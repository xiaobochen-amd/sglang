"""Triton sparse-MLA forward for the DSA fp8 prefill path."""

import math

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz

_IS_FNUZ = is_fp8_fnuz()
_FP8_MAX = 240.0 if _IS_FNUZ else 448.0
_LOG2_FP8_MAX = math.log2(_FP8_MAX)


@triton.jit
def _sparse_mla_prefill_kernel(
    q_nope_ptr,
    q_rope_ptr,
    kv_ptr,
    idx_ptr,
    o_ptr,
    sm_scale,
    log2_fp8_max,
    topk,
    n_tok,
    H: tl.constexpr,
    H_PAD: tl.constexpr,
    DIM: tl.constexpr,
    D_V: tl.constexpr,
    D_TAIL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_XCD: tl.constexpr,
    WIDE_KV_OFFSET: tl.constexpr,
):
    pid = tl.program_id(0)
    # Workgroups land on XCD (pid % N_XCD), so hand each XCD one contiguous run
    # of tokens. n_tok need not divide N_XCD: the first (n_tok % N_XCD) XCDs take
    # one extra token, which keeps pid -> s_i a bijection onto [0, n_tok). A
    # plain (pid % N_XCD) * (n_tok // N_XCD) + pid // N_XCD is only a bijection
    # when the division is exact, and otherwise leaves output rows unwritten.
    per = n_tok // N_XCD
    rem = n_tok % N_XCD
    xcd = pid % N_XCD
    s_i = xcd * per + tl.minimum(xcd, rem) + pid // N_XCD
    s_ok = s_i < n_tok

    h = tl.arange(0, H_PAD)
    hm = h < H
    dv = tl.arange(0, D_V)
    dt = tl.arange(0, D_TAIL)
    q_main = tl.load(
        q_nope_ptr + s_i * H * D_V + h[:, None] * D_V + dv[None, :],
        mask=hm[:, None] & s_ok,
        other=0.0,
    ).to(q_nope_ptr.dtype.element_ty)
    q_tail = tl.load(
        q_rope_ptr + s_i * H * D_TAIL + h[:, None] * D_TAIL + dt[None, :],
        mask=hm[:, None] & s_ok,
        other=0.0,
    ).to(q_nope_ptr.dtype.element_ty)

    qk_scale = sm_scale * 1.4426950408889634
    m_i = tl.full([H_PAD], -float("inf"), tl.float32)
    l_i = tl.zeros([H_PAD], tl.float32)
    acc = tl.zeros([H_PAD, D_V], tl.float32)

    n = tl.arange(0, BLOCK_N)
    # A block's KV addresses cannot be formed until its page ids land, and at
    # one wavefront per SIMD there is no co-resident wave to cover that stall.
    # Keep the next block's ids in flight so the load issues alongside this
    # block's math rather than after it. Costs 3 VGPRs and buys 7%.
    kmask = (n < topk) & s_ok
    idx = tl.load(idx_ptr + s_i * topk + n, mask=kmask, other=-1)

    for k0 in range(0, topk, BLOCK_N):
        kmask = ((k0 + n) < topk) & s_ok
        valid_k = (idx >= 0) & kmask
        valid_qk = valid_k[None, :]
        page = tl.where(valid_k, idx, 0)

        # Reads past topk are masked off, so the tail iteration is a no-op.
        nk = k0 + BLOCK_N
        nmask = ((nk + n) < topk) & s_ok
        idx = tl.load(idx_ptr + s_i * topk + nk + n, mask=nmask, other=-1)

        if WIDE_KV_OFFSET:
            kbase = kv_ptr + page[:, None].to(tl.int64) * DIM
        else:
            kbase = kv_ptr + page[:, None] * DIM
        # No mask on the KV loads. `page` is already clamped to 0 for invalid
        # lanes, so the address is in bounds and the read returns real fp8 --
        # finite, since e4m3 in this pool has no inf. Those columns are then
        # forced to -inf below, giving p == 0, so whatever was read multiplies
        # out of the PV dot. Masking instead costs a v_cndmask per element of a
        # [BLOCK_N, D_V] tile, which was the single largest term in the loop.
        kv_main = tl.load(kbase + dv[None, :]).to(q_nope_ptr.dtype.element_ty)
        kv_tail = tl.load(kbase + (D_V + dt)[None, :]).to(q_nope_ptr.dtype.element_ty)

        # The transpose here is not what the loop's 192 LDS instructions are
        # paying for. Computing qk the other way round -- dot(kv_main, q^T),
        # which puts kv in the operand position it was loaded for and leaves
        # only the loop-invariant q to transpose -- moves the LDS count to 195,
        # i.e. nowhere, because the staging is what feeds MFMA from a
        # globally-loaded tile at all, not the transpose. It also pushes the
        # softmax reductions onto axis 0 and the loop to 981 instructions, and
        # measures 0.89x. Not a promising direction to revisit.
        qk = tl.dot(q_main, tl.trans(kv_main)).to(tl.float32)
        qk += tl.dot(q_tail, tl.trans(kv_tail)).to(tl.float32)
        # The whole softmax runs in log2 space. exp() on this hardware is
        # v_exp_f32, which computes 2**x, so tl.exp emits a multiply by log2(e)
        # in front of it for every element of the [H_PAD, BLOCK_N] tile; folding
        # that constant into sm_scale deletes those multiplies. The -inf mask
        # rides along as the addend of the same fma, which is cheaper than the
        # v_cndmask it replaces.
        qk = qk * qk_scale + tl.where(valid_qk, 0.0, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        # exp2 lands directly in the fp8 domain: subtracting log2(fp8_max) from
        # the running max scales p by fp8_max, using the subtract that was
        # already there instead of a multiply per element.
        p = tl.exp2(qk - (m_safe - log2_fp8_max)[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)

        p_fp8 = p.to(q_nope_ptr.dtype.element_ty)
        # Leave the fp8_max factor in. It is now carried by both pv and l_i, so
        # it cancels in the final divide, and undoing it here would cost a
        # multiply across the [H_PAD, D_V] tile -- 128 registers per lane, the
        # largest single block of VALU that was left in the loop.
        pv = tl.dot(p_fp8, kv_main).to(tl.float32)
        # Rescaling costs a read, a packed multiply and a write across all 128
        # accumulator registers, a quarter of the loop, and alpha is 1 on the
        # blocks where no head saw a new maximum. Guarding it on
        # tl.max(m_new - m_i) > 0 is nonetheless 13% slower: the branch makes
        # the compiler materialise acc on both paths and AGPR traffic inside
        # the loop goes from 136 to 182 moves, outweighing the multiply saved.
        acc = acc * alpha[:, None] + pv
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    # Dividing the accumulator directly makes the compiler emit a full IEEE
    # divide per element -- v_div_scale/v_rcp/v_div_fmas/v_div_fixup, ~10
    # dependent instructions -- 128 times per lane, because it will not hoist
    # the reciprocal out of the broadcast. Reciprocating the H_PAD-wide
    # denominator first leaves one divide and a multiply per element.
    acc = acc * (1.0 / l_safe)[:, None]
    tl.store(
        o_ptr + s_i * H * D_V + h[:, None] * D_V + dv[None, :],
        acc.to(o_ptr.dtype.element_ty),
        mask=hm[:, None] & s_ok,
    )


# One config for every shape leaves a lot on the table. The per-iteration tile
# is [H_PAD, D_V] x [D_V, BLOCK_N], so a single warp keeps up at H_PAD=16 but
# falls behind as H_PAD grows: BLOCK_N=64/num_warps=1 is optimal at H_PAD=16 yet
# 2.1x off at H_PAD=32 and 8.5x off at H_PAD=64. Measured bests on MI355X at
# seq=16384 (benchmark/kernels/sparse_mla_chunk32k), over
# BLOCK_N in 32..256 x num_warps in 1..8:
#
#   H_PAD=16  (64, 1, 1)      H_PAD=32  (64, 2, 1)
#
# num_warps = H_PAD // 16 reproduces those, and beats a runtime autotune: on
# Triton 3.7.0 the config (BLOCK_N=128, num_warps=1, num_stages=2) aborts the
# compiler outright -- an LLVM assertion, so SIGABRT rather than an exception
# autotune could catch and skip. num_stages=2 never won here anyway.
_MAX_WARPS = 4


def _pick_cfg(h_pad: int) -> dict:
    return dict(
        BLOCK_N=64,
        num_warps=max(1, min(_MAX_WARPS, h_pad // 16)),
        num_stages=1,
    )

# XCD count on gfx950. N_XCD=1 disables the swizzle (s_i == pid).
_N_XCD = 8


def _sanitize_indices(indices: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
    """Clamp page ids to valid KV range; negatives stay masked."""
    if indices.numel() == 0:
        return indices
    max_page = kv.shape[0] - 1
    if max_page < 0:
        return indices
    # Masking with a boolean index costs a nonzero(), a gather/scatter pair and
    # a host sync on every call, all on the serving path. The kernel only tests
    # idx >= 0, so clamping the whole tensor is equivalent and is one pass.
    return indices.clamp(min=-1, max=max_page)


def _launch_sparse_mla_prefill(
    q_nope,
    q_rope,
    kv,
    indices,
    out,
    sm_scale,
    d_v,
    *,
    n_xcd=_N_XCD,
    wide_kv_offset=None,
):
    seq, H, d_v_in = q_nope.shape
    assert d_v_in == d_v
    d_tail = q_rope.shape[-1]
    dim = kv.shape[-1]
    topk = indices.shape[-1]
    h_pad = max(16, 1 << (H - 1).bit_length())
    args = (
        q_nope,
        q_rope,
        kv,
        indices,
        out,
        sm_scale,
        _LOG2_FP8_MAX,
        topk,
        seq,
    )
    # Triton evaluates `page * DIM` in int32, so a KV pool larger than
    # 2**31 // DIM tokens wraps the offset negative and reads a wild address.
    # Pool size tracks free HBM, so this fires unpredictably: 2.5M tokens is
    # fine, 4.0M (seen with a lighter co-tenant) is not. Widening the whole
    # inner loop to 64-bit costs throughput, so only do it past the limit.
    if wide_kv_offset is None:
        wide_kv_offset = kv.shape[0] > (2**31 - 1) // dim
    kw = dict(
        H=H,
        H_PAD=h_pad,
        DIM=dim,
        D_V=d_v,
        D_TAIL=d_tail,
        N_XCD=n_xcd,
        WIDE_KV_OFFSET=wide_kv_offset,
    )
    _sparse_mla_prefill_kernel[(seq,)](*args, **kw, **_pick_cfg(h_pad))


def triton_sparse_mla_prefill_fwd(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> torch.Tensor:
    seq, H, d_v_in = q_nope.shape
    assert d_v_in == d_v
    q_nope = q_nope.contiguous()
    q_rope = q_rope.contiguous()
    out = torch.empty(seq, H, d_v, device=q_nope.device, dtype=torch.bfloat16)
    indices = _sanitize_indices(indices, kv)
    _launch_sparse_mla_prefill(q_nope, q_rope, kv, indices, out, sm_scale, d_v)
    return out.unsqueeze(0)
