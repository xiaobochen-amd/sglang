"""Triton sparse-MLA forward for the DSA fp8 decode path (gfx950).

Prefill packs all H heads into one program per token. Decode tiles heads by 16
and splits the topk reduction across groups so the grid fills the machine at
the token counts MTP actually produces. Enable with `--dsa-decode-backend triton`.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsa.triton_sparse_mla_prefill import (
    _LOG2_FP8_MAX,
    _N_XCD,
)

_HEAD_TILE = 16


@triton.jit
def _sparse_mla_decode_splitk_partial_kernel(
    q_nope_ptr,
    q_rope_ptr,
    kv_ptr,
    idx_ptr,
    po_ptr,
    pm_ptr,
    pl_ptr,
    o_ptr,
    bar_ptr,
    sm_scale,
    log2_fp8_max,
    topk,
    seq,
    max_page,
    H: tl.constexpr,
    H_TILE: tl.constexpr,
    DIM: tl.constexpr,
    D_V: tl.constexpr,
    D_TAIL: tl.constexpr,
    Q_MAIN_PITCH: tl.constexpr,
    Q_TAIL_PITCH: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_GROUPS: tl.constexpr,
    KEYS_PER_GROUP: tl.constexpr,
    N_XCD: tl.constexpr,
    WIDE_KV_OFFSET: tl.constexpr,
    FUSED: tl.constexpr,
    SPIN_CAP: tl.constexpr,
):
    """One (token, head tile, key group) each. Emits unnormalised acc + m + l.

    The exp2/log2_fp8_max bias is the same constant in every group, so it
    cancels between the numerator and denominator of the combine and the
    partials can be merged as if they were plain softmax terms.

    With FUSED the merge happens here too, behind a barrier, instead of in a
    second launch. That is worth doing because an empty kernel costs 2.7 us of
    device time on gfx950 whatever its grid, and the standalone combine only
    has ~1.4 us of real work in it -- two thirds of that launch was pure
    overhead. See the wrapper for why the fused path is not always safe.
    """
    pid = tl.program_id(0)
    head_blocks = (H + H_TILE - 1) // H_TILE
    n_prog = seq * head_blocks * N_GROUPS

    per = n_prog // N_XCD
    rem = n_prog % N_XCD
    xcd = pid % N_XCD
    flat = xcd * per + tl.minimum(xcd, rem) + pid // N_XCD
    ok = flat < n_prog

    g = flat % N_GROUPS
    rest = flat // N_GROUPS
    s_i = rest // head_blocks
    h_tile = rest % head_blocks
    h = h_tile * H_TILE + tl.arange(0, H_TILE)
    hm = (h < H) & ok

    dv = tl.arange(0, D_V)
    dt = tl.arange(0, D_TAIL)
    # Pitches, not D_V/D_TAIL, because the server never has the two halves as
    # separate tensors: the gfx95 fused rope+cache path leaves q as one
    # [seq, H, DIM] buffer, and slicing it costs two copies per call -- 4.4-7.1
    # us, 21-58% of this kernel. Reading it in place with a stride costs
    # nothing; a contiguous half just passes its own D_V / D_TAIL.
    q_main = tl.load(
        q_nope_ptr + s_i * H * Q_MAIN_PITCH + h[:, None] * Q_MAIN_PITCH + dv[None, :],
        mask=hm[:, None],
        other=0.0,
    ).to(q_nope_ptr.dtype.element_ty)
    q_tail = tl.load(
        q_rope_ptr + s_i * H * Q_TAIL_PITCH + h[:, None] * Q_TAIL_PITCH + dt[None, :],
        mask=hm[:, None],
        other=0.0,
    ).to(q_rope_ptr.dtype.element_ty)

    # Score transposed. The obvious form, tl.dot(q, tl.trans(kv)), transposes a
    # [BLOCK_N, D_V] key tile on every iteration -- 64 KB at BLOCK_N=128, the
    # largest thing in the loop. q is loop-invariant, so transposing it once
    # here buys the same product from tl.dot(kv, q_t), and the only per-
    # iteration transpose left is of p at [BLOCK_N, H_TILE], 32x smaller. Worth
    # 1.13-1.17x at two warps and 1.26-1.82x at four; the key-tile transpose was
    # what made four warps the slower choice at BLOCK_N=128.
    q_main_t = tl.trans(q_main)
    q_tail_t = tl.trans(q_tail)

    qk_scale = sm_scale * 1.4426950408889634
    m_i = tl.full([H_TILE], -float("inf"), tl.float32)
    l_i = tl.zeros([H_TILE], tl.float32)
    acc = tl.zeros([H_TILE, D_V], tl.float32)

    n = tl.arange(0, BLOCK_N)
    k_lo = g * KEYS_PER_GROUP
    for k0 in range(0, KEYS_PER_GROUP, BLOCK_N):
        kpos = k_lo + k0 + n
        if KEYS_PER_GROUP % BLOCK_N == 0:
            kmask = (kpos < topk) & ok
        else:
            # A tile that overruns its group would walk into the next group's
            # keys and count them twice -- kpos < topk does not catch that,
            # since those keys are real. Only reachable when block_n had to be
            # rounded down to a power of two.
            kmask = (k0 + n < KEYS_PER_GROUP) & (kpos < topk) & ok
        idx = tl.load(idx_ptr + s_i * topk + kpos, mask=kmask, other=-1)
        # Range-check here rather than clamping the index tensor on the host.
        # That clamp was a whole elementwise launch per call -- 1.57 us of
        # device time to touch 2048 int32 at one token, 15% of the kernel pair
        # -- where the comparison rides along free in a test the loop already
        # does.
        valid_k = (idx >= 0) & (idx <= max_page) & kmask
        page = tl.where(valid_k, idx, 0)

        if WIDE_KV_OFFSET:
            kbase = kv_ptr + page[:, None].to(tl.int64) * DIM
        else:
            kbase = kv_ptr + page[:, None] * DIM
        kv_main = tl.load(kbase + dv[None, :]).to(q_nope_ptr.dtype.element_ty)
        kv_tail = tl.load(kbase + (D_V + dt)[None, :]).to(q_nope_ptr.dtype.element_ty)

        qkt = tl.dot(kv_main, q_main_t).to(tl.float32)
        qkt += tl.dot(kv_tail, q_tail_t).to(tl.float32)
        qkt = qkt * qk_scale + tl.where(valid_k[:, None], 0.0, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qkt, axis=0))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        pt = tl.exp2(qkt - (m_safe - log2_fp8_max)[None, :])
        l_i = l_i * alpha + tl.sum(pt, axis=0)
        acc = acc * alpha[:, None] + tl.dot(
            tl.trans(pt).to(q_nope_ptr.dtype.element_ty), kv_main
        ).to(tl.float32)
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)

    if N_GROUPS == 1:
        # Nothing to reduce against, so this program owns the whole output row
        # and writes it. Going through the partial buffers here would cost a
        # round trip and a second launch to merge a single group with itself,
        # which is what made n_groups=1 look uncompetitive at token counts
        # where the grid no longer needs splitting.
        tl.store(
            o_ptr + s_i * H * D_V + h[:, None] * D_V + dv[None, :],
            (acc * (1.0 / l_safe)[:, None]).to(o_ptr.dtype.element_ty),
            mask=hm[:, None],
        )
        return

    # Rescale the partial to its own max before narrowing so a bf16 partial
    # store costs mantissa but not range; the combine folds m_i back in.
    po_off = (s_i * H + h) * N_GROUPS * D_V + g * D_V
    tl.store(
        po_ptr + po_off[:, None] + dv[None, :],
        (acc * (1.0 / l_safe)[:, None]).to(po_ptr.dtype.element_ty),
        mask=hm[:, None],
    )
    pml_off = (s_i * H + h) * N_GROUPS + g
    tl.store(pm_ptr + pml_off, m_i, mask=hm)
    tl.store(pl_ptr + pml_off, l_i, mask=hm)

    if FUSED:
        # Ticket barrier over the N_GROUPS programs sharing this (token, head
        # tile). The counter is never reset: each program derives its round
        # from the value its own arrival returned, so the target it waits for
        # is computed on device. A host-side round counter would be baked into
        # the graph at capture time and every replay after the first would
        # wait on a stale target.
        # The atomics address a [1]-shaped block, not a bare scalar pointer:
        # a scalar atomic is issued per lane, so the counter would advance by
        # the block width instead of by one and every lane would spin on a
        # different target. tl.max collapses the [1] result back to a value
        # all lanes agree on, which the while condition needs.
        tile = s_i * head_blocks + h_tile
        bar = bar_ptr + tile + tl.arange(0, 1)
        ticket = tl.max(tl.atomic_add(bar, 1, sem="release", scope="gpu"))
        target = (ticket // N_GROUPS + 1) * N_GROUPS
        # The spin counter is load bearing, not instrumentation: with the poll
        # as the only thing in the loop the compiler treats it as invariant,
        # hoists it, and every program waits forever on a value it re-reads
        # from a register. The counter also caps the wait, so a barrier that
        # cannot be satisfied degrades to a wrong answer instead of wedging
        # the queue -- SPIN_CAP is far above any wait a live barrier produces.
        spins = 0
        seen = tl.max(tl.atomic_add(bar, 0, sem="acquire", scope="gpu"))
        while seen < target and spins < SPIN_CAP:
            seen = tl.max(tl.atomic_add(bar, 0, sem="acquire", scope="gpu"))
            spins += 1

        # Past the barrier every partial for this tile is visible, so the
        # groups split the merge by output slice instead of one program
        # merging all of D_V.
        SLICE: tl.constexpr = D_V // N_GROUPS
        dvs = g * SLICE + tl.arange(0, SLICE)
        gi = tl.arange(0, N_GROUPS)
        cbase = (s_i * H + h) * N_GROUPS

        m_all = tl.load(
            pm_ptr + cbase[:, None] + gi[None, :], mask=hm[:, None], other=-float("inf")
        )
        l_all = tl.load(
            pl_ptr + cbase[:, None] + gi[None, :], mask=hm[:, None], other=0.0
        )
        m_g = tl.max(m_all, axis=1)
        m_gs = tl.where(m_g == -float("inf"), 0.0, m_g)
        w = tl.exp2(
            tl.where(m_all == -float("inf"), -float("inf"), m_all - m_gs[:, None])
        )
        w = tl.where(w == w, w, 0.0)
        wl = l_all * w
        lsum = tl.sum(wl, axis=1)

        o_all = tl.load(
            po_ptr
            + (cbase * D_V)[:, None, None]
            + (gi * D_V)[None, :, None]
            + dvs[None, None, :],
            mask=hm[:, None, None],
            other=0.0,
        ).to(tl.float32)
        merged = tl.sum(o_all * wl[:, :, None], axis=1)
        lsafe = tl.where(lsum == 0.0, 1.0, lsum)
        tl.store(
            o_ptr + s_i * H * D_V + h[:, None] * D_V + dvs[None, :],
            (merged * (1.0 / lsafe)[:, None]).to(o_ptr.dtype.element_ty),
            mask=hm[:, None],
        )


@triton.jit
def _sparse_mla_decode_splitk_combine_kernel(
    po_ptr,
    pm_ptr,
    pl_ptr,
    o_ptr,
    seq,
    H: tl.constexpr,
    D_V: tl.constexpr,
    N_GROUPS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    DV_BLOCKS: tl.constexpr,
):
    """Merge the per-group partials for one (token, head, d_v slice).

    One head per program, not a tile of 16. The partials are [seq, H,
    N_GROUPS, D_V], so a program's read is BLOCK_DV contiguous values then a
    jump of D_V. Whether that is a coalesced access depends entirely on
    BLOCK_DV, and BLOCK_DV is set by how many programs the grid needs.

    Tiling heads made those two demands fight: at one token the grid could
    only come from d_v, which forced BLOCK_DV to 4 -- 8-byte reads out of
    64-byte lines, so the kernel fetched eight times the bytes it used.
    Giving each head its own program supplies H times the grid for free and
    lets BLOCK_DV stay at 32 or more.

    m and l are recomputed in every d_v slice rather than shared. That is
    N_GROUPS extra scalar loads per program against a pm/pl footprint of
    seq*H*N_GROUPS floats -- kilobytes, LLC-resident, and cheaper than any
    scheme for communicating them across workgroups.
    """
    pid = tl.program_id(0)
    ok = pid < seq * H * DV_BLOCKS

    dv_blk = pid % DV_BLOCKS
    rest = pid // DV_BLOCKS
    s_i = rest // H
    h = rest % H

    dv = dv_blk * BLOCK_DV + tl.arange(0, BLOCK_DV)
    dvm = (dv < D_V) & ok
    base = (s_i * H + h) * N_GROUPS

    # The groups are loaded as one [N_GROUPS] / [N_GROUPS, BLOCK_DV] tile
    # rather than in a python loop over g. Looping made every group a
    # dependent round trip to the partials, which left this kernel
    # latency-bound at 0.65 TB/s -- a third of what the same bytes reach
    # when they are all in flight at once.
    g = tl.arange(0, N_GROUPS)
    m_all = tl.load(pm_ptr + base + g, mask=ok, other=-float("inf"))
    l_all = tl.load(pl_ptr + base + g, mask=ok, other=0.0)

    m = tl.max(m_all, axis=0)
    m_safe = tl.where(m == -float("inf"), 0.0, m)

    # The partial store already divided by its own l_g, so reweight by
    # l_g * 2^(m_g - m) here to rebuild the global numerator.
    w = tl.exp2(tl.where(m_all == -float("inf"), -float("inf"), m_all - m_safe))
    w = tl.where(w == w, w, 0.0)
    wl = l_all * w
    lsum = tl.sum(wl, axis=0)

    o_g = tl.load(
        po_ptr + base * D_V + g[:, None] * D_V + dv[None, :],
        mask=dvm[None, :],
        other=0.0,
    ).to(tl.float32)
    acc = tl.sum(o_g * wl[:, None], axis=0)

    l_safe = tl.where(lsum == 0.0, 1.0, lsum)
    acc = acc * (1.0 / l_safe)
    tl.store(
        o_ptr + (s_i * H + h) * D_V + dv,
        acc.to(o_ptr.dtype.element_ty),
        mask=dvm,
    )


def _head_blocks(h: int) -> int:
    return (h + _HEAD_TILE - 1) // _HEAD_TILE


_CUS = 256

_BARRIERS: dict[tuple[int, int], torch.Tensor] = {}

# Measured waits at the production geometry are single-digit polls; this is a
# deadlock guard, not a tuning knob.
_SPIN_CAP = 1 << 22


def _barrier_buffer(n_tiles: int, device: torch.device) -> torch.Tensor:
    """Zeroed counters for the fused merge, allocated once and never reset.

    The kernel's ticket barrier only ever increments, so this has to survive
    across calls -- and across graph replays, which is why it is cached rather
    than allocated per call: a fresh buffer would have to be zeroed by another
    launch, spending the 2.7 us that fusing the merge was meant to save.
    """
    key = (device.index or 0, 1 << max(0, n_tiles - 1).bit_length())
    buf = _BARRIERS.get(key)
    if buf is None:
        buf = torch.zeros(key[1], device=device, dtype=torch.int32)
        _BARRIERS[key] = buf
    return buf


def _can_fuse(requested: bool, grid: int, n_groups: int, d_v: int) -> bool:
    """Whether the merge can run behind an in-kernel barrier.

    Off by default: fusing is correct but does not pay. us per call, 2-kernel
    against fused, at the n_groups each T selects:

        T=6   13.2  18.1        T=18  19.3  18.8        T=32  22.0  39.9

    Level at best, far worse at the ends. The merge's live tile shares a
    register budget with the main loop's [H_TILE, D_V] accumulator, so folding
    it in costs phase 1 more than skipping a launch saves, and spinning keeps
    the early groups on their CUs instead of letting them retire while the
    slowest group finishes.

    The barrier spins, so every program sharing a tile has to be resident at
    once or the ones that arrived will wait forever for ones that were never
    scheduled. One workgroup per CU is the pessimistic assumption, so the grid
    has to fit in the CU count -- which it does at the token counts decode
    actually runs, and does not once seq*n_groups outgrows the machine.

    The merge also splits d_v across the groups of a tile, so d_v has to divide
    into n_groups power-of-two slices for tl.arange.
    """
    if not requested or n_groups <= 1 or grid > _CUS:
        return False
    slice_dv, rem = divmod(d_v, n_groups)
    return rem == 0 and slice_dv & (slice_dv - 1) == 0


def _q_pitch(q: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Head-row pitch of a [seq, heads, d] query, copying only if it needs one.

    A row-contiguous q reports its own d and is passed through untouched. The
    case worth having is the one the server produces: on gfx95 the fused
    rope+cache path hands us one [seq, heads, 576] tensor and dsa_backend
    slices it into a 512 and a 64 half. Those slices are contiguous along d and
    evenly pitched along heads, which is all the kernel needs -- calling
    .contiguous() on them instead costs two copies, 4.4-7.1 us per call.
    """
    pitch = q.stride(1)
    if q.stride(2) == 1 and q.stride(0) == q.shape[1] * pitch and pitch >= q.shape[2]:
        return q, pitch
    q = q.contiguous()
    return q, q.stride(1)


def pick_combine_block_dv(tokens: int, heads: int, d_v: int = 512) -> int:
    """Size the merge tile so the merge grid stays near 768 programs.

    The merge grid is `tokens * heads * d_v/BLOCK_DV`. One head per program
    supplies an 8x or 16x factor that used to have to come out of BLOCK_DV,
    so the floor can sit at 64 -- a program's read is BLOCK_DV contiguous
    fp32 before it jumps D_V, and 64 of them is a 256-byte burst. Targeting
    128 programs oversubscribed the tile at mid T: T=12 H=16 picked
    BLOCK_DV=512 (192 programs) where 128 (768 programs) is 0.4 us faster.

    Four merge warps lost to one at every MTP point, so the wrapper's
    default is 1; this helper only sizes the tile.

    Largest power-of-two tile that still launches 768 programs, clamped to
    [64, d_v].
    """
    target = d_v * tokens * heads // 768
    block_dv = 64
    while block_dv * 2 <= min(target, d_v):
        block_dv *= 2
    return block_dv


def pick_block_n_warps(
    tokens: int, heads: int, n_groups: int, topk: int = 2048
) -> tuple[int, int]:
    """Widest tile that the group can fill; warps by the loop trip count.

    This used to shrink the tile past a grid of ~512. block_n * d_v fp8 is the
    LDS a program holds, so block_n=128 asks for the full 64 KB and only two
    programs fit on a CU, and past 512 the tail of a 3-deep grid appeared to
    run at a third of the width it could -- narrow tiles measured 7-20% faster
    there, consistently enough across four token counts to look structural.

    It was not residency. It was the [BLOCK_N, D_V] transpose the key loop did
    on every iteration, which grows with the tile and which the partial kernel
    no longer does. With that hoisted, the widest tile wins at every token
    count in the MTP range whatever the grid: 84 tokens at H=8, a grid of 672,
    goes from 35.7 us on the old rule to 31.8 on a full-width tile. The only
    clamp left is that a tile wider than the group it walks would run
    half-masked for the same LDS as a full one.

    Three (block_n, num_warps) pairs miscompile on this Triton and silently
    return a wrong answer -- (32, 2), (64, 1) and (128, 1), at 14-20 dB against
    the fp32 oracle where every working pair sits at 32. It is conditional on
    the trip count, so the same pair is correct at one token count and wrong at
    another. They are not slower, so a sweep that only times will select them.
    (128, 2) and (128, 4) are checked against the oracle at every token count
    in the MTP range; re-verify with an SNR gate before returning anything
    else.

    Warps follow the trip count. A group of exactly one tile has no next
    iteration to overlap, so the extra warps only add scheduling and take
    registers -- two wins at all sixteen swept points below 24 tokens, by 0.1
    to 0.4 us. Past one tile there is load latency to hide and four wins by
    more than that.

    block_n has to be a power of two for tl.arange, and topk // n_groups is
    not one for every topk -- 1536 keys over 16 groups is 96, which used to
    reach the kernel and fail to compile. Rounding down leaves a group that
    the tiles do not divide, which the key loop masks for.
    """
    keys_per_group = topk // n_groups
    block_n = 1 << (min(128, keys_per_group).bit_length() - 1)
    # (32, 2) is the remaining unsafe pair on this Triton -- 14-20 dB at a
    # 1-iteration group, which is exactly the g=64 shape. Two warps are
    # otherwise the right call for a single tile, so only this width drops
    # to one.
    if keys_per_group <= block_n:
        return block_n, 1 if block_n <= 32 else 2
    return block_n, 4


def pick_num_stages(
    tokens: int, heads: int, n_groups: int, block_n: int, topk: int = 2048
) -> int:
    """Pipeline the key loop only when nothing else is hiding the load.

    The partial kernel runs at about 1.6 TB/s where a bare gather of the same
    rows reaches 3.1, and it is not arithmetic that holds it back -- the dots
    come to 3% of peak. It is the KV load latency, and there are two ways to
    cover it: another program on the same CU, or another iteration of this
    program's own loop. Which one is available depends on the grid.

    Below one program per CU the machine supplies no overlap of its own and
    pipelining pays; above it the programs already cover each other and the
    stages only cost registers. Both effects are large and the crossover is
    exactly the CU count. us at H=16, one group per 256 keys:

      grid      144   192   240   256 | 288   320   384
      stages=1 14.4  15.3  17.4  17.8 |20.5  21.0  21.5
      stages=3 14.2  14.8  16.6  17.3 |25.6  26.6  27.7

    Sixteen of those eighteen points are outside the MTP token counts this
    kernel was tuned on, and the rule holds at all of them, in both directions
    -- which is the only reason to trust a branch that just one production
    point (24 tokens, C=4 verifying) actually takes.

    A group the tiles do not divide is left unpipelined. Staging the ragged
    trailing iteration returns a wrong answer on this Triton -- 40% off at
    topk=1536, where 16 groups of 96 keys run 64 + 32 -- and it is the case
    with the least to win anyway. topk=2048 divides by every group count this
    picks, so no production point takes that branch.
    """
    keys_per_group = topk // n_groups
    if keys_per_group <= block_n or keys_per_group % block_n:
        return 1
    grid = tokens * _head_blocks(heads) * n_groups
    return 3 if grid <= _CUS else 1


def pick_n_groups(tokens: int, heads: int, topk: int, cap: int = 16) -> int:
    """Split the reduction 64 ways at 1-2 tokens, 16 ways up to 24, then 8/4.

    After the merge went to one head per program it stopped growing linearly
    in n_groups at small T -- the launch is 2.06 us either way -- so the
    partial can take the split it always wanted. us at H=8 / H=16, ctx=100k:

      tokens     g16    g32    g64          auto used to pick
           1    7.78 / 8.16   7.47 / 8.13   7.09 / 7.66     16
           2    7.88 / 8.55   7.84 / 8.65   7.36 / 8.67     16
           6    8.37 / 9.19   8.46 / 9.88   9.31 /12.19     16
          24   11.58 /13.51  14.19 /17.32                  8 (H=8 wanted 16)

    64 groups at one token matches FlyDSL's partial (4.4-4.8 us) because both
    then run a 32-key tile; 16 groups left that kernel 1.2 us behind. Past two
    tokens the merge traffic comes back and 16 is the ceiling again. The grow
    threshold is 384 at H=8 and 256 at H=16 so 24 tokens keeps 16 groups at
    TP8 (11.3 vs 12.1) and 8 at TP4 (13.2 vs 13.4). 48 tokens still wants 8,
    84 still wants 4, so the shrink threshold stays 512.
    """
    base = tokens * _head_blocks(heads)
    n = 8
    while n > 1 and topk % n:
        n //= 2
    while n > 1 and base * n > 512:
        n //= 2
    grow_lim = 384 if heads <= 8 else 256
    while n < cap and base * n * 2 <= grow_lim and topk % (n * 2) == 0:
        n *= 2
    if tokens <= 2 and topk % 64 == 0:
        n = 64
    return n


def triton_sparse_mla_decode_splitk_fwd(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    *,
    block_n: int = 0,
    n_groups: int = 0,
    num_warps: int = 0,
    combine_warps: int = 1,
    combine_block_dv: int = 0,
    fused: bool = False,
    num_stages: int = 0,
    partial_dtype: torch.dtype = torch.bfloat16,
    n_xcd: int = _N_XCD,
) -> torch.Tensor:
    """Decode sparse MLA. Returns [1, seq, H, d_v] bf16.

    `--dsa-decode-backend triton` (gfx950, fp8 KV). Tiling is picked from the
    token and head count; `pick_*` encode measured configs, including pairs
    that miscompile on this Triton. q is two strided halves of one [T, H, 576]
    tensor -- see `_q_pitch`. SNR 32.2-32.4 dB vs an fp32 oracle.

    Idle MI355X, ctx=100k, auto tiling, us/call from timed_graph (16-call
    graphs). FlyDSL is aiter's flydsl_sparse_mla_decode in the same process.
    `floor` is gather-only. 24-point sum: TileLang 510, FlyDSL 338, this 293.

        TP8 (H=8)                            TP4 (H=16)
        C  role     T  floor    TL   fly  this   floor    TL   fly  this
        1  draft    1    2.9  14.3   8.4   6.9     2.9  14.0   8.4   7.3
        1  verify   6    3.1  14.9   9.3   8.0     3.0  15.1   9.4   8.8
        2  draft    2    2.8  14.6   8.5   7.2     2.8  14.1   8.5   7.8
        2  verify  12    3.7  17.3  12.1   8.4     3.7  18.7  12.0   9.6
        4  draft    4    2.8  14.7   8.9   7.9     2.8  14.6   9.0   8.6
        4  verify  24    4.8  21.9  13.3  11.3     4.9  22.8  13.5  13.1
        8  draft    8    3.0  15.2   9.8   8.1     2.9  15.8   9.8   9.0
        8  verify  48   11.2  27.3  20.8  18.1    11.2  27.9  20.7  19.5
        10 draft   10    3.4  16.9  11.5   8.3     3.4  17.7  11.5   9.4
        10 verify  60   14.0  36.7  22.3  19.8    13.9  37.3  22.4  21.8
        14 draft   14    4.2  18.1  12.5   8.7     4.2  19.3  12.5  10.1
        14 verify  84   18.8  39.8  31.3  26.8    18.7  40.4  31.2  27.7
    """
    seq, h, d_v_in = q_nope.shape
    assert d_v_in == d_v
    if n_groups <= 0:
        n_groups = pick_n_groups(seq, h, indices.shape[-1])
    auto_block_n, auto_warps = pick_block_n_warps(
        seq, h, n_groups, indices.shape[-1]
    )
    block_n = block_n or auto_block_n
    num_warps = num_warps or auto_warps
    num_stages = num_stages or pick_num_stages(
        seq, h, n_groups, block_n, indices.shape[-1]
    )
    combine_block_dv = combine_block_dv or pick_combine_block_dv(seq, h, d_v)
    if topk_rem := indices.shape[-1] % n_groups:
        raise ValueError(
            f"topk {indices.shape[-1]} not divisible by n_groups {n_groups} "
            f"(remainder {topk_rem})"
        )
    q_nope, q_main_pitch = _q_pitch(q_nope)
    q_rope, q_tail_pitch = _q_pitch(q_rope)
    if indices.dim() == 3:
        indices = indices.squeeze(1)
    indices = indices.contiguous()
    dim = kv.shape[-1]
    d_tail = q_rope.shape[-1]
    topk = indices.shape[-1]
    wide = kv.shape[0] > (2**31 - 1) // dim
    max_page = kv.shape[0] - 1
    head_blocks = _head_blocks(h)

    single = n_groups == 1
    if single:
        # The kernel normalises and writes `out` itself in this case, so the
        # partial buffers exist only to give the launch something to bind.
        partial_o = partial_m = partial_l = q_nope
    else:
        partial_o = torch.empty(
            seq, h, n_groups, d_v, device=q_nope.device, dtype=partial_dtype
        )
        partial_m = torch.empty(
            seq, h, n_groups, device=q_nope.device, dtype=torch.float32
        )
        partial_l = torch.empty_like(partial_m)

    grid = seq * head_blocks * n_groups
    out = torch.empty(seq, h, d_v, device=q_nope.device, dtype=torch.bfloat16)
    use_fused = _can_fuse(fused, grid, n_groups, d_v)
    bar = _barrier_buffer(seq * head_blocks, q_nope.device) if use_fused else partial_m

    _sparse_mla_decode_splitk_partial_kernel[(grid,)](
        q_nope,
        q_rope,
        kv,
        indices,
        partial_o,
        partial_m,
        partial_l,
        out,
        bar,
        sm_scale,
        _LOG2_FP8_MAX,
        topk,
        seq,
        max_page,
        H=h,
        H_TILE=_HEAD_TILE,
        DIM=dim,
        D_V=d_v,
        D_TAIL=d_tail,
        Q_MAIN_PITCH=q_main_pitch,
        Q_TAIL_PITCH=q_tail_pitch,
        BLOCK_N=block_n,
        N_GROUPS=n_groups,
        KEYS_PER_GROUP=topk // n_groups,
        N_XCD=n_xcd,
        WIDE_KV_OFFSET=wide,
        FUSED=use_fused,
        SPIN_CAP=_SPIN_CAP,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if use_fused or single:
        return out.unsqueeze(0)

    dv_blocks = (d_v + combine_block_dv - 1) // combine_block_dv
    _sparse_mla_decode_splitk_combine_kernel[(seq * h * dv_blocks,)](
        partial_o,
        partial_m,
        partial_l,
        out,
        seq,
        H=h,
        D_V=d_v,
        N_GROUPS=n_groups,
        BLOCK_DV=combine_block_dv,
        DV_BLOCKS=dv_blocks,
        num_warps=combine_warps,
        num_stages=1,
    )
    return out.unsqueeze(0)
