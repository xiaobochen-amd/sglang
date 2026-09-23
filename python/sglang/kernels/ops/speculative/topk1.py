from __future__ import annotations

import torch
import triton
import triton.language as tl

_DRAFT_TOPK1_BLOCK = 8192
# Above this many rows one block per row already fills the GPU, so row_argmax
# hands the reduction back to torch.
_ROW_ARGMAX_ROW_LIMIT = 32


@triton.jit
def _draft_topk1_partial_argmax_kernel(
    logits,
    partial_vals,
    partial_indices,
    logits_row_stride,
    vocab_size: tl.constexpr,
    num_splits: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # int64 row base: row * stride overflows int32 once bs * vocab reaches 2^31.
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    offsets = split * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < vocab_size
    vals = tl.load(
        logits + row * logits_row_stride + offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    # Keep NaNs on valid lanes from selecting the masked tail.
    vals = tl.where(vals == vals, vals, -1e30)

    max_val = tl.max(vals, axis=0)
    local_index = tl.argmax(vals, axis=0)
    out_offset = row * num_splits + split
    tl.store(partial_vals + out_offset, max_val)
    tl.store(partial_indices + out_offset, split * BLOCK + local_index)


@triton.jit
def _draft_topk1_finalize_kernel(
    partial_vals,
    partial_indices,
    topk_p,
    topk_index,
    positions,
    draft_tokens,
    draft_tokens_stride,
    draft_token_column,
    num_splits: tl.constexpr,
    WRITE_DRAFT_TOKEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < num_splits
    vals = tl.load(
        partial_vals + row * num_splits + offsets,
        mask=mask,
        other=-float("inf"),
    )

    split = tl.argmax(vals, axis=0)
    index = tl.load(partial_indices + row * num_splits + split).to(tl.int64)
    tl.store(topk_index + row, index)
    tl.store(topk_p + row, 1.0)
    if WRITE_DRAFT_TOKEN:
        tl.store(draft_tokens + row * draft_tokens_stride + draft_token_column, index)

    position = tl.load(positions + row)
    tl.store(positions + row, position + 1)


@triton.jit
def _row_argmax_finalize_kernel(
    partial_vals,
    partial_indices,
    out_indices,
    num_splits: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < num_splits
    vals = tl.load(
        partial_vals + row * num_splits + offsets,
        mask=mask,
        other=-float("inf"),
    )
    split = tl.argmax(vals, axis=0)
    index = tl.load(partial_indices + row * num_splits + split).to(tl.int64)
    tl.store(out_indices + row, index)


def _partial_argmax_block(rows: int, vocab_size: int) -> int:
    """Widest partial that still spreads the rows over the whole GPU."""
    block = _DRAFT_TOPK1_BLOCK
    while block > 2048 and rows * triton.cdiv(vocab_size, block) < 256:
        block //= 2
    return block


def row_argmax(values: torch.Tensor) -> torch.Tensor:
    """``values.argmax(dim=-1, keepdim=True)`` with the row split across CTAs.

    Same reason as draft_topk1_postprocess: torch walks one block per row, so a
    vocab-wide row at the speculative row counts leaves the GPU idle. MI355X,
    vocab 154880, under CUDA graph replay: 19.7 -> 11.2 us at 1 row, 39.1 -> 11.4
    at 2, 38.7 -> 11.4 at 32. Eager at 1 row is a 2.3x LOSS (12.6 -> 29.0): torch
    has a fast path there and the three launches are not amortized, so this pays
    off only under graph capture, which is how decode runs. Ties go to
    the lowest index, which is also what the split reduction above guarantees and
    ROCm's argmax does not (#26358). Falls back to torch for the shapes the split
    does not pay for, so callers can use it unconditionally.
    """
    if (
        values.ndim != 2
        or values.dtype != torch.float32
        or values.shape[0] == 0
        or values.shape[0] > _ROW_ARGMAX_ROW_LIMIT
        or values.stride(1) != 1
        or not values.is_cuda
    ):
        return values.argmax(dim=-1, keepdim=True)

    bs, vocab_size = values.shape
    block = _partial_argmax_block(bs, vocab_size)
    num_splits = triton.cdiv(vocab_size, block)
    partial_vals = torch.empty(
        (bs, num_splits), dtype=torch.float32, device=values.device
    )
    partial_indices = torch.empty(
        (bs, num_splits), dtype=torch.int32, device=values.device
    )
    out_indices = torch.empty((bs, 1), dtype=torch.int64, device=values.device)

    _draft_topk1_partial_argmax_kernel[(bs, num_splits)](
        values,
        partial_vals,
        partial_indices,
        values.stride(0),
        vocab_size,
        num_splits,
        BLOCK=block,
        num_warps=8,
    )
    _row_argmax_finalize_kernel[(bs,)](
        partial_vals,
        partial_indices,
        out_indices,
        num_splits,
        BLOCK=triton.next_power_of_2(num_splits),
        num_warps=1,
    )
    return out_indices


@triton.jit
def _draft_proposal_partial_kernel(
    probs,
    gumbel,
    top_ks,
    partial_vals,
    partial_indices,
    probs_row_stride,
    gumbel_row_stride,
    vocab_size: tl.constexpr,
    num_splits: tl.constexpr,
    BLOCK: tl.constexpr,
    TINY: tl.constexpr,
):
    # int64 row base: row * stride overflows int32 once bs * vocab reaches 2^31.
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    offsets = split * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < vocab_size
    probs_vals = tl.load(
        probs + row * probs_row_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gumbel_vals = tl.load(
        gumbel + row * gumbel_row_stride + offsets,
        mask=mask,
        other=1.0,
    ).to(tl.float32)
    # A greedy row scores on the probability itself. Dividing it by exactly 1.0
    # keeps one arithmetic path for both arms, and leaves the greedy arm's order
    # bit-identical to ranking probs directly.
    greedy = tl.load(top_ks + row) <= 1
    gumbel_vals = tl.where(greedy, 1.0, tl.maximum(gumbel_vals, TINY))
    vals = probs_vals / gumbel_vals
    vals = tl.where(mask, vals, -float("inf"))
    # Keep NaNs on valid lanes from selecting the masked tail.
    vals = tl.where(vals == vals, vals, -1e30)

    max_val = tl.max(vals, axis=0)
    local_index = tl.argmax(vals, axis=0)
    out_offset = row * num_splits + split
    tl.store(partial_vals + out_offset, max_val)
    tl.store(partial_indices + out_offset, split * BLOCK + local_index)


@triton.jit
def _draft_proposal_finalize_kernel(
    partial_vals,
    partial_indices,
    probs,
    topk_p,
    topk_index,
    probs_row_stride,
    num_splits: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < num_splits
    vals = tl.load(
        partial_vals + row * num_splits + offsets,
        mask=mask,
        other=-float("inf"),
    )

    split = tl.argmax(vals, axis=0)
    index = tl.load(partial_indices + row * num_splits + split).to(tl.int64)
    tl.store(topk_index + row, index)
    # The proposal probability comes from the same pass, so the caller does not
    # pay a separate gather over the row.
    tl.store(topk_p + row, tl.load(probs + row.to(tl.int64) * probs_row_stride + index))


def draft_proposal_select(
    probs: torch.Tensor, gumbel: torch.Tensor, top_ks: torch.Tensor
):
    """Pick one token per row for the draft proposal: ``(q(X), X)``.

    ``X`` is the Gumbel-max draw ``argmax(probs / gumbel)``, or the row's own
    argmax on a greedy row (``top_k <= 1``); both arms ride the one split
    reduction that already has to read the row.

    ``gumbel`` stays the caller's: the accept test needs the draw and the
    distribution produced together. Falls back to eager composition for
    shapes the split does not pay for.
    """
    bs = probs.shape[0]
    if (
        probs.ndim != 2
        or probs.dtype != torch.float32
        or bs == 0
        or bs > _ROW_ARGMAX_ROW_LIMIT
        or probs.stride(1) != 1
        or gumbel.shape != probs.shape
        or gumbel.dtype != probs.dtype
        or gumbel.stride(1) != 1
        or top_ks.ndim != 1
        or top_ks.shape[0] != bs
        or not top_ks.is_contiguous()
        or not probs.is_cuda
    ):
        scores = probs / gumbel.clamp_min(torch.finfo(gumbel.dtype).tiny)
        topk_index = torch.where(
            (top_ks <= 1).view(-1, 1),
            probs.argmax(dim=-1, keepdim=True),
            scores.argmax(dim=-1, keepdim=True),
        )
        return probs.gather(1, topk_index), topk_index

    vocab_size = probs.shape[1]
    block = _partial_argmax_block(bs, vocab_size)
    num_splits = triton.cdiv(vocab_size, block)
    partial_vals = torch.empty(
        (bs, num_splits), dtype=torch.float32, device=probs.device
    )
    partial_indices = torch.empty(
        (bs, num_splits), dtype=torch.int32, device=probs.device
    )
    topk_p = torch.empty((bs, 1), dtype=torch.float32, device=probs.device)
    topk_index = torch.empty((bs, 1), dtype=torch.int64, device=probs.device)

    _draft_proposal_partial_kernel[(bs, num_splits)](
        probs,
        gumbel,
        top_ks,
        partial_vals,
        partial_indices,
        probs.stride(0),
        gumbel.stride(0),
        vocab_size,
        num_splits,
        BLOCK=block,
        TINY=torch.finfo(torch.float32).tiny,
        num_warps=8,
    )
    _draft_proposal_finalize_kernel[(bs,)](
        partial_vals,
        partial_indices,
        probs,
        topk_p,
        topk_index,
        probs.stride(0),
        num_splits,
        BLOCK=triton.next_power_of_2(num_splits),
        num_warps=1,
    )
    return topk_p, topk_index


def draft_topk1_postprocess(
    next_token_logits: torch.Tensor,
    positions: torch.Tensor,
    draft_tokens: torch.Tensor | None = None,
    draft_token_column: int = 0,
):
    """Argmax draft logits for topk=1 and advance positions.

    PyTorch eager argmax reduces each row with too little parallelism for the
    GLM/DSV4 vocab widths in CUDA graph replay. This split reduction exposes
    the vocab dimension across CTAs, then finalizes one token per row.

    If ``draft_tokens`` is given, the finalize kernel also stores the argmax
    into ``draft_tokens[:, draft_token_column]``, mutating the caller-owned
    buffer in place. ``topk_p`` is returned as constant 1.0: topk=1 drafting
    is greedy and the chain probabilities are unused downstream.
    """
    assert next_token_logits.ndim == 2
    assert next_token_logits.stride(1) == 1
    assert positions.ndim == 1
    assert positions.is_contiguous()
    assert positions.shape[0] == next_token_logits.shape[0]
    assert positions.device == next_token_logits.device
    write_draft_token = draft_tokens is not None
    if write_draft_token:
        assert draft_tokens.ndim == 2
        assert draft_tokens.dtype == torch.long
        assert draft_tokens.device == next_token_logits.device
        assert draft_tokens.shape[0] == next_token_logits.shape[0]
        assert draft_tokens.stride(1) == 1
        assert 0 <= draft_token_column < draft_tokens.shape[1]

    bs, vocab_size = next_token_logits.shape
    topk_p = torch.empty((bs, 1), dtype=torch.float32, device=next_token_logits.device)
    topk_index = torch.empty(
        (bs, 1), dtype=torch.int64, device=next_token_logits.device
    )
    if bs == 0:
        return topk_p, topk_index

    block = _DRAFT_TOPK1_BLOCK
    num_splits = triton.cdiv(vocab_size, block)
    partial_vals = torch.empty(
        (bs, num_splits), dtype=torch.float32, device=next_token_logits.device
    )
    partial_indices = torch.empty(
        (bs, num_splits), dtype=torch.int32, device=next_token_logits.device
    )

    _draft_topk1_partial_argmax_kernel[(bs, num_splits)](
        next_token_logits,
        partial_vals,
        partial_indices,
        next_token_logits.stride(0),
        vocab_size,
        num_splits,
        BLOCK=block,
        num_warps=8,
    )
    # Dummy operand for the disabled draft-token slot: the pointer must be
    # valid even though the kernel never dereferences it (gated off by
    # WRITE_DRAFT_TOKEN).
    _draft_topk1_finalize_kernel[(bs,)](
        partial_vals,
        partial_indices,
        topk_p,
        topk_index,
        positions,
        draft_tokens if write_draft_token else topk_index,
        draft_tokens.stride(0) if write_draft_token else 0,
        draft_token_column,
        num_splits,
        WRITE_DRAFT_TOKEN=write_draft_token,
        BLOCK=triton.next_power_of_2(num_splits),
        num_warps=1,
    )
    return topk_p, topk_index
