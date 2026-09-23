"""Ragged fp8 MQA logits for the DSA indexer on ROCm."""

from __future__ import annotations

import functools

import torch

# aiter's gfx950 wrapper sizes the tile from num_heads alone: a 32-head
# indexer gets decode-tuned BLOCK_KV=64/2 warps even at this path's prefill scale.
_PREFILL_BLOCK_KV = 128
_PREFILL_NUM_WARPS = 4

# aiter's own BLOCK_M=2 regime, which is what the retiling above targets.
_PREFILL_MIN_ROWS = 4096
_PREFILL_NUM_HEADS = 32

_BUFFER_LIMIT_BYTES = 2 * 1024 * 1024 * 1024
_LOGITS_ROW_ALIGN = 256


@functools.lru_cache(maxsize=1)
def _tuned_launch_config():
    """aiter's gluon kernel plus the two capability flags its wrapper derives,
    read from the wrapper module so this path cannot drift from it; None when
    this build would not reach that kernel anyway (older aiter without the
    gluon rewrite, or a module that never defines these attributes)."""
    from aiter.ops.triton import fp8_mqa_logits as aiter_mqa_logits

    try:
        kernel = aiter_mqa_logits._gluon_fp8_mqa_logits_kernel
        if kernel is None or aiter_mqa_logits.arch != "gfx950":
            return None
        return (
            kernel,
            aiter_mqa_logits.ASYNC_COPY_SUPPORTS_DISTRIBUTED,
            aiter_mqa_logits.FOLDED_REDUCTED_SUPPORT,
        )
    except AttributeError:
        return None


def _tuned_prefill_logits(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    cu_starts: torch.Tensor,
    cu_ends: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor | None:
    """The retiled launch, or None when the call is outside what was measured."""
    config = _tuned_launch_config()
    if config is None:
        return None
    kernel, padded_shared_layout, folded_reduction = config

    seq_len, num_heads, head_size = q_fp8.shape
    seq_len_kv = k_fp8.shape[0]
    if num_heads != _PREFILL_NUM_HEADS or seq_len <= _PREFILL_MIN_ROWS:
        return None
    if head_size & (head_size - 1):
        return None

    # The wrapper over-allocates each row to 256 elements and hands back a view,
    # which is what gives the logits a top-k-legal row stride; keep that.
    aligned_kv = (
        (seq_len_kv + _LOGITS_ROW_ALIGN - 1) // _LOGITS_ROW_ALIGN * _LOGITS_ROW_ALIGN
    )
    if clean_logits:
        logits = torch.full(
            (seq_len, aligned_kv),
            fill_value=-float("inf"),
            dtype=torch.float32,
            device=q_fp8.device,
        )
    else:
        logits = torch.empty(
            (seq_len, aligned_kv), dtype=torch.float32, device=q_fp8.device
        )
    logits = logits[:, :seq_len_kv]

    kernel[((seq_len + 1) // 2,)](
        Q_ptr=q_fp8,
        KV_ptr=k_fp8,
        kv_scales_ptr=k_scale,
        weights_ptr=weights,
        cu_start_ptr=cu_starts,
        cu_end_ptr=cu_ends,
        logits_ptr=logits,
        seq_len=seq_len,
        seq_len_kv=seq_len_kv,
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_size,
        stride_q_s=q_fp8.stride(0),
        stride_q_h=q_fp8.stride(1),
        stride_q_d=q_fp8.stride(2),
        stride_kv_s=k_fp8.stride(0),
        stride_kv_d=k_fp8.stride(1),
        stride_w_s=weights.stride(0),
        stride_w_h=weights.stride(1),
        stride_logits_s=logits.stride(0),
        stride_logits_k=logits.stride(1),
        BLOCK_KV=_PREFILL_BLOCK_KV,
        NUM_WARPS=_PREFILL_NUM_WARPS,
        NUM_BUFFERS=2,
        NUM_CHAINS=4 if folded_reduction else 0,
        # Buffer ops address through a 32-bit byte offset, as in the wrapper.
        USE_BUFFER_LOAD=k_fp8.numel() * k_fp8.element_size() < _BUFFER_LIMIT_BYTES,
        USE_BUFFER_STORE=logits.numel() * logits.element_size()
        < _BUFFER_LIMIT_BYTES,
        USE_PADDED_SHARED_LAYOUT=padded_shared_layout,
        BLOCK_M=2,
        MFMA_NONK_DIM=32,
        num_warps=_PREFILL_NUM_WARPS,
        waves_per_eu=4,
    )
    return logits


def aiter_ragged_mqa_logits(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    cu_starts: torch.Tensor,
    cu_ends: torch.Tensor,
    *,
    clean_logits: bool,
) -> torch.Tensor:
    """aiter's ragged fp8 MQA logits, retiled for the indexer's prefill launches.

    Same kernel and same bits as `aiter.ops.triton.fp8_mqa_logits`; only the
    tile the launch asks for differs, and only on the shapes priced above.
    Every other shape goes through the wrapper untouched.
    """
    logits = _tuned_prefill_logits(
        q_fp8, k_fp8, k_scale, weights, cu_starts, cu_ends, clean_logits
    )
    if logits is not None:
        return logits

    from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits

    return fp8_mqa_logits(
        q_fp8,
        k_fp8,
        k_scale,
        weights,
        cu_starts,
        cu_ends,
        clean_logits=clean_logits,
    )
