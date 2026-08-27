from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    is_hip_runtime,
    load_jit,
    make_cpp_args,
)

from .utils import make_name


@cache_once
def _jit_topk_v1_module():
    # topk (<= 1024) is a runtime argument, not a compile-time constant, so a
    # single module serves every k. Baking it in via -DSGL_TOPK used to build one
    # module per k, and since the macro fed a `constexpr` rather than a template
    # parameter every module exported identically mangled symbols -- see the
    # comment in topk_v1.cuh for how that broke the second module's launch.
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        make_name("topk_v1"),
        *args,
        cuda_files=["deepseek_v4/topk_v1.cuh"],
        cuda_wrappers=[("topk_transform", f"TopKKernel<{args}>::transform")],
    )


@cache_once
def _jit_topk_v2_module():
    # v2 is universal: topk (<= 2048) is a runtime argument, not a compile-time
    # constant, so a single module serves every k.
    return load_jit(
        make_name("topk_v2"),
        cuda_files=["deepseek_v4/topk_v2.cuh"],
        cuda_wrappers=[
            ("topk_transform", "TopKKernel::transform"),
            ("topk_plan", "TopKKernel::plan"),
        ],
    )


def topk_transform_512(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    if is_hip_runtime():
        torch.ops.sgl_kernel.deepseek_v4_topk_transform_512(
            scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices
        )
    else:
        module = _jit_topk_v1_module()
        module.topk_transform(
            scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices
        )


# aiter templates its kernel on k, so only the instantiated values are callable.
_AITER_SUPPORTED_K = frozenset({512, 2048})


@cache_once
def _aiter_dsa_topk_transform():
    if not is_hip_runtime():
        return None
    try:
        from aiter import dsa_topk_transform
    except ImportError:
        return None
    return dsa_topk_transform


def topk_transform_512_aiter_supported(
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor],
) -> bool:
    """Whether :func:`topk_transform_512_aiter` can serve this call.

    The aiter kernel emits only the page-mapped slots, so a caller that also
    wants the pre-transform positions (capture, hisparse swap-in) has to stay on
    the v1/v2 path.
    """
    if _aiter_dsa_topk_transform() is None or out_raw_indices is not None:
        return False
    return (
        out_page_indices.shape[-1] in _AITER_SUPPORTED_K
        and page_size > 0
        and (page_size & (page_size - 1)) == 0
    )


def topk_transform_512_aiter(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
) -> None:
    """Fused top-k + page-table transform on aiter's cooperative radix select.

    Same contract as :func:`topk_transform_512` minus ``out_raw_indices``: float32
    ``scores`` [B, max_seq_len] contiguous along the last dim, int32 ``seq_lens``,
    int32 ``page_tables`` [B, num_pages], and the winners written into int32
    ``out_page_indices`` [B, k] with short rows padded to -1.
    """
    fn = _aiter_dsa_topk_transform()
    assert fn is not None, "aiter is unavailable; check topk_transform_512_aiter_supported first"
    # rowStarts=None means every row starts at 0, which is what decode wants and
    # saves the per-call zeros tensor.
    fn(
        scores,
        None,
        seq_lens,
        page_tables,
        out_page_indices,
        page_size,
        out_page_indices.shape[-1],
    )


# metadata is (batch+1, 2) int32: row 0 = {cluster_threshold, num_cluster_items};
# rows 1..N = {batch_id, seq_len} of items routed to the persistent cluster pool.
_PLAN_METADATA_INTS_PER_BATCH = 2


def plan_topk_v2(seq_lens: torch.Tensor, static_threshold: int = 0) -> torch.Tensor:
    """Preprocess the per-batch routing plan for :func:`topk_transform_512_v2`.

    IMPORTANT: every entry of ``seq_lens`` must be NON-NEGATIVE. The device
    kernel reads the int32 buffer as ``uint32_t``, so a negative length (e.g.
    -4 from a DP-padded / idle-companion row) reinterprets as ~4e9, poisons
    the plan, and drives the transform kernel into an illegal memory access.
    Producers of padded rows must clamp their lengths to 0 (0 selects the
    trivial all-(-1) output path, which is safe).
    """
    module = _jit_topk_v2_module()
    bs = seq_lens.shape[0]
    metadata = seq_lens.new_empty(bs + 1, _PLAN_METADATA_INTS_PER_BATCH)
    module.topk_plan(seq_lens, metadata, static_threshold)
    return metadata


def topk_transform_512_v2(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    metadata: torch.Tensor,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    """Fused top-k + page-table transform (DeepSeek-V4 top-k v2 kernel).

    IMPORTANT: every entry of ``seq_lens`` must be NON-NEGATIVE, and
    ``metadata`` must come from :func:`plan_topk_v2` over the same ``seq_lens``
    values. The kernel reads lengths as ``uint32_t``: a negative entry
    reinterprets as a ~4e9-token sequence, sending the row down the cluster
    path over garbage scores and crashing with an illegal memory access
    (GLM 5.2 MTP DP-idle companion rows hit exactly this). A length of 0 is
    the valid way to express "no tokens": the row takes the trivial path and
    the output is all -1.
    """
    module = _jit_topk_v2_module()
    module.topk_transform(
        scores,
        seq_lens,
        page_tables,
        out_page_indices,
        page_size,
        metadata,
        out_raw_indices,
    )
