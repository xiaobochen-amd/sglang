"""AITER MegaMoEV2 adapter for ROCm gfx95x."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.distributed.parallel_state import get_moe_ep_group
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_exec
from sglang.srt.utils import is_gfx95_supported

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

logger = logging.getLogger(__name__)

_AITER_MEGA_MOE_RUNTIMES: dict[tuple, object] = {}
_AITER_MEGA_MOE_SHMEM_INITIALIZED = False
_AITER_MEGA_MOE_GROUP_NAME = "sglang_aiter_megamoe"


def build_aiter_mega_moe_experts_weights(experts) -> None:
    """Replace canonical MXFP4 weights with MegaMoEV2's layout."""
    if getattr(experts, "_aiter_mega_moe_weights_built", False):
        return
    if not is_gfx95_supported():
        raise RuntimeError("AITER MegaMoEV2 requires gfx95x (MI35x)")

    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    w13 = shuffle_weight_a16w4(
        experts.w13_weight.data.contiguous(), 16, True
    ).contiguous()
    w13_scale_src = experts.w13_weight_scale.data
    w13_scale = shuffle_scale_a16w4(
        w13_scale_src.view(-1, w13_scale_src.shape[-1]),
        experts.num_local_experts,
        True,
    ).contiguous()
    w2 = shuffle_weight_a16w4(
        experts.w2_weight.data.contiguous(), 16, False
    ).contiguous()
    w2_scale_src = experts.w2_weight_scale.data
    w2_scale = shuffle_scale_a16w4(
        w2_scale_src.view(-1, w2_scale_src.shape[-1]),
        experts.num_local_experts,
        False,
    ).contiguous()

    # MegaMoE is the only valid A2A path for these tensors. Repointing the
    # parameters releases the canonical layout instead of duplicating it for
    # every sparse layer.
    experts.w13_weight.data = w13
    experts.w13_weight_scale.data = w13_scale
    experts.w2_weight.data = w2
    experts.w2_weight_scale.data = w2_scale
    experts.mega_l1_weights = (w13, w13_scale)
    experts.mega_l2_weights = (w2, w2_scale)
    experts._aiter_mega_moe_weights_built = True
    experts._mega_moe_weights_built = True
    _preallocate_aiter_mega_moe_runtimes(experts)


def _initialize_mori_shmem(ep_group) -> None:
    global _AITER_MEGA_MOE_SHMEM_INITIALIZED
    if _AITER_MEGA_MOE_SHMEM_INITIALIZED:
        return

    # MORI reads this at shmem initialization. An explicit deployment value
    # still wins over the conservative single-node default.
    os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "40G")

    import mori.shmem as ms
    import torch._C._distributed_c10d as c10d

    try:
        c10d._register_process_group(_AITER_MEGA_MOE_GROUP_NAME, ep_group.cpu_group)
    except Exception as exc:
        if "already registered" not in str(exc):
            raise
    else:
        ms.shmem_torch_process_group_init(_AITER_MEGA_MOE_GROUP_NAME)
    _AITER_MEGA_MOE_SHMEM_INITIALIZED = True


def _select_max_tokens(
    num_tokens: int, forward_batch: Optional[ForwardBatch]
) -> int:
    is_decode = forward_batch is not None and (
        forward_batch.forward_mode.is_decode()
        or forward_batch.forward_mode.is_target_verify()
    )
    cap = (
        envs.SGLANG_AITER_MEGA_MOE_DECODE_MAX_TOKENS_PER_RANK.get()
        if is_decode
        else envs.SGLANG_AITER_MEGA_MOE_PREFILL_MAX_TOKENS_PER_RANK.get()
    )
    _validate_max_tokens(cap)
    if num_tokens > cap:
        phase = "decode" if is_decode else "prefill"
        raise ValueError(
            f"AITER MegaMoE {phase} tokens={num_tokens} exceed configured cap={cap}"
        )
    return cap


def _validate_max_tokens(cap: int) -> None:
    if cap <= 0 or cap & (cap - 1):
        raise ValueError(
            "AITER MegaMoE max tokens per rank must be a positive power of two, "
            f"got {cap}"
        )
    if cap > 32768:
        raise ValueError(
            f"AITER MegaMoE supports at most 32768 tokens per rank, got {cap}"
        )


def _get_runtime_for_experts(
    experts,
    *,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    swiglu_limit: float,
    max_tokens_per_rank: int,
):
    ep_group = get_moe_ep_group()
    ep_rank = ep_group.rank_in_group
    ep_size = ep_group.world_size
    current_device = torch.cuda.current_device()

    if ep_size > 8:
        raise ValueError(
            f"AITER MegaMoEV2 is intranode-only and supports EP <= 8, got {ep_size}"
        )
    if current_device != ep_rank:
        raise ValueError(
            "AITER MegaMoEV2 currently requires EP rank to match the local CUDA "
            f"device, got ep_rank={ep_rank}, device={current_device}"
        )
    if get_exec().moe.enable_eplb:
        raise ValueError("AITER MegaMoEV2 does not support EPLB weight migration")
    if experts.num_fused_shared_experts:
        raise ValueError(
            "AITER MegaMoEV2 requires shared experts to remain outside the fused kernel"
        )

    num_experts = experts.num_experts
    if num_experts % ep_size:
        raise ValueError(
            f"AITER MegaMoEV2 experts={num_experts} must be divisible by EP={ep_size}"
        )
    if experts.num_local_experts != num_experts // ep_size:
        raise ValueError(
            "AITER MegaMoEV2 requires one contiguous local expert shard per EP rank"
        )

    key = (
        id(ep_group.cpu_group),
        current_device,
        ep_size,
        num_experts,
        top_k,
        hidden_size,
        intermediate_size,
        max_tokens_per_rank,
        swiglu_limit,
    )

    runtime = _AITER_MEGA_MOE_RUNTIMES.get(key)
    l1_weight, l1_scale = experts.mega_l1_weights
    l2_weight, l2_scale = experts.mega_l2_weights
    if runtime is None:
        _initialize_mori_shmem(ep_group)
        from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2

        runtime = MegaMoEV2(
            rank=ep_rank,
            world_size=ep_size,
            model_dim=hidden_size,
            inter_dim=intermediate_size,
            experts=num_experts,
            topk=top_k,
            quant="a8w4",
            w1=l1_weight,
            w1_scale=l1_scale,
            w2=l2_weight,
            w2_scale=l2_scale,
            max_tok_per_rank=max_tokens_per_rank,
            swiglu_limit=swiglu_limit,
        )
        if not hasattr(runtime, "set_weights"):
            raise RuntimeError(
                "Installed AITER MegaMoEV2 lacks set_weights(); use the paired "
                "feat/mi355x-mega-moe-v2 AITER branch"
            )
        _AITER_MEGA_MOE_RUNTIMES[key] = runtime
        logger.info(
            "AITER MegaMoEV2 engaged: EP=%d E=%d topk=%d H=%d I=%d mtpr=%d",
            ep_size,
            num_experts,
            top_k,
            hidden_size,
            intermediate_size,
            max_tokens_per_rank,
        )
    else:
        runtime.set_weights(l1_weight, l1_scale, l2_weight, l2_scale)
    return runtime


def _preallocate_aiter_mega_moe_runtimes(experts) -> None:
    """Reserve workspaces before SGLang sizes its static KV cache pool."""
    decode_cap = envs.SGLANG_AITER_MEGA_MOE_DECODE_MAX_TOKENS_PER_RANK.get()
    prefill_cap = envs.SGLANG_AITER_MEGA_MOE_PREFILL_MAX_TOKENS_PER_RANK.get()
    swiglu_limit = float(experts.moe_runner_config.swiglu_limit or 0.0)
    for cap in dict.fromkeys((decode_cap, prefill_cap)):
        _validate_max_tokens(cap)
        _get_runtime_for_experts(
            experts,
            hidden_size=experts.hidden_size,
            intermediate_size=experts.intermediate_size_per_partition,
            top_k=experts.top_k,
            swiglu_limit=swiglu_limit,
            max_tokens_per_rank=cap,
        )


def _get_runtime(moe: DeepseekV2MoE, max_tokens_per_rank: int):
    return _get_runtime_for_experts(
        moe.experts,
        hidden_size=moe.config.hidden_size,
        intermediate_size=moe.config.moe_intermediate_size,
        top_k=moe.config.num_experts_per_tok,
        swiglu_limit=float(getattr(moe.config, "swiglu_limit", None) or 0.0),
        max_tokens_per_rank=max_tokens_per_rank,
    )


def run_aiter_mega_moe(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    forward_batch: Optional[ForwardBatch],
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    if num_tokens == 0:
        return hidden_states.new_empty((0, hidden_states.shape[-1]))

    max_tokens_per_rank = _select_max_tokens(num_tokens, forward_batch)
    runtime = _get_runtime(moe, max_tokens_per_rank)
    return runtime(
        hidden_states.contiguous(),
        topk_weights.to(dtype=torch.float32).contiguous(),
        topk_ids.to(dtype=torch.int32).contiguous(),
    )
