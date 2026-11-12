"""Hopper GC-MoE integration for the DeepSeek-V4.1 decode worker.

Routing stays with the model. GC-MoE owns the complete routed EP collective;
the ordinary replicated shared expert is added once on the source rank.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.runtime_context import (
    get_buffer,
    get_parallel,
    get_schedule,
    get_spec,
)

logger = logging.getLogger(__name__)


@dataclass
class _Workspace:
    buffer: object
    logged_forward: bool = False


def build_gc_moe_experts_weights(layer) -> None:
    """Transform the original checkpoint before another runner repacks it."""
    if getattr(layer, "_gc_moe_weights_built", False):
        return
    from gc_moe import transform_weights_for_gc_moe_sm90

    if torch.cuda.get_device_capability(layer.w13_weight.device) != (9, 0):
        raise ValueError("GC-MoE currently requires CUDA SM90")
    if layer.moe_tp_size != 1 or layer.num_fused_shared_experts:
        raise ValueError("GC-MoE requires MoE TP1 and unfused shared experts")
    if layer.w13_weight.dtype not in (torch.int8, torch.uint8):
        raise ValueError("GC-MoE requires the original packed MXFP4 checkpoint")
    l1, l2 = transform_weights_for_gc_moe_sm90(
        (layer.w13_weight.data, layer.w13_weight_scale_inv.data),
        (layer.w2_weight.data, layer.w2_weight_scale_inv.data),
    )
    layer.gc_l1_weights, layer.gc_l2_weights = l1, l2
    # Retain only the GC layout. There is no fallback after this transition.
    layer.w13_weight.data = l1[0]
    layer.w13_weight_scale_inv.data = l1[1]
    layer.w2_weight.data = l2[0]
    layer.w2_weight_scale_inv.data = l2[1]
    layer._gc_moe_weights_built = True
    layer._dsv4_mxfp4_backend = "gc_moe"


def _workspace_capacity() -> int:
    """Cover eager decode and full-width verify batches on every DP rank."""
    requests = get_schedule().max_running_requests
    if requests is None or requests <= 0:
        raise ValueError("GC-MoE requires a positive --max-running-requests")
    parallel = get_parallel()
    dp_size = parallel.attn_dp_size if parallel.attn_dp_enabled else 1
    requests_per_rank = (requests + dp_size - 1) // dp_size
    spec = get_spec()
    tokens_per_request = (
        (spec.speculative_num_draft_tokens or 1) if spec.speculative_algorithm else 1
    )
    # The environment setting is a minimum allocation. Verify can require
    # several rows per request even when the configured decode BS is small.
    return max(
        envs.SGLANG_OPT_GC_MOE_NUM_MAX_TOKENS_PER_RANK.get(),
        requests_per_rank * tokens_per_request,
    )


def _get_workspace(moe) -> _Workspace:
    from gc_moe import get_symm_buffer_for_gc_moe_sm90

    group = get_parallel().moe_ep_group.device_group
    capacity = _workspace_capacity()
    key = (
        id(group),
        torch.cuda.current_device(),
        moe.config.n_routed_experts,
        moe.config.num_experts_per_tok,
        moe.config.hidden_size,
        moe.config.moe_intermediate_size,
        capacity,
    )
    workspaces = get_buffer("gc_moe_sm90_workspaces", dict)
    if key not in workspaces:
        buffer = get_symm_buffer_for_gc_moe_sm90(
            group,
            num_experts=moe.config.n_routed_experts,
            num_max_tokens_per_rank=capacity,
            num_topk=moe.config.num_experts_per_tok,
            hidden=moe.config.hidden_size,
            intermediate_hidden=moe.config.moe_intermediate_size,
        )
        workspaces[key] = _Workspace(buffer)
        logger.info(
            "GC-MoE SM90 workspace: EP=%d, capacity=%d, memory=%.3f GiB",
            group.size(),
            buffer.num_max_tokens_per_rank,
            buffer.buffer.nbytes / (1 << 30),
        )
    return workspaces[key]


def allocate_gc_moe_buffers(model, *, is_draft: bool = False) -> None:
    """Charge shared workspaces before KV-cache sizing; initialize Green contexts."""
    if not get_moe_a2a_backend().is_gc_moe():
        return
    count = 0
    for module in model.modules():
        experts = getattr(module, "experts", None)
        if getattr(experts, "_gc_moe_weights_built", False):
            _get_workspace(module)
            count += 1
    if not count:
        if is_draft:
            # Dense drafts have no routed experts; the target owns GC resources.
            return
        raise RuntimeError("GC-MoE selected but no MXFP4 expert weights were prepared")
    from gc_moe.runtime.sm90_gc_moe import get_context

    context = get_context()
    logger.info(
        "GC-MoE SM90 ready: layers=%d, dispatch_sms=%d, compute_sms=%d, balanced=%s",
        count,
        context.dispatch_sms,
        context.compute_sms,
        context.balanced_workqueues,
    )


def forward_gc_moe(moe, hidden_states, forward_batch=None, input_ids_global=None):
    from sglang.kernels.ops.communication.gc_moe import sm90_gc_moe
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    if not getattr(moe.experts, "_gc_moe_weights_built", False):
        raise RuntimeError(
            "GC-MoE cannot fall back to a runner with another weight layout"
        )
    state = _get_workspace(moe)
    buf = state.buffer
    tokens = hidden_states.shape[0]
    if tokens > buf.num_max_tokens_per_rank:
        raise ValueError(
            f"GC-MoE local rows {tokens} exceed capacity {buf.num_max_tokens_per_rank}"
        )
    shared = moe._forward_shared_experts(hidden_states)
    if tokens:
        router_logits = moe.gate(hidden_states, forward_batch=forward_batch)
        kwargs = {"input_ids": input_ids_global} if moe.is_hash else {}
        routed = moe.topk(hidden_states, router_logits, **kwargs)
        buf.topk_idx[:tokens].copy_(routed.topk_ids)
        weights = routed.topk_weights
        if not moe.routed_scaling_factor_fused_in_topk:
            weights = weights * moe.routed_scaling_factor
        buf.topk_weights[:tokens].copy_(weights)
        sglang_per_token_group_quant_fp8(
            hidden_states,
            group_size=128,
            scale_ue8m0=False,
            output_q=buf.x[:tokens],
            output_s=buf.x_sf[:tokens],
        )
    y = torch.empty(
        (tokens, moe.config.hidden_size),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    # A common capacity bound is safe for asymmetric and zero-token DP ranks.
    # Per-batch synchronized bounds can be tuned after startup validation.
    sm90_gc_moe(
        y,
        moe.experts.gc_l1_weights,
        moe.experts.gc_l2_weights,
        buf,
        activation_clamp=getattr(moe.config, "swiglu_limit", None),
        num_tokens_bound=buf.num_max_tokens_per_rank,
    )
    if shared is not None:
        y.add_(shared)
    if not state.logged_forward:
        logger.info("GC-MoE SM90 first forward submitted: local_tokens=%d", tokens)
        state.logged_forward = True
    return y
