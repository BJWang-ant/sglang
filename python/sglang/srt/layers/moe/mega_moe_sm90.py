# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""SM90 FP8 Mega-MoE forward path and expert-weight prep."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8

try:
    import deep_gemm
except ImportError:
    deep_gemm = None

logger = logging.getLogger(__name__)

from sglang.srt.models.deepseek_common.utils import _device_sm

if TYPE_CHECKING:
    from deep_gemm import SymmBuffer


def is_sm90_mega_moe_available(experts) -> bool:
    if _device_sm != 90 or deep_gemm is None:
        return False
    if getattr(experts, "_mega_moe_sm90_mxfp4_weights", False):
        return hasattr(deep_gemm, "fp8_mxfp4_mega_moe")
    return hasattr(deep_gemm, "fp8_mega_moe") and getattr(
        experts, "_mega_moe_sm90_fp8_weights", False
    )


def run_sm90_mega_routed(
    experts,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    buf: SymmBuffer,
    num_tokens: int,
    *,
    hidden_size: int,
    activation_clamp: Optional[float] = None,
    routed_scaling_factor: float = 1.0,
) -> torch.Tensor:
    # SM90 takes FP32 scales per 128 values, unlike SM100's packed UE8M0.
    # Reuse the existing graph-safe quantizer and write into the peer buffer.
    sglang_per_token_group_quant_fp8(
        hidden_states.contiguous(),
        group_size=128,
        output_q=buf.x[:num_tokens].view(torch.float8_e4m3fn),
        output_s=buf.x_sf[:num_tokens],
    )
    buf.topk_idx.fill_(-1)
    buf.topk_weights.zero_()
    buf.topk_idx[:num_tokens].copy_(topk_ids)
    buf.topk_weights[:num_tokens].copy_(topk_weights)
    if routed_scaling_factor != 1.0:
        buf.topk_weights[:num_tokens].mul_(routed_scaling_factor)

    y = torch.empty(
        (max(num_tokens, 1), hidden_size),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    is_mxfp4 = getattr(experts, "_mega_moe_sm90_mxfp4_weights", False)
    kernel = deep_gemm.fp8_mxfp4_mega_moe if is_mxfp4 else deep_gemm.fp8_mega_moe
    kernel(
        y,
        experts.mega_l1_weights,
        experts.mega_l2_weights,
        buf,
        recipe=(1, 1, 32) if is_mxfp4 else (128, 128, 128),
        activation="swiglu",
        activation_clamp=activation_clamp,
        fast_math=True,
    )
    y = y[:num_tokens]

    return y


def _interleave_l1_weight_only(weight: torch.Tensor, gran: int = 8) -> torch.Tensor:
    num_groups, n, *rest = weight.shape
    half = n // 2
    gate = weight[:, :half].reshape(num_groups, half // gran, gran, *rest)
    up = weight[:, half:].reshape(num_groups, half // gran, gran, *rest)
    return torch.stack([gate, up], dim=2).reshape(num_groups, n, *rest)


def build_sm90_mega_moe_experts_weights(experts) -> None:
    if getattr(experts, "_mega_moe_weights_built", False):
        return

    w13 = experts.w13_weight.data
    w13_sf_fp32 = experts.w13_weight_scale_inv.data
    w2 = experts.w2_weight.data
    w2_sf_fp32 = experts.w2_weight_scale_inv.data

    assert w13.dtype == torch.float8_e4m3fn
    assert w2.dtype == torch.float8_e4m3fn

    num_groups, n1, k1 = w13.shape
    _, n2, k2 = w2.shape
    scale_group_mn, scale_group_k = 128, 128

    assert k1 % scale_group_k == 0 and k2 % scale_group_k == 0, (
        f"invalid SM90 mega-moe K/group_size: k1={k1}, k2={k2}, group_k={scale_group_k}"
    )
    expected_n_groups_1 = (n1 + scale_group_mn - 1) // scale_group_mn
    expected_n_groups_2 = (n2 + scale_group_mn - 1) // scale_group_mn
    expected_k_groups_1 = k1 // scale_group_k
    expected_k_groups_2 = k2 // scale_group_k
    assert w13_sf_fp32.shape[1] == expected_n_groups_1, (
        f"w13 scale N groups mismatch: got {w13_sf_fp32.shape[1]}, "
        f"expected {expected_n_groups_1} (n1={n1}, group_mn={scale_group_mn})"
    )
    assert w2_sf_fp32.shape[1] == expected_n_groups_2, (
        f"w2 scale N groups mismatch: got {w2_sf_fp32.shape[1]}, "
        f"expected {expected_n_groups_2} (n2={n2}, group_mn={scale_group_mn})"
    )
    assert w13_sf_fp32.shape[2] == expected_k_groups_1, (
        f"w13 scale K groups mismatch: got {w13_sf_fp32.shape[2]}, "
        f"expected {expected_k_groups_1} (k1={k1}, group_k={scale_group_k})"
    )
    assert w2_sf_fp32.shape[2] == expected_k_groups_2, (
        f"w2 scale K groups mismatch: got {w2_sf_fp32.shape[2]}, "
        f"expected {expected_k_groups_2} (k2={k2}, group_k={scale_group_k})"
    )

    w13_interleaved = _interleave_l1_weight_only(w13)
    experts.w13_weight.data = w13_interleaved
    experts.mega_l1_weights = (
        experts.w13_weight.data,
        experts.w13_weight_scale_inv.data,
    )
    experts.mega_l2_weights = (
        experts.w2_weight.data,
        experts.w2_weight_scale_inv.data,
    )

    experts._mega_moe_sm90_fp8_weights = True
    experts._mega_moe_weights_built = True


def build_sm90_mxfp4_mega_moe_experts_weights(experts) -> None:
    """Convert checkpoint MXFP4 directly to PR #473's tiled Hopper layout."""
    if getattr(experts, "_mega_moe_weights_built", False):
        return
    if deep_gemm is None or not hasattr(
        deep_gemm, "transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90"
    ):
        raise RuntimeError("SM90 MXFP4 MegaMoE requires DeepGEMM PR #473.")
    l1, l2 = deep_gemm.transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90(
        (experts.w13_weight.data, experts.w13_weight_scale_inv.data),
        (experts.w2_weight.data, experts.w2_weight_scale_inv.data),
        layout="tiled",
    )
    # Keep all transformed expert payloads in parameters so EPLB moves the
    # secondary scale together with the packed values and relative scales.
    experts.w13_weight.data = l1[0]
    experts.w13_weight_scale_inv.data = l1[1]
    experts.w2_weight.data = l2[0]
    experts.w2_weight_scale_inv.data = l2[1]
    experts.register_parameter(
        "mega_w13_weight_scale_2", torch.nn.Parameter(l1[2], requires_grad=False)
    )
    experts.register_parameter(
        "mega_w2_weight_scale_2", torch.nn.Parameter(l2[2], requires_grad=False)
    )
    experts.mega_l1_weights = l1
    experts.mega_l2_weights = l2
    experts._mega_moe_sm90_mxfp4_weights = True
    experts._mega_moe_weights_built = True
    logger.info("Prepared SM90 fused MXFP4 MegaMoE experts (DeepGEMM PR #473, tiled)")
