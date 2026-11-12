"""GC-MoE SM90 fused EP dispatch, expert computation and combine.

Requires Hopper, GC-owned symmetric memory, transformed MXFP4 routed weights,
K128 E4M3 activations with FP32 scales, and BF16 output. Calls sharing a
workspace must be serialized on every rank.
"""


def sm90_gc_moe(
    y,
    l1_weights,
    l2_weights,
    workspace,
    *,
    activation_clamp=None,
    num_tokens_bound=None,
):
    from gc_moe import sm90_fp8_fp4_gc_moe

    sm90_fp8_fp4_gc_moe(
        y,
        l1_weights,
        l2_weights,
        workspace,
        recipe=(1, 1, 32),
        activation="swiglu",
        activation_clamp=activation_clamp,
        num_tokens_bound=num_tokens_bound,
    )
