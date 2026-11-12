"""GC-MoE capacity must cover full-width verify rows on every DP rank."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.gc_moe import (
    _workspace_capacity,
    allocate_gc_moe_buffers,
)
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGCMoeCapacity(CustomTestCase):
    def test_verify_rows_are_not_request_counts(self):
        """A 128-row hint must not truncate 32 requests with six verify rows."""
        with (
            get_context().override_server_args(
                tp_size=4,
                ep_size=4,
                attn_dp_size=4,
                max_running_requests=128,
                speculative_algorithm="DSPARK",
                speculative_num_draft_tokens=6,
            ),
            envs.SGLANG_OPT_GC_MOE_NUM_MAX_TOKENS_PER_RANK.override(128),
        ):
            for rank in range(4):
                with (
                    self.subTest(rank=rank),
                    get_parallel().override(attn_dp_rank=rank),
                ):
                    self.assertEqual(_workspace_capacity(), 192)

    def test_unbounded_request_pool_is_rejected(self):
        with get_context().override_server_args(max_running_requests=None):
            with self.assertRaisesRegex(ValueError, "max-running-requests"):
                _workspace_capacity()

    def test_dense_draft_does_not_load_gc_dependencies(self):
        """Only the target must contain GC-prepared experts."""
        with (
            get_context().override_server_args(moe_a2a_backend="gc_moe"),
            get_flags().moe.override(a2a_backend=MoeA2ABackend.GC_MOE),
            patch.dict("sys.modules", {"gc_moe": None}),
        ):
            allocate_gc_moe_buffers(torch.nn.Module(), is_draft=True)
            with self.assertRaisesRegex(RuntimeError, "no MXFP4 expert weights"):
                allocate_gc_moe_buffers(torch.nn.Module())


if __name__ == "__main__":
    unittest.main()
