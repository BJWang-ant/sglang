"""DeepEP auto must record exact routes across prefill and graph decode."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.eplb import expert_distribution as ed


class TestDeepEPAutoRecorder(unittest.TestCase):
    def make_gatherer(self, mode="auto"):
        metadata = SimpleNamespace(num_layers=2, num_physical_experts=8)
        execution = SimpleNamespace(
            moe=SimpleNamespace(
                expert_distribution_recorder_mode="stat",
                moe_a2a_backend="deepep",
                deepep_mode=mode,
                elastic_ep_backend=None,
            )
        )
        with (
            patch.object(ed, "get_exec", return_value=execution),
            patch.object(
                ed, "get_platform", return_value=SimpleNamespace(is_npu=False)
            ),
            patch.object(ed, "get_device", return_value="cpu"),
        ):
            return ed._SinglePassGatherer.init_new(metadata, 0)

    def test_auto_and_normal_use_exact_physical_route_counts(self):
        for mode in ("auto", "normal"):
            with self.subTest(mode=mode):
                gatherer = self.make_gatherer(mode)
                self.assertIsInstance(gatherer, ed._SelectExpertsSinglePassGatherer)
                gatherer.on_select_experts(
                    0, torch.tensor([[1, 1, -1], [2, 7, -1], [-1, -1, -1]])
                )
                gatherer.on_select_experts(1, torch.tensor([[3, 4, 4]]))
                # A dispatch hook must not count the selected routes again.
                gatherer.on_deepep_dispatch_normal(0, [9, 9], None, None, None)
                gatherer.on_deepep_dispatch_low_latency(1, torch.tensor([9, 9]))
                actual = gatherer.collect()["global_physical_count"]
                expected = torch.tensor(
                    [[0, 2, 1, 0, 0, 0, 0, 1], [0, 0, 0, 1, 2, 0, 0, 0]],
                    dtype=torch.int32,
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_reset_starts_a_new_forward_pass(self):
        gatherer = self.make_gatherer()
        gatherer.on_select_experts(1, torch.tensor([[1, 2, 3]]))
        gatherer.reset()
        gatherer.on_select_experts(0, torch.tensor([[0, 6, -1]]))
        actual = gatherer.collect()["global_physical_count"]
        expected = torch.zeros((2, 8), dtype=torch.int32)
        expected[0, [0, 6]] = 1
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_explicit_low_latency_retains_receive_count_gatherer(self):
        sentinel = object()
        with patch.object(
            ed, "_DeepepLowLatencySinglePassGatherer", return_value=sentinel
        ) as constructor:
            self.assertIs(self.make_gatherer("low_latency"), sentinel)
        self.assertFalse(constructor.call_args.kwargs["elastic_ep_enabled"])


if __name__ == "__main__":
    unittest.main()
