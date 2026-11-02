from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import (
    _torch_prefill_request_rows,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize(
    "mode", ["native", "missing", "mismatched", "wrong_total", "cp"]
)
def test_torch_prefill_request_rows(device, ratio, mode):
    positions = torch.tensor(
        [0, 4, 5, 7, 9] if mode == "cp" else [4, 5, 8, 9, 10],
        dtype=torch.int64,
        device=device,
    )
    rows = [2, 0, 3]
    if mode == "missing":
        rows = None
    elif mode == "mismatched":
        rows = [5]
    elif mode == "wrong_total":
        rows = [1, 1, 1]
    inputs = SimpleNamespace(
        compress_ratio=ratio,
        positions=positions,
        req_rows=(
            None
            if mode == "cp"
            else torch.tensor([9, 9, 7, 7, 7], dtype=torch.int32, device=device)
        ),
        req_pool_indices=torch.tensor([9, 2, 7], dtype=torch.int32, device=device),
        seq_lens_cpu=[6, 7, 11],
        rows_per_request=rows,
        rows_per_request_device=torch.tensor(
            [2, 0, 3], dtype=torch.int32, device=device
        ),
    )
    lengths = (positions + 1) // ratio
    if mode == "native":
        # The fast path must use scheduler metadata without any device-to-host
        # layout discovery, even for empty request rows and permuted pool IDs.
        with (
            patch.object(torch.Tensor, "tolist", side_effect=AssertionError),
            patch.object(torch.Tensor, "item", side_effect=AssertionError),
            patch.object(torch.Tensor, "nonzero", side_effect=AssertionError),
            patch.object(torch, "unique_consecutive", side_effect=AssertionError),
        ):
            actual = list(_torch_prefill_request_rows(inputs, lengths))
    else:
        actual = list(_torch_prefill_request_rows(inputs, lengths))

    assert len(actual) == 2
    expected_starts = [0, 6 // ratio + 7 // ratio]
    expected_lc = (
        [5 // ratio, 10 // ratio] if mode == "cp" else [6 // ratio, 11 // ratio]
    )
    expected_lengths = [[1, 5], [6, 8, 10]] if mode == "cp" else [[5, 6], [9, 10, 11]]
    for index, (start, tokens, lens, lc) in enumerate(actual):
        assert start == expected_starts[index]
        assert lc == expected_lc[index]
        torch.testing.assert_close(
            tokens, torch.tensor([[0, 1], [2, 3, 4]][index], device=device)
        )
        torch.testing.assert_close(
            lens,
            torch.tensor(expected_lengths[index], device=device) // ratio,
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("ratio", [1, 2])
def test_torch_prefill_request_rows_empty_batch(device, ratio):
    inputs = SimpleNamespace(
        compress_ratio=ratio,
        positions=torch.empty(0, dtype=torch.int64, device=device),
        req_rows=torch.empty(0, dtype=torch.int32, device=device),
        req_pool_indices=torch.tensor([9, 2], dtype=torch.int32, device=device),
        seq_lens_cpu=[7, 13],
        rows_per_request=[0, 0],
        rows_per_request_device=torch.zeros(2, dtype=torch.int32, device=device),
    )
    assert list(_torch_prefill_request_rows(inputs, inputs.positions)) == []
