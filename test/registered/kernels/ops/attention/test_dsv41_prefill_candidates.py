"""Prefill candidate selection: causal tails, strided CP slices, and replay."""

import pytest
import torch
import torch.nn.functional as F
import triton

from sglang.kernels.ops.attention.dsv4.prefill_candidates import (
    causal_block_max,
    select_prefill_candidate_blocks,
    topk_prefill_candidates,
)
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_ragged_v2
from sglang.srt.layers.attention.dsv4.candidate_indexer import (
    PrefillCandidateBlocks,
    mask_topk_scores,
    select_candidate_blocks,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def reference_scores(logits, lens, block):
    visible = logits.masked_fill(
        torch.arange(logits.shape[1], device=logits.device)[None, :] >= lens[:, None],
        -torch.inf,
    )
    scores = F.pad(visible, (0, -logits.shape[1] % block), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block)).amax(-1)
    last = (lens[:, None] - 1) // block
    scores = scores.masked_fill(
        torch.arange(scores.shape[1], device=logits.device) == last, torch.inf
    )
    return visible, scores


@pytest.mark.parametrize("width", [1, 7, 8, 9, 513, 8193, 67045])
@pytest.mark.parametrize("block", [3, 8, 16])
@pytest.mark.parametrize("tied", [False, True])
def test_source_matches_reference(width, block, tied):
    torch.manual_seed(42)
    rows = 9
    # Noncontiguous rows and columns exercise request views and CP metadata.
    storage = torch.randn((rows + 2, 2 * (width + 32)), device="cuda")
    logits = storage[1:-1, : 2 * width : 2]
    if tied:
        logits.round_()
    before = storage.clone()
    lens = torch.tensor(
        [0, 1, min(width, block - 1), min(width, block), min(width, block + 1),
         width // 2, max(0, width - 1), width, width], device="cuda", dtype=torch.int32
    )
    visible, ref_scores = reference_scores(logits, lens, block)
    actual = causal_block_max(logits, lens[:, None], block)
    torch.testing.assert_close(actual, ref_scores, rtol=0, atol=0)
    for k in [1, 3, 2048]:
        ref = select_candidate_blocks(visible, lens[:, None], k, block)
        actual = select_prefill_candidate_blocks(logits, lens[:, None], k, block)
        if not tied:
            assert torch.equal(actual, ref)
        else:
            # topk does not promise a stable choice among equal scores.
            actual_blocks = actual[:, ::block]
            ref_blocks = ref[:, ::block]
            assert torch.equal(actual_blocks.sum(-1), ref_blocks.sum(-1))
            for row in range(rows):
                torch.testing.assert_close(
                    ref_scores[row][actual_blocks[row]].sort().values,
                    ref_scores[row][ref_blocks[row]].sort().values,
                    rtol=0, atol=0,
                )
                if lens[row] > 0:
                    assert actual_blocks[row, (lens[row] - 1) // block]
    assert torch.equal(storage, before)


def test_source_zero_rows_and_width():
    for rows, width in [(0, 9), (3, 0), (0, 0)]:
        scores = torch.empty((rows, width), device="cuda")
        lens = torch.zeros(rows, dtype=torch.int32, device="cuda")
        out = select_prefill_candidate_blocks(scores, lens, 2, 8)
        assert out.shape == scores.shape


def test_source_nan_and_infinity():
    logits = torch.tensor(
        [[float("nan"), 1, 2, 3, 4, 5, 6, 7, 8],
         [float("-inf")] * 9, [float("inf")] * 9],
        device="cuda", dtype=torch.float32
    )
    lens = torch.tensor([9, 0, 7], dtype=torch.int32, device="cuda")
    _, ref = reference_scores(logits, lens, 4)
    torch.testing.assert_close(causal_block_max(logits, lens, 4), ref, equal_nan=True)


def test_source_cuda_graph_replay():
    logits = torch.randn((17, 8193), device="cuda")
    lens = torch.full((17,), 8193, device="cuda", dtype=torch.int32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            select_prefill_candidate_blocks(logits, lens, 32, 8)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = select_prefill_candidate_blocks(logits, lens, 32, 8)
    for length in [1, 513, 8193]:
        logits.normal_()
        lens.fill_(length)
        graph.replay()
        visible, _ = reference_scores(logits, lens, 8)
        ref = select_candidate_blocks(visible, lens[:, None], 32, 8)
        assert torch.equal(actual, ref)


@pytest.mark.parametrize("width", [7, 511, 513, 8192, 8193, 16384, 16385, 67045])
@pytest.mark.parametrize("k", [512, 2048])
@pytest.mark.parametrize("mode", ["none", "all", "sparse", "underfill"])
def test_consumer_matches_masked_topk(width, k, mode):
    torch.manual_seed(43)
    rows, block = 9, 8
    padded_width = triton.cdiv(width, 32) * 32
    storage = torch.randn((rows + 2, padded_width), device="cuda")
    logits = storage[1:-1, :width]
    before = storage.clone()
    lens = torch.tensor(
        [0, 1, min(width, 7), min(width, 8), min(width, 9),
         width//2, max(0,width-1), width, width], device="cuda",dtype=torch.int32
    )
    blocks = triton.cdiv(width, block)
    mask_storage = torch.rand((rows, blocks+3), device="cuda") > 0.75
    mask = mask_storage[:, :blocks]
    if mode == "none":
        mask.fill_(False)
    elif mode == "all":
        mask.fill_(True)
    elif mode == "underfill":
        mask.fill_(False)
        mask[:, 0] = True
    # Vary request-local to flattened-KV offsets; these are not score row starts.
    offsets = torch.arange(rows, device="cuda", dtype=torch.int32) * 100003
    expanded = mask.repeat_interleave(block, dim=-1)[:, :width]
    ref_storage = storage.clone()
    ref_scores = ref_storage[1:-1, :width]
    ref_scores.masked_fill_(~expanded, -torch.inf)
    ref = torch.empty((rows,k),device="cuda",dtype=torch.int32)
    topk_transform_ragged_v2(ref_scores,lens,out_offsets=offsets,out_indices=ref)
    ref = mask_topk_scores(ref_scores,ref,offsets)
    actual = torch.empty_like(ref)
    topk_prefill_candidates(logits,lens,mask,block,offsets,actual)
    assert torch.equal(storage,before)
    assert torch.equal(actual.sort(-1).values,ref.sort(-1).values)


def test_consumer_ties_and_nonfinite():
    torch.manual_seed(44)
    rows,width,k,block=5,16387,512,8
    storage=torch.randn((rows,triton.cdiv(width,32)*32),device="cuda").round_()
    logits=storage[:,:width]
    logits[0].fill_(-torch.inf)
    logits[1].zero_()
    logits[2,0]=torch.inf
    lens=torch.full((rows,),width,device="cuda",dtype=torch.int32)
    mask=torch.rand((rows,triton.cdiv(width,block)),device="cuda")>0.9
    mask[2,0]=True
    offsets=torch.zeros(rows,device="cuda",dtype=torch.int32)
    out=torch.empty((rows,k),device="cuda",dtype=torch.int32)
    topk_prefill_candidates(logits,lens,mask,block,offsets,out)
    expanded=mask.repeat_interleave(block,-1)[:,:width]
    ref=logits.masked_fill(~expanded,-torch.inf).topk(k,dim=-1).values
    for row in range(rows):
        chosen=out[row][out[row]>=0].long()
        assert chosen.unique().numel()==chosen.numel()
        assert expanded[row,chosen].all()
        expected=ref[row][ref[row]>-torch.inf].sort().values
        torch.testing.assert_close(logits[row,chosen].sort().values,expected,rtol=0,atol=0)


def test_consumer_cuda_graph_tail_rows():
    torch.manual_seed(45)
    rows,width,k,block=13,8193,512,8
    logits_storage=torch.randn((rows,triton.cdiv(width,32)*32),device="cuda")
    logits=logits_storage[:,:width]
    lens=torch.full((rows,),width,device="cuda",dtype=torch.int32)
    offsets=torch.arange(rows,device="cuda",dtype=torch.int32)*width
    full=PrefillCandidateBlocks(
        torch.rand((30,triton.cdiv(width,block)),device="cuda")>0.6,block
    )
    tail_rows=torch.tensor([1,2,5,7,9,11,12,16,17,21,24,27,29],device="cuda")
    candidates=full.select_rows(tail_rows)
    assert torch.equal(candidates.mask,full.mask[tail_rows])
    out=torch.empty((rows,k),device="cuda",dtype=torch.int32)
    # Compile/warm up before capture on a separate stream.
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            topk_prefill_candidates(logits,lens,candidates.mask,block,offsets,out)
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        topk_prefill_candidates(logits,lens,candidates.mask,block,offsets,out)
    for length in [0,7,8193]:
        lens.fill_(length)
        candidates.mask.copy_(torch.rand_like(candidates.mask,dtype=torch.float32)>0.5)
        graph.replay()
        ref=logits_storage.clone()[:,:width]
        ref.masked_fill_(~candidates.mask.repeat_interleave(block,-1)[:,:width],-torch.inf)
        expected=torch.empty_like(out)
        topk_transform_ragged_v2(ref,lens,out_offsets=offsets,out_indices=expected)
        expected=mask_topk_scores(ref,expected,offsets)
        assert torch.equal(out.sort(-1).values,expected.sort(-1).values)


def test_publish_consume_ragged_cp_rows(monkeypatch):
    from types import SimpleNamespace

    from sglang.srt.layers.attention import deepseek_v4_backend as backend_module
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend

    # Force multiple source chunks even for these small rows.
    monkeypatch.setattr(backend_module, "_TORCH_INDEXER_SCORE_BUDGET_BYTES", 16)

    torch.manual_seed(46)
    # A request with no local rows, a request with no compressed KV, then two
    # different query/context lengths. Layout corresponds to one CP rank.
    q_lens, kv_lens = [0, 2, 3, 4], [31, 0, 17, 31]
    logits = torch.randn((9, 32), device="cuda")
    original = logits.clone()
    lens = torch.tensor([0, 0, 1, 8, 17, 7, 16, 23, 31], device="cuda", dtype=torch.int32)
    backend = SimpleNamespace(forward_metadata=SimpleNamespace(candidate_metadata=None))
    indexer = SimpleNamespace(
        is_candidate_source=True, candidate_topk_blocks=2, candidate_block_size=8
    )
    empty = torch.empty((0, 0), dtype=torch.bool, device="cuda")
    publish = DeepseekV4AttnBackend._publish_or_consume_candidates
    assert publish(backend, indexer, logits, lens, kv_lens, q_lens, empty) is None
    candidates = backend.forward_metadata.candidate_metadata
    assert isinstance(candidates, PrefillCandidateBlocks)
    assert torch.equal(logits, original)
    assert not candidates.mask[:2].any()
    start = 0
    for count, width in zip(q_lens, kv_lens):
        if count and width:
            visible, _ = reference_scores(logits[start:start+count, :width],
                                          lens[start:start+count], 8)
            expected = select_candidate_blocks(visible, lens[start:start+count,None], 2, 8)
            expanded = candidates.mask[start:start+count].repeat_interleave(8,-1)
            assert torch.equal(expanded[:, :width], expected)
            assert not expanded[:, width + (-width % 8):].any()
        start += count
    indexer.is_candidate_source = False
    assert publish(backend,indexer,logits,lens,kv_lens,q_lens,empty) is candidates


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
