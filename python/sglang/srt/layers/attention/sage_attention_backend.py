from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

try:
    from sageattention import sageattn

    _sageattention_available = True
except ImportError:
    _sageattention_available = False


class SageAttentionBackend(AttentionBackend):
    """Attention backend using SageAttention2 (INT8/FP8 quantized attention)."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        if not _sageattention_available:
            raise ImportError(
                "SageAttention is not installed. "
                "Please install it with: pip install sageattention"
            )
        self.forward_metadata = None
        self.device = model_runner.device

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        pass

    def _run_sage_forward_extend(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        scaling=None,
        causal=False,
    ):
        """Run the extend forward by using SageAttention.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_kv_heads, head_size]
            v_cache: [max_total_num_tokens, num_kv_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            extend_prefix_lens: [num_seqs]
            extend_seq_lens: [num_seqs]
            scaling: float or None
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """
        assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]

        start_q = 0
        for seq_idx in range(seq_lens.shape[0]):
            extend_seq_len_q = extend_seq_lens[seq_idx]
            prefill_seq_len_q = extend_prefix_lens[seq_idx]
            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + extend_seq_len_q

            per_req_query = query[start_q:end_q, :, :]

            # Build a redundant query padded to kv length so that
            # qo_len == kv_len, which is required by sageattn's is_causal.
            per_req_query_redundant = torch.empty(
                (seq_len_kv, per_req_query.shape[1], per_req_query.shape[2]),
                dtype=per_req_query.dtype,
                device=per_req_query.device,
            )
            per_req_query_redundant[prefill_seq_len_q:, :, :] = per_req_query

            # Gather KV from cache
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens]
            per_req_value = v_cache[per_req_tokens]

            # Align dtypes
            if not (
                per_req_query.dtype == per_req_key.dtype == per_req_value.dtype
            ):
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)

            # Reshape to HND: [1, heads, seq, dim]
            q_hnd = per_req_query_redundant.permute(1, 0, 2).unsqueeze(0)
            k_hnd = per_req_key.permute(1, 0, 2).unsqueeze(0)
            v_hnd = per_req_value.permute(1, 0, 2).unsqueeze(0)

            attn_out = sageattn(
                q_hnd,
                k_hnd,
                v_hnd,
                tensor_layout="HND",
                is_causal=causal,
                sm_scale=scaling,
            )

            # attn_out: [1, heads, seq, dim] -> [seq, heads, dim]
            attn_out = attn_out.squeeze(0).permute(1, 0, 2)
            output[start_q:end_q, :, :] = attn_out[prefill_seq_len_q:, :, :]
            start_q = end_q

        return output

    def _run_sage_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        scaling=None,
    ):
        """Run the decode forward by using SageAttention.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_kv_heads, head_size]
            v_cache: [max_total_num_tokens, num_kv_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            scaling: float or None

        Returns:
            output: [num_tokens, num_heads, head_size]
        """
        start_q = 0
        for seq_idx in range(seq_lens.shape[0]):
            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + 1

            per_req_query = query[start_q:end_q, :, :]

            # Gather KV from cache
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens]
            per_req_value = v_cache[per_req_tokens]

            # Align dtypes
            if not (
                per_req_query.dtype == per_req_key.dtype == per_req_value.dtype
            ):
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)

            # Reshape to HND: [1, heads, seq, dim]
            q_hnd = per_req_query.permute(1, 0, 2).unsqueeze(0)
            k_hnd = per_req_key.permute(1, 0, 2).unsqueeze(0)
            v_hnd = per_req_value.permute(1, 0, 2).unsqueeze(0)

            attn_out = sageattn(
                q_hnd,
                k_hnd,
                v_hnd,
                tensor_layout="HND",
                is_causal=False,
                sm_scale=scaling,
            )

            # attn_out: [1, heads, 1, dim] -> [1, heads, dim]
            attn_out = attn_out.squeeze(0).permute(1, 0, 2)
            output[start_q:end_q, :, :] = attn_out
            start_q = end_q

        return output

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if layer.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        causal = True
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            causal = False

        self._run_sage_forward_extend(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.extend_prefix_lens,
            forward_batch.extend_seq_lens,
            scaling=layer.scaling,
            causal=causal,
        )
        return o

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if layer.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        self._run_sage_forward_decode(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            scaling=layer.scaling,
        )

        return o

    def support_triton(self):
        return False
