# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Registry interface for TPU Pallas Dual-Stream Ragged Paged Attention."""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int32

from ejkernel.kernels._xla.ragged_page_attention_v3._sinks import reduce_sinks_to_heads

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl_fwd import dual_stream_ragged_paged_attention_pallas


@kernel_registry.register("dual_stream_ragged_page_attention", Platform.PALLAS, Backend.TPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def dual_stream_ragged_page_attention(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    local_keys: Float[Array, "total_tokens num_kv_heads head_dim"],
    local_values: Float[Array, "total_tokens num_kv_heads head_dim"],
    global_kv_cache: Float[
        Array, "num_global_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"
    ],
    local_kv_cache: Float[
        Array, "num_local_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"
    ],
    global_kv_lens: Int32[Array, "max_num_seqs"],
    global_block_tables: Int32[Array, "max_num_seqs_times_global_pages"],
    local_kv_lens: Int32[Array, "max_num_seqs"],
    local_block_tables: Int32[Array, "max_num_seqs_times_local_pages"],
    query_start_loc: Int32[Array, "max_num_seqs_plus_1"],
    distribution: Int32[Array, "3"],
    softmax_aux: Float[Array, "..."] | None = None,
    *,
    softmax_scale: float | None = None,
    sliding_window: int = 128,
    overlap: bool = True,
    logits_soft_cap: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
) -> tuple[
    Float[Array, "total_tokens num_q_heads head_dim"],
    Float[Array, "num_local_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"],
]:
    """Dual-stream ragged paged attention (TPU Pallas fused single-launch kernel)."""
    if softmax_scale is None:
        softmax_scale = float(queries.shape[-1]) ** -0.5
    softmax_aux = reduce_sinks_to_heads(softmax_aux, queries.shape[1], local_keys.shape[1])
    return dual_stream_ragged_paged_attention_pallas(
        queries,
        local_keys,
        local_values,
        global_kv_cache,
        local_kv_cache,
        global_kv_lens,
        global_block_tables,
        local_kv_lens,
        local_block_tables,
        query_start_loc,
        distribution,
        softmax_aux,
        softmax_scale=float(softmax_scale),
        sliding_window=int(sliding_window),
        overlap=bool(overlap),
        logits_soft_cap=logits_soft_cap,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        chunk_prefill_size=chunk_prefill_size,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
    )
