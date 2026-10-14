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

"""Public XLA entry point for block-sparse selected (top-k mask) attention with a VJP."""

from __future__ import annotations

from jaxtyping import Array, Bool, Float

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl import sparse_selected_attention_impl


@kernel_registry.register("sparse_selected_attention", Platform.XLA, Backend.ANY)
def sparse_selected_attention(
    query: Float[Array, "batch seq_len num_heads qk_dim"],
    key: Float[Array, "batch seq_len num_kv_heads qk_dim"],
    value: Float[Array, "batch seq_len num_kv_heads v_dim"],
    mask: Bool[Array, "batch seq_len seq_len"],
    softmax_scale: float | None = None,
    causal: bool = True,
    block_q: int = 512,
    block_k: int = 512,
) -> Float[Array, "batch seq_len num_heads v_dim"]:
    """Attention restricted to a boolean selection mask, tiled and skipping empty tiles.

    Query ``i`` attends to key ``j`` iff ``mask[b, i, j]`` (and ``j <= i`` when ``causal``).
    A query with no allowed key outputs zeros. Tiles ``(block_q, block_k)`` with no allowed
    pair in any batch row are skipped in forward and backward.

    Args:
        query: ``[B, S, H, Dk]``.
        key: ``[B, S, H_kv, Dk]`` (``H`` a multiple of ``H_kv``).
        value: ``[B, S, H_kv, Dv]``.
        mask: ``[B, S, S]`` boolean selection (padding / segments / top-k).
        softmax_scale: Logit scale; defaults to ``Dk ** -0.5``.
        causal: Also apply ``j <= i``.
        block_q: Query tile size (static).
        block_k: Key tile size (static).

    Returns:
        ``[B, S, H, Dv]`` in the query dtype.
    """
    if softmax_scale is None:
        softmax_scale = float(query.shape[-1]) ** -0.5
    return sparse_selected_attention_impl(
        query, key, value, mask, float(softmax_scale), bool(causal), int(block_q), int(block_k)
    )
