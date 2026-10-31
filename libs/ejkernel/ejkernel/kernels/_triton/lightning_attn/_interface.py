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
"""Kernel public interface and registration wrappers."""

from __future__ import annotations

import jaxtyping
from beartype import beartype

from ..._registry import Backend, Platform, kernel_registry
from ._triton_impl_fwd import Array, Float, Int
from ._triton_impl_fwd import lightning_attn as _lightning_attn_impl


@kernel_registry.register("lightning_attn", Platform.TRITON, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def lightning_attn(
    query: Float[Array, "batch seq_len num_heads qk_head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads qk_head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads v_head_dim"],
    layer_idx: int,
    num_layers: int,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    block_k: int = 64,
    block_v: int = 64,
    num_warps: int = 4,
    num_stages: int = 1,
) -> tuple[Float[Array, "batch seq_len num_heads v_head_dim"], Float[Array, "... num_heads qk_head_dim v_head_dim"]]:
    """
    Computes Lightning Attention using a recurrent, linear-time mechanism.
    This function implements the Lightning Attention mechanism, a form of linear
    attention where the decay rate (`g_gamma`) is dynamically determined by the
    layer's depth within the model. This allows for different temporal receptive
    fields across layers.
    The computation is performed efficiently using a recurrent formulation,
    making it suitable for long sequences. It serves as a specialized wrapper
    around the generic `recurrent` function and supports both standard batch
    processing and packed variable-length inputs via `cu_seqlens`.
    Args:
        query: The query tensor. Expected shape is `(batch, seq_len, num_heads, head_dim)`
            or `(1, total_tokens, num_heads, head_dim)` if `cu_seqlens` is used.
        key: The key tensor. Must have the same shape as `q`.
        value: The value tensor. Must have the same shape as `q`.
        layer_idx: The 0-indexed index of the current layer, used to compute
            the layer-specific decay factor.
        num_layers: The total number of layers in the model.
        softmax_scale: A scaling factor applied to the query. If `None`, it defaults
            to `1 / sqrt(head_dim)`.
        initial_state: The initial hidden state for the recurrence. Useful for
            chunked processing of long sequences.
        reverse: If `True`, the sequence is processed in reverse order.
        cu_seqlens: Cumulative sequence lengths for variable-length inputs.
            This is a 1D tensor like `[0, len_seq1, len_seq1+len_seq2, ...]`.
            If provided, the input tensors are expected to be "packed" with a
            batch size of 1.
    Returns:
        A tuple containing:
            - o (jax.Array): The output tensor, with the same shape as `q`.
            - final_state (jax.Array): The final hidden state of the recurrence.
    Raises:
        ValueError: If `cu_seqlens` is provided and the batch size of `q` is
            not 1.
        ValueError: If `cu_seqlens` is provided and the number of initial states
            does not match the number of sequences.
    """
    return _lightning_attn_impl(
        query,
        key,
        value,
        layer_idx,
        num_layers,
        softmax_scale,
        initial_state,
        reverse,
        cu_seqlens,
        block_k,
        block_v,
        num_warps,
        num_stages,
    )


__all__ = ("lightning_attn",)
