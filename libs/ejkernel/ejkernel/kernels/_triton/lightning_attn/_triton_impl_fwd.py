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


"""Lightning Attention implementation using Triton kernels.

This module provides an efficient implementation of Lightning Attention, a
linear attention mechanism that uses layer-dependent decay rates to adaptively
adjust the temporal receptive field across different layers of a neural network.

Lightning Attention is designed for efficient sequence processing with O(N)
complexity while allowing different layers to have different memory characteristics.
Shallow layers can focus on local patterns with faster decay, while deeper layers
can capture longer-range dependencies with slower decay.

Key innovation:
The decay rate (g_gamma) is computed based on layer position:
    g_gamma = -(8 / num_heads) * (1 - layer_idx / num_layers) * head_indices

This creates a progressive increase in temporal context as we move deeper into
the network, mimicking the hierarchical feature learning in transformers but
with linear complexity.

Features:
- O(N) time complexity via recurrent formulation
- Layer-adaptive decay rates for hierarchical learning
- Support for variable-length sequences
- GPU-optimized Triton kernels
- Full gradient support via JAX autodiff

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._triton.lightning_attn import lightning_attn
    >>>
    >>> batch, seq_len, num_heads, head_dim = 2, 2048, 8, 64
    >>> q = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> k = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> v = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>>
    >>>
    >>> output, final_state = lightning_attn(q, k, v, layer_idx=5, num_layers=12)
"""

from jax import numpy as jnp
from jaxtyping import Array, Float, Int

from ..recurrent import recurrent


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
    if cu_seqlens is not None:
        if query.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {query.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if softmax_scale is None:
        softmax_scale = key.shape[-1] ** -0.5
    qheads = query.shape[2]
    g_gamma = -(8 / qheads * (1 - layer_idx / num_layers)) * jnp.arange(qheads, dtype="f4")
    return recurrent(
        query=query,
        key=key,
        value=value,
        g_gamma=g_gamma,
        softmax_scale=softmax_scale,
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
        block_k=block_k,
        block_v=block_v,
        num_warps=num_warps,
        num_stages=num_stages,
    )
