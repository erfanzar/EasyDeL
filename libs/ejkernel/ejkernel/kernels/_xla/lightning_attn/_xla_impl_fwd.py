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

"""XLA/JAX implementation of Lightning Attention with layer-dependent decay.

This module is the concrete forward implementation for the Lightning Attention
mechanism. It computes a per-layer decay vector ``g_gamma`` whose values depend
on both the head index and the layer depth, then delegates all computation to
the generic ``recurrent`` kernel.

Decay formula (per head ``h``)::

    g_gamma[h] = -(8 / num_heads) * (1 - layer_idx / num_layers) * h

This means:
- Shallower layers (small ``layer_idx``) receive larger decay magnitudes,
  giving them shorter effective memory.
- Deeper layers approach zero decay (near-undecayed recurrence).
- Earlier heads within a layer have smaller decay magnitudes than later heads.

The actual recurrence, online softmax, and ``cu_seqlens`` unpacking logic all
live in the ``recurrent`` module; this module only constructs ``g_gamma``.
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
) -> tuple[Float[Array, "batch seq_len num_heads v_head_dim"], Float[Array, "... num_heads qk_head_dim v_head_dim"]]:
    """
    Computes Lightning Attention using a recurrent, linear-time mechanism with JAX/XLA.

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

    Examples:
        >>> import jax.numpy as jnp
        >>> q = jnp.ones((2, 100, 8, 64))
        >>> k = jnp.ones((2, 100, 8, 64))
        >>> v = jnp.ones((2, 100, 8, 64))
        >>> output, final_state = lightning_attn(q, k, v, layer_idx=5, num_layers=24)
        >>> output.shape
        (2, 100, 8, 64)

        Packed variable-length sequences (batch_size must be 1):

        >>> q = jnp.ones((1, 150, 8, 64))
        >>> k = jnp.ones((1, 150, 8, 64))
        >>> v = jnp.ones((1, 150, 8, 64))
        >>> cu_seqlens = jnp.array([0, 50, 100, 150])
        >>> output, final_state = lightning_attn(
        ...     q, k, v, layer_idx=10, num_layers=24, cu_seqlens=cu_seqlens
        ... )
    """
    if cu_seqlens is not None:
        if query.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {query.shape[0]} when using `cu_seqlens`. "
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )

    if softmax_scale is None:
        softmax_scale = key.shape[-1] ** -0.5

    num_heads = query.shape[2] if query.ndim == 4 else query.shape[1]
    g_gamma = -(8 / num_heads * (1 - layer_idx / num_layers)) * jnp.arange(num_heads, dtype=jnp.float32)

    return recurrent(
        query=query,
        key=key,
        value=value,
        g_gamma=g_gamma,
        softmax_scale=softmax_scale,
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
    )
