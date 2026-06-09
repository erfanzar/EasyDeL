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

"""Gated Linear Attention (GLA) core implementation for XLA backend.

This module wraps the shared ``recurrent`` primitive with GLA-specific
validation logic and default argument handling.  The actual recurrent scan
is implemented in ``ejkernel.kernels._xla.recurrent``.

The ``recurrent_gla`` function exported here is the direct implementation;
the ``_interface.py`` module re-exports it after applying the beartype/jaxtyped
runtime checks and registering it in the kernel registry.
"""

from jaxtyping import Array, Float, Int

from ..recurrent import recurrent


def recurrent_gla(
    query: Float[Array, "batch seq_len num_heads qk_head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads qk_head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads v_head_dim"],
    g: Float[Array, "batch seq_len num_heads qk_head_dim"] | None = None,
    g_gamma: Float[Array, "... num_heads"] | None = None,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
) -> tuple[Float[Array, "batch seq_len num_heads v_head_dim"], Float[Array, "... num_heads qk_head_dim v_head_dim"]]:
    """Compute Gated Linear Attention (GLA) via a recurrent, linear-time scan.

    Wraps the core ``recurrent`` primitive with GLA-specific validation and
    default argument handling.  Processes sequences step-by-step, making it
    O(N) in time and memory for the recurrent state.

    Both standard batch processing and variable-length (packed) sequences via
    ``cu_seqlens`` are supported.

    Args:
        query: Query tensor.
            Shape: ``[batch, seq_len, num_heads, qk_head_dim]`` (standard), or
            packed ``[total_tokens, num_heads, qk_head_dim]`` when using
            ``cu_seqlens``.
        key: Key tensor, same shape as ``query``.
        value: Value tensor.
            Shape: ``[batch, seq_len, num_kv_heads, v_head_dim]``.
        g: Gate tensor for GLA.  Same shape as ``query``.
            When provided, element-wise gates the key-value outer product.
        g_gamma: Optional per-head scalar decay factor broadcast over the
            sequence.  Shape: ``[..., num_heads]``.
        softmax_scale: Scaling factor applied to queries before the recurrence.
            Defaults to ``key.shape[-1] ** -0.5``.
        initial_state: Initial hidden state for incremental inference.
            Shape: ``[..., num_heads, qk_head_dim, v_head_dim]``.
        reverse: If True, process the sequence right-to-left.
        cu_seqlens: Cumulative sequence lengths for variable-length packed
            inputs.  Format: ``[0, len_seq1, len_seq1+len_seq2, ...]``.
            When provided, input tensors must be packed and batch size must
            be 1.

    Returns:
        Tuple of:
            - output: Attention output, same shape as ``query``.
            - final_state: Final recurrent hidden state with shape
              ``[..., num_heads, qk_head_dim, v_head_dim]``.

    Raises:
        ValueError: If ``cu_seqlens`` is provided and the batch size is not 1.
        ValueError: If ``cu_seqlens`` is provided and the number of initial
            states does not match the number of sequences.

    Example:
        >>> import jax.numpy as jnp
        >>> q = jnp.ones((2, 100, 8, 64))
        >>> k = jnp.ones((2, 100, 8, 64))
        >>> v = jnp.ones((2, 100, 8, 64))
        >>> g = jnp.ones((2, 100, 8, 64))
        >>> output, final_state = recurrent_gla(q, k, v, g=g)
        >>> output.shape
        (2, 100, 8, 64)
    """
    if cu_seqlens is not None:
        if query.shape[0] != 1 and query.ndim == 4:
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

    o, final_state = recurrent(
        query=query,
        key=key,
        value=value,
        g=g,
        g_gamma=g_gamma,
        softmax_scale=softmax_scale,
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
    )
    return o, final_state
