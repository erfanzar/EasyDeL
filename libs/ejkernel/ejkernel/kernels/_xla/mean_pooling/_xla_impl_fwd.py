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

"""XLA implementation of mean pooling with custom VJP.

This module provides the forward implementation of mean pooling over
sequence dimensions.  Two paths are supported:

- **Fixed-length** (``cu_seqlens=None``): ``jnp.mean(x, axis=1)`` over
  a ``[batch, seq_len, hidden_dim]`` tensor.
- **Variable-length / packed** (``cu_seqlens`` provided): segment-sum / seq_len
  over a 2-D ``[total_tokens, hidden_dim]`` tensor.

A ``jax.custom_vjp`` wrapper (``_mean_pooling_core``) broadcasts the
upstream gradient across the tokens of each packed sequence.

The raw forward functions are injected into ``_xla_impl_bwd`` after import
so the custom-VJP forward rule avoids calling the decorated function again.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from . import _xla_impl_bwd as _mean_pooling_bwd_mod
from ._xla_impl_bwd import _mean_pooling_bwd


def _mean_pooling_varlen(
    x: Float[Array, "total_tokens hidden_dim"],
    cu_seqlens: Int[Array, "num_seqs_plus_one"],
) -> Float[Array, "num_seqs hidden_dim"]:
    """Mean pooling for variable-length (packed) sequences.

    Computes the mean of token embeddings for each variable-length sequence
    in a packed tensor.  Sequences are identified by their cumulative lengths,
    avoiding the need for padding.

    Tokens are assigned to segments by cumulative boundaries and accumulated
    into float32 sums. The number of sequences and total token capacity are
    static shapes, while individual sequence lengths may be traced values.
    Empty segments return zeros.

    Args:
        x: Packed token embeddings ``[total_tokens, hidden_dim]``.
        cu_seqlens: Cumulative sequence lengths ``[num_seqs + 1]``.
            Example: ``[0, 10, 25]`` → two sequences of lengths 10 and 15.

    Returns:
        Mean-pooled embeddings ``[num_seqs, hidden_dim]``, one row per
        sequence.
    """
    num_seqs = cu_seqlens.shape[0] - 1
    if num_seqs == 0:
        return jnp.zeros((0, x.shape[-1]), dtype=x.dtype)
    positions = jnp.arange(x.shape[0], dtype=cu_seqlens.dtype)
    seq_ids = jnp.searchsorted(cu_seqlens[1:], positions, side="right")
    valid = (positions >= cu_seqlens[0]) & (positions < cu_seqlens[-1])
    # Accumulate in at least float32 (low-precision inputs) without narrowing float64 inputs.
    acc_dtype = jnp.promote_types(x.dtype, jnp.float32)
    sums = jnp.zeros((num_seqs, x.shape[-1]), dtype=acc_dtype)
    sums = sums.at[jnp.minimum(seq_ids, num_seqs - 1)].add(jnp.where(valid[:, None], x.astype(acc_dtype), 0))
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    return (sums / jnp.maximum(lengths[:, None], 1)).astype(x.dtype)


def _mean_pooling_fixed(
    x: Float[Array, "batch seq_len hidden_dim"],
) -> Float[Array, "batch hidden_dim"]:
    """Mean pooling for fixed-length (padded) sequences.

    Computes the mean of token embeddings along the sequence dimension
    for a standard batched tensor where all sequences have the same length.

    Args:
        x: Input tensor of shape [batch, seq_len, hidden_dim] where all
            sequences are padded to the same length.

    Returns:
        Mean-pooled tensor of shape [batch, hidden_dim] where each row
        is the average across all seq_len tokens for that batch element.
    """
    return jnp.mean(x, axis=1)


@jax.custom_vjp
def _mean_pooling_core(
    x: Float[Array, "... hidden_dim"],
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
) -> Float[Array, "batch hidden_dim"]:
    """Core mean pooling implementation with custom VJP.

    Dispatches to either variable-length or fixed-length mean pooling
    based on whether cumulative sequence lengths are provided. This
    function is wrapped with ``jax.custom_vjp`` to enable efficient
    gradient computation that avoids recomputing the forward pass.

    Args:
        x: Input tensor. Either [total_tokens, hidden_dim] for packed
            sequences or [batch, seq_len, hidden_dim] for fixed-length.
        cu_seqlens: Optional cumulative sequence lengths [num_seqs + 1].
            If provided, uses variable-length pooling.

    Returns:
        Mean-pooled tensor of shape [batch_or_num_seqs, hidden_dim].
    """
    if cu_seqlens is not None:
        return _mean_pooling_varlen(x, cu_seqlens)
    else:
        return _mean_pooling_fixed(x)


_mean_pooling_bwd_mod._mean_pooling_varlen = _mean_pooling_varlen
_mean_pooling_bwd_mod._mean_pooling_fixed = _mean_pooling_fixed
_mean_pooling_core.defvjp(_mean_pooling_bwd_mod._mean_pooling_fwd, _mean_pooling_bwd)


def mean_pooling(
    x: Float[Array, "... hidden_dim"],
    chunk_size: int = 32,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
) -> Float[Array, "batch hidden_dim"]:
    """Perform mean pooling over the sequence dimension using JAX/XLA.

    Computes the mean of token embeddings.  Dispatches to the fixed-length or
    variable-length path depending on whether ``cu_seqlens`` is provided.
    Gradients are computed via the custom VJP defined in ``_mean_pooling_core``.

    Args:
        x: Input tensor.  Shape ``[batch, seq_len, hidden_dim]`` for
            fixed-length batches, or ``[total_tokens, hidden_dim]`` when
            ``cu_seqlens`` is provided.
        chunk_size: Accepted for API compatibility with the Triton backend;
            ignored on this XLA path.
        cu_seqlens: Optional cumulative sequence lengths ``[num_seqs + 1]``.
            When provided, ``x`` must be 2-D (packed tokens).

    Returns:
        Mean-pooled tensor ``[batch_or_num_seqs, hidden_dim]``.  The leading
        dimension equals ``len(cu_seqlens) - 1`` when ``cu_seqlens`` is used.

    Examples:
        >>> import jax.numpy as jnp
        >>> x = jnp.ones((2, 10, 128))
        >>> out = mean_pooling(x)
        >>> out.shape
        (2, 128)

        >>> x = jnp.ones((25, 128))
        >>> cu_seqlens = jnp.array([0, 10, 25])
        >>> out = mean_pooling(x, cu_seqlens=cu_seqlens)
        >>> out.shape
        (2, 128)
    """

    return _mean_pooling_core(x, cu_seqlens)
