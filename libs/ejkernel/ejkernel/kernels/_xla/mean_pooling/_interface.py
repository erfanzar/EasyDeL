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
"""Registry entry point for the XLA mean pooling kernel.

Registers ``mean_pooling`` under ``(Platform.XLA, Backend.ANY)`` so that the
kernel registry dispatches calls on any XLA-compatible device to the JAX
implementation in ``_xla_impl_fwd``.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import Array, Float, Int
from ._xla_impl_fwd import mean_pooling as _mean_pooling_impl


@kernel_registry.register("mean_pooling", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def mean_pooling(
    x: Float[Array, "... hidden_dim"],
    chunk_size: int = 32,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    *,
    block_dim: int = 128,
    block_size: int = 256,
    num_warps: int = 4,
    num_stages: int = 1,
) -> Float[Array, "... hidden_dim"]:
    """Perform mean pooling over the sequence dimension using JAX/XLA.

    Registry wrapper; delegates to ``_mean_pooling_core`` which is decorated
    with ``jax.custom_vjp`` for efficient gradient computation.

    Args:
        x: Input tensor.  Shape must be ``[batch, seq_len, hidden_dim]`` for
            fixed-length batches, or ``[total_tokens, hidden_dim]`` when
            ``cu_seqlens`` is provided.
        chunk_size: Accepted for API compatibility with the Triton backend;
            ignored on the XLA path.
        cu_seqlens: Optional cumulative sequence lengths of shape
            ``[num_seqs + 1]``.  For example, ``[0, 10, 25]`` describes two
            sequences of lengths 10 and 15.  When provided, ``x`` must be
            2-D (packed tokens).
        block_dim: Accepted for API compatibility with Triton; ignored by XLA.
        num_warps: Accepted for API compatibility with Triton; ignored by XLA.
        num_stages: Accepted for API compatibility with Triton; ignored by XLA.

    Returns:
        Mean-pooled tensor of shape ``[batch_or_num_seqs, hidden_dim]``.
        When ``cu_seqlens`` is used the output has ``len(cu_seqlens) - 1`` rows.

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
    return _mean_pooling_impl(x, chunk_size, cu_seqlens)


__all__ = ("mean_pooling",)
