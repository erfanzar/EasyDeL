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


"""Mean pooling operations using Triton kernels.

This module provides GPU-accelerated mean pooling over sequence dimensions,
commonly used in NLP tasks to aggregate token embeddings into fixed-size
representations. The implementation uses custom Triton kernels for optimal
performance on GPUs.

Mean pooling computes the average of all token embeddings in a sequence,
producing a single vector representation. This is particularly useful for:
- Sentence/document embeddings in classification tasks
- Block compression in sparse attention mechanisms
- Sequence-level representations for downstream tasks

The implementation supports:
- Standard batched sequences with uniform lengths
- Variable-length sequences via cumulative sequence lengths (cu_seqlens)
- Efficient GPU parallelization via Triton
- Full automatic differentiation support

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._triton.mean_pooling import mean_pooling
    >>>
    >>>
    >>> batch, seq_len, hidden_dim = 4, 128, 768
    >>> x = jnp.ones((batch, seq_len, hidden_dim))
    >>>
    >>>
    >>> pooled = mean_pooling(x, chunk_size=32)
    >>> print(pooled.shape)
"""

from functools import partial

import jax
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ..._registry import Backend, Platform, kernel_registry
from ._triton_impl_bwd import bwd_triton_impl
from ._triton_impl_fwd import fwd_triton_impl


def _fwd_call(
    x: Float[Array, "... hidden_dim"],
    chunk_size: int,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    block_dim: int = 64,
    block_size: int = 256,
    num_warps: int = 4,
    num_stages: int = 1,
) -> tuple[Float[Array, "... hidden_dim"], tuple[tuple[int, ...], Int[Array, "num_seqs_plus_one"] | None]]:
    """
    Forward pass for mean pooling with custom VJP.

    Args:
        x: The input tensor.
        chunk_size: Semantic block-pooling window for 4-D inputs.
        cu_seqlens: Optional cumulative sequence lengths for variable-length sequences.
        block_dim: Hidden-dimension tile width.
        block_size: Sequence-axis tile size for 2-D/3-D global pooling.
        num_warps: Number of Triton warps per block.
        num_stages: Number of Triton software pipeline stages.

    Returns:
        A tuple containing the output of the forward pass and the residuals
        needed for the backward pass.
    """
    o = fwd_triton_impl(
        x=x,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        block_dim=block_dim,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    residual = x.shape, cu_seqlens
    return o, residual


def _bwd_call(
    chunk_size: int,
    block_dim: int,
    block_size: int,
    num_warps: int,
    num_stages: int,
    residual: tuple[tuple[int, ...], Int[Array, "num_seqs_plus_one"] | None],
    do: Float[Array, "... hidden_dim"],
) -> tuple[Float[Array, "... hidden_dim"], None]:
    """
    Backward pass for mean pooling with custom VJP.

    Args:
        chunk_size: Semantic block-pooling window used in the forward pass.
        block_dim: Hidden-dimension tile width.
        block_size: Sequence-axis tile size for 2-D/3-D global pooling.
        num_warps: Number of Triton warps per block.
        num_stages: Number of Triton software pipeline stages.
        residual: Residuals saved from the forward pass.
        do: The gradient of the output tensor.

    Returns:
        The gradient with respect to the input tensor `x` and `None` for `cu_seqlens`.
    """
    x_shape, cu_seqlens = residual
    dEo = bwd_triton_impl(
        do=do,
        batch_size=int(x_shape[0]),
        seq_len=int(x_shape[1]),
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        block_dim=block_dim,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return dEo, None


@partial(jax.custom_vjp, nondiff_argnums=(1, 3, 4, 5, 6))
def _mean_pooling(
    x: Float[Array, "... hidden_dim"],
    chunk_size: int,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    block_dim: int = 64,
    block_size: int = 256,
    num_warps: int = 4,
    num_stages: int = 1,
) -> Float[Array, "... hidden_dim"]:
    """
    Core JIT-compiled mean pooling function with a custom VJP.

    This is an internal function that directly calls the Triton implementation
    for the forward pass and is registered with JAX's custom differentiation
    system.

    Args:
        x: The input tensor.
        chunk_size: Semantic block-pooling window for 4-D inputs (static argument).
        cu_seqlens: Optional cumulative sequence lengths for variable-length sequences.
        block_dim: Hidden-dimension tile width (static argument).
        block_size: Sequence-axis tile size for 2-D/3-D global pooling (static argument).
        num_warps: Number of Triton warps per block (static argument).
        num_stages: Number of Triton software pipeline stages (static argument).

    Returns:
        The mean-pooled output tensor.
    """
    return fwd_triton_impl(
        x=x,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        block_dim=block_dim,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )


_mean_pooling.defvjp(_fwd_call, _bwd_call)


@kernel_registry.register("mean_pooling", Platform.TRITON, Backend.GPU)
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
    """
    Performs mean pooling over the sequence dimension using a Triton kernel.

    This function calculates the mean of token embeddings for each sequence in a
    batch (2-D varlen or 3-D fixed) or per block (4-D block pooling). It is
    optimized for GPUs using a custom Triton kernel and supports both standard
    and variable-length sequences.

    Args:
        x: The input tensor of shape `(batch_size, sequence_length, hidden_dim)`
            for fixed global pooling, `(total_tokens, hidden_dim)` with
            `cu_seqlens` for packed varlen global pooling, or
            `(batch_size, sequence_length, num_heads, hidden_dim)` for 4-D
            block pooling.
        chunk_size: Semantic block pooling window for 4-D inputs.
        cu_seqlens: An optional 1D tensor of cumulative sequence lengths for
            handling variable-length sequences in a packed format.
            Example: `[0, len_seq1, len_seq1+len_seq2, ...]`.
        block_dim: Hidden-dimension tile width for Triton kernels.
        block_size: Sequence-axis tile size for 2-D/3-D global pooling kernels.
        num_warps: Number of Triton warps per block.
        num_stages: Number of Triton pipeline stages.

    Returns:
        Mean-pooled tensor matching the input pooling mode.
    """
    return _mean_pooling(
        x=x,
        chunk_size=int(chunk_size),
        cu_seqlens=cu_seqlens,
        block_dim=int(block_dim),
        block_size=int(block_size),
        num_warps=num_warps,
        num_stages=num_stages,
    )
