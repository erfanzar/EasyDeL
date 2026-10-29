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

"""tile-lang mean_pooling public interface.

Registers ``mean_pooling`` for ``Platform.TILELANG / Backend.GPU`` in the
kernel registry.  Input validation (rank, dtype, cu_seqlens constraints) is
performed here before delegating to ``mean_pooling_tilelang`` in ``_impl``.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ejkernel.errors import EjkernelRuntimeError

from ..._registry import Backend, Platform, kernel_registry
from ._impl import mean_pooling_tilelang


@kernel_registry.register("mean_pooling", Platform.TILELANG, Backend.GPU)
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
    """Mean-pool token embeddings over the sequence axis (TileLang GPU backend).

    Supports two calling modes:

    **Padded mode** (``cu_seqlens=None``):
        *x* must be rank-3 with shape ``(batch, seq_len, hidden_dim)``.
        Returns ``(batch, hidden_dim)``.

    **Packed mode** (``cu_seqlens`` provided):
        *x* must be rank-2 with shape ``(total_tokens, hidden_dim)`` and
        contain tokens from all sequences concatenated without padding.
        *cu_seqlens* must be a rank-1 int32 array of shape
        ``(num_seqs + 1,)`` where ``cu_seqlens[i]`` is the cumulative token
        offset of sequence ``i``.
        Returns ``(num_seqs, hidden_dim)``.

    Both modes support forward and backward (``jax.grad`` / ``jax.vjp``).

    Args:
        x: Input activations — ``(batch, seq_len, hidden_dim)`` (padded) or
            ``(total_tokens, hidden_dim)`` (packed).  Supported dtypes:
            float16, bfloat16, float32.
        chunk_size: Accepted for API compatibility; currently ignored by the
            TileLang backend (tile sizes are chosen automatically).
        cu_seqlens: Cumulative sequence-length array of shape
            ``(num_seqs + 1,)`` and dtype int32.  Pass ``None`` for the
            padded variant.
        block_dim: Hidden-axis tile size (``BLOCK_D``). The operation
            layer (``MeanPooling`` op via ``MeanPoolingConfig.block_dim``)
            chooses this from shape; the constant default here is the
            cold-start fallback for direct kernel-layer callers.
        block_size: Sequence-axis tile size (``BLOCK_S``). The operation
            layer (``MeanPoolingConfig.block_size``) chooses this; same
            caller policy as ``block_dim``.
        num_warps: Accepted for API compatibility with Triton; ignored by TileLang.
        num_stages: Accepted for API compatibility with Triton; ignored by TileLang.

    Returns:
        ``(batch, hidden_dim)`` or ``(num_seqs, hidden_dim)`` array in the
        same dtype as *x*.

    Raises:
        EjkernelRuntimeError: If input rank/dtype constraints are violated.
    """
    _ = chunk_size, num_warps, num_stages
    if cu_seqlens is not None:
        if x.ndim != 2:
            raise EjkernelRuntimeError("tile-lang mean_pooling with cu_seqlens requires packed (T, D) input.")
        if cu_seqlens.dtype.name != "int32":
            raise EjkernelRuntimeError("tile-lang mean_pooling requires int32 cu_seqlens.")
        return mean_pooling_tilelang(x, cu_seqlens, block_s=block_size, block_d=block_dim)
    if x.ndim != 3:
        raise EjkernelRuntimeError("tile-lang mean_pooling without cu_seqlens requires (B, S, D) input.")
    return mean_pooling_tilelang(x, None, block_s=block_size, block_d=block_dim)


__all__ = ["mean_pooling"]
