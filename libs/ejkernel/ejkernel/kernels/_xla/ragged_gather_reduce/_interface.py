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

"""XLA reference implementation of the ``ragged_gather_reduce`` op.

The fused MoE top-k combine: for each output row ``t``, gather its
``reduce_group_size`` expert rows, weight them, and sum:

    ``out[t] = sum_j weights[t*k + j] * x[indices[t*k + j]]``

Negative indices mark invalid (padded) slots and contribute zero.  Weighting
and accumulation run in float32 regardless of the input dtype (results cast
back), matching the numerics of the sort-based MoE combine path.

JAX's native autodiff is correct here (gather transpose = scatter-add;
weight grads = per-slot row dots), so no custom VJP is needed.
"""

from __future__ import annotations

import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ..._registry import Backend, Platform, kernel_registry


@kernel_registry.register("ragged_gather_reduce", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def ragged_gather_reduce(
    x: Float[Array, "n d"],
    indices: Int[Array, "mk"],
    weights: Float[Array, "mk"],
    reduce_group_size: int,
) -> Float[Array, "m d"]:
    """Weighted grouped row gather-sum (MoE top-k combine).

    Args:
        x: Expert-output table of shape ``[n, d]``.
        indices: Flat row indices ``[m * reduce_group_size]``; entries ``< 0``
            are invalid slots contributing zero.
        weights: Per-slot combine weights ``[m * reduce_group_size]``.
        reduce_group_size: Number of consecutive slots summed per output row
            (the MoE ``top_k``).  Must divide ``indices.shape[0]``.

    Returns:
        Combined rows ``[m, d]`` in ``x``'s dtype, where
        ``m = indices.shape[0] // reduce_group_size``.

    Raises:
        ValueError: If ``reduce_group_size`` does not divide the slot count.
            (Mismatched ``indices``/``weights`` lengths are rejected by the
            shared ``mk`` shape annotation.)
    """
    slots = int(indices.shape[0])
    if reduce_group_size < 1 or slots % reduce_group_size != 0:
        raise ValueError(f"reduce_group_size ({reduce_group_size}) must divide the slot count ({slots}).")

    valid = indices >= 0
    safe = jnp.where(valid, indices, 0)
    rows = x[safe].astype(jnp.float32)
    w = jnp.where(valid, weights.astype(jnp.float32), 0.0)
    weighted = rows * w[:, None]
    grouped = weighted.reshape(slots // reduce_group_size, reduce_group_size, x.shape[-1])
    return jnp.sum(grouped, axis=1).astype(x.dtype)


__all__ = ("ragged_gather_reduce",)
