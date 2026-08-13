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

"""XLA reference implementation of the ``ragged_gather`` op.

Row gather with a negative-index invalid convention: ``out[i] = x[indices[i]]``
for ``indices[i] >= 0`` and a zero row otherwise.  This is the MoE
permute/unpermute primitive (dispatch rows to expert order and back).

JAX's native autodiff for gather (transpose = scatter-add) is correct, so no
custom VJP is needed; invalid rows contribute zero cotangent through the
``where`` mask.
"""

from __future__ import annotations

import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ..._registry import Backend, Platform, kernel_registry


@kernel_registry.register("ragged_gather", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def ragged_gather(
    x: Float[Array, "n d"],
    indices: Int[Array, "m"],
) -> Float[Array, "m d"]:
    """Gather rows of ``x`` by ``indices``; negative indices yield zero rows.

    Args:
        x: Source table of shape ``[n, d]``.
        indices: Row indices of shape ``[m]``.  Entries ``< 0`` mark invalid
            (padded) slots and produce zero rows.

    Returns:
        Gathered rows ``[m, d]`` in ``x``'s dtype.
    """
    valid = indices >= 0
    safe = jnp.where(valid, indices, 0)
    rows = x[safe]
    return jnp.where(valid[:, None], rows, jnp.zeros((), dtype=x.dtype))


__all__ = ("ragged_gather",)
