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

"""Sinkhorn-Knopp doubly-stochastic projection interface (Pallas TPU)."""

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl_fwd import sinkhorn_knopp_tpu


@kernel_registry.register("sinkhorn_knopp", Platform.PALLAS, Backend.TPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def sinkhorn_knopp(
    matrix: Float[Array, "batch seq rows cols"],
    n_iters: int = 20,
    eps: float = 1e-6,
) -> Float[Array, "batch seq rows cols"]:
    """Fused Sinkhorn-Knopp projection (Pallas TPU).

    Registered under ``"sinkhorn_knopp"`` for ``Platform.PALLAS`` /
    ``Backend.TPU``. Numerically matches the XLA reference, which is the
    contract; see that impl for argument semantics.

    Args:
        matrix: Strictly positive matrices ``[batch, seq, rows, cols]``.
        n_iters: Sinkhorn-Knopp iterations (static).
        eps: Denominator floor.

    Returns:
        Normalised matrices, same shape and dtype.
    """
    return sinkhorn_knopp_tpu(matrix, n_iters=n_iters, eps=eps)
