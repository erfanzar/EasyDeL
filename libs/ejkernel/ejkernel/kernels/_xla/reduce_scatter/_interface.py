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

"""XLA reference implementation of the ``reduce_scatter`` op.

A thin wrapper over ``lax.psum_scatter`` registered under
``Platform.XLA / Backend.ANY``.  JAX's native autodiff rules for
``psum_scatter`` already implement the all-gather transpose, so no custom
VJP is needed.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jax import lax
from jaxtyping import Array, Float

from ..._registry import Backend, Platform, kernel_registry


@kernel_registry.register("reduce_scatter", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def reduce_scatter(
    x: Float[Array, "..."],
    axis_name: str | tuple[str, ...],
    scatter_axis: int = 0,
    tiled: bool = True,
    mode: str = "auto",
    tp_size: int | None = None,
    collective_id: int | None = 0,
) -> Float[Array, "..."]:
    """Reduce ``x`` over ``axis_name`` and scatter shards via ``lax.psum_scatter``.

    Must be called inside ``jax.experimental.shard_map`` (or equivalent) so
    that ``axis_name`` is active.  ``mode`` and ``collective_id`` are accepted
    for API compatibility with the Pallas TPU backend and ignored here.

    Args:
        x: Local full-shaped partial values.
        axis_name: pmap / shard_map axis name for the collective.
        scatter_axis: Axis the reduced result is scattered along.
        tiled: Tiled scatter semantics (no new leading axis).
        mode: Ignored (Pallas algorithm hint).
        tp_size: Optional validation-only world size (``>= 1`` if given).
        collective_id: Ignored (TPU barrier-semaphore ID).

    Returns:
        The reduced shard; ``scatter_axis`` is divided by the world size.

    Raises:
        ValueError: If ``tp_size < 1``.
    """
    if tp_size is not None and tp_size < 1:
        raise ValueError(f"tp_size must be >= 1, got {tp_size}.")
    return lax.psum_scatter(x, axis_name, scatter_dimension=scatter_axis, tiled=tiled)


__all__ = ("reduce_scatter",)
