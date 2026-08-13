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

"""XLA reference implementation of the ``all_gather`` op.

A thin wrapper over ``lax.all_gather`` registered under
``Platform.XLA / Backend.ANY``.  JAX's native autodiff rules for
``all_gather`` already implement the reduce-scatter transpose, so no custom
VJP is needed.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jax import lax
from jaxtyping import Array, Float

from ..._registry import Backend, Platform, kernel_registry


@kernel_registry.register("all_gather", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def all_gather(
    x: Float[Array, "..."],
    axis_name: str | tuple[str, ...],
    gather_axis: int = 0,
    tiled: bool = True,
    mode: str = "auto",
    tp_size: int | None = None,
    collective_id: int | None = 0,
) -> Float[Array, "..."]:
    """Gather shards of ``x`` across ``axis_name`` via ``lax.all_gather``.

    Must be called inside ``jax.experimental.shard_map`` (or equivalent) so
    that ``axis_name`` is active.  ``mode`` and ``collective_id`` are accepted
    for API compatibility with the Pallas TPU backend and ignored here.

    Args:
        x: Local shard.
        axis_name: pmap / shard_map axis name for the collective.
        gather_axis: Concatenation axis.
        tiled: Tiled gather semantics (concatenate; no new leading axis).
        mode: Ignored (Pallas algorithm hint).
        tp_size: Optional validation-only world size (``>= 1`` if given).
        collective_id: Ignored (TPU barrier-semaphore ID).

    Returns:
        Gathered array with ``gather_axis`` multiplied by the world size
        (or a new leading axis when ``tiled=False``).

    Raises:
        ValueError: If ``tp_size < 1``.
    """
    if tp_size is not None and tp_size < 1:
        raise ValueError(f"tp_size must be >= 1, got {tp_size}.")
    return lax.all_gather(x, axis_name=axis_name, axis=gather_axis, tiled=tiled)


__all__ = ("all_gather",)
