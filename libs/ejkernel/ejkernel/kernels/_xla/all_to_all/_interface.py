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

"""XLA reference implementation of the dense ``all_to_all`` op.

A thin wrapper over ``lax.all_to_all`` registered under
``Platform.XLA / Backend.ANY``.  JAX's native autodiff (the transpose of an
all-to-all is the inverse all-to-all with split/concat axes swapped) is
correct, so no custom VJP is needed.

Typical uses: sequence-parallel head exchange around attention (swap the
sharded axis between sequence and heads) and dense MoE token exchange.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jax import lax
from jaxtyping import Array, Float

from ..._registry import Backend, Platform, kernel_registry


@kernel_registry.register("all_to_all", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def all_to_all(
    x: Float[Array, "..."],
    axis_name: str | tuple[str, ...],
    split_axis: int,
    concat_axis: int,
    tiled: bool = True,
    tp_size: int | None = None,
    collective_id: int | None = 0,
) -> Float[Array, "..."]:
    """Exchange shards across ``axis_name`` via ``lax.all_to_all``.

    Splits the local array along ``split_axis``, exchanges one piece with
    every peer, and concatenates the received pieces along ``concat_axis``.

    Must be called inside ``jax.experimental.shard_map`` (or equivalent) so
    that ``axis_name`` is active.  ``collective_id`` is accepted for API
    compatibility with TPU backends and ignored here.

    Args:
        x: Local shard.
        axis_name: pmap / shard_map axis name (or tuple of names) for the
            collective.
        split_axis: Axis to split into per-peer pieces.
        concat_axis: Axis to concatenate received pieces along.
        tiled: Tiled semantics (concatenate; no new leading axis).
        tp_size: Optional validation-only world size (``>= 1`` if given).
        collective_id: Ignored (TPU barrier-semaphore ID).

    Returns:
        The exchanged array; ``split_axis`` is divided and ``concat_axis``
        multiplied by the world size (under ``tiled=True``).

    Raises:
        ValueError: If ``tp_size < 1``.
    """
    if tp_size is not None and tp_size < 1:
        raise ValueError(f"tp_size must be >= 1, got {tp_size}.")
    return lax.all_to_all(x, axis_name, split_axis=split_axis, concat_axis=concat_axis, tiled=tiled)


__all__ = ("all_to_all",)
