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

"""Public interface and custom VJP for the TPU one-shot all-gather.

Wraps the low-level Pallas kernel (``_pallas_impl.all_gather``) with a
``jax.custom_vjp`` rule whose backward pass is the transpose dual: a
reduce-scatter of the output cotangent (using the sibling one-shot
reduce-scatter kernel).  Registered under ``Platform.PALLAS / Backend.TPU``.
"""

from __future__ import annotations

from functools import partial

import jax
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl import all_gather as _all_gather_impl


@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4, 5, 6))
def _all_gather_core(
    x: Float[Array, "..."],
    axis_name: str | tuple[str, ...],
    gather_axis: int,
    tiled: bool,
    mode: str,
    tp_size: int | None,
    collective_id: int | None,
) -> Float[Array, "..."]:
    """Forward-only dispatch to the Pallas one-shot all-gather."""
    return _all_gather_impl(
        x,
        axis_name=axis_name,
        gather_axis=gather_axis,
        tiled=tiled,
        mode=mode,
        tp_size=tp_size,
        collective_id=collective_id,
    )


def _all_gather_core_fwd(
    x: Float[Array, "..."],
    axis_name: str | tuple[str, ...],
    gather_axis: int,
    tiled: bool,
    mode: str,
    tp_size: int | None,
    collective_id: int | None,
):
    """Custom VJP forward pass: compute output; no residuals needed."""
    out = _all_gather_impl(
        x,
        axis_name=axis_name,
        gather_axis=gather_axis,
        tiled=tiled,
        mode=mode,
        tp_size=tp_size,
        collective_id=collective_id,
    )
    return out, None


def _all_gather_core_bwd(
    axis_name: str | tuple[str, ...],
    gather_axis: int,
    tiled: bool,
    mode: str,
    tp_size: int | None,
    collective_id: int | None,
    residual,
    dy: Float[Array, "..."],
):
    """Custom VJP backward pass: reduce-scatter the output cotangent.

    The transpose of a (tiled) all-gather is a reduce-scatter along the same
    axis, implemented with the sibling one-shot kernel so forward and backward
    share the same latency profile.
    """
    del residual
    from ..reduce_scatter._pallas_impl import reduce_scatter as _reduce_scatter_impl

    grad_x = _reduce_scatter_impl(
        dy,
        axis_name=axis_name,
        scatter_axis=gather_axis,
        tiled=tiled,
        mode=mode,
        tp_size=tp_size,
        collective_id=collective_id,
    )
    return (grad_x,)


_all_gather_core.defvjp(_all_gather_core_fwd, _all_gather_core_bwd)


@kernel_registry.register("all_gather", Platform.PALLAS, Backend.TPU)
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
    """All-gather with a one-shot direct-exchange Pallas kernel for small shards.

    Must be called inside ``jax.experimental.shard_map`` (or equivalent) so
    that ``axis_name`` is active.  Small, aligned shards on ``gather_axis=0``
    use a single-phase direct exchange writing each shard straight into every
    peer's output window (one logical hop instead of ``tp - 1`` ring hops);
    other requests delegate to ``lax.all_gather``.

    Args:
        x: Local shard.
        axis_name: pmap / shard_map axis name for the collective.
        gather_axis: Concatenation axis.  The Pallas kernel supports axis 0.
        tiled: Tiled gather semantics (concatenate along ``gather_axis``).
        mode: ``"one_shot"`` (force the kernel; raises when unsupported),
            ``"ring"`` (delegate to ``lax.all_gather``), or ``"auto"``
            (resolves to the ``lax.all_gather`` path — measured faster on v5p-8).
        tp_size: Collective world size; inferred when ``None``.
        collective_id: Integer ID for TPU barrier-semaphore allocation.

    Returns:
        Gathered array with ``gather_axis`` multiplied by the world size
        (or a new leading axis when ``tiled=False``).

    Raises:
        ValueError: If ``mode`` is invalid, ``tp_size < 1``, or
            ``mode="one_shot"`` constraints are violated.
    """
    return _all_gather_core(x, axis_name, gather_axis, tiled, mode, tp_size, collective_id)


__all__ = ("all_gather",)
