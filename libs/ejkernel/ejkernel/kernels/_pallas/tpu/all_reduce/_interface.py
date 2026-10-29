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

"""Public interface and custom VJP for the TPU one-shot all-reduce.

Wraps the low-level Pallas kernel (``_pallas_impl.all_reduce``) with a
``jax.custom_vjp`` rule and registers it under
``Platform.PALLAS / Backend.TPU``.

Backward pass:
    The transpose of a psum is a psum (JAX's own rule for ``lax.psum``):
    under ``shard_map`` with ``check_vma=False`` the replicated output's
    cotangent arrives as ``dy / tp`` on each device, and summing it across
    devices reconstructs ``dy`` exactly.  The backward pass therefore calls
    the all-reduce impl again (the op is its own transpose dual), keeping
    gradient semantics identical to ``lax.psum`` autodiff.
"""

from __future__ import annotations

from functools import partial

import jax
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl import all_reduce as _all_reduce_impl


@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4))
def _all_reduce_core(
    x: Float[Array, "..."],
    axis_name: str | tuple[str, ...],
    mode: str,
    tp_size: int | None,
    collective_id: int | None,
) -> Float[Array, "..."]:
    """Forward-only dispatch to the Pallas one-shot all-reduce."""
    return _all_reduce_impl(x, axis_name=axis_name, mode=mode, tp_size=tp_size, collective_id=collective_id)


def _all_reduce_core_fwd(
    x: Float[Array, "..."],
    axis_name: str | tuple[str, ...],
    mode: str,
    tp_size: int | None,
    collective_id: int | None,
):
    """Custom VJP forward pass: compute output; no residuals needed."""
    out = _all_reduce_impl(x, axis_name=axis_name, mode=mode, tp_size=tp_size, collective_id=collective_id)
    return out, None


def _all_reduce_core_bwd(
    axis_name: str | tuple[str, ...],
    mode: str,
    tp_size: int | None,
    collective_id: int | None,
    residual,
    dy: Float[Array, "..."],
):
    """Custom VJP backward pass: all-reduce the cotangent (self-dual op)."""
    del residual
    grad_x = _all_reduce_impl(dy, axis_name=axis_name, mode=mode, tp_size=tp_size, collective_id=collective_id)
    return (grad_x,)


_all_reduce_core.defvjp(_all_reduce_core_fwd, _all_reduce_core_bwd)


@kernel_registry.register("all_reduce", Platform.PALLAS, Backend.TPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def all_reduce(
    x: Float[Array, "..."],
    axis_name: str | tuple[str, ...],
    mode: str = "auto",
    tp_size: int | None = None,
    collective_id: int | None = 0,
) -> Float[Array, "..."]:
    """All-reduce with a one-shot direct-exchange Pallas kernel for small inputs.

    Must be called inside ``jax.experimental.shard_map`` (or equivalent) so
    that ``axis_name`` is active.  The one-shot mode uses a single-phase
    direct exchange; on v5p-8 the ``lax.psum`` lowering measured faster at
    every size, so ``"auto"`` delegates to it (see ``_pallas_impl``).

    Args:
        x: Local partial values of any rank.
        axis_name: pmap / shard_map axis name for the collective.
        mode: ``"one_shot"`` (force the kernel; raises when its alignment or
            size constraints are violated), ``"ring"`` (delegate to
            ``lax.psum``), or ``"auto"`` (resolves to the ``lax.psum``
            path — measured faster at all sizes on v5p-8).
        tp_size: Collective world size.  Inferred from the active axis context
            or ``jax.device_count()`` when ``None``.
        collective_id: Integer ID for TPU barrier-semaphore allocation.  Use
            distinct IDs for concurrent collectives.

    Returns:
        Elementwise sum over all devices on ``axis_name``; same shape and
        dtype as ``x``.  One-shot accumulation is performed in float32 and
        cast back.

    Raises:
        ValueError: If ``mode`` is invalid, ``tp_size < 1``, or
            ``mode="one_shot"`` constraints are violated.
    """
    return _all_reduce_core(x, axis_name, mode, tp_size, collective_id)


__all__ = ("all_reduce",)
