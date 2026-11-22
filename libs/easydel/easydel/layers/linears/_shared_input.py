# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Gather an activation shared by several column-parallel projections once over tensor parallelism.

A hidden state ``[B, S, D]`` lives sharded over ``tp`` on its feature axis. Every column-parallel projection that
reads it needs it whole on each ``tp`` rank, and in the backward every projection's weight gradient ``x^T dy`` needs
it again. Under rematerialization XLA gathers the recomputed input separately for each of those weight-gradient dots.

:func:`gather_shared_input` constrains the input to its tp-replicated layout once, so every projection reads the
same gathered value, and its custom VJP constrains the summed input cotangent straight back to the tp-sharded hidden
layout (one reduce-scatter instead of an all-reduce). The projections themselves are untouched: each still runs its
own module call, so quantized, LoRA and hooked layers keep working, and parameters and checkpoints do not change.

:func:`resolve_gather_shared_projection_input` decides whether a layer uses it, from
``config.gather_shared_projection_input``: ``True`` on, ``False`` off (the escape hatch), ``None`` (the default)
automatic: on in inference mode (:func:`easydel.utils.set_inference_mode`, which eSurge's model step enters) when the
hidden axis of the projections' input is sharded over a mesh axis larger than 1 under the active layout, off
otherwise. Outside inference mode ``None`` behaves like ``False``, so training compiles do not change.
"""

from __future__ import annotations

import typing as tp

import jax
from eformer.loggings import get_logger
from jaxtyping import Array
from spectrax import apply_logical_sharding
from spectrax import common_types as ct

logger = get_logger(__name__)


class HiddenStateTPReplicated(ct.DynamicShardingAxes):
    """``[B, S, D]`` hidden states with the feature axis replicated (batch / sequence sharded as usual)."""

    axes: tp.ClassVar = [ct.BATCH, ct.QUERY_LENGTH, ct.EMPTY]
    mode: tp.ClassVar = 1


def gather_shared_input(x: Array, config: tp.Any) -> Array:
    """Gather the feature axis of a ``[B, S, D]`` hidden state over ``tp`` once for every projection that reads it.

    The forward value is ``x`` (only its layout changes); the cotangent is constrained back to the regular
    tp-sharded hidden-state layout. Reverse-mode only (custom VJP). A no-op (``x`` returned unchanged) without
    tensor parallelism, for non-3-D inputs, or when ``config`` has no runtime sharding resolver (warns once).

    Args:
        x: Hidden state ``[batch, seq, hidden]``.
        config: Owning model config (mesh and runtime sharding resolver).

    Returns:
        ``x`` in its tp-replicated layout.
    """
    if getattr(x, "ndim", 0) != 3:
        return x
    from easydel.layers.layouts._runtime import tensor_parallel_size  # lazy: the layouts package imports linears

    if tensor_parallel_size(config, arr=x) <= 1:
        return x
    resolver = getattr(config, "runtime_sharding_resolver", None)
    if resolver is None:
        logger.warn_once("gather_shared_projection_input: no runtime sharding resolver on the config; not applied.")
        return x

    def full(a):
        return apply_logical_sharding(a, dynamic_axes=HiddenStateTPReplicated, partition_manager=resolver)

    def local(a):
        return apply_logical_sharding(a, dynamic_axes=ct.HiddenStateSharding, partition_manager=resolver)

    @jax.custom_vjp
    def gather(a):
        return full(a)

    gather.defvjp(lambda a: (full(a), None), lambda _res, ct_: (local(ct_),))
    return gather(x)


def _hidden_axis_is_sharded(x: Array, config: tp.Any) -> bool:
    """Whether the hidden (last) axis of the ``[B, S, D]`` input ``x`` lives on a mesh axis of size > 1.

    Resolved through the config's runtime sharding resolver on the stage mesh of ``x``, so the active layout scope
    counts: under :func:`easydel.infra.sharding.decode_mode_specs` (eSurge's pure-decode buckets) the decode hidden
    axis applies, which is replicated by default.

    Args:
        x: The shared projection input.
        config: Owning model config (mesh and runtime sharding resolver).

    Returns:
        ``True`` when the resolved hidden axis spans more than one device.
    """
    resolver = getattr(config, "runtime_sharding_resolver", None)
    shape = tuple(getattr(x, "shape", ()))
    if resolver is None or len(shape) != 3:
        return False
    from easydel.infra.sharding import mesh_axis_size, resolve_stage_mesh  # lazy, like the layouts import above

    mesh = resolve_stage_mesh(getattr(config, "mesh", None), arr=x)
    if mesh is None:
        return False
    spec = resolver.with_mesh(mesh).resolve(dynamic_axes=ct.HiddenStateSharding, shape=shape)
    return len(spec) > 2 and mesh_axis_size(mesh, spec[2]) > 1


def resolve_gather_shared_projection_input(config: tp.Any, x: Array) -> bool:
    """Whether the column projections reading ``x`` share one gathered copy of it (:func:`gather_shared_input`).

    The single decision point for ``config.gather_shared_projection_input``. The field is read with ``getattr``, so
    a config object without it takes the automatic rule.

    - ``True``: gather. :func:`gather_shared_input` stays a no-op where it cannot apply (no tensor parallelism; no
      runtime sharding resolver, which it warns about once).
    - ``False``: never gather (the escape hatch).
    - ``None``: automatic. Gather in inference mode when the hidden axis of ``x`` is sharded over a mesh axis of
      size > 1 under the active layout: eSurge's prefill and mixed buckets on a tensor-parallel mesh, where every
      column projection needs the whole hidden state anyway. eSurge's pure-decode buckets keep the hidden state
      replicated (``decode_mode_specs``), so they take the ``False`` path. Outside inference mode ``None`` is off,
      so training compiles keep today's program.

    Args:
        config: Owning model config.
        x: The shared projection input ``[batch, seq, hidden]``.

    Returns:
        Whether to route ``x`` through :func:`gather_shared_input`.
    """
    value = getattr(config, "gather_shared_projection_input", None)
    if value is not None:
        return bool(value)
    from easydel.utils.inference_mode import is_inference_mode  # lazy, like the imports above

    return is_inference_mode() and _hidden_axis_is_sharded(x, config)


__all__ = ("HiddenStateTPReplicated", "gather_shared_input", "resolve_gather_shared_projection_input")
