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


__all__ = ("HiddenStateTPReplicated", "gather_shared_input")
