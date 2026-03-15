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

"""Sharding and platform TypedDicts.

Defines configuration structures for distributed model sharding across
devices and hardware platform/backend selection.
"""

from __future__ import annotations

import collections.abc
import typing as tp
from typing import Any, NotRequired, TypedDict

from eformer.escale import PartitionAxis

from easydel.infra.etils import EasyDeLBackends, EasyDeLPlatforms

from .aliases import PartitionRules


class ShardingCfg(TypedDict, total=False):
    """Model sharding configuration for distributed training and inference.

    Attributes:
        axis_dims: Dimensions for each mesh axis (e.g., ``(1, 1, 1, -1)``
            where ``-1`` means auto-compute from device count).
        dcn_axis_dims: Data-center network axis dimensions for multi-slice
            TPU topologies.
        axis_names: Names for each mesh axis (e.g.,
            ``("dp", "fsdp", "ep", "tp")``).
        partition_axis: ``PartitionAxis`` instance defining how model tensors
            map to mesh axes. ``None`` for default partitioning.
        shard_fns: Custom sharding functions keyed by parameter path tuples.
        auto_shard_model: Automatically shard the model using partition rules.
        partition_rules: Regex-based rules mapping parameter names to
            ``PartitionSpec`` values.
        use_ring_of_experts: Use ring-based expert parallelism for MoE layers.
        fsdp_is_ep_bound: Bind FSDP axis to expert parallelism axis.
        sp_is_ep_bound: Bind sequence parallelism axis to expert parallelism.
    """

    axis_dims: NotRequired[collections.abc.Sequence[int]]
    dcn_axis_dims: NotRequired[collections.abc.Sequence[int]]
    axis_names: NotRequired[collections.abc.Sequence[str]]
    partition_axis: NotRequired[PartitionAxis | None]
    shard_fns: NotRequired[collections.abc.Mapping[tuple, tp.Callable[..., Any]] | dict]
    auto_shard_model: NotRequired[bool]
    partition_rules: NotRequired[PartitionRules]
    use_ring_of_experts: NotRequired[bool]
    fsdp_is_ep_bound: NotRequired[bool]
    sp_is_ep_bound: NotRequired[bool]


class PlatformCfg(TypedDict, total=False):
    """Platform and backend configuration for hardware selection.

    Attributes:
        backend: JAX backend to use (e.g., ``"tpu"``, ``"gpu"``, ``"cpu"``).
            ``None`` for auto-detection.
        platform: Target hardware platform. ``None`` for auto-detection.
    """

    backend: NotRequired[EasyDeLBackends | None]
    platform: NotRequired[EasyDeLPlatforms | None]
