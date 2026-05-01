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

"""EasyDeL semantic partition-axis extensions.

This module extends Spectrax's ``PartitionAxis`` system with EasyDeL-specific
semantic axes. In particular it introduces the ``ATTN_DP`` axis used to shard
attention/KV-cache data-parallel work independently of the model's primary
``DP`` axis, which is important for eSurge's KV-page sharding.

Public API:
    - ``ATTN_DP``: The string token identifying the attention data-parallel axis.
    - ``register_attention_data_parallel_axis``: Override the axis rule for ``ATTN_DP``.
    - ``reset_attention_data_parallel_axis``: Restore the default rule.
    - ``resolve_attention_data_parallel_axis``: Look up the resolved axis rule
      for the current partition manager.
"""

from __future__ import annotations

import typing as tp

from spectrax import PartitionAxis
from spectrax.common_types import DP, MODE_PREFILL, NOT_GIVEN

from easydel.infra.sharding import AxisPolicy, RuntimeShardingResolver, coerce_runtime_sharding_resolver

ATTN_DP = "__ATTN_DP__"
_DEFAULT_ATTN_DP_RULE = DP


def _normalize_axis_rule(axis_rule: tp.Any) -> tp.Any:
    """Normalize an axis rule value before registration.

    String rules are stripped of surrounding whitespace and validated to be
    non-empty. Non-string rules (e.g. tuples or other partition specs) are
    returned unchanged.

    Args:
        axis_rule: Any value accepted as a ``PartitionAxis`` rule.

    Returns:
        The normalized rule value, suitable for ``PartitionAxis.register``.

    Raises:
        ValueError: If ``axis_rule`` is a string that is empty after stripping.
    """
    if isinstance(axis_rule, str):
        normalized = axis_rule.strip()
        if not normalized:
            raise ValueError("Axis rule strings must be non-empty.")
        return normalized
    return axis_rule


def register_attention_data_parallel_axis(
    axis_rule: tp.Any = _DEFAULT_ATTN_DP_RULE,
    *,
    generation_axis_rule: tp.Any = NOT_GIVEN,
) -> None:
    """Register the semantic axis used for attention/KV-cache data parallelism.

    Args:
        axis_rule: The partition axis rule to use for attention data
            parallelism during training. Defaults to ``DP``.
        generation_axis_rule: Optional separate axis rule to use during
            generation/inference. When ``NOT_GIVEN``, uses ``axis_rule``.
    """
    PartitionAxis.register(
        ATTN_DP,
        _normalize_axis_rule(axis_rule),
        generation_axis_rule=generation_axis_rule,
        override=True,
    )


def reset_attention_data_parallel_axis() -> None:
    """Reset ``ATTN_DP`` to follow ``PartitionAxis.data_parallel_axis``.

    Restores the default behavior where attention data parallelism
    uses the same axis rule as the global ``DP`` partition axis.
    """
    register_attention_data_parallel_axis(_DEFAULT_ATTN_DP_RULE)


def resolve_attention_data_parallel_axis(
    partition_axis_or_manager: PartitionAxis | AxisPolicy | RuntimeShardingResolver,
    *,
    mode: str = MODE_PREFILL,
) -> tp.Any:
    """Resolve the configured attention/KV-cache data-parallel axis rule.

    This keeps eSurge's KV-page sharding independent from the model's standard
    data-parallel axis. The returned value is suitable for mesh-size lookups,
    ``jax.lax.axis_index``, and other low-level sharding helpers.

    Args:
        partition_axis_or_manager: Either a ``PartitionAxis`` object, an
            ``AxisPolicy``, or a ``RuntimeShardingResolver``. Anything that can
            be coerced into a runtime sharding resolver is acceptable.
        mode: Execution mode used to disambiguate between training and
            generation axis rules. Defaults to ``MODE_PREFILL``.

    Returns:
        The resolved axis rule (typically a string mesh-axis name or a tuple
        of names) that should be used for attention/KV-cache data parallelism.
    """
    if isinstance(partition_axis_or_manager, PartitionAxis):
        paxis = partition_axis_or_manager
    else:
        paxis = coerce_runtime_sharding_resolver(partition_axis_or_manager).paxis
    return paxis.resolve_axis([ATTN_DP], mode=mode)[0]


try:
    PartitionAxis.register(ATTN_DP, _DEFAULT_ATTN_DP_RULE)
except ValueError:
    # Respect existing custom registration if ATTN_DP was already configured.
    pass


__all__ = [
    "ATTN_DP",
    "register_attention_data_parallel_axis",
    "reset_attention_data_parallel_axis",
    "resolve_attention_data_parallel_axis",
]
