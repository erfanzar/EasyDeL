# Copyright 2026 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
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

"""Helper utilities for automatic sharding rule generation and analysis.

This submodule provides classes and functions for defining, composing, and
analyzing sharding rules for JAX arrays. It includes rule-based sharding
strategies that can automatically determine optimal partition specifications
based on array shapes, memory constraints, and mesh configurations.

Classes:
    ShardingRule: Abstract base class for all sharding rules.
    AutoShardingRule: Automatically determines sharding based on array shapes.
    ShapeBasedShardingRule: Creates sharding based on array shape patterns.
    MemoryConstrainedShardingRule: Creates sharding to fit memory constraints.
    CompositeShardingRule: Combines multiple rules with priority ordering.
    ShardingAnalyzer: Validates and estimates memory usage for sharding strategies.

Functions:
    barrier_sync: Synchronizes all JAX processes at a barrier point.

Example:
    >>> from eformer.escale.helpers import AutoShardingRule, ShardingAnalyzer
    >>> # Create an auto sharding rule
    >>> rule = AutoShardingRule(mesh=mesh, axis_names=['dp', 'tp'])
    >>> # Apply the rule to a pytree
    >>> partition_specs = rule.apply(model_params)
    >>> # Analyze the sharding strategy
    >>> analyzer = ShardingAnalyzer(mesh=mesh)
    >>> issues = analyzer.validate_partition_specs(model_params, partition_specs)
"""

from .base import (
    AutoShardingRule,
    CompositeShardingRule,
    MemoryConstrainedShardingRule,
    ShapeBasedShardingRule,
    ShardingAnalyzer,
    ShardingRule,
    barrier_sync,
)

__all__ = (
    "AutoShardingRule",
    "CompositeShardingRule",
    "MemoryConstrainedShardingRule",
    "ShapeBasedShardingRule",
    "ShardingAnalyzer",
    "ShardingRule",
    "barrier_sync",
)
