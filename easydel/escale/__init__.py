# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

from .mesh import (
	MeshPartitionHelper,
	create_mesh,
	names_in_current_mesh,
	parse_mesh_from_string,
)
from .partition import (
	analyze_sharding_strategy,
	auto_namedsharding,
	auto_partition_spec,
	auto_shard_array,
	convert_sharding_strategy,
	create_pattern_based_partition_spec,
	get_names_from_partition_spec,
	make_shard_and_gather_fns,
	match_partition_rules,
	optimize_sharding_for_memory,
	validate_sharding_config,
	vrn_auto_partition_spec,
	with_sharding_constraint,
	extract_sharding_structure,
	get_shardings_with_structure,
	PartitionAxis,
)

from .helpers import (
	AutoShardingRule,
	CompositeShardingRule,
	MemoryConstrainedShardingRule,
	ShapeBasedShardingRule,
	ShardingAnalyzer,
	ShardingRule,
)


__all__ = (
	"AutoShardingRule",
	"CompositeShardingRule",
	"MemoryConstrainedShardingRule",
	"ShapeBasedShardingRule",
	"ShardingAnalyzer",
	"ShardingRule",
	"create_mesh",
	"parse_mesh_from_string",
	"names_in_current_mesh",
	"MeshPartitionHelper",
	"auto_namedsharding",
	"auto_partition_spec",
	"auto_shard_array",
	"vrn_auto_partition_spec",
	"with_sharding_constraint",
	"extract_sharding_structure",
	"get_shardings_with_structure",
	"PartitionAxis",
	"get_names_from_partition_spec",
	"convert_sharding_strategy",
	"validate_sharding_config",
	"optimize_sharding_for_memory",
	"analyze_sharding_strategy",
	"create_pattern_based_partition_spec",
	"make_shard_and_gather_fns",
	"match_partition_rules",
)
