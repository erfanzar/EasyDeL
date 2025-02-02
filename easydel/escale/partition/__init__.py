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

from .auto_spec import (
	auto_namedsharding,
	auto_partition_spec,
	auto_shard_array,
	vrn_auto_partition_spec,
	convert_sharding_strategy,
	optimize_sharding_for_memory,
	validate_sharding_config,
)
from .constraints import (
	get_names_from_partition_spec,
	make_shard_and_gather_fns,
	match_partition_rules,
	with_sharding_constraint,
	analyze_sharding_strategy,
	create_pattern_based_partition_spec,
	extract_sharding_structure,
	get_shardings_with_structure,
	PartitionAxis,
)

__all__ = (
	"auto_namedsharding",
	"auto_partition_spec",
	"auto_shard_array",
	"vrn_auto_partition_spec",
	"with_sharding_constraint",
	"get_names_from_partition_spec",
	"make_shard_and_gather_fns",
	"match_partition_rules",
	"convert_sharding_strategy",
	"optimize_sharding_for_memory",
	"validate_sharding_config",
	"analyze_sharding_strategy",
	"create_pattern_based_partition_spec",
	"extract_sharding_structure",
	"get_shardings_with_structure",
	"PartitionAxis",
)
