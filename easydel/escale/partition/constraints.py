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
import contextlib
from functools import partial
import re
import typing as tp
import warnings

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util as tu
from jax.interpreters import pxla
from jax.lax import with_sharding_constraint as _with_sharding_constraint
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from easydel.utils.traversals import named_tree_map

from ..mesh.validation import names_in_current_mesh


def make_shard_and_gather_fns(
	partition_specs: tp.Dict[str, PartitionSpec],
	mesh: tp.Optional[Mesh] = None,
) -> tp.Tuple[tp.Dict[str, tp.Callable], tp.Dict[str, tp.Callable]]:
	"""
	Create shard and gather functions based on given partition specs and mesh.

	This function generates dictionaries of shard and gather functions that can be used
	to distribute and collect arrays across a JAX mesh. The functions are specifically
	designed for use with Flax's `tu.tree_map`.

	Args:
		partition_specs: A dictionary mapping parameter names to their respective `PartitionSpec`.
		mesh: The JAX mesh to use for sharding. If None, the current mesh is used.

	Returns:
		A tuple containing two dictionaries:
			- `shard_fns`: A dictionary mapping parameter names to their corresponding shard functions.
			- `gather_fns`: A dictionary mapping parameter names to their corresponding gather functions.
	"""
	if mesh is None:
		mesh = jax.interpreters.pxla.thread_resources.env.physical_mesh
		assert not mesh.empty, (
			"You should pass 'mesh' to `make_shard_and_gather_fns` or "
			"at least call that under mesh context manager"
		)

	named_shardings = tu.tree_map(
		lambda p: NamedSharding(mesh=mesh, spec=p),
		partition_specs,
	)

	def make_shard_fn(sharding: NamedSharding) -> tp.Callable:
		"""
		Create a shard function for a specific partition spec.
		"""
		if jax.device_count() == jax.local_device_count():

			def shard_fn(tensor: jnp.ndarray) -> jnp.ndarray:
				with mesh:
					tensor = with_sharding_constraint(arr=tensor, sharding=sharding)
				return tensor
		else:

			@partial(jax.jit, out_shardings=named_shardings)
			def shard_fn(tensor: jnp.ndarray) -> jnp.ndarray:
				return tensor

		return shard_fn

	def make_gather_fn(sharding: NamedSharding) -> tp.Callable:
		"""
		Create a gather function for a specific partition spec.
		"""

		def gather_fn(tensor: jnp.ndarray) -> jnp.ndarray:
			return jax.device_get(
				with_sharding_constraint(
					arr=tensor,
					sharding=NamedSharding(mesh, PartitionSpec()),
				)
			)

		return gather_fn

	shard_fns = tu.tree_map(make_shard_fn, named_shardings)
	gather_fns = tu.tree_map(make_gather_fn, named_shardings)
	return shard_fns, gather_fns


def get_names_from_partition_spec(
	partition_specs: tp.Dict[str, PartitionSpec],
) -> tp.List[str]:
	"""
	Extract axis names from a partition specification.

	This function recursively iterates through the provided `partition_specs`
	dictionary and extracts all unique axis names used in the sharding specifications.

	Args:
		partition_specs: A dictionary mapping parameter names to their respective `PartitionSpec`.

	Returns:
		A list of unique axis names used in the partition specs.
	"""
	names = set()
	if isinstance(partition_specs, dict):
		partition_specs = partition_specs.values()
	for item in partition_specs:
		if item is None:
			continue
		elif isinstance(item, str):
			names.add(item)
		else:
			names.update(get_names_from_partition_spec(item))
	return list(names)


@contextlib.contextmanager
def nullcontext(enter_result=None):
	yield enter_result


def with_sharding_constraint(
	arr: jnp.ndarray,
	sharding: tp.Dict[str, tp.Union[PartitionSpec, NamedSharding]],
) -> jnp.ndarray:
	"""
	Apply sharding constraints if axis names are present in the current mesh.

	This is a smarter version of `jax.lax.with_sharding_constraint`. It only applies the
	sharding constraint if all the axis names specified in the `partition_specs` are
	present in the current JAX mesh.

	Args:
		arr: The JAX array to apply sharding constraints to.
		sharding: A dictionary mapping parameter names to their respective `PartitionSpec`.

	Returns:
		The JAX array with sharding constraints applied (if applicable).
	"""
	if isinstance(arr, (jax.Array, jnp.ndarray)):
		if isinstance(sharding, NamedSharding):
			mesh = sharding.mesh
			sharding = sharding.spec
		else:
			mesh = None
		if mesh is None:
			mesh = pxla.thread_resources.env.physical_mesh
		axis_names = get_names_from_partition_spec(sharding)
		if names_in_current_mesh(*axis_names):
			with mesh or nullcontext():
				arr = _with_sharding_constraint(arr, sharding)
	return arr


def match_partition_rules(
	rules: tp.List[tp.Tuple[str, PartitionSpec]],
	tree: tp.Dict,
) -> tp.Dict:
	"""
	Match partition rules to parameters based on their names.

	This function takes a list of partition rules (regular expressions and
	corresponding `PartitionSpec`) and applies them to a dictionary of parameters
	based on their names. It's useful for automatically defining sharding strategies.

	Args:
		rules: A list of tuples, where each tuple contains:
				 - A regular expression to match parameter names.
				 - A `PartitionSpec` to apply if the name matches.
		tree: A dictionary of parameters, where keys are parameter names.

	Returns:
		A dictionary with the same keys as `tree`, but values are replaced
		with the corresponding `PartitionSpec` based on matching rules.
	"""

	def get_partition_spec(name: str, leaf: jnp.ndarray) -> PartitionSpec:
		"""
		Determine the partition spec for a parameter based on its name.
		"""

		if not hasattr(leaf, "shape"):
			return PartitionSpec()
		if len(leaf.shape) == 0 or np.prod(leaf.shape) == 1:
			""" Don't partition scalar values. """
			return PartitionSpec()
		for rule, ps in rules:
			if re.search(rule, name) is not None:
				if len(ps) > leaf.ndim:
					ps = PartitionSpec(*tuple(ps[: leaf.ndim]))
					warnings.warn(
						f"PartitionSpec Related to {name} went out of range (will be auto trimed to {ps}).",
						stacklevel=1,
					)
				return ps
		raise ValueError(f"Partition rule not found for param: {name}")

	return named_tree_map(get_partition_spec, tree, sep="/")


def analyze_sharding_strategy(
	pytree: tp.Any,
	partition_specs: tp.Dict[str, PartitionSpec],
	mesh: tp.Optional[Mesh] = None,
) -> tp.Dict:
	"""
	Analyzes the effectiveness of a sharding strategy.

	Returns metrics like:
	- Memory usage per device
	- Load balance
	- Communication costs
	"""
	if mesh is None:
		mesh = pxla.thread_resources.env.physical_mesh

	analysis = {
		"total_parameters": 0,
		"sharded_parameters": 0,
		"memory_per_device": {},
		"balance_score": 0.0,
		"partition_stats": {},
	}

	def analyze_leaf(path: str, array: np.ndarray, spec: PartitionSpec):
		total_size = np.prod(array.shape) * array.dtype.itemsize
		analysis["total_parameters"] += np.prod(array.shape)

		if spec != PartitionSpec():
			analysis["sharded_parameters"] += np.prod(array.shape)

		# Calculate per-device memory
		sharded_size = total_size
		for axis, name in enumerate(spec):
			if name is not None:
				sharded_size //= mesh.shape[name]

		return sharded_size

	# Traverse the pytree and collect statistics
	tu.tree_map_with_path(analyze_leaf, pytree, partition_specs)

	return analysis


def create_pattern_based_partition_spec(
	pattern: str,
	mesh: tp.Optional[Mesh] = None,
	default_spec: tp.Optional[PartitionSpec] = None,
) -> tp.Callable[[str, chex.Array], PartitionSpec]:
	"""
	Creates a function that returns PartitionSpec based on parameter name patterns.

	Example:
		pattern_fn = create_pattern_based_partition_spec(
			"attention|mlp->data,hidden->model"
		)
	"""
	if default_spec is None:
		default_spec = PartitionSpec()
	if mesh is None:
		mesh = pxla.thread_resources.env.physical_mesh

	rules = []
	for rule in pattern.split(","):
		if "->" in rule:
			patterns, spec = rule.split("->")
			patterns = patterns.split("|")
			spec = PartitionSpec(*spec.split("."))
			rules.extend((pattern, spec) for pattern in patterns)

	def get_partition_spec(name: str, array: chex.Array) -> PartitionSpec:
		for pattern, spec in rules:
			if re.search(pattern, name):
				return spec
		return default_spec

	return get_partition_spec


AxisType = tp.Optional[tp.Union[tp.Tuple[str, ...], str]]


class PartitionAxis(tp.NamedTuple):
	"""
	A NamedTuple representing different axes of partitioning in a model.

	Each field represents an axis and its corresponding partitioning strategy.
	The value of each field can be:

	* None: The axis is not partitioned.
	* str: The name of the single mesh dimension across which the axis is partitioned.
	* Tuple[str, ...]: A tuple of mesh dimension names, indicating a sharding strategy
		where the axis is split across multiple mesh dimensions.

	Attributes:
		batch_axis: Partitioning strategy for the batch dimension. Defaults to ("fsdp", "dp").
		sequence_axis: Partitioning strategy for the sequence dimension. Defaults to "sp".
		query_sequence_axis: Partitioning strategy for the query sequence dimension. Defaults to "sp".
		head_axis: Partitioning strategy for the attention head dimension. Defaults to "tp".
		key_sequence_axis: Partitioning strategy for the key sequence dimension. Defaults to "sp".
		hidden_state_axis: Partitioning strategy for the hidden state dimension. Defaults to "tp".
		attention_dim_axis: Partitioning strategy for the attention dimension. Defaults to None.
		bias_head_sequence_axis: Partitioning strategy for the bias head sequence dimension. Defaults to None.
		bias_key_sequence_axis: Partitioning strategy for the bias key sequence dimension. Defaults to None.
		generation_query_sequence_axis: Partitioning strategy for the query sequence dimension during generation.
			Defaults to None.
		generation_head_axis: Partitioning strategy for the attention head dimension during generation.
			Defaults to "tp".
		generation_key_sequence_axis: Partitioning strategy for the key sequence dimension during generation.
			Defaults to "sp".
		generation_attention_dim_axis: Partitioning strategy for the attention dimension during generation.
			Defaults to None.
	"""

	batch_axis: AxisType = ("fsdp", "dp")
	sequence_axis: AxisType = "sp"
	query_sequence_axis: AxisType = "sp"
	head_axis: AxisType = "tp"
	key_sequence_axis: AxisType = "sp"
	hidden_state_axis: AxisType = "tp"
	attention_dim_axis: AxisType = None
	bias_head_sequence_axis: AxisType = None
	bias_key_sequence_axis: AxisType = None

	generation_query_sequence_axis: AxisType = None
	generation_head_axis: AxisType = "tp"
	generation_key_sequence_axis: AxisType = "sp"
	generation_attention_dim_axis: AxisType = None
