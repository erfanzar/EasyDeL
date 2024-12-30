import re
from typing import Callable, List, Optional, Sequence, Tuple, Union
import warnings

import chex
import flax
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils
from jax.experimental.mesh_utils import create_device_mesh
from jax.interpreters import pxla
from jax.lax import with_sharding_constraint as _with_sharding_constraint
from jax.sharding import Mesh, NamedSharding, PartitionSpec


def make_shard_and_gather_fns(
	partition_specs: dict[str, PartitionSpec],
	mesh: Optional[jax.sharding.Mesh] = None,
) -> Tuple[dict[Callable], dict[Callable]]:
	"""
	Create shard and gather functions based on given partition specs and mesh.

	This function generates dictionaries of shard and gather functions that can be used
	to distribute and collect arrays across a JAX mesh. The functions are specifically
	designed for use with Flax's `jax.tree_util.tree_map`.

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

	named_shardings = jax.tree_util.tree_map(
		lambda p: NamedSharding(mesh=mesh, spec=p), partition_specs
	)

	def make_shard_fn(partition_spec: NamedSharding) -> Callable:
		"""
		Create a shard function for a specific partition spec.
		"""
		jax_shard_function = jax.jit(
			lambda x: x, in_shardings=None, out_shardings=partition_spec
		)

		def shard_fn(tensor: jnp.ndarray) -> jnp.ndarray:
			return jax_shard_function(tensor).block_until_ready()

		return shard_fn

	def make_gather_fn(partition_spec: NamedSharding) -> Callable:
		"""
		Create a gather function for a specific partition spec.
		"""
		jax_gather_fn = jax.jit(
			lambda x: x,
			in_shardings=partition_spec,
			out_shardings=NamedSharding(mesh, PartitionSpec()),
		)

		def gather_fn(tensor: jnp.ndarray) -> jnp.ndarray:
			return jax.device_get(jax_gather_fn(tensor))

		return gather_fn

	shard_fns = jax.tree_util.tree_map(make_shard_fn, named_shardings)
	gather_fns = jax.tree_util.tree_map(make_gather_fn, named_shardings)
	return shard_fns, gather_fns


def get_jax_mesh(axis_dims: str, names: Sequence[str]) -> jax.sharding.Mesh:
	"""
	Create a JAX mesh based on axis dimensions and names.

	Args:
	    axis_dims: A comma-separated string specifying the dimensions of the mesh, e.g., "1,2,4".
	               A "!" prefix indicates mesh axis splitting.
	    names: A sequence of names for the mesh axes, e.g., ["data", "model"].

	Returns:
	    A JAX mesh object.
	"""
	if axis_dims.startswith("!"):
		mesh_axis_splitting = True
		axis_dims = axis_dims[1:]
	else:
		mesh_axis_splitting = False

	if ":" in axis_dims:
		dims = []
		dim_names = []
		for axis in axis_dims.split(","):
			name, dim = axis.split(":")
			assert name in names, f"Axis name '{name}' not found in provided names: {names}"
			dims.append(int(dim))
			dim_names.append(name)
		assert set(dim_names) == set(names), "Not all axis names were used in 'axis_dims'"
	else:
		dims = [int(x) for x in axis_dims.split(",")]
		dim_names = names
	assert len(dims) == len(names), "Number of dimensions and names must match"

	mesh_shape = np.arange(jax.device_count()).reshape(dims).shape
	if mesh_axis_splitting:
		physical_mesh = np.array(jax.devices()).reshape(mesh_shape)
	else:
		physical_mesh = mesh_utils.create_device_mesh(mesh_shape)
	return Mesh(physical_mesh, dim_names)


def auto_partition_spec(
	x: chex.Array,
	mesh: Optional[Mesh] = None,
	names: Optional[List[Union[str, Tuple[str, ...]]]] = None,
	min_sharding_size: Optional[int] = None,
	reverse: bool = False,
) -> PartitionSpec:
	"""
	Create an optimized PartitionSpec to shard an array across a device mesh.

	Args:
	    x: The input array to be sharded.
	    mesh: The device mesh to shard across. If None, uses the current thread's mesh.
	    names: List of mesh axis names to use for sharding. If None, derives from mesh shape.
	    min_sharding_size: Minimum size of array to shard. If None, uses mesh device count.
	    reverse: If True, reverses dimension sorting order for sharding assignment.

	Returns:
	    PartitionSpec: Optimized sharding specification for the input array.

	Raises:
	    ValueError: If mesh is unavailable or invalid names are provided.
	    TypeError: If input types are incorrect.
	"""
	# Validate input array
	if not isinstance(x, (chex.Array, np.ndarray)):
		raise TypeError(f"Expected array input, got {type(x)}")

	# Get or validate mesh
	if mesh is None:
		mesh = pxla.thread_resources.env.physical_mesh
		if mesh.empty:
			raise ValueError(
				"No mesh available. Provide a mesh or use within a mesh context."
			)

	# Calculate minimum sharding size
	min_sharding_size = min_sharding_size or np.prod(mesh.devices.shape)

	# Early return for small arrays
	array_size = np.prod(x.shape)
	if array_size < min_sharding_size:
		return PartitionSpec()

	# Prepare mesh names
	if not names:
		# Sort mesh axes by size in descending order
		names = [mesh.axis_names[i] for i in np.argsort([-s for s in mesh.devices.shape])]

	# Create mesh size lookup for efficient access
	mesh_sizes = {
		name: (
			np.prod([mesh.shape[n] for n in name])
			if isinstance(name, tuple)
			else mesh.shape[name]
		)
		for name in names
	}

	# Sort dimensions by size
	dim_indices = np.argsort([-dim if not reverse else dim for dim in x.shape])

	# Initialize partition spec
	partition_spec = [None] * len(x.shape)
	remaining_names = set(names)

	# Assign sharding
	for dim_idx in dim_indices:
		dim_size = x.shape[dim_idx]

		# Find best matching mesh axis
		best_name = None
		for name in remaining_names:
			mesh_size = mesh_sizes[name]
			if dim_size % mesh_size == 0:
				best_name = name
				break

		if best_name:
			partition_spec[dim_idx] = best_name
			remaining_names.remove(best_name)

		if not remaining_names:
			break

	return PartitionSpec(*partition_spec)


def vrn_auto_partition_spec(
	x: chex.Array,
	mesh: Optional[Mesh] = None,
	names: Optional[List[Union[str, Tuple[str, ...]]]] = None,
	min_sharding_size: Optional[int] = None,
	reverse: bool = False,
) -> PartitionSpec:
	"""
	Create an optimized PartitionSpec to shard an array across a device mesh.

	Args:
	    x: The input array to be sharded.
	    mesh: The device mesh to shard across. If None, uses the current thread's mesh.
	    names: List of mesh axis names to use for sharding. If None, derives from mesh shape.
	    min_sharding_size: Minimum size of array to shard. If None, uses the product of mesh device shape.
	    reverse: If True, reverses the sorting order of array dimensions.

	Returns:
	    A PartitionSpec describing optimal array sharding.

	Raises:
	    ValueError: If mesh is unavailable or invalid names are provided.
	    TypeError: If input types are incorrect.
	"""
	# Input validation
	if not isinstance(x, (np.ndarray, chex.Array)):
		raise TypeError(f"Expected array input, got {type(x)}")

	# Get mesh
	if mesh is None:
		mesh = pxla.thread_resources.env.physical_mesh
		if mesh.empty:
			raise ValueError(
				"No mesh available. Provide a mesh or use within a mesh context manager."
			)

	# Calculate minimum sharding size
	min_sharding_size = min_sharding_size or int(np.prod(mesh.devices.shape))

	# Early return for small arrays
	array_size = np.prod(x.shape)
	if array_size < min_sharding_size:
		return PartitionSpec()

	# Prepare mesh names
	if not names:
		# Sort mesh axes by size in descending order
		names = [mesh.axis_names[i] for i in np.argsort([-s for s in mesh.devices.shape])]

	# Pre-calculate mesh sizes for performance
	mesh_sizes = {
		name: (
			np.prod([mesh.shape[n] for n in name])
			if isinstance(name, tuple)
			else mesh.shape[name]
		)
		for name in names
	}

	# Initialize partition spec
	partition_spec = [None] * len(x.shape)

	# Calculate dimension ordering
	dim_order = np.argsort([-dim for dim in x.shape] if not reverse else x.shape)

	# Assign sharding
	remaining_names = names.copy()
	for dim_idx in dim_order:
		dim_size = x.shape[dim_idx]

		# Find the best matching mesh axis
		for name in remaining_names:
			mesh_size = mesh_sizes[name]

			if dim_size % mesh_size == 0:
				partition_spec[dim_idx] = name
				remaining_names.remove(name)
				break

	return PartitionSpec(*partition_spec)


def auto_shard_array(
	x: chex.Array,
	mesh: Optional[Mesh] = None,
	names: Optional[List[Union[str, Tuple[str, ...]]]] = None,
	min_sharding_size: Optional[int] = None,
	reverse: bool = False,
):
	"""
	Shards an array across a device mesh according to an automatically derived PartitionSpec.

	This function acts as a wrapper around `pjit(x, in_axis_resources=...)`.

	Args:
	    x: The input array to be sharded.
	    mesh: The device mesh to shard across. If None, uses the current thread's mesh.
	    names: List of mesh axis names to use for sharding. If None, derives from mesh shape.
	    min_sharding_size: Minimum size of array to shard. If None, uses the product of mesh device shape.
	    reverse: If True, reverses the sorting order of array dimensions.

	Returns:
	    The sharded array.
	"""
	if mesh is None:
		mesh = pxla.thread_resources.env.physical_mesh
		if mesh.empty:
			raise ValueError(
				"`auto_shard_array` needs to be used with a mesh. Pass a mesh as an argument "
				"or use this function under a mesh context manager."
			)
	partition_spec = auto_partition_spec(
		x=x,
		mesh=mesh,
		names=names,
		min_sharding_size=min_sharding_size,
		reverse=reverse,
	)
	with mesh:
		return with_sharding_constraint(x=x, partition_specs=partition_spec)


def auto_namedsharding(
	mesh: Optional[Mesh] = None,
	names: Optional[List[Union[str, Tuple[str, ...]]]] = None,
	min_sharding_size: Optional[int] = None,
	reverse: bool = False,
):
	"""
	Returns a function that creates a NamedSharding for an array based on the provided parameters.

	Args:
	    mesh: The device mesh to shard across. If None, uses the current thread's mesh.
	    names: List of mesh axis names to use for sharding. If None, derives from mesh shape.
	    min_sharding_size: Minimum size of array to shard. If None, uses the product of mesh device shape.
	    reverse: If True, reverses the sorting order of array dimensions.

	Returns:
	    A function that takes an array as input and returns a NamedSharding object.
	"""

	def _named_sharding_fn(x: chex.Array):
		return NamedSharding(
			mesh,
			auto_partition_spec(
				x=x,
				mesh=mesh,
				names=names,
				min_sharding_size=min_sharding_size,
				reverse=reverse,
			),
		)

	return _named_sharding_fn


def names_in_current_mesh(*names: str) -> bool:
	"""
	Check if the given names are present in the current JAX mesh.

	Args:
	    *names: Variable number of axis names to check.

	Returns:
	    True if all given names are present in the current mesh, False otherwise.
	"""
	mesh_axis_names = pxla.thread_resources.env.physical_mesh.axis_names
	return set(names) <= set(mesh_axis_names)


def get_names_from_partition_spec(
	partition_specs: dict[str, PartitionSpec],
) -> list[str]:
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


def with_sharding_constraint(
	x: jnp.ndarray, partition_specs: dict[str, PartitionSpec]
) -> jnp.ndarray:
	"""
	Apply sharding constraints if axis names are present in the current mesh.

	This is a smarter version of `jax.lax.with_sharding_constraint`. It only applies the
	sharding constraint if all the axis names specified in the `partition_specs` are
	present in the current JAX mesh.

	Args:
	    x: The JAX array to apply sharding constraints to.
	    partition_specs: A dictionary mapping parameter names to their respective `PartitionSpec`.

	Returns:
	    The JAX array with sharding constraints applied (if applicable).
	"""
	axis_names = get_names_from_partition_spec(partition_specs)
	if names_in_current_mesh(*axis_names):
		x = _with_sharding_constraint(x, partition_specs)
	return x


def wrap_function_with_rng(rng: jnp.ndarray) -> Callable:
	"""
	Wrap a function to automatically manage RNG splitting.

	This decorator simplifies the use of RNGs within functions by handling the
	splitting of the RNG key. When the wrapped function is called, it splits
	the provided RNG key, passes the split key to the original function, and
	updates the RNG state.

	Args:
	    rng: The initial JAX RNG key.

	Returns:
	    A wrapped function that manages RNG splitting internally.
	"""

	def wrap_function(function: Callable) -> Callable:
		"""
		Inner decorator function.
		"""

		def wrapped(*args, **kwargs):
			"""
			The wrapped function that handles RNG splitting.
			"""
			nonlocal rng
			rng, split_rng = jax.random.split(rng)
			return function(split_rng, *args, **kwargs)

		return wrapped

	return wrap_function


def get_metrics(metrics: dict, unreplicate: bool = False, stack: bool = False) -> dict:
	"""
	Process and aggregate metrics.

	Args:
	    metrics: A dictionary of metrics, potentially replicated across devices.
	    unreplicate: If True, unreplicate the metrics before processing.
	    stack: If True, stack the metrics along a new axis.

	Returns:
	    A dictionary of processed metrics.
	"""
	if unreplicate:
		metrics = flax.jax_utils.unreplicate(metrics)
	metrics = jax.device_get(metrics)
	if stack:
		return jax.tree_util.tree_map(lambda *args: np.stack(args), *metrics)
	else:
		return {key: float(val) for key, val in metrics.items()}


def tree_path_to_string(path: tuple, sep: Optional[str] = None) -> str:
	"""
	Convert a JAX tree path to a string representation.

	Args:
	    path: The JAX tree path tuple.
	    sep: Separator to use when joining path elements.

	Returns:
	    The string representation of the path.
	"""
	keys = []
	for key in path:
		if isinstance(key, jax.tree_util.SequenceKey):
			keys.append(str(key.idx))
		elif isinstance(key, jax.tree_util.DictKey):
			keys.append(str(key.key))
		elif isinstance(key, jax.tree_util.GetAttrKey):
			keys.append(str(key.name))
		elif isinstance(key, jax.tree_util.FlattenedIndexKey):
			keys.append(str(key.key))
		else:
			keys.append(str(key))
	if sep is None:
		return tuple(keys)  # Return a tuple of strings if no separator
	return sep.join(keys)


def flatten_tree(
	xs: dict,
	is_leaf: Optional[Callable] = None,
	sep: Optional[str] = None,
) -> dict:
	"""
	Flatten a JAX tree and convert paths to strings.

	Args:
	    xs: The JAX tree to flatten.
	    is_leaf: Optional function to determine leaf nodes.
	    sep: Separator to use when joining path elements.

	Returns:
	    A flattened dictionary with string keys representing the tree paths.
	"""
	flattened, _ = jax.tree_util.tree_flatten_with_path(xs, is_leaf=is_leaf)
	output = {}
	for key, val in flattened:
		output[tree_path_to_string(key, sep=sep)] = val
	return output


def named_tree_map(
	f: Callable,
	tree: dict,
	*rest,
	is_leaf: Optional[Callable] = None,
	sep: Optional[str] = None,
):
	"""
	An extended version of `jax.tree_util.tree_map`.

	This function extends `jax.tree_util.tree_map` by providing the path
	(as a string) to the current leaf node as an argument to the mapped function `f`.

	Args:
	    f: The function to apply to each leaf node, taking the path and value as input.
	    tree: The JAX tree to map over.
	    *rest: Additional arguments to be passed to `f`.
	    is_leaf: Optional function to determine leaf nodes.
	    sep: Separator to use when joining path elements.

	Returns:
	    A new tree with the same structure as `tree` but with the values modified by `f`.
	"""
	return jax.tree_util.tree_map_with_path(
		lambda path, x, *r: f(tree_path_to_string(path, sep=sep), x, *r),
		tree,
		*rest,
		is_leaf=is_leaf,
	)


def match_partition_rules(rules: list[Tuple[str, PartitionSpec]], params: dict) -> dict:
	"""
	Match partition rules to parameters based on their names.

	This function takes a list of partition rules (regular expressions and
	corresponding `PartitionSpec`) and applies them to a dictionary of parameters
	based on their names. It's useful for automatically defining sharding strategies.

	Args:
	    rules: A list of tuples, where each tuple contains:
	           - A regular expression to match parameter names.
	           - A `PartitionSpec` to apply if the name matches.
	    params: A dictionary of parameters, where keys are parameter names.

	Returns:
	    A dictionary with the same keys as `params`, but values are replaced
	    with the corresponding `PartitionSpec` based on matching rules.
	"""

	def get_partition_spec(name: str, leaf: jnp.ndarray) -> PartitionSpec:
		"""
		Determine the partition spec for a parameter based on its name.
		"""
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

	return named_tree_map(get_partition_spec, params, sep="/")


def get_weight_decay_mask(exclusions: list[str]) -> Callable:
	"""
	Create a weight decay mask function based on exclusion rules.

	Args:
	    exclusions: A list of regular expressions defining parameter names
	               to exclude from weight decay.

	Returns:
	    A function that takes a parameter dictionary and returns a mask
	    (a PyTree with the same structure as the parameters) indicating
	    which parameters should be subject to weight decay.
	"""

	def decay(name: str, _) -> bool:
		"""
		Determine if a parameter should be decayed based on its name.
		"""
		for rule in exclusions:
			if re.search(rule, name) is not None:
				return False
		return True

	def weight_decay_mask(params: dict) -> dict:
		"""
		Apply the weight decay mask to a parameter dictionary.
		"""
		return named_tree_map(decay, params, sep="/")

	return weight_decay_mask


def tree_apply(fns: dict, tree: dict):
	"""
	Apply a dictionary of functions to a corresponding PyTree.

	Args:
	    fns: A dictionary where keys match the PyTree structure and values are functions.
	    tree: The PyTree to apply functions to.

	Returns:
	    A new PyTree with the same structure as `tree`, but with values modified by the functions in `fns`.
	"""
	return jax.tree_util.tree_map(lambda fn, x: fn(x), fns, tree)


def create_mesh(
	axis_dims: Sequence[int] = (1, -1, 1, 1),
	axis_names: Sequence[str] = ("dp", "fsdp", "tp", "sp"),
	backend: str = "",
) -> jax.sharding.Mesh:
	"""
	Create a JAX mesh with specified dimensions and names.

	Args:
	    axis_dims: A sequence of integers representing the size of each mesh dimension.
	               A dimension of -1 indicates that it should be inferred automatically.
	    axis_names: A sequence of strings representing the names of the mesh dimensions.
	    backend: The JAX backend to use. If "", the default backend is used.

	Returns:
	    A JAX mesh object.
	"""
	array_devices = jax.numpy.ones(
		(len(jax.devices(backend)) if backend else len(jax.devices()), 1)
	)
	resh = array_devices.reshape(axis_dims).shape

	return jax.sharding.Mesh(create_device_mesh(resh), axis_names)
