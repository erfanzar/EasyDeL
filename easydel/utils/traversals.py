"""Utility functions for managing and manipulating nnx module states."""

import typing as tp
import warnings

import chex
import jax
import jax.numpy as jnp
from flax import nnx, struct
from flax.nnx import traversals


class MetaValueRecreator:
	"""Helper class for recreating meta values with state tracking"""

	def __init__(self, seed: int = 42):
		self._count = 0
		self._rng = jax.random.PRNGKey(seed)

	def get_count(self) -> jnp.ndarray:
		count = self._count
		self._count += 1
		return jnp.array(count, dtype=jnp.uint32)

	def get_rng(self) -> jax.random.PRNGKey:
		key, self._rng = jax.random.split(self._rng)
		return key


class TreePath:
	"""Helper class for managing nested dictionary paths"""

	def __init__(self, parts: tuple, separator: tp.Optional[str] = None):
		self.parts = parts
		self.separator = separator

	def __str__(self) -> str:
		if self.separator is None:
			return self.parts
		return self.separator.join(self.parts)


@struct.dataclass
class _EmptyNode:
	pass


@chex.dataclass
class StateValidationResult:
	is_valid: bool
	missing_keys: set
	invalid_types: tp.Dict[str, type]


empty_node = _EmptyNode()
M = tp.TypeVar("M")


def _dict_flatten_dict(xs, keep_empty_nodes=False, is_leaf=None, sep=None):
	assert isinstance(xs, dict), f"expected dict; got {type(xs)}"

	def _key(path):
		if sep is None:
			return path
		return sep.join(path)

	def _flatten(xs, prefix):
		if not isinstance(xs, dict) or (is_leaf and is_leaf(prefix, xs)):
			return {_key(prefix): xs}
		result = {}
		is_empty = True
		for key, value in xs.items():
			is_empty = False
			path = prefix + (key,)
			result.update(_flatten(value, path))
		if keep_empty_nodes and is_empty:
			if prefix == ():  # when the whole input is empty
				return {}
			return {_key(prefix): empty_node}
		return result

	return _flatten(xs, ())


def _dict_unflatten_dict(xs, sep=None):
	assert isinstance(xs, dict), f"input is not a dict; it is a {type(xs)}"
	result = {}
	for path, value in xs.items():
		if sep is not None:
			path = path.split(sep)
		if value is empty_node:
			value = {}
		cursor = result
		for key in path[:-1]:
			if key not in cursor:
				cursor[key] = {}
			cursor = cursor[key]
		cursor[path[-1]] = value
	return result


def flatten_dict(
	xs: tp.Union[dict, tp.Mapping],
	keep_empty_nodes: bool = False,
	is_leaf: tp.Optional[tp.Callable[[tuple, tp.Any], bool]] = None,
	sep: tp.Optional[str] = None,
	_prefix: tuple = (),
) -> tp.Dict[tp.Union[tuple, str], tp.Any]:
	"""
	Enhanced dictionary flattening with better type handling and validation.

	Args:
	    xs: Dictionary or mapping to flatten
	    keep_empty_nodes: Whether to keep empty dictionary nodes
	    is_leaf: Optional function to determine leaf nodes
	    sep: Optional separator for string keys
	    _prefix: Internal use for recursion

	Returns:
	    Flattened dictionary

	Raises:
	    TypeError: If input is not a dictionary or mapping
	"""
	if not isinstance(xs, (dict, tp.Mapping)):
		raise TypeError(f"Expected dict or Mapping, got {type(xs)}")

	result = {}

	def add_item(path: tuple, value: tp.Any):
		key = TreePath(path, sep).__str__()
		result[key] = value

	def should_flatten(path: tuple, value: tp.Any) -> bool:
		if is_leaf and is_leaf(path, value):
			return False
		return isinstance(value, (dict, tp.Mapping))

	for key, value in xs.items():
		path = _prefix + (key,)
		if should_flatten(path, value):
			nested = flatten_dict(
				value,
				keep_empty_nodes=keep_empty_nodes,
				is_leaf=is_leaf,
				sep=sep,
				_prefix=path,
			)
			if nested or keep_empty_nodes:
				result.update(nested)
		else:
			add_item(path, value)

	return result


def unflatten_dict(xs, sep=None):
	if isinstance(xs, dict):
		return _dict_unflatten_dict(
			xs=xs,
			sep=sep,
		)
	return traversals.unflatten_mapping(
		xs,
		sep=sep,
	)


def nnx_init(
	module: tp.Type[M],
	_add_rngs: bool = True,
	_rng_key: str = "rngs",
	_seed: int = 0,
	_lazy: bool = True,
	**kwargs,
) -> M:
	"""Initializes an nnx module with lazy initialization support.

	This function provides a convenient way to initialize nnx modules while
	handling random number generation and optional lazy initialization.

	Args:
	    module: The nnx module to initialize.
	    _add_rngs: Whether to add a `rngs` attribute to the module's
	        arguments for random number generation. Defaults to True.
	    _rng_key: The key to use for the `rngs` attribute. Defaults to "rngs".
	    _seed: The seed value for random number generation. Defaults to 0.
	    _lazy: Whether to perform lazy initialization. If True, the module's
	        parameters will be initialized lazily when first used. Defaults
	        to True.
	    **kwargs: Additional keyword arguments to pass to the module's
	        constructor.

	Returns:
	    nnx.State: The initialized nnx state.
	"""
	if not _lazy:
		return module(**kwargs, **({_rng_key: nnx.Rngs(_seed)} if _add_rngs else {}))

	return nnx.eval_shape(
		lambda: module(**kwargs, **({_rng_key: nnx.Rngs(_seed)} if _add_rngs else {}))
	)


def create_graphdef(
	module: nnx.Module,
	_add_rngs: bool = True,
	_rng_key: str = "rngs",
	_seed: int = 0,
	**kwargs,
) -> dict:
	"""Creates a graph definition from an nnx module.

	This function initializes the module lazily and extracts the graph
	definition, which represents the structure of the module without any
	parameter values.

	Args:
	    module: The nnx module to create the graph definition from.
	    _add_rngs: Whether to add a `rngs` attribute to the module's
	        arguments for random number generation. Defaults to True.
	    _rng_key: The key to use for the `rngs` attribute. Defaults to "rngs".
	    _seed: The seed value for random number generation. Defaults to 0.
	    **kwargs: Additional keyword arguments to pass to the module's
	        constructor.

	Returns:
	    dict: The graph definition of the module.
	"""
	return nnx.split(
		nnx_init(
			module=module,
			_rng_key=_rng_key,
			_add_rngs=_add_rngs,
			_seed=_seed,
			_lazy=True,
			**kwargs,
		)
	)[0]


def init_garphstate(
	module: nnx.Module,
	_add_rngs: bool = True,
	_rng_key: str = "rngs",
	_seed: int = 0,
	_lazy: bool = True,
	**kwargs,
) -> dict:
	"""Initializes the graph state of an nnx module.

	This function initializes the module and returns the graph state, which
	contains the initialized parameter values and other state information.

	Args:
	    module: The nnx module to initialize.
	    _add_rngs: Whether to add a `rngs` attribute to the module's
	        arguments for random number generation. Defaults to True.
	    _rng_key: The key to use for the `rngs` attribute. Defaults to "rngs".
	    _seed: The seed value for random number generation. Defaults to 0.
	    _lazy: Whether to perform lazy initialization. If True, the module's
	        parameters will be initialized lazily when first used. Defaults
	        to True.
	    **kwargs: Additional keyword arguments to pass to the module's
	        constructor.

	Returns:
	    dict: The initialized graph state of the module.
	"""
	return nnx.split(
		nnx_init(
			module=module,
			_rng_key=_rng_key,
			_add_rngs=_add_rngs,
			_seed=_seed,
			_lazy=_lazy,
			**kwargs,
		)
	)[1]


def validate_state(
	state: tp.Dict[str, tp.Any], init_state: tp.Dict[str, tp.Any]
) -> StateValidationResult:
	"""Validates state against init_state before differentiation."""
	missing_keys = set(init_state.keys()) - set(state.keys())
	invalid_types = {
		k: type(v)
		for k, v in state.items()
		if k in init_state and not isinstance(v, type(init_state[k]))
	}
	return StateValidationResult(
		is_valid=len(missing_keys) == 0 and len(invalid_types) == 0,
		missing_keys=missing_keys,
		invalid_types=invalid_types,
	)


def diffrentiate_state(
	state: tp.Dict[str, tp.Any],
	init_state: tp.Dict[str, tp.Any],
	validate: bool = True,
) -> tp.Dict[str, nnx.VariableState]:
	"""
	Enhanced state differentiation with validation and error handling.

	Args:
	    state: Current state dictionary
	    init_state: Initial state dictionary
	    validate: Whether to perform validation

	Returns:
	    Dictionary of missing attributes

	Raises:
	    ValueError: If validation fails and validate=True
	"""
	if validate:
		validation = validate_state(state, init_state)
		if not validation.is_valid:
			raise ValueError(
				f"State validation failed:\n"
				f"Missing keys: {validation.missing_keys}\n"
				f"Invalid types: {validation.invalid_types}"
			)

	missing_attributes = {}
	for key, value in init_state.items():
		if key not in state:
			if not isinstance(value, nnx.VariableState):
				raise TypeError(f"Value for key {key} must be VariableState, got {type(value)}")
			missing_attributes[key] = value

	return missing_attributes


def redefine_state(state: dict, missings: dict[str, nnx.VariableState]) -> dict:
	"""Redefines missing attributes in a state dictionary.

	This function takes a state dictionary `state` and a dictionary
	`missings` containing missing attributes. It iterates over the
	`missings` dictionary and redefines the missing attributes in the `state`
	dictionary based on their type.

	Args:
	    state: The state dictionary to redefine.
	    missings: A dictionary of missing attributes.

	Returns:
	    dict: The redefined state dictionary.

	Raises:
	    AttributeError: If an unexpected type is encountered in the `missings`
	        dictionary.
	"""
	_miss_count: int = 0
	_state_rngs: jax.random.PRNGKey = jax.random.PRNGKey(42)
	for key, value in missings.items():
		if isinstance(value.type, nnx.Param) or issubclass(value.type, nnx.Param):
			assert (
				value.value is None
			), "there's missing parameter in state which can't be None."
			state[key] = value
		elif isinstance(value.type, nnx.RngCount) or issubclass(value.type, nnx.RngCount):
			state[key] = nnx.VariableState(
				nnx.RngCount,
				jax.numpy.array(_miss_count, dtype=jax.numpy.uint32),
			)
			_miss_count += 1
		elif isinstance(value.type, nnx.RngKey) or issubclass(value.type, nnx.RngKey):
			state[key] = nnx.VariableState(nnx.RngKey, _state_rngs)
			_state_rngs = jax.random.split(_state_rngs)[0]
		else:
			raise AttributeError(
				f"Unexcepted type({value.type}) found which cannot be redefined."
			)
	return state


def is_flatten(tree: dict) -> bool:
	"""Checks if a dictionary represents a flattened tree.

	A flattened tree is a dictionary where the keys are tuples representing
	the path to the leaf nodes. This function checks if any of the keys in the
	input dictionary is a tuple, indicating a flattened tree.

	Args:
	    tree: The dictionary to check.

	Returns:
	    bool: True if the dictionary is a flattened tree, False otherwise.
	"""
	return True in set(isinstance(k, tuple) for k in tree.keys())


def recreate_meta_values(
	values: tp.Dict[str, tp.Any], seed: tp.Optional[int] = None
) -> tp.Dict[str, tp.Any]:
	"""
	Enhanced meta value recreation with better state management.

	Args:
	    values: Dictionary of values to recreate
	    seed: Optional seed for random number generation

	Returns:
	    Dictionary with recreated meta values

	Raises:
	    TypeError: For unexpected value types
	"""
	recreator = MetaValueRecreator(seed or 42)
	input_is_flatten = is_flatten(values)

	if not input_is_flatten:
		values = traversals.flatten_mapping(values)

	try:
		for key, value in values.items():
			if isinstance(value.type, (nnx.RngCount, type)) and issubclass(
				value.type, nnx.RngCount
			):
				values[key].value = recreator.get_count()
			elif isinstance(value.type, (nnx.RngKey, type)) and issubclass(
				value.type, nnx.RngKey
			):
				values[key].value = recreator.get_rng()
			else:
				raise TypeError(f"Unexpected type {value.type} for key {key}")
	except Exception as e:
		raise ValueError(f"Failed to recreate meta values: {str(e)}") from e

	return traversals.unflatten_mapping(values) if not input_is_flatten else values


def refine_graphs(*graphs: dict) -> nnx.State:
	"""Refines and merges multiple graph representations into a single nnx.State.

	This function takes multiple graph representations, which can be either
	dictionaries or nnx.State instances, and merges them into a single
	nnx.State object. It ensures that all inputs are converted to
	nnx.State instances before merging.

	Args:
	    *graphs: The graph representations to merge.

	Returns:
	    nnx.State: The merged nnx.State object.
	"""
	_state_creators = ()
	for graph in graphs:
		if isinstance(graph, nnx.State):
			_state_creators += (graph,)
		else:
			if is_flatten(graph):
				graph = traversals.unflatten_mapping(graph)
			_state_creators += (nnx.State(graph),)
	return nnx.State.merge(*_state_creators)


def merge_state_and_tree(tree: dict, state: nnx.State) -> nnx.State:
	"""
	Attaches a parameter tree to an nnx state.

	This function takes a parameter tree, which is a dictionary containing
	parameter values, and attaches it to an existing nnx state. It first
	splits the nnx state into parameters and other state elements. Then,
	it flattens the parameter tree and the nnx state's parameters for
	easy traversal. For each parameter key in the flattened nnx state,
	if the corresponding value is not None (indicating an existing
	parameter), it replaces the value with the corresponding value from
	the input parameter tree. Finally, it recreates the meta values in
	the "others" part of the state (which includes things like RNG keys
	and counts), and then merges the updated parameters and "others"
	back into a single nnx.State object.

	Args:
	    tree: The parameter tree to attach.
	    state: The nnx state to attach the tree to.

	Returns:
	    nnx.State: The updated nnx state with the attached parameter tree.
	"""
	params, others = nnx.State.split(state, nnx.Param, ...)

	if not is_flatten(params):
		params = flatten_dict(params)
	if not is_flatten(tree):
		tree = flatten_dict(tree)
	# for k, v in tree.items():
	# 	print(k)
	for key in list(params.keys()):
		tree_value = tree.get(key, None)
		if tree_value is not None:
			params[key].value = tree_value
		else:
			if key[-1] == "kernel":
				warnings.warn(
					f"A Params/Kernel Might be missing, please double check ({key}).",
					stacklevel=1,
				)
	others = recreate_meta_values(others)
	state = refine_graphs(others, params)
	return state


def merge_model_and_tree(model: M, tree: dict) -> M:
	graphdef, graphstate = nnx.split(model)
	graphstate = merge_state_and_tree(tree=tree, state=graphstate)
	return nnx.merge(graphdef, graphstate)
