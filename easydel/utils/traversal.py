"""Utility functions for managing and manipulating nnx module states."""

from typing import Type, TypeVar

import jax
from flax import nnx

M = TypeVar("M", nnx.Module)


def nnx_init(
	module: Type[M],
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


def diffrentiate_state(state: dict, init_state: dict) -> dict:
	"""Differentiates two state dictionaries and returns the differences.

	This function compares two state dictionaries, `state` and `init_state`,
	and returns a new dictionary containing only the keys and values that are
	present in `init_state` but not in `state`. Only `nnx.VariableState` types
	are considered for restoration.

	Args:
	    state: The current state dictionary.
	    init_state: The initial state dictionary.

	Returns:
	    dict: A dictionary containing the missing attributes from `init_state`.
	"""
	missing_attributes = {}
	restored_keys = list(state.keys())
	for key in init_state.keys():
		if key not in restored_keys:
			assert isinstance(
				init_state[key], nnx.VariableState
			), "only VariableState types are restoreable"
			missing_attributes[key] = init_state[key]
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


def recreate_meta_values(values: dict) -> dict:
	"""Recreates meta values in a dictionary.

	This function iterates over the input dictionary and recreates the values
	for specific types: `nnx.RngCount` and `nnx.RngKey`. For `nnx.RngCount`,
	it assigns a unique integer count. For `nnx.RngKey`, it generates a new
	random key.

	Args:
	    values: The dictionary containing the values to recreate.

	Returns:
	    dict: The dictionary with recreated meta values.

	Raises:
	    AttributeError: If an unexpected type is encountered in the input
	        dictionary.
	"""
	input_is_flatten = True
	if not is_flatten(values):
		input_is_flatten = False
		values = nnx.traversals.flatten_mapping(values)
	_miss_count: int = 0
	_state_rngs: jax.random.PRNGKey = jax.random.PRNGKey(42)
	for key, value in values.items():
		if isinstance(value.type, nnx.RngCount) or issubclass(value.type, nnx.RngCount):
			values[key].value = jax.numpy.array(_miss_count, dtype=jax.numpy.uint32)
			_miss_count += 1
		elif isinstance(value.type, nnx.RngKey) or issubclass(value.type, nnx.RngKey):
			values[key].value = _state_rngs
			_state_rngs = jax.random.split(_state_rngs)[0]
		else:
			raise AttributeError(
				f"Unexcepted type({value.type}) found which cannot be redefined."
			)
	if not input_is_flatten:
		values = nnx.traversals.unflatten_mapping(values)
	return values


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
				graph = nnx.traversals.unflatten_mapping(graph)
			_state_creators += (nnx.State(graph),)
	return nnx.State.merge(*_state_creators)


def attach_tree_to_nnx_state(tree: dict, state: nnx.State) -> nnx.State:
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
	params = nnx.traversals.flatten_mapping(params)
	if not is_flatten(tree):
		tree = nnx.traversals.flatten_mapping(tree)
	for key in params.keys():
		if params[key].value is not None:
			params[key].value = tree[key]
	others = recreate_meta_values(others)
	state = refine_graphs(others, params)
	return state


def attech_tree_to_nnx_model(model: M, tree: dict) -> M:
	graphdef, graphstate = nnx.split(model)
	graphstate = attach_tree_to_nnx_state(tree=tree, state=graphstate)
	return nnx.merge(graphdef, graphstate)
