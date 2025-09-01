# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Utility functions for managing and manipulating nnx module states."""

import typing as tp
from collections.abc import Iterable
from copy import deepcopy

import jax
import jax.numpy as jnp
from eformer.loggings import get_logger
from eformer.pytree import auto_pytree
from flax import nnx, struct
from flax import nnx as nn
from flax.nnx import traversals
from jax.interpreters import pxla
from jax.sharding import Mesh, NamedSharding

T = tp.TypeVar("T", bound=nn.Module)
ModulePath = tuple[str, ...]

PyTree = dict
FnDict = dict[tp.Any, tp.Callable[[tp.Any], tp.Any]]
TreeDict = dict[tp.Any, tp.Any]
Path = tuple[tp.Any, ...]


logger = get_logger(__name__)


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


@struct.dataclass
class _EmptyNode:
    pass


@auto_pytree
class StateValidationResult:
    is_valid: bool
    missing_keys: set
    invalid_types: dict[str, type]


empty_node = _EmptyNode()
M = tp.TypeVar("M")


def int_key_to_string(xs):
    flatten = False
    if not is_flatten(xs):
        flatten = True
        xs = flatten_dict(xs)
    for key in list(xs.keys()):
        if not isinstance(key, str):
            xs[tuple([str(k) for k in key])] = xs.pop(key)
    if flatten:
        xs = unflatten_dict(xs)
    return xs


def string_key_to_int(xs):
    flatten = False
    if not is_flatten(xs):
        flatten = True
        xs = flatten_dict(xs)
    for key in list(xs.keys()):
        if not isinstance(key, str):
            new_key = tuple((int(k) if str(k).isdigit() else k) for k in key)
            xs[new_key] = xs.pop(key)
    if flatten:
        xs = unflatten_dict(xs)
    return xs


def _dict_flatten_dict(xs, keep_empty_nodes=False, is_leaf=None, sep=None, fumap=False):
    if not fumap:
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
            path = (*prefix, key)
            result.update(_flatten(value, path))
        if keep_empty_nodes and is_empty:
            if prefix == ():  # when the whole input is empty
                return {}
            return {_key(prefix): empty_node}
        return result

    return _flatten(xs, ())


def is_iterable(obj):
    return isinstance(obj, Iterable)


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
    xs: dict | tp.Mapping,
    keep_empty_nodes: bool = False,
    is_leaf: tp.Callable[[tuple, tp.Any], bool] | None = None,
    sep: str | None = None,
    fumap: bool = False,
) -> dict[tuple | str, tp.Any]:
    """
    Enhanced dictionary flattening with better type handling and validation.

    Args:
        xs: Dictionary or mapping to flatten
        keep_empty_nodes: Whether to keep empty dictionary nodes
        is_leaf: Optional function to determine leaf nodes
        sep: Optional separator for string keys

    Returns:
        Flattened dictionary

    Raises:
        TypeError: If input is not a dictionary or mapping
    """

    if isinstance(xs, dict) or fumap:
        if sep is not None:
            xs = int_key_to_string(xs)
        return _dict_flatten_dict(
            xs=xs,
            keep_empty_nodes=keep_empty_nodes,
            is_leaf=is_leaf,
            sep=sep,
            fumap=fumap,
        )
    return traversals.flatten_mapping(
        xs,
        keep_empty_nodes=keep_empty_nodes,
        is_leaf=is_leaf,
        sep=sep,
    )


def unflatten_dict(xs, sep=None):
    if isinstance(xs, dict):
        return _dict_unflatten_dict(xs=xs, sep=sep)
    return traversals.unflatten_mapping(xs, sep=sep)


def nnx_init(
    module: type[M],
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

    return nnx.eval_shape(lambda: module(**kwargs, **({_rng_key: nnx.Rngs(_seed)} if _add_rngs else {})))


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


def validate_state(state: dict[str, tp.Any], init_state: dict[str, tp.Any]) -> StateValidationResult:
    """Validates state against init_state before differentiation."""
    missing_keys = set(init_state.keys()) - set(state.keys())
    invalid_types = {k: type(v) for k, v in state.items() if k in init_state and not isinstance(v, type(init_state[k]))}
    return StateValidationResult(
        is_valid=len(missing_keys) == 0 and len(invalid_types) == 0,
        missing_keys=missing_keys,
        invalid_types=invalid_types,
    )


def diffrentiate_state(
    state: dict[str, tp.Any],
    init_state: dict[str, tp.Any],
    validate: bool = True,
) -> dict[str, nnx.VariableState]:
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
            assert value.value is None, "there's missing parameter in state which can't be None."
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
            raise AttributeError(f"Unexcepted type({value.type}) found which cannot be redefined.")
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


def recreate_meta_values(values: dict[str, tp.Any], seed: int | None = None) -> dict[str, tp.Any]:
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
            if isinstance(value.type, nnx.RngCount | type) and issubclass(value.type, nnx.RngCount):
                values[key].value = recreator.get_count()
            elif isinstance(value.type, nnx.RngKey | type) and issubclass(value.type, nnx.RngKey):
                values[key].value = recreator.get_rng()
            else:
                raise TypeError(f"Unexpected type {value.type} for key {key}")
    except Exception as e:
        raise ValueError(f"Failed to recreate meta values: {e!s}") from e

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
    tree = string_key_to_int(tree)
    for keys in list(params.keys()):
        tree_values = tree.get(keys, None)
        if tree_values is not None:
            params[keys].value = tree_values
        else:
            if keys[-1] != "bias":
                _path = ".".join([str(k) for k in keys])
                logger.info(f"a parameter's missing at {_path}, please double check.")

            # Avoid type '<class 'jax._src.api.ShapeDtypeStruct'>' is not a valid JAX type
            params[keys].value = None

    others = recreate_meta_values(others)
    state = refine_graphs(others, params)
    return state


def merge_model_and_tree(model: M, tree: dict) -> M:
    """
    Attaches a parameter tree to an nnx model.

    This function takes a parameter tree, which is a dictionary containing
    parameter values, and attaches it to an existing nnx model. It first
    splits the nnx model into parameters and other model elements. Then,
    it flattens the parameter tree and the nnx model's parameters for
    easy traversal. For each parameter key in the flattened nnx model,
    if the corresponding value is not None (indicating an existing
    parameter), it replaces the value with the corresponding value from
    the input parameter tree. Finally, it recreates the meta values in
    the "others" part of the model (which includes things like RNG keys
    and counts), and then merges the updated parameters and "others"
    back into a single nnx.Module object.

    Args:
        tree: The parameter tree to attach.
        model: The nnx model to attach the tree to.

    Returns:
        nnx.Module: The updated nnx model with the attached parameter tree.
    """
    graphdef, graphstate = nnx.split(model)
    graphstate = merge_state_and_tree(tree=tree, state=graphstate)
    return nnx.merge(graphdef, graphstate)


def specs_to_name_sharding(tree: dict, mesh: Mesh | None = None) -> dict:
    """
    Converts a dictionary of specifications to a dictionary of NamedSharding objects.

    Args:
            tree (Dict): A dictionary where the keys are names and the values are specifications.
            mesh (Optional[Mesh]): An optional Mesh object. If not provided, the default physical mesh from
                                                             pxla.thread_resources.env.physical_mesh is used.

    Returns:
            Dict: A dictionary where the keys are the same as the input dictionary, and the values are NamedSharding
                            objects created from the specifications and the provided or default mesh.
    """
    mesh = mesh or pxla.thread_resources.env.physical_mesh
    return jax.tree_util.tree_map(lambda spec: NamedSharding(spec=spec, mesh=mesh), tree)


def tree_apply(fns: FnDict, tree: TreeDict) -> TreeDict:
    """
    Apply a dictionary of functions to a corresponding PyTree.

    Args:
            fns: A dictionary where keys match the PyTree structure and values are functions.
            tree: The PyTree to apply functions to.

    Returns:
            A new PyTree with the same structure as `tree`, but with values modified by the functions in `fns`.
    """
    return jax.tree_util.tree_map(lambda fn, x: fn(x), fns, tree)


def tree_path_to_string(path: Path, sep: str | None = None) -> str:
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
    xs: PyTree,
    is_leaf: tp.Callable[[tp.Any], bool] | None = None,
    sep: str | None = None,
) -> dict[str, tp.Any]:
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
    f: tp.Callable[[str, tp.Any, tp.Any], tp.Any],
    tree: PyTree,
    *rest: tp.Any,
    is_leaf: tp.Callable[[tp.Any], bool] | None = None,
    sep: str | None = None,
) -> PyTree:
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


def deepcopy_model(model):
    """
    Creates a deep copy of a JAX model.

    This function takes a JAX model, extracts its leaves (the individual
    components of the model), deep copies them, and then reconstructs the
    model with the copied leaves.

    Args:
            model: A JAX model to be deep copied. This can be any nested structure
                             of JAX arrays, lists, tuples, dicts, etc.

    Returns:
            A deep copy of the input model with the same structure but with all
            leaves deep copied.
    """
    leaves = deepcopy(jax.tree_util.tree_leaves(model))
    struct = jax.tree_util.tree_structure(model)
    return jax.tree_util.tree_unflatten(struct, leaves)


def recursive_merge(full_tree, updates):
    """
    Recursively merge two PyTrees where updates may have fewer parameters.

    Args:
        full_tree: The complete parameter tree
        updates: Tree with updated values (subset of full_tree)

    Returns:
        Merged tree with updated values where available
    """
    if updates is None:
        return full_tree

    if isinstance(full_tree, dict) and isinstance(updates, dict):
        result = {}
        for key in full_tree:
            if key in updates:
                result[key] = recursive_merge(full_tree[key], updates[key])
            else:
                result[key] = full_tree[key]
        return result
    elif isinstance(full_tree, list | tuple) and isinstance(updates, list | tuple):
        result = []
        for i, item in enumerate(full_tree):
            if i < len(updates):
                result.append(recursive_merge(item, updates[i]))
            else:
                result.append(item)
        return type(full_tree)(result)
    else:
        return updates


def iter_module_search(model: nn.Module, instance: type[T] | None = None) -> tp.Iterator[tuple[ModulePath, T]]:
    """
    Iterates through a model and yields paths and modules of a specific type.

    Args:
        model: The root module to search through.
        instance: The type of module to search for.

    Yields:
        tp.Tuple containing:
            - Path to the module as a tuple of strings/integers
            - The module instance matching the specified type

    Example:
        >>> for path, module in iter_module_search(model, ParallelLinear):
        ...   print(f"Found Linear layer at {path}")
    """
    if instance is None:
        for path, module in nn.graph.iter_graph(model):
            yield path, module
    else:
        for path, module in nn.graph.iter_graph(model):
            if isinstance(module, instance):
                yield path, module


def get_module_from_path(model: nn.Module, path: ModulePath) -> nn.Module | None:
    """
    Retrieves a module from a model given its path.

    Args:
        model: The root module to traverse.
        path: tp.Tuple of strings/integers representing the path to the module.

    Returns:
        The module at the specified path, or None if path is empty.

    Example:
        >>> module = get_module_from_path(model, ("encoder", "layer1", "attention"))
    """
    if not path:
        return None

    current = model
    for item in path:
        current = current[item] if isinstance(item, int) else getattr(current, item)
    return current


def set_module_from_path(model: nn.Module, path: ModulePath, new_value: tp.Any) -> None:
    """
    Sets a module at a specific path in the model.

    Args:
        model: The root module to modify.
        path: tp.Tuple of strings/integers representing the path to the module.
        new_value: The new value/module to set at the specified path.

    Raises:
        AttributeError: If the path is invalid.
        IndexError: If trying to access an invalid index.

    Example:
        >>> new_layer = ParallelLinear(64, 128)
        >>> set_module_from_path(model, ("encoder", "layer1"), new_layer)
    """
    if not path:
        return

    current = model
    # Navigate to the parent of the target location
    for item in path[:-1]:
        current = current[item] if isinstance(item, int) else getattr(current, item)

    # Set the new value at the target location
    last_item = path[-1]
    if isinstance(last_item, int):
        current[last_item] = new_value
    else:
        setattr(current, last_item, new_value)
