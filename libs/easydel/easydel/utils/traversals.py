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

"""SpectraX module navigation and backward-compatible PyTree utilities.

Generic dictionary/path utilities are re-exported from eformer so checkpoint
and model callers share one implementation without changing their imports.
"""

import typing as tp
from collections.abc import Generator
from copy import deepcopy

import jax
import spectrax as spx
from eformer.loggings import get_logger
from eformer.pytree import flatten_dict as flatten_dict
from eformer.pytree import flatten_tree as flatten_tree
from eformer.pytree import int_key_to_string as int_key_to_string
from eformer.pytree import is_flatten as is_flatten
from eformer.pytree import is_iterable as is_iterable
from eformer.pytree import named_tree_map as named_tree_map
from eformer.pytree import recursive_merge as recursive_merge
from eformer.pytree import string_key_to_int as string_key_to_int
from eformer.pytree import tree_apply as tree_apply
from eformer.pytree import tree_path_to_string as tree_path_to_string
from eformer.pytree import unflatten_dict as unflatten_dict
from eformer.pytree._tree_util import MetaValueRecreator as MetaValueRecreator
from eformer.pytree._tree_util import StateValidationResult as StateValidationResult
from eformer.pytree._tree_util import empty_node as empty_node

from easydel.infra.sharding import MeshLike, specs_to_named_sharding

T = tp.TypeVar("T", bound=spx.Module)
M = tp.TypeVar("M")
ModulePath = tuple[str, ...]
PyTree = dict
FnDict = dict[tp.Any, tp.Callable[[tp.Any], tp.Any]]
TreeDict = dict[tp.Any, tp.Any]
Path = tuple[tp.Any, ...]
logger = get_logger(__name__)


def recreate_meta_values(values: spx.State | dict, seed: int | None = None) -> spx.State | dict:
    """Return SpectraX state unchanged; RNGs are stored separately.

    Args:
        values: State or dictionary (returned unchanged).
        seed: Ignored; kept for API compatibility.

    Returns:
        The input values unchanged.
    """
    return values


def merge_model_and_tree(model: M, tree: dict, *, silence: bool = False) -> M:
    """Attach parameter values from a tree to a new SpectraX model instance.

    Args:
        model: The SpectraX model to attach the tree to.
        tree: The parameter tree to attach, with optional collection prefixes.
        silence: Suppress missing-parameter warnings.

    Returns:
        The updated SpectraX model with the attached parameter tree.
    """
    gdef, state = spx.export(model)
    if not is_flatten(tree):
        tree = flatten_dict(tree)
    tree = string_key_to_int(tree)

    new_data: dict[str, dict[str, tp.Any]] = {}
    for c, p, v in state.items():
        new_data.setdefault(c, {})[p] = v

    for keys, value in tree.items():
        if not keys:
            continue
        c = keys[0]
        path_str = ".".join(str(k) for k in keys[1:])
        full_path = ".".join(str(k) for k in keys)
        placed = False
        if path_str in new_data.get(c, {}):
            new_data[c][path_str] = value
            placed = True
        if not placed:
            for coll in new_data:
                if full_path in new_data[coll]:
                    new_data[coll][full_path] = value
                    placed = True
                    break
        if not placed and not silence:
            logger.info(f"a parameter's missing at {c}/{path_str}, please double check.")

    bound = spx.bind(gdef, spx.State(new_data))
    # spx.bind does not restore _spx_opaque; retain transparent Opaque unwrapping.
    object.__setattr__(bound, "_spx_opaque", dict(model._spx_opaque))
    for opaque_name in model._spx_attr_order:
        if opaque_name not in bound._spx_attr_order:
            bound._spx_attr_order.append(opaque_name)
    return tp.cast(M, bound)


def specs_to_name_sharding(tree: dict, mesh: MeshLike | None = None) -> dict:
    """Convert PartitionSpecs to NamedShardings using EasyDeL mesh handling.

    Args:
        tree: PyTree whose leaves are PartitionSpec instances.
        mesh: Optional mesh, defaulting to the active EasyDeL mesh.

    Returns:
        A PyTree with NamedSharding leaves.
    """
    return specs_to_named_sharding(tree, mesh)


def deepcopy_model(model):
    """Deep-copy a registered model through its PyTree leaves.

    Args:
        model: PyTree-shaped object to clone.

    Returns:
        A deep copy with identical structure and independent leaf storage.
    """
    leaves = deepcopy(jax.tree_util.tree_leaves(model))
    struct = jax.tree_util.tree_structure(model)
    return jax.tree_util.tree_unflatten(struct, leaves)


def iter_module_search(model: spx.Module, instance: type[T] | None = None) -> Generator[tuple[tp.Any, T], None, None]:
    """Yield modules with tuple-form navigation paths, excluding RNG containers.

    Args:
        model: Root module to search.
        instance: Concrete class to filter by; None selects all modules.

    Yields:
        Pairs of path tuples and matching modules.
    """
    _skip_types = (spx.Rngs,)
    if instance is None:
        for path_str, module in spx.iter_modules(model):
            if isinstance(module, _skip_types):
                continue
            yield tuple(path_str.split(".")), tp.cast(T, module)
    else:
        for path_str, module in spx.iter_modules(model, select=instance):
            if isinstance(module, _skip_types):
                continue
            yield tuple(path_str.split(".")), tp.cast(T, module)


def get_module_from_path(model: spx.Module, path: ModulePath) -> spx.Module | None:
    """Retrieve a submodule by walking attribute and container-index segments.

    Args:
        model: Root module to traverse.
        path: Tuple of path segments to walk.

    Returns:
        The module at path, or None when path is empty.

    Raises:
        AttributeError: If an attribute path is invalid.
        IndexError: If a container index is out of range.
    """
    if not path:
        return None
    current = model
    for item in path:
        if isinstance(item, int):
            current = current[item]
        else:
            try:
                current = getattr(current, item)
            except AttributeError:
                try:
                    current = current[int(item)]
                except (ValueError, IndexError, TypeError):
                    raise
    return current


def set_module_from_path(model: spx.Module, path: ModulePath, new_value: tp.Any) -> None:
    """Install a replacement at an attribute/container path; empty paths do nothing.

    Args:
        model: Root module to mutate.
        path: Tuple of path segments identifying the slot to overwrite.
        new_value: Replacement module or value.

    Raises:
        AttributeError: If an attribute path is invalid.
        IndexError: If a container index is out of range.
    """
    if not path:
        return
    current = model
    for item in path[:-1]:
        if isinstance(item, int):
            current = current[item]
        else:
            try:
                current = getattr(current, item)
            except AttributeError:
                try:
                    current = current[int(item)]
                except (ValueError, IndexError, TypeError):
                    raise
    last_item = path[-1]
    if isinstance(last_item, int):
        current[last_item] = new_value
    else:
        try:
            setattr(current, last_item, new_value)
        except (AttributeError, TypeError):
            try:
                current[int(last_item)] = new_value
            except (ValueError, IndexError, TypeError):
                raise
