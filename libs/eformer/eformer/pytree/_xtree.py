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


import dataclasses
import functools
import json
import threading
import types
import typing as tp
from collections.abc import Callable

import jax
from contextlib2 import contextmanager
from typing_extensions import dataclass_transform

_T = tp.TypeVar("_T")
T = tp.TypeVar("__T")

_PRIMITIVE_TYPES = (
    str,
    bytes,
    types.FunctionType,
    types.MethodType,
    type,
    tp.Callable,
)
"""Tuple of types considered as primitive (non-PyTree node) types.

These types are treated as leaves in PyTree operations and are not
recursively traversed. They are typically non-array data that should
be preserved as-is during tree operations.
"""

_STATE_DICT_REGISTRY: dict[tp.Any, tp.Any] = {}


class _NamedTuple:
    """Sentinel type marker for namedtuple serialization registration.

    This class serves as a key in the state dict registry to handle
    namedtuple types uniformly. Since namedtuples are dynamically created
    and don't share a common base class, this marker allows registering
    a single pair of serialization functions for all namedtuple types.
    """


def _is_namedtuple(x: tp.Any) -> bool:
    """
    Duck typing test for namedtuple factory-generated objects.

    Args:
            x: The object to test.

    Returns:
            True if the object appears to be a namedtuple, False otherwise.
    """
    return isinstance(x, tuple) and hasattr(x, "_fields")


class _ErrorContext(threading.local):
    """Thread-local context for tracking the path during deserialization.

    This class maintains a thread-safe stack of path components that tracks
    the current location within a nested data structure during deserialization.
    This enables meaningful error messages that indicate exactly where in the
    structure a problem occurred.

    Attributes:
        path: List of string path components representing the current location
            in the nested structure being processed.

    Examples:
        >>> ctx = _ErrorContext()
        >>> ctx.path.append("layer1")
        >>> ctx.path.append("weights")
        >>> "/".join(ctx.path)
        'layer1/weights'
    """

    def __init__(self):
        """Initialize the error context with an empty path stack."""
        self.path = []


def register_serialization_state(
    ty: tp.Any,
    ty_to_state_dict: Callable[[tp.Any], dict[str, tp.Any]],
    ty_from_state_dict: Callable[[tp.Any, dict[str, tp.Any]], tp.Any],
    override: bool = False,
):
    """
    Registers serialization and deserialization functions for a given type.

    Args:
            ty: The type to register handlers for.
            ty_to_state_dict: A callable that converts an instance of `ty` to a state dictionary.
            ty_from_state_dict: A callable that updates an instance of `ty` from a state dictionary.
            override: If True, overrides an existing registration for the type.
                              If False and a registration exists, raises a ValueError.

    Raises:
            ValueError: If a handler for the type is already registered and `override` is False.
    """
    if ty in _STATE_DICT_REGISTRY and not override:
        raise ValueError(f'a serialization handler for "{ty.__name__}" is already registered')
    _STATE_DICT_REGISTRY[ty] = (ty_to_state_dict, ty_from_state_dict)


_error_context = _ErrorContext()


@contextmanager
def _record_path(name: str):
    """
    Context manager to record the current path component during deserialization.

    Args:
            name: The name of the current path component (e.g., a field name).

    Yields:
            None. The context manager pushes the name onto the path on entry
            and pops it on exit.
    """
    try:
        _error_context.path.append(name)
        yield
    finally:
        _error_context.path.pop()


def xfrom_state_dict(target: _T, state: dict[str, tp.Any], name: str = ".") -> _T:
    """
    Recursively deserializes the state dictionary into the target object.

    Uses the registered `from_state_dict` function for the target's type if available,
    otherwise returns the state directly.

    Args:
            target: The object to deserialize into.
            state: The state dictionary.
            name: The name of the current object in the parent structure (used for error reporting).

    Returns:
            The deserialized object.
    """
    if _is_namedtuple(target):
        ty = _NamedTuple
    else:
        ty = type(target)
    if ty not in _STATE_DICT_REGISTRY:
        return state
    ty_from_state_dict = _STATE_DICT_REGISTRY[ty][1]
    with _record_path(name):
        return ty_from_state_dict(target, state)


def xto_state_dict(target: tp.Any) -> dict[str, tp.Any]:
    """
    Recursively converts the target object into a state dictionary.

    Uses the registered `to_state_dict` function for the target's type if available,
    otherwise returns the target directly.

    Args:
            target: The object to serialize.

    Returns:
            A dictionary representing the state of the target object, or the target itself
            if no serialization handler is registered.
    """
    if _is_namedtuple(target):
        ty = _NamedTuple
    else:
        ty = type(target)
    if ty not in _STATE_DICT_REGISTRY:
        return target

    ty_to_state_dict = _STATE_DICT_REGISTRY[ty][0]
    state_dict = ty_to_state_dict(target)
    if isinstance(state_dict, dict):
        for key in state_dict.keys():
            if not isinstance(key, str):
                raise TypeError("A state dict must only have string keys.")
    return state_dict


def _is_pytree_node_annotation(annotation: tp.Any) -> bool:
    """
    Determines whether a type annotation should be treated as a JAX PyTree node.

    Primitive types and simple containers of primitives are considered leaves.
    More complex types, custom classes, and containers of non-primitives are
    considered nodes.

    Args:
            annotation: The type annotation to check.

    Returns:
            True if the annotation indicates a PyTree node, False if it indicates a leaf.
    """
    origin = tp.get_origin(annotation)
    args = tp.get_args(annotation)

    if annotation in _PRIMITIVE_TYPES:
        return False

    if origin is tp.Union:
        return any(_is_pytree_node_annotation(arg) for arg in args if arg is not type(None))

    if origin in (list, tuple, set, frozenset):
        return not all(arg in _PRIMITIVE_TYPES for arg in args)

    return True


def field(*, pytree_node: bool | None = None, metadata: dict | None = None, **kwargs):
    """
    Define a dataclass field and optionally mark it explicitly as a PyTree node.

    This function is a wrapper around `dataclasses.field` that adds a `pytree_node`
    option to the metadata.

    Args:
            pytree_node: Explicitly mark the field as a PyTree node (True) or leaf (False).
                                     If None, the type annotation will be used to infer behavior.
            metadata: A dictionary of metadata for the field. The `pytree_node` key
                              will be added or updated in this dictionary.
            **kwargs: Additional keyword arguments passed to `dataclasses.field`.

    Returns:
            A `dataclasses.Field` object.
    """
    md = dict(metadata or {})
    if pytree_node is not None:
        md["pytree_node"] = pytree_node
    return dataclasses.field(metadata=md, **kwargs)


@dataclass_transform(field_specifiers=(field,))
@tp.overload
def dataclass(clz: _T, **kwargs) -> _T:
    """
    Overload for `dataclass` when used as a decorator with a class argument.
    """
    ...


@dataclass_transform(field_specifiers=(field,))
@tp.overload
def dataclass(**kwargs) -> Callable[[_T], _T]:
    """
    Overload for `dataclass` when used as a decorator factory with keyword arguments.
    """
    ...


@dataclass_transform(field_specifiers=(field,))
def dataclass(clz: _T | None = None, **kwargs) -> _T | Callable[[_T], _T]:
    """
    A decorator that enhances standard dataclasses to be JAX PyTree compatible
    and adds serialization/deserialization capabilities.

    It automatically registers the dataclass with `jax.tree_util` and defines
    `to_state_dict` and `from_state_dict` methods based on the field types
    and explicit `pytree_node` markings.

    Args:
            clz: The class to decorate.
            **kwargs: Additional keyword arguments passed to `dataclasses.dataclass`.
                              Defaults to `frozen=True`.

    Returns:
            The decorated class.
    """
    if clz is None:
        return functools.partial(dataclass, **kwargs)
    if getattr(clz, "_eformer_dataclass", False):
        return clz

    kwargs.setdefault("frozen", True)
    data_clz = dataclasses.dataclass(**kwargs)(clz)
    data_fields: list[str] = []
    meta_fields: list[str] = []

    annotations = getattr(data_clz, "__annotations__", {})
    for field_info in dataclasses.fields(data_clz):
        if "pytree_node" in field_info.metadata:
            is_node = field_info.metadata["pytree_node"]
        else:
            ann = annotations.get(field_info.name, tp.Any)
            is_node = _is_pytree_node_annotation(ann)
        (data_fields if is_node else meta_fields).append(field_info.name)

    def replace(self, **updates):
        """
        Returns a new instance of the dataclass with specified fields updated.

        Args:
                **updates: Keyword arguments where keys are field names and values
                                   are the new values for those fields.

        Returns:
                A new instance of the dataclass with the updated fields.
        """
        return dataclasses.replace(self, **updates)

    data_clz.replace = replace
    jax.tree_util.register_dataclass(data_clz, data_fields, meta_fields)

    def to_state_dict(x):
        """
        Converts the dataclass instance to a state dictionary.

        Args:
                x: The dataclass instance.

        Returns:
                A dictionary containing the state of the dataclass instance.
        """
        return {name: xto_state_dict(getattr(x, name)) for name in data_fields}

    def from_state_dict(x, state):
        """
        Updates the dataclass instance from a state dictionary.

        Args:
                x: The dataclass instance to update.
                state: The state dictionary.

        Returns:
                A new dataclass instance with the state loaded from the dictionary.

        Raises:
                ValueError: If a required field is missing in the state dictionary
                                        or if unknown fields are present.
        """
        state = state.copy()
        updates = {}
        for name in data_fields:
            if name not in state:
                raise ValueError(f"Missing field {name} in state dict for {clz.__name__}")
            value = getattr(x, name)
            updates[name] = xfrom_state_dict(value, state.pop(name), name=name)
        if state:
            raise ValueError(f"Unknown field(s) {list(state.keys())} in state dict for {clz.__name__}")
        return x.replace(**updates)

    register_serialization_state(data_clz, to_state_dict, from_state_dict)

    setattr(data_clz, "_eformer_dataclass", True)  # noqa

    return data_clz


def _list_to_state_dict(target: list) -> list:
    """Convert a list to its state dictionary representation.

    Args:
        target: The list to serialize.

    Returns:
        list: List of serialized elements.
    """
    return [xto_state_dict(item) for item in target]


def _list_from_state_dict(target: list | None, state: list) -> list:
    """Restore a list from its state dictionary representation.

    Args:
        target: Optional target list (unused, included for API consistency).
        state: List of serialized element states.

    Returns:
        list: Deserialized list.
    """
    return [xfrom_state_dict(None, item_state, name=f"[{i}]") for i, item_state in enumerate(state)]


def _tuple_to_state_dict(target: tuple) -> list:
    """Convert a tuple to its state dictionary representation.

    Args:
        target: The tuple to serialize.

    Returns:
        list: List of serialized elements (tuples are stored as lists in JSON).
    """
    return [xto_state_dict(item) for item in target]


def _tuple_from_state_dict(target: tuple | None, state: list) -> tuple:
    """Restore a tuple from its state dictionary representation.

    Args:
        target: Optional target tuple for type information.
        state: List of serialized element states.

    Returns:
        tuple: Deserialized tuple.
    """
    elems = []
    target_elems = target if target and len(target) == len(state) else [None] * len(state)
    for i, item_state in enumerate(state):
        elems.append(xfrom_state_dict(target_elems[i], item_state, name=f"[{i}]"))
    return tuple(elems)


def _set_to_state_dict(target: set) -> list:
    """Convert a set to its state dictionary representation.

    Attempts to sort elements for deterministic serialization; falls back
    to arbitrary order if elements are not comparable.

    Args:
        target: The set to serialize.

    Returns:
        list: List of serialized elements.
    """
    try:
        sorted_items = sorted(list(target))
        return [xto_state_dict(item) for item in sorted_items]
    except TypeError:
        return [xto_state_dict(item) for item in target]


def _set_from_state_dict(target: set | None, state: list) -> set:
    """Restore a set from its state dictionary representation.

    Args:
        target: Optional target set (unused, included for API consistency).
        state: List of serialized element states.

    Returns:
        set: Deserialized set.
    """
    return {xfrom_state_dict(None, item_state, name=f"{{{i}}}") for i, item_state in enumerate(state)}


def _frozenset_to_state_dict(target: frozenset) -> list:
    """Convert a frozenset to its state dictionary representation.

    Attempts to sort elements for deterministic serialization; falls back
    to arbitrary order if elements are not comparable.

    Args:
        target: The frozenset to serialize.

    Returns:
        list: List of serialized elements.
    """
    try:
        sorted_items = sorted(list(target))
        return [xto_state_dict(item) for item in sorted_items]
    except TypeError:
        return [xto_state_dict(item) for item in target]


def _frozenset_from_state_dict(target: frozenset | None, state: list) -> frozenset:
    """Restore a frozenset from its state dictionary representation.

    Args:
        target: Optional target frozenset (unused, included for API consistency).
        state: List of serialized element states.

    Returns:
        frozenset: Deserialized frozenset.
    """
    return frozenset(xfrom_state_dict(None, item_state, name=f"f{{{i}}}") for i, item_state in enumerate(state))


def _dict_to_state_dict(target: dict) -> dict[str, tp.Any]:
    """Convert a dictionary to its state dictionary representation.

    Converts all keys to strings for JSON compatibility. Supports int, float,
    bool, None, and string keys.

    Args:
        target: The dictionary to serialize.

    Returns:
        dict[str, Any]: Dictionary with string keys and serialized values.

    Raises:
        TypeError: If a key cannot be converted to a JSON-compatible string.
    """
    new_dict = {}
    for k, v in target.items():
        if not isinstance(k, str):
            if isinstance(k, int | float | bool) or k is None:
                key_str = str(k)
            else:
                raise TypeError(
                    f"Dictionary key {k!r} of type {type(k).__name__} is not serializable to a JSON string key."
                )
        else:
            key_str = k
        new_dict[key_str] = xto_state_dict(v)
    return new_dict


def _dict_from_state_dict(target: dict | None, state: dict[str, tp.Any]) -> dict:
    """Restore a dictionary from its state dictionary representation.

    Args:
        target: Optional target dictionary for type hints on values.
        state: Dictionary with string keys and serialized values.

    Returns:
        dict: Deserialized dictionary.
    """
    new_dict = {}
    target_dict = target if target else {}
    for k, v_state in state.items():
        v_target = target_dict.get(k)
        new_dict[k] = xfrom_state_dict(v_target, v_state, name=k)
    return new_dict


def _namedtuple_to_state_dict(target: tuple) -> dict[str, tp.Any]:
    """Convert a namedtuple to its state dictionary representation.

    Uses _asdict() if available, otherwise manually extracts field values.

    Args:
        target: The namedtuple instance to serialize.

    Returns:
        dict[str, Any]: Dictionary mapping field names to serialized values.
    """
    if hasattr(target, "_asdict"):
        data = target._asdict()
    else:
        data = {field: getattr(target, field) for field in target._fields}
    return {k: xto_state_dict(v) for k, v in data.items()}


def _namedtuple_from_state_dict(target: tuple | None, state: dict[str, tp.Any]) -> tuple:
    """Restore a namedtuple from its state dictionary representation.

    Args:
        target: A prototype namedtuple instance providing type information.
        state: Dictionary mapping field names to serialized values.

    Returns:
        tuple: A new namedtuple instance of the same type as target.

    Raises:
        TypeError: If target is None (namedtuples require a prototype).
        ValueError: If required fields are missing from the state dict.
        TypeError: If namedtuple reconstruction fails.
    """
    if target is None:
        raise TypeError("Cannot deserialize a namedtuple without a target prototype instance.")
    target_type = type(target)

    missing_fields = set(target._fields) - set(state.keys())
    if missing_fields:
        path_str = ".".join(_error_context.path) or "."
        raise ValueError(
            f"Missing field(s) {missing_fields} in state dict for namedtuple {target_type.__name__} at path '{path_str}'"
        )

    deserialized_values = {}
    for field_name in target._fields:
        field_state = state[field_name]
        field_target_value = getattr(target, field_name)
        with _record_path(field_name):
            deserialized_values[field_name] = xfrom_state_dict(
                field_target_value,
                field_state,
                name=field_name,
            )
    try:
        return target_type(**deserialized_values)
    except Exception as e:
        path_str = ".".join(_error_context.path) or "."
        raise TypeError(f"Failed to recreate namedtuple {target_type.__name__} at path '{path_str}': {e}") from e


def _jax_array_to_state_dict(target: jax.Array) -> list:
    """Convert a JAX array to its state dictionary representation.

    Converts the array to a nested Python list for JSON serialization.

    Args:
        target: The JAX array to serialize.

    Returns:
        list: Nested list representation of the array.
    """
    return target.tolist()


def _jax_array_from_state_dict(target: jax.Array | None, state: list) -> jax.Array:
    """Restore a JAX array from its state dictionary representation.

    Args:
        target: Optional target array providing dtype information.
        state: Nested list representation of the array.

    Returns:
        jax.Array: Reconstructed JAX array.

    Raises:
        TypeError: If array reconstruction fails.
    """
    dtype = target.dtype if target is not None else None
    try:
        arr = jax.numpy.array(state, dtype=dtype)
        return arr
    except Exception as e:
        path_str = ".".join(_error_context.path) or "."
        raise TypeError(f"Failed to convert state back to jax.Array at path '{path_str}' (dtype: {dtype}): {e}") from e


register_serialization_state(
    list,
    _list_to_state_dict,
    _list_from_state_dict,
)
register_serialization_state(
    tuple,
    _tuple_to_state_dict,
    _tuple_from_state_dict,
)
register_serialization_state(
    set,
    _set_to_state_dict,
    _set_from_state_dict,
)
register_serialization_state(
    frozenset,
    _frozenset_to_state_dict,
    _frozenset_from_state_dict,
)
register_serialization_state(
    dict,
    _dict_to_state_dict,
    _dict_from_state_dict,
)
register_serialization_state(
    _NamedTuple,
    _namedtuple_to_state_dict,
    _namedtuple_from_state_dict,
)
register_serialization_state(
    jax.Array,
    _jax_array_to_state_dict,
    _jax_array_from_state_dict,
)
register_serialization_state(
    tp.Sequence,
    _list_to_state_dict,
    _list_from_state_dict,
    override=True,
)
register_serialization_state(
    tp.Mapping,
    _dict_to_state_dict,
    _dict_from_state_dict,
    override=True,
)
register_serialization_state(
    set,
    _set_to_state_dict,
    _set_from_state_dict,
    override=True,
)

STree = tp.TypeVar("STree", bound="xTree")


@dataclass_transform(field_specifiers=(field,))
class xTree:
    """
    Base class for dataclasses acting as JAX PyTree nodes with built-in
    serialization support.

    Classes inheriting from `xTree` are automatically processed by the
    `dataclass` decorator upon definition, making them JAX PyTree compatible
    and adding `to_state_dict` and `from_state_dict` methods.
    """

    def __init_subclass__(cls, **kwargs):
        """
        Automatically applies the `dataclass` decorator to subclasses.
        """
        dataclass(cls, **kwargs)

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def replace(self: STree, **overrides) -> STree:
        """
        Returns a new instance of the xTree subclass with specified fields updated.

        This method is added dynamically by the `dataclass` decorator.

        Args:
                **overrides: Keyword arguments where keys are field names and values
                                         are the new values for those fields.

        Returns:
                A new instance of the xTree subclass with the updated fields.
        """
        raise NotImplementedError

    def to_state_dict(self) -> tp.Any:
        """Serializes this instance to a JSON-compatible state."""
        return xto_state_dict(self)

    @classmethod
    def from_state_dict(cls: type[STree], state: tp.Any) -> STree:
        """Deserializes state into an instance of this class."""
        return xfrom_state_dict(None, state, name=cls.__name__)

    def to_dict(self) -> dict[str, tp.Any]:
        """Serializes the PyTree object to a dictionary."""
        result = {}
        for field_obj in dataclasses.fields(self):
            value = getattr(self, field_obj.name)
            if value is Ellipsis:
                continue
            if isinstance(value, tuple):
                result[field_obj.name] = list(value)
            elif value is None:
                result[field_obj.name] = None
            elif hasattr(value, "to_dict") and callable(value.to_dict):
                result[field_obj.name] = value.to_dict()
            else:
                try:
                    json.dumps(value)
                    result[field_obj.name] = value
                except (TypeError, OverflowError):
                    result[field_obj.name] = str(value)
        return result

    @classmethod
    def from_dict(cls_inner_classmethod: type[T], data: dict[str, tp.Any]) -> T:
        """Deserializes a dictionary into a PyTree object."""
        processed_data = {}

        type_hints_local = tp.get_type_hints(cls_inner_classmethod)

        for field_obj in dataclasses.fields(cls_inner_classmethod):
            field_name = field_obj.name
            if field_name not in data:
                continue
            value = data[field_name]
            field_type = type_hints_local.get(field_name)

            if (
                value is not None
                and isinstance(value, list)
                and field_type is not None
                and tp.get_origin(field_type) is tuple
            ):
                processed_data[field_name] = tuple(value)
            else:
                processed_data[field_name] = value

        return cls_inner_classmethod(**processed_data)
