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


from __future__ import annotations

import dataclasses
import json
import types
import typing as tp
from functools import lru_cache, wraps

import typing_extensions
from jax import tree_util as tu

T = tp.TypeVar("T")
FnDict = dict[tp.Any, tp.Callable[[tp.Any], tp.Any]]
TreeDict = dict[tp.Any, tp.Any]
Path = tuple[tp.Any, ...]
FilterSpec = bool | tp.Callable[[tp.Any], bool]
IsLeafFn = tp.Callable[[tp.Any], bool]


@lru_cache(maxsize=1024)
def _is_non_jax_type(typ: type) -> bool:
    """
    Checks if a given type is considered a non-JAX type.

    Non-JAX types are typically those that should not be treated as leaves
    in a PyTree structure by default, such as strings, functions, or types.

    Args:
        typ: The type to check.

    Returns:
        True if the type is considered a non-JAX type, False otherwise.
    """
    NON_JAX_TYPES = (
        str,
        bytes,
        types.FunctionType,
        types.MethodType,
        type,
        tp.Callable,
    )

    if typ is tp.Any:
        return False

    origin = tp.get_origin(typ)
    if origin is tp.Union:  # type:ignore
        args = tp.get_args(typ)
        return any(_is_non_jax_type(arg) for arg in args)

    for non_jax_type in NON_JAX_TYPES:
        try:
            if issubclass(typ, non_jax_type):
                return True
        except TypeError:
            pass

    return False


def field(pytree_node: bool = True, *, metadata: dict | None = None, **kwargs) -> dataclasses.Field:
    """
    A dataclass field replacement that allows specifying whether a field
    should be treated as a PyTree node.

    Args:
        pytree_node: If True (default), the field is treated as a PyTree leaf/node.
                     If False, the field is treated as metadata.
        metadata: Optional dictionary of metadata to pass to `dataclasses.field`.
        **kwargs: Additional keyword arguments passed to `dataclasses.field`.

    Returns:
        A dataclasses.Field object.
    """
    metadata_dict = (metadata or {}).copy()
    metadata_dict["pytree_node"] = pytree_node
    return dataclasses.field(metadata=metadata_dict, **kwargs)


class PyTreeClassInfo:
    """Stores metadata about a class registered as a PyTree."""

    __slots__ = ["data_fields", "frozen", "meta_fields", "type_hints"]

    def __init__(
        self,
        data_fields: tuple[str, ...],
        meta_fields: tuple[str, ...],
        frozen: bool,
        type_hints: dict[str, type],
    ):
        """
        Initializes the PyTreeClassInfo.

        Args:
            data_fields: Tuple of field names treated as PyTree data (children).
            meta_fields: Tuple of field names treated as PyTree metadata.
            frozen: Boolean indicating if the original dataclass was frozen.
            type_hints: Dictionary mapping field names to their type hints.
        """
        self.data_fields = data_fields
        self.meta_fields = meta_fields
        self.frozen = frozen
        self.type_hints = type_hints


_CLASS_INFO_REGISTRY: dict[type, PyTreeClassInfo] = {}


@typing_extensions.dataclass_transform(field_specifiers=(field,))
def auto_pytree(
    cls: type[T] | None = None,
    meta_fields: tuple[str, ...] | None = None,
    json_serializable: bool = True,
    frozen: bool = False,
    max_print_length: int = 500,
):
    """
    A class decorator that automatically registers a dataclass as a JAX PyTree.

    It uses `dataclasses.dataclass` to make the class a dataclass if it isn't already,
    determines which fields are data (PyTree children) and which are metadata,
    and registers the class with `jax.tree_util.register_dataclass`.

    Fields are considered metadata if:
    - They are explicitly listed in `meta_fields`.
    - They are marked with `field(pytree_node=False)`.
    - Their type hint suggests they are non-JAX types (checked by `_is_non_jax_type`).

    Args:
        cls: The class to be decorated.
        meta_fields: A tuple of field names to always treat as metadata.
        json_serializable: If True (default), adds `to_dict`, `from_dict`, `to_json`,
                           and `from_json` methods to the class.
        frozen: If True, makes the dataclass frozen (immutable). Defaults to False.

    Returns:
        The decorated class, registered as a PyTree.
    """

    def wrap(cls_inner: type[T]) -> type[T]:
        """Internal wrapper function for the decorator."""
        cls_inner = dataclasses.dataclass(cls_inner, frozen=frozen)

        fields = [f for f in dataclasses.fields(cls_inner) if f.init]
        all_field_names = tuple(f.name for f in fields)

        final_meta_fields: set[str] = set(meta_fields or ())

        for field_obj in fields:
            field_metadata = field_obj.metadata
            if field_metadata and "pytree_node" in field_metadata:
                if field_metadata["pytree_node"] is False:
                    final_meta_fields.add(field_obj.name)
                elif field_metadata["pytree_node"] is True and field_obj.name in final_meta_fields:
                    final_meta_fields.remove(field_obj.name)

        type_hints = tp.get_type_hints(cls_inner)

        for field_obj in fields:
            if field_obj.name in final_meta_fields:
                continue

            if field_obj.metadata and field_obj.metadata.get("pytree_node") is True:
                continue

            field_type = type_hints.get(field_obj.name)
            if field_type is not None and _is_non_jax_type(field_type):
                final_meta_fields.add(field_obj.name)

        data_fields = tuple(f for f in all_field_names if f not in final_meta_fields)
        meta_fields_tuple = tuple(final_meta_fields)

        def _replace(self, **kwargs):
            """Creates a new instance with specified fields replaced."""
            if not kwargs:
                return self
            return dataclasses.replace(self, **kwargs)

        cls_inner.replace = _replace

        def enhanced_repr(self):
            """Provides a more detailed representation of the object."""
            cls_name = self.__class__.__name__
            items = []

            for k, v in self.__dict__.items():
                if not k.startswith("_"):
                    try:
                        repr_str = str(v).replace("\n", "\n  ")
                        if len(repr_str) > max_print_length:
                            repr_str = f"{v.__class__.__name__}(...)"
                        items.append(f"  {k} : {repr_str}")
                    except TypeError:
                        items.append(f"  {k} : <unrepresentable>")

            return f"{cls_name}(\n" + "\n".join(items) + "\n)"

        cls_inner.__repr__ = enhanced_repr
        cls_inner.__str__ = enhanced_repr

        class_info = PyTreeClassInfo(
            data_fields=data_fields,
            meta_fields=meta_fields_tuple,
            frozen=frozen,
            type_hints=type_hints,
        )
        _CLASS_INFO_REGISTRY[cls_inner] = class_info

        cls_inner.__pytree_meta__ = {
            "data_fields": data_fields,
            "meta_fields": meta_fields_tuple,
            "frozen": frozen,
        }

        if json_serializable:

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

            cls_inner.to_dict = to_dict

            @classmethod
            def from_dict(cls_inner_classmethod: type[T], data: dict[str, tp.Any]) -> T:
                """Deserializes a dictionary into a PyTree object."""
                processed_data = {}

                class_info_local = _CLASS_INFO_REGISTRY.get(cls_inner_classmethod)
                type_hints_local = (
                    class_info_local.type_hints if class_info_local else tp.get_type_hints(cls_inner_classmethod)
                )

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

            cls_inner.from_dict = from_dict

            def to_json(self, **kwargs) -> str:
                """Serializes the PyTree object to a JSON string."""
                return json.dumps(self.to_dict(), **kwargs)

            cls_inner.to_json = to_json

            @classmethod
            def from_json(cls_inner_classmethod: type[T], json_str: str) -> T:
                """Deserializes a JSON string into a PyTree object."""
                data = json.loads(json_str)
                return cls_inner_classmethod.from_dict(data)

            cls_inner.from_json = from_json

            if not hasattr(json.JSONEncoder, "_pytree_patched"):
                original_default = json.JSONEncoder.default

                @wraps(original_default)
                def json_default(encoder_self, obj):
                    """JSON encoder default method patched to handle PyTrees."""
                    if hasattr(obj, "to_dict") and callable(obj.to_dict):
                        return obj.to_dict()
                    return original_default(encoder_self, obj)

                json.JSONEncoder.default = json_default
                json.JSONEncoder._pytree_patched = True

        return tu.register_dataclass(
            cls_inner,
            data_fields=data_fields,
            meta_fields=meta_fields_tuple,
        )

    if cls is None:
        return wrap
    return wrap(cls)


class _PyTreeNodeBase:
    """Base class providing a default `replace` method."""

    def replace(self: T, **kwargs) -> T:
        """Creates a new instance with specified fields replaced.

        This method is typically overridden by the `auto_pytree` decorator
        if the class is decorated directly. It serves as a fallback or
        base for classes inheriting from PyTree/FrozenPyTree.

        Args:
            **kwargs: Field names and their new values.

        Returns:
            A new instance of the class with the specified fields updated.
        """
        return dataclasses.replace(self, **kwargs)


@typing_extensions.dataclass_transform(field_specifiers=(field,))
class PyTree(_PyTreeNodeBase):
    """
    Base class for mutable PyTree dataclasses.

    Inheriting from this class automatically applies the `auto_pytree`
    decorator to the subclass, registering it as a JAX PyTree.
    """

    def __init_subclass__(
        cls,
        *,
        frozen: bool = False,
        json_serializable: bool = True,
        meta_fields: tuple[str, ...] | None = None,
        **kwargs,
    ):
        """
        Applies `auto_pytree` to subclasses.

        Args:
            frozen: If True, makes the dataclass frozen. Defaults to False.
            json_serializable: If True (default), adds JSON serialization methods.
            meta_fields: Tuple of field names to always treat as metadata.
            **kwargs: Additional arguments passed to `auto_pytree`.
        """
        super().__init_subclass__(**kwargs)
        auto_pytree(
            cls,
            meta_fields=meta_fields,
            json_serializable=json_serializable,
            frozen=frozen,
        )


@typing_extensions.dataclass_transform(field_specifiers=(field,))
class FrozenPyTree(_PyTreeNodeBase):
    """
    Base class for immutable (frozen) PyTree dataclasses.

    Inheriting from this class automatically applies the `auto_pytree`
    decorator with `frozen=True` to the subclass, registering it as a
    frozen JAX PyTree.
    """

    def __init_subclass__(
        cls,
        *,
        json_serializable: bool = True,
        meta_fields: tuple[str, ...] | None = None,
        **kwargs,
    ):
        """
        Applies `auto_pytree` with frozen=True to subclasses.

        Args:
            json_serializable: If True (default), adds JSON serialization methods.
            meta_fields: Tuple of field names to always treat as metadata.
            **kwargs: Additional arguments passed to `auto_pytree`.
        """
        super().__init_subclass__(**kwargs)
        auto_pytree(
            cls,
            meta_fields=meta_fields,
            json_serializable=json_serializable,
            frozen=True,
        )
