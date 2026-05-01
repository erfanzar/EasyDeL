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

from __future__ import annotations

import dataclasses
import types
import typing as tp
from collections.abc import Mapping

T = tp.TypeVar("T", bound="ConfigDataclass")
S = tp.TypeVar("S")

__all__ = (
    "ConfigDataclass",
    "ConfigDict",
    "typed_config",
    "typed_config_dataclass",
)


# ---------------------------------------------------------------------------
# New pattern: one class definition that is BOTH a TypedDict (for ``Unpack``)
# and a runtime container with ``from_dict`` / attribute access.
# ---------------------------------------------------------------------------


class ConfigDict(dict):
    """Runtime instance type for classes decorated with :func:`typed_config`.

    A ``dict`` subclass that also supports attribute access. ``Spec.from_dict(...)``
    returns one of these (a per-spec subclass), so callers can use any of:

        c = MyConfig.from_dict(model=m)
        c.model            # attribute access
        c["model"]         # dict access (type-checked via the TypedDict)
        {**c}              # spreadable
        c.replace(model=n) # immutable update via the spec
        c.to_dict()        # plain nested dict
    """

    __slots__ = ()
    __spec__: tp.ClassVar[type | None] = None

    def __getattr__(self, name: str) -> tp.Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name: str, value: tp.Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def to_dict(self) -> dict[str, tp.Any]:
        return {k: _config_to_plain(v) for k, v in self.items()}

    def as_dict(self) -> dict[str, tp.Any]:
        return self.to_dict()

    def replace(self, **updates: tp.Any) -> "ConfigDict":
        spec = type(self).__spec__
        if spec is not None and hasattr(spec, "from_dict"):
            return spec.from_dict(self, **updates)
        merged = type(self)(self)
        merged.update(updates)
        return merged


PostInit = tp.Callable[[tp.Any], None]


@tp.overload
def typed_config(spec: type[S], /) -> type[S]: ...
@tp.overload
def typed_config(
    *,
    defaults: Mapping[str, tp.Any] | None = ...,
    post_init: PostInit | None = ...,
    namespace: Mapping[str, tp.Any] | None = ...,
) -> tp.Callable[[type[S]], type[S]]: ...


def typed_config(
    spec: type | None = None,
    *,
    defaults: Mapping[str, tp.Any] | None = None,
    post_init: PostInit | None = None,
    namespace: Mapping[str, tp.Any] | None = None,
):
    """Augment a ``TypedDict`` spec with ``from_dict`` / ``to_dict`` / ``replace``.

    The decorated class is **returned unchanged for the type system** — it remains
    a ``TypedDict``, so ``Unpack[Spec]`` keeps working in ``**kwargs`` annotations.
    At runtime, ``Spec.from_dict(**kwargs)`` returns a per-spec :class:`ConfigDict`
    instance (a ``dict`` subclass with attribute access).

    Args:
        spec: The TypedDict class. Pass directly with ``@typed_config``, or omit
            and use ``@typed_config(defaults=..., post_init=...)``.
        defaults: Per-field default values. Mutable defaults (dict/list/set) are
            shallow-copied into each instance.
        post_init: Optional ``(instance) -> None`` validator/normalizer called
            after defaults + values are applied. May raise; may also mutate.
        namespace: Extra attributes to attach to the per-spec instance class
            (rarely needed).

    Example::

        def _validate(self):
            if self.model is None:
                raise ValueError("model is required")

        @typed_config(defaults={"tokenizer": None}, post_init=_validate)
        class PreTrainedLoading(TypedDict, total=False):
            model: Required[Any]
            tokenizer: NotRequired[Any | None]

            @classmethod
            def from_dict(
                cls,
                data: Mapping[str, Any] | None = None,
                **kw: Unpack["PreTrainedLoading"],
            ) -> "PreTrainedLoading": ...
    """

    def wrap(spec: type) -> type:
        defaults_ = dict(defaults or {})
        ns = dict(namespace or {})

        try:
            annotations = tp.get_type_hints(spec, include_extras=True)
        except (NameError, TypeError):
            annotations = dict(getattr(spec, "__annotations__", {}))

        # Compute required/optional from resolved annotations rather than
        # __required_keys__ / __optional_keys__ — those are computed by the
        # TypedDict metaclass at class-creation time, but `from __future__ import
        # annotations` makes the raw annotations strings, so the metaclass can't
        # see Required[]/NotRequired[] wrappers and treats them all as plain.
        total = bool(getattr(spec, "__total__", True))
        required_keys: set[str] = set()
        optional_keys: set[str] = set()
        for name, ann in annotations.items():
            origin = tp.get_origin(ann)
            if origin is tp.Required:
                required_keys.add(name)
            elif origin is tp.NotRequired:
                optional_keys.add(name)
            elif total:
                required_keys.add(name)
            else:
                optional_keys.add(name)
        required_keys -= set(defaults_)
        optional_keys |= set(defaults_)
        all_keys = required_keys | optional_keys

        unwrapped = {name: _strip_required(ann) for name, ann in annotations.items()}

        instance_ns: dict[str, tp.Any] = {
            "__module__": getattr(spec, "__module__", __name__),
            "__qualname__": getattr(spec, "__qualname__", spec.__name__),
            "__spec__": spec,
        }
        for k, v in ns.items():
            instance_ns.setdefault(k, v)

        instance_cls: type[ConfigDict] = type(spec.__name__, (ConfigDict,), instance_ns)

        @classmethod
        def from_dict(cls, data=None, **overrides):
            if isinstance(data, instance_cls):
                if not overrides:
                    return data
                merged = dict(data)
                merged.update(overrides)
                return cls.from_dict(merged)

            if data is None:
                values: dict[str, tp.Any] = {}
            elif isinstance(data, Mapping):
                values = dict(data)
            else:
                raise TypeError(f"{cls.__name__}.from_dict() expected a mapping, got {type(data).__name__}.")
            values.update(overrides)

            unknown = sorted(set(values) - all_keys)
            if unknown:
                raise TypeError(f"{cls.__name__}.from_dict() got unknown field(s): {', '.join(unknown)}.")

            instance = instance_cls()
            for k, v in defaults_.items():
                instance[k] = _copy_default(v)
            instance.update(values)

            missing = sorted(required_keys - set(instance))
            if missing:
                raise TypeError(f"{cls.__name__}.from_dict() missing required field(s): {', '.join(missing)}.")

            for k in list(instance.keys()):
                ann = unwrapped.get(k)
                instance[k] = _coerce_value(ann, instance[k])

            if post_init is not None:
                post_init(instance)

            return instance

        @classmethod
        def coerce_config(cls, value=None):
            """Coerce ``value`` (None / Mapping / instance) to a ``cls`` instance.

            Idempotent: passing an existing instance of this spec returns it
            unchanged. ``None`` builds with defaults (errors if required fields
            are missing). A mapping is fed through :meth:`from_dict`.
            """
            return cls.from_dict(value)

        spec.from_dict = from_dict
        spec.coerce_config = coerce_config
        spec.__typed_config_instance_cls__ = instance_cls
        spec.__typed_config_defaults__ = types.MappingProxyType(defaults_)
        return spec

    if spec is not None and isinstance(spec, type):
        return wrap(spec)
    return wrap


def _strip_required(annotation: tp.Any) -> tp.Any:
    origin = tp.get_origin(annotation)
    if origin in (tp.Required, tp.NotRequired):
        return tp.get_args(annotation)[0]
    return annotation


def _copy_default(value: tp.Any) -> tp.Any:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return list(value)
    if isinstance(value, set):
        return set(value)
    return value


def _coerce_value(annotation: tp.Any, value: tp.Any) -> tp.Any:
    if annotation is None or value is None:
        return value

    origin = tp.get_origin(annotation)
    args = tp.get_args(annotation)

    if origin in (tp.Union, types.UnionType):
        for arg in args:
            if arg is type(None):
                continue
            coerced = _try_coerce(arg, value)
            if coerced is not _COERCE_FAIL:
                return coerced
        return value

    coerced = _try_coerce(annotation, value)
    if coerced is not _COERCE_FAIL:
        return coerced
    return value


_COERCE_FAIL = object()


def _try_coerce(annotation: tp.Any, value: tp.Any) -> tp.Any:
    origin = tp.get_origin(annotation)
    args = tp.get_args(annotation)

    target = _config_target(annotation)
    if target is not None:
        if isinstance(value, getattr(target, "__typed_config_instance_cls__", ())):
            return value
        if isinstance(value, ConfigDataclass) and isinstance(value, target):
            return value
        if isinstance(value, Mapping):
            return target.from_dict(value)
        return _COERCE_FAIL

    if origin is tuple and isinstance(value, (list, tuple)):
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce_value(args[0], item) for item in value)
        if args and len(args) == len(value):
            return tuple(_coerce_value(arg, item) for arg, item in zip(args, value, strict=True))
        return tuple(value)

    if origin is list and isinstance(value, list):
        item_ann = args[0] if args else None
        return [_coerce_value(item_ann, item) for item in value]

    if origin is dict and isinstance(value, dict):
        key_ann = args[0] if args else None
        val_ann = args[1] if len(args) > 1 else None
        return {_coerce_value(key_ann, k): _coerce_value(val_ann, v) for k, v in value.items()}

    return _COERCE_FAIL


def _config_target(annotation: tp.Any) -> type | None:
    """Return the config class to coerce a Mapping into, or None if N/A."""
    if isinstance(annotation, type):
        if hasattr(annotation, "__typed_config_instance_cls__"):
            return annotation
        if issubclass(annotation, ConfigDataclass):
            return annotation
    return None


def _config_to_plain(value: tp.Any) -> tp.Any:
    if isinstance(value, ConfigDict):
        return value.to_dict()
    if isinstance(value, ConfigDataclass):
        return value.to_dict()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _config_to_plain(getattr(value, field.name)) for field in dataclasses.fields(value) if field.init
        }
    if isinstance(value, tuple):
        return tuple(_config_to_plain(item) for item in value)
    if isinstance(value, list):
        return [_config_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {_config_to_plain(k): _config_to_plain(v) for k, v in value.items()}
    return value


class ConfigDataclass:
    """Runtime mixin for config classes whose schema is written as a ``TypedDict``.

    The intended pattern is:

    ``class MyConfig(TypedDict, total=False): ...``
    ``if not TYPE_CHECKING: MyConfig = typed_config_dataclass(MyConfig, ...)``

    That keeps the field schema single-source: type checkers see a ``TypedDict``
    usable with ``Unpack[MyConfig]``, while runtime gets a dataclass object with
    validation and stable dict round-tripping.
    """

    def to_dict(self) -> dict[str, tp.Any]:
        if not dataclasses.is_dataclass(self):
            raise TypeError(f"{type(self).__name__} must be a dataclass to use to_dict().")
        return {
            field.name: _config_to_plain(getattr(self, field.name)) for field in dataclasses.fields(self) if field.init
        }

    def as_dict(self) -> dict[str, tp.Any]:
        return self.to_dict()

    @classmethod
    def coerce_config(cls: type[T], value: Mapping[str, tp.Any] | None = None) -> T:
        """Coerce ``value`` (None / Mapping / instance) into a ``cls`` instance."""
        return cls.from_dict(value)

    @classmethod
    def from_dict(cls: type[T], data: Mapping[str, tp.Any] | None = None, **overrides: tp.Any) -> T:
        if isinstance(data, cls):
            return data
        if data is None:
            values: dict[str, tp.Any] = {}
        elif isinstance(data, Mapping):
            values = dict(data)
        else:
            raise TypeError(f"{cls.__name__}.from_dict() expected a mapping, got {type(data).__name__}.")

        values.update(overrides)
        hints = tp.get_type_hints(cls)
        field_names = {field.name for field in dataclasses.fields(cls) if field.init}
        unknown = tuple(sorted(set(values) - field_names))
        if unknown:
            joined = ", ".join(unknown)
            raise TypeError(f"{cls.__name__}.from_dict() got unknown field(s): {joined}.")

        kwargs: dict[str, tp.Any] = {}
        for field in dataclasses.fields(cls):
            if not field.init or field.name not in values:
                continue
            kwargs[field.name] = _coerce_value(hints.get(field.name), values[field.name])
        return cls(**kwargs)

    def replace(self: T, **updates: tp.Any) -> T:
        return dataclasses.replace(self, **updates)


def typed_config_dataclass(
    spec: type,
    *,
    defaults: Mapping[str, tp.Any] | None = None,
    post_init: PostInit | None = None,
    namespace: Mapping[str, tp.Any] | None = None,
) -> type[ConfigDataclass]:
    """Build a runtime dataclass from a ``TypedDict`` spec (legacy)."""

    defaults = dict(defaults or {})
    try:
        annotations = tp.get_type_hints(spec, include_extras=True)
    except (NameError, TypeError):
        annotations = dict(getattr(spec, "__annotations__", {}))

    required_fields: list[tuple[tp.Any, ...]] = []
    optional_fields: list[tuple[tp.Any, ...]] = []
    for name, annotation in annotations.items():
        required, inner_annotation = _unwrap_typeddict_requiredness(annotation)
        if name in defaults:
            item = (name, inner_annotation, _default_field(defaults[name]))
        elif required:
            item = (name, inner_annotation)
        else:
            item = (name, inner_annotation, dataclasses.field(default=None))

        if required and name not in defaults:
            required_fields.append(item)
        else:
            optional_fields.append(item)

    class_namespace: dict[str, tp.Any] = dict(namespace or {})
    if post_init is not None:

        def __post_init__(self):
            post_init(self)

        class_namespace["__post_init__"] = __post_init__

    return dataclasses.make_dataclass(
        cls_name=spec.__name__,
        fields=required_fields + optional_fields,
        bases=(ConfigDataclass,),
        namespace=class_namespace,
        module=getattr(spec, "__module__", None),
    )


def _unwrap_typeddict_requiredness(annotation: tp.Any) -> tuple[bool, tp.Any]:
    origin = tp.get_origin(annotation)
    args = tp.get_args(annotation)
    if origin is tp.Required:
        return True, args[0]
    if origin is tp.NotRequired:
        return False, args[0]
    return True, annotation


def _default_field(value: tp.Any) -> dataclasses.Field:
    if isinstance(value, (dict, list, set)):
        return dataclasses.field(default_factory=lambda value=value: value.copy())
    return dataclasses.field(default=value)
