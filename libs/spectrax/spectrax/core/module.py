# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The :class:`Module` base class.

:class:`Module` is the PyTorch-shaped base class every user module
subclasses. It is deliberately *not* registered as a JAX pytree — state
lives externally in :class:`~spectrax.State` and modules own
:class:`~spectrax.Variable` cells that are the sole mutable leaves.

Attribute discipline: an attribute assigned on a module must be one of

* another :class:`Module` (including a container);
* a :class:`~spectrax.Variable`
  (:class:`~spectrax.Parameter`, :class:`~spectrax.Buffer`, …);
* a :class:`~spectrax.Static` marker or an immutable hashable scalar
  (contributes to :class:`~spectrax.GraphDef`);
* an :class:`Opaque` escape hatch;
* or a name starting with ``"_"`` (implementation detail; not
  graph-visible).

Attribute assignment order is recorded and becomes the stable path
naming scheme used throughout spectrax.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import threading
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, fields, is_dataclass
from typing import TYPE_CHECKING, ClassVar, Self

import jax
import jax.numpy as jnp

from ._typing import ForwardHook, ForwardPreHook
from .context import scope as _scope
from .errors import LazyInitUnderTransformError
from .lazy_init import _allow_materialization, _explicit_lazy_mode, _materialization_allowed
from .policy import Policy, push_policy
from .static import Static, is_static_scalar
from .variable import DeferredBuffer, DeferredParameter, Variable

if TYPE_CHECKING:
    from ..rng.rngs import Rngs


__all__ = ["Module", "Opaque"]


_TRANSFORM_FLAG: threading.local = threading.local()


def _inside_transform() -> bool:
    """Return ``True`` iff a spectrax transform is currently active in this thread.

    Returns:
        Return ``True`` iff a spectrax transform is currently active in this thread.
    """
    return bool(getattr(_TRANSFORM_FLAG, "active", False))


def _set_inside_transform(active: bool) -> None:
    """Set the thread-local "inside a spectrax transform" flag.

    Called by the split/merge shim on entry and exit of every transform.

    Args:
        active: Active value consumed by this operation.
    """
    _TRANSFORM_FLAG.active = active


_GRAPH_EPOCH: int = 0
"""Monotonic global counter bumped whenever any Module's structural
shape changes (a Module/Variable attribute is added, replaced, or
deleted anywhere in the program).

Used by :mod:`spectrax.core.graph` to short-circuit repeat :func:`export`
calls on the dispatch hot path: a Module caches its last
``(epoch, gdef, var_paths)`` snapshot, and when the global epoch is
unchanged the cache is still valid (the graph shape has not been
touched). Value mutations on :class:`~spectrax.Variable` (``.value = …``
or ``_raw_set``) do **not** bump the epoch — they change leaf contents,
not shape — so normal training loops keep the cache hot.
"""


def _bump_graph_epoch() -> None:
    """Invalidate every :func:`spectrax.export` cache across the program.

    Called from :meth:`Module.__setattr__` and :meth:`Module.__delattr__`
    whenever a graph-visible attribute is added, replaced, or removed.
    Only the global counter is mutated; stale cache entries are detected
    lazily on the next :func:`export` call.
    """
    global _GRAPH_EPOCH
    _GRAPH_EPOCH += 1


def _graph_epoch() -> int:
    """Return the current global graph-structure epoch.

    Returns:
        Return the current global graph-structure epoch.
    """
    return _GRAPH_EPOCH


def _public_value(value: object) -> object:
    """Return the value users should see for wrapper-backed attributes.

    Args:
        value: Value consumed by the helper.

    Returns:
        Return the value users should see for wrapper-backed attributes.
    """
    if isinstance(value, Opaque | Static):
        return value.value
    return value


def _opaque_hash_payload(value: object) -> object:
    """Build a best-effort stable payload for an opaque Python object.

    Args:
        value: Value consumed by the helper.

    Returns:
        Result described by this helper.
    """
    if isinstance(value, Opaque):
        return {"opaque": _opaque_hash_payload(value.value)}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return {
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "to_dict": value.to_dict(),
            }
        except Exception:
            pass
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return {
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "model_dump": value.model_dump(),
            }
        except Exception:
            pass
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


def _normalize_hash_payload(value: object, *, _seen: set[int] | None = None) -> object:
    """Normalize arbitrary metadata into a deterministic JSON-like tree.

    Arrays are represented by shape and dtype only, never by values.

    Args:
        value: Value consumed by the helper.
        _seen:  seen value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    if _seen is None:
        _seen = set()

    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return {"float": repr(value)}
    if isinstance(value, complex):
        return {"complex": [repr(value.real), repr(value.imag)]}
    if isinstance(value, bytes):
        return {"bytes": value.hex()}

    if isinstance(value, Static):
        return {"Static": _normalize_hash_payload(value.value, _seen=_seen)}
    if isinstance(value, Opaque):
        return {"Opaque": _normalize_hash_payload(_opaque_hash_payload(value.value), _seen=_seen)}

    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None or dtype is not None:
        return {
            "array": {
                "shape": tuple(int(dim) for dim in tuple(shape or ())),
                "dtype": str(dtype),
            }
        }

    value_id = id(value)
    if isinstance(value, dict | list | tuple | set | frozenset) or is_dataclass(value):
        if value_id in _seen:
            return {"cycle": f"{type(value).__module__}.{type(value).__qualname__}"}
        _seen.add(value_id)
        try:
            if isinstance(value, dict):
                return {
                    "dict": [
                        [
                            _normalize_hash_payload(k, _seen=_seen),
                            _normalize_hash_payload(v, _seen=_seen),
                        ]
                        for k, v in sorted(value.items(), key=lambda item: repr(item[0]))
                    ]
                }
            if isinstance(value, list | tuple):
                return {
                    type(value).__name__: [_normalize_hash_payload(v, _seen=_seen) for v in value],
                }
            if isinstance(value, set | frozenset):
                return {
                    type(value).__name__: sorted(
                        (_normalize_hash_payload(v, _seen=_seen) for v in value),
                        key=repr,
                    )
                }
            if is_dataclass(value):
                return {
                    "dataclass": f"{type(value).__module__}.{type(value).__qualname__}",
                    "fields": {
                        f.name: _normalize_hash_payload(getattr(value, f.name), _seen=_seen) for f in fields(value)
                    },
                }
        finally:
            _seen.discard(value_id)

    try:
        json.dumps(value)
    except TypeError:
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "repr": repr(value),
        }
    return value


def _digest_payload(payload: object) -> str:
    """Return a stable hex digest for a normalized payload.

    Args:
        payload: Payload value consumed by this operation.

    Returns:
        Return a stable hex digest for a normalized payload.
    """
    normalized = _normalize_hash_payload(payload)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


class Opaque:
    """Escape hatch for attributes that are neither Modules nor Variables.

    Wrap any Python object in ``Opaque`` to assign it as a module
    attribute without triggering :meth:`Module.__setattr__`'s strictness
    check. Opaque values are invisible to graph traversal — they do not
    show up in :func:`~spectrax.iter_variables` or
    :func:`~spectrax.iter_modules` and are not pytree leaves — but they
    *do* round-trip through :func:`~spectrax.export` /
    :func:`~spectrax.bind` because :class:`Module` records them
    separately in :attr:`Module._spx_opaque`.

    Most user code never instantiates :class:`Opaque` directly:
    :meth:`Module.__setattr__` automatically wraps any value that
    isn't a Module / Variable / static scalar into an :class:`Opaque`,
    which is how attributes such as ``self.config = config_object``
    Just Work.

    Attributes:
        value: The wrapped Python object, exposed unchanged when the
            owning module reads back the attribute.
    """

    __slots__ = ("value",)

    value: object

    def __init__(self, value: object) -> None:
        """Wrap ``value`` as an opaque attribute.

        Args:
            value: object Python object. No coercion or copy is performed.
        """
        self.value = value

    def __repr__(self) -> str:
        """Return ``Opaque(<class name of value>)`` for diagnostics.

        Returns:
            Return ``Opaque(<class name of value>)`` for diagnostics.
        """
        return f"Opaque({type(self.value).__name__})"


class _HookHandle:
    """Lightweight handle returned by ``register_forward_*`` and
    :meth:`Module.register_context`.

    Hold the handle to be able to :meth:`remove` the hook later. The
    handle keeps a strong reference to the module-side hook list it was
    appended to, so removing a hook does not require knowing the
    owning module.
    """

    __slots__ = ("_fn", "_list")

    _list: list[object]
    _fn: object

    def __init__(self, lst: list[object], fn: object) -> None:
        """Record the destination list and the registered callable.

        Args:
            lst: The hook list on the owning module
                (e.g. ``module._spx_pre_hooks``).
            fn: The callable that was appended to ``lst``.
        """
        self._list = lst
        self._fn = fn

    def remove(self) -> None:
        """Detach the hook from its containing list.

        Idempotent: if the hook was already removed (or the list has
        been mutated externally so the entry no longer exists) the
        :class:`ValueError` raised by ``list.remove`` is swallowed.

        Returns:
            ``None``.
        """
        with contextlib.suppress(ValueError):
            self._list.remove(self._fn)


_HOOK_WARNING_ONCE: set[int] = set()


def _is_context_manager(value: object) -> bool:
    """Return ``True`` when ``value`` implements the context-manager protocol.

    Duck-typed: any object exposing both ``__enter__`` and ``__exit__``
    is accepted, matching the looseness of :keyword:`with`.

    Args:
        value: Value consumed by the helper.

    Returns:
        Return ``True`` when ``value`` implements the context-manager protocol.
    """
    return hasattr(value, "__enter__") and hasattr(value, "__exit__")


def _context_factory(value: object) -> object:
    """Normalize a registered context or zero-arg context factory.

    Reusable context-manager objects (such as a single
    :class:`jax.sharding.Mesh`) are wrapped in an identity factory so
    they can be re-entered for every call. Plain callables are
    returned unchanged on the assumption they construct a fresh
    context manager on every call.

    Args:
        value: Either a context manager or a zero-argument callable
            producing one.

    Returns:
        A zero-argument callable suitable for
        :meth:`contextlib.ExitStack.enter_context` invocation.

    Raises:
        TypeError: If ``value`` is neither a context manager nor a
            callable.
    """
    if _is_context_manager(value):
        return lambda value=value: value
    if callable(value):
        return value
    raise TypeError(
        "Module.register_context() positional arguments must be context "
        "managers or zero-argument factories returning context managers."
    )


class Module:
    """PyTorch-shaped module base class.

    Subclasses override :meth:`__init__` to declare children and
    parameters, and :meth:`forward` to define the computation.

    Modules are **JAX pytrees**: their leaves are the arrays stored in
    descendant :class:`~spectrax.Variable` cells, flattened via
    :func:`~spectrax.export` and reconstituted via :func:`~spectrax.bind`
    on unflatten. That means :func:`jax.jit`, :func:`jax.tree.map`,
    :func:`jax.value_and_grad`, and every other JAX pytree consumer
    accept modules directly.

    **Mutation gotcha.** Inside a *plain* :func:`jax.jit`, mutations
    via ``var.value = new`` happen on a freshly-unflattened module
    inside the trace and are **silently dropped** — the outer live
    module sees no change. Use :func:`~spectrax.jit` with the
    ``mutable=`` selector (or the other spectrax transforms) when you
    want mutations to survive the transform boundary: those wrappers
    run :func:`~spectrax.core.graph.export`/``bind`` with
    mutation-detection around the traced body and write changes back
    through the :class:`~spectrax.Variable` write hook.

    Attributes:
        _spx_attr_order: Ordered list of public attribute names in
            declaration order. Becomes the stable graph naming scheme.
        _spx_static: Dict of static (graph-def-contributing) attribute
            values.
        _spx_training: Training-mode flag, toggled by :meth:`train` /
            :meth:`eval`, propagated recursively to children.
        _spx_fwd_hooks: List of post-hooks invoked after
            :meth:`forward` completes.
        _spx_pre_hooks: List of pre-hooks invoked before
            :meth:`forward`.
        _spx_contexts: List of call contexts entered around every
            :meth:`forward` invocation.
        _spx_policy: Optional dtype :class:`~spectrax.Policy` governing
            this subtree.
        _spx_opaque: Dict recording :class:`Opaque` attributes.
    """

    _spx_container_kind: ClassVar[str] = "module"
    """Graph-def container classification.

    Overridden by containers (:class:`Sequential` -> ``"sequential"``,
    :class:`ModuleList` / :class:`ParameterList` -> ``"list"``,
    :class:`ModuleDict` and :class:`~spectrax.Rngs` -> ``"dict"``) so
    :func:`~spectrax.bind` knows how to reconstruct their children.
    """

    _SPX_RESERVED: ClassVar[tuple[str, ...]] = (
        "_spx_attr_order",
        "_spx_static",
        "_spx_training",
        "_spx_fwd_hooks",
        "_spx_pre_hooks",
        "_spx_contexts",
        "_spx_policy",
        "_spx_opaque",
        "_spx_scan_plan_cache",
    )
    """Private slot names reserved by the :class:`Module` machinery."""

    _spx_attr_order: list[str]
    _spx_static: dict[str, object]
    _spx_training: bool
    _spx_fwd_hooks: list[ForwardHook]
    _spx_pre_hooks: list[ForwardPreHook]
    _spx_contexts: list[object]
    _spx_policy: Policy | None
    _spx_opaque: dict[str, Opaque]
    _spx_scan_plan_cache: object
    _spx_scan_safe_static_fields: ClassVar[frozenset[str] | tuple[str, ...]] = frozenset()
    _spx_scan_safe_opaque_fields: ClassVar[frozenset[str] | tuple[str, ...]] = frozenset()
    """Private scan metadata.

    Subclasses may list static/opaque field names whose per-layer values are
    metadata-only for repeated-layer scans. Differing unlisted values are
    treated as behavior-changing and form separate scan graph families.
    """

    def __new__(cls, *args: object, **kwargs: object) -> Self:
        """Allocate the instance and pre-initialise private slots.

        Slots are created here (rather than in ``__init__``) so that
        subclasses which assign attributes *before* calling
        ``super().__init__()`` do not crash in ``__setattr__``.

        Args:
            *args: Additional positional arguments forwarded to the wrapped callable or backend.
            **kwargs: Additional keyword arguments forwarded to the wrapped callable or backend.

        Returns:
            Result described by this helper.
        """
        instance = super().__new__(cls)
        object.__setattr__(instance, "_spx_attr_order", [])
        object.__setattr__(instance, "_spx_static", {})
        object.__setattr__(instance, "_spx_training", True)
        object.__setattr__(instance, "_spx_fwd_hooks", [])
        object.__setattr__(instance, "_spx_pre_hooks", [])
        object.__setattr__(instance, "_spx_contexts", [])
        object.__setattr__(instance, "_spx_policy", None)
        object.__setattr__(instance, "_spx_opaque", {})
        object.__setattr__(instance, "_spx_export_cache", None)
        object.__setattr__(instance, "_spx_scan_plan_cache", None)
        return instance

    def __init__(self) -> None:
        """No-op — all state is initialised in :meth:`__new__`.

        Subclasses may assign attributes before or after calling
        ``super().__init__()``.
        """

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Register every :class:`Module` subclass as a JAX pytree.

        JAX's :func:`jax.tree_util.register_pytree_with_keys` takes a
        specific class; pytree registration does *not* inherit. To keep
        the user-facing ergonomics simple (every user-defined
        ``class MyModel(spx.Module)`` Just Works with
        :func:`jax.jit` / :func:`jax.tree.map` / …), we auto-register
        each subclass at class-definition time. Safe to call
        repeatedly — JAX raises if a class is re-registered, so we
        guard with a sentinel attribute.

        Args:
            **kwargs: Additional keyword arguments forwarded to the wrapped callable or backend.
        """
        super().__init_subclass__(**kwargs)
        _register_module_pytree(cls)

    def __setattr__(self, name: str, value: object) -> None:
        """Enforce module attribute discipline.

        Names starting with ``_spx_`` bypass all checks (implementation
        details). Single-underscore runtime attributes are preserved as
        opaque runtime metadata across :func:`spectrax.export` /
        :func:`spectrax.bind`. The name ``policy`` is special and funnels
        into the private slot :attr:`_spx_policy`. Otherwise ``value`` must be a
        :class:`Module`, a :class:`~spectrax.Variable`, a static scalar,
        or an :class:`Opaque`.

        One convenience makes the ergonomics friendlier:

        **Annotation-driven static.** If the class declares
        ``name: spx.Static[T]``, the value is automatically wrapped in
        :class:`Static` and stored in :attr:`_spx_static`.

        Everything else falls back to :class:`Opaque`. This keeps
        ergonomic attributes such as ``self.config = config`` available
        on the module without accidentally treating large mutable config
        objects as graph-def static fields.

        Args:
            name: Name used for lookup, logging, or registration.
            value: Value consumed by the helper.
        """
        if name.startswith("_spx_"):
            object.__setattr__(self, name, value)
            return

        if name.startswith("_"):
            object.__setattr__(self, name, value)
            if hasattr(self, "_spx_opaque"):
                if isinstance(value, Module | Variable):
                    self._spx_opaque.pop(name, None)
                else:
                    self._spx_opaque[name] = value if isinstance(value, Opaque) else Opaque(value)
                object.__setattr__(self, "_spx_export_cache", None)
                _bump_graph_epoch()
            return

        if name == "policy":
            if value is not None and not isinstance(value, Policy):
                raise TypeError("`.policy` must be a spectrax.Policy or None")
            object.__setattr__(self, "_spx_policy", value)
            return

        is_module = isinstance(value, Module)
        is_variable = isinstance(value, Variable)
        is_opaque = isinstance(value, Opaque)
        is_static = is_static_scalar(value)

        if not (is_module or is_variable or is_opaque or is_static):
            annotations = getattr(type(self), "__annotations__", {})
            ann = annotations.get(name)
            if ann is not None:
                origin = getattr(ann, "__origin__", None)
                if origin is Static or (isinstance(ann, type) and issubclass(ann, Static)):
                    value = Static(value) if not isinstance(value, Static) else value
                    is_static = True

            if not is_static:
                value = Opaque(value)
                is_opaque = True

        attr_order = self._spx_attr_order
        if name not in attr_order:
            attr_order.append(name)

        _bump_graph_epoch()

        if is_static and not (is_module or is_variable or is_opaque):
            self._spx_static[name] = value
            object.__setattr__(self, name, _public_value(value))
            return

        if is_opaque:
            self._spx_opaque[name] = value
            object.__setattr__(self, name, _public_value(value))
            return

        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        """Delete an attribute and remove it from the graph order/static dicts.

        Args:
            name: Name used for lookup, logging, or registration.
        """
        if name in self._spx_attr_order:
            self._spx_attr_order.remove(name)
        self._spx_static.pop(name, None)
        self._spx_opaque.pop(name, None)
        _bump_graph_epoch()
        object.__delattr__(self, name)

    def _spx_graph_children(self) -> Iterator[tuple[str | int, Module | Variable]]:
        """Yield ``(key, child)`` pairs for Modules/Variables in declaration order.

        The base implementation iterates :attr:`_spx_attr_order` and
        yields attribute-name keys. Containers override this method to
        yield integer or string keys reflecting their native addressing.

        Returns:
            Result described by this helper.
        """
        for name in self._spx_attr_order:
            value = getattr(self, name, None)
            if isinstance(value, Module | Variable):
                yield name, value

    def _spx_static_fields(self) -> dict[str, object]:
        """Return a shallow copy of :attr:`_spx_static` for graph-def export.

        Returns:
            Return a shallow copy of :attr:`_spx_static` for graph-def export.
        """
        return dict(self._spx_static)

    def structure_hash(self) -> str:
        """Return a stable hash of this module's non-value structure.

        The digest includes the exported :class:`~spectrax.GraphDef`,
        static fields, opaque metadata such as configs, variable
        collections/metadata, sharing topology, and canonical paths. It
        does **not** hash parameter or buffer array values.

        Returns:
            A SHA-256 hex digest string representing the structural
            identity of the module.
        """
        from .graph import export

        graphdef, _state = export(self)
        return _digest_payload(
            {
                "version": 1,
                "kind": "spectrax.structure_hash",
                "graphdef": graphdef,
            }
        )

    def shape_hash(self) -> str:
        """Return a stable hash of structure plus state leaf shape/dtype.

        This is useful for checkpoint or compile-cache compatibility:
        it includes all information from :meth:`structure_hash` plus
        each exported state leaf's collection, path, shape, and dtype,
        but never hashes array contents.

        Returns:
            A SHA-256 hex digest string representing the structure and
            leaf signature of the module.
        """
        from .graph import export

        graphdef, state = export(self)
        state_signature = tuple(
            (
                collection,
                path,
                tuple(int(dim) for dim in tuple(getattr(leaf, "shape", ()))),
                str(getattr(leaf, "dtype", type(leaf).__name__)),
            )
            for collection, path, leaf in state.items()
        )
        return _digest_payload(
            {
                "version": 1,
                "kind": "spectrax.shape_hash",
                "graphdef": graphdef,
                "state_signature": state_signature,
            }
        )

    @property
    def training(self) -> bool:
        """Whether this module is currently in training mode.

        Returns:
            Result described by this helper.
        """
        return self._spx_training

    def train(self, mode: bool = True) -> Module:
        """Set this module (and all descendants) to training mode.

        Updates :attr:`_spx_training`, mirrors the value into the
        opaque-attribute map (so the flag survives
        :func:`~spectrax.export` / :func:`~spectrax.bind`), invalidates
        the export cache, bumps the global graph epoch, and recurses
        into every child module.

        Args:
            mode: ``True`` (default) enables training; ``False`` disables.

        Returns:
            ``self`` for chaining.
        """
        object.__setattr__(self, "_spx_training", bool(mode))
        self._spx_opaque["_spx_training"] = bool(mode)
        self._spx_export_cache = None
        _bump_graph_epoch()
        for _, child in self._spx_graph_children():
            if isinstance(child, Module):
                child.train(mode)
        return self

    def eval(self) -> Module:
        """Shorthand for ``self.train(False)``.

        Returns:
            ``self`` for chaining.
        """
        return self.train(False)

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Invoke the module's forward pipeline.

        The full ordering is:

        1. Pre-hooks (registered with
           :meth:`register_forward_pre_hook`) run; a non-``None``
           return value rewrites the ``(args, kwargs)`` seen by
           :meth:`forward`.
        2. Every context registered via :meth:`register_context` is
           entered, then the active :attr:`_spx_policy` is pushed onto
           the dtype-policy stack.
        3. :meth:`forward` runs.
        4. Post-hooks (registered with :meth:`register_forward_hook`)
           run; a non-``None`` return value replaces the output.
        5. If the active policy specifies an
           :attr:`~spectrax.Policy.output_dtype`, the output is cast
           accordingly.

        When invoked under a spectrax transform, hooks are silently
        skipped (with a single warning per offending module) because
        their side effects do not survive tracing. Use
        :meth:`sow` for transform-safe activation capture instead.

        Args:
            *args: Forwarded to :meth:`forward` after pre-hook
                rewriting.
            **kwargs: Forwarded to :meth:`forward` after pre-hook
                rewriting.

        Returns:
            Whatever :meth:`forward` returns (potentially rewritten by
            post-hooks or cast by the active policy).
        """
        if self._spx_pre_hooks and not _inside_transform():
            for h in list(self._spx_pre_hooks):
                r = h(self, args, kwargs)
                if r is not None:
                    args, kwargs = r
        elif self._spx_pre_hooks and _inside_transform():
            _warn_hooks_suppressed(self)

        policy = self._spx_policy
        if self._spx_contexts:
            with contextlib.ExitStack() as stack:
                for make_context in list(self._spx_contexts):
                    stack.enter_context(make_context())
                stack.enter_context(push_policy(policy))
                out: object = self.forward(*args, **kwargs)
        else:
            with push_policy(policy):
                out = self.forward(*args, **kwargs)

        if self._spx_fwd_hooks and not _inside_transform():
            for h in list(self._spx_fwd_hooks):
                r = h(self, args, kwargs, out)
                if r is not None:
                    out = r
        elif self._spx_fwd_hooks and _inside_transform():
            _warn_hooks_suppressed(self)

        if policy is not None and policy.output_dtype is not None:
            out = policy.cast_output(out)
        return out

    def forward(self, *args: object, **kwargs: object) -> object:
        """Compute and return the module output.

        Subclasses override this method. The default implementation
        raises :class:`NotImplementedError`.

        Args:
            *args: Positional inputs forwarded by :meth:`__call__`.
            **kwargs: Keyword inputs forwarded by :meth:`__call__`.

        Returns:
            The output of the module computation.

        Raises:
            NotImplementedError: When the subclass does not override
                this method.
        """
        raise NotImplementedError(f"{type(self).__name__} must override `forward`.")

    def register_forward_pre_hook(self, fn: ForwardPreHook) -> _HookHandle:
        """Register a pre-hook invoked before every :meth:`forward` call.

        See :class:`~spectrax.core.typing.ForwardPreHook` for the call
        signature; returning a non-``None`` ``(args, kwargs)`` rewrites
        the inputs seen by :meth:`forward`. Hooks are skipped under
        spectrax transforms.

        Args:
            fn: The hook callable.

        Returns:
            A :class:`_HookHandle`. Call :meth:`~_HookHandle.remove` on
            it to detach the hook.
        """
        self._spx_pre_hooks.append(fn)
        return _HookHandle(self._spx_pre_hooks, fn)

    def register_forward_hook(self, fn: ForwardHook) -> _HookHandle:
        """Register a post-hook invoked after every :meth:`forward` call.

        See :class:`~spectrax.core.typing.ForwardHook` for the call
        signature; returning a non-``None`` value replaces the
        forward output. Hooks are skipped under spectrax transforms.

        Args:
            fn: The hook callable.

        Returns:
            A :class:`_HookHandle`. Call :meth:`~_HookHandle.remove` on
            it to detach the hook.
        """
        self._spx_fwd_hooks.append(fn)
        return _HookHandle(self._spx_fwd_hooks, fn)

    def register_context(self, *contexts: object, **scope_values: object) -> _HookHandle:
        """Register contexts entered around every ``forward`` invocation.

        Positional arguments may be reusable context-manager objects or
        zero-argument factories returning fresh context managers. Keyword
        arguments are made available through :func:`spx.scope`; keyword
        values that are context managers (for example a JAX mesh) are
        also entered before the scope frame is pushed.

        Example::

            mesh = jax.sharding.Mesh(devices, ("fsdp", "tp"))
            self.register_context(mesh=mesh)

        Args:
            *contexts: Context managers or zero-argument factories.
            **scope_values: Keyword values exposed via :func:`spx.scope`.
                Values that are context managers are also entered.

        Returns:
            A handle with a :meth:`~_HookHandle.remove` method.
        """
        factories = [_context_factory(ctx) for ctx in contexts]
        for value in scope_values.values():
            if _is_context_manager(value):
                factories.append(_context_factory(value))
        if scope_values:
            values = dict(scope_values)
            factories.append(lambda values=values: _scope(**values))

        @contextlib.contextmanager
        def call_context() -> Iterator[None]:
            """Combined ``with``-block that enters every registered context for one call.

            Built lazily from ``factories`` so that argument re-binding
            (e.g. via :meth:`Module.register_context` with a fresh
            value) is picked up on every call. Raises
            :class:`TypeError` if any factory yields something that
            isn't a context manager.

            Returns:
                Result described by this helper.
            """
            with contextlib.ExitStack() as stack:
                for make_context in factories:
                    ctx = make_context()
                    if not _is_context_manager(ctx):
                        raise TypeError("Module.register_context() factories must return context managers.")
                    stack.enter_context(ctx)
                yield

        self._spx_contexts.append(call_context)
        return _HookHandle(self._spx_contexts, call_context)

    def sow(
        self,
        collection: str,
        name: str,
        value: object,
        *,
        reduce: str = "last",
    ) -> None:
        """Capture an intermediate value into a named :class:`~spectrax.Variable` slot.

        The slot lives on ``self`` under the attribute
        ``sow_{collection}_{name}``. On the first call a fresh
        :class:`~spectrax.Variable` is allocated with the given
        ``collection`` as its kind and the metadata pair
        ``{"reduce": reduce, "sow_name": name}``. On subsequent calls
        the existing slot is combined with ``value`` according to
        ``reduce``.

        Sowing into ``"intermediates"`` is the standard mechanism for
        transform-safe activation capture; the captured values survive
        :func:`spectrax.jit` / :func:`spectrax.grad` provided the
        transform marks the collection as mutable.

        Args:
            collection: Variable :attr:`~spectrax.Variable.kind` for the
                slot (typically ``"intermediates"``).
            name: User-facing label appended to the slot's attribute
                name.
            value: Captured value to record or combine.
            reduce: Combination strategy for repeated calls. One of:

                * ``"last"`` — replace the stored value with ``value``.
                * ``"sum"`` — add ``value`` to the running accumulator.
                * ``"stack"`` — concatenate ``value`` along a freshly
                  added leading axis.

        Returns:
            ``None``.

        Raises:
            TypeError: If the slot exists but is not a
                :class:`~spectrax.Variable`.
            ValueError: If ``reduce`` is not one of the supported
                strategies.
        """
        attr = f"sow_{collection}_{name}"
        existing = getattr(self, attr, None)
        if existing is None:
            initial = value[None] if reduce == "stack" else value
            var = Variable(initial, kind=collection, metadata={"reduce": reduce, "sow_name": name})
            self.__setattr__(attr, var)
            return
        if not isinstance(existing, Variable):
            raise TypeError(f"sow slot {attr} is not a Variable")
        if reduce == "last":
            existing.value = value
        elif reduce == "sum":
            cur = existing._raw_get()
            existing.value = value if cur is None else cur + value
        elif reduce == "stack":
            cur = existing._raw_get()
            if cur is None:
                existing.value = value[None]
            else:
                existing.value = jnp.concatenate([cur, value[None]], axis=0)
        else:
            raise ValueError(f"Unknown reduce strategy: {reduce!r}")

    def _spx_guard_not_in_transform(self, what: str) -> None:
        """Raise :class:`LazyInitUnderTransformError` if a transform is active.

        Used by lazy layers to refuse materialization under ``jit`` /
        ``grad`` / ``vmap`` / ``scan`` / ``remat``.

        Args:
            what: What value consumed by this operation.
        """
        if _inside_transform():
            raise LazyInitUnderTransformError(
                f"{what} cannot run inside a spectrax transform. "
                f"Call `module.init(rngs, ...)` before entering jit/grad/vmap/scan/remat."
            )

    def init(
        self,
        rngs: Rngs | int | None = None,
        *example_args: object,
        **example_kwargs: object,
    ) -> Module:
        """Attach an :class:`~spectrax.Rngs` and optionally run a dry forward.

        When ``rngs`` is non-``None`` it is normalized to a
        :class:`~spectrax.Rngs` (integers are wrapped via
        ``Rngs(int)``), assigned as the attribute ``rngs``, and pinned
        to the head of :attr:`_spx_attr_order` so it sorts consistently
        in graph-defs. The graph epoch is bumped so any cached export
        is rebuilt on next access.

        Passing example inputs additionally invokes
        :meth:`sequential_init`, which runs the forward pass under
        the lazy-materialization permission so that
        :class:`~spectrax.DeferredParameter` /
        :class:`~spectrax.DeferredBuffer` cells resolve their shapes.

        Args:
            rngs: Either an :class:`~spectrax.Rngs`, an integer seed, or
                ``None``. Non-``None`` values are attached to ``self``
                as the attribute ``rngs``.
            *example_args: Positional example inputs passed to
                :meth:`forward` to trigger lazy materialization.
            **example_kwargs: Keyword example inputs passed to
                :meth:`forward`.

        Returns:
            ``self`` for chaining.
        """
        from ..rng.rngs import Rngs as _Rngs

        if rngs is not None and not isinstance(rngs, _Rngs):
            rngs = _Rngs(rngs)
        if rngs is not None:
            object.__setattr__(self, "rngs", rngs)
            if "rngs" not in self._spx_attr_order:
                self._spx_attr_order.insert(0, "rngs")
            object.__setattr__(self, "_spx_export_cache", None)
            _bump_graph_epoch()

        if example_args or example_kwargs:
            self.sequential_init(*example_args, **example_kwargs)
        return self

    def sequential_init(self, *example_args: object, **example_kwargs: object) -> Module:
        """Materialize lazy descendants explicitly using example inputs.

        Runs ``self(*example_args, **example_kwargs)`` once with the
        thread-local materialization permission active, which lets
        :class:`~spectrax.DeferredParameter` and
        :class:`~spectrax.DeferredBuffer` cells observe and record their
        true shapes from the supplied inputs. Then invokes
        :meth:`materialize` to allocate every deferred leaf.

        This is the explicit counterpart to the legacy "first eager
        forward materializes lazy modules" behavior. Modules created
        under :func:`spectrax.lazy_init` require this call (or
        :meth:`init` with example inputs) before ordinary forward calls.

        Args:
            *example_args: Positional inputs shaped like the real
                forward inputs. Skipping them runs only
                :meth:`materialize`.
            **example_kwargs: Keyword inputs to forward.

        Returns:
            ``self`` for chaining.
        """
        if example_args or example_kwargs:
            with _allow_materialization():
                self(*example_args, **example_kwargs)
        self.materialize()
        return self

    def materialize(self) -> Module:
        """Resolve and initialize every :class:`~spectrax.DeferredParameter`
        and :class:`~spectrax.DeferredBuffer` in this subtree.

        Typically called automatically during the first eager forward
        pass (or via :meth:`sequential_init`).  Safe to call repeatedly —
        already-materialized variables are skipped.

        Returns:
            ``self`` for chaining.
        """
        seen: set[int] = set()

        def walk(m: Module) -> None:
            """Recurse through ``m``'s graph, materializing any deferred variables.

            ``seen`` (closure variable) guards against revisiting the
            same module via shared references, which would otherwise
            cause unbounded recursion or duplicate ``materialize()``
            calls on the same deferred variable.

            Args:
                m: M value consumed by this operation.
            """
            mid = id(m)
            if mid in seen:
                return
            seen.add(mid)
            for _, child in m._spx_graph_children():
                if isinstance(child, DeferredParameter | DeferredBuffer):
                    child.materialize()
                elif isinstance(child, Module):
                    walk(child)

        walk(self)
        return self

    def _resolve_deferred(self, var: Variable, shape: tuple[int, ...]) -> None:
        """Resolve ``var``'s shape if it is a deferred variable.

        Bridges layer construction code (which knows the inferred
        per-layer shape) with the deferred-variable resolve/
        materialize protocol. Refuses to run under a spectrax
        transform via :meth:`_spx_guard_not_in_transform`. After
        resolving the shape, materializes the variable immediately
        unless we're inside :func:`spectrax.lazy_init` without an
        explicit materialization permission (in which case the actual
        materialization is deferred to
        :meth:`sequential_init` / :meth:`materialize`).

        Args:
            var: A :class:`~spectrax.Variable` instance. No-op when
                ``var`` is not a deferred subclass or is already
                materialized.
            shape: Concrete shape to record on the variable.
        """
        if isinstance(var, DeferredParameter | DeferredBuffer) and not var.is_materialized:
            self._spx_guard_not_in_transform(f"{type(var).__name__} materialization")
            var.resolve_shape(shape)
            if _materialization_allowed() or not _explicit_lazy_mode():
                var.materialize()

    def freeze(self, selector: object = "parameters") -> Module:
        """Move every matched :class:`~spectrax.Variable`'s kind to ``"buffers"``.

        Each matched variable's previous :attr:`~spectrax.Variable.kind`
        is recorded in its metadata under the key ``"frozen_from"`` so
        that :meth:`unfreeze` can restore it. After freezing, the
        graph-export cache is cleared and the global graph epoch bumps
        so downstream :func:`~spectrax.export` callers see the new
        collection layout. Variables already in ``"buffers"`` are
        skipped without recording metadata.

        Args:
            selector: Anything :func:`~spectrax.core.selector.as_selector`
                accepts. Defaults to ``"parameters"`` so a bare
                ``module.freeze()`` freezes every parameter.

        Returns:
            ``self`` for chaining.
        """
        from .selector import as_selector

        sel = as_selector(selector)
        changed = False
        for _p, v in sel.apply(self):
            if v.kind != "buffers":
                v.metadata.setdefault("frozen_from", v.kind)
                v.kind = "buffers"
                changed = True
        if changed:
            object.__setattr__(self, "_spx_export_cache", None)
            _bump_graph_epoch()
        return self

    def unfreeze(self, selector: object = "buffers") -> Module:
        """Reverse :meth:`freeze` — move matched variables back to their pre-freeze kind.

        For every matched variable that carries a ``"frozen_from"``
        metadata entry, the entry is popped and the variable's
        :attr:`~spectrax.Variable.kind` is reset to the recorded value.
        Variables without a recorded prior kind are left unchanged.
        Bumps the global graph epoch when at least one variable is
        restored.

        Args:
            selector: Anything :func:`~spectrax.core.selector.as_selector`
                accepts. Defaults to ``"buffers"`` to walk the
                collection where :meth:`freeze` parks frozen variables.

        Returns:
            ``self`` for chaining.
        """
        from .selector import as_selector

        sel = as_selector(selector)
        changed = False
        for _p, v in sel.apply(self):
            prev = v.metadata.pop("frozen_from", None)
            if prev is not None:
                v.kind = prev
                changed = True
        if changed:
            object.__setattr__(self, "_spx_export_cache", None)
            _bump_graph_epoch()
        return self

    def perturb(self, name: str, x: object) -> object:
        """Insert an identity perturbation variable and add it to ``x``.

        On the first call a zero-valued :class:`~spectrax.Variable` is
        created on ``self`` under the attribute ``perturb_{name}`` with
        :attr:`~spectrax.Variable.kind` set to ``"perturbations"``. The
        returned value is ``x + perturbation.value`` — algebraically
        identity, but differentiating with respect to the perturbation
        variable yields ``dL/dx`` at the call site, which is the
        canonical trick for capturing intermediate-activation gradients
        without rewriting the forward pass. Subsequent calls reuse the
        existing variable.

        Args:
            name: Identifier for the perturbation slot. The actual
                attribute name is ``perturb_{name}``.
            x: Tensor (or pytree thereof) to perturb. Must be additively
                compatible with a same-shape zero tensor.

        Returns:
            ``x`` with the (zero) perturbation added.
        """
        attr = f"perturb_{name}"
        existing = getattr(self, attr, None)
        if existing is None:
            var = Variable(jnp.zeros_like(x), kind="perturbations", metadata={"perturb_name": name})
            self.__setattr__(attr, var)
            existing = var
        return x + existing.value

    def set_attributes(
        self,
        *,
        filter_fn: object = None,
        **attrs: object,
    ) -> Module:
        """Bulk-set attributes on ``self`` and every descendant module.

        Walks the live graph (deduplicating shared submodules by
        ``id``) and, for each module, assigns the given attributes
        only when the attribute already exists on the target. This
        guards against accidentally injecting an unrelated attribute
        on a layer that does not understand it (e.g. a
        :class:`~spectrax.nn.Linear` with no ``deterministic`` field
        is untouched). ``filter_fn`` further narrows the targeted
        modules.

        Args:
            filter_fn: Optional ``(module) -> bool`` predicate. When
                provided, attribute updates are applied only to
                modules for which the predicate returns truthy.
            **attrs: Attributes to set. Each ``key=value`` pair is
                pushed via ``setattr(module, key, value)`` on every
                eligible module that already exposes ``key``.

        Returns:
            ``self`` for chaining.
        """
        seen: set[int] = set()

        def walk(m: Module) -> None:
            """Recurse through ``m``'s sub-modules, applying ``attrs`` where they already exist.

            Like :meth:`materialize`'s ``walk``, ``seen`` (a
            closure ``set``) guards against revisiting the same module
            via shared references and so against double-applying
            attribute updates.

            Args:
                m: M value consumed by this operation.
            """
            mid = id(m)
            if mid in seen:
                return
            seen.add(mid)
            if filter_fn is None or filter_fn(m):
                for k, v in attrs.items():
                    if hasattr(m, k):
                        setattr(m, k, v)
            for _, child in m._spx_graph_children():
                if isinstance(child, Module):
                    walk(child)

        walk(self)
        return self

    def __repr__(self) -> str:
        """Render a pretty tree repr via
        :func:`spectrax.inspect.repr_module`, falling back to the class
        name if rendering raises.

        Returns:
            Result described by this helper.
        """
        try:
            from ..inspect.repr import repr_module

            return repr_module(self)
        except Exception:
            return f"<{type(self).__name__}>"


def _warn_hooks_suppressed(module: Module) -> None:
    """Emit a one-shot warning when hooks are skipped under a transform.

    Each distinct module instance produces at most one warning to keep
    traces clean during training loops.

    Args:
        module: SpectraX module instance operated on by the helper.
    """
    key = id(module)
    if key in _HOOK_WARNING_ONCE:
        return
    _HOOK_WARNING_ONCE.add(key)
    warnings.warn(
        f"Forward hooks on {type(module).__name__} are suppressed under a "
        f"spectrax transform. Use `self.sow('intermediates', ...)` to capture "
        f"values transform-safely.",
        stacklevel=3,
    )


@dataclass(frozen=True)
class _ModuleAux:
    """Aux data for the :class:`Module` pytree registration.

    Carries everything JAX needs to reconstruct a module that ``bind``
    alone can't recover — the structural :class:`~spectrax.GraphDef`,
    the ordered ``(collection, path)`` leaf specification, the
    training flag, forward/pre hooks, call contexts, the dtype policy,
    and the ``Opaque`` attribute map.

    JAX requires ``aux_data`` to be hashable for jit cache-key
    deduplication. The :class:`GraphDef`, the leaf spec, and the
    training bool are already hashable. Hooks/contexts (lists of closures),
    policy (unspecified), and the opaque dict are not generally
    hashable, so :meth:`__hash__` and :meth:`__eq__` fall back to
    ``id(...)``-based comparison for those three — producing
    conservative jit cache misses when hook/policy/opaque identity
    differs, while deduplicating on the hot path where a user re-uses
    the same model instance across steps.
    """

    gdef: object
    leaf_spec: tuple[tuple[str, str], ...]
    training: bool
    fwd_hooks: list
    pre_hooks: list
    contexts: list
    policy: object
    opaque: dict

    def __hash__(self) -> int:
        """Hash structural parts directly; hash hook/policy/opaque by identity.

        Returns:
            Result described by this helper.
        """
        return hash(
            (
                self.gdef,
                self.leaf_spec,
                self.training,
                id(self.fwd_hooks),
                id(self.pre_hooks),
                id(self.contexts),
                id(self.policy),
                id(self.opaque),
            )
        )

    def __eq__(self, other: object) -> bool:
        """Match on structural fields, and on identity for mutable containers.

        Args:
            other: Other value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        if not isinstance(other, _ModuleAux):
            return NotImplemented
        return (
            self.gdef == other.gdef
            and self.leaf_spec == other.leaf_spec
            and self.training == other.training
            and self.fwd_hooks is other.fwd_hooks
            and self.pre_hooks is other.pre_hooks
            and self.contexts is other.contexts
            and self.policy is other.policy
            and self.opaque is other.opaque
        )


_PYTREE_REGISTERED: set[type] = set()
"""Track which Module subclasses have been pytree-registered.

:func:`jax.tree_util.register_pytree_node` raises if a class is
registered twice. Test fixtures that redefine subclasses at module
reload, or a module imported transitively, can trigger repeat
registration; the set guards against that.
"""


def _module_collection_children(m: Module) -> tuple[dict[str, dict[str, object]], _ModuleAux]:
    """Build collection subtrees and shared aux data for module pytree flattening."""
    from .graph import export
    from .paths import str_to_path
    from .state import _nested_set

    cache = m._spx_export_cache
    if cache is None or cache[0] != _graph_epoch():
        export(m)
        cache = m._spx_export_cache
    assert cache is not None
    gdef = cache[1]
    var_entries = cache[7] if len(cache) >= 8 else cache[2]
    leaf_spec = cache[6] if len(cache) >= 7 else tuple((kind, path) for kind, path, _ in var_entries)

    collection_children: dict[str, dict[str, object]] = {}
    for (collection, path), (_kind, _path, var) in zip(leaf_spec, var_entries, strict=True):
        _nested_set(collection_children.setdefault(collection, {}), str_to_path(path), var._raw_get())
    aux = _ModuleAux(
        gdef=gdef,
        leaf_spec=leaf_spec,
        training=m._spx_training,
        fwd_hooks=m._spx_fwd_hooks,
        pre_hooks=m._spx_pre_hooks,
        contexts=m._spx_contexts,
        policy=m._spx_policy,
        opaque=m._spx_opaque,
    )
    return collection_children, aux


def _module_flatten_with_keys(m: Module) -> tuple[tuple[tuple[object, object], ...], _ModuleAux]:
    """Pytree flatten-with-keys for any :class:`Module` instance.

    Reads the hot :func:`spectrax.export` cache and returns the raw
    variable leaves directly, in canonical-path order, without building
    an intermediate :class:`~spectrax.State` pytree. Static /
    structural information, runtime flags, and non-graph attributes all
    go into :class:`_ModuleAux`.

    Args:
        m: M value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    from .state import _KeyedSubtree

    collection_children, aux = _module_collection_children(m)
    key_children = tuple(
        (jax.tree_util.DictKey(collection), _KeyedSubtree(collection_children[collection]))
        for collection in collection_children
    )
    return key_children, aux


def _module_flatten(m: Module) -> tuple[tuple[object, ...], _ModuleAux]:
    """Pytree flatten (no keys) — complement of :func:`_module_flatten_with_keys`.

    Needed by :func:`jax.tree_util.register_pytree_with_keys` so
    ``tree_flatten`` / ``tree_leaves`` can skip the key computation
    when the caller doesn't need paths.

    Args:
        m: M value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    from .state import _KeyedSubtree

    collection_children, aux = _module_collection_children(m)
    return tuple(_KeyedSubtree(collection_children[collection]) for collection in collection_children), aux


def _module_unflatten(aux: _ModuleAux, leaves: object) -> Module:
    """Rebuild a :class:`Module` from ``(aux, leaves)``.

    Reassembles the :class:`~spectrax.State` directly from the ordered
    ``(collection, path)`` leaf spec, hands it to :func:`~spectrax.bind`
    to rebuild the module tree, then restores
    the runtime-state attributes from ``aux``. Hooks and the opaque
    dict are shallow-copied so mutations on the unflattened copy don't
    leak back to the original list/dict references held in ``aux``.

    Args:
        aux: Aux value consumed by this operation.
        leaves: Leaves value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    from .graph import bind
    from .paths import str_to_path
    from .state import State, _KeyedSubtree, _nested_set

    state_data: dict[str, dict[str, object]] = {}
    collection_order: list[str] = []
    for collection, _path in aux.leaf_spec:
        if collection not in collection_order:
            collection_order.append(collection)
    if len(leaves) == len(collection_order) and all(
        isinstance(collection_tree, _KeyedSubtree) for collection_tree in leaves
    ):
        for collection, collection_tree in zip(collection_order, leaves, strict=True):
            if isinstance(collection_tree, _KeyedSubtree):
                state_data[collection] = collection_tree.data
    elif len(aux.leaf_spec) != len(leaves):
        raise ValueError(
            "Module pytree leaf count mismatch during unflatten: "
            f"expected {len(aux.leaf_spec)} leaves from the auxiliary spec, got {len(leaves)}."
        )
    else:
        for (collection, path), leaf in zip(aux.leaf_spec, leaves, strict=True):
            _nested_set(state_data.setdefault(collection, {}), str_to_path(path), leaf)
    state = State._from_raw(state_data)
    m = bind(aux.gdef, state)
    object.__setattr__(m, "_spx_training", aux.training)
    object.__setattr__(m, "_spx_fwd_hooks", list(aux.fwd_hooks))
    object.__setattr__(m, "_spx_pre_hooks", list(aux.pre_hooks))
    object.__setattr__(m, "_spx_contexts", list(aux.contexts))
    object.__setattr__(m, "_spx_policy", aux.policy)
    object.__setattr__(m, "_spx_opaque", dict(aux.opaque))
    for opaque_name, opaque_value in aux.opaque.items():
        if not opaque_name.startswith("_") and opaque_name not in m._spx_attr_order:
            m._spx_attr_order.append(opaque_name)
        object.__setattr__(m, opaque_name, _public_value(opaque_value))
    return m


def _register_module_pytree(cls: type) -> None:
    """Register ``cls`` with JAX's pytree registry (idempotent).

    The three callbacks all delegate to the class-agnostic helpers
    above, so every Module subclass shares a single flatten/unflatten
    implementation — no per-class code generation.
    """
    if cls in _PYTREE_REGISTERED:
        return
    _PYTREE_REGISTERED.add(cls)
    jax.tree_util.register_pytree_with_keys(
        cls,
        _module_flatten_with_keys,
        _module_unflatten,
        flatten_func=_module_flatten,
    )


_register_module_pytree(Module)
