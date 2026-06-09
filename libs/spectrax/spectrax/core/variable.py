# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The :class:`Variable` reference cell.

:class:`Variable` is the only mutable value spectrax modules own. It is:

* identified by a process-wide unique ``ref_id`` allocated at
  construction, used by :func:`~spectrax.export` to detect shared state;
* tagged with a ``kind`` (its collection — ``"parameters"``, ``"buffers"``,
  ``"batch_stats"``, ``"cache"``, ``"intermediates"``, ``"rng"``, or any
  user-defined string);
* a carrier of free-form ``metadata`` (sharding, logical axis names,
  tie group tags, …); and
* transparently array-like: arithmetic and the array protocols
  ``__array__`` and ``__jax_array__`` delegate to ``.value``.  JAX ≥0.9.2
  no longer calls ``__jax_array__`` during abstractification (PyTree
  registration handles that), so the method is safe to keep for eager-mode
  convenience.

Sharing by identity (the same :class:`Variable` instance appearing at
two attribute paths) is the canonical mechanism for tied weights. Each
``bind`` allocates fresh ``ref_id`` values, but sharing topology is
preserved by the :class:`~spectrax.GraphDef`'s normalized ref ids and
``shared_paths``.

Writes to ``.value`` are immediate in eager mode. Under a spectrax
transform the transforms module installs a thread-local write hook that
intercepts writes and redirects them into the traced state, which is
how mutations (e.g. batch-norm running stats) survive the transform
boundary.
"""

from __future__ import annotations

import contextlib
import itertools
import threading
from collections.abc import Callable, Iterator
from typing import ClassVar, cast

import jax
import jax.numpy as jnp
import numpy as np

from ._typing import Array, ArrayLike, DType, VariableObserver
from .sharding import AxisNames, Sharding, normalize_sharding
from .stage_assignment import (
    PIPELINE_STAGE_METADATA_KEY,
    current_stage_assignment,
    metadata_stage_assignment,
    resolve_stage_rank,
)

InitPlacementHook = Callable[[object, dict[str, object], bool], object | None]

__all__ = [
    "KINDS_BUILTIN",
    "Buffer",
    "DeferredBuffer",
    "DeferredParameter",
    "InitPlacementHook",
    "Parameter",
    "Variable",
    "variable_init_placement",
]


_REF_COUNTER = itertools.count(1)
_ref_lock = threading.Lock()


def _fresh_ref_id() -> int:
    """Allocate a fresh process-unique ``ref_id`` (thread-safe).

    Returns:
        Result described by this helper.
    """
    with _ref_lock:
        return next(_REF_COUNTER)


def _maybe_stamped_sharding(value: object) -> Sharding | None:
    """Return any initializer-stamped sharding metadata on ``value``.

    Args:
        value: Value consumed by the helper.

    Returns:
        Return any initializer-stamped sharding metadata on ``value``.
    """
    stamped = getattr(value, "_spx_sharding", None)
    return normalize_sharding(stamped) if stamped is not None else None


def _existing_value_sharding(value: object) -> object | None:
    """Return the concrete sharding already carried by ``value``, if any.

    Tracers carry only a derived/origin sharding — accessing it triggers
    ``find_progenitors`` over the entire jaxpr, which is O(jaxpr-size)
    per call and O(N^2) when initializing many variables inside one
    ``jax.eval_shape`` (e.g. lazy model construction). Skip them.

    Args:
        value: Value consumed by the helper.

    Returns:
        Return the concrete sharding already carried by ``value``, if any.
    """
    if isinstance(value, jax.core.Tracer):
        return None
    return getattr(value, "sharding", None)


_WRITE_HOOK: threading.local = threading.local()


def _set_write_hook(hook: Callable[[Variable], bool] | None) -> None:
    """Install a thread-local write hook for :class:`Variable` writes.

    The hook signature is ``hook(var, new) -> bool``. If the hook returns
    ``True`` it is considered to have handled the write, and the normal
    eager write is suppressed. Passing ``None`` restores the default eager
    behavior.

    Used by the transforms module to redirect writes into the traced
    state carried across the transform boundary.

    Args:
        hook: Hook value consumed by this operation.
    """
    _WRITE_HOOK.hook = hook


def _get_write_hook() -> Callable[[Variable], bool] | None:
    """Return the currently-installed write hook, or ``None``.

    Returns:
        Return the currently-installed write hook, or ``None``.
    """
    return getattr(_WRITE_HOOK, "hook", None)


_READ_HOOK: threading.local = threading.local()


def _set_read_hook(hook: Callable[[Variable]] | None) -> None:
    """Install a thread-local read hook for :class:`Variable` reads.

    The hook signature is ``hook(var) -> object`` and its return value is
    used in place of the underlying ``_value``. Passing ``None`` restores
    the default eager read behavior.

    Args:
        hook: Hook value consumed by this operation.
    """
    _READ_HOOK.hook = hook


def _get_read_hook() -> Callable[[Variable]] | None:
    """Return the currently-installed read hook, or ``None``.

    Returns:
        Return the currently-installed read hook, or ``None``.
    """
    return getattr(_READ_HOOK, "hook", None)


_INIT_PLACEMENT_HOOKS: threading.local = threading.local()


@contextlib.contextmanager
def variable_init_placement(hook: InitPlacementHook) -> Iterator[None]:
    """Install a metadata-driven placement hook for new variable values.

    The hook is invoked from :func:`_initialize_value` for every freshly
    constructed variable — after dtype coercion, before the built-in
    SpectraX mesh fallback. Returning a non-``None`` value claims the
    initialization and that returned value becomes the variable's
    storage. Returning ``None`` defers to the default placement
    behavior. Hooks stack: nested ``with`` blocks layer additional
    hooks on top of the outer ones, but only the innermost hook is
    consulted (subsequent hooks may opt back in by re-entering the
    context).

    The hook signature is
    ``(value, metadata, explicit_sharding) -> object | None`` where
    ``explicit_sharding`` is ``True`` when the variable's constructor
    received an explicit ``sharding=`` argument.

    Args:
        hook: Callable matching :data:`InitPlacementHook`.

    Yields:
        ``None``. The hook is uninstalled on exit.

    Raises:
        TypeError: If ``hook`` is not callable.
    """
    if not callable(hook):
        raise TypeError("variable_init_placement expected a callable hook")
    previous = tuple(getattr(_INIT_PLACEMENT_HOOKS, "stack", ()))
    _INIT_PLACEMENT_HOOKS.stack = (*previous, hook)
    try:
        yield
    finally:
        if previous:
            _INIT_PLACEMENT_HOOKS.stack = previous
        elif hasattr(_INIT_PLACEMENT_HOOKS, "stack"):
            delattr(_INIT_PLACEMENT_HOOKS, "stack")


def _get_init_placement_hook() -> InitPlacementHook | None:
    """Return the innermost active :class:`InitPlacementHook`, or ``None``.

    Reads the thread-local stack pushed by
    :func:`variable_init_placement`. Used by deferred-variable
    materialization to decide where to place a freshly initialized
    leaf (e.g. on a specific stage's sub-mesh).

    Returns:
        Return the innermost active :class:`InitPlacementHook`, or ``None``.
    """
    stack = getattr(_INIT_PLACEMENT_HOOKS, "stack", ())
    return stack[-1] if stack else None


KINDS_BUILTIN: tuple[str, ...] = (
    "parameters",
    "buffers",
    "batch_stats",
    "cache",
    "intermediates",
    "rng",
)
"""The collection names spectrax reserves for built-in use.

User-defined collections are permitted; avoid the reserved names.
"""


class Variable:
    """Reference cell with identity, kind, metadata, and array delegation.

    :class:`Variable` is the base class. Use :class:`Parameter` for
    trainable weights and :class:`Buffer` for non-trainable state
    (running statistics, caches, …). Subclasses may override
    :attr:`default_kind` to set a per-class default collection.
    Subclasses may set :attr:`inherit_stage_assignment` to ``False``
    for global state that should not be owned by a pipeline stage.

    Attributes:
        _value: The underlying array. Access via the :attr:`value`
            property so read/write hooks run.
        kind: The collection this variable belongs to.
        metadata: A free-form dict of per-variable metadata
            (``"sharding"``, ``"axis_names"``, ``"tie_group"``, …).
        ref_id: Process-unique integer identity. Used by
            :func:`~spectrax.export` to detect shared variables; ``bind``
            allocates fresh ``ref_id`` values.
        _observers: Callbacks invoked on each successful write in eager
            mode. See :meth:`add_observer`.
    """

    default_kind: ClassVar[str] = "buffers"
    """Default :attr:`kind` for instances of this class."""

    inherit_stage_assignment: ClassVar[bool] = True
    """Whether active ``assign_stage`` scopes stamp new instances."""

    __slots__ = (
        "_observers",
        "_value",
        "kind",
        "metadata",
        "ref_id",
    )

    _value: object
    kind: str
    metadata: dict[str, object]
    ref_id: int
    _observers: list[VariableObserver]

    def __init__(
        self,
        value: ArrayLike,
        *,
        kind: str | None = None,
        metadata: dict[str, object] | None = None,
        ref_id: int | None = None,
    ) -> None:
        """Construct a reference cell wrapping ``value``.

        Stamps the variable with an active :func:`assign_stage` hint
        when the subclass opts in (:attr:`inherit_stage_assignment`)
        and no pipeline-stage entry is already present in
        ``metadata``. Initializes the observer list as empty.

        Args:
            value: The initial array (or array-like). Not coerced here;
                subclasses that need coercion do it themselves.
            kind: Override the default collection. ``None`` falls back
                to :attr:`default_kind`.
            metadata: Free-form per-variable metadata. A shallow copy
                is made; ``None`` is treated as an empty dict.
            ref_id: An explicit ``ref_id`` to adopt. When ``None`` (the
                common case) a fresh process-unique id is allocated
                via :func:`_fresh_ref_id`.
        """
        self._value = value
        self.kind = kind if kind is not None else type(self).default_kind
        self.metadata = dict(metadata) if metadata else {}
        assignment = current_stage_assignment()
        if (
            type(self).inherit_stage_assignment
            and assignment is not None
            and PIPELINE_STAGE_METADATA_KEY not in self.metadata
        ):
            self.metadata[PIPELINE_STAGE_METADATA_KEY] = assignment
        self.ref_id = ref_id if ref_id is not None else _fresh_ref_id()
        self._observers = []

    @property
    def value(self) -> Array:
        """Read the stored array.

        If a thread-local read hook is installed it is consulted first;
        otherwise the raw underlying value is returned.

        Returns:
            Result described by this helper.
        """
        read_hook = _get_read_hook()
        if read_hook is not None:
            return cast(Array, read_hook(self))
        return cast(Array, self._value)

    @value.setter
    def value(self, new: ArrayLike) -> None:
        """Write ``new`` to the cell.

        Under a spectrax transform the installed write hook may claim
        the write (returning ``True``), in which case the underlying
        storage is left untouched. Otherwise the value is stored and
        every registered :class:`~spectrax.typing.VariableObserver` is
        notified; observer exceptions are swallowed.

        Args:
            new: The new array value to store.
        """
        hook = _get_write_hook()
        if hook is not None and hook(self, new):
            return
        old = self._value
        self._value = new
        for obs in list(self._observers):
            with contextlib.suppress(Exception):
                obs(self, old, new)

    def _raw_get(self) -> object:
        """Return the underlying value, bypassing any installed hooks.

        Used by the graph / transform machinery that must not be
        redirected by its own hooks.

        Returns:
            Return the underlying value, bypassing any installed hooks.
        """
        return self._value

    def _raw_set(self, new: object) -> None:
        """Write ``new`` to the cell, bypassing hooks and observers.

        Used by the graph / transform machinery to apply a state patch
        back to a live module after a transform.

        Args:
            new: New value consumed by this operation.
        """
        self._value = new

    def add_observer(self, fn: VariableObserver) -> None:
        """Register ``fn`` to be called on each eager write.

        Observers are not invoked under spectrax transforms (writes are
        redirected by the write hook before observers run).

        Args:
            fn: A callable ``(var, old, new) -> None`` to invoke on
                every successful eager write.
        """
        self._observers.append(fn)

    def remove_observer(self, fn: VariableObserver) -> None:
        """Unregister a previously-added observer (no-op if absent).

        Args:
            fn: The callable previously passed to :meth:`add_observer`.
        """
        if fn in self._observers:
            self._observers.remove(fn)

    @property
    def sharding(self) -> Sharding | None:
        """Return the :class:`Sharding` from :attr:`metadata` or ``None``.

        Returns:
            Return the :class:`Sharding` from :attr:`metadata` or ``None``.
        """
        s = self.metadata.get("sharding")
        return normalize_sharding(s) if s is not None else None

    @property
    def axis_names(self) -> AxisNames | None:
        """Return the logical axis names from :attr:`metadata` or ``None``.

        Returns:
            Return the logical axis names from :attr:`metadata` or ``None``.
        """
        names = self.metadata.get("axis_names")
        return tuple(names) if names is not None else None

    @property
    def stage_assignment(self) -> tuple[int, int] | None:
        """Return the logical ``(current, total)`` stage hint, if any.

        Returns:
            Return the logical ``(current, total)`` stage hint, if any.
        """
        return metadata_stage_assignment(self.metadata)

    @property
    def stage_index(self) -> int | None:
        """Return the variable's logical stage index within :attr:`stage_count`.

        This is the construction-time ``current`` value from
        ``assign_stage(total=..., current=...)``. Use
        :meth:`resolved_stage_index` to map it onto a concrete MPMD mesh.

        Returns:
            Return the variable's logical stage index within :attr:`stage_count`.
        """
        assignment = self.stage_assignment
        return assignment[0] if assignment is not None else None

    @property
    def stage_count(self) -> int | None:
        """Return the logical number of stage slots the variable was tagged in.

        Returns:
            Return the logical number of stage slots the variable was tagged in.
        """
        assignment = self.stage_assignment
        return assignment[1] if assignment is not None else None

    def resolved_stage_index(self, mesh_or_dim: object) -> int | None:
        """Resolve :attr:`stage_index` onto a physical MPMD stage index.

        ``mesh_or_dim`` may be:

        * an integer ``mpmd_dim``
        * a :class:`~spectrax.sharding.SpxMesh`
        * a :class:`~spectrax.runtime.types.MpMdMesh`

        Returns ``None`` when the variable has no stage hint or when an
        ``SpxMesh`` without an MPMD axis is supplied.

        Args:
            mesh_or_dim: The mesh or pipeline dimension to resolve
                against.

        Returns:
            The zero-based physical stage rank, or ``None``.

        Raises:
            TypeError: If ``mesh_or_dim`` is not an ``int``, ``SpxMesh``,
                or ``MpMdMesh``.
        """
        mpmd_dim: int | None
        if isinstance(mesh_or_dim, int):
            mpmd_dim = mesh_or_dim
        else:
            from ..runtime.types.mesh import MpMdMesh
            from ..sharding.mesh import SpxMesh

            if isinstance(mesh_or_dim, SpxMesh):
                mpmd_dim = mesh_or_dim.mpmd_mesh.mpmd_dim if mesh_or_dim.is_mpmd else None
            elif isinstance(mesh_or_dim, MpMdMesh):
                mpmd_dim = mesh_or_dim.mpmd_dim
            else:
                raise TypeError(
                    "resolved_stage_index(mesh_or_dim): expected an int, "
                    f"SpxMesh, or MpMdMesh; got {type(mesh_or_dim).__name__}."
                )
        if mpmd_dim is None:
            return None
        return resolve_stage_rank(self.stage_assignment, mpmd_dim)

    def stage_mesh(self, mesh: object) -> object:
        """Return the stage-local mesh this variable should live on.

        For non-MPMD meshes, returns the input mesh unchanged (or its
        underlying ``jax_mesh`` for :class:`~spectrax.sharding.SpxMesh`).
        For MPMD meshes plus a stage-tagged variable, returns the owning
        stage sub-mesh.

        Args:
            mesh: A JAX mesh, ``SpxMesh``, or ``MpMdMesh``.

        Returns:
            The appropriate mesh for this variable's pipeline stage.
        """
        from ..runtime.types.mesh import MpMdMesh
        from ..sharding.mesh import SpxMesh

        if isinstance(mesh, SpxMesh):
            if not mesh.is_mpmd:
                return mesh.jax_mesh
            owner = self.resolved_stage_index(mesh)
            return mesh.mpmd_mesh.submesh(owner) if owner is not None else mesh.jax_mesh
        if isinstance(mesh, MpMdMesh):
            owner = self.resolved_stage_index(mesh)
            return mesh.submesh(owner) if owner is not None else mesh.jax_mesh
        return mesh

    def named_sharding(self, mesh: object) -> object:
        """Resolve this variable's metadata to a ``NamedSharding``.

        Args:
            mesh: The mesh against which logical axis names are resolved.

        Returns:
            A :class:`jax.sharding.NamedSharding` instance, or ``None``
            when the metadata contains no sharding information.
        """
        from ..sharding.partition import named_sharding_for_variable

        return named_sharding_for_variable(self, mesh)

    def __array__(self, dtype: DType | None = None, copy: bool | None = None) -> np.ndarray:
        """Implement the NumPy array protocol so ``np.asarray(var)`` works.

        The NumPy 2 ``copy`` keyword is accepted and ignored; copy
        semantics are delegated to :func:`jax.numpy.asarray`.

        Args:
            dtype: Array dtype requested for the produced value.
            copy: Copy value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        del copy
        v = self.value
        if dtype is None:
            return np.asarray(v)
        return np.asarray(v, dtype=dtype)

    def __jax_array__(self) -> Array:
        """Implement the JAX array protocol so ``jnp.split(var)`` et al. work.

        JAX ≥0.9.2 no longer calls this during abstractification; PyTree
        registration handles that.  This method is only invoked on concrete
        values (or tracers inside a jitted function) and makes Variables
        transparently array-like in eager code.

        Returns:
            Result described by this helper.
        """
        return self.value

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the stored array.

        Returns:
            Result described by this helper.
        """
        return tuple(self.value.shape)

    @property
    def dtype(self) -> object:
        """Dtype of the stored array.

        Returns:
            Result described by this helper.
        """
        return self.value.dtype

    @property
    def ndim(self) -> int:
        """Rank (number of dimensions) of the stored array.

        Returns:
            Result described by this helper.
        """
        return int(self.value.ndim)

    @property
    def size(self) -> int:
        """Total number of elements in the stored array.

        Returns:
            Result described by this helper.
        """
        return int(self.value.size)

    def astype(self, dtype: DType) -> Array:
        """Return the value cast to ``dtype`` (not a Variable).

        Args:
            dtype: Target JAX dtype.

        Returns:
            The stored array cast to ``dtype``.
        """
        return jnp.asarray(self.value, dtype=dtype)

    def __getattr__(self, name: str) -> object:
        """Delegate unknown attribute lookups to the underlying array.

        This lets array methods (``transpose``, ``reshape``, ``swapaxes``,
        ``flatten``, ``ravel``, …) and properties (``T``, ``device``, …)
        work transparently without explicitly unwrapping ``.value``.

        Args:
            name: Name used for lookup, logging, or registration.

        Returns:
            Result described by this helper.
        """
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self.value, name)

    @staticmethod
    def _v(other: object) -> object:
        """Unwrap a :class:`Variable` (or pass through) for arithmetic ops.

        Args:
            other: Other value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        return other.value if isinstance(other, Variable) else other

    def __pos__(self) -> Array:
        """Unary ``+``.

        Returns:
            Result described by this helper.
        """
        return +self.value

    def __neg__(self) -> Array:
        """Unary ``-``.

        Returns:
            Result described by this helper.
        """
        return -self.value

    def __abs__(self) -> Array:
        """``abs(var)``.

        Returns:
            Result described by this helper.
        """
        return abs(self.value)

    def __add__(self, o: object) -> Array:
        """``var + o``.

        Args:
            o: O value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        return self.value + self._v(o)

    def __radd__(self, o: object) -> Array:
        """``o + var``.

        Args:
            o: O value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        return self._v(o) + self.value

    def __sub__(self, o: object) -> Array:
        """``var - o``.

        Args:
            o: O value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        return self.value - self._v(o)

    def __rsub__(self, o: object) -> Array:
        """``o - var``.

        Args:
            o: O value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        return self._v(o) - self.value

    def __mul__(self, o: object) -> Array:
        """``var * o``.

        Args:
            o: O value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        return self.value * self._v(o)

    def __rmul__(self, o: object) -> Array:
        """``o * var``.

        Args:
            o: O value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        return self._v(o) * self.value

    def __truediv__(self, o: object) -> Array:
        """``var / o``.

        Args:
            o: O value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        return self.value / self._v(o)

    def __rtruediv__(self, o: object) -> Array:
        """``o / var``.

        Args:
            o: O value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        return self._v(o) / self.value

    def __floordiv__(self, o: object) -> Array:
        """``var // o``.

        Args:
            o: O value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        return self.value // self._v(o)

    def __mod__(self, o: object) -> Array:
        """``var % o``.

        Args:
            o: O value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        return self.value % self._v(o)

    def __pow__(self, o: object) -> Array:
        """``var ** o``.

        Args:
            o: O value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        return self.value ** self._v(o)

    def __rpow__(self, o: object) -> Array:
        """``o ** var``.

        Args:
            o: O value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        return self._v(o) ** self.value

    def __matmul__(self, o: object) -> Array:
        """``var @ o``.

        Args:
            o: O value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        return self.value @ self._v(o)

    def __rmatmul__(self, o: object) -> Array:
        """``o @ var``.

        Args:
            o: O value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        return self._v(o) @ self.value

    def __getitem__(self, idx: object) -> Array:
        """``var[idx]`` — indexes into the stored array.

        Args:
            idx: Idx value consumed by this operation.

        Returns:
            Selected item from the container.
        """
        return self.value[idx]

    def __eq__(self, o: object) -> bool:
        """Identity-based equality: two Variables compare equal iff their
        ``ref_id`` matches. This keeps :class:`Variable` hashable and
        usable as a dict key.

        Args:
            o: O value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        if isinstance(o, Variable):
            return self.ref_id == o.ref_id
        return NotImplemented

    def __ne__(self, o: object) -> bool:
        """Inverse of :meth:`__eq__`.

        Args:
            o: O value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        r = self.__eq__(o)
        return r if r is NotImplemented else not r

    def __hash__(self) -> int:
        """Hash derived from :attr:`ref_id`.

        Returns:
            Result described by this helper.
        """
        return hash(("spectrax.Variable", self.ref_id))

    def __bool__(self) -> bool:
        """``bool(var)`` — delegates to the stored array's truth value.

        Returns:
            Result described by this helper.
        """
        return bool(self.value)

    def __repr__(self) -> str:
        """Compact diagnostic repr with class, kind, shape, dtype, ref id.

        Returns:
            Result described by this helper.
        """
        try:
            shape = tuple(self.value.shape)
            dtype = self.value.dtype
            return f"{type(self).__name__}(kind={self.kind!r}, shape={shape}, dtype={dtype}, ref={self.ref_id})"
        except Exception:
            return f"{type(self).__name__}(kind={self.kind!r}, ref={self.ref_id})"


class Parameter(Variable):
    """Trainable weight.

    Default :attr:`kind` is ``"parameters"`` (or ``"buffers"`` when
    ``trainable=False``). :func:`~spectrax.grad` differentiates with
    respect to ``"parameters"`` by default.
    """

    default_kind: ClassVar[str] = "parameters"

    def __init__(
        self,
        value: ArrayLike,
        *,
        dtype: DType | None = None,
        sharding: Sharding | AxisNames | None = None,
        axis_names: AxisNames | None = None,
        trainable: bool = True,
        metadata: dict[str, object] | None = None,
        ref_id: int | None = None,
    ) -> None:
        """Construct a parameter cell.

        Args:
            value: Initial array.
            dtype: Storage dtype. ``None`` preserves the value's dtype.
            sharding: Sharding spec — a :class:`Sharding` or a tuple of
              logical axis names.
            axis_names: Per-dimension logical axis names.
            trainable: ``True`` stores under ``"parameters"``; ``False``
              stores under ``"buffers"`` so :func:`~spectrax.grad`
              excludes it by default.
            metadata: Additional metadata (merged into :attr:`metadata`).
            ref_id: Explicit ``ref_id`` to adopt.
        """
        meta: dict[str, object] = dict(metadata) if metadata else {}
        stamped_sharding = _maybe_stamped_sharding(value)
        assignment = current_stage_assignment()
        if sharding is not None:
            meta["sharding"] = normalize_sharding(sharding)
        elif stamped_sharding is not None:
            meta["sharding"] = stamped_sharding
        if axis_names is not None:
            meta["axis_names"] = tuple(axis_names)
        if assignment is not None and PIPELINE_STAGE_METADATA_KEY not in meta:
            meta[PIPELINE_STAGE_METADATA_KEY] = assignment
        arr = _initialize_value(value, dtype, metadata=meta, explicit_sharding=sharding is not None)
        super().__init__(
            arr,
            kind="parameters" if trainable else "buffers",
            metadata=meta,
            ref_id=ref_id,
        )


class Buffer(Variable):
    """Non-trainable state (running means, counters, KV caches, ...).

    Default :attr:`kind` is ``"buffers"``; pass an explicit ``kind=`` to
    route to another collection (``"batch_stats"``, ``"cache"``, …).
    """

    default_kind: ClassVar[str] = "buffers"

    def __init__(
        self,
        value: ArrayLike,
        *,
        dtype: DType | None = None,
        kind: str | None = None,
        sharding: Sharding | AxisNames | None = None,
        axis_names: AxisNames | None = None,
        metadata: dict[str, object] | None = None,
        ref_id: int | None = None,
    ) -> None:
        """Construct a buffer cell.

        Args:
            value: Initial array.
            dtype: Storage dtype. ``None`` preserves the value's dtype.
            kind: Override the collection. ``None`` uses
              :attr:`default_kind` (``"buffers"``).
            sharding: Sharding spec.
            axis_names: Per-dimension logical axis names.
            metadata: Additional metadata.
            ref_id: Explicit ``ref_id`` to adopt.
        """
        meta: dict[str, object] = dict(metadata) if metadata else {}
        stamped_sharding = _maybe_stamped_sharding(value)
        assignment = current_stage_assignment()
        if sharding is not None:
            meta["sharding"] = normalize_sharding(sharding)
        elif stamped_sharding is not None:
            meta["sharding"] = stamped_sharding
        if axis_names is not None:
            meta["axis_names"] = tuple(axis_names)
        if assignment is not None and PIPELINE_STAGE_METADATA_KEY not in meta:
            meta[PIPELINE_STAGE_METADATA_KEY] = assignment
        arr = _initialize_value(value, dtype, metadata=meta, explicit_sharding=sharding is not None)
        super().__init__(
            arr,
            kind=kind if kind is not None else "buffers",
            metadata=meta,
            ref_id=ref_id,
        )


class DeferredParameter(Parameter):
    """Parameter whose shape is resolved lazily during the first forward pass.

    Stores an initializer and a *shape specification* (which may contain
    ``None`` placeholders) instead of a concrete array.  On first access
    :attr:`value` returns a zero array of the resolved shape so the forward
    pass can execute for shape inference.  Calling :meth:`materialize`
    replaces the placeholder with a real initialized array.

    Built-in layers such as :class:`~spectrax.nn.Linear` create a
    ``DeferredParameter`` automatically when an input dimension is passed as
    ``None``.
    """

    def __init__(
        self,
        shape_spec: tuple[int | None, ...],
        init: Callable[[object, tuple[int, ...]]],
        rngs: object,
        dtype: object,
        *,
        sharding: Sharding | AxisNames | None = None,
        axis_names: AxisNames | None = None,
        trainable: bool = True,
        metadata: dict[str, object] | None = None,
        ref_id: int | None = None,
    ) -> None:
        """Construct a deferred parameter with a (possibly partial) shape spec.

        Stores the initializer, RNG handle, and dtype for later use and
        installs a single-element zero placeholder so the variable is
        immediately a valid pytree leaf. Shape resolution is performed
        lazily by :meth:`resolve_shape` (typically driven by the first
        forward pass that observes a real input shape), then
        :meth:`materialize` replaces the placeholder with the real
        initialized array.

        Args:
            shape_spec: Per-dimension shape specification. Entries may
                be concrete ``int`` sizes or ``None`` placeholders to
                be resolved later from observed input shapes; the rank
                fixes the parameter's rank.
            init: Initializer callable matching the
                :class:`~spectrax.core.typing.Initializer` signature
                ``(rngs, shape, dtype) -> Array``.
            rngs: First positional argument forwarded to ``init``;
                typically an :class:`~spectrax.Rngs` handle.
            dtype: Storage dtype. Used for both the placeholder and
                the materialized array.
            sharding: Optional sharding spec; recorded in metadata.
            axis_names: Optional logical axis names; recorded in
                metadata.
            trainable: ``True`` (default) routes the variable to the
                ``"parameters"`` collection; ``False`` routes it to
                ``"buffers"``.
            metadata: Additional metadata merged into
                :attr:`Variable.metadata`.
            ref_id: Optional explicit ``ref_id`` to adopt.
        """
        meta: dict[str, object] = dict(metadata) if metadata else {}
        if sharding is not None:
            meta["sharding"] = normalize_sharding(sharding)
        if axis_names is not None:
            meta["axis_names"] = tuple(axis_names)
        super().__init__(jnp.zeros(1, dtype=dtype), dtype=dtype, metadata=meta, ref_id=ref_id, trainable=trainable)
        self._deferred_shape_spec = tuple(shape_spec)
        self._deferred_init = init
        self._deferred_rngs = rngs
        self._deferred_dtype = dtype
        self._deferred_resolved_shape: tuple[int, ...] | None = None
        self._deferred_materialized = False

    @property
    def is_materialized(self) -> bool:
        """Whether this deferred parameter has been materialized.

        Returns:
            Result described by this helper.
        """
        return getattr(self, "_deferred_materialized", False)

    def resolve_shape(self, shape: tuple[int, ...]) -> None:
        """Set the concrete shape.

        No-op once the parameter has been materialized.

        Args:
            shape: Concrete shape tuple. Must have the same length as
                the original ``shape_spec`` so the rank is preserved.

        Returns:
            ``None``.

        Raises:
            ValueError: If ``shape`` has a different rank than
                ``shape_spec``.
        """
        if self.is_materialized:
            return
        if len(shape) != len(self._deferred_shape_spec):
            raise ValueError(f"DeferredParameter expected rank {len(self._deferred_shape_spec)}, got {len(shape)}")
        self._deferred_resolved_shape = tuple(int(s) for s in shape)

    def materialize(self) -> None:
        """Call the stored initializer and replace the placeholder value.

        Runs ``init(rngs, resolved_shape, dtype)`` and stores the
        result through :func:`_initialize_value`, so any active
        sharding metadata or placement hook is honored. Idempotent —
        repeated calls after the first materialization are no-ops.

        Returns:
            ``None``.

        Raises:
            RuntimeError: If :meth:`resolve_shape` has not yet been
                called.
        """
        if self.is_materialized:
            return
        if getattr(self, "_deferred_resolved_shape", None) is None:
            raise RuntimeError("DeferredParameter shape not resolved. Run a forward pass first.")
        arr = self._deferred_init(self._deferred_rngs, self._deferred_resolved_shape, self._deferred_dtype)
        arr = _initialize_value(arr, None, metadata=self.metadata, explicit_sharding="sharding" in self.metadata)
        self._raw_set(arr)
        self._deferred_materialized = True

    @property
    def value(self) -> Array:
        """Return the materialized array; until materialized, return a zero placeholder.

        The placeholder lets shape-inference forward passes run before
        the real init has happened. Reading before
        :meth:`resolve_shape` raises :class:`RuntimeError`.

        Returns:
            Return the materialized array; until materialized, return a zero placeholder.
        """
        if self.is_materialized:
            return super().value
        if getattr(self, "_deferred_resolved_shape", None) is None:
            raise RuntimeError("DeferredParameter shape not resolved. Run a forward pass first.")
        return jnp.zeros(self._deferred_resolved_shape, self._deferred_dtype)

    @value.setter
    def value(self, new: ArrayLike) -> None:
        """Write a real value into the variable and mark it as materialized.

        Args:
            new: The new array value to store.
        """
        self._deferred_materialized = True
        super(DeferredParameter, self.__class__).value.fset(self, new)


class DeferredBuffer(Buffer):
    """Buffer whose shape is resolved lazily during the first forward pass.

    Mirrors :class:`DeferredParameter` for non-trainable state.
    """

    def __init__(
        self,
        shape_spec: tuple[int | None, ...],
        init: Callable[[object, tuple[int, ...]]],
        rngs: object,
        dtype: object,
        *,
        kind: str | None = None,
        sharding: Sharding | AxisNames | None = None,
        axis_names: AxisNames | None = None,
        metadata: dict[str, object] | None = None,
        ref_id: int | None = None,
    ) -> None:
        """Construct a deferred buffer.

        Same lazy semantics as :class:`DeferredParameter` but creates
        a non-trainable :class:`Buffer`.

        Args:
            shape_spec: Per-dimension shape specification with
                ``None`` placeholders for as-yet-unknown sizes.
            init: Initializer callable
                ``(rngs, shape, dtype) -> Array``.
            rngs: First positional argument forwarded to ``init``.
            dtype: Storage dtype.
            kind: Optional override for the buffer's collection name;
                defaults to ``"buffers"``.
            sharding: Optional sharding spec; recorded in metadata.
            axis_names: Optional logical axis names; recorded in
                metadata.
            metadata: Additional metadata.
            ref_id: Optional explicit ``ref_id`` to adopt.
        """
        meta: dict[str, object] = dict(metadata) if metadata else {}
        if sharding is not None:
            meta["sharding"] = normalize_sharding(sharding)
        if axis_names is not None:
            meta["axis_names"] = tuple(axis_names)
        super().__init__(jnp.zeros(1, dtype=dtype), dtype=dtype, kind=kind, metadata=meta, ref_id=ref_id)
        self._deferred_shape_spec = tuple(shape_spec)
        self._deferred_init = init
        self._deferred_rngs = rngs
        self._deferred_dtype = dtype
        self._deferred_resolved_shape: tuple[int, ...] | None = None
        self._deferred_materialized = False

    @property
    def is_materialized(self) -> bool:
        """Whether this deferred buffer has been materialized.

        Returns:
            Result described by this helper.
        """
        return getattr(self, "_deferred_materialized", False)

    def resolve_shape(self, shape: tuple[int, ...]) -> None:
        """Set the concrete shape; rank must match the original ``shape_spec``.

        Args:
            shape: Concrete shape tuple. Must have the same length as
                the original ``shape_spec``.

        Returns:
            ``None``.

        Raises:
            ValueError: If ``shape`` has a different rank than
                ``shape_spec``.
        """
        if self.is_materialized:
            return
        if len(shape) != len(self._deferred_shape_spec):
            raise ValueError(f"DeferredBuffer expected rank {len(self._deferred_shape_spec)}, got {len(shape)}")
        self._deferred_resolved_shape = tuple(int(s) for s in shape)

    def materialize(self) -> None:
        """Run the stored initializer and replace the placeholder buffer value.

        Returns:
            ``None``.

        Raises:
            RuntimeError: If :meth:`resolve_shape` has not yet been
                called.
        """
        if self.is_materialized:
            return
        if getattr(self, "_deferred_resolved_shape", None) is None:
            raise RuntimeError("DeferredBuffer shape not resolved. Run a forward pass first.")
        arr = self._deferred_init(self._deferred_rngs, self._deferred_resolved_shape, self._deferred_dtype)
        arr = _initialize_value(arr, None, metadata=self.metadata, explicit_sharding="sharding" in self.metadata)
        self._raw_set(arr)
        self._deferred_materialized = True

    @property
    def value(self) -> Array:
        """Return the materialized array; before materialization, return a zero placeholder.

        Returns:
            Return the materialized array; before materialization, return a zero placeholder.
        """
        if self.is_materialized:
            return super().value
        if getattr(self, "_deferred_resolved_shape", None) is None:
            raise RuntimeError("DeferredBuffer shape not resolved. Run a forward pass first.")
        return jnp.zeros(self._deferred_resolved_shape, self._deferred_dtype)

    @value.setter
    def value(self, new: ArrayLike) -> None:
        """Write a real value into the buffer and mark it as materialized.

        Args:
            new: The new array value to store.
        """
        self._deferred_materialized = True
        super(DeferredBuffer, self.__class__).value.fset(self, new)


def _initialize_value(
    value: ArrayLike,
    dtype: DType | None,
    *,
    metadata: dict[str, object],
    explicit_sharding: bool,
) -> object:
    """Coerce ``value`` and apply constructor-time sharding when available.

    Pipeline:

    1. Run ``value`` through :func:`_coerce_value` to honor an explicit
       ``dtype`` while preserving any sharding the value already carries.
    2. Short-circuit on tracers — under abstract evaluation the
       ``.sharding`` probe is O(jaxpr-size) and must be skipped.
    3. Consult any active :func:`variable_init_placement` hook; the
       hook's non-``None`` return value claims the placement.
    4. Otherwise fall back to the spectrax mesh: if a current mesh
       resolves the metadata to a :class:`~jax.sharding.NamedSharding`,
       :func:`jax.device_put` the value onto that sharding (unless the
       value already carries an equivalent sharding and no explicit
       override was requested).

    Args:
        value: The raw initial value (Python scalar, NumPy / JAX array,
            tracer, …).
        dtype: Optional explicit storage dtype. ``None`` preserves the
            value's existing dtype.
        metadata: The variable's metadata dict; consulted for sharding
            and pipeline-stage hints.
        explicit_sharding: ``True`` when the caller passed
            ``sharding=`` to the variable constructor; controls
            whether an existing sharding on ``value`` may be
            overridden.

    Returns:
        Either ``value`` unchanged or a placed copy on the resolved
        sharding.
    """
    arr = _coerce_value(value, dtype)
    if not hasattr(arr, "shape") or not hasattr(arr, "dtype"):
        return arr

    # Under abstract evaluation (jax.eval_shape, e.g. lazy model
    # construction) ``arr`` is a Tracer. Device placement on a tracer is
    # meaningless, and reading its ``.sharding`` triggers JAX's
    # ``find_progenitors`` walk over the entire growing jaxpr — fatal for
    # large modules (O(N^2) over parameter count). Skip the hook and the
    # downstream sharding probing entirely so all placement hooks are
    # protected without requiring each hook to defend itself.
    if isinstance(arr, jax.core.Tracer):
        return arr

    hook = _get_init_placement_hook()
    if hook is not None:
        placed = hook(arr, metadata, explicit_sharding)
        if placed is not None:
            return placed

    from ..sharding.mesh import current_mesh
    from ..sharding.partition import named_sharding_for_metadata

    mesh = current_mesh()
    if mesh is None:
        return arr

    existing = _existing_value_sharding(arr)
    if existing is not None and not explicit_sharding and PIPELINE_STAGE_METADATA_KEY not in metadata:
        return arr

    sharding = named_sharding_for_metadata(metadata, mesh)
    if sharding is None:
        return arr
    if existing is not None and existing == sharding:
        return arr

    return jax.device_put(arr, sharding)


def _coerce_value(value: ArrayLike, dtype: DType | None) -> object:
    """Coerce ``value`` while preserving existing JAX-array sharding.

    Note: prefer ``isinstance(value, jax.Array)`` over
    ``hasattr(value, "sharding")``. The ``hasattr`` form *invokes* the
    ``sharding`` property to test for it, and on a tracer that property
    walks the entire growing jaxpr (``find_progenitors``) — paying
    O(jaxpr-size) per call, O(N^2) when initializing many variables
    inside a single ``jax.eval_shape`` (e.g. lazy model construction).

    Args:
        value: Value consumed by the helper.
        dtype: Array dtype requested for the produced value.

    Returns:
        Result described by this helper.
    """
    if isinstance(value, jax.Array):
        return value.astype(dtype) if dtype is not None else value
    if dtype is not None:
        return jnp.asarray(value, dtype=dtype)
    return _as_array(value)


def _as_array(value: ArrayLike) -> object:
    """Coerce ``value`` to a JAX array when it is concrete enough to do so.

    Pass-through on values that are already traced or unrecognized so
    that abstract tracers flow untouched through layer construction.

    Args:
        value: Value consumed by the helper.

    Returns:
        Result described by this helper.
    """
    if isinstance(value, np.ndarray | jnp.ndarray | int | float | complex | bool | list | tuple):
        return jnp.asarray(value)
    return value


def _deferred_aux(v: Variable) -> dict[str, object] | None:
    """Capture DeferredParameter/DeferredBuffer private state for pytree round-trips.

    Args:
        v: V value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    if not isinstance(v, DeferredParameter | DeferredBuffer):
        return None
    return {
        "shape_spec": getattr(v, "_deferred_shape_spec", None),
        "init": getattr(v, "_deferred_init", None),
        "rngs": getattr(v, "_deferred_rngs", None),
        "dtype": getattr(v, "_deferred_dtype", None),
        "resolved_shape": getattr(v, "_deferred_resolved_shape", None),
        "materialized": getattr(v, "_deferred_materialized", False),
    }


def _restore_deferred_aux(v: Variable, deferred: dict[str, object] | None) -> None:
    """Restore DeferredParameter/DeferredBuffer private state captured in aux.

    Args:
        v: V value consumed by this operation.
        deferred: Deferred value consumed by this operation.
    """
    if deferred is None:
        return
    object.__setattr__(v, "_deferred_shape_spec", deferred["shape_spec"])
    object.__setattr__(v, "_deferred_init", deferred["init"])
    object.__setattr__(v, "_deferred_rngs", deferred["rngs"])
    object.__setattr__(v, "_deferred_dtype", deferred["dtype"])
    object.__setattr__(v, "_deferred_resolved_shape", deferred["resolved_shape"])
    object.__setattr__(v, "_deferred_materialized", deferred["materialized"])


_VariableAux = tuple[type[Variable], str, dict[str, object], int, dict[str, object] | None]


def _variable_flatten(v: Variable) -> tuple[list[object], _VariableAux]:
    """Flatten a Variable to its value leaf.

    Args:
        v: V value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    return ([v.value], (type(v), v.kind, dict(v.metadata), v.ref_id, _deferred_aux(v)))


def _variable_unflatten(aux: _VariableAux, children: list[object]) -> Variable:
    """Reconstruct a Variable (or subclass) from its value.

    We use ``object.__new__`` to bypass custom ``__init__`` logic in
    subclasses (e.g. EasyDeL's ``ModuleCaches`` hard-codes ``kind``).
    Metadata, ``ref_id``, and deferred-variable private state are also
    restored so standalone variable pytree round-trips do not corrupt
    sharding/stage metadata or lazy parameter state.

    Args:
        aux: Aux value consumed by this operation.
        children: Children value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    cls, kind, metadata, ref_id, deferred = aux
    obj = object.__new__(cls)
    object.__setattr__(obj, "_value", children[0])
    object.__setattr__(obj, "kind", kind)
    object.__setattr__(obj, "metadata", dict(metadata))
    object.__setattr__(obj, "ref_id", ref_id)
    object.__setattr__(obj, "_observers", [])
    _restore_deferred_aux(obj, deferred)
    return obj


def _variable_flatten_with_keys(
    v: Variable,
) -> tuple[list[tuple[jax.tree_util.KeyEntry, Array]], _VariableAux]:
    """Keyed flatten for :func:`jax.tree_util.register_pytree_with_keys`.

    Args:
        v: V value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    return (
        [(jax.tree_util.GetAttrKey("value"), v.value)],
        (type(v), v.kind, dict(v.metadata), v.ref_id, _deferred_aux(v)),
    )


for _var_cls in (Variable, Parameter, Buffer, DeferredParameter, DeferredBuffer):
    jax.tree_util.register_pytree_with_keys(
        _var_cls,
        _variable_flatten_with_keys,
        _variable_unflatten,
        flatten_func=_variable_flatten,
    )


def _auto_register_init_subclass(cls, **kwargs: object) -> None:
    """Hook that PyTree-registers every new Variable subclass.

    Args:
        **kwargs: Additional keyword arguments forwarded to the wrapped callable or backend.
    """
    super(Variable, cls).__init_subclass__(**kwargs)
    if cls not in (Variable, Parameter, Buffer, DeferredParameter, DeferredBuffer):
        jax.tree_util.register_pytree_with_keys(
            cls,
            _variable_flatten_with_keys,
            _variable_unflatten,
            flatten_func=_variable_flatten,
        )


Variable.__init_subclass__ = classmethod(_auto_register_init_subclass)
