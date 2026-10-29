# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Private shared type aliases and protocols.

Every type defined here is re-exported from :mod:`spectrax.typing`; the
split exists to avoid forcing users to depend on a private path. Types
are intentionally narrow: ``Array`` is the JAX array type,
``ArrayLike`` anything JAX can coerce, ``DType`` anything JAX accepts as
a dtype spec, ``Shape`` a concrete integer tuple, ``PRNGKey`` a JAX
PRNG key array, ``Path`` and ``PathComponent`` the pieces of a
graph path, and the remaining protocols describe the callable shapes
accepted by layer initializers, module hooks, and variable observers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, TypeAlias, TypeVar, runtime_checkable

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from jax.typing import ArrayLike as _ArrayLike
    from jax.typing import DTypeLike as _DTypeLike
else:
    _ArrayLike = object
    _DTypeLike = object


Array: TypeAlias = jax.Array
"""The concrete JAX array type (``jax.Array``)."""

ArrayLike: TypeAlias = "_ArrayLike"
"""Any value JAX can coerce to :class:`Array`.

This includes :class:`Array`, :class:`numpy.ndarray`, Python scalars,
and anything implementing the array protocol (:class:`~spectrax.Variable`
qualifies via ``__jax_array__``).
"""

DType: TypeAlias = "_DTypeLike"
"""A JAX-compatible dtype specifier.

Accepts :class:`jnp.dtype`, a NumPy dtype class, a string like ``"f32"``,
or a Python scalar type like ``int``.
"""

Shape: TypeAlias = tuple[int, ...]
"""A concrete tensor shape (no symbolic dimensions)."""

PRNGKey: TypeAlias = jax.Array
"""A JAX PRNG key array, as returned by :func:`jax.random.PRNGKey`."""

PyTree: TypeAlias = object
"""An arbitrary JAX pytree.

This intentionally uses ``object`` as the dynamic top type so callers
do not lose type checking around values that happen to be pytrees.
"""


PathComponent: TypeAlias = str | int
"""Either an attribute name / dict key (``str``) or a list index (``int``)."""

Path: TypeAlias = tuple[PathComponent, ...]
"""An ordered tuple of :data:`PathComponent` values identifying a location
in a module graph.
"""


T = TypeVar("T")


@runtime_checkable
class Initializer(Protocol):
    """The callable shape of a parameter initializer.

    Every function in :mod:`spectrax.init` implements this protocol. An
    initializer is a *pure* function that, given a PRNG key, a target
    shape, and a dtype, returns an :class:`Array` of that shape and dtype.
    """

    def __call__(self, key: PRNGKey, shape: Shape, dtype: DType = jnp.float32) -> Array:
        """Return a freshly-initialized array of ``shape`` and ``dtype`` from ``key``.

        Args:
            key: Logical key, path segment, or PRNG key used by the operation.
            shape: Array shape requested by the initializer or helper.
            dtype: Array dtype requested for the produced value.

        Returns:
            Result of invoking the wrapped callable or module.
        """
        ...


ModulePredicate: TypeAlias = "Callable[[object, str], bool]"
"""Callable ``(module, path) -> bool`` predicate used by
:class:`~spectrax.Selector` to filter modules.
"""

VariablePredicate: TypeAlias = "Callable[[object, str], bool]"
"""Callable ``(variable, path) -> bool`` predicate used by
:class:`~spectrax.Selector` to filter variables.
"""


class ForwardPreHook(Protocol):
    """The callable shape of a forward *pre-*hook.

    Invoked with the owning module, the positional args tuple, and the
    keyword args dict. Returning a new ``(args, kwargs)`` pair overrides
    the arguments that are passed to ``forward``. Returning ``None``
    leaves them unchanged.
    """

    def __call__(
        self,
        module: object,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> tuple[tuple[object, ...], dict[str, object]] | None:
        """Optionally rewrite ``(args, kwargs)`` before ``forward`` runs.

        Args:
            module: SpectraX module instance operated on by the helper.
            args: Positional arguments forwarded to the wrapped callable.
            kwargs: Keyword arguments forwarded to the wrapped callable.

        Returns:
            Result of invoking the wrapped callable or module.
        """
        ...


class ForwardHook(Protocol):
    """The callable shape of a forward *post-*hook.

    Invoked after ``forward`` with the owning module, the args, the kwargs,
    and the forward output. Returning a non-``None`` value replaces the
    output seen by the caller; returning ``None`` leaves it unchanged.
    """

    def __call__(
        self,
        module: object,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        output: object,
    ) -> object | None:
        """Optionally rewrite the forward output. Return ``None`` to keep it unchanged.

        Args:
            module: SpectraX module instance operated on by the helper.
            args: Positional arguments forwarded to the wrapped callable.
            kwargs: Keyword arguments forwarded to the wrapped callable.
            output: Output value consumed by this operation.

        Returns:
            Result of invoking the wrapped callable or module.
        """
        ...


class VariableObserver(Protocol):
    """The callable shape of a variable write observer.

    Invoked eagerly on every successful write to a
    :class:`~spectrax.Variable` with the variable itself, the previous
    value, and the new value. Observers run on a best-effort basis;
    exceptions raised by an observer are swallowed.
    """

    def __call__(self, var: object, old: object, new: object) -> None:
        """React to ``var`` having its value changed from ``old`` to ``new``.

        Args:
            var: Var value consumed by this operation.
            old: Old value consumed by this operation.
            new: New value consumed by this operation.
        """
        ...
