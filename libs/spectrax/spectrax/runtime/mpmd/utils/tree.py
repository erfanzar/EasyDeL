# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pytree and cotangent leaf helpers for the SpectraX MPMD runtime."""

from __future__ import annotations

import jax

from ....core.variable import Variable


def _delete_if_possible(x: object) -> None:
    """Free an array's device buffer if JAX allows it.

    :class:`jax.Array` exposes a ``.delete()`` method on newer JAX
    versions; older versions / non-committed arrays silently ignore.
    Exceptions are swallowed so the schedule loop never crashes on a
    donation miss.

    Args:
        x: Input value consumed by the operation.
    """
    try:
        delete = getattr(x, "delete", None)
        if callable(delete):
            delete()
    except Exception:
        pass


def _delete_tree_arrays(x: object) -> None:
    """Best-effort deletion for every array-like leaf in a pytree."""
    if x is None:
        return
    try:
        jax.tree.map(lambda leaf: _delete_if_possible(leaf), x, is_leaf=_is_leaf)
    except Exception:
        _delete_if_possible(x)


def _is_leaf(x: object) -> bool:
    """Stop pytree traversal at JAX arrays and Spectrax :class:`Variable` nodes.

    Used as the ``is_leaf`` argument throughout the MPMD runtime so
    that :class:`Variable` containers (which are themselves pytrees of
    metadata + array) are kept whole — otherwise their internal
    metadata leaks out as separate flat-leaf entries and breaks the
    flat-arg <-> outer-jaxpr-invar correspondence.

    Args:
        x: object pytree node.

    Returns:
        ``True`` when ``x`` is a :class:`jax.Array` or
        :class:`Variable`.
    """
    return isinstance(x, jax.Array | Variable)


def _is_float0(x: object) -> bool:
    """Return ``True`` when ``x`` carries the JAX ``float0`` zero-sized sentinel.

    JAX uses ``float0`` to mark cotangents of integer-valued primals
    (produced when ``allow_int=True`` is passed to autodiff). These
    leaves cannot participate in arithmetic; the runtime must short
    them out before scaling or addition.

    Args:
        x: object pytree leaf.

    Returns:
        ``True`` iff ``x.dtype == jax.dtypes.float0``.
    """
    return getattr(x, "dtype", None) == jax.dtypes.float0


def _scale_grad(x: object, scale: object) -> object:
    """Multiply ``x`` by ``scale`` unless ``x`` is a ``float0`` sentinel.

    ``float0`` leaves are returned unchanged so the resulting pytree
    can still be passed back through JAX's autodiff plumbing.

    Args:
        x: Cotangent leaf.
        scale: Scalar multiplier.

    Returns:
        ``x * scale`` for normal arrays, ``x`` for ``float0``.
    """
    if x is None or _is_float0(x):
        return x
    return x * scale


def _add_grad(a: object, b: object) -> object:
    """Add two cotangent leaves treating ``float0`` as the additive identity.

    When either operand is ``float0`` the other is returned untouched.
    Mirrors JAX's autodiff convention so accumulating grads from
    integer-input branches does not raise.

    Args:
        a: First cotangent leaf.
        b: Second cotangent leaf.

    Returns:
        ``a + b`` (or whichever operand is non-``float0``).
    """
    if a is None:
        return b
    if b is None:
        return a
    if _is_float0(a):
        return b
    if _is_float0(b):
        return a
    return a + b


def _cast_cotangent_like(cotangent: object, primal: object) -> object:
    """Cast ``cotangent`` to its matching ``primal`` dtype before transport.

    Some XLA backends complain when a cotangent's dtype differs from
    the producer's output dtype; casting here keeps the transport
    well-typed without forcing the upstream backward jit to widen its
    grads. ``float0`` cotangents are returned untouched.

    Args:
        cotangent: The incoming cotangent array.
        primal: The forward output whose dtype defines the target.

    Returns:
        ``cotangent`` cast to ``primal.dtype`` (or unchanged when
        already matching, when one side has no dtype, or when the
        cotangent is ``float0``).
    """
    if _is_float0(cotangent):
        return cotangent
    cot_dtype = getattr(cotangent, "dtype", None)
    primal_dtype = getattr(primal, "dtype", None)
    if cot_dtype is not None and primal_dtype is not None and cot_dtype != primal_dtype and hasattr(cotangent, "astype"):
        return cotangent.astype(primal_dtype)
    return cotangent
