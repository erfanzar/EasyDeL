# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Module-aware :mod:`jax.lax` scan wrappers.

Two functions are exported:

* :func:`scan` is the module-aware analogue of :func:`jax.lax.scan` for
  the ``(module, x) -> y`` step shape. The module's state is partitioned
  by the ``mutable`` selector into a *carry* (the declared-mutable
  collections) and an *invariant* (everything else). Only the carry is
  threaded through :func:`jax.lax.scan`; the invariant is required to
  stay structurally identical from step to step, otherwise the
  ``mutable=`` declaration was incomplete and a :class:`ValueError`
  with a remediation hint is raised. After the scan returns, the final
  carry is overlaid back on top of the invariant and the resulting
  state is written back to ``init_module`` via
  :func:`~spectrax.transforms.split_merge.apply_mutations`.

* :func:`associative_scan` wraps :func:`jax.lax.associative_scan` for
  *pure* associative binary combine functions. Because that primitive
  performs a tree-shaped parallel prefix with no carry, there is no
  well-defined place to thread module mutations; any direct write to
  the captured module raises :class:`~spectrax.IllegalMutationError`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import jax.lax as lax

from ..core.errors import IllegalMutationError
from ..core.graph import bind, export, live_variables
from ..core.module import Module, _graph_epoch, _set_inside_transform
from ..core.selector import SelectorSugar
from ..core.state import State
from .split_merge import _ModuleRef, apply_mutations, assert_state_unchanged, make_direct_readonly, resolve_mutable

__all__ = ["associative_scan", "scan"]


def associative_scan(
    fn: Callable[[Module]],
    module: Module,
    elems: object,
    *,
    reverse: bool = False,
    axis: int = 0,
    mutable: SelectorSugar = (),
) -> object:
    """Module-aware :func:`jax.lax.associative_scan`.

    ``fn`` has the shape ``(module, a, b) -> c`` and must be associative
    over ``a`` and ``b`` in the same way required by upstream JAX.

    Unlike :func:`scan`, :func:`jax.lax.associative_scan` performs a
    tree-structured parallel prefix with no state carry. That means
    module mutations have no well-defined semantics here, so the module
    is rebound from its original state for each combine and any write
    to that rebound module raises :class:`IllegalMutationError`.

    Args:
        fn: Associative binary combine function ``(module, a, b) -> c``.
        module: Read-only :class:`Module` captured by the combine.
        elems: Input pytree whose ``axis`` dimension is scanned.
        reverse: Forwarded to :func:`jax.lax.associative_scan`.
        axis: Forwarded to :func:`jax.lax.associative_scan`.
        mutable: Unsupported for associative scans; anything other than
            the empty value raises :class:`ValueError`.

    Returns:
        The prefix-combined values with the same structure as ``elems``.
    """
    mutable_sel = resolve_mutable(mutable)
    if mutable_sel is not None:
        raise ValueError(
            "associative_scan() does not support mutable= because "
            "jax.lax.associative_scan has no module-state carry. "
            "Keep the combine function pure."
        )
    if not isinstance(module, Module):
        raise TypeError("associative_scan() requires a Module as module")

    gdef, state = export(module)

    def combine(a: object, b: object) -> object:
        """Rebind a fresh module for this pairwise combine and forbid every write.

        :func:`jax.lax.associative_scan` calls the combine ``O(n log n)``
        times in a parallel-prefix pattern; rebinding ``(gdef, state)``
        per call gives each call a fresh, identically-seeded module
        instance. The graph epoch is recorded before invoking ``fn`` so
        any structural mutation (adding/removing a child module) is
        caught, and per-variable identity snapshots are checked
        afterward to guarantee no value-level write slipped through.

        Args:
            a: Positional arguments forwarded to the wrapped callable.
            b: B value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        m = bind(gdef, state)
        epoch_before = _graph_epoch()
        snapshots = [(path, var.kind, var, var._value) for path, var in live_variables(m)]
        _set_inside_transform(True)
        try:
            out = fn(m, a, b)
        finally:
            _set_inside_transform(False)

        if _graph_epoch() != epoch_before:
            raise IllegalMutationError(
                "associative_scan() does not support structural module "
                "mutations because jax.lax.associative_scan has no "
                "module-state carry."
            )

        for path, kind, var, initial in snapshots:
            if var._value is not initial:
                raise IllegalMutationError(
                    "associative_scan() does not support module mutations "
                    "because jax.lax.associative_scan has no module-state "
                    f"carry. The combine function mutated {kind!r} at {path!r}; "
                    "keep it pure."
                )
        return out

    return lax.associative_scan(combine, elems, reverse=reverse, axis=axis)


def scan(
    fn: Callable[[Module]],
    init_module: Module,
    xs: object,
    *,
    length: int | None = None,
    mutable: SelectorSugar = (),
    unroll: int = 1,
) -> object:
    """Scan ``fn`` over ``xs`` threading ``init_module`` as state.

    Args:
        fn: Per-step function ``(module, x) -> y``.
        init_module: :class:`Module` providing the initial state.
        xs: Scanned sequence (pytree with a leading axis).
        length: Optional explicit sequence length.
        mutable: Selector for collections that may be carried through
            the scan. Collections outside this selector must be
            structurally invariant from step to step.
        unroll: Forwarded to :func:`jax.lax.scan`.

    Returns:
        The stacked ``ys`` from every step. When ``mutable=`` declares
        any collections, the final carry for those collections is
        written back to ``init_module`` in place; otherwise the module
        is left untouched (and any mutation raises).

    Raises:
        TypeError: If ``init_module`` is not a :class:`~spectrax.Module`.
        ValueError: If the invariant portion of the state changes
            structurally across iterations.
        IllegalMutationError: If a direct write to the live module
            occurs on the readonly fast path, or if a mutation lands
            outside the ``mutable`` selector.
    """
    mutable_sel = resolve_mutable(mutable)
    if not isinstance(init_module, Module):
        raise TypeError("scan() requires a Module as init_module")

    if mutable_sel is None:
        guarded_step = make_direct_readonly(
            lambda _carry, x: (None, fn(init_module, x)),
            explicit_modules=(init_module,),
            structural_error_message=(
                "scan() invariant state changed across iterations. Declare the changing collection via `mutable=...`."
            ),
        )

        _carry, ys = lax.scan(guarded_step, None, xs, length=length, unroll=unroll)
        return ys
    gdef, state = export(init_module)
    carry_state, invariant = mutable_sel.partition_state(init_module, state)

    def step(carry: State, x: object) -> tuple[State, object]:
        """Single scan step: merge invariant onto carry, run ``fn``, re-partition.

        Overlays the per-iteration ``carry`` on top of the captured
        invariant, rebinds a fresh module from the merged state, runs
        the user-provided ``fn(module, x)`` inside the inside-transform
        thread-local, and partitions the resulting state into a new
        carry plus invariant. The new invariant is checked against the
        original via :func:`_check_invariant_equal` (key-set equality)
        and :func:`~spectrax.transforms.split_merge.assert_state_unchanged`
        (per-leaf identity); any difference indicates the caller's
        ``mutable=`` selector did not cover a collection that ``fn``
        actually wrote to.

        Args:
            carry: Loop or scan carry value.
            x: Input value consumed by the operation.

        Returns:
            Result described by this helper.
        """
        full = carry.overlay(invariant)
        m = bind(gdef, full)
        _set_inside_transform(True)
        try:
            y = fn(m, x)
        finally:
            _set_inside_transform(False)
        _, new_state = export(m)
        new_carry, new_invariant = (
            (State({}), new_state) if mutable_sel is None else mutable_sel.partition_state(m, new_state)
        )
        _check_invariant_equal(invariant, new_invariant)
        assert_state_unchanged(m, invariant, new_invariant)
        return new_carry, cast(object, y)

    final_carry, ys = lax.scan(step, carry_state, xs, length=length, unroll=unroll)
    apply_mutations(
        [_ModuleRef("arg", 0, init_module, gdef, state)],
        [final_carry.overlay(invariant)],
        mutable_sel,
    )
    return ys


def _check_invariant_equal(a: State, b: State) -> None:
    """Assert two states share the same ``(collection, path)`` key set.

    Used by :func:`scan` and the control-flow primitives in
    :mod:`spectrax.transforms.control_flow` to catch the case where a
    user's body added or removed entries from the supposedly-invariant
    portion of the module state across iterations / branches. Only the
    key set is compared because value equality on tracers is undefined
    during JAX tracing — per-leaf identity is checked separately by
    :func:`~spectrax.transforms.split_merge.assert_state_unchanged`.

    Args:
        a: Reference state captured before the loop / branch.
        b: State observed after one iteration / branch run.

    Raises:
        ValueError: If the key sets differ.
    """
    a_keys = {(c, p) for c, p in a.paths()}
    b_keys = {(c, p) for c, p in b.paths()}
    if a_keys != b_keys:
        raise ValueError(
            "scan() invariant state changed across iterations. Declare the changing collection via `mutable=...`."
        )
