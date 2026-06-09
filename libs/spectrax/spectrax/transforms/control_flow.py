# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Module-aware lifts of :mod:`jax.lax` control-flow primitives.

This module exports the spectrax counterparts of
:func:`jax.lax.cond`, :func:`jax.lax.switch`, :func:`jax.lax.while_loop`,
:func:`jax.lax.fori_loop`, plus :func:`remat_scan` (a
:func:`jax.checkpoint`-wrapped variant of :func:`~spectrax.scan`).

All wrappers use the same recipe:

1. Export the module to ``(GraphDef, State)``.
2. Partition the state by the ``mutable`` selector into a *carry*
   (collections that may be written) and an *invariant* (everything
   else).
3. Wrap the user-provided body into a pure ``(carry_state, ...) ->
   (new_carry, ...)`` form by binding a fresh module from the merged
   state at every call.
4. Hand the pure body to the underlying :mod:`jax.lax` primitive.
5. After the primitive returns, overlay the final carry on top of the
   invariant and write the merged state back to the live module via
   :func:`~spectrax.transforms.split_merge.apply_mutations`.

When ``mutable=()`` (the readonly default), the wrappers take a
fast path that uses
:func:`~spectrax.transforms.split_merge.make_direct_readonly` to
operate on the live module pytree directly and forbid every variable
write — :func:`~spectrax.transforms.split_merge.apply_mutations` is
not invoked at all.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import jax
import jax.lax as lax

from ..core.graph import bind, export
from ..core.module import Module, _set_inside_transform
from ..core.selector import SelectorSugar
from ..core.state import State
from .scan import _check_invariant_equal
from .split_merge import _ModuleRef, apply_mutations, assert_state_unchanged, make_direct_readonly, resolve_mutable

__all__ = ["cond", "fori_loop", "remat_scan", "switch", "while_loop"]


def _find_modulelists(module: Module, prefix: tuple[str, ...] = ()):
    """Recursively yield ``(attr_path, ModuleList)`` for every nested ``ModuleList``.

    Used by :func:`fori_loop` to pre-compute and inject stacked-state
    caches for every :class:`~spectrax.ModuleList` it can find inside
    ``init_module``, so per-iteration ``bind`` calls don't pay the cost
    of re-stacking the constituent modules.

    Args:
        module: The root module to walk.
        prefix: Attribute-name path prefix accumulated by recursion;
            callers should leave at the default.

    Yields:
        ``((attr_name, …), modulelist)`` pairs identifying each
        :class:`~spectrax.ModuleList` inside ``module`` by its
        attribute path from the root.
    """
    from ..core.containers import ModuleList

    for name in module._spx_attr_order:
        value = getattr(module, name)
        if isinstance(value, ModuleList):
            yield (*prefix, name), value
        elif isinstance(value, Module):
            yield from _find_modulelists(value, (*prefix, name))


def _inject_traced_caches(module: Module, caches: dict[tuple[str, ...], tuple[object]]) -> None:
    """Attach pre-computed ``(gdef, stacked_state)`` caches to nested ``ModuleList`` instances.

    For each ``(path, cache)`` entry, walks ``module`` along ``path`` to
    locate the corresponding :class:`~spectrax.ModuleList` and stores
    the cache on its ``_spx_traced_cache`` slot via
    :func:`object.__setattr__` (bypassing the normal frozen-dataclass
    machinery). Subsequent ``bind`` operations on the
    :class:`~spectrax.ModuleList` reuse this cache instead of
    re-stacking its items, which is critical for performance inside a
    ``lax.fori_loop`` body.

    Args:
        module: The freshly-bound module to mutate.
        caches: Mapping from attribute-path tuple to ``(gdef, stacked)``.
    """
    for path, cache in caches.items():
        target = module
        for attr in path:
            target = getattr(target, attr)
        object.__setattr__(target, "_spx_traced_cache", cache)


def _run_branch(
    gdef: object,
    state: State,
    fn: Callable[..., object],
    operands: tuple[object, ...],
) -> tuple[object, State]:
    """Rebind a fresh module from ``(gdef, state)``, run the branch body, and re-export.

    Helper for the :func:`cond` and :func:`switch` branch wrappers. The
    inside-transform thread-local is set for the duration of ``fn`` so
    the variable-write path knows it is running inside a transform,
    then the resulting state is re-exported from the bound module so
    captured mutations show up as new leaves.

    Args:
        gdef: Graph-def of the original module.
        state: Merged carry+invariant state for this branch invocation.
        fn: The user's branch body, called as ``fn(module, *operands)``.
        operands: Positional operands forwarded to the branch body.

    Returns:
        A pair ``(branch_output, new_state)``.
    """
    m = bind(gdef, state)
    _set_inside_transform(True)
    try:
        y = fn(m, *operands)
    finally:
        _set_inside_transform(False)
    _, new_state = export(m)
    return y, new_state


def cond(
    pred: object,
    on_true: Callable[..., object],
    on_false: Callable[..., object],
    module: Module,
    *operands: object,
    mutable: SelectorSugar = (),
) -> object:
    """Module-aware :func:`jax.lax.cond`.

    Selects between ``on_true`` and ``on_false`` based on ``pred``,
    threading ``module`` through both branches. Each branch is called
    as ``branch(module, *operands)``. The two branches must produce
    output pytrees of the same structure and dtypes — JAX requires
    this — *and* their mutation footprints must agree: the
    ``(collection, path)`` set of variables they touch must be
    identical, and every collection outside the ``mutable`` selector
    must remain structurally invariant between branches.

    When ``mutable=()`` the implementation takes a fast path that runs
    each branch on the live module directly, guarded by
    :func:`~spectrax.transforms.split_merge.make_direct_readonly` so
    any direct module write is rejected at trace time. Otherwise the
    state is partitioned into carry+invariant by ``mutable``'s
    selector, both branches are wrapped into pure
    ``(carry, ops) -> (y, new_carry)`` bodies, and after
    :func:`jax.lax.cond` returns the final carry is overlaid back onto
    the invariant and written to the live module via
    :func:`~spectrax.transforms.split_merge.apply_mutations`.

    Args:
        pred: Predicate for the cond.
        on_true: Branch to run when ``pred`` is true.
        on_false: Branch to run when ``pred`` is false.
        module: Module passed to each branch as the first positional
            arg.
        *operands: Extra operands forwarded as ``(module, *operands)``
            to the chosen branch.
        mutable: Selector controlling which collections may be written
            during the branches.

    Returns:
        The branch output.

    Raises:
        ValueError: If the branches differ in their mutation footprint
            or attempt to mutate a collection outside ``mutable``.
        IllegalMutationError: If a branch writes to the live module on
            the readonly fast path.
    """
    mutable_sel = resolve_mutable(mutable)
    gdef, state = export(module)
    if mutable_sel is None:
        invariant_msg = (
            "cond() invariant state changed across branches. Declare the changing collection via `mutable=...`."
        )
        t_fn = make_direct_readonly(
            lambda ops: on_true(module, *ops),
            explicit_modules=(module,),
            structural_error_message=invariant_msg,
        )
        f_fn = make_direct_readonly(
            lambda ops: on_false(module, *ops),
            explicit_modules=(module,),
            structural_error_message=invariant_msg,
        )
        return lax.cond(pred, t_fn, f_fn, operands)
    carry, invariant = mutable_sel.partition_state(module, state)

    def _wrap(branch: Callable[..., object]) -> Callable[[State, tuple[object, ...]], tuple[object, State]]:
        """Wrap a user branch into a ``(carry, ops) -> (y, new_carry)`` ``lax.cond`` body.

        The wrapper merges the invariant state back in before running
        the branch, then re-partitions the resulting state into
        carry+invariant and verifies that the invariant didn't change
        (if it had, the user's ``mutable=`` declaration was
        incomplete, which would corrupt cross-branch state).

        Args:
            branch: Branch value consumed by this operation.

        Returns:
            Result described by this helper.
        """

        def wrapped(c: State, ops: tuple[object, ...]) -> tuple[object, State]:
            """Merge invariant+carry, run the branch, re-partition the new state.

            Single iteration of the cond body: overlay the captured
            invariant on the carry, run the user branch via
            :func:`_run_branch`, partition the resulting state, verify
            the invariant did not change, and return the new carry plus
            the branch output.

            Args:
                c: C value consumed by this operation.
                ops: Ops value consumed by this operation.

            Returns:
                Result described by this helper.
            """
            full = c.overlay(invariant)
            y, new_state = _run_branch(gdef, full, branch, ops)
            new_c, new_inv = (
                (State({}), new_state)
                if mutable_sel is None
                else mutable_sel.partition_state(
                    bind(gdef, new_state),
                    new_state,
                )
            )
            _check_invariant_equal(invariant, new_inv)
            assert_state_unchanged(module, invariant, new_inv)
            return y, new_c

        return wrapped

    t_fn = _wrap(on_true)
    f_fn = _wrap(on_false)
    (y, new_carry) = lax.cond(pred, t_fn, f_fn, carry, operands)
    apply_mutations(
        [_ModuleRef("arg", 0, module, gdef, state)],
        [new_carry.overlay(invariant)],
        mutable_sel,
    )
    return y


def switch(
    index: object,
    branches: Sequence[Callable[..., object]],
    module: Module,
    *operands: object,
    mutable: SelectorSugar = (),
) -> object:
    """Module-aware :func:`jax.lax.switch`.

    Selects ``branches[index]`` and runs it as
    ``branches[index](module, *operands)``. Same constraints as
    :func:`cond` apply, generalized to ``len(branches)`` branches:
    every branch must produce the same output structure and the same
    mutation footprint; any collection outside ``mutable`` must remain
    structurally identical across all branches.

    The readonly fast path (``mutable=()``) wraps each branch in
    :func:`~spectrax.transforms.split_merge.make_direct_readonly`; the
    state-partitioned slow path threads a carry through
    :func:`jax.lax.switch` and writes back via
    :func:`~spectrax.transforms.split_merge.apply_mutations`.

    Args:
        index: Integer-valued index selecting the branch.
        branches: Sequence of branch callables.
        module: Module passed to each branch as the first positional
            arg.
        *operands: Extra operands forwarded to the chosen branch.
        mutable: Selector controlling which collections may be written.

    Returns:
        The chosen branch's output.

    Raises:
        ValueError: If ``branches`` is empty or branches disagree on
            invariant structure.
        IllegalMutationError: If a branch writes to the live module on
            the readonly fast path.
    """
    if not branches:
        raise ValueError("switch() requires at least one branch")
    mutable_sel = resolve_mutable(mutable)
    gdef, state = export(module)
    if mutable_sel is None:
        invariant_msg = (
            "switch() invariant state changed across branches. Declare the changing collection via `mutable=...`."
        )
        wrapped_branches = [
            make_direct_readonly(
                lambda ops, branch=branch: branch(module, *ops),
                explicit_modules=(module,),
                structural_error_message=invariant_msg,
            )
            for branch in branches
        ]
        return lax.switch(index, wrapped_branches, operands)
    carry, invariant = mutable_sel.partition_state(module, state)

    def _wrap(branch: Callable[..., object]) -> Callable[[State, tuple[object, ...]], tuple[object, State]]:
        """Wrap a user branch into a ``(carry, ops) -> (y, new_carry)`` ``lax.switch`` body.

        Same semantics as the :func:`cond` wrapper: merge invariant
        state, run the branch, re-partition, verify that the invariant
        survived. Per-branch closures capture the branch by default
        argument so the loop variable doesn't leak between branches.

        Args:
            branch: Branch value consumed by this operation.

        Returns:
            Result described by this helper.
        """

        def wrapped(c: State, ops: tuple[object, ...]) -> tuple[object, State]:
            """Merge invariant+carry, run the branch, re-partition the resulting state.

            Per-branch :func:`jax.lax.switch` body: behaves identically
            to the body inside :func:`cond`, but a default-argument
            capture pins the per-branch closure so the loop variable
            does not leak between branch wrappers.

            Args:
                c: C value consumed by this operation.
                ops: Ops value consumed by this operation.

            Returns:
                Result described by this helper.
            """
            full = c.overlay(invariant)
            y, new_state = _run_branch(gdef, full, branch, ops)
            new_c, new_inv = (
                (State({}), new_state)
                if mutable_sel is None
                else mutable_sel.partition_state(
                    bind(gdef, new_state),
                    new_state,
                )
            )
            _check_invariant_equal(invariant, new_inv)
            assert_state_unchanged(module, invariant, new_inv)
            return y, new_c

        return wrapped

    wrapped_branches = [_wrap(b) for b in branches]
    (y, new_carry) = lax.switch(index, wrapped_branches, carry, operands)
    apply_mutations(
        [_ModuleRef("arg", 0, module, gdef, state)],
        [new_carry.overlay(invariant)],
        mutable_sel,
    )
    return y


def while_loop(
    cond_fn: Callable[[Module]],
    body_fn: Callable[[Module]],
    init_module: Module,
    init_carry: object,
    *,
    mutable: SelectorSugar = (),
) -> object:
    """Module-aware :func:`jax.lax.while_loop`.

    Iterates ``body_fn(module, user_carry)`` until
    ``cond_fn(module, user_carry)`` returns false. Internally the loop
    threads a tuple ``(state_carry, user_carry)``: ``state_carry`` is
    the declared-mutable portion of the module's state, and
    ``user_carry`` is the user's loop carry value. On each iteration a
    fresh module is rebound from ``state_carry.overlay(invariant)``
    before invoking either user function.

    The body's invariant footprint is checked after each iteration via
    :func:`~spectrax.transforms.scan._check_invariant_equal` and
    :func:`~spectrax.transforms.split_merge.assert_state_unchanged`;
    any drift means the user's ``mutable=`` declaration is incomplete.

    The ``mutable=()`` fast path runs ``cond_fn`` / ``body_fn`` against
    the live module under
    :func:`~spectrax.transforms.split_merge.make_direct_readonly`, with
    no state-carry threaded through the loop.

    Args:
        cond_fn: Loop predicate ``(module, user_carry) -> bool``.
        body_fn: Loop body ``(module, user_carry) -> new_user_carry``.
        init_module: Initial :class:`~spectrax.Module` value.
        init_carry: Initial user carry value.
        mutable: Selector controlling which collections may be written.

    Returns:
        The final ``user_carry`` after the loop terminates.

    Raises:
        ValueError: If the invariant portion of the state changes
            structurally across iterations.
        IllegalMutationError: If a direct write to the live module
            occurs on the readonly fast path, or if a mutation lands
            outside the ``mutable`` selector.
    """
    mutable_sel = resolve_mutable(mutable)
    gdef, state = export(init_module)
    if mutable_sel is None:
        invariant_msg = (
            "while_loop() invariant state changed across iterations. Declare the changing collection via `mutable=...`."
        )
        cond_wrap = make_direct_readonly(
            lambda user_carry: cond_fn(init_module, user_carry),
            explicit_modules=(init_module,),
            structural_error_message=invariant_msg,
        )
        body_wrap = make_direct_readonly(
            lambda user_carry: body_fn(init_module, user_carry),
            explicit_modules=(init_module,),
            structural_error_message=invariant_msg,
        )
        return lax.while_loop(cond_wrap, body_wrap, init_carry)
    carry_state, invariant = mutable_sel.partition_state(init_module, state)

    def cond_wrap(loop_carry: tuple[State]) -> object:
        """:func:`jax.lax.while_loop` predicate.

        Unpacks the ``(state_carry, user_carry)`` loop tuple, overlays
        the captured invariant on ``state_carry``, rebinds a fresh
        module, and calls the user's ``cond_fn`` inside the
        inside-transform thread-local. Returns whatever boolean-shaped
        scalar the user produced.

        Args:
            loop_carry: Loop carry value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        c_state, uc = loop_carry
        full = c_state.overlay(invariant)
        m = bind(gdef, full)
        _set_inside_transform(True)
        try:
            return cond_fn(m, uc)
        finally:
            _set_inside_transform(False)

    def body_wrap(loop_carry: tuple[State, object]) -> tuple[State, object]:
        """:func:`jax.lax.while_loop` body.

        Same module-binding dance as :func:`cond_wrap`: merge invariant
        onto the carry, rebind, run the user's ``body_fn``, re-export
        the resulting state, partition into new carry+invariant, verify
        the invariant did not drift, and return the new
        ``(state_carry, user_carry)`` tuple.

        Args:
            loop_carry: Loop carry value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        c_state, uc = loop_carry
        full = c_state.overlay(invariant)
        m = bind(gdef, full)
        _set_inside_transform(True)
        try:
            new_uc = body_fn(m, uc)
        finally:
            _set_inside_transform(False)
        _, new_state = export(m)
        new_c, new_inv = (State({}), new_state) if mutable_sel is None else mutable_sel.partition_state(m, new_state)
        _check_invariant_equal(invariant, new_inv)
        assert_state_unchanged(init_module, invariant, new_inv)
        return new_c, cast(object, new_uc)

    final_c, final_uc = lax.while_loop(cond_wrap, body_wrap, (carry_state, init_carry))
    apply_mutations(
        [_ModuleRef("arg", 0, init_module, gdef, state)],
        [final_c.overlay(invariant)],
        mutable_sel,
    )
    return final_uc


def fori_loop(
    lower: int,
    upper: int,
    body_fn: Callable[[int, Module]],
    init_module: Module,
    init_carry: object,
    *,
    mutable: SelectorSugar = (),
) -> object:
    """Module-aware :func:`jax.lax.fori_loop`.

    Iterates ``body_fn(i, module, user_carry)`` for ``i`` in
    ``[lower, upper)``. Matches the carry-threading semantics of
    :func:`while_loop`: the loop carry is internally
    ``(state_carry, user_carry)``, with the module rebound each iteration
    from the merged ``state_carry.overlay(invariant)``.

    As an optimization, every nested :class:`~spectrax.ModuleList` in
    ``init_module`` is pre-stacked once via
    :func:`~spectrax.containers._stack_module_states` and the resulting
    cache is injected onto each per-iteration module via
    :func:`_inject_traced_caches`, so ``bind`` does not re-stack the
    list elements on every step.

    Three execution paths:

    * ``mutable=()`` and no ``ModuleList``: pure direct-readonly call
      against ``init_module``.
    * ``mutable=()`` with at least one ``ModuleList``: rebind per
      iteration with cache injection but verify the entire state stays
      identical.
    * ``mutable!=()``: full carry+invariant split with write-back to
      ``init_module`` after the loop.

    Args:
        lower: Inclusive lower bound for the loop counter.
        upper: Exclusive upper bound for the loop counter.
        body_fn: Loop body ``(i, module, user_carry) -> new_user_carry``.
        init_module: Initial :class:`~spectrax.Module`.
        init_carry: Initial user carry.
        mutable: Selector controlling which collections may be written.

    Returns:
        The final ``user_carry`` after ``upper - lower`` iterations.

    Raises:
        ValueError: If the invariant portion of the state changes
            structurally across iterations.
        IllegalMutationError: If a direct write to the live module
            occurs on the readonly fast path, or if a mutation lands
            outside the ``mutable`` selector.
    """
    mutable_sel = resolve_mutable(mutable)
    gdef, state = export(init_module)

    from ..core.containers import _stack_module_states

    modulelist_caches: dict[tuple[str, ...], tuple[object]] = {}
    for path, ml in _find_modulelists(init_module):
        if not ml._spx_items:
            continue
        gdef_ml, stacked = _stack_module_states(ml._spx_items, context="fori_loop ModuleList cache")
        modulelist_caches[path] = (gdef_ml, stacked)

    if mutable_sel is None and not modulelist_caches:
        body_wrap = make_direct_readonly(
            lambda i, user_carry: body_fn(i, init_module, user_carry),
            explicit_modules=(init_module,),
            structural_error_message="fori_loop() invariant state changed across iterations. Declare the changing collection via `mutable=...`.",
        )
        return lax.fori_loop(lower, upper, body_wrap, init_carry)
    if mutable_sel is None:

        def body_wrap(i: object, user_carry: object) -> object:
            """``fori_loop`` body for the no-mutation-but-has-ModuleList path.

            Binds a fresh module from the original ``state``, injects
            the pre-stacked :class:`~spectrax.ModuleList` caches via
            :func:`_inject_traced_caches`, runs ``body_fn(i, module,
            user_carry)`` with the inside-transform flag set, and
            asserts that the entire state survived unchanged before
            returning the new user carry.

            Args:
                i: I value consumed by this operation.
                user_carry: User carry value consumed by this operation.

            Returns:
                Result described by this helper.
            """
            m = bind(gdef, state)
            if modulelist_caches:
                _inject_traced_caches(m, modulelist_caches)
            _set_inside_transform(True)
            try:
                new_user_carry = body_fn(i, m, user_carry)
            finally:
                _set_inside_transform(False)
            _, new_state = export(m)
            _check_invariant_equal(state, new_state)
            assert_state_unchanged(init_module, state, new_state)
            return new_user_carry

        return lax.fori_loop(lower, upper, body_wrap, init_carry)
    carry_state, invariant = mutable_sel.partition_state(init_module, state)

    def body_wrap(i: object, loop_carry: tuple[State, object]) -> tuple[State, object]:
        """``fori_loop`` body for the mutable-state path.

        Unpacks ``(state_carry, user_carry)``, binds a fresh module
        from the merged state with :class:`~spectrax.ModuleList` caches
        injected, runs ``body_fn(i, module, user_carry)``, and
        re-partitions the resulting state into a new carry plus
        invariant. Verifies the invariant did not drift before
        returning the new ``(state_carry, user_carry)`` tuple.

        Args:
            i: I value consumed by this operation.
            loop_carry: Loop carry value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        c_state, uc = loop_carry
        full = c_state.overlay(invariant)
        m = bind(gdef, full)
        if modulelist_caches:
            _inject_traced_caches(m, modulelist_caches)
        _set_inside_transform(True)
        try:
            new_uc = body_fn(i, m, uc)
        finally:
            _set_inside_transform(False)
        _, new_state = export(m)
        new_c, new_inv = (State({}), new_state) if mutable_sel is None else mutable_sel.partition_state(m, new_state)
        _check_invariant_equal(invariant, new_inv)
        assert_state_unchanged(init_module, invariant, new_inv)
        return new_c, cast(object, new_uc)

    final_c, final_uc = lax.fori_loop(lower, upper, body_wrap, (carry_state, init_carry))
    apply_mutations(
        [_ModuleRef("arg", 0, init_module, gdef, state)],
        [final_c.overlay(invariant)],
        mutable_sel,
    )
    return final_uc


def remat_scan(
    fn: Callable[[Module]],
    init_module: Module,
    xs: object,
    *,
    length: int | None = None,
    mutable: SelectorSugar = (),
    policy: Callable[..., object] | None = None,
    prevent_cse: bool = True,
    unroll: int = 1,
) -> object:
    """Scan ``fn`` with each step run under :func:`jax.checkpoint`.

    Equivalent to :func:`~spectrax.scan` (``fn(module, x) -> y``) but
    every step body is additionally wrapped in :func:`jax.checkpoint`
    so the forward pass only stores activations at step boundaries.
    Backward then recomputes the per-step activations on demand. This
    is the standard pattern for very deep transformer stacks where
    naively storing every layer's activations would exhaust device
    memory.

    Same carry+invariant split as :func:`~spectrax.scan`: collections
    in ``mutable`` form the carry that is threaded through
    :func:`jax.lax.scan`; everything else must remain structurally
    invariant per step. Final carry is written back to ``init_module``
    via :func:`~spectrax.transforms.split_merge.apply_mutations`.

    Args:
        fn: Per-step body ``(module, x) -> y``.
        init_module: Module providing the initial state.
        xs: Scanned sequence (pytree with a leading axis).
        length: Optional explicit sequence length (forwarded to
            :func:`jax.lax.scan`).
        mutable: Selector controlling which collections may be carried.
        policy: Optional :func:`jax.checkpoint` save policy.
        prevent_cse: Forwarded to :func:`jax.checkpoint`.
        unroll: Forwarded to :func:`jax.lax.scan`.

    Returns:
        The stacked per-step ``y`` outputs.

    Raises:
        ValueError: If the invariant portion of the state changes
            structurally across iterations.
        IllegalMutationError: If a direct write to the live module
            occurs on the readonly fast path, or if a mutation lands
            outside the ``mutable`` selector.
    """
    mutable_sel = resolve_mutable(mutable)
    if mutable_sel is None:
        step = make_direct_readonly(
            lambda _carry, x: (None, fn(init_module, x)),
            explicit_modules=(init_module,),
            structural_error_message="scan() invariant state changed across iterations. Declare the changing collection via `mutable=...`.",
        )
        rematted = jax.checkpoint(step, policy=policy, prevent_cse=prevent_cse)
        _carry, ys = lax.scan(rematted, None, xs, length=length, unroll=unroll)
        return ys
    gdef, state = export(init_module)
    carry_state, invariant = mutable_sel.partition_state(init_module, state)

    def step(c: State, x: object) -> tuple[State, object]:
        """Single ``remat_scan`` step body, wrapped in :func:`jax.checkpoint` by the caller.

        Overlays the captured invariant on the current carry, rebinds a
        fresh module, runs the user's ``fn(module, x)`` inside the
        inside-transform thread-local, re-exports the resulting state,
        partitions it into a new carry plus invariant, and verifies
        the invariant did not drift.

        Args:
            c: C value consumed by this operation.
            x: Input value consumed by the operation.

        Returns:
            Result described by this helper.
        """
        full = c.overlay(invariant)
        m = bind(gdef, full)
        _set_inside_transform(True)
        try:
            y = fn(m, x)
        finally:
            _set_inside_transform(False)
        _, new_state = export(m)
        new_c, new_inv = (State({}), new_state) if mutable_sel is None else mutable_sel.partition_state(m, new_state)
        _check_invariant_equal(invariant, new_inv)
        assert_state_unchanged(init_module, invariant, new_inv)
        return new_c, y

    rematted = jax.checkpoint(step, policy=policy, prevent_cse=prevent_cse)
    final_c, ys = lax.scan(rematted, carry_state, xs, length=length, unroll=unroll)
    apply_mutations(
        [_ModuleRef("arg", 0, init_module, gdef, state)],
        [final_c.overlay(invariant)],
        mutable_sel,
    )
    return ys
