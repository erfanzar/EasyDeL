# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The shared split/merge shim used by every spectrax transform.

Every spectrax transform follows the same recipe:

1. Walk ``args`` / ``kwargs`` to locate :class:`~spectrax.Module`
   instances via :func:`locate_and_strip` (or
   :func:`locate_and_strip_fast` on the kwargs-empty hot path).
2. :func:`~spectrax.export` each to produce ``(GraphDef, State)``.
3. Build a pure function that, given the list of states, rebinds fresh
   modules, calls the user function, and captures mutations by
   snapshotting variable ``_value`` identities (:func:`make_pure`).
4. Apply the underlying JAX transform to that pure function.
5. On exit, apply mutations back to the original live modules,
   restricted to the ``mutable`` selector, via :func:`apply_mutations`.

Non-module arguments flow through unchanged via :func:`strip_modules`
and :func:`splice_modules` (they are replaced with ``None`` placeholders
through the pure function and re-spliced on the other side).
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import NoReturn, cast

import jax

from ..core.context import _enter as _ctx_enter
from ..core.errors import IllegalMutationError
from ..core.graph import GraphDef, bind, export, live_variables
from ..core.module import Module, _graph_epoch, _inside_transform, _set_inside_transform
from ..core.selector import Selector, SelectorSugar, as_selector
from ..core.selector import select as _sel
from ..core.state import State
from ..core.variable import _get_write_hook, _set_write_hook

__all__ = [
    "apply_mutations",
    "assert_state_unchanged",
    "locate_and_strip",
    "locate_modules",
    "make_direct_readonly",
    "make_pure",
    "make_pure_ctx",
    "make_pure_readonly",
    "make_pure_readonly_ctx",
    "make_pure_readonly_single_positional",
    "make_pure_single_positional",
    "resolve_mutable",
    "splice_modules",
    "strip_modules",
]


PureFn = Callable[
    [tuple[State, ...], tuple[object, ...], dict[str, object]],
    tuple[object, tuple[State, ...]],
]
"""Type of the pure function produced by :func:`make_pure`."""


@dataclass(slots=True)
class _ModuleRef:
    """A located module inside a transform's argument tree.

    Attributes:
        kind: ``"arg"`` or ``"kwarg"``.
        locator: Integer index (when ``kind == "arg"``) or string key
            (when ``kind == "kwarg"``).
        module: The live module reference.
        gdef: The graph-def snapshotted at :func:`locate_modules`.
        state: The state snapshotted at :func:`locate_modules`.
    """

    kind: str
    locator: int | str
    module: Module
    gdef: GraphDef
    state: State


def locate_modules(args: tuple[object, ...], kwargs: dict[str, object]) -> list[_ModuleRef]:
    """Scan ``args`` / ``kwargs`` and return one :class:`_ModuleRef` per :class:`Module`.

    Each reference carries a freshly-computed ``(gdef, state)`` snapshot
    used as the input to the transform. Positional refs come first, in
    argument order; keyword refs follow in dict-iteration order.

    Args:
        args: Positional arguments forwarded by the wrapper.
        kwargs: Keyword arguments forwarded by the wrapper.

    Returns:
        A list of :class:`_ModuleRef` instances, one per module
        located.
    """
    refs: list[_ModuleRef] = []
    for i, a in enumerate(args):
        if isinstance(a, Module):
            g, s = export(a)
            refs.append(_ModuleRef("arg", i, a, g, s))
    for k, v in kwargs.items():
        if isinstance(v, Module):
            g, s = export(v)
            refs.append(_ModuleRef("kwarg", k, v, g, s))
    return refs


def locate_and_strip_fast(args: tuple[object, ...]) -> tuple[list[_ModuleRef], tuple[object, ...]]:
    """Kwargs-less fast variant of :func:`locate_and_strip`.

    Single pass over positional ``args``; each :class:`Module` is
    exported into a :class:`_ModuleRef` and replaced by ``None`` in the
    stripped tuple. Used by :func:`~spectrax.jit`'s kwargs-empty hot
    path (the common case for training-step wrappers).

    Args:
        args: Positional arguments passed to the wrapped function.

    Returns:
        A pair ``(refs, stripped_args)`` where ``refs`` is a list of
        :class:`_ModuleRef` and ``stripped_args`` is ``args`` with every
        :class:`~spectrax.Module` replaced by ``None``.
    """
    refs: list[_ModuleRef] = []
    stripped: list[object] = []
    for i, a in enumerate(args):
        if isinstance(a, Module):
            g, s = export(a)
            refs.append(_ModuleRef("arg", i, a, g, s))
            stripped.append(None)
        else:
            stripped.append(a)
    return refs, tuple(stripped)


def locate_and_strip(
    args: tuple[object, ...], kwargs: dict[str, object]
) -> tuple[list[_ModuleRef], tuple[object, ...], dict[str, object]]:
    """Locate modules and strip them to ``None`` placeholders in one pass.

    Fused equivalent of :func:`locate_modules` followed by
    :func:`strip_modules`. Each ``args`` / ``kwargs`` sequence is
    iterated exactly once; non-module values flow through to the
    stripped tuple/dict while module values are exported (gdef + state
    snapshot) and recorded in a :class:`_ModuleRef`. Used by every
    module-aware transform wrapper (:func:`~spectrax.jit`,
    :func:`~spectrax.vmap`, :func:`~spectrax.remat`) on the dispatch
    hot path.

    Args:
        args: Positional arguments passed to the wrapped function.
        kwargs: Keyword arguments passed to the wrapped function.

    Returns:
        A triple ``(refs, stripped_args, stripped_kwargs)``:

        * ``refs`` — one :class:`_ModuleRef` per located module, in the
          order ``args`` first then ``kwargs``.
        * ``stripped_args`` — ``args`` with every module replaced by
          ``None`` so the rest passes cleanly through :func:`jax.jit`.
        * ``stripped_kwargs`` — same substitution applied to ``kwargs``.
    """
    refs: list[_ModuleRef] = []
    new_args_list: list[object] = [None] * len(args)
    for i, a in enumerate(args):
        if isinstance(a, Module):
            g, s = export(a)
            refs.append(_ModuleRef("arg", i, a, g, s))
        else:
            new_args_list[i] = a
    new_kwargs: dict[str, object] = {}
    for k, v in kwargs.items():
        if isinstance(v, Module):
            g, s = export(v)
            refs.append(_ModuleRef("kwarg", k, v, g, s))
            new_kwargs[k] = None
        else:
            new_kwargs[k] = v
    return refs, tuple(new_args_list), new_kwargs


def strip_modules(args: tuple[object, ...], kwargs: dict[str, object]) -> tuple[tuple[object, ...], dict[str, object]]:
    """Replace :class:`Module` entries with ``None`` placeholders.

    Used by callers that have already located the modules via
    :func:`locate_modules` and want a separate args/kwargs pair with
    the modules removed. The placeholders flow through the pure
    function as ordinary JAX pytree inputs; the corresponding modules
    are passed alongside as states. Order is preserved so
    :func:`splice_modules` can reverse the operation.

    Args:
        args: Positional arguments.
        kwargs: Keyword arguments.

    Returns:
        ``(stripped_args, stripped_kwargs)`` with every module slot set
        to ``None``.
    """
    new_args = tuple(None if isinstance(a, Module) else a for a in args)
    new_kwargs = {k: None if isinstance(v, Module) else v for k, v in kwargs.items()}
    return new_args, new_kwargs


def splice_modules(
    stripped_args: tuple[object, ...],
    stripped_kwargs: dict[str, object],
    refs: list[_ModuleRef],
    modules: list[Module],
) -> tuple[tuple[object, ...], dict[str, object]]:
    """Write freshly-bound ``modules`` back into the stripped-arg tree.

    The inverse of :func:`strip_modules`. Iterates ``refs`` and
    ``modules`` in lockstep and writes each rebound module into the
    positional or keyword slot recorded in the corresponding ref. The
    ``modules`` list typically comes from :func:`bind` calls inside a
    pure body once it has received the per-module states from JAX.

    Args:
        stripped_args: Positional args with module slots set to ``None``.
        stripped_kwargs: Keyword args with module slots set to ``None``.
        refs: Located module refs from :func:`locate_modules` /
            :func:`locate_and_strip`.
        modules: Newly-bound modules to splice in, parallel to ``refs``.

    Returns:
        ``(args, kwargs)`` with every module slot replaced by the
        corresponding entry in ``modules``.
    """
    args = list(stripped_args)
    kwargs = dict(stripped_kwargs)
    for ref, m in zip(refs, modules, strict=False):
        if ref.kind == "arg":
            args[ref.locator] = m
        else:
            kwargs[ref.locator] = m
    return tuple(args), kwargs


def _run_pure_body(
    fn: Callable[..., object],
    gdefs: tuple[GraphDef, ...],
    orig_modules: tuple[Module, ...],
    refs: list[_ModuleRef],
    states: tuple[State, ...],
    stripped_args: tuple[object, ...],
    stripped_kwargs: dict[str, object],
    ctx: dict[str, object] | None,
) -> tuple[object, tuple[State, ...]]:
    """Shared trace-time body of :func:`make_pure` and :func:`make_pure_ctx`.

    Rebinds fresh modules from ``states``, snapshots each live
    variable's ``_value`` identity, invokes ``fn``, and re-packs only
    the mutated leaves into ``new_states``. When ``ctx`` is provided
    (the :func:`make_pure_ctx` path), the invocation is wrapped in
    :func:`_ctx_enter` so :func:`~spectrax.scope` lookups inside the
    trace see the traced values instead of concrete ones.

    Args:
        fn: Callable being wrapped, traced, transformed, or executed.
        gdefs: Gdefs value consumed by this operation.
        orig_modules: Orig modules value consumed by this operation.
        refs: Refs value consumed by this operation.
        states: States value consumed by this operation.
        stripped_args: Stripped args value consumed by this operation.
        stripped_kwargs: Stripped kwargs value consumed by this operation.
        ctx: Ctx value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    modules = [bind(g, s) for g, s in zip(gdefs, states, strict=False)]
    for src, dst in zip(orig_modules, modules, strict=False):
        _copy_runtime_state(src, dst)
    initial: list[list[tuple[str]]] = []
    for m in modules:
        entries: list[tuple[str]] = []
        for path, v in live_variables(m):
            entries.append((path, v, v._value))
        initial.append(entries)
    fargs, fkwargs = splice_modules(stripped_args, stripped_kwargs, refs, modules)
    _set_inside_transform(True)
    try:
        if ctx is None:
            out = fn(*fargs, **fkwargs)
        else:
            with _ctx_enter(ctx):
                out = fn(*fargs, **fkwargs)
    finally:
        _set_inside_transform(False)
    new_states: list[State] = []
    for entries in initial:
        changed: dict[str, dict[str, object]] = {}
        for path, var, init_val in entries:
            cur = var._value
            if cur is not init_val:
                changed.setdefault(var.kind, {})[path] = cur
        new_states.append(State(changed))
    return out, tuple(new_states)


def _raise_illegal_mutation(collection: str, path: str) -> NoReturn:
    """Raise the standard undeclared-mutation error.

    Centralized helper so every callsite produces an identically-worded
    :class:`~spectrax.IllegalMutationError`. The message points users
    at the ``mutable=`` selector knob that would have allowed the
    write.

    Args:
            collection: The variable's collection name (``var.kind``).
            path: Dotted attribute path to the variable.

    Raises:
            IllegalMutationError: Always.

    Returns:
        Result described by this helper.
    """
    raise IllegalMutationError(
        f"Collection {collection!r} at path {path!r} changed under a "
        f"transform but was not declared mutable. Add it to `mutable=`."
    )


def make_direct_readonly(
    fn: Callable[..., object],
    *,
    explicit_modules: tuple[Module, ...] | None = None,
    structural_error_message: str | None = None,
) -> Callable[..., object]:
    """Wrap ``fn`` so any direct module write raises :class:`IllegalMutationError`.

    Used by transforms that bypass the ``(GraphDef, State)`` round-trip
    on the immutable fast path — :func:`~spectrax.jit` (when
    ``mutable=()`` and modules go straight to :func:`jax.jit`),
    :func:`~spectrax.scan` / :func:`~spectrax.cond` /
    :func:`~spectrax.switch` / :func:`~spectrax.while_loop` /
    :func:`~spectrax.fori_loop` body bodies, and the direct
    :func:`~spectrax.vjp` / :func:`~spectrax.value_and_grad` paths.

    On every call the wrapper:

    1. Snapshots the graph epoch and the ``(path, kind, var, _value)``
       identity tuple of every live variable in either ``explicit_modules``
       (if supplied) or every :class:`Module` it finds in
       ``args``/``kwargs``.
    2. Installs a write hook (via :func:`_set_write_hook`) that raises
       on any direct ``var.value = ...`` assignment.
    3. Sets the inside-transform thread-local while ``fn`` runs.
    4. Restores the previous hook and inside-transform flag in a
       ``finally`` block.
    5. Asserts the global graph epoch did not advance (no structural
       mutations) and that every snapshot's ``_value`` is still
       identity-equal to its original.

    ``structural_error_message`` overrides the default
    "module structure changed" wording — used by the control-flow
    wrappers to direct users at ``mutable=...`` instead.

    Args:
        fn: The user function to guard.
        explicit_modules: Optional tuple of module instances to guard;
            when omitted the wrapper auto-discovers modules in the
            args/kwargs of each call.
        structural_error_message: Custom error message for structural
            mutations.

    Returns:
        A guarded callable with the same signature as ``fn``.
    """

    @functools.wraps(fn)
    def guarded(*args: object, **kwargs: object) -> object:
        """Variable-write-protected wrapper around ``fn``.

        Snapshots all live variables of the module(s) involved,
        installs a write hook that raises on any direct mutation,
        runs ``fn``, and asserts that no value or structural change
        slipped through. Module collections that the caller wants
        to mutate must use the explicit :func:`apply_mutations`
        path instead.

        Args:
            *args: Additional positional arguments forwarded to the wrapped callable or backend.
            **kwargs: Additional keyword arguments forwarded to the wrapped callable or backend.

        Returns:
            Result described by this helper.
        """
        modules = list(explicit_modules) if explicit_modules is not None else []
        if explicit_modules is None:
            for value in args:
                if isinstance(value, Module):
                    modules.append(value)
            for value in kwargs.values():
                if isinstance(value, Module):
                    modules.append(value)

        epoch_before = _graph_epoch()
        snapshots: list[tuple[str, str]] = []
        for module in modules:
            for path, var in live_variables(module):
                snapshots.append((path, var.kind, var, var._value))

        prev_hook = _get_write_hook()
        prev_active = _inside_transform()

        def readonly_hook(var: object, _new: object) -> bool:
            """Write hook installed during a readonly transform — every write is an error.

            Looks up the variable in the pre-recorded snapshots so the
            error message includes the variable's path; falls back to
            ``"<unknown>"`` for variables that were created during the
            wrapped function (which is itself a structural mutation).

            Args:
                var: Var value consumed by this operation.
                _new:  new value consumed by this operation.

            Returns:
                Result described by this helper.
            """
            for path, kind, snap_var, _initial in snapshots:
                if snap_var is var:
                    _raise_illegal_mutation(kind, path)
            _raise_illegal_mutation(var.kind, "<unknown>")

        _set_write_hook(readonly_hook)
        _set_inside_transform(True)
        try:
            out = fn(*args, **kwargs)
        finally:
            _set_inside_transform(prev_active)
            _set_write_hook(prev_hook)

        if _graph_epoch() != epoch_before:
            if structural_error_message is not None:
                raise ValueError(structural_error_message)
            raise IllegalMutationError(
                "Module structure changed under a transform but was not declared mutable. "
                "Structural mutations are not supported here."
            )
        for path, kind, var, initial in snapshots:
            if var._value is not initial:
                _raise_illegal_mutation(kind, path)
        return out

    return guarded


def _run_readonly_body(
    fn: Callable[..., object],
    gdefs: tuple[GraphDef, ...],
    orig_modules: tuple[Module, ...],
    refs: list[_ModuleRef],
    states: tuple[State, ...],
    stripped_args: tuple[object, ...],
    stripped_kwargs: dict[str, object],
    ctx: dict[str, object] | None,
) -> object:
    """Trace-time body shared by every readonly pure-fn factory.

    Mirror image of :func:`_run_pure_body` but enforces that no
    variable values change and no graph structure mutates while ``fn``
    runs. Variable identities are snapshotted before invocation;
    afterwards the global graph epoch is checked and every snapshot is
    re-compared, with any drift triggering
    :class:`~spectrax.IllegalMutationError`.

    Args:
        fn: User function being traced.
        gdefs: Per-module graph-defs captured at locate time.
        orig_modules: Original live modules; used by
            :func:`_copy_runtime_state` to forward training-mode flags
            into the rebound modules.
        refs: Located module refs (used by :func:`splice_modules`).
        states: Per-module states JAX traced into the pure call.
        stripped_args: Positional args with module slots set to ``None``.
        stripped_kwargs: Keyword args with module slots set to ``None``.
        ctx: Optional context-frame dict (the
            :func:`make_pure_readonly_ctx` path passes this through).

    Returns:
        Whatever ``fn`` returned.

    Raises:
        IllegalMutationError: If any variable was written or the
            module graph changed structurally during the call.
    """
    modules = [bind(g, s) for g, s in zip(gdefs, states, strict=False)]
    for src, dst in zip(orig_modules, modules, strict=False):
        _copy_runtime_state(src, dst)
    epoch_before = _graph_epoch()
    initial: list[list[tuple[str, str]]] = []
    for m in modules:
        entries: list[tuple[str, str]] = []
        for path, v in live_variables(m):
            entries.append((path, v.kind, v, v._value))
        initial.append(entries)
    fargs, fkwargs = splice_modules(stripped_args, stripped_kwargs, refs, modules)
    _set_inside_transform(True)
    try:
        if ctx is None:
            out = fn(*fargs, **fkwargs)
        else:
            with _ctx_enter(ctx):
                out = fn(*fargs, **fkwargs)
    finally:
        _set_inside_transform(False)
    if _graph_epoch() != epoch_before:
        raise IllegalMutationError(
            "Module structure changed under a transform but was not declared mutable. "
            "Structural mutations are not supported here."
        )
    for entries in initial:
        for path, kind, var, initial_value in entries:
            if var._value is not initial_value:
                _raise_illegal_mutation(kind, path)
    return out


def _run_single_positional_body(
    fn: Callable[..., object],
    gdef: GraphDef,
    orig_module: Module,
    locator: int,
    state: State,
    args_without_module: tuple[object, ...],
    *,
    readonly: bool,
) -> object:
    """Specialized trace body for one positional Module and no kwargs.

    Roughly 2x faster than the general path on the dispatch hot path —
    a single :func:`bind` instead of a tuple, no kwargs handling, no
    splicing of module-shaped placeholders. Used by
    :func:`make_pure_single_positional` and
    :func:`make_pure_readonly_single_positional`.

    Args:
        fn: User function being traced.
        gdef: Graph-def of the single module.
        orig_module: The original live module (for runtime-state
            forwarding).
        locator: Position the module occupied in the user call (so the
            module can be re-inserted at the same index).
        state: The state JAX traced in.
        args_without_module: User positional args with the module slot
            removed.
        readonly: When ``True`` enforces no-mutation semantics; when
            ``False`` returns ``(out, mutated_state)``.

    Returns:
        Either the function output (readonly) or
        ``(output, mutated_state)`` (mutating).

    Raises:
        IllegalMutationError: On undeclared mutations in either mode
            (readonly: any mutation; mutating: structural mutations).
    """
    module = bind(gdef, state)
    _copy_runtime_state(orig_module, module)
    initial: list[tuple[str, str]] = []
    for path, v in live_variables(module):
        initial.append((path, v.kind, v, v._value))
    epoch_before = _graph_epoch()
    fargs = list(args_without_module)
    fargs.insert(locator, module)
    _set_inside_transform(True)
    try:
        out = fn(*fargs)
    finally:
        _set_inside_transform(False)
    if _graph_epoch() != epoch_before:
        raise IllegalMutationError(
            "Module structure changed under a transform but was not declared mutable. "
            "Structural mutations are not supported here."
        )
    if readonly:
        for path, kind, var, initial_value in initial:
            if var._value is not initial_value:
                _raise_illegal_mutation(kind, path)
        return out
    changed: dict[str, dict[str, object]] = {}
    for path, _kind, var, initial_value in initial:
        cur = var._value
        if cur is not initial_value:
            changed.setdefault(var.kind, {})[path] = cur
    return out, State(changed)


def make_pure(fn: Callable[..., object], refs: list[_ModuleRef]) -> PureFn:
    """Construct the pure function fed into :mod:`jax`.

    The returned function has signature
    ``pure(states, stripped_args, stripped_kwargs) -> (out, new_states)``.
    It rebinds fresh :class:`Module` instances from the states, sets
    the thread-local "inside transform" flag, calls ``fn`` with the
    spliced arguments, and captures mutations by comparing each live
    variable's ``_value`` identity against the pre-call snapshot.

    This runs at trace time (once per compile). The trick:
    :func:`~spectrax.bind` creates fresh :class:`~spectrax.Variable`
    instances whose ``_value`` is the incoming tracer from
    ``states``. Before calling ``fn`` we snapshot the ``_value``
    identity of every live variable; after ``fn`` returns, any
    variable whose ``_value`` is no longer identity-equal to its
    snapshot was mutated during the call (a ``var.value = ...``
    assignment rebinds ``_value`` to a new tracer).

    Only those mutated leaves are packed into ``new_states``. The
    unmutated majority is absent from the output pytree entirely,
    so the post-jit :func:`apply_mutations` step has no per-leaf
    comparison work to do on the dispatch hot path. The net effect
    is that a pure forward pass through jit returns an empty state
    tuple for every module.

    Args:
        fn: The user function being traced.
        refs: Module refs produced by :func:`locate_and_strip`.

    Returns:
        A pure callable with signature
        ``(states, stripped_args, stripped_kwargs) -> (out, new_states)``
        consumable by :func:`jax.jit`.
    """
    gdefs = tuple(r.gdef for r in refs)
    orig_modules = tuple(r.module for r in refs)

    def pure(
        states: tuple[State, ...],
        stripped_args: tuple[object, ...],
        stripped_kwargs: dict[str, object],
    ) -> tuple[object, tuple[State, ...]]:
        """JAX-traced body: rebind modules, invoke ``fn``, emit only mutated leaves.

        Args:
            states: States value consumed by this operation.
            stripped_args: Stripped args value consumed by this operation.
            stripped_kwargs: Stripped kwargs value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        return _run_pure_body(fn, gdefs, orig_modules, refs, states, stripped_args, stripped_kwargs, None)

    return pure


def make_pure_readonly(fn: Callable[..., object], refs: list[_ModuleRef]) -> Callable[..., object]:
    """Construct a pure function that rejects all module mutations.

    Same shape as :func:`make_pure` but uses :func:`_run_readonly_body`
    instead of :func:`_run_pure_body`, so any variable write or
    structural mutation triggers
    :class:`~spectrax.IllegalMutationError` at trace time. Returns
    only the user output — there is no second tuple element for
    captured states because none are allowed to change.

    Args:
        fn: The user function being traced.
        refs: Module refs produced by :func:`locate_and_strip`.

    Returns:
        A pure callable with signature
        ``(states, stripped_args, stripped_kwargs) -> out`` consumable
        by :func:`jax.jit`.
    """
    gdefs = tuple(r.gdef for r in refs)
    orig_modules = tuple(r.module for r in refs)

    def pure(
        states: tuple[State, ...],
        stripped_args: tuple[object, ...],
        stripped_kwargs: dict[str, object],
    ) -> object:
        """JAX-traced body: rebind modules read-only, invoke ``fn``, return its output.

        Args:
            states: States value consumed by this operation.
            stripped_args: Stripped args value consumed by this operation.
            stripped_kwargs: Stripped kwargs value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        return _run_readonly_body(fn, gdefs, orig_modules, refs, states, stripped_args, stripped_kwargs, None)

    return pure


def make_pure_single_positional(fn: Callable[..., object], ref: _ModuleRef) -> Callable[..., object]:
    """Specialized pure fn factory for the one-positional-Module hot path.

    The returned callable has shape
    ``pure(state, *args_without_module) -> (out, mutated_state)`` and
    skips the tuple-of-states packaging used by :func:`make_pure`.
    Used on the no-kwargs single-Module dispatch path of
    :func:`~spectrax.jit`, :func:`~spectrax.vmap`, :func:`~spectrax.remat`,
    :func:`~spectrax.vjp`, and :func:`~spectrax.jvp`.

    Args:
        fn: User function being traced.
        ref: The single positional module ref located at dispatch time.

    Returns:
        The pure callable.

    Raises:
        TypeError: If ``ref`` is not a positional ref.
    """
    if ref.kind != "arg":
        raise TypeError("make_pure_single_positional() requires a positional Module ref")
    locator = int(ref.locator)

    def pure(state: State, *args_without_module: object) -> tuple[object, State]:
        """One-module variant: rebind, run ``fn``, return ``(output, mutated_state)``.

        Args:
            state: SpectraX state tree or transform state passed into the operation.
            *args_without_module: Additional positional arguments forwarded to the wrapped callable or backend.

        Returns:
            Result described by this helper.
        """
        return cast(
            tuple[object, State],
            _run_single_positional_body(
                fn,
                ref.gdef,
                ref.module,
                locator,
                state,
                args_without_module,
                readonly=False,
            ),
        )

    return pure


def make_pure_ctx(fn: Callable[..., object], refs: list[_ModuleRef]) -> Callable[..., object]:
    """Variant of :func:`make_pure` that reinstates a scope frame before ``fn``.

    Used by :func:`~spectrax.jit` when the caller has active
    :func:`~spectrax.scope` bindings. The extra ``traced_ctx`` argument
    carries the array-typed scope values as a pytree, which JAX
    lowers to tracers; installing them as a scope frame before
    invoking ``fn`` lets deep ``spx.scope.get(...)`` calls inside the
    traced body resolve to tracer values instead of the concrete
    constants that were live at trace time.

    Signature:
    ``pure(states, traced_ctx, stripped_args, stripped_kwargs) ->
    (out, new_states)``.

    Args:
        fn: The user function being traced.
        refs: Module refs produced by :func:`locate_and_strip`.

    Returns:
        A pure callable consumable by :func:`jax.jit`.
    """
    gdefs = tuple(r.gdef for r in refs)
    orig_modules = tuple(r.module for r in refs)

    def pure(
        states: tuple[State, ...],
        traced_ctx: dict[str, object],
        stripped_args: tuple[object, ...],
        stripped_kwargs: dict[str, object],
    ) -> tuple[object, tuple[State, ...]]:
        """Same shape as :func:`make_pure`'s pure, plus a ``traced_ctx`` arg.

        Args:
            states: States value consumed by this operation.
            traced_ctx: Traced ctx value consumed by this operation.
            stripped_args: Stripped args value consumed by this operation.
            stripped_kwargs: Stripped kwargs value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        return _run_pure_body(fn, gdefs, orig_modules, refs, states, stripped_args, stripped_kwargs, traced_ctx)

    return pure


def make_pure_readonly_ctx(fn: Callable[..., object], refs: list[_ModuleRef]) -> Callable[..., object]:
    """Readonly variant of :func:`make_pure_ctx`.

    Same scope-frame reinstatement as :func:`make_pure_ctx` but routes
    through :func:`_run_readonly_body` so any variable write raises
    :class:`~spectrax.IllegalMutationError` at trace time. Used by the
    scope-aware readonly path of :func:`~spectrax.jit`.

    Args:
        fn: The user function being traced.
        refs: Module refs produced by :func:`locate_and_strip`.

    Returns:
        A pure callable with signature
        ``(states, traced_ctx, stripped_args, stripped_kwargs) -> out``
        consumable by :func:`jax.jit`.
    """
    gdefs = tuple(r.gdef for r in refs)
    orig_modules = tuple(r.module for r in refs)

    def pure(
        states: tuple[State, ...],
        traced_ctx: dict[str, object],
        stripped_args: tuple[object, ...],
        stripped_kwargs: dict[str, object],
    ) -> object:
        """Same shape as :func:`make_pure_readonly`'s ``pure``, plus a ``traced_ctx`` arg.

        Args:
            states: States value consumed by this operation.
            traced_ctx: Traced ctx value consumed by this operation.
            stripped_args: Stripped args value consumed by this operation.
            stripped_kwargs: Stripped kwargs value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        return _run_readonly_body(
            fn,
            gdefs,
            orig_modules,
            refs,
            states,
            stripped_args,
            stripped_kwargs,
            traced_ctx,
        )

    return pure


def make_pure_readonly_single_positional(fn: Callable[..., object], ref: _ModuleRef) -> Callable[..., object]:
    """Readonly counterpart of :func:`make_pure_single_positional`.

    Returns a callable shaped ``pure(state, *args_without_module) -> out``
    (no second tuple element, since no mutations are permitted) that
    delegates to :func:`_run_single_positional_body` with
    ``readonly=True``. Used on the no-kwargs single-Module
    dispatch path whenever ``mutable`` resolves to ``None``.

    Raises:
        TypeError: If ``ref`` is not a positional ref.

    Args:
        fn: Callable being wrapped, traced, transformed, or executed.
        ref: Ref value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    if ref.kind != "arg":
        raise TypeError("make_pure_readonly_single_positional() requires a positional Module ref")
    locator = int(ref.locator)

    def pure(state: State, *args_without_module: object) -> object:
        """One-module readonly variant: rebind, run ``fn``, return only its output.

        Args:
            state: SpectraX state tree or transform state passed into the operation.
            *args_without_module: Additional positional arguments forwarded to the wrapped callable or backend.

        Returns:
            Result described by this helper.
        """
        return _run_single_positional_body(
            fn,
            ref.gdef,
            ref.module,
            locator,
            state,
            args_without_module,
            readonly=True,
        )

    return pure


def assert_state_unchanged(module: Module, before: State, after: State) -> None:
    """Raise if any leaf value changed between two same-structure states.

    Used by the scan / cond / switch / while_loop / fori_loop wrappers
    after each iteration / branch to verify the supposedly-invariant
    portion of the module state really did not move. Only same-keyed
    states are compared per-leaf — a key-set mismatch is intentionally
    silently ignored here because :func:`~spectrax.transforms.scan._check_invariant_equal`
    handles that case with its own dedicated error message.

    Args:
        module: The module whose state is being checked (used only for
            future contextual error messages).
        before: The reference state.
        after: The state observed after one iteration / branch.

    Raises:
        IllegalMutationError: If any leaf with the same key changed
            identity.
    """
    before_paths = set(before.paths())
    after_paths = set(after.paths())
    if before_paths != after_paths:
        return
    for collection, path, new_val in after.items():
        old_val = before.get(collection, path)
        if old_val is not new_val:
            _raise_illegal_mutation(collection, path)


def apply_mutations(
    refs: list[_ModuleRef],
    new_states: list[State],
    mutable: Selector | None,
) -> None:
    """Write ``new_states`` back to the live modules, honoring ``mutable``.

    The inverse of the trace-time capture done in :func:`make_pure`: for
    every variable whose leaf differs from the pre-transform snapshot,
    the new value is copied into the live :class:`~spectrax.Variable`'s
    underlying storage. Leaves that are identity-equal to
    :attr:`_ModuleRef.state` are treated as unchanged — either
    :func:`make_pure` elided them at trace time, or a control-flow path
    (:func:`~spectrax.cond`, :func:`~spectrax.scan`, …) merged the
    untouched invariant portion of the state back in. Mutations outside
    the ``mutable`` selector raise :class:`~spectrax.IllegalMutationError`.

    The ``vars_by_path`` dict is built once per call by reusing the live
    module's :func:`~spectrax.export` cache — no extra tree walk. The
    ``allowed`` set is constructed lazily only when a real mutation is
    detected, so a pure forward pass (no mutations) pays only an
    ``if not new_raw: continue`` check per module.

    Args:
        refs: One :class:`_ModuleRef` per live module that flowed
            through the transform.
        new_states: List of :class:`State` objects parallel to ``refs``,
            typically the second half of the pure function's return.
        mutable: Selector picking which (collection, path) pairs are
            allowed to mutate. ``None`` means "nothing mutable".

    Raises:
        IllegalMutationError: When a variable's leaf changed under a
            transform but the (collection, path) is not picked by
            ``mutable``.
    """
    mutable_kind_set: set[str] | None = None
    if (
        mutable is not None
        and not mutable.subselectors
        and not mutable.module_types
        and not mutable.exclude_module_types
        and not mutable.module_where
        and not mutable.variable_types
        and not mutable.exclude_variable_types
        and not mutable.path_globs
        and not mutable.variable_where
        and not mutable.invert
        and mutable.variable_kinds
    ):
        mutable_kind_set = set(mutable.variable_kinds)

    if mutable_kind_set is not None:
        for ref, new_state in zip(refs, new_states, strict=False):
            if not new_state:
                continue
            cache = ref.module._spx_export_cache
            vars_by_collection = cache[5] if cache is not None and len(cache) >= 6 else None
            for c, p, new_val in new_state.items():
                old_val = ref.state.get(c, p)
                if old_val is new_val:
                    continue
                if c not in mutable_kind_set:
                    raise IllegalMutationError(
                        f"Collection {c!r} at path {p!r} changed under a "
                        f"transform but was not declared mutable. "
                        f"Add it to `mutable=`."
                    )
                if vars_by_collection is not None:
                    inner = vars_by_collection.get(c)
                    if inner is not None:
                        var = inner.get(p)
                        if var is not None:
                            var._value = new_val
                else:
                    vars_by_path = {(_v.kind, _p): _v for _p, _v in _sel().apply(ref.module)}
                    var = vars_by_path.get((c, p))
                    if var is not None:
                        var._value = new_val
        return

    for ref, new_state in zip(refs, new_states, strict=False):
        if not new_state:
            continue

        vars_by_path: dict[tuple[str, str], object] | None = None
        allowed: set[tuple[str, str]] | None = None

        for c, p, new_val in new_state.items():
            old_val = ref.state.get(c, p)
            if old_val is new_val:
                continue
            if allowed is None:
                allowed = set()
                if mutable is not None:
                    for _p, _v in mutable.apply(ref.module):
                        allowed.add((_v.kind, _p))
            if (c, p) not in allowed:
                raise IllegalMutationError(
                    f"Collection {c!r} at path {p!r} changed under a "
                    f"transform but was not declared mutable. "
                    f"Add it to `mutable=`."
                )
            if vars_by_path is None:
                cache = ref.module._spx_export_cache
                if cache is not None and len(cache) >= 4:
                    vars_by_path = cache[3]
                else:
                    vars_by_path = {(_v.kind, _p): _v for _p, _v in _sel().apply(ref.module)}
            var = vars_by_path.get((c, p))
            if var is not None:
                var._raw_set(new_val)


def _copy_runtime_state(src: Module, dst: Module) -> None:
    """Copy non-graph runtime state (training flag) from ``src`` onto ``dst``.

    :class:`Module` mode flags are not part of :class:`GraphDef` and
    therefore do not survive :func:`bind`. This helper walks both trees in
    lockstep and restores them so eager-mode toggles (``train()`` /
    ``eval()``) take effect inside transforms.

    Args:
        src: Src value consumed by this operation.
        dst: Dst value consumed by this operation.
    """
    object.__setattr__(dst, "_spx_training", src._spx_training)
    src_children = list(src._spx_graph_children())
    dst_children = list(dst._spx_graph_children())
    for (_, sc), (_, dc) in zip(src_children, dst_children, strict=False):
        if isinstance(sc, Module) and isinstance(dc, Module):
            _copy_runtime_state(sc, dc)


def _is_same(a: object, b: object) -> bool:
    """Return whether ``a`` and ``b`` represent the same array value.

    Identity-equal values short-circuit to ``True``. Otherwise the
    helper falls back to :func:`jax.numpy.array_equal`, swallowing any
    exception (typically tracer-time comparisons that raise) and
    returning ``False``. Used by callers that need a "best effort"
    equality check that never crashes mid-trace.

    Args:
        a: First value.
        b: Second value.

    Returns:
        ``True`` when the values are identity-equal or array-equal;
        ``False`` otherwise (including when comparison raises).
    """
    if a is b:
        return True
    try:
        return bool(jax.numpy.array_equal(a, b))
    except Exception:
        return False


def resolve_mutable(mutable: SelectorSugar | tuple[()] | list[object]) -> Selector | None:
    """Coerce a ``mutable=`` argument into a :class:`~spectrax.Selector` or ``None``.

    Centralizes the user-facing convention that an empty container
    (``()``, ``[]``) or ``None`` means "no collections are mutable",
    which lets every transform short-circuit to its readonly fast path
    without parsing selectors. Anything else is delegated to
    :func:`~spectrax.as_selector` which handles strings, tuples of
    strings, full :class:`~spectrax.Selector` instances, etc.

    Args:
        mutable: Whatever the user passed as ``mutable=``.

    Returns:
        The resolved :class:`~spectrax.Selector`, or ``None`` if the
        input meant "nothing mutable".
    """
    if mutable is None or mutable == () or mutable == []:
        return None
    return as_selector(mutable)
