# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Module-aware ``jax.jit`` wrapper.

Every keyword supported by :func:`jax.jit` is available here with the
same default as upstream. SpecTrax adds ``mutable`` (selects which
collections, e.g. ``"batch_stats"``, ``"cache"``, may be written back
to live modules after the transformed call — any other write raises
:class:`~spectrax.IllegalMutationError`) plus ``mesh`` and ``schedule``
for routing to the MPMD ``sxjit`` runtime. For SPMD meshes (or when
``mesh`` is not given) the call dispatches to :func:`jax.jit`; if the
mesh is MPMD-shaped it dispatches to
:func:`spectrax.runtime.mpmd.sxjit`.

When ``mutable`` is set, the compiled function's argument layout is
``(states, stripped_args, stripped_kwargs)`` rather than the user's
original signature, so ``static_argnums`` / ``donate_argnums`` /
``in_shardings`` / ``out_shardings`` index into that 3-tuple. When in
doubt use the ``*_argnames`` variants.
"""

from __future__ import annotations

import functools
import weakref
from collections.abc import Callable, Iterable, Sequence
from typing import TypeVar, cast

import jax

from ..core.context import _STACK as _CTX_STACK
from ..core.context import partition as _ctx_partition
from ..core.graph import export
from ..core.module import Module, _graph_epoch
from ..core.selector import SelectorSugar
from ..core.state import State, StateCallABI
from .split_merge import (
    apply_mutations,
    locate_and_strip,
    locate_and_strip_fast,
    make_direct_readonly,
    make_pure,
    make_pure_ctx,
    make_pure_readonly,
    make_pure_readonly_single_positional,
    make_pure_single_positional,
    resolve_mutable,
)

__all__ = ["jit"]

F = TypeVar("F", bound=Callable[..., object])

_UNSET: object = object()
"""Sentinel indicating that a keyword was not supplied, so JAX's own
``UnspecifiedValue`` default is used.
"""

_StateArgRef = tuple[str, int | str, StateCallABI]
_StateArgCache = weakref.WeakKeyDictionary[State, tuple[int, tuple[object, ...], StateCallABI]]


def _live_module_refs(
    args: tuple[object, ...], kwargs: dict[str, object]
) -> tuple[tuple[object, ...], tuple[object, ...], tuple[int, ...]]:
    """Build cache-key tuples for the top-level :class:`~spectrax.Module` arguments.

    Used by the readonly fast path in :func:`jit`: instead of going
    through the full split/merge shim, the wrapped call dispatches a
    direct :func:`jax.jit` over the live module pytree, keyed by the
    triple ``(layout, gdef, id)``. This avoids an export per call when
    the same module instance is repeatedly passed in.

    For each module the export cache is consulted; if it is stale (the
    global graph epoch advanced since the last export) the cache is
    refreshed via :func:`~spectrax.export`.

    Args:
        args: Positional arguments passed to the wrapped function.
        kwargs: Keyword arguments passed to the wrapped function.

    Returns:
        A triple ``(layout_key, gdef_key, id_key)``:

        * ``layout_key`` records the ``("arg"|"kwarg", index|name)``
          position of every module.
        * ``gdef_key`` collects the corresponding :class:`~spectrax.GraphDef`
          values for the structural compile cache.
        * ``id_key`` collects the Python ``id`` of each module instance
          for the identity-based hot-path cache.
    """
    layout: list[object] = []
    gdefs: list[object] = []
    ids: list[int] = []
    epoch = _graph_epoch()
    for i, value in enumerate(args):
        if isinstance(value, Module):
            cache = value._spx_export_cache
            if cache is None or cache[0] != epoch:
                export(value)
                cache = value._spx_export_cache
            assert cache is not None
            layout.append(("arg", i))
            gdefs.append(cache[1])
            ids.append(id(value))
    for key, value in kwargs.items():
        if isinstance(value, Module):
            cache = value._spx_export_cache
            if cache is None or cache[0] != epoch:
                export(value)
                cache = value._spx_export_cache
            assert cache is not None
            layout.append(("kwarg", key))
            gdefs.append(cache[1])
            ids.append(id(value))
    return tuple(layout), tuple(gdefs), tuple(ids)


def _normalize_static_argnums_set(argnums: int | Sequence[int] | None) -> set[int]:
    """Return positional static argnums as a set for call-boundary rewriting."""
    if argnums is None:
        return set()
    if isinstance(argnums, int):
        return {argnums}
    return set(argnums)


def _normalize_static_argnames_set(argnames: str | Iterable[str] | None) -> set[str]:
    """Return keyword static argnames as a set for call-boundary rewriting."""
    if argnames is None:
        return set()
    if isinstance(argnames, str):
        return {argnames}
    return set(argnames)


def _flatten_state_call_args(
    args: tuple[object, ...],
    kwargs: dict[str, object],
    static_argnums: int | Sequence[int] | None,
    static_argnames: str | Iterable[str] | None,
    donate_argnums: int | Sequence[int] | None,
    donate_argnames: str | Iterable[str] | None,
    state_arg_cache: _StateArgCache,
) -> tuple[tuple[_StateArgRef, ...], tuple[object, ...], dict[str, object]]:
    """Replace dynamic top-level ``State`` arguments with cached leaf tuples."""
    static_nums = {index if index >= 0 else len(args) + index for index in _normalize_static_argnums_set(static_argnums)}
    static_names = _normalize_static_argnames_set(static_argnames)
    donated_nums = {
        index if index >= 0 else len(args) + index for index in _normalize_static_argnums_set(donate_argnums)
    }
    donated_names = _normalize_static_argnames_set(donate_argnames)
    refs: list[_StateArgRef] = []
    call_args = list(args)
    for index, value in enumerate(args):
        if index in static_nums or not isinstance(value, State):
            continue
        leaves, abi = (
            _uncached_state_call_flatten(value)
            if index in donated_nums
            else _cached_state_call_flatten(value, state_arg_cache)
        )
        call_args[index] = leaves
        refs.append(("arg", index, abi))

    if not kwargs:
        return tuple(refs), tuple(call_args), kwargs

    call_kwargs = dict(kwargs)
    for name, value in kwargs.items():
        if name in static_names or not isinstance(value, State):
            continue
        leaves, abi = (
            _uncached_state_call_flatten(value)
            if name in donated_names
            else _cached_state_call_flatten(value, state_arg_cache)
        )
        call_kwargs[name] = leaves
        refs.append(("kwarg", name, abi))
    return tuple(refs), tuple(call_args), call_kwargs


def _uncached_state_call_flatten(state: State) -> tuple[tuple[object, ...], StateCallABI]:
    """Flatten a State once without storing donated leaves in the call cache."""
    leaves, treedef = jax.tree_util.tree_flatten(state)
    leaves_tuple = tuple(leaves)
    return leaves_tuple, StateCallABI._from_flattened(leaves_tuple, treedef)


def _cached_state_call_flatten(
    state: State,
    state_arg_cache: _StateArgCache,
) -> tuple[tuple[object, ...], StateCallABI]:
    """Return flat State leaves from a per-jit cache when the State is unchanged."""
    version = state._version
    cached = state_arg_cache.get(state)
    if cached is not None and cached[0] == version:
        return cached[1], cached[2]

    leaves, treedef = jax.tree_util.tree_flatten(state)
    leaves_tuple = tuple(leaves)
    abi = StateCallABI._from_flattened(leaves_tuple, treedef)
    state_arg_cache[state] = (version, leaves_tuple, abi)
    return leaves_tuple, abi


def _state_refs_key(refs: tuple[_StateArgRef, ...]) -> tuple[object, ...]:
    """Return the structural cache key for flattened ``State`` arguments."""
    return tuple((kind, locator, abi.num_leaves, abi.treedef_key) for kind, locator, abi in refs)


def _with_unflattened_state_args(fn: Callable[..., object], refs: tuple[_StateArgRef, ...]) -> Callable[..., object]:
    """Wrap ``fn`` so flat state leaves are rebound before entering user code."""
    if not refs:
        return fn

    @functools.wraps(fn)
    def inner(*args: object, **kwargs: object) -> object:
        restored_args = list(args)
        restored_kwargs = dict(kwargs)
        for kind, locator, abi in refs:
            if kind == "arg":
                index = int(locator)
                restored_args[index] = abi.unflatten(restored_args[index])
            else:
                restored_kwargs[str(locator)] = abi.unflatten(restored_kwargs[str(locator)])
        return fn(*tuple(restored_args), **restored_kwargs)

    return inner


def _flatten_state_in_shardings(in_shardings: object, refs: tuple[_StateArgRef, ...]) -> object:
    """Adapt top-level state shardings to the flattened call ABI when possible."""
    if in_shardings is _UNSET or not refs:
        return in_shardings
    if not isinstance(in_shardings, (tuple, list)):
        return in_shardings

    flattened = list(in_shardings)
    changed = False
    for kind, locator, abi in refs:
        if kind != "arg":
            continue
        index = int(locator)
        if index >= len(flattened):
            continue
        sharding = flattened[index]
        if sharding is None:
            continue
        leaves = tuple(jax.tree_util.tree_leaves(sharding))
        if len(leaves) == 1 and abi.num_leaves != 1:
            continue
        flattened[index] = abi.flatten_sharding(sharding)
        changed = True
    if not changed:
        return in_shardings
    return tuple(flattened) if isinstance(in_shardings, tuple) else flattened


def jit(
    fn: F | None = None,
    *,
    mutable: SelectorSugar = (),
    mesh: object | None = None,
    schedule: object | None = None,
    in_shardings: object = _UNSET,
    out_shardings: object = _UNSET,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
    donate_argnums: int | Sequence[int] | None = None,
    donate_argnames: str | Iterable[str] | None = None,
    batch_argnums: int | Sequence[int] | None = None,
    keep_unused: bool = False,
    device: object = None,
    backend: str | None = None,
    inline: bool = False,
    compiler_options: dict[str, object] | None = None,
) -> F:
    """Module-aware ``jax.jit``.

    Compiles ``fn`` via :func:`jax.jit` while transparently handling
    :class:`~spectrax.Module` arguments. On the first call with a given
    module structure, each :class:`~spectrax.Module` argument is
    exported to ``(GraphDef, State)``, the state is threaded through the
    compiled function, and after the call any mutations to
    declared-mutable collections are written back to the live module.

    **Compiled signature when ``mutable`` is set**

    When ``mutable`` resolves to a non-empty selector, the compiled
    function's argument layout becomes ``(states, stripped_args,
    stripped_kwargs)`` rather than the user's original signature. This
    means ``static_argnums``, ``donate_argnums``, ``in_shardings``, and
    ``out_shardings`` index into that 3-tuple. When in doubt, prefer
    the ``*_argnames`` variants which resolve by name before stripping.

    **In-Out kwargs**

    ``in_shardings`` and ``out_shardings`` (plus ``static_argnums``,
    ``static_argnames``, ``donate_argnums``, ``donate_argnames``,
    ``batch_argnums``, ``keep_unused``, ``device``, ``backend``,
    ``inline``, ``compiler_options``) are forwarded verbatim to
    :func:`jax.jit`.

    **State call ABI**

    On the default readonly path, dynamic top-level :class:`~spectrax.State`
    positional and keyword arguments are passed to the underlying
    :func:`jax.jit` as flat leaf tuples, then reconstructed before user code
    runs. This preserves the public call signature while avoiding repeated
    ``State`` wrapper pytree overhead in hot serving loops.

    **Compile caching**

    The returned wrapper maintains two internal caches:

    * ``_spx_id_cache`` — identity-based hot-path cache keyed by Python
      ``id()`` of input modules plus the current graph epoch. Same
      instance + no structural change reuses the jitted callable in O(1).
    * ``_spx_compile_cache`` — structural cache keyed by the full
      ``GraphDef`` tuple. Handles model swaps and distinct instances
      with identical structure.

    **MPMD dispatch**

    If ``mesh`` is an MPMD mesh, the call routes to
    :func:`spectrax.runtime.mpmd.sxjit` instead of :func:`jax.jit`. In
    that mode ``mutable``, ``donate_argnames``, ``keep_unused``,
    ``device``, ``backend``, ``inline``, and ``compiler_options`` are
    unsupported and raise :class:`ValueError`.

    **Lowered representation**

    The returned callable has a ``.lower(*args, **kwargs)`` method that
    mirrors :meth:`jax.jit.lower` and returns a
    :class:`jax.stages.Lowered` object without dispatching the compiled
    function.

    Args:
        fn: The function to compile. When called without ``fn`` the
            decorator returns a factory: ``@spx.jit(mutable=...)``.
        mutable: Selector (or collection-name sugar) controlling which
            collections may be written back after the call.
        mesh: Optional SpectraX mesh. If this is an MPMD mesh, dispatches
            directly to :func:`spectrax.runtime.mpmd.sxjit`.
        schedule: Optional MPMD schedule, forwarded only when ``mesh`` is
            MPMD.
        in_shardings: Optional sharding constraint for inputs; forwarded
            to :func:`jax.jit`.
        out_shardings: Optional sharding constraint for outputs;
            forwarded to :func:`jax.jit`.
        static_argnums: Indices of positional arguments that should be
            treated as compile-time constants; forwarded to
            :func:`jax.jit`.
        static_argnames: Names of keyword arguments that should be
            treated as compile-time constants; forwarded to
            :func:`jax.jit`.
        donate_argnums: Indices of positional arguments whose buffers
            may be donated to the output; forwarded to :func:`jax.jit`.
        donate_argnames: Names of keyword arguments whose buffers may
            be donated; forwarded to :func:`jax.jit`.
        batch_argnums: MPMD-only; indices of positional arguments that
            represent batch dimensions. Requires an MPMD mesh.
        keep_unused: Forwarded to :func:`jax.jit`.
        device: Forwarded to :func:`jax.jit`.
        backend: Forwarded to :func:`jax.jit`.
        inline: Forwarded to :func:`jax.jit`.
        compiler_options: Forwarded to :func:`jax.jit`.

    Returns:
        A wrapped function. The first call per distinct
        :class:`~spectrax.GraphDef` tuple triggers a JAX trace; later
        calls with matching graph-defs re-use the cached compile.

    Raises:
        ValueError: If ``schedule`` is provided without an MPMD mesh, or
            if ``batch_argnums`` is provided without an MPMD mesh with
            a schedule, or if an unsupported option is passed when
            routing to ``sxjit``.
    """
    if fn is None:
        return cast(
            F,
            lambda f: jit(
                f,
                mutable=mutable,
                mesh=mesh,
                schedule=schedule,
                in_shardings=in_shardings,
                out_shardings=out_shardings,
                static_argnums=static_argnums,
                static_argnames=static_argnames,
                donate_argnums=donate_argnums,
                donate_argnames=donate_argnames,
                batch_argnums=batch_argnums,
                keep_unused=keep_unused,
                device=device,
                backend=backend,
                inline=inline,
                compiler_options=compiler_options,
            ),
        )

    if mesh is not None and _is_mpmd_mesh(mesh):
        _raise_if_unsupported_mpmd_jit_options(
            mutable=mutable,
            donate_argnames=donate_argnames,
            keep_unused=keep_unused,
            device=device,
            backend=backend,
            inline=inline,
            compiler_options=compiler_options,
        )
        from ..runtime.mpmd import sxjit

        return cast(
            F,
            sxjit(
                fn,
                mesh=mesh,
                schedule=schedule,
                static_argnums=_normalize_argnums_for_sxjit(static_argnums),
                static_argnames=_normalize_argnames_for_sxjit(static_argnames),
                donate_argnums=_normalize_argnums_for_sxjit(donate_argnums),
                batch_argnums=_normalize_argnums_for_sxjit(batch_argnums),
                in_shardings=None if in_shardings is _UNSET else in_shardings,
                out_shardings=None if out_shardings is _UNSET else out_shardings,
            ),
        )
    if schedule is not None:
        raise ValueError("spx.jit(..., schedule=...) requires an MPMD mesh.")
    if batch_argnums is not None:
        raise ValueError("spx.jit(..., batch_argnums=...) requires an MPMD mesh with schedule=.")

    mutable_sel = resolve_mutable(mutable)

    jit_kwargs: dict[str, object] = {
        "static_argnums": static_argnums,
        "static_argnames": static_argnames,
        "donate_argnums": donate_argnums,
        "donate_argnames": donate_argnames,
        "keep_unused": keep_unused,
        "device": device,
        "backend": backend,
        "inline": inline,
        "compiler_options": compiler_options,
    }
    if in_shardings is not _UNSET:
        jit_kwargs["in_shardings"] = in_shardings
    if out_shardings is not _UNSET:
        jit_kwargs["out_shardings"] = out_shardings

    _compile_cache: dict[tuple[object, ...], object] = {}
    _id_cache: dict[tuple[int, ...], tuple[int, object, object]] = {}
    _id_cache_one: dict[int, tuple[int, object, object]] = {}
    _ctx_compile_cache: dict[tuple[object, ...], object] = {}
    _state_arg_cache: _StateArgCache = weakref.WeakKeyDictionary()

    _locate = locate_and_strip
    _locate_fast = locate_and_strip_fast
    _epoch_fn = _graph_epoch
    _apply = apply_mutations
    _make_pure = make_pure
    _make_pure_readonly = make_pure_readonly
    _make_pure_single = make_pure_single_positional
    _make_pure_readonly_single = make_pure_readonly_single_positional
    _make_pure_ctx = make_pure_ctx
    _jax_jit = jax.jit
    _ctx_stack_get = _CTX_STACK.get
    _empty_kwargs: dict[str, object] = {}
    _direct_guarded = make_direct_readonly(fn)

    def _jit_kwargs_for_state_refs(refs: tuple[_StateArgRef, ...]) -> dict[str, object]:
        """Return JAX jit kwargs adapted to any flattened State arguments."""
        if not refs or in_shardings is _UNSET:
            return jit_kwargs
        flattened_in_shardings = _flatten_state_in_shardings(in_shardings, refs)
        if flattened_in_shardings is in_shardings:
            return jit_kwargs
        updated = dict(jit_kwargs)
        updated["in_shardings"] = flattened_in_shardings
        return updated

    @functools.wraps(fn)
    def wrapped(*args: object, **kwargs: object) -> object:
        """Dispatch through the graph-def-keyed compile cache.

        Two-level cache:

        1. Identity cache (``_id_cache``) keyed by the Python ``id``
            tuple of the input modules plus the current global graph
        epoch. Hot path: same model instance + no structural change
        returns the cached jitted callable in O(1) without ever
        touching the graph-def hash.
        2. Structural cache (``_compile_cache``) keyed by the full
            graph-def tuple. Handles model swaps, reloads, and distinct
        instances with identical structure.

        A kwargs-empty fast path uses :func:`locate_and_strip_fast`
        which does a single pass over ``args`` and skips the kwargs
        iteration entirely.

        Scope-aware slow path activates when the caller has active
        :func:`~spectrax.scope` bindings: static context values are
        folded into the compile cache key (so different static
        snapshots specialize cleanly); array-typed context values are
        lifted into the jit input tuple and reinstated as a scope
        frame inside the traced body (see :func:`make_pure_ctx`). The
        no-scope hot path above is completely unaffected — a single
        :class:`~contextvars.ContextVar` read (~50 ns) decides which
        path to take.

        Args:
            *args: Additional positional arguments forwarded to the wrapped callable or backend.
            **kwargs: Additional keyword arguments forwarded to the wrapped callable or backend.

        Returns:
            Result described by this helper.
        """
        ctx_stack = _ctx_stack_get()
        if ctx_stack:
            return _wrapped_with_ctx(ctx_stack, args, kwargs)

        if mutable_sel is None:
            state_refs, call_args, call_kwargs = _flatten_state_call_args(
                args,
                kwargs,
                static_argnums,
                static_argnames,
                donate_argnums,
                donate_argnames,
                _state_arg_cache,
            )
            state_key = _state_refs_key(state_refs)
            layout_key, gdef_key, id_key = _live_module_refs(args, kwargs)
            call_layout_key = (layout_key, state_key)
            if len(id_key) == 1:
                id_hit = _id_cache_one.get(id_key[0])
                if id_hit is not None and id_hit[0] == _epoch_fn() and id_hit[1] == call_layout_key:
                    jitted = id_hit[2]
                else:
                    epoch = _epoch_fn()
                    key = ("direct", layout_key, gdef_key, state_key)
                    jitted = _compile_cache.get(key)
                    if jitted is None:
                        direct = _with_unflattened_state_args(_direct_guarded, state_refs)
                        jitted = _jax_jit(direct, **_jit_kwargs_for_state_refs(state_refs))
                        _compile_cache[key] = jitted
                    _id_cache_one[id_key[0]] = (epoch, call_layout_key, jitted)
                return jitted(*call_args, **call_kwargs)

            id_hit = _id_cache.get(id_key)
            if id_hit is not None and id_hit[0] == _epoch_fn() and id_hit[1] == call_layout_key:
                jitted = id_hit[2]
            else:
                epoch = _epoch_fn()
                key = ("direct", layout_key, gdef_key, state_key)
                jitted = _compile_cache.get(key)
                if jitted is None:
                    direct = _with_unflattened_state_args(_direct_guarded, state_refs)
                    jitted = _jax_jit(direct, **_jit_kwargs_for_state_refs(state_refs))
                    _compile_cache[key] = jitted
                _id_cache[id_key] = (epoch, call_layout_key, jitted)
            return jitted(*call_args, **call_kwargs)

        if kwargs:
            refs, stripped_args, stripped_kwargs = _locate(args, kwargs)
        else:
            refs, stripped_args = _locate_fast(args)
            stripped_kwargs = _empty_kwargs
        n = len(refs)
        layout_key = tuple((r.kind, r.locator) for r in refs)
        if n == 1:
            r0 = refs[0]
            mid = id(r0.module)
            states_in: tuple = (r0.state,)
            single_positional = (not kwargs) and r0.kind == "arg"
            call_layout_key = (layout_key, single_positional)
            id_hit = _id_cache_one.get(mid)
            if id_hit is not None and id_hit[0] == _epoch_fn() and id_hit[1] == call_layout_key:
                jitted = id_hit[2]
            else:
                epoch = _epoch_fn()
                key = (layout_key, r0.gdef, single_positional)
                jitted = _compile_cache.get(key)
                if jitted is None:
                    if single_positional:
                        pure = _make_pure_readonly_single(fn, r0) if mutable_sel is None else _make_pure_single(fn, r0)
                    else:
                        pure = _make_pure_readonly(fn, refs) if mutable_sel is None else _make_pure(fn, refs)
                    jitted = _jax_jit(pure, **jit_kwargs)
                    _compile_cache[key] = jitted
                _id_cache_one[mid] = (epoch, call_layout_key, jitted)
            if single_positional:
                other_args = stripped_args[: int(r0.locator)] + stripped_args[int(r0.locator) + 1 :]
                if mutable_sel is None:
                    return jitted(r0.state, *other_args)
                out, new_state = jitted(r0.state, *other_args)
                _apply([r0], [new_state], mutable_sel)
                return out
        else:
            id_key_list = []
            states_list = []
            for r in refs:
                id_key_list.append(id(r.module))
                states_list.append(r.state)
            id_key = tuple(id_key_list)
            states_in = tuple(states_list)
            id_hit = _id_cache.get(id_key)
            if id_hit is not None and id_hit[0] == _epoch_fn() and id_hit[1] == layout_key:
                jitted = id_hit[2]
            else:
                epoch = _epoch_fn()
                key = (layout_key, tuple([r.gdef for r in refs]))
                jitted = _compile_cache.get(key)
                if jitted is None:
                    pure = _make_pure_readonly(fn, refs) if mutable_sel is None else _make_pure(fn, refs)
                    jitted = _jax_jit(pure, **jit_kwargs)
                    _compile_cache[key] = jitted
                _id_cache[id_key] = (epoch, layout_key, jitted)
        if mutable_sel is None:
            return jitted(states_in, stripped_args, stripped_kwargs)
        out, new_states = jitted(states_in, stripped_args, stripped_kwargs)
        _apply(refs, new_states, mutable_sel)
        return out

    def _wrapped_with_ctx(
        ctx_stack: tuple[dict[str, object], ...],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        """Scope-active dispatch path.

        Flattens the scope stack, splits it into
        :data:`~spectrax.core.context.partition`'s traced/static halves,
        augments the compile-cache key with the static snapshot, and
        routes the call through a separate ``pure`` built by
        :func:`make_pure_ctx` so deep ``spx.scope.get(...)`` calls
        inside the traced body resolve to tracers rather than the
        constants captured at the trace-time snapshot.

        Args:
            ctx_stack: Ctx stack value consumed by this operation.
            args: Positional arguments forwarded to the wrapped callable.
            kwargs: Keyword arguments forwarded to the wrapped callable.

        Returns:
            Result described by this helper.
        """
        snap: dict[str, object] = {}
        for frame in ctx_stack:
            snap.update(frame)
        traced_ctx, static_ctx = _ctx_partition(snap)
        if kwargs:
            refs, stripped_args, stripped_kwargs = _locate(args, kwargs)
        else:
            refs, stripped_args = _locate_fast(args)
            stripped_kwargs = _empty_kwargs
        key = (tuple((r.kind, r.locator) for r in refs), tuple([r.gdef for r in refs]), static_ctx)
        jitted = _ctx_compile_cache.get(key)
        if jitted is None:
            pure = _make_pure_ctx(fn, refs)
            jitted = _jax_jit(pure, **jit_kwargs)
            _ctx_compile_cache[key] = jitted
        states_in = tuple([r.state for r in refs])
        out, new_states = jitted(states_in, traced_ctx, stripped_args, stripped_kwargs)
        _apply(refs, new_states, mutable_sel)
        return out

    def lower(*args: object, **kwargs: object) -> object:
        """Lower the call to a :class:`jax.stages.Lowered` without dispatching it.

        Mirrors :meth:`jax.jit.lower` for ahead-of-time users while still
        preserving the graph-def-keyed compile cache. Picks the same
        scope-aware / direct / pure dispatch path as ``wrapped`` would,
        builds the right pure body for that path, calls ``.lower(...)``
        on the resulting jitted function, and returns the
        :class:`~jax.stages.Lowered`. Side effects of the runtime
        wrapper (specifically
        :func:`~spectrax.transforms.split_merge.apply_mutations`) are
        skipped — lowering is purely about preparing the compiled
        artifact.

        Args:
            *args: Additional positional arguments forwarded to the wrapped callable or backend.
            **kwargs: Additional keyword arguments forwarded to the wrapped callable or backend.

        Returns:
            Result described by this helper.
        """
        ctx_stack = _ctx_stack_get()
        if ctx_stack:
            snap: dict[str, object] = {}
            for frame in ctx_stack:
                snap.update(frame)
            traced_ctx, static_ctx = _ctx_partition(snap)
            if kwargs:
                refs, stripped_args, stripped_kwargs = _locate(args, kwargs)
            else:
                refs, stripped_args = _locate_fast(args)
                stripped_kwargs = _empty_kwargs
            key = (tuple((r.kind, r.locator) for r in refs), tuple([r.gdef for r in refs]), static_ctx)
            jitted = _ctx_compile_cache.get(key)
            if jitted is None:
                pure = _make_pure_ctx(fn, refs)
                jitted = _jax_jit(pure, **jit_kwargs)
                _ctx_compile_cache[key] = jitted
            states_in = tuple([r.state for r in refs])
            return jitted.lower(states_in, traced_ctx, stripped_args, stripped_kwargs)

        if mutable_sel is None:
            state_refs, call_args, call_kwargs = _flatten_state_call_args(
                args,
                kwargs,
                static_argnums,
                static_argnames,
                donate_argnums,
                donate_argnames,
                _state_arg_cache,
            )
            state_key = _state_refs_key(state_refs)
            layout_key, gdef_key, id_key = _live_module_refs(args, kwargs)
            call_layout_key = (layout_key, state_key)
            key = ("direct", layout_key, gdef_key, state_key)
            jitted = _compile_cache.get(key)
            if jitted is None:
                direct = _with_unflattened_state_args(_direct_guarded, state_refs)
                jitted = _jax_jit(direct, **_jit_kwargs_for_state_refs(state_refs))
                _compile_cache[key] = jitted
            epoch = _epoch_fn()
            if len(id_key) == 1:
                _id_cache_one[id_key[0]] = (epoch, call_layout_key, jitted)
            elif id_key:
                _id_cache[id_key] = (epoch, call_layout_key, jitted)
            return jitted.lower(*call_args, **call_kwargs)

        if kwargs:
            refs, stripped_args, stripped_kwargs = _locate(args, kwargs)
        else:
            refs, stripped_args = _locate_fast(args)
            stripped_kwargs = _empty_kwargs
        layout_key = tuple((r.kind, r.locator) for r in refs)
        if len(refs) == 1:
            r0 = refs[0]
            single_positional = (not kwargs) and r0.kind == "arg"
            key = (layout_key, r0.gdef, single_positional)
            jitted = _compile_cache.get(key)
            if jitted is None:
                pure = _make_pure_readonly_single(fn, r0) if mutable_sel is None else _make_pure_single(fn, r0)
                if not single_positional:
                    pure = _make_pure_readonly(fn, refs) if mutable_sel is None else _make_pure(fn, refs)
                jitted = _jax_jit(pure, **jit_kwargs)
                _compile_cache[key] = jitted
            _id_cache_one[id(r0.module)] = (_epoch_fn(), (layout_key, single_positional), jitted)
            if single_positional:
                other_args = stripped_args[: int(r0.locator)] + stripped_args[int(r0.locator) + 1 :]
                return jitted.lower(r0.state, *other_args)
        else:
            key = (layout_key, tuple([r.gdef for r in refs]))
            jitted = _compile_cache.get(key)
            if jitted is None:
                pure = _make_pure_readonly(fn, refs) if mutable_sel is None else _make_pure(fn, refs)
                jitted = _jax_jit(pure, **jit_kwargs)
                _compile_cache[key] = jitted
            _id_cache[tuple(id(r.module) for r in refs)] = (_epoch_fn(), layout_key, jitted)
        states_in = tuple([r.state for r in refs])
        return jitted.lower(states_in, stripped_args, stripped_kwargs)

    wrapped._spx_compile_cache = _compile_cache
    wrapped._spx_id_cache = _id_cache
    wrapped._spx_ctx_compile_cache = _ctx_compile_cache
    wrapped.lower = lower
    return cast(F, wrapped)


def _is_mpmd_mesh(mesh: object) -> bool:
    """Return whether ``mesh`` is an MPMD mesh requiring the ``sxjit`` runtime.

    Treats either an explicit ``is_mpmd`` boolean attribute or the
    structural duck-typing trio (``mpmd_dim``, ``submesh``,
    ``sub_sharding``) as evidence of an MPMD mesh.

    Args:
        mesh: JAX mesh or SpectraX mesh descriptor used for placement.

    Returns:
        Return whether ``mesh`` is an MPMD mesh requiring the ``sxjit`` runtime.
    """
    return bool(getattr(mesh, "is_mpmd", False)) or (
        hasattr(mesh, "mpmd_dim") and hasattr(mesh, "submesh") and hasattr(mesh, "sub_sharding")
    )


def _normalize_argnums_for_sxjit(argnums: int | Sequence[int] | None) -> int | tuple[int, ...] | None:
    """Coerce ``argnums`` into the ``None``/``int``/``tuple`` shape ``sxjit`` accepts.

    Args:
        argnums: Argnums value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    if argnums is None or isinstance(argnums, int):
        return argnums
    return tuple(argnums)


def _normalize_argnames_for_sxjit(argnames: str | Iterable[str] | None) -> str | tuple[str, ...] | None:
    """Coerce ``argnames`` into the ``None``/``str``/``tuple`` shape ``sxjit`` accepts.

    Args:
        argnames: Argnames value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    if argnames is None or isinstance(argnames, str):
        return argnames
    return tuple(argnames)


def _raise_if_unsupported_mpmd_jit_options(
    *,
    mutable: SelectorSugar,
    donate_argnames: str | Iterable[str] | None,
    keep_unused: bool,
    device: object,
    backend: str | None,
    inline: bool,
    compiler_options: dict[str, object] | None,
) -> None:
    """Raise if the user passed a ``jax.jit`` option that ``sxjit`` cannot honor.

    ``spx.jit`` routes to :func:`sxjit` whenever the mesh is MPMD-shaped,
    but several ``jax.jit`` knobs (``mutable``, ``donate_argnames``,
    ``keep_unused``, ``device``, ``backend``, ``inline``,
    ``compiler_options``) have no MPMD analog. Bundling all rejected
    options into a single error makes the message actionable.

    Args:
        mutable: Mutable value consumed by this operation.
        donate_argnames: Donate argnames value consumed by this operation.
        keep_unused: Keep unused value consumed by this operation.
        device: Device value consumed by this operation.
        backend: Backend value consumed by this operation.
        inline: Inline value consumed by this operation.
        compiler_options: Compiler options value consumed by this operation.
    """
    unsupported: list[str] = []
    if resolve_mutable(mutable) is not None:
        unsupported.append("mutable")
    if donate_argnames is not None:
        unsupported.append("donate_argnames")
    if keep_unused:
        unsupported.append("keep_unused")
    if device is not None:
        unsupported.append("device")
    if backend is not None:
        unsupported.append("backend")
    if inline:
        unsupported.append("inline")
    if compiler_options is not None:
        unsupported.append("compiler_options")
    if unsupported:
        opts = ", ".join(unsupported)
        raise ValueError(f"spx.jit(..., mesh=<MPMD>) routes to sxjit, which does not support: {opts}.")
