# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Module-aware autodiff: :func:`grad`, :func:`value_and_grad`,
:func:`jvp`, and :func:`vjp`.

These wrappers add :class:`~spectrax.Module` awareness on top of JAX's
autodiff primitives. For :func:`grad` and :func:`value_and_grad` the
differentiation target is selected by ``wrt`` — a
:class:`~spectrax.Selector` or any of its sugar forms (``"parameters"``,
``("parameters", "batch_stats")``, etc.) — and is partitioned out of the
target :class:`~spectrax.Module`'s state via
:meth:`Selector.partition_state <spectrax.Selector.partition_state>`. The
non-target portion is captured as constant ``rest_state`` so JAX never
traces through it.

For :func:`jvp` and :func:`vjp` the same module-aware split/merge shim
from :mod:`spectrax.transforms.split_merge` is used: cotangents and
tangents that the user supplies as :class:`~spectrax.Module` (or
:class:`~spectrax.State`) are automatically reshaped to match the JAX
internal representation, and the returned cotangents are converted back
to user-friendly :class:`~spectrax.State` form before being handed back.

Every non-spectrax keyword (``has_aux``, ``holomorphic``, ``allow_int``,
``reduce_axes``) is forwarded to the underlying
:func:`jax.value_and_grad` / :func:`jax.vjp` / :func:`jax.jvp` verbatim.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import TypeVar, cast

import jax

from ..core.graph import bind, export
from ..core.module import Module, _set_inside_transform
from ..core.selector import SelectorSugar, as_selector
from ..core.state import State
from .split_merge import (
    apply_mutations,
    locate_and_strip_fast,
    make_direct_readonly,
    make_pure,
    make_pure_readonly,
    make_pure_readonly_single_positional,
    make_pure_single_positional,
    resolve_mutable,
)

__all__ = ["grad", "jvp", "value_and_grad", "vjp"]

F = TypeVar("F", bound=Callable[..., object])

AxisName = object
"""Type alias for a JAX axis-name sentinel (no canonical type exists)."""

_MISSING = object()
"""Sentinel for deferred ``jvp`` arguments."""


def value_and_grad(
    fn: F | None = None,
    *,
    wrt: SelectorSugar = "parameters",
    argnum: int | None = None,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> F:
    """Differentiate ``fn`` with respect to a selected module subset, returning value and grads.

    Module-aware analogue of :func:`jax.value_and_grad`. Internally,
    :func:`~spectrax.export` snapshots the target module, ``wrt``'s
    selector partitions the state into ``(target, rest)``, and
    :func:`jax.value_and_grad` differentiates against ``target`` only —
    ``rest`` is closed over as a non-traced constant. When the partition
    leaves nothing in ``rest`` (the common case of differentiating all
    parameters) the implementation falls back to a direct
    :func:`jax.value_and_grad` over the whole module pytree, guarded by
    :func:`~spectrax.transforms.split_merge.make_direct_readonly` so the
    function body is forbidden from mutating the module during the
    forward pass.

    Args:
        fn: Function whose ``argnum``-th argument is the target
            :class:`~spectrax.Module`. When called without ``fn``
            returns a decorator factory so ``@spx.value_and_grad(...)``
            works.
        wrt: :class:`~spectrax.Selector` (or selector sugar, e.g.
            ``"parameters"``) selecting the differentiation target.
            Defaults to the ``"parameters"`` collection.
        argnum: Positional index of the differentiated module. Defaults
            to the first :class:`~spectrax.Module` argument.
        has_aux: When ``True``, ``fn`` is expected to return
            ``(value, aux)`` and the wrapper returns
            ``((value, aux), grads)``. Forwarded to
            :func:`jax.value_and_grad`.
        holomorphic: Forwarded to :func:`jax.value_and_grad`; treat
            inputs/outputs as holomorphic functions of complex inputs.
        allow_int: Forwarded to :func:`jax.value_and_grad`; permit
            integer-typed inputs to be differentiated (yielding zero
            grads).
        reduce_axes: Forwarded to :func:`jax.value_and_grad` when
            non-empty.

    Returns:
        A wrapped callable. Without ``has_aux``: returns
        ``(value, grads)``. With ``has_aux``: returns
        ``((value, aux), grads)``. ``grads`` is a
        :class:`~spectrax.State` mirroring the selected subset.

    Raises:
        TypeError: If the resolved positional argument is not a
            :class:`~spectrax.Module`.
    """
    if fn is None:
        return cast(
            F,
            lambda f: value_and_grad(
                f,
                wrt=wrt,
                argnum=argnum,
                has_aux=has_aux,
                holomorphic=holomorphic,
                allow_int=allow_int,
                reduce_axes=reduce_axes,
            ),
        )

    wrt_sel = as_selector(wrt)
    direct_guarded = make_direct_readonly(fn)

    @functools.wraps(fn)
    def wrapped(*args: object, **kwargs: object) -> object:
        """Partition the module's state, differentiate against the target subset, and re-merge.

        Looks up the target module at position ``argnum`` (or the first
        :class:`~spectrax.Module` if unset), splits its state via
        ``wrt``'s selector, and dispatches either to a direct
        :func:`jax.value_and_grad` over the module pytree (when ``rest``
        is empty) or to a state-partitioned pure closure that overlays
        ``target`` and ``rest`` before calling ``fn``.

        Args:
            *args: Additional positional arguments forwarded to the wrapped callable or backend.
            **kwargs: Additional keyword arguments forwarded to the wrapped callable or backend.

        Returns:
            Result described by this helper.
        """
        idx = argnum if argnum is not None else _find_first_module(args)
        model = args[idx]
        if not isinstance(model, Module):
            raise TypeError(f"Argument {idx} must be a Module, got {type(model).__name__}")
        gdef, state = export(model)
        target, rest = wrt_sel.partition_state(model, state)
        if _state_is_empty(rest):
            vg_kwargs: dict[str, object] = {
                "argnums": idx,
                "has_aux": has_aux,
                "holomorphic": holomorphic,
                "allow_int": allow_int,
            }
            if reduce_axes:
                vg_kwargs["reduce_axes"] = reduce_axes
            vg = jax.value_and_grad(direct_guarded, **vg_kwargs)
            out, grads_module = vg(*args, **kwargs)
            return out, _module_like_to_state(grads_module)
        other_args = tuple(args[:idx]) + tuple(args[idx + 1 :])

        def pure(
            target_state: State,
            rest_state: State,
            other: tuple[object, ...],
            kw: dict[str, object],
        ) -> tuple[object, object | None]:
            """Pure closure fed to :func:`jax.value_and_grad`.

            Overlays the differentiation target on top of the captured
            non-target state, rebinds a fresh module, splices it into
            the user's positional arguments at ``idx``, and runs ``fn``
            inside the inside-transform thread-local. Returns
            ``(value, aux_or_None)`` — the wrapper always sets
            ``has_aux=True`` on the inner :func:`jax.value_and_grad`
            call so it can pass ``rest_state`` and ``aux`` through
            uniformly.

            Args:
                target_state: Target state value consumed by this operation.
                rest_state: Rest state value consumed by this operation.
                other: Other value consumed by this operation.
                kw: Keyword arguments forwarded to the wrapped callable.

            Returns:
                Result described by this helper.
            """
            merged = target_state.overlay(rest_state)
            m = bind(gdef, merged)
            spliced = list(other)
            spliced.insert(idx, m)
            _set_inside_transform(True)
            try:
                out = fn(*spliced, **kw)
            finally:
                _set_inside_transform(False)
            if has_aux:
                val, aux = out
                return val, aux
            return out, None

        vg_kwargs: dict[str, object] = {
            "has_aux": True,
            "holomorphic": holomorphic,
            "allow_int": allow_int,
        }
        if reduce_axes:
            vg_kwargs["reduce_axes"] = reduce_axes
        vg = jax.value_and_grad(pure, **vg_kwargs)
        (value, aux), grads_target = vg(target, rest, other_args, kwargs)
        if has_aux:
            return (value, aux), grads_target
        return value, grads_target

    return cast(F, wrapped)


def grad(
    fn: F | None = None,
    *,
    wrt: SelectorSugar = "parameters",
    argnum: int | None = None,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> F:
    """Differentiate ``fn`` and return only the gradients.

    Thin module-aware wrapper around :func:`value_and_grad` that
    discards the value component. With ``has_aux=True`` returns
    ``(grads, aux)``; otherwise returns ``grads`` directly. See
    :func:`value_and_grad` for the meaning of every argument.

    Args:
        fn: Function to differentiate. When omitted returns a decorator
            factory.
        wrt: Selector for the differentiation target.
        argnum: Positional index of the differentiated module.
        has_aux: When ``True`` return ``(grads, aux)``.
        holomorphic: Forwarded to :func:`jax.value_and_grad`.
        allow_int: Forwarded to :func:`jax.value_and_grad`.
        reduce_axes: Forwarded to :func:`jax.value_and_grad` when
            non-empty.

    Returns:
        A wrapped callable returning ``grads`` (or ``(grads, aux)`` if
        ``has_aux`` is set), where ``grads`` is a
        :class:`~spectrax.State` parallel to the selected subset.

    Raises:
        TypeError: If the resolved positional argument is not a
            :class:`~spectrax.Module`.
    """
    if fn is None:
        return cast(
            F,
            lambda f: grad(
                f,
                wrt=wrt,
                argnum=argnum,
                has_aux=has_aux,
                holomorphic=holomorphic,
                allow_int=allow_int,
                reduce_axes=reduce_axes,
            ),
        )

    vg = value_and_grad(
        fn,
        wrt=wrt,
        argnum=argnum,
        has_aux=has_aux,
        holomorphic=holomorphic,
        allow_int=allow_int,
        reduce_axes=reduce_axes,
    )

    @functools.wraps(fn)
    def wrapped(*args: object, **kwargs: object) -> object:
        """Return only the gradient half of the :func:`value_and_grad` output.

        Strips the value (and aux when ``has_aux`` is unset) from the
        underlying :func:`value_and_grad` return so the user sees the
        same surface as :func:`jax.grad`.

        Args:
            *args: Additional positional arguments forwarded to the wrapped callable or backend.
            **kwargs: Additional keyword arguments forwarded to the wrapped callable or backend.

        Returns:
            Return only the gradient half of the :func:`value_and_grad` output.
        """
        out = vg(*args, **kwargs)
        if has_aux:
            (_, aux), grads = out
            return grads, aux
        _, grads = out
        return grads

    return cast(F, wrapped)


def vjp(
    fn: F | None = None,
    *primals: object,
    has_aux: bool = False,
    reduce_axes: Sequence[AxisName] = (),
    mutable: SelectorSugar = (),
) -> object:
    """Module-aware :func:`jax.vjp`.

    Two call shapes are supported:

    * **Direct**: ``spx.vjp(fn, primal_a, primal_b, ...)`` immediately
      runs the forward pass and returns ``(out, pullback)`` (or
      ``(out, pullback, aux)`` when ``has_aux``).
    * **Decorator/wrapper**: ``spx.vjp(fn, has_aux=...)`` returns a
      wrapper accepting positional primals — useful for building partial
      VJPs with shared options.

    Module primals are exported to ``(GraphDef, State)`` and the pure
    body is built by
    :func:`~spectrax.transforms.split_merge.make_pure` /
    :func:`~spectrax.transforms.split_merge.make_pure_readonly` (or
    their single-positional fast-path variants). ``mutable=`` controls
    which collections may be written back during the primal forward
    pass: when set, mutations are applied to the live module before
    returning. The returned pullback is itself pure — it only computes
    cotangents and never re-mutates modules.

    Cotangents handed back are converted into user-friendly form by
    :func:`_splice_module_cotangents` /
    :func:`_splice_one_cotangent`: the cotangent for each
    :class:`~spectrax.Module` primal is a :class:`~spectrax.State`,
    while non-module primals receive ordinary JAX pytree cotangents.

    Args:
        fn: Function whose VJP is being computed.
        *primals: Optional positional primals; when supplied, the
            forward pass executes immediately.
        has_aux: When ``True``, ``fn`` is expected to return
            ``(value, aux)``.
        reduce_axes: Forwarded to :func:`jax.vjp` when non-empty.
        mutable: Selector controlling write-back of module mutations
            during the primal pass.

    Returns:
        Either ``(out, pullback)`` / ``(out, pullback, aux)`` when
        primals are supplied, or a wrapper callable otherwise.

    Raises:
        TypeError: If keyword arguments are supplied in the wrapped-call
            form (the decorator-mode wrapper rejects kwargs to match
            :func:`jax.vjp` semantics).
    """
    if fn is None:
        return lambda f: vjp(f, has_aux=has_aux, reduce_axes=reduce_axes, mutable=mutable)

    if primals:
        return _vjp_call(fn, primals, has_aux=has_aux, reduce_axes=reduce_axes, mutable=mutable)

    @functools.wraps(fn)
    def wrapped(*args: object, **kwargs: object) -> object:
        """Decorator-mode wrapper: defer ``_vjp_call`` until primals arrive.

        Rejects keyword primals (since :func:`jax.vjp` does not support
        them), then forwards positional primals into the shared
        :func:`_vjp_call` implementation.

        Args:
            *args: Additional positional arguments forwarded to the wrapped callable or backend.
            **kwargs: Additional keyword arguments forwarded to the wrapped callable or backend.

        Returns:
            Result described by this helper.
        """
        _ensure_no_kwargs("vjp", kwargs)
        return _vjp_call(fn, args, has_aux=has_aux, reduce_axes=reduce_axes, mutable=mutable)

    return wrapped


def jvp(
    fn: F | None = None,
    primals: Sequence[object] | None | object = _MISSING,
    tangents: Sequence[object] | None | object = _MISSING,
    *,
    has_aux: bool = False,
    mutable: SelectorSugar = (),
) -> object:
    """Module-aware :func:`jax.jvp`.

    Two call shapes:

    * **Direct**: ``spx.jvp(fn, primals, tangents, has_aux=...)`` runs
      forward + tangent computation immediately.
    * **Decorator/wrapper**: ``spx.jvp(fn, has_aux=...)(primals, tangents)``
      defers until ``(primals, tangents)`` are supplied.

    For each :class:`~spectrax.Module` primal the user may supply a
    matching tangent as either a :class:`~spectrax.Module` (which is
    re-exported to its state), a :class:`~spectrax.State` directly, or
    any pytree matching the module's state structure.

    ``mutable=`` controls write-back of module mutations performed
    during the primal forward pass — those updates are applied to the
    live module via
    :func:`~spectrax.transforms.split_merge.apply_mutations` before
    returning. The tangent computation itself is always pure.

    Args:
        fn: Function whose JVP is being computed.
        primals: Sequence of primal values. Sentinel-defaulted so the
            decorator form is detectable.
        tangents: Sequence of tangent values, parallel to ``primals``.
        has_aux: When ``True``, ``fn`` returns ``(value, aux)`` and the
            wrapper returns ``(out, tangent_out, aux)``.
        mutable: Selector controlling write-back of module mutations
            during the forward pass.

    Returns:
        ``(out, tangent_out)`` (or ``(out, tangent_out, aux)`` when
        ``has_aux``) in direct-call form, or a wrapper callable
        otherwise.

    Raises:
        TypeError: If ``primals`` and ``tangents`` have different
            lengths.
    """
    if fn is None:
        return lambda f: jvp(f, has_aux=has_aux, mutable=mutable)

    if primals is not _MISSING and tangents is not _MISSING:
        return _jvp_call(fn, primals, tangents, has_aux=has_aux, mutable=mutable)

    @functools.wraps(fn)
    def wrapped(
        primals_: Sequence[object],
        tangents_: Sequence[object],
    ) -> object:
        """Decorator-mode wrapper: defer ``_jvp_call`` until ``(primals, tangents)`` arrive.

        Forwards the eventual ``(primals, tangents)`` invocation through
        the shared :func:`_jvp_call` implementation, preserving
        ``has_aux`` and ``mutable`` from the outer ``spx.jvp`` call.

        Args:
            primals_: Primals  value consumed by this operation.
            tangents_: Tangents  value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        return _jvp_call(fn, primals_, tangents_, has_aux=has_aux, mutable=mutable)

    return wrapped


def _find_first_module(args: tuple[object, ...]) -> int:
    """Return the positional index of the first :class:`~spectrax.Module` in ``args``.

    Used by :func:`value_and_grad` (and transitively :func:`grad`) to
    pick the differentiation target when the caller did not pass an
    explicit ``argnum``.

    Args:
        args: Positional arguments forwarded to the wrapped function.

    Returns:
        The index of the first :class:`~spectrax.Module`.

    Raises:
        TypeError: If no positional argument is a :class:`~spectrax.Module`.
    """
    for i, a in enumerate(args):
        if isinstance(a, Module):
            return i
    raise TypeError("spectrax.grad requires at least one Module argument")


def _ensure_no_kwargs(name: str, kwargs: dict[str, object]) -> None:
    """Reject keyword arguments for transforms whose public API mirrors raw JAX.

    :func:`jax.vjp` and :func:`jax.jvp` do not accept keyword arguments
    in their wrapped-call form, so the spectrax wrappers refuse them
    too rather than silently dropping the kwargs on the floor.

    Args:
        name: Public name of the calling transform, embedded in the
            error message.
        kwargs: Keyword arguments collected by the wrapper.

    Raises:
        TypeError: If ``kwargs`` is non-empty.
    """
    if kwargs:
        raise TypeError(
            f"spectrax.{name}() does not support keyword arguments in wrapped-call form. "
            "Wrap the function with lambda/partial if you need kwargs."
        )


def _split_module_tangents(
    refs: list[object], tangents: tuple[object, ...]
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Separate the module-state tangents from the rest of the positional tangents.

    For every located module in ``refs``, looks up the tangent at the
    same positional index, exports it to a :class:`~spectrax.State` if
    the user supplied a :class:`~spectrax.Module` (otherwise keeps the
    pytree as-is), and replaces the entry in the stripped-tangents tuple
    with ``None``. The result is two parallel tuples that match the
    pure function's ``(states, stripped_args, stripped_kwargs)`` shape.

    Args:
        refs: Located module refs from
            :func:`~spectrax.transforms.split_merge.locate_and_strip_fast`.
        tangents: User-supplied positional tangents.

    Returns:
        A pair ``(state_tangents, stripped_tangents)`` parallel to
        ``refs`` and ``tangents`` respectively.
    """
    stripped_tangents = list(tangents)
    state_tangents: list[object] = []
    for ref in refs:
        tangent = tangents[ref.locator]
        if isinstance(tangent, Module):
            _, tangent = export(tangent)
        state_tangents.append(tangent)
        stripped_tangents[ref.locator] = None
    return tuple(state_tangents), tuple(stripped_tangents)


def _splice_module_cotangents(
    refs: list[object], stripped_cotangents: tuple[object, ...], state_cotangents: tuple[object, ...]
) -> tuple[object, ...]:
    """Rebuild a cotangent tuple matching the original primal positions.

    Inverse of :func:`_split_module_tangents`: takes the cotangents JAX
    produced for the non-module pytree positions plus the cotangents for
    the module states, and slots the module cotangents back into the
    positions they occupied in the user's primal tuple.

    Args:
        refs: Located module refs from
            :func:`~spectrax.transforms.split_merge.locate_and_strip_fast`.
        stripped_cotangents: Cotangents for the non-module pytree
            positions, in the same order as the stripped positional args.
        state_cotangents: Cotangents for each module state, parallel to
            ``refs``.

    Returns:
        A tuple parallel to the user's original primal tuple with module
        cotangents re-inserted at their original positions.
    """
    out = list(stripped_cotangents)
    for ref, ct in zip(refs, state_cotangents, strict=False):
        out[ref.locator] = ct
    return tuple(out)


def _splice_one_cotangent(
    locator: int, other_cotangents: tuple[object, ...], state_cotangent: object
) -> tuple[object, ...]:
    """Rebuild a cotangent tuple for the single-positional-Module fast path.

    Args:
        locator: Position the module occupied in the user's primal
            tuple.
        other_cotangents: Cotangents JAX produced for every non-module
            primal, in the same order as the original user call minus
            the module slot.
        state_cotangent: Cotangent produced for the module's state.

    Returns:
        A tuple parallel to the user's original primal tuple.
    """
    out = list(other_cotangents)
    out.insert(locator, state_cotangent)
    return tuple(out)


def _zeros_like_tree(tree: object) -> object:
    """Build a zero-filled cotangent pytree with the same structure as ``tree``.

    Used to seed the cotangent for the ``new_states`` half of a
    ``pure_with_updates`` pullback so the user only ever sees the
    cotangent for the actual function output.

    Args:
        tree: PyTree consumed or produced by the helper.

    Returns:
        Result described by this helper.
    """
    return jax.tree.map(jax.numpy.zeros_like, tree)


def _state_is_empty(state: State) -> bool:
    """Return ``True`` when ``state`` contains no collections.

    Args:
        state: SpectraX state tree or transform state passed into the operation.

    Returns:
        Return ``True`` when ``state`` contains no collections.
    """
    return not state.collections()


def _module_like_to_state(value: object) -> State:
    """Coerce a Module-shaped cotangent into the public :class:`~spectrax.State` form.

    JAX returns cotangents whose pytree structure matches the primal's
    structure — so cotangents for a :class:`~spectrax.Module` primal
    arrive as a :class:`~spectrax.Module`-shaped object. This helper
    normalizes them to :class:`~spectrax.State` so users see one
    consistent cotangent type regardless of which internal autodiff
    path was taken.

    Args:
        value: The cotangent to convert.

    Returns:
        A :class:`~spectrax.State` carrying the same leaves.

    Raises:
        TypeError: If ``value`` is neither a :class:`~spectrax.Module`
            nor a :class:`~spectrax.State`.
    """
    if isinstance(value, State):
        return value
    if isinstance(value, Module):
        _, state = export(value)
        return state
    raise TypeError(f"Expected Module or State value, got {type(value).__name__}")


def _convert_direct_tangents(primals: tuple[object, ...], tangents: tuple[object, ...]) -> tuple[object, ...] | None:
    """Convert :class:`~spectrax.State` tangents into Module-pytree tangents.

    When ``mutable=()`` :func:`vjp` / :func:`jvp` use the direct-JAX
    autodiff path that operates on the live :class:`~spectrax.Module`
    pytree directly. JAX's autodiff requires tangents whose pytree
    structure matches the primal's — so :class:`~spectrax.State`-shaped
    tangents must be re-bound onto a temporary module and re-flattened
    against the primal's treedef before being handed to JAX.

    Args:
        primals: User-supplied primal tuple.
        tangents: User-supplied tangent tuple, parallel to ``primals``.

    Returns:
        A tuple of tangents matching each primal's pytree structure,
        or ``None`` when a tangent for a module primal was neither a
        :class:`~spectrax.Module` nor a :class:`~spectrax.State` (the
        caller falls back to the slow split/merge path in that case).
    """
    converted = list(tangents)
    for i, primal in enumerate(primals):
        if not isinstance(primal, Module):
            continue
        primal_treedef = jax.tree.structure(primal)
        tangent = tangents[i]
        if isinstance(tangent, Module):
            converted[i] = jax.tree_util.tree_unflatten(primal_treedef, jax.tree.leaves(tangent))
        elif isinstance(tangent, State):
            gdef, _ = export(primal)
            tangent_module = bind(gdef, tangent)
            converted[i] = jax.tree_util.tree_unflatten(primal_treedef, jax.tree.leaves(tangent_module))
        else:
            return None
    return tuple(converted)


def _convert_direct_cotangents(primals: tuple[object, ...], cotangents: tuple[object, ...]) -> tuple[object, ...]:
    """Translate raw JAX cotangents into the user-facing types.

    Walks ``primals`` / ``cotangents`` in parallel and converts every
    cotangent that corresponds to a :class:`~spectrax.Module` primal
    into a :class:`~spectrax.State` via :func:`_module_like_to_state`.
    Non-module cotangents pass through unchanged.

    Args:
        primals: User-supplied primal tuple; used to identify which
            cotangent positions correspond to modules.
        cotangents: JAX-produced cotangents, parallel to ``primals``.

    Returns:
        A tuple parallel to ``cotangents`` where every module-associated
        entry has been converted to :class:`~spectrax.State`.
    """
    out: list[object] = []
    for primal, cotangent in zip(primals, cotangents, strict=False):
        out.append(_module_like_to_state(cotangent) if isinstance(primal, Module) else cotangent)
    return tuple(out)


def _vjp_call(
    fn: Callable[..., object],
    primals: Sequence[object],
    *,
    has_aux: bool,
    reduce_axes: Sequence[AxisName],
    mutable: SelectorSugar,
) -> object:
    """Shared backend for direct and wrapped :func:`vjp` invocations.

    Branches on ``mutable_sel`` and on whether the call is the
    single-positional-Module fast path; selects the appropriate
    pure-body factory from :mod:`spectrax.transforms.split_merge` and
    threads the result through :func:`jax.vjp`. After a forward pass
    that captures mutations, write-back is performed via
    :func:`~spectrax.transforms.split_merge.apply_mutations`. The
    returned pullback uses :func:`_splice_module_cotangents` /
    :func:`_splice_one_cotangent` to reshape JAX cotangents back into
    the primal-tuple layout.

    Args:
        fn: Callable being wrapped, traced, transformed, or executed.
        primals: Primal inputs supplied to a differentiation transform.
        has_aux: Has aux value consumed by this operation.
        reduce_axes: Reduce axes value consumed by this operation.
        mutable: Mutable value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    args = tuple(primals)
    mutable_sel = resolve_mutable(mutable)
    direct_guarded = make_direct_readonly(fn)
    if mutable_sel is None:
        vjp_kwargs: dict[str, object] = {"has_aux": has_aux} if has_aux else {}
        if reduce_axes:
            vjp_kwargs["reduce_axes"] = reduce_axes
        if has_aux:
            out, pullback, aux = jax.vjp(direct_guarded, *args, **vjp_kwargs)

            def wrapped_pullback(cotangent: object) -> tuple[object, ...]:
                """User-facing pullback: lifts a JAX VJP into module/state-aware tangents.

                Translates the cotangent into the right shape for the wrapped
                function, calls the underlying ``vjp_fn``, and re-packages the
                result so the user sees module-shaped grads (where applicable)
                rather than raw state pytrees.

                Args:
                    cotangent: Cotangent supplied to a pullback or transpose rule.

                Returns:
                    Result described by this helper.
                """
                cotangents = pullback(cotangent)
                return _convert_direct_cotangents(args, cotangents)

            return out, wrapped_pullback, aux

        out, pullback = jax.vjp(direct_guarded, *args, **vjp_kwargs)

        def wrapped_pullback(cotangent: object) -> tuple[object, ...]:
            """User-facing pullback: lifts a JAX VJP into module/state-aware tangents.

            Translates the cotangent into the right shape for the wrapped
            function, calls the underlying ``vjp_fn``, and re-packages the
            result so the user sees module-shaped grads (where applicable)
            rather than raw state pytrees.

            Args:
                cotangent: Cotangent supplied to a pullback or transpose rule.

            Returns:
                Result described by this helper.
            """
            cotangents = pullback(cotangent)
            return _convert_direct_cotangents(args, cotangents)

        return out, wrapped_pullback

    refs, stripped_args = locate_and_strip_fast(args)
    if len(refs) == 1 and refs[0].kind == "arg":
        ref = refs[0]
        locator = int(ref.locator)
        other_args = args[:locator] + args[locator + 1 :]
        state_in = ref.state
        pure_one = (
            make_pure_readonly_single_positional(fn, ref)
            if mutable_sel is None
            else make_pure_single_positional(fn, ref)
        )

        if has_aux:
            vjp_kwargs: dict[str, object] = {"has_aux": True}
            if reduce_axes:
                vjp_kwargs["reduce_axes"] = reduce_axes
            if mutable_sel is None:
                out, pullback, aux = jax.vjp(pure_one, state_in, *other_args, **vjp_kwargs)

                def wrapped_pullback(cotangent: object) -> tuple[object, ...]:
                    """User-facing pullback: lifts a JAX VJP into module/state-aware tangents.

                    Translates the cotangent into the right shape for the wrapped
                    function, calls the underlying ``vjp_fn``, and re-packages the
                    result so the user sees module-shaped grads (where applicable)
                    rather than raw state pytrees.

                    Args:
                        cotangent: Cotangent supplied to a pullback or transpose rule.

                    Returns:
                        Result described by this helper.
                    """
                    cotangents = pullback(cotangent)
                    return _splice_one_cotangent(locator, tuple(cotangents[1:]), cotangents[0])

                return out, wrapped_pullback, aux

            def pure_with_updates(state: State, *other: object) -> tuple[tuple[object, State], object]:
                """Closure that runs the (re-bound) function and returns ``(out, new_state)``.

                Used by ``jax.vjp`` / ``jax.grad`` so the autodiff transform sees a
                pure ``(state, *args) -> (output, new_state)`` interface even though
                the user wrote a stateful module-method. The caller separately
                applies ``new_state`` back to the live module via :func:`apply_mutations`.

                Args:
                    state: SpectraX state tree or transform state passed into the operation.
                    *other: Additional positional arguments forwarded to the wrapped callable or backend.

                Returns:
                    Result described by this helper.
                """
                (out, aux), new_state = pure_one(state, *other)
                return (out, new_state), aux

            (out, new_state), pullback, aux = jax.vjp(pure_with_updates, state_in, *other_args, **vjp_kwargs)
            apply_mutations([ref], [new_state], mutable_sel)
            zero_state = _zeros_like_tree(new_state)

            def wrapped_pullback(cotangent: object) -> tuple[object, ...]:
                """User-facing pullback: lifts a JAX VJP into module/state-aware tangents.

                Translates the cotangent into the right shape for the wrapped
                function, calls the underlying ``vjp_fn``, and re-packages the
                result so the user sees module-shaped grads (where applicable)
                rather than raw state pytrees.

                Args:
                    cotangent: Cotangent supplied to a pullback or transpose rule.

                Returns:
                    Result described by this helper.
                """
                cotangents = pullback((cotangent, zero_state))
                return _splice_one_cotangent(locator, tuple(cotangents[1:]), cotangents[0])

            return out, wrapped_pullback, aux

        vjp_kwargs = {}
        if reduce_axes:
            vjp_kwargs["reduce_axes"] = reduce_axes
        if mutable_sel is None:
            out, pullback = jax.vjp(pure_one, state_in, *other_args, **vjp_kwargs)

            def wrapped_pullback(cotangent: object) -> tuple[object, ...]:
                """User-facing pullback: lifts a JAX VJP into module/state-aware tangents.

                Translates the cotangent into the right shape for the wrapped
                function, calls the underlying ``vjp_fn``, and re-packages the
                result so the user sees module-shaped grads (where applicable)
                rather than raw state pytrees.

                Args:
                    cotangent: Cotangent supplied to a pullback or transpose rule.

                Returns:
                    Result described by this helper.
                """
                cotangents = pullback(cotangent)
                return _splice_one_cotangent(locator, tuple(cotangents[1:]), cotangents[0])

            return out, wrapped_pullback

        def pure_with_updates(state: State, *other: object) -> tuple[object, State]:
            """Closure that runs the (re-bound) function and returns ``(out, new_state)``.

            Used by ``jax.vjp`` / ``jax.grad`` so the autodiff transform sees a
            pure ``(state, *args) -> (output, new_state)`` interface even though
            the user wrote a stateful module-method. The caller separately
            applies ``new_state`` back to the live module via :func:`apply_mutations`.

            Args:
                state: SpectraX state tree or transform state passed into the operation.
                *other: Additional positional arguments forwarded to the wrapped callable or backend.

            Returns:
                Result described by this helper.
            """
            return cast(tuple[object, State], pure_one(state, *other))

        (out, new_state), pullback = jax.vjp(pure_with_updates, state_in, *other_args, **vjp_kwargs)
        apply_mutations([ref], [new_state], mutable_sel)
        zero_state = _zeros_like_tree(new_state)

        def wrapped_pullback(cotangent: object) -> tuple[object, ...]:
            """User-facing pullback: lifts a JAX VJP into module/state-aware tangents.

            Translates the cotangent into the right shape for the wrapped
            function, calls the underlying ``vjp_fn``, and re-packages the
            result so the user sees module-shaped grads (where applicable)
            rather than raw state pytrees.

            Args:
                cotangent: Cotangent supplied to a pullback or transpose rule.

            Returns:
                Result described by this helper.
            """
            cotangents = pullback((cotangent, zero_state))
            return _splice_one_cotangent(locator, tuple(cotangents[1:]), cotangents[0])

        return out, wrapped_pullback

    states_in = tuple(r.state for r in refs)
    pure = make_pure_readonly(fn, refs) if mutable_sel is None else make_pure(fn, refs)
    empty_kwargs: dict[str, object] = {}

    if has_aux:
        vjp_kwargs: dict[str, object] = {"has_aux": True}
        if reduce_axes:
            vjp_kwargs["reduce_axes"] = reduce_axes
        if mutable_sel is None:
            out, pullback, aux = jax.vjp(pure, states_in, stripped_args, empty_kwargs, **vjp_kwargs)

            def wrapped_pullback(cotangent: object) -> tuple[object, ...]:
                """User-facing pullback: lifts a JAX VJP into module/state-aware tangents.

                Translates the cotangent into the right shape for the wrapped
                function, calls the underlying ``vjp_fn``, and re-packages the
                result so the user sees module-shaped grads (where applicable)
                rather than raw state pytrees.

                Args:
                    cotangent: Cotangent supplied to a pullback or transpose rule.

                Returns:
                    Result described by this helper.
                """
                state_cts, arg_cts, _ = pullback(cotangent)
                return _splice_module_cotangents(refs, arg_cts, state_cts)

            return out, wrapped_pullback, aux

        def pure_with_updates(
            states: tuple[State, ...], stripped: tuple[object, ...], kwargs: dict[str, object]
        ) -> tuple[tuple[object, tuple[State, ...]], object]:
            """Closure that runs the (re-bound) function and returns ``(out, new_state)``.

            Used by ``jax.vjp`` / ``jax.grad`` so the autodiff transform sees a
            pure ``(state, *args) -> (output, new_state)`` interface even though
            the user wrote a stateful module-method. The caller separately
            applies ``new_state`` back to the live module via :func:`apply_mutations`.

            Args:
                states: States value consumed by this operation.
                stripped: Stripped value consumed by this operation.
                kwargs: Keyword arguments forwarded to the wrapped callable.

            Returns:
                Result described by this helper.
            """
            (out, aux), new_states = pure(states, stripped, kwargs)
            return (out, cast(tuple[State, ...], new_states)), aux

        (out, new_states), pullback, aux = jax.vjp(
            pure_with_updates, states_in, stripped_args, empty_kwargs, **vjp_kwargs
        )
        apply_mutations(refs, list(new_states), mutable_sel)
        zero_states = _zeros_like_tree(new_states)

        def wrapped_pullback(cotangent: object) -> tuple[object, ...]:
            """User-facing pullback: lifts a JAX VJP into module/state-aware tangents.

            Translates the cotangent into the right shape for the wrapped
            function, calls the underlying ``vjp_fn``, and re-packages the
            result so the user sees module-shaped grads (where applicable)
            rather than raw state pytrees.

            Args:
                cotangent: Cotangent supplied to a pullback or transpose rule.

            Returns:
                Result described by this helper.
            """
            state_cts, arg_cts, _ = pullback((cotangent, zero_states))
            return _splice_module_cotangents(refs, arg_cts, state_cts)

        return out, wrapped_pullback, aux

    vjp_kwargs = {}
    if reduce_axes:
        vjp_kwargs["reduce_axes"] = reduce_axes
    if mutable_sel is None:
        out, pullback = jax.vjp(pure, states_in, stripped_args, empty_kwargs, **vjp_kwargs)

        def wrapped_pullback(cotangent: object) -> tuple[object, ...]:
            """User-facing pullback: lifts a JAX VJP into module/state-aware tangents.

            Translates the cotangent into the right shape for the wrapped
            function, calls the underlying ``vjp_fn``, and re-packages the
            result so the user sees module-shaped grads (where applicable)
            rather than raw state pytrees.

            Args:
                cotangent: Cotangent supplied to a pullback or transpose rule.

            Returns:
                Result described by this helper.
            """
            state_cts, arg_cts, _ = pullback(cotangent)
            return _splice_module_cotangents(refs, arg_cts, state_cts)

        return out, wrapped_pullback

    def pure_with_updates(
        states: tuple[State, ...], stripped: tuple[object, ...], kwargs: dict[str, object]
    ) -> tuple[object, tuple[State, ...]]:
        """Closure that runs the (re-bound) function and returns ``(out, new_state)``.

        Used by ``jax.vjp`` / ``jax.grad`` so the autodiff transform sees a
        pure ``(state, *args) -> (output, new_state)`` interface even though
        the user wrote a stateful module-method. The caller separately
        applies ``new_state`` back to the live module via :func:`apply_mutations`.

        Args:
            states: States value consumed by this operation.
            stripped: Stripped value consumed by this operation.
            kwargs: Keyword arguments forwarded to the wrapped callable.

        Returns:
            Result described by this helper.
        """
        out, new_states = pure(states, stripped, kwargs)
        return out, new_states

    (out, new_states), pullback = jax.vjp(pure_with_updates, states_in, stripped_args, empty_kwargs, **vjp_kwargs)
    apply_mutations(refs, list(new_states), mutable_sel)
    zero_states = _zeros_like_tree(new_states)

    def wrapped_pullback(cotangent: object) -> tuple[object, ...]:
        """User-facing pullback: lifts a JAX VJP into module/state-aware tangents.

        Translates the cotangent into the right shape for the wrapped
        function, calls the underlying ``vjp_fn``, and re-packages the
        result so the user sees module-shaped grads (where applicable)
        rather than raw state pytrees.

        Args:
            cotangent: Cotangent supplied to a pullback or transpose rule.

        Returns:
            Result described by this helper.
        """
        state_cts, arg_cts, _ = pullback((cotangent, zero_states))
        return _splice_module_cotangents(refs, arg_cts, state_cts)

    return out, wrapped_pullback


def _jvp_call(
    fn: Callable[..., object],
    primals: Sequence[object] | None | object,
    tangents: Sequence[object] | None | object,
    *,
    has_aux: bool,
    mutable: SelectorSugar,
) -> object:
    """Shared backend for direct and wrapped :func:`jvp` invocations.

    Validates arity, exports module primals to states, splits tangents
    via :func:`_split_module_tangents`, and dispatches to
    :func:`jax.jvp` against either the single-positional or general
    pure-body factory. When ``has_aux`` is set the implementation
    bifurcates: the primal pass runs once to obtain ``(out, aux)`` and
    a separate ``has_aux=False``-flavored pure body is built for the
    actual JVP so JAX never sees the auxiliary output. When
    ``mutable_sel`` is non-``None``, captured mutations from the primal
    pass are written back to the live module before returning.

    Args:
        fn: Callable being wrapped, traced, transformed, or executed.
        primals: Primal inputs supplied to a differentiation transform.
        tangents: Tangent inputs supplied to a JVP transform.
        has_aux: Has aux value consumed by this operation.
        mutable: Mutable value consumed by this operation.

    Returns:
        Result described by this helper.
    """
    args = tuple(primals)
    tangent_args = tuple(tangents)
    if len(args) != len(tangent_args):
        raise TypeError(
            f"spectrax.jvp() requires primals and tangents with the same arity; "
            f"got {len(args)} primals and {len(tangent_args)} tangents."
        )

    mutable_sel = resolve_mutable(mutable)
    refs, stripped_args = locate_and_strip_fast(args)
    if len(refs) == 1 and refs[0].kind == "arg":
        ref = refs[0]
        locator = int(ref.locator)
        other_args = args[:locator] + args[locator + 1 :]
        module_tangent = tangent_args[locator]
        if isinstance(module_tangent, Module):
            _, module_tangent = export(module_tangent)
        other_tangents = tangent_args[:locator] + tangent_args[locator + 1 :]
        pure_one = (
            make_pure_readonly_single_positional(fn, ref)
            if mutable_sel is None
            else make_pure_single_positional(fn, ref)
        )

        if has_aux:
            if mutable_sel is None:
                out, aux = pure_one(ref.state, *other_args)
            else:
                (out, aux), new_state = pure_one(ref.state, *other_args)
                apply_mutations([ref], [new_state], mutable_sel)

            def fn_noaux(*a: object) -> object:
                """``has_aux=False`` adapter: drop the aux output so ``jax.grad`` sees a scalar.

                Args:
                    *a: Additional positional arguments forwarded to the wrapped callable or backend.

                Returns:
                    Result described by this helper.
                """
                value, _aux = fn(*a)
                return value

            pure_noaux = (
                make_pure_readonly_single_positional(fn_noaux, ref)
                if mutable_sel is None
                else make_pure_single_positional(fn_noaux, ref)
            )
            if mutable_sel is None:
                _, tangent_out = jax.jvp(
                    pure_noaux,
                    (ref.state, *other_args),
                    (module_tangent, *other_tangents),
                )
            else:

                def out_only(state: State, *other: object) -> object:
                    """Pure-output adapter: return only the primary output (no state) for ``jax.value_and_grad``.

                    Args:
                        state: SpectraX state tree or transform state passed into the operation.
                        *other: Additional positional arguments forwarded to the wrapped callable or backend.

                    Returns:
                        Result described by this helper.
                    """
                    out_only_val, _ignored_state = pure_noaux(state, *other)
                    return out_only_val

                _, tangent_out = jax.jvp(
                    out_only,
                    (ref.state, *other_args),
                    (module_tangent, *other_tangents),
                )
            return out, tangent_out, aux

        if mutable_sel is None:
            out, tangent_out = jax.jvp(
                pure_one,
                (ref.state, *other_args),
                (module_tangent, *other_tangents),
            )
            return out, tangent_out

        def pure_with_updates(state: State, *other: object) -> tuple[object, State]:
            """Closure that runs the (re-bound) function and returns ``(out, new_state)``.

            Used by ``jax.jvp`` so the autodiff transform sees a
            pure ``(state, *args) -> (output, new_state)`` interface even though
            the user wrote a stateful module-method. The caller separately
            applies ``new_state`` back to the live module via :func:`apply_mutations`.

            Args:
                state: SpectraX state tree or transform state passed into the operation.
                *other: Additional positional arguments forwarded to the wrapped callable or backend.

            Returns:
                Result described by this helper.
            """
            return cast(tuple[object, State], pure_one(state, *other))

        (out, new_state), (tangent_out, _tangent_state) = jax.jvp(
            pure_with_updates,
            (ref.state, *other_args),
            (module_tangent, *other_tangents),
        )
        apply_mutations([ref], [new_state], mutable_sel)
        return out, tangent_out

    states_in = tuple(r.state for r in refs)
    state_tangents, stripped_tangents = _split_module_tangents(refs, tangent_args)
    empty_kwargs: dict[str, object] = {}
    pure = make_pure_readonly(fn, refs) if mutable_sel is None else make_pure(fn, refs)

    if has_aux:
        if mutable_sel is None:
            out, aux = pure(states_in, stripped_args, empty_kwargs)
        else:
            (out, aux), new_states = pure(states_in, stripped_args, empty_kwargs)
            apply_mutations(refs, list(new_states), mutable_sel)

        def fn_noaux(*a: object) -> object:
            """``has_aux=False`` adapter: drop the aux output so ``jax.grad`` sees a scalar.

            Args:
                *a: Additional positional arguments forwarded to the wrapped callable or backend.

            Returns:
                Result described by this helper.
            """
            value, _aux = fn(*a)
            return value

        pure_noaux = make_pure_readonly(fn_noaux, refs) if mutable_sel is None else make_pure(fn_noaux, refs)
        if mutable_sel is None:
            _, tangent_out = jax.jvp(
                pure_noaux,
                (states_in, stripped_args, empty_kwargs),
                (state_tangents, stripped_tangents, empty_kwargs),
            )
        else:

            def out_only(states: tuple[State, ...], stripped: tuple[object, ...], kwargs: dict[str, object]) -> object:
                """Pure-output adapter: return only the primary output (no state) for ``jax.value_and_grad``.

                Args:
                    states: States value consumed by this operation.
                    stripped: Stripped value consumed by this operation.
                    kwargs: Keyword arguments forwarded to the wrapped callable.

                Returns:
                    Result described by this helper.
                """
                out_only_val, _ignored_states = pure_noaux(states, stripped, kwargs)
                return out_only_val

            _, tangent_out = jax.jvp(
                out_only,
                (states_in, stripped_args, empty_kwargs),
                (state_tangents, stripped_tangents, empty_kwargs),
            )
        return out, tangent_out, aux

    if mutable_sel is None:
        out, tangent_out = jax.jvp(
            pure,
            (states_in, stripped_args, empty_kwargs),
            (state_tangents, stripped_tangents, empty_kwargs),
        )
        return out, tangent_out

    def pure_with_updates(
        states: tuple[State, ...], stripped: tuple[object, ...], kwargs: dict[str, object]
    ) -> tuple[object, tuple[State, ...]]:
        """Closure that runs the (re-bound) function and returns ``(out, new_state)``.

        Used by ``jax.jvp`` so the autodiff transform sees a
        pure ``(state, *args) -> (output, new_state)`` interface even though
        the user wrote a stateful module-method. The caller separately
        applies ``new_state`` back to the live module via :func:`apply_mutations`.

        Args:
            states: States value consumed by this operation.
            stripped: Stripped value consumed by this operation.
            kwargs: Keyword arguments forwarded to the wrapped callable.

        Returns:
            Result described by this helper.
        """
        out, new_states = pure(states, stripped, kwargs)
        return out, new_states

    (out, new_states), (tangent_out, _tangent_states) = jax.jvp(
        pure_with_updates,
        (states_in, stripped_args, empty_kwargs),
        (state_tangents, stripped_tangents, empty_kwargs),
    )
    apply_mutations(refs, list(new_states), mutable_sel)
    return out, tangent_out
