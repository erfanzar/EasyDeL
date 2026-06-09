# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Module-aware :func:`jax.vmap` wrapper.

The wrapped call walks ``args`` / ``kwargs``, exports each
:class:`~spectrax.Module` to a ``(GraphDef, State)`` snapshot, and
constructs a pure ``(states, args, kwargs) -> (out, new_states)`` body
that the underlying :func:`jax.vmap` consumes. Because the module
states are passed *outside* the user signature, the user's ``in_axes``
spec only refers to the non-module pytree arguments — the module is
always lifted with ``in_axes=None`` so its parameters and buffers are
broadcast across the mapped axis (use :func:`split_rngs` /
:class:`StateAxes` to override that for rng-bearing collections).

Mutations to declared-mutable collections are extracted post-vmap with
``out_axes=None`` and applied back to the live module exactly once via
:func:`~spectrax.transforms.split_merge.apply_mutations`. When
``mutable=()`` the readonly path is used and any captured mutation
raises :class:`~spectrax.IllegalMutationError`.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TypeVar, cast

import jax

from ..core.selector import SelectorSugar
from .split_merge import (
    apply_mutations,
    locate_and_strip,
    locate_and_strip_fast,
    make_pure,
    make_pure_readonly,
    make_pure_readonly_single_positional,
    make_pure_single_positional,
    resolve_mutable,
)

__all__ = ["vmap"]

F = TypeVar("F", bound=Callable[..., object])

AxisName = object
"""Placeholder for JAX axis-name sentinels (hashable values)."""


def _specialized_in_axes(in_axes: object, nargs: int, locator: int) -> tuple[object, ...]:
    """Strip the module position from a user-provided positional ``in_axes`` spec.

    Used on the single-positional-Module fast path: the underlying pure
    function takes ``(state, *non_module_args)`` so the per-arg axis spec
    handed to :func:`jax.vmap` must omit the entry that corresponded to
    the module argument in the user's original signature.

    Args:
        in_axes: Either a per-arg tuple/list of axis specs, or a scalar
            spec broadcast to every argument.
        nargs: Total number of positional arguments the user passed.
        locator: Index of the module argument in the user's call.

    Returns:
        A tuple of length ``nargs - 1`` containing the axis specs for
        every non-module positional argument, in their original order.
    """
    if isinstance(in_axes, tuple):
        axes = in_axes
    elif isinstance(in_axes, list):
        axes = tuple(in_axes)
    else:
        return tuple(in_axes for _ in range(nargs - 1))
    return axes[:locator] + axes[locator + 1 :]


def vmap(
    fn: F | None = None,
    *,
    mutable: SelectorSugar = (),
    in_axes: object = 0,
    out_axes: object = 0,
    axis_name: AxisName | None = None,
    axis_size: int | None = None,
    spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None,
    sum_match: bool = False,
) -> F:
    """Vectorize ``fn`` along a leading batch axis with full module support.

    Behaves like :func:`jax.vmap` except that any
    :class:`~spectrax.Module` positional or keyword argument is exported
    to a ``(GraphDef, State)`` snapshot, threaded through the underlying
    :func:`jax.vmap` with ``in_axes=None`` (broadcast), and rebound on
    the inside before ``fn`` is called. The user's ``in_axes`` /
    ``out_axes`` only describe the non-module pytree leaves.

    Two execution paths are taken:

    * **Single positional Module + no kwargs**: dispatches through
      :func:`~spectrax.transforms.split_merge.make_pure_single_positional`
      (or its readonly variant when ``mutable_sel`` resolves to ``None``).
      The ``in_axes`` spec is stripped of the module position via
      :func:`_specialized_in_axes` so the lifted state slots in cleanly.
    * **General case**: uses
      :func:`~spectrax.transforms.split_merge.make_pure` and lifts all
      module states as a single ``(states, args, kwargs)`` triple with
      ``in_axes=(None, in_axes, None)``.

    Mutations declared via ``mutable=`` come out of vmap with
    ``out_axes=None`` (single, axis-free post-vmap state) and are pushed
    back into the live module by
    :func:`~spectrax.transforms.split_merge.apply_mutations`. When
    ``mutable`` resolves to ``None`` the readonly pure body raises
    :class:`~spectrax.IllegalMutationError` on any captured write.

    Args:
        fn: Function to vmap. When omitted, returns a decorator factory
            so ``@spx.vmap(in_axes=...)`` works.
        mutable: Selector (or selector sugar) controlling which module
            collections may be written back after the transform.
        in_axes: Forwarded to :func:`jax.vmap`; refers to non-module
            argument positions only. Modules are always handled with
            ``in_axes=None``.
        out_axes: Forwarded to :func:`jax.vmap`; the captured
            ``new_states`` are independently mapped with
            ``out_axes=None``.
        axis_name: Optional named axis used by collective primitives.
        axis_size: Optional explicit axis size; required when no input
            specifies one.
        spmd_axis_name: Optional axis name (or tuple) for SPMD lowering.
        sum_match: Forwarded to :func:`jax.vmap`.

    Returns:
        A wrapped function with the same call signature as ``fn``.
    """
    if fn is None:
        return cast(
            F,
            lambda f: vmap(
                f,
                mutable=mutable,
                in_axes=in_axes,
                out_axes=out_axes,
                axis_name=axis_name,
                axis_size=axis_size,
                spmd_axis_name=spmd_axis_name,
                sum_match=sum_match,
            ),
        )

    mutable_sel = resolve_mutable(mutable)
    empty_kwargs: dict[str, object] = {}

    @functools.wraps(fn)
    def wrapped(*args: object, **kwargs: object) -> object:
        """Locate modules, build a pure callable, and dispatch through :func:`jax.vmap`.

        Selects the single-positional-Module fast path when there are no
        kwargs and exactly one positional :class:`~spectrax.Module`; falls
        back to the general states-tuple form otherwise. After the vmap
        call returns, declared-mutable updates are written back to the
        live modules via
        :func:`~spectrax.transforms.split_merge.apply_mutations`.

        Args:
            *args: Additional positional arguments forwarded to the wrapped callable or backend.
            **kwargs: Additional keyword arguments forwarded to the wrapped callable or backend.

        Returns:
            Result described by this helper.
        """
        if kwargs:
            refs, stripped_args, stripped_kwargs = locate_and_strip(args, kwargs)
        else:
            refs, stripped_args = locate_and_strip_fast(args)
            stripped_kwargs = empty_kwargs
        if not kwargs and len(refs) == 1 and refs[0].kind == "arg":
            ref = refs[0]
            locator = int(ref.locator)
            other_args = stripped_args[:locator] + stripped_args[locator + 1 :]
            other_in_axes = _specialized_in_axes(in_axes, len(args), locator)
            pure = (
                make_pure_readonly_single_positional(fn, ref)
                if mutable_sel is None
                else make_pure_single_positional(fn, ref)
            )
            vmapped = jax.vmap(
                pure,
                in_axes=(None, *other_in_axes),
                out_axes=out_axes if mutable_sel is None else (out_axes, None),
                axis_name=axis_name,
                axis_size=axis_size,
                spmd_axis_name=spmd_axis_name,
                sum_match=sum_match,
            )
            if mutable_sel is None:
                return vmapped(ref.state, *other_args)
            out, new_state = vmapped(ref.state, *other_args)
            apply_mutations([ref], [new_state], mutable_sel)
            return out
        pure = make_pure_readonly(fn, refs) if mutable_sel is None else make_pure(fn, refs)
        pure_in_axes = (None, in_axes, None)
        states_in = tuple(r.state for r in refs)
        if mutable_sel is None:
            vmapped = jax.vmap(
                pure,
                in_axes=pure_in_axes,
                out_axes=out_axes,
                axis_name=axis_name,
                axis_size=axis_size,
                spmd_axis_name=spmd_axis_name,
                sum_match=sum_match,
            )
            return vmapped(states_in, stripped_args, stripped_kwargs)
        vmapped = jax.vmap(
            pure,
            in_axes=pure_in_axes,
            out_axes=(out_axes, None),
            axis_name=axis_name,
            axis_size=axis_size,
            spmd_axis_name=spmd_axis_name,
            sum_match=sum_match,
        )
        out, new_states = vmapped(states_in, stripped_args, stripped_kwargs)
        apply_mutations(refs, list(new_states), mutable_sel)
        return out

    return cast(F, wrapped)
