# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Module-aware abstract evaluation via :func:`jax.eval_shape`.

This module exposes a single function, :func:`eval_shape`, which mirrors
:func:`jax.eval_shape` but understands :class:`~spectrax.Module` arguments.
It is the read-only exception in the spectrax transform family — it
participates in the same locate/strip/pure pipeline as :func:`~spectrax.jit`
and :func:`~spectrax.vmap`, but it discards any captured mutations so
the live input modules are never disturbed by an abstract trace.
"""

from __future__ import annotations

from collections.abc import Callable

import jax

from .split_merge import locate_and_strip, make_pure

__all__ = ["eval_shape"]


def eval_shape(fn: Callable[..., object], *args: object, **kwargs: object) -> object:
    """Compute the abstract output shape of ``fn`` without running it.

    Module-aware analogue of :func:`jax.eval_shape`. Module inputs are
    discovered by :func:`~spectrax.transforms.split_merge.locate_and_strip`,
    converted to ``(GraphDef, State)`` snapshots, and rebound inside the
    pure callable produced by :func:`~spectrax.transforms.split_merge.make_pure`
    so that the abstract trace sees the same calling convention as a
    real transform.

    Unlike mutating transforms (:func:`~spectrax.jit`, :func:`~spectrax.vmap`,
    :func:`~spectrax.scan`, …), the second element of the pure tuple — the
    captured ``new_states`` — is intentionally dropped. No ``mutable=``
    selector is consulted and :func:`~spectrax.transforms.split_merge.apply_mutations`
    is never invoked, so any variable writes that occur while abstract-
    evaluating remain confined to the abstract trace and never propagate
    back to the live input modules. This keeps :func:`eval_shape` safe to
    call on stateful module methods purely for shape inference.

    Args:
        fn: The function to abstract-evaluate. Receives the original
            module arguments (rebound from snapshots) plus any
            non-module pytree leaves.
        *args: Positional arguments to ``fn``; any
            :class:`~spectrax.Module` instance is snapshotted and rebound.
        **kwargs: Keyword arguments to ``fn``; same module handling.

    Returns:
        A pytree mirroring ``fn``'s return structure where every JAX
        array leaf is replaced by an abstract value carrying ``shape``
        and ``dtype``.
    """
    refs, stripped_args, stripped_kwargs = locate_and_strip(args, kwargs)
    pure = make_pure(fn, refs)
    states_in = tuple(ref.state for ref in refs)
    out, _new_states = jax.eval_shape(pure, states_in, stripped_args, stripped_kwargs)
    return out
