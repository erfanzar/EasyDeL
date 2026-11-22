# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bridge between stamped quantization rules and the ``nn`` layers.

Layers do not reach into :mod:`spectrax.quantization` directly. They ask
the two helpers here whether a rule was stamped on them, and get back
either the plain JAX op or a quantization-aware substitute with the rule
already bound. That keeps the quantized and unquantized code paths in a
layer to a single line, and keeps the "is there a rule?" question in one
place so it stays cheap — an attribute fetch and a short scan, no regular
expressions at call time.

A layer that was never passed through
:func:`spectrax.quantization.quantize_model` gets exactly
:func:`jax.lax.dot_general` and :func:`jax.numpy.einsum` back, so the
unquantized path is bit-for-bit what it was before quantization existed.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp

from ..core._typing import Array
from ..core.module import Module
from ..quantization import qdot_general, qeinsum, rule_for

__all__ = ["quantized_dot_general_for", "quantized_einsum_for"]


def quantized_dot_general_for(module: Module, *, rhs_is_weight: bool = True) -> Callable[..., Array]:
    """Return the ``dot_general`` a module should use for its weight contraction.

    Args:
        module: The layer performing the contraction.
        rhs_is_weight: Whether the learned weight is the right operand,
            which it is for the ``x @ W`` convention every dense layer
            here follows.

    Returns:
        :func:`jax.lax.dot_general` when no rule governs this module, or
        :func:`spectrax.quantization.qdot_general` with the rule bound.
    """
    rule = rule_for(module, "dot_general")
    if rule is None:
        return jax.lax.dot_general
    return functools.partial(qdot_general, rule=rule, rhs_is_weight=rhs_is_weight, lhs_is_weight=not rhs_is_weight)


def quantized_einsum_for(module: Module, *, rhs_is_weight: bool = True) -> Callable[..., Array]:
    """Return the ``einsum`` a module should use for its weight contraction.

    Args:
        module: The layer performing the contraction.
        rhs_is_weight: Whether the learned weight is the second operand.

    Returns:
        :func:`jax.numpy.einsum` when no rule governs this module, or
        :func:`spectrax.quantization.qeinsum` with the rule bound.
    """
    rule = rule_for(module, "einsum")
    if rule is None:
        return jnp.einsum
    return functools.partial(qeinsum, rule=rule, rhs_is_weight=rhs_is_weight, lhs_is_weight=not rhs_is_weight)
