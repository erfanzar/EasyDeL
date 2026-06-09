# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for mixed-precision policy wiring and :func:`promote_dtype`."""

from __future__ import annotations

import jax.numpy as jnp

from spectrax.core.policy import Policy, current_policy, push_policy
from spectrax.functional.util import promote_dtype
from spectrax.nn.linear import Linear
from spectrax.rng.rngs import Rngs


def test_current_policy_empty_outside_context():
    """Outside any push the stack is empty."""
    assert current_policy() is None


def test_push_policy_sets_and_restores_current():
    """``push_policy`` updates and restores the stack."""
    pol = Policy(compute_dtype=jnp.bfloat16)
    with push_policy(pol):
        assert current_policy() is pol
    assert current_policy() is None


def test_push_policy_none_is_noop():
    """Pushing ``None`` does not change the stack."""
    with push_policy(None):
        assert current_policy() is None


def test_linear_with_compute_bf16_produces_bf16_output():
    """Compute-dtype policy casts inputs/parameters to bf16."""
    m = Linear(4, 4, rngs=Rngs(0))
    m.policy = Policy(compute_dtype=jnp.bfloat16)
    y = m(jnp.ones((1, 4), dtype=jnp.float32))
    assert y.dtype == jnp.bfloat16


def test_linear_with_output_fp32_casts_output():
    """Output-dtype policy forces the output dtype."""
    m = Linear(4, 4, rngs=Rngs(0))
    m.policy = Policy(compute_dtype=jnp.bfloat16, output_dtype=jnp.float32)
    y = m(jnp.ones((1, 4), dtype=jnp.float32))
    assert y.dtype == jnp.float32


def test_promote_dtype_promotes_to_common_dtype():
    """``promote_dtype`` picks the higher-precision common dtype."""
    a = jnp.ones((1,), dtype=jnp.float16)
    b = jnp.ones((1,), dtype=jnp.float32)
    ap, bp = promote_dtype(a, b)
    assert ap.dtype == bp.dtype == jnp.float32


def test_promote_dtype_explicit_dtype_overrides():
    """Explicit ``dtype`` beats inference."""
    a = jnp.ones((1,), dtype=jnp.float16)
    b = jnp.ones((1,), dtype=jnp.float32)
    ap, bp = promote_dtype(a, b, dtype=jnp.float16)
    assert ap.dtype == bp.dtype == jnp.float16


def test_policy_descendant_reads_ancestor_policy():
    """Descendants read the policy pushed by the outermost ``__call__``."""
    from spectrax.core.module import Module

    class Wrapper(Module):
        """Fixture module for testing."""

        def __init__(self, rngs):
            """Initialize with inner."""
            super().__init__()
            self.inner = Linear(4, 4, rngs=rngs)

        def forward(self, x):
            """Run the forward pass."""
            return self.inner(x)

    w = Wrapper(Rngs(0))
    w.policy = Policy(compute_dtype=jnp.bfloat16)
    y = w(jnp.ones((1, 4), dtype=jnp.float32))
    assert y.dtype == jnp.bfloat16
