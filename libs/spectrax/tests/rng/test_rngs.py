# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.rng.rngs`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import spectrax as spx
from spectrax.core.graph import export
from spectrax.core.stage_assignment import PIPELINE_STAGE_METADATA_KEY, assign_stage
from spectrax.rng.rngs import Rngs, RngStream


def test_rngstream_rejects_non_1d_key():
    """A non-1-D key raises :class:`ValueError`."""
    with pytest.raises(ValueError):
        RngStream(jnp.ones((2, 2), dtype=jnp.uint32))


def test_rngstream_packed_leaf_contains_key_and_counter():
    """The packed leaf is longer than the key by two (counter_hi, counter_lo)."""
    key = jax.random.PRNGKey(0)
    rs = RngStream(key)
    assert rs.value.shape[0] == key.shape[0] + 2


def test_rngstream_does_not_inherit_pipeline_stage_assignment():
    """RNG stream state is global mutable state, not stage-owned model state."""
    with assign_stage(total=4, current=2):
        rs = RngStream(jax.random.PRNGKey(0))
        rngs = Rngs(0)
        _ = rngs.parameters

    assert PIPELINE_STAGE_METADATA_KEY not in rs.metadata
    assert PIPELINE_STAGE_METADATA_KEY not in rngs.stream("default").metadata
    assert PIPELINE_STAGE_METADATA_KEY not in rngs.stream("parameters").metadata


def test_rngstream_next_key_advances_counter():
    """Each ``next_key`` call advances the low counter word."""
    rs = RngStream(jax.random.PRNGKey(0))
    k1 = rs.next_key()
    k2 = rs.next_key()
    assert not jnp.array_equal(k1, k2)


def test_rngstream_next_key_uses_high_counter_word():
    """Streams separated by 2**32 draws must not collide."""
    base = jax.random.PRNGKey(0)
    start = RngStream(base, counter=0)
    wrapped = RngStream(base, counter=2**32)

    assert not jnp.array_equal(start.next_key(), wrapped.next_key())


def test_rngstream_fold_in_resets_counter_and_changes_key():
    """``fold_in`` returns a new stream derived from the current key."""
    rs = RngStream(jax.random.PRNGKey(0))
    a = rs.fold_in(1)
    b = rs.fold_in(2)
    assert not jnp.array_equal(a.value, b.value)


def test_rngs_default_seed():
    """``Rngs(seed)`` creates a default stream producing a typed PRNG key."""
    r = Rngs(0)
    k = r.key()
    assert jnp.issubdtype(k.dtype, jax.dtypes.prng_key)


def test_rngs_named_stream_explicit_seed():
    """Keyword streams in the constructor are seeded independently."""
    r = Rngs(0, dropout=99)
    k_def = r.key()
    k_drop = r.key("dropout")
    assert not jnp.array_equal(k_def, k_drop)


def test_rngs_lazy_stream_via_getattr_fold_in():
    """Accessing an unseeded stream derives it from ``default``; repeat
    accesses advance its counter and produce different keys.
    """
    r = Rngs(0)
    k1 = r.dropout
    k2 = r.dropout
    assert not jnp.array_equal(k1, k2)


def test_rngs_lazy_stream_invalidates_export_cache():
    """Explicit stream creation mutates graph structure and must re-export."""
    r = Rngs(0)
    export(r)

    r.stream("dropout")

    assert ("rng", "dropout") in {(c, p) for c, p, _ in export(r)[1].items()}


def test_rngs_missing_named_key_advances_state_under_jit():
    """Lazy named keys must not create untracked graph leaves inside jit."""
    r = Rngs(0)

    @spx.jit(mutable="rng")
    def draw(rngs):
        """Draw a random sample."""
        return jax.random.key_data(rngs.key("custom"))

    k1 = draw(r)
    k2 = draw(r)

    assert not jnp.array_equal(k1, k2)
    assert ("rng", "custom") not in {(c, p) for c, p, _ in export(r)[1].items()}
    _, _, lo = r.stream("default")._unpack()
    assert int(lo) == 2


def test_rngs_same_seed_same_keys():
    """Identical seeds yield identical first keys."""
    assert jnp.array_equal(Rngs(5).key(), Rngs(5).key())


def test_rngs_fold_in_returns_new_rngs():
    """``fold_in`` returns a fresh :class:`Rngs` with a different default stream."""
    r = Rngs(0)
    r2 = r.fold_in("tag")
    assert isinstance(r2, Rngs)
    assert not jnp.array_equal(r2.stream("default").value, r.stream("default").value)


def test_rngs_fork_returns_n_independent_rngs():
    """``fork(n)`` produces ``n`` rngs with pairwise-distinct default keys."""
    r = Rngs(0)
    forked = r.fork(4)
    assert len(forked) == 4
    keys = [forked[i].key() for i in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not jnp.array_equal(keys[i], keys[j])


def test_rngs_fork_advances_parent_stream():
    """Repeated forks should not replay identical child seeds."""
    r = Rngs(0)
    first = r.fork(2).as_stack()
    second = r.fork(2).as_stack()

    assert not jnp.array_equal(first, second)


def test_rngs_fork_rejects_non_positive_count():
    """Invalid fork counts fail before reaching JAX internals."""
    with pytest.raises(ValueError, match="fork count"):
        Rngs(0).fork(0)


def test_rngs_fork_as_stack_shape():
    """``_ForkedRngs.as_stack`` returns an ``(n, key_size)`` stacked array."""
    r = Rngs(0)
    forked = r.fork(3)
    stack = forked.as_stack()
    assert stack.shape[0] == 3


def test_rngs_getattr_underscore_raises():
    """Accessing ``_private`` via ``__getattr__`` raises :class:`AttributeError`."""
    r = Rngs(0)
    with pytest.raises(AttributeError):
        _ = r._no_such


def test_rngs_accepts_jax_key_directly():
    """Passing a raw PRNGKey as ``default`` works."""
    key = jax.random.PRNGKey(7)
    r = Rngs(key)
    assert r.stream("default").value.shape[0] == key.shape[0] + 2


def test_rngs_key_named_returns_fresh_key_each_call():
    """Successive calls on a named stream yield distinct keys."""
    r = Rngs(0)
    a = r.parameters
    b = r.parameters
    assert not jnp.array_equal(a, b)
