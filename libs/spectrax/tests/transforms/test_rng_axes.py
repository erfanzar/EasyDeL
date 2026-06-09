# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for :mod:`spectrax.transforms.rng_axes`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from spectrax.rng.rngs import Rngs
from spectrax.transforms.rng_axes import StateAxes, split_rngs, split_stream_keys


def test_state_axes_get_defaults():
    """Unknown collection returns the supplied default."""
    sa = StateAxes({"parameters": None, "rng": "split"})
    assert sa.get("parameters") is None
    assert sa.get("rng") == "split"
    assert sa.get("missing", 0) == 0


def test_state_axes_iter_yields_collection_axis_pairs():
    """Iteration yields ``(collection, axis)``."""
    sa = StateAxes({"a": None, "b": 1})
    d = dict(sa)
    assert d == {"a": None, "b": 1}


def test_split_stream_keys_produces_requested_count_of_distinct_keys():
    """Splitting produces distinct keys along a leading axis."""
    rngs = Rngs(0)
    keys = split_stream_keys(rngs.stream("default"), axis_size=4)
    assert keys.shape[0] == 4
    for i in range(4):
        for j in range(i + 1, 4):
            assert not jnp.array_equal(keys[i], keys[j])


def test_split_stream_keys_uses_stream_counter():
    """Repeated splits from one stream should not replay the same fork keys."""
    stream = Rngs(0).stream("default")
    first = split_stream_keys(stream, axis_size=2)
    second = split_stream_keys(stream, axis_size=2)

    assert not jnp.array_equal(first, second)


def test_split_stream_keys_rejects_non_positive_axis_size():
    """Invalid axis sizes fail before reaching JAX internals."""
    with pytest.raises(ValueError, match="axis_size"):
        split_stream_keys(Rngs(0).stream("default"), axis_size=0)
    with pytest.raises(ValueError, match="axis_size"):
        split_stream_keys(Rngs(0).stream("default"), axis_size=-1)


def test_split_rngs_yields_independent_forks():
    """Each fork produces a distinct key on the same stream name."""
    rngs = Rngs(0)
    with split_rngs(rngs, axis_size=3) as forks:
        keys = [fork.key("default") for fork in forks]
    assert not jnp.array_equal(jax.random.key_data(keys[0]), jax.random.key_data(keys[1]))
    assert not jnp.array_equal(jax.random.key_data(keys[1]), jax.random.key_data(keys[2]))


def test_split_rngs_rejects_non_positive_axis_size():
    """``split_rngs`` validates the public axis-size argument."""
    with pytest.raises(ValueError, match="axis_size"):
        with split_rngs(Rngs(0), axis_size=0):
            pass


def test_split_rngs_only_splits_requested_streams():
    """Streams outside ``only`` are cloned so forks cannot mutate each other."""
    rngs = Rngs(0, dropout=5)
    _, _, parent_lo_before = rngs.stream("dropout")._unpack()
    with split_rngs(rngs, axis_size=2, only=("default",)) as forks:
        assert forks[0].stream("dropout") is not forks[1].stream("dropout")
        assert forks[0].stream("dropout") is not rngs.stream("dropout")
        d0 = forks[0].key("default")
        d1 = forks[1].key("default")
        dr0 = forks[0].key("dropout")
        _, _, fork1_lo_after = forks[1].stream("dropout")._unpack()
        _, _, parent_lo_after = rngs.stream("dropout")._unpack()
    assert not jnp.array_equal(jax.random.key_data(d0), jax.random.key_data(d1))
    _ = dr0
    assert int(fork1_lo_after) == int(parent_lo_before)
    assert int(parent_lo_after) == int(parent_lo_before)


def test_split_rngs_clones_unsplit_streams_under_jit():
    """Cloning non-split streams must not convert traced counters to Python ints."""
    rngs = Rngs(0, dropout=5)

    @jax.jit
    def draw_from_unsplit_stream(r):
        """Draw from an unsplit RNG stream."""
        with split_rngs(r, axis_size=2, only=("default",)) as forks:
            return jax.random.key_data(forks[0].key("dropout"))

    out = draw_from_unsplit_stream(rngs)

    assert out.shape == (2,)


def test_split_rngs_advances_parent_counter():
    """Splitting advances the parent stream's counter by 1."""
    rngs = Rngs(0)
    stream = rngs.stream("default")
    _, _, lo_before = stream._unpack()
    split_stream_keys(stream, axis_size=2)
    _, _, lo_after = stream._unpack()
    assert int(lo_after) == int(lo_before) + 1
