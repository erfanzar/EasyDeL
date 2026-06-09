# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for :class:`spectrax.pipeline.StagesArray`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spectrax.runtime.types import StagesArray


def _two_device_shards():
    """Return a 2-shard StagesArray on the first two devices."""
    devs = jax.devices()[:2]
    if len(devs) < 2:
        pytest.skip("need 2 devices")
    return StagesArray(
        {
            0: jax.device_put(jnp.ones((3,)), devs[0]),
            1: jax.device_put(2 * jnp.ones((3,)), devs[1]),
        }
    )


def test_construction_and_properties():
    """Constructed StagesArray exposes shape, dtype, and shard indices."""
    a = _two_device_shards()
    assert a.shape == (3,)
    assert a.dtype == jnp.float32
    assert a.mpmd_idxs == frozenset({0, 1})


def test_rejects_empty_shards():
    """Empty shard dict is rejected with a clear error."""
    with pytest.raises(ValueError, match="at least one shard"):
        StagesArray({})


def test_rejects_mismatched_shapes():
    """Shards with differing shapes are rejected."""
    with pytest.raises(ValueError, match="identical shapes"):
        StagesArray({0: jnp.ones((3,)), 1: jnp.ones((4,))})


def test_rejects_mismatched_dtypes():
    """Shards with differing dtypes are rejected."""
    with pytest.raises(ValueError, match="identical dtypes"):
        StagesArray({0: jnp.ones((3,), dtype=jnp.float32), 1: jnp.ones((3,), dtype=jnp.int32)})


def test_getitem_and_contains():
    """``__getitem__`` / ``__contains__`` honour the shard-index set."""
    a = _two_device_shards()
    assert 0 in a
    assert 1 in a
    assert 2 not in a
    with pytest.raises(KeyError):
        a[2]
    assert np.array_equal(np.asarray(a[0]), np.ones((3,)))


def test_with_shard_returns_copy():
    """``with_shard`` returns a new StagesArray, leaving the original intact."""
    a = _two_device_shards()
    b = a.with_shard(0, jnp.zeros((3,)))
    assert np.array_equal(np.asarray(a[0]), np.ones((3,)))
    assert np.array_equal(np.asarray(b[0]), np.zeros((3,)))
    assert np.array_equal(np.asarray(b[1]), 2 * np.ones((3,)))


def test_with_shard_rejects_shape_change():
    """``with_shard`` rejects a replacement with a different shape/dtype."""
    a = _two_device_shards()
    with pytest.raises(ValueError, match="shape/dtype"):
        a.with_shard(0, jnp.zeros((4,)))


def test_reduce_sum_cross_device():
    """``reduce_sum`` sums every shard across devices."""
    a = _two_device_shards()
    total = a.reduce_sum()
    assert np.array_equal(np.asarray(total), 3 * np.ones((3,)))


def test_partially_addressable_single_process():
    """In single-process runs every shard is local, so not partially addressable."""
    a = _two_device_shards()
    assert a.partially_addressable is False


def test_pytree_roundtrip():
    """StagesArray round-trips through ``jax.tree_util`` flatten/unflatten."""
    a = _two_device_shards()
    leaves, treedef = jax.tree_util.tree_flatten(a)
    assert len(leaves) == 2
    b = jax.tree_util.tree_unflatten(treedef, leaves)
    assert b.mpmd_idxs == a.mpmd_idxs
    assert b.shape == a.shape


def test_tree_map_preserves_structure():
    """``jax.tree.map`` transforms each shard and preserves the pytree structure."""
    a = _two_device_shards()
    b = jax.tree.map(lambda x: x + 10, a)
    assert isinstance(b, StagesArray)
    assert b.mpmd_idxs == a.mpmd_idxs
    assert np.array_equal(np.asarray(b[0]), 11 * np.ones((3,)))
    assert np.array_equal(np.asarray(b[1]), 12 * np.ones((3,)))


def test_to_local_array_single_shard():
    """``to_local_array`` returns the lone shard when there is exactly one."""
    a = StagesArray({0: jnp.arange(3, dtype=jnp.float32)})
    arr = a.to_local_array()
    assert np.array_equal(np.asarray(arr), np.arange(3))


def test_to_local_array_requires_single_shard():
    """``to_local_array`` errors when the array has more than one shard."""
    a = _two_device_shards()
    with pytest.raises(ValueError, match="exactly one shard"):
        a.to_local_array()


def test_replicated_value_requires_flag():
    """``replicated_value`` requires the ``replicated=True`` flag."""
    a = _two_device_shards()
    with pytest.raises(ValueError, match="replicated=True"):
        a.replicated_value()
    b = StagesArray(a.shards, replicated=True)
    out = b.replicated_value()
    assert out.shape == (3,)


def test_no_implicit_ndarray_conversion():
    """``np.asarray`` on a multi-shard StagesArray raises TypeError."""
    a = _two_device_shards()
    with pytest.raises(TypeError, match="cannot be converted"):
        np.asarray(a)


def test_process_index_reports_jax_process_index():
    """``StagesArray.process_index`` returns :func:`jax.process_index`."""
    a = StagesArray({0: jnp.asarray(1.0)})
    assert a.process_index == int(jax.process_index())


def test_local_shards_is_full_in_single_process():
    """Single-process runs always have every shard local."""
    a = _two_device_shards()
    assert set(a.local_shards.keys()) == {0, 1}
    assert a.remote_mpmd_idxs == frozenset()


def test_gather_to_process_single_process_preserves_values():
    """Single-process gather_to_process yields shards with identical values."""
    a = _two_device_shards()
    b = a.gather_to_process(0)
    assert set(b.shards.keys()) == set(a.shards.keys())
    for k in a.shards:
        assert jnp.allclose(a.shards[k], b.shards[k])


def test_gather_to_process_rejects_unknown_process_index():
    """An out-of-range target_process raises a clear ValueError."""
    a = _two_device_shards()
    with pytest.raises(ValueError, match="no devices found"):
        a.gather_to_process(999)


def test_abstract_stages_array_allocates_zeros_on_requested_stages():
    """``abstract_stages_array`` builds zero shards on every requested stage."""
    from spectrax.runtime.types.array import abstract_stages_array

    a = abstract_stages_array((4, 2), jnp.float32, [0, 2, 3])
    assert a.mpmd_idxs == frozenset({0, 2, 3})
    assert a.shape == (4, 2)
    assert a.dtype == jnp.float32
    for arr in a.shards.values():
        assert jnp.all(arr == 0.0)


def test_abstract_stages_array_dedupes_and_sorts():
    """Duplicate indices are collapsed; iteration is stable."""
    from spectrax.runtime.types.array import abstract_stages_array

    a = abstract_stages_array((1,), jnp.int32, [2, 0, 2, 0, 1])
    assert a.mpmd_idxs == frozenset({0, 1, 2})


def test_abstract_stages_array_replicated_hint():
    """``replicated=True`` propagates through the constructor."""
    from spectrax.runtime.types.array import abstract_stages_array

    a = abstract_stages_array((3,), jnp.float32, [0, 1], replicated=True)
    assert a.replicated is True
