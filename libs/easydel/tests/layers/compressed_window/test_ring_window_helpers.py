# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Ring-window write / visibility helpers and selected-entry gather against NumPy references."""

import numpy as np
import pytest
from easydel.layers.compressed_window import (
    gather_selected_entries,
    ring_decode_write,
    ring_prefill_write,
    ring_slot_positions,
    ring_visibility_bias,
)
from jax import numpy as jnp

MASK_MIN = float(np.finfo(np.float32).min)
DIM = 3
BATCH = 2


@pytest.mark.parametrize("seq", [2, 4, 5, 9])
def test_prefill_write_matches_numpy(seq):
    window = 4
    rng = np.random.default_rng(seq)
    ring = rng.normal(size=(BATCH, window, DIM)).astype(np.float32)
    kv = rng.normal(size=(BATCH, seq, DIM)).astype(np.float32)
    expected = ring.copy()
    for position in range(max(0, seq - window), seq):
        expected[:, position % window] = kv[:, position]
    got = ring_prefill_write(jnp.asarray(ring), jnp.asarray(kv), window)
    assert got.shape == ring.shape and got.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(got), expected)


def test_prefill_write_casts_to_ring_dtype():
    ring = jnp.zeros((1, 4, DIM), jnp.bfloat16)
    kv = jnp.ones((1, 2, DIM), jnp.float32)
    assert ring_prefill_write(ring, kv, 4).dtype == jnp.bfloat16


def test_decode_write_matches_numpy_with_per_row_positions():
    window = 4
    rng = np.random.default_rng(3)
    ring = rng.normal(size=(BATCH, window, DIM)).astype(np.float32)
    kv_t = rng.normal(size=(BATCH, DIM)).astype(np.float32)
    position = np.array([6, 3], np.int32)
    expected = ring.copy()
    for row in range(BATCH):
        expected[row, position[row] % window] = kv_t[row]
    got = ring_decode_write(jnp.asarray(ring), jnp.asarray(kv_t), jnp.asarray(position), window)
    np.testing.assert_array_equal(np.asarray(got), expected)


@pytest.mark.parametrize("position", [0, 1, 3, 4, 9])
def test_slot_positions_and_visibility_bias_match_brute_force(position):
    window = 4
    got = np.asarray(ring_slot_positions(jnp.asarray([position, position + 2], jnp.int32), window))
    assert got.shape == (2, window)
    for row, pos in enumerate((position, position + 2)):
        for slot in range(window):
            held = [p for p in range(pos + 1) if p % window == slot]
            if held:
                assert got[row, slot] == max(held)
            else:
                assert got[row, slot] < 0
    bias = ring_visibility_bias(jnp.asarray(got))
    assert bias.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(bias), np.where(got >= 0, 0.0, MASK_MIN).astype(np.float32))


def test_gather_selected_entries_matches_numpy():
    rng = np.random.default_rng(5)
    entries = rng.normal(size=(BATCH, 6, DIM)).astype(np.float32)
    topk = np.array([[0, 3, -1, 5], [2, -1, -1, 1]], np.int32)
    gathered, bias = gather_selected_entries(jnp.asarray(entries), jnp.asarray(topk))
    assert gathered.shape == (BATCH, 4, DIM) and bias.shape == (BATCH, 4)
    assert bias.dtype == jnp.float32
    valid = topk >= 0
    for row in range(BATCH):
        for k in range(4):
            if valid[row, k]:
                np.testing.assert_array_equal(np.asarray(gathered[row, k]), entries[row, topk[row, k]])
    np.testing.assert_array_equal(np.asarray(bias), np.where(valid, 0.0, MASK_MIN).astype(np.float32))
