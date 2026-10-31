# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pinned contracts for the indexer selection primitives and config guards.

``indices_to_bool_mask`` is the single place ``-1``-padded top-k selections
become dense attention masks, so its padding / duplicate / range handling is
load-bearing for every indexer family. ``IndexerConfig`` validation guards
(pool kind without tail-pool selection) keep structurally broken indexers
from instantiating.
"""

import os

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from easydel.layers.indexer import IndexerConfig, IndexerKind, indices_to_bool_mask

KV = 6


def test_pool_config_requires_select_tail():
    """``kind='pool'`` without ``select_tail=True`` is rejected at config time.

    Without the always-selected tail pool, queries seeing fewer than
    ``kpool_size`` valid tokens (the first rows of every prefill, and every
    row when kv < kpool) would select nothing and produce all-masked
    attention rows. Everything else about the config must be valid so this
    is the only fired guard.
    """
    with pytest.raises(ValueError, match="select_tail"):
        IndexerConfig(
            kind=IndexerKind.POOL,
            index_n_heads=2,
            index_head_dim=8,
            index_topk=8,
            hidden_size=16,
            q_input_dim=16,
            kpool_size=4,
            packed_state="key_gate_valid",
            select_tail=False,
        )
    # the same config with the tail pool enabled is accepted
    config = IndexerConfig(
        kind=IndexerKind.POOL,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=8,
        hidden_size=16,
        q_input_dim=16,
        kpool_size=4,
        packed_state="key_gate_valid",
        select_tail=True,
    )
    assert config.select_tail is True


def test_minus_one_slots_select_nothing():
    """``-1`` padding lanes must not map to position 0.

    Probe: indices ``[[3, -1, -1]]`` over kv=6 select position 3 only —
    position 0 stays unselected even though a naive clip would map ``-1``
    there. This is the padding lane of every indexer's top-k output.
    Shape note: the reduction collapses the *selection* axis (axis -2), so a
    ``(batch, k)`` input yields ``(batch, kv)`` and ``(batch, q, k)`` yields
    ``(batch, q, kv)``.
    """
    indices = jnp.asarray([[3, -1, -1]], dtype=jnp.int32)
    mask = indices_to_bool_mask(indices, kv_length=KV)
    assert mask.shape == (1, KV)
    assert mask.dtype == jnp.bool_
    assert bool(mask[0, 0]) is False
    assert bool(mask[0, 3]) is True
    assert int(mask[0].sum()) == 1


def test_all_invalid_row_selects_nothing():
    """A row of all ``-1`` lanes yields an all-False mask, not position 0."""
    indices = jnp.full((1, 3), -1, dtype=jnp.int32)
    mask = indices_to_bool_mask(indices, kv_length=KV)
    assert int(mask.sum()) == 0


def test_duplicate_selections_collapse():
    """Repeated indices collapse to one selected position (``any`` reduce)."""
    indices = jnp.asarray([[2, 2, 2, -1]], dtype=jnp.int32)
    mask = indices_to_bool_mask(indices, kv_length=KV)
    assert int(mask[0].sum()) == 1
    assert bool(mask[0, 2]) is True


def test_out_of_range_indices_clip():
    """Out-of-range indices clip into the valid kv range (current contract).

    ``6`` and ``100`` over kv=6 both land on the last position ``5``. The
    clipping is intentional: production callers guarantee in-range indices
    and rely on clip for jit-safety of the ``one_hot`` scatter.
    """
    indices = jnp.asarray([[6, 100, -1]], dtype=jnp.int32)
    mask = indices_to_bool_mask(indices, kv_length=KV)
    assert bool(mask[0, 5]) is True
    assert int(mask[0].sum()) == 1


def test_batched_indices_preserve_row_structure():
    """Batched (b, q, k) inputs scatter per row and keep ``-1`` lanes silent."""
    generator = np.random.default_rng(3)
    raw = generator.integers(0, KV, size=(2, 3, 4))
    raw[:, :, -1] = -1  # every query has one padding lane
    indices = jnp.asarray(raw, dtype=jnp.int32)
    mask = indices_to_bool_mask(indices, kv_length=KV)

    assert mask.shape == (2, 3, KV)
    np_mask = np.asarray(mask)
    for b in range(2):
        for q in range(3):
            expected = set(int(x) for x in raw[b, q] if x >= 0)
            got = set(int(i) for i in np_mask[b, q].nonzero()[0])
            assert got == expected


def test_mask_is_pure_function_of_indices():
    """No hidden state: the same indices always produce the same mask."""
    indices = jnp.asarray([[1, -1, 4]], dtype=jnp.int32)
    first = indices_to_bool_mask(indices, kv_length=KV)
    second = indices_to_bool_mask(indices, kv_length=KV)
    assert bool((first == second).all())
    assert isinstance(first, jax.Array)
