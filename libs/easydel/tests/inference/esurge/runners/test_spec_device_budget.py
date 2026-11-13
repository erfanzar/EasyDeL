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

"""The on-device verification budget split equals a sequential greedy reference over cumulative confidences."""

import numpy as np
import pytest
from easydel.inference.esurge.runners.spec.device_budget import split_verify_budget
from jax import numpy as jnp


def _reference(confidence: np.ndarray, active: np.ndarray, extra_rows: int) -> np.ndarray:
    """Greedy selection: sort every active (row, position) by cumulative confidence, take the first ``extra_rows``."""
    rows, k = confidence.shape
    items = []
    for r in range(rows):
        if not active[r]:
            continue
        cum = np.float32(1.0)
        for j in range(k):
            c = np.float32(min(max(float(confidence[r, j]), 0.0), 1.0))
            cum = np.float32(cum * c) if j else c
            items.append((-float(cum), r, j))
    items.sort()
    out = np.zeros(rows, dtype=np.int32)
    for _, r, _ in items[: max(extra_rows, 0)]:
        out[r] += 1
    return out


def _run(confidence: np.ndarray, active: np.ndarray, extra_rows: int) -> np.ndarray:
    return np.asarray(
        split_verify_budget(jnp.asarray(confidence, jnp.float32), jnp.asarray(active), jnp.int32(extra_rows))
    )


@pytest.mark.parametrize("rows,k", [(1, 1), (4, 3), (16, 5), (64, 7), (128, 4)])
def test_matches_reference_on_random_inputs(rows, k):
    rng = np.random.default_rng(rows * 31 + k)
    for _ in range(6):
        confidence = rng.uniform(0.05, 1.0, size=(rows, k)).astype(np.float32)
        active = rng.random(rows) < 0.75
        extra = int(rng.integers(0, rows * k + 3))
        got = _run(confidence, active, extra)
        np.testing.assert_array_equal(got, _reference(confidence, active, extra))
        assert got.sum() == min(extra, k * int(active.sum()))


def test_coarse_values_with_many_ties():
    rng = np.random.default_rng(7)
    confidence = rng.choice(np.array([0.0, 0.5, 1.0], np.float32), size=(32, 4))
    active = rng.random(32) < 0.8
    for extra in range(0, 32 * 4 + 2, 5):
        np.testing.assert_array_equal(_run(confidence, active, extra), _reference(confidence, active, extra))


def test_zero_budget_verifies_nothing():
    confidence = np.full((8, 3), 0.9, np.float32)
    np.testing.assert_array_equal(_run(confidence, np.ones(8, bool), 0), np.zeros(8, np.int32))
    np.testing.assert_array_equal(_run(confidence, np.ones(8, bool), -4), np.zeros(8, np.int32))


def test_budget_past_all_positions_verifies_every_active_draft():
    confidence = np.random.default_rng(1).uniform(0.0, 1.0, size=(6, 4)).astype(np.float32)
    active = np.array([1, 0, 1, 1, 0, 1], bool)
    for extra in (16, 24, 1000):
        np.testing.assert_array_equal(_run(confidence, active, extra), np.where(active, 4, 0).astype(np.int32))


def test_inactive_rows_get_nothing_even_when_most_confident():
    confidence = np.array([[1.0, 1.0], [0.2, 0.1], [0.3, 0.3]], np.float32)
    active = np.array([False, True, True])
    np.testing.assert_array_equal(_run(confidence, active, 1), [0, 0, 1])
    np.testing.assert_array_equal(_run(confidence, active, 3), [0, 1, 2])


def test_ties_break_by_row_then_position():
    confidence = np.ones((4, 3), np.float32)
    active = np.ones(4, bool)
    np.testing.assert_array_equal(_run(confidence, active, 2), [2, 0, 0, 0])
    np.testing.assert_array_equal(_run(confidence, active, 4), [3, 1, 0, 0])
    np.testing.assert_array_equal(_run(confidence, active, 7), [3, 3, 1, 0])


def test_cumulative_product_orders_positions():
    # Row 0's second draft (0.9 * 0.5 = 0.45) loses to row 1's first (0.6) though 0.5 alone is not the comparison.
    confidence = np.array([[0.9, 0.5], [0.6, 0.99]], np.float32)
    active = np.ones(2, bool)
    np.testing.assert_array_equal(_run(confidence, active, 2), [1, 1])
    np.testing.assert_array_equal(_run(confidence, active, 3), [1, 2])


def test_chosen_positions_are_prefixes():
    rng = np.random.default_rng(3)
    rows, k = 40, 6
    confidence = rng.uniform(0.0, 1.0, size=(rows, k)).astype(np.float32)
    confidence[rng.random((rows, k)) < 0.1] = 1.0
    active = rng.random(rows) < 0.9
    cum = np.cumprod(np.clip(confidence, 0.0, 1.0), axis=1, dtype=np.float32)
    for extra in range(0, rows * k + 1, 7):
        counts = _run(confidence, active, extra)
        assert counts.min() >= 0 and counts.max() <= k
        # Every chosen position outranks every unchosen one: a prefix per row, a global threshold across rows.
        chosen = np.arange(k)[None, :] < counts[:, None]
        if chosen.any() and (~chosen & active[:, None]).any():
            assert cum[chosen].min() >= cum[~chosen & active[:, None]].max()


def test_traced_budget_reuses_one_compile():
    confidence = jnp.full((8, 3), 0.5, jnp.float32)
    active = jnp.ones(8, bool)
    before = split_verify_budget._cache_size()
    for extra in (0, 3, 11, 24):
        split_verify_budget(confidence, active, jnp.int32(extra)).block_until_ready()
    assert split_verify_budget._cache_size() - before <= 1
