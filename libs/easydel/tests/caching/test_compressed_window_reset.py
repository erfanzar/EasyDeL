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

"""Slot reset for the compressed-window cache (DeepSeek-V4).

Freeing a request slot has to wipe ~12 state tensors per layer. Doing that
eagerly per view meant ~700 separate device dispatches on a 43-layer model;
profiling V4 decode showed those costing ~4.6 ms apiece and dominating the step
while the TPU sat 93% idle. The reset is now one compiled call.

The behaviour it must preserve, exactly:

* selected rows go to zero state, position 0, and ``-inf`` overlap gates
  (``-inf`` matters -- it is zero pre-first-window softmax weight, and a 0.0
  there would silently give a fresh request a real attention weight);
* unselected rows are untouched, or a finished request corrupts a live one;
* out-of-range indices are dropped rather than wrapping onto a live slot.
"""

import jax
import numpy as np
import pytest
from easydel.caching.compressed_window.cache import (
    CompressedWindowCache,
    CompressedWindowCacheView,
)
from jax import numpy as jnp

BATCH, HEADS, DIM, WINDOW, ENTRIES = 4, 2, 8, 6, 3


def _view(seed: int) -> CompressedWindowCacheView:
    rng = np.random.default_rng(seed)

    def arr(*shape, dtype=jnp.float32):
        return jnp.asarray(rng.standard_normal(shape).astype(np.float32), dtype=dtype)

    return CompressedWindowCacheView(
        window_kv=arr(BATCH, WINDOW, HEADS, DIM),
        cache_position=jnp.full((BATCH,), 5, dtype=jnp.int32),
        compressor_buffer_kv=arr(BATCH, WINDOW, HEADS, DIM),
        compressor_buffer_gate=arr(BATCH, WINDOW, HEADS),
        compressor_entries=arr(BATCH, ENTRIES, HEADS, DIM),
        compressor_overlap_kv=arr(BATCH, HEADS, DIM),
        compressor_overlap_gate=arr(BATCH, HEADS),
        indexer_buffer_kv=arr(BATCH, WINDOW, HEADS, DIM),
        indexer_buffer_gate=arr(BATCH, WINDOW, HEADS),
        indexer_entries=arr(BATCH, ENTRIES, HEADS, DIM),
        indexer_overlap_kv=arr(BATCH, HEADS, DIM),
        indexer_overlap_gate=arr(BATCH, HEADS),
    )


def _cache(n_layers: int = 3) -> CompressedWindowCache:
    return CompressedWindowCache(views=[_view(i) for i in range(n_layers)])


def _eager_reset(cache: CompressedWindowCache, slots) -> CompressedWindowCache:
    """The per-view path the compiled whole-cache reset replaced."""
    return CompressedWindowCache(
        views=[None if v is None else v.reset_slots(jnp.asarray(slots, jnp.int32)) for v in cache.views]
    )


@pytest.mark.parametrize("slots", [[0], [1, 3], [0, 1, 2, 3], []])
def test_compiled_reset_matches_the_per_view_path(slots):
    """The optimization must be a pure dispatch change, not a behaviour change."""
    cache = _cache()
    got = cache.reset_slots(jnp.asarray(slots, jnp.int32))
    want = _eager_reset(cache, slots)
    for gv, wv in zip(got.views, want.views, strict=True):
        for name, g in vars(gv).items():
            w = getattr(wv, name)
            if isinstance(g, jax.Array):
                np.testing.assert_array_equal(np.asarray(g), np.asarray(w), err_msg=name)


def test_selected_rows_are_emptied_and_others_preserved():
    cache = _cache(n_layers=2)
    before = [np.asarray(v.window_kv) for v in cache.views]
    out = cache.reset_slots(jnp.asarray([1], jnp.int32))
    for view, orig in zip(out.views, before, strict=True):
        kv = np.asarray(view.window_kv)
        assert np.all(kv[1] == 0.0), "reset row must be zeroed"
        np.testing.assert_array_equal(kv[0], orig[0])
        np.testing.assert_array_equal(kv[2], orig[2])
        assert int(np.asarray(view.cache_position)[1]) == 0
        assert int(np.asarray(view.cache_position)[0]) == 5


def test_overlap_gates_reset_to_negative_infinity_not_zero():
    """-inf is zero pre-first-window softmax weight; 0.0 is a real weight."""
    out = _cache(n_layers=1).reset_slots(jnp.asarray([2], jnp.int32))
    view = out.views[0]
    for name in ("compressor_overlap_gate", "indexer_overlap_gate"):
        gate = np.asarray(getattr(view, name))
        assert np.all(np.isneginf(gate[2])), f"{name} row must be -inf"
        assert not np.any(np.isneginf(gate[0])), f"{name} untouched row must be unchanged"


def test_out_of_range_slots_are_dropped_not_wrapped():
    """A stray index must not silently wipe a live request's row."""
    cache = _cache(n_layers=1)
    before = np.asarray(cache.views[0].window_kv)
    out = cache.reset_slots(jnp.asarray([BATCH + 5, -99], jnp.int32))
    np.testing.assert_array_equal(np.asarray(out.views[0].window_kv), before)


def test_uninitialized_cache_short_circuits():
    """A cache whose views are all placeholders has nothing to reset."""
    empty = CompressedWindowCache(views=[None, None])
    assert empty.reset_slots(jnp.asarray([0], jnp.int32)) is empty


def test_reset_is_one_dispatch_not_one_per_layer():
    """The point of the change: cost must not scale with layer count.

    Compiled once per pytree structure, so a 3-layer and a 12-layer cache each
    cost a single call rather than 12x more.
    """
    from easydel.caching.compressed_window.cache import _reset_rows_compiled

    assert hasattr(_reset_rows_compiled, "lower"), "must be a jitted callable"
    big = _cache(n_layers=12)
    out = big.reset_slots(jnp.asarray([0, 2], jnp.int32))
    assert len(out.views) == 12
    assert np.all(np.asarray(out.views[-1].window_kv)[0] == 0.0)


def test_freeing_different_slot_counts_does_not_recompile():
    """Compile once, whatever the free-count -- this regressed once already.

    The first version of the compiled reset took the variable-length index
    array, so freeing 1 slot and freeing 3 slots were different input shapes
    and each triggered a fresh trace. Profiling caught a 1.4 s XLA compile
    firing mid-serve. The mask handed to the jitted helper is now always
    ``[batch]``, so the cache is hit for every free-count.
    """
    from easydel.caching.compressed_window.cache import _reset_rows_compiled

    cache = _cache(n_layers=2)
    _reset_rows_compiled.clear_cache()
    for slots in ([0], [1, 2], [0, 1, 3], [2]):
        cache.reset_slots(jnp.asarray(slots, jnp.int32))
    n = _reset_rows_compiled._cache_size()
    assert n == 1, f"expected a single compiled variant, got {n}"
