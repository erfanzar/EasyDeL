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

"""State-layout tests for the compressed sliding-window cache (DeepSeek-V4)."""

import jax.numpy as jnp
import pytest
from easydel.caching import CompressedWindowCache, CompressedWindowCacheConfig
from spectrax import PartitionAxis

LAYER_TYPES = (
    "sliding_attention",
    "compressed_sparse_attention",
    "heavily_compressed_attention",
)


def _make_config(**overrides):
    kwargs = dict(
        num_hidden_layers=3,
        partition_axis=PartitionAxis(),
        batch_size=2,
        max_length=64,
        sliding_window=8,
        head_dim=32,
        index_head_dim=16,
        csa_rate=2,
        hca_rate=4,
        layer_types=LAYER_TYPES,
    )
    kwargs.update(overrides)
    return CompressedWindowCacheConfig.create(**kwargs)


class TestCompressedWindowCacheConfig:
    def test_validation_rejects_bad_layer_types(self):
        with pytest.raises(ValueError, match="layer_types entries"):
            _make_config(layer_types=("sliding_attention", "full_attention", "sliding_attention"))

    def test_validation_rejects_layer_count_mismatch(self):
        with pytest.raises(ValueError, match="num_hidden_layers"):
            _make_config(layer_types=("sliding_attention",))

    def test_rate_lookup(self):
        config = _make_config()
        assert config.rate_for("compressed_sparse_attention") == 2
        assert config.rate_for("heavily_compressed_attention") == 4
        with pytest.raises(KeyError):
            config.rate_for("sliding_attention")


class TestCompressedWindowCacheLayout:
    """Per-layer-type field presence, shapes, and dtypes."""

    def test_state_layout_per_layer_type(self):
        config = _make_config()
        cache = CompressedWindowCache.init_cache(config=config, dtype=jnp.float32)
        assert len(cache) == 3

        sliding, csa, hca = cache.views

        for view in (sliding, csa, hca):
            assert view.window_kv.shape == (2, 8, 32)
            assert view.window_kv.dtype == jnp.float32
            assert view.cache_position.dtype == jnp.int32
            # one independent stream-position counter per batch row (slot)
            assert view.cache_position.shape == (2,)
            assert bool(jnp.all(view.cache_position == 0))

        # sliding layers carry no compressor state
        assert sliding.compressor_buffer_kv is None
        assert sliding.compressor_entries is None
        assert sliding.indexer_entries is None
        assert sliding.rate == 0
        assert sliding.num_entry_slots == 0

        # HCA: single-series buffers at head_dim, entries at max_length // rate
        assert hca.compressor_buffer_kv.shape == (2, 4, 32)
        assert hca.compressor_buffer_gate.shape == (2, 4, 32)
        assert hca.compressor_entries.shape == (2, 64 // 4, 32)
        assert hca.compressor_overlap_kv is None
        assert hca.indexer_entries is None
        assert hca.num_entry_slots == 16

        # CSA: two-series buffers at 2 * head_dim, overlap at head_dim,
        # plus the full indexer stream at index_head_dim.
        assert csa.compressor_buffer_kv.shape == (2, 2, 64)
        assert csa.compressor_entries.shape == (2, 64 // 2, 32)
        assert csa.compressor_overlap_kv.shape == (2, 2, 32)
        assert csa.compressor_overlap_gate.shape == (2, 2, 32)
        assert csa.compressor_overlap_gate.dtype == jnp.float32
        # overlap gates start at -inf so the Ca half has zero softmax weight
        # before the first window closes.
        assert bool(jnp.all(jnp.isneginf(csa.compressor_overlap_gate)))
        assert csa.indexer_buffer_kv.shape == (2, 2, 32)
        assert csa.indexer_entries.shape == (2, 64 // 2, 16)
        assert csa.indexer_overlap_kv.shape == (2, 2, 16)
        assert bool(jnp.all(jnp.isneginf(csa.indexer_overlap_gate)))
        assert csa.num_entry_slots == 32

    def test_reset_slots_clears_only_selected_rows(self):
        """reset_slots reproduces the init pattern (zeros / -inf gates / pos 0) per row."""
        config = _make_config()
        cache = CompressedWindowCache.init_cache(config=config, dtype=jnp.float32)

        # Populate every array field of every view with a nonzero pattern and
        # advance both rows to distinct positions.
        def _fill(view):
            updates = {"cache_position": jnp.array([5, 9], dtype=jnp.int32)}
            for name in (
                "window_kv",
                "compressor_buffer_kv",
                "compressor_buffer_gate",
                "compressor_entries",
                "compressor_overlap_kv",
                "compressor_overlap_gate",
                "indexer_buffer_kv",
                "indexer_buffer_gate",
                "indexer_entries",
                "indexer_overlap_kv",
                "indexer_overlap_gate",
            ):
                value = getattr(view, name)
                if value is not None:
                    updates[name] = jnp.full(value.shape, 3.0, dtype=value.dtype)
            return view.replace(**updates)

        cache = CompressedWindowCache(views=[_fill(view) for view in cache.views])
        reset = cache.reset_slots(jnp.array([1], dtype=jnp.int32))

        for view in reset.views:
            # Row 0 untouched, row 1 back to init state.
            assert int(view.cache_position[0]) == 5
            assert int(view.cache_position[1]) == 0
            assert bool(jnp.all(view.window_kv[0] == 3.0))
            assert bool(jnp.all(view.window_kv[1] == 0.0))
            for name in ("compressor_buffer_kv", "compressor_entries", "indexer_entries"):
                value = getattr(view, name)
                if value is not None:
                    assert bool(jnp.all(value[0] == 3.0))
                    assert bool(jnp.all(value[1] == 0.0))
            for name in ("compressor_overlap_gate", "indexer_overlap_gate"):
                value = getattr(view, name)
                if value is not None:
                    assert bool(jnp.all(value[0] == 3.0))
                    assert bool(jnp.all(jnp.isneginf(value[1])))

    def test_init_empty_placeholders(self):
        cache = CompressedWindowCache.init_empty(num_hidden_layers=5)
        assert len(cache) == 5
        assert all(view is None for view in cache.views)

    def test_views_are_pytree_carry_safe(self):
        """Round-tripping through flatten/unflatten preserves structure (while_loop carry)."""
        import jax

        config = _make_config()
        cache = CompressedWindowCache.init_cache(config=config, dtype=jnp.float32)
        leaves, treedef = jax.tree.flatten(cache)
        rebuilt = jax.tree.unflatten(treedef, leaves)
        assert isinstance(rebuilt, CompressedWindowCache)
        assert rebuilt.views[1].compressor_entries.shape == cache.views[1].compressor_entries.shape
        assert rebuilt.views[2].layer_type == "heavily_compressed_attention"


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
