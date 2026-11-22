# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Accelerator tests (array computation): hybrid recurrent dtype, ``update_conv_state``, ``create_empty``."""

import jax.numpy as jnp
import numpy as np
from easydel.caching import HybridCacheConfig, HybridCacheView, RaggedPagesMetadata
from spectrax import PartitionAxis


def _config(**kwargs):
    return HybridCacheConfig.create(
        num_hidden_layers=2,
        partition_axis=PartitionAxis(),
        batch_size=2,
        sequence_length=8,
        num_key_value_heads=2,
        head_dim=4,
        d_inner=8,
        d_conv=3,
        d_state=4,
        layer_types=("linear_attention", "full_attention"),
        **kwargs,
    )


def test_recurrent_state_is_float32_by_default_and_follows_config(monkeypatch):
    # The recurrent dtype comes from the config only; EASYDEL_HYBRID_RECURRENT_FLOAT32 is ignored.
    monkeypatch.setenv("EASYDEL_HYBRID_RECURRENT_FLOAT32", "0")
    default_view = HybridCacheView.init(_config(), 0, dtype=jnp.bfloat16)
    assert default_view.recurrent_state.dtype == jnp.float32
    assert default_view.conv_state.dtype == jnp.bfloat16

    cache_dtype_view = HybridCacheView.init(_config(recurrent_state_dtype="cache"), 0, dtype=jnp.bfloat16)
    assert cache_dtype_view.recurrent_state.dtype == jnp.bfloat16

    bf16_view = HybridCacheView.init(_config(recurrent_state_dtype=jnp.bfloat16), 0, dtype=jnp.float32)
    assert bf16_view.recurrent_state.dtype == jnp.bfloat16


def test_update_conv_state_rolls_buffer_and_advances_positions():
    view = HybridCacheView.init(_config(), 0, dtype=jnp.float32)
    new_state = jnp.ones((2, 8), dtype=jnp.float32)

    updated = view.update_conv_state(new_state)
    np.testing.assert_array_equal(np.asarray(updated.conv_state[:, :, -1]), np.ones((2, 8)))
    np.testing.assert_array_equal(np.asarray(updated.positions), np.zeros((2,)))

    advanced = view.update_conv_state(new_state, cache_position=jnp.array([3, 3], dtype=jnp.int32))
    np.testing.assert_array_equal(np.asarray(advanced.positions), np.array([4, 4]))


def test_create_empty_v2_uses_real_slot_mapping_and_token_position_layout():
    meta = RaggedPagesMetadata.create_empty(
        num_tokens=256, max_num_reqs=4, max_pages=2, page_size=128, version="v2", num_slices_per_kv_cache_update_page=8
    )
    # 2 * max_num_reqs + num_tokens // page_size = 10 slices, aligned up to 8-slice pages -> 16.
    assert meta.slot_mapping.shape == (3, 16)
    assert meta.position_ids.shape == (256,)
    assert meta.num_kv_update_slices.shape == (1,)
    assert meta.num_seqs.shape == (1,)
    assert meta.request_distribution is None

    v3 = RaggedPagesMetadata.create_empty(num_tokens=256, max_num_reqs=4, max_pages=2, version="v3")
    assert v3.slot_mapping is None
    assert v3.request_distribution.shape == (3,)
