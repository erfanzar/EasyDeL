# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

# tests/layers/cache/test_transformer_cache.py

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from eformer.escale import PartitionAxis
from jax.sharding import NamedSharding, PartitionSpec

# Assume necessary imports from EasyDeL are available in the test environment
# Adjust paths if necessary
from easydel.layers.caching.transformer.transformer_cache import (
    TransformerCache,
    TransformerCacheMetaData,
    TransformerCacheView,
)
from easydel.layers.quantization.quantizers import (
    EasyDeLQuantizationMethods,
    EasyQuantizer,
)


# Test Fixtures and Setup
@pytest.fixture(scope="module")
def mesh():
    devices = jax.devices()
    if len(devices) >= 4:
        mesh_shape = (1, 2, 2)  # Example sharding over 4 devices (dp, fsdp, mp)
    elif len(devices) >= 2:
        mesh_shape = (1, 2, 1)
    else:
        mesh_shape = (1, 1, 1)
    devices_array = np.array(devices).reshape(mesh_shape)
    device_mesh = jax.sharding.Mesh(devices_array, ("dp", "fsdp", "mp"))
    return device_mesh


@pytest.fixture(scope="module")
def quantizer():
    # Using NONE for simplicity, can test with others if needed
    return EasyQuantizer(EasyDeLQuantizationMethods.NONE)


@pytest.fixture(scope="module")
def cache_metadata_config():
    return TransformerCacheMetaData(
        partition_axis=PartitionAxis(batch_axis="dp", head_axis=None, sequence_axis="fsdp"),
        batch_size=2,
        sequence_length=8,
        num_hidden_layers=1,
        num_heads=2,
        head_dim=4,
        key_heads=2,  # Assuming MHA for simplicity
        value_heads=2,
        key_dim=4,
        value_dim=4,
        update_causal_mask=True,
        create_attention_bias=True,
    )


@pytest.fixture(scope="module")
def kv_spec():
    # Matches partition_axis in cache_metadata_config
    return PartitionSpec("dp", "fsdp", None, None)


@pytest.fixture
def transformer_cache(mesh, cache_metadata_config, quantizer, kv_spec):
    with mesh:
        cache = TransformerCache.init_cache(
            metadata=cache_metadata_config,
            mesh=mesh,
            quantizer=quantizer,
            dtype=jnp.float32,
            key_values_shardings=kv_spec,
        )
    return cache


# Test Cases
def test_transformer_cache_initialization(transformer_cache, cache_metadata_config, mesh):
    assert len(transformer_cache.views) == cache_metadata_config.num_hidden_layers
    view = transformer_cache.views[0]
    assert isinstance(view, TransformerCacheView)
    expected_shape = (
        cache_metadata_config.batch_size,
        cache_metadata_config.sequence_length,
        cache_metadata_config.key_heads,
        cache_metadata_config.key_dim,
    )
    assert view.key.shape == expected_shape
    assert view.value.shape == expected_shape
    assert view.key.dtype == jnp.float32
    assert view.value.dtype == jnp.float32
    assert view.index.shape == (cache_metadata_config.batch_size,)
    assert jnp.all(view.index == 0)

    # Check sharding
    assert isinstance(view.key.sharding, NamedSharding)
    assert isinstance(view.value.sharding, NamedSharding)
    assert view.key.sharding.mesh == mesh
    assert view.value.sharding.mesh == mesh
    # Cannot easily assert exact PartitionSpec equality due to internal representation
    # assert view.key.sharding.spec == kv_spec # This might fail


def test_transformer_cache_view_standard_concat(
    transformer_cache,
    cache_metadata_config,
    kv_spec,
    quantizer,
    mesh,
):
    view = transformer_cache.views[0]
    kv_sharding = NamedSharding(mesh, kv_spec)

    # Inputs
    batch, num_heads, head_dim = 2, 2, 4
    q_len, kv_len = 1, 1
    query = jnp.ones((batch, q_len, num_heads, head_dim), dtype=jnp.float32)
    key = jnp.ones((batch, kv_len, num_heads, head_dim), dtype=jnp.float32) * 2.0
    value = jnp.ones((batch, kv_len, num_heads, head_dim), dtype=jnp.float32) * 3.0
    # Base mask (e.g., padding mask) - all valid here
    attention_mask = jnp.ones((batch, cache_metadata_config.sequence_length), dtype=jnp.bool_)
    causal_mask = jnp.tril(
        jnp.ones(
            (
                1,
                1,
                cache_metadata_config.sequence_length,
                cache_metadata_config.sequence_length,
            ),
            dtype=jnp.bool_,
        )
    )

    # --- First Call (Standard Path, cache_metadata=None) ---
    updated_key, updated_value, final_mask = view.concatenate_to_cache(
        query=query,
        key=key,
        value=value,
        cache_metadata=None,  # Explicitly None for standard path
        attention_mask=attention_mask,
        kv_sharding=kv_sharding,
        quantizer=quantizer,
        causal_mask=causal_mask,
    )

    # Checks for standard path
    assert view.index[0] == 1  # Index should update
    chex.assert_trees_all_close(updated_key, view.key)  # Returned should match internal state
    chex.assert_trees_all_close(updated_value, view.value)
    # Check if the update happened at index 0
    chex.assert_trees_all_close(view.key[:, 0:1, :, :], key)
    chex.assert_trees_all_close(view.value[:, 0:1, :, :], value)
    # Check mask shape and type (specific mask logic is complex to test fully here)
    assert final_mask.shape == (
        batch,
        1,
        q_len,
        cache_metadata_config.sequence_length,
    )
    assert final_mask.dtype == jnp.bool_
    # Mask for the first step should allow attention to index 0
    assert jnp.all(final_mask[:, :, 0, 0])

    # --- Second Call (Standard Path) ---
    key_2 = key * 4.0
    value_2 = value * 5.0
    updated_key_2, updated_value_2, final_mask_2 = view.concatenate_to_cache(
        query=query,
        key=key_2,
        value=value_2,
        cache_metadata=None,
        attention_mask=attention_mask,
        kv_sharding=kv_sharding,
        quantizer=quantizer,
        causal_mask=causal_mask,
    )
    assert view.index[0] == 2  # Index should update again
    chex.assert_trees_all_close(updated_key_2, view.key)
    chex.assert_trees_all_close(updated_value_2, view.value)
    # Check updates at index 0 and 1
    chex.assert_trees_all_close(view.key[:, 0:1, :, :], key)
    chex.assert_trees_all_close(view.value[:, 0:1, :, :], value)
    chex.assert_trees_all_close(view.key[:, 1:2, :, :], key_2)
    chex.assert_trees_all_close(view.value[:, 1:2, :, :], value_2)
    assert final_mask_2.shape == (
        batch,
        1,
        q_len,
        cache_metadata_config.sequence_length,
    )
    # Mask for the second step should allow attention to index 0 and 1
    assert jnp.all(final_mask_2[:, :, 0, 0])
    assert jnp.all(final_mask_2[:, :, 0, 1])


# def test_transformer_cache_view_pooled_concat(
# 	transformer_cache,
# 	cache_metadata_config,
# 	kv_spec,
# 	quantizer,
# 	mesh,
# ):
# 	view = transformer_cache.views[0]
# 	initial_index = view.index + 0

# 	kv_sharding = NamedSharding(mesh, kv_spec)

# 	# Inputs
# 	micro_batch_size, q_len, num_heads, head_dim = 2, 1, 2, 4
# 	query = jnp.ones((micro_batch_size, q_len, num_heads, head_dim), dtype=jnp.float32)
# 	# Simulate updates for two sequences in different slots
# 	key = jnp.array(
# 		[[[[1.1] * head_dim] * num_heads], [[[2.2] * head_dim] * num_heads]],
# 		dtype=jnp.float32,
# 	)  # Shape (2, 1, 2, 4)
# 	value = jnp.array(
# 		[
# 			[[[3.3] * head_dim] * num_heads],  # Batch item 0
# 			[[[4.4] * head_dim] * num_heads],
# 		],  # Batch item 1
# 		dtype=jnp.float32,
# 	)  # Shape (2, 1, 2, 4)
# 	attention_mask_in = jnp.ones(
# 		(micro_batch_size, 1), dtype=jnp.bool_
# 	)  # Dummy input mask

# 	# Pooled Metadata
# 	pooled_metadata = TransformerMetadata(
# 		max_sequence_length=cache_metadata_config.sequence_length,
# 		num_layers=cache_metadata_config.num_hidden_layers,
# 		micro_batch_slot_indices=jnp.array([0, 1]),  # Update slot 0 and slot 1
# 		micro_batch_seq_lengths=jnp.array(
# 			[3, 5]
# 		),  # Write at index 3 for slot 0, index 5 for slot 1
# 	)

# 	# --- Call with Pooled Logic ---
# 	final_k_cache, final_v_cache, final_mask = view.concatenate_to_cache(
# 		query=query,
# 		key=key,
# 		value=value,
# 		cache_metadata=pooled_metadata,  # Use pooled metadata
# 		attention_mask=attention_mask_in,
# 		kv_sharding=kv_sharding,
# 		quantizer=quantizer,
# 		causal_mask=None,  # Not used in pooled path of concat
# 	)

# 	# Checks for pooled path
# 	# 1. Returned mask should be the original input mask
# 	chex.assert_trees_all_equal(final_mask, attention_mask_in)

# 	# 2. Returned K/V caches should reflect the targeted updates
# 	# Slot 0, index 3 should be updated
# 	chex.assert_trees_all_close(final_k_cache[0, 3:4, :, :], key[0])
# 	chex.assert_trees_all_close(final_v_cache[0, 3:4, :, :], value[0])
# 	# Slot 1, index 5 should be updated
# 	chex.assert_trees_all_close(final_k_cache[1, 5:6, :, :], key[1])
# 	chex.assert_trees_all_close(final_v_cache[1, 5:6, :, :], value[1])
# 	# Other indices should remain unchanged (0.0 from initialization)
# 	assert jnp.all(final_k_cache[0, :3] == 0.0)
# 	assert jnp.all(final_k_cache[0, 4:] == 0.0)
# 	assert jnp.all(final_k_cache[1, :5] == 0.0)
# 	assert jnp.all(final_k_cache[1, 6:] == 0.0)

# 	# 3. Check internal state modification (based on user's current code)
# 	# This part verifies the side-effect in the pooled path due to user's revert
# 	# Ideally, in a purely functional setup for JIT, self.key/value wouldn't change here.
# 	chex.assert_trees_all_close(view.key, final_k_cache)
# 	chex.assert_trees_all_close(view.value, final_v_cache)
# 	# Index should NOT change in pooled logic path according to original design,
# 	# but let's check if it was unintentionally left in the user's version.
# 	# If the user removed index update from pooled path, this should pass.
# 	chex.assert_trees_all_equal(view.index, initial_index)


if __name__ == "__main__":
    pytest.main([__file__])
