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

import jax.numpy as jnp
import pytest
from jax.sharding import PartitionSpec

from .mamba2_cache import (
	Mamba2Cache,
	Mamba2CacheMetaData,
	Mamba2CacheView,
)


def test_mamba2_cache_metadata_creation():
	# Test valid creation
	metadata = Mamba2CacheMetaData.create(
		batch_size=2,
		intermediate_size=64,
		num_heads=4,
		head_dim=16,
		state_size=16,
		conv_kernel_size=4,
		n_groups=1,
	)

	assert metadata.batch_size == 2
	assert metadata.intermediate_size == 64
	assert metadata.num_heads == 4
	assert metadata.head_dim == 16
	assert metadata.state_size == 16
	assert metadata.conv_kernel_size == 4
	assert metadata.n_groups == 1

	# Test invalid parameters
	with pytest.raises(ValueError):
		Mamba2CacheMetaData.create(
			batch_size=-1,
			intermediate_size=64,
			num_heads=4,
			head_dim=16,
			state_size=16,
			conv_kernel_size=4,
			n_groups=1,
		)


def test_mamba2_cache_view_initialization():
	metadata = Mamba2CacheMetaData.create(
		batch_size=2,
		intermediate_size=64,
		num_heads=4,
		head_dim=16,
		state_size=16,
		conv_kernel_size=4,
		n_groups=1,
	)

	view = Mamba2CacheView.init(
		metadata=metadata,
		partition_specs=PartitionSpec("fsdp"),
		dtype=jnp.float32,
		layer_index=0,
	)

	expanded_size = (
		metadata.intermediate_size + 2 * metadata.n_groups * metadata.state_size
	)

	# Check shapes
	assert view.conv_states.shape == (
		metadata.batch_size,
		expanded_size,
		metadata.conv_kernel_size,
	)
	assert view.ssm_states.shape == (
		metadata.batch_size,
		metadata.num_heads,
		metadata.head_dim,
		metadata.state_size,
	)
	assert view.positions.shape == (metadata.batch_size,)
	assert view.layer_index == 0


def test_mamba2_cache_view_updates():
	metadata = Mamba2CacheMetaData.create(
		batch_size=2,
		intermediate_size=64,
		num_heads=4,
		head_dim=16,
		state_size=16,
		conv_kernel_size=4,
		n_groups=1,
	)

	view = Mamba2CacheView.init(
		metadata=metadata,
		partition_specs=PartitionSpec("fsdp"),
		dtype=jnp.float32,
		layer_index=0,
	)

	# Test conv state update
	expanded_size = (
		metadata.intermediate_size + 2 * metadata.n_groups * metadata.state_size
	)
	new_conv_state = jnp.ones((metadata.batch_size, expanded_size))
	cache_position = jnp.array(1)

	updated_view = view.update_conv_state(new_conv_state, cache_position)
	assert jnp.any(updated_view.conv_states[:, :, 1] == 1.0)

	# Test ssm state update
	new_ssm_state = jnp.ones(
		(metadata.batch_size, metadata.num_heads, metadata.head_dim, metadata.state_size)
	)
	updated_view = view.update_ssm_state(new_ssm_state)
	assert jnp.all(updated_view.ssm_states == 1.0)


def test_mamba2_cache_initialization():
	metadata = Mamba2CacheMetaData.create(
		batch_size=2,
		intermediate_size=64,
		num_heads=4,
		head_dim=16,
		state_size=16,
		conv_kernel_size=4,
		n_groups=1,
	)

	num_layers = 3
	cache = Mamba2Cache.init_layers_cache(
		num_hidden_layers=num_layers, metadata=metadata, dtype=jnp.float32
	)

	assert len(cache.views) == num_layers
	assert all(isinstance(view, Mamba2CacheView) for view in cache.views)


def test_mamba2_cache_updates():
	metadata = Mamba2CacheMetaData.create(
		batch_size=2,
		intermediate_size=64,
		num_heads=4,
		head_dim=16,
		state_size=16,
		conv_kernel_size=4,
		n_groups=1,
	)

	cache = Mamba2Cache.init_layers_cache(
		num_hidden_layers=3, metadata=metadata, dtype=jnp.float32
	)

	# Test conv state update
	expanded_size = (
		metadata.intermediate_size + 2 * metadata.n_groups * metadata.state_size
	)
	new_conv_state = jnp.ones((metadata.batch_size, expanded_size))
	cache_position = jnp.array(1)

	updated_cache = cache.update_conv_state(0, new_conv_state, cache_position)
	assert jnp.any(updated_cache.views[0].conv_states[:, :, 1] == 1.0)

	# Test ssm state update
	new_ssm_state = jnp.ones(
		(metadata.batch_size, metadata.num_heads, metadata.head_dim, metadata.state_size)
	)
	updated_cache = cache.update_ssm_state(0, new_ssm_state)
	assert jnp.all(updated_cache.views[0].ssm_states == 1.0)


def test_mamba2_cache_reset():
	metadata = Mamba2CacheMetaData.create(
		batch_size=2,
		intermediate_size=64,
		num_heads=4,
		head_dim=16,
		state_size=16,
		conv_kernel_size=4,
		n_groups=1,
	)

	cache = Mamba2Cache.init_layers_cache(
		num_hidden_layers=3, metadata=metadata, dtype=jnp.float32
	)

	# Fill cache with ones
	expanded_size = (
		metadata.intermediate_size + 2 * metadata.n_groups * metadata.state_size
	)
	new_conv_state = jnp.ones((metadata.batch_size, expanded_size))
	new_ssm_state = jnp.ones(
		(metadata.batch_size, metadata.num_heads, metadata.head_dim, metadata.state_size)
	)

	cache = cache.update_conv_state(0, new_conv_state, jnp.array(1))
	cache = cache.update_ssm_state(0, new_ssm_state)

	# Reset cache
	reset_cache = cache.reset()

	# Verify all states are zero
	for view in reset_cache.views:
		assert jnp.all(view.conv_states == 0.0)
		assert jnp.all(view.ssm_states == 0.0)


def test_empty_cache_initialization():
	num_layers = 3
	empty_cache = Mamba2Cache.init_empty(num_layers)

	assert len(empty_cache.views) == num_layers
	assert all(view is None for view in empty_cache.views)


def test_cache_string_representation():
	metadata = Mamba2CacheMetaData.create(
		batch_size=2,
		intermediate_size=64,
		num_heads=4,
		head_dim=16,
		state_size=16,
		conv_kernel_size=4,
		n_groups=1,
	)

	cache = Mamba2Cache.init_layers_cache(
		num_hidden_layers=2, metadata=metadata, dtype=jnp.float32
	)

	# Test string representation
	str_repr = str(cache)
	assert "Mamba2Cache" in str_repr
	assert "Mamba2CacheView" in str_repr
	assert "conv_states" in str_repr
	assert "ssm_states" in str_repr


def test_invalid_layer_access():
	cache = Mamba2Cache.init_empty(2)

	# Test accessing invalid layer
	with pytest.raises(ValueError):
		cache.update_conv_state(0, jnp.ones((2, 64)), jnp.array(1))

	with pytest.raises(ValueError):
		cache.update_ssm_state(0, jnp.ones((2, 4, 16, 16)))


def test_cache_position_clipping():
	metadata = Mamba2CacheMetaData.create(
		batch_size=2,
		intermediate_size=64,
		num_heads=4,
		head_dim=16,
		state_size=16,
		conv_kernel_size=4,
		n_groups=1,
	)

	view = Mamba2CacheView.init(
		metadata=metadata,
		partition_specs=PartitionSpec("fsdp"),
		dtype=jnp.float32,
		layer_index=0,
	)

	expanded_size = (
		metadata.intermediate_size + 2 * metadata.n_groups * metadata.state_size
	)
	new_conv_state = jnp.ones((metadata.batch_size, expanded_size))

	# Test position clipping at lower bound
	cache_position = jnp.array(-1)
	updated_view = view.update_conv_state(new_conv_state, cache_position)
	assert jnp.any(updated_view.conv_states[:, :, 0] == 1.0)

	# Test position clipping at upper bound
	cache_position = jnp.array(10)  # larger than conv_kernel_size
	updated_view = view.update_conv_state(new_conv_state, cache_position)
	assert jnp.any(updated_view.conv_states[:, :, metadata.conv_kernel_size - 1] == 1.0)


def test_dtype_consistency():
	metadata = Mamba2CacheMetaData.create(
		batch_size=2,
		intermediate_size=64,
		num_heads=4,
		head_dim=16,
		state_size=16,
		conv_kernel_size=4,
		n_groups=1,
	)

	# Test with different dtypes
	dtypes = [jnp.float32, jnp.float16, jnp.bfloat16]

	for dtype in dtypes:
		cache = Mamba2Cache.init_layers_cache(
			num_hidden_layers=2, metadata=metadata, dtype=dtype
		)

		assert cache.views[0].conv_states.dtype == dtype
		assert cache.views[0].ssm_states.dtype == dtype


def test_batch_dimension_consistency():
	batch_sizes = [1, 2, 4, 8]

	for batch_size in batch_sizes:
		metadata = Mamba2CacheMetaData.create(
			batch_size=batch_size,
			intermediate_size=64,
			num_heads=4,
			head_dim=16,
			state_size=16,
			conv_kernel_size=4,
			n_groups=1,
		)

		cache = Mamba2Cache.init_layers_cache(
			num_hidden_layers=2, metadata=metadata, dtype=jnp.float32
		)

		assert cache.views[0].conv_states.shape[0] == batch_size
		assert cache.views[0].ssm_states.shape[0] == batch_size
		assert cache.views[0].positions.shape[0] == batch_size


def test_partition_specs():
	metadata = Mamba2CacheMetaData.create(
		batch_size=2,
		intermediate_size=64,
		num_heads=4,
		head_dim=16,
		state_size=16,
		conv_kernel_size=4,
		n_groups=1,
	)

	# Test with custom partition specs
	custom_partition_spec = PartitionSpec("batch", "head", None)
	cache = Mamba2Cache.init_layers_cache(
		num_hidden_layers=2,
		metadata=metadata,
		dtype=jnp.float32,
		partition_specs=custom_partition_spec,
	)

	# Test with default partition specs
	default_cache = Mamba2Cache.init_layers_cache(
		num_hidden_layers=2, metadata=metadata, dtype=jnp.float32
	)

	assert isinstance(cache.views[0], Mamba2CacheView)
	assert isinstance(default_cache.views[0], Mamba2CacheView)


if __name__ == "__main__":
	pytest.main([__file__])
