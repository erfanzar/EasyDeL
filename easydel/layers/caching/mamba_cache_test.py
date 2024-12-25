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

import chex as cx
import pytest
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from .mamba_cache import (
	MambaCache,
	MambaCacheMetaData,
	MambaCacheView,
)


@pytest.fixture
def cache_metadata():
	return MambaCacheMetaData.create(
		batch_size=2,
		intermediate_size=64,
		ssm_state_size=16,
		conv_kernel_size=4,
	)


@pytest.fixture
def basic_cache_view(cache_metadata):
	return MambaCacheView.init(
		metadata=cache_metadata,
		partition_specs=None,
		dtype=jnp.float32,
		layer_index=0,
	)


@pytest.fixture
def basic_cache(cache_metadata):
	return MambaCache.init_layers_cache(
		num_hidden_layers=2,
		metadata=cache_metadata,
		dtype=jnp.float32,
		partition_specs=None,
	)


class TestMambaCacheMetaData:
	def test_create_valid(self):
		metadata = MambaCacheMetaData.create(
			batch_size=2,
			intermediate_size=64,
			ssm_state_size=16,
			conv_kernel_size=4,
		)
		assert metadata.batch_size == 2
		assert metadata.intermediate_size == 64
		assert metadata.ssm_state_size == 16
		assert metadata.conv_kernel_size == 4

	@pytest.mark.parametrize(
		"field,value",
		[
			("batch_size", 0),
			("batch_size", -1),
			("intermediate_size", 0),
			("ssm_state_size", -1),
			("conv_kernel_size", 0),
		],
	)
	def test_create_invalid(self, field, value):
		kwargs = {
			"batch_size": 2,
			"intermediate_size": 64,
			"ssm_state_size": 16,
			"conv_kernel_size": 4,
		}
		kwargs[field] = value
		with pytest.raises(ValueError):
			MambaCacheMetaData.create(**kwargs)


class TestMambaCache:
	def test_init_layers_cache(self, basic_cache):
		assert len(basic_cache.views) == 2
		assert all(isinstance(view, MambaCacheView) for view in basic_cache.views)

		for view in basic_cache.views:
			cx.assert_shape(view.conv_states, (2, 64, 4))
			cx.assert_shape(view.ssm_states, (2, 64, 16))

	def test_update_conv_state(self, basic_cache):
		new_conv_state = jnp.ones((2, 64))
		cache_position = jnp.array(1)

		updated_cache = basic_cache.update_conv_state(
			layer_idx=0,
			new_conv_state=new_conv_state,
			cache_position=cache_position,
		)

		# Check original cache is unchanged
		assert jnp.all(basic_cache.views[0].conv_states == 0)

		# Check updated cache
		assert jnp.all(updated_cache.views[0].conv_states[:, :, 1] == 1)
		assert jnp.all(updated_cache.views[1].conv_states == 0)  # Other layer unchanged

	def test_update_ssm_state(self, basic_cache):
		new_ssm_state = jnp.ones((2, 64, 16))

		updated_cache = basic_cache.update_ssm_state(
			layer_idx=0,
			new_ssm_state=new_ssm_state,
		)

		assert jnp.all(basic_cache.views[0].ssm_states == 0)  # Original unchanged
		assert jnp.all(updated_cache.views[0].ssm_states == 1)
		assert jnp.all(updated_cache.views[1].ssm_states == 0)  # Other layer unchanged

	def test_reset(self, basic_cache):
		# First update some states
		new_conv_state = jnp.ones((2, 64))
		new_ssm_state = jnp.ones((2, 64, 16))

		updated_cache = basic_cache.update_conv_state(
			layer_idx=0,
			new_conv_state=new_conv_state,
			cache_position=jnp.array(0),
		)
		updated_cache = updated_cache.update_ssm_state(
			layer_idx=0,
			new_ssm_state=new_ssm_state,
		)

		# Then reset
		reset_cache = updated_cache.reset()

		for view in reset_cache.views:
			assert jnp.all(view.conv_states == 0)
			assert jnp.all(view.ssm_states == 0)

	def test_invalid_layer_idx(self, basic_cache):
		with pytest.raises(IndexError):
			basic_cache.update_conv_state(
				layer_idx=2,  # Invalid layer index
				new_conv_state=jnp.ones((2, 64)),
				cache_position=jnp.array(0),
			)

	def test_empty_cache(self):
		empty_cache = MambaCache.init_empty(num_hidden_layers=2)
		assert len(empty_cache.views) == 2
		assert all(view is None for view in empty_cache.views)

	def test_update_with_none_view(self):
		empty_cache = MambaCache.init_empty(num_hidden_layers=2)
		with pytest.raises(ValueError, match="Cache view for layer 0 is None"):
			empty_cache.update_conv_state(
				layer_idx=0,
				new_conv_state=jnp.ones((2, 64)),
				cache_position=jnp.array(0),
			)


class TestCacheShapes:
	@pytest.mark.parametrize(
		"batch_size,intermediate_size,ssm_state_size,conv_kernel_size",
		[
			(1, 32, 8, 2),
			(4, 128, 32, 8),
			(8, 256, 64, 16),
		],
	)
	def test_different_shapes(
		self, batch_size, intermediate_size, ssm_state_size, conv_kernel_size
	):
		metadata = MambaCacheMetaData.create(
			batch_size=batch_size,
			intermediate_size=intermediate_size,
			ssm_state_size=ssm_state_size,
			conv_kernel_size=conv_kernel_size,
		)

		cache = MambaCache.init_layers_cache(
			num_hidden_layers=2,
			metadata=metadata,
			dtype=jnp.float32,
			partition_specs=None,
		)

		for view in cache.views:
			cx.assert_shape(
				view.conv_states, (batch_size, intermediate_size, conv_kernel_size)
			)
			cx.assert_shape(view.ssm_states, (batch_size, intermediate_size, ssm_state_size))


class TestCacheDTypes:
	@pytest.mark.parametrize(
		"dtype",
		[
			jnp.float32,
			jnp.float16,
			jnp.bfloat16,
		],
	)
	def test_different_dtypes(self, cache_metadata, dtype):
		cache = MambaCache.init_layers_cache(
			num_hidden_layers=2,
			metadata=cache_metadata,
			dtype=dtype,
			partition_specs=None,
		)

		for view in cache.views:
			cx.assert_type(view.conv_states, dtype)
			cx.assert_type(view.ssm_states, dtype)


class TestCachePartitioning:
	def test_custom_partition_specs(self, cache_metadata):
		custom_partition_spec = PartitionSpec("batch", "model", None)
		cache = MambaCache.init_layers_cache(
			num_hidden_layers=2,
			metadata=cache_metadata,
			dtype=jnp.float32,
			partition_specs=custom_partition_spec,
		)

		# Note: Actual sharding tests would depend on your specific setup
		# Here we just verify the cache is created successfully with custom specs
		assert len(cache.views) == 2
		assert all(isinstance(view, MambaCacheView) for view in cache.views)


class TestCacheOperations:
	def test_cache_position_clipping(self, basic_cache):
		new_conv_state = jnp.ones((2, 64))

		# Test with position beyond kernel size
		updated_cache = basic_cache.update_conv_state(
			layer_idx=0,
			new_conv_state=new_conv_state,
			cache_position=jnp.array(10),  # Should be clipped to 3
		)

		assert jnp.all(updated_cache.views[0].conv_states[:, :, 3] == 1)
		assert jnp.all(updated_cache.views[0].conv_states[:, :, :3] == 0)

	def test_negative_cache_position(self, basic_cache):
		new_conv_state = jnp.ones((2, 64))

		# Test with negative position
		updated_cache = basic_cache.update_conv_state(
			layer_idx=0,
			new_conv_state=new_conv_state,
			cache_position=jnp.array(-1),  # Should be clipped to 0
		)

		assert jnp.all(updated_cache.views[0].conv_states[:, :, 0] == 1)
		assert jnp.all(updated_cache.views[0].conv_states[:, :, 1:] == 0)


if __name__ == "__main__":
	pytest.main([__file__])
