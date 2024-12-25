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

import pytest
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from easydel.etils.etils import EasyDeLQuantizationMethods
from easydel.etils.partition_module import PartitionAxis
from easydel.utils.quantizers import EasyQuantizer

from .transformer_cache import (
	TransformerCache,
	TransformerCacheMetaData,
	TransformerCacheView,
)


class TestTransformerCacheMetaData:
	def test_create_valid(self):
		metadata = TransformerCacheMetaData.create(
			batch_size=4,
			sequence_length=10,
			num_heads=8,
			head_dim=64,
			update_causal_mask=True,
			create_attention_bias=False,
		)
		assert metadata.batch_size == 4
		assert metadata.sequence_length == 10
		assert metadata.num_heads == 8
		assert metadata.head_dim == 64
		assert metadata.key_heads == 8
		assert metadata.value_heads == 8
		assert metadata.key_dim == 64
		assert metadata.value_dim == 64
		assert metadata.update_causal_mask is True
		assert metadata.create_attention_bias is False

	def test_create_with_key_value_dims(self):
		metadata = TransformerCacheMetaData.create(
			batch_size=4,
			sequence_length=10,
			key_heads=4,
			value_heads=2,
			key_dim=128,
			value_dim=64,
			update_causal_mask=False,
			create_attention_bias=True,
		)
		assert metadata.batch_size == 4
		assert metadata.sequence_length == 10
		assert metadata.num_heads is None
		assert metadata.head_dim is None
		assert metadata.key_heads == 4
		assert metadata.value_heads == 2
		assert metadata.key_dim == 128
		assert metadata.value_dim == 64
		assert metadata.update_causal_mask is False
		assert metadata.create_attention_bias is True

	def test_create_missing_head_dim_and_key_value_dims_fails(self):
		with pytest.raises(
			ValueError,
			match="Either head_dim or both key_dim and value_dim must be specified",
		):
			TransformerCacheMetaData.create(
				batch_size=4,
				sequence_length=10,
				num_heads=8,
				update_causal_mask=True,
				create_attention_bias=False,
			)

	def test_create_missing_num_heads_and_key_value_heads_fails(self):
		with pytest.raises(
			ValueError,
			match="Either num_heads or both key_heads and value_heads must be specified",
		):
			TransformerCacheMetaData.create(
				batch_size=4,
				sequence_length=10,
				head_dim=64,
				update_causal_mask=True,
				create_attention_bias=False,
			)

	def test_create_invalid_batch_size(self):
		with pytest.raises(ValueError, match="batch_size must be positive"):
			TransformerCacheMetaData.create(
				batch_size=0,
				sequence_length=10,
				num_heads=8,
				head_dim=64,
			)

	def test_create_invalid_sequence_length(self):
		with pytest.raises(ValueError, match="sequence_length must be positive"):
			TransformerCacheMetaData.create(
				batch_size=4,
				sequence_length=0,
				num_heads=8,
				head_dim=64,
			)

	def test_create_only_head_dim(self):
		metadata = TransformerCacheMetaData.create(
			batch_size=4,
			sequence_length=10,
			num_heads=8,
			head_dim=64,
		)
		assert metadata.key_dim == 64
		assert metadata.value_dim == 64

	def test_create_only_num_heads(self):
		metadata = TransformerCacheMetaData.create(
			batch_size=4,
			sequence_length=10,
			num_heads=8,
			head_dim=64,
		)
		assert metadata.key_heads == 8
		assert metadata.value_heads == 8


class TestTransformerCacheView:
	def test_init(self):
		metadata = TransformerCacheMetaData.create(
			batch_size=2,
			sequence_length=5,
			num_heads=4,
			head_dim=32,
		)
		quantizer = EasyQuantizer(EasyDeLQuantizationMethods.NONE)
		paxis = PartitionAxis()
		key_values_partition_specs = PartitionSpec(
			paxis.batch_axis,
			paxis.key_sequence_axis,
			paxis.head_axis,
			paxis.attention_dim_axis,
		)
		cache_view = TransformerCacheView.init(
			metadata=metadata,
			quantizer=quantizer,
			key_values_partition_specs=key_values_partition_specs,
			dtype=jnp.float32,
		)

		assert isinstance(cache_view.key, jnp.ndarray)
		assert isinstance(cache_view.value, jnp.ndarray)
		assert isinstance(cache_view.index, jnp.ndarray)
		assert cache_view.key.shape == (2, 5, 4, 32)
		assert cache_view.value.shape == (2, 5, 4, 32)
		assert cache_view.index.shape == (2,)
		assert cache_view.index.dtype == jnp.int32
		assert cache_view.metadata == metadata

	def test_repr(self):
		metadata = TransformerCacheMetaData.create(
			batch_size=2,
			sequence_length=5,
			num_heads=4,
			head_dim=32,
		)
		quantizer = EasyQuantizer(EasyDeLQuantizationMethods.NONE)
		paxis = PartitionAxis()
		key_values_partition_specs = PartitionSpec(
			paxis.batch_axis,
			paxis.key_sequence_axis,
			paxis.head_axis,
			paxis.attention_dim_axis,
		)
		cache_view = TransformerCacheView.init(
			metadata=metadata,
			quantizer=quantizer,
			key_values_partition_specs=key_values_partition_specs,
			dtype=jnp.float32,
			layer_index=1,
		)
		expected_repr = (
			"TransformerCacheView(key=(2, 5, 4, 32), value=(2, 5, 4, 32), layer_index=1)"
		)
		assert repr(cache_view) == expected_repr
		assert str(cache_view) == expected_repr


class TestTransformerCache:
	def test_init_layers_cache(self):
		num_layers = 3
		metadata = TransformerCacheMetaData.create(
			batch_size=2,
			sequence_length=5,
			num_heads=4,
			head_dim=32,
		)
		cache = TransformerCache.init_layers_cache(
			num_hidden_layers=num_layers,
			metadata=metadata,
		)

		assert len(cache.views) == num_layers
		for i, view in enumerate(cache.views):
			assert isinstance(view, TransformerCacheView)
			assert view.key.shape == (2, 5, 4, 32)
			assert view.value.shape == (2, 5, 4, 32)
			assert view.layer_index == i
			assert view.metadata == metadata

	def test_init_layers_cache_with_custom_dtype_and_partition_spec(self):
		num_layers = 2
		metadata = TransformerCacheMetaData.create(
			batch_size=1,
			sequence_length=10,
			num_heads=8,
			head_dim=64,
		)
		paxis = PartitionAxis()
		custom_partition_spec = PartitionSpec(
			paxis.batch_axis,
			paxis.key_sequence_axis,
			None,
			paxis.attention_dim_axis,
		)
		cache = TransformerCache.init_layers_cache(
			num_hidden_layers=num_layers,
			metadata=metadata,
			dtype=jnp.float16,
			key_values_partition_specs=custom_partition_spec,
		)
		for view in cache.views:
			assert view.key.dtype == jnp.float16
			assert view.value.dtype == jnp.float16
			# Placeholder check for partition spec
			assert True  # TODO(add a more accurate check after looking for how to do it).

	def test_init_empty(self):
		num_layers = 4
		cache = TransformerCache.init_empty(num_hidden_layers=num_layers)
		assert len(cache.views) == num_layers
		for view in cache.views:
			assert view is None

	def test_repr(self):
		num_layers = 2
		metadata = TransformerCacheMetaData.create(
			batch_size=2,
			sequence_length=5,
			num_heads=4,
			head_dim=32,
		)

		cache = TransformerCache.init_layers_cache(
			num_hidden_layers=num_layers,
			metadata=metadata,
		)
		expected_repr = "TransformerCache(\n  TransformerCacheView(key=(2, 5, 4, 32), value=(2, 5, 4, 32), layer_index=0)\n  TransformerCacheView(key=(2, 5, 4, 32), value=(2, 5, 4, 32), layer_index=1)\n)"
		assert repr(cache) == expected_repr
		assert str(cache) == expected_repr


if __name__ == "__main__":
	pytest.main([__file__])
