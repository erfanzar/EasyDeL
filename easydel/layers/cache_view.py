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
from dataclasses import dataclass
from functools import partial

import einops
import jax
import jax.experimental
import jax.extend
import jax.lib
import jax.tree_util
from fjformer import with_sharding_constraint
from git import Optional
from jax import lax
from jax import numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec

from easydel.etils.etils import EasyDeLQuantizationMethods
from easydel.etils.partition_module import PartitionAxis
from easydel.utils.quantizers import EasyQuantizer


@dataclass
class AttentionCacheViewConfig:
	batch_size: int
	num_heads: int
	sequence_length: int
	head_dim: int
	cache_dtype: jnp.dtype = jnp.float32
	use_sharded_kv_caching: bool = True
	kv_quantization_method: EasyDeLQuantizationMethods = EasyDeLQuantizationMethods.NONE
	kv_quantization_blocks: int = 64
	num_value_heads: Optional[int] = None
	value_head_dim: Optional[int] = None
	partition_axis: PartitionAxis = PartitionAxis()
	sequence_axis: bool = "sp"

	def __post_init__(self):
		self.num_value_heads = self.num_value_heads or self.num_heads
		self.value_head_dim = self.value_head_dim or self.head_dim


class AttentionCacheView:
	def __init__(self, config: AttentionCacheViewConfig):
		self.config = config
		paxs = self.config.partition_axis
		self.kv_partition = PartitionSpec(
			paxs.batch_axis,
			paxs.key_sequence_axis,
			paxs.head_axis,
			paxs.attention_dim_axis,
		)
		self.generation_kv_partition = PartitionSpec(
			paxs.batch_axis,
			None,
			paxs.head_axis,
			paxs.attention_dim_axis,
		)
		key_shape = (
			config.batch_size,
			config.sequence_length,
			config.num_heads,
			config.head_dim,
		)
		value_shape = (
			config.batch_size,
			config.sequence_length,
			config.num_value_heads,
			config.value_head_dims,
		)
		self.quantizer = EasyQuantizer(
			quantization_method=config.kv_quantization_method,
			block_size=config.kv_quantization_blocks,
		)
		self._cached_key = self.quantizer(
			jnp.zeros(
				shape=key_shape,
				dtype=config.cache_dtype,
				device=self.kv_partition,
			)
		)
		self._cached_value = self.quantizer(
			jnp.zeros(
				shape=value_shape,
				dtype=config.cache_dtype,
				device=self.kv_partition,
			)
		)
		self._cache_index = jnp.array(0, dtype=jnp.int32)

	@property
	def index(self):
		return self._cache_index

	def concatenate_to_cache(
		self,
		query: jax.Array,
		key: jax.Array,
		value: jax.Array,
		attention_mask: jax.Array,
	):
		cur_index = self._cache_index
		if (
			query.shape[1] == 1
			and self.config.use_sharded_kv_caching
			and self.config.kv_quantization_method != EasyDeLQuantizationMethods.NONE
		):
			mesh = self.config.mesh

			@partial(
				shard_map,
				mesh=mesh,
				in_specs=(
					self.kv_partition,
					self.kv_partition,
					self.generation_kv_partition,
					self.generation_kv_partition,
					PartitionSpec(),
				),
				out_specs=(self.kv_partition, self.kv_partition),
				check_rep=False,
			)
			def fn(_cached_key, _cached_value, _key, _value, _cur_index):
				assert _key.shape[1] == 1 and _value.shape[1] == 1, (
					_key.shape,
					_value.shape,
				)
				sp_size = max_length // mesh.shape[self.config.sequence_axis]
				_cur_index = (
					_cur_index - jax.lax.axis_index(self.config.sequence_axis) * sp_size
				)
				_key, _value = jax.lax.cond(
					jnp.logical_and(_cur_index >= 0, _cur_index < sp_size),
					lambda: (
						_cached_key.at[:, _cur_index].set(_key[:, -1]),
						_cached_value.at[:, _cur_index].set(_value[:, -1]),
					),
					lambda: (_cached_key, _cached_value),
				)
				return _key, _value

			key, value = fn(self._cached_key, self._cached_value, key, value, cur_index)
		else:
			*batch_dims, max_length, num_heads, depth_per_head = self._cached_key.shape
			cur_index = self._cache_index
			indices = (0,) * len(batch_dims) + (cur_index, 0, 0)  # type:ignore
			key_val = self._cached_key
			value_val = self._cached_value
			if hasattr(key_val, "materialize"):
				key_val = key_val.materialize()
			if hasattr(value_val, "materialize"):
				value_val = value_val.materialize()

			key = lax.dynamic_update_slice(key_val, key, indices)
			value = lax.dynamic_update_slice(value_val, value, indices)
			num_updated_cache_vectors = query.shape[1]
			pad_mask = jnp.broadcast_to(
				jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
				tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
			)
			attention_mask = jnp.logical_and(pad_mask, attention_mask)
		if self.config.kv_quantization_method != EasyDeLQuantizationMethods.NONE:
			self._cached_key = self.quantizer(key)
			self._cached_value = self.quantizer(value)
		else:
			self._cached_key = with_sharding_constraint(key, self.kv_partition)
			self._cached_value = with_sharding_constraint(value, self.kv_partition)

		num_updated_cache_vectors = query.shape[1]
		self._cache_index = self._cache_index + num_updated_cache_vectors
		return key, value, attention_mask

	@staticmethod
	def repeat_key_value(key, value, num_reps: int):
		return (
			einops.repeat(key, "b s h d -> b s (h r) d", r=num_reps),
			einops.repeat(value, "b s h d -> b s (h r) d", r=num_reps),
		)
