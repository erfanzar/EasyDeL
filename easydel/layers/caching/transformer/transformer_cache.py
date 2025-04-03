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
from __future__ import annotations

import typing as tp

import chex as cx
import jax
from eformer.escale import PartitionAxis, with_sharding_constraint
from eformer.jaximus import ImplicitArray
from eformer.pytree import auto_pytree
from jax import lax
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from .._abstracts import (
	BaseCache,
	BaseCacheMetadata,
	BaseCacheView,
	BaseRunTimeMetadata,
)

if tp.TYPE_CHECKING:
	from easydel.layers.quantization.quantizers import EasyQuantizer
else:
	EasyQuantizer = object


@auto_pytree
class TransformerCacheMetaData(BaseCacheMetadata):
	"""Metadata for transformer cache configuration."""

	partition_axis: PartitionAxis
	# Required fields
	batch_size: int
	sequence_length: int
	num_hidden_layers: int
	# Optional attention-related fields
	num_heads: tp.Optional[int]
	head_dim: tp.Optional[int]
	key_heads: tp.Optional[int]
	value_heads: tp.Optional[int]
	key_dim: tp.Optional[int]
	value_dim: tp.Optional[int]

	# Configuration flags
	update_causal_mask: bool
	create_attention_bias: bool

	@classmethod
	def create(
		cls,
		partition_axis: PartitionAxis,
		batch_size: int,
		sequence_length: int,
		num_hidden_layers: int,
		num_heads: tp.Optional[int] = None,
		head_dim: tp.Optional[int] = None,
		key_heads: tp.Optional[int] = None,
		value_heads: tp.Optional[int] = None,
		key_dim: tp.Optional[int] = None,
		value_dim: tp.Optional[int] = None,
		update_causal_mask: bool = True,
		create_attention_bias: bool = True,
	) -> "TransformerCacheMetaData":
		"""
		Create a TransformerCacheMetaData instance with validation.

		Arguments:
				partition_axis: Partition axis.
		    batch_size: Size of the batch.
		    sequence_length: Length of the sequence.
				num_hidden_layers: number of hidden layers.
		    num_heads: Number of attention heads.
		    head_dim: Dimension of each head.
		    key_heads: Number of key heads.
		    value_heads: Number of value heads.
		    key_dim: Dimension of keys.
		    value_dim: Dimension of values.
		    update_causal_mask: Whether to update causal mask.
		    create_attention_bias: Whether to create attention bias.

		Returns:
		    TransformerCacheMetaData instance

		Raises:
		    ValueError: If required parameters are missing or invalid.
		"""

		if batch_size <= 0:
			raise ValueError("batch_size must be positive")
		if sequence_length <= 0:
			raise ValueError("sequence_length must be positive")

		if head_dim is not None:
			key_dim = key_dim or head_dim
			value_dim = value_dim or head_dim
		else:
			if key_dim is None or value_dim is None:
				raise ValueError(
					"Either head_dim or both key_dim and value_dim must be specified"
				)

		# Derive heads from num_heads if not specified
		if num_heads is not None:
			key_heads = key_heads or num_heads
			value_heads = value_heads or num_heads
		else:
			if key_heads is None or value_heads is None:
				raise ValueError(
					"Either num_heads or both key_heads and value_heads must be specified"
				)

		return cls(
			partition_axis=partition_axis,
			batch_size=batch_size,
			sequence_length=sequence_length,
			num_hidden_layers=num_hidden_layers,
			num_heads=num_heads,
			head_dim=head_dim,
			key_heads=key_heads,
			value_heads=value_heads,
			key_dim=key_dim,
			value_dim=value_dim,
			update_causal_mask=update_causal_mask,
			create_attention_bias=create_attention_bias,
		)


@auto_pytree
class TransformerCacheView(BaseCacheView):
	key: tp.Union[cx.Array, ImplicitArray]
	value: tp.Union[cx.Array, ImplicitArray]
	index: tp.Union[cx.Array, ImplicitArray]
	metadata: TransformerCacheMetaData
	layer_index: tp.Optional[int] = None

	@classmethod
	def init(
		cls,
		metadata: TransformerCacheMetaData,
		quantizer: EasyQuantizer,
		key_values_partition_specs: PartitionSpec,
		dtype: jnp.dtype,
		mesh: Mesh,
		layer_index: tp.Optional[int] = None,
	):
		with jax.named_scope("easydel-transformer-cacheview-init"):
			device = NamedSharding(mesh=mesh, spec=key_values_partition_specs)

			out = cls(
				key=quantizer(
					jnp.zeros(
						shape=(
							metadata.batch_size,
							metadata.sequence_length,
							metadata.key_heads,
							metadata.key_dim,
						),
						dtype=dtype,
						device=device,
					),
				),
				value=quantizer(
					jnp.zeros(
						shape=(
							metadata.batch_size,
							metadata.sequence_length,
							metadata.value_heads,
							metadata.value_dim,
						),
						dtype=dtype,
						device=device,
					)
				),
				index=jnp.zeros((metadata.batch_size,), dtype=jnp.int32),
				metadata=metadata,
				layer_index=layer_index,
			)
		return out

	@jax.named_scope("easydel-transformer-cacheview-concatenate-to-cache")
	def concatenate_to_cache(
		self,
		query: cx.Array,
		key: cx.Array,
		value: cx.Array,
		cache_metadata: TransformerMetadata,
		attention_mask: cx.Array,
		kv_sharding: PartitionSpec,
		quantizer: EasyQuantizer,
		causal_mask: tp.Optional[cx.Array] = None,
		token_type_ids: tp.Optional[cx.Array] = None,
	) -> tp.Tuple[cx.Array, cx.Array, cx.Array]:
		"""
		Updates the KV cache with new key/value states and adjusts the attention mask.

		Internal helper function used when KV caching is enabled.

		Args:
		    query (Array): Current query states.
		    key (Array): Current key states.
		    value (Array): Current value states.
		    attention_mask (Array): Base attention mask.
		    causal_mask (tp.Optional[Array], optional): Causal mask. Defaults to None.
		    token_type_ids (tp.Optional[Array], optional): Token type IDs for segment-based masking.
		                                                    Defaults to None.

		Returns:
		    tp.Tuple[Array, Array, Array]:
		        - Updated key cache tensor.
		        - Updated value cache tensor.
		        - Updated attention mask reflecting the cached sequence length.
		"""
		_ = cache_metadata  # not used
		num_updated_cache_vectors = query.shape[1]
		end_index = self.index[0]

		*batch_dims, max_length, num_heads, depth_per_head = self.value.shape

		if attention_mask.ndim == 2:
			attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

		if causal_mask is not None:
			if hasattr(causal_mask, "value"):
				causal_mask = causal_mask.value
			causal_mask = lax.dynamic_slice(
				causal_mask,
				(0, 0, end_index, 0),
				(1, 1, num_updated_cache_vectors, max_length),
			)
			if token_type_ids is not None and num_updated_cache_vectors != 1:
				token_type_mask = jnp.equal(
					jnp.expand_dims(token_type_ids, 2),
					jnp.expand_dims(token_type_ids, 1),
				)

				token_type_mask = jnp.where(token_type_ids == 0, False, token_type_mask)
				token_type_mask = jnp.expand_dims(token_type_mask, 1)
				sequence_length = token_type_ids.shape[1]
				masked_portion = jnp.logical_or(
					token_type_mask[:, :, :num_updated_cache_vectors, :],
					causal_mask[:, :, :, :sequence_length],
				)
				causal_mask = causal_mask.at[:, :, :, :sequence_length].set(masked_portion)

			causal_mask = jnp.broadcast_to(
				causal_mask,
				(query.shape[0],) + causal_mask.shape[1:],
			)

			attention_mask = jnp.broadcast_to(attention_mask, causal_mask.shape)
			attention_mask = jnp.logical_and(attention_mask, causal_mask)

		slice_indices = (0, end_index % self.value.shape[1], 0, 0)

		value_cache = with_sharding_constraint(
			lax.dynamic_update_slice(
				self.value,
				value.astype(self.value.dtype),
				slice_indices,
			),
			sharding=kv_sharding,
		)
		key_cache = with_sharding_constraint(
			lax.dynamic_update_slice(
				self.key,
				key.astype(self.key.dtype),
				slice_indices,
			),
			sharding=kv_sharding,
		)
		pad_mask = jnp.broadcast_to(
			jnp.arange(max_length) < end_index + num_updated_cache_vectors,
			tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
		)
		attention_mask = jnp.logical_and(pad_mask, attention_mask)

		self.key = quantizer(key_cache)
		self.value = quantizer(value_cache)

		self.index = self.index + num_updated_cache_vectors
		return key_cache, value_cache, attention_mask

	def __repr__(self):
		try:
			return (
				self.__class__.__name__
				+ f"(key={self.key.shape}, value={self.value.shape}, layer_index={self.layer_index})"
			)
		except AttributeError:
			return (
				self.__class__.__name__
				+ f"(key={self.key}, value={self.value}, layer_index={self.layer_index})"
			)

	__str__ = __repr__


@auto_pytree
class TransformerCache(BaseCache):
	views: tp.List[tp.Optional[TransformerCacheView]]

	@classmethod
	def init_cache(
		cls,
		metadata: TransformerCacheMetaData,
		mesh: Mesh,
		quantizer: tp.Optional[EasyQuantizer] = None,
		dtype: tp.Optional[jnp.dtype] = None,
		key_values_partition_specs: tp.Optional[PartitionSpec] = None,
	):
		from easydel.infra.etils import EasyDeLQuantizationMethods
		from easydel.layers.quantization.quantizers import EasyQuantizer

		paxis = metadata.partition_axis
		quantizer = quantizer or EasyQuantizer(EasyDeLQuantizationMethods.NONE)
		key_values_partition_specs = key_values_partition_specs or PartitionSpec(
			paxis.batch_axis,
			paxis.key_sequence_axis,
			paxis.head_axis,
			paxis.attention_dim_axis,
		)
		if dtype is None:
			dtype = jnp.bfloat16
		return cls(
			views=[
				TransformerCacheView.init(
					metadata=metadata,
					quantizer=quantizer,
					key_values_partition_specs=key_values_partition_specs,
					dtype=dtype,
					mesh=mesh,
					layer_index=layer_index,
				)
				for layer_index in range(metadata.num_hidden_layers)
			]
		)

	@classmethod
	def init_empty(cls, num_hidden_layers):
		return cls(views=[None for _ in range(num_hidden_layers)])

	def __repr__(self):
		return (
			f"{self.__class__.__name__}(\n  "
			+ "\n  ".join(str(view) for view in self.views)
			+ "\n)"
		)

	__str__ = __repr__


@auto_pytree
class TransformerMetadata(BaseRunTimeMetadata):
	"""
	holds optional metadata for attention runtime
	"""
