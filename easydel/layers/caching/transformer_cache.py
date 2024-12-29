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
import typing as tp

import chex as cx
from fjformer import with_sharding_constraint
from fjformer.core import ImplicitArray
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from easydel.etils.etils import EasyDeLQuantizationMethods
from easydel.etils.partition_module import PartitionAxis
from easydel.utils.quantizers import EasyQuantizer


@cx.dataclass
class TransformerCacheMetaData:
	"""Metadata for transformer cache configuration."""

	# Required fields
	batch_size: int
	sequence_length: int

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
		batch_size: int,
		sequence_length: int,
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
		    batch_size: Size of the batch.
		    sequence_length: Length of the sequence.
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
			batch_size=batch_size,
			sequence_length=sequence_length,
			num_heads=num_heads,
			head_dim=head_dim,
			key_heads=key_heads,
			value_heads=value_heads,
			key_dim=key_dim,
			value_dim=value_dim,
			update_causal_mask=update_causal_mask,
			create_attention_bias=create_attention_bias,
		)


@cx.dataclass
class TransformerCacheView:
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
		layer_index: tp.Optional[int] = None,
	):
		return cls(
			key=quantizer(
				with_sharding_constraint(
					x=jnp.zeros(
						shape=(
							metadata.batch_size,
							metadata.sequence_length,
							metadata.key_heads,
							metadata.key_dim,
						),
						dtype=dtype,
					),
					partition_specs=key_values_partition_specs,
				)
			),
			value=quantizer(
				with_sharding_constraint(
					x=jnp.zeros(
						shape=(
							metadata.batch_size,
							metadata.sequence_length,
							metadata.value_heads,
							metadata.value_dim,
						),
						dtype=dtype,
					),
					partition_specs=key_values_partition_specs,
				)
			),
			index=jnp.zeros((metadata.batch_size,), dtype=jnp.int32),
			metadata=metadata,
			layer_index=layer_index,
		)

	def __repr__(self):
		return (
			self.__class__.__name__
			+ f"(key={self.key.shape}, value={self.value.shape}, layer_index={self.layer_index})"
		)

	__str__ = __repr__


@cx.dataclass
class TransformerCache:
	views: tp.List[tp.Optional[TransformerCacheView]]

	@classmethod
	def init_layers_cache(
		cls,
		num_hidden_layers: int,
		metadata: TransformerCacheMetaData,
		quantizer: tp.Optional[EasyQuantizer] = None,
		dtype: tp.Optional[jnp.dtype] = None,
		key_values_partition_specs: tp.Optional[PartitionSpec] = None,
	):
		paxis = PartitionAxis()
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
					layer_index=layer_index,
				)
				for layer_index in range(num_hidden_layers)
			]
		)

	@classmethod
	def init_empty(cls, num_hidden_layers):
		return cls(views=[None for _ in range(num_hidden_layers)])

	def __repr__(self):
		return (
			f"{self.__class__.__name__ }(\n  "
			+ "\n  ".join(str(view) for view in self.views)
			+ "\n)"
		)

	__str__ = __repr__
