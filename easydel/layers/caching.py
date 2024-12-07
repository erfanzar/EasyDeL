from functools import cached_property
from flax import struct
import typing as tp
import chex as cx
from jax import lax, numpy as jnp
from fjformer.core import ImplicitArray
from fjformer import with_sharding_constraint
from jax.sharding import PartitionSpec

from easydel.etils.partition_module import PartitionAxis
from easydel.utils.quantizers import (
	EasyQuantizer,
	EasyDeLQuantizationMethods,
	EasyDeLPlatforms,
)


@struct.dataclass
class TransformerCacheMetaData:
	"""Metadata for transformer cache configuration."""

	# Required fields
	batch_size: int
	sequence_length: int
	dtype: jnp.dtype
	partition_axis: PartitionAxis

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

	# Quantization settings
	kv_cache_quantization_methods: EasyDeLQuantizationMethods
	quantization_platform: EasyDeLPlatforms
	quantization_block_size: int

	# Default values
	DEFAULT_DTYPE: tp.ClassVar[jnp.dtype] = jnp.float32
	DEFAULT_BLOCK_SIZE: tp.ClassVar[int] = 64

	@classmethod
	def create(
		cls,
		batch_size: int,
		sequence_length: int,
		dtype: jnp.dtype = DEFAULT_DTYPE,
		partition_axis: tp.Optional[PartitionAxis] = None,
		num_heads: tp.Optional[int] = None,
		head_dim: tp.Optional[int] = None,
		key_heads: tp.Optional[int] = None,
		value_heads: tp.Optional[int] = None,
		key_dim: tp.Optional[int] = None,
		value_dim: tp.Optional[int] = None,
		update_causal_mask: bool = True,
		create_attention_bias: bool = True,
		kv_cache_quantization_methods: EasyDeLQuantizationMethods = EasyDeLQuantizationMethods.NONE,
		quantization_platform: EasyDeLPlatforms = EasyDeLPlatforms.JAX,
		quantization_block_size: int = DEFAULT_BLOCK_SIZE,
	) -> "TransformerCacheMetaData":
		"""
		Create a TransformerCacheMetaData instance with validation.

		Arguments:
		    batch_size: Size of the batch.
		    sequence_length: Length of the sequence.
		    dtype: Data type for the cache.
		    partition_axis: Axis for partitioning.
		    num_heads: Number of attention heads.
		    head_dim: Dimension of each head.
		    key_heads: Number of key heads.
		    value_heads: Number of value heads.
		    key_dim: Dimension of keys.
		    value_dim: Dimension of values.
		    update_causal_mask: Whether to update causal mask.
		    create_attention_bias: Whether to create attention bias.
		    kv_cache_quantization_methods: Quantization method for KV cache.
		    quantization_platform: Platform for quantization.
		    quantization_block_size: Block size for quantization.

		Returns:
		    TransformerCacheMetaData instance

		Raises:
		    ValueError: If required parameters are missing or invalid.
		"""

		if batch_size <= 0:
			raise ValueError("batch_size must be positive")
		if sequence_length <= 0:
			raise ValueError("sequence_length must be positive")

		partition_axis = partition_axis or PartitionAxis()

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
			dtype=dtype,
			partition_axis=partition_axis,
			num_heads=num_heads,
			head_dim=head_dim,
			key_heads=key_heads,
			value_heads=value_heads,
			key_dim=key_dim,
			value_dim=value_dim,
			update_causal_mask=update_causal_mask,
			create_attention_bias=create_attention_bias,
			kv_cache_quantization_methods=kv_cache_quantization_methods,
			quantization_platform=quantization_platform,
			quantization_block_size=quantization_block_size,
		)


@struct.dataclass
class TransformerCacheView:
	key: tp.Union[cx.Array, ImplicitArray]
	value: tp.Union[cx.Array, ImplicitArray]
	index: tp.Union[cx.Array, ImplicitArray]
	metadata: TransformerCacheMetaData
	layer_index: tp.Optional[int] = None

	def concatenate(
		self,
		query: cx.Array,
		key: cx.Array,
		value: cx.Array,
		attention_mask: cx.Array,
		causal_mask: tp.Optional[cx.Array] = None,
	) -> tp.Tuple[cx.Array, cx.Array, cx.Array, cx.Array]:
		num_updated_cache_vectors = query.shape[1]
		end_index = self.index[0]
		*batch_dims, max_length, num_heads, depth_per_head = self.value.shape

		if attention_mask.ndim == 2:
			attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

		if causal_mask is not None:
			causal_mask = lax.dynamic_slice(
				causal_mask,
				(0, 0, end_index, 0),
				(1, 1, num_updated_cache_vectors, max_length),
			)
			causal_mask = jnp.broadcast_to(
				causal_mask,
				(query.shape[0],) + causal_mask.shape[1:],
			)
			attention_mask = jnp.broadcast_to(attention_mask, causal_mask.shape)
			attention_mask = jnp.logical_and(attention_mask, causal_mask)

		slice_indices = (0, end_index % self.value.shape[1], 0, 0)
		value_cache = self.value
		key_cache = self.key
		if self.metadata.kv_cache_quantization_methods != EasyDeLQuantizationMethods.NONE:
			key_cache = key_cache.materialize()
			value_cache = value_cache.materialize()

		value_cache = lax.dynamic_update_slice(value_cache, value, slice_indices)
		key_cache = lax.dynamic_update_slice(key_cache, key, slice_indices)
		pad_mask = jnp.broadcast_to(
			jnp.arange(max_length) < self.index[0] + num_updated_cache_vectors,
			tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
		)
		attention_mask = jnp.logical_and(pad_mask, attention_mask)

		self.key = self.quantizer(with_sharding_constraint(key, self.key_values_spec))
		self.value = self.quantizer(with_sharding_constraint(value, self.key_values_spec))
		self.index = self.index + num_updated_cache_vectors

		attention_bias = lax.select(
			attention_mask > 0,
			jnp.full(attention_mask.shape, 0.0).astype(key.dtype),
			jnp.full(attention_mask.shape, jnp.finfo(key.dtype).min).astype(key.dtype),
		)

		return key, value, attention_mask, attention_bias

	@cached_property
	def quantizer(self):
		return EasyQuantizer(
			quantization_method=self.metadata.kv_cache_quantization_methods,
			quantization_platform=self.metadata.quantization_platform,
			block_size=self.metadata.quantization_block_size,
		)

	@cached_property
	def key_values_spec(self):
		return PartitionSpec(
			self.metadata.partition_axis.batch_axis,
			self.metadata.partition_axis.key_sequence_axis,
			self.metadata.partition_axis.head_axis,
			self.metadata.partition_axis.attention_dim_axis,
		)

	@classmethod
	def init(
		cls,
		metadata: TransformerCacheMetaData,
		layer_index: tp.Optional[int] = None,
	):
		key_values_partition_specs = PartitionSpec(
			metadata.partition_axis.batch_axis,
			metadata.partition_axis.key_sequence_axis,
			metadata.partition_axis.head_axis,
			metadata.partition_axis.attention_dim_axis,
		)
		quantizer = EasyQuantizer(
			quantization_method=metadata.kv_cache_quantization_methods,
			quantization_platform=metadata.quantization_platform,
			block_size=metadata.quantization_block_size,
		)

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
						dtype=metadata.dtype,
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
						dtype=metadata.dtype,
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


@struct.dataclass
class TransformerCache:
	views: tp.List[TransformerCacheView]

	@classmethod
	def init_layers_cache(
		cls,
		num_hidden_layers: int,
		metadata: TransformerCacheMetaData,
	):
		return cls(
			views=[
				TransformerCacheView.init(metadata=metadata, layer_index=layer_index)
				for layer_index in range(num_hidden_layers)
			]
		)

	def __repr__(self):
		return (
			f"{self.__class__.__name__ }(\n  "
			+ "\n  ".join(str(view) for view in self.views)
			+ "\n)"
		)

	__str__ = __repr__


if __name__ == "__main__":
	metadata = TransformerCacheMetaData.create(
		batch_size=1,
		sequence_length=2048,
		num_heads=4,
		head_dim=64,
	)
	past_key_values = TransformerCache.init_layers_cache(4, metadata=metadata)
	print(past_key_values)
