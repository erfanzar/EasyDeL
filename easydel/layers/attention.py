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
import warnings
from enum import Enum
from functools import cached_property

import einops
import flax.nnx as nn
import jax
import jax.experimental
import jax.extend
import jax.lib
import jax.tree_util
from chex import Array
from eformer.escale import with_sharding_constraint
from jax import NamedSharding, lax, random
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from jax import tree_util as jtu
from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.layers.caching import TransformerCacheView
from easydel.layers.quantization.quantizers import EasyQuantizer
from easydel.utils.helpers import get_logger
from .attention_operator import AttentionMetadata, AttentionOutput, AttentionRegistry

logger = get_logger(__name__)


def _get_jax_dtype_from_string(dtype_string):
	dtype_mapping = {
		"<class 'jax.numpy.float32'>": jnp.float32,
		"<class 'jax.numpy.float64'>": jnp.float64,
		"<class 'jax.numpy.int32'>": jnp.int32,
		"<class 'jax.numpy.int64'>": jnp.int64,
		"<class 'jax.numpy.bool_'>": jnp.bool_,
		"<class 'jax.numpy.complex64'>": jnp.complex64,
		"<class 'jax.numpy.complex128'>": jnp.complex128,
	}
	return dtype_mapping.get(dtype_string, dtype_string)


class AttentionMechanisms(str, Enum):
	AUTO = "auto"
	FLASH_ATTN2 = "flash_attn2"
	RING = "ring"
	VANILLA = "vanilla"
	SPLASH = "splash"
	CUDNN = "cudnn"
	BLOCKWISE = "blockwise"
	SDPA = "sdpa"
	CUDA_FLASH_ATTN2 = "cuda_flash_attn2"


def tpu_version_check(version: str = "v4"):
	if version in getattr(jax.local_devices()[0], "device_kind", "").lower():
		return True

	return False


def get_optimal_config() -> tp.Tuple[AttentionMechanisms, jnp.dtype]:
	"""
	Returns the optimal attention mechanism and dtype for the current JAX device.

	Returns:
	    A tuple of (attention_mechanism, dtype)
	"""

	match jax.default_backend():
		case "tpu":
			if tpu_version_check("v3"):
				return AttentionMechanisms.FLASH_ATTN2, jnp.float32
			return AttentionMechanisms.SPLASH, jnp.bfloat16
		case "gpu":
			return AttentionMechanisms.FLASH_ATTN2, jnp.bfloat16
		case _:
			return AttentionMechanisms.VANILLA, jnp.bfloat16


DEFAULT_ATTENTION_MECHANISM = "auto"


class FlexibleAttentionModule(nn.Module):
	"""
	Manages different attention mechanisms for efficient computation in EasyDeL models.

	This class serves as a central hub for handling various attention mechanisms, including
	optimized implementations like FlashAttention, SplashAttention, RingAttention, and more traditional
	approaches like vanilla (dot-product) attention. It provides a unified interface to
	select and execute the appropriate attention mechanism based on the model's configuration and
	hardware platform.

	Key Features:

	* **Attention Mechanism Selection:** Supports a wide range of attention mechanisms,
	  allowing users to choose the most suitable option based on performance and hardware constraints.
	* **Sharding and Partitioning:** Integrates with JAX's sharding capabilities, enabling efficient
	  distribution of computations and data across multiple devices.
	* **Block-wise Computation:** Implements block-wise attention computations for optimized memory
	  usage and speed, particularly beneficial for large models.
	* **Performance Optimization:** Includes support for highly optimized implementations like
	  FlashAttention, SplashAttention, and RingAttention for TPU and GPU acceleration.
	* **Flexibility and Customization:** Offers fine-grained control over attention parameters,
	  sharding specifications, and block sizes, providing flexibility for different use cases.
	* **Testing and Evaluation:** Includes a `run_attention_benchmarks` method to systematically evaluate
	  different attention mechanisms and help users identify the best-performing option.


	The FlexibleAttentionModule class is a crucial component within EasyDeL, responsible for managing and optimizing attention
	computations. It provides a user-friendly way to select and execute different attention mechanisms,
	leveraging JAX's sharding capabilities and offering performance enhancements through specialized implementations
	 like FlashAttention and SplashAttention. Its ability to handle block-wise computations and customization options
	  makes it adaptable to a variety of model architectures and hardware configurations.
	"""

	def __init__(
		self,
		base_config: EasyDeLBaseConfig,
		softmax_scale: float,
		dropout_prob: float = 0.0,
	):
		if isinstance(base_config.attn_dtype, str):
			base_config.attn_dtype = _get_jax_dtype_from_string(base_config.attn_dtype)
		if isinstance(base_config.attn_softmax_dtype, str):
			base_config.attn_softmax_dtype = _get_jax_dtype_from_string(
				base_config.attn_softmax_dtype
			)
		if base_config.attn_mechanism == AttentionMechanisms.AUTO:
			impl_name, runtime_dtype = get_optimal_config()
			logger.debug(f"Automatically select AttentionImpl {impl_name} | {runtime_dtype}")
			base_config.attn_mechanism = impl_name
			base_config.attn_dtype = runtime_dtype

		metadata = AttentionMetadata.from_config(
			config=base_config,
			softmax_scale=softmax_scale,
			dropout_prob=dropout_prob,
		)
		self.impl = AttentionRegistry.create(
			impl_name=base_config.attn_mechanism,
			metadata=metadata,
		)
		self.deterministic = True

	@jax.named_scope("easydel-flexible-attention")
	def forward(
		self,
		query_states: Array,
		key_states: Array,
		value_states: Array,
		bias: tp.Optional[Array] = None,
		init_bias: tp.Optional[tp.Callable[[], Array]] = None,
		attention_mask: tp.Optional[Array] = None,
		segment_ids: tp.Optional[Array] = None,
		causal: bool = True,
		dropout_rng: tp.Optional[random.PRNGKey] = None,
	) -> AttentionOutput:
		return jtu.tree_map(
			lambda x: x.astype(self.impl.metadata.runtime_dtype),
			self.impl(
				q=query_states,
				k=key_states,
				v=value_states,
				bias=bias,
				init_bias=init_bias,
				mask=attention_mask,
				segment_ids=segment_ids,
				causal=causal,
				deterministic=self.deterministic,
				dropout_rng=dropout_rng,
			),
		)

	__call__ = forward


SC = tp.TypeVar("SC")


class FlaxAttentionModule(nn.Module):
	def __init__(
		self,
		config: SC,
	):
		super().__init__()
		self.config: SC | EasyDeLBaseConfig = config

		self.cached_key: nn.Cache[Array] | None = None
		self.cached_value: nn.Cache[Array] | None = None
		self.cache_index: nn.Cache[Array] | None = None

	def make_flexible_sliding_window(
		self,
		attention_mask: jax.Array,
		cache_view: TransformerCacheView,
		sliding_window: int,
	):
		attention_mask = jnp.logical_and(
			self._create_sliding_mask(
				cache_pos=self.build_cache_pos(attention_mask, cache_view),
				curr_index=cache_view.index[0] if cache_view is not None else 0,
				cache_length=attention_mask.shape[-1],
				sliding_windows=sliding_window,
			),
			attention_mask,
		)

		def init_attention_bias():
			return jax.lax.select(
				attention_mask > 0,
				jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
				jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
			)

		return attention_mask, init_attention_bias

	@staticmethod
	def build_cache_pos(
		attention_mask: jax.Array,
		cache_view: TransformerCacheView = None,
	) -> jax.Array:
		end_index = cache_view.index[0] if cache_view is not None else 0
		inipos = jnp.cumsum(jnp.any(attention_mask, -1)[:, -1, :], axis=-1)
		return (inipos - (inipos >= 1)) + end_index

	@cached_property
	def quantizer(self):
		return EasyQuantizer(
			quantization_method=self.config.kv_cache_quantization_method,
			block_size=self.config.kv_cache_quantization_blocksize,
		)

	@property
	def default_key_value_sharding(self):
		paxis = self.config.partition_axis
		return NamedSharding(
			mesh=self.config.mesh,
			spec=PartitionSpec(
				paxis.batch_axis,
				paxis.key_sequence_axis,
				paxis.head_axis,
				paxis.attention_dim_axis,
			),
		)

	def get_sharding_safely(self, tensor: jax.Array) -> PartitionSpec:
		return getattr(tensor, "sharding", self.default_key_value_sharding).spec

	@staticmethod
	def _transpose_sequence_head(*args):
		"""The _transpose_sequence_head function transposes the query, key and value matrices.

		Args:
		    *args: arrays to transpose

		Returns:
		    The transpose of the query, key and value matrices
		"""
		return map(lambda x: jnp.transpose(x, (0, 2, 1, 3)), args)

	@jax.named_scope("easydel-flax-attention-concatenate-to-cache")
	def _concatenate_to_cache(
		self,
		query: Array,
		key: Array,
		value: Array,
		cache_view: TransformerCacheView,
		attention_mask: Array,
		causal_mask: tp.Optional[Array] = None,
		token_type_ids: tp.Optional[Array] = None,
	) -> tp.Tuple[Array, Array, Array]:
		num_updated_cache_vectors = query.shape[1]
		end_index = cache_view.index[0]

		*batch_dims, max_length, num_heads, depth_per_head = cache_view.value.shape

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

		slice_indices = (0, end_index % cache_view.value.shape[1], 0, 0)

		value_cache = lax.dynamic_update_slice(
			cache_view.value,
			value.astype(cache_view.value.dtype),
			slice_indices,
		)
		key_cache = lax.dynamic_update_slice(
			cache_view.key,
			key.astype(cache_view.key.dtype),
			slice_indices,
		)
		pad_mask = jnp.broadcast_to(
			jnp.arange(max_length) < end_index + num_updated_cache_vectors,
			tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
		)
		attention_mask = jnp.logical_and(pad_mask, attention_mask)
		cache_view.key = self.quantizer(
			with_sharding_constraint(
				arr=key_cache,
				sharding=self.get_sharding_safely(cache_view.key),
			)
		)
		cache_view.value = self.quantizer(
			with_sharding_constraint(
				arr=value_cache,
				sharding=self.get_sharding_safely(cache_view.value),
			)
		)
		cache_view.index = cache_view.index + num_updated_cache_vectors
		return key_cache, value_cache, attention_mask

	@staticmethod
	def _create_sliding_mask(
		cache_pos: jnp.ndarray,
		curr_index: int,
		cache_length: int,
		sliding_windows: int,
	):
		total_tokens = curr_index + cache_pos.shape[1]

		def _reconstruct_rotated_cache_positions():
			cache_positions = jnp.arange(cache_length) + total_tokens - cache_length
			cache_positions = (
				jnp.zeros_like(cache_positions)
				.at[cache_positions % cache_length]
				.set(cache_positions)
			)
			return cache_positions

		cache_positions = jax.lax.cond(
			total_tokens <= cache_length,
			lambda: jnp.arange(cache_length),
			_reconstruct_rotated_cache_positions,
		)

		cache_positions = cache_positions[None, None, :]
		cache_pos = cache_pos[:, :, None]
		sliding_mask = cache_positions > cache_pos - sliding_windows
		sliding_mask *= cache_positions < cache_pos + sliding_windows
		return sliding_mask

	@jax.named_scope("easydel-flax-attention-concatenate")
	def concatenate(
		self,
		*,
		query: Array,
		key: Array,
		value: Array,
		attention_mask: Array,
		cache_view: tp.Optional[TransformerCacheView] = None,
		causal_mask: tp.Optional[Array] = None,
		token_type_ids: tp.Optional[Array] = None,
		fcm_mask: tp.Optional[Array] = None,
		sliding_windows: tp.Optional[int] = None,
	) -> tp.Tuple[Array, Array, Array, tp.Callable[[], Array]]:
		if attention_mask is not None:
			if attention_mask.dtype != jnp.bool:
				warnings.warn("attention_mask should be a boolean array", stacklevel=1)
				attention_mask = (attention_mask == 1).astype("b1")
		if cache_view is None:
			query_length = query.shape[1]
			key_length = key.shape[1]
			if causal_mask is not None:
				causal_mask = causal_mask[:, :, :query_length, :key_length]
				if token_type_ids is not None and query_length != 1:
					token_type_mask = jnp.equal(
						jnp.expand_dims(token_type_ids, 2),
						jnp.expand_dims(token_type_ids, 1),
					)

					token_type_mask = token_type_mask.at[token_type_ids == 0].set(False)
					token_type_mask = jnp.expand_dims(token_type_mask, 1)
					sequence_length = token_type_ids.shape[1]

					masked_portion = jnp.logical_or(
						token_type_mask,
						causal_mask[
							:,
							:,
							:,
							:sequence_length,
						],
					)
					causal_mask = causal_mask.at[
						:,
						:,
						:,
						:sequence_length,
					].set(masked_portion)
				causal_mask = jnp.broadcast_to(
					causal_mask, (query.shape[0],) + causal_mask.shape[1:]
				)
				if attention_mask.ndim == 2:
					attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

				attention_mask = jnp.broadcast_to(attention_mask, causal_mask.shape)
				attention_mask = nn.combine_masks(attention_mask, causal_mask, fcm_mask)

			else:
				attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
				attention_mask = jnp.repeat(attention_mask, query.shape[1], -2)
		else:
			key, value, attention_mask = self._concatenate_to_cache(
				query=query,
				key=key,
				value=value,
				cache_view=cache_view,
				attention_mask=attention_mask,
				causal_mask=causal_mask,
				token_type_ids=token_type_ids,
			)
		if sliding_windows is not None and attention_mask is not None:
			sliding_window_mask = jnp.tril(
				jnp.ones_like(attention_mask, dtype=jnp.bool),
				k=-sliding_windows,
			)
			window_mask = jnp.where(sliding_window_mask, 0, 1)
			attention_mask = jnp.logical_and(window_mask, attention_mask)
			if attention_mask.shape[-1] <= 1:
				attention_mask = attention_mask[:, :, :, -sliding_windows:]

		def init_attention_bias():
			return lax.select(
				attention_mask > 0,
				jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
				jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
			)

		return key, value, attention_mask, init_attention_bias

	def shard_attention_prod(self, attn_output: jax.Array) -> jax.Array:
		"""
		shards attention output before passing that to output_proj

		Args:
		    attn_output (jax.Array): merged output of dot product attention with 3 dims, (batch, seqlen, hidden_size).

		Returns:
		    jax.Array: sharded version of `attn_output`
		"""
		return with_sharding_constraint(
			arr=attn_output,
			sharding=PartitionSpec(
				self.config.partition_axis.batch_axis,
				(
					self.config.partition_axis.sequence_axis
					if attn_output.shape[1] != 1
					else None
				),
				self.config.partition_axis.hidden_state_axis,
			),
		)

	def _merge_heads(self, hidden_states: jax.Array) -> jax.Array:
		"""
		Merges the attention heads into a single hidden state tensor.

		Args:
		    hidden_states (jax.Array): The hidden states with separate head dimensions.

		Returns:
		    jax.Array: The hidden states with merged head dimensions.
		"""
		return hidden_states.reshape(hidden_states.shape[:2] + (-1,))

	@staticmethod
	def repeat_key_value(key, value, num_reps: int):
		with jax.named_scope("easydel-flax-attention-repeat-kvheads"):
			key = einops.repeat(key, "b s h d -> b s (h r) d", r=num_reps)
			value = einops.repeat(value, "b s h d -> b s (h r) d", r=num_reps)
		return key, value
