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
from functools import partial

import jax
from eformer import common_types
from jax import Array
from jax import numpy as jnp
from jax import random as jr
from jax.experimental import shard_map
from jax.sharding import PartitionSpec as Ps
from easydel.kernels.tpu_ops import pallas_ragged_decode
from easydel.layers.caching.transformer.transformer_cache import TransformerCacheView
from .._attention_impl import (
	AttentionImpl,
	AttentionMetadata,
	AttentionOutput,
	AttentionRegistry,
)

shard_map = shard_map.shard_map


@AttentionRegistry.register
class AutoRegressiveDecodeAttn(AttentionImpl):
	"""
	Attention implementation tailored for the autoregressive decoding step.

	This class handles the attention mechanism when generating tokens one by one,
	attending to the previously generated sequence stored in a cache. It utilizes
	`shard_map` for distributed computation and supports different backends,
	including a potential Pallas-optimized version for TPUs. It assumes the
	query sequence length is 1.

	Attributes:
	    metadata (AttentionMetadata): Configuration metadata for the attention mechanism.
	"""

	@classmethod
	def get_impl_name(cls) -> tp.Union[str, tp.Tuple[str]]:
		"""
		Returns the registered name of this attention implementation.

		Returns:
		    The string "autoregressive_decodeattn".
		"""
		return "autoregressive_decodeattn"

	def get_impl_metadata(self) -> AttentionMetadata:
		"""
		Returns the metadata associated with this attention implementation instance.

		Returns:
		    The `AttentionMetadata` provided during initialization.
		"""
		return self.metadata

	@jax.named_scope("easydel-autoregressive_decodeattn-native-xla")
	def forward_native(
		self,
		q: Array,
		k: Array,
		v: Array,
		cache_view: TransformerCacheView,
		**ignores,
	) -> AttentionOutput:
		"""
		Performs the native JAX/XLA forward pass for autoregressive decoding attention.

		This implementation uses `shard_map` to distribute the computation and relies
		on standard JAX operations (`einsum`, `softmax`). It calculates attention weights
		between the single query token and the keys in the cache, applies masking
		based on the valid range defined in `cache_view`, computes the softmax,
		and finally computes the weighted sum of values.

		Args:
		    q (Array): Query tensor of shape (batch_size, 1, num_query_heads, head_dim).
		    k (Array): Key tensor (from cache) of shape (batch_size, kv_sequence_length, num_kv_heads, head_dim).
		    v (Array): Value tensor (from cache) of shape (batch_size, kv_sequence_length, num_kv_heads, head_dim).
		    cache_view (TransformerCacheView): Contains metadata about the cache, specifically
		        `starts` (start index for valid keys/values) and `index` (current index or length).
		    **ignores: Ignored keyword arguments.

		Returns:
		    AttentionOutput: An object containing the attention outputs (`attention_outputs`)
		        of shape (batch_size, 1, num_query_heads, head_dim). Attention weights are not returned.
		"""
		sm_scale = self.metadata.softmax_scale
		sm_scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
		model_mode = self.get_mode(q=q, BTHD=True)
		assert model_mode == common_types.MODE_DECODE
		(
			query_sharding,
			key_sharding,
			value_sharding,
			bias_sharding,
			mask_sharding,
			attention_sharding,
		) = self.metadata.get_shardings(model_mode, True)
		views_sharding = Ps(query_sharding[0])

		@partial(
			shard_map,
			mesh=self.metadata.mesh,
			in_specs=(
				self.create_stable_sharding(
					query_sharding,
					dep=q,
					tensor=q,
					preserved_indices=[0],
				),
				self.create_stable_sharding(
					key_sharding,
					dep=k,
					tensor=k,
					preserved_indices=[0],
				),
				self.create_stable_sharding(
					value_sharding,
					dep=v,
					tensor=v,
					preserved_indices=[0],
				),
				self.create_stable_sharding(
					views_sharding,
					dep=cache_view.index,
					tensor=cache_view.index,
				),
				self.create_stable_sharding(
					views_sharding,
					dep=cache_view.starts,
					tensor=cache_view.starts,
				),
			),
			out_specs=self.create_stable_sharding(
				attention_sharding,
				tensor=q,
				preserved_indices=[0],
			),
			check_rep=True,
		)
		def _compute(q, k, v, start, index):
			Bs, qlen, qhead, qdim = q.shape
			_, kvlen, kvhead, kvdim = k.shape
			assert qlen == 1
			repeats = qhead // kvhead
			q = q.reshape(Bs, 1, kvhead, repeats, qdim)
			weight = jnp.einsum("bskhd,bmkd->bkhsm", q * sm_scale, k)
			ranges = jnp.arange(kvlen).reshape(1, -1)
			mask = (start.reshape(-1, 1) <= ranges) & (ranges < index.reshape(-1, 1))
			weight = jnp.where(
				mask[:, None, None, None, :],
				weight,
				common_types.DEFAULT_MASK_VALUE,
			)
			weight = jax.nn.softmax(weight)
			return jnp.einsum("bkhsm,bmkd->bskhd", weight, v).reshape(Bs, qlen, qhead, qdim)

		attn_output = _compute(
			q,
			k,
			v,
			cache_view.starts.reshape(-1, 1),
			cache_view.index.reshape(-1, 1),
		)
		return AttentionOutput(
			attention_weights=None,
			attention_outputs=attn_output,
		)

	def forward_gpu(self, *args, **kwargs) -> AttentionOutput:
		"""
		GPU forward pass for autoregressive decoding attention.

		Currently delegates to `forward_cuda`.
		"""
		return self.forward_cuda(*args, **kwargs)

	def forward_tpu(self, *args, **kwargs) -> AttentionOutput:
		"""
		TPU forward pass for autoregressive decoding attention.

		Currently delegates to the native JAX/XLA implementation (`forward_native`).
		Consider using `_forward_tpu` for potential Pallas optimization if available/enabled.
		"""
		return self.forward_native(*args, **kwargs)

	def _forward_tpu(
		self,
		q: Array,
		k: Array,
		v: Array,
		cache_view: TransformerCacheView,
		**ignores,
	) -> AttentionOutput:
		"""
		TPU-specific forward pass using the Pallas ragged decode kernel.

		This method is intended for TPU execution and leverages the potentially
		optimized `pallas_ragged_decode` kernel for better performance compared
		to the native `einsum`-based implementation.

		Args:
		    q (Array): Query tensor of shape (batch_size, 1, num_query_heads, head_dim).
		    k (Array): Key tensor (from cache) of shape (batch_size, kv_sequence_length, num_kv_heads, head_dim).
		    v (Array): Value tensor (from cache) of shape (batch_size, kv_sequence_length, num_kv_heads, head_dim).
		    cache_view (TransformerCacheView): Contains cache metadata (`starts`, `index`).
		    **ignores: Ignored keyword arguments.

		Returns:
		    AttentionOutput: An object containing the attention outputs (`attention_outputs`)
		        calculated by the Pallas kernel. Attention weights are not returned.
		"""
		sm_scale = self.metadata.softmax_scale
		sm_scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
		model_mode = self.get_mode(q=q, BTHD=True)
		assert model_mode == common_types.MODE_DECODE
		(
			query_sharding,
			key_sharding,
			value_sharding,
			bias_sharding,
			mask_sharding,
			attention_sharding,
		) = self.metadata.get_shardings(model_mode, True)
		views_sharding = Ps(query_sharding[0])

		@partial(
			shard_map,
			mesh=self.metadata.mesh,
			in_specs=(
				self.create_stable_sharding(
					query_sharding,
					dep=q,
					tensor=q,
					preserved_indices=[0],
				),
				self.create_stable_sharding(
					key_sharding,
					dep=k,
					tensor=k,
					preserved_indices=[0],
				),
				self.create_stable_sharding(
					value_sharding,
					dep=v,
					tensor=v,
					preserved_indices=[0],
				),
				self.create_stable_sharding(
					views_sharding,
					dep=cache_view.index,
					tensor=cache_view.index,
				),
				self.create_stable_sharding(
					views_sharding,
					dep=cache_view.starts,
					tensor=cache_view.starts,
				),
			),
			out_specs=self.create_stable_sharding(
				attention_sharding,
				tensor=q,
				preserved_indices=[0],
			),
			check_rep=False,
		)
		def _compute(q, k, v, start, index):
			return pallas_ragged_decode(
				q * sm_scale,
				k,
				v,
				index.reshape(-1),
				start.reshape(-1),
			)[0]

		attn_output = _compute(
			q,
			k,
			v,
			cache_view.starts.reshape(-1, 1),
			cache_view.index.reshape(-1, 1),
		)

		return AttentionOutput(
			attention_weights=None,
			attention_outputs=attn_output,
		)

	def forward_cpu(self, *args, **kwargs) -> AttentionOutput:
		"""
		CPU forward pass for autoregressive decoding attention.

		Delegates to the native JAX/XLA implementation (`forward_native`).
		"""
		return self.forward_native(*args, **kwargs)

	def forward_cuda(self, *args, **kwargs) -> AttentionOutput:
		"""
		CUDA GPU forward pass for autoregressive decoding attention.

		Delegates to the native JAX/XLA implementation (`forward_native`).
		Future optimizations might add CUDA-specific kernels here.
		"""
		return self.forward_native(*args, **kwargs)

	def forward_rocm(self, *args, **kwargs) -> AttentionOutput:
		"""
		ROCm GPU forward pass for autoregressive decoding attention.

		Delegates to the native JAX/XLA implementation (`forward_native`).
		Future optimizations might add ROCm-specific kernels here.
		"""
		return self.forward_native(*args, **kwargs)

	def __call__(
		self,
		q: Array,
		k: Array,
		v: Array,
		cache_view: TransformerCacheView,
		**ignores,
	) -> AttentionOutput:
		"""
		Makes the class instance callable.

		This method routes the call to the appropriate backend-specific forward method
		(e.g., `forward_tpu`, `forward_gpu`) based on the configuration determined
		by the parent class `AttentionImpl`. It passes the necessary arguments:
		query, key, value, and cache view information.

		Args:
		    q (Array): Query tensor.
		    k (Array): Key tensor (from cache).
		    v (Array): Value tensor (from cache).
		    cache_view (TransformerCacheView): Cache metadata.
		    **kwargs: Additional keyword arguments passed to the underlying forward method.

		Returns:
		    AttentionOutput: The result of the attention computation.
		"""
		return super().__call__(
			q=q,
			k=k,
			v=v,
			cache_view=cache_view,
			**ignores,
		)


if __name__ == "__main__":
	from easydel.infra import EasyDeLBaseConfig

	# Test cace when qkv might refer to mla
	b, qs, ks, qh, kh, d, vd = 1, 1024, 1024, 32, 8, 128, 128 + 64
	q = jr.normal(jr.key(0), (b, qs, qh, d), "f2")
	k = jr.normal(jr.key(1), (b, ks, kh, d), "f2")
	v = jr.normal(jr.key(2), (b, ks, kh, vd), "f2")
	a = jnp.astype(jr.randint(jr.key(3), (b, 1, qs, ks), 0, 4) > 2, "b1")

	metadata = AttentionMetadata(
		runtime_dtype=jnp.float16,
		runtime_softmax_dtype=jnp.float32,
		base_config=EasyDeLBaseConfig(),
		# backend="cpu",
	)

	attn = AutoRegressiveDecodeAttn(metadata)
	out = attn(q=q, k=k, v=v, mask=a)
