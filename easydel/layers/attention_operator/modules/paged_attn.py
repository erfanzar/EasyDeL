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


from functools import partial
import typing as tp

import jax
from eformer import escale as es
from jax import Array
from jax import numpy as jnp
from jax.sharding import NamedSharding as Ns, PartitionSpec as Ps
import numpy as np
from jax.experimental.shard_map import shard_map
from easydel.kernels.tpu_ops.paged_attention_pallas import (
	pallas_paged_attention,
	pallas_prefill_attention,
)
from easydel.layers.caching import PagedAttentionCacheView
from easydel.layers.caching.paged_attention.paged_attention_cache import (
	PagedAttentionCacheMetaData,
	PagedAttentionMetadata,
)

from easydel.layers.attention_operator._attention_impl import (
	AttentionImpl,
	AttentionMetadata,
	AttentionOutput,
	AttentionRegistry,
)


@AttentionRegistry.register
class PagedAttn(AttentionImpl):
	@classmethod
	def get_impl_name(cls) -> tp.Union[str, tp.Tuple[str]]:
		"""
		Returns the registered name of this attention implementation.

		Returns:
		    The string "paged_attention".
		"""
		return "paged_attention"

	def get_impl_metadata(self) -> AttentionMetadata:
		"""
		Returns the metadata associated with this attention implementation instance.

		Returns:
		    The `AttentionMetadata` provided during initialization.
		"""
		return self.metadata

	def forward_native(self, *args, **kwargs) -> AttentionOutput:
		"""Native (CPU) forward pass. Not implemented for Paged Attention."""
		raise NotImplementedError(
			"Paged Attention does not have a native CPU implementation."
		)

	def forward_gpu(self, *args, **kwargs) -> AttentionOutput:
		"""GPU forward pass. Not implemented for Paged Attention."""
		raise NotImplementedError(
			"Paged Attention does not have a generic GPU implementation."
		)

	def _prefill_tpu(
		self,
		q: Array,
		k: Array,
		v: Array,
		cache_view: PagedAttentionCacheView,
		cache_metadata: PagedAttentionMetadata,
	):
		cache_view.write_prefill_to_cache(k, v, cache_metadata)
		return pallas_prefill_attention(
			q=q,
			k_pages=cache_view.key_pages,
			v_pages=cache_view.value_pages,
			length=cache_metadata.prefill_length,
			page_indices=cache_metadata.prefill_page_table,
			sm_scale=self.metadata.softmax_scale,
		).reshape(q.shape)

	def _decode_tpu(
		self,
		q: Array,
		k: Array,
		v: Array,
		cache_view: PagedAttentionCacheView,
		cache_metadata: PagedAttentionMetadata,
	):
		cache_view.write_generate_to_cache(k, v, cache_metadata)
		kv_pages_sharding = cache_view.kv_pages_sharding.spec
		return shard_map(
			partial(
				pallas_paged_attention,
				pages_per_compute_block=8,
				sm_scale=self.metadata.softmax_scale,
			),
			mesh=es.get_incontext_mesh(),
			in_specs=(
				Ps(None, kv_pages_sharding[0], None),
				kv_pages_sharding,
				kv_pages_sharding,
				Ps(),
				Ps(),
			),
			out_specs=Ps(None, kv_pages_sharding[0], None),
			check_rep=False,
		)(
			q,
			cache_view.key_pages,
			cache_view.value_pages,
			cache_metadata.generate_pos + 1,
			cache_metadata.generate_page_table,
		).reshape(q.shape)

	def _mixed_tpu(
		self,
		q: Array,
		k: Array,
		v: Array,
		cache_view: PagedAttentionCacheView,
		cache_metadata: PagedAttentionMetadata,
	):
		total_len, num_attn_heads_per_device, head_dim = q.shape
		output = jnp.zeros(
			shape=(total_len, num_attn_heads_per_device, head_dim),
			dtype=q.dtype,
		)
		padded_prompt_length = cache_metadata.prefill_pos.shape[0]
		prefill_output = self._prefill_tpu(
			q=q[:padded_prompt_length, :, :],
			k=k[:padded_prompt_length, :, :],
			v=v[:padded_prompt_length, :, :],
			cache_view=cache_view,
			cache_metadata=cache_metadata,
		)

		generate_output = self._generate_tpu(
			q=q[padded_prompt_length:, :, :],
			k=k[padded_prompt_length:, :, :],
			v=v[padded_prompt_length:, :, :],
			cache_view=cache_view,
			cache_metadata=cache_metadata,
		)

		output = jax.lax.dynamic_update_slice_in_dim(
			output,
			prefill_output,
			start_index=0,
			axis=0,
		)

		output = jax.lax.dynamic_update_slice_in_dim(
			output,
			generate_output,
			start_index=padded_prompt_length,
			axis=0,
		)
		return cache_view

	@jax.named_scope("easydel-splashimpl-tpu")
	def forward_tpu(
		self,
		q: Array,
		k: Array,
		v: Array,
		cache_view: PagedAttentionCacheView,
		cache_metadata: PagedAttentionMetadata,
		**ignore,
	) -> AttentionOutput:
		sm_scale = self.metadata.softmax_scale
		sm_scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5

		if (
			hasattr(cache_metadata.generate_pos, "shape")
			and len(cache_metadata.generate_pos.shape) == 0
		):
			out = self._prefill_tpu(
				q=q,
				k=k,
				v=v,
				cache_view=cache_view,
				cache_metadata=cache_metadata,
			)
		elif (
			hasattr(cache_metadata.prefill_pos, "shape")
			and len(cache_metadata.prefill_pos.shape) == 0
		):
			out = self._decode_tpu(
				q=q,
				k=k,
				v=v,
				cache_view=cache_view,
				cache_metadata=cache_metadata,
			)
		else:
			out = self._mixed_tpu(
				q=q,
				k=k,
				v=v,
				cache_view=cache_view,
				cache_metadata=cache_metadata,
			)
		return AttentionOutput(
			attention_weights=None,
			attention_outputs=jnp.expand_dims(out, 0),
		)

	def forward_cpu(self, *args, **kwargs) -> AttentionOutput:
		"""CPU forward pass. Not implemented for Paged Attention."""
		raise NotImplementedError("Paged Attention does not have a CPU implementation.")

	def forward_cuda(self, *args, **kwargs) -> AttentionOutput:
		"""CUDA GPU forward pass. Not implemented for Paged Attention."""
		raise NotImplementedError("Paged Attention does not have a CUDA implementation.")

	def forward_rocm(self, *args, **kwargs) -> AttentionOutput:
		"""ROCm GPU forward pass. Not implemented for Paged Attention."""
		raise NotImplementedError("Paged Attention does not have a ROCm implementation.")

	def __call__(
		self,
		q: Array,
		k: Array,
		v: Array,
		cache_view: PagedAttentionCacheView,
		cache_metadata: PagedAttentionMetadata,
		**ignore,
	) -> AttentionOutput:
		if q.ndim == 4:
			q = q.reshape(q.shape[1:])
		if k.ndim == 4:
			k = k.reshape(k.shape[1:])
		if v.ndim == 4:
			v = v.reshape(v.shape[1:])
		return super().__call__(
			q=q,
			k=k,
			v=v,
			cache_view=cache_view,
			cache_metadata=cache_metadata,
			**ignore,
		)


if __name__ == "__main__":
	from easydel.infra import EasyDeLBaseConfig

	prefill_len = 16
	prefill_non_padding_len = 12
	metadata = AttentionMetadata(
		runtime_dtype=jnp.bfloat16,
		base_config=EasyDeLBaseConfig(axis_dims=(1, 1, 1, -1)),
	)
	attn = PagedAttn(metadata)
	mesh = es.create_mesh((1, 1, -1, 1, 1))
	paxis = es.PartitionAxis()
	max_sequences = 64
	head_dim = 128
	num_attn_heads = 16
	num_kv_heads = 8
	total_page_num = 16
	page_size = 8
	num_generate_tokens = 2
	num_page_to_use = prefill_len // page_size
	attention_metadata = PagedAttentionCacheMetaData.create(
		mesh=mesh,
		partition_axis=paxis,
		batch_size=1,
		num_hidden_layers=1,
		max_sequences=max_sequences,
		dtype=jnp.bfloat16,
		hbm_utilization=0.5,
		kv_head_dim_size=head_dim,
		num_kv_heads=num_kv_heads,
		page_size=page_size,
	)
	kv_shape = (num_kv_heads, total_page_num, page_size, head_dim)
	kv_cache = PagedAttentionCacheView(
		metadata=attention_metadata,
		layer_index=0,
		key_pages=jnp.zeros(kv_shape, dtype=jnp.float32),
		value_pages=jnp.zeros(kv_shape, dtype=jnp.float32),
		kv_pages_sharding=Ns(mesh, Ps("tp", None, None, None)),
	)

	def _prefill_test():
		page_table = jnp.array([1, 3, 0, 0, 0, 0])

		q = jnp.ones((prefill_len, num_attn_heads, head_dim), dtype=jnp.float32)
		k = jnp.ones((prefill_len, num_kv_heads, head_dim), dtype=jnp.float32)
		v = jnp.ones((prefill_len, num_kv_heads, head_dim), dtype=jnp.float32)
		with mesh:
			out = attn._prefill_tpu(
				q=q,
				k=k,
				v=v,
				cache_view=kv_cache,
				cache_metadata=PagedAttentionMetadata(
					prefill_length=prefill_non_padding_len,
					prefill_pos=jnp.arange(0, 8),
					prefill_page_table=page_table,
					generate_pos=0,
					generate_page_table=0,
				),
			)
			np.testing.assert_allclose(
				kv_cache.key_pages[:, page_table[:2], :, :],
				jnp.ones((num_kv_heads, num_page_to_use, page_size, head_dim)),
			)
			np.testing.assert_allclose(
				kv_cache.value_pages[:, page_table[:2], :, :],
				jnp.ones((num_kv_heads, num_page_to_use, page_size, head_dim)),
			)
			zero_index = [i for i in range(page_size)]
			zero_index = (
				zero_index[0 : page_table[0]]
				+ zero_index[page_table[0] + 1 : page_table[1]]
				+ zero_index[page_table[1] + 1 :]
			)

			np.testing.assert_allclose(
				kv_cache.key_pages[:, zero_index, :, :],
				jnp.zeros((num_kv_heads, page_size - num_page_to_use, page_size, head_dim)),
			)
			np.testing.assert_allclose(
				kv_cache.value_pages[:, zero_index, :, :],
				jnp.zeros((num_kv_heads, page_size - num_page_to_use, page_size, head_dim)),
			)
			assert out.shape == (prefill_len, num_attn_heads, head_dim)
			print(out.shape)

	def _decode_test():
		q = jnp.ones((num_generate_tokens, num_attn_heads, head_dim), dtype=jnp.float32)
		k = jnp.ones((num_generate_tokens, num_kv_heads, head_dim), dtype=jnp.float32)
		v = jnp.ones((num_generate_tokens, num_kv_heads, head_dim), dtype=jnp.float32)
		prng = jax.random.PRNGKey(99)
		page_table = jnp.asarray(
			np.random.choice(
				total_page_num,
				(num_generate_tokens, max_sequences // page_size),
				replace=False,
			)
		)
		page_pos = jax.random.randint(
			prng,
			shape=(num_generate_tokens,),
			minval=0,
			maxval=total_page_num * page_size,
		)
		with mesh:
			out = attn._decode_tpu(
				q=q,
				k=k,
				v=v,
				cache_view=kv_cache,
				cache_metadata=PagedAttentionMetadata(
					prefill_length=0,
					prefill_pos=0,
					prefill_page_table=0,
					generate_pos=page_pos,
					generate_page_table=page_table,
				),
			)

			print(out.shape)

	_decode_test()
	_prefill_test()
