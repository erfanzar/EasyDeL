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


import functools
import typing as tp

import jax
from jax import Array
from jax import numpy as jnp
from jax import random as jr
from jax.experimental.pallas.ops.tpu.splash_attention import (
	BlockSizes,
	CausalMask,
	MultiHeadMask,
	SegmentIds,
	make_splash_mqa_single_device,
)
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as Ps
from easydel.kernels.tpu_ops import ragged_attention_pallas
from easydel.layers.caching.transformer.transformer_cache import TransformerCacheView
from .._attention_impl import (
	AttentionImpl,
	AttentionMetadata,
	AttentionOutput,
	AttentionRegistry,
)
from .vanilla import VanillaAttn


@AttentionRegistry.register
class SplashAttn(AttentionImpl):
	"""
	An attention implementation using the Pallas Splash Attention kernel for TPUs.

	Splash Attention is an optimized attention mechanism designed for TPUs.
	This implementation provides a wrapper around the `make_splash_mqa_single_device`
	primitive.

	Note:
	    - This implementation is primarily intended for TPUs.
	    - It falls back to `VanillaAttn` under certain conditions:
	        - Query sequence length is 1 (generation mode).
	        - `causal` is False.
	        - Query sequence length is not divisible by 128 (kernel constraint).
	    - Non-TPU forward methods (`forward_native`, `forward_gpu`, etc.) are not
	      implemented and will raise `NotImplementedError`.

	Registered under the name "splash".
	"""

	@classmethod
	def get_impl_name(cls) -> tp.Union[str, tp.Tuple[str]]:
		"""
		Returns the registered name of this attention implementation.

		Returns:
		    The string "splash".
		"""
		return "splash"

	def get_impl_metadata(self) -> AttentionMetadata:
		"""
		Returns the metadata associated with this attention implementation instance.

		Returns:
		    The `AttentionMetadata` provided during initialization.
		"""
		return self.metadata

	def forward_native(self, *args, **kwargs) -> AttentionOutput:
		"""Native (CPU) forward pass. Not implemented for Splash Attention."""
		raise NotImplementedError(
			"Splash Attention does not have a native CPU implementation."
		)

	def forward_gpu(self, *args, **kwargs) -> AttentionOutput:
		"""GPU forward pass. Not implemented for Splash Attention."""
		raise NotImplementedError(
			"Splash Attention does not have a generic GPU implementation."
		)

	@jax.named_scope("easydel-splashimpl-tpu")
	def forward_tpu(
		self,
		q: Array,
		k: Array,
		v: Array,
		mask: tp.Optional[Array] = None,
		causal: bool = True,
		cache_view: tp.Optional[TransformerCacheView] = None,
		**ignore,
	) -> AttentionOutput:
		"""
		Performs Splash Attention on TPU using the Pallas kernel.

		Handles fallback logic, mask processing, block size configuration, and
		sharding via `shard_map`. Expects inputs potentially in BTHD format and
		transposes them to BHTD for the kernel.

		Args:
		    q: Query tensor (B, T, Hq, D).
		    k: Key tensor (B, S, Hkv, D).
		    v: Value tensor (B, S, Hkv, Dv).
		    mask: Optional boolean attention mask (broadcastable to B, 1, T, S).
		        Used to generate segment IDs if provided.
		    causal: If True, applies causal masking via the kernel's mask configuration.
		        If False, falls back to VanillaAttn.
		    **ignore: Ignored keyword arguments.

		Returns:
		    An `AttentionOutput` object containing the attention outputs. Attention weights
		    are not computed or returned by Splash Attention.
		"""
		query_lenght = q.shape[1]
		value_lenght = v.shape[1]
		if (
			not causal
			or ((query_lenght % 128) != 0)
			or ((q.shape[-1] % 128) != 0)
			or ((v.shape[-1] % 128) != 0)
		):
			return VanillaAttn(self.metadata)(
				q=q,
				k=k,
				v=v,
				mask=mask,
				causal=causal,
			)
		sm_scale = self.metadata.softmax_scale
		sm_scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
		dtype = self.metadata.runtime_dtype
		runtime_type = self.get_runtime_type(q=q, BTHD=False)

		(
			query_partition_spec,
			key_partition_spec,
			value_partition_spec,
			bias_partition_spec,
			mask_partition_spec,
			attention_partition_spec,
		) = self.metadata.get_partition_specs(runtime_type, BTHD=False)
		if mask is not None and mask.shape[0] != q.shape[0]:
			num_reps_mask = q.shape[0] // mask.shape[0]
			mask = jnp.repeat(mask, num_reps_mask, 0)

		block_sizes = BlockSizes(
			block_q=min(self.metadata.blocksize_q, query_lenght),
			block_kv_compute=min(self.metadata.blocksize_k, value_lenght),
			block_kv=min(self.metadata.blocksize_k, value_lenght),
			block_q_dkv=min(self.metadata.blocksize_q, query_lenght),
			block_kv_dkv=min(self.metadata.blocksize_k, value_lenght),
			block_kv_dkv_compute=min(self.metadata.blocksize_k, value_lenght),
			block_q_dq=min(self.metadata.blocksize_q, query_lenght),
			block_kv_dq=min(self.metadata.blocksize_k, value_lenght),
		)
		qkv_mask_partition_spec = Ps(query_partition_spec[0], query_partition_spec[2])
		views_partition_spec = Ps(query_partition_spec[0])
		q_mask, kv_mask = [None] * 2
		if mask is not None:
			q_mask, kv_mask = self._split_attention_mask(mask)
			q_mask, kv_mask = (q_mask.astype("i4"), kv_mask.astype("i4"))
			# pallas dont support int1 or bool in shardmap idk why
		pi = [0, 1, 3]
		mpi = [0]
		# query_partition_spec is like PB,PH,PS,PD
		# v is like BSHD since it's not transposed yet
		gather_ps = query_partition_spec[1]
		tparallel = self.metadata.mesh.shape[gather_ps] if gather_ps is not None else None
		pi = [0]
		if tparallel is not None:
			if (v.shape[2] % tparallel) == 0 and tparallel <= v.shape[2]:
				pi = [0, 3]  # shard DP, FSDP and TP
		index, prefill_length = [None] * 2
		if cache_view is not None:
			index, prefill_length = cache_view.index, cache_view.prefill_length

		@functools.partial(
			shard_map,
			mesh=self.metadata.mesh,
			in_specs=(
				self.create_stable_sharding(query_partition_spec, pi, dep=q),
				self.create_stable_sharding(key_partition_spec, pi, dep=k),
				self.create_stable_sharding(value_partition_spec, pi, dep=v),
				self.create_stable_sharding(qkv_mask_partition_spec, mpi, dep=q_mask),
				self.create_stable_sharding(qkv_mask_partition_spec, mpi, dep=kv_mask),
				self.create_stable_sharding(views_partition_spec, mpi, dep=index),
				self.create_stable_sharding(views_partition_spec, mpi, dep=prefill_length),
			),
			out_specs=self.create_stable_sharding(attention_partition_spec, pi),
			check_rep=False,
		)
		def _wraped_flash_attn(q, k, v, q_mask, kv_mask, index, prefill_length):
			output_shape = q.shape[:-1] + (v.shape[-1],)
			num_reps = q.shape[1] // k.shape[1]
			q = q.reshape(q.shape[:-3] + (k.shape[-3], num_reps, q.shape[-2], q.shape[-1]))
			if q.shape[-2] != 1:
				fn = jax.vmap(
					jax.vmap(
						make_splash_mqa_single_device(
							mask=MultiHeadMask(
								[CausalMask((q.shape[-2], k.shape[-2])) for _ in range(q.shape[-3])]
							),
							block_sizes=block_sizes,
						),
						in_axes=(0, 0, 0, None),
					),
					in_axes=(0, 0, 0, 0),
				)
				m = None
				if kv_mask is not None:
					m = SegmentIds(q_mask, kv_mask)
				out = fn(q * sm_scale, k, v, m)
			else:
				out = jax.vmap(
					functools.partial(
						ragged_attention_pallas,
						scale=sm_scale,
						block_kv=512,
						block_bs=32,
					),
					(1, 1, 1, None, None),
					1,
				)(q[..., 0, :], k, v, prefill_length, index)
			return out.reshape(output_shape)

		attn = _wraped_flash_attn(
			q.transpose(0, 2, 1, 3).astype(dtype),
			k.transpose(0, 2, 1, 3).astype(dtype),
			v.transpose(0, 2, 1, 3).astype(dtype),
			q_mask,
			kv_mask,
			index,
			prefill_length,
		).transpose(0, 2, 1, 3)

		return AttentionOutput(attention_weights=None, attention_outputs=attn)

	def forward_cpu(self, *args, **kwargs) -> AttentionOutput:
		"""CPU forward pass. Not implemented for Splash Attention."""
		raise NotImplementedError("Splash Attention does not have a CPU implementation.")

	def forward_cuda(self, *args, **kwargs) -> AttentionOutput:
		"""CUDA GPU forward pass. Not implemented for Splash Attention."""
		raise NotImplementedError("Splash Attention does not have a CUDA implementation.")

	def forward_rocm(self, *args, **kwargs) -> AttentionOutput:
		"""ROCm GPU forward pass. Not implemented for Splash Attention."""
		raise NotImplementedError("Splash Attention does not have a ROCm implementation.")

	def __call__(
		self,
		q: Array,
		k: Array,
		v: Array,
		mask: tp.Optional[Array] = None,
		causal: bool = True,
		cache_view: tp.Optional[TransformerCacheView] = None,
		**ignore,
	) -> AttentionOutput:
		"""
		Executes the Splash Attention computation or falls back to Vanilla Attention.

		Calls the appropriate backend-specific forward method (`forward_tpu`) via
		`super().__call__`. If the backend is not TPU or fallback conditions are met,
		it relies on the fallback mechanism within `forward_tpu`.

		Args:
		    q: Query tensor.
		    k: Key tensor.
		    v: Value tensor.
		    mask: Optional attention mask.
		    causal: If True, applies causal masking. Affects fallback logic and
		        kernel configuration.
				cache_view: cache view for current layer.
		    **ignore: Additional ignored keyword arguments.

		Returns:
		    An `AttentionOutput` object containing the attention results.
		"""
		return super().__call__(
			q=q,
			k=k,
			v=v,
			mask=mask,
			causal=causal,
			cache_view=cache_view,
			**ignore,
		)


if __name__ == "__main__":
	from easydel.infra import EasyDeLBaseConfig

	test_cases = [
		# (batch_size, q_seq_len, k_seq_len, q_heads, k_heads)
		(1, 2048, 2048, 32, 4),
		(2, 2**13, 2**13, 32, 8),
		(4, 2**14, 2**14, 16, 8),
		(4, 2**13, 2**14, 16, 4),
	]

	metadata = AttentionMetadata(
		runtime_dtype=jnp.bfloat16,
		base_config=EasyDeLBaseConfig(axis_dims=(1, 1, 1, -1)),
	)

	splash_attn = SplashAttn(metadata)
	vanilla_attn = VanillaAttn(metadata)

	for idx, (b, qs, ks, qh, kh) in enumerate(test_cases):
		d, vd = 128, 128
		print(
			f"Running test case {idx + 1}/{len(test_cases)}: "
			f"b={b}, qs={qs}, ks={ks}, qh={qh}, kh={kh}, d={d}, vd={vd}"
		)
		key_q, key_k, key_v = jr.split(jr.PRNGKey(0), 3)

		q = jr.normal(key_q, (b, qs, qh, d), dtype=jnp.float32)
		k = jr.normal(key_k, (b, ks, kh, d), dtype=jnp.float32)
		v = jr.normal(key_v, (b, ks, kh, vd), dtype=jnp.float32)

		mask = SplashAttn._create_causal_mask(max(qs, ks))[-qs:, :ks]
		mask = jnp.broadcast_to(mask, (b, 1, qs, ks))
		splash_out = splash_attn(q=q, k=k, v=v, mask=None).attention_outputs
		vanilla_out = vanilla_attn(q=q, k=k, v=v, mask=None).attention_outputs
		is_close = jnp.allclose(splash_out, vanilla_out, atol=0.125)
		max_diff = jnp.max(jnp.abs(splash_out - vanilla_out))

		print(f"Test case {idx + 1} result: {'PASS' if is_close else 'FAIL'}")
		print(f"Maximum absolute difference: {max_diff}\n")
