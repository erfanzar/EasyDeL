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
from eformer.escale import with_sharding_constraint
from jax import Array
from jax import numpy as jnp
from jax import random as jr
from jax.experimental.shard_map import shard_map

from .._attention_impl import (
	AttentionImpl,
	AttentionMetadata,
	AttentionOutput,
	AttentionRegistry,
	RuntimeType,
)


@AttentionRegistry.register
class ScaledDotProductAttn(AttentionImpl):
	@classmethod
	def get_impl_name(cls) -> tp.Union[str, tp.Tuple[str]]:
		return "sdpa", "cudnn", "cuda_flash_attn2"

	def get_impl_metadata(self) -> AttentionMetadata:
		return self.metadata

	@jax.named_scope("easydel-sdpaimpl-native-xla")
	def forward_native(
		self,
		q: Array,
		k: Array,
		v: Array,
		mask: tp.Optional[Array] = None,
		bias: tp.Optional[Array] = None,
		init_bias: tp.Optional[tp.Callable[[], Array]] = None,
		causal: bool = False,
	) -> AttentionOutput:
		sm_scale = self.metadata.softmax_scale
		sm_scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
		dtype = self.metadata.runtime_dtype
		runtime_type = self.get_runtime_type(q=q, BTHD=True)
		func = functools.partial(
			jax.nn.dot_product_attention,
			implementation="xla",
			scale=sm_scale,
			is_causal=causal if bias is None else False,
		)

		(
			query_partition_spec,
			key_partition_spec,
			value_partition_spec,
			bias_partition_spec,
			mask_partition_spec,
			attention_partition_spec,
		) = self.metadata.get_partition_specs(runtime_type, True)
		if mask is None and bias is None and init_bias is not None:
			bias = init_bias()
		with self.metadata.mesh:
			attention_output = shard_map(
				func,
				mesh=self.metadata.mesh,
				in_specs=(
					self.create_stable_sharding(query_partition_spec, dep=q),
					self.create_stable_sharding(key_partition_spec, dep=k),
					self.create_stable_sharding(value_partition_spec, dep=v),
					self.create_stable_sharding(bias_partition_spec, dep=bias),
					self.create_stable_sharding(mask_partition_spec, dep=mask),
				),
				out_specs=self.create_stable_sharding(attention_partition_spec, [0, 2]),
				check_rep=False,
			)(
				q.astype(dtype),
				k.astype(dtype),
				v.astype(dtype),
				bias.astype(dtype) if bias is not None else None,
				mask.astype("b1") if mask is not None else None,
			)
			return AttentionOutput(
				attention_weights=None,
				attention_outputs=with_sharding_constraint(
					arr=attention_output,
					sharding=attention_partition_spec,
				),
			)

	def forward_gpu(self, *args, **kwargs) -> AttentionOutput:
		return self.forward_cuda(*args, **kwargs)

	def forward_tpu(self, *args, **kwargs) -> AttentionOutput:
		return self.forward_native(*args, **kwargs)

	def forward_cpu(self, *args, **kwargs) -> AttentionOutput:
		return self.forward_native(*args, **kwargs)

	@jax.named_scope("easydel-sdpaimpl-gpu-cuda")
	def forward_cuda(
		self,
		q: Array,
		k: Array,
		v: Array,
		mask: tp.Optional[Array] = None,
		bias: tp.Optional[Array] = None,
		init_bias: tp.Optional[tp.Callable[[], Array]] = None,
		causal: bool = False,
		**ignore,
	) -> AttentionOutput:
		sm_scale = self.metadata.softmax_scale
		sm_scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
		dtype = jnp.float16
		runtime_type = self.get_runtime_type(q=q, BTHD=True)
		func = functools.partial(
			jax.nn.dot_product_attention,
			implementation="cudnn",
			scale=sm_scale,
			is_causal=(causal if not runtime_type == RuntimeType.generation else False),
		)

		(
			query_partition_spec,
			key_partition_spec,
			value_partition_spec,
			bias_partition_spec,
			mask_partition_spec,
			attention_partition_spec,
		) = self.metadata.get_partition_specs(runtime_type, BTHD=True)
		if mask is None and bias is None and init_bias is not None:
			bias = init_bias()
		with self.metadata.mesh:
			attention_output = shard_map(
				func,
				mesh=self.metadata.mesh,
				in_specs=(
					self.create_stable_sharding(query_partition_spec, [0, 2], dep=q),
					self.create_stable_sharding(key_partition_spec, [0, 2], dep=k),
					self.create_stable_sharding(value_partition_spec, [0, 2], dep=v),
					self.create_stable_sharding(bias_partition_spec, dep=bias),
					self.create_stable_sharding(mask_partition_spec, dep=mask),
				),
				out_specs=self.create_stable_sharding(attention_partition_spec, [0, 2]),
				check_rep=False,
			)(
				q.astype(dtype),
				k.astype(dtype),
				v.astype(dtype),
				bias.astype(dtype) if bias is not None else None,
				mask.astype("b1") if mask is not None else None,
			)
			return AttentionOutput(
				attention_weights=None,
				attention_outputs=with_sharding_constraint(
					arr=attention_output,
					sharding=attention_partition_spec,
				),
			)

	def forward_rocm(self, *args, **kwargs) -> AttentionOutput:
		return self.forward_native(*args, **kwargs)

	def __call__(
		self,
		q: Array,
		k: Array,
		v: Array,
		mask: tp.Optional[Array] = None,
		bias: tp.Optional[Array] = None,
		init_bias: tp.Optional[tp.Callable[[], Array]] = None,
		causal: bool = False,
		**ignore,
	) -> AttentionOutput:
		return super().__call__(
			q=q,
			k=k,
			v=v,
			mask=mask,
			bias=bias,
			init_bias=init_bias,
			causal=causal,
		)


if __name__ == "__main__":
	from easydel.infra import EasyDeLBaseConfig

	# Test cace when qkv might refer to mla
	b, qs, ks, qh, kh, d, vd = 1, 1024, 1024, 32, 8, 128, 128
	q = jr.normal(jr.key(0), (b, qs, qh, d), "f2")
	k = jr.normal(jr.key(1), (b, ks, kh, d), "f2")
	v = jr.normal(jr.key(2), (b, ks, kh, vd), "f2")
	a = jnp.astype(jr.randint(jr.key(3), (b, 1, qs, ks), 0, 4) > 2, "b1")

	gpu_attn = ScaledDotProductAttn(
		AttentionMetadata(
			runtime_dtype=jnp.float16,
			base_config=EasyDeLBaseConfig(),
			backend="gpu",
		)
	)
	cpu_attn = ScaledDotProductAttn(
		AttentionMetadata(
			runtime_dtype=jnp.float16,
			base_config=EasyDeLBaseConfig(),
			backend="cpu",
		)
	)
	tpu_attn = ScaledDotProductAttn(
		AttentionMetadata(
			runtime_dtype=jnp.float16,
			base_config=EasyDeLBaseConfig(),
			backend="tpu",
		)
	)  # Fallback to CPU or XLA impl

	cout = cpu_attn(q=q, k=k, v=v, mask=a).attention_outputs
	gout = gpu_attn(q=q, k=k, v=v, mask=a).attention_outputs
	tout = tpu_attn(q=q, k=k, v=v, mask=a).attention_outputs

	print(jnp.allclose(cout, gout, atol=1e-3), jnp.allclose(tout, gout, atol=1e-3))
