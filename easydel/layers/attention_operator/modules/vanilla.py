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

import jax
from eformer.escale import with_sharding_constraint
from flax.nnx.nn.dtypes import promote_dtype
from jax import Array
from jax import numpy as jnp
from jax import random as jr

from .._attention_impl import (
	AttentionImpl,
	AttentionMetadata,
	AttentionOutput,
	AttentionRegistry,
)


@AttentionRegistry.register
class VanillaAttn(AttentionImpl):
	@classmethod
	def get_impl_name(cls) -> tp.Union[str, tp.Tuple[str]]:
		return "vanilla"

	def get_impl_metadata(self) -> AttentionMetadata:
		return self.metadata

	@jax.named_scope("easydel-vanillaimpl-native-xla")
	def forward_native(
		self,
		q: Array,
		k: Array,
		v: Array,
		mask: tp.Optional[Array] = None,
		bias: tp.Optional[Array] = None,
		init_bias: tp.Optional[tp.Callable[[], Array]] = None,
		deterministic: bool = False,
		dropout_rng: tp.Optional[jax.random.PRNGKey] = None,
		**ignore,
	) -> AttentionOutput:
		sm_scale = self.metadata.softmax_scale
		sm_scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
		dtype = self.metadata.runtime_dtype
		softmax_dtype = (
			jnp.float32
			if self.metadata.runtime_softmax_dtype is None
			else self.metadata.runtime_softmax_dtype
		)
		runtime_type = self.get_runtime_type(q=q, BTHD=True)
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
			if bias is None and mask is None and init_bias is not None:
				bias = init_bias()

			b, qs, qh, d = q.shape
			b, ks, kh, d = k.shape
			*_, vd = v.shape
			num_reps = qh // kh
			q = with_sharding_constraint(arr=q, sharding=query_partition_spec)
			k = with_sharding_constraint(arr=k, sharding=key_partition_spec)
			v = with_sharding_constraint(arr=v, sharding=value_partition_spec)

			bias = (
				with_sharding_constraint(arr=bias, sharding=bias_partition_spec)
				if bias is not None
				else bias
			)
			mask = (
				with_sharding_constraint(arr=mask, sharding=mask_partition_spec)
				if mask is not None
				else mask
			)

			q = jnp.reshape(q, (b, qs, kh, num_reps, d))
			q, k, v = promote_dtype((q, k, v), dtype=dtype)

			aw = jnp.einsum("bskhd,bmkd->bkhsm", q * sm_scale, k, optimize=True)

		if bias is not None:
			if bias.shape[1] == (kh * num_reps):
				bias = bias.reshape(b, kh, num_reps, qs, ks)
			elif bias.shape[1] == kh:
				bias = bias.reshape(b, kh, 1, qs, ks)
			elif bias.shape[1] == 1:
				bias = bias.reshape(b, 1, 1, qs, ks)
			else:
				raise NotImplementedError("bias heads wont match!")
			aw = jnp.add(aw, bias.astype(aw))
		elif mask is not None:
			aw = jnp.where(jnp.expand_dims(mask, 1), aw, jnp.finfo(aw).min)
		aw = jax.nn.softmax(aw.astype(softmax_dtype)).astype(dtype)
		dp = self.metadata.dropout_prob
		if not deterministic and dp > 0.0 and dropout_rng is not None:
			keep_prob = 1.0 - dp
			dropout_shape = tuple([1] * (k.ndim - 2)) + aw.shape[-2:]
			keep = jax.random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore

			multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
			aw = aw * multiplier

		attention = jnp.einsum("bkhsm,bmkd->bskhd", aw, v, optimize=True).reshape(
			b,
			qs,
			qh,
			vd,
		)

		return AttentionOutput(
			attention_weights=aw,
			attention_outputs=with_sharding_constraint(
				arr=attention,
				sharding=attention_partition_spec,
			),
		)

	def forward_gpu(self, *args, **kwargs) -> AttentionOutput:
		return self.forward_cuda(*args, **kwargs)

	def forward_tpu(self, *args, **kwargs) -> AttentionOutput:
		return self.forward_native(*args, **kwargs)

	def forward_cpu(self, *args, **kwargs) -> AttentionOutput:
		return self.forward_native(*args, **kwargs)

	def forward_cuda(self, *args, **kwargs) -> AttentionOutput:
		return self.forward_native(*args, **kwargs)

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
		deterministic: bool = False,
		dropout_rng: tp.Optional[jax.random.PRNGKey] = None,
		**ignore,
	) -> AttentionOutput:
		return super().__call__(
			q=q,
			k=k,
			v=v,
			mask=mask,
			bias=bias,
			init_bias=init_bias,
			deterministic=deterministic,
			dropout_rng=dropout_rng,
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

	attn = VanillaAttn(metadata)
	out = attn(q=q, k=k, v=v, mask=a)
