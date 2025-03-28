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
	"""
	A standard, non-optimized implementation of multi-head attention.

	This implementation uses basic JAX operations like `jnp.einsum` and standard
	softmax. It serves as a reference implementation and a fallback for platforms
	where optimized kernels (like Flash Attention) are not available or desired.
	It supports features like attention bias, masking, dropout, and Grouped Query
	Attention (GQA)/Multi-Query Attention (MQA) via reshaping.

	Registered under the name "vanilla".
	"""

	@classmethod
	def get_impl_name(cls) -> tp.Union[str, tp.Tuple[str]]:
		"""
		Returns the registered name of this attention implementation.

		Returns:
		    The string "vanilla".
		"""
		return "vanilla"

	def get_impl_metadata(self) -> AttentionMetadata:
		"""
		Returns the metadata associated with this attention implementation instance.

		Returns:
		    The `AttentionMetadata` provided during initialization.
		"""
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
		deterministic: bool = True,  # Default to deterministic (no dropout)
		dropout_rng: tp.Optional[jax.random.PRNGKey] = None,
		**ignore,
	) -> AttentionOutput:
		"""
		Computes multi-head attention using standard JAX operations.

		Supports GQA/MQA by reshaping the query tensor to match the number of
		key/value heads. Applies scaling, optional bias/mask, softmax (potentially
		in float32), and optional dropout.

		Args:
		    q: Query tensor (B, T, H_q, D).
		    k: Key tensor (B, S, H_kv, D).
		    v: Value tensor (B, S, H_kv, D_v).
		    mask: Optional boolean attention mask (broadcastable to B, 1, T, S).
		        Used if `bias` is not provided.
		    bias: Optional attention bias tensor (broadcastable to B, H_q, T, S).
		        Takes precedence over `mask`.
		    init_bias: Optional callable to initialize bias if mask/bias are None.
		    deterministic: If True, disables dropout.
		    dropout_rng: JAX PRNG key for dropout. Required if `deterministic` is False
		        and `dropout_prob` > 0.
		    **ignore: Ignored keyword arguments.

		Returns:
		    An `AttentionOutput` object containing the attention weights (if computed)
		    and the final attention outputs.

		Raises:
		    NotImplementedError: If the bias head dimension cannot be reshaped correctly
		        to match the query head structure for GQA/MQA.
		"""

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
		"""GPU forward pass. Delegates to `forward_native`."""
		return self.forward_cuda(*args, **kwargs)

	def forward_tpu(self, *args, **kwargs) -> AttentionOutput:
		"""TPU forward pass. Delegates to `forward_native`."""
		return self.forward_native(*args, **kwargs)

	def forward_cpu(self, *args, **kwargs) -> AttentionOutput:
		"""CPU forward pass. Delegates to `forward_native`."""
		return self.forward_native(*args, **kwargs)

	def forward_cuda(self, *args, **kwargs) -> AttentionOutput:
		"""CUDA GPU forward pass. Delegates to `forward_native`."""
		return self.forward_native(*args, **kwargs)

	def forward_rocm(self, *args, **kwargs) -> AttentionOutput:
		"""ROCm GPU forward pass. Delegates to `forward_native`."""
		return self.forward_native(*args, **kwargs)

	def __call__(
		self,
		q: Array,
		k: Array,
		v: Array,
		mask: tp.Optional[Array] = None,
		bias: tp.Optional[Array] = None,
		init_bias: tp.Optional[tp.Callable[[], Array]] = None,
		deterministic: bool = True,
		dropout_rng: tp.Optional[jax.random.PRNGKey] = None,
		**ignore,
	) -> AttentionOutput:
		"""
		Executes the vanilla attention computation.

		Calls the appropriate backend-specific forward method via `super().__call__`.
		Since all backend methods delegate to `forward_native`, this effectively
		always runs the native JAX implementation.

		Args:
		    q: Query tensor.
		    k: Key tensor.
		    v: Value tensor.
		    mask: Optional attention mask.
		    bias: Optional attention bias.
		    init_bias: Optional callable to initialize bias.
		    deterministic: If True, disables dropout.
		    dropout_rng: JAX PRNG key for dropout if deterministic is False.
		    **ignore: Additional ignored keyword arguments.

		Returns:
		    An `AttentionOutput` object containing the attention results.
		"""
		# Uses the BaseOperation.__call__ which reads self.metadata.backend for dispatch,
		# but all paths in VanillaAttn lead back to forward_native.
		return super().__call__(
			q=q,
			k=k,
			v=v,
			mask=mask,
			bias=bias,
			init_bias=init_bias,
			deterministic=deterministic,
			dropout_rng=dropout_rng,
			**ignore,
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
