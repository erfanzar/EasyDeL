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

import chex
import jax
from jax import lax
from jax import numpy as jnp
from jax import random as jrnd
from jax.extend.backend import get_backend

from .cpu_ops import jax_ring_attention_mu
from .tpu_ops import pallas_ring_attention_tpu

AVAILABLE_RING_ATTENTION_PLATFORM = tp.Literal["pallas", "jax"]
BACKEND = get_backend().platform


def ring_attention(
	query: chex.Array,
	key: chex.Array,
	value: chex.Array,
	bias: tp.Optional[chex.Array] = None,
	segment_ids: tp.Optional[chex.Array] = None,
	axis_name: tp.Optional[str] = None,
	float32_logits: bool = True,
	softmax_scale: tp.Optional[float] = None,
	blocksize_q: int = 512,
	blocksize_k: int = 512,
	blocksize_c: tp.Optional[int] = None,
	deterministic: bool = True,
	dropout_rng: tp.Optional[chex.PRNGKey] = None,
	pdrop: float = 0.0,
	dtype: jnp.dtype = jnp.float32,
	policy=jax.checkpoint_policies.nothing_saveable,
	precision: lax.PrecisionLike = jax.lax.Precision.DEFAULT,
	prevent_cse: bool = True,
	cache_idx=None,
	backend: tp.Literal["cpu", "gpu", "tpu"] = ...,
	platform: AVAILABLE_RING_ATTENTION_PLATFORM = ...,
	autocheck: bool = True,
):
	"""
	Computes ring attention with blockwise transformers.
	Supports JAX, Pallas backends for TPU,GPU,CPU
	Args:
		query: Query array of shape (batch, q_len, num_heads, dim_per_head).
		key: Key array of shape (batch, kv_len, num_heads, dim_per_head).
		value: Value array of shape (batch, kv_len, num_heads, dim_per_head).
		bias: tp.Optional bias array of shape (batch, num_heads, q_len, kv_len).
		segment_ids: tp.Optional segment ids array of shape (batch, seq_len).
		axis_name: Name of the axis to ppermute over.
		float32_logits: Whether to compute logits in float32.
		softmax_scale: scale for softmax or depth ** -0.5.
		blocksize_q: Size of query chunks.
		blocksize_k: Size of key chunks.
		blocksize_c: Size of causal blocks.
		deterministic: Whether to apply dropout.
		dropout_rng: PRNG key for dropout.
		pdrop: Dropout probability.
		dtype: dtype of the computation.
		policy: Checkpoint policy.
		precision: Precision of the computation.
		prevent_cse: Whether to prevent common subexpression elimination.
		platform: platform to be used for func (JAX, Pallas)
		backend: requested backend for func (cpu, tpu, gpu)
		autocheck: whenever to auto check blocksizes(q/k)

	Returns:
		Output array of shape (batch, q_len, num_heads, dim_per_head).
	"""
	if backend == Ellipsis or backend is None:
		backend = BACKEND
	if platform == Ellipsis or platform is None:
		match backend:
			case "gpu":
				platform = "jax"
			case "cpu":
				platform = "jax"
			case "tpu":
				platform = "pallas"
			case _:
				platform = ...

	if platform == Ellipsis:
		raise NotImplementedError(f"there's no available platform for backend {backend}")

	if autocheck:
		blocksize_q = min(blocksize_q, query.shape[1])
		blocksize_k = min(blocksize_k, key.shape[1])
		if backend == "gpu":
			float32_logits = False
	match backend:
		case "gpu":
			match platform:
				case "jax":
					return jax_ring_attention_mu(
						query=query,
						key=key,
						value=value,
						bias=bias,
						segment_ids=segment_ids,
						axis_name=axis_name,
						float32_logits=float32_logits,
						softmax_scale=softmax_scale,
						blocksize_q=blocksize_q,
						blocksize_k=blocksize_k,
						blocksize_c=blocksize_c,
						deterministic=deterministic,
						dropout_rng=dropout_rng,
						pdrop=pdrop,
						dtype=dtype,
						policy=policy,
						precision=precision,
						prevent_cse=prevent_cse,
					)
		case "cpu":
			match platform:
				case "jax":
					return jax_ring_attention_mu(
						query=query,
						key=key,
						value=value,
						bias=bias,
						segment_ids=segment_ids,
						axis_name=axis_name,
						float32_logits=float32_logits,
						softmax_scale=softmax_scale,
						blocksize_q=blocksize_q,
						blocksize_k=blocksize_k,
						blocksize_c=blocksize_c,
						deterministic=deterministic,
						dropout_rng=dropout_rng,
						pdrop=pdrop,
						dtype=dtype,
						policy=policy,
						precision=precision,
						prevent_cse=prevent_cse,
					)
		case "tpu":
			match platform:
				case "jax":
					return jax_ring_attention_mu(
						query=query,
						key=key,
						value=value,
						bias=bias,
						segment_ids=segment_ids,
						axis_name=axis_name,
						float32_logits=float32_logits,
						softmax_scale=softmax_scale,
						blocksize_q=blocksize_q,
						blocksize_k=blocksize_k,
						blocksize_c=blocksize_c,
						deterministic=deterministic,
						dropout_rng=dropout_rng,
						pdrop=pdrop,
						dtype=dtype,
						policy=policy,
						precision=precision,
						prevent_cse=prevent_cse,
					)
				case "pallas":
					return pallas_ring_attention_tpu(
						query=query,
						key=key,
						value=value,
						bias=bias,
						segment_ids=segment_ids,
						cache_idx=cache_idx,
						axis_name=axis_name,
						float32_logits=float32_logits,
						softmax_scale=softmax_scale,
						blocksize_q=blocksize_q,
						blocksize_k=blocksize_k,
						blocksize_c=blocksize_c,
					)
	raise NotImplementedError(
		f"NotImplemented for platform {platform} and backend {backend}."
	)


def _test_forward():
	q_key, k_key, v_key = jrnd.split(jrnd.PRNGKey(8), 3)
	B, H, QS, KS, D = 1, 32, 2048, 2048, 128
	blocksize_k = 512
	blocksize_q = 512
	dtype = jnp.float16
	q = jax.nn.initializers.normal(2)(q_key, (B, QS, H, D), dtype=dtype)
	k = jax.nn.initializers.normal(2)(k_key, (B, KS, H, D), dtype=dtype)
	v = jax.nn.initializers.normal(2)(v_key, (B, KS, H, D), dtype=dtype)
	b = (
		jnp.where(
			jrnd.randint(v_key, (B, H, QS, KS), 0, 4) > 2,
			jnp.finfo(dtype).min,
			0,
		)
		if False
		else None
	)
	print("QKV Allocated")
	try:
		co = ring_attention(
			query=q,
			key=k,
			value=v,
			bias=b,
			blocksize_k=blocksize_k,
			blocksize_q=blocksize_q,
			float32_logits=False,
		)
		print(co[-1, -1, -1, :5])
	except Exception as er:
		print("ring OOM", er)
		co = None
	try:
		import flax

		fo = flax.nnx.dot_product_attention(q, k, v, b)
		print(fo[-1, -1, -1, :5])
	except Exception as er:
		print("Flax OOM", er)
		fo = None
	if fo is not None and co is not None:
		print("Results are Close" if jnp.allclose(co, fo, 0, 0.125) else "Wrong results!")


def _test_backward():
	"""Tests the backward pass of the attention mechanism."""
	q_key, k_key, v_key = jrnd.split(jrnd.PRNGKey(8), 3)
	B, H, S, D = 1, 32, 2048, 16
	blocksize_k = 512
	blocksize_q = 512
	dtype = jnp.float16
	q = jax.nn.initializers.normal(2)(q_key, (B, S, H, D), dtype=dtype)
	k = jax.nn.initializers.normal(2)(k_key, (B, S, H, D), dtype=dtype)
	v = jax.nn.initializers.normal(2)(v_key, (B, S, H, D), dtype=dtype)

	b = (
		jnp.where(
			jrnd.randint(v_key, (B, H, S, S), 0, 4) > 2,
			jnp.finfo(dtype).min,
			0,
		)
		if True  # Set to True to test with bias
		else None
	)

	# try:
	co = jax.grad(
		lambda *x: ring_attention(
			*x,
			None,
			None,
			None,
			None,
			blocksize_q,
			blocksize_k,
		).sum()
	)(q, k, v, b)
	print("Custom op backward pass gradients:")
	print(co[-1][-1, -1, :5])  # Print last 5 elements of last head of last batch
	# except Exception as er:
	# 	print(f"Custom op backward pass failed: {er}")
	# 	co = None

	try:
		import flax

		fo = jax.grad(lambda *x: flax.nnx.dot_product_attention(*x).sum())(q, k, v, b)
		print("Flax backward pass gradients:")
		print(fo[-1][-1, -1, :5])  # Print last 5 elements of last head of last batch
	except Exception as e:
		print(f"Flax backward pass failed : {e}")
		fo = None
		exit()

	if fo is not None and co is not None:
		if jnp.allclose(co, fo, atol=0.125):
			print("Backward pass results are close.")
		else:
			print("Backward pass results differ significantly!")


if __name__ == "__main__":
	_test_forward()
	_test_backward()
