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


# Implementation based on FlashAttention 2 (https://arxiv.org/pdf/2307.08691) by @erfanzar,
# with a few bug fixes and adjustments.

import functools
import math
import typing as tp

import jax
import jax.numpy as jnp
import jax.sharding
from jax import lax

from ._backward_jax import _bwd_flash_attn
from ._forward_jax import _fwd_flash_attn


@functools.partial(
	jax.jit,
	static_argnames=[
		"dtype",
		"precision",
		"blocksize_q",
		"blocksize_k",
	],
)
def flash_attention(
	query_state: jax.Array,
	key_state: jax.Array,
	value_state: jax.Array,
	mask: tp.Optional[jax.Array] = None,
	bias: tp.Optional[jax.Array] = None,
	*,
	dropout: float = 0.0,
	inference: bool = True,
	key: tp.Optional[jax.random.PRNGKey] = None,
	blocksize_q: tp.Optional[int] = None,
	blocksize_k: tp.Optional[int] = None,
	dtype: tp.Optional[jnp.dtype] = None,
	precision: lax.PrecisionLike = None,
	head_dim: tp.Optional[int] = None,
	softmax_scale: tp.Optional[float] = None,
) -> jax.Array:
	"""
	Computes multi-head attention using FlashAttention implementation.

	This implementation makes use of the FlashAttention algorithm for faster
	and more memory-efficient computation of attention. It is particularly
	beneficial for long sequences.

	Args:
		query_state: Query, shape (`batch_size`, `q_len`, `num_heads`, `head_dim`).
		key_state: Key, shape (`batch_size`, `kv_len`, `num_heads`, `head_dim`).
		value_state: Value, shape (`batch_size`, `kv_len`, `num_heads`, `head_dim`).
		mask: tp.Optional attention mask. This can be any of the following:

			- No mask (default):  All attention weights are computed.
			- Boolean mask (2D): shape (`batch_size`, `q_len`), with `True` for
				valid and `False` for masked positions.
			- Integer mask (2D): shape (`batch_size`, `q_len`), where the value at
				each position indicates the length of the sequence to attend to.
			- 4D mask: shape (`batch_size`, `q_len`, `kv_len`), with `True` for
				valid and `False` for masked positions.

		bias: tp.Optional attention bias.
		dropout: Dropout rate.
		inference: Whether to run in inference mode.
		key: PRNG key for dropout.
		blocksize_q: Block size for query processing.
		blocksize_k: Block size for key/value processing.
		dtype: tp.Optional dtype for the output.
		precision: tp.Optional precision for matrix multiplication.
		head_dim: tp.Optional head dim to be used at `query_state = query_state / math.sqrt(float(head_dim or query_state.shape[-1]))`.
		softmax_scale tp.Optional softmax_scale to be used for `query_state = query_state * softmax_scale`

	Returns:
		Output of multi-head attention, with shape
		(`batch_size`, `q_len`, `num_heads`, `head_dim`).

	Raises:
		ValueError: If `dropout` is not in the range [0, 1], or if `key` is not
			provided during training when `dropout` > 0.
	"""
	query_state, key_state, value_state = map(
		lambda x: x.transpose(0, 2, 1, 3),
		[query_state, key_state, value_state],
	)
	if not inference and dropout > 0 and key is None:
		raise ValueError("key must be provided for training")
	if dropout < 0 or dropout > 1:
		raise ValueError(f"invalid dropout {dropout}")
	if dtype is not None:
		query_state = query_state.astype(dtype)
		key_state = key_state.astype(dtype)

	blocksize_k = min(key_state.shape[2], blocksize_k or 128)
	blocksize_q = min(query_state.shape[2], blocksize_q or 128)
	if head_dim is not None and softmax_scale is not None:
		raise ValueError("you can't pass both `head_dim` and `softmax_scale`.")
	if head_dim is not None:
		query_state = query_state / math.sqrt(float(head_dim))
	elif softmax_scale is not None:
		query_state = query_state * softmax_scale
	else:
		query_state = query_state / math.sqrt(float(query_state.shape[-1]))
	return _flash_attn2(
		query_state,
		key_state,
		value_state,
		mask,
		bias,
		dropout,
		inference,
		key,
		blocksize_q,
		blocksize_k,
		dtype,
		precision,
	).transpose(0, 2, 1, 3)


@functools.partial(
	jax.custom_vjp,
	nondiff_argnums=(5, 6, 7, 8, 9, 10, 11),
)
def _flash_attn2(
	query_state: jax.Array,
	key_state: jax.Array,
	value_state: jax.Array,
	mask: tp.Optional[jax.Array] = None,
	bias: tp.Optional[jax.Array] = None,
	dropout: float = 0.0,
	inference: bool = False,
	key: tp.Optional[jax.random.PRNGKey] = None,
	blocksize_q: int = 128,
	blocksize_k: int = 128,
	dtype: tp.Optional[jnp.dtype] = jnp.float32,
	precision: lax.PrecisionLike = None,
) -> jax.Array:
	"""Custom VJP-enabled wrapper for FlashAttention forward pass."""
	return _fwd_flash_attn(
		query_state,
		key_state,
		value_state,
		mask,
		bias,
		dropout,
		inference,
		key,
		blocksize_q,
		blocksize_k,
		dtype,
		precision,
	)[0]


_flash_attn2.defvjp(_fwd_flash_attn, _bwd_flash_attn)


def fwd_test():
	import flax
	from jax import random as jrand

	from easydel.utils import GenerateRNG

	rng = GenerateRNG()

	b, h, qs, s, d = 1, 32, 2048, 2048, 128
	dtype = jnp.float16

	q = jrand.normal(rng.rng, shape=(b, qs, h, d), dtype=dtype)
	k = jrand.normal(rng.rng, shape=(b, s, h, d), dtype=dtype)
	v = jrand.normal(rng.rng, shape=(b, s, h, d), dtype=dtype)
	b = jnp.where(
		jrand.randint(rng.rng, shape=(b, h, qs, s), minval=0, maxval=3) > 1,
		0,
		jnp.finfo(dtype).min,
	)
	excepted_result = flax.nnx.dot_product_attention(
		query=q,
		key=k,
		value=v,
		bias=b,
	)
	result = flash_attention(
		query_state=q,
		key_state=k,
		value_state=v,
		bias=b,
		dtype=dtype,
		blocksize_q=64,
		blocksize_k=64,
	)

	print(f"PRED : {result[0, 0, 0, :5]}")
	print(f"ORGN : {excepted_result[0, 0, 0, :5]}")

	print(jnp.allclose(excepted_result, result, atol=0.125, rtol=0))


def bwd_test():
	import flax
	from jax import random as jrand

	from easydel.utils import GenerateRNG

	rng = GenerateRNG()
	b, h, qs, s, d = 2, 32, 64, 64, 64
	dtype = jnp.float16

	q = jrand.normal(rng.rng, shape=(b, qs, h, d), dtype=dtype)
	k = jrand.normal(rng.rng, shape=(b, s, h, d), dtype=dtype)
	v = jrand.normal(rng.rng, shape=(b, s, h, d), dtype=dtype)
	b = jnp.where(
		jrand.randint(rng.rng, shape=(b, h, qs, s), minval=0, maxval=3) > 1,
		0,
		jnp.finfo(dtype).min,
	)

	excepted_result = jax.grad(lambda *x: flax.nnx.dot_product_attention(*x).sum())(
		q, k, v
	)
	result = jax.grad(
		lambda *x: flash_attention(
			*x,
			dtype=dtype,
			blocksize_q=qs,
			blocksize_k=s,
			precision=jax.lax.Precision("HIGHEST".lower()),
		).sum()
	)(q, k, v)

	print(f"PRED BWD : {result[0, 0, 0, :5]}")
	print(f"ORGN BWD : {excepted_result[0, 0, 0, :5]}")

	print(jnp.allclose(excepted_result, result, atol=0.125, rtol=0))


jax_flash_attn_2_mu = flash_attention

__all__ = ["jax_flash_attn_2_mu"]

if __name__ == "__main__":
	# fwd_test()
	bwd_test()
