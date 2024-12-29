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

import flax
import flax.nnx
import jax
import jax.numpy as jnp
import jax.random as jrand
import jax.sharding
from fjformer import GenerateRNG
from fjformer.sharding import with_sharding_constraint
from jax import lax

rng = GenerateRNG()


@functools.partial(
	jax.jit,
	static_argnames=[
		"dtype",
		"precision",
		"blocksize_q",
		"blocksize_k",
	],
)
def flash_attention2(
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
	nondiff_argnums=(
		5,
		6,
		7,
		8,
		9,
		10,
		11,
	),
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


@functools.partial(jax.named_call, name="_fwd_flash_attn")
def _fwd_flash_attn(
	query_state: jax.Array,
	key_state: jax.Array,
	value_state: jax.Array,
	mask: tp.Optional[jax.Array],
	bias: tp.Optional[jax.Array],
	dropout: float,
	inference: bool,
	key: tp.Optional[jax.random.PRNGKey],
	blocksize_q: int,
	blocksize_k: int,
	dtype: tp.Optional[jnp.dtype],
	precision: lax.PrecisionLike,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
	"""Forward pass of FlashAttention."""
	b, h, _, d = query_state.shape
	q_seq = query_state.shape[2]
	k_seq = key_state.shape[2]
	assert (
		q_seq % blocksize_q == 0
	), "Query sequence length is not visible by queryblock size"
	assert k_seq % blocksize_k == 0, "Key sequence length is not visible by keyblock size"
	Tr = q_seq // blocksize_q
	Tc = k_seq // blocksize_k
	o_shape = jax.eval_shape(
		lambda: (query_state @ key_state.transpose(0, 1, 3, 2)) @ value_state
	).shape
	o = jnp.zeros(o_shape, dtype=dtype)

	lse = jnp.full((b, h, q_seq), fill_value=-jnp.inf, dtype=jnp.float32)
	if hasattr(query_state, "sharding"):
		if isinstance(query_state.sharding, jax.sharding.NamedSharding):
			with query_state.sharding.mesh:
				o = with_sharding_constraint(o, query_state.sharding.spec)
				lse = with_sharding_constraint(
					lse,
					jax.sharding.PartitionSpec(*query_state.sharding.spec[:3]),
				)
		elif isinstance(
			query_state.sharding, jax.sharding.SingleDeviceSharding
		) and hasattr(query_state.sharding, "_device"):
			o = jax.device_put(o, query_state.sharding._device)
			lse = jax.device_put(lse, query_state.sharding._device)

	global_mask = mask

	@jax.jit
	@functools.partial(jax.named_call, name="_fwd_flash_attn_call_o")
	def call_o(state):
		i, o, lse = state
		q_i = jax.lax.dynamic_slice_in_dim(query_state, i * blocksize_q, blocksize_q, 2)
		o_i = jax.lax.dynamic_slice_in_dim(o, i * blocksize_q, blocksize_q, 2)
		lse_i = jax.lax.dynamic_slice_in_dim(lse, i * blocksize_q, blocksize_q, 2)
		m_i = jnp.full((b, h, blocksize_q), fill_value=-jnp.inf, dtype=dtype)

		@jax.jit
		@functools.partial(jax.named_call, name="_fwd_flash_attn_call_o_call_qk")
		def call_qk(state):
			i, j, o_i, q_i, lse_i, m_i = state
			k_j = jax.lax.dynamic_slice_in_dim(key_state, j * blocksize_k, blocksize_k, 2)
			v_j = jax.lax.dynamic_slice_in_dim(value_state, j * blocksize_k, blocksize_k, 2)

			s_ij = jnp.einsum(
				"bhqd,bhdk->bhqk",
				q_i,
				k_j.transpose(0, 1, 3, 2),
				precision=precision,
			)

			if bias is not None:
				b_i = jax.lax.dynamic_slice_in_dim(bias, i * blocksize_q, blocksize_q, 2)
				b_ij = jax.lax.dynamic_slice_in_dim(b_i, j * blocksize_k, blocksize_k, 3)
				s_ij = s_ij + b_ij
			if global_mask is not None:
				ma_i = jax.lax.dynamic_slice_in_dim(
					global_mask, i * blocksize_q, blocksize_q, 2
				)
				ma_ij = jax.lax.dynamic_slice_in_dim(ma_i, j * blocksize_k, blocksize_k, 3)
				s_ij = jnp.where(ma_ij, s_ij, -1e10)

			if dropout > 0 and not inference:
				rng = jax.random.fold_in(key, i * Tc + j)
				keep_prob = 1.0 - dropout
				broadcast_shape = list(s_ij.shape)
				mask = jax.random.bernoulli(rng, p=keep_prob, shape=broadcast_shape)
				mask = jnp.broadcast_to(mask, s_ij.shape)
				s_ij = lax.select(mask, s_ij / keep_prob, jnp.zeros_like(s_ij))

			m_ij = jnp.maximum(m_i, jnp.max(s_ij, axis=-1))
			p = jnp.exp(s_ij - jnp.expand_dims(m_ij, -1))

			l_ij = jnp.sum(p, -1)

			o_scale = jnp.exp(m_i - m_ij)
			o_i = o_i * jnp.expand_dims(o_scale, -1)

			o_i = o_i + jnp.einsum(
				"bhqk,bhkd->bhqd",
				p,
				v_j,
				precision=precision,
			)

			return (
				i,
				j + 1,
				o_i.astype(dtype),
				q_i.astype(dtype),
				jnp.log(jnp.exp(lse_i - m_ij) + l_ij) + m_ij,
				m_ij.astype(dtype),
			)

		j_end = jnp.minimum(i + 1, Tc) if mask is not None else Tc

		_, _, o_i, _, lse_i, m_i = jax.lax.while_loop(
			lambda state: state[1] < j_end,
			call_qk,
			(i, 0, o_i, q_i, lse_i, m_i),
		)
		o_scale = jnp.exp(m_i - lse_i)
		o_i = o_i * jnp.expand_dims(o_scale, -1)

		o = jax.lax.dynamic_update_slice_in_dim(o, o_i.astype(o.dtype), i * blocksize_q, 2)
		lse = jax.lax.dynamic_update_slice_in_dim(
			lse,
			lse_i.astype(lse.dtype),
			i * blocksize_q,
			2,
		)
		return i + 1, o, lse

	_, o, lse = jax.lax.while_loop(lambda state: state[0] < Tr, call_o, (0, o, lse))

	return o, (
		o,
		lse,
		query_state,  #: jax.Array
		key_state,  #: jax.Array
		value_state,  #: jax.Array
		mask,
		bias,
	)


@functools.partial(jax.named_call, name="_bwd_flash_attn")
def _bwd_flash_attn(
	dropout: float,
	inference: bool,
	key: tp.Optional[jax.random.PRNGKey],
	blocksize_q: int,
	blocksize_k: int,
	dtype: tp.Optional[jnp.dtype],
	precision: lax.PrecisionLike,
	residuals,
	grad_in: jax.Array,
):
	"""Backward pass of FlashAttention."""

	del dtype
	(
		O,  # noqa: E741
		L,
		query_state,
		key_state,
		value_state,
		mask,
		bias,
	) = residuals
	dO = grad_in

	b, h, _, d = query_state.shape
	q_seq = query_state.shape[2]
	k_seq = key_state.shape[2]
	assert q_seq % blocksize_q == 0
	assert k_seq % blocksize_k == 0
	Tr = q_seq // blocksize_q
	Tc = k_seq // blocksize_k

	D = jnp.sum(dO * O, axis=-1)

	dQ = (query_state * 0.0).astype(query_state.dtype)
	dK = (key_state * 0.0).astype(key_state.dtype)
	dV = (value_state * 0.0).astype(value_state.dtype)
	global_mask = mask
	is_causal = mask is not None

	@jax.jit
	@functools.partial(jax.named_call, name="_bwd_flash_attn_call_o")
	def call_o(state):
		j, dQ, dK, dV = state
		k_j = jax.lax.dynamic_slice_in_dim(key_state, j * blocksize_k, blocksize_k, 2)
		v_j = jax.lax.dynamic_slice_in_dim(value_state, j * blocksize_k, blocksize_k, 2)

		dK_j = jax.lax.dynamic_slice_in_dim(dK, j * blocksize_k, blocksize_k, 2)
		dV_j = jax.lax.dynamic_slice_in_dim(dV, j * blocksize_k, blocksize_k, 2)

		@jax.jit
		@functools.partial(jax.named_call, name="_bwd_flash_attn_call_o_call_qk")
		def do_inner_block(state):
			i, j, dQ, dK_j, dV_j = state
			q_i = jax.lax.dynamic_slice_in_dim(query_state, i * blocksize_q, blocksize_q, 2)
			dQ_i = jax.lax.dynamic_slice_in_dim(dQ, i * blocksize_q, blocksize_q, 2)
			dO_i = jax.lax.dynamic_slice_in_dim(dO, i * blocksize_q, blocksize_q, 2)

			L_i = jax.lax.dynamic_slice_in_dim(L, i * blocksize_q, blocksize_q, 2)
			D_i = jax.lax.dynamic_slice_in_dim(D, i * blocksize_q, blocksize_q, 2)
			s_ij = q_i @ k_j.transpose(0, 1, 3, 2)
			if dropout > 0 and not inference:
				rng = jax.random.fold_in(key, i * Tc + j)
				keep_prob = 1.0 - dropout
				broadcast_shape = list(s_ij.shape)
				mask = jax.random.bernoulli(rng, p=keep_prob, shape=broadcast_shape)
				mask = jnp.broadcast_to(mask, s_ij.shape)
				s_ij = lax.select(mask, s_ij / keep_prob, jnp.zeros_like(s_ij))

			if bias is not None:
				b_i = jax.lax.dynamic_slice_in_dim(bias, i * blocksize_q, blocksize_q, 2)
				b_ij = jax.lax.dynamic_slice_in_dim(b_i, j * blocksize_k, blocksize_k, 3)
				s_ij = s_ij + b_ij

			if global_mask is not None:
				ma_i = jax.lax.dynamic_slice_in_dim(
					global_mask, i * blocksize_q, blocksize_q, 2
				)
				ma_ij = jax.lax.dynamic_slice_in_dim(ma_i, j * blocksize_k, blocksize_k, 3)
				s_ij = jnp.where(ma_ij, s_ij, -1e10)

			p_ij = jnp.exp(s_ij - jnp.expand_dims(L_i, -1))

			if dropout > 0 and not inference:
				rng = jax.random.fold_in(key, i * Tc + j)
				keep_prob = 1.0 - dropout
				broadcast_shape = list(p_ij.shape)
				mask = jax.random.bernoulli(rng, p=keep_prob, shape=broadcast_shape)
				mask = jnp.broadcast_to(mask, p_ij.shape)
				p_ij = lax.select(mask, p_ij / keep_prob, jnp.zeros_like(p_ij))

			dV_j = dV_j + jnp.matmul(p_ij.transpose(0, 1, 3, 2), dO_i, precision=precision)

			dP_ij = jnp.matmul(dO_i, v_j.transpose(0, 1, 3, 2), precision=precision)

			dS_ij = p_ij * (dP_ij - D_i[..., None])
			dQ_i = dQ_i + jnp.matmul(dS_ij, k_j, precision=precision)
			dK_j = dK_j + jnp.matmul(dS_ij.transpose(0, 1, 3, 2), q_i, precision=precision)
			dQ = jax.lax.dynamic_update_slice_in_dim(
				dQ,
				dQ_i.astype(dQ.dtype),
				i * blocksize_q,
				2,
			)
			return (
				i + 1,
				j,
				dQ.astype(query_state.dtype),
				dK_j.astype(key_state.dtype),
				dV_j.astype(value_state.dtype),
			)

		i_start = j if is_causal else 0
		_, j, dQ, dK_j, dV_j = jax.lax.while_loop(
			lambda state: state[0] < Tr,
			do_inner_block,
			(i_start, j, dQ, dK_j, dV_j),
		)

		dK = jax.lax.dynamic_update_slice_in_dim(
			dK, dK_j.astype(dK.dtype), j * blocksize_q, 2
		)
		dV = jax.lax.dynamic_update_slice_in_dim(
			dV, dV_j.astype(dV.dtype), j * blocksize_q, 2
		)

		return j + 1, dQ, dK, dV

	_, dQ, dK, dV = jax.lax.while_loop(
		lambda state: state[0] < Tc,
		call_o,
		(0, dQ, dK, dV),
	)

	return dQ, dK, dV, None, None


_flash_attn2.defvjp(_fwd_flash_attn, _bwd_flash_attn)


def fwd_test():
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
	result = flash_attention2(
		query_state=q,
		key_state=k,
		value_state=v,
		bias=b,
		dtype=dtype,
		blocksize_q=64,
		blocksize_k=64,
	)

	print(f"PRED : {result[0,0,0,:5]}")
	print(f"ORGN : {excepted_result[0,0,0,:5]}")

	print(jnp.allclose(excepted_result, result, atol=0.125, rtol=0))


def bwd_test():
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
		lambda *x: flash_attention2(
			*x,
			dtype=dtype,
			blocksize_q=qs,
			blocksize_k=s,
			precision=jax.lax.Precision("HIGHEST".lower()),
		).sum()
	)(q, k, v)

	print(f"PRED BWD : {result[0,0,0,:5]}")
	print(f"ORGN BWD : {excepted_result[0,0,0,:5]}")

	print(jnp.allclose(excepted_result, result, atol=0.125, rtol=0))


jax_flash_attn_2_mu = flash_attention2

__all__ = ["jax_flash_attn_2_mu"]

if __name__ == "__main__":
	# fwd_test()
	bwd_test()
