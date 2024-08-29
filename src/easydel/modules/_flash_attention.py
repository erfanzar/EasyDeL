# Implementation based on FlashAttention 2 (https://arxiv.org/pdf/2307.08691) by @erfanzar,
# with a few bug fixes and adjustments.

import functools
import jax.sharding
import math
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random
from fjformer.utils import GenerateRNG
from fjformer.sharding import with_sharding_constraint
from jax import lax

rng = GenerateRNG()


def flash_attention2(
	q: jax.Array,
	k: jax.Array,
	v: jax.Array,
	mask: Optional[jax.Array] = None,
	bias: Optional[jax.Array] = None,
	*,
	dropout: float = 0.0,
	inference: bool = True,
	key: Optional[jax.random.PRNGKey] = None,
	q_block: Optional[int] = None,
	k_block: Optional[int] = None,
	dtype: Optional[jnp.dtype] = None,
	precision: lax.PrecisionLike = None,
) -> jax.Array:
	"""
	Computes multi-head attention using FlashAttention implementation.

	This implementation makes use of the FlashAttention algorithm for faster
	and more memory-efficient computation of attention. It is particularly
	beneficial for long sequences.

	Args:
	  q: Query, shape (`batch_size`, `num_heads`, `q_len`, `head_dim`).
	  k: Key, shape (`batch_size`, `num_heads`, `kv_len`, `head_dim`).
	  v: Value, shape (`batch_size`, `num_heads`, `kv_len`, `head_dim`).
	  mask: Optional attention mask. This can be any of the following:

	    - No mask (default):  All attention weights are computed.
	    - Boolean mask (2D): shape (`batch_size`, `q_len`), with `True` for
	      valid and `False` for masked positions.
	    - Integer mask (2D): shape (`batch_size`, `q_len`), where the value at
	      each position indicates the length of the sequence to attend to.
	    - 3D mask: shape (`batch_size`, `q_len`, `kv_len`), with `True` for
	      valid and `False` for masked positions.

	  bias: Optional attention bias.
	  dropout: Dropout rate.
	  inference: Whether to run in inference mode.
	  key: PRNG key for dropout.
	  q_block: Block size for query processing.
	  k_block: Block size for key/value processing.
	  dtype: Optional dtype for the output.
	  precision: Optional precision for matrix multiplication.

	Returns:
	  Output of multi-head attention, with shape
	  (`batch_size`, `num_heads`, `q_len`, `head_dim`).

	Raises:
	  ValueError: If `dropout` is not in the range [0, 1], or if `key` is not
	    provided during training when `dropout` > 0.
	"""
	if not inference and dropout > 0 and key is None:
		raise ValueError("key must be provided for training")
	if dropout < 0 or dropout > 1:
		raise ValueError(f"invalid dropout {dropout}")
	if dtype is not None:
		q = q.astype(dtype)
		k = k.astype(dtype)

	k_block = min(k.shape[2], k_block or 128)
	q_block = min(q.shape[2], q_block or 128)

	q = q / math.sqrt(float(q.shape[-1]))
	return _flash_attn(
		q,
		k,
		v,
		mask,
		bias,
		dropout,
		inference,
		key,
		q_block,
		k_block,
		dtype,
		precision,
	)


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
def _flash_attn(
	q: jax.Array,
	k: jax.Array,
	v: jax.Array,
	mask: Optional[jax.Array],
	bias: Optional[jax.Array],
	dropout: float,
	inference: bool,
	key: Optional[jax.random.PRNGKey],
	q_block: int,
	k_block: int,
	dtype: Optional[jnp.dtype],
	precision: lax.PrecisionLike,
) -> jax.Array:
	"""Custom VJP-enabled wrapper for FlashAttention forward pass."""
	return _fwd_flash_attn(
		q,
		k,
		v,
		mask,
		bias,
		dropout,
		inference,
		key,
		q_block,
		k_block,
		dtype,
		precision,
	)[0]


@functools.partial(jax.named_call, name="_fwd_flash_attn")
def _fwd_flash_attn(
	q: jax.Array,
	k: jax.Array,
	v: jax.Array,
	mask: Optional[jax.Array],
	bias: Optional[jax.Array],
	dropout: float,
	inference: bool,
	key: Optional[jax.random.PRNGKey],
	q_block: int,
	k_block: int,
	dtype: Optional[jnp.dtype],
	precision: lax.PrecisionLike,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
	"""Forward pass of FlashAttention."""
	b, h, _, d = q.shape
	q_seq = q.shape[2]
	k_seq = k.shape[2]
	assert q_seq % q_block == 0
	assert k_seq % k_block == 0
	Tr = q_seq // q_block
	Tc = k_seq // k_block
	o_shape = jax.eval_shape(lambda: (q @ k.transpose(0, 1, 3, 2)) @ v).shape
	o = jnp.zeros(o_shape, dtype=dtype)

	lse = jnp.full((b, h, q_seq), fill_value=-jnp.inf, dtype=jnp.float32)
	if isinstance(q.sharding, jax.sharding.NamedSharding):
		with q.sharding.mesh:
			o = with_sharding_constraint(o, q.sharding.spec)
			lse = with_sharding_constraint(lse, q.sharding.spec[:3])
	elif isinstance(q.sharding, jax.sharding.SingleDeviceSharding) and hasattr(
		q.sharding, "_device"
	):
		o = jax.device_put(o, q.sharding._device)
		lse = jax.device_put(lse, q.sharding._device)

	global_mask = mask

	@functools.partial(jax.named_call, name="_fwd_flash_attn_call_o")
	def call_o(state):
		i, o, lse = state
		q_i = jax.lax.dynamic_slice_in_dim(q, i * q_block, q_block, 2)
		o_i = jax.lax.dynamic_slice_in_dim(o, i * q_block, q_block, 2)
		lse_i = jax.lax.dynamic_slice_in_dim(lse, i * q_block, q_block, 2)
		m_i = jnp.full((b, h, q_block), fill_value=-jnp.inf, dtype=dtype)

		@functools.partial(jax.named_call, name="_fwd_flash_attn_call_o_call_qk")
		def call_qk(state):
			i, j, o_i, q_i, lse_i, m_i = state
			k_j = jax.lax.dynamic_slice_in_dim(k, j * k_block, k_block, 2)
			v_j = jax.lax.dynamic_slice_in_dim(v, j * k_block, k_block, 2)

			s_ij = jnp.einsum(
				"bhqd,bhdk->bhqk",
				q_i,
				k_j.transpose(0, 1, 3, 2),
				precision=precision,
			)
			# assert_equal_shape([s_ij], (b, h, q_block, k_block))
			if bias is not None:
				b_i = jax.lax.dynamic_slice_in_dim(bias, i * q_block, q_block, 2)
				b_ij = jax.lax.dynamic_slice_in_dim(b_i, j * k_block, k_block, 3)
				s_ij = s_ij + b_ij
			if global_mask is not None:
				ma_i = jax.lax.dynamic_slice_in_dim(global_mask, i * q_block, q_block, 2)
				ma_ij = jax.lax.dynamic_slice_in_dim(ma_i, j * k_block, k_block, 3)
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
				o_i,
				q_i,
				jnp.log(jnp.exp(lse_i - m_ij) + l_ij) + m_ij,
				m_ij,
			)

		j_end = jnp.minimum(i + 1, Tc) if mask is not None else Tc
		_, _, o_i, _, lse_i, m_i = jax.lax.while_loop(
			lambda state: state[1] < j_end,
			call_qk,
			(i, 0, o_i, q_i, lse_i, m_i),
		)
		o_scale = jnp.exp(m_i - lse_i)
		o_i = o_i * jnp.expand_dims(o_scale, -1)

		o = jax.lax.dynamic_update_slice_in_dim(o, o_i, i * q_block, 2)
		lse = jax.lax.dynamic_update_slice_in_dim(
			lse,
			lse_i.astype(lse.dtype),
			i * q_block,
			2,
		)
		return i + 1, o, lse

	_, o, lse = jax.lax.while_loop(lambda state: state[0] < Tr, call_o, (0, o, lse))

	return o, (
		o,
		lse,
		q,  #: jax.Array
		k,  #: jax.Array
		v,  #: jax.Array
		mask,
		bias,
	)


def _bwd_flash_attn(
	dropout: float,
	inference: bool,
	key: Optional[jax.random.PRNGKey],
	q_block: int,
	k_block: int,
	dtype: Optional[jnp.dtype],
	precision: lax.PrecisionLike,
	residuals,
	grad_in: jax.Array,
):
	"""Backward pass of FlashAttention."""

	del dtype
	del precision
	(
		O,  # noqa: E741
		L,
		q,
		k,
		v,
		mask,
		bias,
	) = residuals
	dO = grad_in

	b, h, _, d = q.shape
	q_seq = q.shape[2]
	k_seq = k.shape[2]
	assert q_seq % q_block == 0
	assert k_seq % k_block == 0
	Tr = q_seq // q_block
	Tc = k_seq // k_block

	D = jnp.sum(dO * O, axis=-1)

	dQ = (q * 0.0).astype(q.dtype)
	dK = (k * 0.0).astype(k.dtype)
	dV = (v * 0.0).astype(v.dtype)
	global_mask = mask
	is_causal = mask is not None

	@functools.partial(jax.named_call, name="_bwd_flash_attn_call_o")
	def call_o(state):
		j, dQ, dK, dV = state
		k_j = jax.lax.dynamic_slice_in_dim(k, j * k_block, k_block, 2)
		v_j = jax.lax.dynamic_slice_in_dim(v, j * k_block, k_block, 2)

		dK_j = jax.lax.dynamic_slice_in_dim(dK, j * k_block, k_block, 2)
		dV_j = jax.lax.dynamic_slice_in_dim(dV, j * k_block, k_block, 2)

		@functools.partial(jax.named_call, name="_bwd_flash_attn_call_o_call_qk")
		def do_inner_block(state):
			i, j, dQ, dK_j, dV_j = state
			q_i = jax.lax.dynamic_slice_in_dim(q, i * q_block, q_block, 2)
			dQ_i = jax.lax.dynamic_slice_in_dim(dQ, i * q_block, q_block, 2)
			dO_i = jax.lax.dynamic_slice_in_dim(dO, i * q_block, q_block, 2)

			L_i = jax.lax.dynamic_slice_in_dim(L, i * q_block, q_block, 2)
			D_i = jax.lax.dynamic_slice_in_dim(D, i * q_block, q_block, 2)
			s_ij = q_i @ k_j.transpose(0, 1, 3, 2)
			if dropout > 0 and not inference:
				rng = jax.random.fold_in(key, i * Tc + j)
				keep_prob = 1.0 - dropout
				broadcast_shape = list(s_ij.shape)
				mask = jax.random.bernoulli(rng, p=keep_prob, shape=broadcast_shape)
				mask = jnp.broadcast_to(mask, s_ij.shape)
				s_ij = lax.select(mask, s_ij / keep_prob, jnp.zeros_like(s_ij))

			if bias is not None:
				b_i = jax.lax.dynamic_slice_in_dim(bias, i * q_block, q_block, 2)
				b_ij = jax.lax.dynamic_slice_in_dim(b_i, j * k_block, k_block, 3)
				# s_ij = (s_ij.astype(jnp.float32) + b_ij.astype(jnp.float32)).astype(q.dtype)
				s_ij = s_ij + b_ij

			if global_mask is not None:
				ma_i = jax.lax.dynamic_slice_in_dim(global_mask, i * q_block, q_block, 2)
				ma_ij = jax.lax.dynamic_slice_in_dim(ma_i, j * k_block, k_block, 3)
				s_ij = jnp.where(ma_ij, s_ij, -1e10)

			p_ij = jnp.exp(s_ij - jnp.expand_dims(L_i, -1))

			if dropout > 0 and not inference:
				rng = jax.random.fold_in(key, i * Tc + j)
				keep_prob = 1.0 - dropout
				broadcast_shape = list(p_ij.shape)
				mask = jax.random.bernoulli(rng, p=keep_prob, shape=broadcast_shape)
				mask = jnp.broadcast_to(mask, p_ij.shape)
				p_ij = lax.select(mask, p_ij / keep_prob, jnp.zeros_like(p_ij))
			dV_j = dV_j + p_ij.transpose(0, 1, 3, 2) @ dO_i
			dP_ij = dO_i @ v_j.transpose(0, 1, 3, 2)
			dS_ij = p_ij * (dP_ij - D_i[..., None])
			dQ_i = dQ_i + dS_ij @ k_j
			dK_j = dK_j + dS_ij.transpose(0, 1, 3, 2) @ q_i

			dQ = jax.lax.dynamic_update_slice_in_dim(
				dQ,
				dQ_i.astype(dQ.dtype),
				i * q_block,
				2,
			)
			return (
				i + 1,
				j,
				dQ.astype(q.dtype),
				dK_j.astype(k.dtype),
				dV_j.astype(v.dtype),
			)

		i_start = j if is_causal else 0
		i, j, dQ, dK_j, dV_j = jax.lax.while_loop(
			lambda state: state[0] < Tr,
			do_inner_block,
			(i_start, j, dQ, dK_j, dV_j),
		)

		dK = jax.lax.dynamic_update_slice_in_dim(dK, dK_j.astype(dK.dtype), j * q_block, 2)
		dV = jax.lax.dynamic_update_slice_in_dim(dV, dV_j.astype(dV.dtype), j * q_block, 2)

		return j + 1, dQ, dK, dV

	j, dQ, dK, dV = jax.lax.while_loop(
		lambda state: state[0] < Tc,
		call_o,
		(0, dQ, dK, dV),
	)
	return dQ, dK, dV, None, None


_flash_attn.defvjp(_fwd_flash_attn, _bwd_flash_attn)
