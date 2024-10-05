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

import math
from functools import partial
from typing import Optional

import chex
import jax
from fjformer import GenerateRNG
from flax.linen.attention import dot_product_attention as _dot_product_attention
from jax import custom_vjp, extend
from jax import numpy as jnp
from jax import random as jrand
from jax.experimental import pallas as pl

_PLATFORM = extend.backend.get_backend().platform
_INTERPRET = _PLATFORM == "cpu"
_TEST_DTYPE = jnp.float16
# _INTERPRET = True
_rng = GenerateRNG()


def _jax_bwd_flash_attn(
	dtype: jnp.dtype,
	qblock: int,
	kblock: int,
	softmax_scale: float,
	residuals,
	dO,
):
	"""Backward pass of FlashAttention."""

	del dtype
	(
		query_state,
		key_state,
		value_state,
		bias,
		O,  # noqa: E741
		L,
	) = residuals

	(query_state, key_state, value_state, O, dO) = map(  # noqa: E741
		lambda x: x.transpose(0, 2, 1, 3), [query_state, key_state, value_state, O, dO]
	)

	softmax_scale = softmax_scale or 1.0 / math.sqrt(query_state.shape[-1])

	q_seq = query_state.shape[2]
	k_seq = key_state.shape[2]
	assert q_seq % qblock == 0
	assert k_seq % kblock == 0
	Tr = q_seq // qblock
	Tc = k_seq // kblock

	D = jnp.sum(dO * O, axis=-1)
	dQ = (query_state * 0.0).astype(query_state.dtype)
	dK = (key_state * 0.0).astype(key_state.dtype)
	dV = (value_state * 0.0).astype(value_state.dtype)

	@partial(jax.named_call, name="_bwd_flash_attn_call_o")
	def call_o(state):
		j, dQ, dK, dV = state
		k_j = jax.lax.dynamic_slice_in_dim(key_state, j * kblock, kblock, 2)
		v_j = jax.lax.dynamic_slice_in_dim(value_state, j * kblock, kblock, 2)

		dK_j = jax.lax.dynamic_slice_in_dim(dK, j * kblock, kblock, 2)
		dV_j = jax.lax.dynamic_slice_in_dim(dV, j * kblock, kblock, 2)

		@partial(jax.named_call, name="_bwd_flash_attn_call_o_call_qk")
		def do_inner_block(state):
			i, j, dQ, dK_j, dV_j = state
			q_i = jax.lax.dynamic_slice_in_dim(query_state, i * qblock, qblock, 2)
			dQ_i = jax.lax.dynamic_slice_in_dim(dQ, i * qblock, qblock, 2)
			dO_i = jax.lax.dynamic_slice_in_dim(dO, i * qblock, qblock, 2)
			L_i = jax.lax.dynamic_slice_in_dim(L, i * qblock, qblock, 2)
			D_i = jax.lax.dynamic_slice_in_dim(D, i * qblock, qblock, 2)
			s_ij = (q_i * softmax_scale) @ k_j.transpose(0, 1, 3, 2)
			if bias is not None:
				b_i = jax.lax.dynamic_slice_in_dim(bias, i * qblock, qblock, 2)
				b_ij = jax.lax.dynamic_slice_in_dim(b_i, j * kblock, kblock, 3)
				s_ij = s_ij + b_ij

			p_ij = jnp.exp(s_ij - jnp.expand_dims(L_i, -1))

			dV_j = dV_j + jnp.einsum("...nm,...mk->...nk", p_ij, dO_i)

			dP_ij = jnp.einsum("...ik,...jk->...ij", dO_i, v_j)
			dS_ij = p_ij * (dP_ij - D_i[..., None]) * softmax_scale
			dQ_i = dQ_i + jnp.einsum("...id,...dj->...ij", dS_ij, k_j)
			dK_j = dK_j + jnp.einsum("...nm,...mk->...nk", dS_ij, q_i)
			dQ = jax.lax.dynamic_update_slice_in_dim(
				dQ,
				dQ_i.astype(dQ.dtype),
				i * qblock,
				2,
			)
			return (
				i + 1,
				j,
				dQ.astype(query_state.dtype),
				dK_j.astype(key_state.dtype),
				dV_j.astype(value_state.dtype),
			)

		_, j, dQ, dK_j, dV_j = jax.lax.while_loop(
			lambda state: state[0] < Tr,
			do_inner_block,
			(0, j, dQ, dK_j, dV_j),
		)

		dK = jax.lax.dynamic_update_slice_in_dim(
			dK, dK_j.astype(dK.dtype), j * qblock, 2
		)
		dV = jax.lax.dynamic_update_slice_in_dim(
			dV, dV_j.astype(dV.dtype), j * qblock, 2
		)

		return j + 1, dQ, dK, dV

	_, dQ, dK, dV = jax.lax.while_loop(
		lambda state: state[0] < Tc,
		call_o,
		(0, dQ, dK, dV),
	)
	return (
		dQ.transpose(0, 2, 1, 3),
		dK.transpose(0, 2, 1, 3),
		dV.transpose(0, 2, 1, 3),
		None,
	)


def _gpu_fwd_flash_attn_kernel(
	q_ref,
	k_ref,
	v_ref,
	b_ref,
	# out_refs,
	o_ref,
	lse_ref,
	*,
	dtype,
	qblock,
	kblock,
	seqlen_q,
	seqlen_k,
	softmax_scale,
	nheads,
	block_headdim,
):
	qg, nhbg = pl.program_id(0), pl.program_id(1)
	bg, nhg = nhbg // nheads, nhbg % nheads
	q_seq_mask = qg * qblock + jnp.arange(0, qblock) < seqlen_q
	headdim_mask = jnp.arange(0, block_headdim) < q_ref.shape[3]
	qmask = q_seq_mask[:, None] & headdim_mask[None, :]

	q = pl.load(
		q_ref,
		idx=(bg, pl.dslice(qg * qblock, qblock), nhg, slice(None)),
		mask=qmask,
		other=0.0,
	)

	def body(carry):
		idx, acc_o, max_i, lse_i = carry
		kv_seq_mask = idx * kblock + jnp.arange(0, kblock) < seqlen_k
		key_mask = kv_seq_mask[:, None] & headdim_mask[None, :]
		k = pl.load(
			k_ref,
			idx=(bg, pl.dslice(idx * kblock, kblock), nhg, slice(None)),
			mask=key_mask,
			other=0.0,
		)
		qk = jnp.dot(q, k.transpose(1, 0)).astype(jnp.float32)
		qk = jnp.where(q_seq_mask[:, None] & kv_seq_mask[None, :], qk, float("-inf"))
		if b_ref is not None:
			qk = (qk * softmax_scale) + pl.load(
				b_ref,
				idx=(
					bg,
					nhg,
					pl.dslice(qg * qblock, qblock),
					pl.dslice(idx * kblock, kblock),
				),
				mask=q_seq_mask[:, None] & kv_seq_mask[None, :],
				other=0.0,
			).astype(jnp.float32)
			max_ij = jnp.maximum(jnp.max(qk, -1), lse_i)
			p = jnp.exp(qk - max_ij[:, None])
		else:
			max_ij = jnp.maximum(jnp.max(qk, -1) * softmax_scale, lse_i)
			p = jnp.exp(qk * softmax_scale - max_ij[:, None])
		lse_ij = jnp.sum(p, -1)
		acc_o_scale = jnp.exp(max_i - max_ij)
		acc_o = acc_o * acc_o_scale[:, None]
		v = pl.load(
			v_ref,
			idx=(bg, pl.dslice(idx * kblock, kblock), nhg, slice(None)),
			mask=key_mask,
			other=0.0,
		)

		acc_o += jnp.dot(p.astype(v.dtype), v).astype(acc_o.dtype)
		lin = jnp.exp(lse_i - max_ij) + lse_ij
		return (idx + 1, acc_o, max_ij, max_ij + jnp.log(lin))

	end_krange = pl.cdiv(seqlen_k, kblock)
	init_block = (
		0,
		jnp.zeros((qblock, block_headdim), dtype=jnp.float32),
		jnp.full((qblock,), -float("inf"), dtype=jnp.float32),
		jnp.full((qblock,), -float("inf"), dtype=jnp.float32),
	)

	_, o, max_i, lse_i = jax.lax.while_loop(
		lambda state: state[0] < end_krange,
		body,
		init_block,
	)
	o_scale = jnp.exp(max_i - lse_i)
	o = o * o_scale[:, None]
	pl.store(
		o_ref,
		val=o.astype(dtype),
		idx=(bg, pl.dslice(qg * qblock, qblock), nhg, slice(None)),
		mask=qmask,
	)
	if lse_ref is not None:
		pl.store(
			lse_ref,
			val=lse_i,
			idx=(bg, nhg, pl.dslice(qg * qblock, qblock)),
			mask=q_seq_mask,
		)


def _call_gpu_fwd_flash_attn(
	q: jax.Array,
	k: jax.Array,
	v: jax.Array,
	b: Optional[jax.Array] = None,
	dtype: jnp.dtype = None,
	qblock: int = 128,
	kblock: int = 128,
	softmax_scale: float = None,
	inference_mode: bool = False,
):
	batch_size, seqlen_q, nheads, dim = q.shape
	seqlen_k = k.shape[1]
	softmax_scale = softmax_scale or 1.0 / math.sqrt(dim)
	dtype = dtype or q.dtype

	lse_shape = (batch_size, nheads, seqlen_q)
	in_specs = [
		pl.BlockSpec(q.shape, lambda *_: (0,) * q.ndim),
		pl.BlockSpec(k.shape, lambda *_: (0,) * k.ndim),
		pl.BlockSpec(v.shape, lambda *_: (0,) * v.ndim),
	]
	in_specs.append(
		None if b is None else pl.BlockSpec(b.shape, lambda *_: (0,) * b.ndim)
	)

	out_shape = [
		jax.ShapeDtypeStruct(shape=q.shape, dtype=dtype),
		jax.ShapeDtypeStruct(shape=lse_shape, dtype=jnp.float32),
	]
	out_specs = [
		pl.BlockSpec(q.shape, lambda *_: (0,) * q.ndim),
		pl.BlockSpec(lse_shape, lambda *_: (0,) * len(lse_shape)),
	]

	grid = (pl.cdiv(seqlen_q, qblock), batch_size * nheads)

	method = pl.pallas_call(
		partial(
			_gpu_fwd_flash_attn_kernel,
			dtype=dtype,
			qblock=qblock,
			kblock=kblock,
			seqlen_q=seqlen_q,
			seqlen_k=seqlen_k,
			softmax_scale=softmax_scale,
			nheads=nheads,
			block_headdim=dim,
		),
		grid=grid,
		out_specs=out_specs,
		in_specs=in_specs,
		out_shape=out_shape,
		compiler_params=dict(
			triton=dict(num_wraps=4 if dim <= 64 else 8, num_stages=1)
		),
		interpret=_INTERPRET,
		debug=False,
		name=f"gpu_fwd_flash_attn_{inference_mode}",
	)

	o, lse = method(q, k, v, b)
	#  = outs if len(outs) == 2 else outs[0], None
	if inference_mode:
		del lse
		lse = None
	return o, (
		q,
		k,
		v,
		b,
		o,
		lse,
	)


def _gpu_bwd_do_o_dot(
	o_ref,
	dO_ref,
	delta_ref,
	*,
	nheads,
	seqlen_q,
	headdim,
	qblock,
	block_headdim,
):
	qg, bnhg = pl.program_id(0), pl.program_id(1)
	bg = bnhg // nheads
	nhg = bnhg % nheads

	q_seq_mask = qg * qblock + jnp.arange(0, qblock) < seqlen_q
	headdim_mask = jnp.arange(0, block_headdim) < headdim
	omask = q_seq_mask[:, None] & headdim_mask[None, :]
	idx = (bg, pl.dslice(qg * qblock, qblock), nhg, slice(None))

	o = pl.load(o_ref, idx=idx, mask=omask, other=0.0).astype(jnp.float32)
	dO = pl.load(dO_ref, idx=idx, mask=omask, other=0.0).astype(jnp.float32)
	delta = jnp.sum(o * dO, -1)
	pl.store(
		delta_ref,
		idx=(bg, nhg, pl.dslice(qg * qblock, qblock)),
		val=delta.astype(delta_ref.dtype),
		mask=q_seq_mask,
	)


def _gpu_bwd_flash_attn_kernel(
	q_ref,
	k_ref,
	v_ref,
	b_ref,
	delta_ref,
	lse_ref,
	dO_ref,
	_,
	dQ_ref,
	dK_ref,
	dV_ref,
	*,
	dtype,
	qblock,
	kblock,
	seqlen_q,
	seqlen_k,
	softmax_scale,
	nheads,
	block_headdim,
):
	j, bnhg = pl.program_id(0), pl.program_id(1)

	bg = bnhg // nheads
	nhg = bnhg % nheads

	headdim_mask = jnp.arange(0, block_headdim) < k_ref.shape[3]

	kv_seq_mask = j * kblock + jnp.arange(0, kblock) < seqlen_k

	kj = pl.load(
		k_ref,
		idx=(bg, pl.dslice(j * kblock, kblock), nhg, slice(None)),
		mask=kv_seq_mask[:, None] & headdim_mask[None, :],
		other=0.0,
	)
	vj = pl.load(
		v_ref,
		idx=(bg, pl.dslice(j * kblock, kblock), nhg, slice(None)),
		mask=kv_seq_mask[:, None] & headdim_mask[None, :],
		other=0.0,
	)

	def body_q(state):
		i, dKj, dVj = state
		dKj, dVj = dKj.astype(jnp.float32), dVj.astype(jnp.float32)
		q_seq_mask = i * qblock + jnp.arange(0, qblock) < seqlen_q
		q_mask = q_seq_mask[:, None] & headdim_mask[None, :]

		q_idx = (bg, pl.dslice(i * qblock, qblock), nhg, slice(None))
		ld_idx = (bg, nhg, pl.dslice(i * qblock, qblock))

		qi = pl.load(q_ref, idx=q_idx, mask=q_mask, other=0.0).astype(jnp.float32)
		dOi = pl.load(
			dO_ref,
			idx=q_idx,
			mask=q_mask,
			other=0.0,
		).astype(jnp.float32)
		pQ = pl.load(
			dQ_ref,
			idx=q_idx,
			mask=q_mask,
			other=0.0,
			eviction_policy="evict_last",
		)

		qk = jnp.dot(qi, kj.transpose(1, 0).astype(qi.dtype)).astype(jnp.float32)
		qk = jnp.where(q_seq_mask[:, None] & kv_seq_mask[None, :], qk, float("-inf"))
		lse_i = pl.load(
			lse_ref,
			idx=ld_idx,
			mask=q_seq_mask,
			other=0,
		)[:, None].astype(jnp.float32)
		if b_ref is not None:
			qk = qk * softmax_scale + pl.load(
				b_ref,
				idx=(
					bg,
					nhg,
					pl.dslice(i * qblock, qblock),
					pl.dslice(j * kblock, kblock),
				),
				mask=q_seq_mask[:, None] & kv_seq_mask[None, :],
				other=0.0,
			).astype(jnp.float32)
			p = jnp.exp(qk - lse_i)
		else:
			p = jnp.exp(qk * softmax_scale - lse_i)
		dVj = dVj + jnp.dot(
			p.astype(jnp.float32).transpose(1, 0), dOi.astype(jnp.float32)
		)
		dP = jnp.dot(dOi.astype(jnp.float32), vj.transpose(1, 0).astype(jnp.float32))
		delta = pl.load(delta_ref, idx=ld_idx, mask=q_seq_mask, other=0.0)[..., None]
		dS = p * (dP - delta) * softmax_scale
		dQi = pQ + jnp.dot(dS, kj.astype(dS.dtype))
		dKj += jnp.dot(dS.transpose(1, 0), qi)
		pl.store(
			dQ_ref,
			idx=q_idx,
			val=dQi.astype(dQ_ref.dtype),
			mask=q_mask,
			eviction_policy="evict_last",
		)
		return (i + 1, dKj.astype(dK_ref.dtype), dVj.astype(dV_ref.dtype))

	_, dKj, dVj = jax.lax.while_loop(
		lambda state: state[0] < pl.cdiv(seqlen_q, qblock),
		body_q,
		(
			0,
			pl.load(
				dK_ref,
				idx=(bg, pl.dslice(j * kblock, kblock), nhg, slice(None)),
				mask=kv_seq_mask[:, None] & headdim_mask[None, :],
				other=0.0,
			).astype(dK_ref.dtype),
			pl.load(
				dV_ref,
				idx=(bg, pl.dslice(j * kblock, kblock), nhg, slice(None)),
				mask=kv_seq_mask[:, None] & headdim_mask[None, :],
				other=0.0,
			).astype(dV_ref.dtype),
		),
	)
	pl.store(
		dK_ref,
		idx=(bg, pl.dslice(j * kblock, kblock), nhg, slice(None)),
		val=dKj.astype(dK_ref),
		mask=kv_seq_mask[:, None] & headdim_mask[None, :],
	)
	pl.store(
		dV_ref,
		idx=(bg, pl.dslice(j * kblock, kblock), nhg, slice(None)),
		val=dVj.astype(dV_ref),
		mask=kv_seq_mask[:, None] & headdim_mask[None, :],
	)


def _call_gpu_bwd_flash_attn(
	dtype: jnp.dtype,
	qblock: int,
	kblock: int,
	softmax_scale: float,
	res,
	dO,
):
	(q, k, v, b, o, lse) = res
	batch_size, seqlen_q, nheads, dim = q.shape
	seqlen_k = k.shape[1]
	softmax_scale = softmax_scale or 1.0 / math.sqrt(dim)
	dtype = dtype or q.dtype

	bias_spec = None if b is None else pl.BlockSpec(b.shape, lambda *_: (0,) * b.ndim)
	in_specs = [
		pl.BlockSpec(q.shape, lambda *_: (0,) * q.ndim),
		pl.BlockSpec(k.shape, lambda *_: (0,) * k.ndim),
		pl.BlockSpec(v.shape, lambda *_: (0,) * v.ndim),
		bias_spec,
		pl.BlockSpec(lse.shape, lambda *_: (0,) * lse.ndim),  # DELTA.
		pl.BlockSpec(lse.shape, lambda *_: (0,) * lse.ndim),
		pl.BlockSpec(dO.shape, lambda *_: (0,) * dO.ndim),
		pl.BlockSpec(q.shape, lambda *_: (0,) * q.ndim),
	]

	out_shape = [
		jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
		jax.ShapeDtypeStruct(shape=k.shape, dtype=k.dtype),
		jax.ShapeDtypeStruct(shape=v.shape, dtype=v.dtype),
	]

	out_specs = [
		pl.BlockSpec(q.shape, lambda *_: (0,) * q.ndim),
		pl.BlockSpec(k.shape, lambda *_: (0,) * k.ndim),
		pl.BlockSpec(v.shape, lambda *_: (0,) * v.ndim),
	]

	delta = pl.pallas_call(
		partial(
			_gpu_bwd_do_o_dot,
			qblock=qblock,
			seqlen_q=seqlen_q,
			nheads=nheads,
			block_headdim=dim,
			headdim=dim,
		),
		grid=(pl.cdiv(seqlen_q, qblock), batch_size * nheads),
		compiler_params=dict(
			triton=dict(num_wraps=4 if dim <= 64 else 8, num_stages=1)
		),
		in_specs=[
			pl.BlockSpec(o.shape, lambda *_: (0,) * o.ndim),
			pl.BlockSpec(dO.shape, lambda *_: (0,) * dO.ndim),
		],
		out_specs=pl.BlockSpec(lse.shape, lambda *_: (0,) * lse.ndim),
		out_shape=jax.ShapeDtypeStruct(shape=lse.shape, dtype=o.dtype),
		interpret=_INTERPRET,
		debug=False,
		name="gpu_bwd_flash_attn",
	)(o, dO)

	method = pl.pallas_call(
		partial(
			_gpu_bwd_flash_attn_kernel,
			dtype=dtype,
			qblock=qblock,
			kblock=kblock,
			seqlen_q=seqlen_q,
			seqlen_k=seqlen_k,
			softmax_scale=softmax_scale,
			nheads=nheads,
			block_headdim=dim,
		),
		grid=(pl.cdiv(seqlen_k, kblock), batch_size * nheads),
		out_shape=out_shape,
		out_specs=out_specs,
		in_specs=in_specs,
		compiler_params=dict(
			triton=dict(num_wraps=4 if dim <= 64 else 8, num_stages=1)
		),
		interpret=True,  # TODO: DEBUG THIS
		input_output_aliases={6: 0},
		debug=False,
	)

	dq, dk, dv = method(
		q,
		k,
		v,
		b,
		delta,
		lse,
		dO,
		jnp.zeros_like(q),
	)
	return dq, dk, dv, None


@partial(custom_vjp, nondiff_argnums=(4, 5, 6, 7))
def _gpu_flash_attn(
	q: jax.Array,
	k: jax.Array,
	v: jax.Array,
	b: Optional[jax.Array] = None,
	dtype: jnp.dtype = None,
	qblock: int = 128,
	kblock: int = 128,
	softmax_scale: float = None,
):
	return _call_gpu_fwd_flash_attn(
		q,
		k,
		v,
		b,
		dtype,
		qblock,
		kblock,
		softmax_scale,
		True,
	)[0]


_gpu_flash_attn.defvjp(
	_call_gpu_fwd_flash_attn,
	_call_gpu_bwd_flash_attn,
	# _jax_bwd_flash_attn,
)


@jax.named_scope("ed_flash_attention_2")
def _flash_attn2(
	q: jax.Array,
	k: jax.Array,
	v: jax.Array,
	b: Optional[jax.Array] = None,
	dtype: jnp.dtype = None,
	qblock: int = 128,
	kblock: int = 128,
	softmax_scale: float = None,
):
	chex.assert_equal_shape(inputs=(k, v), dims=(0, 1, 2, 3))
	chex.assert_equal_rank([q, k, v])
	if b is not None:
		if q.shape[2] != b.shape[1]:
			raise AssertionError(f"nheads missmatched in q,b {q.shape=},{b.shape=}")
	if _PLATFORM in ["gpu", "cpu"]:
		return _gpu_flash_attn(
			q=q,
			k=k,
			v=v,
			b=b,
			dtype=dtype,
			qblock=qblock,
			kblock=kblock,
			softmax_scale=softmax_scale,
		)
	raise NotImplementedError(
		f"`_flash_attn2` is not implemented for requested platform {_PLATFORM}"
	)


def gpu_fwd_test():
	b, h, qs, s, d = 2, 32, 2048, 2048, 128
	dtype = _TEST_DTYPE

	q = jrand.normal(_rng.rng, shape=(b, qs, h, d), dtype=dtype)
	k = jrand.normal(_rng.rng, shape=(b, s, h, d), dtype=dtype)
	v = jrand.normal(_rng.rng, shape=(b, s, h, d), dtype=dtype)
	b = jnp.where(
		jrand.randint(_rng.rng, shape=(b, h, qs, s), minval=0, maxval=3) > 1,
		0,
		jnp.finfo(dtype).min,
	)

	excepted_result = _dot_product_attention(
		query=q,
		key=k,
		value=v,
		bias=b,
	)
	result = _flash_attn2(
		q=q,
		k=k,
		v=v,
		b=b,
		dtype=jnp.float16,
		qblock=32,
		kblock=32,
	)
	print(f"PRED : {result[-1,-1,-1,:5]}")
	print(f"ORGN : {excepted_result[-1,-1,-1,:5]}")

	print(jnp.allclose(excepted_result, result, atol=0.125, rtol=0))


def gpu_bwd_test():
	q_key, k_key, v_key = jrand.split(jrand.PRNGKey(8), 3)
	B, H, S, D = 1, 2, 16, 16
	blocksize_k = 16
	blocksize_q = 16
	q = jax.nn.initializers.normal(2)(q_key, (B, S, H, D), dtype=jnp.float32)
	k = jax.nn.initializers.normal(2)(k_key, (B, S, H, D), dtype=jnp.float32)
	v = jax.nn.initializers.normal(2)(v_key, (B, S, H, D), dtype=jnp.float32)
	b = (
		jnp.where(
			jrand.randint(v_key, (B, H, S, S), 0, 4) > 2,
			jnp.finfo(jnp.float32).min,
			0,
		)
		if False
		else None
	)
	dtype = _TEST_DTYPE

	excepted_result = jax.grad(lambda *x: _dot_product_attention(*x).sum())(q, k, v, b)
	result = jax.grad(
		lambda *x: _gpu_flash_attn(
			*x,
			dtype=dtype,
			qblock=blocksize_q,
			kblock=blocksize_k,
		).sum()
	)(q, k, v, b)
	print(f"PRED BWD : {result[-1,-1,-1,:5]}")
	print(f"ORGN BWD : {excepted_result[-1,-1,-1,:5]}")

	print(jnp.allclose(excepted_result, result, atol=0.125, rtol=0))


pallas_flash_attn_2_gpu = _flash_attn2
__all__ = ["pallas_flash_attn_2_gpu"]

if __name__ == "__main__":
	gpu_fwd_test()
	gpu_bwd_test()
