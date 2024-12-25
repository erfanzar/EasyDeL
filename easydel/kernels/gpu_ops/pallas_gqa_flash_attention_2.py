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
import typing as tp
from functools import partial

import chex
import jax
from jax import custom_vjp
from jax import numpy as jnp
from jax import random as jrand
from jax.experimental import pallas as pl


def _attn_refrence(query_states, key_states, value_states, bias):
	b, qs, num_q_heads, d = query_states.shape
	num_kv_heads = value_states.shape[2]
	ks = value_states.shape[1]
	query_states = jnp.reshape(
		query_states,
		(b, qs, num_kv_heads, num_q_heads // num_kv_heads, d),
	)

	query_states = query_states * (d**-0.5)
	attention_weight = jnp.einsum(
		"bskhd,bmkd->bkhsm",
		query_states,
		key_states,
	)

	if bias is not None:
		if bias.shape[1] == num_q_heads:
			attention_weight = jnp.add(
				attention_weight,
				bias.reshape(b, num_kv_heads, num_q_heads // num_kv_heads, qs, ks),
			)
		elif bias.shape[1] == num_kv_heads:
			attention_weight = jnp.add(
				attention_weight,
				bias.reshape(b, num_kv_heads, 1, qs, ks),
			)
		elif bias.shape[1] == 1:
			attention_weight = jnp.add(
				attention_weight,
				bias.reshape(b, 1, 1, qs, ks),
			)
		else:
			raise NotImplementedError("bias heads wont match!")

	attention_weight = jax.nn.softmax(attention_weight)

	return jnp.einsum(
		"bkhsm,bmkd->bskhd",
		attention_weight,
		value_states,
	).reshape(b, qs, num_q_heads, d)


@partial(
	jax.jit,
	static_argnames=[
		"dtype",
		"BLOCK_M",
		"BLOCK_N",
		"softmax_scale",
		"inference_mode",
		"interpret",
		"debug",
	],
)
def forward_flash_attention(
	query: jax.Array,
	key: jax.Array,
	value: jax.Array,
	bias: tp.Optional[jax.Array] = None,
	dtype: jnp.dtype = None,
	BLOCK_M: int = 128,
	BLOCK_N: int = 128,
	softmax_scale: float = None,
	inference_mode: bool = False,
	interpret=False,
	debug=False,
):
	batch, query_length, nheads, dim = query.shape
	batch, kv_length, kv_heads, dim = key.shape
	groups = nheads // kv_heads
	kv_length = key.shape[1]
	softmax_scale = softmax_scale or 1.0 / math.sqrt(dim)
	dtype = dtype or query.dtype
	BLOCK_HDIM = pl.next_power_of_2(dim)

	lse_shape = (batch, kv_heads * groups, query_length)
	in_specs = [
		pl.BlockSpec(query.shape, lambda *_: (0,) * query.ndim),
		pl.BlockSpec(key.shape, lambda *_: (0,) * key.ndim),
		pl.BlockSpec(value.shape, lambda *_: (0,) * value.ndim),
	]
	in_specs.append(
		None if bias is None else pl.BlockSpec(bias.shape, lambda *_: (0,) * bias.ndim)
	)
	bias_single_head = True if bias is not None and bias.shape[1] == 1 else False

	@partial(
		pl.pallas_call,
		grid=(pl.cdiv(query_length, BLOCK_M), batch * kv_heads, groups),
		out_specs=[
			pl.BlockSpec(query.shape, lambda *_: (0,) * query.ndim),
			pl.BlockSpec(lse_shape, lambda *_: (0,) * len(lse_shape)),
		],
		in_specs=in_specs,
		out_shape=[
			jax.ShapeDtypeStruct(shape=query.shape, dtype=dtype),
			jax.ShapeDtypeStruct(shape=lse_shape, dtype=jnp.float32),
		],
		compiler_params=dict(triton=dict(num_wraps=4 if dim <= 64 else 8, num_stages=1)),
		interpret=interpret,
		debug=debug,
		name=f"fwd_flash_attention_{inference_mode}",
	)
	def kernel(
		qref,
		kref,
		vref,
		bref,
		oref,
		lref,
	):
		off_qs, nhbg, off_g = pl.program_id(0), pl.program_id(1), pl.program_id(2) + 1
		bg, nhg = nhbg // nheads, nhbg % nheads

		M_SEQUENCE_MASK = off_qs * BLOCK_M + jnp.arange(0, BLOCK_M) < query_length
		HDIM_MASK = jnp.arange(0, BLOCK_HDIM) < dim
		QUERY_MASK = M_SEQUENCE_MASK[:, None] & HDIM_MASK[None, :]

		query = pl.load(
			qref,
			idx=(bg, pl.ds(off_qs * BLOCK_M, BLOCK_M), nhg * off_g, pl.ds(0, BLOCK_HDIM)),
			mask=QUERY_MASK,
			other=0.0,
		)

		def body(carry):
			idx, acc_o, max_i, lse_i = carry
			N_SEQUENCE_MASK = idx * BLOCK_N + jnp.arange(0, BLOCK_N) < kv_length
			KEY_MASK = N_SEQUENCE_MASK[:, None] & HDIM_MASK[None, :]
			key = pl.load(
				kref,
				idx=(bg, pl.ds(idx * BLOCK_N, BLOCK_N), nhg, pl.ds(0, BLOCK_HDIM)),
				mask=KEY_MASK,
				other=0.0,
			)
			qk = jnp.dot(query, key.transpose(1, 0)).astype(jnp.float32) * softmax_scale
			if kv_length % BLOCK_N != 0:
				qk = jnp.where(
					M_SEQUENCE_MASK[:, None] & N_SEQUENCE_MASK[None, :], qk, float("-inf")
				)
			if bref is not None:
				b = pl.load(
					bref,
					idx=(
						bg,
						0 if bias_single_head else nhg * off_g,
						pl.ds(off_qs * BLOCK_M, BLOCK_M),
						pl.ds(idx * BLOCK_N, BLOCK_N),
					),
					mask=M_SEQUENCE_MASK[:, None] & N_SEQUENCE_MASK[None, :],
					other=0.0,
				).astype(jnp.float32)
				qk = qk + b
			max_ij = jnp.maximum(jnp.max(qk, -1), lse_i)
			p = jnp.exp(qk - max_ij[:, None])
			lse_ij = jnp.sum(p, -1)
			acc_o_scale = jnp.exp(max_i - max_ij)
			acc_o = acc_o * acc_o_scale[:, None]
			value = pl.load(
				vref,
				idx=(bg, pl.ds(idx * BLOCK_N, BLOCK_N), nhg, pl.ds(0, BLOCK_HDIM)),
				mask=KEY_MASK,
				other=0.0,
			)

			acc_o += jnp.dot(p.astype(value.dtype), value).astype(acc_o.dtype)
			lin = jnp.exp(lse_i - max_ij) + lse_ij
			return (idx + 1, acc_o, max_ij, max_ij + jnp.log(lin))

		end_krange = pl.cdiv(kv_length, BLOCK_N)
		init_block = (
			0,
			jnp.zeros([BLOCK_M, BLOCK_HDIM], dtype=jnp.float32),
			jnp.full([BLOCK_M], -float("inf"), dtype=jnp.float32),
			jnp.full([BLOCK_M], -float("inf"), dtype=jnp.float32),
		)

		_, o, max_i, lse_i = jax.lax.while_loop(
			lambda state: state[0] < end_krange,
			body,
			init_block,
		)
		o_scale = jnp.exp(max_i - lse_i)
		o = o * o_scale[:, None]
		pl.store(
			oref,
			val=o.astype(dtype),
			idx=(bg, pl.dslice(off_qs * BLOCK_M, BLOCK_M), nhg * off_g, pl.ds(0, BLOCK_HDIM)),
			mask=QUERY_MASK,
		)
		if lref is not None:
			pl.store(
				lref,
				val=lse_i,
				idx=(bg, nhg * off_g, pl.dslice(off_qs * BLOCK_M, BLOCK_M)),
				mask=M_SEQUENCE_MASK,
			)

	out, lse = kernel(query, key, value, bias)
	if inference_mode:
		del lse
		lse = None
	return out, (
		query,
		key,
		value,
		bias,
		out,
		lse,
	)


def backward_flash_attention(
	dtype: jnp.dtype,
	BLOCK_M: int,
	BLOCK_N: int,
	softmax_scale: float,
	inference_mode: bool,
	interpret: bool,
	debug: bool,
	res,
	dO,
):
	raise NotImplementedError("Under Development! (use triton or sdpa).")
	(query, key, value, bias, o, lse) = res
	batch, query_length, nheads, dim = query.shape
	batch, kv_length, kv_heads, dim = key.shape
	groups = nheads // kv_heads
	softmax_scale = softmax_scale or 1.0 / math.sqrt(dim)
	dtype = dtype or query.dtype
	BLOCK_HDIM = pl.next_power_of_2(dim)
	bias_single_head = True if bias is not None and bias.shape[1] == 1 else False
	bias_spec = (
		None if bias is None else pl.BlockSpec(bias.shape, lambda *_: (0,) * bias.ndim)
	)

	@partial(
		pl.pallas_call,
		grid=(pl.cdiv(query_length, BLOCK_M), batch * kv_heads, groups),
		compiler_params=dict(triton=dict(num_wraps=4 if dim <= 64 else 8, num_stages=1)),
		in_specs=[
			pl.BlockSpec(o.shape, lambda *_: (0,) * o.ndim),
			pl.BlockSpec(dO.shape, lambda *_: (0,) * dO.ndim),
		],
		out_specs=pl.BlockSpec(lse.shape, lambda *_: (0,) * lse.ndim),
		out_shape=jax.ShapeDtypeStruct(shape=lse.shape, dtype=o.dtype),
		interpret=interpret,
		debug=debug,
		name="bwd_flash_attention_dO",
	)
	def call_dO(
		oref,
		dOref,
		dEref,
	):
		off_qs, bnhg, off_g = pl.program_id(0), pl.program_id(1), pl.program_id(2) + 1
		bg = bnhg // nheads
		nhg = bnhg % nheads

		M_SEQUENCE_MASK = off_qs * BLOCK_M + jnp.arange(0, BLOCK_M) < query_length
		HDIM_MASK = jnp.arange(0, BLOCK_HDIM) < dim
		OMASK = M_SEQUENCE_MASK[:, None] & HDIM_MASK[None, :]
		idx = (bg, pl.dslice(off_qs * BLOCK_M, BLOCK_M), nhg * off_g, slice(None))

		o = pl.load(oref, idx=idx, mask=OMASK, other=0.0).astype(jnp.float32)
		dO = pl.load(dOref, idx=idx, mask=OMASK, other=0.0).astype(jnp.float32)
		delta = jnp.sum(o * dO, -1)
		pl.store(
			dEref,
			idx=(bg, nhg * off_g, pl.dslice(off_qs * BLOCK_M, BLOCK_M)),
			val=delta.astype(dEref.dtype),
			mask=M_SEQUENCE_MASK,
		)

	delta = call_dO(o, dO)

	@partial(
		pl.pallas_call,
		grid=(pl.cdiv(kv_length, BLOCK_N), batch * kv_heads, groups),
		out_shape=[
			jax.ShapeDtypeStruct(shape=query.shape, dtype=query.dtype),
			jax.ShapeDtypeStruct(shape=key.shape, dtype=key.dtype),
			jax.ShapeDtypeStruct(shape=value.shape, dtype=value.dtype),
		],
		out_specs=[
			pl.BlockSpec(query.shape, lambda *_: (0,) * query.ndim),
			pl.BlockSpec(key.shape, lambda *_: (0,) * key.ndim),
			pl.BlockSpec(value.shape, lambda *_: (0,) * value.ndim),
		],
		in_specs=[
			pl.BlockSpec(query.shape, lambda *_: (0,) * query.ndim),
			pl.BlockSpec(key.shape, lambda *_: (0,) * key.ndim),
			pl.BlockSpec(value.shape, lambda *_: (0,) * value.ndim),
			bias_spec,
			pl.BlockSpec(lse.shape, lambda *_: (0,) * lse.ndim),  # delta.
			pl.BlockSpec(lse.shape, lambda *_: (0,) * lse.ndim),
			pl.BlockSpec(dO.shape, lambda *_: (0,) * dO.ndim),
			pl.BlockSpec(query.shape, lambda *_: (0,) * query.ndim),
		],
		compiler_params=dict(triton=dict(num_wraps=4 if dim <= 64 else 8, num_stages=1)),
		interpret=interpret,
		input_output_aliases={7: 0},
		debug=debug,
	)
	def call_backward(
		qref,
		kref,
		vref,
		bref,
		delta_ref,
		lref,
		dO_ref,
		_,
		dQ_ref,
		dK_ref,
		dV_ref,
	):
		j, bnhg, off_g = pl.program_id(0), pl.program_id(1), pl.program_id(2) + 1

		bg = bnhg // nheads
		nhg = bnhg % nheads

		HDIM_MASK = jnp.arange(0, BLOCK_HDIM) < kref.shape[3]

		N_SEQUENCE_MASK = j * BLOCK_N + jnp.arange(0, BLOCK_N) < kv_length

		kj = pl.load(
			kref,
			idx=(bg, pl.dslice(j * BLOCK_N, BLOCK_N), nhg, slice(None)),
			mask=N_SEQUENCE_MASK[:, None] & HDIM_MASK[None, :],
			other=0.0,
		)
		vj = pl.load(
			vref,
			idx=(bg, pl.dslice(j * BLOCK_N, BLOCK_N), nhg, slice(None)),
			mask=N_SEQUENCE_MASK[:, None] & HDIM_MASK[None, :],
			other=0.0,
		)

		def body_q(state):
			i, dKj, dVj = state
			dKj, dVj = dKj.astype(jnp.float32), dVj.astype(jnp.float32)
			M_SEQUENCE_MASK = i * BLOCK_M + jnp.arange(0, BLOCK_M) < query_length
			q_mask = M_SEQUENCE_MASK[:, None] & HDIM_MASK[None, :]

			q_idx = (bg, pl.dslice(i * BLOCK_M, BLOCK_M), nhg * off_g, slice(None))
			ld_idx = (bg, nhg * off_g, pl.dslice(i * BLOCK_M, BLOCK_M))

			qi = pl.load(qref, idx=q_idx, mask=q_mask, other=0.0).astype(jnp.float32)
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
			if kv_length % BLOCK_N != 0:
				qk = jnp.where(
					M_SEQUENCE_MASK[:, None] & N_SEQUENCE_MASK[None, :],
					qk,
					float("-inf"),
				)
			lse_i = pl.load(
				lref,
				idx=ld_idx,
				mask=M_SEQUENCE_MASK,
				other=0,
			)[:, None].astype(jnp.float32)
			if bref is not None:
				qk = qk * softmax_scale + pl.load(
					bref,
					idx=(
						bg,
						0 if bias_single_head else nhg,
						pl.dslice(i * BLOCK_M, BLOCK_M),
						pl.dslice(j * BLOCK_N, BLOCK_N),
					),
					mask=M_SEQUENCE_MASK[:, None] & N_SEQUENCE_MASK[None, :],
					other=0.0,
				).astype(jnp.float32)
				p = jnp.exp(qk - lse_i)
			else:
				p = jnp.exp(qk * softmax_scale - lse_i)
			dVj = dVj + jnp.dot(
				p.astype(jnp.float32).transpose(1, 0), dOi.astype(jnp.float32)
			)
			dP = jnp.dot(dOi.astype(jnp.float32), vj.transpose(1, 0).astype(jnp.float32))
			delta = pl.load(delta_ref, idx=ld_idx, mask=M_SEQUENCE_MASK, other=0.0)[..., None]
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
			lambda state: state[0] < pl.cdiv(query_length, BLOCK_M),
			body_q,
			(
				0,
				pl.load(
					dK_ref,
					idx=(bg, pl.dslice(j * BLOCK_N, BLOCK_N), nhg, slice(None)),
					mask=N_SEQUENCE_MASK[:, None] & HDIM_MASK[None, :],
					other=0.0,
				).astype(dK_ref.dtype),
				pl.load(
					dV_ref,
					idx=(bg, pl.dslice(j * BLOCK_N, BLOCK_N), nhg, slice(None)),
					mask=N_SEQUENCE_MASK[:, None] & HDIM_MASK[None, :],
					other=0.0,
				).astype(dV_ref.dtype),
			),
		)
		pl.store(
			dK_ref,
			idx=(bg, pl.dslice(j * BLOCK_N, BLOCK_N), nhg, slice(None)),
			val=dKj.astype(dK_ref),
			mask=N_SEQUENCE_MASK[:, None] & HDIM_MASK[None, :],
		)
		pl.store(
			dV_ref,
			idx=(bg, pl.dslice(j * BLOCK_N, BLOCK_N), nhg, slice(None)),
			val=dVj.astype(dV_ref),
			mask=N_SEQUENCE_MASK[:, None] & HDIM_MASK[None, :],
		)

	dq, dk, dv = call_backward(
		query,  # qref,
		key,  # kref,
		value,  # vref,
		bias,  # bref,
		delta,  # delta_ref,
		lse,  # lref,
		dO,  # dO_ref,
		jnp.zeros_like(query),  # dQ_ref,
	)
	return dq, dk, dv, None


@partial(custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8, 9, 10))
def _flash_attention2_hook(
	query: jax.Array,
	key: jax.Array,
	value: jax.Array,
	bias: tp.Optional[jax.Array] = None,
	dtype: jnp.dtype = None,
	BLOCK_M: int = 128,
	BLOCK_N: int = 128,
	softmax_scale: float = None,
	inference_mode: bool = False,
	interpret=False,
	debug=False,
):
	return forward_flash_attention(
		query=query,
		key=key,
		value=value,
		bias=bias,
		dtype=dtype,
		BLOCK_M=BLOCK_M,
		BLOCK_N=BLOCK_N,
		softmax_scale=softmax_scale,
		inference_mode=inference_mode,
		interpret=interpret,
		debug=debug,
	)[0]


_flash_attention2_hook.defvjp(
	forward_flash_attention,
	backward_flash_attention,
)


@jax.named_scope("easydel_pallas_flash_attention_2")
def flash_attention2(
	query: jax.Array,
	key: jax.Array,
	value: jax.Array,
	bias: tp.Optional[jax.Array] = None,
	dtype: jnp.dtype = None,
	BLOCK_M: int = 128,
	BLOCK_N: int = 128,
	softmax_scale: float = None,
	inference_mode: bool = False,
	interpret=False,
	debug=False,
):
	chex.assert_equal_shape(inputs=(key, value), dims=(0, 1, 2, 3))
	chex.assert_equal_rank([query, key, value])
	return _flash_attention2_hook(
		query=query,
		key=key,
		value=value,
		bias=bias,
		dtype=dtype,
		BLOCK_M=BLOCK_M,
		BLOCK_N=BLOCK_N,
		softmax_scale=softmax_scale,
		inference_mode=inference_mode,
		interpret=interpret,
		debug=debug,
	)


def _get_dummy_inputs(B, H, Kh, Qs, S, D, dtype=jnp.float16):
	query = jrand.normal(jrand.key(0), shape=(B, Qs, H, D), dtype=dtype)
	key = jrand.normal(jrand.key(1), shape=(B, S, Kh, D), dtype=dtype)
	value = jrand.normal(jrand.key(2), shape=(B, S, Kh, D), dtype=dtype)
	minim = jnp.finfo(dtype).min
	pos = jrand.randint(jrand.key(3), shape=(B, 1, Qs, S), minval=0, maxval=3) > 1
	bias = jnp.where(pos, 0, minim)
	return query, key, value, bias


def _forward_test():
	B, H, Kh, Qs, S, D = 1, 32, 4, 1024, 1024, 64
	dtype = jnp.float16
	BLOCK_M = 32
	BLOCK_N = 32
	query, key, value, bias = _get_dummy_inputs(B, H, Kh, Qs, S, D, dtype)
	excepted_result = _attn_refrence(query, key, value, bias)

	result = flash_attention2(
		query=query,
		key=key,
		value=value,
		bias=bias,
		dtype=jnp.float16,
		BLOCK_M=BLOCK_M,
		BLOCK_N=BLOCK_N,
	)
	print(f"PRED : {result[-1,-1,-1,:5]}")
	print(f"ORGN : {excepted_result[-1,-1,-1,:5]}")

	print(jnp.allclose(excepted_result, result, atol=0.125, rtol=0))


def _backward_test():
	B, H, Kh, Qs, S, D = 1, 32, 4, 1024, 1024, 64
	dtype = jnp.float16
	BLOCK_M = 32
	BLOCK_N = 32
	query, key, value, bias = _get_dummy_inputs(B, H, Kh, Qs, S, D, dtype)

	excepted_result = jax.grad(lambda *x: _attn_refrence(*x).sum())(
		query, key, value, bias
	)
	result = jax.grad(
		lambda *x: flash_attention2(
			*x,
			dtype=dtype,
			BLOCK_M=BLOCK_M,
			BLOCK_N=BLOCK_N,
		).sum()
	)(query, key, value, bias)
	print(f"PRED BWD : {result[-1,-1,-1,:5]}")
	print(f"ORGN BWD : {excepted_result[-1,-1,-1,:5]}")

	print(jnp.allclose(excepted_result, result, atol=0.125, rtol=0))


pallas_gqa_flash_attention2_gpu = flash_attention2
__all__ = ["pallas_gqa_flash_attention2_gpu"]

if __name__ == "__main__":
	_forward_test()
	# _backward_test()
