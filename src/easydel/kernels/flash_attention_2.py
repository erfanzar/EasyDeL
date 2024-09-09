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

import jax
from fjformer import GenerateRNG
from flax.linen.attention import dot_product_attention as _dot_product_attention
from jax import custom_vjp
from jax import numpy as jnp
from jax import random as jrand
from jax.experimental import pallas as pl
from jax.lib import xla_bridge

PLATFORM = xla_bridge.get_backend().platform
INTERPRET = PLATFORM == "cpu"
# INTERPRET = True
rng = GenerateRNG()


def _gpu_fwd_flash_attn_kernel(
	q_ref,
	k_ref,
	v_ref,
	b_ref,
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
		idx=(bg, pl.dslice(qg * qblock, qblock), nhg, pl.dslice(None)),
		mask=qmask,
		other=0.0,
	)

	def body(carry):
		idx, acc_o, max_i, lse_i = carry
		kv_seq_mask = idx * kblock + jnp.arange(0, kblock) < seqlen_k
		key_mask = kv_seq_mask[:, None] & headdim_mask[None, :]
		k = pl.load(
			k_ref,
			idx=(bg, pl.dslice(idx * kblock, kblock), nhg, pl.dslice(None)),
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
			idx=(bg, pl.dslice(idx * kblock, kblock), nhg, pl.dslice(None)),
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
		idx=(bg, pl.dslice(qg * qblock, qblock), nhg, pl.dslice(None)),
		mask=qmask,
	)

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
	dtype: jnp.dtype = jnp.float32,
	qblock: int = 128,
	kblock: int = 128,
	softmax_scale: float = None,
):
	batch_size, seqlen_q, nheads, dim = q.shape
	seqlen_k = k.shape[1]
	softmax_scale = softmax_scale or 1.0 / math.sqrt(dim)

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
		jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
		jax.ShapeDtypeStruct(shape=lse_shape, dtype=jnp.float32),
	]

	out_specs = [
		pl.BlockSpec(q.shape, lambda *_: (0,) * q.ndim),
		pl.BlockSpec(lse_shape, lambda *_: (0,) * len(lse_shape)),
	]
	grid = (pl.cdiv(seqlen_q, qblock), batch_size * nheads)

	method = pl.pallas_call(
		f=partial(
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
		compiler_params=dict(triton=dict(num_wraps=4 if dim <= 64 else 8, num_stages=1)),
		interpret=INTERPRET,
		debug=False,
		name="gpu_fwd_flash_attn",
	)

	o, lse = method(q, k, v, b)

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
	idx = (bg, pl.dslice(qg * qblock, qblock), nhg, pl.dslice(None))

	o = pl.load(o_ref, idx=idx, mask=omask, other=0.0).astype(jnp.float32)
	dO = pl.load(dO_ref, idx=idx, mask=omask, other=0.0).astype(jnp.float32)
	delta = jnp.sum(o * dO, -1)
	pl.store(
		delta_ref,
		idx=(bg, nhg, pl.dslice(qg * qblock, qblock)),
		val=delta,
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
	dQ_ref,
	dK_ref,
	dV_ref,
	*,
	dtype,
	qblock,
	kblock,
	seqlen_q,
	seqlen_k,
	# seqlen_q_rounded,
	softmax_scale,
	nheads,
	block_headdim,
):
	j, bnhg = pl.program_id(0), pl.program_id(1)

	@pl.when(j == 0)
	def _():
		@pl.when(bnhg == 0)
		def _():
			dQ_ref[...] = jnp.zeros_like(dQ_ref)
			dK_ref[...] = jnp.zeros_like(dK_ref)
			dV_ref[...] = jnp.zeros_like(dV_ref)

	bg = bnhg // nheads
	nhg = bnhg % nheads

	headdim_mask = jnp.arange(0, block_headdim) < k_ref.shape[3]

	kv_seq_mask = j * kblock + jnp.arange(0, kblock) < seqlen_k

	kj = pl.load(
		k_ref,
		idx=(bg, pl.dslice(j * kblock, kblock), nhg, pl.dslice(None)),
		mask=kv_seq_mask[:, None] & headdim_mask[None, :],
		other=0.0,
	)
	vj = pl.load(
		v_ref,
		idx=(bg, pl.dslice(j * kblock, kblock), nhg, pl.dslice(None)),
		mask=kv_seq_mask[:, None] & headdim_mask[None, :],
		other=0.0,
	)

	def body_q(state):
		i, dKj, dVj = state

		q_seq_mask = i * qblock + jnp.arange(0, qblock) < seqlen_q
		q_mask = q_seq_mask[:, None] & headdim_mask[None, :]

		q_idx = (bg, pl.dslice(i * qblock, qblock), nhg, pl.dslice(None))
		ld_idx = (bg, nhg, pl.dslice(i * qblock, qblock))

		qi = pl.load(q_ref, idx=q_idx, mask=q_mask, other=0.0)
		dOi = pl.load(dO_ref, idx=q_idx, mask=q_mask, other=0.0)

		qk = jnp.dot(qi, kj.transpose(1, 0))
		qk = jnp.where(q_seq_mask[:, None] & kv_seq_mask[None, :], qk, float("-inf"))
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
			p = jnp.exp(qk - pl.load(lse_ref, idx=ld_idx, mask=q_seq_mask, other=0.0))
		else:
			p = jnp.exp(
				qk * softmax_scale - pl.load(lse_ref, idx=ld_idx, mask=q_seq_mask, other=0.0)
			)
		dVj += jnp.dot(p.transpose(1, 0), dOi)
		dP = jnp.dot(dOi, vj.transpose(1, 0))
		dS = (
			p
			* (dP - pl.load(delta_ref, idx=ld_idx, mask=q_seq_mask, other=0.0))
			* softmax_scale
		)
		dQi = pl.load(dQ_ref, idx=q_idx, mask=q_mask, other=0.0) + jnp.dot(dS, kj)
		dKj += jnp.dot(dS.transpose(1, 0), qi)
		pl.store(dQ_ref, idx=q_idx, val=dQi, mask=q_mask)
		return (i + 1, dKj, dVj)

	_, dKj, dVj = jax.lax.while_loop(
		lambda state: state[0] < pl.cdiv(seqlen_q, qblock),
		body_q,
		(
			0,
			pl.load(
				dK_ref,
				idx=(bg, pl.dslice(j * kblock, kblock), nhg, pl.dslice(None)),
				mask=kv_seq_mask[:, None] & headdim_mask[None, :],
				other=0.0,
			),
			pl.load(
				dV_ref,
				idx=(bg, pl.dslice(j * kblock, kblock), nhg, pl.dslice(None)),
				mask=kv_seq_mask[:, None] & headdim_mask[None, :],
				other=0.0,
			),
		),
	)
	pl.store(
		dK_ref,
		idx=(bg, pl.dslice(j * kblock, kblock), nhg, pl.dslice(None)),
		val=dKj,
		mask=kv_seq_mask[:, None] & headdim_mask[None, :],
	)
	pl.store(
		dV_ref,
		idx=(bg, pl.dslice(j * kblock, kblock), nhg, pl.dslice(None)),
		val=dVj,
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

	bias_spec = None if b is None else pl.BlockSpec(b.shape, lambda *_: (0,) * b.ndim)
	in_specs = [
		pl.BlockSpec(q.shape, lambda *_: (0,) * q.ndim),
		pl.BlockSpec(k.shape, lambda *_: (0,) * k.ndim),
		pl.BlockSpec(v.shape, lambda *_: (0,) * v.ndim),
		bias_spec,
		pl.BlockSpec(lse.shape, lambda *_: (0,) * lse.ndim),  # DELTA.
		pl.BlockSpec(lse.shape, lambda *_: (0,) * lse.ndim),
		pl.BlockSpec(dO.shape, lambda *_: (0,) * dO.ndim),
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
		f=partial(
			_gpu_bwd_do_o_dot,
			qblock=qblock,
			seqlen_q=seqlen_q,
			nheads=nheads,
			block_headdim=dim,
			headdim=dim,
		),
		grid=(pl.cdiv(seqlen_q, qblock), batch_size * nheads),
		compiler_params=dict(triton=dict(num_wraps=4 if dim <= 64 else 8, num_stages=1)),
		in_specs=[
			pl.BlockSpec(o.shape, lambda *_: (0,) * o.ndim),
			pl.BlockSpec(dO.shape, lambda *_: (0,) * dO.ndim),
		],
		out_specs=pl.BlockSpec(lse.shape, lambda *_: (0,) * lse.ndim),
		out_shape=jax.ShapeDtypeStruct(shape=lse.shape, dtype=o.dtype),
		interpret=INTERPRET,
		debug=False,
		name="gpu_bwd_flash_attn",
	)(o, dO)

	method = pl.pallas_call(
		f=partial(
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
		compiler_params=dict(triton=dict(num_wraps=4 if dim <= 64 else 8, num_stages=1)),
		interpret=INTERPRET,
		debug=False,
	)
	dq, dk, dv = method(q, k, v, b, delta, lse, dO)
	return dq, dk, dv, None


@partial(custom_vjp, nondiff_argnums=(4, 5, 6, 7))
def _gpu_flash_attn(
	q: jax.Array,
	k: jax.Array,
	v: jax.Array,
	b: Optional[jax.Array] = None,
	dtype: jnp.dtype = jnp.float32,
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
	)[0]


_gpu_flash_attn.defvjp(_call_gpu_fwd_flash_attn, _call_gpu_bwd_flash_attn)


def flash_attn2(
	q: jax.Array,
	k: jax.Array,
	v: jax.Array,
	b: Optional[jax.Array] = None,
	dtype: jnp.dtype = jnp.float32,
	qblock: int = 128,
	kblock: int = 128,
	softmax_scale: float = None,
):
	if PLATFORM in ["gpu", "cpu"]:
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
		f"`flash_attn2` is not implemented for requested platform {PLATFORM}"
	)


def gpu_fwd_test():
	b, h, qs, s, d = 1, 8, 1, 64, 128
	dtype = jnp.float16

	q = jrand.normal(rng.rng, shape=(b, qs, h, d), dtype=dtype)
	k = jrand.normal(rng.rng, shape=(b, s, h, d), dtype=dtype)
	v = jrand.normal(rng.rng, shape=(b, s, h, d), dtype=dtype)
	b = jnp.where(
		jrand.randint(rng.rng, shape=(b, h, qs, s), minval=0, maxval=3) > 1,
		0,
		jnp.finfo(dtype).min,
	)

	excepted_result = _dot_product_attention(
		query=q,
		key=k,
		value=v,
		bias=b,
	)
	result = _gpu_flash_attn(
		q=q,
		k=k,
		v=v,
		b=b,
		dtype=dtype,
		qblock=64,
		kblock=64,
	)
	print(f"PRED : {result[0,0,0,:5]}")
	print(f"ORGN : {excepted_result[0,0,0,:5]}")

	print(jnp.allclose(excepted_result, result, atol=0.125, rtol=0))


def gpu_bwd_test():
	b, h, qs, s, d = 1, 1, 16, 16, 16
	dtype = jnp.float32

	q = jrand.normal(rng.rng, shape=(b, qs, h, d), dtype=dtype)
	k = jrand.normal(rng.rng, shape=(b, s, h, d), dtype=dtype)
	v = jrand.normal(rng.rng, shape=(b, s, h, d), dtype=dtype)
	b = jnp.where(
		jrand.randint(rng.rng, shape=(b, h, qs, s), minval=0, maxval=3) > 1,
		0,
		jnp.finfo(dtype).min,
	)

	excepted_result = jax.grad(lambda *x: _dot_product_attention(*x).sum())(
		q,
		k,
		v,
	)
	result = jax.grad(
		lambda *x: _gpu_flash_attn(
			*x,
			dtype=dtype,
			qblock=64,
			kblock=64,
		).sum()
	)(
		q,
		k,
		v,
	)
	print(f"PRED BWD : {result[0,0,0,:5]}")
	print(f"ORGN BWD : {excepted_result[0,0,0,:5]}")

	print(jnp.allclose(excepted_result, result, atol=0.125, rtol=0))


if __name__ == "__main__":
	gpu_fwd_test()
	# gpu_bwd_test()
