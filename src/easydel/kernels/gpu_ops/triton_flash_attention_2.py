import functools
import math
from typing import Optional

import chex
import flax
import flax.linen
import flax.linen.attention
import jax
import triton
from fjformer.jax_triton import triton_call
from jax import custom_vjp
from jax import numpy as jnp
from jax import random as jrnd
from triton import language as tl
import numpy as np


def strides_from_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
	size = np.prod(shape)
	strides = []
	for s in shape:
		size = int(size // s)
		if size != 1:
			strides.append(size)
	if len(strides) == 0:
		strides.append(1)
	return tuple(strides)


# fmt:off
@triton.heuristics(
	{
		"EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
		"EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
		"EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
	}
)
@triton.jit
def _fwd_attn_kernel(
	Q, K, V, B, O, L,
	softmax_scale,
	stride_qb, stride_qh, stride_qm,
	stride_kb, stride_kh, stride_kn,
	stride_vb, stride_vh, stride_vn,
	stride_bb, stride_bh, stride_bm,
	stride_ob, stride_oh, stride_om,
	nheads, headdim,
	seqlen_q, seqlen_k, seqlen_q_rounded,
	HAVE_BIAS: tl.constexpr, BLOCK_HEADDIM: tl.constexpr,
	EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr,
	BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
): 
	off_bm, off_hb = tl.program_id(0), tl.program_id(1)
	off_b = off_hb // nheads
	off_h = off_hb % nheads
	offs_m = off_bm * BLOCK_M + tl.arange(0, BLOCK_M)
	offs_n = tl.arange(0, BLOCK_N)
	offs_d = tl.arange(0, BLOCK_HEADDIM)

	q_ptrs = Q + (off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :]))
	k_ptrs = K + (off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :]))
	v_ptrs = V + (off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :]))

	softmax_scale = softmax_scale.to(tl.float32)
	if HAVE_BIAS:
		b_ptrs = B + (off_b * stride_bb + off_h * stride_bh + (offs_m[:, None] * stride_bm + offs_n[None, :]))

	lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
	max_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
	acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

	if EVEN_N & EVEN_M:
		if EVEN_HEADDIM:
			q = tl.load(q_ptrs)
		else:
			q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
	else:
		if EVEN_HEADDIM:
			q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
		else:
			q = tl.load(q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
	for j in range(0, seqlen_k, BLOCK_N):
		j = tl.multiple_of(j, BLOCK_N)
		if EVEN_N & EVEN_M:
			if EVEN_HEADDIM:
				k = tl.load(k_ptrs + j * stride_kn)
			else:
				k = tl.load(k_ptrs + j * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
		else:
			if EVEN_HEADDIM:
				k = tl.load(k_ptrs + j * stride_kn, mask=(j + offs_n)[:, None] < seqlen_k, other=0.0)
			else:
				k = tl.load(k_ptrs + j * stride_kn, mask=((j + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
		qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
		qk += tl.dot(q, k.T)
		if not EVEN_N:
			qk += tl.where((j + offs_n)[None, :] < seqlen_k, 0, float("-inf")).to(tl.float32)
		if HAVE_BIAS:
			if EVEN_N & EVEN_M:
				b = tl.load(b_ptrs + j).to(tl.float32)
			else:
				b = tl.load(b_ptrs + j, mask=(offs_m[:, None] < seqlen_q) & (j + offs_n)[None, :] < seqlen_k, other=0.0).to(tl.float32)
			qk = (qk * softmax_scale) + b
			max_ij = tl.maximum(tl.max(qk, 1), lse_i)
			p = tl.exp(qk - max_ij[:, None])
		else:
			max_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
			p = tl.exp(qk * softmax_scale - max_ij[:, None])

		l_ij = tl.sum(p, 1)
		acc_o_scale = tl.exp(max_i - max_ij)
		acc_o = acc_o * acc_o_scale[:, None]
		if EVEN_M & EVEN_N:
			if EVEN_HEADDIM:
				v = tl.load(v_ptrs + j * stride_vn)
			else:
				v = tl.load(v_ptrs + j * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
		else:
			if EVEN_HEADDIM:
				v = tl.load(v_ptrs + j * stride_vn,	mask=(j + offs_n)[:, None] < seqlen_k, other=0.0)
			else:
				v = tl.load(v_ptrs + j * stride_vn, mask=((j + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
		acc_o += tl.dot(p.to(v.dtype), v)
		max_i = max_ij
		lin = tl.exp(lse_i - max_ij) + l_ij
		lse_i = max_ij + tl.log(lin)
	o_scale = tl.exp(max_i - lse_i)
	acc_o = acc_o * o_scale[:, None]
	lse_ptrs = L + off_hb * seqlen_q_rounded + offs_m
	tl.store(lse_ptrs, lse_i)

	out_ptrs = O + (off_b * stride_ob + off_h * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :]))
	if EVEN_M:
		if EVEN_HEADDIM:
			tl.store(out_ptrs, acc_o)
		else:
			tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
	else:
		if EVEN_HEADDIM:
			tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
		else:
			tl.store(out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim))
# fmt:on


def _fwd_attn_kernel_call(
	query: Optional[chex.Array],
	key: Optional[chex.Array],
	value: Optional[chex.Array],
	bias: Optional[chex.Array] = None,
	softmax_scale: Optional[float] = None,
	blocksize_q: int = 128,
	blocksize_k: int = 128,
):
	batch, seqlen_q, nheads, headdim = query.shape
	_, seqlen_k, _, _ = key.shape
	assert key.shape == (
		batch,
		seqlen_k,
		nheads,
		headdim,
	), "shape missmatch between key, value."
	assert value.shape == (
		batch,
		seqlen_k,
		nheads,
		headdim,
	), "shape missmatch between key, value."
	assert headdim in {16, 32, 64, 128, 256}, "given headdim is not supported."
	assert query.dtype == key.dtype == value.dtype, "tensors must have the same dtype."
	assert query.dtype in [jnp.float16, jnp.bfloat16], "only support fp16 and bf16."
	softmax_scale = softmax_scale or 1.0 / math.sqrt(headdim)
	HAVE_BIAS = True if bias is not None else False
	BLOCK_HEADDIM = max(triton.next_power_of_2(headdim), 16)
	stride_bb, stride_bh, stride_bm = (
		strides_from_shape(bias.shape) if HAVE_BIAS else (0, 0, 0)
	)
	stride_qb, stride_qh, stride_qm = strides_from_shape(query.shape)
	stride_kb, stride_kh, stride_kn = strides_from_shape(key.shape)
	stride_vb, stride_vh, stride_vn = strides_from_shape(value.shape)
	seqlen_q_rounded = math.ceil(seqlen_q / 64) * 64
	out = jnp.zeros_like(query, device=query.sharding)
	lse = jnp.zeros((batch, nheads, seqlen_q_rounded), dtype=jnp.float32)
	metaparams = dict(
		HAVE_BIAS=HAVE_BIAS,
		BLOCK_HEADDIM=BLOCK_HEADDIM,
		BLOCK_M=blocksize_q,
		BLOCK_N=blocksize_k,
	)
	# fmt:off
	num_warps = 4 if headdim <= 64 else 8
	return triton_call(
		query, key, value, bias, out, lse,
		softmax_scale,   
		stride_qb, stride_qh, stride_qm,
		stride_kb, stride_kh, stride_kn,
		stride_vb, stride_vh, stride_vn,
		stride_bb, stride_bh, stride_bm,
		stride_qb, stride_qh, stride_qm,
		nheads, headdim,  
		seqlen_q, seqlen_k, seqlen_q_rounded,
		kernel=_fwd_attn_kernel,
		out_shape=[
			jax.ShapeDtypeStruct(query.shape, query.dtype, sharding=query.sharding),  # O,
			jax.ShapeDtypeStruct((batch, nheads, seqlen_q_rounded), jnp.float32),  # L,
		],
		grid=lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads),
		input_output_aliases={4: 0, 5: 1},
		num_warps=num_warps,
		num_stages=1,
		**metaparams
	)
	# fmt:on


# fmt:off
@triton.jit
def _bwd_store_dk_dv(
	dk_ptrs, dv_ptrs,
	dk, dv,
	offs_n, offs_d,
	seqlen_k, headdim,
	EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr,
):
	if EVEN_N & EVEN_M:
		if EVEN_HEADDIM:
			tl.store(dv_ptrs, dv)
			tl.store(dk_ptrs, dk)
		else:
			tl.store(dv_ptrs, dv, mask=offs_d[None, :] < headdim)
			tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
	else:
		if EVEN_HEADDIM:
			tl.store(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
			tl.store(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
		else:
			tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
			tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))

# fmt:on


# fmt:off
@triton.jit
def _bwd_do_attn_kernel(
	O, Do, De,
	stride_ob, stride_om, stride_oh,
	stride_dob, stride_dom, stride_doh, 
	nheads, headdim,
	seqlen_q, seqlen_q_rounded,
	BLOCK_M: tl.constexpr, BLOCK_HEADDIM: tl.constexpr,
):
# fmt:on
	off_q = tl.program_id(0)
	off_hb = tl.program_id(1)
	off_b = off_hb // nheads
	off_h = off_hb % nheads
	offs_m = off_q * BLOCK_M + tl.arange(0, BLOCK_M)
	offs_d = tl.arange(0, BLOCK_HEADDIM)
	o_ptrs = O + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :]
	do_ptrs = Do + off_b * stride_dob + off_h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :]
	o = tl.load(o_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0).to(tl.float32)
	do = tl.load(do_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0).to(tl.float32)
	delta = tl.sum(o * do, axis=1)
	tl.store(De + off_hb * seqlen_q_rounded + offs_m, delta)



# fmt:off
@triton.jit
def _bwd_kernel_one_col_block(
	start_n,
	Q, K, V, B,
	Do, Dq, Dk, Dv,
	L, D,
	softmax_scale,
	stride_qm, stride_kn, stride_vn, stride_bm, 
	stride_dom,
	stride_dqm, stride_dkn, stride_dvn,
	seqlen_q, seqlen_k,
	headdim,
	ATOMIC_ADD: tl.constexpr,
	HAVE_BIAS: tl.constexpr,
	BLOCK_HEADDIM: tl.constexpr,
	EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr,
	BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
	begin_m = 0
	offs_qm = begin_m + tl.arange(0, BLOCK_M)
	offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
	offs_m = tl.arange(0, BLOCK_M)
	offs_d = tl.arange(0, BLOCK_HEADDIM)
	q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_d[None, :])
	k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
	v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])
	do_ptrs = Do + (offs_qm[:, None] * stride_dom + offs_d[None, :])
	dq_ptrs = Dq + (offs_qm[:, None] * stride_dqm + offs_d[None, :])
	if HAVE_BIAS:
		b_ptrs = B + (offs_qm[:, None] * stride_bm + offs_n[None, :]) 
	dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
	dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32) 
	if begin_m >= seqlen_q:
		dv_ptrs = Dv + (offs_n[:, None] * stride_dvn + offs_d[None, :])
		dk_ptrs = Dk + (offs_n[:, None] * stride_dkn + offs_d[None, :])
		_bwd_store_dk_dv(
			dk_ptrs,
			dv_ptrs,
			dk,
			dv,
			offs_n,
			offs_d,
			seqlen_k,
			headdim,
			EVEN_M=EVEN_M,
			EVEN_N=EVEN_N,
			EVEN_HEADDIM=EVEN_HEADDIM,
		)
		return 
	if EVEN_N & EVEN_M:
		if EVEN_HEADDIM:
			k = tl.load(k_ptrs)
			v = tl.load(v_ptrs)
		else:
			k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
			v = tl.load(v_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
	else:
		if EVEN_HEADDIM:
			k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
			v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
		else:
			k = tl.load(k_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
			v = tl.load(v_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
			
	num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
	for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
		start_m = tl.multiple_of(start_m, BLOCK_M)
		offs_m_curr = start_m + offs_m 
		if EVEN_M & EVEN_HEADDIM:
			q = tl.load(q_ptrs)
		else:
			if EVEN_HEADDIM:
				q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
			else:
				q = tl.load(q_ptrs, mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
		qk = tl.dot(q, k.T)
		if not EVEN_N: 
			qk = tl.where(offs_n[None, :] < seqlen_k, qk, float("-inf")) 
		if HAVE_BIAS:
			tl.debug_barrier()  # Race condition otherwise
  
			if EVEN_M & EVEN_N:
				bias = tl.load(b_ptrs).to(tl.float32)
			else:
				bias = tl.load(b_ptrs, mask=(offs_m_curr[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k), other=0.0).to(tl.float32)
			qk = qk * softmax_scale + bias 
		if not (EVEN_M & EVEN_HEADDIM):
			tl.debug_barrier()
		lse_i = tl.load(L + offs_m_curr)
		if HAVE_BIAS:
			p = tl.exp(qk * softmax_scale - lse_i[:, None])
		else:
			p = tl.exp(qk - lse_i[:, None]) 
		if EVEN_M & EVEN_HEADDIM:
			do = tl.load(do_ptrs)
		else:
			do = tl.load(do_ptrs, mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
		dv += tl.dot(p.to(do.dtype).T, do)
		if not (EVEN_M & EVEN_HEADDIM):
			tl.debug_barrier()
		dp = tl.dot(do, v.T)
		if not EVEN_HEADDIM:
			tl.debug_barrier() 
		Di = tl.load(D + offs_m_curr) 
		ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype) 
		dk += tl.dot(ds.T, q) 
		if not (EVEN_M & EVEN_HEADDIM): 
			tl.debug_barrier()
		if not ATOMIC_ADD:
			if EVEN_M & EVEN_HEADDIM: 
				dq = tl.load(dq_ptrs, eviction_policy="evict_last")
				dq += tl.dot(ds, k)
				tl.store(dq_ptrs, dq, eviction_policy="evict_last")
			else:
				if EVEN_HEADDIM:
					dq = tl.load(dq_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0, eviction_policy="evict_last")
					dq += tl.dot(ds, k)
					tl.store(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q, eviction_policy="evict_last")
				else:
					dq = tl.load(dq_ptrs, mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0, eviction_policy="evict_last")
					dq += tl.dot(ds, k)
					tl.store( dq_ptrs, dq, mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim), eviction_policy="evict_last")
		else: 
			dq = tl.dot(ds, k)
			if EVEN_M & EVEN_HEADDIM: 
				tl.atomic_add(dq_ptrs, dq)
			else:
				if EVEN_HEADDIM:
					tl.atomic_add(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q)
				else:
					tl.atomic_add(dq_ptrs, dq, mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim))
		dq_ptrs += BLOCK_M * stride_dqm
		q_ptrs += BLOCK_M * stride_qm
		do_ptrs += BLOCK_M * stride_dom
		if HAVE_BIAS:
			b_ptrs += BLOCK_M * stride_bm 
	dv_ptrs = Dv + (offs_n[:, None] * stride_dvn + offs_d[None, :])
	dk_ptrs = Dk + (offs_n[:, None] * stride_dkn + offs_d[None, :])
	_bwd_store_dk_dv(
		dk_ptrs, dv_ptrs,
		dk, dv,
		offs_n, offs_d,
		seqlen_k, headdim,
		EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_HEADDIM=EVEN_HEADDIM
	)
# fmt:on


# fmt:off
@triton.heuristics(
	{
		"EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
		"EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
		"EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
	}
)
@triton.jit
def _bwd_attn_kernel(
	Q, K, V, B,
	Do, Dq, Dk, Dv,
	L, D,
	softmax_scale,
	stride_qb, stride_qh, stride_qm,
	stride_kb, stride_kh, stride_kn,
	stride_vb, stride_vh, stride_vn,
	stride_bb, stride_bh, stride_bm,
	stride_dob, stride_doh, stride_dom,
	stride_dqb, stride_dqh, stride_dqm, 
	stride_dkb, stride_dkh, stride_dkn,
	stride_dvb, stride_dvh, stride_dvn,
	seqlen_q, seqlen_k, seqlen_q_rounded,
	headdim, nheads, 
	HAVE_BIAS: tl.constexpr,
	BLOCK_HEADDIM: tl.constexpr, 
	EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr,
	BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
	off_hb = tl.program_id(1)
	off_b = off_hb // nheads
	off_h = off_hb % nheads
	Q += off_b * stride_qb + off_h * stride_qh
	K += off_b * stride_kb + off_h * stride_kh
	V += off_b * stride_vb + off_h * stride_vh
	Do += off_b * stride_dob + off_h * stride_doh
	Dq += off_b * stride_dqb + off_h * stride_dqh
	Dk += off_b * stride_dkb + off_h * stride_dkh
	Dv += off_b * stride_dvb + off_h * stride_dvh
	if HAVE_BIAS:
		B += off_b * stride_bb + off_h * stride_bh
	D += off_hb * seqlen_q_rounded
	L += off_hb * seqlen_q_rounded 
	start_n = tl.program_id(0)
	_bwd_kernel_one_col_block(
		start_n,
		Q, K, V, B,
		Do, Dq, Dk, Dv,
		L, D,
		softmax_scale,
		stride_qm, stride_kn, stride_vn, stride_bm,
		stride_dom,
		stride_dqm, stride_dkn, stride_dvn,
		seqlen_q, seqlen_k,
		headdim,
		ATOMIC_ADD=True,
		HAVE_BIAS=HAVE_BIAS,
		BLOCK_HEADDIM=BLOCK_HEADDIM,
		EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_HEADDIM=EVEN_HEADDIM,
		BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
	)

# fmt:on
# fmt:off
def _bwd_attn_kernel_call(
	softmax_scale,
	blocksize_q,
	blocksize_k,
	residual,
	Do,
):
	(o, l, query, key, value, bias) = residual
	batch, seqlen_q, nheads, headdim = query.shape
	_, seqlen_k, _, _ = key.shape
	assert key.shape == (
		batch,
		seqlen_k,
		nheads,
		headdim,
	), "shape missmatch between key, value."
	assert value.shape == (
		batch,
		seqlen_k,
		nheads,
		headdim,
	), "shape missmatch between key, value."
	assert headdim in {16, 32, 64, 128, 256}, "given headdim is not supported."
	assert query.dtype == key.dtype == value.dtype, "tensors must have the same dtype."
	assert query.dtype in [jnp.float16, jnp.bfloat16], "only support fp16 and bf16."
	softmax_scale = softmax_scale or 1.0 / math.sqrt(headdim)
	HAVE_BIAS = True if bias is not None else False
	BLOCK_HEADDIM = max(triton.next_power_of_2(headdim), 16)
	Dq = jnp.zeros_like(a=query, dtype=query.dtype, device=query.sharding)
	Dk = jnp.zeros_like(a=key, dtype=key.dtype, device=key.sharding)
	Dv = jnp.zeros_like(a=value, dtype=value.dtype, device=value.sharding)
	stride_bb, stride_bh, stride_bm = (
		strides_from_shape(bias.shape) if HAVE_BIAS else (0, 0, 0)
	)
	stride_qb, stride_qh, stride_qm = strides_from_shape(query.shape)
	stride_kb, stride_kh, stride_kn = strides_from_shape(key.shape)
	stride_vb, stride_vh, stride_vn = strides_from_shape(value.shape)
	stride_ob, stride_oh, stride_om = strides_from_shape(o.shape)
	stride_dqb, stride_dqh, stride_dqm = strides_from_shape(Dq.shape)
	stride_dkb, stride_dkh, stride_dkn = strides_from_shape(Dk.shape)
	stride_dvb, stride_dvh, stride_dvn = strides_from_shape(Dv.shape)  
	stride_dob, stride_doh, stride_dom = strides_from_shape(Do.shape)
	seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
	delta = jnp.empty_like(l)

	num_warps = 4 if headdim <= 64 else 8
	# kernel kwargs
	metaparams = dict(
		BLOCK_M=blocksize_q,
		BLOCK_HEADDIM=BLOCK_HEADDIM,
		num_warps=num_warps,
		num_stages=1,
	)
	delta = triton_call(
		o, Do, delta,
		stride_ob, stride_oh, stride_om,
		stride_dob, stride_doh, stride_dom,
		nheads, headdim,
		seqlen_q, seqlen_q_rounded,
		# triton call kwargs
		out_shape=[
			jax.ShapeDtypeStruct(
				shape=delta.shape,
				dtype=delta.dtype,
				sharding=delta.sharding,
			)
		],
		input_output_aliases={2: 0},
		grid=lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads),
		kernel=_bwd_do_attn_kernel,
		**metaparams,
	)[0]
	metaparams = dict(
		BLOCK_M=blocksize_q,
		BLOCK_N=blocksize_k,
		num_warps=num_warps,
		num_stages=1, 
		BLOCK_HEADDIM=BLOCK_HEADDIM,
		HAVE_BIAS=HAVE_BIAS,
	)

	Dq, Dk, Dv = triton_call(
		query, key, value, bias,
		Do, Dq, Dk, Dv,
		l, delta,
		softmax_scale,
		stride_qb, stride_qh, stride_qm,
		stride_kb, stride_kh, stride_kn,
		stride_vb, stride_vh, stride_vn,
		stride_bb, stride_bh, stride_bm,
		stride_dob, stride_doh, stride_dom,
		stride_dqb, stride_dqh, stride_dqm, 
		stride_dkb, stride_dkh, stride_dkn,
		stride_dvb, stride_dvh, stride_dvn,
		seqlen_q, seqlen_k, seqlen_q_rounded,
		headdim, nheads, 
		kernel=_bwd_attn_kernel,
		grid=lambda META: (triton.cdiv(seqlen_k, META["BLOCK_N"]), batch * nheads),
		out_shape=[
			jax.ShapeDtypeStruct(shape=Dq.shape, dtype=Dq.dtype, sharding=Dq.sharding),
			jax.ShapeDtypeStruct(shape=Dk.shape, dtype=Dk.dtype, sharding=Dk.sharding),
			jax.ShapeDtypeStruct(shape=Dv.shape, dtype=Dv.dtype, sharding=Dv.sharding),
		],
		input_output_aliases={5:0, 6:1, 7:2},
		**metaparams,
	)

	return Dq, Dk, Dv, None

# fmt:on


def _fwd_attn_kernel_call_with_residual(
	query: Optional[chex.Array],
	key: Optional[chex.Array],
	value: Optional[chex.Array],
	bias: Optional[chex.Array] = None,
	softmax_scale: Optional[float] = None,
	blocksize_q: int = 128,
	blocksize_k: int = 128,
):
	o, l = _fwd_attn_kernel_call(
		query=query,
		key=key,
		value=value,
		bias=bias,
		softmax_scale=softmax_scale,
		blocksize_k=blocksize_k,
		blocksize_q=blocksize_q,
	)
	return o, (o, l, query, key, value, bias)


@functools.partial(custom_vjp, nondiff_argnums=[4, 5, 6])
def flash_attn2_gpu(
	query: Optional[chex.Array],
	key: Optional[chex.Array],
	value: Optional[chex.Array],
	bias: Optional[chex.Array] = None,
	softmax_scale: Optional[float] = None,
	blocksize_q: int = 128,
	blocksize_k: int = 128,
):
	return _fwd_attn_kernel_call(
		query=query,
		key=key,
		value=value,
		bias=bias,
		softmax_scale=softmax_scale,
		blocksize_k=blocksize_k,
		blocksize_q=blocksize_q,
	)[0]


flash_attn2_gpu.defvjp(
	_fwd_attn_kernel_call_with_residual,
	_bwd_attn_kernel_call,
)


def _test_forward():
	q_key, k_key, v_key = jrnd.split(jrnd.PRNGKey(8), 3)
	B, H, S, D = 1, 23, 2021, 128
	q = jax.nn.initializers.normal(0.02)(q_key, (B, S, H, D), dtype=jnp.float16)
	k = jax.nn.initializers.normal(0.02)(k_key, (B, S, H, D), dtype=jnp.float16)
	v = jax.nn.initializers.normal(0.02)(v_key, (B, S, H, D), dtype=jnp.float16)
	b = jnp.where(
		jrnd.randint(v_key, (B, H, S, S), 0, 4) > 2,
		jnp.finfo(jnp.float16).min,
		0,
	)
	o = flash_attn2_gpu(q, k, v, b, blocksize_k=128, blocksize_q=64)
	try:
		fo = flax.linen.attention.dot_product_attention(q, k, v, b)
		print(jnp.allclose(o, fo, 0, 0.125))
	except Exception:
		print("Flax OOM")


def _test_backward():
	q_key, k_key, v_key = jrnd.split(jrnd.PRNGKey(8), 3)
	B, H, S, D = 1, 32, 256, 64
	q = jax.nn.initializers.normal(0.02)(q_key, (B, S, H, D), dtype=jnp.float16)
	k = jax.nn.initializers.normal(0.02)(k_key, (B, S, H, D), dtype=jnp.float16)
	v = jax.nn.initializers.normal(0.02)(v_key, (B, S, H, D), dtype=jnp.float16)
	b = (
		jnp.where(
			jrnd.randint(v_key, (B, H, S, S), 0, 4) > 2,
			jnp.finfo(jnp.float16).min,
			0,
		)
		if False
		else None
	)
	o = jax.grad(lambda *x: (flash_attn2_gpu(*x, blocksize_k=64, blocksize_q=64)).sum())(
		q, k, v, b
	)
	try:
		fo = jax.grad(lambda *x: flax.linen.attention.dot_product_attention(*x).sum())(
			q, k, v, b
		)
	except Exception as e:
		print(f"Flax OOM : {e}")
		exit()
	print(o[-1, -1, -1, :5])
	print(fo[-1, -1, -1, :5])
	print(jnp.allclose(o, fo, 0, 5e-5))


if __name__ == "__main__":
	# _test_forward()
	_test_backward()
