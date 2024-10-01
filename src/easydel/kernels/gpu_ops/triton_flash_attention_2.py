import math

import jax
import triton
from fjformer.jax_triton import strides_from_shape, triton_call
from jax import numpy as jnp
from jax import random as jrnd
from triton import language as tl


@triton.heuristics(
	{
		"EVEN_Q_BLOCK": lambda args: args["seqlen_q"] % args["BLOCK_SIZE_Q"] == 0,
		"EVEN_K_BLOCK": lambda args: args["seqlen_k"] % args["BLOCK_SIZE_K"] == 0,
		"EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
	}
)
@triton.jit
def _fwd_attn_kernel_bpt(): ...


@triton.heuristics(
	{
		"EVEN_Q_BLOCK": lambda args: args["seqlen_q"] % args["BLOCK_SIZE_Q"] == 0,
		"EVEN_K_BLOCK": lambda args: args["seqlen_k"] % args["BLOCK_SIZE_K"] == 0,
		"EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
	}
)  # fmt: off
@triton.jit
def _fwd_attn_kernel(
	Q,
	K,
	V,
	B,
	softmax_scale,
	stride_qb,
	stride_qh,
	stride_qs,
	stride_kb,
	stride_kh,
	stride_ks,
	stride_vb,
	stride_vh,
	stride_vs,
	stride_bb,
	stride_bh,
	stride_bs,
	stride_ob,
	stride_oh,
	stride_os,
	nheads,
	seqlen_q,
	seqlen_k,
	seqlen_q_rounded,
	headdim,
	O,
	L,
	HAVE_BIAS: tl.constexpr,
	BLOCK_HEADDIM: tl.constexpr,
	BLOCK_SIZE_Q: tl.constexpr,
	BLOCK_SIZE_K: tl.constexpr,
	EVEN_Q_BLOCK: tl.constexpr,
	EVEN_K_BLOCK: tl.constexpr,
	EVEN_HEADDIM: tl.constexpr,
):
	# fmt: off
	off_q, off_hb = tl.program_id(0), tl.program_id(1)
	off_b = off_hb // nheads
	off_h = off_hb % nheads

	offs_qs = off_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
	offs_ks = tl.arange(0, BLOCK_SIZE_K)
	offs_d = tl.arange(0, BLOCK_HEADDIM)

	q_ptr = Q + (off_b * stride_qb + off_h * stride_qh + (offs_qs[:, None] * stride_qs + offs_d[None, :]))  
	k_ptr = K + (off_b * stride_kb + off_h * stride_kh + (offs_ks[:, None] * stride_ks + offs_d[None, :])) 
	v_ptr = V + (off_b * stride_vb + off_h * stride_vh + (offs_ks[:, None] * stride_vs + offs_d[None, :]))  
	if HAVE_BIAS:
		b_ptr = B + (off_b * stride_bb + off_h * stride_bh + (offs_qs[:, None] * stride_bs + offs_ks[None, :]))  
	lse_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
	max_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
	acc_o = tl.zeros([BLOCK_SIZE_Q, BLOCK_HEADDIM], dtype=tl.float32)
	softmax_scale = softmax_scale.to(tl.float32)

	if EVEN_K_BLOCK & EVEN_Q_BLOCK:
		if EVEN_HEADDIM:
			q = tl.load(q_ptr)
		else:
			q = tl.load(q_ptr, mask=offs_d[None, :] < headdim, other=0.0)
	else:
		if EVEN_HEADDIM:
			q = tl.load(q_ptr, mask=offs_qs[:, None] < seqlen_q, other=0.0)
		else:
			q = tl.load(q_ptr, mask=(offs_qs[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)

	# fmt:on
	for j in range(0, seqlen_k, BLOCK_SIZE_K):
		j = tl.multiple_of(j, BLOCK_SIZE_K)
		# fmt: off
		if EVEN_K_BLOCK & EVEN_Q_BLOCK:
			if EVEN_HEADDIM:
				k = tl.load(k_ptr + j * stride_ks)
			else:
				k = tl.load(k_ptr + j * stride_ks, mask=offs_d[None, :] < headdim, other=0.0)
		else:
			if EVEN_HEADDIM:
				k = tl.load(k_ptr + j * stride_ks, mask=(j + offs_ks)[:, None] < seqlen_k, other=0.0)
			else:
				k = tl.load(k_ptr + j * stride_ks, mask=((j + offs_ks)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
		# fmt: on
		qk = tl.zeros([BLOCK_SIZE_Q, BLOCK_SIZE_K], dtype=tl.float32)
		qk += tl.dot(q, k.T)
		if not EVEN_K_BLOCK:
			qk += tl.where((j + offs_ks)[None, :] < seqlen_k, 0, float("-inf")).to(tl.float32)
		if HAVE_BIAS:
			# fmt: off
			if EVEN_K_BLOCK & EVEN_Q_BLOCK:
				b = tl.load(b_ptr + j).to(tl.float32)
			else:
				b = tl.load(b_ptr + j, mask=(offs_qs[:, None] < seqlen_q) & (j + offs_ks)[None, :] < seqlen_k, other=0.0).to(tl.float32)
			# fmt: on
			qk = qk * softmax_scale + b
			max_ij = tl.maximum(tl.max(qk, 1), lse_i)
			p = tl.exp(qk - max_ij[:, None])
		else:
			max_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
			p = tl.exp(qk * softmax_scale - max_ij[:, None])

		l_ij = tl.sum(p, 1)
		acc_o_scale = tl.exp(max_i - max_ij)
		acc_o = acc_o * acc_o_scale[:, None]
		# fmt: off
		if EVEN_Q_BLOCK & EVEN_K_BLOCK:
			if EVEN_HEADDIM:
				v = tl.load(v_ptr + j * stride_vs)
			else:
				v = tl.load(v_ptr + j * stride_vs, mask=offs_d[None, :] < headdim, other=0.0)
		else:
			if EVEN_HEADDIM:
				v = tl.load(v_ptr + j * stride_vs, mask=(j + offs_ks)[:, None] < seqlen_k, other=0.0)
			else:
				v = tl.load(v_ptr + j * stride_vs, mask=((j + offs_ks)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
		# fmt: on
		acc_o += tl.dot(p.to(v.dtype), v)
		max_i = max_ij
		lin = tl.exp(lse_i - max_ij) + l_ij
		lse_i = max_ij + tl.log(lin)
		o_scale = tl.exp(max_i - lse_i)
		acc_o = acc_o * o_scale[:, None]
		lse_ptrs = L + off_hb * seqlen_q_rounded + offs_qs
		tl.store(lse_ptrs, lse_i)

		# fmt: off
		out_ptrs = O + (off_b * stride_ob+ off_h * stride_oh+ (offs_qs[:, None] * stride_os + offs_d[None, :]))
		if EVEN_Q_BLOCK:
			if EVEN_HEADDIM:
				tl.store(out_ptrs, acc_o)
			else:
				tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
		else:
			if EVEN_HEADDIM:
				tl.store(out_ptrs, acc_o, mask=offs_qs[:, None] < seqlen_q)
			else:
				tl.store(out_ptrs, acc_o, mask=(offs_qs[:, None] < seqlen_q) & (offs_d[None, :] < headdim))
		# fmt: on


def _fwd_attn_kernel_call(
	query,
	key,
	value,
	bias,
	softmax_scale=None,
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
	assert headdim in {16, 32, 64, 128}, "given headdim is not supported."
	assert query.dtype == key.dtype == value.dtype, "tensors must have the same dtype."
	assert query.dtype in [jnp.float16, jnp.bfloat16], "only support fp16 and bf16."
	softmax_scale = softmax_scale or 1.0 / math.sqrt(headdim)

	HAVE_BIAS = True if bias is not None else False
	BLOCK_HEADDIM = max(triton.next_power_of_2(headdim), 16)
	b_strides = strides_from_shape(bias.shape)[:-1] if HAVE_BIAS else (0, 0, 0)
	q_strides = strides_from_shape(query.shape)[:-1]
	k_strides = strides_from_shape(key.shape)[:-1]
	v_strides = strides_from_shape(value.shape)[:-1]

	seqlen_q_rounded = math.ceil(seqlen_q / blocksize_q) * blocksize_q
	num_warps = 4 if headdim <= 64 else 4
	out_shape = [
		jax.ShapeDtypeStruct(
			query.shape,
			query.dtype,
		),  # O,
		jax.ShapeDtypeStruct(
			(batch, nheads, seqlen_q_rounded),
			jnp.float32,
		),  # L,
	]
	metaparams = dict(
		HAVE_BIAS=HAVE_BIAS,
		BLOCK_SIZE_Q=blocksize_q,
		BLOCK_SIZE_K=blocksize_k,
		BLOCK_HEADDIM=BLOCK_HEADDIM,
	)

	return triton_call(
		query,  # Q,
		key,  # K,
		value,  # V,
		bias,  # B,
		softmax_scale,  # softmax_scale,
		*q_strides,
		*k_strides,
		*v_strides,
		*b_strides,
		*q_strides,
		nheads,  # nheads,
		seqlen_q,  # seqlen_q,
		seqlen_k,  # seqlen_k,
		seqlen_q_rounded,  # seqlen_q_rounded,
		headdim,  # headdim,
		# call args
		kernel=_fwd_attn_kernel,
		out_shape=out_shape,
		num_warps=num_warps,
		num_stages=1,
		grid=lambda META: (triton.cdiv(seqlen_q, META["BLOCK_SIZE_Q"]), batch * nheads),
		**metaparams,
	)


def _bwd_do_attn_kernel(): ...
def main():
	q_key, k_key, v_key = jrnd.split(jrnd.PRNGKey(8), 3)
	B, H, S, D = 1, 8, 32, 32
	q = jrnd.normal(q_key, (B, S, H, D), dtype=jnp.float16)
	k = jrnd.normal(k_key, (B, S, H, D), dtype=jnp.float16)
	v = jrnd.normal(v_key, (B, S, H, D), dtype=jnp.float16)
	b = jrnd.randint(v_key, (B, H, S, S), 0, 4) > 2
	o, l = _fwd_attn_kernel_call(
		query=q,
		key=k,
		value=v,
		bias=b,
		blocksize_q=16,
		blocksize_k=16,
	)
	print(o)


if __name__ == "__main__":
	main()
