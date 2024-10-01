import math

import flax
import flax.linen
import flax.linen.attention
import jax
import triton
from fjformer.jax_triton import strides_from_shape, triton_call
from jax import numpy as jnp
from jax import random as jrnd
from triton import language as tl


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
	query,
	key,
	value,
	bias,
	softmax_scale: float = None,
	blocksize_q=128,
	blocksize_k=128,
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
	b_strides = strides_from_shape(bias.shape)[:-1] if HAVE_BIAS else (0, 0, 0)
	q_strides = strides_from_shape(query.shape)[:-1]
	k_strides = strides_from_shape(key.shape)[:-1]
	v_strides = strides_from_shape(value.shape)[:-1]
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
		*q_strides, *k_strides, *v_strides, *b_strides, *q_strides, 
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


def _bwd_do_attn_kernel(): ...
def main():
	q_key, k_key, v_key = jrnd.split(jrnd.PRNGKey(8), 3)
	B, H, S, D = 1, 8, 1024, 128
	q = jax.nn.initializers.normal(0.02)(q_key, (B, S, H, D), dtype=jnp.float16)
	k = jax.nn.initializers.normal(0.02)(k_key, (B, S, H, D), dtype=jnp.float16)
	v = jax.nn.initializers.normal(0.02)(v_key, (B, S, H, D), dtype=jnp.float16)
	b = jnp.where(
		jrnd.randint(v_key, (B, H, S, S), 0, 4) > 2,
		jnp.finfo(jnp.float16).min,
		0,
	)
	o, l = _fwd_attn_kernel_call(
		query=q,
		key=k,
		value=v,
		bias=b,
		blocksize_k=128,
		blocksize_q=64,
	)
	fo = flax.linen.attention.dot_product_attention(q, k, v, b)
	print(jnp.allclose(o, fo, 0, 0.125))


if __name__ == "__main__":
	main()
