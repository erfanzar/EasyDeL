import triton
from triton import language as tl
from fjformer.jax_triton import triton_call, strides_from_shape
import jax
from jax import numpy as jnp, random as jrnd
import math


@triton.heuristics(
	{
		"EVEN_QS": lambda args: args["seqlen_q"] % args["BLOCK_QS"] == 0,
		"EVEN_KS": lambda args: args["seqlen_k"] % args["BLOCK_KS"] == 0,
		"EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
	}
)
@triton.jit
def _fwd_attn_kernel_bpt(): ...


@triton.heuristics(
	{
		"EVEN_QS": lambda args: args["seqlen_q"] % args["BLOCK_QS"] == 0,
		"EVEN_KS": lambda args: args["seqlen_k"] % args["BLOCK_KS"] == 0,
		"EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
	}
)
@triton.jit
def _fwd_attn_kernel(
	Q,  # noqa:E741
	K,  # noqa:E741
	V,  # noqa:E741
	B,  # noqa:E741
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
	O,  # noqa:E741
	L,  # noqa:E741
	HAVE_BIAS: tl.constexpr,
	BLOCK_HEADDIM: tl.constexpr,
	BLOCK_QS: tl.constexpr,
	BLOCK_KS: tl.constexpr,
	# heuristics
	EVEN_QS: tl.constexpr,
	EVEN_KS: tl.constexpr,
	EVEN_HEADDIM: tl.constexpr,
	**META,
):
	off_q, off_hb = tl.program_id(0), tl.program_id(1)
	off_b = off_hb // nheads
	off_h = off_hb % nheads
	offs_qs = off_q * BLOCK_QS + tl.arange(0, BLOCK_QS)
	offs_ks = tl.arange(0, BLOCK_KS)
	offs_d = tl.arange(0, BLOCK_HEADDIM)
	q_ptr = Q + (
		off_b * stride_qb
		+ off_h * stride_qh
		+ (offs_qs[:, None] * stride_qs + offs_d[None, :])
	)
	k_ptr = K + (
		off_b * stride_kb
		+ off_h * stride_kh
		+ (offs_ks[:, None] * stride_ks + offs_d[None, :])
	)
	v_ptr = V + (
		off_b * stride_vb
		+ off_h * stride_vh
		+ (offs_ks[:, None] * stride_vs + offs_d[None, :])
	)
	if HAVE_BIAS == "true":
		b_ptr = B + (
			off_b * stride_bb
			+ off_h * stride_bh
			+ (offs_qs[:, None] * stride_bs + offs_ks[None, :])
		)
	lse_i = tl.zeros([BLOCK_QS], dtype=tl.float32) - float("inf")
	max_i = tl.zeros([BLOCK_QS], dtype=tl.float32) - float("inf")
	acc_o = tl.zeros([BLOCK_QS, BLOCK_HEADDIM], dtype=tl.float32)
	softmax_scale = softmax_scale.to(tl.float32)
	if EVEN_KS & EVEN_QS:
		if EVEN_HEADDIM:
			q = tl.load(q_ptr)
		else:
			q = tl.load(
				q_ptr,
				mask=offs_d[None, :] < headdim,
				other=0.0,
			)
	else:
		if EVEN_HEADDIM:
			q = tl.load(
				q_ptr,
				mask=offs_qs[:, None] < seqlen_q,
				other=0.0,
			)
		else:
			q = tl.load(
				q_ptr,
				mask=(offs_qs[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
				other=0.0,
			)
	for j in range(0, seqlen_k, BLOCK_KS):
		j = tl.multiple_of(j, BLOCK_KS)
		if EVEN_KS & EVEN_QS:
			if EVEN_HEADDIM:
				k = tl.load(k_ptr + j * stride_ks)
			else:
				k = tl.load(
					k_ptr + j * stride_ks,
					mask=offs_d[None, :] < headdim,
					other=0.0,
				)
		else:
			if EVEN_HEADDIM:
				k = tl.load(
					k_ptr + j * stride_ks,
					mask=(j * offs_ks)[:, None] < seqlen_k,
					other=0.0,
				)
			else:
				k = tl.load(
					k_ptr + j * stride_ks,
					mask=((j * offs_ks)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
					other=0.0,
				)
		qk = tl.zeros([BLOCK_QS, BLOCK_KS], dtype=tl.float32)
		qk += tl.dot(q, k.T)
		if not EVEN_KS:
			qk += tl.where((j * offs_ks)[None, :] < seqlen_k, 0, float("-inf")).to(tl.float32)
		if HAVE_BIAS == "true":
			if EVEN_KS & EVEN_QS:
				b = tl.load(b_ptr + j).to(tl.float32)
			else:
				b = tl.load(
					b_ptr + j,
					mask=(offs_qs[:, None] < seqlen_q) & (j + offs_ks)[None, :] < seqlen_k,
					other=0.0,
				).to(tl.float32)
			qk = qk * softmax_scale + b
			max_ij = tl.maximum(tl.max(qk, 1), lse_i)
			p = tl.exp(qk - max_ij[:, None])
		else:
			max_ij = tl.maximum(tl.max(qk * softmax_scale, 1), lse_i)
			p = tl.exp(qk - max_ij[:, None])
		l_ij = tl.sum(p, 1)
		acc_o_scale = tl.exp(max_i - max_ij).to(tl.float32)
		acc_o = acc_o * acc_o_scale[:, None]
		if EVEN_QS & EVEN_KS:
			if EVEN_HEADDIM:
				v = tl.load(v_ptr + j * stride_vs)
			else:
				v = tl.load(
					v_ptr + j * stride_vs,
					mask=offs_d[None, :] < headdim,
					other=0.0,
				)
		else:
			if EVEN_HEADDIM:
				v = tl.load(
					v_ptr + j * stride_vs,
					mask=(j + offs_ks)[:, None] < seqlen_k,
					other=0.0,
				)
			else:
				v = tl.load(
					v_ptr + j * stride_vs,
					mask=((j + offs_ks)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
					other=0.0,
				)
		acc_o += tl.dot(p.to(tl.float16), v)
		max_i = max_ij
		lin = tl.exp(lse_i - max_ij) + l_ij
		lse_i = max_ij + tl.log(lin)
		o_scale = tl.exp(max_i - lse_i)
		acc_o = acc_o * o_scale[:, None]
		qs = tl.program_id(0)
		offs_qs = qs * BLOCK_QS + tl.arange(0, BLOCK_QS)

		lse_ptrs = L + off_hb * seqlen_q_rounded + offs_qs
		tl.store(lse_ptrs, lse_i)
		offs_d = tl.arange(0, BLOCK_HEADDIM)
		out_ptrs = (
			O
			+ off_b * stride_ob
			+ off_h * stride_oh
			+ (offs_qs[:, None] * stride_os + offs_d[None, :])
		)
		if EVEN_QS:
			if EVEN_HEADDIM:
				tl.store(out_ptrs, acc_o)
			else:
				tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
		else:
			if EVEN_HEADDIM:
				tl.store(out_ptrs, acc_o, mask=offs_qs[:, None] < seqlen_q)
			else:
				tl.store(
					out_ptrs,
					acc_o,
					mask=(offs_qs[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
				)


def _fwd_attn_kernel_call(q, k, v, b, softmax_scale=None):
	batch, seqlen_q, nheads, d = q.shape
	_, seqlen_k, _, _ = k.shape
	assert k.shape == (batch, seqlen_k, nheads, d)
	assert v.shape == (batch, seqlen_k, nheads, d)

	assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
	assert q.dtype in [jnp.float16, jnp.bfloat16], "Only support fp16 and bf16"
	softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

	HAVE_BIAS = "true" if b is not None else "false"
	BLOCK = 128
	BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)

	b_strides = strides_from_shape(b.shape) if b is not None else (0, 0, 0)
	q_strides = strides_from_shape(q.shape)
	k_strides = strides_from_shape(k.shape)
	v_strides = strides_from_shape(v.shape)

	seqlen_q_rounded = math.ceil(seqlen_q / BLOCK) * BLOCK
	num_warps = 4 if d <= 64 else 8

	out_shape = [
		jax.ShapeDtypeStruct(
			q.shape,
			q.dtype,
		),  # O,  # noqa:E741
		jax.ShapeDtypeStruct(
			(batch, nheads, seqlen_q_rounded),
			jnp.float32,
		),  # L,  # noqa:E741
	]

	metaparams = dict(
		HAVE_BIAS=HAVE_BIAS,
		BLOCK_QS=BLOCK,
		BLOCK_KS=BLOCK,
		BLOCK_HEADDIM=BLOCK_HEADDIM,
	)

	# HAVE_BIAS: tl.constexpr,
	# BLOCK_HEADDIM: tl.constexpr,
	# BLOCK_QS: tl.constexpr,
	# BLOCK_KS: tl.constexpr,
	# # heuristics
	# EVEN_QS: tl.constexpr,
	# EVEN_KS: tl.constexpr,
	# EVEN_HEADDIM: tl.constexpr,
	return triton_call(
		q,  # Q,  # noqa:E741
		k,  # K,  # noqa:E741
		v,  # V,  # noqa:E741
		b,  # B,  # noqa:E741
		softmax_scale,  # softmax_scale,
		q_strides[0],  # stride_qb,
		q_strides[1],  # stride_qh,
		q_strides[2],  # stride_qs,
		k_strides[0],  # stride_kb,
		k_strides[1],  # stride_kh,
		k_strides[2],  # stride_ks,
		v_strides[0],  # stride_vb,
		v_strides[1],  # stride_vh,
		v_strides[2],  # stride_vs,
		b_strides[0],  # stride_bb,
		b_strides[1],  # stride_bh,
		b_strides[2],  # stride_bs,
		q_strides[0],  # stride_ob,
		q_strides[1],  # stride_oh,
		q_strides[2],  # stride_os,
		nheads,  # nheads,
		seqlen_q,  # seqlen_q,
		seqlen_k,  # seqlen_k,
		seqlen_q_rounded,  # seqlen_q_rounded,
		d,  # headdim,
		# call args
		kernel=_fwd_attn_kernel,
		out_shape=out_shape,
		num_warps=num_warps,
		num_stages=1,
		grid=lambda META: (triton.cdiv(seqlen_q, META["BLOCK_QS"]), batch * nheads),
		**metaparams,
	)


def _bwd_do_attn_kernel(): ...
def main():
	q_key, k_key, v_key = jrnd.split(jrnd.PRNGKey(0), 3)
	B, H, S, D = 1, 8, 1024, 128
	q = jrnd.normal(q_key, (B, S, H, D), dtype=jnp.float16)
	k = jrnd.normal(k_key, (B, S, H, D), dtype=jnp.float16)
	v = jrnd.normal(v_key, (B, S, H, D), dtype=jnp.float16)
	b = jrnd.randint(v_key, (B, H, S, S), 0, 4) > 2
	o, l = _fwd_attn_kernel_call(q, k, v, b)
	print(o)


if __name__ == "__main__":
	main()
