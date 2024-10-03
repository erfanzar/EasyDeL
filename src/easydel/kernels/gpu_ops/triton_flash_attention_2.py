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


def get_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
	size = np.prod(shape)
	strides = []
	for s in shape:
		size = int(size // s)
		strides.append(size)
	return tuple(strides)


def apply_stride(self):
	def func(idx) -> int:
		size = np.prod(self.shape)
		strides = []
		for s in self.shape:
			size = int(size // s)
			if size != 1:
				strides.append(size)
		if len(strides) == 0:
			strides.append(1)
		return int(strides[idx])

	self.stride = func
	return self


def get_sharding(arr):
	return getattr(arr, "sharding", None)


def check_shapes_and_dtypes(
	query, key, value, batch, seqlen_k, nheads, headdim, blocksize_k, blocksize_q
):
	chex.assert_shape(
		key, (batch, seqlen_k, nheads, headdim), custom_message="Shape mismatch for key."
	)
	chex.assert_shape(
		value,
		(batch, seqlen_k, nheads, headdim),
		custom_message="Shape mismatch for value.",
	)
	chex.assert_equal(
		query.dtype, key.dtype, custom_message="Dtype mismatch between query and key."
	)
	chex.assert_equal(
		key.dtype, value.dtype, custom_message="Dtype mismatch between key and value."
	)
	if query.dtype not in [jnp.float16, jnp.bfloat16]:
		raise AssertionError("Only fp16 and bf16 are supported.") from None
	chex.assert_is_divisible(
		blocksize_k, 16, custom_message="blocksize_k should be divisible by 16."
	)
	chex.assert_is_divisible(
		blocksize_q, 16, custom_message="blocksize_q should be divisible by 16."
	)
	if headdim not in [16, 32, 64, 128, 256]:
		raise AssertionError("Unsupported headdim value.")


@triton.heuristics(
	{
		"EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
		"EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
		"EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
	}
)
@triton.jit
def _fwd_attn_kernel(
	Q,
	K,
	V,
	B,
	softmax_scale,
	stride_qb,
	stride_qh,
	stride_qm,
	stride_kb,
	stride_kh,
	stride_kn,
	stride_vb,
	stride_vh,
	stride_vn,
	stride_bb,
	stride_bh,
	stride_bm,
	stride_bn,
	stride_ob,
	stride_oh,
	stride_om,
	stride_lb,
	stride_lh,
	headdim,
	seqlen_q,
	seqlen_k,
	O,
	L,
	HAVE_BIAS: tl.constexpr,
	BLOCK_HEADDIM: tl.constexpr,
	EVEN_M: tl.constexpr,
	EVEN_N: tl.constexpr,
	EVEN_HEADDIM: tl.constexpr,
	BLOCK_M: tl.constexpr,
	BLOCK_N: tl.constexpr,
):
	off_m, off_b, off_h = (
		tl.program_id(0),
		tl.program_id(1),
		tl.program_id(2),
	)  # this one was the bug the whole time ...
	offs_m = off_m * BLOCK_M + tl.arange(0, BLOCK_M)
	offs_d = tl.arange(0, BLOCK_HEADDIM)
	offs_n = tl.arange(0, BLOCK_N)
	q_ptrs = Q + (
		off_b * stride_qb
		+ off_h * stride_qh
		+ (offs_m[:, None] * stride_qm + offs_d[None, :])
	)

	if EVEN_N & EVEN_M:
		if EVEN_HEADDIM:
			q = tl.load(q_ptrs)
		else:
			q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
	else:
		if EVEN_HEADDIM:
			q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
		else:
			q = tl.load(
				q_ptrs,
				mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
				other=0.0,
			)
	k_ptrs = K + (
		off_b * stride_kb
		+ off_h * stride_kh
		+ (offs_n[:, None] * stride_kn + offs_d[None, :])
	)
	v_ptrs = V + (
		off_b * stride_vb
		+ off_h * stride_vh
		+ (offs_n[:, None] * stride_vn + offs_d[None, :])
	)
	softmax_scale = softmax_scale.to(tl.float32)
	if HAVE_BIAS:
		b_ptrs = B + (
			off_b * stride_bb
			+ off_h * stride_bh
			+ (offs_m[:, None] * stride_bm + offs_n[None, :])
		)

	lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
	max_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
	acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

	for j in range(0, seqlen_k, BLOCK_N):
		j = tl.multiple_of(j, BLOCK_N)
		if EVEN_N & EVEN_M:
			if EVEN_HEADDIM:
				k = tl.load(k_ptrs + j * stride_kn)
			else:
				k = tl.load(k_ptrs + j * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
		else:
			if EVEN_HEADDIM:
				k = tl.load(
					k_ptrs + j * stride_kn, mask=(j + offs_n)[:, None] < seqlen_k, other=0.0
				)
			else:
				k = tl.load(
					k_ptrs + j * stride_kn,
					mask=((j + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
					other=0.0,
				)
		qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
		qk += tl.dot(q, k.T)
		if not EVEN_N:
			qk += tl.where((j + offs_n)[None, :] < seqlen_k, 0, float("-inf")).to(tl.float32)
		if HAVE_BIAS:
			if EVEN_N & EVEN_M:
				b = tl.load(b_ptrs + j).to(tl.float32)
			else:
				b = tl.load(
					b_ptrs + j,
					mask=(offs_m[:, None] < seqlen_q) & (j + offs_n)[None, :] < seqlen_k,
					other=0.0,
				).to(tl.float32)
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
				v = tl.load(
					v_ptrs + j * stride_vn, mask=(j + offs_n)[:, None] < seqlen_k, other=0.0
				)
			else:
				v = tl.load(
					v_ptrs + j * stride_vn,
					mask=((j + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
					other=0.0,
				)
		acc_o += tl.dot(p.to(v.dtype), v)
		max_i = max_ij
		lin = tl.exp(lse_i - max_ij) + l_ij
		lse_i = max_ij + tl.log(lin)

	o_scale = tl.exp(max_i - lse_i)
	acc_o = acc_o * o_scale[:, None]
	lse_ptrs = L + (off_b * stride_lb + off_h * stride_lh + offs_m)
	tl.store(lse_ptrs, lse_i)
	# Here's a Bug
	out_ptrs = O + (
		off_b * stride_ob
		+ off_h * stride_oh
		+ (offs_m[:, None] * stride_om + offs_d[None, :])
	)
	if EVEN_M:
		if EVEN_HEADDIM:
			tl.store(out_ptrs, acc_o)
		else:
			tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
	else:
		if EVEN_HEADDIM:
			tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
		else:
			tl.store(
				out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
			)


def _fwd_attn_kernel_call(
	query: Optional[chex.Array],
	key: Optional[chex.Array],
	value: Optional[chex.Array],
	bias: Optional[chex.Array] = None,
	softmax_scale: Optional[float] = None,
	blocksize_q: int = 128,
	blocksize_k: int = 128,
):
	query = apply_stride(query)
	key = apply_stride(key)
	value = apply_stride(value)
	batch, seqlen_q, nheads, headdim = query.shape
	_, seqlen_k, _, _ = key.shape
	check_shapes_and_dtypes(
		query=query,
		key=key,
		value=value,
		batch=batch,
		seqlen_k=seqlen_k,
		nheads=nheads,
		headdim=headdim,
		blocksize_k=blocksize_k,
		blocksize_q=blocksize_q,
	)
	softmax_scale = softmax_scale or 1.0 / math.sqrt(headdim)
	HAVE_BIAS = True if bias is not None else False
	BLOCK_HEADDIM = max(triton.next_power_of_2(headdim), 16)
	seqlen_q_rounded = math.ceil(seqlen_q / blocksize_q) * blocksize_q
	stride_bb, stride_bh, stride_bm, stride_bn = (
		get_strides(bias.shape) if HAVE_BIAS else (0, 0, 0, 0)
	)
	stride_lb, stride_lh, stride_lm = get_strides((batch, nheads, seqlen_q_rounded))
	metaparams = dict(
		HAVE_BIAS=HAVE_BIAS,
		BLOCK_HEADDIM=BLOCK_HEADDIM,
		BLOCK_M=blocksize_q,
		BLOCK_N=blocksize_k,
	)
	num_warps = 4 if headdim <= 64 else 8
	return triton_call(
		query,
		key,
		value,
		bias if bias is not None else False,
		softmax_scale,
		query.stride(0),
		query.stride(2),
		query.stride(1),
		key.stride(0),
		key.stride(2),
		key.stride(1),
		value.stride(0),
		value.stride(2),
		value.stride(1),
		stride_bb,
		stride_bh,
		stride_bm,
		stride_bn,
		query.stride(0),
		query.stride(2),
		query.stride(1),
		stride_lb,
		stride_lh,
		headdim,
		seqlen_q,
		seqlen_k,
		kernel=_fwd_attn_kernel,
		out_shape=[
			jax.ShapeDtypeStruct(query.shape, query.dtype, sharding=get_sharding(query)),  # O
			jax.ShapeDtypeStruct((batch, nheads, seqlen_q_rounded), jnp.float32),  # L
		],
		grid=lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch, nheads),
		num_warps=num_warps,
		num_stages=1,
		**metaparams,
	)


@triton.jit
def _bwd_do_attn_kernel(
	O,
	Do,
	De,
	stride_ob,
	stride_om,
	stride_oh,
	stride_dob,
	stride_dom,
	stride_doh,
	nheads,
	headdim,
	seqlen_q,
	seqlen_q_rounded,
	BLOCK_M: tl.constexpr,
	BLOCK_HEADDIM: tl.constexpr,
):
	off_q = tl.program_id(0)
	off_hb = tl.program_id(1)
	off_b = off_hb // nheads
	off_h = off_hb % nheads
	offs_m = off_q * BLOCK_M + tl.arange(0, BLOCK_M)
	offs_d = tl.arange(0, BLOCK_HEADDIM)
	o_ptrs = (
		O
		+ off_b * stride_ob
		+ off_h * stride_oh
		+ offs_m[:, None] * stride_om
		+ offs_d[None, :]
	)
	do_ptrs = (
		Do
		+ off_b * stride_dob
		+ off_h * stride_doh
		+ offs_m[:, None] * stride_dom
		+ offs_d[None, :]
	)
	o = tl.load(
		o_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0
	).to(tl.float32)
	do = tl.load(
		do_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0
	).to(tl.float32)
	delta = tl.sum(o * do, axis=1)
	tl.store(De + off_hb * seqlen_q_rounded + offs_m, delta)


@triton.heuristics(
	{
		"EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
		"EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
		"EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
	}
)
@triton.jit
def _bwd_attn_kernel(
	Q,
	K,
	V,
	B,
	Do,
	Dq,
	Dk,
	Dv,
	L,
	D,
	softmax_scale,
	stride_qb,
	stride_qh,
	stride_qm,
	stride_kb,
	stride_kh,
	stride_kn,
	stride_vb,
	stride_vh,
	stride_vn,
	stride_bb,
	stride_bh,
	stride_bm,
	stride_dob,
	stride_doh,
	stride_dom,
	stride_dqb,
	stride_dqh,
	stride_dqm,
	stride_dkb,
	stride_dkh,
	stride_dkn,
	stride_dvb,
	stride_dvh,
	stride_dvn,
	seqlen_q,
	seqlen_k,
	seqlen_q_rounded,
	headdim,
	nheads,
	HAVE_BIAS: tl.constexpr,
	BLOCK_HEADDIM: tl.constexpr,
	EVEN_M: tl.constexpr,
	EVEN_N: tl.constexpr,
	EVEN_HEADDIM: tl.constexpr,
	BLOCK_M: tl.constexpr,
	BLOCK_N: tl.constexpr,
):
	# fmt:on
	off_n, off_b, off_h = (
		tl.program_id(0),
		tl.program_id(1),
		tl.program_id(2),
	)
	offs_n = off_n * BLOCK_N + tl.arange(0, BLOCK_N)
	offs_d = tl.arange(0, BLOCK_HEADDIM)
	k_ptrs = K + (
		off_b * stride_kb
		+ off_h * stride_kh
		+ (offs_n[:, None] * stride_kn + offs_d[None, :])
	)
	v_ptrs = V + (
		off_b * stride_vb
		+ off_h * stride_vh
		+ (offs_n[:, None] * stride_vn + offs_d[None, :])
	)
	dk_ptrs = Dk + (
		off_b * stride_dkb
		+ off_h * stride_dkh
		+ (offs_n[:, None] * stride_dkn + offs_d[None, :])
	)
	dv_ptrs = Dv + (
		off_b * stride_dvb
		+ off_h * stride_dvh
		+ (offs_n[:, None] * stride_dvn + offs_d[None, :])
	)

	if EVEN_HEADDIM:
		if EVEN_N:
			k = tl.load(k_ptrs)
			v = tl.load(v_ptrs)
			dk = tl.load(dk_ptrs)
			dv = tl.load(dv_ptrs)
		else:
			k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
			v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
			dk = tl.load(dk_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
			dv = tl.load(dv_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
	else:
		if EVEN_N:
			k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
			v = tl.load(v_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
			dk = tl.load(dk_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
			dv = tl.load(dv_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
		else:
			k = tl.load(
				k_ptrs,
				mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
				other=0.0,
			)
			v = tl.load(
				v_ptrs,
				mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
				other=0.0,
			)
			dk = tl.load(
				dk_ptrs,
				mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
				other=0.0,
			)
			dv = tl.load(
				dv_ptrs,
				mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
				other=0.0,
			)


def _bwd_attn_kernel_call(
	bias,
	softmax_scale,
	blocksize_q,
	blocksize_k,
	residual,
	Do,
):
	(o, l, query, key, value) = residual
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
	Dq = jnp.zeros_like(a=query, dtype=query.dtype, device=get_sharding(query))
	Dk = jnp.zeros_like(a=key, dtype=key.dtype, device=get_sharding(key))
	Dv = jnp.zeros_like(a=value, dtype=value.dtype, device=get_sharding(value))
	stride_bb, stride_bh, stride_bm, stride_bn = (
		apply_stride(bias.shape) if HAVE_BIAS else (0, 0, 0, 0)
	)
	query = apply_stride(query)
	key = apply_stride(key)
	value = apply_stride(value)
	seqlen_q_rounded = math.ceil(seqlen_q / blocksize_q) * blocksize_q
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
		o,
		Do,
		delta,
		query.sride(0),
		query.sride(2),
		query.sride(1),
		query.sride(0),
		query.sride(2),
		query.sride(1),
		nheads,
		headdim,
		seqlen_q,
		seqlen_q_rounded,
		out_shape=[
			jax.ShapeDtypeStruct(
				shape=delta.shape,
				dtype=delta.dtype,
				sharding=delta.sharding,
			)
		],
		input_output_aliases={2: 0},
		grid=lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch, nheads),
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
		query,
		key,
		value,
		bias if bias is not None else False,
		Do,
		Dq,
		Dk,
		Dv,
		l,
		delta,
		softmax_scale,
		query.stride(0),
		query.stride(2),
		query.stride(1),
		key.stride(0),
		key.stride(2),
		key.stride(1),
		value.stride(0),
		value.stride(2),
		value.stride(1),
		stride_bb,
		stride_bh,
		stride_bm,
		query.stride(0),
		query.stride(2),
		query.stride(1),
		query.stride(0),
		query.stride(2),
		query.stride(1),
		key.stride(0),
		key.stride(2),
		key.stride(1),
		value.stride(0),
		value.stride(2),
		value.stride(1),
		seqlen_q,
		seqlen_k,
		seqlen_q_rounded,
		headdim,
		nheads,
		kernel=_bwd_attn_kernel,
		grid=lambda META: (triton.cdiv(seqlen_k, META["BLOCK_N"]), batch * nheads),
		out_shape=[
			jax.ShapeDtypeStruct(shape=Dq.shape, dtype=Dq.dtype, sharding=Dq.sharding),
			jax.ShapeDtypeStruct(shape=Dk.shape, dtype=Dk.dtype, sharding=Dk.sharding),
			jax.ShapeDtypeStruct(shape=Dv.shape, dtype=Dv.dtype, sharding=Dv.sharding),
		],
		input_output_aliases={5: 0, 6: 1, 7: 2},
		**metaparams,
	)

	return Dq, Dk, Dv


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
	return o, (o, l, query, key, value)


@functools.partial(custom_vjp, nondiff_argnums=[3, 4, 5, 6])
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
	B, H, S, D = 2, 32, 251, 128
	blocksize_k = 64
	blocksize_q = 128
	q = jax.nn.initializers.normal(2)(q_key, (B, S, H, D), dtype=jnp.float16)
	k = jax.nn.initializers.normal(2)(k_key, (B, S, H, D), dtype=jnp.float16)
	v = jax.nn.initializers.normal(2)(v_key, (B, S, H, D), dtype=jnp.float16)
	b = (
		jnp.where(
			jrnd.randint(v_key, (B, H, S, S), 0, 4) > 2,
			jnp.finfo(jnp.float16).min,
			0,
		)
		if False
		else None
	)
	o = flash_attn2_gpu(
		q,
		k,
		v,
		b,
		blocksize_k=blocksize_k,
		blocksize_q=blocksize_q,
	)

	try:
		fo = flax.linen.attention.dot_product_attention(q, k, v, b)

		print(o[-1, -1, -1, :5])
		print(fo[-1, -1, -1, :5])
		print(jnp.allclose(o, fo, 0, 0.125))
	except Exception as er:
		print("Flax OOM", er)


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
	o = jax.grad(lambda *x: (flash_attn2_gpu(*x, blocksize_k=128, blocksize_q=64)).sum())(
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
	_test_forward()
	# _test_backward()
