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


# impl GQA ATTN by @erfanzar

import functools
import math
import os
import typing as tp

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"


import chex
import jax
import numpy as np
import triton
from fjformer.jax_triton import triton_call
from jax import custom_vjp
from jax import numpy as jnp
from jax import random as jrnd
from triton import language as tl

FLASH_ATTN_BWD_ = True


def calculate_num_warps(
	head_dim: int,
	q_block_size: int = 0,
	k_block_size: int = 0,
) -> int:
	"""
	Calculate the number of warps based on head dimension and block sizes.

	Args:
	head_dim (int): The dimension of the attention head.
	q_block_size (int): The size of the query block. Default is 0.
	k_block_size (int): The size of the key block. Default is 0.

	Returns:
	int: The number of warps.
	"""
	if 16 < head_dim < 64:
		return 8
	elif 64 < head_dim < 128:
		return 4
	else:
		if q_block_size > 32 and k_block_size > 64:
			return 1
		elif q_block_size > 64 and k_block_size > 32:
			return 1
		else:
			return 4


def get_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
	"""Calculates strides for a given shape.

	Args:
		shape: Shape of the array.

	Returns:
		Tuple of strides.
	"""
	size = np.prod(shape)
	strides = []
	for s in shape:
		size = int(size // s)
		strides.append(size)
	return tuple(strides)


def get_sharding(arr: chex.Array):
	"""Gets the sharding of an array.

	Args:
		arr: Array to get sharding from.

	Returns:
		Sharding of the array.
	"""
	return getattr(arr, "sharding", None)


def check_shapes_and_dtypes(
	query: chex.Array,
	key: chex.Array,
	value: chex.Array,
	headdim: int,
):
	"""Checks the shapes and dtypes of the input arrays.

	Args:
		query: Query array.
		key: Key array.
		value: Value array.
		batch: Batch size.
		seqlen_k: Sequence length of the key.
		nheads: Number of heads.
		headdim: Head dimension.

	Raises:
		AssertionError: If the shapes or dtypes are not valid.
	"""
	chex.assert_equal(
		query.dtype, key.dtype, custom_message="Dtype mismatch between query and key."
	)
	chex.assert_equal(
		key.dtype, value.dtype, custom_message="Dtype mismatch between key and value."
	)
	if query.dtype not in [jnp.float16]:
		raise AssertionError("Only fp16 is supported.") from None

	if headdim > 256:
		raise AssertionError("Unsupported headdim value.")


def is_hip():
	try:
		return triton.runtime.driver.active.get_current_target().backend == "hip"
	except:  # noqa
		return True


fwd_configs = [
	triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
	for BM in [16, 32, 64, 128]
	for BN in [16, 32, 64, 128]
	for s in ([1] if is_hip() else [1, 3, 4, 7])
	for w in [2, 4, 8]
]


def fwd_keep(conf):
	BLOCK_M = conf.kwargs["BLOCK_M"]
	BLOCK_N = conf.kwargs["BLOCK_N"]
	if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
		return False
	return True


@triton.heuristics({"EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0})
@triton.jit
def _fwd_attention_kernel(
	Q,
	K,
	V,
	B,
	softmax_scale: tl.constexpr,
	stride_qb,
	stride_qh,
	stride_qg,
	stride_qm,
	stride_kb,
	stride_kh,
	stride_kn,
	stride_vb,
	stride_vh,
	stride_vn,
	stride_bb,
	stride_bh,
	stride_bg,
	stride_bm,
	stride_bn,
	stride_ob,
	stride_oh,
	stride_og,
	stride_om,
	stride_lb,
	stride_lh,
	stride_lg,
	headdim: tl.constexpr,
	num_kv_heads: tl.constexpr,
	num_groups: tl.constexpr,
	CQL: tl.constexpr,
	CKL: tl.constexpr,
	seqlen_q,
	seqlen_k,
	O,
	L,
	HAVE_BIAS: tl.constexpr,
	BIAS_SINGLE_HEAD: tl.constexpr,
	BLOCK_HEADDIM: tl.constexpr,
	EVEN_N: tl.constexpr,
	BLOCK_M: tl.constexpr,
	BLOCK_N: tl.constexpr,
):
	start_m, off_bh, off_gp = (
		tl.program_id(0),
		tl.program_id(1),
		tl.program_id(2),
	)
	off_h = off_bh % num_kv_heads
	off_b = off_bh // num_kv_heads

	if not EVEN_N:
		offs_n = tl.arange(0, BLOCK_N)

	Q_Block_ptr = tl.make_block_ptr(
		base=Q + (off_b * stride_qb + off_h * stride_qh + off_gp * stride_qg),
		shape=(seqlen_q, headdim),
		block_shape=(BLOCK_M, BLOCK_HEADDIM),
		strides=(stride_qm, 1),
		offsets=(start_m * BLOCK_M, 0),
		order=(0, 1),
	)
	O_Block_ptr = tl.make_block_ptr(
		base=O + (off_b * stride_ob + off_h * stride_oh + off_gp * stride_og),
		shape=(seqlen_q, headdim),
		block_shape=(BLOCK_M, BLOCK_HEADDIM),
		strides=(stride_om, 1),
		offsets=(start_m * BLOCK_M, 0),
		order=(0, 1),
	)
	L_Block_ptr = tl.make_block_ptr(
		base=L + (off_b * stride_lb + off_h * stride_lh + off_gp * stride_lg),
		shape=(seqlen_q,),
		strides=(1,),
		offsets=(start_m * BLOCK_M,),
		block_shape=(BLOCK_M,),
		order=(0,),
	)
	kv_stride = off_b * stride_kb + off_h * stride_kh
	K_Block_ptr = tl.make_block_ptr(
		base=K + kv_stride,
		shape=(headdim, seqlen_k),
		block_shape=(BLOCK_HEADDIM, BLOCK_N),
		strides=(1, stride_kn),
		offsets=(0, 0),
		order=(1, 0),
	)
	V_Block_ptr = tl.make_block_ptr(
		base=V + kv_stride,
		shape=(seqlen_k, headdim),
		block_shape=(BLOCK_N, BLOCK_HEADDIM),
		strides=(stride_vn, 1),
		offsets=(0, 0),
		order=(0, 1),
	)
	q = tl.load(Q_Block_ptr, boundary_check=(0, 1))
	softmax_scale = softmax_scale.to(tl.float32)
	if HAVE_BIAS:
		bias_h_pos: tl.constexpr = (
			0 if BIAS_SINGLE_HEAD else off_h * stride_bh + off_gp * stride_bg
		)
		B_Block_ptr = tl.make_block_ptr(
			base=B + (off_b * stride_bb + bias_h_pos),
			shape=(seqlen_q, seqlen_k),
			block_shape=(BLOCK_M, BLOCK_N),
			strides=(stride_bm, stride_bn),
			offsets=(start_m * BLOCK_M, 0),
			order=(0, 1),
		)
	lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
	max_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
	acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
	for j in range(0, seqlen_k, BLOCK_N):
		j = tl.multiple_of(j, BLOCK_N)
		k = tl.load(K_Block_ptr, boundary_check=(0, 1))
		qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
		qk += tl.dot(q, k) * softmax_scale
		if not EVEN_N:
			qk += tl.where((j + offs_n)[None, :] < seqlen_k, 0, float("-inf")).to(tl.float32)
		if HAVE_BIAS:
			b = tl.load(B_Block_ptr, boundary_check=(0, 1)).to(tl.float32)
			B_Block_ptr = tl.advance(B_Block_ptr, (0, BLOCK_N))
			qk = qk + b
			max_ij = tl.maximum(tl.max(qk, 1), lse_i)
			p = tl.exp(qk - max_ij[:, None])
		else:
			max_ij = tl.maximum(tl.max(qk, 1), lse_i)
			p = tl.exp(qk - max_ij[:, None])
		l_ij = tl.sum(p, 1)
		acc_o_scale = tl.exp(max_i - max_ij)
		acc_o = acc_o * acc_o_scale[:, None]
		v = tl.load(V_Block_ptr, boundary_check=(0, 1))
		acc_o += tl.dot(p.to(v.dtype), v)
		max_i = max_ij
		lse_i = max_ij + tl.log(tl.exp(lse_i - max_ij) + l_ij)
		K_Block_ptr = tl.advance(K_Block_ptr, (0, BLOCK_N))
		V_Block_ptr = tl.advance(V_Block_ptr, (BLOCK_N, 0))

	o_scale = tl.exp(max_i - lse_i)
	acc_o = acc_o * o_scale[:, None]
	tl.store(L_Block_ptr, lse_i, boundary_check=(0,))
	tl.store(O_Block_ptr, acc_o.to(q.dtype), boundary_check=(0, 1))


try:
	_fwd_attention_kernel = triton.autotune(
		list(filter(fwd_keep, fwd_configs)),
		key=["CQL", "CKL", "HAVE_BIAS", "BIAS_SINGLE_HEAD", "BLOCK_HEADDIM"],
	)(_fwd_attention_kernel)
except ModuleNotFoundError:
	...


def _fwd_attention_kernel_call(
	query: tp.Optional[chex.Array],
	key: tp.Optional[chex.Array],
	value: tp.Optional[chex.Array],
	bias: tp.Optional[chex.Array] = None,
	softmax_scale: tp.Optional[float] = None,
):
	"""Calls the Triton kernel for the forward pass of the attention mechanism.

	Args:
		query: Query array.
		key: Key array.
		value: Value array.
		bias: Bias array.
		softmax_scale: Scaling factor for the softmax function.

	Returns:
		Tuple of the output array and the log-sum-exp array.
	"""
	batch, seqlen_q, num_q_heads, headdim = query.shape
	_, seqlen_k, num_kv_heads, _ = key.shape
	num_groups = num_q_heads // num_kv_heads
	query = query.reshape(
		batch,
		seqlen_q,
		num_kv_heads,
		num_groups,
		headdim,
	)
	if bias is not None:
		if bias.shape[1] == 1:
			bias = bias.reshape(
				batch,
				1,
				1,
				seqlen_q,
				seqlen_k,
			)
		else:
			bias = bias.reshape(
				batch,
				num_kv_heads,
				num_groups,
				seqlen_q,
				seqlen_k,
			)
		HAVE_BIAS = True
		stride_bb, stride_bh, stride_bg, stride_bm, stride_bn = get_strides(bias.shape)
	else:
		HAVE_BIAS = False
		stride_bb, stride_bh, stride_bg, stride_bm, stride_bn = (0, 0, 0, 0, 0)

	BIAS_SINGLE_HEAD = True if bias is None else (True if bias.shape[1] == 1 else False)
	check_shapes_and_dtypes(
		query=query,
		key=key,
		value=value,
		headdim=headdim,
	)
	softmax_scale = softmax_scale or 1.0 / math.sqrt(headdim)
	BLOCK_HEADDIM = max(triton.next_power_of_2(headdim), 16)
	stride_lb, stride_lh, stride_lg, stride_lm = get_strides(
		(batch, num_kv_heads, num_groups, seqlen_q)
	)
	metaparams = dict(
		BIAS_SINGLE_HEAD=BIAS_SINGLE_HEAD,
		HAVE_BIAS=HAVE_BIAS,
		BLOCK_HEADDIM=BLOCK_HEADDIM,
		BLOCK_M=64,
		BLOCK_N=64,
		num_stages=1,
		num_warps=8,
	)

	stride_qb, stride_qm, stride_qh, stride_qg, stride_qd = get_strides(query.shape)
	stride_kb, stride_kn, stride_kh, stride_kd = get_strides(key.shape)
	stride_vb, stride_vn, stride_vh, stride_vd = get_strides(value.shape)
	out, lse = triton_call(
		query,
		key,
		value,
		bias if bias is not None else jnp.zeros((1,), jnp.float16),
		softmax_scale,
		stride_qb,
		stride_qh,
		stride_qg,
		stride_qm,
		stride_kb,
		stride_kh,
		stride_kn,
		stride_vb,
		stride_vh,
		stride_vn,
		stride_bb,
		stride_bh,
		stride_bg,
		stride_bm,
		stride_bn,
		stride_qb,
		stride_qh,
		stride_qg,
		stride_qm,
		stride_lb,
		stride_lh,
		stride_lg,
		headdim,
		num_kv_heads,
		num_groups,
		seqlen_q // 64,
		seqlen_k // 64,
		seqlen_q,
		seqlen_k,
		kernel=_fwd_attention_kernel,
		out_shape=[
			jax.ShapeDtypeStruct(query.shape, query.dtype, sharding=get_sharding(query)),
			jax.ShapeDtypeStruct((batch, num_kv_heads, num_groups, seqlen_q), jnp.float32),
		],
		grid=lambda META: (
			triton.cdiv(seqlen_q, META["BLOCK_M"]),
			batch * num_kv_heads,
			num_groups,
		),
		name="triton::ops::_fwd_attn_kernel",
		**metaparams,
	)
	return out.reshape(batch, seqlen_q, num_q_heads, headdim), lse


@triton.jit
def _bwd_do_attention_kernel(
	O,
	Do,
	De,
	stride_ob,
	stride_om,
	stride_oh,
	stride_og,
	stride_dob,
	stride_dom,
	stride_doh,
	stride_dog,
	stride_deb,
	stride_deh,
	stride_deg,
	num_kv_heads,
	num_groups,
	headdim,
	seqlen_q,
	BLOCK_M: tl.constexpr,
	BLOCK_HEADDIM: tl.constexpr,
):
	off_q = tl.program_id(0)
	off_bh = tl.program_id(1)
	off_gp = tl.program_id(2)
	off_b = off_bh // num_kv_heads
	off_h = off_bh % num_kv_heads
	offs_m = off_q * BLOCK_M + tl.arange(0, BLOCK_M)
	offs_d = tl.arange(0, BLOCK_HEADDIM)
	o_ptrs = (
		O
		+ off_b * stride_ob
		+ off_h * stride_oh
		+ off_gp * stride_og
		+ offs_m[:, None] * stride_om
		+ offs_d[None, :]
	)
	do_ptrs = (
		Do
		+ off_b * stride_dob
		+ off_h * stride_doh
		+ off_gp * stride_dog
		+ offs_m[:, None] * stride_dom
		+ offs_d[None, :]
	)
	o = tl.load(
		o_ptrs,
		mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
		other=0.0,
	).to(tl.float32)
	do = tl.load(
		do_ptrs,
		mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
		other=0.0,
	).to(tl.float32)
	delta = tl.sum(o * do, axis=1)
	tl.store(
		De + off_b * stride_deb + off_h * stride_deh + off_gp * stride_deg + offs_m,
		delta,
		mask=offs_m < seqlen_q,
	)


@triton.heuristics(
	{
		"EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
		"EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
		"EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
	}
)
@triton.jit
def _bwd_attention_kernel(
	Q,
	K,
	V,
	B,
	Do,
	L,
	D,
	softmax_scale: tl.constexpr,
	stride_qb,
	stride_qm,
	stride_qh,
	stride_qg,
	stride_kb,
	stride_kn,
	stride_kh,
	stride_vb,
	stride_vn,
	stride_vh,
	stride_bb,
	stride_bh,
	stride_bg,
	stride_bm,
	stride_dob,
	stride_dom,
	stride_doh,
	stride_dog,
	stride_dqb,
	stride_dqm,
	stride_dqh,
	stride_dqg,
	stride_dkb,
	stride_dkn,
	stride_dkh,
	stride_dvb,
	stride_dvn,
	stride_dvh,
	stride_lb,
	stride_lh,
	stride_lg,
	seqlen_q,
	seqlen_k,
	headdim,
	num_kv_heads,
	num_groups,
	Dq,
	Dk,
	Dv,
	HAVE_BIAS: tl.constexpr,
	BIAS_SINGLE_HEAD: tl.constexpr,
	BLOCK_HEADDIM: tl.constexpr,
	EVEN_M: tl.constexpr,
	EVEN_N: tl.constexpr,
	EVEN_HEADDIM: tl.constexpr,
	BLOCK_M: tl.constexpr,
	BLOCK_N: tl.constexpr,
):
	start_n, off_bh, off_gp = (
		tl.program_id(0),
		tl.program_id(1),
		tl.program_id(2),
	)
	softmax_scale = softmax_scale.to(tl.float32)
	off_h = off_bh % num_kv_heads
	off_b = off_bh // num_kv_heads
	offs_qm = tl.arange(0, BLOCK_M)
	offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
	offs_m = tl.arange(0, BLOCK_M)
	offs_d = tl.arange(0, BLOCK_HEADDIM)
	l_ptrs = L + (off_b * stride_lb + off_h * stride_lh + offs_qm + off_gp * stride_lg)
	d_ptrs = D + (off_b * stride_lb + off_h * stride_lh + offs_qm + off_gp * stride_lg)
	q_ptrs = (
		Q
		+ (off_b * stride_qb + off_h * stride_qh + off_gp * stride_qg)
		+ (offs_qm[:, None] * stride_qm + offs_d[None, :])
	)
	k_ptrs = (
		K
		+ (off_b * stride_kb + off_h * stride_kh)
		+ (offs_n[:, None] * stride_kn + offs_d[None, :])
	)
	v_ptrs = (
		V
		+ (off_b * stride_vb + off_h * stride_vh)
		+ (offs_n[:, None] * stride_vn + offs_d[None, :])
	)
	do_ptrs = (
		Do
		+ (off_b * stride_dob + off_h * stride_doh + off_gp * stride_dog)
		+ (offs_qm[:, None] * stride_dom + offs_d[None, :])
	)
	dq_ptrs = (
		Dq
		+ (off_b * stride_dqb + off_h * stride_dqh + off_gp * stride_dqg)
		+ (offs_qm[:, None] * stride_dqm + offs_d[None, :])
	)
	if HAVE_BIAS:
		bias_h_pos: tl.constexpr = (
			0 if BIAS_SINGLE_HEAD else off_h * stride_bh + off_gp * stride_bg
		)
		b_ptrs = (
			B
			+ (off_b * stride_bb + bias_h_pos)
			+ (offs_qm[:, None] * stride_bm + offs_n[None, :])
		)
	dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
	dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
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

	num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
	for start_m in range(0, num_block_m * BLOCK_M, BLOCK_M):
		start_m = tl.multiple_of(start_m, BLOCK_M)
		offs_m_curr = start_m + offs_m
		q = tl.load(
			q_ptrs,
			mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
			other=0.0,
		)
		qk = tl.dot(q, k.T) * softmax_scale
		if not EVEN_N:
			qk = tl.where(offs_n[None, :] < seqlen_k, qk, float("-inf"))

		if HAVE_BIAS:
			bias = tl.load(
				b_ptrs,
				mask=(offs_m_curr[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k),
				other=0.0,
			).to(tl.float32)
			qk = qk + bias
		lse_i = tl.load(l_ptrs + start_m, mask=offs_m_curr < seqlen_q, other=0.0)

		p = tl.exp(qk - lse_i[:, None])
		do = tl.load(
			do_ptrs,
			mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
			other=0.0,
		)
		dv += tl.dot(p.to(do.dtype).T, do)
		dp = tl.dot(do, v.T)

		Di = tl.load(d_ptrs + start_m, mask=offs_m_curr < seqlen_q, other=0.0)
		ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)
		dk += tl.dot(ds.T, q)

		dq = tl.dot(ds, k)
		tl.atomic_add(
			dq_ptrs,
			dq,
			mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
		)
		# ADVANCE TO NEXT POINT
		dq_ptrs += BLOCK_M * stride_dqm
		q_ptrs += BLOCK_M * stride_qm
		do_ptrs += BLOCK_M * stride_dom
		if HAVE_BIAS:
			b_ptrs += BLOCK_M * stride_bm
	dv_ptrs = (
		Dv
		+ (off_b * stride_dvb + off_h * stride_dvh)
		+ (offs_n[:, None] * stride_dvn + offs_d[None, :])
	)
	dk_ptrs = (
		Dk
		+ (off_b * stride_dkb + off_h * stride_dkh)
		+ (offs_n[:, None] * stride_dkn + offs_d[None, :])
	)

	tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
	tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))


def _bwd_attention_kernel_call(
	softmax_scale: float,
	residual,
	Do: chex.Array,
):
	"""Calls the Triton kernel for the backward pass of the attention mechanism.

	Args:
		softmax_scale: Scaling factor for the softmax function.
		residual: Residual from the forward pass.
		Do: Output gradient array.

	Returns:
		Tuple of the gradients of the query, key, value, and bias arrays.
	"""
	(o, l, query, key, value, bias) = residual

	batch, seqlen_q, num_q_heads, headdim = query.shape
	_, seqlen_k, num_kv_heads, _ = key.shape
	num_groups = num_q_heads // num_kv_heads
	if num_groups > 2:
		raise NotImplementedError(
			"triton_gqa_flash_attn2 is not performing well for num groups over 2 please use FORCE_MHA"
		)

	if FLASH_ATTN_BWD_:
		query = query.reshape(batch, seqlen_q, num_kv_heads, num_groups, headdim)
		o = o.reshape(batch, seqlen_q, num_kv_heads, num_groups, headdim)
		Do = Do.reshape(batch, seqlen_q, num_kv_heads, num_groups, headdim)
	if bias is not None:
		if bias.shape[1] == 1:
			bias = bias.reshape(
				batch,
				1,
				1,
				seqlen_q,
				seqlen_k,
			)
		else:
			bias = bias.reshape(
				batch,
				num_kv_heads,
				num_groups,
				seqlen_q,
				seqlen_k,
			)
		HAVE_BIAS = True
		stride_bb, stride_bh, stride_bg, stride_bm, _ = get_strides(bias.shape)
	else:
		HAVE_BIAS = False
		stride_bb, stride_bh, stride_bg, stride_bm, _ = (0, 0, 0, 0, 0)

	BIAS_SINGLE_HEAD = True if bias is None else (True if bias.shape[1] == 1 else False)
	softmax_scale = softmax_scale or 1.0 / math.sqrt(headdim)
	if FLASH_ATTN_BWD_:
		assert headdim <= 256, "given headdim is not supported."
		assert query.dtype == key.dtype == value.dtype, "tensors must have the same dtype."
		assert query.dtype in [jnp.float16], "only support fp16."
		HAVE_BIAS = True if bias is not None else False
		BLOCK_HEADDIM = max(triton.next_power_of_2(headdim), 16)
		bwd_kernel_out_shapes = [
			jax.ShapeDtypeStruct(
				shape=query.shape,
				dtype=query.dtype,
				sharding=get_sharding(query),
			),
			jax.ShapeDtypeStruct(
				shape=key.shape,
				dtype=key.dtype,
				sharding=get_sharding(key),
			),
			jax.ShapeDtypeStruct(
				shape=value.shape,
				dtype=value.dtype,
				sharding=get_sharding(value),
			),
		]

		delta = jnp.empty_like(l)
		# BATCH  , SEQUENCE  , HEADS   , _
		stride_qb, stride_qm, stride_qh, stride_qg, _ = get_strides(query.shape)
		stride_kb, stride_kn, stride_kh, _ = get_strides(key.shape)
		stride_vb, stride_vn, stride_vh, _ = get_strides(value.shape)
		stride_ob, stride_om, stride_oh, stride_og, _ = get_strides(o.shape)

		# BATCH  , HEADS    , _
		stride_lb, stride_lh, stride_lg, _ = get_strides(l.shape)
		stride_deb, stride_deh, stride_deg, _ = get_strides(delta.shape)

		# BATCH   , SEQUENCE  , HEADS     , _
		stride_dqb, stride_dqm, stride_dqh, stride_dqg, _ = get_strides(query.shape)
		stride_dkb, stride_dkn, stride_dkh, _ = get_strides(key.shape)
		stride_dvb, stride_dvn, stride_dvh, _ = get_strides(value.shape)
		stride_dob, stride_dom, stride_doh, stride_dog, _ = get_strides(Do.shape)

		num_warps = 4 if headdim <= 64 else 8

		# kernel kwargs
		metaparams = dict(
			BLOCK_M=128,
			BLOCK_HEADDIM=BLOCK_HEADDIM,
			num_warps=num_warps,
			num_stages=1,
		)
		(delta,) = triton_call(
			o,
			Do,
			delta,
			stride_ob,
			stride_om,
			stride_oh,
			stride_og,
			stride_dob,
			stride_dom,
			stride_doh,
			stride_dog,
			stride_deb,
			stride_deh,
			stride_deg,
			num_kv_heads,
			num_groups,
			headdim,
			seqlen_q,
			out_shape=[
				jax.ShapeDtypeStruct(
					shape=delta.shape,
					dtype=delta.dtype,
					sharding=get_sharding(delta),
				)
			],
			input_output_aliases={2: 0},
			grid=lambda META: (
				triton.cdiv(seqlen_q, META["BLOCK_M"]),
				batch * num_kv_heads,
				num_groups,
			),
			kernel=_bwd_do_attention_kernel,
			name="triton::ops::_bwd_do_attention_kernel",
			**metaparams,
		)
		metaparams = dict(
			BLOCK_M=int(os.environ.get("BLOCKSIZE_M_FLASH_ATTN", 64)),
			BLOCK_N=int(os.environ.get("BLOCKSIZE_N_FLASH_ATTN", 64)),
			num_warps=num_warps,
			num_stages=1,
			BLOCK_HEADDIM=BLOCK_HEADDIM,
			HAVE_BIAS=HAVE_BIAS,
			BIAS_SINGLE_HEAD=BIAS_SINGLE_HEAD,
		)

		Dq, Dk, Dv = triton_call(
			query,
			key,
			value,
			bias if bias is not None else jnp.zeros((1,), jnp.float16),
			Do,
			l,
			delta,
			softmax_scale,
			stride_qb,
			stride_qm,
			stride_qh,
			stride_qg,
			stride_kb,
			stride_kn,
			stride_kh,
			stride_vb,
			stride_vn,
			stride_vh,
			stride_bb,
			stride_bh,
			stride_bg,
			stride_bm,
			stride_dob,
			stride_dom,
			stride_doh,
			stride_dog,
			stride_dqb,
			stride_dqm,
			stride_dqh,
			stride_dqg,
			stride_dkb,
			stride_dkn,
			stride_dkh,
			stride_dvb,
			stride_dvn,
			stride_dvh,
			stride_lb,
			stride_lh,
			stride_lg,
			seqlen_q,
			seqlen_k,
			headdim,
			num_kv_heads,
			num_groups,
			kernel=_bwd_attention_kernel,
			grid=lambda META: (
				triton.cdiv(seqlen_k, META["BLOCK_N"]),
				batch * num_kv_heads,
				num_groups,
			),
			out_shape=bwd_kernel_out_shapes,
			name="triton::ops::_bwd_attention_kernel",
			**metaparams,
		)

		return Dq.reshape(batch, seqlen_q, num_q_heads, headdim), Dk, Dv, None
	else:
		_, f_vjp = jax.vjp(
			functools.partial(_attn_refrence, softmax_scale=softmax_scale),
			query,
			key,
			value,
			bias,
		)
		return f_vjp(Do)


def _fwd_attention_kernel_call_with_residual(
	query: tp.Optional[chex.Array],
	key: tp.Optional[chex.Array],
	value: tp.Optional[chex.Array],
	bias: tp.Optional[chex.Array] = None,
	softmax_scale: tp.Optional[float] = None,
):
	"""Calls the Triton kernel for the forward pass of the attention mechanism and returns the residual.

	Args:
		query: Query array.
		key: Key array.
		value: Value array.
		bias: Bias array.
		softmax_scale: Scaling factor for the softmax function.

	Returns:
		Tuple of the output array and the residual.
	"""
	o, l = _fwd_attention_kernel_call(
		query=query,
		key=key,
		value=value,
		bias=bias,
		softmax_scale=softmax_scale,
	)
	return o, (o, l, query, key, value, bias)


@functools.partial(custom_vjp, nondiff_argnums=[4])
@functools.partial(jax.jit, static_argnums=[4])
def _flash_attn2_gqa(
	query: chex.Array,
	key: chex.Array,
	value: chex.Array,
	bias: tp.Optional[chex.Array] = None,
	softmax_scale: tp.Optional[float] = None,
) -> chex.Array:
	"""Computes the attention mechanism using the Triton kernel.

	Args:
		query: Query array of shape (batch, seq_len_q, num_heads, head_dim).
		key: Key array of shape (batch, seq_len_k, num_heads, head_dim).
		value: Value array of shape (batch, seq_len_k, num_heads, head_dim).
		bias: tp.Optional bias array of shape (batch, num_heads, seq_len_q, seq_len_k).
		softmax_scale: Scaling factor for the softmax function.

	Returns:
		Output array of shape (batch, seq_len_q, num_heads, head_dim).
	"""
	return _fwd_attention_kernel_call(
		query=query,
		key=key,
		value=value,
		bias=bias,
		softmax_scale=softmax_scale,
	)[0]


_flash_attn2_gqa.defvjp(
	_fwd_attention_kernel_call_with_residual,
	_bwd_attention_kernel_call,
)


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


triton_gqa_flash_attention2_gpu = _flash_attn2_gqa
__all__ = ["triton_gqa_flash_attention2_gpu"]


def _test_forward():
	"""Tests the forward pass of the attention mechanism."""
	q_key, k_key, v_key = jrnd.split(jrnd.PRNGKey(8), 3)
	q_key, k_key, v_key = jrnd.split(jrnd.PRNGKey(8), 3)
	B, QH, KVH, QS, KS, D = 1, 32, 8, 1024, 1024, 128
	q = jax.nn.initializers.normal(2)(q_key, (B, QS, QH, D), dtype=jnp.float16)
	k = jax.nn.initializers.normal(2)(k_key, (B, KS, KVH, D), dtype=jnp.float16)
	v = jax.nn.initializers.normal(2)(v_key, (B, KS, KVH, D), dtype=jnp.float16)
	b = (
		jnp.where(
			jrnd.randint(v_key, (B, 1, QS, KS), 0, 4) > 2,
			jnp.finfo(jnp.float16).min,
			0,
		)
		if True
		else None
	)
	print("QKV Allocated")
	co = triton_gqa_flash_attention2_gpu(q, k, v, b)
	print(co[-1, -1, -1, :5])
	fo = _attn_refrence(q, k, v, b)
	print(fo[-1, -1, -1, :5])
	print("Results are Close" if jnp.allclose(co, fo, 0, 0.125) else "Wrong results!")


def _test_backward():
	"""Tests the backward pass of the attention mechanism."""
	q_key, k_key, v_key = jrnd.split(jrnd.PRNGKey(8), 3)
	B, QH, KVH, QS, KS, D = 1, 32, 16, 1024, 1024, 128
	q = jax.nn.initializers.normal(2)(q_key, (B, QS, QH, D), dtype=jnp.float16)
	k = jax.nn.initializers.normal(2)(k_key, (B, KS, KVH, D), dtype=jnp.float16)
	v = jax.nn.initializers.normal(2)(v_key, (B, KS, KVH, D), dtype=jnp.float16)
	b = (
		jnp.where(
			jrnd.randint(v_key, (B, 1, QS, KS), 0, 4) > 2,
			jnp.finfo(jnp.float16).min,
			0,
		)
		if True
		else None
	)

	try:
		co = jax.grad(lambda *x: triton_gqa_flash_attention2_gpu(*x).sum())(q, k, v, b)
		print("Custom op backward pass gradients:")
		print(co[-1][-1, -1, :5])  # Print last 5 elements of last head of last batch
	except Exception as er:
		print(f"Custom op backward pass failed: {er}")
		co = None

	try:
		fo = jax.grad(lambda *x: _attn_refrence(*x).sum())(q, k, v, b)

		print(fo[-1, -1, -1, :5])  # Print last 5 elements of last head of last batch
	except Exception as e:
		print(f"Flax backward pass failed : {e}")
		fo = None
		exit()

	if fo is not None and co is not None:
		if jnp.allclose(co, fo, atol=0.125):
			print("Backward pass results are close.")
		else:
			print("Backward pass results differ significantly!")


if __name__ == "__main__":
	_test_forward()
	_test_forward()
	_test_forward()
	_test_forward()
	_test_forward()
	_test_backward()
