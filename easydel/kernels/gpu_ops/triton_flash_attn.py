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
# improved from org oai triton implementation

# TODO: fix bwd kernel


import triton
import triton.language as tl
import triton.tools.experimental_descriptor


def is_hip():
	return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cuda():
	return triton.runtime.driver.active.get_current_target().backend == "cuda"


@triton.jit
def _attn_fwd_inner(
	acc,
	l_i,
	m_i,
	q,
	K_block_ptr,
	V_block_ptr,
	AM_block_ptr,
	B_block_ptr,
	start_m,
	qk_scale,
	BLOCK_M: tl.constexpr,
	HEAD_DIM: tl.constexpr,
	BLOCK_N: tl.constexpr,
	STAGE: tl.constexpr,
	offs_m: tl.constexpr,
	offs_n: tl.constexpr,
	QSeq: tl.constexpr,
	KSeq: tl.constexpr,
	fp8_v: tl.constexpr,
):
	if STAGE == 1:
		lo, hi = 0, start_m * BLOCK_M
	elif STAGE == 2:
		lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
		lo = tl.multiple_of(lo, BLOCK_M)
	else:
		lo, hi = 0, KSeq
	K_block_ptr = tl.advance(K_block_ptr, (0, lo))
	V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
	for start_n in range(lo, hi, BLOCK_N):
		start_n = tl.multiple_of(start_n, BLOCK_N)
		k = tl.load(K_block_ptr)
		qk = tl.dot(q, k)
		if STAGE == 2:
			mask = offs_m[:, None] >= (start_n + offs_n[None, :])
			qk = (qk * qk_scale) + tl.where(mask, 0, -1.0e6)
			m_ij = tl.maximum(m_i, tl.max(qk, 1))
			qk -= m_ij[:, None]
		elif B_block_ptr is not None:
			bias = tl.load(B_block_ptr)
			qk = (qk * qk_scale) + bias
			m_ij = tl.maximum(m_i, tl.max(qk, 1))
			qk -= m_ij[:, None]
			B_block_ptr = tl.advance(B_block_ptr, (0, BLOCK_N))
		elif AM_block_ptr is not None:
			attn_mask = tl.cast(tl.load(AM_block_ptr), tl.int1)
			qk = (qk * qk_scale) + tl.where(attn_mask, 0, -1.0e6)
			m_ij = tl.maximum(m_i, tl.max(qk, 1))
			qk -= m_ij[:, None]
			AM_block_ptr = tl.advance(AM_block_ptr, (0, BLOCK_N))
		else:
			m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
			qk = (qk * qk_scale) - m_ij[:, None]
		p = tl.math.exp2(qk)
		l_ij = tl.sum(p, 1)
		alpha = tl.math.exp2(m_i - m_ij)
		l_i = l_i * alpha + l_ij
		acc = acc * alpha[:, None]
		v = tl.load(V_block_ptr)
		if fp8_v:
			p = p.to(tl.float8e5)
		else:
			p = p.to(tl.float16)
		acc = tl.dot(p, v, acc)
		m_i = m_ij
		V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
		K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
	return acc, l_i, m_i


configs = [
	triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
	for BM in [64, 128]
	for BN in [32, 64]
	for s in ([1] if is_hip() else [3, 4, 7])
	for w in [4, 8]
]


def keep(conf):
	BLOCK_M = conf.kwargs["BLOCK_M"]
	BLOCK_N = conf.kwargs["BLOCK_N"]
	if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
		return False
	return True


@triton.autotune(
	list(filter(keep, configs)),
	key=["QSeq", "KSeq", "HEAD_DIM"],
)
@triton.jit
def _attn_fwd(
	Q,
	K,
	V,
	AM,
	B,
	sm_scale,
	M,
	Out,
	stride_qz,
	stride_qg,
	stride_qh,
	stride_qm,
	stride_qk,
	stride_kz,
	stride_kh,
	stride_kn,
	stride_kk,
	stride_vz,
	stride_vh,
	stride_vk,
	stride_vn,
	stride_az,
	stride_ag,
	stride_ah,
	stride_am,
	stride_an,
	stride_bz,
	stride_bg,
	stride_bh,
	stride_bm,
	stride_bn,
	stride_oz,
	stride_og,
	stride_oh,
	stride_om,
	stride_ok,
	stride_mz,
	stride_mg,
	stride_mh,
	stride_mm,
	Z,
	single_headed,
	QSeq,
	KSeq,
	QHEAD,
	KHEAD,
	HEAD_DIM: tl.constexpr,
	BLOCK_M: tl.constexpr,
	BLOCK_N: tl.constexpr,
	STAGE: tl.constexpr,
):
	tl.static_assert(BLOCK_N <= HEAD_DIM)
	start_m = tl.program_id(0)
	off_hz = tl.program_id(1)
	off_qa = tl.program_id(2)
	off_z = off_hz // KHEAD
	off_h = off_hz % KHEAD
	q_offset = off_z * stride_qz + off_qa * stride_qg + off_h * stride_qh
	k_offset = off_z * stride_kz + off_h * stride_kh
	v_offset = off_z * stride_vz + off_h * stride_vh

	Q_block_ptr = tl.make_block_ptr(
		base=Q + q_offset,
		shape=(QSeq, HEAD_DIM),
		strides=(stride_qm, stride_qk),
		offsets=(start_m * BLOCK_M, 0),
		block_shape=(BLOCK_M, HEAD_DIM),
		order=(1, 0),
	)
	v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
	V_block_ptr = tl.make_block_ptr(
		base=V + k_offset,
		shape=(KSeq, HEAD_DIM),
		strides=(stride_vk, stride_vn),
		offsets=(0, 0),
		block_shape=(BLOCK_N, HEAD_DIM),
		order=v_order,
	)
	K_block_ptr = tl.make_block_ptr(
		base=K + v_offset,
		shape=(HEAD_DIM, KSeq),
		strides=(stride_kk, stride_kn),
		offsets=(0, 0),
		block_shape=(HEAD_DIM, BLOCK_N),
		order=(0, 1),
	)
	if AM is not None:
		h_pos: tl.constexpr = 0 if single_headed else off_h * stride_ah + off_qa * stride_ag
		AM_block_ptr = tl.make_block_ptr(
			base=AM + (off_z * stride_az + h_pos),
			shape=(QSeq, KSeq),
			strides=(stride_am, stride_an),
			offsets=(start_m * BLOCK_M, 0),
			block_shape=(BLOCK_M, BLOCK_N),
			order=(1, 0),
		)
	else:
		AM_block_ptr = None
	if B is not None:
		h_pos: tl.constexpr = 0 if single_headed else off_h * stride_bh + off_qa * stride_bg
		B_block_ptr = tl.make_block_ptr(
			base=B + (off_z * stride_bz + h_pos),
			shape=(QSeq, KSeq),
			strides=(stride_bm, stride_bn),
			offsets=(start_m * BLOCK_M, 0),
			block_shape=(BLOCK_M, BLOCK_N),
			order=(1, 0),
		)
	else:
		B_block_ptr = None
	O_block_ptr = tl.make_block_ptr(
		base=Out + q_offset,
		shape=(QSeq, HEAD_DIM),
		strides=(stride_om, stride_ok),
		offsets=(start_m * BLOCK_M, 0),
		block_shape=(BLOCK_M, HEAD_DIM),
		order=(1, 0),
	)

	offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
	offs_n = tl.arange(0, BLOCK_N)
	m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
	l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
	acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
	qk_scale = sm_scale
	qk_scale *= 1.44269504
	q = tl.load(Q_block_ptr)
	if STAGE & 1:
		acc, l_i, m_i = _attn_fwd_inner(
			acc,
			l_i,
			m_i,
			q,
			K_block_ptr,
			V_block_ptr,
			AM_block_ptr,
			B_block_ptr,
			start_m,
			qk_scale,
			BLOCK_M,
			HEAD_DIM,
			BLOCK_N,
			4 - STAGE,
			offs_m,
			offs_n,
			QSeq,
			KSeq,
			V.dtype.element_ty == tl.float8e5,
		)
	if STAGE & 2:
		acc, l_i, m_i = _attn_fwd_inner(
			acc,
			l_i,
			m_i,
			q,
			K_block_ptr,
			V_block_ptr,
			AM_block_ptr,
			B_block_ptr,
			start_m,
			qk_scale,
			BLOCK_M,
			HEAD_DIM,
			BLOCK_N,
			2,
			offs_m,
			offs_n,
			QSeq,
			KSeq,
			V.dtype.element_ty == tl.float8e5,
		)
	m_i += tl.math.log2(l_i)
	acc = acc / l_i[:, None]
	M_block_ptr = tl.make_block_ptr(
		base=M + off_z * stride_mz + off_qa * stride_mg + off_h * stride_mh,
		shape=(QSeq,),
		strides=(stride_mm,),
		offsets=(start_m * BLOCK_M,),
		block_shape=(BLOCK_M,),
		order=(0,),
	)
	tl.store(M_block_ptr, m_i)
	tl.store(O_block_ptr, acc.to(Out.type.element_ty))


@triton.jit
def _attn_bwd_preprocess(
	OP,
	DO,
	Delta,
	stride_opz,
	stride_opg,
	stride_oph,
	stride_opm,
	stride_opk,
	stride_oz,
	stride_og,
	stride_oh,
	stride_om,
	stride_ok,
	stride_mz,
	stride_mg,
	stride_mh,
	stride_mm,
	QSeq: tl.constexpr,
	BLOCK_M: tl.constexpr,
	HEAD_DIM: tl.constexpr,
):
	off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
	off_hz = tl.program_id(1)
	off_n = tl.arange(0, HEAD_DIM)
	o = tl.load(
		OP + off_hz * HEAD_DIM * QSeq + off_m[:, None] * HEAD_DIM + off_n[None, :]
	)
	do = tl.load(
		DO + off_hz * HEAD_DIM * QSeq + off_m[:, None] * HEAD_DIM + off_n[None, :]
	).to(tl.float32)
	delta = tl.sum(o * do, axis=1)
	tl.store(Delta + off_hz * QSeq + off_m, delta)


@triton.jit
def _attn_bwd_dkdv(
	dk,
	dv,
	Q,
	k,
	v,
	sm_scale,
	DO,
	M,
	D,
	stride_tok,
	stride_d,
	H,
	N_CTX,
	BLOCK_M1: tl.constexpr,
	BLOCK_N1: tl.constexpr,
	HEAD_DIM: tl.constexpr,
	start_n,
	start_m,
	num_steps,
	MASK: tl.constexpr,
):
	offs_m = start_m + tl.arange(0, BLOCK_M1)
	offs_n = start_n + tl.arange(0, BLOCK_N1)
	offs_k = tl.arange(0, HEAD_DIM)
	qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
	do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
	tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
	curr_m = start_m
	step_m = BLOCK_M1
	for blk_idx in range(num_steps):
		qT = tl.load(qT_ptrs)
		offs_m = curr_m + tl.arange(0, BLOCK_M1)
		m = tl.load(M + offs_m)
		qkT = tl.dot(k, qT)
		pT = tl.math.exp2(qkT - m[None, :])
		if MASK:
			mask = offs_m[None, :] >= offs_n[:, None]
			pT = tl.where(mask, pT, 0.0)
		do = tl.load(do_ptrs)
		ppT = pT
		ppT = ppT.to(tl.float16)
		dv += tl.dot(ppT, do)
		Di = tl.load(D + offs_m)
		dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
		dsT = pT * (dpT - Di[None, :])
		dsT = dsT.to(tl.float16)
		dk += tl.dot(dsT, tl.trans(qT))
		curr_m += step_m
		qT_ptrs += step_m * stride_tok
		do_ptrs += step_m * stride_tok
	return dk, dv


@triton.jit
def _attn_bwd_dq(
	dq,
	q,
	K,
	V,
	do,
	m,
	D,
	stride_tok,
	stride_d,
	H,
	N_CTX,
	BLOCK_M2: tl.constexpr,
	BLOCK_N2: tl.constexpr,
	HEAD_DIM: tl.constexpr,
	start_m,
	start_n,
	num_steps,
	MASK: tl.constexpr,
):
	offs_m = start_m + tl.arange(0, BLOCK_M2)
	offs_n = start_n + tl.arange(0, BLOCK_N2)
	offs_k = tl.arange(0, HEAD_DIM)
	kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
	vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
	Di = tl.load(D + offs_m)
	tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
	curr_n = start_n
	step_n = BLOCK_N2
	for blk_idx in range(num_steps):
		kT = tl.load(kT_ptrs)
		vT = tl.load(vT_ptrs)
		qk = tl.dot(q, kT)
		p = tl.math.exp2(qk - m)
		if MASK:
			offs_n = curr_n + tl.arange(0, BLOCK_N2)
			mask = offs_m[:, None] >= offs_n[None, :]
			p = tl.where(mask, p, 0.0)
		dp = tl.dot(do, vT).to(tl.float32)
		ds = p * (dp - Di[:, None])
		ds = ds.to(tl.float16)
		dq += tl.dot(ds, tl.trans(kT))
		curr_n += step_n
		kT_ptrs += step_n * stride_tok
		vT_ptrs += step_n * stride_tok
	return dq


@triton.jit
def _attn_bwd(
	Q,
	K,
	V,
	sm_scale,
	DO,
	DQ,
	DK,
	DV,
	M,
	D,
	stride_z,
	stride_h,
	stride_tok,
	stride_d,
	H,
	N_CTX,
	BLOCK_M1: tl.constexpr,
	BLOCK_N1: tl.constexpr,
	BLOCK_M2: tl.constexpr,
	BLOCK_N2: tl.constexpr,
	BLK_SLICE_FACTOR: tl.constexpr,
	HEAD_DIM: tl.constexpr,
):
	LN2: tl.constexpr = 0.6931471824645996
	bhid = tl.program_id(2)
	off_chz = bhid * N_CTX
	adj = stride_h * (bhid % H) + stride_z * (bhid // H)
	pid = tl.program_id(0)

	Q += adj
	K += adj
	V += adj
	DO += adj
	DQ += adj
	DK += adj
	DV += adj
	M += off_chz
	D += off_chz
	offs_k = tl.arange(0, HEAD_DIM)

	start_n = pid * BLOCK_N1
	start_m = start_n

	MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
	offs_n = start_n + tl.arange(0, BLOCK_N1)

	dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
	dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

	k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
	v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

	num_steps = BLOCK_N1 // MASK_BLOCK_M1

	dk, dv = _attn_bwd_dkdv(
		dk,
		dv,
		Q,
		k,
		v,
		sm_scale,
		DO,
		M,
		D,
		stride_tok,
		stride_d,
		H,
		N_CTX,
		MASK_BLOCK_M1,
		BLOCK_N1,
		HEAD_DIM,
		start_n,
		start_m,
		num_steps,
		MASK=True,
	)

	start_m += num_steps * MASK_BLOCK_M1
	num_steps = (N_CTX - start_m) // BLOCK_M1

	dk, dv = _attn_bwd_dkdv(
		dk,
		dv,
		Q,
		k,
		v,
		sm_scale,
		DO,
		M,
		D,
		stride_tok,
		stride_d,
		H,
		N_CTX,
		BLOCK_M1,
		BLOCK_N1,
		HEAD_DIM,
		start_n,
		start_m,
		num_steps,
		MASK=False,
	)

	dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
	tl.store(dv_ptrs, dv)

	dk *= sm_scale
	dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
	tl.store(dk_ptrs, dk)

	start_m = pid * BLOCK_M2
	end_n = start_m + BLOCK_M2

	MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
	offs_m = start_m + tl.arange(0, BLOCK_M2)

	q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
	dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
	do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

	m = tl.load(M + offs_m)
	m = m[:, None]

	num_steps = BLOCK_M2 // MASK_BLOCK_N2
	dq = _attn_bwd_dq(
		dq,
		q,
		K,
		V,
		do,
		m,
		D,
		stride_tok,
		stride_d,
		H,
		N_CTX,
		BLOCK_M2,
		MASK_BLOCK_N2,
		HEAD_DIM,
		start_m,
		end_n - num_steps * MASK_BLOCK_N2,
		num_steps,
		MASK=True,
	)
	end_n -= num_steps * MASK_BLOCK_N2
	num_steps = end_n // BLOCK_N2
	dq = _attn_bwd_dq(
		dq,
		q,
		K,
		V,
		do,
		m,
		D,
		stride_tok,
		stride_d,
		H,
		N_CTX,
		BLOCK_M2,
		BLOCK_N2,
		HEAD_DIM,
		start_m,
		end_n - num_steps * BLOCK_N2,
		num_steps,
		MASK=False,
	)
	dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
	dq *= LN2
	tl.store(dq_ptrs, dq)
