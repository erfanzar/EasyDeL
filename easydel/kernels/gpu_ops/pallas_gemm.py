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


# Implementation by @erfanzar,
# with a few bug fixes and adjustments.

import typing as tp
from functools import partial

import jax
import jax.interpreters
import jax.interpreters.pxla
import jax.random
from fjformer import GenerateRNG
from jax import numpy as jnp
from jax.experimental import pallas as pl

PLATFORM = jax.extend.backend.get_backend().platform
INTERPRET = PLATFORM == "cpu"
rng = GenerateRNG()

# GPU KERNEL


def _gpu_matmul_kernel_fwd(
	a_ref,
	b_ref,
	o_ref,
	*,
	blocksize_m,
	blocksize_k,
	blocksize_n,
	prod_dtype,
	precision,
):
	row_id, col_id = pl.program_id(0), pl.program_id(1)

	m, n = a_ref.shape[0], b_ref.shape[1]
	col_slice = pl.dslice(col_id * blocksize_n, blocksize_n)
	row_slice = pl.dslice(row_id * blocksize_m, blocksize_m)
	a_mask_i = (row_id * blocksize_m + jnp.arange(blocksize_m) < a_ref.shape[0])[:, None]
	b_mask_j = (col_id * blocksize_n + jnp.arange(blocksize_n) < b_ref.shape[1])[None, :]
	ij = jnp.arange(blocksize_k)

	def body(start_i, carry_i):
		o_p = carry_i
		a_mask_j = (start_i * blocksize_k + ij < jnp.arange(blocksize_k))[None, :] < m
		b_mask_i = (start_i * blocksize_k + ij < jnp.arange(blocksize_k))[:, None] < n
		inner_slice = pl.dslice(start_i * blocksize_k, blocksize_k)

		a = pl.load(
			a_ref,
			(row_slice, inner_slice),
			mask=a_mask_i & a_mask_j,
			other=0.0,
		)

		b = pl.load(
			b_ref,
			(inner_slice, col_slice),
			mask=b_mask_i & b_mask_j,
			other=0.0,
		)
		return jnp.dot(a, b, precision=precision) + o_p

	o = jax.lax.fori_loop(
		0,
		pl.cdiv(b_ref.shape[0], blocksize_k),
		body,
		jnp.zeros((blocksize_m, blocksize_n), dtype=prod_dtype),
	)
	omi = (row_id * blocksize_m + jnp.arange(blocksize_m) < o_ref.shape[0])[:, None]
	omj = (col_id * blocksize_n + jnp.arange(blocksize_n) < o_ref.shape[1])[None, :]
	o_mask = omi & omj
	pl.store(
		o_ref,
		(
			pl.dslice(row_id * blocksize_m, blocksize_m),
			pl.dslice(col_id * blocksize_n, blocksize_n),
		),
		o.astype(o_ref.dtype),
		mask=o_mask,
	)


def _get_compiler_params(
	blocksize_m: tp.Optional[int],
	blocksize_k: tp.Optional[int],
	blocksize_n: tp.Optional[int],
	dtype: jnp.dtype,
):
	params = None
	platform = jax.extend.backend.get_backend().platform
	dtype_size = jnp.dtype(dtype).itemsize
	if platform == "gpu":
		dtype_size = jnp.dtype(dtype).itemsize

		# Calculate shared memory usage
		shared_mem = 2 * blocksize_k * (blocksize_m + blocksize_n) * dtype_size

		# Improved num_warps calculation
		num_threads = blocksize_m * blocksize_n
		num_warps = min(max(num_threads // 32, 1), 8)

		# Adjust num_warps based on shared memory usage and block size
		if shared_mem > 32768:  # 32 KB shared memory limit for many GPUs
			num_warps = min(num_warps, 4)
		elif max(blocksize_m, blocksize_n) >= 128:
			num_warps = max(num_warps, 4)

		# num_stages calculation (as before)
		l2_cache_size = 1024 * 1024
		working_set_size = shared_mem * 2
		num_stages = min(max(l2_cache_size // working_set_size, 2), 5)

		if blocksize_k <= 32:
			num_stages = min(num_stages, 2)
		elif blocksize_k <= 64:
			num_stages = min(num_stages, 3)

		params = dict(triton=dict(num_stages=num_stages, num_warps=num_warps))

	return params


def get_best_block_size(A, B):
	# A is assumed to be of shape (m, k) and B is of shape (k, n)
	m, k = A.shape[0], A.shape[1]
	n = B.shape[1]

	# Initialize block sizes
	bm, bk, bn = 16, 16, 16

	# Adjust block size for m
	if m > 1000:
		bm = 32
	if m > 2000:
		bm = 64
	if m > 8000:
		bm = 128

	# Adjust block size for k
	if k > 1000:
		bk = 32
	if k > 2000:
		bk = 64
	if k > 8000:
		bk = 128

	# Adjust block size for n
	if n > 1000:
		bn = 32  # bn maxes out at 32

	return bm, bk, bn


@partial(
	jax.jit,
	static_argnames=[
		"blocksize_m",
		"blocksize_k",
		"blocksize_n",
		"prod_dtype",
		"precision",
	],
)
def _call_gpu_matmul_kernel_fwd(
	A: jax.Array,
	B: jax.Array,
	blocksize_m: tp.Optional[int],
	blocksize_k: tp.Optional[int],
	blocksize_n: tp.Optional[int],
	prod_dtype: jnp.dtype,
	precision: jax.lax.PrecisionLike = None,
):
	# A(mk)@B(kn)=C(mn)
	assert A.ndim == 2 and B.ndim == 2, f"got {A.shape=} and {B.shape=}"
	assert (
		A.shape[1] == B.shape[0]
	), f"matmul can't be operated with these shapes {A.shape=} {B.shape=} "
	m, n = A.shape[0], B.shape[1]
	pbm, pbk, pbn = get_best_block_size(A, B)

	blocksize_m = blocksize_m or pbm
	blocksize_n = blocksize_n or pbn
	blocksize_k = blocksize_k or pbk

	grid = (pl.cdiv(m, blocksize_m), pl.cdiv(n, blocksize_n))
	in_specs = [
		pl.BlockSpec(A.shape, lambda *_: (0,) * A.ndim),
		pl.BlockSpec(B.shape, lambda *_: (0,) * B.ndim),
	]

	out_specs = pl.BlockSpec((m, n), lambda *_: (0,) * A.ndim)
	return pl.pallas_call(
		f=partial(
			_gpu_matmul_kernel_fwd,
			blocksize_m=blocksize_m,
			blocksize_n=blocksize_n,
			blocksize_k=blocksize_k,
			prod_dtype=prod_dtype,
			precision=precision,
		),
		out_shape=jax.ShapeDtypeStruct(shape=(m, n), dtype=A.dtype),
		debug=False,
		interpret=INTERPRET,
		grid=grid,
		in_specs=in_specs,
		out_specs=out_specs,
		compiler_params=_get_compiler_params(
			blocksize_m=blocksize_m,
			blocksize_n=blocksize_n,
			blocksize_k=blocksize_k,
			dtype=A.dtype,
		),
		name="gpu_matmul_kernel_fwd",
	)(A, B)


def _call_gpu_matmul_kernel_fwd_residual(
	A: jax.Array,
	B: jax.Array,
	blocksize_m: tp.Optional[int],
	blocksize_k: tp.Optional[int],
	blocksize_n: tp.Optional[int],
	prod_dtype: jnp.dtype,
	precision: jax.lax.PrecisionLike = None,
):
	return _call_gpu_matmul_kernel_fwd(
		A=A,
		B=B,
		blocksize_m=blocksize_m,
		blocksize_k=blocksize_k,
		blocksize_n=blocksize_n,
		prod_dtype=prod_dtype,
		precision=precision,
	), (
		A,
		B,
	)


def _call_gpu_matmul_kernel_bwd(
	blocksize_m,
	blocksize_k,
	blocksize_n,
	prod_dtype,
	precision,
	res,
	gO,
):
	# A(mk)@B(kn)=C(mn)

	(
		A,
		B,
	) = res

	gA = _call_gpu_matmul_kernel_fwd(
		A=gO,
		B=B.transpose(1, 0),
		blocksize_m=blocksize_m,
		blocksize_k=blocksize_k,
		blocksize_n=blocksize_n,
		prod_dtype=prod_dtype,
		precision=precision,
	)
	gB = _call_gpu_matmul_kernel_fwd(
		A=A.transpose(1, 0),
		B=gO,
		blocksize_m=blocksize_m,
		blocksize_k=blocksize_k,
		blocksize_n=blocksize_n,
		prod_dtype=prod_dtype,
		precision=precision,
	)
	return gA, gB


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5, 6))
def gpu_matmul(
	A: jax.Array,
	B: jax.Array,
	blocksize_m: tp.Optional[int],
	blocksize_k: tp.Optional[int],
	blocksize_n: tp.Optional[int],
	prod_dtype: jnp.dtype,
	precision: jax.lax.PrecisionLike = None,
):
	return _call_gpu_matmul_kernel_fwd(
		A=A,
		B=B,
		blocksize_m=blocksize_m,
		blocksize_k=blocksize_k,
		blocksize_n=blocksize_n,
		prod_dtype=prod_dtype,
		precision=precision,
	)


gpu_matmul.defvjp(_call_gpu_matmul_kernel_fwd_residual, _call_gpu_matmul_kernel_bwd)

__all__ = ["gpu_matmul"]
