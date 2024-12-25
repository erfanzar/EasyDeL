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
from jax.experimental.pallas import tpu as pltpu

PLATFORM = jax.extend.backend.get_backend().platform
INTERPRET = PLATFORM == "cpu"
rng = GenerateRNG()


def get_best_block_size_tpu(A, B):
	# A is assumed to be of shape (m, k) and B is of shape (k, n)
	m, k = A.shape[0], A.shape[1]
	n = B.shape[1]

	# Initialize block sizes
	bm, bk, bn = 16, 16, 16

	# Adjust block size for m
	if m >= 1024:
		bm = 1024

	# Adjust block size for k
	if k > 128:
		bk = 128
	if k >= 2048:
		bk = 1024

	# Adjust block size for n
	if n >= 1024:
		bn = 1024
	if n >= 2048:
		bn = 256

	return bm, bk, bn


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


def _tpu_matmul_kernel_fwd(a_ref, b_ref, o_ref, ac_ref, *, precision, k_grid):
	@pl.when(pl.program_id(2) == 0)
	def _():
		ac_ref[...] = jnp.zeros_like(ac_ref)

	ac_ref[...] += jnp.dot(
		a_ref[...].astype(jnp.float32),
		b_ref[...].astype(jnp.float32),
		preferred_element_type=jnp.float32,
		precision=precision,
	)

	@pl.when(pl.program_id(2) == k_grid - 1)
	def _():
		o_ref[...] = ac_ref[...].astype(o_ref.dtype)


@partial(
	jax.jit,
	static_argnames=[
		"blocksize_m",
		"blocksize_k",
		"blocksize_n",
		"precision",
	],
)
def _call_tpu_matmul_kernel_fwd(
	A: jax.Array,
	B: jax.Array,
	*,
	blocksize_m: tp.Optional[int],
	blocksize_k: tp.Optional[int],
	blocksize_n: tp.Optional[int],
	precision: jax.lax.PrecisionLike = None,
):
	assert A.ndim == 2 and B.ndim == 2, f"got {A.shape=} and {B.shape=}"
	assert (
		A.shape[1] == B.shape[0]
	), f"matmul can't be operated with these shapes {A.shape=} {B.shape=} "
	m, k, n = A.shape[0], B.shape[0], B.shape[1]
	pbm, pbk, pbn = get_best_block_size(A, B)

	blocksize_m = blocksize_m or pbm
	blocksize_n = blocksize_n or pbn
	blocksize_k = blocksize_k or pbk

	grid = (
		pl.cdiv(m, blocksize_m),
		pl.cdiv(n, blocksize_n),
		pl.cdiv(k, blocksize_k),
	)

	in_specs = [
		pl.BlockSpec((blocksize_m, blocksize_k), lambda mi, ni, ki: (mi, ki)),
		pl.BlockSpec((blocksize_k, blocksize_n), lambda mi, ni, ki: (ki, ni)),
	]

	out_specs = pl.BlockSpec((blocksize_m, blocksize_n), lambda mi, ni, ki: (mi, ni))
	return pl.pallas_call(
		f=partial(_tpu_matmul_kernel_fwd, precision=precision, k_grid=grid[-1]),
		out_shape=jax.ShapeDtypeStruct(shape=(m, n), dtype=A.dtype),
		debug=False,
		interpret=INTERPRET,
		grid_spec=pltpu.PrefetchScalarGridSpec(
			num_scalar_prefetch=0,
			grid=grid,
			in_specs=in_specs,
			out_specs=out_specs,
			scratch_shapes=[pltpu.VMEM((blocksize_m, blocksize_n), jnp.float32)],
		),
		compiler_params=dict(
			mosaic=dict(dimension_semantics=("parallel", "parallel", "arbitrary"))
		),
		name="tpu_matmul_kernel_fwd",
	)(A, B)


def _call_tpu_matmul_kernel_fwd_residual(
	A: jax.Array,
	B: jax.Array,
	blocksize_m: tp.Optional[int],
	blocksize_k: tp.Optional[int],
	blocksize_n: tp.Optional[int],
	precision: jax.lax.PrecisionLike = None,
):
	return _call_tpu_matmul_kernel_fwd(
		A=A,
		B=B,
		blocksize_m=blocksize_m,
		blocksize_k=blocksize_k,
		blocksize_n=blocksize_n,
		precision=precision,
	), (A, B)


def _call_tpu_matmul_kernel_bwd(
	blocksize_m: tp.Optional[int],
	blocksize_k: tp.Optional[int],
	blocksize_n: tp.Optional[int],
	precision: jax.lax.PrecisionLike,
	res,
	gO,
):
	(A, B) = res
	gA = _call_tpu_matmul_kernel_fwd(
		A=gO,
		B=B.transpose(1, 0),
		blocksize_m=blocksize_m,
		blocksize_k=blocksize_k,
		blocksize_n=blocksize_n,
		precision=precision,
	)

	gB = _call_tpu_matmul_kernel_fwd(
		A=A.transpose(1, 0),
		B=gO,
		blocksize_m=blocksize_m,
		blocksize_k=blocksize_k,
		blocksize_n=blocksize_n,
		precision=precision,
	)
	return gA, gB


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5))
def pallas_gemm(
	A: jax.Array,
	B: jax.Array,
	blocksize_m: tp.Optional[int],
	blocksize_k: tp.Optional[int],
	blocksize_n: tp.Optional[int],
	precision: jax.lax.PrecisionLike = None,
):
	return _call_tpu_matmul_kernel_fwd(
		A=A,
		B=B,
		blocksize_m=blocksize_m,
		blocksize_k=blocksize_k,
		blocksize_n=blocksize_n,
		precision=precision,
	)


pallas_gemm.defvjp(_call_tpu_matmul_kernel_fwd_residual, _call_tpu_matmul_kernel_bwd)

__all__ = ["pallas_gemm"]
