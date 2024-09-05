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

import logging
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

import jax
import jax.interpreters
import jax.interpreters.pxla
import jax.random
from fjformer import GenerateRNG
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.lax import PrecisionLike

# from jax.experimental.pallas import gpu as pltr
from jax.lib import xla_bridge

PLATFORM = xla_bridge.get_backend().platform
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
	blocksize_m: int,
	blocksize_k: int,
	blocksize_n: int,
	dtype: jnp.dtype,
):
	params = None
	platform = xla_bridge.get_backend().platform
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
	blocksize_m: Optional[int],
	blocksize_k: Optional[int],
	blocksize_n: Optional[int],
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
	)(A, B)


def _call_gpu_matmul_kernel_fwd_residual(
	A: jax.Array,
	B: jax.Array,
	blocksize_m: Optional[int],
	blocksize_k: Optional[int],
	blocksize_n: Optional[int],
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
def _gpu_matmul(
	A: jax.Array,
	B: jax.Array,
	blocksize_m: Optional[int],
	blocksize_k: Optional[int],
	blocksize_n: Optional[int],
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


_gpu_matmul.defvjp(_call_gpu_matmul_kernel_fwd_residual, _call_gpu_matmul_kernel_bwd)

# TPU KERNEL


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
	blocksize_m: int,
	blocksize_k: int,
	blocksize_n: int,
	precision: jax.lax.PrecisionLike = None,
):
	assert A.ndim == 2 and B.ndim == 2, f"got {A.shape=} and {B.shape=}"
	assert (
		A.shape[1] == B.shape[0]
	), f"matmul can't be operated with these shapes {A.shape=} {B.shape=} "
	m, k, n = A.shape[0], B.shape[0], B.shape[1]

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
		interpret=False,
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
	)(A, B)


def _call_tpu_matmul_kernel_fwd_residual(
	A: jax.Array,
	B: jax.Array,
	blocksize_m: int,
	blocksize_k: int,
	blocksize_n: int,
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
	blocksize_m: int,
	blocksize_k: int,
	blocksize_n: int,
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
def _tpu_matmul(
	A: jax.Array,
	B: jax.Array,
	blocksize_m: int,
	blocksize_k: int,
	blocksize_n: int,
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


_tpu_matmul.defvjp(_call_tpu_matmul_kernel_fwd_residual, _call_tpu_matmul_kernel_bwd)


def matmul_kernel(
	A: jax.Array,
	B: jax.Array,
	*,
	blocksize_m: Optional[int] = None,
	blocksize_k: Optional[int] = None,
	blocksize_n: Optional[int] = None,
	prod_dtype: jnp.dtype = jnp.float32,
	precision: PrecisionLike = None,
):
	if PLATFORM in ["gpu", "cpu"]:
		return _gpu_matmul(
			A,
			B,
			blocksize_m,
			blocksize_k,
			blocksize_n,
			prod_dtype,
			precision,
		)
	elif PLATFORM in ["tpu", "cpu"]:
		org_dtype = A.dtype
		A = A.astype(jnp.promote_types(jnp.bfloat16, A.dtype))
		B = B.astype(jnp.promote_types(jnp.bfloat16, B.dtype))
		return _tpu_matmul(
			A,
			B,
			blocksize_m,
			blocksize_k,
			blocksize_n,
			precision,
		).astype(org_dtype)
	else:
		raise NotImplementedError(
			f"`matmul_kernel` is not implemented for request platform {PLATFORM}"
		)


def matmul_test():
	print("RES TEST")
	intermediate_size = 2048
	hidden_size = 512
	dtype = jnp.float16
	A = jax.nn.initializers.normal(0.02, dtype=dtype)(
		rng.rng, (hidden_size, intermediate_size)
	)
	B = jax.nn.initializers.normal(0.02, dtype=dtype)(
		rng.rng, (intermediate_size, hidden_size)
	)
	y_ = matmul_kernel(A, B)
	y = A @ B
	print(jnp.allclose(y_, y, atol=0.125, rtol=0))
	print(y[0, :5])
	print(y_[0, :5])
	print(y[-1, :5])
	print(y_[-1, :5])


def matmul_grad_test():
	print("GRAD TEST")
	intermediate_size = 2048
	hidden_size = 512
	dtype = jnp.float16
	A = jax.nn.initializers.normal(0.02, dtype=dtype)(
		rng.rng, (hidden_size, intermediate_size)
	)
	B = jax.nn.initializers.normal(0.02, dtype=dtype)(
		rng.rng, (intermediate_size, hidden_size)
	)
	g = jax.grad(lambda x, e: jnp.sum(x @ e))(A, B)
	g_ = jax.grad(lambda x, e: jnp.sum(matmul_kernel(x, e)))(A, B)
	print(jnp.allclose(g, g_, atol=0.125, rtol=0))
	print(g_[0])
	print(g[0])


def _matmul_flops(m: int, k: int, n: int):
	return 2 * m * k * n


def _matmul_membw(m: int, k: int, n: int, dtype: jnp.dtype):
	return (m * k + k * n + m * n) * jnp.dtype(dtype).itemsize


def _matmul_flops_intensity(m: int, k: int, n: int, dtype: jnp.dtype):
	flops = _matmul_flops(m, k, n)
	membw = _matmul_membw(m, k, n, dtype)
	return flops / membw


def _benchmark(f, ntrials: int = 100):
	import timeit

	def run(*args, **kwargs):
		jax.block_until_ready(f(*args, **kwargs))
		result = timeit.timeit(
			lambda: jax.block_until_ready(f(*args, **kwargs)), number=ntrials
		)
		time = result / ntrials
		return time

	return run


def _analyze_matmul(m: int, k: int, n: int, dtype: jnp.dtype, mm_func):
	from tabulate import tabulate

	x = jnp.ones((m, k), dtype=dtype)
	y = jnp.ones((k, n), dtype=dtype)
	time = _benchmark(mm_func)(x, y)
	time_org = _benchmark(jnp.matmul)(x, y)
	mm_flops = _matmul_flops(m, k, n) / time
	mm_flops_org = _matmul_flops(m, k, n) / time_org

	print(
		tabulate(
			[
				["Time (s)", f"{time:.6f}", f"{time_org:.6f}"],
				["FLOP/s", f"{mm_flops:.2e}", f"{mm_flops_org:.2e}"],
			],
			headers=[f"Metric ({m}x{k}x{n})", "PAL Matmul", "JAX Matmul"],
			tablefmt="grid",
		)
	)
	return time, mm_flops


@dataclass
class _MatMulConfig:
	block_m: int
	block_n: int
	block_k: int
	time: float
	flops: float


def matmul_benchmark():
	BLOCK_SIZES = [16, 32, 64, 128] if PLATFORM == "gpu" else [128, 256, 512, 1024]
	logging.basicConfig(
		level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
	)

	def log_configuration(config: _MatMulConfig, is_time_best: bool, is_flops_best: bool):
		log_message = f"Configuration: block_m={config.block_m}, block_n={config.block_n}, block_k={config.block_k}"
		if is_time_best:
			logging.info(f"New best time {log_message} - Time: {config.time:.6f} seconds")
		if is_flops_best:
			logging.info(f"New best FLOP/s {log_message} - FLOP/s: {config.flops:.2e}")

	def autotune_block_sizes(
		m: int, n: int, k: int, dtype: jnp.dtype = jnp.float32
	) -> Tuple[Tuple[int, int, int], float]:
		best_time_config = None
		best_flops_config = None

		for block_m in BLOCK_SIZES:
			for block_n in BLOCK_SIZES:
				for block_k in BLOCK_SIZES:
					try:
						time, flops = _analyze_matmul(
							m,
							k,
							n,
							dtype,
							partial(
								matmul_kernel,
								blocksize_m=block_m,
								blocksize_n=block_n,
								blocksize_k=block_k,
								prod_dtype=dtype,
								precision=None,
							),
						)
						current_config = _MatMulConfig(block_m, block_n, block_k, time, flops)

						is_time_best = best_time_config is None or time < best_time_config.time
						is_flops_best = best_flops_config is None or flops > best_flops_config.flops

						if is_time_best:
							best_time_config = current_config
						if is_flops_best:
							best_flops_config = current_config

						log_configuration(current_config, is_time_best, is_flops_best)

					except Exception as e:
						if "RESOURCE_EXHAUSTED" in str(e):
							logging.warning(
								f"OOM error for configuration: block_m={block_m}, block_n={block_n}, block_k={block_k}"
							)
						else:
							logging.error(
								f"Unexpected error for configuration: block_m={block_m}, block_n={block_n}, block_k={block_k}: {str(e)}"
							)

		return (
			best_time_config.block_m,
			best_time_config.block_n,
			best_time_config.block_k,
		), best_time_config.time

	logging.basicConfig(
		level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
	)
	# Define a list of matrix sizes to benchmark
	matrix_sizes = [
		(1024, 1024, 1024),
		(2048, 2048, 2048),
		(4096, 4096, 4096),
		(8192, 8192, 8192),
		(10240, 10240, 10240),
	]

	# Run the benchmark for each matrix size
	best_configs = {}
	for m, k, n in matrix_sizes:
		logging.info(f"\nBenchmarking matrix multiplication: ({m} x {k}) * ({k} x {n})")
		best_config, best_time = autotune_block_sizes(m, n, k, dtype=jnp.float32)
		logging.info(
			f"Best configuration ({m}x{k}x{n}): block_m={best_config[0]}, "
			f"block_n={best_config[1]}, block_k={best_config[2]}"
		)
		best_configs[f"{m}x{k}x{n}"] = best_config
		logging.info(f"Best time: {best_time:.6f} seconds")
	print(best_configs)


if __name__ == "__main__":
	# matmul_test()
	# matmul_grad_test()
	matmul_benchmark()
