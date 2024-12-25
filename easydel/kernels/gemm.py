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

# import os
# import sys

# sys.path.append(
# 	os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")
# )

import logging
import typing as tp
from dataclasses import dataclass
from functools import partial

import jax
import jax.interpreters
import jax.interpreters.pxla
import jax.random
import numpy as np
from fjformer import GenerateRNG
from jax import lax
from jax import numpy as jnp
from jax.lax import PrecisionLike

from .gpu_ops.triton_gemm import gemm as triton_gemm
from .tpu_ops.pallas_gemm import pallas_gemm

PLATFORM = jax.extend.backend.get_backend().platform
INTERPRET = PLATFORM == "cpu"
rng = GenerateRNG()


def gemm(
	A: jax.Array,
	B: jax.Array,
	*,
	blocksize_m: tp.Optional[int] = None,
	blocksize_k: tp.Optional[int] = None,
	blocksize_n: tp.Optional[int] = None,
	precision: PrecisionLike = None,
	**_,
):
	if PLATFORM == "gpu":
		return triton_gemm(A, B)
	elif PLATFORM == "tpu":
		org_dtype = A.dtype
		A = A.astype(jnp.promote_types(jnp.bfloat16, A.dtype))
		B = B.astype(jnp.promote_types(jnp.bfloat16, B.dtype))
		return pallas_gemm(
			A,
			B,
			blocksize_m,
			blocksize_k,
			blocksize_n,
			precision,
		).astype(org_dtype)
	elif PLATFORM == "cpu":
		return jax.lax.batch_matmul(A, B, precision=precision)
	else:
		raise NotImplementedError(
			f"`gemm` is not implemented for request platform {PLATFORM}"
		)


def custom_dot_general_kernel(
	lhs: jnp.ndarray,
	rhs: jnp.ndarray,
	dimension_numbers: tp.Optional[
		tp.Tuple[
			tp.Tuple[tp.Sequence[int], tp.Sequence[int]],
			tp.Tuple[tp.Sequence[int], tp.Sequence[int]],
		]
	] = None,
	precision=None,
	preferred_element_type=None,
	*args,
	**kwargs,
):
	if preferred_element_type is None:
		preferred_element_type = rhs.dtype

	if dimension_numbers is None:
		raise ValueError(
			"dimension_numbers must be provided for general tensor contractions"
		)

	((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)) = dimension_numbers

	# Helper function to reshape inputs to 2D based on contract and batch dimensions
	def reshape_for_contraction(x, contract_dims, batch_dims):
		other_dims = [
			i for i in range(x.ndim) if i not in contract_dims and i not in batch_dims
		]
		perm = list(batch_dims) + other_dims + list(contract_dims)
		x = jnp.transpose(x, perm)
		batch_shape = [int(x.shape[i]) for i in range(len(batch_dims))]
		other_shape = [
			int(x.shape[i]) for i in range(len(batch_dims), x.ndim - len(contract_dims))
		]
		contract_shape = tuple(
			int(x.shape[i]) for i in range(x.ndim - len(contract_dims), x.ndim)
		)
		return (
			x.reshape(
				-1,
				np.prod(other_shape).astype("i4"),
				np.prod(contract_shape).astype("i4"),
			),
			batch_shape,
			other_shape,
		)

	# Reshape lhs and rhs for contraction
	lhs_reshaped, lhs_batch_shape, lhs_other_shape = reshape_for_contraction(
		lhs, lhs_contract, lhs_batch
	)
	rhs_reshaped, rhs_batch_shape, rhs_other_shape = reshape_for_contraction(
		rhs, rhs_contract, rhs_batch
	)

	# Ensure batch dimensions are compatible
	if lhs_batch_shape != rhs_batch_shape:
		raise ValueError("Batch dimensions must match for batched matrix multiplication")

	# Perform batched matrix multiplication using vmap
	result_3d = jax.vmap(gemm)(lhs_reshaped, jnp.transpose(rhs_reshaped, (0, 2, 1)))

	# Reshape result back to the original batch and output dimensions
	final_shape = lhs_batch_shape + lhs_other_shape + rhs_other_shape
	return result_3d.reshape(final_shape).astype(preferred_element_type)


def replace_dot_general_with_gemm():
	jax.lax.dot_general = custom_dot_general_kernel


def matmul_test():
	print("RES TEST")
	intermediate_size = 2048
	hidden_size = 512
	dtype = jnp.float16
	A = jax.nn.initializers.normal(0.02, dtype=dtype)(
		rng.rng,
		(hidden_size, intermediate_size),
	)
	B = jax.nn.initializers.normal(0.02, dtype=dtype)(
		rng.rng,
		(intermediate_size, hidden_size),
	)
	y_ = triton_gemm(A, B)
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
		rng.rng,
		(hidden_size, intermediate_size),
	)
	B = jax.nn.initializers.normal(0.02, dtype=dtype)(
		rng.rng,
		(intermediate_size, hidden_size),
	)
	g = jax.grad(lambda x, e: jnp.sum(x @ e))(A, B)
	g_ = jax.grad(lambda x, e: jnp.sum(triton_gemm(x, e)))(A, B)
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


def _analyze_matmul(
	m: int,
	k: int,
	n: int,
	block_m,
	block_n,
	block_k,
	dtype: jnp.dtype,
	mm_func,
):
	from tabulate import tabulate  # type:ignore # noqa

	x = jnp.ones((m, k), dtype=dtype)
	y = jnp.ones((k, n), dtype=dtype)
	try:
		time = _benchmark(mm_func)(x, y)
	except Exception as e:
		time = 1e6
		if "RESOURCE_EXHAUSTED" in str(e):
			logging.warning(
				f"OOM error for configuration: block_m={block_m}, block_n={block_n}, block_k={block_k}"
			)
		else:
			logging.error(
				f"Unexpected error for configuration: block_m={block_m}, block_n={block_n}, block_k={block_k}: {str(e)}"
			)

	try:
		time_org = _benchmark(jnp.matmul)(x, y)
	except Exception as e:
		time_org = 1e6
		if "RESOURCE_EXHAUSTED" in str(e):
			logging.warning("OOM error for native matmul")
		else:
			logging.error(
				f"Unexpected error for configuration: block_m={block_m}, block_n={block_n}, block_k={block_k}: {str(e)}"
			)
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


def matmul_benchmark(unused_args=None):
	BLOCK_SIZES = [1] if PLATFORM == "gpu" else [128, 256, 512, 1024]  # GPU is autotuned
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
	) -> tp.Tuple[tp.Tuple[int, int, int], float]:
		best_time_config = None
		best_flops_config = None

		for block_m in BLOCK_SIZES:
			for block_n in BLOCK_SIZES:
				for block_k in BLOCK_SIZES:
					time, flops = _analyze_matmul(
						block_m=block_m,
						m=m,
						block_k=block_k,
						k=k,
						block_n=block_n,
						n=n,
						dtype=dtype,
						mm_func=partial(
							gemm,
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

		return (
			best_time_config.block_m,
			best_time_config.block_n,
			best_time_config.block_k,
		), best_time_config.time

	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s - %(levelname)s - %(message)s",
	)
	# Define a list of matrix sizes to benchmark
	matrix_sizes = [
		(1024, 1024, 1024),
		(2048, 2048, 2048),
		(4096, 4096, 4096),
		(8192, 8192, 8192),
		(10240, 10240, 10240),
		(12288, 12288, 12288),
		(14336, 14336, 14336),
		(16384, 16384, 16384),
		(18432, 18432, 18432),
	]

	# Run the benchmark for each matrix size
	best_configs = {}
	for m, k, n in matrix_sizes:
		if PLATFORM != "gpu":
			logging.info(f"\nBenchmarking matrix multiplication: ({m} x {k}) * ({k} x {n})")
		best_config, best_time = autotune_block_sizes(m, n, k, dtype=jnp.float32)

		if PLATFORM != "gpu":
			logging.info(
				f"Best configuration ({m}x{k}x{n}): block_m={best_config[0]}, "
				f"block_n={best_config[1]}, block_k={best_config[2]}"
			)
			best_configs[f"{m}x{k}x{n}"] = best_config
			logging.info(f"Best time: {best_time:.6f} seconds")

	if PLATFORM != "gpu":
		print(best_configs)


def test_dot_general_replacer():
	print("Example 1: Standard matrix multiplication")
	lhs1 = jnp.arange(6).reshape(2, 3)
	rhs1 = jnp.arange(6).reshape(3, 2)

	result1_custom = custom_dot_general_kernel(
		lhs1,
		rhs1,
		dimension_numbers=(((1,), (0,)), ((), ())),
	)
	result1_original = lax.dot_general(
		lhs1,
		rhs1,
		dimension_numbers=(((1,), (0,)), ((), ())),
	)

	print("Custom Result:\n", result1_custom)
	print("Original Result:\n", result1_original)
	assert jnp.allclose(
		result1_custom,
		result1_original,
		atol=0.125,
		rtol=0,
	), "Test 1 failed: Results do not match!"

	# Example 2: Batched matrix multiplication
	print("\nExample 2: Batched matrix multiplication")
	lhs2 = jnp.arange(24).reshape(2, 3, 4)
	rhs2 = jnp.arange(24).reshape(2, 4, 3)

	result2_custom = custom_dot_general_kernel(
		lhs2,
		rhs2,
		dimension_numbers=(((2,), (1,)), ((0,), (0,))),
	)
	result2_original = lax.dot_general(
		lhs2,
		rhs2,
		dimension_numbers=(((2,), (1,)), ((0,), (0,))),
	)

	print("Custom Result:\n", result2_custom)
	print("Original Result:\n", result2_original)
	assert jnp.allclose(
		result2_custom,
		result2_original,
		atol=0.125,
		rtol=0,
	), "Test 2 failed: Results do not match!"

	# Example 3: Explicit dimension numbers
	print("\nExample 3: Explicit dimension numbers")
	lhs3 = jnp.arange(24).reshape(2, 3, 4)
	rhs3 = jnp.arange(24).reshape(4, 3, 2)

	result3_custom = custom_dot_general_kernel(
		lhs3,
		rhs3,
		dimension_numbers=(((2,), (0,)), ((0,), (2,))),
	)
	result3_original = lax.dot_general(
		lhs3,
		rhs3,
		dimension_numbers=(((2,), (0,)), ((0,), (2,))),
	)

	print("Custom Result:\n", result3_custom)
	print("Original Result:\n", result3_original)
	assert jnp.allclose(
		result3_custom,
		result3_original,
		atol=0.125,
		rtol=0,
	), "Test 3 failed: Results do not match!"

	# Example 4: Vector dot product
	print("\nExample 4: Vector dot product")
	lhs4 = jnp.arange(3)
	rhs4 = jnp.arange(3)

	result4_custom = custom_dot_general_kernel(
		lhs4,
		rhs4,
		dimension_numbers=(((0,), (0,)), ((), ())),
	)
	result4_original = lax.dot_general(
		lhs4,
		rhs4,
		dimension_numbers=(((0,), (0,)), ((), ())),
	)

	print("Custom Result:\n", result4_custom)
	print("Original Result:\n", result4_original)
	assert jnp.allclose(
		result4_custom,
		result4_original,
		atol=0.125,
		rtol=0,
	), "Test 4 failed: Results do not match!"
	print("\nExample 5: Complex tensor contraction")
	lhs5 = jax.random.normal(jax.random.key(0), (128 * 2 * 2,)).reshape(2, -1, 128)
	rhs5 = jax.random.normal(jax.random.key(1), (128 * 2,)).reshape(128, -1)

	result5_original = lax.dot_general(
		lhs5,
		rhs5,
		dimension_numbers=(((2,), (0,)), ((), ())),
	)
	result5_custom = custom_dot_general_kernel(
		lhs5,
		rhs5,
		dimension_numbers=(((2,), (0,)), ((), ())),
	)

	print("Custom Result:\n", result5_custom)
	print("Original Result:\n", result5_original)
	assert jnp.allclose(
		result5_custom,
		result5_original,
		atol=0.125,
		rtol=0,
	), "Test 5 failed: Results do not match!"
	print("ALL Tests are passed!")


if __name__ == "__main__":
	matmul_test()
	matmul_grad_test()
	test_dot_general_replacer()
	# from absl import app
	# app.run(matmul_benchmark)
