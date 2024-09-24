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

import os
import sys

sys.path.append(
	os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../src")
)
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
from jax.lax import PrecisionLike
from jax.lib import xla_bridge

from easydel.kernels.gpu_ops.triton_gemm import gemm as triton_gemm
from easydel.kernels.tpu_ops.pallas_gemm import pallas_gemm

PLATFORM = xla_bridge.get_backend().platform
INTERPRET = PLATFORM == "cpu"
rng = GenerateRNG()


def gemm_kernel(
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
		return triton_gemm(A, B)
	elif PLATFORM in ["tpu", "cpu"]:
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
	else:
		raise NotImplementedError(
			f"`gemm_kernel` is not implemented for request platform {PLATFORM}"
		)


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
	from tabulate import tabulate

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
	) -> Tuple[Tuple[int, int, int], float]:
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
							gemm_kernel,
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
		level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
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


if __name__ == "__main__":
	matmul_test()
	matmul_grad_test()
	# from absl import app
	# app.run(matmul_benchmark)
