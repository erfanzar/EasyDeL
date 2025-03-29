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


import argparse
import functools
import time
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from eformer import escale as es
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as Ps

# Import the distributed matmul functions
from easydel.kernels.collective_matmul import (
	MatrixMultiplyMethod,
	create_distributed_matmul,
	prepare_matrix_for_all_gather,
	prepare_matrix_for_reduce_scatter,
)


class MatrixMultiplyBenchmark:
	"""
	Benchmark for distributed matrix multiplication methods.

	This class provides functionality to benchmark and compare different
	distributed matrix multiplication implementations across various
	matrix sizes and device configurations.
	"""

	def __init__(
		self,
		mesh_shape: Tuple[int, ...] = (1, 1, -1, 1, 1),
		mesh_axis: str = "tp",
		dtype: jnp.dtype = jnp.float32,
		warmup_iterations: int = 2,
		benchmark_iterations: int = 5,
	):
		"""
		Initialize the benchmark environment.

		Args:
		    mesh_shape: Shape of the device mesh to use for distributed computation
		    mesh_axis: The mesh axis name to use for parallelization
		    dtype: Data type for test matrices
		    warmup_iterations: Number of warmup iterations before timing
		    benchmark_iterations: Number of iterations to average timing over
		"""
		self.mesh_shape = mesh_shape
		self.mesh_axis = mesh_axis
		self.dtype = dtype
		self.warmup_iterations = warmup_iterations
		self.benchmark_iterations = benchmark_iterations

		# Create device mesh
		self.device_mesh = es.create_mesh(mesh_shape)

		# Initialize results storage
		self.results = {"standard": [], "all_gather": [], "reduce_scatter": []}
		self.matrix_sizes = []

	def _generate_test_matrices(
		self, m: int, k: int, n: int
	) -> Tuple[jax.Array, jax.Array]:
		"""
		Generate random test matrices of the specified dimensions.

		Args:
		    m: Number of rows in the left matrix
		    k: Inner dimension (columns in left, rows in right)
		    n: Number of columns in the right matrix

		Returns:
		    Tuple of (left_matrix, right_matrix)
		"""
		key1, key2 = jax.random.split(jax.random.PRNGKey(0))
		left_matrix = jax.random.uniform(key1, shape=(m, k), dtype=self.dtype)
		right_matrix = jax.random.uniform(key2, shape=(k, n), dtype=self.dtype)
		return left_matrix, right_matrix

	def _benchmark_standard_matmul(
		self,
		left_matrix: jax.Array,
		right_matrix: jax.Array,
	) -> float:
		"""
		Benchmark standard matrix multiplication.

		Args:
		    left_matrix: Left-hand side matrix
		    right_matrix: Right-hand side matrix

		Returns:
		    Average execution time in milliseconds
		"""
		# Compile the operation
		left_matrix_spec = Ps(("sp", "fsdp"), self.mesh_axis)
		right_matrix_spec = Ps(self.mesh_axis, ("sp", "fsdp"))

		matmul_fn = jax.jit(
			lambda x, y: x @ y,
			in_shardings=(
				NamedSharding(self.device_mesh, left_matrix_spec),
				NamedSharding(self.device_mesh, right_matrix_spec),
			),
			out_shardings=NamedSharding(self.device_mesh, Ps(("sp", "fsdp"), self.mesh_axis)),
		)

		# Warmup

		with self.device_mesh:
			left_matrix = es.with_sharding_constraint(left_matrix, left_matrix_spec)
			right_matrix = es.with_sharding_constraint(right_matrix, right_matrix_spec)

			for _ in range(self.warmup_iterations):
				result = matmul_fn(left_matrix, right_matrix)
				result.block_until_ready()

			# Benchmark
			start_time = time.time()
			for _ in range(self.benchmark_iterations):
				result = matmul_fn(left_matrix, right_matrix)
				result.block_until_ready()
		end_time = time.time()

		# Return average time in milliseconds
		return ((end_time - start_time) / self.benchmark_iterations) * 1000

	def _benchmark_distributed_matmul(
		self,
		left_matrix: jax.Array,
		right_matrix: jax.Array,
		method: MatrixMultiplyMethod,
	) -> float:
		"""
		Benchmark a distributed matrix multiplication method.

		Args:
		    left_matrix: Left-hand side matrix
		    right_matrix: Right-hand side matrix
		    method: The distributed matrix multiplication method to benchmark

		Returns:
		    Average execution time in milliseconds
		"""
		# Prepare matrices based on the method
		if method == MatrixMultiplyMethod.ALL_GATHER:
			# Prepare for all-gather
			sharded_right = jax.device_put(
				right_matrix,
				NamedSharding(self.device_mesh, Ps(("sp", "fsdp"), self.mesh_axis)),
			)
			prepared_right = prepare_matrix_for_all_gather(
				sharded_right, self.device_mesh, self.mesh_axis
			)
			in_specs = (
				Ps(("sp", "fsdp"), self.mesh_axis),
				Ps(("sp", "fsdp"), self.mesh_axis),
			)
		else:
			# Prepare for reduce-scatter
			sharded_right = jax.device_put(
				right_matrix, NamedSharding(self.device_mesh, Ps(self.mesh_axis, None))
			)
			prepared_right = prepare_matrix_for_reduce_scatter(
				sharded_right, self.device_mesh, self.mesh_axis
			)
			in_specs = (
				Ps(("sp", "fsdp"), self.mesh_axis),
				Ps(self.mesh_axis, ("sp", "fsdp")),
			)

		# Function to be executed on each device
		def distributed_matmul_wrapper(left, right, method, dims):
			return create_distributed_matmul(method, dims)(left, right)

		# Create the distributed function
		distributed_fn = jax.jit(
			jax.experimental.shard_map.shard_map(
				f=functools.partial(
					distributed_matmul_wrapper,
					method=method,
					dims=self.mesh_axis,
				),
				mesh=self.device_mesh,
				in_specs=in_specs,
				out_specs=Ps(("sp", "fsdp"), self.mesh_axis),
			),
		)
		with self.device_mesh:
			left_matrix = es.with_sharding_constraint(left_matrix, in_specs[0])
			right_matrix = es.with_sharding_constraint(right_matrix, in_specs[1])
		# Warmup
		for _ in range(self.warmup_iterations):
			result = distributed_fn(left_matrix, prepared_right)
			result.block_until_ready()

		# Benchmark
		start_time = time.time()
		for _ in range(self.benchmark_iterations):
			result = distributed_fn(left_matrix, prepared_right)
			result.block_until_ready()
		end_time = time.time()

		# Return average time in milliseconds
		return ((end_time - start_time) / self.benchmark_iterations) * 1000

	def run_benchmark(
		self,
		matrix_sizes: List[Tuple[int, int, int]],
	) -> Dict[str, List[float]]:
		"""
		Run benchmarks for all methods across specified matrix sizes.

		Args:
		    matrix_sizes: List of (m, k, n) tuples defining matrix dimensions to test

		Returns:
		    Dictionary with benchmark results
		"""
		self.matrix_sizes = matrix_sizes

		for m, k, n in matrix_sizes:
			print(f"\nBenchmarking matrices of size ({m}x{k}) @ ({k}x{n})...")

			# Generate test matrices
			left_matrix, right_matrix = self._generate_test_matrices(m, k, n)

			# Benchmark standard matrix multiplication
			standard_time = self._benchmark_standard_matmul(left_matrix, right_matrix)
			self.results["standard"].append(standard_time)
			print(f"  Standard MatMul: {standard_time:.2f} ms")

			# Benchmark all-gather distributed matrix multiplication
			all_gather_time = self._benchmark_distributed_matmul(
				left_matrix,
				right_matrix,
				MatrixMultiplyMethod.ALL_GATHER,
			)
			self.results["all_gather"].append(all_gather_time)
			print(f"  All-Gather MatMul: {all_gather_time:.2f} ms")

			# Benchmark reduce-scatter distributed matrix multiplication
			reduce_scatter_time = self._benchmark_distributed_matmul(
				left_matrix,
				right_matrix,
				MatrixMultiplyMethod.REDUCE_SCATTER,
			)
			self.results["reduce_scatter"].append(reduce_scatter_time)
			print(f"  Reduce-Scatter MatMul: {reduce_scatter_time:.2f} ms")

			# Calculate speedups
			ag_speedup = standard_time / all_gather_time
			rs_speedup = standard_time / reduce_scatter_time
			print(f"  All-Gather Speedup: {ag_speedup:.2f}x")
			print(f"  Reduce-Scatter Speedup: {rs_speedup:.2f}x")

		return self.results

	def plot_results(
		self, save_path: str = None, figsize: Tuple[int, int] = (12, 8)
	) -> None:
		"""
		Plot benchmark results.

		Args:
		    save_path: Path to save the plot image, or None to display
		    figsize: Figure size (width, height) in inches
		"""
		if not self.results["standard"]:
			print("No benchmark results to plot. Run benchmarks first.")
			return

		# Create labels for x-axis
		labels = [f"{m}x{k}x{n}" for m, k, n in self.matrix_sizes]

		# Create figure with two subplots
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

		# Plot execution times
		x = np.arange(len(labels))
		width = 0.25

		ax1.bar(x - width, self.results["standard"], width, label="Standard")
		ax1.bar(x, self.results["all_gather"], width, label="All-Gather")
		ax1.bar(x + width, self.results["reduce_scatter"], width, label="Reduce-Scatter")

		ax1.set_xlabel("Matrix Sizes (mxkxn)")
		ax1.set_ylabel("Execution Time (ms)")
		ax1.set_title("Matrix Multiplication Performance")
		ax1.set_xticks(x)
		ax1.set_xticklabels(labels, rotation=45)
		ax1.legend()
		ax1.grid(axis="y", linestyle="--", alpha=0.7)

		# Plot speedups
		ag_speedups = [
			s / a for s, a in zip(self.results["standard"], self.results["all_gather"])
		]
		rs_speedups = [
			s / r for s, r in zip(self.results["standard"], self.results["reduce_scatter"])
		]

		ax2.bar(x - width / 2, ag_speedups, width, label="All-Gather")
		ax2.bar(x + width / 2, rs_speedups, width, label="Reduce-Scatter")

		ax2.set_xlabel("Matrix Sizes (mxkxn)")
		ax2.set_ylabel("Speedup vs Standard (higher is better)")
		ax2.set_title("Matrix Multiplication Speedup")
		ax2.set_xticks(x)
		ax2.set_xticklabels(labels, rotation=45)
		ax2.legend()
		ax2.grid(axis="y", linestyle="--", alpha=0.7)

		# Add horizontal line at y=1 (baseline)
		ax2.axhline(y=1, color="r", linestyle="-", alpha=0.3)

		plt.tight_layout()

		if save_path:
			plt.savefig(save_path, dpi=300, bbox_inches="tight")
			print(f"Plot saved to {save_path}")
		else:
			plt.show()

	def print_summary(self) -> None:
		"""Print a summary of the benchmark results."""
		if not self.results["standard"]:
			print("No benchmark results to summarize. Run benchmarks first.")
			return

		print("\n===== Distributed Matrix Multiplication Benchmark Summary =====")
		print(f"Device Mesh Shape: {self.mesh_shape}")
		print(f"Parallelization Axis: {self.mesh_axis}")
		print(f"Data Type: {self.dtype}")
		print(f"Benchmark Iterations: {self.benchmark_iterations}")
		print("\nMatrix Sizes and Execution Times (milliseconds):")

		# Calculate averages
		avg_standard = sum(self.results["standard"]) / len(self.results["standard"])
		avg_all_gather = sum(self.results["all_gather"]) / len(self.results["all_gather"])
		avg_reduce_scatter = sum(self.results["reduce_scatter"]) / len(
			self.results["reduce_scatter"]
		)

		# Calculate average speedups
		avg_ag_speedup = avg_standard / avg_all_gather
		avg_rs_speedup = avg_standard / avg_reduce_scatter

		# Print results table
		print(
			"\n{:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format(
				"Matrix Size",
				"Standard",
				"All-Gather",
				"AG Speedup",
				"Reduce-Scatter",
				"RS Speedup",
			)
		)
		print("-" * 90)

		for i, (m, k, n) in enumerate(self.matrix_sizes):
			size_str = f"{m}x{k}x{n}"
			std_time = self.results["standard"][i]
			ag_time = self.results["all_gather"][i]
			rs_time = self.results["reduce_scatter"][i]
			ag_speedup = std_time / ag_time
			rs_speedup = std_time / rs_time

			print(
				"{:<15} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f}".format(
					size_str, std_time, ag_time, ag_speedup, rs_time, rs_speedup
				)
			)

		print("-" * 90)
		print(
			"{:<15} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f}".format(
				"AVERAGE",
				avg_standard,
				avg_all_gather,
				avg_ag_speedup,
				avg_reduce_scatter,
				avg_rs_speedup,
			)
		)
		print("\n=== End of Summary ===")


def main():
	"""Main function to run benchmarks from command line."""
	parser = argparse.ArgumentParser(
		description="Benchmark distributed matrix multiplication methods"
	)
	parser.add_argument(
		"--sizes",
		type=str,
		default="1024,1024,1024 2048,1024,2048 4096,1024,4096 8192,1024,8192 8192,4096,8192",
		help='Matrix sizes to benchmark in format "m,k,n m,k,n ..."',
	)
	parser.add_argument(
		"--warmup", type=int, default=2, help="Number of warmup iterations"
	)
	parser.add_argument(
		"--iterations", type=int, default=5, help="Number of benchmark iterations"
	)
	parser.add_argument(
		"--plot", type=str, default=None, help="Path to save benchmark plot"
	)
	parser.add_argument(
		"--dtype",
		type=str,
		choices=["float32", "float16", "bfloat16"],
		default="bfloat16",
		help="Data type for matrices",
	)
	args = parser.parse_args()

	# Parse matrix sizes
	matrix_sizes = []
	for size_str in args.sizes.split():
		m, k, n = map(int, size_str.split(","))
		matrix_sizes.append((m, k, n))

	# Set dtype
	if args.dtype == "float32":
		dtype = jnp.float32
	elif args.dtype == "float16":
		dtype = jnp.float16
	else:  # bfloat16
		dtype = jnp.bfloat16

	# Create and run benchmark
	benchmark = MatrixMultiplyBenchmark(
		dtype=dtype,
		warmup_iterations=args.warmup,
		benchmark_iterations=args.iterations,
	)

	benchmark.run_benchmark(matrix_sizes)
	benchmark.print_summary()

	if args.plot:
		benchmark.plot_results(save_path=args.plot)
	else:
		benchmark.plot_results()


if __name__ == "__main__":
	main()
