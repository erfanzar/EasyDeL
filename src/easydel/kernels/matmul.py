# Implementation by @erfanzar,
# with a few bug fixes and adjustments.
from functools import partial
import os as _os


if bool(
	_os.environ.get("EASYDEL_AUTO", "true")
):  # Taking care of some optional GPU FLAGs
	_os.environ["XLA_FLAGS"] = (
		_os.environ.get("XLA_FLAGS", "") + " "
		"--xla_gpu_enable_triton_softmax_fusion=true \ "
		"--xla_gpu_triton_gemm_any=True \ "
		"--xla_gpu_enable_async_collectives=true \ "
		"--xla_gpu_enable_latency_hiding_scheduler=true \ "
		"--xla_gpu_enable_highest_priority_async_stream=true \ "
	)
	_os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
	_os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.99"
	_os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import jax
import jax.interpreters
import jax.interpreters.pxla
import jax.random
from fjformer import GenerateRNG
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.lib import xla_bridge

rng = GenerateRNG()


def _matmul_kernel_fwd(
	a_ref,
	b_ref,
	o_ref,
	*,
	blocksize_m,
	blocksize_k,
	blocksize_n,
	po_dtype,
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
		jnp.zeros((blocksize_m, blocksize_n), dtype=po_dtype),
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


def _call_matmul_kernel_fwd(
	A: jax.Array,
	B: jax.Array,
	*,
	blocksize_m: int = 32,
	blocksize_k: int = 128,
	blocksize_n: int = 32,
	po_dtype: jnp.dtype = jnp.float32,
	precision: jax.lax.PrecisionLike = None,
):
	# A(mk)@B(kn)=C(mn)
	assert A.ndim == 2 and B.ndim == 2, f"got {A.shape=} and {B.shape=}"
	assert (
		A.shape[1] == B.shape[0]
	), f"matmul can't be operated with these shapes {A.shape=} {B.shape=} "
	m, n = A.shape[0], B.shape[1]
	grid = (pl.cdiv(m, blocksize_m), pl.cdiv(n, blocksize_n))

	in_specs = [
		pl.BlockSpec(lambda *_: (0,) * A.ndim, A.shape),
		pl.BlockSpec(lambda *_: (0,) * B.ndim, B.shape),
	]
	platform = xla_bridge.get_backend().platform
	interpret = platform == "cpu"
	out_specs = pl.BlockSpec(lambda *_: (0,) * A.ndim, (m, n))
	return pl.pallas_call(
		f=partial(
			_matmul_kernel_fwd,
			blocksize_m=blocksize_m,
			blocksize_n=blocksize_n,
			blocksize_k=blocksize_k,
			po_dtype=po_dtype,
			precision=precision,
		),
		out_shape=jax.ShapeDtypeStruct(shape=(m, n), dtype=A.dtype),
		debug=False,
		interpret=interpret,
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


def _call_matmul_kernel_fwd_g(
	A: jax.Array,
	B: jax.Array,
	blocksize_m: int = 32,
	blocksize_k: int = 128,
	blocksize_n: int = 32,
	po_dtype: jnp.dtype = jnp.float32,
	precision: jax.lax.PrecisionLike = None,
):
	return _call_matmul_kernel_fwd(
		A=A,
		B=B,
		blocksize_m=blocksize_m,
		blocksize_k=blocksize_k,
		blocksize_n=blocksize_n,
		po_dtype=po_dtype,
		precision=precision,
	), (
		A,
		B,
	)


def _call_matmul_kernel_bwd(
	blocksize_m,
	blocksize_k,
	blocksize_n,
	po_dtype,
	precision,
	res,
	gin,
):
	# A(mk)@B(kn)=C(mn)

	(
		A,
		B,
	) = res

	gA = _call_matmul_kernel_fwd(
		A=gin,
		B=B.transpose(1, 0),
		blocksize_m=blocksize_m,
		blocksize_k=blocksize_k,
		blocksize_n=blocksize_n,
		po_dtype=po_dtype,
		precision=precision,
	)
	gB = _call_matmul_kernel_fwd(
		A=A.transpose(1, 0),
		B=gin,
		blocksize_m=blocksize_m,
		blocksize_k=blocksize_k,
		blocksize_n=blocksize_n,
		po_dtype=po_dtype,
		precision=precision,
	)
	return gA, gB


def matmul_kernel(
	A: jax.Array,
	B: jax.Array,
	*,
	blocksize_m: int = 32,
	blocksize_k: int = 128,
	blocksize_n: int = 32,
	po_dtype: jnp.dtype = jnp.float32,
	precision: jax.lax.PrecisionLike = None,
):
	return _m(
		A,
		B,
		blocksize_m,
		blocksize_k,
		blocksize_n,
		po_dtype,
		precision,
	)


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5, 6))
# @jax.custom_vjp
def _m(
	A: jax.Array,
	B: jax.Array,
	blocksize_m: int = 32,
	blocksize_k: int = 128,
	blocksize_n: int = 32,
	po_dtype: jnp.dtype = jnp.float32,
	precision: jax.lax.PrecisionLike = None,
):
	return _call_matmul_kernel_fwd_g(
		A=A,
		B=B,
		blocksize_m=blocksize_m,
		blocksize_k=blocksize_k,
		blocksize_n=blocksize_n,
		po_dtype=po_dtype,
		precision=precision,
	)[0]


_m.defvjp(_call_matmul_kernel_fwd_g, _call_matmul_kernel_bwd)


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


def matmul_benchmark():
	import time

	def benchmark_matmul(m, n, k, block_m, block_n, block_k, dtype=jnp.float32):
		# Compile the kernel with given block sizes
		kernel = matmul_kernel

		# Generate random matrices
		a = jax.random.normal(jax.random.PRNGKey(0), (m, k), dtype=dtype)
		b = jax.random.normal(jax.random.PRNGKey(1), (k, n), dtype=dtype)

		# Warm-up run
		kernel(
			a,
			b,
			blocksize_m=block_m,
			blocksize_k=block_k,
			blocksize_n=block_n,
			po_dtype=dtype,
		)

		# Timed run
		start = time.time()
		for _ in range(10):  # Run multiple times for more stable results
			kernel(
				a,
				b,
				blocksize_m=block_m,
				blocksize_k=block_k,
				blocksize_n=block_n,
				po_dtype=dtype,
			)
		jax.block_until_ready(
			kernel(
				a,
				b,
				blocksize_m=block_m,
				blocksize_k=block_k,
				blocksize_n=block_n,
				po_dtype=dtype,
			)
		)
		end = time.time()

		return (end - start) / 10

	def autotune_block_sizes(m, n, k, dtype=jnp.float32):
		best_time = float("inf")
		best_config = None

		for block_m in [16, 32, 64, 128]:
			for block_n in [16, 32, 64, 128]:
				for block_k in [16, 32, 64, 128]:
					try:
						time = benchmark_matmul(m, n, k, block_m, block_n, block_k, dtype)
						print(
							f"Configuration: block_m={block_m}, block_n={block_n}, "
							f"block_k={block_k} -> Time: {time:.4f} seconds"
						)
						if time < best_time:
							best_time = time
							best_config = (block_m, block_n, block_k)
							print(
								f"New best configuration found: block_m={block_m}, "
								f"block_n={block_n}, block_k={block_k} with Time: {best_time:.4f} seconds"
							)
					except Exception as e:  # noqa
						print(
							f"Skipping configuration due to OOM : block_m={block_m},"
							f" block_n={block_n}, block_k={block_k}"
						)
						pass

		return best_config, best_time

	# Example usage
	m, n, k = 1, 4096, 4096 * 4
	best_config, best_time = autotune_block_sizes(m, n, k, jnp.float16)
	print(f"Best configuration for {m}x{n}x{k} matmul: {best_config}")
	print(f"Best time: {best_time:.6f} seconds")


if __name__ == "__main__":
	# matmul_test()
	# matmul_grad_test()
	matmul_benchmark()
