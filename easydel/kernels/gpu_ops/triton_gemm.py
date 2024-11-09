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


from functools import partial

import fjformer.jax_triton as jt
import jax
import triton
from jax import core as jcore
from jax import numpy as jnp
from jax.interpreters import batching, mlir, xla
from triton import language as tl


def _is_cuda():
	return triton.runtime.driver.active.get_current_target().backend == "cuda"


def _get_cuda_autotune_config():
	return [
		triton.Config(
			{
				"BLOCK_SIZE_M": 128,
				"BLOCK_SIZE_N": 256,
				"BLOCK_SIZE_K": 64,
				"GROUP_SIZE_M": 8,
			},
			num_stages=3,
			num_warps=8,
		),
		triton.Config(
			{
				"BLOCK_SIZE_M": 64,
				"BLOCK_SIZE_N": 256,
				"BLOCK_SIZE_K": 32,
				"GROUP_SIZE_M": 8,
			},
			num_stages=4,
			num_warps=4,
		),
		triton.Config(
			{
				"BLOCK_SIZE_M": 128,
				"BLOCK_SIZE_N": 128,
				"BLOCK_SIZE_K": 32,
				"GROUP_SIZE_M": 8,
			},
			num_stages=4,
			num_warps=4,
		),
		triton.Config(
			{
				"BLOCK_SIZE_M": 128,
				"BLOCK_SIZE_N": 64,
				"BLOCK_SIZE_K": 32,
				"GROUP_SIZE_M": 8,
			},
			num_stages=4,
			num_warps=4,
		),
		triton.Config(
			{
				"BLOCK_SIZE_M": 64,
				"BLOCK_SIZE_N": 128,
				"BLOCK_SIZE_K": 32,
				"GROUP_SIZE_M": 8,
			},
			num_stages=4,
			num_warps=4,
		),
		triton.Config(
			{
				"BLOCK_SIZE_M": 128,
				"BLOCK_SIZE_N": 32,
				"BLOCK_SIZE_K": 32,
				"GROUP_SIZE_M": 8,
			},
			num_stages=4,
			num_warps=4,
		),
		triton.Config(
			{
				"BLOCK_SIZE_M": 64,
				"BLOCK_SIZE_N": 32,
				"BLOCK_SIZE_K": 32,
				"GROUP_SIZE_M": 8,
			},
			num_stages=5,
			num_warps=2,
		),
		triton.Config(
			{
				"BLOCK_SIZE_M": 32,
				"BLOCK_SIZE_N": 64,
				"BLOCK_SIZE_K": 32,
				"GROUP_SIZE_M": 8,
			},
			num_stages=5,
			num_warps=2,
		),
	]


def _get_hip_autotune_config():
	return [
		triton.Config(
			{
				"BLOCK_SIZE_M": 128,
				"BLOCK_SIZE_N": 256,
				"BLOCK_SIZE_K": 16,
				"GROUP_SIZE_M": 1,
				"waves_per_eu": 2,
			},
			num_warps=4,
			num_stages=0,
		),
		triton.Config(
			{
				"BLOCK_SIZE_M": 256,
				"BLOCK_SIZE_N": 256,
				"BLOCK_SIZE_K": 16,
				"GROUP_SIZE_M": 4,
				"waves_per_eu": 2,
			},
			num_warps=8,
			num_stages=0,
		),
		triton.Config(
			{
				"BLOCK_SIZE_M": 128,
				"BLOCK_SIZE_N": 128,
				"BLOCK_SIZE_K": 32,
				"GROUP_SIZE_M": 1,
				"waves_per_eu": 2,
			},
			num_warps=8,
			num_stages=0,
		),
		triton.Config(
			{
				"BLOCK_SIZE_M": 64,
				"BLOCK_SIZE_N": 128,
				"BLOCK_SIZE_K": 32,
				"GROUP_SIZE_M": 8,
				"waves_per_eu": 3,
			},
			num_warps=4,
			num_stages=0,
		),
		triton.Config(
			{
				"BLOCK_SIZE_M": 64,
				"BLOCK_SIZE_N": 64,
				"BLOCK_SIZE_K": 32,
				"GROUP_SIZE_M": 1,
				"waves_per_eu": 8,
			},
			num_warps=4,
			num_stages=0,
		),
	]


def _get_autotune_config():
	try:
		if _is_cuda():
			return _get_cuda_autotune_config()
		else:
			return _get_hip_autotune_config()
	except:  # noqa
		return _get_cuda_autotune_config()


@triton.jit
def _triton_gemm(
	a_ptr,
	b_ptr,
	c_ptr,
	stride_am,
	stride_ak,
	stride_bk,
	stride_bn,
	stride_cm,
	stride_cn,
	M,
	N,
	K,
	BLOCK_SIZE_M: tl.constexpr,
	BLOCK_SIZE_N: tl.constexpr,
	BLOCK_SIZE_K: tl.constexpr,
	GROUP_SIZE_M: tl.constexpr,
):
	pid = tl.program_id(0)
	num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
	num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
	num_pid_in_group = GROUP_SIZE_M * num_pid_n
	group_id = pid // num_pid_in_group
	first_pid_m = group_id * GROUP_SIZE_M
	group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
	pid_m = first_pid_m + (pid % group_size_m)
	pid_n = (pid % num_pid_in_group) // group_size_m
	offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
	offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
	offs_k = tl.arange(0, BLOCK_SIZE_K)
	a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
	b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
	acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
	for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
		a_data = tl.load(a_ptrs, mask=offs_k[None, :] < (K - k * BLOCK_SIZE_K), other=0.0)
		b_data = tl.load(b_ptrs, mask=offs_k[:, None] < (K - k * BLOCK_SIZE_K), other=0.0)
		acc += tl.dot(a_data, b_data)
		a_ptrs += BLOCK_SIZE_K * stride_ak
		b_ptrs += BLOCK_SIZE_K * stride_bk
	acc = acc.to(dtype=tl.float16)
	offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
	offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
	c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
	tl.store(c_ptrs, acc, mask=c_mask)


try:
	_triton_gemm = triton.autotune(configs=_get_autotune_config(), key=["M", "N", "K"])(
		_triton_gemm
	)
except ModuleNotFoundError:
	...


@triton.jit
def _gemm_activation_kernel(
	a_ptr,
	b_ptr,
	c_ptr,
	M,
	N,
	K,
	stride_am,
	stride_ak,
	stride_bk,
	stride_bn,
	stride_cm,
	stride_cn,
	BLOCK_SIZE_M: tl.constexpr,
	BLOCK_SIZE_N: tl.constexpr,
	BLOCK_SIZE_K: tl.constexpr,
	GROUP_SIZE_M: tl.constexpr,
	activation: tl.constexpr,
):
	pid = tl.program_id(0)
	num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
	num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
	num_pid_in_group = GROUP_SIZE_M * num_pid_n
	group_id = pid // num_pid_in_group
	first_pid_m = group_id * GROUP_SIZE_M
	group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
	pid_m = first_pid_m + (pid % group_size_m)
	pid_n = (pid % num_pid_in_group) // group_size_m
	offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
	offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	offs_k = tl.arange(0, BLOCK_SIZE_K)
	a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
	b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
	acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
	for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
		a_data = tl.load(a_ptrs, mask=offs_k[None, :] < (K - k * BLOCK_SIZE_K), other=0.0)
		b_data = tl.load(b_ptrs, mask=offs_k[:, None] < (K - k * BLOCK_SIZE_K), other=0.0)
		acc += tl.dot(a_data, b_data)
		a_ptrs += BLOCK_SIZE_K * stride_ak
		b_ptrs += BLOCK_SIZE_K * stride_bk
	acc = activation(acc.to(dtype=tl.float16))
	offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
	offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
	c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
	tl.store(c_ptrs, acc, mask=c_mask)


def _triton_call_gemm(A, B):
	M, K, N = A.shape[0], A.shape[1], B.shape[1]
	out_shape = jax.ShapeDtypeStruct(
		(A.shape[0], B.shape[1]),
		dtype=A.dtype,
	)

	# stride_am: tl.constexpr,
	# stride_ak: tl.constexpr,
	# stride_bk: tl.constexpr,
	# stride_bn: tl.constexpr,
	# stride_cm: tl.constexpr,
	# stride_cn: tl.constexpr,
	# M: tl.constexpr,
	# N: tl.constexpr,
	# K: tl.constexpr,
	# BLOCK_SIZE_M: tl.constexpr,
	# BLOCK_SIZE_N: tl.constexpr,
	# BLOCK_SIZE_K: tl.constexpr,
	# GROUP_SIZE_M: tl.constexpr,
	metaparams = dict(
		stride_am=K,
		stride_ak=1,
		stride_bk=N,
		stride_bn=1,
		stride_cm=N,
		stride_cn=1,
		M=M,
		N=N,
		K=K,
	)
	return jt.triton_call(
		A,
		B,
		kernel=_triton_gemm,
		grid=lambda META: (
			triton.cdiv(META["M"], META["BLOCK_SIZE_M"])
			* triton.cdiv(META["N"], META["BLOCK_SIZE_N"]),
		),
		out_shape=out_shape,
		**metaparams,
	)


op_prim = jcore.Primitive("op_prim")


def impt_prim(A, B):
	# operate in float16 or bfloat16 since they are both accurate on GPUs
	if A.dtype == jnp.bfloat16 or B.dtype == jnp.bfloat16:
		compute_dtype = jnp.bfloat16
	elif A.dtype == jnp.float16 or B.dtype == jnp.float16:
		compute_dtype = jnp.float16
	else:
		compute_dtype = jnp.float32
	return _triton_call_gemm(
		A.astype(compute_dtype),
		B.astype(compute_dtype),
	).astype(A.dtype)


op_prim.def_impl(partial(xla.apply_primitive, op_prim))

mlir.register_lowering(
	op_prim,
	mlir.lower_fun(fun=impt_prim, multiple_results=False),
	platform="gpu",
)


@op_prim.def_abstract_eval
def triton_gemm_abstract_eval(A, B):
	if A.ndim != 2 or B.ndim != 2:
		raise ValueError("Both inputs must be 2-dimensional")
	if A.shape[1] != B.shape[0]:
		raise ValueError(
			f"Incompatible shapes for matrix multiplication: {A.shape} and {B.shape}"
		)
	return jax.core.ShapedArray((A.shape[0], B.shape[1]), A.dtype)


def triton_gemm_vjp(A, B):
	def vjp(y_bar):
		return (op_prim.bind(y_bar, B.T), op_prim.bind(A.T, y_bar))

	return op_prim.bind(A, B), vjp


def triton_gemm_batch(args, batch_axes):
	A, B = args
	A_axis, B_axis = batch_axes

	if A_axis is None and B_axis is None:
		return op_prim.bind(A, B), None

	# Helper function to prepare inputs
	def prepare_input(X, axis):
		if axis is not None:
			return jnp.swapaxes(X, axis, 0) if axis != 0 else X
		return jnp.expand_dims(X, 0)

	A = prepare_input(A, A_axis)
	B = prepare_input(B, B_axis)

	# Ensure both A and B have the same batch dimension
	batch_size = max(A.shape[0], B.shape[0])
	A = jnp.broadcast_to(A, (batch_size,) + A.shape[1:])
	B = jnp.broadcast_to(B, (batch_size,) + B.shape[1:])

	# Perform batched matrix multiplication
	def batched_gemm(A, B):
		assert A.ndim == 3 and B.ndim == 3, "Inputs must be 3D after batching"
		return jax.lax.map(lambda x: op_prim.bind(x[0], x[1]), (A, B))

	result = batched_gemm(A, B)
	return result, 0  # The result is always batched along the first dimension


batching.primitive_batchers[op_prim] = triton_gemm_batch


@jax.custom_vjp
def gemm(A, B):
	return op_prim.bind(A, B)


def _fwd_gemm(A, B):
	return op_prim.bind(A, B), (A, B)


def _bwd_gemm(residual, gO):
	return (
		op_prim.bind(gO, residual[1].transpose(1, 0)),
		op_prim.bind(residual[0].transpose(1, 0), gO),
	)


gemm.defvjp(_fwd_gemm, _bwd_gemm)
gemm = jax.jit(gemm)


def test_run(argv):
	M, K, N = 1027 * 10, 4096, 1021 * 8
	dtype = jnp.float16
	A = jax.nn.initializers.normal(0.02, dtype=dtype)(jax.random.key(0), (M, K))
	B = jax.nn.initializers.normal(0.02, dtype=dtype)(jax.random.key(1), (K, N))
	res = gemm(A, B)
	g = jax.grad(lambda x, e: jnp.sum(x @ e))(A, B)
	g_ = jax.grad(lambda x, e: jnp.sum(gemm(x, e)))(A, B)
	print("Result Close : ", jnp.allclose(res, A @ B, atol=0.125, rtol=0))
	print("Grad Close   : ", jnp.allclose(g, g_, atol=0.125, rtol=0))


def test_vmap(argv):
	A = jnp.array(
		[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
	)  # Shape: (2, 2, 2)
	B = jnp.array([[5.0, 6.0], [7.0, 8.0]])  # Shape: (2, 2)

	result = jax.vmap(gemm, in_axes=(0, None))(A, B)
	print("Batched Result:", result)

	# Test gradients with batching
	grad_func = jax.grad(lambda A, B: jax.vmap(gemm, in_axes=(0, None))(A, B).sum())
	grad_A, grad_B = grad_func(A, B)
	print("Batched Gradient w.r.t. A:", grad_A)
	print("Batched Gradient w.r.t. B:", grad_B)


def bench(argv):
	import timeit

	M, K, N = 1027 * 10, 4096, 1021 * 8
	ntrials = 100
	dtype = jnp.float16
	A = jax.nn.initializers.normal(0.02, dtype=dtype)(jax.random.key(0), (M, K))
	B = jax.nn.initializers.normal(0.02, dtype=dtype)(jax.random.key(1), (K, N))
	_ = gemm(A, B)  # skip autotuning process

	def _benchmark(f, ntrials: int = 100):
		def run(*args, **kwargs):
			jax.block_until_ready(f(*args, **kwargs))
			result = timeit.timeit(
				lambda: jax.block_until_ready(f(*args, **kwargs)), number=ntrials
			)
			time = result / ntrials
			return time

		return run

	print("TRITON CALL : ", _benchmark(gemm, ntrials=ntrials)(A, B))
	print("JAX.LAX CALL : ", _benchmark(jax.lax.batch_matmul, ntrials=ntrials)(A, B))


__all__ = ["gemm"]
if __name__ == "__main__":
	from absl import app

	# app.run(test_run)
	# app.run(test_vmap)
	app.run(bench)
