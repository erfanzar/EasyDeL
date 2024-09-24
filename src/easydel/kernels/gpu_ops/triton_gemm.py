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


import fjformer.jax_triton as jt
import jax
import triton
from jax import custom_vjp
from jax import numpy as jnp
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
	if _is_cuda():
		return _get_cuda_autotune_config()
	else:
		return _get_hip_autotune_config()


@triton.autotune(configs=_get_autotune_config(), key=["M", "N", "K"])
@triton.jit
def _triton_gemm_kernel(
	a_ptr,
	b_ptr,
	M,
	N,
	K,
	stride_am,
	stride_ak,
	stride_bk,
	stride_bn,
	stride_cm,
	stride_cn,
	c_ptr,
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


@triton.autotune(configs=_get_autotune_config(), key=["M", "N", "K"])
@triton.jit
def _gemm_activation_kernel(
	a_ptr,
	b_ptr,
	M,
	N,
	K,
	stride_am,
	stride_ak,
	stride_bk,
	stride_bn,
	stride_cm,
	stride_cn,
	c_ptr,
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


def _triton_call_gemm_kernel(A, B):
	M, K, N = A.shape[0], A.shape[1], B.shape[1]
	out_shape = jax.ShapeDtypeStruct(
		(A.shape[0], B.shape[1]),
		dtype=A.dtype,
	)
	return jt.triton_call(
		A,
		B,
		M,
		N,
		K,
		*jt.strides_from_shape(A.shape),
		*jt.strides_from_shape(B.shape),
		*jt.strides_from_shape(out_shape.shape),
		kernel=_triton_gemm_kernel,
		grid=lambda META: (
			triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
		),
		out_shape=out_shape,
	)


def _gemm_fwd(A, B):
	return _triton_call_gemm_kernel(A, B), (A, B)


def _gemm_bwd(res, gO):
	A, B = res
	return gemm(A=gO, B=B.transpose(1, 0)), gemm(A=A.transpose(1, 0), B=gO)


@custom_vjp
def gemm(A, B):
	return _triton_call_gemm_kernel(A, B)


gemm.defvjp(_gemm_fwd, _gemm_bwd)
gemm = jax.custom_batching.custom_vmap(gemm)


def _vmap_rule(axis_size, in_batched, a, b):
	print(axis_size, in_batched)
	if in_batched[0] and not in_batched[1] and a.ndim == 3 and b.ndim == 2:
		results = []
		for axis in range(axis_size):
			results.append(jnp.expand_dims(gemm(a[axis, :, :], b), 0))
		return jnp.concatenate(results, axis=0, dtype=results[0].dtype)
	raise NotImplementedError("`_vmap_rule` is not fully implemented yet!")


gemm.def_vmap(_vmap_rule)
gemm = jax.jit(gemm)


def test(argv):
	M, K, N = 1027 * 10, 4096, 1021 * 8
	dtype = jnp.float16
	A = jax.nn.initializers.normal(0.02, dtype=dtype)(jax.random.key(0), (M, K))
	B = jax.nn.initializers.normal(0.02, dtype=dtype)(jax.random.key(1), (K, N))
	res = gemm(A, B)
	g = jax.grad(lambda x, e: jnp.sum(x @ e))(A, B)
	g_ = jax.grad(lambda x, e: jnp.sum(gemm(x, e)))(A, B)
	print("Result Close : ", jnp.allclose(res, A @ B, atol=0.125, rtol=0))
	print("Grad Close   : ", jnp.allclose(g, g_, atol=0.125, rtol=0))


__all__ = ["gemm"]
if __name__ == "__main__":
	from absl import app

	app.run(test)
