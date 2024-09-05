
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

from jax import lax
from jax import numpy as jnp

from easydel.kernels.matmul import matmul_kernel


def dbrx_mlp_pallas(
	x,
	expert_w1,
	expert_v1,
	expert_w2,
	*,
	act_fn,
	blocksize_m: int = 16,
	blocksize_k: int = 64,
	blocksize_n: int = 16,
	prod_dtype: jnp.dtype = jnp.float32,
	precision: lax.PrecisionLike = None,
):
	args = dict(
		blocksize_k=blocksize_k,
		blocksize_m=blocksize_m,
		blocksize_n=blocksize_n,
		prod_dtype=prod_dtype,
		precision=precision,
	)
	x1 = matmul_kernel(x, expert_w1.T, **args)
	x2 = matmul_kernel(x, expert_v1.T, **args)
	x1 = act_fn(x1)
	x1 = matmul_kernel(x1 * x2, expert_w2, **args)
	return x1
