
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

from easydel.kernels.gemm import gemm_kernel


def phi3_mlp_pallas(
	x,
	proj_1,
	proj_2,
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
	y_12 = gemm_kernel(x, proj_1, **args)
	y_1, y_2 = jnp.split(y_12, 2, axis=-1)
	return gemm_kernel(act_fn(y_1) * y_2, proj_2, **args)