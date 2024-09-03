from jax import lax
from jax import numpy as jnp

from easydel.kernels.matmul import matmul_kernel


def phi3_mlp_pallas(
	x,
	proj_1,
	proj_2,
	*,
	act_fn,
	blocksize_m: int = 16,
	blocksize_k: int = 64,
	blocksize_n: int = 16,
	po_dtype: jnp.dtype = jnp.float32,
	precision: lax.PrecisionLike = None,
):
	args = dict(
		blocksize_k=blocksize_k,
		blocksize_m=blocksize_m,
		blocksize_n=blocksize_n,
		po_dtype=po_dtype,
		precision=precision,
	)
	y_12 = matmul_kernel(x, proj_1, **args)
	y_1, y_2 = jnp.split(y_12, 2, axis=-1)
	return matmul_kernel(act_fn(y_1) * y_2, proj_2, **args)