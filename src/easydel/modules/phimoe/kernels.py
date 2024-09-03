from jax import lax
from jax import numpy as jnp

from easydel.kernels.matmul import matmul_kernel


def phimoe_mlp_pallas(
	x,
	w1,
	w2,
	w3,
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
	return matmul_kernel(
		(act_fn(matmul_kernel(x, w1, **args)) * matmul_kernel(x, w3, **args)), w2, **args
	)
