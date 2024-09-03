from jax import lax
from jax import numpy as jnp
import jax

from easydel.kernels.matmul import matmul_kernel


def grok1_mlp_pallas(
	x,
	linear,
	linear_1,
	linear_v,
	*,
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
		(
			jax.nn.gelu(matmul_kernel(x, linear, **args)) * matmul_kernel(x, linear_v, **args)
		),
		linear_1,
		**args,
	)
