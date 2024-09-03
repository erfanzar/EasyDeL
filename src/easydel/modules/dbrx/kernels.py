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
	x1 = matmul_kernel(x, expert_w1.T, **args)
	x2 = matmul_kernel(x, expert_v1.T, **args)
	x1 = act_fn(x1)
	x1 = matmul_kernel(x1 * x2, expert_w2, **args)
	return x1
