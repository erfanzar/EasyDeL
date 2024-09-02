# Implementation by @erfanzar,
# with a few bug fixes and adjustments.
from functools import partial

import jax
import jax.random
from fjformer import GenerateRNG
from jax import numpy as jnp
from jax.experimental import pallas as pl

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
	assert A.ndim == 2 and B.ndim == 2
	assert A.shape[1] == B.shape[0]
	m, k, n = A.shape[0], A.shape[1], B.shape[1]
	blocksize_k = min(k, blocksize_k)
	grid = (pl.cdiv(m, blocksize_m), pl.cdiv(n, blocksize_n))

	in_specs = [
		pl.BlockSpec(lambda *_: (0,) * A.ndim, A.shape),
		pl.BlockSpec(lambda *_: (0,) * B.ndim, B.shape),
	]

	# interpret = list(A.devices())[0].platform == "cpu"
	interpret = True
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
	), (A, B)


def _call_matmul_kernel_bwd(
	blocksize_m: int,
	blocksize_k: int,
	blocksize_n: int,
	po_dtype: jnp.dtype,
	precision: jax.lax.PrecisionLike,
	res,
	gin,
):
	# A(mk)@B(kn)=C(mn)

	(A, B) = res

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
	intermediate_size = 2048
	hidden_size = 512
	dtype = jnp.float16
	A = jax.random.normal(rng.rng, (hidden_size, intermediate_size), dtype=dtype)
	B = jax.random.normal(rng.rng, (intermediate_size, hidden_size), dtype=dtype)
	y_ = matmul_kernel(A, B)
	y = A @ B
	print(jnp.allclose(y_, y, atol=0.125, rtol=0))
	print(y[0, :5])
	print(y_[0, :5])
	print(y[-1, :5])
	print(y_[-1, :5])


def matmul_grad_test():
	intermediate_size = 2048
	hidden_size = 512
	dtype = jnp.float16
	A = jax.random.normal(rng.rng, (hidden_size, intermediate_size), dtype=dtype)
	B = jax.random.normal(rng.rng, (intermediate_size, hidden_size), dtype=dtype)
	g = jax.grad(lambda x, e: jnp.sum(x @ e))(A, B)
	g_ = jax.grad(lambda x, e: jnp.sum(matmul_kernel(x, e)))(A, B)

	print(g_[0])
	print(g[0])


if __name__ == "__main__":
	matmul_grad_test()
