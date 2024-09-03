from functools import partial

import jax.experimental.pallas as pl
import jax.random
from flax import linen as nn
from jax import lax
from jax import numpy as jnp


class RMSNorm(nn.Module):
	dim: int
	eps: float = 1e-6
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32

	def setup(self) -> None:
		self.weight = self.param(
			"kernel",
			nn.initializers.ones,
			(self.dim,),
			self.param_dtype,
		)

	def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
		return x * lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)

	def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
		x = x.astype(jnp.promote_types(self.dtype, jnp.float32))
		output = self._norm(x).astype(self.dtype)

		weight = self.weight.astype(self.dtype)
		return weight * output


def _pl_fwd_rms_kernel(x_ref, w_ref, r_ref, o_ref, *, blocksize_x, eps, po_dtype):
	pid = pl.program_id(0)
	xslice = (pl.dslice(pid * blocksize_x, blocksize_x), pl.dslice(None))
	xi = (pid * blocksize_x + jnp.arange(blocksize_x) < x_ref.shape[0])[:, None]
	xj = (jnp.arange(x_ref.shape[1]) < x_ref.shape[1])[None, :]
	xmask = xi & xj
	x = pl.load(x_ref, xslice, mask=xmask).astype(po_dtype)
	scale = jax.lax.rsqrt(jnp.mean(jnp.square(x), keepdims=True, axis=-1) + eps)
	pl.store(
		r_ref,
		(pl.dslice(pid * blocksize_x, blocksize_x), pl.dslice(None)),
		scale,
		mask=xi & jnp.array([True])[None, :],
	)
	_normed = scale * x
	w = pl.load(w_ref, pl.dslice(0, x_ref.shape[1]))
	o = _normed.astype(o_ref.dtype) * w
	pl.store(o_ref, xslice, o, mask=xmask)


def _call_fwd_rms_kernel(x, w, blocksize_x, eps, po_dtype):
	assert x.ndim == 2 and w.ndim == 1
	in_specs = [
		pl.BlockSpec(x.shape, lambda *p: (0, 0)),
		pl.BlockSpec(w.shape, lambda *p: (0,)),
	]
	out_specs = [
		pl.BlockSpec((x.shape[0], 1), lambda *p: (0, 0)),
		pl.BlockSpec(x.shape, lambda *p: (0, 0)),
	]
	out_shape = [
		jax.ShapeDtypeStruct(shape=(x.shape[0], 1), dtype=po_dtype),
		jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype),
	]
	return pl.pallas_call(
		f=partial(
			_pl_fwd_rms_kernel,
			blocksize_x=blocksize_x,
			eps=eps,
			po_dtype=po_dtype,
		),
		in_specs=in_specs,
		out_specs=out_specs,
		out_shape=out_shape,
		interpret=True,
		debug=False,
		grid=(pl.cdiv(x.shape[0], blocksize_x)),
	)(x, w)


def _call_fwd_rms_kernel_g(x, w, blocksize_x, eps, po_dtype):
	scale, o = _call_fwd_rms_kernel(
		x=x,
		w=w,
		blocksize_x=blocksize_x,
		eps=eps,
		po_dtype=po_dtype,
	)

	return o, (
		x,
		w,
		scale,
		blocksize_x,
		eps,
	)


def _call_bwd_rms_kernel(po_dtype, res, gin):
	(
		x,
		w,
		scale,
		blocksize_x,
		eps,
	) = res


@partial(jax.custom_vjp, nondiff_argnums=(4,))
def _rms_call(
	x,
	w,
	blocksize_x,
	eps,
	po_dtype,
):
	return _call_fwd_rms_kernel_g(
		x=x,
		w=w,
		blocksize_x=blocksize_x,
		eps=eps,
		po_dtype=po_dtype,
	)[0]


_rms_call.defvjp(_call_fwd_rms_kernel_g, _call_bwd_rms_kernel)


def rms_norm(
	x: jax.Array,
	w: jax.Array,
	*,
	blocksize_x: int = 8,
	eps: float = 1e-5,
	po_dtype: jnp.dtype = jnp.float32,
):
	return _rms_call(
		x,
		w,
		blocksize_x,
		eps,
		po_dtype,
	)


def test_fwd_call():
	dim = 4
	seq = 4
	eps = 1e-5
	inputs = jax.random.normal(jax.random.key(564), (seq, dim), dtype=jnp.float16)

	norm = RMSNorm(dim, eps, jnp.float16, jnp.float16)
	params = norm.init(jax.random.PRNGKey(0), inputs)
	y = norm.apply(params, inputs)
	y_ = rms_norm(
		inputs,
		params["params"]["kernel"],
		blocksize_x=8,
		eps=eps,
		po_dtype=jnp.float32,
	)
	print(jnp.allclose(y_, y, atol=0.125, rtol=0))
	print(y_)
	# print(y)


def test_bwd_call():
	# FIX BWD
	dim = 4
	seq = 4
	eps = 1e-5
	inputs = jax.random.normal(jax.random.key(564), (seq, dim), dtype=jnp.float16)

	norm = RMSNorm(dim, eps, jnp.float16, jnp.float16)
	params = norm.init(jax.random.PRNGKey(0), inputs)
	g = jax.grad(lambda *x: jnp.sum(norm.apply(*x)))(params, inputs)["params"]["kernel"]
	g_ = jax.grad(
		lambda *x: jnp.sum(
			rms_norm(
				*x,
				blocksize_x=8,
				eps=eps,
				po_dtype=jnp.float32,
			)
		)
	)(inputs, params["params"]["kernel"])
	print(jnp.allclose(g, g_, atol=0.125, rtol=0))
	print(g_)
	print(g)


if __name__ == "__main__":
	# test_fwd_call()
	test_bwd_call()
