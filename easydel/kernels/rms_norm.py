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

import jax.experimental.pallas as pl
import jax.extend
import jax.random
from jax import numpy as jnp

PLATFORM = jax.extend.backend.get_backend().platform
INTERPRET = PLATFORM == "cpu"
# INTERPRET = True  # Debuging
# TODO :ISSUE IN JAX 0.4.33


def basic_layer_norm(
	x: jnp.ndarray,
	weight: jnp.ndarray,
	eps: float,
) -> jnp.ndarray:
	dtype = x.dtype
	x = x.astype(jnp.promote_types(dtype, jnp.float32))
	normed = x * jax.lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + eps)
	return (weight.astype(dtype) * normed).astype(dtype)


def _get_compiler_params():
	return None


def _pl_fwd_rms_kernel(x_ref, w_ref, o_ref, *, blocksize_x, eps, prod_dtype):
	pid = pl.program_id(0)
	xslice = (pl.dslice(pid * blocksize_x, blocksize_x), pl.dslice(None))
	xi = (pid * blocksize_x + jnp.arange(blocksize_x) < x_ref.shape[0])[:, None]
	xj = (jnp.arange(x_ref.shape[1]) < x_ref.shape[1])[None, :]
	xmask = xi & xj
	x = pl.load(x_ref, xslice, mask=xmask).astype(prod_dtype)
	scale = 1 / jnp.sqrt(jnp.mean(jnp.square(x), keepdims=True, axis=-1) + eps).astype(
		jnp.float32
	)
	_normed = scale.astype(x) * x
	w = pl.load(w_ref, pl.dslice(0, x_ref.shape[1]))
	o = _normed.astype(o_ref.dtype) * w
	pl.store(o_ref, xslice, o, mask=xmask)


def _pl_bwd_rms_kernel(
	x_ref,
	w_ref,
	gO_ref,
	dX_ref,
	*,
	blocksize_x,
	eps,
	n_cols,
	dtype,
):
	pid = pl.program_id(axis=0)
	xslice = (pl.dslice(pid * blocksize_x, blocksize_x), pl.dslice(None))
	gO_slice = (pl.dslice(pid * blocksize_x, blocksize_x), pl.dslice(None))
	x_mask_i = (pid * blocksize_x + jnp.arange(blocksize_x) < x_ref.shape[0])[:, None]
	x_mask_j = (jnp.arange(n_cols) < n_cols)[None, :]
	xmask = x_mask_i & x_mask_j
	gO_mask_i = (pid * blocksize_x + jnp.arange(blocksize_x) < x_ref.shape[0])[:, None]
	gO_mask_j = (jnp.arange(n_cols) < n_cols)[None, :]
	gO_mask = gO_mask_i & gO_mask_j
	x = pl.load(x_ref, xslice, mask=xmask, other=0.0)
	w = pl.load(w_ref, pl.dslice(None))
	gO = pl.load(gO_ref, gO_slice, mask=gO_mask, other=0.0)
	x_squared = x * x
	x_squared_sum = x_squared.sum(axis=-1, keepdims=True).astype(jnp.float32)
	x_norm = 1 / jax.lax.sqrt(x_squared_sum / n_cols + eps)
	x_norm = x_norm.astype(dtype)
	grad_x_norm = gO * w
	grad_x_part1 = grad_x_norm * x_norm

	grad_x_squared_sum = (-0.5 * (x_squared_sum / n_cols + eps) ** (-1.5)) * (
		2 * x.astype(jnp.float32) / n_cols
	).astype(dtype)
	grad_x_part2 = grad_x_squared_sum * (x * grad_x_norm).sum(axis=-1, keepdims=True)
	grad_x = grad_x_part1 + grad_x_part2
	dX_slice = (pl.dslice(pid * blocksize_x, blocksize_x), pl.dslice(None))
	dX_mask_i = (pid * blocksize_x + jnp.arange(blocksize_x) < x_ref.shape[0])[:, None]
	dX_mask_j = (jnp.arange(n_cols) < n_cols)[None, :]
	dX_mask = dX_mask_i & dX_mask_j
	pl.store(dX_ref, dX_slice, val=grad_x.astype(dX_ref), mask=dX_mask)


def _call_fwd_rms_kernel(x, w, blocksize_x, eps, prod_dtype):
	assert x.ndim == 2 and w.ndim == 1
	in_specs = [
		pl.BlockSpec(x.shape, lambda *p: (0, 0)),
		pl.BlockSpec(w.shape, lambda *p: (0,)),
	]
	out_specs = pl.BlockSpec(x.shape, lambda *p: (0, 0))
	out_shape = jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)

	method = pl.pallas_call(
		partial(
			_pl_fwd_rms_kernel,
			blocksize_x=blocksize_x,
			eps=eps,
			prod_dtype=prod_dtype,
		),
		in_specs=in_specs,
		out_specs=out_specs,
		interpret=INTERPRET,
		debug=False,
		grid=(pl.cdiv(x.shape[0], blocksize_x),),
		out_shape=out_shape,
		compiler_params=_get_compiler_params(),
	)
	out = method(x, w)
	return out


def _call_fwd_rms_kernel_with_residual(x, w, blocksize_x, eps, prod_dtype):
	o = _call_fwd_rms_kernel(
		x=x,
		w=w,
		blocksize_x=blocksize_x,
		eps=eps,
		prod_dtype=prod_dtype,
	)
	return o, (x, w, blocksize_x, eps)


def _call_bwd_rms_kernel(prod_dtype, res, gO):
	(
		x,
		w,
		blocksize_x,
		eps,
	) = res
	N = x.shape[-1]
	in_specs = [
		pl.BlockSpec(x.shape, lambda *p: (0, 0)),
		pl.BlockSpec(w.shape, lambda *p: (0,)),
		pl.BlockSpec(gO.shape, lambda *p: (0, 0)),
	]
	out_specs = pl.BlockSpec(x.shape, lambda *p: (0, 0))
	out_shape = jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
	method = pl.pallas_call(
		partial(
			_pl_bwd_rms_kernel,
			blocksize_x=blocksize_x,
			eps=eps,
			n_cols=N,
			dtype=prod_dtype,
		),
		in_specs=in_specs,
		out_specs=out_specs,
		out_shape=out_shape,
		grid=(pl.cdiv(x.shape[0], blocksize_x),),
		compiler_params=_get_compiler_params(),
		interpret=INTERPRET,
	)
	dX = method(x, w, gO)
	return dX, None, None, None


@partial(jax.custom_vjp, nondiff_argnums=(4,))
def _rms_call(x, w, blocksize_x, eps, prod_dtype):
	return _call_fwd_rms_kernel(
		x=x,
		w=w,
		blocksize_x=blocksize_x,
		eps=eps,
		prod_dtype=prod_dtype,
	)


_rms_call.defvjp(_call_fwd_rms_kernel_with_residual, _call_bwd_rms_kernel)


def rms_norm(
	x: jax.Array,
	w: jax.Array,
	blocksize_x: int = 8,
	eps: float = 1e-5,
	prod_dtype: jnp.dtype = jnp.float32,
):
	return _rms_call(x, w, blocksize_x, eps, prod_dtype)


def test_fwd_call():
	dim = 4
	seq = 4
	eps = 1e-5
	dtype = jnp.float16
	inputs = jax.random.normal(jax.random.key(564), (seq, dim), dtype=dtype)

	w = jnp.ones((dim,), dtype=dtype)

	print(jax.vjp(basic_layer_norm, inputs, w, eps)[1])
	# y = basic_layer_norm(inputs, w, eps)
	# y_ = rms_norm(
	# 	inputs,
	# 	w,
	# 	blocksize_x=64,
	# 	eps=eps,
	# 	prod_dtype=jnp.float32,
	# )
	# print("is Prediciton Ok?      = ", jnp.allclose(y_, y, atol=0.125, rtol=0))
	# print("Orginal Prediciton     = ", y[0, :4])
	# print("Kernel  Prediction     = ", y_[0, :4])


def test_bwd_call():
	# FIX BWD
	dim = 4
	seq = 4
	eps = 1e-5
	dtype = jnp.float16
	x = jax.random.normal(jax.random.key(564), (seq, dim), dtype=dtype)

	w = jnp.ones((dim,), dtype=dtype)

	g = jax.grad(lambda e: basic_layer_norm(e, w, eps).sum())(x)
	g_ = jax.grad(lambda e: rms_norm(e, w, 64, eps, dtype).sum())(x)
	print("is Prediciton G Ok?    = ", jnp.allclose(g, g_, 0.125, 0))
	print("Orginal w.r.t G        = ", g[0, :4])
	print("Calculated w.r.t G     = ", g_[0, :4])


if __name__ == "__main__":
	test_fwd_call()
	test_bwd_call()
