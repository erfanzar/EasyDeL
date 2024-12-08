from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
from jax import lax

from flax import nnx
from flax.nnx import rnglib
from flax.nnx.module import Module
from flax.nnx.nn import dtypes, initializers
from flax.typing import (
	Dtype,
	Initializer,
	PrecisionLike,
	DotGeneralT,
)

Array = jax.Array
Axis = int
Size = int


default_kernel_init = initializers.lecun_normal()
default_bias_init = initializers.zeros_init()


class Linear(Module):
	"""A linear transformation applied over the last dimension of the input.

	Example usage::

	  >>> from flax import nnx
	  >>> import jax, jax.numpy as jnp

	  >>> layer = nnx.Linear(in_features=3, out_features=4, rngs=nnx.Rngs(0))
	  >>> jax.tree.map(jnp.shape, nnx.state(layer))
	  State({
	    'bias': VariableState(
	      type=Param,
	      value=(4,)
	    ),
	    'kernel': VariableState(
	      type=Param,
	      value=(3, 4)
	    )
	  })

	Attributes:
	  in_features: the number of input features.
	  out_features: the number of output features.
	  use_bias: whether to add a bias to the output (default: True).
	  dtype: the dtype of the computation (default: infer from input and params).
	  param_dtype: the dtype passed to parameter initializers (default: float32).
	  precision: numerical precision of the computation see ``jax.lax.Precision``
	    for details.
	  kernel_init: initializer function for the weight matrix.
	  bias_init: initializer function for the bias.
	  dot_general: dot product function.
	  rngs: rng key.
	"""

	def __init__(
		self,
		in_features: int,
		out_features: int,
		*,
		use_bias: bool = True,
		dtype: tp.Optional[Dtype] = None,
		param_dtype: Dtype = jnp.float32,
		precision: PrecisionLike = None,
		kernel_init: Initializer = default_kernel_init,
		bias_init: Initializer = default_bias_init,
		dot_general: DotGeneralT = lax.dot_general,
		rngs: rnglib.Rngs,
	):
		kernel_key = rngs.params()
		self.kernel = nnx.Param(
			kernel_init(kernel_key, (in_features, out_features), param_dtype)
		)
		if use_bias:
			bias_key = rngs.params()
			self.bias = nnx.Param(bias_init(bias_key, (out_features,), param_dtype))
		else:
			self.bias = nnx.Param(None)

		self.in_features = in_features
		self.out_features = out_features
		self.use_bias = use_bias
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.kernel_init = kernel_init
		self.bias_init = bias_init
		self.dot_general = dot_general

	def __call__(self, inputs: Array) -> Array:
		"""Applies a linear transformation to the inputs along the last dimension.

		Args:
		  inputs: The nd-array to be transformed.

		Returns:
		  The transformed input.
		"""
		kernel = self.kernel.value
		bias = self.bias.value
		try:
			kernel = kernel.materialize()
			bias = bias.materialize()
		except Exception:
			...
		inputs, kernel, bias = dtypes.promote_dtype(
			(inputs, kernel, bias), dtype=self.dtype
		)
		y = self.dot_general(
			inputs,
			kernel,
			(((inputs.ndim - 1,), (0,)), ((), ())),
			precision=self.precision,
		)
		assert self.use_bias == (bias is not None)
		if bias is not None:
			y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
		return y
