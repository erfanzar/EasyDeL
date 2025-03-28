# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
# Copyright 2024 The Improved Version Contributors.
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

import enum
import typing as tp
import warnings  # Use dataclass for config
from dataclasses import dataclass

import jax.numpy as jnp
from flax import nnx as nn
from flax.nnx.nn.dtypes import promote_dtype
from jax import lax
from jax.sharding import Mesh

# Type Aliases
Array = jnp.ndarray
Dtype = jnp.dtype
Initializer = nn.initializers.Initializer
PrecisionLike = lax.PrecisionLike
Shape = tp.Sequence[int]
AxisNames = tp.Union[str, tp.Sequence[str], tp.Tuple[str, ...]]

# Default initializers
default_kernel_init = nn.initializers.lecun_normal()
default_bias_init = nn.initializers.zeros


@enum.unique
class TensorParallelType(enum.Enum):
	"""Type of tensor parallelism for layers."""

	ROW = enum.auto()
	# Shards input features, aggregates output (AllReduce)
	COLUMN = enum.auto()
	# Shards output features, gathers output (AllGather/ReduceScatter)


@dataclass
class TensorParallelConfig:
	"""Configuration for Tensor Parallelism.

	Attributes:
	    mesh: The JAX device mesh.
	    axis_name: The name of the mesh axis to use for tensor parallelism.
	    parallel_type: The type of parallelism (ROW or COLUMN).
	    reduce_scatter_output: If True and parallel_type is COLUMN,
	        use reduce-scatter instead of all-gather for the output.
	        This keeps the output sharded (useful for sequence parallelism
	        or subsequent RowParallel layers). Defaults to False.
	"""

	mesh: Mesh
	axis_name: str = "tp"
	parallel_type: tp.Optional[TensorParallelType] = None
	reduce_scatter_output: bool = False

	def __post_init__(self):
		if self.parallel_type is not None:
			if not isinstance(self.mesh, Mesh):
				msg = "`mesh` must be provided and be a `Mesh` when using tensor parallelism."
			if self.axis_name not in self.mesh.axis_names:
				msg = f"axis_name '{self.axis_name}' not found in mesh axis names: {self.mesh.axis_names}"
			if self.reduce_scatter_output and self.parallel_type != TensorParallelType.COLUMN:
				msg = "`reduce_scatter_output=True` is only valid for `COLUMN`."

			raise ValueError(msg)


class ParallelLinear(nn.Module):
	"""A Linear layer with optional parallelism.

	Behaves like `nnx.Linear` but can distribute computation and parameters
	across devices based on the `TensorParallelConfig`.

	Attributes:
	    in_features: Number of input features.
	    out_features: Number of output features.
	    use_bias: Whether to include a bias term. Default is True.
	    dtype: The dtype of the computation (defaults to inferred from input).
	    param_dtype: The dtype of the parameters. Default is float32.
	    precision: JAX precision for the dot product. Default is None.
	    kernel_init: Initializer for the kernel weights.
	    bias_init: Initializer for the bias.
	    parallel_config: Configuration for tensor parallelism. If None,
	      the layer behaves like a standard non-parallel Linear layer.
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
		parallel_config: tp.Optional[TensorParallelConfig] = None,
		rngs: nn.Rngs,
	):
		self.in_features = in_features
		self.out_features = out_features
		self.use_bias = use_bias
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.kernel_init = kernel_init
		self.bias_init = bias_init
		self.parallel_config = parallel_config

		if parallel_config:
			mesh = parallel_config.mesh
			axis_name = parallel_config.axis_name
			tp_size = mesh.shape[axis_name]

			if parallel_config.parallel_type == TensorParallelType.COLUMN:
				if out_features % tp_size != 0:
					raise ValueError(
						f"Output features ({out_features}) must be divisible by "
						f"tensor parallel size ({tp_size}) for COLUMN parallelism."
					)

			elif parallel_config.parallel_type == TensorParallelType.ROW:
				if in_features % tp_size != 0:
					raise ValueError(
						f"Input features ({in_features}) must be divisible by "
						f"tensor parallel size ({tp_size}) for ROW parallelism."
					)
			else:
				raise ValueError(f"Invalid parallel_type: {parallel_config.parallel_type}")
		self._num_merged = len(out_features) if isinstance(out_features, tp.Sequence) else 1
		out_features_sum = sum(out_features) if self._num_merged > 1 else out_features
		kernel_key = rngs.params()
		kernel_shape = (in_features, out_features_sum)
		self.kernel = nn.Param(kernel_init(kernel_key, kernel_shape, param_dtype))

		if use_bias:
			bias_key = rngs.params()
			bias_shape = (out_features,)
			self.bias = nn.Param(bias_init(bias_key, bias_shape, param_dtype))
		else:
			self.bias = None

	def __call__(self, inputs: Array) -> Array:
		"""Applies the linear transformation with optional tensor parallelism.

		Args:
		    inputs: The input array. Shape: (..., in_features).
		            For ROW parallelism, the input is expected to be sharded
		            along the feature dimension (`axis_name`).

		Returns:
		    The transformed output array.
		    Shape: (..., out_features).
		    Output is sharded for COLUMN parallelism if `reduce_scatter_output` is True.
		    Otherwise, output is fully replicated.
		"""
		kernel = self.kernel.value
		bias = self.bias.value if self.use_bias else None

		if bias is not None:
			inputs, kernel, bias = promote_dtype((inputs, kernel, bias), dtype=self.dtype)
		else:
			inputs, kernel = promote_dtype((inputs, kernel), dtype=self.dtype)

		if self.parallel_config is None:
			y = lax.dot_general(
				inputs,
				kernel,
				(((inputs.ndim - 1,), (0,)), ((), ())),
				precision=self.precision,
			)
		else:
			axis_name = self.parallel_config.axis_name
			parallel_type = self.parallel_config.parallel_type

			if parallel_type == TensorParallelType.COLUMN:
				y_local = lax.dot_general(
					inputs,
					kernel,
					(((inputs.ndim - 1,), (0,)), ((), ())),
					precision=self.precision,
				)

				if self.parallel_config.reduce_scatter_output:
					if self.parallel_config.reduce_scatter_output:
						y = lax.psum_scatter(
							y_local,
							axis_name=axis_name,
							scatter_dimension=-1,
							tiled=True,
						)
						warnings.warn(
							"reduce_scatter_output=True in ColumnParallel forward pass is unusual. "
							"Ensure this behavior (summing then scattering) is intended. "
							"Typically requires subsequent RowParallel or specific gradient handling.",
							stacklevel=1,
						)
					else:
						y = lax.all_gather(y_local, axis_name=axis_name, axis=-1)

				else:
					y = lax.all_gather(y_local, axis_name=axis_name, axis=-1)

			elif parallel_type == TensorParallelType.ROW:
				y_partial = lax.dot_general(
					inputs,
					kernel,
					(((inputs.ndim - 1,), (0,)), ((), ())),
					precision=self.precision,
				)

				y = lax.psum(y_partial, axis_name=axis_name)

			else:
				raise AssertionError("Should be unreachable due to config validation")

		if self.parallel_config is None:
			y = lax.dot_general(
				inputs, kernel, (((inputs.ndim - 1,), (0,)), ((), ())), precision=self.precision
			)
			if bias is not None:
				y = y + jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))

		else:
			axis_name = self.parallel_config.axis_name
			parallel_type = self.parallel_config.parallel_type

			if parallel_type == TensorParallelType.COLUMN:
				y_local = lax.dot_general(
					inputs,
					kernel,
					(((inputs.ndim - 1,), (0,)), ((), ())),
					precision=self.precision,
				)
				if bias is not None:
					y_local = y_local + jnp.reshape(bias, (1,) * (y_local.ndim - 1) + (-1,))

				if self.parallel_config.reduce_scatter_output:
					y = lax.psum_scatter(
						y_local,
						axis_name=axis_name,
						scatter_dimension=-1,
						tiled=True,
					)
				else:
					y = lax.all_gather(y_local, axis_name=axis_name, axis=-1)

			elif parallel_type == TensorParallelType.ROW:
				y_partial = lax.dot_general(
					inputs,
					kernel,
					(((inputs.ndim - 1,), (0,)), ((), ())),
					precision=self.precision,
				)
				y = lax.psum(y_partial, axis_name=axis_name)
				if bias is not None:
					y = y + jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))

			else:
				raise AssertionError("Should be unreachable")

		return y
