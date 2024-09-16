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

import itertools
from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar, Union

import chex
import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
from chex import Array, PRNGKey, Shape
from einops import einsum
from flax import linen as nn
from flax.linen.dtypes import promote_dtype
from flax.linen.linear import (
	ConvGeneralDilatedT,
	Dtype,
	PaddingLike,
	PrecisionLike,
	_conv_dimension_numbers,
	canonicalize_padding,
	default_kernel_init,
)
from jax import eval_shape, lax
from jax.core import ShapedArray

from easydel.modules.flax_modeling_utils import (
	ACT2FN,
)
from easydel.modules.mamba2.mamba2_configuration import Mamba2Config as Mamba2Config
from easydel.modules.modeling_flax_outputs import FlaxBaseModelOutput


def init_to_value(x, dtype):
	return lambda _: x.astype(dtype)


@flax.struct.dataclass
class Mamba2Output(FlaxBaseModelOutput):
	last_hidden_state: chex.Array = None
	cache_params: Optional[List[chex.Array]] = None
	hidden_states: Optional[Tuple[chex.Array]] = None


@flax.struct.dataclass
class Mamba2CausalLMOutput(FlaxBaseModelOutput):
	logits: chex.Array = None
	cache_params: Optional[List[chex.Array]] = None
	hidden_states: Optional[Tuple[chex.Array]] = None


def pad_tensor_by_size(input_tensor: jnp.ndarray, pad_size: int):
	"""
	Padding x tensor with `pad_size` on the seq_len dim (dim=1)
	"""
	if input_tensor.ndim == 4:
		pad_width = [(0, 0), (0, pad_size), (0, 0), (0, 0)]
	else:
		pad_width = [(0, 0), (0, pad_size), (0, 0)]

	return jnp.pad(input_tensor, pad_width, mode="constant", constant_values=0)


def reshape_into_chunks(input_tensor, pad_size, chunk_size):
	"""
	Padding input_tensor with `pad_size` on the seq_len dim (dim=1) and
	simultaneously splitting it into chunk sequences.
	"""
	input_tensor = pad_tensor_by_size(input_tensor, pad_size)

	if input_tensor.ndim == 3:
		return input_tensor.reshape(
			input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2]
		)
	else:
		return input_tensor.reshape(
			input_tensor.shape[0],
			-1,
			chunk_size,
			input_tensor.shape[2],
			input_tensor.shape[3],
		)


def segment_sum(input_tensor):
	"""
	More stable segment sum calculation. Uses cumulative sums and masking instead of direct subtractions.
	"""
	chunk_size = input_tensor.shape[-1]
	input_tensor = jnp.expand_dims(input_tensor, axis=-1)
	input_tensor = jnp.tile(input_tensor, (1,) * (input_tensor.ndim - 1) + (chunk_size,))

	mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool), k=-1)
	input_tensor = jnp.where(mask, input_tensor, 0)

	tensor_segsum = jnp.cumsum(input_tensor, axis=-2)

	mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool), k=0)
	tensor_segsum = jnp.where(mask, tensor_segsum, -jnp.inf)

	return tensor_segsum


_T = TypeVar("_T")


class FlaxMamba2Cache:
	def __init__(
		self,
		config: Mamba2Config,
		batch_size: int,
		dtype=jnp.float16,
	):
		self.seqlen_offset = 0
		self.dtype = dtype
		intermediate_size = config.intermediate_size
		ssm_state_size = config.state_size
		conv_kernel_size = config.conv_kernel

		self.conv_states = {
			i: jnp.zeros((batch_size, intermediate_size, conv_kernel_size), dtype=dtype)
			for i in range(config.num_hidden_layers)
		}
		self.ssm_states = {
			i: jnp.zeros((batch_size, intermediate_size, ssm_state_size), dtype=dtype)
			for i in range(config.num_hidden_layers)
		}


def create_tuple_parser(n: int) -> Callable[[Union[_T, Sequence[_T]]], tuple[_T, ...]]:
	def parse(x: Union[_T, Sequence[_T]]) -> tuple[_T, ...]:
		if isinstance(x, Sequence):
			if len(x) == n:
				return tuple(x)
			else:
				raise ValueError(f"x!=n ({x}!=({n}))")
		else:
			return tuple(itertools.repeat(x, n))

	return parse


class Conv1D(nn.Module):
	features: int
	kernel_size: int = 1
	stride: int = 1
	padding: int = 0
	dilation: int = 1
	groups: int = 1
	use_bias: bool = True
	num_spatial_dims: int = 1
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[str, lax.Precision]] = None

	@nn.compact
	def __call__(self, x):
		kernel = self.param(
			"kernel",
			nn.initializers.lecun_normal(dtype=self.param_dtype),
			(self.features, 1, self.kernel_size),
			self.param_dtype,
		)
		unbatched_rank = self.num_spatial_dims + 2
		if x.ndim != unbatched_rank:
			raise ValueError(
				f"Input to `Conv` needs to have rank {unbatched_rank},"
				f" but input has shape {x.shape}.",
			)

		x = lax.conv_general_dilated(
			lhs=x,
			rhs=jnp.asarray(kernel, dtype=self.dtype),
			window_strides=(self.stride,),
			padding=((self.padding, self.padding),),
			rhs_dilation=(self.dilation,),
			feature_group_count=self.groups,
		)
		if self.use_bias:
			bias = self.param(
				"bias", nn.initializers.zeros_init(), (self.features,), self.param_dtype
			)
			x = x + jnp.asarray(bias.reshape(1, -1, 1), dtype=self.dtype)
		return x


def mamba_ssm(
	u: jax.Array,
	delta: jax.Array,
	A: jax.Array,
	B: jax.Array,
	C: jax.Array,
	D: Optional[jax.Array] = None,
	delta_bias: Optional[jax.Array] = None,
	delta_softplus: bool = False,
	associative_scan: bool = True,
) -> jax.Array:
	if delta_bias is not None:
		raise NotImplementedError("delta_bias not implemented yet.")

	_, d_in = u.shape
	n = A.shape[1]

	delta = jnp.asarray(delta, dtype=jnp.float32)

	if delta_softplus:
		delta = jax.nn.softplus(delta)

	delta_A = jnp.exp(einsum(delta, A, "l d_in, d_in n -> l d_in n"))
	delta_B_u = einsum(delta, B, u, "l d_in, l n, l d_in -> l d_in n")

	x = jnp.zeros((d_in, n))

	def _scan_fn(x, params):
		d_A, d_Bu, C = params

		x = d_A * x + d_Bu
		return x, einsum(x, C, "d_in n, n -> d_in")

	def _associative_scan_fn(s, c):
		return tuple((c[0] * s[0], c[0] * s[1] + c[1]))

	if associative_scan:
		_, y = jax.lax.associative_scan(_associative_scan_fn, (delta_A, delta_B_u))
		y = einsum(y, C, "L d_in n, L n -> L d_in")
	else:
		_, y = jax.lax.scan(_scan_fn, init=x, xs=[delta_A, delta_B_u, C])

	y = y + u * D
	return y


class Conv(nn.Module):
	features: int
	kernel_size: Sequence[int]
	strides: Union[None, int, Sequence[int]] = 1
	padding: PaddingLike = "SAME"
	input_dilation: Union[None, int, Sequence[int]] = 1
	kernel_dilation: Union[None, int, Sequence[int]] = 1
	feature_group_count: int = 1
	use_bias: bool = True
	mask: Optional[Array] = None
	dtype: Optional[Dtype] = None
	param_dtype: Dtype = jnp.float32
	precision: PrecisionLike = None
	kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
	bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros_init()
	conv_general_dilated: Optional[ConvGeneralDilatedT] = None
	conv_general_dilated_cls: Any = None

	@property
	def shared_weights(self) -> bool:
		return True

	@nn.compact
	def __call__(self, inputs: Array) -> Array:
		if isinstance(self.kernel_size, int):
			raise TypeError(
				"Expected Conv kernel_size to be a"
				" tuple/list of integers (eg.: [3, 3]) but got"
				f" {self.kernel_size}."
			)
		else:
			kernel_size = tuple(self.kernel_size)

		def maybe_broadcast(x: Optional[Union[int, Sequence[int]]]) -> Tuple[int, ...]:
			if x is None:
				# backward compatibility with using None as sentinel for
				# broadcast 1
				x = 1
			if isinstance(x, int):
				return (x,) * len(kernel_size)
			return tuple(x)

		# Combine all input batch dimensions into a single leading batch axis.
		num_batch_dimensions = inputs.ndim - (len(kernel_size) + 1)
		if num_batch_dimensions != 1:
			input_batch_shape = inputs.shape[:num_batch_dimensions]
			total_batch_size = int(np.prod(input_batch_shape))
			flat_input_shape = (total_batch_size,) + inputs.shape[num_batch_dimensions:]
			inputs = jnp.reshape(inputs, flat_input_shape)

		# self.strides or (1,) * (inputs.ndim - 2)
		strides = maybe_broadcast(self.strides)
		input_dilation = maybe_broadcast(self.input_dilation)
		kernel_dilation = maybe_broadcast(self.kernel_dilation)

		padding_lax = canonicalize_padding(self.padding, len(kernel_size))
		if padding_lax == "CIRCULAR":
			kernel_size_dilated = [
				(k - 1) * d + 1 for k, d in zip(kernel_size, kernel_dilation)
			]
			zero_pad: List[Tuple[int, int]] = [(0, 0)]
			pads = zero_pad + [((k - 1) // 2, k // 2) for k in kernel_size_dilated] + [(0, 0)]
			inputs = jnp.pad(inputs, pads, mode="wrap")
			padding_lax = "VALID"
		elif padding_lax == "CAUSAL":
			if len(kernel_size) != 1:
				raise ValueError("Causal padding is only implemented for 1D convolutions.")
			left_pad = kernel_dilation[0] * (kernel_size[0] - 1)
			pads = [(0, 0), (left_pad, 0), (0, 0)]
			inputs = jnp.pad(inputs, pads)
			padding_lax = "VALID"

		dimension_numbers = _conv_dimension_numbers(inputs.shape)
		in_features = jnp.shape(inputs)[-1]

		if self.shared_weights:
			# One shared convolutional kernel for all pixels in the output.

			inf_f = in_features // self.feature_group_count
			# inf_f = 1
			kernel_shape = (
				self.features,
				inf_f,
			) + kernel_size

		else:
			if self.feature_group_count != 1:
				raise NotImplementedError(
					"`lax.conv_general_dilated_local` does not support "
					f"`feature_group_count != 1`, got `{self.feature_group_count}`."
				)

			# Need to know the spatial output shape of a standard convolution to
			# create the unshared convolution kernel.
			if self.conv_general_dilated_cls is not None:
				conv_general_dilated = self.conv_general_dilated_cls()
			elif self.conv_general_dilated is not None:
				conv_general_dilated = self.conv_general_dilated
			else:
				conv_general_dilated = lax.conv_general_dilated
			conv_output_shape = eval_shape(
				lambda lhs, rhs: conv_general_dilated(  # pylint: disable=g-long-lambda
					lhs=lhs,
					rhs=rhs,
					window_strides=strides,
					padding=padding_lax,
					dimension_numbers=dimension_numbers,
					lhs_dilation=input_dilation,
					rhs_dilation=kernel_dilation,
				),
				inputs,
				ShapedArray(kernel_size + (in_features, self.features), inputs.dtype),
			).shape

			# One (unshared) convolutional kernel per each pixel in the output.
			kernel_shape = conv_output_shape[1:-1] + (
				np.prod(kernel_size) * in_features,
				self.features,
			)

		if self.mask is not None and self.mask.shape != kernel_shape:
			raise ValueError(
				"Mask needs to have the same shape as weights. "
				f"Shapes are: {self.mask.shape}, {kernel_shape}"
			)

		kernel = self.param("kernel", self.kernel_init, kernel_shape, self.param_dtype)
		kernel = jnp.asarray(kernel.transpose(2, 1, 0), self.dtype)
		if self.mask is not None:
			kernel *= self.mask

		if self.use_bias:
			if self.shared_weights:
				bias_shape = (self.features,)
			else:
				bias_shape = conv_output_shape[1:]

			bias = self.param("bias", self.bias_init, bias_shape, self.param_dtype)
		else:
			bias = None

		inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)
		if self.shared_weights:
			if self.conv_general_dilated_cls is not None:
				conv_general_dilated = self.conv_general_dilated_cls()
			elif self.conv_general_dilated is not None:
				conv_general_dilated = self.conv_general_dilated
			else:
				conv_general_dilated = lax.conv_general_dilated
			y = conv_general_dilated(
				lhs=inputs,
				rhs=kernel,
				window_strides=strides,
				padding=padding_lax,
				lhs_dilation=input_dilation,
				rhs_dilation=kernel_dilation,
				dimension_numbers=dimension_numbers,
				feature_group_count=self.feature_group_count,
				precision=self.precision,
			)
		else:
			y = lax.conv_general_dilated_local(
				lhs=inputs,
				rhs=kernel,
				window_strides=strides,
				padding=padding_lax,
				filter_shape=kernel_size,
				lhs_dilation=input_dilation,
				rhs_dilation=kernel_dilation,
				dimension_numbers=dimension_numbers,
				precision=self.precision,
			)

		if self.use_bias:
			bias = bias.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)
			y += bias

		if num_batch_dimensions != 1:
			output_shape = input_batch_shape + y.shape[1:]
			y = jnp.reshape(y, output_shape)
		return y


class FlaxMambaRMSNormGated(nn.Module):
	hidden_size: int
	eps: float = 1e-6
	dtype: jnp.dtype = jnp.float32

	def setup(self):
		self.weight = self.param(
			"kernel",
			nn.initializers.ones,
			(self.hidden_size,),
			dtype=self.dtype,
		)

	def __call__(self, hidden_states, gate=None):
		input_dtype = hidden_states.dtype
		hidden_states = hidden_states.astype(jnp.float32)

		if gate is not None:
			gate = gate.astype(jnp.float32)
			hidden_states = hidden_states * jax.nn.silu(gate)

		variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
		hidden_states = hidden_states * jax.lax.rsqrt(variance + self.eps)

		return (self.weight * hidden_states).astype(input_dtype)


class FlaxMamba2Mixer(nn.Module):
	config: Mamba2Config
	layer_idx: int
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[str, lax.Precision]] = None

	def setup(self):
		self.num_heads = self.config.num_heads
		self.hidden_size = self.config.hidden_size
		self.ssm_state_size = self.config.state_size
		self.conv_kernel_size = self.config.conv_kernel
		self.intermediate_size = int(self.config.expand * self.hidden_size)
		self.time_step_rank = int(self.config.time_step_rank)
		self.use_conv_bias = self.config.use_conv_bias
		self.activation = self.config.hidden_act
		self.act = ACT2FN[self.config.hidden_act]

		self.norm_before_gate = self.config.norm_before_gate
		self.layer_norm_epsilon = self.config.layer_norm_epsilon
		self.rms_norm = self.config.rms_norm

		self.n_groups = self.config.n_groups
		self.head_dim = self.config.head_dim
		self.chunk_size = self.config.chunk_size

		self.time_step_limit = self.config.time_step_limit
		self.time_step_min = self.config.time_step_min
		self.time_step_max = self.config.time_step_max

		self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
		self.conv1d = nn.Conv(
			features=self.conv_dim,
			kernel_size=(self.config.conv_kernel,),
			feature_group_count=self.conv_dim,
			padding="SAME",
			use_bias=self.config.use_conv_bias,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)

		projection_size = self.intermediate_size + self.conv_dim + self.num_heads
		self.in_proj = nn.Dense(
			features=projection_size,
			use_bias=self.config.use_bias,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)

		self.dt_bias = self.param(
			"dt_bias",
			nn.initializers.ones,
			(self.num_heads,),
			dtype=self.dtype,
		)

		A = jnp.arange(1, self.num_heads + 1)
		self.A_log = self.param(
			"A_log", lambda _: jnp.log(A).astype(self.dtype), (self.num_heads,)
		)
		self.D = self.param(
			"D",
			nn.initializers.ones,
			(self.num_heads,),
			dtype=self.dtype,
		)

		self.norm = FlaxMambaRMSNormGated(
			self.intermediate_size,
			eps=self.layer_norm_epsilon,
			dtype=self.dtype,
		)
		self.out_proj = nn.Dense(
			features=self.hidden_size,
			use_bias=self.config.use_bias,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)

	def __call__(
		self,
		input_states,
		cache_params: FlaxMamba2Cache = None,
		cache_position=None,
		attention_mask=None,
	):
		batch_size, seq_len, _ = input_states.shape
		dtype = input_states.dtype

		# Gated MLP's linear projection
		projected_states = self.in_proj(jnp.squeeze(input_states, axis=1))
		d_mlp = (
			projected_states.shape[-1]
			- 2 * self.intermediate_size
			- 2 * self.n_groups * self.ssm_state_size
			- self.num_heads
		) // 2
		_, _, gate, hidden_states, dt = jnp.split(
			projected_states,
			[
				d_mlp,
				d_mlp * 2,
				d_mlp * 2 + self.intermediate_size,
				d_mlp * 2 + self.intermediate_size + self.conv_dim,
			],
			axis=-1,
		)
		if cache_params is not None:
			ssm_state = cache_params.ssm_states[self.layer_idx].copy()
			if cache_params.seqlen_offset > 0:
				conv_state = cache_params.conv_states[
					self.layer_idx
				]  # [batch, intermediate_size, conv_kernel_size]
				conv_state = jnp.roll(conv_state, shifts=-1, axis=-1)
				# handle batched generation - states are copied through
				conv_state[:, :, -1] = (
					hidden_states[:, 0, :] if hidden_states.ndim == 3 else hidden_states
				)
				cache_params.conv_states[self.layer_idx] = jax.lax.dynamic_update_slice(
					cache_params.conv_states[self.layer_idx],
					conv_state,
					(0, 0, 0, 0),
				)
				hidden_states = jnp.sum(
					conv_state * self.conv1d.variables["params"]["kernel"][:, 0, :],
					dim=-1,
				)
				if self.use_conv_bias:
					hidden_states += self.conv1d.variables["params"]["bias"]
				hidden_states = self.act(hidden_states).astype(dtype)[
					:, None, ...
				]  # [batch, 1, intermediate_size] : decoding
			else:
				hidden_states = jnp.transpose(hidden_states, (0, 2, 1))

				pad_width = [
					(0, 0),
					(0, 0),
					(self.conv_kernel_size - hidden_states.shape[-1], 0),
				]
				conv_state = jnp.pad(hidden_states, pad_width)

				cache_params.conv_states[self.layer_idx] = jax.lax.dynamic_update_slice(
					cache_params.conv_states[self.layer_idx],
					conv_state,
					(0, 0, 0, 0),
				)

				# Apply convolution and activation
				hidden_states = self.conv1d(hidden_states)
				hidden_states = jnp.transpose(hidden_states, (0, 2, 1))
				hidden_states = self.act(hidden_states)
				hidden_states = hidden_states[:, :seq_len, :]

				# Apply attention mask if necessary
				def apply_mask(hidden_states, attention_mask):
					return hidden_states * attention_mask[:, :, None]

				def identity(hidden_states):
					return hidden_states

				mask_condition = (
					attention_mask is not None
					and attention_mask.shape[1] > 1
					and attention_mask.shape[0] > 1
				)

				hidden_states = jax.lax.cond(
					mask_condition,
					lambda: apply_mask(hidden_states, attention_mask),
					lambda: identity(hidden_states),
				)
		else:
			ssm_state = jnp.zeros(
				(batch_size, self.num_heads, self.head_dim, self.ssm_state_size),
				dtype=dtype,
			)
			hidden_states = self.act(
				self.conv1d(hidden_states.transpose(1, 2))[..., :seq_len].transpose(1, 2)
			)
			hidden_states, B, C = jnp.split(
				hidden_states,
				[
					self.intermediate_size,
					self.n_groups * self.ssm_state_size,
					self.n_groups * self.ssm_state_size,
				],
				axis=-1,
			)
			A = -jnp.exp(self.A_log.float())  # [num_heads]
			if cache_params is not None and cache_params.seqlen_offset > 0:
				# Note: there is no need to pad parameter matrices here, as there is just one new token
				# for batched generation
				dt = dt[:, None, ...] if dt.ndim == 2 else dt[:, 0, :][:, None, ...]
				dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], self.head_dim)
				# [num_heads] -> [num_heads, head_dim]
				dt_bias = self.dt_bias[..., None].expand(self.dt_bias.shape[0], self.head_dim)

				dt = jax.nn.softplus(dt + dt_bias.astype(dt.dtype))
				dt = jnp.clip(dt, min=self.time_step_min)  # , self.time_step_max)
				A = (
					A[..., None, None]
					.expand(self.num_heads, self.head_dim, self.ssm_state_size)
					.astype(dtype=jnp.float32)
				)
				# [bsz, num_heads, head_dim, state_size]
				dA = jnp.exp(dt[..., None] * A)

				# Discretize B
				# [bsz, n_groups * state_size] -> [bsz, n_groups, 1, state_size] ->
				# -> [bsz, n_groups, group to head repetition factor, state_size] -> [bsz, num_heads, state_size]
				batch_size = B.shape[0]

				# Process B
				B = B.reshape(batch_size, self.n_groups, -1, 1)
				B = jnp.tile(B, (1, 1, self.num_heads // self.n_groups, 1))
				B = B.reshape(batch_size, -1, B.shape[-1])
				dB = dt[..., None] * B[..., None, :]

				# Process hidden_states
				hidden_states = hidden_states.reshape(batch_size, -1, self.head_dim)
				dBx = dB * hidden_states[..., None]

				# State calculation
				dA = cache_params.dA  # Assuming dA is pre-computed and stored in cache_params
				new_ssm_states = cache_params.ssm_states[self.layer_idx] * dA + dBx
				cache_params = cache_params.ssm_states[self.layer_idx] = new_ssm_states

				# Process C
				C = C.reshape(batch_size, self.n_groups, -1, 1)
				C = jnp.tile(C, (1, 1, self.num_heads // self.n_groups, 1))
				C = C.reshape(batch_size, -1, C.shape[-1])

				# Compute y
				ssm_states = cache_params.ssm_states[self.layer_idx]
				ssm_states_reshaped = ssm_states.reshape(
					batch_size * self.num_heads, self.head_dim, self.ssm_state_size
				)
				C_reshaped = C.reshape(batch_size * self.num_heads, self.ssm_state_size, 1)
				y = jnp.matmul(ssm_states_reshaped, C_reshaped)
				y = y.reshape(batch_size, self.num_heads, self.head_dim)

				# D skip connection
				D = jnp.tile(self.D[:, None], (1, self.head_dim))
				y = y + hidden_states * D

				# Reshape y
				y = y.reshape(batch_size, -1)[:, None, ...]
			else:
				# begin ssd naive implementation without einsums
				dt = jax.nn.softplus(dt + self.dt_bias)
				dt = jnp.clip(dt, min=self.time_step_min)
				hidden_states = hidden_states.reshape(
					batch_size, seq_len, -1, self.head_dim
				).float()
				B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
				C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
				B = B.repeat(1, 1, self.num_heads // self.n_groups, 1)
				C = C.repeat(1, 1, self.num_heads // self.n_groups, 1)
				pad_size = self.chunk_size - (seq_len % self.chunk_size)

				D_residual = self.D[..., None] * pad_tensor_by_size(hidden_states, pad_size)

				# Discretize x and A
				hidden_states = hidden_states * dt[..., None]
				A = A.astype(hidden_states.dtype) * dt

				# Rearrange into blocks/chunks
				hidden_states, A, B, C = [
					reshape_into_chunks(t, pad_size, self.chunk_size)
					for t in (hidden_states, A, B, C)
				]

				# [bsz, -1, chunk_size, num_heads] -> [bsz, num_heads, -1, chunk_size]
				A = jnp.permute_dims(A, axes=(0, 3, 1, 2))
				A_cumsum = jnp.cumsum(A, axis=-1)

				# 1. Compute the output for each intra-chunk (diagonal blocks)
				# This is the analog of a causal mask
				L = jnp.exp(segment_sum(A))

				# First, contraction of C and B to get G (attention-weights like)
				G_intermediate = (
					C[:, :, :, None, :, :] * B[:, :, None, :, :, :]
				)  # shape: (b, c, l, s, h, n)
				G = G_intermediate.sum(axis=-1)  # shape: (b, c, l, s, h)

				# Step 2: Compute M, equivalent to applying attention mask to weights
				M_intermediate = G[..., None] * jnp.permute_dims(L, (0, 2, 3, 4, 1))[..., None]
				M = M_intermediate.sum(axis=-1)

				# Step 3: Compute Y_diag (apply to values)
				Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(3)

				# (right term of low-rank factorization of off-diagonal blocks; B terms)

				decay_states = jnp.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
				B_decay_contraction = (
					B * jnp.permute_dims(decay_states, (0, 2, 3, 1))[..., None]
				)
				# permute back B * decay states
				states = jnp.permute_dims(
					(
						jnp.permute_dims(B_decay_contraction, axes=(0, 1, 3, 2, 4))[..., None]
						* jnp.permute_dims(hidden_states, axes=(0, 1, 3, 2, 4))[..., None, :]
					).sum(axis=3),
					axes=(0, 1, 2, 4, 3),
				)
				if cache_params is not None and cache_params.seqlen_offset > 0:
					previous_states = cache_params.ssm_states[self.layer_idx][:, None, ...]
				else:
					previous_states = jnp.zeros_like(states[:, :1])
				states = jnp.concatenate([previous_states, states], axis=1)
				decay_chunk = jnp.exp(
					segment_sum(jnp.pad(A_cumsum[:, :, :, -1], ((0, 0), (0, 0), (0, 0), (1, 0))))
				)

				states_permuted = jnp.permute_dims(states, axes=(0, 2, 1, 3, 4))
				result = (decay_chunk[..., None, None] * states_permuted[:, :, None, ...]).sum(
					dim=2
				)
				new_states = jnp.permute_dims(result, (0, 2, 1, 3, 4))
				states, ssm_state = new_states[:, :-1], new_states[:, -1]

				# Compute state -> output conversion per chunk
				# (left term of low-rank factorization of off-diagonal blocks; C terms)
				state_decay_out = jnp.exp(A_cumsum)
				# compute Yoff
				C_times_states = C[..., None, :] * states[:, :, None, ...]
				state_decay_out_permuted = jnp.permute_dims(state_decay_out, axes=(0, 2, 3, 1))
				Y_off = C_times_states.sum(-1) * state_decay_out_permuted[..., None]
				# Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)

				y = Y_diag + Y_off
				# [bsz, -1, self.chunk_size, num_heads, head_dim] -> [bsz, (padded) seq_len, num_heads, head_dim]
				y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)

				y = y + D_residual
				# Cutting off padded chunks
				if pad_size > 0:
					y = y[:, :seq_len, :, :]
				y = y.reshape(batch_size, seq_len, -1)
				if ssm_state is not None and cache_params is not None:
					cache_params.ssm_states[self.layer_idx] = ssm_state

				scan_output = self.norm(y, gate)
				contextualized_states = self.out_proj(
					scan_output.astype(dtype)
				)  # [batch, seq_len, hidden_size]
				return contextualized_states
