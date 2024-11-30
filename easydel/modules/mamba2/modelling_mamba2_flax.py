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
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import chex
import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
from chex import Array, PRNGKey, Shape
from einops import einsum
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import partitioning as nn_partitioning
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
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import eval_shape, lax
from jax.core import ShapedArray

from easydel.etils.etils import EasyDeLGradientCheckPointers
from easydel.layers.common import Conv1D
from easydel.layers.norms import RMSNorm as FlaxMamba2RMSNorm
from easydel.modules.factory import register_module
from easydel.modules.flax_modeling_utils import (
	ACT2FN,
	get_gradient_checkpoint_policy,
)
from easydel.modules.mamba2.mamba2_configuration import Mamba2Config as Mamba2Config
from easydel.modules.modeling_flax_outputs import FlaxBaseModelOutput
from easydel.modules.modeling_utils import EasyDeLBaseModule


def init_to_value(x, dtype):
	return lambda *_: x.astype(dtype)


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


@jax.tree_util.register_pytree_node_class
@dataclass
class FlaxMamba2Cache:
	config: Mamba2Config
	batch_size: int
	dtype = jnp.float16
	conv_states: Optional[chex.Array] = None
	ssm_states: Optional[chex.Array] = None
	seqlen_offset: Optional[int] = None

	def __post_init__(self):
		self.seqlen_offset = 0 if self.seqlen_offset is None else self.seqlen_offset
		if self.conv_states is None:
			self.conv_states = {
				i: jnp.zeros(
					(
						self.batch_size,
						self.intermediate_size + 2 * self.config.n_groups * self.config.state_size,
						self.conv_kernel_size,
					),
					dtype=self.dtype,
				)
				for i in range(self.config.num_hidden_layers)
			}
		if self.ssm_states is None:
			self.ssm_states = {
				i: jnp.zeros(
					(
						self.batch_size,
						self.config.num_heads,
						self.config.head_dim,
						self.config.state_size,
					),
					dtype=self.dtype,
				)
				for i in range(self.config.num_hidden_layers)
			}

	def reset(self):
		self.conv_states = {
			i: jnp.zeros(
				(
					self.batch_size,
					self.intermediate_size + 2 * self.config.n_groups * self.config.state_size,
					self.conv_kernel_size,
				),
				dtype=self.dtype,
			)
			for i in range(self.config.num_hidden_layers)
		}
		self.ssm_states = {
			i: jnp.zeros(
				(
					self.batch_size,
					self.config.num_heads,
					self.config.head_dim,
					self.config.state_size,
				),
				dtype=self.dtype,
			)
			for i in range(self.config.num_hidden_layers)
		}

	def tree_flatten(self):
		return (
			self.config,
			self.batch_size,
			self.dtype,
			self.conv_states,
			self.ssm_states,
			self.seqlen_offset,
		), {}

	def tree_unflatten(cls, aux, children):
		return cls(*children)


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
			self.dtype,
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
		self.conv1d = Conv1D(
			features=self.conv_dim,
			kernel_size=self.config.conv_kernel,
			groups=self.conv_dim,
			stride=1,
			padding=self.config.conv_kernel - 1,
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

		dt = jax.lax.clamp(
			self.config.time_step_floor,
			jnp.exp(
				jax.random.normal(
					key=self.make_rng("params"),
					shape=(self.config.num_heads,),
					dtype=self.param_dtype,
				)
				* (jnp.log(self.config.time_step_max) - jnp.log(self.config.time_step_min))
				+ jnp.log(self.config.time_step_min)
			).astype(jnp.float32),
			1e9,
		)

		inv_dt = dt + jnp.log(-jnp.expm1(-dt))
		self.dt_bias = self.param(
			"dt_bias",
			init_to_value(inv_dt, self.param_dtype),
			(self.num_heads,),
		)

		self.A_log = self.param(
			"A_log",
			init_to_value(
				jnp.log(jnp.arange(1, self.num_heads + 1, dtype=jnp.float32)),
				self.param_dtype,
			),
		)
		self.D = self.param("D", init_to_value(jnp.ones(self.num_heads), self.param_dtype))

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
		input_states: chex.Array,
		cache_params: Optional[FlaxMamba2Cache] = None,
		attention_mask: Optional[chex.Array] = None,
	):
		dtype = input_states.dtype
		if (
			attention_mask is not None
			and attention_mask.shape[1] > 1
			and attention_mask.shape[0] > 1
		):
			input_states = (input_states * attention_mask[:, :, None]).to(dtype)
		batch_size, seq_len, _ = input_states.shape
		dtype = input_states.dtype

		# Gated MLP's linear projection
		projected_states = self.in_proj(input_states)
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
				self.conv1d(hidden_states.transpose(0, 2, 1))[..., :seq_len].transpose(0, 2, 1)
			)
			hidden_states, B, C = jnp.split(
				hidden_states,
				[
					self.intermediate_size,
					self.intermediate_size + self.n_groups * self.ssm_state_size,
				],
				axis=-1,
			)
			A = -jnp.exp(self.A_log.astype("float32"))  # [num_heads]
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
				dA = jnp.exp(dt[..., None] * A)
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
				).astype(jnp.float32)
				B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).astype(jnp.float32)
				C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).astype(jnp.float32)
				B = B.repeat(self.num_heads // self.n_groups, 2)
				C = C.repeat(self.num_heads // self.n_groups, 2)
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
				A = jnp.transpose(A, axes=(0, 3, 1, 2))
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
				M_intermediate = G[..., None] * jnp.transpose(L, (0, 2, 3, 4, 1))[..., None]
				M = M_intermediate.sum(axis=-1)

				# Step 3: Compute Y_diag (apply to values)
				Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(3)

				# (right term of low-rank factorization of off-diagonal blocks; B terms)

				decay_states = jnp.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
				B_decay_contraction = B * jnp.transpose(decay_states, (0, 2, 3, 1))[..., None]
				# permute back B * decay states
				states = jnp.transpose(
					(
						jnp.transpose(B_decay_contraction, axes=(0, 1, 3, 2, 4))[..., None]
						* jnp.transpose(hidden_states, axes=(0, 1, 3, 2, 4))[..., None, :]
					).sum(axis=3),
					axes=(0, 1, 2, 4, 3),
				)
				if cache_params is not None and cache_params.seqlen_offset > 0:
					previous_states = cache_params.ssm_states[self.layer_idx][:, None, ...]
				else:
					previous_states = jnp.zeros_like(states[:, :1])
				states = jnp.concatenate([previous_states, states], axis=1)
				decay_chunk = jnp.exp(
					segment_sum(jnp.pad(A_cumsum[:, :, :, -1], ((0, 0), (0, 0), (1, 0))))
				)

				states_permuted = jnp.transpose(states, axes=(0, 2, 1, 3, 4))
				result = (decay_chunk[..., None, None] * states_permuted[:, :, None, ...]).sum(
					axis=2
				)
				new_states = jnp.transpose(result, (0, 2, 1, 3, 4))
				states, ssm_state = new_states[:, :-1], new_states[:, -1]

				# Compute state -> output conversion per chunk
				# (left term of low-rank factorization of off-diagonal blocks; C terms)
				state_decay_out = jnp.exp(A_cumsum)
				# compute Yoff
				C_times_states = C[..., None, :] * states[:, :, None, ...]
				state_decay_out_permuted = jnp.transpose(state_decay_out, axes=(0, 2, 3, 1))
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


class FlaxMamba2Block(nn.Module):
	config: Mamba2Config
	layer_idx: int
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[str, lax.Precision]] = None

	def setup(self):
		config = self.config
		self.residual_in_fp32 = config.residual_in_fp32
		self.norm = FlaxMamba2RMSNorm(
			config.hidden_size,
			eps=config.layer_norm_epsilon,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
		)
		block = FlaxMamba2Mixer
		if self.config.gradient_checkpointing != EasyDeLGradientCheckPointers.NONE:
			block = nn_partitioning.remat(
				block,
				static_argnums=(1,),
				policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
			)
		self.mixer = block(
			config,
			self.layer_idx,
			self.dtype,
			self.param_dtype,
			self.precision,
		)

	def __call__(
		self,
		hidden_states: chex.Array,
		cache_params: Optional[FlaxMamba2Cache] = None,
		attention_mask: Optional[chex.Array] = None,
	) -> chex.Array:
		residual = hidden_states
		hidden_states = self.norm(hidden_states)
		if self.residual_in_fp32:
			residual = residual.astype(jnp.float32)
		hidden_states = self.mixer(
			hidden_states,
			cache_params,
			attention_mask,
		)
		hidden_states = residual + hidden_states
		return hidden_states


class FlaxMamba2LayerCollection(nn.Module):
	config: Mamba2Config
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[str, lax.Precision]] = None

	def setup(self) -> None:
		self.blocks = [
			FlaxMamba2Block(
				config=self.config,
				layer_idx=layer_idx,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
				name=str(layer_idx),
			)
			for layer_idx in range(self.config.num_hidden_layers)
		]

	def __call__(
		self,
		hidden_states: chex.Array,
		cache_params: Optional[FlaxMamba2Cache] = None,
		attention_mask: Optional[chex.Array] = None,
		output_hidden_states: bool = False,
	) -> Tuple[chex.Array, Union[None, Tuple[chex.Array, ...]]]:
		all_hidden_states = () if output_hidden_states else None
		for block in self.blocks:
			hidden_states = block(
				hidden_states=hidden_states,
				cache_params=cache_params,
				attention_mask=attention_mask,
			)

			if output_hidden_states:
				all_hidden_states = all_hidden_states + (hidden_states,)

		return hidden_states, all_hidden_states


class FlaxMamba2Module(nn.Module):
	config: Mamba2Config
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[str, lax.Precision]] = None

	def setup(self) -> None:
		config = self.config
		self.embeddings = nn.Embed(
			config.vocab_size,
			config.hidden_size,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
		)
		self.layers = FlaxMamba2LayerCollection(
			config=config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)

		self.norm_f = FlaxMamba2RMSNorm(
			config.hidden_size,
			eps=config.layer_norm_epsilon,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
		)

	def __call__(
		self,
		input_ids: Optional[chex.Array] = None,
		input_embeds: Optional[chex.Array] = None,
		attention_mask: Optional[chex.Array] = None,
		cache_params: Optional[chex.Array] = None,
		deterministic: bool = True,
		use_cache: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		**kwargs,
	) -> Union[Tuple, Mamba2Output]:
		output_hidden_states = (
			output_hidden_states
			if output_hidden_states is not None
			else self.config.output_hidden_states
		)
		use_cache = (
			use_cache
			if use_cache is not None
			else (self.config.use_cache if not deterministic else False)
		)
		return_dict = (
			return_dict if return_dict is not None else self.config.use_return_dict
		)

		if (input_ids is None) ^ (input_embeds is not None):
			raise ValueError(
				"You cannot specify both input_ids and input_embeds at the same time, and must specify either one"
			)

		if input_embeds is None:
			input_embeds = self.embeddings(input_ids)

		if deterministic and use_cache:
			use_cache = False

		if cache_params is None and use_cache:
			cache_params = FlaxMamba2Cache(
				self.config,
				input_embeds.shape[0],
				dtype=input_embeds.dtype,
			)

		hidden_states = input_embeds
		hidden_states, all_hidden_states = self.layers(
			hidden_states=hidden_states,
			cache_params=cache_params,
			attention_mask=attention_mask,
			output_hidden_states=output_hidden_states,
		)

		if use_cache:
			cache_params.seqlen_offset += input_embeds.shape[1]

		hidden_states = self.norm_f(hidden_states)

		if output_hidden_states:
			all_hidden_states = all_hidden_states + (hidden_states,)

		if not return_dict:
			return tuple(
				v for v in [hidden_states, cache_params, all_hidden_states] if v is not None
			)

		return Mamba2Output(
			last_hidden_state=hidden_states,
			cache_params=cache_params if use_cache else None,
			hidden_states=all_hidden_states,
		)


class FlaxMamba2ForCausalLMModule(nn.Module):
	config: Mamba2Config
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[str, lax.Precision]] = None

	def setup(self) -> None:
		self.backbone = FlaxMamba2Module(
			self.config, self.dtype, self.param_dtype, self.precision
		)
		self.lm_head = nn.Dense(
			self.config.vocab_size,
			use_bias=False,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)

	def __call__(
		self,
		input_ids: Optional[chex.Array] = None,
		input_embeds: Optional[chex.Array] = None,
		attention_mask: Optional[chex.Array] = None,
		cache_params: Optional[chex.Array] = None,
		deterministic: bool = True,
		use_cache: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		**kwargs,
	) -> Union[Tuple, Mamba2CausalLMOutput]:
		return_dict = (
			return_dict if return_dict is not None else self.config.use_return_dict
		)
		mamba_outputs = self.backbone(
			input_ids=input_ids,
			input_embeds=input_embeds,
			attention_mask=attention_mask,
			deterministic=deterministic,
			cache_params=cache_params,
			use_cache=use_cache,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)
		hidden_states = mamba_outputs[0]

		logits = self.lm_head(hidden_states).astype(jnp.float32)

		if not return_dict:
			return (logits,) + mamba_outputs[1:]

		return Mamba2CausalLMOutput(
			logits=logits,
			cache_params=mamba_outputs.cache_params,
			hidden_states=mamba_outputs.hidden_states,
		)


class FlaxMambaPretrainedModel(EasyDeLBaseModule):
	config_class = Mamba2Config
	base_model_prefix = "backbone"
	module_class: nn.Module = None

	def __init__(
		self,
		config: Mamba2Config,
		input_shape: Tuple = (1, 1),
		seed: int = 0,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: Optional[Union[str, lax.Precision]] = None,
		_do_init: bool = True,
		**kwargs,
	):
		"""The __init__ function is called when the class is instantiated.
		It sets up the instance of the class, and defines what happens when it's created.
		The __init__ function can take arguments, but self is always required (it refers to the instance of the object).

		Args:
		    self: Refer to the object itself
		    config: MambaConfig: Pass the configuration to the module
		    input_shape: Tuple: Specify the shape of the input to the
		        model
		    seed: int: Set the seed for random number generation
		    dtype: jnp.dtype: Specify the data type of the model ra
		    param_dtype: jnp.dtype: Specify the data type of the
		        param_dtype
		    precision: Optional[Union[str, lax.Precision]]: precision
		        for model operations
		    _do_init: bool: Control whether the module is initialized or
		        not
		    **kwargs: Pass in any additional parameters that the
		        module_class might need
		:param : Specify the number of layers in the network

		Returns:
		    The super() of the class
		"""

		super().__init__(
			config=config,
			param_dtype=param_dtype,
			precision=precision,
			input_shape=(input_shape[0], 1),
			seed=seed,
			dtype=dtype,
			_do_init=_do_init,
			**kwargs,
		)

	def init_weights(
		self,
		rng: jax.random.PRNGKey,
		input_shape: Tuple,
		params: FrozenDict = None,
	) -> FrozenDict:
		"""The init_weights function is used to initialize the weights of a model.

		Args:
		    self: Access variables that belong to the class
		    rng: jax.random.PRNGKey: Initialize the weights of the model
		    input_shape: Tuple: Specify the shape of the input tensor
		    params: FrozenDict: Pass in the parameters of a pre-trained
		        model

		Returns:
		    A frozendict of parameters
		"""
		input_ids = jnp.zeros(input_shape, dtype="i4")
		params_rng, dropout_rng = jax.random.split(rng)
		rngs = {"params": params_rng, "dropout": dropout_rng}

		module_init_outputs = self.module.init(rngs, input_ids, return_dict=False)

		random_params = module_init_outputs["params"]

		if params is not None:
			random_params = flatten_dict(unfreeze(random_params))
			params = flatten_dict(unfreeze(params))
			for missing_key in self._missing_keys:
				params[missing_key] = random_params[missing_key]
			self._missing_keys = set()
			return freeze(unflatten_dict(params))
		else:
			return random_params

	def init_cache(self, batch_size, max_length):
		return None

	def __call__(
		self,
		input_ids: Optional[chex.Array] = None,
		input_embeds: Optional[chex.Array] = None,
		attention_mask: Optional[chex.Array] = None,
		cache_params: dict = None,
		deterministic: bool = True,
		params: dict = None,
		dropout_rng: jax.random.PRNGKey = None,
		train: bool = False,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		extra_embedding: Optional[Union[jnp.ndarray, None]] = None,
		add_params_field: bool = False,
		use_cache: bool = False,
		**kwargs,
	):
		"""The __call__ function is the main function of a JAX module.

		Args:
		    self: Represent the instance of the class
		    input_ids: Optional[chex.Array]: Pass in the input tokens
		    input_embeds: Optional[chex.Array]: Pass in the embedded
		        tokens
		    cache_params: dict: Pass in the past cache_params from a
		        previous call to __call__
		    params: dict: Pass in the parameters of the model
		    dropout_rng: jax.random.PRNGKey: Make sure that the dropout
		        is applied in a random way
		    train: bool: Determine whether to use dropout or not
		    output_hidden_states: Optional[bool]: Return the hidden
		        states of all layers
		    return_dict: Optional[bool]: Determine whether to return a
		        dictionary or not
		    extra_embedding: Optional[Union[jnp.ndarray,None]]: Pass in
		        the embedding for the input_ids
		    add_params_field: bool: Add the params field to the inputs
		        dictionary

		Returns:
		    A tuple of the following:
		"""
		output_hidden_states = (
			output_hidden_states
			if output_hidden_states is not None
			else self.config.output_hidden_states
		)
		return_dict = return_dict if return_dict is not None else self.config.return_dict

		if cache_params is not None:
			assert isinstance(
				cache_params, FlaxMamba2Cache
			), f"Wrong cache input_type of {type(cache_params)}"
		rngs = {}
		if dropout_rng is not None:
			rngs["dropout"] = dropout_rng

		rngs["params"] = jax.random.key(0)

		inputs = (
			{"params": params or self.params} if add_params_field else params or self.params
		)

		return self.module.apply(
			inputs,
			input_ids=input_ids,
			input_embeds=input_embeds,
			attention_mask=attention_mask,
			cache_params=cache_params,
			deterministic=not train,
			use_cache=use_cache,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
			rngs=rngs,
			mutable=False,
		)


@register_module(
	"base-module",
	config=Mamba2Config,
	model_type="mamba2",
	embedding_layer_names=["embeddings"],
)
class FlaxMamba2Model(FlaxMambaPretrainedModel):
	module_class = FlaxMamba2Module


@register_module(
	"causal-language-model",
	config=Mamba2Config,
	model_type="mamba2",
	embedding_layer_names=["embeddings"],
)
class FlaxMamba2ForCausalLM(FlaxMambaPretrainedModel):
	module_class = FlaxMamba2ForCausalLMModule

	def update_inputs_for_generation(
		self,
		outputs: Mamba2Output,
		model_kwargs: Dict[str, Any],
		**kwargs,
	) -> Dict[str, Any]:
		model_kwargs["cache_params"] = outputs.get("cache_params", None)
		return model_kwargs

	def prepare_inputs_for_generation(
		self,
		input_ids,
		max_length,
		attention_mask: Optional[chex.Array] = None,
	):
		batch_size, seq_length = input_ids.shape
		extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
		if attention_mask is not None:
			extended_attention_mask = jax.lax.dynamic_update_slice(
				extended_attention_mask,
				attention_mask,
				(0, 0),
			)
		return {
			"cache_params": self.init_cache(batch_size=batch_size, max_length=max_length),
			"attention_mask": extended_attention_mask,
		}

	def init_cache(self, batch_size, max_length):
		return FlaxMamba2Cache(self.config, batch_size, self.module.dtype)
