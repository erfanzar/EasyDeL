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
import typing as tp

import chex
import flax.struct
import jax
import jax.numpy as jnp
from flax import nnx as nn
from jax import lax

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import register_module
from easydel.infra.modeling_outputs import FlaxBaseModelOutput
from easydel.infra.utils import (
	ACT2FN,
	auto_remat,
)
from easydel.layers.caching.mamba2_cache import (
	Mamba2Cache,
	Mamba2CacheMetaData,
	Mamba2CacheView,
)
from easydel.layers.norms import RMSNorm as FlaxMamba2RMSNorm
from easydel.modules.mamba2.mamba2_configuration import Mamba2Config as Mamba2Config


def init_to_value(x, dtype):
	return lambda *_: x.astype(dtype)


@flax.struct.dataclass
class Mamba2Output(FlaxBaseModelOutput):
	last_hidden_state: chex.Array = None
	cache_params: tp.Optional[Mamba2Cache] = None
	hidden_states: tp.Optional[tp.Tuple[chex.Array]] = None


@flax.struct.dataclass
class Mamba2CausalLMOutput(FlaxBaseModelOutput):
	logits: chex.Array = None
	cache_params: tp.Optional[Mamba2Cache] = None
	hidden_states: tp.Optional[tp.Tuple[chex.Array]] = None


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


_T = tp.TypeVar("_T")


def create_tuple_parser(
	n: int,
) -> tp.Callable[[tp.Union[_T, tp.Sequence[_T]]], tuple[_T, ...]]:
	def parse(x: tp.Union[_T, tp.Sequence[_T]]) -> tuple[_T, ...]:
		if isinstance(x, tp.Sequence):
			if len(x) == n:
				return tuple(x)
			else:
				raise ValueError(f"x!=n ({x}!=({n}))")
		else:
			return tuple(itertools.repeat(x, n))

	return parse


class Conv1D(nn.Module):
	def __init__(
		self,
		features: int,
		kernel_size: int = 1,
		stride: int = 1,
		padding: int = 0,
		dilation: int = 1,
		groups: int = 1,
		use_bias: bool = True,
		num_spatial_dims: int = 1,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[str, lax.Precision]] = None,
		*,
		rngs: nn.Rngs,
	):
		kernel_shape = (features, 1, kernel_size)
		self.kernel = nn.Param(
			nn.initializers.lecun_normal(dtype=param_dtype)(
				rngs.params(),
				kernel_shape,
				param_dtype,
			),
		)

		if use_bias:
			self.bias = nn.Param(
				nn.initializers.zeros(
					rngs.params(),
					shape=(features,),
					dtype=param_dtype,
				)
			)

		self.features = features
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.dilation = dilation
		self.groups = groups
		self.use_bias = use_bias
		self.num_spatial_dims = num_spatial_dims
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision

	def __call__(self, x):
		unbatched_rank = self.num_spatial_dims + 2
		if x.ndim != unbatched_rank:
			raise ValueError(
				f"Input to `Conv` needs to have rank {unbatched_rank},"
				f" but input has shape {x.shape}.",
			)

		x = lax.conv_general_dilated(
			lhs=x,
			rhs=jnp.asarray(jnp.swapaxes(self.kernel.value, 0, 2), dtype=self.dtype),
			window_strides=(self.stride,),
			padding=((self.padding, self.padding),),
			rhs_dilation=(self.dilation,),
			feature_group_count=self.groups,
		)

		if self.use_bias:
			x = x + jnp.asarray(self.bias.value.reshape(1, -1, 1), dtype=self.dtype)

		return x


class MambaRMSNormGated(nn.Module):
	def __init__(
		self,
		hidden_size: int,
		eps: float = 1e-6,
		dtype: jnp.dtype = jnp.float32,
	):
		self.hidden_size = hidden_size
		self.eps = eps
		self.dtype = dtype
		self.kernel = nn.Param(
			jnp.ones((self.hidden_size,), self.dtype),
		)

	def __call__(self, hidden_states, gate=None):
		input_dtype = hidden_states.dtype
		hidden_states = hidden_states.astype(jnp.float32)

		if gate is not None:
			gate = gate.astype(jnp.float32)
			hidden_states = hidden_states * jax.nn.silu(gate)

		variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
		hidden_states = hidden_states * jax.lax.rsqrt(variance + self.eps)

		return (self.kernel.value * hidden_states).astype(input_dtype)


class Mamba2Mixer(nn.Module):
	def __init__(
		self,
		config: Mamba2Config,
		layer_idx: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[str, lax.Precision]] = None,
		*,
		rngs: nn.Rngs,
	) -> None:
		self.config = config
		self.layer_idx = layer_idx
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision

		self.num_heads = config.num_heads
		self.hidden_size = config.hidden_size
		self.ssm_state_size = config.state_size
		self.conv_kernel_size = config.conv_kernel
		self.intermediate_size = int(config.expand * self.hidden_size)
		self.time_step_rank = int(config.time_step_rank)
		self.use_conv_bias = config.use_conv_bias
		self.activation = config.hidden_act
		self.act = ACT2FN[config.hidden_act]

		self.norm_before_gate = config.norm_before_gate
		self.layer_norm_epsilon = config.layer_norm_epsilon
		self.rms_norm = config.rms_norm

		self.n_groups = config.n_groups
		self.head_dim = config.head_dim
		self.chunk_size = config.chunk_size

		self.time_step_limit = config.time_step_limit
		self.time_step_min = config.time_step_min
		self.time_step_max = config.time_step_max

		self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
		self.conv1d = Conv1D(
			features=self.conv_dim,
			kernel_size=self.config.conv_kernel,
			groups=self.conv_dim,
			stride=1,
			padding=self.config.conv_kernel - 1,
			use_bias=self.config.use_conv_bias,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		projection_size = self.intermediate_size + self.conv_dim + self.num_heads

		self.in_proj = nn.Linear(
			self.hidden_size,
			projection_size,
			use_bias=self.config.use_bias,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		dt = jax.lax.clamp(
			self.config.time_step_floor,
			jnp.exp(
				jax.random.normal(
					key=rngs.params(),
					shape=(self.config.num_heads,),
					dtype=self.param_dtype,
				)
				* (jnp.log(self.config.time_step_max) - jnp.log(self.config.time_step_min))
				+ jnp.log(self.config.time_step_min)
			).astype(jnp.float32),
			1e9,
		)

		inv_dt = dt + jnp.log(-jnp.expm1(-dt))
		self.dt_bias = nn.Param(inv_dt.astype(self.param_dtype))

		self.A_log = nn.Param(
			jnp.log(
				jnp.arange(1, self.num_heads + 1, dtype=jnp.float32),
			).astype(self.param_dtype),
		)
		self.D = nn.Param(jnp.ones(self.num_heads, dtype=self.param_dtype))

		self.norm = MambaRMSNormGated(
			self.intermediate_size,
			eps=self.layer_norm_epsilon,
			dtype=self.param_dtype,
		)
		self.out_proj = nn.Linear(
			self.intermediate_size,
			self.hidden_size,
			use_bias=self.config.use_bias,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

	def __call__(
		self,
		input_states: chex.Array,
		cache_params: tp.Optional[Mamba2CacheView] = None,
		cache_position: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
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
				conv_state = cache_params.conv_states
				# [batch, intermediate_size, conv_kernel_size]
				conv_state = jnp.roll(conv_state, shifts=-1, axis=-1)
				# handle batched generation - states are copied through
				conv_state[:, :, -1] = (
					hidden_states[:, 0, :] if hidden_states.ndim == 3 else hidden_states
				)
				cache_params.conv_states = jax.lax.dynamic_update_slice(
					cache_params.conv_states,
					conv_state,
					(0, 0, 0, 0),
				)
				hidden_states = jnp.sum(conv_state * self.conv1d.kernel.value[:, 0, :], dim=-1)
				if self.use_conv_bias:
					hidden_states += self.conv1d.bias.value
				hidden_states = self.act(hidden_states).astype(dtype)[:, None, ...]
			# [batch, 1, intermediate_size] : decoding
			else:
				hidden_states = jnp.swapaxes(hidden_states, 2, 1)

				pad_width = [
					(0, 0),
					(0, 0),
					(self.conv_kernel_size - hidden_states.shape[-1], 0),
				]
				conv_state = jnp.pad(hidden_states, pad_width)

				cache_params.conv_states = jax.lax.dynamic_update_slice(
					cache_params.conv_states,
					conv_state,
					(0, 0, 0, 0),
				)

				# Apply convolution and activation
				hidden_states = self.conv1d(hidden_states)
				hidden_states = jnp.swapaxes(hidden_states, 2, 1)
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
				jnp.swapaxes(
					self.conv1d(
						jnp.swapaxes(
							hidden_states,
							2,
							1,
						)
					)[..., :seq_len],
					2,
					1,
				)
			)
			hidden_states, B, C = jnp.split(
				hidden_states,
				[
					self.intermediate_size,
					self.intermediate_size + self.n_groups * self.ssm_state_size,
				],
				axis=-1,
			)
			A = -jnp.exp(self.A_log.value.astype("float32"))  # [num_heads]
			if cache_params is not None and cache_params.seqlen_offset > 0:
				dt = dt[:, None, ...] if dt.ndim == 2 else dt[:, 0, :][:, None, ...]
				dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], self.head_dim)
				dt_bias = self.dt_bias[..., None].expand(self.dt_bias.shape[0], self.head_dim)

				dt = jax.nn.softplus(dt + dt_bias.astype(dt.dtype))
				dt = jnp.clip(dt, min=self.time_step_min)
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


class Mamba2Block(nn.Module):
	def __init__(
		self,
		config: Mamba2Config,
		layer_idx: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[str, lax.Precision]] = None,
		*,
		rngs: nn.Rngs,
	) -> None:
		self.config = config
		self.layer_idx = layer_idx
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.residual_in_fp32 = config.residual_in_fp32
		self.norm = FlaxMamba2RMSNorm(
			dim=config.hidden_size,
			eps=config.layer_norm_epsilon,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)
		block = Mamba2Mixer
		(block,) = auto_remat(
			block,
			policy=config.gradient_checkpointing,
		)
		self.mixer = block(
			config=config,
			layer_idx=layer_idx,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

	def __call__(
		self,
		hidden_states: chex.Array,
		cache_params: tp.Optional[Mamba2CacheView] = None,
		cache_position: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
	) -> chex.Array:
		residual = hidden_states
		hidden_states = self.norm(hidden_states)
		if self.residual_in_fp32:
			residual = residual.astype(jnp.float32)
		hidden_states = self.mixer(
			hidden_states,
			cache_params,
			cache_position,
			attention_mask,
		)
		hidden_states = residual + hidden_states
		return hidden_states


@register_module(
	"base-module",
	config=Mamba2Config,
	model_type="mamba2",
	embedding_layer_names=["embeddings"],
)
class Mamba2Model(EasyDeLBaseModule):
	def __init__(
		self,
		config: Mamba2Config,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[str, lax.Precision]] = None,
		*,
		rngs: nn.Rngs,
	) -> None:
		super().__init__(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.embeddings = nn.Embed(
			config.vocab_size,
			config.hidden_size,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)
		self.layers = [
			Mamba2Block(
				config=config,
				layer_idx=layer_idx,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			for layer_idx in range(config.num_hidden_layers)
		]

		self.norm_f = FlaxMamba2RMSNorm(
			config.hidden_size,
			eps=config.layer_norm_epsilon,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

	def __call__(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		cache_params: tp.Optional[Mamba2Cache] = None,
		output_hidden_states: tp.Optional[bool] = None,
		return_dict: tp.Optional[bool] = None,
		cache_position: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		**kwargs,
	) -> tp.Union[tp.Tuple, Mamba2Output]:
		output_hidden_states = (
			output_hidden_states
			if output_hidden_states is not None
			else self.config.output_hidden_states
		)
		all_hidden_states = () if output_hidden_states else None
		return_dict = (
			return_dict if return_dict is not None else self.config.use_return_dict
		)

		if (input_ids is None) ^ (inputs_embeds is not None):
			raise ValueError(
				"You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
			)

		if inputs_embeds is None:
			inputs_embeds = self.embeddings(input_ids)
		if cache_params is None:
			cache_params = Mamba2Cache.init_empty(len(self.layers))
		hidden_states = inputs_embeds
		for idx, block in enumerate(self.layers):
			hidden_states = block(
				hidden_states=hidden_states,
				cache_params=cache_params.views[idx],
				cache_position=cache_position,
				attention_mask=attention_mask,
			)

			if output_hidden_states:
				all_hidden_states = all_hidden_states + (hidden_states,)

		hidden_states = self.norm_f(hidden_states)

		if output_hidden_states:
			all_hidden_states = all_hidden_states + (hidden_states,)

		if not return_dict:
			return tuple(
				v for v in [hidden_states, cache_params, all_hidden_states] if v is not None
			)

		return Mamba2Output(
			last_hidden_state=hidden_states,
			cache_params=cache_params,
			hidden_states=all_hidden_states,
		)


@register_module(
	"causal-language-model",
	config=Mamba2Config,
	model_type="mamba2",
	embedding_layer_names=["embeddings"],
)
class Mamba2ForCausalLM(EasyDeLBaseModule):
	def __init__(
		self,
		config: Mamba2Config,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[str, lax.Precision]] = None,
		*,
		rngs: nn.Rngs,
	) -> None:
		super().__init__(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.backbone = Mamba2Model(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.lm_head = nn.Linear(
			config.hidden_size,
			config.vocab_size,
			use_bias=False,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

	def __call__(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		cache_params: tp.Optional[Mamba2Cache] = None,
		output_hidden_states: tp.Optional[bool] = None,
		return_dict: tp.Optional[bool] = None,
		cache_position: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		**kwargs,
	) -> tp.Union[tp.Tuple, Mamba2CausalLMOutput]:
		return_dict = (
			return_dict if return_dict is not None else self.config.use_return_dict
		)
		mamba_outputs = self.backbone(
			input_ids=input_ids,
			inputs_embeds=inputs_embeds,
			attention_mask=attention_mask,
			cache_params=cache_params,
			cache_position=cache_position,
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

	def init_cache(self, batch_size: int, max_length: int):
		return Mamba2Cache.init_layers_cache(
			metadata=Mamba2CacheMetaData(
				batch_size=batch_size,
				intermediate_size=int(self.config.expand * self.config.hidden_size),
				conv_kernel_size=self.config.conv_kernel,
				head_dim=self.config.head_dim,
				n_groups=self.config.n_groups,
				state_size=self.config.state_size,
				num_heads=self.config.num_heads,
			),
			dtype=self.dtype,
			num_hidden_layers=self.config.num_hidden_layers,
		)

	def prepare_inputs_for_generation(
		self,
		input_ids,
		inputs_embeds=None,
		cache_params: tp.Optional[Mamba2Cache] = None,
		cache_position: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		**kwargs,
	):
		if inputs_embeds is not None:
			past_len = inputs_embeds.shape[1] + input_ids.shape[1]
		else:
			past_len = input_ids.shape[1]
		if cache_params is None:
			cache_params = self.init_cache(input_ids.shape[0], 0)
		if attention_mask.shape[1] < past_len:
			extended_mask = jnp.ones(
				(
					attention_mask.shape[0],
					past_len - attention_mask.shape[1],
				),
				"i4",
			)
			attention_mask = jnp.concatenate([attention_mask, extended_mask], axis=1)
		model_inputs = {}
		if inputs_embeds is not None and cache_params is None:
			model_inputs.update({"inputs_embeds": inputs_embeds})

		model_inputs.update(
			{
				"attention_mask": attention_mask,
				"cache_params": cache_params,
				"cache_position": cache_position,
			}
		)
		return self.prepare_inputs_for_call(model_inputs)

	def update_inputs_for_generation(self, model_outputs, model_kwargs):
		model_outputs.cache_params.update_seq(1)
		model_kwargs["cache_params"] = model_outputs.cache_params
		return model_kwargs
