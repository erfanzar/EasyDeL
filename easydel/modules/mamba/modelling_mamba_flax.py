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

import functools
import itertools
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, TypeVar, Union

import chex
import flax.struct
import jax
import jax.numpy as jnp
from einops import einsum, repeat
from flax import nnx as nn
from jax import lax

from easydel.etils.etils import EasyDeLGradientCheckPointers
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import register_module
from easydel.infra.modeling_outputs import FlaxBaseModelOutput
from easydel.infra.utils import (
	ACT2FN,
	get_dot_general_by_bits,
	get_gradient_checkpoint_policy,
)
from easydel.layers.caching import MambaCache
from easydel.layers.caching.mamba_cache import MambaCacheMetaData, MambaCacheView
from easydel.layers.norms import RMSNorm as MambaRMSNorm
from easydel.modules.mamba.mamba_configuration import MambaConfig as MambaConfig


def init_to_value(x, dtype):
	return lambda _, shape, dtype: jnp.broadcast_to(jnp.asarray(x, dtype=dtype), shape)


@flax.struct.dataclass
class MambaOutput(FlaxBaseModelOutput):
	last_hidden_state: chex.Array = None
	cache: Optional[MambaCache] = None
	hidden_states: Optional[Tuple[chex.Array]] = None


@flax.struct.dataclass
class MambaCausalLMOutput(FlaxBaseModelOutput):
	logits: chex.Array = None
	cache: Optional[MambaCache] = None
	hidden_states: Optional[Tuple[chex.Array]] = None


_T = TypeVar("_T")


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


class Lambda(nn.Module):
	fn: Callable

	def __call__(self, x, **kwargs):
		return self.fn(x, **kwargs)


class MambaConv1D(nn.Module):
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
		precision: Optional[Union[str, lax.Precision]] = None,
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
			rhs=jnp.asarray(self.kernel.value, dtype=self.dtype),
			window_strides=(self.stride,),
			padding=((self.padding, self.padding),),
			rhs_dilation=(self.dilation,),
			feature_group_count=self.groups,
		)

		if self.use_bias:
			x = x + jnp.asarray(self.bias.value.reshape(1, -1, 1), dtype=self.dtype)

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


class MambaMixer(nn.Module):
	def __init__(
		self,
		config: MambaConfig,
		layer_idx: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: Optional[Union[str, lax.Precision]] = None,
		*,
		rngs: nn.Rngs,
	) -> None:
		self.config = config
		self.layer_idx = layer_idx
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision

		hidden_size = config.hidden_size
		ssm_state_size = config.state_size
		intermediate_size = config.intermediate_size
		time_step_rank = config.time_step_rank
		conv_kernel_size = config.conv_kernel

		self.conv1d = nn.Conv(
			in_features=intermediate_size,
			out_features=intermediate_size,
			use_bias=config.use_conv_bias,
			kernel_size=config.conv_kernel,
			feature_group_count=intermediate_size,
			padding=config.conv_kernel - 1,
			rngs=rngs,
		)

		self.activation = config.hidden_act
		self.act = ACT2FN[config.hidden_act]

		dt_init_std = time_step_rank**-0.5 * config.time_step_scale
		if config.time_step_init_scheme == "constant":
			init_kernel_dt = nn.initializers.constant(dt_init_std, dtype=param_dtype)
		elif config.time_step_init_scheme == "random":

			def init_kernel_dt(key, _shape, _dtype):
				return (
					jax.nn.initializers.uniform(scale=dt_init_std * 2, dtype=param_dtype)(
						key, _shape, _dtype
					)
					- dt_init_std
				)

		else:
			init_kernel_dt = nn.initializers.normal(config.initializer_range, param_dtype)

		dt = jax.lax.clamp(
			config.time_step_floor,
			jnp.exp(
				jax.random.normal(
					key=rngs.params(),
					shape=(intermediate_size,),
					dtype=jnp.float32,
				)
				* (jnp.log(config.time_step_max) - jnp.log(config.time_step_min))
				+ jnp.log(config.time_step_min)
			),
			config.time_step_max,
		)
		inv_dt = dt + jnp.log(-jnp.expm1(-dt))

		linear_class = functools.partial(
			nn.Linear,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		self.in_proj = linear_class(
			hidden_size,
			intermediate_size * 2,
			use_bias=config.use_bias,
			rngs=rngs,
		)
		self.x_proj = linear_class(
			intermediate_size,
			time_step_rank + ssm_state_size * 2,
			use_bias=False,
			rngs=rngs,
		)
		self.dt_proj = linear_class(
			time_step_rank,
			intermediate_size,
			use_bias=True,
			kernel_init=init_kernel_dt,
			bias_init=lambda _, shape, dtype: inv_dt,
			rngs=rngs,
		)
		self.out_proj = linear_class(
			intermediate_size,
			hidden_size,
			use_bias=config.use_bias,
			rngs=rngs,
		)
		A = repeat(jnp.arange(1, ssm_state_size + 1), "n -> d n", d=intermediate_size)

		self.A_log = nn.Param(jnp.log(A))
		self.D = nn.Param(jnp.ones(intermediate_size))

		self.ssm_state_size = ssm_state_size
		self.intermediate_size = intermediate_size
		self.conv_kernel_size = conv_kernel_size
		self.time_step_rank = time_step_rank

	def __call__(
		self,
		input_states: chex.Array,
		cache: Optional[MambaCacheView] = None,
		position_ids: Optional[chex.Array] = None,
		attention_mask: Optional[chex.Array] = None,
	):
		batch_size, seq_len, _ = input_states.shape
		dtype = input_states.dtype

		# 1. Gated MLP's linear projection
		projected_states = self.in_proj(input_states)
		projected_states = jnp.transpose(
			projected_states, (0, 2, 1)
		)  # [batch, 2 * intermediate_size, seq_len]
		hidden_states, gate = jnp.split(projected_states, 2, axis=1)

		if attention_mask is not None:
			hidden_states = hidden_states * jnp.expand_dims(attention_mask, 1)

		# 2. Convolution sequence transformation
		if cache is not None:
			ssm_state = jnp.array(cache.ssm_states)

			if position_ids.shape[0] == self.conv_kernel_size:
				conv_state = jnp.pad(
					hidden_states,
					((0, 0), (0, 0), (self.conv_kernel_size - hidden_states.shape[-1], 0)),
				)

				cache.update_conv_state(conv_state, position_ids)
				hidden_states = self.act(
					self.conv1d(hidden_states)[..., :seq_len]
				)  # [batch, intermediate_size, seq_len]
			else:
				conv_state = cache.update_conv_state(hidden_states, position_ids)
				hidden_states = jnp.sum(conv_state * self.conv1d.weight[:, 0, :], axis=-1)
				if self.use_conv_bias:
					hidden_states = hidden_states + self.conv1d.bias
				hidden_states = jnp.expand_dims(
					self.act(hidden_states).astype(dtype), -1
				)  # [batch, intermediate_size, 1]
		else:
			ssm_state = jnp.zeros(
				(batch_size, self.intermediate_size, self.ssm_state_size), dtype=dtype
			)
			hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])
			# [batch, intermediate_size, seq_len]

		if attention_mask is not None:
			hidden_states = hidden_states * jnp.expand_dims(attention_mask, 1)

		# 3. State Space Model sequence transformation
		# 3.a. Selection
		ssm_parameters = self.x_proj(jnp.transpose(hidden_states, (0, 2, 1)))
		time_step, B, C = jnp.split(
			ssm_parameters,
			[self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
			axis=-1,
		)
		discrete_time_step = self.dt_proj(time_step)  # [batch, seq_len, intermediate_size]
		discrete_time_step = jnp.transpose(
			jax.nn.softplus(discrete_time_step), (0, 2, 1)
		)  # [batch, intermediate_size, seq_len]

		# 3.b. Discretization
		A = -jnp.exp(self.A_log.astype(jnp.float32))  # [intermediate_size, ssm_state_size]
		discrete_A = jnp.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None])
		discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].astype(
			jnp.float32
		)
		deltaB_u = discrete_B * hidden_states[:, :, :, None].astype(jnp.float32)

		# 3.c perform the recurrence
		def scan_fn(carry, x):
			ssm_state = carry
			discrete_A_t, deltaB_u_t = x
			new_state = discrete_A_t * ssm_state + deltaB_u_t
			return new_state, new_state

		if cache is not None:
			# Explicit loop for inference or when using cache
			scan_outputs = []
			for i in range(seq_len):
				ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]
				scan_output = jnp.matmul(
					ssm_state.astype(dtype), jnp.expand_dims(C[:, i, :], -1)
				)
				scan_outputs.append(scan_output[:, :, 0])
			scan_output = jnp.stack(scan_outputs, axis=-1)
		else:
			# Use scan for training
			_, scan_output = jax.lax.scan(
				scan_fn,
				ssm_state,
				(jnp.moveaxis(discrete_A, 2, 0), jnp.moveaxis(deltaB_u, 2, 0)),
			)
			scan_output = jnp.matmul(scan_output, jnp.expand_dims(C, -1)).squeeze(-1)
			scan_output = jnp.moveaxis(scan_output, 0, 2)

		scan_output = scan_output + (hidden_states * self.D[None, :, None])
		scan_output = scan_output * self.act(gate)

		if cache is not None:
			cache.ssm_states = ssm_state

		# 4. Final linear projection
		contextualized_states = self.out_proj(jnp.transpose(scan_output, (0, 2, 1)))
		return contextualized_states


class MambaBlock(nn.Module):
	def __init__(
		self,
		config: MambaConfig,
		layer_idx: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: Optional[Union[str, lax.Precision]] = None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.layer_idx = layer_idx
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.residual_in_fp32 = config.residual_in_fp32
		self.norm = MambaRMSNorm(
			config.hidden_size,
			eps=config.layer_norm_epsilon,
			dtype=dtype,
			param_dtype=param_dtype,
		)
		block = MambaMixer
		if self.config.gradient_checkpointing != EasyDeLGradientCheckPointers.NONE:
			block = nn.remat(
				block,
				static_argnums=(1,),
				policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
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
		cache: Optional[MambaCacheView] = None,
		position_ids: Optional[chex.Array] = None,
		attention_mask: Optional[chex.Array] = None,
	) -> chex.Array:
		residual = hidden_states
		hidden_states = self.norm(hidden_states)
		if self.residual_in_fp32:
			residual = residual.astype(jnp.float32)
		hidden_states = self.mixer(
			hidden_states,
			cache,
			position_ids,
			attention_mask,
		)
		hidden_states = residual + hidden_states
		return hidden_states


@register_module(
	"base-module",
	config=MambaConfig,
	model_type="mamba",
	embedding_layer_names=["embeddings"],
)
class MambaModel(EasyDeLBaseModule):
	def __init__(
		self,
		config: MambaConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: Optional[Union[str, lax.Precision]] = None,
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
			num_embeddings=config.vocab_size,
			features=config.hidden_size,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)
		self.layers = [
			MambaBlock(
				config=config,
				layer_idx=layer_idx,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			for layer_idx in range(config.num_hidden_layers)
		]
		self.norm_f = MambaRMSNorm(
			config.hidden_size,
			eps=config.layer_norm_epsilon,
			dtype=dtype,
			param_dtype=param_dtype,
		)

	def __call__(
		self,
		input_ids: Optional[chex.Array] = None,
		input_embeds: Optional[chex.Array] = None,
		cache: Optional[MambaCache] = None,
		position_ids: Optional[chex.Array] = None,
		attention_mask: Optional[chex.Array] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		**kwargs,
	) -> Union[Tuple, MambaOutput]:
		output_hidden_states = (
			output_hidden_states
			if output_hidden_states is not None
			else self.config.output_hidden_states
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

		batch_size, sequence_length = input_embeds.shape[:2]
		if attention_mask is None:
			attention_mask = jnp.ones((batch_size, sequence_length), "i4")
		if position_ids is None:
			position_ids = jnp.broadcast_to(
				jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
				(batch_size, sequence_length),
			).astype(jnp.int32)
		if cache is None:
			cache = MambaCache.init_empty(len(self.layers))

		hidden_states = input_embeds
		all_hidden_states = () if output_hidden_states else None
		for idx, block in enumerate(self.layers):
			hidden_states = block(
				hidden_states=hidden_states,
				cache=cache.views[idx],
				attention_mask=attention_mask,
				position_ids=position_ids,
			)

			if output_hidden_states:
				all_hidden_states = all_hidden_states + (hidden_states,)

		hidden_states = self.norm_f(hidden_states)

		if output_hidden_states:
			all_hidden_states = all_hidden_states + (hidden_states,)

		if not return_dict:
			return tuple(
				v
				for v in [
					hidden_states,
					all_hidden_states,
					cache,
				]
				if v is not None
			)

		return MambaOutput(
			last_hidden_state=hidden_states,
			cache=cache,
			hidden_states=all_hidden_states,
		)

	def init_cache(self, batch_size: int, max_length: int):
		return MambaCache.init_layers_cache(
			num_hidden_layers=self.config.num_hidden_layers,
			dtype=self.dtype,
			partition_specs=jax.sharding.PartitionSpec(
				self.config.partition_axis.batch_axis,
				self.config.partition_axis.key_sequence_axis,
				self.config.partition_axis.head_axis,
				self.config.partition_axis.attention_dim_axis,
			),
			metadata=MambaCacheMetaData.create(
				batch_size=batch_size,
				sequence_length=max_length,
				num_heads=self.config.num_key_value_heads,
				head_dim=self.config.head_dim,
			),
		)


@register_module(
	"causal-language-model",
	config=MambaConfig,
	model_type="mamba",
	embedding_layer_names=["embeddings"],
)
class MambaForCausalLM(EasyDeLBaseModule):
	def __init__(
		self,
		config: MambaConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: Optional[Union[str, lax.Precision]] = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.backbone = MambaModel(
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
		input_ids: Optional[chex.Array] = None,
		input_embeds: Optional[chex.Array] = None,
		cache: Optional[MambaCache] = None,
		position_ids: Optional[chex.Array] = None,
		attention_mask: Optional[chex.Array] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		**kwargs,
	) -> Union[Tuple, MambaCausalLMOutput]:
		return_dict = (
			return_dict if return_dict is not None else self.config.use_return_dict
		)

		mamba_outputs = self.backbone(
			input_ids=input_ids,
			input_embeds=input_embeds,
			attention_mask=attention_mask,
			position_ids=position_ids,
			cache=cache,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)
		hidden_states = mamba_outputs[0]

		self.lm_head.kernel.value = self.backbone.embeddings.embedding.value.T
		logits = self.lm_head(hidden_states).astype(jnp.float32)

		if not return_dict:
			return (logits,) + mamba_outputs[1:]

		return MambaCausalLMOutput(
			logits=logits,
			cache=mamba_outputs.cache,
			hidden_states=mamba_outputs.hidden_states,
		)

	def update_inputs_for_generation(
		self,
		outputs: MambaOutput,
		model_kwargs: Dict[str, Any],
		**kwargs,
	) -> Dict[str, Any]:
		model_kwargs["cache"] = outputs.get("cache", None)
		return model_kwargs

	def prepare_inputs_for_generation(self, input_ids, max_length, **kwargs):
		return {
			"cache": kwargs.get("cache", None),
			"input_ids": input_ids,
		}

	def init_cache(self, batch_size: int, max_length: int):
		return MambaCache.init_layers_cache(
			num_hidden_layers=self.config.num_hidden_layers,
			dtype=self.dtype,
			partition_specs=jax.sharding.PartitionSpec(
				self.config.partition_axis.batch_axis,
				self.config.partition_axis.key_sequence_axis,
				self.config.partition_axis.head_axis,
				self.config.partition_axis.attention_dim_axis,
			),
			metadata=MambaCacheMetaData.create(
				batch_size=batch_size,
				sequence_length=max_length,
				num_heads=self.config.num_key_value_heads,
				head_dim=self.config.head_dim,
			),
		)
