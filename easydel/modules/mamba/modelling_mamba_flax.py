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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import chex
import flax.struct
import jax
import jax.numpy as jnp
from einops import rearrange, repeat, einsum
from jax import lax

from easydel.etils.etils import EasyDeLGradientCheckPointers
from easydel.layers.caching import MambaCache
from easydel.layers.caching.mamba_cache import MambaCacheMetaData, MambaCacheView
from easydel.layers.norms import RMSNorm as MambaRMSNorm
from easydel.modules._base.base_module import EasyDeLBaseModule
from easydel.modules._base.factory import register_module
from easydel.modules._base.flax_modeling_utils import (
	ACT2FN,
	get_dot_general_by_bits,
	get_gradient_checkpoint_policy,
)
from easydel.modules.mamba.mamba_configuration import MambaConfig as MambaConfig
from easydel.modules.modeling_flax_outputs import FlaxBaseModelOutput
from flax import nnx as nn


def init_to_value(x, dtype):
	return lambda _, shape, dtype: jnp.broadcast_to(jnp.asarray(x, dtype=dtype), shape)


@flax.struct.dataclass
class MambaOutput(FlaxBaseModelOutput):
	last_hidden_state: chex.Array = None
	cache_params: Optional[List[chex.Array]] = None
	hidden_states: Optional[Tuple[chex.Array]] = None


@flax.struct.dataclass
class MambaCausalLMOutput(FlaxBaseModelOutput):
	logits: chex.Array = None
	cache_params: Optional[List[chex.Array]] = None
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

	def setup(self):
		kernel_shape = (self.features, 1, self.kernel_size)
		self.kernel = nn.Param(
			init_fn=nn.initializers.lecun_normal(dtype=self.param_dtype),
			shape=kernel_shape,
			name="kernel",
		)

		if self.use_bias:
			self.bias = nn.Param(
				init_fn=nn.initializers.zeros,
				shape=(self.features,),
				name="bias",
			)

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

		self.conv1d = MambaConv1D(
			features=intermediate_size,
			kernel_size=conv_kernel_size,
			groups=intermediate_size,
			stride=1,
			padding=conv_kernel_size - 1,
			use_bias=config.use_conv_bias,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
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
					key=rngs.params,
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

		self.A_log = nn.Param(
			init_fn=init_to_value(jnp.log(A), dtype),
			shape=(intermediate_size, ssm_state_size),
			name="A_log",
		)

		self.D = nn.Param(
			init_fn=init_to_value(jnp.ones(intermediate_size), dtype),
			shape=(intermediate_size,),
			name="D",
		)

		self.ssm_state_size = ssm_state_size
		self.intermediate_size = intermediate_size
		self.conv_kernel_size = conv_kernel_size
		self.time_step_rank = time_step_rank

	def __call__(
		self,
		input_states,
		cache: Optional[MambaCacheView] = None,
		inference: Optional[bool] = True,
	):
		batch_size, seq_len, _ = input_states.shape
		dtype = input_states.dtype
		projected_states = self.in_proj(input_states).transpose(0, 2, 1)
		hidden_states, gate = jnp.split(projected_states, 2, axis=1)

		# 2. Convolution sequence transformation
		if inference:
			ssm_state = cache.ssm_states
			if cache.seqlen_offset > 0:
				conv_state = cache.conv_states
				conv_state = jax.lax.dynamic_update_slice(conv_state, hidden_states, (0, 0, 0))
				hidden_states = jnp.sum(
					conv_state * rearrange(self.conv1d.kernel.value, "d 1 w -> 1 d w"),
					axis=-1,
				)
				if self.config.use_conv_bias:
					hidden_states += self.conv1d.bias.value
				hidden_states = jnp.expand_dims(self.act(hidden_states).astype(dtype), -1)
				# [batch, intermediate_size, 1] : decoding
			else:
				padding_amount = self.conv_kernel_size - hidden_states.shape[-1]
				conv_state = jnp.pad(
					hidden_states, ((0, 0), (0, 0), (padding_amount, 0)), mode="constant"
				)
				hidden_states = self.act(
					self.conv1d(hidden_states.transpose(0, 2, 1)).transpose(0, 2, 1)
				)
				cache.update_conv_state(conv_state)
		else:
			ssm_state = jnp.zeros(
				(batch_size, self.intermediate_size, self.ssm_state_size), dtype=dtype
			)
			hidden_states = self.act(
				self.conv1d(hidden_states.transpose(0, 2, 1)).transpose(0, 2, 1)
			)

		ssm_parameters = self.x_proj(hidden_states)
		time_step, B, C = jnp.split(
			ssm_parameters,
			indices_or_sections=[
				self.time_step_rank,
				self.time_step_rank + self.ssm_state_size,
			],
			axis=-1,
		)
		discrete_time_step = self.dt_proj(time_step)
		# [batch, seq_len, intermediate_size]
		discrete_time_step = jax.nn.softplus(discrete_time_step)
		# [batch, intermediate_size, seq_len]
		A = -jnp.exp(self.A_log.value.astype(jnp.float32))
		# [intermediate_size, ssm_state_size]
		modified_a = jnp.expand_dims(A, axis=0)
		modified_time_step = jnp.expand_dims(discrete_time_step, axis=-1)
		discrete_A = jnp.exp(modified_a * modified_time_step)
		# [batch, intermediate_size, seq_len, ssm_state_size]

		discrete_B = jnp.expand_dims(
			rearrange(B, "b l d -> b d l"),
			axis=2,
		).astype(jnp.float32)

		# [batch, intermediate_size, 1, ssm_state_size]

		deltaB_u = discrete_B * jnp.expand_dims(hidden_states, axis=-1).astype(jnp.float32)

		# 3.c perform the recurrence y ← SSM(A, B, C)(x)
		scan_outputs = []
		for i in range(seq_len):
			ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]
			# [batch, intermediate_size, ssm_state]

			scan_output = jax.lax.batch_matmul(
				ssm_state.astype(dtype),
				jnp.expand_dims(rearrange(C[:, i, :], "b d -> b d 1"), -1).astype(dtype),
			)
			# [batch, intermediate_size, 1]

			scan_outputs.append(scan_output[:, :, 0])

		scan_output = jnp.stack(scan_outputs, axis=1)
		# [batch, seq_len, intermediate_size]
		scan_output = scan_output + (
			hidden_states * self.D.value[jnp.newaxis, jnp.newaxis, :]
		)
		scan_output = scan_output * self.act(gate)

		if inference:
			cache.update_ssm_state(ssm_state)
			cache.positions += 1

		# 4. Final linear projection
		contextualized_states = self.out_proj(scan_output)
		# [batch, seq_len, hidden_size]
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
		inference: Optional[bool] = True,
	) -> chex.Array:
		residual = hidden_states
		hidden_states = self.norm(hidden_states)
		if self.residual_in_fp32:
			residual = residual.astype(jnp.float32)
		hidden_states = self.mixer(
			hidden_states,
			cache=cache,
			inference=inference,
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
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		inference: Optional[bool] = True,
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

		if inference and cache is None:
			cache = MambaCache.init_empty(len(self.layers.blocks))

		hidden_states = input_embeds
		all_hidden_states = () if output_hidden_states else None
		for idx, block in enumerate(self.blocks):
			hidden_states = block(
				hidden_states,
				cache=cache.views[idx],
				inference=inference,
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
					cache if inference else None,
					all_hidden_states,
				]
				if v is not None
			)

		return MambaOutput(
			last_hidden_state=hidden_states,
			cache_params=cache if inference else None,
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
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		inference: Optional[bool] = True,
		**kwargs,
	) -> Union[Tuple, MambaCausalLMOutput]:
		return_dict = (
			return_dict if return_dict is not None else self.config.use_return_dict
		)

		mamba_outputs = self.backbone(
			input_ids=input_ids,
			input_embeds=input_embeds,
			cache=cache,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
			inference=inference,
		)
		hidden_states = mamba_outputs[0]

		self.lm_head.kernel.value = self.backbone.embeddings.embedding.value.T
		logits = self.lm_head(hidden_states).astype(jnp.float32)

		if not return_dict:
			return (logits,) + mamba_outputs[1:]

		return MambaCausalLMOutput(
			logits=logits,
			cache_params=mamba_outputs.cache_params,
			hidden_states=mamba_outputs.hidden_states,
		)

	def update_inputs_for_generation(
		self,
		outputs: MambaOutput,
		model_kwargs: Dict[str, Any],
		**kwargs,
	) -> Dict[str, Any]:
		model_kwargs["cache"] = outputs.get("cache_params", None)
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
