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


import math
import typing as tp

import chex
import jax.lax
from flax import linen as nn
from flax.core import FrozenDict, freeze, unfreeze
from flax.struct import dataclass
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import numpy as jnp

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import register_module
from easydel.infra.modeling_outputs import ModelOutput
from easydel.modules.rwkv.rwkv_configuration import RwkvConfig as RwkvConfig


@dataclass
class RwkvOutput(ModelOutput):
	last_hidden_state: chex.Array = None
	state: tp.Optional[tp.Tuple[chex.Array, ...]] = None
	hidden_states: tp.Optional[tp.Tuple[chex.Array, ...]] = None
	attentions: tp.Optional[tp.Tuple[chex.Array, ...]] = None


@dataclass
class RwkvCausalLMOutput(ModelOutput):
	logits: chex.Array = None
	state: tp.Optional[tp.List[chex.Array]] = None
	hidden_states: tp.Optional[tp.Tuple[chex.Array, ...]] = None
	attentions: tp.Optional[tp.Tuple[chex.Array, ...]] = None


def init_to_value(x, dtype):
	return lambda _: x.astype(dtype)


def init_state(hidden_size):
	zeros = jnp.zeros(hidden_size)
	min_values = jnp.full(hidden_size, -jnp.inf)
	time_mix_state = (zeros, zeros, zeros, min_values)
	channel_mix_state = zeros
	return time_mix_state, channel_mix_state


@jax.jit
def rwkv_linear_attention(
	time_decay, time_first, key, value, state=None, return_state=False
):
	current_sequence_length = key.shape[1]
	output = jnp.zeros_like(key)

	if state is None:
		num_state = jnp.zeros_like(key[:, 0], dtype=jnp.float32)
		den_state = jnp.zeros_like(key[:, 0], dtype=jnp.float32)
		max_state = jnp.zeros_like(key[:, 0], dtype=jnp.float32) - 1e38
	else:
		num_state, den_state, max_state = state

	time_decay = -jnp.exp(time_decay)

	for current_index in range(current_sequence_length):
		current_key = key[:, current_index].float()
		current_value = value[:, current_index]

		max_for_output = jnp.maximum(max_state, current_key + time_first)
		e1 = jnp.exp(max_state - max_for_output)
		e2 = jnp.exp(current_key + time_first - max_for_output)
		numerator = e1 * num_state + e2 * current_value
		denominator = e1 * den_state + e2
		output[:, current_index] = (numerator / denominator).astype(output.dtype)

		max_for_state = jnp.maximum(max_state + time_decay, current_key)
		e1 = jnp.exp(max_state + time_decay - max_for_state)
		e2 = jnp.exp(current_key - max_for_state)
		num_state = e1 * num_state + e2 * current_value
		den_state = e1 * den_state + e2
		max_state = max_for_state

	if return_state or state is not None:
		state = [num_state, den_state, max_state]

	return output, state


class FlaxRwkvSelfAttention(nn.Module):
	config: RwkvConfig
	layer_id: int
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: tp.Optional[tp.Union[str, jax.lax.Precision]] = None

	def setup(self) -> None:
		config = self.config
		num_hidden_layers = config.num_hidden_layers
		layer_id = self.layer_id
		hidden_size = self.config.hidden_size
		attention_hidden_size = (
			config.attention_hidden_size
			if config.attention_hidden_size is not None
			else hidden_size
		)
		self.attention_hidden_size = attention_hidden_size

		ratio_0_to_1 = layer_id / (num_hidden_layers - 1)
		ratio_1_to_almost_0 = 1.0 - (layer_id / num_hidden_layers)
		zigzag = 0.5 * (jnp.arange(1, hidden_size + 1) % 3 - 1)
		time_first = jnp.full(hidden_size, math.log(0.3)) + zigzag
		h = jnp.arange(0, hidden_size)
		time_decay = -5 + 8 * (h / (hidden_size - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
		x = jnp.arange(hidden_size) / hidden_size

		time_mix_key = jnp.power(x, ratio_1_to_almost_0)
		time_mix_value = time_mix_key + 0.3 * ratio_0_to_1
		time_mix_receptance = jnp.power(x, 0.5 * ratio_1_to_almost_0)

		# This makes it easier to convert torch model into easydel since we use automated/small translation between
		# jax and torch

		# time_decay = time_decay.reshape(1, 1, hidden_size)
		# time_first = time_first.reshape(1, 1, hidden_size)
		# time_mix_key = time_mix_key.reshape(1, 1, hidden_size)
		# time_mix_value = time_mix_value.reshape(1, 1, hidden_size)
		# time_mix_receptance = time_mix_receptance.reshape(1, 1, hidden_size)

		self.time_decay = self.param(
			"time_decay",
			init_fn=init_to_value(time_decay, self.dtype),
		)
		self.time_first = self.param(
			"time_first",
			init_fn=init_to_value(time_first, self.dtype),
		)
		self.time_mix_key = self.param(
			"time_mix_key",
			init_fn=init_to_value(time_mix_key, self.dtype),
		)
		self.time_mix_value = self.param(
			"time_mix_value",
			init_fn=init_to_value(time_mix_value, self.dtype),
		)
		self.time_mix_receptance = self.param(
			"time_mix_receptance",
			init_fn=init_to_value(time_mix_receptance, self.dtype),
		)

		self.key = Dense(
			attention_hidden_size,
			use_bias=False,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
		)
		self.value = Dense(
			attention_hidden_size,
			use_bias=False,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
		)
		self.receptance = Dense(
			attention_hidden_size,
			use_bias=False,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
		)
		self.output = Dense(
			hidden_size,
			use_bias=False,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
		)

	def __call__(
		self,
		hidden: chex.Array,
		state: tp.Tuple[chex.Array, chex.Array, chex.Array, chex.Array],
	):
		sx, aa, bb, pp = state
		c_x = jnp.concatenate(
			(jnp.expand_dims(sx, 0), hidden[:-1, :]),
		)
		key_x = hidden * self.time_mix_key.reshape(-1) + c_x * (
			1 - self.time_mix_key.reshape(-1)
		)
		value_x = hidden * self.time_mix_value.reshape(-1) + c_x * (
			1 - self.time_mix_value.reshape(-1)
		)
		receptance_x = hidden * self.time_mix_receptance.reshape(-1) + c_x * (
			1 - self.time_mix_receptance.reshape(-1)
		)
		receptance_state = nn.sigmoid(self.receptance(receptance_x))
		key_states = self.key(key_x)
		value_states = self.value(value_x)

		def step(in_state, kv):
			(inner_aa, inner_bb, inner_p), (kk, vv) = in_state, kv
			ww = self.time_first.reshape(-1) + kk
			p = jnp.maximum(inner_p, ww)
			e1 = jnp.exp(inner_p - p)
			e2 = jnp.exp(ww - p)
			next_c_x = ((e1 * inner_aa + e2 * vv) / (e1 * inner_bb + e2)).astype(
				dtype=receptance_state.dtype
			)

			ww = -jnp.exp(self.time_decay.reshape(-1)) + inner_p
			p = jnp.maximum(ww, kk)
			e1 = jnp.exp(ww - p)
			e2 = jnp.exp(kk - p)
			inner_aa = e1 * inner_aa + e2 * vv
			inner_bb = e1 * inner_bb + e2
			inner_p = p
			next_inner_state = (inner_aa, inner_bb, inner_p)
			return next_inner_state, next_c_x

		(aa, bb, pp), c_x = jax.lax.scan(step, (aa, bb, pp), (key_states, value_states))
		out = hidden + self.output(receptance_state * c_x)
		next_state = (hidden[-1, :], aa, bb, pp)
		return out, next_state


class FlaxRwkvFeedForward(nn.Module):
	config: RwkvConfig
	layer_id: int
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: tp.Optional[tp.Union[str, jax.lax.Precision]] = None

	def setup(self):
		config = self.config
		hidden_size = config.hidden_size
		layer_id = self.layer_id
		num_hidden_layers = self.config.num_hidden_layers
		intermediate_size = (
			config.intermediate_size
			if config.intermediate_size is not None
			else 4 * config.hidden_size
		)

		x = jnp.arange(hidden_size) / hidden_size

		ratio_1_to_almost_0 = 1.0 - (layer_id / num_hidden_layers)
		time_mix_key = jnp.power(x, ratio_1_to_almost_0)
		time_mix_receptance = jnp.power(x, 0.5 * ratio_1_to_almost_0)

		# This makes it easier to convert torch model into easydel since we use automated/small translation between
		# jax and torch

		# time_mix_key = time_mix_key.reshape(1, 1, -1)
		# time_mix_receptance = time_mix_receptance.reshape(1, 1, -1)

		self.time_mix_key = self.param(
			"time_mix_key", init_fn=init_to_value(time_mix_key, self.param_dtype)
		)
		self.time_mix_receptance = self.param(
			"time_mix_receptance",
			init_fn=init_to_value(time_mix_receptance, self.param_dtype),
		)

		self.key = Dense(
			intermediate_size,
			use_bias=False,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
		)
		self.receptance = Dense(
			hidden_size,
			use_bias=False,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
		)
		self.value = Dense(
			hidden_size,
			use_bias=False,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
		)

	def __call__(self, hidden, state):
		sx = jnp.concatenate((jnp.expand_dims(state, 0), hidden[:-1, :]))
		xk = hidden * self.time_mix_key.reshape(-1) + sx * (
			1 - self.time_mix_key.reshape(-1)
		)
		xr = hidden * self.time_mix_receptance.reshape(-1) + sx * (
			1 - self.time_mix_receptance.reshape(-1)
		)
		r = nn.sigmoid(self.receptance(xr))
		k = jnp.square(nn.relu(self.key(xk)))
		return r * self.value(k), hidden[-1, :]


class SingleStandFlaxRwkvBlock(nn.Module):
	config: RwkvConfig
	layer_id: int
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: tp.Optional[tp.Union[str, jax.lax.Precision]] = None

	def setup(self):
		config = self.config
		layer_id = self.layer_id

		if layer_id == 0:
			self.pre_ln = nn.LayerNorm(
				epsilon=config.layer_norm_epsilon,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
			)

		self.ln1 = nn.LayerNorm(
			epsilon=config.layer_norm_epsilon,
			dtype=dtype,
			param_dtype=param_dtype,
		)
		self.ln2 = nn.LayerNorm(
			epsilon=config.layer_norm_epsilon,
			dtype=dtype,
			param_dtype=param_dtype,
		)

		self.attention = FlaxRwkvSelfAttention(
			config=config,
			layer_id=layer_id,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
		)
		self.feed_forward = FlaxRwkvFeedForward(
			config=config,
			layer_id=layer_id,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
		)

	def __call__(self, hidden, state=None, output_attentions: bool = False):
		if state is None:
			state = init_state(self.config.hidden_size)

		self_state, ffd_state = state
		if self.layer_id == 0:
			hidden = self.pre_ln(hidden)

		attention, self_state = self.attention(
			self.ln1(hidden),
			state=self_state,
		)
		hidden = hidden + attention

		feed_forward, ffd_state = self.feed_forward(self.ln2(hidden), state=ffd_state)
		hidden = hidden + feed_forward

		outputs = (hidden, (self_state, ffd_state))
		if output_attentions:
			outputs += (attention,)
		else:
			outputs += (None,)

		return outputs


FlaxRwkvBlock = nn.vmap(
	SingleStandFlaxRwkvBlock,
	in_axes=0,
	out_axes=0,
	split_rngs={"params": False},
	variable_axes={"params": None},
)


class FlaxRwkvBlockCollection(nn.Module):
	config: RwkvConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: tp.Optional[tp.Union[str, jax.lax.Precision]] = None

	def setup(self) -> None:
		self.blocks = [
			FlaxRwkvBlock(
				config=config,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				layer_id=idx,
				name=str(idx),
			)
			for idx in range(self.config.num_hidden_layers)
		]

		self.layers_are_rescaled = False

	def __call__(
		self,
		hidden_states: chex.Array,
		attention_mask: tp.Optional[chex.Array] = None,
		state: tp.Optional[tp.List[chex.Array]] = None,
		use_cache: tp.Optional[bool] = None,
		deterministic: tp.Optional[bool] = True,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		return_dict: tp.Optional[bool] = None,
	):
		all_hidden_states = ()
		all_self_attentions = ()
		use_cache = (
			use_cache
			if use_cache is not None
			else (self.config.use_cache if not deterministic else False)
		)
		for idx, block in enumerate(self.blocks):
			hidden_states, state, attentions = block(
				hidden_states, state=state, output_attentions=output_attentions
			)

			if (
				self.layers_are_rescaled
				and self.config.rescale_every > 0
				and (idx + 1) % self.config.rescale_every == 0
			):
				hidden_states = hidden_states / 2

			if output_hidden_states:
				all_hidden_states = all_hidden_states + (hidden_states,)

			if output_attentions:
				all_self_attentions = all_self_attentions + (attentions,)
		return hidden_states, all_hidden_states, all_self_attentions


class FlaxRwkvPretrainedModel(EasyDeLBaseModule):
	module_class: nn.Module
	config_class = RwkvConfig

	def __init__(
		self,
		config: RwkvConfig,
		input_shape: tp.Tuple = (1, 1),
		seed: int = 0,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
		_do_init: bool = True,
		**kwargs,
	):
		super().__init__(
			config=config,
			module=self.module_class(
				config=config,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				**kwargs,
			),
			input_shape=input_shape,
			seed=seed,
			dtype=dtype,
			_do_init=_do_init,
		)

	def init_weights(
		self, rng: jax.random.PRNGKey, input_shape: tp.Tuple, params: FrozenDict = None
	) -> FrozenDict[tp.Any, tp.Any] | tp.Mapping[str, tp.Any] | tp.Any:
		input_ids = jnp.zeros(input_shape, dtype="i4")
		attention_mask = jnp.ones((batch_size, sequence_length), "i4")
		params_rng, dropout_rng = jax.random.split(rng)
		rng_s = {"params": params_rng, "dropout": dropout_rng}
		module_init_outputs = self.module.init(
			rng_s, input_ids, attention_mask, return_dict=False
		)

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

	def __call__(  # noqa
		self,
		input_ids: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		state: tp.Optional[tp.List[chex.Array]] = None,
		use_cache: tp.Optional[bool] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		return_dict: tp.Optional[bool] = None,
		params: dict = None,
		dropout_rng: jax.random.PRNGKey = None,
		train: bool = False,
		extra_embedding: tp.Optional[tp.Union[jnp.ndarray, None]] = None,
		add_params_field: bool = False,
	):
		output_attentions = (
			output_attentions
			if output_attentions is not None
			else self.config.output_attentions
		)
		output_hidden_states = (
			output_hidden_states
			if output_hidden_states is not None
			else self.config.output_hidden_states
		)
		return_dict = return_dict if return_dict is not None else self.config.return_dict

		batch_size, sequence_length = input_ids.shape

		if attention_mask is None:
			attention_mask = jnp.ones((batch_size, sequence_length))

		rng_s = {}
		if dropout_rng is not None:
			rng_s["dropout"] = dropout_rng

		inputs = (
			{"params": params or self.params} if add_params_field else params or self.params
		)

		if self.config.bits is not None:
			rng_s["params"] = jax.random.key(0)

		mutable = False

		return self.module.apply(
			inputs,
			input_ids,
			attention_mask,
			inputs_embeds,
			state,
			use_cache,
			train,
			output_attentions,
			output_hidden_states,
			return_dict,
			rngs=rng_s,
			mutable=mutable,
		)

	def generate(self, *args, **kwargs):
		try:
			gen_output = super().generate(*args, **kwargs)
		except AttributeError as exc:
			if "past_key_values" in str(exc):
				raise AttributeError(
					"You tried to call `generate` with a decoding strategy that manipulates `past_key_values`. RWKV "
					"doesn't have that attribute, try another generation strategy instead. For the available "
					"generation strategies, check this doc:"
					" https://huggingface.co/docs/transformers/en/generation_strategies#decoding-strategies"
				) from None
			else:
				raise exc
		return gen_output

	def prepare_inputs_for_generation(
		self, input_ids, state=None, inputs_embeds=None, **kwargs
	):
		if state is not None:
			input_ids = input_ids[:, -1].unsqueeze(-1)
		if inputs_embeds is not None and state is None:
			model_inputs = {"inputs_embeds": inputs_embeds}
		else:
			model_inputs = {"input_ids": input_ids}

		model_inputs["state"] = state
		return model_inputs


@register_module(
	"base-module",
	config=RwkvConfig,
	model_type="rwkv",
	embedding_layer_names=["embed_tokens"],
	layernorm_names=["ln_out", "ln2", "ln1", "pre_ln"],
	rnn_based_or_rwkv=True,
)
class FlaxRwkvModel(nn.Module):
	config: RwkvConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: tp.Optional[tp.Union[str, jax.lax.Precision]] = None

	def setup(self):
		config = self.config
		self.embeddings = nn.Embed(
			config.vocab_size,
			config.hidden_size,
			dtype=dtype,
			param_dtype=param_dtype,
		)
		self.blocks = FlaxRwkvBlockCollection(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		self.ln_out = nn.LayerNorm(
			dtype=dtype,
			param_dtype=param_dtype,
		)

	def __call__(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		state: tp.Optional[tp.List[chex.Array]] = None,
		deterministic: bool = True,
		use_cache: tp.Optional[bool] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		return_dict: tp.Optional[bool] = None,
	) -> tp.Union[tp.Tuple, RwkvOutput]:
		output_attentions = (
			output_attentions
			if output_attentions is not None
			else self.config.output_attentions
		)
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

		if (input_ids is None) ^ (inputs_embeds is not None):
			raise ValueError(
				"You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
			)

		if inputs_embeds is None:
			inputs_embeds = self.embeddings(input_ids)

		if use_cache and state is None:
			shape = (
				inputs_embeds.shape[0],
				self.config.hidden_size,
				self.config.num_hidden_layers,
			)
			state = [
				jnp.zeros(
					*shape,
					dtype=inputs_embeds.dtype if i <= 1 else jnp.float32,
				)
				for i in range(5)
			]
			state[4] -= 1e30

		hidden_states = inputs_embeds

		hidden_states, all_hidden_states, all_self_attentions = self.blocks(
			hidden_states,
			attention_mask,
			state,
			use_cache,
			deterministic,
			output_attentions,
			output_hidden_states,
			return_dict,
		)

		hidden_states = self.ln_out(hidden_states)

		if output_hidden_states:
			all_hidden_states = all_hidden_states + (hidden_states,)

		if not return_dict:
			return tuple(
				x
				for x in [hidden_states, state, all_hidden_states, all_self_attentions]
				if x is not None
			)

		return RwkvOutput(
			last_hidden_state=hidden_states,
			state=state,
			hidden_states=all_hidden_states,
			attentions=all_self_attentions,
		)


@register_module(
	"causal-language-model",
	config=RwkvConfig,
	model_type="rwkv",
	embedding_layer_names=["embed_tokens"],
	layernorm_names=["ln_out", "ln2", "ln1", "pre_ln"],
	rnn_based_or_rwkv=True,
)
class FlaxRwkvForCausalLM(nn.Module):
	config: RwkvConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: tp.Optional[tp.Union[str, jax.lax.Precision]] = None

	def setup(self):
		config = self.config
		self.rwkv = FlaxRwkvModel(
			config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
		)
		self.head = Dense(
			config.vocab_size,
			use_bias=False,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
		)

	def __call__(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		state: tp.Optional[tp.List[chex.Array]] = None,
		deterministic: bool = True,
		use_cache: tp.Optional[bool] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		return_dict: tp.Optional[bool] = None,
	) -> tp.Union[tp.Tuple, RwkvCausalLMOutput]:
		return_dict = (
			return_dict if return_dict is not None else self.config.use_return_dict
		)

		rwkv_outputs = self.rwkv(
			input_ids,
			inputs_embeds=inputs_embeds,
			state=state,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
			deterministic=deterministic,
		)
		hidden_states = rwkv_outputs[0]

		logits = self.head(hidden_states)

		if not return_dict:
			return (logits,) + rwkv_outputs[1:]

		return RwkvCausalLMOutput(
			logits=logits,
			state=rwkv_outputs.state,
			hidden_states=rwkv_outputs.hidden_states,
			attentions=rwkv_outputs.attentions,
		)
