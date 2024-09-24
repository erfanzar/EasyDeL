
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
from typing import Dict, Optional, Tuple, Union

import chex
import flax
import jax
from einops import rearrange
from flax import linen as nn
from flax.core import FrozenDict
from flax.linen import Dense
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from easydel.modules.flax_modeling_utils import (
	ACT2FN,
	FlaxAttentionModule,
	control_mlp_sharding,
	get_gradient_checkpoint_policy,
	with_sharding_constraint,
)
from easydel.modules.gpt_neo_x.gpt_neo_x_configuration import (
	GPTNeoXConfig as GPTNeoXConfig,
)
from easydel.modules.gpt_neo_x.kernels import gptneox_mlp_pallas
from easydel.modules.modeling_flax_outputs import FlaxBaseModelOutput
from easydel.modules.modeling_utils import EDPretrainedModel


def precompute_freqs_cis(
	dim: int, end: int, theta: float = 10000.0, dtype: jnp.dtype = jnp.bfloat16
) -> jnp.ndarray:
	freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
	t = jnp.arange(end)  # type: ignore
	freqs = jnp.outer(t, freqs).astype(dtype)
	sin, cos = jnp.sin(freqs), jnp.cos(freqs)
	freqs_cis = jnp.complex64(cos + 1j * sin)
	return jnp.asarray(freqs_cis)


def apply_rotary_emb(
	xq: jnp.ndarray,
	xk: jnp.ndarray,
	freqs_cis: jnp.ndarray,
	dtype: jnp.dtype = jnp.bfloat16,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
	reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
	reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)

	xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
	xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])

	freqs_cis = jnp.reshape(freqs_cis, (*freqs_cis.shape[:2], 1, *freqs_cis.shape[2:]))

	xq_out = xq_ * freqs_cis
	xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(
		*xq_out.shape[:-1], -1
	)

	xk_out = xk_ * freqs_cis
	xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(
		*xk_out.shape[:-1], -1
	)

	return xq_out.astype(dtype), xk_out.astype(dtype)


class FlaxGPTNeoXAttention(FlaxAttentionModule):
	config: GPTNeoXConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		self.head_size = self.config.hidden_size // self.config.num_attention_heads
		self.frequencies = precompute_freqs_cis(
			dtype=self.dtype,
			dim=self.head_size,
			end=self.config.max_position_embeddings,
		)
		self.w_qkv = Dense(3 * self.config.hidden_size)
		self.w_o = Dense(self.config.hidden_size)

		self.factor = jnp.sqrt(jnp.asarray(self.head_size, dtype=jnp.float32))
		self.bias = flax.linen.make_causal_mask(
			jnp.ones(
				(
					1,
					getattr(
						self.config,
						"mask_max_position_embeddings",
						self.config.max_position_embeddings,
					),
				)
			)
		)

	def __call__(
		self,
		hidden_states: chex.Array,
		attention_mask: chex.Array = None,
	):
		b, s, d = hidden_states.shape
		q, k, v = jnp.split(self.w_qkv(hidden_states), indices_or_sections=3, axis=-1)
		freq = self.frequencies[:s].reshape(1, s, -1)

		q = rearrange(q, "b s (h d) -> b s h d", h=self.config.num_attention_heads)
		k = rearrange(k, "b s (h d) -> b s h d", h=self.config.num_attention_heads)
		v = rearrange(v, "b s (h d) -> b s h d", h=self.config.num_attention_heads)
		bias = jnp.where(self.bias == 1, 0, jnp.finfo(hidden_states.dtype).min)
		q, k = apply_rotary_emb(q, k, freqs_cis=freq, dtype=self.dtype)

		q = with_sharding_constraint(
			q,
			jax.sharding.PartitionSpec(
				("dp", "fsdp"), "sp" if q.shape[1] != 1 else None, "tp", None
			),
		)
		k = with_sharding_constraint(
			k, jax.sharding.PartitionSpec(("dp", "fsdp"), "sp", "tp", None)
		)
		v = with_sharding_constraint(
			v, jax.sharding.PartitionSpec(("dp", "fsdp"), "sp", "tp", None)
		)
		attn = (
			jnp.einsum("...qhd,...khd->...hqk", q, k, precision=self.precision) * self.factor
		)
		attn = attn + bias[:, :, :s, :s]
		if attention_mask is not None:
			attn += attention_mask
		attn = jax.nn.softmax(attn, axis=-1)
		attn = with_sharding_constraint(
			attn, PartitionSpec(("dp", "fsdp"), "sp", None, None)
		)
		attn = jnp.einsum("...hqk,..khd->qhd", attn, v, precision=self.precision)
		attn = self.w_o(attn.reshape(b, s, d))
		return attn


class FlaxGPTNeoXMlp(nn.Module):
	config: GPTNeoXConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		self.dense_h_to_4h = Dense(self.config.intermediate_size)
		self.dense_4h_to_h = Dense(self.config.hidden_size)
		self.act = ACT2FN[self.config.hidden_act]

	def __call__(self, x):
		x = control_mlp_sharding(x, self.config.partition_axis)
		if (
			self.config.hardware_abstraction
			and self.dense_4h_to_h.variables.get("params", None) is not None
		):
			return jax.vmap(
				functools.partial(
					gptneox_mlp_pallas,
					act_fn=self.act,
					blocksize_k=self.config.pallas_k_block_size,
					blocksize_m=self.config.pallas_m_block_size,
					blocksize_n=self.config.pallas_n_block_size,
					prod_dtype=self.dtype,
					precision=self.precision,
				),
				in_axes=(0, None, None),
			)(
				x,
				self.dense_h_to_4h.variables["params"]["kernel"],
				self.dense_4h_to_h.variables["params"]["kernel"],
			)

		return self.dense_4h_to_h(self.act(self.dense_h_to_4h(x)))


class FlaxGPTNeoXBlock(nn.Module):
	config: GPTNeoXConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		self.use_parallel_residual = self.config.use_parallel_residual
		self.input_layernorm = nn.LayerNorm(
			epsilon=self.config.layer_norm_eps, dtype=self.dtype
		)
		self.post_attention_layernorm = nn.LayerNorm(
			epsilon=self.config.layer_norm_eps, dtype=self.dtype
		)
		self.attention = FlaxGPTNeoXAttention(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.mlp = FlaxGPTNeoXMlp(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)

	def __call__(
		self,
		hidden_states: chex.Array,
		attention_mask: chex.Array,
	):
		attn = self.attention(
			self.input_layernorm(hidden_states), attention_mask=attention_mask
		)

		if self.use_parallel_residual:
			mlp = self.mlp(self.post_attention_layernorm(hidden_states))
			hidden_states = mlp + hidden_states + attn
		else:
			hidden_states = attn + hidden_states
			hidden_states = (
				self.mlp(self.post_attention_layernorm(hidden_states)) + hidden_states
			)
		return hidden_states


class FlaxGPTNeoXCollection(nn.Module):
	config: GPTNeoXConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		block = FlaxGPTNeoXBlock
		if self.config.gradient_checkpointing != "":
			block = nn.remat(
				block,
				static_argnums=None,
				policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
			)
		self.blocks = [
			block(
				config=self.config,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
				name=str(i),
			)
			for i in range(self.config.num_hidden_layers)
		]

	def __call__(
		self,
		hidden_states: chex.Array,
		attention_mask: chex.Array,
	):
		for block in self.blocks:
			hidden_states = block(hidden_states, attention_mask=attention_mask)
		return hidden_states


class FlaxGPTNeoXModule(nn.Module):
	config: GPTNeoXConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		self.embed_in = nn.Embed(self.config.vocab_size, self.config.hidden_size)
		self.layers = FlaxGPTNeoXCollection(
			config=self.config,
			param_dtype=self.param_dtype,
			dtype=self.dtype,
			precision=self.precision,
		)
		self.final_layer_norm = nn.LayerNorm(
			epsilon=self.config.layer_norm_eps, dtype=self.dtype
		)

	def __call__(
		self,
		input_ids: jnp.dtype = None,
		attention_mask: Optional[chex.Array] = None,
		return_dict: Optional[bool] = None,
	):
		b, s = input_ids.shape
		hidden_state = self.embed_in(inputs=input_ids)
		hidden_state = self.final_layer_norm(
			self.layers(hidden_state=hidden_state, attention_mask=attention_mask)
		)
		if return_dict:
			return FlaxBaseModelOutput(last_hidden_state=hidden_state)
		else:
			return (hidden_state,)


class FlaxGPTNeoXPretrainedModel(EDPretrainedModel):
	module_class: nn.Module = None
	config_class = GPTNeoXConfig

	def __init__(
		self,
		config,
		_do_init=False,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		input_shape: Tuple = (1, 12),
	):
		module = self.module_class(config=config, dtype=dtype, param_dtype=param_dtype)
		super().__init__(
			_do_init=_do_init,
			module=module,
			config=config,
			dtype=dtype,
			input_shape=input_shape,
		)

	def init_weights(
		self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None
	) -> Dict:
		if params is None:
			params = self.module.init(
				rngs=rng,
				input_ids=jnp.ones(input_shape),
				attention_mask=jnp.ones(input_shape),
			)
		return params["params"]

	def __call__(
		self,
		input_ids,
		attention_mask=None,
		params: FrozenDict = None,
		add_params_field: bool = False,
		return_dict: bool = True,
		**kwargs,
	):
		params = (
			{"params": params or self.params} if add_params_field else params or self.params
		)
		predict = self.module.apply(
			params,
			input_ids=jnp.asarray(input_ids, dtype=jnp.int32),
			attention_mask=(
				jnp.asarray(attention_mask, dtype=jnp.int32)
				if attention_mask is not None
				else attention_mask
			),
			return_dict=return_dict,
		)
		return predict

	def prepare_inputs_for_generation(
		self, input_ids, max_length, attention_mask: Optional[chex.Array] = None
	):
		return {
			"attention_mask": attention_mask,
		}

	def update_inputs_for_generation(self, model_outputs, model_kwargs):
		return model_kwargs


class FlaxGPTNeoXModel(FlaxGPTNeoXPretrainedModel):
	module_class = FlaxGPTNeoXModule

	def get_input_embeddings(self):
		return self.module.wte

	def set_input_embeddings(self, value):
		self.module.wte = value


class FlaxGPTNeoXForCausalLMModule(nn.Module):
	config: GPTNeoXConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		self.transformer = FlaxGPTNeoXModule(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.lm_head = Dense(self.config.vocab_size, use_bias=False)

	def __call__(self, input_ids, attention_mask, return_dict: bool = False):
		pred = self.transformer(
			input_ids=input_ids, attention_mask=attention_mask, return_dict=True
		).last_hidden_state
		return self.lm_head(pred)


class FlaxGPTNeoXForCausalLM(FlaxGPTNeoXPretrainedModel):
	module_class = FlaxGPTNeoXForCausalLMModule

	def get_output_embeddings(self):
		return self.module.lm_head

	def get_decoder(self):
		return self.module.transformer

	def get_input_embeddings(self):
		return self.module.transformer.wte

	def set_output_embeddings(self, new_embeddings):
		self.module.lm_head = new_embeddings

	def set_decoder(self, decoder):
		self.module.transformer = decoder

	def set_input_embeddings(self, value):
		self.module.transformer.wte = value
