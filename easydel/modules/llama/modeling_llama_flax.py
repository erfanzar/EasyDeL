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
from functools import cached_property, partial
from typing import Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp

from easydel.etils.etils import EasyDeLGradientCheckPointers
from easydel.layers.attention import FlaxAttentionModule, FlexibleAttentionModule
from easydel.layers.norms import RMSNorm
from easydel.modules.factory import register_module
from easydel.modules.flax_modeling_utils import (
	ACT2FN,
	block_wise_ffn,
	control_mlp_sharding,
	get_dot_general_by_bits,
	get_gradient_checkpoint_policy,
	get_static_param_indices,
)

# easydel.modules
from easydel.modules.llama.llama_configuration import (
	LlamaConfig as LlamaConfig,
	VisionLlamaConfig as VisionLlamaConfig,
)
from easydel.modules.modeling_flax_outputs import (
	FlaxBaseModelOutput,
	FlaxCausalLMOutput,
)
from easydel.modules.modeling_utils import EasyDeLBaseModule
from flax import nnx as nn


class LlamaAttention(FlaxAttentionModule):
	"""
	FlaxLlamaAttention implements an attention mechanism with rotary embeddings.

	Attributes:
		config (LlamaConfig): Configuration for the attention module.
		dtype (jnp.dtype): Data type for computations (default is jnp.bfloat16).
		param_dtype (jnp.dtype): Data type for parameters (default is jnp.bfloat16).
		precision (Optional[Union[str, jax.lax.Precision]]): Precision setting for JAX operations (default is "fastest").
	"""

	def __init__(
		self,
		config: LlamaConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: Optional[Union[jax.lax.Precision, str]] = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__(config=config)
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.rngs = rngs
		self.hidden_size = config.hidden_size
		head_dim = config.hidden_size // config.num_attention_heads
		self.head_dim = getattr(config, "head_dim", head_dim)
		self.num_key_value_groups = (
			self.config.num_attention_heads // self.config.num_key_value_heads
		)

		if self.num_key_value_groups == 1:
			assert self.config.num_attention_heads == self.config.num_key_value_heads

		linear_class = partial(
			nn.Linear,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			use_bias=self.config.attention_bias,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			precision=self.precision,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)
		self.q_proj = linear_class(
			config.hidden_size,
			config.num_attention_heads * self.head_dim,
			rngs=rngs,
		)
		self.k_proj = linear_class(
			config.hidden_size,
			config.num_key_value_heads * self.head_dim,
			rngs=rngs,
		)
		self.v_proj = linear_class(
			config.hidden_size,
			config.num_key_value_heads * self.head_dim,
			rngs=rngs,
		)
		self.o_proj = linear_class(
			config.num_attention_heads * self.head_dim,
			config.hidden_size,
			rngs=rngs,
		)

		self.rotary = self.config.get_basic_rope(
			self.dtype,
			self.head_dim,
			self.head_dim,
			True,
		)

		self.attention_performer = FlexibleAttentionModule(
			attention_dropout=self.config.attention_dropout,
			num_q_heads=self.config.num_attention_heads,
			num_kv_heads=self.config.num_key_value_heads,
			head_dims=self.head_dim,
			precision=self.precision,
			force_float32_tpu=True,
			attn_mechanism=self.config.attn_mechanism,
			dtype=self.config.attn_dtype,
			mesh=self.config.mesh,
			sm_scale=1 / math.sqrt(self.head_dim),
			axis_name=self.config.attention_axis_name,
			base_config=self.config,
		)
		self.resid_dropout = nn.Dropout(
			rate=config.resid_pdrop,
			rngs=rngs,
		)

	def __call__(
		self,
		hidden_states: chex.Array,
		attention_mask: chex.Array,
		position_ids: chex.Array,
		causal_mask: chex.Array,
		segment_ids: Optional[chex.Array] = None,
		init_cache: bool = False,
		output_attentions: bool = False,
		fcm_mask: Optional[chex.Array] = None,
		frequencies: Optional[chex.Array] = None,
	) -> Tuple[chex.Array, chex.Array]:
		batch_size, sequence_length = hidden_states.shape[:2]
		query_states, key_states, value_states = (
			self.q_proj(hidden_states),
			self.k_proj(hidden_states),
			self.v_proj(hidden_states),
		)
		qshape = (
			batch_size,
			sequence_length,
			self.config.num_attention_heads,
			self.head_dim,
		)
		kv_shape = (
			batch_size,
			sequence_length,
			self.config.num_key_value_heads,
			self.head_dim,
		)
		query_states = query_states.reshape(qshape)
		key_states = key_states.reshape(kv_shape)
		value_states = value_states.reshape(kv_shape)

		query_states, key_states = self.rotary(
			positions=position_ids,
			query=query_states,
			key=key_states,
			frequencies=frequencies,
		)

		(
			query_states,
			key_states,
			value_states,
			attention_mask,
			attention_bias,
		) = self.concatenate_to_cache(
			init_cache=init_cache,
			query=query_states,
			key=key_states,
			value=value_states,
			attention_mask=attention_mask,
			causal_mask=causal_mask,
			fcm_mask=fcm_mask,
		)

		attentions = self.attention_performer(
			query_states=query_states,
			key_states=key_states,
			value_states=value_states,
			bias=attention_bias,
			attention_mask=attention_mask,
			causal=True,
			dropout_rng=self.rngs.params(),
			deterministic=True,
			query_sequence_length=query_states.shape[1],
			key_value_sequence_length=key_states.shape[1],
			uses_cache=init_cache,
			segment_ids=segment_ids,
			causal_mask=causal_mask,
		)
		attn_output = self.resid_dropout(
			self.o_proj(
				self.shard_attention_prod(
					attn_output=self._merge_heads(attentions.attention_outputs)
				)
			),
		)
		outputs = (
			(attn_output, attentions.attention_weights)
			if output_attentions
			else (attn_output, None)
		)
		return outputs


class LlamaMLP(nn.Module):
	"""
	Attributes:
		config (LlamaConfig): Configuration for the attention module.
		dtype (jnp.dtype): Data type for computations (default is jnp.bfloat16).
		param_dtype (jnp.dtype): Data type for parameters (default is jnp.bfloat16).
		precision (Optional[Union[str, jax.lax.Precision]]): Precision setting for JAX operations (default is "fastest").
	"""

	def __init__(
		self,
		config: LlamaConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: Optional[Union[jax.lax.Precision, str]] = None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		linear_class = partial(
			nn.Linear,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			use_bias=self.config.mlp_bias,
			kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
			precision=self.precision,
			rngs=rngs,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)
		self.gate_proj = linear_class(
			config.hidden_size,
			config.intermediate_size,
			rngs=rngs,
		)
		self.down_proj = linear_class(
			config.intermediate_size,
			config.hidden_size,
			rngs=rngs,
		)
		self.up_proj = linear_class(
			config.hidden_size,
			config.intermediate_size,
			rngs=rngs,
		)
		self.dropout = nn.Dropout(rate=self.config.resid_pdrop, rngs=rngs)
		self.act_fn = ACT2FN[self.config.hidden_act]

	def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
		"""The __call__ function is the main function of a class.
		It is called when an instance of the class (an object) is invoked as a function, i.e., obj(arguments).
		The __call__ method enables instances of a class to be called like standard Python functions.

		Args:
		    self: Represent the instance of the class
		    x: jnp.ndarray: Pass in the input to the layer

		Returns:
		    A tensor that is the result of applying a dropout function
		    to x
		"""

		x = control_mlp_sharding(x, self.config.partition_axis)
		x = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
		x = self.dropout(x)
		return x


class LlamaBlock(nn.Module):
	"""
	Attributes:
		config (LlamaConfig): Configuration for the attention module.
		dtype (jnp.dtype): Data type for computations (default is jnp.bfloat16).
		param_dtype (jnp.dtype): Data type for parameters (default is jnp.bfloat16).
		precision (Optional[Union[str, jax.lax.Precision]]): Precision setting for JAX operations (default is "fastest").
	"""

	def __init__(
		self,
		config: LlamaConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: Optional[Union[jax.lax.Precision, str]] = None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		attn_block = LlamaAttention
		mlp_block = LlamaMLP
		if self.config.gradient_checkpointing != EasyDeLGradientCheckPointers.NONE:
			attn_block.__call__ = nn.remat(
				attn_block.__call__,
				static_argnums=get_static_param_indices(attn_block.__call__),
				policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
			)

			mlp_block.__call__ = nn.remat(
				mlp_block.__call__,
				static_argnums=get_static_param_indices(mlp_block.__call__),
				policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
			)

		self.self_attn = attn_block(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
			rngs=rngs,
		)

		self.mlp = mlp_block(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
			rngs=rngs,
		)
		self.input_layernorm = RMSNorm(
			dim=self.config.hidden_size,
			eps=self.config.rms_norm_eps,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			rngs=rngs,
		)
		self.post_attention_layernorm = RMSNorm(
			dim=self.config.hidden_size,
			eps=self.config.rms_norm_eps,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			rngs=rngs,
		)

	def __call__(
		self,
		hidden_states: chex.Array,
		attention_mask: chex.Array,
		position_ids: chex.Array,
		causal_mask: chex.Array,
		segment_ids: Optional[chex.Array] = None,
		init_cache: bool = False,
		output_attentions: bool = False,
		fcm_mask: Optional[chex.Array] = None,
		frequencies: Optional[chex.Array] = None,
	):
		"""
		Forward pass of the module block.

		Args:
		    hidden_states (chex.Array): Input hidden states.
		    attention_mask (chex.Array): Mask to apply on the attention scores.
		    position_ids (chex.Array): Position indices for the tokens.
		    causal_mask (chex.Array): Causal mask for ensuring autoregressive behavior.
		    segment_ids (Optional[chex.Array]): Segment IDs for segment-based attention (optional).
		    init_cache (bool): If True, initializes cache for caching keys and values.
		    output_attentions (bool): If True, outputs attention weights alongside the hidden states.
		    fcm_mask (Optional[chex.Array]): fcm mask to be combined with attn mask and causal mask.
		Returns:
		    Tuple[chex.Array, chex.Array]: A tuple containing the attention output and the attention weights.
		"""
		attn_outputs = self.self_attn(
			self.input_layernorm(hidden_states),
			attention_mask,
			position_ids,
			causal_mask,
			segment_ids,
			init_cache,
			output_attentions,
			fcm_mask,
			frequencies,
		)
		attn_output = attn_outputs[0]
		hidden_states = hidden_states + attn_output

		feed_forward_input = self.post_attention_layernorm(hidden_states)

		if self.config.use_scan_mlp:
			feed_forward_hidden_states = block_wise_ffn(
				self.mlp,
				feed_forward_input,
				self.config.scan_mlp_chunk_size,
			)
		else:
			feed_forward_hidden_states = self.mlp(feed_forward_input)

		hidden_states = hidden_states + feed_forward_hidden_states

		return (hidden_states,) + attn_outputs[1:]


@register_module(
	"base-module",
	config=LlamaConfig,
	model_type="llama",
	embedding_layer_names=["embed_tokens"],
)
class LlamaModel(EasyDeLBaseModule):
	"""
	Attributes:
		config (LlamaConfig): Configuration for the attention module.
		dtype (jnp.dtype): Data type for computations (default is jnp.bfloat16).
		param_dtype (jnp.dtype): Data type for parameters (default is jnp.bfloat16).
		precision (Optional[Union[str, jax.lax.Precision]]): Precision setting for JAX operations (default is "fastest").
	"""

	def __init__(
		self,
		config: LlamaConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: Optional[Union[jax.lax.Precision, str]] = None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision

		self.embed_tokens = nn.Embed(
			num_embeddings=self.config.vocab_size,
			features=self.config.hidden_size,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
			rngs=rngs,
		)
		self.dropout = nn.Dropout(rate=self.config.embd_pdrop, rngs=rngs)
		self.layers = [
			LlamaBlock(
				config=self.config,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
				rngs=rngs,
			)
			for _ in range(self.config.num_hidden_layers)
		]
		self.norm = RMSNorm(
			self.config.hidden_size,
			eps=self.config.rms_norm_eps,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			rngs=rngs,
		)
		# self.causal_mask = RecreatorVariable(
		# 	self.config.get_basic_causal_mask(),
		# 	recreator=lambda: self.config.get_basic_causal_mask(),
		# )
		# self.frequencies = RecreatorVariable(
		# 	self.config.get_basic_frequencies(),
		# 	recreator=lambda: self.config.get_basic_frequencies(),
		# )

	@cached_property
	def causal_mask(self):
		return self.config.get_basic_causal_mask()

	@cached_property
	def frequencies(self):
		return self.config.get_basic_frequencies()

	def __call__(
		self,
		input_ids: chex.Array,
		attention_mask: Optional[chex.Array] = None,
		position_ids: Optional[chex.Array] = None,
		segment_ids: Optional[chex.Array] = None,
		input_embeds: Optional[chex.Array] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		init_cache: bool = False,
		return_dict: bool = True,
	) -> Union[FlaxBaseModelOutput, Tuple]:
		"""
		Forward pass through the Llama module.

		Args:
		    input_ids (chex.Array): Input tensor containing token IDs.
		    attention_mask (chex.Array): Mask for attention.
		    position_ids (chex.Array): Positional indices.
		    segment_ids (Optional[chex.Array]): Segment IDs for different input parts.
		    input_embeds (Optional[chex.Array]): Embedded input tensor.
		    output_attentions (Optional[bool]): If True, output attention weights.
		    output_hidden_states (Optional[bool]): If True, output hidden states.
		    init_cache (bool): If True, initialize cache for decoding.
		    return_dict (bool): If True, return a dictionary of outputs.

		Returns:
		    FlaxBaseModelOutput | Tuple: Model output, either as a named tuple or a standard tuple.
		"""
		if input_embeds is None and input_ids is not None:
			input_embeds = self.embed_tokens(input_ids.astype("i4"))
		else:
			raise ValueError("you should specify input_embeds or input_ids one of them")
		batch_size, sequence_length, _ = input_embeds.shape

		all_attentions = () if output_attentions else None
		all_hidden_states = () if output_hidden_states else None
		assert (
			sequence_length <= self.config.max_position_embeddings
		), f"Maximum Position Embedding Reached ! (Excepted <= {self.config.max_position_embeddings} got {sequence_length})"
		if attention_mask.ndim == 2:
			attention_mask = jnp.expand_dims(attention_mask, (1, 2))

		hidden_states = self.dropout(input_embeds)

		for block in self.layers:
			if output_hidden_states:
				all_hidden_states += (hidden_states,)

			layer_outputs = block(
				hidden_states=hidden_states,
				attention_mask=attention_mask,
				position_ids=position_ids,
				causal_mask=self.config.get_basic_causal_mask(),
				init_cache=init_cache,
				output_attentions=output_attentions,
				segment_ids=segment_ids,
				frequencies=self.config.get_basic_frequencies(),
			)
			hidden_states = layer_outputs[0]

			if output_attentions:
				all_attentions += (layer_outputs[1],)

		hidden_states = self.norm(hidden_states)

		if output_hidden_states:
			all_hidden_states += (hidden_states,)
			outputs = (hidden_states, all_hidden_states, all_attentions)
		else:
			outputs = (hidden_states, all_attentions)

		if not return_dict:
			return tuple(v for v in outputs if v is not None)

		return FlaxBaseModelOutput(
			last_hidden_state=hidden_states,
			hidden_states=outputs[1],
			attentions=outputs[-1],
		)


@register_module(
	"causal-language-model",
	config=LlamaConfig,
	model_type="llama",
	embedding_layer_names=["embed_tokens"],
)
class LlamaForCausalLM(EasyDeLBaseModule):
	"""
	Attributes:
		config (LlamaConfig): Configuration for the attention module.
		dtype (jnp.dtype): Data type for computations (default is jnp.bfloat16).
		param_dtype (jnp.dtype): Data type for parameters (default is jnp.bfloat16).
		precision (Optional[Union[str, jax.lax.Precision]]): Precision setting for JAX operations (default is "fastest").
	"""

	def __init__(
		self,
		config: LlamaConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: Optional[Union[jax.lax.Precision, str]] = None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.model = LlamaModel(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
			rngs=rngs,
		)

		self.lm_head = nn.Linear(
			self.config.hidden_size,
			self.config.vocab_size,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			use_bias=False,
			kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
			precision=self.precision,
			rngs=rngs,
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)

	def __call__(
		self,
		input_ids: Optional[chex.Array] = None,
		attention_mask: Optional[chex.Array] = None,
		position_ids: Optional[chex.Array] = None,
		segment_ids: Optional[chex.Array] = None,
		input_embeds: Optional[chex.Array] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		init_cache: bool = False,
		return_dict: bool = True,
	) -> Union[FlaxCausalLMOutput, Tuple]:
		"""
		Forward pass through the Llama module.

		Args:
		    input_ids (Optional[chex.Array]): Input tensor containing token IDs.
		    attention_mask (Optional[chex.Array]): Mask for attention.
		    position_ids (Optional[chex.Array]): Positional indices.
		    segment_ids (Optional[chex.Array]): Segment IDs for different input parts.
		    input_embeds (Optional[chex.Array]): Embedded input tensor.
		    output_attentions (Optional[bool]): If True, output attention weights.
		    output_hidden_states (Optional[bool]): If True, output hidden states.
		    init_cache (bool): If True, initialize cache for decoding.
		    return_dict (bool): If True, return a dictionary of outputs.

		Returns:
		    FlaxCausalLMOutput | Tuple: Model output, either as a named tuple or a standard tuple.
		"""

		batch_size, seq_length = (
			input_ids.shape if input_ids is not None else input_embeds.shape[:2]
		)
		if attention_mask is None:
			attention_mask = jnp.ones_like(input_ids)
		if position_ids is None:
			position_ids = jnp.broadcast_to(
				jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
				(batch_size, seq_length),
			).astype(jnp.int32)
		outputs = self.model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			init_cache=init_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
			input_embeds=input_embeds,
			segment_ids=segment_ids,
		)

		hidden_states = outputs[0]

		if self.config.tie_word_embeddings:
			shared_kernel = self.model.variables["params"]["embed_tokens"][
				"embedding"
			].T.astype(self.param_dtype)
			lm_logits = self.lm_head.apply(
				{"params": {"kernel": shared_kernel}},
				hidden_states,
			)
		else:
			lm_logits = self.lm_head(hidden_states)

		if not return_dict:
			return (lm_logits,) + outputs[1:]

		return FlaxCausalLMOutput(
			logits=lm_logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)


# @register_module(
# 	"sequence-classification",
# 	config=LlamaConfig,
# 	model_type="llama",
# 	embedding_layer_names=["embed_tokens"],
# )
# @wrap_easydel_module(LlamaConfig, base_model_prefix="model")
# class FlaxLlamaForSequenceClassification(nn.Module):
# 	num_labels: int
# 	config: LlamaConfig
# 	dtype: jnp.dtype = jnp.float32
# 	param_dtype: jnp.dtype = jnp.float32
# 	precision: Optional[Union[jax.lax.Precision, str]] = None

# 	def setup(self):
# 		"""The setup function is called once at the beginning of training.
# 		It initializes the model and optimizer, and sets up any other state that needs to be initialized.

# 		Args:
# 		    self: Access variables that belong to the class

# 		Returns:
# 		    A tuple of the model and the classifier
# 		"""
# 		self.model = FlaxLlamaModel.flax_module(
# 			config=self.config,
# 			dtype=self.dtype,
# 			param_dtype=self.param_dtype,
# 			precision=self.precision,
# 		)
# 		self.score = Dense(
# 			self.num_labels,
# 			dtype=self.dtype,
# 			param_dtype=self.param_dtype,
# 			use_bias=False,
# 			kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
# 			precision=self.precision,
# 		)

# 	def __call__(
# 		self,
# 		input_ids: Optional[chex.Array] = None,
# 		attention_mask: Optional[chex.Array] = None,
# 		position_ids: Optional[chex.Array] = None,
# 		segment_ids: Optional[chex.Array] = None,
# 		input_embeds: Optional[chex.Array] = None,
# 		output_attentions: Optional[bool] = None,
# 		output_hidden_states: Optional[bool] = None,
# 		init_cache: bool = False,
# 		deterministic: bool = True,
# 		return_dict: bool = True,
# 	) -> Union[FlaxSequenceClassifierOutput, Tuple]:
# 		"""
# 		Forward pass through the Llama module.

# 		Args:
# 		    input_ids (Optional[chex.Array]): Input tensor containing token IDs.
# 		    attention_mask (Optional[chex.Array]): Mask for attention.
# 		    position_ids (Optional[chex.Array]): Positional indices.
# 		    segment_ids (Optional[chex.Array]): Segment IDs for different input parts.
# 		    input_embeds (Optional[chex.Array]): Embedded input tensor.
# 		    output_attentions (Optional[bool]): If True, output attention weights.
# 		    output_hidden_states (Optional[bool]): If True, output hidden states.
# 		    init_cache (bool): If True, initialize cache for decoding.
# 		    deterministic (bool): If True, disable dropout.
# 		    return_dict (bool): If True, return a dictionary of outputs.

# 		Returns:
# 		    FlaxSequenceClassifierOutput | Tuple: Model output, either as a named tuple or a standard tuple.
# 		"""

# 		batch_size, seq_length = (
# 			input_ids.shape if input_ids is not None else input_embeds.shape[:2]
# 		)
# 		if attention_mask is None:
# 			attention_mask = jnp.ones_like(input_ids)
# 		if position_ids is None:
# 			position_ids = jnp.broadcast_to(
# 				jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
# 				(batch_size, seq_length),
# 			)
# 		outputs = self.model(
# 			input_ids=input_ids,
# 			attention_mask=attention_mask,
# 			position_ids=position_ids,
# 			deterministic=deterministic,
# 			init_cache=init_cache,
# 			output_attentions=output_attentions,
# 			output_hidden_states=output_hidden_states,
# 			return_dict=return_dict,
# 			input_embeds=input_embeds,
# 			segment_ids=segment_ids,
# 		)

# 		hidden_states = outputs[0]
# 		prediction = self.score(hidden_states)
# 		if return_dict:
# 			return FlaxSequenceClassifierOutput(
# 				logits=prediction,
# 				hidden_states=hidden_states,
# 			)
# 		else:
# 			return (prediction,)


# @register_module(
# 	"vision-module",
# 	config=VisionLlamaConfig,
# 	model_type="llama",
# 	embedding_layer_names=["embed_tokens"],
# )
# @wrap_custom_easydel_module(
# 	base=EasyDeLBaseVisionModule,
# 	config_class=VisionLlamaConfig,
# 	base_model_prefix="model",
# )
# class FlaxVisionLlamaModel(nn.Module):
# 	config: VisionLlamaConfig
# 	dtype: jnp.dtype = jnp.float32
# 	param_dtype: jnp.dtype = jnp.float32
# 	precision: Optional[Union[jax.lax.Precision, str]] = None

# 	def setup(self):
# 		config = self.config
# 		self.embed_dim = config.hidden_size

# 		self.embed_vision = nn.Embed(
# 			config.vision_vocab_size,
# 			config.hidden_size,
# 			embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
# 			dtype=self.dtype,
# 			param_dtype=self.param_dtype,
# 		)

# 		self.embed_tokens = nn.Embed(
# 			config.vocab_size,
# 			config.hidden_size,
# 			embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
# 			dtype=self.dtype,
# 			param_dtype=self.param_dtype,
# 		)
# 		self.dropout = nn.Dropout(rate=config.embd_pdrop)
# 		self.layers = FlaxLlamaBlockCollection(
# 			self.config,
# 			dtype=self.dtype,
# 			param_dtype=self.param_dtype,
# 			precision=self.precision,
# 		)
# 		self.norm = RMSNorm(
# 			self.config.hidden_size,
# 			eps=self.config.rms_norm_eps,
# 			dtype=self.dtype,
# 			param_dtype=self.param_dtype,
# 		)

# 		self.causal_mask = nn.make_causal_mask(
# 			jnp.ones(
# 				shape=(1, self.config.granted_mask_max_position_embedding),
# 				dtype="bool",
# 			),
# 			dtype="bool",
# 		)

# 	def __call__(
# 		self,
# 		input_ids: jax.Array,
# 		vision_mask: jax.Array,
# 		attention_mask: Optional[jax.Array] = None,
# 		position_ids: Optional[jax.Array] = None,
# 		deterministic=True,
# 		init_cache: bool = False,
# 		output_attentions: bool = False,
# 		output_hidden_states: bool = False,
# 		return_dict: bool = True,
# 	):
# 		input_ids = input_ids.astype("i4")

# 		if input_ids.shape[1] == 1:
# 			if self.config.sample_mode == "text":
# 				input_embeds = self.embed_tokens(input_ids)
# 			elif self.config.sample_mode == "vision":
# 				input_embeds = self.embed_vision(input_ids)
# 			elif self.config.sample_mode == "all":
# 				raise NotImplementedError
# 			else:
# 				raise ValueError(f"Invalid sample_mode: {self.config.sample_mode}")
# 		else:
# 			input_text_embeds = self.embed_tokens(jnp.where(vision_mask, 0, input_ids))
# 			input_vision_embeds = self.embed_vision(jnp.where(vision_mask, input_ids, 0))
# 			vision_mask = vision_mask[..., None].astype("f4")
# 			input_embeds = (
# 				input_text_embeds * (1 - vision_mask) + input_vision_embeds * vision_mask
# 			)

# 		hidden_states = self.dropout(input_embeds, deterministic=deterministic)

# 		outputs = self.layers(
# 			hidden_states=hidden_states,
# 			attention_mask=attention_mask,
# 			position_ids=position_ids,
# 			deterministic=deterministic,
# 			init_cache=init_cache,
# 			output_attentions=output_attentions,
# 			output_hidden_states=output_hidden_states,
# 			return_dict=return_dict,
# 			causal_mask=self.causal_mask,
# 		)

# 		hidden_states = outputs[0]
# 		hidden_states = self.norm(hidden_states)

# 		if output_hidden_states:
# 			all_hidden_states = outputs[1] + (hidden_states,)
# 			outputs = (hidden_states, all_hidden_states) + outputs[2:]
# 		else:
# 			outputs = (hidden_states,) + outputs[1:]

# 		if not return_dict:
# 			return tuple(v for v in outputs if v is not None)

# 		return FlaxBaseModelOutput(
# 			last_hidden_state=hidden_states,
# 			hidden_states=outputs[1],
# 			attentions=outputs[-1],
# 		)


# @register_module(
# 	"vision-language-model",
# 	config=VisionLlamaConfig,
# 	model_type="llama",
# 	embedding_layer_names=["embed_tokens"],
# )
# @wrap_custom_easydel_module(
# 	base=EasyDeLBaseVisionModule,
# 	config_class=VisionLlamaConfig,
# 	base_model_prefix="model",
# )
# class FlaxVisionLlamaForCausalLM(nn.Module):
# 	config: VisionLlamaConfig
# 	dtype: jnp.dtype = jnp.float32
# 	param_dtype: jnp.dtype = jnp.float32
# 	precision: Optional[Union[jax.lax.Precision, str]] = None

# 	def setup(self):
# 		self.model = FlaxVisionLlamaModel.flax_module(
# 			self.config,
# 			dtype=self.dtype,
# 			param_dtype=self.param_dtype,
# 			precision=self.precision,
# 		)
# 		self.vision_head = Dense(
# 			self.config.vision_vocab_size,
# 			dtype=self.dtype,
# 			param_dtype=self.param_dtype,
# 			use_bias=False,
# 			kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
# 			precision=self.precision,
# 		)
# 		self.lm_head = Dense(
# 			self.config.vocab_size,
# 			dtype=self.dtype,
# 			param_dtype=self.param_dtype,
# 			use_bias=False,
# 			kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
# 			precision=self.precision,
# 		)

# 	def __call__(
# 		self,
# 		input_ids: jax.Array,
# 		vision_mask: jax.Array,
# 		attention_mask: Optional[jax.Array] = None,
# 		position_ids: Optional[jax.Array] = None,
# 		deterministic=True,
# 		init_cache: bool = False,
# 		output_attentions: bool = False,
# 		output_hidden_states: bool = False,
# 		return_dict: bool = True,
# 	):
# 		batch_size, seq_length = input_ids.shape
# 		if attention_mask is None:
# 			attention_mask = jnp.ones_like(input_ids)
# 		if position_ids is None:
# 			position_ids = jnp.broadcast_to(
# 				jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
# 				(batch_size, seq_length),
# 			)

# 		outputs = self.transformer(
# 			input_ids,
# 			vision_mask,
# 			attention_mask,
# 			position_ids,
# 			deterministic=deterministic,
# 			init_cache=init_cache,
# 			output_attentions=output_attentions,
# 			output_hidden_states=output_hidden_states,
# 			return_dict=return_dict,
# 		)

# 		hidden_states = outputs[0]

# 		if self.config.tie_vision_embeddings:
# 			shared_kernel = self.transformer.variables["params"]["embed_vision"][
# 				"embedding"
# 			].T
# 			vision_logits = self.vision_head.apply(
# 				{"params": {"kernel": shared_kernel}}, hidden_states
# 			)
# 		else:
# 			vision_logits = self.vision_head(hidden_states)

# 		if self.config.tie_word_embeddings:
# 			shared_kernel = self.transformer.variables["params"]["embed_tokens"][
# 				"embedding"
# 			].T.astype(self.param_dtype)
# 			lm_logits = self.lm_head.apply(
# 				{"params": {"kernel": shared_kernel}},
# 				hidden_states,
# 			)
# 		else:
# 			lm_logits = self.lm_head(hidden_states)

# 		if self.config.sample_mode == "all":
# 			if not return_dict:
# 				return (
# 					vision_logits,
# 					lm_logits,
# 				) + outputs[1:]

# 			return FlaxCausalLMOutput(
# 				logits=(vision_logits, lm_logits),
# 				hidden_states=outputs.hidden_states,
# 				attentions=outputs.attentions,
# 			)
# 		elif self.config.sample_mode == "vision":
# 			if not return_dict:
# 				return (vision_logits,) + outputs[1:]

# 			return FlaxCausalLMOutput(
# 				logits=vision_logits,
# 				hidden_states=outputs.hidden_states,
# 				attentions=outputs.attentions,
# 			)
# 		elif self.config.sample_mode == "text":
# 			if not return_dict:
# 				return (lm_logits,) + outputs[1:]

# 			return FlaxCausalLMOutput(
# 				logits=lm_logits,
# 				hidden_states=outputs.hidden_states,
# 				attentions=outputs.attentions,
# 			)
# 		else:
# 			raise ValueError(f"Invalid sample_mode: {self.config.sample_mode}")
