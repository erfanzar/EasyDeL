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


import typing as tp

import chex
import jax
import jax.numpy as jnp
from flax import nnx as nn

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType
from easydel.infra.modeling_outputs import (
	ModelOutput,
)
from easydel.infra.utils import (
	ACT2FN,
)
from easydel.layers.caching import TransformerCache
from easydel.modules.auto.auto_modeling import (
	AutoEasyDeLModelForCausalLM,
	AutoEasyDeLVisionModel,
)
from easydel.utils import traversals as etr
from easydel.utils.helpers import get_logger
from easydel.infra.factory import register_module
from .llava_configuration import LlavaConfig

logger = get_logger(__name__)


@etr.auto_pytree
class LlavaCausalLMOutputWithPast(ModelOutput):
	"""
	Base class for Llava causal language model (or autoregressive) outputs.

	Args:
	    loss (`chex.Array` of shape `(1,)`, *optional*, returned when `labels` is provided):
	        Language modeling loss (for next-token prediction).
	    logits (`chex.Array` of shape `(batch_size, sequence_length, config.vocab_size)`):
	        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
	    past_key_values (`tuple(tuple(chex.Array))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
	        Tuple of `tuple(chex.Array)` of length `config.n_layers`, with each tuple having 2 tensors of shape
	        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

	        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
	        `past_key_values` input) to speed up sequential decoding.
	    hidden_states (`tuple(chex.Array)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
	        Tuple of `chex.Array` (one for the output of the embeddings, if the model has an embedding layer, +
	        one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

	        Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
	    attentions (`tuple(chex.Array)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
	        Tuple of `chex.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
	        sequence_length)`.

	        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
	        heads.
	    image_hidden_states (`chex.Array`, *optional*):
	        A `chex.Array` of size (batch_size * num_patches, num_images, sequence_length, hidden_size)`.
	        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
	"""

	loss: tp.Optional[chex.Array] = None
	logits: chex.Array = None
	past_key_values: tp.Optional[TransformerCache] = None
	hidden_states: tp.Optional[tp.Tuple[chex.Array]] = None
	attentions: tp.Optional[tp.Tuple[chex.Array]] = None
	image_hidden_states: tp.Optional[chex.Array] = None


class LlavaMultiModalProjector(nn.Module):
	def __init__(
		self,
		config: LlavaConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.rngs = rngs

		num_feature_layers = (
			1
			if isinstance(config.vision_feature_layer, int)
			else len(config.vision_feature_layer)
		)

		self.linear_1 = nn.Linear(
			config.vision_config.hidden_size * num_feature_layers,
			config.text_config.hidden_size,
			use_bias=config.multimodal_projector_bias,
			kernel_init=nn.initializers.normal(0.02),
			param_dtype=param_dtype,
			dtype=dtype,
			precision=precision,
			rngs=rngs,
		)

		self.act = ACT2FN[config.projector_hidden_act]
		self.linear_2 = nn.Linear(
			config.text_config.hidden_size,
			config.text_config.hidden_size,
			use_bias=config.multimodal_projector_bias,
			kernel_init=nn.initializers.normal(0.02),
			param_dtype=param_dtype,
			dtype=dtype,
			precision=precision,
			rngs=rngs,
		)

	def __call__(self, image_features: jax.Array) -> jax.Array:
		hidden_states = self.linear_1(image_features)
		hidden_states = self.act(hidden_states)
		hidden_states = self.linear_2(hidden_states)
		return hidden_states


@register_module(
	TaskType.IMAGE_TEXT_TO_TEXT,
	config=LlavaConfig,
	model_type="llava",
)
class LlavaForConditionalGeneration(EasyDeLBaseModule):
	loss_type = "ForCausalLM"

	def __init__(
		self,
		config: LlavaConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.PrecisionLike = None,
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
		self.vision_tower = AutoEasyDeLVisionModel.from_config(
			config=config.vision_config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.multi_modal_projector = LlavaMultiModalProjector(
			config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.vocab_size = config.text_config.vocab_size
		self.language_model = AutoEasyDeLModelForCausalLM.from_config(
			config=config.text_config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		self.pad_token_id = (
			self.config.pad_token_id if self.config.pad_token_id is not None else -1
		)
		self.vision_feature_layer = config.vision_feature_layer
		self.vision_feature_select_strategy = getattr(
			config,
			"vision_feature_select_strategy",
			"default",
		)

	def get_image_features(self, pixel_values: chex.Array) -> chex.Array:
		image_features = self.vision_tower(pixel_values, output_hidden_states=True)
		selected_image_feature = image_features.hidden_states[self.vision_feature_layer]
		if self.vision_feature_select_strategy == "default":
			selected_image_feature = selected_image_feature[:, 1:]
		elif self.vision_feature_select_strategy == "full":
			selected_image_feature = selected_image_feature
		image_features = self.multi_modal_projector(selected_image_feature)

		return image_features

	def __call__(
		self,
		input_ids: chex.Array = None,
		pixel_values: chex.Array = None,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		return_dict: tp.Optional[bool] = None,
		**lm_kwargs,
	):
		if (input_ids is None) ^ (inputs_embeds is not None):
			raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

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
		return_dict = (
			return_dict if return_dict is not None else self.config.use_return_dict
		)
		if input_ids is not None and self.config.image_token_index >= self.vocab_size:
			special_image_mask = input_ids == self.config.image_token_index
			llm_input_ids = input_ids
			llm_input_ids = jnp.where(special_image_mask, 0, llm_input_ids)
		else:
			llm_input_ids = input_ids

		if inputs_embeds is None:
			inputs_embeds = self.language_model.model.embed_tokens(llm_input_ids)

		if pixel_values is not None:
			image_features = self.get_image_features(pixel_values)

			special_image_mask = jnp.expand_dims(
				(input_ids == self.config.image_token_index), -1
			)
			special_image_mask = jnp.broadcast_to(special_image_mask, inputs_embeds.shape)
			image_features = image_features.astype(inputs_embeds.dtype)
			inputs_embeds = jnp.place(
				inputs_embeds,
				special_image_mask,
				image_features,
				inplace=False,
			)
		outputs = self.language_model(
			attention_mask=attention_mask,
			position_ids=position_ids,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			past_key_values=past_key_values,
			return_dict=return_dict,
			inputs_embeds=inputs_embeds,
			segment_ids=segment_ids,
			**lm_kwargs,
		)

		return LlavaCausalLMOutputWithPast(
			loss=None,
			logits=outputs.logits,
			past_key_values=outputs.past_key_values,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
			image_hidden_states=image_features if pixel_values is not None else None,
		)

	def _get_compile_model_kwargs(
		self,
		batch_size: int,
		input_tokens_length: int,
		input_sharding: jax.sharding.PartitionSpec,
		rngs: jax.random.PRNGKey,
		vision_included: bool = False,
		vision_batch_size: int = 1,
		vision_channels: int = 3,
		vision_height: tp.Optional[int] = None,
		vision_width: tp.Optional[int] = None,
		required_props: tp.Optional[tp.Mapping[str, tp.Dict[str, tp.Any]]] = None,
		**kwargs,
	):
		basics = self.language_model._get_compile_model_kwargs(
			batch_size=batch_size,
			input_tokens_length=input_tokens_length,
			input_sharding=input_sharding,
			rngs=rngs,
			vision_included=vision_included,
			vision_batch_size=vision_batch_size,
			vision_channels=vision_channels,
			vision_height=vision_height,
			vision_width=vision_width,
			required_props=required_props,
			**kwargs,
		)
		if vision_included:
			pixel_values = jnp.ones(
				(
					vision_batch_size or 1,
					vision_channels or 3,
					self.config.vision_config.image_size,
					self.config.vision_config.image_size,
				),
				dtype="f4",
			)
			basics.update({"pixel_values": pixel_values})
		return basics

	def prepare_inputs_for_generation(
		self,
		input_ids: chex.Array,
		max_length: int,
		pixel_values: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
	):
		model_inputs = self.language_model.prepare_inputs_for_generation(
			input_ids=input_ids,
			max_length=max_length,
			attention_mask=attention_mask,
		)
		model_inputs["pixel_values"] = pixel_values
		return model_inputs

	def update_inputs_for_generation(self, model_outputs, model_kwargs):
		model_kwargs = self.language_model.update_inputs_for_generation(
			model_outputs, model_kwargs
		)
		model_kwargs.pop("pixel_values", None)  # only effect first iter
		return model_kwargs
