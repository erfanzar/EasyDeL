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
from functools import partial
from typing import Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from flax import nnx as nn

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import register_module
from easydel.infra.modeling_outputs import (
	FlaxBaseModelOutput,
	FlaxCausalLMOutput,
	FlaxSequenceClassifierOutput,
	ModelOutput,
)
from easydel.infra.utils import (
	ACT2FN,
	auto_remat,
	block_wise_ffn,
	control_mlp_sharding,
	get_dot_general_by_bits,
)
from easydel.layers.attention import FlaxAttentionModule, FlexibleAttentionModule
from easydel.layers.caching import TransformerCache, TransformerCacheView
from easydel.layers.norms import RMSNorm
from easydel.modules.clip.clip_configuration import (
	CLIPConfig,
	CLIPTextConfig,
	CLIPVisionConfig,
)
from flax import struct
from flax import nnx as nn


@struct.dataclass
class FlaxCLIPTextModelOutput(ModelOutput):
	"""
	Base class for text model's outputs that also contains a pooling of the last hidden states.

	Args:
	    text_embeds (`jnp.ndarray` of shape `(batch_size, output_dim`):
	        The text embeddings obtained by applying the projection layer to the pooled output of
	        [`FlaxCLIPTextModel`].
	    last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
	        Sequence of hidden-states at the output of the last layer of the model.
	    hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
	        Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
	        `(batch_size, sequence_length, hidden_size)`.

	        Hidden-states of the model at the output of each layer plus the initial embedding outputs.
	    attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
	        Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
	        sequence_length)`.

	        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
	        heads.
	"""

	text_embeds: jnp.ndarray = None
	last_hidden_state: jnp.ndarray = None
	hidden_states: Optional[Tuple[jnp.ndarray, ...]] = None
	attentions: Optional[Tuple[jnp.ndarray, ...]] = None


@struct.dataclass
class FlaxCLIPOutput(ModelOutput):
	"""
	Args:
	    logits_per_image:(`jnp.ndarray` of shape `(image_batch_size, text_batch_size)`):
	        The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
	        similarity scores.
	    logits_per_text:(`jnp.ndarray` of shape `(text_batch_size, image_batch_size)`):
	        The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
	        similarity scores.
	    text_embeds(`jnp.ndarray` of shape `(batch_size, output_dim`):
	        The text embeddings obtained by applying the projection layer to the pooled output of
	        [`FlaxCLIPTextModel`].
	    image_embeds(`jnp.ndarray` of shape `(batch_size, output_dim`):
	        The image embeddings obtained by applying the projection layer to the pooled output of
	        [`FlaxCLIPVisionModel`].
	    text_model_output(`FlaxBaseModelOutputWithPooling`):
	        The output of the [`FlaxCLIPTextModel`].
	    vision_model_output(`FlaxBaseModelOutputWithPooling`):
	        The output of the [`FlaxCLIPVisionModel`].
	"""

	logits_per_image: jnp.ndarray = None
	logits_per_text: jnp.ndarray = None
	text_embeds: jnp.ndarray = None
	image_embeds: jnp.ndarray = None
	text_model_output: FlaxBaseModelOutputWithPooling = None
	vision_model_output: FlaxBaseModelOutputWithPooling = None

	def to_tuple(self) -> Tuple[Any]:
		return tuple(
			self[k]
			if k not in ["text_model_output", "vision_model_output"]
			else getattr(self, k).to_tuple()
			for k in self.keys()
		)


class CLIPVisionEmbeddings(nn.Module):
	def __init__(
		self,
		config: CLIPVisionConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: Optional[Union[jax.lax.Precision, str]] = None,
		*,
		rngs: nn.Rngs,
	):
		embed_dim = config.hidden_size
		image_size = config.image_size
		patch_size = config.patch_size

		self.class_embedding = nn.Param(
			jax.nn.initializers.normal(stddev=0.02)(
				rngs.params(),
				shape=(embed_dim,),
				dtype=param_dtype,
			),
		)

		self.patch_embedding = nn.Conv(
			config.num_channels,
			embed_dim,
			kernel_size=(patch_size, patch_size),
			strides=(patch_size, patch_size),
			padding="VALID",
			use_bias=False,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			kernel_init=jax.nn.initializers.normal(),
			rngs=rngs,
		)

		self.num_patches = (image_size // patch_size) ** 2
		num_positions = self.num_patches + 1
		self.position_embedding = nn.Embed(
			num_positions,
			embed_dim,
			embedding_init=jax.nn.initializers.normal(),
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

	def __call__(self, pixel_values):
		patch_embeds = self.patch_embedding(pixel_values)
		batch_size, height, width, channels = patch_embeds.shape
		patch_embeds = jnp.reshape(patch_embeds, (batch_size, height * width, channels))

		class_embeds = jnp.expand_dims(self.class_embedding.value, axis=(0, 1))
		class_embeds = jnp.tile(class_embeds, (batch_size, 1, 1))
		embeddings = jnp.concatenate([class_embeds, patch_embeds], axis=1)

		embeddings = embeddings + self.position_embedding(
			jnp.expand_dims(
				jnp.arange(
					0, ((self.config.image_size // self.config.patch_size) ** 2) + 1, dtype="i4"
				),
				axis=0,
			)
		)
		return embeddings


class CLIPTextEmbeddings(nn.Module):
	def __init__(
		self,
		config: CLIPVisionConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: Optional[Union[jax.lax.Precision, str]] = None,
		*,
		rngs: nn.Rngs,
	):
		embed_dim = config.hidden_size

		self.token_embedding = nn.Embed(
			config.vocab_size,
			embed_dim,
			embedding_init=jax.nn.initializers.normal(),
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)
		self.position_embedding = nn.Embed(
			config.max_position_embeddings,
			embed_dim,
			embedding_init=jax.nn.initializers.normal(),
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

	def __call__(self, input_ids, position_ids):
		input_embeds = self.token_embedding(input_ids.astype("i4"))
		position_embeds = self.position_embedding(position_ids.astype("i4"))

		embeddings = input_embeds + position_embeds
		return embeddings


class CLIPAttention(FlaxAttentionModule):
	def __init__(
		self,
		config: Union[CLIPTextConfig, CLIPVisionConfig],
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: Optional[Union[jax.lax.Precision, str]] = None,
		*,
		rngs: nn.Rngs,
	):
		self.embed_dim = config.hidden_size
		self.num_heads = config.num_attention_heads
		self.head_dim = self.embed_dim // self.num_heads
		if self.head_dim * self.num_heads != self.embed_dim:
			raise ValueError(
				f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
				f" {self.num_heads})."
			)
		self.scale = self.head_dim**-0.5
		self.dropout = config.attention_dropout
		linear_class = partial(
			nn.Linear,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
			kernel_init=jax.nn.initializers.normal(0.01),
		)
		self.k_proj = linear_class(self.embed_dim, self.embed_dim)
		self.v_proj = linear_class(self.embed_dim, self.embed_dim)
		self.q_proj = linear_class(self.embed_dim, self.embed_dim)
		self.out_proj = linear_class(self.embed_dim, self.embed_dim)

		self.causal = isinstance(config, CLIPTextConfig)
