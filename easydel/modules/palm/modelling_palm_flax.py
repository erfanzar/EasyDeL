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

from typing import Optional, Union

import chex
import jax
import jax.numpy as jnp
import numpy as onp
import transformers.modeling_flax_outputs
from einops import rearrange
from flax import linen as nn
from jax import numpy as np

from easydel.etils.etils import EasyDeLGradientCheckPointers
from easydel.layers.norms import RMSNorm
from easydel.modules.factory import register_module
from easydel.modules.flax_modeling_utils import get_gradient_checkpoint_policy
from easydel.modules.modeling_flax_outputs import FlaxCausalLMOutput
from easydel.modules.modeling_utils import wrap_easydel_module
from easydel.modules.palm.palm_configuration import PalmConfig as PalmConfig


def pre_compute_freq_cis(dim, max_length, theta: int = 10000.0, dtype=jnp.bfloat16):
	frequencies = 1 / (theta ** (jnp.arange(0, dim, 2).astype(dtype=dtype) / dim))
	length = jnp.arange(max_length)
	cis = jnp.outer(length, frequencies).astype(dtype)
	sin = jnp.sin(cis)
	cos = jnp.cos(cis)
	frequencies = jnp.complex64(cos + 1j * sin)
	return jnp.asarray(frequencies)


def apply_rotary_embedding(xq, xk, frequencies, dtype=jnp.bfloat16):
	reshape_xq = xq.astype(jnp.flaot32).reshape(xq.shape[:-1], -1, 2)
	reshape_xk = xk.astype(jnp.flaot32).reshape(xk.shape[:-1], -1, 2)

	complex_q = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
	complex_k = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])

	frequencies = frequencies.reshape(*frequencies[:2], 1, *frequencies[2:])
	xq = complex_q * frequencies
	xk = complex_k * frequencies
	xq = jnp.stack([jnp.real(xq), jnp.imag(xq)], axis=-1).reshape(xq.shape[:-1], -1)
	xk = jnp.stack([jnp.real(xk), jnp.imag(xk)], axis=-1).reshape(xk.shape[:-1], -1)
	return xq.astype(dtype), xk.astype(dtype)


class ParallelPalmBlock(nn.Module):
	config: PalmConfig
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		attn_inner_dim = self.config.dim_head * self.config.num_attention_heads
		ff_inner_dim = self.config.hidden_size * self.config.up_inner_dim
		self.fused_dims = (
			attn_inner_dim,
			self.config.dim_head,
			self.config.dim_head,
			ff_inner_dim,
			ff_inner_dim,
		)

		# INPUT WEIGHTS
		self.wi = self.param(
			"kernel",
			nn.initializers.normal,
			(self.config.hidden_size, sum(self.fused_dims)),
			self.param_dtype,
		).astype(self.param_dtype)

		# ATTENTION WEIGHT OUTPUT
		self.attn_wo = self.param(
			"kernel",
			nn.initializers.normal,
			(attn_inner_dim, self.config.hidden_size),
			self.param_dtype,
		).astype(
			self.param_dtype,
		)

		self.ff_wo = self.param(
			"kernel",
			nn.initializers.normal,
			(attn_inner_dim, self.config.hidden_size),
			self.param_dtype,
		).astype(
			self.param_dtype,
		)

		self.norm = RMSNorm(
			dim=self.config.hidden_size, dtype=self.dtype, param_dtype=self.param_dtype
		)
		self.post_norm = RMSNorm(
			dim=self.config.hidden_size, dtype=self.dtype, param_dtype=self.param_dtype
		)

		self.num_attention_heads: int = self.config.num_attention_heads
		self.scale: float = self.config.dim_head**-0.5

	def __call__(self, hidden_state, frequencies, causal_mask):
		split_indices = onp.cumsum(self.fused_dims[:-1])

		hidden_state = self.norm(hidden_state)

		q, k, v, ff, ff_gate = np.split(hidden_state @ self.wi, split_indices, axis=-1)
		q = rearrange(q, "b s (h d)-> b s h d", h=self.num_attention_heads)
		k = rearrange(k, "b s (h d)-> b s h d", h=self.num_attention_heads)

		q, k = apply_rotary_embedding(q, k, frequencies, self.dtype)
		q = rearrange(q, "b s h d -> b s (h d)")
		k = rearrange(k, "b s h d -> b s (h d)")
		q = (
			rearrange(q, "... n (h d) -> ... h n d", h=self.num_attention_heads) * self.scale
		)

		sim = jnp.einsum("... h i d, ... j d -> ... h i j", q, k)
		# if self.config.use_sharding_constraint:
		#     sim = with_sharding_constraint(sim, PartitionSpec(("dp", "fsdp"), "sp", None, None))
		mask_value = jnp.finfo(hidden_state).min
		attn = nn.softmax(np.where(causal_mask, sim, mask_value), axis=-1)

		out = jnp.einsum("... h i j, ... j d -> ... h i d", attn, v)
		# if self.config.use_sharding_constraint:
		#     out = with_sharding_constraint(out, PartitionSpec(("dp", "fsdp"), "sp", None, None))
		attn_out = rearrange(out, "... h n hd -> ... n (h hd)") @ self.attn_wo

		ff_out = (ff * nn.swish(ff_gate)) @ self.ff_wo

		return attn_out + ff_out


class ParallelCollection(nn.Module):
	config: PalmConfig
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		block = ParallelPalmBlock
		if self.config.gradient_checkpointing != EasyDeLGradientCheckPointers.NONE:
			block = nn.remat(
				block,
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

	def __call__(self, hidden_state, frequencies, causal_mask, output_attention=False):
		saves = []
		for block in self.blocks:
			hidden_state = (
				block(
					hidden_state=hidden_state,
					frequencies=frequencies,
					causal_mask=causal_mask,
				)
				+ hidden_state
			)
			if output_attention:
				saves.append(hidden_state)
		return hidden_state, saves


@register_module(
	"base-module",
	config=PalmConfig,
	model_type="palm",
	embedding_layer_names=["wte"],
)
@wrap_easydel_module(config_class=PalmConfig, base_model_prefix="path_way")
class FlaxPalmModel(nn.Module):
	config: PalmConfig
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		self.wte = nn.Embed(
			self.config.vocab_size,
			self.config.hidden_size,
			dtype=self.dtype,
			param_dtype=self.dtype,
			embedding_init=jax.nn.initializers.normal,
		)
		self.block = ParallelCollection(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.frequencies = pre_compute_freq_cis(
			self.config.dim_head, self.config.max_length, dtype=self.dtype
		)

		self.ln_f = RMSNorm(
			dim=self.config.hidden_size,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			eps=self.config.eps,
		)
		self.causal_mask = nn.make_causal_mask(
			jnp.ones((1, self.config.max_length), dtype="bool"), dtype="bool"
		)

	def make_causal_mask(self, attention_mask=None):
		assert attention_mask is not None
		b, s = attention_mask.shape
		mask = attention_mask + self.causal_mask
		mask = jnp.where(mask == 2, 1, 0).astype(jnp.bool_)
		return mask.reshape(b, 1, 1, s)

	def __call__(
		self,
		input_ids: chex.Array,
		attention_mask: chex.Array = None,
		return_dict: bool = True,
		output_attention: bool = False,
	):
		batch, seq_len = input_ids.shape
		if attention_mask is None:
			attention_mask = jnp.ones((batch, seq_len), dtype=jnp.int32)

		mask = self.make_causal_mask(attention_mask=attention_mask)
		hidden_state = self.wte(inputs=input_ids)
		hidden_state, atn = self.block(
			hidden_state=hidden_state,
			causal_mask=mask,
			output_attention=output_attention,
			frequencies=self.frequencies[:seq_len].reshape(1, seq_len, -1),
		)
		hidden_state = self.ln_f(hidden_state)

		if return_dict:
			return transformers.modeling_flax_outputs.FlaxBaseModelOutput(
				last_hidden_state=hidden_state, hidden_states=atn
			)
		else:
			return hidden_state, atn


@register_module(
	"causal-language-model",
	config=PalmConfig,
	model_type="palm",
	embedding_layer_names=["wte"],
)
@wrap_easydel_module(config_class=PalmConfig, base_model_prefix="path_way")
class FlaxPalmForCausalLM(nn.Module):
	config: PalmConfig
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self) -> None:
		self.path_way = FlaxPalmModel.flax_module(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		if not self.config.use_tie_word_embedding:
			self.lm_head = self.param(
				"kernel",
				jax.nn.initializers.normal,
				(self.config.hidden_size, self.config.vocab_size),
				self.param_dtype,
			).astype(
				self.param_dtype,
			)

	def __call__(
		self,
		input_ids: chex.Array,
		attention_mask: chex.Array = None,
		return_dict: bool = True,
		output_attention: bool = False,
	):
		out = self.path_way(
			input_ids=input_ids,
			attention_mask=attention_mask,
			return_dict=True,
			output_attention=output_attention,
		)
		last_state = out.last_hidden_state
		if not self.config.use_tie_word_embedding:
			last_state = last_state @ self.lm_head
		else:
			last_state = last_state @ self.path_way.wte.embedding.T

		if return_dict:
			return FlaxCausalLMOutput(logits=last_state, hidden_states=out.hidden_states)
		else:
			return (
				last_state,
				out.hidden_states if output_attention else last_state,
			)
