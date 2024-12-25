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
import math
import typing as tp

import chex
import jax
from flax import nnx as nn
from jax import lax
from jax import numpy as jnp

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import register_module
from easydel.infra.modeling_outputs import (
	FlaxBaseModelOutput,
	FlaxCausalLMOutput,
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
from easydel.modules.deepseek_v2.deepseek_configuration import (
	DeepseekV2Config as DeepseekV2Config,
)


def yarn_find_correction_dim(
	num_rotations, dim, base=10000, max_position_embeddings=2048
):
	return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
		2 * math.log(base)
	)


def yarn_find_correction_range(
	low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
	low = math.floor(
		yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
	)
	high = math.ceil(
		yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
	)
	return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_get_mscale(scale=1.0, mscale=1.0):
	if scale <= 1:
		return 1.0
	return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min, max, dim):
	if min == max:
		max += 0.001  # Prevent singularity

	linear_func = (jnp.arange(dim, dtype=jnp.float32) - min) / (max - min)
	return jnp.clip(linear_func, 0, 1)


def init_deepseek_rotary_embedding(
	dim,
	max_position_embeddings=2048,
	base=10000,
	method: tp.Literal["linear", "yarn", "dynamic", None] = None,
	kwargs: tp.Optional[dict] = None,
):
	if method is None:
		inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2).astype("float32") / dim))
		t = jnp.arange(max_position_embeddings, dtype=inv_freq.dtype)
		freqs = jnp.outer(t, inv_freq)
		emb = jnp.concatenate((freqs, freqs), axis=-1)
		return jnp.sin(emb), jnp.cos(emb)
	elif method == "linear":
		assert kwargs is not None
		inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2).astype("float32") / dim))
		t = jnp.arange(max_position_embeddings, dtype=inv_freq.dtype) / kwargs.get(
			"scaling_factor"
		)
		freqs = jnp.outer(t, inv_freq)
		emb = jnp.concatenate((freqs, freqs), axis=-1)
		return jnp.sin(emb), jnp.cos(emb)
	elif method == "dynamic":
		assert kwargs is not None
		targeted_len = kwargs.get("targeted_len", max_position_embeddings)
		if targeted_len > max_position_embeddings:
			base = base * (
				(kwargs.get("scaling_factor") * targeted_len / max_position_embeddings)
				- (kwargs.get("scaling_factor") - 1)
			) ** (dim / (dim - 2))
			inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2).astype("float32") / dim))

		else:
			inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2).astype("float32") / dim))
		t = jnp.arange(max_position_embeddings, dtype=inv_freq.dtype) / kwargs.get(
			"scaling_factor"
		)

		freqs = jnp.outer(t, inv_freq)
		emb = jnp.concatenate((freqs, freqs), axis=-1)
		return jnp.sin(emb), jnp.cos(emb)
	elif method == "yarn":
		scaling_factor = kwargs.get("scaling_factor", 1.0)
		original_max_position_embeddings = kwargs.get(
			"original_max_position_embeddings", 4096
		)
		beta_fast = kwargs.get("beta_fast", 32)
		beta_slow = kwargs.get("beta_slow", 1)
		mscale = kwargs.get("mscale", 1)
		mscale_all_dim = kwargs.get("mscale_all_dim", 0)
		freq_extra = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
		freq_inter = 1.0 / (
			scaling_factor * base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
		)

		low, high = yarn_find_correction_range(
			beta_fast,
			beta_slow,
			dim,
			base,
			original_max_position_embeddings,
		)
		inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).astype("float32")
		inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
		t = jnp.arange(max_position_embeddings, dtype=jnp.float32)

		freqs = jnp.outer(t, inv_freq)

		_mscale = float(
			yarn_get_mscale(scaling_factor, mscale)
			/ yarn_get_mscale(scaling_factor, mscale_all_dim)
		)

		emb = jnp.concatenate((freqs, freqs), axis=-1)
		return (jnp.sin(emb) * _mscale).astype("float32"), (jnp.cos(emb) * _mscale).astype(
			"float32"
		)


def rotate_half(x):
	x1 = x[..., : x.shape[-1] // 2]
	x2 = x[..., x.shape[-1] // 2 :]
	return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
	cos = jnp.expand_dims(cos[position_ids], unsqueeze_dim)
	sin = jnp.expand_dims(sin[position_ids], unsqueeze_dim)
	b, h, s, d = q.shape
	q = q.reshape(b, h, s, d // 2, 2).transpose(0, 1, 2, 4, 3).reshape(b, h, s, d)
	b, h, s, d = k.shape
	k = k.reshape(b, h, s, d // 2, 2).transpose(0, 1, 2, 4, 3).reshape(b, h, s, d)
	q_embed = (q * cos) + (rotate_half(q) * sin)
	k_embed = (k * cos) + (rotate_half(k) * sin)
	return q_embed, k_embed


class FlaxDeepseekV2MLP(nn.Module):
	def __init__(
		self,
		config: DeepseekV2Config,
		dtype: jnp.dtype = jnp.bfloat16,
		param_dtype: jnp.dtype = jnp.bfloat16,
		precision: tp.Optional[tp.Union[str, jax.lax.Precision]] = None,
		hidden_size: tp.Optional[int] = None,
		intermediate_size: tp.Optional[int] = None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		linear = functools.partial(
			nn.Linear,
			use_bias=False,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			kernel_init=nn.initializers.normal(),
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)
		imz = intermediate_size or config.intermediate_size
		hs = hidden_size or config.hidden_size
		self.gate_proj = linear(hs, imz, rngs=rngs)
		self.up_proj = linear(hs, imz, rngs=rngs)
		self.down_proj = linear(imz, hs, rngs=rngs)
		self.act_fn = ACT2FN[config.hidden_act]

	def __call__(self, hidden_states: chex.Array):
		hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
		return self.down_proj(
			self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
		)


class FlaxMoEGate(nn.Module):
	def __init__(
		self,
		config: DeepseekV2Config,
		dtype: jnp.dtype = jnp.bfloat16,
		param_dtype: jnp.dtype = jnp.bfloat16,
		precision: tp.Optional[tp.Union[str, jax.lax.Precision]] = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__()
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.rngs = rngs
		self.top_k = config.num_experts_per_tok
		self.n_routed_experts = config.n_routed_experts
		self.routed_scaling_factor = config.routed_scaling_factor
		self.scoring_func = config.scoring_func
		self.alpha = config.aux_loss_alpha
		self.seq_aux = config.seq_aux
		self.topk_method = config.topk_method
		self.n_group = config.n_group
		self.topk_group = config.topk_group

		self.norm_topk_prob = config.norm_topk_prob
		self.gating_dim = config.hidden_size
		self.kernel = nn.Param(
			nn.initializers.kaiming_uniform(dtype=self.param_dtype)(
				rngs.params(), (self.n_routed_experts, self.gating_dim)
			),
		)
		self.dp = nn.Dropout(0, rngs=rngs)

	def __call__(self, hidden_states):
		bsz, seq_len, h = hidden_states.shape
		hidden_states = hidden_states.reshape(-1, h)
		logits = jax.lax.batch_matmul(
			hidden_states.astype(jnp.float32),
			self.kernel.value.astype(jnp.float32),
			precision=self.precision,
		)
		if self.scoring_func == "softmax":
			scores = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
		else:
			raise NotImplementedError(
				f"insupportable scoring function for MoE gating: {self.scoring_func}"
			)

		### select top-k experts
		if self.topk_method == "gready":
			topk_weight, topk_idx = jax.lax.top_k(scores, k=self.top_k)
		elif self.topk_method == "group_limited_greedy":
			group_scores = scores.reshape(bsz * seq_len, self.n_group, -1).max(
				axis=-1
			)  # [n, n_group]

			# Find the indices of the top k scores in each group
			top_k_indices = lax.top_k(group_scores, self.topk_group)[1]  # [n, topk_group]

			# Initialize a mask with zeros
			group_mask = jnp.zeros_like(group_scores)  # [n, n_group]

			# Update the mask: this is a bit tricky in JAX as there is no direct scatter function
			n_indices = jnp.arange(group_mask.shape[0])[:, None]
			group_mask = group_mask.at[n_indices, top_k_indices].set(1)  # [n, n_group]

			# Expand and reshape the group_mask
			score_mask = jnp.repeat(
				group_mask[:, :, None], self.n_routed_experts // self.n_group, axis=2
			)
			score_mask = score_mask.reshape(bsz * seq_len, -1)  # [n, e]

			# Apply the mask to scores
			masked_scores = jnp.where(score_mask, scores, 0.0)  # [n, e]

			# Compute the top k scores after masking
			topk_weight, topk_idx = lax.top_k(masked_scores, self.top_k)
		else:
			raise ValueError()
		### norm gate to sum 1
		if self.top_k > 1 and self.norm_topk_prob:
			denominator = jnp.sum(topk_weight, axis=-1, keepdims=True) + 1e-20
			topk_weight = topk_weight / denominator
		else:
			topk_weight = topk_weight * self.routed_scaling_factor
		### expert-level computation auxiliary loss

		if not self.dp.deterministic and self.alpha > 0.0:
			scores_for_aux = scores
			aux_topk = self.top_k
			topk_idx_for_aux_loss = topk_idx.reshape(bsz, -1)
			if self.seq_aux:
				scores_for_seq_aux = scores_for_aux.reshape(bsz, seq_len, -1)
				ce = jnp.zeros(bsz, self.n_routed_experts)
				ce = ce.at[1, topk_idx_for_aux_loss].add(
					jnp.ones(bsz, seq_len * aux_topk),
				)
				ce = jnp.divide(ce, (seq_len * aux_topk / self.n_routed_experts))
				aux_loss = (
					jnp.mean(jnp.sum((ce * jnp.mean(scores_for_seq_aux, axis=-1)), axis=1))
					* self.alpha
				)
			else:
				mask_ce = jax.nn.one_hot(
					topk_idx_for_aux_loss.reshape(-1), num_classes=self.n_routed_experts
				)
				ce = jnp.mean(mask_ce.astype("float32"), axis=0)
				Pi = jnp.mean(scores_for_aux, axis=0)
				fi = ce * self.n_routed_experts
				aux_loss = jnp.sum(Pi * fi) * self.alpha
		else:
			aux_loss = None
		return topk_idx, topk_weight, aux_loss


class FlaxDeepseekV2MoE(nn.Module):
	def __init__(
		self,
		config: DeepseekV2Config,
		dtype: jnp.dtype = jnp.bfloat16,
		param_dtype: jnp.dtype = jnp.bfloat16,
		precision: tp.Optional[tp.Union[str, jax.lax.Precision]] = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__()
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.rngs = rngs
		self.num_experts_per_tok = config.num_experts_per_tok

		self.ep_size = 1
		self.experts_per_rank = config.n_routed_experts
		self.ep_rank = 0
		self.experts = self.experts = [
			FlaxDeepseekV2MLP(
				config=config,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				intermediate_size=self.config.moe_intermediate_size,
				rngs=rngs,
			)
			for i in range(self.config.n_routed_experts)
		]
		self.gate = FlaxMoEGate(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		if config.n_shared_experts is not None:
			intermediate_size = config.moe_intermediate_size * config.n_shared_experts
			self.shared_experts = FlaxDeepseekV2MoE(
				config=config,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				intermediate_size=intermediate_size,
				rngs=rngs,
			)

	def __call__(self, hidden_states: chex.Array):
		hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
		identity = hidden_states
		orig_shape = hidden_states.shape
		topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
		hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
		flat_topk_idx = topk_idx.reshape(-1)
		hidden_states = hidden_states.repeat(self.num_experts_per_tok, axis=0)
		y = jnp.empty_like(hidden_states)
		for i, expert in enumerate(self.experts):
			y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
		y = (y.reshape(*topk_weight.shape, -1) * jnp.expand_dims(topk_weight, -1)).sum(
			axis=1
		)
		y = y.reshape(*orig_shape)
		if self.config.n_shared_experts is not None:
			y = y + self.shared_experts(identity)
		return y


class FlaxDeepseekV2Attention(FlaxAttentionModule):
	def __init__(
		self,
		config: DeepseekV2Config,
		dtype: jnp.dtype = jnp.bfloat16,
		param_dtype: jnp.dtype = jnp.bfloat16,
		precision: tp.Optional[tp.Union[str, jax.lax.Precision]] = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__(config=config)
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.rngs = rngs

		self.attention_dropout = config.attention_dropout
		self.hidden_size = config.hidden_size
		self.num_heads = config.num_attention_heads

		self.max_position_embeddings = config.max_position_embeddings
		self.rope_theta = config.rope_theta
		self.q_lora_rank = config.q_lora_rank
		self.qk_rope_head_dim = config.qk_rope_head_dim
		self.kv_lora_rank = config.kv_lora_rank
		self.v_head_dim = config.v_head_dim
		self.qk_nope_head_dim = config.qk_nope_head_dim
		self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

		self.is_causal = True

		linear = functools.partial(
			nn.Linear,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
		)
		if self.config.q_lora_rank is None:
			self.q_proj = nn.Linear(
				self.hidden_size,
				self.num_heads * self.q_head_dim,
				use_bias=False,
				rngs=rngs,
			)
		else:
			self.q_a_proj = linear(
				self.hidden_size,
				config.q_lora_rank,
				use_bias=config.attention_bias,
				rngs=rngs,
			)
			self.q_a_layernorm = RMSNorm(
				config.q_lora_rank,
				eps=1e-6,
				dtype=dtype,
				param_dtype=param_dtype,
				rngs=rngs,
			)
			self.q_b_proj = linear(
				config.q_lora_rank,
				self.num_heads * self.q_head_dim,
				use_bias=False,
				rngs=rngs,
			)

		self.kv_a_proj_with_mqa = linear(
			self.hidden_size,
			config.kv_lora_rank + config.qk_rope_head_dim,
			use_bias=config.attention_bias,
			rngs=rngs,
		)
		self.kv_a_layernorm = RMSNorm(
			config.kv_lora_rank,
			dtype=dtype,
			eps=1e-6,
			param_dtype=param_dtype,
			rngs=rngs,
		)
		self.kv_b_proj = linear(
			config.kv_lora_rank,
			self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
			use_bias=False,
			rngs=rngs,
		)

		self.o_proj = linear(
			self.num_heads * self.v_head_dim,
			self.hidden_size,
			use_bias=config.attention_bias,
			rngs=rngs,
		)

		softmax_scale = self.q_head_dim**-0.5
		if self.config.rope_scaling is not None:
			mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
			scaling_factor = self.config.rope_scaling["factor"]
			if mscale_all_dim:
				mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
				softmax_scale = self.softmax_scale * mscale * mscale
		self.attention_performer = FlexibleAttentionModule(
			num_q_heads=self.config.num_attention_heads,
			num_kv_heads=self.config.num_attention_heads,
			attention_dropout=self.config.attention_dropout,
			head_dims=self.q_head_dim,
			precision=self.precision,
			force_float32_tpu=True,
			attn_mechanism="vanilla",
			mesh=self.config.mesh,
			sm_scale=softmax_scale,
			axis_name=self.config.attention_axis_name,
			base_config=self.config,
			_do_check=False,
		)

	def __call__(
		self,
		hidden_states: chex.Array,
		frequencies: tp.Tuple[chex.Array, chex.Array],
		attention_mask: chex.Array,
		position_ids: chex.Array,
		causal_mask: chex.Array,
		segment_ids: tp.Optional[chex.Array] = None,
		cache_view: tp.Optional[TransformerCacheView] = None,
		output_attentions: bool = False,
		fcm_mask: tp.Optional[chex.Array] = None,
	):
		"""
		Forward pass of the attention module.

		Args:
		    hidden_states (chex.Array): Input hidden states.
		    frequencies (tp.Tuple[chex.Array, chex.Array]): Cosine and sine components for rotary embeddings.
		    attention_mask (chex.Array): Mask to apply on the attention scores.
		    position_ids (chex.Array): Position indices for the tokens.
		    causal_mask (chex.Array): Causal mask for ensuring autoregressive behavior.
		    segment_ids (tp.Optional[chex.Array]): Segment IDs for segment-based attention (optional).
		    deterministic (bool): If True, disables dropout for deterministic behavior.
		    init_cache (bool): If True, initializes cache for caching keys and values.
		    output_attentions (bool): If True, outputs attention weights alongside the hidden states.
		    fcm_mask (tp.Optional[chex.Array]): fcm mask to be combined with attn mask and causal mask.
		Returns:
		    tp.Tuple[chex.Array, chex.Array]: A tuple containing the attention output and the attention weights.
		"""
		bsz, q_len, _ = hidden_states.shape

		if self.config.q_lora_rank is None:
			q = self.q_proj(hidden_states)
		else:
			q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
		q = q.reshape(bsz, q_len, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
		# Split into nope and pe parts
		q_nope, q_pe = q[..., : self.qk_nope_head_dim], q[..., self.qk_nope_head_dim :]
		# Key and Value projections with MQA (Multi-Query Attention) considerations
		compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
		k_pe = compressed_kv[..., self.kv_lora_rank :]
		compressed_kv = compressed_kv[..., : self.kv_lora_rank]

		k_pe = k_pe.reshape(bsz, q_len, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
		kv = (
			self.kv_b_proj(
				self.kv_a_layernorm(compressed_kv),
			)
			.reshape(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
			.transpose(0, 2, 1, 3)
		)

		k_nope = kv[..., : self.qk_nope_head_dim]
		value_states = kv[
			..., self.qk_nope_head_dim : self.qk_nope_head_dim + self.v_head_dim
		]

		sin, cos = frequencies

		q_pe, k_pe = apply_rotary_pos_emb(
			q=q_pe,
			k=k_pe,
			cos=cos,
			sin=sin,
			position_ids=position_ids,
		)

		query_states = jnp.zeros((bsz, self.num_heads, q_len, self.q_head_dim), q_pe.dtype)
		query_states.at[..., : self.qk_nope_head_dim].set(q_nope)
		query_states.at[..., self.qk_nope_head_dim :].set(q_pe)

		key_states = jnp.zeros((bsz, self.num_heads, q_len, self.q_head_dim), k_pe.dtype)
		key_states.at[..., : self.qk_nope_head_dim].set(k_nope)
		key_states.at[..., self.qk_nope_head_dim :].set(k_pe)

		query_states = query_states.transpose(0, 2, 1, 3)
		key_states = key_states.transpose(0, 2, 1, 3)
		value_states = value_states.transpose(0, 2, 1, 3)

		(
			key_states,
			value_states,
			attention_mask,
			attention_bias,
		) = self.concatenate(
			query=query_states,
			key=key_states,
			cache_view=cache_view,
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
			query_sequence_length=query_states.shape[1],
			key_value_sequence_length=key_states.shape[1],
			uses_cache=cache_view is not None,
			segment_ids=segment_ids,
			causal_mask=causal_mask,
		)

		attn_output = self.shard_attention_prod(
			self._merge_heads(attentions.attention_outputs)
		)
		attn_output = self.o_proj(attn_output)

		outputs = (
			(attn_output, attentions.attention_weights)
			if output_attentions
			else (attn_output,)
		)
		return outputs


class FlaxDeepseekV2DecoderLayer(nn.Module):
	def __init__(
		self,
		config: DeepseekV2Config,
		layer_idx: int,
		dtype: jnp.dtype = jnp.bfloat16,
		param_dtype: jnp.dtype = jnp.bfloat16,
		precision: tp.Optional[tp.Union[str, jax.lax.Precision]] = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__()
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.rngs = rngs
		self.layer_idx = layer_idx
		self.hidden_size = config.hidden_size

		attn_block = FlaxDeepseekV2Attention
		mlp_block = FlaxDeepseekV2MLP
		mlp_moe_block = FlaxDeepseekV2MoE

		attn_block, mlp_block = auto_remat(
			attn_block,
			mlp_block,
			policy=config.gradient_checkpointing,
		)
		self.self_attn = attn_block(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		self.mlp = (
			mlp_moe_block(
				config=config,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			if (
				config.n_routed_experts is not None
				and layer_idx >= config.first_k_dense_replace
				and layer_idx % config.moe_layer_freq == 0
			)
			else mlp_block(
				config=config,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
		)
		self.input_layernorm = RMSNorm(
			config.hidden_size,
			eps=config.rms_norm_eps,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)
		self.post_attention_layernorm = RMSNorm(
			config.hidden_size,
			eps=config.rms_norm_eps,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

	def __call__(
		self,
		hidden_states: chex.Array,
		frequencies: tp.Tuple[chex.Array, chex.Array],
		attention_mask: chex.Array,
		position_ids: chex.Array,
		causal_mask: chex.Array,
		segment_ids: tp.Optional[chex.Array] = None,
		cache_view: tp.Optional[TransformerCacheView] = None,
		output_attentions: bool = False,
		fcm_mask: tp.Optional[chex.Array] = None,
	):
		"""
		Forward pass of the module block.

		Args:
		    hidden_states (chex.Array): Input hidden states.
		    frequencies (tp.Tuple[chex.Array, chex.Array]): Cosine and sine components for rotary embeddings.
		    attention_mask (chex.Array): Mask to apply on the attention scores.
		    position_ids (chex.Array): Position indices for the tokens.
		    causal_mask (chex.Array): Causal mask for ensuring autoregressive behavior.
		    segment_ids (tp.Optional[chex.Array]): Segment IDs for segment-based attention (optional).
		    deterministic (bool): If True, disables dropout for deterministic behavior.
		    init_cache (bool): If True, initializes cache for caching keys and values.
		    output_attentions (bool): If True, outputs attention weights alongside the hidden states.
		    fcm_mask (tp.Optional[chex.Array]): fcm mask to be combined with attn mask and causal mask.
		Returns:
		    tp.Tuple[chex.Array, chex.Array]: A tuple containing the attention output and the attention weights.
		"""
		residual = hidden_states

		hidden_states = self.input_layernorm(hidden_states)

		# Self Attention
		attn_out = self.self_attn(
			hidden_states,
			frequencies,
			attention_mask,
			position_ids,
			causal_mask,
			segment_ids,
			cache_view,
			output_attentions,
			fcm_mask,
		)
		hidden_states, self_attn_weights = (
			attn_out if output_attentions else (attn_out[0], None)
		)
		hidden_states = residual + hidden_states

		# Fully Connected
		residual = hidden_states
		hidden_states = self.post_attention_layernorm(hidden_states)

		if self.config.use_scan_mlp:
			feed_forward_hidden_states = block_wise_ffn(
				self.mlp,
				hidden_states,
				self.config.scan_mlp_chunk_size,
			)
		else:
			feed_forward_hidden_states = self.mlp(
				hidden_states,
			)
		hidden_states = residual + feed_forward_hidden_states

		outputs = (hidden_states,)

		if output_attentions:
			outputs += (self_attn_weights,)

		return outputs  # type:ignore


@register_module(
	"base-module",
	DeepseekV2Config,
	model_type="deepseek_v2",
	embedding_layer_names=["embed_tokens"],
)
class DeepseekV2Model(EasyDeLBaseModule):
	def __init__(
		self,
		config: DeepseekV2Config,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
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
		self.embed_tokens = nn.Embed(
			self.config.vocab_size,
			self.config.hidden_size,
			embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

		self.layers = [
			FlaxDeepseekV2DecoderLayer(
				config=config,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				layer_idx=i,
				rngs=rngs,
			)
			for i in range(self.config.num_hidden_layers)
		]
		self.norm = RMSNorm(
			self.config.hidden_size,
			eps=self.config.rms_norm_eps,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

	@functools.cached_property
	def frequencies(self):
		initial_rope_kwargs = {}
		method = None
		if self.config.rope_scaling is not None:
			scaling_type = self.config.rope_scaling["type"]
			method = scaling_type
			if scaling_type != "yarn":
				initial_rope_kwargs = dict(scaling_factor=self.config.rope_scaling["factor"])
			else:
				initial_rope_kwargs = {
					key: self.config.rope_scaling[key]
					for key in [
						"original_max_position_embeddings",
						"beta_fast",
						"beta_slow",
						"mscale",
						"mscale_all_dim",
					]
					if key in self.config.rope_scaling
				}
				initial_rope_kwargs["scaling_factor"] = self.config.rope_scaling["factor"]
		return init_deepseek_rotary_embedding(
			dim=self.config.qk_rope_head_dim,
			max_position_embeddings=self.config.granted_freq_max_position_embedding,
			base=self.config.rope_theta,
			method=method,  # type:ignore
			kwargs=initial_rope_kwargs,
		)

	def __call__(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		return_dict: bool = True,
	) -> tp.Union[FlaxBaseModelOutput, tp.Tuple]:
		"""
		Forward pass through the Deepseekv2 module.

		Args:
		    input_ids (chex.Array): Input tensor containing token IDs.
		    attention_mask (chex.Array): Mask for attention.
		    position_ids (chex.Array): Positional indices.
		    segment_ids (tp.Optional[chex.Array]): Segment IDs for different input parts.
		    inputs_embeds (tp.Optional[chex.Array]): Embedded input tensor.
		    output_attentions (tp.Optional[bool]): If True, output attention weights.
		    output_hidden_states (tp.Optional[bool]): If True, output hidden states.
		    init_cache (bool): If True, initialize cache for decoding.
		    deterministic (bool): If True, disable dropout.
		    return_dict (bool): If True, return a dictionary of outputs.

		Returns:
		    FlaxBaseModelOutput | tp.Tuple: Model output, either as a named tuple or a standard tuple.
		"""
		if (input_ids is None) ^ (inputs_embeds is not None):
			raise ValueError(
				"You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
			)
		if inputs_embeds is None:
			inputs_embeds = self.embed_tokens(input_ids.astype("i4"))
		batch_size, sequence_length, _ = inputs_embeds.shape

		all_attentions = () if output_attentions else None
		all_hidden_states = () if output_hidden_states else None
		assert (
			sequence_length <= self.config.max_position_embeddings
		), f"Maximum Position Embedding Reached ! (Excepted <= {self.config.max_position_embeddings} got {sequence_length})"
		if attention_mask is None:
			attention_mask = jnp.ones((batch_size, sequence_length), "i4")
		if position_ids is None:
			position_ids = jnp.broadcast_to(
				jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
				(batch_size, sequence_length),
			).astype(jnp.int32)
		if attention_mask.ndim == 2:
			attention_mask = jnp.expand_dims(attention_mask, (1, 2))
		hidden_states = inputs_embeds
		if past_key_values is None:
			past_key_values = TransformerCache.init_empty(len(self.layers))
		for idx, layer in enumerate(self.layers):
			if output_hidden_states:
				all_hidden_states += (hidden_states,)

			output = layer(
				hidden_states=hidden_states,
				frequencies=self.frequencies,
				attention_mask=attention_mask,
				position_ids=position_ids,
				causal_mask=self.causal_mask,
				output_attentions=output_attentions,
				segment_ids=segment_ids,
				cache_view=past_key_values.views[idx],
			)
			hidden_states = output[0]

			if output_attentions:
				all_attentions += (output[1],)

		hidden_states = self.norm(hidden_states)

		if output_hidden_states:
			all_hidden_states += (hidden_states,)
		outputs = (hidden_states, all_hidden_states, all_attentions, past_key_values)

		if not return_dict:
			return tuple(value for value in outputs if value is not None)

		return FlaxBaseModelOutput(
			last_hidden_state=hidden_states,
			hidden_states=all_hidden_states,
			attentions=all_attentions,
			past_key_values=past_key_values,
		)


@register_module(
	"causal-language-model",
	DeepseekV2Config,
	model_type="deepseek_v2",
	embedding_layer_names=["embed_tokens"],
)
class DeepseekV2ForCausalLM(EasyDeLBaseModule):
	def __init__(
		self,
		config: DeepseekV2Config,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: tp.Optional[tp.Union[jax.lax.Precision, str]] = None,
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
		self.model = DeepseekV2Model(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)
		self.lm_head = nn.Linear(
			config.hidden_size,
			config.vocab_size,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			use_bias=False,
			kernel_init=nn.initializers.normal(config.initializer_range),
			rngs=rngs,
			**get_dot_general_by_bits(config.bits, config.easy_method),
		)

	def __call__(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		inputs_embeds: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		return_dict: bool = True,
	) -> tp.Union[FlaxCausalLMOutput, tp.Tuple]:
		outputs = self.model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			inputs_embeds=inputs_embeds,
			past_key_values=past_key_values,
			segment_ids=segment_ids,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		hidden_states = outputs[0]

		if self.config.tie_word_embeddings:
			# self.lm_head.kernel.value = self.model.embed_tokens.embedding.value.T
			# lm_logits = self.lm_head(hidden_states)
			lm_logits = hidden_states @ self.model.embed_tokens.embedding.value.T
		else:
			lm_logits = self.lm_head(hidden_states)

		if not return_dict:
			return (lm_logits,) + outputs[1:]

		return FlaxCausalLMOutput(
			logits=lm_logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
			past_key_values=outputs.past_key_values,
		)
