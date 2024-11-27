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
import typing
import warnings
from typing import Optional, Tuple, Union

import chex
import flax
import jax
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from jax import lax
from jax import numpy as jnp

from easydel.etils.etils import EasyDeLGradientCheckPointers
from easydel.layers.attention import FlaxAttentionModule, FlexibleAttentionModule
from easydel.layers.norms import RMSNorm
from easydel.modules.deepseek_v2.deepseek_configuration import (
	DeepseekV2Config as DeepseekV2Config,
)
from easydel.modules.factory import register_module
from easydel.modules.flax_modeling_utils import (
	ACT2FN,
	block_wise_ffn,
	control_mlp_sharding,
	get_dot_general_by_bits,
	get_gradient_checkpoint_policy,
)
from easydel.modules.modeling_flax_outputs import (
	FlaxBaseModelOutput,
	FlaxCausalLMOutput,
	FlaxMaskedLMOutput,
)
from easydel.modules.modeling_utils import wrap_easydel_module

re_mat = nn_partitioning.remat


@flax.struct.dataclass
class MoeModelOutput:
	last_hidden_state: chex.Array = None
	hidden_states: Optional[Tuple[chex.Array]] = None
	attentions: Optional[Tuple[chex.Array]] = None
	router_logits: Optional[Tuple[chex.Array]] = None


@flax.struct.dataclass
class MoeCausalLMOutput(FlaxMaskedLMOutput):
	aux_loss: Optional[chex.Array] = None
	router_logits: Optional[Tuple[chex.Array]] = None


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
	method: typing.Literal["linear", "yarn", "dynamic", None] = None,
	kwargs: typing.Optional[dict] = None,
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
	config: DeepseekV2Config
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16
	precision: Optional[Union[str, jax.lax.Precision]] = None
	hidden_size: Optional[int] = None
	intermediate_size: Optional[int] = None

	def setup(self) -> None:
		dense = functools.partial(
			nn.Dense,
			use_bias=False,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
			kernel_init=nn.initializers.normal(),
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)
		self.gate_proj = dense(self.intermediate_size or self.config.intermediate_size)
		self.up_proj = dense(self.intermediate_size or self.config.intermediate_size)
		self.down_proj = dense(self.hidden_size or self.config.hidden_size)
		self.act_fn = ACT2FN[self.config.hidden_act]

	def __call__(
		self,
		x: chex.Array,
		e: bool = False,  # Ignored
	):
		x = control_mlp_sharding(x, self.config.partition_axis)
		return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class FlaxMoEGate(nn.Module):
	config: DeepseekV2Config
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16
	precision: Optional[Union[str, jax.lax.Precision]] = None

	def setup(self) -> None:
		config = self.config
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
		self.weight = self.param(
			"kernel",
			nn.initializers.kaiming_uniform(dtype=self.param_dtype),
			(self.n_routed_experts, self.gating_dim),
		)

	def __call__(self, hidden_states, deterministic: bool = True):
		bsz, seq_len, h = hidden_states.shape
		hidden_states = hidden_states.reshape(-1, h)
		logits = jax.lax.batch_matmul(
			hidden_states.astype(jnp.float32),
			self.weight.astype(jnp.float32),
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
		if not deterministic and self.alpha > 0.0:
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


class FlaxDeepseekV2MLPCollection(nn.Module):
	config: DeepseekV2Config
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16
	precision: Optional[Union[str, jax.lax.Precision]] = None

	def setup(self) -> None:
		self.experts = [
			FlaxDeepseekV2MLP(
				config=self.config,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
				intermediate_size=self.config.moe_intermediate_size,
				name=str(i),
			)
			for i in range(self.config.n_routed_experts)
		]

	def __call__(self, hidden_states, flat_topk_idx):
		y = jnp.empty_like(hidden_states)
		for i, expert in enumerate(self.experts):
			y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
		return y


class FlaxDeepseekV2MoE(nn.Module):
	config: DeepseekV2Config
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16
	precision: Optional[Union[str, jax.lax.Precision]] = None

	def setup(self) -> None:
		config = self.config
		self.num_experts_per_tok = config.num_experts_per_tok

		self.ep_size = 1
		self.experts_per_rank = config.n_routed_experts
		self.ep_rank = 0
		self.experts = FlaxDeepseekV2MLPCollection(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.gate = FlaxMoEGate(
			config=config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		if config.n_shared_experts is not None:
			intermediate_size = config.moe_intermediate_size * config.n_shared_experts
			self.shared_experts = FlaxDeepseekV2MoE(
				config=self.config,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
				intermediate_size=intermediate_size,
			)

	def __call__(
		self,
		hidden_states: chex.Array,
		e: bool = False,  # ignored !
	):
		hidden_states = control_mlp_sharding(hidden_states, self.config.partition_axis)
		identity = hidden_states
		orig_shape = hidden_states.shape
		topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
		hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
		flat_topk_idx = topk_idx.reshape(-1)
		hidden_states = hidden_states.repeat(self.num_experts_per_tok, axis=0)
		y = self.experts(hidden_states=hidden_states, flat_topk_idx=flat_topk_idx)
		y = (y.reshape(*topk_weight.shape, -1) * jnp.expand_dims(topk_weight, -1)).sum(
			axis=1
		)
		y = y.reshape(*orig_shape)
		if self.config.n_shared_experts is not None:
			y = y + self.shared_experts(identity)
		return y


class FlaxDeepseekV2Attention(FlaxAttentionModule):
	config: DeepseekV2Config
	layer_idx: int
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16
	precision: Optional[Union[str, jax.lax.Precision]] = None

	def setup(self) -> None:
		config = self.config

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

		dense_class = functools.partial(
			nn.Dense,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)

		self.q_a_proj = dense_class(config.q_lora_rank, use_bias=config.attention_bias)
		self.q_a_layernorm = RMSNorm(config.q_lora_rank)
		self.q_b_proj = dense_class(self.num_heads * self.q_head_dim, use_bias=False)

		self.kv_a_proj_with_mqa = dense_class(
			config.kv_lora_rank + config.qk_rope_head_dim,
			use_bias=config.attention_bias,
		)
		self.kv_a_layernorm = RMSNorm(config.kv_lora_rank)
		self.kv_b_proj = dense_class(
			self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
			use_bias=False,
		)

		self.o_proj = dense_class(self.hidden_size, use_bias=config.attention_bias)

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
		frequencies: Tuple[chex.Array, chex.Array],
		attention_mask: chex.Array,
		position_ids: chex.Array,
		causal_mask: chex.Array,
		segment_ids: Optional[chex.Array] = None,
		deterministic: bool = True,
		init_cache: bool = False,
		output_attentions: bool = False,
		fcm_mask: Optional[chex.Array] = None,
	):
		"""
		Forward pass of the attention module.

		Args:
		    hidden_states (chex.Array): Input hidden states.
		    frequencies (Tuple[chex.Array, chex.Array]): Cosine and sine components for rotary embeddings.
		    attention_mask (chex.Array): Mask to apply on the attention scores.
		    position_ids (chex.Array): Position indices for the tokens.
		    causal_mask (chex.Array): Causal mask for ensuring autoregressive behavior.
		    segment_ids (Optional[chex.Array]): Segment IDs for segment-based attention (optional).
		    deterministic (bool): If True, disables dropout for deterministic behavior.
		    init_cache (bool): If True, initializes cache for caching keys and values.
		    output_attentions (bool): If True, outputs attention weights alongside the hidden states.
		    fcm_mask (Optional[chex.Array]): fcm mask to be combined with attn mask and causal mask.
		Returns:
		    Tuple[chex.Array, chex.Array]: A tuple containing the attention output and the attention weights.
		"""
		bsz, q_len, _ = hidden_states.shape

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

		dropout_rng = None

		if not deterministic and self.config.attn_config.attn_pdrop > 0.0:
			dropout_rng = self.make_rng("dropout")
		query_states, key_states, value_states, attention_mask, attention_bias = (
			self.concatenate_to_cache(
				init_cache=init_cache,
				query=query_states,
				key=key_states,
				value=value_states,
				attention_mask=attention_mask,
				causal_mask=causal_mask,
				fcm_mask=fcm_mask,
			)
		)
		query_length, key_length = query_states.shape[1], key_states.shape[1]

		attentions = self.attention_performer(
			query_states=query_states,
			key_states=key_states,
			value_states=value_states,
			bias=attention_bias,
			attention_mask=attention_mask,
			causal=True,
			dropout_rng=dropout_rng,
			deterministic=deterministic,
			query_sequence_length=query_length,
			key_value_sequence_length=key_length,
			uses_cache=self.has_variable("cache", "cached_key") or init_cache,
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
	config: DeepseekV2Config
	layer_idx: int
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16
	precision: Optional[Union[str, jax.lax.Precision]] = None

	def setup(self):
		config = self.config
		layer_idx = self.layer_idx
		self.hidden_size = config.hidden_size

		attn_block = FlaxDeepseekV2Attention
		mlp_block = FlaxDeepseekV2MLP
		mlp_moe_block = FlaxDeepseekV2MoE

		if self.config.gradient_checkpointing != EasyDeLGradientCheckPointers.NONE:
			attn_block = re_mat(
				attn_block,
				policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
				static_argnums=(1, 4, 6, 7, 8),
			)
			mlp_block = re_mat(
				mlp_block,
				policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
				static_argnums=(1,),
			)

			mlp_moe_block = re_mat(
				mlp_moe_block,
				policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing),
				static_argnums=(1,),
			)

		self.self_attn = attn_block(
			config=config,
			layer_idx=self.layer_idx,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)

		self.mlp = (
			mlp_moe_block(
				config=config,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
			)
			if (
				config.n_routed_experts is not None
				and layer_idx >= config.first_k_dense_replace
				and layer_idx % config.moe_layer_freq == 0
			)
			else mlp_block(
				config=config,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
			)
		)
		self.input_layernorm = RMSNorm(
			config.hidden_size,
			eps=config.rms_norm_eps,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
		)
		self.post_attention_layernorm = RMSNorm(
			config.hidden_size,
			eps=config.rms_norm_eps,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
		)

	def __call__(
		self,
		hidden_states: chex.Array,
		frequencies: Tuple[chex.Array, chex.Array],
		attention_mask: chex.Array,
		position_ids: chex.Array,
		causal_mask: chex.Array,
		segment_ids: Optional[chex.Array] = None,
		deterministic: bool = True,
		init_cache: bool = False,
		output_attentions: bool = False,
		fcm_mask: Optional[chex.Array] = None,
	):
		"""
		Forward pass of the module block.

		Args:
		    hidden_states (chex.Array): Input hidden states.
		    frequencies (Tuple[chex.Array, chex.Array]): Cosine and sine components for rotary embeddings.
		    attention_mask (chex.Array): Mask to apply on the attention scores.
		    position_ids (chex.Array): Position indices for the tokens.
		    causal_mask (chex.Array): Causal mask for ensuring autoregressive behavior.
		    segment_ids (Optional[chex.Array]): Segment IDs for segment-based attention (optional).
		    deterministic (bool): If True, disables dropout for deterministic behavior.
		    init_cache (bool): If True, initializes cache for caching keys and values.
		    output_attentions (bool): If True, outputs attention weights alongside the hidden states.
		    fcm_mask (Optional[chex.Array]): fcm mask to be combined with attn mask and causal mask.
		Returns:
		    Tuple[chex.Array, chex.Array]: A tuple containing the attention output and the attention weights.
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
			deterministic,
			init_cache,
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
				deterministic,
			)
		else:
			feed_forward_hidden_states = self.mlp(
				hidden_states,
				deterministic,
			)
		hidden_states = residual + feed_forward_hidden_states

		outputs = (hidden_states,)

		if output_attentions:
			outputs += (self_attn_weights,)

		return outputs  # type:ignore


class FlaxDeepseekV2DecoratorCollection(nn.Module):
	config: DeepseekV2Config
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16
	precision: Optional[Union[str, jax.lax.Precision]] = None

	def setup(self) -> None:
		if self.config.attn_mechanism != "vanilla":
			warnings.warn("Deepseek2 only support vanilla attention", stacklevel=3)
		self.layers = [
			FlaxDeepseekV2DecoderLayer(
				config=self.config,
				dtype=self.dtype,
				param_dtype=self.param_dtype,
				precision=self.precision,
				layer_idx=i,
				name=str(i),
			)
			for i in range(self.config.num_hidden_layers)
		]

	def __call__(
		self,
		hidden_states: chex.Array,
		frequencies: Tuple[chex.Array, chex.Array],
		attention_mask: chex.Array,
		causal_mask: chex.Array,
		position_ids: chex.Array,
		segment_ids: Optional[chex.Array] = None,
		deterministic: bool = True,
		init_cache: bool = False,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
	) -> Tuple[chex.Array, Optional[chex.Array], chex.Array]:
		"""
		Forward pass through the collection of decoder layers.

		Args:
		    hidden_states (chex.Array): Input tensor containing the hidden states.
		    frequencies (Tuple[chex.Array, chex.Array]): Frequency positional encodings.
		    attention_mask (chex.Array): Mask to apply during attention.
		    causal_mask (chex.Array): Causal mask for autoregressive decoding.
		    position_ids (chex.Array): Positional indices for the sequence.
		    segment_ids (Optional[chex.Array]): Segment IDs for distinguishing different parts of the input.
		    deterministic (bool): If True, disables dropout.
		    init_cache (bool): If True, initializes caching mechanism for fast decoding.
		    output_attentions (bool): If True, returns attention weights.
		    output_hidden_states (bool): If True, returns hidden states.

		Returns:
		    Tuple[chex.Array, Optional[chex.Array], chex.Array]:
		        - hidden_states: The output tensor after layer processing.
		        - all_hidden_states: all of Hidden states (if `output_hidden_states` is True).
		        - self_attn_weights: Attention weights (if `output_attentions` is True).

		"""
		all_attentions = () if output_attentions else None
		all_hidden_states = () if output_hidden_states else None

		if not deterministic and self.config.fcm_max_ratio > 0:
			# Apply forgetful causal mask
			batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]
			fcm_ratio = jax.random.uniform(
				self.make_rng("fcm"),
				shape=(batch_size, 1, 1, 1),
				minval=self.config.fcm_min_ratio,
				maxval=self.config.fcm_max_ratio,
			)
			fcm_mask = (
				jax.random.uniform(
					self.make_rng("fcm"), shape=(batch_size, 1, seq_length, seq_length)
				)
				> fcm_ratio
			)
			fcm_mask = fcm_mask.at[:, :, :, 0].set(True)
			fcm_mask = fcm_mask.astype("bool")
		else:
			fcm_mask = None

		for layer in self.layers:
			if output_hidden_states:
				all_hidden_states += (hidden_states,)

			output = layer(
				hidden_states=hidden_states,
				frequencies=frequencies,
				attention_mask=attention_mask,
				position_ids=position_ids,
				causal_mask=causal_mask,
				deterministic=deterministic,
				init_cache=init_cache,
				output_attentions=output_attentions,
				fcm_mask=fcm_mask,
				segment_ids=segment_ids,
			)
			hidden_states = output[0]

			if output_attentions:
				output_attentions += (output[1],)

		return hidden_states, all_hidden_states, all_attentions


@register_module(
	"base-module",
	DeepseekV2Config,
	model_type="deepseek_v2",
	embedding_layer_names=["embed_tokens"],
)
@wrap_easydel_module(config_class=DeepseekV2Config, base_model_prefix="model")
class FlaxDeepseekV2Model(nn.Module):
	config: DeepseekV2Config
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self):
		self.embed_tokens = nn.Embed(
			self.config.vocab_size,
			self.config.hidden_size,
			embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
			dtype=self.dtype,
			param_dtype=self.param_dtype,
		)

		self.layers = FlaxDeepseekV2DecoratorCollection(
			self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.norm = RMSNorm(
			self.config.hidden_size,
			eps=self.config.rms_norm_eps,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
		)

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
		self.frequencies = init_deepseek_rotary_embedding(
			dim=self.config.qk_rope_head_dim,
			max_position_embeddings=self.config.granted_freq_max_position_embedding,
			base=self.config.rope_theta,
			method=method,  # type:ignore
			kwargs=initial_rope_kwargs,
		)
		self.causal_mask = flax.linen.make_causal_mask(
			jnp.ones(
				shape=(1, self.config.granted_mask_max_position_embedding),
				dtype="bool",
			),
			dtype="bool",
		)

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
		deterministic: bool = True,
		return_dict: bool = True,
	) -> Union[FlaxBaseModelOutput, Tuple]:
		"""
		Forward pass through the Deepseekv2 module.

		Args:
		    input_ids (chex.Array): Input tensor containing token IDs.
		    attention_mask (chex.Array): Mask for attention.
		    position_ids (chex.Array): Positional indices.
		    segment_ids (Optional[chex.Array]): Segment IDs for different input parts.
		    input_embeds (Optional[chex.Array]): Embedded input tensor.
		    output_attentions (Optional[bool]): If True, output attention weights.
		    output_hidden_states (Optional[bool]): If True, output hidden states.
		    init_cache (bool): If True, initialize cache for decoding.
		    deterministic (bool): If True, disable dropout.
		    return_dict (bool): If True, return a dictionary of outputs.

		Returns:
		    FlaxBaseModelOutput | Tuple: Model output, either as a named tuple or a standard tuple.
		"""
		if input_embeds is None and input_ids is not None:
			input_embeds = self.embed_tokens(input_ids.astype("i4"))
		else:
			raise ValueError("you should specify input_embeds or input_ids one of them")
		batch_size, sequence_length, _ = input_embeds.shape

		assert (
			sequence_length <= self.config.max_position_embeddings
		), f"Maximum Position Embedding Reached ! (Excepted <= {self.config.max_position_embeddings} got {sequence_length})"
		if attention_mask.ndim == 2:
			attention_mask = jnp.expand_dims(attention_mask, (1, 2))

		outputs = self.layers(
			hidden_states=input_embeds,
			frequencies=self.frequencies,
			attention_mask=attention_mask,
			position_ids=position_ids,
			causal_mask=self.causal_mask,
			deterministic=deterministic,
			init_cache=init_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			segment_ids=segment_ids,
		)

		hidden_states = outputs[0]
		hidden_states = self.norm(hidden_states)

		if output_hidden_states:
			all_hidden_states = outputs[1] + (hidden_states,)
			outputs = (hidden_states, all_hidden_states) + outputs[2:]
		else:
			outputs = (hidden_states,) + outputs[1:]

		if not return_dict:
			return tuple(value for value in outputs if value is not None)

		return FlaxBaseModelOutput(
			last_hidden_state=hidden_states,
			hidden_states=outputs[1],
			attentions=outputs[-1],
		)


@register_module(
	"causal-language-model",
	DeepseekV2Config,
	model_type="deepseek_v2",
	embedding_layer_names=["embed_tokens"],
)
@wrap_easydel_module(config_class=DeepseekV2Config, base_model_prefix="model")
class FlaxDeepseekV2ForCausalLM(nn.Module):
	config: DeepseekV2Config
	dtype: jnp.dtype = jnp.bfloat16
	param_dtype: jnp.dtype = jnp.bfloat16
	precision: Optional[jax.lax.Precision] = None

	def setup(self) -> None:
		self.model = FlaxDeepseekV2Model.flax_module(
			config=self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.lm_head = nn.Dense(
			self.config.vocab_size,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
			use_bias=False,
			kernel_init=nn.initializers.normal(self.config.initializer_range),
			**get_dot_general_by_bits(self.config.bits, self.config.easy_method),
		)

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
		deterministic: bool = True,
		return_dict: bool = True,
	) -> Union[FlaxCausalLMOutput, Tuple]:
		"""
		Forward pass through the Deepseekv2 module.

		Args:
		    input_ids (chex.Array): Input tensor containing token IDs.
		    attention_mask (chex.Array): Mask for attention.
		    position_ids (chex.Array): Positional indices.
		    segment_ids (Optional[chex.Array]): Segment IDs for different input parts.
		    input_embeds (Optional[chex.Array]): Embedded input tensor.
		    output_attentions (Optional[bool]): If True, output attention weights.
		    output_hidden_states (Optional[bool]): If True, output hidden states.
		    init_cache (bool): If True, initialize cache for decoding.
		    deterministic (bool): If True, disable dropout.
		    return_dict (bool): If True, return a dictionary of outputs.

		Returns:
		    FlaxCausalLMOutput | Tuple: Model output, either as a named tuple or a standard tuple.
		"""
		batch_size, seq_length = input_ids.shape

		if attention_mask is None:
			attention_mask = jnp.ones_like(input_ids)
		if position_ids is None:
			position_ids = jnp.broadcast_to(
				jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
				(batch_size, seq_length),
			)
		outputs = self.model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			deterministic=deterministic,
			input_embeds=input_embeds,
			init_cache=init_cache,
			segment_ids=segment_ids,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		hidden_states = outputs[0]

		if self.config.tie_word_embeddings:
			shared_kernel = self.transformer.variables["params"]["embed_tokens"][
				"embedding"
			].T.astype(self.param_dtype)
			lm_logits = self.lm_head.apply(
				{"params": {"kernel": shared_kernel}}, hidden_states
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
