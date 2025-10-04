# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""
DiT-MoE (Mixture of Experts Diffusion Transformer) implementation.

This module implements a DiT architecture with sparse Mixture of Experts (MoE)
following DeepSeek V2's MoE design. It replaces standard MLP layers with MoE blocks
containing shared experts (always active) and routed experts (selected via top-k routing).
"""

import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
from eformer import common_types
from flax import nnx as nn
from jax import lax

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import BaseModelOutput
from easydel.infra.utils import auto_remat
from easydel.layers.attention import FlexibleAttentionModule
from easydel.layers.linear import ColumnParallelLinear, RowParallelLinear
from easydel.layers.moe import BaseMoeModule, MoeLoadBalancingStrategy, MoeRoutingStrategy

from .dit_moe_configuration import DiTMoEConfig


def modulate(x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
	"""Apply adaptive layer norm modulation: (1 + scale) * x + shift."""
	return x * (1 + scale) + shift


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> jnp.ndarray:
	"""
	Generate 2D sinusoidal positional embeddings.

	Args:
		embed_dim: Embedding dimension (must be divisible by 2)
		grid_size: Size of the spatial grid (grid_size x grid_size patches)

	Returns:
		Positional embeddings of shape [1, grid_size**2, embed_dim]
	"""
	assert embed_dim % 2 == 0

	# Create grid coordinates
	grid_h = jnp.arange(grid_size, dtype=jnp.float32)
	grid_w = jnp.arange(grid_size, dtype=jnp.float32)
	grid = jnp.meshgrid(grid_w, grid_h, indexing="xy")
	grid = jnp.stack(grid, axis=0)
	grid = grid.reshape(2, -1)

	# Generate embeddings
	omega = jnp.arange(embed_dim // 4, dtype=jnp.float32)
	omega /= embed_dim / 4.0
	omega = 1.0 / (10000**omega)

	out = jnp.einsum("hw,d->hwd", grid, omega)
	emb_sin = jnp.sin(out)
	emb_cos = jnp.cos(out)

	emb = jnp.concatenate([emb_sin, emb_cos], axis=-1)
	emb = emb.reshape(2, -1, embed_dim // 2)
	emb = jnp.concatenate([emb[0], emb[1]], axis=-1)

	return jnp.expand_dims(emb, 0)


class TimestepEmbedding(nn.Module):
	"""Embeds scalar timesteps into vector representations using sinusoidal encoding."""

	def __init__(
		self,
		hidden_size: int,
		frequency_embedding_size: int = 256,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		*,
		rngs: nn.Rngs,
	):
		self.hidden_size = hidden_size
		self.frequency_embedding_size = frequency_embedding_size
		self.dtype = dtype
		self.param_dtype = param_dtype

		self.mlp = [
			nn.Linear(
				frequency_embedding_size,
				hidden_size,
				dtype=dtype,
				param_dtype=param_dtype,
				rngs=rngs,
			),
			nn.Linear(
				hidden_size,
				hidden_size,
				dtype=dtype,
				param_dtype=param_dtype,
				rngs=rngs,
			),
		]

	def timestep_embedding(self, t: jnp.ndarray, max_period: float = 10000.0) -> jnp.ndarray:
		"""Generate sinusoidal timestep embeddings."""
		t = jax.lax.convert_element_type(t, jnp.float32)
		half = self.frequency_embedding_size // 2
		freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half)
		args = t[:, None] * freqs[None]
		embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
		return embedding.astype(self.dtype)

	def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
		"""Forward pass."""
		t_freq = self.timestep_embedding(t)
		t_emb = self.mlp[0](t_freq)
		t_emb = nn.silu(t_emb)
		t_emb = self.mlp[1](t_emb)
		return t_emb


class LabelEmbedding(nn.Module):
	"""Embeds class labels into vector representations."""

	def __init__(
		self,
		num_classes: int,
		hidden_size: int,
		dropout_prob: float = 0.1,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		*,
		rngs: nn.Rngs,
	):
		self.num_classes = num_classes
		self.hidden_size = hidden_size
		self.dropout_prob = dropout_prob

		# +1 for unconditional class token
		self.embedding_table = nn.Embed(
			num_classes + 1,
			hidden_size,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

	def __call__(self, labels: jnp.ndarray, force_drop_ids: jnp.ndarray | None = None) -> jnp.ndarray:
		"""Forward pass with classifier-free guidance dropout."""
		use_dropout = self.dropout_prob > 0
		if (force_drop_ids is None) and use_dropout:
			drop_ids = jax.random.bernoulli(
				nn.rng_key("dropout"),
				self.dropout_prob,
				shape=labels.shape,
			)
		else:
			drop_ids = force_drop_ids if force_drop_ids is not None else jnp.zeros_like(labels, dtype=bool)

		# Use num_classes as the unconditional token
		labels = jnp.where(drop_ids, self.num_classes, labels)
		embeddings = self.embedding_table(labels)
		return embeddings


class PatchEmbed(nn.Module):
	"""Converts images into a sequence of patch embeddings."""

	def __init__(
		self,
		image_size: int,
		patch_size: int,
		in_channels: int,
		hidden_size: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		*,
		rngs: nn.Rngs,
	):
		self.image_size = image_size
		self.patch_size = patch_size
		self.in_channels = in_channels
		self.hidden_size = hidden_size
		self.num_patches = (image_size // patch_size) ** 2

		self.proj = nn.Conv(
			in_features=in_channels,
			out_features=hidden_size,
			kernel_size=(patch_size, patch_size),
			strides=(patch_size, patch_size),
			padding="VALID",
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

	def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
		"""Forward pass."""
		B, H, W, C = x.shape
		assert H == self.image_size and W == self.image_size, \
			f"Input image size ({H}x{W}) doesn't match model ({self.image_size}x{self.image_size})"

		x = self.proj(x)
		x = x.reshape(B, -1, self.hidden_size)
		return x


class MoEGate(nn.Module):
	"""MoE gating network following DeepSeek V2 design."""

	def __init__(
		self,
		config: DiTMoEConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.PrecisionLike = None,
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

	def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
		"""Compute gating weights for experts."""
		seu = hidden_states.shape[0] * hidden_states.shape[1]  # batch * seq_len
		hidden_states_flat = hidden_states.reshape(seu, -1)

		logits = jax.lax.batch_matmul(
			hidden_states_flat.astype(jnp.float32),
			self.kernel.value.astype(jnp.float32),
			precision=self.precision,
		)

		if self.scoring_func == "softmax":
			scores = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
		else:
			raise NotImplementedError(f"Unsupported scoring function: {self.scoring_func}")

		if self.topk_method == "greedy":
			topk_weight, _ = jax.lax.top_k(scores, k=self.top_k)
		elif self.topk_method == "group_limited_greedy":
			group_scores = scores.reshape(seu, self.n_group, -1).max(axis=-1)
			top_k_indices = lax.top_k(group_scores, self.topk_group)[1]

			group_mask = jnp.zeros_like(group_scores)
			n_indices = jnp.arange(group_mask.shape[0])[:, None]
			group_mask = group_mask.at[n_indices, top_k_indices].set(1)

			score_mask = jnp.repeat(group_mask[:, :, None], self.n_routed_experts // self.n_group, axis=2)
			score_mask = score_mask.reshape(seu, -1)
			masked_scores = jnp.where(score_mask, scores, 0.0)
			topk_weight, _ = lax.top_k(masked_scores, self.top_k)
		else:
			raise ValueError(f"Unknown topk_method: {self.topk_method}")

		if self.top_k > 1 and self.norm_topk_prob:
			denominator = jnp.sum(topk_weight, axis=-1, keepdims=True) + 1e-20
			topk_weight = topk_weight / denominator
		else:
			topk_weight = topk_weight * self.routed_scaling_factor

		return topk_weight


class DiTMLP(nn.Module):
	"""Standard MLP used in dense layers and shared experts."""

	def __init__(
		self,
		config: DiTMoEConfig,
		intermediate_size: int,
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

		linear_cls = partial(
			nn.Linear,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=config.use_bias,
			rngs=rngs,
		)

		self.fc1 = linear_cls(config.hidden_size, intermediate_size)
		self.fc2 = linear_cls(intermediate_size, config.hidden_size)
		self.dropout = nn.Dropout(config.mlp_dropout, rngs=rngs) if config.mlp_dropout > 0 else lambda x: x

	def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
		"""Forward pass."""
		hidden_states = self.fc1(hidden_states)
		hidden_states = nn.gelu(hidden_states) if self.config.hidden_act == "gelu" else nn.silu(hidden_states)
		hidden_states = self.dropout(hidden_states)
		hidden_states = self.fc2(hidden_states)
		hidden_states = self.dropout(hidden_states)
		return hidden_states


class DiTMLPMoE(nn.Module):
	"""MoE expert layer with multiple MLPs."""

	def __init__(
		self,
		config: DiTMoEConfig,
		intermediate_size: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.intermediate_size = intermediate_size

		# Create routed experts
		self.experts = [
			DiTMLP(
				config=config,
				intermediate_size=intermediate_size,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			for _ in range(config.n_routed_experts)
		]

	def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
		"""Forward pass - routes to different experts."""
		# This will be called by BaseMoeModule's routing logic
		# Each expert processes its assigned tokens
		return jnp.stack([expert(hidden_states) for expert in self.experts])


class DiTMoE(BaseMoeModule):
	"""
	DiT MoE layer following DeepSeek V2 architecture.

	Combines shared experts (always active) with routed experts (selected via top-k).
	Unlike standard MoE, DeepSeek doesn't use router auxiliary losses.
	"""

	def __init__(
		self,
		config: DiTMoEConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__(
			config=config,
			n_routed_experts=config.n_routed_experts,
			num_experts_per_tok=config.num_experts_per_tok,
			hidden_size=config.hidden_size,
			lbl_coef=None,  # DeepSeek doesn't use load balancing loss
			rzl_coef=None,  # DeepSeek doesn't use router z-loss
			routing_strategy=MoeRoutingStrategy.TOP_K,
			load_balancing_strategy=MoeLoadBalancingStrategy.STANDARD,
		)
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision
		self.rngs = rngs

		# Routed experts
		self.experts = DiTMLPMoE(
			config=config,
			intermediate_size=config.moe_intermediate_size,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		# Gate network
		self.gate = MoEGate(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		# Shared experts (always active)
		if config.n_shared_experts > 0:
			shared_intermediate_size = config.moe_intermediate_size * config.n_shared_experts
			self.shared_experts = DiTMLP(
				config=config,
				intermediate_size=shared_intermediate_size,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)

	def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
		"""
		Forward pass.

		Args:
			hidden_states: Input tensor [batch_size, seq_len, hidden_size]

		Returns:
			Output tensor [batch_size, seq_len, hidden_size]
		"""
		identity = hidden_states

		# Route to selected experts
		y, router_logits = self._moe_call(
			gate_layer=self.gate,
			expert_layer=self.experts,
			hidden_state=hidden_states,
			output_metrics=False,
			validate_inputs=False,
		)

		# Add shared expert output
		if self.config.n_shared_experts > 0:
			y = y + self.shared_experts(identity)

		return y


class DiTMoEBlock(nn.Module):
	"""DiT-MoE transformer block with MoE replacing the standard MLP."""

	def __init__(
		self,
		config: DiTMoEConfig,
		layer_idx: int,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.layer_idx = layer_idx
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision

		# Attention
		self.attn = FlexibleAttentionModule(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		# MLP or MoE
		is_moe_layer = (layer_idx >= config.first_k_dense_replace) and \
		               (layer_idx % config.moe_layer_freq == 0)

		if is_moe_layer:
			self.mlp = DiTMoE(
				config=config,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
		else:
			# Dense MLP for first_k_dense_replace layers
			self.mlp = DiTMLP(
				config=config,
				intermediate_size=config.intermediate_size,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)

		# Adaptive layer norm modulation
		linear_cls = partial(
			nn.Linear,
			dtype=dtype,
			param_dtype=param_dtype,
			use_bias=config.use_bias,
			rngs=rngs,
		)

		self.adaLN_modulation = nn.Sequential(
			nn.silu,
			linear_cls(config.hidden_size, 6 * config.hidden_size),
		)

		# Layer norms
		self.norm1 = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps, rngs=rngs)
		self.norm2 = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps, rngs=rngs)

	def __call__(
		self,
		hidden_states: jnp.ndarray,
		conditioning: jnp.ndarray,
		attention_mask: jnp.ndarray | None = None,
	) -> jnp.ndarray:
		"""Forward pass."""
		# Adaptive layer norm parameters
		shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
			self.adaLN_modulation(conditioning).chunk(6, axis=-1)

		# Attention block with adaptive LN
		normed = modulate(self.norm1(hidden_states), shift_msa[:, None, :], scale_msa[:, None, :])
		attn_output = self.attn(
			normed,
			attention_mask=attention_mask,
			causal_mask=None,
		)
		hidden_states = hidden_states + gate_msa[:, None, :] * attn_output

		# MLP/MoE block with adaptive LN
		normed = modulate(self.norm2(hidden_states), shift_mlp[:, None, :], scale_mlp[:, None, :])
		mlp_output = self.mlp(normed)
		hidden_states = hidden_states + gate_mlp[:, None, :] * mlp_output

		return hidden_states


class FinalLayer(nn.Module):
	"""Final layer that unpatchifies and outputs the denoised image/velocity."""

	def __init__(
		self,
		config: DiTMoEConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		*,
		rngs: nn.Rngs,
	):
		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype

		self.norm_final = nn.LayerNorm(
			config.hidden_size,
			epsilon=config.layer_norm_eps,
			rngs=rngs,
		)

		self.linear = nn.Linear(
			config.hidden_size,
			config.patch_size * config.patch_size * config.out_channels,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

		self.adaLN_modulation = nn.Sequential(
			nn.silu,
			nn.Linear(
				config.hidden_size,
				2 * config.hidden_size,
				dtype=dtype,
				param_dtype=param_dtype,
				rngs=rngs,
			),
		)

	def __call__(self, x: jnp.ndarray, conditioning: jnp.ndarray) -> jnp.ndarray:
		"""Forward pass."""
		shift, scale = self.adaLN_modulation(conditioning).chunk(2, axis=-1)
		x = modulate(self.norm_final(x), shift[:, None, :], scale[:, None, :])
		x = self.linear(x)
		return x


@register_module("dit_moe", TaskType.BASE_MODEL)
class DiTMoEModel(EasyDeLBaseModule):
	"""
	DiT-MoE (Mixture of Experts Diffusion Transformer) base model.

	This model extends DiT with sparse MoE layers following DeepSeek V2's architecture.
	"""

	def __init__(
		self,
		config: DiTMoEConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__(config=config, dtype=dtype, param_dtype=param_dtype, rngs=rngs)

		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype
		self.precision = precision

		# Patch embedding
		self.patch_embed = PatchEmbed(
			image_size=config.image_size,
			patch_size=config.patch_size,
			in_channels=config.in_channels,
			hidden_size=config.hidden_size,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

		# Positional embedding
		grid_size = config.image_size // config.patch_size
		pos_embed = get_2d_sincos_pos_embed(config.hidden_size, grid_size)
		self.pos_embed = nn.Variable(pos_embed.astype(param_dtype))

		# Timestep embedding
		self.time_embed = TimestepEmbedding(
			hidden_size=config.hidden_size,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

		# Label embedding
		if config.use_conditioning:
			self.label_embed = LabelEmbedding(
				num_classes=config.num_classes,
				hidden_size=config.hidden_size,
				dropout_prob=config.class_dropout_prob,
				dtype=dtype,
				param_dtype=param_dtype,
				rngs=rngs,
			)

		# Transformer blocks with MoE
		self.blocks = [
			DiTMoEBlock(
				config=config,
				layer_idx=i,
				dtype=dtype,
				param_dtype=param_dtype,
				precision=precision,
				rngs=rngs,
			)
			for i in range(config.num_hidden_layers)
		]

	def __call__(
		self,
		pixel_values: jnp.ndarray,
		timesteps: jnp.ndarray,
		labels: jnp.ndarray | None = None,
		attention_mask: jnp.ndarray | None = None,
		return_dict: bool = True,
	) -> BaseModelOutput | tuple:
		"""Forward pass."""
		# Embed patches
		hidden_states = self.patch_embed(pixel_values)

		# Add positional embedding
		hidden_states = hidden_states + self.pos_embed.value

		# Embed timesteps
		time_emb = self.time_embed(timesteps)

		# Embed labels and combine with timestep embedding
		if self.config.use_conditioning and labels is not None:
			label_emb = self.label_embed(labels)
			conditioning = time_emb + label_emb
		else:
			conditioning = time_emb

		# Apply transformer blocks
		for block in self.blocks:
			if self.config.gradient_checkpointing != "nothing_saveable":
				hidden_states = auto_remat(
					block,
					policy=self.config.gradient_checkpointing,
				)(hidden_states, conditioning, attention_mask)
			else:
				hidden_states = block(hidden_states, conditioning, attention_mask)

		if return_dict:
			return BaseModelOutput(last_hidden_state=hidden_states)
		return (hidden_states,)


@register_module("dit_moe", TaskType.IMAGE_DIFFUSION)
class DiTMoEForImageDiffusion(EasyDeLBaseModule):
	"""
	DiT-MoE model for image diffusion with rectified flow.

	This model adds the final unpatchification layer for diffusion training.
	"""

	def __init__(
		self,
		config: DiTMoEConfig,
		dtype: jnp.dtype = jnp.float32,
		param_dtype: jnp.dtype = jnp.float32,
		precision: jax.lax.PrecisionLike = None,
		*,
		rngs: nn.Rngs,
	):
		super().__init__(config=config, dtype=dtype, param_dtype=param_dtype, rngs=rngs)

		self.config = config
		self.dtype = dtype
		self.param_dtype = param_dtype

		# Base DiT-MoE model
		self.model = DiTMoEModel(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			precision=precision,
			rngs=rngs,
		)

		# Final layer
		self.final_layer = FinalLayer(
			config=config,
			dtype=dtype,
			param_dtype=param_dtype,
			rngs=rngs,
		)

	def unpatchify(self, x: jnp.ndarray) -> jnp.ndarray:
		"""Convert patch embeddings back to image format."""
		B = x.shape[0]
		p = self.config.patch_size
		h = w = self.config.image_size // p
		c = self.config.out_channels

		x = x.reshape(B, h, w, p, p, c)
		x = jnp.einsum("bhwpqc->bhpwqc", x)
		x = x.reshape(B, h * p, w * p, c)
		return x

	def __call__(
		self,
		pixel_values: jnp.ndarray,
		timesteps: jnp.ndarray,
		labels: jnp.ndarray | None = None,
		attention_mask: jnp.ndarray | None = None,
		return_dict: bool = True,
	) -> BaseModelOutput | tuple:
		"""Forward pass."""
		# Get hidden states from base model
		outputs = self.model(
			pixel_values=pixel_values,
			timesteps=timesteps,
			labels=labels,
			attention_mask=attention_mask,
			return_dict=True,
		)

		hidden_states = outputs.last_hidden_state

		# Get conditioning for final layer
		time_emb = self.model.time_embed(timesteps)
		if self.config.use_conditioning and labels is not None:
			label_emb = self.model.label_embed(labels)
			conditioning = time_emb + label_emb
		else:
			conditioning = time_emb

		# Apply final layer
		patch_outputs = self.final_layer(hidden_states, conditioning)

		# Unpatchify to image space
		predictions = self.unpatchify(patch_outputs)

		if return_dict:
			return BaseModelOutput(last_hidden_state=predictions)
		return (predictions,)
