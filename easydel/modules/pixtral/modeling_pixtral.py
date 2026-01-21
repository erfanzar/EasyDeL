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


import functools

import jax.lax
from eformer import common_types
from eformer.escale import apply_logical_sharding
from ejkernel.types import MaskInfo
from flax import nnx as nn
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import AttentionLayerOutput, BaseModelOutput, DecoderLayerOutput
from easydel.infra.utils import ACT2FN, auto_remat, block_wise_ffn
from easydel.layers.attention import AttentionModule, FlexibleAttentionModule
from easydel.layers.components import ColumnParallelLinear, RMSNorm, RowParallelLinear

from .pixtral_configuration import PixtralVisionConfig


def position_ids_in_meshgrid(patch_embeds_list, max_width):
    """Generates position IDs based on a meshgrid for a list of patch embeddings.

    Args:
        patch_embeds_list (list[Array]): A list of patch embeddings, where each element
            has shape (..., height, width).
        max_width (int): The maximum width across all patches, used for calculating the linear index.

    Returns:
        Array: A 1D array of position IDs corresponding to the flattened patches.
    """
    positions = []
    for patch in patch_embeds_list:
        height, width = patch.shape[-2:]
        mesh = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
        h_grid, v_grid = jnp.stack(mesh, axis=-1).reshape(-1, 2).T
        ids = h_grid * max_width + v_grid
        positions.append(ids)
    return jnp.concatenate(positions)


def generate_block_attention_mask(patch_embeds_list, tensor):
    """Generates a block-diagonal attention mask for multi-image processing.

    This mask ensures that attention is only computed within each image's patches,
    preventing cross-image attention.

    Args:
        patch_embeds_list (list[int]): A list containing the number of patches for each image.
        tensor (Array): The input tensor (e.g., hidden states) with shape
            (batch_size, sequence_length, ...).

    Returns:
        Array: A block-diagonal attention mask of shape
            (batch_size, 1, sequence_length, sequence_length).
            The mask contains 0.0 for allowed attention positions and a large negative number
            (minimum float value) for masked positions.
    """
    dtype = tensor.dtype
    seq_len = tensor.shape[1]
    d_min = jnp.finfo(dtype).min
    patch_lengths = jnp.asarray(patch_embeds_list, dtype=jnp.int32)
    block_end_idx = jnp.cumsum(patch_lengths)

    positions = jnp.arange(seq_len, dtype=jnp.int32)
    block_ids = jnp.sum(positions[:, None] >= block_end_idx[None, :], axis=-1)
    same_block = block_ids[:, None] == block_ids[None, :]

    block_mask = jnp.where(same_block, jnp.array(0, dtype=dtype), d_min)
    block_mask = jnp.expand_dims(block_mask, axis=(0, 1))
    block_mask = jnp.broadcast_to(block_mask, (tensor.shape[0], 1, seq_len, seq_len))
    return block_mask


def compute_frequencies(dim: int, max_patches_per_side: int, theta: float = 10000.0):
    """
    Computes frequencies with a fixed max length for RoPE.

    Args:
        dim: Embedding dimension.
        max_patches_per_side: Maximum number of patches per side of the image.
        theta: Scaling factor for frequencies.

    Returns:
        inv_freq: Computed frequencies of shape (max_patches_per_side**2, dim).
    """
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2).astype(jnp.float32) / dim))

    h = jnp.arange(max_patches_per_side)
    w = jnp.arange(max_patches_per_side)

    freqs_h = jnp.outer(h, freqs[::2])
    freqs_w = jnp.outer(w, freqs[1::2])

    inv_freq = jnp.concatenate(
        [
            jnp.tile(freqs_h[:, None, :], (1, max_patches_per_side, 1)),
            jnp.tile(freqs_w[None, :, :], (max_patches_per_side, 1, 1)),
        ],
        axis=-1,
    ).reshape(-1, dim // 2)
    # we reshape to only index on the position indexes, not tuple of indexes

    inv_freq = jnp.concatenate((inv_freq, inv_freq), axis=-1)
    return inv_freq


# Adapted from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


# Adapted from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=0):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`jnp.ndarray`): The query tensor.
        k (`jnp.ndarray`): The key tensor.
        cos (`jnp.ndarray`): The cosine part of the rotary embedding.
        sin (`jnp.ndarray`): The sine part of the rotary embedding.
        position_ids (`jnp.ndarray`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos and sin
            so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos and sin have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos and sin broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(jnp.ndarray)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # Pixtral uses `[batch, seq_len, heads, head_dim]` layout for Q/K. Accept RoPE
    # inputs as either `[seq_len, head_dim]` or `[batch, seq_len, head_dim]` and
    # reshape to broadcast across `heads`.
    if cos.ndim == 2:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    elif cos.ndim == 3:
        cos = cos[:, :, None, :]
        sin = sin[:, :, None, :]
    else:
        cos = jnp.expand_dims(cos, axis=unsqueeze_dim)
        sin = jnp.expand_dims(sin, axis=unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class PixtralMLP(nn.Module):
    """Multi-Layer Perceptron module for Pixtral vision models.

    Implements the feedforward network with a Gated Linear Unit (GLU) structure
    using SiLU activation for effective visual representation learning.
    """

    def __init__(
        self,
        config: PixtralVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Pixtral MLP block.

        Args:
            config (PixtralVisionConfig): Model configuration with MLP parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        column_parallel_linear = functools.partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        row_parallel_linear = functools.partial(
            RowParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.gate_proj = column_parallel_linear(
            config.hidden_size,
            config.intermediate_size,
            rngs=rngs,
        )
        self.down_proj = row_parallel_linear(
            config.intermediate_size,
            config.hidden_size,
            rngs=rngs,
        )
        self.up_proj = column_parallel_linear(
            config.hidden_size,
            config.intermediate_size,
            rngs=rngs,
        )
        self.act_fn = ACT2FN[self.config.hidden_act]

    def __call__(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        """Apply SiLU-gated feedforward transformation.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]

        Returns:
            Transformed hidden states [batch, seq_len, hidden_dim]
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        gate = checkpoint_name(self.act_fn(self.gate_proj(hidden_states)), "mlp_gate")
        up = checkpoint_name(self.up_proj(hidden_states), "mlp_up")
        hidden_states = checkpoint_name(self.down_proj(gate * up), "mlp_down")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return checkpoint_name(hidden_states, "mlp_output")


class PixtralAttention(AttentionModule):
    """Multi-head attention layer with 2D RoPE embeddings for Pixtral vision models.

    Implements vision-specific attention with:
    - 2D Rotary Position Embeddings for spatial awareness
    - Block-diagonal attention for multi-image batches
    - Bidirectional attention within image patches
    """

    def __init__(
        self,
        config: PixtralVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Pixtral attention layer with 2D RoPE support.

        Args:
            config (PixtralVisionConfig): Model configuration with attention parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(config=config)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.hidden_size = config.hidden_size
        self.head_dim = self.config.head_dim

        self.num_key_value_groups = self.config.num_attention_heads // self.config.num_attention_heads

        if self.num_key_value_groups == 1:
            assert self.config.num_attention_heads == self.config.num_attention_heads

        column_parallel_linear = functools.partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
        )
        row_parallel_linear = functools.partial(
            RowParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
        )
        self.q_proj = column_parallel_linear(config.hidden_size, config.num_attention_heads * self.head_dim, rngs=rngs)
        self.k_proj = column_parallel_linear(config.hidden_size, config.num_attention_heads * self.head_dim, rngs=rngs)
        self.v_proj = column_parallel_linear(config.hidden_size, config.num_attention_heads * self.head_dim, rngs=rngs)
        self.o_proj = row_parallel_linear(config.num_attention_heads * self.head_dim, config.hidden_size, rngs=rngs)

        self.attention_performer = FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=config.attention_dropout,
            requires_cache=False,  # Vision encoder doesn't need KV cache
        )

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo,
        position_ids: Int[Array, "batch seq_len"],
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ) -> AttentionLayerOutput:
        """Apply multi-head self-attention with 2D RoPE.

        Args:
            hidden_states (Array): Input tensor of shape (batch_size, num_patches, hidden_dim).
            mask_info (MaskInfo): Attention mask information for block-diagonal masking.
            position_ids (Array): Position indices for patches, shape (batch_size, num_patches).
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            frequencies (Array | None, optional): Precomputed 2D RoPE frequencies. Defaults to None.

        Returns:
            AttentionLayerOutput: Contains attention_output, optional attention_weight, and cache_view.
        """
        batch_size, sequence_length = hidden_states.shape[:2]
        query_states, key_states, value_states = (
            checkpoint_name(self.q_proj(hidden_states), "attn_query"),
            checkpoint_name(self.k_proj(hidden_states), "attn_key"),
            checkpoint_name(self.v_proj(hidden_states), "attn_value"),
        )

        query_states = query_states.reshape(
            batch_size,
            sequence_length,
            self.config.num_attention_heads,
            self.head_dim,
        )
        key_states = key_states.reshape(
            batch_size,
            sequence_length,
            self.config.num_attention_heads,
            self.head_dim,
        )
        value_states = value_states.reshape(
            batch_size,
            sequence_length,
            self.config.num_attention_heads,
            self.head_dim,
        )

        query_states, key_states = apply_rotary_pos_emb(
            q=query_states,
            k=key_states,
            cos=jnp.cos(frequencies),
            sin=jnp.sin(frequencies),
            position_ids=position_ids,
            unsqueeze_dim=0,
        )

        (
            key_states,
            value_states,
            mask_info,
            init_attention_bias,
            cache_view,
            _cache_metadata,
        ) = self.concatenate(
            query=query_states,
            key=key_states,
            value=value_states,
            mask_info=mask_info,
            cache_view=None,
            cache_metadata=None,
        )

        attentions = self.attention_performer.forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            mode=common_types.MODE_TRAIN,
            bias=None,
            cache_metadata=None,
            init_bias=init_attention_bias,
            mask_info=mask_info,
            causal=True,
        )

        attn_output = self.shard_attention_prod(self._merge_heads(attentions.attention_outputs))
        attn_output = checkpoint_name(self.o_proj(attn_output), "attn_output")

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class PixtralBlock(nn.Module):
    """Single transformer block for Pixtral vision models.

    Combines bidirectional attention with feedforward networks,
    using RMS normalization and residual connections in a pre-norm architecture.
    """

    def __init__(
        self,
        config: PixtralVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Pixtral transformer block.

        Args:
            config (PixtralVisionConfig): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        attn_block = PixtralAttention
        mlp_block = PixtralMLP

        attn_block, mlp_block = auto_remat(
            attn_block,
            mlp_block,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.attention = attn_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.feed_forward = mlp_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.attention_norm = RMSNorm(
            dim=config.hidden_size,
            eps=1e-5,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.ffn_norm = RMSNorm(
            dim=config.hidden_size,
            eps=1e-5,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo,
        position_ids: Int[Array, "batch seq_len"],
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ) -> DecoderLayerOutput:
        """Forward pass through the transformer block.

        Applies pre-normalization architecture: x + attn(norm(x)) followed by x + mlp(norm(x))

        Args:
            hidden_states (Array): Input tensor of shape (batch_size, num_patches, hidden_dim).
            mask_info (MaskInfo): Attention mask information for block-diagonal masking.
            position_ids (Array): Position indices for patches, shape (batch_size, num_patches).
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            frequencies (Array | None, optional): Precomputed 2D RoPE frequencies. Defaults to None.

        Returns:
            DecoderLayerOutput: Contains hidden_states, attention_weight, and cache_view.
        """
        residual = hidden_states
        attention_output = self.attention(
            self.attention_norm(hidden_states),
            mask_info,
            position_ids,
            output_attentions,
            frequencies,
        )

        hidden_states = checkpoint_name(attention_output.attention_output + residual, "residual")
        ffd_inp = self.ffn_norm(hidden_states)
        if self.config.use_scan_mlp:
            feed_forward_hidden_states = block_wise_ffn(self.feed_forward, ffd_inp, self.config.scan_mlp_chunk_size)
        else:
            feed_forward_hidden_states = self.feed_forward(ffd_inp)

        hidden_states = checkpoint_name(hidden_states + feed_forward_hidden_states, "layer_output")
        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attention_output.attention_weight,
            cache_view=attention_output.cache_view,
        )


class PixtralTransformer(nn.Module):
    """Transformer stack for Pixtral vision models.

    Processes patch embeddings through multiple transformer blocks with
    2D RoPE and block-diagonal attention for variable-resolution images.
    """

    def __init__(
        self,
        config: PixtralVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Pixtral transformer stack.

        Args:
            config (PixtralVisionConfig): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.layers = [
            PixtralBlock(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"],
        position_embeddings: Array | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        """Forward pass through the Pixtral transformer stack.

        Processes patch embeddings through all transformer blocks with RoPE and
        block-diagonal attention.

        Args:
            inputs_embeds (Array): Input patch embeddings of shape (batch_size, num_patches, hidden_size).
            position_embeddings (Array | None, optional): Precomputed 2D RoPE frequencies. Defaults to None.
            attention_mask (Array | None, optional): Attention mask of shape
                (batch_size, 1, num_patches, num_patches). Defaults to None.
            mask_info (MaskInfo | None, optional): Structured mask information. Defaults to None.
            position_ids (Array | None, optional): Position indices for patches. Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention weights. Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return all hidden states. Defaults to None.

        Returns:
            BaseModelOutput: Contains last_hidden_state, optional hidden_states, optional attentions,
                and past_key_values (always None for vision encoder).
        """
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        if position_ids is None:
            position_ids = mask_info.q_position_ids

        hidden_states = inputs_embeds
        for _idx, block in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = block(
                hidden_states=hidden_states,
                mask_info=mask_info,
                position_ids=position_ids,
                output_attentions=output_attentions,
                frequencies=position_embeddings,
            )
            hidden_states = layer_outputs.hidden_states

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            past_key_values=None,
        )


@register_module(TaskType.BASE_VISION, config=PixtralVisionConfig, model_type="pixtral")
class PixtralVisionModel(EasyDeLBaseModule):
    """Pixtral vision encoder model.

    Implements the complete Pixtral vision model with patch embedding via convolution,
    2D RoPE, and transformer stack. Supports variable-resolution images through
    block-diagonal attention masking.

    Attributes:
        config (PixtralVisionConfig): Configuration for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.
    """

    def __init__(
        self,
        config: PixtralVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Pixtral vision model.

        Args:
            config (PixtralVisionConfig): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.patch_conv = nn.Conv(
            in_features=config.num_channels,
            out_features=config.hidden_size,
            kernel_size=(config.patch_size,) * 2,
            strides=(config.patch_size,) * 2,
            use_bias=False,
            precision=precision,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.ln_pre = RMSNorm(
            config.hidden_size,
            eps=1e-5,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.transformer = PixtralTransformer(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    @functools.cached_property
    def frequencies(self) -> Float[Array, "max_patches head_dim"]:
        """Compute and cache 2D RoPE frequencies for patch positions.

        Returns:
            Array: Precomputed frequency embeddings of shape (max_patches_per_side^2, head_dim).
        """
        return compute_frequencies(
            dim=self.config.head_dim,
            theta=self.config.rope_theta,
            max_patches_per_side=self.config.image_size // self.config.patch_size,
        )

    def __call__(
        self,
        pixel_values: list[Array],
        output_hidden_states: bool | None = False,
        output_attentions: bool | None = None,
        *args,
        **kwargs,
    ) -> BaseModelOutput:
        """Forward pass through the Pixtral vision model.

        Processes variable-resolution images through patch embedding, 2D RoPE,
        and transformer with block-diagonal attention for multi-image batches.

        Args:
            pixel_values (list[Array]): List of input images, each of shape (channels, height, width).
                Images can have different resolutions.
            output_hidden_states (bool | None, optional): Whether to return hidden states from all
                layers. Defaults to False.
            output_attentions (bool | None, optional): Whether to return attention weights.
                Defaults to None.
            *args: Additional positional arguments (unused, for compatibility).
            **kwargs: Additional keyword arguments (unused, for compatibility).

        Returns:
            BaseModelOutput: Contains last_hidden_state of shape (1, total_patches, hidden_size),
                optional hidden_states, and optional attentions.
        """
        patch_embeds_list = [
            self.patch_conv(jnp.expand_dims(img, 0).astype(self.dtype).transpose(0, 2, 3, 1)) for img in pixel_values
        ]
        patch_embeds_list = [p.transpose(0, 3, 1, 2) for p in patch_embeds_list]
        # flatten to a single sequence
        patch_embeds = jnp.concatenate(
            [jnp.transpose(jnp.reshape(p, (p.shape[0], p.shape[1], -1)), (0, 2, 1)) for p in patch_embeds_list],
            axis=1,
        )
        patch_embeds = checkpoint_name(self.ln_pre(patch_embeds), "embeddings")

        # positional embeddings
        position_ids = position_ids_in_meshgrid(
            patch_embeds_list,
            max_width=self.config.image_size // self.config.patch_size,
        )

        attention_mask = generate_block_attention_mask(
            [p.shape[-2] * p.shape[-1] for p in patch_embeds_list], patch_embeds
        )
        transformer_output = self.transformer(
            inputs_embeds=patch_embeds,
            attention_mask=attention_mask,
            position_embeddings=self.frequencies[position_ids],
        )
        return BaseModelOutput(
            last_hidden_state=checkpoint_name(transformer_output.last_hidden_state, "model_output"),
            hidden_states=transformer_output.hidden_states,
            attentions=transformer_output.attentions,
        )

    def get_encoder(self):
        """Returns the encoder (this vision model acts as the encoder)."""
        return self

    def get_decoder(self):
        """Returns the decoder (not applicable for encoder-only vision model)."""
        raise NotImplementedError("This is an encoder-only model and does not have a decoder.")

    def get_lm_head(self):
        """Returns the language model head (not applicable for vision encoder)."""
        raise NotImplementedError("This vision model does not have a language model head.")

    def get_embedding(self):
        """Returns the patch embedding layer."""
        return self.patch_conv
