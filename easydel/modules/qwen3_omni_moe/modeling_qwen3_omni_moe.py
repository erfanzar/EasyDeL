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

"""Qwen3OmniMoe multimodal model implementation for EasyDeL.

This module implements the Qwen3OmniMoe architecture, a multimodal
model that processes text, vision (images/videos), and audio inputs.

Components:
- Audio encoder: Processes mel-spectrogram audio features
- Vision encoder: Processes images and videos (from Qwen3VL)
- Text decoder: MoE language model for text generation
"""

import typing
from functools import partial

import jax
import jax.numpy as jnp
from eformer import common_types
from eformer.escale import apply_logical_sharding
from eformer.pytree import auto_pytree
from ejkernel.types import MaskInfo
from flax import nnx as nn
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import (
    BaseModelOutput,
    DecoderLayerOutput,
    ModelOutput,
    MoeCausalLMOutput,
    VLMCausalLMOutput,
)
from easydel.infra.utils import ACT2FN, auto_remat
from easydel.layers.attention import FlexibleAttentionModule
from easydel.layers.attention_unified import UnifiedAttention
from easydel.layers.base_modules import BaseConditionalGenerationModule, BaseVisionLanguageModule
from easydel.layers.caching import (
    HybridCache,
    OperationsMetadata,
    RaggedPagesCache,
    RaggedPagesCacheView,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.layers.linear import ColumnParallelLinear, RowParallelLinear
from easydel.layers.moe import (
    BaseMoeModule,
    ColumnParallelMoELinear,
    MoeLoadBalancingStrategy,
    MoeRoutingStrategy,
    RowParallelMoELinear,
)
from easydel.layers.norms import RMSNorm

from .qwen3_omni_moe_configuration import (
    Qwen3OmniMoeAudioConfig,
    Qwen3OmniMoeAudioEncoderConfig,
    Qwen3OmniMoeCode2WavConfig,
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeTalkerCodePredictorConfig,
    Qwen3OmniMoeTalkerConfig,
    Qwen3OmniMoeTalkerTextConfig,
    Qwen3OmniMoeTextConfig,
    Qwen3OmniMoeThinkerConfig,
    Qwen3OmniMoeVisionConfig,
    Qwen3OmniMoeVisionEncoderConfig,
)


@auto_pytree
class Qwen3OmniMoeCausalLMOutputWithPast(ModelOutput):
    """Output class for Qwen3OmniMoe causal language model."""

    loss: Array | None = None
    logits: Array = None
    past_key_values: list[Array] | None = None
    hidden_states: tuple[Array] | None = None
    attentions: tuple[Array] | None = None
    rope_deltas: Array | None = None
    image_hidden_states: Float[Array, "batch seq_len hidden_dim"] | None = None
    audio_hidden_states: Float[Array, "batch seq_len hidden_dim"] | None = None
    router_logits: tuple[Array] | None = None


class Qwen3OmniMoeAudioMLP(nn.Module):
    """Feed-forward network for audio encoder."""

    def __init__(
        self,
        config: Qwen3OmniMoeAudioConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        self.fc1 = ColumnParallelLinear(
            config.d_model,
            config.encoder_ffn_dim,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.fc2 = RowParallelLinear(
            config.encoder_ffn_dim,
            config.d_model,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.act = ACT2FN[config.activation_function]

    def __call__(self, hidden_states: Array) -> Array:
        return self.fc2(self.act(self.fc1(hidden_states)))


class Qwen3OmniMoeAudioAttention(nn.Module):
    """Self-attention for audio encoder."""

    def __init__(
        self,
        config: Qwen3OmniMoeAudioConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        self.layer_idx = layer_idx
        self.d_model = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.d_model // self.num_heads

        self.q_proj = ColumnParallelLinear(
            self.d_model,
            self.d_model,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.k_proj = ColumnParallelLinear(
            self.d_model,
            self.d_model,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.v_proj = ColumnParallelLinear(
            self.d_model,
            self.d_model,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.out_proj = RowParallelLinear(
            self.d_model,
            self.d_model,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.attention = FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=config.attention_dropout,
            requires_cache=False,  # Audio encoder doesn't need KV cache
        )

    def __call__(
        self,
        hidden_states: Array,
        attention_mask: Array | None = None,
    ) -> Array:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        attn_output = self.attention.forward(
            query_states=q,
            key_states=k,
            value_states=v,
            mode=common_types.MODE_TRAIN,
            attention_mask=attention_mask,
            causal=False,
        ).attention_outputs

        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        return self.out_proj(attn_output)


class Qwen3OmniMoeAudioEncoderLayer(nn.Module):
    """Transformer layer for audio encoder.

    Matches HuggingFace structure with fc1/fc2 directly on the layer.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeAudioConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        self.self_attn = Qwen3OmniMoeAudioAttention(
            config=config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.self_attn_layer_norm = nn.LayerNorm(
            config.d_model,
            epsilon=1e-5,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.fc1 = ColumnParallelLinear(
            config.d_model,
            config.encoder_ffn_dim,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.fc2 = RowParallelLinear(
            config.encoder_ffn_dim,
            config.d_model,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.act = ACT2FN[config.activation_function]
        self.final_layer_norm = nn.LayerNorm(
            config.d_model,
            epsilon=1e-5,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: Array,
        attention_mask: Array | None = None,
    ) -> Array:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc2(self.act(self.fc1(hidden_states)))
        hidden_states = residual + hidden_states

        return hidden_states


def create_sinusoidal_positions(length: int, channels: int, max_timescale: int = 10000) -> Array:
    """Create sinusoidal position embeddings.

    Args:
        length: Maximum sequence length.
        channels: Embedding dimension (must be even).
        max_timescale: Maximum timescale for position encoding.

    Returns:
        Position embeddings of shape [length, channels].
    """
    if channels % 2 != 0:
        raise ValueError("SinusoidsPositionEmbedding needs even channels input")
    log_timescale_increment = jnp.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = jnp.exp(-log_timescale_increment * jnp.arange(channels // 2))
    scaled_time = jnp.arange(length)[:, None] * inv_timescales[None, :]
    return jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=1)


class Qwen3OmniMoeAudioEncoder(EasyDeLBaseModule):
    """Audio encoder for Qwen3OmniMoe.

    Processes mel-spectrogram audio inputs through:
    1. 2D convolutional feature extraction (conv2d1, conv2d2, conv2d3)
    2. Linear projection (conv_out)
    3. Sinusoidal positional encoding
    4. Transformer encoder layers
    5. Output projection (proj1 + act + proj2)

    Matches HuggingFace transformers structure.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeAudioConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
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
        self.config = config
        self.n_window = config.n_window
        self.n_window_infer = config.n_window_infer
        self.conv_chunksize = config.conv_chunksize

        self.conv2d1 = nn.Conv(
            in_features=1,
            out_features=config.downsample_hidden_size,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.conv2d2 = nn.Conv(
            in_features=config.downsample_hidden_size,
            out_features=config.downsample_hidden_size,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.conv2d3 = nn.Conv(
            in_features=config.downsample_hidden_size,
            out_features=config.downsample_hidden_size,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        mel_after_conv = (((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2
        self.conv_out = ColumnParallelLinear(
            config.downsample_hidden_size * mel_after_conv,
            config.d_model,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.max_source_positions = config.max_source_positions
        self.d_model = config.d_model

        self.layers = [
            Qwen3OmniMoeAudioEncoderLayer(
                config=config,
                layer_idx=i,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for i in range(config.encoder_layers)
        ]

        self.ln_post = nn.LayerNorm(
            config.d_model,
            epsilon=1e-5,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.proj1 = ColumnParallelLinear(
            config.d_model,
            config.d_model,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.act = ACT2FN[config.activation_function]
        self.proj2 = RowParallelLinear(
            config.d_model,
            config.output_dim,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        input_features: Float[Array, "batch mel_bins time"],
        feature_lens: Int[Array, "batch"] | None = None,
        attention_mask: Bool[Array, "batch time"] | None = None,
    ) -> BaseModelOutput:
        """Forward pass of audio encoder.

        Args:
            input_features: Mel-spectrogram features [batch, mel_bins, time].
            feature_lens: Length of each audio in the batch.
            attention_mask: Optional attention mask.

        Returns:
            BaseModelOutput with encoded audio features.
        """

        hidden_states = input_features.transpose(0, 2, 1)[..., None]

        hidden_states = nn.gelu(self.conv2d1(hidden_states))
        hidden_states = nn.gelu(self.conv2d2(hidden_states))
        hidden_states = nn.gelu(self.conv2d3(hidden_states))

        b, t, f, c = hidden_states.shape
        hidden_states = hidden_states.reshape(b, t, f * c)

        hidden_states = self.conv_out(hidden_states)

        seq_len = hidden_states.shape[1]
        pos_emb = create_sinusoidal_positions(seq_len, self.d_model).astype(hidden_states.dtype)
        hidden_states = hidden_states + pos_emb[None, :, :]

        # Note: For simplicity, we process without cu_seqlens chunking
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        hidden_states = self.ln_post(hidden_states)

        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)

        return BaseModelOutput(last_hidden_state=hidden_states)

    def get_encoder(self):
        return self

    def get_decoder(self):
        raise NotImplementedError("Audio encoder does not have a decoder.")

    def get_lm_head(self):
        raise NotImplementedError("Audio encoder does not have a language model head.")

    def get_embedding(self):
        return self.conv2d1


def rotate_half(x: Array) -> Array:
    """Rotate half the hidden dims for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision(
    q: Array,
    k: Array,
    cos: Array,
    sin: Array,
) -> tuple[Array, Array]:
    """Apply rotary positional embeddings to vision features."""
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.astype("f4"), k.astype("f4")
    cos, sin = jnp.expand_dims(cos, -2).astype("f4"), jnp.expand_dims(sin, -2).astype("f4")
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.astype(orig_q_dtype), k_embed.astype(orig_k_dtype)


def create_attention_mask(cu_seqlens: Array, seq_length: int, dtype: jnp.dtype) -> Array:
    """Create attention mask from cumulative sequence lengths."""
    attention_mask = jnp.full((1, seq_length, seq_length), jnp.finfo(dtype).min, dtype=dtype)
    mask_updates = jnp.zeros((1, seq_length, seq_length), dtype=dtype)

    for i in range(1, len(cu_seqlens)):
        start_idx = cu_seqlens[i - 1]
        end_idx = cu_seqlens[i]
        mask_updates = mask_updates.at[..., start_idx:end_idx, start_idx:end_idx].set(0)

    attention_mask = jax.lax.dynamic_update_slice(attention_mask, mask_updates, (0, 0, 0))
    return attention_mask


class Qwen3OmniMoeVisionPatchEmbed(nn.Module):
    """3D patch embedding for vision encoder."""

    def __init__(
        self,
        config: Qwen3OmniMoeVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        self.dtype = dtype
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.hidden_size = config.hidden_size

        kernel_size = (config.temporal_patch_size, config.patch_size, config.patch_size)
        self.proj = nn.Conv(
            in_features=config.in_channels,
            out_features=config.hidden_size,
            kernel_size=kernel_size,
            strides=kernel_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, hidden_states: Array) -> Array:
        hidden_states = jnp.transpose(
            hidden_states.reshape(
                -1,
                self.in_channels,
                self.temporal_patch_size,
                self.patch_size,
                self.patch_size,
            ),
            (0, 2, 3, 4, 1),
        )
        hidden_states = self.proj(hidden_states.astype(self.dtype))
        return hidden_states.reshape(-1, self.hidden_size)


class Qwen3OmniMoeVisionPatchMerger(nn.Module):
    """Spatial patch merger for vision encoder.

    Matches HuggingFace structure with ln_q and mlp list.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        use_postshuffle_norm: bool = False,
        *,
        rngs: nn.Rngs,
    ):
        self.dtype = dtype
        self.spatial_merge_size = config.spatial_merge_size
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm

        self.ln_q = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else config.hidden_size,
            epsilon=1e-6,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.mlp = [
            ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                use_bias=True,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            ),
            None,
            RowParallelLinear(
                self.hidden_size,
                config.out_hidden_size,
                use_bias=True,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            ),
        ]

    def __call__(self, x: Array) -> Array:
        x = self.ln_q(x.reshape(-1, self.hidden_size) if self.use_postshuffle_norm else x).reshape(-1, self.hidden_size)
        x = self.mlp[0](x)
        x = nn.gelu(x, approximate=False)
        x = self.mlp[2](x)
        return x


class Qwen3OmniMoeVisionMLP(nn.Module):
    """Feed-forward network for vision encoder."""

    def __init__(
        self,
        config: Qwen3OmniMoeVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        self.linear_fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.act = ACT2FN[config.hidden_act]
        self.linear_fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, x: Array) -> Array:
        return self.linear_fc2(self.act(self.linear_fc1(x)))


class Qwen3OmniMoeVisionAttention(nn.Module):
    """Self-attention for vision encoder."""

    def __init__(
        self,
        config: Qwen3OmniMoeVisionConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.layer_idx = layer_idx
        self.qkv = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size * 3,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.attention = FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=0.0,
            requires_cache=False,  # Vision encoder doesn't need KV cache
        )

    def __call__(
        self,
        hidden_states: Array,
        cu_seqlens: Array,
        rotary_pos_emb: Array,
    ) -> Array:
        seq_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states)
        q, k, v = map(
            lambda x: x.squeeze(0),
            jnp.split(qkv.reshape(seq_length, 3, self.num_heads, -1).transpose(1, 0, 2, 3), 3, 0),
        )

        cos = jnp.cos(rotary_pos_emb)
        sin = jnp.sin(rotary_pos_emb)
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        attn_output = self.attention.forward(
            query_states=q,
            key_states=k,
            value_states=v,
            mode=common_types.MODE_TRAIN,
            attention_mask=create_attention_mask(cu_seqlens, seq_length, q.dtype),
            causal=False,
        ).attention_outputs

        attn_output = attn_output.reshape(seq_length, -1)
        return checkpoint_name(self.proj(attn_output), "vision_attn_output")


class Qwen3OmniMoeVisionBlock(nn.Module):
    """Transformer block for vision encoder."""

    def __init__(
        self,
        config: Qwen3OmniMoeVisionConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        self.norm1 = nn.LayerNorm(config.hidden_size, epsilon=1e-6, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.norm2 = nn.LayerNorm(config.hidden_size, epsilon=1e-6, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.attn = Qwen3OmniMoeVisionAttention(
            config=config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.mlp = Qwen3OmniMoeVisionMLP(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: Array,
        cu_seqlens: Array,
        rotary_pos_emb: Array,
    ) -> Array:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens, rotary_pos_emb)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


def create_vision_rotary_embedding(dim: int, theta: float = 10000.0, max_seqlen: int = 8192) -> Array:
    """Create vision rotary position embeddings.

    Args:
        dim: Dimension of the embedding (head_dim // 2).
        theta: Base for computing frequencies.
        max_seqlen: Maximum sequence length.

    Returns:
        Frequencies of shape [max_seqlen, dim].
    """
    inv_freq = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    seq = jnp.arange(max_seqlen, dtype=jnp.float32)
    freqs = jnp.outer(seq, inv_freq)
    return freqs


class Qwen3OmniMoeVisionEncoder(EasyDeLBaseModule):
    """Vision encoder for Qwen3OmniMoe.

    Matches HuggingFace structure with merger_list, rotary_pos_emb.
    """

    config_class = Qwen3OmniMoeVisionConfig

    def __init__(
        self,
        config: Qwen3OmniMoeVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
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

        self.merger_list = [
            Qwen3OmniMoeVisionPatchMerger(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                use_postshuffle_norm=True,
                rngs=rngs,
            )
            for _ in range(len(config.deepstack_visual_indexes))
        ]

        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen3OmniMoeVisionPatchEmbed(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.pos_embed = nn.Embed(
            num_embeddings=config.num_position_embeddings,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        head_dim = config.hidden_size // config.num_heads
        self._head_dim_ro = head_dim // 2

        self._rotary_dim = head_dim // 2

        self.blocks = [
            Qwen3OmniMoeVisionBlock(
                config=config,
                layer_idx=idx,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for idx in range(config.depth)
        ]

        self.merger = Qwen3OmniMoeVisionPatchMerger(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            use_postshuffle_norm=False,
            rngs=rngs,
        )

        self.deepstack_visual_indexes = config.deepstack_visual_indexes

    def rot_pos_emb(self, grid_thw: Array, max_grid_size: int) -> Array:
        """Compute rotary position embeddings for vision features."""
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = jnp.arange(h)[:, None]
            hpos_ids = jnp.broadcast_to(hpos_ids, (h, w))
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = jnp.transpose(hpos_ids, (0, 2, 1, 3)).flatten()

            wpos_ids = jnp.arange(w)[None, :]
            wpos_ids = jnp.broadcast_to(wpos_ids, (h, w))
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = jnp.transpose(wpos_ids, (0, 2, 1, 3)).flatten()

            stacked = jnp.stack([hpos_ids, wpos_ids], axis=-1)
            pos_ids.append(jnp.repeat(stacked, t, axis=1))

        pos_ids = jnp.concatenate(pos_ids, axis=0)
        rotary_pos_emb_full = jnp.outer(
            jnp.arange(0, max_grid_size, dtype="f4"),
            1.0 / (10000 ** (jnp.arange(0, self._head_dim_ro, 2, dtype="f4") / self._head_dim_ro)),
        )
        rotary_pos_emb = jnp.take(rotary_pos_emb_full, pos_ids, axis=0)
        return rotary_pos_emb.reshape(pos_ids.shape[0], -1)

    def __call__(
        self,
        hidden_states: Array,
        grid_thw: Array,
        max_grid_size: int,
    ) -> Array:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw, max_grid_size)

        grid_lens = grid_thw[:, 1] * grid_thw[:, 2]
        repeated = jnp.repeat(grid_lens, grid_thw[:, 0])
        cu_seqlens = jnp.cumsum(repeated, dtype="i4")
        cu_seqlens = jnp.pad(cu_seqlens, (1, 0), constant_values=0)

        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, rotary_pos_emb)

        return self.merger(hidden_states)

    def get_encoder(self):
        return self

    def get_decoder(self):
        raise NotImplementedError("Vision model does not have a decoder.")

    def get_lm_head(self):
        raise NotImplementedError("Vision model does not have a language model head.")

    def get_embedding(self):
        return self.patch_embed


class Qwen3OmniMoeTextMLP(nn.Module):
    """Dense MLP for non-MoE layers."""

    def __init__(
        self,
        config: Qwen3OmniMoeTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        column_linear = partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        row_linear = partial(
            RowParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )

        self.gate_proj = column_linear(config.hidden_size, config.intermediate_size, rngs=rngs)
        self.up_proj = column_linear(config.hidden_size, config.intermediate_size, rngs=rngs)
        self.down_proj = row_linear(config.intermediate_size, config.hidden_size, rngs=rngs)
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(self, hidden_states: Array) -> Array:
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        gate = checkpoint_name(self.act_fn(self.gate_proj(hidden_states)), "mlp_gate")
        up = checkpoint_name(self.up_proj(hidden_states), "mlp_up")
        hidden_states = checkpoint_name(self.down_proj(gate * up), "mlp_down")
        return checkpoint_name(hidden_states, "mlp_output")


class Qwen3OmniMoeMLPStack(nn.Module):
    """MoE MLP stack."""

    reform_param: typing.ClassVar = {
        "gate_up_proj$": {
            "splits": [
                {"name": "gate_proj.kernel", "spliter": lambda x: x[..., : x.shape[-1] // 2]},
                {"name": "up_proj.kernel", "spliter": lambda x: x[..., x.shape[-1] // 2 :]},
            ],
            "inverse_spliter": lambda torch, gate, up: torch.stack((gate, up), dim=-1).flatten(-2),
        },
        "down_proj$": {
            "splits": [
                {"name": "down_proj.kernel", "spliter": lambda x: x},
            ],
            "inverse_spliter": lambda x: x,
        },
    }

    def __init__(
        self,
        config: Qwen3OmniMoeTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        self.gate_proj = ColumnParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.hidden_size,
            out_features=config.moe_intermediate_size,
            rngs=rngs,
            kernel_init=nn.initializers.normal(),
            use_bias=False,
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.down_proj = RowParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.moe_intermediate_size,
            out_features=config.hidden_size,
            rngs=rngs,
            use_bias=False,
            kernel_init=nn.initializers.normal(),
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.up_proj = ColumnParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.hidden_size,
            out_features=config.moe_intermediate_size,
            rngs=rngs,
            use_bias=False,
            kernel_init=nn.initializers.normal(),
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(
        self,
        hidden_states: Array,
        group_sizes: Array,
        sorted_experts: Array | None = None,
    ) -> Array:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states, group_sizes, sorted_experts))
            * self.up_proj(hidden_states, group_sizes, sorted_experts),
            group_sizes,
            sorted_experts,
        )


class Qwen3OmniMoeTextSparseBlock(BaseMoeModule):
    """Sparse MoE block for text decoder."""

    def __init__(
        self,
        config: Qwen3OmniMoeTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(
            config=config,
            n_routed_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            lbl_coef=None,
            rzl_coef=None,
            routing_strategy=MoeRoutingStrategy.TOP_K if config.norm_topk_prob else MoeRoutingStrategy.TOP_K_NDIV,
            load_balancing_strategy=MoeLoadBalancingStrategy.STANDARD,
        )
        self.config = config
        self.gate = ColumnParallelLinear(
            config.hidden_size,
            config.num_experts,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=nn.initializers.normal(config.initializer_range),
        )
        self.experts = Qwen3OmniMoeMLPStack(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, hidden_states: Array) -> tuple[Array, Array]:
        out, router_logits = self.moe_call(
            hidden_state=hidden_states,
            gate_layer=self.gate,
            expert_layer=self.experts,
            wi_kernel=self.experts.gate_proj.kernel.value,
            wu_kernel=self.experts.up_proj.kernel.value,
            wd_kernel=self.experts.down_proj.kernel.value,
            act_fn=self.experts.act_fn,
        )
        return checkpoint_name(out, "moe_expert_output"), checkpoint_name(router_logits, "moe_router_logits")


class Qwen3OmniMoeTextAttention(UnifiedAttention):
    """Causal self-attention for text decoder."""

    def __init__(
        self,
        config: Qwen3OmniMoeTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        sliding_window = None
        if config.use_sliding_window and config.sliding_window is not None and layer_idx >= config.max_window_layers:
            sliding_window = config.sliding_window

        super().__init__(
            config,
            dtype,
            param_dtype,
            precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="standard",
            causal=True,
            sliding_window=sliding_window,
            use_qk_norm=True,
        )

    def _postprocess_qkv(self, query_states, key_states, value_states):
        return self.query_normalization(query_states), self.key_normalization(key_states), value_states


class Qwen3OmniMoeTextDecoderLayer(nn.Module):
    """Decoder layer for text model."""

    def __init__(
        self,
        config: Qwen3OmniMoeTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        self.config = config
        self.layer_idx = layer_idx

        attn_block = Qwen3OmniMoeTextAttention
        mlp_block = Qwen3OmniMoeTextMLP
        moe_block = Qwen3OmniMoeTextSparseBlock

        attn_block, mlp_block, moe_block = auto_remat(
            attn_block,
            mlp_block,
            moe_block,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )

        self.self_attn = attn_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )

        self.is_moe = (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        )

        if self.is_moe:
            self.mlp = moe_block(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        else:
            self.mlp = mlp_block(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
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
        hidden_states: Array,
        mask_info: MaskInfo,
        position_ids: Array,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        frequencies: Array | None = None,
    ) -> DecoderLayerOutput:
        attn_outputs = self.self_attn(
            self.input_layernorm(hidden_states),
            mask_info,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            output_attentions,
            frequencies,
        )
        hidden_states = checkpoint_name(hidden_states + attn_outputs.attention_output, "residual")

        feed_forward_input = self.post_attention_layernorm(hidden_states)
        feed_forward_output = self.mlp(feed_forward_input)

        router_logits = None
        if self.is_moe:
            feed_forward_output, router_logits = feed_forward_output

        hidden_states = checkpoint_name(hidden_states + feed_forward_output, "residual")

        return DecoderLayerOutput(
            hidden_states=checkpoint_name(hidden_states, "layer_output"),
            attention_weight=attn_outputs.attention_weight,
            router_logits=router_logits if output_router_logits else None,
            cache_view=attn_outputs.cache_view,
        )


class Qwen3OmniMoeTalkerResizeMLP(nn.Module):
    """Resize MLP for projecting thinker hidden states to talker dimensions.

    Uses two linear layers (linear_fc1, linear_fc2) with activation in between.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        text_config = config.text_config

        self.linear_fc1 = ColumnParallelLinear(
            config.thinker_hidden_size,
            text_config.intermediate_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.linear_fc2 = ColumnParallelLinear(
            text_config.intermediate_size,
            text_config.hidden_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.act_fn = ACT2FN[text_config.hidden_act]

    def __call__(self, hidden_states: Array) -> Array:
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_states)))


class Qwen3OmniMoeTalkerTextMLP(nn.Module):
    """Dense MLP for Talker (used as shared expert)."""

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerTextConfig,
        intermediate_size: int | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        imz = intermediate_size or config.intermediate_size
        column_linear = partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        row_linear = partial(
            RowParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )

        self.gate_proj = column_linear(config.hidden_size, imz, rngs=rngs)
        self.up_proj = column_linear(config.hidden_size, imz, rngs=rngs)
        self.down_proj = row_linear(imz, config.hidden_size, rngs=rngs)
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(self, hidden_states: Array) -> Array:
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        gate = checkpoint_name(self.act_fn(self.gate_proj(hidden_states)), "mlp_gate")
        up = checkpoint_name(self.up_proj(hidden_states), "mlp_up")
        hidden_states = checkpoint_name(self.down_proj(gate * up), "mlp_down")
        return checkpoint_name(hidden_states, "mlp_output")


class Qwen3OmniMoeTalkerMLPStack(nn.Module):
    """MoE MLP stack for Talker."""

    reform_param: typing.ClassVar = {
        "gate_up_proj$": {
            "splits": [
                {"name": "gate_proj.kernel", "spliter": lambda x: x[..., : x.shape[-1] // 2]},
                {"name": "up_proj.kernel", "spliter": lambda x: x[..., x.shape[-1] // 2 :]},
            ],
            "inverse_spliter": lambda torch, gate, up: torch.stack((gate, up), dim=-1).flatten(-2),
        },
        "down_proj$": {
            "splits": [
                {"name": "down_proj.kernel", "spliter": lambda x: x},
            ],
            "inverse_spliter": lambda x: x,
        },
    }

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        self.gate_proj = ColumnParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.hidden_size,
            out_features=config.moe_intermediate_size,
            rngs=rngs,
            kernel_init=nn.initializers.normal(),
            use_bias=False,
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.down_proj = RowParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.moe_intermediate_size,
            out_features=config.hidden_size,
            rngs=rngs,
            use_bias=False,
            kernel_init=nn.initializers.normal(),
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.up_proj = ColumnParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.hidden_size,
            out_features=config.moe_intermediate_size,
            rngs=rngs,
            use_bias=False,
            kernel_init=nn.initializers.normal(),
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(
        self,
        hidden_states: Array,
        group_sizes: Array,
        sorted_experts: Array | None = None,
    ) -> Array:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states, group_sizes, sorted_experts))
            * self.up_proj(hidden_states, group_sizes, sorted_experts),
            group_sizes,
            sorted_experts,
        )


class Qwen3OmniMoeTalkerTextSparseMoeBlock(BaseMoeModule):
    """Sparse MoE block with SHARED EXPERT for Talker.

    This is the key new component that differs from Thinker's MoE:
    - Has a shared_expert (dense MLP) that processes ALL tokens
    - Has a shared_expert_gate (sigmoid gate) to weight the shared expert output
    - Output = routed_expert_output + sigmoid(gate(x)) * shared_expert(x)
    """

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(
            config=config,
            n_routed_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            lbl_coef=None,
            rzl_coef=None,
            routing_strategy=MoeRoutingStrategy.TOP_K if config.norm_topk_prob else MoeRoutingStrategy.TOP_K_NDIV,
            load_balancing_strategy=MoeLoadBalancingStrategy.STANDARD,
        )
        self.config = config

        self.gate = ColumnParallelLinear(
            config.hidden_size,
            config.num_experts,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=nn.initializers.normal(config.initializer_range),
        )

        self.experts = Qwen3OmniMoeTalkerMLPStack(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.shared_expert = Qwen3OmniMoeTalkerTextMLP(
            config=config,
            intermediate_size=config.shared_expert_intermediate_size,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.shared_expert_gate = ColumnParallelLinear(
            config.hidden_size,
            1,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=nn.initializers.normal(config.initializer_range),
        )

    def __call__(self, hidden_states: Array) -> tuple[Array, Array]:
        routed_output, router_logits = self.moe_call(
            hidden_state=hidden_states,
            gate_layer=self.gate,
            expert_layer=self.experts,
            wi_kernel=self.experts.gate_proj.kernel.value,
            wu_kernel=self.experts.up_proj.kernel.value,
            wd_kernel=self.experts.down_proj.kernel.value,
            act_fn=self.experts.act_fn,
        )

        shared_output = self.shared_expert(hidden_states)
        shared_gate = jax.nn.sigmoid(self.shared_expert_gate(hidden_states))

        output = routed_output + shared_gate * shared_output

        return checkpoint_name(output, "moe_expert_output"), checkpoint_name(router_logits, "moe_router_logits")


class Qwen3OmniMoeTalkerTextAttention(UnifiedAttention):
    """Causal self-attention for Talker text model."""

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        super().__init__(
            config,
            dtype,
            param_dtype,
            precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="standard",
            causal=True,
            sliding_window=config.sliding_window,
            use_qk_norm=True,
        )

    def _postprocess_qkv(self, query_states, key_states, value_states):
        return self.query_normalization(query_states), self.key_normalization(key_states), value_states


class Qwen3OmniMoeTalkerTextDecoderLayer(nn.Module):
    """Decoder layer for Talker text model with shared expert MoE."""

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        self.config = config
        self.layer_idx = layer_idx

        attn_block = Qwen3OmniMoeTalkerTextAttention
        mlp_block = Qwen3OmniMoeTalkerTextMLP
        moe_block = Qwen3OmniMoeTalkerTextSparseMoeBlock

        attn_block, mlp_block, moe_block = auto_remat(
            attn_block,
            mlp_block,
            moe_block,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )

        self.self_attn = attn_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )

        self.is_moe = (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        )

        if self.is_moe:
            self.mlp = moe_block(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        else:
            self.mlp = mlp_block(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
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
        hidden_states: Array,
        mask_info: MaskInfo,
        position_ids: Array,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        frequencies: Array | None = None,
    ) -> DecoderLayerOutput:
        attn_outputs = self.self_attn(
            self.input_layernorm(hidden_states),
            mask_info,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            output_attentions,
            frequencies,
        )
        hidden_states = checkpoint_name(hidden_states + attn_outputs.attention_output, "residual")

        feed_forward_input = self.post_attention_layernorm(hidden_states)
        feed_forward_output = self.mlp(feed_forward_input)

        router_logits = None
        if self.is_moe:
            feed_forward_output, router_logits = feed_forward_output

        hidden_states = checkpoint_name(hidden_states + feed_forward_output, "residual")

        return DecoderLayerOutput(
            hidden_states=checkpoint_name(hidden_states, "layer_output"),
            attention_weight=attn_outputs.attention_weight,
            router_logits=router_logits if output_router_logits else None,
            cache_view=attn_outputs.cache_view,
        )


class Qwen3OmniMoeTalkerCodePredictorMLP(nn.Module):
    """Dense MLP for Talker code predictor (non-MoE)."""

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerCodePredictorConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        column_linear = partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        row_linear = partial(
            RowParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )

        self.gate_proj = column_linear(config.hidden_size, config.intermediate_size, rngs=rngs)
        self.up_proj = column_linear(config.hidden_size, config.intermediate_size, rngs=rngs)
        self.down_proj = row_linear(config.intermediate_size, config.hidden_size, rngs=rngs)
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(self, hidden_states: Array) -> Array:
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        gate = checkpoint_name(self.act_fn(self.gate_proj(hidden_states)), "mlp_gate")
        up = checkpoint_name(self.up_proj(hidden_states), "mlp_up")
        hidden_states = checkpoint_name(self.down_proj(gate * up), "mlp_down")
        return checkpoint_name(hidden_states, "mlp_output")


class Qwen3OmniMoeTalkerCodePredictorAttention(UnifiedAttention):
    """Causal self-attention for Talker code predictor."""

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerCodePredictorConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        super().__init__(
            config,
            dtype,
            param_dtype,
            precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="standard",
            causal=True,
            sliding_window=config.sliding_window,
            use_qk_norm=True,
        )

    def _postprocess_qkv(self, query_states, key_states, value_states):
        return self.query_normalization(query_states), self.key_normalization(key_states), value_states


class Qwen3OmniMoeTalkerCodePredictorDecoderLayer(nn.Module):
    """Decoder layer for Talker code predictor (non-MoE)."""

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerCodePredictorConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        self.config = config
        self.layer_idx = layer_idx

        attn_block = Qwen3OmniMoeTalkerCodePredictorAttention
        mlp_block = Qwen3OmniMoeTalkerCodePredictorMLP

        attn_block, mlp_block = auto_remat(
            attn_block,
            mlp_block,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )

        self.self_attn = attn_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )
        self.mlp = mlp_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
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
        hidden_states: Array,
        mask_info: MaskInfo,
        position_ids: Array,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Array | None = None,
    ) -> DecoderLayerOutput:
        attn_outputs = self.self_attn(
            self.input_layernorm(hidden_states),
            mask_info,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            output_attentions,
            frequencies,
        )
        hidden_states = checkpoint_name(hidden_states + attn_outputs.attention_output, "residual")

        feed_forward_input = self.post_attention_layernorm(hidden_states)
        feed_forward_output = self.mlp(feed_forward_input)
        hidden_states = checkpoint_name(hidden_states + feed_forward_output, "residual")

        return DecoderLayerOutput(
            hidden_states=checkpoint_name(hidden_states, "layer_output"),
            attention_weight=attn_outputs.attention_weight,
            router_logits=None,
            cache_view=attn_outputs.cache_view,
        )


class Qwen3OmniMoeTalkerCodePredictorModel(EasyDeLBaseModule):
    """Talker code predictor model for acoustic token prediction.

    This is a lightweight transformer that predicts codec tokens
    from the talker text model's hidden states.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerCodePredictorConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
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

        self.codec_embedding = [
            nn.Embed(
                config.vocab_size,
                config.hidden_size,
                embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            for _ in range(config.num_code_groups - 1)
        ]

        self.layers = [
            Qwen3OmniMoeTalkerCodePredictorDecoderLayer(
                config=config,
                layer_idx=layer_idx,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for layer_idx in range(config.num_hidden_layers)
        ]

        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"],
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> BaseModelOutput:
        sequence_length = inputs_embeds.shape[1]
        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        if position_ids is None:
            position_ids = mask_info.q_position_ids

        hidden_states = inputs_embeds
        if mode is None:
            mode = (
                common_types.MODE_DECODE
                if sequence_length == 1 and past_key_values is not None
                else common_types.MODE_TRAIN
            )
        if past_key_values is None:
            past_key_values = TransformerCache.init_empty(len(self.layers))

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for idx, block in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = block(
                hidden_states=hidden_states,
                mask_info=mask_info,
                position_ids=position_ids,
                mode=mode,
                cache_view=past_key_values.views[idx],
                cache_metadata=cache_metadata,
                output_attentions=output_attentions,
                frequencies=self.frequencies,
            )
            hidden_states = layer_outputs.hidden_states

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            past_key_values[idx] = layer_outputs.cache_view

        hidden_states = self.norm(hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

    def get_encoder(self):
        raise NotImplementedError("This is a decoder-only model.")

    def get_decoder(self):
        return self

    def get_lm_head(self):
        raise NotImplementedError("Use Qwen3OmniMoeTalkerCodePredictorForConditionalGeneration for LM head.")

    def get_embedding(self):
        return self.codec_embedding[0] if self.codec_embedding else None


class Qwen3OmniMoeTalkerCodePredictorForConditionalGeneration(
    BaseConditionalGenerationModule[Qwen3OmniMoeTalkerCodePredictorModel, Qwen3OmniMoeTalkerCodePredictorConfig]
):
    """Talker code predictor with per-group LM heads."""

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerCodePredictorConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(
            config=config,
            base_model_class=Qwen3OmniMoeTalkerCodePredictorModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            create_lm_head=False,
            lm_head_name="lm_head",
        )

        self.lm_head = [
            ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
                kernel_init=jax.nn.initializers.normal(config.initializer_range),
            )
            for _ in range(config.num_code_groups - 1)
        ]

    def __call__(
        self,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"],
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> MoeCausalLMOutput:
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        logits = jnp.stack([head(outputs.last_hidden_state) for head in self.lm_head], axis=2)

        return MoeCausalLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            last_hidden_state=outputs.last_hidden_state,
        )

    def get_encoder(self):
        raise NotImplementedError("This is a decoder-only model.")

    def get_decoder(self):
        return self.model.get_decoder()

    def get_lm_head(self):
        return self.lm_head

    def get_embedding(self):
        return self.model.get_embedding()


class Qwen3OmniMoeTalkerModel(EasyDeLBaseModule):
    """Talker text model that processes codec embeddings."""

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
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

        self.codec_embedding = nn.Embed(
            config.vocab_size,
            config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.layers = [
            Qwen3OmniMoeTalkerTextDecoderLayer(
                config=config,
                layer_idx=layer_idx,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for layer_idx in range(config.num_hidden_layers)
        ]

        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_router_logits: bool = False,
    ) -> VLMCausalLMOutput:
        if inputs_embeds is None:
            inputs_embeds = self.codec_embedding(input_ids.astype("i4"))

        sequence_length = inputs_embeds.shape[1]
        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        if position_ids is None:
            position_ids = mask_info.q_position_ids

        hidden_states = inputs_embeds
        if mode is None:
            mode = (
                common_types.MODE_DECODE
                if sequence_length == 1 and past_key_values is not None
                else common_types.MODE_TRAIN
            )
        if past_key_values is None:
            past_key_values = TransformerCache.init_empty(len(self.layers))

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        for idx, block in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = block(
                hidden_states=hidden_states,
                mask_info=mask_info,
                position_ids=position_ids,
                mode=mode,
                cache_view=past_key_values.views[idx],
                cache_metadata=cache_metadata,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                frequencies=self.frequencies,
            )
            hidden_states = layer_outputs.hidden_states

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            past_key_values[idx] = layer_outputs.cache_view
            if output_router_logits and layer_outputs.router_logits is not None:
                all_router_logits += (layer_outputs.router_logits,)

        hidden_states = self.norm(hidden_states)

        return VLMCausalLMOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            past_key_values=past_key_values,
            router_logits=all_router_logits,
        )

    def get_encoder(self):
        raise NotImplementedError("This is a decoder-only model.")

    def get_decoder(self):
        return self

    def get_lm_head(self):
        raise NotImplementedError("Base model does not have a language model head.")

    def get_embedding(self):
        return self.codec_embedding


class Qwen3OmniMoeTalkerForConditionalGeneration(
    BaseConditionalGenerationModule[Qwen3OmniMoeTalkerModel, Qwen3OmniMoeTalkerConfig]
):
    """Full Talker model combining text model and code predictor.

    This model takes hidden states from the Thinker and generates
    acoustic tokens for speech synthesis.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        talker_model = Qwen3OmniMoeTalkerModel(
            config=config.text_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        super().__init__(
            config=config,
            base_model=talker_model,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            create_lm_head=False,
            lm_head_name="codec_head",
        )

        self.code_predictor = Qwen3OmniMoeTalkerCodePredictorForConditionalGeneration(
            config=config.code_predictor_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.text_projection = Qwen3OmniMoeTalkerResizeMLP(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.hidden_projection = Qwen3OmniMoeTalkerResizeMLP(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.codec_head = ColumnParallelLinear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        thinker_hidden_states: Float[Array, "batch seq_len thinker_hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_router_logits: bool = False,
        apply_codec_head: bool = True,
    ) -> MoeCausalLMOutput:
        if thinker_hidden_states is not None and inputs_embeds is None:
            inputs_embeds = self.text_projection(thinker_hidden_states)

        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
        )

        logits = None
        if apply_codec_head:
            logits = self.codec_head(outputs.last_hidden_state)

        return MoeCausalLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
            last_hidden_state=outputs.last_hidden_state,
            router_logits=outputs.router_logits,
        )

    def get_encoder(self):
        raise NotImplementedError("This is a decoder-only model.")

    def get_decoder(self):
        return self.model.get_decoder()

    def get_lm_head(self):
        return self.codec_head

    def get_embedding(self):
        return self.model.get_embedding()


class Qwen3OmniMoeCode2WavLayerScale(nn.Module):
    """Learnable per-channel scaling for residual branches.

    Helps stabilize training of deep networks.
    """

    def __init__(
        self,
        dim: int,
        init_value: float = 1e-4,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: nn.Rngs,
    ):
        self.scale = nn.Param(
            jnp.full((dim,), init_value, dtype=param_dtype),
        )
        self.dtype = dtype

    def __call__(self, x: Array) -> Array:
        return x * self.scale.value.astype(self.dtype)


class Qwen3OmniMoeCode2WavMLP(nn.Module):
    """MLP for Code2Wav transformer."""

    def __init__(
        self,
        config: Qwen3OmniMoeCode2WavConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        column_linear = partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            precision=precision,
            rngs=rngs,
        )
        row_linear = partial(
            RowParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            precision=precision,
            rngs=rngs,
        )

        self.gate_proj = column_linear(config.hidden_size, config.intermediate_size, rngs=rngs)
        self.up_proj = column_linear(config.hidden_size, config.intermediate_size, rngs=rngs)
        self.down_proj = row_linear(config.intermediate_size, config.hidden_size, rngs=rngs)
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(self, hidden_states: Array) -> Array:
        gate = self.act_fn(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


class Qwen3OmniMoeCode2WavAttention(nn.Module):
    """Sliding window attention for Code2Wav vocoder."""

    def __init__(
        self,
        config: Qwen3OmniMoeCode2WavConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.sliding_window = config.sliding_window

        self.q_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            use_bias=config.attention_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.k_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            use_bias=config.attention_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.v_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            use_bias=config.attention_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            use_bias=config.attention_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.attention = FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=config.attention_dropout,
            requires_cache=False,  # Code2Wav encoder doesn't need KV cache
        )

    def __call__(
        self,
        hidden_states: Array,
        attention_mask: Array | None = None,
        position_ids: Array | None = None,
    ) -> Array:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        if self.sliding_window is not None:
            causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
            window_mask = jnp.triu(jnp.ones((seq_len, seq_len), dtype=jnp.bool_), k=-self.sliding_window + 1)
            mask = causal_mask & window_mask
            if attention_mask is not None:
                mask = mask & attention_mask
            attention_mask = mask

        attn_output = self.attention.forward(
            query_states=q,
            key_states=k,
            value_states=v,
            mode=common_types.MODE_TRAIN,
            attention_mask=attention_mask,
            causal=True,
        ).attention_outputs

        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


class Qwen3OmniMoeCode2WavTransformerLayer(nn.Module):
    """Transformer layer with LayerScale for Code2Wav vocoder."""

    def __init__(
        self,
        config: Qwen3OmniMoeCode2WavConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        self.layer_idx = layer_idx

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.self_attn = Qwen3OmniMoeCode2WavAttention(
            config=config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.self_attn_layer_scale = Qwen3OmniMoeCode2WavLayerScale(
            config.hidden_size,
            init_value=config.layer_scale_initial_scale,
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
        self.mlp = Qwen3OmniMoeCode2WavMLP(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.mlp_layer_scale = Qwen3OmniMoeCode2WavLayerScale(
            config.hidden_size,
            init_value=config.layer_scale_initial_scale,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: Array,
        attention_mask: Array | None = None,
        position_ids: Array | None = None,
    ) -> Array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = residual + self.self_attn_layer_scale(attn_output)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + self.mlp_layer_scale(mlp_output)

        return hidden_states


class Qwen3OmniMoeCode2WavTransformerModel(nn.Module):
    """Transformer model for Code2Wav, containing layers and norm."""

    def __init__(
        self,
        config: Qwen3OmniMoeCode2WavConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config

        self.layers = [
            Qwen3OmniMoeCode2WavTransformerLayer(
                config=config,
                layer_idx=layer_idx,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for layer_idx in range(config.num_hidden_layers)
        ]

        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        inputs_embeds: Array,
        attention_mask: Array | None = None,
        position_ids: Array | None = None,
    ) -> BaseModelOutput:
        hidden_states = inputs_embeds

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids)

        hidden_states = self.norm(hidden_states)

        return BaseModelOutput(last_hidden_state=hidden_states)


class Qwen3OmniMoeCode2Wav(EasyDeLBaseModule):
    """Code2Wav vocoder: converts codec tokens to waveform.

    Uses transformer with sliding window attention and ConvNeXt
    upsampling blocks.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeCode2WavConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
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

        # Single embedding for all quantizers with offset-based indexing
        self.code_embedding = nn.Embed(
            config.codebook_size * config.num_quantizers,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        # Offset buffer for each quantizer
        self.code_offset = jnp.arange(config.num_quantizers).reshape(1, -1, 1) * config.codebook_size

        self.pre_transformer = Qwen3OmniMoeCode2WavTransformerModel(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        # Note: upsample and decoder blocks are not implemented here as they require
        # ConvNeXt blocks. The HuggingFace implementation uses these for audio synthesis.

    def __call__(
        self,
        codec_tokens: Int[Array, "batch num_quantizers seq_len"],
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
    ) -> BaseModelOutput:
        """Forward pass of Code2Wav vocoder.

        Args:
            codec_tokens: Codec tokens from talker [batch, num_quantizers, seq_len].
            attention_mask: Optional attention mask.

        Returns:
            BaseModelOutput with audio features (not full waveform - upsampling done externally).
        """
        # codes shape: [batch, num_quantizers, seq_len]
        codes_with_offset = codec_tokens + self.code_offset
        hidden_states = self.code_embedding(codes_with_offset.astype("i4")).mean(axis=1)

        # Pass through transformer
        outputs = self.pre_transformer(hidden_states, attention_mask)

        return outputs

    def get_encoder(self):
        raise NotImplementedError("Code2Wav is a decoder-only model.")

    def get_decoder(self):
        return self

    def get_lm_head(self):
        raise NotImplementedError("Code2Wav uses upsampling blocks, not an LM head.")

    def get_embedding(self):
        return self.code_embedding


def merge_multimodal_embeddings(
    input_ids: jax.Array,
    inputs_embeds: jax.Array,
    multimodal_embeddings: jax.Array,
    placeholder_token_id: int | list[int],
) -> jax.Array:
    """Overwrite inputs_embeds wherever input_ids matches placeholder tokens."""
    if isinstance(placeholder_token_id, list):
        placeholder_token_id = jnp.array(placeholder_token_id)
        is_multimodal = jnp.isin(input_ids, placeholder_token_id)
    else:
        is_multimodal = input_ids == placeholder_token_id

    dummy_row = jnp.zeros_like(multimodal_embeddings[0:1])
    flattened_padded = jnp.concatenate([dummy_row, multimodal_embeddings], axis=0)
    gather_indices = jnp.cumsum(is_multimodal)
    update_values = flattened_padded[gather_indices]
    condition = jnp.expand_dims(is_multimodal, axis=-1)
    return jnp.where(condition, update_values, inputs_embeds)


class Qwen3OmniMoeThinkerTextModel(EasyDeLBaseModule):
    """Text decoder model for Qwen3OmniMoe Thinker.

    Contains only the text processing components (embed_tokens, layers, norm).
    This matches HuggingFace's Qwen3OmniMoeThinkerTextModel structure where
    audio_tower and visual are at the top level of ForConditionalGeneration.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
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

        embed_block = auto_remat(
            nn.Embed,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )

        self.embed_tokens = embed_block(
            config.vocab_size,
            config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.layers = [
            Qwen3OmniMoeTextDecoderLayer(
                config=config,
                layer_idx=layer_idx,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for layer_idx in range(config.num_hidden_layers)
        ]

        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
    ) -> VLMCausalLMOutput:
        """Forward pass through text decoder only."""
        config = self.config
        output_router_logits = output_router_logits if output_router_logits is not None else config.output_router_logits
        output_hidden_states = output_hidden_states if output_hidden_states is not None else config.output_hidden_states
        output_attentions = output_attentions if output_attentions is not None else config.output_attentions

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        if position_ids is None:
            position_ids = mask_info.q_position_ids

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        hidden_states = inputs_embeds

        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states=hidden_states,
                mask_info=mask_info,
                position_ids=position_ids,
                output_attentions=output_attentions,
                mode=mode,
                cache_view=past_key_values.views[layer_idx] if past_key_values is not None else None,
                cache_metadata=cache_metadata,
            )
            hidden_states = layer_outputs.hidden_states

            if output_attentions:
                all_self_attentions += (layer_outputs.attentions,)
            if output_router_logits:
                all_router_logits += (layer_outputs.router_logits,)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return VLMCausalLMOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            router_logits=all_router_logits,
            past_key_values=past_key_values,
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_embedding(self):
        return self.embed_tokens


@register_module(TaskType.BASE_MODULE, config=Qwen3OmniMoeThinkerConfig, model_type="qwen3_omni_moe")
class Qwen3OmniMoeModel(EasyDeLBaseModule):
    """Base Qwen3OmniMoe Thinker model combining audio, vision, and text.

    Note: This class is kept for backward compatibility. The audio and visual
    components are nested under this model, while in HuggingFace they're at
    the ForConditionalGeneration level.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeThinkerConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
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

        self.audio = Qwen3OmniMoeAudioEncoder(
            config=config.audio_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.visual = Qwen3OmniMoeVisionEncoder(
            config=config.vision_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        text_config = config.text_config
        embed_block = auto_remat(
            nn.Embed,
            policy=text_config.gradient_checkpointing,
            save_names=text_config.gradient_checkpointing_targets,
            exclude_names=text_config.gradient_checkpointing_targets,
        )

        self.embed_tokens = embed_block(
            text_config.vocab_size,
            text_config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=text_config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.layers = [
            Qwen3OmniMoeTextDecoderLayer(
                config=text_config,
                layer_idx=layer_idx,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for layer_idx in range(text_config.num_hidden_layers)
        ]

        self.norm = RMSNorm(
            text_config.hidden_size,
            eps=text_config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def compute_embedding(
        self,
        input_ids: Int[Array, "batch seq_len"],
        *,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        input_features: Float[Array, "batch mel_bins time"] | None = None,
        audio_embeds: Array | None = None,
        pixel_values: Float[Array, "num_patches channels h w"] | None = None,
        image_grid_thw: Int[Array, "num_images 3"] | None = None,
        image_max_grid_size: int | None = None,
        image_embeds: Array | None = None,
        pixel_values_videos: Float[Array, "num_patches channels h w"] | None = None,
        video_grid_thw: Int[Array, "num_videos 3"] | None = None,
        video_max_grid_size: int | None = None,
        video_embeds: Array | None = None,
        **kwargs,
    ) -> Array:
        if inputs_embeds is None:
            inputs_embeds = checkpoint_name(super().compute_embedding(input_ids), "embeddings")

        if input_ids is None and (
            input_features is not None
            or audio_embeds is not None
            or pixel_values is not None
            or image_embeds is not None
            or pixel_values_videos is not None
            or video_embeds is not None
        ):
            raise ValueError("`input_ids` must be provided to merge multimodal embeddings.")

        if audio_embeds is None and input_features is not None:
            audio_outputs = self.audio(input_features)
            audio_embeds = audio_outputs.last_hidden_state

        if audio_embeds is not None:
            if audio_embeds.ndim == 3:
                audio_embeds = audio_embeds.reshape(-1, self.config.text_config.hidden_size)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                audio_embeds.astype(inputs_embeds.dtype),
                self.config.audio_token_id,
            )

        if image_embeds is None and pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("`image_grid_thw` must be provided when `pixel_values` is not None.")
            if image_max_grid_size is None:
                image_max_grid_size = int(jnp.max(image_grid_thw[:, 1:]).item())
            image_embeds = self.visual(pixel_values, image_grid_thw, image_max_grid_size)

        if image_embeds is not None:
            if image_embeds.ndim > 2:
                image_embeds = image_embeds.reshape(-1, image_embeds.shape[-1])
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                image_embeds.astype(inputs_embeds.dtype),
                self.config.image_token_id,
            )

        if video_embeds is None and pixel_values_videos is not None:
            if video_grid_thw is None:
                raise ValueError("`video_grid_thw` must be provided when `pixel_values_videos` is not None.")
            if video_max_grid_size is None:
                video_max_grid_size = int(jnp.max(video_grid_thw[:, 1:]).item())
            video_embeds = self.visual(pixel_values_videos, video_grid_thw, video_max_grid_size)

        if video_embeds is not None:
            if video_embeds.ndim > 2:
                video_embeds = video_embeds.reshape(-1, video_embeds.shape[-1])
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                video_embeds.astype(inputs_embeds.dtype),
                self.config.video_token_id,
            )

        return inputs_embeds

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        input_features: Float[Array, "batch mel_bins time"] | None = None,
        pixel_values: Float[Array, "num_patches channels h w"] | None = None,
        image_grid_thw: Int[Array, "num_images 3"] | None = None,
        image_max_grid_size: int | None = None,
        pixel_values_videos: Float[Array, "num_patches channels h w"] | None = None,
        video_grid_thw: Int[Array, "num_videos 3"] | None = None,
        video_max_grid_size: int | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
    ) -> VLMCausalLMOutput:
        text_config = self.config.text_config
        output_router_logits = (
            output_router_logits if output_router_logits is not None else text_config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else text_config.output_hidden_states
        )

        all_router_logits = () if output_router_logits else None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must provide either `input_ids` or `inputs_embeds`.")

        if (
            inputs_embeds is None
            or input_features is not None
            or pixel_values is not None
            or pixel_values_videos is not None
        ):
            if input_ids is None:
                raise ValueError("`input_ids` must be provided to compute or merge multimodal embeddings.")
            inputs_embeds = self.compute_embedding(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                input_features=input_features,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                image_max_grid_size=image_max_grid_size,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                video_max_grid_size=video_max_grid_size,
            )

        sequence_length = inputs_embeds.shape[1]
        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        if position_ids is None:
            position_ids = mask_info.q_position_ids

        hidden_states = inputs_embeds
        if mode is None:
            mode = (
                common_types.MODE_DECODE
                if sequence_length == 1 and past_key_values is not None
                else common_types.MODE_TRAIN
            )
        if past_key_values is None:
            past_key_values = TransformerCache.init_empty(len(self.layers))

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=text_config.partition_manager,
        )

        for idx, block in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = block(
                hidden_states=hidden_states,
                mask_info=mask_info,
                position_ids=position_ids,
                mode=mode,
                cache_view=past_key_values.views[idx],
                cache_metadata=cache_metadata,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                frequencies=self.frequencies,
            )
            hidden_states = layer_outputs.hidden_states

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            past_key_values[idx] = layer_outputs.cache_view
            if output_router_logits and layer_outputs.router_logits is not None:
                all_router_logits += (layer_outputs.router_logits,)

        hidden_states = checkpoint_name(self.norm(hidden_states), "model_output")

        return VLMCausalLMOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            past_key_values=past_key_values,
            router_logits=all_router_logits,
        )

    def get_encoder(self):
        raise NotImplementedError("This is a decoder-only model.")

    def get_decoder(self):
        return self

    def get_lm_head(self):
        raise NotImplementedError("Base model does not have a language model head.")

    def get_embedding(self):
        return self.embed_tokens


@register_module(
    TaskType.ANY_TO_ANY,
    config=Qwen3OmniMoeThinkerConfig,
    model_type="qwen3_omni_moe_thinker",
)
class Qwen3OmniMoeThinkerForConditionalGeneration(
    BaseVisionLanguageModule[Qwen3OmniMoeThinkerTextModel, Qwen3OmniMoeThinkerConfig]
):
    """Qwen3OmniMoe Thinker for multimodal understanding (text output only).

    This is the Thinker component that handles:
    - Audio input encoding via audio_tower
    - Vision (image/video) input encoding via visual
    - MoE text decoder for understanding and text generation via model

    Structure matches HuggingFace:
    - audio_tower: Qwen3OmniMoeAudioEncoder (top level)
    - visual: Qwen3OmniMoeVisionEncoder (top level)
    - model: Qwen3OmniMoeThinkerTextModel (text-only, top level)
    - lm_head: Linear projection (top level)
    """

    _task_type = TaskType.ANY_TO_ANY
    _model_type = "qwen3_omni_moe_thinker"
    _config_class = Qwen3OmniMoeThinkerConfig
    _auto_register = False
    _supports_video = True
    _uses_mrope = True

    loss_type = "ForCausalLM"

    def __init__(
        self,
        config: Qwen3OmniMoeThinkerConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        text_model = Qwen3OmniMoeThinkerTextModel(
            config=config.text_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        super().__init__(
            config=config,
            base_model=text_model,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            image_token_index=config.image_token_id,
            video_token_index=config.video_token_id,
            spatial_merge_size=getattr(config.vision_config, "spatial_merge_size", 2),
            create_lm_head=False,
        )

        self.audio_tower = Qwen3OmniMoeAudioEncoder(
            config=config.audio_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.visual = Qwen3OmniMoeVisionEncoder(
            config=config.vision_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        text_config = config.text_config
        self.vocab_size = text_config.vocab_size
        self.lm_head = nn.Linear(
            text_config.hidden_size,
            text_config.vocab_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.image_token_id = config.image_token_id
        self.video_token_id = config.video_token_id
        self.audio_token_id = config.audio_token_id

    @property
    def audio(self):
        """Property to access the audio encoder (alias for audio_tower)."""
        return self.audio_tower

    def compute_embedding(
        self,
        input_ids: Int[Array, "batch seq_len"],
        *,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        input_features: Float[Array, "batch mel_bins time"] | None = None,
        audio_embeds: Array | None = None,
        pixel_values: Float[Array, "num_patches channels h w"] | None = None,
        image_grid_thw: Int[Array, "num_images 3"] | None = None,
        image_max_grid_size: int | None = None,
        image_embeds: Array | None = None,
        pixel_values_videos: Float[Array, "num_patches channels h w"] | None = None,
        video_grid_thw: Int[Array, "num_videos 3"] | None = None,
        video_max_grid_size: int | None = None,
        video_embeds: Array | None = None,
        **kwargs,
    ) -> Array:
        if inputs_embeds is None:
            inputs_embeds = checkpoint_name(self.model.compute_embedding(input_ids), "embeddings")

        if input_ids is None and (
            input_features is not None
            or audio_embeds is not None
            or pixel_values is not None
            or image_embeds is not None
            or pixel_values_videos is not None
            or video_embeds is not None
        ):
            raise ValueError("`input_ids` must be provided to merge multimodal embeddings.")

        if audio_embeds is None and input_features is not None:
            audio_outputs = self.audio_tower(input_features)
            audio_embeds = audio_outputs.last_hidden_state

        if audio_embeds is not None:
            if audio_embeds.ndim == 3:
                audio_embeds = audio_embeds.reshape(-1, self.config.text_config.hidden_size)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                audio_embeds.astype(inputs_embeds.dtype),
                self.audio_token_id,
            )

        if image_embeds is None and pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("`image_grid_thw` must be provided when `pixel_values` is not None.")
            if image_max_grid_size is None:
                image_max_grid_size = int(jnp.max(image_grid_thw[:, 1:]).item())
            image_embeds = self.visual(pixel_values, image_grid_thw, image_max_grid_size)

        if image_embeds is not None:
            if image_embeds.ndim > 2:
                image_embeds = image_embeds.reshape(-1, image_embeds.shape[-1])
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                image_embeds.astype(inputs_embeds.dtype),
                self.image_token_id,
            )

        if video_embeds is None and pixel_values_videos is not None:
            if video_grid_thw is None:
                raise ValueError("`video_grid_thw` must be provided when `pixel_values_videos` is not None.")
            if video_max_grid_size is None:
                video_max_grid_size = int(jnp.max(video_grid_thw[:, 1:]).item())
            video_embeds = self.visual(pixel_values_videos, video_grid_thw, video_max_grid_size)

        if video_embeds is not None:
            if video_embeds.ndim > 2:
                video_embeds = video_embeds.reshape(-1, video_embeds.shape[-1])
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                video_embeds.astype(inputs_embeds.dtype),
                self.video_token_id,
            )

        return inputs_embeds

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        input_features: Float[Array, "batch mel_bins time"] | None = None,
        pixel_values: Float[Array, "num_patches channels h w"] | None = None,
        image_grid_thw: Int[Array, "num_images 3"] | None = None,
        image_max_grid_size: int | None = None,
        pixel_values_videos: Float[Array, "num_patches channels h w"] | None = None,
        video_grid_thw: Int[Array, "num_videos 3"] | None = None,
        video_max_grid_size: int | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> MoeCausalLMOutput:
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must provide either `input_ids` or `inputs_embeds`.")

        if (
            inputs_embeds is None
            or input_features is not None
            or pixel_values is not None
            or pixel_values_videos is not None
        ):
            if input_ids is None:
                raise ValueError("`input_ids` must be provided to compute or merge multimodal embeddings.")
            inputs_embeds = self.compute_embedding(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                input_features=input_features,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                image_max_grid_size=image_max_grid_size,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                video_max_grid_size=video_max_grid_size,
            )

        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
        )

        logits = None
        if apply_lm_head:
            logits = self.lm_head(outputs.last_hidden_state)

        return MoeCausalLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
            last_hidden_state=outputs.last_hidden_state,
            router_logits=outputs.router_logits,
        )

    def get_encoder(self):
        return self.visual

    def get_decoder(self):
        return self.model

    def get_embedding(self):
        return self.model.embed_tokens

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_lm_head(self):
        return self.lm_head

    def get_vision_tower(self) -> nn.Module:
        return self.visual

    def get_language_model(self) -> nn.Module:
        return self.model

    def get_image_features(
        self,
        pixel_values: Float[Array, "num_patches channels height width"],
        image_grid_thw: Int[Array, "num_images 3"] | None = None,
        **kwargs,
    ) -> Float[Array, "num_patches hidden"]:
        """Extract image features from the vision encoder.

        Args:
            pixel_values: Input pixel values for the images.
            image_grid_thw: Grid dimensions (temporal, height, width) for each image.
            **kwargs: Additional arguments passed to the vision encoder.

        Returns:
            Image features ready for merging with text embeddings.
        """
        return self.visual(
            pixel_values=pixel_values,
            grid_thw=image_grid_thw,
        ).last_hidden_state

    def get_video_features(
        self,
        pixel_values_videos: Float[Array, "num_patches channels height width"],
        video_grid_thw: Int[Array, "num_videos 3"] | None = None,
        **kwargs,
    ) -> Float[Array, "num_patches hidden"]:
        """Extract video features from the vision encoder.

        Args:
            pixel_values_videos: Input pixel values for the videos.
            video_grid_thw: Grid dimensions (temporal, height, width) for each video.
            **kwargs: Additional arguments passed to the vision encoder.

        Returns:
            Video features ready for merging with text embeddings.
        """
        return self.visual(
            pixel_values=pixel_values_videos,
            grid_thw=video_grid_thw,
        ).last_hidden_state

    def get_audio_features(
        self,
        input_features: Float[Array, "batch mel_bins time"],
        **kwargs,
    ) -> Float[Array, "batch seq hidden"]:
        """Extract audio features from the audio encoder.

        Args:
            input_features: Input mel-spectrogram features.
            **kwargs: Additional arguments passed to the audio encoder.

        Returns:
            Audio features ready for merging with text embeddings.
        """
        audio_outputs = self.audio_tower(input_features)
        return audio_outputs.last_hidden_state


@register_module(TaskType.ANY_TO_ANY, config=Qwen3OmniMoeConfig, model_type="qwen3_omni_moe")
class Qwen3OmniMoeForConditionalGeneration(
    BaseConditionalGenerationModule[Qwen3OmniMoeThinkerForConditionalGeneration, Qwen3OmniMoeConfig]
):
    """Full Qwen3OmniMoe model with Thinker, Talker, and Code2Wav.

    This is the top-level model that combines:
    - Thinker: Multimodal understanding (always loaded)
    - Talker: Speech generation with MoE + shared experts (optional)
    - Code2Wav: Codec-to-waveform vocoder (optional)

    Args:
        config: Full Qwen3OmniMoe configuration.
        dtype: Computation dtype.
        param_dtype: Parameter dtype.
        precision: JAX precision.
        rngs: Random number generators.
    """

    _task_type = TaskType.ANY_TO_ANY
    _model_type = "qwen3_omni_moe"
    _config_class = Qwen3OmniMoeConfig
    _auto_register = False

    def __init__(
        self,
        config: Qwen3OmniMoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        thinker = Qwen3OmniMoeThinkerForConditionalGeneration(
            config=config.thinker_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        super().__init__(
            config=config,
            base_model=thinker,
            base_model_name="thinker",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            create_lm_head=False,
        )

        self.has_talker = config.enable_audio_output
        if self.has_talker:
            self.talker = Qwen3OmniMoeTalkerForConditionalGeneration(
                config=config.talker_config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            self.code2wav = Qwen3OmniMoeCode2Wav(
                config=config.code2wav_config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )

    @property
    def visual(self):
        """Property to access the vision encoder."""
        return self.thinker.visual

    @property
    def audio(self):
        """Property to access the audio encoder."""
        return self.thinker.audio

    def compute_embedding(self, input_ids, *args, **kwargs):
        return self.thinker.compute_embedding(input_ids, *args, **kwargs)

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        input_features: Float[Array, "batch mel_bins time"] | None = None,
        pixel_values: Float[Array, "num_patches channels h w"] | None = None,
        image_grid_thw: Int[Array, "num_images 3"] | None = None,
        pixel_values_videos: Float[Array, "num_patches channels h w"] | None = None,
        video_grid_thw: Int[Array, "num_videos 3"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> MoeCausalLMOutput:
        """Forward pass through the Thinker component.

        For text generation, use this method. For audio output,
        use the talker and code2wav components separately.
        """
        return self.thinker(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            input_features=input_features,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            apply_lm_head=apply_lm_head,
        )

    def get_encoder(self):
        raise NotImplementedError("This is a decoder-only model.")

    def get_decoder(self):
        return self.thinker.get_decoder()

    def get_embedding(self):
        return self.thinker.get_embedding()

    def get_image_features(
        self,
        pixel_values: Float[Array, "num_patches channels height width"],
        image_grid_thw: Int[Array, "num_images 3"] | None = None,
        **kwargs,
    ) -> Float[Array, "num_patches hidden"]:
        """Extract image features from the vision encoder."""
        return self.thinker.get_image_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **kwargs,
        )


Qwen3OmniMoeThinkerModel = Qwen3OmniMoeModel


__all__ = [
    "Qwen3OmniMoeAudioConfig",
    "Qwen3OmniMoeAudioEncoder",
    "Qwen3OmniMoeAudioEncoderConfig",
    "Qwen3OmniMoeCode2Wav",
    "Qwen3OmniMoeCode2WavAttention",
    "Qwen3OmniMoeCode2WavConfig",
    "Qwen3OmniMoeCode2WavLayerScale",
    "Qwen3OmniMoeCode2WavMLP",
    "Qwen3OmniMoeCode2WavTransformerLayer",
    "Qwen3OmniMoeConfig",
    "Qwen3OmniMoeForConditionalGeneration",
    "Qwen3OmniMoeMLPStack",
    "Qwen3OmniMoeModel",
    "Qwen3OmniMoeTalkerCodePredictorAttention",
    "Qwen3OmniMoeTalkerCodePredictorConfig",
    "Qwen3OmniMoeTalkerCodePredictorDecoderLayer",
    "Qwen3OmniMoeTalkerCodePredictorForConditionalGeneration",
    "Qwen3OmniMoeTalkerCodePredictorMLP",
    "Qwen3OmniMoeTalkerCodePredictorModel",
    "Qwen3OmniMoeTalkerConfig",
    "Qwen3OmniMoeTalkerForConditionalGeneration",
    "Qwen3OmniMoeTalkerMLPStack",
    "Qwen3OmniMoeTalkerModel",
    "Qwen3OmniMoeTalkerTextAttention",
    "Qwen3OmniMoeTalkerTextConfig",
    "Qwen3OmniMoeTalkerTextDecoderLayer",
    "Qwen3OmniMoeTalkerTextMLP",
    "Qwen3OmniMoeTalkerTextSparseMoeBlock",
    "Qwen3OmniMoeTextAttention",
    "Qwen3OmniMoeTextConfig",
    "Qwen3OmniMoeTextDecoderLayer",
    "Qwen3OmniMoeTextMLP",
    "Qwen3OmniMoeTextSparseBlock",
    "Qwen3OmniMoeThinkerConfig",
    "Qwen3OmniMoeThinkerForConditionalGeneration",
    "Qwen3OmniMoeThinkerModel",
    "Qwen3OmniMoeVisionConfig",
    "Qwen3OmniMoeVisionEncoder",
    "Qwen3OmniMoeVisionEncoderConfig",
]
