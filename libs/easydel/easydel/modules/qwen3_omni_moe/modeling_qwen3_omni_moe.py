# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

import jax
import jax.numpy as jnp
import spectrax as spx
from eformer.pytree import auto_pytree
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int
from spectrax import apply_logical_sharding, common_types, nn

from easydel.caching import (
    HybridCache,
    OperationsMetadata,
    RaggedPagesCache,
    RaggedPagesCacheView,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.infra.base_module import EasyDeLBaseModule, EasyDeLLayerStackMixin
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import (
    BaseModelOutput,
    DecoderLayerOutput,
    EmbeddingInfo,
    ModelOutput,
    MoeCausalLMOutput,
    VLMCausalLMOutput,
)
from easydel.infra.utils import ACT2FN, auto_remat, blockwise_ffn
from easydel.layers import (
    BaseMoeModule,
    ColumnParallelLinear,
    ColumnParallelMoELinear,
    Embed,
    MoeLoadBalancingStrategy,
    MoeRoutingStrategy,
    ParallelLinear,
    RMSNorm,
    RowParallelLinear,
    RowParallelMoELinear,
    dense_qkv_layout,
    moe_down_projection_reform_param,
    moe_fused_gate_up_reform_param,
)
from easydel.layers.attention import FlexibleAttentionModule, UnifiedAttention, block_diagonal_bias
from easydel.layers.mlp import GatedMLP, moe_gated_expert_forward
from easydel.layers.norms import LayerNorm
from easydel.layers.rotary import (
    apply_rope_split_half,
    vision_grid_pos_embed_interpolate,
    vision_grid_rotary_embeddings,
)
from easydel.modules._base import BaseConditionalGenerationModule, BaseVisionLanguageModule

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
from .rope_utils import build_omni_rope_positions, build_omni_rope_positions_traced


@auto_pytree
class Qwen3OmniMoeCausalLMOutputWithPast(ModelOutput):
    """Multimodal causal-LM outputs with optional audio, vision, and router features.

    Attributes:
        loss: Language-model loss when labels are supplied.
        logits: Vocabulary prediction scores.
        past_key_values: Cached key/value states for generation.
        hidden_states: Optional hidden states from the decoder layers.
        attentions: Optional per-layer attention weights.
        rope_deltas: Multimodal RoPE position offsets.
        image_hidden_states: Encoded image features.
        audio_hidden_states: Encoded audio features.
        router_logits: Optional per-layer expert-routing logits.
    """

    loss: Array | None = None
    logits: Array = None
    past_key_values: list[Array] | None = None
    hidden_states: tuple[Array] | None = None
    attentions: tuple[Array] | None = None
    rope_deltas: Array | None = None
    image_hidden_states: Float[Array, "batch seq_len hidden_dim"] | None = None
    audio_hidden_states: Float[Array, "batch seq_len hidden_dim"] | None = None
    router_logits: tuple[Array] | None = None


class Qwen3OmniMoeAudioMLP(spx.Module):
    """Two-layer audio feed-forward network, also shared by the encoder layers."""

    def __init__(
        self,
        config: Qwen3OmniMoeAudioConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the audio feed-forward projections.

        Args:
            config: Audio dimensions and activation configuration.
            dtype: Computation dtype.
            param_dtype: Parameter dtype.
            precision: Matmul precision.
            rngs: Parameter initialization streams.
        """
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

    def forward(self, hidden_states: Array) -> Array:
        """Apply the feed-forward transform, preserving the input hidden dimension.

        Args:
            hidden_states: Input audio states of shape [batch, seq_len, d_model].

        Returns:
            Transformed states with the same shape.
        """
        return self.fc2(self.act(self.fc1(hidden_states)))


class Qwen3OmniMoeAudioAttention(spx.Module):
    """Self-attention module for audio encoder.

    Implements multi-head self-attention for processing audio features
    without causal masking. Uses FlexibleAttentionModule for efficient
    attention computation.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeAudioConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize audio encoder attention layer.

        Args:
            config (Qwen3OmniMoeAudioConfig): Audio encoder configuration.
            layer_idx (int): Index of this layer in the encoder stack.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
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
            use_bias=True,
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

    def forward(
        self,
        hidden_states: Array,
        attention_mask: Array | None = None,
    ) -> Array:
        """Apply self-attention to audio features.

        Args:
            hidden_states: Input tensor [batch, seq_len, d_model].
            attention_mask: Optional attention mask [batch, seq_len].

        Returns:
            Attention output tensor [batch, seq_len, d_model].
        """
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        q = apply_logical_sharding(
            q,
            dynamic_axes=common_types.AttnQSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        k = apply_logical_sharding(
            k,
            dynamic_axes=common_types.AttnKVSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        v = apply_logical_sharding(
            v,
            dynamic_axes=common_types.AttnKVSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )

        attn_output = self.attention.forward(
            query_states=q,
            key_states=k,
            value_states=v,
            mode=common_types.MODE_TRAIN,
            mask_info=MaskInfo.from_attention_mask(attention_mask) if attention_mask is not None else None,
            causal=False,
        ).attention_outputs

        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        attn_output = apply_logical_sharding(
            attn_output,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        attn_output = self.out_proj(attn_output)
        attn_output = apply_logical_sharding(
            attn_output,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        return typing.cast(Array, attn_output)


class Qwen3OmniMoeAudioEncoderLayer(Qwen3OmniMoeAudioMLP):
    """Transformer layer for audio encoder.

    Implements a single transformer layer with pre-normalization architecture,
    self-attention, and feedforward network for processing audio features.
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
        rngs: spx.Rngs,
    ):
        """Initialize audio encoder layer.

        Args:
            config (Qwen3OmniMoeAudioConfig): Audio encoder configuration.
            layer_idx (int): Index of this layer in the encoder stack.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.self_attn = Qwen3OmniMoeAudioAttention(
            config=config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.self_attn_layer_norm = LayerNorm(
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
        self.final_layer_norm = LayerNorm(
            config.d_model,
            epsilon=1e-5,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def forward(
        self,
        hidden_states: Array,
        attention_mask: Array | None = None,
    ) -> Array:
        """Forward pass through audio encoder layer.

        Args:
            hidden_states: Input tensor [batch, seq_len, d_model].
            attention_mask: Optional attention mask.

        Returns:
            Output tensor [batch, seq_len, d_model] after attention and FFN.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = super().forward(hidden_states)
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
        rngs: spx.Rngs,
    ):
        """Initialize audio encoder.

        Args:
            config (Qwen3OmniMoeAudioConfig): Audio encoder configuration with
                conv and transformer parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
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

        self.conv2d1 = nn.Conv2d(
            in_channels=1,
            out_channels=config.downsample_hidden_size,
            kernel_size=3,
            stride=2,
            padding=((1, 1), (1, 1)),
            dtype=dtype,
            rngs=rngs,
            precision=precision,
        )
        self.conv2d2 = nn.Conv2d(
            in_channels=config.downsample_hidden_size,
            out_channels=config.downsample_hidden_size,
            kernel_size=3,
            stride=2,
            padding=((1, 1), (1, 1)),
            dtype=dtype,
            rngs=rngs,
            precision=precision,
        )
        self.conv2d3 = nn.Conv2d(
            in_channels=config.downsample_hidden_size,
            out_channels=config.downsample_hidden_size,
            kernel_size=3,
            stride=2,
            padding=((1, 1), (1, 1)),
            dtype=dtype,
            rngs=rngs,
            precision=precision,
        )

        mel_after_conv = (((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2
        with self.assign_layer_stage(0, total_layers=config.encoder_layers):
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

        self.layers = nn.ModuleList([])
        for i in range(config.encoder_layers):
            with self.assign_layer_stage(i, total_layers=config.encoder_layers):
                self.layers.append(
                    Qwen3OmniMoeAudioEncoderLayer(
                        config=config,
                        layer_idx=i,
                        dtype=dtype,
                        param_dtype=param_dtype,
                        precision=precision,
                        rngs=rngs,
                    )
                )

        with self.assign_layer_stage(config.encoder_layers - 1, total_layers=config.encoder_layers):
            self.ln_post = LayerNorm(
                config.d_model,
                epsilon=1e-5,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

        with self.assign_layer_stage(config.encoder_layers - 1, total_layers=config.encoder_layers):
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
        with self.assign_layer_stage(config.encoder_layers - 1, total_layers=config.encoder_layers):
            self.proj2 = RowParallelLinear(
                config.d_model,
                config.output_dim,
                use_bias=True,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )

    def forward(
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

        batch_size, mel_bins, time = input_features.shape
        chunk_size = 2 * self.n_window
        if chunk_size <= 0 or self.conv_chunksize <= 0 or self.n_window_infer < chunk_size:
            raise ValueError("Audio windows and conv_chunksize must be positive; n_window_infer must cover a chunk.")
        if feature_lens is None:
            feature_lens = (
                jnp.sum(attention_mask, axis=-1, dtype=jnp.int32)
                if attention_mask is not None
                else jnp.full((batch_size,), time, dtype=jnp.int32)
            )
        feature_lens = jnp.clip(jnp.asarray(feature_lens, dtype=jnp.int32), 0, time)
        valid_frames = jnp.arange(time)[None, :] < feature_lens[:, None]
        if attention_mask is not None:
            valid_frames = valid_frames & attention_mask.astype(jnp.bool_)
        input_features = jnp.where(valid_frames[:, None, :], input_features, 0)

        # Static padded chunks keep the encoder JIT-compatible even when lengths are traced.
        num_chunks = (time + chunk_size - 1) // chunk_size
        conv_chunk_size = min(time, chunk_size)
        padded = jnp.pad(input_features, ((0, 0), (0, 0), (0, num_chunks * conv_chunk_size - time)))
        chunks = padded.reshape(batch_size, mel_bins, num_chunks, conv_chunk_size)
        chunks = chunks.transpose(0, 2, 3, 1).reshape(batch_size * num_chunks, conv_chunk_size, mel_bins, 1)
        # HF pads every chunk to the longest *actual* chunk in the batch, not
        # the padded input width. Keep static shapes, but reproduce that common
        # convolution boundary after every stage so out-of-bound activations
        # (including bias and the last valid frame's receptive field) cannot feed
        # back into retained positions at the next stage. Do not mask to each
        # chunk's own length: HF retains padding activations up to the batch max.
        conv_extent = jnp.minimum(jnp.max(feature_lens), conv_chunk_size)
        conv_outputs = []
        for start in range(0, chunks.shape[0], self.conv_chunksize):
            chunk = chunks[start : start + self.conv_chunksize].astype(self.dtype)
            extent = conv_extent
            for conv in (self.conv2d1, self.conv2d2, self.conv2d3):
                # The checkpoint's CNN uses exact GELU, independently of the
                # configurable transformer/projection activation.
                chunk = jax.nn.gelu(conv(chunk), approximate=False)
                extent = (extent + 1) // 2
                inside_extent = jnp.arange(chunk.shape[1]) < extent
                chunk = jnp.where(inside_extent[None, :, None, None], chunk, 0)
            conv_outputs.append(chunk)
        hidden_states = jnp.concatenate(conv_outputs, axis=0)
        b, t, f, c = hidden_states.shape
        # HF flattens [channels, frequency], not [frequency, channels].
        hidden_states = hidden_states.transpose(0, 1, 3, 2).reshape(b, t, c * f)
        hidden_states = self.conv_out(hidden_states)
        pos_emb = create_sinusoidal_positions(t, self.d_model).astype(hidden_states.dtype)
        hidden_states = hidden_states + pos_emb[None, :, :]
        hidden_states = hidden_states.reshape(batch_size, num_chunks * t, self.d_model)

        chunk_lengths = jnp.clip(feature_lens[:, None] - jnp.arange(num_chunks)[None, :] * chunk_size, 0, chunk_size)
        encoded_lengths = (chunk_lengths + 7) // 8
        valid_tokens = (jnp.arange(t)[None, None, :] < encoded_lengths[..., None]).reshape(batch_size, -1)
        window_size = t * (self.n_window_infer // chunk_size)
        window_ids = jnp.arange(num_chunks * t) // window_size
        attention_mask = (window_ids[:, None] == window_ids[None, :])[None, None, :, :] & valid_tokens[:, None, None, :]

        def _layer_loop(layer, carry):
            """Apply a single audio-encoder layer inside the layer-stack scan.

            Body of ``self.layers.scan``; runs ``layer`` on the current
            audio hidden states under the appropriate stage context and
            returns the updated carry tuple.
            """
            hidden_states, idx = carry
            with self._layer_stage_context(idx, layers=self.layers):
                hidden_states = layer(hidden_states, attention_mask)
            hidden_states = self._mark_layer_stage_boundary(hidden_states, idx, layers=self.layers)

            return hidden_states, idx + 1

        hidden_states, _ = self.layers.scan(
            _layer_loop,
            (hidden_states, 0),
            trace=not self.config.scan_layers or self._pipeline_stage_count() > 1,
            **self._layer_scan_kwargs("audio"),
        )
        hidden_states = self.ln_post(hidden_states)

        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        hidden_states = jnp.where(valid_tokens[..., None], hidden_states, 0)

        return BaseModelOutput(last_hidden_state=hidden_states)

    def get_encoder(self):
        """Return the audio encoder itself.

        The Qwen3-Omni audio tower is encoder-only; calling code that
        expects an encoder/decoder split (e.g. via the
        ``EncoderDecoderProtocol`` contract) gets the same module back.

        Returns:
            spx.Module: ``self``.
        """
        return self

    def get_decoder(self):
        """Encoder-only audio model has no decoder.

        Raises:
            NotImplementedError: Always — the audio encoder produces
                continuous features that are merged into the text
                trunk; there is no separate decoder stage.
        """
        raise NotImplementedError("Audio encoder does not have a decoder.")

    def get_lm_head(self):
        """Audio encoder does not project to a vocabulary.

        Raises:
            NotImplementedError: Always — audio features are consumed by
                the text decoder via projection layers, not by an LM head
                on the encoder itself.
        """
        raise NotImplementedError("Audio encoder does not have a language model head.")

    def get_embedding(self):
        """Return the front-end convolution that embeds mel-spectrograms.

        The audio encoder's "embedding" is the first 2-D convolution
        (``conv2d1``) which strides over the mel-spectrogram input and
        produces the initial sequence of audio tokens consumed by the
        transformer blocks.

        Returns:
            spx.Module: The initial ``Conv2D`` patchifier.
        """
        return self.conv2d1


def _encode_audio_for_merging(audio_tower, input_features, feature_lens=None, feature_attention_mask=None):
    """Encode audio and compact valid tokens before text-placeholder merging.

    The static-sized tail is unused by placeholder merging; valid tokens from every
    sample must precede it rather than leaving padding between different samples.
    """
    if feature_lens is None:
        feature_lens = (
            jnp.sum(feature_attention_mask, axis=-1, dtype=jnp.int32)
            if feature_attention_mask is not None
            else jnp.full((input_features.shape[0],), input_features.shape[-1], dtype=jnp.int32)
        )
    features = audio_tower(
        input_features, feature_lens=feature_lens, attention_mask=feature_attention_mask
    ).last_hidden_state
    chunk_size = 2 * audio_tower.n_window
    lengths = jnp.clip(jnp.asarray(feature_lens), 0, input_features.shape[-1])
    output_lengths = (lengths // chunk_size) * ((chunk_size + 7) // 8) + (lengths % chunk_size + 7) // 8
    valid = jnp.arange(features.shape[1])[None, :] < output_lengths[:, None]
    flat_features = features.reshape(-1, features.shape[-1])
    indices = jnp.nonzero(valid.reshape(-1), size=flat_features.shape[0], fill_value=0)[0]
    return flat_features[indices], jnp.sum(output_lengths, dtype=jnp.int32)


def create_attention_mask(cu_seqlens: Array, seq_length: int, dtype: jnp.dtype) -> Array:
    """Build a block-diagonal additive attention mask from a cu_seqlens vector.

    Vision/audio encoders pack multiple variable-length samples into a
    single sequence of length ``seq_length`` and use ``cu_seqlens``
    (cumulative token offsets, length ``num_samples + 1``) to delimit
    them. This helper produces the additive mask that allows attention
    only within a single sample's span: positions in different blocks
    receive ``-inf`` and positions in the same block receive ``0``.

    Args:
        cu_seqlens: Cumulative token offsets, shape
            ``(num_samples + 1,)``. Block ``i`` covers indices
            ``[cu_seqlens[i], cu_seqlens[i + 1])``.
        seq_length: Total packed sequence length.
        dtype: Dtype of the resulting mask. ``finfo(dtype).min`` is used
            for the masked-out positions so the mask can be added
            directly to attention logits in the same dtype.

    Returns:
        Additive mask of shape ``(1, seq_length, seq_length)`` ready to
        be broadcast across attention heads.
    """
    # Shared segment-membership mask (no slicing by traced ``cu_seqlens`` values, so it runs under ``lax.scan``).
    return block_diagonal_bias(jnp.asarray(cu_seqlens, dtype=jnp.int32), seq_length, dtype)


class Qwen3OmniMoeVisionPatchEmbed(spx.Module):
    """3D patch embedding for vision encoder.

    Converts image/video patches into embedding vectors using 3D convolution
    that processes temporal, height, and width dimensions simultaneously.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize vision patch embedding layer.

        Args:
            config (Qwen3OmniMoeVisionConfig): Vision encoder configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.dtype = dtype
        self.config = config
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.hidden_size = config.hidden_size

        kernel_size = (config.temporal_patch_size, config.patch_size, config.patch_size)
        self.proj = nn.Conv3d(
            in_channels=config.in_channels,
            out_channels=config.hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            use_bias=True,
            dtype=dtype,
            rngs=rngs,
            precision=precision,
        )

    def forward(self, hidden_states: Array) -> Array:
        """Convert image/video patches to embeddings.

        Args:
            hidden_states: Input patches flattened into [num_patches, patch_elements].

        Returns:
            Patch embeddings [num_patches, hidden_size].
        """
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
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        return hidden_states


class Qwen3OmniMoeVisionPatchMerger(spx.Module):
    """Spatial patch merger for vision encoder.

    Merges spatially adjacent patches to reduce sequence length while
    preserving information. Uses layer normalization and MLP for processing.
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
        rngs: spx.Rngs,
    ):
        """Initialize vision patch merger.

        Args:
            config (Qwen3OmniMoeVisionConfig): Vision encoder configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            use_postshuffle_norm (bool, optional): Whether to apply norm after shuffling.
                Defaults to False.
            rngs (spx.Rngs): Random number generator state.
        """
        self.dtype = dtype
        self.spatial_merge_size = config.spatial_merge_size
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm

        self.ln_q = LayerNorm(
            self.hidden_size if use_postshuffle_norm else config.hidden_size,
            epsilon=1e-6,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.mlp_linear_1 = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.mlp_linear_2 = RowParallelLinear(
            self.hidden_size,
            config.out_hidden_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def forward(self, x: Array) -> Array:
        """Merge adjacent patches and project to output dimension.

        Args:
            x: Input patch embeddings.

        Returns:
            Merged and projected embeddings.
        """
        x = self.ln_q(x.reshape(-1, self.hidden_size) if self.use_postshuffle_norm else x).reshape(-1, self.hidden_size)
        x = self.mlp_linear_1(x)
        x = jax.nn.gelu(x, approximate=False)
        x = self.mlp_linear_2(x)
        return x


class Qwen3OmniMoeVisionMLP(spx.Module):
    """Feed-forward network for vision encoder.

    Implements a two-layer feedforward network with GELU activation
    for processing vision features in the vision encoder.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize vision encoder MLP.

        Args:
            config (Qwen3OmniMoeVisionConfig): Vision encoder configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
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

    def forward(self, x: Array) -> Array:
        """Apply feedforward transformation to vision features.

        Args:
            x: Input tensor [seq_len, hidden_size].

        Returns:
            Transformed hidden states [seq_len, hidden_size].
        """
        return self.linear_fc2(self.act(self.linear_fc1(x)))


class Qwen3OmniMoeVisionAttention(spx.Module):
    """Self-attention module for vision encoder.

    Implements multi-head self-attention with rotary position embeddings
    for processing vision features. Uses FlexibleAttentionModule for
    efficient attention computation.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeVisionConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize vision encoder attention layer.

        Args:
            config (Qwen3OmniMoeVisionConfig): Vision encoder configuration.
            layer_idx (int): Index of this layer in the encoder stack.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
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

    def forward(
        self,
        hidden_states: Array,
        cu_seqlens: Array,
        rotary_pos_emb: Array,
    ) -> Array:
        """Apply self-attention with RoPE to vision features.

        Args:
            hidden_states: Input tensor [seq_len, hidden_size].
            cu_seqlens: Cumulative sequence lengths for attention masking.
            rotary_pos_emb: Rotary position embeddings.

        Returns:
            Attention output tensor [seq_len, hidden_size].
        """
        seq_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states)
        q, k, v = map(
            lambda x: x.squeeze(0),
            jnp.split(qkv.reshape(seq_length, 3, self.num_heads, -1).transpose(1, 0, 2, 3), 3, 0),
        )

        cos = jnp.cos(rotary_pos_emb)
        sin = jnp.sin(rotary_pos_emb)
        q = apply_rope_split_half(q, cos, sin)
        k = apply_rope_split_half(k, cos, sin)

        # Attention ops take [batch, seq, heads, dim]; packed vision tokens form one batch row and
        # the cu_seqlens block mask is additive (same as the Qwen3-VL vision tower).
        q = jnp.expand_dims(q, 0)
        k = jnp.expand_dims(k, 0)
        v = jnp.expand_dims(v, 0)
        attn_bias = create_attention_mask(cu_seqlens, seq_length, q.dtype)
        attn_bias = jnp.broadcast_to(attn_bias, (1, self.num_heads, seq_length, seq_length))

        attn_output = self.attention.forward(
            query_states=q,
            key_states=k,
            value_states=v,
            bias=attn_bias,
            mode=common_types.MODE_TRAIN,
            causal=False,
        ).attention_outputs

        attn_output = attn_output.squeeze(0).reshape(seq_length, -1)
        attn_output = apply_logical_sharding(
            attn_output,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        attn_output = checkpoint_name(self.proj(attn_output), "vision_attn_output")
        attn_output = apply_logical_sharding(
            attn_output,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        return typing.cast(Array, attn_output)


class Qwen3OmniMoeVisionBlock(spx.Module):
    """Transformer block for vision encoder.

    Implements a single transformer layer with pre-normalization architecture,
    self-attention with RoPE, and feedforward network for processing vision features.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeVisionConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize vision encoder block.

        Args:
            config (Qwen3OmniMoeVisionConfig): Vision encoder configuration.
            layer_idx (int): Index of this layer in the encoder stack.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.norm1 = LayerNorm(config.hidden_size, epsilon=1e-6, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.norm2 = LayerNorm(config.hidden_size, epsilon=1e-6, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
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

    def forward(
        self,
        hidden_states: Array,
        cu_seqlens: Array,
        rotary_pos_emb: Array,
    ) -> Array:
        """Forward pass through vision encoder block.

        Args:
            hidden_states: Input tensor [seq_len, hidden_size].
            cu_seqlens: Cumulative sequence lengths for attention masking.
            rotary_pos_emb: Rotary position embeddings.

        Returns:
            Output tensor [seq_len, hidden_size] after attention and FFN.
        """
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens, rotary_pos_emb)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3OmniMoeVisionEncoder(EasyDeLBaseModule):
    """Vision encoder for Qwen3OmniMoe.

    Processes images and videos through:
    1. 3D patch embedding for temporal and spatial patches
    2. Position embeddings for spatial locations
    3. Transformer encoder layers with RoPE
    4. Patch merging to reduce sequence length

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
        rngs: spx.Rngs,
    ):
        """Initialize vision encoder.

        Args:
            config (Qwen3OmniMoeVisionConfig): Vision encoder configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.merger_list = nn.ModuleList(
            [
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
        )

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

        with self.assign_layer_stage(0, total_layers=config.depth):
            self.pos_embed = Embed(
                num_embeddings=config.num_position_embeddings,
                features=config.hidden_size,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        head_dim = config.hidden_size // config.num_heads
        self._head_dim_ro = head_dim // 2

        self.blocks = nn.ModuleList([])
        for idx in range(config.depth):
            with self.assign_layer_stage(idx, total_layers=config.depth):
                self.blocks.append(
                    Qwen3OmniMoeVisionBlock(
                        config=config,
                        layer_idx=idx,
                        dtype=dtype,
                        param_dtype=param_dtype,
                        precision=precision,
                        rngs=rngs,
                    )
                )

        self.merger = Qwen3OmniMoeVisionPatchMerger(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            use_postshuffle_norm=False,
            rngs=rngs,
        )

        self.deepstack_visual_indexes = config.deepstack_visual_indexes

    def fast_pos_embed_interpolate(self, grid_thw: Array) -> Array:
        """Interpolate learned positions and reorder patches into spatial-merge blocks."""
        return vision_grid_pos_embed_interpolate(
            self.pos_embed, grid_thw, self.num_grid_per_side, self.spatial_merge_size
        )

    def rot_pos_emb(self, grid_thw: Array, max_grid_size: int) -> Array:
        """Compute 2-D RoPE frequencies for the packed vision tokens.

        For every visual sample described in ``grid_thw`` (one row per
        image/video, columns ``T, H, W``), generates per-token height
        and width position indices, expands them with
        ``spatial_merge_size`` so the patch-merge layout matches the
        rest of the encoder, repeats them along the temporal axis, and
        finally indexes a precomputed inverse-frequency table to obtain
        the rotary embedding for each spatial token.

        Args:
            grid_thw: Per-sample grid sizes of shape ``(num_samples, 3)``
                with columns ``(T, H, W)``.
            max_grid_size: Upper bound on grid extent used to size the
                shared inverse-frequency table.

        Returns:
            Rotary position embedding of shape
            ``(total_tokens, head_dim_ro)`` where ``total_tokens`` is
            the sum of ``T * H * W`` over the batch.
        """
        return vision_grid_rotary_embeddings(grid_thw, max_grid_size, self._head_dim_ro, self.spatial_merge_size)

    def forward(
        self,
        hidden_states: Array,
        grid_thw: Array,
        max_grid_size: int,
    ) -> tuple[Array, list[Array]]:
        """Encode visual patches and return merged and intermediate DeepStack features.

        Args:
            hidden_states: Input pixel values [num_patches, channels * t * h * w].
            grid_thw: Grid dimensions (temporal, height, width) for each image/video.
            max_grid_size: Maximum grid size for rotary embedding computation.

        Returns:
            Merged features and one intermediate feature per configured DeepStack layer.
        """
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds.astype(hidden_states.dtype)
        rotary_pos_emb = self.rot_pos_emb(grid_thw, max_grid_size)
        # HF: ``emb = cat((rotary_pos_emb, rotary_pos_emb), -1)`` before cos/sin (full head_dim width).
        rotary_pos_emb = jnp.concatenate([rotary_pos_emb, rotary_pos_emb], axis=-1)

        grid_lens = grid_thw[:, 1] * grid_thw[:, 2]
        repeated = jnp.repeat(grid_lens, grid_thw[:, 0])
        cu_seqlens = jnp.cumsum(repeated, dtype="i4")
        cu_seqlens = jnp.pad(cu_seqlens, (1, 0), constant_values=0)

        deepstack_features = []

        def _layer_loop(block, carry):
            """Run a visual block, collecting configured intermediate features."""
            hidden_states, idx = carry
            with self._layer_stage_context(idx, layers=self.blocks):
                hidden_states = block(hidden_states, cu_seqlens, rotary_pos_emb)
            hidden_states = self._mark_layer_stage_boundary(hidden_states, idx, layers=self.blocks)
            if idx in self.deepstack_visual_indexes:
                merger_idx = self.deepstack_visual_indexes.index(idx)
                deepstack_features.append(self.merger_list[merger_idx](hidden_states))
            return hidden_states, idx + 1

        # A compiled scan cannot append intermediate arrays to a Python list.
        hidden_states, _ = self.blocks.scan(
            _layer_loop,
            (hidden_states, 0),
            trace=not self.config.scan_layers or self._pipeline_stage_count() > 1 or bool(self.deepstack_visual_indexes),
            **self._layer_scan_kwargs("vision"),
        )
        return typing.cast(Array, self.merger(hidden_states)), deepstack_features

    def get_encoder(self):
        """Return the vision encoder itself.

        Returns:
            spx.Module: ``self`` — Qwen3-Omni's vision tower is
            encoder-only.
        """
        return self

    def get_decoder(self):
        """Encoder-only vision model has no decoder.

        Raises:
            NotImplementedError: Always — visual features are merged into
                the text decoder via the patch merger and projector;
                there is no separate vision decoder.
        """
        raise NotImplementedError("Vision model does not have a decoder.")

    def get_lm_head(self):
        """Vision encoder does not project to a vocabulary.

        Raises:
            NotImplementedError: Always — the LM head lives on the text
                generation wrapper, not the vision tower.
        """
        raise NotImplementedError("Vision model does not have a language model head.")

    def get_embedding(self):
        """Return the patch embedding that turns pixels into vision tokens.

        Returns:
            spx.Module: The ``patch_embed`` module, which projects
            flattened ``(channels, T, H, W)`` patches to
            ``hidden_size``-dimensional tokens consumed by the
            transformer blocks.
        """
        return self.patch_embed


class Qwen3OmniMoeTextMLP(GatedMLP):
    """Dense MLP for non-MoE layers in Qwen3OmniMoe text decoder.

    Implements the feedforward network with SwiGLU activation function
    for enhanced representation learning. Used for non-MoE layers.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        intermediate_size: int | None = None,
        kernel_init=None,
    ):
        """Initialize the shared gated MLP with the submodel's width and initializer.

        Args:
            config (Qwen3OmniMoeTextConfig): Submodel dimensions and activation.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
            intermediate_size: Optional shared-expert width override.
            kernel_init: Projection initializer. Defaults to normal(config.initializer_range).
        """
        super().__init__(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            intermediate_size=intermediate_size,
            kernel_init=kernel_init,
        )


class Qwen3OmniMoeMLPStack(spx.Module):
    """Stacked MoE MLP module using ParallelMoELinear layers.

    Implements the expert MLP stack with SwiGLU activation function using
    column and row parallel MoE linear layers for efficient expert computation.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize MoE MLP stack.

        Args:
            config (Qwen3OmniMoeTextConfig): Text decoder configuration with MoE parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        self.gate_up_proj = ColumnParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.hidden_size,
            out_features=2 * config.moe_intermediate_size,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(),
            use_bias=False,
            partition_manager=config.runtime_sharding_resolver,
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
            kernel_init=jax.nn.initializers.normal(),
            partition_manager=config.runtime_sharding_resolver,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    @property
    def reform_param(self):
        return {
            **moe_fused_gate_up_reform_param(config=self.config),
            **moe_down_projection_reform_param(),
        }

    def forward(
        self,
        hidden_states: Array,
        group_sizes: Array,
        sorted_experts: Array | None = None,
    ) -> Array:
        """Apply SwiGLU feedforward transformation through MoE experts.

        Args:
            hidden_states: Input tensor after expert routing.
            group_sizes: Array specifying the number of tokens routed to each expert.
            sorted_experts: Optional array of sorted expert indices.

        Returns:
            Transformed hidden states after MoE MLP processing.
        """
        return moe_gated_expert_forward(self, hidden_states, group_sizes, sorted_experts)


class Qwen3OmniMoeTextSparseBlock(BaseMoeModule):
    """Sparse Mixture of Experts (MoE) block for Qwen3OmniMoe text decoder.

    Implements token-level expert routing with top-k selection, combining
    outputs from multiple experts based on learned routing weights.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize text decoder MoE sparse block.

        Args:
            config (Qwen3OmniMoeTextConfig): Text decoder configuration with MoE parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
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
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.config = config
        self.gate = ColumnParallelLinear(
            config.hidden_size,
            config.num_experts,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
        )
        self.experts = Qwen3OmniMoeMLPStack(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def forward(self, hidden_states: Array) -> tuple[Array, Array]:
        """Route tokens through experts and combine outputs.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim].

        Returns:
            Tuple containing:
                - Output hidden states [batch, seq_len, hidden_dim] after expert processing
                - Router logits [batch * seq_len, num_experts] for auxiliary loss computation
        """
        out, router_logits = self.moe_call(
            hidden_state=hidden_states,
            gate_layer=self.gate,
            expert_layer=self.experts,
            gate_up_kernel=self.experts.gate_up_proj.kernel_view(),
            wd_kernel=self.experts.down_proj.kernel_view(),
            act_fn=self.experts.act_fn,
        )
        return checkpoint_name(out, "moe_expert_output"), checkpoint_name(router_logits, "moe_router_logits")


class Qwen3OmniMoeTextAttention(UnifiedAttention):
    """Causal self-attention for Qwen3OmniMoe text decoder.

    Multi-head attention layer with RoPE embeddings and Q/K normalization.
    Features layer-specific sliding window attention support.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize text decoder attention layer.

        Args:
            config (Qwen3OmniMoeTextConfig): Text decoder configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
            layer_idx (int): Index of this layer for sliding window configuration.
        """
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

    def _apply_rotary(self, query_states, key_states, position_ids, frequencies=None):
        """Keep fractional video timestamps on the direct mRoPE frequency path."""
        if jnp.issubdtype(position_ids.dtype, jnp.floating):
            # Integer lookup tables cannot represent timed video coordinates.
            frequencies = None
        return super()._apply_rotary(query_states, key_states, position_ids, frequencies)

    def _postprocess_qkv(self, query_states, key_states, value_states):
        """Apply Q/K normalization after computing query, key, and value projections.

        Args:
            query_states: Query tensor from projection layer.
            key_states: Key tensor from projection layer.
            value_states: Value tensor from projection layer.

        Returns:
            Tuple of normalized query, normalized key, and value tensors.
        """
        return self.query_normalization(query_states), self.key_normalization(key_states), value_states


class Qwen3OmniMoeTextDecoderLayer(spx.Module):
    """Single decoder layer for Qwen3OmniMoe text model.

    Combines multi-head attention with Q/K normalization and feedforward networks
    (either standard MLP or MoE) with RMS normalization and residual connections.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize text decoder layer.

        Args:
            config (Qwen3OmniMoeTextConfig): Text decoder configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
            layer_idx (int): Index of this layer, used to determine MoE vs MLP.
        """
        self.config = config
        self.layer_idx = layer_idx

        self.self_attn = Qwen3OmniMoeTextAttention(
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
            self.mlp = Qwen3OmniMoeTextSparseBlock(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        else:
            self.mlp = Qwen3OmniMoeTextMLP(
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

    def forward(
        self,
        hidden_states: Array,
        mask_info: MaskInfo,
        position_ids: Array,
        mode: common_types.RUNTIME_MODE_TYPES,  # type: ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        frequencies: Array | None = None,
    ) -> DecoderLayerOutput:
        """Forward pass through the decoder layer.

        Args:
            hidden_states (Array): Input tensor of shape (batch_size, sequence_length, hidden_dim).
            mask_info (MaskInfo): Attention mask information including causal masks.
            position_ids (Array): Position indices for tokens.
            mode (RUNTIME_MODE_TYPES): Runtime mode (train, decode, etc.).
            cache_view: Cache view for key-value states. Defaults to None.
            cache_metadata: Cache metadata. Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            output_router_logits (bool, optional): Whether to return router logits. Defaults to False.
            frequencies (Array | None, optional): Precomputed RoPE frequencies. Defaults to None.

        Returns:
            DecoderLayerOutput: Contains hidden states, attention weights, router logits, and cache view.
        """
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

        if self.config.use_scan_mlp and not self.is_moe:
            feed_forward_output = blockwise_ffn(
                self.mlp,
                feed_forward_input,
                self.config.scan_mlp_chunk_size,
            )
        else:
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


class Qwen3OmniMoeTalkerResizeMLP(spx.Module):
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
        rngs: spx.Rngs,
    ):
        """Initialize talker resize MLP.

        Args:
            config (Qwen3OmniMoeTalkerConfig): Talker configuration with input/output dimensions.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
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

    def forward(self, hidden_states: Array) -> Array:
        """Project hidden states from thinker to talker dimension.

        Args:
            hidden_states: Input tensor from thinker [batch, seq_len, thinker_hidden_dim].

        Returns:
            Resized hidden states [batch, seq_len, talker_hidden_dim].
        """
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_states)))


class Qwen3OmniMoeTalkerTextMLP(Qwen3OmniMoeTextMLP):
    """Dense MLP for Talker text model.

    Implements the feedforward network with SwiGLU activation function
    for enhanced representation learning. Used as shared expert in MoE layers.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerTextConfig,
        intermediate_size: int | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize talker text MLP.

        Args:
            config (Qwen3OmniMoeTalkerTextConfig): Talker text configuration.
            intermediate_size (int | None, optional): Override for intermediate layer size.
                Defaults to config.intermediate_size if None.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            intermediate_size=intermediate_size,
        )


class Qwen3OmniMoeTalkerMLPStack(Qwen3OmniMoeMLPStack):
    """Talker experts sharing the Thinker expert implementation and checkpoint layout."""


class Qwen3OmniMoeTalkerTextSparseMoeBlock(BaseMoeModule):
    """Sparse MoE block with SHARED EXPERT for Talker.

    This is the key new component that differs from Thinker's MoE:
    - Has a shared_expert (dense MLP) that processes ALL tokens
    - Has a shared_expert_gate (sigmoid gate) to weight the shared expert output
    - Output = routed_expert_output + sigmoid(gate(x)) * shared_expert(x)

    This architecture allows for better knowledge sharing across all tokens
    while still maintaining the benefits of sparse expert routing.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Talker MoE sparse block with shared expert.

        Args:
            config (Qwen3OmniMoeTalkerTextConfig): Talker text configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
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
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.config = config

        self.gate = ColumnParallelLinear(
            config.hidden_size,
            config.num_experts,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
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
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
        )

    def forward(self, hidden_states: Array) -> tuple[Array, Array]:
        """Route tokens through experts and shared expert, then combine outputs.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim].

        Returns:
            Tuple containing:
                - Output hidden states after MoE and shared expert processing
                - Router logits for auxiliary loss computation
        """
        routed_output, router_logits = self.moe_call(
            hidden_state=hidden_states,
            gate_layer=self.gate,
            expert_layer=self.experts,
            gate_up_kernel=self.experts.gate_up_proj.kernel_view(),
            wd_kernel=self.experts.down_proj.kernel_view(),
            act_fn=self.experts.act_fn,
        )

        shared_output = self.shared_expert(hidden_states)
        shared_gate = jax.nn.sigmoid(self.shared_expert_gate(hidden_states))

        output = routed_output + shared_gate * shared_output

        return checkpoint_name(output, "moe_expert_output"), checkpoint_name(router_logits, "moe_router_logits")


class Qwen3OmniMoeTalkerTextAttention(UnifiedAttention):
    """Causal self-attention for Talker text model.

    Multi-head attention layer with RoPE embeddings and Q/K normalization.
    Features sliding window attention support.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize Talker text attention layer.

        Args:
            config (Qwen3OmniMoeTalkerTextConfig): Talker text configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the decoder stack.
        """
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
        """Apply Q/K normalization after computing query, key, and value projections.

        Args:
            query_states: Query tensor from projection layer.
            key_states: Key tensor from projection layer.
            value_states: Value tensor from projection layer.

        Returns:
            Tuple of normalized query, normalized key, and value tensors.
        """
        return self.query_normalization(query_states), self.key_normalization(key_states), value_states


class Qwen3OmniMoeTalkerTextDecoderLayer(spx.Module):
    """Decoder layer for Talker text model with shared expert MoE.

    Combines multi-head attention with Q/K normalization and MoE feedforward
    network with shared expert using RMS normalization and residual connections.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize Talker text decoder layer.

        Args:
            config (Qwen3OmniMoeTalkerTextConfig): Talker text configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
            layer_idx (int): Index of this layer, used to determine MoE vs MLP.
        """
        self.config = config
        self.layer_idx = layer_idx

        self.self_attn = Qwen3OmniMoeTalkerTextAttention(
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
            self.mlp = Qwen3OmniMoeTalkerTextSparseMoeBlock(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        else:
            self.mlp = Qwen3OmniMoeTalkerTextMLP(
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

    def forward(
        self,
        hidden_states: Array,
        mask_info: MaskInfo,
        position_ids: Array,
        mode: common_types.RUNTIME_MODE_TYPES,  # type: ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        frequencies: Array | None = None,
    ) -> DecoderLayerOutput:
        """Forward pass through the Talker decoder layer.

        Args:
            hidden_states (Array): Input tensor of shape (batch_size, sequence_length, hidden_dim).
            mask_info (MaskInfo): Attention mask information including causal masks.
            position_ids (Array): Position indices for tokens.
            mode (RUNTIME_MODE_TYPES): Runtime mode (train, decode, etc.).
            cache_view: Cache view for key-value states. Defaults to None.
            cache_metadata: Cache metadata. Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            output_router_logits (bool, optional): Whether to return router logits. Defaults to False.
            frequencies (Array | None, optional): Precomputed RoPE frequencies. Defaults to None.

        Returns:
            DecoderLayerOutput: Contains hidden states, attention weights, router logits, and cache view.
        """
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

        if self.config.use_scan_mlp and not self.is_moe:
            feed_forward_output = blockwise_ffn(
                self.mlp,
                feed_forward_input,
                self.config.scan_mlp_chunk_size,
            )
        else:
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


class Qwen3OmniMoeTalkerCodePredictorMLP(Qwen3OmniMoeTextMLP):
    """Code-predictor SwiGLU block sharing the text MLP implementation and checkpoint layout."""


class Qwen3OmniMoeTalkerCodePredictorAttention(UnifiedAttention):
    """Causal self-attention for Talker code predictor.

    Multi-head attention layer with RoPE embeddings and Q/K normalization
    for processing codec token sequences.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerCodePredictorConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize code predictor attention with Q/K normalization and sliding window.

        Args:
            config: Code predictor configuration.
            dtype: Computation data type.
            param_dtype: Parameter storage data type.
            precision: JAX numerical precision.
            rngs: Random number generator state.
            layer_idx: Index of this layer in the stack.
        """
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
        """Apply per-head RMSNorm to query and key projections.

        Overrides :class:`UnifiedAttention._postprocess_qkv` to perform
        the Qwen3-style QK normalization before attention. The value
        projection is passed through unchanged.

        Args:
            query_states: Projected query of shape
                ``(batch, seq_len, num_heads, head_dim)``.
            key_states: Projected key of shape
                ``(batch, seq_len, num_kv_heads, head_dim)``.
            value_states: Projected value of shape
                ``(batch, seq_len, num_kv_heads, head_dim)``.

        Returns:
            ``(query_states, key_states, value_states)`` triple with
            QK normalization applied.
        """
        return self.query_normalization(query_states), self.key_normalization(key_states), value_states


class Qwen3OmniMoeTalkerCodePredictorDecoderLayer(spx.Module):
    """Single transformer block of the Talker code-predictor stack.

    Combines :class:`Qwen3OmniMoeTalkerCodePredictorAttention` (causal
    self-attention with sliding window and Q/K RMSNorm) with a dense
    SwiGLU MLP, both wrapped in pre-norm RMSNorm + residual connections.

    Unlike the upstream Qwen3-Omni text decoder, the code-predictor uses
    a *non-MoE* feed-forward block: every token sees the same dense MLP
    weights, which keeps acoustic-token prediction deterministic and
    avoids router stochasticity in the speech path.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerCodePredictorConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize code predictor decoder layer with attention and MLP.

        Args:
            config: Code predictor configuration.
            dtype: Computation data type.
            param_dtype: Parameter storage data type.
            precision: JAX numerical precision.
            rngs: Random number generator state.
            layer_idx: Index of this layer in the stack.
        """
        self.config = config
        self.layer_idx = layer_idx

        self.self_attn = Qwen3OmniMoeTalkerCodePredictorAttention(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )
        self.mlp = Qwen3OmniMoeTalkerCodePredictorMLP(
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

    def forward(
        self,
        hidden_states: Array,
        mask_info: MaskInfo,
        position_ids: Array,
        mode: common_types.RUNTIME_MODE_TYPES,  # type: ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Array | None = None,
    ) -> DecoderLayerOutput:
        """Run the decoder layer: self-attention followed by MLP with residual connections.

        Args:
            hidden_states: Input hidden states.
            mask_info: Attention mask information.
            position_ids: Position IDs for RoPE.
            mode: Runtime mode (train, decode, etc.).
            cache_view: KV cache view for this layer.
            cache_metadata: Cache metadata for efficient caching.
            output_attentions: Whether to return attention weights.
            frequencies: Precomputed RoPE frequencies.

        Returns:
            DecoderLayerOutput with updated hidden states and optional attention weights.
        """
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

        if self.config.use_scan_mlp:
            feed_forward_output = blockwise_ffn(
                self.mlp,
                feed_forward_input,
                self.config.scan_mlp_chunk_size,
            )
        else:
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
        rngs: spx.Rngs,
    ):
        """Initialize code predictor model with codec embeddings and decoder layers.

        Args:
            config: Code predictor configuration.
            dtype: Computation data type.
            param_dtype: Parameter storage data type.
            precision: JAX numerical precision.
            rngs: Random number generator state.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.codec_embedding = nn.ModuleList(
            [
                Embed(
                    config.vocab_size,
                    config.hidden_size,
                    embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
                    dtype=dtype,
                    param_dtype=param_dtype,
                    rngs=rngs,
                )
                for _ in range(config.num_code_groups - 1)
            ]
        )

        remat_layer_block = auto_remat(
            Qwen3OmniMoeTalkerCodePredictorDecoderLayer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.layers = nn.ModuleList([])
        for layer_idx in range(config.num_hidden_layers):
            with self.assign_layer_stage(layer_idx, total_layers=config.num_hidden_layers):
                self.layers.append(
                    remat_layer_block(
                        config=config,
                        layer_idx=layer_idx,
                        dtype=dtype,
                        param_dtype=param_dtype,
                        precision=precision,
                        rngs=rngs,
                    )
                )

        final_layer_idx = max(0, config.num_hidden_layers - 1)
        with self.assign_layer_stage(final_layer_idx, total_layers=config.num_hidden_layers):
            self.norm = RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

    def forward(
        self,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"],
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> BaseModelOutput:
        """Forward pass through the code predictor transformer.

        Args:
            inputs_embeds: Input embeddings [batch, seq_len, hidden_dim].
            attention_mask: Optional attention mask for padding.
            mask_info: Mask information for efficient attention.
            position_ids: Position IDs for RoPE.
            mode: Runtime mode (train, decode, etc.).
            past_key_values: Cached key-value states for generation.
            cache_metadata: Cache metadata for efficient caching.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return all hidden states.

        Returns:
            BaseModelOutput with last hidden state and optional intermediate outputs.
        """
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

        views = past_key_values.views if past_key_values is not None else None
        has_cache_views = views is not None and any(v is not None for v in views)
        needs_trace_cache = mode == common_types.MODE_DECODE or has_cache_views
        trace_layers = self._layer_scan_trace(
            False,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            cache_views=views,
            extra=needs_trace_cache,
        )
        cache_views = views if trace_layers else None
        # Hoisted out of the loop body: the caching property must not be
        # first materialized inside the lax.scan trace (tracer leak).
        frequencies = self.frequencies

        def _layer_loop(block, carry):
            """Apply a single decoder layer inside the layer-stack scan.

            Body of ``self.layers.scan`` for a non-MoE decoder stack;
            carries ``(hidden_states, cache_views, all_hidden_states,
            all_attentions, layer_index)``, optionally accumulating
            per-layer hidden states / attention weights.
            """
            hidden_states, cv, all_hidden_states, all_attentions, idx = carry
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            with self._layer_stage_context(idx, layers=self.layers):
                layer_outputs = block(
                    hidden_states=hidden_states,
                    mask_info=mask_info,
                    position_ids=position_ids,
                    mode=mode,
                    cache_view=self._layer_cache_view_at(cv, idx, enabled=trace_layers, cache=past_key_values),
                    cache_metadata=cache_metadata,
                    output_attentions=output_attentions,
                    frequencies=frequencies,
                )
            hidden_states = self._mark_layer_stage_boundary(layer_outputs.hidden_states, idx, layers=self.layers)

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            cv = self._layer_cache_view_update(
                cv,
                idx,
                layer_outputs.cache_view,
                enabled=trace_layers,
                cache=past_key_values,
            )

            return hidden_states, cv, all_hidden_states, all_attentions, idx + 1

        hidden_states, _, all_hidden_states, all_attentions, _ = self.layers.scan(
            _layer_loop,
            (hidden_states, cache_views, all_hidden_states, all_attentions, 0),
            trace=trace_layers,
            **self._layer_scan_kwargs("text"),
        )
        hidden_states = self.norm(hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            past_key_values=past_key_values,
        )

    def get_encoder(self):
        """Code predictor is decoder-only — no separate encoder.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("This is a decoder-only model.")

    def get_decoder(self):
        """Return the code-predictor decoder stack (``self``).

        Returns:
            spx.Module: ``self``. The base model *is* the decoder; the
            per-group LM heads live on
            :class:`Qwen3OmniMoeTalkerCodePredictorForConditionalGeneration`.
        """
        return self

    def get_lm_head(self):
        """Base code predictor has no LM head.

        Per-group acoustic LM heads are attached by the
        ``ForConditionalGeneration`` wrapper because there are
        ``num_code_groups - 1`` independent heads (one per residual
        codec quantizer level), which does not fit the single-head
        contract of the base module.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Use Qwen3OmniMoeTalkerCodePredictorForConditionalGeneration for LM head.")

    def get_embedding(self):
        """Return the first codec quantizer's embedding table.

        The code predictor maintains one ``Embed`` per quantizer level;
        the first table is treated as the canonical embedding for
        weight-tying / introspection purposes.

        Returns:
            spx.Module | None: The first ``Embed`` in
            ``self.codec_embedding``, or ``None`` if no codec embeddings
            are configured.
        """
        return self.codec_embedding[0] if self.codec_embedding else None


class Qwen3OmniMoeTalkerCodePredictorForConditionalGeneration(
    BaseConditionalGenerationModule[Qwen3OmniMoeTalkerCodePredictorModel, Qwen3OmniMoeTalkerCodePredictorConfig]  # type: ignore
):
    """Talker code-predictor wrapper with per-quantizer LM heads.

    Wraps :class:`Qwen3OmniMoeTalkerCodePredictorModel` and attaches
    ``num_code_groups - 1`` independent linear heads — one per residual
    quantizer level — so the model emits a separate logits tensor per
    codec group at every position. This enables Talker to predict the
    full residual codec token stack autoregressively while sharing a
    single transformer backbone.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerCodePredictorConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize code predictor with per-group LM heads for codec token prediction.

        Creates one linear head per code group (num_code_groups - 1) to predict
        codec tokens independently for each quantizer group.

        Args:
            config: Code predictor configuration.
            dtype: Computation data type.
            param_dtype: Parameter storage data type.
            precision: JAX numerical precision.
            rngs: Random number generator state.
        """
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

        self.lm_head = nn.ModuleList(
            [
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
        )

    def forward(
        self,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"],
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> MoeCausalLMOutput:
        """Forward pass through code predictor with per-group logit computation.

        Runs the transformer backbone and applies each per-group LM head to produce
        stacked logits of shape [batch, seq_len, num_groups, vocab_size].

        Args:
            inputs_embeds: Input embeddings [batch, seq_len, hidden_dim].
            attention_mask: Optional attention mask for padding.
            mask_info: Mask information for efficient attention.
            position_ids: Position IDs for RoPE.
            mode: Runtime mode (train, decode, etc.).
            past_key_values: Cached key-value states for generation.
            cache_metadata: Cache metadata for efficient caching.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return all hidden states.

        Returns:
            MoeCausalLMOutput with stacked per-group logits.
        """
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
            past_key_values=outputs.past_key_values,
            last_hidden_state=outputs.last_hidden_state,
        )

    def get_encoder(self):
        """Code-predictor wrapper is decoder-only — no encoder.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("This is a decoder-only model.")

    def get_decoder(self):
        """Return the wrapped code-predictor decoder.

        Returns:
            spx.Module: ``self.model`` — the decoder owns the
            transformer stack and codec embeddings.
        """
        return self.model.get_decoder()

    def get_lm_head(self):
        """Return the per-quantizer LM head list.

        Returns:
            spx.nn.ModuleList: One :class:`ColumnParallelLinear` head per
            residual quantizer level (length ``num_code_groups - 1``).
            Index ``g`` produces logits for the ``g+1``-th codec group
            (group 0 is consumed by the upstream Talker text model).
        """
        return self.lm_head

    def get_embedding(self):
        """Return the canonical codec embedding from the wrapped model.

        Returns:
            spx.Module | None: The first codec quantizer's embedding
            table, as exposed by
            :meth:`Qwen3OmniMoeTalkerCodePredictorModel.get_embedding`.
        """
        return self.model.get_embedding()


class Qwen3OmniMoeTalkerModel(EasyDeLBaseModule):
    """Talker text-side model that conditions on codec embeddings.

    The Talker stack consumes a sequence of codec tokens (acoustic
    quantizer indices) plus optional textual context, embeds the codec
    tokens through :attr:`codec_embedding`, and passes the result
    through a stack of MoE-aware decoder layers to produce hidden states
    consumed by the downstream code predictor and Code2Wav vocoder.

    This module is the speech-side analogue of a text decoder: it owns
    the codec embedding table and transformer body but no LM head; the
    head is supplied by the surrounding ``ForCausalLM`` wrapper.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the Talker text model with codec embedding and decoder layers.

        Args:
            config: Talker text configuration.
            dtype: Computation data type.
            param_dtype: Parameter storage data type.
            precision: JAX numerical precision.
            rngs: Random number generator state.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        with self.assign_layer_stage(0, total_layers=config.num_hidden_layers):
            self.codec_embedding = Embed(
                config.vocab_size,
                config.hidden_size,
                embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

        remat_layer_block = auto_remat(
            Qwen3OmniMoeTalkerTextDecoderLayer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.layers = nn.ModuleList([])
        for layer_idx in range(config.num_hidden_layers):
            with self.assign_layer_stage(layer_idx, total_layers=config.num_hidden_layers):
                self.layers.append(
                    remat_layer_block(
                        config=config,
                        layer_idx=layer_idx,
                        dtype=dtype,
                        param_dtype=param_dtype,
                        precision=precision,
                        rngs=rngs,
                    )
                )

        final_layer_idx = max(0, config.num_hidden_layers - 1)
        with self.assign_layer_stage(final_layer_idx, total_layers=config.num_hidden_layers):
            self.norm = RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_router_logits: bool = False,
    ) -> VLMCausalLMOutput:
        """Forward pass through the Talker text model.

        Embeds codec tokens (or uses provided embeddings) and processes them
        through the decoder layers with optional MoE routing.

        Args:
            input_ids: Codec token IDs [batch, seq_len].
            inputs_embeds: Pre-computed embeddings (overrides input_ids).
            attention_mask: Attention mask for padding.
            mask_info: Mask information for efficient attention.
            position_ids: Position IDs for RoPE.
            mode: Runtime mode (train, decode, etc.).
            past_key_values: Cached key-value states for generation.
            cache_metadata: Cache metadata for efficient caching.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return all hidden states.
            output_router_logits: Whether to return MoE router logits.

        Returns:
            VLMCausalLMOutput with hidden states and optional router logits.
        """
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
            partition_manager=self.config.runtime_sharding_resolver,
        )

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        views = past_key_values.views if past_key_values is not None else None
        has_cache_views = views is not None and any(v is not None for v in views)
        needs_trace_cache = mode == common_types.MODE_DECODE or has_cache_views
        # Router-logit aggregation grows a Python tuple per layer, which
        # lax.scan cannot carry — collecting it forces the trace path too.
        trace_layers = self._layer_scan_trace(
            False,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            cache_views=views,
            extra=needs_trace_cache or bool(output_router_logits),
        )
        cache_views = views if trace_layers else None
        # Hoisted out of the loop body: the caching property must not be
        # first materialized inside the lax.scan trace (tracer leak).
        frequencies = self.frequencies

        def _layer_loop(block, carry):
            """Apply a single Thinker MoE decoder layer inside the layer-stack scan.

            Body of ``self.layers.scan`` for the Thinker text decoder;
            carries ``(hidden_states, cache_views, all_hidden_states,
            all_attentions, all_router_logits, layer_index)``, optionally
            accumulating per-layer hidden states, attention weights, and
            MoE router logits.
            """
            hidden_states, cv, all_hidden_states, all_attentions, all_router_logits, idx = carry
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            with self._layer_stage_context(idx, layers=self.layers):
                layer_outputs = block(
                    hidden_states=hidden_states,
                    mask_info=mask_info,
                    position_ids=position_ids,
                    mode=mode,
                    cache_view=self._layer_cache_view_at(cv, idx, enabled=trace_layers, cache=past_key_values),
                    cache_metadata=cache_metadata,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    frequencies=frequencies,
                )
            hidden_states = self._mark_layer_stage_boundary(layer_outputs.hidden_states, idx, layers=self.layers)

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            cv = self._layer_cache_view_update(
                cv,
                idx,
                layer_outputs.cache_view,
                enabled=trace_layers,
                cache=past_key_values,
            )
            if output_router_logits and layer_outputs.router_logits is not None:
                all_router_logits += (layer_outputs.router_logits,)

            return hidden_states, cv, all_hidden_states, all_attentions, all_router_logits, idx + 1

        hidden_states, _, all_hidden_states, all_attentions, all_router_logits, _ = self.layers.scan(
            _layer_loop,
            (hidden_states, cache_views, all_hidden_states, all_attentions, all_router_logits, 0),
            trace=trace_layers,
            **self._layer_scan_kwargs("text"),
        )
        hidden_states = self.norm(hidden_states)

        return VLMCausalLMOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            past_key_values=past_key_values,
            router_logits=all_router_logits,
        )

    def get_encoder(self):
        """Talker is decoder-only — no separate encoder.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("This is a decoder-only model.")

    def get_decoder(self):
        """Return the Talker text decoder stack (``self``).

        Returns:
            spx.Module: ``self``. The first-group codec head is attached
            by the surrounding ``ForConditionalGeneration`` wrapper.
        """
        return self

    def get_lm_head(self):
        """Base Talker text model has no LM head.

        Raises:
            NotImplementedError: Always — use
                :class:`Qwen3OmniMoeTalkerForConditionalGeneration` to
                attach the codec head.
        """
        raise NotImplementedError("Base model does not have a language model head.")

    def get_embedding(self):
        """Return the codec token embedding table.

        Returns:
            spx.Module: The :class:`Embed` mapping codec quantizer
            indices to ``hidden_size``-dimensional vectors.
        """
        return self.codec_embedding


class Qwen3OmniMoeTalkerForConditionalGeneration(
    BaseConditionalGenerationModule[Qwen3OmniMoeTalkerModel, Qwen3OmniMoeTalkerConfig]  # type: ignore
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
        rngs: spx.Rngs,
    ):
        """Initialize full Talker model with text model, code predictor, and projections.

        Combines the talker text model, a code predictor for multi-group codec
        token generation, text/hidden projections, and a codec head for
        first-group token prediction.

        Args:
            config: Talker configuration with text and code predictor configs.
            dtype: Computation data type.
            param_dtype: Parameter storage data type.
            precision: JAX numerical precision.
            rngs: Random number generator state.
        """
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

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        thinker_hidden_states: Float[Array, "batch seq_len thinker_hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_router_logits: bool = False,
        apply_codec_head: bool = True,
        thinker_inputs_embeds: Float[Array, "batch seq_len thinker_hidden_dim"] | None = None,
        thinker_multimodal_mask: Bool[Array, "batch seq_len"] | None = None,
    ) -> MoeCausalLMOutput:
        """Forward pass through the full Talker model.

        Projects thinker hidden states (if provided), runs the talker text model,
        and optionally applies the codec head for first-group token logits.

        Args:
            input_ids: Codec token IDs [batch, seq_len].
            inputs_embeds: Pre-computed embeddings (overrides input_ids and thinker_hidden_states).
            thinker_hidden_states: Hidden states from the Thinker model to project.
            attention_mask: Attention mask for padding.
            mask_info: Mask information for efficient attention.
            position_ids: Position IDs for RoPE.
            mode: Runtime mode (train, decode, etc.).
            past_key_values: Cached key-value states for generation.
            cache_metadata: Cache metadata for efficient caching.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return all hidden states.
            output_router_logits: Whether to return MoE router logits.
            apply_codec_head: Whether to apply the codec head for logits.
            thinker_inputs_embeds: Thinker token embeddings, projected with text_projection.
            thinker_multimodal_mask: Select hidden_projection for multimodal tokens and
                text_projection for text tokens when both thinker inputs are provided.

        Returns:
            MoeCausalLMOutput with codec logits and optional intermediate outputs.
        """
        if inputs_embeds is None:
            if thinker_hidden_states is not None:
                inputs_embeds = self.hidden_projection(thinker_hidden_states)
            if thinker_inputs_embeds is not None:
                text_embeds = self.text_projection(thinker_inputs_embeds)
                if inputs_embeds is not None:
                    if thinker_multimodal_mask is None:
                        raise ValueError("thinker_multimodal_mask is required when both Thinker inputs are supplied.")
                    inputs_embeds = jnp.where(thinker_multimodal_mask[..., None], inputs_embeds, text_embeds)
                else:
                    inputs_embeds = text_embeds

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
        """Talker generator is decoder-only — no encoder.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("This is a decoder-only model.")

    def get_decoder(self):
        """Return the wrapped Talker text decoder.

        Returns:
            spx.Module: ``self.model.get_decoder()`` — the underlying
            :class:`Qwen3OmniMoeTalkerModel`.
        """
        return self.model.get_decoder()

    def get_lm_head(self):
        """Return the first-group codec acoustic head.

        The full Talker generator predicts the codec stack jointly:
        group 0 is produced by ``self.codec_head`` from the Talker's
        own hidden states, while groups ``1 .. num_code_groups - 1``
        are produced by ``self.code_predictor``.

        Returns:
            ColumnParallelLinear: The codec head projecting
            ``hidden_size`` to the codec vocabulary.
        """
        return self.codec_head

    def get_embedding(self):
        """Return the codec token embedding from the wrapped Talker.

        Returns:
            spx.Module: The codec :class:`Embed` table owned by
            :class:`Qwen3OmniMoeTalkerModel`.
        """
        return self.model.get_embedding()


class Qwen3OmniMoeCode2WavLayerScale(spx.Module):
    """Learnable per-channel scaling factor applied to residual branches.

    Implements the LayerScale trick (CaiT, He et al. 2021): after each
    sublayer the residual contribution is multiplied element-wise by a
    learned ``hidden_size``-shaped vector ``gamma``. ``gamma`` is
    initialised to a small constant so that early in training each
    residual branch is essentially identity, which stabilises training
    of deep vocoder stacks.
    """

    def __init__(
        self,
        dim: int,
        init_value: float = 1e-4,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize learnable per-channel scale parameter.

        Args:
            dim: Number of channels to scale.
            init_value: Initial scale value for each channel.
            dtype: Computation data type.
            param_dtype: Parameter storage data type.
            rngs: Random number generator state.
        """
        self.weight = spx.Parameter(
            jnp.full((dim,), init_value, dtype=param_dtype),
        )
        self.dtype = dtype

    def forward(self, x: Array) -> Array:
        """Apply per-channel scaling to the input tensor.

        Args:
            x: Input tensor to scale.

        Returns:
            Scaled tensor with same shape as input.
        """
        return x * self.weight.value.astype(self.dtype)


class Qwen3OmniMoeCode2WavMLP(Qwen3OmniMoeTextMLP):
    """SwiGLU feed-forward block used inside the Code2Wav transformer.

    Implements the standard ``down_proj(act(gate_proj(x)) * up_proj(x))``
    gated activation: a ``ColumnParallelLinear`` produces the gate and a
    parallel ``ColumnParallelLinear`` produces the up-projection; the
    elementwise product is then projected back to ``hidden_size`` via a
    ``RowParallelLinear``. The activation function is selected by
    ``config.hidden_act`` and ranges over ``ACT2FN``.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeCode2WavConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize gated MLP with gate, up, and down projections.

        Args:
            config: Code2Wav configuration.
            dtype: Computation data type.
            param_dtype: Parameter storage data type.
            precision: JAX numerical precision.
            rngs: Random number generator state.
        """
        super().__init__(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            # Preserve Code2Wav's default ParallelLinear initializer, not the text normal initializer.
            kernel_init=jax.nn.initializers.lecun_normal(),
        )


def _code2wav_position_embeddings(config, hidden_states, position_ids=None):
    """Build full-width split-half RoPE tables for the Code2Wav sequence."""
    if position_ids is None:
        position_ids = jnp.broadcast_to(jnp.arange(hidden_states.shape[1]), hidden_states.shape[:2])
    inv_freq = config.rope_theta ** (-jnp.arange(0, config.head_dim, 2, dtype=jnp.float32) / config.head_dim)
    phases = position_ids.astype(jnp.float32)[..., None] * inv_freq
    phases = jnp.concatenate((phases, phases), axis=-1)
    return jnp.cos(phases), jnp.sin(phases)


class Qwen3OmniMoeCode2WavAttention(spx.Module):
    """Causal multi-head attention with sliding-window mask for Code2Wav.

    Specialised attention block used by the Code2Wav vocoder: the same
    QKVO projection pattern as the text decoder but with a fixed
    ``config.sliding_window`` causal mask, which limits each query to a
    bounded local context. This bounded attention preserves the
    locality required for high-fidelity waveform synthesis while
    keeping the per-step cost independent of the total decoded
    duration.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeCode2WavConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize sliding window attention with Q/K/V/O projections.

        Uses causal attention with optional sliding window masking for
        localized context in audio waveform generation.

        Args:
            config: Code2Wav configuration.
            layer_idx: Index of this layer in the stack.
            dtype: Computation data type.
            param_dtype: Parameter storage data type.
            precision: JAX numerical precision.
            rngs: Random number generator state.
        """
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.sliding_window = config.sliding_window

        q_size = config.num_attention_heads * config.head_dim
        kv_size = config.num_key_value_heads * config.head_dim
        qkv_layout = dense_qkv_layout(q_size, kv_size)
        self.qkv_proj = ColumnParallelLinear(
            config.hidden_size,
            qkv_layout.segment_sizes,
            use_bias=config.attention_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layout=qkv_layout,
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

    @property
    def reform_param(self):
        """Expose HF split-QKV checkpoint conversion on the projection owner."""
        return self.qkv_proj.build_reform_param("qkv_proj", config=self.config)

    def forward(
        self,
        hidden_states: Array,
        attention_mask: Array | None = None,
        position_ids: Array | None = None,
        position_embeddings: tuple[Array, Array] | None = None,
    ) -> Array:
        """Apply sliding window causal attention.

        Args:
            hidden_states: Input hidden states [batch, seq_len, hidden_dim].
            attention_mask: Optional attention mask.
            position_ids: Optional per-token rotary position IDs.
            position_embeddings: Optional shared cosine/sine tables from the transformer trunk.

        Returns:
            Attention output with same shape as input hidden states.
        """
        batch_size, seq_len, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        q, k, v = self.qkv_proj.split(qkv, config=self.config)

        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        q = apply_logical_sharding(
            q,
            dynamic_axes=common_types.AttnQSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        k = apply_logical_sharding(
            k,
            dynamic_axes=common_types.AttnKVSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        v = apply_logical_sharding(
            v,
            dynamic_axes=common_types.AttnKVSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )

        cos, sin = (
            _code2wav_position_embeddings(self.config, hidden_states, position_ids)
            if position_embeddings is None
            else position_embeddings
        )
        q = apply_rope_split_half(q, cos, sin)
        k = apply_rope_split_half(k, cos, sin)

        if self.sliding_window is not None:
            causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
            window_mask = jnp.triu(jnp.ones((seq_len, seq_len), dtype=jnp.bool_), k=-self.sliding_window + 1)
            mask = (causal_mask & window_mask)[None, None, :, :]
            if attention_mask is not None:
                if attention_mask.ndim == 2:
                    attention_mask = attention_mask[:, None, None, :]
                mask = mask & attention_mask.astype(jnp.bool_)
            attention_mask = jnp.broadcast_to(mask, (batch_size, mask.shape[1], seq_len, seq_len))

        attn_output = self.attention.forward(
            query_states=q,
            key_states=k,
            value_states=v,
            mode=common_types.MODE_TRAIN,
            mask_info=MaskInfo.from_attention_mask(attention_mask) if attention_mask is not None else None,
            causal=True,
        ).attention_outputs

        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        attn_output = apply_logical_sharding(
            attn_output,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        attn_output = self.o_proj(attn_output)
        attn_output = apply_logical_sharding(
            attn_output,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        return typing.cast(Array, attn_output)


class Qwen3OmniMoeCode2WavTransformerLayer(spx.Module):
    """Pre-norm transformer block for the Code2Wav vocoder.

    Stacks RMSNorm + sliding-window self-attention + LayerScale, then a
    second RMSNorm + SwiGLU MLP + LayerScale. Each residual branch is
    multiplied by its own learnable per-channel ``gamma`` so the early
    block is near-identity, which is critical for stable training of
    the deep stack used to predict raw waveform samples from codec
    tokens.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeCode2WavConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize transformer layer with attention, MLP, and LayerScale branches.

        Args:
            config: Code2Wav configuration.
            layer_idx: Index of this layer in the stack.
            dtype: Computation data type.
            param_dtype: Parameter storage data type.
            precision: JAX numerical precision.
            rngs: Random number generator state.
        """
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

    def forward(
        self,
        hidden_states: Array,
        attention_mask: Array | None = None,
        position_ids: Array | None = None,
        position_embeddings: tuple[Array, Array] | None = None,
    ) -> Array:
        """Run pre-norm transformer layer with LayerScale on both residual branches.

        Args:
            hidden_states: Input hidden states [batch, seq_len, hidden_dim].
            attention_mask: Optional attention mask.
            position_ids: Optional position IDs.

        Returns:
            Updated hidden states with same shape as input.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(hidden_states, attention_mask, position_ids, position_embeddings)
        hidden_states = residual + self.self_attn_layer_scale(attn_output)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.config.use_scan_mlp:
            mlp_output = blockwise_ffn(
                self.mlp,
                hidden_states,
                self.config.scan_mlp_chunk_size,
            )
        else:
            mlp_output = self.mlp(hidden_states)
        hidden_states = residual + self.mlp_layer_scale(mlp_output)

        return hidden_states


class Qwen3OmniMoeCode2WavTransformerModel(EasyDeLLayerStackMixin, spx.Module):
    """Stacked Code2Wav transformer trunk consumed by the upsampler.

    Owns the list of :class:`Qwen3OmniMoeCode2WavTransformerLayer`
    blocks and the final RMSNorm. The output is a sequence of
    ``hidden_size``-dimensional vectors that downstream upsampling
    convolutions transform into raw audio samples; no LM head is
    attached because Code2Wav synthesises waveforms directly rather
    than predicting discrete tokens.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeCode2WavConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize transformer model with stacked layers and final norm.

        Args:
            config: Code2Wav configuration.
            dtype: Computation data type.
            param_dtype: Parameter storage data type.
            precision: JAX numerical precision.
            rngs: Random number generator state.
        """
        self.config = config

        self.layers = nn.ModuleList([])
        for layer_idx in range(config.num_hidden_layers):
            with self.assign_layer_stage(layer_idx, total_layers=config.num_hidden_layers):
                self.layers.append(
                    Qwen3OmniMoeCode2WavTransformerLayer(
                        config=config,
                        layer_idx=layer_idx,
                        dtype=dtype,
                        param_dtype=param_dtype,
                        precision=precision,
                        rngs=rngs,
                    )
                )

        final_layer_idx = max(0, config.num_hidden_layers - 1)
        with self.assign_layer_stage(final_layer_idx, total_layers=config.num_hidden_layers):
            self.norm = RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

    def forward(
        self,
        inputs_embeds: Array,
        attention_mask: Array | None = None,
        position_ids: Array | None = None,
    ) -> BaseModelOutput:
        """Forward pass through all transformer layers followed by final norm.

        Args:
            inputs_embeds: Input embeddings [batch, seq_len, hidden_dim].
            attention_mask: Optional attention mask.
            position_ids: Optional position IDs.

        Returns:
            BaseModelOutput with normalized last hidden state.
        """
        hidden_states = inputs_embeds
        position_embeddings = _code2wav_position_embeddings(self.config, hidden_states, position_ids)

        def _layer_loop(layer, carry):
            """Apply a single code-predictor decoder layer inside the layer-stack scan.

            Body of ``self.layers.scan`` for the Talker code-predictor
            decoder; runs ``layer`` on the current hidden states with
            ``attention_mask`` and ``position_ids``, and returns the
            updated carry tuple.
            """
            hidden_states, idx = carry
            with self._layer_stage_context(idx, layers=self.layers):
                hidden_states = layer(hidden_states, attention_mask, position_ids, position_embeddings)
            hidden_states = self._mark_layer_stage_boundary(hidden_states, idx, layers=self.layers)

            return hidden_states, idx + 1

        hidden_states, _ = self.layers.scan(
            _layer_loop,
            (hidden_states, 0),
            trace=not self.config.scan_layers or self._pipeline_stage_count() > 1,
            **self._layer_scan_kwargs("text"),
        )
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
        rngs: spx.Rngs,
    ):
        """Initialize Code2Wav vocoder with codec embeddings and transformer.

        Uses offset-based indexing for a single shared embedding across all
        quantizers, followed by a pre-transformer for feature processing.

        Args:
            config: Code2Wav vocoder configuration.
            dtype: Computation data type.
            param_dtype: Parameter storage data type.
            precision: JAX numerical precision.
            rngs: Random number generator state.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        # Single embedding for all quantizers with offset-based indexing
        self.code_embedding = Embed(
            config.codebook_size * config.num_quantizers,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
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

    def forward(
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

        outputs = typing.cast(BaseModelOutput, self.pre_transformer(hidden_states, attention_mask))

        return outputs

    def get_encoder(self):
        """Code2Wav is decoder-only — no encoder.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Code2Wav is a decoder-only model.")

    def get_decoder(self):
        """Return ``self`` — Code2Wav owns its own pre-transformer body.

        Returns:
            spx.Module: ``self``.
        """
        return self

    def get_lm_head(self):
        """Code2Wav synthesises waveforms via upsampling, not via an LM head.

        Raises:
            NotImplementedError: Always — there is no token vocabulary at
                the output; the post-transformer upsampling blocks emit
                continuous audio samples directly.
        """
        raise NotImplementedError("Code2Wav uses upsampling blocks, not an LM head.")

    def get_embedding(self):
        """Return the codec token embedding consumed by the pre-transformer.

        Returns:
            spx.Module: The :class:`Embed` table indexed by quantizer
            tokens (after applying the per-quantizer offset stored in
            ``self.code_offset``).
        """
        return self.code_embedding


# The cumsum-gather merge is shared infrastructure; these families each had
# their own copy of it. `BaseVisionLanguageModule.merge_multimodal_embeddings`
# is the same algorithm, so bind the module-level name to it rather than
# restating it. Keeping the name means call sites (and importers) are unchanged.
merge_multimodal_embeddings = BaseVisionLanguageModule.merge_multimodal_embeddings


def _visual_deepstack_embeddings(
    visual: Qwen3OmniMoeVisionEncoder,
    input_ids: Array,
    pixel_values: Array | None,
    image_grid_thw: Array | None,
    image_max_grid_size: int | None,
    pixel_values_videos: Array | None,
    video_grid_thw: Array | None,
    video_max_grid_size: int | None,
    image_token_id: int,
    video_token_id: int,
    image_embeds: Array | None = None,
    video_embeds: Array | None = None,
    image_deepstack: list[Array] | None = None,
    video_deepstack: list[Array] | None = None,
) -> tuple[Array | None, Array | None, Array | None, list[Array] | None]:
    """Collect raw/cached visual features and align every intermediate residual."""
    image_mask = input_ids == image_token_id if image_embeds is not None else None
    video_mask = input_ids == video_token_id if video_embeds is not None else None
    if pixel_values is not None:
        if image_grid_thw is None:
            raise ValueError("`image_grid_thw` must be provided when `pixel_values` is not None.")
        if image_max_grid_size is None:
            image_max_grid_size = int(jnp.max(image_grid_thw[:, 1:]).item())
        image_embeds, image_deepstack = visual(pixel_values, image_grid_thw, image_max_grid_size)
        image_mask = input_ids == image_token_id
    if pixel_values_videos is not None:
        if video_grid_thw is None:
            raise ValueError("`video_grid_thw` must be provided when `pixel_values_videos` is not None.")
        if video_max_grid_size is None:
            video_max_grid_size = int(jnp.max(video_grid_thw[:, 1:]).item())
        video_embeds, video_deepstack = visual(pixel_values_videos, video_grid_thw, video_max_grid_size)
        video_mask = input_ids == video_token_id

    image_deepstack = [] if image_deepstack is None else image_deepstack
    video_deepstack = [] if video_deepstack is None else video_deepstack
    for features, residuals in ((image_embeds, image_deepstack), (video_embeds, video_deepstack)):
        if residuals and (features is None or any(layer.shape != features.shape for layer in residuals)):
            raise ValueError("Cached DeepStack residuals require matching visual embedding shapes.")
    if image_mask is not None and video_mask is not None:
        visual_mask = image_mask | video_mask
        # Use a fixed output width (from the vision towers) so this also
        # works when the thinker forward is traced under jax.jit.
        num_visual = image_embeds.shape[0] + video_embeds.shape[0]
        visual_indices = jnp.nonzero(visual_mask.reshape(-1), size=num_visual, fill_value=0)[0]
        image_slots = image_mask.reshape(-1)[visual_indices]
        video_slots = video_mask.reshape(-1)[visual_indices]
        deepstack = []
        for image_layer, video_layer in zip(image_deepstack, video_deepstack, strict=True):
            image_idx = jnp.where(image_slots, jnp.cumsum(image_slots) - 1, 0)
            video_idx = jnp.where(video_slots, jnp.cumsum(video_slots) - 1, 0)
            merged = jnp.where(image_slots[:, None], image_layer[image_idx], 0)
            merged = jnp.where(video_slots[:, None], video_layer[video_idx], merged)
            deepstack.append(merged)
        return image_embeds, video_embeds, visual_mask, deepstack
    if image_mask is not None:
        return image_embeds, None, image_mask, image_deepstack
    if video_mask is not None:
        return None, video_embeds, video_mask, video_deepstack
    return None, None, None, None


def _inject_visual_deepstack(hidden_states: Array, visual_mask: Array, visual_embeds: Array) -> Array:
    """Add compact or token-aligned serving residuals at visual positions."""
    flattened = hidden_states.reshape(-1, hidden_states.shape[-1])
    mask = visual_mask.reshape(-1)
    residuals = visual_embeds.reshape(-1, visual_embeds.shape[-1])
    if residuals.shape[0] == flattened.shape[0]:
        aligned = residuals
    else:
        index = jnp.where(mask, jnp.cumsum(mask) - 1, 0)
        aligned = residuals[index]
    updated = jnp.where(mask[:, None], flattened + aligned.astype(flattened.dtype), flattened)
    return updated.reshape(hidden_states.shape)


class Qwen3OmniMoeThinkerTextModel(EasyDeLBaseModule):
    """Text decoder model for Qwen3OmniMoe Thinker.

    A transformer decoder with MoE layers for text generation. Contains
    only the text processing components (embed_tokens, layers, norm).
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
        rngs: spx.Rngs,
    ):
        """Initialize Thinker text model.

        Args:
            config (Qwen3OmniMoeTextConfig): Text decoder configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        with self.assign_layer_stage(0, total_layers=config.num_hidden_layers):
            self.embed_tokens = Embed(
                config.vocab_size,
                config.hidden_size,
                embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

        remat_layer_block = auto_remat(
            Qwen3OmniMoeTextDecoderLayer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.layers = nn.ModuleList([])
        for layer_idx in range(config.num_hidden_layers):
            with self.assign_layer_stage(layer_idx, total_layers=config.num_hidden_layers):
                self.layers.append(
                    remat_layer_block(
                        config=config,
                        layer_idx=layer_idx,
                        dtype=dtype,
                        param_dtype=param_dtype,
                        precision=precision,
                        rngs=rngs,
                    )
                )

        final_layer_idx = max(0, config.num_hidden_layers - 1)
        with self.assign_layer_stage(final_layer_idx, total_layers=config.num_hidden_layers):
            self.norm = RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        visual_pos_masks: Array | None = None,
        deepstack_visual_embeds: list[Array] | None = None,
    ) -> VLMCausalLMOutput:
        """Forward pass through text decoder.

        Args:
            input_ids: Input token IDs [batch, seq_len].
            inputs_embeds: Input embeddings if provided instead of input_ids.
            attention_mask: Attention mask for padding.
            mask_info: Mask information for efficient attention.
            position_ids: Position IDs for tokens.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return all hidden states.
            output_router_logits: Whether to return MoE router logits.
            mode: Runtime mode (train, decode, etc.).
            past_key_values: Cached key-value states for generation.
            cache_metadata: Cache metadata for efficient caching.
            visual_pos_masks: Positions occupied by image or video tokens.
            deepstack_visual_embeds: Intermediate visual features to inject after early layers.

        Returns:
            VLMCausalLMOutput: Model outputs including hidden states, attentions,
                and router logits if requested.
        """
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
        if deepstack_visual_embeds is not None and visual_pos_masks is None:
            raise ValueError("`visual_pos_masks` must be provided when `deepstack_visual_embeds` is not None.")

        views = past_key_values.views if past_key_values is not None else None
        has_cache_views = views is not None and any(v is not None for v in views)
        needs_trace_cache = mode == common_types.MODE_DECODE or has_cache_views
        # Router-logit aggregation grows a Python tuple per layer, which
        # lax.scan cannot carry — collecting it forces the trace path too.
        trace_layers = self._layer_scan_trace(
            False,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            cache_views=views,
            extra=needs_trace_cache or bool(output_router_logits) or bool(deepstack_visual_embeds),
        )
        cache_views = views if trace_layers else None
        # Hoisted out of the loop body: the caching property must not be
        # first materialized inside the lax.scan trace (tracer leak).
        frequencies = self.frequencies

        def _layer_loop(layer, carry):
            """Apply a single Thinker text MoE decoder layer inside the layer-stack scan.

            Body of ``self.layers.scan`` for the Thinker text decoder;
            runs ``layer`` on the current hidden states, writes the
            returned KV cache view back, optionally accumulates per-layer
            hidden states, self-attention weights, and MoE router logits,
            and returns the updated carry tuple.
            """
            hidden_states, cv, all_hidden_states, all_self_attentions, all_router_logits, layer_idx = carry
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            with self._layer_stage_context(layer_idx, layers=self.layers):
                layer_outputs = layer(
                    hidden_states=hidden_states,
                    mask_info=mask_info,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    mode=mode,
                    cache_view=self._layer_cache_view_at(cv, layer_idx, enabled=trace_layers, cache=past_key_values),
                    cache_metadata=cache_metadata,
                    frequencies=frequencies,
                )
            hidden_states = layer_outputs.hidden_states
            # Truthiness (not ``is not None``) so an empty list never reaches the traced index
            # comparison: an empty list lets the layer stack take the lax.scan path.
            if deepstack_visual_embeds and layer_idx < len(deepstack_visual_embeds):
                hidden_states = _inject_visual_deepstack(
                    hidden_states, visual_pos_masks, deepstack_visual_embeds[layer_idx]
                )
            hidden_states = self._mark_layer_stage_boundary(hidden_states, layer_idx, layers=self.layers)

            if output_attentions:
                all_self_attentions += (layer_outputs.attention_weight,)
            cv = self._layer_cache_view_update(
                cv,
                layer_idx,
                layer_outputs.cache_view,
                enabled=trace_layers,
                cache=past_key_values,
            )
            if output_router_logits and layer_outputs.router_logits is not None:
                all_router_logits += (layer_outputs.router_logits,)

            return hidden_states, cv, all_hidden_states, all_self_attentions, all_router_logits, layer_idx + 1

        hidden_states, _, all_hidden_states, all_self_attentions, all_router_logits, _ = self.layers.scan(
            _layer_loop,
            (hidden_states, cache_views, all_hidden_states, all_self_attentions, all_router_logits, 0),
            trace=trace_layers,
            **self._layer_scan_kwargs("text"),
        )
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
        """Return the text token embedding (HuggingFace-compatible alias).

        Returns:
            spx.Module: The :attr:`embed_tokens` table.
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """Replace the text token embedding in-place.

        Args:
            value: Replacement embedding module — must accept
                integer ``input_ids`` of shape ``(batch, seq_len)`` and
                produce hidden states of shape
                ``(batch, seq_len, hidden_size)``.
        """
        self.embed_tokens = value

    def get_embedding(self):
        """Return the text token embedding (EasyDeL-style alias).

        Returns:
            spx.Module: Same module as :meth:`get_input_embeddings`;
            duplicated to satisfy both the EasyDeL and HF naming
            conventions used elsewhere in the codebase.
        """
        return self.embed_tokens


@register_module(TaskType.BASE_MODULE, config=Qwen3OmniMoeThinkerConfig, model_type="qwen3_omni_moe")
class Qwen3OmniMoeModel(EasyDeLBaseModule):
    """Base Qwen3OmniMoe Thinker model combining audio, vision, and text.

    A multimodal encoder-decoder model that processes:
    - Audio inputs via mel-spectrogram encoding
    - Visual inputs (images/videos) via patch embedding and transformer
    - Text inputs via token embedding

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
        rngs: spx.Rngs,
    ):
        """Initialize base Qwen3OmniMoe model.

        Args:
            config (Qwen3OmniMoeThinkerConfig): Thinker configuration with audio,
                vision, and text configurations.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
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

        with self.assign_layer_stage(0, total_layers=text_config.num_hidden_layers):
            self.embed_tokens = Embed(
                text_config.vocab_size,
                text_config.hidden_size,
                embedding_init=jax.nn.initializers.normal(stddev=text_config.initializer_range),
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

        remat_layer_block = auto_remat(
            Qwen3OmniMoeTextDecoderLayer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.layers = nn.ModuleList([])
        for layer_idx in range(text_config.num_hidden_layers):
            with self.assign_layer_stage(layer_idx, total_layers=text_config.num_hidden_layers):
                self.layers.append(
                    remat_layer_block(
                        config=text_config,
                        layer_idx=layer_idx,
                        dtype=dtype,
                        param_dtype=param_dtype,
                        precision=precision,
                        rngs=rngs,
                    )
                )

        final_layer_idx = max(0, text_config.num_hidden_layers - 1)
        with self.assign_layer_stage(final_layer_idx, total_layers=text_config.num_hidden_layers):
            self.norm = RMSNorm(
                text_config.hidden_size,
                eps=text_config.rms_norm_eps,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

    def compute_embedding(
        self,
        input_ids: Int[Array, "batch seq_len"] | None,
        *,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        input_features: Float[Array, "batch mel_bins time"] | None = None,
        feature_lens: Int[Array, "batch"] | None = None,
        feature_attention_mask: Bool[Array, "batch time"] | None = None,
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
        """Compute embeddings by merging text, audio, image, and video inputs.

        Processes token IDs into text embeddings, then merges in any multimodal
        embeddings (audio, image, video) at their respective placeholder positions.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length).
            inputs_embeds: Pre-computed text embeddings. If None, computed from input_ids.
            input_features: Raw audio features (mel-spectrogram) to encode.
            feature_lens: Valid mel-frame count for each audio sample.
            feature_attention_mask: Padding mask over mel frames.
            audio_embeds: Pre-computed audio embeddings to merge.
            pixel_values: Raw image pixel values to encode via vision encoder.
            image_grid_thw: Grid dimensions (temporal, height, width) for each image.
            image_max_grid_size: Maximum grid size for image position embeddings.
            image_embeds: Pre-computed image embeddings to merge.
            pixel_values_videos: Raw video pixel values to encode via vision encoder.
            video_grid_thw: Grid dimensions (temporal, height, width) for each video.
            video_max_grid_size: Maximum grid size for video position embeddings.
            video_embeds: Pre-computed video embeddings to merge.
            **kwargs: Additional keyword arguments.

        Returns:
            Array: Merged embeddings of shape (batch_size, sequence_length, hidden_size).

        Raises:
            ValueError: If input_ids is None when multimodal inputs are provided.
        """
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

        audio_valid_length = None
        if audio_embeds is None and input_features is not None:
            audio_embeds, audio_valid_length = _encode_audio_for_merging(
                self.audio, input_features, feature_lens, feature_attention_mask
            )

        if audio_embeds is not None:
            if audio_embeds.ndim == 3:
                audio_embeds = audio_embeds.reshape(-1, self.config.text_config.hidden_size)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                audio_embeds.astype(inputs_embeds.dtype),
                self.config.audio_token_id,
                multimodal_embeddings_valid_length=audio_valid_length,
            )

        if image_embeds is None and pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("`image_grid_thw` must be provided when `pixel_values` is not None.")
            if image_max_grid_size is None:
                image_max_grid_size = int(jnp.max(image_grid_thw[:, 1:]).item())
            image_embeds, _ = self.visual(pixel_values, image_grid_thw, image_max_grid_size)

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
            video_embeds, _ = self.visual(pixel_values_videos, video_grid_thw, video_max_grid_size)

        if video_embeds is not None:
            if video_embeds.ndim > 2:
                video_embeds = video_embeds.reshape(-1, video_embeds.shape[-1])
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                video_embeds.astype(inputs_embeds.dtype),
                self.config.video_token_id,
            )

        return inputs_embeds  # pyright: ignore[reportReturnType]

    def forward(
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
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        feature_lens: Int[Array, "batch"] | None = None,
        feature_attention_mask: Bool[Array, "batch time"] | None = None,
    ) -> VLMCausalLMOutput:
        """Forward pass through the base Qwen3OmniMoe model with multimodal embedding fusion.

        Computes embeddings from text, audio, image, and video inputs, then
        processes them through the MoE transformer decoder.

        Args:
            input_ids: Input token IDs [batch, seq_len].
            inputs_embeds: Pre-computed embeddings (overrides input_ids).
            attention_mask: Attention mask for padding.
            mask_info: Mask information for efficient attention.
            position_ids: Position IDs for RoPE.
            input_features: Audio mel-spectrogram features.
            feature_lens: Valid mel-frame count for each audio sample.
            feature_attention_mask: Padding mask over mel frames.
            pixel_values: Image pixel values.
            image_grid_thw: Grid dimensions for image patches.
            image_max_grid_size: Maximum grid size for images.
            pixel_values_videos: Video pixel values.
            video_grid_thw: Grid dimensions for video patches.
            video_max_grid_size: Maximum grid size for videos.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return all hidden states.
            output_router_logits: Whether to return MoE router logits.
            mode: Runtime mode (train, decode, etc.).
            past_key_values: Cached key-value states for generation.
            cache_metadata: Cache metadata for efficient caching.

        Returns:
            VLMCausalLMOutput with hidden states and optional router logits.
        """
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
        if input_ids is None and (pixel_values is not None or pixel_values_videos is not None):
            raise ValueError("`input_ids` must be provided to merge multimodal embeddings.")
        image_embeds, video_embeds, visual_pos_masks, deepstack_visual_embeds = _visual_deepstack_embeddings(
            self.visual,
            input_ids,
            pixel_values,
            image_grid_thw,
            image_max_grid_size,
            pixel_values_videos,
            video_grid_thw,
            video_max_grid_size,
            self.config.image_token_id,
            self.config.video_token_id,
        )

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
                feature_lens=feature_lens,
                feature_attention_mask=feature_attention_mask,
                image_embeds=image_embeds,
                video_embeds=video_embeds,
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
            partition_manager=text_config.runtime_sharding_resolver,
        )

        views = past_key_values.views if past_key_values is not None else None
        has_cache_views = views is not None and any(v is not None for v in views)
        needs_trace_cache = mode == common_types.MODE_DECODE or has_cache_views
        # Router-logit aggregation grows a Python tuple per layer, which
        # lax.scan cannot carry — collecting it forces the trace path too.
        trace_layers = self._layer_scan_trace(
            False,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            cache_views=views,
            extra=needs_trace_cache or bool(output_router_logits) or bool(deepstack_visual_embeds),
        )
        cache_views = views if trace_layers else None
        # Hoisted out of the loop body: the caching property must not be
        # first materialized inside the lax.scan trace (tracer leak).
        frequencies = self.frequencies

        def _layer_loop(block, carry):
            """Apply a single Talker MoE decoder layer inside the layer-stack scan.

            Body of ``self.layers.scan`` for the Talker text decoder;
            carries ``(hidden_states, cache_views, all_hidden_states,
            all_attentions, all_router_logits, layer_index)``, optionally
            accumulating per-layer hidden states, attention weights, and
            MoE router logits.
            """
            hidden_states, cv, all_hidden_states, all_attentions, all_router_logits, idx = carry
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            with self._layer_stage_context(idx, layers=self.layers):
                layer_outputs = block(
                    hidden_states=hidden_states,
                    mask_info=mask_info,
                    position_ids=position_ids,
                    mode=mode,
                    cache_view=self._layer_cache_view_at(cv, idx, enabled=trace_layers, cache=past_key_values),
                    cache_metadata=cache_metadata,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    frequencies=frequencies,
                )
            hidden_states = layer_outputs.hidden_states
            # Truthiness so an empty list never compares against a traced scan index.
            if deepstack_visual_embeds and idx < len(deepstack_visual_embeds):
                hidden_states = _inject_visual_deepstack(hidden_states, visual_pos_masks, deepstack_visual_embeds[idx])
            hidden_states = self._mark_layer_stage_boundary(hidden_states, idx, layers=self.layers)

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            cv = self._layer_cache_view_update(
                cv,
                idx,
                layer_outputs.cache_view,
                enabled=trace_layers,
                cache=past_key_values,
            )
            if output_router_logits and layer_outputs.router_logits is not None:
                all_router_logits += (layer_outputs.router_logits,)

            return hidden_states, cv, all_hidden_states, all_attentions, all_router_logits, idx + 1

        hidden_states, _, all_hidden_states, all_attentions, all_router_logits, _ = self.layers.scan(
            _layer_loop,
            (hidden_states, cache_views, all_hidden_states, all_attentions, all_router_logits, 0),
            trace=trace_layers,
            **self._layer_scan_kwargs("text"),
        )
        hidden_states = checkpoint_name(self.norm(hidden_states), "model_output")

        return VLMCausalLMOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            past_key_values=past_key_values,
            router_logits=all_router_logits,
        )

    def get_encoder(self):
        """Base Qwen3-Omni model is decoder-only — no encoder.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("This is a decoder-only model.")

    def get_decoder(self):
        """Return ``self`` — the model is its own decoder.

        Returns:
            spx.Module: ``self``.
        """
        return self

    def get_lm_head(self):
        """Base text model has no LM head.

        Raises:
            NotImplementedError: Always — the LM head lives on
                :class:`Qwen3OmniMoeThinkerForConditionalGeneration`.
        """
        raise NotImplementedError("Base model does not have a language model head.")

    def get_embedding(self):
        """Return the text token embedding table.

        Returns:
            spx.Module: The :attr:`embed_tokens` module.
        """
        return self.embed_tokens


@register_module(
    TaskType.ANY_TO_ANY,
    config=Qwen3OmniMoeThinkerConfig,
    model_type="qwen3_omni_moe_thinker",
)
class Qwen3OmniMoeThinkerForConditionalGeneration(
    BaseVisionLanguageModule[Qwen3OmniMoeThinkerTextModel, Qwen3OmniMoeThinkerConfig]  # type: ignore
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
        rngs: spx.Rngs,
    ):
        """Initialize Qwen3OmniMoe Thinker for conditional generation.

        Args:
            config (Qwen3OmniMoeThinkerConfig): Thinker configuration with
                audio, vision, and text configurations.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
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
        self.lm_head = ParallelLinear(
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

    def get_rope_index(
        self,
        input_ids=None,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        use_audio_in_video=False,
        audio_seqlens=None,
        second_per_grids=None,
    ):
        """Build prompt positions, including timed video and interleaved audio.

        Generation preparation validates concrete metadata; raw-media compiled
        training uses a token scan without host transfers or dynamic shapes.
        """
        traced = any(
            isinstance(value, jax.core.Tracer)
            for value in jax.tree.leaves((input_ids, image_grid_thw, video_grid_thw, attention_mask, second_per_grids))
        )
        build_positions = build_omni_rope_positions_traced if traced else build_omni_rope_positions
        positions, deltas = build_positions(
            self.config,
            input_ids,
            image_grid_thw,
            video_grid_thw,
            attention_mask,
            use_audio_in_video,
            audio_seqlens,
            second_per_grids,
        )
        return jnp.asarray(positions), jnp.asarray(deltas)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        max_length,
        pad_token_id,
        starts=None,
        pixel_values=None,
        attention_mask=None,
        **kwargs,
    ):
        """Prepare the text cache and Omni's multimodal positional metadata."""
        model_inputs = self.base_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            max_length=max_length,
            pad_token_id=pad_token_id,
            starts=starts,
            attention_mask=attention_mask,
        )
        model_inputs.update({key: value for key, value in kwargs.items() if value is not None})
        model_inputs["pixel_values"] = pixel_values
        if kwargs.get("position_ids") is None:
            audio_seqlens = kwargs.get("audio_seqlens", kwargs.get("feature_lens"))
            if audio_seqlens is None and kwargs.get("feature_attention_mask") is not None:
                audio_seqlens = jnp.sum(kwargs["feature_attention_mask"], axis=-1)
            model_inputs["position_ids"], model_inputs["rope_deltas"] = self.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=kwargs.get("image_grid_thw"),
                video_grid_thw=kwargs.get("video_grid_thw"),
                attention_mask=attention_mask,
                use_audio_in_video=kwargs.get("use_audio_in_video", False),
                audio_seqlens=audio_seqlens,
                second_per_grids=kwargs.get("second_per_grids"),
            )
        return model_inputs

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        """Drop already-encoded audio along with vision tensors after prefill."""
        model_kwargs = super().update_inputs_for_generation(model_outputs, model_kwargs)
        for name in ("input_features", "feature_lens", "feature_attention_mask", "audio_seqlens", "second_per_grids"):
            model_kwargs.pop(name, None)
        return model_kwargs

    def compute_embedding_with_info(
        self,
        input_ids,
        *,
        inputs_embeds=None,
        pixel_values=None,
        image_grid_thw=None,
        image_max_grid_size=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        video_max_grid_size=None,
        image_embeds=None,
        video_embeds=None,
        deepstack_image_embeds=None,
        deepstack_video_embeds=None,
        attention_mask=None,
        position_ids=None,
        rope_deltas=None,
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
        use_audio_in_video=False,
        audio_seqlens=None,
        second_per_grids=None,
        **kwargs,
    ):
        """Return merged embeddings plus every residual and position needed for replay.

        Raw vision towers run once. Callers can pass the returned position IDs,
        visual mask and DeepStack residuals to a forward with only inputs_embeds.
        The legacy compute_embedding/feature-getter array APIs are unchanged.
        """
        if input_ids is None and any(
            value is not None
            for value in (
                pixel_values,
                pixel_values_videos,
                image_embeds,
                video_embeds,
            )
        ):
            raise ValueError("`input_ids` must be provided to merge multimodal embeddings.")
        images, videos, raw_mask, raw_deepstack = _visual_deepstack_embeddings(
            self.visual,
            input_ids,
            pixel_values,
            image_grid_thw,
            image_max_grid_size,
            pixel_values_videos,
            video_grid_thw,
            video_max_grid_size,
            self.config.image_token_id,
            self.config.video_token_id,
            image_embeds=image_embeds,
            video_embeds=video_embeds,
            image_deepstack=deepstack_image_embeds,
            video_deepstack=deepstack_video_embeds,
        )
        if images is not None:
            kwargs["image_embeds"] = images
        if videos is not None:
            kwargs["video_embeds"] = videos
        embeddings = self.compute_embedding(input_ids, inputs_embeds=inputs_embeds, **kwargs)
        if raw_mask is not None:
            visual_pos_masks, deepstack_visual_embeds = raw_mask, raw_deepstack
        if position_ids is None and input_ids is not None:
            audio_seqlens = kwargs.get("feature_lens") if audio_seqlens is None else audio_seqlens
            if audio_seqlens is None and kwargs.get("feature_attention_mask") is not None:
                audio_seqlens = jnp.sum(kwargs["feature_attention_mask"], axis=-1)
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask,
                use_audio_in_video,
                audio_seqlens,
                second_per_grids,
            )
        if raw_mask is not None and raw_deepstack:
            compact_index = jnp.cumsum(raw_mask.reshape(-1)) - 1
            if images is not None:
                slots = jnp.nonzero((input_ids == self.image_token_id).reshape(-1), size=images.shape[0])[0]
                deepstack_image_embeds = [layer[compact_index[slots]] for layer in raw_deepstack]
            if videos is not None:
                slots = jnp.nonzero((input_ids == self.video_token_id).reshape(-1), size=videos.shape[0])[0]
                deepstack_video_embeds = [layer[compact_index[slots]] for layer in raw_deepstack]
        return embeddings, EmbeddingInfo(
            position_ids=position_ids,
            rope_deltas=rope_deltas,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            deepstack_image_embeds=deepstack_image_embeds,
            deepstack_video_embeds=deepstack_video_embeds,
        )

    @property
    def audio(self):
        """HuggingFace-compatible alias for the audio encoder.

        Some upstream configs reference ``model.audio`` rather than the
        EasyDeL-canonical ``model.audio_tower``; this property forwards
        to the same underlying :class:`Qwen3OmniMoeAudioEncoder` so both
        naming conventions resolve identically.

        Returns:
            spx.Module: ``self.audio_tower``.
        """
        return self.audio_tower

    def compute_embedding(
        self,
        input_ids: Int[Array, "batch seq_len"] | None,
        *,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        input_features: Float[Array, "batch mel_bins time"] | None = None,
        feature_lens: Int[Array, "batch"] | None = None,
        feature_attention_mask: Bool[Array, "batch time"] | None = None,
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
        """Compute embeddings by merging text, audio, image, and video inputs.

        Delegates text embedding to the underlying text model, then merges
        multimodal embeddings (audio, image, video) at placeholder positions
        using the top-level audio_tower and visual encoders.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length).
            inputs_embeds: Pre-computed text embeddings. If None, computed from input_ids.
            input_features: Raw audio features (mel-spectrogram) to encode.
            feature_lens: Valid mel-frame count for each audio sample.
            feature_attention_mask: Padding mask over mel frames.
            audio_embeds: Pre-computed audio embeddings to merge.
            pixel_values: Raw image pixel values to encode via vision encoder.
            image_grid_thw: Grid dimensions (temporal, height, width) for each image.
            image_max_grid_size: Maximum grid size for image position embeddings.
            image_embeds: Pre-computed image embeddings to merge.
            pixel_values_videos: Raw video pixel values to encode via vision encoder.
            video_grid_thw: Grid dimensions (temporal, height, width) for each video.
            video_max_grid_size: Maximum grid size for video position embeddings.
            video_embeds: Pre-computed video embeddings to merge.
            **kwargs: Additional keyword arguments.

        Returns:
            Array: Merged embeddings of shape (batch_size, sequence_length, hidden_size).

        Raises:
            ValueError: If input_ids is None when multimodal inputs are provided.
        """
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

        audio_valid_length = None
        if audio_embeds is None and input_features is not None:
            audio_embeds, audio_valid_length = _encode_audio_for_merging(
                self.audio_tower, input_features, feature_lens, feature_attention_mask
            )

        if audio_embeds is not None:
            if audio_embeds.ndim == 3:
                audio_embeds = audio_embeds.reshape(-1, self.config.text_config.hidden_size)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                audio_embeds.astype(inputs_embeds.dtype),
                self.audio_token_id,
                multimodal_embeddings_valid_length=audio_valid_length,
            )

        if image_embeds is None and pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("`image_grid_thw` must be provided when `pixel_values` is not None.")
            if image_max_grid_size is None:
                image_max_grid_size = int(jnp.max(image_grid_thw[:, 1:]).item())
            image_embeds, _ = self.visual(pixel_values, image_grid_thw, image_max_grid_size)

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
            video_embeds, _ = self.visual(pixel_values_videos, video_grid_thw, video_max_grid_size)

        if video_embeds is not None:
            if video_embeds.ndim > 2:
                video_embeds = video_embeds.reshape(-1, video_embeds.shape[-1])
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                video_embeds.astype(inputs_embeds.dtype),
                self.video_token_id,
            )

        return inputs_embeds  # pyright: ignore[reportReturnType]

    def forward(
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
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        feature_lens: Int[Array, "batch"] | None = None,
        feature_attention_mask: Bool[Array, "batch time"] | None = None,
        visual_pos_masks: Array | None = None,
        deepstack_visual_embeds: list[Array] | None = None,
        rope_deltas: Array | None = None,
        use_audio_in_video: bool = False,
        audio_seqlens: Array | None = None,
        second_per_grids: Array | None = None,
    ) -> MoeCausalLMOutput:
        """Forward pass through the Thinker for multimodal conditional generation.

        Fuses audio, vision, and text embeddings, runs the text decoder, and
        optionally applies the LM head for next-token prediction.

        Args:
            input_ids: Input token IDs [batch, seq_len].
            inputs_embeds: Pre-computed embeddings (overrides input_ids).
            attention_mask: Attention mask for padding.
            mask_info: Mask information for efficient attention.
            position_ids: Position IDs for RoPE.
            input_features: Audio mel-spectrogram features.
            feature_lens: Valid mel-frame count for each audio sample.
            feature_attention_mask: Padding mask over mel frames.
            pixel_values: Image pixel values.
            image_grid_thw: Grid dimensions for image patches.
            image_max_grid_size: Maximum grid size for images.
            pixel_values_videos: Video pixel values.
            video_grid_thw: Grid dimensions for video patches.
            video_max_grid_size: Maximum grid size for videos.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return all hidden states.
            output_router_logits: Whether to return MoE router logits.
            mode: Runtime mode (train, decode, etc.).
            past_key_values: Cached key-value states for generation.
            cache_metadata: Cache metadata for efficient caching.
            apply_lm_head: Whether to apply the LM head for logits.

        Returns:
            MoeCausalLMOutput with logits (if apply_lm_head) and model outputs.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must provide either `input_ids` or `inputs_embeds`.")
        if input_ids is None and (pixel_values is not None or pixel_values_videos is not None):
            raise ValueError("`input_ids` must be provided to merge multimodal embeddings.")
        if (
            inputs_embeds is None
            or input_features is not None
            or pixel_values is not None
            or pixel_values_videos is not None
        ):
            if input_ids is None:
                raise ValueError("`input_ids` must be provided to compute or merge multimodal embeddings.")
            # Text-only compiled training does not need host multimodal parsing.
            if pixel_values is None and pixel_values_videos is None and input_features is None:
                inputs_embeds = self.compute_embedding(input_ids, inputs_embeds=inputs_embeds)
            else:
                inputs_embeds, info = self.compute_embedding_with_info(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    input_features=input_features,
                    feature_lens=feature_lens,
                    feature_attention_mask=feature_attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    image_max_grid_size=image_max_grid_size,
                    pixel_values_videos=pixel_values_videos,
                    video_grid_thw=video_grid_thw,
                    video_max_grid_size=video_max_grid_size,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    rope_deltas=rope_deltas,
                    visual_pos_masks=visual_pos_masks,
                    deepstack_visual_embeds=deepstack_visual_embeds,
                    use_audio_in_video=use_audio_in_video,
                    audio_seqlens=audio_seqlens,
                    second_per_grids=second_per_grids,
                )
                position_ids, rope_deltas = info.position_ids, info.rope_deltas
                visual_pos_masks, deepstack_visual_embeds = info.visual_pos_masks, info.deepstack_visual_embeds

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
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

        logits = None
        if apply_lm_head:
            logits = self.compute_lm_logits(self.prepare_lm_head_inputs(outputs.last_hidden_state))

        return MoeCausalLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
            last_hidden_state=outputs.last_hidden_state,
            router_logits=outputs.router_logits,
        )

    def get_encoder(self):
        """Return the vision encoder for compatibility with VLM utilities.

        VLMs that follow ``EncoderDecoderProtocol`` expose the visual
        tower as the encoder; the audio encoder is reachable separately
        via :attr:`audio_tower` / :attr:`audio`.

        Returns:
            spx.Module: The :attr:`visual` :class:`Qwen3OmniMoeVisionEncoder`.
        """
        return self.visual

    def get_decoder(self):
        """Return the text decoder backbone.

        Returns:
            spx.Module: The wrapped
            :class:`Qwen3OmniMoeThinkerTextModel` (a stack of MoE
            decoder layers).
        """
        return self.model

    def get_embedding(self):
        """Return the text token embedding from the wrapped text model.

        Returns:
            spx.Module: ``self.model.embed_tokens``.
        """
        return self.model.embed_tokens

    def get_input_embeddings(self):
        """HuggingFace-compatible alias of :meth:`get_embedding`.

        Returns:
            spx.Module: ``self.model.embed_tokens``.
        """
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """Replace the text token embedding in-place.

        Args:
            value: Replacement embedding module accepting ``input_ids``
                of shape ``(batch, seq_len)``.
        """
        self.model.embed_tokens = value

    def get_lm_head(self):
        """Return the text-side LM head.

        The Thinker emits text logits via :attr:`lm_head` even though
        the inputs may include audio and visual tokens; speech tokens
        are produced by the separate Talker / Code2Wav stack.

        Returns:
            ParallelLinear: The text LM head.
        """
        return self.lm_head

    def get_vision_tower(self) -> spx.Module:
        """Return the vision encoder tower (VLM protocol method).

        Returns:
            spx.Module: ``self.visual`` (a
            :class:`Qwen3OmniMoeVisionEncoder`).
        """
        return self.visual

    def get_language_model(self) -> spx.Module:
        """Return the text decoder (VLM protocol method).

        Returns:
            spx.Module: ``self.model``.
        """
        return self.model

    def get_image_features(
        self,
        pixel_values: Float[Array, "num_patches channels height width"],
        image_grid_thw: Int[Array, "num_images 3"] | None = None,
        return_deepstack: bool = False,
        **kwargs,
    ) -> Float[Array, "num_patches hidden"] | tuple[tuple[Array, ...], list[Array]]:
        """Extract image features from the vision encoder.

        Args:
            pixel_values: Input pixel values for the images.
            image_grid_thw: Grid dimensions (temporal, height, width) for each image.
            return_deepstack: Return ((merged_features,), residuals) instead of only features.
            **kwargs: Additional arguments passed to the vision encoder.

        Returns:
            Image features, or the features and all intermediate residuals when requested.
        """
        max_grid_size = kwargs.get("image_max_grid_size")
        if max_grid_size is None:
            max_grid_size = int(jnp.max(image_grid_thw[:, 1:]).item())
        merged, deepstack = self.visual(pixel_values, image_grid_thw, max_grid_size)
        return ((merged,), deepstack) if return_deepstack else merged

    def get_video_features(
        self,
        pixel_values_videos: Float[Array, "num_patches channels height width"],
        video_grid_thw: Int[Array, "num_videos 3"] | None = None,
        return_deepstack: bool = False,
        **kwargs,
    ) -> Float[Array, "num_patches hidden"] | tuple[tuple[Array, ...], list[Array]]:
        """Extract video features from the vision encoder.

        Args:
            pixel_values_videos: Input pixel values for the videos.
            video_grid_thw: Grid dimensions (temporal, height, width) for each video.
            return_deepstack: Return ((merged_features,), residuals) instead of only features.
            **kwargs: Additional arguments passed to the vision encoder.

        Returns:
            Video features, or the features and all intermediate residuals when requested.
        """
        max_grid_size = kwargs.get("video_max_grid_size")
        if max_grid_size is None:
            max_grid_size = int(jnp.max(video_grid_thw[:, 1:]).item())
        merged, deepstack = self.visual(pixel_values_videos, video_grid_thw, max_grid_size)
        return ((merged,), deepstack) if return_deepstack else merged

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
        audio_outputs = self.audio_tower(input_features, **kwargs)
        return audio_outputs.last_hidden_state


@register_module(TaskType.ANY_TO_ANY, config=Qwen3OmniMoeConfig, model_type="qwen3_omni_moe")
class Qwen3OmniMoeForConditionalGeneration(
    BaseConditionalGenerationModule[Qwen3OmniMoeThinkerForConditionalGeneration, Qwen3OmniMoeConfig]  # type: ignore
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
        rngs: spx.Rngs,
    ):
        """Initialize full Qwen3OmniMoe model with Thinker, Talker, and Code2Wav.

        Args:
            config (Qwen3OmniMoeConfig): Full configuration including thinker,
                talker, and code2wav configurations.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
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
        """Forward to the Thinker's vision encoder.

        The top-level Omni wrapper does not own the vision tower itself;
        instead it delegates to the Thinker, which is the multimodal
        understanding component that fuses vision/audio/text. This
        property keeps the HF-style ``model.visual`` access pattern
        working at the outer layer.

        Returns:
            spx.Module: ``self.thinker.visual`` (a
            :class:`Qwen3OmniMoeVisionEncoder`).
        """
        return self.thinker.visual

    @property
    def audio(self):
        """Forward to the Thinker's audio encoder.

        Returns:
            spx.Module: ``self.thinker.audio`` (a
            :class:`Qwen3OmniMoeAudioEncoder`).
        """
        return self.thinker.audio

    def compute_embedding(self, input_ids, *args, **kwargs):
        """Delegate embedding computation to the Thinker component.

        Args:
            input_ids: Input token IDs.
            *args: Positional arguments forwarded to thinker.compute_embedding.
            **kwargs: Keyword arguments forwarded to thinker.compute_embedding.

        Returns:
            Array: Merged multimodal embeddings from the Thinker.
        """
        return self.thinker.compute_embedding(input_ids, *args, **kwargs)

    def compute_embedding_with_info(self, input_ids, *args, **kwargs):
        """Preserve Thinker position metadata and DeepStack residuals for prefill."""
        return self.thinker.compute_embedding_with_info(input_ids, *args, **kwargs)

    def get_rope_index(self, *args, **kwargs):
        """Delegate Omni prompt positional indexing to the Thinker."""
        return self.thinker.get_rope_index(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Prepare the Thinker's multimodal prompt and continuation cache."""
        return self.thinker.prepare_inputs_for_generation(*args, **kwargs)

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        """Delegate continuation updates and encoded-media cleanup to the Thinker."""
        return self.thinker.update_inputs_for_generation(model_outputs, model_kwargs)

    def forward(
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
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        feature_lens: Int[Array, "batch"] | None = None,
        feature_attention_mask: Bool[Array, "batch time"] | None = None,
        visual_pos_masks: Array | None = None,
        deepstack_visual_embeds: list[Array] | None = None,
        rope_deltas: Array | None = None,
        use_audio_in_video: bool = False,
        audio_seqlens: Array | None = None,
        second_per_grids: Array | None = None,
    ) -> MoeCausalLMOutput:
        """Forward pass through the Thinker component.

        For text generation use this method directly. For end-to-end
        audio output, pair the Thinker logits with the Talker and the
        Code2Wav (token-to-waveform) components, which the surrounding
        Qwen3-Omni wrapper exposes separately.

        Args:
            input_ids: Token ids ``(batch, seq_len)``; mutually exclusive
                with ``inputs_embeds``.
            inputs_embeds: Pre-computed embeddings
                ``(batch, seq_len, hidden_size)``.
            attention_mask: Boolean mask ``(batch, seq_len)``.
            mask_info: Optional pre-built ``MaskInfo`` carrying
                causal/sliding masks.
            position_ids: 1-D or 3-D position ids; the Thinker chooses
                the layout based on whether mRoPE is enabled.
            input_features: Mel-spectrogram features
                ``(batch, mel_bins, time)`` for the audio encoder.
            feature_lens: Valid mel-frame count for each audio sample.
            feature_attention_mask: Padding mask over mel frames.
            pixel_values: Packed image pixel values for the vision tower.
            image_grid_thw: Per-image ``(T, H, W)`` grids.
            pixel_values_videos: Packed video pixel values for the vision
                tower.
            video_grid_thw: Per-video ``(T, H, W)`` grids.
            output_attentions: Whether to return per-layer attention
                weights.
            output_hidden_states: Whether to return per-layer hidden
                states.
            output_router_logits: Whether to return MoE router logits
                (used for the auxiliary load-balancing loss).
            mode: Runtime mode (train/decode); auto-detected if ``None``.
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata accompanying ``past_key_values``.
            apply_lm_head: When ``True`` (default) project the final
                hidden states through the LM head; when ``False`` return
                the hidden states only.

        Returns:
            MoeCausalLMOutput: ``logits`` (or hidden states when
            ``apply_lm_head=False``), updated KV cache, optional hidden
            states / attentions, and router logits / aux loss when MoE
            tracing is enabled.
        """
        return self.thinker(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            input_features=input_features,
            feature_lens=feature_lens,
            feature_attention_mask=feature_attention_mask,
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
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            rope_deltas=rope_deltas,
            use_audio_in_video=use_audio_in_video,
            audio_seqlens=audio_seqlens,
            second_per_grids=second_per_grids,
        )

    def get_encoder(self):
        """Top-level Omni model has no monolithic encoder.

        Audio, vision, and text encoders live inside the Thinker
        component and are reachable via :attr:`audio` / :attr:`visual`
        / :meth:`get_decoder`. Surfacing a single encoder here would
        misrepresent the architecture, so this method intentionally
        raises.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("This is a decoder-only model.")

    def get_decoder(self):
        """Return the Thinker's text decoder.

        Returns:
            spx.Module: ``self.thinker.get_decoder()`` — the underlying
            :class:`Qwen3OmniMoeThinkerTextModel` that processes the
            merged multimodal sequence.
        """
        return self.thinker.get_decoder()

    def get_embedding(self):
        """Return the Thinker's text token embedding.

        Returns:
            spx.Module: ``self.thinker.get_embedding()``.
        """
        return self.thinker.get_embedding()

    def get_image_features(
        self,
        pixel_values: Float[Array, "num_patches channels height width"],
        image_grid_thw: Int[Array, "num_images 3"] | None = None,
        return_deepstack: bool = False,
        **kwargs,
    ) -> Float[Array, "num_patches hidden"] | tuple[tuple[Array, ...], list[Array]]:
        """Extract image features via the Thinker's vision encoder.

        Convenience wrapper that proxies to
        :meth:`Qwen3OmniMoeThinkerForConditionalGeneration.get_image_features`
        so callers operating on the top-level model do not need to
        reach into the Thinker explicitly.

        Args:
            pixel_values: Packed image patches of shape
                ``(num_patches, channels, height, width)``.
            image_grid_thw: Per-image grid sizes ``(T, H, W)`` (``T == 1``
                for still images), needed to delimit per-image token
                spans in the vision tower output.
            **kwargs: Forwarded keyword arguments.

        Returns:
            Visual features of shape ``(num_patches, hidden_size)`` ready
            for placeholder substitution into the text sequence.
        """
        return self.thinker.get_image_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_deepstack=return_deepstack,
            **kwargs,
        )

    def get_video_features(self, pixel_values_videos, video_grid_thw=None, return_deepstack=False, **kwargs):
        """Extract video features, optionally retaining intermediate residuals."""
        return self.thinker.get_video_features(
            pixel_values_videos,
            video_grid_thw,
            return_deepstack=return_deepstack,
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
