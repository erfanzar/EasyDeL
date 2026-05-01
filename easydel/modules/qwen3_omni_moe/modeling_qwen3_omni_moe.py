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
from functools import partial

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
    ModelOutput,
    MoeCausalLMOutput,
    VLMCausalLMOutput,
)
from easydel.infra.utils import ACT2FN, auto_remat
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
)
from easydel.layers.attention import FlexibleAttentionModule, UnifiedAttention
from easydel.layers.norms import LayerNorm
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


@auto_pytree
class Qwen3OmniMoeCausalLMOutputWithPast(ModelOutput):
    """Output class for Qwen3OmniMoe causal language model.

    Contains model outputs including logits, hidden states, attention weights,
    and multimodal embeddings from audio and image encoders.

    Attributes:
        loss (Array | None): Language modeling loss if labels provided.
        logits (Array): Prediction scores from language model head.
        past_key_values (list[Array] | None): Cached key-value states for generation.
        hidden_states (tuple[Array] | None): Hidden states from all layers if requested.
        attentions (tuple[Array] | None): Attention weights from all layers if requested.
        rope_deltas (Array | None): RoPE position deltas for multimodal inputs.
        image_hidden_states (Array | None): Encoded image features [batch, seq_len, hidden_dim].
        audio_hidden_states (Array | None): Encoded audio features [batch, seq_len, hidden_dim].
        router_logits (tuple[Array] | None): Router logits from MoE layers if requested.
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
    """Feed-forward network for audio encoder.

    Implements a two-layer feedforward network with configurable activation
    function for processing audio features in the audio encoder.
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
        """Initialize audio encoder MLP.

        Args:
            config (Qwen3OmniMoeAudioConfig): Audio encoder configuration with
                d_model and encoder_ffn_dim parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
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
        """Apply feedforward transformation to audio features.

        Args:
            hidden_states: Input tensor [batch, seq_len, d_model].

        Returns:
            Transformed hidden states [batch, seq_len, d_model].
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
            attention_mask=attention_mask,
            causal=False,
        ).attention_outputs

        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        attn_output = apply_logical_sharding(
            attn_output,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        return self.out_proj(attn_output)


class Qwen3OmniMoeAudioEncoderLayer(spx.Module):
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
        )
        self.conv2d2 = nn.Conv2d(
            in_channels=config.downsample_hidden_size,
            out_channels=config.downsample_hidden_size,
            kernel_size=3,
            stride=2,
            padding=((1, 1), (1, 1)),
            dtype=dtype,
            rngs=rngs,
        )
        self.conv2d3 = nn.Conv2d(
            in_channels=config.downsample_hidden_size,
            out_channels=config.downsample_hidden_size,
            kernel_size=3,
            stride=2,
            padding=((1, 1), (1, 1)),
            dtype=dtype,
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

        self.layers = nn.ModuleList([])
        for i in range(config.encoder_layers):
            with spx.assign_stage(total=config.encoder_layers, current=i):
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

        self.ln_post = LayerNorm(
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

        hidden_states = input_features.transpose(0, 2, 1)[..., None]

        hidden_states = jax.nn.gelu(self.conv2d1(hidden_states))
        hidden_states = jax.nn.gelu(self.conv2d2(hidden_states))
        hidden_states = jax.nn.gelu(self.conv2d3(hidden_states))

        b, t, f, c = hidden_states.shape
        hidden_states = hidden_states.reshape(b, t, f * c)

        hidden_states = self.conv_out(hidden_states)

        seq_len = hidden_states.shape[1]
        pos_emb = create_sinusoidal_positions(seq_len, self.d_model).astype(hidden_states.dtype)
        hidden_states = hidden_states + pos_emb[None, :, :]

        # Note: For simplicity, we process without cu_seqlens chunking
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
        )
        hidden_states = self.ln_post(hidden_states)

        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)

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


def rotate_half(x: Array) -> Array:
    """Rotate the second half of the trailing dim by negation.

    Helper used by the vision RoPE implementation: given a tensor whose
    last dimension is split into two halves ``[a, b]``, returns
    ``[-b, a]``. Combined with element-wise ``cos``/``sin``
    multiplications this realises a 2-D rotation per pair of channels.

    Args:
        x: Input tensor with an even-sized trailing dimension.

    Returns:
        Tensor of identical shape with the lower / upper halves swapped
        and the new lower half negated.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision(
    q: Array,
    k: Array,
    cos: Array,
    sin: Array,
) -> tuple[Array, Array]:
    """Apply rotary position embeddings to vision query/key tensors.

    Promotes ``q``/``k`` and the precomputed ``cos``/``sin`` tables to
    float32 to avoid bf16 round-off in the trigonometric mixing, applies
    the standard RoPE rotation ``x * cos + rotate_half(x) * sin``, and
    casts the results back to the original dtypes of the inputs.

    Args:
        q: Query tensor with trailing rotated dim.
        k: Key tensor with trailing rotated dim.
        cos: Cosine table broadcastable to ``q``/``k`` after a heads-axis
            expand (``..., heads_axis, dim``).
        sin: Sine table with the same shape as ``cos``.

    Returns:
        Tuple of ``(q_rot, k_rot)`` with the same shapes/dtypes as the
        inputs, rotated according to the supplied frequency table.
    """
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.astype("f4"), k.astype("f4")
    cos, sin = jnp.expand_dims(cos, -2).astype("f4"), jnp.expand_dims(sin, -2).astype("f4")
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.astype(orig_q_dtype), k_embed.astype(orig_k_dtype)


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
    attention_mask = jnp.full((1, seq_length, seq_length), jnp.finfo(dtype).min, dtype=dtype)
    mask_updates = jnp.zeros((1, seq_length, seq_length), dtype=dtype)

    for i in range(1, len(cu_seqlens)):
        start_idx = cu_seqlens[i - 1]
        end_idx = cu_seqlens[i]
        mask_updates = mask_updates.at[..., start_idx:end_idx, start_idx:end_idx].set(0)

    attention_mask = jax.lax.dynamic_update_slice(attention_mask, mask_updates, (0, 0, 0))
    return attention_mask


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
        return hidden_states.reshape(-1, self.hidden_size)


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

        self._rotary_dim = head_dim // 2

        self.blocks = nn.ModuleList([])
        for idx in range(config.depth):
            with spx.assign_stage(total=config.depth, current=idx):
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

    def forward(
        self,
        hidden_states: Array,
        grid_thw: Array,
        max_grid_size: int,
    ) -> Array:
        """Forward pass through vision encoder.

        Args:
            hidden_states: Input pixel values [num_patches, channels * t * h * w].
            grid_thw: Grid dimensions (temporal, height, width) for each image/video.
            max_grid_size: Maximum grid size for rotary embedding computation.

        Returns:
            Encoded vision features [num_merged_patches, out_hidden_size].
        """
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw, max_grid_size)

        grid_lens = grid_thw[:, 1] * grid_thw[:, 2]
        repeated = jnp.repeat(grid_lens, grid_thw[:, 0])
        cu_seqlens = jnp.cumsum(repeated, dtype="i4")
        cu_seqlens = jnp.pad(cu_seqlens, (1, 0), constant_values=0)

        def _layer_loop(block, carry):
            """Apply a single vision-encoder block inside the layer-stack scan.

            Body of ``self.blocks.scan``; runs ``block`` on the current
            vision hidden states with the precomputed ``cu_seqlens`` and
            ``rotary_pos_emb``, and returns the updated carry tuple.
            """
            hidden_states, idx = carry
            with self._layer_stage_context(idx, layers=self.blocks):
                hidden_states = block(hidden_states, cu_seqlens, rotary_pos_emb)
            hidden_states = self._mark_layer_stage_boundary(hidden_states, idx, layers=self.blocks)

            return hidden_states, idx + 1

        hidden_states, _ = self.blocks.scan(
            _layer_loop,
            (hidden_states, 0),
            trace=not self.config.scan_layers or self._pipeline_stage_count() > 1,
        )
        return self.merger(hidden_states)

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


class Qwen3OmniMoeTextMLP(spx.Module):
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
    ):
        """Initialize text decoder MLP.

        Args:
            config (Qwen3OmniMoeTextConfig): Text decoder configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
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

    def forward(self, hidden_states: Array) -> Array:
        """Apply SwiGLU feedforward transformation.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim].

        Returns:
            Transformed hidden states [batch, seq_len, hidden_dim].
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        gate = checkpoint_name(self.act_fn(self.gate_proj(hidden_states)), "mlp_gate")
        up = checkpoint_name(self.up_proj(hidden_states), "mlp_up")
        hidden_states = checkpoint_name(self.down_proj(gate * up), "mlp_down")
        return checkpoint_name(hidden_states, "mlp_output")


class Qwen3OmniMoeMLPStack(spx.Module):
    """Stacked MoE MLP module using ParallelMoELinear layers.

    Implements the expert MLP stack with SwiGLU activation function using
    column and row parallel MoE linear layers for efficient expert computation.
    """

    reform_param: typing.ClassVar = {
        "gate_up_proj$": {
            "splits": [
                {
                    "name": "gate_proj.weight",
                    "spliter": lambda x: x[:, : x.shape[1] // 2, :].swapaxes(-1, -2),
                },
                {
                    "name": "up_proj.weight",
                    "spliter": lambda x: x[:, x.shape[1] // 2 :, :].swapaxes(-1, -2),
                },
            ],
            "inverse_spliter": lambda torch, gate, up: torch.cat(
                (gate.transpose(-1, -2), up.transpose(-1, -2)),
                dim=1,
            ),
        },
        "down_proj$": {
            "splits": [
                {"name": "down_proj.weight", "spliter": lambda x: x.swapaxes(-1, -2)},
            ],
            "inverse_spliter": lambda x: x.swapaxes(-1, -2),
        },
    }

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
        self.gate_proj = ColumnParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.hidden_size,
            out_features=config.moe_intermediate_size,
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
        self.up_proj = ColumnParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.hidden_size,
            out_features=config.moe_intermediate_size,
            rngs=rngs,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(),
            partition_manager=config.runtime_sharding_resolver,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.act_fn = ACT2FN[config.hidden_act]

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
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states, group_sizes, sorted_experts))
            * self.up_proj(hidden_states, group_sizes, sorted_experts),
            group_sizes,
            sorted_experts,
        )


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
            wi_kernel=self.experts.gate_proj.weight.value,
            wu_kernel=self.experts.up_proj.weight.value,
            wd_kernel=self.experts.down_proj.weight.value,
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
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
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


class Qwen3OmniMoeTalkerTextMLP(spx.Module):
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

    def forward(self, hidden_states: Array) -> Array:
        """Apply SwiGLU feedforward transformation.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim].

        Returns:
            Transformed hidden states [batch, seq_len, hidden_dim].
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        gate = checkpoint_name(self.act_fn(self.gate_proj(hidden_states)), "mlp_gate")
        up = checkpoint_name(self.up_proj(hidden_states), "mlp_up")
        hidden_states = checkpoint_name(self.down_proj(gate * up), "mlp_down")
        return checkpoint_name(hidden_states, "mlp_output")


class Qwen3OmniMoeTalkerMLPStack(spx.Module):
    """Stacked MoE MLP module for Talker using ParallelMoELinear layers.

    Implements the expert MLP stack with SwiGLU activation function for
    the Talker's MoE layers.
    """

    reform_param: typing.ClassVar = {
        "gate_up_proj$": {
            "splits": [
                {
                    "name": "gate_proj.weight",
                    "spliter": lambda x: x[:, : x.shape[1] // 2, :].swapaxes(-1, -2),
                },
                {
                    "name": "up_proj.weight",
                    "spliter": lambda x: x[:, x.shape[1] // 2 :, :].swapaxes(-1, -2),
                },
            ],
            "inverse_spliter": lambda torch, gate, up: torch.cat(
                (gate.transpose(-1, -2), up.transpose(-1, -2)),
                dim=1,
            ),
        },
        "down_proj$": {
            "splits": [
                {"name": "down_proj.weight", "spliter": lambda x: x.swapaxes(-1, -2)},
            ],
            "inverse_spliter": lambda x: x.swapaxes(-1, -2),
        },
    }

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Talker MoE MLP stack.

        Args:
            config (Qwen3OmniMoeTalkerTextConfig): Talker text configuration with MoE parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        self.gate_proj = ColumnParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.hidden_size,
            out_features=config.moe_intermediate_size,
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
        self.up_proj = ColumnParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.hidden_size,
            out_features=config.moe_intermediate_size,
            rngs=rngs,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(),
            partition_manager=config.runtime_sharding_resolver,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.act_fn = ACT2FN[config.hidden_act]

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
            wi_kernel=self.experts.gate_proj.weight.value,
            wu_kernel=self.experts.up_proj.weight.value,
            wd_kernel=self.experts.down_proj.weight.value,
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
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
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


class Qwen3OmniMoeTalkerCodePredictorMLP(spx.Module):
    """Dense MLP for Talker code predictor.

    Implements the feedforward network with SwiGLU activation function
    for the code predictor (non-MoE) layers.
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
        """Initialize code predictor MLP.

        Args:
            config (Qwen3OmniMoeTalkerCodePredictorConfig): Code predictor configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
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

    def forward(self, hidden_states: Array) -> Array:
        """Apply SwiGLU feedforward transformation.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim].

        Returns:
            Transformed hidden states [batch, seq_len, hidden_dim].
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        gate = checkpoint_name(self.act_fn(self.gate_proj(hidden_states)), "mlp_gate")
        up = checkpoint_name(self.up_proj(hidden_states), "mlp_up")
        hidden_states = checkpoint_name(self.down_proj(gate * up), "mlp_down")
        return checkpoint_name(hidden_states, "mlp_output")


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
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
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
            with spx.assign_stage(total=config.num_hidden_layers, current=layer_idx):
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
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
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

        def _layer_loop(block, carry):
            """Apply a single decoder layer inside the layer-stack scan.

            Body of ``self.layers.scan`` for a non-MoE decoder stack;
            runs ``block`` on the current hidden states, optionally
            accumulates per-layer hidden states / attention weights, and
            returns the updated carry tuple.
            """
            hidden_states, all_hidden_states, all_attentions, idx = carry
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            with self._layer_stage_context(idx, layers=self.layers):
                layer_outputs = block(
                    hidden_states=hidden_states,
                    mask_info=mask_info,
                    position_ids=position_ids,
                    mode=mode,
                    cache_view=self._layer_cache_view_at(None, idx, enabled=True, cache=past_key_values),
                    cache_metadata=cache_metadata,
                    output_attentions=output_attentions,
                    frequencies=self.frequencies,
                )
            hidden_states = self._mark_layer_stage_boundary(layer_outputs.hidden_states, idx, layers=self.layers)

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            self._layer_cache_view_update(None, idx, layer_outputs.cache_view, enabled=True, cache=past_key_values)

            return hidden_states, all_hidden_states, all_attentions, idx + 1

        hidden_states, all_hidden_states, all_attentions, _ = self.layers.scan(
            _layer_loop,
            (hidden_states, all_hidden_states, all_attentions, 0),
            trace=True,
        )
        hidden_states = self.norm(hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
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
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
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
            with spx.assign_stage(total=config.num_hidden_layers, current=layer_idx):
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
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
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

        def _layer_loop(block, carry):
            """Apply a single Thinker MoE decoder layer inside the layer-stack scan.

            Body of ``self.layers.scan`` for the Thinker text decoder;
            runs ``block`` on the current hidden states, optionally
            accumulates per-layer hidden states, attention weights, and
            MoE router logits, and returns the updated carry tuple.
            """
            hidden_states, all_hidden_states, all_attentions, all_router_logits, idx = carry
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            with self._layer_stage_context(idx, layers=self.layers):
                layer_outputs = block(
                    hidden_states=hidden_states,
                    mask_info=mask_info,
                    position_ids=position_ids,
                    mode=mode,
                    cache_view=self._layer_cache_view_at(None, idx, enabled=True, cache=past_key_values),
                    cache_metadata=cache_metadata,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    frequencies=self.frequencies,
                )
            hidden_states = self._mark_layer_stage_boundary(layer_outputs.hidden_states, idx, layers=self.layers)

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            self._layer_cache_view_update(None, idx, layer_outputs.cache_view, enabled=True, cache=past_key_values)
            if output_router_logits and layer_outputs.router_logits is not None:
                all_router_logits += (layer_outputs.router_logits,)

            return hidden_states, all_hidden_states, all_attentions, all_router_logits, idx + 1

        hidden_states, all_hidden_states, all_attentions, all_router_logits, _ = self.layers.scan(
            _layer_loop,
            (hidden_states, all_hidden_states, all_attentions, all_router_logits, 0),
            trace=True,
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
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_router_logits: bool = False,
        apply_codec_head: bool = True,
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

        Returns:
            MoeCausalLMOutput with codec logits and optional intermediate outputs.
        """
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


class Qwen3OmniMoeCode2WavMLP(spx.Module):
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

    def forward(self, hidden_states: Array) -> Array:
        """Apply gated MLP: down_proj(act(gate_proj(x)) * up_proj(x)).

        Args:
            hidden_states: Input hidden states.

        Returns:
            Transformed hidden states with same hidden dimension.
        """
        gate = self.act_fn(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


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

    def forward(
        self,
        hidden_states: Array,
        attention_mask: Array | None = None,
        position_ids: Array | None = None,
    ) -> Array:
        """Apply sliding window causal attention.

        Args:
            hidden_states: Input hidden states [batch, seq_len, hidden_dim].
            attention_mask: Optional attention mask.
            position_ids: Optional position IDs (unused, included for interface compatibility).

        Returns:
            Attention output with same shape as input hidden states.
        """
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

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
        attn_output = apply_logical_sharding(
            attn_output,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        return self.o_proj(attn_output)


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
        attn_output = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = residual + self.self_attn_layer_scale(attn_output)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
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
            with spx.assign_stage(total=config.num_hidden_layers, current=layer_idx):
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

        def _layer_loop(layer, carry):
            """Apply a single code-predictor decoder layer inside the layer-stack scan.

            Body of ``self.layers.scan`` for the Talker code-predictor
            decoder; runs ``layer`` on the current hidden states with
            ``attention_mask`` and ``position_ids``, and returns the
            updated carry tuple.
            """
            hidden_states, idx = carry
            with self._layer_stage_context(idx, layers=self.layers):
                hidden_states = layer(hidden_states, attention_mask, position_ids)
            hidden_states = self._mark_layer_stage_boundary(hidden_states, idx, layers=self.layers)

            return hidden_states, idx + 1

        hidden_states, _ = self.layers.scan(
            _layer_loop,
            (hidden_states, 0),
            trace=not self.config.scan_layers or self._pipeline_stage_count() > 1,
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

        # Pass through transformer
        outputs = self.pre_transformer(hidden_states, attention_mask)

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


def merge_multimodal_embeddings(
    input_ids: jax.Array,
    inputs_embeds: jax.Array,
    multimodal_embeddings: jax.Array,
    placeholder_token_id: int | list[int],
) -> jax.Array:
    """Splice multimodal embeddings into a text embedding sequence.

    Replaces every position in ``inputs_embeds`` whose corresponding
    ``input_ids`` value is a registered placeholder token (e.g.
    ``<image>``, ``<video>``, ``<audio>``) with the next vector from
    ``multimodal_embeddings``. The fill order follows the natural
    left-to-right scan of ``input_ids``.

    The implementation uses the cumsum-gather pattern shared by
    :class:`MultiModalMergeFeature`: we prepend a zero row to the
    multimodal table so that non-placeholder positions index the dummy
    row, and then ``jnp.where`` picks the original or replaced
    embedding per-position. This formulation is JIT-safe and produces
    O(seq_len * hidden) memory traffic.

    Args:
        input_ids: Token ids of shape ``(batch, seq_len)``.
        inputs_embeds: Text embeddings of shape
            ``(batch, seq_len, hidden_size)``.
        multimodal_embeddings: Concatenated visual / audio embeddings of
            shape ``(num_multimodal_tokens, hidden_size)``.
        placeholder_token_id: Single token id or list of ids that mark
            multimodal slots in ``input_ids``.

    Returns:
        Merged embedding tensor with the same shape as ``inputs_embeds``.
    """
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
            with spx.assign_stage(total=config.num_hidden_layers, current=layer_idx):
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
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
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

        def _layer_loop(layer, carry):
            """Apply a single Talker code-predictor MoE layer inside the layer-stack scan.

            Body of ``self.layers.scan`` for the Talker code-predictor
            MoE decoder; runs ``layer`` on the current hidden states,
            optionally accumulates per-layer hidden states, self-attention
            weights, and MoE router logits, and returns the updated
            carry tuple.
            """
            hidden_states, all_hidden_states, all_self_attentions, all_router_logits, layer_idx = carry
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            with self._layer_stage_context(idx, layers=self.layers):
                layer_outputs = layer(
                    hidden_states=hidden_states,
                    mask_info=mask_info,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                    mode=mode,
                    cache_view=past_key_values.views[layer_idx] if past_key_values is not None else None,
                    cache_metadata=cache_metadata,
                )
            hidden_states = self._mark_layer_stage_boundary(layer_outputs.hidden_states, idx, layers=self.layers)

            if output_attentions:
                all_self_attentions += (layer_outputs.attentions,)
            if output_router_logits:
                all_router_logits += (layer_outputs.router_logits,)

            return hidden_states, all_hidden_states, all_self_attentions, all_router_logits, layer_idx + 1

        hidden_states, all_hidden_states, all_self_attentions, all_router_logits, _ = self.layers.scan(
            _layer_loop,
            (hidden_states, all_hidden_states, all_self_attentions, all_router_logits, 0),
            trace=True,
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
            with spx.assign_stage(total=text_config.num_hidden_layers, current=layer_idx):
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
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
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
            partition_manager=text_config.runtime_sharding_resolver,
        )

        def _layer_loop(block, carry):
            """Apply a single Talker MoE decoder layer inside the layer-stack scan.

            Body of ``self.layers.scan`` for the Talker text decoder;
            runs ``block`` on the current hidden states, optionally
            accumulates per-layer hidden states, attention weights, and
            MoE router logits, and returns the updated carry tuple.
            """
            hidden_states, all_hidden_states, all_attentions, all_router_logits, idx = carry
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            with self._layer_stage_context(idx, layers=self.layers):
                layer_outputs = block(
                    hidden_states=hidden_states,
                    mask_info=mask_info,
                    position_ids=position_ids,
                    mode=mode,
                    cache_view=self._layer_cache_view_at(None, idx, enabled=True, cache=past_key_values),
                    cache_metadata=cache_metadata,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    frequencies=self.frequencies,
                )
            hidden_states = self._mark_layer_stage_boundary(layer_outputs.hidden_states, idx, layers=self.layers)

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            self._layer_cache_view_update(None, idx, layer_outputs.cache_view, enabled=True, cache=past_key_values)
            if output_router_logits and layer_outputs.router_logits is not None:
                all_router_logits += (layer_outputs.router_logits,)

            return hidden_states, all_hidden_states, all_attentions, all_router_logits, idx + 1

        hidden_states, all_hidden_states, all_attentions, all_router_logits, _ = self.layers.scan(
            _layer_loop,
            (hidden_states, all_hidden_states, all_attentions, all_router_logits, 0),
            trace=True,
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
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
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
        **kwargs,
    ) -> Float[Array, "num_patches hidden"]:
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
