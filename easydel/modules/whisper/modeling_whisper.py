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


import math
import random
import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
from eformer import common_types
from eformer.escale import apply_logical_sharding
from ejkernel.types import MaskInfo
from flax import nnx as nn
from jax import lax
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from easydel.inference.logits_process import (
    ForceTokensLogitsProcessor,
    LogitsProcessorList,
    WhisperTimeStampLogitsProcessor,
)
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.loss_utils import LossConfig, LossMetrics
from easydel.infra.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    SequenceClassifierOutput,
)
from easydel.infra.utils import ACT2FN, auto_remat
from easydel.layers.attention import AttentionModule, FlexibleAttentionModule
from easydel.layers.base_modules import BaseConditionalGenerationModule
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

from .whisper_configuration import WhisperConfig as WhisperConfig

remat = nn.remat


def shift_tokens_right(
    input_ids: Int[Array, "batch seq_len"],
    pad_token_id: int,
    decoder_start_token_id: int,
):
    """
    Shift input ids one token to the right using JAX.
    """
    batch_size, seq_length = input_ids.shape
    shifted_input_ids = jnp.full(
        (batch_size, seq_length),
        pad_token_id,
        dtype=input_ids.dtype,
    )
    shifted_input_ids = shifted_input_ids.at[:, 1:].set(input_ids[:, :-1])
    shifted_input_ids = shifted_input_ids.at[:, 0].set(decoder_start_token_id)
    shifted_input_ids = jnp.where(input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids


def sinusoidal_embedding_init(key, shape, dtype=jnp.float_) -> jax.Array:
    """Initializes sinusoidal positional embeddings.

    Args:
        key: JAX PRNG key (unused, but part of standard initializer signature).
        shape (tuple): Shape of the embedding matrix (length, channels).
        dtype: Data type of the embeddings (default: jnp.float_).

    Returns:
        jax.Array: Sinusoidal positional embedding matrix.

    Raises:
        ValueError: If the number of channels is not even.
    """
    length, channels = shape
    if channels % 2 != 0:
        raise ValueError(
            f"Number of channels has to be divisible by 2 for sinusoidal positional embeddings, got {channels} channels."
        )
    log_timescale_increment = math.log(10000) / (channels // 2 - 1)
    inv_timescales = jnp.exp(-log_timescale_increment * jnp.arange(channels // 2))
    scaled_time = jnp.arange(length).reshape(-1, 1) * inv_timescales.reshape(1, -1)
    return jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=1).astype(dtype)


class WhisperAttention(AttentionModule):
    """Whisper Attention mechanism.

    This module implements the standard multi-head attention mechanism used
    in both the encoder and decoder of the Whisper model.

    Attributes:
        config (WhisperConfig): Configuration object for the model.
        embed_dim (int): Dimensionality of the embedding layer.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        causal (bool): Whether this attention is causal (used in decoder self-attention).
        bias (bool): Whether to include bias in linear projections.
        head_dim (int): Dimensionality of each attention head.
        q_proj (ParallelLinear): Linear layer for query projection.
        k_proj (ParallelLinear): Linear layer for key projection.
        v_proj (ParallelLinear): Linear layer for value projection.
        out_proj (ParallelLinear): Linear layer for output projection.
        attention_performer (FlexibleAttentionModule): Module for performing attention computation.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for matrix multiplications.
        rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: WhisperConfig,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        causal: bool = False,
        bias: bool = True,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        """Initializes the WhisperAttention module.

        Args:
            config (WhisperConfig): The configuration object for the model.
            embed_dim (int): Dimensionality of the input and output features.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability for attention weights (default: 0.0).
            causal (bool): Whether to apply causal masking (default: False).
            bias (bool): Whether to include bias terms in the projection layers (default: True).
            dtype (jnp.dtype): Data type for computations (default: jnp.float32).
            param_dtype (jnp.dtype): Data type for parameters (default: jnp.float32).
            precision (tp.Optional[tp.Union[str, lax.Precision]]): Precision setting for JAX operations (default: None).
            rngs (nn.Rngs): Random number generators.

        Raises:
            ValueError: If `embed_dim` is not divisible by `num_heads`.
        """
        super().__init__(config=config)
        self.rngs = rngs
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.causal = causal
        self.bias = bias
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        linear = partial(
            ColumnParallelLinear,
            self.embed_dim,
            self.embed_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )

        self.q_proj = linear(use_bias=self.bias, rngs=rngs)
        self.k_proj = linear(use_bias=False, rngs=rngs)
        self.v_proj = linear(use_bias=self.bias, rngs=rngs)
        self.out_proj = linear(use_bias=self.bias, rngs=rngs)

        # Only causal (decoder) attention needs KV cache.
        # Encoder attention (causal=False) processes full sequences at once.
        self.attention_performer = FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=config.attention_dropout,
            requires_cache=causal,  # Only decoder self-attention needs cache
        )

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None = None,
        key_value_states: jnp.ndarray | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
    ) -> tuple[tp.Any, tp.Any, tp.Any]:
        """Forward pass of the attention module.

        Args:
            hidden_states (jnp.ndarray): Input hidden states (batch, seq_len, embed_dim).
            mask_info (MaskInfo | None): Mask information for attention.
            key_value_states (tp.Optional[jnp.ndarray]): Optional key/value states for cross-attention
                (batch, kv_seq_len, embed_dim). If None, self-attention is performed.
            mode: Runtime mode (train/decode/prefill).
            cache_view (tp.Optional[TransformerCacheView]): Cache view for key/value states, used in causal attention.
            cache_metadata (tp.Optional[TransformerMetadata]): Metadata for paged attention.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray, cache_view]: A tuple containing:
                - attn_output (jnp.ndarray): Attention output (batch, seq_len, embed_dim).
                - attn_weights (jnp.ndarray): Attention weights (batch, num_heads, seq_len, kv_seq_len).
                - cache_view: Updated cache view.
        """
        is_cross_attention = key_value_states is not None
        query_states = checkpoint_name(self.q_proj(hidden_states), "attn_query")

        if is_cross_attention:
            key_states = checkpoint_name(self.k_proj(key_value_states), "attn_key")
            value_states = checkpoint_name(self.v_proj(key_value_states), "attn_value")
        else:
            key_states = checkpoint_name(self.k_proj(hidden_states), "attn_key")
            value_states = checkpoint_name(self.v_proj(hidden_states), "attn_value")

        query_states = self._split_heads(query_states)
        key_states = self._split_heads(key_states)
        value_states = self._split_heads(value_states)

        init_attention_bias = lambda: None  # noqa

        if self.causal:
            # For causal (decoder) attention, use concatenate to handle KV cache
            (
                key_states,
                value_states,
                mask_info,
                init_attention_bias,
                cache_view,
                cache_metadata,
            ) = self.concatenate(
                query=query_states,
                key=key_states,
                cache_view=cache_view,
                value=value_states,
                mask_info=mask_info,
            )

        attentions = self.attention_performer.forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            mode=mode,
            bias=None,
            cache_metadata=cache_metadata,
            cache_view=cache_view,
            init_bias=init_attention_bias,
            mask_info=mask_info,
            causal=self.causal,
        )

        attn_output = checkpoint_name(
            self.out_proj(self.shard_attention_prod(self._merge_heads(attentions.attention_outputs))), "attn_output"
        )

        return attn_output, attentions.attention_weights, cache_view

    def _split_heads(self, hidden_state) -> jnp.ndarray:
        """Splits the last dimension of the hidden state into (num_heads, head_dim)."""
        return hidden_state.reshape((*hidden_state.shape[:2], self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_state) -> jnp.ndarray:
        """Merges the last two dimensions (num_heads, head_dim) into embed_dim."""
        return hidden_state.reshape((*hidden_state.shape[:2], self.embed_dim))


class WhisperEncoderLayer(nn.Module):
    """A single layer for the Whisper encoder.

    This layer consists of a self-attention mechanism followed by a feed-forward
    network (FFN), with residual connections and layer normalization.

    Attributes:
        config (WhisperConfig): Configuration object for the model.
        embed_dim (int): Dimensionality of the input and output features.
        self_attn (WhisperAttention): Self-attention module.
        self_attn_layer_norm (nn.LayerNorm): Layer normalization before self-attention.
        fc1 (ParallelLinear): First linear layer of the FFN.
        fc2 (ParallelLinear): Second linear layer of the FFN.
        final_layer_norm (nn.LayerNorm): Layer normalization after the FFN.
        activation_fn (callable): Activation function for the FFN.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for matrix multiplications.
        rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: WhisperConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        """Initializes the WhisperEncoderLayer module.

        Args:
            config (WhisperConfig): The configuration object for the model.
            dtype (jnp.dtype): Data type for computations (default: jnp.float32).
            param_dtype (jnp.dtype): Data type for parameters (default: jnp.float32).
            precision (tp.Optional[tp.Union[str, lax.Precision]]): Precision setting for JAX operations (default: None).
            rngs (nn.Rngs): Random number generators.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.embed_dim = self.config.d_model

        self.self_attn = WhisperAttention(
            config=config,
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        linear = partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            epsilon=1e-05,
            rngs=rngs,
        )
        self.dropout_layer = nn.Dropout(rate=self.config.dropout, rngs=rngs)
        self.activation_fn = ACT2FN[self.config.activation_function]
        self.activation_dropout_layer = nn.Dropout(
            rate=self.config.activation_dropout,
            rngs=rngs,
        )
        self.fc1 = linear(self.embed_dim, self.config.encoder_ffn_dim, rngs=rngs)
        self.fc2 = linear(self.config.encoder_ffn_dim, self.embed_dim, rngs=rngs)
        self.final_layer_norm = nn.LayerNorm(
            self.embed_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            epsilon=1e-05,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None = None,
        output_attentions: bool = True,
    ) -> tuple[jnp.ndarray, ...]:
        """Forward pass of the encoder layer.

        Args:
            hidden_states (jnp.ndarray): Input hidden states (batch, seq_len, embed_dim).
            mask_info (MaskInfo | None): Mask information for attention.
            output_attentions (bool): Whether to return attention weights (default: True).

        Returns:
            tp.Tuple[jnp.ndarray, ...]: A tuple containing:
                - hidden_states (jnp.ndarray): Output hidden states (batch, seq_len, embed_dim).
                - attn_weights (jnp.ndarray, optional): Attention weights if `output_attentions` is True.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            mask_info=mask_info,
            mode=common_types.MODE_TRAIN,
            cache_view=None,
            key_value_states=None,
        )
        hidden_states = self.dropout_layer(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.activation_dropout_layer(hidden_states)
        hidden_states = checkpoint_name(self.fc2(hidden_states), "mlp_down")
        hidden_states = self.dropout_layer(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class WhisperDecoderLayer(nn.Module):
    """A single layer for the Whisper decoder.

    This layer consists of self-attention, cross-attention (attending to encoder outputs),
    and a feed-forward network (FFN), each followed by residual connections and layer normalization.

    Attributes:
        config (WhisperConfig): Configuration object for the model.
        embed_dim (int): Dimensionality of the input and output features.
        self_attn (WhisperAttention): Self-attention module (causal).
        encoder_attn (WhisperAttention): Cross-attention module (attends to encoder outputs).
        self_attn_layer_norm (nn.LayerNorm): Layer normalization before self-attention.
        encoder_attn_layer_norm (nn.LayerNorm): Layer normalization before cross-attention.
        fc1 (ParallelLinear): First linear layer of the FFN.
        fc2 (ParallelLinear): Second linear layer of the FFN.
        final_layer_norm (nn.LayerNorm): Layer normalization after the FFN.
        activation_fn (callable): Activation function for the FFN.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for matrix multiplications.
        rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: WhisperConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        """Initializes the WhisperDecoderLayer module.

        Args:
            config (WhisperConfig): The configuration object for the model.
            dtype (jnp.dtype): Data type for computations (default: jnp.float32).
            param_dtype (jnp.dtype): Data type for parameters (default: jnp.float32).
            precision (tp.Optional[tp.Union[str, lax.Precision]]): Precision setting for JAX operations (default: None).
            rngs (nn.Rngs): Random number generators.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.embed_dim = self.config.d_model
        self.self_attn = WhisperAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            causal=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.dropout_layer = nn.Dropout(
            rate=self.config.dropout,
            rngs=rngs,
        )
        self.activation_fn = ACT2FN[self.config.activation_function]
        self.activation_dropout_layer = nn.Dropout(
            rate=self.config.activation_dropout,
            rngs=rngs,
        )

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            epsilon=1e-05,
            rngs=rngs,
        )
        self.encoder_attn = WhisperAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(
            self.embed_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            epsilon=1e-05,
            rngs=rngs,
        )
        linear = partial(
            ColumnParallelLinear,
            param_dtype=self.param_dtype,
            precision=self.precision,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        self.fc1 = linear(
            self.embed_dim,
            self.config.decoder_ffn_dim,
            rngs=rngs,
        )
        self.fc2 = linear(
            self.config.decoder_ffn_dim,
            self.embed_dim,
            rngs=rngs,
        )
        self.final_layer_norm = nn.LayerNorm(
            self.embed_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            epsilon=1e-05,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None = None,
        encoder_hidden_states: Float[Array, "batch seq_len hidden_dim"] | None = None,
        encoder_mask_info: MaskInfo | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = True,
    ) -> tuple[jnp.ndarray, ...]:
        """Forward pass of the decoder layer.

        Args:
            hidden_states (jnp.ndarray): Input hidden states (batch, seq_len, embed_dim).
            mask_info (MaskInfo | None): Mask information for self-attention.
            encoder_hidden_states (tp.Optional[jnp.ndarray]): Hidden states from the encoder
                (batch, encoder_seq_len, embed_dim).
            encoder_mask_info (MaskInfo | None): Mask information for cross-attention.
            mode: Runtime mode (train/decode/prefill).
            cache_view (tp.Optional[TransformerCacheView]): Cache view for key/value states.
            cache_metadata (tp.Optional[TransformerMetadata]): Metadata for paged attention.
            output_attentions (bool): Whether to return attention weights (default: True).

        Returns:
            tp.Tuple[jnp.ndarray, ...]: A tuple containing:
                - hidden_states (jnp.ndarray): Output hidden states (batch, seq_len, embed_dim).
                - self_attn_weights (jnp.ndarray, optional): Self-attention weights if `output_attentions` is True.
                - cross_attn_weights (jnp.ndarray, optional): Cross-attention weights if `output_attentions` is True.
                - cache_view: Updated cache view.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, cache_view = self.self_attn(
            hidden_states=hidden_states,
            mask_info=mask_info,
            mode=mode,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )
        hidden_states = self.dropout_layer(hidden_states)
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            hidden_states, cross_attn_weights, _ = self.encoder_attn(
                hidden_states=hidden_states,
                mask_info=encoder_mask_info,
                key_value_states=encoder_hidden_states,
            )
            hidden_states = self.dropout_layer(hidden_states)
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.activation_dropout_layer(hidden_states)
        hidden_states = checkpoint_name(self.fc2(hidden_states), "mlp_down")
        hidden_states = self.dropout_layer(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        outputs += (cache_view,)
        return outputs


class WhisperEncoder(EasyDeLBaseModule):
    """The Whisper Encoder transformer stack.

    This module processes the input audio features (log-Mel spectrogram) through
    convolutional layers followed by a stack of `WhisperEncoderLayer` modules.

    Attributes:
        config (WhisperConfig): Configuration object for the model.
        conv1 (nn.Conv): First convolutional layer.
        conv2 (nn.Conv): Second convolutional layer.
        embed_positions (nn.Embed): Positional embedding layer.
        layers (nn.List[WhisperEncoderLayer]): List of encoder layers.
        layer_norm (nn.LayerNorm): Final layer normalization.
        embed_dim (int): Dimensionality of the model.
        num_mel_bins (int): Number of Mel frequency bins in the input features.
        padding_idx (int): Index of the padding token.
        max_source_positions (int): Maximum sequence length for the encoder.
        scale_embedding (float | None): Scaling factor for embeddings.
        embed_scale (float | None): Alias for scale_embedding.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: WhisperConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the WhisperEncoder module.

        Args:
            config (WhisperConfig): The configuration object for the model.
            dtype (jnp.dtype): Data type for computations (default: jnp.float32).
            param_dtype (jnp.dtype): Data type for parameters (default: jnp.float32).
            precision (tp.Optional[tp.Union[str, lax.Precision]]): Precision setting for JAX operations (default: None).
            rngs (nn.Rngs): Random number generators.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.conv1 = nn.Conv(
            self.config.d_model,
            self.config.d_model,
            kernel_size=(3,),
            padding=1,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.conv2 = nn.Conv(
            self.config.d_model,
            self.config.d_model,
            kernel_size=(3,),
            strides=2,
            padding=1,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.dropout_layer = nn.Dropout(
            rate=self.config.dropout,
            rngs=rngs,
        )

        block = WhisperEncoderLayer
        self.layers = [
            block(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for i in range(self.config.encoder_layers)
        ]

        self.embed_positions = nn.Embed(
            self.config.max_source_positions,
            self.config.d_model,
            dtype=self.dtype,
            embedding_init=sinusoidal_embedding_init,
            param_dtype=self.param_dtype,
            rngs=rngs,
        )

        self.layer_norm = nn.LayerNorm(
            self.config.d_model,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            epsilon=1e-05,
            rngs=rngs,
        )
        self.layerdrop = self.config.decoder_layerdrop

    def __call__(
        self,
        input_features: jnp.ndarray,
        mask_info: MaskInfo | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> tuple[tp.Any | None, ...] | BaseModelOutput:
        """Forward pass of the Whisper encoder.

        Args:
            input_features (jnp.ndarray): Input audio features (log-Mel spectrogram)
                of shape (batch_size, num_mel_bins, sequence_length).
            mask_info (MaskInfo | None): Mask information for attention.
            output_attentions (bool): Whether to return attention weights (default: False).
            output_hidden_states (bool): Whether to return hidden states for all layers (default: False).

        Returns:
            BaseModelOutput | tuple: The encoder output. returns a `BaseModelOutput`
                containing `last_hidden_state`, `hidden_states` (optional), and `attentions` (optional).
        """
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        if input_features.shape[1:] != (
            self.config.num_mel_bins,
            self.config.max_source_positions * 2,
        ):
            raise ValueError(
                "input_features.shape[1:], must be equal to (self.config.num_mel_bins,"
                f" self.config.max_source_positions * 2) (got {input_features.shape[1:]}, but should be"
                f" ({self.config.num_mel_bins}, {self.config.max_source_positions * 2}))"
            )

        input_features = input_features.transpose(0, 2, 1)
        hidden_states = jax.nn.gelu(self.conv1(input_features), approximate=False)
        hidden_states = jax.nn.gelu(self.conv2(hidden_states), approximate=False)

        embed_positions = self.embed_positions(jnp.arange(self.config.max_source_positions))
        embed_positions = jax.lax.stop_gradient(embed_positions)
        hidden_states = hidden_states + embed_positions

        hidden_states = self.dropout_layer(hidden_states)

        for encoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = (*all_hidden_states, hidden_states)
            dropout_probability = random.uniform(0, 1)
            if not self.dropout_layer.deterministic and (dropout_probability < self.layerdrop):
                # skip the layer
                layer_outputs = (None, None)
            else:
                layer_outputs = encoder_layer(
                    hidden_states=hidden_states,
                    mask_info=mask_info,
                    output_attentions=output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = (*all_attentions, layer_outputs[1])

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class WhisperDecoder(EasyDeLBaseModule):
    """The Whisper Decoder transformer stack.

    This module processes the target token IDs, incorporates positional embeddings,
    and attends to both the input sequence (self-attention) and the encoder outputs
    (cross-attention) through a stack of `WhisperDecoderLayer` modules.

    Attributes:
        config (WhisperConfig): Configuration object for the model.
        embed_tokens (nn.Embed): Embedding layer for target tokens.
        embed_positions (nn.Embed): Positional embedding layer.
        layers (nn.List[WhisperDecoderLayer]): List of decoder layers.
        layer_norm (nn.LayerNorm): Final layer normalization (applied to pre-final outputs).
        dropout (nn.Dropout): Dropout layer.
        padding_idx (int): Index of the padding token.
        max_target_positions (int): Maximum sequence length for the decoder.
        embed_scale (float | None): Scaling factor for embeddings.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: WhisperConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the WhisperDecoder module.

        Args:
            config (WhisperConfig): The configuration object for the model.
            dtype (jnp.dtype): Data type for computations (default: jnp.float32).
            param_dtype (jnp.dtype): Data type for parameters (default: jnp.float32).
            precision (tp.Optional[tp.Union[str, lax.Precision]]): Precision setting for JAX operations (default: None).
            rngs (nn.Rngs): Random number generators.
        """
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
            self.config.vocab_size,
            self.config.d_model,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.embed_positions = nn.Embed(
            self.config.max_target_positions,
            self.config.d_model,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.layers = [
            WhisperDecoderLayer(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for i in range(self.config.decoder_layers)
        ]

        self.layerdrop = self.config.decoder_layerdrop
        self.dropout_layer = nn.Dropout(
            rate=self.config.dropout,
            rngs=rngs,
        )

        self.layer_norm = nn.LayerNorm(
            self.config.d_model,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            epsilon=1e-05,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"],
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        encoder_hidden_states: Float[Array, "batch seq_len hidden_dim"] | None = None,
        encoder_mask_info: MaskInfo | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> tuple[tp.Any, ...] | BaseModelOutputWithPastAndCrossAttentions:
        """Forward pass of the Whisper decoder.

        Args:
            input_ids (jnp.ndarray): Input token IDs (batch, target_sequence_length).
            mask_info (MaskInfo | None): Mask information for self-attention.
            position_ids (jnp.ndarray): Position IDs (batch, target_sequence_length).
            encoder_hidden_states (tp.Optional[jnp.ndarray]): Hidden states from the encoder
                (batch, encoder_sequence_length, embed_dim).
            encoder_mask_info (MaskInfo | None): Mask information for cross-attention.
            mode: Runtime mode (train/decode/prefill).
            past_key_values (tp.Optional[TransformerCache]): Cached key/value states for fast decoding.
            cache_metadata (tp.Optional[TransformerMetadata]): Metadata for paged attention.
            output_attentions (bool): Whether to return attention weights (default: False).
            output_hidden_states (bool): Whether to return hidden states for all layers (default: False).

        Returns:
            BaseModelOutputWithPastAndCrossAttentions: The decoder output.
                returns a `BaseModelOutputWithPastAndCrossAttentions` containing `last_hidden_state`,
                `past_key_values` (if `use_cache` is True), `hidden_states` (optional), `attentions` (optional),
                and `cross_attentions` (optional).
        """
        if mask_info is None:
            attention_mask = jnp.ones((input_ids.shape[0], input_ids.shape[1]), dtype=jnp.bool_)
            mask_info = MaskInfo.from_attention_mask(attention_mask)

        if encoder_hidden_states is not None and encoder_mask_info is None:
            cross_attention_mask = jnp.ones(
                (input_ids.shape[0], input_ids.shape[1], encoder_hidden_states.shape[1]),
                dtype=jnp.bool_,
            )
            encoder_mask_info = MaskInfo.from_attention_mask(cross_attention_mask)

        inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = (
                jnp.arange(inputs_embeds.shape[1])
                .reshape(1, -1)
                .repeat(
                    inputs_embeds.shape[0],
                    0,
                )
            )

        position_ids = position_ids.astype("i4")
        position_embeds = self.embed_positions(position_ids)

        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.dropout_layer(hidden_states)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        if mode is None:
            mode = (
                common_types.MODE_DECODE
                if input_ids.shape[1] == 1 and past_key_values is not None
                else common_types.MODE_TRAIN
            )
        if past_key_values is None:
            past_key_values = TransformerCache.init_empty(len(self.layers))

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not self.dropout_layer.deterministic and (dropout_probability < self.layerdrop):
                layer_outputs = (None, None, None, None)
            else:
                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    mask_info=mask_info,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_mask_info=encoder_mask_info,
                    mode=mode,
                    cache_view=past_key_values[idx],
                    cache_metadata=cache_metadata,
                    output_attentions=output_attentions,
                )
            past_key_values[idx] = layer_outputs[-1]
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
            past_key_values=past_key_values,
        )


@register_module(TaskType.BASE_MODULE, config=WhisperConfig, model_type="whisper")
class WhisperModel(EasyDeLBaseModule):
    """The base Whisper Model transformer implementing the encoder-decoder architecture.

    Attributes:
        config (WhisperConfig): Configuration object for the model.
        encoder (WhisperEncoder): The encoder stack.
        decoder (WhisperDecoder): The decoder stack.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: WhisperConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the WhisperModel module.

        Args:
            config (WhisperConfig): The configuration object for the model.
            dtype (jnp.dtype): Data type for computations (default: jnp.float32).
            param_dtype (jnp.dtype): Data type for parameters (default: jnp.float32).
            precision (tp.Optional[tp.Union[str, lax.Precision]]): Precision setting for JAX operations (default: None).
            rngs (nn.Rngs): Random number generators.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.encoder = WhisperEncoder(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.decoder = WhisperDecoder(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def _get_decoder_module(self):
        """Returns the decoder module."""
        return self.decoder

    def _get_encoder_module(self):
        """Returns the encoder module."""
        return self.encoder

    def __call__(
        self,
        input_features: jnp.ndarray,
        decoder_input_ids: Int[Array, "batch seq_len"],
        decoder_mask_info: MaskInfo | None = None,
        decoder_position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Forward pass of the complete Whisper model (encoder + decoder).

        Args:
            input_features (jnp.ndarray): Input audio features (batch, num_mel_bins, seq_len).
            decoder_input_ids (jnp.ndarray): Decoder input token IDs (batch, target_seq_len).
            decoder_mask_info (MaskInfo | None): Mask information for decoder self-attention.
            decoder_position_ids (tp.Optional[jnp.ndarray]): Position IDs for decoder inputs.
            mode: Runtime mode (train/decode/prefill).
            past_key_values (tp.Optional[TransformerCache]): Cached key/value states for fast decoding.
            cache_metadata (tp.Optional[TransformerMetadata]): Metadata for paged attention.
            output_attentions (bool): Whether to return attention weights (default: False).
            output_hidden_states (bool): Whether to return hidden states for all layers (default: False).

        Returns:
            Seq2SeqModelOutput: The model output. returns a `Seq2SeqModelOutput`.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        batch_size, sequence_length = decoder_input_ids.shape

        if decoder_position_ids is None:
            if past_key_values is not None:
                raise ValueError("Make sure to provide `decoder_position_ids` when passing `past_key_values`.")

            decoder_position_ids = jnp.broadcast_to(
                jnp.arange(sequence_length)[None, :],
                (batch_size, sequence_length),
            )
        decoder_position_ids = decoder_position_ids.astype("i4")
        encoder_outputs = self.encoder(
            input_features=input_features,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            mask_info=decoder_mask_info,
            position_ids=decoder_position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            encoder_hidden_states=encoder_outputs[0],
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            past_key_values=decoder_outputs.past_key_values,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def decode(
        self,
        encoder_hidden_states: Float[Array, "batch seq_len hidden_dim"],
        decoder_input_ids: Int[Array, "batch seq_len"],
        decoder_mask_info: MaskInfo | None = None,
        decoder_position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Performs decoding using the decoder module.

        Args:
            encoder_hidden_states (jnp.ndarray): Hidden states from the encoder.
            decoder_input_ids (jnp.ndarray): Decoder input token IDs.
            decoder_mask_info (MaskInfo | None): Mask information for decoder self-attention.
            decoder_position_ids (tp.Optional[jnp.ndarray]): Position IDs for decoder inputs.
            mode: Runtime mode (train/decode/prefill).
            past_key_values (tp.Optional[TransformerCache]): Cached key/value states.
            cache_metadata (tp.Optional[TransformerMetadata]): Metadata for paged attention.
            output_attentions (bool): Whether to return attention weights.
            output_hidden_states (bool): Whether to return hidden states for all layers.

        Returns:
            BaseModelOutputWithPastAndCrossAttentions | tuple: Decoder output.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        batch_size, sequence_length = decoder_input_ids.shape

        if decoder_position_ids is None:
            if past_key_values is not None:
                raise ValueError("Make sure to provide `decoder_position_ids` when passing `past_key_values`.")

            decoder_position_ids = jnp.broadcast_to(
                jnp.arange(sequence_length)[None, :],
                (batch_size, sequence_length),
            )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            mask_info=decoder_mask_info,
            position_ids=decoder_position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            encoder_hidden_states=encoder_hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            past_key_values=decoder_outputs.past_key_values,
        )

    def encode(
        self,
        input_features: jnp.ndarray,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Performs encoding using the encoder module.

        Args:
            input_features (jnp.ndarray): Input audio features.
            output_attentions (bool): Whether to return attention weights.
            output_hidden_states (bool): Whether to return hidden states for all layers.

        Returns:
            BaseModelOutput | tuple: Encoder output.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        encoder_outputs = self.encoder(
            input_features=input_features,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        return Seq2SeqModelOutput(
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        """
        return self.encoder

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        """
        return self.decoder

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        Base Models don't have a Language Model Head.
        """
        raise NotImplementedError("The base model does not have a language model head.")

    def get_embedding(self):
        """
        Returns the embedding layer of the decoder.
        """
        return self.decoder.embed_tokens


@register_module(TaskType.SPEECH_SEQUENCE_TO_SEQUENCE, config=WhisperConfig, model_type="whisper")
class WhisperForConditionalGeneration(BaseConditionalGenerationModule[WhisperModel, WhisperConfig]):
    """Whisper encoder-decoder with projection head for speech-to-text generation.

    This model extends the base Whisper architecture with a language modeling head for
    generating text transcriptions and translations from audio. It supports multilingual
    speech recognition, translation to English, and timestamp prediction.

    The model consists of:
        - WhisperModel: The base encoder-decoder transformer
        - proj_out: Linear projection layer mapping decoder hidden states to vocabulary logits

    Attributes:
        model (WhisperModel): The base encoder-decoder model.
        proj_out (RowParallelLinear): Output projection to vocabulary space.
        config (WhisperConfig): Model configuration.
        loss_type (str): Loss type identifier for training ("ForCausalLM").
    """

    loss_type = "ForCausalLM"

    _task_type = TaskType.SPEECH_SEQUENCE_TO_SEQUENCE
    _model_type = "whisper"
    _config_class = WhisperConfig

    def __init__(
        self,
        config: WhisperConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes WhisperForConditionalGeneration.

        Args:
            config (WhisperConfig): Model configuration object specifying architecture
                parameters such as vocab_size, num_mel_bins, layer counts, hidden dimensions,
                and attention head counts.
            dtype (jnp.dtype): Data type for activations and computations during forward pass.
                Default: jnp.bfloat16.
            param_dtype (jnp.dtype): Data type for model parameters (weights and biases).
                Default: jnp.bfloat16.
            precision (str | lax.Precision, optional): JAX precision setting for matrix
                multiplications. Options include 'default', 'high', 'highest', or None.
                Default: None.
            rngs (nn.Rngs): Random number generator state for dropout and parameter
                initialization.
        """
        super().__init__(
            config=config,
            base_model_class=WhisperModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_name="proj_out",
            lm_head_class=RowParallelLinear,
            lm_head_bias=False,
            lm_head_kernel_init=jax.nn.initializers.normal(config.init_std),
            create_lm_head=True,
        )

    def _get_encoder_module(self):
        return self.model.encoder

    def _get_decoder_module(self):
        return self.model.decoder

    def __call__(
        self,
        input_features,
        decoder_input_ids,
        decoder_mask_info: MaskInfo | None = None,
        decoder_position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Forward pass of WhisperForConditionalGeneration.

        Processes audio features through the encoder and generates text through the decoder
        with an optional language modeling head for next-token prediction.

        Args:
            input_features (jnp.ndarray): Log-Mel spectrogram features from audio input.
                Shape: (batch_size, num_mel_bins, sequence_length) where num_mel_bins is
                typically 80 and sequence_length is max_source_positions * 2 (usually 3000).
            decoder_input_ids (jnp.ndarray): Input token IDs for the decoder.
                Shape: (batch_size, target_sequence_length). For training, these are typically
                the shifted target sequences. For generation, starts with decoder_start_token_id.
            decoder_mask_info (MaskInfo, optional): Mask information for decoder self-attention,
                encoding causal attention patterns and padding. If None, creates default causal mask.
            decoder_position_ids (Int[Array, "batch seq_len"], optional): Position indices for
                decoder inputs. Shape: (batch_size, target_sequence_length). If None, uses
                sequential positions [0, 1, 2, ...].
            mode (RUNTIME_MODE_TYPES, optional): Execution mode controlling attention computation.
                Options: MODE_TRAIN (full attention), MODE_DECODE (single-token generation),
                MODE_PREFILL (initial KV cache population). Auto-detected if None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache, optional):
                Cached key/value states from previous decoder layers for fast autoregressive
                generation. Contains cached states for all decoder layers.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata, optional):
                Metadata for managing paged attention caches, including sequence lengths and
                valid token positions.
            apply_lm_head (bool): Whether to apply the language modeling head to produce logits.
                Set to False if you only need hidden states. Default: True.
            output_attentions (bool): Whether to return attention weights from all layers.
                Default: False.
            output_hidden_states (bool): Whether to return hidden states from all layers
                (encoder and decoder). Default: False.

        Returns:
            Seq2SeqLMOutput: Model output containing:
                - logits (jnp.ndarray, optional): Next-token prediction logits if apply_lm_head=True.
                  Shape: (batch_size, target_sequence_length, vocab_size).
                - decoder_hidden_states (tuple[jnp.ndarray], optional): Hidden states from each
                  decoder layer if output_hidden_states=True.
                - decoder_attentions (tuple[jnp.ndarray], optional): Self-attention weights from
                  each decoder layer if output_attentions=True.
                - cross_attentions (tuple[jnp.ndarray], optional): Cross-attention weights from
                  each decoder layer if output_attentions=True.
                - encoder_last_hidden_state (jnp.ndarray): Final encoder hidden states.
                  Shape: (batch_size, encoder_sequence_length, d_model).
                - encoder_hidden_states (tuple[jnp.ndarray], optional): Hidden states from each
                  encoder layer if output_hidden_states=True.
                - encoder_attentions (tuple[jnp.ndarray], optional): Attention weights from each
                  encoder layer if output_attentions=True.
                - past_key_values (TransformerCache | RaggedPagesCache | HybridCache, optional):
                  Updated cache with new key/value states.
        """
        outputs = self.model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            decoder_mask_info=decoder_mask_info,
            decoder_position_ids=decoder_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
        )

        hidden_states = outputs[0]

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        lm_logits = None
        if apply_lm_head:
            lm_logits = self.apply_lm_head(hidden_states)

        return Seq2SeqLMOutput(
            logits=lm_logits,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            past_key_values=outputs.past_key_values,
        )

    def decode(
        self,
        decoder_input_ids,
        encoder_outputs,
        encoder_attention_mask: Bool[Array, "batch seq_len"] | None = None,
        decoder_mask_info: MaskInfo | None = None,
        decoder_position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        encoder_hidden_states = encoder_outputs[0]

        outputs = self.model.decode(
            encoder_hidden_states=encoder_hidden_states,
            decoder_mask_info=decoder_mask_info,
            decoder_input_ids=decoder_input_ids,
            decoder_position_ids=decoder_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
        )
        hidden_states = outputs[0]

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        lm_logits = self.proj_out(
            hidden_states,
            self.model.decoder.embed_tokens.embedding.value.T.astype(self.param_dtype)
            if self.config.tie_word_embeddings
            else None,
        )

        return Seq2SeqLMOutput(
            logits=lm_logits,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            past_key_values=outputs.past_key_values,
        )

    def encode(
        self,
        input_features: jnp.ndarray,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        return self.model.encode(
            input_features=jnp.array(input_features, dtype="f4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

    def generate(
        self,
        input_features,
        generation_config=None,
        logits_processor=None,
        return_timestamps=None,
        task=None,
        language=None,
        is_multilingual=None,
        **kwargs,
    ):
        if generation_config is None:
            generation_config = self.generation_config

        if return_timestamps is not None:
            generation_config.return_timestamps = return_timestamps

        if task is not None:
            generation_config.task = task

        if is_multilingual is not None:
            generation_config.is_multilingual = is_multilingual

        if language is not None:
            generation_config.language = language

        if kwargs is not None and "decoder_input_ids" in kwargs:
            decoder_input_length = len(kwargs["decoder_input_ids"])
        else:
            decoder_input_length = 1

        forced_decoder_ids = []

        if hasattr(generation_config, "is_multilingual") and generation_config.is_multilingual:
            if hasattr(generation_config, "language"):
                forced_decoder_ids.append((1, generation_config.lang_to_id[generation_config.language]))
            else:
                forced_decoder_ids.append((1, None))

            if hasattr(generation_config, "task"):
                forced_decoder_ids.append((2, generation_config.task_to_id[generation_config.task]))
            else:
                forced_decoder_ids.append((2, generation_config.task_to_id["transcribe"]))

        if (
            hasattr(generation_config, "return_timestamps") and generation_config.return_timestamps
        ) or return_timestamps:
            logits_processor = [WhisperTimeStampLogitsProcessor(generation_config, self.config, decoder_input_length)]
        else:
            if forced_decoder_ids and forced_decoder_ids[-1][0] != generation_config.no_timestamps_token_id:
                idx = forced_decoder_ids[-1][0] + 1 if forced_decoder_ids else 1
                forced_decoder_ids.append((idx, generation_config.no_timestamps_token_id))

        if len(forced_decoder_ids) > 0:
            generation_config.forced_decoder_ids = forced_decoder_ids

        return super().generate(
            input_features,
            generation_config,
            logits_processor=logits_processor,
            **kwargs,
        )

    def _force_generate(
        self,
        input_features: jax.Array,
        forced_decoder_ids: jax.Array,
        return_timestamps: bool = False,
        generation_config: tp.Optional["transformers.GenerationConfig"] = None,  # type:ignore
        **kwargs,
    ):
        if generation_config is None:
            generation_config = self.generation_config
        generation_config.forced_decoder_ids = None
        logits_processor = LogitsProcessorList()
        logits_processor.append(ForceTokensLogitsProcessor(forced_decoder_ids))
        if return_timestamps:
            logits_processor.append(WhisperTimeStampLogitsProcessor(generation_config, self.config, 1))
        return super().generate(
            input_features,
            generation_config,
            logits_processor=logits_processor,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        max_length: int,
        pad_token_id: int,
        starts: int | None = None,
        shardings=None,
        attention_mask: jax.Array | None = None,
        decoder_attention_mask: jax.Array | None = None,
        encoder_outputs=None,
        **kwargs,
    ):
        # initializing the cache
        batch_size, seq_length = decoder_input_ids.shape

        if starts is None:
            starts = self.compute_prefill_length(decoder_input_ids, pad_token_id)
        past_key_values = self.init_cache(
            batch_size,
            max_length,
            starts,
            shardings,
            pad_token_id,
        )
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="b1")
        if decoder_attention_mask is not None:
            position_ids = decoder_attention_mask.cumsum(-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, decoder_attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        decoder_mask_info = MaskInfo.from_attention_mask(extended_attention_mask)

        return self.prepare_inputs_for_call(
            **{
                "past_key_values": past_key_values,
                "encoder_outputs": encoder_outputs,
                "encoder_attention_mask": attention_mask,
                "decoder_mask_info": decoder_mask_info,
                "decoder_position_ids": position_ids,
            }
        )

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["decoder_position_ids"] = model_kwargs["decoder_position_ids"][:, -1:] + 1
        return model_kwargs

    def compute_loss(
        self,
        *,
        labels: Array | None = None,
        loss_config: LossConfig | None = None,
        loss_kwargs: dict | None = None,
        **batch,
    ) -> tuple[tp.Any, LossMetrics]:
        if loss_config is None:
            loss_config = LossConfig()
        loss_config.reduction = "mean"
        loss_config.shift_tokens = False
        if labels is not None:
            if batch.get("decoder_input_ids", None) is None and batch.get("decoder_inputs_embeds", None) is None:
                batch["decoder_input_ids"] = shift_tokens_right(
                    labels,
                    self.config.pad_token_id,
                    self.config.decoder_start_token_id,
                )
        return super().compute_loss(
            labels=labels,
            loss_config=loss_config,
            loss_kwargs=loss_kwargs,
            **batch,
        )

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        """
        return self.model.encoder

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        """
        return self.model.decoder

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        """
        return self.proj_out

    def get_embedding(self):
        """
        Returns the embedding layer of the decoder.
        """
        return self.model.decoder.embed_tokens


@register_module(TaskType.AUDIO_CLASSIFICATION, config=WhisperConfig, model_type="whisper")
class WhisperForAudioClassification(EasyDeLBaseModule):
    """Encoder-only Whisper variant with pooling and classifier for audio tagging.

    This model uses the Whisper encoder for audio feature extraction, followed by
    mean-pooling and a classification head for tasks like audio event detection,
    music genre classification, or acoustic scene classification.

    Architecture:
        - WhisperEncoder: Processes log-Mel spectrograms into contextualized representations
        - Optional weighted layer sum: Combines encoder layer outputs with learned weights
        - Projector: Linear layer reducing d_model to classifier_proj_size
        - Mean pooling: Aggregates temporal representations
        - Classifier: Linear layer mapping to num_labels classes

    Attributes:
        encoder (WhisperEncoder): The Whisper encoder stack.
        layer_weights (jnp.ndarray, optional): Learned weights for combining encoder layers
            when use_weighted_layer_sum=True.
        projector (ColumnParallelLinear): Dimension reduction layer before classification.
        classifier (ColumnParallelLinear): Final classification layer.
        config (WhisperConfig): Model configuration.
    """

    def __init__(
        self,
        config: WhisperConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes WhisperForAudioClassification.

        Args:
            config (WhisperConfig): Model configuration object. Must include num_labels
                for the classification task and optionally use_weighted_layer_sum and
                classifier_proj_size.
            dtype (jnp.dtype): Data type for activations and computations during forward pass.
                Default: jnp.bfloat16.
            param_dtype (jnp.dtype): Data type for model parameters (weights and biases).
                Default: jnp.bfloat16.
            precision (str | lax.Precision, optional): JAX precision setting for matrix
                multiplications. Options include 'default', 'high', 'highest', or None.
                Default: None.
            rngs (nn.Rngs): Random number generator state for dropout and parameter
                initialization.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.encoder = WhisperEncoder(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        config.is_encoder_decoder = False
        num_layers = config.num_hidden_layers + 1
        if config.use_weighted_layer_sum:
            self.layer_weights = jnp.repeat(1 / num_layers, num_layers)
        self.projector = ColumnParallelLinear(
            config.d_model,
            config.classifier_proj_size,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.classifier = ColumnParallelLinear(
            config.classifier_proj_size,
            config.num_labels,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        input_features,
        encoder_outputs=None,
        output_attentions=None,
        output_hidden_states: bool = True,
    ):
        """Forward pass of WhisperForAudioClassification.

        Processes audio features through the Whisper encoder, applies optional weighted layer
        summation, pools the representations, and produces classification logits.

        Args:
            input_features (jnp.ndarray): Log-Mel spectrogram features from audio input.
                Shape: (batch_size, num_mel_bins, sequence_length) where num_mel_bins is
                typically 80 and sequence_length is max_source_positions * 2 (usually 3000).
            encoder_outputs (BaseModelOutput, optional): Pre-computed encoder outputs to reuse.
                If provided, skips encoder computation. Must include hidden_states if
                use_weighted_layer_sum=True. Default: None (compute encoder outputs).
            output_attentions (bool, optional): Whether to return attention weights from all
                encoder layers. If None, uses config.output_attentions. Default: None.
            output_hidden_states (bool): Whether to return hidden states from all encoder layers.
                Required when use_weighted_layer_sum=True to compute weighted average across
                layers. Default: True.

        Returns:
            SequenceClassifierOutput: Classification output containing:
                - logits (jnp.ndarray): Classification logits over all label classes.
                  Shape: (batch_size, num_labels).
                - hidden_states (tuple[jnp.ndarray], optional): Hidden states from each encoder
                  layer if output_hidden_states=True. Each tensor has shape
                  (batch_size, sequence_length, d_model).
                - attentions (tuple[jnp.ndarray], optional): Attention weights from each encoder
                  layer if output_attentions=True. Each tensor has shape
                  (batch_size, num_heads, sequence_length, sequence_length).

        Note:
            This model uses mean pooling over the temporal dimension to aggregate encoder
            outputs into a fixed-size representation before classification. If
            use_weighted_layer_sum=True in the config, it computes a learned weighted
            average across all encoder layer outputs instead of using only the final layer.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_features,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

        if self.config.use_weighted_layer_sum:
            hidden_states = jnp.stack(encoder_outputs, axis=1)
            norm_weights = jax.nn.softmax(self.layer_weights, axis=-1)
            hidden_states = jnp.sum(hidden_states * jnp.reshape(norm_weights, [-1, 1, 1]), axis=1)
        else:
            hidden_states = encoder_outputs[0]

        hidden_states = self.projector(hidden_states)
        pooled_output = jnp.mean(hidden_states, axis=1)

        logits = self.classifier(pooled_output)

        return SequenceClassifierOutput(
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        """
        return self.encoder

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        This is an encoder-only model for classification.
        """
        raise NotImplementedError("This is an encoder-only model and does not have a decoder.")

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        This model has an audio classification head, not a language model head.
        """
        raise NotImplementedError("This model has an audio classification head, not a language model head.")

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        The encoder uses convolutional layers for feature extraction, not a standard token embedding.
        Returning the first convolutional layer as the "embedding" layer.
        """
        return self.encoder.conv1
