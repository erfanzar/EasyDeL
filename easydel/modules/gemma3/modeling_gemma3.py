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


from functools import cached_property, partial

import jax
import jax.numpy as jnp
from eformer import common_types
from eformer.escale import apply_logical_sharding
from eformer.loggings import get_logger
from eformer.pytree import auto_pytree
from ejkernel.types import MaskInfo
from flax import nnx as nn
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    DecoderLayerOutput,
    ModelOutput,
    SequenceClassifierOutput,
    VLMCausalLMOutput,
)
from easydel.infra.utils import ACT2FN, ArrayParam, auto_remat, block_wise_ffn
from easydel.layers.attention import FlexibleAttentionModule
from easydel.layers.attention_unified import UnifiedAttention
from easydel.layers.base_modules import BaseCausalLMModule, BaseSequenceClassificationModule, BaseVisionLanguageModule
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
from easydel.layers.norms import float8s
from easydel.modules.auto.auto_modeling import AutoEasyDeLVisionModel

from .gemma3_configuration import Gemma3Config, Gemma3TextConfig

logger = get_logger(__name__)


@auto_pytree
class Gemma3ModelOutputWithPast(ModelOutput):
    r"""
    past_key_values (`tuple(tuple(Array))`):
        Tuple of `tuple(Array)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`Array`, *optional*):
        A `Array` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    last_hidden_state: Array | None = None
    image_hidden_states: Float[Array, "batch seq_len hidden_dim"] | None = None
    past_key_values: TransformerCache | None = None
    hidden_states: tuple[Array] | None = None
    attentions: tuple[Array] | None = None


@auto_pytree
class Gemma3CausalLMOutputWithPast(ModelOutput):
    """
    Base class for Gemma3 causal language model (or autoregressive) outputs.

    Args:
        loss (`Array` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`Array` of shape `(batch_size, sequence_length, config.get_text_config().vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(Array))`):
            Tuple of `tuple(Array)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(Array)`, *optional*, returned when `output_hidden_states=True` is passed or
            when `config.output_hidden_states=True`):
            Tuple of `Array` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(Array)`, *optional*, returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`):
            Tuple of `Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`Array`, *optional*):
            A `Array` of size `(batch_size, sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder after projecting last hidden state.
    """

    loss: Array | None = None
    logits: Array | None = None
    last_hidden_state: Array | None = None
    past_key_values: TransformerCache | None = None
    hidden_states: tuple[Array] | None = None
    attentions: tuple[Array] | None = None
    image_hidden_states: Float[Array, "batch seq_len hidden_dim"] | None = None


class Gemma3RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization for Gemma3 models.

    Implements RMS normalization with Float8 support for efficient computation
    and memory usage in Gemma3 architecture.
    """

    kernel_init = staticmethod(nn.initializers.ones)

    def __init__(
        self,
        config: Gemma3TextConfig,
        param_dtype: jnp.dtype = jnp.float32,
        dim: int | None = None,
        epsilon: float | None = None,
    ):
        self.config = config
        self.epsilon = self.config.rms_norm_eps if epsilon is None else epsilon
        self.param_dtype = param_dtype
        dim = self.config.hidden_size if dim is None else dim
        self.kernel = ArrayParam.bound(
            shape=(dim,),
            dtype=param_dtype,
            init_method="ones",
            key=None,
        )

    def _norm(self, x: jax.Array) -> jax.Array:
        return x * (1 / jnp.sqrt(jnp.power(x, 2).mean(-1, keepdims=True) + self.epsilon))

    def __call__(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> jax.Array:
        variance = self._norm(hidden_states.astype(jnp.float32)).astype(self.param_dtype)
        out = (1 + self.kernel.value.astype(self.param_dtype)) * variance

        if out.dtype in float8s:
            out = out.astype(jnp.bfloat16)
        return out


class Gemma3Attention(UnifiedAttention):
    """Gemma3 Attention with Q/K normalization.

    Inherits Q/K normalization from QKNormAttention.
    Features:
    - Custom Gemma3RMSNorm for Q/K normalization
    - Layer-specific sliding window
    - Custom softmax scaling (query_pre_attn_scalar)
    """

    def __init__(
        self,
        config: Gemma3TextConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        causal: bool = True,
        is_cross_attention: bool = False,
        *,
        rngs: nn.Rngs,
    ):
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"

        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="standard",
            causal=True,
            sliding_window=config.sliding_window if self.is_sliding else None,
            use_qk_norm=True,
        )

        self.layer_idx = layer_idx
        self.is_cross_attention = is_cross_attention
        self.causal = causal

    def _create_q_norm(self, config, dtype, param_dtype, rngs):
        """Override to use Gemma3RMSNorm instead of standard RMSNorm."""
        return Gemma3RMSNorm(config, param_dtype=param_dtype, dim=self.head_dim)

    def _create_k_norm(self, config, dtype, param_dtype, rngs):
        """Override to use Gemma3RMSNorm instead of standard RMSNorm."""
        return Gemma3RMSNorm(config, param_dtype=param_dtype, dim=self.head_dim)

    def _create_attention_performer(self, config, rngs):
        """Override to use custom softmax scale with query_pre_attn_scalar."""
        return FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=config.query_pre_attn_scalar**-0.5,
            dropout_prob=config.attention_dropout,
        )

    def _postprocess_qkv(self, query_states, key_states, value_states):
        """Post-process Q/K/V after projection and reshape, before RoPE/sharding.

        **KEY METHOD**: Override this to apply Q/K normalization or other transformations.
        By default, applies Q/K norm if configured, otherwise returns unchanged.

        Pattern for Q/K normalization:
            >>> def _postprocess_qkv(self, q, k, v):
            ...     if hasattr(self, 'q_norm'):
            ...         q = self.q_norm(q)
            ...         k = self.k_norm(k)
            ...     return q, k, v

        Args:
            query_states: Query tensor [batch, seq_len, num_heads, head_dim]
            key_states: Key tensor [batch, seq_len, num_kv_heads, head_dim]
            value_states: Value tensor [batch, seq_len, num_kv_heads, head_dim]

        Returns:
            Tuple of (processed_query, processed_key, processed_value)
        """

        return self.q_norm(query_states), self.k_norm(key_states), value_states


class Gemma3MLP(nn.Module):
    """Multi-Layer Perceptron module for Gemma3 models.

    Implements the feedforward network with gated activation functions
    and optional Float8 scaling for improved performance.
    """

    def __init__(
        self,
        config: Gemma3TextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        embed_dim = self.config.hidden_size
        inner_dim = self.config.intermediate_size if self.config.intermediate_size is not None else 4 * embed_dim
        kernel_init = jax.nn.initializers.normal(config.initializer_range)

        self.act = ACT2FN[self.config.hidden_activation]

        column_parallel_linear = partial(
            ColumnParallelLinear,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        row_parallel_linear = partial(
            RowParallelLinear,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.gate_proj = column_parallel_linear(embed_dim, inner_dim)
        self.down_proj = row_parallel_linear(inner_dim, embed_dim)
        self.up_proj = column_parallel_linear(embed_dim, inner_dim)

    def __call__(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        gate = checkpoint_name(self.act(self.gate_proj(hidden_states)), "mlp_gate")
        up = checkpoint_name(self.up_proj(hidden_states), "mlp_up")
        hidden_states = checkpoint_name(self.down_proj(gate * up), "mlp_down")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return checkpoint_name(hidden_states, "mlp_output")


class Gemma3DecoderLayer(nn.Module):
    """Single decoder layer for Gemma3 models.

    Combines self-attention, optional cross-attention, and feedforward networks
    with residual connections and layer normalization.
    """

    def __init__(
        self,
        config: Gemma3TextConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        mlp_block = Gemma3MLP
        attn_block = Gemma3Attention

        attn_block, mlp_block = auto_remat(
            attn_block,
            mlp_block,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )

        self.self_attn = attn_block(
            self.config,
            layer_idx=self.layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.mlp = mlp_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.input_layernorm = Gemma3RMSNorm(self.config, param_dtype=self.param_dtype)
        self.post_attention_layernorm = Gemma3RMSNorm(self.config, param_dtype=self.param_dtype)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(self.config, param_dtype=self.param_dtype)
        self.post_feedforward_layernorm = Gemma3RMSNorm(self.config, param_dtype=self.param_dtype)

        self.is_sliding = self.self_attn.is_sliding
        self.sliding_window = config.sliding_window

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
        default_frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ):
        """
        Forward pass of the module block.

        Args:
            hidden_states (Array): Input hidden states.
            attention_mask (Array): Mask to apply on the attention scores.
            position_ids (Array): Position indices for the tokens.
            causal_mask (Array): Causal mask for ensuring autoregressive behavior.
            deterministic (bool): If True, disables dropout for deterministic behavior.
            init_cache (bool): If True, initializes cache for caching keys and values.
            output_attentions (bool): If True, outputs attention weights alongside the hidden states.
            fcm_mask (tp.Optional[Array]): fcm mask to be combined with attn mask and causal mask.
        Returns:
            tp.Tuple[Array, Array]: A tuple containing the attention output and the attention weights.
        """
        residual = hidden_states
        frequencies = default_frequencies if self.is_sliding else frequencies
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        attn_outputs = self.self_attn(
            hidden_states,
            mask_info,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            output_attentions,
            frequencies,
        )
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        hidden_states = self.post_attention_layernorm(attn_outputs.attention_output)
        hidden_states = checkpoint_name(residual + hidden_states, "residual")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        if self.config.use_scan_mlp:
            hidden_states = block_wise_ffn(
                self.mlp,
                hidden_states,
                self.config.scan_mlp_chunk_size,
            )
        else:
            hidden_states = self.mlp(hidden_states)

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = checkpoint_name(residual + hidden_states, "residual")
        hidden_states = checkpoint_name(hidden_states, "layer_output")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
            cache_view=attn_outputs.cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=Gemma3TextConfig, model_type="gemma3_text")
class Gemma3TextModel(EasyDeLBaseModule):
    """Decoder-only Gemma3 text transformer with embeddings and stacked decoder layers."""

    def __init__(
        self,
        config: Gemma3TextConfig,
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
        self.hidden_size = self.config.hidden_size

        embed_block = auto_remat(
            nn.Embed,
            policy=self.config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.embed_tokens = embed_block(
            self.config.vocab_size,
            self.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.layers = [
            Gemma3DecoderLayer(
                self.config,
                layer_idx=i,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for i in range(self.config.num_hidden_layers)
        ]
        self.norm = Gemma3RMSNorm(self.config, param_dtype=self.dtype)

    @cached_property
    def default_frequencies(self):
        from easydel.infra.utils import ModuleCaches
        from easydel.layers.rotary_embedding import get_frequencies

        frequencies = get_frequencies(
            head_size=self.config.head_dim,
            rotary_dim=self.config.head_dim,
            max_position=self.config.granted_freq_max_position_embedding,
            base=self.config.rope_local_base_freq,
            rope_scaling=None,
        ).astype(jnp.bfloat16)

        return ModuleCaches(frequencies)

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        token_type_ids: Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
    ) -> BaseModelOutput:
        """
        Forward pass through the Gemma2 module.

        Args:
            input_ids (Array): Input tensor containing token IDs.
            attention_mask (Array): Mask for attention.
            position_ids (Array): Positional indices.
            inputs_embeds (tp.Optional[Array]): Embedded input tensor.
            output_attentions (tp.Optional[bool]): If True, output attention weights.
            output_hidden_states (tp.Optional[bool]): If True, output hidden states.
            init_cache (bool): If True, initialize cache for decoding.
            deterministic (bool): If True, disable dropout.

        Returns:
            BaseModelOutput | tp.Tuple: Model output, either as a named tuple or a standard tuple.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = checkpoint_name(self.embed_tokens(input_ids.astype("i4")), "embeddings") * (
                self.config.hidden_size**0.5
            )
        sequence_length = inputs_embeds.shape[1]

        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        # HF Gemma3 uses per-layer attention masks:
        # - full attention: causal
        # - sliding attention: causal + sliding window
        # For multimodal (token_type_ids), HF *OR*s a bidirectional image-block mask
        # with the base mask (so image tokens can see each other even outside the window).
        #
        # To match that, we construct two MaskInfo variants and select per layer:
        #   full:    causal | image_block_mask
        #   sliding: (causal & window) | image_block_mask
        mask_info_full = mask_info
        mask_info_sliding = mask_info
        if token_type_ids is not None:
            token_type_ids = jnp.asarray(token_type_ids, dtype=jnp.int32)
            is_image = token_type_ids == 1
            prev_is_image = jnp.pad(is_image, ((0, 0), (1, 0)), constant_values=False)[:, :-1]
            new_image_start = is_image & (~prev_is_image)
            image_group_ids = jnp.cumsum(new_image_start.astype(jnp.int32), axis=1) - 1
            # Use per-image group IDs so equality masking matches HF's "same_image_block" logic.
            # Text tokens are 0 (disabled by MaskInfo.apply_token_type_ids' default zero_policy="q").
            grouped_token_types = jnp.where(is_image, image_group_ids + 1, 0).astype(jnp.int32)

            causal_mask_info = mask_info.apply_causal()
            mask_info_full = causal_mask_info.apply_token_type_ids(grouped_token_types)
            mask_info_sliding = causal_mask_info.apply_sliding_window(self.config.sliding_window).apply_token_type_ids(
                grouped_token_types
            )

            # We've baked causal (and for sliding layers, window) into the attention mask, so the
            # attention kernel shouldn't apply causal/window again.
            object.__setattr__(mask_info_full, "_causal_baked", True)
            object.__setattr__(mask_info_sliding, "_causal_baked", True)
        if position_ids is None:
            position_ids = mask_info.q_position_ids
        inputs_embeds = inputs_embeds
        assert sequence_length <= self.config.max_position_embeddings, (
            f"Maximum Position Embedding Reached ! "
            f"(Excepted <= {self.config.max_position_embeddings} got {sequence_length})"
        )

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
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for idx, block in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_mask_info = (
                mask_info_sliding if self.config.layer_types[idx] == "sliding_attention" else mask_info_full
            )
            layer_outputs = block(
                hidden_states=hidden_states,
                mask_info=layer_mask_info,
                position_ids=position_ids,
                mode=mode,
                cache_view=past_key_values.views[idx],
                cache_metadata=cache_metadata,
                output_attentions=output_attentions,
                frequencies=self.frequencies,
                default_frequencies=self.default_frequencies,
            )
            hidden_states = layer_outputs.hidden_states

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            past_key_values[idx] = layer_outputs.cache_view

        hidden_states = self.norm(hidden_states)
        hidden_states = checkpoint_name(hidden_states, "model_output")
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            past_key_values=past_key_values,
        )

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        Decoder-Only models don't have an encoder.
        """
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        """
        return self

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        Base Models don't have a Language Model Head.
        """
        raise NotImplementedError("The base model does not have a language model head.")

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.embed_tokens


@register_module(TaskType.CAUSAL_LM, config=Gemma3TextConfig, model_type="gemma3_text")
class Gemma3ForCausalLM(BaseCausalLMModule[Gemma3TextModel, Gemma3TextConfig]):
    """Gemma3 model with a language modeling head for causal language modeling tasks."""

    _task_type = TaskType.CAUSAL_LM
    _model_type = "gemma3_text"
    _config_class = Gemma3TextConfig

    def __init__(
        self,
        config: Gemma3TextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        if param_dtype == jnp.float16 or param_dtype == "f2":
            logger.error(
                "Gemma-3's recommended dtype is bfloat16, but you are using float16. "
                "This may result in junk responses or incorrect predictions."
            )
        super().__init__(
            config=config,
            base_model_class=Gemma3TextModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        token_type_ids: Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> CausalLMOutput:  # type:ignore
        """
        Forward pass through the Gemma3 model.

        Args:
            input_ids (tp.Optional[Array]): Input tensor containing token IDs.
            attention_mask (tp.Optional[Array]): Mask for attention.
            position_ids (tp.Optional[Array]): Positional indices.
            token_type_ids (tp.Optional[Array]): Token type IDs for handling different types of tokens.
            inputs_embeds (tp.Optional[Array]): Embedded input tensor.
            output_attentions (tp.Optional[bool]): If True, output attention weights.
            output_hidden_states (tp.Optional[bool]): If True, output hidden states.
            past_key_values (tp.Optional[TransformerCache | RaggedPagesCache]): Cached key values for
                faster inference.
            cache_metadata (tp.Optional[TransformerMetadata | RaggedPagesMetadata]): Metadata for cache handling.

        Returns:
            CausalLMOutput | tp.Tuple: Model output, either as a named tuple or a standard tuple.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            inputs_embeds=inputs_embeds,
            token_type_ids=token_type_ids,
        )

        hidden_states = outputs.last_hidden_state

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        lm_logits = None
        if apply_lm_head:
            lm_logits = checkpoint_name(self.apply_lm_head(hidden_states), "lm_head_output")
        if self.config.final_logit_softcapping is not None:
            cap = jnp.array(self.config.final_logit_softcapping, dtype=lm_logits.dtype)
            lm_logits = cap * jax.nn.tanh(lm_logits / cap)

        return CausalLMOutput(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            last_hidden_state=outputs.last_hidden_state,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
        )

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        Decoder-Only models don't have an encoder.
        """
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        """
        return self.model.get_decoder()

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        Base Models don't have a Language Model Head.
        """
        return self.lm_head

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.model.get_embedding()


@register_module(TaskType.SEQUENCE_CLASSIFICATION, config=Gemma3TextConfig, model_type="gemma3_text")
class Gemma3ForSequenceClassification(BaseSequenceClassificationModule[Gemma3TextModel, Gemma3TextConfig]):
    """Text-only Gemma3 backbone with a classification head for sequence tasks."""

    _task_type = TaskType.SEQUENCE_CLASSIFICATION
    _model_type = "gemma3_text"
    _config_class = Gemma3TextConfig

    def __init__(
        self,
        config: Gemma3TextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(
            config=config,
            base_model_class=Gemma3TextModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            pooling_strategy="last",
            score_head_bias=False,
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
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> SequenceClassifierOutput:
        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = jnp.argmax(jnp.equal(input_ids, self.config.pad_token_id).astype("i4"), -1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
            else:
                sequence_lengths = -1

        pooled_logits = logits[jnp.arange(batch_size), sequence_lengths]

        return SequenceClassifierOutput(
            logits=pooled_logits,
            past_key_values=past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        Decoder-Only models don't have an encoder.
        """
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        """
        return self.model

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        """
        return self.lm_head

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.model.get_embedding()

    def get_task_head(self):
        """Returns the sequence classification head."""
        return self.score


class Gemma3MultiModalProjector(nn.Module):
    """Multi-modal projector for Gemma3 vision-language models.

    Projects vision features into the text embedding space, enabling
    cross-modal understanding and generation in Gemma3.
    """

    def __init__(
        self,
        config: Gemma3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs

        self.mm_input_projection_weight = ArrayParam.bound(
            shape=(
                config.get_text_config().hidden_size,
                config.vision_config.hidden_size,
            ),
            dtype=param_dtype,
            init_method="zeros",
            key=None,
        )
        self.mm_soft_emb_norm = Gemma3RMSNorm(
            config.vision_config,
            param_dtype=param_dtype,
            dim=config.vision_config.hidden_size,
            epsilon=config.vision_config.layer_norm_eps,
        )
        self.patches_per_image = int(config.vision_config.image_size // config.vision_config.patch_size)
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        kernel_size = self.patches_per_image // self.tokens_per_side
        self.kernel_size = kernel_size
        self.avg_pool = lambda x: jax.lax.reduce_window(
            x,
            init_value=0.0,
            computation=jax.lax.add,
            window_dimensions=(1, 1, kernel_size, kernel_size),
            window_strides=(1, 1, kernel_size, kernel_size),
            padding="VALID",
        ) / (kernel_size * kernel_size)

    def __call__(self, vision_outputs):
        batch_size, _, seq_length = vision_outputs.shape

        reshaped_vision_outputs = jnp.transpose(vision_outputs, (0, 2, 1))

        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            batch_size,
            seq_length,
            self.patches_per_image,
            self.patches_per_image,
        )
        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = pooled_vision_outputs.reshape(batch_size, seq_length, -1)
        pooled_vision_outputs = jnp.transpose(pooled_vision_outputs, (0, 2, 1))
        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

        projected_vision_outputs = jax.lax.dot_general(
            normed_vision_outputs,
            self.mm_input_projection_weight.T,
            (((normed_vision_outputs.ndim - 1), (0,)), ((), ())),
        )
        return projected_vision_outputs.astype(vision_outputs.dtype)


@register_module(TaskType.BASE_MODULE, config=Gemma3Config, model_type="gemma3")
class Gemma3Model(EasyDeLBaseModule):
    """Multimodal Gemma3 stack combining a vision tower, projector, and language model."""

    def __init__(
        self,
        config: Gemma3Config,
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
        self.vision_tower = AutoEasyDeLVisionModel.from_config(
            config=config.vision_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.multi_modal_projector = Gemma3MultiModalProjector(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.vocab_size = config.get_text_config().vocab_size
        self.language_model = Gemma3TextModel(
            config=config.get_text_config(),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def get_image_features(self, pixel_values: Array) -> Array:
        vision_outputs = self.vision_tower(pixel_values=pixel_values).last_hidden_state
        image_features = self.multi_modal_projector(vision_outputs)
        return image_features

    def compute_embedding(
        self,
        input_ids: Int[Array, "batch seq_len"],
        *,
        image_features: Array | None = None,
        pixel_values: Array | None = None,
        **kwargs,
    ) -> Array:
        if input_ids is None:
            raise ValueError("`input_ids` must be provided when calling `compute_embedding`.")

        image_token_id = self.config.image_token_id
        if image_token_id >= self.vocab_size:
            llm_input_ids = jnp.where(input_ids == image_token_id, 0, input_ids)
        else:
            llm_input_ids = input_ids

        hidden_size = self.config.get_text_config().hidden_size
        inputs_embeds = super().compute_embedding(llm_input_ids) * (hidden_size**0.5)

        if image_features is None and pixel_values is not None:
            image_features = self.get_image_features(pixel_values)

        if image_features is not None:
            multimodal_embeddings = image_features.reshape(-1, image_features.shape[-1]).astype(inputs_embeds.dtype)
            inputs_embeds = BaseVisionLanguageModule.merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                placeholder_token_id=image_token_id,
            )

        return inputs_embeds

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] = None,
        pixel_values: Array = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        token_type_ids: Array | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **lm_kwargs,
    ) -> Gemma3ModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)

        if inputs_embeds is None:
            inputs_embeds = self.compute_embedding(
                input_ids,
                image_features=image_features,
            )
        elif image_features is not None:
            if input_ids is None:
                special_image_mask = inputs_embeds == self.get_embedding()(
                    jnp.array(self.config.image_token_id, dtype="i4")
                )
            else:
                special_image_mask = jnp.expand_dims(input_ids == self.config.image_token_id, axis=-1)
                special_image_mask = jnp.broadcast_to(special_image_mask, inputs_embeds.shape)
            inputs_embeds = jnp.place(
                inputs_embeds,
                special_image_mask,
                image_features.astype(inputs_embeds.dtype),
                inplace=False,
            )

        outputs = self.language_model(
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            inputs_embeds=inputs_embeds,
            token_type_ids=token_type_ids,
            **lm_kwargs,
        )
        return Gemma3ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )

    def init_cache(
        self,
        batch_size,
        max_length,
        starts=None,
        shardings=None,
        pad_token_id=None,
    ):
        return self.language_model.init_cache(batch_size, max_length, starts, shardings, pad_token_id)

    def prepare_inputs_for_generation(
        self,
        input_ids: Int[Array, "batch seq_len"],
        max_length: int,
        pad_token_id: int,
        starts: int | None = None,
        pixel_values: Array | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        token_type_ids: Array | None = None,
    ):
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            max_length=max_length,
            pad_token_id=pad_token_id,
            starts=starts,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        model_inputs["pixel_values"] = pixel_values
        return model_inputs

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs = super().update_inputs_for_generation(model_outputs, model_kwargs)
        model_kwargs.pop("pixel_values", None)  # only effect first iter
        model_kwargs.pop("token_type_ids", None)  # only effect first iter
        return model_kwargs

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        Gemma3 is a multi-modal model with a vision tower, but for typical LLM usage,
        it's considered a decoder-only architecture.
        """
        return self.vision_tower

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        """
        return self.language_model.get_decoder()

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        Base Models don't have a Language Model Head.
        """
        raise NotImplementedError("The base model does not have a language model head.")

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.language_model.get_embedding()


@register_module(TaskType.IMAGE_TEXT_TO_TEXT, config=Gemma3Config, model_type="gemma3")
class Gemma3ForConditionalGeneration(BaseVisionLanguageModule[Gemma3Model, Gemma3Config]):
    """Gemma3 multimodal language model for conditional generation.

    Combines a vision tower and a language model with a multi-modal projector.
    Inherits from BaseVisionLanguageModule to leverage common VLM infrastructure.

    Features a custom apply_lm_head with final_logit_softcapping for improved
    training stability.

    Class Attributes:
        _task_type: IMAGE_TEXT_TO_TEXT task type
        _model_type: "gemma3" model identifier
        _supports_video: False (Gemma3 is image-only)
        _uses_mrope: False (uses standard RoPE)
    """

    # Class attributes for registration and capabilities
    _task_type = TaskType.IMAGE_TEXT_TO_TEXT
    _model_type = "gemma3"
    _config_class = Gemma3Config
    _auto_register = False  # Already registered via decorator
    _supports_video = False
    _uses_mrope = False

    # Component name mapping
    _vision_tower_name = "vision_tower"
    _projector_name = "multi_modal_projector"
    _language_model_name = "language_model"

    loss_type = "ForCausalLM"

    def __init__(
        self,
        config: Gemma3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the Gemma3ForConditionalGeneration model."""
        super().__init__(
            config=config,
            base_model_class=Gemma3Model,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            # VLM-specific configuration
            vision_feature_layer=getattr(config, "vision_feature_layer", -1),
            vision_feature_select_strategy=getattr(config, "vision_feature_select_strategy", "default"),
            image_token_index=getattr(config, "image_token_id", None),
            # LM head configuration
            tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
            lm_head_bias=False,
        )

    def get_image_features(
        self,
        pixel_values: Float[Array, "batch channels height width"],
        **kwargs,
    ) -> Float[Array, "batch num_patches hidden"]:
        """Extract and project image features from pixel values.

        Delegates to the base model's get_image_features implementation.

        Args:
            pixel_values: Input image pixel values
            **kwargs: Additional arguments (unused for Gemma3)

        Returns:
            Projected image features ready for merging with text embeddings
        """
        return self.base_model.get_image_features(pixel_values)

    def compute_embedding(self, input_ids, *args, **kwargs):
        return self.base_model.compute_embedding(input_ids, *args, **kwargs)

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] = None,
        pixel_values: Array = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        token_type_ids: Array | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **lm_kwargs,
    ) -> VLMCausalLMOutput:
        """Forward pass for the Gemma3 model.

        Args:
            input_ids: Input token IDs (batch_size, sequence_length)
            pixel_values: Input pixel values for images
            attention_mask: Attention mask
            mask_info: Mask information
            position_ids: Position IDs for text
            mode: Runtime mode
            past_key_values: Cached keys/values for language model
            cache_metadata: Metadata for paged attention
            apply_lm_head: Whether to apply the LM head
            token_type_ids: Token type IDs for distinguishing image/text tokens
            inputs_embeds: Input embeddings (alternative to input_ids)
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            **lm_kwargs: Additional arguments passed to the language model

        Returns:
            VLMCausalLMOutput: Model outputs including logits and optional states
        """
        # Forward through base model
        outputs = self.base_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            inputs_embeds=inputs_embeds,
            token_type_ids=token_type_ids,
            **lm_kwargs,
        )
        hidden_states = outputs.last_hidden_state

        lm_logits = None
        if apply_lm_head:
            lm_logits = checkpoint_name(self.apply_lm_head(hidden_states), "lm_head_output")

        return VLMCausalLMOutput(
            logits=lm_logits,
            last_hidden_state=hidden_states,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states if pixel_values is not None else None,
        )

    def apply_lm_head(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> Array:
        """Apply the language modeling head with optional logit softcapping.

        Gemma3 uses final_logit_softcapping to prevent extreme logit values,
        which improves training stability.

        Args:
            hidden_states: Hidden states from the model

        Returns:
            LM logits with optional softcapping applied
        """
        lm_logits = self.lm_head(hidden_states)
        if self.config.get_text_config().final_logit_softcapping is not None:
            cap = jnp.array(self.config.get_text_config().final_logit_softcapping, dtype=lm_logits.dtype)
            lm_logits = cap * jax.nn.tanh(lm_logits / cap)
        return lm_logits

    def init_cache(self, batch_size, max_length, starts=None, shardings=None, pad_token_id=None):
        """Initialize KV cache for generation."""
        return self.base_model.init_cache(batch_size, max_length, starts, shardings, pad_token_id)

    def get_vision_tower(self) -> nn.Module:
        """Returns the vision tower component."""
        return self.base_model.vision_tower

    def get_projector(self) -> nn.Module:
        """Returns the multimodal projector component."""
        return self.base_model.multi_modal_projector

    def get_language_model(self) -> nn.Module:
        """Returns the language model component."""
        return self.base_model.language_model
