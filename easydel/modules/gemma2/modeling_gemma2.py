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


from functools import partial

import jax
import jax.numpy as jnp
from eformer import common_types
from eformer.escale import apply_logical_sharding
from eformer.loggings import get_logger
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
    SequenceClassifierOutput,
)
from easydel.infra.utils import ACT2FN, ArrayParam, auto_remat, block_wise_ffn
from easydel.layers.attention import FlexibleAttentionModule
from easydel.layers.attention_unified import UnifiedAttention
from easydel.layers.base_modules import BaseCausalLMModule, BaseSequenceClassificationModule
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

from .gemma2_configuration import Gemma2Config

logger = get_logger(__name__)


class Gemma2RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization for Gemma2 models.

    This normalization technique normalizes the inputs by the root mean square,
    providing stability during training while being computationally efficient.
    """

    kernel_init = staticmethod(nn.initializers.ones)

    def __init__(self, config: Gemma2Config, dtype: jnp.dtype = jnp.float32):
        self.config = config
        self.epsilon = self.config.rms_norm_eps
        self.dtype = dtype
        self.kernel = ArrayParam.bound(
            shape=(self.config.hidden_size,),
            dtype=dtype,
            init_method="ones",
            key=None,
        )

    def __call__(self, hidden_states):
        variance = hidden_states.astype(jnp.float32)
        variance = jnp.power(variance, 2)
        variance = variance.mean(-1, keepdims=True)
        hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)

        return (1 + self.kernel.value.astype(self.dtype)) * jnp.asarray(hidden_states, dtype=self.dtype)


class Gemma2Attention(UnifiedAttention):
    """Multi-head attention layer with RoPE embeddings for Gemma2 models.

    Inherits from UnifiedAttention with Gemma2-specific customizations:
    - Sliding window attention (layer-specific)
    - Custom query pre-attention scalar
    """

    def __init__(
        self,
        config: Gemma2Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        causal: bool = True,
        is_cross_attention: bool = False,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Gemma2 attention with sliding window configuration."""
        # Set layer-specific attributes before super().__init__
        self.is_cross_attention = is_cross_attention

        super().__init__(
            config,
            dtype,
            param_dtype,
            precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="standard",
            causal=causal,
            sliding_window=config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None,
        )

        # Gemma2-specific attributes
        self.attention_softmax_in_fp32 = self.dtype is not jnp.float32

    def _create_rotary(self, config: Gemma2Config, dtype: jnp.dtype):
        """Create Gemma2-specific rotary embedding layer."""
        return config.get_basic_rope(dtype, self.head_dim, self.head_dim, True)

    def _create_attention_performer(self, config: Gemma2Config, rngs: nn.Rngs):
        """Create attention performer with Gemma2's custom softmax scale."""
        return FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=config.query_pre_attn_scalar**-0.5,
            dropout_prob=config.attention_dropout,
        )

    def _merge_heads(self, hidden_states):
        """
        Merges the attention heads into a single hidden state tensor.

        Args:
            hidden_states (Array): The hidden states with separate head dimensions.

        Returns:
            Array: The hidden states with merged head dimensions.
        """
        return hidden_states.reshape((*hidden_states.shape[:2], self.num_heads * self.head_dim))

    def _split_heads(self, hidden_states, num_heads):
        return hidden_states.reshape((*hidden_states.shape[:2], num_heads, self.head_dim))


class Gemma2MLP(nn.Module):
    """Multi-Layer Perceptron module for Gemma2 models.

    Implements the feedforward network component of the transformer architecture
    with gated linear units and optional activation functions.
    """

    def __init__(
        self,
        config: Gemma2Config,
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
        self.gate_proj = column_parallel_linear(
            embed_dim,
            inner_dim,
            rngs=rngs,
        )
        self.down_proj = row_parallel_linear(
            inner_dim,
            embed_dim,
            rngs=rngs,
        )
        self.up_proj = column_parallel_linear(
            embed_dim,
            inner_dim,
            rngs=rngs,
        )

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


class Gemma2DecoderLayer(nn.Module):
    """Single decoder layer for Gemma2 models.

    Combines multi-head attention and feedforward networks with residual connections
    and layer normalization to form a complete transformer decoder layer.
    """

    def __init__(
        self,
        config: Gemma2Config,
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
        mlp_block = Gemma2MLP
        attn_block = Gemma2Attention

        attn_block, mlp_block = auto_remat(
            attn_block,
            mlp_block,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.is_sliding = bool(self.layer_idx % 2)
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

        self.input_layernorm = Gemma2RMSNorm(self.config, dtype=self.dtype)
        self.post_attention_layernorm = Gemma2RMSNorm(self.config, dtype=self.dtype)
        self.pre_feedforward_layernorm = Gemma2RMSNorm(self.config, dtype=self.dtype)
        self.post_feedforward_layernorm = Gemma2RMSNorm(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ):
        """
        Forward pass of the module block.

        Args:
            hidden_states (Array): Input hidden states.
            attention_mask (Array): Mask to apply on the attention scores.
            position_ids (Array): Position indices for the tokens.
            causal_mask (Array): Causal mask for ensuring autoregressive behavior.
            segment_ids (tp.Optional[Array]): Segment IDs for segment-based attention (optional).
            deterministic (bool): If True, disables dropout for deterministic behavior.
            init_cache (bool): If True, initializes cache for caching keys and values.
            output_attentions (bool): If True, outputs attention weights alongside the hidden states.
            fcm_mask (tp.Optional[Array]): fcm mask to be combined with attn mask and causal mask.
        Returns:
            tp.Tuple[Array, Array]: A tuple containing the attention output and the attention weights.
        """
        residual = hidden_states

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

        hidden_states = self.post_attention_layernorm(attn_outputs.attention_output)
        hidden_states = checkpoint_name(residual + hidden_states, "residual")

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
        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
            cache_view=attn_outputs.cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=Gemma2Config, model_type="gemma2")
class Gemma2Model(EasyDeLBaseModule):
    """Decoder-only Gemma2 transformer composed of embedding, decoder stack, and final norm."""

    def __init__(
        self,
        config: Gemma2Config,
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
            policy=config.gradient_checkpointing,
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
            Gemma2DecoderLayer(
                self.config,
                layer_idx=i,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for i in range(self.config.num_hidden_layers)
        ]
        self.norm = Gemma2RMSNorm(self.config, dtype=self.dtype)

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
    ) -> BaseModelOutput:
        """Forward pass through the Gemma2 base model.

        Processes input tokens through token embedding, applies scaling, and passes them through
        the decoder layers with alternating sliding window and full attention patterns.

        Args:
            input_ids (Int[Array, "batch seq_len"], optional): Input token IDs.
                Shape: (batch_size, sequence_length). Must be provided if inputs_embeds is None.
            inputs_embeds (Float[Array, "batch seq_len hidden_dim"], optional): Pre-computed
                input embeddings. Shape: (batch_size, sequence_length, hidden_size).
                Must be provided if input_ids is None.
            attention_mask (Bool[Array, "batch seq_len"], optional): Attention mask indicating
                which tokens should be attended to (True) and which should be ignored (False).
                Shape: (batch_size, sequence_length). Default: None (all tokens attended).
            mask_info (MaskInfo, optional): Pre-computed mask information encoding attention
                patterns and positions. If None, computed from attention_mask or input_ids.
            position_ids (Int[Array, "batch seq_len"], optional): Position indices for each token.
                Shape: (batch_size, sequence_length). If None, uses sequential positions.
            mode (RUNTIME_MODE_TYPES, optional): Execution mode controlling attention computation.
                Options: MODE_TRAIN (full attention), MODE_DECODE (single-token), MODE_PREFILL.
                Auto-detected if None based on sequence length and cache presence.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache, optional):
                Cached key-value states from previous forward passes for efficient autoregressive
                generation. Hybrid cache recommended for Gemma2's mixed attention pattern.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata, optional):
                Metadata for managing paged attention caches in serving scenarios.
            output_attentions (bool, optional): Whether to return attention weights from all layers.
                Default: None (uses config value).
            output_hidden_states (bool, optional): Whether to return hidden states from all layers.
                Default: None (uses config value).

        Returns:
            BaseModelOutput: Model outputs containing:
                - last_hidden_state (jnp.ndarray): Final layer hidden states after RMSNorm.
                  Shape: (batch_size, sequence_length, hidden_size).
                - hidden_states (tuple[jnp.ndarray], optional): Hidden states from each layer
                  if output_hidden_states=True. Each tensor has shape
                  (batch_size, sequence_length, hidden_size).
                - attentions (tuple[jnp.ndarray], optional): Attention weights from each layer
                  if output_attentions=True. Each tensor has shape
                  (batch_size, num_heads, sequence_length, key_value_length).
                - past_key_values (TransformerCache | RaggedPagesCache | HybridCache): Updated
                  cache with new key-value states for all layers.

        Raises:
            ValueError: If both input_ids and inputs_embeds are None or both are provided.
            AssertionError: If sequence_length exceeds max_position_embeddings.

        Note:
            Gemma2 applies sqrt(hidden_size) scaling to input embeddings before processing.
            The model alternates between sliding window attention (4096 tokens) on odd layers
            and full attention on even layers for efficiency.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = checkpoint_name(self.embed_tokens(input_ids.astype("i4")), "embeddings")
        sequence_length = inputs_embeds.shape[1]

        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        if position_ids is None:
            position_ids = mask_info.q_position_ids
        inputs_embeds = inputs_embeds * (self.config.hidden_size**0.5)
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
            layer_outputs: DecoderLayerOutput = block(
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


@register_module(TaskType.CAUSAL_LM, config=Gemma2Config, model_type="gemma2")
class Gemma2ForCausalLM(BaseCausalLMModule[Gemma2Model, Gemma2Config]):
    """Gemma2 model with a language modeling head for causal language modeling tasks."""

    _task_type = TaskType.CAUSAL_LM
    _model_type = "gemma2"
    _config_class = Gemma2Config

    def __init__(
        self,
        config: Gemma2Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(
            config=config,
            base_model_class=Gemma2Model,
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
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> CausalLMOutput:
        """Forward pass of the Gemma2 causal language model.

        Processes input tokens through the base Gemma2 model and applies the language modeling
        head to produce next-token prediction logits with optional soft-capping.

        Args:
            input_ids (Int[Array, "batch seq_len"], optional): Input token IDs.
                Shape: (batch_size, sequence_length). Must be provided if inputs_embeds is None.
            inputs_embeds (Float[Array, "batch seq_len hidden_dim"], optional): Pre-computed
                input embeddings. Shape: (batch_size, sequence_length, hidden_size).
                Must be provided if input_ids is None.
            attention_mask (Bool[Array, "batch seq_len"], optional): Attention mask indicating
                which tokens should be attended to. Shape: (batch_size, sequence_length).
                Default: None (all tokens attended).
            mask_info (MaskInfo, optional): Pre-computed mask information. If None, computed
                from attention_mask or input_ids.
            position_ids (Int[Array, "batch seq_len"], optional): Position indices for each token.
                Shape: (batch_size, sequence_length). If None, uses sequential positions.
            mode (RUNTIME_MODE_TYPES, optional): Execution mode (MODE_TRAIN, MODE_DECODE, MODE_PREFILL).
                Auto-detected if None based on sequence length and cache.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache, optional):
                Cached key-value states for efficient autoregressive generation. HybridCache
                is recommended for Gemma2's mixed attention patterns.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata, optional):
                Metadata for paged attention cache management.
            apply_lm_head (bool): Whether to apply the language modeling head to produce logits.
                Set to False to only get hidden states. Default: True.
            output_attentions (bool, optional): Whether to return attention weights from all layers.
                Default: None (uses config value).
            output_hidden_states (bool, optional): Whether to return hidden states from all layers.
                Default: None (uses config value).

        Returns:
            CausalLMOutput: Model outputs containing:
                - logits (jnp.ndarray, optional): Next-token prediction logits if apply_lm_head=True.
                  Shape: (batch_size, sequence_length, vocab_size). Soft-capping is applied if
                  final_logit_softcapping is configured (logits = tanh(logits / cap) * cap).
                - hidden_states (tuple[jnp.ndarray], optional): Hidden states from each layer
                  if output_hidden_states=True.
                - attentions (tuple[jnp.ndarray], optional): Attention weights from each layer
                  if output_attentions=True.
                - past_key_values (TransformerCache | RaggedPagesCache | HybridCache): Updated
                  cache with new key-value states.

        Note:
            Gemma2 applies final_logit_softcapping (default: 30.0) via tanh to prevent extreme
            logit values and improve training stability. The formula is:
            logits = tanh(logits / final_logit_softcapping) * final_logit_softcapping
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
        """
        return self.lm_head

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.model.get_embedding()


@register_module(TaskType.SEQUENCE_CLASSIFICATION, config=Gemma2Config, model_type="gemma2")
class Gemma2ForSequenceClassification(BaseSequenceClassificationModule[Gemma2Model, Gemma2Config]):
    """Gemma2 text encoder with a classification head for sequence-level tasks."""

    _task_type = TaskType.SEQUENCE_CLASSIFICATION
    _model_type = "gemma2"
    _config_class = Gemma2Config

    def __init__(
        self,
        config: Gemma2Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(
            config=config,
            base_model_class=Gemma2Model,
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
        This model has a sequence classification head, not an LM Head.
        """
        raise NotImplementedError("This model has a sequence classification head, not a language model head.")

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.model.get_embedding()

    def get_task_head(self):
        """Returns the sequence classification head."""
        return self.score
