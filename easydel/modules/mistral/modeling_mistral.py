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

import chex
import jax
from eformer import common_types
from eformer.escale import apply_logical_sharding
from eformer.loggings import get_logger
from ejkernel.types import MaskInfo
from flax import nnx as nn
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import (
    AttentionLayerOutput,
    BaseModelOutput,
    CausalLMOutput,
    DecoderLayerOutput,
    SequenceClassifierOutput,
)
from easydel.infra.utils import ACT2FN, auto_remat, block_wise_ffn, get_dot_general_by_bits
from easydel.layers.attention import AttentionModule, FlexibleAttentionModule
from easydel.layers.caching import (
    RaggedPagesCache,
    RaggedPagesCacheView,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.layers.linear import ColumnParallelLinear, RowParallelLinear
from easydel.layers.norms import RMSNorm

from .mistral_configuration import MistralConfig

logger = get_logger(__name__)


class MistralMLP(nn.Module):
    """Multi-Layer Perceptron module for Mistral models.

    Implements the feedforward network with SiLU activation function
    for efficient and effective representation learning.
    """

    def __init__(
        self,
        config: MistralConfig,
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
        column_parallel_linear = functools.partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=self.precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        row_parallel_linear = functools.partial(
            RowParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=self.precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
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
        """Apply SiLU feedforward transformation.

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


class MistralAttention(AttentionModule):
    """Multi-head attention layer with RoPE embeddings for Mistral models."""

    def __init__(
        self,
        config: MistralConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize attention layer with config."""
        super().__init__(config=config)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.hidden_size = config.hidden_size
        self.head_dim = self.config.head_dim

        self.num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads

        if self.num_key_value_groups == 1:
            assert self.config.num_attention_heads == self.config.num_key_value_heads

        column_parallel_linear = functools.partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.attention_bias,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        row_parallel_linear = functools.partial(
            RowParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.attention_bias,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        self.q_proj = column_parallel_linear(config.hidden_size, config.num_attention_heads * self.head_dim, rngs=rngs)
        self.k_proj = column_parallel_linear(config.hidden_size, config.num_key_value_heads * self.head_dim, rngs=rngs)
        self.v_proj = column_parallel_linear(config.hidden_size, config.num_key_value_heads * self.head_dim, rngs=rngs)
        self.o_proj = row_parallel_linear(config.num_attention_heads * self.head_dim, config.hidden_size, rngs=rngs)
        self.sliding_window = config.sliding_window
        self.attention_performer = FlexibleAttentionModule(
            rngs=rngs,
            dropout_prob=config.attention_dropout,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
        )

        self.rotary = self.config.get_basic_rope(self.dtype, self.head_dim)

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ) -> AttentionLayerOutput:
        batch_size, sequence_length = hidden_states.shape[:2]
        query_states, key_states, value_states = (
            checkpoint_name(self.q_proj(hidden_states), "attn_query"),
            checkpoint_name(self.k_proj(hidden_states), "attn_key"),
            checkpoint_name(self.v_proj(hidden_states), "attn_value"),
        )

        query_states = query_states.reshape(batch_size, sequence_length, self.config.num_attention_heads, self.head_dim)
        key_states = key_states.reshape(batch_size, sequence_length, self.config.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(batch_size, sequence_length, self.config.num_key_value_heads, self.head_dim)

        query_states, key_states, value_states = self.apply_qkv_shardings(query_states, key_states, value_states)

        query_states, key_states = self.rotary(
            positions=position_ids,
            query=query_states,
            key=key_states,
            frequencies=frequencies,
        )

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
            value=value_states,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            mask_info=mask_info,
            sliding_window=self.sliding_window,
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
            causal=True,
            sliding_window=self.sliding_window,
        )

        attn_output = self.shard_attention_prod(self._merge_heads(attentions.attention_outputs))
        attn_output = checkpoint_name(self.o_proj(attn_output), "attn_output")

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class MistralDecoderLayer(nn.Module):
    """Single decoder layer for Mistral models.

    Combines sliding window attention with feedforward networks,
    using RMS normalization and residual connections.
    """

    def __init__(
        self,
        config: MistralConfig,
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
        attn_block = MistralAttention
        mlp_block = MistralMLP

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
        )
        self.mlp = mlp_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.input_layernorm = RMSNorm(
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.post_attention_layernorm = RMSNorm(
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ) -> DecoderLayerOutput:
        residual = hidden_states
        attention_output = self.self_attn(
            self.input_layernorm(hidden_states),
            mask_info,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            output_attentions,
            frequencies,
        )

        hidden_states = checkpoint_name(attention_output.attention_output + residual, "residual")
        ffd_inp = self.post_attention_layernorm(hidden_states)
        if self.config.use_scan_mlp:
            feed_forward_hidden_states = block_wise_ffn(self.mlp, ffd_inp, self.config.scan_mlp_chunk_size)
        else:
            feed_forward_hidden_states = self.mlp(ffd_inp)

        hidden_states = checkpoint_name(hidden_states + feed_forward_hidden_states, "residual")

        return DecoderLayerOutput(
            hidden_states=checkpoint_name(hidden_states, "layer_output"),
            attention_weight=attention_output.attention_weight,
            cache_view=attention_output.cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=MistralConfig, model_type="mistral")
class MistralModel(EasyDeLBaseModule):
    """Mistral model implementation.

    This implements the Mistral language model architecture, utilizing transformer blocks
    with RMSNorm, sliding window attention, and rotary position embeddings.

    Attributes:
            config (MistralConfig): Configuration for the model.
            dtype (jnp.dtype): Data type for computations.
            param_dtype (jnp.dtype): Data type for parameters.
            precision: Precision setting for JAX operations.
    """

    def __init__(
        self,
        config: MistralConfig,
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
            MistralDecoderLayer(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for i in range(self.config.num_hidden_layers)
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
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        """Forward pass through the Mistral model.

        Args:
            input_ids (chex.Array, optional): Input token IDs, shape (batch_size, sequence_length).
            inputs_embeds (chex.Array, optional): Input embeddings, shape (batch_size, sequence_length, hidden_size).
            attention_mask (chex.Array, optional): Mask to avoid attention on padding tokens.
            position_ids (chex.Array, optional): Indices of positions of each input sequence token.
            past_key_values (TransformerCache | RaggedPagesCache, optional):
                Cache containing precomputed key/value states.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata, optional): Metadata for cache handling.
            output_attentions (bool, optional): Whether to return attention weights.
            output_hidden_states (bool, optional): Whether to return hidden states of all layers.


        Returns:
            Union[BaseModelOutput, Tuple]: Model outputs
                (last hidden state, optional hidden states, optional attentions)
        """
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = checkpoint_name(self.embed_tokens(input_ids.astype("i4")), "embeddings")
        batch_size, sequence_length, _ = inputs_embeds.shape

        assert sequence_length <= self.config.max_position_embeddings, (
            f"Maximum Position Embedding Reached ! "
            f"(Excepted <= {self.config.max_position_embeddings} got {sequence_length})"
        )

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length), "b1")
        else:
            if attention_mask.dtype != jnp.bool:
                attention_mask = jnp.astype(attention_mask == 1, "b1")
        if attention_mask.ndim == 2:
            mask_info = MaskInfo.from_segments(attention_mask)
        else:
            mask_info = MaskInfo.from_attention_mask(attention_mask)

        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                (batch_size, sequence_length),
            ).astype(jnp.int32)

        if attention_mask.ndim == 2:
            mask_info = MaskInfo.from_segments(attention_mask)
        else:
            mask_info = MaskInfo.from_attention_mask(attention_mask)

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


@register_module(TaskType.CAUSAL_LM, config=MistralConfig, model_type="mistral")
class MistralForCausalLM(EasyDeLBaseModule):
    """Mistral model with a language modeling head for causal language modeling tasks.

    This model is a transformer-based language model with sliding window attention
    applied to perform autoregressive language generation.

    Attributes:
            config (MistralConfig): Configuration for the model.
            dtype (jnp.dtype): Data type for computations.
            param_dtype (jnp.dtype): Data type for parameters.
            precision: Precision setting for JAX operations.
    """

    def __init__(
        self,
        config: MistralConfig,
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
        self.model = MistralModel(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        lm_head_block = ColumnParallelLinear
        lm_head_block = auto_remat(
            lm_head_block,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.lm_head = lm_head_block(
            config.hidden_size,
            config.vocab_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            precision=precision,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> CausalLMOutput:
        """Forward pass through the Mistral model for causal language modeling.

        Args:
                input_ids (chex.Array, optional): Input token IDs, shape (batch_size, sequence_length).
                inputs_embeds (chex.Array, optional): Input embeddings, shape (batch_size, sequence_length, hidden_size).
                attention_mask (chex.Array, optional): Mask to avoid attention on padding tokens.
                position_ids (chex.Array, optional): Indices of positions of each input sequence token.
                past_key_values (TransformerCache | RaggedPagesCache, optional):
                    Cache containing precomputed key/value states.
                cache_metadata (TransformerMetadata | RaggedPagesMetadata, optional): Metadata for cache handling.
                output_attentions (bool, optional): Whether to return attention weights.
                output_hidden_states (bool, optional): Whether to return hidden states of all layers.


        Returns:
                Union[CausalLMOutput, Tuple]: Model outputs (logits, optional hidden states, optional attentions)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
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


@register_module(TaskType.SEQUENCE_CLASSIFICATION, config=MistralConfig, model_type="mistral")
class MistralForSequenceClassification(EasyDeLBaseModule):
    """Mistral model for sequence classification tasks.

    This class extends the base Mistral model by adding a linear classification head
    to perform sequence classification tasks such as sentiment analysis or text classification.

    Attributes:
            config (MistralConfig): Configuration for the model.
            dtype (jnp.dtype): Data type for computations.
            param_dtype (jnp.dtype): Data type for parameters.
            precision: Precision setting for JAX operations.
    """

    def __init__(
        self,
        config: MistralConfig,
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
        self.model = MistralModel(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        assert hasattr(config, "num_labels"), (
            "in order to use `SequenceClassification` Models in `EasyDeL` "
            "you first need to attach `num_labels` to model `config`"
        )
        self.score = ColumnParallelLinear(
            self.config.hidden_size,
            config.num_labels,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            precision=self.precision,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> SequenceClassifierOutput:
        """Forward pass through the Mistral model for sequence classification.

        This method processes input sequences through the Mistral model and applies
        a classification head to the output.

        Args:
            input_ids (chex.Array, optional): Input token IDs, shape (batch_size, sequence_length).
            inputs_embeds (chex.Array, optional): Input embeddings, shape (batch_size, sequence_length, hidden_size).
            attention_mask (chex.Array, optional): Mask to avoid attention on padding tokens.
            position_ids (chex.Array, optional): Indices of positions of each input sequence token.
            past_key_values (TransformerCache | RaggedPagesCache, optional):
                Cache containing precomputed key/value states.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata, optional): Metadata for cache handling.
            output_attentions (bool, optional): Whether to return attention weights.
            output_hidden_states (bool, optional): Whether to return hidden states of all layers.


        Returns:
            Union[SequenceClassifierOutput, Tuple]: Classification outputs including logits and optional model outputs
        """
        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
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
        return self.model.get_decoder()

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
