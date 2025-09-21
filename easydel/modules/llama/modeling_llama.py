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
from flax import nnx as nn
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
    PagesCache,
    PagesCacheView,
    PagesMetadata,
    TransformerCache,
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.layers.linear import ColumnParallelLinear, RowParallelLinear
from easydel.layers.norms import RMSNorm

from .llama_configuration import LlamaConfig


class LlamaMLP(nn.Module):
    """Multi-Layer Perceptron module for Llama models.

    Implements the feedforward network with SwiGLU activation function
    for enhanced representation learning in Llama architecture.
    """

    def __init__(
        self,
        config: LlamaConfig,
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
        column_parallel_linear = partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=self.config.mlp_bias,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        row_parallel_linear = partial(
            RowParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=self.config.mlp_bias,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        self.gate_proj = column_parallel_linear(config.hidden_size, config.intermediate_size)
        self.down_proj = row_parallel_linear(config.intermediate_size, config.hidden_size)
        self.up_proj = column_parallel_linear(config.hidden_size, config.intermediate_size)
        self.dropout = nn.Dropout(rate=self.config.resid_pdrop, rngs=rngs)
        self.act_fn = ACT2FN[self.config.hidden_act]

    def __call__(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        """Apply SwiGLU feedforward transformation.

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
        hidden_states = self.dropout(hidden_states)
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return checkpoint_name(hidden_states, "mlp_output")


class LlamaAttention(AttentionModule):
    """Multi-head attention layer with RoPE embeddings for Llama models."""

    def __init__(
        self,
        config: LlamaConfig,
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
        head_dim = config.hidden_size // config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", head_dim)
        self.num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads

        if self.num_key_value_groups == 1:
            assert self.config.num_attention_heads == self.config.num_key_value_heads

        column_parallel_linear = partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.attention_bias,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        row_parallel_linear = partial(
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

        self.rotary = self.config.get_basic_rope(self.dtype, self.head_dim, self.head_dim, True)

        self.attention_performer = FlexibleAttentionModule(
            rngs=rngs,
            base_config=self.config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=self.config.attention_dropout,
        )

        self.resid_dropout = nn.Dropout(rate=config.resid_pdrop, rngs=rngs)

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        attention_mask: Bool[Array, "batch seq_len"],
        position_ids: Int[Array, "batch seq_len"],
        causal_mask: Bool[Array, "batch seq_len seq_len"] | bool | None,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | PagesCacheView | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        segment_ids: Int[Array, "batch seq_len"] | None = None,
        output_attentions: bool = False,
        fcm_mask: Bool[Array, "batch seq_len seq_len"] | None = None,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ) -> AttentionLayerOutput:
        """Apply multi-head attention with RoPE.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            attention_mask: Attention mask for padding
            position_ids: Position indices for RoPE
            causal_mask: Causal attention mask
            mode: Runtime mode (train/eval/infer)
            cache_view: Optional cache view for KV caching
            cache_metadata: Optional cache metadata
            segment_ids: Optional segment IDs
            output_attentions: Whether to return attention weights
            fcm_mask: Optional FCM mask
            frequencies: Optional precomputed RoPE frequencies

        Returns:
            AttentionLayerOutput with attention output and optional weights
        """
        batch_size, sequence_length = hidden_states.shape[:2]
        query_states, key_states, value_states = (
            checkpoint_name(self.q_proj(hidden_states), "attn_query"),
            checkpoint_name(self.k_proj(hidden_states), "attn_key"),
            checkpoint_name(self.v_proj(hidden_states), "attn_value"),
        )
        qshape = (
            batch_size,
            sequence_length,
            self.config.num_attention_heads,
            self.head_dim,
        )
        kv_shape = (
            batch_size,
            sequence_length,
            self.config.num_key_value_heads,
            self.head_dim,
        )
        query_states = query_states.reshape(qshape)
        key_states = key_states.reshape(kv_shape)
        value_states = value_states.reshape(kv_shape)
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
            attention_mask,
            init_attention_bias,
            cache_view,
            cache_metadata,
        ) = self.concatenate(
            query=query_states,
            key=key_states,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            value=value_states,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            fcm_mask=fcm_mask,
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
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            causal=True,
        )
        attn_output = checkpoint_name(
            self.resid_dropout(
                self.o_proj(self.shard_attention_prod(attn_output=self._merge_heads(attentions.attention_outputs)))
            ),
            "attn_output",
        )
        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class LlamaDecoderLayer(nn.Module):
    """Single decoder layer for Llama models.

    Combines multi-head attention and feedforward networks with
    RMS normalization and residual connections.
    """

    def __init__(
        self,
        config: LlamaConfig,
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
        attn_block = LlamaAttention
        mlp_block = LlamaMLP
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
        attention_mask: Bool[Array, "batch seq_len"],
        position_ids: Int[Array, "batch seq_len"],
        causal_mask: Bool[Array, "batch seq_len seq_len"] | bool | None,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | PagesCacheView | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        segment_ids: Int[Array, "batch seq_len"] | None = None,
        output_attentions: bool = False,
        fcm_mask: Bool[Array, "batch seq_len seq_len"] | None = None,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ) -> DecoderLayerOutput:
        attn_outputs = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask,
            position_ids,
            causal_mask,
            mode,
            cache_view,
            cache_metadata,
            segment_ids,
            output_attentions,
            fcm_mask,
            frequencies,
        )
        hidden_states = checkpoint_name(hidden_states + attn_outputs.attention_output, "residual")

        feed_forward_input = self.post_attention_layernorm(hidden_states)

        if self.config.use_scan_mlp:
            feed_forward_hidden_states = block_wise_ffn(
                self.mlp,
                feed_forward_input,
                self.config.scan_mlp_chunk_size,
            )
        else:
            feed_forward_hidden_states = self.mlp(feed_forward_input)

        hidden_states = checkpoint_name(hidden_states + feed_forward_hidden_states, "residual")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        hidden_states = checkpoint_name(hidden_states, "layer_output")
        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
            cache_view=attn_outputs.cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=LlamaConfig, model_type="llama")
class LlamaModel(EasyDeLBaseModule):
    """Llama model implementation.

    This implements the Llama language model architecture, utilizing transformer blocks
    with RMSNorm, rotary position embeddings, and a specific attention mechanism.

    Attributes:
            config (LlamaConfig): Configuration for the model.
            dtype (jnp.dtype): Data type for computations.
            param_dtype (jnp.dtype): Data type for parameters.
            precision: Precision setting for JAX operations.
    """

    def __init__(
        self,
        config: LlamaConfig,
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
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            rngs=rngs,
        )
        self.dropout = nn.Dropout(rate=self.config.embd_pdrop, rngs=rngs)
        self.layers = [
            LlamaDecoderLayer(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for _ in range(self.config.num_hidden_layers)
        ]
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
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
        segment_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        """Forward pass through the Llama model.

        Args:
            input_ids (chex.Array, optional): Input token IDs, shape (batch_size, sequence_length).
            inputs_embeds (chex.Array, optional): Input embeddings, shape (batch_size, sequence_length, hidden_size).
            attention_mask (chex.Array, optional): Mask to avoid attention on padding tokens.
            position_ids (chex.Array, optional): Indices of positions of each input sequence token.
            segment_ids (chex.Array, optional): Segment token indices for segment embeddings.
            past_key_values (TransformerCache | PagesCache, optional): Cache containing
                precomputed key/value states.
            cache_metadata (TransformerMetadata | PagesMetadata, optional): Metadata for cache handling.
            output_attentions (bool, optional): Whether to return attention weights.
            output_hidden_states (bool, optional): Whether to return hidden states of all layers.


        Returns:
            Union[BaseModelOutput, Tuple]: Model outputs (last hidden state, optional hidden states, optional attentions)
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = checkpoint_name(self.embed_tokens(input_ids.astype("i4")), "embeddings")
        batch_size, sequence_length, _ = inputs_embeds.shape

        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        assert sequence_length <= self.config.max_position_embeddings, (
            f"Maximum Position Embedding Reached ! "
            f"(Excepted <= {self.config.max_position_embeddings} got {sequence_length})"
        )
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length), "b1")
        else:
            if attention_mask.dtype != jnp.bool:
                attention_mask = jnp.astype(attention_mask == 1, "b1")
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                (batch_size, sequence_length),
            ).astype(jnp.int32)

        hidden_states = self.dropout(inputs_embeds)
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
                attention_mask=attention_mask,
                position_ids=position_ids,
                mode=mode,
                cache_view=past_key_values.views[idx],
                cache_metadata=cache_metadata,
                causal_mask=self.causal_mask,
                output_attentions=output_attentions,
                segment_ids=segment_ids,
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


@register_module(TaskType.CAUSAL_LM, config=LlamaConfig, model_type="llama")
class LlamaForCausalLM(EasyDeLBaseModule):
    """Llama model with a language modeling head for causal language modeling tasks.

    This model is a transformer-based language model with causal attention masks
    applied to perform autoregressive language generation.

    Attributes:
            config (LlamaConfig): Configuration for the model.
            dtype (jnp.dtype): Data type for computations (default is jnp.float32).
            param_dtype (jnp.dtype): Data type for parameters (default is jnp.float32).
            precision (tp.Optional[tp.Union[str, jax.lax.Precision]]): Precision setting for JAX operations.
    """

    def __init__(
        self,
        config: LlamaConfig,
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
        self.model = LlamaModel(
            config=config,
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
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            precision=self.precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        segment_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> CausalLMOutput:
        """Forward pass through the Llama model for causal language modeling.

        Args:
                input_ids (chex.Array, optional): Input token IDs, shape (batch_size, sequence_length).
                inputs_embeds (chex.Array, optional): Input embeddings, shape (batch_size, sequence_length, hidden_size).
                attention_mask (chex.Array, optional): Mask to avoid attention on padding tokens.
                position_ids (chex.Array, optional): Indices of positions of each input sequence token.
                segment_ids (chex.Array, optional): Segment token indices for segment embeddings.
                past_key_values (TransformerCache | PagesCache, optional): Cache containing
                    precomputed key/value states.
                cache_metadata (TransformerMetadata | PagesMetadata, optional): Metadata for cache handling.
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
            segment_ids=segment_ids,
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


@register_module(TaskType.SEQUENCE_CLASSIFICATION, config=LlamaConfig, model_type="llama")
class LlamaForSequenceClassification(EasyDeLBaseModule):
    """Llama model for sequence classification tasks.

    This class extends the base Llama model by adding a linear classification head
    to perform sequence classification tasks such as sentiment analysis or text classification.

    Attributes:
            config (LlamaConfig): Configuration for the model.
            dtype (jnp.dtype): Data type for computations.
            param_dtype (jnp.dtype): Data type for parameters.
            precision: Precision setting for JAX operations.
    """

    def __init__(
        self,
        config: LlamaConfig,
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
        self.model = LlamaModel(
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
        score_block = ColumnParallelLinear
        score_block = auto_remat(
            score_block,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.score = score_block(
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
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        segment_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> SequenceClassifierOutput:
        """Forward pass through the Llama model for sequence classification.

        This method processes input sequences through the Llama model and applies
        a classification head to the output.

        Args:
            input_ids (chex.Array, optional): Input token IDs, shape (batch_size, sequence_length).
            inputs_embeds (chex.Array, optional): Input embeddings, shape (batch_size, sequence_length, hidden_size).
            attention_mask (chex.Array, optional): Mask to avoid attention on padding tokens.
            position_ids (chex.Array, optional): Indices of positions of each input sequence token.
            segment_ids (chex.Array, optional): Segment token indices for segment embeddings.
            past_key_values (TransformerCache | PagesCache, optional): Cache containing
                precomputed key/value states.
            cache_metadata (TransformerMetadata | PagesMetadata, optional): Metadata for cache handling.
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
            segment_ids=segment_ids,
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
