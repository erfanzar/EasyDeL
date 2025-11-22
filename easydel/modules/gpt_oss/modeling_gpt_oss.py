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


import jax
import jax.numpy as jnp
from eformer import common_types
from eformer.escale import apply_logical_sharding
from ejkernel.types import MaskInfo
from flax import nnx as nn
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.loss_utils import auxiliary_load_balancing_loss_func
from easydel.infra.modeling_outputs import (
    AttentionLayerOutput,
    DecoderLayerOutput,
    MoeCausalLMOutput,
    MoeModelOutput,
    SequenceClassifierOutput,
)
from easydel.infra.utils import ArrayParam, auto_remat
from easydel.layers.attention_unified import UnifiedAttention
from easydel.layers.base_modules import BaseCausalLMModule
from easydel.layers.caching import (
    RaggedPagesCache,
    RaggedPagesCacheView,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.layers.linear import ColumnParallelLinear
from easydel.layers.norms import RMSNorm

from .gpt_oss_configuration import GptOssConfig


class GptOssExperts(nn.Module):
    """Grouped expert feed-forward network used inside GPT-OSS MoE layers."""

    def __init__(
        self,
        config: GptOssConfig,
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

        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size

        initializer = jax.nn.initializers.normal(stddev=self.config.initializer_range)

        NEP = self.num_experts
        HD = self.hidden_size
        ED = self.expert_dim

        self.gate_up_proj = ArrayParam.bound(
            shape=(NEP, HD, 2 * ED),
            dtype=param_dtype,
            init_fn=initializer,
            key=rngs.param(),
        )
        self.gate_up_proj_bias = ArrayParam.bound(
            shape=(NEP, 2 * ED),
            dtype=param_dtype,
            init_fn=initializer,
            key=rngs.param(),
        )
        self.down_proj = ArrayParam.bound(
            shape=(NEP, ED, HD),
            dtype=param_dtype,
            init_fn=initializer,
            key=rngs.param(),
        )
        self.down_proj_bias = ArrayParam.bound(
            shape=(NEP, HD),
            dtype=param_dtype,
            init_fn=initializer,
            key=rngs.param(),
        )

        self.alpha = 1.702
        self.limit = 7.0

    def __call__(self, hidden_states: jax.Array, router_indices=None, routing_weights=None, training=False) -> jax.Array:
        del training
        batch_size, seq_len = hidden_states.shape[:2]
        tokens = batch_size * seq_len

        hidden_states = hidden_states.reshape(tokens, self.hidden_size)
        routing_weights = routing_weights.reshape(tokens, -1)
        router_indices = router_indices.reshape(tokens, -1)

        gate_up_proj = jnp.take(self.gate_up_proj.value, router_indices, axis=0)
        gate_up = jnp.einsum("td,txdf->txf", hidden_states, gate_up_proj, precision=self.precision)
        gate_up += jnp.take(self.gate_up_proj_bias.value, router_indices, axis=0)

        gate = jnp.clip(gate_up[..., ::2], a_max=self.limit)
        up = jnp.clip(gate_up[..., 1::2], a_min=-self.limit, a_max=self.limit)
        glu = gate * jax.nn.sigmoid(gate * self.alpha)
        fused = (up + 1) * glu

        down_proj = jnp.take(self.down_proj.value, router_indices, axis=0)
        next_states = jnp.einsum("txf,txfd->txd", fused, down_proj, precision=self.precision)
        next_states += jnp.take(self.down_proj_bias.value, router_indices, axis=0)

        topk_weights = jnp.take_along_axis(routing_weights, router_indices, axis=1)
        next_states = jnp.einsum("txd,tx->td", next_states, topk_weights, precision=self.precision)

        return next_states.reshape(batch_size, seq_len, self.hidden_size)


class GptOssTopKRouter(nn.Module):
    """Top-k router that scores tokens and selects experts per token."""

    def __init__(
        self,
        config: GptOssConfig,
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
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        initializer = jax.nn.initializers.normal(stddev=self.config.initializer_range)
        self.kernel = ArrayParam.bound(
            shape=(self.num_experts, self.hidden_dim),
            dtype=param_dtype,
            init_fn=initializer,
            key=rngs.param(),
        )
        self.bias = ArrayParam.bound(
            shape=(self.num_experts,),
            dtype=param_dtype,
            init_fn=initializer,
            key=rngs.param(),
        )

    def __call__(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = hidden_states @ self.kernel.value + self.bias.value
        router_top_value, router_indices = jax.lax.top_k(router_logits, self.top_k)
        router_top_value = jax.nn.softmax(router_top_value, axis=1)
        router_scores = jnp.zeros_like(router_logits)
        router_scores = router_scores.at[jnp.arange(router_scores.shape[0])[:, None], router_indices].set(
            router_top_value
        )

        return router_scores, router_indices


class GptOssMLP(nn.Module):
    """Mixture-of-experts MLP combining the router and shared experts."""

    def __init__(
        self,
        config: GptOssConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        self.router = GptOssTopKRouter(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.experts = GptOssExperts(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, hidden_states, training=False):
        router_scores, router_indices = self.router(hidden_states)
        routed_out = self.experts(
            hidden_states,
            router_indices=router_indices,
            routing_weights=router_scores,
            training=training,
        )
        return routed_out, router_scores


class GptOssAttention(UnifiedAttention):
    """GPT-OSS Attention with sink tokens support.

    Inherits from UnifiedAttention.
    Supports layer-specific sliding windows and sink tokens for improved attention.
    """

    def __init__(
        self,
        config: GptOssConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        """Initialize GPT-OSS attention with sink tokens."""
        self.layer_idx = layer_idx

        super().__init__(
            config,
            dtype,
            param_dtype,
            precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="standard",
            causal=True,
        )

        # Set sliding window attribute for layer-specific attention
        is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        if is_sliding:
            self.sliding_window = config.sliding_window

        # Sink tokens for improved attention
        self.sinks = ArrayParam.bound(
            shape=(config.num_attention_heads,),
            dtype=param_dtype,
            init_fn=jax.nn.initializers.normal(config.initializer_range),
            key=rngs.param(),
        )

    def forward_standard(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ):
        """Override to add sink tokens support."""
        batch_size, sequence_length = hidden_states.shape[:2]

        # 1. Project Q/K/V
        query_states = checkpoint_name(self.query_projection(hidden_states), "attn_query")
        key_states = checkpoint_name(self.key_projection(hidden_states), "attn_key")
        value_states = checkpoint_name(self.value_projection(hidden_states), "attn_value")

        # 2. Reshape to multi-head format
        query_states = query_states.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        key_states = key_states.reshape(batch_size, sequence_length, self.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(batch_size, sequence_length, self.num_key_value_heads, self.head_dim)

        # 3. POST-PROCESSING HOOK: Apply Q/K norm or other transformations
        query_states, key_states, value_states = self._postprocess_qkv(query_states, key_states, value_states)

        # 4. Apply sharding
        query_states, key_states, value_states = self.apply_qkv_shardings(query_states, key_states, value_states)

        # 5. Apply RoPE
        query_states, key_states = self._apply_rotary(query_states, key_states, position_ids, frequencies)

        # 6. KV cache concatenation
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

        # 7. Compute attention with sink tokens
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
            sliding_window=self.sliding_window,
            softmax_aux=self.sinks.value,  # Sink tokens
        )

        # 8. Merge heads and output projection
        attn_output = self.shard_attention_prod(self._merge_heads(attentions.attention_outputs))
        attn_output = checkpoint_name(self.output_projection(attn_output), name="attn_output")

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class GptOssDecoderLayer(nn.Module):
    """Single GPT-OSS decoder block with attention and expert MLP."""

    def __init__(
        self,
        config: GptOssConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        attn_block = GptOssAttention
        mlp_block = GptOssMLP
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
        self.attention_type = config.layer_types[layer_idx]

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | None = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ):
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
        hidden_states = hidden_states + attn_outputs.attention_output

        feed_forward_input = self.post_attention_layernorm(hidden_states)

        feed_forward_hidden_states, router_logits = self.mlp(feed_forward_input)
        hidden_states = hidden_states + feed_forward_hidden_states
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
            cache_view=attn_outputs.cache_view,
            router_logits=router_logits,
        )


@register_module(TaskType.BASE_MODULE, config=GptOssConfig, model_type="gpt_oss")
class GptOssModel(EasyDeLBaseModule):
    """The base GptOss model transformer.

    This class represents the core transformer architecture of the GptOss model,
    consisting of an embedding layer, multiple GptOssDecoderLayer layers (with sparse MoE),
    and a final layer normalization.

    Attributes:
        config (GptOssConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
        embed_tokens (nn.Embed): Embedding layer for input tokens.
        layers (tp.List[GptOssDecoderLayer]): List of decoder layers.
        norm (RMSNorm): Final layer normalization.
        gradient_checkpointing (EasyDeLGradientCheckPointers): Gradient checkpointing configuration.
    """

    def __init__(
        self,
        config: GptOssConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the GptOssModel.

        Args:
            config (GptOssConfig): The configuration object for the GptOss model.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
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
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.layers = [
            GptOssDecoderLayer(
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
            dim=config.hidden_size,
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
        past_key_values: TransformerCache | RaggedPagesCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | None = None,
    ) -> MoeModelOutput:
        """Forward pass of the GptOssModel.

        Args:
            input_ids (tp.Optional[chex.Array]): Input token IDs. Shape: (batch_size, sequence_length).
            inputs_embeds (tp.Optional[chex.Array]): Input embeddings.
                Either `input_ids` or `inputs_embeds` must be provided.
            attention_mask (tp.Optional[chex.Array]): Mask to avoid performing attention on padding token indices.
                Shape: (batch_size, sequence_length).
            position_ids (tp.Optional[chex.Array]): Position indices for the tokens.
                Shape: (batch_size, sequence_length).
            segment_ids (tp.Optional[chex.Array]): Segment IDs (unused).
            output_attentions (tp.Optional[bool]): Whether to return attention weights.
                Defaults to `config.output_attentions`.
            output_hidden_states (tp.Optional[bool]): Whether to return hidden states for all layers.
                Defaults to `config.output_hidden_states`.
            output_router_logits (tp.Optional[bool]): Whether to return router logits from the MoE layers.
                Defaults to `config.output_router_logits`.
            past_key_values (tp.Optional[TransformerCache | RaggedPagesCache]):
                Precomputed key/value states for attention.
            cache_metadata (tp.Optional[TransformerMetadata | RaggedPagesMetadata]): Metadata for paged attention.


        Returns:
            MoeModelOutput: The model's output.
                returns a `MoeModelOutput` object containing `last_hidden_state`, `hidden_states` (optional),
                `attentions` (optional), and `router_logits` (optional).

        Raises:
            ValueError: If neither `input_ids` nor `inputs_embeds` is provided.
        """
        if output_router_logits is None:
            output_router_logits = self.config.output_router_logits

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.astype("i4"))

        sequence_length = inputs_embeds.shape[1]

        assert sequence_length <= self.config.max_position_embeddings, (
            f"Maximum Position Embedding Reached ! "
            f"(Excepted <= {self.config.max_position_embeddings} got {sequence_length})"
        )

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
                all_self_attns += (layer_outputs.attention_weight,)

            if output_router_logits:
                all_router_logits += (layer_outputs.router_logits,)

            past_key_values[idx] = layer_outputs.cache_view

        hidden_states = self.norm(hidden_states)

        return MoeModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
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


@register_module(TaskType.CAUSAL_LM, config=GptOssConfig, model_type="gpt_oss")
class GptOssForCausalLM(BaseCausalLMModule[GptOssModel, GptOssConfig]):
    """GPT-OSS model with a Causal Language Modeling head.

    This model consists of the base GPT-OSS transformer (GptOssModel) followed by a
    language modeling head for next token prediction. Supports MoE with auxiliary loss.

    Type Parameters:
        GptOssModel: The base model type
        GptOssConfig: The configuration type
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "gpt_oss"
    _config_class = GptOssConfig

    def __init__(
        self,
        config: GptOssConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize the GPT-OSS Causal LM module.

        Args:
            config: Model configuration
            dtype: Data type for computations
            param_dtype: Data type for parameters
            precision: Precision setting for JAX operations
            rngs: Random number generators
        """
        super().__init__(
            config=config,
            base_model_class=GptOssModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
            router_aux_loss_coef=config.router_aux_loss_coef,
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
        past_key_values: TransformerCache | RaggedPagesCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> MoeCausalLMOutput:
        """Forward pass for GPT-OSS MoE model.

        Args:
            input_ids: Input token IDs
            inputs_embeds: Input embeddings (alternative to input_ids)
            attention_mask: Mask to avoid attention on padding tokens
            position_ids: Position indices for tokens
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            output_router_logits: Whether to return router logits
            mode: Runtime mode (train, eval, etc.)
            past_key_values: Cache containing precomputed key/value states
            cache_metadata: Metadata for cache handling
            apply_lm_head: Whether to apply the LM head

        Returns:
            MoeCausalLMOutput with logits, aux_loss, router_logits, etc.
        """

        def _aux_loss_fn(outputs, attention_mask):
            """Custom auxiliary loss for GPT-OSS."""
            if outputs.router_logits is None:
                return None
            return auxiliary_load_balancing_loss_func(
                gate_logits=outputs.router_logits,
                num_experts=self.config.num_local_experts,
                top_k=self.config.num_experts_per_tok,
                attention_mask=attention_mask,
            )

        return self.forward_moe(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            apply_lm_head=apply_lm_head,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            aux_loss_fn=_aux_loss_fn,
        )


@register_module(TaskType.SEQUENCE_CLASSIFICATION, config=GptOssConfig, model_type="gpt_oss")
class GptOssForSequenceClassification(EasyDeLBaseModule):
    """GptOss model with a Sequence Classification head.

    This model consists of the base GptOss transformer (`GptOssModel`) followed by a
    linear layer (`score`) that projects the transformer's output hidden states
    (typically the hidden state of the first token) to the number of classes for classification.
    It also handles the calculation of the auxiliary loss from the MoE layers.

    Attributes:
        config (GptOssConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
        model (GptOssModel): The core GptOss transformer model.
        score (ParallelLinear): The linear layer for classification.
        num_experts (int): Total number of experts.
        num_experts_per_tok (int): Number of experts to route per token.
    """

    def __init__(
        self,
        config: GptOssConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the GptOssForSequenceClassification model.

        Args:
            config (GptOssConfig): The configuration object for the GptOss model.
                Must include `num_labels`.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            rngs (nn.Rngs): Random number generators.

        Raises:
            AssertionError: If `config.num_labels` is not defined.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.model = GptOssModel(
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
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | None = None,
    ) -> SequenceClassifierOutput:
        """Forward pass of the GptOssForSequenceClassification model.

        Args:
            input_ids (tp.Optional[chex.Array]): Input token IDs. Shape: (batch_size, sequence_length).
            inputs_embeds (tp.Optional[chex.Array]): Input embeddings.
                Either `input_ids` or `inputs_embeds` must be provided.
            attention_mask (tp.Optional[chex.Array]): Mask to avoid performing attention on padding token indices.
                Shape: (batch_size, sequence_length).
            position_ids (tp.Optional[chex.Array]): Position indices for the tokens.
                Shape: (batch_size, sequence_length).
            segment_ids (tp.Optional[chex.Array]): Segment IDs (unused).
            output_attentions (tp.Optional[bool]): Whether to return attention weights. Defaults to
                `config.output_attentions`.
            output_hidden_states (tp.Optional[bool]): Whether to return hidden states for all layers.
                Defaults to `config.output_hidden_states`.
            output_router_logits (tp.Optional[bool]): Whether to return router logits from the MoE layers.
                Defaults to `config.output_router_logits`.
            past_key_values (tp.Optional[TransformerCache | RaggedPagesCache]):
                Precomputed key/value states for attention.
            cache_metadata (tp.Optional[TransformerMetadata | RaggedPagesMetadata]): Metadata for paged attention.


        Returns:
           SequenceClassifierOutput: The model's output.
                returns a `SequenceClassifierOutput` object containing `logits`, `aux_loss` (optional),
                `hidden_states` (optional), `attentions` (optional), and `router_logits` (optional).


        Raises:
            ValueError: If `config.pad_token_id` is None and `batch_size > 1`.
        """
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
            output_router_logits=output_router_logits,
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
        aux_loss = None
        if output_router_logits and transformer_outputs.router_logits is not None:
            aux_loss = auxiliary_load_balancing_loss_func(
                gate_logits=transformer_outputs.router_logits,
                num_experts=self.config.num_local_experts,
                top_k=self.config.num_experts_per_tok,
                attention_mask=attention_mask,
            )
            aux_loss += aux_loss * self.config.router_aux_loss_coef

        return SequenceClassifierOutput(
            logits=pooled_logits,
            past_key_values=past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            aux_loss=aux_loss,
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
