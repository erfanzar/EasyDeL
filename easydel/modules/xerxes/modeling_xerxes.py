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
from easydel.infra.modeling_outputs import BaseModelOutput, CausalLMOutput, DecoderLayerOutput
from easydel.infra.utils import auto_remat, block_wise_ffn
from easydel.layers.attention import FlexibleAttentionModule
from easydel.layers.attention_unified import UnifiedAttention
from easydel.layers.base_modules import BaseCausalLMModule
from easydel.layers.caching import (
    HybridCache,
    RaggedPagesCache,
    RaggedPagesCacheView,
    TransformerCache,
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.layers.linear import ColumnParallelLinear, RowParallelLinear
from easydel.layers.norms import RMSNorm

from .xerxes_configuration import XerxesConfig as XerxesConfig

logger = get_logger(__name__)


class Identity(nn.Module):
    """No-op module used as a placeholder when optional layers are disabled."""

    def __init__(self): ...
    def __call__(self, x):
        return x


class PostCross(nn.Module):
    """Applies a bounded tanh transform after cross attention."""

    def __init__(self): ...
    def __call__(self, x):
        return jax.nn.tanh(x / 30.0) * 30.0


class XerxesMLP(nn.Module):
    """Feed-forward network for Xerxes decoder blocks."""

    def __init__(
        self,
        config: XerxesConfig,
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
        self.rngs = rngs
        kernel_init = jax.nn.initializers.normal(config.initializer_range)

        self.act = nn.swish if config.swish_run else functools.partial(nn.gelu, approximate=True)
        column_parallel_linear = functools.partial(
            ColumnParallelLinear,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        row_parallel_linear = functools.partial(
            RowParallelLinear,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.gate_proj = column_parallel_linear(
            self.config.hidden_size,
            self.config.intermediate_size,
            rngs=rngs,
        )
        self.up_proj = column_parallel_linear(
            self.config.hidden_size,
            self.config.intermediate_size,
            rngs=rngs,
        )
        self.down_proj = row_parallel_linear(
            self.config.intermediate_size,
            self.config.hidden_size,
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
        return hidden_states


class XerxesAttention(UnifiedAttention):
    """Xerxes Attention with conditional Q/K normalization.

    Inherits Q/K normalization from QKNormAttention.
    Features:
    - Conditional Q/K normalization via xe_kvnorm flag
    - Layer-specific sliding window (different patterns based on layer_idx or window_pattern)
    """

    def __init__(
        self,
        config: XerxesConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        causal: bool = True,
        is_cross_attention: bool = False,
        *,
        rngs: nn.Rngs,
    ):
        # Set sliding window BEFORE super().__init__()
        self.is_local_attn = False
        sliding_window = None
        if not config.xe_kvnorm:
            sliding_window = 4096 if bool((layer_idx % 2) == 0) else None
        if config.window_pattern is not None:
            self.is_local_attn = bool((layer_idx + 1) % config.window_pattern)
            sliding_window = config.sliding_window if self.is_local_attn else None

        self.xe_kvnorm = config.xe_kvnorm

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

        self.layer_idx = layer_idx
        self.is_cross_attention = is_cross_attention
        self.causal = causal
        self.attention_softmax_in_fp32 = self.dtype is not jnp.float32

    def _create_q_norm(self, config, dtype, param_dtype, rngs):
        """Override to conditionally create Q norm based on xe_kvnorm flag."""
        if not self.xe_kvnorm:
            return None
        return RMSNorm(
            dim=self.head_dim,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def _create_k_norm(self, config, dtype, param_dtype, rngs):
        """Override to conditionally create K norm based on xe_kvnorm flag."""
        if not self.xe_kvnorm:
            return None
        return RMSNorm(
            dim=self.head_dim,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def _create_attention_performer(self, config, rngs):
        """Override to set dropout_prob to 0.0 for Xerxes."""
        return FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=0.0,
        )

    def _postprocess_qkv(self, query_states, key_states, value_states):
        if not self.xe_kvnorm:
            return query_states, key_states, value_states
        return self.query_normalization(query_states), self.key_normalization(key_states), value_states


class XerxesSparseMoeBlock(nn.Module):
    """Sparse mixture-of-experts feed-forward block used in selected layers."""

    def __init__(
        self,
        config: XerxesConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: None | jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        assert config.swish_run is False

        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.gate = ColumnParallelLinear(
            self.config.hidden_size,
            self.config.num_local_experts,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )
        self.experts = [
            XerxesMLP(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for _ in range(self.config.num_local_experts)
        ]

    def __call__(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> tuple[Array, Array]:
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        router_logits = self.gate(hidden_states).astype(jnp.promote_types(self.dtype, jnp.float32))
        routing_weights, selected_experts = jax.lax.top_k(router_logits, k=self.config.num_experts_per_tok)
        routing_weights = jax.nn.softmax(routing_weights.astype(jnp.promote_types(self.dtype, jnp.float32)), axis=-1)

        final_hidden_state = jnp.zeros_like(hidden_states)
        for index in range(self.config.num_local_experts):
            expert_layer_output = (
                block_wise_ffn(
                    self.experts[index],
                    hidden_states,
                    self.config.scan_mlp_chunk_size,
                )
                if self.config.use_scan_mlp
                else self.experts[index](hidden_states)
            )
            expert_layer_output_exp = (
                jnp.sum(jnp.multiply(selected_experts == index, routing_weights), axis=-1)[:, :, None]
                * expert_layer_output
            )
            final_hidden_state += expert_layer_output_exp
        return final_hidden_state, router_logits


class XerxesDecoderLayer(nn.Module):
    """Transformer decoder block with optional cross-attention and MoE."""

    def __init__(
        self,
        config: XerxesConfig,
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
        self.rngs = rngs

        mlp_block = XerxesSparseMoeBlock if self.config.xe_moe else XerxesMLP
        attn_block = XerxesAttention

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
        rms = functools.partial(
            RMSNorm,
            dim=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        identity = config.xe_kvnorm and not config.xe_moe
        if config.xe_mlpnorm:
            identity = False
        self.identity = identity
        self.input_layernorm = rms()
        self.post_attention_layernorm = rms()
        self.pre_feedforward_layernorm = Identity() if identity else rms()
        self.post_feedforward_layernorm = Identity() if identity else rms()

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesCacheView | None = None,
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
            segment_ids (tp.Optional[Array]): Segment IDs for segment-based attention (optional).
            deterministic (bool): If True, disables dropout for deterministic behavior.
            init_cache (bool): If True, initializes cache for caching keys and values.
            output_attentions (bool): If True, outputs attention weights alongside the hidden states.
            fcm_mask (tp.Optional[Array]): fcm mask to be combined with attn mask and causal mask.
        Returns:
            tp.Tuple[Array, Array]: A tuple containing the attention output and the attention weights.
        """

        attn_outputs = self.self_attn(
            self.input_layernorm(hidden_states),
            mask_info,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            output_attentions,
            default_frequencies if self.self_attn.is_local_attn else frequencies,
        )
        if self.identity:
            hidden_states = hidden_states + attn_outputs.attention_output
            residual = hidden_states
            feed_forward_input = self.post_attention_layernorm(hidden_states)

        else:
            normed = self.post_attention_layernorm(attn_outputs.attention_output)
            hidden_states = hidden_states + normed
            residual = hidden_states
            feed_forward_input = self.pre_feedforward_layernorm(hidden_states)

        if self.config.use_scan_mlp and not self.config.xe_moe:
            feed_forward_hidden_states = block_wise_ffn(
                self.mlp,
                feed_forward_input,
                self.config.scan_mlp_chunk_size,
            )
        else:
            feed_forward_hidden_states = self.mlp(feed_forward_input)

        router_logits = None
        if isinstance(feed_forward_hidden_states, tuple):
            feed_forward_hidden_states, router_logits = feed_forward_hidden_states

        hidden_states = self.post_feedforward_layernorm(feed_forward_hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
            router_logits=router_logits,
            cache_view=attn_outputs.cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=XerxesConfig, model_type="xerxes")
class XerxesModel(EasyDeLBaseModule):
    """Xerxes decoder stack wiring embeddings, decoder layers, and final norm."""

    def __init__(
        self,
        config: XerxesConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
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
            XerxesDecoderLayer(
                self.config,
                layer_idx=i,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for i in range(self.config.num_hidden_layers)
        ]
        self.norm = RMSNorm(
            dim=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.embedding_scale = float(1 if config.xe_kvnorm and not config.xe_mlpnorm else config.hidden_size**0.5)

    @functools.cached_property
    def default_frequencies(self):
        from easydel.infra.utils import ModuleCaches
        from easydel.layers.rotary_embedding import get_frequencies

        frequencies = get_frequencies(
            head_size=self.config.head_dim,
            rotary_dim=self.config.head_dim,
            max_position=self.config.granted_freq_max_position_embedding,
            base=10000,
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
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesCacheView | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        """
        Forward pass through the Xerxes module.

        Args:
            input_ids (Array): Input tensor containing token IDs.
            attention_mask (Array): Mask for attention.
            position_ids (Array): Positional indices.
            segment_ids (tp.Optional[Array]): Segment IDs for different input parts.
            inputs_embeds (tp.Optional[Array]): Embedded input tensor.
            output_attentions (tp.Optional[bool]): If True, output attention weights.
            output_hidden_states (tp.Optional[bool]): If True, output hidden states.
            init_cache (bool): If True, initialize cache for decoding.
            deterministic (bool): If True, disable dropout.

        Returns:
            BaseModelOutput | tp.Tuple: Model output, either as a named tuple or a standard tuple.
        """
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.astype("i4"))
        sequence_length = inputs_embeds.shape[1]
        inputs_embeds = inputs_embeds * self.embedding_scale
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

        if mode is None:
            mode = (
                common_types.MODE_DECODE
                if sequence_length == 1 and past_key_values is not None
                else common_types.MODE_TRAIN
            )
        if past_key_values is None:
            past_key_values = TransformerCache.init_empty(len(self.layers))

        hidden_states = apply_logical_sharding(
            inputs_embeds,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        for idx, block in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            outputs = block(
                hidden_states=hidden_states,
                mask_info=mask_info,
                position_ids=position_ids,
                mode=mode,
                cache_view=past_key_values.views[idx],
                cache_metadata=cache_metadata,
                output_attentions=output_attentions,
                frequencies=self.frequencies,
                default_frequencies=self.default_frequencies,
            )
            hidden_states = outputs.hidden_states

            hidden_states = apply_logical_sharding(
                hidden_states,
                dynamic_axes=common_types.HiddenStateSharding,
                partition_manager=self.config.partition_manager,
            )
            if output_attentions:
                all_attentions += (outputs.attention_weight,)

            past_key_values[idx] = outputs.cache_view

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states, *outputs[2:])
        else:
            outputs = (hidden_states, *outputs[1:])

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


@register_module(TaskType.CAUSAL_LM, config=XerxesConfig, model_type="xerxes")
class XerxesForCausalLM(BaseCausalLMModule[XerxesModel, XerxesConfig]):
    """Xerxes language model with LM head for causal generation."""

    _task_type = TaskType.CAUSAL_LM
    _model_type = "xerxes"
    _config_class = XerxesConfig

    def __init__(
        self,
        config: XerxesConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(
            config=config,
            base_model_class=XerxesModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
        )
        identity = config.xe_kvnorm and not config.xe_moe
        self.post_pross = Identity() if identity else PostCross()

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesCacheView | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> CausalLMOutput:
        """
        Forward pass through the Xerxes module.

        Args:
            input_ids (tp.Optional[Array]): Input tensor containing token IDs.
            attention_mask (tp.Optional[Array]): Mask for attention.
            position_ids (tp.Optional[Array]): Positional indices.
            segment_ids (tp.Optional[Array]): Segment IDs for different input parts.
            inputs_embeds (tp.Optional[Array]): Embedded input tensor.
            output_attentions (tp.Optional[bool]): If True, output attention weights.
            output_hidden_states (tp.Optional[bool]): If True, output hidden states.
            init_cache (bool): If True, initialize cache for decoding.
            deterministic (bool): If True, disable dropout.

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
        )
        hidden_states = outputs.last_hidden_state

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        lm_logits = None
        if apply_lm_head:
            lm_logits = self.apply_lm_head(hidden_states)
        return CausalLMOutput(
            logits=self.post_pross(lm_logits),
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
