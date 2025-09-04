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


from functools import cached_property

import chex
import jax
import jax.numpy as jnp
from eformer import common_types
from eformer.escale import apply_logical_sharding
from flax import nnx as nn

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
from easydel.infra.utils import ACT2FN, auto_remat, get_dot_general_by_bits
from easydel.layers.attention import AttentionModule, FlexibleAttentionModule
from easydel.layers.caching import (
    PagesCache,
    PagesCacheView,
    PagesMetadata,
    TransformerCache,
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.layers.linear import ParallelLinear

from .dbrx_configuration import DbrxConfig


class DbrxAttention(AttentionModule):
    def __init__(
        self,
        config: DbrxConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(config=config)
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs

        self.num_attention_heads = self.config.n_heads
        self.num_key_value_heads = self.config.attn_config.kv_n_heads
        config = self.config
        self.hidden_size = config.hidden_size
        self.head_dim = self.config.d_model // self.config.n_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        if self.num_key_value_groups == 1:
            assert self.num_attention_heads == self.config.attn_config.kv_n_heads
        self.Wqkv = ParallelLinear(
            config.hidden_size,
            self.hidden_size + 2 * self.num_key_value_heads * self.head_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=self.precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        self.out_proj = ParallelLinear(
            config.hidden_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=self.precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )

        self.rotary = self.config.get_basic_rope(
            dtype=self.dtype,
            rotary_dim=self.config.hidden_size // self.config.num_attention_heads,
            head_size=self.config.hidden_size // self.config.num_attention_heads,
            is_neox_style=True,
            base=self.config.attn_config.rope_theta,
        )
        self.attention_performer = FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=0.0,
        )
        self.resid_dropout = nn.Dropout(rate=config.resid_pdrop)

    def __call__(
        self,
        hidden_states: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        causal_mask: chex.Array | bool | None,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | PagesCacheView | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool = False,
        fcm_mask: chex.Array | None = None,
        frequencies: chex.Array | None = None,
    ):
        """
        Forward pass of the attention module.

        Args:
            hidden_states (chex.Array): Input hidden states.
            attention_mask (chex.Array): Mask to apply on the attention scores.
            position_ids (chex.Array): Position indices for the tokens.
            causal_mask (chex.Array): Causal mask for ensuring autoregressive behavior.
            segment_ids (tp.Optional[chex.Array]): Segment IDs for segment-based attention (optional).
            deterministic (bool): If True, disables dropout for deterministic behavior.
            init_cache (bool): If True, initializes cache for caching keys and values.
            output_attentions (bool): If True, outputs attention weights alongside the hidden states.
            fcm_mask (tp.Optional[chex.Array]): fcm mask to be combined with attn mask and causal mask.
        Returns:
            tp.Tuple[chex.Array, chex.Array]: A tuple containing the attention output and the attention weights.
        """
        batch_size, sequence_length = hidden_states.shape[:2]
        qkv_states = self.Wqkv(hidden_states)
        if self.config.attn_config.clip_qkv is not None:
            qkv_states = qkv_states.clip(
                min=-self.config.attn_config.clip_qkv,
                max=self.config.attn_config.clip_qkv,
            )

        query_size = self.hidden_size
        key_size = self.num_key_value_heads * self.head_dim

        query_states, key_value_states = jnp.split(qkv_states, [query_size], axis=2)
        key_states, value_states = jnp.split(key_value_states, [key_size], axis=2)
        query_states = query_states.reshape(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.head_dim,
        )
        key_states = key_states.reshape(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        )
        value_states = value_states.reshape(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        )

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
            value=value_states,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
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

        attn_output = self.shard_attention_prod(self._merge_heads(attentions.attention_outputs))
        attn_output = self.out_proj(attn_output)

        attn_output = self.resid_dropout(attn_output)
        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class DbrxNormAttentionNorm(nn.Module):
    """Normalization-Attention-Normalization module for DBRX models.

    Implements a unique architecture pattern with normalization layers
    surrounding the attention mechanism for improved gradient flow.
    """

    kernel_init = staticmethod(nn.initializers.ones)

    def __init__(
        self,
        config: DbrxConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.norm_1 = nn.LayerNorm(
            self.config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            rngs=rngs,
        )
        self.attn = DbrxAttention(  # statics 3,5,6,7
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.norm_2 = nn.LayerNorm(
            self.config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            rngs=rngs,
        )

        self.dropout = nn.Dropout(
            self.config.resid_pdrop,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        causal_mask: chex.Array | bool | None,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | PagesCacheView | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool = False,
        fcm_mask: chex.Array | None = None,
        frequencies: chex.Array | None = None,
    ) -> DecoderLayerOutput:
        """
        Forward pass of the attentionNrom module.

        Args:
            hidden_states (chex.Array): Input hidden states.
            attention_mask (chex.Array): Mask to apply on the attention scores.
            position_ids (chex.Array): Position indices for the tokens.
            causal_mask (chex.Array): Causal mask for ensuring autoregressive behavior.
            segment_ids (tp.Optional[chex.Array]): Segment IDs for segment-based attention (optional).
            deterministic (bool): If True, disables dropout for deterministic behavior.
            init_cache (bool): If True, initializes cache for caching keys and values.
            output_attentions (bool): If True, outputs attention weights alongside the hidden states.
            fcm_mask (tp.Optional[chex.Array]): fcm mask to be combined with attn mask and causal mask.
        Returns:
            DecoderLayerOutput: A tuple containing the residual_states, hidden states, and the attention weights.
        """
        residual_states = hidden_states
        hidden_states = self.norm_1(hidden_states)

        attn_outputs = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            causal_mask=causal_mask,
            segment_ids=segment_ids,
            fcm_mask=fcm_mask,
            frequencies=frequencies,
            mode=mode,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )
        hidden_states = attn_outputs.attention_output
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + residual_states

        residual_states = hidden_states
        hidden_states = self.norm_2(hidden_states)

        return DecoderLayerOutput(
            hidden_states=hidden_states,
            residual_states=residual_states,
            attention_weight=attn_outputs.attention_weight,
            router_logits=None,
            gate_loss=None,
            cache_view=attn_outputs.cache_view,
        )


class DbrxExpertGLU(nn.Module):
    """Gated Linear Unit expert module for DBRX mixture of experts.

    Implements a single expert network with gated activation for
    specialized processing in the MoE architecture.
    """

    def __init__(
        self,
        config: DbrxConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        shape = (
            self.config.ffn_config.moe_num_experts * self.config.ffn_config.ffn_hidden_size,
            self.config.d_model,
        )
        init_fn = nn.initializers.normal(dtype=self.dtype)
        self.w1 = nn.Param(init_fn(rngs.params(), shape, self.param_dtype))
        self.v1 = nn.Param(init_fn(rngs.params(), shape, self.param_dtype))
        self.w2 = nn.Param(init_fn(rngs.params(), shape, self.param_dtype))
        self.activation_fn = ACT2FN[self.config.ffn_config.ffn_act_fn["name"]]

    def __call__(self, x: chex.Array, expert_idx: int) -> chex.Array:
        expert_shape = (
            self.config.ffn_config.moe_num_experts,
            self.config.ffn_config.ffn_hidden_size,
            self.config.d_model,
        )
        expert_w1 = self.w1.value.reshape(expert_shape)[expert_idx]
        expert_v1 = self.v1.value.reshape(expert_shape)[expert_idx]
        expert_w2 = self.w2.value.reshape(expert_shape)[expert_idx]

        x1 = jnp.matmul(
            x,
            jnp.expand_dims(expert_w1.T, 0),
            precision=self.precision,
        )
        x2 = jnp.matmul(
            x,
            jnp.expand_dims(expert_v1.T, 0),
            precision=self.precision,
        )
        x1 = self.activation_fn(x1)
        x1 = x1 * x2
        x1 = jnp.matmul(
            x1,
            jnp.expand_dims(expert_w2, 0),
            precision=self.precision,
        )
        return x1


class DbrxExperts(nn.Module):
    """Collection of expert networks for DBRX mixture of experts.

    Manages multiple expert networks that can be selected and combined
    based on routing decisions for conditional computation.
    """

    def __init__(
        self,
        config: DbrxConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.mlp = DbrxExpertGLU(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        x: chex.Array,
        weights: chex.Array,
        top_weights: chex.Array,
        top_experts: chex.Array,
    ):
        final_hidden_state = jnp.zeros_like(x)
        for index in range(self.config.ffn_config.moe_num_experts):
            output_moe_layer = self.mlp(x, index)
            final_hidden_state += (
                jnp.sum(jnp.multiply(index == top_experts, top_weights), axis=-1)[:, :, None] * output_moe_layer
            )
        return final_hidden_state


class DbrxRouter(nn.Module):
    """Router module for DBRX mixture of experts.

    Determines which experts to activate for each input token,
    implementing sparse routing for efficient computation.
    """

    def __init__(
        self,
        config: DbrxConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.hidden_size = self.config.d_model
        self.moe_num_experts = self.config.ffn_config.moe_num_experts
        self.moe_top_k = self.config.ffn_config.moe_top_k
        self.moe_jitter_eps = self.config.ffn_config.moe_jitter_eps
        self.moe_normalize_expert_weights = self.config.ffn_config.moe_normalize_expert_weights
        self.uniform_expert_assignment = self.config.ffn_config.uniform_expert_assignment

        self.layer = ParallelLinear(
            config.hidden_size,
            self.moe_num_experts,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def jitter(self, x: chex.Array) -> chex.Array:
        if self.moe_jitter_eps is None:
            raise RuntimeError("The router does not have moe_jitter_eps set.")
        low = 1.0 - self.moe_jitter_eps
        high = 1.0 + self.moe_jitter_eps
        noise = jax.random.normal(self.make_rng("params"), x.shape, dtype=x.dtype)
        return low + noise * (high - low)

    def __call__(self, x: chex.Array, deterministic: bool = True) -> tuple[chex.Array, chex.Array, chex.Array]:
        if not deterministic and self.moe_jitter_eps is not None:
            x = x * self.jitter(x)

        weights = self.layer(x.astype(jnp.promote_types(self.dtype, jnp.float32)))
        weights = jax.nn.softmax(weights.astype(jnp.promote_types(self.dtype, jnp.float32)))
        top_weights, top_experts = jax.lax.top_k(weights, self.moe_top_k)

        if self.moe_normalize_expert_weights:
            top_weights = top_weights / jnp.linalg.norm(
                top_weights,
                ord=int(self.moe_normalize_expert_weights),
                axis=-1,
                keepdims=True,
            )

        if self.uniform_expert_assignment:
            top_experts = jax.lax.stop_gradient(
                (
                    jnp.arange(
                        0,
                        jnp.prod(
                            jnp.asarray(top_experts.shape, dtype=jnp.int32),
                            dtype=jnp.int32,
                        ),
                        dtype=top_experts.dtype,
                    )
                    % self.moe_num_experts
                ).reshape(top_experts.shape)
            )

        weights = weights.astype(x.dtype)
        top_weights = top_weights.astype(x.dtype)
        return weights, top_weights, top_experts


class DbrxFFN(nn.Module):
    """Feedforward network with mixture of experts for DBRX models.

    Combines router and expert networks to implement sparse MoE
    feedforward layers with conditional computation.
    """

    def __init__(
        self,
        config: DbrxConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.router = DbrxRouter(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.experts = DbrxExperts(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, x: chex.Array) -> tuple[chex.Array, chex.Array]:
        x = apply_logical_sharding(
            x,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        weights, top_weights, top_experts = self.router(x)
        out = self.experts(x, weights, top_weights, top_experts)
        out = apply_logical_sharding(
            out,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return out, weights


class DbrxBlock(nn.Module):
    """Single transformer block for DBRX models.

    Integrates attention mechanisms with mixture of experts feedforward
    networks, using residual connections and normalization.
    """

    def __init__(
        self,
        config: DbrxConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.hidden_size = self.config.d_model
        self.resid_pdrop = self.config.resid_pdrop
        attn_block = DbrxNormAttentionNorm
        ffn_block = DbrxFFN
        attn_block, ffn_block = auto_remat(
            attn_block,
            ffn_block,
            policy=config.gradient_checkpointing,
        )
        self.norm_attn_norm = attn_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.ffn = ffn_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        causal_mask: chex.Array | bool | None,
        mode: common_types.RUNTIME_MODE_TYPES | None,  # type:ignore
        cache_view: TransformerCacheView | PagesCacheView | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        fcm_mask: chex.Array | None = None,
        frequencies: chex.Array | None = None,
    ) -> DecoderLayerOutput:
        """
        Forward pass of the attentionNrom module.

        Args:
            hidden_states (chex.Array): Input hidden states.
            attention_mask (chex.Array): Mask to apply on the attention scores.
            position_ids (chex.Array): Position indices for the tokens.
            causal_mask (chex.Array): Causal mask for ensuring autoregressive behavior.
            segment_ids (tp.Optional[chex.Array]): Segment IDs for segment-based attention (optional).
            deterministic (bool): If True, disables dropout for deterministic behavior.
            init_cache (bool): If True, initializes cache for caching keys and values.
            output_attentions (bool): If True, outputs attention weights.
            output_router_logits (bool): If True, outputs router logits.
            fcm_mask (tp.Optional[chex.Array]): fcm mask to be combined with attn mask and causal mask.
        Returns:
            DecoderLayerOutput: A tuple containing the residual_states, hidden states, and the attention weights.
        """

        decoder_output = self.norm_attn_norm(
            hidden_states,
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
        hidden_states = decoder_output.hidden_states
        hidden_states, router_logits = self.ffn(hidden_states)
        hidden_states = decoder_output.residual_states + hidden_states

        return decoder_output.replace(
            hidden_states=hidden_states,
            router_logits=router_logits if output_router_logits else None,
        )


@register_module(TaskType.BASE_MODULE, config=DbrxConfig, model_type="dbrx")
class DbrxModel(EasyDeLBaseModule):
    """
    Base DBRX Model outputting raw hidden-states.

    This model is a Transformer-based model with a mixture of experts (MoE) architecture,
    implementing the DBRX architecture as described in the original paper.

    The model uses specialized attention modules and a router-based MoE FFN layer.
    """

    def __init__(
        self,
        config: DbrxConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize the DbrxModel.

        Args:
                config (DbrxConfig): The model configuration.
                dtype (jnp.dtype, optional): The data type for computation. Defaults to jnp.float32.
                param_dtype (jnp.dtype, optional): The data type for parameters. Defaults to jnp.float32.
                precision (jax.lax.PrecisionLike, optional): The precision to use for matrix multiplication.
                    Defaults to None.
                rngs (nn.Rngs): The random number generators.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.padding_idx = self.config.pad_token_id
        self.vocab_size = self.config.vocab_size
        self.emb_pdrop = self.config.emb_pdrop

        self.wte = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.blocks = [
            DbrxBlock(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for i in range(self.config.n_layers)
        ]
        self.norm_f = nn.LayerNorm(
            self.config.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    @cached_property
    def frequencies(self):
        return self.config.get_basic_frequencies(
            rotary_dim=self.config.hidden_size // self.config.num_attention_heads,
            head_size=self.config.hidden_size // self.config.num_attention_heads,
            base=self.config.attn_config.rope_theta,
        )

    def __call__(
        self,
        input_ids: chex.Array,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
    ) -> MoeModelOutput:
        """
        Forward pass of the model.

        Args:
                input_ids (chex.Array): Token IDs to process.
                attention_mask (Optional[chex.Array], optional): Mask to avoid attention on padding tokens.
                    Defaults to None.
                position_ids (Optional[chex.Array], optional): Position IDs. Defaults to None.
                segment_ids (Optional[chex.Array], optional): Segment IDs for segment-based attention. Defaults to None.
                inputs_embeds (Optional[chex.Array], optional): Pre-computed input embeddings. Defaults to None.
                output_attentions (Optional[bool], optional): Whether to output attention weights. Defaults to None.
                output_hidden_states (Optional[bool], optional): Whether to output hidden states. Defaults to None.
                output_router_logits (Optional[bool], optional): Whether to output router logits. Defaults to None.
                past_key_values (Optional[TransformerCache | PagesCache], optional): Cached key/values.
                    Defaults to None.
                cache_metadata (Optional[TransformerMetadata | PagesMetadata], optional): Cache metadata.
                    Defaults to None.


        Returns:
                MoeModelOutput: The model outputs, either as a named tuple or a standard tuple.
        """
        if output_router_logits is None:
            output_router_logits = self.config.output_router_logits

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids.astype("i4"))

        batch_size, sequence_length = inputs_embeds.shape[:2]
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

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        hidden_states = inputs_embeds
        all_hidden_states = ()
        all_router_logits = ()
        all_attentions = ()
        if mode is None:
            mode = (
                common_types.MODE_DECODE
                if sequence_length == 1 and past_key_values is not None
                else common_types.MODE_TRAIN
            )
        if past_key_values is None:
            past_key_values = TransformerCache.init_empty(len(self.blocks))
        for idx, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            outputs = block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                causal_mask=self.causal_mask,
                segment_ids=segment_ids,
                mode=mode,
                cache_view=past_key_values.views[idx],
                cache_metadata=cache_metadata,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                frequencies=self.frequencies,
            )
            hidden_states = outputs.hidden_states

            hidden_states = apply_logical_sharding(
                hidden_states,
                dynamic_axes=common_types.HiddenStateSharding,
                partition_manager=self.config.partition_manager,
            )

            if output_attentions:
                all_attentions += (outputs.attention_weight,)
            if output_router_logits:
                all_router_logits += (outputs.router_logits,)
            past_key_values[idx] = outputs.cache_view
        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return MoeModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            router_logits=all_router_logits,
        )

    def get_encoder(self) -> nn.Module:
        """
        Returns the encoder part of the model's graph definition.
        For DbrxModel (decoder-only), this is not applicable.
        """
        raise NotImplementedError("DbrxModel is a decoder-only model and does not have a separate encoder.")

    def get_decoder(self) -> nn.Module:
        """
        Returns the decoder part of the model's graph definition.
        For DbrxModel, this is the model itself.
        """
        return self

    def get_lm_head(self) -> nn.Module:
        """
        Returns the language model head of the module.
        DbrxModel does not include the lm_head.
        """
        raise NotImplementedError("DbrxModel does not include the language model head. See DbrxForCausalLM.")

    def get_embedding(self) -> nn.Module:
        """
        Returns the embedding layer of the module.
        """
        return self.wte


@register_module(TaskType.CAUSAL_LM, config=DbrxConfig, model_type="dbrx")
class DbrxForCausalLM(EasyDeLBaseModule):
    def __init__(
        self,
        config: DbrxConfig,
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
        self.transformer = DbrxModel(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.lm_head = ParallelLinear(
            config.hidden_size,
            config.vocab_size,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            use_bias=False,
            rngs=rngs,
            kernel_init=nn.initializers.normal(config.initializer_range),
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )

    def __call__(
        self,
        input_ids: chex.Array,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> MoeCausalLMOutput | tuple:
        if output_router_logits is None:
            output_router_logits = self.config.output_router_logits
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            segment_ids=segment_ids,
        )
        logits = None
        if apply_lm_head:
            logits = self.apply_lm_head(outputs.last_hidden_state)
        aux_loss = None
        if output_router_logits and outputs.router_logits is not None:
            aux_loss = auxiliary_load_balancing_loss_func(
                gate_logits=outputs.router_logits,
                num_experts=self.config.ffn_config.moe_num_experts,
                top_k=self.config.ffn_config.moe_top_k,
                attention_mask=attention_mask,
            )
            aux_loss += aux_loss * self.config.router_aux_loss_coef

        return MoeCausalLMOutput(
            aux_loss=aux_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            last_hidden_state=outputs.last_hidden_state,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            past_key_values=outputs.past_key_values,
        )

    def get_encoder(self) -> nn.Module:
        """
        Returns the encoder part of the model's graph definition.
        For DbrxForCausalLM (decoder-only), this is not applicable.
        """
        raise NotImplementedError("DbrxForCausalLM is a decoder-only model and does not have a separate encoder.")

    def get_decoder(self) -> nn.Module:
        """
        Returns the decoder part of the model's graph definition.
        For DbrxForCausalLM, this is the underlying DbrxModel.
        """
        return self.transformer.get_decoder()

    def get_lm_head(self) -> nn.Module:
        """
        Returns the language model head of the module.
        """
        return self.lm_head

    def get_embedding(self) -> nn.Module:
        """
        Returns the embedding layer of the module.
        """
        # Access the embedding layer through the decoder (DbrxModel)
        return self.transformer.get_embedding()  # Leverages DbrxModel's get_embedding


@register_module(TaskType.SEQUENCE_CLASSIFICATION, config=DbrxConfig, model_type="dbrx")
class DbrxForSequenceClassification(EasyDeLBaseModule):
    def __init__(
        self,
        config: DbrxConfig,
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
        self.transformer = DbrxModel(
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
        self.score = ParallelLinear(
            config.hidden_size,
            config.num_labels,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: chex.Array,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
    ) -> SequenceClassifierOutput:
        if output_router_logits is None:
            output_router_logits = self.config.output_router_logits
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
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
        aux_loss = None
        if output_router_logits and transformer_outputs.router_logits is not None:
            aux_loss = auxiliary_load_balancing_loss_func(
                gate_logits=transformer_outputs.router_logits,
                num_experts=self.config.ffn_config.moe_num_experts,
                top_k=self.config.ffn_config.moe_top_k,
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

    def get_encoder(self) -> nn.Module:
        """
        Returns the encoder part of the model's graph definition.
        For DbrxForSequenceClassification (decoder-only), this is not applicable.
        """
        raise NotImplementedError(
            "DbrxForSequenceClassification is a decoder-only model and does not have a separate encoder."
        )

    def get_decoder(self) -> nn.Module:
        """
        Returns the decoder part of the model's graph definition.
        For DbrxForSequenceClassification, this is the underlying DbrxModel.
        """
        return self.transformer  # self.transformer is the DbrxModel instance

    def get_lm_head(self) -> nn.Module:
        """
        Returns the language model head of the module.
        DbrxForSequenceClassification uses a classification head instead.
        """
        raise NotImplementedError(
            "DbrxForSequenceClassification uses a classification head (self.score), not an lm_head."
        )

    def get_embedding(self) -> nn.Module:
        """
        Returns the embedding layer of the module.
        """
        # Access the embedding layer through the decoder (DbrxModel)
        return self.transformer.get_embedding()  # Leverages DbrxModel's get_embedding
