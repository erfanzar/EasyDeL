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
from easydel.layers.moe import (
    BaseMoeModule,
    ColumnParallelMoELinear,
    MoeLoadBalancingStrategy,
    MoeRoutingStrategy,
    RowParallelMoELinear,
)
from easydel.layers.norms import RMSNorm

from .glm4_moe_configuration import Glm4MoeConfig


class Glm4MoeMLP(nn.Module):
    def __init__(
        self,
        config: Glm4MoeConfig,
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
        linear_class = partial(
            ParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        self.gate_proj = linear_class(config.hidden_size, config.intermediate_size)
        self.up_proj = linear_class(config.hidden_size, config.intermediate_size)
        self.down_proj = linear_class(config.intermediate_size, config.hidden_size)
        self.act_fn = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        gate_output = self.act_fn(self.gate_proj(hidden_states))
        up_output = self.up_proj(hidden_states)
        hidden_states = self.down_proj(gate_output * up_output)
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return hidden_states, None


class Glm4MoeMLPStack(nn.Module):
    """Glm4Moe MoE MLP using the new ParallelMoELinear layers."""

    def __init__(
        self,
        config: Glm4MoeConfig,
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
        self.gate_proj = ColumnParallelMoELinear(
            num_experts=config.n_routed_experts,
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            rngs=rngs,
            kernel_init=nn.initializers.normal(),
            use_bias=False,
            use_pallas_group_matmul=config.use_pallas_group_matmul,
            partition_manager=config.partition_manager,
        )
        self.down_proj = RowParallelMoELinear(
            num_experts=config.n_routed_experts,
            in_features=config.intermediate_size,
            out_features=config.hidden_size,
            rngs=rngs,
            use_bias=False,
            kernel_init=nn.initializers.normal(),
            use_pallas_group_matmul=config.use_pallas_group_matmul,
            partition_manager=config.partition_manager,
        )
        self.up_proj = ColumnParallelMoELinear(
            num_experts=config.n_routed_experts,
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            rngs=rngs,
            use_bias=False,
            kernel_init=nn.initializers.normal(),
            use_pallas_group_matmul=config.use_pallas_group_matmul,
            partition_manager=config.partition_manager,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(self, x: chex.Array, group_sizes: chex.Array) -> chex.Array:
        """Forward pass through MoE MLP."""
        hidden_states = self.act_fn(self.gate_proj(x, group_sizes))
        hidden_states = hidden_states * self.up_proj(x, group_sizes)
        outputs = self.down_proj(hidden_states, group_sizes)
        return outputs


class Glm4MoeTopKRouter(nn.Module):
    def __init__(
        self,
        config: Glm4MoeConfig,
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
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.kernel = nn.Param(
            jax.nn.initializers.normal(stddev=config.initializer_range)(
                rngs.param(),
                shape=(self.n_routed_experts, config.hidden_size),
                dtype=param_dtype,
            )
        )
        self.e_score_correction_bias = nn.Param(jnp.zeros((self.n_routed_experts,), dtype=jnp.float32))

    def get_topk_indices(self, scores):
        scores_for_choice = scores + self.e_score_correction_bias.value
        batch_size = scores_for_choice.shape[0]
        group_scores = scores_for_choice.reshape(batch_size, self.n_group, self.n_routed_experts // self.n_group)
        top2_per_group = jax.lax.top_k(group_scores, k=2)[0]
        group_scores_sum = jnp.sum(top2_per_group, axis=-1)
        scores_for_choice = jnp.where(
            jax.nn.one_hot(
                jax.lax.top_k(
                    group_scores_sum,
                    k=self.topk_group,
                )[-1],
                self.n_group,
                dtype=scores.dtype,
            )[:, :, None]
            .repeat(self.n_routed_experts // self.n_group, axis=2)
            .reshape(batch_size, self.n_routed_experts),
            scores_for_choice,
            0.0,
        )
        _, topk_indices = jax.lax.top_k(scores_for_choice, k=self.top_k)

        return topk_indices

    def __call__(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.config.hidden_size)
        router_logits = jnp.matmul(hidden_states.astype(jnp.float32), self.kernel.value.astype(jnp.float32))
        scores = jax.nn.sigmoid(router_logits)
        topk_indices = self.get_topk_indices(scores)
        batch_size = scores.shape[0]
        batch_indices = jnp.arange(batch_size)[:, None]
        topk_weights = scores[batch_indices, topk_indices]
        if self.norm_topk_prob:
            denominator = jnp.sum(topk_weights, axis=-1, keepdims=True) + 1e-20
            topk_weights = topk_weights / denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_weights


class Glm4MoeMoE(BaseMoeModule):
    def __init__(
        self,
        config: Glm4MoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(
            config=config,
            n_routed_experts=config.n_routed_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            lbl_coef=getattr(config, "router_aux_loss_coef", None),
            rzl_coef=getattr(config, "router_z_loss_coef", None),
            routing_strategy=MoeRoutingStrategy.TOP_K,
            load_balancing_strategy=MoeLoadBalancingStrategy.STANDARD,
        )
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        self.experts = Glm4MoeMLPStack(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.gate = Glm4MoeTopKRouter(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.shared_experts = Glm4MoeMLP(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, hidden_states: chex.Array) -> tuple[chex.Array, chex.Array]:
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        final_hidden_states, router_logits = self._moe_call(
            gate_layer=self.gate,
            expert_layer=self.experts,
            hidden_state=hidden_states,
            output_metrics=False,
            validate_inputs=True,
        )
        shared_output, _ = self.shared_experts(hidden_states)
        final_hidden_states = final_hidden_states + shared_output
        return final_hidden_states, router_logits


class Glm4MoeAttention(AttentionModule):
    def __init__(
        self,
        config: Glm4MoeConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(config=config)
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.hidden_size = config.hidden_size
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.head_dim = head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.use_qk_norm = config.use_qk_norm

        linear_class = partial(
            ParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.attention_bias,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        self.q_proj = linear_class(config.hidden_size, config.num_attention_heads * self.head_dim, rngs=rngs)
        self.k_proj = linear_class(config.hidden_size, config.num_key_value_heads * self.head_dim, rngs=rngs)
        self.v_proj = linear_class(config.hidden_size, config.num_key_value_heads * self.head_dim, rngs=rngs)
        self.o_proj = linear_class(config.num_attention_heads * self.head_dim, config.hidden_size, rngs=rngs)

        if self.use_qk_norm:
            self.q_norm = RMSNorm(
                dim=self.head_dim,
                eps=config.rms_norm_eps,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            self.k_norm = RMSNorm(
                dim=self.head_dim,
                eps=config.rms_norm_eps,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

        self.rotary = self.config.get_basic_rope(self.dtype, self.head_dim, self.head_dim, True)
        self.attention_performer = FlexibleAttentionModule(
            rngs=rngs,
            base_config=self.config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=self.config.attention_dropout,
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
    ) -> tuple[chex.Array, chex.Array]:
        batch_size, sequence_length = hidden_states.shape[:2]
        query_states, key_states, value_states = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
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

        # Apply QK normalization if enabled
        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

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
        attn_output = self.o_proj(self.shard_attention_prod(attn_output=self._merge_heads(attentions.attention_outputs)))

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class Glm4MoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Glm4MoeConfig,
        layer_idx: int,
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
        self.layer_idx = layer_idx
        attn_block = Glm4MoeAttention
        mlp_block = Glm4MoeMLP if layer_idx < config.first_k_dense_replace else Glm4MoeMoE

        attn_block, mlp_block = auto_remat(
            attn_block,
            mlp_block,
            policy=config.gradient_checkpointing,
        )
        self.self_attn = attn_block(
            config=config,
            layer_idx=layer_idx,
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
        hidden_states: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        causal_mask: chex.Array | bool | None,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | PagesCacheView | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        fcm_mask: chex.Array | None = None,
        frequencies: chex.Array | None = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            causal_mask=causal_mask,
            mode=mode,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            segment_ids=segment_ids,
            output_attentions=output_attentions,
            fcm_mask=fcm_mask,
            frequencies=frequencies,
        )
        hidden_states = residual + attn_outputs.attention_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states, router_logits = self.mlp(hidden_states)

        hidden_states = residual + hidden_states
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


@register_module(TaskType.BASE_MODULE, config=Glm4MoeConfig, model_type="glm4_moe")
class Glm4MoeModel(EasyDeLBaseModule):
    """GLM4 MoE model implementation."""

    def __init__(
        self,
        config: Glm4MoeConfig,
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
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            rngs=rngs,
        )
        self.layers = [
            Glm4MoeDecoderLayer(
                config=config,
                layer_idx=layer_idx,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for layer_idx in range(self.config.num_hidden_layers)
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
        input_ids: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
    ) -> MoeModelOutput:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.astype("i4"))
        batch_size, sequence_length, _ = inputs_embeds.shape
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        all_router_logits = () if output_router_logits else None

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
                attention_mask=attention_mask,
                position_ids=position_ids,
                mode=mode,
                cache_view=past_key_values.views[idx],
                cache_metadata=cache_metadata,
                causal_mask=self.causal_mask,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                segment_ids=segment_ids,
                frequencies=self.frequencies,
            )

            hidden_states = layer_outputs.hidden_states

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            if output_router_logits and layer_outputs.router_logits is not None:
                all_router_logits += (layer_outputs.router_logits,)

            past_key_values[idx] = layer_outputs.cache_view

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return MoeModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            router_logits=all_router_logits,
            past_key_values=past_key_values,
        )

    def get_encoder(self):
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        return self

    def get_lm_head(self):
        raise NotImplementedError("The base model does not have a language model head.")

    def get_embedding(self):
        return self.embed_tokens


@register_module(TaskType.CAUSAL_LM, config=Glm4MoeConfig, model_type="glm4_moe")
class Glm4MoeForCausalLM(EasyDeLBaseModule):
    """GLM4 MoE model with a language modeling head for causal language modeling tasks."""

    def __init__(
        self,
        config: Glm4MoeConfig,
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
        self.model = Glm4MoeModel(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.vocab_size = config.vocab_size
        self.lm_head = ParallelLinear(
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
        input_ids: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
    ) -> MoeCausalLMOutput:
        output_router_logits = False  # NOTE: we dont use aux loss for Glm4Moe
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
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
            lm_logits = self.apply_lm_head(hidden_states)

        aux_loss = 0
        if output_router_logits and outputs.router_logits is not None:
            aux_loss = auxiliary_load_balancing_loss_func(
                gate_logits=outputs.router_logits,
                num_experts=self.config.n_routed_experts,
                top_k=self.config.num_experts_per_tok,
                attention_mask=attention_mask,
            )

        return MoeCausalLMOutput(
            aux_loss=aux_loss,
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            last_hidden_state=outputs.last_hidden_state,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            past_key_values=outputs.past_key_values,
        )

    def get_encoder(self):
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        return self.model.get_decoder()

    def get_lm_head(self):
        return self.lm_head

    def get_embedding(self):
        return self.model.get_embedding()


@register_module(TaskType.SEQUENCE_CLASSIFICATION, config=Glm4MoeConfig, model_type="glm4_moe")
class Glm4MoeForSequenceClassification(EasyDeLBaseModule):
    """GLM4 MoE model for sequence classification tasks."""

    def __init__(
        self,
        config: Glm4MoeConfig,
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
        self.model = Glm4MoeModel(
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
        segment_ids: chex.Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
    ) -> SequenceClassifierOutput:
        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
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

        aux_loss = None
        if output_router_logits and transformer_outputs.router_logits is not None:
            aux_loss = auxiliary_load_balancing_loss_func(
                gate_logits=transformer_outputs.router_logits,
                num_experts=self.config.n_routed_experts,
                top_k=self.config.num_experts_per_tok,
                attention_mask=attention_mask,
            )
            aux_loss = aux_loss * self.config.router_aux_loss_coef

        return SequenceClassifierOutput(
            logits=pooled_logits,
            past_key_values=past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            aux_loss=aux_loss,
        )

    def get_encoder(self):
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        return self.model.get_decoder()

    def get_lm_head(self):
        raise NotImplementedError("This model has a sequence classification head, not a language model head.")

    def get_embedding(self):
        return self.model.get_embedding()
