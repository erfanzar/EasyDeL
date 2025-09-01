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
import jax.numpy as jnp
from eformer import common_types
from eformer.escale import apply_logical_sharding
from eformer.loggings import get_logger
from flax import nnx as nn

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.loss_utils import auxiliary_load_balancing_loss_func
from easydel.infra.modeling_outputs import (
    AttentionLayerOutput,
    BaseModelOutput,
    DecoderLayerOutput,
    MoeCausalLMOutput,
    MoeModelOutput,
)
from easydel.infra.utils import ACT2FN, auto_remat, get_dot_general_by_bits
from easydel.layers.attention import AttentionModule, FlexibleAttentionModule
from easydel.layers.caching import (
    PagesCache,
    PagesCacheView,
    PagesMetadata,
    TransformerCache,
    TransformerCacheMetaData,
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

from .xerxes2_configuration import Xerxes2Config as Xerxes2Config

logger = get_logger(__name__)


class Xerxes2Attention(AttentionModule):
    def __init__(
        self,
        config: Xerxes2Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(config=config)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.qhead_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.vhead_dim = config.vhead_dim

        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim

        linear_class = functools.partial(
            ParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )

        if self.config.q_lora_dim is not None:
            self.qa_proj = linear_class(config.hidden_size, config.q_lora_dim)
            self.qa_norm = nn.LayerNorm(
                config.q_lora_dim,
                rngs=rngs,
                dtype=dtype,
                param_dtype=param_dtype,
            )
            self.qb_proj = linear_class(config.q_lora_dim, self.num_heads * self.qhead_dim)
        else:
            self.qc_proj = linear_class(config.hidden_size, self.num_heads * self.qhead_dim)
        self.kv_mqa_proj = linear_class(
            config.hidden_size,
            config.kv_lora_dim + config.qk_rope_head_dim,
        )
        self.kv_norm = nn.LayerNorm(
            config.kv_lora_dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.kvi_proj = linear_class(
            config.kv_lora_dim,
            self.num_heads * (self.qhead_dim - self.qk_rope_head_dim + self.vhead_dim),
        )
        self.o_proj = linear_class(
            self.num_heads * self.vhead_dim,
            self.config.hidden_size,
        )

        self.attention_performer = FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=self.qhead_dim**-0.5,
            dropout_prob=0.0,
        )
        self.rotary = self.config.get_basic_rope(
            self.dtype,
            self.qk_rope_head_dim,
            self.qk_rope_head_dim,
            config.rope_theta,
        )

    def _split_heads(self, hidden_states, num_heads):
        return hidden_states.reshape((*hidden_states.shape[:2], num_heads, self.head_dim))

    def __call__(
        self,
        hidden_states: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        causal_mask: chex.Array | bool | None,
        frequencies: tuple[chex.Array, chex.Array],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | PagesCacheView | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool = False,
    ):
        """Forward pass of the attention module."""
        batch_size, sequence_length = hidden_states.shape[:2]
        if self.config.q_lora_dim is None:
            query_states = self.qc_proj(hidden_states)
        else:
            query_states = self.qb_proj(self.qa_norm(self.qa_proj(hidden_states)))
        query_states = query_states.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.qhead_dim,
        )
        compressed_kv = self.kv_mqa_proj(hidden_states)
        compressed_kv = compressed_kv.reshape(
            batch_size,
            sequence_length,
            1,
            self.config.kv_lora_dim + self.config.qk_rope_head_dim,
        )

        q_nope, q_pe = (
            query_states[..., : self.qk_nope_head_dim],
            query_states[..., self.qk_nope_head_dim :],
        )
        k_pe = compressed_kv[..., self.config.kv_lora_dim :]
        compressed_kv = compressed_kv[..., : self.config.kv_lora_dim]
        kv = self.kvi_proj(self.kv_norm(compressed_kv))
        value_states = kv[..., self.qk_nope_head_dim : self.qk_nope_head_dim + self.vhead_dim]
        k_nope = kv[..., : self.qk_nope_head_dim]

        q_pe, k_pe = self.rotary(
            positions=position_ids,
            query=q_pe,
            key=k_pe,
            frequencies=frequencies,
        )

        query_states = (
            jnp.zeros(
                (batch_size, sequence_length, self.num_heads, self.qhead_dim),
                dtype=q_pe.dtype,
            )
            .at[..., : self.qk_nope_head_dim]
            .set(q_nope)
            .at[..., self.qk_nope_head_dim :]
            .set(q_pe)
        )
        key_states = (
            jnp.zeros(
                (batch_size, sequence_length, 1, self.qhead_dim),
                dtype=q_pe.dtype,
            )
            .at[..., : self.qk_nope_head_dim]
            .set(k_nope)
            .at[..., self.qk_nope_head_dim :]
            .set(k_pe)
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

        attn_output = self.o_proj(self.shard_attention_prod(self._merge_heads(attentions.attention_outputs)))

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class Xerxes2MLP(nn.Module):
    def __init__(
        self,
        config: Xerxes2Config,
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

        self.act = nn.silu
        linear_class = functools.partial(
            ParallelLinear,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        self.gate_up_proj = linear_class(config.hidden_size, 2 * config.intermediate_size, rngs=rngs)
        self.down_proj = linear_class(config.intermediate_size, config.hidden_size, rngs=rngs)

    def __call__(self, hidden_states):
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        up_states = self.gate_up_proj(hidden_states)
        gate, up_states = jnp.split(up_states, 2, axis=-1)
        hidden_states = self.down_proj(up_states * nn.silu(gate))
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return hidden_states


class Xerxes2MoeMLPStack(nn.Module):
    """Xerxes2Moe MoE MLP using the new ParallelMoELinear layers."""

    def __init__(
        self,
        config: Xerxes2Config,
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
            num_experts=config.num_experts,
            in_features=config.hidden_size,
            out_features=config.moe_intermediate_size,
            rngs=rngs,
            kernel_init=nn.initializers.normal(),
            use_bias=False,
            use_pallas_group_matmul=config.use_pallas_group_matmul,
            partition_manager=config.partition_manager,
        )
        self.down_proj = RowParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.moe_intermediate_size,
            out_features=config.hidden_size,
            rngs=rngs,
            use_bias=False,
            kernel_init=nn.initializers.normal(),
            use_pallas_group_matmul=config.use_pallas_group_matmul,
            partition_manager=config.partition_manager,
        )
        self.up_proj = ColumnParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.hidden_size,
            out_features=config.moe_intermediate_size,
            rngs=rngs,
            use_bias=False,
            kernel_init=nn.initializers.normal(),
            use_pallas_group_matmul=config.use_pallas_group_matmul,
            partition_manager=config.partition_manager,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(self, hidden_states: chex.Array, group_sizes: chex.Array) -> chex.Array:
        """Forward pass through MoE MLP."""
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states, group_sizes)) * self.up_proj(hidden_states, group_sizes),
            group_sizes,
        )


class Xerxes2MoeSparseBlock(BaseMoeModule):
    """Sparse Mixture of Experts (MoE) block for Xerxes2 MoE.

    This block routes input hidden states to a selected subset of experts
    and combines their outputs.

    Attributes:
        config (Xerxes2MoeConfig): Configuration object for the model.
        gate (ParallelLinear): Linear layer for the gating network.
        experts (nn.List[Xerxes2MoeMLP]): List of expert MLP modules.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for matrix multiplications.
        rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: Xerxes2Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the Xerxes2MoeSparseBlock module.

        Args:
            config (Xerxes2MoeConfig): The configuration object for the model.
            dtype (jnp.dtype): Data type for computations (default: jnp.float32).
            param_dtype (jnp.dtype): Data type for parameters (default: jnp.float32).
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations (default: None).
            rngs (nn.Rngs): Random number generators.
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
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.gate = ParallelLinear(
            config.hidden_size,
            config.num_experts,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=nn.initializers.normal(config.initializer_range),
        )

        self.experts = Xerxes2MoeMLPStack(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, hidden_states: chex.Array) -> tuple[chex.Array, chex.Array]:
        """Forward pass of the Sparse MoE block.

        Args:
            hidden_states (chex.Array): Input hidden states (batch_size * sequence_length, hidden_dim).

        Returns:
            tp.Tuple[chex.Array, chex.Array]: A tuple containing:
                - final_hidden_states (chex.Array): The output hidden states after MoE processing.
                - router_logits (chex.Array): The logits output by the gating network.
        """
        out, router_logits = self._moe_call(
            gate_layer=self.gate,
            expert_layer=self.experts,
            hidden_state=hidden_states,
            output_metrics=False,
            validate_inputs=True,
            apply_capacity_constraint=False,
        )
        return out, router_logits


class Xerxes2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Xerxes2Config,
        layer_idx: int,
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

        attn_block, mlp_block, moe_block = auto_remat(
            Xerxes2Attention,
            Xerxes2MLP,
            Xerxes2MoeSparseBlock,
            policy=config.gradient_checkpointing,
        )
        self.self_attn = attn_block(
            self.config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.is_moe = (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        )
        if self.is_moe:
            self.mlp = moe_block(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        else:
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
        self.input_layernorm = rms()
        self.post_attention_layernorm = rms()
        self.pre_feedforward_layernorm = rms()
        self.post_feedforward_layernorm = rms()

    def __call__(
        self,
        hidden_states: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        causal_mask: chex.Array | bool | None,
        frequencies: tuple[chex.Array, chex.Array],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | PagesCacheView | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
    ):
        """
        Forward pass of the module block.

        Args:
            hidden_states (chex.Array): Input hidden states.
            attention_mask (chex.Array): Mask to apply on the attention scores.
        Returns:
            tp.Tuple[chex.Array, chex.Array]: A tuple containing the attention output and the attention weights.
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
            attention_mask,
            position_ids,
            causal_mask,
            frequencies,
            mode,
            cache_view,
            cache_metadata,
            segment_ids,
            output_attentions,
        )
        hidden_states = self.post_attention_layernorm(attn_outputs.attention_output)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        router_logits = None
        if self.is_moe:
            hidden_states, router_logits = hidden_states
        hidden_states = self.post_feedforward_layernorm(hidden_states)
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
            router_logits=router_logits if output_router_logits else None,
        )


@register_module(TaskType.BASE_MODULE, config=Xerxes2Config, model_type="xerxes2")
class Xerxes2Model(EasyDeLBaseModule):
    def __init__(
        self,
        config: Xerxes2Config,
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
        self.embed_tokens = nn.Embed(
            config.vocab_size,
            config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.layers = [
            Xerxes2DecoderLayer(
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
        )

    @functools.cached_property
    def frequencies(self) -> jnp.ndarray:
        """Returns frequency values from the config."""
        return self.config.get_basic_frequencies(self.config.qk_rope_head_dim)

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
    ) -> BaseModelOutput:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(inputs=input_ids.astype("i4"))

        batch_size, sequence_length, _ = inputs_embeds.shape

        if output_router_logits is None:
            output_router_logits = self.config.output_router_logits

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
            if output_router_logits:
                all_router_logits += (layer_outputs.router_logits,)

            past_key_values[idx] = layer_outputs.cache_view

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return MoeModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            past_key_values=past_key_values,
            router_logits=all_router_logits,
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


@register_module(TaskType.CAUSAL_LM, config=Xerxes2Config, model_type="xerxes2")
class Xerxes2ForCausalLM(EasyDeLBaseModule):
    def __init__(
        self,
        config: Xerxes2Config,
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
        self.model = Xerxes2Model(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.lm_head = ParallelLinear(
            self.config.hidden_size,
            self.config.vocab_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
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
        aux_loss = None
        if output_router_logits and outputs.router_logits is not None:
            aux_loss = auxiliary_load_balancing_loss_func(
                gate_logits=outputs.router_logits,
                num_experts=self.config.num_experts,
                top_k=self.config.num_experts_per_tok,
                attention_mask=attention_mask,
            )
            aux_loss += aux_loss * self.config.router_aux_loss_coef

        return MoeCausalLMOutput(
            logits=lm_logits,
            aux_loss=aux_loss,
            hidden_states=outputs.hidden_states,
            last_hidden_state=outputs.last_hidden_state,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
        )

    def create_cache_metadata(
        self,
        batch_size: int,
        max_length: int,
        pad_token_id: int | None = None,
    ):
        if pad_token_id is None:
            if hasattr(self, "generation_config"):
                pad_token_id = self.generation_config.pad_token_id
            elif hasattr(self.config, "pad_token_id"):
                pad_token_id = self.config.pad_token_id
            else:
                pad_token_id = 0
        head_dim = getattr(self.config, "head_dim", None)
        if head_dim is None:
            head_dim = self.config.hidden_size // self.config.num_attention_heads
        num_key_value_heads = getattr(self.config, "num_key_value_heads", None)
        if num_key_value_heads is None:
            num_key_value_heads = self.config.num_attention_heads
        return TransformerCacheMetaData.create(
            num_hidden_layers=self.config.num_hidden_layers,
            batch_size=batch_size,
            sequence_length=max_length,
            num_heads=1,
            key_dim=self.config.qk_rope_head_dim + self.config.qk_nope_head_dim,
            value_dim=self.config.vhead_dim,
        )

    def init_cache(
        self,
        batch_size: int,
        max_length: int,
        starts: int | None = None,
        shardings: dict | None = None,
        pad_token_id: int | None = None,
    ):
        shardings = shardings or dict()
        return TransformerCache.init_cache(
            dtype=self.config.kvdtype,
            partition_manager=self.config.partition_manager,
            metadata=self.create_cache_metadata(
                batch_size=batch_size,
                max_length=max_length,
                pad_token_id=pad_token_id,
            ),
            quantizer=self._quant_class(
                quantization_method=self.config.kv_cache_quantization_method,
                block_size=self.config.kv_cache_quantization_blocksize,
                quantization_platform=self.config.platform,
            ),
            mesh=self.config.mesh,
            starts=starts,
            mask_type_details=self.config.get_mask_details(),
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
