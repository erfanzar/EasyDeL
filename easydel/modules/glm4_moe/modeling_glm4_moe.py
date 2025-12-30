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

import typing
from functools import partial

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
from easydel.infra.modeling_outputs import (
    DecoderLayerOutput,
    MoeModelOutput,
)
from easydel.infra.utils import ACT2FN, ArrayParam, auto_remat
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
    """Dense feed-forward block used in non-MoE GLM-4-MoE layers."""

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
        column_parallel_linear = partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        row_parallel_linear = partial(
            RowParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.gate_proj = column_parallel_linear(config.hidden_size, config.intermediate_size)
        self.up_proj = column_parallel_linear(config.hidden_size, config.intermediate_size)
        self.down_proj = row_parallel_linear(config.intermediate_size, config.hidden_size)
        self.act_fn = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> jnp.ndarray:
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        gate_output = self.act_fn(checkpoint_name(self.gate_proj(hidden_states), name="mlp_gate"))
        up_output = checkpoint_name(self.up_proj(hidden_states), name="mlp_up")
        hidden_states = checkpoint_name(self.down_proj(gate_output * up_output), name="mlp_down")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return hidden_states, None


class Glm4MoeMLPStack(nn.Module):
    """Glm4Moe MoE MLP using the new ParallelMoELinear layers."""

    reform_param: typing.ClassVar = {
        "gate_up_proj$": {
            "splits": [
                {"name": "gate_proj.kernel", "spliter": lambda x: x[..., : x.shape[-1] // 2]},
                {"name": "up_proj.kernel", "spliter": lambda x: x[..., x.shape[-1] // 2 :]},
            ],
            "inverse_spliter": lambda torch, gate, up: torch.stack((gate, up), dim=-1).flatten(-2),
        },
        "down_proj$": {
            "splits": [
                {"name": "down_proj.kernel", "spliter": lambda x: x},
            ],
            "inverse_spliter": lambda x: x,
        },
    }

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
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.down_proj = RowParallelMoELinear(
            num_experts=config.n_routed_experts,
            in_features=config.intermediate_size,
            out_features=config.hidden_size,
            rngs=rngs,
            use_bias=False,
            kernel_init=nn.initializers.normal(),
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.up_proj = ColumnParallelMoELinear(
            num_experts=config.n_routed_experts,
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            rngs=rngs,
            use_bias=False,
            kernel_init=nn.initializers.normal(),
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(
        self,
        x: Array,
        group_sizes: Array,
        sorted_experts: Array | None = None,
    ) -> Array:
        """Forward pass through MoE MLP."""
        hidden_states = self.act_fn(checkpoint_name(self.gate_proj(x, group_sizes, sorted_experts), name="moe_gate"))
        hidden_states = hidden_states * checkpoint_name(self.up_proj(x, group_sizes, sorted_experts), name="moe_up")
        outputs = checkpoint_name(self.down_proj(hidden_states, group_sizes, sorted_experts), name="moe_expert_output")
        return outputs


class Glm4MoeTopKRouter(nn.Module):
    """Selects top-k experts per token for GLM-4-MoE routing."""

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

        self.kernel = ArrayParam.bound(
            shape=(self.n_routed_experts, config.hidden_size),
            dtype=param_dtype,
            init_method="normal",
            init_kwargs={"stddev": config.initializer_range},
            key=rngs.param(),
        )
        self.e_score_correction_bias = ArrayParam.bound(
            shape=(self.n_routed_experts,),
            dtype=jnp.float32,
            init_method="zeros",
            key=None,
        )

    def get_selected_experts(self, scores):
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
        _, selected_experts = jax.lax.top_k(scores_for_choice, k=self.top_k)

        return selected_experts

    def __call__(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        hidden_states = hidden_states.reshape(-1, self.config.hidden_size)
        router_logits = checkpoint_name(
            jnp.matmul(hidden_states.astype(jnp.float32), self.kernel.value.astype(jnp.float32)),
            name="moe_router_logits",
        )
        scores = jax.nn.sigmoid(router_logits)
        selected_experts = self.get_selected_experts(scores)
        batch_size = scores.shape[0]
        batch_indices = jnp.arange(batch_size)[:, None]
        selected_weights = scores[batch_indices, selected_experts]
        if self.norm_topk_prob:
            denominator = jnp.sum(selected_weights, axis=-1, keepdims=True) + 1e-20
            selected_weights = selected_weights / denominator
        selected_weights = selected_weights * self.routed_scaling_factor
        return selected_weights


class Glm4MoeMoE(BaseMoeModule):
    """GLM-4-MoE feed-forward wrapper combining router and expert stacks."""

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

    def __call__(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> tuple[Array, Array]:
        out, router_logits = self.moe_call(
            hidden_state=hidden_states,
            gate_layer=self.gate,
            expert_layer=self.experts,
            wi_kernel=self.experts.gate_proj.kernel.value,
            wu_kernel=self.experts.up_proj.kernel.value,
            wd_kernel=self.experts.down_proj.kernel.value,
            act_fn=self.experts.act_fn,
        )
        shared_output, _ = self.shared_experts(hidden_states)
        out = out + shared_output

        return checkpoint_name(out, "moe_expert_output"), checkpoint_name(router_logits, "moe_router_logits")


class Glm4MoeAttention(UnifiedAttention):
    """Attention layer variant used inside GLM-4-MoE decoder blocks."""

    def __init__(
        self,
        config: Glm4MoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        self.layer_idx = layer_idx
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
            use_qk_norm=config.use_qk_norm,
        )


class Glm4MoeDecoderLayer(nn.Module):
    """Single decoder block for GLM-4-MoE with attention and MoE MLP."""

    def __init__(
        self,
        config: Glm4MoeConfig,
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
        self.layer_idx = layer_idx
        attn_block = Glm4MoeAttention
        mlp_block = Glm4MoeMLP if layer_idx < config.first_k_dense_replace else Glm4MoeMoE

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

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
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
        output_router_logits: bool | None = None,
    ) -> MoeModelOutput:
        """Forward pass through GLM-4-MoE base model with grouped expert routing.

        Processes input through transformer layers with hybrid dense-sparse architecture.
        Early layers use dense FFN, deeper layers use grouped mixture-of-experts routing.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length). Mutually exclusive
                with inputs_embeds. Token indices in vocabulary [0, config.vocab_size).
            inputs_embeds: Pre-computed input embeddings of shape (batch_size, sequence_length, hidden_size).
                Mutually exclusive with input_ids. Use for custom embedding manipulation.
            attention_mask: Boolean mask of shape (batch_size, sequence_length) where True indicates
                valid tokens and False indicates padding. Auto-computed if None.
            mask_info: Pre-computed mask information for attention operations. Auto-computed via
                MaskInfo.dynamic_init if None.
            position_ids: Position indices of shape (batch_size, sequence_length) for RoPE. Defaults
                to sequential positions from mask_info if None.
            mode: Runtime mode (MODE_TRAIN/DECODE/EVAL). Auto-detected: MODE_DECODE if seq_len=1 and
                cache exists, else MODE_TRAIN.
            past_key_values: Cached key-value states from previous forward passes for autoregressive
                generation. Can be TransformerCache, RaggedPagesCache, or HybridCache.
            cache_metadata: Metadata for paged attention operations (sequence lengths, block tables).
            output_attentions: Whether to return attention weights from all layers. Defaults to False.
            output_hidden_states: Whether to return hidden states from all layers. Defaults to False.
            output_router_logits: Whether to return MoE router logits for load balancing loss computation.
                Defaults to False. Router logits are only returned from MoE layers (layers >= first_k_dense_replace).

        Returns:
            MoeModelOutput containing:
                - last_hidden_state: Final layer output of shape (batch_size, sequence_length, hidden_size).
                - hidden_states: Tuple of all layer outputs if output_hidden_states=True, else None.
                    Each element has shape (batch_size, sequence_length, hidden_size).
                - attentions: Tuple of attention weights if output_attentions=True, else None.
                    Each element has shape (batch_size, num_heads, sequence_length, sequence_length).
                - router_logits: Tuple of router logits from MoE layers if output_router_logits=True, else None.
                    Each element has shape (batch_size, sequence_length, n_routed_experts).
                - past_key_values: Updated cache for next generation step.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.astype("i4"))
        sequence_length = inputs_embeds.shape[1]
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        all_router_logits = () if output_router_logits else None

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
class Glm4MoeForCausalLM(BaseCausalLMModule[Glm4MoeModel, Glm4MoeConfig]):
    """GLM4 MoE model with a language modeling head for causal language modeling tasks."""

    _task_type = TaskType.CAUSAL_LM
    _model_type = "glm4_moe"
    _config_class = Glm4MoeConfig

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
            base_model_class=Glm4MoeModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
            router_aux_loss_coef=None,  # NOTE: we dont use aux loss for Glm4Moe
        )


@register_module(TaskType.SEQUENCE_CLASSIFICATION, config=Glm4MoeConfig, model_type="glm4_moe")
class Glm4MoeForSequenceClassification(BaseSequenceClassificationModule[Glm4MoeModel, Glm4MoeConfig]):
    """GLM4 MoE model for sequence classification tasks."""

    _task_type = TaskType.SEQUENCE_CLASSIFICATION
    _model_type = "glm4_moe"
    _config_class = Glm4MoeConfig

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
            base_model_class=Glm4MoeModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            score_bias=False,
        )
