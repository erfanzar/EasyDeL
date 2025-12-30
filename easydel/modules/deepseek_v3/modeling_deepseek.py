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
import typing
from functools import partial
from typing import ClassVar

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
    BaseModelOutput,
    DecoderLayerOutput,
    MoeCausalLMOutput,
    MoeModelOutput,
)
from easydel.infra.utils import ACT2FN, ArrayParam, auto_remat
from easydel.layers.attention import FlexibleAttentionModule
from easydel.layers.attention_unified import UnifiedAttention
from easydel.layers.base_modules import BaseCausalLMModule
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
from easydel.layers.rotary_embedding import yarn_get_mscale

from .deepseek_configuration import DeepseekV3Config


class DeepseekV3MLP(nn.Module):
    """Standard DeepSeek V3 feed-forward network used in dense decoder layers."""

    def __init__(
        self,
        config: DeepseekV3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        hidden_size=None,
        intermediate_size=None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        linear_class = partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.gate_proj = linear_class(self.hidden_size, self.intermediate_size)
        self.down_proj = linear_class(self.intermediate_size, self.hidden_size)
        self.up_proj = linear_class(self.hidden_size, self.intermediate_size)
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        if hidden_states.ndim == 3:  # if not in moe infer
            hidden_states = apply_logical_sharding(
                hidden_states,
                dynamic_axes=common_types.HiddenStateSharding,
                partition_manager=self.config.partition_manager,
            )
        gate = checkpoint_name(self.act_fn(self.gate_proj(hidden_states)), "mlp_gate")
        up = checkpoint_name(self.up_proj(hidden_states), "mlp_up")
        hidden_states = checkpoint_name(self.down_proj(gate * up), "mlp_down")
        if hidden_states.ndim == 3:  # if not in moe infer
            hidden_states = apply_logical_sharding(
                hidden_states,
                dynamic_axes=common_types.HiddenStateSharding,
                partition_manager=self.config.partition_manager,
            )
        return checkpoint_name(hidden_states, "mlp_output")


class MoEGate(nn.Module):
    """Top-k routing gate that scores tokens for the mixture-of-experts blocks."""

    def __init__(
        self,
        config: DeepseekV3Config,
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
        self.top_k = self.config.num_experts_per_tok
        self.n_routed_experts = self.config.n_routed_experts
        self.routed_scaling_factor = self.config.routed_scaling_factor
        self.scoring_func = self.config.scoring_func
        self.seq_aux = self.config.seq_aux
        self.topk_method = self.config.topk_method
        self.n_group = self.config.n_group
        self.topk_group = self.config.topk_group
        self.norm_topk_prob = self.config.norm_topk_prob
        self.gating_dim = self.config.hidden_size
        kernel = nn.initializers.kaiming_uniform()(
            rngs.param(),
            (self.gating_dim, self.n_routed_experts),
            param_dtype,
        )

        self.kernel = ArrayParam.bound(
            shape=kernel.shape,
            dtype=param_dtype,
            init_method="kaiming_uniform",
            key=rngs.param(),
            value=kernel,
        )
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = ArrayParam.bound(
                shape=(self.n_routed_experts,),
                dtype=param_dtype,
                init_method="zeros",
                key=rngs.params(),
            )

    def __call__(self, hidden_states):
        squ, _ = hidden_states.shape
        logits = jnp.dot(
            hidden_states.astype(jnp.float32),
            self.kernel.value.astype(jnp.float32),
            precision=self.precision,
        )

        if self.scoring_func == "sigmoid":
            scores = jax.nn.sigmoid(logits)
        else:
            raise NotImplementedError(f"insupportable scoring function for MoE gating: {self.scoring_func}")

        if self.topk_method == "noaux_tc":
            scores_for_choice = scores + self.e_score_correction_bias
            group_scores = scores_for_choice.reshape(squ, self.n_group, -1)
            top2_scores = jax.lax.top_k(group_scores, k=2)[0]
            group_scores = jnp.sum(top2_scores, axis=-1)

            group_idx = jax.lax.top_k(group_scores, k=self.topk_group)[1]

            group_mask = jnp.zeros_like(group_scores)
            indices = jnp.arange(group_mask.shape[0])[:, None]
            group_mask = group_mask.at[indices, group_idx].set(1.0)

            score_mask = jnp.repeat(
                group_mask[:, :, None],
                self.n_routed_experts // self.n_group,
                axis=2,
            ).reshape(squ, -1)

            masked_scores = jnp.where(score_mask > 0, scores_for_choice, 0.0)
            topk_weight, _ = jax.lax.top_k(masked_scores, k=self.top_k)
        else:
            raise NotImplementedError(f"insupportable TopK function for MoE gating: {self.topk_method}")

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = jnp.sum(topk_weight, axis=-1, keepdims=True) + 1e-20
            topk_weight = topk_weight / denominator

        return topk_weight * self.routed_scaling_factor


class DeepseekV3MLPMoE(nn.Module):
    """Mixture-of-experts feed-forward module parameterized by the DeepSeek V3 config."""

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
        config: DeepseekV3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        self.precision = precision

        imz = intermediate_size or config.intermediate_size
        hs = hidden_size or config.hidden_size
        self.gate_proj = ColumnParallelMoELinear(
            num_experts=config.n_routed_experts,
            in_features=hs,
            out_features=imz,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nn.initializers.normal(),
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            rngs=rngs,
        )
        self.up_proj = ColumnParallelMoELinear(
            num_experts=config.n_routed_experts,
            in_features=hs,
            out_features=imz,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nn.initializers.normal(),
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            rngs=rngs,
        )
        self.down_proj = RowParallelMoELinear(
            num_experts=config.n_routed_experts,
            in_features=imz,
            out_features=hs,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nn.initializers.normal(),
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            rngs=rngs,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(
        self,
        hidden_states: Array,
        group_sizes: Array,
        sorted_experts: Array | None = None,
    ):
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        gate = checkpoint_name(self.act_fn(self.gate_proj(hidden_states, group_sizes, sorted_experts)), "mlp_gate")
        up = checkpoint_name(self.up_proj(hidden_states, group_sizes, sorted_experts), "mlp_up")
        down = checkpoint_name(self.down_proj(gate * up, group_sizes, sorted_experts), "mlp_down")
        return checkpoint_name(
            apply_logical_sharding(
                down,
                dynamic_axes=common_types.HiddenStateSharding,
                partition_manager=self.config.partition_manager,
            ),
            "mlp_output",
        )


class DeepseekV3MoE(BaseMoeModule):
    """Wraps gating and expert networks to apply DeepSeek V3 MoE feed-forward processing."""

    def __init__(
        self,
        config: DeepseekV3Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
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
        self.rngs = rngs
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts_per_rank = config.n_routed_experts
        self.experts = DeepseekV3MLPMoE(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            intermediate_size=self.config.moe_intermediate_size,
            rngs=rngs,
        )
        self.gate = MoEGate(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV3MLP(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                intermediate_size=intermediate_size,
                rngs=rngs,
            )

    def __call__(self, hidden_states: Array):
        out, router_logits = self.moe_call(
            hidden_state=hidden_states,
            gate_layer=self.gate,
            expert_layer=self.experts,
            wi_kernel=self.experts.gate_proj.kernel.value,
            wu_kernel=self.experts.up_proj.kernel.value,
            wd_kernel=self.experts.down_proj.kernel.value,
            act_fn=self.experts.act_fn,
        )
        if self.config.n_shared_experts is not None:
            out = out + self.shared_experts(hidden_states)
        return checkpoint_name(out, "moe_expert_output"), checkpoint_name(router_logits, "moe_router_logits")


class DeepseekV3Attention(UnifiedAttention):
    """DeepSeek V3 Multi-head Latent Attention.

    Inherits MLA implementation from UnifiedAttention base class.
    """

    projection_mapping: ClassVar[dict[str, str]] = {
        "mla_q_proj": "q_proj",
        "mla_q_a_proj": "q_a_proj",
        "mla_q_a_layernorm": "q_a_layernorm",
        "mla_q_b_proj": "q_b_proj",
        "mla_kv_a_proj_with_mqa": "kv_a_proj_with_mqa",
        "mla_kv_a_layernorm": "kv_a_layernorm",
        "mla_kv_b_proj": "kv_b_proj",
        "output_projection": "o_proj",
    }

    def __init__(
        self,
        config: DeepseekV3Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        # Set MLA-specific dimensions before calling super().__init__()
        # so they're available in define_network
        self.config = config
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank

        super().__init__(
            config,
            dtype,
            param_dtype,
            precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="mla",
            causal=True,
            use_mla_lora=config.q_lora_rank is not None,
        )

        # Override head_dim for MLA - use value head dimension for output merging
        self.head_dim = self.v_head_dim

    def define_network(
        self,
        config: DeepseekV3Config,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        precision: jax.lax.Precision,
        rngs: nn.Rngs,
    ):
        """Define MLA-specific network structure."""

        # Query projection with optional LoRA
        if not self.use_mla_lora:
            setattr(
                self,
                self.projection_mapping["mla_q_proj"],
                ColumnParallelLinear(
                    config.hidden_size,
                    config.num_attention_heads * self.q_head_dim,
                    rngs=rngs,
                    use_bias=False,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    kernel_init=jax.nn.initializers.normal(config.initializer_range),
                    precision=precision,
                ),
            )
        else:
            setattr(
                self,
                self.projection_mapping["mla_q_a_proj"],
                ColumnParallelLinear(
                    config.hidden_size,
                    config.q_lora_rank,
                    rngs=rngs,
                    use_bias=config.attention_bias,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    kernel_init=jax.nn.initializers.normal(config.initializer_range),
                    precision=precision,
                ),
            )
            setattr(
                self,
                self.projection_mapping["mla_q_a_layernorm"],
                RMSNorm(
                    config.q_lora_rank,
                    eps=config.rms_norm_eps,
                    rngs=rngs,
                    dtype=dtype,
                    param_dtype=param_dtype,
                ),
            )
            setattr(
                self,
                self.projection_mapping["mla_q_b_proj"],
                ColumnParallelLinear(
                    config.q_lora_rank,
                    config.num_attention_heads * self.q_head_dim,
                    rngs=rngs,
                    use_bias=False,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    kernel_init=jax.nn.initializers.normal(config.initializer_range),
                    precision=precision,
                ),
            )

        # KV compression projection
        setattr(
            self,
            self.projection_mapping["mla_kv_a_proj_with_mqa"],
            ColumnParallelLinear(
                config.hidden_size,
                config.kv_lora_rank + config.qk_rope_head_dim,
                rngs=rngs,
                use_bias=config.attention_bias,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=jax.nn.initializers.normal(config.initializer_range),
                precision=precision,
            ),
        )
        setattr(
            self,
            self.projection_mapping["mla_kv_a_layernorm"],
            RMSNorm(
                config.kv_lora_rank,
                eps=config.rms_norm_eps,
                rngs=rngs,
                dtype=dtype,
                param_dtype=param_dtype,
            ),
        )
        setattr(
            self,
            self.projection_mapping["mla_kv_b_proj"],
            ColumnParallelLinear(
                config.kv_lora_rank,
                config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim),
                rngs=rngs,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=jax.nn.initializers.normal(config.initializer_range),
                precision=precision,
            ),
        )

        setattr(
            self,
            self.projection_mapping["output_projection"],
            RowParallelLinear(
                config.num_attention_heads * self.v_head_dim,
                config.hidden_size,
                rngs=rngs,
                use_bias=config.attention_bias,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=jax.nn.initializers.normal(config.initializer_range),
                precision=precision,
            ),
        )

        self.rotary = self._create_rotary(config, dtype)
        self.attention_performer = self._create_attention_performer(config, rngs)

    def _create_attention_performer(self, config, rngs):
        """Create attention performer module.

        Override for custom attention dropout or softmax scale.
        """
        softmax_scale = self.q_head_dim**-0.5
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                softmax_scale = softmax_scale * mscale * mscale
        return FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=float(softmax_scale),
            dropout_prob=getattr(config, "attention_dropout", 0.0),
        )


class DeepseekV3DecoderLayer(nn.Module):
    """Single DeepSeek V3 transformer block with MLA attention and optional MoE MLP."""

    def __init__(
        self,
        config: DeepseekV3Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        attn_block = DeepseekV3Attention
        mlp_block = DeepseekV3MLP
        mlp_moe_block = DeepseekV3MoE

        attn_block, mlp_block, mlp_moe_block = auto_remat(
            attn_block,
            mlp_block,
            mlp_moe_block,
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

        self.mlp = (
            mlp_moe_block(
                config=config,
                layer_idx=layer_idx,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else mlp_block(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: Array,
        mask_info: MaskInfo,
        position_ids: Array,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: tuple[Array, Array] | None = None,
    ) -> DecoderLayerOutput:
        """
        Forward pass of the module block.

        Args:
            hidden_states (Array): Input hidden states.
            frequencies (tp.Tuple[Array, Array]): Cosine and sine components for rotary embeddings.
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

        # Self Attention
        attn_out = self.self_attn(
            hidden_states,
            mask_info,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            output_attentions,
            frequencies,
        )
        hidden_states = attn_out.attention_output
        hidden_states = checkpoint_name(residual + hidden_states, "residual")

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        feed_forward_hidden_states = self.mlp(hidden_states)
        router_logits = None
        if isinstance(feed_forward_hidden_states, tuple):
            feed_forward_hidden_states, router_logits = feed_forward_hidden_states
        hidden_states = checkpoint_name(residual + feed_forward_hidden_states, "residual")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        hidden_states = checkpoint_name(hidden_states, "layer_output")
        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_out.attention_weight,
            cache_view=attn_out.cache_view,
            router_logits=router_logits,
        )


@register_module(TaskType.BASE_MODULE, DeepseekV3Config, model_type="deepseek_v3")
class DeepseekV3Model(EasyDeLBaseModule):
    """Full DeepSeek V3 decoder-only transformer composed of MLA blocks and MoE feed-forward layers."""

    def __init__(
        self,
        config: DeepseekV3Config,
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
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.layers = [
            DeepseekV3DecoderLayer(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                layer_idx=i,
                rngs=rngs,
            )
            for i in range(self.config.num_hidden_layers)
        ]
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    @functools.cached_property
    def frequencies(self):
        """Compute RoPE frequencies using config's get_basic_frequencies method."""
        return self.config.get_basic_frequencies(
            head_size=self.config.qk_rope_head_dim,
            rotary_dim=self.config.qk_rope_head_dim,
            base=self.config.rope_theta,
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
    ) -> BaseModelOutput:
        """
        Forward pass through the Deepseekv3 module.

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
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = checkpoint_name(self.embed_tokens(input_ids.astype("i4")), "embeddings")
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

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            output = layer(
                hidden_states=hidden_states,
                frequencies=self.frequencies,
                mask_info=mask_info,
                position_ids=position_ids,
                output_attentions=output_attentions,
                mode=mode,
                cache_view=past_key_values.views[idx],
                cache_metadata=cache_metadata,
            )
            hidden_states = output.hidden_states

            if output_attentions:
                all_attentions += (output.attention_weight,)

            if output_router_logits and hasattr(output, "router_logits") and output.router_logits is not None:
                all_router_logits += (output.router_logits,)

            past_key_values[idx] = output.cache_view

        hidden_states = self.norm(hidden_states)
        hidden_states = checkpoint_name(hidden_states, "model_output")

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


@register_module(TaskType.CAUSAL_LM, DeepseekV3Config, model_type="deepseek_v3")
class DeepseekV3ForCausalLM(BaseCausalLMModule[DeepseekV3Model, DeepseekV3Config]):
    """
    DeepseekV3 model with a language modeling head for causal language modeling tasks.

    This model extends the base DeepseekV3Model by adding a linear language modeling head
    on top of the transformer model. It incorporates Mixture of Experts (MoE) architecture
    and is designed for generative tasks and text generation.
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "deepseek_v3"
    _config_class = DeepseekV3Config

    def __init__(
        self,
        config: DeepseekV3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize the DeepseekV3ForCausalLM model.

        Args:
            config (DeepseekV3Config): The model configuration.
            dtype (jnp.dtype, optional): The data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype, optional): The data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike, optional): The precision to use for matrix multiplication.
                Defaults to None.
            rngs (nn.Rngs): The random number generators.
        """
        super().__init__(
            config=config,
            base_model_class=DeepseekV3Model,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
            router_aux_loss_coef=getattr(config, "router_aux_loss_coef", None),
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
        output_router_logits: bool | None = None,
    ) -> MoeCausalLMOutput:
        """
        Forward pass of the causal language model.

        Args:
            input_ids (Optional[Array], optional): Token IDs to process. Defaults to None.
            inputs_embeds (Optional[Array], optional): Pre-computed input embeddings. Defaults to None.
            attention_mask (Optional[Array], optional): Mask to avoid attention on padding tokens. Defaults to None.
            position_ids (Optional[Array], optional): Position IDs. Defaults to None.
            output_attentions (Optional[bool], optional): Whether to output attention weights. Defaults to None.
            output_hidden_states (Optional[bool], optional): Whether to output hidden states. Defaults to None.
            output_router_logits (Optional[bool], optional): Whether to output router logits. Defaults to None.
            past_key_values (Optional[TransformerCache | RaggedPagesCache], optional): Cached key/values.
                Defaults to None.
            cache_metadata (Optional[TransformerMetadata | RaggedPagesMetadata], optional): Cache metadata.
                Defaults to None.

        Returns:
                MoeCausalLMOutput: The model outputs with router logits and aux loss.
        """
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
            aux_loss_fn=self._compute_aux_loss,
        )

    def _compute_aux_loss(self, outputs, attention_mask):
        """Compute auxiliary loss for load balancing."""
        if outputs.router_logits is None or len(outputs.router_logits) == 0:
            return None

        all_router_logits = jnp.stack(outputs.router_logits, axis=0)

        aux_loss = auxiliary_load_balancing_loss_func(
            gate_logits=all_router_logits,
            num_experts=self.config.n_routed_experts,
            top_k=self.config.num_experts_per_tok,
            attention_mask=attention_mask,
        )
        return aux_loss + (aux_loss * self.config.router_aux_loss_coef)

    def create_transformer_cache_config(self, batch_size: int, max_length: int):
        """Create cache metadata for MLA attention.

        MLA uses different dimensions for keys and values:
        - Keys: num_attention_heads x q_head_dim (qk_nope_head_dim + qk_rope_head_dim)
        - Values: num_attention_heads x v_head_dim

        Args:
            batch_size: Batch size for the cache
            max_length: Maximum sequence length
            pad_token_id: Padding token ID (optional)

        Returns:
            TransformerCacheConfig configured for MLA
        """
        from easydel.layers.caching import TransformerCacheConfig

        config = self.config

        # MLA dimensions
        q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        v_head_dim = config.v_head_dim

        return TransformerCacheConfig.create(
            num_hidden_layers=config.num_hidden_layers,
            batch_size=batch_size,
            sequence_length=max_length,
            num_heads=config.num_attention_heads,
            key_heads=config.num_attention_heads,
            value_heads=config.num_attention_heads,
            key_dim=q_head_dim,
            value_dim=v_head_dim,
        )

    def create_ragged_page_cache_config(
        self,
        max_length: int,
        *,
        page_size: int = 128,
        hbm_utilization: float = 0.9,
        dtype: jnp.dtype | None = None,
    ):
        """Create paged cache metadata for MLA attention.

        MLA uses different dimensions for keys and values:
        - Keys: num_attention_heads x q_head_dim (qk_nope_head_dim + qk_rope_head_dim)
        - Values: num_attention_heads x v_head_dim

        Args:
            hbm_utilization: Target HBM utilization (0.0 to 1.0)
            page_size: Number of tokens per page
            max_model_length: Maximum model sequence length

        Returns:
            RaggedPagesCacheConfig configured for MLA
        """
        from easydel.layers.attention import AttentionMechanisms
        from easydel.layers.caching import RaggedPagesCacheConfig

        config = self.config
        text_config = config.get_text_config()

        # MLA dimensions
        q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        v_head_dim = config.v_head_dim

        match text_config.attn_mechanism:
            case AttentionMechanisms.RAGGED_PAGE_ATTENTION_V3:
                version = "v3"
            case AttentionMechanisms.RAGGED_PAGE_ATTENTION_V2:
                version = "v2"
            case _:
                version = "v3"

        return RaggedPagesCacheConfig.create(
            mesh=self.mesh,
            partition_manager=text_config.partition_manager,
            kvdtype=text_config.kvdtype,
            max_model_length=max_length,
            num_hidden_layers=config.num_hidden_layers,
            num_kv_heads=config.num_attention_heads,
            kv_head_dim_size=q_head_dim,
            k_headdim=q_head_dim,
            v_headdim=v_head_dim,
            hbm_utilization=hbm_utilization,
            page_size=page_size,
            version=version,
        )
