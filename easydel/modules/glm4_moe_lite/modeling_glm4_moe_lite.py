# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
from eformer.common_types import ColumnWise, Replicated
from eformer.escale import apply_logical_sharding
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from flax import nnx as nn
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from easydel.caching import (
    HybridCache,
    OperationsMetadata,
    RaggedPagesCache,
    RaggedPagesCacheView,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import AttentionLayerOutput, DecoderLayerOutput, MoeModelOutput
from easydel.infra.utils import ACT2FN, auto_remat
from easydel.layers import (
    BaseMoeModule,
    ColumnParallelLinear,
    ColumnParallelMoELinear,
    Embed,
    MoeFusedHooks,
    MoeLoadBalancingStrategy,
    MoeRoutingStrategy,
    RMSNorm,
    RowParallelLinear,
    RowParallelMoELinear,
)
from easydel.layers.attention import FlexibleAttentionModule, UnifiedAttention
from easydel.layers.rotary import yarn_get_mscale
from easydel.modules._base import BaseCausalLMModule

from .glm4_moe_lite_configuration import Glm4MoeLiteConfig


class Glm4MoeLiteMLP(nn.Module):
    """Dense MLP block for GLM-4-MoE-Lite layers."""

    def __init__(
        self,
        config: Glm4MoeLiteConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.hidden_size = hidden_size or config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        column = partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        row = partial(
            RowParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.gate_proj = column(self.hidden_size, self.intermediate_size)
        self.up_proj = column(self.hidden_size, self.intermediate_size)
        self.down_proj = row(self.intermediate_size, self.hidden_size)
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> Array:
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


class Glm4MoeLiteMLPStack(nn.Module):
    """MoE expert MLP stack for GLM-4-MoE-Lite."""

    reform_param: typing.ClassVar = {
        "gate_up_proj$": {
            "splits": [
                # HF layout: [E, 2M, H] -> ED layout: [E, H, M]
                {"name": "gate_proj.kernel", "spliter": lambda x: x[:, : x.shape[1] // 2, :].swapaxes(-1, -2)},
                {"name": "up_proj.kernel", "spliter": lambda x: x[:, x.shape[1] // 2 :, :].swapaxes(-1, -2)},
            ],
            # Reverse after ED kernels are converted to torch layout [E, M, H].
            "inverse_spliter": lambda torch, gate, up: torch.cat((gate, up), dim=1),
        },
        "down_proj$": {
            "splits": [
                # HF layout: [E, H, M] -> ED layout: [E, M, H]
                {"name": "down_proj.kernel", "spliter": lambda x: x.swapaxes(-1, -2)},
            ],
            "inverse_spliter": lambda x: x,
        },
    }

    def __init__(
        self,
        config: Glm4MoeLiteConfig,
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
        self.gate_proj = ColumnParallelMoELinear(
            num_experts=config.n_routed_experts,
            in_features=config.hidden_size,
            out_features=config.moe_intermediate_size,
            rngs=rngs,
            kernel_init=nn.initializers.normal(),
            use_bias=False,
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.up_proj = ColumnParallelMoELinear(
            num_experts=config.n_routed_experts,
            in_features=config.hidden_size,
            out_features=config.moe_intermediate_size,
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
            in_features=config.moe_intermediate_size,
            out_features=config.hidden_size,
            rngs=rngs,
            kernel_init=nn.initializers.normal(),
            use_bias=False,
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(
        self,
        hidden_states: Array,
        group_sizes: Array,
        sorted_experts: Array | None = None,
    ) -> Array:
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return apply_logical_sharding(
            checkpoint_name(
                self.down_proj(
                    self.act_fn(
                        checkpoint_name(self.gate_proj(hidden_states, group_sizes, sorted_experts), name="mlp_gate")
                    )
                    * checkpoint_name(self.up_proj(hidden_states, group_sizes, sorted_experts), name="mlp_up"),
                    group_sizes,
                    sorted_experts,
                ),
                name="mlp_down",
            ),
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )


class Glm4MoeLiteTopKRouter(nn.Module):
    """Router module for GLM-4-MoE-Lite grouped top-k gating."""

    def __init__(
        self,
        config: Glm4MoeLiteConfig,
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
        self.n_routed_experts = config.n_routed_experts
        self.kernel = nn.Param(
            jax.nn.initializers.normal(config.initializer_range)(
                rngs.param(),
                (config.hidden_size, self.n_routed_experts),
                param_dtype,
            )
        )
        self.e_score_correction_bias = nn.Param(jnp.zeros((self.n_routed_experts,), dtype=jnp.float32))

    def craft_sharding(self, *, partition_manager=None, **_kwargs) -> dict[str, object]:
        kernel_spec = Replicated if self.config.use_expert_tensor_mode else ColumnWise
        return {"kernel": kernel_spec, "e_score_correction_bias": Replicated}

    def __call__(self, hidden_states: Float[Array, "tokens hidden_dim"]) -> Array:
        hidden_states = hidden_states.reshape(-1, self.config.hidden_size)
        return checkpoint_name(
            jnp.matmul(hidden_states.astype(jnp.float32), self.kernel.value.astype(jnp.float32)),
            "moe_router_logits",
        )


class Glm4MoeLiteMoE(BaseMoeModule):
    """Mixture-of-experts feed-forward block for GLM-4-MoE-Lite."""

    def __init__(
        self,
        config: Glm4MoeLiteConfig,
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
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.group_topk_k = min(2, (config.n_routed_experts or 0) // max(config.n_group, 1))

        self.experts = Glm4MoeLiteMLPStack(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.gate = Glm4MoeLiteTopKRouter(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.shared_experts = (
            Glm4MoeLiteMLP(
                config=config,
                intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            if config.n_shared_experts
            else None
        )
        self.moe_hooks = MoeFusedHooks(
            normalize_gate_logits=lambda x: x,
            select_hook=partial(
                self._select_experts_static,
                n_routed_experts=self.n_routed_experts,
                n_group=self.n_group,
                topk_group=self.topk_group,
                group_topk_k=self.group_topk_k,
                norm_topk_prob=self.norm_topk_prob,
                routed_scaling_factor=self.routed_scaling_factor,
            ),
        )

    @staticmethod
    def _select_experts_static(
        gate_logits: Array,
        pre_bias_logits: Array | None,
        k: int,
        *,
        n_routed_experts: int,
        n_group: int,
        topk_group: int,
        group_topk_k: int,
        norm_topk_prob: bool,
        routed_scaling_factor: float,
    ) -> tuple[Array, Array]:
        del pre_bias_logits
        scores = jax.nn.sigmoid(gate_logits.astype(jnp.float32))
        scores_for_choice = scores
        batch_size = scores_for_choice.shape[0]
        group_size = n_routed_experts // n_group
        group_scores = scores_for_choice.reshape(batch_size, n_group, group_size)
        top2_per_group = jax.lax.top_k(group_scores, k=group_topk_k)[0]
        group_scores_sum = jnp.sum(top2_per_group, axis=-1)
        group_k = min(topk_group, n_group)
        group_idx = jax.lax.top_k(group_scores_sum, k=group_k)[1]
        group_mask = jax.nn.one_hot(group_idx, n_group, dtype=scores_for_choice.dtype)
        group_mask = jnp.sum(group_mask, axis=1)
        scores_mask = jnp.repeat(group_mask, group_size, axis=1)
        scores_for_choice = jnp.where(scores_mask > 0, scores_for_choice, 0.0)
        _, topk_indices = jax.lax.top_k(scores_for_choice, k=k)
        topk_weights = jnp.take_along_axis(scores, topk_indices, axis=-1)
        if norm_topk_prob:
            denominator = jnp.sum(topk_weights, axis=-1, keepdims=True) + 1e-20
            topk_weights = topk_weights / denominator
        topk_weights = topk_weights * routed_scaling_factor
        return topk_weights, topk_indices

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
        if self.shared_experts is not None:
            out = out + self.shared_experts(hidden_states)
        return checkpoint_name(out, "moe_expert_output"), checkpoint_name(router_logits, "moe_router_logits")


class Glm4MoeLiteAttention(UnifiedAttention):
    """Multi-head Latent Attention for GLM-4-MoE-Lite."""

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
        config: Glm4MoeLiteConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        self.config = config
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank

        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="mla",
            causal=True,
            use_mla_lora=config.q_lora_rank is not None and config.q_lora_rank > 0,
        )
        self.head_dim = self.v_head_dim

    def define_network(
        self,
        config: Glm4MoeLiteConfig,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        precision: jax.lax.Precision,
        rngs: nn.Rngs,
    ):
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
        softmax_scale = self.q_head_dim**-0.5
        rope_scaling = getattr(self.config, "rope_scaling", None)
        if rope_scaling is not None:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = rope_scaling.get("factor", 1.0)
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                softmax_scale = softmax_scale * mscale * mscale
        return FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=softmax_scale,
            dropout_prob=getattr(config, "attention_dropout", 0.0),
        )

    @staticmethod
    def _apply_rope_interleaved(x: Array, cos: Array, sin: Array) -> Array:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return jnp.stack((o1, o2), axis=-1).reshape(x.shape)

    def forward_mla(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
        alibi: Float[Array, "batch_or_1 heads qseq_len_or_1 kvseq_len_or_1"] | None = None,
    ):
        bsz, q_len, _ = hidden_states.shape

        if not self.use_mla_lora:
            q = checkpoint_name(self.mla_q_proj(hidden_states), name="attn_query")
        else:
            q = checkpoint_name(
                self.mla_q_b_proj(
                    self.mla_q_a_layernorm(checkpoint_name(self.mla_q_a_proj(hidden_states), name="attn_query_a"))
                ),
                name="attn_query",
            )

        q = q.reshape(bsz, q_len, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = q[..., : self.qk_nope_head_dim], q[..., self.qk_nope_head_dim :]

        compressed_kv = self.mla_kv_a_proj_with_mqa(hidden_states)
        k_pe = compressed_kv[..., self.kv_lora_rank :]
        compressed_kv = compressed_kv[..., : self.kv_lora_rank]

        k_pe = k_pe.reshape(bsz, q_len, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)

        kv = (
            self.mla_kv_b_proj(self.mla_kv_a_layernorm(compressed_kv))
            .reshape(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(0, 2, 1, 3)
        )
        k_nope = kv[..., : self.qk_nope_head_dim]
        value_states = kv[..., self.qk_nope_head_dim : self.qk_nope_head_dim + self.v_head_dim]

        if frequencies is not None:
            freq_array = frequencies.value if hasattr(frequencies, "value") else frequencies
            freqs = freq_array[position_ids]
            cos, sin = jnp.split(freqs, 2, -1)
            cos = cos[:, None, :, :].astype(q_pe.dtype)
            sin = sin[:, None, :, :].astype(q_pe.dtype)
            if self.config.rope_interleave:
                q_pe = self._apply_rope_interleaved(q_pe, cos, sin)
                k_pe = self._apply_rope_interleaved(k_pe, cos, sin)
            else:
                q1, q2 = jnp.split(q_pe, 2, axis=-1)
                k1, k2 = jnp.split(k_pe, 2, axis=-1)
                q_pe = jnp.concatenate([q1 * cos - q2 * sin, q2 * cos + q1 * sin], axis=-1)
                k_pe = jnp.concatenate([k1 * cos - k2 * sin, k2 * cos + k1 * sin], axis=-1)

        query_states = jnp.zeros((bsz, self.num_heads, q_len, self.q_head_dim), q_pe.dtype)
        query_states = query_states.at[..., : self.qk_nope_head_dim].set(q_nope)
        query_states = query_states.at[..., self.qk_nope_head_dim :].set(q_pe)

        key_states = jnp.zeros((bsz, self.num_heads, q_len, self.q_head_dim), k_pe.dtype)
        key_states = key_states.at[..., : self.qk_nope_head_dim].set(k_nope)
        key_states = key_states.at[..., self.qk_nope_head_dim :].set(k_pe)

        query_states = query_states.transpose(0, 2, 1, 3)
        key_states = key_states.transpose(0, 2, 1, 3)
        value_states = value_states.transpose(0, 2, 1, 3)

        causal_for_kernel = self.causal
        if mask_info is not None and getattr(mask_info, "_causal_baked", False):
            causal_for_kernel = False

        sliding_window_for_kernel = self.sliding_window
        if mask_info is not None and getattr(mask_info, "sliding_window_baked_in", False):
            sliding_window_for_kernel = None

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
        )

        softmax_aux = getattr(self, "sinks", getattr(self, "softmax_aux", None))
        softmax_aux = getattr(softmax_aux, "value", softmax_aux)

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
            causal=causal_for_kernel,
            sliding_window=sliding_window_for_kernel,
            softmax_aux=softmax_aux,
        )

        attn_output = self.shard_attention_prod(self._merge_heads(attentions.attention_outputs))
        attn_output = checkpoint_name(self.output_projection(attn_output), name="attn_output")

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class Glm4MoeLiteDecoderLayer(nn.Module):
    """Single decoder layer for GLM-4-MoE-Lite."""

    def __init__(
        self,
        config: Glm4MoeLiteConfig,
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
        self.rngs = rngs
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        attn_block = Glm4MoeLiteAttention
        mlp_block = Glm4MoeLiteMLP
        mlp_moe_block = Glm4MoeLiteMoE

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

        if (
            config.mlp_layer_types is not None
            and config.mlp_layer_types[layer_idx] == "sparse"
            and config.n_routed_experts is not None
            and config.num_experts_per_tok is not None
        ):
            self.mlp = mlp_moe_block(
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
        hidden_states = residual + attn_outputs.attention_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        router_logits = None
        if isinstance(feed_forward_hidden_states, tuple):
            feed_forward_hidden_states, router_logits = feed_forward_hidden_states
        hidden_states = residual + feed_forward_hidden_states
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


@register_module(TaskType.BASE_MODULE, config=Glm4MoeLiteConfig, model_type="glm4_moe_lite")
class Glm4MoeLiteModel(EasyDeLBaseModule):
    """Base GLM-4-MoE-Lite model."""

    def __init__(
        self,
        config: Glm4MoeLiteConfig,
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
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs

        self.embed_tokens = Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.layers = nn.List(
            [
                Glm4MoeLiteDecoderLayer(
                    config=config,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    precision=precision,
                    layer_idx=i,
                    rngs=rngs,
                )
                for i in range(self.config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    @functools.cached_property
    def frequencies(self):
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
    ) -> MoeModelOutput:
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

        if sequence_length > self.config.max_position_embeddings:
            raise ValueError(
                f"Maximum Position Embedding Reached: {sequence_length} > {self.config.max_position_embeddings}."
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


@register_module(TaskType.CAUSAL_LM, config=Glm4MoeLiteConfig, model_type="glm4_moe_lite")
class Glm4MoeLiteForCausalLM(BaseCausalLMModule[Glm4MoeLiteModel, Glm4MoeLiteConfig]):  # type: ignore
    """GLM-4-MoE-Lite model with causal LM head."""

    _task_type = TaskType.CAUSAL_LM
    _model_type = "glm4_moe_lite"
    _config_class = Glm4MoeLiteConfig

    def __init__(
        self,
        config: Glm4MoeLiteConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(
            config=config,
            base_model_class=Glm4MoeLiteModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
            router_aux_loss_coef=None,
        )

    def create_ragged_page_cache_config(
        self,
        max_length: int,
        *,
        page_size: int = 128,
        hbm_utilization: float = 0.9,
        dtype: jnp.dtype | None = None,
    ):
        """Create paged cache configuration for MLA attention.

        GLM4-MoE-Lite uses MLA, where the runtime attention head width is
        ``qk_nope_head_dim + qk_rope_head_dim`` (not ``config.head_dim``).
        """
        from easydel.caching import RaggedPagesCacheConfig
        from easydel.layers.attention import AttentionMechanisms

        text_config = self.config.get_text_config()

        q_head_dim = self.config.qk_nope_head_dim + self.config.qk_rope_head_dim
        v_head_dim = self.config.v_head_dim

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
            kvdtype=text_config.kvdtype if dtype is None else dtype,
            max_model_length=max_length,
            num_hidden_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_attention_heads,
            kv_head_dim_size=q_head_dim,
            k_headdim=q_head_dim,
            v_headdim=v_head_dim,
            hbm_utilization=hbm_utilization,
            page_size=page_size,
            version=version,
        )


__all__ = ["Glm4MoeLiteForCausalLM", "Glm4MoeLiteModel"]
