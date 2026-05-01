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
"""Spectrax implementation of DeepSeek-V3.

DeepSeek-V3 keeps DeepSeek-V2's Multi-head Latent Attention (low-rank Q/KV
projections, split nope/rope head dims) and pushes the MoE further: 256
routed experts plus a single shared expert, top-8 routing with the
auxiliary-free top-k method (``topk_method="noaux_tc"``) and sigmoid
scoring, optional multi-token prediction, and YaRN-style RoPE for extended
context.

Building blocks:

- :class:`DeepseekV3MLP` — dense SwiGLU FFN used by the dense layers
  (typically the first ``first_k_dense_replace`` layers).
- :class:`DeepseekV3MLPMoE` — expert-parallel SwiGLU FFN.
- :class:`MoEGate` — sigmoid/softmax router with optional group routing,
  ``topk_method="noaux_tc"`` (no auxiliary, learned bias), and sequence-level
  auxiliary loss.
- :class:`DeepseekV3MoE` — router + routed/shared experts wired together.
- :class:`DeepseekV3Attention` — Multi-head Latent Attention (MLA) module.
- :class:`DeepseekV3DecoderLayer` — single decoder layer (dense or MoE).

Public model classes (registered with the factory):

- :class:`DeepseekV3Model` — base decoder.
- :class:`DeepseekV3ForCausalLM` — causal LM head + auxiliary load-balancing
  loss.
"""

import functools
import typing
from functools import partial
from typing import ClassVar

import jax
import jax.numpy as jnp
import spectrax as spx
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int
from spectrax import apply_logical_sharding, common_types, nn

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
from easydel.infra.loss_utils import auxiliary_load_balancing_loss_func
from easydel.infra.modeling_outputs import (
    BaseModelOutput,
    DecoderLayerOutput,
    MoeCausalLMOutput,
    MoeModelOutput,
)
from easydel.infra.utils import ACT2FN, ArrayParam, auto_remat
from easydel.layers import (
    BaseMoeModule,
    ColumnParallelLinear,
    ColumnParallelMoELinear,
    Embed,
    MoeLoadBalancingStrategy,
    MoeRoutingStrategy,
    RMSNorm,
    RowParallelLinear,
    RowParallelMoELinear,
)
from easydel.layers.attention import FlexibleAttentionModule, UnifiedAttention
from easydel.layers.rotary import yarn_get_mscale
from easydel.modules._base import BaseCausalLMModule

from .deepseek_configuration import DeepseekV3Config


class DeepseekV3MLP(spx.Module):
    """SwiGLU feed-forward used by DeepSeek-V3 dense layers and shared experts.

    Computes ``down_proj(silu(gate_proj(x)) * up_proj(x))``. Used in two
    places: as the FFN of the first ``config.first_k_dense_replace`` layers
    (which run as dense transformer blocks), and as the *shared expert*
    branch inside :class:`DeepseekV3MoE` when
    ``config.n_shared_experts`` is set (the shared-expert intermediate width
    is ``config.moe_intermediate_size * n_shared_experts``).
    """

    def __init__(
        self,
        config: DeepseekV3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        hidden_size=None,
        intermediate_size=None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize DeepSeek V3 MLP block.

        Args:
            config (DeepseekV3Config): Model configuration with MLP parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            hidden_size (int | None, optional): Override for hidden size. Defaults to None (uses config).
            intermediate_size (int | None, optional): Override for intermediate size.
                Defaults to None (uses config).
            rngs (spx.Rngs): Random number generator state.
        """
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

    def forward(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        """Apply SwiGLU feedforward transformation.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]

        Returns:
            Transformed hidden states [batch, seq_len, hidden_dim]
        """
        if hidden_states.ndim == 3:  # if not in moe infer
            hidden_states = apply_logical_sharding(
                hidden_states,
                dynamic_axes=common_types.HiddenStateSharding,
                partition_manager=self.config.runtime_sharding_resolver,
            )
        gate = checkpoint_name(self.act_fn(self.gate_proj(hidden_states)), "mlp_gate")
        up = checkpoint_name(self.up_proj(hidden_states), "mlp_up")
        hidden_states = checkpoint_name(self.down_proj(gate * up), "mlp_down")
        if hidden_states.ndim == 3:  # if not in moe infer
            hidden_states = apply_logical_sharding(
                hidden_states,
                dynamic_axes=common_types.HiddenStateSharding,
                partition_manager=self.config.runtime_sharding_resolver,
            )
        return checkpoint_name(hidden_states, "mlp_output")


class MoEGate(spx.Module):
    """DeepSeek-V3 router: sigmoid scoring + grouped ``noaux_tc`` top-k selection.

    DeepSeek-V3 replaces the standard softmax router with a *biased sigmoid*
    routing scheme:

    1. ``hidden -> n_routed_experts`` linear (no bias) produces per-expert
       scores; a learnable ``e_score_correction_bias`` is added to balance
       experts without an auxiliary loss.
    2. Experts are grouped into ``n_group`` groups; only the top
       ``topk_group`` groups (ranked by the sum of their best two members)
       are eligible for selection on each token.
    3. Within the eligible groups, the router picks ``num_experts_per_tok``
       experts (8 by default in V3) and re-normalises their sigmoid scores
       to sum to 1 (scaled by ``routed_scaling_factor``).

    The selection function ``noaux_tc`` keeps the routing differentiable in
    the value path while load balance is handled purely through the
    correction bias — hence the name "no auxiliary loss" balancing.
    """

    def __init__(
        self,
        config: DeepseekV3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize MoE gating module.

        Args:
            config (DeepseekV3Config): Model configuration with MoE routing parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
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
        kernel = jax.nn.initializers.kaiming_uniform()(
            rngs.param,
            (self.gating_dim, self.n_routed_experts),
            param_dtype,
        )

        self.weight = ArrayParam.bound(
            shape=kernel.shape,
            dtype=param_dtype,
            init_method="kaiming_uniform",
            key=rngs.param,
            value=kernel,
        )
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = ArrayParam.bound(
                shape=(self.n_routed_experts,),
                dtype=param_dtype,
                init_method="zeros",
                key=rngs.parameters,
            )

    def forward(self, hidden_states):
        """Compute expert routing weights for input tokens.

        Implements sigmoid scoring with noaux_tc group-based top-k expert selection.

        Args:
            hidden_states (Array): Input tensor of shape [batch * seq_len, hidden_dim].

        Returns:
            Array: Top-k expert weights for each token [batch * seq_len, top_k],
                scaled by routed_scaling_factor.
        """
        squ, _ = hidden_states.shape
        logits = jnp.dot(
            hidden_states.astype(jnp.float32),
            self.weight.value.astype(jnp.float32),
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

            assert self.n_routed_experts is not None
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


class DeepseekV3MLPMoE(spx.Module):
    """Expert-parallel SwiGLU FFN used inside DeepSeek-V3 MoE layers.

    Each call evaluates ``w_down(silu(w_gate(x)) * w_up(x))`` independently
    per expert via ``ColumnParallelMoELinear`` / ``RowParallelMoELinear``,
    so the expert dimension is sharded across the EP/TP mesh axes. Tokens
    have already been grouped by :class:`DeepseekV3MoE` into per-expert
    ragged segments (``group_sizes`` + sorted-expert indices); this module
    never sees the raw ``(batch, seq)`` layout. The intermediate width is
    ``config.moe_intermediate_size`` (smaller than ``intermediate_size`` so
    each expert is cheaper than a dense layer).
    """

    reform_param: typing.ClassVar = {
        "gate_up_proj$": {
            "splits": [
                {"name": "gate_proj.weight", "spliter": lambda x: x[..., : x.shape[-1] // 2]},
                {"name": "up_proj.weight", "spliter": lambda x: x[..., x.shape[-1] // 2 :]},
            ],
            "inverse_spliter": lambda torch, gate, up: torch.stack((gate, up), dim=-1).flatten(-2),
        },
        "down_proj$": {
            "splits": [
                {"name": "down_proj.weight", "spliter": lambda x: x},
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
        rngs: spx.Rngs,
    ):
        """Initialize DeepSeek V3 MoE MLP block.

        Args:
            config (DeepseekV3Config): Model configuration with MoE MLP parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (str | jax.lax.Precision | None, optional): Numerical precision for operations.
                Defaults to None.
            hidden_size (int | None, optional): Override for hidden size. Defaults to None (uses config).
            intermediate_size (int | None, optional): Override for intermediate size.
                Defaults to None (uses config).
            rngs (spx.Rngs): Random number generator state.
        """
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
            kernel_init=jax.nn.initializers.normal(),
            partition_manager=config.runtime_sharding_resolver,
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
            kernel_init=jax.nn.initializers.normal(),
            partition_manager=config.runtime_sharding_resolver,
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
            kernel_init=jax.nn.initializers.normal(),
            partition_manager=config.runtime_sharding_resolver,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            rngs=rngs,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: Array,
        group_sizes: Array,
        sorted_experts: Array | None = None,
    ):
        """Apply SwiGLU feedforward transformation through MoE experts.

        Args:
            hidden_states (Array): Input tensor containing routed tokens.
            group_sizes (Array): Size of each expert group for batched computation.
            sorted_experts (Array | None, optional): Sorted expert indices. Defaults to None.

        Returns:
            Array: Transformed hidden states after expert processing.
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )

        gate = checkpoint_name(self.act_fn(self.gate_proj(hidden_states, group_sizes, sorted_experts)), "mlp_gate")
        up = checkpoint_name(self.up_proj(hidden_states, group_sizes, sorted_experts), "mlp_up")
        down = checkpoint_name(self.down_proj(gate * up, group_sizes, sorted_experts), "mlp_down")
        return checkpoint_name(
            apply_logical_sharding(
                down,
                dynamic_axes=common_types.HiddenStateSharding,
                partition_manager=self.config.runtime_sharding_resolver,
            ),
            "mlp_output",
        )


class DeepseekV3MoE(BaseMoeModule):
    """DeepSeek-V3 MoE layer: ``noaux_tc`` routing + routed experts + shared expert.

    Composes the routing and expert paths into one block:

    * **Router** (:class:`MoEGate`) emits ``num_experts_per_tok`` expert ids
      per token and matching gating weights, using DeepSeek-V3's
      sigmoid-scored ``noaux_tc`` group-top-k strategy.
    * **Routed experts** (:class:`DeepseekV3MLPMoE`) run the SwiGLU FFN per
      expert, scattered through ``BaseMoeModule.moe_call`` with
      ``MoeLoadBalancingStrategy.NO_AUX_LOSS``.
    * **Shared expert** (:class:`DeepseekV3MLP`, optional) processes every
      token unconditionally with intermediate width
      ``moe_intermediate_size * n_shared_experts``; its output is summed
      onto the routed-expert output before exit. This is the
      "shared + routed" trick that lets DeepSeek keep per-token compute
      stable even at low expert activation rates.
    """

    def __init__(
        self,
        config: DeepseekV3Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize DeepSeek V3 MoE layer.

        Args:
            config (DeepseekV3Config): Model configuration with MoE parameters.
            layer_idx (int): Index of this layer in the model.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (str | jax.lax.Precision | None, optional): Numerical precision for operations.
                Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
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

    def forward(self, hidden_states: Array):
        """Process tokens through MoE experts with routing.

        Args:
            hidden_states (Array): Input tensor of shape (batch_size, sequence_length, hidden_dim).

        Returns:
            tuple[Array, Array]: Tuple containing:
                - Expert output tensor of shape (batch_size, sequence_length, hidden_dim)
                - Router logits for auxiliary loss computation
        """
        out, router_logits = self.moe_call(
            hidden_state=hidden_states,
            gate_layer=self.gate,
            expert_layer=self.experts,
            wi_kernel=self.experts.gate_proj.weight.value,
            wu_kernel=self.experts.up_proj.weight.value,
            wd_kernel=self.experts.down_proj.weight.value,
            act_fn=self.experts.act_fn,
        )
        if self.config.n_shared_experts is not None:
            out = out + self.shared_experts(hidden_states)
        return checkpoint_name(out, "moe_expert_output"), checkpoint_name(router_logits, "moe_router_logits")


class DeepseekV3Attention(UnifiedAttention):
    """Multi-head Latent Attention (MLA) for DeepSeek-V3.

    MLA splits each query/key into two parts — a "no-position" (NoPE) part of
    width ``qk_nope_head_dim`` and a "position" (RoPE) part of width
    ``qk_rope_head_dim`` — and projects keys/values through a low-rank latent
    of width ``kv_lora_rank``. The KV cache then stores the *latent* vectors
    (size ``kv_lora_rank + qk_rope_head_dim`` per token) instead of full
    per-head keys/values, shrinking the cache to a small fraction of GQA's
    footprint.

    Concretely:

    * **Query path.** When ``config.q_lora_rank`` is set, queries are routed
      through a LoRA-style bottleneck:
      ``hidden -> q_a_proj (rank=q_lora_rank) -> q_a_layernorm ->
      q_b_proj``; otherwise a single ``q_proj`` produces the full
      ``num_heads * (qk_nope_head_dim + qk_rope_head_dim)`` projection.
    * **KV path.** ``kv_a_proj_with_mqa`` produces both the latent
      (``kv_lora_rank``) and the shared RoPE-K (``qk_rope_head_dim``) in one
      matmul; the latent is RMSNorm'd by ``kv_a_layernorm`` and then
      ``kv_b_proj`` expands it to per-head NoPE-K and value channels of
      total width ``num_heads * (qk_nope_head_dim + v_head_dim)``.
    * **Softmax scale.** Uses ``q_head_dim ** -0.5``; when YARN scaling is
      configured (``rope_scaling.mscale_all_dim``) the scale is multiplied
      by ``mscale ** 2`` for stable extrapolation.
    * **Output.** Heads are merged at the value head dim ``v_head_dim`` and
      projected back to ``hidden_size`` by a row-parallel ``o_proj``.
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
        rngs: spx.Rngs,
    ):
        """Initialize DeepSeek V3 MLA attention layer.

        Args:
            config (DeepseekV3Config): Model configuration with attention parameters.
            layer_idx (int): Index of this layer in the model.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (str | jax.lax.Precision | None, optional): Numerical precision.
                Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
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
        rngs: spx.Rngs,
    ):
        """Define MLA-specific network structure.

        Sets up query projection (with optional LoRA), key-value compression
        projections with layer normalization, and output projection layers.

        Args:
            config (DeepseekV3Config): Model configuration.
            dtype (jnp.dtype): Data type for computation.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.Precision): Numerical precision.
            rngs (spx.Rngs): Random number generator state.
        """

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
        """Create attention performer module with custom softmax scale.

        Configures the attention performer with MLA-specific softmax scaling,
        including optional YARN-based scaling for extended context lengths.

        Args:
            config: Model configuration with attention settings.
            rngs: Random number generator state for dropout.

        Returns:
            FlexibleAttentionModule: Configured attention performer.
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


class DeepseekV3DecoderLayer(spx.Module):
    """One DeepSeek-V3 decoder layer (sequential pre-norm MLA + FFN/MoE).

    Standard pre-norm transformer block:

        ``x = x + mla(input_ln(x))``
        ``x = x + ffn(post_attn_ln(x))``

    The FFN swap-out is layer-aware:

    * Layers ``0..first_k_dense_replace - 1`` use a dense
      :class:`DeepseekV3MLP` with intermediate width ``intermediate_size``.
    * Remaining layers use :class:`DeepseekV3MoE` (sigmoid-routed experts +
      optional shared expert) with intermediate width
      ``moe_intermediate_size`` per expert.

    This layout — dense early layers, MoE for the bulk — keeps representation
    quality at the bottom of the stack while reaping MoE's compute savings
    deeper in the model.
    """

    def __init__(
        self,
        config: DeepseekV3Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize DeepSeek V3 decoder layer.

        Args:
            config (DeepseekV3Config): Model configuration.
            layer_idx (int): Index of this layer in the model.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (str | jax.lax.Precision | None, optional): Numerical precision.
                Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        mlp_block = DeepseekV3MLP
        mlp_moe_block = DeepseekV3MoE

        self.self_attn = DeepseekV3Attention(
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

    def forward(
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
        """Forward pass through the decoder layer.

        Applies pre-normalization architecture: x + attn(norm(x)) followed by x + mlp(norm(x))
        where mlp may be either a dense MLP or MoE layer depending on layer configuration.

        Args:
            hidden_states (Array): Input tensor of shape (batch_size, sequence_length, hidden_dim).
            mask_info (MaskInfo): Attention mask information including causal masks.
            position_ids (Array): Position indices for tokens, shape (batch_size, sequence_length).
            mode (RUNTIME_MODE_TYPES): Runtime mode (train, decode, etc.) for optimization.
            cache_view (TransformerCacheView | RaggedPagesCacheView | None, optional): Cache view.
                Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Cache metadata. Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            frequencies (tuple[Array, Array] | None, optional): Precomputed RoPE frequencies.
                Defaults to None.

        Returns:
            DecoderLayerOutput: Contains hidden states, attention weights, cache view, and router logits.
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
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
            partition_manager=self.config.runtime_sharding_resolver,
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
    """DeepSeek-V3 backbone (registered as ``BASE_MODULE``).

    Stack: token embedding -> ``num_hidden_layers`` :class:`DeepseekV3DecoderLayer`
    blocks (the first ``first_k_dense_replace`` are dense, the rest run MoE)
    -> final RMSNorm. Attention uses MLA with low-rank KV compression
    (``kv_lora_rank``, ``qk_nope/rope_head_dim``, ``v_head_dim``) and YARN
    RoPE scaling for long context. Decoder layers wrap with :func:`auto_remat`
    so per-layer activation checkpointing follows
    ``config.gradient_checkpointing``. Forward returns
    :class:`MoeModelOutput` carrying the final hidden state, optional
    all-layer hidden states / attentions, optional router logits (one per
    MoE layer; ``None`` for the dense prefix), and the updated KV cache.

    Attributes:
        config (DeepseekV3Config): Configuration for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.
    """

    def __init__(
        self,
        config: DeepseekV3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize DeepSeek V3 base model.

        Args:
            config (DeepseekV3Config): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.embed_tokens = Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        remat_layer_block = auto_remat(
            DeepseekV3DecoderLayer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.layers = nn.ModuleList([])
        for i in range(self.config.num_hidden_layers):
            with spx.assign_stage(total=self.config.num_hidden_layers, current=i):
                self.layers.append(
                    remat_layer_block(
                        config=config,
                        dtype=dtype,
                        param_dtype=param_dtype,
                        precision=precision,
                        layer_idx=i,
                        rngs=rngs,
                    )
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
        """Compute RoPE frequencies for rotary position embeddings.

        Uses YaRN scaling if configured, otherwise computes standard RoPE frequencies
        for the rope head dimension.

        Returns:
            tuple[Array, Array]: Cosine and sine frequency components for rotary embeddings.
        """
        return self.config.get_basic_frequencies(
            head_size=self.config.qk_rope_head_dim,
            rotary_dim=self.config.qk_rope_head_dim,
            base=self.config.rope_theta,
        )

    def forward(
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
        """Forward pass through the DeepSeek V3 base model.

        Processes input tokens through embedding, all decoder layers with MLA attention
        and MoE/dense FFN, and final normalization.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch_size, sequence_length).
                Must be provided if inputs_embeds is None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings of shape
                (batch_size, sequence_length, hidden_size). Defaults to None.
            attention_mask (Array | None, optional): Boolean mask to avoid attention on padding tokens,
                shape (batch_size, sequence_length). Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information for attention operations.
                Defaults to None.
            position_ids (Array | None, optional): Position indices for each token, shape
                (batch_size, sequence_length). Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode (train/decode) for optimizations.
                Auto-detected if None. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cache with precomputed key-value states for generation. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache management. Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention weights from all layers.
                Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return hidden states from all layers.
                Defaults to None.
            output_router_logits (bool | None, optional): Whether to return router logits from MoE layers.
                Defaults to None.

        Returns:
            MoeModelOutput: Contains last_hidden_state, optional all hidden_states, optional attentions,
                updated past_key_values, and optional router_logits.

        Raises:
            ValueError: If both input_ids and inputs_embeds are provided or both are None.
            AssertionError: If sequence_length exceeds max_position_embeddings.
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
            partition_manager=self.config.runtime_sharding_resolver,
        )

        def _layer_loop(layer, carry):
            hidden_states, all_hidden_states, all_attentions, all_router_logits, idx = carry
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            with self._layer_stage_context(idx, layers=self.layers):
                output = layer(
                    hidden_states=hidden_states,
                    frequencies=self.frequencies,
                    mask_info=mask_info,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                    mode=mode,
                    cache_view=self._layer_cache_view_at(None, idx, enabled=True, cache=past_key_values),
                    cache_metadata=cache_metadata,
                )
            hidden_states = self._mark_layer_stage_boundary(output.hidden_states, idx, layers=self.layers)

            if output_attentions:
                all_attentions += (output.attention_weight,)

            if output_router_logits and hasattr(output, "router_logits") and output.router_logits is not None:
                all_router_logits += (output.router_logits,)

            self._layer_cache_view_update(None, idx, output.cache_view, enabled=True, cache=past_key_values)

            return hidden_states, all_hidden_states, all_attentions, all_router_logits, idx + 1

        hidden_states, all_hidden_states, all_attentions, all_router_logits, _ = self.layers.scan(
            _layer_loop,
            (hidden_states, all_hidden_states, all_attentions, all_router_logits, 0),
            trace=True,
        )
        hidden_states = self.norm(hidden_states)
        hidden_states = checkpoint_name(hidden_states, "model_output")

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return MoeModelOutput(  # pyright: ignore[reportReturnType]
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
    """DeepSeek-V3 backbone + ``vocab_size`` LM head for autoregressive generation.

    Wraps :class:`DeepseekV3Model` with an unbiased linear ``lm_head``. Goes
    through ``forward_moe`` so per-MoE-layer router logits are surfaced;
    however DeepSeek-V3's ``noaux_tc`` routing means the auxiliary
    load-balancing loss is *not* added to the objective (load balance is
    maintained purely through the learnable
    ``e_score_correction_bias``). Routing weights still flow into the
    output for diagnostics and downstream MoE-aware tooling.

    Attributes:
        config (DeepseekV3Config): Configuration for the model.
        dtype (jnp.dtype): Data type for computations (default is jnp.bfloat16).
        param_dtype (jnp.dtype): Data type for parameters (default is jnp.bfloat16).
        precision: Precision setting for JAX operations.
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
        rngs: spx.Rngs,
    ):
        """Initialize DeepSeek V3 model for causal language modeling.

        Args:
            config (DeepseekV3Config): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
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

    def forward(
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
        """Forward pass of the causal language model.

        Processes input through the base model and applies the language modeling head
        to produce next-token logits with MoE auxiliary loss support.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch_size, sequence_length).
                Must be provided if inputs_embeds is None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings of shape
                (batch_size, sequence_length, hidden_size). Defaults to None.
            attention_mask (Array | None, optional): Boolean mask to avoid attention on padding tokens,
                shape (batch_size, sequence_length). Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information for attention operations.
                Defaults to None.
            position_ids (Array | None, optional): Position indices for each token, shape
                (batch_size, sequence_length). Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode (train/decode) for optimizations.
                Auto-detected if None. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cache with precomputed key-value states for generation. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache management. Defaults to None.
            apply_lm_head (bool, optional): Whether to apply the language model head.
                Defaults to True.
            output_attentions (bool | None, optional): Whether to return attention weights from all layers.
                Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return hidden states from all layers.
                Defaults to None.
            output_router_logits (bool | None, optional): Whether to return router logits from MoE layers.
                Defaults to None.

        Returns:
            MoeCausalLMOutput: Contains logits, optional hidden_states, optional attentions,
                updated past_key_values, router_logits, and auxiliary loss.
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
        """Compute auxiliary loss for MoE load balancing.

        Args:
            outputs: Model outputs containing router logits.
            attention_mask: Attention mask for valid tokens.

        Returns:
            Auxiliary loss value for load balancing, or None if no router logits.
        """
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

    def create_transformer_cache_config(self, batch_size: int, max_length: int, **kwargs):
        """Create cache configuration for MLA attention.

        MLA uses different dimensions for keys and values:
        - Keys: num_attention_heads x q_head_dim (qk_nope_head_dim + qk_rope_head_dim)
        - Values: num_attention_heads x v_head_dim

        Args:
            batch_size (int): Batch size for the cache.
            max_length (int): Maximum sequence length.

        Returns:
            TransformerCacheConfig: Configuration object for MLA-compatible transformer cache.
        """
        from easydel.caching import TransformerCacheConfig

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
        """Create paged cache configuration for MLA attention.

        MLA uses different dimensions for keys and values:
        - Keys: num_attention_heads x q_head_dim (qk_nope_head_dim + qk_rope_head_dim)
        - Values: num_attention_heads x v_head_dim

        Args:
            max_length (int): Maximum model sequence length.
            page_size (int, optional): Number of tokens per page. Defaults to 128.
            hbm_utilization (float, optional): Target HBM utilization (0.0 to 1.0).
                Defaults to 0.9.
            dtype (jnp.dtype | None, optional): Data type for cache. Defaults to None.

        Returns:
            RaggedPagesCacheConfig: Configuration object for MLA-compatible paged cache.
        """
        from easydel.caching import MLARaggedPagesCacheConfig, RaggedPagesCacheConfig
        from easydel.layers.attention import AttentionMechanisms

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

        attn_mechanism = getattr(text_config, "attn_mechanism", None)
        if hasattr(attn_mechanism, "value"):
            attn_mechanism = attn_mechanism.value
        is_mla_ragged = str(attn_mechanism) in (
            "multi_latent_ragged_page_attention_v1",
            "multi_latent_ragged_page_attention_v2",
        )
        if is_mla_ragged:
            return MLARaggedPagesCacheConfig.create(
                mesh=self.mesh,
                runtime_sharding_resolver=text_config.runtime_sharding_resolver,
                kvdtype=text_config.kvdtype,
                max_model_length=max_length,
                num_hidden_layers=config.num_hidden_layers,
                num_kv_heads=config.num_attention_heads,
                kv_lora_rank=config.kv_lora_rank,
                qk_rope_head_dim=config.qk_rope_head_dim,
                hbm_utilization=hbm_utilization,
                page_size=page_size,
            )

        return RaggedPagesCacheConfig.create(
            mesh=self.mesh,
            partition_manager=text_config.runtime_sharding_resolver,
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
