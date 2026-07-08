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
"""Spectrax implementation of Mistral4.

Mistral4 is Mistral AI's DeepSeek-V3-shaped architecture: Multi-head Latent
Attention with low-rank Q/KV projections and split nope/rope head dims, plus
a DeepseekV3-style MoE block (stacked routed experts and one shared expert).
Deltas versus DeepSeek-V3, all faithful to
``transformers.models.mistral4.modeling_mistral4``:

- **Router**: routing scores are a plain softmax over the router logits —
  no sigmoid scoring and no ``e_score_correction_bias``. Group selection
  still keeps the ``topk_group`` groups with the largest top-2 mass out of
  ``n_group`` groups.
- **Attention temperature**: after RoPE the query is scaled by the Llama-4
  temperature ``1 + beta * log(1 + floor(position /
  original_max_position_embeddings))`` with ``beta =
  rope_parameters["llama_4_scaling_beta"]``.
- **Softmax scale**: plain ``qk_head_dim ** -0.5`` with *no* YaRN-mscale
  adjustment (DeepSeek-V3 multiplies by ``mscale**2``; Mistral4 does not).
- **RoPE pairing**: ``config.rope_interleave`` (default ``True``) selects
  the interleaved (GPT-J even/odd) channel pairing on the rope slice.

Building blocks:

- :class:`Mistral4MLP` — dense SwiGLU FFN (dense layers + shared expert).
- :class:`Mistral4TopkRouter` — softmax router emitting full per-expert
  scores.
- :class:`Mistral4MLPMoE` — expert-parallel SwiGLU FFN (stacked experts).
- :class:`Mistral4MoE` — router + routed/shared experts wired together.
- :class:`Mistral4Attention` — MLA module with the Mistral4 deltas.
- :class:`Mistral4DecoderLayer` — single decoder layer (dense or MoE).

Public model classes (registered with the factory):

- :class:`Mistral4Model` — base decoder.
- :class:`Mistral4ForCausalLM` — causal LM head.
- :class:`Mistral4ForSequenceClassification` — sequence classification head.
"""

import functools
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
    MLARaggedPagesCacheView,
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
    AttentionLayerOutput,
    BaseModelOutput,
    DecoderLayerOutput,
    MoeCausalLMOutput,
    MoeModelOutput,
)
from easydel.infra.utils import ACT2FN, ArrayParam, auto_remat, blockwise_ffn
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
    dense_gate_up_layout,
    moe_down_projection_reform_param,
    moe_fused_gate_up_reform_param,
    split_fused_gate_up_projection,
)
from easydel.layers.attention import FlexibleAttentionModule, UnifiedAttention
from easydel.modules._base import BaseCausalLMModule, BaseSequenceClassificationModule

from .mistral4_configuration import Mistral4Config


class Mistral4MLP(spx.Module):
    """SwiGLU feed-forward used by Mistral4 dense layers and shared experts.

    Computes ``down_proj(silu(gate_proj(x)) * up_proj(x))``. Used in two
    places: as the FFN of the first ``config.first_k_dense_replace`` layers
    (which run as dense transformer blocks), and as the *shared expert*
    branch inside :class:`Mistral4MoE` when ``config.n_shared_experts`` is
    set (the shared-expert intermediate width is
    ``config.moe_intermediate_size * n_shared_experts``).
    """

    def __init__(
        self,
        config: Mistral4Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the Mistral4 MLP block.

        Args:
            config (Mistral4Config): Model configuration with MLP parameters.
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
        self.gate_up_proj = ColumnParallelLinear(
            self.hidden_size,
            (self.intermediate_size, self.intermediate_size),
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
            layout=dense_gate_up_layout(self.intermediate_size),
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    @property
    def reform_param(self):
        """Checkpoint reform rules for the fused ``gate_up_proj`` projection.

        Returns:
            dict: Mapping consumed by the checkpoint loader to split the fused
                gate/up kernel back into per-projection slices on load.
        """
        return self.gate_up_proj.build_reform_param("gate_up_proj", config=self.config)

    def forward(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        """Apply the SwiGLU feedforward transformation.

        Args:
            hidden_states: Input tensor ``[batch, seq_len, hidden_dim]``.

        Returns:
            Transformed hidden states ``[batch, seq_len, hidden_dim]``.
        """
        if hidden_states.ndim == 3:  # if not in moe infer
            hidden_states = apply_logical_sharding(
                hidden_states,
                dynamic_axes=common_types.HiddenStateSharding,
                partition_manager=self.config.runtime_sharding_resolver,
            )
        gate_up = checkpoint_name(self.gate_up_proj(hidden_states), "mlp_gate_up")
        gate_raw, up = split_fused_gate_up_projection(gate_up, config=self.config)
        gate = checkpoint_name(self.act_fn(gate_raw), "mlp_gate")
        hidden_states = checkpoint_name(self.down_proj(gate * up), "mlp_down")
        if hidden_states.ndim == 3:  # if not in moe infer
            hidden_states = apply_logical_sharding(
                hidden_states,
                dynamic_axes=common_types.HiddenStateSharding,
                partition_manager=self.config.runtime_sharding_resolver,
            )
        return checkpoint_name(hidden_states, "mlp_output")


class Mistral4TopkRouter(spx.Module):
    """Mistral4 router: plain softmax scores over the router logits.

    Unlike DeepSeek-V3's biased-sigmoid ``noaux_tc`` router, Mistral4 scores
    experts with a single softmax over the router logits — there is no
    ``e_score_correction_bias`` and no sigmoid. Group restriction and top-k
    selection happen in :meth:`Mistral4MoE._select_experts_static` (wired as
    the MoE ``select_hook``), which consumes the full per-expert softmax
    scores this module returns.
    """

    def __init__(
        self,
        config: Mistral4Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the MoE gating module.

        Args:
            config (Mistral4Config): Model configuration with MoE routing parameters.
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
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.gating_dim = config.hidden_size
        kernel = jax.nn.initializers.normal(config.initializer_range)(
            rngs.param,
            (self.gating_dim, self.n_routed_experts),
            param_dtype,
        )
        self.weight = ArrayParam.bound(
            shape=kernel.shape,
            dtype=param_dtype,
            init_method="normal",
            key=rngs.param,
            value=kernel,
        )

    def forward(self, hidden_states: Float[Array, "tokens hidden_dim"]) -> Array:
        """Compute full-expert softmax scores for ``moe_call`` to route on.

        Args:
            hidden_states (Array): Input tensor of shape ``[batch * seq_len, hidden_dim]``.

        Returns:
            Array: Per-expert softmax scores ``[batch * seq_len, n_routed_experts]``.
        """
        logits = jnp.dot(
            hidden_states.astype(jnp.float32),
            self.weight.value.astype(jnp.float32),
            precision=self.precision,
        )
        return jax.nn.softmax(logits, axis=-1)


class Mistral4MLPMoE(spx.Module):
    """Expert-parallel SwiGLU FFN used inside Mistral4 MoE layers.

    Each call evaluates ``w_down(silu(w_gate(x)) * w_up(x))`` independently
    per expert via ``ColumnParallelMoELinear`` / ``RowParallelMoELinear``,
    so the expert dimension is sharded across the EP/TP mesh axes. Tokens
    have already been grouped by :class:`Mistral4MoE` into per-expert ragged
    segments; this module never sees the raw ``(batch, seq)`` layout. The
    intermediate width is ``config.moe_intermediate_size``.
    """

    def __init__(
        self,
        config: Mistral4Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the Mistral4 MoE expert stack.

        Args:
            config (Mistral4Config): Model configuration with MoE MLP parameters.
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
        self.gate_up_proj = ColumnParallelMoELinear(
            num_experts=config.n_routed_experts,
            in_features=hs,
            out_features=2 * imz,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
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
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            partition_manager=config.runtime_sharding_resolver,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            rngs=rngs,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    @property
    def reform_param(self):
        """Checkpoint reform rules for the stacked expert kernels.

        HF Mistral4 checkpoints store *pre-fused* stacked expert tensors:
        ``experts.gate_up_proj`` with shape ``[E, 2 * moe_intermediate,
        hidden]`` and ``experts.down_proj`` with shape ``[E, hidden,
        moe_intermediate]``. Both are transposed into EasyDeL's grouped
        matmul layout on load (and back on export); the gate/up halves are
        additionally TP-interleaved.

        Returns:
            dict: Mapping consumed by the checkpoint loader for the fused
                ``gate_up_proj`` and stacked ``down_proj`` kernels.
        """
        return {
            **moe_fused_gate_up_reform_param(config=self.config),
            **moe_down_projection_reform_param(),
        }

    def forward(
        self,
        hidden_states: Array,
        group_sizes: Array,
        sorted_experts: Array | None = None,
    ):
        """Apply the SwiGLU feedforward transformation through MoE experts.

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

        gate_up = checkpoint_name(self.gate_up_proj(hidden_states, group_sizes, sorted_experts), "mlp_gate_up")
        gate, up = split_fused_gate_up_projection(gate_up, config=self.config)
        gate = checkpoint_name(self.act_fn(gate), "mlp_gate")
        down = checkpoint_name(self.down_proj(gate * up, group_sizes, sorted_experts), "mlp_down")
        return checkpoint_name(
            apply_logical_sharding(
                down,
                dynamic_axes=common_types.HiddenStateSharding,
                partition_manager=self.config.runtime_sharding_resolver,
            ),
            "mlp_output",
        )


class Mistral4MoE(BaseMoeModule):
    """Mistral4 MoE layer: softmax grouped top-k routing + routed/shared experts.

    Composes the routing and expert paths into one block:

    * **Router** (:class:`Mistral4TopkRouter`) emits full per-expert softmax
      scores; expert selection happens in the ``select_hook``:
      per-group score = sum of the top-2 softmax scores in the group, the
      ``topk_group`` best groups are kept, then ``num_experts_per_tok``
      experts are chosen among them; weights are the gathered softmax
      scores, renormalised when ``norm_topk_prob`` and scaled by
      ``routed_scaling_factor``.
    * **Routed experts** (:class:`Mistral4MLPMoE`) run the SwiGLU FFN per
      expert, scattered through ``BaseMoeModule.moe_call``.
    * **Shared expert** (:class:`Mistral4MLP`, optional) processes every
      token unconditionally with intermediate width
      ``moe_intermediate_size * n_shared_experts``; its output is summed
      onto the routed-expert output before exit.
    """

    def __init__(
        self,
        config: Mistral4Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the Mistral4 MoE layer.

        Args:
            config (Mistral4Config): Model configuration with MoE parameters.
            layer_idx (int): Index of this layer in the model.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (str | jax.lax.Precision | None, optional): Numerical precision.
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
        self.experts = Mistral4MLPMoE(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            intermediate_size=config.moe_intermediate_size,
            rngs=rngs,
        )
        self.gate = Mistral4TopkRouter(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        # Route on the gate's full per-expert softmax scores: identity-normalize (the gate already emits
        # probabilities) and run Mistral4's grouped top-k in the select_hook, which returns weights AND
        # expert indices (mirrors the DeepSeek-V3 / GLM-MoE-DSA wiring).
        self.moe_hooks = self.moe_hooks.replace(
            normalize_gate_logits=lambda x: x,
            select_hook=functools.partial(
                self._select_experts_static,
                n_group=config.n_group,
                topk_group=config.topk_group,
                n_routed_experts=config.n_routed_experts,
                norm_topk_prob=config.norm_topk_prob,
                routed_scaling_factor=config.routed_scaling_factor,
            ),
        )
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = Mistral4MLP(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                intermediate_size=intermediate_size,
                rngs=rngs,
            )

    @staticmethod
    def _select_experts_static(
        gate_logits: Array,
        pre_bias_logits: Array | None,
        k: int,
        *,
        n_group: int,
        topk_group: int,
        n_routed_experts: int,
        norm_topk_prob: bool,
        routed_scaling_factor: float,
    ) -> tuple[Array, Array]:
        """Mistral4 grouped top-k over the gate's full-expert softmax scores.

        ``gate_logits`` is ``[tokens, n_routed_experts]`` softmax scores.
        Restricts selection to the ``topk_group`` groups with the largest
        top-2 mass, picks ``k`` experts within them, renormalizes when
        ``norm_topk_prob`` (denominator ``sum + 1e-20``), then scales by
        ``routed_scaling_factor`` — returning BOTH the weights and the
        expert indices. Mirrors
        ``transformers.Mistral4MoE.route_tokens_to_experts`` exactly except
        that scores enter pre-computed (the gate applies the softmax).

        Args:
            gate_logits: Full per-expert softmax scores ``[tokens, n_routed_experts]``.
            pre_bias_logits: Unused (kept for select-hook signature parity).
            k: Number of experts to select per token.
            n_group: Number of expert groups.
            topk_group: Number of groups to keep per token.
            n_routed_experts: Total routed expert count.
            norm_topk_prob: Whether to renormalise the selected weights.
            routed_scaling_factor: Multiplier applied to the final weights.

        Returns:
            tuple[Array, Array]: ``(topk_weights, topk_indices)`` of shape
            ``[tokens, k]`` each.
        """
        scores = gate_logits.astype(jnp.float32)
        token_count = scores.shape[0]
        group_scores = scores.reshape(token_count, n_group, n_routed_experts // n_group)
        group_scores = jnp.sum(jax.lax.top_k(group_scores, k=2)[0], axis=-1)
        group_idx = jax.lax.top_k(group_scores, k=topk_group)[1]
        group_mask = jnp.zeros_like(group_scores)
        row_idx = jnp.arange(token_count)[:, None]
        group_mask = group_mask.at[row_idx, group_idx].set(1.0)
        score_mask = jnp.repeat(group_mask[:, :, None], n_routed_experts // n_group, axis=2).reshape(token_count, -1)
        masked_scores = jnp.where(score_mask > 0, scores, 0.0)
        topk_weight, topk_idx = jax.lax.top_k(masked_scores, k=k)
        if norm_topk_prob:
            topk_weight = topk_weight / (jnp.sum(topk_weight, axis=-1, keepdims=True) + 1e-20)
        topk_weight = topk_weight * routed_scaling_factor
        return topk_weight, topk_idx

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
            gate_up_kernel=self.experts.gate_up_proj.weight.value,
            wd_kernel=self.experts.down_proj.weight.value,
            act_fn=self.experts.act_fn,
        )
        if self.config.n_shared_experts is not None:
            out = out + self.shared_experts(hidden_states)
        return checkpoint_name(out, "moe_expert_output"), checkpoint_name(router_logits, "moe_router_logits")


class Mistral4Attention(UnifiedAttention):
    """Multi-head Latent Attention for Mistral4.

    Identical wiring to DeepSeek-V3 MLA (optional q LoRA chain,
    ``kv_a_proj_with_mqa`` latent + shared single-head RoPE key,
    ``kv_b_proj`` decompression into per-head NoPE-K and V), with three
    Mistral4-specific behaviors:

    * **Interleaved RoPE** (``config.rope_interleave``, default ``True``):
      the rope slice rotates even/odd channel pairs (GPT-J pairing) instead
      of NeoX half-split pairing.
    * **Llama-4 attention temperature**: queries are multiplied by
      ``1 + beta * log(1 + floor(position / original_max))`` with ``beta``
      and ``original_max`` read from ``config.rope_parameters``
      (``llama_4_scaling_beta`` / ``original_max_position_embeddings``).
      The scale is position-dependent and therefore consistent across
      prefill and decode as long as ``position_ids`` carry the cache
      offset.
    * **Plain softmax scale** ``qk_head_dim ** -0.5`` — Mistral4 does *not*
      apply DeepSeek-V3's YaRN ``mscale**2`` correction.
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
        config: Mistral4Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the Mistral4 MLA attention layer.

        Args:
            config (Mistral4Config): Model configuration with attention parameters.
            layer_idx (int): Index of this layer in the model.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (str | jax.lax.Precision | None, optional): Numerical precision.
                Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        # Set MLA-specific dimensions before calling super().__init__()
        # so they're available in define_network.
        self.config = config
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank

        rope_params = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
        beta = rope_params.get("llama_4_scaling_beta") if isinstance(rope_params, dict) else None
        original_max = (
            rope_params.get("original_max_position_embeddings") if isinstance(rope_params, dict) else None
        )
        self.llama_4_scaling_beta = float(beta) if beta is not None else None
        self.llama_4_original_max_position_embeddings = float(original_max) if original_max is not None else None

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

        # Override head_dim for MLA - use value head dimension for output merging.
        self.head_dim = self.v_head_dim

    def define_network(
        self,
        config: Mistral4Config,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        precision: jax.lax.Precision,
        rngs: spx.Rngs,
    ):
        """Define the MLA-specific network structure.

        Sets up the query projection (with optional LoRA), the key-value
        compression projections with layer normalization, and the output
        projection.

        Args:
            config (Mistral4Config): Model configuration.
            dtype (jnp.dtype): Data type for computation.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.Precision): Numerical precision.
            rngs (spx.Rngs): Random number generator state.
        """
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
        """Create the attention performer with Mistral4's plain softmax scale.

        Unlike DeepSeek-V3, Mistral4 does **not** fold the YaRN
        ``mscale**2`` correction into the softmax scale; the scale is a
        plain ``qk_head_dim ** -0.5``.

        Args:
            config: Model configuration with attention settings.
            rngs: Random number generator state for dropout.

        Returns:
            FlexibleAttentionModule: Configured attention performer.
        """
        return FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=float(self.q_head_dim**-0.5),
            dropout_prob=getattr(config, "attention_dropout", 0.0),
        )

    @staticmethod
    def _apply_rope_interleaved(x: Array, cos: Array, sin: Array) -> Array:
        """Apply RoPE in the interleaved (GPT-J even/odd) layout.

        Args:
            x: Tensor whose last axis carries the rotated channels with
                even/odd interleaving.
            cos: Per-position cosines broadcastable to ``x``.
            sin: Per-position sines broadcastable to ``x``.

        Returns:
            Rotated tensor with the same shape as ``x``.
        """
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return jnp.stack((o1, o2), axis=-1).reshape(x.shape)

    def _llama_4_attn_scale(self, position_ids: Array) -> Array | None:
        """Compute the Llama-4 attention temperature per query position.

        ``scale = 1 + beta * log(1 + floor(position / original_max))`` —
        equal to 1 within the original context window and growing
        logarithmically beyond it. Returns ``None`` when the config's rope
        parameters do not define ``llama_4_scaling_beta`` /
        ``original_max_position_embeddings``.

        Args:
            position_ids: Position indices ``[batch, seq_len]`` (with cache
                offset applied during decode).

        Returns:
            Array | None: Float32 scale of shape ``[batch, seq_len]``.
        """
        if self.llama_4_scaling_beta is None or self.llama_4_original_max_position_embeddings is None:
            return None
        positions = position_ids.astype(jnp.float32)
        return 1.0 + self.llama_4_scaling_beta * jnp.log1p(
            jnp.floor(positions / self.llama_4_original_max_position_embeddings)
        )

    def forward_mla(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type: ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
        alibi: Float[Array, "batch_or_1 heads qseq_len_or_1 kvseq_len_or_1"] | None = None,
    ) -> AttentionLayerOutput:
        """Run the Mistral4 MLA attention forward pass.

        Mirrors :meth:`UnifiedAttention.forward_mla` (both the standard
        decompressed path and the weight-absorbed MLA ragged-pages path)
        with two Mistral4 deltas: RoPE is applied with the interleaved
        pairing when ``config.rope_interleave`` and, after RoPE, queries
        are scaled by the Llama-4 attention temperature.

        Args:
            hidden_states: Residual stream ``[batch, seq_len, hidden_dim]``.
            mask_info: Mask information for attention.
            position_ids: Position indices used for RoPE and the attention
                temperature.
            mode: Runtime mode (train / prefill / decode).
            cache_view: Optional KV cache view; an
                :class:`MLARaggedPagesCacheView` triggers the weight-absorbed
                path.
            cache_metadata: Optional cache metadata companion.
            output_attentions: Whether to return materialized attention
                weights.
            frequencies: Optional precomputed RoPE cos/sin cache.
            alibi: Accepted for signature parity with the other forward
                paths; unused by MLA.

        Returns:
            :class:`AttentionLayerOutput` containing the attention output,
            optional attention weights, and the updated cache view.
        """
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

        # Reshape and transpose: [batch, heads, seq, dim]
        q = q.reshape(bsz, q_len, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = q[..., : self.qk_nope_head_dim], q[..., self.qk_nope_head_dim :]

        compressed_kv = self.mla_kv_a_proj_with_mqa(hidden_states)
        k_pe = compressed_kv[..., self.kv_lora_rank :]
        compressed_kv = compressed_kv[..., : self.kv_lora_rank]

        k_pe = k_pe.reshape(bsz, q_len, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)

        _use_mla_ragged = isinstance(cache_view, MLARaggedPagesCacheView)

        if not _use_mla_ragged:
            kv = (
                self.mla_kv_b_proj(self.mla_kv_a_layernorm(compressed_kv))
                .reshape(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
                .transpose(0, 2, 1, 3)
            )
            k_nope = kv[..., : self.qk_nope_head_dim]
            value_states = kv[..., self.qk_nope_head_dim : self.qk_nope_head_dim + self.v_head_dim]
        else:
            compressed_kv = self.mla_kv_a_layernorm(compressed_kv)
            k_nope = None
            value_states = None

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

        # Llama-4 attention temperature: scale the full query (nope + pe) by a
        # per-position factor. Applying it to both slices here is equivalent to
        # HF's post-concat multiply and keeps the weight-absorbed path exact.
        attn_scale = self._llama_4_attn_scale(position_ids)
        if attn_scale is not None:
            scale = attn_scale[:, None, :, None]
            q_nope = (q_nope * scale).astype(q_nope.dtype)
            q_pe = (q_pe * scale).astype(q_pe.dtype)

        causal_for_kernel = self.causal
        if mask_info is not None and getattr(mask_info, "_causal_baked", False):
            causal_for_kernel = False

        sliding_window_for_kernel = self.sliding_window
        if mask_info is not None and getattr(mask_info, "sliding_window_baked_in", False):
            sliding_window_for_kernel = None

        if _use_mla_ragged:
            # Weight absorption: split kv_b_proj weight into W_k and W_v.
            kv_b_weight = self.mla_kv_b_proj.weight.value
            kv_b_weight = kv_b_weight.reshape(self.kv_lora_rank, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            w_k = kv_b_weight[:, :, : self.qk_nope_head_dim]
            w_v = kv_b_weight[:, :, self.qk_nope_head_dim :]

            q_absorbed = jnp.einsum(
                "bnsd,and->bnsa",
                q_nope.astype(jnp.float32),
                w_k.astype(jnp.float32),
            ).astype(q_nope.dtype)

            mla_queries_nope = q_absorbed.transpose(0, 2, 1, 3)
            mla_queries_pe = q_pe.transpose(0, 2, 1, 3)
            mla_keys_values = compressed_kv
            mla_keys_pe = k_pe[:, 0, :, :]

            mla_softmax_scale = (self.qk_nope_head_dim + self.qk_rope_head_dim) ** -0.5

            key_for_cache = jnp.concatenate(
                [compressed_kv[:, :, None, :], k_pe.transpose(0, 2, 1, 3)],
                axis=-1,
            )
            query_for_concat = mla_queries_nope
            value_for_cache = key_for_cache  # MLA: value pages = key pages

            (
                _,
                _,
                mask_info,
                init_attention_bias,
                cache_view,
                cache_metadata,
            ) = self.concatenate(
                query=query_for_concat,
                key=key_for_cache,
                value=value_for_cache,
                cache_view=cache_view,
                cache_metadata=cache_metadata,
                mask_info=mask_info,
            )

            softmax_aux = getattr(self, "sinks", getattr(self, "softmax_aux", None))
            softmax_aux = getattr(softmax_aux, "value", softmax_aux)

            attentions = self.attention_performer.forward(
                query_states=query_for_concat,
                key_states=key_for_cache,
                value_states=value_for_cache,
                mode=mode,
                bias=None,
                cache_metadata=cache_metadata,
                cache_view=cache_view,
                init_bias=init_attention_bias,
                mask_info=mask_info,
                causal=causal_for_kernel,
                sliding_window=sliding_window_for_kernel,
                softmax_aux=softmax_aux,
                queries_nope=mla_queries_nope,
                queries_pe=mla_queries_pe,
                keys_values=mla_keys_values,
                keys_pe=mla_keys_pe,
                softmax_scale=mla_softmax_scale,
                output_attentions=output_attentions,
            )

            attn_out = attentions.attention_outputs
            attn_out = jnp.einsum(
                "tna,anh->tnh",
                attn_out.astype(jnp.float32),
                w_v.astype(jnp.float32),
            ).astype(attn_out.dtype)

            attn_output = attn_out.reshape(bsz, q_len, -1)
        else:
            query_states = jnp.zeros((bsz, self.num_heads, q_len, self.q_head_dim), q_pe.dtype)
            query_states = query_states.at[..., : self.qk_nope_head_dim].set(q_nope)
            query_states = query_states.at[..., self.qk_nope_head_dim :].set(q_pe)

            key_states = jnp.zeros((bsz, self.num_heads, q_len, self.q_head_dim), k_pe.dtype)
            key_states = key_states.at[..., : self.qk_nope_head_dim].set(k_nope)
            key_states = key_states.at[..., self.qk_nope_head_dim :].set(k_pe)

            query_states = query_states.transpose(0, 2, 1, 3)
            key_states = key_states.transpose(0, 2, 1, 3)
            value_states = value_states.transpose(0, 2, 1, 3)

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
                output_attentions=output_attentions,
            )

            attn_out = attentions.attention_outputs
            attn_output = self._merge_heads(attn_out)

        attn_output = self.shard_attention_prod(attn_output)
        attn_output = checkpoint_name(self.output_projection(attn_output), name="attn_output")
        attn_output = self.shard_attention_prod(attn_output)

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class Mistral4DecoderLayer(spx.Module):
    """One Mistral4 decoder layer (sequential pre-norm MLA + FFN/MoE).

    Standard pre-norm transformer block:

        ``x = x + mla(input_ln(x))``
        ``x = x + ffn(post_attn_ln(x))``

    The FFN swap-out is layer-aware:

    * Layers ``0..first_k_dense_replace - 1`` use a dense
      :class:`Mistral4MLP` with intermediate width ``intermediate_size``.
    * Remaining layers use :class:`Mistral4MoE` (softmax-routed experts +
      optional shared expert) with intermediate width
      ``moe_intermediate_size`` per expert.
    """

    def __init__(
        self,
        config: Mistral4Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize a Mistral4 decoder layer.

        Args:
            config (Mistral4Config): Model configuration.
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

        self.self_attn = Mistral4Attention(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )

        self.mlp = (
            Mistral4MoE(
                config=config,
                layer_idx=layer_idx,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            if (config.n_routed_experts is not None and layer_idx >= config.first_k_dense_replace)
            else Mistral4MLP(
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
        mode: common_types.RUNTIME_MODE_TYPES,  # type: ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: tuple[Array, Array] | None = None,
    ) -> DecoderLayerOutput:
        """Forward pass through the decoder layer.

        Applies the pre-normalization architecture: ``x + attn(norm(x))``
        followed by ``x + mlp(norm(x))`` where mlp may be either a dense MLP
        or a MoE layer depending on layer configuration.

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

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.config.use_scan_mlp and isinstance(self.mlp, Mistral4MLP):
            feed_forward_hidden_states = blockwise_ffn(
                self.mlp,
                hidden_states,
                self.config.scan_mlp_chunk_size,
            )
        else:
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


@register_module(TaskType.BASE_MODULE, Mistral4Config, model_type="mistral4")
class Mistral4Model(EasyDeLBaseModule):
    """Mistral4 backbone (registered as ``BASE_MODULE``).

    Stack: token embedding -> ``num_hidden_layers``
    :class:`Mistral4DecoderLayer` blocks (the first
    ``first_k_dense_replace`` are dense, the rest run MoE) -> final RMSNorm.
    Attention uses MLA with low-rank KV compression and YaRN RoPE, plus the
    Llama-4 attention temperature. Decoder layers wrap with
    :func:`auto_remat` so per-layer activation checkpointing follows
    ``config.gradient_checkpointing``. Forward returns
    :class:`MoeModelOutput` carrying the final hidden state, optional
    all-layer hidden states / attentions, optional router logits (one per
    MoE layer), and the updated KV cache.

    Attributes:
        config (Mistral4Config): Configuration for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.
    """

    def __init__(
        self,
        config: Mistral4Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the Mistral4 base model.

        Args:
            config (Mistral4Config): Model configuration.
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

        with self.assign_layer_stage(0, total_layers=self.config.num_hidden_layers):
            self.embed_tokens = Embed(
                self.config.vocab_size,
                self.config.hidden_size,
                embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

        remat_layer_block = auto_remat(
            Mistral4DecoderLayer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.layers = nn.ModuleList([])
        for i in range(self.config.num_hidden_layers):
            with self.assign_layer_stage(i, total_layers=self.config.num_hidden_layers):
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
        final_layer_idx = max(0, self.config.num_hidden_layers - 1)
        with self.assign_layer_stage(final_layer_idx, total_layers=self.config.num_hidden_layers):
            self.norm = RMSNorm(
                self.config.hidden_size,
                eps=self.config.rms_norm_eps,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

    @functools.cached_property
    def frequencies(self):
        """Compute RoPE frequencies for the rope slice of the head.

        Uses the config's YaRN parameters sized to ``qk_rope_head_dim``.

        Returns:
            tuple[Array, Array]: Cosine and sine frequency components.
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
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
    ) -> BaseModelOutput:
        """Forward pass through the Mistral4 base model.

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
            """Per-layer body for the Mistral4 decoder ``scan``.

            Carry layout: ``(hidden_states, all_hidden_states,
            all_attentions, all_router_logits, idx)``. Records the input
            hidden state when requested, runs one decoder layer at the
            assigned pipeline stage, accumulates attention weights and MoE
            router logits when requested, updates the per-layer KV cache
            slot, and returns the next carry.
            """
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


@register_module(TaskType.CAUSAL_LM, Mistral4Config, model_type="mistral4")
class Mistral4ForCausalLM(BaseCausalLMModule[Mistral4Model, Mistral4Config]):
    """Mistral4 backbone + ``vocab_size`` LM head for autoregressive generation.

    Wraps :class:`Mistral4Model` with an unbiased linear ``lm_head``. Goes
    through ``forward_moe`` so per-MoE-layer router logits are surfaced;
    Mistral4 uses no auxiliary load-balancing loss by default (no
    ``router_aux_loss_coef`` in the reference config), but router logits
    still flow into the output for diagnostics and MoE-aware tooling.

    Attributes:
        config (Mistral4Config): Configuration for the model.
        dtype (jnp.dtype): Data type for computations (default is jnp.bfloat16).
        param_dtype (jnp.dtype): Data type for parameters (default is jnp.bfloat16).
        precision: Precision setting for JAX operations.
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "mistral4"
    _config_class = Mistral4Config

    def __init__(
        self,
        config: Mistral4Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the Mistral4 model for causal language modeling.

        Args:
            config (Mistral4Config): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=Mistral4Model,
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
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
    ) -> MoeCausalLMOutput:
        """Forward pass of the causal language model.

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
        """Compute the optional auxiliary loss for MoE load balancing.

        Mistral4's reference implementation trains without an auxiliary
        router loss; the loss is only computed when the config defines a
        non-``None`` ``router_aux_loss_coef``.

        Args:
            outputs: Model outputs containing router logits.
            attention_mask: Attention mask for valid tokens.

        Returns:
            Auxiliary loss value for load balancing, or None when disabled
            or no router logits are present.
        """
        coef = getattr(self.config, "router_aux_loss_coef", None)
        if coef is None:
            return None
        if outputs.router_logits is None or len(outputs.router_logits) == 0:
            return None

        all_router_logits = jnp.stack(outputs.router_logits, axis=0)

        aux_loss = auxiliary_load_balancing_loss_func(
            gate_logits=all_router_logits,
            num_experts=self.config.n_routed_experts,
            top_k=self.config.num_experts_per_tok,
            attention_mask=attention_mask,
        )
        return aux_loss * coef

    def create_transformer_cache_config(self, batch_size: int, max_length: int, **kwargs):
        """Create the cache configuration for MLA attention.

        MLA uses different dimensions for keys and values:
        - Keys: num_attention_heads x q_head_dim (qk_nope_head_dim + qk_rope_head_dim)
        - Values: num_attention_heads x v_head_dim

        Args:
            batch_size (int): Batch size for the cache.
            max_length (int): Maximum sequence length.
            **kwargs: Unused extra options (signature parity).

        Returns:
            TransformerCacheConfig: Configuration object for the MLA-compatible transformer cache.
        """
        from easydel.caching import TransformerCacheConfig

        config = self.config

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
        max_cache_tokens: int | None = None,
    ):
        """Create the paged cache configuration for MLA attention.

        Args:
            max_length (int): Maximum model sequence length.
            page_size (int, optional): Number of tokens per page. Defaults to 128.
            hbm_utilization (float, optional): Target HBM utilization (0.0 to 1.0).
                Defaults to 0.9.
            dtype (jnp.dtype | None, optional): Data type for cache. ``None``
                defers to the model's ``kvdtype``.
            max_cache_tokens (int | None, optional): Optional hard ceiling on
                total KV-cache tokens; the page count becomes
                ``min(hbm-derived, cap)``. Defaults to None (no cap).

        Returns:
            RaggedPagesCacheConfig: Configuration object for the MLA-compatible paged cache.
        """
        from easydel.caching import MLARaggedPagesCacheConfig, RaggedPagesCacheConfig
        from easydel.layers.attention import AttentionMechanisms

        config = self.config
        text_config = config.get_text_config()

        if dtype is None:
            dtype = text_config.kvdtype

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
                kvdtype=dtype,
                max_model_length=max_length,
                num_hidden_layers=config.num_hidden_layers,
                num_kv_heads=config.num_attention_heads,
                kv_lora_rank=config.kv_lora_rank,
                qk_rope_head_dim=config.qk_rope_head_dim,
                hbm_utilization=hbm_utilization,
                page_size=page_size,
                max_cache_tokens=max_cache_tokens,
            )

        return RaggedPagesCacheConfig.create(
            mesh=self.mesh,
            runtime_sharding_resolver=text_config.runtime_sharding_resolver,
            kvdtype=dtype,
            max_model_length=max_length,
            num_hidden_layers=config.num_hidden_layers,
            num_kv_heads=config.num_attention_heads,
            kv_head_dim_size=q_head_dim,
            k_headdim=q_head_dim,
            v_headdim=v_head_dim,
            hbm_utilization=hbm_utilization,
            page_size=page_size,
            version=version,
            max_cache_tokens=max_cache_tokens,
        )


@register_module(TaskType.SEQUENCE_CLASSIFICATION, config=Mistral4Config, model_type="mistral4")
class Mistral4ForSequenceClassification(BaseSequenceClassificationModule[Mistral4Model, Mistral4Config]):
    """Mistral4 model for sequence classification tasks.

    Extends the base Mistral4 model with a linear classification head over
    the last (non-padding) token's hidden state, matching
    ``transformers.Mistral4ForSequenceClassification``.

    Attributes:
        config (Mistral4Config): Configuration for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.
    """

    _task_type = TaskType.SEQUENCE_CLASSIFICATION
    _model_type = "mistral4"
    _config_class = Mistral4Config

    def __init__(
        self,
        config: Mistral4Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike | None = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the Mistral4 model for sequence classification.

        Args:
            config (Mistral4Config): Model configuration with num_labels for classification.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=Mistral4Model,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            pooling_strategy="last",
            score_head_bias=False,
        )
