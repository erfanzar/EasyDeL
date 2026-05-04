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

"""Spectrax implementation of THUDM's GLM-4-MoE language model.

GLM-4-MoE is a decoder-only transformer with a hybrid dense/sparse layout:
the first ``first_k_dense_replace`` layers use a standard dense MLP and the
rest use a grouped mixture-of-experts block with a small number of always-on
shared experts plus sparsely routed experts.

Architectural traits:
    - Grouped-query attention with optional bias and Q/K normalization.
    - Partial rotary embeddings (``partial_rotary_factor``).
    - Pre-norm RMSNorm decoder layers.
    - Hybrid dense + grouped MoE FFN: per-token top-k routing inside
      ``topk_group`` selected groups out of ``n_group``, plus shared experts.
    - Optional load-balancing auxiliary loss via ``routed_scaling_factor``.

Exports:
    - :class:`Glm4MoeModel`: Backbone returning hidden states.
    - :class:`Glm4MoeForCausalLM`: Decoder LM with optional tied LM head.
    - :class:`Glm4MoeForSequenceClassification`: Pooled classifier head.
"""

import typing
from functools import partial

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
from easydel.infra.modeling_outputs import (
    DecoderLayerOutput,
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
from easydel.layers.attention import UnifiedAttention
from easydel.modules._base import BaseCausalLMModule, BaseSequenceClassificationModule

from .glm4_moe_configuration import Glm4MoeConfig


class Glm4MoeMLP(spx.Module):
    """Dense feed-forward block used in non-MoE GLM-4-MoE layers.

    Implements the standard gated feedforward network with separate gate and up
    projections for dense layers in the GLM-4-MoE hybrid architecture.
    Used in early layers before the MoE transition point.
    """

    def __init__(
        self,
        config: Glm4MoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        intermediate_size: int | None = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize GLM-4 MoE dense MLP block.

        Args:
            config (Glm4MoeConfig): Model configuration with MLP parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            intermediate_size (int | None, optional): Optional MLP intermediate size override.
                If None, uses config.intermediate_size.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
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
        self.gate_proj = column_parallel_linear(config.hidden_size, self.intermediate_size)
        self.up_proj = column_parallel_linear(config.hidden_size, self.intermediate_size)
        self.down_proj = row_parallel_linear(self.intermediate_size, config.hidden_size)
        self.act_fn = ACT2FN[self.config.hidden_act]

    def forward(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> jnp.ndarray:
        """Apply gated feedforward transformation.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]

        Returns:
            Tuple of (transformed hidden states [batch, seq_len, hidden_dim], None).
            Returns None for router_logits compatibility with MoE interface.
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        gate_output = self.act_fn(checkpoint_name(self.gate_proj(hidden_states), name="mlp_gate"))
        up_output = checkpoint_name(self.up_proj(hidden_states), name="mlp_up")
        hidden_states = checkpoint_name(self.down_proj(gate_output * up_output), name="mlp_down")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        return hidden_states, None  # pyright: ignore[reportReturnType]


class Glm4MoeMLPStack(spx.Module):
    """Expert MLP stack for GLM-4-MoE using parallel MoE linear layers.

    Implements the feedforward network for multiple experts using efficient
    batched computation with ColumnParallelMoELinear and RowParallelMoELinear
    layers. Supports expert tensor mode for optimized expert computation.
    """

    reform_param: typing.ClassVar = {
        "gate_up_proj$": {
            "splits": [
                {
                    "name": "gate_proj.weight",
                    "spliter": lambda x: x[:, : x.shape[1] // 2, :].swapaxes(-1, -2),
                },
                {
                    "name": "up_proj.weight",
                    "spliter": lambda x: x[:, x.shape[1] // 2 :, :].swapaxes(-1, -2),
                },
            ],
            "inverse_spliter": lambda torch, gate, up: torch.cat(
                (gate.transpose(-1, -2), up.transpose(-1, -2)),
                dim=1,
            ),
        },
        "down_proj$": {
            "splits": [
                {"name": "down_proj.weight", "spliter": lambda x: x.swapaxes(-1, -2)},
            ],
            "inverse_spliter": lambda x: x.swapaxes(-1, -2),
        },
    }

    def __init__(
        self,
        config: Glm4MoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize GLM-4 MoE expert MLP stack.

        Args:
            config (Glm4MoeConfig): Model configuration with MoE parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.gate_proj = ColumnParallelMoELinear(
            num_experts=config.n_routed_experts,
            in_features=config.hidden_size,
            out_features=config.moe_intermediate_size,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(),
            use_bias=False,
            partition_manager=config.runtime_sharding_resolver,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.down_proj = RowParallelMoELinear(
            num_experts=config.n_routed_experts,
            in_features=config.moe_intermediate_size,
            out_features=config.hidden_size,
            rngs=rngs,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(),
            partition_manager=config.runtime_sharding_resolver,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.up_proj = ColumnParallelMoELinear(
            num_experts=config.n_routed_experts,
            in_features=config.hidden_size,
            out_features=config.moe_intermediate_size,
            rngs=rngs,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(),
            partition_manager=config.runtime_sharding_resolver,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        x: Array,
        group_sizes: Array,
        sorted_experts: Array | None = None,
    ) -> Array:
        """Forward pass through expert MLP stack.

        Args:
            x: Input tensor for expert computation.
            group_sizes: Sizes of each expert group for batched computation.
            sorted_experts: Optional sorted expert indices for routing.

        Returns:
            Expert output tensor after gated feedforward transformation.
        """
        hidden_states = self.act_fn(checkpoint_name(self.gate_proj(x, group_sizes, sorted_experts), name="moe_gate"))
        hidden_states = hidden_states * checkpoint_name(self.up_proj(x, group_sizes, sorted_experts), name="moe_up")
        outputs = checkpoint_name(self.down_proj(hidden_states, group_sizes, sorted_experts), name="moe_expert_output")
        return outputs


class Glm4MoeTopKRouter(spx.Module):
    """Two-stage grouped top-k router with bias-corrected sigmoid scoring.

    GLM-4-MoE adopts the DeepSeek-style "grouped top-k" routing rather than
    flat top-k over all experts. The ``n_routed_experts`` are partitioned into
    ``n_group`` contiguous groups of equal size; routing then proceeds in two
    nested top-k passes:

    1. **Group selection**. For each token, score every group by the sum of
       its top-2 expert scores (within that group), then pick the
       ``topk_group`` highest-scoring groups. All experts outside the selected
       groups are masked to zero.
    2. **Expert selection**. From the surviving (non-masked) experts, take a
       flat top-k of size ``num_experts_per_tok`` to obtain the final
       routing decision.

    Two further details matter at training time. The router emits **raw
    sigmoid** probabilities (not softmax) so that the per-expert "load" is a
    well-defined fraction in ``[0, 1]`` independent of the other experts. A
    learned ``e_score_correction_bias`` of shape ``(n_routed_experts,)`` is
    *added to scores only for selection*; the value used for the convex
    combination of expert outputs is the un-biased sigmoid. This decouples
    routing from output magnitude so the bias can drive load balancing
    without distorting expert outputs. Finally, the selected weights are
    optionally re-normalised (``norm_topk_prob``) and scaled by
    ``routed_scaling_factor`` before consumption.

    The router projection itself is a single fp32 matmul against
    ``self.weight`` of shape ``(n_routed_experts, hidden_size)`` — kept in
    fp32 regardless of the activation dtype because tiny gating logit
    differences can otherwise be lost in bf16.
    """

    def __init__(
        self,
        config: Glm4MoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize GLM-4 MoE Top-K router.

        Args:
            config (Glm4MoeConfig): Model configuration with routing parameters.
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
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = ArrayParam.bound(
            shape=(self.n_routed_experts, config.hidden_size),
            dtype=param_dtype,
            init_method="normal",
            init_kwargs={"stddev": config.initializer_range},
            key=rngs.param,
        )
        self.e_score_correction_bias = ArrayParam.bound(
            shape=(self.n_routed_experts,),
            dtype=jnp.float32,
            init_method="normal",
            init_kwargs={"stddev": 0.0},
            key=rngs.param,
        )

    def get_selected_experts(self, scores):
        """Select top-k experts using grouped routing strategy.

        Performs two-stage selection: first selects top groups based on sum
        of top-2 scores per group, then selects top-k experts from those groups.

        Args:
            scores: Router scores of shape [batch_size, n_routed_experts].

        Returns:
            Selected expert indices of shape [batch_size, top_k].
        """
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

    def forward(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        """Compute pre-activation router logits for input tokens.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]

        Returns:
            Router logits [batch * seq_len, n_routed_experts].
        """
        hidden_states = hidden_states.reshape(-1, self.config.hidden_size)
        return checkpoint_name(
            jnp.matmul(hidden_states.astype(jnp.float32), self.weight.value.astype(jnp.float32)),
            name="moe_router_logits",
        )


class Glm4MoeMoE(BaseMoeModule):
    """GLM-4-MoE sparse feedforward block: routed experts + always-on shared expert.

    Composes three pieces:

    - :class:`Glm4MoeTopKRouter` to compute per-token gate logits.
    - :class:`Glm4MoeMLPStack` holding the ``n_routed_experts`` parameter
      stacks, executed via the fused ``moe_call`` dispatch in
      :class:`BaseMoeModule` (which sorts tokens by assigned expert and
      processes each contiguous expert group with ``grouped_matmul``).
    - :class:`Glm4MoeMLP` shared expert that *every* token always passes
      through; its output is added unconditionally to the routed-expert
      mixture. The shared expert's intermediate size is widened by
      ``n_shared_experts`` to give it capacity comparable to the routed
      experts being combined.

    Token output is therefore ``shared(h) + sum_i w_i * routed_i(h)`` where
    ``i`` ranges over the ``num_experts_per_tok`` selected experts and
    ``w_i`` come from the (optionally re-normalised, ``routed_scaling_factor``
    -scaled) sigmoid scores returned by the grouped top-k router.

    Auxiliary load-balancing and router-z losses are wired into the parent
    ``BaseMoeModule`` via ``router_aux_loss_coef`` / ``router_z_loss_coef``
    if the config provides them; the actual loss tensors come back as a
    side-output of :meth:`forward` (the ``router_logits`` returned to the
    caller) and are aggregated by the trainer.
    """

    def __init__(
        self,
        config: Glm4MoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize GLM-4 MoE layer.

        Args:
            config (Glm4MoeConfig): Model configuration with MoE parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
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
        self.group_topk_k = min(2, config.n_routed_experts // max(config.n_group, 1))

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
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            rngs=rngs,
        )
        self.moe_hooks = self.moe_hooks.replace(
            normalize_gate_logits=lambda x: x,
            select_hook=partial(
                self._select_experts_static,
                n_routed_experts=self.n_routed_experts,
                n_group=self.config.n_group,
                topk_group=self.config.topk_group,
                group_topk_k=self.group_topk_k,
                norm_topk_prob=self.config.norm_topk_prob,
                routed_scaling_factor=self.config.routed_scaling_factor,
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
        group_mask = jax.nn.one_hot(group_idx, n_group, dtype=scores.dtype)
        group_mask = jnp.sum(group_mask, axis=1)
        score_mask = jnp.repeat(group_mask, group_size, axis=1)
        scores_for_choice = jnp.where(score_mask > 0, scores_for_choice, 0.0)
        topk_indices = jax.lax.top_k(scores_for_choice, k=k)[1]
        topk_weights = jnp.take_along_axis(scores, topk_indices, axis=-1)

        if norm_topk_prob:
            denominator = jnp.sum(topk_weights, axis=-1, keepdims=True) + 1e-20
            topk_weights = topk_weights / denominator
        topk_weights = topk_weights * routed_scaling_factor
        return topk_weights, topk_indices

    def forward(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> tuple[Array, Array]:
        """Forward pass through the MoE layer.

        Routes tokens to top-k experts, computes expert outputs, and combines
        with shared expert outputs for final result.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]

        Returns:
            Tuple of (output tensor [batch, seq_len, hidden_dim], router_logits).
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
        shared_output, _ = self.shared_experts(hidden_states)
        out = out + shared_output

        return checkpoint_name(out, "moe_expert_output"), checkpoint_name(router_logits, "moe_router_logits")


class Glm4MoeAttention(UnifiedAttention):
    """Causal GQA attention with RoPE and optional QK-norm for GLM-4-MoE.

    Thin specialisation of :class:`UnifiedAttention` whose only deviation
    from a stock causal attention is the ``use_qk_norm`` toggle: when
    ``config.use_qk_norm`` is true, query and key heads pass through a
    per-head RMSNorm before RoPE is applied, which empirically stabilises
    the gradient through the routing layers above. The dispatch backend,
    head splits, and RoPE are inherited verbatim from the unified base
    class.
    """

    def __init__(
        self,
        config: Glm4MoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize GLM-4 MoE attention layer with grouped-query attention support.

        Args:
            config (Glm4MoeConfig): Model configuration with attention parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the model.
        """
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


class Glm4MoeDecoderLayer(spx.Module):
    """Hybrid dense/MoE decoder block selected by ``layer_idx``.

    Implements one transformer block with the standard pre-norm shape
    ``x + attn(norm(x))`` followed by ``x + ff(norm(x))``, where the
    feedforward branch is one of two flavours depending on this layer's
    position in the stack:

    - **Dense MLP** (:class:`Glm4MoeMLP`) when
      ``layer_idx < config.first_k_dense_replace``. The first
      ``first_k_dense_replace`` layers stay dense to preserve the strong
      low-frequency feature mixing those early layers do; making them
      sparse would starve every expert of training signal.
    - **MoE block** (:class:`Glm4MoeMoE`) for all later layers. These are
      where the parameter count is concentrated and where sparse routing
      meaningfully reduces compute per token.

    Both flavours expose the same ``(out, router_logits)`` return contract
    (the dense path returns ``router_logits=None``) so the surrounding
    decoder loop can collect auxiliary losses uniformly.
    """

    def __init__(
        self,
        config: Glm4MoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize GLM-4 MoE decoder layer.

        Args:
            config (Glm4MoeConfig): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the model. Determines whether
                to use dense MLP (layer_idx < first_k_dense_replace) or MoE.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.layer_idx = layer_idx
        attn_block = Glm4MoeAttention
        mlp_block = Glm4MoeMLP if layer_idx < config.first_k_dense_replace else Glm4MoeMoE
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

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type: ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ):
        """Forward pass through the decoder layer.

        Applies pre-normalization architecture with attention followed by
        feedforward network (either dense MLP or MoE depending on layer index).

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
            output_router_logits (bool, optional): Whether to return router logits. Defaults to False.
            frequencies (Array | None, optional): Precomputed RoPE frequencies. Defaults to None.

        Returns:
            DecoderLayerOutput: Contains hidden states, attention weights, cache view, and router logits.
        """
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
            partition_manager=self.config.runtime_sharding_resolver,
        )
        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
            cache_view=attn_outputs.cache_view,
            router_logits=router_logits,
        )


@register_module(TaskType.BASE_MODULE, config=Glm4MoeConfig, model_type="glm4_moe")
class Glm4MoeModel(EasyDeLBaseModule):
    """GLM-4 MoE (Mixture-of-Experts) model implementation.

    This implements the GLM-4 MoE architecture, a hybrid transformer model that
    combines dense feedforward layers in early layers with mixture-of-experts
    layers in deeper layers. Features include RMSNorm, rotary position embeddings,
    grouped-query attention, and top-k expert routing with shared experts.

    Attributes:
        config (Glm4MoeConfig): Configuration for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.
    """

    def __init__(
        self,
        config: Glm4MoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize GLM-4 MoE base model.

        Args:
            config (Glm4MoeConfig): Model configuration.
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
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            rngs=rngs,
        )
        remat_layer_block = auto_remat(
            Glm4MoeDecoderLayer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.layers = nn.ModuleList([])
        for layer_idx in range(self.config.num_hidden_layers):
            with self.assign_layer_stage(layer_idx, total_layers=self.config.num_hidden_layers):
                self.layers.append(
                    remat_layer_block(
                        config=config,
                        layer_idx=layer_idx,
                        dtype=dtype,
                        param_dtype=param_dtype,
                        precision=precision,
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
            partition_manager=self.config.runtime_sharding_resolver,
        )

        def _layer_loop(block, carry):
            hidden_states, all_hidden_states, all_attentions, all_router_logits, idx = carry
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            with self._layer_stage_context(idx, layers=self.layers):
                layer_outputs = block(
                    hidden_states=hidden_states,
                    mask_info=mask_info,
                    position_ids=position_ids,
                    mode=mode,
                    cache_view=self._layer_cache_view_at(None, idx, enabled=True, cache=past_key_values),
                    cache_metadata=cache_metadata,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    frequencies=self.frequencies,
                )

            hidden_states = self._mark_layer_stage_boundary(layer_outputs.hidden_states, idx, layers=self.layers)

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            if output_router_logits and layer_outputs.router_logits is not None:
                all_router_logits += (layer_outputs.router_logits,)

            self._layer_cache_view_update(None, idx, layer_outputs.cache_view, enabled=True, cache=past_key_values)

            return hidden_states, all_hidden_states, all_attentions, all_router_logits, idx + 1

        hidden_states, all_hidden_states, all_attentions, all_router_logits, _ = self.layers.scan(
            _layer_loop,
            (hidden_states, all_hidden_states, all_attentions, all_router_logits, 0),
            trace=True,
        )
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
        """Returns the encoder part of the model's graph definition.

        Raises:
            NotImplementedError: Decoder-only models don't have an encoder.
        """
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        """Returns the decoder part of the model's graph definition.

        Returns:
            The model itself, as this is a decoder-only architecture.
        """
        return self

    def get_lm_head(self):
        """Returns the language model head of the module.

        Raises:
            NotImplementedError: Base models don't have a Language Model Head.
        """
        raise NotImplementedError("The base model does not have a language model head.")

    def get_embedding(self):
        """Returns the embedding layer of the module.

        Returns:
            The token embedding layer.
        """
        return self.embed_tokens


@register_module(TaskType.CAUSAL_LM, config=Glm4MoeConfig, model_type="glm4_moe")
class Glm4MoeForCausalLM(BaseCausalLMModule[Glm4MoeModel, Glm4MoeConfig]):  # type: ignore
    """GLM-4 MoE model with a language modeling head for causal language modeling tasks.

    This model is a transformer-based language model with mixture-of-experts layers
    and causal attention masks applied to perform autoregressive language generation.

    Attributes:
        config (Glm4MoeConfig): Configuration for the model.
        dtype (jnp.dtype): Data type for computations (default is jnp.bfloat16).
        param_dtype (jnp.dtype): Data type for parameters (default is jnp.bfloat16).
        precision: Precision setting for JAX operations.
    """

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
        rngs: spx.Rngs,
    ):
        """Initialize GLM-4 MoE model for causal language modeling.

        Args:
            config (Glm4MoeConfig): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
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
class Glm4MoeForSequenceClassification(BaseSequenceClassificationModule[Glm4MoeModel, Glm4MoeConfig]):  # type: ignore
    """GLM-4 MoE model for sequence classification tasks.

    This class extends the base GLM-4 MoE model by adding a linear classification head
    to perform sequence classification tasks such as sentiment analysis or text classification.

    Attributes:
        config (Glm4MoeConfig): Configuration for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.
    """

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
        rngs: spx.Rngs,
    ):
        """Initialize GLM-4 MoE model for sequence classification.

        Args:
            config (Glm4MoeConfig): Model configuration with num_labels for classification.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
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
