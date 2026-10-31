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

"""Spectrax implementation of GLM-5-Next (GLM-5.3-Flash).

GLM-5-Next pairs every decoder layer with one of two attention mechanisms
scheduled 3:1 by ``config.layer_types``:

- **KDA linear attention** (``"linear_attention"``) — Kimi-style gated delta
  rule behind a fused depthwise short conv, with a *per-channel* forget gate
  (``g = lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))`` on
  ``(num_heads, head_dim)`` channels), per-head sigmoid write strength, and a
  gated RMSNorm output. Dispatches through
  :class:`~easydel.operations.kernels.KernelDeltaAttnOp` with
  ``per_channel_decay=True``.
- **DSA/MLA sparse attention** (``"deepseek_sparse_attention"``) — DeepSeek
  Multi-head Latent Attention (NoPE: ``qk_rope_head_dim=0``) whose KV pages
  are restricted per query to the ``index_topk`` tokens picked by a
  k-pooled lightning indexer (keys are pooled in groups of ``index_kpool``
  with a softmax-gated learned average, scored with per-head ReLU dot
  products, and expanded back to raw token indices; the incomplete tail pool
  is optionally appended).

Cross-cutting traits:

- **mHC hyper-connections** (``hc_mult=4``): the residual stream is a stack
  of ``hc_mult`` streams ``[B, S, hc, D]``; two
  :class:`Glm5NextHyperConnection` modules per layer collapse/re-expand
  streams through a Sinkhorn-projected doubly-stochastic mixer, and the
  final collapse is an unweighted mean over streams.
- **Grouped sigmoid-top-k MoE** (288 routed + 1 shared experts, top-8) with
  a score-correction bias, and **clamped SwiGLU** (``swiglu_limit=10``)
  everywhere a gated MLP appears.

Exports:
    - :class:`Glm5NextTextModel`: Text backbone returning hidden states.
    - :class:`Glm5NextForCausalLM`: Decoder LM with (optional) tied LM head.
"""

import functools
import typing
from functools import partial
from typing import ClassVar

import jax
import jax.numpy as jnp
import spectrax as spx
from eformer.loggings import get_logger
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int
from spectrax import apply_logical_sharding, common_types, nn

from easydel.caching import (
    HybridCache,
    OperationsMetadata,
    RaggedPagesCache,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerMetadata,
)
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import (
    DecoderLayerOutput,
    MoeModelOutput,
    SparseDecoderLayerOutput,
)
from easydel.infra.utils import ACT2FN, ArrayParam, auto_remat, blockwise_ffn
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
    clamped_swiglu,
    dense_gate_up_layout,
    gated_mlp_forward,
    split_fused_gate_up_projection,
)
from easydel.layers.attention import DeepSeekSparseAttention
from easydel.layers.indexer import (
    IndexerConfig,
    IndexerKind,
    SparseIndexer,
)
from easydel.layers.linear_attention import KernelDeltaAttention, unnest_legacy_kda_forget_gate
from easydel.layers.moe import fp32_router_logits, moe_group_topk_select
from easydel.layers.residual import (
    HyperStreamSharding,
    ManifoldHyperConnection,
    ManifoldHyperConnectionConfig,
    manifold_residual_write,
    mean_hyper_head,
)
from easydel.modules._base import BaseCausalLMModule
from easydel.utils.checkpoint_keys import KEEP, KeyRuleSet, Map

from .glm5_next_configuration import (
    LINEAR_ATTENTION_LAYER_TYPE,
    Glm5NextTextConfig,
)

logger = get_logger(__name__)


class Glm5NextTextMLP(spx.Module):
    """Dense gated MLP with the GLM-5 SwiGLU clamp.

    Computes ``down(silu(clamp(gate, max=limit)) * clamp(up, ±limit))``. The
    ``gate_proj``/``up_proj`` pair is stored fused as ``gate_up_proj`` (same
    layout convention as the other GLM/DeepSeek families) and declared with
    the ``clamped_swiglu`` combine so both the ejkernel fused path and the
    fallback clamp identically.

    Args:
        config: Model configuration.
        dtype: Computation dtype.
        param_dtype: Parameter storage dtype.
        precision: JAX matmul precision.
        hidden_size: Override for the input/output dim.
        intermediate_size: Override for the inner MLP width.
        rngs: PRNG key container.
    """

    def __init__(
        self,
        config: Glm5NextTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the dense clamped MLP.

        Args:
            config: Model configuration.
            dtype: Computation dtype.
            param_dtype: Parameter dtype.
            precision: JAX matmul precision.
            hidden_size: Optional override for the input/output dim.
            intermediate_size: Optional override for the inner MLP width.
            rngs: Random number generator collection.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.hidden_size = hidden_size or config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.limit = float(config.swiglu_limit)
        # ejkernel's ``clamped_swiglu`` combine hard-codes silu; GLM-5 ships
        # ``hidden_act="silu"`` so the fused declaration always matches.
        if str(config.hidden_act).lower() in ("silu", "swish"):
            self.fused_act_name = "clamped_swiglu"
            self.fused_act_params = (self.limit,)
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

    def forward(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> Array:
        """Applies the clamped gated MLP transformation.

        Args:
            hidden_states: Input tensor of shape ``(batch, seq_len, hidden_dim)``.

        Returns:
            Transformed hidden states with the same shape as input.
        """
        if getattr(self, "fused_act_name", None) is not None:
            return gated_mlp_forward(self, hidden_states)
        gate_up = checkpoint_name(self.gate_up_proj(hidden_states), "mlp_gate_up")
        gate, up = split_fused_gate_up_projection(gate_up, config=self.config)
        activated = clamped_swiglu(gate, up, limit=self.limit, act_fn=self.act_fn)
        return checkpoint_name(self.down_proj(activated), "mlp_down")


class Glm5NextTextExperts(spx.Module):
    """Stacked routed experts with the clamped SwiGLU activation.

    The HF checkpoint ships fused 3-D tensors ``experts.gate_up_proj``
    ``[E, 2M, H]`` (gate rows first, ``chunk(2)`` semantics) and
    ``experts.down_proj`` ``[E, H, M]`` in ``F.linear`` orientation. EasyDeL
    keeps gate/up as *separate* 3-D MoE linears so the per-expert activation
    can clamp between the projections (the fused-MoE kernel offers no hook
    for it); the ``reform_param`` rules split the HF fused tensor on load.

    Args:
        config: Model configuration.
        dtype: Computation dtype.
        param_dtype: Parameter storage dtype.
        precision: JAX matmul precision.
        rngs: PRNG key container.
    """

    def __init__(
        self,
        config: Glm5NextTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the routed-expert projections.

        Args:
            config: Model configuration carrying ``n_routed_experts`` and
                ``moe_intermediate_size``.
            dtype: Computation dtype.
            param_dtype: Parameter dtype.
            precision: JAX matmul precision.
            rngs: Random number generator collection.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.num_experts = config.n_routed_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.limit = float(config.swiglu_limit)
        moe_kwargs = dict(
            rngs=rngs,
            use_bias=False,
            # HF `_init_weights` uses `initializer_range` (0.02) for the
            # expert stacks; the JAX default stddev of 1.0 would hard-saturate
            # the ±swiglu_limit clamp at from-scratch init.
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            partition_manager=config.runtime_sharding_resolver,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.gate_proj = ColumnParallelMoELinear(
            num_experts=self.num_experts,
            in_features=self.hidden_size,
            out_features=self.intermediate_size,
            **moe_kwargs,
        )
        self.up_proj = ColumnParallelMoELinear(
            num_experts=self.num_experts,
            in_features=self.hidden_size,
            out_features=self.intermediate_size,
            **moe_kwargs,
        )
        self.down_proj = RowParallelMoELinear(
            num_experts=self.num_experts,
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            **moe_kwargs,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    @property
    def reform_param(self):
        """Split rules for the HF fused/stacked expert tensors.

        Returns:
            dict: HF fused ``gate_up_proj`` splits into ``gate_proj`` /
                ``up_proj`` kernels; ``down_proj`` is transposed from
                ``F.linear`` to kernel orientation. Keys are module-relative;
                the converter prefixes them with this module's path.
        """
        intermediate_size = self.intermediate_size
        return {
            "gate_up_proj$": {
                "splits": [
                    {
                        "name": "gate_proj.weight",
                        "spliter": lambda x: x[:, :intermediate_size, :].permute(0, 2, 1),
                    },
                    {
                        "name": "up_proj.weight",
                        "spliter": lambda x: x[:, intermediate_size:, :].permute(0, 2, 1),
                    },
                ],
                "inverse_spliter": lambda torch, gate, up: torch.cat(
                    (gate.permute(0, 2, 1), up.permute(0, 2, 1)), dim=1
                ),
            },
            "down_proj$": {
                "splits": [
                    {
                        "name": "down_proj.weight",
                        "spliter": lambda x: x.permute(0, 2, 1),
                    }
                ],
                "inverse_spliter": lambda torch, down: down.permute(0, 2, 1),
            },
        }

    def forward(
        self,
        hidden_states: Array,
        group_sizes: Array,
        sorted_experts: Array | None = None,
    ) -> Array:
        """Apply the clamped SwiGLU expert FFN over expert-grouped tokens.

        Args:
            hidden_states: Expert-grouped token representations.
            group_sizes: Per-expert group sizes.
            sorted_experts: Optional sorted expert indices.

        Returns:
            Expert outputs in the grouped layout.
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        gate = checkpoint_name(self.gate_proj(hidden_states, group_sizes, sorted_experts), name="mlp_gate_up")
        up = checkpoint_name(self.up_proj(hidden_states, group_sizes, sorted_experts), name="mlp_gate_up")
        activated = clamped_swiglu(gate, up, limit=self.limit, act_fn=self.act_fn)
        return typing.cast(
            Array,
            apply_logical_sharding(
                checkpoint_name(self.down_proj(activated, group_sizes, sorted_experts), name="mlp_down"),
                dynamic_axes=common_types.HiddenStateSharding,
                partition_manager=self.config.runtime_sharding_resolver,
            ),
        )


class Glm5NextTextTopKRouter(spx.Module):
    """Routing gate for GLM-5 grouped sigmoid top-k expert selection.

    Projects hidden states into per-expert logits via a learned weight matrix
    (evaluated in fp32) and exposes the per-expert score-correction bias used
    for auxiliary-loss-free load balancing.

    Args:
        config: Model configuration.
        dtype: Computation dtype.
        param_dtype: Parameter storage dtype.
        precision: JAX matmul precision.
        rngs: PRNG key container.
    """

    def __init__(
        self,
        config: Glm5NextTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the GLM-5 top-k routing gate.

        Args:
            config: Model configuration carrying ``n_routed_experts``.
            dtype: Computation dtype.
            param_dtype: Parameter dtype (the router matmul itself runs in
                fp32 regardless).
            precision: JAX matmul precision.
            rngs: Random number generator collection.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.n_routed_experts = config.n_routed_experts
        self.weight = spx.Parameter(
            jax.nn.initializers.normal(config.initializer_range)(
                rngs.param,
                (config.hidden_size, self.n_routed_experts),
                param_dtype,
            )
        )
        # Declared via ArrayParam.bound (as glm4_moe / glm_moe_dsa do) rather
        # than a bare spx.Parameter: under sequential_init the bare form stays
        # an unmaterialised ShapeDtypeStruct, and routing reads it.
        self.e_score_correction_bias = ArrayParam.bound(
            shape=(self.n_routed_experts,),
            dtype=jnp.float32,
            init_method="zeros",
            key=rngs.param,
        )

    def forward(self, hidden_states: Float[Array, "tokens hidden_dim"]) -> Array:
        """Computes per-expert routing logits for all tokens.

        Args:
            hidden_states: Flattened token representations.

        Returns:
            Router logits of shape ``(tokens, n_routed_experts)`` (fp32).
        """
        return fp32_router_logits(hidden_states, self.weight.value, self.config.hidden_size)


class Glm5NextTextMoE(BaseMoeModule):
    """Mixture-of-Experts feed-forward block for GLM-5 sparse layers.

    Routes tokens through ``num_experts_per_tok`` routed experts selected via
    grouped sigmoid top-k gating with a score-correction bias, plus shared
    experts that process every token unconditionally. Combine weights come
    from the raw sigmoid scores (the correction bias steers *selection*
    only), scaled by ``routed_scaling_factor``.

    Args:
        config: Model configuration.
        dtype: Computation dtype.
        param_dtype: Parameter storage dtype.
        precision: JAX matmul precision.
        rngs: PRNG key container.
    """

    def __init__(
        self,
        config: Glm5NextTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the GLM-5 sparse FFN.

        Args:
            config: Model configuration carrying every routing knob.
            dtype: Computation dtype.
            param_dtype: Parameter dtype.
            precision: JAX matmul precision.
            rngs: Random number generator collection.
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
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        # HF scores groups by the sum of their top-2 expert scores
        # (``group_scores.topk(2).sum(-1)``).
        self.group_topk_k = min(2, (config.n_routed_experts or 0) // max(config.n_group, 1))

        self.experts = Glm5NextTextExperts(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.gate = Glm5NextTextTopKRouter(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.shared_experts = (
            Glm5NextTextMLP(
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
        )

    def _select_hook(self):
        """Bind the shared grouped top-k selector to this layer's live bias.

        ``e_score_correction_bias`` steers *selection* without touching the
        combine weights; it must be read here (not captured in ``__init__``)
        so a trained bias reaches routing instead of a frozen zero init.

        Returns:
            partial: Configured :func:`moe_group_topk_select` selector.
        """
        return partial(
            moe_group_topk_select,
            n_routed_experts=self.n_routed_experts,
            score_fn="sigmoid",
            e_score_correction_bias=self.gate.e_score_correction_bias.value,
            n_group=self.config.n_group,
            topk_group=self.config.topk_group,
            group_topk_k=self.group_topk_k,
            group_score="topk_sum",
            norm_topk_prob=self.config.norm_topk_prob,
            routed_scaling_factor=self.config.routed_scaling_factor,
        )

    def forward(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> tuple[Array, Array]:
        """Routes tokens through selected experts and combines outputs.

        Args:
            hidden_states: Input tensor of shape ``(batch, seq_len, hidden_dim)``.

        Returns:
            Tuple of (combined expert output, router logits).
        """
        hooks = self.moe_hooks.replace(select_hook=self._select_hook())
        limit = self.config.swiglu_limit

        def ffn_activation(gate: Array, up: Array) -> Array:
            """Clamped SwiGLU: ``silu(clamp(gate, max=limit)) * clamp(up, ±limit)``.

            Args:
                gate: Gate-projection output.
                up: Up-projection output.

            Returns:
                Activated and combined expert intermediate values.
            """
            return clamped_swiglu(gate, up, limit=limit, act_fn=self.experts.act_fn)

        out, router_logits = self.moe_call(
            hidden_state=hidden_states,
            gate_layer=self.gate,
            expert_layer=self.experts,
            wi_kernel=self.experts.gate_proj.kernel_view(),
            wu_kernel=self.experts.up_proj.kernel_view(),
            wd_kernel=self.experts.down_proj.kernel_view(),
            act_fn=self.experts.act_fn,
            ffn_activation=ffn_activation,
            hooks=hooks,
        )
        if self.shared_experts is not None:
            out = out + self.shared_experts(hidden_states)
        return checkpoint_name(out, "moe_expert_output"), checkpoint_name(router_logits, "moe_router_logits")


class Glm5NextHyperConnection(ManifoldHyperConnection):
    """GLM-5 configuration adapter for shared manifold hyper-connections.

    Preserves direct ``fn`` / ``base`` / ``scale`` checkpoint parameters and
    maps flattened ``hc_mult`` residual streams to three fp32 mixing factors:

    - ``pre`` (``sigmoid + hc_eps``): stream-collapse weights producing the
      sub-layer input.
    - ``post`` (``2 * sigmoid``): placement weights of the sub-layer output
      back onto the streams.
    - ``comb``: an ``hc x hc`` stream mixer, softmax-initialised then
      projected onto doubly-stochastic matrices with ``hc_sinkhorn_iters``
      alternating column/row normalizations (applied *transposed* by the
      caller).

    Args:
        config: Model configuration.
        dtype: Computation dtype (mHC math itself runs in fp32).
        param_dtype: Parameter storage dtype.
        precision: Matmul precision for the learned stream projection.
        rngs: Random number generators.
    """

    def __init__(
        self,
        config: Glm5NextTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the hyper-connection module.

        Args:
            config: Model configuration (reads ``hc_mult``,
                ``hc_sinkhorn_iters``, ``hc_eps``, ``hidden_size``,
                ``rms_norm_eps``, ``initializer_range``,
                ``hc_use_fused_coefficients``).
            dtype: Activation dtype (mHC math itself runs in fp32).
            param_dtype: Parameter storage dtype.
            precision: Matmul precision for the learned stream projection.
            rngs: Random number generators.
        """
        super().__init__(
            ManifoldHyperConnectionConfig.from_model_config(config),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            mesh_source=config,
        )
        # Keep the model configuration public; the shared layer retains scalar
        # settings and resolves the stage-local mesh through mesh_source.
        self.config = config


def _hc_head_collapse(hidden_streams: Float[Array, "batch seq hc hidden"]) -> Array:
    """Collapse mHC streams to a single sequence with an unweighted mean.

    GLM-5's ``Glm5NextTextHyperHead`` (unlike DeepSeek-V4's learned head)
    averages the streams before the final RMSNorm.

    Args:
        hidden_streams: Residual streams ``[B, S, hc, D]``.

    Returns:
        Collapsed hidden states ``[B, S, D]`` in the input dtype.
    """
    return mean_hyper_head(hidden_streams)


class Glm5NextLinearAttention(KernelDeltaAttention):
    """GLM-5-Next KDA linear-attention layer: the shared :class:`KernelDeltaAttention` with the safe forget gate.

    The per-channel log-decay is ``linear_lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))``, in
    ``[linear_lower_bound, 0)``; the bound also lets the fused ``kda_chunk`` kernel
    factor the pairwise decay into MXU matmuls. Hugging Face stores one fused depthwise ``conv1d`` over the
    concatenated q/k/v channels; :attr:`reform_param` slices it into the three per-stream convolutions (identical for a
    depthwise conv), so the layer shares the KDA cache layout with Kimi-Linear.

    Args:
        config: Model configuration.
        layer_idx: Index of this layer in the decoder stack.
        dtype: Computation dtype.
        param_dtype: Parameter storage dtype.
        precision: JAX matmul precision.
        rngs: PRNG key container.
    """

    def __init__(
        self,
        config: Glm5NextTextConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Build the shared KDA layer with ``gate_lower_bound=config.linear_lower_bound``."""
        super().__init__(
            config,
            layer_idx,
            dtype,
            param_dtype,
            precision,
            gate_lower_bound=config.linear_lower_bound,
            rngs=rngs,
        )

    @property
    def reform_param(self):
        """Split HF's fused depthwise ``conv1d.weight`` ``[2 * key_dim + value_dim, 1, d_conv]`` into ``q/k/v_conv1d``.

        Checkpoints that already ship separate convolutions convert through the default 3-D permute; every other
        tensor (including the flat forget-gate ``A_log`` / ``dt_bias`` / ``f_a_proj`` / ``f_b_proj``) maps by name.

        Returns:
            dict: Reform rules, module-relative.
        """
        qkv = self.key_dim
        return {
            "conv1d.weight$": {
                "splits": [
                    {"name": "q_conv1d.weight", "spliter": lambda x: x[:qkv].permute(2, 1, 0)},
                    {"name": "k_conv1d.weight", "spliter": lambda x: x[qkv : 2 * qkv].permute(2, 1, 0)},
                    {"name": "v_conv1d.weight", "spliter": lambda x: x[2 * qkv :].permute(2, 1, 0)},
                ],
                "inverse_spliter": lambda torch, q, k, v: torch.cat(
                    [q.permute(2, 1, 0), k.permute(2, 1, 0), v.permute(2, 1, 0)], dim=0
                ),
            },
        }


class Glm5NextIndexer(SparseIndexer):
    """GLM-5 k-pool DSA indexer on the unified :class:`SparseIndexer` layer.

    Projection plumbing, pool compression, ReLU-weighted scoring, top-k
    selection and tail-pool expansion live in the shared
    :class:`~easydel.layers.indexer.SparseIndexer`. This subclass maps the
    model config and adds opt-in differentiable scores for distillation (the
    shared pool strategy only exposes hard selections). Checkpoint
    parameter names (``wq_b`` / ``wk`` / ``k_norm`` / ``weights_proj`` /
    ``index_kpool_compress_ape`` / ``index_kpool_compress_gate``) are owned by
    the parent class and are unchanged.
    """

    # ``layer_idx`` is informational on the indexer, so it does not split the layer scan.
    _spx_scan_safe_static_fields: ClassVar[frozenset[str]] = frozenset({"layer_idx"})

    def __init__(
        self,
        config: Glm5NextTextConfig,
        layer_idx: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Map the model config onto the unified indexer configuration.

        Args:
            config: Model configuration carrying ``index_n_heads``,
                ``index_head_dim``, ``index_topk``, ``index_kpool``,
                ``q_lora_rank``.
            layer_idx: Index of this layer in the decoder stack.
            dtype: Computation dtype.
            param_dtype: Parameter dtype.
            precision: JAX matmul precision.
            rngs: Random number generator collection.
        """
        super().__init__(
            IndexerConfig(
                kind=IndexerKind.POOL,
                index_n_heads=config.index_n_heads,
                index_head_dim=config.index_head_dim,
                index_topk=config.index_topk,
                hidden_size=config.hidden_size,
                q_input_dim=config.q_lora_rank if config.q_lora_rank is not None else config.hidden_size,
                score_activation="relu",
                head_reduction="weighted",
                query_source="q_lora",
                rope_style="none",
                norm_eps=1e-6,
                packed_state="key_gate_valid",
                stop_gradient=True,
                kpool_size=config.index_kpool,
                select_tail=config.index_kpool_always_select_tail,
                initializer_range=config.initializer_range,
                scoring_kernel=config.index_scoring_kernel,
            ),
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        # Row-parallel TPU top-k resolves its layout from the model mesh.
        self.mesh_source = config


class Glm5NextDSAAttention(DeepSeekSparseAttention):
    """GLM-5-Next DSA attention: NoPE MLA (``qk_rope_head_dim=0``) restricted by the k-pool :class:`Glm5NextIndexer`.

    Ordinary transformer caches keep the packed indexer features in their position-aligned auxiliary channel; MLA
    ragged-pages serving keeps them in indexer pages and selects pools of ``index_kpool`` tokens. The
    ``projection_mapping`` binds the HF checkpoint attribute names.
    """

    projection_mapping: ClassVar[dict[str, str]] = {
        "mla_q_a_proj": "q_a_proj",
        "mla_q_a_layernorm": "q_a_layernorm",
        "mla_q_b_proj": "q_b_proj",
        "mla_kv_a_proj_with_mqa": "kv_a_proj_with_mqa",
        "mla_kv_a_layernorm": "kv_a_layernorm",
        "mla_kv_b_proj": "kv_b_proj",
        "output_projection": "o_proj",
    }

    def _create_indexer(self, config, dtype, param_dtype, precision, rngs) -> Glm5NextIndexer:
        """This layer's k-pool indexer (see :meth:`DeepSeekSparseAttention._create_indexer`)."""
        return Glm5NextIndexer(
            config=config,
            layer_idx=self.layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def _mla_softmax_scale(self, config) -> float:
        """``q_head_dim ** -0.5``: the reference applies no YaRN ``mscale`` correction."""
        return self.q_head_dim**-0.5


class Glm5NextDecoderLayer(spx.Module):
    """Single GLM-5 decoder layer with two mHC sites.

    Dispatches attention on ``config.layer_types[layer_idx]`` —
    :class:`Glm5NextLinearAttention` (``"linear_attention"``) or
    :class:`Glm5NextDSAAttention` (``"deepseek_sparse_attention"``) — and the
    feed-forward on ``config.mlp_layer_types[layer_idx]`` (:class:`Glm5NextTextMoE`
    for ``"sparse"``, :class:`Glm5NextTextMLP` otherwise). Both sub-layer
    sites are wrapped in :class:`Glm5NextHyperConnection` mixing:

        ``post, comb, collapsed = attn_hc(streams)``
        ``streams = post ⊗ attn(norm(collapsed)) + combᵀ @ streams``

    Args:
        config: Model configuration.
        layer_idx: Index of this layer in the decoder stack.
        dtype: Computation dtype.
        param_dtype: Parameter storage dtype.
        precision: JAX matmul precision.
        rngs: PRNG key container.
    """

    # ``layer_idx`` only selects the sub-module classes at init (recorded in ``block_type``), so layers of the same
    # type share one lowered scan body.
    _spx_scan_safe_static_fields: ClassVar[frozenset[str]] = frozenset({"layer_idx"})

    def __init__(
        self,
        config: Glm5NextTextConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize one GLM-5 decoder block.

        Args:
            config: Model configuration.
            layer_idx: Index of this layer in the decoder stack.
            dtype: Computation dtype.
            param_dtype: Parameter dtype.
            precision: JAX matmul precision.
            rngs: Random number generator collection.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        # ``rngs`` is consumed by sub-module initialisers only; it is not kept on the layer (an unused shared RNG
        # would force the sequential-carry scan lowering).
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.block_type = config.layer_types[layer_idx]

        attention_cls = (
            Glm5NextLinearAttention if self.block_type == LINEAR_ATTENTION_LAYER_TYPE else Glm5NextDSAAttention
        )
        self.self_attn = attention_cls(
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
            self.mlp = Glm5NextTextMoE(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        else:
            self.mlp = Glm5NextTextMLP(
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
        self.attn_hc = Glm5NextHyperConnection(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.ffn_hc = Glm5NextHyperConnection(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    @property
    def reform_param(self):
        """Reform rules mapping HF's flat mHC parameter names to EasyDeL nesting.

        HF stores the hyper-connection parameters flat on the decoder layer
        (``hc_attn_fn`` / ``hc_attn_base`` / ``hc_attn_scale`` and the ``ffn``
        triple) while EasyDeL nests them under the
        :class:`Glm5NextHyperConnection` sub-modules (``attn_hc.fn`` ...).
        All three are raw ``Parameter``s (not ``nn.Linear``), so HF orientation
        matches EasyDeL's — ``fn`` is ``[mix, hc*hidden]`` on both sides — and
        every rule is a pure rename.

        Returns:
            dict: Six rename rules, module-relative.
        """

        def _rule(hf_name: str, easydel_name: str):
            return {
                "splits": [{"name": easydel_name, "spliter": lambda x: x}],
                "inverse_spliter": lambda x: x,
            }

        return {
            "hc_attn_fn$": _rule("hc_attn_fn", "attn_hc.fn"),
            "hc_attn_base$": _rule("hc_attn_base", "attn_hc.base"),
            "hc_attn_scale$": _rule("hc_attn_scale", "attn_hc.scale"),
            "hc_ffn_fn$": _rule("hc_ffn_fn", "ffn_hc.fn"),
            "hc_ffn_base$": _rule("hc_ffn_base", "ffn_hc.base"),
            "hc_ffn_scale$": _rule("hc_ffn_scale", "ffn_hc.scale"),
        }

    def forward(
        self,
        hidden_states: Float[Array, "batch seq hc hidden"],
        mask_info: MaskInfo,
        position_ids: Array,
        mode: common_types.RUNTIME_MODE_TYPES,  # type: ignore
        cache_view=None,
        cache_metadata=None,
        output_attentions: bool = False,
        frequencies=None,
        previous_topk_indices: Array | None = None,
        return_topk_indices: bool = False,
    ) -> DecoderLayerOutput:
        """Run one GLM-5 decoder block over the residual streams.

        Args:
            hidden_states: Residual streams ``[B, S, hc, D]``.
            mask_info: Attention mask metadata.
            position_ids: Position indices (threaded to the DSA path).
            mode: Runtime mode.
            cache_view: Per-layer cache view (hybrid KDA / full-attention).
            cache_metadata: Cache metadata.
            output_attentions: Whether to expose attention weights.
            frequencies: Unused (NoPE); kept for call compatibility.

        Returns:
            :class:`DecoderLayerOutput` with the updated streams, optional
            attention weights, updated cache view, and optional router logits.
        """
        post, comb, collapsed = self.attn_hc(hidden_states)
        attn_input = self.input_layernorm(collapsed)
        attn_input = apply_logical_sharding(
            attn_input,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        topk_indices = previous_topk_indices
        if self.block_type == LINEAR_ATTENTION_LAYER_TYPE:
            attn_outputs = self.self_attn(
                hidden_states=attn_input,
                mask_info=mask_info,
                cache_view=cache_view,
                cache_metadata=cache_metadata,
            )
        elif return_topk_indices:
            attn_outputs, topk_indices = self.self_attn.forward_mla(
                attn_input,
                mask_info,
                position_ids,
                mode,
                cache_view,
                cache_metadata,
                output_attentions,
                frequencies,
                previous_topk_indices=previous_topk_indices,
                return_topk_indices=True,
            )
        else:
            attn_outputs = self.self_attn(
                attn_input,
                mask_info,
                position_ids,
                mode,
                cache_view,
                cache_metadata,
                output_attentions,
                frequencies,
            )
        hidden_states = manifold_residual_write(
            hidden_states, attn_outputs.attention_output, post, comb, precision=self.precision
        )

        post, comb, collapsed = self.ffn_hc(hidden_states)
        mlp_input = self.post_attention_layernorm(collapsed)
        if self.config.use_scan_mlp and isinstance(self.mlp, Glm5NextTextMLP):
            feed_forward_hidden_states = blockwise_ffn(
                self.mlp,
                mlp_input,
                self.config.scan_mlp_chunk_size,
            )
        else:
            feed_forward_hidden_states = self.mlp(mlp_input)
        router_logits = None
        if isinstance(feed_forward_hidden_states, tuple):
            feed_forward_hidden_states, router_logits = feed_forward_hidden_states
        hidden_states = manifold_residual_write(
            hidden_states, feed_forward_hidden_states, post, comb, precision=self.precision
        )
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=HyperStreamSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )

        result = SparseDecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
            cache_view=attn_outputs.cache_view,
            router_logits=router_logits,
            indexer_aux_loss=getattr(attn_outputs, "indexer_aux_loss", None),
        )
        return (result, topk_indices) if return_topk_indices else result


class _Glm5NextIndexerCacheMixin:
    """Preallocate pooled indexer features alongside ordinary MLA KV storage."""

    def create_transformer_cache_config(self, batch_size, max_length, pad_token_id=None, layer_idx=None):
        """Keep standard cache geometry and request float32 pooled features."""
        config = super().create_transformer_cache_config(batch_size, max_length, pad_token_id, layer_idx)
        return config.replace(auxiliary_dim=2 * self.config.get_text_config().index_head_dim + 1)


@register_module(TaskType.BASE_MODULE, config=Glm5NextTextConfig, model_type="glm5_next_text")
class Glm5NextTextModel(_Glm5NextIndexerCacheMixin, EasyDeLBaseModule):
    """GLM-5-Next text backbone (no language-model head).

    Embeds tokens, broadcasts the embedding to ``hc_mult`` mHC residual
    streams, runs the heterogeneous KDA/DSA decoder stack, collapses the
    streams with an unweighted mean, and applies the final RMSNorm.

    Args:
        config: Model configuration.
        dtype: Computation dtype.
        param_dtype: Parameter storage dtype.
        precision: JAX matmul precision.
        rngs: PRNG key container.
    """

    fp32_param_patterns: ClassVar[tuple[str, ...]] = ("e_score_correction_bias",)
    # Accepts native checkpoints that nest the forget gate under ``self_attn.forget_gate``.
    _upgrade_legacy_native_state = staticmethod(unnest_legacy_kda_forget_gate)

    def __init__(
        self,
        config: Glm5NextTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the GLM-5-Next backbone.

        Args:
            config: Model configuration.
            dtype: Computation dtype.
            param_dtype: Parameter dtype.
            precision: JAX matmul precision.
            rngs: Random number generator collection.
        """
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
            Glm5NextDecoderLayer,
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
        """RoPE table — always ``None`` (GLM-5-Next is fully NoPE).

        Returns:
            None: Kept for call compatibility with rotary-based models.
        """
        return None

    def get_embedding(self):
        """Return the token embedding layer.

        Returns:
            Embed: The token embedding layer (``embed_tokens``).
        """
        return self.embed_tokens

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
        """Run the GLM-5-Next backbone.

        Args:
            input_ids: Token ids ``(batch, seq_len)``, mutually exclusive with
                ``inputs_embeds``.
            inputs_embeds: Pre-computed embeddings.
            attention_mask: Optional padding mask.
            mask_info: Optional pre-computed mask metadata.
            position_ids: Optional positions (unused — NoPE).
            mode: Runtime mode; auto-detected when ``None``.
            past_key_values: Existing cache to update.
            cache_metadata: Cache metadata for paged attention.
            output_attentions: Collect attention weights.
            output_hidden_states: Collect intermediate hidden states.
            output_router_logits: Collect router logits from MoE layers.

        Returns:
            :class:`MoeModelOutput` with the final hidden state, optional
            collections, and the updated cache.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = checkpoint_name(self.embed_tokens(input_ids.astype("i4")), "embeddings")
        sequence_length = inputs_embeds.shape[1]
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
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

        if mode is None:
            mode = (
                common_types.MODE_DECODE
                if sequence_length == 1 and past_key_values is not None
                else common_types.MODE_TRAIN
            )

        if past_key_values is None:
            past_key_values = HybridCache.init_empty(len(self.layers))

        # mHC residual streams: broadcast the embedding across hc_mult slots.
        hidden_states = jnp.broadcast_to(
            inputs_embeds[:, :, None, :],
            (inputs_embeds.shape[0], sequence_length, self.config.hc_mult, inputs_embeds.shape[-1]),
        ).astype(inputs_embeds.dtype)

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=HyperStreamSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )

        views = past_key_values.views if past_key_values is not None else None
        has_cache_views = views is not None and any(v is not None for v in views)
        needs_trace_cache = mode == common_types.MODE_DECODE or has_cache_views
        has_shared_indexers = "shared" in self.config.indexer_types
        # With ``config.scan_layers`` the heterogeneous stack is scanned as one segment per run of identical layers
        # (KDA vs DSA, dense vs sparse MLP). A scan needs a fixed carry structure, so stacks whose carry gains
        # entries mid-stack (shared-indexer selections, the indexer aux loss) stay Python-traced.
        indexer_aux_active = float(getattr(self.config, "indexer_aux_loss_coef", 0.0) or 0.0) > 0.0
        trace_layers = self._layer_scan_trace(
            False,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            cache_views=views,
            extra=needs_trace_cache or bool(output_router_logits) or has_shared_indexers or indexer_aux_active,
        )
        cache_views = views if trace_layers else None

        def _layer_loop(block, carry):
            """Run one heterogeneous decoder layer inside the stack loop.

            Threads streams, cache views, collected outputs, layer index,
            the previous selection and the unscaled per-layer indexer loss sum
            through the Python-traced loop.

            Args:
                block: The decoder layer module.
                carry: Current loop carry.

            Returns:
                Updated carry ``(streams, cache_views, ..., idx + 1)``.
            """
            hidden_states, cv, all_hidden_states, all_attentions, all_router_logits, idx, prev_topk, indexer_aux = carry
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            with self._layer_stage_context(idx, layers=self.layers):
                layer_outputs = block(
                    hidden_states=hidden_states,
                    mask_info=mask_info,
                    position_ids=position_ids,
                    mode=mode,
                    cache_view=self._layer_cache_view_at(cv, idx, enabled=trace_layers, cache=past_key_values),
                    cache_metadata=cache_metadata,
                    output_attentions=output_attentions,
                    frequencies=self.frequencies,
                    **({"previous_topk_indices": prev_topk, "return_topk_indices": True} if has_shared_indexers else {}),
                )
                if has_shared_indexers:
                    layer_outputs, prev_topk = layer_outputs

            hidden_states = self._mark_layer_stage_boundary(layer_outputs.hidden_states, idx, layers=self.layers)

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            if output_router_logits and layer_outputs.router_logits is not None:
                all_router_logits += (layer_outputs.router_logits,)

            cv = self._layer_cache_view_update(
                cv,
                idx,
                layer_outputs.cache_view,
                enabled=trace_layers,
                cache=past_key_values,
            )

            layer_aux = layer_outputs.indexer_aux_loss
            if layer_aux is not None:
                indexer_aux = layer_aux if indexer_aux is None else indexer_aux + layer_aux
            return (
                hidden_states,
                cv,
                all_hidden_states,
                all_attentions,
                all_router_logits,
                idx + 1,
                prev_topk,
                indexer_aux,
            )

        initial = (hidden_states, cache_views, all_hidden_states, all_attentions, all_router_logits, 0, None, None)
        final = self.layers.scan(_layer_loop, initial, trace=trace_layers, **self._layer_scan_kwargs("text"))
        hidden_states, _, all_hidden_states, all_attentions, all_router_logits, _, _, indexer_aux_loss = final

        # Unweighted mean over mHC streams, then final norm.
        hidden_states = self.norm(_hc_head_collapse(hidden_states))

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return MoeModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            router_logits=all_router_logits,
            past_key_values=past_key_values,
            indexer_aux_loss=indexer_aux_loss,
        )


#: The published `zai-org/GLM-5.3-Flash` checkpoint is the multimodal composite: the language model lives under
#: ``model.language_model.`` (EasyDeL builds it as ``model.``) and the vision tower under ``model.visual.``
#: (unowned by the text model). Text-only extractions already in ``model.layers...`` pass through.
GLM5_NEXT_KEY_RULES = KeyRuleSet(
    [
        Map(r"model\.visual\..*", None),
        Map(r"model\.language_model\.(?P<rest>.*)", "model.{rest}"),
    ],
    otherwise=KEEP,
    name="glm5_next",
)


@register_module(TaskType.CAUSAL_LM, config=Glm5NextTextConfig, model_type="glm5_next_text")
class Glm5NextForCausalLM(_Glm5NextIndexerCacheMixin, BaseCausalLMModule[Glm5NextTextModel, Glm5NextTextConfig]):  # type: ignore
    """GLM-5-Next model with a causal language-modelling head.

    Wraps :class:`Glm5NextTextModel` and adds a bias-free LM head projecting
    the final hidden states to vocabulary logits (tied to the embedding when
    ``config.tie_word_embeddings``).

    Args:
        config: Model configuration.
        dtype: Computation dtype.
        param_dtype: Parameter storage dtype.
        precision: JAX matmul precision.
        rngs: PRNG key container.
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "glm5_next_text"
    _config_class = Glm5NextTextConfig
    _upgrade_legacy_native_state = staticmethod(unnest_legacy_kda_forget_gate)

    fp32_param_patterns: ClassVar[tuple[str, ...]] = ("e_score_correction_bias",)

    def __init__(
        self,
        config: Glm5NextTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the causal-LM wrapper around :class:`Glm5NextTextModel`.

        Args:
            config: Model configuration.
            dtype: Computation dtype.
            param_dtype: Parameter dtype.
            precision: JAX matmul precision.
            rngs: Random number generator collection.
        """
        super().__init__(
            config=config,
            base_model_class=Glm5NextTextModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
            router_aux_loss_coef=getattr(config, "router_aux_loss_coef", None),
        )

    checkpoint_key_rules: ClassVar[KeyRuleSet] = GLM5_NEXT_KEY_RULES

    def has_mtp(self) -> bool:
        """Whether a nextn (MTP) head is attached as ``self.mtp``.

        The head is optional and attached after loading
        (:func:`easydel.modules.glm5_next.modeling_glm5_next_mtp.attach_glm5_next_mtp`), so plain GLM-5-Next
        models are unchanged and report ``False``.

        Returns:
            ``True`` iff an MTP head is attached.
        """
        return getattr(self, "mtp", None) is not None


@register_module(TaskType.CAUSAL_LM, config=Glm5NextTextConfig, model_type="glm5_next")
class Glm5NextCompositeCausalLM(Glm5NextForCausalLM):
    """Registry alias for composite `zai-org/GLM-5.3-Flash` checkpoints.

    The published checkpoint declares the multimodal `model_type: "glm5_next"`
    (`Glm5NextForConditionalGeneration`, nested `text_config`/`vision_config`).
    EasyDeL serves the text stack, so this alias lets
    `get_modules_by_type("glm5_next", CAUSAL_LM)` resolve to the text causal-LM
    module, with :meth:`Glm5NextTextConfig` flattening the nested `text_config`
    during config load. Vision tensors are skipped by the converter (they match
    no EasyDeL parameter).
    """


__all__ = ["Glm5NextCompositeCausalLM", "Glm5NextForCausalLM", "Glm5NextTextModel"]
