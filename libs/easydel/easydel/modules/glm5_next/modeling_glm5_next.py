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
import math
import typing
from functools import partial
from typing import ClassVar

import jax
import jax.numpy as jnp
import spectrax as spx
from eformer.loggings import get_logger
from ejkernel.modules import sinkhorn_knopp
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as Ps
from jaxtyping import Array, Bool, Float, Int
from spectrax import apply_logical_sharding, common_types, nn

from easydel.caching import (
    HybridCache,
    KDACacheView,
    KDAMetadata,
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
from easydel.infra.modeling_outputs import (
    AttentionLayerOutput,
    DecoderLayerOutput,
    MoeModelOutput,
)
from easydel.infra.sequence_packing import (
    packed_segment_ids_from_mask_info,
    pairwise_attention_mask_from_mask_info,
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
    RMSNormGated,
    RowParallelLinear,
    RowParallelMoELinear,
    clamped_swiglu,
    dense_gate_up_layout,
    gated_mlp_forward,
    split_fused_gate_up_projection,
)
from easydel.layers.attention import FlexibleAttentionModule, UnifiedAttention
from easydel.layers.indexer import IndexerConfig, IndexerKind, SparseIndexer
from easydel.layers.linear_attention import apply_conv_with_state, apply_mask_to_padding_states
from easydel.layers.moe import moe_group_topk_select
from easydel.layers.norms import lowfloats
from easydel.modules._base import BaseCausalLMModule
from easydel.operations import OperationMetadata
from easydel.operations.kernels import (
    KDAOutput,
    KernelDeltaAttnOp,
    fused_kda_gate_per_channel,
)
from easydel.operations.kernels.kda import (
    _chunked_scan_per_channel_rows,
    _single_step_kda_per_channel_fwd_bthd,
)

from .glm5_next_configuration import (
    LINEAR_ATTENTION_LAYER_TYPE,
    Glm5NextTextConfig,
)

logger = get_logger(__name__)

_KDA_CHUNK_SIZE = 64


def _unweighted_rms_norm(x: Array, eps: float) -> Array:
    """Unweighted RMSNorm: ``x * rsqrt(mean(x^2) + eps)`` with an fp32 moment.

    Mirrors HF ``Glm5NextTextUnweightedRMSNorm`` used by the mHC mapping.

    Args:
        x: Input array; normalization runs over the last axis.
        eps: Variance epsilon.

    Returns:
        Array of the same shape and dtype as ``x``.
    """
    scale = jax.lax.rsqrt(jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True) + eps)
    return x * scale.astype(x.dtype)


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
            kernel_init=jax.nn.initializers.normal(),
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
        hidden_states = hidden_states.reshape(-1, self.config.hidden_size)
        return checkpoint_name(
            jnp.matmul(hidden_states.astype(jnp.float32), self.weight.value.astype(jnp.float32)),
            "moe_router_logits",
        )


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


class Glm5NextHyperConnection(spx.Module):
    """Manifold-Constrained Hyper-Connection (mHC) mixing module.

    Owns the learned ``fn`` / ``base`` / ``scale`` parameters that map the
    flattened ``hc_mult`` residual streams to three fp32 outputs:

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
                ``rms_norm_eps``, ``initializer_range``).
            dtype: Activation dtype (mHC math itself runs in fp32).
            param_dtype: Parameter storage dtype.
            precision: Matmul precision for the learned stream projection.
            rngs: Random number generators.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = ArrayParam.bound(
            shape=(mix, self.hc_mult * config.hidden_size),
            dtype=param_dtype,
            init_method="normal",
            init_kwargs={"stddev": config.initializer_range},
            key=rngs.param,
        )
        self.base = ArrayParam.bound(
            shape=(mix,),
            dtype=param_dtype,
            init_method="zeros",
            key=rngs.param,
        )
        self.scale = ArrayParam.bound(
            shape=(3,),
            dtype=param_dtype,
            init_method="ones",
            key=rngs.param,
        )

    def forward(self, hidden_streams: Float[Array, "batch seq hc hidden"]) -> tuple[Array, Array, Array]:
        """Compute ``(post, comb, collapsed)`` from the mHC mapping.

        Args:
            hidden_streams: Residual streams of shape ``[B, S, hc, D]``.

        Returns:
            Tuple of ``post`` (``[B, S, hc]``, fp32), ``comb``
            (``[B, S, hc, hc]``, fp32, Sinkhorn-projected), and ``collapsed``
            (``[B, S, D]``, input dtype) — the pre-weighted stream sum fed to
            the sub-layer.

        Note:
            ``comb`` descends from ``hidden_streams``; on multi-device TPU
            meshes the Sinkhorn projection is pinned to replicated sharding
            with an explicit constraint before the fused ``shard_map`` call so
            every device redundantly normalises the (couple-of-KB) matrix
            rather than moving shards.
        """
        hc = self.hc_mult
        eps = self.hc_eps
        batch, seq = hidden_streams.shape[:2]
        flat = hidden_streams.reshape(batch, seq, -1).astype(jnp.float32)
        flat = _unweighted_rms_norm(flat, self.rms_norm_eps)
        fn = self.fn.value.astype(jnp.float32)
        mix_logits = jnp.matmul(flat, fn.T, precision=self.precision)
        pre_w = mix_logits[..., :hc]
        post_w = mix_logits[..., hc : 2 * hc]
        comb_w = mix_logits[..., 2 * hc :]
        base = self.base.value.astype(jnp.float32)
        pre_b, post_b, comb_b = base[:hc], base[hc : 2 * hc], base[2 * hc :]
        scale = self.scale.value.astype(jnp.float32)

        pre = jax.nn.sigmoid(pre_w * scale[0] + pre_b) + eps
        post = 2.0 * jax.nn.sigmoid(post_w * scale[1] + post_b)
        comb_logits = comb_w.reshape(batch, seq, hc, hc) * scale[2] + comb_b.reshape(hc, hc)
        comb = jax.nn.softmax(comb_logits, axis=-1) + eps
        # Sinkhorn knopp: alternate column/row normalization onto the
        # doubly-stochastic manifold (fixed static iteration count).
        # TODO: write akernel for this one aswell ffs that indexing for b takes a lot of time
        mesh = self.config.mesh
        jax_mesh = getattr(mesh, "jax_mesh", mesh)
        if jax_mesh is not None and getattr(jax_mesh, "size", 1) > 1 and jax.default_backend() == "tpu":
            comb = jax.lax.with_sharding_constraint(comb, NamedSharding(jax_mesh, Ps()))
            comb = jax.shard_map(
                lambda c: sinkhorn_knopp(c, self.hc_sinkhorn_iters, eps),
                mesh=jax_mesh,
                in_specs=(Ps(),),
                out_specs=Ps(),
                check_vma=False,
            )(comb)
        else:
            comb = sinkhorn_knopp(comb, self.hc_sinkhorn_iters, eps)

        collapsed = jnp.sum(pre[..., None] * hidden_streams.astype(jnp.float32), axis=2)
        return post, comb, collapsed.astype(hidden_streams.dtype)


def _hc_head_collapse(hidden_streams: Float[Array, "batch seq hc hidden"]) -> Array:
    """Collapse mHC streams to a single sequence with an unweighted mean.

    GLM-5's ``Glm5NextTextHyperHead`` (unlike DeepSeek-V4's learned head)
    averages the streams before the final RMSNorm.

    Args:
        hidden_states: Residual streams ``[B, S, hc, D]``.

    Returns:
        Collapsed hidden states ``[B, S, D]`` in the input dtype.
    """
    return jnp.mean(hidden_streams, axis=2)


class Glm5NextForgetGate(spx.Module):
    """GLM-5 per-channel KDA forget gate.

    Two-layer MLP ``f_b(f_a(x))`` whose ``qkv_dim`` output is viewed as
    ``(num_heads, head_dim)``; the log-decay is
    ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`` per channel (safe
    gate) or ``-exp(A_log) * softplus(g + dt_bias)`` without the bound.
    ``A_log`` stays per-head ``(num_heads,)`` and broadcasts across each
    head's channels.

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
        """Initialize the forget gate projections and decay parameters.

        Args:
            config: Model configuration.
            dtype: Computation dtype.
            param_dtype: Parameter dtype.
            precision: JAX matmul precision.
            rngs: Random number generator collection.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.head_dim = config.linear_head_dim
        self.num_heads = config.linear_num_heads
        self.qkv_dim = self.head_dim * self.num_heads
        self.lower_bound = config.linear_lower_bound

        self.f_a_proj = ColumnParallelLinear(
            config.hidden_size,
            self.head_dim,
            rngs=rngs,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
        )
        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            self.qkv_dim,
            rngs=rngs,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
        )
        # FLA initialization (HF `_init_weights`): A_log = 0 with the safe gate.
        # dt_bias: HF draws dt ~ U(1e-3, 1e-1) and stores its inverse softplus
        # (support ≈ [-6.9, -2.25]); ArrayParam initializers are standard JAX
        # factories, so we use the support midpoint via `constant` — matching
        # the kimi_linear precedent. From-pretrained loads overwrite both.
        # Both stay float32 (HF keeps them in strict-fp32 modules).
        self.A_log = ArrayParam.bound(
            shape=(self.num_heads,),
            dtype=jnp.float32,
            init_method="zeros",
            key=rngs.param,
        )
        dt_bias_mid = 0.5 * (math.log(math.expm1(1e-3)) + math.log(math.expm1(1e-1)))
        self.dt_bias = ArrayParam.bound(
            shape=(self.qkv_dim,),
            dtype=jnp.float32,
            init_method="constant",
            init_kwargs={"value": dt_bias_mid},
            key=rngs.param,
        )

    def forward(self, hidden_states: Float[Array, "batch seq hidden"]) -> Float[Array, "batch seq num_heads head_dim"]:
        """Materialise the per-channel log-decay for the KDA kernels.

        Args:
            hidden_states: Layer input ``(batch, seq, hidden_size)``.

        Returns:
            Float32 per-channel log-decay ``(batch, seq, num_heads, head_dim)``
            with values in ``[lower_bound, 0)`` (safe gate) or non-positive
            (softplus).
        """
        gate = self.f_b_proj(self.f_a_proj(hidden_states))
        return fused_kda_gate_per_channel(
            gate,
            self.A_log.value,
            self.dt_bias.value,
            lower_bound=self.lower_bound,
        )


class Glm5NextLinearAttention(spx.Module):
    """KDA (Kimi-style kernel delta attention) block for GLM-5 linear layers.

    Differences from the Kimi-Linear donor layer:

    - **One fused conv**: HF stores a single depthwise ``conv1d`` over the
      concatenated ``3 * qkv_dim`` channels. EasyDeL keeps three per-stream
      depthwise convs (``q/k/v_conv1d``) — mathematically identical for a
      depthwise conv — so the KDACacheView conv-state slots can be reused
      directly; the ``reform_param`` rules slice the HF fused weight.
    - **Per-channel decay**: the forget gate produces an
      ``(num_heads, head_dim)`` log-decay (see :class:`Glm5NextForgetGate`)
      dispatched into :class:`KernelDeltaAttnOp` with
      ``per_channel_decay=True``.
    - **Low-rank output gate**: ``g_a_proj`` maps to ``head_dim`` (GLM-5's
      bottleneck), then ``g_b_proj`` back to ``qkv_dim``.

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
        """Initialize the GLM-5 KDA linear-attention block.

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
        self.layer_idx = layer_idx

        self.num_heads = config.linear_num_heads
        self.head_dim = config.linear_head_dim
        self.d_conv = config.linear_conv_kernel_dim
        self.chunk_size = _KDA_CHUNK_SIZE

        self.key_dim = self.num_heads * self.head_dim
        self.value_dim = self.key_dim

        column_linear = partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )

        self.q_proj = column_linear(config.hidden_size, self.key_dim)
        self.k_proj = column_linear(config.hidden_size, self.key_dim)
        self.v_proj = column_linear(config.hidden_size, self.value_dim)

        # Three depthwise causal convs (slices of HF's fused conv1d weight).
        self.q_conv1d = nn.Conv1d(
            in_channels=self.key_dim,
            out_channels=self.key_dim,
            kernel_size=self.d_conv,
            groups=self.key_dim,
            padding=((self.d_conv - 1, 0),),
            dtype=dtype,
            rngs=rngs,
            use_bias=False,
        )
        self.k_conv1d = nn.Conv1d(
            in_channels=self.key_dim,
            out_channels=self.key_dim,
            kernel_size=self.d_conv,
            groups=self.key_dim,
            padding=((self.d_conv - 1, 0),),
            dtype=dtype,
            rngs=rngs,
            use_bias=False,
        )
        self.v_conv1d = nn.Conv1d(
            in_channels=self.value_dim,
            out_channels=self.value_dim,
            kernel_size=self.d_conv,
            groups=self.value_dim,
            padding=((self.d_conv - 1, 0),),
            dtype=dtype,
            rngs=rngs,
            use_bias=False,
        )

        self.forget_gate = Glm5NextForgetGate(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.b_proj = column_linear(config.hidden_size, self.num_heads)

        self.g_a_proj = column_linear(config.hidden_size, self.head_dim)
        self.g_b_proj = column_linear(self.head_dim, self.value_dim)

        self.o_proj = RowParallelLinear(
            self.value_dim,
            config.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.o_norm = RMSNormGated(
            self.head_dim,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            activation="sigmoid",
            rngs=rngs,
        )

        metadata = OperationMetadata(
            runtime_dtype=self.dtype,
            runtime_softmax_dtype=jnp.float32,
            base_config=self.config,
        )
        self.kda_op = KernelDeltaAttnOp(metadata)

    @property
    def reform_param(self):
        """Checkpoint reform rules for this attention block.

        Two families of HF→EasyDeL renames:

        - HF nests the forget gate flat under ``self_attn`` (``A_log``,
          ``dt_bias``, ``f_a_proj``, ``f_b_proj``) while EasyDeL groups it
          under the :class:`Glm5NextForgetGate` sub-module; the 2-D
          projections also need the standard transpose.
        - The conv kernels: HF ``conv1d.weight`` (a single fused depthwise
          conv over 3*qkv channels) is split into the per-stream
          ``q/k/v_conv1d`` kernels (identity on checkpoints that, like the
          published bf16 extraction, already ship separate convs — those
          convert through the default 3-D permute).

        Returns:
            dict: Reform rules, module-relative.
        """
        qkv = self.key_dim
        transpose = lambda x: x.permute(1, 0)  # noqa: E731

        def _rename(hf_name: str, easydel_name: str, spliter=lambda x: x):
            return {
                "splits": [{"name": easydel_name, "spliter": spliter}],
                "inverse_spliter": spliter,
            }

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
            # HF keeps the forget-gate tensors flat on ``self_attn``; EasyDeL
            # nests them under the Glm5NextForgetGate sub-module.
            "A_log$": _rename("A_log", "forget_gate.A_log"),
            "dt_bias$": _rename("dt_bias", "forget_gate.dt_bias"),
            "f_a_proj.weight$": _rename(
                "f_a_proj.weight",
                "forget_gate.f_a_proj.weight",
                transpose,
            ),
            "f_b_proj.weight$": _rename(
                "f_b_proj.weight",
                "forget_gate.f_b_proj.weight",
                transpose,
            ),
        }

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None = None,
        cache_view: KDACacheView | None = None,
        cache_metadata: KDAMetadata | None = None,
    ) -> AttentionLayerOutput:
        """Run the per-channel-decay delta rule over a block of tokens.

        Streaming behaviour matches the Kimi donor: with a cache view carrying
        rolling conv windows and the recurrent memory, a single new token is
        spliced onto the conv buffers and the memory advances one step;
        otherwise the conv applies left-padding and the chunked kernel runs
        the parallel intra-chunk / recurrent inter-chunk decomposition.

        Args:
            hidden_states: ``(batch, seq_len, hidden_size)`` layer input.
            mask_info: Optional :class:`MaskInfo`; only the padding mask is
                consumed (the recurrence is causal by construction).
            cache_view: Per-layer KDA cache (conv windows + recurrent state).
            cache_metadata: Optional :class:`KDAMetadata`; unused on the
                dense layout.

        Returns:
            :class:`AttentionLayerOutput` with ``(batch, seq_len,
            hidden_size)`` output, ``attention_weight=None`` and the updated
            cache view.
        """
        if mask_info is not None:
            q_mask: Array | None = typing.cast("Array | None", mask_info.q_attention_mask)
            if q_mask is not None and q_mask.shape[1] != hidden_states.shape[1]:
                q_mask = q_mask[:, : hidden_states.shape[1]]
            hidden_states = apply_mask_to_padding_states(hidden_states, q_mask)
        # Packed-segment resets (kimi's ``segment_ids`` threading): when the
        # batch carries packing metadata, both the depthwise conv and the
        # delta-rule recurrence reset their state at document boundaries so
        # document n+1 never attends to document n's state. Unpacked batches
        # derive no segment ids and keep today's plain (faster chunked)
        # paths; decode (seq_len == 1) is unaffected — the conv decode branch
        # ignores segment ids and the single-step kernel never sees them.
        segment_ids = None
        if mask_info is not None:
            segment_ids = packed_segment_ids_from_mask_info(mask_info, hidden_states.shape[1])

        batch_size, seq_len, _ = hidden_states.shape
        is_inference = seq_len == 1 and cache_view is not None

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # eSurge packed-row serving: the runner feeds one flattened token
        # stream [1, T, D] whose per-request segments are described by
        # ``cache_metadata.query_start_loc``. Per-request state lives in
        # per-row cache slots [max_num_reqs, ...], so the layer must split
        # the stream into rows, advance each row's conv window / recurrent
        # state independently, and scatter the outputs back.
        qsl = getattr(cache_metadata, "query_start_loc", None) if cache_metadata is not None else None
        if cache_view is not None and qsl is not None:
            return self._forward_packed_rows(
                hidden_states=hidden_states,
                query=query,
                key=key,
                value=value,
                beta=jax.nn.sigmoid(self.b_proj(hidden_states)),
                decay=self.forget_gate(hidden_states),
                cache_view=cache_view,
                query_start_loc=qsl,
                cache_metadata=cache_metadata,
            )

        q_conv_state = cache_view.q_conv_state if cache_view is not None else None
        k_conv_state = cache_view.k_conv_state if cache_view is not None else None
        v_conv_state = cache_view.v_conv_state if cache_view is not None else None

        conv_output_dtype = jnp.bfloat16 if self.dtype in lowfloats else self.dtype
        query, new_q_conv_state = apply_conv_with_state(
            query,
            self.q_conv1d,
            q_conv_state,
            is_inference=is_inference,
            d_conv=self.d_conv,
            output_dtype=conv_output_dtype,
            reuse_partial_state=True,
            segment_ids=segment_ids,
        )
        key, new_k_conv_state = apply_conv_with_state(
            key,
            self.k_conv1d,
            k_conv_state,
            is_inference=is_inference,
            d_conv=self.d_conv,
            output_dtype=conv_output_dtype,
            reuse_partial_state=True,
            segment_ids=segment_ids,
        )
        value, new_v_conv_state = apply_conv_with_state(
            value,
            self.v_conv1d,
            v_conv_state,
            is_inference=is_inference,
            d_conv=self.d_conv,
            output_dtype=conv_output_dtype,
            reuse_partial_state=True,
            segment_ids=segment_ids,
        )

        query = query.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        query = apply_logical_sharding(
            query,
            dynamic_axes=common_types.AttnQSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        key = apply_logical_sharding(
            key,
            dynamic_axes=common_types.AttnKVSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        value = apply_logical_sharding(
            value,
            dynamic_axes=common_types.AttnKVSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )

        decay = self.forget_gate(hidden_states)
        beta = jax.nn.sigmoid(self.b_proj(hidden_states))

        output_gate = self.g_b_proj(self.g_a_proj(hidden_states))
        output_gate = output_gate.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        recurrent_state = cache_view.recurrent_state if cache_view is not None else None

        kda_output: KDAOutput = self.kda_op(
            query=query,
            key=key,
            value=value,
            beta=beta,
            decay=decay,
            q_conv_state=new_q_conv_state,
            k_conv_state=new_k_conv_state,
            v_conv_state=new_v_conv_state,
            recurrent_state=recurrent_state,
            chunk_size=self.chunk_size,
            per_channel_decay=True,
            segment_ids=segment_ids,
        )

        output = kda_output.attention_outputs

        output = self.o_norm(output, output_gate)
        output = output.reshape(batch_size, seq_len, -1)
        output = apply_logical_sharding(
            output,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        output = checkpoint_name(self.o_proj(output), name="attn_output")
        output = apply_logical_sharding(
            output,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )

        new_cache_view = cache_view
        if cache_view is not None:
            new_cache_view = cache_view.update_kda_states(
                new_q_conv_state=kda_output.q_conv_state,
                new_k_conv_state=kda_output.k_conv_state,
                new_v_conv_state=kda_output.v_conv_state,
                new_recurrent_state=kda_output.recurrent_state,
            )

        return AttentionLayerOutput(
            attention_output=output,
            attention_weight=None,
            cache_view=new_cache_view,
        )

    def _forward_packed_rows(
        self,
        hidden_states: Float[Array, "1 seq_len hidden_dim"],
        query: Float[Array, "1 seq_len key_dim"],
        key: Float[Array, "1 seq_len key_dim"],
        value: Float[Array, "1 seq_len value_dim"],
        beta: Float[Array, "1 seq_len num_heads"],
        decay: Float[Array, "1 seq_len num_heads head_dim"],
        cache_view: KDACacheView,
        query_start_loc: Array,
        cache_metadata: Array | None = None,
    ) -> AttentionLayerOutput:
        """Advance per-request KDA state over an eSurge packed token stream.

        The runner flattens all scheduled requests into one token stream
        ``[1, T, D]``; ``query_start_loc`` gives each request's (row's)
        segment as ``[start_i, start_{i+1})``. Per-request conv windows and
        recurrent memory live in per-row cache slots, so this method splits
        the stream into rows, runs the causal conv per row (with the carried
        window as prefix for rows that already hold context), advances the
        per-channel delta-rule state per row — single-step on pure decode
        steps, memory-bounded chunk scan otherwise — and gathers the
        per-row outputs back into the packed stream.

        Args:
            hidden_states: Packed layer input ``[1, T, D]``.
            query: Packed Q projection ``[1, T, key_dim]``.
            key: Packed K projection ``[1, T, key_dim]``.
            value: Packed V projection ``[1, T, value_dim]``.
            beta: Packed per-head gating ``[1, T, heads]`` (sigmoid applied).
            decay: Packed fused per-channel log-decay ``[1, T, heads, head_dim]``.
            cache_view: Per-layer KDA cache view (per-row slots).
            query_start_loc: Cumulative segment offsets ``[rows + 1]``.

        Returns:
            :class:`AttentionLayerOutput` with the packed output row and the
            updated cache view.
        """
        if query_start_loc.ndim == 2:  # rank-major DP: [dp, rows + 1]
            query_start_loc = query_start_loc[0]
        rows = query_start_loc.shape[0] - 1
        seq = hidden_states.shape[1]
        starts = query_start_loc[:-1]
        lens = query_start_loc[1:] - starts  # [rows]

        q_row, k_row, v_row = query[0], key[0], value[0]
        b_row, g_row = beta[0], decay[0]

        # Index-clamped gather (NOT start-clamped slicing): the token bucket
        # is padded to the compile width, so `starts[i] + seq` routinely
        # overruns the buffer and a start-clamped dynamic_slice would clamp
        # back to 0, silently feeding row 0's tokens to every later row.
        def gather(x):
            idx = jnp.clip(starts[:, None] + ar_post[None, :], 0, x.shape[0] - 1)
            return x[idx]

        ar_post = jnp.arange(seq)
        q_r, k_r, v_r = gather(q_row), gather(k_row), gather(v_row)
        b_r, g_r = gather(b_row), gather(g_row)

        ar = jnp.arange(seq)
        valid = (ar[None, :] < lens[:, None]).astype(q_r.dtype)  # [rows, T]

        def mask(x):
            shape = valid.shape + (1,) * (x.ndim - 2)
            return jnp.where(valid.reshape(shape), x, 0.0)

        q_r, k_r, v_r, b_r, g_r = mask(q_r), mask(k_r), mask(v_r), mask(b_r), mask(g_r)

        # ---- causal depthwise conv per row -----------------------------
        conv_dtype = jnp.bfloat16 if self.dtype in lowfloats else self.dtype
        conv_states = {
            "q": cache_view.q_conv_state,
            "k": cache_view.k_conv_state,
            "v": cache_view.v_conv_state,
        }
        windows = {"q": q_r, "k": k_r, "v": v_r}
        new_conv = {}
        conv_outs = {}
        # Carried-window rows: any row scheduled tokens beyond a fresh
        # prefill (decode steps and chunked prefill continuations).
        # ``context_lens`` counts tokens *including* this step, so a row
        # with prior context satisfies context_lens > lens.
        ctx = getattr(cache_metadata, "context_lens", None)
        if ctx is None:
            has_prefix = lens <= 1
        else:
            has_prefix = ctx > lens
        for name in ("q", "k", "v"):
            conv1d = getattr(self, f"{name}_conv1d")
            kern = conv1d.weight.value.squeeze(1).T  # [d, d_conv]
            x_rows = windows[name].astype(jnp.float32)
            st = conv_states[name].astype(jnp.float32)  # [rows, d, d_conv]
            d_conv = st.shape[-1]
            prefix = jnp.where(has_prefix[:, None, None], st[:, :, 1:], 0.0).transpose(0, 2, 1)  # [rows, dc-1, d]
            stream = jnp.concatenate([prefix, x_rows], axis=1)  # [rows, T + dc - 1, d]
            out = jnp.zeros_like(stream[:, :seq, :])
            for m in range(d_conv):
                out = out + stream[:, m : m + seq, :] * kern[:, m][None, None, :]
            out = jax.nn.silu(out).astype(conv_dtype)
            conv_outs[name] = out
            # New state = the trailing window ending at this row's LAST real
            # token (stream offset dc-1+lens), not the padded bucket tail.
            win_start = jnp.clip(lens - 1, 0, seq - 1)  # [rows]
            win_idx = win_start[:, None] + jnp.arange(d_conv)[None, :]
            candidate = stream[jnp.arange(rows)[:, None], win_idx].transpose(0, 2, 1)  # [rows, d, d_conv]
            keep = (lens > 0)[:, None, None]
            new_conv[name] = jnp.where(keep, candidate, st).astype(conv_states[name].dtype)

        # ---- delta rule per row ----------------------------------------
        num_heads = b_r.shape[-1]
        head_k = q_r.shape[-1] // num_heads
        head_v = v_r.shape[-1] // num_heads
        qh = q_r.reshape(rows, seq, num_heads, head_k).transpose(0, 2, 1, 3)  # [R, H, T, K]
        kh = k_r.reshape(rows, seq, num_heads, head_k).transpose(0, 2, 1, 3)
        vh = v_r.reshape(rows, seq, num_heads, head_v).transpose(0, 2, 1, 3)
        bh = b_r.transpose(0, 2, 1)  # [R, H, T]
        gh = g_r.reshape(rows, seq, num_heads, head_k).transpose(0, 2, 1, 3)
        init_state = cache_view.recurrent_state  # [R, H, K, V]

        all_single = jnp.all(lens <= 1)

        def _single_path():
            # qh/kh/vh are [R, H, T, D]; the kernel contract is BTHD
            # [batch, seq=1, heads, dim] — take seq position 0 (each row's
            # only token) and transpose the head axis into place.
            out1, new_state = _single_step_kda_per_channel_fwd_bthd(
                query=qh[:, :, :1, :].transpose(0, 2, 1, 3),
                key=kh[:, :, :1, :].transpose(0, 2, 1, 3),
                value=vh[:, :, :1, :].transpose(0, 2, 1, 3),
                beta=bh[:, :, :1].transpose(0, 2, 1),
                decay=gh[:, :, :1, :].transpose(0, 2, 1, 3),
                recurrent_state=init_state,
                use_qk_l2norm=True,
            )
            # unify with the chunk path: [R, H, T, V] float32
            out1 = jnp.transpose(out1, (0, 2, 1, 3)).astype(jnp.float32)
            out1 = jnp.pad(out1, ((0, 0), (0, 0), (0, seq - 1), (0, 0)))
            return out1, new_state.astype(jnp.float32)

        def _chunk_path():
            return _chunked_scan_per_channel_rows(
                query=qh,
                key=kh,
                value=vh,
                beta=bh,
                decay=gh,
                initial_state=init_state,
                token_valid=valid,
                chunk_size=_KDA_CHUNK_SIZE,
            )

        out_h, new_recurrent = jax.lax.cond(all_single, _single_path, _chunk_path)

        # scatter per-row outputs back into the packed stream via gather
        head_v_dim = vh.shape[-1]
        out_rows = out_h.transpose(0, 2, 1, 3).reshape(rows, seq, num_heads, head_v_dim)  # [R, T, H, V]
        seg_ids = jnp.sum(query_start_loc[1:, None] <= ar[None, :], axis=0)  # [T]
        seg_ids = jnp.clip(seg_ids, 0, rows - 1)
        pos_in_row = jnp.clip(ar - query_start_loc[seg_ids], 0, seq - 1)
        out_packed = out_rows[seg_ids, pos_in_row].astype(hidden_states.dtype)[None]  # [1, T, H, V]

        output_gate = self.g_b_proj(self.g_a_proj(hidden_states))
        output_gate = output_gate.reshape(1, seq, num_heads, head_v_dim)
        output = self.o_norm(out_packed, output_gate)
        output = output.reshape(1, seq, -1)
        output = checkpoint_name(self.o_proj(output), name="attn_output")

        new_cache_view = cache_view.update_kda_states(
            new_q_conv_state=new_conv["q"],
            new_k_conv_state=new_conv["k"],
            new_v_conv_state=new_conv["v"],
            new_recurrent_state=new_recurrent.astype(cache_view.recurrent_state.dtype),
        )
        return AttentionLayerOutput(
            attention_output=output,
            attention_weight=None,
            cache_view=new_cache_view,
        )


class Glm5NextIndexer(SparseIndexer):
    """GLM-5 k-pool DSA indexer on the unified :class:`SparseIndexer` layer.

    Thin configuration adapter: the projection plumbing, pool compression,
    ReLU-weighted scoring, top-k selection and tail-pool expansion all live in
    the shared :class:`~easydel.layers.indexer.SparseIndexer`; this subclass
    only maps the model config onto :class:`IndexerConfig`. Checkpoint
    parameter names (``wq_b`` / ``wk`` / ``k_norm`` / ``weights_proj`` /
    ``index_kpool_compress_ape`` / ``index_kpool_compress_gate``) are owned by
    the parent class and are unchanged.
    """

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
            ),
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )


class Glm5NextDSAAttention(UnifiedAttention):
    """Multi-head Latent Attention (NoPE) with k-pool DSA sparse gating.

    Standard DeepSeek MLA: queries factor through the ``q_lora_rank``
    bottleneck; keys/values through the compressed ``kv_lora_rank`` latent
    (``kv_a_proj_with_mqa`` + ``kv_a_layernorm`` + ``kv_b_proj``). GLM-5 is
    fully NoPE (``qk_rope_head_dim=0``): there is no rotary module and the
    whole head dim is the non-positional slice.

    The indexer (:class:`Glm5NextIndexer`) restricts each query's attention
    to the selected ``index_topk`` (+ tail pool) tokens by ANDing a
    scatter mask into the pairwise ``mask_info`` attention mask.

    Cache: full-attention KV pages via ``UnifiedAttention.concatenate``; the
    indexer's packed per-token states ride the same view's
    ``recurrent_state`` slot (unused by full-attention layers).

    The ``projection_mapping`` class variable rewrites attribute names into
    HF-checkpoint-compatible names (``q_a_proj``, ``kv_a_proj_with_mqa``,
    ``indexer.wq_b``, ...).
    """

    projection_mapping: ClassVar[dict[str, str]] = {
        "mla_q_a_proj": "q_a_proj",
        "mla_q_a_layernorm": "q_a_layernorm",
        "mla_q_b_proj": "q_b_proj",
        "mla_kv_a_proj_with_mqa": "kv_a_proj_with_mqa",
        "mla_kv_a_layernorm": "kv_a_layernorm",
        "mla_kv_b_proj": "kv_b_proj",
        "output_projection": "o_proj",
        "dsa_indexer_wq_b": "indexer.wq_b",
        "dsa_indexer_wk": "indexer.wk",
        "dsa_indexer_k_norm": "indexer.k_norm",
        "dsa_indexer_weights_proj": "indexer.weights_proj",
    }

    def __init__(
        self,
        config: Glm5NextTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize the MLA + DSA attention block.

        Args:
            config: Model configuration.
            dtype: Computation dtype.
            param_dtype: Parameter dtype.
            precision: JAX matmul precision.
            rngs: Random number generator collection.
            layer_idx: Index of this layer in the decoder stack.
        """
        self.config = config
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.index_topk = config.index_topk

        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="mla",
            causal=True,
            use_mla_lora=True,
        )
        self.head_dim = self.v_head_dim

    def define_network(
        self,
        config: Glm5NextTextConfig,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        precision: jax.lax.Precision,
        rngs: spx.Rngs,
    ):
        """Creates the MLA projections, attention performer, and k-pool indexer.

        NoPE model: no rotary embedding is created.

        Args:
            config: Model configuration.
            dtype: Computation dtype.
            param_dtype: Parameter storage dtype.
            precision: JAX matmul precision.
            rngs: PRNG key container.
        """
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
                eps=1e-6,
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
                eps=1e-6,
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
        self.attention_performer = self._create_attention_performer(config, rngs)
        self.indexer = Glm5NextIndexer(
            config=config,
            layer_idx=self.layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def _create_attention_performer(self, config, rngs):
        """Construct the flexible attention kernel (NoPE — no rope scaling).

        Args:
            config: Model configuration.
            rngs: Random number generator collection.

        Returns:
            A :class:`FlexibleAttentionModule` configured for MLA + DSA.
        """
        softmax_scale = self.q_head_dim**-0.5
        return FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=softmax_scale,
            dropout_prob=getattr(config, "attention_dropout", 0.0),
        )

    @property
    def reform_param(self):
        """No extra reform rules (all projections load with plain transposes).

        Returns:
            dict: Empty mapping.
        """
        return {}

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
    ):
        """Run the NoPE MLA forward with the k-pool DSA top-k mask.

        Args:
            hidden_states: Layer input ``(batch, seq_len, hidden_size)``.
            mask_info: Attention mask metadata.
            position_ids: Position indices (unused — NoPE; kept for the
                UnifiedAttention call contract).
            mode: Runtime mode propagated to the kernel.
            cache_view: KV cache view (full-attention hybrid or transformer).
            cache_metadata: Cache metadata.
            output_attentions: Whether to return attention weights.
            frequencies: Unused (NoPE).
            alibi: Unused.

        Returns:
            :class:`AttentionLayerOutput` with the attention output, optional
            weights, and the updated cache view (with the indexer packed
            states re-stashed in ``recurrent_state``).
        """
        del position_ids, frequencies, alibi
        bsz, q_len, _ = hidden_states.shape

        q_resid = self.mla_q_a_layernorm(checkpoint_name(self.mla_q_a_proj(hidden_states), name="attn_query_a"))
        query_states = checkpoint_name(self.mla_q_b_proj(q_resid), name="attn_query")
        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)

        compressed_kv = self.mla_kv_a_proj_with_mqa(hidden_states)
        compressed_kv = self.mla_kv_a_layernorm(compressed_kv[..., : self.kv_lora_rank])
        kv = (
            self.mla_kv_b_proj(compressed_kv)
            .reshape(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(0, 2, 1, 3)
        )
        key_states = kv[..., : self.qk_nope_head_dim]
        value_states = kv[..., self.qk_nope_head_dim :]

        # BTHD at the op boundary (matches the concatenate contract).
        query_states = query_states.transpose(0, 2, 1, 3)
        key_states = key_states.transpose(0, 2, 1, 3)
        value_states = value_states.transpose(0, 2, 1, 3)

        cached_packed_states = None
        has_recurrent_slot = cache_view is not None and hasattr(cache_view, "recurrent_state")
        if has_recurrent_slot:
            maybe_cached = getattr(cache_view, "recurrent_state", None)
            if (
                maybe_cached is not None
                and getattr(maybe_cached, "ndim", 0) == 3
                and maybe_cached.shape[0] == bsz
                and maybe_cached.shape[-1] == self.indexer.packed_state_dim
            ):
                cached_packed_states = maybe_cached

        q_mask = None
        if mask_info is not None:
            q_mask = typing.cast("Array | None", mask_info.q_attention_mask)
            if q_mask is not None and q_mask.shape[1] != q_len:
                q_mask = q_mask[:, :q_len]
            q_mask = q_mask.astype(bool)

        indexer_out = self.indexer(
            hidden_states=hidden_states,
            q_resid=q_resid,
            attention_mask=q_mask,
            cached_packed=cached_packed_states,
        )
        topk_indices, packed_states = indexer_out.topk_indices, indexer_out.packed_state

        if has_recurrent_slot and hasattr(cache_view, "replace"):
            try:
                cache_view = cache_view.replace(recurrent_state=packed_states.astype(cache_view.key.dtype))
            except TypeError:
                logger.warning_once(
                    "Failed to store indexer packed states in cache_view.recurrent_state; "
                    "cache structure may not support recurrent_state."
                )

        causal_for_kernel = self.causal
        if mask_info is not None and getattr(mask_info, "_causal_baked", False):
            causal_for_kernel = False
        sliding_window_for_kernel = self.sliding_window
        if mask_info is not None and getattr(mask_info, "sliding_window_baked_in", False):
            sliding_window_for_kernel = None

        # The MLA ragged-pages kernel consumes the weight-absorbed form
        # (q_nope @ W_k, latent kv) rather than decompressed per-head
        # states; the latent (compressed_kv + rope) is what the
        # MLARaggedPagesCacheView stores. GLM-5 is NoPE, so the rope
        # components carry no information — they are zero-padded to the
        # cache rope width (``mla_cache_rope_width``, 128-aligned for the
        # Pallas MLA kernel), which contributes nothing to the scores.
        absorbed_w_v = None
        mla_kwargs: dict = {}
        if isinstance(cache_view, MLARaggedPagesCacheView):
            w = self.mla_kv_b_proj.weight.value
            local_heads = w.shape[1] // (self.qk_nope_head_dim + self.v_head_dim)
            w = w.reshape(self.kv_lora_rank, local_heads, self.qk_nope_head_dim + self.v_head_dim)
            w_nope = w[:, :, : self.qk_nope_head_dim]
            absorbed_w_v = w[:, :, self.qk_nope_head_dim :]

            q_nope = query_states  # [B, S, N, qk_nope] (BTHD)
            q_absorbed = jnp.einsum(
                "bsnd,knd->bsnk",
                q_nope.astype(jnp.float32),
                w_nope.astype(jnp.float32),
            ).astype(q_nope.dtype)

            rope_width = int(getattr(self.config, "mla_cache_rope_width", 0) or 0)
            query_states = q_absorbed  # [B, S, N, kv_lora_rank]
            key_states = jnp.concatenate(
                [compressed_kv[:, :, None, :], jnp.zeros((bsz, q_len, 1, rope_width), compressed_kv.dtype)],
                axis=-1,
            )
            value_states = key_states

            mla_kwargs = {
                "queries_nope": query_states,
                "queries_pe": jnp.zeros((bsz, q_len, self.num_heads, rope_width), query_states.dtype),
                "keys_values": compressed_kv,
                "keys_pe": jnp.zeros((bsz, q_len, rope_width), compressed_kv.dtype),
                "softmax_scale": (self.qk_nope_head_dim + self.config.qk_rope_head_dim) ** -0.5,
            }

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

        if topk_indices is not None:
            kv_len = key_states.shape[1]
            if q_len == kv_len or cached_packed_states is not None:
                topk_mask = jnp.any(jax.nn.one_hot(topk_indices, kv_len, dtype=jnp.bool_), axis=-2)
                attention_mask = (
                    pairwise_attention_mask_from_mask_info(mask_info, q_len, kv_len) if mask_info is not None else None
                )
                if attention_mask is not None:
                    mask_info = mask_info.replace(attention_mask=(attention_mask & topk_mask)[:, None, :, :])

        softmax_aux = self._softmax_aux()

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
            **mla_kwargs,
        )

        attn_out = attentions.attention_outputs
        if absorbed_w_v is not None and attn_out.ndim == 3:
            # Kernel output: [total_tokens, N, kv_lora_rank] -> project W_v.
            attn_out = jnp.einsum(
                "thk,khv->thv",
                attn_out.astype(jnp.float32),
                absorbed_w_v.astype(jnp.float32),
            ).astype(attn_out.dtype)
            attn_output = attn_out.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        else:
            attn_output = self._merge_heads(attentions.attention_outputs)
        expected_attn_dim = self.num_heads * self.v_head_dim
        if attn_output.shape[-1] != expected_attn_dim:
            actual_attn_dim = int(attn_output.shape[-1])
            if actual_attn_dim > expected_attn_dim:
                attn_output = attn_output[..., :expected_attn_dim]
            else:
                pad_width = [(0, 0)] * attn_output.ndim
                pad_width[-1] = (0, expected_attn_dim - actual_attn_dim)
                attn_output = jnp.pad(attn_output, pad_width)
        attn_output = checkpoint_name(self.output_projection(attn_output), name="attn_output")
        attn_output = self.shard_attention_prod(attn_output)

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


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
        self.rngs = rngs
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
        dtype = hidden_states.dtype

        post, comb, collapsed = self.attn_hc(hidden_states)
        attn_input = self.input_layernorm(collapsed)
        attn_input = apply_logical_sharding(
            attn_input,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        if self.block_type == LINEAR_ATTENTION_LAYER_TYPE:
            attn_outputs = self.self_attn(
                hidden_states=attn_input,
                mask_info=mask_info,
                cache_view=cache_view,
                cache_metadata=cache_metadata,
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
        hidden_states = post.astype(dtype)[..., None] * attn_outputs.attention_output[..., None, :] + jnp.einsum(
            "bsji,bsjd->bsid",
            comb.astype(dtype),
            hidden_states,
            precision=self.precision,
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
        hidden_states = post.astype(dtype)[..., None] * feed_forward_hidden_states[..., None, :] + jnp.einsum(
            "bsji,bsjd->bsid",
            comb.astype(dtype),
            hidden_states,
            precision=self.precision,
        )
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


@register_module(TaskType.BASE_MODULE, config=Glm5NextTextConfig, model_type="glm5_next_text")
class Glm5NextTextModel(EasyDeLBaseModule):
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

        if any(t == "shared" for t in (getattr(config, "indexer_types", None) or [])):
            logger.warning_once(
                "GLM-5-Next config declares 'shared' indexer layers, but this EasyDeL "
                "implementation runs a full k-pool indexer on every DSA layer and does not "
                "reuse a previous full layer's top-k selection. Logits match HF only for "
                "all-'full' indexer schedules (the checkpoint default)."
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
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )

        views = past_key_values.views if past_key_values is not None else None
        has_cache_views = views is not None and any(v is not None for v in views)
        needs_trace_cache = mode == common_types.MODE_DECODE or has_cache_views
        # Heterogeneous stack (KDA vs DSA layers) — always Python-trace.
        trace_layers = self._layer_scan_trace(
            True,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            cache_views=views,
            extra=needs_trace_cache or bool(output_router_logits),
        )
        cache_views = views if trace_layers else None

        def _layer_loop(block, carry):
            """Run one heterogeneous decoder layer inside the stack loop.

            Threads ``(streams, cache_views, all_hidden_states,
            all_attentions, all_router_logits, layer_index)`` through the
            Python-traced loop.

            Args:
                block: The decoder layer module.
                carry: Current loop carry.

            Returns:
                Updated carry ``(streams, cache_views, ..., idx + 1)``.
            """
            hidden_states, cv, all_hidden_states, all_attentions, all_router_logits, idx = carry
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
                )

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

            return hidden_states, cv, all_hidden_states, all_attentions, all_router_logits, idx + 1

        hidden_states, _, all_hidden_states, all_attentions, all_router_logits, _ = self.layers.scan(
            _layer_loop,
            (hidden_states, cache_views, all_hidden_states, all_attentions, all_router_logits, 0),
            trace=trace_layers,
        )

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
        )


@register_module(TaskType.CAUSAL_LM, config=Glm5NextTextConfig, model_type="glm5_next_text")
class Glm5NextForCausalLM(BaseCausalLMModule[Glm5NextTextModel, Glm5NextTextConfig]):  # type: ignore
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

    @staticmethod
    def _checkpoint_key_normalizer(key: str) -> str | None:
        """Rewrite composite `zai-org/GLM-5.3-Flash` checkpoint keys.

        The published checkpoint is the multimodal composite: language-model
        tensors live under ``model.language_model.`` and the vision tower under
        ``model.visual.``. EasyDeL builds the text stack as ``model.layers...``,
        so the ``language_model`` segment is stripped and vision tensors are
        reported as unowned (``None`` — the converter skips them). Text-only
        extractions that already use ``model.layers...`` pass through unchanged
        (idempotent).

        Args:
            key: Raw HF state-dict key.

        Returns:
            The EasyDeL-side key, or ``None`` when the runtime does not own
            the tensor (vision tower).
        """
        if key.startswith("model.visual."):
            return None
        if key.startswith("model.language_model."):
            return "model." + key[len("model.language_model.") :]
        return key


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
