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

"""Spectrax implementation of GLM-4-MoE-Lite.

GLM-4-MoE-Lite is a lightweight MoE variant of GLM-4 that combines
Multi-head Latent Attention (MLA) with a hybrid dense/grouped-MoE FFN
schedule. MLA replaces the full Q/K/V projection set with a low-rank
compression of the KV cache, drastically reducing memory while preserving
representational capacity.

Architectural traits:
    - Multi-head Latent Attention with separate compressed KV cache views.
    - Grouped top-k MoE routing with shared (always-on) experts.
    - First ``first_k_dense_replace`` layers use a dense MLP; remaining
      layers use the MoE block.
    - Pre-norm RMSNorm decoder layers with partial RoPE.

Exports:
    - :class:`Glm4MoeLiteModel`: Backbone returning hidden states.
    - :class:`Glm4MoeLiteForCausalLM`: Decoder LM with optional tied LM head.
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
from easydel.infra.modeling_outputs import DecoderLayerOutput, MoeModelOutput
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
    dense_gate_up_layout,
    gated_mlp_forward,
    moe_down_projection_reform_param,
    moe_fused_gate_up_reform_param,
    split_fused_gate_up_projection,
)
from easydel.layers.attention import FlexibleAttentionModule, UnifiedAttention
from easydel.layers.moe import fp32_router_logits, moe_group_topk_select
from easydel.layers.rotary import yarn_get_mscale
from easydel.modules._base import BaseCausalLMModule

from .glm4_moe_lite_configuration import Glm4MoeLiteConfig


class Glm4MoeLiteMLP(spx.Module):
    """Dense MLP block for GLM-4-MoE-Lite layers.

    Implements the standard gated feedforward network with separate gate and up
    projections, used in dense layers of the GLM-4-MoE-Lite hybrid architecture.
    Supports configurable hidden and intermediate sizes.
    """

    def __init__(
        self,
        config: Glm4MoeLiteConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the dense GLM-4-MoE-Lite MLP block.

        Args:
            config: Model configuration.
            dtype: Computation dtype.
            param_dtype: Parameter dtype.
            precision: JAX matmul precision.
            hidden_size: Optional override for the input/output dim;
                defaults to ``config.hidden_size``. Used by the shared
                expert which widens the model dim by ``n_shared_experts``.
            intermediate_size: Optional override for the inner MLP width;
                defaults to ``config.intermediate_size``.
            rngs: Random number generator collection.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.hidden_size = hidden_size or config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
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
        """Run the dense gated MLP.

        Delegates to :func:`easydel.layers.gated_mlp_forward`, which routes
        supported layouts through the ejkernel fused MLP kernels and runs
        the legacy composition otherwise.

        Args:
            hidden_states: Input residual stream
                ``(batch, seq_len, hidden_size)``.

        Returns:
            Transformed hidden states with the same shape as the input.
        """
        return gated_mlp_forward(self, hidden_states)


class Glm4MoeLiteMLPStack(spx.Module):
    """Expert MLP stack for GLM-4-MoE-Lite using parallel MoE linear layers.

    Implements the feedforward network for multiple experts using efficient
    batched computation with ColumnParallelMoELinear and RowParallelMoELinear
    layers. Supports expert tensor mode for optimized expert computation.
    """

    def __init__(
        self,
        config: Glm4MoeLiteConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the routed-expert MLP stack.

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
        self.gate_up_proj = ColumnParallelMoELinear(
            num_experts=config.n_routed_experts,
            in_features=config.hidden_size,
            out_features=2 * config.moe_intermediate_size,
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
            kernel_init=jax.nn.initializers.normal(),
            use_bias=False,
            partition_manager=config.runtime_sharding_resolver,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    @property
    def reform_param(self):
        return {
            **moe_fused_gate_up_reform_param(config=self.config),
            **moe_down_projection_reform_param(),
        }

    def forward(
        self,
        hidden_states: Array,
        group_sizes: Array,
        sorted_experts: Array | None = None,
    ) -> Array:
        """Run the routed gated MLP across all experts at once.

        Args:
            hidden_states: Token rows already permuted by expert
                assignment.
            group_sizes: Per-expert token counts feeding the underlying
                grouped-matmul kernel.
            sorted_experts: Optional sorted expert indices threaded
                through the parallel MoE kernels.

        Returns:
            Expert outputs aligned with the permuted input rows.
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        gate_up = checkpoint_name(self.gate_up_proj(hidden_states, group_sizes, sorted_experts), name="mlp_gate_up")
        gate, up = split_fused_gate_up_projection(gate_up, config=self.config)
        return typing.cast(
            Array,
            apply_logical_sharding(
                checkpoint_name(self.down_proj(self.act_fn(gate) * up, group_sizes, sorted_experts), name="mlp_down"),
                dynamic_axes=common_types.HiddenStateSharding,
                partition_manager=self.config.runtime_sharding_resolver,
            ),
        )


class Glm4MoeLiteTopKRouter(spx.Module):
    """fp32 router projection feeding the grouped top-k selector in :class:`Glm4MoeLiteMoE`.

    Owns only the *projection* parameters — the actual grouped top-k logic
    lives in :func:`easydel.layers.moe.moe_group_topk_select`. The two stored
    parameters are:

    - ``weight``: ``(hidden_size, n_routed_experts)`` learned matrix used to
      produce the per-expert pre-bias logits.
    - ``e_score_correction_bias``: ``(n_routed_experts,)`` learnable
      load-balancing bias that is added to the sigmoid scores at
      *selection* time only — never to the convex combination weights, so
      the bias can drive load balancing without distorting expert
      contributions.

    The matmul is forced into float32 (regardless of activation dtype)
    because the relative gaps between selected and rejected experts can
    fall below bfloat16's resolution near initialisation.
    """

    def __init__(
        self,
        config: Glm4MoeLiteConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the GLM-4-MoE-Lite top-k router projection.

        Args:
            config: Model configuration carrying ``n_routed_experts``.
            dtype: Computation dtype (router matmul is forced to fp32
                regardless).
            param_dtype: Parameter dtype.
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
        # Declared via ArrayParam.bound (as glm4_moe and hy_v3 do) rather than a
        # bare spx.Parameter: under sequential_init the bare form stays an
        # unmaterialised ShapeDtypeStruct, which was harmless only while nothing
        # read it. Routing reads it now.
        self.e_score_correction_bias = ArrayParam.bound(
            shape=(self.n_routed_experts,),
            dtype=jnp.float32,
            init_method="zeros",
            key=rngs.param,
        )

    def forward(self, hidden_states: Float[Array, "tokens hidden_dim"]) -> Array:
        """Project hidden states to per-expert pre-bias logits.

        Args:
            hidden_states: Token activations
                ``(batch * seq_len, hidden_size)`` (or any shape whose
                last dim is ``hidden_size``).

        Returns:
            fp32 router logits of shape
            ``(num_tokens, n_routed_experts)``.
        """
        return fp32_router_logits(hidden_states, self.weight.value, self.config.hidden_size)


class Glm4MoeLiteMoE(BaseMoeModule):
    """Sparse FFN with grouped sigmoid-top-k routing plus an always-on shared expert.

    Identical routing protocol to :class:`Glm4MoeMoE` (DeepSeek-style
    grouped top-k with bias-corrected sigmoid scoring; see that class's
    docstring for the full algorithm) but configured for the *Lite*
    parameter regime: smaller ``n_routed_experts``, smaller
    ``moe_intermediate_size``, and a single layer-uniform routing schedule
    with no first-K-dense replacement at this level (the dense/sparse mix
    is decided one level up at the decoder layer).

    The convex combination is

        ``out = shared(h) + sum_i w_i * routed_i(h)``

    with ``shared`` widened by ``n_shared_experts`` (so the shared branch
    matches the aggregate capacity of the selected routed branch). When
    ``n_shared_experts == 0`` the shared expert is omitted entirely. The
    ``moe_call`` parent dispatch sorts tokens by expert and runs all
    routed-expert kernels through ``grouped_matmul``.
    """

    def __init__(
        self,
        config: Glm4MoeLiteConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the sparse FFN block.

        Args:
            config: Model configuration carrying every routing knob
                (``n_routed_experts``, ``num_experts_per_tok``, ``n_group``,
                ``topk_group``, ``norm_topk_prob``, ``routed_scaling_factor``,
                ``n_shared_experts``, ``moe_intermediate_size``).
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
        )

    def _select_hook(self):
        """Bind the shared grouped top-k selector to this layer's live bias.

        ``e_score_correction_bias`` is the auxiliary-loss-free load-balancing
        term: it steers *selection* without touching the combine weights. It
        has to be read here rather than captured in ``__init__`` -- binding the
        parameter at construction time freezes the zero-initialised tensor, so
        the trained bias never reaches routing. That is exactly the bug this
        replaced: the previous hook took ``pre_bias_logits`` and dropped it.
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
        """Run the sparse FFN with optional shared expert.

        Args:
            hidden_states: Pre-MoE residual ``(batch, seq_len, hidden_size)``.

        Returns:
            Tuple ``(out, router_logits)`` where ``out`` is the combined
            shared + routed expert output and ``router_logits`` are the
            raw fp32 logits used for auxiliary load-balancing losses.
        """
        hooks = self.moe_hooks.replace(select_hook=self._select_hook())
        out, router_logits = self.moe_call(
            hidden_state=hidden_states,
            gate_layer=self.gate,
            expert_layer=self.experts,
            gate_up_kernel=self.experts.gate_up_proj.kernel_view(),
            wd_kernel=self.experts.down_proj.kernel_view(),
            act_fn=self.experts.act_fn,
            hooks=hooks,
        )
        if self.shared_experts is not None:
            out = out + self.shared_experts(hidden_states)
        return checkpoint_name(out, "moe_expert_output"), checkpoint_name(router_logits, "moe_router_logits")


class Glm4MoeLiteAttention(UnifiedAttention):
    """DeepSeek-style Multi-head Latent Attention with split-RoPE heads and YaRN mscale.

    Same MLA layout used in DeepSeek-V2/V3 and the GlmMoeDsaAttention sibling
    class, minus the dynamic-sparse indexer:

    - **Query LoRA** ``hidden -> q_lora_rank -> num_heads * q_head_dim`` when
      ``q_lora_rank`` is set (skipped, i.e., a single dense ``q_proj``, when
      it is ``None``/0).
    - **KV compression** ``hidden -> kv_lora_rank (+ qk_rope_head_dim) ->
      num_kv_heads * (qk_nope + v_head_dim)`` storing only the
      ``kv_lora_rank``-dimensional latent in the KV cache rather than the
      full keys and values.
    - **Split-RoPE heads**: ``q_head_dim = qk_nope_head_dim + qk_rope_head_dim``
      with RoPE applied only to the trailing ``qk_rope_head_dim`` slice; the
      ``qk_nope_head_dim`` slice carries content-only matching. Values use
      a separate ``v_head_dim`` because they need no rotated component.
    - **YaRN mscale** at ``softmax_scale`` time: when long-context RoPE
      scaling is configured, the softmax temperature is multiplied by the
      :func:`yarn_get_mscale` factor so the effective attention temperature
      matches the rescaled RoPE frequencies — this avoids the classic
      attention sharpening that otherwise plagues NTK-style RoPE scaling
      at very long contexts.

    The ``projection_mapping`` class variable rewrites the ``UnifiedAttention``
    factory's logical names into the HF-DeepSeek attribute names so released
    checkpoints load directly.
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
        config: Glm4MoeLiteConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize the MLA attention block.

        Args:
            config: Model configuration carrying every MLA hyper-param
                (``q_lora_rank``, ``kv_lora_rank``, ``qk_nope_head_dim``,
                ``qk_rope_head_dim``, ``v_head_dim``, ``rope_scaling``).
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
        rngs: spx.Rngs,
    ):
        """Build the MLA projection stack.

        The body lived here in four families with byte-identical contents; it
        is now :meth:`UnifiedAttention._build_mla_projections`, driven by this
        class's ``projection_mapping`` so the HF attribute names are unchanged.
        """
        self._build_mla_projections(config, dtype, param_dtype, precision, rngs)

    def _create_attention_performer(self, config, rngs):
        """Construct the flexible attention kernel with YaRN-aware scaling.

        Multiplies the ``softmax_scale`` by the YaRN ``mscale`` factor when
        the config sets a non-zero ``mscale_all_dim`` so the effective
        attention temperature matches the rescaled RoPE frequencies.

        Args:
            config: Model configuration.
            rngs: Random number generator collection.

        Returns:
            A :class:`FlexibleAttentionModule` configured for MLA.
        """
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

    # ``forward_mla`` is inherited from :class:`UnifiedAttention`: it builds the q LoRA chain, splits the
    # ``kv_a`` projection, applies (interleaved or split-half) RoPE per ``config.rope_interleave``, reads the
    # softmax scale from ``self.attention_performer`` (the YaRN-aware performer built above), and skips
    # ``kv_b_proj`` on the weight-absorbed ``MLARaggedPagesCacheView`` path.


class Glm4MoeLiteDecoderLayer(spx.Module):
    """Single decoder layer for GLM-4-MoE-Lite.

    Combines Multi-head Latent Attention (MLA) and either dense MLP or MoE
    feedforward networks with RMS normalization and residual connections.
    Uses dense MLP for layers marked as "dense" in mlp_layer_types and
    sparse MoE for layers marked as "sparse".
    """

    def __init__(
        self,
        config: Glm4MoeLiteConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize one decoder block.

        Picks ``Glm4MoeLiteMoE`` for layers marked ``"sparse"`` in
        ``config.mlp_layer_types`` (and only when MoE is fully configured)
        and ``Glm4MoeLiteMLP`` otherwise.

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

        mlp_moe_block = Glm4MoeLiteMoE

        self.self_attn = Glm4MoeLiteAttention(
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
            self.mlp = Glm4MoeLiteMLP(
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
        """Pre-norm decoder block: ``x + attn(norm(x))`` then ``x + ffn(norm(x))``.

        The feedforward branch returns a ``(out, router_logits)`` tuple
        for sparse layers and a plain tensor for dense layers; the router
        logits are forwarded only for sparse layers.

        Args:
            hidden_states: Residual ``(batch, seq_len, hidden_size)``.
            mask_info: Attention mask metadata.
            position_ids: Position indices for RoPE.
            mode: Runtime mode propagated to attention.
            cache_view: KV cache view, if any.
            cache_metadata: Cache metadata, if any.
            output_attentions: Whether to expose attention weights.
            frequencies: Optional pre-computed RoPE table.

        Returns:
            :class:`DecoderLayerOutput` carrying hidden states, optional
            attention weights, the (possibly updated) cache view, and
            optional router logits for MoE layers.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
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
        if self.config.use_scan_mlp and isinstance(self.mlp, Glm4MoeLiteMLP):
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
        hidden_states = residual + feed_forward_hidden_states
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


@register_module(TaskType.BASE_MODULE, config=Glm4MoeLiteConfig, model_type="glm4_moe_lite")
class Glm4MoeLiteModel(EasyDeLBaseModule):
    """GLM-4-MoE-Lite base model implementation.

    Implements the GLM-4-MoE-Lite architecture, a lightweight sparse mixture-of-experts
    transformer model featuring Multi-head Latent Attention (MLA) with low-rank KV/Q
    compression, grouped top-k MoE routing with shared experts, and a configurable
    dense-to-sparse MLP schedule across layers.
    """

    def __init__(
        self,
        config: Glm4MoeLiteConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the GLM-4-MoE-Lite backbone (embeddings, layers, final norm).

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
            Glm4MoeLiteDecoderLayer,
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
        """Cached RoPE cos/sin table sized to ``qk_rope_head_dim``.

        MLA only rotates the ``qk_rope_head_dim`` slice, so the table is
        narrower than the full attention head dim.
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
    ) -> MoeModelOutput:
        """Run the GLM-4-MoE-Lite backbone.

        Args:
            input_ids: Token ids ``(batch, seq_len)``. Mutually exclusive
                with ``inputs_embeds``.
            inputs_embeds: Pre-computed embeddings ``(batch, seq_len,
                hidden_size)``.
            attention_mask: Optional ``(batch, seq_len)`` boolean padding
                mask.
            mask_info: Optional pre-computed attention mask metadata.
            position_ids: Optional ``(batch, seq_len)`` RoPE positions.
            mode: Runtime mode; defaults to MODE_DECODE when seq_len==1
                and a cache is provided, otherwise MODE_TRAIN.
            past_key_values: Existing KV cache to update.
            cache_metadata: Cache metadata for paged attention.
            output_attentions: Whether to collect attention weights.
            output_hidden_states: Whether to collect intermediate hidden
                states.
            output_router_logits: Whether to collect router logits from
                MoE layers.

        Returns:
            :class:`MoeModelOutput` with the final hidden state, optional
            collections of intermediate states/attentions/router logits,
            and the (updated) KV cache.

        Raises:
            ValueError: If neither/both of ``input_ids``/``inputs_embeds``
                are provided, or if the sequence exceeds
                ``max_position_embeddings``.
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
            partition_manager=self.config.runtime_sharding_resolver,
        )

        views = past_key_values.views if past_key_values is not None else None
        has_cache_views = views is not None and any(v is not None for v in views)
        needs_trace_cache = mode == common_types.MODE_DECODE or has_cache_views
        # Router-logit aggregation grows a Python tuple per layer, which
        # lax.scan cannot carry — collecting it forces the trace path too.
        trace_layers = self._layer_scan_trace(
            False,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            cache_views=views,
            extra=needs_trace_cache or bool(output_router_logits),
        )
        cache_views = views if trace_layers else None
        # Hoisted out of the loop body: the caching property must not be
        # first materialized inside the lax.scan trace (tracer leak).
        frequencies = self.frequencies

        def _layer_loop(block, carry):
            """Run one MLA decoder layer inside ``self.layers.scan``.

            Threads ``(hidden_states, cv, all_hidden_states, all_attentions,
            all_router_logits, layer_index)`` through the scan and
            collects optional outputs requested by the outer ``forward``.
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
                    frequencies=frequencies,
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
            **self._layer_scan_kwargs("text"),
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


@register_module(TaskType.CAUSAL_LM, config=Glm4MoeLiteConfig, model_type="glm4_moe_lite")
class Glm4MoeLiteForCausalLM(BaseCausalLMModule[Glm4MoeLiteModel, Glm4MoeLiteConfig]):  # type: ignore
    """GLM-4-MoE-Lite model with a language modeling head for causal language modeling.

    Combines the GLM-4-MoE-Lite base model with a linear language modeling head
    for autoregressive text generation. Uses the sparse MoE architecture with
    MLA attention for efficient inference and training.
    """

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
        rngs: spx.Rngs,
    ):
        """Initialize the causal-LM wrapper around :class:`Glm4MoeLiteModel`.

        Args:
            config: Model configuration.
            dtype: Computation dtype.
            param_dtype: Parameter dtype.
            precision: JAX matmul precision.
            rngs: Random number generator collection.
        """
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
        max_cache_tokens: int | None = None,
    ):
        """Create MLA or expanded-KV pages, also for config-only callers."""
        text_config = self.config.get_text_config()
        mechanism = getattr(text_config.attn_mechanism, "value", text_config.attn_mechanism)
        if mechanism in ("multi_latent_ragged_page_attention_v1", "multi_latent_ragged_page_attention_v2"):
            return self._create_mla_ragged_page_cache_config(
                max_length=max_length,
                page_size=page_size,
                hbm_utilization=hbm_utilization,
                dtype=dtype,
                max_cache_tokens=max_cache_tokens,
            )
        return Glm4MoeLiteForCausalLM._create_standard_ragged_page_cache_config(
            self,
            max_length=max_length,
            page_size=page_size,
            hbm_utilization=hbm_utilization,
            dtype=dtype,
            max_cache_tokens=max_cache_tokens,
        )

    def _create_standard_ragged_page_cache_config(
        self,
        max_length: int,
        *,
        page_size: int = 128,
        hbm_utilization: float = 0.9,
        dtype: jnp.dtype | None = None,
        num_hidden_layers_override: int | None = None,
        max_cache_tokens: int | None = None,
    ):
        """Size a non-MLA paged cache for the decompressed key/value heads.

        The inherited public builder handles attention-mechanism dispatch and
        MLA latent-cache geometry. Only the standard-cache geometry differs:
        keys have ``qk_nope_head_dim + qk_rope_head_dim`` channels and values
        have ``v_head_dim`` channels, with ``num_attention_heads`` KV heads.

        Args:
            max_length: Maximum model context length.
            page_size: Tokens per page.
            hbm_utilization: Fraction of HBM the cache may consume.
            dtype: Optional cache dtype override.
            num_hidden_layers_override: Explicit number of cache-bearing layers.
            max_cache_tokens: Optional hard ceiling on total KV-cache tokens.

        Returns:
            A paged cache configuration matching decompressed MLA heads.
        """
        from easydel.caching import RaggedPagesCacheConfig
        from easydel.layers.attention import AttentionMechanisms

        text_config = self.config.get_text_config()
        q_head_dim = text_config.qk_nope_head_dim + text_config.qk_rope_head_dim
        if dtype is None:
            dtype = text_config.kvdtype
            if isinstance(dtype, str):
                dtype = getattr(jnp, dtype, jnp.bfloat16)

        if num_hidden_layers_override is not None:
            num_hidden_layers = num_hidden_layers_override
        elif hasattr(self, "_standard_ragged_layer_indices_for_cache_budget") and hasattr(
            self, "_pp_stage_cache_layer_count"
        ):
            layer_indices = self._standard_ragged_layer_indices_for_cache_budget()
            num_hidden_layers = self._pp_stage_cache_layer_count(
                layer_indices,
                total_layers=text_config.num_hidden_layers,
            )
        else:
            # The public builder also accepts config/mesh-only callers.
            num_hidden_layers = text_config.num_hidden_layers

        version = "v2" if text_config.attn_mechanism == AttentionMechanisms.RAGGED_PAGE_ATTENTION_V2 else "v3"
        return RaggedPagesCacheConfig.create(
            mesh=text_config.mesh,
            runtime_sharding_resolver=text_config.runtime_sharding_resolver,
            kvdtype=dtype,
            max_model_length=max_length,
            num_hidden_layers=max(1, int(num_hidden_layers)),
            num_kv_heads=text_config.num_attention_heads,
            kv_head_dim_size=q_head_dim,
            k_headdim=q_head_dim,
            v_headdim=text_config.v_head_dim,
            hbm_utilization=hbm_utilization,
            page_size=page_size,
            max_cache_tokens=max_cache_tokens,
            version=version,
        )


__all__ = ["Glm4MoeLiteForCausalLM", "Glm4MoeLiteModel"]
