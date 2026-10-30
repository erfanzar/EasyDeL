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

"""Spectrax implementation of GLM-MoE-DSA.

GLM-MoE-DSA combines Multi-head Latent Attention (MLA), a Dynamic Sparse
Attention (DSA) indexer that selects top-k key tokens per query, and a
hybrid dense/grouped-MoE FFN schedule. It is targeted at extremely long
contexts where full attention becomes the bottleneck.

Architectural traits:
    - Multi-head Latent Attention with low-rank KV compression and
      compressed KV cache views for inference.
    - DSA indexer producing per-query top-k key indices used to mask the
      attention computation to a sparse subset.
    - First ``first_k_dense_replace`` layers are dense; remaining layers
      use grouped MoE with shared and routed experts.
    - Pre-norm RMSNorm decoder layers with partial RoPE.

Exports:
    - :class:`GlmMoeDsaModel`: Backbone returning hidden states.
    - :class:`GlmMoeDsaForCausalLM`: Decoder LM with optional tied LM head.
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
    RaggedPagesCacheView,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import DecoderLayerOutput, MoeModelOutput, SparseDecoderLayerOutput
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
from easydel.layers.attention import DeepSeekSparseAttention
from easydel.layers.indexer import (
    IndexerLayerNorm,
    TokenIndexer,
    TokenIndexerConfig,
    warn_unsupported_scoring_kernel,
)
from easydel.layers.moe import fp32_router_logits, moe_group_topk_select
from easydel.modules._base import BaseCausalLMModule

from .glm_moe_dsa_configuration import GlmMoeDsaConfig

logger = get_logger(__name__)

# Public family name, shared with the common implementation.
GlmMoeDsaLayerNorm = IndexerLayerNorm


class GlmMoeDsaMLP(spx.Module):
    """Dense gated MLP block used in the first few (dense) layers of GLM-MoE-DSA.

    Applies a SiLU-gated feed-forward network: ``down_proj(act(gate_proj(x)) * up_proj(x))``.

    Args:
        config: Model configuration.
        dtype: Computation dtype.
        param_dtype: Parameter storage dtype.
        precision: JAX matmul precision.
        hidden_size: Override for ``config.hidden_size``.
        intermediate_size: Override for ``config.intermediate_size``.
        rngs: PRNG key container.
    """

    def __init__(
        self,
        config: GlmMoeDsaConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the dense MLP block.

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
        """Applies the gated MLP transformation.

        Delegates to :func:`easydel.layers.gated_mlp_forward`, which routes
        supported layouts through the ejkernel fused MLP kernels and runs
        the legacy composition otherwise.

        Args:
            hidden_states: Input tensor of shape ``(batch, seq_len, hidden_dim)``.

        Returns:
            Transformed hidden states with the same shape as input.
        """
        return gated_mlp_forward(self, hidden_states)


class GlmMoeDsaMLPStack(spx.Module):
    """Stacked expert MLP kernels for GLM-MoE-DSA MoE layers.

    Each expert applies the same gated-MLP architecture as :class:`GlmMoeDsaMLP`
    but parameters are stored as 3-D tensors ``[num_experts, ...]`` so that all
    experts can be evaluated in a single fused operation.

    ``reform_param`` defines the weight-splitting rules used when converting
    between HuggingFace and EasyDeL parameter layouts.

    Args:
        config: Model configuration.
        dtype: Computation dtype.
        param_dtype: Parameter storage dtype.
        precision: JAX matmul precision.
        rngs: PRNG key container.
    """

    def __init__(
        self,
        config: GlmMoeDsaConfig,
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
        """Applies the gated MLP across all experts in a fused manner.

        Args:
            hidden_states: Token representations, pre-sorted by expert assignment.
            group_sizes: Number of tokens assigned to each expert.
            sorted_experts: Expert indices for sorted token ordering.

        Returns:
            Transformed hidden states after expert-parallel MLP.
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


class GlmMoeDsaTopKRouter(spx.Module):
    """Routing gate for GLM-MoE-DSA grouped top-k expert selection.

    Projects hidden states into per-expert logits via a learned weight matrix
    and adds a per-expert score-correction bias.

    Args:
        config: Model configuration.
        dtype: Computation dtype.
        param_dtype: Parameter storage dtype.
        precision: JAX matmul precision.
        rngs: PRNG key container.
    """

    def __init__(
        self,
        config: GlmMoeDsaConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the GLM-MoE-DSA top-k routing gate.

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
        """Computes per-expert routing logits for all tokens.

        Args:
            hidden_states: Flattened token representations.

        Returns:
            Router logits of shape ``(tokens, n_routed_experts)``.
        """
        return fp32_router_logits(hidden_states, self.weight.value, self.config.hidden_size)


class GlmMoeDsaMoE(BaseMoeModule):
    """Mixture-of-Experts feed-forward block for GLM-MoE-DSA sparse layers.

    Routes tokens through ``num_experts_per_tok`` routed experts selected via
    grouped top-k gating with sigmoid scoring and optional shared experts that
    process every token unconditionally.

    Args:
        config: Model configuration.
        dtype: Computation dtype.
        param_dtype: Parameter storage dtype.
        precision: JAX matmul precision.
        rngs: PRNG key container.
    """

    def __init__(
        self,
        config: GlmMoeDsaConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the GLM-MoE-DSA sparse FFN.

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
        self.group_topk_k = min(2, (config.n_routed_experts or 0) // max(config.n_group, 1))

        self.experts = GlmMoeDsaMLPStack(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.gate = GlmMoeDsaTopKRouter(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.shared_experts = (
            GlmMoeDsaMLP(
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
        """Routes tokens through selected experts and combines outputs.

        Args:
            hidden_states: Input tensor of shape ``(batch, seq_len, hidden_dim)``.

        Returns:
            Tuple of (combined expert output, router logits).
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


class GlmMoeDsaIndexer(TokenIndexer):
    """GLM-MoE-DSA config/norm adapter for the operation-backed token indexer."""

    def __init__(
        self,
        config: GlmMoeDsaConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Preserve the GLM checkpoint layout and registered operation dispatch."""
        super().__init__(
            TokenIndexerConfig(
                hidden_size=config.hidden_size,
                q_input_dim=config.q_lora_rank if config.q_lora_rank is not None else config.hidden_size,
                num_heads=config.index_n_heads,
                head_dim=config.index_head_dim,
                topk=config.index_topk,
                rope_dim=config.qk_rope_head_dim,
                rope_interleave=bool(getattr(config, "indexer_rope_interleave", False)),
                initializer_range=config.initializer_range,
            ),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            base_config=config,
            norm_factory=partial(
                IndexerLayerNorm, config.index_head_dim, eps=1e-6, rngs=rngs, dtype=dtype, param_dtype=param_dtype
            ),
        )
        self.config = config
        warn_unsupported_scoring_kernel(config, "GLM-MoE-DSA token")


class GlmMoeDsaAttention(DeepSeekSparseAttention):
    """GLM-MoE-DSA attention: split-RoPE MLA restricted by the per-token :class:`GlmMoeDsaIndexer`.

    Queries go through the ``q_lora_rank`` bottleneck when it is set; scores use the MLA scale with the YaRN
    ``mscale**2`` correction. The ``projection_mapping`` binds the HF checkpoint attribute names.
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

    def _create_indexer(self, config, dtype, param_dtype, precision, rngs) -> GlmMoeDsaIndexer:
        """This layer's token indexer (see :meth:`DeepSeekSparseAttention._create_indexer`)."""
        return GlmMoeDsaIndexer(config=config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)

    def _mla_norm_eps(self, config) -> float:
        """``1e-6``: the reference builds ``q_a_layernorm`` / ``kv_a_layernorm`` with the RMSNorm default."""
        return 1e-6


class GlmMoeDsaDecoderLayer(spx.Module):
    """Single transformer decoder layer for GLM-MoE-DSA.

    Composes an MLA self-attention block and a feed-forward block (either dense
    :class:`GlmMoeDsaMLP` or sparse :class:`GlmMoeDsaMoE` depending on the
    per-layer ``mlp_layer_types`` schedule) with pre-norm residual connections.

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
        config: GlmMoeDsaConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize one MLA + DSA decoder block.

        Picks :class:`GlmMoeDsaMoE` for layers marked ``"sparse"`` in
        ``config.mlp_layer_types`` (and only when MoE is fully configured)
        and :class:`GlmMoeDsaMLP` otherwise.

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

        self.self_attn = GlmMoeDsaAttention(
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
            self.mlp = GlmMoeDsaMoE(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        else:
            self.mlp = GlmMoeDsaMLP(
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
        previous_topk_indices: Array | None = None,
        return_topk_indices: bool = False,
    ) -> DecoderLayerOutput:
        """Pre-norm decoder block: ``x + attn(norm(x))`` then ``x + ffn(norm(x))``.

        The feedforward branch returns a ``(out, router_logits)`` tuple
        for sparse layers and a plain tensor for dense layers; router
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

        if return_topk_indices:
            attn_outputs, topk_indices = self.self_attn.forward_mla(
                hidden_states,
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
        if self.config.use_scan_mlp and isinstance(self.mlp, GlmMoeDsaMLP):
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

        result = SparseDecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
            cache_view=attn_outputs.cache_view,
            router_logits=router_logits,
            indexer_aux_loss=attn_outputs.indexer_aux_loss,
        )
        return (result, topk_indices) if return_topk_indices else result


class _GlmMoeDsaIndexerCacheMixin:
    """Allocate position-aligned token indexer features with ordinary MLA KV."""

    def create_transformer_cache_config(self, batch_size, max_length, pad_token_id=None, layer_idx=None):
        """Retain standard geometry and request persistent float32 indexer keys."""
        config = super().create_transformer_cache_config(batch_size, max_length, pad_token_id, layer_idx)
        return config.replace(auxiliary_dim=self.config.get_text_config().index_head_dim)


@register_module(TaskType.BASE_MODULE, config=GlmMoeDsaConfig, model_type="glm_moe_dsa")
class GlmMoeDsaModel(_GlmMoeDsaIndexerCacheMixin, EasyDeLBaseModule):
    """GLM-MoE-DSA base transformer model (no language-model head).

    Stacks an embedding layer, ``num_hidden_layers`` decoder layers
    (:class:`GlmMoeDsaDecoderLayer`), and a final RMS normalisation.  Each
    decoder layer uses Multi-head Latent Attention with dynamic sparse
    attention indexing and either a dense or sparse (MoE) feed-forward block.

    Args:
        config: Model configuration.
        dtype: Computation dtype.
        param_dtype: Parameter storage dtype.
        precision: JAX matmul precision.
        rngs: PRNG key container.
    """

    def __init__(
        self,
        config: GlmMoeDsaConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the GLM-MoE-DSA backbone.

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
            GlmMoeDsaDecoderLayer,
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

        Returns:
            tuple[Array, Array]: Cosine and sine RoPE frequency tensors.
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
        """Run the GLM-MoE-DSA backbone.

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
            and the (updated) KV cache. ``indexer_aux_loss`` is the unscaled
            per-full-indexer-layer sum only for stateless forwards with a
            positive ``indexer_aux_loss_coef``; otherwise it is ``None``.

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
        aux_active = not has_cache_views and self.config.indexer_aux_loss_coef > 0.0
        # Router-logit aggregation grows a Python tuple per layer, which
        # lax.scan cannot carry — collecting it forces the trace path too.
        has_shared_indexers = "shared" in self.config.indexer_types
        trace_layers = self._layer_scan_trace(
            has_shared_indexers,
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
            """Run one MLA + DSA decoder layer inside ``self.layers.scan``.

            Threads ``(hidden_states, cache_views, all_hidden_states,
            all_attentions, all_router_logits, layer_index)`` through the
            scan and collects optional outputs requested by the outer
            ``forward``.
            """
            if aux_active:
                carry, indexer_aux = carry[:-1], carry[-1]
            if has_shared_indexers:
                hidden_states, cv, all_hidden_states, all_attentions, all_router_logits, idx, prev_topk = carry
            else:
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

            result = (hidden_states, cv, all_hidden_states, all_attentions, all_router_logits, idx + 1)
            if has_shared_indexers:
                result = (*result, prev_topk)
            if aux_active:
                if layer_outputs.indexer_aux_loss is not None:
                    indexer_aux = indexer_aux + layer_outputs.indexer_aux_loss
                result = (*result, indexer_aux)
            return result

        initial = (hidden_states, cache_views, all_hidden_states, all_attentions, all_router_logits, 0)
        if has_shared_indexers:
            initial = (*initial, None)
        if aux_active:
            initial = (*initial, jnp.zeros((), dtype=jnp.float32))
        final = self.layers.scan(_layer_loop, initial, trace=trace_layers, **self._layer_scan_kwargs("text"))
        hidden_states, _, all_hidden_states, all_attentions, all_router_logits, _ = final[:6]
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return MoeModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            router_logits=all_router_logits,
            past_key_values=past_key_values,
            indexer_aux_loss=final[-1] if aux_active else None,
        )


@register_module(TaskType.CAUSAL_LM, config=GlmMoeDsaConfig, model_type="glm_moe_dsa")
class GlmMoeDsaForCausalLM(_GlmMoeDsaIndexerCacheMixin, BaseCausalLMModule[GlmMoeDsaModel, GlmMoeDsaConfig]):  # type: ignore
    """GLM-MoE-DSA model with a causal language-modelling head.

    Wraps :class:`GlmMoeDsaModel` and adds a linear LM head (no bias) that
    projects the final hidden states to vocabulary logits.

    Args:
        config: Model configuration.
        dtype: Computation dtype.
        param_dtype: Parameter storage dtype.
        precision: JAX matmul precision.
        rngs: PRNG key container.
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "glm_moe_dsa"
    _config_class = GlmMoeDsaConfig

    def __init__(
        self,
        config: GlmMoeDsaConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the causal-LM wrapper around :class:`GlmMoeDsaModel`.

        Args:
            config: Model configuration.
            dtype: Computation dtype.
            param_dtype: Parameter dtype.
            precision: JAX matmul precision.
            rngs: Random number generator collection.
        """
        super().__init__(
            config=config,
            base_model_class=GlmMoeDsaModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
            router_aux_loss_coef=getattr(config, "router_aux_loss_coef", None),
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
        """Create paged cache configuration for MLA attention.

        GLM-MoE-DSA uses MLA, where the runtime attention head width is
        ``qk_nope_head_dim + qk_rope_head_dim`` (not ``config.head_dim``).
        When the configured attention mechanism is MLA-paged, defers to
        :meth:`_create_mla_ragged_page_cache_config`.

        Args:
            max_length: Maximum model context length.
            page_size: Tokens per page. Defaults to 128.
            hbm_utilization: Fraction of HBM the cache may consume.
                Defaults to 0.9.
            dtype: Optional override for the cache dtype.

            max_cache_tokens: Optional hard ceiling on total KV-cache tokens;
                the page count becomes ``min(hbm-derived, cap)``. Defaults to None.

        Returns:
            A cache config object suitable for the chosen attention
            mechanism.
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

        attn_mechanism = getattr(text_config, "attn_mechanism", None)
        if hasattr(attn_mechanism, "value"):
            attn_mechanism = attn_mechanism.value
        is_mla_ragged = str(attn_mechanism) in (
            "multi_latent_ragged_page_attention_v1",
            "multi_latent_ragged_page_attention_v2",
        )
        if is_mla_ragged:
            # Post-RoPE indexer keys live in pages sharing the KV page table, so the serving
            # path scores each request's real history and attends through the sparse MLA op.
            return self._create_mla_ragged_page_cache_config(
                max_length=max_length,
                page_size=page_size,
                max_cache_tokens=max_cache_tokens,
                hbm_utilization=hbm_utilization,
                dtype=dtype,
                indexer_page_kwargs=dict(
                    indexer_key_dim=int(self.config.index_head_dim), indexer_gate_dim=0, indexer_kpool=1
                ),
            )

        return RaggedPagesCacheConfig.create(
            mesh=self.mesh,
            runtime_sharding_resolver=text_config.runtime_sharding_resolver,
            kvdtype=text_config.kvdtype if dtype is None else dtype,
            max_model_length=max_length,
            num_hidden_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_attention_heads,
            kv_head_dim_size=q_head_dim,
            k_headdim=q_head_dim,
            v_headdim=v_head_dim,
            hbm_utilization=hbm_utilization,
            page_size=page_size,
            max_cache_tokens=max_cache_tokens,
            version=version,
        )

    def create_unified_attention_cache_config(
        self,
        max_length: int,
        *,
        page_size: int = 128,
        hbm_utilization: float = 0.9,
        dtype: jnp.dtype | None = None,
    ):
        """Create unified-attention cache configuration for MLA attention.

        GLM-MoE-DSA's effective key width is ``qk_nope_head_dim + qk_rope_head_dim``.
        The base implementation uses ``config.head_dim`` which is not the MLA KV width.

        Args:
            max_length: Maximum model context length.
            page_size: Tokens per page. Defaults to 128.
            hbm_utilization: Fraction of HBM the cache may consume.
                Defaults to 0.9.
            dtype: Optional override for the cache dtype.

        Returns:
            UnifiedAttentionCacheConfig: Cache config sized for the MLA
                key width.
        """
        from easydel.caching import UnifiedAttentionCacheConfig

        text_config = self.config.get_text_config()
        q_head_dim = self.config.qk_nope_head_dim + self.config.qk_rope_head_dim

        if dtype is None:
            dtype = text_config.kvdtype
            if isinstance(dtype, str):
                dtype = getattr(jnp, dtype, jnp.bfloat16)

        return UnifiedAttentionCacheConfig.create(
            mesh=self.mesh,
            runtime_sharding_resolver=text_config.runtime_sharding_resolver,
            kvdtype=dtype,
            num_hidden_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_attention_heads,
            max_model_length=max_length,
            head_dim=q_head_dim,
            hbm_utilization=hbm_utilization,
            page_size=page_size,
        )


__all__ = ["GlmMoeDsaForCausalLM", "GlmMoeDsaModel"]
