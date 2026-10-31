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
    gated_mlp_forward,
    moe_down_projection_reform_param,
    moe_fused_gate_up_reform_param,
    split_fused_gate_up_projection,
)
from easydel.layers.attention import FlexibleAttentionModule, UnifiedAttention
from easydel.layers.moe import moe_group_topk_select
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

        The standard 3D path delegates to
        :func:`easydel.layers.gated_mlp_forward` (ejkernel fused MLP where
        supported, legacy composition otherwise). The 2D moe-infer path
        keeps its unsharded composition.

        Args:
            hidden_states: Input tensor ``[batch, seq_len, hidden_dim]``.

        Returns:
            Transformed hidden states ``[batch, seq_len, hidden_dim]``.
        """
        if hidden_states.ndim != 3:  # in moe infer: no logical sharding on 2D tokens
            gate_up = checkpoint_name(self.gate_up_proj(hidden_states), "mlp_gate_up")
            gate_raw, up = split_fused_gate_up_projection(gate_up, config=self.config)
            gate = checkpoint_name(self.act_fn(gate_raw), "mlp_gate")
            hidden_states = checkpoint_name(self.down_proj(gate * up), "mlp_down")
            return checkpoint_name(hidden_states, "mlp_output")
        return gated_mlp_forward(self, hidden_states)


class Mistral4TopkRouter(spx.Module):
    """Mistral4 router emitting raw logits for routing diagnostics and losses.

    Unlike DeepSeek-V3's biased-sigmoid ``noaux_tc`` router, Mistral4 scores
    experts with a single fp32 softmax in the MoE normalization hook — there
    is no ``e_score_correction_bias`` and no sigmoid. Group restriction and
    top-k selection consume those probabilities through
    :func:`easydel.layers.moe.moe_group_topk_select` with ``score_fn="none"``.
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
        """Compute full-expert raw logits, preserving them for auxiliary losses.

        Args:
            hidden_states (Array): Input tensor of shape ``[batch * seq_len, hidden_dim]``.

        Returns:
            Array: Per-expert fp32 logits ``[batch * seq_len, n_routed_experts]``.
        """
        logits = jnp.dot(
            hidden_states.astype(jnp.float32),
            self.weight.value.astype(jnp.float32),
            precision=self.precision,
        )
        return logits


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

    * **Router** (:class:`Mistral4TopkRouter`) emits raw logits, normalized
      once to fp32 softmax scores before expert selection in the ``select_hook``:
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
        self.layer_idx = layer_idx
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
        # Preserve raw logits for outputs/losses; normalize exactly once before
        # grouped selection, which consumes full per-expert probabilities.
        self.moe_hooks = self.moe_hooks.replace(
            normalize_gate_logits=lambda x: jax.nn.softmax(x.astype(jnp.float32), axis=-1),
            select_hook=functools.partial(
                moe_group_topk_select,
                n_routed_experts=config.n_routed_experts,
                score_fn="none",
                n_group=config.n_group,
                topk_group=config.topk_group,
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
            gate_up_kernel=self.experts.gate_up_proj.kernel_view(),
            wd_kernel=self.experts.down_proj.kernel_view(),
            act_fn=self.experts.act_fn,
            # The standard path uses layer_idx to emit tensor-valued debug callbacks.
            # Keep ordinary forwards silent; layer identity is supplied only on explicit opt-in.
            layer_idx=self.layer_idx if self.config.moe_debug else None,
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
        original_max = rope_params.get("original_max_position_embeddings") if isinstance(rope_params, dict) else None
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
        """Build the MLA projection stack.

        The body lived here in four families with byte-identical contents; it
        is now :meth:`UnifiedAttention._build_mla_projections`, driven by this
        class's ``projection_mapping`` so the HF attribute names are unchanged.
        """
        self._build_mla_projections(config, dtype, param_dtype, precision, rngs)

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

    def _postprocess_mla_queries(
        self,
        q_nope: Float[Array, "batch heads seq_len nope_dim"],
        q_pe: Float[Array, "batch heads seq_len rope_dim"],
        position_ids: Int[Array, "batch seq_len"],
    ) -> tuple[Array, Array]:
        """Apply Llama-4 temperature before either shared MLA cache path.

        Args:
            q_nope: Non-positional query components in compute dtype.
            q_pe: Rotary-transformed positional query components in compute dtype.
            position_ids: Query positions, including any cache offset.

        Returns:
            Both query components scaled by the position-dependent temperature,
            with their original shapes and dtypes preserved.
        """
        attn_scale = self._llama_4_attn_scale(position_ids)
        if attn_scale is not None:
            scale = attn_scale[:, None, :, None]
            q_nope = (q_nope * scale).astype(q_nope.dtype)
            q_pe = (q_pe * scale).astype(q_pe.dtype)
        return q_nope, q_pe


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

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else getattr(self.config, "output_router_logits", False)
        )
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

        def _layer_loop(layer, carry):
            """Per-layer body for the Mistral4 decoder ``scan``.

            Carry layout: ``(hidden_states, cv, all_hidden_states,
            all_attentions, all_router_logits, idx)``. Records the input
            hidden state when requested, runs one decoder layer at the
            assigned pipeline stage, accumulates attention weights and MoE
            router logits when requested, updates the per-layer KV cache
            slot, and returns the next carry.
            """
            hidden_states, cv, all_hidden_states, all_attentions, all_router_logits, idx = carry
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            with self._layer_stage_context(idx, layers=self.layers):
                output = layer(
                    hidden_states=hidden_states,
                    frequencies=frequencies,
                    mask_info=mask_info,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                    mode=mode,
                    cache_view=self._layer_cache_view_at(cv, idx, enabled=trace_layers, cache=past_key_values),
                    cache_metadata=cache_metadata,
                )
            hidden_states = self._mark_layer_stage_boundary(output.hidden_states, idx, layers=self.layers)

            if output_attentions:
                all_attentions += (output.attention_weight,)

            if output_router_logits and hasattr(output, "router_logits") and output.router_logits is not None:
                all_router_logits += (output.router_logits,)

            cv = self._layer_cache_view_update(
                cv,
                idx,
                output.cache_view,
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
        )

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
