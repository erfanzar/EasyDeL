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

"""Tencent Hunyuan V3 (HYV3) model implementation for EasyDeL.

HYV3 pairs a Qwen3-style attention block (grouped-query attention with
per-head RMSNorm on queries and keys applied before full-dim RoPE, rotary
base ``11_158_840.0``) with a DeepSeek-flavoured sigmoid MoE feed-forward:

- ``config.mlp_layer_types`` schedules each layer as ``"dense"`` (SwiGLU MLP
  of width ``intermediate_size``) or ``"sparse"`` (routed MoE).
- The sparse router computes fp32 sigmoid scores; the top-k experts are
  *selected* on ``sigmoid(logits) + e_score_correction_bias`` while the
  combination *weights* are the raw sigmoid scores, normalized and scaled
  by ``router_scaling_factor``.
- One always-on shared SwiGLU expert of width
  ``moe_intermediate_size * num_shared_experts`` is added to the routed
  output — in float32 when ``enable_moe_fp32_combine`` is set.

Components:
- :class:`HyV3Attention` and :class:`HyV3DecoderLayer` provide per-layer
  transformer logic.
- :class:`HyV3MoE` performs the bias-corrected sigmoid routing.
- :class:`HyV3Model` is the transformer trunk; :class:`HyV3ForCausalLM`
  wraps it with the LM head.
"""

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
    MoeCausalLMOutput,
    MoeModelOutput,
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
    dense_gate_up_layout,
    gated_mlp_forward,
    moe_down_projection_reform_param,
    moe_fused_gate_up_reform_param,
    split_fused_gate_up_projection,
)
from easydel.layers.attention import UnifiedAttention
from easydel.layers.moe import moe_group_topk_select
from easydel.modules._base import BaseCausalLMModule

from .hy_v3_configuration import HyV3Config


class HyV3MLP(spx.Module):
    """Dense SwiGLU feed-forward block for HYV3.

    Used for ``"dense"`` layers (width ``config.intermediate_size``) and,
    with an ``intermediate_size`` override, as the shared expert inside
    :class:`HyV3MoE` (width ``moe_intermediate_size * num_shared_experts``).
    """

    def __init__(
        self,
        config: HyV3Config,
        intermediate_size: int | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the HYV3 MLP block.

        Args:
            config (HyV3Config): Model configuration with MLP parameters.
            intermediate_size (int, optional): Override for the intermediate layer
                width. Defaults to ``config.intermediate_size`` if None.
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
        intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_up_proj = ColumnParallelLinear(
            config.hidden_size,
            (intermediate_size, intermediate_size),
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.mlp_bias,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
            layout=dense_gate_up_layout(intermediate_size),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.mlp_bias,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    @property
    def reform_param(self):
        """Return checkpoint-reform rules for the fused ``gate_up_proj`` kernel.

        Delegates to :meth:`ColumnParallelLinear.build_reform_param` so HF
        checkpoints carrying separate ``gate_proj`` / ``up_proj`` tensors can be
        re-formed into the EasyDeL fused layout used by this dense MLP path.

        Returns:
            dict: Reform-rule mapping consumable by the checkpoint loader.
        """
        return self.gate_up_proj.build_reform_param("gate_up_proj", config=self.config)

    def forward(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> jnp.ndarray:
        """Apply the SwiGLU feed-forward transformation.

        Delegates to :func:`easydel.layers.gated_mlp_forward`, which routes
        supported layouts through the ejkernel fused MLP kernels and runs
        the legacy composition otherwise.

        Args:
            hidden_states: Input tensor ``[batch, seq_len, hidden_dim]``.

        Returns:
            Transformed hidden states ``[batch, seq_len, hidden_dim]``.
        """
        return gated_mlp_forward(self, hidden_states)


class HyV3MLPStack(spx.Module):
    """Stacked routed-expert SwiGLU MLPs for HYV3 sparse layers.

    Holds the ``num_experts`` fused gate/up and down kernels consumed by
    ``BaseMoeModule.moe_call``'s grouped-matmul dispatch.
    """

    def __init__(
        self,
        config: HyV3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the HYV3 expert MLP stack.

        Args:
            config (HyV3Config): Model configuration with MoE parameters including
                ``num_experts`` and ``moe_intermediate_size``.
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
        self.gate_up_proj = ColumnParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.hidden_size,
            out_features=2 * config.moe_intermediate_size,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            use_bias=False,
            partition_manager=config.runtime_sharding_resolver,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.down_proj = RowParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.moe_intermediate_size,
            out_features=config.hidden_size,
            rngs=rngs,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            partition_manager=config.runtime_sharding_resolver,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    @property
    def reform_param(self):
        """Return reform rules for HF stacked expert tensors.

        HF ships ``experts.gate_up_proj`` as ``[E, 2 * moe_intermediate, hidden]``
        and ``experts.down_proj`` as ``[E, hidden, moe_intermediate]``; both are
        transposed into EasyDeL's ``[E, in, out]`` grouped-matmul layout.

        Returns:
            dict: Reform-rule mapping consumable by the checkpoint loader.
        """
        return {
            **moe_fused_gate_up_reform_param(config=self.config),
            **moe_down_projection_reform_param(),
        }

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        group_sizes: Array,
        sorted_experts: Array | None = None,
    ) -> Array:
        """Apply the SwiGLU feed-forward transformation through the routed experts.

        Args:
            hidden_states: Routed token representations grouped by expert.
            group_sizes: Number of tokens routed to each expert.
            sorted_experts: Optional sorted expert indices for efficient computation.

        Returns:
            Expert outputs in the same grouped layout as the input.
        """
        gate_up = self.gate_up_proj(hidden_states, group_sizes, sorted_experts)
        gate, up = split_fused_gate_up_projection(gate_up, config=self.config)
        return self.down_proj(self.act_fn(gate) * up, group_sizes, sorted_experts)


class HyV3TopKRouter(spx.Module):
    """fp32 router projection feeding the sigmoid top-k selector in :class:`HyV3MoE`.

    Owns only the projection ``weight`` of shape ``(hidden_size, num_experts)``;
    the sigmoid scoring, bias-corrected selection, normalization and
    ``router_scaling_factor`` live in
    :func:`easydel.layers.moe.moe_group_topk_select`.
    The matmul is forced into float32 regardless of the activation dtype —
    mirroring HF's ``hidden.float() @ weight.float()`` — because gating-score
    gaps can fall below bfloat16 resolution.
    """

    def __init__(
        self,
        config: HyV3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the HYV3 top-k router projection.

        Args:
            config (HyV3Config): Model configuration carrying ``num_experts``.
            dtype (jnp.dtype, optional): Computation dtype (the router matmul is
                forced to fp32 regardless). Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Parameter storage dtype.
                Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): JAX matmul precision.
                Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.num_experts = config.num_experts
        self.weight = ArrayParam.bound(
            shape=(config.hidden_size, self.num_experts),
            dtype=param_dtype,
            init_method="normal",
            init_kwargs={"stddev": config.initializer_range},
            key=rngs.param,
        )

    def forward(self, hidden_states: Float[Array, "tokens hidden_dim"]) -> Array:
        """Project hidden states to fp32 per-expert router logits.

        Args:
            hidden_states: Token activations whose last dim is ``hidden_size``.

        Returns:
            fp32 router logits of shape ``(num_tokens, num_experts)``.
        """
        hidden_states = hidden_states.reshape(-1, self.config.hidden_size)
        return checkpoint_name(
            jnp.matmul(
                hidden_states.astype(jnp.float32),
                self.weight.value.astype(jnp.float32),
                precision=self.precision,
            ),
            "moe_router_logits",
        )


class HyV3MoE(BaseMoeModule):
    """HYV3 sparse feed-forward block: sigmoid-routed experts plus a shared expert.

    Routing follows the HF ``HYV3TopKRouter`` / ``HYV3MoE`` semantics exactly:

    1. fp32 sigmoid scores over all ``num_experts`` routed experts.
    2. Top-``num_experts_per_tok`` *selection* on
       ``scores + e_score_correction_bias`` (the bias is a per-expert fp32
       parameter loaded from the checkpoint; it steers selection only).
    3. Combination *weights* are the raw sigmoid scores gathered at the
       selected indices, normalized by their sum (``+ 1e-20``) and scaled by
       ``router_scaling_factor``.
    4. The always-on shared SwiGLU expert (width
       ``moe_intermediate_size * num_shared_experts``) is added to the routed
       output — in float32 when ``config.enable_moe_fp32_combine`` is set.

    The live ``e_score_correction_bias`` value is threaded into the routing
    ``select_hook`` per call so loaded checkpoints route correctly (the hook
    installed at ``__init__`` time is only a zero-bias placeholder).
    """

    def __init__(
        self,
        config: HyV3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the HYV3 sparse MoE layer.

        Args:
            config (HyV3Config): Model configuration with MoE parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            n_routed_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            lbl_coef=None,
            rzl_coef=None,
            routing_strategy=MoeRoutingStrategy.TOP_K,
            load_balancing_strategy=MoeLoadBalancingStrategy.STANDARD,
        )
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.router_scaling_factor = config.router_scaling_factor
        self.enable_moe_fp32_combine = config.enable_moe_fp32_combine

        self.gate = HyV3TopKRouter(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.experts = HyV3MLPStack(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        # HF stores the correction bias as a persistent fp32 buffer on the MoE
        # block itself (``model.layers.N.mlp.e_score_correction_bias``).
        self.e_score_correction_bias = ArrayParam.bound(
            shape=(config.num_experts,),
            dtype=jnp.float32,
            init_method="zeros",
            key=rngs.param,
        )
        self.shared_experts = HyV3MLP(
            config=config,
            intermediate_size=config.moe_intermediate_size * config.num_shared_experts,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        # Placeholder hook (zero bias) so ``moe_call``'s strategy auto-config
        # never mutates hooks at trace time; ``forward`` re-binds the hook with
        # the live bias value on every call.
        self.moe_hooks = MoeFusedHooks(
            normalize_gate_logits=lambda x: x,
            select_hook=partial(
                moe_group_topk_select,
                n_routed_experts=config.num_experts,
                score_fn="sigmoid",
                e_score_correction_bias=None,
                n_group=1,
                norm_topk_prob=True,
                routed_scaling_factor=config.router_scaling_factor,
                norm_eps=1e-20,
            ),
        )

    def forward(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> tuple[Array, Array]:
        """Route tokens through the selected experts and add the shared expert.

        Args:
            hidden_states: Input tensor ``[batch, seq_len, hidden_dim]``.

        Returns:
            Tuple containing:
                - Output hidden states ``[batch, seq_len, hidden_dim]``.
                - fp32 pre-sigmoid router logits ``[batch * seq_len, num_experts]``.
        """
        hooks = self.moe_hooks.replace(
            select_hook=partial(
                moe_group_topk_select,
                n_routed_experts=self.config.num_experts,
                score_fn="sigmoid",
                e_score_correction_bias=self.e_score_correction_bias.value,
                n_group=1,
                norm_topk_prob=True,
                routed_scaling_factor=self.router_scaling_factor,
                norm_eps=1e-20,
            ),
        )
        out, router_logits = self.moe_call(
            hidden_state=hidden_states,
            gate_layer=self.gate,
            expert_layer=self.experts,
            gate_up_kernel=self.experts.gate_up_proj.kernel_view(),
            wd_kernel=self.experts.down_proj.kernel_view(),
            act_fn=self.experts.act_fn,
            hooks=hooks,
        )
        shared_output = self.shared_experts(hidden_states)
        if self.enable_moe_fp32_combine:
            out = (out.astype(jnp.float32) + shared_output.astype(jnp.float32)).astype(hidden_states.dtype)
        else:
            out = out + shared_output
        return checkpoint_name(out, "moe_expert_output"), checkpoint_name(router_logits, "moe_router_logits")


class HyV3Attention(UnifiedAttention):
    """Grouped-query attention with per-head Q/K RMSNorm for HYV3.

    Identical to Qwen3 attention: queries and keys pass through a per-head
    RMSNorm (``head_dim``-sized, ``rms_norm_eps``) *before* full-dim rotary
    embeddings. All layers use full causal attention (no sliding window);
    projections carry no bias by default (``config.attention_bias``).
    """

    def __init__(
        self,
        config: HyV3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize HYV3 attention layer with grouped-query attention support.

        Args:
            config (HyV3Config): Model configuration with attention parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the model.
        """
        super().__init__(
            config,
            dtype,
            param_dtype,
            precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="standard",
            causal=True,
            sliding_window=None,
            use_qk_norm=True,
        )
        self.layer_idx = layer_idx

    def _postprocess_qkv(self, query_states, key_states, value_states):
        """Apply per-head Q/K RMSNorm before rotary embeddings.

        Args:
            query_states: Query tensor from the projection layer.
            key_states: Key tensor from the projection layer.
            value_states: Value tensor from the projection layer.

        Returns:
            Tuple of normalized query, normalized key, and unchanged value tensors.
        """
        return self.query_normalization(query_states), self.key_normalization(key_states), value_states


class HyV3DecoderLayer(spx.Module):
    """Single decoder layer for HYV3.

    Standard pre-norm residual block: ``x + attn(norm(x))`` followed by
    ``x + mlp(norm(x))``, where ``mlp`` is either a dense SwiGLU MLP or the
    sparse :class:`HyV3MoE` depending on ``config.mlp_layer_types[layer_idx]``.
    """

    def __init__(
        self,
        config: HyV3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize a HYV3 decoder layer.

        Args:
            config (HyV3Config): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
            layer_idx (int): Index of this layer, used to pick dense vs sparse MLP.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        self.self_attn = HyV3Attention(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )
        self.is_moe = config.mlp_layer_types[layer_idx] == "sparse"
        if self.is_moe:
            self.mlp = HyV3MoE(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        else:
            self.mlp = HyV3MLP(
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
            output_router_logits (bool, optional): Whether to return router logits for MoE layers.
                Defaults to False.
            frequencies (Array | None, optional): Precomputed RoPE frequencies. Defaults to None.

        Returns:
            DecoderLayerOutput: Contains hidden states, attention weights, router logits, and cache view.
        """
        attn_outputs = self.self_attn(
            self.input_layernorm(hidden_states),
            mask_info,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            output_attentions,
            frequencies,
        )
        hidden_states = checkpoint_name(hidden_states + attn_outputs.attention_output, "residual")
        feed_forward_input = self.post_attention_layernorm(hidden_states)

        router_logits = None
        if self.is_moe:
            feed_forward_hidden_states, router_logits = self.mlp(feed_forward_input)
        elif self.config.use_scan_mlp:
            feed_forward_hidden_states = blockwise_ffn(
                self.mlp,
                feed_forward_input,
                self.config.scan_mlp_chunk_size,
            )
        else:
            feed_forward_hidden_states = self.mlp(feed_forward_input)

        hidden_states = checkpoint_name(hidden_states + feed_forward_hidden_states, "residual")
        return DecoderLayerOutput(
            hidden_states=checkpoint_name(hidden_states, "layer_output"),
            attention_weight=attn_outputs.attention_weight,
            router_logits=router_logits if output_router_logits else None,
            cache_view=attn_outputs.cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=HyV3Config, model_type="hy_v3")
class HyV3Model(EasyDeLBaseModule):
    """HYV3 base transformer model.

    Token embedding, ``num_hidden_layers`` :class:`HyV3DecoderLayer` blocks
    (dense/sparse MLP scheduled by ``config.mlp_layer_types``), and a final
    RMSNorm. All layers use full causal attention, so the standard
    :class:`TransformerCache` applies.

    Attributes:
        config (HyV3Config): Configuration for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.
    """

    def __init__(
        self,
        config: HyV3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the HYV3 base model.

        Args:
            config (HyV3Config): Model configuration.
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

        with self.assign_layer_stage(0, total_layers=config.num_hidden_layers):
            self.embed_tokens = Embed(
                config.vocab_size,
                config.hidden_size,
                embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
        remat_layer_block = auto_remat(
            HyV3DecoderLayer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.layers = nn.ModuleList([])
        for layer_idx in range(config.num_hidden_layers):
            with self.assign_layer_stage(layer_idx, total_layers=config.num_hidden_layers):
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
        final_layer_idx = max(0, config.num_hidden_layers - 1)
        with self.assign_layer_stage(final_layer_idx, total_layers=config.num_hidden_layers):
            self.norm = RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
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
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
    ) -> MoeModelOutput:
        """Forward pass through the HYV3 base model.

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
            output_attentions (bool | None, optional): Whether to return attention weights from all layers.
                Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return hidden states from all layers.
                Defaults to None.
            output_router_logits (bool | None, optional): Whether to return router logits from MoE layers.
                Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode (train/decode) for optimizations.
                Auto-detected if None. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cache with precomputed key-value states for generation. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache management. Defaults to None.

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
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        all_router_logits = () if output_router_logits else None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
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

        def _layer_loop(block, carry):
            """Apply a single decoder layer inside the layer-stack scan.

            Body of ``self.layers.scan``; runs ``block`` on the current
            hidden states, optionally accumulates per-layer hidden states,
            attention weights and MoE router logits, and returns the
            updated carry tuple ``(hidden_states, cv, all_hidden_states,
            all_attentions, all_router_logits, idx)``.
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
                    output_router_logits=output_router_logits,
                    frequencies=frequencies,
                )
            hidden_states = self._mark_layer_stage_boundary(layer_outputs.hidden_states, idx, layers=self.layers)

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            cv = self._layer_cache_view_update(
                cv,
                idx,
                layer_outputs.cache_view,
                enabled=trace_layers,
                cache=past_key_values,
            )
            if output_router_logits and layer_outputs.router_logits is not None:
                all_router_logits += (layer_outputs.router_logits,)

            return hidden_states, cv, all_hidden_states, all_attentions, all_router_logits, idx + 1

        hidden_states, _, all_hidden_states, all_attentions, all_router_logits, _ = self.layers.scan(
            _layer_loop,
            (hidden_states, cache_views, all_hidden_states, all_attentions, all_router_logits, 0),
            trace=trace_layers,
        )
        hidden_states = checkpoint_name(self.norm(hidden_states), "model_output")

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


@register_module(TaskType.CAUSAL_LM, config=HyV3Config, model_type="hy_v3")
class HyV3ForCausalLM(BaseCausalLMModule[HyV3Model, HyV3Config]):  # type: ignore
    """HYV3 model with a language modeling head for causal language modeling tasks.

    HYV3 defines no auxiliary router loss (load balance is steered by the
    ``e_score_correction_bias``), so ``aux_loss`` is always ``None``.

    Attributes:
        config (HyV3Config): Configuration for the model.
        dtype (jnp.dtype): Data type for computations (default is jnp.bfloat16).
        param_dtype (jnp.dtype): Data type for parameters (default is jnp.bfloat16).
        precision: Precision setting for JAX operations.
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "hy_v3"
    _config_class = HyV3Config

    def __init__(
        self,
        config: HyV3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the HYV3 model for causal language modeling.

        Args:
            config (HyV3Config): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=HyV3Model,
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
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> MoeCausalLMOutput:
        """Forward pass through the HYV3 causal language model.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch_size, sequence_length).
            inputs_embeds (Array | None, optional): Pre-computed input embeddings.
            attention_mask (Array | None, optional): Attention mask for padding tokens.
            mask_info (MaskInfo | None, optional): Advanced mask information.
            position_ids (Array | None, optional): Position indices for tokens.
            output_attentions (bool | None, optional): Whether to return attention weights.
            output_hidden_states (bool | None, optional): Whether to return all hidden states.
            output_router_logits (bool | None, optional): Whether to return router logits from MoE layers.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode for optimizations.
            past_key_values: Cache with precomputed key-value states.
            cache_metadata: Metadata for cache management.
            apply_lm_head (bool, optional): Whether to apply the language modeling head. Defaults to True.

        Returns:
            MoeCausalLMOutput: Contains logits, past_key_values, hidden_states, attentions,
                router_logits, and aux_loss (always ``None`` for HYV3).
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
