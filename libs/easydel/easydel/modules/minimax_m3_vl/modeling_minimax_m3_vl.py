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

"""MiniMax M3 VL model implementation for EasyDeL.

MiniMax M3 VL couples a CLIP-style vision tower with a MiniMax-M2-style
full-attention MoE text decoder (the text model is self-contained in this
family; it is *not* the lightning-attention MiniMax-Text-01 architecture in
``modules/minimax``):

- **Text decoder** — grouped-query attention with per-head Gemma-style
  ``(1 + weight)`` RMSNorm on Q/K before partial RoPE
  (``rope_parameters.partial_rotary_factor``); per-layer MLP scheduling via
  ``config.mlp_layer_types`` between a dense clamped SwiGLU-OAI MLP and a
  sigmoid-routed MoE (selection on ``sigmoid(logits) +
  e_score_correction_bias``, weights from the raw sigmoid scores, normalized
  and scaled by ``routed_scaling_factor``) plus one shared expert. Layers
  whose ``config.layer_types`` entry is ``"minimax_m3_sparse"`` additionally
  run a block-sparse Lightning Indexer that restricts attention to the
  top-``index_topk_blocks`` key blocks per query (dense-mask prefill path;
  decode caching for sparse layers is not implemented).
- **Vision tower** — Conv3d patch embedding, pre-LN CLIP blocks with 3D
  (T/H/W) rotary position embeddings on Q/K, no final LayerNorm.
- **Projector** — two-stage GELU MLP: per-patch projection to the text width
  followed by a ``spatial_merge_size**2`` patch-merge MLP.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import spectrax as spx
from eformer.pytree import auto_pytree
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
    AttentionLayerOutput,
    BaseModelOutput,
    DecoderLayerOutput,
    ModelOutput,
    MoeCausalLMOutput,
    MoeModelOutput,
    VLMCausalLMOutput,
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
    RowParallelLinear,
    RowParallelMoELinear,
    dense_gate_up_layout,
    dense_qkv_layout,
    gated_mlp_forward,
    moe_down_projection_reform_param,
    moe_fused_gate_up_reform_param,
    split_fused_gate_up_projection,
)
from easydel.layers.attention import AttentionModule, FlexibleAttentionModule, UnifiedAttention
from easydel.layers.norms import LayerNorm
from easydel.modules._base import BaseCausalLMModule, BaseVisionLanguageModule
from easydel.modules.gemma3.modeling_gemma3 import Gemma3RMSNorm
from easydel.modules.qwen2_vl.modeling_qwen2_vl import Qwen2VLPatchEmbed

from .minimax_m3_vl_configuration import (
    MiniMaxM3VLConfig,
    MiniMaxM3VLTextConfig,
    MiniMaxM3VLVisionConfig,
)


class MiniMaxM3VLRMSNorm(Gemma3RMSNorm):
    """Gemma-style RMSNorm used throughout MiniMax M3 VL.

    Normalizes in float32 and scales by ``(1 + weight)`` so the layer is the
    identity at initialization — checkpoints store weights near zero. Used for
    the decoder layer norms, the final norm, the per-head Q/K norms and the
    Lightning-Indexer Q/K norms.
    """


def _swiglu_oai(gate: Array, up: Array, *, alpha: float, limit: float) -> Array:
    """Apply the clamped SwiGLU-OAI activation used by MiniMax M3.

    Computes ``(clip(up, -limit, limit) + 1) * glu`` where
    ``glu = clip(gate, max=limit) * sigmoid(clip(gate, max=limit) * alpha)``.
    The ``+1`` shift on the up branch keeps the multiplicative path
    well-defined when ``up`` is exactly zero (same recipe as GPT-OSS, but the
    gate/up channels are stored as concatenated halves, not interleaved).

    Args:
        gate: Gate projection output.
        up: Up projection output.
        alpha: Sigmoid gain of the Swish-style gate.
        limit: Clamp bound (gate is clamped from above only; up on both sides).

    Returns:
        The activated intermediate tensor with the same shape as ``gate``.
    """
    gate = jnp.clip(gate, max=limit)
    up = jnp.clip(up, min=-limit, max=limit)
    glu = gate * jax.nn.sigmoid(gate * alpha)
    return (up + 1.0) * glu


def _apply_partial_rotary(x: Array, cos: Array, sin: Array) -> Array:
    """Rotate the first ``cos.shape[-1]`` channels of ``x`` (non-interleaved).

    Mirrors HF's GLM/MiniMax ``apply_rotary_pos_emb``: the leading
    ``rotary_dim`` channels are rotated with the rotate-half convention and
    the tail passes through unchanged.

    Args:
        x: Input tensor ``[..., head_dim]``; broadcastable against ``cos``.
        cos: Cosine table ``[..., rotary_dim]``.
        sin: Sine table ``[..., rotary_dim]``.

    Returns:
        Tensor of the same shape as ``x`` with the leading channels rotated.
    """
    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    half = rotary_dim // 2
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    x_rot = x_rot * cos + rotated * sin
    if x_pass.shape[-1] == 0:
        return x_rot
    return jnp.concatenate([x_rot, x_pass], axis=-1)


def _gather_cos_sin(
    frequencies: Array,
    position_ids: Int[Array, "batch seq_len"],
    width: int | None = None,
) -> tuple[Array, Array]:
    """Gather HF-layout cos/sin tables from an EasyDeL frequency cache.

    EasyDeL frequency caches store ``[max_pos, 2 * rot] = [cos(f) | sin(f)]``
    with ``f`` of width ``rot / 2`` (no duplication). HF's tables are
    ``cat([f, f])`` wide, so the halves are duplicated here before optional
    truncation to ``width`` — matching the indexer's ``cos[..., :head_dim]``
    slice semantics exactly.

    Args:
        frequencies: Frequency cache (possibly a ``ModuleCaches`` wrapper).
        position_ids: Positions to gather, ``[batch, seq_len]``.
        width: Optional truncation width of the returned tables.

    Returns:
        Tuple ``(cos, sin)`` of shape ``[batch, seq_len, min(width, rot)]``
        in float32.
    """
    freq = frequencies.value if hasattr(frequencies, "value") else frequencies
    table = freq[position_ids].astype(jnp.float32)
    cos_half, sin_half = jnp.split(table, 2, axis=-1)
    cos = jnp.concatenate([cos_half, cos_half], axis=-1)
    sin = jnp.concatenate([sin_half, sin_half], axis=-1)
    if width is not None:
        cos = cos[..., :width]
        sin = sin[..., :width]
    return cos, sin


class MiniMaxM3VLDenseMLP(spx.Module):
    """Dense clamped SwiGLU-OAI feed-forward block.

    Used for ``"dense"`` layers (width ``config.dense_intermediate_size``) and,
    with an ``intermediate_size`` override, as the shared expert inside
    :class:`MiniMaxM3VLSparseMoeBlock` (width
    ``config.shared_intermediate_size``). HF stores the fused
    ``gate_up_proj`` tensor pre-fused (gate and up halves concatenated), so
    the layout declares ``source_is_fused=True``.
    """

    def __init__(
        self,
        config: MiniMaxM3VLTextConfig,
        intermediate_size: int | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the dense MLP block.

        Args:
            config (MiniMaxM3VLTextConfig): Model configuration.
            intermediate_size (int, optional): Override for the intermediate
                width. Defaults to ``config.dense_intermediate_size``.
            dtype (jnp.dtype, optional): Computation dtype. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Parameter dtype. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Matmul precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.swiglu_alpha = config.swiglu_alpha
        self.swiglu_limit = config.swiglu_limit
        self.fused_act_name = "swiglu_oai"
        self.fused_act_params = (float(config.swiglu_alpha), float(config.swiglu_limit))
        intermediate_size = intermediate_size if intermediate_size is not None else config.dense_intermediate_size
        self.gate_up_proj = ColumnParallelLinear(
            config.hidden_size,
            (intermediate_size, intermediate_size),
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
            # HF MiniMax M3 checkpoints store gate_up_proj pre-fused (one tensor).
            layout=dense_gate_up_layout(intermediate_size, source_is_fused=True),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )

    @property
    def reform_param(self):
        """Return checkpoint-reform rules for the pre-fused ``gate_up_proj`` tensor.

        Returns:
            dict: Reform-rule mapping consumable by the checkpoint loader.
        """
        return self.gate_up_proj.build_reform_param("gate_up_proj", config=self.config, include_bias=False)

    def forward(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> jnp.ndarray:
        """Apply the clamped SwiGLU-OAI feed-forward transformation.

        Delegates to :func:`easydel.layers.gated_mlp_forward`; the declared
        ``swiglu_oai`` combine runs on both the fused and fallback paths.

        Args:
            hidden_states: Input tensor ``[batch, seq_len, hidden_dim]``.

        Returns:
            Transformed hidden states ``[batch, seq_len, hidden_dim]``.
        """
        return gated_mlp_forward(self, hidden_states)


class MiniMaxM3VLExperts(spx.Module):
    """Stacked routed-expert clamped SwiGLU-OAI MLPs for MoE layers.

    Holds the ``num_local_experts`` fused gate/up and down kernels consumed by
    ``BaseMoeModule.moe_call``'s grouped-matmul dispatch. HF ships the stacked
    tensors as ``gate_up_proj`` ``[E, 2 * intermediate, hidden]`` (fused,
    concatenated halves) and ``down_proj`` ``[E, hidden, intermediate]``.
    """

    def __init__(
        self,
        config: MiniMaxM3VLTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the expert MLP stack.

        Args:
            config (MiniMaxM3VLTextConfig): Model configuration with MoE
                parameters (``num_local_experts``, ``intermediate_size``).
            dtype (jnp.dtype, optional): Computation dtype. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Parameter dtype. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Matmul precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.swiglu_alpha = config.swiglu_alpha
        self.swiglu_limit = config.swiglu_limit
        self.gate_up_proj = ColumnParallelMoELinear(
            num_experts=config.num_local_experts,
            in_features=config.hidden_size,
            out_features=2 * config.intermediate_size,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            use_bias=False,
            partition_manager=config.runtime_sharding_resolver,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.down_proj = RowParallelMoELinear(
            num_experts=config.num_local_experts,
            in_features=config.intermediate_size,
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

        HF ships ``experts.gate_up_proj`` as ``[E, 2 * intermediate, hidden]``
        and ``experts.down_proj`` as ``[E, hidden, intermediate]``; both are
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
        hidden_states: Float[Array, "tokens hidden_dim"],
        group_sizes: Array,
        sorted_experts: Array | None = None,
    ) -> Array:
        """Apply the clamped SwiGLU-OAI transformation through the routed experts.

        Args:
            hidden_states: Routed token representations grouped by expert.
            group_sizes: Number of tokens routed to each expert.
            sorted_experts: Optional sorted expert indices.

        Returns:
            Expert outputs in the same grouped layout as the input.
        """
        gate_up = self.gate_up_proj(hidden_states, group_sizes, sorted_experts)
        gate, up = split_fused_gate_up_projection(gate_up, config=self.config)
        intermediate = _swiglu_oai(gate, up, alpha=self.swiglu_alpha, limit=self.swiglu_limit)
        return self.down_proj(intermediate, group_sizes, sorted_experts)


class MiniMaxM3VLTopKRouter(spx.Module):
    """Router projection plus the per-expert selection-bias buffer.

    Owns the projection ``weight`` of shape ``(hidden_size, num_experts)``
    and the fp32 ``e_score_correction_bias`` buffer (HF stores it on the
    router: ``mlp.gate.e_score_correction_bias``). The sigmoid scoring,
    bias-corrected selection, normalization and ``routed_scaling_factor``
    live in :meth:`MiniMaxM3VLSparseMoeBlock._select_experts_static`.
    """

    def __init__(
        self,
        config: MiniMaxM3VLTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the router.

        Args:
            config (MiniMaxM3VLTextConfig): Model configuration carrying
                ``num_local_experts``.
            dtype (jnp.dtype, optional): Computation dtype. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Parameter dtype. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Matmul precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.num_experts = config.num_local_experts
        self.weight = ArrayParam.bound(
            shape=(config.hidden_size, self.num_experts),
            dtype=param_dtype,
            init_method="normal",
            init_kwargs={"stddev": config.initializer_range},
            key=rngs.param,
        )
        # HF stores the correction bias as a persistent fp32 buffer on the
        # router (``model.layers.N.mlp.gate.e_score_correction_bias``). It
        # steers *selection* only; combination weights use the raw scores.
        self.e_score_correction_bias = ArrayParam.bound(
            shape=(self.num_experts,),
            dtype=jnp.float32,
            init_method="zeros",
            key=rngs.param,
        )

    def forward(self, hidden_states: Float[Array, "tokens hidden_dim"]) -> Array:
        """Project hidden states to per-expert router logits.

        Mirrors HF: the matmul runs in the router weight's dtype (no fp32
        forcing); the fp32 promotion happens at sigmoid time in the select
        hook.

        Args:
            hidden_states: Token activations whose last dim is ``hidden_size``.

        Returns:
            Router logits of shape ``(num_tokens, num_experts)``.
        """
        hidden_states = hidden_states.reshape(-1, self.config.hidden_size)
        weight = self.weight.value
        return checkpoint_name(
            jnp.matmul(hidden_states.astype(weight.dtype), weight, precision=self.precision),
            "moe_router_logits",
        )


class MiniMaxM3VLSparseMoeBlock(BaseMoeModule):
    """MiniMax M3 sparse feed-forward block: sigmoid-routed experts plus a shared expert.

    Routing follows the HF ``MiniMaxM3VLTopKRouter`` / ``LagunaSparseMoeBlock``
    semantics exactly:

    1. Sigmoid scores over all routed experts (fp32 sigmoid on the raw
       logits).
    2. Top-``num_experts_per_tok`` *selection* on
       ``scores + e_score_correction_bias``.
    3. Combination *weights* are the raw sigmoid scores gathered at the
       selected indices, normalized by their sum, then scaled by
       ``routed_scaling_factor`` (HF scales the routed output after the
       experts; scaling the weights is algebraically identical).
    4. The always-on shared clamped SwiGLU-OAI expert (width
       ``shared_intermediate_size``) is added to the routed output.

    The live ``e_score_correction_bias`` value is threaded into the routing
    ``select_hook`` per call so loaded checkpoints route correctly (the hook
    installed at ``__init__`` time is only a zero-bias placeholder), and
    ``normalize_gate_logits`` is pinned to identity so the strategy
    auto-configuration never re-normalizes the scaled weights on later calls.
    """

    def __init__(
        self,
        config: MiniMaxM3VLTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the sparse MoE block.

        Args:
            config (MiniMaxM3VLTextConfig): Model configuration.
            dtype (jnp.dtype, optional): Computation dtype. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Parameter dtype. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Matmul precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            n_routed_experts=config.num_local_experts,
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
        self.routed_scaling_factor = config.routed_scaling_factor

        self.gate = MiniMaxM3VLTopKRouter(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.experts = MiniMaxM3VLExperts(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.shared_experts = MiniMaxM3VLDenseMLP(
            config=config,
            intermediate_size=config.shared_intermediate_size,
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
                self._select_experts_static,
                e_score_correction_bias=None,
                routed_scaling_factor=config.routed_scaling_factor,
            ),
        )

    @staticmethod
    def _select_experts_static(
        gate_logits: Array,
        pre_bias_logits: Array | None,
        k: int,
        *,
        e_score_correction_bias: Array | None,
        routed_scaling_factor: float,
    ) -> tuple[Array, Array]:
        """Run the MiniMax M3 bias-corrected sigmoid top-k expert selection.

        Args:
            gate_logits: Pre-sigmoid router logits ``(num_tokens, num_experts)``.
            pre_bias_logits: Unused; kept for hook signature compatibility.
            k: Number of routed experts selected per token.
            e_score_correction_bias: Per-expert fp32 selection bias, or ``None``
                for zero bias.
            routed_scaling_factor: Multiplier applied to the selected weights
                (algebraically identical to HF scaling the routed output).

        Returns:
            Tuple ``(topk_weights, topk_indices)`` of shape ``(num_tokens, k)``.
        """
        del pre_bias_logits
        scores = jax.nn.sigmoid(gate_logits.astype(jnp.float32))
        if e_score_correction_bias is None:
            scores_for_choice = scores
        else:
            scores_for_choice = scores + e_score_correction_bias.astype(jnp.float32)
        topk_indices = jax.lax.top_k(scores_for_choice, k=k)[1]
        topk_weights = jnp.take_along_axis(scores, topk_indices, axis=-1)
        topk_weights = topk_weights / jnp.sum(topk_weights, axis=-1, keepdims=True)
        topk_weights = topk_weights * routed_scaling_factor
        return topk_weights, topk_indices

    def forward(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> tuple[Array, Array]:
        """Route tokens through the selected experts and add the shared expert.

        Args:
            hidden_states: Input tensor ``[batch, seq_len, hidden_dim]``.

        Returns:
            Tuple containing:
                - Output hidden states ``[batch, seq_len, hidden_dim]``.
                - Pre-sigmoid router logits ``[batch * seq_len, num_experts]``.
        """
        hooks = self.moe_hooks.replace(
            select_hook=partial(
                self._select_experts_static,
                e_score_correction_bias=self.gate.e_score_correction_bias.value,
                routed_scaling_factor=self.routed_scaling_factor,
            ),
        )

        swiglu_alpha = self.experts.swiglu_alpha
        swiglu_limit = self.experts.swiglu_limit

        def ffn_activation(w0: Array, w1: Array) -> Array:
            """Fused-path clamped SwiGLU-OAI over the split gate/up halves."""
            return _swiglu_oai(w0, w1, alpha=swiglu_alpha, limit=swiglu_limit)

        out, router_logits = self.moe_call(
            hidden_state=hidden_states,
            gate_layer=self.gate,
            expert_layer=self.experts,
            gate_up_kernel=self.experts.gate_up_proj.kernel_view(),
            wd_kernel=self.experts.down_proj.kernel_view(),
            act_fn=self.experts.act_fn,
            ffn_activation=ffn_activation,
            hooks=hooks,
        )
        shared_output = self.shared_experts(hidden_states)
        out = out + shared_output
        return checkpoint_name(out, "moe_expert_output"), checkpoint_name(router_logits, "moe_router_logits")


class MiniMaxM3VLIndexer(spx.Module):
    """Lightning Indexer for MiniMax M3 block-sparse attention (prefill path).

    Scores each query against every key with a small ``index_n_heads``-head
    dot-product branch (one head per KV/GQA group), max-pools the per-key
    scores into blocks of ``index_block_size`` keys, and keeps the
    top-``index_topk_blocks`` key blocks per query plus the
    ``index_local_blocks`` blocks immediately preceding the query. The
    selection is expanded into a dense additive attention bias (0 at every
    allowed (query, key) pair; dtype-min elsewhere) — the eager/SDPA reference
    semantics of HF's ``build_block_mask``. Like DeepSeek-V4's indexer this is
    purely a selection branch: no value projection and no residual output.

    Only the stateless (no KV cache) path is implemented; decode caching of
    the indexer keys is not supported.
    """

    def __init__(
        self,
        config: MiniMaxM3VLTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the indexer.

        Args:
            config (MiniMaxM3VLTextConfig): Model configuration carrying the
                ``index_*`` hyperparameters.
            dtype (jnp.dtype, optional): Computation dtype. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Parameter dtype. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Matmul precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.head_dim = config.index_head_dim
        self.num_heads = config.index_n_heads
        self.block_size = config.index_block_size
        self.topk_blocks = config.index_topk_blocks
        self.local_blocks = config.index_local_blocks
        linear = partial(
            ColumnParallelLinear,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )
        self.q_proj = linear(config.hidden_size, config.index_n_heads * config.index_head_dim)
        self.k_proj = linear(config.hidden_size, config.index_head_dim)
        self.q_norm = MiniMaxM3VLRMSNorm(config, param_dtype=param_dtype, dim=config.index_head_dim)
        self.k_norm = MiniMaxM3VLRMSNorm(config, param_dtype=param_dtype, dim=config.index_head_dim)

    def compute_block_bias(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        position_ids: Int[Array, "batch seq_len"],
        frequencies: Array,
        num_attention_heads: int,
        bias_dtype: jnp.dtype,
    ) -> Float[Array, "batch num_attention_heads seq_len seq_len"]:
        """Select key blocks per query and expand them into an additive bias.

        Args:
            hidden_states: Pre-attention residual stream ``[B, S, H]``.
            position_ids: Per-token content positions ``[B, S]`` (the block of
                a query is ``position_ids // index_block_size``).
            frequencies: The decoder's RoPE frequency cache (the indexer reuses
                the main rotary table, truncated to ``index_head_dim``).
            num_attention_heads: Full query-head count to broadcast the
                per-KV-group selection onto.
            bias_dtype: Output dtype of the additive bias.

        Returns:
            Additive attention bias ``[B, num_attention_heads, S, S]`` with
            ``0`` at selected causally-valid (query, key) pairs and the
            dtype's minimum elsewhere.
        """
        batch, seq_len, _ = hidden_states.shape
        idx_q = self.q_norm(self.q_proj(hidden_states).reshape(batch, seq_len, self.num_heads, self.head_dim))
        idx_k = self.k_norm(self.k_proj(hidden_states).reshape(batch, seq_len, 1, self.head_dim))

        cos, sin = _gather_cos_sin(frequencies, position_ids, width=self.head_dim)
        cos = cos[:, :, None, :].astype(idx_q.dtype)
        sin = sin[:, :, None, :].astype(idx_q.dtype)
        idx_q = _apply_partial_rotary(idx_q, cos, sin)
        idx_k = _apply_partial_rotary(idx_k, cos, sin)

        # Per-key scores; fp32 like HF.
        scores = jnp.einsum(
            "bqhd,bkd->bhqk",
            idx_q.astype(jnp.float32),
            idx_k[:, :, 0, :].astype(jnp.float32),
            precision=self.precision,
        )
        k_positions = jnp.arange(seq_len)
        token_future = k_positions[None, None, None, :] > position_ids[:, None, :, None]
        scores = jnp.where(token_future, -jnp.inf, scores)

        pad = (-seq_len) % self.block_size
        if pad:
            scores = jnp.pad(scores, ((0, 0), (0, 0), (0, 0), (0, pad)), constant_values=-jnp.inf)
        num_key_blocks = (seq_len + pad) // self.block_size
        block_scores = scores.reshape(batch, self.num_heads, seq_len, num_key_blocks, self.block_size).max(-1)

        q_block = position_ids // self.block_size  # [B, S]
        if self.local_blocks > 0:
            local = jnp.arange(self.local_blocks)
            local_idx = jnp.clip(q_block[:, :, None] - local[None, None, :], min=0)  # [B, S, L]
            local_idx = jnp.broadcast_to(local_idx[:, None, :, :], (batch, self.num_heads, seq_len, self.local_blocks))
            block_scores = jnp.put_along_axis(block_scores, local_idx, jnp.inf, axis=-1, inplace=False)

        topk = min(self.topk_blocks, num_key_blocks)
        topk_scores, topk_indices = jax.lax.top_k(block_scores, topk)
        # Future/empty blocks keep their -inf score; tag them -1 (dropped).
        topk_indices = jnp.where(topk_scores == -jnp.inf, -1, topk_indices)

        safe = jnp.where(topk_indices < 0, num_key_blocks, topk_indices)
        block_bias = jnp.full((batch, self.num_heads, seq_len, num_key_blocks + 1), -jnp.inf, dtype=jnp.float32)
        block_bias = jnp.put_along_axis(block_bias, safe, 0.0, axis=-1, inplace=False)
        block_keep = block_bias[..., :num_key_blocks] == 0.0

        # Block verdict back onto keys; indexer heads carry one selection per
        # KV/GQA group — expand up to the full query-head count.
        keep = jnp.repeat(block_keep, self.block_size, axis=-1)[..., :seq_len]
        keep = jnp.repeat(keep, num_attention_heads // self.num_heads, axis=1)
        # Compose with token-level causality (HF composes with the causal /
        # padding mask; padding is applied independently by the kernel).
        keep = keep & ~token_future
        min_value = jnp.finfo(bias_dtype).min
        return jnp.where(keep, jnp.asarray(0.0, bias_dtype), jnp.asarray(min_value, bias_dtype))


class MiniMaxM3VLAttention(UnifiedAttention):
    """MiniMax M3 attention: per-head Gemma Q/K norm, partial RoPE, optional indexer.

    Grouped-query attention with per-head Gemma-style ``(1 + weight)``
    RMSNorm applied to queries and keys before rotary embeddings; the rotated
    width follows ``config.partial_rotary_factor``. Layers whose
    ``config.layer_types`` entry is ``"minimax_m3_sparse"`` additionally build
    a :class:`MiniMaxM3VLIndexer` whose dense block-selection bias is added to
    the attention logits (stateless prefill path only).
    """

    def __init__(
        self,
        config: MiniMaxM3VLTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize the attention layer.

        Args:
            config (MiniMaxM3VLTextConfig): Model configuration.
            dtype (jnp.dtype, optional): Computation dtype. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Parameter dtype. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Matmul precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
            layer_idx (int): Index of this layer (selects full vs sparse).
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
        self.is_sparse = config.layer_types[layer_idx] == "minimax_m3_sparse"
        if self.is_sparse:
            self.indexer = MiniMaxM3VLIndexer(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        else:
            self.indexer = None

    def _create_q_norm(self, config, dtype, param_dtype, rngs):
        """Build the per-head Gemma-style query normalization.

        Args:
            config: Text config (read for ``rms_norm_eps``).
            dtype: Compute dtype (unused; kept for API parity).
            param_dtype: Parameter dtype for the learnable scale.
            rngs: Random number generator state (unused).

        Returns:
            MiniMaxM3VLRMSNorm: Per-head query normalization over ``head_dim``.
        """
        del dtype, rngs
        return MiniMaxM3VLRMSNorm(config, param_dtype=param_dtype, dim=self.head_dim)

    def _create_k_norm(self, config, dtype, param_dtype, rngs):
        """Build the per-head Gemma-style key normalization.

        Args:
            config: Text config (read for ``rms_norm_eps``).
            dtype: Compute dtype (unused; kept for API parity).
            param_dtype: Parameter dtype for the learnable scale.
            rngs: Random number generator state (unused).

        Returns:
            MiniMaxM3VLRMSNorm: Per-head key normalization over ``head_dim``.
        """
        del dtype, rngs
        return MiniMaxM3VLRMSNorm(config, param_dtype=param_dtype, dim=self.head_dim)

    def _postprocess_qkv(self, query_states, key_states, value_states):
        """Apply per-head Q/K RMSNorm before rotary embeddings.

        Args:
            query_states: Query tensor ``[B, S, H_q, D]``.
            key_states: Key tensor ``[B, S, H_kv, D]``.
            value_states: Value tensor ``[B, S, H_kv, D]``.

        Returns:
            Tuple of normalized query, normalized key, and unchanged values.
        """
        return self.query_normalization(query_states), self.key_normalization(key_states), value_states

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type: ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
        alibi: Array | None = None,
    ) -> AttentionLayerOutput:
        """Route to the standard path or the block-sparse indexer path.

        Args:
            hidden_states: Residual stream ``[batch, seq_len, hidden_dim]``.
            mask_info: Mask container (mask, segment IDs, positions).
            position_ids: Per-token positions ``[batch, seq_len]``.
            mode: Runtime mode (train / prefill / decode).
            cache_view: Optional KV cache view (full-attention layers only).
            cache_metadata: Optional cache metadata companion.
            output_attentions: Whether to materialize attention weights.
            frequencies: Precomputed RoPE cos/sin cache.
            alibi: Unused; kept for signature parity.

        Returns:
            AttentionLayerOutput with the attention output and cache view.
        """
        if self.indexer is None:
            return super().forward(
                hidden_states,
                mask_info,
                position_ids,
                mode,
                cache_view,
                cache_metadata,
                output_attentions,
                frequencies,
                alibi,
            )
        return self.forward_block_sparse(
            hidden_states,
            mask_info,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            output_attentions,
            frequencies,
        )

    def forward_block_sparse(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type: ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ) -> AttentionLayerOutput:
        """Standard attention flow with the indexer's additive block bias.

        A lean clone of :meth:`UnifiedAttention.forward_standard` that feeds
        the Lightning Indexer's dense block-selection bias (0 for selected
        causally-valid pairs, dtype-min otherwise) to the attention kernel via
        its ``bias`` channel. Stateless only — sparse layers do not support a
        KV cache view.

        Args:
            hidden_states: Residual stream ``[batch, seq_len, hidden_dim]``.
            mask_info: Mask container (padding is composed by the kernel).
            position_ids: Per-token positions ``[batch, seq_len]``.
            mode: Runtime mode.
            cache_view: Must be ``None`` (decode caching unimplemented).
            cache_metadata: Optional cache metadata companion.
            output_attentions: Whether to materialize attention weights.
            frequencies: Precomputed RoPE cos/sin cache.

        Returns:
            AttentionLayerOutput with the attention output.

        Raises:
            NotImplementedError: If a cache view is supplied.
        """
        if cache_view is not None:
            raise NotImplementedError(
                "MiniMax M3 sparse-attention layers do not support KV-cache decoding yet; "
                "run stateless forwards (past_key_values=None) or use a full-attention config."
            )
        batch_size, sequence_length = hidden_states.shape[:2]

        block_bias = self.indexer.compute_block_bias(
            hidden_states,
            position_ids,
            frequencies,
            num_attention_heads=self.num_heads,
            bias_dtype=jnp.promote_types(self.dtype, jnp.float32),
        )

        query_states, key_states, value_states = self._project_qkv(hidden_states)
        query_states, key_states, value_states = self._preprocess_qkv(query_states, key_states, value_states)
        query_states = query_states.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        key_states = key_states.reshape(batch_size, sequence_length, self.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(batch_size, sequence_length, self.num_key_value_heads, self.head_dim)
        query_states, key_states, value_states = self._postprocess_qkv(query_states, key_states, value_states)
        query_states, key_states, value_states = self.apply_qkv_shardings(query_states, key_states, value_states)
        query_states, key_states = self._apply_rotary(query_states, key_states, position_ids, frequencies)

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

        attentions = self.attention_performer.forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            mode=mode,
            bias=block_bias,
            cache_metadata=cache_metadata,
            cache_view=cache_view,
            init_bias=init_attention_bias,
            mask_info=mask_info,
            causal=self.causal,
            output_attentions=output_attentions,
        )

        attn_output = self.shard_attention_prod(self._merge_heads(attentions.attention_outputs))
        attn_output = checkpoint_name(self.output_projection(attn_output), "attn_output")
        attn_output = self.shard_attention_prod(attn_output)

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class MiniMaxM3VLDecoderLayer(spx.Module):
    """Single MiniMax M3 decoder layer.

    Standard pre-norm residual block: ``x + attn(norm(x))`` followed by
    ``x + mlp(norm(x))``, where ``mlp`` is either the dense clamped
    SwiGLU-OAI MLP or the sparse :class:`MiniMaxM3VLSparseMoeBlock` depending
    on ``config.mlp_layer_types[layer_idx]``. All norms are Gemma-style
    ``(1 + weight)`` RMSNorms.
    """

    def __init__(
        self,
        config: MiniMaxM3VLTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize a decoder layer.

        Args:
            config (MiniMaxM3VLTextConfig): Model configuration.
            dtype (jnp.dtype, optional): Computation dtype. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Parameter dtype. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Matmul precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
            layer_idx (int): Index of this layer.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        self.self_attn = MiniMaxM3VLAttention(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )
        self.is_moe = config.mlp_layer_types[layer_idx] == "sparse"
        if self.is_moe:
            self.mlp = MiniMaxM3VLSparseMoeBlock(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        else:
            self.mlp = MiniMaxM3VLDenseMLP(
                config=config,
                intermediate_size=config.dense_intermediate_size,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        self.input_layernorm = MiniMaxM3VLRMSNorm(config, param_dtype=param_dtype)
        self.post_attention_layernorm = MiniMaxM3VLRMSNorm(config, param_dtype=param_dtype)

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
            hidden_states (Array): Input tensor ``[batch, seq_len, hidden_dim]``.
            mask_info (MaskInfo): Attention mask information.
            position_ids (Array): Position indices ``[batch, seq_len]``.
            mode (RUNTIME_MODE_TYPES): Runtime mode (train, decode, etc.).
            cache_view: Optional cache view. Defaults to None.
            cache_metadata: Optional cache metadata. Defaults to None.
            output_attentions (bool, optional): Whether to return attention
                weights. Defaults to False.
            output_router_logits (bool, optional): Whether to return router
                logits for MoE layers. Defaults to False.
            frequencies (Array | None, optional): Precomputed RoPE frequencies.

        Returns:
            DecoderLayerOutput: Hidden states, attention weights, router
                logits, and cache view.
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


@register_module(TaskType.BASE_MODULE, config=MiniMaxM3VLTextConfig, model_type="minimax_m3_vl_text")
class MiniMaxM3VLTextModel(EasyDeLBaseModule):
    """MiniMax M3 text decoder trunk.

    Token embedding, ``num_hidden_layers`` :class:`MiniMaxM3VLDecoderLayer`
    blocks (dense/sparse MLP scheduled by ``config.mlp_layer_types``; full or
    block-sparse attention scheduled by ``config.layer_types``), and a final
    Gemma-style RMSNorm. Full-attention configurations use the standard
    :class:`TransformerCache`; configurations containing
    ``"minimax_m3_sparse"`` layers are stateless-only (no decode cache).

    Attributes:
        config (MiniMaxM3VLTextConfig): Configuration for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.
    """

    def __init__(
        self,
        config: MiniMaxM3VLTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the text decoder trunk.

        Args:
            config (MiniMaxM3VLTextConfig): Model configuration.
            dtype (jnp.dtype, optional): Computation dtype. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Parameter dtype. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Matmul precision. Defaults to None.
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
            MiniMaxM3VLDecoderLayer,
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
            self.norm = MiniMaxM3VLRMSNorm(config, param_dtype=param_dtype)

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
        """Forward pass through the text decoder.

        Args:
            input_ids (Array | None, optional): Token IDs ``[batch, seq_len]``.
                Mutually exclusive with ``inputs_embeds``.
            inputs_embeds (Array | None, optional): Pre-computed embeddings.
            attention_mask (Array | None, optional): Padding mask ``[batch, seq_len]``.
            mask_info (MaskInfo | None, optional): Advanced mask information.
            position_ids (Array | None, optional): Position indices.
            output_attentions (bool | None, optional): Whether to return
                per-layer attention weights.
            output_hidden_states (bool | None, optional): Whether to return
                per-layer hidden states.
            output_router_logits (bool | None, optional): Whether to return
                MoE router logits.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode.
            past_key_values: Optional cache (full-attention configs only).
            cache_metadata: Optional cache metadata.

        Returns:
            MoeModelOutput: last_hidden_state, optional per-layer states,
                cache, and router logits.

        Raises:
            ValueError: If both or neither of ``input_ids``/``inputs_embeds``
                are provided.
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

            Carry layout: ``(hidden_states, cv, all_hidden_states,
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

    def init_cache(self, batch_size, max_length, starts=None, shardings=None, pad_token_id=None):
        """Initialize the KV cache for autoregressive generation.

        Args:
            batch_size (int): Batch size for the cache.
            max_length (int): Maximum sequence length to cache.
            starts (int | None, optional): Starting positions. Defaults to None.
            shardings (Any | None, optional): Cache shardings. Defaults to None.
            pad_token_id (int | None, optional): Padding token ID. Defaults to None.

        Returns:
            TransformerCache: Initialized cache.

        Raises:
            NotImplementedError: If the configuration contains
                ``"minimax_m3_sparse"`` layers — the Lightning Indexer's key
                history is not cached yet, so sparse configs are
                stateless-only.
        """
        if any(t == "minimax_m3_sparse" for t in self.config.layer_types):
            raise NotImplementedError(
                "MiniMax M3 sparse-attention layers ('minimax_m3_sparse' in config.layer_types) do not support "
                "KV-cache decoding yet: the Lightning Indexer's key history is not cached. Run stateless forwards "
                "or use a full-attention configuration."
            )
        return super().init_cache(batch_size, max_length, starts, shardings, pad_token_id)

    def get_encoder(self):
        """Return the encoder part of the model's graph definition.

        Raises:
            NotImplementedError: Decoder-only models have no encoder.
        """
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        """Return the decoder part of the model's graph definition."""
        return self

    def get_lm_head(self):
        """Return the language model head of the module.

        Raises:
            NotImplementedError: Base models have no LM head.
        """
        raise NotImplementedError("The base model does not have a language model head.")

    def get_embedding(self):
        """Return the embedding layer of the module."""
        return self.embed_tokens


@register_module(TaskType.CAUSAL_LM, config=MiniMaxM3VLTextConfig, model_type="minimax_m3_vl_text")
class MiniMaxM3VLForCausalLM(BaseCausalLMModule[MiniMaxM3VLTextModel, MiniMaxM3VLTextConfig]):  # type: ignore
    """MiniMax M3 text decoder with a language modeling head.

    Load-balance for the sigmoid router is steered by the
    ``e_score_correction_bias`` buffer; no auxiliary router loss is computed
    unless ``router_aux_loss_coef`` is consumed by the training loop.

    Attributes:
        config (MiniMaxM3VLTextConfig): Configuration for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "minimax_m3_vl_text"
    _config_class = MiniMaxM3VLTextConfig

    def __init__(
        self,
        config: MiniMaxM3VLTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the causal language model.

        Args:
            config (MiniMaxM3VLTextConfig): Model configuration.
            dtype (jnp.dtype, optional): Computation dtype. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Parameter dtype. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Matmul precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=MiniMaxM3VLTextModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
            router_aux_loss_coef=getattr(config, "router_aux_loss_coef", None),
        )

    def init_cache(self, batch_size, max_length, starts=None, shardings=None, pad_token_id=None):
        """Initialize the KV cache for autoregressive generation.

        Args:
            batch_size (int): Batch size for the cache.
            max_length (int): Maximum sequence length to cache.
            starts (int | None, optional): Starting positions. Defaults to None.
            shardings (Any | None, optional): Cache shardings. Defaults to None.
            pad_token_id (int | None, optional): Padding token ID. Defaults to None.

        Returns:
            TransformerCache: Initialized cache.

        Raises:
            NotImplementedError: If the configuration contains
                ``"minimax_m3_sparse"`` layers (stateless-only; see
                :meth:`MiniMaxM3VLTextModel.init_cache`).
        """
        if any(t == "minimax_m3_sparse" for t in self.config.layer_types):
            raise NotImplementedError(
                "MiniMax M3 sparse-attention layers ('minimax_m3_sparse' in config.layer_types) do not support "
                "KV-cache decoding yet: the Lightning Indexer's key history is not cached. Run stateless forwards "
                "or use a full-attention configuration."
            )
        return super().init_cache(batch_size, max_length, starts, shardings, pad_token_id)

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
        """Forward pass through the causal language model.

        Args:
            input_ids (Array | None, optional): Token IDs ``[batch, seq_len]``.
            inputs_embeds (Array | None, optional): Pre-computed embeddings.
            attention_mask (Array | None, optional): Padding mask.
            mask_info (MaskInfo | None, optional): Advanced mask information.
            position_ids (Array | None, optional): Position indices.
            output_attentions (bool | None, optional): Whether to return
                attention weights.
            output_hidden_states (bool | None, optional): Whether to return
                per-layer hidden states.
            output_router_logits (bool | None, optional): Whether to return
                MoE router logits.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode.
            past_key_values: Optional cache (full-attention configs only).
            cache_metadata: Optional cache metadata.
            apply_lm_head (bool, optional): Whether to apply the LM head.
                Defaults to True.

        Returns:
            MoeCausalLMOutput: Logits, cache, hidden states, attentions and
                router logits.
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


def _vision_rope_tables(
    grid_thw,
    head_dim: int,
    theta: float,
    spatial_merge_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the vision tower's 3D rotary cos/sin tables from patch grids.

    ``2 * (head_dim // 2)`` rotary channels are split evenly across the
    T/H/W axes (each rounded down to a multiple of 2), giving ``axis_dim``
    channels per axis; per-patch (t, h, w) grid coordinates (laid out
    block-major over ``m x m`` spatial-merge blocks) scale each axis' band of
    frequencies, and the concatenated bands are duplicated to pair with the
    rotate-half convention. Any head channels past ``3 * axis_dim`` are never
    rotated.

    Args:
        grid_thw: Concrete ``(num_images, 3)`` array of (t, h, w) patch grids.
        head_dim: Per-head channel dimension of the vision attention.
        theta: Rotary base frequency.
        spatial_merge_size: Side length of the spatial-merge block.

    Returns:
        Tuple ``(cos, sin)`` of shape ``[total_patches, 3 * axis_dim]`` in
        float32.
    """
    grid_thw = np.asarray(grid_thw)
    rope_dims = 2 * (head_dim // 2)
    axis_dim = 2 * ((rope_dims // 3) // 2)
    inv_freq = 1.0 / (theta ** (np.arange(0, axis_dim, 2, dtype=np.float32) / axis_dim))

    coords = []
    for t, h, w in grid_thw.tolist():
        hpos, wpos = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        block_shape = (h // spatial_merge_size, spatial_merge_size, w // spatial_merge_size, spatial_merge_size)
        hpos = hpos.reshape(block_shape).transpose(0, 2, 1, 3).reshape(-1)
        wpos = wpos.reshape(block_shape).transpose(0, 2, 1, 3).reshape(-1)
        tpos = np.repeat(np.arange(t), h * w)
        coords.append(np.stack([tpos, np.tile(hpos, t), np.tile(wpos, t)], axis=-1))
    coords = np.concatenate(coords, axis=0).astype(np.float32)

    freqs = (coords[:, :, None] * inv_freq[None, None, :]).reshape(coords.shape[0], -1)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb), np.sin(emb)


class MiniMaxM3VLVisionAttention(AttentionModule):
    """CLIP-style vision self-attention with 3D rotary embeddings.

    Standard biased MHA (fused QKV projection loaded from HF's separate
    ``q_proj``/``k_proj``/``v_proj`` tensors, ``out_proj`` output) whose
    queries and keys are rotated by the tower's 3D (T/H/W) RoPE before a
    full bidirectional attention over the whole patch sequence.
    """

    def __init__(
        self,
        config: MiniMaxM3VLVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the vision attention layer.

        Args:
            config (MiniMaxM3VLVisionConfig): Vision configuration.
            dtype (jnp.dtype, optional): Computation dtype. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Parameter dtype. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Matmul precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(config=config)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"hidden_size must be divisible by num_attention_heads (got `hidden_size`: {self.embed_dim} "
                f"and `num_attention_heads`: {self.num_heads})."
            )
        linear_class = partial(
            ColumnParallelLinear,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
        )
        qkv_layout = dense_qkv_layout(self.embed_dim, self.embed_dim)
        self.qkv_proj = linear_class(self.embed_dim, qkv_layout.segment_sizes, layout=qkv_layout)
        self.out_proj = linear_class(self.embed_dim, self.embed_dim)
        self.attention_performer = FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=config.attention_dropout,
            attn_mechanism="vanilla",
            requires_cache=False,
        )

    @property
    def reform_param(self):
        """Reform rules loading HF's separate q/k/v tensors into the fused QKV."""
        return self.qkv_proj.build_reform_param("qkv_proj", config=self.config, include_bias=True)

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        position_embeddings: tuple[Array, Array],
    ) -> Array:
        """Apply vision self-attention with 3D rotary embeddings.

        Args:
            hidden_states: Patch features ``[batch, num_patches, hidden]``.
            position_embeddings: Precomputed ``(cos, sin)`` tables of shape
                ``[num_patches, 3 * axis_dim]``.

        Returns:
            Attention output ``[batch, num_patches, hidden]``.
        """
        qkv = checkpoint_name(self.qkv_proj(hidden_states), name="attn_qkv")
        query, key, value = self.qkv_proj.split(qkv, config=self.config)
        shape = (*hidden_states.shape[:2], self.num_heads, self.head_dim)
        query = query.reshape(shape)
        key = key.reshape(shape)
        value = value.reshape(shape)

        cos, sin = position_embeddings
        cos = cos[None, :, None, :].astype(query.dtype)
        sin = sin[None, :, None, :].astype(query.dtype)
        query = _apply_partial_rotary(query, cos, sin)
        key = _apply_partial_rotary(key, cos, sin)

        attentions = self.attention_performer.forward(
            query_states=query,
            key_states=key,
            value_states=value,
            mode=common_types.MODE_TRAIN,
            bias=None,
            mask_info=None,
            causal=False,
        )
        attn_output = self.shard_attention_prod(
            attentions.attention_outputs.reshape(*hidden_states.shape[:2], self.embed_dim)
        )
        attn_output = checkpoint_name(self.out_proj(attn_output), name="attn_output")
        return self.shard_attention_prod(attn_output)


class MiniMaxM3VLVisionMLP(spx.Module):
    """CLIP-style two-layer GELU MLP for the vision encoder (``fc1``/``fc2``)."""

    def __init__(
        self,
        config: MiniMaxM3VLVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the vision MLP.

        Args:
            config (MiniMaxM3VLVisionConfig): Vision configuration.
            dtype (jnp.dtype, optional): Computation dtype. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Parameter dtype. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Matmul precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        linear_class = partial(
            ColumnParallelLinear,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
        )
        self.fc1 = linear_class(config.hidden_size, config.intermediate_size)
        self.fc2 = linear_class(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> Array:
        """Apply the feed-forward transformation.

        Args:
            hidden_states: Input tensor ``[batch, num_patches, hidden]``.

        Returns:
            Transformed tensor of the same shape.
        """
        return self.fc2(self.activation_fn(self.fc1(hidden_states)))


class MiniMaxM3VLVisionEncoderLayer(spx.Module):
    """Pre-LN CLIP encoder layer with 3D-RoPE attention.

    ``x + attn(ln1(x))`` followed by ``x + mlp(ln2(x))`` with biased
    LayerNorms (HF names ``layer_norm1`` / ``layer_norm2``).
    """

    def __init__(
        self,
        config: MiniMaxM3VLVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the encoder layer.

        Args:
            config (MiniMaxM3VLVisionConfig): Vision configuration.
            dtype (jnp.dtype, optional): Computation dtype. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Parameter dtype. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Matmul precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.self_attn = MiniMaxM3VLVisionAttention(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.mlp = MiniMaxM3VLVisionMLP(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.layer_norm1 = LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.layer_norm2 = LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        position_embeddings: tuple[Array, Array],
    ) -> Array:
        """Forward pass through the encoder layer.

        Args:
            hidden_states: Patch features ``[batch, num_patches, hidden]``.
            position_embeddings: Precomputed ``(cos, sin)`` rotary tables.

        Returns:
            Output patch features of the same shape.
        """
        hidden_states = hidden_states + self.self_attn(self.layer_norm1(hidden_states), position_embeddings)
        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


@register_module(TaskType.BASE_VISION, config=MiniMaxM3VLVisionConfig, model_type="minimax_m3_vl_vision")
class MiniMaxM3VLVisionModel(EasyDeLBaseModule):
    """MiniMax M3 VL vision tower: Conv3d patch embed + 3D RoPE CLIP encoder.

    A ``pre_layrnorm`` (HF's spelling) follows the bias-free Conv3d patch
    embedding; the encoder attends over the entire concatenated patch
    sequence with no mask and no final LayerNorm.

    Attributes:
        config (MiniMaxM3VLVisionConfig): Vision configuration.
        dtype (jnp.dtype): Computation dtype.
        param_dtype (jnp.dtype): Parameter dtype.
        precision: Matmul precision.
    """

    config_class = MiniMaxM3VLVisionConfig

    def __init__(
        self,
        config: MiniMaxM3VLVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the vision tower.

        Args:
            config (MiniMaxM3VLVisionConfig): Vision configuration.
            dtype (jnp.dtype, optional): Computation dtype. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Parameter dtype. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Matmul precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.embeddings = Qwen2VLPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.num_channels,
            embed_dim=config.hidden_size,
            precision=precision,
            dtype=dtype,
            param_dtype=param_dtype,
            partition_manager=config.runtime_sharding_resolver,
            rngs=rngs,
        )
        self.pre_layrnorm = LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        block = auto_remat(
            MiniMaxM3VLVisionEncoderLayer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.layers = nn.ModuleList(
            [
                block(
                    config=config,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    precision=precision,
                    rngs=rngs,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.head_dim = config.hidden_size // config.num_attention_heads

    def forward(
        self,
        pixel_values: Float[Array, "num_patches patch_features"],
        grid_thw,
        output_hidden_states: bool = False,
    ) -> BaseModelOutput:
        """Encode flattened image/video patches.

        Args:
            pixel_values: Flattened patches
                ``[total_patches, channels * temporal_patch * patch * patch]``.
            grid_thw: Concrete ``(num_images, 3)`` array of per-image
                (t, h, w) patch grids used to build the 3D RoPE.
            output_hidden_states (bool, optional): Whether to also return the
                per-layer hidden states. Defaults to False.

        Returns:
            BaseModelOutput: ``last_hidden_state`` of shape
                ``[1, total_patches, hidden]`` (no final LayerNorm) and the
                optional per-layer states.
        """
        embeds = self.embeddings(pixel_values)
        cos, sin = _vision_rope_tables(
            grid_thw,
            head_dim=self.head_dim,
            theta=self.config.rope_theta,
            spatial_merge_size=self.config.spatial_merge_size,
        )
        position_embeddings = (
            jnp.asarray(cos, dtype=self.dtype),
            jnp.asarray(sin, dtype=self.dtype),
        )
        hidden_states = self.pre_layrnorm(embeds)[None, ...]
        all_hidden_states = () if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states = layer(hidden_states, position_embeddings)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class MiniMaxM3VLMultiModalProjector(spx.Module):
    """Two-stage GELU projector with spatial patch merging.

    Projects each vision patch from ``vision_config.hidden_size`` to
    ``text_config.hidden_size`` (``linear_1`` -> GELU -> ``linear_2``), then
    groups ``spatial_merge_size**2`` neighbouring patches into the channel dim
    and fuses them back to a single text-width token
    (``merge_linear_1`` -> GELU -> ``merge_linear_2``).
    """

    def __init__(
        self,
        config: MiniMaxM3VLConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the projector.

        Args:
            config (MiniMaxM3VLConfig): Composite configuration.
            dtype (jnp.dtype, optional): Computation dtype. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Parameter dtype. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Matmul precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        text_hidden = config.text_config.hidden_size
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        linear_class = partial(
            ColumnParallelLinear,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(0.02),
        )
        self.linear_1 = linear_class(config.vision_config.hidden_size, config.projector_hidden_size)
        self.linear_2 = linear_class(config.projector_hidden_size, text_hidden)
        self.merge_linear_1 = linear_class(config.merged_hidden_size, config.projector_hidden_size)
        self.merge_linear_2 = linear_class(config.projector_hidden_size, text_hidden)
        self.act = ACT2FN["gelu"]
        self.merge_act = ACT2FN["gelu"]

    def forward(self, image_features: Float[Array, "num_patches vision_hidden"]) -> Array:
        """Project and spatially merge vision features into text tokens.

        Args:
            image_features: Vision-tower features ``[num_patches, vision_hidden]``.

        Returns:
            Merged text-width tokens
            ``[num_patches // spatial_merge_size**2, text_hidden]``.
        """
        hidden_states = self.linear_2(self.act(self.linear_1(image_features)))
        hidden_states = hidden_states.reshape(hidden_states.shape[0] // (self.spatial_merge_size**2), -1)
        return self.merge_linear_2(self.merge_act(self.merge_linear_1(hidden_states)))


@auto_pytree
class MiniMaxM3VLModelOutputWithPast(ModelOutput):
    """Output of :class:`MiniMaxM3VLModel`.

    Attributes:
        last_hidden_state: Final decoder hidden states.
        past_key_values: KV cache (full-attention configs only).
        hidden_states: Optional per-layer hidden states.
        attentions: Optional per-layer attention weights.
        router_logits: Optional MoE router logits.
        image_hidden_states: Projected image features scattered into the text
            stream, when images were provided.
        video_hidden_states: Projected video features, when videos were
            provided.
    """

    last_hidden_state: Array | None = None
    past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None
    hidden_states: tuple[Array, ...] | None = None
    attentions: tuple[Array, ...] | None = None
    router_logits: tuple[Array, ...] | None = None
    image_hidden_states: Array | None = None
    video_hidden_states: Array | None = None


@register_module(TaskType.VISION_LM, config=MiniMaxM3VLConfig, model_type="minimax_m3_vl")
class MiniMaxM3VLModel(EasyDeLBaseModule):
    """MiniMax M3 VL backbone: vision tower + projector + text decoder (no LM head).

    Vision features are projected and spatially merged by
    :class:`MiniMaxM3VLMultiModalProjector` and scattered into the text
    embedding sequence at ``image_token_index`` / ``video_token_index``
    placeholder positions; the merged sequence runs through
    :class:`MiniMaxM3VLTextModel` with standard 1D positions (no mRoPE).

    Attributes:
        vision_tower (MiniMaxM3VLVisionModel): CLIP-style tower with 3D RoPE.
        multi_modal_projector (MiniMaxM3VLMultiModalProjector): Patch-merge MLP.
        language_model (MiniMaxM3VLTextModel): MoE text decoder trunk.
    """

    def __init__(
        self,
        config: MiniMaxM3VLConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the multimodal backbone.

        Args:
            config (MiniMaxM3VLConfig): Composite configuration.
            dtype (jnp.dtype, optional): Computation dtype. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Parameter dtype. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Matmul precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.vision_tower = MiniMaxM3VLVisionModel(
            config.vision_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.multi_modal_projector = MiniMaxM3VLMultiModalProjector(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.language_model = MiniMaxM3VLTextModel(
            config.text_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def get_image_features(self, pixel_values: Array, image_grid_thw=None, **kwargs) -> Array:
        """Encode and project image patches into text-width tokens.

        Args:
            pixel_values: Flattened image patches
                ``[total_patches, channels * temporal_patch * patch * patch]``.
            image_grid_thw: Concrete ``(num_images, 3)`` (t, h, w) patch grids.
            **kwargs: Unused model-specific arguments.

        Returns:
            Projected, spatially merged features
            ``[total_patches // merge**2, text_hidden]``.
        """
        vision_outputs = self.vision_tower(pixel_values=pixel_values, grid_thw=image_grid_thw)
        return self.multi_modal_projector(vision_outputs.last_hidden_state[0])

    def get_video_features(self, pixel_values_videos: Array, video_grid_thw=None, **kwargs) -> Array:
        """Encode and project video patches into text-width tokens.

        Video frames flow through the same vision pipeline as images (the
        tower is grid-agnostic); only the placeholder token differs.

        Args:
            pixel_values_videos: Flattened video patches.
            video_grid_thw: Concrete ``(num_videos, 3)`` patch grids.
            **kwargs: Unused model-specific arguments.

        Returns:
            Projected, spatially merged features ``[num_tokens, text_hidden]``.
        """
        vision_outputs = self.vision_tower(pixel_values=pixel_values_videos, grid_thw=video_grid_thw)
        return self.multi_modal_projector(vision_outputs.last_hidden_state[0])

    def compute_embedding(
        self,
        input_ids: Int[Array, "batch seq_len"] | None,
        *,
        image_features: Array | None = None,
        video_features: Array | None = None,
        pixel_values: Array | None = None,
        pixel_values_videos: Array | None = None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ) -> Array:
        """Compute input embeddings with merged image/video features.

        Args:
            input_ids (Array): Token IDs ``[batch, seq_len]``.
            image_features (Array | None, optional): Pre-extracted image
                features; computed from ``pixel_values`` when None.
            video_features (Array | None, optional): Pre-extracted video
                features; computed from ``pixel_values_videos`` when None.
            pixel_values (Array | None, optional): Raw flattened image patches.
            pixel_values_videos (Array | None, optional): Raw flattened video
                patches.
            image_grid_thw: Per-image (t, h, w) patch grids.
            video_grid_thw: Per-video (t, h, w) patch grids.
            **kwargs: Unused.

        Returns:
            Array: Embeddings ``[batch, seq_len, text_hidden]`` with vision
                features merged at placeholder positions.

        Raises:
            ValueError: If ``input_ids`` is None.
        """
        if input_ids is None:
            raise ValueError("`input_ids` must be provided when calling `compute_embedding`.")

        vocab_size = self.config.get_text_config().vocab_size
        image_token_index = self.config.image_token_index
        video_token_index = self.config.video_token_index
        llm_input_ids = input_ids
        if image_token_index >= vocab_size:
            llm_input_ids = jnp.where(llm_input_ids == image_token_index, 0, llm_input_ids)
        if video_token_index >= vocab_size:
            llm_input_ids = jnp.where(llm_input_ids == video_token_index, 0, llm_input_ids)

        inputs_embeds = super().compute_embedding(llm_input_ids)

        if image_features is None and pixel_values is not None:
            image_features = self.get_image_features(pixel_values, image_grid_thw=image_grid_thw)
        if video_features is None and pixel_values_videos is not None:
            video_features = self.get_video_features(pixel_values_videos, video_grid_thw=video_grid_thw)

        if image_features is not None:
            inputs_embeds = BaseVisionLanguageModule.merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=image_features.reshape(-1, image_features.shape[-1]).astype(inputs_embeds.dtype),
                placeholder_token_id=image_token_index,
            )
        if video_features is not None:
            inputs_embeds = BaseVisionLanguageModule.merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=video_features.reshape(-1, video_features.shape[-1]).astype(inputs_embeds.dtype),
                placeholder_token_id=video_token_index,
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        pixel_values: Array | None = None,
        pixel_values_videos: Array | None = None,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        **lm_kwargs,
    ) -> MiniMaxM3VLModelOutputWithPast:
        """Forward pass through the multimodal backbone.

        Args:
            input_ids (Array | None, optional): Token IDs ``[batch, seq_len]``.
            pixel_values (Array | None, optional): Flattened image patches.
            pixel_values_videos (Array | None, optional): Flattened video
                patches.
            image_grid_thw: Per-image (t, h, w) patch grids.
            video_grid_thw: Per-video (t, h, w) patch grids.
            attention_mask (Array | None, optional): Padding mask.
            mask_info (MaskInfo | None, optional): Advanced mask information.
            position_ids (Array | None, optional): Position indices.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode.
            past_key_values: Optional cache.
            cache_metadata: Optional cache metadata.
            inputs_embeds (Array | None, optional): Pre-computed embeddings.
            output_attentions (bool | None, optional): Whether to return
                attention weights.
            output_hidden_states (bool | None, optional): Whether to return
                per-layer hidden states.
            output_router_logits (bool | None, optional): Whether to return
                MoE router logits.
            **lm_kwargs: Extra arguments forwarded to the language model.

        Returns:
            MiniMaxM3VLModelOutputWithPast: Backbone outputs with cache,
                router logits and vision hidden states.

        Raises:
            ValueError: If both or neither of ``input_ids``/``inputs_embeds``
                are provided, or pixels are given without ``input_ids``.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if (pixel_values is not None or pixel_values_videos is not None) and input_ids is None:
            raise ValueError("`input_ids` must be provided when pixel inputs are given.")

        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values, image_grid_thw=image_grid_thw)
        video_features = None
        if pixel_values_videos is not None:
            video_features = self.get_video_features(pixel_values_videos, video_grid_thw=video_grid_thw)

        if inputs_embeds is None:
            inputs_embeds = self.compute_embedding(
                input_ids,
                image_features=image_features,
                video_features=video_features,
            )

        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            **lm_kwargs,
        )

        return MiniMaxM3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            image_hidden_states=image_features,
            video_hidden_states=video_features,
        )

    def init_cache(self, batch_size, max_length, starts=None, shardings=None, pad_token_id=None):
        """Initialize the KV cache by delegating to the language model.

        Args:
            batch_size (int): Batch size for the cache.
            max_length (int): Maximum sequence length to cache.
            starts (int | None, optional): Starting positions. Defaults to None.
            shardings (Any | None, optional): Cache shardings. Defaults to None.
            pad_token_id (int | None, optional): Padding token ID. Defaults to None.

        Returns:
            TransformerCache: Initialized cache (full-attention configs only).
        """
        return self.language_model.init_cache(batch_size, max_length, starts, shardings, pad_token_id)

    def prepare_inputs_for_generation(
        self,
        input_ids: Int[Array, "batch seq_len"],
        max_length: int,
        pad_token_id: int,
        starts: int | None = None,
        pixel_values: Array | None = None,
        pixel_values_videos: Array | None = None,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
    ):
        """Prepare inputs for autoregressive generation.

        Args:
            input_ids (Array): Prompt token IDs ``[batch, seq_len]``.
            max_length (int): Maximum generation length.
            pad_token_id (int): Padding token ID.
            starts (int | None, optional): Starting positions. Defaults to None.
            pixel_values (Array | None, optional): Flattened image patches.
            pixel_values_videos (Array | None, optional): Flattened video patches.
            image_grid_thw: Per-image patch grids.
            video_grid_thw: Per-video patch grids.
            attention_mask (Array | None, optional): Padding mask.

        Returns:
            dict: Model inputs ready for generation.
        """
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            max_length=max_length,
            pad_token_id=pad_token_id,
            starts=starts,
            attention_mask=attention_mask,
        )
        model_inputs["pixel_values"] = pixel_values
        model_inputs["pixel_values_videos"] = pixel_values_videos
        model_inputs["image_grid_thw"] = image_grid_thw
        model_inputs["video_grid_thw"] = video_grid_thw
        return model_inputs

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        """Update inputs for the next generation step.

        Pixel inputs are merged into the sequence on the first step only.

        Args:
            model_outputs: Outputs from the previous step.
            model_kwargs (dict): Current model keyword arguments.

        Returns:
            dict: Updated model keyword arguments.
        """
        model_kwargs = self.language_model.update_inputs_for_generation(model_outputs, model_kwargs)
        for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
            model_kwargs.pop(key, None)  # only effect first iter
        return model_kwargs

    def get_encoder(self):
        """Return the vision tower (the encoder component)."""
        return self.vision_tower

    def get_decoder(self):
        """Return the text decoder."""
        return self.language_model.get_decoder()

    def get_lm_head(self):
        """Return the language model head of the module.

        Raises:
            NotImplementedError: The backbone has no LM head.
        """
        raise NotImplementedError("The base model does not have a language model head.")

    def get_embedding(self):
        """Return the text embedding layer."""
        return self.language_model.get_embedding()


@register_module(TaskType.IMAGE_TEXT_TO_TEXT, config=MiniMaxM3VLConfig, model_type="minimax_m3_vl")
class MiniMaxM3VLForConditionalGeneration(BaseVisionLanguageModule[MiniMaxM3VLModel, MiniMaxM3VLConfig]):  # type: ignore
    """MiniMax M3 VL model with an LM head for conditional generation.

    Combines the vision tower, projector and MoE text decoder with a language
    modeling head over the text vocabulary. Uses standard 1D RoPE positions
    (no mRoPE); image and video placeholders are scattered Llava-style.

    Attributes:
        _task_type: IMAGE_TEXT_TO_TEXT task type.
        _model_type: ``"minimax_m3_vl"``.
        _supports_video: True (videos share the image pipeline).
        _uses_mrope: False (standard RoPE).
    """

    _task_type = TaskType.IMAGE_TEXT_TO_TEXT
    _model_type = "minimax_m3_vl"
    _config_class = MiniMaxM3VLConfig
    _auto_register = False  # Already registered via decorator
    _supports_video = True
    _uses_mrope = False

    _vision_tower_name = "vision_tower"
    _projector_name = "multi_modal_projector"
    _language_model_name = "language_model"

    loss_type = "ForCausalLM"

    def __init__(
        self,
        config: MiniMaxM3VLConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the conditional-generation model.

        Args:
            config (MiniMaxM3VLConfig): Composite configuration.
            dtype (jnp.dtype, optional): Computation dtype. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Parameter dtype. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Matmul precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=MiniMaxM3VLModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            image_token_index=config.image_token_index,
            video_token_index=config.video_token_index,
            spatial_merge_size=config.vision_config.spatial_merge_size,
            router_aux_loss_coef=getattr(config.text_config, "router_aux_loss_coef", None),
            tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
            lm_head_bias=False,
        )
        self.vocab_size = config.text_config.vocab_size

    @property
    def vision_tower(self):
        """Return the vision tower from the base model."""
        return self.base_model.vision_tower

    @property
    def multi_modal_projector(self):
        """Return the multimodal projector from the base model."""
        return self.base_model.multi_modal_projector

    @property
    def language_model(self):
        """Return the text decoder from the base model."""
        return self.base_model.language_model

    def get_image_features(self, pixel_values: Array, image_grid_thw=None, **kwargs):
        """Extract projected image features via the base model.

        Args:
            pixel_values (Array): Flattened image patches.
            image_grid_thw: Per-image (t, h, w) patch grids.
            **kwargs: Unused.

        Returns:
            Array: Projected, spatially merged image features.
        """
        return self.base_model.get_image_features(pixel_values, image_grid_thw=image_grid_thw)

    def get_video_features(self, pixel_values_videos: Array, video_grid_thw=None, **kwargs):
        """Extract projected video features via the base model.

        Args:
            pixel_values_videos (Array): Flattened video patches.
            video_grid_thw: Per-video (t, h, w) patch grids.
            **kwargs: Unused.

        Returns:
            Array: Projected, spatially merged video features.
        """
        return self.base_model.get_video_features(pixel_values_videos, video_grid_thw=video_grid_thw)

    def compute_embedding(self, input_ids, *args, **kwargs):
        """Compute multimodal embeddings via the base model.

        Args:
            input_ids: Input token IDs.
            *args: Positional arguments passed to the base model.
            **kwargs: Keyword arguments passed to the base model.

        Returns:
            Array: Merged multimodal embeddings.
        """
        return self.base_model.compute_embedding(input_ids, *args, **kwargs)

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        pixel_values: Array | None = None,
        pixel_values_videos: Array | None = None,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        **kwargs,
    ) -> VLMCausalLMOutput:
        """Forward pass for conditional generation.

        Args:
            input_ids (Array | None, optional): Token IDs ``[batch, seq_len]``.
            pixel_values (Array | None, optional): Flattened image patches.
            pixel_values_videos (Array | None, optional): Flattened video patches.
            image_grid_thw: Per-image (t, h, w) patch grids.
            video_grid_thw: Per-video (t, h, w) patch grids.
            attention_mask (Array | None, optional): Padding mask.
            mask_info (MaskInfo | None, optional): Advanced mask information.
            position_ids (Array | None, optional): Position indices.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode.
            past_key_values: Optional cache.
            cache_metadata: Optional cache metadata.
            apply_lm_head (bool, optional): Whether to apply the LM head.
                Defaults to True.
            inputs_embeds (Array | None, optional): Pre-computed embeddings.
            output_attentions (bool | None, optional): Whether to return
                attention weights.
            output_hidden_states (bool | None, optional): Whether to return
                per-layer hidden states.
            output_router_logits (bool | None, optional): Whether to return
                MoE router logits.
            **kwargs: Additional unused arguments.

        Returns:
            VLMCausalLMOutput: Logits and backbone outputs.
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
        )

        hidden_states = apply_logical_sharding(
            outputs.last_hidden_state,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )

        lm_logits = None
        if apply_lm_head:
            lm_logits = self.compute_lm_logits(hidden_states)
            lm_logits = self.apply_logit_cap(lm_logits)

        return VLMCausalLMOutput(
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            last_hidden_state=hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            image_hidden_states=outputs.image_hidden_states,
        )

    def init_cache(self, batch_size, max_length, starts=None, shardings=None, pad_token_id=None):
        """Initialize the KV cache by delegating to the backbone.

        Args:
            batch_size (int): Batch size for the cache.
            max_length (int): Maximum sequence length to cache.
            starts (int | None, optional): Starting positions. Defaults to None.
            shardings (Any | None, optional): Cache shardings. Defaults to None.
            pad_token_id (int | None, optional): Padding token ID. Defaults to None.

        Returns:
            TransformerCache: Initialized cache (full-attention configs only).
        """
        return self.base_model.init_cache(batch_size, max_length, starts, shardings, pad_token_id)


__all__ = [
    "MiniMaxM3VLForCausalLM",
    "MiniMaxM3VLForConditionalGeneration",
    "MiniMaxM3VLModel",
    "MiniMaxM3VLTextModel",
    "MiniMaxM3VLVisionModel",
]
