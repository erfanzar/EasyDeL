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

"""Inkling 2 and Inkling 2 Loop text decoder and multimodal model implementations.

Composes EasyDeL's reusable building blocks cleanly across packages:

- **Cache**: :class:`~easydel.caching.ConvKVCache` (KV + 4 float32 causal
  short-convolution states per logical layer slot, aliased as
  :class:`Inkling2Cache` / :class:`Inkling2CacheView` /
  :class:`Inkling2CacheConfig`).
- **Residual connections**: :class:`~easydel.layers.residual.BlockAttentionResidual`,
  :class:`~easydel.layers.residual.LoopMessageAttentionResidual`, and
  :class:`~easydel.layers.residual.LoopMessageGate`.
- **Short convolutions**: :class:`~easydel.layers.linear_attention.ResidualShortConvolution`
  and :class:`~easydel.layers.linear_attention.DynamicResidualShortConvolution`.
- **Attention**: :class:`~easydel.layers.attention.AttentionModule` +
  :class:`~easydel.layers.attention.FlexibleAttentionModule` (stage-mesh aware
  under MPMD, wired for dropout, and routed through EasyDeL's registered
  ``vanilla`` operation for 4-D per-token ``softmax_aux`` sinks + relative
  position bias).
- **MoE FFN**: :class:`~easydel.layers.BaseMoeModule` +
  :func:`~easydel.layers.mlp.softcapped_swiglu`.
- **Vision tower**: :mod:`.modeling_inkling2_vision`.
- **Checkpoint conversion**: :mod:`._checkpoint`.
- **Pipeline parallelism (MPMD)**: full stage assignment, stage-local mesh
  contexts, and rank-safe ``sxstage_iter`` stage boundaries carrying both
  ``hidden_states`` and ``block_states`` across stages — including atomic
  single-stage placement of the looped core in :class:`Inkling2LoopTextModel`.
"""

from __future__ import annotations

import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import spectrax as spx
from eformer.pytree import auto_pytree
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import PartitionSpec
from jaxtyping import Array, Bool, Float, Int
from spectrax import apply_logical_sharding, common_types, nn

from easydel.caching import (
    ConvKVCache,
    ConvKVCacheConfig,
    ConvKVCacheView,
    OperationsMetadata,
    TransformerMetadata,
)
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import (
    AttentionLayerOutput,
    DecoderLayerOutput,
    MoeCausalLMOutput,
    MoeModelOutput,
    VLMCausalLMOutput,
)
from easydel.infra.utils import ArrayParam, auto_remat
from easydel.layers import (
    BaseMoeModule,
    BlockAttentionResidual,
    ColumnParallelLinear,
    ColumnParallelMoELinear,
    Embed,
    LoopMessageAttentionResidual,
    LoopMessageGate,
    MoeFusedHooks,
    MoeLoadBalancingStrategy,
    MoeRoutingStrategy,
    RowParallelLinear,
    RowParallelMoELinear,
    softcapped_swiglu,
    split_fused_gate_up_projection,
)
from easydel.layers.attention import AttentionMechanisms, AttentionModule, FlexibleAttentionModule
from easydel.layers.linear_attention import DynamicResidualShortConvolution, ResidualShortConvolution
from easydel.layers.moe import moe_group_topk_select
from easydel.layers.norms import RMSNorm
from easydel.modules._base import BaseCausalLMModule, BaseVisionLanguageModule

from ._checkpoint import (
    FP32_PARAM_PATTERNS,
    INKLING2_CAUSAL_LM_KEY_RULES,
    INKLING2_LOOP_CAUSAL_LM_KEY_RULES,
    INKLING2_LOOP_TEXT_KEY_RULES,
    INKLING2_LOOP_VLM_KEY_RULES,
    INKLING2_LOOP_VLM_MODEL_KEY_RULES,
    INKLING2_TEXT_KEY_RULES,
    INKLING2_VISION_KEY_RULES,
    INKLING2_VLM_KEY_RULES,
    INKLING2_VLM_MODEL_KEY_RULES,
    LOOP_FP32_PARAM_PATTERNS,
    Inkling2KeyNormalizer,
    deinterleave_inkling2_gate_up,
    dequantize_inkling2_experts,
    interleave_inkling2_gate_up,
    merge_inkling2_loop_message_queries,
    pad_inkling2_lm_head,
    split_inkling2_loop_message_queries,
    tp_pack_expert_gate_up,
    tp_unpack_expert_gate_up,
    transpose_expert_3d,
    unpad_inkling2_lm_head,
)
from .inkling2_configuration import (
    Inkling2Config,
    Inkling2LoopConfig,
    Inkling2LoopTextConfig,
    Inkling2TextConfig,
)
from .modeling_inkling2_vision import (
    Inkling2VisionAttention,
    Inkling2VisionEmbeddings,
    Inkling2VisionEncoderLayer,
    Inkling2VisionMerger,
    Inkling2VisionMLP,
    Inkling2VisionModel,
    Inkling2VisionModelOutput,
    Inkling2VisionRotaryEmbedding,
    apply_rotary_pos_emb_vision,
)

# Public family aliases for the reusable shared layers and cache.
Inkling2Cache = ConvKVCache
Inkling2CacheConfig = ConvKVCacheConfig
Inkling2CacheView = ConvKVCacheView
Inkling2ShortConvolution = ResidualShortConvolution
Inkling2DynamicShortConvolution = DynamicResidualShortConvolution
Inkling2BlockAttentionResidual = BlockAttentionResidual
Inkling2LoopMessageAttentionResidual = LoopMessageAttentionResidual
Inkling2LoopMessageGate = LoopMessageGate


def create_inkling2_cache_config(
    config: Inkling2TextConfig,
    batch_size: int,
    sequence_length: int,
    pad_token_id: int | None = None,
) -> ConvKVCacheConfig:
    """Build a :class:`ConvKVCacheConfig` matching an Inkling 2 / Inkling 2 Loop text config.

    Each logical cache slot owns one KV view plus four float32 short-convolution
    buffers: ``(k_sconv, v_sconv, post_attention_conv, post_mlp_conv)``.
    """
    num_slots = config.total_num_hidden_layers
    layer_types = tuple(config.effective_layer_types)
    layer_kv_shapes = tuple(config.layer_kv_shape(i) for i in range(num_slots))
    layer_conv_dims = tuple(
        (kv_h * kv_d, kv_h * kv_d, int(config.hidden_size), int(config.hidden_size)) for kv_h, kv_d in layer_kv_shapes
    )
    layer_sliding_windows = tuple(
        int(config.sliding_window_size) if lt == "hybrid_sliding" else None for lt in layer_types
    )
    return ConvKVCacheConfig.create(
        batch_size=batch_size,
        sequence_length=sequence_length,
        conv_kernel_size=config.conv_kernel_size,
        layer_kv_shapes=layer_kv_shapes,
        layer_conv_dims=layer_conv_dims,
        layer_sliding_windows=layer_sliding_windows,
        pad_token_id=pad_token_id if pad_token_id is not None else config.pad_token_id,
        masking_details=config.get_mask_details(),
    )


# ---------------------------------------------------------------------------
# Norms, L2 clipping, logarithmic scaling, and relative position bias
# ---------------------------------------------------------------------------


def _safe_l2_norm(x_f32: Array) -> Array:
    """Compute the trailing-axis L2 norm with finite zero-input gradients."""
    sq_sum = jnp.sum(jnp.square(x_f32), axis=-1, keepdims=True)
    safe_sq = jnp.where(sq_sum > 0.0, sq_sum, 1.0)
    return jnp.where(sq_sum > 0.0, jnp.sqrt(safe_sq), 0.0)


class Inkling2RMSNorm(RMSNorm):
    """Inkling 2 RMSNorm: fp32 statistics and fp32 scale multiply, cast back once."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: spx.Rngs,
    ):
        super().__init__(
            dim=hidden_size,
            eps=eps,
            dtype=dtype,
            param_dtype=param_dtype,
            with_scale=True,
            fp32_scale=True,
            output_dtype=None,
            rngs=rngs,
        )


class Inkling2UnweightedRMSNorm(RMSNorm):
    """Parameter-free RMSNorm running statistics in fp32 and casting back to input dtype."""

    def __init__(self, eps: float = 1e-6, dtype: jnp.dtype = jnp.bfloat16):
        super().__init__(
            dim=1,
            eps=eps,
            dtype=dtype,
            param_dtype=jnp.float32,
            with_scale=False,
            fp32_scale=True,
            output_dtype=None,
            rngs=spx.Rngs(0),
        )


class Inkling2L2Clip(spx.Module):
    """Scales each trailing vector down to ``threshold`` when its L2 norm exceeds ``threshold``."""

    def __init__(self, threshold: float):
        self.threshold = float(threshold)

    def forward(self, hidden_states: Array) -> Array:
        input_dtype = hidden_states.dtype
        x = hidden_states.astype(jnp.float32)
        norm = _safe_l2_norm(x)
        scale = jnp.minimum(self.threshold / (norm + 1e-6), 1.0)
        return (x * scale).astype(input_dtype)


class Inkling2ScaledL2Clip(spx.Module):
    """Learned per-head/per-channel scale followed by :class:`Inkling2L2Clip`."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        threshold: float,
        param_dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: spx.Rngs,
    ):
        self.scale = ArrayParam.bound(
            shape=(num_heads, head_dim),
            dtype=param_dtype,
            init_method="ones",
            key=rngs.param,
        )
        self.l2_clip = Inkling2L2Clip(threshold)

    def forward(self, hidden_states: Array, position_ids: Array | None = None) -> Array:
        del position_ids
        input_dtype = hidden_states.dtype
        scaled = hidden_states.astype(jnp.float32) * self.scale.value.astype(jnp.float32)
        return self.l2_clip(scaled).astype(input_dtype)


class Inkling2LogScale(spx.Module):
    """Position-dependent logarithmic scaling ``1 + alpha * log(max((pos + 1) / n_floor, 1))``."""

    def __init__(self, config: Inkling2TextConfig):
        self.log_scaling_alpha = float(config.log_scaling_alpha)
        self.log_scaling_n_floor = float(config.log_scaling_n_floor)

    def forward(self, hidden_states: Array, position_ids: Array) -> Array:
        input_dtype = hidden_states.dtype
        ratio = jnp.maximum((position_ids.astype(jnp.float32) + 1.0) / self.log_scaling_n_floor, 1.0)
        tau = 1.0 + self.log_scaling_alpha * jnp.log(ratio)
        return (hidden_states.astype(jnp.float32) * tau[..., None, None]).astype(input_dtype)


class Inkling2LogScaledL2Clip(Inkling2ScaledL2Clip):
    """Learned scale + L2 clip followed by :class:`Inkling2LogScale` (used in full-attention layers)."""

    def __init__(
        self,
        config: Inkling2TextConfig,
        num_heads: int,
        head_dim: int,
        threshold: float,
        param_dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: spx.Rngs,
    ):
        super().__init__(
            num_heads=num_heads,
            head_dim=head_dim,
            threshold=threshold,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.log_scale = Inkling2LogScale(config)

    def forward(self, hidden_states: Array, position_ids: Array | None = None) -> Array:
        clipped = super().forward(hidden_states, position_ids)
        return self.log_scale(clipped, position_ids)


class Inkling2RelativePositionBias(spx.Module):
    """Hidden-state conditioned relative position bias gathered by backward distance."""

    def __init__(
        self,
        config: Inkling2TextConfig,
        num_heads: int,
        rel_extent: int,
        log_scaling: bool,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        self.num_heads = num_heads
        self.d_rel = config.d_rel
        self.rel_extent = rel_extent
        self.precision = precision
        self.linear = ColumnParallelLinear(
            config.hidden_size,
            num_heads * config.d_rel,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.norm = (
            Inkling2LogScaledL2Clip(
                config, num_heads, config.d_rel, config.rel_l2_clip_threshold, param_dtype=param_dtype, rngs=rngs
            )
            if log_scaling
            else Inkling2ScaledL2Clip(
                num_heads, config.d_rel, config.rel_l2_clip_threshold, param_dtype=param_dtype, rngs=rngs
            )
        )
        self.proj = ArrayParam.bound(
            shape=(config.d_rel, rel_extent),
            dtype=param_dtype,
            init_method="normal",
            init_kwargs={"stddev": config.initializer_range},
            key=rngs.param,
        )

    def forward(
        self,
        hidden_states: Float[Array, "batch q_len hidden_dim"],
        relative_distance: Int[Array, "... q_len kv_len"],
        position_ids: Int[Array, "batch q_len"],
    ) -> Float[Array, "batch num_heads q_len kv_len"]:
        relative_states = self.linear(hidden_states).reshape(*hidden_states.shape[:-1], self.num_heads, self.d_rel)
        relative_states = self.norm(relative_states, position_ids)

        proj = self.proj.value.astype(relative_states.dtype)
        rel_logits = jnp.matmul(relative_states, proj, precision=self.precision)
        rel_logits = jnp.swapaxes(rel_logits, 1, 2)  # [B, H, Q, rel_extent]

        dist = relative_distance
        if dist.ndim == 2:
            dist = dist[None, None, :, :]
        elif dist.ndim == 3:
            dist = dist[:, None, :, :]
        gather_idx = jnp.clip(dist, 0, self.rel_extent - 1)
        gather_idx = jnp.broadcast_to(gather_idx, (rel_logits.shape[0], self.num_heads, dist.shape[-2], dist.shape[-1]))
        position_bias = jnp.take_along_axis(rel_logits, gather_idx, axis=-1)
        in_range = (dist >= 0) & (dist < self.rel_extent)
        return jnp.where(in_range, position_bias, jnp.zeros((), dtype=position_bias.dtype))


# ---------------------------------------------------------------------------
# Attention projections & AttentionModule subclasses
# ---------------------------------------------------------------------------


class Inkling2QueryProjection(spx.Module):
    """Query projection with learned L2 clipping (and log scaling on full layers)."""

    def __init__(
        self,
        config: Inkling2TextConfig,
        num_heads: int,
        head_dim: int,
        log_scaling: bool,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.linear = ColumnParallelLinear(
            config.hidden_size,
            num_heads * head_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.norm = (
            Inkling2LogScaledL2Clip(
                config, num_heads, head_dim, config.qk_l2_clip_threshold, param_dtype=param_dtype, rngs=rngs
            )
            if log_scaling
            else Inkling2ScaledL2Clip(
                num_heads, head_dim, config.qk_l2_clip_threshold, param_dtype=param_dtype, rngs=rngs
            )
        )

    def forward(self, hidden_states: Array, position_ids: Array) -> Float[Array, "batch seq_len num_heads head_dim"]:
        states = self.linear(hidden_states)
        states = states.reshape(*states.shape[:-1], self.num_heads, self.head_dim)
        return self.norm(states, position_ids)


class Inkling2KeyValueProjection(spx.Module):
    """Key or value projection followed by causal short conv and L2-clip (K) / unweighted RMSNorm (V)."""

    def __init__(
        self,
        config: Inkling2TextConfig,
        num_heads: int,
        head_dim: int,
        layer_idx: int,
        conv_idx: int,
        is_key: bool,
        dynamic: bool = False,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.linear = ColumnParallelLinear(
            config.hidden_size,
            num_heads * head_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.sconv = (
            DynamicResidualShortConvolution(
                hidden_size=num_heads * head_dim,
                input_size=config.hidden_size,
                num_heads=num_heads,
                conv_kernel_size=config.conv_kernel_size,
                layer_idx=layer_idx,
                conv_idx=conv_idx,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                initializer_range=config.initializer_range,
                rngs=rngs,
            )
            if dynamic
            else ResidualShortConvolution(
                hidden_size=num_heads * head_dim,
                conv_kernel_size=config.conv_kernel_size,
                layer_idx=layer_idx,
                conv_idx=conv_idx,
                rngs=rngs,
            )
        )
        self.norm = (
            Inkling2L2Clip(config.qk_l2_clip_threshold)
            if is_key
            else Inkling2UnweightedRMSNorm(eps=config.rms_norm_eps, dtype=dtype)
        )

    def forward(
        self,
        hidden_states: Array,
        cache_view: ConvKVCacheView | None = None,
        conv_mask: Array | None = None,
        seq_idx: Array | None = None,
    ) -> tuple[Float[Array, "batch seq_len num_kv_heads head_dim"], ConvKVCacheView | None]:
        states = self.linear(hidden_states)
        states, cache_view = self.sconv(
            states,
            dynamic_input=hidden_states,
            cache_view=cache_view,
            conv_mask=conv_mask,
            seq_idx=seq_idx,
        )
        states = states.reshape(*states.shape[:-1], self.num_heads, self.head_dim)
        return self.norm(states), cache_view


class Inkling2GatedOutputProjection(spx.Module):
    """Sigmoid-gated attention output projection ``linear(attn_out * sigmoid(gate_proj(x)))``."""

    def __init__(
        self,
        config: Inkling2TextConfig,
        num_heads: int,
        head_dim: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            num_heads * head_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.linear = RowParallelLinear(
            num_heads * head_dim,
            config.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )

    def forward(self, attn_output: Array, hidden_states: Array) -> Array:
        gate = jax.nn.sigmoid(self.gate_proj(hidden_states).astype(jnp.float32))
        gated = (attn_output.astype(jnp.float32) * gate).astype(attn_output.dtype)
        return self.linear(gated)


class _Inkling2BaseAttention(AttentionModule):
    """Shared attention mechanics for Inkling 2 sliding and full attention layers.

    Uses :class:`FlexibleAttentionModule` bound to EasyDeL's registered
    ``vanilla`` operation (preserving MPMD stage-submesh context, logical QKV
    sharding, dropout, 4-D per-token ``softmax_aux`` sinks, per-pair relative
    position bias, and sliding-window masking without double-windowing cached
    steps).
    """

    def __init__(
        self,
        config: Inkling2TextConfig,
        layer_idx: int,
        *,
        is_full: bool,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        rngs: spx.Rngs,
    ):
        super().__init__(config=config)
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.is_sliding = config.is_sliding_layer(layer_idx)
        self.head_dim = config.swa_head_dim if self.is_sliding else config.head_dim
        self.num_heads = config.swa_num_attention_heads if self.is_sliding else config.num_attention_heads
        self.num_key_value_heads = config.swa_num_key_value_heads if self.is_sliding else config.num_key_value_heads
        self.sliding_window = config.sliding_window_size if self.is_sliding else None
        self.rel_extent = config.sliding_window_size if self.is_sliding else config.rel_extent
        self.scaling = float(self.head_dim**-0.5)

        self.q_proj = Inkling2QueryProjection(
            config,
            self.num_heads,
            self.head_dim,
            log_scaling=is_full,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.k_proj = Inkling2KeyValueProjection(
            config,
            self.num_key_value_heads,
            self.head_dim,
            layer_idx=layer_idx,
            conv_idx=0,
            is_key=True,
            dynamic=is_full,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.v_proj = Inkling2KeyValueProjection(
            config,
            self.num_key_value_heads,
            self.head_dim,
            layer_idx=layer_idx,
            conv_idx=1,
            is_key=False,
            dynamic=is_full,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.o_proj = Inkling2GatedOutputProjection(
            config,
            self.num_heads,
            self.head_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.sink_proj = ColumnParallelLinear(
            config.hidden_size,
            self.num_heads,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.relative_position_bias = Inkling2RelativePositionBias(
            config,
            self.num_heads,
            self.rel_extent,
            log_scaling=is_full,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        # 4-D per-token softmax_aux sinks + arbitrary 4-D relative bias require
        # EasyDeL's registered vanilla kernel; clear impl_decode so global
        # decode_attn_mechanism overrides cannot bypass 4-D sink support.
        self.attention_performer = FlexibleAttentionModule(
            base_config=config,
            softmax_scale=self.scaling,
            dropout_prob=config.attention_dropout,
            attn_mechanism=AttentionMechanisms.VANILLA,
            requires_cache=False,
            rngs=rngs,
        )
        self.attention_performer.impl_decode = None

    def _run_attention(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        key_hidden_states: Float[Array, "batch seq_len hidden_dim"],
        value_hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,
        cache_view: ConvKVCacheView | None = None,
        cache_metadata: TransformerMetadata | OperationsMetadata | None = None,
        conv_mask: Bool[Array, "batch seq_len"] | None = None,
        seq_idx: Int[Array, "batch seq_len"] | None = None,
        output_attentions: bool = False,
    ) -> AttentionLayerOutput:
        batch_size, q_len = hidden_states.shape[:2]
        query_states = self.q_proj(hidden_states, position_ids)
        key_states, cache_view = self.k_proj(
            key_hidden_states, cache_view=cache_view, conv_mask=conv_mask, seq_idx=seq_idx
        )
        value_states, cache_view = self.v_proj(
            value_hidden_states, cache_view=cache_view, conv_mask=conv_mask, seq_idx=seq_idx
        )
        query_states, key_states, value_states = self.apply_qkv_shardings(query_states, key_states, value_states)

        if cache_view is not None:
            key_states, value_states, aligned_mask, cache_view, _ = cache_view.concatenate_to_cache(
                query=query_states,
                key=key_states,
                value=value_states,
                mode=mode,
                quantizer=self.quantizer,
                cache_metadata=cache_metadata,
                mask_info=mask_info,
                runtime_sharding_resolver=self.config.runtime_sharding_resolver,
                causal=True,
            )
            q_pos = aligned_mask.q_positions
            kv_pos = aligned_mask.kv_positions
            if q_pos is None:
                q_pos = jnp.broadcast_to(jnp.arange(q_len, dtype=jnp.int32)[None, :], (batch_size, q_len))
            if kv_pos is None:
                kv_len = key_states.shape[1]
                kv_pos = jnp.broadcast_to(jnp.arange(kv_len, dtype=jnp.int32)[None, :], (batch_size, kv_len))
            relative_distance = q_pos[:, :, None] - kv_pos[:, None, :]
        else:
            aligned_mask = mask_info
            q_idx = jnp.arange(q_len, dtype=jnp.int32)
            relative_distance = q_idx[:, None] - q_idx[None, :]

        position_bias = self.relative_position_bias(hidden_states, relative_distance, position_ids)
        sinks = jnp.swapaxes(self.sink_proj(hidden_states), 1, 2)[..., None]  # [B, Hq, Q, 1]

        attn_res = self.attention_performer.forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            mode=mode,
            mask_info=aligned_mask,
            bias=position_bias,
            sliding_window=self.sliding_window,
            cache_view=cache_view.kv if cache_view is not None else None,
            cache_metadata=cache_metadata,
            causal=True,
            softmax_aux=sinks,
            softmax_scale=self.scaling,
            output_attentions=output_attentions,
            precision=self.precision,
            compute_dtype=self.dtype,
        )

        attn_output = attn_res.attention_outputs.reshape(batch_size, q_len, self.num_heads * self.head_dim)
        attn_output = checkpoint_name(self.o_proj(attn_output, hidden_states), "attn_output")
        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attn_res.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class Inkling2SlidingAttention(_Inkling2BaseAttention):
    """Inkling 2 sliding-window attention (static short conv on K/V, no log scaling)."""

    def __init__(
        self,
        config: Inkling2TextConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        super().__init__(
            config=config,
            layer_idx=layer_idx,
            is_full=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,
        cache_view: ConvKVCacheView | None = None,
        cache_metadata: TransformerMetadata | OperationsMetadata | None = None,
        key_hidden_states: Float[Array, "batch seq_len hidden_dim"] | None = None,
        value_hidden_states: Float[Array, "batch seq_len hidden_dim"] | None = None,
        conv_mask: Bool[Array, "batch seq_len"] | None = None,
        seq_idx: Int[Array, "batch seq_len"] | None = None,
        output_attentions: bool = False,
    ) -> AttentionLayerOutput:
        return self._run_attention(
            hidden_states=hidden_states,
            key_hidden_states=hidden_states if key_hidden_states is None else key_hidden_states,
            value_hidden_states=hidden_states if value_hidden_states is None else value_hidden_states,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            conv_mask=conv_mask,
            seq_idx=seq_idx,
            output_attentions=output_attentions,
        )


class Inkling2FullAttention(_Inkling2BaseAttention):
    """Inkling 2 full attention (dynamic short conv on K/V, log scaling on Q and relative bias)."""

    def __init__(
        self,
        config: Inkling2TextConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        super().__init__(
            config=config,
            layer_idx=layer_idx,
            is_full=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        key_hidden_states: Float[Array, "batch seq_len hidden_dim"],
        value_hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,
        cache_view: ConvKVCacheView | None = None,
        cache_metadata: TransformerMetadata | OperationsMetadata | None = None,
        conv_mask: Bool[Array, "batch seq_len"] | None = None,
        seq_idx: Int[Array, "batch seq_len"] | None = None,
        output_attentions: bool = False,
    ) -> AttentionLayerOutput:
        return self._run_attention(
            hidden_states=hidden_states,
            key_hidden_states=key_hidden_states,
            value_hidden_states=value_hidden_states,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            conv_mask=conv_mask,
            seq_idx=seq_idx,
            output_attentions=output_attentions,
        )


# ---------------------------------------------------------------------------
# Softcapped SwiGLU MoE block
# ---------------------------------------------------------------------------


class Inkling2SoftcappedSwiGLU(spx.Module):
    """Module wrapper around :func:`~easydel.layers.mlp.softcapped_swiglu`."""

    def __init__(self, config: Inkling2TextConfig):
        self.gate_softcap = float(config.mlp_gate_softcap)
        self.up_softcap = float(config.mlp_up_softcap)

    def forward(self, gate: Array, up: Array) -> Array:
        return softcapped_swiglu(gate, up, gate_cap=self.gate_softcap, up_cap=self.up_softcap)


class Inkling2Experts(spx.Module):
    """Stacked routed-expert softcapped SwiGLU MLPs supporting HF and vendor checkpoint formats.

    Accepts canonical Hugging Face checkpoints (``gate_up_proj`` / ``down_proj``)
    as well as Thinking Machines' raw checkpoints (interleaved BF16, MXFP8, and
    NVFP4 ``w13_weight*`` / ``w2_weight*`` tensors).
    """

    def __init__(
        self,
        config: Inkling2TextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.gate_softcap = float(config.mlp_gate_softcap)
        self.up_softcap = float(config.mlp_up_softcap)
        self.gate_up_proj = ColumnParallelMoELinear(
            num_experts=config.n_routed_experts,
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
            num_experts=config.n_routed_experts,
            in_features=config.moe_intermediate_size,
            out_features=config.hidden_size,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            use_bias=False,
            partition_manager=config.runtime_sharding_resolver,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.act_fn = Inkling2SoftcappedSwiGLU(config)

    @property
    def reform_param(self) -> dict[str, tp.Any]:
        """Checkpoint conversion rules for canonical HF and vendor BF16/MXFP8/NVFP4 experts."""
        cfg = self.config

        def _canonical_gate_up_in(x):
            return tp_pack_expert_gate_up(x, config=cfg)

        def _canonical_gate_up_out(x):
            return tp_unpack_expert_gate_up(x, config=cfg)

        def _raw_bf16_w13_in(x):
            return tp_pack_expert_gate_up(deinterleave_inkling2_gate_up(x), config=cfg)

        def _mxfp8_w13_in(weight, scale):
            return tp_pack_expert_gate_up(dequantize_inkling2_experts(weight, scale, deinterleave=True), config=cfg)

        def _nvfp4_w13_in(weight, scale, amax, denom):
            deq = dequantize_inkling2_experts(weight, scale, amax, denom, deinterleave=True)
            return tp_pack_expert_gate_up(deq, config=cfg)

        def _mxfp8_w2_in(weight, scale):
            return transpose_expert_3d(dequantize_inkling2_experts(weight, scale, deinterleave=False))

        def _nvfp4_w2_in(weight, scale, amax, denom):
            return transpose_expert_3d(dequantize_inkling2_experts(weight, scale, amax, denom, deinterleave=False))

        def _unimplemented_quant_export(tensor):
            del tensor
            raise NotImplementedError("Exporting EasyDeL experts to raw quantized vendor layouts is not supported.")

        return {
            "gate_up_proj$": {
                "splits": [{"name": "gate_up_proj.weight", "spliter": _canonical_gate_up_in}],
                "inverse_spliter": _canonical_gate_up_out,
            },
            "down_proj$": {
                "splits": [{"name": "down_proj.weight", "spliter": transpose_expert_3d}],
                "inverse_spliter": transpose_expert_3d,
            },
            "w13_weight$": {
                "splits": [{"name": "gate_up_proj.weight", "spliter": _raw_bf16_w13_in}],
                "inverse_spliter": lambda x: interleave_inkling2_gate_up(_canonical_gate_up_out(x)),
            },
            "w2_weight$": {
                "splits": [{"name": "down_proj.weight", "spliter": transpose_expert_3d}],
                "inverse_spliter": transpose_expert_3d,
            },
            "gate_up_proj.weight$": {
                "sources": ("w13_weight", "w13_weight_scale"),
                "fuser": _mxfp8_w13_in,
                "inverse_fuser": _unimplemented_quant_export,
                "already_converted": True,
                "log_label": "Inkling2 MXFP8 gate/up experts",
            },
            "gate_up_proj.weight": {
                "sources": (
                    "w13_weight",
                    "w13_weight_scale",
                    "w13_weight_amax",
                    "w13_weight_outer_scale_denominator",
                ),
                "fuser": _nvfp4_w13_in,
                "inverse_fuser": _unimplemented_quant_export,
                "already_converted": True,
                "log_label": "Inkling2 NVFP4 gate/up experts",
            },
            "down_proj.weight$": {
                "sources": ("w2_weight", "w2_weight_scale"),
                "fuser": _mxfp8_w2_in,
                "inverse_fuser": _unimplemented_quant_export,
                "already_converted": True,
                "log_label": "Inkling2 MXFP8 down experts",
            },
            "down_proj.weight": {
                "sources": (
                    "w2_weight",
                    "w2_weight_scale",
                    "w2_weight_amax",
                    "w2_weight_outer_scale_denominator",
                ),
                "fuser": _nvfp4_w2_in,
                "inverse_fuser": _unimplemented_quant_export,
                "already_converted": True,
                "log_label": "Inkling2 NVFP4 down experts",
            },
        }

    def forward(
        self,
        hidden_states: Float[Array, "tokens hidden_dim"],
        group_sizes: Array,
        sorted_experts: Array | None = None,
    ) -> Array:
        gate_up = self.gate_up_proj(hidden_states, group_sizes, sorted_experts)
        gate, up = split_fused_gate_up_projection(gate_up, config=self.config)
        intermediate = softcapped_swiglu(gate, up, gate_cap=self.gate_softcap, up_cap=self.up_softcap)
        return self.down_proj(intermediate, group_sizes, sorted_experts)


class Inkling2TopKRouter(spx.Module):
    """Float32 MoE router with persistent fp32 ``e_score_correction_bias``."""

    def __init__(
        self,
        config: Inkling2TextConfig,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        self.config = config
        self.precision = precision
        self.num_experts = config.n_routed_experts
        self.weight = ArrayParam.bound(
            shape=(config.hidden_size, self.num_experts),
            dtype=jnp.float32,
            init_method="normal",
            init_kwargs={"stddev": config.initializer_range},
            key=rngs.param,
        )
        self.e_score_correction_bias = ArrayParam.bound(
            shape=(self.num_experts,),
            dtype=jnp.float32,
            init_method="zeros",
            key=rngs.param,
        )

    def forward(self, hidden_states: Float[Array, "... hidden_dim"]) -> Float[Array, "tokens experts"]:
        flat = hidden_states.reshape(-1, self.config.hidden_size).astype(jnp.float32)
        weight = self.weight.value.astype(jnp.float32)
        return checkpoint_name(jnp.matmul(flat, weight), "moe_router_logits")


class Inkling2SparseMoeBlock(BaseMoeModule):
    """Inkling 2 sparse MoE block (bias-steered top-k routing, softcapped SwiGLU experts)."""

    def __init__(
        self,
        config: Inkling2TextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        super().__init__(
            config=config,
            n_routed_experts=config.n_routed_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            lbl_coef=None,
            rzl_coef=None,
            routing_strategy=MoeRoutingStrategy.TOP_K,
            load_balancing_strategy=MoeLoadBalancingStrategy.STANDARD,
        )
        self.config = config
        self.route_scale = float(config.route_scale)
        self.scoring_func = config.scoring_func

        self.gate = Inkling2TopKRouter(config=config, precision=precision, rngs=rngs)
        self.experts = Inkling2Experts(
            config=config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs
        )
        self.moe_hooks = MoeFusedHooks(
            normalize_gate_logits=lambda x: x,
            select_hook=partial(
                moe_group_topk_select,
                n_routed_experts=config.n_routed_experts,
                score_fn=config.scoring_func,
                e_score_correction_bias=None,
                n_group=1,
                norm_topk_prob=True,
                routed_scaling_factor=config.route_scale,
                norm_eps=1e-20,
            ),
        )

    def forward(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> tuple[Array, Array]:
        hooks = self.moe_hooks.replace(
            select_hook=partial(
                moe_group_topk_select,
                n_routed_experts=self.config.n_routed_experts,
                score_fn=self.scoring_func,
                e_score_correction_bias=self.gate.e_score_correction_bias.value,
                n_group=1,
                norm_topk_prob=True,
                routed_scaling_factor=self.route_scale,
                norm_eps=1e-20,
            ),
        )
        gate_cap = self.experts.gate_softcap
        up_cap = self.experts.up_softcap

        def ffn_activation(w0: Array, w1: Array) -> Array:
            return softcapped_swiglu(w0, w1, gate_cap=gate_cap, up_cap=up_cap)

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
        return checkpoint_name(out, "moe_expert_output"), checkpoint_name(router_logits, "moe_router_logits")


# ---------------------------------------------------------------------------
# Decoder layers (single shared base + thin sliding / full / looped variants)
# ---------------------------------------------------------------------------


class _Inkling2BaseDecoderLayer(spx.Module):
    """Shared decoder-layer forward body for Inkling 2 sliding, full, and looped blocks."""

    def _init_common_sublayers(
        self,
        config: Inkling2TextConfig,
        layer_idx: int,
        *,
        num_attn_queries: int,
        self_attn: _Inkling2BaseAttention,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        precision: jax.lax.PrecisionLike,
        rngs: spx.Rngs,
    ) -> None:
        self.config = config
        self.layer_idx = layer_idx
        self.is_block_boundary = (layer_idx % config.n_layers_per_attn_res_block) == 0
        self.attention_input = (
            BlockAttentionResidual(config.hidden_size, num_queries=num_attn_queries, rngs=rngs)
            if layer_idx > 0
            else None
        )
        self.mlp_input = BlockAttentionResidual(config.hidden_size, num_queries=1, rngs=rngs)
        self.self_attn = self_attn
        self.mlp = Inkling2SparseMoeBlock(config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)
        self.post_attention_layernorm = Inkling2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.post_attention_convolution = ResidualShortConvolution(
            config.hidden_size, config.conv_kernel_size, layer_idx=layer_idx, conv_idx=2, rngs=rngs
        )
        self.post_mlp_convolution = ResidualShortConvolution(
            config.hidden_size, config.conv_kernel_size, layer_idx=layer_idx, conv_idx=3, rngs=rngs
        )

    def _prepare_attn_inputs(
        self,
        hidden_states: Array,
        block_states: Array,
        messages: Array | None,
    ) -> tuple[Array, Array, Array, Array | None]:
        raise NotImplementedError

    def _apply_mlp_message(self, hidden_states: Array, mlp_message: Array | None) -> Array:
        del mlp_message
        return hidden_states

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        block_states: Float[Array, "batch seq_len num_blocks hidden_dim"],
        mask_info: MaskInfo,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,
        cache_view: ConvKVCacheView | None = None,
        cache_metadata: TransformerMetadata | OperationsMetadata | None = None,
        conv_mask: Bool[Array, "batch seq_len"] | None = None,
        seq_idx: Int[Array, "batch seq_len"] | None = None,
        messages: Float[Array, "batch seq_len 4 hidden_dim"] | None = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
    ) -> DecoderLayerOutput:
        dtype = hidden_states.dtype
        block_residual = hidden_states

        q_in, k_in, v_in, mlp_message = self._prepare_attn_inputs(hidden_states, block_states, messages)
        attn_out = self.self_attn(
            hidden_states=q_in.astype(dtype),
            key_hidden_states=k_in.astype(dtype),
            value_hidden_states=v_in.astype(dtype),
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            conv_mask=conv_mask,
            seq_idx=seq_idx,
            output_attentions=output_attentions,
        )
        cache_view = attn_out.cache_view
        attn_hidden, cache_view = self.post_attention_convolution(
            attn_out.attention_output, cache_view=cache_view, conv_mask=conv_mask, seq_idx=seq_idx
        )

        residual = hidden_states = attn_hidden.astype(jnp.float32)
        if self.is_block_boundary:
            block_states = jnp.concatenate([block_states, block_residual[..., None, :]], axis=-2)
        else:
            residual = hidden_states = block_residual.astype(jnp.float32) + hidden_states

        hidden_states = self.mlp_input(block_states, hidden_states)[0]
        hidden_states = self._apply_mlp_message(hidden_states, mlp_message)
        mlp_out, router_logits = self.mlp(self.post_attention_layernorm(hidden_states).astype(dtype))
        mlp_out, cache_view = self.post_mlp_convolution(
            mlp_out, cache_view=cache_view, conv_mask=conv_mask, seq_idx=seq_idx
        )
        final_hidden = checkpoint_name((residual + mlp_out.astype(jnp.float32)).astype(dtype), "layer_output")
        return DecoderLayerOutput(
            hidden_states=final_hidden,
            attention_weight=attn_out.attention_weight if output_attentions else None,
            router_logits=router_logits if output_router_logits else None,
            cache_view=cache_view,
        )


class Inkling2SlidingDecoderLayer(_Inkling2BaseDecoderLayer):
    """Inkling 2 sliding-window decoder layer with block attention residuals and short convs."""

    attention_type = "sliding_attention"

    def __init__(
        self,
        config: Inkling2TextConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        self._init_common_sublayers(
            config,
            layer_idx,
            num_attn_queries=1,
            self_attn=Inkling2SlidingAttention(
                config, layer_idx=layer_idx, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs
            ),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.input_layernorm = Inkling2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

    def _prepare_attn_inputs(
        self,
        hidden_states: Array,
        block_states: Array,
        messages: Array | None,
    ) -> tuple[Array, Array, Array, Array | None]:
        del messages
        if self.attention_input is not None:
            hidden_states = self.attention_input(block_states, hidden_states)[0]
        normed = self.input_layernorm(hidden_states)
        return normed, normed, normed, None


class Inkling2FullDecoderLayer(_Inkling2BaseDecoderLayer):
    """Inkling 2 full-attention decoder layer with separate Q/K/V block-residual queries and input norms."""

    attention_type = "full_attention"

    def __init__(
        self,
        config: Inkling2TextConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        self._init_common_sublayers(
            config,
            layer_idx,
            num_attn_queries=3,
            self_attn=Inkling2FullAttention(
                config, layer_idx=layer_idx, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs
            ),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.q_input_layernorm = Inkling2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.k_input_layernorm = Inkling2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.v_input_layernorm = Inkling2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

    def _prepare_attn_inputs(
        self,
        hidden_states: Array,
        block_states: Array,
        messages: Array | None,
    ) -> tuple[Array, Array, Array, Array | None]:
        del messages
        if self.attention_input is None:
            q_h = k_h = v_h = hidden_states
        else:
            q_h, k_h, v_h = self.attention_input(block_states, hidden_states)
        return self.q_input_layernorm(q_h), self.k_input_layernorm(k_h), self.v_input_layernorm(v_h), None


class _Inkling2LoopMessageGatesMixin:
    """Shared four-gate cross-pass controller setup for looped decoder layers."""

    def _init_message_gates(
        self,
        config: Inkling2LoopTextConfig,
        *,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        precision: jax.lax.PrecisionLike,
        rngs: spx.Rngs,
    ) -> None:
        gate_kwargs = dict(
            hidden_size=config.hidden_size,
            gate_rank=config.loop_gate_rank,
            eps=config.rms_norm_eps,
            initializer_range=config.initializer_range,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.q_message_gate = LoopMessageGate(**gate_kwargs)
        self.k_message_gate = LoopMessageGate(**gate_kwargs)
        self.v_message_gate = LoopMessageGate(**gate_kwargs)
        self.mlp_message_gate = LoopMessageGate(**gate_kwargs)

    def _apply_mlp_message(self, hidden_states: Array, mlp_message: Array | None) -> Array:
        if mlp_message is None:
            return hidden_states
        return self.mlp_message_gate(hidden_states, mlp_message)


class Inkling2LoopSlidingLoopDecoderLayer(_Inkling2LoopMessageGatesMixin, Inkling2SlidingDecoderLayer):
    """Looped sliding-window decoder layer with four cross-pass :class:`LoopMessageGate` controllers."""

    def __init__(
        self,
        config: Inkling2LoopTextConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        super().__init__(
            config=config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self._init_message_gates(config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)

    def _prepare_attn_inputs(
        self,
        hidden_states: Array,
        block_states: Array,
        messages: Array | None,
    ) -> tuple[Array, Array, Array, Array | None]:
        hidden_states = self.attention_input(block_states, hidden_states)[0]  # type: ignore[misc]
        if messages is None:
            normed = self.input_layernorm(hidden_states)
            return normed, normed, normed, None
        q_in = self.input_layernorm(self.q_message_gate(hidden_states, messages[:, :, 0, :]))
        k_in = self.input_layernorm(self.k_message_gate(hidden_states, messages[:, :, 1, :]))
        v_in = self.input_layernorm(self.v_message_gate(hidden_states, messages[:, :, 2, :]))
        return q_in, k_in, v_in, messages[:, :, 3, :]


class Inkling2LoopFullLoopDecoderLayer(_Inkling2LoopMessageGatesMixin, Inkling2FullDecoderLayer):
    """Looped full-attention decoder layer with four cross-pass :class:`LoopMessageGate` controllers."""

    def __init__(
        self,
        config: Inkling2LoopTextConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        super().__init__(
            config=config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self._init_message_gates(config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)

    def _prepare_attn_inputs(
        self,
        hidden_states: Array,
        block_states: Array,
        messages: Array | None,
    ) -> tuple[Array, Array, Array, Array | None]:
        q_h, k_h, v_h = self.attention_input(block_states, hidden_states)  # type: ignore[misc]
        mlp_msg = None
        if messages is not None:
            q_h = self.q_message_gate(q_h, messages[:, :, 0, :])
            k_h = self.k_message_gate(k_h, messages[:, :, 1, :])
            v_h = self.v_message_gate(v_h, messages[:, :, 2, :])
            mlp_msg = messages[:, :, 3, :]
        return self.q_input_layernorm(q_h), self.k_input_layernorm(k_h), self.v_input_layernorm(v_h), mlp_msg


# ---------------------------------------------------------------------------
# Normed token embeddings & Text models (full MPMD pipeline parallelism)
# ---------------------------------------------------------------------------


class Inkling2NormedEmbedding(Embed):
    """Token embedding table followed by :class:`Inkling2RMSNorm`."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: spx.Rngs,
    ):
        super().__init__(
            num_embeddings=num_embeddings,
            features=embedding_dim,
            embedding_init=jax.nn.initializers.normal(stddev=initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.embed_norm = Inkling2RMSNorm(embedding_dim, eps=norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs)

    def forward(self, inputs: Array) -> Array:
        return self.embed_norm(super().forward(inputs))


def _resolve_conv_mask_and_seq_idx(
    mask_info: MaskInfo,
    attention_mask: Array | None,
    batch_size: int,
    seq_len: int,
    has_cache: bool,
) -> tuple[Array | None, Array | None]:
    """Derive per-token convolution padding mask and optional packed segment ids."""
    conv_mask: Array | None = None
    seq_idx: Array | None = None
    q_seg = getattr(mask_info, "_q_segment_ids", None)
    if q_seg is not None and q_seg.shape == (batch_size, seq_len):
        conv_mask = q_seg >= 0
        if not has_cache:
            seq_idx = q_seg.astype(jnp.int32)
    elif attention_mask is not None and attention_mask.ndim == 2:
        if attention_mask.shape[1] == seq_len:
            conv_mask = attention_mask.astype(jnp.bool_)
        elif attention_mask.shape[1] > seq_len:
            conv_mask = attention_mask[:, -seq_len:].astype(jnp.bool_)
    return conv_mask, seq_idx


@auto_pytree
class Inkling2ModelOutput(MoeModelOutput):
    """Multimodal backbone output extending :class:`MoeModelOutput` with image features."""

    image_hidden_states: Array | None = None


@register_module(TaskType.BASE_MODULE, config=Inkling2TextConfig, model_type="inkling2_text")
class Inkling2TextModel(EasyDeLBaseModule):
    """Inkling 2 flat MoE text decoder backbone with full SPMD and MPMD pipeline support."""

    fp32_param_patterns: tp.ClassVar[tuple[str, ...]] = FP32_PARAM_PATTERNS
    checkpoint_key_rules: tp.ClassVar[Inkling2KeyNormalizer] = INKLING2_TEXT_KEY_RULES

    def __init__(
        self,
        config: Inkling2TextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        with self.assign_layer_stage(0, total_layers=config.num_hidden_layers):
            self.embed_tokens = Inkling2NormedEmbedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                norm_eps=config.rms_norm_eps,
                initializer_range=config.initializer_range,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

        self.layers = self._build_layers(config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)

        final_layer_idx = max(0, config.num_hidden_layers - 1)
        with self.assign_layer_stage(final_layer_idx, total_layers=config.num_hidden_layers):
            self.norm = Inkling2RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs
            )
            self.final_input = BlockAttentionResidual(config.hidden_size, num_queries=1, rngs=rngs)

    def _build_layers(
        self,
        config: Inkling2TextConfig,
        *,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        precision: jax.lax.PrecisionLike,
        rngs: spx.Rngs,
    ) -> nn.ModuleList:
        remat_sliding = auto_remat(
            Inkling2SlidingDecoderLayer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        remat_full = auto_remat(
            Inkling2FullDecoderLayer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        layers = nn.ModuleList([])
        for layer_idx in range(config.num_hidden_layers):
            layer_cls = remat_sliding if config.is_sliding_layer(layer_idx) else remat_full
            with self.assign_layer_stage(layer_idx, total_layers=config.num_hidden_layers):
                layers.append(
                    layer_cls(
                        config=config,
                        layer_idx=layer_idx,
                        dtype=dtype,
                        param_dtype=param_dtype,
                        precision=precision,
                        rngs=rngs,
                    )
                )
        return layers

    def _stage_boundary_edge_sharding(self, hidden_states: Array) -> PartitionSpec | None:
        """Resolve a 2-axis ``(batch, seq)`` edge sharding compatible with both 3-D and 4-D stage outputs.

        Inkling 2 carries both ``hidden_states`` ``[B, S, H]`` and
        ``block_states`` ``[B, S, num_blocks, H]`` across pipeline stages.
        SpectraX applies the stage marker's ``sharding`` to every stage output
        with ``strict=True``, so restricting the edge spec to the shared
        leading ``(batch, seq)`` axes preserves DP/SP sharding while avoiding
        attempting to shard ``num_blocks`` along ``tp``.
        """
        spec = self._layer_stage_boundary_sharding(hidden_states)
        if spec is None:
            return None
        entries = tuple(spec)
        if len(entries) >= 2:
            return PartitionSpec(entries[0], entries[1])
        return spec

    def _mark_residual_stage_boundary(
        self,
        hidden_states: Array,
        block_states: Array,
        layer_idx: int,
        *,
        next_layer_idx: int | None = None,
    ) -> tuple[Array, Array]:
        """Emit an ``sxstage_iter`` boundary over ``(hidden_states, block_states)`` when crossing PP stages."""
        total = len(self.layers)
        pp = self._pipeline_stage_count()
        nxt_idx = layer_idx + 1 if next_layer_idx is None else next_layer_idx
        if pp <= 1 or nxt_idx >= total:
            return hidden_states, block_states
        current = self._pipeline_logical_stage(layer_idx, total)
        nxt = self._pipeline_logical_stage(nxt_idx, total)
        if current != nxt:
            edge_sharding = self._stage_boundary_edge_sharding(hidden_states)
            hidden_states, block_states = spx.sxstage_iter(
                (hidden_states, block_states),
                stage=current,
                sharding=edge_sharding,
            )
        return hidden_states, block_states

    def create_cache_config(
        self,
        batch_size: int,
        max_length: int,
        pad_token_id: int | None = None,
    ) -> ConvKVCacheConfig:
        """Return the :class:`ConvKVCacheConfig` for this decoder."""
        return create_inkling2_cache_config(
            config=self.config,
            batch_size=batch_size,
            sequence_length=max_length,
            pad_token_id=pad_token_id,
        )

    def _cache_slot_physical_layer(self, slot_idx: int) -> int:
        """Return the physical decoder layer index that owns logical cache slot ``slot_idx``."""
        return int(slot_idx)

    def init_cache(
        self,
        batch_size: int,
        max_length: int,
        starts: Array | None = None,
        shardings: dict | None = None,
        pad_token_id: int | None = None,
    ) -> ConvKVCache:
        """Allocate a stage-aware :class:`ConvKVCache` (KV + 4 fp32 short-conv states per slot)."""
        from easydel.infra.mixins.generation import _apply_cache_shardings

        cache_config = self.create_cache_config(
            batch_size=batch_size,
            max_length=max_length,
            pad_token_id=pad_token_id,
        )
        kv_dtype = getattr(self.config, "kvdtype", None) or self.dtype
        cache = ConvKVCache.init_cache(
            mesh=self.config.mesh,
            metadata=cache_config,
            partition_manager=self.config.runtime_sharding_resolver,
            dtype=kv_dtype,
            starts=starts,
            quantizer=self.config._get_kv_cache_quantizer() if hasattr(self.config, "_get_kv_cache_quantizer") else None,
            layer_context_fn=lambda slot: self._layer_stage_context(
                self._cache_slot_physical_layer(slot), layers=self.layers
            ),
        )
        return _apply_cache_shardings(cache, shardings, self.config.mesh)

    def _prepare_forward_inputs(
        self,
        input_ids: Int[Array, "batch seq_len"] | None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None,
        attention_mask: Bool[Array, "batch seq_len"] | None,
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"] | None,
        mode: common_types.RUNTIME_MODE_TYPES | None,
        past_key_values: ConvKVCache | None,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of `input_ids` or `inputs_embeds`.")
        if inputs_embeds is None:
            with self._layer_stage_context(0, layers=self.layers):
                inputs_embeds = checkpoint_name(self.embed_tokens(input_ids.astype("i4")), "embeddings")

        batch_size, seq_len = inputs_embeds.shape[:2]
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds max_position_embeddings ({self.config.max_position_embeddings})."
            )

        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        if past_key_values is None:
            past_key_values = ConvKVCache.init_empty(self.config.total_num_hidden_layers)

        has_cache = any(v is not None for v in past_key_values.views)
        if mode is None:
            mode = (
                common_types.MODE_DECODE
                if (seq_len == 1 and has_cache)
                else (common_types.MODE_PREFILL if has_cache else common_types.MODE_TRAIN)
            )

        if position_ids is None:
            if has_cache and past_key_values.views[0] is not None:
                kv0 = past_key_values.views[0].kv
                past_offset = kv0.indexes - kv0.starts
                position_ids = past_offset[:, None] + jnp.arange(seq_len, dtype=jnp.int32)[None, :]
            else:
                position_ids = mask_info.q_position_ids

        conv_mask, seq_idx = _resolve_conv_mask_and_seq_idx(mask_info, attention_mask, batch_size, seq_len, has_cache)
        return inputs_embeds, mask_info, position_ids, mode, past_key_values, conv_mask, seq_idx

    def _finalize_hidden_states(self, block_states: Array, hidden_states: Array, output_dtype: jnp.dtype) -> Array:
        """Apply terminal block attention residual and final RMSNorm inside the terminal PP stage."""
        final_layer_idx = max(0, len(self.layers) - 1)
        with self._layer_stage_context(final_layer_idx, layers=self.layers):
            hidden_states = self.final_input(block_states, hidden_states)[0]
            hidden_states = checkpoint_name(self.norm(hidden_states).astype(output_dtype), "model_output")
        return hidden_states

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
        mode: common_types.RUNTIME_MODE_TYPES | None = None,
        past_key_values: ConvKVCache | None = None,
        cache_metadata: TransformerMetadata | OperationsMetadata | None = None,
    ) -> MoeModelOutput:
        inputs_embeds, mask_info, position_ids, mode, past_key_values, conv_mask, seq_idx = self._prepare_forward_inputs(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
        )
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        output_router_logits = self.config.output_router_logits if output_router_logits is None else output_router_logits

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        with self._layer_stage_context(0, layers=self.layers):
            hidden_states = apply_logical_sharding(
                inputs_embeds,
                dynamic_axes=common_types.HiddenStateSharding,
                partition_manager=self.config.runtime_sharding_resolver,
            )
        block_states = jnp.zeros((*inputs_embeds.shape[:-1], 0, inputs_embeds.shape[-1]), dtype=inputs_embeds.dtype)
        views = list(past_key_values.views)

        def _layer_loop(decoder_layer, carry):
            hs, bs, ah, aa, ar, idx = carry
            if output_hidden_states:
                ah = (*ah, hs)
            with self._layer_stage_context(idx, layers=self.layers):
                layer_out = decoder_layer(
                    hidden_states=hs,
                    block_states=bs,
                    mask_info=mask_info,
                    position_ids=position_ids,
                    mode=mode,
                    cache_view=views[idx],
                    cache_metadata=cache_metadata,
                    conv_mask=conv_mask,
                    seq_idx=seq_idx,
                    output_attentions=bool(output_attentions),
                    output_router_logits=bool(output_router_logits),
                )
            views[idx] = layer_out.cache_view
            if decoder_layer.is_block_boundary:
                bs = jnp.concatenate([bs, hs[..., None, :]], axis=-2)
            hs = layer_out.hidden_states
            hs, bs = self._mark_residual_stage_boundary(hs, bs, idx)
            if output_attentions:
                aa = (*aa, layer_out.attention_weight)
            if output_router_logits:
                ar = (*ar, layer_out.router_logits)
            return hs, bs, ah, aa, ar, idx + 1

        # Layers are heterogeneous (sliding vs full, growing block_states width),
        # so trace=True runs the traced Python scan while honoring layer-scan kwargs.
        hidden_states, block_states, all_hidden_states, all_attentions, all_router_logits, _ = self.layers.scan(
            _layer_loop,
            (hidden_states, block_states, all_hidden_states, all_attentions, all_router_logits, 0),
            trace=True,
            **self._layer_scan_kwargs("text"),
        )

        hidden_states = self._finalize_hidden_states(block_states, hidden_states, inputs_embeds.dtype)
        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        return MoeModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=ConvKVCache(views=views),
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            router_logits=all_router_logits,
        )

    def get_encoder(self):
        raise NotImplementedError("Inkling2TextModel is a decoder-only backbone and has no encoder.")

    def get_decoder(self):
        return self

    def get_lm_head(self):
        raise NotImplementedError("Base Inkling2TextModel has no language model head.")

    def get_embedding(self):
        return self.embed_tokens


def _inkling2_loop_unit_count(text_config: Inkling2LoopTextConfig) -> int:
    """Return the acyclic MPMD pipeline unit count: pre-loop + 1 looped core + post-loop."""
    return (
        int(text_config.loop_layer_index_start)
        + 1
        + int(text_config.num_hidden_layers - text_config.loop_layer_index_end)
    )


def _inkling2_loop_pipeline_layer_position(
    module: EasyDeLBaseModule,
    text_config: Inkling2LoopTextConfig,
    layer_idx: int,
) -> tuple[int, int]:
    """Map a physical layer index onto Inkling 2 Loop's acyclic MPMD pipeline units.

    Pre-loop layers occupy ``0 .. loop_start - 1``, the entire looped core
    ``[loop_start, loop_end)`` collapses onto unit ``loop_start``, and post-loop
    layers occupy ``loop_start + 1 .. total_units - 1``.
    """
    loop_start = text_config.loop_layer_index_start
    loop_end = text_config.loop_layer_index_end
    idx = int(layer_idx)
    if idx < loop_start:
        unit = idx
    elif idx < loop_end:
        unit = loop_start
    else:
        unit = loop_start + 1 + (idx - loop_end)
    total_units = _inkling2_loop_unit_count(text_config)
    offset = int(getattr(module.config, "pipeline_layer_offset", 0) or 0)
    explicit_total = getattr(module.config, "pipeline_layer_total", None)
    resolved_total = int(explicit_total) if explicit_total else total_units
    return offset + unit, module._pipeline_effective_layer_total(resolved_total)


@register_module(TaskType.BASE_MODULE, config=Inkling2LoopTextConfig, model_type="inkling2_loop_text")
class Inkling2LoopTextModel(Inkling2TextModel):
    """Inkling 2 Loop text decoder repeating ``[loop_layer_index_start, loop_layer_index_end)``.

    Under MPMD pipeline parallelism (``pp > 1``), the entire looped core
    ``[loop_layer_index_start, loop_layer_index_end)`` and its cross-pass
    :attr:`message_input` are mapped to one atomic pipeline unit so every pass
    executes locally on a single stage rank with zero cyclic dependencies or
    cross-rank weight transfers.
    """

    fp32_param_patterns: tp.ClassVar[tuple[str, ...]] = LOOP_FP32_PARAM_PATTERNS
    checkpoint_key_rules: tp.ClassVar[Inkling2KeyNormalizer] = INKLING2_LOOP_TEXT_KEY_RULES

    def __init__(
        self,
        config: Inkling2LoopTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        self._validate_mpmd_loop_units(config)
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        with self.assign_layer_stage(config.loop_layer_index_start, total_layers=config.num_hidden_layers):
            self.message_input = LoopMessageAttentionResidual(
                hidden_size=config.hidden_size,
                num_loop_layers=config.num_loop_layers,
                num_messages=4,
                rngs=rngs,
            )

    def _pipeline_loop_unit_count(self, config: Inkling2LoopTextConfig | None = None) -> int:
        """Number of acyclic pipeline units: pre-loop layers + 1 looped-core unit + post-loop layers."""
        cfg: Inkling2LoopTextConfig = self.config if config is None else config  # type: ignore[assignment]
        return _inkling2_loop_unit_count(cfg)

    def _validate_mpmd_loop_units(self, config: Inkling2LoopTextConfig) -> None:
        """Ensure logical PP stage count does not exceed the acyclic pipeline unit count."""
        mesh_shape = getattr(getattr(config, "mesh", None), "shape", None)
        try:
            physical_pp = int(mesh_shape["pp"])
        except (KeyError, TypeError, AttributeError):
            physical_pp = 1
        logical_pp = physical_pp * int(getattr(config, "pipeline_virtual_stages", 1) or 1)
        total_units = self._pipeline_loop_unit_count(config)
        if logical_pp > 1 and logical_pp > total_units:
            raise ValueError(
                f"Inkling2LoopTextModel keeps its looped core [{config.loop_layer_index_start}, "
                f"{config.loop_layer_index_end}) on a single pipeline stage, leaving {total_units} "
                f"acyclic pipeline units ({config.loop_layer_index_start} pre-loop + 1 loop core + "
                f"{config.num_hidden_layers - config.loop_layer_index_end} post-loop), which is fewer "
                f"than logical_pp={logical_pp}. Reduce pp/pipeline_virtual_stages or increase non-loop layers."
            )

    def _cache_slot_physical_layer(self, slot_idx: int) -> int:
        """Map both primary and extra-pass logical cache slots to their owning physical layer."""
        cfg: Inkling2LoopTextConfig = self.config  # type: ignore[assignment]
        idx = int(slot_idx)
        if idx < cfg.num_hidden_layers:
            return idx
        return cfg.loop_layer_index_start + ((idx - cfg.num_hidden_layers) % cfg.num_loop_layers)

    def _pipeline_layer_position(self, layer_idx: int, total_layers: int) -> tuple[int, int]:
        """Map physical layer indices into acyclic pipeline unit positions for MPMD."""
        del total_layers
        cfg: Inkling2LoopTextConfig = self.config  # type: ignore[assignment]
        return _inkling2_loop_pipeline_layer_position(self, cfg, layer_idx)

    def _build_layers(
        self,
        config: Inkling2LoopTextConfig,  # type: ignore[override]
        *,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        precision: jax.lax.PrecisionLike,
        rngs: spx.Rngs,
    ) -> nn.ModuleList:
        remat_kw = dict(
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        layer_classes = {
            (False, False): auto_remat(Inkling2FullDecoderLayer, **remat_kw),
            (True, False): auto_remat(Inkling2SlidingDecoderLayer, **remat_kw),
            (False, True): auto_remat(Inkling2LoopFullLoopDecoderLayer, **remat_kw),
            (True, True): auto_remat(Inkling2LoopSlidingLoopDecoderLayer, **remat_kw),
        }
        layers = nn.ModuleList([])
        for layer_idx in range(config.num_hidden_layers):
            is_sliding = config.is_sliding_layer(layer_idx)
            is_loop = config.loop_layer_index_start <= layer_idx < config.loop_layer_index_end
            layer_cls = layer_classes[(is_sliding, is_loop)]
            with self.assign_layer_stage(layer_idx, total_layers=config.num_hidden_layers):
                layers.append(
                    layer_cls(
                        config=config,
                        layer_idx=layer_idx,
                        dtype=dtype,
                        param_dtype=param_dtype,
                        precision=precision,
                        rngs=rngs,
                    )
                )
        return layers

    @property
    def reform_param(self) -> dict[str, tp.Any]:
        """Merge vendor ``middle_cycle_cross_pass_*queries`` into ``message_input.query``."""
        return {
            "message_input.query$": {
                "sources": ("middle_cycle_cross_pass_queries", "middle_cycle_cross_pass_kv_queries"),
                "fuser": merge_inkling2_loop_message_queries,
                "inverse_fuser": split_inkling2_loop_message_queries,
                "already_converted": True,
                "log_label": "Inkling2Loop cross-pass message queries",
            }
        }

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
        mode: common_types.RUNTIME_MODE_TYPES | None = None,
        past_key_values: ConvKVCache | None = None,
        cache_metadata: TransformerMetadata | OperationsMetadata | None = None,
    ) -> MoeModelOutput:
        inputs_embeds, mask_info, position_ids, mode, past_key_values, conv_mask, seq_idx = self._prepare_forward_inputs(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
        )
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        output_router_logits = self.config.output_router_logits if output_router_logits is None else output_router_logits

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        with self._layer_stage_context(0, layers=self.layers):
            hidden_states = apply_logical_sharding(
                inputs_embeds,
                dynamic_axes=common_types.HiddenStateSharding,
                partition_manager=self.config.runtime_sharding_resolver,
            )
        block_states = jnp.zeros((*inputs_embeds.shape[:-1], 0, inputs_embeds.shape[-1]), dtype=inputs_embeds.dtype)
        views = list(past_key_values.views)

        loop_start = self.config.loop_layer_index_start
        loop_end = self.config.loop_layer_index_end
        num_loop_layers = loop_end - loop_start

        def _run_layer(layer_idx: int, cache_slot: int, cur_hidden: Array, cur_blocks: Array, messages: Array | None):
            nonlocal all_hidden_states, all_attentions, all_router_logits
            decoder_layer = self.layers[layer_idx]
            if output_hidden_states:
                all_hidden_states = (*all_hidden_states, cur_hidden)
            with self._layer_stage_context(layer_idx, layers=self.layers):
                layer_out = decoder_layer(
                    hidden_states=cur_hidden,
                    block_states=cur_blocks,
                    mask_info=mask_info,
                    position_ids=position_ids,
                    mode=mode,
                    cache_view=views[cache_slot],
                    cache_metadata=cache_metadata,
                    conv_mask=conv_mask,
                    seq_idx=seq_idx,
                    messages=messages,
                    output_attentions=bool(output_attentions),
                    output_router_logits=bool(output_router_logits),
                )
            views[cache_slot] = layer_out.cache_view
            if decoder_layer.is_block_boundary:
                cur_blocks = jnp.concatenate([cur_blocks, cur_hidden[..., None, :]], axis=-2)
            if output_attentions:
                all_attentions = (*all_attentions, layer_out.attention_weight)
            if output_router_logits:
                all_router_logits = (*all_router_logits, layer_out.router_logits)
            return layer_out.hidden_states, cur_blocks

        # Pre-loop layers (may cross PP stage boundaries between layers and into the loop core).
        for layer_idx in range(loop_start):
            hidden_states, block_states = _run_layer(layer_idx, layer_idx, hidden_states, block_states, None)
            hidden_states, block_states = self._mark_residual_stage_boundary(hidden_states, block_states, layer_idx)

        pre_loop_hidden_states, pre_loop_block_states = hidden_states, block_states
        num_pre_loop_blocks = pre_loop_block_states.shape[-2]

        # All passes through the looped core execute on a single pipeline stage.
        for layer_idx in range(loop_start, loop_end):
            hidden_states, block_states = _run_layer(layer_idx, layer_idx, hidden_states, block_states, None)

        for loop_idx in range(1, self.config.loop_num_passes):
            previous_block_states = block_states[..., num_pre_loop_blocks + 1 :, :]
            previous_hidden_states = hidden_states
            hidden_states, block_states = pre_loop_hidden_states, pre_loop_block_states

            cache_offset = self.config.num_hidden_layers + (loop_idx - 1) * num_loop_layers - loop_start
            for loop_layer_idx, layer_idx in enumerate(range(loop_start, loop_end)):
                with self._layer_stage_context(layer_idx, layers=self.layers):
                    messages = self.message_input(previous_block_states, previous_hidden_states, loop_layer_idx)
                hidden_states, block_states = _run_layer(
                    layer_idx, layer_idx + cache_offset, hidden_states, block_states, messages
                )

        # Transition from looped core into post-loop layers (emits boundary iff stages differ).
        hidden_states, block_states = self._mark_residual_stage_boundary(
            hidden_states, block_states, loop_end - 1, next_layer_idx=loop_end
        )

        # Post-loop layers
        for layer_idx in range(loop_end, self.config.num_hidden_layers):
            hidden_states, block_states = _run_layer(layer_idx, layer_idx, hidden_states, block_states, None)
            hidden_states, block_states = self._mark_residual_stage_boundary(hidden_states, block_states, layer_idx)

        hidden_states = self._finalize_hidden_states(block_states, hidden_states, inputs_embeds.dtype)
        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        return MoeModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=ConvKVCache(views=views),
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            router_logits=all_router_logits,
        )


# ---------------------------------------------------------------------------
# Shared unpadded LM head mixin & CausalLM heads
# ---------------------------------------------------------------------------


def _make_unpadded_lm_head_class(unpadded_vocab_size: int) -> type[ColumnParallelLinear]:
    """Return a :class:`ColumnParallelLinear` subclass projecting to ``unpadded_vocab_size``."""

    class _Inkling2LMHead(ColumnParallelLinear):
        def __init__(self, in_features: int, out_features: int, **kwargs):
            del out_features
            super().__init__(in_features, unpadded_vocab_size, **kwargs)

    return _Inkling2LMHead


class _Inkling2UnpaddedLMHeadMixin:
    """Shared unpadded ``lm_head`` checkpoint rules, fp32 logit softcapping, and cache delegation."""

    vocab_size: int
    unpadded_vocab_size: int
    _final_logit_softcapping: float

    @property
    def reform_param(self) -> dict[str, tp.Any]:
        """Unpad ``lm_head.weight`` on load and zero-pad back to ``vocab_size`` on export."""
        unpadded_v = self.unpadded_vocab_size
        padded_v = self.vocab_size
        return {
            "lm_head.weight$": {
                "splits": [
                    {
                        "name": "lm_head.weight",
                        "spliter": lambda w: unpad_inkling2_lm_head(w, unpadded_v),
                    }
                ],
                "inverse_spliter": lambda w: pad_inkling2_lm_head(w, padded_v),
            }
        }

    def apply_logit_cap(self, logits: Array) -> Array:
        """Promote LM logits to fp32 and apply ``tanh(logits / cap) * cap``."""
        logits = logits.astype(jnp.float32)
        cap = self._final_logit_softcapping
        if cap > 0.0:
            logits = jnp.tanh(logits / cap) * cap
        return logits

    def get_lm_logit_postprocess(self) -> tp.Callable[[Array], Array] | None:
        """Snapshot the fp32 softcapping transform for chunked-loss and MPMD LM-head paths."""
        cap = float(self._final_logit_softcapping)

        def _postprocess(logits: Array) -> Array:
            out = logits.astype(jnp.float32)
            if cap > 0.0:
                out = jnp.tanh(out / cap) * cap
            return out

        return _postprocess

    def init_cache(self, batch_size, max_length, starts=None, shardings=None, pad_token_id=None) -> ConvKVCache:
        return self.base_model.init_cache(batch_size, max_length, starts, shardings, pad_token_id)


@register_module(TaskType.CAUSAL_LM, config=Inkling2TextConfig, model_type="inkling2_text")
@register_module(TaskType.CAUSAL_LM, config=Inkling2Config, model_type="inkling2")
class Inkling2ForCausalLM(_Inkling2UnpaddedLMHeadMixin, BaseCausalLMModule[Inkling2TextModel, Inkling2TextConfig]):
    """Inkling 2 text model with unpadded softcapped language-modeling head."""

    _task_type = TaskType.CAUSAL_LM
    _model_type = "inkling2_text"
    _config_class = Inkling2TextConfig
    _auto_register = False

    fp32_param_patterns: tp.ClassVar[tuple[str, ...]] = FP32_PARAM_PATTERNS
    checkpoint_key_rules: tp.ClassVar[Inkling2KeyNormalizer] = INKLING2_CAUSAL_LM_KEY_RULES

    def __init__(
        self,
        config: Inkling2TextConfig | Inkling2Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        _base_model_class: type[Inkling2TextModel] = Inkling2TextModel,
    ):
        text_config = config.get_text_config() if isinstance(config, Inkling2Config) else config
        self._final_logit_softcapping = float(text_config.final_logit_softcapping)
        super().__init__(
            config=text_config,
            base_model_class=_base_model_class,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            tie_word_embeddings=False,
            logit_cap=None,
            router_aux_loss_coef=getattr(text_config, "router_aux_loss_coef", 0.0) or None,
            lm_head_name="lm_head",
            lm_head_bias=False,
            lm_head_kernel_init=jax.nn.initializers.normal(text_config.initializer_range),
            lm_head_class=_make_unpadded_lm_head_class(text_config.unpadded_vocab_size),
        )
        self.vocab_size = text_config.vocab_size
        self.unpadded_vocab_size = text_config.unpadded_vocab_size

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,
        past_key_values: ConvKVCache | None = None,
        cache_metadata: TransformerMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
    ) -> MoeCausalLMOutput:
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


@register_module(TaskType.CAUSAL_LM, config=Inkling2LoopTextConfig, model_type="inkling2_loop_text")
@register_module(TaskType.CAUSAL_LM, config=Inkling2LoopConfig, model_type="inkling2_loop")
class Inkling2LoopForCausalLM(Inkling2ForCausalLM):
    """Inkling 2 Loop text model with unpadded softcapped language-modeling head."""

    _task_type = TaskType.CAUSAL_LM
    _model_type = "inkling2_loop_text"
    _config_class = Inkling2LoopTextConfig
    _auto_register = False

    fp32_param_patterns: tp.ClassVar[tuple[str, ...]] = LOOP_FP32_PARAM_PATTERNS
    checkpoint_key_rules: tp.ClassVar[Inkling2KeyNormalizer] = INKLING2_LOOP_CAUSAL_LM_KEY_RULES

    def __init__(
        self,
        config: Inkling2LoopTextConfig | Inkling2LoopConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            _base_model_class=Inkling2LoopTextModel,
        )

    def _pipeline_layer_position(self, layer_idx: int, total_layers: int) -> tuple[int, int]:
        """Match :class:`Inkling2LoopTextModel` acyclic MPMD unit positions for task-head stage placement."""
        del total_layers
        text_cfg = self.config.get_text_config() if isinstance(self.config, Inkling2Config) else self.config
        return _inkling2_loop_pipeline_layer_position(self, text_cfg, layer_idx)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Multimodal models & ConditionalGeneration heads (MPMD + process_encoder + gen)
# ---------------------------------------------------------------------------


@register_module(TaskType.VISION_LM, config=Inkling2Config, model_type="inkling2")
@register_module(TaskType.BASE_MODULE, config=Inkling2Config, model_type="inkling2")
class Inkling2Model(EasyDeLBaseModule):
    """Multimodal Inkling 2 backbone combining :class:`Inkling2VisionModel` and :class:`Inkling2TextModel`."""

    fp32_param_patterns: tp.ClassVar[tuple[str, ...]] = FP32_PARAM_PATTERNS
    checkpoint_key_rules: tp.ClassVar[Inkling2KeyNormalizer] = INKLING2_VLM_MODEL_KEY_RULES

    def __init__(
        self,
        config: Inkling2Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        _language_model_class: type[Inkling2TextModel] = Inkling2TextModel,
    ):
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        # Pin the vision tower to stage 0 so MPMD schedules execute vision
        # encoding on the same rank as text token embeddings before stage 0 -> 1.
        with self.assign_layer_stage(0, total_layers=config.text_config.num_hidden_layers):
            self.vision_tower = Inkling2VisionModel(
                config.vision_config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        self.language_model = _language_model_class(
            config.text_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def get_image_features(
        self,
        pixel_values: Array,
        image_grid_thw: Array | np.ndarray,
    ) -> Array:
        """Encode image patches into LM-width token embeddings ``[num_image_tokens, text_hidden]``."""
        with self.language_model._layer_stage_context(0, layers=self.language_model.layers):
            return self.vision_tower(pixel_values=pixel_values, image_grid_thw=image_grid_thw).pooler_output

    def compute_embedding(
        self,
        input_ids: Int[Array, "batch seq_len"] | None,
        *,
        image_features: Array | None = None,
        image_features_valid_length: Int[Array, ""] | int | None = None,
        pixel_values: Array | None = None,
        image_grid_thw: Array | np.ndarray | None = None,
        **kwargs,
    ) -> Array:
        """Embed text tokens and scatter vision features (computed or precomputed) at ``image_token_id``."""
        del kwargs
        if input_ids is None:
            raise ValueError("`input_ids` must be provided when calling `compute_embedding`.")
        vocab_size = self.config.text_config.vocab_size
        image_token_id = self.config.image_token_id
        safe_ids = jnp.where(input_ids == image_token_id, 0, input_ids) if image_token_id >= vocab_size else input_ids
        with self.language_model._layer_stage_context(0, layers=self.language_model.layers):
            inputs_embeds = checkpoint_name(self.language_model.embed_tokens(safe_ids.astype("i4")), "embeddings")

            if image_features is None and pixel_values is not None:
                if image_grid_thw is None:
                    raise ValueError("`image_grid_thw` must be provided when `pixel_values` is given.")
                image_features = self.get_image_features(pixel_values, image_grid_thw)

            if image_features is not None:
                flat_features = image_features.reshape(-1, image_features.shape[-1]).astype(inputs_embeds.dtype)
                if image_features_valid_length is not None:
                    valid_mask = jnp.arange(flat_features.shape[0], dtype=jnp.int32) < jnp.asarray(
                        image_features_valid_length, dtype=jnp.int32
                    )
                    flat_features = jnp.where(valid_mask[:, None], flat_features, jnp.zeros_like(flat_features))
                inputs_embeds = BaseVisionLanguageModule.merge_multimodal_embeddings(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    multimodal_embeddings=flat_features,
                    placeholder_token_id=image_token_id,
                )
        return inputs_embeds

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        pixel_values: Array | None = None,
        image_grid_thw: Array | np.ndarray | None = None,
        image_features: Array | None = None,
        image_features_valid_length: Int[Array, ""] | int | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,
        past_key_values: ConvKVCache | None = None,
        cache_metadata: TransformerMetadata | OperationsMetadata | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
    ) -> Inkling2ModelOutput:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of `input_ids` or `inputs_embeds`.")
        if (pixel_values is not None or image_features is not None) and input_ids is None:
            raise ValueError("`input_ids` must be provided when `pixel_values` or `image_features` is given.")

        if image_features is None and pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("`image_grid_thw` must be provided when `pixel_values` is given.")
            image_features = self.get_image_features(pixel_values, image_grid_thw)

        if inputs_embeds is None:
            inputs_embeds = self.compute_embedding(
                input_ids,
                image_features=image_features,
                image_features_valid_length=image_features_valid_length,
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
        )
        return Inkling2ModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            image_hidden_states=image_features,
        )

    def init_cache(self, batch_size, max_length, starts=None, shardings=None, pad_token_id=None) -> ConvKVCache:
        return self.language_model.init_cache(batch_size, max_length, starts, shardings, pad_token_id)

    def prepare_inputs_for_generation(
        self,
        input_ids: Int[Array, "batch seq_len"],
        max_length: int,
        pad_token_id: int,
        starts: int | None = None,
        pixel_values: Array | None = None,
        image_grid_thw: Array | np.ndarray | None = None,
        image_features: Array | None = None,
        image_features_valid_length: Int[Array, ""] | int | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
    ) -> dict[str, tp.Any]:
        """Prepare KV + short-conv cache and multimodal inputs for autoregressive generation."""
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            max_length=max_length,
            pad_token_id=pad_token_id,
            starts=starts,
            attention_mask=attention_mask,
        )
        model_inputs["pixel_values"] = pixel_values
        model_inputs["image_grid_thw"] = image_grid_thw
        model_inputs["image_features"] = image_features
        model_inputs["image_features_valid_length"] = image_features_valid_length
        return model_inputs

    def update_inputs_for_generation(self, model_outputs, model_kwargs) -> dict[str, tp.Any]:
        """Advance decode state and drop first-step-only multimodal inputs."""
        model_kwargs = self.language_model.update_inputs_for_generation(model_outputs, model_kwargs)
        for key in ("pixel_values", "image_grid_thw", "image_features", "image_features_valid_length"):
            model_kwargs.pop(key, None)
        return model_kwargs

    def get_encoder(self):
        return self.vision_tower

    def get_decoder(self):
        return self.language_model.get_decoder()

    def get_lm_head(self):
        raise NotImplementedError("Base Inkling2Model has no language model head.")

    def get_embedding(self):
        return self.language_model.get_embedding()


@register_module(TaskType.IMAGE_TEXT_TO_TEXT, config=Inkling2Config, model_type="inkling2")
class Inkling2ForConditionalGeneration(
    _Inkling2UnpaddedLMHeadMixin,
    BaseVisionLanguageModule[Inkling2Model, Inkling2Config],  # type: ignore[type-var]
):
    """Inkling 2 multimodal vision-language model with unpadded softcapped LM head."""

    _task_type = TaskType.IMAGE_TEXT_TO_TEXT
    _model_type = "inkling2"
    _config_class = Inkling2Config
    _auto_register = False
    _supports_video = False
    _uses_mrope = False

    _vision_tower_name = "vision_tower"
    _projector_name = "merger"
    _language_model_name = "language_model"
    loss_type = "ForCausalLM"

    fp32_param_patterns: tp.ClassVar[tuple[str, ...]] = FP32_PARAM_PATTERNS
    checkpoint_key_rules: tp.ClassVar[Inkling2KeyNormalizer] = INKLING2_VLM_KEY_RULES

    def __init__(
        self,
        config: Inkling2Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        _base_model_class: type[Inkling2Model] = Inkling2Model,
    ):
        text_config = config.get_text_config()
        self._final_logit_softcapping = float(text_config.final_logit_softcapping)
        super().__init__(
            config=config,
            base_model_class=_base_model_class,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            image_token_index=config.image_token_id,
            spatial_merge_size=config.vision_config.spatial_merge_size,
            router_aux_loss_coef=getattr(text_config, "router_aux_loss_coef", 0.0) or None,
            tie_word_embeddings=False,
            logit_cap=None,
            lm_head_name="lm_head",
            create_lm_head=False,
            lm_head_bias=False,
            lm_head_kernel_init=jax.nn.initializers.normal(text_config.initializer_range),
        )
        head_cls = ColumnParallelLinear
        if self._gradient_checkpointing_feature.should_checkpoint():
            head_cls = auto_remat(head_cls, **self._gradient_checkpointing_feature.get_config())

        def _build_unpadded_head():
            return head_cls(
                text_config.hidden_size,
                text_config.unpadded_vocab_size,
                dtype=dtype,
                param_dtype=param_dtype,
                use_bias=False,
                kernel_init=jax.nn.initializers.normal(text_config.initializer_range),
                precision=precision,
                rngs=rngs,
            )

        self.lm_head = self._create_task_head_on_last_stage(_build_unpadded_head)
        self.vocab_size = text_config.vocab_size
        self.unpadded_vocab_size = text_config.unpadded_vocab_size

    @property
    def vision_tower(self):
        return self.base_model.vision_tower

    @property
    def language_model(self):
        return self.base_model.language_model

    def get_image_features(self, pixel_values: Array, image_grid_thw: Array | np.ndarray, **kwargs) -> Array:
        del kwargs
        return self.base_model.get_image_features(pixel_values, image_grid_thw)

    def compute_embedding(self, input_ids, *args, **kwargs):
        return self.base_model.compute_embedding(input_ids, *args, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, max_length, pad_token_id, starts=None, **kwargs):
        return self.base_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            max_length=max_length,
            pad_token_id=pad_token_id,
            starts=starts,
            **kwargs,
        )

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        return self.base_model.update_inputs_for_generation(model_outputs, model_kwargs)

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        pixel_values: Array | None = None,
        image_grid_thw: Array | np.ndarray | None = None,
        image_features: Array | None = None,
        image_features_valid_length: Int[Array, ""] | int | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,
        past_key_values: ConvKVCache | None = None,
        cache_metadata: TransformerMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        **kwargs,
    ) -> VLMCausalLMOutput:
        del kwargs
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_features=image_features,
            image_features_valid_length=image_features_valid_length,
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
        hidden_states = self.prepare_lm_head_inputs(outputs.last_hidden_state)
        lm_logits = self.compute_lm_logits(hidden_states) if apply_lm_head else None
        aux_loss = self.compute_router_aux_loss(outputs, attention_mask=attention_mask)
        return VLMCausalLMOutput(
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            last_hidden_state=hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            aux_loss=aux_loss,
            image_hidden_states=outputs.image_hidden_states,
        )


@register_module(TaskType.VISION_LM, config=Inkling2LoopConfig, model_type="inkling2_loop")
@register_module(TaskType.BASE_MODULE, config=Inkling2LoopConfig, model_type="inkling2_loop")
class Inkling2LoopModel(Inkling2Model):
    """Multimodal Inkling 2 Loop backbone combining :class:`Inkling2VisionModel` and :class:`Inkling2LoopTextModel`."""

    fp32_param_patterns: tp.ClassVar[tuple[str, ...]] = LOOP_FP32_PARAM_PATTERNS
    checkpoint_key_rules: tp.ClassVar[Inkling2KeyNormalizer] = INKLING2_LOOP_VLM_MODEL_KEY_RULES

    def __init__(
        self,
        config: Inkling2LoopConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            _language_model_class=Inkling2LoopTextModel,
        )

    def _pipeline_layer_position(self, layer_idx: int, total_layers: int) -> tuple[int, int]:
        """Match :class:`Inkling2LoopTextModel` acyclic MPMD unit positions for vision-tower placement."""
        del total_layers
        return _inkling2_loop_pipeline_layer_position(self, self.config.get_text_config(), layer_idx)  # type: ignore[arg-type]


@register_module(TaskType.IMAGE_TEXT_TO_TEXT, config=Inkling2LoopConfig, model_type="inkling2_loop")
class Inkling2LoopForConditionalGeneration(Inkling2ForConditionalGeneration):
    """Inkling 2 Loop multimodal vision-language model with unpadded softcapped LM head."""

    _task_type = TaskType.IMAGE_TEXT_TO_TEXT
    _model_type = "inkling2_loop"
    _config_class = Inkling2LoopConfig
    _auto_register = False

    fp32_param_patterns: tp.ClassVar[tuple[str, ...]] = LOOP_FP32_PARAM_PATTERNS
    checkpoint_key_rules: tp.ClassVar[Inkling2KeyNormalizer] = INKLING2_LOOP_VLM_KEY_RULES

    def __init__(
        self,
        config: Inkling2LoopConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            _base_model_class=Inkling2LoopModel,
        )

    def _pipeline_layer_position(self, layer_idx: int, total_layers: int) -> tuple[int, int]:
        """Match :class:`Inkling2LoopTextModel` acyclic MPMD unit positions for LM-head placement."""
        del total_layers
        return _inkling2_loop_pipeline_layer_position(self, self.config.get_text_config(), layer_idx)  # type: ignore[arg-type]


__all__ = [
    "INKLING2_CAUSAL_LM_KEY_RULES",
    "INKLING2_LOOP_CAUSAL_LM_KEY_RULES",
    "INKLING2_LOOP_TEXT_KEY_RULES",
    "INKLING2_LOOP_VLM_KEY_RULES",
    "INKLING2_LOOP_VLM_MODEL_KEY_RULES",
    "INKLING2_TEXT_KEY_RULES",
    "INKLING2_VISION_KEY_RULES",
    "INKLING2_VLM_KEY_RULES",
    "INKLING2_VLM_MODEL_KEY_RULES",
    "Inkling2BlockAttentionResidual",
    "Inkling2Cache",
    "Inkling2CacheConfig",
    "Inkling2CacheView",
    "Inkling2DynamicShortConvolution",
    "Inkling2Experts",
    "Inkling2ForCausalLM",
    "Inkling2ForConditionalGeneration",
    "Inkling2FullAttention",
    "Inkling2FullDecoderLayer",
    "Inkling2GatedOutputProjection",
    "Inkling2KeyNormalizer",
    "Inkling2KeyValueProjection",
    "Inkling2L2Clip",
    "Inkling2LogScale",
    "Inkling2LogScaledL2Clip",
    "Inkling2LoopForCausalLM",
    "Inkling2LoopForConditionalGeneration",
    "Inkling2LoopFullLoopDecoderLayer",
    "Inkling2LoopMessageAttentionResidual",
    "Inkling2LoopMessageGate",
    "Inkling2LoopModel",
    "Inkling2LoopSlidingLoopDecoderLayer",
    "Inkling2LoopTextModel",
    "Inkling2Model",
    "Inkling2ModelOutput",
    "Inkling2NormedEmbedding",
    "Inkling2QueryProjection",
    "Inkling2RMSNorm",
    "Inkling2RelativePositionBias",
    "Inkling2ScaledL2Clip",
    "Inkling2ShortConvolution",
    "Inkling2SlidingAttention",
    "Inkling2SlidingDecoderLayer",
    "Inkling2SoftcappedSwiGLU",
    "Inkling2SparseMoeBlock",
    "Inkling2TextModel",
    "Inkling2TopKRouter",
    "Inkling2UnweightedRMSNorm",
    "Inkling2VisionAttention",
    "Inkling2VisionEmbeddings",
    "Inkling2VisionEncoderLayer",
    "Inkling2VisionMLP",
    "Inkling2VisionMerger",
    "Inkling2VisionModel",
    "Inkling2VisionModelOutput",
    "Inkling2VisionRotaryEmbedding",
    "apply_rotary_pos_emb_vision",
    "create_inkling2_cache_config",
    "deinterleave_inkling2_gate_up",
    "dequantize_inkling2_experts",
    "interleave_inkling2_gate_up",
    "merge_inkling2_loop_message_queries",
    "pad_inkling2_lm_head",
    "split_inkling2_loop_message_queries",
    "unpad_inkling2_lm_head",
]
