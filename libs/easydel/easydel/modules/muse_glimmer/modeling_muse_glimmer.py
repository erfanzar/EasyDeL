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

"""Muse-Glimmer multimodal model implementation for EasyDeL.

Port of HuggingFace's ``muse_glimmer`` family: a packed windowed ViT tower
feeding a decoder-only language model through a two-layer adapter.

Language model specifics (all mirrored from the reference implementation):

- **Gated attention** — a ``gate_proj`` the width of the query projection whose
  sigmoid multiplies the attention output before ``o_proj``. It is fused into
  the same column-parallel matmul as Q/K/V (segments ``[q | gate | k | v]``), so
  a layer still issues a single projection matmul while HF's four separate
  checkpoint tensors load through the fused layout's reform rules.
- **Scale-less QK-norm** — Q and K are RMS-normalized with no learnable scale,
  and Q is then multiplied by ``qk_scale_factor`` on top of ``1/sqrt(head_dim)``.
- **Per-layer RoPE base** — ``config.layer_rope_theta[i]`` gives the layer's RoPE
  base; ``0`` marks a NoPE layer, which is handed an identity rotation table.
- **Sandwich norms** — every sub-layer output is normalized (with the tighter
  ``post_norm_eps``) before being added back to the residual.
- **Soft-capped logits** — logits are scaled by ``output_multiplier`` and then
  passed through ``T * tanh(x / T)`` with ``T = final_logit_softcapping``.

Vision tower specifics:

- Patches arrive pre-flattened; a single linear embeds them and a learned
  ``pos_emb_height x pos_emb_width`` grid is bilinearly resampled onto each
  image's patch grid.
- Blocks alternate window attention and full attention; both are expressed as a
  block-diagonal additive bias built from the relevant cumulative sequence
  lengths, with tokens permuted into window order for the whole stack.
- The tower head pixel-shuffles ``merge_size x merge_size`` blocks into the
  channel axis, so the adapter sees ``hidden_size * merge_size**2`` features.

Grid-dependent index math (window order, cumulative lengths, position ids,
interpolation taps) is derived from concrete ``grid_thw`` values on the host,
matching the other packed-ViT ports in the zoo.
"""

import typing as tp
from functools import cached_property

import jax
import numpy as np
import spectrax as spx
from eformer.pytree import auto_pytree
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from jax import numpy as jnp
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
    VLMCausalLMOutput,
)
from easydel.infra.sequence_packing import token_attention_mask_from_mask_info
from easydel.infra.utils import ACT2FN, ArrayParam, auto_remat, blockwise_ffn
from easydel.layers import (
    ColumnParallelLinear,
    Embed,
    RMSNorm,
    RowParallelLinear,
    dense_gate_up_layout,
    gated_mlp_forward,
)
from easydel.layers.attention import FlexibleAttentionModule, UnifiedAttention
from easydel.layers.layouts import FusedColumnLayout, FusedSegment
from easydel.layers.norms import LayerNorm, lowfloats
from easydel.modules._base import BaseVisionLanguageModule

from .muse_glimmer_configuration import (
    MuseGlimmerConfig,
    MuseGlimmerTextConfig,
    MuseGlimmerVisionConfig,
)


@auto_pytree
class MuseGlimmerModelOutputWithPast(ModelOutput):
    """Output of :class:`MuseGlimmerModel`.

    Attributes:
        last_hidden_state: Final decoder hidden states,
            shape ``(batch, seq_len, text_hidden_size)``.
        past_key_values: Updated KV cache, when caching is active.
        hidden_states: Per-layer hidden states, when requested.
        attentions: Per-layer attention weights, when requested.
        image_hidden_states: Projected vision features merged into the sequence,
            shape ``(num_visual_tokens, text_hidden_size)``.
    """

    last_hidden_state: Array = None
    past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None
    hidden_states: tuple[Array, ...] | None = None
    attentions: tuple[Array, ...] | None = None
    image_hidden_states: Array | None = None


def _scaleless_rms_norm(hidden_states: Array, eps: float) -> Array:
    """RMS-normalize the trailing axis with no learnable scale.

    Mirrors HF's ``MuseGlimmerRMSNorm(with_scale=False)``: the statistic is
    computed in float32 and the result cast back to the input dtype. Used for
    the token-embedding norm, the attention QK-norm and the norm applied to
    projected vision features.

    Args:
        hidden_states: Tensor whose last axis is normalized.
        eps: Variance epsilon, added before the reciprocal square root.

    Returns:
        Array: Normalized tensor with the input dtype and shape.
    """
    original_dtype = hidden_states.dtype
    promoted = hidden_states.astype(jnp.float32)
    mean_squared = jnp.mean(jnp.square(promoted), axis=-1, keepdims=True) + eps
    return (promoted * jax.lax.rsqrt(mean_squared)).astype(original_dtype)


class MuseGlimmerCenteredRMSNorm(spx.Module):
    """RMSNorm whose learnable scale is applied as ``1 + weight``.

    The Gemma-style parameterization used by every norm inside a Muse-Glimmer
    decoder layer: ``weight`` is initialized to zeros so the layer starts as a
    plain RMS division, and both the normalization and the scaling run in
    float32 before the result is cast back to the input dtype.

    Attributes:
        epsilon (float): Variance epsilon added inside the reciprocal sqrt.
        param_dtype (DTypeLike): Storage dtype of ``weight``.
        weight (ArrayParam): Per-feature offset of shape ``(dim,)``.
    """

    kernel_init = staticmethod(jax.nn.initializers.zeros)

    def __init__(
        self,
        dim: int,
        epsilon: float = 1e-6,
        param_dtype: jnp.dtype = jnp.float32,
    ) -> None:
        """Initialize the centered RMS norm.

        Args:
            dim: Size of the trailing feature axis being normalized.
            epsilon: Variance epsilon. Defaults to 1e-6.
            param_dtype: Storage dtype of the learnable offset. Defaults to float32.
        """
        self.epsilon = epsilon
        self.param_dtype = param_dtype
        self.weight = ArrayParam.bound(
            shape=(dim,),
            dtype=param_dtype,
            init_method="zeros",
            key=None,
        )

    def forward(self, hidden_states: Float[Array, "... dim"]) -> Array:
        """Normalize ``hidden_states`` and scale by ``1 + weight``.

        Args:
            hidden_states: Tensor whose last axis is normalized.

        Returns:
            Array: Normalized, scaled tensor with the input dtype (promoted to
            bfloat16 when the input dtype is a low-precision float that cannot
            represent the scaled result).
        """
        original_dtype = hidden_states.dtype
        promoted = hidden_states.astype(jnp.float32)
        mean_squared = jnp.mean(jnp.square(promoted), axis=-1, keepdims=True) + self.epsilon
        normed = promoted * jax.lax.rsqrt(mean_squared)
        out = normed * (1.0 + self.weight.value.astype(jnp.float32))
        if original_dtype in lowfloats:
            return out.astype(jnp.bfloat16)
        return out.astype(original_dtype)


def _rotate_half(x: Array) -> Array:
    """Rotate the trailing axis by swapping and negating halves.

    Args:
        x: Tensor with an even-sized trailing dimension.

    Returns:
        Array: ``[-x2, x1]`` where ``x1``/``x2`` are the leading/trailing halves.
    """
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _apply_rotary_pos_emb_vision(q: Array, k: Array, cos: Array, sin: Array) -> tuple[Array, Array]:
    """Apply the vision tower's 2-D RoPE to query/key tensors.

    Promotes to float32 for the trigonometric mix (matching the reference
    implementation, which is explicit about the upcast) and restores the
    original dtypes afterwards.

    Args:
        q: Queries of shape ``(1, seq, heads, head_dim)``.
        k: Keys of shape ``(1, seq, heads, head_dim)``.
        cos: Cosine table of shape ``(seq, head_dim)``.
        sin: Sine table of shape ``(seq, head_dim)``.

    Returns:
        tuple[Array, Array]: Rotated ``(q, k)`` with the input shapes/dtypes.
    """
    q_dtype, k_dtype = q.dtype, k.dtype
    q32, k32 = q.astype(jnp.float32), k.astype(jnp.float32)
    cos32 = jnp.expand_dims(cos, -2).astype(jnp.float32)
    sin32 = jnp.expand_dims(sin, -2).astype(jnp.float32)
    q_out = (q32 * cos32) + (_rotate_half(q32) * sin32)
    k_out = (k32 * cos32) + (_rotate_half(k32) * sin32)
    return q_out.astype(q_dtype), k_out.astype(k_dtype)


def _block_diagonal_bias(cu_seqlens: np.ndarray, seq_length: int, dtype: jnp.dtype) -> Array:
    """Build an additive attention bias that confines attention to segments.

    Args:
        cu_seqlens: Cumulative sequence boundaries of shape ``(num_segments + 1,)``;
            segment ``i`` spans ``[cu_seqlens[i], cu_seqlens[i + 1])``.
        seq_length: Total packed length (equal to ``cu_seqlens[-1]``).
        dtype: Output dtype; masked pairs get ``finfo(dtype).min``.

    Returns:
        Array: Bias of shape ``(1, 1, seq_length, seq_length)``, ``0.0`` inside a
        segment and a large negative value across segments.
    """
    segment_ids = np.zeros((seq_length,), dtype=np.int32)
    for index in range(len(cu_seqlens) - 1):
        segment_ids[int(cu_seqlens[index]) : int(cu_seqlens[index + 1])] = index
    same_segment = jnp.asarray(segment_ids[:, None] == segment_ids[None, :])
    bias = jnp.where(same_segment, 0.0, jnp.finfo(dtype).min).astype(dtype)
    return bias[None, None, :, :]


def get_vision_cu_seqlens(grid_thw: np.ndarray) -> np.ndarray:
    """Cumulative per-frame patch counts for a packed vision batch.

    Args:
        grid_thw: Integer array of shape ``(num_images_or_videos, 3)`` holding
            ``(temporal, height, width)`` patch counts per entry.

    Returns:
        np.ndarray: ``(total_frames + 1,)`` int32 cumulative boundaries starting at 0.
    """
    per_frame = np.repeat(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    return np.concatenate([[0], np.cumsum(per_frame)]).astype(np.int32)


def get_vision_position_ids(grid_thw: np.ndarray, spatial_merge_size: int) -> np.ndarray:
    """Per-patch ``(height, width)`` indices in spatial-merge-block order.

    Port of HF ``transformers.vision_utils.get_vision_position_ids`` with
    ``include_temporal=False``.

    Args:
        grid_thw: ``(num_images_or_videos, 3)`` patch-grid dimensions.
        spatial_merge_size: Merge block size; ``1`` keeps raster order.

    Returns:
        np.ndarray: ``(total_patches, 2)`` int32 ``(h_index, w_index)`` pairs.
    """
    merge = spatial_merge_size
    parts = []
    for frames, height, width in grid_thw.tolist():
        frames, height, width = int(frames), int(height), int(width)
        h_ids, w_ids = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        block_shape = (height // merge, merge, width // merge, merge)
        h_ids = h_ids.reshape(block_shape).transpose(0, 2, 1, 3).reshape(-1)
        w_ids = w_ids.reshape(block_shape).transpose(0, 2, 1, 3).reshape(-1)
        parts.append(np.tile(np.stack([h_ids, w_ids], axis=-1), (frames, 1)))
    return np.concatenate(parts, axis=0).astype(np.int32)


def get_vision_window_index(
    grid_thw: np.ndarray,
    spatial_merge_size: int,
    window_size: int,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Window-attention permutation and its cumulative window boundaries.

    Port of HF ``transformers.vision_utils.get_vision_window_index``: patches
    are regrouped so that every attention window occupies a contiguous span.

    Args:
        grid_thw: ``(num_images_or_videos, 3)`` patch-grid dimensions.
        spatial_merge_size: Merge block size used to size the window in tokens.
        window_size: Window edge in pixels.
        patch_size: Patch edge in pixels.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(window_index, cu_window_seqlens)`` — a
        permutation of ``range(total_patches)`` and the cumulative window
        boundaries within the permuted order.
    """
    window_index: list[np.ndarray] = []
    cu_window_seqlens: list[int] = [0]
    offset = 0
    merger_window = window_size // spatial_merge_size // patch_size
    merge_unit = spatial_merge_size**2

    for grid_t, grid_h, grid_w in grid_thw.tolist():
        grid_t, grid_h, grid_w = int(grid_t), int(grid_h), int(grid_w)
        llm_h = grid_h // spatial_merge_size
        llm_w = grid_w // spatial_merge_size
        index = np.arange(grid_t * llm_h * llm_w).reshape(grid_t, llm_h, llm_w)
        pad_h = merger_window - llm_h % merger_window
        pad_w = merger_window - llm_w % merger_window
        num_windows_h = (llm_h + pad_h) // merger_window
        num_windows_w = (llm_w + pad_w) // merger_window
        padded = np.pad(index, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=-100)
        padded = padded.reshape(grid_t, num_windows_h, merger_window, num_windows_w, merger_window)
        padded = padded.transpose(0, 1, 3, 2, 4).reshape(
            grid_t, num_windows_h * num_windows_w, merger_window, merger_window
        )
        seqlens = (padded != -100).sum(axis=(2, 3)).reshape(-1)
        flat = padded.reshape(-1)
        window_index.append(flat[flat != -100] + offset)
        cu_window_seqlens.extend((np.cumsum(seqlens) * merge_unit + cu_window_seqlens[-1]).tolist())
        offset += grid_t * llm_h * llm_w

    merged = np.concatenate(window_index, axis=0).astype(np.int32)
    cumulative = np.asarray(cu_window_seqlens, dtype=np.int32)
    # Windows fully consumed by padding produce repeated boundaries; drop them.
    keep = np.concatenate([[True], cumulative[1:] != cumulative[:-1]])
    return merged, cumulative[keep]


def _interpolation_axis_taps_weights(
    index: np.ndarray,
    size: np.ndarray,
    side: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear taps/weights resampling a ``side``-long axis onto ``size`` points.

    Reproduces the ``align_corners=False`` half-pixel mapping with
    ``padding_mode="zeros"`` that Muse-Glimmer's reference position-embedding
    resampler uses, as two taps per target position.

    Args:
        index: Target index along the axis, one entry per patch.
        size: Target axis length for each patch (ragged batches supported).
        side: Source axis length (the learned position grid side).

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(taps, weights)``, each ``(n, 2)``.
    """
    src = (index.astype(np.float64) + 0.5) * side / size - 0.5
    floor = np.floor(src)
    offsets = np.arange(2)
    raw_taps = floor.astype(np.int64)[:, None] + offsets
    taps = np.clip(raw_taps, 0, side - 1)
    distance = np.abs(src[:, None] - floor[:, None] - offsets)
    weights = np.clip(1.0 - distance, 0.0, None)
    # Out-of-range taps were clamped to the border; zero them so they contribute
    # nothing, matching `F.grid_sample(padding_mode="zeros")`.
    weights = weights * ((raw_taps >= 0) & (raw_taps <= side - 1))
    return taps.astype(np.int32), weights.astype(np.float32)


def get_vision_interpolation_indices_and_weights(
    grid_thw: np.ndarray,
    num_grid_per_side: int,
    spatial_merge_size: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Gather indices/weights resampling the learned position grid per patch.

    Port of HF ``transformers.vision_utils.get_vision_interpolation_indices_and_weights``
    specialized to Muse-Glimmer's settings (bilinear, ``align_corners=False``,
    ``padding="zeros"``).

    Args:
        grid_thw: ``(num_images_or_videos, 3)`` patch-grid dimensions.
        num_grid_per_side: Side length of the square learned position grid.
        spatial_merge_size: Merge block size; ``1`` keeps raster patch order.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(indices, weights)`` of shape
        ``(total_patches, 4)`` — flat indices into the position table and their
        bilinear weights.
    """
    side = num_grid_per_side
    merge = spatial_merge_size

    counts = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
    heights = np.repeat(grid_thw[:, 1], counts)
    widths = np.repeat(grid_thw[:, 2], counts)
    starts = np.repeat(np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
    within = (np.arange(int(counts.sum())) - starts) % (heights * widths)

    blocks_w = widths // merge
    in_col = within % merge
    in_row = (within // merge) % merge
    block_col = (within // (merge * merge)) % blocks_w
    block_row = within // (merge * merge * blocks_w)
    row = block_row * merge + in_row
    col = block_col * merge + in_col

    h_taps, h_weights = _interpolation_axis_taps_weights(row, heights, side)
    w_taps, w_weights = _interpolation_axis_taps_weights(col, widths, side)
    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
    weights = (h_weights[:, :, None] * w_weights[:, None, :]).reshape(-1, 4)
    return indices.astype(np.int32), weights.astype(np.float32)


def get_vision_pixel_shuffle_index(grid_thw: np.ndarray, merge_size: int) -> np.ndarray:
    """Permutation grouping each ``merge_size**2`` spatial block contiguously.

    Args:
        grid_thw: ``(num_images_or_videos, 3)`` patch-grid dimensions.
        merge_size: Pixel-shuffle merge factor per spatial axis.

    Returns:
        np.ndarray: ``(total_patches,)`` gather index laying out every
        ``merge_size x merge_size`` block as consecutive rows.
    """
    indices = []
    offset = 0
    for frames, height, width in grid_thw.tolist():
        frames, height, width = int(frames), int(height), int(width)
        permutation = np.arange(height * width)
        permutation = permutation.reshape(height // merge_size, merge_size, width // merge_size, merge_size)
        permutation = permutation.transpose(0, 2, 1, 3).reshape(-1)
        if frames > 1:
            frame_offsets = (np.arange(frames) * height * width).reshape(frames, 1)
            permutation = (permutation[None, :] + frame_offsets).reshape(-1)
        indices.append(permutation + offset)
        offset += frames * height * width
    return np.concatenate(indices, axis=0).astype(np.int32)


def _muse_glimmer_attention_layout(*, q_size: int, gate_size: int, kv_size: int) -> FusedColumnLayout:
    """Build the fused ``[Q | gate | K | V]`` column-parallel layout.

    Args:
        q_size: Output channels of the query segment.
        gate_size: Output channels of the output-gate segment.
        kv_size: Output channels of each of the key and value segments.

    Returns:
        FusedColumnLayout: Layout describing the four labelled segments, used
        both for the runtime activation split and to derive the checkpoint
        reform rules that fuse HF's separate ``q_proj`` / ``gate_proj`` /
        ``k_proj`` / ``v_proj`` tensors.
    """
    return FusedColumnLayout(
        segments=(
            FusedSegment("q", q_size, "q_proj"),
            FusedSegment("gate", gate_size, "gate_proj"),
            FusedSegment("k", kv_size, "k_proj"),
            FusedSegment("v", kv_size, "v_proj"),
        ),
        log_label="Muse-Glimmer gated-attention Q/gate/K/V groups",
    )


class MuseGlimmerTextMLP(spx.Module):
    """SwiGLU feed-forward network for a Muse-Glimmer decoder layer."""

    def __init__(
        self,
        config: MuseGlimmerTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ) -> None:
        """Initialize the feed-forward block.

        Args:
            config: Text decoder configuration.
            dtype: Compute dtype. Defaults to bfloat16.
            param_dtype: Parameter storage dtype. Defaults to bfloat16.
            precision: JAX matmul precision. Defaults to None.
            rngs: SpecTrax RNGs consumed by the projection initializers.
            layer_idx: Index of the owning decoder layer.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.layer_idx = layer_idx

        self.gate_up_proj = ColumnParallelLinear(
            config.hidden_size,
            (config.intermediate_size, config.intermediate_size),
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
            layout=dense_gate_up_layout(config.intermediate_size),
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.act_fn = ACT2FN[config.hidden_activation]

    @property
    def reform_param(self):
        """Checkpoint reform rules mapping HF ``gate_proj``/``up_proj`` to the fused tensor."""
        return self.gate_up_proj.build_reform_param("gate_up_proj", config=self.config)

    def forward(self, hidden_states: Array) -> Array:
        """Apply the SwiGLU transformation.

        Args:
            hidden_states: Input of shape ``(batch, seq_len, hidden_size)``.

        Returns:
            Array: Output of shape ``(batch, seq_len, hidden_size)``.
        """
        return gated_mlp_forward(self, hidden_states)


class MuseGlimmerTextAttention(UnifiedAttention):
    """Gated causal attention with scale-less QK-norm and per-layer NoPE.

    Structural differences from stock :class:`UnifiedAttention`:

    - The fused projection carries four segments, ``[Q | gate | K | V]``, so
      HF's separate ``gate_proj`` tensor loads without a second matmul.
    - Q and K are RMS-normalized without a learnable scale (so the block owns no
      norm parameters), and Q is scaled by ``config.qk_scale_factor``.
    - Layers whose ``config.layer_rope_theta`` entry is ``0`` are handed an
      identity RoPE table, making them NoPE; the rest use their own base.
    - ``sigmoid(gate)`` multiplies the per-head attention output before ``o_proj``.
    """

    def __init__(
        self,
        config: MuseGlimmerTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ) -> None:
        """Initialize the attention block.

        Args:
            config: Text decoder configuration.
            dtype: Compute dtype. Defaults to bfloat16.
            param_dtype: Parameter storage dtype. Defaults to bfloat16.
            precision: JAX matmul precision. Defaults to None.
            rngs: SpecTrax RNGs consumed by the sub-module initializers.
            layer_idx: Zero-based index of this layer in the decoder stack.
        """
        self.is_sliding = config.layer_types is not None and config.layer_types[layer_idx] == "sliding_attention"
        # Read before `super().__init__` because `define_network` -> `_create_rotary`
        # runs inside it and needs the per-layer base.
        self.layer_rope_theta = float(config.layer_rope_theta[layer_idx])
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="standard",
            causal=True,
            sliding_window=config.sliding_window if self.is_sliding else None,
            use_qk_norm=False,
        )
        self.layer_idx = layer_idx
        self.qk_scale_factor = float(config.qk_scale_factor)
        self.qk_norm_eps = float(config.rms_norm_eps)

    def _create_fused_qkv_proj(self, config, dtype, param_dtype, precision, rngs):
        """Create the packed ``[Q | gate | K | V]`` column-parallel projection.

        Args:
            config: Owning :class:`MuseGlimmerTextConfig`.
            dtype: Compute dtype for the projection.
            param_dtype: Parameter dtype for the kernel.
            precision: JAX matmul precision.
            rngs: SpecTrax RNGs for initialization.

        Returns:
            ColumnParallelLinear: Projection carrying the four-segment layout.
        """
        q_out = self.num_heads * self.head_dim
        kv_out = self.num_key_value_heads * self.head_dim
        layout = _muse_glimmer_attention_layout(q_size=q_out, gate_size=q_out, kv_size=kv_out)
        return ColumnParallelLinear(
            config.hidden_size,
            layout.out_features,
            rngs=rngs,
            use_bias=getattr(config, "attention_bias", False),
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(getattr(config, "initializer_range", 0.02)),
            precision=precision,
            layout=layout,
        )

    def _create_rotary(self, config: MuseGlimmerTextConfig, dtype: jnp.dtype):
        """Build this layer's rotary embedding.

        Every layer gets a rotary module, including the NoPE ones
        (``config.layer_rope_theta[layer_idx] == 0``): that keeps the decoder
        stack structurally invariant so it can be scanned. NoPE is realized in
        the frequency table instead — the model hands those layers an identity
        (``cos = 1``, ``sin = 0``) table, which leaves Q/K untouched. The base
        recorded here is the layer's own theta, falling back to the global one
        for NoPE layers, where it is never consulted.

        Args:
            config: Owning configuration.
            dtype: Compute dtype for the rotary module.

        Returns:
            Rotary embedding module for this layer.
        """
        return config.get_basic_rope(
            dtype=dtype,
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            base=self.layer_rope_theta or config.rope_theta,
        )

    def _postprocess_qkv(self, query_states, key_states, value_states):
        """Apply the scale-less QK-norm and the extra query scale.

        Args:
            query_states: Queries of shape ``(batch, seq, heads, head_dim)``.
            key_states: Keys of shape ``(batch, seq, kv_heads, head_dim)``.
            value_states: Values, passed through unchanged.

        Returns:
            tuple: ``(query, key, value)`` with Q/K normalized and Q scaled.
        """
        query_states = _scaleless_rms_norm(query_states, self.qk_norm_eps) * self.qk_scale_factor
        key_states = _scaleless_rms_norm(key_states, self.qk_norm_eps)
        return query_states, key_states, value_states

    def forward_standard(
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
    ):
        """Run gated attention for one decoder layer.

        Identical to the stock standard path except that the fused projection
        yields a fourth ``gate`` segment whose sigmoid multiplies the per-head
        attention output before the output projection.

        Args:
            hidden_states: Residual stream of shape ``(batch, seq_len, hidden_size)``.
            mask_info: Mask container (mask, segment ids, positions).
            position_ids: Absolute token positions of shape ``(batch, seq_len)``.
            mode: Runtime mode (train / prefill / decode).
            cache_view: Optional KV-cache view, mutated by the cache concat step.
            cache_metadata: Optional companion cache metadata.
            output_attentions: When True, materialize softmax weights.
            frequencies: Precomputed RoPE table; the identity table on NoPE layers.
            alibi: Unused, accepted for signature parity.

        Returns:
            AttentionLayerOutput: Attention output, optional weights, cache view.
        """
        batch_size = hidden_states.shape[0]
        sequence_length = hidden_states.shape[1]

        qkv = checkpoint_name(self.query_key_value_projection(hidden_states), "attn_qkv")
        query_states, gate, key_states, value_states = self.query_key_value_projection.split(qkv, config=self.config)

        head_shape = (batch_size, sequence_length, -1, self.head_dim)
        query_states = query_states.reshape(head_shape)
        gate = gate.reshape(head_shape)
        key_states = key_states.reshape(head_shape)
        value_states = value_states.reshape(head_shape)

        query_states, key_states, value_states = self._postprocess_qkv(query_states, key_states, value_states)
        query_states, key_states, value_states = self.apply_qkv_shardings(query_states, key_states, value_states)
        query_states, key_states = self._apply_rotary(query_states, key_states, position_ids, frequencies)

        causal_for_kernel = self.causal
        if mask_info is not None and getattr(mask_info, "_causal_baked", False):
            causal_for_kernel = False

        sliding_window_for_kernel = self.sliding_window
        if mask_info is not None and getattr(mask_info, "sliding_window_baked_in", False):
            sliding_window_for_kernel = None

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
            sliding_window=sliding_window_for_kernel,
        )

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
            output_attentions=output_attentions,
        )

        if attentions.cache_view is not None:
            cache_view = attentions.cache_view

        attn_output = attentions.attention_outputs
        # Re-assert the query sharding so the gate (which still carries the
        # hidden-state sequence partitioning) and the attention output agree.
        attn_output = apply_logical_sharding(
            attn_output,
            dynamic_axes=common_types.AttnQSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )

        if attn_output.dtype in lowfloats or gate.dtype in lowfloats:
            output_dtype = attn_output.dtype
            attn_output = (attn_output.astype(jnp.float32) * jax.nn.sigmoid(gate.astype(jnp.float32))).astype(
                output_dtype
            )
        else:
            attn_output = attn_output * jax.nn.sigmoid(gate)

        attn_output = self._merge_heads(attn_output)
        attn_output = self.shard_attention_prod(attn_output, pre_projection=True)
        attn_output = checkpoint_name(self.output_projection(attn_output), "attn_output")
        attn_output = self.shard_attention_prod(attn_output)

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class MuseGlimmerTextDecoderLayer(spx.Module):
    """Decoder layer with sandwich norms around gated attention and the FFN."""

    def __init__(
        self,
        config: MuseGlimmerTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ) -> None:
        """Initialize the decoder layer.

        Args:
            config: Text decoder configuration.
            dtype: Compute dtype. Defaults to bfloat16.
            param_dtype: Parameter storage dtype. Defaults to bfloat16.
            precision: JAX matmul precision. Defaults to None.
            rngs: SpecTrax RNGs consumed by the sub-module initializers.
            layer_idx: Zero-based index of this layer.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.layer_idx = layer_idx

        self.self_attn = MuseGlimmerTextAttention(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )
        self.mlp = MuseGlimmerTextMLP(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )
        self.input_layernorm = MuseGlimmerCenteredRMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=param_dtype,
        )
        self.post_attention_layernorm = MuseGlimmerCenteredRMSNorm(
            config.hidden_size,
            epsilon=config.post_norm_eps,
            param_dtype=param_dtype,
        )
        self.pre_feedforward_layernorm = MuseGlimmerCenteredRMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=param_dtype,
        )
        self.post_feedforward_layernorm = MuseGlimmerCenteredRMSNorm(
            config.hidden_size,
            epsilon=config.post_norm_eps,
            param_dtype=param_dtype,
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
        frequencies: Array | None = None,
    ) -> DecoderLayerOutput:
        """Run one decoder layer.

        Applies ``residual + post_norm(sublayer(pre_norm(x)))`` twice — once for
        gated attention and once for the SwiGLU FFN.

        Args:
            hidden_states: Input of shape ``(batch, seq_len, hidden_size)``.
            mask_info: Attention mask information.
            position_ids: Absolute token positions.
            mode: Runtime mode (train / prefill / decode).
            cache_view: Optional KV-cache view.
            cache_metadata: Optional cache metadata.
            output_attentions: Whether to return attention weights.
            frequencies: Precomputed RoPE table for RoPE layers.

        Returns:
            DecoderLayerOutput: Hidden states, attention weights, cache view.
        """
        residual = hidden_states
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
        hidden_states = self.post_attention_layernorm(attn_outputs.attention_output)
        hidden_states = checkpoint_name(residual + hidden_states, "residual")

        residual = hidden_states
        feed_forward_input = self.pre_feedforward_layernorm(hidden_states)
        if self.config.use_scan_mlp:
            feed_forward_hidden_states = blockwise_ffn(
                self.mlp,
                feed_forward_input,
                self.config.scan_mlp_chunk_size,
            )
        else:
            feed_forward_hidden_states = self.mlp(feed_forward_input)
        hidden_states = self.post_feedforward_layernorm(feed_forward_hidden_states)
        hidden_states = checkpoint_name(residual + hidden_states, "residual")

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        return DecoderLayerOutput(
            hidden_states=checkpoint_name(hidden_states, "layer_output"),
            attention_weight=attn_outputs.attention_weight,
            cache_view=attn_outputs.cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=MuseGlimmerTextConfig, model_type="muse_glimmer")
class MuseGlimmerTextModel(EasyDeLBaseModule):
    """Muse-Glimmer language-model trunk (embeddings, decoder stack, final norm).

    The token embedding is followed by a scale-less RMS norm (HF's
    ``MuseGlimmerTextNormedEmbedding``); this is exposed through
    :meth:`embed_input_ids` so the multimodal wrapper can normalize text
    embeddings before splicing vision features into placeholder positions,
    exactly as the reference implementation does.
    """

    def __init__(
        self,
        config: MuseGlimmerTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ) -> None:
        """Initialize the language-model trunk.

        Args:
            config: Text decoder configuration.
            dtype: Compute dtype. Defaults to bfloat16.
            param_dtype: Parameter storage dtype. Defaults to bfloat16.
            precision: JAX matmul precision. Defaults to None.
            rngs: SpecTrax RNGs consumed by the sub-module initializers.
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
                num_embeddings=config.vocab_size,
                features=config.hidden_size,
                dtype=dtype,
                param_dtype=param_dtype,
                embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
                rngs=rngs,
            )

        remat_layer_block = auto_remat(
            MuseGlimmerTextDecoderLayer,
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

    @cached_property
    def frequencies(self):
        """Precomputed RoPE table for the global ``rope_theta``."""
        return self.config.get_basic_frequencies(
            head_size=self.config.head_dim,
            rotary_dim=self.config.head_dim,
            base=self.config.rope_theta,
        )

    @cached_property
    def rope_frequency_bank(self) -> tuple[Array, Array]:
        """Stacked per-layer RoPE tables plus the layer -> table selector.

        One table is built per distinct value in ``config.layer_rope_theta``,
        with ``0`` (NoPE) mapping to an identity rotation (``cos = 1``,
        ``sin = 0``) so those layers pass Q/K through untouched. Selecting the
        table by index — rather than branching on the layer in Python — keeps
        every decoder layer identical, which is what allows the stack to be
        scanned.

        Returns:
            tuple[Array, Array]: ``(tables, selector)`` where ``tables`` has
            shape ``(num_distinct, max_position_embeddings, head_dim)`` and
            ``selector`` has one entry per decoder layer.
        """
        head_dim = self.config.head_dim
        distinct: list[float] = []
        selector: list[int] = []
        for theta in self.config.layer_rope_theta:
            key = float(theta) if theta else 0.0
            if key not in distinct:
                distinct.append(key)
            selector.append(distinct.index(key))

        reference = jnp.asarray(getattr(self.frequencies, "value", self.frequencies))
        tables = []
        for theta in distinct:
            if theta == 0.0:
                half = reference.shape[-1] // 2
                identity = jnp.concatenate(
                    [
                        jnp.ones((reference.shape[0], half), dtype=reference.dtype),
                        jnp.zeros((reference.shape[0], reference.shape[-1] - half), dtype=reference.dtype),
                    ],
                    axis=-1,
                )
                tables.append(identity)
            elif theta == float(self.config.rope_theta):
                tables.append(reference)
            else:
                table = self.config.get_basic_frequencies(
                    head_size=head_dim,
                    rotary_dim=head_dim,
                    base=theta,
                )
                tables.append(jnp.asarray(getattr(table, "value", table)))
        return jnp.stack(tables), jnp.asarray(selector, dtype=jnp.int32)

    def embed_input_ids(self, input_ids: Int[Array, "batch seq_len"]) -> Array:
        """Look up token embeddings and apply the scale-less embedding norm.

        Args:
            input_ids: Token ids of shape ``(batch, seq_len)``.

        Returns:
            Array: Normalized embeddings of shape ``(batch, seq_len, hidden_size)``.
        """
        embeddings = self.embed_tokens(input_ids.astype("i4"))
        return checkpoint_name(_scaleless_rms_norm(embeddings, self.config.rms_norm_eps), "embeddings")

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
        trace: bool = False,
    ) -> BaseModelOutput:
        """Run the decoder stack.

        Args:
            input_ids: Token ids of shape ``(batch, seq_len)``. Mutually
                exclusive with ``inputs_embeds``.
            inputs_embeds: Pre-computed (already normalized) embeddings.
            attention_mask: Padding mask of shape ``(batch, seq_len)``.
            mask_info: Structured mask information.
            position_ids: Absolute token positions.
            mode: Runtime mode; inferred when None.
            past_key_values: KV cache for generation.
            cache_metadata: Cache metadata.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return per-layer hidden states.
            trace: Force the traced (unrolled) layer-scan path.

        Returns:
            BaseModelOutput: Last hidden state, optional hidden states and
            attentions, and the updated cache.

        Raises:
            ValueError: If neither or both of ``input_ids`` / ``inputs_embeds`` are given.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)

        sequence_length = inputs_embeds.shape[1]
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if sequence_length > self.config.max_position_embeddings:
            raise ValueError(
                "Maximum Position Embedding Reached! "
                f"(Expected <= {self.config.max_position_embeddings} got {sequence_length})"
            )

        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        attention_mask = token_attention_mask_from_mask_info(mask_info, sequence_length)

        if position_ids is None:
            position_ids = mask_info.q_position_ids

        hidden_states = apply_logical_sharding(
            inputs_embeds,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )

        if mode is None:
            mode = (
                common_types.MODE_DECODE
                if sequence_length == 1 and past_key_values is not None
                else common_types.MODE_TRAIN
            )

        if past_key_values is None:
            past_key_values = TransformerCache.init_empty(len(self.layers))

        views = past_key_values.views if past_key_values is not None else None
        has_cache_views = views is not None and any(v is not None for v in views)
        needs_trace_cache = mode == common_types.MODE_DECODE or has_cache_views

        trace_layers = self._layer_scan_trace(
            trace,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            cache_views=views,
            extra=needs_trace_cache,
        )
        cache_views = views if trace_layers else None
        frequency_tables, frequency_selector = self.rope_frequency_bank

        def _run_layer(block, carry):
            """Apply one decoder layer inside the layer-stack scan."""
            hs, cv, ah, aa, idx = carry
            if output_hidden_states:
                ah = (*ah, hs)
            # Each layer's RoPE table is gathered by index, so the layer bodies
            # stay identical and NoPE layers simply pick the identity table.
            layer_frequencies = frequency_tables[frequency_selector[idx]]
            with self._layer_stage_context(idx, layers=self.layers):
                layer_outputs = block(
                    hidden_states=hs,
                    mask_info=mask_info,
                    position_ids=position_ids,
                    mode=mode,
                    cache_view=self._layer_cache_view_at(cv, idx, enabled=trace_layers, cache=past_key_values),
                    cache_metadata=cache_metadata,
                    output_attentions=output_attentions,
                    frequencies=layer_frequencies,
                )
            hs = layer_outputs.hidden_states
            hs = self._mark_layer_stage_boundary(hs, idx, layers=self.layers)
            cv = self._layer_cache_view_update(
                cv,
                idx,
                layer_outputs.cache_view,
                enabled=trace_layers,
                cache=past_key_values,
            )
            if output_attentions:
                aa = (*aa, layer_outputs.attention_weight)
            return hs, cv, ah, aa, idx + 1

        init_carry = (hidden_states, cache_views, all_hidden_states, all_attentions, 0)
        hidden_states, _, all_hidden_states, all_attentions, _ = self.layers.scan(
            _run_layer,
            init_carry,
            trace=trace_layers,
        )

        hidden_states = self.norm(hidden_states)
        hidden_states = checkpoint_name(hidden_states, "model_output")
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            past_key_values=past_key_values,
        )

    def get_encoder(self):
        """Raise — the text trunk has no encoder.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Text model does not have an encoder.")

    def get_decoder(self):
        """Return the decoder (this module)."""
        return self

    def get_lm_head(self):
        """Raise — the trunk owns no LM head.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Base model does not have a language model head.")

    def get_embedding(self):
        """Return the token embedding table."""
        return self.embed_tokens


class MuseGlimmerVisionMLP(spx.Module):
    """Two-layer feed-forward block of a vision encoder layer."""

    def __init__(
        self,
        config: MuseGlimmerVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ) -> None:
        """Initialize the vision FFN.

        Args:
            config: Vision tower configuration.
            dtype: Compute dtype. Defaults to bfloat16.
            param_dtype: Parameter storage dtype. Defaults to bfloat16.
            precision: JAX matmul precision. Defaults to None.
            rngs: SpecTrax RNGs consumed by the projection initializers.
        """
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: Array) -> Array:
        """Apply ``fc2(act(fc1(x)))``.

        Args:
            hidden_states: Input of shape ``(seq_len, hidden_size)``.

        Returns:
            Array: Output of shape ``(seq_len, hidden_size)``.
        """
        return self.fc2(self.act(self.fc1(hidden_states)))


class MuseGlimmerVisionAttention(UnifiedAttention):
    """Bidirectional attention over packed vision patches.

    Attention is confined to the segments described by ``cu_seqlens``, which the
    tower supplies per layer: whole-frame boundaries for full-attention layers,
    per-window boundaries for window-attention layers.
    """

    def __init__(
        self,
        config: MuseGlimmerVisionConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ) -> None:
        """Initialize a vision attention block.

        Args:
            config: Vision tower configuration.
            layer_idx: Index of the owning encoder layer.
            dtype: Compute dtype. Defaults to bfloat16.
            param_dtype: Parameter storage dtype. Defaults to bfloat16.
            precision: JAX matmul precision. Defaults to None.
            rngs: SpecTrax RNGs consumed by the sub-module initializers.
        """
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.layer_idx = layer_idx
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="standard",
            causal=False,
            use_gqa=False,
        )

    def define_network(
        self,
        config: MuseGlimmerVisionConfig,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        precision: jax.lax.PrecisionLike,
        rngs: spx.Rngs,
    ) -> None:
        """Build the fused QKV projection, the output projection and the performer.

        Args:
            config: Vision tower configuration.
            dtype: Compute dtype.
            param_dtype: Parameter storage dtype.
            precision: JAX matmul precision.
            rngs: SpecTrax RNGs consumed by the sub-module initializers.
        """
        layout = FusedColumnLayout(
            segments=(
                FusedSegment("q", self.hidden_size, "q_proj"),
                FusedSegment("k", self.hidden_size, "k_proj"),
                FusedSegment("v", self.hidden_size, "v_proj"),
            ),
            log_label="Muse-Glimmer vision Q/K/V groups",
        )
        self.qkv_proj = ColumnParallelLinear(
            self.hidden_size,
            layout.out_features,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
            layout=layout,
        )
        self.qkv_proj.build_reform_param("qkv_proj", config=config, include_bias=True)
        self.proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )
        self.rotary = None
        self.attention_performer = self._create_attention_performer(config, rngs)

    @property
    def reform_param(self):
        """Checkpoint reform rules fusing HF's separate vision Q/K/V tensors."""
        return self.qkv_proj.build_reform_param("qkv_proj", config=self.config, include_bias=True)

    def _create_attention_performer(self, config, rngs: spx.Rngs) -> FlexibleAttentionModule:
        """Build the cache-free bidirectional attention performer.

        Args:
            config: Vision tower configuration.
            rngs: SpecTrax RNGs.

        Returns:
            FlexibleAttentionModule: Performer configured for the vision tower.
        """
        return FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=0.0,
            attn_mechanism="vanilla",
            requires_cache=False,
        )

    def forward(  # type: ignore[override]
        self,
        hidden_states: Array,
        attention_bias: Array,
        position_embeddings: tuple[Array, Array],
    ) -> Array:
        """Run bidirectional attention over packed patches.

        Args:
            hidden_states: Packed patches of shape ``(seq_len, hidden_size)``.
            attention_bias: Additive block-diagonal bias of shape
                ``(1, 1, seq_len, seq_len)``.
            position_embeddings: ``(cos, sin)`` tables of shape ``(seq_len, head_dim)``.

        Returns:
            Array: Attention output of shape ``(seq_len, hidden_size)``.
        """
        seq_length = hidden_states.shape[0]
        qkv = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = self.qkv_proj.split(qkv, config=self.config)

        head_shape = (1, seq_length, self.num_heads, self.head_dim)
        query_states = query_states.reshape(head_shape)
        key_states = key_states.reshape(head_shape)
        value_states = value_states.reshape(head_shape)

        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        # The performer expects a per-head bias; the segment mask is head-invariant.
        attention_bias = jnp.broadcast_to(attention_bias, (1, self.num_heads, seq_length, seq_length))

        attn_output = self.attention_performer.forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            bias=attention_bias,
            causal=False,
            mode=common_types.MODE_TRAIN,
        ).attention_outputs

        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.shard_attention_prod(attn_output, pre_projection=True)
        attn_output = checkpoint_name(self.proj(attn_output), "vision_attn_output")
        return self.shard_attention_prod(attn_output)


class MuseGlimmerVisionEncoderLayer(spx.Module):
    """Pre-norm vision encoder block."""

    def __init__(
        self,
        config: MuseGlimmerVisionConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ) -> None:
        """Initialize a vision encoder block.

        Args:
            config: Vision tower configuration.
            layer_idx: Index of this block.
            dtype: Compute dtype. Defaults to bfloat16.
            param_dtype: Parameter storage dtype. Defaults to bfloat16.
            precision: JAX matmul precision. Defaults to None.
            rngs: SpecTrax RNGs consumed by the sub-module initializers.
        """
        self.layer_idx = layer_idx
        # The reference block norms use a literal 1e-5, independent of the
        # `layer_norm_eps` that governs `ln_pre` / `ln_post`.
        self.norm1 = LayerNorm(config.hidden_size, epsilon=1e-5, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.norm2 = LayerNorm(config.hidden_size, epsilon=1e-5, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.attn = MuseGlimmerVisionAttention(
            config=config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.mlp = MuseGlimmerVisionMLP(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def forward(
        self,
        hidden_states: Array,
        attention_bias: Array,
        position_embeddings: tuple[Array, Array],
    ) -> Array:
        """Run attention and FFN with residual connections.

        Args:
            hidden_states: Packed patches of shape ``(seq_len, hidden_size)``.
            attention_bias: Additive block-diagonal attention bias.
            position_embeddings: ``(cos, sin)`` rotary tables.

        Returns:
            Array: Updated patches of shape ``(seq_len, hidden_size)``.
        """
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class MuseGlimmerVisionPatchEmbedder(spx.Module):
    """Linear patch embedding plus a resampled learned position grid."""

    def __init__(
        self,
        config: MuseGlimmerVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ) -> None:
        """Initialize the patch embedder.

        Args:
            config: Vision tower configuration.
            dtype: Compute dtype. Defaults to bfloat16.
            param_dtype: Parameter storage dtype. Defaults to bfloat16.
            precision: JAX matmul precision. Defaults to None.
            rngs: SpecTrax RNGs consumed by the sub-module initializers.
        """
        self.config = config
        self.dtype = dtype
        self.num_grid_per_side = config.pos_emb_height
        patch_features = config.patch_temporal * config.in_channels * config.patch_size**2
        self.patch_embedding = ColumnParallelLinear(
            patch_features,
            config.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )
        self.position_embedding_table = Embed(
            num_embeddings=config.pos_emb_height * config.pos_emb_width,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            rngs=rngs,
        )

    def forward(self, pixel_values: Array, grid_thw: np.ndarray) -> Array:
        """Embed packed patches and add the resampled position embedding.

        Args:
            pixel_values: Flattened patches of shape
                ``(total_patches, patch_temporal * in_channels * patch_size**2)``.
            grid_thw: Host-side ``(num_images_or_videos, 3)`` patch-grid dims.

        Returns:
            Array: Patch embeddings of shape ``(total_patches, hidden_size)``.
        """
        embeddings = self.patch_embedding(pixel_values.astype(self.dtype))
        interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
            grid_thw,
            num_grid_per_side=self.num_grid_per_side,
            spatial_merge_size=1,
        )
        taps = self.position_embedding_table(jnp.asarray(interp_indices))
        pos_embeds = jnp.sum(taps * jnp.asarray(interp_weights)[:, :, None], axis=1)
        return embeddings + pos_embeds.astype(embeddings.dtype)


@register_module(TaskType.BASE_VISION, config=MuseGlimmerVisionConfig, model_type="muse_glimmer_vision")
class MuseGlimmerVisionModel(EasyDeLBaseModule):
    """Packed windowed ViT tower for Muse-Glimmer.

    Patches are permuted into window order once, run through the whole stack
    (each block seeing either window or whole-frame segment boundaries), then
    permuted back, layer-normalized, and pixel-shuffled by ``merge_size``.
    """

    config_class = MuseGlimmerVisionConfig

    def __init__(
        self,
        config: MuseGlimmerVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ) -> None:
        """Initialize the vision tower.

        Args:
            config: Vision tower configuration.
            dtype: Compute dtype. Defaults to bfloat16.
            param_dtype: Parameter storage dtype. Defaults to bfloat16.
            precision: JAX matmul precision. Defaults to None.
            rngs: SpecTrax RNGs consumed by the sub-module initializers.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.patch_embedder = MuseGlimmerVisionPatchEmbedder(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.ln_pre = LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.layers = nn.ModuleList([])
        for layer_idx in range(config.num_hidden_layers):
            with self.assign_layer_stage(layer_idx, total_layers=config.num_hidden_layers):
                self.layers.append(
                    MuseGlimmerVisionEncoderLayer(
                        config=config,
                        layer_idx=layer_idx,
                        dtype=dtype,
                        param_dtype=param_dtype,
                        precision=precision,
                        rngs=rngs,
                    )
                )
        self.ln_post = LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        # Window/position/interpolation math runs on unmerged patches; the merge
        # is deferred to `pixel_shuffle`, hence a spatial merge size of 1 here.
        self.spatial_merge_size = 1
        self.patch_size = config.patch_size
        self.window_size = config.window_size
        self.merge_size = config.merge_size

    def rotary_tables(self, position_ids: np.ndarray) -> tuple[Array, Array]:
        """Build the interleaved 2-D rotary tables for packed patches.

        Frequencies are computed independently for each spatial axis over
        ``head_dim // 2`` channels and interleaved as ``[w, h, w, h]``, matching
        the reference implementation.

        Args:
            position_ids: ``(total_patches, 2)`` host array of ``(w, h)`` indices.

        Returns:
            tuple[Array, Array]: ``(cos, sin)`` of shape ``(total_patches, head_dim)``.
        """
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        spatial_dim = head_dim // 2
        inv_freq = 1.0 / (self.config.rope_theta ** (np.arange(0, spatial_dim, 2, dtype=np.float32) / spatial_dim))
        freq_w = position_ids[:, 0].astype(np.float32)[:, None] * inv_freq[None, :]
        freq_h = position_ids[:, 1].astype(np.float32)[:, None] * inv_freq[None, :]
        freq = np.concatenate([freq_w, freq_h, freq_w, freq_h], axis=-1)
        return jnp.asarray(np.cos(freq), dtype=self.dtype), jnp.asarray(np.sin(freq), dtype=self.dtype)

    def pixel_shuffle(self, hidden_states: Array, grid_thw: np.ndarray) -> Array:
        """Fold ``merge_size x merge_size`` spatial blocks into the channel axis.

        Args:
            hidden_states: Patches of shape ``(total_patches, hidden_size)``.
            grid_thw: Host-side ``(num_images_or_videos, 3)`` patch-grid dims.

        Returns:
            Array: Merged tokens of shape
            ``(total_patches / merge_size**2, hidden_size * merge_size**2)``.
        """
        factor = self.merge_size
        dim = hidden_states.shape[-1]
        shuffle_index = jnp.asarray(get_vision_pixel_shuffle_index(grid_thw, factor))
        hidden_states = hidden_states[shuffle_index]
        hidden_states = hidden_states.reshape(-1, factor * factor, dim)
        return jnp.transpose(hidden_states, (0, 2, 1)).reshape(-1, dim * factor * factor)

    def forward(self, pixel_values: Array, grid_thw: Array | np.ndarray) -> Array:
        """Encode packed image/video patches into merged vision tokens.

        Args:
            pixel_values: Flattened patches of shape
                ``(total_patches, patch_temporal * in_channels * patch_size**2)``.
            grid_thw: ``(num_images_or_videos, 3)`` patch-grid dimensions. Values
                must be concrete (host-side), since window order and segment
                boundaries are derived from them.

        Returns:
            Array: Merged vision tokens of shape
            ``(total_patches / merge_size**2, hidden_size * merge_size**2)``.
        """
        grid = np.asarray(jax.device_get(grid_thw), dtype=np.int64).reshape(-1, 3)

        cu_seqlens = get_vision_cu_seqlens(grid)
        window_index, cu_window_seqlens = get_vision_window_index(
            grid,
            spatial_merge_size=self.spatial_merge_size,
            window_size=self.window_size,
            patch_size=self.patch_size,
        )

        hidden_states = self.patch_embedder(pixel_values, grid)
        hidden_states = self.ln_pre(hidden_states)
        window_index_device = jnp.asarray(window_index)
        hidden_states = hidden_states[window_index_device]

        # The reference implementation orders rotary positions as (w, h) and
        # offsets them by 1.
        position_ids = get_vision_position_ids(grid, spatial_merge_size=self.spatial_merge_size)
        position_ids = position_ids[:, ::-1] + 1
        position_embeddings = self.rotary_tables(position_ids[window_index])

        seq_length = hidden_states.shape[0]
        # Both bias variants are stacked so the per-layer choice becomes a gather
        # on the (possibly traced) layer index; that keeps the block signature
        # identical across layers, which is what lets the stack be scanned.
        bias_stack = jnp.stack(
            [
                _block_diagonal_bias(cu_seqlens, seq_length, hidden_states.dtype),
                _block_diagonal_bias(cu_window_seqlens, seq_length, hidden_states.dtype),
            ]
        )
        bias_selector = jnp.asarray(
            [0 if layer_type == "full_attention" else 1 for layer_type in self.config.layer_types],
            dtype=jnp.int32,
        )

        def _layer_loop(block, carry):
            """Apply one vision-encoder block inside the layer-stack scan."""
            hidden, layer_num = carry
            with self._layer_stage_context(layer_num, layers=self.layers):
                hidden = block(
                    hidden,
                    attention_bias=bias_stack[bias_selector[layer_num]],
                    position_embeddings=position_embeddings,
                )
            hidden = self._mark_layer_stage_boundary(hidden, layer_num, layers=self.layers)
            return hidden, layer_num + 1

        hidden_states, _ = self.layers.scan(
            _layer_loop,
            (hidden_states, 0),
            trace=not self.config.scan_layers or self._pipeline_stage_count() > 1,
        )

        reverse_indices = jnp.asarray(np.argsort(window_index))
        hidden_states = hidden_states[reverse_indices]
        hidden_states = self.ln_post(hidden_states)
        return self.pixel_shuffle(hidden_states, grid)

    def get_encoder(self):
        """Return the encoder (this module)."""
        return self

    def get_decoder(self):
        """Raise — the vision tower has no decoder.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Vision model does not have a decoder.")

    def get_lm_head(self):
        """Raise — the vision tower has no LM head.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Vision model does not have a language model head.")

    def get_embedding(self):
        """Return the patch embedder."""
        return self.patch_embedder


class MuseGlimmerVisionAdapter(spx.Module):
    """Two-layer bias-free adapter between the vision tower and the projection."""

    def __init__(
        self,
        config: MuseGlimmerConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ) -> None:
        """Initialize the vision adapter.

        Args:
            config: Composite Muse-Glimmer configuration.
            dtype: Compute dtype. Defaults to bfloat16.
            param_dtype: Parameter storage dtype. Defaults to bfloat16.
            precision: JAX matmul precision. Defaults to None.
            rngs: SpecTrax RNGs consumed by the projection initializers.
        """
        self.fc1 = ColumnParallelLinear(
            config.out_hidden_size,
            config.projector_hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.fc2 = RowParallelLinear(
            config.projector_hidden_size,
            config.projector_hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.act = ACT2FN[config.projector_hidden_act]

    def forward(self, hidden_states: Array) -> Array:
        """Apply ``act(fc2(act(fc1(x))))``.

        Args:
            hidden_states: Vision tokens of shape ``(num_tokens, out_hidden_size)``.

        Returns:
            Array: Adapted tokens of shape ``(num_tokens, projector_hidden_size)``.
        """
        return self.act(self.fc2(self.act(self.fc1(hidden_states))))


@register_module(TaskType.VISION_LM, config=MuseGlimmerConfig, model_type="muse_glimmer")
class MuseGlimmerModel(EasyDeLBaseModule):
    """Muse-Glimmer trunk: vision tower + adapter/projection + language model.

    Vision features are adapted, projected into the text hidden size,
    RMS-normalized (scale-less ``perception_emb_norm``) and scattered into the
    normalized text embedding sequence at ``image_token_id`` / ``video_token_id``
    positions before the decoder runs.
    """

    def __init__(
        self,
        config: MuseGlimmerConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ) -> None:
        """Initialize the multimodal trunk.

        Args:
            config: Composite Muse-Glimmer configuration.
            dtype: Compute dtype. Defaults to bfloat16.
            param_dtype: Parameter storage dtype. Defaults to bfloat16.
            precision: JAX matmul precision. Defaults to None.
            rngs: SpecTrax RNGs consumed by the sub-module initializers.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.vision_tower = MuseGlimmerVisionModel(
            config.vision_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.language_model = MuseGlimmerTextModel(
            config.text_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.vision_adapter = MuseGlimmerVisionAdapter(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.vision_projection = ColumnParallelLinear(
            config.projector_hidden_size,
            config.text_config.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def get_input_embeddings(self):
        """Return the language model's token embedding table."""
        return self.language_model.get_embedding()

    def set_input_embeddings(self, value):
        """Replace the language model's token embedding table.

        Args:
            value: New embedding module.
        """
        self.language_model.embed_tokens = value

    def get_image_features(self, pixel_values: Array, image_grid_thw: Array | np.ndarray) -> Array:
        """Encode packed image patches into text-space embeddings.

        Args:
            pixel_values: Flattened patches of shape ``(total_patches, patch_features)``.
            image_grid_thw: ``(num_images, 3)`` patch-grid dimensions.

        Returns:
            Array: Vision embeddings of shape ``(num_visual_tokens, text_hidden_size)``.
        """
        vision_features = self.vision_tower(pixel_values=pixel_values, grid_thw=image_grid_thw)
        vision_features = self.vision_adapter(vision_features)
        vision_features = self.vision_projection(vision_features)
        return _scaleless_rms_norm(vision_features, self.config.text_config.rms_norm_eps)

    def get_video_features(self, pixel_values_videos: Array, video_grid_thw: Array | np.ndarray) -> Array:
        """Encode packed video patches into text-space embeddings.

        Videos share the image path entirely; the temporal axis is already
        folded into ``grid_thw``.

        Args:
            pixel_values_videos: Flattened patches of shape ``(total_patches, patch_features)``.
            video_grid_thw: ``(num_videos, 3)`` patch-grid dimensions.

        Returns:
            Array: Vision embeddings of shape ``(num_visual_tokens, text_hidden_size)``.
        """
        return self.get_image_features(pixel_values_videos, video_grid_thw)

    def compute_embedding(
        self,
        input_ids: Int[Array, "batch seq_len"] | None,
        *,
        image_features: Array | None = None,
        video_features: Array | None = None,
        **kwargs,
    ) -> Array:
        """Embed tokens and splice vision features into placeholder positions.

        Placeholder ids are replaced by ``0`` before the lookup (they may sit
        outside the embedding table's trained range), matching the reference
        implementation, and the normalized vision features are then scattered
        back into those positions.

        Args:
            input_ids: Token ids of shape ``(batch, seq_len)``.
            image_features: Projected image embeddings, when images are present.
            video_features: Projected video embeddings, when videos are present.
            **kwargs: Ignored extra keyword arguments.

        Returns:
            Array: Embeddings of shape ``(batch, seq_len, text_hidden_size)``.

        Raises:
            ValueError: If ``input_ids`` is None.
        """
        if input_ids is None:
            raise ValueError("`input_ids` must be provided when calling `compute_embedding`.")

        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        multimodal_mask = (input_ids == image_token_id) | (input_ids == video_token_id)
        llm_input_ids = jnp.where(multimodal_mask, 0, input_ids)
        inputs_embeds = self.language_model.embed_input_ids(llm_input_ids)

        for features, token_id in ((image_features, image_token_id), (video_features, video_token_id)):
            if features is None:
                continue
            merged = features.reshape(-1, features.shape[-1]).astype(inputs_embeds.dtype)
            inputs_embeds = BaseVisionLanguageModule.merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=merged,
                placeholder_token_id=token_id,
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        pixel_values: Array | None = None,
        image_grid_thw: Array | None = None,
        pixel_values_videos: Array | None = None,
        video_grid_thw: Array | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **lm_kwargs,
    ) -> MuseGlimmerModelOutputWithPast:
        """Run the multimodal trunk.

        Args:
            input_ids: Token ids of shape ``(batch, seq_len)``.
            pixel_values: Packed image patches.
            image_grid_thw: ``(num_images, 3)`` image patch-grid dimensions.
            pixel_values_videos: Packed video patches.
            video_grid_thw: ``(num_videos, 3)`` video patch-grid dimensions.
            attention_mask: Padding mask of shape ``(batch, seq_len)``.
            mask_info: Structured mask information.
            position_ids: Absolute token positions.
            mode: Runtime mode hint.
            past_key_values: KV cache for generation.
            cache_metadata: Cache metadata.
            inputs_embeds: Pre-computed embeddings (skips the vision path).
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return per-layer hidden states.
            **lm_kwargs: Extra arguments forwarded to the language model.

        Returns:
            MuseGlimmerModelOutputWithPast: Hidden states, cache and the merged
            vision embeddings.

        Raises:
            ValueError: If neither or both of ``input_ids`` / ``inputs_embeds``
                are given, or if pixels are supplied without ``input_ids``.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        has_pixels = pixel_values is not None or pixel_values_videos is not None
        if has_pixels and input_ids is None:
            raise ValueError("`input_ids` must be provided when pixel inputs are not None.")

        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values, image_grid_thw)
        video_features = None
        if pixel_values_videos is not None:
            video_features = self.get_video_features(pixel_values_videos, video_grid_thw)

        if inputs_embeds is None:
            inputs_embeds = self.compute_embedding(
                input_ids,
                image_features=image_features,
                video_features=video_features,
            )

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **lm_kwargs,
        )

        visual_features = image_features if image_features is not None else video_features
        return MuseGlimmerModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=visual_features,
        )

    def init_cache(self, batch_size, max_length, starts=None, shardings=None, pad_token_id=None):
        """Initialize the language model's KV cache.

        Args:
            batch_size: Batch size for the cache.
            max_length: Maximum cached sequence length.
            starts: Optional starting positions.
            shardings: Optional sharding specifications.
            pad_token_id: Optional padding token id.

        Returns:
            The initialized cache.
        """
        return self.language_model.init_cache(batch_size, max_length, starts, shardings, pad_token_id)

    def prepare_inputs_for_generation(
        self,
        input_ids: Int[Array, "batch seq_len"],
        max_length: int,
        pad_token_id: int,
        starts: int | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        **kwargs,
    ):
        """Prepare inputs for autoregressive generation.

        Delegates to the language model; the multimodal wrapper re-attaches the
        vision inputs (pixels and ``grid_thw`` tables) for the first step and
        drops them afterwards, so they are intentionally ignored here.

        Args:
            input_ids: Prompt token ids of shape ``(batch, seq_len)``.
            max_length: Maximum generation length.
            pad_token_id: Padding token id.
            starts: Optional starting positions.
            attention_mask: Optional padding mask.
            **kwargs: Vision inputs and other extras, ignored at this level.

        Returns:
            dict: Model inputs ready for generation.
        """
        return self.language_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            max_length=max_length,
            pad_token_id=pad_token_id,
            starts=starts,
            attention_mask=attention_mask,
        )

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        """Update model inputs for the next generation step.

        Args:
            model_outputs: Outputs from the previous generation step.
            model_kwargs: Current model kwargs.

        Returns:
            dict: Updated model kwargs.
        """
        return self.language_model.update_inputs_for_generation(model_outputs, model_kwargs)

    def get_encoder(self):
        """Return the vision tower."""
        return self.vision_tower

    def get_decoder(self):
        """Return the language model trunk."""
        return self.language_model

    def get_lm_head(self):
        """Raise — the trunk owns no LM head.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("The base model does not have a language model head.")

    def get_embedding(self):
        """Return the language model's token embedding table."""
        return self.language_model.get_embedding()


@register_module(TaskType.IMAGE_TEXT_TO_TEXT, config=MuseGlimmerConfig, model_type="muse_glimmer")
class MuseGlimmerForConditionalGeneration(BaseVisionLanguageModule[MuseGlimmerModel, MuseGlimmerConfig]):  # type: ignore
    """Muse-Glimmer with the LM head, for image/video-conditioned generation.

    Logits are scaled by ``output_multiplier`` and then tanh soft-capped at
    ``final_logit_softcapping``, so the head emits
    ``T * tanh(logits * multiplier / T)``.
    """

    _task_type = TaskType.IMAGE_TEXT_TO_TEXT
    _model_type = "muse_glimmer"
    _config_class = MuseGlimmerConfig
    _auto_register = False  # Already registered via decorator
    _supports_video = True
    _uses_mrope = False

    _vision_tower_name = "vision_tower"
    _projector_name = "vision_adapter"
    _language_model_name = "language_model"

    loss_type = "ForCausalLM"

    def __init__(
        self,
        config: MuseGlimmerConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ) -> None:
        """Initialize the conditional-generation wrapper.

        Args:
            config: Composite Muse-Glimmer configuration.
            dtype: Compute dtype. Defaults to bfloat16.
            param_dtype: Parameter storage dtype. Defaults to bfloat16.
            precision: JAX matmul precision. Defaults to None.
            rngs: SpecTrax RNGs consumed by the sub-module initializers.
        """
        super().__init__(
            config=config,
            base_model_class=MuseGlimmerModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            image_token_index=config.image_token_id,
            video_token_index=config.video_token_id,
            spatial_merge_size=config.vision_config.merge_size,
            temporal_patch_size=config.vision_config.patch_temporal,
            tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
            lm_head_bias=False,
        )

    def get_image_features(self, pixel_values: Array, image_grid_thw: Array | None = None, **kwargs) -> Array:
        """Encode packed image patches into text-space embeddings.

        Args:
            pixel_values: Flattened patches of shape ``(total_patches, patch_features)``.
            image_grid_thw: ``(num_images, 3)`` patch-grid dimensions.
            **kwargs: Ignored extra keyword arguments.

        Returns:
            Array: Vision embeddings of shape ``(num_visual_tokens, text_hidden_size)``.
        """
        return self.base_model.get_image_features(pixel_values, image_grid_thw)

    def get_video_features(
        self,
        pixel_values_videos: Array,
        video_grid_thw: Array | None = None,
        **kwargs,
    ) -> Array:
        """Encode packed video patches into text-space embeddings.

        Args:
            pixel_values_videos: Flattened patches of shape ``(total_patches, patch_features)``.
            video_grid_thw: ``(num_videos, 3)`` patch-grid dimensions.
            **kwargs: Ignored extra keyword arguments.

        Returns:
            Array: Vision embeddings of shape ``(num_visual_tokens, text_hidden_size)``.
        """
        return self.base_model.get_video_features(pixel_values_videos, video_grid_thw)

    def compute_embedding(self, input_ids, *args, **kwargs):
        """Embed tokens and splice vision features into placeholder positions.

        Args:
            input_ids: Token ids of shape ``(batch, seq_len)``.
            *args: Forwarded to the base model.
            **kwargs: Forwarded to the base model.

        Returns:
            Array: Merged embeddings.
        """
        return self.base_model.compute_embedding(input_ids, *args, **kwargs)

    def _softcap(self, logits: Array) -> Array:
        """Scale logits by ``output_multiplier`` and tanh soft-cap them.

        Args:
            logits: Raw LM-head outputs.

        Returns:
            Array: ``T * tanh(logits * multiplier / T)``, or just the scaled
            logits when soft-capping is disabled.
        """
        text_config = self.config.get_text_config()
        logits = logits * jnp.asarray(text_config.output_multiplier, dtype=logits.dtype)
        cap_value = text_config.final_logit_softcapping
        if cap_value is None:
            return logits
        cap = jnp.asarray(cap_value, dtype=logits.dtype)
        return cap * jax.nn.tanh(logits / cap)

    def compute_lm_logits(self, hidden_states: Array) -> Array:
        """Project hidden states to vocabulary logits with Muse-Glimmer capping.

        Args:
            hidden_states: Hidden states of shape ``(batch, seq_len, hidden_size)``.

        Returns:
            Array: Scaled, soft-capped logits of shape ``(batch, seq_len, vocab_size)``.
        """
        return self._softcap(super().compute_lm_logits(hidden_states))

    def make_lm_head_fn(self, vocab_shard_stage: int | None = None):
        """Build a trace-safe LM-head projection that applies the same capping.

        Args:
            vocab_shard_stage: Optional pipeline-stage hint forwarded to the base helper.

        Returns:
            Callable[[Array], Array]: Hidden states -> scaled, soft-capped logits.
        """
        base_fn = super().make_lm_head_fn(vocab_shard_stage=vocab_shard_stage)

        def _project(hidden_states):
            """Apply the base LM head then Muse-Glimmer's scale + tanh soft-cap."""
            return self._softcap(base_fn(hidden_states))

        return _project

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        pixel_values: Array | None = None,
        image_grid_thw: Array | None = None,
        pixel_values_videos: Array | None = None,
        video_grid_thw: Array | None = None,
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
        **lm_kwargs,
    ) -> VLMCausalLMOutput:
        """Run the multimodal model and (optionally) the LM head.

        Args:
            input_ids: Token ids of shape ``(batch, seq_len)``.
            pixel_values: Packed image patches.
            image_grid_thw: ``(num_images, 3)`` image patch-grid dimensions.
            pixel_values_videos: Packed video patches.
            video_grid_thw: ``(num_videos, 3)`` video patch-grid dimensions.
            attention_mask: Padding mask of shape ``(batch, seq_len)``.
            mask_info: Structured mask information.
            position_ids: Absolute token positions.
            mode: Runtime mode hint.
            past_key_values: KV cache for generation.
            cache_metadata: Cache metadata.
            apply_lm_head: Whether to compute vocabulary logits. Defaults to True.
            inputs_embeds: Pre-computed embeddings.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return per-layer hidden states.
            **lm_kwargs: Extra arguments forwarded to the language model.

        Returns:
            VLMCausalLMOutput: Logits (when requested), cache, hidden states,
            attentions and the merged vision embeddings.

        Raises:
            ValueError: If neither or both of ``input_ids`` / ``inputs_embeds`` are given.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.base_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
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
            **lm_kwargs,
        )

        hidden_states = apply_logical_sharding(
            outputs.last_hidden_state,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )

        lm_logits = None
        if apply_lm_head:
            lm_logits = self.compute_lm_logits(hidden_states)

        return VLMCausalLMOutput(
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            last_hidden_state=hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def init_cache(self, batch_size, max_length, starts=None, shardings=None, pad_token_id=None):
        """Initialize the KV cache for autoregressive generation.

        Args:
            batch_size: Batch size for the cache.
            max_length: Maximum cached sequence length.
            starts: Optional starting positions.
            shardings: Optional sharding specifications.
            pad_token_id: Optional padding token id.

        Returns:
            The initialized cache.
        """
        return self.base_model.init_cache(batch_size, max_length, starts, shardings, pad_token_id)

    def get_vision_tower(self) -> spx.Module:
        """Return the vision tower."""
        return self.base_model.vision_tower

    def get_projector(self) -> spx.Module:
        """Return the vision adapter."""
        return self.base_model.vision_adapter

    def get_language_model(self) -> spx.Module:
        """Return the language model trunk."""
        return self.base_model.language_model


__all__: tp.Sequence[str] = (
    "MuseGlimmerForConditionalGeneration",
    "MuseGlimmerModel",
    "MuseGlimmerTextModel",
    "MuseGlimmerVisionModel",
)
