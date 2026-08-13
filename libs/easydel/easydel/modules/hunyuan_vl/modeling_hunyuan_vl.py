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

"""Tencent HunYuanVL multimodal model implementation for EasyDeL.

HunYuanVL (the open-source, dense, image-text variant behind
``tencent/HunyuanOCR``) pairs a SigLIP-style vision tower with a HunYuan
dense decoder:

- Vision tower: strided ``Conv2d`` patch embedding plus a learned position
  grid bilinearly interpolated to each image's patch grid, pre-LayerNorm
  encoder blocks with biased q/k/v/o projections (no rotary embeddings, one
  block-diagonal segment per image), and a convolutional patch merger that
  downsamples by ``spatial_merge_size``, appends a newline token per merged
  row, projects into the text hidden size, and wraps every image in learned
  begin/end tokens.
- Text decoder: grouped-query attention where rotary embeddings are applied
  *before* the per-head RMSNorms on queries/keys, a SwiGLU MLP, and RMSNorm
  pre-normalization. Positions use HunYuan's 3-axis multimodal RoPE:
  ``mrope_section`` partitions half of each head across
  ``(width, height, image_index)`` and each doubled section is taken as one
  contiguous chunk from its axis.

Components:
- :class:`HunYuanVLVisionModel`: the vision tower (patch embedding, blocks,
  patch merger).
- :class:`HunYuanVLTextModel`: the language-model trunk.
- :class:`HunYuanVLModel`: the composite base model merging visual and text
  embeddings and computing multimodal RoPE indices.
- :class:`HunYuanVLForConditionalGeneration`: the IMAGE_TEXT_TO_TEXT task
  head with a (tied) LM head.
"""

import itertools
import typing as tp
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
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
    AttentionLayerOutput,
    BaseModelOutput,
    DecoderLayerOutput,
    EmbeddingInfo,
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
from easydel.layers.norms import LayerNorm
from easydel.modules._base import BaseVisionLanguageModule

from .hunyuan_vl_configuration import HunYuanVLConfig, HunYuanVLTextConfig, HunYuanVLVisionConfig


def get_rope_index(
    input_ids: np.ndarray,
    image_grid_thw: np.ndarray | None = None,
    attention_mask: np.ndarray | None = None,
    mm_token_type_ids: np.ndarray | None = None,
    spatial_merge_size: int = 2,
    image_token_id: int = -1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build HunYuanVL 3-axis multimodal RoPE position ids.

    All three axes default to plain 1-D text positions. Inside every image
    span the last three axes are overwritten with ``(width, height,
    image_index)`` positions computed over the pooled visual grid: the patch
    merger appends one newline token per merged row, so the width channel
    spans ``llm_w + 1`` columns. When the span includes the merger's
    begin/end tokens (``span_length == grid_tokens + 2``) those two tokens
    keep their default text positions. ``image_index`` is the global ordinal
    of the image across the whole batch, mirroring the HF reference.

    Note:
        This helper runs in NumPy with data-dependent control flow and is
        NOT JIT-compatible; call it outside traced code.

    Args:
        input_ids: Token ids of shape ``(batch, seq_len)``.
        image_grid_thw: Per-image ``(t, h, w)`` patch grid of shape
            ``(num_images, 3)``; ``None`` for text-only inputs.
        attention_mask: Optional mask of shape ``(batch, seq_len)`` where 1
            marks valid tokens.
        mm_token_type_ids: Optional per-token modality types (1 = image
            span). When ``None``, spans are derived from contiguous runs of
            ``image_token_id`` in ``input_ids``.
        spatial_merge_size: Spatial merge factor of the vision tower.
        image_token_id: Placeholder token id used for image content.

    Returns:
        tuple: ``(position_ids, rope_deltas)`` with shapes
        ``(3, batch, seq_len)`` and ``(batch, 1)``.

    Raises:
        ValueError: If the number of image spans does not match
            ``image_grid_thw``, or a span length is inconsistent with its
            grid.
    """
    input_ids = np.asarray(input_ids)
    batch_size, seq_length = input_ids.shape
    position_ids = np.zeros((3, batch_size, seq_length), dtype=np.int32)
    if attention_mask is not None:
        attention_mask = np.asarray(attention_mask)[:, :seq_length]
    if mm_token_type_ids is not None:
        mm_token_type_ids = np.asarray(mm_token_type_ids)

    grids = np.asarray(image_grid_thw).reshape(-1, 3) if image_grid_thw is not None else None
    grid_cursor = 0
    image_index = 0
    rope_deltas: list[int] = []

    for batch_idx in range(batch_size):
        valid_mask = (
            attention_mask[batch_idx].astype(bool) if attention_mask is not None else np.ones(seq_length, dtype=bool)
        )
        current_ids = input_ids[batch_idx][valid_mask]
        if mm_token_type_ids is not None:
            span_flags = mm_token_type_ids[batch_idx][valid_mask] == 1
        else:
            span_flags = current_ids == image_token_id

        current_positions = np.tile(np.arange(current_ids.shape[0], dtype=np.int32), (3, 1))

        if grids is not None:
            flags = np.concatenate([[False], span_flags, [False]])
            edges = np.flatnonzero(flags[1:] != flags[:-1])
            for span_start, span_end in zip(edges[0::2], edges[1::2], strict=True):
                if grid_cursor >= grids.shape[0]:
                    raise ValueError("Found more image placeholder spans than entries in `image_grid_thw`.")
                _t, grid_h, grid_w = (int(v) for v in grids[grid_cursor])
                grid_cursor += 1

                llm_grid_h = grid_h // spatial_merge_size
                llm_grid_w = grid_w // spatial_merge_size
                # The merger emits one newline token per merged row, so the
                # width channel spans `llm_grid_w + 1` columns.
                height_pos, width_pos = np.meshgrid(
                    np.arange(llm_grid_h, dtype=np.int32),
                    np.arange(llm_grid_w + 1, dtype=np.int32),
                    indexing="ij",
                )
                grid_tokens = llm_grid_h * (llm_grid_w + 1)
                span_length = int(span_end - span_start)
                if span_length == grid_tokens + 2:
                    grid_start = int(span_start) + 1
                elif span_length == grid_tokens:
                    grid_start = int(span_start)
                else:
                    raise ValueError(
                        "Image placeholder span length does not match `image_grid_thw`: "
                        f"span_length={span_length}, expected {grid_tokens} or {grid_tokens + 2}."
                    )
                grid_end = grid_start + grid_tokens
                current_positions[0, grid_start:grid_end] = width_pos.flatten()
                current_positions[1, grid_start:grid_end] = height_pos.flatten()
                current_positions[2, grid_start:grid_end] = image_index
                image_index += 1

        position_ids[:, batch_idx, valid_mask] = current_positions
        rope_deltas.append(int(current_positions.max()) + 1 - int(current_ids.shape[0]))

    if grids is not None and grid_cursor != grids.shape[0]:
        raise ValueError(
            "Found fewer image placeholder spans than entries in `image_grid_thw`: "
            f"spans={grid_cursor}, images={grids.shape[0]}."
        )

    return jnp.asarray(position_ids, dtype="i4"), jnp.asarray(rope_deltas, dtype="i4").reshape(-1, 1)


def merge_multimodal_embeddings(
    input_ids: jax.Array,
    inputs_embeds: jax.Array,
    multimodal_embeddings: jax.Array,
    placeholder_token_id: int,
) -> jax.Array:
    """Scatter visual embeddings into text embeddings at placeholder positions.

    Args:
        input_ids: Token ids of shape ``(batch, seq_len)``.
        inputs_embeds: Text embeddings of shape ``(batch, seq_len, hidden)``.
        multimodal_embeddings: Flattened visual embeddings of shape
            ``(total_tokens, hidden)``.
        placeholder_token_id: Token id marking placeholder positions.

    Returns:
        jax.Array: Merged embeddings of shape ``(batch, seq_len, hidden)``.
    """
    batch_size, seq_len, hidden = inputs_embeds.shape
    is_multimodal = (input_ids == placeholder_token_id).reshape(-1)
    flat_embeds = inputs_embeds.reshape(-1, hidden)

    dummy_row = jnp.zeros_like(multimodal_embeddings[0:1])
    flattened_padded = jnp.concatenate([dummy_row, multimodal_embeddings], axis=0)
    gather_indices = jnp.cumsum(is_multimodal)
    update_values = flattened_padded[gather_indices]

    merged = jnp.where(is_multimodal[:, None], update_values, flat_embeds)
    return merged.reshape(batch_size, seq_len, hidden)


class HunYuanVLVisionEmbeddings(spx.Module):
    """Conv patch embedding plus an interpolated learned position grid.

    Pixels arrive as flat per-patch features. Each patch is embedded with a
    strided convolution and every image's patch sequence receives the
    learned positional grid (row 0 of the table is a legacy CLS slot; the
    remaining ``edge**2`` rows form the grid) bilinearly resized to that
    image's ``(h, w)`` patch shape.
    """

    def __init__(
        self,
        config: HunYuanVLVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ) -> None:
        """Initialize patch and position embeddings.

        Args:
            config (HunYuanVLVisionConfig): Vision tower configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.position_edge = config.max_image_size // config.patch_size
        self.num_positions = self.position_edge**2 + 1

        self.patch_embedding = nn.Conv(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            stride=(self.patch_size, self.patch_size),
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.position_embedding = Embed(
            num_embeddings=self.num_positions,
            features=self.embed_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            rngs=rngs,
        )

    def interpolate_pos_encoding(self, height: int, width: int) -> Array:
        """Resize the learned position grid to the current image grid.

        Args:
            height: Target grid height in patches.
            width: Target grid width in patches.

        Returns:
            Array: Positional embeddings of shape ``(height * width, dim)``.
        """
        grid = self.position_embedding.weight.value[1:, :].astype(jnp.float32)
        grid = grid.reshape(self.position_edge, self.position_edge, self.embed_dim)
        resized = jax.image.resize(
            grid,
            shape=(height, width, self.embed_dim),
            method=self.config.interpolate_mode,
            antialias=False,
        )
        return resized.reshape(height * width, self.embed_dim)

    def forward(self, pixel_values: Float[Array, "num_patches patch_features"], grid_thw: np.ndarray) -> Array:
        """Embed flat pixel patches and add interpolated positions per image.

        Args:
            pixel_values: Flat per-patch pixel features of shape
                ``(num_patches, channels * patch_size**2)``.
            grid_thw: Static per-image ``(t, h, w)`` patch grid.

        Returns:
            Array: Patch embeddings of shape ``(1, num_patches, dim)``.
        """
        num_patches = pixel_values.shape[0]
        patches = pixel_values.reshape(
            num_patches,
            self.config.num_channels,
            self.patch_size,
            self.patch_size,
        )
        patches = jnp.transpose(patches, (0, 2, 3, 1)).astype(self.dtype)
        embeddings = self.patch_embedding(patches).reshape(num_patches, self.embed_dim)

        start = 0
        per_image: list[Array] = []
        for t, h, w in np.asarray(grid_thw).reshape(-1, 3).tolist():
            t, h, w = int(t), int(h), int(w)
            end = start + t * h * w
            position_embedding = jnp.tile(
                self.interpolate_pos_encoding(h, w).astype(embeddings.dtype),
                (t, 1),
            )
            per_image.append(embeddings[start:end, :] + position_embedding)
            start = end

        return jnp.concatenate(per_image, axis=0)[None, ...]


class HunYuanVLVisionMLP(spx.Module):
    """Two-layer feed-forward block of the vision encoder (SigLIP shape)."""

    def __init__(
        self,
        config: HunYuanVLVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ) -> None:
        """Initialize the vision MLP.

        Args:
            config (HunYuanVLVisionConfig): Vision tower configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.act = ACT2FN[config.hidden_act]
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def forward(self, hidden_states: Array) -> Array:
        """Apply the feed-forward transformation.

        Args:
            hidden_states: Input of shape ``(..., hidden_size)``.

        Returns:
            Array: Output of shape ``(..., hidden_size)``.
        """
        return self.fc2(self.act(self.fc1(hidden_states)))


class HunYuanVLVisionAttention(UnifiedAttention):
    """Non-causal vision self-attention with per-image segment masking.

    No rotary embeddings are used (positions come from the learned grid).
    Patches from different images never attend to each other: attention is
    computed independently per ``cu_seqlens`` segment, exactly like the HF
    eager reference.
    """

    def __init__(
        self,
        config: HunYuanVLVisionConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize vision attention.

        Args:
            config (HunYuanVLVisionConfig): Vision tower configuration.
            layer_idx (int): Index of this layer in the encoder stack.
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
            layer_idx=layer_idx,
            attention_type="standard",
            causal=False,
        )

    def _create_rotary(self, config, dtype):
        """Vision attention carries no rotary embeddings.

        Args:
            config: Vision tower configuration (unused).
            dtype: Compute dtype (unused).

        Returns:
            None: The vision tower relies on learned position embeddings.
        """
        return None

    def _create_attention_performer(self, config, rngs: spx.Rngs):
        """Build a cache-free vanilla attention performer.

        Args:
            config: Vision tower configuration.
            rngs (spx.Rngs): Random number generator state.

        Returns:
            FlexibleAttentionModule: Vanilla attention without KV caching.
        """
        return FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=0.0,
            attn_mechanism="vanilla",
            requires_cache=False,
        )

    def forward(self, hidden_states: Float[Array, "one seq_len hidden"], cu_seqlens: np.ndarray) -> Array:
        """Apply segment-wise full self-attention over vision patches.

        Args:
            hidden_states: Patch features of shape ``(1, seq_len, hidden)``.
            cu_seqlens: Static cumulative segment lengths (one segment per
                image), shape ``(num_images + 1,)``.

        Returns:
            Array: Attention output of shape ``(1, seq_len, hidden)``.
        """
        seq_length = hidden_states.shape[1]
        query_states, key_states, value_states = self._project_qkv(hidden_states)
        query_states = query_states.reshape(1, seq_length, self.num_heads, self.head_dim)
        key_states = key_states.reshape(1, seq_length, self.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(1, seq_length, self.num_key_value_heads, self.head_dim)

        query_states = jnp.transpose(query_states, (0, 2, 1, 3))
        key_states = jnp.transpose(key_states, (0, 2, 1, 3))
        value_states = jnp.transpose(value_states, (0, 2, 1, 3))

        boundaries = [int(v) for v in np.asarray(cu_seqlens).tolist()]
        attn_outputs = []
        for seg_start, seg_end in itertools.pairwise(boundaries):
            q_chunk = query_states[:, :, seg_start:seg_end, :]
            k_chunk = key_states[:, :, seg_start:seg_end, :]
            v_chunk = value_states[:, :, seg_start:seg_end, :]
            attn_weights = jnp.matmul(q_chunk, jnp.swapaxes(k_chunk, -1, -2)) * (self.head_dim**-0.5)
            attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(q_chunk.dtype)
            attn_outputs.append(jnp.swapaxes(jnp.matmul(attn_weights, v_chunk), 1, 2))

        attn_output = jnp.concatenate(attn_outputs, axis=1).reshape(1, seq_length, -1)
        attn_output = self.shard_attention_prod(attn_output)
        attn_output = self.output_projection(attn_output)
        return attn_output


class HunYuanVLVisionBlock(spx.Module):
    """Pre-LayerNorm vision transformer block (SigLIP encoder-layer shape)."""

    def __init__(
        self,
        config: HunYuanVLVisionConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ) -> None:
        """Initialize the vision block.

        Args:
            config (HunYuanVLVisionConfig): Vision tower configuration.
            layer_idx (int): Index of this block in the encoder stack.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.layer_norm1 = LayerNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.self_attn = HunYuanVLVisionAttention(
            config=config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.layer_norm2 = LayerNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.mlp = HunYuanVLVisionMLP(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def forward(self, hidden_states: Array, cu_seqlens: np.ndarray) -> Array:
        """Run pre-norm attention and MLP with residual connections.

        Args:
            hidden_states: Patch features of shape ``(1, seq_len, hidden)``.
            cu_seqlens: Static cumulative segment lengths per image.

        Returns:
            Array: Updated patch features of shape ``(1, seq_len, hidden)``.
        """
        hidden_states = hidden_states + self.self_attn(self.layer_norm1(hidden_states), cu_seqlens=cu_seqlens)
        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


class HunYuanVLVisionPatchMerger(spx.Module):
    """Convolutional patch merger emitting newline / begin / end tokens.

    Per image: RMSNorm, a ``spatial_merge_size``-strided conv doubling the
    channel count, activation, a 1x1 conv to ``4 * hidden``, a learned
    newline token appended to every merged row, a linear projection into the
    text hidden size, learned begin/end tokens wrapped around the sequence,
    and a final RMSNorm.
    """

    def __init__(
        self,
        config: HunYuanVLVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ) -> None:
        """Initialize the patch merger.

        Args:
            config (HunYuanVLVisionConfig): Vision tower configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size

        self.before_rms = RMSNorm(
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.proj_conv = nn.Conv(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size * 2,
            kernel_size=(config.spatial_merge_size, config.spatial_merge_size),
            stride=(config.spatial_merge_size, config.spatial_merge_size),
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.proj_act = ACT2FN[config.hidden_act]
        self.proj_out = nn.Conv(
            in_channels=config.hidden_size * 2,
            out_channels=config.hidden_size * 4,
            kernel_size=(1, 1),
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.mlp = ColumnParallelLinear(
            config.hidden_size * 4,
            config.text_hidden_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.image_newline = ArrayParam.bound(
            shape=(config.hidden_size * 4,),
            dtype=param_dtype,
            init_method="normal",
            init_kwargs={"stddev": config.text_hidden_size**-0.5},
            key=rngs.param,
        )
        self.image_begin = ArrayParam.bound(
            shape=(config.text_hidden_size,),
            dtype=param_dtype,
            init_method="normal",
            init_kwargs={"stddev": config.text_hidden_size**-0.5},
            key=rngs.param,
        )
        self.image_end = ArrayParam.bound(
            shape=(config.text_hidden_size,),
            dtype=param_dtype,
            init_method="normal",
            init_kwargs={"stddev": config.text_hidden_size**-0.5},
            key=rngs.param,
        )
        self.image_sep = ArrayParam.bound(
            shape=(config.text_hidden_size,),
            dtype=param_dtype,
            init_method="normal",
            init_kwargs={"stddev": config.text_hidden_size**-0.5},
            key=rngs.param,
        )
        self.after_rms = RMSNorm(
            dim=config.text_hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def forward(self, hidden_states: Float[Array, "one seq_len hidden"], size: tuple[int, int]) -> Array:
        """Merge one image's patch features into text-space tokens.

        Args:
            hidden_states: Patch features of one image, shape
                ``(1, h * w, hidden)``.
            size: The image's ``(h, w)`` patch grid.

        Returns:
            Array: Merged tokens of shape
            ``(1, (h // m) * (w // m + 1) + 2, text_hidden)``.
        """
        height, width = int(size[0]), int(size[1])
        hidden_states = self.before_rms(hidden_states)
        dtype = hidden_states.dtype

        hidden_states = hidden_states.reshape(1, height, width, -1)
        hidden_states = self.proj_conv(hidden_states)
        hidden_states = self.proj_act(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        merged_h, merged_w, channels = hidden_states.shape[1], hidden_states.shape[2], hidden_states.shape[3]
        newline = jnp.broadcast_to(
            self.image_newline.value.astype(dtype).reshape(1, 1, 1, channels),
            (1, merged_h, 1, channels),
        )
        hidden_states = jnp.concatenate([hidden_states, newline], axis=2)
        hidden_states = hidden_states.reshape(1, merged_h * (merged_w + 1), channels)
        hidden_states = self.mlp(hidden_states)

        text_hidden = hidden_states.shape[-1]
        begin = jnp.broadcast_to(self.image_begin.value.astype(dtype).reshape(1, 1, text_hidden), (1, 1, text_hidden))
        end = jnp.broadcast_to(self.image_end.value.astype(dtype).reshape(1, 1, text_hidden), (1, 1, text_hidden))
        hidden_states = jnp.concatenate([begin, hidden_states, end], axis=1)
        return self.after_rms(hidden_states)


@register_module(TaskType.BASE_VISION, config=HunYuanVLVisionConfig, model_type="hunyuan_vl")
class HunYuanVLVisionModel(EasyDeLBaseModule):
    """HunYuanVL vision tower: patch embedding, encoder blocks, patch merger.

    Consumes flat per-patch pixel features plus a static ``grid_thw``
    describing every image's patch layout and returns the concatenation of
    merged per-image token sequences in the text hidden size.
    """

    config_class = HunYuanVLVisionConfig

    def __init__(
        self,
        config: HunYuanVLVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the vision tower.

        Args:
            config (HunYuanVLVisionConfig): Vision tower configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.

        Raises:
            AssertionError: If ``config`` is not a
                :class:`HunYuanVLVisionConfig` instance.
        """
        assert isinstance(config, HunYuanVLVisionConfig), "HunYuanVLVisionModel requires a HunYuanVLVisionConfig"
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.embeddings = HunYuanVLVisionEmbeddings(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.layers = nn.ModuleList([])
        for idx in range(config.num_hidden_layers):
            with self.assign_layer_stage(idx, total_layers=config.num_hidden_layers):
                self.layers.append(
                    HunYuanVLVisionBlock(
                        config=config,
                        layer_idx=idx,
                        dtype=dtype,
                        param_dtype=param_dtype,
                        precision=precision,
                        rngs=rngs,
                    )
                )
        self.patch_merger = HunYuanVLVisionPatchMerger(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def forward(self, pixel_values: Float[Array, "num_patches patch_features"], grid_thw) -> Array:
        """Encode images into merged text-space tokens.

        Args:
            pixel_values: Flat per-patch pixel features of shape
                ``(num_patches, channels * patch_size**2)``.
            grid_thw: Static per-image ``(t, h, w)`` patch grid of shape
                ``(num_images, 3)``.

        Returns:
            Array: Concatenated merged features of shape
            ``(1, total_merged_tokens, text_hidden)``.
        """
        grid = np.asarray(grid_thw).reshape(-1, 3)
        hidden_states = self.embeddings(pixel_values, grid)

        lengths = [int(t) * int(h) * int(w) for t, h, w in grid.tolist()]
        cu_seqlens = np.concatenate([[0], np.cumsum(lengths)])

        def _layer_loop(block, carry):
            """Apply one vision block inside the layer-stack scan."""
            hs, idx = carry
            with self._layer_stage_context(idx, layers=self.layers):
                hs = block(hs, cu_seqlens=cu_seqlens)
            hs = self._mark_layer_stage_boundary(hs, idx, layers=self.layers)
            return hs, idx + 1

        hidden_states, _ = self.layers.scan(
            _layer_loop,
            (hidden_states, 0),
            trace=not self.config.scan_layers or self._pipeline_stage_count() > 1,
        )

        merged: list[Array] = []
        start = 0
        for (t, h, w), length in zip(grid.tolist(), lengths, strict=True):
            del t
            item = hidden_states[:, start : start + length, :]
            merged.append(self.patch_merger(item, size=(int(h), int(w))))
            start += length
        return jnp.concatenate(merged, axis=1)

    def get_encoder(self):
        """Return the encoder (this vision model is the encoder)."""
        return self

    def get_decoder(self):
        """Vision towers have no decoder.

        Raises:
            NotImplementedError: Always; this is an encoder-only model.
        """
        raise NotImplementedError("This is an encoder-only model and does not have a decoder.")

    def get_lm_head(self):
        """Vision towers have no LM head.

        Raises:
            NotImplementedError: Always; vision models carry no LM head.
        """
        raise NotImplementedError("This vision model does not have a language model head.")

    def get_embedding(self):
        """Return the patch-embedding module."""
        return self.embeddings


class HunYuanVLMLP(spx.Module):
    """SwiGLU feed-forward block for the HunYuanVL text decoder."""

    def __init__(
        self,
        config: HunYuanVLTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize the text MLP.

        Args:
            config (HunYuanVLTextConfig): Text decoder configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the decoder stack.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
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
        self.act_fn = ACT2FN[config.hidden_act]

    @property
    def reform_param(self):
        """Fused gate/up checkpoint reform rules for HF conversion."""
        return self.gate_up_proj.build_reform_param("gate_up_proj", config=self.config)

    def forward(self, hidden_states: Float[Array, "batch seq_len hidden"]) -> jnp.ndarray:
        """Apply the SwiGLU transformation.

        Delegates to :func:`easydel.layers.gated_mlp_forward`, which routes
        supported layouts through the ejkernel fused MLP kernels and runs
        the legacy composition otherwise.

        Args:
            hidden_states: Input of shape ``(batch, seq_len, hidden)``.

        Returns:
            jnp.ndarray: Output of shape ``(batch, seq_len, hidden)``.
        """
        return gated_mlp_forward(self, hidden_states)


class HunYuanVLTextAttention(UnifiedAttention):
    """HunYuan dense attention with multimodal RoPE and post-RoPE QK norms.

    Unlike Qwen3-style QK-norm attention, HunYuan applies rotary embeddings
    to raw queries/keys first and then normalizes each head with
    ``query_layernorm`` / ``key_layernorm``.
    """

    norms_mapping: ClassVar[dict[str, str]] = {
        "query_normalization": "query_layernorm",
        "key_normalization": "key_layernorm",
        "value_normalization": "v_norm",
        "output_normalization": "o_norm",
    }

    def __init__(
        self,
        config: HunYuanVLTextConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize text attention.

        Args:
            config (HunYuanVLTextConfig): Text decoder configuration.
            layer_idx (int): Index of this layer in the decoder stack.
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
            layer_idx=layer_idx,
            attention_type="standard",
            causal=True,
            use_qk_norm=True,
        )

    def _apply_rotary(self, query_states, key_states, position_ids, frequencies=None):
        """Rotate queries/keys with mRoPE, then apply the per-head QK norms.

        Args:
            query_states: Queries ``(batch, seq, heads, head_dim)``.
            key_states: Keys ``(batch, seq, kv_heads, head_dim)``.
            position_ids: Positions ``(batch, seq)`` or ``(3, batch, seq)``.
            frequencies: Optional precomputed RoPE frequency cache.

        Returns:
            tuple: Rotated and normalized ``(query_states, key_states)``.
        """
        query_states, key_states = super()._apply_rotary(query_states, key_states, position_ids, frequencies)
        return self.query_normalization(query_states), self.key_normalization(key_states)


class HunYuanVLTextDecoderLayer(spx.Module):
    """Pre-norm decoder layer: HunYuan attention plus a SwiGLU MLP."""

    def __init__(
        self,
        config: HunYuanVLTextConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the decoder layer.

        Args:
            config (HunYuanVLTextConfig): Text decoder configuration.
            layer_idx (int): Index of this layer in the decoder stack.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.self_attn = HunYuanVLTextAttention(
            config=config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.mlp = HunYuanVLMLP(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
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
        hidden_states: Float[Array, "batch seq_len hidden"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "three batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type: ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ) -> DecoderLayerOutput:
        """Run the decoder layer.

        Args:
            hidden_states: Input of shape ``(batch, seq_len, hidden)``.
            mask_info: Attention mask information.
            position_ids: 3-axis mRoPE positions ``(3, batch, seq_len)``.
            mode: Runtime mode (train / prefill / decode).
            cache_view: Optional KV cache view.
            cache_metadata: Optional cache metadata.
            output_attentions: Whether to return attention weights.
            frequencies: Optional precomputed RoPE frequencies.

        Returns:
            DecoderLayerOutput: Updated hidden states, optional attention
            weights, and the cache view.
        """
        attn_outputs: AttentionLayerOutput = self.self_attn(
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
        if self.config.use_scan_mlp:
            feed_forward_hidden_states = blockwise_ffn(self.mlp, feed_forward_input, self.config.scan_mlp_chunk_size)
        else:
            feed_forward_hidden_states = self.mlp(feed_forward_input)
        hidden_states = checkpoint_name(hidden_states + feed_forward_hidden_states, "residual")

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


@register_module(TaskType.BASE_MODULE, config=HunYuanVLTextConfig, model_type="hunyuan_vl")
class HunYuanVLTextModel(EasyDeLBaseModule):
    """HunYuanVL language-model trunk processing text and visual tokens."""

    config_class = HunYuanVLTextConfig

    def __init__(
        self,
        config: HunYuanVLTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the text model.

        Args:
            config (HunYuanVLTextConfig): Text decoder configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.

        Raises:
            AssertionError: If ``config`` is not a
                :class:`HunYuanVLTextConfig` instance.
        """
        assert isinstance(config, HunYuanVLTextConfig), "HunYuanVLTextModel requires a HunYuanVLTextConfig"
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        with self.assign_layer_stage(0, total_layers=self.config.num_hidden_layers):
            self.embed_tokens = Embed(
                num_embeddings=self.config.vocab_size,
                features=self.config.hidden_size,
                dtype=dtype,
                param_dtype=param_dtype,
                embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
                rngs=rngs,
            )

        remat_layer_block = auto_remat(
            HunYuanVLTextDecoderLayer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.layers = nn.ModuleList([])
        for idx in range(self.config.num_hidden_layers):
            with self.assign_layer_stage(idx, total_layers=self.config.num_hidden_layers):
                self.layers.append(
                    remat_layer_block(
                        config=config,
                        layer_idx=idx,
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
        inputs_embeds: Float[Array, "batch seq_len hidden"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        trace: bool = False,
    ) -> BaseModelOutput:
        """Run the decoder stack.

        Args:
            input_ids: Token ids ``(batch, seq_len)``; mutually exclusive
                with ``inputs_embeds``.
            inputs_embeds: Precomputed embeddings ``(batch, seq_len,
                hidden)``, typically with visual tokens merged in.
            attention_mask: Optional attention mask ``(batch, seq_len)``.
            mask_info: Optional structured mask information.
            position_ids: 3-axis mRoPE positions ``(3, batch, seq_len)`` or
                2-D text positions broadcast across axes.
            mode: Runtime mode; inferred when ``None``.
            past_key_values: Optional KV cache.
            cache_metadata: Optional cache metadata.
            apply_lm_head: Unused (API compatibility).
            output_attentions: Whether to collect attention weights.
            output_hidden_states: Whether to collect per-layer states.
            trace: Layer-scan trace override.

        Returns:
            BaseModelOutput: Final hidden state plus optional collections.

        Raises:
            ValueError: If both or neither of ``input_ids`` /
                ``inputs_embeds`` are provided.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = checkpoint_name(self.embed_tokens(input_ids.astype("i4")), "embeddings")
        sequence_length = inputs_embeds.shape[1]

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
            batch_size = inputs_embeds.shape[0]
            pos_2d = mask_info.q_position_ids
            position_ids = jnp.broadcast_to(pos_2d[None, :, :], (3, batch_size, sequence_length))
        elif position_ids.ndim == 2:
            batch_size = position_ids.shape[0]
            position_ids = jnp.broadcast_to(position_ids[None, :, :], (3, batch_size, sequence_length))

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

        trace_layers = self._layer_scan_trace(
            trace,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            cache_views=views,
            extra=needs_trace_cache,
        )
        cache_views = views if trace_layers else None

        def _run_layer(block, carry):
            """Apply one decoder layer inside the layer-stack scan."""
            hs, cv, ah, aa, idx = carry
            if output_hidden_states:
                ah = (*ah, hs)
            with self._layer_stage_context(idx, layers=self.layers):
                layer_outputs = block(
                    hidden_states=hs,
                    mask_info=mask_info,
                    position_ids=position_ids,
                    mode=mode,
                    cache_view=self._layer_cache_view_at(cv, idx, enabled=trace_layers, cache=past_key_values),
                    cache_metadata=cache_metadata,
                    output_attentions=output_attentions,
                    frequencies=self.frequencies,
                )
            hs = self._mark_layer_stage_boundary(layer_outputs.hidden_states, idx, layers=self.layers)
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
        hidden_states = checkpoint_name(self.norm(hidden_states), "model_output")

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            past_key_values=past_key_values,
        )

    def get_encoder(self):
        """Decoder-only models have no encoder.

        Raises:
            NotImplementedError: Always; this model is decoder-only.
        """
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        """Return the decoder trunk (this module)."""
        return self

    def get_lm_head(self):
        """Base models carry no LM head.

        Raises:
            NotImplementedError: Always; the head lives on the task wrapper.
        """
        raise NotImplementedError("The base model does not have a language model head.")

    def get_embedding(self):
        """Return the token embedding."""
        return self.embed_tokens

    def set_embeddings(self, value):
        """Set the token embedding.

        Args:
            value: New embedding module.
        """
        self.embed_tokens = value


@register_module(TaskType.BASE_MODULE, config=HunYuanVLConfig, model_type="hunyuan_vl")
class HunYuanVLModel(EasyDeLBaseModule):
    """Composite HunYuanVL base model (vision tower + language model)."""

    config_class = HunYuanVLConfig

    def __init__(
        self,
        config: HunYuanVLConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the composite model.

        Args:
            config (HunYuanVLConfig): Composite configuration.
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
        self.vision_tower = HunYuanVLVisionModel(
            config.vision_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.language_model = HunYuanVLTextModel(
            config.text_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.rope_deltas = None

    def set_embeddings(self, value):
        """Set the text token embedding.

        Args:
            value: New embedding module.
        """
        self.language_model.set_embeddings(value)

    def get_embedding(self):
        """Return the text token embedding."""
        return self.language_model.get_embedding()

    def get_decoder(self):
        """Return the language-model trunk."""
        return self.language_model

    def get_image_features(self, pixel_values: Array, image_grid_thw=None) -> Array:
        """Encode images through the vision tower.

        Args:
            pixel_values: Flat per-patch pixel features.
            image_grid_thw: Static per-image ``(t, h, w)`` patch grid.

        Returns:
            Array: Merged visual features of shape
            ``(total_tokens, text_hidden)``.
        """
        pixel_values = pixel_values.astype(self.vision_tower.dtype)
        image_embeds = self.vision_tower(pixel_values, grid_thw=image_grid_thw)
        return image_embeds.reshape(-1, image_embeds.shape[-1])

    def get_rope_index(
        self,
        input_ids: Array | None = None,
        image_grid_thw: Array | None = None,
        video_grid_thw: Array | None = None,
        attention_mask: Array | None = None,
        mm_token_type_ids: Array | None = None,
    ) -> tuple[Array, Array]:
        """Compute HunYuanVL multimodal RoPE indices.

        Args:
            input_ids: Token ids ``(batch, seq_len)``.
            image_grid_thw: Per-image ``(t, h, w)`` grid.
            video_grid_thw: Unsupported; must be ``None``.
            attention_mask: Optional attention mask.
            mm_token_type_ids: Optional per-token modality types.

        Returns:
            tuple: ``(position_ids, rope_deltas)`` of shapes
            ``(3, batch, seq_len)`` and ``(batch, 1)``.

        Raises:
            NotImplementedError: If ``video_grid_thw`` is provided.
        """
        if video_grid_thw is not None:
            raise NotImplementedError("HunYuanVL does not support video inputs.")
        return get_rope_index(
            input_ids=np.asarray(input_ids),
            image_grid_thw=np.asarray(image_grid_thw) if image_grid_thw is not None else None,
            attention_mask=np.asarray(attention_mask) if attention_mask is not None else None,
            mm_token_type_ids=np.asarray(mm_token_type_ids) if mm_token_type_ids is not None else None,
            spatial_merge_size=self.config.vision_config.spatial_merge_size,
            image_token_id=self.config.image_token_id,
        )

    def compute_embedding(
        self,
        input_ids: Int[Array, "batch seq_len"] | None,
        *,
        pixel_values: Array | None = None,
        image_grid_thw: Array | None = None,
        image_embeds: Array | None = None,
        **kwargs,
    ) -> Array:
        """Compute input embeddings with visual tokens merged in.

        Args:
            input_ids: Token ids ``(batch, seq_len)``.
            pixel_values: Optional flat per-patch pixel features.
            image_grid_thw: Static per-image ``(t, h, w)`` grid.
            image_embeds: Optional precomputed visual embeddings.
            **kwargs: Ignored extra arguments.

        Returns:
            Array: Embeddings of shape ``(batch, seq_len, hidden)``.

        Raises:
            ValueError: If ``input_ids`` is ``None``.
        """
        if input_ids is None:
            raise ValueError("`input_ids` must be provided when calling `compute_embedding`.")

        inputs_embeds = super().compute_embedding(input_ids)

        if image_embeds is None and pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)

        if image_embeds is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=image_embeds.astype(inputs_embeds.dtype),
                placeholder_token_id=self.config.image_token_id,
            )
        return inputs_embeds

    def compute_embedding_with_info(
        self,
        input_ids: Int[Array, "batch seq_len"],
        *,
        attention_mask: Array | None = None,
        **kwargs,
    ) -> tuple[Array, EmbeddingInfo]:
        """Compute embeddings plus the mRoPE info needed for serving.

        Computes multimodal embeddings and eagerly derives the 3-axis mRoPE
        ``position_ids`` and ``rope_deltas`` via :meth:`get_rope_index` so
        callers that pass ``inputs_embeds`` (or explicit positions) to the
        forward — e.g. paged serving runtimes — stay position-consistent
        with the dense forward.

        Args:
            input_ids: Input token IDs of shape ``(batch, seq_len)``.
            attention_mask: Optional attention mask used by
                :meth:`get_rope_index`.
            **kwargs: Additional arguments (``pixel_values``,
                ``image_grid_thw``, ...) forwarded to
                :meth:`compute_embedding`.

        Returns:
            tuple[Array, EmbeddingInfo]: Embeddings of shape
            ``(batch, seq_len, hidden)`` and an :class:`EmbeddingInfo`
            carrying ``position_ids`` ``(3, batch, seq_len)`` and
            ``rope_deltas`` ``(batch, 1)``.
        """
        inputs_embeds = self.compute_embedding(input_ids, **kwargs)
        position_ids, rope_deltas = self.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=kwargs.get("image_grid_thw") if kwargs.get("pixel_values") is not None else None,
            attention_mask=attention_mask,
        )
        return inputs_embeds, EmbeddingInfo(position_ids=position_ids, rope_deltas=rope_deltas)

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden"] | None = None,
        pixel_values: Array | None = None,
        image_grid_thw: Array | None = None,
        mm_token_type_ids: Array | None = None,
        rope_deltas: Array | None = None,
        mask_info: MaskInfo | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> BaseModelOutput:
        """Run the composite model.

        Note:
            When ``position_ids`` is ``None`` and images are present, the
            multimodal indices are computed with NumPy control flow and are
            NOT JIT-compatible; pass explicit ``position_ids`` inside traced
            code.

        Args:
            input_ids: Token ids ``(batch, seq_len)``.
            attention_mask: Optional attention mask.
            position_ids: Optional positions (2-D text or 3-axis mRoPE).
            past_key_values: Optional KV cache.
            inputs_embeds: Optional precomputed embeddings.
            pixel_values: Optional flat per-patch pixel features.
            image_grid_thw: Static per-image ``(t, h, w)`` grid.
            mm_token_type_ids: Optional per-token modality types.
            rope_deltas: Unused pass-through for generation plumbing.
            mask_info: Optional structured mask information.
            mode: Runtime mode.
            cache_metadata: Optional cache metadata.
            output_attentions: Whether to collect attention weights.
            output_hidden_states: Whether to collect per-layer states.
            **kwargs: Ignored extra arguments.

        Returns:
            BaseModelOutput: Final hidden state plus optional collections.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if inputs_embeds is None:
            inputs_embeds = self.compute_embedding(
                input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        if position_ids is None and input_ids is not None and pixel_values is not None:
            position_ids, _rope_deltas = self.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids,
            )
        elif position_ids is not None and position_ids.ndim == 2:
            batch_size, seq_length = position_ids.shape
            position_ids = jnp.broadcast_to(position_ids[None, :, :], (3, batch_size, seq_length))

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            mask_info=mask_info,
            mode=mode,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_metadata=cache_metadata,
        )
        return BaseModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@register_module(TaskType.IMAGE_TEXT_TO_TEXT, config=HunYuanVLConfig, model_type="hunyuan_vl")
class HunYuanVLForConditionalGeneration(BaseVisionLanguageModule[HunYuanVLModel, HunYuanVLConfig]):  # type: ignore
    """HunYuanVL image-text-to-text task head with a tied LM head.

    Class Attributes:
        _task_type: IMAGE_TEXT_TO_TEXT task type.
        _model_type: ``"hunyuan_vl"``.
        _supports_video: ``False`` (image-only family).
        _uses_mrope: ``True`` (3-axis multimodal RoPE).
    """

    _task_type = TaskType.IMAGE_TEXT_TO_TEXT
    _model_type = "hunyuan_vl"
    _config_class = HunYuanVLConfig
    _auto_register = False
    _supports_video = False
    _uses_mrope = True

    _vision_tower_name = "vision_tower"
    _language_model_name = "language_model"

    loss_type = "ForCausalLM"

    def __init__(
        self,
        config: HunYuanVLConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the task head.

        Args:
            config (HunYuanVLConfig): Composite configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        mrope_section = tuple(config.text_config.rope_scaling["mrope_section"])
        super().__init__(
            config=config,
            base_model_class=HunYuanVLModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            vision_feature_layer=-1,
            vision_feature_select_strategy="default",
            image_token_index=config.image_token_id,
            spatial_merge_size=config.vision_config.spatial_merge_size,
            mrope_section=mrope_section,
            tie_word_embeddings=getattr(config, "tie_word_embeddings", True),
            lm_head_bias=False,
        )
        self.vocab_size = config.text_config.vocab_size

    def get_embedding(self):
        """Return the text token embedding."""
        return self.base_model.get_embedding()

    def set_embeddings(self, value):
        """Set the text token embedding.

        Args:
            value: New embedding module.
        """
        self.base_model.set_embeddings(value)

    def get_vision_tower(self) -> spx.Module:
        """Return the vision tower."""
        return self.base_model.vision_tower

    def get_language_model(self) -> spx.Module:
        """Return the language-model trunk."""
        return self.base_model.language_model

    def get_image_features(
        self,
        pixel_values: Float[Array, "num_patches patch_features"],
        image_grid_thw: tp.Any | None = None,
        **kwargs,
    ) -> Array:
        """Encode images through the vision tower.

        Args:
            pixel_values: Flat per-patch pixel features.
            image_grid_thw: Static per-image ``(t, h, w)`` grid.
            **kwargs: Ignored extra arguments.

        Returns:
            Array: Merged visual features ``(total_tokens, text_hidden)``.
        """
        return self.base_model.get_image_features(pixel_values, image_grid_thw)

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        inputs_embeds: Float[Array, "batch seq_len hidden"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        pixel_values: Array | None = None,
        image_grid_thw: tp.Any | None = None,
        mm_token_type_ids: Array | None = None,
        rope_deltas: Array | None = None,
        **kwargs,
    ) -> VLMCausalLMOutput:
        """Run the multimodal forward pass and project logits.

        Args:
            input_ids: Token ids ``(batch, seq_len)`` with image
                placeholders where visual content goes.
            attention_mask: Optional attention mask.
            mask_info: Optional structured mask information.
            position_ids: Optional positions (2-D text or 3-axis mRoPE).
            mode: Runtime mode.
            past_key_values: Optional KV cache.
            cache_metadata: Optional cache metadata.
            apply_lm_head: Whether to compute vocabulary logits.
            inputs_embeds: Optional precomputed embeddings.
            output_attentions: Whether to collect attention weights.
            output_hidden_states: Whether to collect per-layer states.
            pixel_values: Optional flat per-patch pixel features.
            image_grid_thw: Static per-image ``(t, h, w)`` grid.
            mm_token_type_ids: Optional per-token modality types.
            rope_deltas: Optional mRoPE deltas pass-through.
            **kwargs: Ignored extra arguments.

        Returns:
            VLMCausalLMOutput: Logits (optional), cache, hidden states,
            attentions and rope deltas.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            mask_info=mask_info,
            mode=mode,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs.last_hidden_state
        hidden_states = apply_logical_sharding(
            hidden_states,
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
            rope_deltas=rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        max_length: int,
        pad_token_id: int,
        starts: int | None = None,
        past_key_values=None,
        attention_mask=None,
        mask_info=None,
        inputs_embeds=None,
        position_ids=None,
        pixel_values=None,
        image_grid_thw=None,
        mm_token_type_ids=None,
        rope_deltas=None,
        **kwargs,
    ):
        """Prepare inputs for autoregressive generation.

        Args:
            input_ids: Prompt token ids.
            max_length: Maximum sequence length for generation.
            pad_token_id: Padding token id.
            starts: Optional prefill lengths.
            past_key_values: Optional preallocated KV cache.
            attention_mask: Optional attention mask.
            mask_info: Optional structured mask information.
            inputs_embeds: Optional precomputed embeddings.
            position_ids: Optional positions.
            pixel_values: Optional pixel features (first step only).
            image_grid_thw: Static per-image grid.
            mm_token_type_ids: Optional per-token modality types.
            rope_deltas: Optional precomputed mRoPE deltas.
            **kwargs: Ignored extra arguments.

        Returns:
            dict: Model inputs for the generation loop.
        """
        batch_size, seq_length = input_ids.shape

        if past_key_values is None:
            if starts is None:
                starts = self.compute_prefill_length(input_ids, pad_token_id)
            past_key_values = self.init_cache(batch_size, max_length, starts, None, pad_token_id)

        if inputs_embeds is not None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {}

        if attention_mask is None:
            attn_2d = jnp.ones((batch_size, seq_length), dtype="b1")
        else:
            attn_2d = attention_mask

        extended_attention_mask = jnp.ones((batch_size, max_length), dtype=attn_2d.dtype)
        extended_attention_mask = jax.lax.dynamic_update_slice(extended_attention_mask, attn_2d, (0, 0))

        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids if inputs_embeds is None else None,
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
        )

        # Note: get_rope_index uses NumPy and is NOT JIT-compatible.
        if position_ids is not None and position_ids.ndim == 2:
            position_ids = jnp.broadcast_to(position_ids[None, :, :], (3, batch_size, seq_length))
            if rope_deltas is None:
                max_pos = jnp.max(position_ids)
                rope_deltas = jnp.full((batch_size, 1), max_pos + 1 - seq_length, dtype=jnp.int32)
        elif position_ids is None:
            if image_grid_thw is not None:
                position_ids, rope_deltas = self.base_model.get_rope_index(
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    attention_mask=attn_2d,
                    mm_token_type_ids=mm_token_type_ids,
                )
            else:
                attn_mask_int = attn_2d.astype(jnp.int32)
                position_ids = jnp.cumsum(attn_mask_int, axis=-1) - 1
                position_ids = jnp.where(attn_mask_int == 0, 1, position_ids)
                position_ids = jnp.broadcast_to(position_ids[None, :, :], (3, batch_size, seq_length))
                max_pos = jnp.max(position_ids)
                rope_deltas = jnp.full((batch_size, 1), max_pos + 1 - seq_length, dtype=jnp.int32)

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "attention_mask": extended_attention_mask,
                "mask_info": mask_info,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "rope_deltas": rope_deltas,
            }
        )
        return model_inputs

    def _create_required_props_from_kwargs(
        self,
        model_kwargs: dict[str, Array],
    ) -> tp.Mapping[str, dict[str, tp.Any]] | None:
        """Preserve grid shapes across generation steps.

        Args:
            model_kwargs: Current generation kwargs.

        Returns:
            Mapping | None: Properties to preserve.
        """
        basics = {}
        if "image_grid_thw" in model_kwargs:
            basics.update({"image_grid_thw": {"value": jnp.array(model_kwargs["image_grid_thw"])}})
        return basics

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        """Advance generation state and drop one-time vision inputs.

        Args:
            model_outputs: Outputs of the previous step.
            model_kwargs: Current generation kwargs.

        Returns:
            dict: Updated kwargs for the next step.
        """
        model_kwargs["past_key_values"] = model_outputs.past_key_values

        if getattr(model_outputs, "rope_deltas", None) is not None:
            model_kwargs["rope_deltas"] = model_outputs.rope_deltas

        if model_kwargs.get("position_ids") is not None:
            position_ids = model_kwargs["position_ids"]
            if position_ids.ndim == 3:
                model_kwargs["position_ids"] = position_ids[:, :, -1:] + 1
            else:
                model_kwargs["position_ids"] = position_ids[:, -1:] + 1

        model_kwargs.pop("pixel_values", None)
        model_kwargs.pop("mm_token_type_ids", None)
        return model_kwargs

    def prepare_inputs_for_call(
        self,
        image_grid_thw: tp.Any | None = None,
        drop_ids: bool = True,
        **others,
    ):
        """Prepare inputs with mRoPE positions computed from grid shapes.

        Args:
            image_grid_thw: Static per-image ``(t, h, w)`` grid.
            drop_ids: Whether to drop ``input_ids`` from the result.
            **others: Remaining call arguments.

        Returns:
            dict: Prepared inputs including ``position_ids`` and
            ``mask_info``.
        """
        attention_mask = others.get("attention_mask", None)
        mask_info = others.get("mask_info", None)
        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=others.get("input_ids"),
            inputs_embeds=others.get("inputs_embeds"),
            attention_mask=attention_mask,
        )

        input_ids = others.get("input_ids", None)
        target_length = input_ids.shape[1] if input_ids is not None else None
        attention_mask = token_attention_mask_from_mask_info(mask_info, target_length)
        rope_deltas = others.get("rope_deltas", None)
        position_ids = others.get("position_ids", None)

        if position_ids is None and input_ids is not None and image_grid_thw is not None:
            position_ids, rope_deltas = self.base_model.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
                mm_token_type_ids=others.get("mm_token_type_ids"),
            )
        elif position_ids is not None and position_ids.ndim == 2:
            batch_size, sequence_length = position_ids.shape
            position_ids = jnp.broadcast_to(position_ids[None, :, :], (3, batch_size, sequence_length))
            if rope_deltas is None:
                max_pos = jnp.max(position_ids)
                rope_deltas = jnp.full((batch_size, 1), max_pos + 1 - sequence_length, dtype=jnp.int32)
        elif position_ids is None and input_ids is not None and (attention_mask is None or attention_mask.ndim == 2):
            batch_size, sequence_length = input_ids.shape
            if attention_mask is not None and attention_mask.ndim == 2:
                attn_mask_int = attention_mask.astype(jnp.int32)
                position_ids = jnp.cumsum(attn_mask_int, axis=-1) - 1
                position_ids = jnp.where(attn_mask_int == 0, 1, position_ids)
                position_ids = jnp.broadcast_to(position_ids[None, :, :], (3, batch_size, sequence_length))
                if rope_deltas is None:
                    max_pos = jnp.max(position_ids)
                    rope_deltas = jnp.full((batch_size, 1), max_pos + 1 - sequence_length, dtype=jnp.int32)
            else:
                position_ids = jnp.broadcast_to(
                    jnp.arange(sequence_length, dtype=jnp.int32)[None, None, :],
                    (3, batch_size, sequence_length),
                )
                if rope_deltas is None:
                    rope_deltas = jnp.zeros((batch_size, 1), dtype=jnp.int32)

        # Only drop ``input_ids`` when precomputed ``inputs_embeds`` are supplied.
        # The training/eval loss path passes ``input_ids`` with no
        # ``inputs_embeds``; dropping it there would leave the forward with
        # nothing to embed and raise in ``compute_embedding``.
        if drop_ids and others.get("inputs_embeds") is not None:
            others.pop("input_ids", None)
        # NOTE: ``rope_deltas`` is intentionally NOT written back into the call
        # kwargs. It is a ``VLMCausalLMOutput`` field, and the loss path spreads
        # both ``**outputs`` and ``**batch`` into the loss function; leaving a
        # ``rope_deltas`` entry in the batch collides with the output field and
        # raises ``TypeError: got multiple values for 'rope_deltas'``. The head
        # only passes ``rope_deltas`` through to its output, so the training loss
        # path does not need it as an input.
        others.update(
            dict(
                image_grid_thw=image_grid_thw,
                position_ids=position_ids,
                attention_mask=attention_mask,
                mask_info=mask_info,
            )
        )
        # Strip trainer-only loss-masking keys (e.g. completion_mask) so they are
        # not forwarded into the module call. See BaseVisionLanguageModule.
        return self._drop_trainer_only_kwargs(others)

    def get_static_arguments(self):
        """Return call arguments treated as static under JIT.

        Returns:
            tuple[str, ...]: Static argument names.
        """
        return ("image_grid_thw",)
