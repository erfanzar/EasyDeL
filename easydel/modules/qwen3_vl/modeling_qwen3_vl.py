# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

import math
from functools import cached_property, partial

import jax
import jax.numpy as jnp
import numpy as np
from eformer import common_types
from eformer.escale import apply_logical_sharding
from eformer.pytree import auto_pytree
from ejkernel.types import MaskInfo
from flax import nnx as nn
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import (
    BaseModelOutput,
    DecoderLayerOutput,
    EmbeddingInfo,
    ModelOutput,
    VLMCausalLMOutput,
)
from easydel.infra.utils import ACT2FN, auto_remat, block_wise_ffn
from easydel.layers.attention import FlexibleAttentionModule
from easydel.layers.attention_unified import UnifiedAttention
from easydel.layers.base_modules import BaseVisionLanguageModule
from easydel.layers.caching import (
    HybridCache,
    OperationsMetadata,
    RaggedPagesCache,
    RaggedPagesCacheView,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.layers.components import ColumnParallelLinear, Embed, RMSNorm, RowParallelLinear
from easydel.layers.components.norms import LayerNorm

from .qwen3_vl_configuration import Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig


@auto_pytree
class Qwen3VLCausalLMOutputWithPast(ModelOutput):
    """Output class for Qwen3-VL causal language model.

    Attributes:
        loss: Language modeling loss if labels provided.
        logits: Prediction scores of the language model head.
        past_key_values: Tuple of past key values for efficient generation.
        hidden_states: Tuple of hidden states at each layer.
        attentions: Tuple of attention weights at each layer.
        rope_deltas: RoPE position deltas for mRoPE.
        image_hidden_states: Hidden states from image processing.
    """

    loss: Array | None = None
    logits: Array = None
    past_key_values: list[Array] | None = None
    hidden_states: tuple[Array] | None = None
    attentions: tuple[Array] | None = None
    rope_deltas: Array | None = None
    image_hidden_states: Float[Array, "batch seq_len hidden_dim"] | None = None


@auto_pytree
class Qwen3VLModelOutputWithPast(ModelOutput):
    """Base-model output for Qwen3-VL with optional mRoPE deltas."""

    last_hidden_state: Array = None
    past_key_values: list[Array] | None = None
    hidden_states: tuple[Array] | None = None
    attentions: tuple[Array] | None = None
    rope_deltas: Array | None = None


def get_rope_index(
    input_ids: np.ndarray,
    image_grid_thw: np.ndarray | None = None,
    video_grid_thw: np.ndarray | None = None,
    attention_mask: np.ndarray | None = None,
    spatial_merge_size: int = 1,
    image_token_id: int = -1,
    video_token_id: int = -1,
    vision_start_token_id: int = -1,
    tokens_per_second: float = 1.0,
    second_per_grid_ts: list[float] | None = None,
    context_len: int = 0,
    seq_len: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Calculate 3D RoPE indices for multimodal inputs.

    Computes position IDs with temporal, height, and width dimensions for
    proper 3D rotary position embeddings (mRoPE) in Qwen3-VL.

    Different from Qwen2-VL, Qwen3-VL uses timestamps rather than absolute
    time position IDs for videos.

    Args:
        input_ids: Token IDs of shape (batch_size, sequence_length).
        image_grid_thw: Temporal/height/width grid for images.
        video_grid_thw: Temporal/height/width grid for videos.
        attention_mask: Attention mask for padding.
        spatial_merge_size: Spatial merge factor for vision.
        image_token_id: Token ID representing images.
        video_token_id: Token ID representing videos.
        vision_start_token_id: Token ID for vision sequence start.
        tokens_per_second: Temporal scaling for video tokens.
        second_per_grid_ts: Per-video seconds per temporal grid step.
        context_len: Existing KV context length offset.
        seq_len: Target sequence length.

    Returns:
        Tuple of (position_ids, mrope_position_deltas).
        position_ids has shape (3, batch_size, sequence_length).
    """
    if second_per_grid_ts is None:
        second_per_grid_ts = []

    if video_grid_thw is not None:
        video_grid_thw = video_grid_thw.copy()
        video_grid_thw = np.repeat(video_grid_thw, video_grid_thw[:, 0].astype(int), axis=0)
        video_grid_thw[:, 0] = 1

    if input_ids.shape[-1] != 1 and attention_mask is not None:
        attention_mask = attention_mask[:, : input_ids.shape[-1]]

    batch_size, seq_length = input_ids.shape[:2]
    position_ids = np.ones((3, batch_size, seq_length), dtype=np.int32)
    mrope_position_deltas: list[int] = []
    image_index, video_index = 0, 0

    if image_grid_thw is not None or video_grid_thw is not None:
        for i in range(batch_size):
            valid_mask = attention_mask[i] == 1 if attention_mask is not None else np.ones(seq_length, dtype=bool)
            input_tokens = input_ids[i][valid_mask].tolist()
            vision_start_indices = np.where(np.array(input_tokens) == vision_start_token_id)[0]
            vision_tokens = np.array(input_tokens)[vision_start_indices + 1]
            image_nums = int(np.sum(vision_tokens == image_token_id))
            video_nums = int(np.sum(vision_tokens == video_token_id))

            llm_pos_ids_list: list[np.ndarray] = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums

            for _ in range(image_nums + video_nums):
                video_second_per_grid_t = 0.0
                try:
                    ed_image = input_tokens.index(image_token_id, st) if remain_images > 0 else len(input_tokens) + 1
                except ValueError:
                    ed_image = len(input_tokens) + 1
                try:
                    ed_video = input_tokens.index(video_token_id, st) if remain_videos > 0 else len(input_tokens) + 1
                except ValueError:
                    ed_video = len(input_tokens) + 1

                if ed_image < ed_video:
                    t, h, w = image_grid_thw[image_index]
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = video_grid_thw[video_index]
                    video_second_per_grid_t = 1.0
                    if second_per_grid_ts:
                        video_second_per_grid_t = second_per_grid_ts[video_index]
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                llm_grid_t, llm_grid_h, llm_grid_w = (
                    int(t),
                    int(h) // spatial_merge_size,
                    int(w) // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max().item() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(
                    np.broadcast_to(np.arange(text_len, dtype=np.int32).reshape(1, -1), (3, text_len)) + st_idx
                )

                t_index = (
                    (
                        np.broadcast_to(
                            np.arange(llm_grid_t, dtype=np.int32).reshape(-1, 1),
                            (llm_grid_t, llm_grid_h * llm_grid_w),
                        )
                        * video_second_per_grid_t
                        * tokens_per_second
                    )
                    .astype(np.int32)
                    .flatten()
                )
                h_index = np.broadcast_to(
                    np.arange(llm_grid_h, dtype=np.int32).reshape(1, -1, 1),
                    (llm_grid_t, llm_grid_h, llm_grid_w),
                ).flatten()
                w_index = np.broadcast_to(
                    np.arange(llm_grid_w, dtype=np.int32).reshape(1, 1, -1),
                    (llm_grid_t, llm_grid_h, llm_grid_w),
                ).flatten()

                llm_pos_ids_list.append(
                    np.stack([t_index, h_index, w_index], axis=0) + text_len + st_idx,
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max().item() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    np.broadcast_to(np.arange(text_len, dtype=np.int32).reshape(1, -1), (3, text_len)) + st_idx
                )

            llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
            target_seq_len = seq_len if seq_len is not None else llm_positions.shape[-1]
            llm_positions = llm_positions[:, context_len:target_seq_len]

            position_ids[:, i, valid_mask] = llm_positions
            mrope_position_deltas.append(int(llm_positions.max() + 1 - len(input_tokens)))
    else:
        if attention_mask is not None:
            position_ids = jnp.cumsum(attention_mask, axis=-1) - 1
            position_ids = jnp.where(attention_mask == 0, 1, position_ids)
            position_ids = jnp.expand_dims(position_ids, axis=0).repeat(3, axis=0)
            max_position_ids = jnp.max(position_ids, axis=(0, 2), keepdims=True)
            mrope_position_deltas = (max_position_ids + 1 - attention_mask.shape[-1]).reshape(-1)
        else:
            position_ids = jnp.arange(seq_length).reshape(1, 1, -1).repeat(3, axis=0).repeat(batch_size, axis=1)
            mrope_position_deltas = jnp.zeros((batch_size,), dtype=input_ids.dtype)

    return jnp.asarray(position_ids), jnp.asarray(mrope_position_deltas).reshape(-1, 1)


def _merge_multimodal_embeddings(
    inputs_embeds: jax.Array,
    is_multimodal: jax.Array,
    multimodal_embeddings: jax.Array,
) -> jax.Array:
    """Merge multimodal embeddings into text embeddings at placeholder positions.

    Args:
        inputs_embeds: Text embeddings with shape (batch, seq_len, hidden)
        is_multimodal: Boolean mask with shape (batch, seq_len)
        multimodal_embeddings: Flattened vision embeddings with shape (total_tokens, hidden)

    Returns:
        Merged embeddings with shape (batch, seq_len, hidden)
    """
    batch_size, seq_len, hidden = inputs_embeds.shape

    flat_embeds = inputs_embeds.reshape(-1, hidden)
    flat_mask = is_multimodal.reshape(-1)

    dummy_row = jnp.zeros_like(multimodal_embeddings[0:1])
    flattened_padded = jnp.concatenate([dummy_row, multimodal_embeddings], axis=0)

    gather_indices = jnp.cumsum(flat_mask)
    update_values = flattened_padded[gather_indices]

    condition = jnp.expand_dims(flat_mask, axis=-1)
    merged = jnp.where(condition, update_values, flat_embeds)

    return merged.reshape(batch_size, seq_len, hidden)


def merge_multimodal_embeddings(
    input_ids: jax.Array,
    inputs_embeds: jax.Array,
    multimodal_embeddings: jax.Array,
    placeholder_token_id: int | list[int],
) -> jax.Array:
    """Overwrite inputs_embeds wherever input_ids matches placeholder tokens."""
    if isinstance(placeholder_token_id, list):
        placeholder_token_id = jnp.array(placeholder_token_id)
        is_multimodal = jnp.isin(input_ids, placeholder_token_id)
    else:
        is_multimodal = input_ids == placeholder_token_id
    return _merge_multimodal_embeddings(inputs_embeds, is_multimodal, multimodal_embeddings)


def rotate_half(x: Array) -> Array:
    """Rotate half the hidden dims of the input for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision(q: Array, k: Array, cos: Array, sin: Array) -> tuple[Array, Array]:
    """Apply rotary positional embeddings to vision features."""
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.astype("f4"), k.astype("f4")
    cos, sin = jnp.expand_dims(cos, -2).astype("f4"), jnp.expand_dims(sin, -2).astype("f4")
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.astype(orig_q_dtype)
    k_embed = k_embed.astype(orig_k_dtype)
    return q_embed, k_embed


def create_attention_mask(cu_seqlens: Array, seq_length: int, dtype: jnp.dtype) -> Array:
    """Create block-diagonal attention mask from cumulative sequence lengths.

    Vectorized implementation that works correctly with JAX tracing.
    """
    positions = jnp.arange(seq_length)
    starts = cu_seqlens[:-1]
    ends = cu_seqlens[1:]
    in_segment = (positions[:, None] >= starts[None, :]) & (positions[:, None] < ends[None, :])

    segment_ids = jnp.argmax(in_segment.astype(jnp.int32), axis=-1)
    same_segment = segment_ids[:, None] == segment_ids[None, :]
    attention_mask = jnp.where(same_segment, 0.0, jnp.finfo(dtype).min).astype(dtype)

    return attention_mask[None, :, :]


class Qwen3VLVisionPatchEmbed(nn.Module):
    """3D convolution-based patch embedding for Qwen3-VL vision encoder.

    Converts input image/video pixels into patch embeddings using a 3D convolution
    that operates over temporal and spatial dimensions simultaneously.
    """

    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        """Initialize Qwen3-VL vision patch embedding layer.

        Args:
            config (Qwen3VLVisionConfig): Vision encoder configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.dtype = dtype
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.hidden_size = config.hidden_size

        kernel_size = (config.temporal_patch_size, config.patch_size, config.patch_size)
        self.proj = nn.Conv(
            in_features=config.in_channels,
            out_features=config.hidden_size,
            kernel_size=kernel_size,
            strides=kernel_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, hidden_states: Array) -> Array:
        """Apply 3D convolution to extract patch embeddings.

        Args:
            hidden_states (Array): Input pixel values flattened into patches.

        Returns:
            Array: Patch embeddings of shape (num_patches, hidden_size).
        """
        hidden_states = jnp.transpose(
            hidden_states.reshape(
                -1,
                self.in_channels,
                self.temporal_patch_size,
                self.patch_size,
                self.patch_size,
            ),
            (0, 2, 3, 4, 1),
        )
        hidden_states = self.proj(hidden_states.astype(self.dtype))
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        return hidden_states


class Qwen3VLVisionPatchMerger(nn.Module):
    """Spatial patch merger with MLP gating for Qwen3-VL.

    Merges spatially adjacent patches to reduce sequence length while
    preserving important visual information through a gated MLP.
    """

    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        use_postshuffle_norm: bool = False,
        *,
        rngs: nn.Rngs,
    ) -> None:
        """Initialize Qwen3-VL vision patch merger.

        Args:
            config (Qwen3VLVisionConfig): Vision encoder configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            use_postshuffle_norm (bool, optional): Whether to apply normalization after
                spatial shuffling. Defaults to False.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__()
        self.dtype = dtype
        self.spatial_merge_size = config.spatial_merge_size
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm

        self.norm = LayerNorm(
            self.hidden_size if use_postshuffle_norm else config.hidden_size,
            epsilon=1e-6,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.linear_fc1 = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.linear_fc2 = RowParallelLinear(
            self.hidden_size,
            config.out_hidden_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, x: Array) -> Array:
        """Merge patches through normalization and gated MLP.

        Args:
            x (Array): Input patch embeddings.

        Returns:
            Array: Merged patch embeddings with reduced spatial dimensions.
        """
        x = self.norm(x.reshape(-1, self.hidden_size) if self.use_postshuffle_norm else x).reshape(-1, self.hidden_size)
        x = self.linear_fc2(nn.gelu(self.linear_fc1(x), approximate=False))
        return x


class Qwen3VLVisionMLP(nn.Module):
    """Feed-forward network for Qwen3-VL vision encoder.

    Implements a two-layer MLP with GELU activation for vision feature
    transformation within the vision transformer blocks.
    """

    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        layer_idx: int | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        """Initialize Qwen3-VL vision MLP layer.

        Args:
            config (Qwen3VLVisionConfig): Vision encoder configuration.
            layer_idx (int | None, optional): Index of this layer in the encoder.
                Defaults to None.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.linear_fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.act = ACT2FN[config.hidden_act]
        self.linear_fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, x: Array) -> Array:
        """Apply feedforward transformation with GELU activation.

        Args:
            x (Array): Input tensor.

        Returns:
            Array: Transformed tensor with same shape as input.
        """
        return self.linear_fc2(self.act(self.linear_fc1(x)))


class Qwen3VLVisionAttention(UnifiedAttention):
    """Self-attention for Qwen3-VL vision encoder with rotary embeddings.

    Implements multi-head self-attention with 2D rotary position embeddings
    for encoding spatial relationships in vision features.
    """

    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Qwen3-VL vision attention layer.

        Args:
            config (Qwen3VLVisionConfig): Vision encoder configuration.
            layer_idx (int): Index of this layer in the encoder.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
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
            use_fused_qkv=True,
            use_gqa=False,
        )

    def define_network(
        self,
        config: Qwen3VLVisionConfig,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        precision: jax.lax.PrecisionLike,
        rngs: nn.Rngs,
    ) -> None:
        self.qkv = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size * 3,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.attention_performer = self._create_attention_performer(config, rngs)

    def _create_attention_performer(self, config, rngs: nn.Rngs):
        """Create the attention performer module.

        Args:
            config: Vision configuration.
            rngs (nn.Rngs): Random number generator state.

        Returns:
            FlexibleAttentionModule: Configured attention module.
        """
        return FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=0.0,
            attn_mechanism="vanilla",
            requires_cache=False,  # Vision encoder doesn't need KV cache
        )

    def __call__(
        self,
        hidden_states: Array,
        cu_seqlens: Array,
        rotary_pos_emb: Array = None,
    ) -> Array:
        """Apply self-attention with rotary position embeddings.

        Args:
            hidden_states (Array): Input tensor of shape (seq_len, hidden_size).
            cu_seqlens (Array): Cumulative sequence lengths for block-diagonal attention.
            rotary_pos_emb (Array, optional): Precomputed rotary position embeddings.
                Defaults to None.

        Returns:
            Array: Attention output of shape (seq_len, hidden_size).
        """
        seq_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states)
        q, k, v = map(
            lambda x: x.squeeze(0),
            jnp.split(
                qkv.reshape(seq_length, 3, self.num_heads, -1).transpose(1, 0, 2, 3),
                3,
                0,
            ),
        )

        cos = jnp.cos(rotary_pos_emb)
        sin = jnp.sin(rotary_pos_emb)
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        q = jnp.expand_dims(q, 0)
        k = jnp.expand_dims(k, 0)
        v = jnp.expand_dims(v, 0)

        attn_bias = create_attention_mask(cu_seqlens, seq_length, q.dtype)
        attn_bias = jnp.broadcast_to(attn_bias, (1, self.num_heads, seq_length, seq_length))

        attn_output = self.attention_performer.forward(
            query_states=q,
            key_states=k,
            value_states=v,
            bias=attn_bias,
            causal=False,
            mode=common_types.MODE_TRAIN if seq_length != 1 else common_types.MODE_DECODE,
        ).attention_outputs

        attn_output = attn_output.squeeze(0)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = checkpoint_name(self.proj(attn_output), "vision_attn_output")
        return attn_output


class Qwen3VLVisionBlock(nn.Module):
    """Transformer block for Qwen3-VL vision encoder.

    Combines self-attention and MLP layers with pre-normalization
    architecture and residual connections for vision feature processing.
    """

    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        """Initialize Qwen3-VL vision transformer block.

        Args:
            config (Qwen3VLVisionConfig): Vision encoder configuration.
            layer_idx (int): Index of this layer in the encoder.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.norm1 = LayerNorm(
            config.hidden_size,
            epsilon=1e-6,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.norm2 = LayerNorm(
            config.hidden_size,
            epsilon=1e-6,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.attn = Qwen3VLVisionAttention(
            config=config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.mlp = Qwen3VLVisionMLP(
            config=config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: Array,
        cu_seqlens: Array,
        rotary_pos_emb: Array,
    ) -> Array:
        """Forward pass through the vision transformer block.

        Applies pre-normalization: x + attn(norm(x)) followed by x + mlp(norm(x)).

        Args:
            hidden_states (Array): Input tensor of shape (seq_len, hidden_size).
            cu_seqlens (Array): Cumulative sequence lengths for block-diagonal attention.
            rotary_pos_emb (Array): Precomputed rotary position embeddings.

        Returns:
            Array: Output tensor of shape (seq_len, hidden_size).
        """
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


@register_module(TaskType.BASE_VISION, config=Qwen3VLConfig, model_type="qwen3_vl")
class Qwen3VisionTransformerPretrainedModel(EasyDeLBaseModule):
    """Vision transformer encoder for Qwen3-VL.

    Processes images and videos through patch embedding, positional encoding,
    transformer blocks, and patch merging to produce visual features for the
    language model.

    Attributes:
        config_class: Configuration class for the vision encoder.
        patch_embed: 3D convolution-based patch embedding.
        pos_embed: Learnable position embeddings.
        blocks: List of vision transformer blocks.
        merger: Patch merger for reducing spatial dimensions.
        deepstack_merger_list: Additional mergers for deepstack visual features.
    """

    config_class = Qwen3VLVisionConfig

    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Qwen3-VL vision transformer encoder.

        Args:
            config (Qwen3VLVisionConfig): Vision encoder configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.patch_embed = Qwen3VLVisionPatchEmbed(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.pos_embed = Embed(
            num_embeddings=config.num_position_embeddings,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.spatial_merge_size = config.spatial_merge_size
        head_dim = config.hidden_size // config.num_heads
        self._head_dim_ro = head_dim // 2

        self.blocks = nn.List(
            [
                Qwen3VLVisionBlock(
                    config=config,
                    layer_idx=idx,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    precision=precision,
                    rngs=rngs,
                )
                for idx in range(config.depth)
            ]
        )

        self.merger = Qwen3VLVisionPatchMerger(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.deepstack_merger_list = nn.List(
            [
                Qwen3VLVisionPatchMerger(
                    config=config,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    precision=precision,
                    use_postshuffle_norm=True,
                    rngs=rngs,
                )
                for _ in config.deepstack_visual_indexes
            ]
        )

        self.num_grid_per_side = int(math.sqrt(config.num_position_embeddings))

    def get_dtype(self) -> jnp.dtype:
        """Get the data type used by the model parameters.

        Returns:
            jnp.dtype: Data type of the model parameters.
        """
        return self.blocks[0].mlp.linear_fc2.kernel.value.dtype

    def fast_pos_embed_interpolate(self, grid_thw: Array) -> Array:
        """Compute positional embeddings with bilinear interpolation.

        Args:
            grid_thw: Grid dimensions (temporal, height, width) per image, shape (num_images, 3)

        Returns:
            Positional embeddings with shape (total_tokens, hidden_size)
        """
        grid_ts = grid_thw[:, 0]
        grid_hs = grid_thw[:, 1]
        grid_ws = grid_thw[:, 2]
        merge_size = self.spatial_merge_size

        idx_list = [[], [], [], []]
        weight_list = [[], [], [], []]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws, strict=False):
            t, h, w = int(t), int(h), int(w)

            h_idxs = jnp.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = jnp.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = jnp.floor(h_idxs).astype(jnp.int32)
            w_idxs_floor = jnp.floor(w_idxs).astype(jnp.int32)
            h_idxs_ceil = jnp.clip(h_idxs_floor + 1, max=self.num_grid_per_side - 1)
            w_idxs_ceil = jnp.clip(w_idxs_floor + 1, max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[:, None] + w_idxs_floor[None, :]).flatten(),
                (base_h[:, None] + w_idxs_ceil[None, :]).flatten(),
                (base_h_ceil[:, None] + w_idxs_floor[None, :]).flatten(),
                (base_h_ceil[:, None] + w_idxs_ceil[None, :]).flatten(),
            ]

            weights = [
                ((1 - dh)[:, None] * (1 - dw)[None, :]).flatten(),
                ((1 - dh)[:, None] * dw[None, :]).flatten(),
                (dh[:, None] * (1 - dw)[None, :]).flatten(),
                (dh[:, None] * dw[None, :]).flatten(),
            ]

            for i in range(4):
                idx_list[i].append(indices[i])
                weight_list[i].append(weights[i])

        idx_arrays = [jnp.concatenate(idx_list[i], axis=0) for i in range(4)]
        weight_arrays = [jnp.concatenate(weight_list[i], axis=0) for i in range(4)]

        pos_embeds = [self.pos_embed(idx_arrays[i].astype(jnp.int32)) * weight_arrays[i][:, None] for i in range(4)]

        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        splits = [int(h * w) for h, w in zip(grid_hs, grid_ws, strict=False)]
        split_pos = jnp.cumsum(jnp.array(splits[:-1])) if len(splits) > 1 else []
        patch_pos_embeds_list = jnp.split(patch_pos_embeds, split_pos, axis=0)

        patch_pos_embeds_permute = []
        for pos_embed, t, h, w in zip(patch_pos_embeds_list, grid_ts, grid_hs, grid_ws, strict=False):
            t, h, w = int(t), int(h), int(w)
            if t > 1:
                pos_embed = jnp.tile(pos_embed, (t, 1))
            pos_embed = pos_embed.reshape(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
            pos_embed = jnp.transpose(pos_embed, (0, 1, 3, 2, 4, 5))
            pos_embed = pos_embed.reshape(-1, pos_embed.shape[-1])
            patch_pos_embeds_permute.append(pos_embed)

        return jnp.concatenate(patch_pos_embeds_permute, axis=0)

    def rot_pos_emb(self, grid_thw: Array, max_grid_size: int) -> Array:
        """Compute rotary position embeddings for vision features.

        Matches HuggingFace's Qwen3VLVisionModel.rot_pos_emb exactly.
        """
        merge_size = self.spatial_merge_size

        freq_table = jnp.outer(
            jnp.arange(0, max_grid_size, dtype=jnp.float32),
            1.0 / (10000 ** (jnp.arange(0, self._head_dim_ro, 2, dtype=jnp.float32) / self._head_dim_ro)),
        )

        pos_ids_list = []
        for num_frames, height, width in grid_thw:
            num_frames, height, width = int(num_frames), int(height), int(width)
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = jnp.arange(merged_h)
            block_cols = jnp.arange(merged_w)
            intra_row = jnp.arange(merge_size)
            intra_col = jnp.arange(merge_size)

            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            row_idx = jnp.broadcast_to(row_idx, (merged_h, merged_w, merge_size, merge_size)).reshape(-1)

            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
            col_idx = jnp.broadcast_to(col_idx, (merged_h, merged_w, merge_size, merge_size)).reshape(-1)

            coords = jnp.stack([row_idx, col_idx], axis=-1)

            if num_frames > 1:
                coords = jnp.tile(coords, (num_frames, 1))

            pos_ids_list.append(coords)

        pos_ids = jnp.concatenate(pos_ids_list, axis=0)

        embeddings = freq_table[pos_ids]
        embeddings = embeddings.reshape(pos_ids.shape[0], -1)

        return embeddings

    def __call__(
        self,
        hidden_states: Array,
        grid_thw: Array,
        max_grid_size: int,
    ) -> tuple[Array, list[Array]]:
        """Forward pass through the vision transformer encoder.

        Processes input pixels through patch embedding, adds positional embeddings,
        applies transformer blocks with rotary embeddings, and merges patches.

        Args:
            hidden_states (Array): Input pixel values.
            grid_thw (Array): Grid dimensions (temporal, height, width) per image/video.
            max_grid_size (int): Maximum grid size for positional embeddings.

        Returns:
            tuple[Array, list[Array]]: Tuple of (merged_features, deepstack_features).
                merged_features: Final vision embeddings after patch merging.
                deepstack_features: Intermediate features from specified layers.
        """
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds
        rotary_pos_emb = self.rot_pos_emb(grid_thw, max_grid_size)
        rotary_pos_emb = jnp.concatenate([rotary_pos_emb, rotary_pos_emb], axis=-1)

        grid_lens = grid_thw[:, 1] * grid_thw[:, 2]
        repeated = jnp.repeat(grid_lens, grid_thw[:, 0])
        cu_seqlens = jnp.cumsum(repeated, dtype="i4")
        cu_seqlens = jnp.pad(cu_seqlens, (1, 0), constant_values=0)

        deepstack_feature_lists = []
        for layer_num, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
            )
            if layer_num in self.config.deepstack_visual_indexes:
                deepstack_idx = self.config.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = self.deepstack_merger_list[deepstack_idx](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.merger(hidden_states)
        return hidden_states, deepstack_feature_lists

    def get_encoder(self):
        """Get the encoder component.

        Returns:
            Qwen3VisionTransformerPretrainedModel: The vision encoder itself.
        """
        return self

    def get_decoder(self):
        """Get the decoder component.

        Raises:
            NotImplementedError: Vision model does not have a decoder.
        """
        raise NotImplementedError("Vision model does not have a decoder.")

    def get_lm_head(self):
        """Get the language model head.

        Raises:
            NotImplementedError: Vision model does not have a language model head.
        """
        raise NotImplementedError("Vision model does not have a language model head.")

    def get_embedding(self):
        """Get the embedding layer.

        Returns:
            Qwen3VLVisionPatchEmbed: The patch embedding layer.
        """
        return self.patch_embed


class Qwen3VLTextMLP(nn.Module):
    """SwiGLU feed-forward network for Qwen3-VL text decoder.

    Implements the feedforward network with SwiGLU activation function
    for enhanced representation learning in the text decoder.
    """

    def __init__(
        self,
        config: Qwen3VLTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        """Initialize Qwen3-VL text MLP block.

        Args:
            config (Qwen3VLTextConfig): Text decoder configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the decoder.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.layer_idx = layer_idx

        column_linear = partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        row_linear = partial(
            RowParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )

        self.gate_proj = column_linear(config.hidden_size, config.intermediate_size, rngs=rngs)
        self.up_proj = column_linear(config.hidden_size, config.intermediate_size, rngs=rngs)
        self.down_proj = row_linear(config.intermediate_size, config.hidden_size, rngs=rngs)
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(self, hidden_states: Array) -> Array:
        """Apply SwiGLU feedforward transformation.

        Args:
            hidden_states (Array): Input tensor of shape (batch, seq_len, hidden_dim).

        Returns:
            Array: Transformed hidden states of shape (batch, seq_len, hidden_dim).
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        gate = checkpoint_name(self.act_fn(self.gate_proj(hidden_states)), "mlp_gate")
        up = checkpoint_name(self.up_proj(hidden_states), "mlp_up")
        hidden_states = checkpoint_name(self.down_proj(gate * up), "mlp_down")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return checkpoint_name(hidden_states, "mlp_output")


class Qwen3VLTextAttention(UnifiedAttention):
    """Causal self-attention for Qwen3-VL text decoder with mRoPE and QK-norm support.

    This attention module supports multi-dimensional rotary position embeddings (mRoPE)
    for proper handling of multimodal inputs with temporal, height, and width dimensions.
    """

    def __init__(
        self,
        config: Qwen3VLTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        """Initialize Qwen3-VL text attention layer.

        Args:
            config (Qwen3VLTextConfig): Text decoder configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the decoder.
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
            sliding_window=config.sliding_window if config.use_sliding_window else None,
            use_qk_norm=True,
        )

    def _postprocess_qkv(self, query_states, key_states, value_states):
        """Apply Q/K normalization after computing query, key, and value projections.

        Args:
            query_states: Query tensor from projection layer.
            key_states: Key tensor from projection layer.
            value_states: Value tensor from projection layer.

        Returns:
            Tuple of normalized query, normalized key, and value tensors.
        """
        return self.query_normalization(query_states), self.key_normalization(key_states), value_states


class Qwen3VLTextDecoderLayer(nn.Module):
    """Transformer decoder layer for Qwen3-VL text model.

    Combines multi-head attention with Q/K normalization and feedforward networks
    with RMS normalization and residual connections.
    """

    def __init__(
        self,
        config: Qwen3VLTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        """Initialize Qwen3-VL text decoder layer.

        Args:
            config (Qwen3VLTextConfig): Text decoder configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the decoder.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.layer_idx = layer_idx

        attn_block = Qwen3VLTextAttention
        mlp_block = Qwen3VLTextMLP
        attn_block, mlp_block = auto_remat(
            attn_block,
            mlp_block,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )

        self.self_attn = attn_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )

        self.mlp = mlp_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
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

    def __call__(
        self,
        hidden_states: Array,
        mask_info: MaskInfo,
        position_ids: Array,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Array | None = None,
    ) -> DecoderLayerOutput:
        """Forward pass through the decoder layer.

        Applies pre-normalization architecture: x + attn(norm(x)) followed by x + mlp(norm(x)).

        Args:
            hidden_states (Array): Input tensor of shape (batch, seq_len, hidden_dim).
            mask_info (MaskInfo): Attention mask information including causal masks.
            position_ids (Array): Position indices, shape (3, batch, seq_len) for mRoPE.
            mode (RUNTIME_MODE_TYPES): Runtime mode (train, decode, etc.) for optimization.
            cache_view (TransformerCacheView | RaggedPagesCacheView | None, optional): Cache view.
                Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Cache metadata. Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            frequencies (Array | None, optional): Precomputed RoPE frequencies. Defaults to None.

        Returns:
            DecoderLayerOutput: Contains hidden states, attention weights, and cache view.
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

        if self.config.use_scan_mlp:
            feed_forward_hidden_states = block_wise_ffn(
                self.mlp,
                feed_forward_input,
                self.config.scan_mlp_chunk_size,
            )
        else:
            feed_forward_hidden_states = self.mlp(feed_forward_input)

        hidden_states = checkpoint_name(hidden_states + feed_forward_hidden_states, "residual")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        return DecoderLayerOutput(
            hidden_states=checkpoint_name(hidden_states, "layer_output"),
            attention_weight=attn_outputs.attention_weight,
            cache_view=attn_outputs.cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=Qwen3VLTextConfig, model_type="qwen3_vl")
class Qwen3VLTextModel(EasyDeLBaseModule):
    """Text decoder model for Qwen3-VL.

    Implements the text decoder component of Qwen3-VL, utilizing transformer blocks
    with RMSNorm, Q/K normalization, and multi-dimensional rotary position embeddings
    (mRoPE) for handling multimodal inputs.

    Attributes:
        config: Configuration for the text decoder.
        embed_tokens: Token embedding layer.
        layers: List of transformer decoder layers.
        norm: Final layer normalization.
    """

    def __init__(
        self,
        config: Qwen3VLTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Qwen3-VL text decoder model.

        Args:
            config (Qwen3VLTextConfig): Text decoder configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            rngs=rngs,
        )

        self.layers = nn.List(
            [
                Qwen3VLTextDecoderLayer(
                    config=config,
                    layer_idx=i,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    precision=precision,
                    rngs=rngs,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    @cached_property
    def frequencies(self):
        """Get cached RoPE frequency tensor from config.

        Computes and caches the rotary position embedding frequencies
        based on the model's head dimension and rope_theta configuration.

        Returns:
            Array: Precomputed RoPE frequencies for position encoding.
        """
        head_dim = getattr(self.config, "qk_rope_head_dim", None) or self.config.head_dim
        return self.config.get_basic_frequencies(
            head_size=head_dim,
            rotary_dim=head_dim,
            base=self.config.rope_theta,
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        visual_pos_masks: Bool[Array, "batch seq_len"] | None = None,
        deepstack_visual_embeds: list[Array] | None = None,
    ) -> BaseModelOutput:
        """Forward pass through the Qwen3-VL text decoder model.

        Processes input tokens through embedding, all decoder layers with mRoPE
        and RMSNorm, and final normalization. Supports deepstack visual embedding
        injection at intermediate layers.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch, seq_len).
                Must be provided if inputs_embeds is None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings of shape
                (batch, seq_len, hidden_size). Defaults to None.
            attention_mask (Array | None, optional): Boolean mask to avoid attention on
                padding tokens, shape (batch, seq_len). Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information for attention.
                Defaults to None.
            position_ids (Array | None, optional): 3D position indices for mRoPE,
                shape (3, batch, seq_len). Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode (train/decode).
                Auto-detected if None. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cache with precomputed key-value states. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache management. Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention weights.
                Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return all hidden states.
                Defaults to None.
            visual_pos_masks (Array | None, optional): Mask indicating visual token positions.
                Defaults to None.
            deepstack_visual_embeds (list[Array] | None, optional): Intermediate visual
                embeddings for deepstack fusion. Defaults to None.

        Returns:
            BaseModelOutput: Contains last_hidden_state, optional all hidden_states,
                optional attentions, and updated past_key_values.

        Raises:
            ValueError: If both input_ids and inputs_embeds are provided or both are None.
            AssertionError: If sequence_length exceeds max_position_embeddings.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify either input_ids or inputs_embeds, but not both.")

        if inputs_embeds is None:
            inputs_embeds = checkpoint_name(self.embed_tokens(input_ids.astype("i4")), "embeddings")

        sequence_length = inputs_embeds.shape[1]

        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        assert sequence_length <= self.config.max_position_embeddings, (
            f"Maximum Position Embedding Reached! "
            f"(Expected <= {self.config.max_position_embeddings} got {sequence_length})"
        )

        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        attention_mask = mask_info.attention_mask

        if position_ids is None:
            batch_size = inputs_embeds.shape[0]
            pos_2d = mask_info.q_position_ids
            position_ids = jnp.broadcast_to(pos_2d[None, :, :], (3, batch_size, sequence_length))

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
            partition_manager=self.config.partition_manager,
        )

        for idx, block in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = block(
                hidden_states=hidden_states,
                mask_info=mask_info,
                position_ids=position_ids,
                mode=mode,
                cache_view=past_key_values.views[idx],
                cache_metadata=cache_metadata,
                output_attentions=output_attentions,
                frequencies=self.frequencies,
            )
            hidden_states = layer_outputs.hidden_states
            if deepstack_visual_embeds is not None and idx < len(deepstack_visual_embeds):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[idx],
                )

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            past_key_values[idx] = layer_outputs.cache_view

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

    def _deepstack_process(
        self,
        hidden_states: Array,
        visual_pos_masks: Array,
        visual_embeds: Array,
    ) -> Array:
        """Add visual embeddings to hidden states at visual positions."""
        visual_embeds = visual_embeds.astype(hidden_states.dtype)
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, hidden_dim)
        flat_mask = visual_pos_masks.reshape(-1)
        cumsum_mask = jnp.cumsum(flat_mask) - 1
        cumsum_mask = jnp.where(flat_mask, cumsum_mask, 0)

        visual_updates = visual_embeds[cumsum_mask]

        flat_hidden = jnp.where(flat_mask[:, None], flat_hidden + visual_updates, flat_hidden)

        return flat_hidden.reshape(batch_size, seq_len, hidden_dim)

    def get_encoder(self):
        """Get the encoder component.

        Raises:
            NotImplementedError: Text model does not have an encoder.
        """
        raise NotImplementedError("Text model does not have an encoder.")

    def get_decoder(self):
        """Get the decoder component.

        Returns:
            Qwen3VLTextModel: The text decoder itself.
        """
        return self

    def get_lm_head(self):
        """Get the language model head.

        Raises:
            NotImplementedError: Base model does not have a language model head.
        """
        raise NotImplementedError("Base model does not have a language model head.")

    def get_embedding(self):
        """Get the token embedding layer.

        Returns:
            Embed: The token embedding layer.
        """
        return self.embed_tokens


@register_module(TaskType.VISION_LM, config=Qwen3VLConfig, model_type="qwen3_vl")
class Qwen3VLModel(EasyDeLBaseModule):
    """Qwen3-VL multimodal model integrating vision encoder and text decoder.

    This model handles the fusion of visual and textual information by:
    1. Processing images/videos through the vision encoder
    2. Merging visual embeddings into text embeddings at placeholder positions
    3. Computing 3D position IDs for proper mRoPE handling
    4. Running the fused embeddings through the text decoder

    Architecture matches HuggingFace:
    - visual: Qwen3VisionTransformerPretrainedModel
    - language_model: Qwen3VLTextModel
    """

    def __init__(
        self,
        config: Qwen3VLConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Qwen3-VL multimodal model.

        Args:
            config (Qwen3VLConfig): Multimodal model configuration containing
                vision_config and text_config.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.visual = Qwen3VisionTransformerPretrainedModel(
            config.vision_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.language_model = Qwen3VLTextModel(
            config.text_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.rope_deltas = None

    def get_input_embeddings(self):
        """Get the input embedding layer.

        Returns:
            Embed: The token embedding layer from the language model.
        """
        return self.language_model.get_embedding()

    def set_input_embeddings(self, value):
        """Set the input embedding layer.

        Args:
            value: New embedding layer to use.
        """
        self.language_model.embed_tokens = value

    def set_decoder(self, decoder):
        """Set the language model decoder.

        Args:
            decoder: New decoder model to use.
        """
        self.language_model = decoder

    def get_decoder(self):
        """Get the language model decoder.

        Returns:
            Qwen3VLTextModel: The language model component.
        """
        return self.language_model

    def get_rope_index(
        self,
        input_ids: Array,
        image_grid_thw: Array | None = None,
        video_grid_thw: Array | None = None,
        attention_mask: Array | None = None,
    ) -> tuple[Array, Array]:
        """Calculate 3D RoPE indices for multimodal inputs.

        Different from Qwen2-VL, Qwen3-VL uses timestamps rather than absolute
        time position IDs for videos.

        Args:
            input_ids: Token IDs of shape (batch_size, sequence_length).
            image_grid_thw: Temporal/height/width grid for images.
            video_grid_thw: Temporal/height/width grid for videos.
            attention_mask: Attention mask for padding.

        Returns:
            Tuple of (position_ids, mrope_position_deltas).
            position_ids has shape (3, batch_size, sequence_length).
        """
        tokens_per_second = getattr(self.config.vision_config, "tokens_per_second", 2.0)
        return get_rope_index(
            input_ids=input_ids,
            image_grid_thw=np.array(image_grid_thw) if image_grid_thw is not None else None,
            video_grid_thw=np.array(video_grid_thw) if video_grid_thw is not None else None,
            attention_mask=attention_mask,
            spatial_merge_size=self.config.vision_config.spatial_merge_size,
            image_token_id=self.config.image_token_id,
            video_token_id=self.config.video_token_id,
            vision_start_token_id=self.config.vision_start_token_id,
            tokens_per_second=tokens_per_second,
        )

    def compute_embedding(
        self,
        input_ids: Int[Array, "batch seq_len"],
        *,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        pixel_values: Array | None = None,
        pixel_values_videos: Array | None = None,
        image_grid_thw: Array | None = None,
        video_grid_thw: Array | None = None,
        image_max_grid_size: int | None = None,
        video_max_grid_size: int | None = None,
        image_embeds: Array | None = None,
        video_embeds: Array | None = None,
        **kwargs,
    ) -> Array:
        """Compute embeddings with multimodal fusion.

        Merges visual embeddings from images/videos into text embeddings
        at placeholder token positions.

        Args:
            input_ids (Array): Input token IDs of shape (batch, seq_len).
            inputs_embeds (Array | None, optional): Pre-computed embeddings. Defaults to None.
            pixel_values (Array | None, optional): Image pixel values. Defaults to None.
            pixel_values_videos (Array | None, optional): Video pixel values. Defaults to None.
            image_grid_thw (Array | None, optional): Image grid dimensions. Defaults to None.
            video_grid_thw (Array | None, optional): Video grid dimensions. Defaults to None.
            image_max_grid_size (int | None, optional): Max image grid size. Defaults to None.
            video_max_grid_size (int | None, optional): Max video grid size. Defaults to None.
            image_embeds (Array | None, optional): Pre-computed image embeddings. Defaults to None.
            video_embeds (Array | None, optional): Pre-computed video embeddings. Defaults to None.
            **kwargs: Additional arguments.

        Returns:
            Array: Fused embeddings with visual content merged at placeholder positions.

        Raises:
            ValueError: If input_ids is None when required.
        """
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("`input_ids` must be provided when calling `compute_embedding`.")
            inputs_embeds = super().compute_embedding(input_ids)

        if input_ids is None and (image_embeds is not None or video_embeds is not None):
            raise ValueError("`input_ids` must be provided to merge multimodal embeddings.")

        if image_embeds is None and pixel_values is not None:
            image_embeds_tuple, _deepstack_image_embeds = self.get_image_features(
                pixel_values,
                image_grid_thw,
                image_max_grid_size,
            )
            image_embeds = jnp.concatenate(image_embeds_tuple, axis=0)

        if image_embeds is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=image_embeds.astype(inputs_embeds.dtype),
                placeholder_token_id=self.config.image_token_id,
            )

        if video_embeds is None and pixel_values_videos is not None:
            video_embeds_tuple, _deepstack_video_embeds = self.get_video_features(
                pixel_values_videos,
                video_grid_thw,
                video_max_grid_size,
            )
            video_embeds = jnp.concatenate(video_embeds_tuple, axis=0)

        if video_embeds is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=video_embeds.astype(inputs_embeds.dtype),
                placeholder_token_id=self.config.video_token_id,
            )

        return inputs_embeds

    def compute_embedding_with_info(
        self,
        input_ids: Int[Array, "batch seq_len"],
        *,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        pixel_values: Array | None = None,
        pixel_values_videos: Array | None = None,
        image_grid_thw: Array | None = None,
        video_grid_thw: Array | None = None,
        image_max_grid_size: int | None = None,
        video_max_grid_size: int | None = None,
        image_embeds: Array | None = None,
        video_embeds: Array | None = None,
        attention_mask: Array | None = None,
        **kwargs,
    ) -> tuple[Array, EmbeddingInfo]:
        """Compute embeddings with multimodal fusion and return additional info.

        Similar to compute_embedding but also returns EmbeddingInfo containing
        position IDs, rope deltas, and deepstack visual embeddings.

        Args:
            input_ids (Array): Input token IDs of shape (batch, seq_len).
            inputs_embeds (Array | None, optional): Pre-computed embeddings. Defaults to None.
            pixel_values (Array | None, optional): Image pixel values. Defaults to None.
            pixel_values_videos (Array | None, optional): Video pixel values. Defaults to None.
            image_grid_thw (Array | None, optional): Image grid dimensions. Defaults to None.
            video_grid_thw (Array | None, optional): Video grid dimensions. Defaults to None.
            image_max_grid_size (int | None, optional): Max image grid size. Defaults to None.
            video_max_grid_size (int | None, optional): Max video grid size. Defaults to None.
            image_embeds (Array | None, optional): Pre-computed image embeddings. Defaults to None.
            video_embeds (Array | None, optional): Pre-computed video embeddings. Defaults to None.
            attention_mask (Array | None, optional): Attention mask. Defaults to None.
            **kwargs: Additional arguments.

        Returns:
            tuple[Array, EmbeddingInfo]: Tuple of (fused_embeddings, embedding_info).

        Raises:
            ValueError: If input_ids is None.
        """
        if input_ids is None:
            raise ValueError("`input_ids` must be provided when calling `compute_embedding_with_info`.")

        if inputs_embeds is None:
            inputs_embeds = super().compute_embedding(input_ids)

        text_embeds = inputs_embeds

        deepstack_image_embeds = None
        deepstack_video_embeds = None

        if image_embeds is None and pixel_values is not None:
            image_embeds_tuple, deepstack_image_embeds = self.get_image_features(
                pixel_values,
                image_grid_thw,
                image_max_grid_size,
            )
            image_embeds = jnp.concatenate(image_embeds_tuple, axis=0)

        if video_embeds is None and pixel_values_videos is not None:
            video_embeds_tuple, deepstack_video_embeds = self.get_video_features(
                pixel_values_videos,
                video_grid_thw,
                video_max_grid_size,
            )
            video_embeds = jnp.concatenate(video_embeds_tuple, axis=0)

        if image_embeds is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=image_embeds.astype(inputs_embeds.dtype),
                placeholder_token_id=self.config.image_token_id,
            )

        if video_embeds is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=video_embeds.astype(inputs_embeds.dtype),
                placeholder_token_id=self.config.video_token_id,
            )

        visual_pos_masks = None
        deepstack_visual_embeds = None

        has_deepstack_images = deepstack_image_embeds is not None and len(deepstack_image_embeds) > 0
        has_deepstack_videos = deepstack_video_embeds is not None and len(deepstack_video_embeds) > 0
        if has_deepstack_images or has_deepstack_videos:
            image_mask, video_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=text_embeds,
                image_features=image_embeds.astype(text_embeds.dtype) if image_embeds is not None else None,
                video_features=video_embeds.astype(text_embeds.dtype) if video_embeds is not None else None,
            )

            if image_embeds is not None and video_embeds is not None and has_deepstack_images and has_deepstack_videos:
                visual_pos_masks = image_mask | video_mask
                deepstack_visual_embeds = []

                image_mask_joint = image_mask[visual_pos_masks]
                video_mask_joint = video_mask[visual_pos_masks]

                for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds, strict=False):
                    embed_joint = jnp.zeros((image_mask_joint.shape[0], img_embed.shape[-1]), dtype=img_embed.dtype)

                    img_idx = jnp.cumsum(image_mask_joint) - 1
                    img_idx = jnp.where(image_mask_joint, img_idx, 0)
                    embed_joint = jnp.where(image_mask_joint[:, None], img_embed[img_idx], embed_joint)

                    vid_idx = jnp.cumsum(video_mask_joint) - 1
                    vid_idx = jnp.where(video_mask_joint, vid_idx, 0)
                    embed_joint = jnp.where(video_mask_joint[:, None], vid_embed[vid_idx], embed_joint)

                    deepstack_visual_embeds.append(embed_joint)
            elif image_embeds is not None and has_deepstack_images:
                visual_pos_masks = image_mask
                deepstack_visual_embeds = deepstack_image_embeds
            elif video_embeds is not None and has_deepstack_videos:
                visual_pos_masks = video_mask
                deepstack_visual_embeds = deepstack_video_embeds

        position_ids, rope_deltas = self.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )

        info = EmbeddingInfo(
            position_ids=position_ids,
            rope_deltas=rope_deltas,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            deepstack_image_embeds=deepstack_image_embeds,
            deepstack_video_embeds=deepstack_video_embeds,
        )
        return inputs_embeds, info

    def get_video_features(
        self,
        pixel_values_videos: Array,
        video_grid_thw: Array | None = None,
        video_max_grid_size: int | None = None,
    ) -> tuple[tuple[Array, ...], list[Array]]:
        """Encodes videos into continuous embeddings.

        The deepstack visual features are also returned.

        Args:
            pixel_values_videos: The tensors corresponding to the input videos.
            video_grid_thw: The temporal, height and width of feature shape
                of each video in LLM.
            video_max_grid_size: Maximum grid size for videos.

        Returns:
            Tuple of (video_embeds_tuple, deepstack_video_embeds)
        """
        return self.get_image_features(pixel_values_videos, video_grid_thw, video_max_grid_size)

    def get_image_features(
        self,
        pixel_values: Array,
        image_grid_thw: Array | None = None,
        image_max_grid_size: int | None = None,
    ) -> tuple[tuple[Array, ...], list[Array]]:
        """Encodes images into continuous embeddings.

        The deepstack visual features are also returned.

        Args:
            pixel_values: The tensors corresponding to the input images.
            image_grid_thw: The temporal, height and width of feature shape
                of each image in LLM.
            image_max_grid_size: Maximum grid size for images.

        Returns:
            Tuple of (image_embeds_tuple, deepstack_image_embeds)
        """
        pixel_values = pixel_values.astype(self.visual.get_dtype())

        if image_max_grid_size is None and image_grid_thw is not None:
            image_max_grid_size = int(np.array(image_grid_thw)[:, 1:].max())

        grid_thw = np.array(image_grid_thw) if image_grid_thw is not None else None

        image_embeds, deepstack_image_embeds = self.visual(
            pixel_values,
            grid_thw=grid_thw,
            max_grid_size=image_max_grid_size,
        )

        split_sizes = (np.prod(grid_thw, axis=-1) // (self.visual.spatial_merge_size**2)).tolist()
        split_points = np.cumsum(split_sizes[:-1]) if len(split_sizes) > 1 else []
        image_embeds_tuple = tuple(jnp.split(image_embeds, split_points, axis=0))

        return image_embeds_tuple, deepstack_image_embeds

    def get_placeholder_mask(
        self,
        input_ids: Array | None,
        inputs_embeds: Array,
        image_features: Array | None = None,
        video_features: Array | None = None,
    ) -> tuple[Array, Array]:
        """Obtains multimodal placeholder mask from input_ids or inputs_embeds.

        Checks that the placeholder token count equals the length of multimodal features.
        If the lengths are different, an error is raised.

        Args:
            input_ids: Input token IDs.
            inputs_embeds: Input embeddings.
            image_features: Image features to validate against mask.
            video_features: Video features to validate against mask.

        Returns:
            Tuple of (special_image_mask, special_video_mask).
        """
        if input_ids is None:
            image_embed_ref = self.get_input_embeddings()(jnp.array(self.config.image_token_id, dtype=jnp.int32))
            special_image_mask = jnp.all(inputs_embeds == image_embed_ref, axis=-1)

            video_embed_ref = self.get_input_embeddings()(jnp.array(self.config.video_token_id, dtype=jnp.int32))
            special_video_mask = jnp.all(inputs_embeds == video_embed_ref, axis=-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask_expanded = jnp.broadcast_to(special_image_mask[..., None], inputs_embeds.shape)
        if image_features is not None:
            image_feature_size = image_features.size
            masked_size = inputs_embeds[special_image_mask_expanded].size
            if masked_size != image_feature_size:
                raise ValueError(
                    f"Image features and image tokens do not match: "
                    f"tokens: {n_image_tokens}, features {image_features.shape[0]}"
                )

        n_video_tokens = special_video_mask.sum()
        special_video_mask_expanded = jnp.broadcast_to(special_video_mask[..., None], inputs_embeds.shape)
        if video_features is not None:
            video_feature_size = video_features.size
            masked_size = inputs_embeds[special_video_mask_expanded].size
            if masked_size != video_feature_size:
                raise ValueError(
                    f"Video features and video tokens do not match: "
                    f"tokens: {n_video_tokens}, features {video_features.shape[0]}"
                )

        return special_image_mask, special_video_mask

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        visual_pos_masks: Bool[Array, "batch seq_len"] | None = None,
        deepstack_visual_embeds: list[Array] | None = None,
        pixel_values: Array | None = None,
        pixel_values_videos: Array | None = None,
        image_grid_thw: tuple | None = None,
        video_grid_thw: tuple | None = None,
        image_max_grid_size: int | None = None,
        video_max_grid_size: int | None = None,
        cache_position: Array | None = None,
    ) -> BaseModelOutput:
        """Forward pass through the Qwen3-VL multimodal model.

        Processes multimodal inputs by:
        1. Encoding images/videos through the vision encoder
        2. Merging visual embeddings into text embeddings at placeholder positions
        3. Computing 3D position IDs for proper mRoPE handling
        4. Running the fused embeddings through the text decoder

        Args:
            input_ids (Array | None, optional): Input token IDs. Defaults to None.
            inputs_embeds (Array | None, optional): Pre-computed embeddings. Defaults to None.
            attention_mask (Array | None, optional): Attention mask. Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information. Defaults to None.
            position_ids (Array | None, optional): 3D position IDs for mRoPE. Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode. Defaults to None.
            past_key_values (Cache | None, optional): KV cache for generation. Defaults to None.
            cache_metadata (Metadata | None, optional): Cache metadata. Defaults to None.
            output_attentions (bool | None, optional): Return attention weights. Defaults to None.
            output_hidden_states (bool | None, optional): Return hidden states. Defaults to None.
            visual_pos_masks (Array | None, optional): Visual token position mask. Defaults to None.
            deepstack_visual_embeds (list[Array] | None, optional): Deepstack embeddings.
                Defaults to None.
            pixel_values (Array | None, optional): Image pixels. Defaults to None.
            pixel_values_videos (Array | None, optional): Video pixels. Defaults to None.
            image_grid_thw (tuple | None, optional): Image grid dimensions. Defaults to None.
            video_grid_thw (tuple | None, optional): Video grid dimensions. Defaults to None.
            image_max_grid_size (int | None, optional): Max image grid size. Defaults to None.
            video_max_grid_size (int | None, optional): Max video grid size. Defaults to None.
            cache_position (Array | None, optional): Cache position. Defaults to None.

        Returns:
            Qwen3VLModelOutputWithPast: Model outputs with hidden states and rope deltas.

        Raises:
            ValueError: If neither or both input_ids and inputs_embeds are provided.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        precomputed_visual_pos_masks = visual_pos_masks
        precomputed_deepstack_visual_embeds = deepstack_visual_embeds

        if inputs_embeds is None:
            inputs_embeds = self.compute_embedding(input_ids)

        image_mask = None
        video_mask = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None
        image_embeds = None
        video_embeds = None

        if pixel_values is not None:
            image_embeds_tuple, deepstack_image_embeds = self.get_image_features(
                pixel_values,
                image_grid_thw,
                image_max_grid_size,
            )
            image_embeds = jnp.concatenate(image_embeds_tuple, axis=0).astype(inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )

        if pixel_values_videos is not None:
            video_embeds_tuple, deepstack_video_embeds = self.get_video_features(
                pixel_values_videos,
                video_grid_thw,
                video_max_grid_size,
            )
            video_embeds = jnp.concatenate(video_embeds_tuple, axis=0).astype(inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                video_features=video_embeds,
            )

        if image_embeds is not None or video_embeds is not None:
            inputs_embeds = self.compute_embedding(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_embeds=image_embeds,
                video_embeds=video_embeds,
            )

        computed_visual_pos_masks = None
        computed_deepstack_visual_embeds = None

        if image_mask is not None and video_mask is not None:
            computed_visual_pos_masks = image_mask | video_mask
            computed_deepstack_visual_embeds = []

            image_mask_joint = image_mask[computed_visual_pos_masks]
            video_mask_joint = video_mask[computed_visual_pos_masks]

            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds, strict=False):
                embed_joint = jnp.zeros((image_mask_joint.shape[0], img_embed.shape[-1]), dtype=img_embed.dtype)

                img_idx = jnp.cumsum(image_mask_joint) - 1
                img_idx = jnp.where(image_mask_joint, img_idx, 0)
                embed_joint = jnp.where(image_mask_joint[:, None], img_embed[img_idx], embed_joint)

                vid_idx = jnp.cumsum(video_mask_joint) - 1
                vid_idx = jnp.where(video_mask_joint, vid_idx, 0)
                embed_joint = jnp.where(video_mask_joint[:, None], vid_embed[vid_idx], embed_joint)

                computed_deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            computed_visual_pos_masks = image_mask
            computed_deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            computed_visual_pos_masks = video_mask
            computed_deepstack_visual_embeds = deepstack_video_embeds

        if computed_visual_pos_masks is not None:
            visual_pos_masks = computed_visual_pos_masks
            deepstack_visual_embeds = computed_deepstack_visual_embeds
        else:
            visual_pos_masks = precomputed_visual_pos_masks
            deepstack_visual_embeds = precomputed_deepstack_visual_embeds
            if deepstack_visual_embeds is not None and visual_pos_masks is None:
                raise ValueError("`visual_pos_masks` must be provided when `deepstack_visual_embeds` is not None.")

        rope_deltas = None
        if position_ids is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw if pixel_values is not None else None,
                video_grid_thw if pixel_values_videos is not None else None,
                attention_mask=attention_mask,
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
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=rope_deltas,
        )

    def get_encoder(self):
        """Get the vision encoder.

        Returns:
            Qwen3VisionTransformerPretrainedModel: The vision encoder.
        """
        return self.visual

    def get_lm_head(self):
        """Get the language model head.

        Raises:
            NotImplementedError: Qwen3VLModel does not have a language model head.
        """
        raise NotImplementedError("Qwen3VLModel does not have a language model head.")

    def get_embedding(self):
        """Get the token embedding layer.

        Returns:
            Embed: The token embedding layer from the language model.
        """
        return self.language_model.embed_tokens


@register_module(TaskType.IMAGE_TEXT_TO_TEXT, config=Qwen3VLConfig, model_type="qwen3_vl")
class Qwen3VLForConditionalGeneration(BaseVisionLanguageModule[Qwen3VLModel, Qwen3VLConfig]):
    """Qwen3-VL model for conditional generation from images/videos and text.

    Inherits from BaseVisionLanguageModule to leverage common VLM infrastructure.

    Architecture matches HuggingFace:
    - model: Qwen3VLModel (contains visual + language_model)
    - lm_head: Linear projection to vocab

    Class Attributes:
        _task_type: IMAGE_TEXT_TO_TEXT task type
        _model_type: "qwen3_vl" model identifier
        _supports_video: True (Qwen3-VL supports video input)
        _uses_mrope: True (uses multi-dimensional RoPE)
    """

    _task_type = TaskType.IMAGE_TEXT_TO_TEXT
    _model_type = "qwen3_vl"
    _config_class = Qwen3VLConfig
    _auto_register = False
    _supports_video = True
    _uses_mrope = True

    _vision_tower_name = "visual"
    _projector_name = "merger"
    _language_model_name = "language_model"

    loss_type = "ForCausalLM"

    def __init__(
        self,
        config: Qwen3VLConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Qwen3-VL model for conditional generation.

        Args:
            config (Qwen3VLConfig): Multimodal model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=Qwen3VLModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            vision_feature_layer=-1,
            vision_feature_select_strategy="default",
            image_token_index=config.image_token_id,
            video_token_index=config.video_token_id,
            spatial_merge_size=config.vision_config.spatial_merge_size,
            tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
            lm_head_bias=False,
        )
        self.vocab_size = config.text_config.vocab_size

    def get_input_embeddings(self):
        """Get the input embedding layer.

        Returns:
            Embed: The token embedding layer.
        """
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        """Set the input embedding layer.

        Args:
            value: New embedding layer to use.
        """
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        """Set the language model decoder.

        Args:
            decoder: New decoder model to use.
        """
        self.model.set_decoder(decoder)

    def get_decoder(self):
        """Get the language model decoder.

        Returns:
            Qwen3VLTextModel: The language model component.
        """
        return self.model.get_decoder()

    @property
    def visual(self):
        """Property to access the vision transformer for backward compatibility."""
        return self.model.visual

    @property
    def language_model(self):
        """Property to access the language model for backward compatibility."""
        return self.model.language_model

    def get_video_features(
        self,
        pixel_values_videos: Array,
        video_grid_thw: Array | None = None,
        video_max_grid_size: int | None = None,
    ) -> tuple[tuple[Array, ...], list[Array]]:
        """Encode videos into continuous embeddings.

        Args:
            pixel_values_videos (Array): Video pixel values.
            video_grid_thw (Array | None, optional): Video grid dimensions. Defaults to None.
            video_max_grid_size (int | None, optional): Max grid size. Defaults to None.

        Returns:
            tuple[tuple[Array, ...], list[Array]]: Video embeddings and deepstack features.
        """
        return self.model.get_video_features(pixel_values_videos, video_grid_thw, video_max_grid_size)

    def get_image_features(
        self,
        pixel_values: Array,
        image_grid_thw: Array | None = None,
        image_max_grid_size: int | None = None,
    ) -> tuple[tuple[Array, ...], list[Array]]:
        """Encode images into continuous embeddings.

        Args:
            pixel_values (Array): Image pixel values.
            image_grid_thw (Array | None, optional): Image grid dimensions. Defaults to None.
            image_max_grid_size (int | None, optional): Max grid size. Defaults to None.

        Returns:
            tuple[tuple[Array, ...], list[Array]]: Image embeddings and deepstack features.
        """
        return self.model.get_image_features(pixel_values, image_grid_thw, image_max_grid_size)

    def compute_embedding(self, input_ids, *args, **kwargs):
        """Compute embeddings with multimodal fusion.

        Delegates to the underlying model's compute_embedding method.

        Args:
            input_ids: Input token IDs.
            *args: Positional arguments passed to model.compute_embedding.
            **kwargs: Keyword arguments passed to model.compute_embedding.

        Returns:
            Array: Fused embeddings.
        """
        return self.model.compute_embedding(input_ids, *args, **kwargs)

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        visual_pos_masks: Bool[Array, "batch seq_len"] | None = None,
        deepstack_visual_embeds: list[Array] | None = None,
        pixel_values: Array | None = None,
        pixel_values_videos: Array | None = None,
        image_grid_thw: tuple | None = None,
        video_grid_thw: tuple | None = None,
        cache_position: Array | None = None,
        image_max_grid_size: int | None = None,
        video_max_grid_size: int | None = None,
        **kwargs,
    ) -> VLMCausalLMOutput:
        """Forward pass for the Qwen3-VL model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            mask_info: Mask information
            position_ids: 3D position IDs for mRoPE (3, batch, seq_len)
            mode: Runtime mode
            past_key_values: Cached keys/values
            cache_metadata: Metadata for paged attention
            apply_lm_head: Whether to apply the LM head
            inputs_embeds: Input embeddings
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            pixel_values: Image pixel values
            pixel_values_videos: Video pixel values
            image_grid_thw: Image grid shape for mRoPE
            video_grid_thw: Video grid shape for mRoPE
            cache_position: Cache position for incremental decoding
            image_max_grid_size: Maximum grid size for images
            video_max_grid_size: Maximum grid size for videos
            **kwargs: Additional arguments

        Returns:
            VLMCausalLMOutput: Model outputs including logits and optional states
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            image_max_grid_size=image_max_grid_size,
            video_max_grid_size=video_max_grid_size,
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_state

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        lm_logits = None
        if apply_lm_head:
            lm_logits = checkpoint_name(self.apply_lm_head(hidden_states), "lm_head_output")
            lm_logits = self.apply_logit_cap(lm_logits)

        return VLMCausalLMOutput(
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            last_hidden_state=hidden_states,
            attentions=outputs.attentions,
            rope_deltas=getattr(outputs, "rope_deltas", None),
            image_hidden_states=None,
        )

    def apply_lm_head(self, hidden_states: Array) -> Array:
        """Apply the language modeling head to hidden states.

        Args:
            hidden_states (Array): Hidden states from the model.

        Returns:
            Array: Logits over the vocabulary.
        """
        return self.lm_head(hidden_states)

    def get_vision_tower(self) -> nn.Module:
        """Get the vision tower component.

        Returns:
            Qwen3VisionTransformerPretrainedModel: The vision encoder.
        """
        return self.model.visual

    def get_language_model(self) -> nn.Module:
        """Get the language model component.

        Returns:
            Qwen3VLTextModel: The text decoder.
        """
        return self.model.language_model
