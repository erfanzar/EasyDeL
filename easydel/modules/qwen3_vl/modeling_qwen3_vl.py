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


from functools import partial

import chex
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
from easydel.infra.modeling_outputs import BaseModelOutput, DecoderLayerOutput, ModelOutput
from easydel.infra.utils import ACT2FN, auto_remat, block_wise_ffn, get_dot_general_by_bits
from easydel.layers.attention import FlexibleAttentionModule
from easydel.layers.attention_unified import UnifiedAttention
from easydel.layers.caching import (
    RaggedPagesCache,
    RaggedPagesCacheView,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.layers.linear import ColumnParallelLinear, RowParallelLinear
from easydel.layers.norms import RMSNorm

from .qwen3_vl_configuration import Qwen3VLConfig, Qwen3VLVisionConfig


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

    loss: chex.Array | None = None
    logits: chex.Array = None
    past_key_values: list[chex.Array] | None = None
    hidden_states: tuple[chex.Array] | None = None
    attentions: tuple[chex.Array] | None = None
    rope_deltas: chex.Array | None = None
    image_hidden_states: Float[Array, "batch seq_len hidden_dim"] | None = None


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
            position_ids = np.cumsum(attention_mask, axis=-1) - 1
            position_ids = np.where(attention_mask == 0, 1, position_ids)
            position_ids = np.expand_dims(position_ids, axis=0).repeat(3, axis=0)
            max_position_ids = np.max(position_ids, axis=(0, 2), keepdims=True)
            mrope_position_deltas = (max_position_ids + 1 - attention_mask.shape[-1]).reshape(-1)
        else:
            position_ids = np.arange(seq_length).reshape(1, 1, -1).repeat(3, axis=0).repeat(batch_size, axis=1)
            mrope_position_deltas = np.zeros((batch_size,), dtype=input_ids.dtype)

    return jnp.asarray(position_ids), jnp.asarray(mrope_position_deltas).reshape(-1, 1)


def _merge_multimodal_embeddings(
    inputs_embeds: jax.Array,
    is_multimodal: jax.Array,
    multimodal_embeddings: jax.Array,
) -> jax.Array:
    """Merge multimodal embeddings into text embeddings at placeholder positions."""
    dummy_row = jnp.zeros_like(multimodal_embeddings[0:1])
    flattened_padded = jnp.concatenate([dummy_row, multimodal_embeddings], axis=0)
    gather_indices = jnp.cumsum(is_multimodal)
    update_values = flattened_padded[gather_indices]
    condition = jnp.expand_dims(is_multimodal, axis=-1)
    return jnp.where(condition, update_values, inputs_embeds)


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


def rotate_half(x: chex.Array) -> chex.Array:
    """Rotate half the hidden dims of the input for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision(
    q: chex.Array, k: chex.Array, cos: chex.Array, sin: chex.Array
) -> tuple[chex.Array, chex.Array]:
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


def create_attention_mask(cu_seqlens: chex.Array, seq_length: int, dtype: jnp.dtype) -> chex.Array:
    """Create attention mask from cumulative sequence lengths."""
    attention_mask = jnp.full(
        (1, seq_length, seq_length),
        jnp.finfo(dtype).min,
        dtype=dtype,
    )
    mask_updates = jnp.zeros((1, seq_length, seq_length), dtype=dtype)

    for i in range(1, len(cu_seqlens)):
        start_idx = cu_seqlens[i - 1]
        end_idx = cu_seqlens[i]
        mask_updates = mask_updates.at[..., start_idx:end_idx, start_idx:end_idx].set(0)

    attention_mask = jax.lax.dynamic_update_slice(attention_mask, mask_updates, (0, 0, 0))
    return attention_mask


class Qwen3VLVisionPatchEmbed(nn.Module):
    """3D convolution-based patch embedding for Qwen3-VL vision encoder."""

    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
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
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, hidden_states: chex.Array) -> chex.Array:
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
    """Spatial patch merger with MLP gating for Qwen3-VL."""

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
        super().__init__()
        self.dtype = dtype
        self.spatial_merge_size = config.spatial_merge_size
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm

        self.norm = nn.LayerNorm(
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

    def __call__(self, x: chex.Array) -> chex.Array:
        x = self.norm(x.reshape(-1, self.hidden_size) if self.use_postshuffle_norm else x).reshape(-1, self.hidden_size)
        x = self.linear_fc2(nn.gelu(self.linear_fc1(x), approximate=False))
        return x


class Qwen3VLVisionMLP(nn.Module):
    """Feed-forward network for Qwen3-VL vision encoder."""

    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        super().__init__()
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

    def __call__(self, x: chex.Array) -> chex.Array:
        return self.linear_fc2(self.act(self.linear_fc1(x)))


class Qwen3VLVisionAttention(UnifiedAttention):
    """Self-attention for Qwen3-VL vision encoder with rotary embeddings."""

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
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
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
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def _create_attention_performer(self, config, rngs: nn.Rngs):
        return FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=0.0,
        )

    def __call__(
        self,
        hidden_states: chex.Array,
        cu_seqlens: chex.Array,
        rotary_pos_emb: chex.Array = None,
    ) -> chex.Array:
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
        # Compute cos/sin from freqs
        cos = jnp.cos(rotary_pos_emb)
        sin = jnp.sin(rotary_pos_emb)
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        attn_output = self.attention_performer.forward(
            query_states=q,
            key_states=k,
            value_states=v,
            mode=common_types.MODE_TRAIN,
            attention_mask=create_attention_mask(cu_seqlens, seq_length, q.dtype),
            causal=False,
        ).attention_outputs
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = checkpoint_name(self.proj(attn_output), "vision_attn_output")
        return attn_output


class Qwen3VLVisionBlock(nn.Module):
    """Transformer block for Qwen3-VL vision encoder."""

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
        super().__init__()
        self.norm1 = nn.LayerNorm(
            config.hidden_size,
            epsilon=1e-6,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.norm2 = nn.LayerNorm(
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
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: chex.Array,
        cu_seqlens: chex.Array,
        rotary_pos_emb: chex.Array,
    ) -> chex.Array:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


@register_module(TaskType.BASE_VISION, config=Qwen3VLConfig, model_type="qwen3_vl")
class Qwen3VisionTransformerPretrainedModel(EasyDeLBaseModule):
    """Vision transformer encoder for Qwen3-VL."""

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

        # Positional embeddings
        self.pos_embed = nn.Embed(
            num_embeddings=config.num_position_embeddings,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.spatial_merge_size = config.spatial_merge_size
        head_dim = config.hidden_size // config.num_heads
        self._head_dim_ro = head_dim // 2

        self.blocks = [
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

        self.merger = Qwen3VLVisionPatchMerger(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        # Deepstack mergers for intermediate feature extraction
        self.deepstack_merger_list = [
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

    def get_dtype(self) -> jnp.dtype:
        return self.blocks[0].mlp.linear_fc2.kernel.value.dtype

    def rot_pos_emb(self, grid_thw: chex.Array, max_grid_size: int) -> chex.Array:
        """Compute rotary position embeddings for vision features."""
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = jnp.arange(h)
            hpos_ids = jnp.expand_dims(hpos_ids, 1)
            hpos_ids = jnp.broadcast_to(hpos_ids, (h, w))
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = jnp.transpose(hpos_ids, (0, 2, 1, 3))
            hpos_ids = hpos_ids.flatten()

            wpos_ids = jnp.arange(w)
            wpos_ids = jnp.expand_dims(wpos_ids, 0)
            wpos_ids = jnp.broadcast_to(wpos_ids, (h, w))
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = jnp.transpose(wpos_ids, (0, 2, 1, 3))
            wpos_ids = wpos_ids.flatten()

            stacked = jnp.stack([hpos_ids, wpos_ids], axis=-1)
            repeated = jnp.repeat(stacked, t, axis=1)
            pos_ids.append(repeated)

        pos_ids = jnp.concatenate(pos_ids, axis=0)
        rotary_pos_emb_full = jnp.outer(
            jnp.arange(0, max_grid_size, dtype="f4"),
            1.0 / (10000 ** (jnp.arange(0, self._head_dim_ro, 2, dtype="f4") / self._head_dim_ro)),
        )
        rotary_pos_emb = jnp.take(rotary_pos_emb_full, pos_ids, axis=0)
        rotary_pos_emb = rotary_pos_emb.reshape(pos_ids.shape[0], -1)
        return rotary_pos_emb

    def __call__(
        self,
        hidden_states: chex.Array,
        grid_thw: chex.Array,
        max_grid_size: int,
    ) -> chex.Array:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw, max_grid_size)

        grid_lens = grid_thw[:, 1] * grid_thw[:, 2]
        repeated = jnp.repeat(grid_lens, grid_thw[:, 0])
        cu_seqlens = jnp.cumsum(repeated, dtype="i4")
        cu_seqlens = jnp.pad(cu_seqlens, (1, 0), constant_values=0)

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
            )

        return self.merger(hidden_states)

    def get_encoder(self):
        return self

    def get_decoder(self):
        raise NotImplementedError("Vision model does not have a decoder.")

    def get_lm_head(self):
        raise NotImplementedError("Vision model does not have a language model head.")

    def get_embedding(self):
        return self.patch_embed


class Qwen3VLTextMLP(nn.Module):
    """SwiGLU feed-forward network for Qwen3-VL text decoder."""

    def __init__(
        self,
        config: Qwen3VLConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        column_linear = partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        row_linear = partial(
            RowParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )

        self.gate_proj = column_linear(config.hidden_size, config.intermediate_size, rngs=rngs)
        self.up_proj = column_linear(config.hidden_size, config.intermediate_size, rngs=rngs)
        self.down_proj = row_linear(config.intermediate_size, config.hidden_size, rngs=rngs)
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(self, hidden_states: chex.Array) -> chex.Array:
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
    """Causal self-attention for Qwen3-VL text decoder with mRoPE and QK-norm support."""

    def __init__(
        self,
        config: Qwen3VLConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
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


class Qwen3VLTextDecoderLayer(nn.Module):
    """Transformer decoder layer for Qwen3-VL text model."""

    def __init__(
        self,
        config: Qwen3VLConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

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
        hidden_states: chex.Array,
        mask_info: MaskInfo,
        position_ids: chex.Array,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | None = None,
        output_attentions: bool = False,
        frequencies: chex.Array | None = None,
    ) -> DecoderLayerOutput:
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


@register_module(TaskType.BASE_MODULE, config=Qwen3VLConfig, model_type="qwen3_vl")
class Qwen3VLTextModel(EasyDeLBaseModule):
    """Text decoder model for Qwen3-VL."""

    def __init__(
        self,
        config: Qwen3VLConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        embed_block = auto_remat(
            nn.Embed,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.embed_tokens = embed_block(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            rngs=rngs,
        )

        self.layers = [
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

        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
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
                frequencies=None,
            )
            hidden_states = layer_outputs.hidden_states

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

    def get_encoder(self):
        raise NotImplementedError("Text model does not have an encoder.")

    def get_decoder(self):
        return self

    def get_lm_head(self):
        raise NotImplementedError("Base model does not have a language model head.")

    def get_embedding(self):
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
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        # Vision encoder
        self.visual = Qwen3VisionTransformerPretrainedModel(
            config.vision_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        # Language model (text decoder)
        self.language_model = Qwen3VLTextModel(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        pixel_values: chex.Array | None = None,
        pixel_values_videos: chex.Array | None = None,
        image_grid_thw: tuple | None = None,
        video_grid_thw: tuple | None = None,
        image_max_grid_size: int | None = None,
        video_max_grid_size: int | None = None,
    ) -> BaseModelOutput:
        # Process vision inputs if provided
        if inputs_embeds is None:
            inputs_embeds = self.language_model.embed_tokens(input_ids)

            if pixel_values is not None:
                pixel_values = pixel_values.astype(self.visual.get_dtype())
                image_embeds = self.visual(
                    pixel_values,
                    grid_thw=np.array(image_grid_thw),
                    max_grid_size=image_max_grid_size,
                )
                inputs_embeds = merge_multimodal_embeddings(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    multimodal_embeddings=image_embeds,
                    placeholder_token_id=self.config.image_token_id,
                )

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.astype(self.visual.get_dtype())
                video_embeds = self.visual(
                    pixel_values_videos,
                    grid_thw=np.array(video_grid_thw),
                    max_grid_size=video_max_grid_size,
                )
                inputs_embeds = merge_multimodal_embeddings(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    multimodal_embeddings=video_embeds,
                    placeholder_token_id=self.config.video_token_id,
                )

        # Forward through language model
        return self.language_model(
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
        )

    def get_encoder(self):
        return self.visual

    def get_decoder(self):
        return self.language_model

    def get_lm_head(self):
        raise NotImplementedError("Qwen3VLModel does not have a language model head.")

    def get_embedding(self):
        return self.language_model.embed_tokens


@register_module(TaskType.IMAGE_TEXT_TO_TEXT, config=Qwen3VLConfig, model_type="qwen3_vl")
class Qwen3VLForConditionalGeneration(EasyDeLBaseModule):
    """Qwen3-VL model for conditional generation from images/videos and text.

    Architecture matches HuggingFace:
    - model: Qwen3VLModel (contains visual + language_model)
    - lm_head: Linear projection to vocab
    """

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
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.model = Qwen3VLModel(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.vocab_size = config.vocab_size

        lm_head_block = ColumnParallelLinear
        lm_head_block = auto_remat(
            lm_head_block,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.lm_head = lm_head_block(
            config.hidden_size,
            config.vocab_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | None = None,
        apply_lm_head: bool = True,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        pixel_values: chex.Array | None = None,
        pixel_values_videos: chex.Array | None = None,
        image_grid_thw: tuple | None = None,
        video_grid_thw: tuple | None = None,
        rope_deltas: chex.Array | None = None,
        image_max_grid_size: int | None = None,
        video_max_grid_size: int | None = None,
    ) -> Qwen3VLCausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        tokens_per_second = getattr(self.config.vision_config, "tokens_per_second", 2.0)
        second_per_grid_ts = None

        # Vision processing is now handled inside self.model
        if inputs_embeds is None:
            inputs_embeds = self.model.language_model.embed_tokens(input_ids)

            if pixel_values is not None:
                pixel_values = pixel_values.astype(self.model.visual.get_dtype())
                image_embeds = self.model.visual(
                    pixel_values,
                    grid_thw=np.array(image_grid_thw),
                    max_grid_size=image_max_grid_size,
                )
                inputs_embeds = merge_multimodal_embeddings(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    multimodal_embeddings=image_embeds,
                    placeholder_token_id=self.config.image_token_id,
                )

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.astype(self.model.visual.get_dtype())
                video_embeds = self.model.visual(
                    pixel_values_videos,
                    grid_thw=np.array(video_grid_thw),
                    max_grid_size=video_max_grid_size,
                )
                inputs_embeds = merge_multimodal_embeddings(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    multimodal_embeddings=video_embeds,
                    placeholder_token_id=self.config.video_token_id,
                )

        batch_size = inputs_embeds.shape[0]

        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        attention_mask = mask_info.attention_mask

        if position_ids is None and input_ids is not None and (attention_mask is None or attention_mask.ndim == 2):
            if past_key_values is None or rope_deltas is None:
                position_ids, rope_deltas = get_rope_index(
                    input_ids=np.array(input_ids),
                    image_grid_thw=np.array(image_grid_thw) if image_grid_thw is not None else None,
                    video_grid_thw=np.array(video_grid_thw) if video_grid_thw is not None else None,
                    attention_mask=np.array(attention_mask) if attention_mask is not None else None,
                    spatial_merge_size=self.model.visual.spatial_merge_size,
                    image_token_id=self.config.image_token_id,
                    video_token_id=self.config.video_token_id,
                    vision_start_token_id=self.config.vision_start_token_id,
                    tokens_per_second=tokens_per_second,
                    second_per_grid_ts=second_per_grid_ts,
                )
            else:
                sequence_length = inputs_embeds.shape[1]
                position_ids = jnp.arange(sequence_length).reshape(1, -1).repeat(batch_size, 0)
                position_ids = jnp.expand_dims(position_ids, 0).repeat(3, 0)

        # Forward through model (language_model only since vision is already processed)
        outputs = self.model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            inputs_embeds=inputs_embeds,
            mask_info=mask_info,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs.last_hidden_state

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        logits = None
        if apply_lm_head:
            logits = checkpoint_name(self.apply_lm_head(outputs.last_hidden_state), "lm_head_output")

        return Qwen3VLCausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=rope_deltas,
            image_hidden_states=hidden_states,
        )

    def get_encoder(self):
        return self.model.visual

    def get_decoder(self):
        return self.model.language_model

    def get_lm_head(self):
        return self.lm_head

    def get_embedding(self):
        return self.model.language_model.embed_tokens
