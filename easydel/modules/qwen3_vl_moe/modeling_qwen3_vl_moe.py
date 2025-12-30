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
import typing
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
    MoeCausalLMOutput,
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
from easydel.layers.linear import ColumnParallelLinear, RowParallelLinear
from easydel.layers.moe import (
    BaseMoeModule,
    ColumnParallelMoELinear,
    MoeLoadBalancingStrategy,
    MoeRoutingStrategy,
    RowParallelMoELinear,
)
from easydel.layers.norms import RMSNorm

from .qwen3_vl_moe_configuration import Qwen3VLMoeConfig, Qwen3VLMoeTextConfig, Qwen3VLMoeVisionConfig


@auto_pytree
class Qwen3VLMoeCausalLMOutputWithPast(ModelOutput):
    """Output class for Qwen3-VL-MoE causal language model."""

    loss: Array | None = None
    logits: Array = None
    past_key_values: list[Array] | None = None
    hidden_states: tuple[Array] | None = None
    attentions: tuple[Array] | None = None
    rope_deltas: Array | None = None
    image_hidden_states: Float[Array, "batch seq_len hidden_dim"] | None = None
    router_logits: tuple[Array] | None = None


@auto_pytree
class Qwen3VLMoeModelOutputWithPast(ModelOutput):
    """Base-model output for Qwen3-VL-MoE with optional mRoPE deltas."""

    last_hidden_state: Array = None
    past_key_values: list[Array] | None = None
    hidden_states: tuple[Array] | None = None
    attentions: tuple[Array] | None = None
    rope_deltas: Array | None = None
    router_logits: tuple[Array] | None = None


def _dbg_tail(x: Array) -> Array:
    """Return the last 5 flattened elements for compact debug prints."""
    flat = jnp.ravel(x)
    return flat[-5:]


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
    proper 3D rotary position embeddings (mRoPE) in Qwen3-VL-MoE.

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
    """Apply rotary positional embeddings to vision features.

    RoPE is only applied to the first half of the head dimensions (head_dim_ro = head_dim // 2).
    The second half of head dimensions remain unchanged.
    """
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.astype("f4"), k.astype("f4")
    cos, sin = jnp.expand_dims(cos, -2).astype("f4"), jnp.expand_dims(sin, -2).astype("f4")
    rot_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
    k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]
    q_rot_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    q_embed = jnp.concatenate([q_rot_embed, q_pass], axis=-1)
    k_embed = jnp.concatenate([k_rot_embed, k_pass], axis=-1)
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


class Qwen3VLMoeVisionPatchEmbed(nn.Module):
    """3D convolution-based patch embedding for Qwen3-VL-MoE vision encoder."""

    def __init__(
        self,
        config: Qwen3VLMoeVisionConfig,
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
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, hidden_states: Array) -> Array:
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


class Qwen3VLMoeVisionPatchMerger(nn.Module):
    """Spatial patch merger with MLP gating for Qwen3-VL-MoE."""

    def __init__(
        self,
        config: Qwen3VLMoeVisionConfig,
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

    def __call__(self, x: Array) -> Array:
        x = self.norm(x.reshape(-1, self.hidden_size) if self.use_postshuffle_norm else x).reshape(-1, self.hidden_size)
        x = self.linear_fc2(nn.gelu(self.linear_fc1(x), approximate=False))
        return x


class Qwen3VLMoeVisionMLP(nn.Module):
    """Feed-forward network for Qwen3-VL-MoE vision encoder."""

    def __init__(
        self,
        config: Qwen3VLMoeVisionConfig,
        layer_idx: int | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
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
        return self.linear_fc2(self.act(self.linear_fc1(x)))


class Qwen3VLMoeVisionAttention(UnifiedAttention):
    """Self-attention for Qwen3-VL-MoE vision encoder with rotary embeddings."""

    def __init__(
        self,
        config: Qwen3VLMoeVisionConfig,
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
        config: Qwen3VLMoeVisionConfig,
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
        q = jnp.expand_dims(q, axis=0)
        k = jnp.expand_dims(k, axis=0)
        v = jnp.expand_dims(v, axis=0)
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


class Qwen3VLMoeVisionBlock(nn.Module):
    """Transformer block for Qwen3-VL-MoE vision encoder."""

    def __init__(
        self,
        config: Qwen3VLMoeVisionConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
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
        self.attn = Qwen3VLMoeVisionAttention(
            config=config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.mlp = Qwen3VLMoeVisionMLP(
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
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


@register_module(TaskType.BASE_VISION, config=Qwen3VLMoeConfig, model_type="qwen3_vl_moe")
class Qwen3VLMoeVisionTransformerPretrainedModel(EasyDeLBaseModule):
    """Vision transformer encoder for Qwen3-VL-MoE."""

    config_class = Qwen3VLMoeVisionConfig

    def __init__(
        self,
        config: Qwen3VLMoeVisionConfig,
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

        self.patch_embed = Qwen3VLMoeVisionPatchEmbed(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

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
            Qwen3VLMoeVisionBlock(
                config=config,
                layer_idx=idx,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for idx in range(config.depth)
        ]

        self.merger = Qwen3VLMoeVisionPatchMerger(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.deepstack_merger_list = [
            Qwen3VLMoeVisionPatchMerger(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                use_postshuffle_norm=True,
                rngs=rngs,
            )
            for _ in config.deepstack_visual_indexes
        ]

        self.num_grid_per_side = int(math.sqrt(config.num_position_embeddings))

    def get_dtype(self) -> jnp.dtype:
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
        """Compute rotary position embeddings for vision features."""
        merge_size = self.spatial_merge_size

        freq_table = jnp.outer(
            jnp.arange(0, max_grid_size, dtype="f4"),
            1.0 / (10000 ** (jnp.arange(0, self._head_dim_ro, 2, dtype="f4") / self._head_dim_ro)),
        )

        pos_ids_list = []
        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            merged_h, merged_w = h // merge_size, w // merge_size

            block_rows = jnp.arange(merged_h)
            block_cols = jnp.arange(merged_w)
            intra_row = jnp.arange(merge_size)
            intra_col = jnp.arange(merge_size)

            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = jnp.broadcast_to(row_idx, (merged_h, merged_w, merge_size, merge_size)).reshape(-1)
            col_idx = jnp.broadcast_to(col_idx, (merged_h, merged_w, merge_size, merge_size)).reshape(-1)

            coords = jnp.stack([row_idx, col_idx], axis=-1)
            if t > 1:
                coords = jnp.tile(coords, (t, 1))

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
                merger_idx = self.config.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = self.deepstack_merger_list[merger_idx](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        return self.merger(hidden_states), deepstack_feature_lists

    def get_encoder(self):
        return self

    def get_decoder(self):
        raise NotImplementedError("Vision model does not have a decoder.")

    def get_lm_head(self):
        raise NotImplementedError("Vision model does not have a language model head.")

    def get_embedding(self):
        return self.patch_embed


class Qwen3VLMoeTextMLP(nn.Module):
    """SwiGLU feed-forward network for Qwen3-VL-MoE text decoder (dense MLP)."""

    def __init__(
        self,
        config: Qwen3VLMoeTextConfig,
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


class Qwen3VLMoeMLPStack(nn.Module):
    """Qwen3-VL-MoE MoE MLP using the new ParallelMoELinear layers."""

    reform_param: typing.ClassVar = {
        "gate_up_proj$": {
            "splits": [
                {"name": "gate_proj.kernel", "spliter": lambda x: x[..., : x.shape[-1] // 2]},
                {"name": "up_proj.kernel", "spliter": lambda x: x[..., x.shape[-1] // 2 :]},
            ],
            "inverse_spliter": lambda torch, gate, up: torch.cat((gate, up), dim=-1),
        },
        "down_proj$": {
            "splits": [
                {"name": "down_proj.kernel", "spliter": lambda x: x},
            ],
            "inverse_spliter": lambda x: x,
        },
    }

    def __init__(
        self,
        config: Qwen3VLMoeTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.gate_proj = ColumnParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.hidden_size,
            out_features=config.moe_intermediate_size,
            rngs=rngs,
            kernel_init=nn.initializers.normal(),
            use_bias=False,
            partition_manager=config.partition_manager,
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
            kernel_init=nn.initializers.normal(),
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.up_proj = ColumnParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.hidden_size,
            out_features=config.moe_intermediate_size,
            rngs=rngs,
            use_bias=False,
            kernel_init=nn.initializers.normal(),
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        group_sizes: Array,
        sorted_experts: Array | None = None,
    ) -> Array:
        """Forward pass through MoE MLP."""
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states, group_sizes, sorted_experts))
            * self.up_proj(hidden_states, group_sizes, sorted_experts),
            group_sizes,
            sorted_experts,
        )


class Qwen3VLMoeTextSparseBlock(BaseMoeModule):
    """Sparse Mixture of Experts (MoE) block for Qwen3-VL-MoE text decoder.

    This block routes input hidden states to a selected subset of experts
    and combines their outputs.

    Attributes:
        config (Qwen3VLMoeTextConfig): Configuration object for the model.
        gate (ParallelLinear): Linear layer for the gating network.
        experts (Qwen3VLMoeMLPStack): Stack of expert MLP modules.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for matrix multiplications.
        rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: Qwen3VLMoeTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the Qwen3VLMoeTextSparseBlock module.

        Args:
            config (Qwen3VLMoeTextConfig): The configuration object for the model.
            dtype (jnp.dtype): Data type for computations (default: jnp.bfloat16).
            param_dtype (jnp.dtype): Data type for parameters (default: jnp.bfloat16).
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations (default: None).
            rngs (nn.Rngs): Random number generators.
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
        self.gate = ColumnParallelLinear(
            config.hidden_size,
            config.num_experts,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=nn.initializers.normal(config.initializer_range),
        )

        self.experts = Qwen3VLMoeMLPStack(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.layer_idx: int | None = None

    def __call__(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> tuple[Array, Array]:
        """Forward pass of the Sparse MoE block.

        Args:
            hidden_states (Array): Input hidden states (batch_size * sequence_length, hidden_dim).

        Returns:
            tuple[Array, Array]: A tuple containing:
                - final_hidden_states (Array): The output hidden states after MoE processing.
                - router_logits (Array): The logits output by the gating network.
        """
        out, router_logits = self.moe_call(
            hidden_state=hidden_states,
            gate_layer=self.gate,
            expert_layer=self.experts,
            wi_kernel=self.experts.gate_proj.kernel.value,
            wu_kernel=self.experts.up_proj.kernel.value,
            wd_kernel=self.experts.down_proj.kernel.value,
            act_fn=self.experts.act_fn,
        )
        return checkpoint_name(out, "moe_expert_output"), checkpoint_name(router_logits, "moe_router_logits")


class Qwen3VLMoeTextAttention(UnifiedAttention):
    """Causal self-attention for Qwen3-VL-MoE text decoder with mRoPE and QK-norm support."""

    def __init__(
        self,
        config: Qwen3VLMoeTextConfig,
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

    def _postprocess_qkv(self, query_states, key_states, value_states):
        return self.query_normalization(query_states), self.key_normalization(key_states), value_states


class Qwen3VLMoeTextDecoderLayer(nn.Module):
    """Transformer decoder layer for Qwen3-VL-MoE text model with conditional MoE/dense MLP."""

    def __init__(
        self,
        config: Qwen3VLMoeTextConfig,
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
        self.layer_idx = layer_idx

        attn_block = Qwen3VLMoeTextAttention
        mlp_block = Qwen3VLMoeTextMLP
        moe_block = Qwen3VLMoeTextSparseBlock
        attn_block, mlp_block, moe_block = auto_remat(
            attn_block,
            mlp_block,
            moe_block,
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

        self.is_moe = (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and ((layer_idx + 1) % config.decoder_sparse_step == 0)
        )

        if self.is_moe:
            self.mlp = moe_block(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            self.mlp.layer_idx = layer_idx
        else:
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
        output_router_logits: bool = False,
        frequencies: Array | None = None,
    ) -> DecoderLayerOutput:
        attn_input = self.input_layernorm(hidden_states)
        attn_outputs = self.self_attn(
            attn_input,
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

        if self.config.use_scan_mlp and not self.is_moe:
            feed_forward_hidden_states = block_wise_ffn(
                self.mlp,
                feed_forward_input,
                self.config.scan_mlp_chunk_size,
            )
        else:
            feed_forward_hidden_states = self.mlp(feed_forward_input)

        router_logits = None
        if self.is_moe:
            feed_forward_hidden_states, router_logits = feed_forward_hidden_states

        hidden_states = checkpoint_name(hidden_states + feed_forward_hidden_states, "residual")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        return DecoderLayerOutput(
            hidden_states=checkpoint_name(hidden_states, "layer_output"),
            attention_weight=attn_outputs.attention_weight,
            router_logits=router_logits if output_router_logits else None,
            cache_view=attn_outputs.cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=Qwen3VLMoeConfig, model_type="qwen3_vl_moe")
class Qwen3VLMoeTextModel(EasyDeLBaseModule):
    """Text decoder model for Qwen3-VL-MoE with MoE support."""

    def __init__(
        self,
        config: Qwen3VLMoeTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        assert isinstance(config, Qwen3VLMoeTextConfig), (
            f"expected config to be of type Qwen3VLMoeTextConfig but got {type(config)}"
        )
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
            Qwen3VLMoeTextDecoderLayer(
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

    @cached_property
    def frequencies(self):
        """Cached RoPE frequency cache from config."""
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
        output_router_logits: bool | None = None,
        visual_pos_masks: Bool[Array, "batch seq_len"] | None = None,
        deepstack_visual_embeds: list[Array] | None = None,
    ) -> BaseModelOutput:
        """Forward pass through text decoder with MoE and optional deepstack visual integration.

        Processes input embeddings through transformer layers with mixture-of-experts feedforward
        networks. Supports deepstack visual feature injection at specified layer positions.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length). Mutually exclusive
                with inputs_embeds. Converted to embeddings via self.embed_tokens.
            inputs_embeds: Pre-computed input embeddings of shape (batch_size, sequence_length, hidden_size).
                Mutually exclusive with input_ids. Use when embeddings are already prepared.
            attention_mask: Boolean mask of shape (batch_size, sequence_length) where True indicates
                valid tokens, False indicates padding.
            mask_info: Pre-computed mask information for efficient attention operations. Auto-computed
                if None using MaskInfo.dynamic_init.
            position_ids: Position IDs of shape (3, batch_size, sequence_length) for 3D mRoPE.
                Dimensions: [temporal, height, width]. Defaults to standard 1D positions if None.
            mode: Runtime mode (MODE_TRAIN/DECODE/EVAL). Auto-detected: MODE_DECODE if seq_len=1
                and cache exists, else MODE_TRAIN.
            past_key_values: Cached key-value states from previous steps. Initialized as empty
                TransformerCache if None.
            cache_metadata: Metadata for paged attention (sequence lengths, block tables, etc.).
            output_attentions: Whether to return attention weights from all layers. Defaults to
                config.output_attentions.
            output_hidden_states: Whether to return hidden states from all layers. Defaults to
                config.output_hidden_states.
            output_router_logits: Whether to return MoE router logits for auxiliary loss. Defaults
                to config.output_router_logits.
            visual_pos_masks: Boolean mask of shape (batch_size, sequence_length) indicating positions
                where deepstack visual embeddings should be injected. Required if deepstack_visual_embeds provided.
            deepstack_visual_embeds: List of visual embedding arrays (one per layer) for deepstack injection.
                Each array shape: (num_visual_tokens, hidden_size). Injected additively at visual_pos_masks positions.

        Returns:
            MoeCausalLMOutput containing:
                - last_hidden_state: Final layer hidden states of shape (batch_size, sequence_length, hidden_size).
                - hidden_states: Tuple of all layer hidden states if output_hidden_states=True, else None.
                - attentions: Tuple of attention weights from all layers if output_attentions=True, else None.
                - past_key_values: Updated cache for next generation step.
                - router_logits: Tuple of router logits from MoE layers if output_router_logits=True, else None.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify either input_ids or inputs_embeds, but not both.")

        if inputs_embeds is None:
            inputs_embeds = checkpoint_name(self.embed_tokens(input_ids.astype("i4")), "embeddings")

        sequence_length = inputs_embeds.shape[1]

        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        all_router_logits = () if output_router_logits else None

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
                output_router_logits=output_router_logits,
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

            if output_router_logits and layer_outputs.router_logits is not None:
                all_router_logits += (layer_outputs.router_logits,)

            past_key_values[idx] = layer_outputs.cache_view

        hidden_states = self.norm(hidden_states)
        hidden_states = checkpoint_name(hidden_states, "model_output")

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return MoeCausalLMOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            past_key_values=past_key_values,
            router_logits=all_router_logits if output_router_logits else None,
        )

    def _deepstack_process(
        self,
        hidden_states: Array,
        visual_pos_masks: Array,
        visual_embeds: Array,
    ) -> Array:
        """Add visual embeddings to hidden states at visual token positions."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, hidden_dim)
        flat_mask = visual_pos_masks.reshape(-1)

        cumsum_mask = jnp.cumsum(flat_mask) - 1

        visual_update = jnp.where(
            flat_mask[:, None],
            visual_embeds[cumsum_mask],
            jnp.zeros_like(flat_hidden),
        )

        flat_hidden = flat_hidden + visual_update
        return flat_hidden.reshape(batch_size, seq_len, hidden_dim)

    def get_encoder(self):
        raise NotImplementedError("Text model does not have an encoder.")

    def get_decoder(self):
        return self

    def get_lm_head(self):
        raise NotImplementedError("Base model does not have a language model head.")

    def get_embedding(self):
        return self.embed_tokens


@register_module(TaskType.VISION_LM, config=Qwen3VLMoeConfig, model_type="qwen3_vl_moe")
class Qwen3VLMoeModel(EasyDeLBaseModule):
    """Qwen3-VL-MoE multimodal model integrating vision encoder and text decoder with MoE.

    This model handles the fusion of visual and textual information by:
    1. Processing images/videos through the vision encoder
    2. Merging visual embeddings into text embeddings at placeholder positions
    3. Computing 3D position IDs for proper mRoPE handling
    4. Running the fused embeddings through the text decoder with MoE layers

    Architecture matches HuggingFace pattern:
    - visual: Qwen3VLMoeVisionTransformerPretrainedModel
    - language_model: Qwen3VLMoeTextModel
    """

    def __init__(
        self,
        config: Qwen3VLMoeConfig,
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

        self.visual = Qwen3VLMoeVisionTransformerPretrainedModel(
            config.vision_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.language_model = Qwen3VLMoeTextModel(
            config.text_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.rope_deltas = None

    def get_input_embeddings(self):
        return self.language_model.get_embedding()

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_encoder(self):
        return self.visual

    def get_rope_index(
        self,
        input_ids: Array,
        image_grid_thw: Array | None = None,
        video_grid_thw: Array | None = None,
        attention_mask: Array | None = None,
    ) -> tuple[Array, Array]:
        """Calculate 3D RoPE indices for multimodal inputs.

        Different from Qwen2-VL, Qwen3-VL-MoE uses timestamps rather than absolute
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
        output_router_logits: bool | None = None,
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
        """Forward pass for multimodal base model without LM head.

        Orchestrates vision-language fusion by:
        1. Processing images/videos through vision encoder
        2. Merging visual embeddings into text token sequence
        3. Computing 3D mRoPE position IDs
        4. Running fused embeddings through text decoder with MoE

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length). Must contain
                special tokens (image_token_id, video_token_id) at positions where visual
                features should be inserted.
            inputs_embeds: Pre-computed input embeddings of shape (batch_size, sequence_length, hidden_size).
                If provided, input_ids must still be provided for multimodal merging.
            attention_mask: Boolean mask of shape (batch_size, sequence_length) where True=valid, False=padding.
            mask_info: Pre-computed mask information for attention. Auto-computed if None.
            position_ids: 3D position IDs of shape (3, batch_size, sequence_length) for mRoPE.
                Dimensions: [temporal, height, width]. Auto-computed from grid_thw if None.
            mode: Runtime mode (MODE_TRAIN/DECODE/EVAL). Auto-detected from context if None.
            past_key_values: Cached key-value states for autoregressive generation.
            cache_metadata: Metadata for paged attention (sequence lengths, block tables).
            output_attentions: Whether to return attention weights. Defaults to config.output_attentions.
            output_hidden_states: Whether to return all layer hidden states. Defaults to config.output_hidden_states.
            output_router_logits: Whether to return MoE router logits. Defaults to config.output_router_logits.
            visual_pos_masks: Boolean mask of shape (batch_size, sequence_length) indicating visual token
                positions. Auto-computed from pixel_values if not provided.
            deepstack_visual_embeds: List of visual embeddings for injection at each decoder layer.
                Auto-computed from pixel_values if not provided.
            pixel_values: Image pixel values of shape (num_images, channels, height, width).
            pixel_values_videos: Video pixel values of shape (num_videos, channels, frames, height, width).
            image_grid_thw: List of (temporal, height, width) tuples for each image grid.
            video_grid_thw: List of (temporal, height, width) tuples for each video grid.
            image_max_grid_size: Maximum grid size for adaptive image resolution.
            video_max_grid_size: Maximum grid size for adaptive video resolution.
            cache_position: Cache position indices for generation (internal use).

        Returns:
            Qwen3VLMoeModelOutputWithPast containing:
                - last_hidden_state: Final hidden states of shape (batch_size, sequence_length, hidden_size).
                - past_key_values: Updated cache for next generation step.
                - hidden_states: Tuple of all layer hidden states if output_hidden_states=True.
                - attentions: Tuple of attention weights if output_attentions=True.
                - rope_deltas: mRoPE position deltas of shape (batch_size, 1).
                - router_logits: Tuple of MoE router logits if output_router_logits=True.
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
            output_router_logits=output_router_logits,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

        router_logits = getattr(outputs, "router_logits", None) if output_router_logits else None
        return Qwen3VLMoeModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=rope_deltas,
            router_logits=router_logits,
        )

    def get_lm_head(self):
        raise NotImplementedError("Qwen3VLMoeModel does not have a language model head.")

    def get_embedding(self):
        return self.language_model.embed_tokens


@register_module(TaskType.IMAGE_TEXT_TO_TEXT, config=Qwen3VLMoeConfig, model_type="qwen3_vl_moe")
class Qwen3VLMoeForConditionalGeneration(BaseVisionLanguageModule[Qwen3VLMoeModel, Qwen3VLMoeConfig]):
    """Qwen3-VL-MoE model for conditional generation from images/videos and text.

    Inherits from BaseVisionLanguageModule to leverage common VLM infrastructure.

    Architecture matches HuggingFace:
    - model: Qwen3VLMoeModel (contains visual + language_model with MoE)
    - lm_head: Linear projection to vocab

    Class Attributes:
        _task_type: IMAGE_TEXT_TO_TEXT task type
        _model_type: "qwen3_vl_moe" model identifier
        _supports_video: True (supports video input)
        _uses_mrope: True (uses multi-dimensional RoPE)
    """

    _task_type = TaskType.IMAGE_TEXT_TO_TEXT
    _model_type = "qwen3_vl_moe"
    _config_class = Qwen3VLMoeConfig
    _auto_register = False
    _supports_video = True
    _uses_mrope = True

    _vision_tower_name = "visual"
    _projector_name = "merger"
    _language_model_name = "language_model"

    loss_type = "ForCausalLM"

    def __init__(
        self,
        config: Qwen3VLMoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the Qwen3VLMoeForConditionalGeneration model."""
        super().__init__(
            config=config,
            base_model_class=Qwen3VLMoeModel,
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
            router_aux_loss_coef=getattr(config.text_config, "router_aux_loss_coef", 0.001),
            tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
            lm_head_bias=False,
        )
        self.vocab_size = config.text_config.vocab_size

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
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
        """Delegates to self.model.get_video_features."""
        return self.model.get_video_features(pixel_values_videos, video_grid_thw, video_max_grid_size)

    def get_image_features(
        self,
        pixel_values: Array,
        image_grid_thw: Array | None = None,
        image_max_grid_size: int | None = None,
    ) -> tuple[tuple[Array, ...], list[Array]]:
        """Delegates to self.model.get_image_features."""
        return self.model.get_image_features(pixel_values, image_grid_thw, image_max_grid_size)

    def compute_embedding(self, input_ids, *args, **kwargs):
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
        output_router_logits: bool | None = None,
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
        """Forward pass for vision-language generation with MoE.

        Processes multimodal inputs (text, images, videos) through the full model pipeline
        to generate next-token predictions. Handles vision encoding, multimodal fusion,
        3D RoPE positioning, MoE routing, and language modeling.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length).
                Special tokens: config.image_token_id for images, config.video_token_id for videos.
            attention_mask: Boolean mask of shape (batch_size, sequence_length) where True indicates
                valid tokens and False indicates padding. Defaults to all-ones if not provided.
            mask_info: Pre-computed mask information for attention operations. If None, computed
                automatically from input_ids and attention_mask.
            position_ids: 3D position IDs for mRoPE of shape (3, batch_size, sequence_length).
                Dimensions are [temporal, height, width]. Auto-computed from grid_thw if None.
            mode: Runtime mode (MODE_TRAIN, MODE_DECODE, MODE_EVAL). Auto-detected if None based on
                sequence_length and past_key_values presence.
            past_key_values: Cached key-value states from previous forward passes for autoregressive
                generation. Can be TransformerCache, RaggedPagesCache, or HybridCache.
            cache_metadata: Metadata for paged attention operations (sequence lengths, block tables).
            apply_lm_head: Whether to apply the language modeling head to get logits. Set to False
                if you only need hidden states. Defaults to True.
            inputs_embeds: Pre-computed input embeddings of shape (batch_size, sequence_length, hidden_size).
                If provided, input_ids is ignored. Useful for custom embedding manipulation.
            output_attentions: Whether to return attention weights from all layers. Defaults to
                config.output_attentions.
            output_hidden_states: Whether to return hidden states from all layers. Defaults to
                config.output_hidden_states.
            output_router_logits: Whether to return MoE router logits for auxiliary loss computation.
                Defaults to config.text_config.output_router_logits.
            visual_pos_masks: Pre-computed boolean mask of shape (batch_size, sequence_length) indicating
                positions where visual tokens appear. Auto-computed if pixel_values provided.
            deepstack_visual_embeds: Pre-computed list of visual embeddings for deepstack injection at
                each decoder layer. Auto-computed if pixel_values provided.
            pixel_values: Image pixel values of shape (num_images, channels, height, width).
                Images are processed through vision encoder and merged at image_token_id positions.
            pixel_values_videos: Video pixel values of shape (num_videos, channels, frames, height, width).
                Videos are processed similarly to images but with temporal modeling.
            image_grid_thw: List of (temporal, height, width) tuples for each image specifying the
                grid structure after patch embedding. Required if pixel_values provided.
            video_grid_thw: List of (temporal, height, width) tuples for each video. Required if
                pixel_values_videos provided.
            cache_position: Position indices for cached states during generation. Used internally.
            image_max_grid_size: Maximum grid size for adaptive image resolution. If None, uses
                original resolution.
            video_max_grid_size: Maximum grid size for adaptive video resolution. If None, uses
                original resolution.
            **kwargs: Additional arguments (unused, for API compatibility).

        Returns:
            VLMCausalLMOutput containing:
                - logits: Language modeling logits of shape (batch_size, sequence_length, vocab_size)
                    if apply_lm_head=True, else None.
                - past_key_values: Updated cache for next generation step.
                - hidden_states: Tuple of hidden states from all layers if output_hidden_states=True.
                - last_hidden_state: Final hidden state of shape (batch_size, sequence_length, hidden_size).
                - attentions: Tuple of attention weights if output_attentions=True.
                - rope_deltas: mRoPE position deltas of shape (batch_size, 1) for tracking position offsets.
                - router_logits: Tuple of router logits from MoE layers if output_router_logits=True.
                - image_hidden_states: None (reserved for future use).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.get_text_config().output_router_logits
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
            output_router_logits=output_router_logits,
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
            router_logits=getattr(outputs, "router_logits", None) if output_router_logits else None,
        )

    def apply_lm_head(self, hidden_states: Array) -> Array:
        """Apply the language modeling head."""
        return self.lm_head(hidden_states)

    def get_vision_tower(self) -> nn.Module:
        """Returns the vision tower component."""
        return self.model.visual

    def get_language_model(self) -> nn.Module:
        """Returns the language model component."""
        return self.model.language_model
