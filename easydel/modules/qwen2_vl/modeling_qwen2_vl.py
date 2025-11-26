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
import typing as tp
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
from easydel.infra.modeling_outputs import (
    AttentionLayerOutput,
    BaseModelOutput,
    DecoderLayerOutput,
    ModelOutput,
    VLMCausalLMOutput,
)
from easydel.infra.utils import ACT2FN, auto_remat, block_wise_ffn, get_dot_general_by_bits
from easydel.layers.attention import FlexibleAttentionModule
from easydel.layers.attention_unified import UnifiedAttention
from easydel.layers.base_modules import BaseVisionLanguageModule
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

from .qwen2_vl_configuration import Qwen2VLConfig, Qwen2VLVisionConfig


# TODO: Convert this to a jitable jax fn and use that inside model instead of precall
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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the 3D rope index based on image and video's temporal, height, and width in LLM.

    Args:
        input_ids (`np.ndarray` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`np.ndarray` of shape `(num_images, 3)`, *optional*):
            The temporal, height, and width of feature shape of each image in LLM.
        video_grid_thw (`np.ndarray` of shape `(num_videos, 3)`, *optional*):
            The temporal, height, and width of feature shape of each video in LLM.
        attention_mask (`np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        spatial_merge_size (int):
            The spatial merge size for vision embeddings.
        image_token_id (int):
            The token ID representing an image.
        video_token_id (int):
            The token ID representing a video.
        vision_start_token_id (int):
            The token ID representing the start of a vision sequence.
        tokens_per_second (float):
            Temporal scaling applied to video tokens.
        second_per_grid_ts (list[float] | None):
            Per-video seconds per temporal grid step, if available.
        context_len (int):
            Length of any existing KV context to offset positions.
        seq_len (int | None):
            Target sequence length to slice positions to. Defaults to full length.

    Returns:
        position_ids (`np.ndarray` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`np.ndarray` of shape `(batch_size)`)
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


@auto_pytree
class Qwen2VLCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Qwen2VL causal language model (or autoregressive) outputs.
    """

    loss: chex.Array | None = None
    logits: chex.Array = None
    past_key_values: list[chex.Array] | None = None
    hidden_states: tuple[chex.Array] | None = None
    attentions: tuple[chex.Array] | None = None
    rope_deltas: chex.Array | None = None


def create_attention_mask(cu_seqlens, seq_length, dtype):
    """
    Creates an attention mask matrix.

    Args:
        cu_seqlens: Cumulative sequence lengths.
        seq_length: Length of each sequence.
        dtype: Data type of the mask.

    Returns:
        Attention mask matrix.
    """
    attention_mask = jnp.full(
        (1, seq_length, seq_length),
        jnp.finfo(dtype).min,
        dtype=dtype,
    )

    mask_updates = jnp.zeros((1, seq_length, seq_length), dtype=dtype)

    for i in range(1, len(cu_seqlens)):
        start_idx = cu_seqlens[i - 1]
        end_idx = cu_seqlens[i]
        mask_updates = mask_updates.at[
            ...,
            start_idx:end_idx,
            start_idx:end_idx,
        ].set(0)

    attention_mask = jax.lax.dynamic_update_slice(attention_mask, mask_updates, (0, 0, 0))
    return attention_mask


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
    """Overwrite `inputs_embeds` wherever `input_ids` matches placeholder tokens."""
    if isinstance(placeholder_token_id, list):
        placeholder_token_id = jnp.array(placeholder_token_id)
        is_multimodal = jnp.isin(input_ids, placeholder_token_id)
    else:
        is_multimodal = input_ids == placeholder_token_id
    return _merge_multimodal_embeddings(inputs_embeds, is_multimodal, multimodal_embeddings)


def precompute_vl_rotary(dim, theta, max_position):
    """Precompute rotary angle matrix for the vision-language attention stack."""
    inv = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype="f4") / dim))
    seq = jnp.arange(0, max_position, "f4")
    return jnp.outer(seq, inv)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision(array: chex.Array, freqs: chex.Array) -> chex.Array:
    """Apply rotary positional embedding to vision features."""
    orig_dtype = array.dtype
    array = array.astype("f4")
    cos = jnp.cos(freqs)
    sin = jnp.sin(freqs)
    cos = jnp.expand_dims(jnp.repeat(jnp.expand_dims(cos, 1), 2, -1), 0).astype("f4")
    sin = jnp.expand_dims(jnp.repeat(jnp.expand_dims(sin, 1), 2, -1), 0).astype("f4")
    output = (array * cos) + (rotate_half(array) * sin)
    output = output.astype(orig_dtype)
    return output.squeeze(0)


class Qwen2VLPatchEmbed(nn.Module):
    """Convert images or video frames into patch embeddings for Qwen2-VL."""

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
        precision: jax.lax.PrecisionLike = None,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: nn.Rngs,
    ) -> None:
        self.dtype = dtype
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv(
            in_features=in_channels,
            out_features=embed_dim,
            kernel_size=kernel_size,
            strides=kernel_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> chex.Array:
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
        hidden_states = hidden_states.reshape(-1, self.embed_dim)
        return hidden_states


class Qwen2VLPatchMerger(nn.Module):
    """Merge neighboring spatial patches to downsample visual tokens."""

    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        precision: jax.lax.PrecisionLike = None,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: nn.Rngs,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(
            context_dim,
            epsilon=1e-6,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.mlp = [
            ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            ),
            partial(nn.gelu, approximate=False),
            RowParallelLinear(
                self.hidden_size,
                dim,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            ),
        ]

    def __call__(self, x: chex.Array) -> chex.Array:
        x = self.ln_q(x).reshape(-1, self.hidden_size)
        for mlp in self.mlp:  # make easy attach work with no effort
            x = mlp(x)
        return x


class Qwen2VLVisionMLP(nn.Module):
    """Feed-forward module for the Qwen2-VL vision encoder."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        hidden_act: str,
        precision: jax.lax.PrecisionLike = None,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: nn.Rngs,
    ) -> None:
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            dim,
            hidden_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.act = ACT2FN[hidden_act]
        self.fc2 = RowParallelLinear(
            hidden_dim,
            dim,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, x: chex.Array) -> chex.Array:
        return self.fc2(self.act(self.fc1(x)))


class Qwen2VLVisionAttention(UnifiedAttention):
    """Self-attention layer for vision patches with rotary position encoding."""

    def __init__(
        self,
        config,
        layer_idx: int,
        dim: int,
        num_heads: int = 16,
        precision: jax.lax.PrecisionLike = None,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: nn.Rngs,
    ):
        self.embed_dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Create a proxy config to adapt VisionConfig to UnifiedAttention requirements
        class ConfigAdapter:
            def __init__(self, config, dim, num_heads):
                self.hidden_size = dim
                self.num_attention_heads = num_heads
                self.num_key_value_heads = num_heads
                self.head_dim = dim // num_heads
                self.attention_bias = True  # Vision model usually has bias
                # Copy other attributes from config
                for k, v in config.__dict__.items():
                    if not hasattr(self, k):
                        setattr(self, k, v)
                self.bits = getattr(config, "bits", None)
                self.easy_method = getattr(config, "easy_method", None)
                self.scan_mlp_chunk_size = getattr(config, "scan_mlp_chunk_size", 1024)

        adapted_config = ConfigAdapter(config, dim, num_heads)

        super().__init__(
            config=adapted_config,
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
        config,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        precision: jax.lax.PrecisionLike,
        rngs: nn.Rngs,
    ) -> None:
        self.qkv = ColumnParallelLinear(
            self.embed_dim,
            self.embed_dim * 3,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.proj = RowParallelLinear(
            self.embed_dim,
            self.embed_dim,
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
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
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
        q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
        k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)
        row_ids = jnp.arange(seq_length)[None, None, :]
        col_ids = jnp.arange(seq_length)[None, :, None]
        starts = cu_seqlens[:-1][:, None, None]
        ends = cu_seqlens[1:][:, None, None]
        is_valid = (row_ids >= starts) & (row_ids < ends) & (col_ids >= starts) & (col_ids < ends)
        combined_mask = jnp.any(is_valid, axis=0)
        attention_mask = jnp.where(combined_mask, 0.0, jnp.finfo(q.dtype).min)
        q = q.swapaxes(0, 1)
        k = k.swapaxes(0, 1)
        v = v.swapaxes(0, 1)
        attn_weights = jnp.matmul(q, k.swapaxes(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(q.dtype)
        attn_output = jnp.matmul(attn_weights, v)
        attn_output = attn_output.swapaxes(0, 1).reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2VLVisionBlock(nn.Module):
    """Vision transformer block combining attention and MLP with pre-normalization."""

    def __init__(
        self,
        config: Qwen2VLVisionConfig,
        layer_idx: int,
        precision: jax.lax.PrecisionLike = None,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: nn.Rngs,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(
            config.embed_dim,
            epsilon=1e-6,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.norm2 = nn.LayerNorm(
            config.embed_dim,
            epsilon=1e-6,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)

        self.attn = Qwen2VLVisionAttention(
            config=config,
            layer_idx=layer_idx,
            dim=config.embed_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.mlp = Qwen2VLVisionMLP(
            dim=config.embed_dim,
            hidden_dim=mlp_hidden_dim,
            hidden_act=config.hidden_act,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, hidden_states, cu_seqlens, rotary_pos_emb) -> chex.Array:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2VLMLP(nn.Module):
    """Feed-forward network used in the Qwen2-VL language decoder."""

    def __init__(
        self,
        config: Qwen2VLConfig,
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
        column_parallel_linear = partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        row_parallel_linear = partial(
            RowParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        self.gate_proj = column_parallel_linear(
            config.hidden_size,
            config.intermediate_size,
            rngs=rngs,
        )
        self.down_proj = row_parallel_linear(
            config.intermediate_size,
            config.hidden_size,
            rngs=rngs,
        )
        self.up_proj = column_parallel_linear(
            config.hidden_size,
            config.intermediate_size,
            rngs=rngs,
        )
        self.act_fn = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> jnp.ndarray:
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


class Qwen2VLAttention(UnifiedAttention):
    """Causal self-attention used in the Qwen2-VL language decoder."""

    def __init__(
        self,
        config: Qwen2VLConfig,
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
        )

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ) -> AttentionLayerOutput:
        if position_ids is not None and position_ids.ndim == 3:
            position_ids = position_ids[0]
        return super().__call__(
            hidden_states=hidden_states,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
            frequencies=frequencies,
        )


class Qwen2VLDecoderLayer(nn.Module):
    """Transformer decoder layer coupling Qwen2-VL attention and feed-forward modules."""

    def __init__(
        self,
        config: Qwen2VLConfig,
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
        attn_block = Qwen2VLAttention
        mlp_block = Qwen2VLMLP
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

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ):
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


@register_module(TaskType.BASE_VISION, config=Qwen2VLConfig, model_type="qwen2_vl")
class Qwen2VLVisionTransformer(EasyDeLBaseModule):
    """Vision transformer encoder used to extract image features for Qwen2-VL."""

    config_class = Qwen2VLVisionConfig

    def __init__(
        self,
        config: Qwen2VLVisionConfig,
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

        self.patch_embed = Qwen2VLPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.spatial_merge_size = config.spatial_merge_size
        head_dim = config.embed_dim // config.num_heads
        self._head_dim_ro = head_dim // 2

        self.blocks = [
            Qwen2VLVisionBlock(
                config=config,
                layer_idx=idx,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for idx in range(config.depth)
        ]

        self.merger = Qwen2VLPatchMerger(
            dim=config.hidden_size,
            context_dim=config.embed_dim,
            spatial_merge_size=config.spatial_merge_size,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def get_dtype(self) -> jnp.dtype:
        return self.blocks[0].mlp.fc2.kernel.value.dtype

    def rot_pos_emb(self, grid_thw, max_grid_size):
        pos_ids = []

        for t, h, w in grid_thw:
            # Create height position ids
            hpos_ids = jnp.arange(h)
            hpos_ids = jnp.expand_dims(hpos_ids, 1)
            hpos_ids = jnp.broadcast_to(hpos_ids, (h, w))

            # Reshape and permute height positions
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = jnp.transpose(hpos_ids, (0, 2, 1, 3))
            hpos_ids = hpos_ids.flatten()

            # Create width position ids
            wpos_ids = jnp.arange(w)
            wpos_ids = jnp.expand_dims(wpos_ids, 0)
            wpos_ids = jnp.broadcast_to(wpos_ids, (h, w))

            # Reshape and permute width positions
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = jnp.transpose(wpos_ids, (0, 2, 1, 3))
            wpos_ids = wpos_ids.flatten()

            # Stack and repeat
            stacked = jnp.stack([hpos_ids, wpos_ids], axis=-1)
            repeated = jnp.repeat(stacked, t, axis=1)
            pos_ids.append(repeated)

        pos_ids = jnp.concatenate(pos_ids, axis=0)
        rotary_pos_emb_full = jnp.outer(
            jnp.arange(0, max_grid_size, dtype="f4"),
            1.0 / (10000 ** (jnp.arange(0, self._head_dim_ro, 2, dtype="f4") / self._head_dim_ro)),
        )
        # Index into embeddings and flatten
        rotary_pos_emb = jnp.take(rotary_pos_emb_full, pos_ids, axis=0)
        rotary_pos_emb = rotary_pos_emb.reshape(pos_ids.shape[0], -1)

        return rotary_pos_emb

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        grid_thw: chex.Array,
        max_grid_size,
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
        """
        Returns the encoder part of the model's graph definition.
        This vision model acts as the encoder.
        """
        return self

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        This is an encoder-only model and does not have a decoder.
        """
        raise NotImplementedError("This is an encoder-only model and does not have a decoder.")

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        This vision model does not have a language model head.
        """
        raise NotImplementedError("This vision model does not have a language model head.")

    def get_embedding(self):
        """
        Returns the embedding layer of the module. In this case, it's the patch embedding layer.
        """
        return self.patch_embed


@register_module(TaskType.BASE_MODULE, config=Qwen2VLConfig, model_type="qwen2_vl")
class Qwen2VLTextModel(EasyDeLBaseModule):
    """Language decoder stack for Qwen2-VL that consumes projected vision tokens."""

    def __init__(
        self,
        config: Qwen2VLConfig,
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
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            rngs=rngs,
        )

        self.layers = [
            Qwen2VLDecoderLayer(
                config=config,
                layer_idx=idx,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for idx in range(self.config.num_hidden_layers)
        ]
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
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
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
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
                frequencies=self.frequencies,
            )
            hidden_states = layer_outputs.hidden_states

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            past_key_values[idx] = layer_outputs.cache_view

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

    def get_input_embeddings(self):
        """
        Returns the input embedding layer of the module.
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """
        Sets the input embedding layer of the module.
        """
        self.embed_tokens = value


@register_module(TaskType.BASE_MODULE, config=Qwen2VLConfig, model_type="qwen2_vl")
class Qwen2VLModel(EasyDeLBaseModule):
    """
    The Qwen2-VL model which consists of a vision encoder and a language model.
    """

    def __init__(
        self,
        config: Qwen2VLConfig,
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
        self.visual = Qwen2VLVisionTransformer(
            config.vision_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.language_model = Qwen2VLTextModel(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.rope_deltas = None

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_video_features(self, pixel_values_videos: chex.Array, video_grid_thw: chex.Array | None = None):
        pixel_values_videos = pixel_values_videos.astype(self.visual.dtype)
        video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
        if video_grid_thw is not None:
            split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
            indices = np.cumsum(split_sizes)[:-1]
            video_embeds = jnp.split(video_embeds, indices)
        return video_embeds

    def get_image_features(self, pixel_values: chex.Array, image_grid_thw: chex.Array | None = None):
        pixel_values = pixel_values.astype(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        if image_grid_thw is not None:
            split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
            indices = np.cumsum(split_sizes)[:-1]
            image_embeds = jnp.split(image_embeds, indices)
        return image_embeds

    def get_rope_index(
        self,
        input_ids: chex.Array = None,
        image_grid_thw: chex.Array = None,
        video_grid_thw: chex.Array = None,
        attention_mask: chex.Array = None,
    ):
        return get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            spatial_merge_size=self.config.vision_config.spatial_merge_size,
            image_token_id=self.config.image_token_id,
            video_token_id=self.config.video_token_id,
            vision_start_token_id=self.config.vision_start_token_id,
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        past_key_values: TransformerCache | RaggedPagesCache | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        pixel_values: chex.Array | None = None,
        pixel_values_videos: chex.Array | None = None,
        image_grid_thw: chex.Array | None = None,
        video_grid_thw: chex.Array | None = None,
        rope_deltas: chex.Array | None = None,
        cache_position: chex.Array | None = None,
        mask_info: MaskInfo | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | None = None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            if isinstance(image_embeds, (list, tuple)):
                image_embeds = jnp.concatenate(image_embeds, axis=0)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=image_embeds,
                placeholder_token_id=self.config.image_token_id,
            )

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            if isinstance(video_embeds, (list, tuple)):
                video_embeds = jnp.concatenate(video_embeds, axis=0)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=video_embeds,
                placeholder_token_id=self.config.video_token_id,
            )

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


@register_module(TaskType.IMAGE_TEXT_TO_TEXT, config=Qwen2VLConfig, model_type="qwen2_vl")
class Qwen2VLForConditionalGeneration(BaseVisionLanguageModule[Qwen2VLModel, Qwen2VLConfig]):
    """Multimodal Qwen2-VL model for conditional generation from images/video and text.

    Inherits from BaseVisionLanguageModule to leverage common VLM infrastructure.

    Attributes:
        config (Qwen2VLConfig): Configuration object.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): JAX precision level.
        rngs (nn.Rngs): Random number generators.

    Class Attributes:
        _task_type: IMAGE_TEXT_TO_TEXT task type
        _model_type: "qwen2_vl" model identifier
        _supports_video: True (Qwen2-VL supports video input)
        _uses_mrope: True (uses multi-dimensional RoPE)
    """

    # Class attributes for registration and capabilities
    _task_type = TaskType.IMAGE_TEXT_TO_TEXT
    _model_type = "qwen2_vl"
    _config_class = Qwen2VLConfig
    _auto_register = False  # Already registered via decorator
    _supports_video = True
    _uses_mrope = True

    # Component name mapping (Qwen uses "visual" not "vision_tower")
    _vision_tower_name = "visual"
    _projector_name = "merger"  # Qwen2VL uses merger in visual
    _language_model_name = "language_model"

    loss_type = "ForCausalLM"

    def __init__(
        self,
        config: Qwen2VLConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the Qwen2VLForConditionalGeneration model."""
        super().__init__(
            config=config,
            base_model_class=Qwen2VLModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            # VLM-specific configuration
            vision_feature_layer=-1,
            vision_feature_select_strategy="default",
            image_token_index=config.image_token_id,
            video_token_index=config.video_token_id,
            # mRoPE config
            spatial_merge_size=config.vision_config.spatial_merge_size,
            # LM head configuration
            tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
            lm_head_bias=False,
        )
        self.vocab_size = config.vocab_size

    @property
    def visual(self):
        """Property to access the vision transformer for backward compatibility."""
        return self.base_model.visual

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.base_model.set_input_embeddings(value)

    def get_video_features(
        self,
        pixel_values_videos: Float[Array, "batch temporal channels height width"],
        video_grid_thw: tuple | None = None,
        **kwargs,
    ) -> Float[Array, "batch num_tokens hidden"]:
        """Extract and project video features.

        Args:
            pixel_values_videos: Input video pixel values
            video_grid_thw: Video grid shape (temporal, height, width)
            **kwargs: Additional arguments

        Returns:
            Projected video features
        """
        return self.base_model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(
        self,
        pixel_values: Float[Array, "batch channels height width"],
        image_grid_thw: tuple | None = None,
        **kwargs,
    ) -> Float[Array, "batch num_patches hidden"]:
        """Extract and project image features.

        Args:
            pixel_values: Input image pixel values
            image_grid_thw: Image grid shape (temporal=1, height, width)
            **kwargs: Additional arguments

        Returns:
            Projected image features
        """
        return self.base_model.get_image_features(pixel_values, image_grid_thw)

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
        cache_position: chex.Array | None = None,
        **kwargs,
    ) -> VLMCausalLMOutput:
        """Forward pass for the Qwen2-VL model.

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
            rope_deltas: Position deltas for mRoPE
            cache_position: Cache position for generation
            **kwargs: Additional arguments

        Returns:
            VLMCausalLMOutput: Model outputs including logits and optional states
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
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            mask_info=mask_info,
            mode=mode,
            cache_metadata=cache_metadata,
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
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        batch_size, _seq_length = input_ids.shape

        if past_key_values is None:
            if starts is None:
                starts = self.compute_prefill_length(input_ids, pad_token_id)
            past_key_values = self.init_cache(
                batch_size,
                max_length,
                starts,
                None,
                pad_token_id,
            )

        if inputs_embeds is not None:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids if inputs_embeds is None else None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        attn = mask_info.attention_mask if mask_info is not None else None
        if attn is None:
            attn = jnp.ones((batch_size, _seq_length), dtype="b1")

        extended_attention_mask = jnp.ones((batch_size, max_length), dtype=attn.dtype)
        extended_attention_mask = jax.lax.dynamic_update_slice(extended_attention_mask, attn, (0, 0))
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "attention_mask": extended_attention_mask,
                "mask_info": mask_info,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
            }
        )
        return model_inputs

    def _create_required_props_from_kwargs(
        self,
        model_kwargs: dict[str, chex.Array],
    ) -> tp.Mapping[str, dict[str, tp.Any]] | None:
        basics = {}
        if "image_grid_thw" in model_kwargs.keys():
            basics.update({"image_grid_thw": {"value": jnp.array(model_kwargs["image_grid_thw"])}})
        if "video_grid_thw" in model_kwargs.keys():
            basics.update({"video_grid_thw": {"value": jnp.array(model_kwargs["video_grid_thw"])}})
        return basics

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, :, -1:] + 1
        model_kwargs.pop("pixel_values", None)  # only effect first iter
        model_kwargs.pop("pixel_values_videos", None)  # only effect first iter
        model_kwargs.pop("token_type_ids", None)  # only effect first iter
        return model_kwargs

    def apply_lm_head(self, hidden_states: Array) -> Array:
        """Apply the language modeling head."""
        return self.lm_head(hidden_states)

    def get_vision_tower(self) -> nn.Module:
        """Returns the vision tower component."""
        return self.base_model.visual

    def get_language_model(self) -> nn.Module:
        """Returns the language model component."""
        return self.base_model.language_model

    def prepare_inputs_for_call(
        self,
        image_grid_thw: chex.Array | None = None,
        video_grid_thw: chex.Array | None = None,
        drop_ids: bool = True,
        **others,
    ):
        """Prepare inputs with mRoPE position IDs computed from grid shapes."""
        attention_mask = others.get("attention_mask", None)
        mask_info = others.get("mask_info", None)
        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=others.get("input_ids"),
            inputs_embeds=others.get("inputs_embeds"),
            attention_mask=attention_mask,
        )
        attention_mask = mask_info.attention_mask

        rope_deltas = others.get("rope_deltas", None)
        position_ids = others.get("position_ids", None)
        tokens_per_second = getattr(self.config.vision_config, "tokens_per_second", 1.0)

        if (
            position_ids is None
            and others.get("input_ids", None) is not None
            and (attention_mask is None or attention_mask.ndim == 2)
        ):
            if others.get("past_key_values", None) is not None or others.get("rope_deltas", None) is None:
                position_ids, rope_deltas = get_rope_index(
                    input_ids=others.get("input_ids"),
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=np.array(attention_mask) if attention_mask is not None else None,
                    spatial_merge_size=self.visual.spatial_merge_size,
                    image_token_id=self.config.image_token_id,
                    video_token_id=self.config.video_token_id,
                    vision_start_token_id=self.config.vision_start_token_id,
                    tokens_per_second=tokens_per_second,
                )
            else:
                batch_size, sequence_length = others.get("input_ids").shape
                position_ids = jnp.arange(sequence_length).reshape(1, -1).repeat(batch_size, 0)
                position_ids = jnp.expand_dims(position_ids, 0).repeat(3, 0)
        if drop_ids:
            others.pop("input_ids", None)
        others.update(
            dict(
                video_grid_thw=video_grid_thw,
                image_grid_thw=image_grid_thw,
                position_ids=position_ids,
                rope_deltas=rope_deltas,
                attention_mask=attention_mask,
                mask_info=mask_info,
            )
        )
        return others

    def get_static_arguments(self):
        return (
            "image_grid_thw",
            "video_grid_thw",
        )

    def _get_compile_model_kwargs(
        self,
        batch_size: int,
        input_tokens_length: int,
        input_sharding: jax.sharding.PartitionSpec,
        rngs: jax.random.PRNGKey,
        vision_included: bool = False,
        vision_batch_size: int = 1,
        vision_channels: int = 3,
        vision_height: int | None = None,
        vision_width: int | None = None,
        required_props: tp.Mapping[str, dict[str, tp.Any]] | None = None,
        **kwargs,
    ):
        basics = super()._get_compile_model_kwargs(
            batch_size=batch_size,
            input_tokens_length=input_tokens_length,
            input_sharding=input_sharding,
            rngs=rngs,
            vision_included=vision_included,
            vision_batch_size=vision_batch_size,
            vision_channels=vision_channels,
            vision_height=vision_height,
            vision_width=vision_width,
            required_props=required_props,
            **kwargs,
        )

        if vision_included:
            assert required_props is not None
            assert "image_grid_thw" in required_props.keys()

            pixel_values = jnp.ones((vision_height, vision_width), dtype="f4")
            basics.update(
                {
                    "pixel_values": pixel_values,
                    "image_grid_thw": jnp.array(required_props["image_grid_thw"]["value"]),
                }
            )
        return basics

    def _create_required_props_from_kwargs(
        self,
        model_kwargs: dict[str, chex.Array],
    ) -> tp.Mapping[str, dict[str, tp.Any]] | None:
        basics = {}
        if "image_grid_thw" in model_kwargs.keys():
            basics.update({"image_grid_thw": {"value": jnp.array(model_kwargs["image_grid_thw"])}})
        if "video_grid_thw" in model_kwargs.keys():
            basics.update({"video_grid_thw": {"value": jnp.array(model_kwargs["video_grid_thw"])}})
        return basics
