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
from flax import nnx as nn

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import AttentionLayerOutput, BaseModelOutput, DecoderLayerOutput, ModelOutput
from easydel.infra.utils import ACT2FN, auto_remat, block_wise_ffn, get_dot_general_by_bits
from easydel.layers.attention import AttentionModule, FlexibleAttentionModule
from easydel.layers.caching import (
    PagesCache,
    PagesCacheView,
    PagesMetadata,
    TransformerCache,
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.layers.linear import ParallelLinear
from easydel.layers.norms import RMSNorm
from easydel.utils.compiling_utils import ejit

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

    Returns:
        position_ids (`np.ndarray` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`np.ndarray` of shape `(batch_size)`)
    """
    if input_ids.shape[-1] != 1:
        attention_mask = attention_mask[:, : input_ids.shape[-1]]
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = np.ones_like(total_input_ids)

        position_ids = np.ones((3, input_ids.shape[0], input_ids.shape[1]), dtype=input_ids.dtype)
        image_index, video_index = 0, 0
        mrope_position_deltas = []

        for i in range(input_ids.shape[0]):
            input_ids_masked = input_ids[i][attention_mask[i] == 1]
            vision_start_indices = np.where(input_ids_masked == vision_start_token_id)[0]
            vision_tokens = input_ids_masked[vision_start_indices + 1]
            image_nums = np.sum(vision_tokens == image_token_id)
            video_nums = np.sum(vision_tokens == video_token_id)
            input_tokens = input_ids_masked.tolist()
            llm_pos_ids_list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums

            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    int(t),
                    int(h) // spatial_merge_size,
                    int(w) // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(np.arange(text_len).reshape(1, -1).repeat(3, axis=0) + st_idx)

                t_index = np.arange(llm_grid_t).reshape(-1, 1).repeat(llm_grid_h * llm_grid_w, axis=1).flatten()
                h_index = (
                    np.arange(llm_grid_h)
                    .reshape(1, -1, 1)
                    .repeat(llm_grid_t, axis=0)
                    .repeat(llm_grid_w, axis=2)
                    .flatten()
                )
                w_index = (
                    np.arange(llm_grid_w)
                    .reshape(1, 1, -1)
                    .repeat(llm_grid_t, axis=0)
                    .repeat(llm_grid_h, axis=1)
                    .flatten()
                )
                llm_pos_ids_list.append(np.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(np.arange(text_len).reshape(1, -1).repeat(3, axis=0) + st_idx)

            llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
            position_ids[:, i, attention_mask[i] == 1] = llm_positions
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

        mrope_position_deltas = np.array(mrope_position_deltas).reshape(-1, 1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = jnp.cumsum(attention_mask, axis=-1) - 1
            position_ids = jnp.where(attention_mask == 0, 1, position_ids)
            position_ids = jnp.expand_dims(position_ids, axis=0).repeat(3, axis=0)
            max_position_ids = jnp.max(position_ids, axis=(0, 2), keepdims=True)
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                np.arange(input_ids.shape[1]).reshape(1, 1, -1).repeat(3, axis=0).repeat(input_ids.shape[0], axis=1)
            )
            mrope_position_deltas = np.zeros((input_ids.shape[0], 1), dtype=input_ids.dtype)

        return position_ids, mrope_position_deltas


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


# some of my garbage ideas but they always endup workin
# TODO: Fix this structure somehow
@ejit(static_argnames=["TKN_ID"])
def jax_scatter(sec_embeds, ids, fir_embeds, TKN_ID):
    image_embeds = sec_embeds.astype(fir_embeds.dtype)

    image_indices = (
        jnp.where(
            jnp.broadcast_to(jnp.expand_dims(ids == TKN_ID, axis=-1), fir_embeds.shape).reshape(-1),
            size=fir_embeds.size,
            fill_value=-1,
        )[0]
        + 1
    )
    flatten_emb = fir_embeds.reshape(-1)
    flatten_img_emb = image_embeds.reshape(-1)[: len(image_indices)]
    flatten_emb = jnp.pad(flatten_emb, (1, 0))
    flatten_img_emb = jnp.pad(
        flatten_img_emb,
        (0, flatten_emb.size - flatten_img_emb.size),
        # this will default be known as 0 so it wont be used anyway
    )
    image_indices = jnp.pad(
        image_indices,
        (0, flatten_emb.size - image_indices.size),
        # this will default be known as 0 so it wont be used anyway
    )
    scattered_embeds = flatten_emb.at[image_indices].set(flatten_img_emb)[1:]
    fir_embeds = scattered_embeds.reshape(fir_embeds.shape)
    return fir_embeds


def precompute_vl_rotary(dim, theta, max_position):
    inv = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype="f4") / dim))
    seq = jnp.arange(0, max_position, "f4")
    return jnp.outer(seq, inv)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision(array: chex.Array, freqs: chex.Array) -> chex.Array:
    orig_dtype = array.dtype
    array = array.astype("f4")
    cos = jnp.cos(freqs)
    sin = jnp.sin(freqs)
    cos = jnp.expand_dims(jnp.repeat(jnp.expand_dims(cos, 1), 2, -1), 0).astype("f4")
    sin = jnp.expand_dims(jnp.repeat(jnp.expand_dims(sin, 1), 2, -1), 0).astype("f4")
    output = (array * cos) + (rotate_half(array) * sin)
    output = output.astype(orig_dtype)
    return output.squeeze(0)


class PatchEmbed(nn.Module):
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
        hidden_states = hidden_states.reshape(-1, self.embed_dim)
        return hidden_states


class PatchMerger(nn.Module):
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
            ParallelLinear(
                self.hidden_size,
                self.hidden_size,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            ),
            partial(nn.gelu, approximate=False),
            ParallelLinear(
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


class VisionMlp(nn.Module):
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
        self.fc1 = ParallelLinear(
            dim,
            hidden_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.act = ACT2FN[hidden_act]
        self.fc2 = ParallelLinear(
            hidden_dim,
            dim,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, x: chex.Array) -> chex.Array:
        return self.fc2(self.act(self.fc1(x)))


class VisionAttention(AttentionModule):
    def __init__(
        self,
        config,
        dim: int,
        num_heads: int = 16,
        precision: jax.lax.PrecisionLike = None,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(config)
        self.rngs = rngs
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = ParallelLinear(
            dim,
            dim * 3,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.proj = ParallelLinear(
            dim,
            dim,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.attention_performer = FlexibleAttentionModule(
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
        q, k, v = map(
            lambda x: x.squeeze(0),
            jnp.split(
                self.qkv(hidden_states)
                .reshape(seq_length, 3, self.num_heads, -1)
                .transpose(1, 0, 2, 3),  # seq spl nhd fea -> spl seq nhd fea
                3,
                0,
            ),
        )
        q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
        k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)

        # q = jnp.expand_dims(q, 0)
        # k = jnp.expand_dims(k, 0)
        # v = jnp.expand_dims(v, 0)
        # attention_mask = jnp.full(
        # 	(1, seq_length, seq_length),
        # 	jnp.finfo(q.dtype).min,
        # 	dtype=q.dtype,
        # )
        # for i in range(1, len(cu_seqlens)):
        # 	mask = attention_mask.at[
        # 		...,
        # 		cu_seqlens[i - 1] : cu_seqlens[i],
        # 		cu_seqlens[i - 1] : cu_seqlens[i],
        # 	].set(0)
        # 	attention_mask = mask
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
        attn_weights = jax.nn.softmax(
            attn_weights.astype(jnp.float32),
            axis=-1,
        ).astype(q.dtype)
        attn_output = jnp.matmul(attn_weights, v)
        attn_output = attn_output.swapaxes(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2VLVisionBlock(nn.Module):
    def __init__(
        self,
        config: Qwen2VLVisionConfig,
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

        self.attn = VisionAttention(
            config=config,
            dim=config.embed_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.mlp = VisionMlp(
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
    def __init__(
        self,
        config: Qwen2VLConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        linear_class = partial(
            ParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        self.gate_proj = linear_class(
            config.hidden_size,
            config.intermediate_size,
            rngs=rngs,
        )
        self.down_proj = linear_class(
            config.intermediate_size,
            config.hidden_size,
            rngs=rngs,
        )
        self.up_proj = linear_class(
            config.hidden_size,
            config.intermediate_size,
            rngs=rngs,
        )
        self.act_fn = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        gate = self.act_fn(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        hidden_states = self.down_proj(gate * up)
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        return hidden_states


class Qwen2VLAttention(AttentionModule):
    def __init__(
        self,
        config: Qwen2VLConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(config=config)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs

        self.hidden_size = config.hidden_size
        head_dim = config.hidden_size // config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", head_dim)
        self.num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads

        if self.num_key_value_groups == 1:
            assert self.config.num_attention_heads == self.config.num_key_value_heads

        linear_class = partial(
            ParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        self.q_proj = linear_class(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            rngs=rngs,
            use_bias=True,
        )
        self.k_proj = linear_class(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            rngs=rngs,
            use_bias=True,
        )
        self.v_proj = linear_class(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            rngs=rngs,
            use_bias=True,
        )
        self.o_proj = linear_class(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            rngs=rngs,
            use_bias=False,
        )

        self.sliding_window = config.sliding_window if config.use_sliding_window else None
        self.rotary = self.config.get_basic_rope(self.dtype, self.head_dim, self.head_dim, True)

        self.attention_performer = FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=config.attention_dropout,
        )

    def __call__(
        self,
        hidden_states: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        causal_mask: chex.Array | bool | None,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | PagesCacheView | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool = False,
        fcm_mask: chex.Array | None = None,
        frequencies: chex.Array | None = None,
    ) -> tuple[chex.Array, chex.Array]:
        batch_size, sequence_length = hidden_states.shape[:2]
        query_states, key_states, value_states = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )
        qshape = (batch_size, sequence_length, self.config.num_attention_heads, self.head_dim)
        kv_shape = (batch_size, sequence_length, self.config.num_key_value_heads, self.head_dim)
        query_states = query_states.reshape(qshape)
        key_states = key_states.reshape(kv_shape)
        value_states = value_states.reshape(kv_shape)
        if position_ids.ndim == 3:
            position_ids = position_ids[0]
            # cond vision gen issue will be fixed with no mem issue.
        query_states, key_states, value_states = self.apply_qkv_shardings(query_states, key_states, value_states)

        query_states, key_states = self.rotary(
            positions=position_ids,
            query=query_states,
            key=key_states,
            frequencies=frequencies,
        )

        (
            key_states,
            value_states,
            attention_mask,
            init_attention_bias,
            cache_view,
            cache_metadata,
        ) = self.concatenate(
            query=query_states,
            key=key_states,
            value=value_states,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            fcm_mask=fcm_mask,
            sliding_window=self.sliding_window,
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
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            causal=True,
            sliding_window=self.sliding_window,
        )

        attn_output = self.o_proj(self.shard_attention_prod(attn_output=self._merge_heads(attentions.attention_outputs)))
        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class Qwen2VLDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen2VLConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
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
        )

        self.self_attn = attn_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.mlp = mlp_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
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
        hidden_states: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        causal_mask: chex.Array | bool | None,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | PagesCacheView | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool = False,
        fcm_mask: chex.Array | None = None,
        frequencies: chex.Array | None = None,
    ):
        attn_outputs = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask,
            position_ids,
            causal_mask,
            mode,
            cache_view,
            cache_metadata,
            segment_ids,
            output_attentions,
            fcm_mask,
            frequencies,
        )
        hidden_states = hidden_states + attn_outputs.attention_output

        feed_forward_input = self.post_attention_layernorm(hidden_states)

        if self.config.use_scan_mlp:
            feed_forward_hidden_states = block_wise_ffn(
                self.mlp,
                feed_forward_input,
                self.config.scan_mlp_chunk_size,
            )
        else:
            feed_forward_hidden_states = self.mlp(feed_forward_input)

        hidden_states = hidden_states + feed_forward_hidden_states
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
            cache_view=attn_outputs.cache_view,
        )


@register_module(TaskType.BASE_VISION, config=Qwen2VLConfig, model_type="qwen2_vl")
class Qwen2VisionTransformerPretrainedModel(EasyDeLBaseModule):
    config_class = Qwen2VLVisionConfig

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

        self.patch_embed = PatchEmbed(
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
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for _ in range(config.depth)
        ]

        self.merger = PatchMerger(
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
        hidden_states: chex.Array,
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
class Qwen2VLModel(EasyDeLBaseModule):
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

        self.embed_tokens = nn.Embed(
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
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for _ in range(self.config.num_hidden_layers)
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
        input_ids: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.astype("i4"))
        batch_size, sequence_length, _ = inputs_embeds.shape

        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        assert sequence_length <= self.config.max_position_embeddings, (
            f"Maximum Position Embedding Reached ! "
            f"(Excepted <= {self.config.max_position_embeddings} got {sequence_length})"
        )
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length), "b1")
        else:
            if attention_mask.dtype != jnp.bool:
                attention_mask = jnp.astype(attention_mask == 1, "b1")
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                (batch_size, sequence_length),
            ).astype(jnp.int32)

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
                attention_mask=attention_mask,
                position_ids=position_ids,
                mode=mode,
                cache_view=past_key_values.views[idx],
                cache_metadata=cache_metadata,
                causal_mask=self.causal_mask,
                output_attentions=output_attentions,
                segment_ids=segment_ids,
                frequencies=self.frequencies,
            )
            hidden_states = layer_outputs.hidden_states

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            past_key_values[idx] = layer_outputs.cache_view

        hidden_states = self.norm(hidden_states)

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


@register_module(TaskType.IMAGE_TEXT_TO_TEXT, config=Qwen2VLConfig, model_type="qwen2_vl")
class Qwen2VLForConditionalGeneration(EasyDeLBaseModule):
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
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.visual = Qwen2VisionTransformerPretrainedModel(
            config.vision_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.model = Qwen2VLModel(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.vocab_size = config.vocab_size
        self.lm_head = ParallelLinear(
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
        input_ids: chex.Array = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        apply_lm_head: bool = True,
        inputs_embeds: chex.Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        pixel_values: chex.Array | None = None,
        pixel_values_videos: chex.Array | None = None,
        image_grid_thw: tuple | None = None,
        video_grid_thw: tuple | None = None,
        rope_deltas: chex.Array | None = None,
        image_max_grid_size: int | None = None,
        video_max_grid_size: int | None = None,
    ) -> tuple | Qwen2VLCausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.astype(self.visual.get_dtype())
                image_embeds = self.visual(
                    pixel_values,
                    grid_thw=np.array(image_grid_thw),
                    max_grid_size=image_max_grid_size,
                )
                inputs_embeds = jax_scatter(
                    image_embeds,
                    input_ids,
                    inputs_embeds,
                    self.config.image_token_id,
                )

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.astype(self.visual.get_dtype())
                video_embeds = self.visual(
                    pixel_values_videos,
                    grid_thw=np.array(video_grid_thw),
                    max_grid_size=video_max_grid_size,
                )

                inputs_embeds = jax_scatter(
                    video_embeds,
                    input_ids,
                    inputs_embeds,
                    self.config.video_token_id,
                )

        if position_ids is None and input_ids is not None and (attention_mask is None or attention_mask.ndim == 2):
            if past_key_values is not None or rope_deltas is None:
                position_ids, rope_deltas = get_rope_index(
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask,
                    spatial_merge_size=self.visual.spatial_merge_size,
                    image_token_id=self.config.image_token_id,
                    video_token_id=self.config.video_token_id,
                    vision_start_token_id=self.config.vision_start_token_id,
                )
            else:
                batch_size, sequence_length = inputs_embeds.shape[:2]
                position_ids = jnp.arange(sequence_length).reshape(1, -1).repeat(batch_size, 0)
                position_ids = jnp.expand_dims(position_ids, 0).repeat(3, 0)
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            inputs_embeds=inputs_embeds,
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
            logits = self.apply_lm_head(outputs.last_hidden_state)

        return Qwen2VLCausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
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
        inputs_embeds=None,
        position_ids=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        batch_size, seq_length = input_ids.shape

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

        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="b1")
        if attention_mask is not None:
            extended_attention_mask = jax.lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "attention_mask": extended_attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
            }
        )
        return self.prepare_inputs_for_call(**model_inputs)

    def prepare_inputs_for_call(
        self,
        image_grid_thw: chex.Array | None = None,
        video_grid_thw: chex.Array | None = None,
        image_max_grid_size: int | None = None,
        video_max_grid_size: int | None = None,
        drop_ids: bool = True,
        **others,
    ):
        if image_grid_thw is not None:
            if image_max_grid_size is None:
                image_max_grid_size = jnp.max(image_grid_thw[:, 1:])

        if video_grid_thw is not None:
            if video_max_grid_size is None:
                video_max_grid_size = jnp.max(video_grid_thw[:, 1:])

        attention_mask = others.get("attention_mask", None)
        rope_deltas = others.get("rope_deltas", None)
        position_ids = others.get("position_ids", None)

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
                    attention_mask=attention_mask,
                    spatial_merge_size=self.visual.spatial_merge_size,
                    image_token_id=self.config.image_token_id,
                    video_token_id=self.config.video_token_id,
                    vision_start_token_id=self.config.vision_start_token_id,
                )
            else:
                batch_size, sequence_length = others.get("input_ids").shape
                position_ids = jnp.arange(sequence_length).reshape(1, -1).repeat(batch_size, 0)
                position_ids = jnp.expand_dims(position_ids, 0).repeat(3, 0)
        if drop_ids:
            others.pop("input_ids", None)
        others.update(
            dict(
                video_max_grid_size=video_max_grid_size,
                image_max_grid_size=image_max_grid_size,
                video_grid_thw=video_grid_thw,
                image_grid_thw=image_grid_thw,
                position_ids=position_ids,
                rope_deltas=rope_deltas,
            )
        )
        return others

    def get_static_arguments(self):
        return (
            "video_max_grid_size",
            "image_max_grid_size",
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

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, :, -1:] + 1
        model_kwargs.pop("pixel_values", None)  # only effect first iter
        model_kwargs.pop("token_type_ids", None)  # only effect first iter
        return model_kwargs

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        The vision tower acts as the encoder in this multi-modal setup.
        """
        return self.visual

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        """
        return self.model

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        """
        return self.lm_head

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.model.embed_tokens
