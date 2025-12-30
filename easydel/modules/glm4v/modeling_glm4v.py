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
from itertools import groupby

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
from easydel.layers.linear import ColumnParallelLinear, RowParallelLinear
from easydel.layers.norms import RMSNorm

from .glm4v_configuration import Glm4vConfig, Glm4vTextConfig, Glm4vVisionConfig


@auto_pytree
class Glm4vModelOutputWithPast(ModelOutput):
    """Base model output for GLM4V multimodal model."""

    last_hidden_state: Array | None = None
    past_key_values: TransformerCache | None = None
    hidden_states: tuple[Array] | None = None
    attentions: tuple[Array] | None = None
    rope_deltas: Array | None = None


def _rotate_half(x: Array) -> Array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision(q: Array, k: Array, cos: Array, sin: Array) -> tuple[Array, Array]:
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def create_attention_mask(cu_seqlens: Array, seq_length: int, dtype: jnp.dtype) -> Array:
    """Create block-diagonal attention mask from cumulative sequence lengths."""
    positions = jnp.arange(seq_length)
    starts = cu_seqlens[:-1]
    ends = cu_seqlens[1:]
    in_segment = (positions[:, None] >= starts[None, :]) & (positions[:, None] < ends[None, :])
    segment_ids = jnp.argmax(in_segment.astype(jnp.int32), axis=-1)
    same_segment = segment_ids[:, None] == segment_ids[None, :]
    attention_mask = jnp.where(same_segment, 0.0, jnp.finfo(dtype).min).astype(dtype)
    return attention_mask[None, :, :]


class Glm4vVisionPatchEmbed(nn.Module):
    """3D convolution-based patch embedding for GLM4V vision encoder."""

    def __init__(
        self,
        config: Glm4vVisionConfig,
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


class Glm4vVisionMLP(nn.Module):
    """SwiGLU-style MLP for GLM4V vision encoder blocks."""

    def __init__(
        self,
        config: Glm4vVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act = ACT2FN[config.hidden_act]
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, x: Array) -> Array:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class Glm4vVisionAttention(UnifiedAttention):
    """Self-attention for GLM4V vision encoder with rotary embeddings."""

    def __init__(
        self,
        config: Glm4vVisionConfig,
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
        config: Glm4vVisionConfig,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        precision: jax.lax.PrecisionLike,
        rngs: nn.Rngs,
    ) -> None:
        self.qkv = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size * 3,
            use_bias=getattr(config, "attention_bias", False),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            use_bias=False,
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
            requires_cache=False,
        )

    def __call__(
        self,
        hidden_states: Array,
        cu_seqlens: Array,
        rotary_pos_emb: Array,
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
        return checkpoint_name(self.proj(attn_output), "vision_attn_output")


class Glm4vVisionBlock(nn.Module):
    """Transformer block for GLM4V vision encoder."""

    def __init__(
        self,
        config: Glm4vVisionConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        self.norm1 = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.norm2 = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.attn = Glm4vVisionAttention(
            config=config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.mlp = Glm4vVisionMLP(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, hidden_states: Array, *, cu_seqlens: Array, rotary_pos_emb: Array) -> Array:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Glm4vVisionPatchMerger(nn.Module):
    """Projection + gated MLP merger for GLM4V vision features."""

    def __init__(
        self,
        dim: int,
        context_dim: int,
        hidden_act: str,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        self.proj = ColumnParallelLinear(
            dim,
            dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.norm = nn.LayerNorm(dim, epsilon=1e-6, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.act1 = jax.nn.gelu
        self.act = ACT2FN[hidden_act]
        self.gate_proj = ColumnParallelLinear(
            dim,
            context_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.up_proj = ColumnParallelLinear(
            dim,
            context_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.down_proj = RowParallelLinear(
            context_dim,
            dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.reform_param = {
            "post_projection_norm": {
                "splits": [
                    {
                        "name": "norm",
                        "spliter": lambda x: x,
                    }
                ]
            }
        }

    def __call__(self, hidden_state: Array) -> Array:
        hidden_state = self.proj(hidden_state)
        hidden_state = self.act1(self.norm(hidden_state))
        return self.down_proj(self.act(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


@register_module(TaskType.BASE_VISION, config=Glm4vConfig, model_type="glm4v")
class Glm4vVisionModel(EasyDeLBaseModule):
    """Vision transformer encoder for GLM4V."""

    config_class = Glm4vVisionConfig

    def __init__(
        self,
        config: Glm4vVisionConfig,
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
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = Glm4vVisionPatchEmbed(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.post_conv_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.post_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

        num_positions = int((config.image_size // config.patch_size) ** 2)
        self.pos_embed = nn.Embed(
            num_embeddings=num_positions,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.num_grid_per_side = int(math.sqrt(num_positions))
        self.reform_param = {
            "embeddings.position_embedding": {
                "splits": [
                    {
                        "name": "pos_embed",
                        "spliter": lambda x: x,
                    }
                ]
            }
        }

        head_dim = config.hidden_size // config.num_heads
        self._head_dim_ro = (head_dim // 2) // 2

        self.blocks = [
            Glm4vVisionBlock(
                config,
                layer_idx=idx,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for idx in range(config.depth)
        ]

        self.downsample = nn.Conv(
            in_features=config.hidden_size,
            out_features=config.out_hidden_size,
            kernel_size=(config.spatial_merge_size, config.spatial_merge_size),
            strides=(config.spatial_merge_size, config.spatial_merge_size),
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.merger = Glm4vVisionPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.intermediate_size,
            hidden_act=config.hidden_act,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def get_dtype(self) -> jnp.dtype:
        return self.pos_embed.embedding.value.dtype

    def fast_pos_embed_interpolate(self, grid_thw: Array) -> Array:
        """Bilinear-interpolate 2D position embeddings and apply merge-size permutation."""
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

            h_floor = jnp.floor(h_idxs).astype(jnp.int32)
            w_floor = jnp.floor(w_idxs).astype(jnp.int32)
            h_ceil = jnp.clip(h_floor + 1, max=self.num_grid_per_side - 1)
            w_ceil = jnp.clip(w_floor + 1, max=self.num_grid_per_side - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            base_h = h_floor * self.num_grid_per_side
            base_h_ceil = h_ceil * self.num_grid_per_side

            indices = [
                (base_h[:, None] + w_floor[None, :]).flatten(),
                (base_h[:, None] + w_ceil[None, :]).flatten(),
                (base_h_ceil[:, None] + w_floor[None, :]).flatten(),
                (base_h_ceil[:, None] + w_ceil[None, :]).flatten(),
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
        merge_size = self.spatial_merge_size
        freq_table = jnp.outer(
            jnp.arange(0, max_grid_size, dtype=jnp.float32),
            1.0 / (10000 ** (jnp.arange(0, self._head_dim_ro * 2, 2, dtype=jnp.float32) / (self._head_dim_ro * 2))),
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

    def __call__(self, hidden_states: Array, *, grid_thw: Array) -> Array:
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = self.post_conv_layernorm(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        grid_lens = grid_thw[:, 1] * grid_thw[:, 2]
        repeated = jnp.repeat(grid_lens, grid_thw[:, 0])
        cu_seqlens = jnp.cumsum(repeated, dtype="i4")
        cu_seqlens = jnp.pad(cu_seqlens, (1, 0), constant_values=0)

        max_grid_size = int(np.array(grid_thw)[:, 1:].max()) if grid_thw is not None else 1
        rotary_pos_emb = self.rot_pos_emb(grid_thw, max_grid_size=max_grid_size)
        rotary_pos_emb = jnp.concatenate([rotary_pos_emb, rotary_pos_emb], axis=-1)

        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

        hidden_states = self.post_layernorm(hidden_states)

        merge_size = self.spatial_merge_size
        hidden_states = hidden_states.reshape(-1, merge_size, merge_size, hidden_states.shape[-1])
        hidden_states = self.downsample(hidden_states).reshape(-1, self.config.out_hidden_size)

        hidden_states = self.merger(hidden_states)
        return hidden_states

    def get_encoder(self):
        return self

    def get_decoder(self):
        raise NotImplementedError("Vision model does not have a decoder.")

    def get_lm_head(self):
        raise NotImplementedError("Vision model does not have a language model head.")

    def get_embedding(self):
        return self.patch_embed


class Glm4vTextMLP(nn.Module):
    """SwiGLU feed-forward network for GLM4V text decoder."""

    def __init__(
        self,
        config: Glm4vTextConfig,
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

        self.gate_up_proj = ColumnParallelLinear(
            config.hidden_size,
            2 * config.intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(self, hidden_states: Array) -> Array:
        gate_up_states = checkpoint_name(self.gate_up_proj(hidden_states), name="mlp_gate_up")
        gate, up_states = jnp.split(gate_up_states, 2, axis=-1)
        hidden_states = checkpoint_name(self.down_proj(up_states * self.act_fn(gate)), name="mlp_down")
        return hidden_states


class Glm4vTextAttention(UnifiedAttention):
    """GLM4V attention with bias-free output projection (HF-compatible)."""

    def __init__(
        self,
        config: Glm4vTextConfig,
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
        )

    def _create_rotary(self, config: Glm4vTextConfig, dtype: jnp.dtype):
        # HF Glm4vText uses GPT-J style rotary (even/odd rotation) with partial rotary factor.
        return config.get_basic_rope(
            dtype=dtype,
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            base=getattr(config, "rope_theta", 10000.0),
            is_neox_style=False,
        )

    def _create_o_proj(
        self,
        config: Glm4vTextConfig,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        precision: jax.lax.PrecisionLike,
        rngs: nn.Rngs,
    ) -> RowParallelLinear:
        return RowParallelLinear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            rngs=rngs,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(getattr(config, "initializer_range", 0.02)),
            precision=precision,
        )


class Glm4vTextDecoderLayer(nn.Module):
    """Single GLM4V text decoder block combining attention and MLP."""

    def __init__(
        self,
        config: Glm4vTextConfig,
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

        attn_block, mlp_block = auto_remat(
            Glm4vTextAttention,
            Glm4vTextMLP,
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
        )

        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.post_self_attn_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.post_mlp_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ) -> DecoderLayerOutput:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
            frequencies=frequencies,
        )
        hidden_states = self.post_self_attn_layernorm(attn_outputs.attention_output)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.config.use_scan_mlp:
            feed_forward_hidden_states = block_wise_ffn(self.mlp, hidden_states, self.config.scan_mlp_chunk_size)
        else:
            feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_layernorm(feed_forward_hidden_states)
        hidden_states = residual + hidden_states

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


@register_module(TaskType.BASE_MODULE, config=Glm4vTextConfig, model_type="glm4v")
class Glm4vTextModel(EasyDeLBaseModule):
    """GLM4V text decoder model."""

    config_class = Glm4vTextConfig

    def __init__(
        self,
        config: Glm4vTextConfig,
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
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.layers = [
            Glm4vTextDecoderLayer(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
                layer_idx=i,
            )
            for i in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs)

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
    ) -> BaseModelOutput:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.astype("i4"))

        sequence_length = inputs_embeds.shape[1]
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

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
        raise NotImplementedError("Text model does not have an encoder.")

    def get_decoder(self):
        return self

    def get_lm_head(self):
        raise NotImplementedError("Text model does not have a language model head.")

    def get_embedding(self):
        return self.embed_tokens


@register_module(TaskType.VISION_LM, config=Glm4vConfig, model_type="glm4v")
class Glm4vModel(EasyDeLBaseModule):
    """GLM4V multimodal model integrating vision encoder and text decoder."""

    def __init__(
        self,
        config: Glm4vConfig,
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
        self.visual = Glm4vVisionModel(
            config.vision_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.language_model = Glm4vTextModel(
            config.text_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def get_input_embeddings(self):
        return self.language_model.get_embedding()

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def get_decoder(self):
        return self.language_model

    def get_rope_index(
        self,
        *,
        input_ids: Array | None = None,
        image_grid_thw: Array | None = None,
        video_grid_thw: Array | None = None,
        attention_mask: Array | None = None,
    ) -> tuple[Array, Array]:
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_start_token_id = self.config.video_start_token_id
        video_end_token_id = self.config.video_end_token_id

        if input_ids is None:
            raise ValueError("`input_ids` must be provided to compute RoPE indices.")

        batch_size, seq_length = input_ids.shape

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids, dtype=jnp.int32)
        else:
            attention_mask = attention_mask.astype(jnp.int32)

        position_ids = np.ones((3, batch_size, seq_length), dtype=np.int32)
        mrope_position_deltas: list[int] = []

        image_index = 0
        video_index = 0
        video_group_index = 0

        if image_grid_thw is None and video_grid_thw is None:
            # Text-only fallback
            pos = jnp.cumsum(attention_mask, axis=-1) - 1
            pos = jnp.where(attention_mask == 0, 1, pos).astype(jnp.int32)
            position_ids = jnp.broadcast_to(pos[None, ...], (3, batch_size, seq_length))
            max_pos = position_ids.max(axis=(0, 2), keepdims=True)
            deltas = (max_pos + 1 - seq_length).reshape(-1, 1).astype(jnp.int32)
            return position_ids, deltas

        total_input_ids = np.array(input_ids)
        total_attention_mask = np.array(attention_mask)
        image_grid = np.array(image_grid_thw) if image_grid_thw is not None else None
        video_grid = np.array(video_grid_thw) if video_grid_thw is not None else None

        for i in range(batch_size):
            valid = total_attention_mask[i] == 1
            tokens = total_input_ids[i][valid].tolist()

            token_types: list[str] = []
            video_check_flg = False
            for token in tokens:
                if token == video_start_token_id:
                    video_check_flg = True
                elif token == video_end_token_id:
                    video_check_flg = False

                if token == image_token_id and not video_check_flg:
                    token_types.append("image")
                elif token == image_token_id and video_check_flg:
                    token_types.append("video")
                else:
                    token_types.append("text")

            input_type_group = []
            for key, group in groupby(enumerate(token_types), key=lambda x: x[1]):
                group = list(group)
                start_idx = group[0][0]
                end_idx = group[-1][0] + 1
                input_type_group.append((key, start_idx, end_idx))

            llm_pos_ids_list: list[np.ndarray] = []
            video_frame_num = 1
            for modality_type, start_idx, end_idx in input_type_group:
                st_idx = int(llm_pos_ids_list[-1].max() + 1) if len(llm_pos_ids_list) > 0 else 0

                if modality_type == "image":
                    assert image_grid is not None, "`image_grid_thw` must be provided for image tokens."
                    t, h, w = image_grid[image_index]
                    llm_grid_t = int(t)
                    llm_grid_h = int(h) // spatial_merge_size
                    llm_grid_w = int(w) // spatial_merge_size

                    t_index = np.arange(llm_grid_t)[:, None].repeat(llm_grid_h * llm_grid_w, axis=1).reshape(-1)
                    h_index = (
                        np.arange(llm_grid_h)[None, :, None]
                        .repeat(llm_grid_t, axis=0)
                        .repeat(llm_grid_w, axis=2)
                        .reshape(-1)
                    )
                    w_index = (
                        np.arange(llm_grid_w)[None, None, :]
                        .repeat(llm_grid_t, axis=0)
                        .repeat(llm_grid_h, axis=1)
                        .reshape(-1)
                    )
                    llm_pos_ids_list.append(np.stack([t_index, h_index, w_index], axis=0) + st_idx)
                    image_index += 1
                    video_frame_num = 1
                elif modality_type == "video":
                    assert video_grid is not None, "`video_grid_thw` must be provided for video tokens."
                    _t = video_frame_num
                    h = int(video_grid[video_index][1])
                    w = int(video_grid[video_index][2])
                    llm_grid_t = int(_t)
                    llm_grid_h = h // spatial_merge_size
                    llm_grid_w = w // spatial_merge_size

                    for t_idx in range(llm_grid_t):
                        t_index = np.full((llm_grid_h * llm_grid_w,), t_idx, dtype=np.int32)
                        h_index = (
                            np.arange(llm_grid_h)[None, :, None].repeat(1, axis=0).repeat(llm_grid_w, axis=2).reshape(-1)
                        )
                        w_index = (
                            np.arange(llm_grid_w)[None, None, :].repeat(1, axis=0).repeat(llm_grid_h, axis=1).reshape(-1)
                        )
                        llm_pos_ids_list.append(np.stack([t_index, h_index, w_index], axis=0) + st_idx)

                    video_group_index += 1
                    if video_group_index >= int(video_grid[video_index][0]):
                        video_index += 1
                        video_group_index = 0
                    video_frame_num += 1
                else:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(np.arange(text_len, dtype=np.int32)[None, :].repeat(3, axis=0) + st_idx)
                    video_frame_num = 1

            llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
            position_ids[:, i, valid] = llm_positions
            mrope_position_deltas.append(int(llm_positions.max() + 1 - seq_length))

        deltas = np.array(mrope_position_deltas, dtype=np.int32).reshape(-1, 1)
        return jnp.asarray(position_ids), jnp.asarray(deltas)

    def get_video_features(self, pixel_values_videos: Array, video_grid_thw: Array | None = None):
        pixel_values_videos = pixel_values_videos.astype(self.visual.get_dtype())
        if video_grid_thw is None:
            raise ValueError("`video_grid_thw` must be provided when `pixel_values_videos` is not None.")

        video_grid = np.array(video_grid_thw)
        temp_frames_hw = []
        for t, h, w in video_grid:
            temp_frames_hw.append(np.tile(np.array([1, int(h), int(w)], dtype=video_grid.dtype), (int(t), 1)))
        flattened_video_grid_thw = np.concatenate(temp_frames_hw, axis=0)

        video_embeds = self.visual(pixel_values_videos, grid_thw=jnp.asarray(flattened_video_grid_thw))
        split_sizes = (video_grid.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        indices = np.cumsum(split_sizes)[:-1]
        video_embeds = jnp.split(video_embeds, indices) if len(indices) > 0 else (video_embeds,)
        return video_embeds

    def get_image_features(self, pixel_values: Array, image_grid_thw: Array | None = None):
        pixel_values = pixel_values.astype(self.visual.get_dtype())
        if image_grid_thw is None:
            raise ValueError("`image_grid_thw` must be provided when `pixel_values` is not None.")
        image_grid = np.array(image_grid_thw)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        indices = np.cumsum(split_sizes)[:-1]
        image_embeds = jnp.split(image_embeds, indices) if len(indices) > 0 else (image_embeds,)
        return image_embeds

    def compute_embedding(
        self,
        input_ids: Int[Array, "batch seq_len"],
        *,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        pixel_values: Array | None = None,
        pixel_values_videos: Array | None = None,
        image_grid_thw: Array | None = None,
        video_grid_thw: Array | None = None,
        image_embeds: Array | None = None,
        video_embeds: Array | None = None,
        **kwargs,
    ) -> Array:
        if input_ids is None:
            raise ValueError("`input_ids` must be provided when calling `compute_embedding`.")
        if pixel_values is not None and pixel_values_videos is not None:
            raise ValueError("Passing both `pixel_values` and `pixel_values_videos` is not supported.")

        if inputs_embeds is None:
            inputs_embeds = super().compute_embedding(input_ids)

        if image_embeds is None and pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            if isinstance(image_embeds, (list, tuple)):
                image_embeds = jnp.concatenate(image_embeds, axis=0)

        if image_embeds is not None:
            inputs_embeds = BaseVisionLanguageModule.merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=image_embeds.astype(inputs_embeds.dtype),
                placeholder_token_id=self.config.image_token_id,
            )

        if video_embeds is None and pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            if isinstance(video_embeds, (list, tuple)):
                video_embeds = jnp.concatenate(video_embeds, axis=0)

        if video_embeds is not None:
            inputs_embeds = BaseVisionLanguageModule.merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=video_embeds.astype(inputs_embeds.dtype),
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
        inputs_embeds = self.compute_embedding(input_ids, **kwargs)
        position_ids, rope_deltas = self.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=kwargs.get("image_grid_thw") if kwargs.get("pixel_values") is not None else None,
            video_grid_thw=kwargs.get("video_grid_thw") if kwargs.get("pixel_values_videos") is not None else None,
            attention_mask=attention_mask,
        )
        return inputs_embeds, EmbeddingInfo(position_ids=position_ids, rope_deltas=rope_deltas)

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        pixel_values: Array | None = None,
        pixel_values_videos: Array | None = None,
        image_grid_thw: Array | None = None,
        video_grid_thw: Array | None = None,
        rope_deltas: Array | None = None,
        cache_position: Array | None = None,
        mask_info: MaskInfo | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        **kwargs,
    ) -> Glm4vModelOutputWithPast:
        del cache_position

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if inputs_embeds is None:
            inputs_embeds = self.compute_embedding(
                input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
            )

        if position_ids is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw if pixel_values is not None else None,
                video_grid_thw=video_grid_thw if pixel_values_videos is not None else None,
                attention_mask=attention_mask,
            )
        elif position_ids.ndim == 2:
            batch_size, seq_len = position_ids.shape
            position_ids = jnp.broadcast_to(position_ids[None, :, :], (3, batch_size, seq_len))

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
        )

        return Glm4vModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=rope_deltas,
        )

    def get_encoder(self):
        return self.visual

    def get_lm_head(self):
        raise NotImplementedError("Glm4vModel does not have a language model head.")

    def get_embedding(self):
        return self.language_model.embed_tokens


@register_module(TaskType.IMAGE_TEXT_TO_TEXT, config=Glm4vConfig, model_type="glm4v")
class Glm4vForConditionalGeneration(BaseVisionLanguageModule[Glm4vModel, Glm4vConfig]):
    """GLM4V model for conditional generation."""

    _task_type = TaskType.IMAGE_TEXT_TO_TEXT
    _model_type = "glm4v"
    _config_class = Glm4vConfig
    _auto_register = False
    _supports_video = True
    _uses_mrope = True

    _vision_tower_name = "visual"
    _projector_name = "merger"
    _language_model_name = "language_model"

    loss_type = "ForCausalLM"

    def __init__(
        self,
        config: Glm4vConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(
            config=config,
            base_model_class=Glm4vModel,
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

    @property
    def visual(self):
        return self.base_model.visual

    @property
    def language_model(self):
        return self.base_model.language_model

    def get_video_features(self, pixel_values_videos: Array, video_grid_thw: Array | None = None, **kwargs):
        return self.base_model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: Array, image_grid_thw: Array | None = None, **kwargs):
        return self.base_model.get_image_features(pixel_values, image_grid_thw)

    def compute_embedding(self, input_ids, *args, **kwargs):
        return self.base_model.compute_embedding(input_ids, *args, **kwargs)

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
        pixel_values: Array | None = None,
        pixel_values_videos: Array | None = None,
        image_grid_thw: Array | None = None,
        video_grid_thw: Array | None = None,
        rope_deltas: Array | None = None,
        cache_position: Array | None = None,
        **kwargs,
    ) -> VLMCausalLMOutput:
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
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
        )

        hidden_states = apply_logical_sharding(
            outputs.last_hidden_state,
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
        )


__all__ = [
    "Glm4vConfig",
    "Glm4vForConditionalGeneration",
    "Glm4vModel",
    "Glm4vModelOutputWithPast",
    "Glm4vTextConfig",
    "Glm4vTextModel",
    "Glm4vVisionConfig",
    "Glm4vVisionModel",
]
