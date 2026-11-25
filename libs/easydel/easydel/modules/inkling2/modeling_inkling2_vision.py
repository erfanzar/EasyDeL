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

"""Vision encoder and 2x2 spatial-merge projector for the Inkling 2 family.

Implements Thinking Machines' ViT encoder using EasyDeL's shared primitives:

- Patch coordinates and bilinear position-embedding resampling from
  :mod:`easydel.layers.rotary` (:func:`vision_grid_position_ids`,
  :func:`vision_grid_pos_embed_interpolation_indices_and_weights`,
  :func:`apply_rope_interleaved`).
- Multi-head attention via :class:`~easydel.layers.attention.AttentionModule`
  and :class:`~easydel.layers.attention.FlexibleAttentionModule` (stage-mesh
  aware under MPMD and wired for ``attention_dropout``).
- Layer-stack scanning via :meth:`~spectrax.nn.ModuleList.scan`.
"""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
import spectrax as spx
from eformer.pytree import auto_pytree
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from jaxtyping import Array, Float, Int
from spectrax import common_types, nn

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import ModelOutput
from easydel.infra.sharding import sharding_for_layout
from easydel.infra.utils import ACT2FN, ArrayParam, auto_remat
from easydel.layers import ColumnParallelLinear, Embed, RowParallelLinear
from easydel.layers.attention import AttentionMechanisms, AttentionModule, FlexibleAttentionModule
from easydel.layers.norms import LayerNorm
from easydel.layers.rotary import (
    apply_rope_interleaved,
    vision_grid_pos_embed_interpolation_indices_and_weights,
    vision_grid_position_ids,
)

from ._checkpoint import INKLING2_VISION_KEY_RULES, Inkling2KeyNormalizer
from .inkling2_configuration import Inkling2VisionConfig


def apply_rotary_pos_emb_vision(
    q: Array,
    k: Array,
    cos: Array,
    sin: Array,
) -> tuple[Array, Array]:
    """Apply interleaved axial 2-D rotary embeddings in float32 using :func:`apply_rope_interleaved`."""
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    cos_h = cos[:, None, :].astype(jnp.float32)
    sin_h = sin[:, None, :].astype(jnp.float32)
    q_rot = apply_rope_interleaved(q.astype(jnp.float32), cos_h, sin_h).astype(orig_q_dtype)
    k_rot = apply_rope_interleaved(k.astype(jnp.float32), cos_h, sin_h).astype(orig_k_dtype)
    return q_rot, k_rot


class Inkling2VisionRotaryEmbedding(spx.Module):
    """Axial 2-D rotary embedding for Inkling 2 vision tokens (width half then height half)."""

    def __init__(self, config: Inkling2VisionConfig):
        head_dim = config.hidden_size // config.num_attention_heads
        if head_dim % 4 != 0:
            raise ValueError(f"Vision head_dim ({head_dim}) must be divisible by 4 for axial 2-D RoPE.")
        self.dim = head_dim // 2
        self.theta = float(config.rope_theta)

    def forward(self, positions: Int[Array, "seq_len 2"]) -> tuple[Array, Array]:
        """Compute half-width ``(cos, sin)`` tables of shape ``[seq_len, head_dim // 2]``."""
        inv_freq = 1.0 / (self.theta ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        pos_f32 = positions.astype(jnp.float32)
        # Upstream Inkling 2 packs width frequencies first and height frequencies second.
        freqs_h = pos_f32[:, 0:1] * inv_freq[None, :]
        freqs_w = pos_f32[:, 1:2] * inv_freq[None, :]
        freqs = jnp.concatenate([freqs_w, freqs_h], axis=-1)  # [seq_len, head_dim // 2]
        return jnp.cos(freqs), jnp.sin(freqs)


class Inkling2VisionAttention(AttentionModule):
    """Multi-head bidirectional self-attention for packed Inkling 2 vision sequences."""

    def __init__(
        self,
        config: Inkling2VisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        super().__init__(config=config)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"Vision hidden_size ({self.embed_dim}) must be divisible by num_attention_heads ({self.num_heads})."
            )
        self.scaling = float(self.head_dim**-0.5)
        init = jax.nn.initializers.normal(config.initializer_range)
        self.q_proj = ColumnParallelLinear(
            self.embed_dim,
            self.embed_dim,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=init,
            precision=precision,
            rngs=rngs,
        )
        self.k_proj = ColumnParallelLinear(
            self.embed_dim,
            self.embed_dim,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=init,
            precision=precision,
            rngs=rngs,
        )
        self.v_proj = ColumnParallelLinear(
            self.embed_dim,
            self.embed_dim,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=init,
            precision=precision,
            rngs=rngs,
        )
        self.out_proj = RowParallelLinear(
            self.embed_dim,
            self.embed_dim,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=init,
            precision=precision,
            rngs=rngs,
        )
        # Packed vision sequences include per-image CLS tokens whose segment
        # lengths are arbitrary (not multiples of 128), so route through
        # EasyDeL's registered vanilla backend via FlexibleAttentionModule.
        self.attention_performer = FlexibleAttentionModule(
            base_config=config,
            softmax_scale=self.scaling,
            dropout_prob=config.attention_dropout,
            attn_mechanism=AttentionMechanisms.VANILLA,
            requires_cache=False,
            rngs=rngs,
        )
        self.attention_performer.impl_decode = None

    def forward(
        self,
        hidden_states: Float[Array, "seq_len embed_dim"],
        mask_info: MaskInfo,
        position_embeddings: tuple[Array, Array],
        output_attentions: bool = False,
    ) -> tuple[Array, Array | None]:
        seq_len = hidden_states.shape[0]
        q = self.q_proj(hidden_states).reshape(seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(seq_len, self.num_heads, self.head_dim)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        attn_res = self.attention_performer.forward(
            query_states=q[None, ...],
            key_states=k[None, ...],
            value_states=v[None, ...],
            mode=common_types.MODE_TRAIN,
            mask_info=mask_info,
            causal=False,
            output_attentions=output_attentions,
            precision=self.precision,
            compute_dtype=self.dtype,
        )
        attn_out = attn_res.attention_outputs[0].reshape(seq_len, self.embed_dim)
        attn_out = self.out_proj(attn_out)
        return attn_out, (attn_res.attention_weights if output_attentions else None)


class Inkling2VisionMLP(spx.Module):
    """Two-layer GELU MLP for Inkling 2 vision encoder blocks."""

    def __init__(
        self,
        config: Inkling2VisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        self.activation_fn = ACT2FN[config.hidden_act]
        init = jax.nn.initializers.normal(config.initializer_range)
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=init,
            precision=precision,
            rngs=rngs,
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=init,
            precision=precision,
            rngs=rngs,
        )

    def forward(self, hidden_states: Array) -> Array:
        return self.fc2(self.activation_fn(self.fc1(hidden_states)))


class Inkling2VisionEncoderLayer(spx.Module):
    """Pre-LN vision transformer block with LayerScale parameters ``lambda_1`` and ``lambda_2``."""

    def __init__(
        self,
        config: Inkling2VisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        self.self_attn = Inkling2VisionAttention(
            config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs
        )
        self.mlp = Inkling2VisionMLP(config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)
        self.layernorm_before = LayerNorm(
            config.hidden_size, epsilon=config.layer_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.layernorm_after = LayerNorm(
            config.hidden_size, epsilon=config.layer_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.lambda_1 = spx.Parameter(
            jnp.full((config.hidden_size,), config.layer_scale_init_value, dtype=param_dtype),
            sharding=sharding_for_layout(common_types.Replicated),
        )
        self.lambda_2 = spx.Parameter(
            jnp.full((config.hidden_size,), config.layer_scale_init_value, dtype=param_dtype),
            sharding=sharding_for_layout(common_types.Replicated),
        )

    def forward(
        self,
        hidden_states: Array,
        mask_info: MaskInfo,
        position_embeddings: tuple[Array, Array],
        output_attentions: bool = False,
    ) -> tuple[Array, Array | None]:
        residual = hidden_states
        attn_out, attn_weights = self.self_attn(
            self.layernorm_before(hidden_states),
            mask_info=mask_info,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )
        hidden_states = residual + self.lambda_1.value.astype(attn_out.dtype) * attn_out

        residual = hidden_states
        mlp_out = self.mlp(self.layernorm_after(hidden_states))
        hidden_states = residual + self.lambda_2.value.astype(mlp_out.dtype) * mlp_out
        return hidden_states, attn_weights


class Inkling2VisionEmbeddings(spx.Module):
    """Linear patch embedding + bilinearly interpolated 2-D position embeddings + per-image CLS token."""

    def __init__(
        self,
        config: Inkling2VisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        self.patch_size = config.patch_size
        self.merge_size = config.spatial_merge_size
        self.grid_size = config.posemb_grid_size
        self.patch_in_features = config.num_channels * (config.patch_size**2)
        self.patch_embedding = ColumnParallelLinear(
            self.patch_in_features,
            config.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.position_embedding = Embed(
            num_embeddings=config.posemb_grid_size**2,
            features=config.hidden_size,
            embedding_init=jax.nn.initializers.normal(config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.class_embedding = ArrayParam.bound(
            shape=(config.hidden_size,),
            dtype=param_dtype,
            init_method="normal",
            init_kwargs={"stddev": config.initializer_range},
            key=rngs.param,
        )
        self.cls_position_embedding = ArrayParam.bound(
            shape=(config.hidden_size,),
            dtype=param_dtype,
            init_method="normal",
            init_kwargs={"stddev": config.initializer_range},
            key=rngs.param,
        )

    def forward(
        self,
        pixel_values: Array,
        patch_grid_thw_np: np.ndarray,
        patch_index: Array,
    ) -> Array:
        merge, patch = self.merge_size, self.patch_size
        channels = pixel_values.shape[-1]
        # (num_image_tokens, merge*patch, merge*patch, C) -> (num_image_tokens * merge^2, patch^2 * C)
        patches = pixel_values.reshape(-1, merge, patch, merge, patch, channels).transpose(0, 1, 3, 2, 4, 5)
        patch_embeddings = self.patch_embedding(patches.reshape(-1, self.patch_in_features))

        interp_indices_np, interp_weights_np = vision_grid_pos_embed_interpolation_indices_and_weights(
            patch_grid_thw_np,
            num_grid_per_side=self.grid_size,
            spatial_merge_size=merge,
            align_corners=False,
        )
        interp_indices = jnp.asarray(interp_indices_np, dtype=jnp.int32)
        interp_weights = jnp.asarray(interp_weights_np, dtype=jnp.float32)
        pos_embs = self.position_embedding(interp_indices).astype(jnp.float32)
        position_embeddings = jnp.sum(pos_embs * interp_weights[:, :, None], axis=1)
        patch_embeddings = patch_embeddings + position_embeddings.astype(patch_embeddings.dtype)

        cls_embedding = (self.class_embedding.value + self.cls_position_embedding.value).astype(patch_embeddings.dtype)
        total_seq = patch_embeddings.shape[0] + int(patch_grid_thw_np.shape[0])
        embeddings = jnp.broadcast_to(cls_embedding[None, :], (total_seq, patch_embeddings.shape[-1]))
        return embeddings.at[patch_index].set(patch_embeddings)


class Inkling2VisionMerger(spx.Module):
    """Two-layer MLP projecting spatially merged ``merge^2 * vision_hidden`` patches to ``text_hidden_size``."""

    def __init__(
        self,
        config: Inkling2VisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        init = jax.nn.initializers.normal(config.initializer_range)
        self.linear_1 = ColumnParallelLinear(
            (config.spatial_merge_size**2) * config.hidden_size,
            config.projection_hidden_dim,
            use_bias=config.projector_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=init,
            precision=precision,
            rngs=rngs,
        )
        self.act = ACT2FN[config.hidden_act]
        self.linear_2 = RowParallelLinear(
            config.projection_hidden_dim,
            config.text_hidden_size,
            use_bias=config.projector_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=init,
            precision=precision,
            rngs=rngs,
        )

    def forward(self, image_features: Array) -> Array:
        return self.linear_2(self.act(self.linear_1(image_features)))


@auto_pytree
class Inkling2VisionModelOutput(ModelOutput):
    """Output of :class:`Inkling2VisionModel` containing encoder tokens and merged LM-width features."""

    last_hidden_state: Array
    pooler_output: Array
    hidden_states: tuple[Array, ...] | None = None
    attentions: tuple[Array, ...] | None = None


@register_module(TaskType.BASE_VISION, config=Inkling2VisionConfig, model_type="inkling2_vision")
@register_module(TaskType.BASE_MODULE, config=Inkling2VisionConfig, model_type="inkling2_vision")
class Inkling2VisionModel(EasyDeLBaseModule):
    """Inkling 2 vision encoder and spatial-merge projector."""

    checkpoint_key_rules: tp.ClassVar[Inkling2KeyNormalizer] = INKLING2_VISION_KEY_RULES

    def __init__(
        self,
        config: Inkling2VisionConfig,
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
        self.embeddings = Inkling2VisionEmbeddings(
            config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs
        )
        self.pre_layernorm = LayerNorm(
            config.hidden_size, epsilon=config.layer_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.rotary_emb = Inkling2VisionRotaryEmbedding(config)
        remat_layer = auto_remat(
            Inkling2VisionEncoderLayer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.layers = nn.ModuleList(
            [
                remat_layer(config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.merger = Inkling2VisionMerger(config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)

    def forward(
        self,
        pixel_values: Float[Array, "num_image_tokens merged_h merged_w channels"],
        image_grid_thw: Int[Array, "num_images 3"] | np.ndarray,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> Inkling2VisionModelOutput:
        merge = self.config.spatial_merge_size
        grid_np = np.asarray(image_grid_thw, dtype=np.int64)
        patch_grid_np = grid_np * np.array([1, merge, merge], dtype=np.int64)
        patches_per_image = patch_grid_np.prod(axis=-1)
        num_images = int(patch_grid_np.shape[0])
        num_patches = int(pixel_values.shape[0]) * (merge**2)
        total_seq = num_patches + num_images

        image_ids = np.arange(1, num_images + 1, dtype=np.int64)
        patch_index_np = np.arange(num_patches, dtype=np.int64) + np.repeat(image_ids, patches_per_image)
        patch_index = jnp.asarray(patch_index_np, dtype=jnp.int32)

        seg_lengths = patches_per_image + 1
        segment_ids_np = np.repeat(np.arange(num_images, dtype=np.int32), seg_lengths)
        mask_info = MaskInfo.from_segments(jnp.asarray(segment_ids_np)[None, :])

        pixel_values = pixel_values.astype(self.dtype)
        hidden_states = self.embeddings(pixel_values, patch_grid_np, patch_index)
        hidden_states = self.pre_layernorm(hidden_states)

        # Patches are 1-indexed (row, col); per-image CLS tokens sit at (0, 0).
        patch_pos_np = vision_grid_position_ids(patch_grid_np, merge)
        full_pos_np = np.zeros((total_seq, 2), dtype=np.int32)
        full_pos_np[patch_index_np] = patch_pos_np + 1
        position_embeddings = self.rotary_emb(jnp.asarray(full_pos_np, dtype=jnp.int32))

        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        trace_layers = self._layer_scan_trace(
            False,
            output_hidden_states=bool(output_hidden_states),
            output_attentions=bool(output_attentions),
        )

        def _layer_loop(block, carry):
            hs, ah, aa = carry
            if output_hidden_states:
                ah = (*ah, hs)
            hs, attn_w = block(
                hs,
                mask_info=mask_info,
                position_embeddings=position_embeddings,
                output_attentions=bool(output_attentions),
            )
            if output_attentions:
                aa = (*aa, attn_w)
            return hs, ah, aa

        hidden_states, all_hidden_states, all_attentions = self.layers.scan(
            _layer_loop,
            (hidden_states, all_hidden_states, all_attentions),
            trace=trace_layers,
            **self._layer_scan_kwargs("vision"),
        )
        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        patch_states = hidden_states[patch_index]
        pooler_output = self.merger(patch_states.reshape(-1, (merge**2) * hidden_states.shape[-1]))
        return Inkling2VisionModelOutput(
            last_hidden_state=hidden_states,
            pooler_output=pooler_output,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

    def get_encoder(self):
        return self

    def get_decoder(self):
        raise NotImplementedError("Inkling2VisionModel is an encoder and has no decoder.")

    def get_lm_head(self):
        raise NotImplementedError("Inkling2VisionModel has no language model head.")

    def get_embedding(self):
        return self.embeddings


__all__ = [
    "Inkling2VisionAttention",
    "Inkling2VisionEmbeddings",
    "Inkling2VisionEncoderLayer",
    "Inkling2VisionMLP",
    "Inkling2VisionMerger",
    "Inkling2VisionModel",
    "Inkling2VisionModelOutput",
    "Inkling2VisionRotaryEmbedding",
    "apply_rotary_pos_emb_vision",
]
