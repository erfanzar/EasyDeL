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

"""EasyDeL implementation of MoonshotAI Kimi-VL.

Implements the vision tower (MoonViT) and the multimodal wrapper around
DeepSeek-V3 (MoE) language model, matching the HuggingFace trust_remote_code
structure and parameter naming:

- `vision_tower.*`
- `multi_modal_projector.*`
- `language_model.*` (DeepseekV3ForCausalLM)
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cached_property

import jax
import jax.numpy as jnp
from eformer import common_types
from flax import nnx as nn
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import VLMCausalLMOutput
from easydel.layers.attention import FlexibleAttentionModule
from easydel.layers.base_modules import BaseVisionLanguageModule

from ..deepseek_v3.modeling_deepseek import DeepseekV3ForCausalLM
from .kimi_vl_configuration import KimiVLConfig, MoonViTConfig


def _create_block_diagonal_bias(cu_seqlens: Array, seq_length: int, dtype: jnp.dtype) -> Array:
    """Create block-diagonal attention bias from cumulative sequence lengths.

    Returns:
        Array[1, seq_length, seq_length] with 0 for allowed positions and -inf for disallowed.
    """
    positions = jnp.arange(seq_length)
    starts = cu_seqlens[:-1]
    ends = cu_seqlens[1:]
    in_segment = (positions[:, None] >= starts[None, :]) & (positions[:, None] < ends[None, :])
    segment_ids = jnp.argmax(in_segment.astype(jnp.int32), axis=-1)
    same_segment = segment_ids[:, None] == segment_ids[None, :]
    mask_value = jnp.finfo(dtype).min
    bias = jnp.where(same_segment, 0.0, mask_value).astype(dtype)
    return bias[None, :, :]


def _apply_rope(
    xq: Array,
    xk: Array,
    freqs_cis: Array,
) -> tuple[Array, Array]:
    """Apply complex RoPE to q/k.

    Args:
        xq, xk: (..., num_heads, head_dim)
        freqs_cis: (..., head_dim/2) complex
    """
    freqs_cis = freqs_cis[..., None, :]  # (..., 1, head_dim/2)

    def _to_complex(x: Array) -> Array:
        x = x.astype(jnp.float32).reshape(*x.shape[:-1], -1, 2)
        return jax.lax.complex(x[..., 0], x[..., 1])

    def _to_real(x: Array, dtype: jnp.dtype) -> Array:
        x = jnp.stack([jnp.real(x), jnp.imag(x)], axis=-1)
        return x.reshape(*x.shape[:-2], -1).astype(dtype)

    q_c = _to_complex(xq)
    k_c = _to_complex(xk)
    q_out = q_c * freqs_cis
    k_out = k_c * freqs_cis
    return _to_real(q_out, xq.dtype), _to_real(k_out, xk.dtype)


class Learnable2DInterpPosEmb(nn.Module):
    """Learnable 2D positional embeddings with cubic interpolation."""

    def __init__(
        self,
        height: int,
        width: int,
        dim: int,
        interpolation_mode: jax.image.ResizeMethod = jax.image.ResizeMethod.CUBIC,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: nn.Rngs,
    ) -> None:
        self.height = height
        self.width = width
        self.dim = dim
        self.interpolation_mode = interpolation_mode
        self.kernel = nn.Param(
            nn.initializers.normal()(rngs.params(), (dim, width, height), param_dtype),
        )
        self.dtype = dtype

    def __call__(self, x: Array, grid_hws: Array) -> Array:
        pos_embs = []
        kernel = self.kernel.value.astype(jnp.float32).transpose(2, 1, 0)
        for h, w in grid_hws:
            h = int(h)
            w = int(w)
            if (h, w) == (self.height, self.width):
                pos_embs.append(kernel.reshape(-1, self.dim))
            else:
                resized = jax.image.resize(
                    kernel,
                    shape=(h, w, self.dim),
                    method=self.interpolation_mode,
                    antialias=True,
                )
                pos_embs.append(resized.reshape(-1, self.dim))
        return x + jnp.concatenate(pos_embs, axis=0).astype(x.dtype)


class MoonVisionPatchEmbed(nn.Module):
    """Patch embedding for MoonViT (expects patchified inputs)."""

    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: int = 14,
        pos_emb_height: int = 64,
        pos_emb_width: int = 64,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike | None = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        self.patch_size = patch_size
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dtype = dtype

        self.proj = nn.Conv(
            in_features=in_dim,
            out_features=out_dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.pos_emb = Learnable2DInterpPosEmb(
            height=pos_emb_height,
            width=pos_emb_width,
            dim=out_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, pixel_values: Array, grid_hws: Array) -> Array:
        if pixel_values.ndim == 4:
            # Convert from NCHW -> NHWC for JAX conv.
            pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))
        x = self.proj(pixel_values.astype(self.dtype))
        x = x.reshape(x.shape[0], -1)
        return self.pos_emb(x, grid_hws)


class Rope2DPosEmb(nn.Module):
    """2D rotary position embedding (complex cis) with multi-resolution support."""

    def __init__(self, dim: int, max_height: int, max_width: int, theta_base: float = 10000.0):
        self.dim = dim
        if dim % 4 != 0:
            raise ValueError("dim must be divisible by 4")
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

    @cached_property
    def freqs_cis(self) -> Array:
        n = self.max_height * self.max_width
        flat_pos = jnp.arange(n, dtype=jnp.float32)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = jnp.arange(0, self.dim, 4, dtype=jnp.float32)[: self.dim // 4]
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = jnp.outer(x_pos, freqs).astype(jnp.float32)
        y_freqs = jnp.outer(y_pos, freqs).astype(jnp.float32)
        x_cis = jnp.exp(1j * x_freqs).astype(jnp.complex64)
        y_cis = jnp.exp(1j * y_freqs).astype(jnp.complex64)
        freqs_cis = jnp.concatenate([x_cis[..., None], y_cis[..., None]], axis=-1)
        return freqs_cis.reshape(self.max_height, self.max_width, -1)

    def get_freqs_cis(self, grid_hws: Array) -> Array:
        shapes = [(int(h), int(w)) for h, w in grid_hws]
        if not all(1 <= h <= self.max_height and 1 <= w <= self.max_width for h, w in shapes):
            raise ValueError(f"grid_hws out of range: {shapes} vs {(self.max_height, self.max_width)}")
        pieces = [self.freqs_cis[:h, :w].reshape(-1, self.dim // 2) for h, w in shapes]
        return jnp.concatenate(pieces, axis=0)


class MLP2(nn.Module):
    """Two-layer MLP used in MoonViT blocks (matches HF naming `fc0`/`fc1`)."""

    def __init__(
        self,
        dims: tuple[int, int, int],
        activation: Callable[[Array], Array],
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike | None = None,
        *,
        rngs: nn.Rngs,
    ):
        in_dim, hidden_dim, out_dim = dims
        self.fc0 = nn.Linear(
            in_dim,
            hidden_dim,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.fc1 = nn.Linear(
            hidden_dim,
            out_dim,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.activation = activation

    def __call__(self, x: Array) -> Array:
        return self.fc1(self.activation(self.fc0(x)))


class MoonVitEncoderLayer(nn.Module):
    """Transformer encoder layer for MoonViT."""

    def __init__(
        self,
        base_config: MoonViTConfig,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        activation: Callable[[Array], Array],
        attn_bias: bool = True,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike | None = None,
        *,
        rngs: nn.Rngs,
    ):
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.dtype = dtype

        self.norm0 = nn.LayerNorm(
            hidden_dim,
            epsilon=1e-5,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.norm1 = nn.LayerNorm(
            hidden_dim,
            epsilon=1e-5,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.mlp = MLP2(
            (hidden_dim, mlp_dim, hidden_dim),
            activation=activation,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.wqkv = nn.Linear(
            hidden_dim,
            hidden_dim * 3,
            use_bias=attn_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.wo = nn.Linear(
            hidden_dim,
            hidden_dim,
            use_bias=attn_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.attention_performer = FlexibleAttentionModule(
            rngs=rngs,
            base_config=base_config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=0.0,
            attn_mechanism="vanilla",
            requires_cache=False,
        )

    def _attention(self, x: Array, cu_seqlens: Array, rope_freqs_cis: Array) -> Array:
        seq_length = x.shape[0]
        qkv = self.wqkv(x)
        qkv = qkv.reshape(seq_length, 3, self.num_heads, self.head_dim)
        q = qkv[:, 0]
        k = qkv[:, 1]
        v = qkv[:, 2]

        q, k = _apply_rope(q, k, rope_freqs_cis)

        q = q[None, :, :, :]
        k = k[None, :, :, :]
        v = v[None, :, :, :]

        bias = _create_block_diagonal_bias(cu_seqlens, seq_length, q.dtype)
        bias = jnp.broadcast_to(bias, (1, self.num_heads, seq_length, seq_length))

        attn_out = self.attention_performer.forward(
            query_states=q,
            key_states=k,
            value_states=v,
            bias=bias,
            causal=False,
            mode=common_types.MODE_TRAIN if seq_length != 1 else common_types.MODE_DECODE,
        ).attention_outputs

        attn_out = attn_out.squeeze(0).reshape(seq_length, -1)
        return self.wo(attn_out)

    def __call__(self, hidden_states: Array, cu_seqlens: Array, rope_freqs_cis: Array) -> Array:
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)
        hidden_states = residual + self._attention(hidden_states, cu_seqlens, rope_freqs_cis)

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class MoonVitEncoder(nn.Module):
    """Vision transformer encoder for MoonViT (packed sequences)."""

    def __init__(
        self,
        base_config: MoonViTConfig,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_dim: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike | None = None,
        *,
        rngs: nn.Rngs,
    ):
        self.rope_2d = Rope2DPosEmb(hidden_dim // num_heads, 512, 512)

        def activation(x):
            return jax.nn.gelu(x, approximate=True)

        self.blocks = [
            MoonVitEncoderLayer(
                base_config=base_config,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                mlp_dim=mlp_dim,
                activation=activation,
                attn_bias=True,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for _ in range(num_layers)
        ]
        self.final_layernorm = nn.LayerNorm(
            hidden_dim,
            epsilon=1e-5,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, hidden_states: Array, grid_hws: Array) -> Array:
        rope_freqs_cis = self.rope_2d.get_freqs_cis(grid_hws)

        grid_lens = (grid_hws[:, 0] * grid_hws[:, 1]).astype(jnp.int32)
        lengths = jnp.concatenate([jnp.zeros((1,), dtype=jnp.int32), grid_lens], axis=0)
        cu_seqlens = jnp.cumsum(lengths, axis=0)

        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, rope_freqs_cis)

        return self.final_layernorm(hidden_states)


def patch_merger(
    hidden_states: Array,
    grid_hws: Array,
    merge_kernel_size: tuple[int, int],
) -> list[Array]:
    """Merge patches into spatial groups (matches HF `patch_merger`)."""
    d_model = hidden_states.shape[-1]
    kernel_h, kernel_w = merge_kernel_size
    outputs = []
    pre_sum = 0
    for h, w in grid_hws:
        h = int(h)
        w = int(w)
        seq = hidden_states[pre_sum : pre_sum + (h * w)]
        new_h, new_w = h // kernel_h, w // kernel_w
        reshaped = seq.reshape(new_h, kernel_h, new_w, kernel_w, d_model)
        reshaped = jnp.transpose(reshaped, (0, 2, 1, 3, 4))
        outputs.append(reshaped.reshape(new_h * new_w, kernel_h * kernel_w, d_model))
        pre_sum += h * w
    return outputs


class MoonVitPretrainedModel(nn.Module):
    """MoonViT vision tower used by Kimi-VL (matches HF `MoonVitPretrainedModel`)."""

    def __init__(
        self,
        config: MoonViTConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike | None = None,
        *,
        rngs: nn.Rngs,
    ):
        self.merge_kernel_size = tuple(config.merge_kernel_size)
        self.patch_size = config.patch_size
        self.patch_embed = MoonVisionPatchEmbed(
            out_dim=config.hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.init_pos_emb_height,
            pos_emb_width=config.init_pos_emb_width,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.encoder = MoonVitEncoder(
            base_config=config,
            hidden_dim=config.hidden_size,
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            mlp_dim=config.intermediate_size,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, pixel_values: Array, grid_hws: Array) -> list[Array]:
        hidden_states = self.patch_embed(pixel_values, grid_hws)
        hidden_states = self.encoder(hidden_states, grid_hws)
        return patch_merger(hidden_states, grid_hws, merge_kernel_size=self.merge_kernel_size)


class KimiVLMultiModalProjector(nn.Module):
    """Project MoonViT patch groups into DeepSeek-V3 hidden size."""

    def __init__(
        self,
        config: KimiVLConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike | None = None,
        *,
        rngs: nn.Rngs,
    ):
        merge_kernel = tuple(config.vision_config.merge_kernel_size)
        hidden_size = config.vision_config.hidden_size * merge_kernel[0] * merge_kernel[1]

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(
            config.vision_config.hidden_size,
            epsilon=1e-5,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.linear_1 = nn.Linear(
            hidden_size,
            hidden_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.linear_2 = nn.Linear(
            hidden_size,
            config.text_config.hidden_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, image_features: list[Array]) -> Array:
        image_features = jnp.concatenate(image_features, axis=0)
        hidden_states = self.pre_norm(image_features).reshape(-1, self.hidden_size)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = jax.nn.gelu(hidden_states, approximate=False)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


@register_module(TaskType.IMAGE_TEXT_TO_TEXT, config=KimiVLConfig, model_type="kimi_vl")
class KimiVLForConditionalGeneration(BaseVisionLanguageModule[DeepseekV3ForCausalLM, KimiVLConfig]):
    """Kimi-VL model for image-text to text generation."""

    _task_type = TaskType.IMAGE_TEXT_TO_TEXT
    _model_type = "kimi_vl"
    _config_class = KimiVLConfig
    _auto_register = False

    loss_type = "ForCausalLM"

    def __init__(
        self,
        config: KimiVLConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike | None = None,
        *,
        rngs: nn.Rngs,
    ):
        language_model = DeepseekV3ForCausalLM(
            config=config.text_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        super().__init__(
            config=config,
            base_model=language_model,
            base_model_name="language_model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            image_token_index=int(config.media_placeholder_token_id),
            tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
            create_lm_head=False,
        )
        self.vision_tower = MoonVitPretrainedModel(
            config=config.vision_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.multi_modal_projector = KimiVLMultiModalProjector(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def _merge_with_image_features(
        self,
        inputs_embeds: Array,
        input_ids: Array,
        image_features: Array,
    ) -> Array:
        placeholder = int(self.config.media_placeholder_token_id)
        multimodal_embeddings = image_features.reshape(-1, image_features.shape[-1])
        return BaseVisionLanguageModule.merge_multimodal_embeddings(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            placeholder_token_id=placeholder,
        )

    def _extract_image_features(self, pixel_values: Array, image_grid_hws: Array) -> Array:
        image_features = self.vision_tower(pixel_values, image_grid_hws)
        return self.multi_modal_projector(image_features)

    def get_image_features(
        self,
        pixel_values: Float[Array, "num_patches channels patch_h patch_w"],
        image_grid_hws: Int[Array, "num_images 2"] | None = None,
        **kwargs,
    ) -> Float[Array, "num_patches hidden_dim"]:
        if image_grid_hws is None:
            raise ValueError("`image_grid_hws` must be provided when `pixel_values` is not None.")
        return self._extract_image_features(pixel_values.astype(self.dtype), image_grid_hws)

    def compute_embedding(
        self,
        input_ids: Int[Array, "batch seq_len"],
        *,
        image_features: Array | None = None,
        pixel_values: Float[Array, "num_patches channels patch_h patch_w"] | None = None,
        image_grid_hws: Int[Array, "num_images 2"] | None = None,
        **kwargs,
    ) -> Array:
        if input_ids is None:
            raise ValueError("`input_ids` must be provided when calling `compute_embedding`.")

        placeholder = int(self.config.media_placeholder_token_id)
        vocab_size = int(getattr(self.config.text_config, "vocab_size", 0) or 0)
        if vocab_size and placeholder >= vocab_size:
            llm_input_ids = jnp.where(input_ids == placeholder, 0, input_ids)
        else:
            llm_input_ids = input_ids

        inputs_embeds = super().compute_embedding(llm_input_ids)

        if image_features is None and pixel_values is not None:
            if image_grid_hws is None:
                raise ValueError("`image_grid_hws` must be provided when `pixel_values` is not None.")
            image_features = self._extract_image_features(pixel_values.astype(self.dtype), image_grid_hws)

        if image_features is not None:
            inputs_embeds = self._merge_with_image_features(
                inputs_embeds,
                input_ids,
                image_features.astype(inputs_embeds.dtype),
            )

        return inputs_embeds

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: object | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: object | None = None,
        cache_metadata: object | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        pixel_values: Float[Array, "num_patches channels patch_h patch_w"] | None = None,
        image_grid_hws: Int[Array, "num_images 2"] | None = None,
    ) -> VLMCausalLMOutput:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        image_hidden_states = None
        image_features = None
        if pixel_values is not None:
            if input_ids is None:
                raise ValueError("`input_ids` must be provided when `pixel_values` is not None.")
            if image_grid_hws is None:
                raise ValueError("`image_grid_hws` must be provided when `pixel_values` is not None.")

            image_features = self._extract_image_features(pixel_values.astype(self.dtype), image_grid_hws)

        if inputs_embeds is None:
            inputs_embeds = self.compute_embedding(
                input_ids,
                image_features=image_features,
            )

        if image_features is not None:
            image_hidden_states = image_features.astype(inputs_embeds.dtype)

        outputs = self.language_model(
            input_ids=None,
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

        return VLMCausalLMOutput(
            logits=outputs.logits,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
            image_hidden_states=image_hidden_states,
            router_logits=getattr(outputs, "router_logits", None),
            aux_loss=getattr(outputs, "aux_loss", None),
            loss=getattr(outputs, "loss", None),
        )

    def init_cache(self, batch_size, max_length, starts=None, shardings=None, pad_token_id=None):
        return self.language_model.init_cache(batch_size, max_length, starts, shardings, pad_token_id)

    def prepare_inputs_for_generation(
        self,
        input_ids: Array,
        max_length: int,
        pad_token_id: int,
        starts: int | None = None,
        pixel_values: Array | None = None,
        attention_mask: Array | None = None,
        image_grid_hws: Array | None = None,
    ):
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            max_length=max_length,
            pad_token_id=pad_token_id,
            starts=starts,
            attention_mask=attention_mask,
        )
        model_inputs["pixel_values"] = pixel_values
        model_inputs["image_grid_hws"] = image_grid_hws
        return model_inputs

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs = self.language_model.update_inputs_for_generation(model_outputs, model_kwargs)
        model_kwargs.pop("pixel_values", None)
        model_kwargs.pop("image_grid_hws", None)
        return model_kwargs

    def get_encoder(self):
        return self.vision_tower

    def get_decoder(self):
        return self.language_model.get_decoder()

    def get_lm_head(self):
        return self.language_model.get_lm_head()

    def get_embedding(self):
        return self.language_model.get_embedding()

    def get_vision_tower(self) -> nn.Module:
        return self.vision_tower

    def get_projector(self) -> nn.Module:
        return self.multi_modal_projector

    def get_language_model(self) -> nn.Module:
        return self.language_model


__all__ = [
    "KimiVLForConditionalGeneration",
    "KimiVLMultiModalProjector",
    "MoonVitPretrainedModel",
]
