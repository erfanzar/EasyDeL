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

"""DeepSeek-V4.1 vision tower and aligner (reference ``inference/vision.py``).

The ViT embeds ``14 x 14 x 3`` patches with a biased linear layer, runs pre-norm blocks of full bidirectional
attention (2D rotate-half RoPE over the patch grid) and a SwiGLU MLP, then applies a final RMSNorm. The aligner folds
``r x r`` neighbouring patch features (channel-major ``(c, kh, kw)`` order, right/bottom zero padding to a multiple of
``r``) and maps them to the text width with ``w2(gelu(w1(x)))``.

Module/leaf names mirror the released checkpoint exactly (``vision.patch_embed.proj``, ``vision.blocks.N.{norm1, attn.
wqkv, attn.wo, norm2, mlp.w1, mlp.w2}``, ``vision.norm``, ``aligner.{w1, w2}``), so mapping a native key only needs the
model prefix.

Image-span token types (``ref_dl/inference/image_processor.py``): every position of a span carries ``image_token_id``
in ``input_ids`` and is told apart by its type; the span is ``[IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) *
n_llm_h + [IMAGE_END]``.

Attention math is delegated to the shared ``FlexibleAttentionModule`` with the vanilla backend, float32 Q/K/V and
softmax accumulation, explicit ``head_dim ** -0.5`` scaling, and bidirectional visibility. Long patch sequences retain
query chunking (each chunk attends to every key). The 2D grid-specific rotate-half RoPE remains local because it assigns
independent frequency lanes to row and column coordinates.
"""

from __future__ import annotations

import functools
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
import spectrax as spx
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Float
from spectrax import common_types, nn

from easydel.infra.utils import auto_remat
from easydel.layers import ColumnParallelLinear, RMSNorm, RowParallelLinear
from easydel.layers.attention import FlexibleAttentionModule

from .deepseek_v41_configuration import DeepseekV41Config

#: Token types of ``image_processor.py``: text is ``-1``; the others are positions inside an image span.
TOKEN_TYPE_TEXT = -1
TOKEN_TYPE_IMAGE_START = 0
TOKEN_TYPE_IMAGE = 1
TOKEN_TYPE_IMAGE_NEW_LINE = 2
TOKEN_TYPE_IMAGE_END = 3

#: Queries per attention chunk (bounds the ``[heads, chunk, patches]`` score tensor of large images).
_ATTN_QUERY_CHUNK = 1024


def llm_grid(n_vit_h: int, n_vit_w: int, downsample_ratio: int) -> tuple[int, int]:
    """Token grid the aligner produces from a ``n_vit_h x n_vit_w`` patch grid (ceil division).

    Args:
        n_vit_h: Patch rows.
        n_vit_w: Patch columns.
        downsample_ratio: Aligner fold size ``r``.

    Returns:
        ``(n_llm_h, n_llm_w)``.
    """
    return -(-n_vit_h // downsample_ratio), -(-n_vit_w // downsample_ratio)


@functools.lru_cache(maxsize=16)
def vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """2D RoPE tables of the patch grid (reference ``get_vision_cos_sin``), float32 NumPy.

    Args:
        n_h: Patch rows.
        n_w: Patch columns.
        dim: Rotary width (``head_dim // 2``): half is driven by the row index, half by the column index.
        theta: RoPE base.

    Returns:
        ``(cos, sin)``, each ``[n_h * n_w, 1, dim]`` in row-major patch order.
    """
    inv_freq = 1.0 / (np.float32(theta) ** (np.arange(0, dim, 2, dtype=np.float32) / np.float32(dim)))
    hpos = np.broadcast_to(np.arange(n_h)[:, None], (n_h, n_w))
    wpos = np.broadcast_to(np.arange(n_w)[None, :], (n_h, n_w))
    pos = np.stack([hpos, wpos], axis=-1).reshape(-1, 2, 1).astype(np.float32)
    freqs = (pos * inv_freq).reshape(n_h * n_w, -1)
    return np.cos(freqs)[:, None, :], np.sin(freqs)[:, None, :]


def _apply_rotary(x: Array, cos: Array, sin: Array) -> Array:
    """Rotate-half RoPE over the full head (float32 math, result in ``x.dtype``); ``x`` is ``[N, P, H, D]``."""
    x1, x2 = jnp.split(x.astype(jnp.float32), 2, axis=-1)
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)


def _linear_kwargs(
    initializer_range: float, dtype, param_dtype, precision, rngs, *, use_bias: bool
) -> dict[str, tp.Any]:
    """Common ColumnParallel/RowParallel constructor arguments."""
    return dict(
        use_bias=use_bias,
        dtype=dtype,
        param_dtype=param_dtype,
        precision=precision,
        kernel_init=jax.nn.initializers.normal(initializer_range),
        rngs=rngs,
    )


class DeepseekV41VisionPatchEmbed(spx.Module):
    """``Linear(3 * patch ** 2 -> hidden)`` with bias over flattened ``[3, patch, patch]`` patches."""

    def __init__(
        self,
        config: DeepseekV41Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        params = config.vision_params()
        self.proj = ColumnParallelLinear(
            3 * params["patch_size"] ** 2,
            params["hidden_size"],
            **_linear_kwargs(config.initializer_range, dtype, param_dtype, precision, rngs, use_bias=True),
        )

    def forward(self, patches: Float[Array, "images patches patch_dim"]) -> Array:
        """Embed patches ``[N, P, 3 * patch ** 2]`` to ``[N, P, hidden]``."""
        return self.proj(patches)


class DeepseekV41VisionAttention(spx.Module):
    """Full bidirectional multi-head attention with fused biased ``wqkv`` / ``wo`` and 2D rotate-half RoPE."""

    def __init__(
        self,
        config: DeepseekV41Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        params = config.vision_params()
        hidden, heads = params["hidden_size"], params["num_attention_heads"]
        if hidden % heads or (hidden // heads) % 4:
            raise ValueError(
                f"vision hidden_size {hidden} must split into {heads} heads whose width is a multiple of 4 "
                "(2D RoPE rotates half the head by row and half by column)."
            )
        self.precision = precision
        self.num_heads = heads
        self.head_dim = hidden // heads
        kwargs = _linear_kwargs(config.initializer_range, dtype, param_dtype, precision, rngs, use_bias=True)
        self.wqkv = ColumnParallelLinear(hidden, 3 * hidden, **kwargs)
        self.wo = RowParallelLinear(hidden, hidden, **kwargs)
        self.attention_performer = FlexibleAttentionModule(
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=0.0,
            rngs=rngs,
            attn_mechanism="vanilla",
            requires_cache=False,
        )

    def forward(self, hidden_states: Array, cos: Array, sin: Array) -> Array:
        """Attend over all patches of each image.

        Args:
            hidden_states: ``[N, P, hidden]``.
            cos: ``[P, 1, head_dim // 2]`` rotary cosine.
            sin: ``[P, 1, head_dim // 2]`` rotary sine.

        Returns:
            ``[N, P, hidden]``.
        """
        n, p, _ = hidden_states.shape
        q, k, v = (t.reshape(n, p, self.num_heads, self.head_dim) for t in jnp.split(self.wqkv(hidden_states), 3, -1))
        q = _apply_rotary(q, cos, sin).astype(jnp.float32)
        k = _apply_rotary(k, cos, sin).astype(jnp.float32)
        v = v.astype(jnp.float32)
        chunks = []
        for start in range(0, p, _ATTN_QUERY_CHUNK):
            chunk_output = self.attention_performer.forward(
                query_states=q[:, start : start + _ATTN_QUERY_CHUNK],
                key_states=k,
                value_states=v,
                mode=common_types.MODE_TRAIN,
                causal=False,
                precision=self.precision,
                compute_dtype=jnp.float32,
            ).attention_outputs
            chunks.append(chunk_output)
        out = jnp.concatenate(chunks, axis=1).reshape(n, p, -1).astype(hidden_states.dtype)
        return checkpoint_name(self.wo(out), "attn_output")


class DeepseekV41VisionMLP(spx.Module):
    """Bias-free SwiGLU MLP: ``w1`` emits ``[gate | up]``, ``w2`` projects back."""

    def __init__(
        self,
        config: DeepseekV41Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        params = config.vision_params()
        kwargs = _linear_kwargs(config.initializer_range, dtype, param_dtype, precision, rngs, use_bias=False)
        self.w1 = ColumnParallelLinear(params["hidden_size"], 2 * params["intermediate_size"], **kwargs)
        self.w2 = RowParallelLinear(params["intermediate_size"], params["hidden_size"], **kwargs)

    def forward(self, hidden_states: Array) -> Array:
        """``w2(silu(gate) * up)`` with ``gate, up = chunk(w1(x), 2)``."""
        gate, up = jnp.split(self.w1(hidden_states), 2, axis=-1)
        return checkpoint_name(self.w2(jax.nn.silu(gate) * up), "mlp_output")


class DeepseekV41VisionBlock(spx.Module):
    """Pre-norm ViT block: ``x + attn(norm1(x))`` then ``x + mlp(norm2(x))`` (RMSNorm eps 1e-6, float32 weight)."""

    def __init__(
        self,
        config: DeepseekV41Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        params = config.vision_params()
        kwargs = dict(dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)
        norm_kwargs = dict(dim=params["hidden_size"], eps=params["rms_norm_eps"], dtype=jnp.float32)
        self.norm1 = RMSNorm(**norm_kwargs, param_dtype=jnp.float32, rngs=rngs)
        self.attn = DeepseekV41VisionAttention(config, **kwargs)
        self.norm2 = RMSNorm(**norm_kwargs, param_dtype=jnp.float32, rngs=rngs)
        self.mlp = DeepseekV41VisionMLP(config, **kwargs)

    def forward(self, hidden_states: Array, cos: Array, sin: Array) -> Array:
        """Apply attention and MLP residual branches to ``[N, P, hidden]``."""
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cos, sin)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class DeepseekV41VisionModel(spx.Module):
    """DeepSeek ViT: patch embedding, ``num_hidden_layers`` blocks, final RMSNorm (reference ``ViT``)."""

    def __init__(
        self,
        config: DeepseekV41Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        params = config.vision_params()
        kwargs = dict(dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)
        self.patch_size = params["patch_size"]
        self.rope_theta = params["rope_theta"]
        self.rope_dim = params["hidden_size"] // params["num_attention_heads"] // 2
        self.patch_embed = DeepseekV41VisionPatchEmbed(config, **kwargs)
        block_cls = auto_remat(
            DeepseekV41VisionBlock,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.blocks = nn.ModuleList([])
        for _ in range(params["num_hidden_layers"]):
            self.blocks.append(block_cls(config, **kwargs))
        self.norm = RMSNorm(
            params["hidden_size"],
            eps=params["rms_norm_eps"],
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=rngs,
        )

    def forward(self, patches: Array, grid: tuple[int, int]) -> Array:
        """Encode images that share one patch grid.

        Args:
            patches: ``[N, P, 3 * patch ** 2]`` (or ``[N, P, 3, patch, patch]``, or ``[P, ...]`` for one image) in
                row-major patch order, ``P == n_vit_h * n_vit_w``.
            grid: Static python ints ``(n_vit_h, n_vit_w)``.

        Returns:
            Patch features ``[N, P, hidden]`` (``[1, P, hidden]`` for an unbatched input).

        Raises:
            ValueError: If the patch count does not match the grid.
        """
        n_h, n_w = int(grid[0]), int(grid[1])
        if patches.ndim in (2, 4):
            patches = patches[None]
        patches = patches.reshape(patches.shape[0], patches.shape[1], -1)
        if patches.shape[1] != n_h * n_w:
            raise ValueError(f"{patches.shape[1]} patches do not fill a {n_h}x{n_w} grid.")
        cos, sin = vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
        cos, sin = jnp.asarray(cos), jnp.asarray(sin)
        hidden_states = self.patch_embed(patches)
        for block in self.blocks:
            hidden_states = block(hidden_states, cos, sin)
        return self.norm(hidden_states)


class DeepseekV41Aligner(spx.Module):
    """Fold ``r x r`` patch neighbourhoods and project to the text width: ``w2(gelu(w1(x)))`` (reference ``Aligner``)."""

    def __init__(
        self,
        config: DeepseekV41Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        params = config.vision_params()
        self.downsample_ratio = params["downsample_ratio"]
        kwargs = _linear_kwargs(config.initializer_range, dtype, param_dtype, precision, rngs, use_bias=True)
        self.w1 = ColumnParallelLinear(params["hidden_size"] * self.downsample_ratio**2, config.hidden_size, **kwargs)
        self.w2 = RowParallelLinear(config.hidden_size, config.hidden_size, **kwargs)

    def forward(self, features: Array, grid: tuple[int, int]) -> Array:
        """Downsample ViT features to LLM tokens.

        Args:
            features: ``[N, n_vit_h * n_vit_w, C]`` ViT output.
            grid: Static python ints ``(n_vit_h, n_vit_w)``.

        Returns:
            ``[N, n_llm_h * n_llm_w, hidden_size]`` in reading order (``F.unfold`` order, channel-major ``(c, kh, kw)``
            inside each fold, right/bottom zero padding).
        """
        n_h, n_w = int(grid[0]), int(grid[1])
        r = self.downsample_ratio
        n, _, c = features.shape
        x = features.reshape(n, n_h, n_w, c)
        x = jnp.pad(x, ((0, 0), (0, -n_h % r), (0, -n_w % r), (0, 0)))
        gh, gw = x.shape[1] // r, x.shape[2] // r
        x = x.reshape(n, gh, r, gw, r, c).transpose(0, 1, 3, 5, 2, 4).reshape(n, gh * gw, c * r * r)
        return self.w2(jax.nn.gelu(self.w1(x), approximate=False))


__all__ = [
    "TOKEN_TYPE_IMAGE",
    "TOKEN_TYPE_IMAGE_END",
    "TOKEN_TYPE_IMAGE_NEW_LINE",
    "TOKEN_TYPE_IMAGE_START",
    "TOKEN_TYPE_TEXT",
    "DeepseekV41Aligner",
    "DeepseekV41VisionAttention",
    "DeepseekV41VisionBlock",
    "DeepseekV41VisionMLP",
    "DeepseekV41VisionModel",
    "DeepseekV41VisionPatchEmbed",
    "llm_grid",
    "vision_cos_sin",
]
