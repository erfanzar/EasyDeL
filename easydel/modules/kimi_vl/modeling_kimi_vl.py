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

This module implements the Kimi-VL vision-language model, which combines:
- MoonViT: A vision transformer encoder for processing images
- Multi-modal projector: Projects vision features to language model space
- DeepSeek-V3: A Mixture of Experts language model backbone

The architecture follows the HuggingFace trust_remote_code structure
and parameter naming conventions:
- `vision_tower.*`: MoonViT vision encoder components
- `multi_modal_projector.*`: Vision-to-language projection layers
- `language_model.*`: DeepseekV3ForCausalLM backbone

Key features:
- 2D rotary position embeddings for vision transformers
- Learnable 2D position embeddings with interpolation
- Block-diagonal attention for multi-image processing
- Patch merging for efficient feature compression
- Seamless integration with DeepSeek-V3 MoE language model

References:
    - Kimi-VL: https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cached_property

import jax
import jax.numpy as jnp
from eformer import common_types
from eformer.common_types import Replicated
from flax import nnx as nn
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import VLMCausalLMOutput
from easydel.layers.attention import FlexibleAttentionModule
from easydel.layers.base_modules import BaseVisionLanguageModule
from easydel.layers.components.norms import LayerNorm

from ..deepseek_v3.modeling_deepseek import DeepseekV3ForCausalLM
from .kimi_vl_configuration import KimiVLConfig, MoonViTConfig


def _create_block_diagonal_bias(cu_seqlens: Array, seq_length: int, dtype: jnp.dtype) -> Array:
    """Create block-diagonal attention bias from cumulative sequence lengths.

    This function creates a block-diagonal attention mask where each block
    corresponds to one image/segment. Positions within the same segment
    can attend to each other (bias=0), while cross-segment attention
    is blocked (bias=-inf).

    Args:
        cu_seqlens (Array): Cumulative sequence lengths array of shape (num_segments + 1,).
            For example, [0, 196, 392] for two 14x14 images.
        seq_length (int): Total sequence length (sum of all segment lengths).
        dtype (jnp.dtype): Data type for the output bias tensor.

    Returns:
        Array: Attention bias of shape (1, seq_length, seq_length) with 0 for
            allowed positions and -inf for disallowed cross-segment positions.
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
    """Apply complex-valued rotary position embeddings to queries and keys.

    Uses complex number multiplication for efficient RoPE application.
    The input tensors are converted to complex form, multiplied with
    frequency components, and converted back to real representation.

    Args:
        xq (Array): Query tensor of shape (..., num_heads, head_dim).
        xk (Array): Key tensor of shape (..., num_heads, head_dim).
        freqs_cis (Array): Complex frequency tensor of shape (..., head_dim/2).
            Contains precomputed cos + i*sin values for each position.

    Returns:
        tuple[Array, Array]: Tuple of (rotated_queries, rotated_keys), each with
            the same shape as the input tensors.
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
    """Learnable 2D positional embeddings with resolution interpolation.

    This module provides learnable position embeddings that can be interpolated
    to different spatial resolutions. Uses cubic interpolation by default for
    smooth scaling to arbitrary image sizes.

    Attributes:
        height (int): Base height for learned position embeddings.
        width (int): Base width for learned position embeddings.
        dim (int): Embedding dimension for each position.
        interpolation_mode: Interpolation method for resizing (default: CUBIC).
    """

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
        """Initialize learnable 2D position embeddings.

        Args:
            height (int): Base height for the position embedding grid.
            width (int): Base width for the position embedding grid.
            dim (int): Embedding dimension for each spatial position.
            interpolation_mode (jax.image.ResizeMethod, optional): Method for
                interpolating to different resolutions. Defaults to CUBIC.
            dtype (jnp.dtype, optional): Data type for computation.
                Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters.
                Defaults to jnp.bfloat16.
            rngs (nn.Rngs): Random number generator state.
        """
        self.height = height
        self.width = width
        self.dim = dim
        self.interpolation_mode = interpolation_mode
        self.kernel = nn.Param(
            nn.initializers.normal()(rngs.params(), (dim, width, height), param_dtype),
        )
        self.dtype = dtype

    def craft_sharding(self, *, partition_manager=None, **_kwargs) -> dict[str, object]:
        """Return sharding specs for position embedding parameters."""
        return {"kernel": Replicated}

    def __call__(self, x: Array, grid_hws: Array) -> Array:
        """Add interpolated position embeddings to input features.

        For each image in the batch (specified by grid_hws), interpolates
        the learned position embeddings to match the image's patch grid
        size and adds them to the corresponding features.

        Args:
            x (Array): Input feature tensor of shape (total_patches, dim).
            grid_hws (Array): Array of (height, width) pairs for each image,
                shape (num_images, 2).

        Returns:
            Array: Features with position embeddings added, same shape as input.
        """
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
    """Patch embedding layer for MoonViT vision transformer.

    Converts image patches into embeddings using a convolution operation,
    then adds learnable 2D position embeddings. Supports variable resolution
    images through position embedding interpolation.

    Attributes:
        patch_size (int): Size of each image patch.
        in_dim (int): Number of input channels (typically 3 for RGB).
        out_dim (int): Output embedding dimension.
    """

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
        """Initialize MoonViT patch embedding layer.

        Args:
            out_dim (int): Output embedding dimension for each patch.
            in_dim (int, optional): Number of input channels. Defaults to 3.
            patch_size (int, optional): Size of each square patch. Defaults to 14.
            pos_emb_height (int, optional): Base height for position embeddings.
                Defaults to 64.
            pos_emb_width (int, optional): Base width for position embeddings.
                Defaults to 64.
            dtype (jnp.dtype, optional): Data type for computation.
                Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters.
                Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike | None, optional): Numerical precision.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
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
        """Embed image patches with position information.

        Applies convolution to extract patch features and adds interpolated
        2D position embeddings based on each image's grid dimensions.

        Args:
            pixel_values (Array): Input images of shape (num_patches, H, W, C)
                or (num_patches, C, H, W). Automatically handles NCHW to NHWC
                conversion.
            grid_hws (Array): Array of (height, width) pairs for each image's
                patch grid, shape (num_images, 2).

        Returns:
            Array: Embedded patches with position information added,
                shape (total_patches, out_dim).
        """
        if pixel_values.ndim == 4:
            # Convert from NCHW -> NHWC for JAX conv.
            pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))
        x = self.proj(pixel_values.astype(self.dtype))
        x = x.reshape(x.shape[0], -1)
        return self.pos_emb(x, grid_hws)


class Rope2DPosEmb(nn.Module):
    """2D rotary position embedding for vision transformers.

    Implements rotary position embeddings extended to 2D spatial grids
    using complex number representation. Supports variable resolution
    images by computing position embeddings on a fixed grid that can
    be sliced for different image sizes.

    The embedding uses separate frequency components for x and y positions,
    concatenated along the last dimension to form the full 2D RoPE.

    Attributes:
        dim (int): Embedding dimension (must be divisible by 4).
        max_height (int): Maximum supported grid height.
        max_width (int): Maximum supported grid width.
        theta_base (float): Base for frequency computation.
    """

    def __init__(self, dim: int, max_height: int, max_width: int, theta_base: float = 10000.0):
        """Initialize 2D rotary position embedding.

        Args:
            dim (int): Embedding dimension (must be divisible by 4 for x/y split).
            max_height (int): Maximum grid height to precompute embeddings for.
            max_width (int): Maximum grid width to precompute embeddings for.
            theta_base (float, optional): Base for computing frequency bands.
                Defaults to 10000.0.

        Raises:
            ValueError: If dim is not divisible by 4.
        """
        self.dim = dim
        if dim % 4 != 0:
            raise ValueError("dim must be divisible by 4")
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

    @cached_property
    def freqs_cis(self) -> Array:
        """Precomputed complex frequency tensor for the full grid.

        Computes e^(i * pos * freq) for all positions in the maximum grid,
        combining x and y position frequencies.

        Returns:
            Array: Complex frequency tensor of shape (max_height, max_width, dim/2).
        """
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
        """Get concatenated frequency tensors for multiple images.

        Slices the precomputed frequency tensor for each image's grid
        dimensions and concatenates them along the sequence dimension.

        Args:
            grid_hws (Array): Array of (height, width) pairs for each image,
                shape (num_images, 2).

        Returns:
            Array: Concatenated frequency tensor of shape (total_patches, dim/2).

        Raises:
            ValueError: If any grid dimension exceeds the maximum supported size.
        """
        shapes = [(int(h), int(w)) for h, w in grid_hws]
        if not all(1 <= h <= self.max_height and 1 <= w <= self.max_width for h, w in shapes):
            raise ValueError(f"grid_hws out of range: {shapes} vs {(self.max_height, self.max_width)}")
        pieces = [self.freqs_cis[:h, :w].reshape(-1, self.dim // 2) for h, w in shapes]
        return jnp.concatenate(pieces, axis=0)


class MLP2(nn.Module):
    """Two-layer feedforward network for MoonViT transformer blocks.

    Standard MLP with configurable activation function. Uses HuggingFace-compatible
    naming (`fc0`, `fc1`) for weight loading compatibility.

    Attributes:
        fc0: First linear layer (input -> hidden).
        fc1: Second linear layer (hidden -> output).
        activation: Activation function applied after first layer.
    """

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
        """Initialize two-layer MLP.

        Args:
            dims (tuple[int, int, int]): Tuple of (input_dim, hidden_dim, output_dim).
            activation (Callable): Activation function to apply after first layer.
            dtype (jnp.dtype, optional): Data type for computation.
                Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters.
                Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike | None, optional): Numerical precision.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
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
        """Apply two-layer MLP transformation.

        Args:
            x (Array): Input tensor of shape (..., input_dim).

        Returns:
            Array: Output tensor of shape (..., output_dim).
        """
        return self.fc1(self.activation(self.fc0(x)))


class MoonVitEncoderLayer(nn.Module):
    """Transformer encoder layer for MoonViT vision transformer.

    Implements a standard vision transformer encoder layer with:
    - Pre-normalization architecture (LayerNorm before attention and MLP)
    - Multi-head self-attention with 2D RoPE
    - Block-diagonal attention for multi-image processing
    - Two-layer MLP with configurable activation

    Attributes:
        num_heads (int): Number of attention heads.
        hidden_dim (int): Hidden dimension size.
        head_dim (int): Dimension per attention head.
    """

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
        """Initialize MoonViT encoder layer.

        Args:
            base_config (MoonViTConfig): Configuration for the vision transformer.
            num_heads (int): Number of attention heads.
            hidden_dim (int): Hidden dimension size.
            mlp_dim (int): MLP intermediate dimension.
            activation (Callable): Activation function for MLP.
            attn_bias (bool, optional): Whether to use bias in attention layers.
                Defaults to True.
            dtype (jnp.dtype, optional): Data type for computation.
                Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters.
                Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike | None, optional): Numerical precision.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.dtype = dtype

        self.norm0 = LayerNorm(
            hidden_dim,
            epsilon=1e-5,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.norm1 = LayerNorm(
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
        """Apply multi-head self-attention with 2D RoPE and block-diagonal masking.

        Args:
            x (Array): Input tensor of shape (seq_len, hidden_dim).
            cu_seqlens (Array): Cumulative sequence lengths for block-diagonal attention.
            rope_freqs_cis (Array): 2D rotary position embedding frequencies.

        Returns:
            Array: Attention output of shape (seq_len, hidden_dim).
        """
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
        """Forward pass through the encoder layer.

        Applies pre-norm self-attention and MLP with residual connections:
        x = x + attention(norm(x))
        x = x + mlp(norm(x))

        Args:
            hidden_states (Array): Input tensor of shape (seq_len, hidden_dim).
            cu_seqlens (Array): Cumulative sequence lengths for multi-image attention.
            rope_freqs_cis (Array): 2D rotary position embedding frequencies.

        Returns:
            Array: Output tensor of shape (seq_len, hidden_dim).
        """
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)
        hidden_states = residual + self._attention(hidden_states, cu_seqlens, rope_freqs_cis)

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class MoonVitEncoder(nn.Module):
    """Vision transformer encoder for MoonViT with packed sequence support.

    Processes packed sequences of image patches from multiple images
    using block-diagonal attention. Uses 2D rotary position embeddings
    for spatial awareness.

    Attributes:
        rope_2d (Rope2DPosEmb): 2D rotary position embedding module.
        blocks (list): Stack of transformer encoder layers.
        final_layernorm: Final layer normalization.
    """

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
        """Initialize MoonViT encoder.

        Args:
            base_config (MoonViTConfig): Configuration for the vision transformer.
            hidden_dim (int): Hidden dimension size.
            num_layers (int): Number of transformer layers.
            num_heads (int): Number of attention heads.
            mlp_dim (int): MLP intermediate dimension.
            dtype (jnp.dtype, optional): Data type for computation.
                Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters.
                Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike | None, optional): Numerical precision.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.rope_2d = Rope2DPosEmb(hidden_dim // num_heads, 512, 512)

        def activation(x):
            return jax.nn.gelu(x, approximate=True)

        self.blocks = nn.List(
            [
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
        )
        self.final_layernorm = LayerNorm(
            hidden_dim,
            epsilon=1e-5,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, hidden_states: Array, grid_hws: Array) -> Array:
        """Forward pass through the MoonViT encoder.

        Processes packed image patches through transformer layers with
        block-diagonal attention and 2D rotary position embeddings.

        Args:
            hidden_states (Array): Packed patch embeddings of shape
                (total_patches, hidden_dim).
            grid_hws (Array): Array of (height, width) pairs for each image,
                shape (num_images, 2).

        Returns:
            Array: Encoded features of shape (total_patches, hidden_dim).
        """
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
    """Merge adjacent patches into spatial groups for feature compression.

    Groups patches according to the merge kernel size, reducing the number
    of tokens while preserving spatial locality. This is used to reduce
    the sequence length before projecting to the language model.

    For example, with merge_kernel_size=(2, 2), a 14x14 grid becomes
    7x7 groups, each containing 4 patches.

    Args:
        hidden_states (Array): Packed patch features of shape
            (total_patches, hidden_dim).
        grid_hws (Array): Array of (height, width) pairs for each image,
            shape (num_images, 2).
        merge_kernel_size (tuple[int, int]): Kernel size for merging patches
            as (kernel_h, kernel_w).

    Returns:
        list[Array]: List of merged features for each image, where each
            element has shape (num_groups, kernel_h * kernel_w, hidden_dim).
    """
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
    """MoonViT vision tower for Kimi-VL multimodal model.

    Complete vision encoder that processes images through:
    1. Patch embedding with learnable 2D position embeddings
    2. Vision transformer encoder with 2D RoPE
    3. Patch merging for sequence compression

    Matches HuggingFace `MoonVitPretrainedModel` naming for weight compatibility.

    Attributes:
        merge_kernel_size (tuple): Kernel size for patch merging.
        patch_size (int): Size of image patches.
        patch_embed: Patch embedding layer.
        encoder: Vision transformer encoder.
    """

    def __init__(
        self,
        config: MoonViTConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize MoonViT vision tower.

        Args:
            config (MoonViTConfig): Configuration for the vision model.
            dtype (jnp.dtype, optional): Data type for computation.
                Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters.
                Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike | None, optional): Numerical precision.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
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
        """Process images through the vision tower.

        Embeds image patches, processes them through the transformer encoder,
        and merges adjacent patches for sequence compression.

        Args:
            pixel_values (Array): Input images of shape (num_patches, C, H, W)
                or (num_patches, H, W, C).
            grid_hws (Array): Array of (height, width) pairs for each image's
                patch grid, shape (num_images, 2).

        Returns:
            list[Array]: List of merged features for each image, where each
                element has shape (num_groups, kernel_h * kernel_w, hidden_dim).
        """
        hidden_states = self.patch_embed(pixel_values, grid_hws)
        hidden_states = self.encoder(hidden_states, grid_hws)
        return patch_merger(hidden_states, grid_hws, merge_kernel_size=self.merge_kernel_size)


class KimiVLMultiModalProjector(nn.Module):
    """Multi-modal projector for bridging vision and language models.

    Projects merged patch features from MoonViT into the DeepSeek-V3
    language model's hidden dimension. Uses a two-layer MLP with
    LayerNorm and GELU activation.

    Architecture:
        1. Pre-normalization on input features
        2. Flatten merged patches
        3. Linear projection with GELU activation
        4. Final linear projection to language model dimension

    Attributes:
        hidden_size (int): Flattened feature dimension after patch merging.
        pre_norm: Layer normalization before projection.
        linear_1: First linear layer with GELU activation.
        linear_2: Second linear layer projecting to LM hidden size.
    """

    def __init__(
        self,
        config: KimiVLConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize multi-modal projector.

        Args:
            config (KimiVLConfig): Configuration for the VLM model.
            dtype (jnp.dtype, optional): Data type for computation.
                Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters.
                Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike | None, optional): Numerical precision.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        merge_kernel = tuple(config.vision_config.merge_kernel_size)
        hidden_size = config.vision_config.hidden_size * merge_kernel[0] * merge_kernel[1]

        self.hidden_size = hidden_size
        self.pre_norm = LayerNorm(
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
        """Project merged image features to language model dimension.

        Concatenates features from all images, normalizes, flattens the
        merged patch groups, and projects to the language model hidden size.

        Args:
            image_features (list[Array]): List of merged features from
                MoonViT, each of shape (num_groups, kernel_h * kernel_w, hidden_dim).

        Returns:
            Array: Projected features of shape (total_tokens, lm_hidden_size).
        """
        image_features = jnp.concatenate(image_features, axis=0)
        hidden_states = self.pre_norm(image_features).reshape(-1, self.hidden_size)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = jax.nn.gelu(hidden_states, approximate=False)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


@register_module(TaskType.IMAGE_TEXT_TO_TEXT, config=KimiVLConfig, model_type="kimi_vl")
class KimiVLForConditionalGeneration(BaseVisionLanguageModule[DeepseekV3ForCausalLM, KimiVLConfig]):
    """Kimi-VL vision-language model for conditional text generation.

    Combines MoonViT vision encoder with DeepSeek-V3 MoE language model
    for multimodal understanding and generation. Processes images through
    the vision tower and projects them into the language model's embedding
    space.

    Architecture:
        1. Vision tower (MoonViT): Encodes images into patch features
        2. Multi-modal projector: Projects vision features to LM dimension
        3. Language model (DeepSeek-V3): Generates text conditioned on
           combined image and text embeddings

    This model supports:
        - Single and multi-image inputs
        - Variable resolution images
        - Interleaved image-text generation
        - Efficient MoE-based language modeling

    Attributes:
        vision_tower (MoonVitPretrainedModel): Vision encoder component.
        multi_modal_projector (KimiVLMultiModalProjector): Vision-to-LM projector.
        language_model (DeepseekV3ForCausalLM): Language model backbone.
    """

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
        """Initialize Kimi-VL model for conditional generation.

        Args:
            config (KimiVLConfig): Configuration for the VLM model.
            dtype (jnp.dtype, optional): Data type for computation.
                Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters.
                Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike | None, optional): Numerical precision.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
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
        """Merge text embeddings with image features at placeholder positions.

        Replaces the placeholder token embeddings with the corresponding
        image feature embeddings.

        Args:
            inputs_embeds (Array): Text embeddings of shape (batch, seq_len, hidden_dim).
            input_ids (Array): Token IDs of shape (batch, seq_len).
            image_features (Array): Projected image features.

        Returns:
            Array: Combined embeddings with image features replacing placeholders.
        """
        placeholder = int(self.config.media_placeholder_token_id)
        multimodal_embeddings = image_features.reshape(-1, image_features.shape[-1])
        return BaseVisionLanguageModule.merge_multimodal_embeddings(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            placeholder_token_id=placeholder,
        )

    def _extract_image_features(self, pixel_values: Array, image_grid_hws: Array) -> Array:
        """Extract and project image features from pixel values.

        Processes images through the vision tower and multi-modal projector
        to produce features compatible with the language model.

        Args:
            pixel_values (Array): Input images.
            image_grid_hws (Array): Grid dimensions for each image.

        Returns:
            Array: Projected image features ready for merging with text.
        """
        image_features = self.vision_tower(pixel_values, image_grid_hws)
        return self.multi_modal_projector(image_features)

    def get_image_features(
        self,
        pixel_values: Float[Array, "num_patches channels patch_h patch_w"],
        image_grid_hws: Int[Array, "num_images 2"] | None = None,
        **kwargs,
    ) -> Float[Array, "num_patches hidden_dim"]:
        """Get projected image features from pixel values.

        Public interface for extracting image features. Useful for pre-computing
        image features for efficient multi-turn conversation.

        Args:
            pixel_values (Array): Input images of shape (num_patches, C, H, W).
            image_grid_hws (Array | None): Grid dimensions for each image,
                shape (num_images, 2). Required when pixel_values is provided.
            **kwargs: Additional arguments (ignored for compatibility).

        Returns:
            Array: Projected image features of shape (num_patches, hidden_dim).

        Raises:
            ValueError: If image_grid_hws is None.
        """
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
        """Compute combined text and image embeddings.

        Embeds input tokens and optionally merges with image features at
        placeholder positions. Handles placeholder token ID replacement
        for out-of-vocabulary image tokens.

        Args:
            input_ids (Array): Token IDs of shape (batch, seq_len).
            image_features (Array | None, optional): Pre-computed image features.
                Defaults to None.
            pixel_values (Array | None, optional): Raw pixel values for images.
                Used if image_features is None. Defaults to None.
            image_grid_hws (Array | None, optional): Grid dimensions for images.
                Required if pixel_values is provided. Defaults to None.
            **kwargs: Additional arguments (ignored for compatibility).

        Returns:
            Array: Combined embeddings of shape (batch, seq_len, hidden_dim).

        Raises:
            ValueError: If input_ids is None or if pixel_values is provided
                without image_grid_hws.
        """
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
        """Forward pass for multimodal generation.

        Processes both image and text inputs through their respective encoders,
        merges the embeddings, and generates output through the language model.

        Args:
            input_ids (Array | None, optional): Token IDs of shape (batch, seq_len).
                Must be provided if inputs_embeds is None.
            inputs_embeds (Array | None, optional): Pre-computed embeddings.
                Defaults to None.
            attention_mask (Array | None, optional): Attention mask for padding.
                Defaults to None.
            mask_info (object | None, optional): Advanced mask information.
                Defaults to None.
            position_ids (Array | None, optional): Position indices for tokens.
                Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode (train/decode).
                Defaults to None.
            past_key_values (object | None, optional): Cache for generation.
                Defaults to None.
            cache_metadata (object | None, optional): Cache metadata.
                Defaults to None.
            apply_lm_head (bool, optional): Whether to apply LM head projection.
                Defaults to True.
            output_attentions (bool | None, optional): Return attention weights.
                Defaults to None.
            output_hidden_states (bool | None, optional): Return hidden states.
                Defaults to None.
            output_router_logits (bool | None, optional): Return MoE router logits.
                Defaults to None.
            pixel_values (Array | None, optional): Input images.
                Defaults to None.
            image_grid_hws (Array | None, optional): Grid dimensions for images.
                Defaults to None.

        Returns:
            VLMCausalLMOutput: Contains logits, hidden states, attentions,
                image features, router logits, and auxiliary loss.

        Raises:
            ValueError: If both or neither of input_ids and inputs_embeds are provided,
                or if pixel_values is provided without input_ids or image_grid_hws.
        """
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
        """Initialize KV cache for generation.

        Delegates to the language model's cache initialization.

        Args:
            batch_size: Batch size for the cache.
            max_length: Maximum sequence length.
            starts: Starting positions (optional).
            shardings: Sharding specifications (optional).
            pad_token_id: Padding token ID (optional).

        Returns:
            Initialized cache for autoregressive generation.
        """
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
        """Prepare inputs for autoregressive generation.

        Extends the language model's input preparation with image-related
        inputs for multimodal generation.

        Args:
            input_ids (Array): Input token IDs.
            max_length (int): Maximum generation length.
            pad_token_id (int): Padding token ID.
            starts (int | None, optional): Starting positions. Defaults to None.
            pixel_values (Array | None, optional): Input images. Defaults to None.
            attention_mask (Array | None, optional): Attention mask. Defaults to None.
            image_grid_hws (Array | None, optional): Image grid dimensions.
                Defaults to None.

        Returns:
            dict: Model inputs including text and image components.
        """
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
        """Update inputs for the next generation step.

        Updates model kwargs and removes image inputs after the first
        forward pass (images are only needed once).

        Args:
            model_outputs: Outputs from the previous generation step.
            model_kwargs: Current model keyword arguments.

        Returns:
            dict: Updated model kwargs for the next step.
        """
        model_kwargs = self.language_model.update_inputs_for_generation(model_outputs, model_kwargs)
        model_kwargs.pop("pixel_values", None)
        model_kwargs.pop("image_grid_hws", None)
        return model_kwargs

    def get_encoder(self):
        """Get the vision encoder (vision tower).

        Returns:
            MoonVitPretrainedModel: The vision tower component.
        """
        return self.vision_tower

    def get_decoder(self):
        """Get the language model decoder.

        Returns:
            The decoder component of the language model.
        """
        return self.language_model.get_decoder()

    def get_lm_head(self):
        """Get the language model head.

        Returns:
            The language model output projection.
        """
        return self.language_model.get_lm_head()

    def get_embedding(self):
        """Get the token embedding layer.

        Returns:
            Embed: The language model's token embeddings.
        """
        return self.language_model.get_embedding()

    def get_vision_tower(self) -> nn.Module:
        """Get the vision tower module.

        Returns:
            MoonVitPretrainedModel: The MoonViT vision encoder.
        """
        return self.vision_tower

    def get_projector(self) -> nn.Module:
        """Get the multi-modal projector.

        Returns:
            KimiVLMultiModalProjector: The vision-to-language projector.
        """
        return self.multi_modal_projector

    def get_language_model(self) -> nn.Module:
        """Get the language model backbone.

        Returns:
            DeepseekV3ForCausalLM: The DeepSeek-V3 language model.
        """
        return self.language_model


__all__ = [
    "KimiVLForConditionalGeneration",
    "KimiVLMultiModalProjector",
    "MoonVitPretrainedModel",
]
