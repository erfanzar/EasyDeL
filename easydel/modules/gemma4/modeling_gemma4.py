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

"""Gemma4 model implementation for EasyDeL.

This module provides the Gemma4 multimodal model architecture including text-only
and vision-language variants. Gemma4 extends Gemma3 with several architectural
innovations designed for improved efficiency and capacity:

Key Architectural Features:
    - Mixed Attention Patterns: Alternates between local sliding window attention
      and global full attention with a configurable 5:1 ratio (default). Sliding
      layers use standard RoPE (theta=10k) while global layers use proportional
      RoPE (theta=1M) with partial_rotary_factor=0.25.
    - Per-Layer Input Embeddings: Each decoder layer receives an additional
      residual signal computed from a separate per-layer embedding table combined
      with a learned projection of the main embeddings. This provides each layer
      with a unique view of the input tokens.
    - KV Sharing: Later layers in the network can reuse key-value projections
      from earlier layers of the same attention type (sliding/global), reducing
      memory footprint and parameter count.
    - Mixture of Experts (MoE): Optional sparse MoE blocks run in parallel with
      the dense MLP at each layer. A learned router selects the top-k experts per
      token, with per-expert scaling and normalized routing weights.
    - Key-Equals-Value (k_eq_v): For global attention layers, the key projection
      can be reused as the value projection (no separate v_proj), reducing
      parameter count while preserving representation capacity.
    - V-Normalization: Value states are normalized using RMSNorm without a
      learnable scale parameter, improving training stability.
    - Per-Layer Head Dimensions: Global attention layers can use a different head
      dimension (``global_head_dim``) and number of KV heads
      (``num_global_key_value_heads``) from sliding window layers.
    - Double-Wide MLP: Layers that share KV projections can optionally use a 2x
      wider MLP to compensate for the reduced attention capacity.
    - Layer Scalar: Each decoder layer has a learnable scalar weight (initialized
      to 1.0) that scales the layer's output, enabling fine-grained capacity
      control.
    - Multimodal Embedder: Vision features are projected into the language model's
      embedding space via RMSNorm (without scale) followed by a linear projection,
      replacing Gemma3's average-pooling projector.

Model Variants:
    - ``Gemma4TextModel``: Text-only decoder transformer for language modeling.
    - ``Gemma4ForCausalLM``: Text model with language modeling head for generation.
    - ``Gemma4Model``: Multimodal model with vision tower and language model.
    - ``Gemma4ForConditionalGeneration``: Full VLM with LM head for
      vision-language generation tasks.

References:
    - HuggingFace: https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma4
"""

import typing
from functools import cached_property, partial

import jax
import jax.numpy as jnp
import spectrax as spx
from ejkernel.types import MaskInfo
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int
from spectrax import apply_logical_sharding, common_types, nn

from easydel.caching import (
    HybridCache,
    OperationsMetadata,
    RaggedPagesCache,
    RaggedPagesCacheView,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.infra.base_module import EasyDeLBaseModule, EasyDeLLayerStackMixin
from easydel.infra.factory import TaskType, register_module, registry
from easydel.infra.modeling_outputs import (
    AttentionLayerOutput,
    BaseModelOutput,
    CausalLMOutput,
    DecoderLayerOutput,
    EmbeddingInfo,
    EncoderLayerOutput,
    VLMCausalLMOutput,
)
from easydel.infra.utils import ACT2FN, ArrayParam, auto_remat, block_wise_ffn
from easydel.layers import (
    BaseMoeModule,
    ColumnParallelLinear,
    ColumnParallelMoELinear,
    Embed,
    MoeLoadBalancingStrategy,
    MoeRoutingStrategy,
    ParallelLinear,
    RowParallelLinear,
    RowParallelMoELinear,
)
from easydel.layers.attention import FlexibleAttentionModule, UnifiedAttention
from easydel.layers.norms._norms import lowfloats
from easydel.modules._base import BaseCausalLMModule, BaseVisionLanguageModule
from easydel.modules.auto.auto_modeling import AutoEasyDeLVisionModel

from .gemma4_configuration import Gemma4Config, Gemma4TextConfig, Gemma4VisionConfig


def _has_registered_gemma4_vision_backend(config: Gemma4VisionConfig | None) -> bool:
    """Return True when the configured Gemma4 vision encoder is registered.

    Args:
        config (Gemma4VisionConfig | None): Vision config or ``None``.

    Returns:
        bool: ``True`` if a ``BASE_VISION`` module is registered for
        ``config.model_type``; ``False`` if ``config`` is ``None`` or no
        registration exists.
    """
    if config is None:
        return False
    try:
        registry.get_module_registration(TaskType.BASE_VISION, config.model_type)
        return True
    except KeyError:
        return False


def _gemma4_vision_pixel_values_to_nhwc(pixel_values: Array) -> Array:
    """Convert image inputs to NHWC, accepting both NCHW and NHWC layouts.

    Args:
        pixel_values (Array): A rank-4 tensor of either NCHW or NHWC layout.

    Returns:
        Array: NHWC-laid out tensor.

    Raises:
        ValueError: If ``pixel_values`` is not rank-4.
    """
    if pixel_values.ndim != 4:
        raise ValueError("Gemma4 vision inputs must be rank-4 tensors.")

    if pixel_values.shape[-1] in (1, 3, 4):
        return pixel_values
    if pixel_values.shape[1] in (1, 3, 4):
        return jnp.transpose(pixel_values, (0, 2, 3, 1))
    return pixel_values


class Gemma4RMSNorm(spx.Module):
    """Root Mean Square Layer Normalization for Gemma4 models.

    Implements Hugging Face Gemma4 RMS normalization with ``kernel * norm(x)``
    scaling and Float8/lowfloat support. When ``with_scale=False``, the
    learnable scale parameter is omitted entirely, producing a pure
    normalization without any learned affine transform. This scale-free variant
    is used for value normalization in Gemma4's attention layers.

    The normalization is always computed in float32 for numerical stability,
    regardless of the input dtype.

    Args:
        config: Model configuration providing ``rms_norm_eps`` and ``hidden_size``.
            Can be ``None`` if both ``dim`` and ``epsilon`` are provided explicitly.
        param_dtype: Data type for the learnable scale parameter. Defaults to float32.
        dim: Dimension of the normalization kernel. If ``None``, uses
            ``config.hidden_size``. Must be provided if ``config`` is ``None``.
        epsilon: Small constant added to the denominator for numerical stability.
            If ``None``, uses ``config.rms_norm_eps``. Defaults to 1e-6 when
            ``config`` is also ``None``.
        with_scale: Whether to include a learnable scale parameter. When ``True``
            (default), the output is ``kernel * norm(x)``. When ``False``, the
            output is simply ``norm(x)`` with no learned parameters.
    """

    kernel_init = staticmethod(jax.nn.initializers.ones)

    def __init__(
        self,
        config: Gemma4TextConfig | Gemma4VisionConfig | None = None,
        param_dtype: jnp.dtype = jnp.float32,
        dim: int | None = None,
        epsilon: float | None = None,
        with_scale: bool = True,
    ):
        self.epsilon = (config.rms_norm_eps if config is not None else 1e-6) if epsilon is None else epsilon
        self.param_dtype = param_dtype
        self.with_scale = with_scale
        if with_scale:
            dim = (config.hidden_size if config is not None else dim) if dim is None else dim
            self.weight = ArrayParam.bound(
                shape=(dim,),
                dtype=param_dtype,
                init_method="ones",
                key=None,
            )

    def _norm(self, x: jax.Array) -> jax.Array:
        """Apply RMS normalization without any learnable parameters.

        Computes ``x / sqrt(mean(x^2) + epsilon)`` element-wise along the last
        dimension.

        Args:
            x: Input tensor of arbitrary shape. Normalization is applied along
                the last axis.

        Returns:
            Normalized tensor with the same shape as ``x``.
        """
        return x * (1 / jnp.sqrt(jnp.power(x, 2).mean(-1, keepdims=True) + self.epsilon))

    def forward(self, hidden_states: Array) -> jax.Array:
        """Apply RMS normalization with optional learnable scale.

        The input is upcast to float32 for the normalization computation, then
        cast back to the input dtype. When ``with_scale=True``, the result is
        scaled by ``kernel`` following Hugging Face Gemma4. If the output dtype
        is a lowfloat (e.g., Float8), it is automatically promoted to bfloat16
        for downstream compatibility.

        Args:
            hidden_states: Input tensor of shape ``(..., dim)`` where ``dim``
                matches the kernel dimension.

        Returns:
            Normalized (and optionally scaled) tensor with the same shape as
            the input.
        """
        variance = self._norm(hidden_states.astype(jnp.float32))
        if self.with_scale:
            out = self.weight.value.astype(jnp.float32) * variance
        else:
            out = variance
        target_dtype = hidden_states.dtype
        if target_dtype in lowfloats:
            target_dtype = jnp.bfloat16
        return out.astype(target_dtype)


def _gemma4_vision_patchify(pixel_values: Array, patch_size: int) -> tuple[Array, Array]:
    """Convert raw images into flat patches and 2-D patch position ids.

    Args:
        pixel_values (Array): NHWC pixel tensor of shape
            ``(batch, height, width, channels)``.
        patch_size (int): Side length of each square patch.

    Returns:
        tuple[Array, Array]: ``(patches, pixel_position_ids)`` of shapes
        ``(batch, num_patches, patch_size**2 * channels)`` and
        ``(batch, num_patches, 2)``.

    Raises:
        ValueError: If the image dimensions are not divisible by
            ``patch_size``.
    """
    pixel_values = _gemma4_vision_pixel_values_to_nhwc(pixel_values)
    batch_size, height, width, channels = pixel_values.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"Gemma4 vision inputs must be divisible by the patch size ({patch_size}), got {(height, width)}."
        )

    grid_height = height // patch_size
    grid_width = width // patch_size
    patches = pixel_values.reshape(
        batch_size,
        grid_height,
        patch_size,
        grid_width,
        patch_size,
        channels,
    )
    patches = jnp.transpose(patches, (0, 1, 3, 2, 4, 5))
    patches = patches.reshape(batch_size, grid_height * grid_width, patch_size * patch_size * channels)

    grid_x, grid_y = jnp.meshgrid(
        jnp.arange(grid_width, dtype=jnp.int32),
        jnp.arange(grid_height, dtype=jnp.int32),
        indexing="xy",
    )
    pixel_position_ids = jnp.stack((grid_x, grid_y), axis=-1).reshape(1, grid_height * grid_width, 2)
    pixel_position_ids = jnp.broadcast_to(pixel_position_ids, (batch_size, grid_height * grid_width, 2))
    return patches, pixel_position_ids


def _gemma4_vision_prepare_inputs(
    pixel_values: Array,
    patch_size: int,
    pixel_position_ids: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Normalize Gemma4 vision inputs to flat patches plus patch positions.

    Accepts either rank-4 pixel inputs (which are patchified internally) or
    rank-3 pre-patchified inputs. When ``pixel_position_ids`` is ``None`` for
    rank-3 inputs, square positional ids are auto-generated provided the patch
    count is a perfect square.

    Args:
        pixel_values (Array): Rank-4 pixel tensor or rank-3 patch tensor.
        patch_size (int): Patch side length, used only for rank-4 inputs.
        pixel_position_ids (Array | None, optional): Optional explicit
            ``(x, y)`` patch ids. Defaults to None.

    Returns:
        tuple[Array, Array, Array]: ``(patches, pixel_position_ids,
        padding_positions)``. ``padding_positions`` is True wherever
        ``pixel_position_ids == -1``.

    Raises:
        ValueError: If the input rank is unsupported, or if rank-3 inputs lack
            position ids and do not have a square patch count.
    """
    if pixel_values.ndim == 4:
        patches, inferred_position_ids = _gemma4_vision_patchify(pixel_values, patch_size)
        if pixel_position_ids is None:
            pixel_position_ids = inferred_position_ids
        else:
            pixel_position_ids = pixel_position_ids.astype(jnp.int32)
        padding_positions = jnp.zeros(patches.shape[:2], dtype=jnp.bool_)
        return patches, pixel_position_ids, padding_positions

    if pixel_values.ndim != 3:
        raise ValueError(
            "Gemma4 vision inputs must be either raw images `[batch, channels, height, width]` "
            "or flat patches `[batch, num_patches, patch_dim]`."
        )

    patches = pixel_values
    if pixel_position_ids is None:
        num_patches = patches.shape[1]
        grid_size = int(num_patches**0.5)
        if grid_size * grid_size != num_patches:
            raise ValueError(
                "Flat patch inputs require `pixel_position_ids` unless the patch count forms a perfect square."
            )
        grid_x, grid_y = jnp.meshgrid(
            jnp.arange(grid_size, dtype=jnp.int32),
            jnp.arange(grid_size, dtype=jnp.int32),
            indexing="xy",
        )
        pixel_position_ids = jnp.stack((grid_x, grid_y), axis=-1).reshape(1, num_patches, 2)
        pixel_position_ids = jnp.broadcast_to(pixel_position_ids, (patches.shape[0], num_patches, 2))
    else:
        pixel_position_ids = pixel_position_ids.astype(jnp.int32)

    padding_positions = jnp.all(pixel_position_ids == -1, axis=-1)
    return patches, pixel_position_ids, padding_positions


class Gemma4VisionPatchEmbedder(spx.Module):
    """Project flat image patches and add learned 2-D position embeddings.

    Implements the HF-compatible Gemma4 vision stem: pixel patches are linearly
    projected to ``hidden_size`` and added to learned 2-D positional embeddings
    indexed by per-patch ``(x, y)`` ids. Padded patches receive zero positional
    embeddings.

    Attributes:
        input_proj (ColumnParallelLinear): Linear projection from flattened
            patch pixels (``3 * patch_size**2``) to ``hidden_size``.
        position_embedding_table (ArrayParam): Learned table of shape
            ``(2, position_embedding_size, hidden_size)`` storing per-axis
            (x then y) positional embeddings.
    """

    def __init__(
        self,
        config: Gemma4VisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the Gemma4 vision patch embedder.

        Args:
            config (Gemma4VisionConfig): Vision encoder configuration providing
                ``patch_size``, ``hidden_size``, ``position_embedding_size``,
                and ``initializer_range``.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to
                jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters.
                Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for
                matmul. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        patch_dim = 3 * config.patch_size * config.patch_size
        kernel_init = jax.nn.initializers.normal(config.initializer_range)
        self.input_proj = ColumnParallelLinear(
            patch_dim,
            config.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.position_embedding_table = ArrayParam.bound(
            shape=(2, config.position_embedding_size, config.hidden_size),
            dtype=param_dtype,
            init_method="ones",
            key=None,
        )

    def _position_embeddings(self, pixel_position_ids: Array, padding_positions: Array) -> Array:
        """Gather per-axis position embeddings and zero them on padded patches.

        Args:
            pixel_position_ids (Array): Integer ids of shape
                ``(batch, num_patches, 2)`` for the ``(x, y)`` patch positions.
            padding_positions (Array): Boolean mask of shape
                ``(batch, num_patches)`` indicating padded (True) entries.

        Returns:
            Array: Positional embeddings of shape
            ``(batch, num_patches, hidden_size)`` with zeros on padded slots.

        Raises:
            ValueError: If ``pixel_position_ids`` does not have a trailing
                dimension of size 2.
        """
        clamped_positions = jnp.clip(
            pixel_position_ids.astype(jnp.int32),
            min=0,
            max=self.config.position_embedding_size - 1,
        )
        if clamped_positions.shape[-1] != 2:
            raise ValueError(
                f"Gemma4 vision position ids must have shape `[batch, num_patches, 2]`, got {clamped_positions.shape}."
            )
        position_table = self.position_embedding_table.value
        x_embeddings = jnp.take(position_table[0], clamped_positions[..., 0], axis=0)
        y_embeddings = jnp.take(position_table[1], clamped_positions[..., 1], axis=0)
        position_embeddings = x_embeddings + y_embeddings
        return jnp.where(padding_positions[..., None], 0.0, position_embeddings)

    def forward(self, pixel_values: Array, pixel_position_ids: Array, padding_positions: Array) -> Array:
        """Embed already patchified pixel inputs using the HF-compatible stem.

        The pixel values are first rescaled from ``[0, 1]`` to ``[-1, 1]``,
        projected through ``input_proj``, and then added to the per-axis 2-D
        positional embeddings.

        Args:
            pixel_values (Array): Patchified pixel inputs of shape
                ``(batch, num_patches, patch_dim)``.
            pixel_position_ids (Array): Integer ``(x, y)`` ids of shape
                ``(batch, num_patches, 2)``.
            padding_positions (Array): Boolean padding mask of shape
                ``(batch, num_patches)``.

        Returns:
            Array: Embeddings of shape ``(batch, num_patches, hidden_size)``
            named ``"vision_embeddings"`` for rematerialization.
        """
        pixel_values = 2.0 * (pixel_values.astype(jnp.float32) - 0.5)
        hidden_states = self.input_proj(pixel_values.astype(self.input_proj.weight.dtype))
        position_embeddings = self._position_embeddings(pixel_position_ids, padding_positions)
        return checkpoint_name(hidden_states + position_embeddings.astype(hidden_states.dtype), "vision_embeddings")


class Gemma4VisionClippableLinear(spx.Module):
    """Wrapper that preserves the HF ``*.linear.weight`` parameter layout.

    Wraps either a :class:`ColumnParallelLinear` or :class:`RowParallelLinear`
    inside a ``self.linear`` attribute so that checkpoint parameter paths match
    Hugging Face's Gemma4 layout (``*.linear.weight``). The "clippable" name
    derives from the optional weight-clipping behaviour controlled by
    ``Gemma4VisionConfig.use_clipped_linears``.
    """

    def __init__(
        self,
        config: Gemma4VisionConfig,
        in_features: int,
        out_features: int,
        *,
        parallel_mode: str,
        use_bias: bool,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        rngs: spx.Rngs,
    ):
        """Initialize the clippable linear wrapper.

        Args:
            config (Gemma4VisionConfig): Vision config providing
                ``initializer_range``.
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            parallel_mode (str): Either ``"row"`` for row-parallel or any other
                value for column-parallel.
            use_bias (bool): Whether to add a bias term.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to
                jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters.
                Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision.
                Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        kernel_init = jax.nn.initializers.normal(config.initializer_range)
        linear_cls = RowParallelLinear if parallel_mode == "row" else ColumnParallelLinear
        self.linear = linear_cls(
            in_features,
            out_features,
            use_bias=use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            rngs=rngs,
        )

    def forward(self, hidden_states: Array) -> Array:
        """Apply the wrapped linear projection.

        Args:
            hidden_states (Array): Input tensor of shape ``(..., in_features)``.

        Returns:
            Array: Output tensor of shape ``(..., out_features)``.
        """
        return self.linear(hidden_states)


class Gemma4VisionRotaryEmbedding(spx.Module):
    """2-D rotary embedding shared across all Gemma4 vision encoder layers.

    Splits the per-head feature dimension in two halves and applies independent
    NeoX-style RoPE rotations driven by the patch ``x`` and ``y`` coordinates.
    The remaining channels are passed through unchanged.

    Attributes:
        head_dim (int): Per-head feature dimension.
        base (float): RoPE base frequency (typically very small for spatial
            positions, e.g. 100.0).
        rotary_dim_per_axis (int): Number of channels rotated per spatial axis
            (``2 * (head_dim // 4)``).
        frequencies (Array): Precomputed RoPE frequency tensor.
    """

    def __init__(self, config: Gemma4VisionConfig, dtype: jnp.dtype = jnp.bfloat16):
        """Initialize the 2-D rotary embedding.

        Args:
            config (Gemma4VisionConfig): Vision config providing ``head_dim``,
                ``rope_parameters``, and ``max_position_embeddings``.
            dtype (jnp.dtype, optional): Data type of the precomputed
                frequencies. Defaults to jnp.bfloat16.

        Raises:
            ValueError: If ``head_dim < 4`` (no channels left to rotate).
        """
        from easydel.layers import get_frequencies

        self.head_dim = config.head_dim
        self.base = config.rope_parameters.get("rope_theta", 100.0)
        self.rotary_dim_per_axis = 2 * (self.head_dim // 4)
        if self.rotary_dim_per_axis <= 0:
            raise ValueError(f"Gemma4 vision head_dim must be at least 4, got {self.head_dim}.")
        with jax.ensure_compile_time_eval():
            self.frequencies = get_frequencies(
                head_size=self.rotary_dim_per_axis,
                rotary_dim=self.rotary_dim_per_axis,
                max_position=config.max_position_embeddings,
                base=self.base,
                rope_scaling=None,
            ).astype(dtype)

    def _apply_axis(self, query: Array, key: Array, positions: Array) -> tuple[Array, Array]:
        """Apply NeoX-style RoPE along a single spatial axis.

        Args:
            query (Array): Query slice for a single axis.
            key (Array): Key slice for a single axis.
            positions (Array): Integer positions along the chosen axis.

        Returns:
            tuple[Array, Array]: Rotated ``(query, key)`` tensors of the same
            shape as the inputs.
        """
        from easydel.layers.rotary._compute_fns import apply_basic_rope

        return apply_basic_rope(
            query=query,
            key=key,
            positions=positions.astype(jnp.int32),
            frequencies=self.frequencies,
            rotary_dim=query.shape[-1],
            is_neox_style=True,
            dtype=query.dtype,
        )

    def forward(self, query: Array, key: Array, pixel_position_ids: Array | None) -> tuple[Array, Array]:
        """Apply independent RoPE rotations for the x and y patch coordinates.

        The first ``2 * (head_dim // (2 * ndim))`` channels per spatial axis are
        rotated using the corresponding axis position; the remainder of the head
        is passed through unchanged.

        Args:
            query (Array): Query tensor of shape
                ``(batch, seq_len, num_heads, head_dim)``.
            key (Array): Key tensor of the same shape as ``query``.
            pixel_position_ids (Array | None): ``(x, y)`` patch ids of shape
                ``(batch, seq_len, 2)``. When ``None``, both inputs are
                returned unchanged.

        Returns:
            tuple[Array, Array]: ``(query, key)`` with RoPE applied per axis.
        """
        if pixel_position_ids is None:
            return query, key

        pixel_position_ids = jnp.clip(pixel_position_ids.astype(jnp.int32), min=0)
        ndim = pixel_position_ids.shape[-1]
        num_rotated_channels_per_dim = 2 * (query.shape[-1] // (2 * ndim))
        rotated_channels = num_rotated_channels_per_dim * ndim

        query_rot, query_pass = query[..., :rotated_channels], query[..., rotated_channels:]
        key_rot, key_pass = key[..., :rotated_channels], key[..., rotated_channels:]
        query_parts = jnp.split(query_rot, ndim, axis=-1)
        key_parts = jnp.split(key_rot, ndim, axis=-1)

        rotated_query_parts = []
        rotated_key_parts = []
        for axis, (query_part, key_part) in enumerate(zip(query_parts, key_parts, strict=False)):
            query_axis, key_axis = self._apply_axis(query_part, key_part, pixel_position_ids[..., axis])
            rotated_query_parts.append(query_axis)
            rotated_key_parts.append(key_axis)

        query = jnp.concatenate([*rotated_query_parts, query_pass], axis=-1)
        key = jnp.concatenate([*rotated_key_parts, key_pass], axis=-1)
        return query, key


class Gemma4VisionAttention(spx.Module):
    """HF-compatible bidirectional vision attention block for Gemma4.

    Implements the multi-head bidirectional attention used inside each Gemma4
    vision encoder layer. Q/K/V are produced by clippable linear projections,
    Q and K are normalized by RMSNorm, V is normalized scale-free, and 2-D RoPE
    is applied via :class:`Gemma4VisionRotaryEmbedding` before the attention
    kernel.
    """

    def __init__(
        self,
        config: Gemma4VisionConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the vision attention block.

        Args:
            config (Gemma4VisionConfig): Vision encoder configuration.
            layer_idx (int): Index of this layer in the encoder stack.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to
                jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters.
                Defaults to jnp.bfloat16.
            precision (str | jax.lax.Precision | None, optional): Numerical
                precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        self.q_proj = Gemma4VisionClippableLinear(
            config,
            config.hidden_size,
            self.num_attention_heads * self.head_dim,
            parallel_mode="column",
            use_bias=bool(config.attention_bias),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.k_proj = Gemma4VisionClippableLinear(
            config,
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            parallel_mode="column",
            use_bias=bool(config.attention_bias),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.v_proj = Gemma4VisionClippableLinear(
            config,
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            parallel_mode="column",
            use_bias=bool(config.attention_bias),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.o_proj = Gemma4VisionClippableLinear(
            config,
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            parallel_mode="row",
            use_bias=bool(config.attention_bias),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.q_norm = Gemma4RMSNorm(param_dtype=param_dtype, dim=self.head_dim, epsilon=config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(param_dtype=param_dtype, dim=self.head_dim, epsilon=config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(
            config=None,
            param_dtype=param_dtype,
            dim=self.head_dim,
            epsilon=config.rms_norm_eps,
            with_scale=False,
        )
        self.attention_performer = FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=1.0,
            dropout_prob=config.attention_dropout,
            requires_cache=False,
        )

    def forward(
        self,
        hidden_states: Array,
        rotary_emb: Gemma4VisionRotaryEmbedding,
        pixel_position_ids: Array | None = None,
        mask_info: MaskInfo | None = None,
        output_attentions: bool = False,
    ) -> EncoderLayerOutput:
        """Apply attention, including 2-D patch rotary embeddings.

        Args:
            hidden_states (Array): Input tensor of shape
                ``(batch, seq_len, hidden_size)``.
            rotary_emb (Gemma4VisionRotaryEmbedding): Shared rotary embedding
                module driven by the patch positions.
            pixel_position_ids (Array | None, optional): ``(x, y)`` patch ids
                of shape ``(batch, seq_len, 2)`` for RoPE application. Defaults
                to None.
            mask_info (MaskInfo | None, optional): Optional bidirectional
                attention mask. Defaults to None.
            output_attentions (bool, optional): Whether to return attention
                weights. Defaults to False.

        Returns:
            EncoderLayerOutput: Hidden states after projection and optional
            attention weights.
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        batch_size, sequence_length, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states).reshape(
            batch_size, sequence_length, self.num_attention_heads, self.head_dim
        )
        key_states = self.k_proj(hidden_states).reshape(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        )
        value_states = self.v_proj(hidden_states).reshape(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        )

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        value_states = self.v_norm(value_states)
        query_states, key_states = rotary_emb(query_states, key_states, pixel_position_ids)

        attentions = self.attention_performer.forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            mode=common_types.MODE_TRAIN,
            mask_info=mask_info,
            causal=False,
        )
        attention_output = self.o_proj(attentions.attention_outputs.reshape(batch_size, sequence_length, -1))
        attention_output = apply_logical_sharding(
            attention_output,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        return EncoderLayerOutput(
            hidden_states=checkpoint_name(attention_output, "vision_attn_output"),
            attention_weight=attentions.attention_weights if output_attentions else None,
        )


class Gemma4VisionMLP(spx.Module):
    """HF-compatible Gemma4 vision MLP using ``*.linear.weight`` wrappers.

    Gated MLP with the same structure as the text decoder MLP
    (``down(act(gate(x)) * up(x))``) but built from
    :class:`Gemma4VisionClippableLinear` so that parameter paths match the
    Hugging Face checkpoint layout.
    """

    def __init__(
        self,
        config: Gemma4VisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the Gemma4 vision MLP.

        Args:
            config (Gemma4VisionConfig): Vision encoder configuration providing
                ``hidden_size``, ``intermediate_size``, and
                ``hidden_activation``.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to
                jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters.
                Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision.
                Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        self.act = ACT2FN[config.hidden_activation]
        self.gate_proj = Gemma4VisionClippableLinear(
            config,
            config.hidden_size,
            config.intermediate_size,
            parallel_mode="column",
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.up_proj = Gemma4VisionClippableLinear(
            config,
            config.hidden_size,
            config.intermediate_size,
            parallel_mode="column",
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.down_proj = Gemma4VisionClippableLinear(
            config,
            config.intermediate_size,
            config.hidden_size,
            parallel_mode="row",
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def forward(self, hidden_states: Array) -> Array:
        """Apply the gated MLP transformation.

        Args:
            hidden_states (Array): Input tensor of shape
                ``(batch, seq_len, hidden_size)``.

        Returns:
            Array: Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        gate = checkpoint_name(self.act(self.gate_proj(hidden_states)), "vision_mlp_gate")
        up = checkpoint_name(self.up_proj(hidden_states), "vision_mlp_up")
        hidden_states = checkpoint_name(self.down_proj(gate * up), "vision_mlp_down")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        return checkpoint_name(hidden_states, "vision_mlp_output")


class Gemma4VisionEncoderLayer(spx.Module):
    """HF-compatible vision transformer block for Gemma4.

    A single bidirectional encoder block with pre/post-norm structure around
    both the attention and MLP sub-blocks, matching the Hugging Face Gemma4
    vision encoder reference layout.
    """

    def __init__(
        self,
        config: Gemma4VisionConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize one Gemma4 vision encoder layer.

        Args:
            config (Gemma4VisionConfig): Vision encoder configuration.
            layer_idx (int): Index of this layer in the encoder stack.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to
                jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters.
                Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision.
                Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        self.layer_idx = layer_idx
        self.input_layernorm = Gemma4RMSNorm(config, param_dtype=param_dtype)
        self.self_attn = Gemma4VisionAttention(
            config=config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.post_attention_layernorm = Gemma4RMSNorm(config, param_dtype=param_dtype)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(config, param_dtype=param_dtype)
        self.post_feedforward_layernorm = Gemma4RMSNorm(config, param_dtype=param_dtype)
        self.mlp = Gemma4VisionMLP(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def forward(
        self,
        hidden_states: Array,
        rotary_emb: Gemma4VisionRotaryEmbedding,
        pixel_position_ids: Array | None = None,
        mask_info: MaskInfo | None = None,
        attention_mask: Array | None = None,
        output_attentions: bool = False,
    ) -> EncoderLayerOutput:
        """Run one Gemma4 vision encoder block.

        Applies pre-norm bidirectional attention with both pre- and
        post-feedforward RMSNorm, residual connections, and an optional zero
        mask on padded patch positions.

        Args:
            hidden_states (Array): Input tensor of shape
                ``(batch, num_patches, hidden_size)``.
            rotary_emb (Gemma4VisionRotaryEmbedding): Shared RoPE module.
            pixel_position_ids (Array | None, optional): ``(x, y)`` patch ids
                of shape ``(batch, num_patches, 2)``. Defaults to None.
            mask_info (MaskInfo | None, optional): Bidirectional attention
                mask. Defaults to None.
            attention_mask (Array | None, optional): Boolean per-token mask of
                shape ``(batch, num_patches)``; padded tokens are zeroed in
                the residual path. Defaults to None.
            output_attentions (bool, optional): Whether to return attention
                weights. Defaults to False.

        Returns:
            EncoderLayerOutput: Updated hidden states and optional attention
            weights.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_outputs = self.self_attn(
            hidden_states=hidden_states,
            rotary_emb=rotary_emb,
            pixel_position_ids=pixel_position_ids,
            mask_info=mask_info,
            output_attentions=output_attentions,
        )
        hidden_states = checkpoint_name(
            residual + self.post_attention_layernorm(attention_outputs.hidden_states), "residual"
        )

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = checkpoint_name(
            residual + self.post_feedforward_layernorm(self.mlp(hidden_states)),
            "residual",
        )
        if attention_mask is not None:
            hidden_states = jnp.where(attention_mask[..., None], hidden_states, 0.0)

        return EncoderLayerOutput(
            hidden_states=checkpoint_name(hidden_states, "vision_layer_output"),
            attention_weight=attention_outputs.attention_weight,
        )


class Gemma4VisionEncoder(EasyDeLLayerStackMixin, spx.Module):
    """Shared rotary embedding plus stacked Gemma4 vision encoder layers.

    Stacks ``num_hidden_layers`` :class:`Gemma4VisionEncoderLayer` blocks under
    a shared :class:`Gemma4VisionRotaryEmbedding` and runs them sequentially via
    the layer-stack scan, with support for gradient checkpointing and pipeline
    staging through :class:`EasyDeLLayerStackMixin`.
    """

    def __init__(
        self,
        config: Gemma4VisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the Gemma4 vision encoder stack.

        Args:
            config (Gemma4VisionConfig): Vision encoder configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to
                jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters.
                Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision.
                Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        self.rotary_emb = Gemma4VisionRotaryEmbedding(config=config, dtype=dtype)
        remat_layer_block = auto_remat(
            Gemma4VisionEncoderLayer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.layers = nn.ModuleList([])
        for layer_idx in range(config.num_hidden_layers):
            with self.assign_layer_stage(layer_idx, total_layers=config.num_hidden_layers):
                self.layers.append(
                    remat_layer_block(
                        config=config,
                        layer_idx=layer_idx,
                        dtype=dtype,
                        param_dtype=param_dtype,
                        precision=precision,
                        rngs=rngs,
                    )
                )

    def forward(
        self,
        inputs_embeds: Array,
        attention_mask: Array | None,
        pixel_position_ids: Array | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> BaseModelOutput:
        """Encode patch embeddings with bidirectional attention.

        Args:
            inputs_embeds (Array): Patch embeddings of shape
                ``(batch, num_patches, hidden_size)`` from the patch embedder.
            attention_mask (Array | None): Boolean mask of shape
                ``(batch, num_patches)`` marking valid (True) tokens; used to
                build a bidirectional attention mask.
            pixel_position_ids (Array | None, optional): ``(x, y)`` patch ids
                of shape ``(batch, num_patches, 2)`` for RoPE. Defaults to None.
            output_attentions (bool, optional): Whether to collect attention
                weights from each layer. Defaults to False.
            output_hidden_states (bool, optional): Whether to collect hidden
                states from each layer. Defaults to False.

        Returns:
            BaseModelOutput: Final hidden states with optional per-layer
            hidden states and attention weights.
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        mask_info = None
        if attention_mask is not None:
            attention_mask = attention_mask.astype(jnp.bool_)
            bidirectional_mask = attention_mask[:, None, :, None] & attention_mask[:, None, None, :]
            mask_info = MaskInfo(_attention_mask=bidirectional_mask)

        hidden_states = inputs_embeds

        def _layer_loop(layer, carry):
            hidden_states, all_hidden_states, all_attentions, idx = carry
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            with self._layer_stage_context(idx, layers=self.layers):
                layer_outputs = layer(
                    hidden_states=hidden_states,
                    rotary_emb=self.rotary_emb,
                    pixel_position_ids=pixel_position_ids,
                    mask_info=mask_info,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                )
            hidden_states = self._mark_layer_stage_boundary(layer_outputs.hidden_states, idx, layers=self.layers)

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            return hidden_states, all_hidden_states, all_attentions, idx + 1

        hidden_states, all_hidden_states, all_attentions, _ = self.layers.scan(
            _layer_loop,
            (hidden_states, all_hidden_states, all_attentions, 0),
            trace=output_hidden_states
            or output_attentions
            or not self.config.scan_layers
            or self._pipeline_stage_count() > 1,
        )
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class Gemma4VisionPooler(spx.Module):
    """Spatial pooling and hidden-size scaling for Gemma4 vision tokens.

    Reduces the encoder output from ``num_patches`` patches to ``output_length``
    soft tokens by averaging positional groups, zeroes padded entries, then
    scales the result by ``sqrt(hidden_size)`` to match the language model's
    embedding scale.

    Attributes:
        hidden_size (int): Encoder hidden size.
        root_hidden_size (float): Square root of ``hidden_size`` used as the
            output scale factor.
    """

    def __init__(self, config: Gemma4VisionConfig):
        """Initialize the vision pooler.

        Args:
            config (Gemma4VisionConfig): Vision config providing
                ``hidden_size``.
        """
        self.hidden_size = config.hidden_size
        self.root_hidden_size = config.hidden_size**0.5

    def _avg_pool_by_positions(
        self,
        hidden_states: Array,
        pixel_position_ids: Array,
        length: int,
    ) -> tuple[Array, Array]:
        """Average-pool tokens to ``length`` soft tokens via positional groups.

        Computes per-token weights from the patch positions so that each output
        soft token is the average of a square ``kernel_size x kernel_size`` block.

        Args:
            hidden_states (Array): Encoder outputs of shape
                ``(batch, num_patches, hidden_size)``.
            pixel_position_ids (Array): ``(x, y)`` patch ids of shape
                ``(batch, num_patches, 2)``.
            length (int): Target number of soft tokens.

        Returns:
            tuple[Array, Array]: ``(pooled_hidden_states, valid_mask)`` of
            shapes ``(batch, length, hidden_size)`` and ``(batch, length)``.

        Raises:
            ValueError: If ``length`` does not divide ``num_patches`` into a
                square kernel.
        """
        input_seq_len = hidden_states.shape[1]
        kernel_size = int((input_seq_len // length) ** 0.5)
        kernel_size_squared = kernel_size**2
        if kernel_size_squared * length != input_seq_len:
            raise ValueError(
                f"Cannot pool hidden states of shape {hidden_states.shape} to length {length}: "
                f"{kernel_size=}^2 times {length=} must equal {input_seq_len}."
            )

        clamped_positions = jnp.clip(pixel_position_ids.astype(jnp.int32), min=0)
        max_x = jnp.max(clamped_positions[..., 0], axis=-1, keepdims=True) + 1
        kernel_indices = clamped_positions // kernel_size
        kernel_indices = kernel_indices[..., 0] + (max_x // kernel_size) * kernel_indices[..., 1]
        weights = jax.nn.one_hot(kernel_indices, length, dtype=jnp.float32) / kernel_size_squared
        output = jnp.einsum("bnl,bnh->blh", weights, hidden_states.astype(jnp.float32)).astype(hidden_states.dtype)
        mask = jnp.any(weights != 0, axis=1)
        return output, mask

    def forward(
        self,
        hidden_states: Array,
        pixel_position_ids: Array,
        padding_positions: Array,
        output_length: int | None = None,
    ) -> tuple[Array, Array]:
        """Pool encoder outputs into soft tokens and scale by ``sqrt(hidden)``.

        Args:
            hidden_states (Array): Encoder outputs of shape
                ``(batch, num_patches, hidden_size)``.
            pixel_position_ids (Array): ``(x, y)`` patch ids.
            padding_positions (Array): Boolean padding mask of shape
                ``(batch, num_patches)``.
            output_length (int | None, optional): Target number of soft tokens.
                If None, no spatial pooling is applied. Defaults to None.

        Returns:
            tuple[Array, Array]: ``(soft_tokens, valid_mask)``.

        Raises:
            ValueError: If ``output_length`` exceeds the number of input
                patches.
        """
        if output_length is None:
            output_length = hidden_states.shape[1]
        if output_length > hidden_states.shape[1]:
            raise ValueError(
                f"Cannot output more soft tokens ({output_length}) than there are patches ({hidden_states.shape[1]})."
            )

        hidden_states = jnp.where(padding_positions[..., None], 0.0, hidden_states)
        valid_mask = ~padding_positions
        if hidden_states.shape[1] != output_length:
            hidden_states, valid_mask = self._avg_pool_by_positions(hidden_states, pixel_position_ids, output_length)

        hidden_states = hidden_states * self.root_hidden_size
        return hidden_states, valid_mask


@register_module(TaskType.BASE_VISION, config=Gemma4VisionConfig, model_type="gemma4_vision")
@register_module(TaskType.BASE_MODULE, config=Gemma4VisionConfig, model_type="gemma4_vision")
class Gemma4VisionModel(EasyDeLBaseModule):
    """HF-compatible Gemma4 vision encoder for multimodal checkpoints.

    Wraps the patch embedder, encoder stack, and pooler into a single
    end-to-end vision tower that accepts either NCHW/NHWC raw pixels or
    pre-patchified inputs and returns flattened soft-token embeddings ready
    for the multimodal embedder.
    """

    config_class = Gemma4VisionConfig

    def __init__(
        self,
        config: Gemma4VisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the Gemma4 vision tower.

        Args:
            config (Gemma4VisionConfig): Vision encoder configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to
                jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters.
                Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision.
                Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.patch_embedder = Gemma4VisionPatchEmbedder(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.encoder = Gemma4VisionEncoder(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.pooler = Gemma4VisionPooler(config)
        if config.standardize:
            self.std_bias = ArrayParam.bound(
                shape=(config.hidden_size,),
                dtype=param_dtype,
                init_method="zeros",
                key=None,
            )
            self.std_scale = ArrayParam.bound(
                shape=(config.hidden_size,),
                dtype=param_dtype,
                init_method="ones",
                key=None,
            )

    def forward(
        self,
        pixel_values: Array,
        pixel_position_ids: Array | None = None,
        image_position_ids: Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        """Encode images or flat patches into Gemma4 soft tokens.

        Accepts rank-4 pixel inputs (NCHW or NHWC, auto-detected) or rank-3
        flat patches. Patches are embedded, encoded by the bidirectional
        transformer stack, spatially pooled, optionally standardized, and then
        flattened across the (batch, valid-token) axis.

        Args:
            pixel_values (Array): Either ``(batch, channels, height, width)``,
                ``(batch, height, width, channels)``, or
                ``(batch, num_patches, patch_dim)``.
            pixel_position_ids (Array | None, optional): Per-patch ``(x, y)``
                ids. Defaults to None.
            image_position_ids (Array | None, optional): Alias for
                ``pixel_position_ids`` accepted for API compatibility.
                Defaults to None.
            output_attentions (bool | None, optional): Whether to return
                attention weights. Defaults to None (False).
            output_hidden_states (bool | None, optional): Whether to return
                per-layer hidden states. Defaults to None (False).

        Returns:
            BaseModelOutput: Flattened soft-token embeddings as
            ``last_hidden_state`` plus optional per-layer hidden states and
            attentions.
        """
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        pixel_position_ids = pixel_position_ids if pixel_position_ids is not None else image_position_ids

        patches, pixel_position_ids, padding_positions = _gemma4_vision_prepare_inputs(
            pixel_values=pixel_values,
            patch_size=self.config.patch_size,
            pixel_position_ids=pixel_position_ids,
        )
        output_length = patches.shape[-2] // max(int(self.config.pooling_kernel_size), 1) ** 2
        inputs_embeds = self.patch_embedder(
            pixel_values=patches,
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
        )
        encoder_outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=~padding_positions,
            pixel_position_ids=pixel_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states, pooler_mask = self.pooler(
            hidden_states=encoder_outputs.last_hidden_state,
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
            output_length=output_length,
        )
        hidden_states = jnp.where(pooler_mask[..., None], hidden_states, 0).reshape(-1, hidden_states.shape[-1])
        if self.config.standardize:
            hidden_states = (hidden_states - self.std_bias.value) * self.std_scale.value
        hidden_states = checkpoint_name(hidden_states, "vision_model_output")
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def get_encoder(self):
        """Return the vision encoder stack."""
        return self.encoder

    def get_decoder(self):
        """Gemma4 vision is encoder-only."""
        raise NotImplementedError("Gemma4 vision is an encoder-only model.")

    def get_lm_head(self):
        """Gemma4 vision does not expose an LM head."""
        raise NotImplementedError("Gemma4 vision does not define an LM head.")

    def get_embedding(self):
        """Return the patch projection stem."""
        return self.patch_embedder.input_proj


class Gemma4Attention(UnifiedAttention):
    """Multi-head attention layer for Gemma4 with per-layer architectural variation.

    Extends ``UnifiedAttention`` with the following Gemma4-specific features:

    **Per-layer attention type selection**: Each layer is configured as either
    ``"sliding_attention"`` (local window) or ``"full_attention"`` (global), with
    different head dimensions, KV head counts, and RoPE parameters for each type.

    **Q/K/V normalization**: All three projection outputs are normalized via
    RMSNorm. Query and key norms use learned scale parameters (standard Gemma
    convention), while value normalization is scale-free (``with_scale=False``).

    **Key-equals-value (k_eq_v)**: For global attention layers when
    ``attention_k_eq_v=True``, the key projection output is reused as the value
    input (before normalization). This eliminates the separate ``v_proj`` weight
    matrix, reducing parameters by ~33% in the attention block while the v_norm
    still produces a distinct representation from the k_norm output.

    **KV sharing**: Layers beyond the ``num_kv_shared_layers`` threshold reuse
    cached key-value states from the last non-shared layer of the same attention
    type, avoiding redundant KV computation in the later layers of deep models.

    **Different head geometry per layer type**: Global layers can use
    ``global_head_dim`` and ``num_global_key_value_heads`` (typically larger and
    fewer respectively) while sliding layers use the standard ``head_dim`` and
    ``num_key_value_heads``.

    Args:
        config: Text model configuration containing all attention parameters.
        layer_idx: Zero-based index of this layer in the decoder stack, used to
            determine the attention type from ``config.layer_types``.
        dtype: Data type for computation tensors. Defaults to bfloat16.
        param_dtype: Data type for learnable parameters. Defaults to bfloat16.
        precision: JAX numerical precision for matrix multiplications.
        causal: Whether to apply causal (autoregressive) attention masking.
        is_cross_attention: Whether this layer performs cross-attention.
        rngs: SpecTrax random number generator state for parameter initialization.
    """

    def __init__(
        self,
        config: Gemma4TextConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        causal: bool = True,
        is_cross_attention: bool = False,
        *,
        rngs: spx.Rngs,
    ):
        self.layer_type = config.layer_types[layer_idx] if config.layer_types else None
        self.is_sliding = self.layer_type == "sliding_attention"
        self.use_alternative_attention = config.attention_k_eq_v and not self.is_sliding

        self._head_dim = self._resolve_head_dim(config)
        self._num_kv_heads = self._resolve_num_kv_heads(config)

        first_kv_shared_layer_idx = config.num_hidden_layers - getattr(config, "num_kv_shared_layers", 0)
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        self.kv_shared_layer_index = None
        if self.is_kv_shared_layer:
            prev_layers = config.layer_types[:first_kv_shared_layer_idx]
            layer_type = config.layer_types[layer_idx]
            if layer_type in prev_layers:
                self.kv_shared_layer_index = len(prev_layers) - 1 - prev_layers[::-1].index(layer_type)
            else:
                self.is_kv_shared_layer = False

        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="standard",
            causal=True,
            sliding_window=config.sliding_window if self.is_sliding else None,
            use_qk_norm=True,
        )

        self.layer_idx = layer_idx
        self.is_cross_attention = is_cross_attention
        self.causal = causal
        self._head_dim = self._resolve_head_dim(config)
        self._num_kv_heads = self._resolve_num_kv_heads(config)

        self.v_norm = Gemma4RMSNorm(
            config=config,
            param_dtype=param_dtype,
            dim=self._head_dim,
            with_scale=False,
        )

    @staticmethod
    def _hf_sliding_window(sliding_window: int | None) -> tuple[int, int] | None:
        """Convert HF Gemma4 sliding-window size to EasyDeL's inclusive tuple form."""
        if sliding_window is None:
            return None
        return (max(int(sliding_window) - 1, 0), 0)

    def _kernel_sliding_window(self) -> int | tuple[int, int] | None:
        """Choose the sliding-window representation expected by the active attention kernel."""
        if self.sliding_window is None:
            return None

        impl = getattr(getattr(self, "attention_performer", None), "impl", None)
        impl_name = type(impl).__name__.lower() if impl is not None else ""

        # ejkernel ragged/page backends accept an integer window size, but they
        # still follow Gemma4's HF semantics where a window of N means the
        # current token plus the previous N-1 cache positions.
        if any(token in impl_name for token in ("ragged", "page", "decode", "unified")):
            return max(int(self.sliding_window) - 1, 0)

        return self._hf_sliding_window(self.sliding_window)

    @property
    def head_dim(self):
        """Per-head dimension, which varies between global and sliding layers."""
        return self._head_dim

    @head_dim.setter
    def head_dim(self, value):
        self._head_dim = value

    def _resolve_head_dim(self, config: Gemma4TextConfig) -> int:
        """Return the effective head dimension for this layer type."""
        if not self.is_sliding and config.global_head_dim:
            return config.global_head_dim
        return config.head_dim

    def _resolve_num_kv_heads(self, config: Gemma4TextConfig) -> int:
        """Return the effective KV head count for this layer type."""
        if self.use_alternative_attention and config.num_global_key_value_heads is not None:
            return config.num_global_key_value_heads
        return config.num_key_value_heads

    def define_network(self, config, dtype, param_dtype, precision, rngs):
        """Build projection layers, RoPE, attention performer, and QK norms.

        Overrides the parent to configure layer-type-specific head dimensions
        and KV head counts before creating the projection matrices. Also handles
        the k_eq_v case where ``v_proj`` aliases ``k_proj``.

        Args:
            config: Text model configuration.
            dtype: Computation data type.
            param_dtype: Parameter data type.
            precision: JAX precision setting.
            rngs: Random number generators for initialization.
        """
        resolved_head_dim = self._resolve_head_dim(config)
        resolved_num_kv_heads = self._resolve_num_kv_heads(config)

        self._head_dim = resolved_head_dim
        self._num_kv_heads = resolved_num_kv_heads
        self.head_dim = resolved_head_dim
        self.num_key_value_heads = resolved_num_kv_heads
        self.num_key_value_groups = config.num_attention_heads // resolved_num_kv_heads

        kernel_init = jax.nn.initializers.normal(config.initializer_range)
        column = partial(
            ColumnParallelLinear,
            use_bias=config.attention_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        row = partial(
            RowParallelLinear,
            use_bias=config.attention_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            rngs=rngs,
        )

        self.q_proj = column(config.hidden_size, config.num_attention_heads * resolved_head_dim)
        self.k_proj = column(config.hidden_size, resolved_num_kv_heads * resolved_head_dim)
        if not self.use_alternative_attention:
            self.v_proj = column(config.hidden_size, resolved_num_kv_heads * resolved_head_dim)
        else:
            self.v_proj = self.k_proj
        self.o_proj = row(config.num_attention_heads * resolved_head_dim, config.hidden_size)

        self.rotary = self._create_rotary(config, dtype)
        self.attention_performer = self._create_attention_performer(config, rngs)

        if self.use_qk_norm:
            self.q_norm = self._create_q_norm(config, dtype, param_dtype, rngs)
            self.k_norm = self._create_k_norm(config, dtype, param_dtype, rngs)

    def _create_q_norm(self, config, dtype, param_dtype, rngs):
        """Create query normalization using Gemma4's (1+kernel)*norm convention."""
        return Gemma4RMSNorm(config, param_dtype=param_dtype, dim=self._head_dim)

    def _create_k_norm(self, config, dtype, param_dtype, rngs):
        """Create key normalization using Gemma4's (1+kernel)*norm convention."""
        return Gemma4RMSNorm(config, param_dtype=param_dtype, dim=self._head_dim)

    def _create_attention_performer(self, config, rngs):
        """Create the attention computation module.

        Gemma4 uses ``scaling=1.0`` because the Q/K normalization already
        controls the magnitude of the attention logits, making the traditional
        ``1/sqrt(d_k)`` scaling unnecessary.
        """
        return FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=1.0,
            dropout_prob=config.attention_dropout,
        )

    def _postprocess_qkv(self, query_states, key_states, value_states):
        """Apply Q/K/V normalization after projection and multi-head reshape.

        For standard layers, all three states are independently normalized.
        For k_eq_v layers, value_states (which is the raw key projection
        aliased by ``_preprocess_qkv``) is passed through the scale-free
        v_norm *before* key_states is overwritten by k_norm.  This matches
        the HuggingFace reference where ``v_norm`` operates on the raw
        projection output, not the already-normed key.

        Args:
            query_states: Query tensor ``[batch, seq_len, num_heads, head_dim]``.
            key_states: Key tensor ``[batch, seq_len, num_kv_heads, head_dim]``.
            value_states: Value tensor ``[batch, seq_len, num_kv_heads, head_dim]``.

        Returns:
            Tuple of normalized (query, key, value) tensors.
        """
        query_states = self.q_norm(query_states)
        if self.use_alternative_attention:
            # k_eq_v: value_states is the raw key projection (aliased in
            # _preprocess_qkv).  Normalize it *before* k_norm overwrites
            # key_states so that v_norm sees the raw projection, not a
            # double-normed tensor.
            value_states = self.v_norm(value_states)
            key_states = self.k_norm(key_states)
        else:
            key_states = self.k_norm(key_states)
            value_states = self.v_norm(value_states)
        return query_states, key_states, value_states

    def _preprocess_qkv(self, query_states, key_states, value_states):
        """Handle k_eq_v aliasing before the multi-head reshape.

        When ``use_alternative_attention`` is enabled, the key projection output
        is used directly as the value input (they share the same linear
        projection). The actual differentiation between K and V happens later
        in ``_postprocess_qkv`` via separate normalization.

        Args:
            query_states: Raw query projection output.
            key_states: Raw key projection output.
            value_states: Raw value projection output (ignored when k_eq_v).

        Returns:
            Tuple of (query, key, value) with value aliased to key when k_eq_v.
        """
        if self.use_alternative_attention:
            value_states = key_states
        return query_states, key_states, value_states

    def _forward_with_kv_capture(
        self,
        hidden_states,
        mask_info,
        position_ids,
        mode,
        cache_view,
        cache_metadata,
        output_attentions,
        frequencies,
    ):
        """Standard forward that also captures post-norm, post-RoPE K/V.

        The captured K/V are stored in ``self._captured_kv`` so the model
        loop can pass them to downstream shared layers.

        Args:
            hidden_states: Input tensor ``[batch, seq_len, hidden_dim]``.
            mask_info: Attention mask configuration.
            position_ids: Position indices ``[batch, seq_len]``.
            mode: Runtime execution mode.
            cache_view: KV cache view for this layer.
            cache_metadata: Cache management metadata.
            output_attentions: Whether to return attention weights.
            frequencies: Precomputed RoPE frequencies.

        Returns:
            ``AttentionLayerOutput`` with attention output, optional weights,
            and updated cache view.
        """
        batch_size = hidden_states.shape[0]
        sequence_length = hidden_states.shape[1]

        query_states = checkpoint_name(self.query_projection(hidden_states), "attn_query")
        key_states = checkpoint_name(self.key_projection(hidden_states), "attn_key")
        value_states = checkpoint_name(self.value_projection(hidden_states), "attn_value")

        query_states, key_states, value_states = self._preprocess_qkv(
            query_states,
            key_states,
            value_states,
        )

        query_states = query_states.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        key_states = key_states.reshape(batch_size, sequence_length, self.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(batch_size, sequence_length, self.num_key_value_heads, self.head_dim)

        query_states, key_states, value_states = self._postprocess_qkv(
            query_states,
            key_states,
            value_states,
        )
        query_states, key_states, value_states = self.apply_qkv_shardings(
            query_states,
            key_states,
            value_states,
        )
        query_states, key_states = self._apply_rotary(
            query_states,
            key_states,
            position_ids,
            frequencies,
        )

        # Capture post-norm, post-RoPE K/V for potential downstream sharing.
        # Stored temporarily on the object to be harvested by the model loop.
        # Using object.__setattr__ to bypass SpecTrax pytree validation.
        object.__setattr__(self, "_captured_kv", (key_states, value_states))

        causal_for_kernel = self.causal
        if mask_info is not None and getattr(mask_info, "_causal_baked", False):
            causal_for_kernel = False

        sliding_window_for_kernel = self._kernel_sliding_window()
        if mask_info is not None and getattr(mask_info, "sliding_window_baked_in", False):
            sliding_window_for_kernel = None

        (
            key_states,
            value_states,
            mask_info,
            init_attention_bias,
            cache_view,
            cache_metadata,
        ) = self.concatenate(
            query=query_states,
            key=key_states,
            value=value_states,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            mask_info=mask_info,
            sliding_window=sliding_window_for_kernel,
        )
        if mask_info is not None and getattr(mask_info, "sliding_window_baked_in", False):
            sliding_window_for_kernel = None

        softmax_aux = getattr(self, "sinks", getattr(self, "softmax_aux", None))
        softmax_aux = getattr(softmax_aux, "value", softmax_aux)

        attentions = self.attention_performer.forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            mode=mode,
            bias=None,
            cache_metadata=cache_metadata,
            cache_view=cache_view,
            init_bias=init_attention_bias,
            mask_info=mask_info,
            causal=causal_for_kernel,
            sliding_window=sliding_window_for_kernel,
            softmax_aux=softmax_aux,
        )

        if attentions.cache_view is not None:
            cache_view = attentions.cache_view

        attention_out = self._merge_heads(attentions.attention_outputs)
        attn_output = self.shard_attention_prod(attention_out)
        attn_output = checkpoint_name(self.output_projection(attn_output), "attn_output")

        if hasattr(self, "resid_dropout"):
            attn_output = self.resid_dropout(attn_output)

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weight if output_attentions else None,
            cache_view=cache_view,
        )

    def forward(
        self,
        hidden_states,
        mask_info,
        position_ids,
        mode,
        cache_view=None,
        cache_metadata=None,
        output_attentions=False,
        frequencies=None,
        alibi=None,
        shared_key_value=None,
    ):
        """Gemma4 attention forward with optional KV sharing.

        When ``shared_key_value`` is provided (a ``(key, value)`` tuple of
        post-norm, post-RoPE tensors from the donor layer), the K/V
        projections, norms, and rotary application are skipped and the shared
        states are used directly.  Only Q projection, Q-norm, and Q-RoPE are
        computed fresh.

        Args:
            hidden_states: Input tensor ``[batch, seq_len, hidden_dim]``.
            mask_info: Attention mask configuration.
            position_ids: Position indices ``[batch, seq_len]``.
            mode: Runtime execution mode.
            cache_view: KV cache view for this layer.
            cache_metadata: Cache management metadata.
            output_attentions: Whether to return attention weights.
            frequencies: Precomputed RoPE frequencies.
            alibi: ALiBi bias (unused for standard attention).
            shared_key_value: Optional ``(key_states, value_states)`` tuple
                from the donor layer.  Both tensors are already normed and
                rotary-embedded with shape
                ``[batch, seq_len, num_kv_heads, head_dim]``.

        Returns:
            ``AttentionLayerOutput`` with attention output, optional weights,
            and post-attention key/value states for downstream sharing.
        """
        if shared_key_value is None:
            return self._forward_with_kv_capture(
                hidden_states,
                mask_info,
                position_ids,
                mode,
                cache_view,
                cache_metadata,
                output_attentions,
                frequencies,
            )

        # KV sharing path: use shared K/V, compute only Q
        batch_size = hidden_states.shape[0]
        sequence_length = hidden_states.shape[1]

        query_states = checkpoint_name(self.query_projection(hidden_states), "attn_query")
        query_states = query_states.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )
        query_states = self.q_norm(query_states)

        key_states, value_states = shared_key_value
        query_states, key_states, value_states = self.apply_qkv_shardings(
            query_states,
            key_states,
            value_states,
        )
        query_states, _ = self._apply_rotary(
            query_states,
            key_states,
            position_ids,
            frequencies,
        )

        causal_for_kernel = self.causal
        if mask_info is not None and getattr(mask_info, "_causal_baked", False):
            causal_for_kernel = False

        sliding_window_for_kernel = self._kernel_sliding_window()
        if mask_info is not None and getattr(mask_info, "sliding_window_baked_in", False):
            sliding_window_for_kernel = None

        # Shared layers read from the donor's cache view (which is aliased
        # to this layer's view).  The donor already wrote K/V for the current
        # tokens, so we call concatenate for cache reads (mask setup, bias
        # init) but the write is effectively a no-op since the pages already
        # contain the correct data.  For the eager (no-cache) path the shared
        # K/V are used directly without any cache interaction.
        init_attention_bias = None
        if cache_view is not None:
            (
                key_states,
                value_states,
                mask_info,
                init_attention_bias,
                cache_view,
                cache_metadata,
            ) = self.concatenate(
                query=query_states,
                key=key_states,
                value=value_states,
                cache_view=cache_view,
                cache_metadata=cache_metadata,
                mask_info=mask_info,
                sliding_window=sliding_window_for_kernel,
            )
            if mask_info is not None and getattr(mask_info, "sliding_window_baked_in", False):
                sliding_window_for_kernel = None

        softmax_aux = getattr(self, "sinks", getattr(self, "softmax_aux", None))
        softmax_aux = getattr(softmax_aux, "value", softmax_aux)

        attentions = self.attention_performer.forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            mode=mode,
            bias=None,
            cache_metadata=cache_metadata,
            cache_view=cache_view,
            init_bias=init_attention_bias,
            mask_info=mask_info,
            causal=causal_for_kernel,
            sliding_window=sliding_window_for_kernel,
            softmax_aux=softmax_aux,
        )

        if attentions.cache_view is not None:
            cache_view = attentions.cache_view

        attention_out = self._merge_heads(attentions.attention_outputs)
        attn_output = self.shard_attention_prod(attention_out)
        attn_output = checkpoint_name(self.output_projection(attn_output), "attn_output")

        if hasattr(self, "resid_dropout"):
            attn_output = self.resid_dropout(attn_output)

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weight if output_attentions else None,
            cache_view=cache_view,
        )


class Gemma4MLP(spx.Module):
    """Gated feed-forward network for Gemma4 decoder layers.

    Implements the standard gated MLP: ``down_proj(act(gate_proj(x)) * up_proj(x))``
    with support for double-wide intermediate dimensions in KV-shared layers.

    When ``use_double_wide_mlp=True`` and the layer falls within the KV-sharing
    region (last ``num_kv_shared_layers`` layers), the intermediate size is
    doubled to compensate for the reduced attention capacity from sharing KV
    projections.

    Uses ``ColumnParallelLinear`` for gate/up projections and ``RowParallelLinear``
    for the down projection to enable tensor-parallel execution.

    Args:
        config: Text model configuration with MLP parameters.
        layer_idx: Layer index, used to determine KV-sharing status.
        dtype: Computation data type.
        param_dtype: Parameter data type.
        precision: JAX numerical precision for matrix operations.
        rngs: Random number generator state.
    """

    def __init__(
        self,
        config: Gemma4TextConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: spx.Rngs,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        first_kv_shared_layer_idx = config.num_hidden_layers - config.num_kv_shared_layers
        is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        use_double_wide = config.use_double_wide_mlp and is_kv_shared_layer

        embed_dim = config.hidden_size
        inner_dim = config.intermediate_size * (2 if use_double_wide else 1)
        kernel_init = jax.nn.initializers.normal(config.initializer_range)

        self.act = ACT2FN[config.hidden_activation]

        column = partial(
            ColumnParallelLinear,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        row = partial(
            RowParallelLinear,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.gate_proj = column(embed_dim, inner_dim)
        self.up_proj = column(embed_dim, inner_dim)
        self.down_proj = row(inner_dim, embed_dim)

    def forward(self, hidden_states: Array) -> Array:
        """Apply the gated MLP transformation.

        Computes ``down_proj(activation(gate_proj(x)) * up_proj(x))`` with
        logical sharding applied at input and output boundaries for tensor
        parallelism.

        Args:
            hidden_states: Input tensor of shape
                ``(batch_size, sequence_length, hidden_size)``.

        Returns:
            Transformed tensor with the same shape as the input.
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        gate = self.gate_proj(hidden_states)
        up = checkpoint_name(self.up_proj(hidden_states), "mlp_up")
        if self.config.activations_in_float32:
            gate = gate.astype(jnp.float32)
            up = up.astype(jnp.float32)
        gate = checkpoint_name(self.act(gate), "mlp_gate")
        hidden_states = checkpoint_name(self.down_proj((gate * up).astype(self.dtype)), "mlp_down")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        return checkpoint_name(hidden_states, "mlp_output")


class Gemma4TextMLPStack(spx.Module):
    """Expert MLP stack for Gemma4 using ``ColumnParallelMoELinear`` / ``RowParallelMoELinear``.

    Each expert implements a gated MLP: ``down_proj(act(gate_proj(x)) * up_proj(x))``.
    The gate and up projections use column-parallel sharding and the down projection
    uses row-parallel sharding, enabling efficient tensor-parallel execution via
    ``grouped_matmul`` over sorted token groups.

    HuggingFace Gemma4 stores the gate and up weights as a single fused
    ``gate_up_proj`` tensor of shape ``[num_experts, 2*intermediate, hidden]``.
    The ``reform_param`` class variable defines the split/merge transformations
    for converting between the fused HF layout and EasyDeL's separate
    ``gate_proj``/``up_proj`` layout during weight loading.

    Args:
        config: Text model configuration with ``num_experts``,
            ``moe_intermediate_size``, ``hidden_size``, and ``hidden_activation``.
        dtype: Computation data type.
        param_dtype: Parameter data type.
        precision: JAX numerical precision.
        rngs: Random number generator state.
    """

    reform_param: typing.ClassVar = {
        "gate_up_proj$": {
            "splits": [
                {
                    "name": "gate_proj.weight",
                    "spliter": lambda x: x[:, : x.shape[1] // 2, :].swapaxes(-1, -2),
                },
                {
                    "name": "up_proj.weight",
                    "spliter": lambda x: x[:, x.shape[1] // 2 :, :].swapaxes(-1, -2),
                },
            ],
            "inverse_spliter": lambda torch, gate, up: torch.cat(
                (gate.transpose(-1, -2), up.transpose(-1, -2)),
                dim=1,
            ),
        },
        "down_proj$": {
            "splits": [
                {"name": "down_proj.weight", "spliter": lambda x: x.swapaxes(-1, -2)},
            ],
            "inverse_spliter": lambda x: x.swapaxes(-1, -2),
        },
    }

    def __init__(
        self,
        config: Gemma4TextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        moe_linear_col = partial(
            ColumnParallelMoELinear,
            num_experts=config.num_experts,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(),
            use_bias=False,
            partition_manager=config.runtime_sharding_resolver,
            use_expert_tensor_mode=getattr(config, "use_expert_tensor_mode", False),
            dtype=dtype,
            param_dtype=param_dtype,
        )
        moe_linear_row = partial(
            RowParallelMoELinear,
            num_experts=config.num_experts,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(),
            use_bias=False,
            partition_manager=config.runtime_sharding_resolver,
            use_expert_tensor_mode=getattr(config, "use_expert_tensor_mode", False),
            dtype=dtype,
            param_dtype=param_dtype,
        )

        self.gate_proj = moe_linear_col(in_features=config.hidden_size, out_features=config.moe_intermediate_size)
        self.up_proj = moe_linear_col(in_features=config.hidden_size, out_features=config.moe_intermediate_size)
        self.down_proj = moe_linear_row(in_features=config.moe_intermediate_size, out_features=config.hidden_size)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(
        self,
        hidden_states: Array,
        group_sizes: Array,
        sorted_experts: Array | None = None,
    ) -> Array:
        """Apply the gated MLP transformation through the expert stack.

        Tokens are assumed to be pre-sorted by expert assignment. The
        ``group_sizes`` array indicates how many tokens are routed to each
        expert, enabling ``grouped_matmul`` to batch the computation
        efficiently.

        Args:
            hidden_states: Sorted token representations ``(total_tokens, hidden_size)``.
            group_sizes: Number of tokens per expert ``(num_experts,)``.
            sorted_experts: Optional expert indices aligned with tokens.

        Returns:
            Expert-processed hidden states ``(total_tokens, hidden_size)``.
        """
        gate = self.gate_proj(hidden_states, group_sizes, sorted_experts)
        up = self.up_proj(hidden_states, group_sizes, sorted_experts)
        if self.config.activations_in_float32:
            gate = gate.astype(jnp.float32)
            up = up.astype(jnp.float32)
        return self.down_proj(
            (self.act_fn(gate) * up).astype(self.dtype),
            group_sizes,
            sorted_experts,
        )


class Gemma4TextRouter(BaseMoeModule):
    """Gemma4 MoE router module with HF-compatible parameter layout.

    This module owns Gemma4's learned router state while delegating expert
    compute to an external ``Gemma4TextMLPStack``. Keeping the router and
    expert weights as sibling modules on the decoder layer preserves the
    Hugging Face checkpoint layout:

    - ``layers.*.router.norm``
    - ``layers.*.router.proj``
    - ``layers.*.router.weight``
    - ``layers.*.router.per_expert_scale``
    - ``layers.*.experts.*``

    Gemma4's router is distinctive in several ways:

    - **Pre-routing normalization**: Hidden states are RMSNorm'd (without
      learnable scale) before the router projection, unlike standard MoE
      which routes raw hidden states.
    - **Per-dimension scaling**: A learned scale vector multiplied by
      ``hidden_size^(-0.5)`` is applied before the router projection.
    - **Per-expert scaling**: After top-k selection and weight normalization,
      each selected expert's weight is multiplied by a learned per-expert
      scalar, allowing the model to learn relative expert importance.

    The router's custom logic is encapsulated in the ``reform_router_probs``
    method which is passed to the base ``moe_call`` dispatch. The actual
    token-to-expert dispatch uses EasyDeL's fused MoE implementation with
    ``grouped_matmul`` for efficient execution.

    Args:
        config: Text model configuration with ``num_experts``,
            ``top_k_experts``, ``hidden_size``, ``rms_norm_eps``, and
            ``moe_intermediate_size``.
        dtype: Computation data type.
        param_dtype: Parameter data type.
        precision: JAX numerical precision.
        rngs: Random number generator state.
    """

    def __init__(
        self,
        config: Gemma4TextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        super().__init__(
            config=config,
            n_routed_experts=config.num_experts,
            num_experts_per_tok=config.top_k_experts,
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

        self.norm = Gemma4RMSNorm(
            config=config,
            param_dtype=param_dtype,
            dim=config.hidden_size,
            with_scale=False,
        )
        self.weight = ArrayParam.bound(
            shape=(config.hidden_size,),
            dtype=param_dtype,
            init_method="ones",
            key=None,
        )
        self.router_scalar_root_size = config.hidden_size**-0.5
        self.per_expert_scale = ArrayParam.bound(
            shape=(config.num_experts,),
            dtype=param_dtype,
            init_method="ones",
            key=None,
        )

        self.proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_experts,
            use_bias=False,
            dtype=jnp.float32 if config.float32_gate_logits else dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
        )

    @staticmethod
    def preserve_router_scores(router_scores: Array) -> Array:
        """Keep raw router logits so custom selection can apply Gemma4 routing."""
        return router_scores

    def reform_router_probs(self, hidden_states: Array) -> Array:
        """Apply Gemma4's custom pre-routing transformation.

        Normalizes the hidden states, applies per-dimension scaling, then
        projects to router logits. This method is called by the base
        ``moe_call`` infrastructure in place of directly feeding hidden states
        to the gate layer.

        Args:
            hidden_states: Raw hidden states ``[batch*seq, hidden_size]``.

        Returns:
            Transformed hidden states ready for the gate projection.
        """
        hidden_states = self.norm(hidden_states)
        if self.config.float32_gate_logits:
            hidden_states = hidden_states.astype(jnp.float32)
        router_dtype = hidden_states.dtype
        return (
            hidden_states
            * self.weight.value.astype(router_dtype)
            * jnp.asarray(self.router_scalar_root_size, dtype=router_dtype)
        )

    def select_experts_with_scale(
        self,
        router_scores: Array,
        _pre_bias_logits: Array | None,
        num_experts_per_tok: int,
    ) -> tuple[Array, Array]:
        """Select top-k experts from router probabilities and apply expert scaling."""
        router_scores_f32 = router_scores.astype(jnp.float32)
        row_sums = router_scores_f32.sum(axis=-1)
        appears_normalized = jnp.logical_and(
            jnp.all(router_scores_f32 >= 0.0),
            jnp.all(jnp.abs(row_sums - 1.0) <= 1e-3),
        )
        router_probs = jax.lax.cond(
            appears_normalized,
            lambda scores: scores,
            lambda scores: jax.nn.softmax(scores, axis=-1),
            router_scores_f32,
        )
        top_k_weights, top_k_indices = jax.lax.top_k(router_probs, num_experts_per_tok)
        if num_experts_per_tok > 1:
            top_k_weights = top_k_weights / jnp.maximum(
                top_k_weights.sum(axis=-1, keepdims=True),
                jnp.finfo(top_k_weights.dtype).eps,
            )
        top_k_weights = top_k_weights * self.per_expert_scale.value[top_k_indices].astype(top_k_weights.dtype)
        return top_k_weights, top_k_indices

    def forward(
        self,
        router_hidden_states: Array,
        expert_hidden_states: Array,
        expert_layer: Gemma4TextMLPStack,
    ) -> tuple[Array, Array]:
        """Route tokens to experts, compute expert outputs, and aggregate.

        Uses EasyDeL's fused MoE dispatch (``moe_call``) with
        ``grouped_matmul`` for efficient expert computation. The custom
        router pre-processing is applied via ``before_gate`` hooks.

        Args:
            router_hidden_states: Residual stream used for routing logits.
            expert_hidden_states: Pre-normalized expert input activations.
            expert_layer: Expert MLP stack used for grouped expert execution.

        Returns:
            A tuple of:
            - Expert-aggregated output ``[batch, seq_len, hidden_size]``.
            - Router logits ``[batch*seq_len, num_experts]`` for auxiliary
              loss computation.
        """
        runtime_hooks = self.moe_hooks.replace(
            before_gate=self.reform_router_probs,
            normalize_gate_logits=self.preserve_router_scores,
            select_hook=self.select_experts_with_scale,
        )
        out, router_logits = self.moe_call(
            hidden_state=expert_hidden_states,
            gate_hidden_state=router_hidden_states,
            gate_layer=self.proj,
            expert_layer=expert_layer,
            wi_kernel=expert_layer.gate_proj.weight.value,
            wu_kernel=expert_layer.up_proj.weight.value,
            wd_kernel=expert_layer.down_proj.weight.value,
            hooks=runtime_hooks,
            act_fn=expert_layer.act_fn,
        )
        return checkpoint_name(out, "moe_expert_output"), checkpoint_name(router_logits, "moe_router_logits")


class Gemma4DecoderLayer(spx.Module):
    """Single transformer decoder layer for Gemma4.

    Implements the full Gemma4 decoder block with the following sequential
    stages, each wrapped in a residual connection:

    1. **Self-attention**: Pre-norm → attention → post-norm → residual add.
    2. **Feed-forward**: Pre-norm → MLP → (optional MoE parallel path) →
       post-norm → residual add.
    3. **Per-layer input** (optional): Gate → activation → element-wise multiply
       with per-layer embedding → project → post-norm → residual add.
    4. **Layer scalar**: Multiply the final output by a learned per-layer scalar.

    The MoE path (when ``enable_moe_block=True``) runs in parallel with the
    dense MLP: the MLP output is post-normalized independently, the MoE branch
    processes the pre-MLP residual through the router and experts with its own
    pre/post norms, and the two outputs are summed before the shared
    ``post_feedforward_layernorm``.

    Args:
        config: Text model configuration with all layer parameters.
        layer_idx: Zero-based layer index in the decoder stack.
        dtype: Computation data type.
        param_dtype: Parameter data type.
        precision: JAX numerical precision.
        rngs: Random number generator state.
    """

    def __init__(
        self,
        config: Gemma4TextConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: spx.Rngs,
    ):
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        self.self_attn = Gemma4Attention(
            config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.mlp = Gemma4MLP(
            config=config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.input_layernorm = Gemma4RMSNorm(config, param_dtype=param_dtype)
        self.post_attention_layernorm = Gemma4RMSNorm(config, param_dtype=param_dtype)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(config, param_dtype=param_dtype)
        self.post_feedforward_layernorm = Gemma4RMSNorm(config, param_dtype=param_dtype)

        self.layer_scalar = ArrayParam.bound(shape=(1,), dtype=param_dtype, init_method="ones", key=None)

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.per_layer_input_gate = ParallelLinear(
                config.hidden_size,
                self.hidden_size_per_layer_input,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            self.per_layer_projection = ParallelLinear(
                self.hidden_size_per_layer_input,
                config.hidden_size,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            self.post_per_layer_input_norm = Gemma4RMSNorm(config, param_dtype=param_dtype)
            self.per_layer_act = ACT2FN[config.hidden_activation]

        self.enable_moe_block = config.enable_moe_block
        if self.enable_moe_block:
            self.router = Gemma4TextRouter(config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)
            self.experts = Gemma4TextMLPStack(
                config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs
            )
            self.post_feedforward_layernorm_1 = Gemma4RMSNorm(config, param_dtype=param_dtype)
            self.post_feedforward_layernorm_2 = Gemma4RMSNorm(config, param_dtype=param_dtype)
            self.pre_feedforward_layernorm_2 = Gemma4RMSNorm(config, param_dtype=param_dtype)

        self.is_sliding = self.self_attn.is_sliding

    def forward(
        self,
        hidden_states: Array,
        mask_info: MaskInfo,
        position_ids: Array,
        mode: common_types.RUNTIME_MODE_TYPES,  # type: ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Array | None = None,
        default_frequencies: Array | None = None,
        per_layer_input: Array | None = None,
        shared_key_value: tuple[Array, Array] | None = None,
    ) -> DecoderLayerOutput:
        """Execute one decoder layer forward pass.

        Sliding layers use ``default_frequencies`` (local RoPE) while global
        layers use ``frequencies`` (global RoPE with proportional scaling).

        Args:
            hidden_states: Input tensor ``[batch, seq_len, hidden_size]``.
            mask_info: Attention mask configuration for this layer.
            position_ids: Position indices ``[batch, seq_len]``.
            mode: Runtime execution mode (train/decode/prefill).
            cache_view: KV cache view for this layer, or ``None`` during training.
            cache_metadata: Cache management metadata.
            output_attentions: Whether to return attention weight matrices.
            frequencies: RoPE frequencies for global attention layers.
            default_frequencies: RoPE frequencies for sliding window layers.
            per_layer_input: Per-layer embedding for this specific layer,
                shape ``[batch, seq_len, hidden_size_per_layer_input]``, or
                ``None`` if per-layer inputs are disabled.
            shared_key_value: Optional ``(key_states, value_states)`` tuple
                from the donor layer for KV sharing.

        Returns:
            ``DecoderLayerOutput`` containing the transformed hidden states,
            optional attention weights, and updated cache view.
        """
        frequencies = default_frequencies if self.is_sliding else frequencies

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )

        attn_outputs = self.self_attn(
            hidden_states,
            mask_info,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            output_attentions,
            frequencies,
            shared_key_value=shared_key_value,
        )
        hidden_states = self.post_attention_layernorm(attn_outputs.attention_output)
        hidden_states = checkpoint_name(residual + hidden_states, "residual")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        if self.config.use_scan_mlp:
            hidden_states = block_wise_ffn(self.mlp, hidden_states, self.config.scan_mlp_chunk_size)
        else:
            hidden_states = self.mlp(hidden_states)

        if self.enable_moe_block:
            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

            moe_input = self.pre_feedforward_layernorm_2(residual)
            hidden_states_2, _ = self.router(residual, moe_input, self.experts)
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

            hidden_states = hidden_states_1 + hidden_states_2

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = checkpoint_name(residual + hidden_states, "residual")

        if self.hidden_size_per_layer_input and per_layer_input is not None:
            residual = hidden_states
            hidden_states = self.per_layer_input_gate(hidden_states)
            hidden_states = self.per_layer_act(hidden_states)
            hidden_states = hidden_states * per_layer_input
            hidden_states = self.per_layer_projection(hidden_states)
            hidden_states = self.post_per_layer_input_norm(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states = hidden_states * self.layer_scalar.value
        hidden_states = checkpoint_name(hidden_states, "layer_output")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )

        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
            cache_view=attn_outputs.cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=Gemma4TextConfig, model_type="gemma4_text")
class Gemma4TextModel(EasyDeLBaseModule):
    """Gemma4 text decoder model.

    A decoder-only transformer with mixed sliding/global attention, optional
    per-layer input embeddings, KV sharing, and Mixture-of-Experts blocks.

    The model processes input tokens through the following pipeline:

    1. **Token embedding**: Lookup in the main embedding table, scaled by
       ``sqrt(hidden_size)`` following the Gemma convention.
    2. **Per-layer input preparation** (optional): A separate embedding table
       produces per-layer residual signals. These are combined with a learned
       projection of the main embeddings via addition and ``sqrt(2)`` scaling.
    3. **Decoder layers**: Each layer applies self-attention (sliding or global
       based on ``layer_types``), feed-forward (with optional MoE), and
       per-layer input injection, all with residual connections and RMSNorm.
    4. **Final normalization**: RMSNorm applied to the output hidden states.

    For multimodal inputs (when used inside ``Gemma4Model``), the model accepts
    ``token_type_ids`` to construct bidirectional attention masks for vision
    token blocks, following the same masking strategy as Gemma3.

    Args:
        config: ``Gemma4TextConfig`` with all model hyperparameters.
        dtype: Computation data type. Defaults to bfloat16.
        param_dtype: Parameter storage data type. Defaults to bfloat16.
        precision: JAX precision setting for matrix operations.
        rngs: SpecTrax random number generator state.
    """

    def __init__(
        self,
        config: Gemma4TextConfig,
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
        self.hidden_size = config.hidden_size

        self.embed_tokens = Embed(
            config.vocab_size,
            self.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        remat_layer_block = auto_remat(
            Gemma4DecoderLayer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.layers = nn.ModuleList([])
        for i in range(config.num_hidden_layers):
            with self.assign_layer_stage(i, total_layers=config.num_hidden_layers):
                self.layers.append(
                    remat_layer_block(
                        config,
                        layer_idx=i,
                        dtype=dtype,
                        param_dtype=param_dtype,
                        precision=precision,
                        rngs=rngs,
                    )
                )
        final_layer_idx = max(0, config.num_hidden_layers - 1)
        with self.assign_layer_stage(final_layer_idx, total_layers=config.num_hidden_layers):
            self.norm = Gemma4RMSNorm(config, param_dtype=param_dtype)

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = Embed(
                config.vocab_size_per_layer_input,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            self.per_layer_input_scale = 2.0**-0.5
            self.per_layer_model_projection = ParallelLinear(
                config.hidden_size,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            self.per_layer_model_projection_scale = config.hidden_size**-0.5
            self.per_layer_projection_norm = Gemma4RMSNorm(
                config,
                param_dtype=param_dtype,
                dim=config.hidden_size_per_layer_input,
            )

        # Keep the cached layer-type summary hashable so split graphdefs can
        # flow through static JAX/SpecTrax compilation paths.
        self.unique_layer_types = tuple(dict.fromkeys(config.layer_types))

    def get_per_layer_inputs(self, input_ids: Array) -> Array:
        """Look up per-layer input embeddings from the dedicated embedding table.

        Each token ID maps to a vector of size
        ``num_hidden_layers * hidden_size_per_layer_input``, which is reshaped
        into per-layer slices. The embeddings are scaled by
        ``sqrt(hidden_size_per_layer_input)`` following the Gemma convention.

        Args:
            input_ids: Token IDs of shape ``(batch_size, sequence_length)``.

        Returns:
            Per-layer embeddings of shape
            ``(batch, seq_len, num_hidden_layers, hidden_size_per_layer_input)``.
        """
        per_layer_embs = self.embed_tokens_per_layer(input_ids.astype("i4")) * (self.hidden_size_per_layer_input**0.5)
        return per_layer_embs.reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(self, inputs_embeds: Array, per_layer_inputs: Array | None = None) -> Array:
        """Project main embeddings to per-layer space and combine with token-level per-layer embeddings.

        The main embedding is linearly projected to produce a per-layer signal,
        then normalized. If per-layer token embeddings are provided, they are
        added and the sum is scaled by ``1/sqrt(2)`` to maintain variance.

        Args:
            inputs_embeds: Main token embeddings of shape
                ``(batch, seq_len, hidden_size)``.
            per_layer_inputs: Optional per-layer token embeddings from
                ``get_per_layer_inputs``, shape
                ``(batch, seq_len, num_layers, hidden_size_per_layer_input)``.

        Returns:
            Combined per-layer inputs of shape
            ``(batch, seq_len, num_layers, hidden_size_per_layer_input)``.
        """
        per_layer_projection = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None:
            return per_layer_projection
        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale

    @cached_property
    def default_frequencies(self):
        """Precomputed RoPE frequencies for sliding window (local) attention layers.

        Uses the ``sliding_attention`` rope parameters from the config, typically
        with ``rope_theta=10000`` and the standard ``head_dim``.

        Returns:
            ``ModuleCaches`` wrapping the frequency tensor of shape
            ``(max_position, head_dim // 2)``.
        """
        from easydel.infra.utils import ModuleCaches
        from easydel.layers import get_frequencies

        local_params = self.config.rope_parameters.get("sliding_attention", {})
        base = local_params.get("rope_theta", 10_000.0)

        frequencies = get_frequencies(
            head_size=self.config.head_dim,
            rotary_dim=self.config.head_dim,
            max_position=self.config.granted_freq_max_position_embedding,
            base=base,
            rope_scaling=None,
        )
        return ModuleCaches(frequencies)

    @cached_property
    def global_frequencies(self):
        """Precomputed RoPE frequencies for global (full) attention layers.

        Uses the ``full_attention`` rope parameters from the config, typically
        with ``rope_theta=1_000_000``, ``partial_rotary_factor=0.25``, and
        ``global_head_dim`` (which may differ from the standard ``head_dim``).

        The ``partial_rotary_factor`` controls what fraction of the head
        dimension receives rotary embeddings. For example, with
        ``global_head_dim=512`` and ``partial_rotary_factor=0.25``, only the
        first 128 dimensions of each head use RoPE, while the remaining 384
        dimensions receive no positional encoding.

        Returns:
            ``ModuleCaches`` wrapping the frequency tensor of shape
            ``(max_position, rotary_dim // 2)``.
        """
        from easydel.infra.utils import ModuleCaches
        from easydel.layers import get_frequencies

        global_params = self.config.rope_parameters.get("full_attention", {})
        base = global_params.get("rope_theta", 1_000_000.0)
        partial_rotary = global_params.get("partial_rotary_factor", 1.0)
        rope_type = global_params.get("rope_type", "default")
        head_dim = self.config.global_head_dim if self.config.global_head_dim else self.config.head_dim

        if rope_type == "proportional":
            rope_angles = int(partial_rotary * head_dim // 2)
            inv_freq_rotated = 1.0 / (base ** (jnp.arange(0, 2 * rope_angles, 2, dtype=jnp.float32) / head_dim))
            nope_angles = head_dim // 2 - rope_angles
            inv_freq = (
                jnp.concatenate((inv_freq_rotated, jnp.zeros((nope_angles,), dtype=jnp.float32)), axis=0)
                if nope_angles > 0
                else inv_freq_rotated
            )
            positions = jnp.arange(self.config.granted_freq_max_position_embedding, dtype=jnp.float32)[:, None]
            phase = positions * inv_freq[None, :]
            frequencies = jnp.concatenate((jnp.cos(phase), jnp.sin(phase)), axis=-1)
        elif partial_rotary < 1.0:
            rotated_dim = int(head_dim * partial_rotary)
            rotated_frequencies = get_frequencies(
                head_size=rotated_dim,
                rotary_dim=rotated_dim,
                max_position=self.config.granted_freq_max_position_embedding,
                base=base,
                rope_scaling=None,
                partial_rotary_factor=1.0,
            )
            rotated_cos, rotated_sin = jnp.split(rotated_frequencies, 2, axis=-1)
            pass_dim = head_dim // 2 - rotated_cos.shape[-1]
            frequencies = jnp.concatenate(
                (
                    jnp.concatenate(
                        (rotated_cos, jnp.ones((rotated_cos.shape[0], pass_dim), dtype=rotated_cos.dtype)), axis=-1
                    ),
                    jnp.concatenate(
                        (rotated_sin, jnp.zeros((rotated_sin.shape[0], pass_dim), dtype=rotated_sin.dtype)), axis=-1
                    ),
                ),
                axis=-1,
            )
        else:
            frequencies = get_frequencies(
                head_size=head_dim,
                rotary_dim=head_dim,
                max_position=self.config.granted_freq_max_position_embedding,
                base=base,
                rope_scaling=None,
            )
        return ModuleCaches(frequencies)

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        token_type_ids: Array | None = None,
        per_layer_inputs: Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
    ) -> BaseModelOutput:
        """Forward pass through the Gemma4 text decoder.

        Args:
            input_ids: Input token IDs ``[batch, seq_len]``. Exactly one of
                ``input_ids`` or ``inputs_embeds`` must be provided.
            inputs_embeds: Pre-computed input embeddings
                ``[batch, seq_len, hidden_size]``.
            attention_mask: Boolean padding mask ``[batch, seq_len]``.
            mask_info: Pre-computed mask information. If ``None``, constructed
                automatically from ``attention_mask``.
            position_ids: Explicit position indices ``[batch, seq_len]``.
            token_type_ids: Token type indicators for multimodal masking. Values
                of 1 or 2 indicate vision tokens which receive bidirectional
                attention within each contiguous vision block.
            per_layer_inputs: Pre-computed per-layer embeddings
                ``[batch, seq_len, num_layers, hidden_size_per_layer_input]``.
                If ``None`` and ``hidden_size_per_layer_input > 0``, computed
                automatically from ``input_ids``.
            output_attentions: Whether to return attention weights from all layers.
            output_hidden_states: Whether to return hidden states from all layers
                (including the final normalized output).
            mode: Runtime execution mode. Auto-detected if ``None``.
            past_key_values: Cached key-value states for autoregressive generation.
            cache_metadata: Metadata for cache management operations.

        Returns:
            ``BaseModelOutput`` with ``last_hidden_state``, optional
            ``hidden_states`` tuple, optional ``attentions`` tuple, and
            ``past_key_values``.

        Raises:
            ValueError: If both or neither of ``input_ids`` and ``inputs_embeds``
                are provided.
            AssertionError: If the sequence length exceeds
                ``max_position_embeddings``.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = checkpoint_name(self.embed_tokens(input_ids.astype("i4")), "embeddings") * (
                self.config.hidden_size**0.5
            )

        if self.hidden_size_per_layer_input:
            if per_layer_inputs is None and input_ids is not None:
                per_layer_inputs = self.get_per_layer_inputs(input_ids)
            per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        sequence_length = inputs_embeds.shape[1]

        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        mask_info_full = mask_info
        mask_info_sliding = mask_info

        if token_type_ids is not None:
            token_type_ids = jnp.asarray(token_type_ids, dtype=jnp.int32)
            is_vision = (token_type_ids == 1) | (token_type_ids == 2)
            prev_is_vision = jnp.pad(is_vision, ((0, 0), (1, 0)), constant_values=False)[:, :-1]
            new_vision_start = is_vision & (~prev_is_vision)
            vision_group_ids = jnp.cumsum(new_vision_start.astype(jnp.int32), axis=1) - 1
            grouped_token_types = jnp.where(is_vision, vision_group_ids + 1, 0).astype(jnp.int32)

            causal_mask_info = mask_info.apply_causal()
            mask_info_full = causal_mask_info.apply_token_type_ids(grouped_token_types)
            mask_info_sliding = causal_mask_info.apply_sliding_window(
                Gemma4Attention._hf_sliding_window(self.config.sliding_window)
            ).apply_token_type_ids(grouped_token_types)
            object.__setattr__(mask_info_full, "_causal_baked", True)
            object.__setattr__(mask_info_sliding, "_causal_baked", True)

        if position_ids is None:
            position_ids = mask_info.q_position_ids

        assert sequence_length <= self.config.max_position_embeddings

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
            partition_manager=self.config.runtime_sharding_resolver,
        )
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # KV sharing: shared layers reuse K/V from the last non-shared layer
        # of the same attention type instead of computing their own projections.
        # For cached inference, shared layers also use the donor's cache view
        # (their own views[idx] is None to avoid duplicate buffer donation).
        shared_kv: dict[int, tuple[Array, Array]] = {}
        donor_cache_views: dict[int, typing.Any] = {}

        def _layer_loop(block, carry):
            hidden_states, all_hidden_states, all_attentions, idx = carry
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_mask_info = (
                mask_info_sliding if self.config.layer_types[idx] == "sliding_attention" else mask_info_full
            )
            per_layer_input = per_layer_inputs[:, :, idx, :] if per_layer_inputs is not None else None

            attn = block.self_attn
            shared_key_value = shared_kv.get(attn.kv_shared_layer_index) if attn.is_kv_shared_layer else None

            # Shared layers use the donor's cache view; their own slot is None.
            cache_view = self._layer_cache_view_at(None, idx, enabled=True, cache=past_key_values)
            if cache_view is None and attn.is_kv_shared_layer:
                cache_view = donor_cache_views.get(attn.kv_shared_layer_index)

            with self._layer_stage_context(idx, layers=self.layers):
                layer_outputs = block(
                    hidden_states=hidden_states,
                    mask_info=layer_mask_info,
                    position_ids=position_ids,
                    mode=mode,
                    cache_view=cache_view,
                    cache_metadata=cache_metadata,
                    output_attentions=output_attentions,
                    frequencies=self.global_frequencies,
                    default_frequencies=self.default_frequencies,
                    per_layer_input=per_layer_input,
                    shared_key_value=shared_key_value,
                )
            hidden_states = self._mark_layer_stage_boundary(layer_outputs.hidden_states, idx, layers=self.layers)

            # Store captured K/V for potential downstream sharing.
            captured = getattr(attn, "_captured_kv", None)
            if captured is not None and not attn.is_kv_shared_layer:
                shared_kv[idx] = captured
                object.__setattr__(attn, "_captured_kv", None)
                # Track the donor's (potentially updated) cache view.
                self._layer_cache_view_update(donor_cache_views, idx, layer_outputs.cache_view, enabled=True)

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)
            if not attn.is_kv_shared_layer:
                self._layer_cache_view_update(None, idx, layer_outputs.cache_view, enabled=True, cache=past_key_values)

            return hidden_states, all_hidden_states, all_attentions, idx + 1

        hidden_states, all_hidden_states, all_attentions, _ = self.layers.scan(
            _layer_loop,
            (hidden_states, all_hidden_states, all_attentions, 0),
            trace=True,
        )
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
        """Not applicable for decoder-only models."""
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        """Return the decoder (i.e., this model itself)."""
        return self

    def get_lm_head(self):
        """Not applicable for the base model without a language modeling head."""
        raise NotImplementedError("The base model does not have a language model head.")

    def get_embedding(self):
        """Return the token embedding layer."""
        return self.embed_tokens


@register_module(TaskType.CAUSAL_LM, config=Gemma4TextConfig, model_type="gemma4_text")
class Gemma4ForCausalLM(BaseCausalLMModule[Gemma4TextModel, Gemma4TextConfig]):
    """Gemma4 model with a language modeling head for causal (autoregressive) generation.

    Wraps ``Gemma4TextModel`` with a tied linear projection from the final hidden
    states to vocabulary logits. Supports optional logit soft-capping via
    ``config.final_logit_softcapping``, which smoothly bounds logit magnitudes to
    ``[-cap, cap]`` using ``cap * tanh(logits / cap)``.

    Args:
        config: ``Gemma4TextConfig`` with all model hyperparameters.
        dtype: Computation data type. Defaults to bfloat16.
        param_dtype: Parameter data type. Defaults to bfloat16.
        precision: JAX precision setting for matrix operations.
        rngs: SpecTrax random number generator state.
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "gemma4_text"
    _config_class = Gemma4TextConfig

    def __init__(
        self,
        config: Gemma4TextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        super().__init__(
            config=config,
            base_model_class=Gemma4TextModel,
            base_model_name="model",
            tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
        )

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        token_type_ids: Array | None = None,
        per_layer_inputs: Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> CausalLMOutput:
        """Forward pass through the causal language model.

        Runs the base text model and optionally applies the language modeling
        head to project hidden states to vocabulary logits.

        Args:
            input_ids: Input token IDs ``[batch, seq_len]``.
            inputs_embeds: Pre-computed embeddings ``[batch, seq_len, hidden_size]``.
            attention_mask: Padding mask ``[batch, seq_len]``.
            mask_info: Pre-computed attention mask information.
            position_ids: Position indices ``[batch, seq_len]``.
            token_type_ids: Vision/text token type indicators for multimodal masking.
            per_layer_inputs: Pre-computed per-layer embeddings.
            output_attentions: Whether to return attention weight matrices.
            output_hidden_states: Whether to return intermediate hidden states.
            mode: Runtime execution mode (auto-detected if ``None``).
            past_key_values: Cached KV states for generation.
            cache_metadata: Cache management metadata.
            apply_lm_head: Whether to compute vocabulary logits. Set to ``False``
                to return only hidden states (e.g., for feature extraction).

        Returns:
            ``CausalLMOutput`` with ``logits`` (or ``None`` if
            ``apply_lm_head=False``), ``last_hidden_state``, and optional
            ``hidden_states``, ``attentions``, and ``past_key_values``.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            inputs_embeds=inputs_embeds,
            token_type_ids=token_type_ids,
            per_layer_inputs=per_layer_inputs,
        )

        lm_logits = None
        if apply_lm_head:
            lm_logits = self.compute_lm_logits(self.prepare_lm_head_inputs(outputs.last_hidden_state))

        return CausalLMOutput(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            last_hidden_state=outputs.last_hidden_state,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
        )

    def compute_lm_logits(self, hidden_states: Array) -> Array:
        """Project hidden states to vocabulary logits with optional soft-capping.

        Calls the base LM-head projection, then applies Gemma4's logit
        soft-capping when ``config.final_logit_softcapping`` is set:
        ``cap * tanh(logits / cap)``, which smoothly bounds logit magnitudes
        to ``[-cap, cap]``.

        Args:
            hidden_states: Final hidden representations ``[batch, seq_len, hidden_size]``.

        Returns:
            Vocabulary logits ``[batch, seq_len, vocab_size]``, optionally soft-capped.
        """
        return self.apply_lm_head(hidden_states)

    def apply_lm_head(self, hidden_states: Array) -> Array:
        """Project hidden states to Gemma4 text logits for runtime sampling paths."""
        if getattr(self.config, "tie_word_embeddings", False):
            # Gemma4 checkpoints tie the text LM head to the input embedding
            # matrix. Going through the generic ColumnParallelLinear `w=...`
            # override can drift under TPU TP layouts, so use the embedding's
            # native attend path directly for tied-text logits.
            lm_logits = self.get_embedding().attend(hidden_states)
        else:
            lm_logits = super().apply_lm_head(hidden_states)
        if self.config.final_logit_softcapping is not None:
            cap = jnp.array(self.config.final_logit_softcapping, dtype=lm_logits.dtype)
            lm_logits = cap * jax.nn.tanh(lm_logits / cap)
        return lm_logits

    def make_lm_head_fn(self):
        """Trace-safe projection preserving Gemma-4 tied-embedding attend path and soft-capping."""
        cap_value = self.config.final_logit_softcapping
        if getattr(self.config, "tie_word_embeddings", False):
            # Embed.attend only reads .value — already trace-safe.
            _attend = self.get_embedding().attend

            def _project(hidden_states):
                lm_logits = _attend(hidden_states)
                if cap_value is not None:
                    cap = jnp.array(cap_value, dtype=lm_logits.dtype)
                    lm_logits = cap * jax.nn.tanh(lm_logits / cap)
                return lm_logits

        else:
            # Untied: delegate to base (native_forward bypass) + add capping.
            base_fn = super().make_lm_head_fn()
            if cap_value is None:
                return base_fn

            def _project(hidden_states):
                logits = base_fn(hidden_states)
                cap = jnp.array(cap_value, dtype=logits.dtype)
                return cap * jax.nn.tanh(logits / cap)

        return _project

    def get_encoder(self):
        """Not applicable for decoder-only models."""
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        """Return the decoder component of the model."""
        return self.model.get_decoder()

    def get_lm_head(self):
        """Return the language modeling head projection layer."""
        return self.lm_head

    def get_embedding(self):
        """Return the token embedding layer from the base model."""
        return self.model.get_embedding()


class Gemma4MultimodalEmbedder(spx.Module):
    """Projects multimodal (vision or audio) features into the language model's embedding space.

    Applies scale-free RMSNorm followed by a linear projection to transform
    features from the multimodal encoder's hidden dimension to the text model's
    hidden dimension. This replaces Gemma3's average-pooling projector with a
    simpler normalize-then-project approach.

    Args:
        multimodal_hidden_size: Hidden dimension of the multimodal encoder output.
        text_hidden_size: Hidden dimension of the language model.
        rms_norm_eps: Epsilon for the pre-projection RMSNorm.
        dtype: Computation data type.
        param_dtype: Parameter data type.
        rngs: Random number generator state.
    """

    def __init__(
        self,
        multimodal_hidden_size: int,
        text_hidden_size: int,
        rms_norm_eps: float = 1e-6,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: spx.Rngs,
    ):
        self.embedding_pre_projection_norm = Gemma4RMSNorm(
            dim=multimodal_hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=param_dtype,
            with_scale=False,
        )
        kernel_init = jax.nn.initializers.normal(0.02)
        self.embedding_projection = ParallelLinear(
            multimodal_hidden_size,
            text_hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            rngs=rngs,
        )

    def forward(self, inputs_embeds: Array) -> Array:
        """Normalize and project multimodal features to text embedding space.

        Args:
            inputs_embeds: Multimodal features of shape
                ``(..., multimodal_hidden_size)``.

        Returns:
            Projected features of shape ``(..., text_hidden_size)``.
        """
        return self.embedding_projection(self.embedding_pre_projection_norm(inputs_embeds))


@register_module(TaskType.BASE_MODULE, config=Gemma4Config, model_type="gemma4")
class Gemma4Model(EasyDeLBaseModule):
    """Gemma4 multimodal model combining a vision encoder with a language decoder.

    Integrates an optional vision tower (loaded via ``AutoEasyDeLVisionModel``)
    with the ``Gemma4TextModel`` language backbone through a
    ``Gemma4MultimodalEmbedder`` that projects vision features into the text
    embedding space. Image features are inserted at ``image_token_id``
    placeholder positions in the token sequence using a cumsum-based gathering
    algorithm.

    When per-layer input embeddings are enabled, they are computed from the
    original ``input_ids`` before vision token merging (since the original token
    IDs are not recoverable after soft token insertion). Multimodal token
    positions use the padding token ID for the per-layer embedding lookup.

    Args:
        config: ``Gemma4Config`` containing text, vision, and multimodal settings.
        dtype: Computation data type.
        param_dtype: Parameter data type.
        precision: JAX precision setting.
        rngs: Random number generator state.
    """

    def __init__(
        self,
        config: Gemma4Config,
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

        self.language_model = Gemma4TextModel(
            config=config.text_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self._missing_vision_backend_model_type: str | None = None

        if config.vision_config is not None and _has_registered_gemma4_vision_backend(config.vision_config):
            self.vision_tower = AutoEasyDeLVisionModel.from_config(
                config.vision_config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            self.embed_vision = Gemma4MultimodalEmbedder(
                multimodal_hidden_size=config.vision_config.hidden_size,
                text_hidden_size=config.text_config.hidden_size,
                rms_norm_eps=config.vision_config.rms_norm_eps,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
        else:
            if config.vision_config is not None:
                self._missing_vision_backend_model_type = config.vision_config.model_type
            self.vision_tower = None
            self.embed_vision = None

    def _require_vision_tower(self) -> None:
        """Validate that a usable vision encoder is available for image inputs."""
        if self.vision_tower is not None:
            return
        if self.config.vision_config is None:
            raise ValueError("Model was initialized without a vision config.")
        raise NotImplementedError(
            "Gemma4 image inputs require a registered BASE_VISION backend for "
            f"`{self._missing_vision_backend_model_type or self.config.vision_config.model_type}`."
        )

    def compute_embedding(
        self,
        input_ids: Array,
        pixel_values: Array | None = None,
        image_position_ids: Array | None = None,
    ) -> Array:
        """Compute text embeddings and merge vision features at placeholder positions.

        Token embeddings are looked up and scaled by ``sqrt(hidden_size)``.
        If ``pixel_values`` are provided, the vision tower encodes them into
        features which are projected into the text embedding space and inserted
        at positions where ``input_ids == config.image_token_id``.

        The merging uses a cumsum-based algorithm: a boolean mask identifies
        image token positions, and ``cumsum`` over that mask creates gather
        indices that map each placeholder to the corresponding vision feature.

        Args:
            input_ids: Token IDs ``[batch, seq_len]``.
            pixel_values: Image pixel values for the vision encoder, or ``None``
                for text-only inputs.

        Returns:
            Merged embeddings ``[batch, seq_len, hidden_size]`` with vision
            features at image token positions and text embeddings elsewhere.
        """
        inputs_embeds = self.language_model.embed_tokens(input_ids.astype("i4")) * (
            self.config.text_config.hidden_size**0.5
        )

        if pixel_values is not None:
            self._require_vision_tower()
            vision_outputs = self.vision_tower(
                pixel_values=pixel_values,
                pixel_position_ids=image_position_ids,
            )
            image_features = self.embed_vision(vision_outputs.last_hidden_state)
            image_features = image_features.astype(inputs_embeds.dtype)

            special_image_mask = (input_ids == self.config.image_token_id).astype(jnp.int32)
            image_features_flat = image_features.reshape(-1, image_features.shape[-1])

            cumsum = jnp.cumsum(special_image_mask.reshape(-1), axis=0)
            gather_indices = jnp.where(
                special_image_mask.reshape(-1),
                cumsum - 1,
                0,
            )
            image_features_at_pos = jnp.where(
                special_image_mask.reshape(-1)[:, None],
                image_features_flat[gather_indices],
                jnp.zeros((1, image_features.shape[-1]), dtype=inputs_embeds.dtype),
            )
            inputs_embeds = jnp.where(
                special_image_mask[:, :, None],
                image_features_at_pos.reshape(inputs_embeds.shape),
                inputs_embeds,
            )

        return inputs_embeds

    def _compute_per_layer_inputs(self, input_ids: Array | None) -> Array | None:
        """Build per-layer inputs while masking multimodal placeholder tokens."""
        if not self.language_model.hidden_size_per_layer_input or input_ids is None:
            return None

        multimodal_mask = input_ids == self.config.image_token_id
        if self.config.video_token_id is not None:
            multimodal_mask = multimodal_mask | (input_ids == self.config.video_token_id)
        safe_ids = jnp.where(multimodal_mask, self.config.text_config.pad_token_id, input_ids)
        return self.language_model.get_per_layer_inputs(safe_ids)

    def compute_embedding_with_info(
        self,
        input_ids: Array,
        pixel_values: Array | None = None,
        image_position_ids: Array | None = None,
        **_kwargs,
    ) -> tuple[Array, EmbeddingInfo | None]:
        """Compute multimodal embeddings and preserve auxiliary per-layer inputs."""
        inputs_embeds = self.compute_embedding(input_ids, pixel_values, image_position_ids=image_position_ids)
        per_layer_inputs = self._compute_per_layer_inputs(input_ids)
        if per_layer_inputs is None:
            return inputs_embeds, None
        return inputs_embeds, EmbeddingInfo(per_layer_inputs=per_layer_inputs)

    def forward(
        self,
        input_ids: Array | None = None,
        pixel_values: Array | None = None,
        image_position_ids: Array | None = None,
        attention_mask: Array | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Array | None = None,
        token_type_ids: Array | None = None,
        per_layer_inputs: Array | None = None,
        inputs_embeds: Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
    ) -> BaseModelOutput:
        """Forward pass through the multimodal model.

        If ``inputs_embeds`` is not provided, computes text+vision merged
        embeddings via ``compute_embedding``. Per-layer inputs are computed
        from the original ``input_ids`` with multimodal positions replaced by
        the padding token ID.

        Args:
            input_ids: Token IDs ``[batch, seq_len]``.
            pixel_values: Image data for the vision encoder.
            attention_mask: Padding mask ``[batch, seq_len]``.
            mask_info: Pre-computed attention mask information.
            position_ids: Position indices ``[batch, seq_len]``.
            token_type_ids: Vision/text token type indicators.
            per_layer_inputs: Optional pre-computed per-layer embeddings for the
                text decoder.
            inputs_embeds: Pre-computed embeddings (skips embedding computation).
            output_attentions: Return attention weights from all layers.
            output_hidden_states: Return hidden states from all layers.
            mode: Runtime execution mode.
            past_key_values: Cached KV states.
            cache_metadata: Cache management metadata.

        Returns:
            ``BaseModelOutput`` from the language model.
        """
        if inputs_embeds is None:
            inputs_embeds = self.compute_embedding(
                input_ids,
                pixel_values,
                image_position_ids=image_position_ids,
            )

        if per_layer_inputs is None:
            per_layer_inputs = self._compute_per_layer_inputs(input_ids)

        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            per_layer_inputs=per_layer_inputs,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
        )

    def get_encoder(self):
        """Not applicable."""
        raise NotImplementedError()

    def get_decoder(self):
        """Return the language model decoder."""
        return self.language_model

    def get_lm_head(self):
        """Not applicable for the base multimodal model."""
        raise NotImplementedError()

    def get_embedding(self):
        """Return the token embedding layer from the language model."""
        return self.language_model.get_embedding()


@register_module(TaskType.IMAGE_TEXT_TO_TEXT, config=Gemma4Config, model_type="gemma4")
class Gemma4ForConditionalGeneration(BaseVisionLanguageModule[Gemma4Model, Gemma4Config]):
    """Gemma4 vision-language model for conditional text generation from images.

    Combines the ``Gemma4Model`` multimodal backbone with a language modeling
    head for tasks like image captioning, visual question answering, and
    multimodal chat. Supports optional logit soft-capping inherited from the
    text configuration.

    During generation, ``pixel_values`` and ``token_type_ids`` are automatically
    removed from model inputs after the first forward pass (prefill), since
    vision features are cached in the KV states.

    Args:
        config: ``Gemma4Config`` with text, vision, and multimodal settings.
        dtype: Computation data type. Defaults to bfloat16.
        param_dtype: Parameter data type. Defaults to bfloat16.
        precision: JAX precision setting.
        rngs: Random number generator state.
    """

    _supports_video = False
    _uses_mrope = False
    _vision_tower_name = "vision_tower"
    _projector_name = "embed_vision"
    _language_model_name = "language_model"

    def __init__(
        self,
        config: Gemma4Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        super().__init__(
            config=config,
            base_model_class=Gemma4Model,
            base_model_name="model",
            tie_word_embeddings=getattr(config.text_config, "tie_word_embeddings", False),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def get_image_features(
        self,
        pixel_values: Array,
        image_position_ids: Array | None = None,
    ) -> Array:
        """Extract and project image features from pixel values.

        Runs the vision tower encoder followed by the multimodal embedder to
        produce image features in the language model's embedding space.

        Args:
            pixel_values: Image data accepted by the vision encoder.

        Returns:
            Projected image features ready for merging into the text sequence.

        Raises:
            ValueError: If the model was initialized without a vision config.
        """
        self.base_model._require_vision_tower()
        vision_outputs = self.base_model.vision_tower(
            pixel_values=pixel_values,
            pixel_position_ids=image_position_ids,
        )
        return self.base_model.embed_vision(vision_outputs.last_hidden_state)

    def compute_embedding(
        self,
        input_ids: Array,
        pixel_values: Array | None = None,
        image_position_ids: Array | None = None,
    ) -> Array:
        """Delegate to the base model's embedding computation with vision merging.

        Args:
            input_ids: Token IDs ``[batch, seq_len]``.
            pixel_values: Optional image data for the vision encoder.

        Returns:
            Merged text+vision embeddings ``[batch, seq_len, hidden_size]``.
        """
        return self.base_model.compute_embedding(
            input_ids,
            pixel_values,
            image_position_ids=image_position_ids,
        )

    def compute_embedding_with_info(
        self,
        input_ids: Array,
        pixel_values: Array | None = None,
        image_position_ids: Array | None = None,
        **kwargs,
    ) -> tuple[Array, EmbeddingInfo | None]:
        """Delegate multimodal embedding computation and auxiliary info to the base model."""
        return self.base_model.compute_embedding_with_info(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
            **kwargs,
        )

    def apply_lm_head(self, hidden_states: Array) -> Array:
        """Project hidden states to vocabulary logits with optional soft-capping.

        Args:
            hidden_states: Final hidden representations
                ``[batch, seq_len, hidden_size]``.

        Returns:
            Vocabulary logits ``[batch, seq_len, vocab_size]``.
        """
        logits = super().apply_lm_head(hidden_states)
        cap = getattr(self.config.text_config, "final_logit_softcapping", None)
        if cap is not None:
            cap = jnp.array(cap, dtype=logits.dtype)
            logits = cap * jax.nn.tanh(logits / cap)
        return logits

    def make_lm_head_fn(self):
        """Trace-safe projection with Gemma-4 VLM soft-capping."""
        base_fn = super().make_lm_head_fn()
        cap_value = getattr(self.config.text_config, "final_logit_softcapping", None)
        if cap_value is None:
            return base_fn

        def _project(hidden_states):
            logits = base_fn(hidden_states)
            cap = jnp.array(cap_value, dtype=logits.dtype)
            return cap * jax.nn.tanh(logits / cap)

        return _project

    def forward(
        self,
        input_ids: Array | None = None,
        pixel_values: Array | None = None,
        image_position_ids: Array | None = None,
        attention_mask: Array | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Array | None = None,
        token_type_ids: Array | None = None,
        per_layer_inputs: Array | None = None,
        inputs_embeds: Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> VLMCausalLMOutput:
        """Forward pass through the vision-language model.

        Runs the multimodal base model (vision encoding + text decoding) and
        optionally applies the LM head to produce vocabulary logits.

        Args:
            input_ids: Token IDs ``[batch, seq_len]``.
            pixel_values: Image data for the vision encoder.
            attention_mask: Padding mask ``[batch, seq_len]``.
            mask_info: Pre-computed attention mask information.
            position_ids: Position indices ``[batch, seq_len]``.
            token_type_ids: Vision/text indicators for bidirectional masking.
            inputs_embeds: Pre-computed embeddings (skips vision+text merge).
            output_attentions: Return attention weights.
            output_hidden_states: Return intermediate hidden states.
            mode: Runtime execution mode.
            past_key_values: Cached KV states.
            cache_metadata: Cache management metadata.
            apply_lm_head: Whether to compute vocabulary logits.

        Returns:
            ``VLMCausalLMOutput`` with ``logits``, ``last_hidden_state``, and
            optional ``hidden_states``, ``attentions``, ``past_key_values``.
        """
        outputs = self.base_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            per_layer_inputs=per_layer_inputs,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
        )

        lm_logits = None
        if apply_lm_head:
            lm_logits = self.apply_lm_head(outputs.last_hidden_state)

        return VLMCausalLMOutput(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            last_hidden_state=outputs.last_hidden_state,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
        )

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        """Strip vision-specific inputs after the first generation step.

        After the initial prefill, vision features are cached in the KV states
        and should not be re-processed. This removes ``pixel_values``,
        ``token_type_ids``, and prompt-length auxiliary multimodal inputs such
        as ``per_layer_inputs`` from the generation kwargs.

        Args:
            model_outputs: Outputs from the previous generation step.
            model_kwargs: Current generation keyword arguments.

        Returns:
            Updated kwargs with vision inputs removed.
        """
        model_kwargs = super().update_inputs_for_generation(model_outputs, model_kwargs)
        model_kwargs.pop("pixel_values", None)
        model_kwargs.pop("image_position_ids", None)
        model_kwargs.pop("token_type_ids", None)
        model_kwargs.pop("per_layer_inputs", None)
        return model_kwargs
