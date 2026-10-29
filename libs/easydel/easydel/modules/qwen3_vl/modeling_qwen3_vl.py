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

"""Qwen3-VL multimodal model implementation for EasyDeL.

This module implements Alibaba's Qwen3-VL family: a Qwen3 language
backbone paired with a vision tower derived from the Qwen2-VL design.
The vision encoder uses windowed/full attention over patch tokens and
2-D rotary position embeddings; the language model uses 3-D rotary
position embeddings (mRoPE) so that interleaved image/video and text
tokens can share a single RoPE schedule.

Exposes the vision tower, the language trunk
:class:`Qwen3VLModel`, and the multimodal wrapper
:class:`Qwen3VLForConditionalGeneration`, which merges projected visual
features into placeholder positions in the input token sequence and
forwards through the LM head.
"""

import math
from functools import cached_property

import jax
import jax.numpy as jnp
import numpy as np
import spectrax as spx
from eformer.pytree import auto_pytree
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int
from spectrax import apply_logical_sharding, common_types, nn
from spectrax.functional import conv

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
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import (
    BaseModelOutput,
    DecoderLayerOutput,
    EmbeddingInfo,
    ModelOutput,
    VLMCausalLMOutput,
)
from easydel.infra.sequence_packing import token_attention_mask_from_mask_info
from easydel.infra.utils import ACT2FN, auto_remat, blockwise_ffn
from easydel.layers import (
    ColumnParallelLinear,
    Embed,
    RMSNorm,
    RowParallelLinear,
    dense_gate_up_layout,
    gated_mlp_forward,
)
from easydel.layers.attention import FlexibleAttentionModule, UnifiedAttention, block_diagonal_bias
from easydel.layers.norms import LayerNorm
from easydel.layers.rotary import (
    apply_rope_split_half,
    vision_grid_pos_embed_interpolate,
    vision_grid_rotary_embeddings,
)
from easydel.modules._base import BaseVisionLanguageModule, DeepStackVisualMixin, QwenVLMultimodalMixin
from easydel.modules.qwen2_vl.modeling_qwen2_vl import get_rope_index as qwen2_vl_get_rope_index

from .qwen3_vl_configuration import Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig


@auto_pytree
class Qwen3VLCausalLMOutputWithPast(VLMCausalLMOutput):
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

    loss: Array | None = None
    logits: Array = None
    past_key_values: list[Array] | None = None
    hidden_states: tuple[Array] | None = None
    attentions: tuple[Array] | None = None
    rope_deltas: Array | None = None
    image_hidden_states: Float[Array, "batch seq_len hidden_dim"] | None = None

    def __init__(
        self,
        loss=None,
        logits=None,
        past_key_values=None,
        hidden_states=None,
        attentions=None,
        rope_deltas=None,
        image_hidden_states=None,
        **kwargs,
    ):
        """Preserve the original positional API while accepting unified VLM fields."""
        super().__init__(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=attentions,
            rope_deltas=rope_deltas,
            image_hidden_states=image_hidden_states,
            **kwargs,
        )

    def __reduce__(self):
        """Restore fields and ordered mapping through pickle state, not positional arguments.

        ModelOutput's positional reconstruction follows inherited dataclass order,
        whereas this class deliberately keeps its legacy constructor order.
        OrderedDict's state and item iterator already preserve every field.
        """
        constructor, _args, *state = super().__reduce__()
        return constructor, (), *state

    def __post_init__(self):
        """Keep legacy mapping, tuple and integer-index ordering before added VLM fields."""
        super().__post_init__()
        self._restore_legacy_order()

    def _sync_output_field(self, name, value):
        """Preserve legacy ordering on mutation and pickle/deepcopy reconstruction."""
        super()._sync_output_field(name, value)
        self._restore_legacy_order()

    def _restore_legacy_order(self):
        """Place historical fields ahead of newly inherited VLM fields."""
        for name in reversed(
            ("loss", "logits", "past_key_values", "hidden_states", "attentions", "rope_deltas", "image_hidden_states")
        ):
            if name in self:
                self.move_to_end(name, last=False)


@auto_pytree
class Qwen3VLModelOutputWithPast(ModelOutput):
    """Base-model output for Qwen3-VL with optional mRoPE position deltas.

    Mirrors the standard ``BaseModelOutputWithPast`` but additionally
    threads through the precomputed mRoPE position offsets used by the
    multimodal RoPE branch. Holding the deltas here keeps the language
    model agnostic to whether the inputs were pure text or contained
    vision/video placeholders — downstream wrappers reuse the same
    deltas when extending generation.

    Attributes:
        last_hidden_state: Final-layer hidden states of shape
            ``(batch, seq_len, hidden_size)``.
        past_key_values: Updated KV cache for autoregressive decoding,
            in the format expected by :class:`TransformerCache`.
        hidden_states: Per-layer hidden states when
            ``output_hidden_states`` is requested.
        attentions: Per-layer attention weights when ``output_attentions``
            is requested.
        rope_deltas: Per-batch integer offsets that align the 1-D text
            RoPE indices with the 3-D visual mRoPE positions; required
            for correct continuation of multimodal sequences during
            decode.
    """

    last_hidden_state: Array = None
    past_key_values: list[Array] | None = None
    hidden_states: tuple[Array] | None = None
    attentions: tuple[Array] | None = None
    rope_deltas: Array | None = None


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

    Different from Qwen2-VL, Qwen3-VL uses timestamps rather than absolute
    time position IDs for videos.

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
    if video_grid_thw is not None:
        # Qwen3-VL timestamps videos per frame: split each video's grid into one row per temporal step, then the
        # position layout is exactly Qwen2-VL's.
        video_grid_thw = np.asarray(video_grid_thw).copy()
        video_grid_thw = np.repeat(video_grid_thw, video_grid_thw[:, 0].astype(int), axis=0)
        video_grid_thw[:, 0] = 1
    return qwen2_vl_get_rope_index(
        input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
        spatial_merge_size=spatial_merge_size,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        vision_start_token_id=vision_start_token_id,
        tokens_per_second=tokens_per_second,
        second_per_grid_ts=second_per_grid_ts,
        context_len=context_len,
        seq_len=seq_len,
    )


# The cumsum-gather merge is shared infrastructure; these families each had
# their own copy of it. `BaseVisionLanguageModule.merge_multimodal_embeddings`
# is the same algorithm, so bind the module-level name to it rather than
# restating it. Keeping the name means call sites (and importers) are unchanged.
merge_multimodal_embeddings = BaseVisionLanguageModule.merge_multimodal_embeddings


# The packed-vision block-diagonal mask is shared infrastructure; this family
# had its own copy. Bind the module-level name so call sites are unchanged.
create_attention_mask = block_diagonal_bias


class Qwen3VLVisionPatchEmbed(spx.Module):
    """3D convolution-based patch embedding for Qwen3-VL vision encoder.

    Converts input image/video pixels into patch embeddings using a 3D convolution
    that operates over temporal and spatial dimensions simultaneously.
    """

    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ) -> None:
        """Initialize Qwen3-VL vision patch embedding layer.

        Args:
            config (Qwen3VLVisionConfig): Vision encoder configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__()
        self.dtype = dtype
        self.config = config
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.hidden_size = config.hidden_size

        kernel_size = (config.temporal_patch_size, config.patch_size, config.patch_size)
        self.proj = nn.Conv3d(
            in_channels=config.in_channels,
            out_channels=config.hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            precision=precision,
        )

    def forward(self, hidden_states: Array) -> Array:
        """Apply 3D convolution to extract patch embeddings.

        Args:
            hidden_states (Array): Input pixel values flattened into patches.

        Returns:
            Array: Patch embeddings of shape (num_patches, hidden_size).
        """
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
        # SpectraX Conv3d's dtype arguments govern storage. Cast operands,
        # not parameter leaves, so mixed storage/compute dtypes remain valid.
        hidden_states = conv(
            hidden_states.astype(self.dtype),
            self.proj.weight.value.astype(self.dtype),
            self.proj.bias.value.astype(self.dtype),
            stride=self.proj.stride,
            padding=self.proj.padding,
            dilation=self.proj.dilation,
            groups=self.proj.groups,
            precision=self.proj.precision,
        )
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        return hidden_states


class Qwen3VLVisionPatchMerger(spx.Module):
    """Spatial patch merger with MLP gating for Qwen3-VL.

    Merges spatially adjacent patches to reduce sequence length while
    preserving important visual information through a gated MLP.
    """

    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        use_postshuffle_norm: bool = False,
        *,
        rngs: spx.Rngs,
    ) -> None:
        """Initialize Qwen3-VL vision patch merger.

        Args:
            config (Qwen3VLVisionConfig): Vision encoder configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            use_postshuffle_norm (bool, optional): Whether to apply normalization after
                spatial shuffling. Defaults to False.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__()
        self.dtype = dtype
        self.spatial_merge_size = config.spatial_merge_size
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm

        self.norm = LayerNorm(
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

    def forward(self, x: Array) -> Array:
        """Merge patches through normalization and gated MLP.

        Args:
            x (Array): Input patch embeddings.

        Returns:
            Array: Merged patch embeddings with reduced spatial dimensions.
        """
        x = self.norm(x.reshape(-1, self.hidden_size) if self.use_postshuffle_norm else x).reshape(-1, self.hidden_size)
        x = self.linear_fc2(jax.nn.gelu(self.linear_fc1(x), approximate=False))
        return x


class Qwen3VLVisionMLP(spx.Module):
    """Feed-forward network for Qwen3-VL vision encoder.

    Implements a two-layer MLP with GELU activation for vision feature
    transformation within the vision transformer blocks.
    """

    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        layer_idx: int | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ) -> None:
        """Initialize Qwen3-VL vision MLP layer.

        Args:
            config (Qwen3VLVisionConfig): Vision encoder configuration.
            layer_idx (int | None, optional): Index of this layer in the encoder.
                Defaults to None.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
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

    def forward(self, x: Array) -> Array:
        """Apply feedforward transformation with GELU activation.

        Args:
            x (Array): Input tensor.

        Returns:
            Array: Transformed tensor with same shape as input.
        """
        return self.linear_fc2(self.act(self.linear_fc1(x)))


class Qwen3VLVisionAttention(UnifiedAttention):
    """Self-attention for Qwen3-VL vision encoder with rotary embeddings.

    Implements multi-head self-attention with 2D rotary position embeddings
    for encoding spatial relationships in vision features.
    """

    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Qwen3-VL vision attention layer.

        Args:
            config (Qwen3VLVisionConfig): Vision encoder configuration.
            layer_idx (int): Index of this layer in the encoder.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
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
            use_gqa=False,
        )

    def define_network(
        self,
        config: Qwen3VLVisionConfig,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        precision: jax.lax.PrecisionLike,
        rngs: spx.Rngs,
    ) -> None:
        """Define the QKV and output projection layers for vision attention.

        Args:
            config: Vision encoder configuration.
            dtype: Data type for computation.
            param_dtype: Data type for parameters.
            precision: Numerical precision for matrix operations.
            rngs: Random number generator state.
        """
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

    def _create_attention_performer(self, config, rngs: spx.Rngs):
        """Create the attention performer module.

        Args:
            config: Vision configuration.
            rngs (spx.Rngs): Random number generator state.

        Returns:
            FlexibleAttentionModule: Configured attention module.
        """
        return FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=0.0,
            attn_mechanism="vanilla",
            requires_cache=False,  # Vision encoder doesn't need KV cache
        )

    def forward(
        self,
        hidden_states: Array,
        cu_seqlens: Array,
        rotary_pos_emb: Array = None,
    ) -> Array:
        """Apply self-attention with rotary position embeddings.

        Args:
            hidden_states (Array): Input tensor of shape (seq_len, hidden_size).
            cu_seqlens (Array): Cumulative sequence lengths for block-diagonal attention.
            rotary_pos_emb (Array, optional): Precomputed rotary position embeddings.
                Defaults to None.

        Returns:
            Array: Attention output of shape (seq_len, hidden_size).
        """
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
        q = apply_rope_split_half(q, cos, sin)
        k = apply_rope_split_half(k, cos, sin)

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
        attn_output = self.shard_attention_prod(attn_output)
        attn_output = checkpoint_name(self.proj(attn_output), "vision_attn_output")
        attn_output = self.shard_attention_prod(attn_output)
        return attn_output


class Qwen3VLVisionBlock(spx.Module):
    """Transformer block for Qwen3-VL vision encoder.

    Combines self-attention and MLP layers with pre-normalization
    architecture and residual connections for vision feature processing.
    """

    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ) -> None:
        """Initialize Qwen3-VL vision transformer block.

        Args:
            config (Qwen3VLVisionConfig): Vision encoder configuration.
            layer_idx (int): Index of this layer in the encoder.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.norm1 = LayerNorm(
            config.hidden_size,
            epsilon=1e-6,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.norm2 = LayerNorm(
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
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def forward(
        self,
        hidden_states: Array,
        cu_seqlens: Array,
        rotary_pos_emb: Array,
    ) -> Array:
        """Forward pass through the vision transformer block.

        Applies pre-normalization: x + attn(norm(x)) followed by x + mlp(norm(x)).

        Args:
            hidden_states (Array): Input tensor of shape (seq_len, hidden_size).
            cu_seqlens (Array): Cumulative sequence lengths for block-diagonal attention.
            rotary_pos_emb (Array): Precomputed rotary position embeddings.

        Returns:
            Array: Output tensor of shape (seq_len, hidden_size).
        """
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


@register_module(TaskType.BASE_VISION, config=Qwen3VLConfig, model_type="qwen3_vl")
class Qwen3VisionTransformerPretrainedModel(EasyDeLBaseModule):
    """Vision transformer encoder for Qwen3-VL.

    Processes images and videos through patch embedding, positional encoding,
    transformer blocks, and patch merging to produce visual features for the
    language model.

    Attributes:
        config_class: Configuration class for the vision encoder.
        patch_embed: 3D convolution-based patch embedding.
        pos_embed: Learnable position embeddings.
        blocks: List of vision transformer blocks.
        merger: Patch merger for reducing spatial dimensions.
        deepstack_merger_list: Additional mergers for deepstack visual features.
    """

    config_class = Qwen3VLVisionConfig

    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Qwen3-VL vision transformer encoder.

        Args:
            config (Qwen3VLVisionConfig): Vision encoder configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
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

        with self.assign_layer_stage(layer_idx=0, total_layers=config.depth):
            self.pos_embed = Embed(
                num_embeddings=config.num_position_embeddings,
                features=config.hidden_size,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

        self.spatial_merge_size = config.spatial_merge_size
        head_dim = config.hidden_size // config.num_heads
        self._head_dim_ro = head_dim // 2

        self.blocks = nn.ModuleList([])
        for idx in range(config.depth):
            with self.assign_layer_stage(idx, total_layers=config.depth):
                self.blocks.append(
                    Qwen3VLVisionBlock(
                        config=config,
                        layer_idx=idx,
                        dtype=dtype,
                        param_dtype=param_dtype,
                        precision=precision,
                        rngs=rngs,
                    )
                )

        self.merger = Qwen3VLVisionPatchMerger(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.deepstack_merger_list = nn.ModuleList(
            [
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
        )

        self.num_grid_per_side = int(math.sqrt(config.num_position_embeddings))

    def get_dtype(self) -> jnp.dtype:
        """Get the data type used by the model parameters.

        Returns:
            jnp.dtype: Data type of the model parameters.
        """
        probe = self.blocks[0].mlp.linear_fc2
        weight = getattr(probe, "weight", None)
        if weight is not None:
            return weight.value.dtype
        # Quantized linears carry no dense `weight`; their param_dtype is the
        # dtype activations/dequantized weights use.
        param_dtype = getattr(probe, "param_dtype", None)
        if param_dtype is not None:
            return jnp.dtype(param_dtype)
        return self.patch_embed.proj.weight.value.dtype

    def fast_pos_embed_interpolate(self, grid_thw: Array) -> Array:
        """Compute positional embeddings with bilinear interpolation.

        Args:
            grid_thw: Grid dimensions (temporal, height, width) per image, shape (num_images, 3)

        Returns:
            Positional embeddings with shape (total_tokens, hidden_size)
        """
        return vision_grid_pos_embed_interpolate(
            self.pos_embed, grid_thw, self.num_grid_per_side, self.spatial_merge_size
        )

    def rot_pos_emb(self, grid_thw: Array, max_grid_size: int) -> Array:
        """Compute rotary position embeddings for vision features.

        Matches HuggingFace's Qwen3VLVisionModel.rot_pos_emb exactly.
        """
        return vision_grid_rotary_embeddings(grid_thw, max_grid_size, self._head_dim_ro, self.spatial_merge_size)

    def forward(
        self,
        hidden_states: Array,
        grid_thw: Array,
        max_grid_size: int,
    ) -> tuple[Array, list[Array]]:
        """Forward pass through the vision transformer encoder.

        Processes input pixels through patch embedding, adds positional embeddings,
        applies transformer blocks with rotary embeddings, and merges patches.

        Args:
            hidden_states (Array): Input pixel values.
            grid_thw (Array): Grid dimensions (temporal, height, width) per image/video.
            max_grid_size (int): Maximum grid size for positional embeddings.

        Returns:
            tuple[Array, list[Array]]: Tuple of (merged_features, deepstack_features).
                merged_features: Final vision embeddings after patch merging.
                deepstack_features: Intermediate features from specified layers.
        """
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

        def _layer_loop(block, carry):
            """Apply a single vision-encoder block inside the layer-stack scan.

            Body of ``self.blocks.scan``; runs ``block`` on the current
            visual hidden states and returns the updated carry tuple.
            """
            hidden_states, layer_num = carry
            with self._layer_stage_context(layer_num, layers=self.blocks):
                hidden_states = block(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    rotary_pos_emb=rotary_pos_emb,
                )
            hidden_states = self._mark_layer_stage_boundary(hidden_states, layer_num, layers=self.blocks)
            if layer_num in self.config.deepstack_visual_indexes:
                deepstack_idx = self.config.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = self.deepstack_merger_list[deepstack_idx](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

            return hidden_states, layer_num + 1

        # Deepstack features are gathered into a Python list keyed by the
        # traced layer number, which is incompatible with a compiled
        # lax.scan body (``layer_num in deepstack_visual_indexes`` needs a
        # concrete value and the append is a Python side effect). Keep the
        # unrolled (traced) path whenever deepstack is configured.
        hidden_states, _ = self.blocks.scan(
            _layer_loop,
            (hidden_states, 0),
            trace=not self.config.scan_layers
            or self._pipeline_stage_count() > 1
            or bool(self.config.deepstack_visual_indexes),
            **self._layer_scan_kwargs("vision"),
        )
        hidden_states = self.merger(hidden_states)
        return hidden_states, deepstack_feature_lists

    def get_encoder(self):
        """Get the encoder component.

        Returns:
            Qwen3VisionTransformerPretrainedModel: The vision encoder itself.
        """
        return self

    def get_decoder(self):
        """Get the decoder component.

        Raises:
            NotImplementedError: Vision model does not have a decoder.
        """
        raise NotImplementedError("Vision model does not have a decoder.")

    def get_lm_head(self):
        """Get the language model head.

        Raises:
            NotImplementedError: Vision model does not have a language model head.
        """
        raise NotImplementedError("Vision model does not have a language model head.")

    def get_embedding(self):
        """Get the embedding layer.

        Returns:
            Qwen3VLVisionPatchEmbed: The patch embedding layer.
        """
        return self.patch_embed


class Qwen3VLTextMLP(spx.Module):
    """SwiGLU feed-forward network for Qwen3-VL text decoder.

    Implements the feedforward network with SwiGLU activation function
    for enhanced representation learning in the text decoder.
    """

    def __init__(
        self,
        config: Qwen3VLTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize Qwen3-VL text MLP block.

        Args:
            config (Qwen3VLTextConfig): Text decoder configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the decoder.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.layer_idx = layer_idx

        self.gate_up_proj = ColumnParallelLinear(
            config.hidden_size,
            (config.intermediate_size, config.intermediate_size),
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
            layout=dense_gate_up_layout(config.intermediate_size),
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    @property
    def reform_param(self):
        """Checkpoint reform rules for the fused gate-up projection.

        Returns:
            dict: Mapping from external ``gate_up_proj`` checkpoint
            entries to the layout used internally by the fused
            :attr:`gate_up_proj` ColumnParallelLinear, enabling
            round-trip conversion of HuggingFace-style separate gate /
            up weights into the fused layout used here.
        """
        return self.gate_up_proj.build_reform_param("gate_up_proj", config=self.config)

    def forward(self, hidden_states: Array) -> Array:
        """Apply SwiGLU feedforward transformation.

        Delegates to :func:`easydel.layers.gated_mlp_forward`, which routes
        supported layouts through the ejkernel fused MLP kernels and runs
        the legacy composition otherwise.

        Args:
            hidden_states (Array): Input tensor of shape (batch, seq_len, hidden_dim).

        Returns:
            Array: Transformed hidden states of shape (batch, seq_len, hidden_dim).
        """
        return gated_mlp_forward(self, hidden_states)


class Qwen3VLTextAttention(UnifiedAttention):
    """Causal self-attention for Qwen3-VL text decoder with mRoPE and QK-norm support.

    This attention module supports multi-dimensional rotary position embeddings (mRoPE)
    for proper handling of multimodal inputs with temporal, height, and width dimensions.
    """

    def __init__(
        self,
        config: Qwen3VLTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize Qwen3-VL text attention layer.

        Args:
            config (Qwen3VLTextConfig): Text decoder configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the decoder.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="standard",
            causal=True,
            sliding_window=(
                config.sliding_window
                if config.use_sliding_window and config.layer_types[layer_idx] == "sliding_attention"
                else None
            ),
            use_qk_norm=True,
        )

    def _postprocess_qkv(self, query_states, key_states, value_states):
        """Apply Q/K normalization after computing query, key, and value projections.

        Args:
            query_states: Query tensor from projection layer.
            key_states: Key tensor from projection layer.
            value_states: Value tensor from projection layer.

        Returns:
            Tuple of normalized query, normalized key, and value tensors.
        """
        return self.query_normalization(query_states), self.key_normalization(key_states), value_states


class Qwen3VLTextDecoderLayer(spx.Module):
    """Transformer decoder layer for Qwen3-VL text model.

    Combines multi-head attention with Q/K normalization and feedforward networks
    with RMS normalization and residual connections.
    """

    def __init__(
        self,
        config: Qwen3VLTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize Qwen3-VL text decoder layer.

        Args:
            config (Qwen3VLTextConfig): Text decoder configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the decoder.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.layer_idx = layer_idx

        self.self_attn = Qwen3VLTextAttention(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )

        self.mlp = Qwen3VLTextMLP(
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
    ) -> DecoderLayerOutput:
        """Forward pass through the decoder layer.

        Applies pre-normalization architecture: x + attn(norm(x)) followed by x + mlp(norm(x)).

        Args:
            hidden_states (Array): Input tensor of shape (batch, seq_len, hidden_dim).
            mask_info (MaskInfo): Attention mask information including causal masks.
            position_ids (Array): Position indices, shape (3, batch, seq_len) for mRoPE.
            mode (RUNTIME_MODE_TYPES): Runtime mode (train, decode, etc.) for optimization.
            cache_view (TransformerCacheView | RaggedPagesCacheView | None, optional): Cache view.
                Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Cache metadata. Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            frequencies (Array | None, optional): Precomputed RoPE frequencies. Defaults to None.

        Returns:
            DecoderLayerOutput: Contains hidden states, attention weights, and cache view.
        """
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
            feed_forward_hidden_states = blockwise_ffn(
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
            partition_manager=self.config.runtime_sharding_resolver,
        )

        return DecoderLayerOutput(
            hidden_states=checkpoint_name(hidden_states, "layer_output"),
            attention_weight=attn_outputs.attention_weight,
            cache_view=attn_outputs.cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=Qwen3VLTextConfig, model_type="qwen3_vl")
class Qwen3VLTextModel(DeepStackVisualMixin, EasyDeLBaseModule):
    """Text decoder model for Qwen3-VL.

    Implements the text decoder component of Qwen3-VL, utilizing transformer blocks
    with RMSNorm, Q/K normalization, and multi-dimensional rotary position embeddings
    (mRoPE) for handling multimodal inputs.

    Attributes:
        config: Configuration for the text decoder.
        embed_tokens: Token embedding layer.
        layers: List of transformer decoder layers.
        norm: Final layer normalization.
    """

    def __init__(
        self,
        config: Qwen3VLTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Qwen3-VL text decoder model.

        Args:
            config (Qwen3VLTextConfig): Text decoder configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        with self.assign_layer_stage(0, total_layers=config.num_hidden_layers):
            self.embed_tokens = Embed(
                num_embeddings=config.vocab_size,
                features=config.hidden_size,
                dtype=dtype,
                param_dtype=param_dtype,
                embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
                rngs=rngs,
            )

        remat_layer_block = auto_remat(
            Qwen3VLTextDecoderLayer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.layers = nn.ModuleList([])
        for i in range(config.num_hidden_layers):
            with self.assign_layer_stage(i, total_layers=config.num_hidden_layers):
                self.layers.append(
                    remat_layer_block(
                        config=config,
                        layer_idx=i,
                        dtype=dtype,
                        param_dtype=param_dtype,
                        precision=precision,
                        rngs=rngs,
                    )
                )

        final_layer_idx = max(0, config.num_hidden_layers - 1)
        with self.assign_layer_stage(final_layer_idx, total_layers=config.num_hidden_layers):
            self.norm = RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

    @cached_property
    def frequencies(self):
        """Get cached RoPE frequency tensor from config.

        Computes and caches the rotary position embedding frequencies
        based on the model's head dimension and rope_theta configuration.

        Returns:
            Array: Precomputed RoPE frequencies for position encoding.
        """
        head_dim = getattr(self.config, "qk_rope_head_dim", None) or self.config.head_dim
        return self.config.get_basic_frequencies(
            head_size=head_dim,
            rotary_dim=head_dim,
            base=self.config.rope_theta,
        )

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        visual_pos_masks: Bool[Array, "batch seq_len"] | None = None,
        deepstack_visual_embeds: list[Array] | None = None,
        trace: bool = False,
    ) -> BaseModelOutput:
        """Forward pass through the Qwen3-VL text decoder model.

        Processes input tokens through embedding, all decoder layers with mRoPE
        and RMSNorm, and final normalization. Supports deepstack visual embedding
        injection at intermediate layers.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch, seq_len).
                Must be provided if inputs_embeds is None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings of shape
                (batch, seq_len, hidden_size). Defaults to None.
            attention_mask (Array | None, optional): Boolean mask to avoid attention on
                padding tokens, shape (batch, seq_len). Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information for attention.
                Defaults to None.
            position_ids (Array | None, optional): 3D position indices for mRoPE,
                shape (3, batch, seq_len). Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode (train/decode).
                Auto-detected if None. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cache with precomputed key-value states. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache management. Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention weights.
                Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return all hidden states.
                Defaults to None.
            visual_pos_masks (Array | None, optional): Mask indicating visual token positions.
                Defaults to None.
            deepstack_visual_embeds (list[Array] | None, optional): Intermediate visual
                embeddings for deepstack fusion. Defaults to None.

        Returns:
            BaseModelOutput: Contains last_hidden_state, optional all hidden_states,
                optional attentions, and updated past_key_values.

        Raises:
            ValueError: If both input_ids and inputs_embeds are provided or both are None.
            AssertionError: If sequence_length exceeds max_position_embeddings.
        """
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
        attention_mask = token_attention_mask_from_mask_info(mask_info, sequence_length)

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
            partition_manager=self.config.runtime_sharding_resolver,
        )

        views = past_key_values.views if past_key_values is not None else None
        has_cache_views = views is not None and any(v is not None for v in views)
        needs_trace_cache = mode == common_types.MODE_DECODE or has_cache_views
        # DeepStack injection indexes a Python list by layer index, so the stack must run
        # traced (unrolled, integer ``idx``) whenever features are present (r11 S10-2).
        has_deepstack = bool(deepstack_visual_embeds)

        trace_layers = self._layer_scan_trace(
            trace,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            cache_views=views,
            extra=needs_trace_cache or has_deepstack,
        )
        cache_views = views if trace_layers else None
        # Materialize the cached property outside the scan trace to avoid
        # storing a scan-local tracer on the model on the first forward.
        frequencies = self.frequencies

        def _run_layer(block, carry):
            """Apply a single language-model decoder layer inside the layer-stack scan.

            Body of ``self.layers.scan``; runs ``block`` on the current
            hidden states with the appropriate cache view, optionally
            mixes in deepstack visual embeddings at the configured layer
            indices, accumulates per-layer hidden states / attention
            weights when requested, and returns the updated carry tuple.
            """
            hs, cv, ah, aa, idx = carry
            if output_hidden_states:
                ah = (*ah, hs)
            with self._layer_stage_context(idx, layers=self.layers):
                layer_outputs = block(
                    hidden_states=hs,
                    mask_info=mask_info,
                    position_ids=position_ids,
                    mode=mode,
                    cache_view=self._layer_cache_view_at(cv, idx, enabled=trace_layers, cache=past_key_values),
                    cache_metadata=cache_metadata,
                    output_attentions=output_attentions,
                    frequencies=frequencies,
                )
            hs = layer_outputs.hidden_states
            if has_deepstack and idx < len(deepstack_visual_embeds):
                hs = self._deepstack_process(
                    hs,
                    visual_pos_masks,
                    deepstack_visual_embeds[idx],
                )
            hs = self._mark_layer_stage_boundary(hs, idx, layers=self.layers)
            cv = self._layer_cache_view_update(
                cv,
                idx,
                layer_outputs.cache_view,
                enabled=trace_layers,
                cache=past_key_values,
            )
            if output_attentions:
                aa = (*aa, layer_outputs.attention_weight)
            return hs, cv, ah, aa, idx + 1

        init_carry = (hidden_states, cache_views, all_hidden_states, all_attentions, 0)
        hidden_states, _, all_hidden_states, all_attentions, _ = self.layers.scan(
            _run_layer,
            init_carry,
            trace=trace_layers,
            **self._layer_scan_kwargs("text"),
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
        """Get the encoder component.

        Raises:
            NotImplementedError: Text model does not have an encoder.
        """
        raise NotImplementedError("Text model does not have an encoder.")

    def get_decoder(self):
        """Get the decoder component.

        Returns:
            Qwen3VLTextModel: The text decoder itself.
        """
        return self

    def get_lm_head(self):
        """Get the language model head.

        Raises:
            NotImplementedError: Base model does not have a language model head.
        """
        raise NotImplementedError("Base model does not have a language model head.")

    def get_embedding(self):
        """Get the token embedding layer.

        Returns:
            Embed: The token embedding layer.
        """
        return self.embed_tokens


@register_module(TaskType.VISION_LM, config=Qwen3VLConfig, model_type="qwen3_vl")
class Qwen3VLModel(QwenVLMultimodalMixin, EasyDeLBaseModule):
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
        rngs: spx.Rngs,
    ):
        """Initialize Qwen3-VL multimodal model.

        Args:
            config (Qwen3VLConfig): Multimodal model configuration containing
                vision_config and text_config.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.visual = Qwen3VisionTransformerPretrainedModel(
            config.vision_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.language_model = Qwen3VLTextModel(
            config.text_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def get_input_embeddings(self):
        """Get the input embedding layer.

        Returns:
            Embed: The token embedding layer from the language model.
        """
        return self.language_model.get_embedding()

    def set_input_embeddings(self, value):
        """Set the input embedding layer.

        Args:
            value: New embedding layer to use.
        """
        self.language_model.embed_tokens = value

    def set_decoder(self, decoder):
        """Set the language model decoder.

        Args:
            decoder: New decoder model to use.
        """
        self.language_model = decoder

    def get_decoder(self):
        """Get the language model decoder.

        Returns:
            Qwen3VLTextModel: The language model component.
        """
        return self.language_model

    def get_rope_index(
        self,
        input_ids: Array,
        image_grid_thw: Array | None = None,
        video_grid_thw: Array | None = None,
        attention_mask: Array | None = None,
    ) -> tuple[Array, Array]:
        """Calculate 3D RoPE indices for multimodal inputs.

        Different from Qwen2-VL, Qwen3-VL uses timestamps rather than absolute
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
        input_ids: Int[Array, "batch seq_len"] | None,
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
        """Compute embeddings with multimodal fusion.

        Merges visual embeddings from images/videos into text embeddings
        at placeholder token positions.

        Args:
            input_ids (Array): Input token IDs of shape (batch, seq_len).
            inputs_embeds (Array | None, optional): Pre-computed embeddings. Defaults to None.
            pixel_values (Array | None, optional): Image pixel values. Defaults to None.
            pixel_values_videos (Array | None, optional): Video pixel values. Defaults to None.
            image_grid_thw (Array | None, optional): Image grid dimensions. Defaults to None.
            video_grid_thw (Array | None, optional): Video grid dimensions. Defaults to None.
            image_max_grid_size (int | None, optional): Max image grid size. Defaults to None.
            video_max_grid_size (int | None, optional): Max video grid size. Defaults to None.
            image_embeds (Array | None, optional): Pre-computed image embeddings. Defaults to None.
            video_embeds (Array | None, optional): Pre-computed video embeddings. Defaults to None.
            **kwargs: Additional arguments.

        Returns:
            Array: Fused embeddings with visual content merged at placeholder positions.

        Raises:
            ValueError: If input_ids is None when required.
        """
        return super().compute_embedding(
            input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            image_max_grid_size=image_max_grid_size,
            video_max_grid_size=video_max_grid_size,
            image_embeds=image_embeds,
            video_embeds=video_embeds,
            **kwargs,
        )

    def compute_embedding_with_info(
        self,
        input_ids: Int[Array, "batch seq_len"] | None,
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
        deepstack_image_embeds: list[Array] | None = None,
        deepstack_video_embeds: list[Array] | None = None,
        **kwargs,
    ) -> tuple[Array, EmbeddingInfo]:
        """Compute embeddings with multimodal fusion and return additional info.

        Similar to compute_embedding but also returns EmbeddingInfo containing
        position IDs, rope deltas, and deepstack visual embeddings.

        Args:
            input_ids (Array): Input token IDs of shape (batch, seq_len).
            inputs_embeds (Array | None, optional): Pre-computed embeddings. Defaults to None.
            pixel_values (Array | None, optional): Image pixel values. Defaults to None.
            pixel_values_videos (Array | None, optional): Video pixel values. Defaults to None.
            image_grid_thw (Array | None, optional): Image grid dimensions. Defaults to None.
            video_grid_thw (Array | None, optional): Video grid dimensions. Defaults to None.
            image_max_grid_size (int | None, optional): Max image grid size. Defaults to None.
            video_max_grid_size (int | None, optional): Max video grid size. Defaults to None.
            image_embeds (Array | None, optional): Pre-computed image embeddings. Defaults to None.
            video_embeds (Array | None, optional): Pre-computed video embeddings. Defaults to None.
            attention_mask (Array | None, optional): Attention mask. Defaults to None.
            deepstack_image_embeds: Optional cached per-layer image residuals, paired with image_embeds.
            deepstack_video_embeds: Optional cached per-layer video residuals, paired with video_embeds.
            **kwargs: Additional arguments.

        Returns:
            tuple[Array, EmbeddingInfo]: Tuple of (fused_embeddings, embedding_info).

        Raises:
            ValueError: If input_ids is None or cached residual shapes/layer counts disagree.
        """
        for name, features, residuals in (
            ("image", image_embeds, deepstack_image_embeds),
            ("video", video_embeds, deepstack_video_embeds),
        ):
            if residuals is not None and (
                features is None or any(residual.shape != features.shape for residual in residuals)
            ):
                raise ValueError(f"Cached {name} DeepStack residuals require matching `{name}_embeds` shapes.")
        embeddings, info = super().compute_embedding_with_info(
            input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            image_max_grid_size=image_max_grid_size,
            video_max_grid_size=video_max_grid_size,
            image_embeds=image_embeds,
            video_embeds=video_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )
        if deepstack_image_embeds is not None or deepstack_video_embeds is not None:
            images = info.deepstack_image_embeds if deepstack_image_embeds is None else deepstack_image_embeds
            videos = info.deepstack_video_embeds if deepstack_video_embeds is None else deepstack_video_embeds
            if images and videos and len(images) != len(videos):
                raise ValueError("Image and video DeepStack residuals must have the same number of layers.")
            image_mask, video_mask = self.get_placeholder_mask(input_ids, embeddings)
            info.visual_pos_masks, info.deepstack_visual_embeds = self.merge_deepstack_features(
                image_mask if images else None, video_mask if videos else None, images, videos
            )
            info.deepstack_image_embeds, info.deepstack_video_embeds = images, videos
        return embeddings, info

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
        return super().get_image_features(pixel_values, image_grid_thw, image_max_grid_size)

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
        return super().get_placeholder_mask(input_ids, inputs_embeds, image_features, video_features)

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
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
        """Forward pass through the Qwen3-VL multimodal model.

        Processes multimodal inputs by:
        1. Encoding images/videos through the vision encoder
        2. Merging visual embeddings into text embeddings at placeholder positions
        3. Computing 3D position IDs for proper mRoPE handling
        4. Running the fused embeddings through the text decoder

        Args:
            input_ids (Array | None, optional): Input token IDs. Defaults to None.
            inputs_embeds (Array | None, optional): Pre-computed embeddings. Defaults to None.
            attention_mask (Array | None, optional): Attention mask. Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information. Defaults to None.
            position_ids (Array | None, optional): 3D position IDs for mRoPE. Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode. Defaults to None.
            past_key_values (Cache | None, optional): KV cache for generation. Defaults to None.
            cache_metadata (Metadata | None, optional): Cache metadata. Defaults to None.
            output_attentions (bool | None, optional): Return attention weights. Defaults to None.
            output_hidden_states (bool | None, optional): Return hidden states. Defaults to None.
            visual_pos_masks (Array | None, optional): Visual token position mask. Defaults to None.
            deepstack_visual_embeds (list[Array] | None, optional): Deepstack embeddings.
                Defaults to None.
            pixel_values (Array | None, optional): Image pixels. Defaults to None.
            pixel_values_videos (Array | None, optional): Video pixels. Defaults to None.
            image_grid_thw (tuple | None, optional): Image grid dimensions. Defaults to None.
            video_grid_thw (tuple | None, optional): Video grid dimensions. Defaults to None.
            image_max_grid_size (int | None, optional): Max image grid size. Defaults to None.
            video_max_grid_size (int | None, optional): Max video grid size. Defaults to None.
            cache_position (Array | None, optional): Cache position. Defaults to None.

        Returns:
            Qwen3VLModelOutputWithPast: Model outputs with hidden states and rope deltas.

        Raises:
            ValueError: If neither or both input_ids and inputs_embeds are provided.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        inputs_embeds, embedding_info = self.prepare_multimodal_embeddings(
            input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            image_max_grid_size=image_max_grid_size,
            video_max_grid_size=video_max_grid_size,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        visual_pos_masks = embedding_info.visual_pos_masks
        deepstack_visual_embeds = embedding_info.deepstack_visual_embeds

        rope_deltas = None
        # Without token IDs there is no multimodal placeholder layout to parse.
        # The text decoder derives all three position axes from MaskInfo, just
        # as for its own embeddings-only API (including padding/packed inputs).
        if position_ids is None and input_ids is not None:
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
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

        return Qwen3VLModelOutputWithPast(  # pyright: ignore[reportReturnType]
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=rope_deltas,
        )

    def get_encoder(self):
        """Get the vision encoder.

        Returns:
            Qwen3VisionTransformerPretrainedModel: The vision encoder.
        """
        return self.visual

    def get_lm_head(self):
        """Get the language model head.

        Raises:
            NotImplementedError: Qwen3VLModel does not have a language model head.
        """
        raise NotImplementedError("Qwen3VLModel does not have a language model head.")

    def get_embedding(self):
        """Get the token embedding layer.

        Returns:
            Embed: The token embedding layer from the language model.
        """
        return self.language_model.embed_tokens


@register_module(TaskType.IMAGE_TEXT_TO_TEXT, config=Qwen3VLConfig, model_type="qwen3_vl")
class Qwen3VLForConditionalGeneration(BaseVisionLanguageModule[Qwen3VLModel, Qwen3VLConfig]):
    """Qwen3-VL model for conditional generation from images/videos and text.

    Inherits from BaseVisionLanguageModule to leverage common VLM infrastructure.

    Architecture matches HuggingFace:
    - model: Qwen3VLModel (contains visual + language_model)
    - lm_head: Linear projection to vocab

    Class Attributes:
        _task_type: IMAGE_TEXT_TO_TEXT task type
        _model_type: "qwen3_vl" model identifier
        _supports_video: True (Qwen3-VL supports video input)
        _uses_mrope: True (uses multi-dimensional RoPE)
    """

    _task_type = TaskType.IMAGE_TEXT_TO_TEXT
    _model_type = "qwen3_vl"
    _config_class = Qwen3VLConfig
    _auto_register = False
    _supports_video = True
    _uses_mrope = True

    _vision_tower_name = "visual"
    _projector_name = "merger"
    _language_model_name = "language_model"

    loss_type = "ForCausalLM"

    def __init__(
        self,
        config: Qwen3VLConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Qwen3-VL model for conditional generation.

        Args:
            config (Qwen3VLConfig): Multimodal model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=Qwen3VLModel,
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

    def get_input_embeddings(self):
        """Get the input embedding layer.

        Returns:
            Embed: The token embedding layer.
        """
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        """Set the input embedding layer.

        Args:
            value: New embedding layer to use.
        """
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        """Set the language model decoder.

        Args:
            decoder: New decoder model to use.
        """
        self.model.set_decoder(decoder)

    def get_decoder(self):
        """Get the language model decoder.

        Returns:
            Qwen3VLTextModel: The language model component.
        """
        return self.model.get_decoder()

    @property
    def visual(self):
        """Backward-compatible alias for the Qwen3-VL vision tower.

        Mirrors the HuggingFace attribute layout (``model.visual``) so
        external code that walks the parameter tree by name continues
        to find the vision transformer at the expected location.

        Returns:
            spx.Module: The
            :class:`Qwen3VisionTransformerPretrainedModel` owned by the
            wrapped :class:`Qwen3VLModel`.
        """
        return self.model.visual

    @property
    def language_model(self):
        """Backward-compatible alias for the text decoder.

        Returns:
            spx.Module: The Qwen3 language-model trunk owned by
            :class:`Qwen3VLModel`. The LM head, in contrast, lives on
            this outer wrapper rather than on the inner model.
        """
        return self.model.language_model

    def get_video_features(
        self,
        pixel_values_videos: Array,
        video_grid_thw: Array | None = None,
        video_max_grid_size: int | None = None,
    ) -> tuple[tuple[Array, ...], list[Array]]:
        """Encode videos into continuous embeddings.

        Args:
            pixel_values_videos (Array): Video pixel values.
            video_grid_thw (Array | None, optional): Video grid dimensions. Defaults to None.
            video_max_grid_size (int | None, optional): Max grid size. Defaults to None.

        Returns:
            tuple[tuple[Array, ...], list[Array]]: Video embeddings and deepstack features.
        """
        return self.model.get_video_features(pixel_values_videos, video_grid_thw, video_max_grid_size)

    def get_image_features(
        self,
        pixel_values: Array,
        image_grid_thw: Array | None = None,
        image_max_grid_size: int | None = None,
    ) -> tuple[tuple[Array, ...], list[Array]]:
        """Encode images into continuous embeddings.

        Args:
            pixel_values (Array): Image pixel values.
            image_grid_thw (Array | None, optional): Image grid dimensions. Defaults to None.
            image_max_grid_size (int | None, optional): Max grid size. Defaults to None.

        Returns:
            tuple[tuple[Array, ...], list[Array]]: Image embeddings and deepstack features.
        """
        return self.model.get_image_features(pixel_values, image_grid_thw, image_max_grid_size)

    def compute_embedding_with_info(
        self, input_ids, *args, deepstack_image_embeds=None, deepstack_video_embeds=None, **kwargs
    ):
        """Prepare embeddings and cached/native DeepStack metadata in one pass.

        Args:
            input_ids: Input token IDs.
            *args: Positional arguments forwarded to the multimodal model.
            deepstack_image_embeds: Optional cached per-layer image residuals.
            deepstack_video_embeds: Optional cached per-layer video residuals.
            **kwargs: Embedding, pixel, grid and attention inputs for the model.

        Returns:
            Fused embeddings and EmbeddingInfo, without a second vision encode.
        """
        return self.model.compute_embedding_with_info(
            input_ids,
            *args,
            deepstack_image_embeds=deepstack_image_embeds,
            deepstack_video_embeds=deepstack_video_embeds,
            **kwargs,
        )

    def compute_embedding(self, input_ids, *args, **kwargs):
        """Compute embeddings with multimodal fusion.

        Delegates to the underlying model's compute_embedding method.

        Args:
            input_ids: Input token IDs.
            *args: Positional arguments passed to model.compute_embedding.
            **kwargs: Keyword arguments passed to model.compute_embedding.

        Returns:
            Array: Fused embeddings.
        """
        return self.model.compute_embedding(input_ids, *args, **kwargs)

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
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
        """Forward pass for the Qwen3-VL model.

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
            cache_position: Cache position for incremental decoding
            image_max_grid_size: Maximum grid size for images
            video_max_grid_size: Maximum grid size for videos
            **kwargs: Additional arguments

        Returns:
            VLMCausalLMOutput: Model outputs including logits and optional states
        """
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
            partition_manager=self.config.runtime_sharding_resolver,
        )

        lm_logits = None
        if apply_lm_head:
            lm_logits = self.compute_lm_logits(hidden_states)
            lm_logits = self.apply_logit_cap(lm_logits)

        return Qwen3VLCausalLMOutputWithPast(
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            last_hidden_state=hidden_states,
            attentions=outputs.attentions,
            rope_deltas=getattr(outputs, "rope_deltas", None),
            image_hidden_states=None,
        )

    def get_vision_tower(self) -> spx.Module:
        """Get the vision tower component.

        Returns:
            Qwen3VisionTransformerPretrainedModel: The vision encoder.
        """
        return self.model.visual

    def get_language_model(self) -> spx.Module:
        """Get the language model component.

        Returns:
            Qwen3VLTextModel: The text decoder.
        """
        return self.model.language_model
