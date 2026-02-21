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
import math
import typing as tp
from functools import partial

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
    AttentionLayerOutput,
    BaseModelOutput,
    DecoderLayerOutput,
    ModelOutput,
    VLMCausalLMOutput,
)
from easydel.infra.utils import ACT2FN, auto_remat, block_wise_ffn
from easydel.layers import ColumnParallelLinear, Embed, RMSNorm, RowParallelLinear
from easydel.layers.attention import FlexibleAttentionModule, UnifiedAttention
from easydel.layers.norms import LayerNorm
from easydel.modules._base import BaseVisionLanguageModule

from .qwen2_vl_configuration import Qwen2VLConfig, Qwen2VLTextConfig, Qwen2VLVisionConfig


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
            position_ids = jnp.cumsum(attention_mask, axis=-1) - 1
            position_ids = jnp.where(attention_mask == 0, 1, position_ids)
            position_ids = jnp.expand_dims(position_ids, axis=0).repeat(3, axis=0)
            max_position_ids = jnp.max(position_ids, axis=(0, 2), keepdims=True)
            mrope_position_deltas = (max_position_ids + 1 - attention_mask.shape[-1]).reshape(-1)
        else:
            position_ids = jnp.arange(seq_length).reshape(1, 1, -1).repeat(3, axis=0).repeat(batch_size, axis=1)
            mrope_position_deltas = jnp.zeros((batch_size,), dtype=input_ids.dtype)

    return jnp.asarray(position_ids), jnp.asarray(mrope_position_deltas).reshape(-1, 1)


@auto_pytree
class Qwen2VLCausalLMOutputWithPast(ModelOutput):
    """Output class for Qwen2VL causal language model with past key-values.

    Extends ModelOutput to include rope_deltas for multi-dimensional rotary
    position embeddings used in vision-language models.

    Attributes:
        loss (Array | None): Language modeling loss when labels are provided.
        logits (Array): Prediction scores from the language modeling head,
            shape (batch_size, sequence_length, vocab_size).
        past_key_values (list[Array] | None): Cached key-value states for
            efficient autoregressive generation.
        hidden_states (tuple[Array] | None): Hidden states from all layers
            if output_hidden_states=True.
        attentions (tuple[Array] | None): Attention weights from all layers
            if output_attentions=True.
        rope_deltas (Array | None): Position deltas for multi-dimensional RoPE,
            shape (batch_size, 1), accounting for variable vision token lengths.
    """

    loss: Array | None = None
    logits: Array = None
    past_key_values: list[Array] | None = None
    hidden_states: tuple[Array] | None = None
    attentions: tuple[Array] | None = None
    rope_deltas: Array | None = None


def create_attention_mask(cu_seqlens: Array, seq_length: int, dtype: jnp.dtype) -> Array:
    """Create block-diagonal attention mask from cumulative sequence lengths.

    Creates a mask that allows attention only within segments defined by
    cumulative sequence lengths, used for processing multiple images/videos
    as separate sequences within a batch.

    Args:
        cu_seqlens (Array): Cumulative sequence lengths of shape (num_segments + 1,).
            For example, [0, 100, 256] indicates two segments: positions 0-99 and 100-255.
        seq_length (int): Total sequence length for the output mask.
        dtype (jnp.dtype): Data type for the output mask (typically float32 or bfloat16).

    Returns:
        Array: Block-diagonal attention mask of shape (1, seq_length, seq_length).
            Positions within the same segment have value 0.0, positions across
            different segments have value -inf (dtype minimum).
    """
    positions = jnp.arange(seq_length)
    starts = cu_seqlens[:-1]
    ends = cu_seqlens[1:]
    in_segment = (positions[:, None] >= starts[None, :]) & (positions[:, None] < ends[None, :])

    segment_ids = jnp.argmax(in_segment.astype(jnp.int32), axis=-1)
    same_segment = segment_ids[:, None] == segment_ids[None, :]
    attention_mask = jnp.where(same_segment, 0.0, jnp.finfo(dtype).min).astype(dtype)

    return attention_mask[None, :, :]


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
    """Merge multimodal embeddings into text embeddings at placeholder token positions.

    Replaces embeddings at positions where input_ids match the placeholder token(s)
    with the corresponding multimodal (image/video) embeddings.

    Args:
        input_ids (jax.Array): Input token IDs of shape (batch_size, seq_length).
        inputs_embeds (jax.Array): Text embeddings of shape (batch_size, seq_length, hidden_dim).
        multimodal_embeddings (jax.Array): Flattened vision embeddings of shape (total_tokens, hidden_dim).
        placeholder_token_id (int | list[int]): Token ID(s) indicating where to insert vision embeddings.
            Can be a single int or list of ints for multiple placeholder types.

    Returns:
        jax.Array: Merged embeddings of shape (batch_size, seq_length, hidden_dim) with vision
            embeddings inserted at placeholder positions.
    """
    if isinstance(placeholder_token_id, list):
        placeholder_token_id = jnp.array(placeholder_token_id)
        is_multimodal = jnp.isin(input_ids, placeholder_token_id)
    else:
        is_multimodal = input_ids == placeholder_token_id
    return _merge_multimodal_embeddings(inputs_embeds, is_multimodal, multimodal_embeddings)


def precompute_vl_rotary(dim, theta, max_position):
    """Precompute rotary position embedding frequencies for vision-language attention.

    Computes the inverse frequency matrix used for rotary position embeddings
    in the vision-language attention stack.

    Args:
        dim (int): Dimension of the rotary embedding (typically head_dim).
        theta (float): Base value for computing inverse frequencies (typically 10000.0).
        max_position (int): Maximum sequence position to precompute embeddings for.

    Returns:
        jnp.ndarray: Rotary angle matrix of shape (max_position, dim // 2) containing
            the outer product of position indices and inverse frequencies.
    """
    inv = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype="f4") / dim))
    seq = jnp.arange(0, max_position, "f4")
    return jnp.outer(seq, inv)


def rotate_half(x):
    """Rotate half the hidden dimensions of the input for rotary embeddings.

    Splits the input tensor along the last dimension and swaps the two halves
    with negation, which is the core operation for applying rotary position embeddings.

    Args:
        x (Array): Input tensor of shape (..., head_dim) where head_dim is even.

    Returns:
        Array: Rotated tensor of the same shape, where the first half of the last
            dimension is negated and swapped with the second half.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision(
    q: Array,
    k: Array,
    cos: Array,
    sin: Array,
) -> tuple[Array, Array]:
    """Apply rotary positional embedding to vision features.

    Matches HuggingFace's implementation exactly.

    Args:
        q: Query tensor of shape (seq_len, num_heads, head_dim)
        k: Key tensor of shape (seq_len, num_heads, head_dim)
        cos: Cosine embeddings of shape (seq_len, head_dim)
        sin: Sine embeddings of shape (seq_len, head_dim)

    Returns:
        Tuple of (q_embed, k_embed) with same shapes as inputs.
    """
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q = q.astype("f4")
    k = k.astype("f4")
    # unsqueeze(-2) adds a dimension at axis -2 (before last)
    cos = jnp.expand_dims(cos, axis=-2).astype("f4")  # (seq_len, 1, head_dim)
    sin = jnp.expand_dims(sin, axis=-2).astype("f4")  # (seq_len, 1, head_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.astype(orig_q_dtype)
    k_embed = k_embed.astype(orig_k_dtype)
    return q_embed, k_embed


class Qwen2VLPatchEmbed(nn.Module):
    """Patch embedding layer for converting images/video frames into embeddings.

    Uses 3D convolution to convert image or video patches into embeddings,
    handling both spatial (height, width) and temporal dimensions for video input.
    """

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
        """Initialize the patch embedding layer.

        Args:
            patch_size (int, optional): Spatial patch size (height and width). Defaults to 14.
            temporal_patch_size (int, optional): Temporal patch size for video frames. Defaults to 2.
            in_channels (int, optional): Number of input channels (RGB=3). Defaults to 3.
            embed_dim (int, optional): Output embedding dimension. Defaults to 1152.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            rngs (nn.Rngs): Random number generator state.
        """
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

    def __call__(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> Array:
        """Apply patch embedding to input pixels.

        Args:
            hidden_states (Array): Input pixel values of shape
                (num_patches * temporal * channels, patch_size, patch_size) flattened.

        Returns:
            Array: Patch embeddings of shape (num_patches, embed_dim).
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
        hidden_states = self.proj(hidden_states.astype(self.dtype))
        hidden_states = hidden_states.reshape(-1, self.embed_dim)
        return hidden_states


class Qwen2VLPatchMerger(nn.Module):
    """Patch merger module to downsample visual tokens by merging spatial neighbors.

    Reduces the number of visual tokens by merging adjacent patches spatially,
    projecting them through an MLP to match the language model's hidden dimension.
    """

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
        """Initialize the patch merger.

        Args:
            dim (int): Output dimension (language model hidden size).
            context_dim (int): Input context dimension from vision encoder.
            spatial_merge_size (int, optional): Number of patches to merge spatially.
                Defaults to 2 (merges 2x2=4 patches into 1).
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__()
        self.dtype = dtype
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = LayerNorm(
            context_dim,
            epsilon=1e-6,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.mlp = nn.List(
            [
                ColumnParallelLinear(
                    self.hidden_size,
                    self.hidden_size,
                    use_bias=True,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    precision=precision,
                    rngs=rngs,
                ),
                partial(nn.gelu, approximate=False),
                RowParallelLinear(
                    self.hidden_size,
                    dim,
                    use_bias=True,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    precision=precision,
                    rngs=rngs,
                ),
            ]
        )

    def __call__(self, x: Array) -> Array:
        """Merge and project patches.

        Args:
            x (Array): Input patches of shape (num_patches, context_dim).

        Returns:
            Array: Merged and projected patches of shape (num_merged_patches, dim).
        """
        x = self.ln_q(x).reshape(-1, self.hidden_size)
        for mlp in self.mlp:
            x = mlp(x)
        return x


class Qwen2VLVisionMLP(nn.Module):
    """Multi-Layer Perceptron module for the Qwen2-VL vision encoder.

    Implements a two-layer feedforward network with configurable activation function
    for use in vision transformer blocks.
    """

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
        """Initialize the vision MLP.

        Args:
            dim (int): Input and output dimension.
            hidden_dim (int): Hidden layer dimension.
            hidden_act (str): Activation function name (e.g., 'gelu', 'silu').
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            dim,
            hidden_dim,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.act = ACT2FN[hidden_act]
        self.fc2 = RowParallelLinear(
            hidden_dim,
            dim,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, x: Array) -> Array:
        """Apply feedforward transformation.

        Args:
            x (Array): Input tensor of shape (seq_len, dim).

        Returns:
            Array: Transformed tensor of shape (seq_len, dim).
        """
        return self.fc2(self.act(self.fc1(x)))


class Qwen2VLVisionAttention(UnifiedAttention):
    """Self-attention layer for vision patches with 2D rotary position encoding.

    Implements multi-head self-attention for the vision encoder with spatial
    rotary embeddings that encode 2D patch positions (height, width).
    """

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
        """Initialize vision attention layer.

        Args:
            config: Vision configuration object.
            layer_idx (int): Index of this layer in the encoder stack.
            dim (int): Hidden dimension for attention.
            num_heads (int, optional): Number of attention heads. Defaults to 16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            rngs (nn.Rngs): Random number generator state.
        """
        self.embed_dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        class ConfigAdapter:
            def __init__(self, config, dim, num_heads):
                self.hidden_size = dim
                self.num_attention_heads = num_heads
                self.num_key_value_heads = num_heads
                self.head_dim = dim // num_heads
                self.attention_bias = True
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
        """Define network layers for vision attention.

        Creates fused QKV projection and output projection layers for
        efficient vision attention computation.

        Args:
            config: Configuration object with model parameters.
            dtype (jnp.dtype): Data type for computation.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): Numerical precision.
            rngs (nn.Rngs): Random number generator state.
        """
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
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def _create_attention_performer(self, config, rngs: nn.Rngs):
        """Create attention performer for vision attention.

        Args:
            config: Configuration object.
            rngs (nn.Rngs): Random number generator state.

        Returns:
            FlexibleAttentionModule: Configured attention module with vanilla attention
                and no KV caching (not needed for vision encoder).
        """
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
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        cu_seqlens: Array,
        position_embeddings: tuple[Array, Array] | None = None,
    ) -> Array:
        """Apply vision self-attention with rotary embeddings.

        Computes multi-head self-attention over vision patches with 2D rotary
        position embeddings. Processes each image/video segment separately.

        Args:
            hidden_states (Array): Input patches of shape (seq_len, hidden_dim).
            cu_seqlens (Array): Cumulative sequence lengths for segment boundaries.
            position_embeddings (tuple[Array, Array] | None, optional): Precomputed
                (cos, sin) rotary embeddings. Defaults to None.

        Returns:
            Array: Attention output of shape (seq_len, hidden_dim).
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
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        # Match HF: process each segment defined by cu_seqlens separately
        q = q.swapaxes(0, 1)[None, ...]  # (1, num_heads, seq, head_dim)
        k = k.swapaxes(0, 1)[None, ...]
        v = v.swapaxes(0, 1)[None, ...]
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        if lengths.ndim == 0:
            lengths = lengths[None]
        splits = [jnp.split(tensor, jnp.cumsum(lengths)[:-1], axis=2) for tensor in (q, k, v)]

        attn_outputs = []
        for q_chunk, k_chunk, v_chunk in zip(*splits, strict=False):
            attn_weights = jnp.matmul(q_chunk, k_chunk.swapaxes(-1, -2)) / math.sqrt(self.head_dim)
            attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(q_chunk.dtype)
            attn_output = jnp.matmul(attn_weights, v_chunk)
            attn_output = attn_output.swapaxes(1, 2)  # (1, seq_chunk, num_heads, head_dim)
            attn_outputs.append(attn_output)

        attn_output = jnp.concatenate(attn_outputs, axis=1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2VLVisionBlock(nn.Module):
    """Vision transformer block combining self-attention and MLP with pre-normalization.

    Implements a standard ViT block architecture with pre-layer normalization,
    multi-head self-attention with 2D rotary embeddings, and an MLP feedforward network.
    """

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
        """Initialize vision transformer block.

        Args:
            config (Qwen2VLVisionConfig): Vision encoder configuration.
            layer_idx (int): Index of this block in the encoder stack.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__()
        self.norm1 = LayerNorm(
            config.embed_dim,
            epsilon=1e-6,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.norm2 = LayerNorm(
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

    def __call__(self, hidden_states, cu_seqlens, position_embeddings) -> Array:
        """Forward pass through the vision block.

        Applies pre-normalization architecture: x + attn(norm(x)) followed by x + mlp(norm(x)).

        Args:
            hidden_states (Array): Input patches of shape (seq_len, hidden_dim).
            cu_seqlens (Array): Cumulative sequence lengths for segment boundaries.
            position_embeddings (tuple[Array, Array]): Precomputed (cos, sin) rotary embeddings.

        Returns:
            Array: Output patches of shape (seq_len, hidden_dim).
        """
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2VLMLP(nn.Module):
    """Multi-Layer Perceptron module for Qwen2-VL language decoder.

    Implements a SwiGLU feedforward network for the language model decoder layers,
    with gate, up, and down projections for enhanced representation learning.
    """

    def __init__(
        self,
        config: Qwen2VLTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        """Initialize Qwen2-VL language decoder MLP.

        Args:
            config (Qwen2VLTextConfig): Text model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the decoder stack.
        """
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
        )
        row_parallel_linear = partial(
            RowParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
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
        """Apply SwiGLU feedforward transformation.

        Args:
            hidden_states (Array): Input tensor of shape (batch, seq_len, hidden_dim).

        Returns:
            jnp.ndarray: Transformed tensor of shape (batch, seq_len, hidden_dim).
        """
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
    """Causal self-attention layer for Qwen2-VL language decoder.

    Implements multi-head causal attention with multi-dimensional RoPE (M-RoPE)
    that encodes 3D positions (temporal, height, width) for vision-language alignment.
    Supports sliding window attention when configured.
    """

    def __init__(
        self,
        config: Qwen2VLTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        """Initialize Qwen2-VL language decoder attention.

        Args:
            config (Qwen2VLTextConfig): Text model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the decoder stack.
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
            sliding_window=config.sliding_window if config.use_sliding_window else None,
        )

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"] | Int[Array, "3 batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ) -> AttentionLayerOutput:
        """Apply causal attention with M-RoPE position embeddings.

        Args:
            hidden_states (Array): Input tensor of shape (batch, seq_len, hidden_dim).
            mask_info (MaskInfo | None): Attention mask information.
            position_ids (Array): Position IDs, either 2D (batch, seq_len) or 3D
                (3, batch, seq_len) for M-RoPE temporal/height/width encoding.
            mode (RUNTIME_MODE_TYPES): Runtime mode (train, decode, etc.).
            cache_view (TransformerCacheView | RaggedPagesCacheView | None, optional):
                Cache view for KV caching. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache operations. Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights.
                Defaults to False.
            frequencies (Array | None, optional): Precomputed RoPE frequencies.
                Defaults to None.

        Returns:
            AttentionLayerOutput: Contains attention output, optional attention weights,
                and updated cache view.
        """
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
    """Single decoder layer for Qwen2-VL language model.

    Combines causal self-attention with M-RoPE position encoding and SwiGLU
    feedforward networks, using RMS normalization and residual connections.
    """

    def __init__(
        self,
        config: Qwen2VLTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        """Initialize Qwen2-VL decoder layer.

        Args:
            config (Qwen2VLTextConfig): Text model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the decoder stack.
        """
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
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ):
        """Forward pass through the decoder layer.

        Applies pre-normalization architecture: x + attn(norm(x)) followed by x + mlp(norm(x)).

        Args:
            hidden_states (Array): Input tensor of shape (batch, seq_len, hidden_dim).
            mask_info (MaskInfo | None): Attention mask information including causal masks.
            position_ids (Array): Position indices for M-RoPE encoding, shape (3, batch, seq_len)
                or (batch, seq_len).
            mode (RUNTIME_MODE_TYPES): Runtime mode (train, decode, etc.).
            cache_view (TransformerCacheView | RaggedPagesCacheView | None, optional):
                Cache view for KV caching. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache operations. Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights.
                Defaults to False.
            frequencies (Array | None, optional): Precomputed RoPE frequencies.
                Defaults to None.

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


@register_module(TaskType.BASE_VISION, config=Qwen2VLVisionConfig, model_type="qwen2_vl")
class Qwen2VLVisionTransformer(EasyDeLBaseModule):
    """Vision transformer encoder for extracting image and video features.

    Implements a vision transformer architecture with 2D rotary position embeddings
    and patch merging for efficient visual feature extraction in Qwen2-VL.

    Attributes:
        config (Qwen2VLVisionConfig): Vision encoder configuration.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.
    """

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
        """Initialize the vision transformer.

        Args:
            config (Qwen2VLVisionConfig): Vision encoder configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.

        Raises:
            AssertionError: If config is not a Qwen2VLVisionConfig instance.
        """
        assert isinstance(config, Qwen2VLVisionConfig), (
            "Qwen2VLVisionTransformer requires a Qwen2VLVisionConfig instance"
        )
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

        self.blocks = nn.List(
            [
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
        )

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
        """Get the data type of the model parameters.

        Returns:
            jnp.dtype: The data type of the MLP kernel parameters.
        """
        return self.blocks[0].mlp.fc2.kernel.value.dtype

    def rot_pos_emb(self, grid_thw, max_grid_size):
        """Compute 2D rotary position embeddings for vision patches.

        Creates position embeddings that encode spatial (height, width) positions
        for each patch, accounting for spatial merge operations.

        Args:
            grid_thw (Array): Grid dimensions of shape (num_images/videos, 3) where
                each row contains (temporal, height, width) in patches.
            max_grid_size (int): Maximum grid size for computing rotary frequencies.

        Returns:
            jnp.ndarray: Rotary position embeddings of shape (total_patches, head_dim).
        """
        pos_ids = []

        for t, h, w in grid_thw:
            # Convert to Python ints for reshape operations
            t, h, w = int(t), int(h), int(w)

            hpos_ids = jnp.arange(h)
            hpos_ids = jnp.expand_dims(hpos_ids, 1)
            hpos_ids = jnp.broadcast_to(hpos_ids, (h, w))

            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = jnp.transpose(hpos_ids, (0, 2, 1, 3))
            hpos_ids = hpos_ids.flatten()

            wpos_ids = jnp.arange(w)
            wpos_ids = jnp.expand_dims(wpos_ids, 0)
            wpos_ids = jnp.broadcast_to(wpos_ids, (h, w))

            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = jnp.transpose(wpos_ids, (0, 2, 1, 3))
            wpos_ids = wpos_ids.flatten()

            stacked = jnp.stack([hpos_ids, wpos_ids], axis=-1)
            repeated = jnp.tile(stacked, (t, 1))
            pos_ids.append(repeated)

        pos_ids = jnp.concatenate(pos_ids, axis=0)
        rotary_pos_emb_full = jnp.outer(
            jnp.arange(0, max_grid_size, dtype="f4"),
            1.0 / (10000 ** (jnp.arange(0, self._head_dim_ro, 2, dtype="f4") / self._head_dim_ro)),
        )
        rotary_pos_emb = jnp.take(rotary_pos_emb_full, pos_ids, axis=0)
        rotary_pos_emb = rotary_pos_emb.reshape(pos_ids.shape[0], -1)

        return rotary_pos_emb

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        grid_thw: Array,
        max_grid_size,
    ) -> Array:
        """Forward pass through the vision transformer.

        Encodes image or video pixel values into visual features suitable for
        fusion with the language model.

        Args:
            hidden_states (Array): Flattened pixel values ready for patch embedding.
            grid_thw (Array): Grid dimensions of shape (num_images/videos, 3) where
                each row contains (temporal, height, width) in patches.
            max_grid_size: Maximum grid size for computing rotary position embeddings.

        Returns:
            Array: Visual features of shape (num_merged_patches, hidden_size) after
                patch merging, ready for language model fusion.
        """
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw, max_grid_size)

        # Match HuggingFace: concatenate and compute cos/sin
        emb = jnp.concatenate([rotary_pos_emb, rotary_pos_emb], axis=-1)
        position_embeddings = (jnp.cos(emb), jnp.sin(emb))

        grid_lens = grid_thw[:, 1] * grid_thw[:, 2]
        repeated = jnp.repeat(grid_lens, grid_thw[:, 0])
        cu_seqlens = jnp.cumsum(repeated, dtype="i4")
        cu_seqlens = jnp.pad(cu_seqlens, (1, 0), constant_values=0)
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
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


@register_module(TaskType.BASE_MODULE, config=Qwen2VLTextConfig, model_type="qwen2_vl")
class Qwen2VLTextModel(EasyDeLBaseModule):
    """Language decoder stack for Qwen2-VL that processes text and projected vision tokens.

    Implements the transformer decoder backbone for the Qwen2-VL model, processing
    interleaved text and vision embeddings using M-RoPE position encoding for
    multimodal understanding.

    Attributes:
        config (Qwen2VLTextConfig): Text model configuration.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.
    """

    def __init__(
        self,
        config: Qwen2VLTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize the Qwen2-VL text model.

        Args:
            config (Qwen2VLTextConfig): Text model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.

        Raises:
            AssertionError: If config is not a Qwen2VLTextConfig instance.
        """
        assert isinstance(config, Qwen2VLTextConfig), "Qwen2VLTextModel requires a Qwen2VLTextConfig instance"
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.embed_tokens = Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            rngs=rngs,
        )

        self.layers = nn.List(
            [
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
        )
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
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        """Forward pass through the Qwen2-VL text model.

        Processes input tokens (or embeddings) through all decoder layers with
        M-RoPE position encoding and RMS normalization.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch, seq_len).
                Must be provided if inputs_embeds is None.
            inputs_embeds (Array | None, optional): Pre-computed embeddings of shape
                (batch, seq_len, hidden_size). Typically includes fused vision tokens.
            attention_mask (Array | None, optional): Attention mask of shape (batch, seq_len).
                Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information. Defaults to None.
            position_ids (Array | None, optional): 3D position IDs for M-RoPE of shape
                (3, batch, seq_len). Auto-computed if None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode. Auto-detected if None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cached KV states for generation. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Cache metadata. Defaults to None.
            apply_lm_head (bool, optional): Unused parameter for API compatibility.
                Defaults to True.
            output_attentions (bool | None, optional): Whether to return attention weights.
                Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return all hidden states.
                Defaults to None.

        Returns:
            BaseModelOutput: Contains last_hidden_state, optional hidden_states,
                optional attentions, and past_key_values.

        Raises:
            ValueError: If both input_ids and inputs_embeds are provided or both are None.
            AssertionError: If sequence length exceeds max_position_embeddings.
        """
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

    def set_embeddings(self, value):
        """
        Sets the input embedding layer of the module.
        """
        self.embed_tokens = value


@register_module(TaskType.BASE_MODULE, config=Qwen2VLConfig, model_type="qwen2_vl")
class Qwen2VLModel(EasyDeLBaseModule):
    """Qwen2-VL base model combining vision encoder and language decoder.

    This model integrates a vision transformer encoder for processing images and videos
    with a language model decoder, enabling multimodal understanding through M-RoPE
    position encoding and vision-text embedding fusion.

    Attributes:
        config (Qwen2VLConfig): Combined model configuration.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.
        visual (Qwen2VLVisionTransformer): Vision encoder component.
        language_model (Qwen2VLTextModel): Language decoder component.
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
        """Initialize the Qwen2-VL base model.

        Args:
            config (Qwen2VLConfig): Combined model configuration containing
                vision_config and text_config.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
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
            config.text_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.rope_deltas = None

    def set_embeddings(self, value):
        """Set the token embedding layer.

        Args:
            value: New embedding layer to set.
        """
        self.language_model.set_embeddings(value)

    def get_video_features(self, pixel_values_videos: Array, video_grid_thw: Array | None = None):
        """Extract visual features from video input.

        Args:
            pixel_values_videos (Array): Video pixel values of shape
                (num_videos, channels, num_frames, height, width).
            video_grid_thw (Array | None, optional): Video grid dimensions of shape
                (num_videos, 3) where each row is (temporal, height, width).
                Defaults to None.

        Returns:
            list[Array] | Array: Video embeddings, either as a list of per-video
                embeddings or concatenated if video_grid_thw is None.
        """
        pixel_values_videos = pixel_values_videos.astype(self.visual.dtype)
        max_grid_size = int(video_grid_thw[:, 1:].max()) if video_grid_thw is not None else 1
        video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw, max_grid_size=max_grid_size)
        if video_grid_thw is not None:
            split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
            indices = np.cumsum(split_sizes)[:-1]
            video_embeds = jnp.split(video_embeds, indices)
        return video_embeds

    def get_image_features(self, pixel_values: Array, image_grid_thw: Array | None = None):
        """Extract visual features from image input.

        Args:
            pixel_values (Array): Image pixel values of shape
                (num_images, channels, height, width).
            image_grid_thw (Array | None, optional): Image grid dimensions of shape
                (num_images, 3) where each row is (1, height, width).
                Defaults to None.

        Returns:
            list[Array] | Array: Image embeddings, either as a list of per-image
                embeddings or concatenated if image_grid_thw is None.
        """
        pixel_values = pixel_values.astype(self.visual.dtype)
        max_grid_size = int(image_grid_thw[:, 1:].max()) if image_grid_thw is not None else 1
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw, max_grid_size=max_grid_size)
        if image_grid_thw is not None:
            split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
            indices = np.cumsum(split_sizes)[:-1]
            image_embeds = jnp.split(image_embeds, indices)
        return image_embeds

    def get_rope_index(
        self,
        input_ids: Array = None,
        image_grid_thw: Array = None,
        video_grid_thw: Array = None,
        attention_mask: Array = None,
    ):
        """Compute 3D M-RoPE position indices for multimodal inputs.

        Calculates position indices that encode temporal, height, and width
        dimensions for vision tokens, enabling proper position encoding
        for images and videos in the language model.

        Args:
            input_ids (Array, optional): Input token IDs of shape (batch, seq_len).
            image_grid_thw (Array, optional): Image grid dimensions of shape
                (num_images, 3) where each row is (1, height, width).
            video_grid_thw (Array, optional): Video grid dimensions of shape
                (num_videos, 3) where each row is (temporal, height, width).
            attention_mask (Array, optional): Attention mask of shape (batch, seq_len).

        Returns:
            tuple[Array, Array]: Position IDs of shape (3, batch, seq_len) and
                rope_deltas of shape (batch, 1).
        """
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

    def compute_embedding(
        self,
        input_ids: Int[Array, "batch seq_len"],
        *,
        pixel_values: Array | None = None,
        pixel_values_videos: Array | None = None,
        image_grid_thw: Array | None = None,
        video_grid_thw: Array | None = None,
        image_embeds: Array | None = None,
        video_embeds: Array | None = None,
        **kwargs,
    ) -> Array:
        """Compute input embeddings with vision tokens merged in.

        Creates input embeddings by combining text token embeddings with
        vision embeddings at placeholder token positions.

        Args:
            input_ids (Array): Input token IDs of shape (batch, seq_len).
            pixel_values (Array | None, optional): Image pixel values. Defaults to None.
            pixel_values_videos (Array | None, optional): Video pixel values. Defaults to None.
            image_grid_thw (Array | None, optional): Image grid dimensions. Defaults to None.
            video_grid_thw (Array | None, optional): Video grid dimensions. Defaults to None.
            image_embeds (Array | None, optional): Pre-computed image embeddings.
                Defaults to None.
            video_embeds (Array | None, optional): Pre-computed video embeddings.
                Defaults to None.
            **kwargs: Additional arguments (unused).

        Returns:
            Array: Combined embeddings of shape (batch, seq_len, hidden_size) with
                vision embeddings inserted at placeholder token positions.

        Raises:
            ValueError: If input_ids is None.
        """
        if input_ids is None:
            raise ValueError("`input_ids` must be provided when calling `compute_embedding`.")

        inputs_embeds = super().compute_embedding(input_ids)

        if image_embeds is None and pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            if isinstance(image_embeds, (list, tuple)):
                image_embeds = jnp.concatenate(image_embeds, axis=0)

        if image_embeds is not None:
            inputs_embeds = merge_multimodal_embeddings(
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
            inputs_embeds = merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=video_embeds.astype(inputs_embeds.dtype),
                placeholder_token_id=self.config.video_token_id,
            )

        return inputs_embeds

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
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
    ):
        """Forward pass through the Qwen2-VL base model.

        Processes multimodal inputs by encoding vision features and fusing them
        with text embeddings, then passing through the language model.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch, seq_len).
                Defaults to None.
            attention_mask (Array | None, optional): Attention mask. Defaults to None.
            position_ids (Array | None, optional): 3D position IDs for M-RoPE.
                Auto-computed if None. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cached KV states. Defaults to None.
            inputs_embeds (Array | None, optional): Pre-computed embeddings.
                Defaults to None.
            use_cache (bool | None, optional): Whether to return cache. Defaults to None.
            output_attentions (bool | None, optional): Return attention weights.
                Defaults to None.
            output_hidden_states (bool | None, optional): Return hidden states.
                Defaults to None.
            return_dict (bool | None, optional): Return dict vs tuple. Defaults to None.
            pixel_values (Array | None, optional): Image pixels. Defaults to None.
            pixel_values_videos (Array | None, optional): Video pixels. Defaults to None.
            image_grid_thw (Array | None, optional): Image grid dimensions. Defaults to None.
            video_grid_thw (Array | None, optional): Video grid dimensions. Defaults to None.
            rope_deltas (Array | None, optional): Position deltas. Defaults to None.
            cache_position (Array | None, optional): Cache positions. Defaults to None.
            mask_info (MaskInfo | None, optional): Mask information. Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Cache metadata. Defaults to None.
            **kwargs: Additional arguments (unused).

        Returns:
            BaseModelOutput: Contains last_hidden_state, past_key_values,
                hidden_states, and attentions.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.compute_embedding(
                input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
            )

        if position_ids is None:
            position_ids, _rope_deltas = self.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw if pixel_values is not None else None,
                video_grid_thw=video_grid_thw if pixel_values_videos is not None else None,
                attention_mask=attention_mask,
            )
        elif position_ids.ndim == 2:
            batch_size, seq_length = position_ids.shape
            position_ids = jnp.broadcast_to(position_ids[None, :, :], (3, batch_size, seq_length))

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

    def get_embedding(self):
        """Returns the embedding layer of the module."""
        return self.language_model.get_embedding()

    def get_decoder(self):
        """Returns the decoder part of the model."""
        return self.language_model


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

    _task_type = TaskType.IMAGE_TEXT_TO_TEXT
    _model_type = "qwen2_vl"
    _config_class = Qwen2VLConfig
    _auto_register = False
    _supports_video = True
    _uses_mrope = True

    _vision_tower_name = "visual"
    _projector_name = "merger"
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
        """Initialize the Qwen2-VL model for conditional generation.

        Args:
            config (Qwen2VLConfig): Combined model configuration containing
                vision_config and text_config.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=Qwen2VLModel,
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
        self.vocab_size = config.vocab_size

    @property
    def visual(self):
        """Access the vision transformer component.

        Returns:
            Qwen2VLVisionTransformer: The vision encoder component for backward compatibility.
        """
        return self.base_model.visual

    def get_embedding(self):
        """Get the token embedding layer.

        Returns:
            Embed: The token embedding layer from the language model.
        """
        return self.base_model.get_embedding()

    def set_embeddings(self, value):
        """Set the token embedding layer.

        Args:
            value: New embedding layer to set.
        """
        self.base_model.set_embeddings(value)

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

    def compute_embedding(self, input_ids, *args, **kwargs):
        """Compute input embeddings with vision tokens merged in.

        Delegates to the base model's compute_embedding method.

        Args:
            input_ids: Input token IDs.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments including pixel_values.

        Returns:
            Array: Combined embeddings with vision tokens inserted.
        """
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
        image_grid_thw: tuple | None = None,
        video_grid_thw: tuple | None = None,
        rope_deltas: Array | None = None,
        cache_position: Array | None = None,
        **kwargs,
    ) -> VLMCausalLMOutput:
        """Forward pass for Qwen2-VL conditional generation model.

        This method processes multimodal inputs (text, images, and/or videos) and generates
        language model outputs. Vision inputs are encoded via the vision encoder and fused
        with text embeddings using placeholder tokens.

        Args:
            input_ids (Int[Array, "batch seq_len"] | None): Input token IDs of shape (batch_size, sequence_length).
                Should contain special vision tokens (image_token_id, video_token_id) at positions where
                visual content should be inserted. Defaults to None.
            attention_mask (Bool[Array, "batch seq_len"] | None): Attention mask of shape (batch_size, sequence_length).
                Defaults to None.
            mask_info (MaskInfo | None): Precomputed mask information. Overrides attention_mask if provided.
                Defaults to None.
            position_ids (Int[Array, "3 batch seq_len"] | None): 3D position IDs for multi-dimensional RoPE
                of shape (3, batch_size, sequence_length) encoding [temporal, height, width] dimensions.
                Auto-computed from image/video grids if None. Defaults to None.
            mode (common_types.RUNTIME_MODE_TYPES | None): Runtime mode controlling attention implementation.
                Defaults to None (auto-inferred).
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None): Cached key-value states
                from previous forward passes for efficient generation. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None): Metadata for
                paged attention operations. Defaults to None.
            apply_lm_head (bool): Whether to apply the language modeling head to produce logits. Defaults to True.
            inputs_embeds (Float[Array, "batch seq_len hidden_dim"] | None): Pre-computed text embeddings of shape
                (batch_size, sequence_length, hidden_size). Mutually exclusive with input_ids. Defaults to None.
            output_attentions (bool | None): Whether to return attention weights from all layers. Defaults to None.
            output_hidden_states (bool | None): Whether to return hidden states from all layers. Defaults to None.
            pixel_values (Array | None): Image pixel values of shape (num_images, channels, height, width) where
                channels=3 (RGB). Images are patchified and encoded by the vision encoder. Defaults to None.
            pixel_values_videos (Array | None): Video pixel values of shape
                (num_videos, channels, num_frames, height, width). Defaults to None.
            image_grid_thw (tuple | None): Image grid dimensions as array of shape (num_images, 3) where each row
                contains [temporal, height_patches, width_patches]. Used for computing M-RoPE positions. Defaults to None.
            video_grid_thw (tuple | None): Video grid dimensions as array of shape (num_videos, 3) where each row
                contains [temporal_frames, height_patches, width_patches]. Defaults to None.
            rope_deltas (Array | None): Position deltas for M-RoPE of shape (batch_size, 1). Represents offset
                added to position indices due to variable-length vision sequences. Defaults to None.
            cache_position (Array | None): Explicit cache position indices for generation. Defaults to None.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            VLMCausalLMOutput: Named tuple containing:
                - logits (Array | None): Language modeling logits of shape (batch_size, sequence_length, vocab_size)
                  if apply_lm_head=True, otherwise None.
                - past_key_values (TransformerCache | RaggedPagesCache | HybridCache): Updated KV cache.
                - hidden_states (tuple[Array, ...] | None): Hidden states from all layers if output_hidden_states=True.
                - last_hidden_state (Array): Final hidden state of shape (batch_size, sequence_length, hidden_size).
                - attentions (tuple[Array, ...] | None): Attention weights from all layers if output_attentions=True.
                - rope_deltas (Array | None): Position deltas used for M-RoPE.

        Note:
            The model uses special token IDs to determine where to insert vision embeddings:
            - vision_start_token_id: Marks the beginning of a vision sequence
            - image_token_id or video_token_id: Indicates the type of vision content
            - vision_end_token_id: Marks the end of a vision sequence
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
        rope_deltas=None,
        **kwargs,
    ):
        """Prepare model inputs for autoregressive generation.

        Sets up the cache, attention mask, position IDs, and other inputs
        needed for efficient autoregressive generation.

        Args:
            input_ids: Input token IDs.
            max_length (int): Maximum sequence length for generation.
            pad_token_id (int): Padding token ID.
            starts (int | None, optional): Prefill length. Defaults to None.
            past_key_values: Cached KV states. Defaults to None.
            attention_mask: Attention mask. Defaults to None.
            mask_info: Mask information. Defaults to None.
            inputs_embeds: Pre-computed embeddings. Defaults to None.
            position_ids: Position IDs. Defaults to None.
            pixel_values: Image pixel values. Defaults to None.
            pixel_values_videos: Video pixel values. Defaults to None.
            image_grid_thw: Image grid dimensions. Defaults to None.
            video_grid_thw: Video grid dimensions. Defaults to None.
            rope_deltas: Position deltas for M-RoPE. Defaults to None.
            **kwargs: Additional arguments.

        Returns:
            dict: Model inputs dictionary for generation.
        """
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

        # Note: Don't include input_ids in model_inputs - it will be passed as a positional arg
        if inputs_embeds is not None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {}

        if attention_mask is None:
            attn_2d = jnp.ones((batch_size, _seq_length), dtype="b1")
        else:
            attn_2d = attention_mask

        extended_attention_mask = jnp.ones((batch_size, max_length), dtype=attn_2d.dtype)
        extended_attention_mask = jax.lax.dynamic_update_slice(extended_attention_mask, attn_2d, (0, 0))

        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids if inputs_embeds is None else None,
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
        )

        # Note: get_rope_index uses numpy and is NOT JIT-compatible (uses data-dependent control flow)
        if position_ids is not None and position_ids.ndim == 2:
            position_ids = jnp.broadcast_to(position_ids[None, :, :], (3, batch_size, _seq_length))
            if rope_deltas is None:
                max_pos = jnp.max(position_ids)
                rope_deltas = jnp.full((batch_size, 1), max_pos + 1 - _seq_length, dtype=jnp.int32)
        elif position_ids is None:
            if image_grid_thw is not None or video_grid_thw is not None:
                position_ids, rope_deltas = self.base_model.get_rope_index(
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attn_2d,
                )
            elif attn_2d is not None:
                attn_mask_int = attn_2d.astype(jnp.int32)
                position_ids = jnp.cumsum(attn_mask_int, axis=-1) - 1
                position_ids = jnp.where(attn_mask_int == 0, 1, position_ids)
                position_ids = jnp.expand_dims(position_ids, axis=0)
                position_ids = jnp.broadcast_to(position_ids, (3, batch_size, _seq_length))
                max_pos = jnp.max(position_ids)
                rope_deltas = jnp.full((batch_size, 1), max_pos + 1 - _seq_length, dtype=jnp.int32)
            else:
                position_ids = jnp.broadcast_to(
                    jnp.arange(_seq_length, dtype=jnp.int32)[None, None, :],
                    (3, batch_size, _seq_length),
                )
                rope_deltas = jnp.zeros((batch_size, 1), dtype=jnp.int32)

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
                "rope_deltas": rope_deltas,
            }
        )
        return model_inputs

    def _create_required_props_from_kwargs(
        self,
        model_kwargs: dict[str, Array],
    ) -> tp.Mapping[str, dict[str, tp.Any]] | None:
        """Create required properties from model kwargs for generation.

        Extracts grid dimensions that need to be preserved across generation steps.

        Args:
            model_kwargs (dict[str, Array]): Model keyword arguments.

        Returns:
            tp.Mapping[str, dict[str, tp.Any]] | None: Dictionary of properties
                to preserve, or None if no relevant properties found.
        """
        basics = {}
        if "image_grid_thw" in model_kwargs.keys():
            basics.update({"image_grid_thw": {"value": jnp.array(model_kwargs["image_grid_thw"])}})
        if "video_grid_thw" in model_kwargs.keys():
            basics.update({"video_grid_thw": {"value": jnp.array(model_kwargs["video_grid_thw"])}})
        return basics

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        """Update model inputs for the next generation step.

        Updates cached KV states, position IDs, and removes vision inputs
        that only need to be processed once.

        Args:
            model_outputs: Outputs from the previous generation step.
            model_kwargs: Current model keyword arguments.

        Returns:
            dict: Updated model keyword arguments for the next step.
        """
        model_kwargs["past_key_values"] = model_outputs.past_key_values

        if hasattr(model_outputs, "rope_deltas") and model_outputs.rope_deltas is not None:
            model_kwargs["rope_deltas"] = model_outputs.rope_deltas

        if model_kwargs.get("position_ids") is not None:
            position_ids = model_kwargs["position_ids"]
            if position_ids.ndim == 3:
                model_kwargs["position_ids"] = position_ids[:, :, -1:] + 1
            else:
                model_kwargs["position_ids"] = position_ids[:, -1:] + 1

        model_kwargs.pop("pixel_values", None)
        model_kwargs.pop("pixel_values_videos", None)
        model_kwargs.pop("token_type_ids", None)
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
        image_grid_thw: Array | None = None,
        video_grid_thw: Array | None = None,
        drop_ids: bool = True,
        **others,
    ):
        """Prepare inputs with M-RoPE position IDs computed from grid shapes.

        Computes 3D position IDs for multi-dimensional RoPE based on the
        provided image and video grid dimensions.

        Args:
            image_grid_thw (Array | None, optional): Image grid dimensions.
                Defaults to None.
            video_grid_thw (Array | None, optional): Video grid dimensions.
                Defaults to None.
            drop_ids (bool, optional): Whether to drop input_ids from result.
                Defaults to True.
            **others: Additional arguments to include in the result.

        Returns:
            dict: Prepared inputs including computed position_ids, rope_deltas,
                and mask_info.
        """
        attention_mask = others.get("attention_mask", None)
        mask_info = others.get("mask_info", None)
        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=others.get("input_ids"),
            inputs_embeds=others.get("inputs_embeds"),
            attention_mask=attention_mask,
        )
        attention_mask = mask_info.attention_mask

        input_ids = others.get("input_ids", None)
        rope_deltas = others.get("rope_deltas", None)
        position_ids = others.get("position_ids", None)

        if position_ids is None and input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            position_ids, rope_deltas = self.base_model.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
        elif position_ids is not None and position_ids.ndim == 2:
            batch_size, sequence_length = position_ids.shape
            position_ids = jnp.expand_dims(position_ids, axis=0)
            position_ids = jnp.broadcast_to(position_ids, (3, batch_size, sequence_length))
            if rope_deltas is None:
                max_pos = jnp.max(position_ids)
                rope_deltas = jnp.full((batch_size, 1), max_pos + 1 - sequence_length, dtype=jnp.int32)

        # Note: get_rope_index uses numpy and is NOT JIT-compatible (uses data-dependent control flow)
        elif (
            position_ids is None
            and others.get("input_ids", None) is not None
            and (attention_mask is None or attention_mask.ndim == 2)
        ):
            input_ids = others.get("input_ids")
            batch_size, sequence_length = input_ids.shape

            if attention_mask is not None and attention_mask.ndim == 2:
                attn_mask_int = attention_mask.astype(jnp.int32)
                position_ids = jnp.cumsum(attn_mask_int, axis=-1) - 1
                position_ids = jnp.where(attn_mask_int == 0, 1, position_ids)
                position_ids = jnp.expand_dims(position_ids, axis=0)
                position_ids = jnp.broadcast_to(position_ids, (3, batch_size, sequence_length))
                if rope_deltas is None:
                    max_pos = jnp.max(position_ids)
                    rope_deltas = jnp.full((batch_size, 1), max_pos + 1 - sequence_length, dtype=jnp.int32)
            else:
                position_ids = jnp.broadcast_to(
                    jnp.arange(sequence_length, dtype=jnp.int32)[None, None, :],
                    (3, batch_size, sequence_length),
                )
                if rope_deltas is None:
                    rope_deltas = jnp.zeros((batch_size, 1), dtype=jnp.int32)
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
        """Get arguments that should be treated as static for JIT compilation.

        Returns:
            tuple[str, ...]: Names of arguments that are static (grid dimensions).
        """
        return ("image_grid_thw", "video_grid_thw")
