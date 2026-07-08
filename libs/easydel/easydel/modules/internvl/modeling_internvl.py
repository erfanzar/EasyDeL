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

"""InternVL vision-language model implementation.

InternVL couples the InternViT vision tower (a timm/BEiT-style ViT with a CLS
token, learned absolute position embeddings, per-layer layer-scale multipliers
and optional full-width QK RMSNorm) with an auto-resolved decoder-only language
model (Qwen2 by default; Llama variants exist) through a pixel-shuffle
downsampling step and a LayerNorm+MLP multimodal projector. Visual features
are extracted from the vision tower, spatially downsampled by pixel shuffle
(halving each spatial dim and 4x-ing channels at the default 0.5 ratio),
projected into the LM embedding space, and merged into the text embedding
sequence at positions marked by ``image_token_id``.

Exports:
    - ``InternVLVisionModel``: the standalone InternViT tower.
    - ``InternVLMultiModalProjector``: LayerNorm + two-layer MLP bridge.
    - ``InternVLModel``: base multimodal model returning hidden states.
    - ``InternVLForConditionalGeneration``: full model with LM head.
"""

import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
import spectrax as spx
from eformer.loggings import get_logger
from eformer.pytree import auto_pytree
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int
from spectrax import apply_logical_sharding, common_types, nn

from easydel.caching import (
    HybridCache,
    OperationsMetadata,
    RaggedPagesCache,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerMetadata,
)
from easydel.infra.base_module import EasyDeLBaseModule, EasyDeLLayerStackMixin
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import (
    AttentionLayerOutput,
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ModelOutput,
    VLMCausalLMOutput,
)
from easydel.infra.utils import ACT2FN, ArrayParam, auto_remat
from easydel.layers import ColumnParallelLinear, RowParallelLinear, dense_qkv_layout
from easydel.layers.attention import AttentionModule, FlexibleAttentionModule
from easydel.layers.norms import LayerNorm, RMSNorm
from easydel.modules._base import BaseVisionLanguageModule

from ..auto.auto_modeling import AutoEasyDeLModel, AutoEasyDeLVisionModel
from .internvl_configuration import InternVLConfig, InternVLVisionConfig

logger = get_logger(__name__)


@auto_pytree
class InternVLCausalLMOutputWithPast(ModelOutput):
    """Output of the InternVL base model for autoregressive runs.

    Args:
        loss (Array | None): Language modeling loss when labels are provided.
        logits (Array | None): LM head scores (populated by the task wrapper).
        past_key_values (TransformerCache | None): Updated KV cache.
        hidden_states (tuple[Array] | None): Per-layer hidden states when requested.
        last_hidden_state (Array | None): Final decoder hidden state.
        attentions (tuple[Array] | None): Attention weights when requested.
        image_hidden_states (Array | None): Projected image features produced by
            the vision tower + projector.
    """

    loss: Array | None = None
    logits: Array = None
    past_key_values: TransformerCache | None = None
    hidden_states: tuple[Array] | None = None
    last_hidden_state: Array | None = None
    attentions: tuple[Array] | None = None
    image_hidden_states: Float[Array, "batch seq_len hidden_dim"] | None = None


class InternVLVisionPatchEmbeddings(spx.Module):
    """Conv2d patch embedder for the InternViT tower.

    Slices ``(batch, height, width, channels)`` pixels into non-overlapping
    ``patch_size`` patches with a strided convolution and flattens them into
    ``(batch, num_patches, hidden_size)`` tokens.
    """

    def __init__(
        self,
        config: InternVLVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the patch embedder.

        Args:
            config (InternVLVisionConfig): Vision tower configuration.
            dtype (jnp.dtype, optional): Computation dtype. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Parameter dtype. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Matmul precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        image_size, patch_size = config.image_size, config.patch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = config.num_channels
        self.num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])

        self.projection = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=tuple(patch_size),
            stride=tuple(patch_size),
            padding="VALID",
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def forward(self, pixel_values: Float[Array, "batch height width channels"]) -> Array:
        """Embed pixels into patch tokens.

        Args:
            pixel_values: NHWC pixel tensor of shape (batch, height, width, channels).

        Returns:
            Array: Patch embeddings of shape (batch, num_patches, hidden_size).
        """
        embeddings = self.projection(pixel_values)
        batch_size, height, width, channels = embeddings.shape
        return jnp.reshape(embeddings, (batch_size, height * width, channels))


class InternVLVisionEmbeddings(EasyDeLLayerStackMixin, spx.Module):
    """CLS token + patch + absolute position embeddings for InternViT.

    Follows the timm ViT input contract: patch tokens from the conv embedder,
    a learnable CLS token prepended, optional mask-token substitution for
    masked-image-modeling inputs, and learned absolute position embeddings
    (bicubically interpolated when the input grid differs from the trained
    grid).
    """

    def __init__(
        self,
        config: InternVLVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize InternVL vision embeddings.

        Args:
            config (InternVLVisionConfig): Vision tower configuration.
            dtype (jnp.dtype, optional): Computation dtype. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Parameter dtype. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Matmul precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype

        self.cls_token = ArrayParam.bound(
            shape=(1, 1, config.hidden_size),
            dtype=param_dtype,
            init_method="zeros",
            key=rngs.parameters,
        )
        if config.use_mask_token:
            self.mask_token = ArrayParam.bound(
                shape=(1, 1, config.hidden_size),
                dtype=param_dtype,
                init_method="zeros",
                key=rngs.parameters,
            )
        else:
            self.mask_token = None

        with self.assign_layer_stage(0, total_layers=config.num_hidden_layers):
            self.patch_embeddings = InternVLVisionPatchEmbeddings(
                config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        num_patches = self.patch_embeddings.num_patches
        if config.use_absolute_position_embeddings:
            self.position_embeddings = ArrayParam.bound(
                shape=(1, num_patches + 1, config.hidden_size),
                dtype=param_dtype,
                init_method="zeros",
                key=rngs.parameters,
            )
        else:
            self.position_embeddings = None

    def interpolate_pos_encoding(self, embeddings: Array, height: int, width: int) -> Array:
        """Interpolate trained position encodings to a new spatial grid.

        Mirrors the HF/timm behavior: the CLS position is kept and the patch
        grid is resized bicubically. Note ``jax.image.resize`` bicubic uses a
        slightly different cubic kernel than ``torch.nn.functional.interpolate``;
        this path only fires for non-square or resolution-mismatched inputs.

        Args:
            embeddings: Token embeddings of shape (batch, num_patches + 1, dim).
            height: Input image height in pixels.
            width: Input image width in pixels.

        Returns:
            Array: Position embeddings of shape (1, num_patches + 1, dim).
        """
        position_embeddings = self.position_embeddings.value
        num_patches = embeddings.shape[1] - 1
        num_positions = position_embeddings.shape[1] - 1

        if num_patches == num_positions and height == width:
            return position_embeddings

        class_pos_embed = position_embeddings[:, :1]
        patch_pos_embed = position_embeddings[:, 1:]
        dim = embeddings.shape[-1]

        new_height = height // self.patch_size[0]
        new_width = width // self.patch_size[1]
        sqrt_num_positions = int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = jax.image.resize(
            patch_pos_embed.astype(jnp.float32),
            shape=(1, new_height, new_width, dim),
            method="bicubic",
        ).astype(position_embeddings.dtype)
        patch_pos_embed = patch_pos_embed.reshape(1, -1, dim)
        return jnp.concatenate([class_pos_embed, patch_pos_embed], axis=1)

    def forward(
        self,
        pixel_values: Float[Array, "batch height width channels"],
        bool_masked_pos: Bool[Array, "batch num_patches"] | None = None,
    ) -> Array:
        """Compute the ViT input embedding sequence.

        Args:
            pixel_values: NHWC pixel tensor of shape (batch, height, width, channels).
            bool_masked_pos: Optional boolean mask of masked patches (masked
                image modeling); masked positions are replaced by the mask token.

        Returns:
            Array: Embeddings of shape (batch, num_patches + 1, hidden_size).
        """
        batch_size, height, width, _ = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)
        seq_len = embeddings.shape[1]

        if bool_masked_pos is not None and self.mask_token is not None:
            mask_tokens = jnp.broadcast_to(
                self.mask_token.value.astype(embeddings.dtype),
                embeddings.shape,
            )
            w = bool_masked_pos[..., None].astype(embeddings.dtype)
            embeddings = embeddings * (1 - w) + mask_tokens * w
        del seq_len

        cls_tokens = jnp.broadcast_to(
            self.cls_token.value.astype(embeddings.dtype),
            (batch_size, 1, embeddings.shape[-1]),
        )
        embeddings = jnp.concatenate([cls_tokens, embeddings], axis=1)

        if self.position_embeddings is not None:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width).astype(embeddings.dtype)

        return checkpoint_name(embeddings, name="embeddings")


class InternVLVisionAttention(AttentionModule):
    """Bidirectional multi-head self-attention for the InternViT tower.

    Standard non-causal MHA with a fused ``[Q | K | V]`` projection and an
    output ``projection_layer``. When ``config.use_qk_norm`` is set, a
    full-width RMSNorm (eps 1e-6, matching HF's ``InternVLVisionRMSNorm``)
    is applied to the query/key projections *before* the head split.
    """

    def __init__(
        self,
        config: InternVLVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize InternVL vision attention.

        Args:
            config (InternVLVisionConfig): Vision tower configuration.
            dtype (jnp.dtype, optional): Computation dtype. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Parameter dtype. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Matmul precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.

        Raises:
            ValueError: If ``hidden_size`` is not divisible by ``num_attention_heads``.
        """
        super().__init__(config=config)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        linear_class = partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
        )
        qkv_layout = dense_qkv_layout(self.embed_dim, self.embed_dim)
        self.qkv_proj = linear_class(
            self.embed_dim,
            qkv_layout.segment_sizes,
            layout=qkv_layout,
            use_bias=config.attention_bias,
        )
        self.projection_layer = linear_class(self.embed_dim, self.embed_dim, use_bias=True)

        if config.use_qk_norm:
            # HF's InternVLVisionRMSNorm is LlamaRMSNorm(embed_dim) with its
            # default eps=1e-6, applied on the full projection width.
            self.q_norm = RMSNorm(self.embed_dim, eps=1e-6, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
            self.k_norm = RMSNorm(self.embed_dim, eps=1e-6, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        else:
            self.q_norm = None
            self.k_norm = None

        self.causal = False
        self.attention_performer = FlexibleAttentionModule(
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=config.attention_dropout,
            rngs=rngs,
            # Vision encoder runs dense, cache-free attention. Pinning "vanilla"
            # keeps the tower functional when serving stacks recursively retarget
            # `attn_mechanism` to paged mechanisms (mirrors Qwen2-VL's tower).
            attn_mechanism="vanilla",
            requires_cache=False,  # Vision encoder doesn't need KV cache
        )

    @property
    def reform_param(self):
        """Checkpoint reform rules mapping HF's split q/k/v onto the fused projection."""
        return self.qkv_proj.build_reform_param("qkv_proj", config=self.config)

    def _split_heads(self, hidden_states: Array) -> Array:
        """Reshape (batch, seq, embed_dim) into (batch, seq, num_heads, head_dim).

        Args:
            hidden_states: Full-width projection output.

        Returns:
            Array: Per-head view of the projection.
        """
        return hidden_states.reshape((*hidden_states.shape[:2], self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states: Array) -> Array:
        """Reshape (batch, seq, num_heads, head_dim) back to (batch, seq, embed_dim).

        Args:
            hidden_states: Per-head attention outputs.

        Returns:
            Array: Merged full-width representation.
        """
        return hidden_states.reshape((*hidden_states.shape[:2], self.embed_dim))

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None = None,
        output_attentions: bool = False,
    ):
        """Apply non-causal multi-head self-attention.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_dim).
            mask_info: Optional attention mask information.
            output_attentions: Whether to return attention weights. Defaults to False.

        Returns:
            AttentionLayerOutput: Attention output and optional attention weights.
        """
        qkv = checkpoint_name(self.qkv_proj(hidden_states), name="attn_qkv")
        query, key, value = self.qkv_proj.split(qkv, config=self.config)

        if self.q_norm is not None:
            query = self.q_norm(query)
        if self.k_norm is not None:
            key = self.k_norm(key)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        attentions = self.attention_performer.forward(
            query_states=query,
            key_states=key,
            value_states=value,
            mode=common_types.MODE_TRAIN,
            mask_info=mask_info,
            causal=self.causal,
        )

        attn_output = self.shard_attention_prod(self._merge_heads(attentions.attention_outputs))
        attn_output = checkpoint_name(self.projection_layer(attn_output), name="attn_output")
        attn_output = self.shard_attention_prod(attn_output)

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
        )


class InternVLVisionMLP(spx.Module):
    """Two-layer feed-forward block for InternViT encoder layers.

    Computes ``fc2(act(fc1(x)))`` with ``config.hidden_act`` (GELU by default),
    matching HF's CLIP-style MLP used by InternVL.
    """

    def __init__(
        self,
        config: InternVLVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the InternViT MLP block.

        Args:
            config (InternVLVisionConfig): Vision tower configuration.
            dtype (jnp.dtype, optional): Computation dtype. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Parameter dtype. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Matmul precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        linear_class = partial(
            ColumnParallelLinear,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
        )
        self.fc1 = linear_class(config.hidden_size, config.intermediate_size)
        self.fc2 = linear_class(config.intermediate_size, config.hidden_size)

    def forward(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        """Apply the feed-forward transformation.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_dim).

        Returns:
            Array: Transformed hidden states of the same shape.
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        hidden_states = checkpoint_name(self.fc1(hidden_states), name="mlp_up")
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = checkpoint_name(self.fc2(hidden_states), name="mlp_down")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        return checkpoint_name(hidden_states, name="mlp_output")


class InternVLVisionLayer(spx.Module):
    """Single InternViT encoder block (timm ``Block`` equivalent).

    Pre-norm attention and MLP sub-blocks, each scaled by a learnable
    layer-scale vector (``lambda_1``/``lambda_2``) before the residual add:

    .. code-block:: text

        h = h + lambda_1 * attn(norm_before(h))
        h = h + lambda_2 * mlp(norm_after(h))

    The norm flavor (LayerNorm vs RMSNorm) is selected by ``config.norm_type``.
    """

    def __init__(
        self,
        config: InternVLVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize an InternViT encoder layer.

        Args:
            config (InternVLVisionConfig): Vision tower configuration.
            dtype (jnp.dtype, optional): Computation dtype. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Parameter dtype. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Matmul precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        self.attention = InternVLVisionAttention(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.mlp = InternVLVisionMLP(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        norm_class = self._norm_class(config)
        self.layernorm_before = norm_class(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.layernorm_after = norm_class(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.lambda_1 = ArrayParam.bound(
            shape=(config.hidden_size,),
            dtype=param_dtype,
            init_method="constant",
            init_kwargs={"value": config.layer_scale_init_value},
            key=rngs.parameters,
        )
        self.lambda_2 = ArrayParam.bound(
            shape=(config.hidden_size,),
            dtype=param_dtype,
            init_method="constant",
            init_kwargs={"value": config.layer_scale_init_value},
            key=rngs.parameters,
        )

    @staticmethod
    def _norm_class(config: InternVLVisionConfig):
        """Resolve the per-layer norm constructor from ``config.norm_type``.

        Args:
            config: Vision tower configuration.

        Returns:
            Callable: A ``(config, *, dtype, param_dtype, rngs)`` constructor.
        """
        if config.norm_type == "rms_norm":

            def _make(cfg, *, dtype, param_dtype, rngs):
                return RMSNorm(cfg.hidden_size, eps=cfg.layer_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        else:

            def _make(cfg, *, dtype, param_dtype, rngs):
                return LayerNorm(
                    cfg.hidden_size,
                    epsilon=cfg.layer_norm_eps,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    rngs=rngs,
                )

        return _make

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None = None,
        output_attentions: bool = False,
    ):
        """Forward pass through the encoder layer.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_dim).
            mask_info: Optional attention mask information.
            output_attentions: Whether to return attention weights. Defaults to False.

        Returns:
            EncoderLayerOutput-like tuple carried through the layer stack: the
            updated hidden states and optional attention weights.
        """
        attn_outputs = self.attention(
            hidden_states=self.layernorm_before(hidden_states),
            mask_info=mask_info,
            output_attentions=output_attentions,
        )
        attention_output = attn_outputs.attention_output * self.lambda_1.value.astype(hidden_states.dtype)
        hidden_states = checkpoint_name(hidden_states + attention_output, name="residual")

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.mlp(layer_output)
        layer_output = layer_output * self.lambda_2.value.astype(hidden_states.dtype)
        hidden_states = checkpoint_name(layer_output + hidden_states, name="layer_output")

        return hidden_states, attn_outputs.attention_weight


class InternVLVisionEncoder(EasyDeLLayerStackMixin, spx.Module):
    """Stack of :class:`InternVLVisionLayer` blocks.

    The module attribute is named ``layer`` (not ``layers``) to match the HF
    checkpoint key layout ``encoder.layer.{i}.*``.
    """

    def __init__(
        self,
        config: InternVLVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the InternViT encoder.

        Args:
            config (InternVLVisionConfig): Vision tower configuration.
            dtype (jnp.dtype, optional): Computation dtype. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Parameter dtype. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Matmul precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        remat_layer_block = auto_remat(
            InternVLVisionLayer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.layer = nn.ModuleList([])
        for idx in range(config.num_hidden_layers):
            with self.assign_layer_stage(idx, total_layers=config.num_hidden_layers):
                self.layer.append(
                    remat_layer_block(
                        config=config,
                        dtype=dtype,
                        param_dtype=param_dtype,
                        precision=precision,
                        rngs=rngs,
                    )
                )

    def forward(
        self,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Forward pass through all encoder layers.

        Args:
            inputs_embeds: Input embeddings of shape (batch, seq_len, hidden_dim).
            mask_info: Optional attention mask information.
            output_attentions: Whether to collect attention weights. Defaults to False.
            output_hidden_states: Whether to collect per-layer hidden states.
                Defaults to False.

        Returns:
            BaseModelOutput: Last hidden state plus optional per-layer hidden
            states / attention weights.
        """
        hidden_states = inputs_embeds
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        def _layer_loop(layer, carry):
            """Apply one encoder layer inside the layer-stack scan.

            Args:
                layer: The encoder layer module for this step.
                carry: Tuple of (hidden_states, all_hidden_states,
                    all_attentions, layer_index).

            Returns:
                tuple: Updated carry.
            """
            hidden_states, all_hidden_states, all_attentions, idx = carry
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            with self._layer_stage_context(idx, layers=self.layer):
                hidden_states, attention_weight = layer(
                    hidden_states=hidden_states,
                    mask_info=mask_info,
                    output_attentions=output_attentions,
                )
            hidden_states = self._mark_layer_stage_boundary(hidden_states, idx, layers=self.layer)

            if output_attentions:
                all_attentions += (attention_weight,)

            return hidden_states, all_hidden_states, all_attentions, idx + 1

        hidden_states, all_hidden_states, all_attentions, _ = self.layer.scan(
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


@register_module(TaskType.BASE_VISION, config=InternVLVisionConfig, model_type="internvl_vision")
@register_module(TaskType.BASE_MODULE, config=InternVLVisionConfig, model_type="internvl_vision")
class InternVLVisionModel(EasyDeLBaseModule):
    """Standalone InternViT vision tower (registered under ``internvl_vision``).

    Embeds pixels with :class:`InternVLVisionEmbeddings`, runs the
    :class:`InternVLVisionEncoder` stack and applies the final layernorm only
    when ``config.use_mean_pooling`` is False (matching HF, where the final
    norm is the identity for mean-pooled checkpoints).

    Unlike EasyDeL's CLIP/SigLIP towers, the HF class stores its parameters
    flat (``embeddings.*`` / ``encoder.layer.*``) in both the standalone and
    the composite checkpoints, so no ``_hf_flattened_wrapper`` declaration is
    needed here.
    """

    def __init__(
        self,
        config: InternVLVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the InternViT vision model.

        Args:
            config (InternVLVisionConfig): Vision tower configuration.
            dtype (jnp.dtype, optional): Computation dtype. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Parameter dtype. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Matmul precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.embeddings = InternVLVisionEmbeddings(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.encoder = InternVLVisionEncoder(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        if config.use_mean_pooling:
            self.layernorm = None
        else:
            self.layernorm = LayerNorm(
                config.hidden_size,
                epsilon=config.layer_norm_eps,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

    def forward(
        self,
        pixel_values: Array,
        bool_masked_pos: Bool[Array, "batch num_patches"] | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Forward pass through the vision tower.

        Args:
            pixel_values: Pixel tensor in NCHW (PyTorch) layout of shape
                (batch, channels, height, width); transposed to NHWC internally.
            bool_masked_pos: Optional boolean masked-patch positions for masked
                image modeling.
            output_attentions: Whether to return attention weights. Defaults to False.
            output_hidden_states: Whether to return per-layer hidden states.
                Defaults to False.

        Returns:
            BaseModelOutputWithPooling: Last hidden state (post final layernorm
            when configured) plus optional hidden states / attentions.
        """
        if pixel_values is not None and pixel_values.ndim == 4:
            # Convert from NCHW (PyTorch) to NHWC (JAX)
            pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))

        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
        encoder_outputs = self.encoder(
            inputs_embeds=embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs.last_hidden_state
        if self.layernorm is not None:
            sequence_output = self.layernorm(sequence_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=checkpoint_name(sequence_output, name="model_output"),
            pooler_output=None,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def get_encoder(self):
        """Return the encoder part of the model.

        Returns:
            InternVLVisionEncoder: The ViT encoder stack.
        """
        return self.encoder

    def get_decoder(self):
        """Encoder-only model: no decoder exists.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("This is an encoder-only model and does not have a decoder.")

    def get_lm_head(self):
        """Vision model has no language model head.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("This vision model does not have a language model head.")

    def get_embedding(self):
        """Return the embedding layer of the module.

        Returns:
            InternVLVisionEmbeddings: The patch/CLS/position embedding module.
        """
        return self.embeddings


class InternVLMultiModalProjector(spx.Module):
    """LayerNorm + two-layer MLP bridging pixel-shuffled vision features to the LM.

    Applies::

        h = layer_norm(image_features)                  # vision_dim * K
        h = act(linear_1(h))                            # -> text_dim
        out = linear_2(h)                               # text_dim -> text_dim

    where ``K = int(1 / downsample_ratio) ** 2`` accounts for the channel
    expansion introduced by pixel shuffle (4x at the default 0.5 ratio) and
    ``act`` is ``config.projector_hidden_act``.
    """

    def __init__(
        self,
        config: InternVLConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the InternVL multimodal projector.

        Args:
            config (InternVLConfig): Composite model configuration.
            dtype (jnp.dtype, optional): Computation dtype. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Parameter dtype. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Matmul precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        input_dim = config.vision_config.hidden_size * int(1 / config.downsample_ratio) ** 2

        self.layer_norm = LayerNorm(
            input_dim,
            epsilon=1e-5,  # torch.nn.LayerNorm default used by HF here
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.linear_1 = RowParallelLinear(
            input_dim,
            config.get_text_config().hidden_size,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(0.02),
            param_dtype=param_dtype,
            dtype=dtype,
            precision=precision,
            rngs=rngs,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = RowParallelLinear(
            config.get_text_config().hidden_size,
            config.get_text_config().hidden_size,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(0.02),
            param_dtype=param_dtype,
            dtype=dtype,
            precision=precision,
            rngs=rngs,
        )

    def forward(self, image_features: jax.Array) -> jax.Array:
        """Project pixel-shuffled vision features into the LM embedding space.

        Args:
            image_features: Vision features of shape
                (batch, num_tokens, vision_hidden * int(1/downsample_ratio)**2).

        Returns:
            jax.Array: Projected features of shape (batch, num_tokens, text_hidden).
        """
        hidden_states = self.layer_norm(image_features)
        hidden_states = checkpoint_name(self.linear_1(hidden_states), name="projector_linear1")
        hidden_states = self.act(hidden_states)
        hidden_states = checkpoint_name(self.linear_2(hidden_states), name="projector_linear2")
        return hidden_states


@register_module(TaskType.BASE_VISION, config=InternVLConfig, model_type="internvl")
class InternVLModel(EasyDeLBaseModule):
    """InternVL base trunk: vision tower + pixel shuffle + projector + language model.

    On a forward pass:

    1. ``vision_tower`` (:class:`InternVLVisionModel`, resolved through
       :class:`AutoEasyDeLVisionModel`) embeds the input images; features come
       from ``config.vision_feature_layer`` (-1 selects the tower output).
    2. The CLS token is dropped (``vision_feature_select_strategy="default"``),
       the patch grid is downsampled by :meth:`pixel_shuffle`
       (``config.downsample_ratio``), and ``multi_modal_projector`` maps the
       expanded channels to the language model's hidden dim.
    3. ``language_model`` (an :class:`AutoEasyDeLModel`, no LM head) consumes
       the merged embedding sequence formed by replacing every
       ``image_token_id`` position with the projected vision tokens.
    """

    def __init__(
        self,
        config: InternVLConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the InternVL base model.

        Args:
            config (InternVLConfig): Composite configuration with vision and text configs.
            dtype (jnp.dtype, optional): Computation dtype. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Parameter dtype. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Matmul precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.vision_tower = AutoEasyDeLVisionModel.from_config(
            config=config.vision_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.multi_modal_projector = InternVLMultiModalProjector(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.language_model = AutoEasyDeLModel.from_config(
            config=config.get_text_config(),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.vision_feature_layer = config.vision_feature_layer
        self.vision_feature_select_strategy = getattr(config, "vision_feature_select_strategy", "default")

    @staticmethod
    def pixel_shuffle(vision_features: Array, scale_factor: float = 0.5) -> Array:
        """Pixel-shuffle downsampling on spatial vision features.

        Faithful port of HF's ``InternVLModel.pixel_shuffle``: trades spatial
        resolution for channel width, mapping ``(batch, width, height, channels)``
        to ``(batch, width * s, height * s, channels / s**2)``.

        Args:
            vision_features: Input tensor of shape (batch, width, height, channels).
            scale_factor: Downsampling factor. Defaults to 0.5 (halves each
                spatial dim, 4x channels).

        Returns:
            Array: Downsampled tensor of shape
                (batch, width * s, height * s, channels / s**2).

        Raises:
            ValueError: If height or width is not divisible by the scale factor.
        """
        batch_size, width, height, channels = vision_features.shape

        if (height * scale_factor) % 1 != 0 or (width * scale_factor) % 1 != 0:
            raise ValueError("Height and width must be divisible by scale_factor for proper downsampling.")

        # Reshape to allow downsampling
        vision_features = vision_features.reshape(
            batch_size, width, int(height * scale_factor), int(channels / scale_factor)
        )
        # Permute dimensions to align downsampled axis correctly
        vision_features = jnp.transpose(vision_features, (0, 2, 1, 3))
        # Reshape to achieve final downsampled dimensions
        vision_features = vision_features.reshape(
            batch_size, int(height * scale_factor), int(width * scale_factor), int(channels / (scale_factor**2))
        )
        # Swap height and width back for proper orientation
        vision_features = jnp.transpose(vision_features, (0, 2, 1, 3))

        return vision_features

    def get_image_features(self, pixel_values: Array) -> Array:
        """Extract, pixel-shuffle and project image features.

        Args:
            pixel_values: Input pixel values of shape (batch, channels, height, width).

        Returns:
            Array: Projected image features of shape
                (batch, num_tokens, text_hidden_size).
        """
        vision_feature_layer = self.vision_feature_layer
        output_hidden_states = vision_feature_layer != -1
        vision_outputs = self.vision_tower(pixel_values, output_hidden_states=output_hidden_states)
        if vision_feature_layer == -1:
            vision_features = vision_outputs.last_hidden_state
        else:
            vision_features = vision_outputs.hidden_states[vision_feature_layer]
        if self.vision_feature_select_strategy == "default":
            vision_features = vision_features[:, 1:, :]

        # Reshape tokens back to their (square) spatial grid.
        num_tokens = vision_features.shape[1]
        feature_size = int(num_tokens**0.5)
        batch_size = vision_features.shape[0]
        vision_features = vision_features.reshape(batch_size, feature_size, feature_size, -1)

        vision_features = self.pixel_shuffle(vision_features, scale_factor=self.config.downsample_ratio)
        vision_features = vision_features.reshape(batch_size, -1, vision_features.shape[-1])

        image_features = tp.cast(Array, self.multi_modal_projector(vision_features))
        return image_features

    def compute_embedding(
        self,
        input_ids: Int[Array, "batch seq_len"] | None,
        *,
        image_features: Array | None = None,
        pixel_values: Array | None = None,
        **kwargs,
    ) -> Array:
        """Compute input embeddings with merged image and text features.

        Args:
            input_ids (Array): Input token IDs of shape (batch, seq_len).
            image_features (Array | None, optional): Pre-extracted image features.
                If None and pixel_values is provided, features are extracted.
            pixel_values (Array | None, optional): Raw pixel values.
            **kwargs: Unused extra keyword arguments.

        Returns:
            Array: Combined embeddings of shape (batch, seq_len, hidden) with
                image features merged at placeholder positions.

        Raises:
            ValueError: If ``input_ids`` is None.
        """
        if input_ids is None:
            raise ValueError("`input_ids` must be provided when calling `compute_embedding`.")

        vocab_size = self.config.get_text_config().vocab_size
        image_token_id = self.config.image_token_id
        if image_token_id >= vocab_size:
            llm_input_ids = jnp.where(input_ids == image_token_id, 0, input_ids)
        else:
            llm_input_ids = input_ids

        inputs_embeds = super().compute_embedding(llm_input_ids)

        if image_features is None and pixel_values is not None:
            image_features = self.get_image_features(pixel_values)

        if image_features is not None:
            multimodal_embeddings = image_features.reshape(-1, image_features.shape[-1]).astype(inputs_embeds.dtype)
            inputs_embeds = BaseVisionLanguageModule.merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                placeholder_token_id=image_token_id,
            )

        return inputs_embeds

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        pixel_values: Array | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **lm_kwargs,
    ):
        """Forward pass through the InternVL base model.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape
                (batch, seq_len). Must be provided when inputs_embeds is None.
            pixel_values (Array | None, optional): Image pixels of shape
                (batch, channels, height, width).
            attention_mask (Array | None, optional): Boolean attention mask of
                shape (batch, seq_len).
            mask_info (MaskInfo | None, optional): Structured mask information.
            position_ids (Array | None, optional): Position indices per token.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode hint.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                KV cache for generation.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Cache management metadata.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings.
            output_attentions (bool | None, optional): Whether to return attention weights.
            output_hidden_states (bool | None, optional): Whether to return hidden states.
            **lm_kwargs: Extra arguments passed to the language model.

        Returns:
            InternVLCausalLMOutputWithPast: Hidden states, cache, attentions and
                projected image hidden states.

        Raises:
            ValueError: If both or neither of input_ids / inputs_embeds are given,
                or if pixel_values is provided without input_ids.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        if pixel_values is not None and input_ids is None:
            raise ValueError("`input_ids` must be provided when `pixel_values` is not None.")

        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)

        if inputs_embeds is None:
            inputs_embeds = self.compute_embedding(
                input_ids,
                image_features=image_features,
            )
        outputs = self.language_model(
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            inputs_embeds=inputs_embeds,
            **lm_kwargs,
        )

        return InternVLCausalLMOutputWithPast(
            loss=None,
            logits=None,
            past_key_values=outputs.past_key_values,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )

    def init_cache(
        self,
        batch_size,
        max_length,
        starts=None,
        shardings=None,
        pad_token_id=None,
    ):
        """Initialize the language model KV cache for generation.

        Args:
            batch_size (int): Batch size for the cache.
            max_length (int): Maximum sequence length to cache.
            starts (int | None, optional): Starting positions.
            shardings (Any | None, optional): Sharding specifications.
            pad_token_id (int | None, optional): Padding token ID.

        Returns:
            TransformerCache: Initialized cache for the language model.
        """
        return self.language_model.init_cache(batch_size, max_length, starts, shardings, pad_token_id)

    def prepare_inputs_for_generation(
        self,
        input_ids: Int[Array, "batch seq_len"],
        max_length: int,
        pad_token_id: int,
        starts: int | None = None,
        pixel_values: Array | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
    ):
        """Prepare inputs for autoregressive generation, including pixels.

        Args:
            input_ids (Array): Prompt token IDs of shape (batch, seq_len).
            max_length (int): Maximum generation length.
            pad_token_id (int): Padding token ID.
            starts (int | None, optional): Starting positions.
            pixel_values (Array | None, optional): Image pixels for the first step.
            attention_mask (Array | None, optional): Attention mask.

        Returns:
            dict: Model inputs ready for generation.
        """
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            max_length=max_length,
            pad_token_id=pad_token_id,
            starts=starts,
            attention_mask=attention_mask,
        )
        model_inputs["pixel_values"] = pixel_values
        return model_inputs

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        """Update model inputs for the next generation step.

        Removes pixel_values after the first step: image features are already
        encoded into the KV cache.

        Args:
            model_outputs: Outputs from the previous generation step.
            model_kwargs (dict): Current model kwargs.

        Returns:
            dict: Updated model kwargs.
        """
        model_kwargs = self.language_model.update_inputs_for_generation(model_outputs, model_kwargs)
        model_kwargs.pop("pixel_values", None)  # only effect first iter
        return model_kwargs

    def get_encoder(self):
        """Return the vision tower (the model's encoder).

        Returns:
            InternVLVisionModel: The vision encoder.
        """
        return self.vision_tower

    def get_decoder(self):
        """Return the language model's decoder.

        Returns:
            spx.Module: The decoder stack.
        """
        return self.language_model.get_decoder()

    def get_lm_head(self):
        """Base models don't own an LM head.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("The base model does not have a language model head.")

    def get_embedding(self):
        """Return the text embedding layer.

        Returns:
            Embed: The language model's token embedding.
        """
        return self.language_model.get_embedding()


@register_module(TaskType.IMAGE_TEXT_TO_TEXT, config=InternVLConfig, model_type="internvl")
class InternVLForConditionalGeneration(BaseVisionLanguageModule[InternVLModel, InternVLConfig]):  # type: ignore
    """InternVL model for conditional text generation from image + text inputs.

    Wraps :class:`InternVLModel` with an LM head (tied to the text embeddings
    when ``config.tie_word_embeddings`` is set, matching the HF default).

    Class Attributes:
        _task_type: IMAGE_TEXT_TO_TEXT task type.
        _model_type: "internvl" model identifier.
        _supports_video: False (frames are handled as image batches upstream).
        _uses_mrope: False (standard RoPE from the text backbone).
    """

    _task_type = TaskType.IMAGE_TEXT_TO_TEXT
    _model_type = "internvl"
    _config_class = InternVLConfig
    _auto_register = False  # Already registered via decorator
    _supports_video = False
    _uses_mrope = False

    _vision_tower_name = "vision_tower"
    _projector_name = "multi_modal_projector"
    _language_model_name = "language_model"

    loss_type = "ForCausalLM"

    def __init__(
        self,
        config: InternVLConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize InternVL for conditional generation.

        Args:
            config (InternVLConfig): Composite model configuration.
            dtype (jnp.dtype, optional): Computation dtype. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Parameter dtype. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Matmul precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=InternVLModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            # VLM-specific configuration
            vision_feature_layer=config.vision_feature_layer,
            vision_feature_select_strategy=getattr(config, "vision_feature_select_strategy", "default"),
            image_token_index=config.image_token_id,
            # LM head configuration
            tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
            lm_head_bias=False,
        )

    def get_image_features(
        self,
        pixel_values: Float[Array, "batch channels height width"],
        **kwargs,
    ) -> Float[Array, "batch num_patches hidden"]:
        """Extract, pixel-shuffle and project image features.

        Delegates to the base model, which runs the vision tower, drops the CLS
        token, applies pixel shuffle and projects to the LM hidden size.

        Args:
            pixel_values: Image pixels of shape (batch, channels, height, width).
            **kwargs: Unused extra arguments.

        Returns:
            Array: Projected image features ready for multimodal merge.
        """
        return self.base_model.get_image_features(pixel_values)

    def compute_embedding(self, input_ids, *args, **kwargs):
        """Compute input embeddings with merged image and text features.

        Args:
            input_ids (Array): Input token IDs of shape (batch, seq_len).
            *args: Positional arguments passed to the base model.
            **kwargs: Keyword arguments (e.g. pixel_values, image_features).

        Returns:
            Array: Combined embeddings with image features merged.
        """
        return self.base_model.compute_embedding(input_ids, *args, **kwargs)

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        pixel_values: Array | None = None,
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
        **lm_kwargs,
    ) -> VLMCausalLMOutput:
        """Forward pass for image-conditioned text generation.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape
                (batch, seq_len). Must be provided when inputs_embeds is None.
            pixel_values (Array | None, optional): Image pixels of shape
                (batch, channels, height, width).
            attention_mask (Array | None, optional): Boolean attention mask.
            mask_info (MaskInfo | None, optional): Structured mask information.
            position_ids (Array | None, optional): Position indices per token.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode hint.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                KV cache for generation.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Cache management metadata.
            apply_lm_head (bool, optional): Whether to compute vocabulary logits.
                Defaults to True.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings.
            output_attentions (bool | None, optional): Whether to return attention weights.
            output_hidden_states (bool | None, optional): Whether to return hidden states.
            **lm_kwargs: Extra arguments passed to the language model.

        Returns:
            VLMCausalLMOutput: Logits (when requested), cache, hidden states,
                attentions and image hidden states.

        Raises:
            ValueError: If both or neither of input_ids / inputs_embeds are given.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.base_model(
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
            pixel_values=pixel_values,
            **lm_kwargs,
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

        return VLMCausalLMOutput(
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            last_hidden_state=hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states if pixel_values is not None else None,
        )

    def init_cache(
        self,
        batch_size,
        max_length,
        starts=None,
        shardings=None,
        pad_token_id=None,
    ):
        """Initialize the KV cache for autoregressive generation.

        Args:
            batch_size (int): Batch size for the cache.
            max_length (int): Maximum sequence length to cache.
            starts (int | None, optional): Starting positions.
            shardings (Any | None, optional): Sharding specifications.
            pad_token_id (int | None, optional): Padding token ID.

        Returns:
            TransformerCache: Initialized cache for the language model.
        """
        return self.base_model.init_cache(batch_size, max_length, starts, shardings, pad_token_id)

    def get_vision_tower(self) -> spx.Module:
        """Return the vision tower component.

        Returns:
            spx.Module: The InternViT vision encoder.
        """
        return self.base_model.vision_tower

    def get_projector(self) -> spx.Module:
        """Return the multimodal projector component.

        Returns:
            spx.Module: The LayerNorm+MLP projector.
        """
        return self.base_model.multi_modal_projector

    def get_language_model(self) -> spx.Module:
        """Return the language model component.

        Returns:
            spx.Module: The text decoder backbone.
        """
        return self.base_model.language_model
