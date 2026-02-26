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
import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
from eformer import common_types
from eformer.common_types import ColumnWise, Replicated
from eformer.escale import apply_logical_sharding
from eformer.pytree import auto_pytree
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from flax import nnx as nn
from jax import image as jimg
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import (
    AttentionLayerOutput,
    BaseModelOutput,
    BaseModelOutputWithPooling,
    EncoderLayerOutput,
    ImageClassifierOutput,
    ModelOutput,
)
from easydel.infra.utils import ACT2FN, ArrayParam
from easydel.layers import ColumnParallelLinear, Embed, RowParallelLinear
from easydel.layers.attention import AttentionModule, FlexibleAttentionModule
from easydel.layers.norms import LayerNorm
from easydel.modules._base import BaseImageClassificationModule

from .configuration_siglip import SiglipConfig, SiglipTextConfig, SiglipVisionConfig


@auto_pytree
class SiglipVisionModelOutput(ModelOutput):
    """Outputs from the SigLIP vision tower including pooled embeddings."""

    image_embeds: Array | None = None
    last_hidden_state: Array = None
    hidden_states: tuple[Array, ...] | None = None
    attentions: tuple[Array, ...] | None = None


@auto_pytree
class SiglipTextModelOutput(ModelOutput):
    """Outputs from the SigLIP text encoder with optional attentions."""

    text_embeds: Array | None = None
    last_hidden_state: Array = None
    hidden_states: tuple[Array, ...] | None = None
    attentions: tuple[Array, ...] | None = None


@auto_pytree
class SiglipOutput(ModelOutput):
    """Contrastive SigLIP output bundling text/vision logits and embeddings."""

    loss: Array | None = None
    logits_per_image: Array = None
    logits_per_text: Array = None
    text_embeds: Array = None
    image_embeds: Array = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> tuple[tp.Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class SiglipVisionEmbeddings(nn.Module):
    """Vision embeddings module for SigLIP models.

    Converts image pixel values into patch embeddings with position encodings
    for vision transformer processing. Unlike CLIP, SigLIP does not use a class token.
    """

    def __init__(
        self,
        config: SiglipVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize SigLIP vision embeddings.

        Args:
            config (SiglipVisionConfig): Vision model configuration with embedding parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = Embed(
            self.num_positions,
            self.embed_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.patch_embedding = nn.Conv(
            in_features=config.num_channels,
            out_features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            precision=precision,
        )

    def interpolate(self, embeddings: Array, height: int, width: int):
        """Interpolate position embeddings for different image sizes.

        Args:
            embeddings: Patch embeddings of shape (batch_size, num_patches, embed_dim).
            height: Original image height in pixels.
            width: Original image width in pixels.

        Returns:
            Interpolated position embeddings matching the number of patches.
        """
        num_patches = embeddings.shape[1]
        num_positions = self.position_embedding.weight.shape[0]
        if num_patches == num_positions and height == width:
            return self.position_embedding(
                jnp.arange(
                    self.num_positions,
                    dtype="i4",
                ).reshape(1, -1)
            )
        patch_pos_embed = self.position_embedding.embedding.unsqueeze(0)

        dim = embeddings.shape[-1]
        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = int(num_positions**0.5)

        patch_pos_embed = jnp.reshape(patch_pos_embed, (1, sqrt_num_positions, sqrt_num_positions, dim))
        patch_pos_embed = jnp.transpose(patch_pos_embed, (0, 3, 1, 2))

        patch_pos_embed = jimg.resize(
            patch_pos_embed,
            (1, dim, new_height, new_width),
            method="cubic",
        )

        return jnp.reshape(jnp.transpose(patch_pos_embed, (0, 2, 3, 1)), (1, -1, dim))

    def __call__(self, pixel_values: Array, interpolate_pos_encoding=False):
        """Create vision embeddings from pixel values.

        Args:
            pixel_values: Input pixel values of shape (batch_size, channels, height, width).
            interpolate_pos_encoding: Whether to interpolate position embeddings for
                different image resolutions. Defaults to False.

        Returns:
            Patch embeddings with position encodings of shape
            (batch_size, num_patches, hidden_size).
        """
        _, _, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.kernel.dtype

        pixel_values = pixel_values.transpose(0, 2, 3, 1).astype(dtype=target_dtype)
        patch_embeds = self.patch_embedding(pixel_values).transpose(0, 3, 1, 2)

        embeddings = jnp.reshape(patch_embeds, (*patch_embeds.shape[:2], -1))
        embeddings = jnp.transpose(embeddings, (0, 2, 1))
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(jnp.arange(self.num_positions, dtype="i4").reshape(1, -1))
        return embeddings


class SiglipTextEmbeddings(nn.Module):
    """Text embeddings module for SigLIP models.

    Combines token embeddings and position embeddings for text
    transformer processing in the SigLIP architecture.
    """

    def __init__(
        self,
        config: SiglipTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize SigLIP text embeddings.

        Args:
            config (SiglipTextConfig): Text model configuration with embedding parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        embed_dim = config.hidden_size

        self.token_embedding = Embed(
            config.vocab_size,
            embed_dim,
            embedding_init=jax.nn.initializers.normal(),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.position_embedding = Embed(
            config.max_position_embeddings,
            embed_dim,
            embedding_init=jax.nn.initializers.normal(),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
    ) -> Array:
        """Create text embeddings from token IDs.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length).
            position_ids: Position indices of shape (batch_size, sequence_length).
                Auto-generated if not provided.
            inputs_embeds: Pre-computed token embeddings. If provided, input_ids is ignored.

        Returns:
            Combined token and position embeddings of shape
            (batch_size, sequence_length, hidden_size).

        Raises:
            ValueError: If sequence length exceeds max_position_embeddings.
        """
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        max_position_embedding = self.position_embedding.embedding.shape[0]

        if seq_length > max_position_embedding:
            raise ValueError(
                f"Sequence length must be less than max_position_embeddings (got `sequence length`: "
                f"{seq_length} and max_position_embeddings: {max_position_embedding}"
            )

        if position_ids is None:
            position_ids = jnp.arange(seq_length, dtype="i4").reshape(1, -1)

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class SiglipAttention(AttentionModule):
    """Multi-head attention module for SigLIP models.

    Implements bidirectional self-attention for both text and vision
    encoders in the SigLIP architecture.
    """

    def __init__(
        self,
        config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize SigLIP attention layer.

        Args:
            config: Model configuration with attention parameters (hidden_size,
                num_attention_heads, attention_dropout).
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state.
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

        self.dropout = config.attention_dropout
        linear_class = partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(0.01),
        )
        self.k_proj = linear_class(self.embed_dim, self.embed_dim)
        self.v_proj = linear_class(self.embed_dim, self.embed_dim)
        self.q_proj = linear_class(self.embed_dim, self.embed_dim)
        self.out_proj = linear_class(self.embed_dim, self.embed_dim)

        self.causal = False
        self.attention_performer = FlexibleAttentionModule(
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=config.attention_dropout,
            rngs=rngs,
            requires_cache=False,  # Vision/text encoder doesn't need KV cache
        )

    def _split_heads(self, hidden_states):
        """Split hidden states into multiple attention heads.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, embed_dim).

        Returns:
            Reshaped tensor of shape (batch, seq_len, num_heads, head_dim).
        """
        return hidden_states.reshape((*hidden_states.shape[:2], self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        """Merge attention heads back into hidden states.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, num_heads, head_dim).

        Returns:
            Merged tensor of shape (batch, seq_len, embed_dim).
        """
        return hidden_states.reshape((*hidden_states.shape[:2], self.embed_dim))

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None = None,
        output_attentions: bool = False,
    ):
        """Apply multi-head self-attention.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_dim).
            mask_info: Attention mask information for masking padded positions.
            output_attentions: Whether to return attention weights. Defaults to False.

        Returns:
            AttentionLayerOutput containing attention output and optional attention weights.
        """
        query = checkpoint_name(self.q_proj(hidden_states), "attn_query")
        key = checkpoint_name(self.k_proj(hidden_states), "attn_key")
        value = checkpoint_name(self.v_proj(hidden_states), "attn_value")

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

        attn_output = self._merge_heads(attentions.attention_outputs)
        attn_output = checkpoint_name(self.out_proj(attn_output), "attn_output")

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
        )


class SiglipMLP(nn.Module):
    """Multi-Layer Perceptron module for SigLIP models.

    Implements the feedforward network with configurable activation function
    for both text and vision encoders in the SigLIP architecture.
    """

    def __init__(
        self,
        config: SiglipTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize SigLIP MLP block.

        Args:
            config (SiglipTextConfig): Model configuration with MLP parameters
                (hidden_size, intermediate_size, hidden_act).
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.activation_fn = ACT2FN[config.hidden_act]
        linear_class = partial(
            ColumnParallelLinear,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(0.01),
        )
        self.fc1 = linear_class(config.hidden_size, config.intermediate_size)
        self.fc2 = linear_class(config.intermediate_size, config.hidden_size)

    def __call__(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        """Apply feedforward transformation.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_dim).

        Returns:
            Transformed hidden states of shape (batch, seq_len, hidden_dim).
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        hidden_states = checkpoint_name(self.fc2(self.activation_fn(self.fc1(hidden_states))), "mlp_output")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return hidden_states


class SiglipEncoderLayer(nn.Module):
    """Single encoder layer for SigLIP models.

    Combines multi-head self-attention and feedforward networks with
    layer normalization and residual connections using pre-norm architecture.
    """

    def __init__(
        self,
        config: SiglipTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize SigLIP encoder layer.

        Args:
            config (SiglipTextConfig): Model configuration with layer parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.self_attn = SiglipAttention(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.layer_norm1 = LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.mlp = SiglipMLP(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.layer_norm2 = LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None = None,
        output_attentions: bool = False,
    ):
        """Forward pass through the encoder layer.

        Applies pre-normalization architecture: x + attn(norm(x)) followed by x + mlp(norm(x)).

        Args:
            hidden_states: Input tensor of shape (batch_size, sequence_length, hidden_dim).
            mask_info: Attention mask information for masking positions.
            output_attentions: Whether to return attention weights. Defaults to False.

        Returns:
            EncoderLayerOutput containing hidden states and optional attention weights.
        """
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            mask_info=mask_info,
            output_attentions=output_attentions,
        )
        hidden_states = attn_outputs.attention_output
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return EncoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
        )


class SiglipEncoder(nn.Module):
    """Transformer encoder for SigLIP models.

    Stacks multiple SiglipEncoderLayer instances to form the complete
    encoder for either text or vision processing.
    """

    def __init__(
        self,
        config: SiglipTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize SigLIP encoder.

        Args:
            config (SiglipTextConfig): Model configuration with encoder parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.layers = nn.List(
            [
                SiglipEncoderLayer(
                    config=config,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    precision=precision,
                    rngs=rngs,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

    def __call__(
        self,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Forward pass through all encoder layers.

        Args:
            inputs_embeds: Input embeddings of shape (batch_size, sequence_length, hidden_dim).
            mask_info: Attention mask information. Defaults to None.
            output_attentions: Whether to return attention weights from all layers.
                Defaults to False.
            output_hidden_states: Whether to return hidden states from all layers.
                Defaults to False.

        Returns:
            BaseModelOutput containing last hidden state, optional all hidden states,
            and optional attention weights.
        """
        hidden_states = inputs_embeds
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states=hidden_states,
                mask_info=mask_info,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs.hidden_states

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class SiglipTextTransformer(EasyDeLBaseModule):
    """Text transformer encoder for SigLIP models.

    Processes text tokens through embeddings, multiple encoder layers,
    and final layer normalization with a projection head to produce
    text representations for contrastive learning.
    """

    def __init__(
        self,
        config: SiglipTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize SigLIP text transformer.

        Args:
            config (SiglipTextConfig): Text model configuration.
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
        embed_dim = config.hidden_size
        self.embeddings = SiglipTextEmbeddings(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.encoder = SiglipEncoder(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.final_layer_norm = LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.head = ColumnParallelLinear(
            embed_dim,
            config.projection_size,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"],
        mask_info: MaskInfo,
        position_ids: Int[Array, "batch seq_len"],
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = False,
    ):
        """Forward pass through the text transformer.

        Processes text tokens through embeddings, encoder layers, and produces
        pooled output from the last token position with a projection head.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length).
            mask_info: Attention mask information for masking padded tokens.
            position_ids: Position indices of shape (batch_size, sequence_length).
            output_attentions: Whether to return attention weights. Defaults to False.
            output_hidden_states: Whether to return hidden states from all layers.
                Defaults to False.

        Returns:
            BaseModelOutputWithPooling containing last hidden state, pooled output,
            optional hidden states, and optional attention weights.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        input_shape = input_ids.shape
        input_ids = input_ids.reshape(-1, input_shape[-1])
        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
        )

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            mask_info=mask_info,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.final_layer_norm(last_hidden_state)
        pooled_output = last_hidden_state[:, -1, :]
        pooled_output = self.head(pooled_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        """
        return self

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        This is an encoder-only model.
        """
        raise NotImplementedError("This is an encoder-only model and does not have a decoder.")

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        This model has a projection head, not a language model head.
        """
        raise NotImplementedError("This model has a projection head, not a language model head.")

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.embeddings


@register_module(TaskType.BASE_MODULE, config=SiglipTextConfig, model_type="siglip_text_model")
class SiglipTextModel(EasyDeLBaseModule):
    """SigLIP text model outputting raw hidden states.

    A bare transformer encoder for text that produces embeddings without
    any task-specific head, suitable for use in contrastive learning.
    """

    def __init__(
        self,
        config: SiglipTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize SigLIP text model.

        Args:
            config (SiglipTextConfig): Text model configuration.
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
        self.text_model = SiglipTextTransformer(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> tuple | BaseModelOutputWithPooling:
        """Forward pass through the SigLIP text model.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length).
            attention_mask: Boolean mask of shape (batch_size, sequence_length) indicating
                which tokens to attend to. Auto-generated if not provided.
            mask_info: Pre-computed attention mask information. Overrides attention_mask if provided.
            position_ids: Position indices of shape (batch_size, sequence_length).
                Auto-generated if not provided.
            output_attentions: Whether to return attention weights. Defaults to config value.
            output_hidden_states: Whether to return hidden states from all layers.
                Defaults to config value.

        Returns:
            BaseModelOutputWithPooling containing last hidden state, pooled output,
            optional hidden states, and optional attention weights.

        Raises:
            ValueError: If input_ids is not provided.
        """
        if input_ids is None:
            raise ValueError("`input_ids` must be provided.")

        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        if position_ids is None:
            position_ids = mask_info.q_position_ids

        return self.text_model(
            input_ids=input_ids,
            mask_info=mask_info,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        """
        return self.text_model

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        This is an encoder-only model.
        """
        raise NotImplementedError("This is an encoder-only model and does not have a decoder.")

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        This model has a projection head, not a language model head.
        """
        raise NotImplementedError("This model has a projection head, not a language model head.")

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.text_model.embeddings


class SiglipVisionTransformer(EasyDeLBaseModule):
    """Vision transformer encoder for SigLIP models.

    Processes images through patch embeddings, multiple encoder layers with
    bidirectional attention, and layer normalization with an optional
    multi-head attention pooling head to produce vision representations.
    """

    def __init__(
        self,
        config: SiglipTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize SigLIP vision transformer.

        Args:
            config (SiglipVisionConfig): Vision model configuration.
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
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.encoder = SiglipEncoder(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.post_layernorm = LayerNorm(
            embed_dim,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.use_head = True if not hasattr(config, "vision_use_head") else config.vision_use_head
        if self.use_head:
            self.head = SiglipMultiheadAttentionPoolingHead(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )

    def __call__(
        self,
        pixel_values,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        interpolate_pos_encoding: bool | None = False,
    ) -> tuple | BaseModelOutputWithPooling:
        """Forward pass through the vision transformer.

        Processes images through patch embeddings, encoder layers, post-layer norm,
        and produces pooled output using multi-head attention pooling.

        Args:
            pixel_values: Input pixel values of shape (batch_size, channels, height, width).
            output_attentions: Whether to return attention weights. Defaults to config value.
            output_hidden_states: Whether to return hidden states from all layers.
                Defaults to config value.
            interpolate_pos_encoding: Whether to interpolate position embeddings for
                different image resolutions. Defaults to False.

        Returns:
            BaseModelOutputWithPooling containing last hidden state, pooled output
            from attention pooling, optional hidden states, and optional attention weights.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        hidden_states = self.embeddings(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)
        pooler_output = self.head(last_hidden_state) if self.use_head else None

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        """
        return self

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        This is an encoder-only model.
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
        Returns the embedding layer of the module.
        """
        return self.embeddings


class MultiheadAttention(nn.Module):
    """Multi-head attention module for SigLIP vision pooling.

    Implements standard multi-head attention without causal masking,
    used by the attention pooling head for aggregating patch representations.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        bias=True,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize multi-head attention module.

        Args:
            embed_dim: Embedding dimension for the attention layer.
            num_heads: Number of attention heads.
            bias: Whether to include bias in projection layers. Defaults to True.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.

        Raises:
            ValueError: If embed_dim or num_heads is non-positive.
            AssertionError: If embed_dim is not divisible by num_heads.
        """
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = ArrayParam.bound(
            shape=(embed_dim * 3, embed_dim),
            dtype=param_dtype,
            init_method="xavier_uniform",
            key=rngs.param(),
        )
        self.in_proj_bias = ArrayParam.bound(
            shape=(3 * embed_dim,),
            dtype=param_dtype,
            init_method="zeros",
            key=rngs.param(),
        )
        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            use_bias=bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def craft_sharding(self, *, partition_manager=None, **_kwargs) -> dict[str, object]:
        """Return sharding specs for custom attention parameters."""
        return {
            "in_proj_weight": ColumnWise,
            "in_proj_bias": Replicated,
        }

    def __call__(
        self,
        query: Array,
        key: Array,
        value: Array,
    ):
        """Apply multi-head attention.

        Args:
            query: Query tensor of shape (batch_size, query_len, embed_dim).
            key: Key tensor of shape (batch_size, key_len, embed_dim).
            value: Value tensor of shape (batch_size, key_len, embed_dim).

        Returns:
            Attention output of shape (batch_size, query_len, embed_dim).
        """
        qbs, qss, qds = query.shape
        b, s, _d = value.shape

        qb, kb, vb = jnp.split(self.in_proj_bias, 3, -1)
        qw, kw, vw = jnp.split(self.in_proj_weight, 3, -1)

        qout = ((query @ qw) + qb).reshape(qbs, qss, self.num_heads, -1)
        kout = ((key @ kw) + kb).reshape(b, s, self.num_heads, -1)
        vout = ((value @ vw) + vb).reshape(b, s, self.num_heads, -1)

        attn = jnp.einsum(
            "bhqk,bkhd->bqhd",
            jax.nn.softmax(
                jnp.einsum(
                    "bqhd,bkhd->bhqk",
                    qout * (qout.shape[-1] ** -0.5),
                    kout,
                )
            ),
            vout,
        )

        return checkpoint_name(self.out_proj(attn.reshape(qbs, qss, qds)), "attn_output")


class SiglipMultiheadAttentionPoolingHead(nn.Module):
    """Multi-head attention pooling head for SigLIP vision model.

    Uses a learned probe token to attend over all patch representations,
    producing a single pooled representation followed by MLP refinement.
    This is SigLIP's approach to aggregating patch-level features into
    image-level representations.
    """

    def __init__(
        self,
        config: SiglipTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize attention pooling head.

        Args:
            config (SiglipVisionConfig): Vision model configuration with pooling parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.probe = ArrayParam.bound(
            shape=(1, 1, config.hidden_size),
            dtype=param_dtype,
            init_method="normal",
            key=rngs.param(),
        )
        self.attention = MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.layernorm = LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.mlp = SiglipMLP(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def craft_sharding(self, *, partition_manager=None, **_kwargs) -> dict[str, object]:
        """Return sharding specs for attention pooling parameters."""
        return {"probe": Replicated}

    def __call__(self, hidden_state):
        """Apply attention pooling over patch representations.

        Args:
            hidden_state: Patch representations of shape (batch_size, num_patches, hidden_size).

        Returns:
            Pooled representation of shape (batch_size, hidden_size).
        """
        batch_size = hidden_state.shape[0]
        probe = self.probe.value.repeat(batch_size, 0)
        hidden_state = self.attention(probe, hidden_state, hidden_state)
        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)
        return hidden_state[:, 0]


@register_module(TaskType.BASE_VISION, config=SiglipVisionConfig, model_type="siglip_vision_model")
class SiglipVisionModel(EasyDeLBaseModule):
    """SigLIP vision model outputting raw hidden states.

    A bare vision transformer that processes images and produces embeddings
    without any task-specific head, suitable for use in contrastive learning.
    """

    def __init__(
        self,
        config: SiglipVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize SigLIP vision model.

        Args:
            config (SiglipVisionConfig): Vision model configuration.
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
        self.vision_model = SiglipVisionTransformer(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        pixel_values,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        interpolate_pos_encoding: bool = False,
    ) -> tuple | BaseModelOutputWithPooling:
        """Forward pass through the SigLIP vision model.

        Args:
            pixel_values: Input pixel values of shape (batch_size, channels, height, width).
            output_attentions: Whether to return attention weights. Defaults to config value.
            output_hidden_states: Whether to return hidden states from all layers.
                Defaults to config value.
            interpolate_pos_encoding: Whether to interpolate position embeddings for
                different image resolutions. Defaults to False.

        Returns:
            BaseModelOutputWithPooling containing last hidden state, pooled output,
            optional hidden states, and optional attention weights.
        """
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        """
        return self.vision_model

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        This is an encoder-only model.
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
        Returns the embedding layer of the module.
        """
        return self.vision_model.embeddings


@register_module(TaskType.BASE_MODULE, config=SiglipConfig, model_type="siglip")
class SiglipModel(EasyDeLBaseModule):
    """Sigmoid Language-Image Pre-training (SigLIP) model.

    Combines text and vision encoders with sigmoid loss for contrastive
    learning. Unlike CLIP, SigLIP uses a sigmoid loss function instead
    of softmax, enabling more efficient batch-level contrastive learning.
    """

    def __init__(
        self,
        config: SiglipConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize SigLIP model.

        Args:
            config (SiglipConfig): Combined text and vision configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.

        Raises:
            TypeError: If config.get_text_config() is not SiglipTextConfig or
                config.vision_config is not SiglipVisionConfig.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        if not isinstance(config.get_text_config(), SiglipTextConfig):
            raise TypeError(
                "config.get_text_config() is expected to be of type SiglipTextConfig but is of type"
                f" {type(config.get_text_config())}."
            )

        if not isinstance(config.vision_config, SiglipVisionConfig):
            raise TypeError(
                "config.vision_config is expected to be of type SiglipVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.get_text_config()
        vision_config = config.vision_config

        text_model = SiglipTextModel(
            text_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        vision_model = SiglipVisionModel(
            vision_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.text_model = text_model.text_model
        self.vision_model = vision_model.vision_model

        self.logit_scale = ArrayParam.bound(
            shape=(1,),
            dtype=param_dtype,
            init_method="normal",
            key=rngs.param(),
        )
        self.logit_bias = ArrayParam.bound(
            shape=(1,),
            dtype=param_dtype,
            init_method="normal",
            key=rngs.param(),
        )

    def craft_sharding(self, *, partition_manager=None, **_kwargs) -> dict[str, object]:
        """Return sharding specs for logit scaling parameters."""
        return {
            "logit_scale": Replicated,
            "logit_bias": Replicated,
        }

    def get_text_features(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> Array:
        """Extract text features from input tokens.

        Processes text through the text encoder and returns the pooled output
        from the projection head.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length).
            attention_mask: Boolean mask indicating tokens to attend to.
            mask_info: Pre-computed mask information.
            position_ids: Position indices for tokens.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            Pooled text features of shape (batch_size, projection_size).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if mask_info is None and attention_mask is not None:
            mask_info = MaskInfo.from_attention_mask(attention_mask)

        text_outputs = self.text_model(
            input_ids=input_ids,
            mask_info=mask_info,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = text_outputs[1]

        return pooled_output

    def get_image_features(
        self,
        pixel_values: Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        interpolate_pos_encoding: bool = False,
    ) -> Array:
        """Extract image features from pixel values.

        Processes images through the vision encoder and returns the pooled output
        from the attention pooling head.

        Args:
            pixel_values: Input pixel values of shape (batch_size, channels, height, width).
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.
            interpolate_pos_encoding: Whether to interpolate position embeddings.

        Returns:
            Pooled image features of shape (batch_size, hidden_size).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        pooled_output = vision_outputs[1]

        return pooled_output

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        pixel_values: Array | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        return_loss: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        interpolate_pos_encoding: bool = False,
    ) -> tuple | SiglipOutput:
        """Performs forward pass through SigLIP's dual-encoder architecture.

        Processes images through the vision transformer and text through the text transformer,
        normalizes both embeddings, and computes sigmoid-based contrastive similarity logits
        between all image-text pairs in the batch.

        Args:
            input_ids: Text token IDs of shape (batch_size, sequence_length). Tokenized text
                inputs for the text encoder.
            pixel_values: Image pixel values of shape (batch_size, channels, height, width).
                Preprocessed images for the vision encoder.
            attention_mask: Boolean mask of shape (batch_size, sequence_length) for text,
                indicating which tokens to attend to.
            mask_info: Pre-computed mask information for text. If provided, overrides
                `attention_mask`.
            position_ids: Explicit position indices of shape (batch_size, sequence_length)
                for text. Auto-generated if not provided.
            return_loss: Whether to compute and return the contrastive loss.
            output_attentions: Whether to return attention weights from vision and text encoders.
            output_hidden_states: Whether to return hidden states from all layers of both encoders.
            interpolate_pos_encoding: Whether to interpolate position embeddings for
                different image resolutions.

        Returns:
            SiglipOutput containing:
                - loss: Sigmoid contrastive loss if return_loss is True
                - logits_per_image: Similarity scores of shape (batch, batch)
                - logits_per_text: Similarity scores of shape (batch, batch)
                - text_embeds: L2-normalized text embeddings
                - image_embeds: L2-normalized image embeddings
                - text_model_output: Full output from text encoder
                - vision_model_output: Full output from vision encoder
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        if mask_info is None and attention_mask is not None:
            mask_info = MaskInfo.from_attention_mask(attention_mask)

        text_outputs = self.text_model(
            input_ids=input_ids,
            mask_info=mask_info,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        image_embeds = vision_outputs[1]
        text_embeds = text_outputs[1]

        # normalized features
        image_embeds = image_embeds / jnp.linalg.norm(
            image_embeds,
            ord=2,
            axis=-1,
            keepdims=True,
        )
        text_embeds = text_embeds / jnp.linalg.norm(
            text_embeds,
            ord=2,
            axis=-1,
            keepdims=True,
        )

        # cosine similarity as logits
        logits_per_text = jnp.matmul(text_embeds, image_embeds.T)

        logit_scale, logit_bias = (self.logit_scale, self.logit_bias)
        logits_per_text = logits_per_text * jnp.exp(logit_scale) + logit_bias

        logits_per_image = logits_per_text.T

        loss = None
        if return_loss:
            m1_diag1 = -jnp.ones_like(logits_per_text) + 2 * jnp.eye(logits_per_text.shape[0])
            loglik = jax.nn.log_sigmoid(m1_diag1 * logits_per_text)
            nll = -jnp.sum(loglik, axis=-1)
            loss = nll.mean()

        return SiglipOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        The vision tower acts as the encoder in this multi-modal setup.
        """
        return self.vision_model

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        The text model acts as the decoder in this multi-modal setup.
        """
        return self.text_model

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        This model does not have a traditional language model head, but a projection head.
        """
        raise NotImplementedError("This model does not have a language model head.")

    def get_embedding(self):
        """
        Returns the embedding layer of the text model.
        """
        return self.text_model.embeddings


@register_module(TaskType.IMAGE_CLASSIFICATION, config=SiglipConfig, model_type="siglip")
class SiglipForImageClassification(BaseImageClassificationModule[SiglipVisionModel, SiglipConfig]):  # type: ignore
    """SigLIP vision model with image classification head.

    Extends the SigLIP vision transformer with a linear classification layer
    on top of the mean-pooled patch embeddings for image classification tasks.
    """

    def __init__(
        self,
        config: SiglipConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize SigLIP for image classification.

        Args:
            config (SiglipConfig): Model configuration with vision_config and num_labels.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        vision_model = SiglipVisionModel(
            config.vision_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        super().__init__(
            config=config,
            base_model=vision_model,
            base_model_name="vision_model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            pooling_strategy="mean",
            classifier_bias=True,
        )
        self.num_labels = config.num_labels
        self.use_classif = self.classifier is not None

    def __call__(
        self,
        pixel_values: Array | None = None,
        labels: Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        interpolate_pos_encoding: bool = False,
    ) -> tuple | ImageClassifierOutput:
        """Forward pass for image classification.

        Args:
            pixel_values: Input pixel values of shape (batch_size, channels, height, width).
            labels: Ground truth labels for computing loss (unused in this implementation).
            output_attentions: Whether to return attention weights. Defaults to config value.
            output_hidden_states: Whether to return hidden states from all layers.
                Defaults to config value.
            interpolate_pos_encoding: Whether to interpolate position embeddings for
                different image resolutions.

        Returns:
            ImageClassifierOutput containing classification logits, optional hidden states,
            and optional attention weights.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.vision_model(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        sequence_output = outputs[0]

        logits = jnp.mean(sequence_output, axis=1)
        if self.classifier is not None:
            logits = self.classifier(logits)

        return ImageClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        """
        return self.vision_model

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        This is an encoder-only model for classification.
        """
        raise NotImplementedError("This is an encoder-only model and does not have a decoder.")

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        This model has an image classification head, not a language model head.
        """
        raise NotImplementedError("This model has an image classification head, not a language model head.")

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.base_model.get_embedding()

    def get_task_head(self):
        """Returns the image classification head."""
        return self.classifier
