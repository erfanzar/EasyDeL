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
"""Spectrax implementation of OpenAI CLIP (Contrastive Language-Image Pre-training).

CLIP couples a vision transformer encoder with a causal text transformer
encoder; both project into a shared embedding space where matched image-text
pairs have high cosine similarity. Training optimizes a symmetric InfoNCE
loss with a learnable temperature.

Loss helpers:

- :func:`contrastive_loss` — single-direction softmax cross-entropy.
- :func:`clip_loss` — symmetric image/text contrastive loss.

Building blocks:

- :class:`CLIPVisionEmbeddings`, :class:`CLIPTextEmbeddings` — patch/word +
  position embeddings (vision adds a learnable CLS token).
- :class:`CLIPAttention`, :class:`CLIPMLP`, :class:`CLIPEncoderLayer`,
  :class:`CLIPEncoder` — shared transformer stack.
- :class:`CLIPTextTransformer`, :class:`CLIPVisionTransformer` — encoder
  wrappers used by the text/vision sub-models.

Public model classes (registered with the factory):

- :class:`CLIPTextModel` — text encoder (+ EOS pooling).
- :class:`CLIPTextModelWithProjection` — text encoder + projection head.
- :class:`CLIPVisionModel` — vision encoder.
- :class:`CLIPForImageClassification` — vision encoder + linear classifier.
- :class:`CLIPModel` — joint dual-encoder for contrastive training and
  zero-shot image classification.
"""

import inspect
import typing as tp
from functools import cached_property, partial

import jax
import jax.numpy as jnp
import spectrax as spx
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int
from spectrax import apply_logical_sharding, common_types, nn

from easydel.infra.base_module import EasyDeLBaseModule, EasyDeLLayerStackMixin
from easydel.infra.factory import TaskType, register_module
from easydel.infra.loss_utils import LossMetrics
from easydel.infra.modeling_outputs import (
    AttentionLayerOutput,
    BaseModelOutput,
    BaseModelOutputWithPooling,
    CLIPOutput,
    CLIPTextModelOutput,
    EncoderLayerOutput,
    ImageClassifierOutput,
)
from easydel.infra.utils import ACT2FN, ArrayParam, auto_remat
from easydel.layers import ColumnParallelLinear, Embed
from easydel.layers.attention import AttentionModule, FlexibleAttentionModule
from easydel.layers.norms import LayerNorm
from easydel.modules._base import BaseImageClassificationModule

from .clip_configuration import CLIPConfig, CLIPTextConfig, CLIPVisionConfig


def contrastive_loss(logits: jax.Array) -> jax.Array:
    """Compute the one-direction softmax cross-entropy used by CLIP.

    Treats each row of ``logits`` as the similarities of a single anchor
    (e.g. one image) against all candidates (e.g. all texts in the batch);
    the correct match is on the diagonal. Returns the mean
    log-softmax cross-entropy with diagonal targets — i.e. the InfoNCE
    objective for that direction.

    Args:
        logits (jax.Array): Square similarity matrix of shape ``(N, N)``.

    Returns:
        jax.Array: Scalar mean cross-entropy loss.
    """
    labels = jnp.arange(len(logits))
    return jnp.mean(-jnp.sum(jax.nn.log_softmax(logits) * jax.nn.one_hot(labels, len(logits)), axis=-1))


def clip_loss(similarity: jax.Array) -> jax.Array:
    """Symmetric image/text contrastive loss used for CLIP pre-training.

    Averages :func:`contrastive_loss` applied to ``similarity`` (image -> text
    direction) and to its transpose (text -> image), so neither modality is
    privileged.

    Args:
        similarity (jax.Array): Square cosine similarity matrix of shape
            ``(batch, batch)`` already scaled by ``logit_scale``. The
            diagonal entries correspond to the correct image-text pairs.

    Returns:
        jax.Array: Scalar symmetric InfoNCE loss.
    """
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.T)
    return (caption_loss + image_loss) / 2.0


class CLIPVisionEmbeddings(spx.Module):
    """Patch + position + CLS embedding for the CLIP ViT tower.

    Computes the standard ViT input contract:

    1. ``patch_embedding``: a 2D conv with ``kernel_size = stride =
       config.patch_size`` slices the input image
       ``(batch, image_size, image_size, num_channels)`` into
       ``(image_size // patch_size) ** 2`` non-overlapping patch tokens of
       width ``hidden_size``.
    2. A learnable ``[CLS]`` token (``class_embedding``) is prepended,
       producing ``num_patches + 1`` tokens.
    3. Learned absolute position embeddings (``num_positions = num_patches +
       1``) are added.

    Output shape: ``(batch, num_patches + 1, hidden_size)``. The ``[CLS]``
    token at position 0 is what the vision tower's pooled head reads.
    """

    def __init__(
        self,
        config: CLIPVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize CLIP vision embeddings.

        Args:
            config (CLIPVisionConfig): Vision model configuration with embedding parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        embed_dim = config.hidden_size
        image_size = config.image_size
        patch_size = config.patch_size

        self.class_embedding = ArrayParam.bound(
            shape=(embed_dim,),
            dtype=param_dtype,
            init_method="normal",
            init_kwargs={"stddev": 0.02},
            key=rngs.parameters,
        )

        self.patch_embedding = nn.Conv2d(
            config.num_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding="VALID",
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
        )

        self.num_patches = (image_size // patch_size) ** 2
        num_positions = self.num_patches + 1
        self.position_embedding = Embed(
            num_positions,
            embed_dim,
            embedding_init=jax.nn.initializers.normal(),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def forward(self, pixel_values):
        """Create vision embeddings from pixel values.

        Args:
            pixel_values: Input pixel values of shape (batch_size, height, width, channels).

        Returns:
            Combined class token and patch embeddings with position encodings,
            shape (batch_size, num_patches + 1, hidden_size).
        """
        patch_embeds = self.patch_embedding(pixel_values)
        batch_size, height, width, channels = patch_embeds.shape
        patch_embeds = jnp.reshape(patch_embeds, (batch_size, height * width, channels))

        class_embeds = jnp.expand_dims(self.class_embedding.value, axis=(0, 1))
        class_embeds = jnp.tile(class_embeds, (batch_size, 1, 1))
        embeddings = jnp.concatenate([class_embeds, patch_embeds], axis=1)

        embeddings = embeddings + self.position_embedding(
            jnp.expand_dims(
                jnp.arange(0, ((self.config.image_size // self.config.patch_size) ** 2) + 1, dtype="i4"),
                axis=0,
            )
        )
        return checkpoint_name(embeddings, name="embeddings")


class CLIPTextEmbeddings(spx.Module):
    """Token + learned absolute position embedding for the CLIP text encoder.

    Looks up token ids in ``token_embedding`` (vocab ``config.vocab_size``,
    width ``hidden_size``), then adds a learned absolute position embedding
    sliced to the input sequence length (``config.max_position_embeddings``,
    typically 77). Output: ``(batch, seq_len, hidden_size)``.
    """

    def __init__(
        self,
        config: CLIPVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize CLIP text embeddings.

        Args:
            config (CLIPTextConfig): Text model configuration with embedding parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (spx.Rngs): Random number generator state.
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

    def forward(self, input_ids, position_ids):
        """Create text embeddings from token IDs.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length).
            position_ids: Position indices of shape (batch_size, sequence_length).

        Returns:
            Combined token and position embeddings of shape
            (batch_size, sequence_length, hidden_size).
        """
        input_embeds = self.token_embedding(input_ids.astype("i4"))
        position_embeds = self.position_embedding(position_ids.astype("i4"))

        embeddings = input_embeds + position_embeds
        return checkpoint_name(embeddings, name="embeddings")


class CLIPAttention(AttentionModule):
    """Multi-head self-attention shared by both CLIP encoders.

    Standard pre-norm MHA: ``q``/``k``/``v``/``out_proj`` are
    :class:`ColumnParallelLinear` projections of width ``hidden_size``
    (no GQA — ``num_heads = num_attention_heads``). The masking convention is
    config-driven: a :class:`CLIPTextConfig` instance forces a causal mask
    (so the text encoder is autoregressive in the contrastive sense, with
    EOS-token pooling), while a :class:`CLIPVisionConfig` instance disables
    causal masking so every patch can attend to every other patch.
    """

    def __init__(
        self,
        config: CLIPTextConfig | CLIPVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize CLIP attention layer.

        Args:
            config (CLIPTextConfig | CLIPVisionConfig): Model configuration with
                attention parameters. Determines causal/non-causal behavior.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (spx.Rngs): Random number generator state.
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

        self.causal = isinstance(config, CLIPTextConfig)
        self.attention_performer = FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=config.attention_dropout,
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

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
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
        query = checkpoint_name(self.q_proj(hidden_states), name="attn_query")
        key = checkpoint_name(self.k_proj(hidden_states), name="attn_key")
        value = checkpoint_name(self.v_proj(hidden_states), name="attn_value")

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        attentions = self.attention_performer.forward(
            query_states=query,
            key_states=key,
            value_states=value,
            mode=common_types.MODE_TRAIN,
            bias=None,
            mask_info=mask_info,
            causal=self.causal,
        )

        attn_output = self._merge_heads(attentions.attention_outputs)
        attn_output = checkpoint_name(self.out_proj(attn_output), name="attn_output")

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=None,
        )


class CLIPMLP(spx.Module):
    """Two-layer feed-forward block for CLIP encoder layers.

    Computes ``fc2(act(fc1(x)))`` where ``fc1`` widens to
    ``config.intermediate_size`` and ``fc2`` projects back to ``hidden_size``.
    The activation is selected by ``config.hidden_act`` (typically
    ``"quick_gelu"`` for CLIP). No gating; this is the original
    transformer-style MLP.
    """

    def __init__(
        self,
        config: CLIPTextConfig | CLIPVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize CLIP MLP block.

        Args:
            config (CLIPTextConfig | CLIPVisionConfig): Model configuration with MLP parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (spx.Rngs): Random number generator state.
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

    def forward(
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


class CLIPEncoderLayer(spx.Module):
    """One CLIP encoder block (pre-norm attention + MLP residuals).

    The forward path is the canonical pre-LN transformer block:
    ``x = x + attn(layer_norm1(x))`` then
    ``x = x + mlp(layer_norm2(x))``. Used identically by both the text
    encoder and the vision encoder; the only behavioural difference is the
    causal-mask flag on :class:`CLIPAttention`, which is set by the config
    type (text → causal, vision → bidirectional).
    """

    def __init__(
        self,
        config: CLIPTextConfig | CLIPVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize CLIP encoder layer.

        Args:
            config (CLIPTextConfig | CLIPVisionConfig): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.self_attn = CLIPAttention(
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
        self.mlp = CLIPMLP(
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

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
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
        hidden_states = checkpoint_name(residual + hidden_states, name="residual")

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = checkpoint_name(residual + hidden_states, name="residual")
        hidden_states = checkpoint_name(hidden_states, name="layer_output")

        return EncoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
        )


class CLIPEncoder(EasyDeLLayerStackMixin, spx.Module):
    """Stack of :class:`CLIPEncoderLayer` blocks, scan-friendly.

    Holds ``config.num_hidden_layers`` encoder blocks in a
    :class:`spx.nn.ModuleList`. Inherits :class:`EasyDeLLayerStackMixin` so
    the layer loop runs through ``self.layers.scan(...)`` (single trace,
    optional remat) and so cache-views / hidden-state collection follow the
    same contract as the decoder stacks elsewhere in EasyDeL. Used by both
    :class:`CLIPTextTransformer` and :class:`CLIPVisionTransformer`.
    """

    def __init__(
        self,
        config: CLIPTextConfig | CLIPVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize CLIP encoder.

        Args:
            config (CLIPTextConfig | CLIPVisionConfig): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        remat_layer_block = auto_remat(
            CLIPEncoderLayer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.layers = nn.ModuleList([])
        for _ in range(config.num_hidden_layers):
            with self.assign_layer_stage(_, total_layers=config.num_hidden_layers):
                self.layers.append(
                    remat_layer_block(
                        config=config,
                        dtype=dtype,
                        param_dtype=param_dtype,
                        precision=precision,
                        rngs=rngs,
                    )
                )

    @cached_property
    def causal_mask(self):
        """Get causal mask for text encoder.

        Returns:
            Causal attention mask for text encoder, None for vision encoder.
        """
        if isinstance(self.config, CLIPTextConfig):
            return self.config.get_basic_causal_mask()
        return None

    def forward(
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

        def _layer_loop(layer, carry):
            hidden_states, all_hidden_states, all_attentions, idx = carry
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            with self._layer_stage_context(idx, layers=self.layers):
                layer_outputs = layer(
                    hidden_states=hidden_states,
                    mask_info=mask_info,
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


class CLIPTextTransformer(EasyDeLBaseModule):
    """Causal text encoder for CLIP — embeddings + encoder + EOS pooling.

    Pipeline: :class:`CLIPTextEmbeddings` -> :class:`CLIPEncoder` (causal) ->
    final ``LayerNorm``. The pooled representation used downstream for
    contrastive similarity is the hidden state at the EOS position
    (``argmax(input_ids, axis=-1)``, matching OpenAI's reference
    tokenization where EOS is the highest-id token in the sequence). Returns
    a :class:`BaseModelOutputWithPooling` carrying both the per-token hidden
    states and the EOS-pooled vector.
    """

    def __init__(
        self,
        config: CLIPTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize CLIP text transformer.

        Args:
            config (CLIPTextConfig): Text model configuration.
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
        self.embeddings = CLIPTextEmbeddings(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.encoder = CLIPEncoder(
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

        self.eos_token_id = self.config.eos_token_id

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"],
        mask_info: MaskInfo,
        position_ids: Int[Array, "batch seq_len"],
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = False,
    ):
        """Forward pass through the text transformer.

        Processes text tokens through embeddings, encoder layers with causal attention,
        and produces pooled output from the EOS token position.

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

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)
        mask_info = MaskInfo.dynamic_init(mask_info=mask_info, input_ids=input_ids)
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            mask_info=mask_info,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = checkpoint_name(self.final_layer_norm(last_hidden_state), name="model_output")

        if self.eos_token_id == 2:
            pooled_output = last_hidden_state[
                jnp.arange(last_hidden_state.shape[0]),
                input_ids.argmax(axis=-1),
            ]
        else:
            pooled_output = last_hidden_state[
                jnp.arange(last_hidden_state.shape[0]),
                (input_ids == self.eos_token_id).argmax(axis=-1),
            ]

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


class CLIPVisionTransformer(EasyDeLBaseModule):
    """Bidirectional ViT image encoder used by CLIP.

    Pipeline: :class:`CLIPVisionEmbeddings` (Conv2d patchify + CLS +
    learned position embedding) -> ``pre_layrnorm`` -> :class:`CLIPEncoder`
    (bidirectional self-attention) -> ``post_layernorm`` applied to the
    ``[CLS]`` token. Pixel input contract is
    ``(batch, image_size, image_size, num_channels)`` (NHWC); the output is
    a :class:`BaseModelOutputWithPooling` whose ``pooler_output`` is the
    layer-normed CLS hidden state — this is the vector projected into the
    shared image-text embedding space by :class:`CLIPModel`.
    """

    def __init__(
        self,
        config: CLIPVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize CLIP vision transformer.

        Args:
            config (CLIPVisionConfig): Vision model configuration.
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
        self.embeddings = CLIPVisionEmbeddings(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.pre_layrnorm = LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.encoder = CLIPEncoder(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.post_layernorm = LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def forward(
        self,
        pixel_values: Array | None = None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        """Forward pass through the vision transformer.

        Processes images through patch embeddings, pre-layer norm, encoder layers,
        and produces pooled output from the class token.

        Args:
            pixel_values: Input pixel values of shape (batch_size, channels, height, width)
                or (batch_size, height, width, channels). Automatically transposed if needed.
            output_attentions: Whether to return attention weights. Defaults to config value.
            output_hidden_states: Whether to return hidden states from all layers.
                Defaults to config value.

        Returns:
            BaseModelOutputWithPooling containing last hidden state, pooled output
            from class token, optional hidden states, and optional attention weights.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if pixel_values is not None and pixel_values.ndim == 4:
            # Convert from NCHW (PyTorch) to NHWC (JAX)
            pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = checkpoint_name(encoder_outputs.last_hidden_state, name="model_output")
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

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
        This vision model does not have a language model head.
        """
        raise NotImplementedError("This vision model does not have a language model head.")

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.embeddings


@register_module(config=CLIPTextConfig, model_type="clip_text_model", task_type=TaskType.BASE_MODULE)
class CLIPTextModel(EasyDeLBaseModule):
    """Top-level CLIP text encoder (registered under ``clip_text_model``).

    Thin wrapper over :class:`CLIPTextTransformer` exposing the standard
    EasyDeL ``BASE_MODULE`` interface (``compute_embedding``, ``get_encoder``,
    …). Returns a :class:`BaseModelOutputWithPooling` — same contract as the
    underlying transformer, just promoted to a registered module so
    :class:`AutoEasyDeLModel` can resolve it from a config alone.
    """

    def __init__(
        self,
        config: CLIPTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize CLIP text model.

        Args:
            config (CLIPTextConfig): Text model configuration.
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
        self.text_model = CLIPTextTransformer(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"],
        mask_info: MaskInfo | None = None,
        attention_mask: Array | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Forward pass through the CLIP text model.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length).
            mask_info: Pre-computed attention mask information. Overrides attention_mask if provided.
            attention_mask: Boolean mask of shape (batch_size, sequence_length) indicating
                which tokens to attend to. Auto-generated as all-ones if not provided.
            position_ids: Position indices of shape (batch_size, sequence_length).
                Auto-generated if not provided.
            output_attentions: Whether to return attention weights. Defaults to False.
            output_hidden_states: Whether to return hidden states from all layers.
                Defaults to False.

        Returns:
            BaseModelOutputWithPooling containing last hidden state, pooled output,
            optional hidden states, and optional attention weights.
        """
        batch_size, seq_len = input_ids.shape

        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[None, :], (batch_size, seq_len))

        if mask_info is None:
            if attention_mask is None:
                attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)
            mask_info = MaskInfo.from_attention_mask(attention_mask)

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
        Base Models don't have a Language Model Head.
        """
        raise NotImplementedError("The base model does not have a language model head.")

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.text_model.embeddings


class CLIPTextModelWithProjection(EasyDeLBaseModule):
    """CLIP text encoder + linear projection into the shared embedding space.

    Composes :class:`CLIPTextTransformer` with an unbiased ``text_projection``
    of shape ``(hidden_size, projection_dim)``. The projection is applied to
    the EOS-pooled hidden state, producing a vector directly comparable
    (via cosine similarity, after ``l2`` norm) with the image projection
    from :class:`CLIPVisionModel`. Returns :class:`CLIPTextModelOutput`.
    """

    def __init__(
        self,
        config: CLIPTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize CLIP text model with projection.

        Args:
            config (CLIPTextConfig): Text model configuration with projection_dim.
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
        self.text_model = CLIPTextTransformer(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.text_projection = ColumnParallelLinear(
            config.hidden_size,
            config.projection_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"],
        mask_info: MaskInfo,
        position_ids: Int[Array, "batch seq_len"],
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> CLIPTextModelOutput:
        """Forward pass through the CLIP text model with projection.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length).
            mask_info: Attention mask information for masking padded tokens.
            position_ids: Position indices of shape (batch_size, sequence_length).
            output_attentions: Whether to return attention weights. Defaults to False.
            output_hidden_states: Whether to return hidden states from all layers.
                Defaults to False.

        Returns:
            CLIPTextModelOutput containing projected text embeddings, last hidden state,
            optional hidden states, and optional attention weights.
        """
        text_outputs = self.text_model(
            input_ids=input_ids,
            mask_info=mask_info,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = text_outputs[1]
        text_embeds = checkpoint_name(self.text_projection(pooled_output), name="text_projection_output")

        return CLIPTextModelOutput(
            text_embeds=text_embeds,
            last_hidden_state=text_outputs.last_hidden_state,
            hidden_states=text_outputs.hidden_states,
            attentions=text_outputs.attentions,
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


@register_module(config=CLIPVisionConfig, model_type="clip_vision_model", task_type=TaskType.BASE_VISION)
@register_module(config=CLIPVisionConfig, model_type="clip_vision_model", task_type=TaskType.BASE_MODULE)
class CLIPVisionModel(EasyDeLBaseModule):
    """Top-level CLIP vision encoder (registered under ``clip_vision_model``).

    Thin wrapper over :class:`CLIPVisionTransformer` exposing the registered
    ``BASE_MODULE`` interface; pixel input contract is the NHWC ``(batch,
    image_size, image_size, num_channels)`` tensor expected by the underlying
    Conv2d patch embedder. Returns :class:`BaseModelOutputWithPooling` with
    the ``[CLS]`` token's post-layernorm vector as ``pooler_output``.
    """

    def __init__(
        self,
        config: CLIPVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize CLIP vision model.

        Args:
            config (CLIPVisionConfig): Vision model configuration.
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
        self.vision_model = CLIPVisionTransformer(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def forward(
        self,
        pixel_values: Array,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Forward pass through the CLIP vision model.

        Args:
            pixel_values: Input pixel values of shape (batch_size, channels, height, width)
                or (batch_size, height, width, channels).
            output_attentions: Whether to return attention weights. Defaults to False.
            output_hidden_states: Whether to return hidden states from all layers.
                Defaults to False.

        Returns:
            BaseModelOutputWithPooling containing last hidden state, pooled output,
            optional hidden states, and optional attention weights.
        """
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
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


@register_module(config=CLIPVisionConfig, model_type="clip", task_type=TaskType.IMAGE_CLASSIFICATION)
class CLIPForImageClassification(BaseImageClassificationModule[CLIPVisionTransformer, CLIPVisionConfig]):  # type: ignore
    """CLIP vision tower + linear ``classifier`` head for supervised image classification.

    Backbone: :class:`CLIPVisionTransformer`. Pooling: mean over the patch
    tokens (the ``[CLS]`` token at position 0 is excluded), giving one
    ``hidden_size`` vector per image. Head: a single ``classifier`` linear
    of shape ``(hidden_size, num_labels)``. Suitable for fine-tuning the
    CLIP image encoder on closed-vocabulary classification benchmarks
    (ImageNet, etc.). Pixel input contract matches
    :class:`CLIPVisionTransformer`.
    """

    def __init__(
        self,
        config: CLIPVisionConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize CLIP for image classification.

        Args:
            config (CLIPVisionConfig): Vision model configuration with num_labels.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=CLIPVisionTransformer,
            base_model_name="vision_model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            pooling_strategy="mean",
        )

    def forward(
        self,
        pixel_values: Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> tuple | ImageClassifierOutput:
        """Forward pass for image classification.

        Args:
            pixel_values: Input pixel values of shape (batch_size, channels, height, width)
                or (batch_size, height, width, channels).
            output_attentions: Whether to return attention weights. Defaults to config value.
            output_hidden_states: Whether to return hidden states from all layers.
                Defaults to config value.

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
        )

        sequence_output = outputs[0]

        sequence_output = jnp.mean(sequence_output[:, 1:, :], axis=1)
        if self.config.num_labels > 0:
            logits = self.classifier(sequence_output)
        else:
            logits = sequence_output

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
        return self.vision_model.embeddings

    def get_task_head(self):
        """Returns the image classification head."""
        return self.classifier


@register_module(config=CLIPConfig, model_type="clip", task_type=TaskType.ZERO_SHOT_IMAGE_CLASSIFICATION)
class CLIPModel(EasyDeLBaseModule):
    """Joint dual-encoder CLIP model (registered under ``ZERO_SHOT_IMAGE_CLASSIFICATION``).

    Combines :class:`CLIPTextTransformer` and :class:`CLIPVisionTransformer`
    with two unbiased projection heads (``text_projection``,
    ``visual_projection``) of shape ``(text/vision_hidden, projection_dim)``
    that map both modalities into a shared embedding space. A learnable
    scalar ``logit_scale`` (initialised at ``log(1 / 0.07)``, exponentiated
    at use, optionally clipped by ``logit_scale_init_value``) controls the
    softmax temperature of the contrastive similarity. Forward returns a
    :class:`CLIPOutput` with both image and text embeddings, the
    ``logits_per_image`` / ``logits_per_text`` similarity matrices, and the
    symmetric InfoNCE :func:`clip_loss` when training. Used at inference for
    zero-shot classification (text prompts as classes) and image-text
    retrieval.
    """

    def __init__(
        self,
        config: CLIPConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize CLIP model.

        Args:
            config (CLIPConfig): Combined text and vision configuration.
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
        text_config = self.config.get_text_config()
        vision_config = self.config.vision_config

        self.projection_dim = self.config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = CLIPTextTransformer(
            text_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.vision_model = CLIPVisionTransformer(
            vision_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        linear_class = partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(0.02),
            use_bias=False,
            rngs=rngs,
        )
        self.visual_projection = linear_class(config.vision_config.hidden_size, self.projection_dim)
        self.text_projection = linear_class(config.get_text_config().hidden_size, self.projection_dim)

        self.logit_scale = ArrayParam.bound(
            shape=(),
            dtype=jnp.float32,
            init_method="ones",
            key=None,
            value=jnp.ones((), dtype=jnp.float32) * self.config.logit_scale_init_value,
        )

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None,
        pixel_values: Array,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        output_attentions=None,
        output_hidden_states=None,
    ) -> CLIPOutput:
        """Performs forward pass through CLIP's dual-encoder architecture.

        Processes images through the vision transformer and text through the text transformer,
        projects both to a shared embedding space, and computes contrastive similarity logits
        between all image-text pairs in the batch.

        Args:
            input_ids: Text token IDs of shape (batch_size, sequence_length). Tokenized text
                inputs for the text encoder.
            pixel_values: Image pixel values of shape (batch_size, channels, height, width)
                or (batch_size, height, width, channels) depending on data format. Preprocessed
                images for the vision encoder.
            attention_mask: Boolean mask of shape (batch_size, sequence_length) for text,
                indicating which tokens to attend to. Auto-generated as all-ones if not provided.
            mask_info: Pre-computed mask information for text. If provided, overrides
                `attention_mask`.
            position_ids: Explicit position indices of shape (batch_size, sequence_length)
                for text. Auto-generated from attention_mask if not provided.
            output_attentions: Whether to return attention weights from vision and text encoders.
            output_hidden_states: Whether to return hidden states from all layers of both encoders.

        Returns:
            CLIPOutput containing:
                - logits_per_image: Similarity scores of shape (batch, batch) where entry [i,j]
                    is the scaled cosine similarity between image i and text j
                - logits_per_text: Transposed similarity scores of shape (batch, batch)
                - text_embeds: L2-normalized text embeddings of shape (batch, projection_dim)
                - image_embeds: L2-normalized image embeddings of shape (batch, projection_dim)
                - text_model_output: Full output from text encoder including hidden states
                - vision_model_output: Full output from vision encoder including hidden states
        """
        if attention_mask is None and input_ids is not None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None and attention_mask is not None:
            position_ids = attention_mask.cumsum(-1) - 1

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        image_embeds = vision_outputs[1]
        image_embeds = checkpoint_name(self.visual_projection(image_embeds), name="visual_projection_output")

        text_embeds = text_outputs[1]
        text_embeds = checkpoint_name(self.text_projection(text_embeds), name="text_projection_output")

        image_embeds = image_embeds / jnp.linalg.norm(image_embeds, axis=-1, keepdims=True)
        text_embeds = text_embeds / jnp.linalg.norm(text_embeds, axis=-1, keepdims=True)

        logit_scale = jnp.exp(self.logit_scale)
        logits_per_text = jnp.matmul(text_embeds, image_embeds.T) * logit_scale
        logits_per_image = logits_per_text.T

        return CLIPOutput(
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )

    def get_text_features(
        self,
        input_ids: Int[Array, "batch seq_len"],
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
    ):
        """Extract text features from input tokens.

        Processes text through the text encoder and projects to the shared
        embedding space.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length).
            attention_mask: Boolean mask indicating tokens to attend to.
            mask_info: Pre-computed mask information.
            position_ids: Position indices for tokens.

        Returns:
            Projected text features of shape (batch_size, projection_dim).
        """
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
        )
        pooled_output = text_outputs[1]
        text_features = checkpoint_name(self.text_projection(pooled_output), name="text_projection_output")
        return text_features

    def get_image_features(self, pixel_values: Array):
        """Extract image features from pixel values.

        Processes images through the vision encoder and projects to the shared
        embedding space.

        Args:
            pixel_values: Input pixel values of shape (batch_size, channels, height, width)
                or (batch_size, height, width, channels).

        Returns:
            Projected image features of shape (batch_size, projection_dim).
        """
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        pooled_output = vision_outputs[1]  # pooled_output
        image_features = checkpoint_name(self.visual_projection(pooled_output), name="visual_projection_output")
        return image_features

    def compute_loss(
        self,
        *,
        labels=None,  # just to extract
        loss_config=None,  # just to extract
        loss_kwargs=None,  # just to extract
        **batch,
    ) -> tuple[tp.Any, CLIPOutput]:
        """Compute contrastive loss for CLIP training.

        Calculates the symmetric cross-entropy loss between image-text similarity
        scores for contrastive learning.

        Args:
            labels: Unused, extracted for API compatibility.
            loss_config: Unused, extracted for API compatibility.
            loss_kwargs: Unused, extracted for API compatibility.
            **batch: Input batch containing input_ids, pixel_values, and masks.

        Returns:
            Tuple of (CLIPOutput with loss, LossMetrics).
        """
        forward_batch = batch
        try:
            call_signature = inspect.signature(self.forward)
        except (TypeError, ValueError):
            call_signature = None
        if call_signature is not None:
            call_parameters = call_signature.parameters
            if not any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in call_parameters.values()):
                accepted_keys = set(call_parameters.keys())
                forward_batch = {key: value for key, value in batch.items() if key in accepted_keys}
        if forward_batch.get("input_ids", None) is not None and forward_batch.get("inputs_embeds", None) is not None:
            forward_batch = dict(forward_batch)
            forward_batch.pop("inputs_embeds", None)
        outputs = self(**forward_batch)

        loss = LossMetrics(loss=clip_loss(outputs.logits_per_text))
        outputs = outputs.replace(loss=loss.loss)
        return outputs, loss  # pyright: ignore[reportReturnType]

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        The vision tower acts as the encoder in this multi-modal setup.
        """
        return self.vision_model

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        The text model acts as the "decoder" or text processor in this multi-modal setup.
        """
        return self.text_model

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        This model does not have a traditional language model head, but projection heads.
        """
        raise NotImplementedError("This model does not have a language model head.")

    def get_embedding(self):
        """
        Returns the embedding layer of the text model.
        """
        return self.text_model.embeddings
