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

import jax
import jax.numpy as jnp
from eformer import common_types
from eformer.escale import apply_logical_sharding
from eformer.pytree import auto_pytree
from ejkernel.types import MaskInfo
from flax import nnx as nn
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import (
    DecoderLayerOutput,
    ModelOutput,
    MoeModelOutput,
    VLMCausalLMOutput,
)
from easydel.infra.utils import auto_remat
from easydel.layers.attention_unified import UnifiedAttention
from easydel.layers.base_modules import BaseVisionLanguageModule
from easydel.layers.caching import (
    HybridCache,
    OperationsMetadata,
    RaggedPagesCache,
    RaggedPagesCacheView,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.layers.components import Embed, RMSNorm, RowParallelLinear
from easydel.modules.glm4_moe.modeling_glm4_moe import Glm4MoeMLP, Glm4MoeMoE
from easydel.modules.glm4v.modeling_glm4v import Glm4vModel, Glm4vVisionModel

from .glm4v_moe_configuration import Glm4vMoeConfig, Glm4vMoeTextConfig, Glm4vMoeVisionConfig


@auto_pytree
class Glm4vMoeModelOutputWithPast(ModelOutput):
    """Base model output for GLM4V-MoE multimodal model.

    Contains outputs from the GLM4V-MoE multimodal model including hidden states,
    cached key-values for generation, position information for multimodal
    rotary position embeddings, and MoE-specific routing information.

    Attributes:
        last_hidden_state (Array | None): Hidden states from the final layer.
        past_key_values (TransformerCache | None): Cached key-value states for generation.
        hidden_states (tuple[Array] | None): Hidden states from all layers if requested.
        attentions (tuple[Array] | None): Attention weights from all layers if requested.
        rope_deltas (Array | None): Position deltas for multimodal rotary embeddings.
        router_logits (tuple[Array] | None): Router logits from MoE layers for load balancing.
        all_router_losses (tuple[Array] | None): Auxiliary routing losses from MoE layers.
    """

    last_hidden_state: Array | None = None
    past_key_values: TransformerCache | None = None
    hidden_states: tuple[Array] | None = None
    attentions: tuple[Array] | None = None
    rope_deltas: Array | None = None
    router_logits: tuple[Array] | None = None
    all_router_losses: tuple[Array] | None = None


class Glm4vMoeTextAttention(UnifiedAttention):
    """GLM4V-MoE attention with bias-free output projection.

    Implements multi-head self-attention with GPT-J style rotary position
    embeddings and grouped-query attention support for the text decoder.
    Uses bias-free output projection for HuggingFace compatibility.
    """

    def __init__(
        self,
        config: Glm4vMoeTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        """Initialize text decoder attention layer.

        Args:
            config (Glm4vMoeTextConfig): Text decoder configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
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
            use_qk_norm=getattr(config, "use_qk_norm", False),
        )

    def _create_o_proj(
        self,
        config: Glm4vMoeTextConfig,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        precision: jax.lax.PrecisionLike,
        rngs: nn.Rngs,
    ) -> RowParallelLinear:
        """Create the output projection layer without bias.

        Args:
            config (Glm4vMoeTextConfig): Text decoder configuration.
            dtype (jnp.dtype): Data type for computation.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): Numerical precision for operations.
            rngs (nn.Rngs): Random number generator state.

        Returns:
            RowParallelLinear: Output projection layer.
        """
        return RowParallelLinear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            rngs=rngs,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(getattr(config, "initializer_range", 0.02)),
            precision=precision,
        )


class Glm4vMoeTextDecoderLayer(nn.Module):
    """Single decoder block for GLM4V-MoE with attention and MoE MLP.

    Implements a transformer decoder layer with pre-normalization,
    self-attention, and either a dense MLP or Mixture-of-Experts MLP
    depending on the layer index. The first `first_k_dense_replace` layers
    use dense MLPs, while remaining layers use MoE.
    """

    def __init__(
        self,
        config: Glm4vMoeTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        """Initialize text decoder layer.

        Args:
            config (Glm4vMoeTextConfig): Text decoder configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the decoder. Determines whether
                to use dense MLP or MoE based on first_k_dense_replace config.
        """
        self.config = config
        self.layer_idx = layer_idx

        attn_block = Glm4vMoeTextAttention
        mlp_block = Glm4MoeMLP if layer_idx < config.first_k_dense_replace else Glm4MoeMoE

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
        )

        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ):
        """Forward pass through the text decoder layer.

        Applies pre-normalization architecture with attention and MoE/dense MLP:
        x + attn(norm(x)) followed by x + mlp(norm(x))

        Args:
            hidden_states (Array): Input tensor of shape (batch_size, sequence_length, hidden_dim).
            mask_info (MaskInfo): Attention mask information including causal masks.
            position_ids (Array): Position indices for tokens, shape (batch_size, sequence_length).
            mode (RUNTIME_MODE_TYPES): Runtime mode (train, decode, etc.) for optimization.
            cache_view (TransformerCacheView | RaggedPagesCacheView | None, optional): Cache view.
                Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Cache metadata. Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            output_router_logits (bool, optional): Whether to return MoE router logits. Defaults to False.
            frequencies (Array | None, optional): Precomputed RoPE frequencies. Defaults to None.

        Returns:
            DecoderLayerOutput: Contains hidden states, attention weights, cache view,
                and router logits from MoE layers.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
            frequencies=frequencies,
        )
        hidden_states = residual + attn_outputs.attention_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states, router_logits = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
            cache_view=attn_outputs.cache_view,
            router_logits=router_logits,
        )


@register_module(TaskType.BASE_VISION, config=Glm4vMoeConfig, model_type="glm4v_moe")
class Glm4vMoeVisionModel(Glm4vVisionModel):
    """Vision transformer encoder for GLM4V-MoE.

    Processes images and videos through patch embedding, transformer blocks with
    2D rotary position embeddings, spatial downsampling, and patch merging to
    produce vision features for the multimodal language model.

    Inherits all functionality from Glm4vVisionModel as the vision architecture
    is shared between GLM4V and GLM4V-MoE variants.

    Attributes:
        config_class: The configuration class for this model (Glm4vMoeVisionConfig).
    """

    config_class = Glm4vMoeVisionConfig


@register_module(TaskType.BASE_MODULE, config=Glm4vMoeTextConfig, model_type="glm4v_moe")
class Glm4vMoeTextModel(EasyDeLBaseModule):
    """GLM4V-MoE text decoder model with Mixture-of-Experts.

    Implements the text decoder component of GLM4V-MoE, utilizing transformer
    blocks with RMSNorm, rotary position embeddings, and a hybrid architecture
    where the first `first_k_dense_replace` layers use dense MLPs and the
    remaining layers use Mixture-of-Experts for improved capacity.

    Attributes:
        config_class: The configuration class for this model.
        embed_tokens: Token embedding layer.
        layers: List of decoder layers (dense and MoE).
        norm: Final layer normalization.
    """

    config_class = Glm4vMoeTextConfig

    def __init__(
        self,
        config: Glm4vMoeTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize GLM4V-MoE text decoder.

        Args:
            config (Glm4vMoeTextConfig): Text decoder configuration including
                MoE parameters like first_k_dense_replace and num_experts.
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
        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            rngs=rngs,
        )
        self.layers = nn.List([
            Glm4vMoeTextDecoderLayer(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
                layer_idx=i,
            )
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs)

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
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
    ) -> MoeModelOutput:
        """Forward pass through the GLM4V-MoE text decoder.

        Processes input tokens through embedding, all decoder layers with RoPE,
        RMSNorm, and Mixture-of-Experts, then final normalization.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch_size, sequence_length).
                Must be provided if inputs_embeds is None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings of shape
                (batch_size, sequence_length, hidden_size). Defaults to None.
            attention_mask (Array | None, optional): Boolean mask to avoid attention on padding tokens,
                shape (batch_size, sequence_length). Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information for attention operations.
                Defaults to None.
            position_ids (Array | None, optional): Position indices for each token, shape
                (batch_size, sequence_length). Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode (train/decode) for optimizations.
                Auto-detected if None. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cache with precomputed key-value states for generation. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache management. Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention weights from all layers.
                Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return hidden states from all layers.
                Defaults to None.
            output_router_logits (bool | None, optional): Whether to return router logits from MoE layers.
                Defaults to None.

        Returns:
            MoeModelOutput: Contains last_hidden_state, optional all hidden_states, optional attentions,
                updated past_key_values, router_logits, and auxiliary router losses.

        Raises:
            ValueError: If both input_ids and inputs_embeds are provided or both are None.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else getattr(self.config, "output_router_logits", False)
        )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.astype("i4"))

        sequence_length = inputs_embeds.shape[1]
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        all_router_logits = () if output_router_logits else None

        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        if position_ids is None:
            position_ids = mask_info.q_position_ids

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
                output_router_logits=output_router_logits,
                frequencies=self.frequencies,
            )
            hidden_states = layer_outputs.hidden_states
            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)
            if output_router_logits:
                all_router_logits += (layer_outputs.router_logits,)
            past_key_values[idx] = layer_outputs.cache_view

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return MoeModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            past_key_values=past_key_values,
            router_logits=all_router_logits,
            all_router_losses=getattr(layer_outputs, "all_router_losses", None),
        )

    def get_decoder(self):
        """Return the decoder (text model itself).

        Returns:
            Glm4vMoeTextModel: This text decoder model.
        """
        return self

    def get_embedding(self):
        """Return the token embedding layer.

        Returns:
            Embed: The token embedding layer.
        """
        return self.embed_tokens


@register_module(TaskType.VISION_LM, config=Glm4vMoeConfig, model_type="glm4v_moe")
class Glm4vMoeModel(Glm4vModel):
    """GLM4V-MoE multimodal model integrating vision encoder and MoE text decoder.

    Combines a vision transformer encoder with a MoE-enhanced text decoder to process
    both image/video and text inputs, supporting multimodal understanding
    and generation with 3D rotary position embeddings for spatial-temporal reasoning.

    Inherits multimodal glue logic from Glm4vModel including position encoding
    computation, embedding merging, and vision feature extraction. The key difference
    is the use of Glm4vMoeTextModel as the language model component.

    Attributes:
        visual (Glm4vMoeVisionModel): Vision encoder for processing images/videos.
        language_model (Glm4vMoeTextModel): MoE-enhanced text decoder for language understanding.
    """

    def __init__(
        self,
        config: Glm4vMoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize GLM4V-MoE multimodal model.

        Args:
            config (Glm4vMoeConfig): Full model configuration including vision, text,
                and MoE parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        EasyDeLBaseModule.__init__(
            self,
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.visual = Glm4vMoeVisionModel(
            config.vision_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.language_model = Glm4vMoeTextModel(
            config.text_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
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
    ) -> Glm4vMoeModelOutputWithPast:
        """Forward pass through the GLM4V-MoE multimodal model.

        Processes text and image/video inputs through the vision encoder and
        MoE text decoder to produce hidden representations with router information.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch, seq_len).
            attention_mask (Array | None, optional): Attention mask of shape (batch, seq_len).
            position_ids (Array | None, optional): Position IDs for M-RoPE,
                shape (3, batch, seq_len) or (batch, seq_len).
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cached key-value states for generation.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings.
            output_attentions (bool | None, optional): Whether to return attention weights.
            output_hidden_states (bool | None, optional): Whether to return all hidden states.
            output_router_logits (bool | None, optional): Whether to return MoE router logits.
            pixel_values (Array | None, optional): Image pixel values.
            pixel_values_videos (Array | None, optional): Video pixel values.
            image_grid_thw (Array | None, optional): Grid dimensions for images.
            video_grid_thw (Array | None, optional): Grid dimensions for videos.
            rope_deltas (Array | None, optional): Pre-computed rope position deltas.
            cache_position (Array | None, optional): Cache position (unused).
            mask_info (MaskInfo | None, optional): Attention mask information.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode for optimization.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Cache metadata for efficient caching.
            **kwargs: Additional unused arguments.

        Returns:
            Glm4vMoeModelOutputWithPast: Model outputs including hidden states, cache,
                rope deltas, router logits, and auxiliary router losses.
        """
        del cache_position

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else getattr(self.config, "output_router_logits", False)
        )

        if inputs_embeds is None:
            inputs_embeds = self.compute_embedding(
                input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
            )

        if position_ids is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw if pixel_values is not None else None,
                video_grid_thw=video_grid_thw if pixel_values_videos is not None else None,
                attention_mask=attention_mask,
            )
        elif position_ids.ndim == 2:
            batch_size, seq_len = position_ids.shape
            position_ids = jnp.broadcast_to(position_ids[None, :, :], (3, batch_size, seq_len))

        text_outputs = self.language_model(
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
            output_router_logits=output_router_logits,
        )

        return Glm4vMoeModelOutputWithPast(
            last_hidden_state=text_outputs.last_hidden_state,
            past_key_values=text_outputs.past_key_values,
            hidden_states=text_outputs.hidden_states,
            attentions=text_outputs.attentions,
            rope_deltas=rope_deltas,
            router_logits=text_outputs.router_logits,
            all_router_losses=text_outputs.all_router_losses,
        )


@register_module(TaskType.IMAGE_TEXT_TO_TEXT, config=Glm4vMoeConfig, model_type="glm4v_moe")
class Glm4vMoeForConditionalGeneration(BaseVisionLanguageModule[Glm4vMoeModel, Glm4vMoeConfig]):
    """GLM4V-MoE model for conditional generation.

    Vision-language model that combines a vision encoder with a MoE-enhanced text decoder
    and language modeling head for generating text conditioned on images,
    videos, and/or text prompts.

    Supports both image and video understanding with 3D rotary position
    embeddings for spatial-temporal reasoning and Mixture-of-Experts for
    improved model capacity.

    Attributes:
        _task_type: The task type for this model (IMAGE_TEXT_TO_TEXT).
        _model_type: Model type identifier ("glm4v_moe").
        _supports_video: Whether this model supports video inputs (True).
        _uses_mrope: Whether this model uses multimodal RoPE (True).
        vocab_size: Vocabulary size from the text configuration.
    """

    _task_type = TaskType.IMAGE_TEXT_TO_TEXT
    _model_type = "glm4v_moe"
    _config_class = Glm4vMoeConfig
    _auto_register = False
    _supports_video = True
    _uses_mrope = True

    _vision_tower_name = "visual"
    _projector_name = "merger"
    _language_model_name = "language_model"

    loss_type = "ForCausalLM"

    def __init__(
        self,
        config: Glm4vMoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize GLM4V-MoE for conditional generation.

        Args:
            config (Glm4vMoeConfig): Full model configuration including vision, text,
                MoE parameters, and generation settings.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=Glm4vMoeModel,
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
            router_aux_loss_coef=getattr(config.text_config, "router_aux_loss_coef", None),
            tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
            lm_head_bias=False,
        )
        self.vocab_size = config.text_config.vocab_size

    @property
    def visual(self):
        """Return the vision encoder.

        Returns:
            Glm4vMoeVisionModel: The vision encoder from the base model.
        """
        return self.base_model.visual

    @property
    def language_model(self):
        """Return the language model decoder.

        Returns:
            Glm4vMoeTextModel: The MoE language model from the base model.
        """
        return self.base_model.language_model

    def get_video_features(self, pixel_values_videos: Array, video_grid_thw: Array | None = None, **kwargs):
        """Extract video features using the vision encoder.

        Args:
            pixel_values_videos (Array): Video pixel values.
            video_grid_thw (Array | None, optional): Grid dimensions for videos.
            **kwargs: Additional arguments (unused).

        Returns:
            tuple[Array, ...]: Video embeddings split by video.
        """
        return self.base_model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: Array, image_grid_thw: Array | None = None, **kwargs):
        """Extract image features using the vision encoder.

        Args:
            pixel_values (Array): Image pixel values.
            image_grid_thw (Array | None, optional): Grid dimensions for images.
            **kwargs: Additional arguments (unused).

        Returns:
            tuple[Array, ...]: Image embeddings split by image.
        """
        return self.base_model.get_image_features(pixel_values, image_grid_thw)

    def compute_embedding(self, input_ids, *args, **kwargs):
        """Compute multimodal embeddings.

        Delegates to the base model's compute_embedding method.

        Args:
            input_ids: Input token IDs.
            *args: Positional arguments passed to base model.
            **kwargs: Keyword arguments passed to base model.

        Returns:
            Array: Merged multimodal embeddings.
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
        output_router_logits: bool | None = None,
        pixel_values: Array | None = None,
        pixel_values_videos: Array | None = None,
        image_grid_thw: Array | None = None,
        video_grid_thw: Array | None = None,
        rope_deltas: Array | None = None,
        cache_position: Array | None = None,
        **kwargs,
    ) -> VLMCausalLMOutput:
        """Forward pass for conditional generation.

        Processes multimodal inputs through the vision encoder and MoE text decoder,
        then applies the language modeling head to produce logits for generation.

        Args:
            input_ids (Array, optional): Input token IDs of shape (batch, seq_len).
            attention_mask (Array | None, optional): Attention mask of shape (batch, seq_len).
            mask_info (MaskInfo | None, optional): Attention mask information.
            position_ids (Array | None, optional): Position IDs for M-RoPE.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode for optimization.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cached key-value states for generation.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Cache metadata for efficient caching.
            apply_lm_head (bool, optional): Whether to apply the LM head. Defaults to True.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings.
            output_attentions (bool | None, optional): Whether to return attention weights.
            output_hidden_states (bool | None, optional): Whether to return all hidden states.
            output_router_logits (bool | None, optional): Whether to return MoE router logits.
            pixel_values (Array | None, optional): Image pixel values.
            pixel_values_videos (Array | None, optional): Video pixel values.
            image_grid_thw (Array | None, optional): Grid dimensions for images.
            video_grid_thw (Array | None, optional): Grid dimensions for videos.
            rope_deltas (Array | None, optional): Pre-computed rope position deltas.
            cache_position (Array | None, optional): Cache position (unused).
            **kwargs: Additional unused arguments.

        Returns:
            VLMCausalLMOutput: Model outputs including logits, hidden states, cache,
                rope deltas, and router logits from MoE layers.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else getattr(self.config, "output_router_logits", False)
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
            output_router_logits=output_router_logits,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
        )

        hidden_states = apply_logical_sharding(
            outputs.last_hidden_state,
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
            rope_deltas=getattr(outputs, "rope_deltas", None),
            router_logits=getattr(outputs, "router_logits", None) if output_router_logits else None,
        )


__all__ = [
    "Glm4vMoeConfig",
    "Glm4vMoeForConditionalGeneration",
    "Glm4vMoeModel",
    "Glm4vMoeModelOutputWithPast",
    "Glm4vMoeTextConfig",
    "Glm4vMoeTextModel",
    "Glm4vMoeVisionConfig",
    "Glm4vMoeVisionModel",
]
