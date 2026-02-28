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

"""Qwen3.5-MoE text and multimodal model wrappers."""

import jax
import jax.numpy as jnp
from eformer import common_types
from flax import nnx as nn

from easydel.caching import (
    HybridCache,
    OperationsMetadata,
    RaggedPagesCache,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerMetadata,
)
from easydel.infra.factory import TaskType, register_module
from easydel.modules._base import BaseCausalLMModule, BaseVisionLanguageModule
from easydel.modules.qwen3_5.modeling_qwen3_5 import (
    _get_rope_index_from_mm_token_types,
    _maybe_flatten_position_ids_for_text,
)
from easydel.modules.qwen3_next.modeling_qwen3_next import Qwen3NextForCausalLM, Qwen3NextModel
from easydel.modules.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeModel,
    Qwen3VLMoeModelOutputWithPast,
    Qwen3VLMoeVisionTransformerPretrainedModel,
)
from easydel.modules.qwen3_vl_moe.qwen3_vl_moe_configuration import Qwen3VLMoeConfig, Qwen3VLMoeTextConfig

from .qwen3_5_moe_configuration import Qwen3_5MoeConfig, Qwen3_5MoeTextConfig


@register_module(TaskType.BASE_MODULE, config=Qwen3_5MoeTextConfig, model_type="qwen3_5_moe_text")
class Qwen3_5MoeTextModel(Qwen3NextModel):
    """Qwen3.5-MoE text-only base model (no LM head).

    Thin wrapper around :class:`Qwen3NextModel` registered with the
    ``qwen3_5_moe_text`` model type.
    """


@register_module(TaskType.CAUSAL_LM, config=Qwen3_5MoeTextConfig, model_type="qwen3_5_moe")
@register_module(TaskType.CAUSAL_LM, config=Qwen3_5MoeTextConfig, model_type="qwen3_5_moe_text")
class Qwen3_5MoeForCausalLM(Qwen3NextForCausalLM):
    """Qwen3.5-MoE text causal language model.

    Wraps :class:`Qwen3_5MoeTextModel` with a linear LM head for next-token
    prediction.  Supports MoE routing with auxiliary load-balancing loss.

    Args:
        config: Qwen3.5-MoE text configuration.
        dtype: Computation dtype.
        param_dtype: Parameter storage dtype.
        precision: JAX matmul precision.
        rngs: PRNG key container.
    """

    _model_type = "qwen3_5_moe"
    _config_class = Qwen3_5MoeTextConfig

    def __init__(
        self,
        config: Qwen3_5MoeTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        BaseCausalLMModule.__init__(
            self,
            config=config,
            base_model_class=Qwen3_5MoeTextModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
            router_aux_loss_coef=getattr(config, "router_aux_loss_coef", None),
        )


@register_module(TaskType.BASE_MODULE, config=Qwen3_5MoeConfig, model_type="qwen3_5_moe")
@register_module(TaskType.VISION_LM, config=Qwen3_5MoeConfig, model_type="qwen3_5_moe")
class Qwen3_5MoeModel(Qwen3VLMoeModel):
    """Qwen3.5-MoE multimodal (vision-language) base model.

    Combines a :class:`Qwen3VLMoeVisionTransformerPretrainedModel` vision encoder
    with a :class:`Qwen3_5MoeTextModel` MoE language backbone. Image and video
    pixels are encoded into continuous embeddings, fused into the token embedding
    stream, and processed by the language model with 3D mRoPE position ids.

    Args:
        config: Qwen3.5-MoE multimodal configuration.
        dtype: Computation dtype.
        param_dtype: Parameter storage dtype.
        precision: JAX matmul precision.
        rngs: PRNG key container.
    """

    def __init__(
        self,
        config: Qwen3_5MoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        bootstrap_text_config = Qwen3VLMoeTextConfig(
            vocab_size=config.text_config.vocab_size,
            hidden_size=config.text_config.hidden_size,
            intermediate_size=config.text_config.intermediate_size,
            num_hidden_layers=config.text_config.num_hidden_layers,
            num_attention_heads=config.text_config.num_attention_heads,
            num_key_value_heads=config.text_config.num_key_value_heads,
            head_dim=config.text_config.head_dim,
            hidden_act=config.text_config.hidden_act,
            max_position_embeddings=config.text_config.max_position_embeddings,
            initializer_range=config.text_config.initializer_range,
            rms_norm_eps=config.text_config.rms_norm_eps,
            use_cache=config.text_config.use_cache,
            tie_word_embeddings=getattr(config.text_config, "tie_word_embeddings", False),
            rope_theta=config.text_config.rope_theta,
            attention_bias=config.text_config.attention_bias,
            attention_dropout=config.text_config.attention_dropout,
            rope_scaling=getattr(config.text_config, "rope_scaling", None),
            decoder_sparse_step=getattr(config.text_config, "decoder_sparse_step", 1),
            moe_intermediate_size=getattr(config.text_config, "moe_intermediate_size", 512),
            num_experts_per_tok=getattr(config.text_config, "num_experts_per_tok", 4),
            num_experts=getattr(config.text_config, "num_experts", 8),
            norm_topk_prob=getattr(config.text_config, "norm_topk_prob", False),
            output_router_logits=getattr(config.text_config, "output_router_logits", False),
            router_aux_loss_coef=getattr(config.text_config, "router_aux_loss_coef", 0.001),
            mlp_only_layers=getattr(config.text_config, "mlp_only_layers", None),
            layer_types=getattr(config.text_config, "layer_types", None),
        )
        bootstrap_config = Qwen3VLMoeConfig(
            vision_config=(
                config.vision_config.to_dict()
                if hasattr(config.vision_config, "to_dict")
                else vars(config.vision_config)
            ),
            text_config=bootstrap_text_config.to_dict(),
            image_token_id=config.image_token_id,
            video_token_id=config.video_token_id,
            vision_start_token_id=config.vision_start_token_id,
            vision_end_token_id=config.vision_end_token_id,
            tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
        )
        super().__init__(
            config=bootstrap_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.config = config
        # Rebuild vision tower from the final Qwen3.5-MoE vision config so
        # deepstack settings and parameter names match HF checkpoints.
        self.visual = Qwen3VLMoeVisionTransformerPretrainedModel(
            config=config.vision_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.language_model = Qwen3_5MoeTextModel(
            config=config.text_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.rope_deltas = None

    def __call__(
        self,
        input_ids: jax.Array | None = None,
        inputs_embeds: jax.Array | None = None,
        attention_mask: jax.Array | None = None,
        mask_info: object | None = None,
        position_ids: jax.Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        visual_pos_masks: jax.Array | None = None,  # compatibility no-op
        deepstack_visual_embeds: list[jax.Array] | None = None,  # compatibility no-op
        pixel_values: jax.Array | None = None,
        pixel_values_videos: jax.Array | None = None,
        image_grid_thw: tuple | None = None,
        video_grid_thw: tuple | None = None,
        image_max_grid_size: int | None = None,
        video_max_grid_size: int | None = None,
        cache_position: jax.Array | None = None,  # compatibility no-op
        mm_token_type_ids: jax.Array | None = None,
    ) -> Qwen3VLMoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.compute_embedding(input_ids)

        image_embeds = None
        video_embeds = None
        if pixel_values is not None:
            image_embeds_tuple, _deepstack_image_embeds = self.get_image_features(
                pixel_values,
                image_grid_thw,
                image_max_grid_size,
            )
            image_embeds = jnp.concatenate(image_embeds_tuple, axis=0).astype(inputs_embeds.dtype)
        if pixel_values_videos is not None:
            video_embeds_tuple, _deepstack_video_embeds = self.get_video_features(
                pixel_values_videos,
                video_grid_thw,
                video_max_grid_size,
            )
            video_embeds = jnp.concatenate(video_embeds_tuple, axis=0).astype(inputs_embeds.dtype)
        if image_embeds is not None or video_embeds is not None:
            inputs_embeds = self.compute_embedding(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_embeds=image_embeds,
                video_embeds=video_embeds,
            )

        if position_ids is None:
            if mm_token_type_ids is not None:
                position_ids, rope_deltas = _get_rope_index_from_mm_token_types(
                    input_ids=input_ids,
                    mm_token_type_ids=mm_token_type_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask,
                    spatial_merge_size=self.config.vision_config.spatial_merge_size,
                )
            else:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw if pixel_values is not None else None,
                    video_grid_thw=video_grid_thw if pixel_values_videos is not None else None,
                    attention_mask=attention_mask,
                )
            self.rope_deltas = rope_deltas

        position_ids = _maybe_flatten_position_ids_for_text(self.config.text_config, position_ids)

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
        )

        return Qwen3VLMoeModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )


@register_module(TaskType.IMAGE_TEXT_TO_TEXT, config=Qwen3_5MoeConfig, model_type="qwen3_5_moe")
class Qwen3_5MoeForConditionalGeneration(BaseVisionLanguageModule[Qwen3_5MoeModel, Qwen3_5MoeConfig]):
    """Qwen3.5-MoE multimodal conditional generation model.

    End-to-end vision-language model that wraps :class:`Qwen3_5MoeModel` and
    adds a causal LM head for image/video-conditioned text generation.
    Supports both image and video inputs via the underlying vision encoder
    and uses MoE routing in the language backbone.

    Args:
        config: Qwen3.5-MoE multimodal configuration.
        dtype: Computation dtype.
        param_dtype: Parameter storage dtype.
        precision: JAX matmul precision.
        rngs: PRNG key container.
    """

    _task_type = TaskType.IMAGE_TEXT_TO_TEXT
    _model_type = "qwen3_5_moe"
    _config_class = Qwen3_5MoeConfig
    _auto_register = False
    _supports_video = True
    _uses_mrope = True

    _vision_tower_name = "visual"
    _projector_name = "merger"
    _language_model_name = "language_model"

    loss_type = "ForCausalLM"

    def __init__(
        self,
        config: Qwen3_5MoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(
            config=config,
            base_model_class=Qwen3_5MoeModel,
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
            router_aux_loss_coef=getattr(config.text_config, "router_aux_loss_coef", 0.001),
            tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
            lm_head_bias=False,
        )
        self.vocab_size = config.text_config.vocab_size

    def get_input_embeddings(self):
        """Get the input embedding layer."""
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        """Set the input embedding layer."""
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        """Set the language model decoder."""
        self.model.set_decoder(decoder)

    def get_decoder(self):
        """Get the language model decoder."""
        return self.model.get_decoder()

    @property
    def visual(self):
        """Property to access the vision transformer for backward compatibility."""
        return self.model.visual

    @property
    def language_model(self):
        """Property to access the language model for backward compatibility."""
        return self.model.language_model

    def get_video_features(
        self,
        pixel_values_videos: jax.Array,
        video_grid_thw: jax.Array | None = None,
        video_max_grid_size: int | None = None,
    ) -> tuple[tuple[jax.Array, ...], list[jax.Array]]:
        """Encode videos into continuous embeddings."""
        return self.model.get_video_features(pixel_values_videos, video_grid_thw, video_max_grid_size)

    def get_image_features(
        self,
        pixel_values: jax.Array,
        image_grid_thw: jax.Array | None = None,
        image_max_grid_size: int | None = None,
    ) -> tuple[tuple[jax.Array, ...], list[jax.Array]]:
        """Encode images into continuous embeddings."""
        return self.model.get_image_features(pixel_values, image_grid_thw, image_max_grid_size)

    def compute_embedding(self, input_ids, *args, **kwargs):
        """Compute embeddings with multimodal fusion."""
        return self.model.compute_embedding(input_ids, *args, **kwargs)


__all__ = [
    "Qwen3_5MoeConfig",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5MoeModel",
    "Qwen3_5MoeTextConfig",
    "Qwen3_5MoeTextModel",
]
