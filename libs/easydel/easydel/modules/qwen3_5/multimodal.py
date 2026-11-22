# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0
"""Shared Qwen3.5 dense and MoE multimodal embedding/mRoPE forward driver."""

from jax import numpy as jnp

from easydel.modules._base import (
    flatten_mrope_position_ids_for_text,
    mrope_position_ids_from_mm_token_types,
)


def qwen3_5_position_ids(
    model,
    *,
    input_ids=None,
    inputs_embeds=None,
    attention_mask=None,
    mask_info=None,
    position_ids=None,
    mm_token_type_ids=None,
    image_grid_thw=None,
    video_grid_thw=None,
):
    """Resolve the same text/multimodal rotary positions for the trunk and MTP."""
    rope_deltas = None
    if position_ids is None:
        if mm_token_type_ids is not None:
            position_ids, rope_deltas = mrope_position_ids_from_mm_token_types(
                input_ids=input_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                spatial_merge_size=model.config.vision_config.spatial_merge_size,
            )
        elif image_grid_thw is None and video_grid_thw is None and (mask_info is not None or input_ids is None):
            from ejkernel.types import MaskInfo

            mask_info = MaskInfo.dynamic_init(
                mask_info=mask_info, input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask
            )
            position_ids = mask_info.q_position_ids
        else:
            position_ids, rope_deltas = model.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
    return flatten_mrope_position_ids_for_text(model.config.text_config, position_ids), rope_deltas


def qwen3_5_multimodal_forward(
    model,
    *,
    output_class,
    input_ids=None,
    inputs_embeds=None,
    attention_mask=None,
    mask_info=None,
    position_ids=None,
    mode=None,
    past_key_values=None,
    cache_metadata=None,
    output_attentions=None,
    output_hidden_states=None,
    pixel_values=None,
    image_embeds=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    image_max_grid_size=None,
    video_max_grid_size=None,
    mm_token_type_ids=None,
    output_router_logits=None,
):
    """Merge pixels or precomputed image embeddings, then run the language trunk.

    ``output_class`` selects the dense/MoE output protocol. Remaining arguments
    follow the public Qwen3.5 model forwards. Pixel inputs take precedence over
    precomputed image embeddings, which still retain their grid for 3D mRoPE.
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds = model.compute_embedding(input_ids)

    video_embeds = None
    if pixel_values is not None:
        image_embeds_tuple, _ = model.get_image_features(pixel_values, image_grid_thw, image_max_grid_size)
        image_embeds = jnp.concatenate(image_embeds_tuple, axis=0).astype(inputs_embeds.dtype)
    if pixel_values_videos is not None:
        video_embeds_tuple, _ = model.get_video_features(pixel_values_videos, video_grid_thw, video_max_grid_size)
        video_embeds = jnp.concatenate(video_embeds_tuple, axis=0).astype(inputs_embeds.dtype)
    if image_embeds is not None or video_embeds is not None:
        inputs_embeds = model.compute_embedding(
            input_ids, inputs_embeds=inputs_embeds, image_embeds=image_embeds, video_embeds=video_embeds
        )

    position_ids, rope_deltas = qwen3_5_position_ids(
        model,
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        mask_info=mask_info,
        position_ids=position_ids,
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_grid_thw if (pixel_values is not None or image_embeds is not None) else None,
        video_grid_thw=video_grid_thw if pixel_values_videos is not None else None,
    )
    outputs = model.language_model(
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
    router_outputs = {}
    if hasattr(output_class, "router_logits"):
        router_outputs["router_logits"] = getattr(outputs, "router_logits", None)
    return output_class(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=rope_deltas,
        **router_outputs,
    )
