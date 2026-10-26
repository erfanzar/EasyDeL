# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0
"""Shared Qwen3.5 dense/MoE text and vision-language MTP training API."""

import inspect
import typing as tp

import jax
from ejkernel.types import MaskInfo
from jax import numpy as jnp
from spectrax import common_types

from easydel.infra.sequence_packing import packed_segment_ids_from_mask_info, token_attention_mask_from_mask_info


def _shift_left(value, amount, fill=0):
    amount = min(amount, value.shape[1])
    pad = jnp.full((value.shape[0], amount, *value.shape[2:]), fill, dtype=value.dtype)
    return jnp.concatenate([value[:, amount:], pad], axis=1)


def _mtp_shift_validity(labels, shift, attention_mask=None, mask_info=None):
    """Require every token from the source through the target to share a segment."""
    valid = token_attention_mask_from_mask_info(mask_info, labels.shape[1])
    if valid is None:
        valid = jnp.ones(labels.shape, dtype=jnp.bool_)
    if attention_mask is not None and attention_mask.ndim == 2:
        valid = valid & attention_mask.astype(jnp.bool_)
    segments = packed_segment_ids_from_mask_info(mask_info, labels.shape[1])
    same_run = valid
    for offset in range(1, min(shift, labels.shape[1]) + 1):
        same_run = same_run & _shift_left(valid, offset, False)
        if segments is not None:
            same_run = same_run & (segments == _shift_left(segments, offset, -1))
    return same_run


class Qwen3_5MTPMixin:
    """Teacher-forced MTP API using a model's shared embedding and LM head.

    Subclasses instantiate ``self.mtp`` without changing checkpoint paths.
    The embedding, rotary-table and coefficient hooks support text and VLM trunks.
    """

    def _mtp_embed(self, token_ids_i4):
        return self.model.get_embedding()(token_ids_i4)

    def _mtp_frequencies(self):
        return getattr(self.model, "frequencies", None)

    def _mtp_loss_coef(self) -> float:
        return float(getattr(self.config, "mtp_loss_coef", 0.0))

    def has_mtp(self) -> bool:
        """Return whether an MTP head is instantiated."""
        return self.mtp is not None

    def compute_mtp_outputs(
        self,
        outputs,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        mask_info=None,
        position_ids: jax.Array | None = None,
        mode=None,
        mtp_past_key_values=None,
        cache_metadata=None,
    ) -> tp.Any:
        """Fuse main hidden states with next-token embeddings in the MTP head.

        Args:
            outputs: Main forward result exposing ``last_hidden_state``.
            input_ids: Ground-truth token IDs, shaped ``(batch, sequence)``.
            attention_mask: Optional padding mask reused by MTP attention.
            mask_info: Optional structured causal mask.
            position_ids: Optional rotary position IDs.
            mode: Runtime mode, defaulting to training.
            mtp_past_key_values: MTP-local KV cache, separate from the trunk cache.
            cache_metadata: Metadata for the MTP cache.

        Returns:
            MTP hidden states and updated cache, or None without an MTP head.
            Project hidden states with ``compute_mtp_logits`` to predict t + 2.
        """
        if self.mtp is None:
            return None
        next_input_ids = _shift_left(input_ids, 1)
        next_token_embeds = self._mtp_embed(next_input_ids.astype("i4"))
        valid = _mtp_shift_validity(input_ids, 1, attention_mask, mask_info)
        next_token_embeds = jnp.where(valid[..., None], next_token_embeds, 0)
        return self.mtp(
            prev_hidden_states=outputs.last_hidden_state,
            next_token_embeds=next_token_embeds,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=mtp_past_key_values,
            cache_metadata=cache_metadata,
            frequencies=self._mtp_frequencies(),
            attention_mask=attention_mask,
        )

    def compute_mtp_logits(self, mtp_output) -> jax.Array:
        """Project MTP hidden states to (batch, sequence, vocabulary) logits."""
        hidden = mtp_output.last_hidden_state
        if hasattr(self, "prepare_lm_head_inputs"):
            hidden = self.prepare_lm_head_inputs(hidden)
        return self.compute_lm_logits(hidden)

    def compute_mtp_chain(
        self, outputs, input_ids, n_steps: int, attention_mask=None, mask_info=None, position_ids=None
    ):
        """Return teacher-forced (steps, batch, sequence, vocabulary) logits.

        Step k fuses the previous hidden states with tokens shifted by k and
        predicts t + k + 1. Returns None without a head or when n_steps < 1.
        """
        if self.mtp is None or n_steps < 1:
            return None
        frequencies = self._mtp_frequencies()
        prev_hidden = outputs.last_hidden_state
        step_logits = []
        for k in range(1, n_steps + 1):
            shifted = _shift_left(input_ids, k)
            embeds = self._mtp_embed(shifted.astype("i4"))
            valid = _mtp_shift_validity(input_ids, k, attention_mask, mask_info)
            mtp_out = self.mtp(
                prev_hidden_states=prev_hidden,
                next_token_embeds=jnp.where(valid[..., None], embeds, 0),
                frequencies=frequencies,
                attention_mask=attention_mask,
                mask_info=mask_info,
                position_ids=position_ids,
            )
            prev_hidden = mtp_out.last_hidden_state
            step_logits.append(self.compute_mtp_logits(mtp_out))
        return jnp.stack(step_logits, axis=0)

    def compute_mtp_loss(self, mtp_logits, labels, attention_mask=None, ignore_index: int = -100, mask_info=None):
        """Return unscaled fp32 skip-one CE, excluding ignored/padded targets.

        ``mtp_logits[:, t]`` predicts ``labels[:, t + 2]``. The trailing two
        positions have no target. Padding and packed ``mask_info`` exclude
        shifts that cross a masked token or an example boundary. VLM callers
        can also mask visual labels with ``ignore_index``. Returns zero when
        no valid targets remain.
        """
        shifted_labels = _shift_left(labels, 2, ignore_index)
        valid_targets = _mtp_shift_validity(labels, 2, attention_mask, mask_info)
        shifted_labels = jnp.where(valid_targets, shifted_labels, ignore_index)
        log_probs = jax.nn.log_softmax(mtp_logits.astype(jnp.float32), axis=-1)
        nll = -jnp.take_along_axis(log_probs, jnp.maximum(shifted_labels, 0)[..., None], axis=-1).squeeze(-1)
        valid = (shifted_labels != ignore_index).astype(jnp.float32)
        return jnp.sum(nll * valid) / jnp.maximum(jnp.sum(valid), 1.0)

    def _maybe_add_mtp_aux_loss(self, outputs, input_ids, attention_mask, mode, mask_info=None, position_ids=None):
        """Expose scaled MTP loss and logits and add it to training aux_loss."""
        if self.mtp is None or input_ids is None:
            return outputs
        coef = self._mtp_loss_coef()
        if coef <= 0.0 or mode in (common_types.MODE_DECODE, common_types.MODE_PREFILL, common_types.MODE_INSERT):
            return outputs
        if getattr(outputs, "last_hidden_state", None) is None:
            return outputs
        mask_info = MaskInfo.dynamic_init(mask_info=mask_info, input_ids=input_ids, attention_mask=attention_mask)
        if position_ids is None:
            position_ids = mask_info.q_position_ids
        mtp_output = self.compute_mtp_outputs(
            outputs,
            input_ids=input_ids,
            attention_mask=attention_mask,
            mode=mode,
            mask_info=mask_info,
            position_ids=position_ids,
        )
        if mtp_output is None:
            return outputs
        mtp_logits = self.compute_mtp_logits(mtp_output)
        mtp_loss = self.compute_mtp_loss(mtp_logits, input_ids, attention_mask, mask_info=mask_info) * coef
        existing = getattr(outputs, "aux_loss", None)
        new_aux = mtp_loss if existing is None else existing + mtp_loss
        return outputs.replace(aux_loss=new_aux, mtp_logits=mtp_logits, mtp_loss=mtp_loss)


class Qwen3_5VisionMTPMixin(Qwen3_5MTPMixin):
    """MTP hooks for dense and MoE multimodal trunks."""

    def _vision_forward_with_mtp(self, forward, *args, **kwargs):
        """Reuse resolved multimodal/packed positions in automatic MTP training."""
        from easydel.modules._base import BaseVisionLanguageModule

        from .multimodal import qwen3_5_position_ids

        outputs = forward(*args, **kwargs)
        if self.mtp is None:
            return outputs
        bound = inspect.signature(BaseVisionLanguageModule.forward).bind(self, *args, **kwargs).arguments
        extra = bound.pop("kwargs", {})
        bound.update(extra)
        input_ids = bound.get("input_ids")
        mode = bound.get("mode")
        if mode is None and (bound.get("past_key_values") is not None or bound.get("cache_metadata") is not None):
            inputs = input_ids if input_ids is not None else bound.get("inputs_embeds")
            mode = common_types.MODE_DECODE if inputs is not None and inputs.shape[1] == 1 else common_types.MODE_PREFILL
        if (
            input_ids is None
            or self._mtp_loss_coef() <= 0
            or mode in (common_types.MODE_DECODE, common_types.MODE_PREFILL, common_types.MODE_INSERT)
        ):
            return outputs
        position_ids, _ = qwen3_5_position_ids(
            self.model,
            input_ids=input_ids,
            inputs_embeds=bound.get("inputs_embeds"),
            attention_mask=bound.get("attention_mask"),
            mask_info=bound.get("mask_info"),
            position_ids=bound.get("position_ids"),
            mm_token_type_ids=bound.get("mm_token_type_ids"),
            image_grid_thw=bound.get("image_grid_thw")
            if (bound.get("pixel_values") is not None or bound.get("image_embeds") is not None)
            else None,
            video_grid_thw=bound.get("video_grid_thw") if bound.get("pixel_values_videos") is not None else None,
        )
        return self._maybe_add_mtp_aux_loss(
            outputs,
            input_ids,
            bound.get("attention_mask"),
            mode,
            mask_info=bound.get("mask_info"),
            position_ids=position_ids,
        )

    def _mtp_embed(self, token_ids_i4):
        return self.model.get_input_embeddings()(token_ids_i4)

    def _mtp_frequencies(self):
        return getattr(self.model.language_model, "frequencies", None)

    def _mtp_loss_coef(self) -> float:
        return float(getattr(self.config.text_config, "mtp_loss_coef", 0.0))
