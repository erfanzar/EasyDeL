# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared preference-pair batch preparation, independent of any trainer."""

from jax import numpy as jnp
from jaxtyping import Array


def concatenated_inputs(
    batch: dict[str, list | Array],
    padding_value: int,
) -> dict[str, Array]:
    """Concatenate chosen/rejected examples into a single batch for a fused forward pass.

    Used by preference ``concatenated_forward`` implementations to halve the
    number of model calls per pair: the prompt is duplicated and chosen and
    rejected completions are stacked along the batch axis. Completion ids and
    attention masks are padded to the maximum completion length. Multimodal
    side-inputs are likewise duplicated along the batch axis.

    Args:
        batch: Preference batch with ``prompt_input_ids``,
            ``prompt_attention_mask``, ``chosen_input_ids``,
            ``chosen_attention_mask``, ``rejected_input_ids``, and
            ``rejected_attention_mask``. Optionally includes ``pixel_values``,
            ``pixel_attention_mask``, and ``image_sizes`` for VLM training,
            and ``chosen_loss_mask`` / ``rejected_loss_mask`` (same shape as
            the completion ids) selecting which attended completion tokens
            enter the log-prob sums. When either loss mask is present the
            output carries ``completion_loss_mask`` (a missing side defaults
            to all ones); otherwise the output is unchanged.
        padding_value: Padding token id used to right-pad completions.

    Returns:
        Prompt ids/masks, completion ids/masks, and optional image inputs with
        chosen examples first and rejected examples second (``2 * batch`` rows).
    """
    # Keep the shared package independent of the heavyweight collator module
    # until batch preparation is actually requested.
    from ..utils import pad_to_length

    output = {}
    output["prompt_input_ids"] = jnp.concatenate(
        [batch["prompt_input_ids"], batch["prompt_input_ids"]],
        axis=0,
    )
    output["prompt_attention_mask"] = jnp.concatenate(
        [batch["prompt_attention_mask"], batch["prompt_attention_mask"]],
        axis=0,
    )
    if "pixel_values" in batch:
        output["pixel_values"] = jnp.concatenate(
            [batch["pixel_values"], batch["pixel_values"]],
            axis=0,
        )
    if "pixel_attention_mask" in batch:
        output["pixel_attention_mask"] = jnp.concatenate(
            [batch["pixel_attention_mask"], batch["pixel_attention_mask"]],
            axis=0,
        )
    if "image_sizes" in batch:
        output["image_sizes"] = jnp.concatenate(
            [batch["image_sizes"], batch["image_sizes"]],
            axis=0,
        )

    max_completion_length = max(
        batch["chosen_input_ids"].shape[1],
        batch["rejected_input_ids"].shape[1],
    )
    output["completion_input_ids"] = jnp.concatenate(
        (
            pad_to_length(
                batch["chosen_input_ids"],
                max_completion_length,
                pad_value=padding_value,
            ),
            pad_to_length(
                batch["rejected_input_ids"],
                max_completion_length,
                pad_value=padding_value,
            ),
        ),
    )
    output["completion_attention_mask"] = jnp.concatenate(
        (
            pad_to_length(
                batch["chosen_attention_mask"],
                max_completion_length,
                pad_value=0,
            ),
            pad_to_length(
                batch["rejected_attention_mask"],
                max_completion_length,
                pad_value=0,
            ),
        ),
    )
    chosen_loss_mask = batch.get("chosen_loss_mask")
    rejected_loss_mask = batch.get("rejected_loss_mask")
    if chosen_loss_mask is not None or rejected_loss_mask is not None:
        # A side without an explicit loss mask scores every attended completion token.
        if chosen_loss_mask is None:
            chosen_loss_mask = jnp.ones_like(batch["chosen_attention_mask"])
        if rejected_loss_mask is None:
            rejected_loss_mask = jnp.ones_like(batch["rejected_attention_mask"])
        for name, mask, ids in (
            ("chosen_loss_mask", chosen_loss_mask, batch["chosen_input_ids"]),
            ("rejected_loss_mask", rejected_loss_mask, batch["rejected_input_ids"]),
        ):
            if tuple(mask.shape) != tuple(ids.shape):
                raise ValueError(
                    f"`{name}` must have the same shape as its completion ids {tuple(ids.shape)}, "
                    f"got {tuple(mask.shape)}."
                )
        output["completion_loss_mask"] = jnp.concatenate(
            (
                pad_to_length(jnp.asarray(chosen_loss_mask), max_completion_length, pad_value=0),
                pad_to_length(jnp.asarray(rejected_loss_mask), max_completion_length, pad_value=0),
            ),
        )
    return output
