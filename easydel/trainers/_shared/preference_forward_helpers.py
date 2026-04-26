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

"""Small helpers shared by preference-style ``concatenated_forward`` implementations.

Each preference trainer (DPO, CPO, BCO, KTO, etc.) implements its own
``concatenated_forward`` because the per-trainer output shape and reduction
logic legitimately differs.  But the *boilerplate* around forward-pass
preparation -- gathering multimodal kwargs from the batch, applying paired
truncation along the sequence axis -- is identical and was copy-pasted
between trainers during the spectrax migration.  Centralising those tiny
chunks here removes ~60 lines of literal duplication without forcing an
artificial unified abstraction over trainers whose contracts differ.
"""

from __future__ import annotations

import typing as tp

import jax

_MULTIMODAL_BATCH_KEYS: tuple[str, ...] = ("pixel_values", "pixel_attention_mask", "image_sizes")


def gather_multimodal_kwargs(
    batch: tp.Mapping[str, tp.Any],
    *,
    aux_loss_enabled: bool = False,
) -> dict[str, jax.Array]:
    """Build the optional model kwargs threaded through preference forward passes.

    Picks up any vision-tower inputs that happen to be present in *batch*
    (``pixel_values``, ``pixel_attention_mask``, ``image_sizes``) and adds
    ``output_router_logits`` when MoE auxiliary loss is requested.  Keys that
    are absent are silently skipped so this works on text-only batches.
    """
    model_kwargs: dict[str, jax.Array] = {}
    if aux_loss_enabled:
        model_kwargs["output_router_logits"] = True
    for key in _MULTIMODAL_BATCH_KEYS:
        if key in batch:
            model_kwargs[key] = batch[key]
    return model_kwargs


def apply_paired_truncation(
    *arrays: jax.Array,
    max_length: int | None,
    truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
) -> tuple[jax.Array, ...]:
    """Truncate each input array along ``axis=1`` using a shared mode.

    Used by the decoder-only branch of ``concatenated_forward`` to clip
    ``input_ids`` / ``attention_mask`` / ``loss_mask`` (or any number of
    parallel sequence-aligned tensors) to *max_length*.  When *max_length*
    is ``None`` the inputs are returned unchanged so callers can apply this
    unconditionally.
    """
    if max_length is None:
        return arrays
    if truncation_mode == "keep_end":
        return tuple(arr[:, -max_length:] for arr in arrays)
    if truncation_mode == "keep_start":
        return tuple(arr[:, :max_length] for arr in arrays)
    raise ValueError(
        f"Unknown truncation mode: {truncation_mode!r}. Expected 'keep_end' or 'keep_start'."
    )


__all__ = [
    "apply_paired_truncation",
    "gather_multimodal_kwargs",
]
