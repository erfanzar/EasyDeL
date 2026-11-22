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

"""Input guards for fixed-rate KV compression, whose groups follow physical token offsets."""

from __future__ import annotations

import typing as tp

import jax
from jax import numpy as jnp
from jaxtyping import Array

__all__ = ("raise_if_runtime_true", "reject_packed_or_padded_batch")


def raise_if_runtime_true(condition: Array | bool, message: str) -> None:
    """Raise ``ValueError(message)`` eagerly, or through a JIT-safe host callback, when ``condition`` holds."""

    def _check(value):
        if bool(value):
            raise ValueError(message)

    if isinstance(condition, jax.core.Tracer):
        # No ordering dependency on other effects; ordered callbacks cannot be lowered across multiple devices (TP).
        jax.debug.callback(_check, condition, ordered=False)
    else:
        _check(condition)


def reject_packed_or_padded_batch(
    mask_info: tp.Any | None,
    attention_mask: Array | None = None,
    *,
    family: str = "Compressed-window attention",
) -> None:
    """Reject packed documents or padding that fixed-rate compressors cannot represent.

    All-valid masks pass. A ``MaskInfo`` carrying segment ids is rejected when any position is padding or a row holds
    more than one segment; a raw mask (with no segments) or an explicit ``attention_mask`` is rejected when any position
    is masked.

    Args:
        mask_info: ``ejkernel.types.MaskInfo`` (or any object exposing ``_q_segment_ids`` / ``_attention_mask``), or
            ``None``.
        attention_mask: Optional ``[batch, seq]`` mask.
        family: Model name used in the error message.

    Raises:
        ValueError: When the batch is packed or padded (eagerly, or from the device callback under ``jit``).
    """
    message = (
        f"{family} packed-document training or padding-mask prefill is not yet supported: "
        "fixed-rate KV compression follows physical offsets. Use unpacked, unpadded batches."
    )
    conditions = []
    if mask_info is not None:
        segments = getattr(mask_info, "_q_segment_ids", None)
        if segments is not None:
            valid = segments >= 0
            safe = jnp.where(valid, segments, 0)
            has_padding = jnp.any(~valid)
            has_multiple = jnp.max(safe, axis=-1) != jnp.min(jnp.where(valid, safe, jnp.iinfo(safe.dtype).max), axis=-1)
            has_multiple = jnp.any(has_multiple & jnp.any(valid, axis=-1))
            conditions.append(has_padding | has_multiple)
        raw_mask = getattr(mask_info, "_attention_mask", None)
        if raw_mask is not None and segments is None:
            conditions.append(jnp.any(~raw_mask.astype(bool)))
    if attention_mask is not None:
        conditions.append(jnp.any(~attention_mask.astype(bool)))
    if conditions:
        condition = conditions[0]
        for item in conditions[1:]:
            condition = condition | item
        raise_if_runtime_true(condition, message)
