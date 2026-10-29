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

"""Host-side per-group loss weighting for distillation.

The pooled distillation loss is the supervised-token mean, so each data group ``g`` weighs ``N_g / N``
(its share of supervised tokens). With target weights ``w_g`` every supervised token of group ``g`` is
scaled by ``s_g = w_g * N / N_g`` on the full step batch; every loss path reduces as
``sum(s * kl) / sum(s)`` and gradient accumulation recombines microbatches by ``sum(s)``, so the step
optimises ``sum_g w_g * mean_g``. The scales ride in a separate fp32 ``loss_token_weights`` leaf so
``completion_mask`` stays binary for every other consumer.
"""

from __future__ import annotations

import collections.abc
import typing as tp

import numpy as np

LOSS_GROUP_IDS_KEY = "loss_group_ids"
LOSS_TOKEN_WEIGHTS_KEY = "loss_token_weights"
LOSS_GROUP_BATCH_KEYS = (LOSS_GROUP_IDS_KEY, LOSS_TOKEN_WEIGHTS_KEY)


def loss_group_ids_from_dataset_ids(
    dataset_ids: np.ndarray,
    *,
    num_groups: int,
    dataset_map: collections.abc.Sequence[int] | None,
) -> np.ndarray:
    """Map per-token dataset ids to loss-group ids (``-1`` = no group).

    Args:
        dataset_ids: ``[B, S]`` integer dataset ids (``-1`` on padding).
        num_groups: Number of loss groups.
        dataset_map: Entry ``i`` is the group of dataset ``i`` (``-1`` = none); ``None`` maps dataset ``i``
            to group ``i`` when ``i < num_groups``.

    Returns:
        ``[B, S]`` int32 group ids.
    """
    ids = np.asarray(dataset_ids).astype(np.int64)
    if dataset_map is None:
        out = np.where((ids >= 0) & (ids < num_groups), ids, -1)
    else:
        lut = np.asarray(list(dataset_map), dtype=np.int64)
        inside = (ids >= 0) & (ids < lut.shape[0])
        out = np.where(inside, lut[np.clip(ids, 0, max(lut.shape[0] - 1, 0))], -1)
    return out.astype(np.int32)


def loss_supervision_mask(batch: collections.abc.Mapping[str, tp.Any], *, shift_tokens: bool) -> np.ndarray:
    """Host mirror of the distillation loss mask on the unshifted ``[B, S]`` label grid (float64).

    ``completion_mask`` > ``attention_mask`` > all-ones, AND ``labels != -100``; under ``shift_tokens``
    column 0 is zeroed because token 0 is never a prediction target.
    """
    ids = batch.get("input_ids", batch.get("inputs_embeds"))
    mask = batch.get("completion_mask", batch.get("attention_mask"))
    if mask is None:
        mask = np.ones(tuple(np.shape(ids)[:2]), np.float64)
    mask = np.asarray(mask).astype(np.float64)
    labels = batch.get("labels")
    if labels is not None:
        mask = mask * (np.asarray(labels) != -100)
    if shift_tokens:
        mask = mask.copy()
        mask[:, 0] = 0.0
    return mask


def loss_group_token_weights(
    group_ids: np.ndarray,
    supervised: np.ndarray,
    group_weights: collections.abc.Sequence[float],
    *,
    max_multiplier: float | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Per-token fp32 scales that turn the pooled token mean into ``sum_g w_g * mean_g``.

    Groups absent from the batch drop out and the remaining weights renormalise to sum 1. ``max_multiplier``
    caps ``s_g`` at that multiple of the pooled weight. Scales are divided by their smallest present value
    (the loss is a ratio, so this is exact) and computed in float64 before the fp32 cast, so weights equal to
    the token shares give exactly ``1.0`` everywhere.

    Args:
        group_ids: ``[B, S]`` int group ids (``-1`` = none).
        supervised: ``[B, S]`` supervision mask (see :func:`loss_supervision_mask`).
        group_weights: Target weight per group.
        max_multiplier: Optional cap on ``w_g * N / N_g``.

    Returns:
        ``(token_weights, stats)``: ``[B, S]`` float32 scales (``0`` outside every group) and float64 ``[G]``
        vectors ``tokens`` (``N_g``), ``share`` (``N_g / N``) and ``weight`` (effective weight after
        renormalisation and capping).

    Raises:
        ValueError: When a supervised token has no group.
    """
    group_ids = np.asarray(group_ids).astype(np.int64)
    supervised = np.asarray(supervised).astype(np.float64)
    num_groups = len(group_weights)
    orphan = (supervised != 0) & (group_ids < 0)
    if orphan.any():
        raise ValueError(
            f"loss-group weighting: {int(orphan.sum())} supervised tokens have no loss group; extend "
            "`loss_group_dataset_map` or mask them out."
        )
    in_group = (group_ids >= 0) & (group_ids < num_groups)
    safe_ids = np.where(in_group, group_ids, 0)
    tokens = np.bincount(safe_ids[in_group], weights=supervised[in_group], minlength=num_groups)[:num_groups]
    total = float(tokens.sum())
    present = tokens > 0
    weights = np.asarray(group_weights, np.float64) * present
    if weights.sum() > 0:
        weights = weights / weights.sum()
    scale = np.where(present, weights * total / np.where(present, tokens, 1.0), 0.0)
    if max_multiplier is not None:
        scale = np.minimum(scale, float(max_multiplier))
    positive = present & (scale > 0)
    if positive.any():
        scale = scale / scale[positive].min()
    token_weights = np.where(in_group, scale[safe_ids], 0.0).astype(np.float32)
    effective = scale * tokens
    stats = {
        "tokens": tokens,
        "share": tokens / max(total, 1.0),
        "weight": effective / max(float(effective.sum()), 1e-30),
    }
    return token_weights, stats
