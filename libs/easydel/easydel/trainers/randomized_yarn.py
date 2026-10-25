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

"""Randomized YaRN: randomized positional encoding with a length curriculum.

Implements the position-sampling half of *Randomized YaRN Improves Length Generalization for
Long-Context Reasoning* (Mehta, Yin & Durrett, arXiv:2606.23687):

1. **Randomized positional encoding.** A document of ``L`` tokens gets ``L`` positions drawn
   uniformly without replacement from ``[0, L_t)`` and sorted, so token order survives while the
   absolute indices span a much larger range than ``L``.
2. **Length curriculum.** ``L_t`` follows a non-decreasing schedule over training.

YaRN scaling itself stays in the model's rope-scaling config; this module only rewrites
``position_ids``. Like :mod:`easydel.trainers.pose` it is a host-side, per-microbatch transform
that touches nothing else, and it shares PoSE's document segmentation.

``run_length=r`` keeps positions contiguous inside runs of ``r`` tokens while scattering the runs
over ``[0, L_t)``; ``r=1`` is the published method.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from itertools import pairwise

import numpy as np

from easydel.trainers.pose import (
    IGNORE_INDEX,
    document_segment_ranges,
    extract_position_batch,
    position_axes,
    valid_sequence_end,
    validate_rope_table_covers,
)

__all__ = [
    "RandomizedYaRNConfig",
    "apply_randomized_yarn_to_batch",
    "apply_randomized_yarn_to_positions",
    "current_l_max",
    "randomized_yarn_rng",
    "sample_monotone_positions",
    "validate_randomized_yarn_target",
]

_RNG_STREAM = 0x52594E  # separates this augmentation's random stream from any other seeded by (seed, step)


@dataclass
class RandomizedYaRNConfig:
    """Configuration for Randomized YaRN position augmentation.

    Attributes:
        enabled: Master switch; when ``False`` every entry point is a no-op.
        l_max_schedule: Non-decreasing length curriculum ``[L_1, ..., L_T]``; each value is the exclusive
            upper bound on positions sampled during that phase.
        curriculum_total_steps: Steps the phases divide evenly; ``0`` pins the last (largest) value.
        run_length: Contiguous run length ``r`` (``1`` reproduces the paper).
        p_apply: Per-document probability of applying the augmentation.
        target_max_pos: Hard ceiling on emitted positions, checked against the model's RoPE table; must be
            ``> 1`` when enabled and ``>= max(l_max_schedule)``.
        only_buckets: Optional training-bucket indices this applies to (``None`` = every step).
    """

    enabled: bool = False
    l_max_schedule: list[int] = field(default_factory=list)
    curriculum_total_steps: int = 0
    run_length: int = 1
    p_apply: float = 1.0
    target_max_pos: int = 0
    only_buckets: list[int] | None = None

    def __post_init__(self) -> None:
        self.run_length = int(self.run_length)
        self.curriculum_total_steps = int(self.curriculum_total_steps)
        self.target_max_pos = int(self.target_max_pos)
        self.p_apply = float(self.p_apply)
        self.l_max_schedule = [int(v) for v in (self.l_max_schedule or [])]
        if self.run_length < 1:
            raise ValueError(f"randomized_yarn.run_length must be >= 1, got {self.run_length}.")
        if not 0.0 <= self.p_apply <= 1.0:
            raise ValueError(f"randomized_yarn.p_apply must be in [0, 1], got {self.p_apply}.")
        if self.curriculum_total_steps < 0:
            raise ValueError(f"randomized_yarn.curriculum_total_steps must be >= 0, got {self.curriculum_total_steps}.")
        if self.only_buckets is not None:
            buckets = list(self.only_buckets)
            if not buckets:
                raise ValueError("randomized_yarn.only_buckets must be None or a non-empty list, got [].")
            if any(not isinstance(b, int) or isinstance(b, bool) or b < 0 for b in buckets):
                raise ValueError(f"randomized_yarn.only_buckets entries must be non-negative ints, got {buckets}.")
            self.only_buckets = buckets
        if not self.enabled:
            return
        if not self.l_max_schedule:
            raise ValueError("randomized_yarn.enabled requires a non-empty l_max_schedule.")
        if any(v <= 1 for v in self.l_max_schedule):
            raise ValueError(f"randomized_yarn.l_max_schedule entries must be > 1, got {self.l_max_schedule}.")
        if any(b < a for a, b in pairwise(self.l_max_schedule)):
            raise ValueError(f"randomized_yarn.l_max_schedule must be non-decreasing, got {self.l_max_schedule}.")
        if self.target_max_pos <= 1:
            raise ValueError(f"randomized_yarn.target_max_pos must be > 1 when enabled, got {self.target_max_pos}.")
        if max(self.l_max_schedule) > self.target_max_pos:
            raise ValueError(
                "randomized_yarn.target_max_pos must be >= every l_max_schedule entry, got "
                f"target_max_pos={self.target_max_pos}, max(l_max_schedule)={max(self.l_max_schedule)}."
            )

    def applies_to_bucket(self, bucket_index: int | None) -> bool:
        """Whether this should run on a step routed to ``bucket_index`` (same rule as PoSE)."""
        if self.only_buckets is None:
            return True
        return bucket_index is not None and bucket_index in self.only_buckets

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for TrainingArguments persistence."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object] | RandomizedYaRNConfig) -> RandomizedYaRNConfig:
        """Build a config from a serialized dictionary."""
        if isinstance(data, cls):
            return data
        return cls(**data)


def current_l_max(config: RandomizedYaRNConfig, step: int) -> int:
    """Exclusive position ceiling ``L_t`` at ``step`` under the length curriculum.

    The ``T`` schedule entries split ``curriculum_total_steps`` into ``T`` even phases; steps at or past
    the total stay in the last phase. ``curriculum_total_steps == 0`` uses the last entry throughout.
    """
    if not config.enabled or not config.l_max_schedule:
        return 0
    phases = len(config.l_max_schedule)
    if config.curriculum_total_steps <= 0 or phases == 1:
        return config.l_max_schedule[-1]
    phase = (max(int(step), 0) * phases) // config.curriculum_total_steps
    return config.l_max_schedule[min(phase, phases - 1)]


def validate_randomized_yarn_target(config: RandomizedYaRNConfig, *, model_max_freq_pos: int | None, who: str) -> None:
    """Fail loudly if ``target_max_pos`` exceeds the model's RoPE frequency-table length."""
    if not config.enabled:
        return
    validate_rope_table_covers(
        config.target_max_pos, model_max_freq_pos=model_max_freq_pos, who=who, feature="randomized_yarn"
    )


def randomized_yarn_rng(seed: int, step: int) -> np.random.Generator:
    """Generator for one microbatch, keyed on ``(seed, step)`` so distinct pairs never share a stream."""
    return np.random.default_rng(np.random.SeedSequence([int(seed) & 0xFFFFFFFF, int(step), _RNG_STREAM]))


def sample_monotone_positions(rng: np.random.Generator, *, n: int, l_max: int, run_length: int = 1) -> np.ndarray:
    """Draw ``n`` strictly increasing positions in ``[0, l_max)``, contiguous within runs of ``run_length``.

    ``run_length == 1`` is a uniformly random size-``n`` subset, sorted. For ``r > 1`` the range is tiled
    into ``l_max // r`` slots of width ``r``; ``ceil(n / r)`` slots are drawn without replacement, sorted
    and expanded into runs (the last run is trimmed).

    Raises:
        ValueError: When ``n`` positions do not fit in ``l_max`` at this run length.
    """
    n, l_max, run_length = int(n), int(l_max), int(run_length)
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    if run_length < 1:
        raise ValueError(f"run_length must be >= 1, got {run_length}.")
    n_runs = math.ceil(n / run_length)
    n_slots = l_max // run_length
    if n_runs > n_slots:
        raise ValueError(f"cannot place {n_runs} runs of {run_length} in [0, {l_max}) ({n_slots} slots).")
    slots = np.sort(rng.choice(n_slots, size=n_runs, replace=False)).astype(np.int64)
    out = (slots[:, None] * run_length + np.arange(run_length, dtype=np.int64)[None, :]).reshape(-1)
    return out[:n]


def apply_randomized_yarn_to_positions(
    positions: np.ndarray,
    *,
    config: RandomizedYaRNConfig,
    step: int,
    rng: np.random.Generator,
    tokens: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    pad_id: int | None = None,
    ignore_index: int = IGNORE_INDEX,
    seq_lens: np.ndarray | None = None,
    seq_lens_num: np.ndarray | None = None,
    cu_seqlens: np.ndarray | None = None,
    cu_seqlens_num: np.ndarray | None = None,
    image_token_id: int | None = None,
) -> None:
    """Apply Randomized YaRN in place to a writable rank-2 ``[B, S]`` position array.

    Per row: strip the padded tail, split into documents, then per document draw Bernoulli(``p_apply``)
    and, when it fires, replace the document's positions with ``base + sorted draw`` where ``base`` is the
    document's original first position and the draw lies in ``[0, min(L_t, target_max_pos - base))``.
    Documents that do not fit, or contain ``image_token_id``, are left untouched.
    """
    if not config.enabled or config.p_apply <= 0.0:
        return
    if positions.ndim != 2:
        raise ValueError(f"Randomized YaRN expects rank-2 [B, S] positions, got shape {tuple(positions.shape)}.")
    l_max = current_l_max(config, step)
    seq_len = positions.shape[1]
    if l_max <= 1 or seq_len < 2:
        return

    for sample_idx in range(positions.shape[0]):
        valid_end = valid_sequence_end(sample_idx, seq_len, tokens, labels, pad_id, ignore_index)
        if valid_end < 2:
            continue
        segment_ranges = document_segment_ranges(
            sample_idx, valid_end, positions, seq_lens, seq_lens_num, cu_seqlens, cu_seqlens_num
        )
        sample_pos = positions[sample_idx]
        seg_bases = {start: int(sample_pos[start]) for start, end in segment_ranges if end - start >= 2}
        for start, end in segment_ranges:
            segment_len = end - start
            if segment_len < 2:
                continue
            if (
                image_token_id is not None
                and tokens is not None
                and bool(np.any(tokens[sample_idx, start:end] == image_token_id))
            ):
                continue
            if rng.random() >= config.p_apply:
                continue
            base = seg_bases[start]
            span = min(l_max, config.target_max_pos - base)
            if math.ceil(segment_len / config.run_length) > span // config.run_length:
                continue
            drawn = sample_monotone_positions(rng, n=segment_len, l_max=span, run_length=config.run_length)
            sample_pos[start:end] = base + drawn.astype(sample_pos.dtype)


def apply_randomized_yarn_to_batch(
    batch: dict,
    *,
    config: RandomizedYaRNConfig,
    step: int,
    seed: int,
    image_token_id: int | None,
    pad_id: int | None,
    ignore_index: int = IGNORE_INDEX,
) -> dict:
    """Randomized YaRN on a collated batch; ``batch["position_ids"]`` is replaced by a fresh host array.

    Handles ``[B, S]`` and mRoPE ``[axes, B, S]`` layouts; every mRoPE axis gets the identical draw.

    Raises:
        ValueError: When enabled but the batch carries no ``position_ids``.
    """
    if not config.enabled or config.p_apply <= 0.0 or current_l_max(config, step) <= 1:
        return batch
    positions, inputs = extract_position_batch(batch, feature="randomized_yarn")
    for axis_positions in position_axes(positions):
        apply_randomized_yarn_to_positions(
            axis_positions,
            config=config,
            step=step,
            rng=randomized_yarn_rng(seed, step),
            pad_id=pad_id,
            ignore_index=ignore_index,
            image_token_id=image_token_id,
            **inputs,
        )
    batch["position_ids"] = positions
    return batch
