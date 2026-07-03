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

"""PoSE (Positional Skip-wisE) training — a pure ``position_ids`` augmentation.

PoSE (Zhu et al. 2023, arXiv:2309.10400) extends a model's usable context without
paying long-sequence compute: it keeps a short physical training window but manipulates
the position indices so RoPE sees relative distances up to a long-context target. Each
document is split into two chunks and a large positional *skip* (gap) is inserted between
them; the largest emitted position stays ``< target_max_pos``. It touches ONLY
``position_ids`` — tokens, labels, loss, and the attention mask are unchanged (the skip
changes RoPE rotation angles, not attention connectivity).

This is the data-side core: a host-side, per-microbatch transform applied to a rank-2
``[B, S]`` (1-D positions) array right before the forward pass. It is deliberately NOT a
jit'd op — the per-document segmentation, Bernoulli gate, split, and gap are data-dependent
control flow. RoPE in EasyDeL gathers cos/sin at the passed ``position_ids``
(``frequencies[positions]``), so this transform takes effect; the model's RoPE frequency
table must cover ``target_max_pos`` (see :func:`validate_pose_target_max_pos`).

RNG: a threaded :class:`numpy.random.Generator` consumed in a fixed per-document order —
(1) Bernoulli gate, (2) split point, (3) gap — so a given ``(seed, step)`` is reproducible.
The caller creates one generator per microbatch (e.g. ``np.random.default_rng(seed + step)``).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import pairwise

import numpy as np

IGNORE_INDEX = -100


@dataclass
class PoSEConfig:
    """Configuration for PoSE position augmentation.

    Attributes:
        enabled: Master switch; when ``False`` the transform is a no-op.
        p_pose: Per-document probability of applying a skip (the value at step 0 when
            ``warmup_steps`` is used).
        max_p_pose: Probability reached at the end of warmup; defaults to ``p_pose``.
        warmup_steps: Linear ramp length for the probability (``0`` ⇒ constant ``p_pose``).
        target_max_pos: Long-context target; the largest position PoSE emits is
            ``< target_max_pos``. Must be ``> 1`` when enabled and ``<=`` the model's RoPE
            frequency-table length (validated separately via
            :func:`validate_pose_target_max_pos`).
        min_chunk_frac: Lower bound of the split point as a fraction of document length
            (``min_chunk = ceil(L * min_chunk_frac)``).
        max_chunk_frac: Upper bound of the split point (``max_chunk = floor(L * max_chunk_frac)``).
        bias_large_gap: When ``True`` warp the uniform gap draw toward large gaps via
            ``g = 1 - (1 - u)**2``; otherwise draw the gap uniformly.
        only_buckets: Optional list of training-bucket indices (see
            ``TrainingArguments.buckets``) PoSE applies to; steps routed to any other
            bucket are left untouched. ``None`` (default) applies PoSE on every step.
            Requires the trainer to have training buckets configured (validated there).
    """

    enabled: bool = False
    p_pose: float = 0.0
    max_p_pose: float | None = None
    warmup_steps: int = 0
    target_max_pos: int = 0
    min_chunk_frac: float = 0.25
    max_chunk_frac: float = 0.75
    bias_large_gap: bool = True
    only_buckets: list[int] | None = None

    def __post_init__(self) -> None:
        if self.max_p_pose is None:
            self.max_p_pose = self.p_pose
        if not 0.0 <= self.p_pose <= 1.0:
            raise ValueError(f"pose.p_pose must be in [0, 1], got {self.p_pose}.")
        if not 0.0 <= self.max_p_pose <= 1.0:
            raise ValueError(f"pose.max_p_pose must be in [0, 1], got {self.max_p_pose}.")
        if self.warmup_steps < 0:
            raise ValueError(f"pose.warmup_steps must be >= 0, got {self.warmup_steps}.")
        if not 0.0 <= self.min_chunk_frac <= 1.0:
            raise ValueError(f"pose.min_chunk_frac must be in [0, 1], got {self.min_chunk_frac}.")
        if not 0.0 <= self.max_chunk_frac <= 1.0:
            raise ValueError(f"pose.max_chunk_frac must be in [0, 1], got {self.max_chunk_frac}.")
        if self.min_chunk_frac > self.max_chunk_frac:
            raise ValueError(
                f"pose.min_chunk_frac must be <= pose.max_chunk_frac, got {self.min_chunk_frac} > {self.max_chunk_frac}."
            )
        if self.enabled and self.target_max_pos <= 1:
            raise ValueError(f"pose.target_max_pos must be > 1 when pose.enabled=true, got {self.target_max_pos}.")
        if self.only_buckets is not None:
            buckets = list(self.only_buckets)
            if not buckets:
                raise ValueError("pose.only_buckets must be None or a non-empty list of bucket indices, got [].")
            if any(not isinstance(b, int) or isinstance(b, bool) or b < 0 for b in buckets):
                raise ValueError(f"pose.only_buckets entries must be non-negative ints, got {self.only_buckets}.")
            self.only_buckets = buckets

    def applies_to_bucket(self, bucket_index: int | None) -> bool:
        """Whether PoSE should run on a step routed to ``bucket_index``.

        ``None`` ``only_buckets`` matches every step. When restricted, a step matches
        only if the trainer reports a bucket index in the list (``bucket_index=None``
        — no buckets configured — never matches a restricted config).
        """
        if self.only_buckets is None:
            return True
        return bucket_index is not None and bucket_index in self.only_buckets

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for TrainingArguments persistence."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object] | "PoSEConfig") -> "PoSEConfig":
        """Build a PoSE config from a serialized dictionary."""
        if isinstance(data, cls):
            return data
        return cls(**data)


def current_pose_probability(config: PoSEConfig, step: int) -> float:
    """Return the PoSE probability at ``step`` under the linear warmup schedule.

    ``0`` when disabled; constant ``p_pose`` when ``warmup_steps <= 0``; otherwise a linear
    interpolation from ``p_pose`` (step 0) to ``max_p_pose`` (step ``>= warmup_steps``).
    """
    if not config.enabled:
        return 0.0
    if config.warmup_steps <= 0:
        return config.p_pose
    assert config.max_p_pose is not None  # set in __post_init__
    progress = min(max(step, 0), config.warmup_steps) / float(config.warmup_steps)
    return config.p_pose + progress * (config.max_p_pose - config.p_pose)


def validate_pose_target_max_pos(config: PoSEConfig, *, model_max_freq_pos: int | None, who: str) -> None:
    """Fail loudly if ``target_max_pos`` exceeds the model's RoPE frequency-table length.

    RoPE cos/sin are a fixed precomputed table gathered by ``position_ids``; a position
    ``>= len(table)`` indexes out of bounds (silently wraps under XLA). The table length is
    ``config.granted_freq_max_position_embedding`` (``freq_max_position_embeddings`` or
    ``max_position_embeddings``) — pass that as ``model_max_freq_pos``. When PoSE is enabled
    the model config must set it ``>= target_max_pos``.
    """
    if not config.enabled or model_max_freq_pos is None or model_max_freq_pos <= 0:
        return
    if config.target_max_pos > model_max_freq_pos:
        raise ValueError(
            "pose.target_max_pos must be <= the model RoPE frequency-table length to avoid "
            "out-of-bounds RoPE indexing, but got "
            f"pose.target_max_pos={config.target_max_pos}, {who} freq table={model_max_freq_pos}. "
            "Raise freq_max_position_embeddings (or max_position_embeddings) to >= target_max_pos."
        )


def _sample_valid_end(
    sample_idx: int,
    seq_len: int,
    tokens: np.ndarray | None,
    labels: np.ndarray | None,
    pad_id: int | None,
    ignore_index: int,
) -> int:
    """Index one-past the last non-(padded-tail) token; the trailing pad region is untouched.

    A padded-tail position has ``token == pad_id`` AND ``label == ignore_index``. Requires
    rank-2 ``tokens``+``labels`` and a ``pad_id``; otherwise the whole row is valid.
    """
    if tokens is None or labels is None or pad_id is None or tokens.ndim != 2 or labels.ndim != 2:
        return seq_len
    pad_tail = (tokens[sample_idx] == pad_id) & (labels[sample_idx] == ignore_index)
    non_tail = np.flatnonzero(~pad_tail)
    return int(non_tail[-1]) + 1 if non_tail.size > 0 else 0


def _sample_segment_ranges(
    sample_idx: int,
    valid_end: int,
    positions: np.ndarray,
    seq_lens: np.ndarray | None,
    seq_lens_num: np.ndarray | None,
    cu_seqlens: np.ndarray | None,
    cu_seqlens_num: np.ndarray | None,
) -> list[tuple[int, int]]:
    """Per-document ``(start, end)`` ranges within ``[0, valid_end)``; first source wins.

    Priority: explicit ``seq_lens`` (cumulative lengths) → ``cu_seqlens`` (cumulative
    offsets) → ``positions == 0`` resets → a single ``[0, valid_end)`` segment.
    """
    if valid_end <= 0:
        return []

    if seq_lens is not None and seq_lens.ndim == 2 and sample_idx < seq_lens.shape[0]:
        if seq_lens_num is not None and seq_lens_num.ndim == 1 and sample_idx < seq_lens_num.shape[0]:
            n_segments = int(seq_lens_num[sample_idx])
        else:
            n_segments = int((seq_lens[sample_idx] > 0).sum())
        boundaries = [0]
        cursor = 0
        for seg_len_val in seq_lens[sample_idx][:n_segments].tolist():
            cursor += int(seg_len_val)
            seg_end = min(cursor, valid_end)
            if seg_end > boundaries[-1]:
                boundaries.append(seg_end)
            if cursor >= valid_end:
                break
        if boundaries[-1] != valid_end:
            boundaries.append(valid_end)
    elif cu_seqlens is not None and cu_seqlens.ndim == 2 and sample_idx < cu_seqlens.shape[0]:
        if cu_seqlens_num is not None and cu_seqlens_num.ndim == 1 and sample_idx < cu_seqlens_num.shape[0]:
            n_boundaries = int(cu_seqlens_num[sample_idx])
        else:
            row = cu_seqlens[sample_idx]
            n_boundaries = int((row[1:] > row[:-1]).sum()) + 1
        n_boundaries = max(1, min(n_boundaries, cu_seqlens.shape[1]))
        boundaries = [0]
        for boundary_val in cu_seqlens[sample_idx, 1:n_boundaries].tolist():
            boundary = min(int(boundary_val), valid_end)
            if boundary > boundaries[-1]:
                boundaries.append(boundary)
            if boundary >= valid_end:
                break
        if boundaries[-1] != valid_end:
            boundaries.append(valid_end)
    else:
        reset_points = (np.flatnonzero(positions[sample_idx, 1:valid_end] == 0) + 1).tolist()
        boundaries = [0, *reset_points, valid_end]

    return list(pairwise(boundaries))


def apply_pose_to_positions(
    positions: np.ndarray,
    *,
    config: PoSEConfig,
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
    """Apply PoSE in place, mutating only ``positions`` (a rank-2 ``[B, S]`` int array).

    Per row: strip the padded tail, find per-document segments, then for each document draw
    Bernoulli(p) → split point → gap (in that order from ``rng``) and rewrite the two chunks
    with the skip. No-op when disabled, ``p <= 0``, ``S < 2``, ``valid_end < 2``, ``L < 2``,
    or ``max_gap <= 0``. ``positions`` must be rank-2 (1-D positions); callers handle any
    multi-axis (mRoPE) layout upstream.

    When ``image_token_id`` is given, any document whose tokens contain it is left untouched
    (text-only PoSE) — its multimodal positions are not 1-D contiguous and must not be split.
    """
    if not config.enabled:
        return
    p_pose = current_pose_probability(config, step)
    if p_pose <= 0.0:
        return
    if positions.ndim != 2:
        raise ValueError(f"PoSE expects rank-2 [B, S] positions, got shape {tuple(positions.shape)}.")
    seq_len = positions.shape[1]
    if seq_len < 2:
        return

    for sample_idx in range(positions.shape[0]):
        valid_end = _sample_valid_end(sample_idx, seq_len, tokens, labels, pad_id, ignore_index)
        if valid_end < 2:
            continue
        segment_ranges = _sample_segment_ranges(
            sample_idx, valid_end, positions, seq_lens, seq_lens_num, cu_seqlens, cu_seqlens_num
        )
        sample_pos = positions[sample_idx]
        # capture per-doc bases BEFORE any mutation so each doc uses its own original base
        seg_bases = {start: int(sample_pos[start]) for start, end in segment_ranges if end - start >= 2}

        for start, end in segment_ranges:
            segment_len = end - start
            if segment_len < 2:
                continue
            # text-only PoSE: skip documents that contain image placeholder tokens (their
            # positions are multimodal/non-contiguous) — before drawing, so RNG stays aligned.
            if (
                image_token_id is not None
                and tokens is not None
                and bool(np.any(tokens[sample_idx, start:end] == image_token_id))
            ):
                continue
            if rng.random() >= p_pose:  # per-document Bernoulli(p)
                continue

            min_chunk = max(1, min(math.ceil(segment_len * config.min_chunk_frac), segment_len - 1))
            max_chunk = max(1, min(math.floor(segment_len * config.max_chunk_frac), segment_len - 1))
            if min_chunk > max_chunk:
                split_idx = max(1, min(segment_len // 2, segment_len - 1))
            elif min_chunk == max_chunk:
                split_idx = min_chunk
            else:
                split_idx = int(rng.integers(min_chunk, max_chunk + 1))  # inclusive upper

            base = seg_bases[start]
            max_gap = config.target_max_pos - base - segment_len
            if max_gap <= 0:
                continue

            if config.bias_large_gap:
                u = rng.random()
                u = 1.0 - (1.0 - u) * (1.0 - u)  # concave warp toward large gaps
                gap = int(u * (max_gap + 1))
            else:
                gap = int(rng.integers(0, max_gap + 1))  # uniform, inclusive

            second_start = base + split_idx + gap
            second_len = segment_len - split_idx
            sample_pos[start : start + split_idx] = base + np.arange(split_idx, dtype=sample_pos.dtype)
            sample_pos[start + split_idx : end] = second_start + np.arange(second_len, dtype=sample_pos.dtype)


def _seq_lens_from_segment_ids(segment_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row document lengths from packed ``segment_ids`` ``[B, S]`` (contiguous-equal runs).

    Returns ``(seq_lens[B, max_segs], seq_lens_num[B])`` for the core's ``seq_lens`` path. Every
    run (including any trailing padding run) becomes a segment; the core then caps them at
    ``valid_end`` so the padded tail is dropped.
    """
    rows: list[list[int]] = []
    for row in segment_ids:
        change = (np.flatnonzero(np.diff(row)) + 1).tolist()
        bounds = [0, *change, len(row)]
        rows.append([bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)])
    max_segs = max((len(r) for r in rows), default=1)
    seq_lens = np.zeros((len(rows), max_segs), dtype=np.int64)
    seq_lens_num = np.zeros(len(rows), dtype=np.int64)
    for b, lens in enumerate(rows):
        seq_lens[b, : len(lens)] = lens
        seq_lens_num[b] = len(lens)
    return seq_lens, seq_lens_num


def apply_pose_to_batch(
    batch: dict,
    *,
    config: PoSEConfig,
    step: int,
    seed: int,
    image_token_id: int | None,
    pad_id: int | None,
    ignore_index: int = IGNORE_INDEX,
) -> dict:
    """Text-only PoSE on a collated training batch, mutating ``batch["position_ids"]`` in place.

    Host-side, per-microbatch, run before the forward (KD: mutate once so teacher+student share
    the positions). Handles both 3-D mRoPE ``[3, B, S]`` and 1-D ``[B, S]`` ``position_ids``; for
    3-D the identical skip is applied to all three axes (same ``seed`` ⇒ same draws), so text
    documents (whose axes are equal arange) stay consistent and image documents are untouched on
    every axis. Documents containing ``image_token_id`` are skipped (Option A: text-only).

    ``seed`` + ``step`` deterministically seed a fresh ``numpy`` generator per microbatch.
    No-op when disabled or ``p <= 0``. Raises if enabled but the batch has no precomputed
    ``position_ids`` (a real misconfiguration — the packed collator always emits them; we do not
    fabricate positions, which would break mRoPE).
    """
    if not config.enabled or current_pose_probability(config, step) <= 0.0:
        return batch
    if "position_ids" not in batch:
        raise ValueError(
            "pose.enabled but the collated batch has no 'position_ids' to transform "
            "(the packed collator is expected to emit them)."
        )
    positions = np.asarray(batch["position_ids"])
    tokens = np.asarray(batch["input_ids"]) if "input_ids" in batch else None
    labels = np.asarray(batch["labels"]) if "labels" in batch else None
    seq_lens = seq_lens_num = None
    if "segment_ids" in batch:
        seq_lens, seq_lens_num = _seq_lens_from_segment_ids(np.asarray(batch["segment_ids"]))

    common = dict(
        config=config,
        step=step,
        tokens=tokens,
        labels=labels,
        pad_id=pad_id,
        ignore_index=ignore_index,
        seq_lens=seq_lens,
        seq_lens_num=seq_lens_num,
        image_token_id=image_token_id,
    )
    if positions.ndim == 3:  # mRoPE [axes, B, S] — identical skip on every axis
        for axis in range(positions.shape[0]):
            apply_pose_to_positions(positions[axis], rng=np.random.default_rng(seed + step), **common)
    elif positions.ndim == 2:  # 1-D [B, S]
        apply_pose_to_positions(positions, rng=np.random.default_rng(seed + step), **common)
    else:
        raise ValueError(f"batch position_ids must be rank-2 [B,S] or rank-3 [axes,B,S], got {positions.ndim}.")
    batch["position_ids"] = positions
    return batch
