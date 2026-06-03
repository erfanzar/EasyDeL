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

"""Collators for the precomputed "embeds-only" VLM data pack.

QAT-distillation of the Qwen3.5-VL teacher consumes parquet rows that already carry
POST-vision-tower image embeddings (bf16). Training decodes those blobs, scatters them into
the ``<image>`` placeholder positions (``IMAGE_PLACEHOLDER_ID``) and SKIPS the vision tower.
``HIDDEN`` is the teacher hidden size. These collators turn a list of pack rows into a
batch dict the Qwen3.5 / Qwen3.5-MoE forward consumes via its ``image_embeds`` /
``image_grid_thw`` arguments.

They were originally reference fixtures inside ``tests/data/test_embeds_pack.py`` and are
promoted here so a real training run can import them; the contract test now imports from this
module, keeping a single source of truth.

Three collators:
  * :func:`collate_embeds_pack`   - pads to the per-batch-max length (variable shape -> a JIT
    recompile whenever the batch-max changes). Simple; fine for eval / dynamic shapes.
  * :func:`collate_bucket_static` - pads every batch to a caller-fixed ``seq_len`` and
    ``max_embed_rows`` so one bucket compiles once (no per-batch recompiles). Use for
    training throughput.
  * :func:`collate_packed_embeds` - greedily concatenates several examples into each fixed
    ``seq_len`` window (killing the per-example padding ``collate_bucket_static`` leaves) and
    emits ``segment_ids`` + per-segment-reset 3D M-RoPE ``position_ids`` so packed documents stay
    isolated (block-diagonal attention + GDR recurrence reset, both folded off ``segment_ids`` by
    the trainer's ``compute_loss``). Highest MFU when examples are short relative to the bucket.
    All three are single-``list[dict]`` callables once their hyper-params are bound
    (``functools.partial``), matching the ``collate_fn`` hook on ``AsyncDataLoader`` /
    ``batch_iterator``.

WIRING (no new abstraction needed)
----------------------------------
Collation is a many-rows -> one-batch reduction, so it does NOT fit the row-level
:class:`~easydel.data.transforms.base.Transform` contract (one-in / one-out) or
:class:`ExpandTransform` (one-in / many-out). The framework already owns collation at the
batch boundary: the pipeline's ``LoadStage`` reads ``collate_fn`` off its
:class:`~easydel.data.core.config.LoadStageConfig` and hands it to every ``AsyncDataLoader``
it builds (so the same callable plugs into both the imperative ``batch_iterator`` path and the
Stage/Pipeline DSL). Bind the hyper-params and drop the result straight into that config::

    from functools import partial
    LoadStageConfig(..., collate_fn=partial(collate_bucket_static,
                                            pad_id=0, seq_len=<band edge>, max_embed_rows=<cap>))

i.e. these stay plain ``collate_fn`` callables rather than a bespoke collate Stage, which would
only duplicate ``LoadStage``.

PARTITION-COLUMN SEAM (read before wiring this into a loader)
-------------------------------------------------------------
The pack is laid out as Hive partitions ``source=<>/area_bucket=<>/slen_band=<>/part-*.parquet``.
Those three keys live in the directory PATH, not in the parquet row body. ``ParquetShardedSource``
reads the body only, so rows it yields do NOT carry ``source`` / ``area_bucket`` / ``slen_band``.

Consequence: :func:`collate_bucket_static`'s partition-homogeneity guard and its
``slen_band == seq_len`` cross-check are BEST-EFFORT — they fire only when rows happen to carry
those keys and silently no-op otherwise (they are ``if key in r``-tolerant by design). The hard
``len(input_ids) <= seq_len`` overflow guard is always live and independent of this.

Usage that stays correct under that limitation: bind ONE loader to ONE partition directory
(``data_files`` = a single ``source=.../area_bucket=.../slen_band=.../`` dir — no glob spanning
partitions, no cross-partition shuffle) and pass the matching ``seq_len`` (that partition's
``slen_band`` upper edge) and ``max_embed_rows``. Every batch is then naturally homogeneous and
the inert guards cost nothing.

To make the guards LIVE (defense against a misconfigured spanning loader), the fix belongs on
the read side, not here: teach ``ParquetShardedSource`` to infer the Hive partition columns from
the file path and inject them into each row dict. That touches a shared loader used by every
parquet source, so it is intentionally left as a separate decision rather than baked in here.
"""

from __future__ import annotations

import numpy as np

# Pack contract constants (Qwen3_5Config ``image_token_id`` default / teacher hidden size).
IMAGE_PLACEHOLDER_ID = 248056
HIDDEN = 5120
# Teacher vision ``spatial_merge_size`` (Qwen3.5-VL = 2). The pack's per-image placeholder run
# length equals ``T * (grid_h // SPATIAL_MERGE_SIZE) * (grid_w // SPATIAL_MERGE_SIZE)``; the M-RoPE
# helper needs this to lay out image position ids that line up with those placeholder runs.
SPATIAL_MERGE_SIZE = 2


def _decode_row_embeds(row: dict) -> np.ndarray:
    """Decode a row's per-image bf16 blobs to a single ``(sum(embed_n_tok), embed_dim)`` f32 array."""
    import ml_dtypes

    ed = int(row["embed_dim"])
    ent = list(row["embed_n_tok"])
    blobs = row["image_embeds"]
    mats = [np.frombuffer(b, dtype=ml_dtypes.bfloat16).reshape(ent[i], ed).astype(np.float32) for i, b in enumerate(blobs)]
    if not mats:
        return np.zeros((0, ed), dtype=np.float32)
    return np.concatenate(mats, axis=0)


def collate_embeds_pack(rows: list[dict], pad_id: int, max_total: int) -> dict:
    """Collate embed-pack rows into a static-shape batch for training (no vision tower).

    Pads ``input_ids`` with ``pad_id`` (must differ from the image placeholder so it is not
    mistaken for a visual slot), ``labels`` with ``-100``, and ``attention_mask`` with ``0`` to
    the batch-max length. All rows' decoded embeds are concatenated row-major and zero-padded to
    a fixed ``max_total`` rows so the scatter target has a static shape. ``image_grid_thw`` is
    stacked to ``(total_images, 3)``.
    """
    import jax.numpy as jnp

    assert pad_id != IMAGE_PLACEHOLDER_ID, "pad_id must not collide with the image placeholder id"
    bsz = len(rows)
    max_len = max(len(r["input_ids"]) for r in rows)

    input_ids = np.full((bsz, max_len), pad_id, dtype=np.int32)
    attention_mask = np.zeros((bsz, max_len), dtype=np.int32)
    labels = np.full((bsz, max_len), -100, dtype=np.int32)

    embeds_list, grids = [], []
    for bi, r in enumerate(rows):
        ids = np.asarray(r["input_ids"], dtype=np.int32)
        length = ids.shape[0]
        input_ids[bi, :length] = ids
        attention_mask[bi, :length] = np.asarray(r["attention_mask"], dtype=np.int32)
        labels[bi, :length] = np.asarray(r["labels"], dtype=np.int32)
        embeds_list.append(_decode_row_embeds(r))
        grids.append(np.asarray(r["image_grid_thw"], dtype=np.int32).reshape(-1, 3))

    all_embeds = np.concatenate(embeds_list, axis=0) if embeds_list else np.zeros((0, HIDDEN), np.float32)
    n_real = all_embeds.shape[0]
    assert n_real <= max_total, f"decoded embed rows {n_real} exceed max_total {max_total}"
    padded = np.zeros((max_total, HIDDEN), dtype=np.float32)
    padded[:n_real] = all_embeds
    grid_thw = np.concatenate(grids, axis=0) if grids else np.zeros((0, 3), np.int32)

    return {
        "input_ids": jnp.asarray(input_ids),
        "attention_mask": jnp.asarray(attention_mask),
        "labels": jnp.asarray(labels),
        "image_embeds": jnp.asarray(padded),
        "image_grid_thw": jnp.asarray(grid_thw),
        "n_real_embeds": n_real,
    }


def collate_bucket_static(rows: list[dict], pad_id: int, seq_len: int, max_embed_rows: int) -> dict:
    """Bucketed STATIC-shape collator: pad every batch to a caller-fixed ``seq_len`` (the area_bucket
    sequence ceiling) and a fixed ``max_embed_rows`` embed-row cap, so all batches drawn from one
    bucket share an identical jitted shape and the training step compiles once per bucket (no
    per-batch recompiles). Contrast ``collate_embeds_pack``, which pads to the per-batch max length
    (variable shape -> a recompile whenever the batch-max changes).

    Hard overflow guard: a row with ``len(input_ids) > seq_len`` raises. The producer caps IMAGE
    tokens per bucket but NOT total seq_len (build.py records seq_len, never truncates), so a
    text-heavy row can exceed the static window. Silently overflowing would drop trailing tokens —
    including image placeholders — breaking the ``#placeholders == #embed rows`` invariant and making
    the scatter OOB-clamp wrong embeds onto real positions. The guard forces an explicit upstream
    policy (raise the ceiling / route to a larger bucket / truncate TEXT before collation) instead.
    """
    import jax.numpy as jnp

    assert pad_id != IMAGE_PLACEHOLDER_ID, "pad_id must not collide with the image placeholder id"

    # Belt-and-suspenders to the seq_len overflow assert below: a static-S batch must be homogeneous
    # in its partition key -- every row from ONE (source x area_bucket x slen_band) partition. The
    # loader guarantees this only when each AsyncDataLoader wraps a single partition dir (no glob
    # spanning partitions, no cross-partition shuffle); a misconfig that interleaves partitions would
    # force a single wrong static window. Key on the HIVE PARTITION columns (source/area_bucket/
    # slen_band), NOT on `subset`: at full-pack scale one source can hold many subsets, so a subset
    # check would falsely reject a valid single-source partition. Tolerant of rows lacking a label --
    # skip that level rather than raise KeyError for the wrong reason. NOTE: ParquetShardedSource does
    # not surface the Hive partition columns (they are path-only), so under that loader these checks
    # no-op and homogeneity rests on the one-loader-per-partition-dir usage pattern (see module docstring).
    for key in ("source", "area_bucket", "slen_band"):
        vals = {r[key] for r in rows if key in r}
        assert len(vals) <= 1, (
            f"heterogeneous batch: static-S collation requires one source x area_bucket x slen_band "
            f"partition per batch, got mixed {key} {sorted(vals)} -- configure one loader per "
            "partition dir (no spanning glob / no cross-partition shuffle)"
        )

    # static-S must equal this partition's slen_band upper edge -- the band IS the compiled window
    # (sink keys slen_band on band(seq_len); the collator's seq_len must be that same edge). Skipped
    # for rows lacking the label (pre-sub-band fixtures), where seq_len is the area_bucket ceiling.
    band_vals = {int(r["slen_band"]) for r in rows if "slen_band" in r}
    if band_vals:
        (band_edge,) = band_vals  # homogeneity asserted above
        assert band_edge == seq_len, (
            f"static-S mismatch: collator seq_len {seq_len} != partition slen_band edge {band_edge} -- "
            "the bucketed window must equal the band's upper edge"
        )

    bsz = len(rows)
    input_ids = np.full((bsz, seq_len), pad_id, dtype=np.int32)
    attention_mask = np.zeros((bsz, seq_len), dtype=np.int32)
    labels = np.full((bsz, seq_len), -100, dtype=np.int32)

    embeds_list, grids = [], []
    for bi, r in enumerate(rows):
        ids = np.asarray(r["input_ids"], dtype=np.int32)
        length = ids.shape[0]
        assert length <= seq_len, (
            f"overflow guard: row seq_len {length} exceeds bucket static window {seq_len} -- "
            "raise the bucket ceiling, route to a larger bucket, or truncate TEXT before collation"
        )
        input_ids[bi, :length] = ids
        attention_mask[bi, :length] = np.asarray(r["attention_mask"], dtype=np.int32)
        labels[bi, :length] = np.asarray(r["labels"], dtype=np.int32)
        embeds_list.append(_decode_row_embeds(r))
        grids.append(np.asarray(r["image_grid_thw"], dtype=np.int32).reshape(-1, 3))

    all_embeds = np.concatenate(embeds_list, axis=0) if embeds_list else np.zeros((0, HIDDEN), np.float32)
    n_real = all_embeds.shape[0]
    assert n_real <= max_embed_rows, f"decoded embed rows {n_real} exceed max_embed_rows {max_embed_rows}"
    padded = np.zeros((max_embed_rows, HIDDEN), dtype=np.float32)
    padded[:n_real] = all_embeds
    grid_thw = np.concatenate(grids, axis=0) if grids else np.zeros((0, 3), np.int32)

    return {
        "input_ids": jnp.asarray(input_ids),
        "attention_mask": jnp.asarray(attention_mask),
        "labels": jnp.asarray(labels),
        "image_embeds": jnp.asarray(padded),
        "image_grid_thw": jnp.asarray(grid_thw),
        "n_real_embeds": n_real,
    }


def collate_packed_embeds(
    rows: list[dict],
    pad_id: int,
    seq_len: int,
    max_embed_rows: int,
    spatial_merge_size: int = SPATIAL_MERGE_SIZE,
) -> dict:
    """Packed STATIC-shape collator: greedily concatenate several examples into each fixed
    ``seq_len`` window (FCFS; a row that will not fit starts the next window), removing the
    per-example padding ``collate_bucket_static`` leaves (which pads EVERY short example out to the
    full bucket ceiling). Denser windows -> less wasted compute -> higher training MFU.

    Document isolation inside a window is carried by ``segment_ids`` -- a per-token example index
    that RESTARTS at 0 in each window. The trainer's ``compute_loss`` folds it into the universal
    ``mask_info`` (``MaskInfo.from_segments``), which drives BOTH block-diagonal full attention and
    the GDR linear-attention recurrence/conv reset, so examples neither attend to nor carry state
    across one another (proven equal to per-document runs in tests/modules/test_qwen3_5_packing.py,
    forward and backward).

    3D M-RoPE ``position_ids`` are reset per example: each segment's positions restart at 0. This is
    obtained by calling the model's own ``_get_rope_index_from_mm_token_types`` on each example in
    isolation (its ``current_pos`` starts at 0) and concatenating -- no reimplementation of the
    mRoPE math, and it keeps positions from running cumulatively past ``max_position_embeddings``.
    The forward consumes caller ``position_ids`` directly (modeling_qwen3_5.py:1110) and flattens
    3D->1D itself when the text config does not enable mRoPE.

    Image embeds are decoded and concatenated in EXAMPLE order (== the row-major placeholder scan of
    the flattened ``(M, seq_len)`` batch) and zero-padded to ``max_embed_rows``; ``image_grid_thw``
    concatenated likewise. ASSERTS #image-placeholder tokens == #decoded embed rows so the scatter
    target stays aligned.

    Overflow guard (mirrors :func:`collate_bucket_static`): a single row longer than ``seq_len``
    raises rather than silently truncating placeholders.

    Returns ``(M, seq_len)`` for the M windows greedy packing produced. NOTE: M is data-dependent,
    so the leading batch dim is NOT static across calls -- ``seq_len`` and ``max_embed_rows`` are the
    statically pinned dims. For a compile-once leading dim, pack upstream (PackStage) or cap the
    window count; left out here to keep the single-``list[dict]`` contract of the sibling collators.
    """
    import jax.numpy as jnp

    from easydel.modules.qwen3_5.modeling_qwen3_5 import _get_rope_index_from_mm_token_types

    assert pad_id != IMAGE_PLACEHOLDER_ID, "pad_id must not collide with the image placeholder id"

    # Partition homogeneity (best-effort; no-op when rows lack the path-only Hive keys -- see module
    # docstring). Keep source/area_bucket: decoded embeds + the scatter assume one teacher/resolution
    # regime. Do NOT check slen_band: packing DELIBERATELY mixes sub-band lengths into one window, so
    # the ``collate_bucket_static`` ``slen_band == seq_len`` equality does not apply here.
    for key in ("source", "area_bucket"):
        vals = {r[key] for r in rows if key in r}
        assert len(vals) <= 1, (
            f"heterogeneous batch: packing requires one source x area_bucket partition per batch, "
            f"got mixed {key} {sorted(vals)} -- configure one loader per source/area_bucket dir"
        )

    # Greedy FCFS packing into fixed seq_len windows. A row that would overflow the current window
    # flushes it and starts the next; a single row longer than the window is unpackable -> raise.
    packs: list[list[dict]] = []
    current: list[dict] = []
    current_len = 0
    for r in rows:
        length = len(r["input_ids"])
        assert length <= seq_len, (
            f"overflow guard: row seq_len {length} exceeds packed window {seq_len} -- "
            "raise the window, route to a larger bucket, or truncate TEXT before collation"
        )
        if current and current_len + length > seq_len:
            packs.append(current)
            current, current_len = [], 0
        current.append(r)
        current_len += length
    if current:
        packs.append(current)

    n_win = len(packs)
    input_ids = np.full((n_win, seq_len), pad_id, dtype=np.int32)
    attention_mask = np.zeros((n_win, seq_len), dtype=np.int32)
    labels = np.full((n_win, seq_len), -100, dtype=np.int32)
    # segment_ids pad fill = -1 (the "no segment" value compute_loss also assigns to masked
    # positions); real tokens get their per-window example index below.
    segment_ids = np.full((n_win, seq_len), -1, dtype=np.int32)
    position_ids = np.zeros((3, n_win, seq_len), dtype=np.int32)

    embeds_list, grids = [], []
    for wi, pack in enumerate(packs):
        offset = 0
        for seg_idx, r in enumerate(pack):
            ids = np.asarray(r["input_ids"], dtype=np.int32)
            length = ids.shape[0]
            window_slice = slice(offset, offset + length)
            input_ids[wi, window_slice] = ids
            attention_mask[wi, window_slice] = np.asarray(r["attention_mask"], dtype=np.int32)
            labels[wi, window_slice] = np.asarray(r["labels"], dtype=np.int32)
            segment_ids[wi, window_slice] = seg_idx

            # Per-example reset 3D M-RoPE: run the model helper on THIS example alone so its
            # current_pos starts at 0, then drop the result into the window slice -> the segment's
            # positions reset at its start. mm_token_type_ids marks image placeholders (1) vs text
            # (0); image groups consume image_grid_thw to lay out 2D spatial positions.
            grid = np.asarray(r["image_grid_thw"], dtype=np.int32).reshape(-1, 3)
            mm_token_type_ids = (ids == IMAGE_PLACEHOLDER_ID).astype(np.int32).reshape(1, -1)
            example_positions, _ = _get_rope_index_from_mm_token_types(
                input_ids=ids.reshape(1, -1),
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=grid if grid.shape[0] else None,
                attention_mask=None,
                spatial_merge_size=spatial_merge_size,
            )
            position_ids[:, wi, window_slice] = np.asarray(example_positions)[:, 0, :]

            embeds_list.append(_decode_row_embeds(r))
            grids.append(grid)
            offset += length

    all_embeds = np.concatenate(embeds_list, axis=0) if embeds_list else np.zeros((0, HIDDEN), np.float32)
    n_real = all_embeds.shape[0]
    n_place = int((input_ids == IMAGE_PLACEHOLDER_ID).sum())
    assert n_place == n_real, (
        f"placeholder/embed mismatch: {n_place} image-placeholder tokens but {n_real} decoded embed "
        "rows -- packing must preserve the per-example placeholder<->embed alignment"
    )
    assert n_real <= max_embed_rows, f"decoded embed rows {n_real} exceed max_embed_rows {max_embed_rows}"
    padded = np.zeros((max_embed_rows, HIDDEN), dtype=np.float32)
    padded[:n_real] = all_embeds
    grid_thw = np.concatenate(grids, axis=0) if grids else np.zeros((0, 3), np.int32)

    return {
        "input_ids": jnp.asarray(input_ids),
        "attention_mask": jnp.asarray(attention_mask),
        "labels": jnp.asarray(labels),
        "segment_ids": jnp.asarray(segment_ids),
        "position_ids": jnp.asarray(position_ids),
        "image_embeds": jnp.asarray(padded),
        "image_grid_thw": jnp.asarray(grid_thw),
        "n_real_embeds": n_real,
    }
