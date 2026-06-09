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

"""Collators for precomputed "embeds-only" VLM data packs.

These rows already carry post-vision-tower image embeddings. Training decodes those blobs,
scatters them into caller-specified image placeholder positions, and skips the vision tower.
The collators do not own model-family constants: callers pass the image token id, optional
embedding width for all-text batches, and the model-specific mRoPE position helper when needed.

They were originally reference fixtures inside ``tests/data/test_embeds_pack.py`` and are
promoted here so a real training run can import them; the contract test now imports from this
module, keeping a single source of truth.

Three collators:
  * :func:`collate_embeds_pack`   - pads to the per-batch-max length (variable shape -> a JIT
    recompile whenever the batch-max changes). Simple; fine for eval / dynamic shapes.
  * :func:`collate_bucket_static` - pads every batch to a caller-fixed ``seq_len`` and
    ``max_embed_rows`` so one bucket compiles once (no per-batch recompiles). Use for
    training throughput.
  * :func:`collate_packed_embeds` - TRAINING-STATIC packed MATERIALIZER: turns one batch's
    per-window row assignments (``list[list[dict]]``, produced by
    :class:`~easydel.data.transforms.packer.EmbedsWindowPacker`) into a constant
    ``(n_windows, seq_len)`` batch that compiles once (the packed sibling of
    ``collate_bucket_static``; kills the per-example padding it leaves). Carries ``segment_ids`` +
    per-segment-reset 3D M-RoPE ``position_ids`` so packed documents stay isolated (block-diagonal
    attention + GDR recurrence reset, both folded off ``segment_ids`` by the trainer's
    ``compute_loss``). Highest MFU when examples are short relative to the bucket. Packing
    (which rows share a window) is the packer's job; this function is the pure deterministic shape
    layer -- the two-layer split keeps placement policy and M-RoPE/embed-scatter independent.

The first two are single-``list[dict]`` callables once their hyper-params are bound
(``functools.partial``), matching the ``collate_fn`` hook on ``AsyncDataLoader`` /
``batch_iterator``. The packed path is driven instead by an ``EmbedsWindowPacker`` (stateful,
carries rows across batches) whose ``emit()`` / ``flush()`` output is handed to
:func:`collate_packed_embeds`.

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

import typing as tp

import numpy as np

PositionIdFn = tp.Callable[..., tuple[tp.Any, tp.Any]]


def _resolve_embed_dtype(embed_dtype: tp.Any) -> np.dtype:
    return np.dtype(embed_dtype)


def _as_list(value: tp.Any) -> list[tp.Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def _infer_row_embed_dim(row: dict, embed_dim: int | None = None) -> int:
    """Resolve one row's image-embed width from explicit value, row metadata, or bf16 blob bytes."""
    if embed_dim is not None:
        return int(embed_dim)
    row_dim = row.get("embed_dim")
    if row_dim is not None:
        return int(row_dim)

    embed_n_tok = [int(v) for v in _as_list(row.get("embed_n_tok"))]
    blobs = _as_list(row.get("image_embeds"))
    for n_tok, blob in zip(embed_n_tok, blobs, strict=False):
        if n_tok <= 0:
            continue
        byte_len = len(blob)
        denom = n_tok * 2
        if byte_len % denom != 0:
            raise ValueError(f"cannot infer embed_dim: blob has {byte_len} bytes for {n_tok} bf16 rows")
        return byte_len // denom

    raise ValueError("embed_dim is required for an all-text batch/row with no image embedding blobs")


def _infer_batch_embed_dim(rows: list[dict], embeds_list: list[np.ndarray], embed_dim: int | None = None) -> int:
    """Resolve a single embed width for the materialized batch and reject mixed-width packs."""
    dims: set[int] = set()
    if embed_dim is not None:
        dims.add(int(embed_dim))
    for row in rows:
        row_dim = row.get("embed_dim")
        if row_dim is not None:
            dims.add(int(row_dim))
    for embeds in embeds_list:
        if embeds.ndim == 2 and embeds.shape[1] > 0:
            dims.add(int(embeds.shape[1]))
    if not dims:
        raise ValueError("embed_dim is required when every packed row is text-only")
    if len(dims) != 1:
        raise ValueError(f"mixed embed_dim values in one packed batch: {sorted(dims)}")
    return dims.pop()


def _decode_row_embeds(row: dict, embed_dim: int | None = None, embed_dtype: tp.Any = np.float32) -> np.ndarray:
    """Decode a row's per-image bf16 blobs to a single ``(sum(embed_n_tok), embed_dim)`` array."""
    import ml_dtypes

    resolved_dtype = _resolve_embed_dtype(embed_dtype)
    ed = _infer_row_embed_dim(row, embed_dim)
    ent = [int(v) for v in _as_list(row.get("embed_n_tok"))]
    blobs = _as_list(row.get("image_embeds"))
    if len(ent) != len(blobs):
        raise ValueError(f"embed_n_tok/image_embeds length mismatch: {len(ent)} != {len(blobs)}")
    mats = [
        np.frombuffer(b, dtype=ml_dtypes.bfloat16).reshape(ent[i], ed).astype(resolved_dtype, copy=False)
        for i, b in enumerate(blobs)
    ]
    if not mats:
        return np.zeros((0, ed), dtype=resolved_dtype)
    return np.concatenate(mats, axis=0)


def collate_embeds_pack(
    rows: list[dict],
    pad_id: int,
    max_total: int,
    *,
    image_token_id: int,
    embed_dim: int | None = None,
    embed_dtype: tp.Any = np.float32,
) -> dict:
    """Collate embed-pack rows into a static-shape batch for training (no vision tower).

    Pads ``input_ids`` with ``pad_id`` (must differ from the image placeholder so it is not
    mistaken for a visual slot), ``labels`` with ``-100``, and ``attention_mask`` with ``0`` to
    the batch-max length. All rows' decoded embeds are concatenated row-major and zero-padded to
    a fixed ``max_total`` rows so the scatter target has a static shape. ``image_grid_thw`` is
    stacked to ``(total_images, 3)``.
    """
    assert pad_id != image_token_id, "pad_id must not collide with the image placeholder id"
    resolved_dtype = _resolve_embed_dtype(embed_dtype)
    bsz = len(rows)
    max_len = max(len(r["input_ids"]) for r in rows)

    input_ids = np.full((bsz, max_len), pad_id, dtype=np.int32)
    attention_mask = np.zeros((bsz, max_len), dtype=np.int32)
    labels = np.full((bsz, max_len), -100, dtype=np.int32)

    embeds_list, grids = [], []
    image_embed_positions = np.zeros((max_total, 2), dtype=np.int32)
    image_embed_mask = np.zeros((max_total,), dtype=np.int32)
    embed_cursor = 0
    for bi, r in enumerate(rows):
        ids = np.asarray(r["input_ids"], dtype=np.int32)
        length = ids.shape[0]
        input_ids[bi, :length] = ids
        attention_mask[bi, :length] = np.asarray(r["attention_mask"], dtype=np.int32)
        labels[bi, :length] = np.asarray(r["labels"], dtype=np.int32)
        embeds_list.append(_decode_row_embeds(r, embed_dim, resolved_dtype))
        grids.append(np.asarray(r["image_grid_thw"], dtype=np.int32).reshape(-1, 3))
        place_positions = np.flatnonzero(ids == image_token_id).astype(np.int32, copy=False)
        n_place = place_positions.shape[0]
        image_embed_positions[embed_cursor : embed_cursor + n_place, 0] = bi
        image_embed_positions[embed_cursor : embed_cursor + n_place, 1] = place_positions
        image_embed_mask[embed_cursor : embed_cursor + n_place] = 1
        embed_cursor += n_place

    resolved_embed_dim = _infer_batch_embed_dim(rows, embeds_list, embed_dim)
    all_embeds = (
        np.concatenate(embeds_list, axis=0) if embeds_list else np.zeros((0, resolved_embed_dim), resolved_dtype)
    )
    n_real = all_embeds.shape[0]
    assert embed_cursor == n_real, (
        f"placeholder/embed mismatch: {embed_cursor} image-placeholder tokens but {n_real} decoded embed rows"
    )
    assert n_real <= max_total, f"decoded embed rows {n_real} exceed max_total {max_total}"
    padded = np.zeros((max_total, resolved_embed_dim), dtype=resolved_dtype)
    padded[:n_real] = all_embeds
    grid_thw = np.concatenate(grids, axis=0) if grids else np.zeros((0, 3), np.int32)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "image_embeds": padded,
        "image_embed_positions": image_embed_positions,
        "image_embed_mask": image_embed_mask,
        "image_grid_thw": grid_thw,
        "n_real_embeds": n_real,
    }


def collate_bucket_static(
    rows: list[dict],
    pad_id: int,
    seq_len: int,
    max_embed_rows: int,
    *,
    image_token_id: int,
    embed_dim: int | None = None,
    embed_dtype: tp.Any = np.float32,
) -> dict:
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
    assert pad_id != image_token_id, "pad_id must not collide with the image placeholder id"
    resolved_dtype = _resolve_embed_dtype(embed_dtype)

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
    image_embed_positions = np.zeros((max_embed_rows, 2), dtype=np.int32)
    image_embed_mask = np.zeros((max_embed_rows,), dtype=np.int32)
    embed_cursor = 0
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
        embeds_list.append(_decode_row_embeds(r, embed_dim, resolved_dtype))
        grids.append(np.asarray(r["image_grid_thw"], dtype=np.int32).reshape(-1, 3))
        place_positions = np.flatnonzero(ids == image_token_id).astype(np.int32, copy=False)
        n_place = place_positions.shape[0]
        image_embed_positions[embed_cursor : embed_cursor + n_place, 0] = bi
        image_embed_positions[embed_cursor : embed_cursor + n_place, 1] = place_positions
        image_embed_mask[embed_cursor : embed_cursor + n_place] = 1
        embed_cursor += n_place

    resolved_embed_dim = _infer_batch_embed_dim(rows, embeds_list, embed_dim)
    all_embeds = (
        np.concatenate(embeds_list, axis=0) if embeds_list else np.zeros((0, resolved_embed_dim), resolved_dtype)
    )
    n_real = all_embeds.shape[0]
    assert embed_cursor == n_real, (
        f"placeholder/embed mismatch: {embed_cursor} image-placeholder tokens but {n_real} decoded embed rows"
    )
    assert n_real <= max_embed_rows, f"decoded embed rows {n_real} exceed max_embed_rows {max_embed_rows}"
    padded = np.zeros((max_embed_rows, resolved_embed_dim), dtype=resolved_dtype)
    padded[:n_real] = all_embeds
    grid_thw = np.concatenate(grids, axis=0) if grids else np.zeros((0, 3), np.int32)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "image_embeds": padded,
        "image_embed_positions": image_embed_positions,
        "image_embed_mask": image_embed_mask,
        "image_grid_thw": grid_thw,
        "n_real_embeds": n_real,
    }


def collate_packed_embeds(
    packs: list[list[dict]],
    pad_id: int,
    seq_len: int,
    max_embed_rows: int,
    n_windows: int,
    *,
    image_token_id: int,
    embed_dim: int | None = None,
    embed_dtype: tp.Any = np.float32,
    position_id_fn: PositionIdFn | None = None,
) -> dict:
    """Packed TRAINING-STATIC MATERIALIZER: turn one batch's per-window row assignments into a
    constant ``(n_windows, seq_len)`` batch on EVERY call (so the pod compiles the step once).
    ``packs`` is ``list[list[dict]]`` -- the ``M`` (<= ``n_windows``) non-empty windows produced by
    one :meth:`~easydel.data.transforms.packer.EmbedsWindowPacker.emit` / ``flush`` step; window
    ``wi`` holds ``packs[wi]`` concatenated in order. PLACEMENT (which rows share a window) is the
    packer's job -- this is the pure deterministic shape layer. Versus ``collate_bucket_static`` this
    kills the per-example padding (it pads EVERY short example out to the bucket ceiling): denser
    windows -> less wasted compute -> higher MFU.

    Window-count contract (the packer owns row sizing; this layer pins the static shape):
      * ``M`` real windows are filled from ``packs``;
      * ``M < n_windows`` -> the trailing ``n_windows - M`` windows are emitted WHOLE-padded
        (``attention_mask`` 0, ``segment_ids`` -1, ``labels`` -100, no image placeholders) so they
        contribute zero loss and zero attention -- a fully inert pad batch row (proven backward-finite);
      * ``M > n_windows`` -> raise. The packer never produces this (it opens exactly ``n_windows``
        bins per emit and carries overflow to the next batch, never dropping rows); the assert is a
        defensive shape contract against a hand-built ``packs``.

    Document isolation inside a window is carried by ``segment_ids`` -- a per-token example index
    that RESTARTS at 0 in each window. The trainer's ``compute_loss`` folds it into the universal
    ``mask_info`` (``MaskInfo.from_segments``), which drives BOTH block-diagonal full attention and
    the GDR linear-attention recurrence/conv reset, so examples neither attend to nor carry state
    across one another (proven equal to per-document runs in tests/modules/test_qwen3_5_packing.py,
    forward and backward).

    3D M-RoPE ``position_ids`` are reset per example: each segment's positions restart at 0. This is
    obtained by calling the caller-supplied model position helper on each example in isolation and
    concatenating. Text-only examples use a simple per-example 1D sequence repeated over the 3 mRoPE
    axes when no helper is supplied.

    Image embeds are decoded and concatenated in EXAMPLE order (== the row-major placeholder scan of
    the flattened ``(n_windows, seq_len)`` batch) and zero-padded to ``max_embed_rows``;
    ``image_grid_thw`` concatenated likewise. ASSERTS #image-placeholder tokens == #decoded embed
    rows so the scatter target stays aligned.

    The per-row admission guards (``len(input_ids) <= seq_len`` and ``embed_rows <= max_embed_rows``)
    live on the packer (:meth:`EmbedsWindowPacker.push`); the defensive per-window capacity assert
    below catches a hand-built ``packs`` whose rows overflow ``seq_len``.
    """
    assert pad_id != image_token_id, "pad_id must not collide with the image placeholder id"
    resolved_dtype = _resolve_embed_dtype(embed_dtype)

    # Partition homogeneity (best-effort; no-op when rows lack the path-only Hive keys -- see module
    # docstring). Keep source/area_bucket: decoded embeds + the scatter assume one teacher/resolution
    # regime. Do NOT check slen_band: packing DELIBERATELY mixes sub-band lengths into one window, so
    # the ``collate_bucket_static`` ``slen_band == seq_len`` equality does not apply here.
    flat_rows = [r for pack in packs for r in pack]
    for key in ("source", "area_bucket"):
        vals = {r[key] for r in flat_rows if key in r}
        assert len(vals) <= 1, (
            f"heterogeneous batch: packing requires one source x area_bucket partition per batch, "
            f"got mixed {key} {sorted(vals)} -- configure one loader per source/area_bucket dir"
        )

    # Static leading-dim contract: at most n_windows real windows so (n_windows, seq_len) is constant
    # and the step compiles once. The packer guarantees this (opens exactly n_windows bins per emit,
    # carries overflow forward); the assert defends against a bad hand-built packs. Rows are never
    # dropped -- overflow rows are carried to the next batch by the packer.
    assert len(packs) <= n_windows, (
        f"window overflow: {len(packs)} windows handed in but n_windows={n_windows} -- the packer "
        "opens exactly n_windows bins per emit and carries overflow forward (it never drops rows)"
    )

    # Allocate the FULL static (n_windows, seq_len). Windows M..n_windows-1 are left at these init
    # values -> whole-padded inert rows: pad_id ids, attn 0, labels -100, segment_ids -1 (== the
    # masked "no segment" value), positions 0. They carry no placeholders and contribute zero loss.
    input_ids = np.full((n_windows, seq_len), pad_id, dtype=np.int32)
    attention_mask = np.zeros((n_windows, seq_len), dtype=np.int32)
    labels = np.full((n_windows, seq_len), -100, dtype=np.int32)
    segment_ids = np.full((n_windows, seq_len), -1, dtype=np.int32)
    position_ids = np.zeros((3, n_windows, seq_len), dtype=np.int32)

    embeds_list, grids = [], []
    image_embed_positions = np.zeros((max_embed_rows, 2), dtype=np.int32)
    image_embed_mask = np.zeros((max_embed_rows,), dtype=np.int32)
    embed_cursor = 0
    for wi, pack in enumerate(packs):
        offset = 0
        for seg_idx, r in enumerate(pack):
            ids = np.asarray(r["input_ids"], dtype=np.int32)
            length = ids.shape[0]
            assert offset + length <= seq_len, (
                f"window capacity exceeded: window {wi} fills to {offset + length} tokens > seq_len "
                f"{seq_len} -- the packer must never overfill a bin (hand-built packs?)"
            )
            window_slice = slice(offset, offset + length)
            input_ids[wi, window_slice] = ids
            attention_mask[wi, window_slice] = np.asarray(r["attention_mask"], dtype=np.int32)
            labels[wi, window_slice] = np.asarray(r["labels"], dtype=np.int32)
            segment_ids[wi, window_slice] = seg_idx

            # Per-example reset 3D M-RoPE: run the caller's model helper on THIS example alone,
            # then drop the result into the window slice. The generic collator only marks image
            # placeholders; model-family details live in ``position_id_fn``.
            grid = np.asarray(r["image_grid_thw"], dtype=np.int32).reshape(-1, 3)
            has_images = bool(np.any(ids == image_token_id))
            if has_images:
                if position_id_fn is None:
                    raise ValueError("position_id_fn is required for packed rows containing image placeholders")
                mm_token_type_ids = (ids == image_token_id).astype(np.int32).reshape(1, -1)
                example_positions, _ = position_id_fn(
                    input_ids=ids.reshape(1, -1),
                    mm_token_type_ids=mm_token_type_ids,
                    image_grid_thw=grid if grid.shape[0] else None,
                    attention_mask=None,
                )
            else:
                example_positions = np.arange(length, dtype=np.int32).reshape(1, 1, -1).repeat(3, axis=0)
            position_ids[:, wi, window_slice] = np.asarray(example_positions)[:, 0, :]

            embeds_list.append(_decode_row_embeds(r, embed_dim, resolved_dtype))
            grids.append(grid)
            place_positions = np.flatnonzero(ids == image_token_id).astype(np.int32, copy=False)
            n_place = place_positions.shape[0]
            image_embed_positions[embed_cursor : embed_cursor + n_place, 0] = wi
            image_embed_positions[embed_cursor : embed_cursor + n_place, 1] = offset + place_positions
            image_embed_mask[embed_cursor : embed_cursor + n_place] = 1
            embed_cursor += n_place
            offset += length

    flat_rows = [r for pack in packs for r in pack]
    resolved_embed_dim = _infer_batch_embed_dim(flat_rows, embeds_list, embed_dim)
    all_embeds = (
        np.concatenate(embeds_list, axis=0) if embeds_list else np.zeros((0, resolved_embed_dim), resolved_dtype)
    )
    n_real = all_embeds.shape[0]
    n_place = int((input_ids == image_token_id).sum())
    assert n_place == n_real, (
        f"placeholder/embed mismatch: {n_place} image-placeholder tokens but {n_real} decoded embed "
        "rows -- packing must preserve the per-example placeholder<->embed alignment"
    )
    assert embed_cursor == n_real, (
        f"placeholder/embed metadata mismatch: {embed_cursor} coordinate rows but {n_real} decoded embed rows"
    )
    assert n_real <= max_embed_rows, f"decoded embed rows {n_real} exceed max_embed_rows {max_embed_rows}"
    padded = np.zeros((max_embed_rows, resolved_embed_dim), dtype=resolved_dtype)
    padded[:n_real] = all_embeds
    grid_thw = np.concatenate(grids, axis=0) if grids else np.zeros((0, 3), np.int32)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "segment_ids": segment_ids,
        "position_ids": position_ids,
        "image_embeds": padded,
        "image_embed_positions": image_embed_positions,
        "image_embed_mask": image_embed_mask,
        "image_grid_thw": grid_thw,
        "n_real_embeds": n_real,
    }
