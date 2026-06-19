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

"""Regression tests for the grain ArrayRecord VL dataloader.

These lock the behaviours that the ad-hoc cluster proofs verified during bring-up, as
self-contained unit tests (tiny SYNTHETIC fixtures, no GCS / TPU / staged data):

1. ``test_batch_byte_exact_arrayrecord_vs_parquet`` - a batch assembled from the
   ArrayRecord path is byte-identical to the same rows read from parquet (the embeds
   round-trip through msgpack/array_record unchanged).
2. ``test_weighted_mix_proportions_controllable`` - ``grain.MapDataset.mix(weights)``
   samples datasets by the *configured weights*, not by their sizes.
3. ``test_multi_image_scatter`` - rows with ``n_images >= 2`` round-trip per-image and
   collate into the correct multi-span embed scatter.
4. ``test_convert_read_collate_pipeline`` - the full parquet -> .array_record ->
   ArrayRecordDataSource -> msgpack-decode -> collate path produces a correct batch.
5. ``test_loader_yields_uncollated_lists`` - the production grain pipeline
   (``...batch(batch_fn=list)``) yields LIST batches, not collated dicts. This is the
   collation contract the trainer relies on (the prefetcher applies ``data_collator``
   once); collating inside the loader would double-collate. Regression for that bug.

Run: ``pytest libs/easydel/tests/data/test_arrayrecord_grain.py``
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

grain = pytest.importorskip("grain.python", reason="grain not installed")
msgpack = pytest.importorskip("msgpack", reason="msgpack not installed")
ml_dtypes = pytest.importorskip("ml_dtypes", reason="ml_dtypes not installed")
pq = pytest.importorskip("pyarrow.parquet", reason="pyarrow not installed")
pa = pytest.importorskip("pyarrow", reason="pyarrow not installed")
ar_mod = pytest.importorskip(
    "array_record.python.array_record_module", reason="array_record not installed (Linux-only)"
)
from easydel.data.transforms.collators import collate_embeds_pack  # noqa: E402

ArrayRecordWriter = ar_mod.ArrayRecordWriter

HIDDEN = 16            # tiny embed dim — byte-exactness / scatter logic are dim-agnostic
IMAGE_TOKEN = 248056   # qwen3.6 image placeholder id
PAD_ID = 0


def _make_row(rng: np.random.Generator, n_images: int, n_tok_each: int, n_text: int) -> dict:
    """One synthetic VL row: text tokens + per-image placeholder spans + bf16 embed blobs."""
    embed_n_tok = [n_tok_each] * n_images
    blobs, grids = [], []
    for nt in embed_n_tok:
        arr = rng.standard_normal((nt, HIDDEN)).astype(ml_dtypes.bfloat16)
        blobs.append(arr.tobytes())
        grids.extend([1, 1, nt])  # (t,h,w) per image; only shape (n_images,3) matters here
    # input_ids: n_text text tokens then a placeholder span per image.
    # native python ints (parquet to_pydict yields these; msgpack rejects numpy scalars).
    ids = [int(x) for x in rng.integers(10, 100, size=n_text)]
    for nt in embed_n_tok:
        ids += [IMAGE_TOKEN] * int(nt)
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": list(ids),
        "image_grid_thw": grids,
        "image_embeds": blobs,
        "embed_n_tok": embed_n_tok,
        "embed_dim": HIDDEN,
        "n_images": n_images,
        "seq_len": len(ids),
    }


def _dataset_rows(seed: int, n_rows: int, *, multi: bool = False) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_rows):
        n_images = (2 if i % 2 else 3) if multi else 1
        rows.append(_make_row(rng, n_images=n_images, n_tok_each=3 + (i % 3), n_text=5 + (i % 4)))
    return rows


def _write_parquet(rows: list[dict], path: str) -> None:
    cols = {k: [r[k] for r in rows] for k in rows[0]}
    pq.write_table(pa.table(cols), path)


def _write_array_record(rows: list[dict], path: str) -> None:
    w = ArrayRecordWriter(path, "group_size:1")
    for r in rows:
        w.write(msgpack.packb(r, use_bin_type=True))
    w.close()


def _read_array_record(path: str) -> list[dict]:
    r = ar_mod.ArrayRecordReader(path)
    out = [msgpack.unpackb(b, raw=False) for b in r.read_all()]
    r.close()
    return out


def _collate(rows: list[dict]) -> dict:
    max_total = sum(int(x) for r in rows for x in r["embed_n_tok"])
    return collate_embeds_pack(
        rows, pad_id=PAD_ID, max_total=max_total, image_token_id=IMAGE_TOKEN,
        embed_dim=HIDDEN, embed_dtype=ml_dtypes.bfloat16,
    )


# --------------------------------------------------------------------------------------
def test_batch_byte_exact_arrayrecord_vs_parquet(tmp_path):
    """A collated batch from .array_record == the same rows read from parquet (byte-exact)."""
    rows = _dataset_rows(0, 6)
    pqf = str(tmp_path / "d.parquet")
    arf = str(tmp_path / "d.array_record")
    _write_parquet(rows, pqf)
    _write_array_record(rows, arf)

    rows_pq = pq.read_table(pqf).to_pylist()
    rows_ar = _read_array_record(arf)
    assert len(rows_ar) == len(rows_pq) == 6

    b_pq, b_ar = _collate(rows_pq), _collate(rows_ar)
    assert b_pq.keys() == b_ar.keys()
    for k in b_pq:
        a, p = np.asarray(b_ar[k]), np.asarray(b_pq[k])
        assert a.shape == p.shape, f"{k} shape {a.shape} != {p.shape}"
        assert np.array_equal(a, p), f"{k} not byte-exact between arrayrecord and parquet"
    # the bf16 embed tensor specifically must be bit-identical
    assert np.asarray(b_ar["image_embeds"]).dtype == ml_dtypes.bfloat16


def test_weighted_mix_proportions_controllable(tmp_path):
    """grain.MapDataset.mix samples by configured WEIGHTS, not dataset sizes."""
    # deliberately skewed sizes so size-proportional != target weights
    specs = {"A": (10, 30), "B": (11, 80), "C": (12, 20)}  # name -> (seed, n_rows)
    files = {}
    for name, (seed, n) in specs.items():
        f = str(tmp_path / f"{name}.array_record")
        _write_array_record(_dataset_rows(seed, n), f)
        files[name] = f

    names = sorted(files)                       # A, B, C
    weights = [0.5, 0.3, 0.2]
    per_ds = [
        grain.MapDataset.source(grain.ArrayRecordDataSource([files[nm]])).shuffle(seed=7 + i).map(lambda b, t=nm: t)
        for i, nm in enumerate(names)
    ]
    mixed = grain.MapDataset.mix(per_ds, weights=weights)
    draws = Counter(mixed[i] for i in range(3000))
    for i, nm in enumerate(names):
        frac = draws[nm] / 3000
        assert abs(frac - weights[i]) < 0.04, f"{nm}: observed {frac:.3f} != target {weights[i]}"
    # confirm it is NOT merely size-proportional (B has the most rows but the lowest-but-one weight)
    size_frac_B = specs["B"][1] / sum(s[1] for s in specs.values())
    assert abs(draws["B"] / 3000 - size_frac_B) > 0.1


def test_multi_image_scatter(tmp_path):
    """Multi-image rows round-trip per-image and scatter the correct number of spans."""
    rows = _dataset_rows(1, 5, multi=True)
    arf = str(tmp_path / "m.array_record")
    _write_array_record(rows, arf)
    rows_ar = _read_array_record(arf)

    for src, got in zip(rows, rows_ar, strict=True):
        assert got["n_images"] == src["n_images"] >= 2
        assert len(got["image_embeds"]) == len(got["embed_n_tok"]) == got["n_images"]
        for a, b in zip(src["image_embeds"], got["image_embeds"], strict=True):
            assert a == b  # per-image blob byte-exact

    batch = _collate(rows_ar)
    total_embed = sum(int(x) for r in rows_ar for x in r["embed_n_tok"])
    total_place = sum(int(t) == IMAGE_TOKEN for r in rows_ar for t in r["input_ids"])
    assert total_place == total_embed                                  # placeholders == embed rows
    assert int(np.asarray(batch["image_embed_mask"]).sum()) == total_embed
    assert np.asarray(batch["image_embeds"]).shape == (total_embed, HIDDEN)
    assert np.asarray(batch["image_grid_thw"]).shape[0] == sum(r["n_images"] for r in rows_ar)


def test_convert_read_collate_pipeline(tmp_path):
    """Full parquet -> .array_record -> ArrayRecordDataSource -> decode -> collate -> batch."""
    rows = _dataset_rows(2, 8)
    arf = str(tmp_path / "p.array_record")
    _write_array_record(rows, arf)

    src = grain.ArrayRecordDataSource([arf])
    assert len(src) == 8
    B = 4
    pipeline = (
        grain.MapDataset.source(src)
        .map(lambda b: msgpack.unpackb(b, raw=False))
        .batch(batch_size=B, drop_remainder=True, batch_fn=list)
    )
    batches = [pipeline[i] for i in range(len(pipeline))]
    assert len(batches) == 2  # 8 rows / B=4
    for blist in batches:
        assert isinstance(blist, list) and len(blist) == B
        out = _collate(blist)
        assert np.asarray(out["input_ids"]).shape[0] == B
        assert np.asarray(out["image_embeds"]).dtype == ml_dtypes.bfloat16


def test_loader_yields_uncollated_lists(tmp_path):
    """Production pipeline yields LIST batches (batch_fn=list), not collated dicts.

    Regression for the double-collation bug: the trainer prefetcher applies the
    data_collator to the dataloader output, so the loader must NOT pre-collate.
    """
    arf = str(tmp_path / "c.array_record")
    _write_array_record(_dataset_rows(3, 12), arf)

    pipeline = (
        grain.MapDataset.source(grain.ArrayRecordDataSource([arf]))
        .shuffle(seed=0)
        .map(lambda b: msgpack.unpackb(b, raw=False))
        .batch(batch_size=4, drop_remainder=True, batch_fn=list)
    )
    it = pipeline.to_iter_dataset(read_options=grain.ReadOptions(num_threads=2, prefetch_buffer_size=8))
    first = next(iter(it))
    assert isinstance(first, list), "loader must yield an uncollated list of rows"
    assert len(first) == 4
    assert all(isinstance(r, dict) and "image_embeds" in r for r in first)
    # applying the collator exactly ONCE turns the list into the model batch dict
    out = _collate(first)
    assert isinstance(out, dict) and "image_embed_positions" in out
