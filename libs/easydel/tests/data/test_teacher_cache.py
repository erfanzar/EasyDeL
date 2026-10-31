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

"""Offline teacher cache: on-disk format round trip, manifest contract, fail-closed lookup, window grid
placement, and the device-side top-k / tail math against an independent dense reference.

The format tests are host-side NumPy (``cpu_ok``); the device-math tests compile XLA and run on the
accelerator."""

from __future__ import annotations

import json

import jax
import numpy as np
import pytest
from easydel.data.teacher_cache import (
    LAYOUT_KEYS_FIELD,
    LAYOUT_SPAN_FIELD,
    MANIFEST_NAME,
    TeacherCacheMismatch,
    TeacherCacheMiss,
    TeacherCacheReader,
    TeacherCacheWriter,
    chunked_teacher_topk,
    layout_sidecar,
    records_from_windows,
    row_key,
    teacher_topk_from_logits,
    write_manifest,
)
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec

K = 4
META = {"teacher_uri": "teacher-a", "tokenizer": "tok-a", "k": K, "segment_align": 1}


def _record(rng, n, k=K, vocab=64):
    """A synthetic, self-consistent record: real top-k of a random distribution plus its exact tail."""
    logits = rng.normal(0, 2.0, (n, vocab))
    logp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    idx = np.argsort(-logp, axis=-1)[:, :k]
    top = np.take_along_axis(logp, idx, -1)
    tail = np.log(np.clip(1.0 - np.exp(top).sum(-1), 1e-12, 1.0))
    return {
        "input_ids": rng.integers(1, 1000, n).astype(np.int32),
        "supervised": (rng.random(n) < 0.7).astype(np.uint8),
        "topk_idx": idx.astype(np.int32),
        "topk_logp": top.astype(np.float32),
        "tail_logp": tail.astype(np.float32),
    }


def _dump(root, records_by_host, *, part_bytes=256 << 20):
    for host, records in records_by_host.items():
        writer = TeacherCacheWriter(str(root), host_index=host, k=K, part_bytes=part_bytes)
        for rec in records:
            writer.append(**rec)
        writer.close()
    return write_manifest(str(root), meta=META, host_count=len(records_by_host))


@pytest.mark.cpu_ok
def test_row_key_depends_on_tokens_only_and_is_dtype_stable():
    assert row_key([1, 2, 3]) == row_key(np.asarray([1, 2, 3], np.int64)) == row_key(np.asarray([1, 2, 3], np.int32))
    assert row_key([1, 2, 3]) != row_key([1, 2, 4])
    assert len(row_key([7])) == 16
    with pytest.raises(ValueError, match="1-D"):
        row_key(np.zeros((2, 2), np.int32))


@pytest.mark.cpu_ok
def test_multi_host_round_trip_with_part_rotation(tmp_path):
    rng = np.random.default_rng(0)
    host0 = [_record(rng, n) for n in (5, 9, 3, 12)]
    host1 = [_record(rng, n) for n in (7, 4)]
    manifest = _dump(tmp_path, {0: host0, 1: host1}, part_bytes=200)  # tiny parts -> several per host
    assert manifest["rows"] == 6 and manifest["tokens"] == sum(len(r["input_ids"]) for r in host0 + host1)
    assert len(manifest["parts"]) > 2
    assert all(p.startswith(("host-0000/", "host-0001/")) for p in manifest["parts"])

    reader = TeacherCacheReader(str(tmp_path), expect={"teacher_uri": "teacher-a", "tokenizer": "tok-a"})
    assert len(reader) == 6
    for rec in host0 + host1:
        got = reader.get(row_key(rec["input_ids"]), input_ids=rec["input_ids"])
        np.testing.assert_array_equal(got["input_ids"], rec["input_ids"])
        np.testing.assert_array_equal(got["supervised"], rec["supervised"])
        np.testing.assert_array_equal(got["topk_idx"], rec["topk_idx"])
        assert got["topk_logp"].dtype == np.float16
        np.testing.assert_allclose(got["topk_logp"].astype(np.float32), rec["topk_logp"], rtol=2e-3, atol=2e-3)
        np.testing.assert_allclose(got["tail_logp"].astype(np.float32), rec["tail_logp"], rtol=2e-3, atol=2e-3)


@pytest.mark.cpu_ok
def test_duplicate_rows_resolve_to_one_record(tmp_path):
    rng = np.random.default_rng(1)
    rec = _record(rng, 6)
    _dump(tmp_path, {0: [rec, dict(rec)]})
    assert len(TeacherCacheReader(str(tmp_path))) == 1


@pytest.mark.cpu_ok
def test_writer_rejects_bad_shapes_and_reuse(tmp_path):
    rng = np.random.default_rng(2)
    writer = TeacherCacheWriter(str(tmp_path), host_index=0, k=K)
    rec = _record(rng, 5)
    with pytest.raises(ValueError, match="shapes disagree"):
        writer.append(**{**rec, "topk_idx": rec["topk_idx"][:, :2]})
    with pytest.raises(ValueError, match="non-empty"):
        writer.append(**_record(rng, 0))
    writer.close()
    with pytest.raises(RuntimeError, match="closed"):
        writer.append(**rec)
    with pytest.raises(ValueError, match="k must be positive"):
        TeacherCacheWriter(str(tmp_path), host_index=0, k=0)


@pytest.mark.cpu_ok
def test_manifest_requires_contract_fields_and_every_host(tmp_path):
    rng = np.random.default_rng(3)
    writer = TeacherCacheWriter(str(tmp_path), host_index=0, k=K)
    writer.append(**_record(rng, 4))
    writer.close()
    with pytest.raises(ValueError, match="segment_align"):
        write_manifest(str(tmp_path), meta={k: v for k, v in META.items() if k != "segment_align"}, host_count=1)
    with pytest.raises(RuntimeError, match=r"hosts \[1\]"):
        write_manifest(str(tmp_path), meta=META, host_count=2, wait_seconds=0.1, poll_seconds=0.05)
    with pytest.raises(ValueError, match="k="):
        write_manifest(str(tmp_path), meta={**META, "k": K + 1}, host_count=1)
    assert not (tmp_path / MANIFEST_NAME).exists(), "a refused manifest must not be written"


@pytest.mark.cpu_ok
def test_reader_refuses_unfinished_or_mismatched_caches(tmp_path):
    with pytest.raises(TeacherCacheMismatch, match="did not finish"):
        TeacherCacheReader(str(tmp_path))
    rng = np.random.default_rng(4)
    _dump(tmp_path, {0: [_record(rng, 4)]})
    with pytest.raises(TeacherCacheMismatch, match="tokenizer"):
        TeacherCacheReader(str(tmp_path), expect={"tokenizer": "tok-b"})
    with pytest.raises(TeacherCacheMismatch, match="top-8"):
        TeacherCacheReader(str(tmp_path), topk=8)
    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({**manifest, "format_version": 99}))
    with pytest.raises(TeacherCacheMismatch, match="format_version"):
        TeacherCacheReader(str(tmp_path))


@pytest.mark.cpu_ok
def test_miss_and_token_mismatch_fail_closed(tmp_path):
    rng = np.random.default_rng(5)
    rec = _record(rng, 6)
    _dump(tmp_path, {0: [rec]})
    reader = TeacherCacheReader(str(tmp_path))
    with pytest.raises(TeacherCacheMiss):
        reader.get(row_key([1, 2, 3]), input_ids=np.asarray([1, 2, 3]))
    other = rec["input_ids"].copy()
    other[0] += 1
    with pytest.raises(TeacherCacheMismatch, match="differ"):
        reader.get(row_key(rec["input_ids"]), input_ids=other)


@pytest.mark.cpu_ok
def test_sliced_topk_rederives_the_tail_from_kept_mass(tmp_path):
    rng = np.random.default_rng(6)
    rec = _record(rng, 8)
    _dump(tmp_path, {0: [rec]})
    got = TeacherCacheReader(str(tmp_path), topk=2).get(row_key(rec["input_ids"]))
    assert got["topk_idx"].shape == (8, 2)
    np.testing.assert_array_equal(got["topk_idx"], rec["topk_idx"][:, :2])
    expected = np.log(np.clip(1.0 - np.exp(rec["topk_logp"][:, :2]).sum(-1), 1e-7, 1.0))
    np.testing.assert_allclose(got["tail_logp"].astype(np.float32), expected, rtol=5e-3, atol=5e-3)


@pytest.mark.cpu_ok
@pytest.mark.parametrize("align", [1, 4])
def test_layout_sidecar_fill_windows_and_records_round_trip(tmp_path, align):
    rng = np.random.default_rng(7)
    seq_len, n_windows = 24, 3
    recs = [_record(rng, n) for n in (5, 6, 7, 3)]
    _dump(tmp_path, {0: recs})
    placement = [[0, 1], [2, 3]]  # third window left empty
    keys, span = layout_sidecar(recs, placement, n_windows=n_windows, align=align)
    assert keys.shape == (n_windows, 2, 16) and span.shape == (n_windows, 2, 2)
    assert (span[2] == 0).all()
    assert span[0, 1, 0] == -(-5 // align) * align  # second doc starts at the aligned footprint of the first

    input_ids = np.zeros((n_windows, seq_len), np.int32)
    labels = np.full((n_windows, seq_len), -100, np.int32)
    for w, indices in enumerate(placement):
        for d, i in enumerate(indices):
            off, n = span[w, d]
            input_ids[w, off : off + n] = recs[i]["input_ids"]
            labels[w, off : off + n] = np.where(recs[i]["supervised"] > 0, recs[i]["input_ids"], -100)

    reader = TeacherCacheReader(str(tmp_path))
    grid = reader.fill_windows(keys, span, seq_len=seq_len, input_ids=input_ids)
    assert grid["teacher_topk_idx"].shape == (n_windows, seq_len, K)
    assert grid["teacher_tail_logp"].shape == (n_windows, seq_len)
    covered = np.zeros((n_windows, seq_len), bool)
    for w, indices in enumerate(placement):
        for d, i in enumerate(indices):
            off, n = span[w, d]
            covered[w, off : off + n] = True
            np.testing.assert_array_equal(grid["teacher_topk_idx"][w, off : off + n], recs[i]["topk_idx"])
    assert (grid["teacher_topk_idx"][~covered] == 0).all() and (grid["teacher_tail_logp"][~covered] == 0).all()

    batch = {"input_ids": input_ids, "labels": labels, LAYOUT_KEYS_FIELD: keys, LAYOUT_SPAN_FIELD: span}
    out = list(
        records_from_windows(
            batch,
            grid["teacher_topk_idx"],
            grid["teacher_topk_logp"],
            grid["teacher_tail_logp"],
            window_indices=range(n_windows),
            step=3,
        )
    )
    assert [o["window"] for o in out] == [0, 0, 1, 1] and all(o["step"] == 3 for o in out)
    for o, i in zip(out, [0, 1, 2, 3], strict=True):
        np.testing.assert_array_equal(o["input_ids"], recs[i]["input_ids"])
        np.testing.assert_array_equal(o["supervised"], recs[i]["supervised"])


@pytest.mark.cpu_ok
def test_fill_windows_refuses_missing_rows_and_bad_spans(tmp_path):
    rng = np.random.default_rng(8)
    recs = [_record(rng, 4)]
    _dump(tmp_path, {0: recs})
    reader = TeacherCacheReader(str(tmp_path))
    keys, span = layout_sidecar([*recs, _record(rng, 3)], [[0, 1]], n_windows=1)
    with pytest.raises(TeacherCacheMiss):
        reader.fill_windows(keys, span, seq_len=16)
    keys, span = layout_sidecar(recs, [[0]], n_windows=1)
    with pytest.raises(TeacherCacheMismatch, match="exceeds seq_len"):
        reader.fill_windows(keys, span, seq_len=2)
    bad = span.copy()
    bad[0, 0, 1] = 3
    with pytest.raises(TeacherCacheMismatch, match="slot says 3"):
        reader.fill_windows(keys, bad, seq_len=16)


@pytest.mark.cpu_ok
def test_records_from_windows_rejects_a_stale_sidecar():
    ids = np.asarray([[5, 6, 7, 0]], np.int32)
    keys, span = layout_sidecar([{"input_ids": [5, 6, 8]}], [[0]], n_windows=1)
    batch = {"input_ids": ids, "labels": ids, LAYOUT_KEYS_FIELD: keys, LAYOUT_SPAN_FIELD: span}
    z = np.zeros((1, 4, K))
    with pytest.raises(ValueError, match="does not match"):
        list(records_from_windows(batch, z, z, np.zeros((1, 4)), window_indices=[0]))


def _dense_reference(logits: np.ndarray, k: int):
    """Independent float64 reference: full log-softmax, top-k, and the exact tail log-mass."""
    x = logits.astype(np.float64)
    lse = np.log(np.exp(x - x.max(-1, keepdims=True)).sum(-1)) + x.max(-1)
    logp = x - lse[..., None]
    idx = np.argsort(-x, axis=-1, kind="stable")[..., :k]
    top = np.take_along_axis(logp, idx, -1)
    rest = x.copy()
    np.put_along_axis(rest, idx, -np.inf, -1)
    m = rest.max(-1)
    tail = np.log(np.exp(rest - m[..., None]).sum(-1)) + m - lse
    return idx, top, tail


def test_topk_from_logits_matches_dense_reference():
    logits = np.random.default_rng(10).normal(0, 3.0, (2, 7, 96)).astype(np.float32)
    idx, logp, tail = jax.jit(lambda x: teacher_topk_from_logits(x, 8))(jnp.asarray(logits))
    ref_idx, ref_logp, ref_tail = _dense_reference(logits, 8)
    np.testing.assert_array_equal(np.asarray(idx), ref_idx)
    np.testing.assert_allclose(np.asarray(logp), ref_logp, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(tail), ref_tail, rtol=1e-4, atol=1e-4)


def test_tail_stays_exact_when_topk_mass_is_nearly_one():
    """log(1 - sum p) would cancel to 0 here; lse(non-top-k) - lse(all) keeps the true ~-40 value."""
    logits = np.full((1, 32), -40.0, np.float32)
    logits[0, :2] = 0.0
    _, _, tail = jax.jit(lambda x: teacher_topk_from_logits(x, 2))(jnp.asarray(logits))
    _, _, ref_tail = _dense_reference(logits, 2)
    assert float(ref_tail[0]) < -35.0
    np.testing.assert_allclose(np.asarray(tail), ref_tail, rtol=1e-4)


def test_empty_tail_and_oversized_k():
    _, _, tail = jax.jit(lambda x: teacher_topk_from_logits(x, 4))(jnp.zeros((3, 4), jnp.float32))
    np.testing.assert_allclose(np.asarray(tail), -88.0)
    with pytest.raises(ValueError, match="exceeds the vocabulary"):
        teacher_topk_from_logits(jnp.zeros((2, 4), jnp.float32), 5)


@pytest.mark.parametrize("chunk_size", [3, 7, 16])
def test_chunked_topk_matches_unchunked(chunk_size):
    rng = np.random.default_rng(11)
    hidden = jnp.asarray(rng.normal(0, 1.0, (2, 13, 8)), jnp.float32)
    head = jnp.asarray(rng.normal(0, 1.0, (8, 80)), jnp.float32)

    def lm_head(h):
        return h @ head

    idx, logp, tail = jax.jit(lambda h: chunked_teacher_topk(h, lm_head, 6, chunk_size=chunk_size))(hidden)
    assert idx.shape == (2, 13, 6) and logp.dtype == jnp.float16 and tail.shape == (2, 13)
    ref_idx, ref_logp, ref_tail = _dense_reference(np.asarray(lm_head(hidden)), 6)
    np.testing.assert_array_equal(np.asarray(idx), ref_idx)
    np.testing.assert_allclose(np.asarray(logp, np.float32), ref_logp, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(np.asarray(tail, np.float32), ref_tail, rtol=2e-3, atol=2e-3)


def test_vocab_parallel_chunked_topk_matches_single_device():
    devices = np.array(jax.devices())
    if devices.size < 2:
        pytest.skip(f"vocabulary-parallel path needs >= 2 devices, have {devices.size}")
    tp = 2 if devices.size % 2 == 0 else int(devices.size)
    mesh = Mesh(devices.reshape(devices.size // tp, tp), ("dp", "tp"))
    rng = np.random.default_rng(12)
    vocab = 32 * tp
    hidden = jnp.asarray(rng.normal(0, 1.0, (devices.size // tp, 10, 8)), jnp.float32)
    head = jnp.asarray(rng.normal(0, 1.0, (8, vocab)), jnp.float32)

    def lm_head(h):
        return h @ head

    sharded = jax.jit(
        lambda h: chunked_teacher_topk(
            h,
            lm_head,
            5,
            chunk_size=4,
            vp_mesh=mesh,
            vp_logit_spec=PartitionSpec("dp", None, "tp"),
            vp_token_spec=PartitionSpec("dp", None),
        )
    )(hidden)
    ref_idx, ref_logp, ref_tail = _dense_reference(np.asarray(lm_head(hidden)), 5)
    np.testing.assert_array_equal(np.asarray(sharded[0]), ref_idx)
    np.testing.assert_allclose(np.asarray(sharded[1], np.float32), ref_logp, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(np.asarray(sharded[2], np.float32), ref_tail, rtol=2e-3, atol=2e-3)
