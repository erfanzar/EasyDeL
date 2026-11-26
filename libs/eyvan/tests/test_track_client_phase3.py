# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Rollouts, completion groups, tables and media in the in-job ``eyvan.track`` client."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

import eyvan.track as track
from eyvan.track import _format as fmt
from eyvan.track._store import LocalTrackingStore, TrackingStoreError

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x01" * 40


class BrokenStore:
    @property
    def root(self) -> str:
        return "gs://unreachable/tracking/default/p"

    def put(self, key: str, data: bytes) -> None:
        raise TrackingStoreError("network unreachable")

    def get(self, key: str) -> bytes:
        raise TrackingStoreError("network unreachable")

    def list(self, prefix: str = "") -> list[str]:
        raise TrackingStoreError("network unreachable")


def _records(store: LocalTrackingStore, run_id: str, stream: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for key in store.list(f"{run_id}/{stream}"):
        out.extend(fmt.decode_segment(store.get(key)))
    return out


def _start(tmp_path: Path, store: object | None = None, **kwargs: object) -> track.Run:
    kwargs.setdefault("environment", {})
    kwargs.setdefault("flush_interval_s", 3600.0)
    kwargs.setdefault("spill_dir", tmp_path / "spill")
    return track.start(store=store or LocalTrackingStore(tmp_path / "store"), **kwargs)  # type: ignore[arg-type]


def test_rollout_from_steps_records_tracks_actions_return_and_video(tmp_path: Path) -> None:
    store = LocalTrackingStore(tmp_path / "store")
    run = _start(tmp_path, store, run_id="r")
    steps = [
        {"reward": 0.0, "value": 0.5, "action": 2},
        {"reward": 1.0, "value": 0.7, "action": 3},
        {"reward": float("nan"), "value": 0.1, "action": None, "done": 1},
    ]
    assert run.rollout(steps, action_names=["NOOP", "FIRE", "UP", "DOWN"], actor=3, video=MP4, fps=30, step=7)
    run.finish()

    (record,) = _records(store, "r", fmt.STREAM_ROLLOUTS)
    assert record["length"] == 3 and record["step"] == 7 and record["actor"] == "3"
    assert record["tracks"]["reward"] == [0.0, 1.0, "nan"]
    assert record["tracks"]["done"] == [None, None, 1.0]
    assert record["actions"] == [2, 3, None]
    assert record["action_names"] == ["NOOP", "FIRE", "UP", "DOWN"]
    assert record["ret"] == 1.0  # NaN step kept in the track, excluded from the return
    digest = hashlib.sha256(MP4).hexdigest()
    assert record["video"] == {"sha256": digest, "content_type": "video/mp4", "bytes": len(MP4)}
    assert record["fps"] == 30.0
    assert store.get(fmt.blob_key("r", digest)) == MP4


def test_rollout_sampling_every_n_and_tracks_form(tmp_path: Path) -> None:
    store = LocalTrackingStore(tmp_path / "store")
    run = _start(tmp_path, store, run_id="r")
    kept = [run.rollout(tracks={"reward": [i, 0.0]}, actions=[0, 1], every=3) for i in range(7)]
    run.finish()

    assert kept == [True, False, False, True, False, False, True]
    assert [r["tracks"]["reward"][0] for r in _records(store, "r", fmt.STREAM_ROLLOUTS)] == [0.0, 3.0, 6.0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"steps": [{"reward": 1}], "tracks": {"reward": [1]}},
        {"tracks": {"reward": [1, 2]}, "actions": [1]},
        {"tracks": {"bad name": [1]}},
        {"tracks": {"reward": ["x"]}},
        {"tracks": {"reward": [1]}, "every": 0},
        {"tracks": {"reward": [1]}, "kind": "test"},
        {"tracks": {"reward": [1]}, "fps": 0},
        {"tracks": {"reward": [1]}, "video": PNG, "video_type": "image/png"},
        {"tracks": {"reward": [1]}, "video": b"x", "video_type": "text/html"},
        {"tracks": {f"t{i}": [1] for i in range(33)}},
        {"tracks": {"reward": []}},
    ],
)
def test_rollout_rejects_bad_arguments(tmp_path: Path, kwargs: dict[str, object]) -> None:
    run = _start(tmp_path, run_id="r")
    with pytest.raises((TypeError, ValueError)):
        run.rollout(**kwargs)  # type: ignore[arg-type]
    run.finish()


def test_completions_compute_group_normalized_advantages_from_components(tmp_path: Path) -> None:
    store = LocalTrackingStore(tmp_path / "store")
    run = _start(tmp_path, store, run_id="g")
    run.completions(
        "If 3x + 7 = 25, what is x²?",
        ["\\boxed{36}", "<b>x</b>", "12"],
        rewards={"correct": [1, 1, 0], "format": [0.2, 0.0, 0.2]},
        tokens=[9, 4, 3],
        kl=[0.01, float("nan"), 0.02],
        truncated=[False, None, True],
        answer="36",
        group="g-812",
        step=260,
    )
    run.completions("same", ["a", "b"], rewards=[1.0, 1.0], step=260)
    run.finish()

    first, second = _records(store, "g", fmt.STREAM_COMPLETIONS)
    assert (first["group"], first["step"], first["answer"]) == ("g-812", 260, "36")
    totals = [r["reward"] for r in first["responses"]]
    assert totals == pytest.approx([1.2, 1.0, 0.2])
    mean = sum(totals) / 3
    std = math.sqrt(sum((t - mean) ** 2 for t in totals) / 3)
    assert [r["advantage"] for r in first["responses"]] == pytest.approx([(t - mean) / std for t in totals])
    assert first["responses"][1]["text"] == "<b>x</b>"
    assert first["responses"][0]["rewards"] == {"correct": 1.0, "format": 0.2}
    assert [r["kl"] for r in first["responses"]] == [0.01, None, 0.02]
    assert [r["truncated"] for r in first["responses"]] == [False, None, True]
    assert [r["advantage"] for r in second["responses"]] == [0.0, 0.0]
    assert second["group"].startswith("g1-0-")


def test_completions_clip_long_text_visibly_and_reject_mismatches(tmp_path: Path) -> None:
    store = LocalTrackingStore(tmp_path / "store")
    run = _start(tmp_path, store, run_id="g")
    run.completions("p" * 40_000, ["r"], rewards=[1.0], step=0)
    for kwargs in (
        {"prompt": "p", "responses": []},
        {"prompt": "p", "responses": ["a"], "rewards": [1, 2]},
        {"prompt": "p", "responses": ["a"], "truncated": [1]},
        {"prompt": 3, "responses": ["a"]},
        {"prompt": "p", "responses": ["a"] * 65},
    ):
        with pytest.raises((TypeError, ValueError)):
            run.completions(**kwargs)  # type: ignore[arg-type]
    run.finish()

    (record,) = _records(store, "g", fmt.STREAM_COMPLETIONS)
    assert len(record["prompt"]) == 32_768
    stats = [r for r in _records(store, "g", fmt.STREAM_META) if r["op"] == "stats"][-1]
    assert stats["dropped"]["invalid"] == 1


def test_table_with_mapping_rows_and_blob_cells(tmp_path: Path) -> None:
    store = LocalTrackingStore(tmp_path / "store")
    run = _start(tmp_path, store, run_id="t")
    frame = run.blob(PNG, content_type="image/png")
    assert frame is not None
    run.table(
        "eval/cases", [{"case": "a", "score": 0.5, "ok": True, "frame": frame}, {"case": "b", "score": None}], step=3
    )
    run.table("plain", [[1, "x"], [2.5, "y"]], columns=["n", "s"], step=3)
    run.finish()

    by_key = {r["key"]: r for r in _records(store, "t", fmt.STREAM_TABLES)}
    assert by_key["eval/cases"]["columns"] == ["case", "score", "ok", "frame"]
    assert by_key["eval/cases"]["rows"] == [["a", 0.5, True, frame], ["b", None, None, None]]
    assert by_key["plain"]["rows"] == [[1, "x"], [2.5, "y"]]
    assert store.get(fmt.blob_key("t", frame["sha256"])) == PNG


@pytest.mark.parametrize(
    "args",
    [
        ([[1, 2]], None),
        ([[1]], ["a", "a"]),
        ([[1, 2]], ["a"]),
        ([[{"x": 1}]], ["a"]),
        ([[object()]], ["a"]),
        ([[1]] * 1001, ["a"]),
    ],
)
def test_table_rejects_bad_shapes(tmp_path: Path, args: tuple[object, object]) -> None:
    run = _start(tmp_path, run_id="t")
    with pytest.raises((TypeError, ValueError)):
        run.table("k", args[0], columns=args[1])  # type: ignore[arg-type]
    run.finish()


def test_media_from_path_is_content_addressed_and_deduplicated(tmp_path: Path) -> None:
    store = LocalTrackingStore(tmp_path / "store")
    image = tmp_path / "frame.png"
    image.write_bytes(PNG)
    run = _start(tmp_path, store, run_id="m")
    first = run.media("eval/frame", image, caption="step 3", step=3)
    second = run.media("eval/frame", PNG, content_type="image/png", step=4)
    run.finish()

    assert first == second == {"sha256": hashlib.sha256(PNG).hexdigest(), "content_type": "image/png", "bytes": len(PNG)}
    records = _records(store, "m", fmt.STREAM_MEDIA)
    assert [(r["step"], r["caption"]) for r in records] == [(3, "step 3"), (4, None)]
    assert store.list("m/blobs") == [fmt.blob_key("m", first["sha256"])]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"data_or_path": b"<html>", "content_type": "text/html"},
        {"data_or_path": b"x"},
        {"data_or_path": 42},
        {"data_or_path": "notes.txt"},
    ],
)
def test_media_rejects_unsupported_input(tmp_path: Path, kwargs: dict[str, object]) -> None:
    run = _start(tmp_path, run_id="m")
    with pytest.raises((TypeError, ValueError)):
        run.media("k", **kwargs)  # type: ignore[arg-type]
    run.finish()


def test_blob_budget_and_media_key_caps_drop_visibly(tmp_path: Path) -> None:
    store = LocalTrackingStore(tmp_path / "store")
    limits = track.TrackLimits(max_media_keys=1, max_blob_bytes=len(PNG) + 1)
    run = _start(tmp_path, store, run_id="c", limits=limits)
    assert run.media("a", PNG, content_type="image/png") is not None
    assert run.media("b", PNG, content_type="image/png") is None  # second key
    assert run.media("a", MP4, content_type="video/mp4") is None  # over the byte budget
    run.finish()

    assert len(_records(store, "c", fmt.STREAM_MEDIA)) == 1
    stats = [r for r in _records(store, "c", fmt.STREAM_META) if r["op"] == "stats"][-1]
    assert stats["dropped"]["keys"] == 1 and stats["dropped"]["queue"] == 1


def test_blobs_and_records_spill_and_replay_when_the_store_is_down(tmp_path: Path) -> None:
    spill = tmp_path / "spill"
    run = _start(tmp_path, BrokenStore(), run_id="off", spill_dir=spill)
    ref = run.media("frame", PNG, content_type="image/png", step=0)
    run.rollout(tracks={"reward": [1.0]}, step=0)
    assert run.finish() is False

    target = tmp_path / "recovered"
    result = track.replay_spill(spill, store_uri=target.as_uri())
    assert result.failed == 0
    recovered = LocalTrackingStore(target)
    assert ref is not None
    assert recovered.get(fmt.blob_key("off", ref["sha256"])) == PNG
    assert _records(recovered, "off", fmt.STREAM_MEDIA)[0]["blob"] == ref
    assert len(_records(recovered, "off", fmt.STREAM_ROLLOUTS)) == 1


def test_nonzero_rank_rollouts_are_kept_but_tables_and_media_are_not(tmp_path: Path) -> None:
    store = LocalTrackingStore(tmp_path / "store")
    actor = _start(tmp_path, store, run_id="a", environment={"JAX_PROCESS_ID": "2"})
    assert actor.rollout(tracks={"reward": [1.0]}, step=0)
    assert actor.media("k", PNG, content_type="image/png") is None
    actor.table("k", [[1]], columns=["a"])
    actor.completions("p", ["r"], rewards=[1.0])
    actor.finish()

    assert len(_records(store, "a", fmt.STREAM_ROLLOUTS)) == 1
    assert store.list("a/media") == store.list("a/tables") == store.list("a/completions") == []
    assert store.list("a/blobs") == []
