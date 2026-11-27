# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for W&B-style run naming, zero-config ``eyvan.track.start()``, and ``eyvan track pull``."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pytest
from click.testing import CliRunner

import eyvan.track as track
from eyvan.sdk.commands import track as track_commands
from eyvan.storage import (
    MonitoringBucketConfig,
    StorageConfig,
    StorageConfigStore,
    clear_gcp_context_cache,
)
from eyvan.track import _format as fmt
from eyvan.track._store import LocalTrackingStore, TrackingStoreError
from eyvan.track.tracker.index import TrackingIndex

_NAME_PATTERN = re.compile(r"^[a-z]+-[a-z]+-[1-9][0-9]?$")


def test_generate_run_name_format_determinism_and_prefix_sanitization() -> None:
    name = track.generate_run_name()
    assert _NAME_PATTERN.fullmatch(name), name
    assert fmt.validate_run_id(name) == name

    # Deterministic with seed and independent of global random state when unseeded.
    assert track.generate_run_name(seed=12345) == track.generate_run_name(seed=12345)
    assert track.generate_run_name(seed=1) != track.generate_run_name(seed=2)
    random.seed(42)
    seen = {track.generate_run_name() for _ in range(12)}
    assert len(seen) > 1

    # Prefix sanitization and 64-char cap.
    prefixed = track.generate_run_name(prefix="Qwen/Qwen3-1.7B_GRPO", seed=7)
    assert prefixed.startswith("qwen-qwen3-1-7b-grpo-")
    assert len(prefixed) <= 64
    assert fmt.validate_run_id(prefixed) == prefixed

    very_long = track.generate_run_name(prefix="model-" + "x" * 100, seed=7)
    assert len(very_long) <= 64
    assert not very_long.endswith("-")
    assert fmt.validate_run_id(very_long) == very_long

    # Punctuation-only prefix falls back cleanly to base slug.
    assert _NAME_PATTERN.fullmatch(track.generate_run_name(prefix="///...", seed=7))

    with pytest.raises(TypeError):
        track.generate_run_name(prefix=123)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        track.generate_run_name(seed=True)


def test_track_start_auto_naming_and_backwards_compatibility(tmp_path: Path) -> None:
    store = LocalTrackingStore(tmp_path / "store")
    spill = tmp_path / "spill"

    # Zero-config fresh run gets a W&B-style readable name.
    auto_run = track.start(store=store, environment={}, spill_dir=spill, flush_interval_s=3600)
    try:
        assert _NAME_PATTERN.fullmatch(auto_run.name), auto_run.name
        assert auto_run.id.startswith(f"{auto_run.name}-")
    finally:
        auto_run.finish()

    # Explicit run_id preserves legacy run.name == run_id unless auto_name=True.
    explicit_id_run = track.start(run_id="ppo-1", store=store, environment={}, spill_dir=spill, flush_interval_s=3600)
    try:
        assert explicit_id_run.name == "ppo-1"
    finally:
        explicit_id_run.finish()

    explicit_with_auto = track.start(
        run_id="ppo-2",
        auto_name=True,
        store=store,
        environment={},
        spill_dir=spill,
        flush_interval_s=3600,
    )
    try:
        assert explicit_with_auto.id == "ppo-2"
        assert _NAME_PATTERN.fullmatch(explicit_with_auto.name)
    finally:
        explicit_with_auto.finish()

    # Workload-resumed run keeps run.name == workload_id.
    wl_run = track.start(
        store=store,
        environment={"EYVAN_WORKLOAD_ID": "wl_abc123"},
        spill_dir=spill,
        flush_interval_s=3600,
    )
    try:
        assert wl_run.id == "wl_abc123"
        assert wl_run.name == "wl_abc123"
    finally:
        wl_run.finish()

    # auto_name=False restores legacy run-<timestamp>-<hex> naming.
    legacy = track.start(
        store=store,
        environment={},
        spill_dir=spill,
        flush_interval_s=3600,
        auto_name=False,
    )
    try:
        assert legacy.name == legacy.id
        assert legacy.id.startswith("run-")
    finally:
        legacy.finish()


def test_track_start_zero_config_uses_configured_monitoring_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(fmt.TRACKING_URI_ENV, raising=False)
    monkeypatch.delenv("EYVAN_WORKLOAD_ID", raising=False)
    clear_gcp_context_cache()

    StorageConfigStore(tmp_path / ".eyvan" / "storage.json").save(
        StorageConfig(
            project="example-project-a",
            monitoring=MonitoringBucketConfig(
                bucket="eyvan_monitoring_example-project-a",
                location="us-central1",
                prefix="tracking",
            ),
        )
    )

    # Without explicit store/uri/environment -> resolves configured monitoring bucket!
    remote_run = track.start(
        project="grpo-exp",
        spill_dir=tmp_path / "spill",
        flush_interval_s=3600,
    )
    try:
        assert remote_run.store_root == "gs://eyvan_monitoring_example-project-a/tracking/default/grpo-exp"
    finally:
        # Don't block waiting on real network upload in this offline test.
        remote_run.finish(timeout=0.01)

    # Hermetic callers passing environment={} bypass ~/.eyvan/storage.json and stay local!
    local_run = track.start(
        project="grpo-exp",
        environment={},
        spill_dir=tmp_path / "spill",
        flush_interval_s=3600,
    )
    try:
        assert local_run.store_root == (tmp_path / ".eyvan" / "tracking" / "default" / "grpo-exp").resolve().as_uri()
    finally:
        local_run.finish()

    # Explicit EYVAN_TRACKING_URI overrides storage.json.
    override_dir = tmp_path / "override"
    monkeypatch.setenv(fmt.TRACKING_URI_ENV, override_dir.as_uri())
    env_run = track.start(project="grpo-exp", spill_dir=tmp_path / "spill", flush_interval_s=3600)
    try:
        assert env_run.store_root == override_dir.resolve().as_uri()
    finally:
        env_run.finish()


def test_track_pull_syncs_from_configured_monitoring_bucket_and_indexes_locally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    clear_gcp_context_cache()

    # Seed a fake "remote" monitoring store with two scopes and a tracked run.
    remote_dir = tmp_path / "fake-gcs-bucket" / "tracking"
    run1_store = LocalTrackingStore(remote_dir / "default" / "rl-proj")
    run1 = track.start(
        "qwen3-grpo",
        run_id="run-alpha",
        store=run1_store,
        environment={},
        spill_dir=tmp_path / "spill",
        flush_interval_s=3600,
    )
    run1.log({"train/loss": 0.42}, step=1)
    run1.log({"train/loss": 0.21}, step=2)
    run1.finish()

    run2_store = LocalTrackingStore(remote_dir / "default" / "other-proj")
    run2 = track.start(
        "sft-baseline",
        run_id="run-beta",
        store=run2_store,
        environment={},
        spill_dir=tmp_path / "spill",
        flush_interval_s=3600,
    )
    run2.log({"train/loss": 1.0}, step=1)
    run2.finish()

    # Without configured monitoring bucket and no --store -> helpful error.
    runner = CliRunner()
    no_cfg = runner.invoke(track_commands.track, ["pull"])
    assert no_cfg.exit_code != 0
    assert "eyvan storage init" in no_cfg.output

    # Configure monitoring bucket in ~/.eyvan/storage.json and wire _open_pull_store to our fake remote.
    StorageConfigStore(tmp_path / ".eyvan" / "storage.json").save(
        StorageConfig(
            project="example-project-a",
            monitoring=MonitoringBucketConfig(
                bucket="eyvan_monitoring_example-project-a",
                location="us-central1",
                prefix="tracking",
            ),
        )
    )
    opened_uris: list[str] = []

    def fake_open_pull_store(uri: str) -> LocalTrackingStore:
        opened_uris.append(uri)
        assert uri == "gs://eyvan_monitoring_example-project-a/tracking"
        return LocalTrackingStore(remote_dir)

    monkeypatch.setattr(track_commands, "_open_pull_store", fake_open_pull_store)

    dest_dir = tmp_path / "pulled-tracking"
    first = runner.invoke(track_commands.track, ["pull", "--dest", str(dest_dir), "--project", "rl-proj", "--json"])
    assert first.exit_code == 0, first.output
    first_data = json.loads(first.output)
    assert opened_uris == ["gs://eyvan_monitoring_example-project-a/tracking"]
    assert first_data["prefix"] == "default/rl-proj"
    assert first_data["downloaded"] >= 2
    assert first_data["skipped"] == 0
    assert first_data["failed"] == 0

    # Verify pulled data is directly readable by TrackingIndex!
    pulled_store = LocalTrackingStore(dest_dir)
    index = TrackingIndex(tmp_path / "index.sqlite3")
    index.scan(pulled_store)
    runs = index.list_runs("default", "rl-proj")
    assert [r["run_id"] for r in runs] == ["run-alpha"]
    assert index.list_runs("default", "other-proj") == []

    # Second identical pull skips all existing immutable segments.
    second = runner.invoke(track_commands.track, ["pull", "--dest", str(dest_dir), "--project", "rl-proj", "--json"])
    assert second.exit_code == 0, second.output
    second_data = json.loads(second.output)
    assert second_data["downloaded"] == 0
    assert second_data["skipped"] == first_data["downloaded"]

    # Pulling with --run filters to the requested run across scopes.
    filtered = runner.invoke(track_commands.track, ["pull", "--dest", str(dest_dir), "--run", "run-beta", "--json"])
    assert filtered.exit_code == 0, filtered.output
    filtered_data = json.loads(filtered.output)
    assert filtered_data["downloaded"] >= 2
    index.scan(pulled_store)
    assert [r["run_id"] for r in index.list_runs("default", "other-proj")] == ["run-beta"]


def test_track_pull_reports_partial_errors_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FlakyStore:
        root = "gs://flaky/tracking"

        def list(self, prefix: str = "") -> list[str]:
            return ["default/p/r/meta/a0001-r0000-0123456789ab-00000000.ndjson.gz"]

        def get(self, key: str) -> bytes:
            raise TrackingStoreError("transient GCS 503")

        def put(self, key: str, data: bytes) -> None:
            raise NotImplementedError

    monkeypatch.setattr(track_commands, "_open_pull_store", lambda _uri: FlakyStore())
    res = CliRunner().invoke(
        track_commands.track,
        ["pull", "--store", "gs://flaky/tracking", "--dest", str(tmp_path / "dest"), "--json"],
    )
    assert res.exit_code != 0
    assert '"failed": 1' in res.output


def test_track_pull_watch_tails_new_segments_and_exits_cleanly_on_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote_dir = tmp_path / "remote"
    dest_dir = tmp_path / "local"
    remote_store = LocalTrackingStore(remote_dir)

    # Seed initial run before poll 1.
    run1 = track.start(
        "watch-run-1",
        run_id="run-watch-1",
        store=LocalTrackingStore(remote_dir / "default" / "rl-proj"),
        environment={},
        spill_dir=tmp_path / "spill",
        flush_interval_s=3600,
    )
    run1.log({"train/loss": 0.8}, step=1)
    run1.finish()

    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 1:
            # Add a second segment before poll 2.
            run2 = track.start(
                "watch-run-2",
                run_id="run-watch-2",
                store=LocalTrackingStore(remote_dir / "default" / "rl-proj"),
                environment={},
                spill_dir=tmp_path / "spill",
                flush_interval_s=3600,
            )
            run2.log({"train/loss": 0.4}, step=2)
            run2.finish()
        else:
            raise KeyboardInterrupt

    monkeypatch.setattr(track_commands, "_open_pull_store", lambda _uri: remote_store)
    monkeypatch.setattr(track_commands, "_watch_sleep", fake_sleep)

    runner = CliRunner()
    res = runner.invoke(
        track_commands.track,
        ["pull", "--store", str(remote_dir), "--dest", str(dest_dir), "--watch", "2.5"],
    )
    assert res.exit_code == 0, res.output
    assert sleeps == [2.5, 2.5]
    assert "Watching" in res.output
    assert "Stopped watching after 2 poll(s):" in res.output

    index = TrackingIndex(tmp_path / "watch-index.sqlite3")
    index.scan(LocalTrackingStore(dest_dir))
    assert {r["run_id"] for r in index.list_runs("default", "rl-proj")} == {"run-watch-1", "run-watch-2"}

    # Invalid watch interval fails cleanly.
    bad = runner.invoke(track_commands.track, ["pull", "--store", str(remote_dir), "--watch", "0"])
    assert bad.exit_code != 0
    assert "positive number" in bad.output


def test_track_start_handles_unwritable_store_and_spill_without_crashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    ro_root = tmp_path / "readonly"
    ro_root.mkdir()
    os.chmod(ro_root, 0o555)
    try:
        # Case 1: Read-only store and spill directories never crash start/log/flush/finish.
        run = track.start(
            "ro-test",
            uri=str(ro_root / "tracking"),
            spill_dir=ro_root / "spill",
            environment={},
            flush_interval_s=3600,
        )
        run.log({"train/loss": 1.23}, step=1)
        assert run.flush(timeout=5.0) is False
        assert run.finish(timeout=5.0) is False
        assert run.stats()["dropped"]["queue"] >= 1

        # Case 2: PermissionError during store_for_uri falls back gracefully instead of crashing.
        from eyvan.track import _run as run_module

        def denied_store(_uri: str):
            raise PermissionError("read-only home mount")

        with monkeypatch.context() as scoped:
            scoped.setattr(run_module, "store_for_uri", denied_store)
            run2 = track.start(
                "ro-store-init",
                uri=str(ro_root / "blocked"),
                spill_dir=ro_root / "spill",
                environment={},
                flush_interval_s=3600,
            )
            run2.log({"train/loss": 0.5}, step=1)
            assert run2.finish(timeout=5.0) is False
    finally:
        os.chmod(ro_root, 0o755)

    # Caller argument errors (bad URI scheme) still raise ValueError.
    with pytest.raises(ValueError, match="unsupported tracking store URI scheme"):
        track.start("bad-uri", uri="http://not-a-store", environment={})
