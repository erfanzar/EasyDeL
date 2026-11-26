# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Local tracker ownership, CLI lifecycle and cloud dry-run safety."""

from __future__ import annotations

import json
import stat
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import eyvan.track
from eyvan.sdk.commands import track as commands
from eyvan.server.local import tracker


@pytest.fixture
def supervisor(tmp_path):
    live = set()
    arguments = {}
    launches = []

    def launch(argv, environment, log):
        pid = 98765
        launches.append((argv, environment, log))
        live.add(pid)
        arguments[pid] = argv
        return pid

    instance = tracker.LocalTrackerSupervisor(
        directory=tmp_path / "tracker",
        launcher=launch,
        health_probe=lambda url: bool(live),
        process_liveness=lambda pid: pid in live,
        argument_reader=arguments.get,
        process_terminator=lambda pid: live.discard(pid),
        process_killer=lambda pid: live.discard(pid),
    )
    return instance, launches, live, arguments


def test_local_tracker_lifecycle_private_record_and_idempotence(supervisor, tmp_path) -> None:
    instance, launches, live, _arguments = supervisor
    record = instance.up(store_uri=str(tmp_path / "runs"))
    assert len(launches) == 1
    assert launches[0][0][:3] == (tracker.sys.executable, "-m", "eyvan.track.tracker")
    assert launches[0][1]["EYVAN_TRACKER_TOKEN"] == tracker.read_tracker_token(record)
    assert stat.S_IMODE(instance.directory.stat().st_mode) == 0o700
    assert stat.S_IMODE((instance.directory / "token").stat().st_mode) == 0o600
    assert stat.S_IMODE((instance.directory / "record.json").stat().st_mode) == 0o600
    assert set(json.loads((instance.directory / "record.json").read_text())) == {
        "url",
        "pid",
        "store",
        "db",
        "log",
        "token_path",
    }
    assert tracker.read_tracker_record(instance.directory) == record
    assert instance.up(store_uri=str(tmp_path / "runs")) == record
    assert len(launches) == 1
    assert instance.status()["status"] == "running"
    instance.down()
    assert live == set()
    assert not (instance.directory / "record.json").exists()
    assert not (instance.directory / "token").exists()
    assert instance.status()["status"] == "stopped"


def test_local_tracker_refuses_to_kill_reused_pid(supervisor, tmp_path) -> None:
    instance, _launches, live, arguments = supervisor
    record = instance.up(store_uri=str(tmp_path / "runs"))
    arguments[record.pid] = ("unrelated", "process")
    with pytest.raises(tracker.LocalTrackerError, match="identity"):
        instance.down()
    assert record.pid in live
    assert (instance.directory / "record.json").exists()


def test_local_tracker_timeout_cleans_owned_child(supervisor, tmp_path) -> None:
    instance, _launches, live, _arguments = supervisor
    instance.health_probe = lambda url: False
    with pytest.raises(tracker.LocalTrackerError, match="health"):
        instance.up(store_uri=str(tmp_path / "runs"), timeout_s=0)
    assert live == set()
    assert not (instance.directory / "record.json").exists()


def test_cli_local_up_status_down(supervisor, tmp_path, monkeypatch) -> None:
    instance, _launches, live, _arguments = supervisor
    monkeypatch.setattr(commands, "LocalTrackerSupervisor", lambda: instance)
    runner = CliRunner()
    result = runner.invoke(commands.track, ["up", "--local", "--store", str(tmp_path / "runs")])
    assert result.exit_code == 0, result.output
    assert "8095" in result.output
    result = runner.invoke(commands.track, ["status", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "running"
    assert "EYVAN_TRACKER_TOKEN" not in result.output
    result = runner.invoke(commands.track, ["down"])
    assert result.exit_code == 0, result.output
    assert live == set()


def test_cli_sync_is_lazy_and_uses_replay_result(tmp_path, monkeypatch) -> None:
    calls = []

    def replay(directory, *, store_uri=None):
        calls.append((directory, store_uri))
        return SimpleNamespace(uploaded=3, failed=0)

    monkeypatch.setattr(eyvan.track, "replay_spill", replay, raising=False)
    result = CliRunner().invoke(commands.track, ["sync", str(tmp_path), "--store", "gs://runs/tracking"])
    assert result.exit_code == 0, result.output
    assert calls == [(tmp_path, "gs://runs/tracking")]
    assert "3" in result.output


def test_vm_plan_never_executes_without_flag(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    calls = []
    monkeypatch.setattr(commands, "run_command", lambda argv: calls.append(argv))
    result = CliRunner().invoke(commands.track, ["up", "--project", "test-project", "--store", "gs://runs/tracking"])
    assert result.exit_code == 0, result.output
    assert "SPOT" in result.output
    assert "e2" in result.output
    assert calls == []
    result = CliRunner().invoke(
        commands.track, ["up", "--project", "test-project", "--store", "gs://runs/tracking", "--execute"]
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0][:4] == ("gcloud", "compute", "instances", "create")
    assert "--provisioning-model=SPOT" in calls[0]
    assert "--quiet" in calls[0]


def test_vm_plan_requires_project_without_saved_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    result = CliRunner().invoke(commands.track, ["up"])
    assert result.exit_code != 0
    assert "--project" in result.output


def test_vm_plan_uses_saved_autoscale_project_without_writes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ".eyvan" / "autoscale"
    root.mkdir(parents=True)
    profile = root / "defaults.yaml"
    profile.write_text("project: saved-project\n")
    original = profile.read_bytes()
    monkeypatch.setattr(commands, "run_command", lambda argv: pytest.fail("dry run attempted cloud execution"))
    result = CliRunner().invoke(commands.track, ["up"])
    assert result.exit_code == 0, result.output
    assert "saved-project" in result.output
    assert profile.read_bytes() == original
    assert not (tmp_path / ".eyvan" / "tracker").exists()


def test_cli_sync_returns_nonzero_for_remaining_spills(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        eyvan.track, "replay_spill", lambda *args, **kwargs: SimpleNamespace(replayed=0, failed=2), raising=False
    )
    result = CliRunner().invoke(commands.track, ["sync", str(tmp_path)])
    assert result.exit_code == 1
    assert "2" in result.output
    assert "unsynced" in result.output


def test_local_tracker_refuses_token_symlink(supervisor, tmp_path) -> None:
    instance, _launches, _live, _arguments = supervisor
    record = instance.up(store_uri=str(tmp_path / "runs"))
    token = instance.directory / "token"
    original = tmp_path / "unrelated-secret"
    original.write_text("not-a-tracker-token")
    token.unlink()
    token.symlink_to(original)
    with pytest.raises(tracker.daemon.LocalDaemonStateError, match="regular file"):
        tracker.read_tracker_token(record)
    assert original.read_text() == "not-a-tracker-token"


def test_cli_registration_preserves_other_commands(tmp_path, monkeypatch) -> None:
    from eyvan.sdk.cli import main

    monkeypatch.setenv("HOME", str(tmp_path))
    result = CliRunner().invoke(main, ["track", "--help"])
    assert result.exit_code == 0, result.output
    for name in ("up", "status", "down", "sync"):
        assert name in result.output
    assert "up" in main.commands
    assert "capacity" in main.commands
    assert "runs" in main.commands


class _RecordingClient:
    """Answer plane requests from a table and record exactly what was asked."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def request(self, method, path, *, params=None, json=None, headers=None):
        self.calls.append((method, path, dict(params or {})))
        return self.responses[path]


def _runs_session(responses):
    client = _RecordingClient(responses)
    scope = SimpleNamespace(org_id="acme", project_id="rl")
    return SimpleNamespace(client=client, workload_scope=lambda: scope), client


def test_runs_list_requests_the_scoped_plane_route_and_renders_unknown(tmp_path) -> None:
    session, client = _runs_session(
        {
            "/api/v1/tracking/runs": {
                "runs": [
                    {"run_id": "wl_1", "status": "unknown", "last_step": None, "attempts": 2, "name": "ppo"},
                    {"run_id": "wl_2", "status": "finished", "last_step": 400, "attempts": 1, "name": "ppo-s1"},
                ]
            }
        }
    )
    result = CliRunner().invoke(commands.runs, ["list", "--status", "running", "--limit", "5"], obj=session)

    assert result.exit_code == 0, result.output
    assert client.calls == [
        ("GET", "/api/v1/tracking/runs", {"org_id": "acme", "project_id": "rl", "limit": 5, "status": "running"})
    ]
    assert "wl_1" in result.output and "unknown" in result.output
    assert "400" in result.output


def test_runs_show_and_series_hit_the_run_routes_with_encoded_ids(tmp_path) -> None:
    detail = {
        "run_id": "wl_1",
        "status": "running",
        "name": "ppo",
        "workload_id": "wl_1",
        "attempts": 3,
        "last_step": 330,
        "keys": ["loss/policy", "episode/return"],
        "stats": {"records": 10, "dropped": {"keys": 0, "rate": 1, "invalid": 0, "queue": 0}, "spilled": 0},
    }
    series_path = "/api/v1/tracking/runs/wl_1/series"
    session, client = _runs_session(
        {
            "/api/v1/tracking/runs/wl_1": detail,
            series_path: {"run_id": "wl_1", "x": "step", "series": [{"key": "k", "points": [[0, 1, 1, 1]]}]},
        }
    )
    shown = CliRunner().invoke(commands.runs, ["show", "wl_1"], obj=session)
    assert shown.exit_code == 0, shown.output
    assert "attempts   3" in shown.output and "loss/policy" in shown.output

    series = CliRunner().invoke(
        commands.runs, ["series", "wl_1", "--key", "a", "--key", "b", "--x", "env_steps", "--points", "50"], obj=session
    )
    assert series.exit_code == 0, series.output
    assert len(json.loads(series.output)["series"]) == 2
    assert client.calls[0] == ("GET", "/api/v1/tracking/runs/wl_1", {"org_id": "acme", "project_id": "rl"})
    assert [call[2]["key"] for call in client.calls[1:]] == ["a", "b"]
    assert {call[1] for call in client.calls[1:]} == {series_path}
    assert all(call[2]["x"] == "env_steps" and call[2]["points"] == 50 for call in client.calls[1:])


def test_runs_refuses_path_escaping_ids_before_any_request() -> None:
    session, client = _runs_session({})
    result = CliRunner().invoke(commands.runs, ["show", "../escape"], obj=session)
    assert result.exit_code != 0
    assert client.calls == []


def test_runs_episodes_list_breakdown_and_hist_hit_the_phase2_routes() -> None:
    base = "/api/v1/tracking/runs/wl_1"
    session, client = _runs_session(
        {
            f"{base}/episodes": {
                "episodes": [{"kind": "eval", "step": 40, "ret": 18.5, "length": 900, "actor": None}],
                "total": 1,
            },
            f"{base}/episodes/breakdown": {
                "actors": [{"actor": "3", "count": 12, "mean_return": 14.0, "max_lag": 2}],
                "envs": [],
                "lag": {},
                "evals": [{"checkpoint": "ckpt-40", "mean_return": 18.5, "best": True}],
                "replayed_excluded": 0,
            },
            f"{base}/histograms": {"run_id": "wl_1", "key": "adv", "edges": [], "columns": []},
        }
    )
    listed = CliRunner().invoke(
        commands.runs, ["episodes", "wl_1", "--kind", "eval", "--actor", "3", "--limit", "5"], obj=session
    )
    assert listed.exit_code == 0, listed.output
    assert "18.5" in listed.output
    assert client.calls[0] == (
        "GET",
        f"{base}/episodes",
        {"org_id": "acme", "project_id": "rl", "limit": 5, "kind": "eval", "actor": "3"},
    )

    broken = CliRunner().invoke(commands.runs, ["episodes", "wl_1", "--breakdown"], obj=session)
    assert broken.exit_code == 0, broken.output
    assert "ckpt-40" in broken.output and "*best" in broken.output
    assert client.calls[1] == ("GET", f"{base}/episodes/breakdown", {"org_id": "acme", "project_id": "rl"})

    hist = CliRunner().invoke(commands.runs, ["hist", "wl_1", "--key", "adv", "--bins", "16"], obj=session)
    assert hist.exit_code == 0, hist.output
    assert json.loads(hist.output)["key"] == "adv"
    assert client.calls[2] == (
        "GET",
        f"{base}/histograms",
        {"org_id": "acme", "project_id": "rl", "key": "adv", "x": "step", "points": 50, "bins": 16},
    )


def test_runs_curve_requests_the_episode_curve_route() -> None:
    path = "/api/v1/tracking/runs/wl_1/episodes/curve"
    session, client = _runs_session({path: {"run_id": "wl_1", "metric": "length", "buckets": []}})
    result = CliRunner().invoke(
        commands.runs, ["curve", "wl_1", "--metric", "length", "--kind", "eval", "--points", "10"], obj=session
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["metric"] == "length"
    assert client.calls == [
        (
            "GET",
            path,
            {"org_id": "acme", "project_id": "rl", "metric": "length", "kind": "eval", "x": "step", "points": 10},
        )
    ]


def test_runs_phase3_commands_hit_the_scoped_routes() -> None:
    base = "/api/v1/tracking/runs/wl_1"
    session, client = _runs_session(
        {
            f"{base}/rollouts": {"rollouts": [{"id": "a1-r0-w-0-0", "step": 5, "length": 3, "ret": 1.0}], "total": 1},
            f"{base}/rollouts/a1-r0-w-0-0": {"id": "a1-r0-w-0-0", "tracks": {"reward": [0, 1]}},
            f"{base}/completions": {"groups": [{"id": "g", "step": 2, "k": 8, "zero_variance": True}], "total": 1},
            f"{base}/completions/stats": {"steps": []},
            f"{base}/tables": {"keys": [{"key": "eval/cases", "count": 1, "last_step": 3}]},
            f"{base}/media": {"media": [{"key": "frame", "step": 1, "blob": {"available": False}}], "total": 1},
        }
    )
    runner = CliRunner()
    assert runner.invoke(commands.runs, ["rollouts", "wl_1", "--kind", "eval"], obj=session).exit_code == 0
    shown = runner.invoke(commands.runs, ["rollouts", "wl_1", "--id", "a1-r0-w-0-0"], obj=session)
    assert json.loads(shown.output)["tracks"]["reward"] == [0, 1]
    listed = runner.invoke(commands.runs, ["completions", "wl_1", "--sort", "zero_variance", "--step", "2"], obj=session)
    assert "ZERO-VARIANCE" in listed.output
    assert runner.invoke(commands.runs, ["completions", "wl_1", "--stats"], obj=session).exit_code == 0
    assert runner.invoke(commands.runs, ["tables", "wl_1"], obj=session).exit_code == 0
    media = runner.invoke(commands.runs, ["media", "wl_1"], obj=session)
    assert "not uploaded" in media.output
    scope = {"org_id": "acme", "project_id": "rl"}
    assert client.calls == [
        ("GET", f"{base}/rollouts", {**scope, "limit": 20, "kind": "eval"}),
        ("GET", f"{base}/rollouts/a1-r0-w-0-0", scope),
        ("GET", f"{base}/completions", {**scope, "limit": 20, "sort": "zero_variance", "step": 2}),
        ("GET", f"{base}/completions/stats", scope),
        ("GET", f"{base}/tables", scope),
        ("GET", f"{base}/media", {**scope, "limit": 50}),
    ]


def test_runs_blob_verifies_the_digest_before_writing(tmp_path) -> None:
    import hashlib

    data = b"\x89PNG-frame"
    digest = hashlib.sha256(data).hexdigest()
    calls = []

    def request_bytes(method, path, *, params=None, headers=None):
        calls.append((method, path, dict(params or {})))
        return (b"tampered" if "bad" in str(params) else data), "image/png"

    scope = SimpleNamespace(org_id="acme", project_id="rl")
    session = SimpleNamespace(client=SimpleNamespace(request_bytes=request_bytes), workload_scope=lambda: scope)
    out = tmp_path / "frame.png"
    result = CliRunner().invoke(commands.runs, ["blob", "wl_1", digest, "-o", str(out)], obj=session)
    assert result.exit_code == 0, result.output
    assert out.read_bytes() == data
    assert calls == [("GET", f"/api/v1/tracking/runs/wl_1/blobs/{digest}", {"org_id": "acme", "project_id": "rl"})]
    again = CliRunner().invoke(commands.runs, ["blob", "wl_1", digest, "-o", str(out)], obj=session)
    assert again.exit_code != 0 and "already exists" in again.output

    scope.project_id = "bad"
    other = tmp_path / "other.png"
    tampered = CliRunner().invoke(commands.runs, ["blob", "wl_1", digest, "-o", str(other)], obj=session)
    assert tampered.exit_code != 0 and not other.exists()
    malformed = CliRunner().invoke(commands.runs, ["blob", "wl_1", "nope", "-o", str(other)], obj=session)
    assert malformed.exit_code != 0


def test_runs_health_reports_the_plane_answer() -> None:
    session, client = _runs_session(
        {"/api/v1/tracking/health": {"status": "ok", "store": "gs://b/tracking", "indexed_segments": 7}}
    )
    result = CliRunner().invoke(commands.runs, ["health"], obj=session)
    assert result.exit_code == 0, result.output
    assert "ok" in result.output and "7 segments" in result.output
    assert client.calls == [("GET", "/api/v1/tracking/health", {})]
