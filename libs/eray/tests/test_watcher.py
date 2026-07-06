# Copyright 2026 The EasyDeL/eray Author @erfanzar (Erfan Zare Chavoshi).
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

"""Tests for the spot watcher: pure plan() truth table, effects, resubmission."""

from types import SimpleNamespace

import eray.provision.watcher as watcher_module
import pytest
from eray.provision.registry import ClusterRecord, ClusterRegistry, LocalBackend
from eray.provision.watcher import (
    Action,
    Observed,
    WatchPolicy,
    plan,
    resubmit_jobs,
    watch_and_reconnect,
)

NOW = 1_000_000.0


def make_record(**overrides) -> ClusterRecord:
    defaults = dict(
        name="trainer1",
        project="proj",
        zone="us-east5-a",
        accelerator_type="v5p-64",
        qr_id="trainer1",
        generation=0,
        state="HEALTHY",
    )
    defaults.update(overrides)
    return ClusterRecord(**defaults)


def obs(**overrides) -> Observed:
    defaults = dict(
        node_state="READY",
        node_head_ip="10.0.0.5",
        num_hosts=1,
        qr_state="ACTIVE",
        qr_error="",
        head_up=True,
        jobs=None,
        now=NOW,
    )
    defaults.update(overrides)
    return Observed(**defaults)


def kinds(actions: list[Action]) -> list[str]:
    return [a.kind for a in actions]


class TestPlanTruthTable:
    def test_healthy_snapshot_and_no_event_when_already_healthy(self):
        actions = plan(make_record(), obs(jobs=[{"submission_id": "j1"}]), WatchPolicy())
        assert kinds(actions) == ["set_state", "snapshot_jobs"]
        assert actions[0].args["state"] == "HEALTHY"
        assert actions[0].args["reset_incident"] is True

    def test_first_healthy_emits_healthy_not_recovered(self):
        actions = plan(make_record(state="UNKNOWN"), obs(), WatchPolicy())
        assert actions[-1].args["event"] == "healthy"

    def test_recovery_completion_emits_recovered(self):
        actions = plan(make_record(state="CONNECTING"), obs(), WatchPolicy())
        assert actions[-1].args["event"] == "cluster_recovered"

    def test_adopted_node_without_qr_is_healthy_and_never_deleted(self):
        # Truth-table row: QR gone out-of-band but node alive → watch only.
        actions = plan(make_record(state="ADOPTED"), obs(qr_state=None), WatchPolicy())
        assert "delete_qr" not in kinds(actions)
        assert actions[0].args["state"] == "HEALTHY"

    def test_dark_head_first_tick_only_counts(self):
        actions = plan(make_record(), obs(head_up=False), WatchPolicy())
        assert kinds(actions) == ["set_state"]
        assert actions[0].args["extra"] == {"unreach_ticks": 1}

    def test_dark_head_second_tick_attempts_repair_once(self):
        record = make_record(extra={"unreach_ticks": 1})
        actions = plan(record, obs(head_up=False), WatchPolicy())
        assert kinds(actions) == ["event", "set_state", "bootstrap", "connect"]
        assert actions[1].args["extra"] == {"repair_attempted": True}

    def test_dark_head_after_repair_recreates(self):
        record = make_record(extra={"unreach_ticks": 5, "repair_attempted": True})
        actions = plan(record, obs(head_up=False), WatchPolicy())
        assert "create_qr" in kinds(actions)
        # Node still exists → the QR delete must use --force.
        delete = next(a for a in actions if a.kind == "delete_qr")
        assert delete.args["force"] is True

    def test_preempted_node_triggers_recovery_sequence(self):
        actions = plan(make_record(generation=3, qr_id="trainer1-r3"), obs(node_state="PREEMPTED"), WatchPolicy())
        assert kinds(actions) == ["event", "set_state", "delete_qr", "create_qr", "record_recreate", "set_state"]
        assert actions[0].args["event"] == "preemption_detected"
        assert actions[1].args["state"] == "DEGRADED"
        assert actions[3].args["qr_id"] == "trainer1-r4"
        assert actions[-1].args["state"] == "WAITING"

    def test_suspended_qr_without_node_recovers_without_force(self):
        actions = plan(make_record(), obs(node_state=None, qr_state="SUSPENDED"), WatchPolicy())
        delete = next(a for a in actions if a.kind == "delete_qr")
        assert delete.args["force"] is False

    def test_deleting_node_waits_for_terminal_state(self):
        # A delete operation is in flight — acting now races it (observed
        # live: QR delete --force during node deletion fails, code 10
        # ABORTED). The plan must wait, not recover.
        actions = plan(make_record(state="HEALTHY"), obs(node_state="DELETING", qr_state="ACTIVE"), WatchPolicy())
        assert kinds(actions) == ["set_state"]
        assert "delete_qr" not in kinds(actions)

    def test_node_gone_but_qr_still_active_waits(self):
        # Right after a node death the QR can briefly still read ACTIVE
        # before SUSPENDING; recovery starts once the QR reaches a
        # gone-or-dead state.
        actions = plan(make_record(state="HEALTHY"), obs(node_state=None, qr_state="ACTIVE"), WatchPolicy())
        assert "delete_qr" not in kinds(actions)
        assert "create_qr" not in kinds(actions)

    def test_everything_gone_skips_delete(self):
        actions = plan(make_record(), obs(node_state=None, qr_state=None, head_up=None), WatchPolicy())
        assert "delete_qr" not in kinds(actions)
        assert "create_qr" in kinds(actions)

    def test_quota_failure_halts_never_loops(self):
        actions = plan(
            make_record(),
            obs(node_state=None, qr_state="FAILED", qr_error="User does not have permission ..."),
            WatchPolicy(),
        )
        assert kinds(actions) == ["event", "halt"]
        assert actions[1].args["state"] == "HALTED_QUOTA"

    def test_transient_failure_recreates(self):
        actions = plan(
            make_record(), obs(node_state=None, qr_state="FAILED", qr_error="capacity unavailable"), WatchPolicy()
        )
        assert "create_qr" in kinds(actions)

    def test_hourly_budget_halts(self):
        record = make_record(recreate_ts=[NOW - 100, NOW - 200, NOW - 300, NOW - 400])
        actions = plan(record, obs(node_state="PREEMPTED"), WatchPolicy(max_recreates_per_hour=4))
        assert actions[-1].kind == "halt"
        assert actions[-1].args["state"] == "HALTED_BUDGET"

    def test_daily_budget_halts(self):
        stamps = [NOW - i * 5000 for i in range(12)]  # 12 within a day, spread out of the hour window
        record = make_record(recreate_ts=stamps)
        actions = plan(record, obs(node_state="PREEMPTED"), WatchPolicy(max_recreates_per_day=12))
        assert actions[-1].args["state"] == "HALTED_BUDGET"

    def test_waiting_and_provisioning_just_mark(self):
        pending = obs(node_state=None, qr_state="WAITING_FOR_RESOURCES", head_up=None)
        waiting = plan(make_record(), pending, WatchPolicy())
        assert kinds(waiting) == ["set_state"] and waiting[0].args["state"] == "WAITING"
        prov = plan(make_record(), obs(node_state="CREATING", qr_state="ACTIVE", head_up=None), WatchPolicy())
        assert prov[0].args["state"] == "PROVISIONING"

    def test_parked_states_do_nothing(self):
        for state in ("HALTED_QUOTA", "HALTED_BUDGET", "NEEDS_BOOTSTRAP"):
            assert plan(make_record(state=state), obs(node_state="PREEMPTED"), WatchPolicy()) == []

    def test_desired_down_does_nothing(self):
        assert plan(make_record(desired_state="down"), obs(node_state="PREEMPTED"), WatchPolicy()) == []


class TestExecuteActions:
    @pytest.fixture
    def registry(self, tmp_path):
        reg = ClusterRegistry(LocalBackend(tmp_path / "clusters.json"))
        reg.upsert(make_record(state="DEGRADED"))
        return reg

    def _run(self, registry, actions, monkeypatch, *, qr_exists=False, dry_run=False):
        calls = {"delete": [], "create": []}
        monkeypatch.setattr(
            watcher_module,
            "delete_queued_resource",
            lambda qr_id, **k: calls["delete"].append((qr_id, k.get("force"))),
        )
        monkeypatch.setattr(
            watcher_module,
            "create_queued_resource",
            lambda spec, qr_id=None: calls["create"].append(qr_id) or SimpleNamespace(qr_id=qr_id, state="ACCEPTED"),
        )
        monkeypatch.setattr(
            watcher_module,
            "describe_queued_resource",
            lambda qr_id, **k: SimpleNamespace(qr_id=qr_id) if qr_exists else None,
        )
        events = []
        record = registry.get("trainer1")
        watcher_module.execute_actions(
            record,
            actions,
            registry,
            WatchPolicy(),
            dry_run=dry_run,
            emit=lambda e, d="": events.append((e, d)),
        )
        return calls, events

    def test_recovery_sequence_executes_in_order(self, registry, monkeypatch):
        actions = plan(registry.get("trainer1"), obs(node_state="PREEMPTED"), WatchPolicy())
        calls, events = self._run(registry, actions, monkeypatch)
        assert calls["delete"] == [("trainer1", True)]
        assert calls["create"] == ["trainer1-r1"]
        record = registry.get("trainer1")
        assert record.generation == 1
        assert record.qr_id == "trainer1-r1"
        assert record.intent is None
        assert record.state == "WAITING"
        assert len(record.recreate_ts) == 1
        assert [e for e, _ in events][:2] == ["preemption_detected", "qr_delete"]

    def test_create_is_idempotent_on_crash_replay(self, registry, monkeypatch):
        # If the intent's target already exists (crash between create and
        # intent-clear), the watcher adopts it instead of re-creating.
        actions = [Action("create_qr", {"qr_id": "trainer1-r1"})]
        calls, _ = self._run(registry, actions, monkeypatch, qr_exists=True)
        assert calls["create"] == []
        record = registry.get("trainer1")
        assert record.qr_id == "trainer1-r1"
        assert record.intent is None

    def test_dry_run_executes_nothing(self, registry, monkeypatch):
        actions = plan(registry.get("trainer1"), obs(node_state="PREEMPTED"), WatchPolicy())
        calls, events = self._run(registry, actions, monkeypatch, dry_run=True)
        assert calls["delete"] == [] and calls["create"] == []
        assert registry.get("trainer1").generation == 0
        assert all(e == "dry_run" for e, _ in events)

    def test_halt_parks_the_cluster(self, registry, monkeypatch):
        self._run(registry, [Action("halt", {"state": "HALTED_BUDGET"})], monkeypatch)
        assert registry.get("trainer1").state == "HALTED_BUDGET"


class FakeJobsClient:
    def __init__(self, existing=()):
        self.existing = [SimpleNamespace(submission_id=sid) for sid in existing]
        self.submitted = []

    def list_jobs(self):
        return self.existing

    def submit_job(self, **kwargs):
        self.submitted.append(kwargs)
        return kwargs["submission_id"]


class TestResubmitJobs:
    def _record(self, snapshot):
        return make_record(generation=2, job_snapshot=snapshot)

    def _client(self, monkeypatch, existing=()):
        client = FakeJobsClient(existing)
        fake_module = SimpleNamespace(JobSubmissionClient=lambda addr: client)
        monkeypatch.setitem(__import__("sys").modules, "ray.job_submission", fake_module)
        return client

    def snapshot_entry(self, sid="train-abc", *, restartable="1", cwd=None, restart_count=None):
        meta = {"restartable": restartable}
        if cwd is not None:
            meta["cwd"] = cwd
        if restart_count is not None:
            meta["restart_count"] = str(restart_count)
            meta["resume_of"] = sid
        return {"submission_id": sid, "entrypoint": "python train.py", "metadata": meta, "status": "RUNNING"}

    def test_restartable_job_resubmitted_with_contract(self, monkeypatch, tmp_path):
        client = self._client(monkeypatch)
        events = []
        record = self._record([self.snapshot_entry(cwd=str(tmp_path))])
        submitted = resubmit_jobs(record, "10.0.0.5", WatchPolicy(), emit=lambda e, d="": events.append((e, d)))
        assert submitted == ["train-abc-p1"]
        sub = client.submitted[0]
        assert sub["runtime_env"]["working_dir"] == str(tmp_path)
        env = sub["runtime_env"]["env_vars"]
        assert env["ERAY_RESTART_COUNT"] == "1"
        assert env["ERAY_PREEMPTED_FROM"] == "train-abc"
        assert env["ERAY_CLUSTER_GENERATION"] == "2"
        assert sub["metadata"]["resume_of"] == "train-abc"
        assert sub["metadata"]["restart_count"] == "1"

    def test_non_restartable_skipped(self, monkeypatch):
        client = self._client(monkeypatch)
        record = self._record([self.snapshot_entry(restartable="0")])
        assert resubmit_jobs(record, "10.0.0.5", WatchPolicy(), emit=lambda e, d="": None) == []
        assert client.submitted == []

    def test_restart_cap_enforced(self, monkeypatch, tmp_path):
        self._client(monkeypatch)
        record = self._record([self.snapshot_entry("train-abc-p3", cwd=str(tmp_path), restart_count=3)])
        events = []
        submitted = resubmit_jobs(
            record, "10.0.0.5", WatchPolicy(max_restarts_per_job=3), emit=lambda e, d="": events.append(e)
        )
        assert submitted == []
        assert "job_skipped" in events

    def test_increments_across_generations(self, monkeypatch, tmp_path):
        client = self._client(monkeypatch)
        record = self._record([self.snapshot_entry("train-abc-p2", cwd=str(tmp_path), restart_count=2)])
        submitted = resubmit_jobs(record, "10.0.0.5", WatchPolicy(), emit=lambda e, d="": None)
        assert submitted == ["train-abc-p3"]
        assert client.submitted[0]["runtime_env"]["env_vars"]["ERAY_RESTART_COUNT"] == "3"

    def test_missing_cwd_skipped(self, monkeypatch):
        client = self._client(monkeypatch)
        record = self._record([self.snapshot_entry(cwd="/nonexistent/path")])
        events = []
        assert resubmit_jobs(record, "10.0.0.5", WatchPolicy(), emit=lambda e, d="": events.append((e, d))) == []
        assert client.submitted == []
        assert events and events[0][0] == "job_skipped"

    def test_duplicate_resubmission_is_noop(self, monkeypatch, tmp_path):
        client = self._client(monkeypatch, existing=["train-abc-p1"])
        record = self._record([self.snapshot_entry(cwd=str(tmp_path))])
        assert resubmit_jobs(record, "10.0.0.5", WatchPolicy(), emit=lambda e, d="": None) == []
        assert client.submitted == []


class TestWatchLoop:
    @pytest.fixture
    def registry(self, tmp_path, monkeypatch):
        reg = ClusterRegistry(LocalBackend(tmp_path / "clusters.json"))
        monkeypatch.setattr(watcher_module, "EVENTS_PATH", tmp_path / "events.jsonl")
        monkeypatch.setattr(watcher_module, "PAUSE_DIR", tmp_path)
        return reg

    def test_once_ticks_every_cluster(self, registry, monkeypatch):
        registry.upsert(make_record(name="a"))
        registry.upsert(make_record(name="b"))
        observed = []
        monkeypatch.setattr(watcher_module, "observe", lambda r, **k: observed.append(r.name) or obs())
        watch_and_reconnect(once=True, registry=registry)
        assert observed == ["a", "b"]
        assert registry.lease_holder() is None  # released on exit

    def test_paused_cluster_skipped(self, registry, monkeypatch, tmp_path):
        registry.upsert(make_record(name="a"))
        registry.upsert(make_record(name="b"))
        (tmp_path / "pause-a").touch()
        observed = []
        monkeypatch.setattr(watcher_module, "observe", lambda r, **k: observed.append(r.name) or obs())
        watch_and_reconnect(once=True, registry=registry)
        assert observed == ["b"]

    def test_lease_conflict_raises(self, registry, monkeypatch):
        from unittest import mock

        with mock.patch.object(ClusterRegistry, "_holder", return_value="other:1"):
            other = ClusterRegistry(registry.backend)
            assert other.acquire_lease()
        with pytest.raises(RuntimeError, match="lease"):
            watch_and_reconnect(once=True, registry=registry)

    def test_one_cluster_error_does_not_stop_loop(self, registry, monkeypatch, tmp_path):
        registry.upsert(make_record(name="a"))
        registry.upsert(make_record(name="b"))
        seen = []

        def flaky_observe(record, **k):
            seen.append(record.name)
            if record.name == "a":
                raise RuntimeError("boom")
            return obs()

        monkeypatch.setattr(watcher_module, "observe", flaky_observe)
        watch_and_reconnect(once=True, registry=registry)
        assert seen == ["a", "b"]
        events = (tmp_path / "events.jsonl").read_text()
        assert "watch_error" in events

    def test_dry_run_needs_no_lease_and_mutates_nothing(self, registry, monkeypatch):
        registry.upsert(make_record(name="a", state="DEGRADED"))
        monkeypatch.setattr(watcher_module, "observe", lambda r, **k: obs(node_state="PREEMPTED"))
        monkeypatch.setattr(
            watcher_module, "delete_queued_resource", lambda *a, **k: pytest.fail("must not mutate")
        )
        monkeypatch.setattr(
            watcher_module, "create_queued_resource", lambda *a, **k: pytest.fail("must not mutate")
        )
        watch_and_reconnect(once=True, dry_run=True, registry=registry)
        assert registry.get("a").generation == 0
