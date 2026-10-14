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

import time
from types import SimpleNamespace
from unittest import mock

import eray.provision.watcher as watcher_module
import pytest
from eray.provision.registry import ClusterRecord, ClusterRegistry, LocalBackend
from eray.provision.watcher import (
    Action,
    LeaseKeeper,
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
        assert kinds(actions) == ["event", "set_state", "record_recreate", "delete_qr", "create_qr", "set_state"]
        assert actions[0].args["event"] == "preemption_detected"
        assert actions[1].args["state"] == "DEGRADED"
        assert actions[4].args["qr_id"] == "trainer1-r4"
        assert actions[-1].args["state"] == "WAITING"

    def test_recovery_resets_incident_counters(self):
        # The replaced slice's repair_attempted/unreach_ticks must not carry
        # over to generation N+1.
        record = make_record(extra={"unreach_ticks": 5, "repair_attempted": True})
        actions = plan(record, obs(head_up=False), WatchPolicy())
        states = [a for a in actions if a.kind == "set_state"]
        assert [a.args["state"] for a in states] == ["DEGRADED", "WAITING"]
        assert all(a.args.get("reset_incident") for a in states)

    def test_recovery_is_budgeted_before_any_mutation(self):
        # record_recreate precedes delete/create, so a mutation that fails
        # immediately still counts toward the budget.
        actions = kinds(plan(make_record(), obs(node_state="PREEMPTED"), WatchPolicy()))
        assert actions.index("record_recreate") < actions.index("delete_qr") < actions.index("create_qr")

    def test_resumed_quota_failure_is_replaced_once(self):
        failed = obs(node_state=None, qr_state="FAILED", qr_error="quota exceeded")
        acked = make_record(state="UNKNOWN", qr_id="trainer1-r2", generation=2, extra={"resumed_qr": "trainer1-r2"})
        actions = plan(acked, failed, WatchPolicy())
        assert "halt" not in kinds(actions)
        assert next(a for a in actions if a.kind == "create_qr").args["qr_id"] == "trainer1-r3"
        # The acknowledgement covers that QR only: a fresh quota failure halts.
        replaced = make_record(state="WAITING", qr_id="trainer1-r3", generation=3, extra={"resumed_qr": "trainer1-r2"})
        assert plan(replaced, failed, WatchPolicy())[-1].args["state"] == "HALTED_QUOTA"

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

    def test_replacement_slice_is_not_recreated_on_sight(self, registry, monkeypatch):
        # Gen N went dark, the one repair failed → recreate. Gen N+1 then
        # comes up READY before Ray is started on it: it must get its own
        # blip/repair cycle, not inherit "repair already attempted".
        registry.mutate_record("trainer1", lambda r: r.extra.update(unreach_ticks=5, repair_attempted=True))
        actions = plan(registry.get("trainer1"), obs(head_up=False), WatchPolicy())
        assert "create_qr" in kinds(actions)
        self._run(registry, actions, monkeypatch)
        record = registry.get("trainer1")
        assert record.generation == 1 and record.state == "WAITING"
        assert "repair_attempted" not in record.extra and "unreach_ticks" not in record.extra
        policy = WatchPolicy()
        first = plan(record, obs(head_up=False), policy)
        assert kinds(first) == ["set_state"]  # blip count, not a recreate
        self._run(registry, first, monkeypatch)
        second = plan(registry.get("trainer1"), obs(head_up=False), policy)
        assert kinds(second) == ["event", "set_state", "bootstrap", "connect"]  # repair (connect) gen N+1

    def _failing_create(self, monkeypatch, message):
        def boom(spec, qr_id=None):
            raise RuntimeError(message)

        monkeypatch.setattr(watcher_module, "delete_queued_resource", lambda qr_id, **k: None)
        monkeypatch.setattr(watcher_module, "create_queued_resource", boom)
        monkeypatch.setattr(watcher_module, "describe_queued_resource", lambda qr_id, **k: None)

    def test_immediate_create_failure_counts_toward_budget_and_halts(self, registry, monkeypatch):
        self._failing_create(monkeypatch, "gcloud compute tpus failed: INTERNAL error")
        policy = WatchPolicy(max_recreates_per_hour=3)
        gone = obs(node_state=None, qr_state=None, head_up=None)
        for attempt in range(3):
            actions = plan(registry.get("trainer1"), gone, policy)
            assert "create_qr" in kinds(actions), attempt
            with pytest.raises(RuntimeError, match="INTERNAL"):
                watcher_module.execute_actions(registry.get("trainer1"), actions, registry, policy, emit=lambda *a: None)
            record = registry.get("trainer1")
            assert len(record.recreate_ts) == attempt + 1
            assert record.intent is None
        final = plan(registry.get("trainer1"), gone, policy)
        assert kinds(final) == ["event", "halt"]
        assert final[-1].args["state"] == "HALTED_BUDGET"

    def test_quota_create_failure_halts_immediately(self, registry, monkeypatch):
        self._failing_create(monkeypatch, "gcloud compute tpus failed: Quota 'TPUV5P' exceeded")
        events = []
        actions = plan(registry.get("trainer1"), obs(node_state="PREEMPTED"), WatchPolicy())
        watcher_module.execute_actions(
            registry.get("trainer1"), actions, registry, WatchPolicy(), emit=lambda e, d="": events.append(e)
        )
        record = registry.get("trainer1")
        assert record.state == "HALTED_QUOTA"
        assert record.intent is None
        assert "qr_failed_quota" in events

    def test_lost_lease_stops_before_next_action(self, registry, monkeypatch):
        answers = iter([True, True, False])
        actions = plan(registry.get("trainer1"), obs(node_state="PREEMPTED"), WatchPolicy())
        deleted, created, events = [], [], []
        monkeypatch.setattr(watcher_module, "delete_queued_resource", lambda qr_id, **k: deleted.append(qr_id))
        monkeypatch.setattr(watcher_module, "create_queued_resource", lambda *a, **k: created.append(1))
        monkeypatch.setattr(watcher_module, "describe_queued_resource", lambda qr_id, **k: None)
        watcher_module.execute_actions(
            registry.get("trainer1"),
            actions,
            registry,
            WatchPolicy(),
            emit=lambda e, d="": events.append(e),
            lease_ok=lambda: next(answers),
        )
        # event + set_state ran; the lease was gone before record_recreate.
        assert deleted == [] and created == []
        assert registry.get("trainer1").generation == 0
        assert events == ["preemption_detected", "lease_lost"]


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

    def snapshot_entry(
        self, sid="train-abc", *, restartable="1", cwd=None, restart_count=None, working_dir=None, runtime_env=None
    ):
        meta = {"restartable": restartable}
        if cwd is not None:
            meta["cwd"] = cwd
        if working_dir is not None:
            meta["working_dir"] = working_dir
        if restart_count is not None:
            meta["restart_count"] = str(restart_count)
            meta["resume_of"] = sid
        entry = {"submission_id": sid, "entrypoint": "python train.py", "metadata": meta, "status": "RUNNING"}
        if runtime_env is not None:
            entry["runtime_env"] = runtime_env
        return entry

    def test_original_runtime_env_is_preserved(self, monkeypatch, tmp_path):
        client = self._client(monkeypatch)
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        entry = self.snapshot_entry(
            cwd=str(tmp_path),
            working_dir=str(pkg),
            runtime_env={
                "working_dir": "gcs://_ray_pkg_dead.zip",
                "py_modules": ["gcs://_ray_pkg_mod.zip", "s3://bucket/mod.zip"],
                "pip": ["einops"],
                "env_vars": {"HF_TOKEN": "hf_x", "WANDB_PROJECT": "p", "ERAY_RESTART_COUNT": "stale"},
            },
        )
        assert resubmit_jobs(self._record([entry]), "10.0.0.5", WatchPolicy(), emit=lambda e, d="": None) == [
            "train-abc-p1"
        ]
        env = client.submitted[0]["runtime_env"]
        # The packaged dir, not the dead cluster's gcs:// package or the cwd.
        assert env["working_dir"] == str(pkg)
        assert env["py_modules"] == ["s3://bucket/mod.zip"]
        assert env["pip"] == ["einops"]
        assert env["env_vars"]["HF_TOKEN"] == "hf_x"
        assert env["env_vars"]["WANDB_PROJECT"] == "p"
        assert env["env_vars"]["ERAY_RESTART_COUNT"] == "1"  # restart contract wins
        # The snapshot itself is not mutated.
        assert entry["runtime_env"]["env_vars"]["ERAY_RESTART_COUNT"] == "stale"

    def test_no_working_dir_job_is_not_repackaged(self, monkeypatch, tmp_path):
        client = self._client(monkeypatch)
        record = self._record([self.snapshot_entry(cwd=str(tmp_path), working_dir="")])
        assert resubmit_jobs(record, "10.0.0.5", WatchPolicy(), emit=lambda e, d="": None) == ["train-abc-p1"]
        assert "working_dir" not in client.submitted[0]["runtime_env"]

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
        monkeypatch.setattr(watcher_module, "delete_queued_resource", lambda *a, **k: pytest.fail("must not mutate"))
        monkeypatch.setattr(watcher_module, "create_queued_resource", lambda *a, **k: pytest.fail("must not mutate"))
        watch_and_reconnect(once=True, dry_run=True, registry=registry)
        assert registry.get("a").generation == 0


class TestListJobs:
    def _module(self, monkeypatch, client_factory):
        fake_module = SimpleNamespace(JobSubmissionClient=client_factory)
        monkeypatch.setitem(__import__("sys").modules, "ray.job_submission", fake_module)

    def test_api_failure_is_none_not_empty(self, monkeypatch):
        def broken(addr):
            raise ConnectionError("dashboard down")

        self._module(monkeypatch, broken)
        assert watcher_module._list_jobs("10.0.0.5") is None
        # ... so plan() keeps the last good snapshot instead of wiping it.
        assert "snapshot_jobs" not in kinds(plan(make_record(), obs(jobs=None), WatchPolicy()))

    def test_snapshot_captures_runtime_env(self, monkeypatch):
        job = SimpleNamespace(
            submission_id="j1",
            entrypoint="python t.py",
            metadata={"restartable": "1"},
            runtime_env={"env_vars": {"HF_TOKEN": "x"}},
            status="RUNNING",
        )
        done = SimpleNamespace(submission_id="j0", entrypoint="x", metadata={}, runtime_env=None, status="SUCCEEDED")
        self._module(monkeypatch, lambda addr: SimpleNamespace(list_jobs=lambda: [job, done]))
        jobs = watcher_module._list_jobs("10.0.0.5")
        assert [j["submission_id"] for j in jobs] == ["j1"]
        assert jobs[0]["runtime_env"] == {"env_vars": {"HF_TOKEN": "x"}}


class TestLeaseKeeper:
    @pytest.fixture
    def registry(self, tmp_path):
        return ClusterRegistry(LocalBackend(tmp_path / "clusters.json"))

    def test_background_renewal_outlives_the_ttl(self, registry):
        # A long action (QR delete: up to 10 min) must not let the lease lapse.
        assert registry.acquire_lease(ttl=1.0)
        keeper = LeaseKeeper(registry, ttl=1.0, interval=0.1)
        keeper.start()
        try:
            time.sleep(2.0)  # "long operation", 2x the TTL
            assert keeper.held()
            assert registry.lease_holder() == ClusterRegistry._holder()
        finally:
            keeper.stop()

    def test_other_holder_marks_lost(self, registry):
        assert registry.acquire_lease()
        keeper = LeaseKeeper(registry)
        assert keeper.held()
        registry.backend.update(lambda doc: doc.update(lease={"holder": "other:1", "expires": time.time() + 600}))
        assert keeper.renew() is False
        assert not keeper.held()

    def test_transient_errors_expire_the_margin(self, registry, monkeypatch):
        assert registry.acquire_lease()
        keeper = LeaseKeeper(registry, ttl=120, interval=30)
        monkeypatch.setattr(registry, "acquire_lease", mock.Mock(side_effect=RuntimeError("gcs 503")))
        assert keeper.renew() is True  # recent success still inside the margin
        keeper._last_ok -= 100  # renewals kept failing for 100s of a 120s lease
        assert keeper.renew() is False


class TestWatchLoopLease:
    @pytest.fixture
    def registry(self, tmp_path, monkeypatch):
        reg = ClusterRegistry(LocalBackend(tmp_path / "clusters.json"))
        monkeypatch.setattr(watcher_module, "EVENTS_PATH", tmp_path / "events.jsonl")
        monkeypatch.setattr(watcher_module, "PAUSE_DIR", tmp_path)
        return reg

    @staticmethod
    def _steal(registry):
        registry.backend.update(lambda doc: doc.update(lease={"holder": "other:1", "expires": time.time() + 600}))

    def test_lease_lost_mid_tick_stops_acting(self, registry, monkeypatch):
        class EagerKeeper(LeaseKeeper):
            def held(self):  # as if the renewal thread had just run
                if not self.registry.acquire_lease(ttl=self.ttl):
                    self._lost.set()
                return super().held()

        monkeypatch.setattr(watcher_module, "LeaseKeeper", EagerKeeper)
        registry.upsert(make_record(name="a", state="UNKNOWN"))
        registry.upsert(make_record(name="b"))
        observed = []

        def stealing_observe(record, **k):
            observed.append(record.name)
            self._steal(registry)  # another watcher took over during a's long step
            return obs()

        monkeypatch.setattr(watcher_module, "observe", stealing_observe)
        with pytest.raises(RuntimeError, match="lost the fleet lease"):
            watch_and_reconnect(once=True, registry=registry)
        assert observed == ["a"]
        assert registry.get("a").state == "UNKNOWN"  # a's planned actions were gated, none ran
        assert "lease_lost" in (registry.backend.path.parent / "events.jsonl").read_text()
        assert registry.lease_holder() == "other:1"  # never released someone else's lease

    def test_heartbeat_result_is_checked(self, registry, monkeypatch):
        registry.upsert(make_record(name="a"))
        monkeypatch.setattr(watcher_module, "observe", lambda r, **k: self._steal(registry) or obs())
        monkeypatch.setattr(watcher_module.time, "sleep", lambda s: pytest.fail("must not keep looping"))
        with pytest.raises(RuntimeError, match="lost the fleet lease"):
            watch_and_reconnect(registry=registry)
