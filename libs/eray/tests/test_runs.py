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

"""Tests for eray.runs: the planner truth table, budgets and quarantine,
preemption-vs-failure separation, the store and lease, the manager with a
fake Jobs API, and the CLI."""

from __future__ import annotations

import time

import eray.cli.runs as cli_runs
import eray.runs.store as store_module
import pytest
from click.testing import CliRunner
from eray.capacity.fake import FakeClock
from eray.runs.manager import RunsManager
from eray.runs.model import (
    LAUNCH_VISIBILITY_TIMEOUT_S,
    HealthPolicy,
    RunObservation,
    RunRecord,
    RunSpec,
    RunState,
    plan_run,
    reset_for_retry,
)
from eray.runs.store import Lease, LeaseHeldError, RunStore

NOW = 1_700_000_000.0


def make_record(**overrides) -> RunRecord:
    spec_fields = {
        "name": overrides.pop("name", "arm1"),
        "cluster": overrides.pop("cluster", "http://10.0.0.1:8265"),
        "entrypoint": overrides.pop("entrypoint", "python train.py"),
        "max_failures": overrides.pop("max_failures", 10),
        "max_preemptions": overrides.pop("max_preemptions", 1000),
        "health": overrides.pop(
            "health", HealthPolicy(compile_grace_s=600.0, step_timeout_s=120.0, max_futile=3)
        ),
    }
    record = RunRecord(spec=RunSpec(**spec_fields))
    for key, value in overrides.items():
        setattr(record, key, value)
    return record


def obs(**overrides) -> RunObservation:
    defaults = dict(cluster_ok=True, job_status="RUNNING", latest_step=None, now=NOW)
    defaults.update(overrides)
    return RunObservation(**defaults)


def kinds(actions) -> list[str]:
    return [a.kind for a in actions]


def final_state(record, actions) -> RunState | None:
    for action in reversed(actions):
        if action.kind == "set" and "state" in action.changes:
            return action.changes["state"]
    return None


def running_record(**overrides) -> RunRecord:
    defaults = dict(
        state=RunState.RUNNING,
        attempt=1,
        submission_id="arm1-a1",
        attempt_started=NOW - 1000.0,
        last_step_ts=NOW - 30.0,
    )
    defaults.update(overrides)
    return make_record(**defaults)


# ============================================================================
# Planner truth table
# ============================================================================


class TestPlanLaunchPaths:
    def test_pending_with_cluster_launches_first_attempt(self):
        actions = plan_run(make_record(), obs(job_status=None))
        assert kinds(actions) == ["launch", "set"]
        assert actions[0].submission_id == "arm1-a1"
        assert actions[1].changes["state"] is RunState.LAUNCHING
        assert actions[1].changes["attempt"] == 1
        assert actions[1].changes["start_step"] == 0

    def test_pending_without_cluster_waits_as_preempted(self):
        actions = plan_run(make_record(), obs(cluster_ok=False, job_status=None, cluster_note="head down"))
        assert final_state(make_record(), actions) is RunState.PREEMPTED

    def test_preempted_waits_quietly_while_cluster_down(self):
        record = make_record(state=RunState.PREEMPTED)
        assert plan_run(record, obs(cluster_ok=False, job_status=None)) == []

    def test_preempted_relaunches_when_cluster_returns(self):
        record = make_record(state=RunState.PREEMPTED, attempt=3, last_step=500)
        actions = plan_run(record, obs(job_status=None))
        assert actions[0].submission_id == "arm1-a4"
        assert actions[1].changes["start_step"] == 500  # bar to beat carries over

    def test_launching_becomes_running_when_job_visible(self):
        record = make_record(state=RunState.LAUNCHING, submission_id="arm1-a1", attempt=1, attempt_started=NOW - 5)
        actions = plan_run(record, obs(job_status="RUNNING"))
        assert final_state(record, actions) is RunState.RUNNING

    def test_launching_tolerates_brief_invisibility(self):
        record = make_record(state=RunState.LAUNCHING, submission_id="arm1-a1", attempt_started=NOW - 10)
        assert plan_run(record, obs(job_status=None)) == []

    def test_launching_invisible_too_long_is_a_failure(self):
        record = make_record(
            state=RunState.LAUNCHING, submission_id="arm1-a1",
            attempt_started=NOW - LAUNCH_VISIBILITY_TIMEOUT_S - 1,
        )
        actions = plan_run(record, obs(job_status=None))
        assert final_state(record, actions) is RunState.PENDING
        assert actions[-1].changes["failures"] == 1

    def test_pending_with_live_submission_readopts_instead_of_relaunching(self):
        # A transient Jobs-API blip parks the run PREEMPTED; when the API
        # answers again the old attempt is often still RUNNING. Launching
        # attempt N+1 would put two trainers on the same checkpoints.
        for state in (RunState.PENDING, RunState.PREEMPTED):
            record = make_record(state=state, attempt=1, submission_id="arm1-a1")
            actions = plan_run(record, obs(job_status="RUNNING"))
            assert "launch" not in kinds(actions)
            assert final_state(record, actions) is RunState.RUNNING

    def test_pending_with_dead_submission_still_relaunches(self):
        record = make_record(state=RunState.PREEMPTED, attempt=1, submission_id="arm1-a1")
        actions = plan_run(record, obs(job_status="FAILED"))
        assert actions[0].kind == "launch"
        assert actions[0].submission_id == "arm1-a2"

    def test_launching_stopped_is_killed_not_retried(self):
        record = make_record(state=RunState.LAUNCHING, submission_id="arm1-a1", attempt_started=NOW - 5)
        actions = plan_run(record, obs(job_status="STOPPED"))
        assert final_state(record, actions) is RunState.KILLED
        assert "launch" not in kinds(actions)

    def test_launching_cluster_lost_is_preemption(self):
        record = make_record(state=RunState.LAUNCHING, submission_id="arm1-a1", attempt_started=NOW - 5)
        actions = plan_run(record, obs(cluster_ok=False, job_status="FAILED"))
        assert final_state(record, actions) is RunState.PREEMPTED
        assert actions[-1].changes["preemptions"] == 1


class TestPlanRunningPaths:
    def test_succeeded(self):
        actions = plan_run(running_record(), obs(job_status="SUCCEEDED"))
        assert final_state(None, actions) is RunState.SUCCEEDED

    def test_hollow_success_below_min_steps_is_a_failure(self):
        # A driver that swallows its exception and exits 0 reports
        # SUCCEEDED at step 0 — with a min-steps floor that is a failure.
        record = running_record(
            health=HealthPolicy(compile_grace_s=600.0, step_timeout_s=120.0, max_futile=3, min_steps_for_success=1)
        )
        actions = plan_run(record, obs(job_status="SUCCEEDED", latest_step=None))
        assert final_state(None, actions) is RunState.PENDING
        assert actions[-1].changes["failures"] == 1
        assert actions[-1].changes["futile"] == 1

    def test_success_at_or_above_min_steps_stands(self):
        record = running_record(
            last_step=1500,
            health=HealthPolicy(compile_grace_s=600.0, step_timeout_s=120.0, max_futile=3, min_steps_for_success=100),
        )
        actions = plan_run(record, obs(job_status="SUCCEEDED"))
        assert final_state(None, actions) is RunState.SUCCEEDED

    def test_hollow_success_counts_dying_log_steps(self):
        # Steps visible only in the final logs still clear the floor.
        record = running_record(
            health=HealthPolicy(compile_grace_s=600.0, step_timeout_s=120.0, max_futile=3, min_steps_for_success=5)
        )
        actions = plan_run(record, obs(job_status="SUCCEEDED", latest_step=9))
        assert final_state(None, actions) is RunState.SUCCEEDED

    def test_external_stop_is_killed_not_restarted(self):
        actions = plan_run(running_record(), obs(job_status="STOPPED"))
        assert final_state(None, actions) is RunState.KILLED
        assert "launch" not in kinds(actions)

    def test_job_failed_with_healthy_cluster_retries_and_counts_failure(self):
        actions = plan_run(running_record(), obs(job_status="FAILED"))
        assert final_state(None, actions) is RunState.PENDING
        assert actions[-1].changes["failures"] == 1
        assert actions[-1].changes["futile"] == 1  # no progress this attempt

    def test_job_failed_after_progress_resets_futile(self):
        record = running_record(start_step=100, last_step=250, futile=2)
        actions = plan_run(record, obs(job_status="FAILED"))
        assert actions[-1].changes["futile"] == 0
        assert actions[-1].changes["failures"] == 1

    def test_fast_crash_between_ticks_credits_final_log_steps(self):
        # The job crashed before any RUNNING tick recorded a step, but its
        # dying logs show progress — that must not count as futile.
        record = running_record(start_step=0, last_step=0, futile=1)
        actions = plan_run(record, obs(job_status="FAILED", latest_step=5))
        assert actions[-1].changes["futile"] == 0
        assert actions[-1].changes["last_step"] == 5

    def test_job_vanished_with_healthy_cluster_is_failure(self):
        actions = plan_run(running_record(), obs(job_status=None))
        assert final_state(None, actions) is RunState.PENDING
        assert actions[-1].changes["failures"] == 1

    def test_job_dead_with_dead_cluster_is_preemption_not_failure(self):
        actions = plan_run(running_record(), obs(cluster_ok=False, job_status=None, cluster_note="preempted"))
        assert final_state(None, actions) is RunState.PREEMPTED
        assert actions[-1].changes["preemptions"] == 1
        assert "failures" not in actions[-1].changes

    def test_futile_ladder_reaches_quarantine(self):
        record = running_record(futile=2)  # max_futile=3 → this failure is the third
        actions = plan_run(record, obs(job_status="FAILED"))
        assert final_state(None, actions) is RunState.QUARANTINED

    def test_failure_budget_exhaustion_fails_the_run(self):
        record = running_record(failures=10, start_step=0, last_step=50)  # progress → no quarantine
        actions = plan_run(record, obs(job_status="FAILED"))
        assert final_state(None, actions) is RunState.FAILED

    def test_preemption_budget_exhaustion_fails_the_run(self):
        record = running_record(max_preemptions=2, preemptions=2)
        actions = plan_run(record, obs(cluster_ok=False, job_status=None))
        assert final_state(None, actions) is RunState.FAILED

    def test_step_progress_is_recorded(self):
        actions = plan_run(running_record(last_step=10), obs(latest_step=42))
        assert kinds(actions) == ["set"]
        assert actions[0].changes["last_step"] == 42

    def test_progress_past_start_step_ends_futile_streak(self):
        record = running_record(start_step=10, last_step=10, futile=2)
        actions = plan_run(record, obs(latest_step=11))
        assert actions[0].changes["futile"] == 0

    def test_no_progress_inside_compile_grace_is_fine(self):
        record = running_record(attempt_started=NOW - 100.0, last_step_ts=0.0)
        assert plan_run(record, obs(latest_step=None)) == []

    def test_stall_past_grace_and_timeout_stops_and_retries(self):
        record = running_record(attempt_started=NOW - 1000.0, last_step_ts=NOW - 200.0)
        actions = plan_run(record, obs(latest_step=None))
        assert kinds(actions)[0] == "stop_job"
        assert final_state(None, actions) is RunState.PENDING
        assert actions[-1].changes["failures"] == 1

    def test_recent_step_within_timeout_is_fine(self):
        record = running_record(attempt_started=NOW - 1000.0, last_step_ts=NOW - 60.0)
        assert plan_run(record, obs(latest_step=None)) == []

    def test_attempt_restarting_below_lifetime_max_is_not_stall_killed(self):
        # Lost checkpoint: attempt 2 restarts at step 1 while last_step=500.
        # Per-attempt progress must refresh the heartbeat.
        record = running_record(
            attempt=2, start_step=500, last_step=500, attempt_step=0,
            attempt_started=NOW - 1000.0, last_step_ts=NOW - 900.0,
        )
        actions = plan_run(record, obs(latest_step=3))
        assert kinds(actions) == ["set"]
        assert actions[0].changes["attempt_step"] == 3
        assert actions[0].changes["last_step"] == 500  # lifetime max unchanged
        assert actions[0].changes["last_step_ts"] == NOW  # heartbeat refreshed

    def test_invalid_step_regex_rejected_at_spec_construction(self):
        with pytest.raises(ValueError, match="step_regex"):
            make_record(health=HealthPolicy(step_regex="step ([0-9"))

    def test_terminal_states_plan_nothing(self):
        for state in (RunState.SUCCEEDED, RunState.FAILED, RunState.KILLED, RunState.QUARANTINED):
            assert plan_run(make_record(state=state), obs()) == []


class TestModelRoundtrip:
    def test_record_roundtrip_with_unknown_keys(self):
        record = running_record(last_step=77, failures=2)
        record.log("progress", "step 77", now=NOW)
        data = record.to_dict()
        data["future_field"] = {"x": 1}
        data["spec"]["future_spec_field"] = True
        loaded = RunRecord.from_dict(data)
        assert loaded.spec == record.spec
        assert loaded.state is RunState.RUNNING
        assert loaded.last_step == 77
        assert loaded.history[-1][1] == "progress"

    def test_reset_for_retry_clears_budgets_keeps_steps(self):
        record = running_record(state=RunState.QUARANTINED, failures=7, futile=3, last_step=900)
        fresh = reset_for_retry(record)
        assert fresh.state is RunState.PENDING
        assert fresh.failures == 0 and fresh.futile == 0
        assert fresh.last_step == 900


# ============================================================================
# Store and lease
# ============================================================================


@pytest.fixture(autouse=True)
def _isolate_runs_state(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "STATE_PATH", tmp_path / "runs.json")
    monkeypatch.setattr(store_module, "LEASE_PATH", tmp_path / "runs.lease")


class TestStoreAndLease:
    def test_store_roundtrip(self):
        store = RunStore()
        store.upsert(running_record(last_step=5))
        loaded = store.load()["arm1"]
        assert loaded.state is RunState.RUNNING
        assert loaded.last_step == 5
        assert store.remove("arm1") is True
        assert store.remove("arm1") is False

    def test_lease_excludes_second_watcher(self):
        first = Lease()
        first.acquire()
        with pytest.raises(LeaseHeldError):
            Lease().acquire()
        first.release()
        Lease().acquire()  # released → free

    def test_expired_lease_is_stolen_automatically(self):
        stale = Lease(ttl_s=0.001)
        stale.acquire()
        time.sleep(0.01)
        Lease().acquire()  # expired → no raise

    def test_steal_flag_overrides_live_holder(self):
        Lease().acquire()
        Lease().acquire(steal=True)


# ============================================================================
# Manager with a fake Jobs API
# ============================================================================


class FakeJobsClient:
    """Minimal JobSubmissionClient stand-in shared across addresses."""

    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self.stopped: list[str] = []

    def submit_job(self, *, entrypoint, submission_id, runtime_env=None, metadata=None):
        if submission_id in self.jobs:
            raise RuntimeError(f"Job with submission_id {submission_id} already exists")
        self.jobs[submission_id] = {
            "status": "RUNNING",
            "logs": "",
            "entrypoint": entrypoint,
            "runtime_env": runtime_env or {},
        }
        return submission_id

    def get_job_status(self, submission_id):
        if submission_id not in self.jobs:
            raise RuntimeError(f"Job {submission_id} does not exist")
        return self.jobs[submission_id]["status"]

    def get_job_logs(self, submission_id):
        if submission_id not in self.jobs:
            raise RuntimeError(f"Job {submission_id} does not exist")
        return self.jobs[submission_id]["logs"]

    def stop_job(self, submission_id):
        self.stopped.append(submission_id)
        self.jobs[submission_id]["status"] = "STOPPED"


def make_manager(clock=None):
    clock = clock or FakeClock(NOW)
    fake = FakeJobsClient()
    manager = RunsManager(
        clock=clock,
        jobs_client_factory=lambda address: fake,
        address_resolver=lambda cluster: ("http://fake:8265", ""),
    )
    return manager, fake, clock


class TestManager:
    def test_full_lifecycle_launch_progress_crash_relaunch(self):
        manager, fake, _clock = make_manager()
        manager.store.upsert(make_record())

        records = manager.reconcile_once()  # PENDING → launch
        assert "arm1-a1" in fake.jobs
        assert records["arm1"].state is RunState.LAUNCHING

        records = manager.reconcile_once()  # visible → RUNNING
        assert records["arm1"].state is RunState.RUNNING

        fake.jobs["arm1-a1"]["logs"] = "compiling...\nstep 12 loss 1.0\nstep 13 loss 0.9\n"
        records = manager.reconcile_once()
        assert records["arm1"].last_step == 13

        fake.jobs["arm1-a1"]["status"] = "FAILED"
        records = manager.reconcile_once()
        assert records["arm1"].state is RunState.PENDING
        assert records["arm1"].failures == 1
        assert records["arm1"].futile == 0  # it had made progress

        records = manager.reconcile_once()  # relaunch as attempt 2
        assert "arm1-a2" in fake.jobs
        assert records["arm1"].state is RunState.LAUNCHING
        assert records["arm1"].start_step == 13

    def test_stall_is_stopped_and_retried(self):
        manager, fake, clock = make_manager()
        record = make_record(health=HealthPolicy(compile_grace_s=10.0, step_timeout_s=20.0, max_futile=3))
        manager.store.upsert(record)
        manager.reconcile_once()
        manager.reconcile_once()  # RUNNING, silent logs
        clock.advance(120.0)  # past grace + step timeout
        records = manager.reconcile_once()
        assert fake.stopped == ["arm1-a1"]
        assert records["arm1"].state is RunState.PENDING
        assert records["arm1"].failures == 1

    def test_duplicate_submission_is_adopted(self):
        manager, fake, _clock = make_manager()
        fake.jobs["arm1-a1"] = {"status": "RUNNING", "logs": "", "entrypoint": "x", "runtime_env": {}}
        manager.store.upsert(make_record())
        records = manager.reconcile_once()
        assert records["arm1"].state is RunState.LAUNCHING
        assert records["arm1"].submission_id == "arm1-a1"
        records = manager.reconcile_once()
        assert records["arm1"].state is RunState.RUNNING

    def test_jobs_api_down_is_preemption_not_failure(self):
        clock = FakeClock(NOW)

        class ExplodingClient:
            def get_job_status(self, _):
                raise ConnectionError("connection refused")

        manager = RunsManager(
            clock=clock,
            jobs_client_factory=lambda address: ExplodingClient(),
            address_resolver=lambda cluster: ("http://fake:8265", ""),
        )
        manager.store.upsert(running_record())
        records = manager.reconcile_once()
        assert records["arm1"].state is RunState.PREEMPTED
        assert records["arm1"].failures == 0

    def test_persistent_submit_failure_consumes_budgets(self):
        # A submit that always raises (e.g. workdir over the upload cap)
        # must not loop forever at zero budget: the set-action still lands,
        # and the visibility timeout meters it into the failure budget.
        clock = FakeClock(NOW)

        class RejectingClient(FakeJobsClient):
            def submit_job(self, **kwargs):
                raise RuntimeError("Job payload exceeds upload limit")

        fake = RejectingClient()
        manager = RunsManager(
            clock=clock,
            jobs_client_factory=lambda address: fake,
            address_resolver=lambda cluster: ("http://fake:8265", ""),
        )
        manager.store.upsert(make_record(max_failures=1, health=HealthPolicy(max_futile=99)))
        records = manager.reconcile_once()
        assert records["arm1"].state is RunState.LAUNCHING
        assert records["arm1"].attempt == 1
        clock.advance(LAUNCH_VISIBILITY_TIMEOUT_S + 1)
        records = manager.reconcile_once()
        assert records["arm1"].failures == 1
        clock.advance(1)
        records = manager.reconcile_once()  # relaunch attempt 2 (also fails)
        clock.advance(LAUNCH_VISIBILITY_TIMEOUT_S + 1)
        records = manager.reconcile_once()
        assert records["arm1"].state is RunState.FAILED  # budget exhausted, no infinite loop

    def test_watch_dry_run_does_not_take_the_lease(self):
        manager, _fake, _clock = make_manager()
        manager.dry_run = True
        Lease().acquire()  # a live real watcher
        manager.watch(once=True)  # must not raise LeaseHeldError

    def test_mid_pass_cli_write_to_other_record_survives(self):
        # The watcher persists per record; a CLI upsert to record B while
        # the pass reconciles record A must not be clobbered.
        manager, fake, _clock = make_manager()
        manager.store.upsert(make_record(name="arm1"))

        cli_written = make_record(name="armz", state=RunState.KILLED)
        original_submit = fake.submit_job

        def submit_and_interleave(**kwargs):
            manager.store.upsert(cli_written)  # CLI write lands mid-pass
            return original_submit(**kwargs)

        fake.submit_job = submit_and_interleave
        manager.reconcile_once()
        persisted = manager.store.load()
        assert persisted["armz"].state is RunState.KILLED  # not clobbered
        assert persisted["arm1"].state is RunState.LAUNCHING

    def test_dry_run_reports_without_acting_or_persisting(self):
        manager, fake, _clock = make_manager()
        manager.store.upsert(make_record())
        manager.dry_run = True
        manager.reconcile_once()
        assert fake.jobs == {}
        assert manager.store.load()["arm1"].state is RunState.PENDING

    def test_watch_once_holds_and_releases_lease(self):
        manager, _fake, _clock = make_manager()
        manager.store.upsert(make_record())
        manager.watch(once=True)
        Lease().acquire()  # released after watch → acquirable

    def test_watch_refuses_second_instance(self):
        manager, _fake, _clock = make_manager()
        Lease().acquire()  # simulate a live watcher (same pid → alive)
        with pytest.raises(LeaseHeldError):
            manager.watch(once=True)


# ============================================================================
# CLI
# ============================================================================


@pytest.fixture()
def cli_env(monkeypatch):
    fake = FakeJobsClient()

    def fake_manager(**kwargs):
        return RunsManager(
            jobs_client_factory=lambda address: fake,
            address_resolver=lambda cluster: ("http://fake:8265", ""),
            **kwargs,
        )

    monkeypatch.setattr(cli_runs, "RunsManager", fake_manager)
    monkeypatch.setattr(cli_runs, "resolve_jobs_address", lambda cluster: ("http://fake:8265", ""))
    from eray.cli.main import cli

    return CliRunner(), cli, fake


class TestCli:
    def test_add_watch_once_and_list(self, cli_env):
        runner, cli, fake = cli_env
        result = runner.invoke(
            cli, ["runs", "add", "arm1", "-c", "http://10.0.0.1:8265", "-x", "python train.py"]
        )
        assert result.exit_code == 0, result.output
        result = runner.invoke(cli, ["runs", "watch", "--once"])
        assert result.exit_code == 0, result.output
        assert "arm1-a1" in fake.jobs
        result = runner.invoke(cli, ["runs", "list"])
        assert result.exit_code == 0
        assert "LAUNCHING" in result.output

    def test_add_refuses_live_duplicate_without_force(self, cli_env):
        runner, cli, _ = cli_env
        runner.invoke(cli, ["runs", "add", "arm1", "-c", "x", "-x", "python t.py"])
        result = runner.invoke(cli, ["runs", "add", "arm1", "-c", "x", "-x", "python t.py"])
        assert result.exit_code != 0
        result = runner.invoke(cli, ["runs", "add", "arm1", "-c", "x", "-x", "python t.py", "--force"])
        assert result.exit_code == 0

    def test_stop_then_retry_rearms(self, cli_env):
        runner, cli, fake = cli_env
        runner.invoke(cli, ["runs", "add", "arm1", "-c", "x", "-x", "python t.py"])
        runner.invoke(cli, ["runs", "watch", "--once"])
        result = runner.invoke(cli, ["runs", "stop", "arm1"])
        assert result.exit_code == 0, result.output
        assert fake.stopped == ["arm1-a1"]
        assert RunStore().load()["arm1"].state is RunState.KILLED
        result = runner.invoke(cli, ["runs", "retry", "arm1"])
        assert result.exit_code == 0
        assert RunStore().load()["arm1"].state is RunState.PENDING

    def test_retry_refuses_non_terminal(self, cli_env):
        runner, cli, _ = cli_env
        runner.invoke(cli, ["runs", "add", "arm1", "-c", "x", "-x", "python t.py"])
        runner.invoke(cli, ["runs", "watch", "--once"])  # LAUNCHING now
        result = runner.invoke(cli, ["runs", "retry", "arm1"])
        assert result.exit_code != 0
        assert "not terminal" in result.output

    def test_add_force_stops_live_job_and_keeps_attempt_numbering(self, cli_env):
        runner, cli, fake = cli_env
        runner.invoke(cli, ["runs", "add", "arm1", "-c", "x", "-x", "python old.py"])
        runner.invoke(cli, ["runs", "watch", "--once"])  # launches arm1-a1
        result = runner.invoke(cli, ["runs", "add", "arm1", "-c", "x", "-x", "python new.py", "--force"])
        assert result.exit_code == 0, result.output
        assert fake.stopped == ["arm1-a1"]  # old attempt stopped
        runner.invoke(cli, ["runs", "watch", "--once"])
        assert "arm1-a2" in fake.jobs  # fresh id, no stale adoption
        assert fake.jobs["arm1-a2"]["entrypoint"] == "python new.py"

    def test_rm_requires_terminal_or_force(self, cli_env):
        runner, cli, _ = cli_env
        runner.invoke(cli, ["runs", "add", "arm1", "-c", "x", "-x", "python t.py"])
        result = runner.invoke(cli, ["runs", "rm", "arm1"])
        assert result.exit_code != 0
        result = runner.invoke(cli, ["runs", "rm", "arm1", "--force"])
        assert result.exit_code == 0
        assert RunStore().load() == {}

    def test_status_shows_history(self, cli_env):
        runner, cli, _ = cli_env
        runner.invoke(cli, ["runs", "add", "arm1", "-c", "x", "-x", "python t.py"])
        runner.invoke(cli, ["runs", "watch", "--once"])
        result = runner.invoke(cli, ["runs", "status", "arm1"])
        assert result.exit_code == 0
        assert "launch" in result.output


class TestFleetBindingResolution:
    """resolve_jobs_address's fleet path must construct a real registry —
    the fake-resolver seam in the manager tests never exercises it."""

    def test_fleet_binding_resolves_head_address(self, tmp_path, monkeypatch):
        import eray.runs.manager as manager_module
        from eray.provision.registry import ClusterRecord, ClusterRegistry, LocalBackend

        reg = ClusterRegistry(LocalBackend(tmp_path / "clusters.json"))
        reg.upsert(
            ClusterRecord(
                name="trainer1", project="p", zone="z", accelerator_type="v5p-8",
                head_ip="10.0.0.5", state="HEALTHY",
            )
        )
        monkeypatch.setattr(ClusterRegistry, "from_config", classmethod(lambda cls: reg))
        monkeypatch.setattr(manager_module, "head_reachable", lambda ip: True)
        address, note = manager_module.resolve_jobs_address("fleet:trainer1")
        assert address == "http://10.0.0.5:8265"
        assert note == ""

    def test_fleet_binding_reports_missing_and_unreachable(self, tmp_path, monkeypatch):
        import eray.runs.manager as manager_module
        from eray.provision.registry import ClusterRecord, ClusterRegistry, LocalBackend

        reg = ClusterRegistry(LocalBackend(tmp_path / "clusters.json"))
        monkeypatch.setattr(ClusterRegistry, "from_config", classmethod(lambda cls: reg))
        address, note = manager_module.resolve_jobs_address("fleet:ghost")
        assert address is None and "not in registry" in note

        reg.upsert(ClusterRecord(name="dark", project="p", zone="z", accelerator_type="v5p-8", head_ip="10.0.0.9"))
        monkeypatch.setattr(manager_module, "head_reachable", lambda ip: False)
        address, note = manager_module.resolve_jobs_address("fleet:dark")
        assert address is None and "unreachable" in note
