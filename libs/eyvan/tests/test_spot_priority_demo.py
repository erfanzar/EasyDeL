# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The operator demo plans bounded owned Spot work before any remote mutation."""

import runpy
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "spot_priority_demo.py"


def _module():
    assert SCRIPT.is_file(), "the reusable Spot operator workflow is not implemented"
    return runpy.run_path(str(SCRIPT))


def _advancing_clock(monkeypatch):
    import time

    clock = [time.monotonic()]

    def sleep(seconds):
        assert seconds > 0
        clock[0] += seconds

    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", sleep)
    return clock


def _arguments(**changes):
    return (
        dict(
            url="http://127.0.0.1:18181",
            org="research",
            project="experiment",
            gcp_project="demo-project-123",
            zone="us-east5-a",
            agent_url="http://127.0.0.1:8090",
            artifact_prefix="gs://example-bucket/priority-demo",
            run_id="a" * 32,
            timeout=2400,
        )
        | changes
    )


def test_plan_is_one_owned_spot_pool_and_zero_initial_allocation() -> None:
    plan = _module()["build_plan"](**_arguments())
    assert plan["fleet_name"] == "eyvan-priority-" + "a" * 32
    assert plan["checkpoint_uri"] == "gs://example-bucket/priority-demo/" + "a" * 32 + "/checkpoint.json"
    assert plan["url"] == "http://127.0.0.1:18181"
    assert plan["timeout"] == 2400
    (pool,) = plan["fleet_request"]["pools"]
    assert pool["capacity_class"] == "spot" and pool["provider_project"] == "demo-project-123"
    assert pool["max_cells"] == 1 and pool["min_cells"] == pool["initial_cells"] == 0
    assert pool["allocatable"]["tpu"] == 4
    assert pool["executor"] == "eyvan-native"


def test_workload_resources_are_bound_to_the_compiled_four_chip_pool():
    module = _module()
    plan = module["build_plan"](**_arguments())
    plan["fleet_request"]["pools"][0]["allocatable"]["tpu"] = 8
    with pytest.raises(ValueError):
        module["workload_request"](plan, "batch", "print('payload')")


@pytest.mark.parametrize(
    "changes",
    [
        {"url": "https://user:password@example.test"},  # pragma: allowlist secret -- rejected dummy credentials
        {"url": "https://example.test/?token=private"},
        {"artifact_prefix": "gs://user:password@bucket/prefix"},  # pragma: allowlist secret -- dummy credentials
        {"artifact_prefix": "gs://bucket/prefix?token=private"},
        {"artifact_prefix": "gs://bucket"},
        {"artifact_prefix": "https://bucket/prefix"},
        {"artifact_prefix": "gs://bucket/pre\nfix"},
        {"artifact_prefix": "gs://bucket/pre fix"},
        {"artifact_prefix": "gs://bucket/prefix//child"},
        {"url": "http://127.0.0.1:\n8080"},
        {"run_id": "../foreign"},
        {"gcp_project": ""},
        {"zone": "us-east5"},
        {"timeout": float("nan")},
        {"timeout": 0},
        {"timeout": 100000},
    ],
)
def test_plan_rejects_ambiguous_targets_and_unbounded_runs(changes) -> None:
    with pytest.raises(ValueError):
        _module()["build_plan"](**_arguments(**changes))


def test_dry_run_does_not_resolve_identity_or_create_clients(monkeypatch, capsys) -> None:
    import socket
    import subprocess

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not access the network or runtime profiles")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    args = [
        "--url",
        "http://127.0.0.1:18181",
        "--org",
        "research",
        "--project",
        "experiment",
        "--gcp-project",
        "demo-project-123",
        "--zone",
        "us-east5-a",
        "--agent-url",
        "http://127.0.0.1:8090",
        "--artifact-prefix",
        "gs://example-bucket/priority-demo",
    ]
    assert _module()["main"](args) == 0
    assert '"execute": false' in capsys.readouterr().out


def test_requests_use_real_priority_bands_and_bound_checkpoint_identity() -> None:
    module = _module()
    plan = module["build_plan"](**_arguments())
    batch = module["workload_request"](plan, "batch", "print('payload')")
    urgent = module["workload_request"](plan, "urgent", "print('payload')")
    assert batch.spec.priority_band.name == "BATCH"
    assert urgent.spec.priority_band.name == "INTERACTIVE"
    assert batch.spec.checkpointable and batch.spec.checkpoint_hook
    assert batch.spec.checkpoint_output_uri == plan["checkpoint_uri"]
    assert batch.spec.max_failure_retries == 0
    assert batch.spec.required_labels == {"eyvan_fleet": plan["fleet_name"]}
    assert batch.spec.gang.per_replica.tpu == 4
    assert not batch.spec.additional_capacity_classes
    assert not urgent.spec.checkpointable


def test_marker_parser_uses_message_json_and_exact_run_identity() -> None:
    import json

    find = _module()["find_marker"]
    valid = {"marker": "checkpoint_ready", "run_id": "a" * 32, "step": 20}
    page = {
        "records": [
            {"text": json.dumps(valid)},
            {"message": str(valid)},
            {"message": json.dumps(valid | {"run_id": "foreign"})},
        ]
    }
    assert find(page, "checkpoint_ready", "a" * 32) is None
    page["records"].append({"message": json.dumps(valid)})
    assert find(page, "checkpoint_ready", "a" * 32) == valid


@pytest.mark.parametrize("fail_save", [False, True])
@pytest.mark.parametrize("fail_reads", [False, True])
def test_cleanup_reconciles_ambiguous_admission_and_only_cancels_owned_ids(fail_save, fail_reads, monkeypatch) -> None:
    from types import SimpleNamespace

    module = _module()
    plan = module["build_plan"](**_arguments())
    request = module["workload_request"](plan, "batch", "print('payload')")
    item = SimpleNamespace(
        id="wl_owned",
        org_id=plan["org"],
        project_id=plan["project"],
        spec=request.spec,
        state=SimpleNamespace(value="running"),
    )
    events = []
    _advancing_clock(monkeypatch)

    class Client:
        def submit_workload(self, supplied, *, idempotency_key):
            assert supplied is request
            assert idempotency_key == plan["run_id"] + "-batch"
            events.append("reconcile")
            return item

        def get_workload(self, workload_id):
            assert workload_id == "wl_owned"
            if fail_reads and "workload-read-failed" not in events:
                events.append("workload-read-failed")
                raise OSError("transient read outage")
            if item.state.value == "cancelling":
                item.state.value = "cancelled"
                events.append("terminal-observed")
            return item

        def workload_action(self, workload_id, action, *, idempotency_key):
            assert idempotency_key == plan["run_id"] + "-cancel-batch"
            events.append((workload_id, action))
            item.state.value = "cancelling"

    class Transport:
        def request_checked(self, method, path):
            assert path == "/api/v1/fleets/" + plan["fleet_name"]
            if fail_reads and "fleet-read-failed" not in events:
                events.append("fleet-read-failed")
                raise OSError("transient read outage")
            return SimpleNamespace(json=lambda: {"pools": plan["fleet_request"]["pools"], "cells": []})

        def mutate(self, method, path, **kwargs):
            assert method == "DELETE" and path.endswith(plan["fleet_name"])
            events.append("disable")
            return SimpleNamespace(raise_for_status=lambda: None, status_code=200)

    pending = {"batch": {"request": request, "id": None}}
    receipt = {}

    def save():
        if fail_save:
            raise OSError("receipt storage failed")

    module["cleanup_owned"](plan, Client(), Transport(), pending, receipt, fleet_started=True, save=save)
    assert "disable" in events and "reconcile" in events and ("wl_owned", "cancel") in events
    assert events.count(("wl_owned", "cancel")) == 1
    assert "terminal-observed" in events
    assert pending["batch"]["id"] == "wl_owned"
    assert receipt["cleanup"]["api_cells_deleted"] is True
    assert receipt["cleanup"]["provider_absence_verified"] is False
    assert bool(receipt["cleanup"]["errors"]) is fail_save


@pytest.mark.parametrize("expire_cancellation", [False, True])
def test_cleanup_still_attempts_fleet_disable_after_cancel_failure(expire_cancellation, monkeypatch) -> None:
    from types import SimpleNamespace

    module = _module()
    plan = module["build_plan"](**_arguments())
    request = module["workload_request"](plan, "batch", "print('payload')")
    events = []
    clock = _advancing_clock(monkeypatch)

    class Client:
        def get_workload(self, workload_id):
            if expire_cancellation:
                clock[0] += 121
            return SimpleNamespace(
                id=workload_id,
                org_id=plan["org"],
                project_id=plan["project"],
                spec=request.spec,
                state=SimpleNamespace(value="running"),
            )

        def workload_action(self, *args, **kwargs):
            raise RuntimeError("private diagnostic must not enter the receipt")

    class Transport:
        def request_checked(self, *args):
            return SimpleNamespace(json=lambda: {"pools": plan["fleet_request"]["pools"], "cells": []})

        def mutate(self, *args, **kwargs):
            events.append("disable")
            return SimpleNamespace(raise_for_status=lambda: None, status_code=200)

    receipt = {}
    module["cleanup_owned"](
        plan, Client(), Transport(), {"batch": {"request": request, "id": "wl_owned"}}, receipt, fleet_started=True
    )
    assert events == ["disable"]
    assert receipt["cleanup"]["errors"]
    assert "private diagnostic" not in str(receipt)


def test_existing_fleet_is_neither_overwritten_nor_disabled() -> None:
    from types import SimpleNamespace

    module = _module()
    plan = module["build_plan"](**_arguments())

    class Transport:
        def request(self, *args, **kwargs):
            return SimpleNamespace(status_code=200)

        def mutate(self, *args, **kwargs):
            raise AssertionError("existing fleets are not owned by this invocation")

    receipt = module["run_workflow"](plan, object(), Transport(), "print('payload')", save=lambda receipt: None)
    assert receipt["priority_restoration_verified"] is False
    assert receipt["failure_type"] == "RuntimeError"
    assert receipt["cleanup"]["teardown_requested"] is False


def test_success_requires_checkpoint_attempt_and_computation_contracts() -> None:
    from types import SimpleNamespace

    module = _module()
    plan = module["build_plan"](**_arguments())
    batch = SimpleNamespace(
        attempt=2, failure_count=0, preemption_count=0, capacity_loss_count=0, checkpoint_uri=plan["checkpoint_uri"]
    )
    restored = {
        "restored_step": 20,
        "final_step": 40,
        "device_count": 4,
        "matches_uninterrupted_control": True,
        "checkpoint_instance": "owned-spot-vm",
        "restored_instance": "owned-spot-vm",
    }
    module["verify_restoration"](plan, batch, restored)
    for update in (
        {"matches_uninterrupted_control": False},
        {"device_count": 0},
        {"restored_step": 0},
        {"checkpoint_instance": ""},
        {"restored_instance": "another-vm"},
    ):
        with pytest.raises(RuntimeError):
            module["verify_restoration"](plan, batch, restored | update)
    batch.attempt = 1
    with pytest.raises(RuntimeError):
        module["verify_restoration"](plan, batch, restored)


@pytest.mark.parametrize("lose_response", [False, True, "interrupt", "timeout", "receipt"])
def test_orchestration_and_admission_loss_with_api_doubles(lose_response, monkeypatch) -> None:
    import json
    from types import SimpleNamespace

    from eyvan.records import WorkloadRecord, WorkloadState

    module = _module()
    plan = module["build_plan"](**_arguments())
    records = {}
    cancellations = []
    calls = []
    clock = _advancing_clock(monkeypatch)

    class Client:
        submissions = 0

        def submit_workload(self, request, *, idempotency_key):
            self.submissions += 1
            role = "batch" if request.spec.checkpointable else "urgent"
            records.setdefault(
                role,
                WorkloadRecord(
                    id="wl_" + role,
                    org_id=plan["org"],
                    project_id=plan["project"],
                    spec=request.spec,
                    created_by="demo-test",
                ),
            )
            if self.submissions == 1:
                if lose_response is True:
                    raise OSError("simulated lost admission response")
                if lose_response == "interrupt":
                    raise KeyboardInterrupt
                if lose_response == "timeout":
                    clock[0] += plan["timeout"] + 1
            return records[role]

        def get_workload(self, workload_id):
            role = workload_id.removeprefix("wl_")
            item = records[role]
            if item.state == WorkloadState.CANCELLED:
                return item
            if len(records) == 2:
                return item.model_copy(
                    update={
                        "state": WorkloadState.SUCCEEDED,
                        "attempt": 2 if role == "batch" else 1,
                        "checkpoint_uri": plan["checkpoint_uri"],
                    }
                )
            return item.model_copy(update={"state": WorkloadState.RUNNING})

        def workload_log_page(self, workload_id, **kwargs):
            if workload_id == "wl_urgent":
                marker = {"marker": "urgent_complete", "device_count": 4}
            elif len(records) == 1:
                marker = {"marker": "checkpoint_ready", "step": 20}
            else:
                marker = {
                    "marker": "training_restored",
                    "restored_step": 20,
                    "final_step": 40,
                    "device_count": 4,
                    "matches_uninterrupted_control": True,
                    "checkpoint_instance": "owned-spot-vm",
                    "restored_instance": "owned-spot-vm",
                }
            return {"records": [{"message": json.dumps(marker | {"run_id": plan["run_id"]})}]}

        def workload_action(self, workload_id, action, **kwargs):
            assert kwargs["idempotency_key"] == plan["run_id"] + "-cancel-batch"
            cancellations.append((workload_id, action))
            role = workload_id.removeprefix("wl_")
            records[role] = records[role].model_copy(update={"state": WorkloadState.CANCELLED})

    class Transport:
        def request(self, method, path):
            return SimpleNamespace(status_code=404)

        def request_checked(self, *args):
            return SimpleNamespace(json=lambda: {"pools": plan["fleet_request"]["pools"], "cells": []})

        def mutate(self, method, path, **kwargs):
            assert path == "/api/v1/fleets/" + plan["fleet_name"]
            calls.append(method)
            return SimpleNamespace(raise_for_status=lambda: None, status_code=200)

    def save(receipt):
        if lose_response == "receipt" and receipt.get("admission_keys"):
            raise OSError("pre-admission receipt write failed")

    client = Client()
    receipt = module["run_workflow"](plan, client, Transport(), "print('payload')", save=save)
    assert calls == ["PUT", "DELETE"]
    assert receipt["priority_restoration_verified"] is (not lose_response), receipt.get("failure_type")
    assert bool(receipt["cleanup"]["errors"]) is (lose_response == "receipt")
    if lose_response == "receipt":
        assert client.submissions == 0 and not records and not cancellations
    elif lose_response:
        assert client.submissions == (1 if lose_response == "timeout" else 2) and len(records) == 1
        assert cancellations == [("wl_batch", "cancel")]
        if lose_response == "timeout":
            assert receipt["failure_type"] == "TimeoutError"
        elif lose_response == "interrupt":
            assert receipt["failure_type"] == "KeyboardInterrupt"
    else:
        assert not cancellations
        assert receipt["restored"]["matches_uninterrupted_control"] is True


def test_cleanup_reports_deadline_when_owned_workload_never_becomes_terminal(monkeypatch):
    from types import SimpleNamespace

    module = _module()
    plan = module["build_plan"](**_arguments())
    request = module["workload_request"](plan, "batch", "print('payload')")
    clock = _advancing_clock(monkeypatch)
    started = clock[0]
    cancellations = []
    events = []

    class Client:
        def get_workload(self, workload_id):
            return SimpleNamespace(
                id=workload_id,
                org_id=plan["org"],
                project_id=plan["project"],
                spec=request.spec,
                state=SimpleNamespace(value="cancelling"),
            )

        def workload_action(self, workload_id, action, *, idempotency_key):
            cancellations.append((workload_id, action, idempotency_key))

    class Transport:
        def request_checked(self, *args):
            return SimpleNamespace(json=lambda: {"pools": plan["fleet_request"]["pools"], "cells": []})

        def mutate(self, *args, **kwargs):
            events.append("disable")
            return SimpleNamespace(raise_for_status=lambda: None, status_code=200)

    receipt = {}
    module["cleanup_owned"](
        plan, Client(), Transport(), {"batch": {"request": request, "id": "wl_owned"}}, receipt, fleet_started=True
    )
    assert events == ["disable"]
    assert cancellations == [("wl_owned", "cancel", plan["run_id"] + "-cancel-batch")]
    # The fake clock starts at the real monotonic value, so the difference is
    # a float subtraction and is not exactly 120 for every starting value.
    assert clock[0] - started == pytest.approx(120)
    assert receipt["cleanup"]["errors"] == [{"operation": "cancel-batch", "type": "TimeoutError"}]
    assert receipt["cleanup"]["api_cells_deleted"] is True
    assert receipt["cleanup"]["provider_absence_verified"] is False


def test_log_polling_advances_cursor_and_retains_prior_markers():
    import json
    from types import SimpleNamespace

    module = _module()
    run_id = "a" * 32
    marker = {"marker": "checkpoint_ready", "run_id": run_id, "step": 20}
    pages = [
        {"records": [{"message": "startup output"}] * 200, "next_cursor": "first-page"},
        {"records": [{"message": json.dumps(marker)}], "next_cursor": "second-page"},
        {"records": [], "next_cursor": "second-page"},
    ]
    cursors = []

    def read(workload_id, *, after, limit):
        assert workload_id == "owned" and limit == 200
        cursors.append(after)
        return pages.pop(0)

    client = SimpleNamespace(workload_log_page=read)
    progress = {}
    assert not module["poll_markers"](client, "owned", run_id, progress)
    assert module["poll_markers"](client, "owned", run_id, progress)["checkpoint_ready"] == marker
    assert module["poll_markers"](client, "owned", run_id, progress)["checkpoint_ready"] == marker
    assert cursors == ["", "first-page", "second-page"]


def test_total_operation_deadline_interrupts_a_slow_drip_response():
    import signal
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import httpx

    if not hasattr(signal, "setitimer"):
        pytest.skip("the paid operator example requires POSIX interval timers")
    module = _module()
    finished = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            try:
                self.send_response(200)
                self.send_header("Content-Length", "200")
                self.end_headers()
                for _ in range(200):
                    self.wfile.write(b"x")
                    self.wfile.flush()
                    time.sleep(0.01)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                finished.set()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    previous = signal.getsignal(signal.SIGALRM)
    try:
        with httpx.Client(timeout=10, trust_env=False) as client:
            with pytest.raises(TimeoutError):
                module["bounded_call"](lambda: client.get(f"http://127.0.0.1:{server.server_port}"), 0.15)
        assert finished.wait(5)
        assert signal.getsignal(signal.SIGALRM) is previous
        assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_termination_requests_cleanup_and_restores_prior_handler(monkeypatch):
    import signal

    module = _module()
    previous = object()
    installed = []
    monkeypatch.setattr(signal, "getsignal", lambda signum: previous)
    monkeypatch.setattr(signal, "signal", lambda signum, handler: installed.append((signum, handler)))
    with pytest.raises(KeyboardInterrupt):
        with module["termination_cleanup"]():
            installed[-1][1](signal.SIGTERM, None)
    assert installed[0][0] == signal.SIGTERM
    assert installed[-1] == (signal.SIGTERM, previous)


def test_cleanup_does_not_confuse_missing_cell_data_with_verified_deletion():
    from types import SimpleNamespace

    module = _module()
    plan = module["build_plan"](**_arguments())
    transport = SimpleNamespace(
        request_checked=lambda *args: SimpleNamespace(json=lambda: {"pools": plan["fleet_request"]["pools"]}),
        mutate=lambda *args, **kwargs: SimpleNamespace(raise_for_status=lambda: None),
    )
    receipt = {}
    module["cleanup_owned"](plan, SimpleNamespace(), transport, {}, receipt, fleet_started=True)
    assert receipt["cleanup"]["teardown_requested"] is True
    assert receipt["cleanup"]["api_cells_deleted"] is False
    assert receipt["cleanup"]["errors"]


def test_termination_during_cleanup_is_deferred_until_owned_teardown():
    import signal
    from types import SimpleNamespace

    module = _module()
    plan = module["build_plan"](**_arguments())
    events = []

    def read(*args):
        if not events:
            events.append("termination")
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        return SimpleNamespace(json=lambda: {"pools": plan["fleet_request"]["pools"], "cells": []})

    def mutate(*args, **kwargs):
        events.append("disable")
        return SimpleNamespace(raise_for_status=lambda: None)

    receipt = {}
    with module["termination_cleanup"]():
        try:
            module["cleanup_owned"](
                plan,
                SimpleNamespace(),
                SimpleNamespace(request_checked=read, mutate=mutate),
                {},
                receipt,
                fleet_started=True,
            )
        except KeyboardInterrupt:
            pass  # Turn an unsafe abort into an assertion failure, not a stopped pytest session.
    assert events == ["termination", "disable"]
    assert receipt["cleanup"]["api_cells_deleted"] is True
    assert receipt["cleanup"]["interrupted"] is True
