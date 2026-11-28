# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Explicit, dry-run-first operator workflow for one Spot TPU priority demo."""

from __future__ import annotations

import argparse
import contextlib
import ipaddress
import json
import math
import re
import uuid
from urllib.parse import urlsplit


def _endpoint(value: str) -> str:
    if not isinstance(value, str) or any(ord(char) < 33 for char in value):
        raise ValueError("endpoint contains whitespace or control characters")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("endpoint must be an explicit HTTP(S) origin without credentials")
    try:
        loopback = ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        loopback = parsed.hostname.lower() == "localhost"
    if parsed.scheme == "http" and not loopback:
        raise ValueError("non-loopback endpoints require HTTPS")
    if parsed.port is not None and not 0 < parsed.port < 65536:
        raise ValueError("invalid endpoint port")
    return value.rstrip("/")


def build_plan(
    *,
    url: str,
    org: str,
    project: str,
    gcp_project: str,
    zone: str,
    agent_url: str,
    artifact_prefix: str,
    run_id: str | None = None,
    timeout: float = 2400,
) -> dict:
    """Compile a unique, zero-initial-cell Spot fleet without contacting any service."""
    from eyvan.capacity.fleet_config import EyvanFleetConfig
    from eyvan.sdk.commands.fleets import FleetCommandService

    url, agent_url = _endpoint(url), _endpoint(agent_url)
    run_id = uuid.uuid4().hex if run_id is None else run_id
    if not re.fullmatch(r"[a-f0-9]{32}", run_id):
        raise ValueError("run ID must be 32 lowercase hexadecimal characters")
    if not re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", gcp_project):
        raise ValueError("an explicit GCP project ID is required")
    if not re.fullmatch(r"[a-z]+-[a-z]+[0-9]+-[a-z]", zone):
        raise ValueError("an explicit GCP zone is required")
    if any(not isinstance(value, str) or not value.strip() or len(value) > 128 for value in (org, project)):
        raise ValueError("explicit Eyvan organization and project IDs are required")
    if isinstance(timeout, bool) or not math.isfinite(timeout) or not 60 <= timeout <= 3600:
        raise ValueError("timeout must be between 60 and 3600 seconds")
    prefix = urlsplit(artifact_prefix)
    if (
        any(ord(char) < 33 for char in artifact_prefix)
        or prefix.scheme != "gs"
        or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]", prefix.netloc)
        or prefix.query
        or prefix.fragment
        or not prefix.path.strip("/")
        or any(part in {"", ".", ".."} for part in prefix.path.strip("/").split("/"))
    ):
        raise ValueError("artifact prefix must be an explicit gs://bucket/nonempty-prefix without credentials")
    fleet_name = "eyvan-priority-" + run_id
    fleet = EyvanFleetConfig.model_validate(
        {
            "version": 1,
            "name": fleet_name,
            "labels": {"eyvan_demo_run": run_id},
            "provider": {"type": "gcp", "project": gcp_project, "region": zone.rsplit("-", 1)[0], "zone": zone},
            "executor": "eyvan-native",
            "capacity_class": "spot",
            "runtime": {
                "python": "3.11",
                "init_commands": [
                    "uv pip install --python /opt/eyvan-native/venv/bin/python "
                    "'jax[tpu]==0.11.2' 'google-cloud-storage==3.3.1'"
                ],
            },
            "autoscaling": {"idle_timeout_s": 300, "provisioning_timeout_s": 1200, "launch_batch": 1},
            "node_types": {
                "trainer": {
                    "kind": "tpu",
                    "accelerator_type": "v5p-8",
                    "min_workers": 0,
                    "max_workers": 1,
                    "initial_workers": 0,
                    "agent_endpoint": agent_url,
                    "provider_options": {"bootstrap_egress": "public-internet"},
                }
            },
        }
    )
    request = FleetCommandService.apply_request(fleet).model_dump(mode="json")
    return {
        "url": url,
        "org": org,
        "project": project,
        "gcp_project": gcp_project,
        "zone": zone,
        "run_id": run_id,
        "fleet_name": fleet_name,
        "fleet_request": request,
        "timeout": timeout,
        "checkpoint_uri": artifact_prefix.rstrip("/") + "/" + run_id + "/checkpoint.json",
    }


def workload_request(plan: dict, role: str, payload_source: str):
    """Build typed wire requests; batch is a priority band, not a workload tree."""
    import shlex

    from eyvan.server.api.schemas import WorkloadSubmitRequest

    if role not in {"batch", "urgent"}:
        raise ValueError("unknown demo role")
    (pool,) = plan["fleet_request"]["pools"]
    chips = pool["allocatable"]["tpu"]
    if chips != 4:
        raise ValueError("this payload requires a compiled four-chip pool")

    def command(mode):
        return shlex.join(
            [
                "/opt/eyvan-native/venv/bin/python",
                "-c",
                payload_source,
                mode,
                "--gcp-project",
                plan["gcp_project"],
                "--checkpoint-uri",
                plan["checkpoint_uri"],
                "--run-id",
                plan["run_id"],
                "--devices",
                "4",
            ]
        )

    spec = {
        "name": plan["fleet_name"] + "-" + role,
        "execution": {"kind": "command", "command": command("train" if role == "batch" else "urgent")},
        "gang": {"replicas": 1, "per_replica": {"cpu": 1, "memory_gib": 2, "tpu": chips}},
        "provider": "gcp",
        "region": plan["zone"].rsplit("-", 1)[0],
        "accelerator_type": "v5p-8",
        "capacity_class": "spot",
        "additional_capacity_classes": [],
        "required_executor": "eyvan-native",
        "required_labels": {"eyvan_fleet": plan["fleet_name"]},
        "estimated_duration_seconds": 300,
        "priority_band": "batch" if role == "batch" else "interactive",
        "max_failure_retries": 0,
        "max_preemption_retries": 2,
    }
    if role == "batch":
        spec.update(
            checkpointable=True,
            checkpoint_hook=command("hook"),
            checkpoint_output_uri=plan["checkpoint_uri"],
            checkpoint_grace_seconds=30,
        )
    return WorkloadSubmitRequest.model_validate({"org_id": plan["org"], "project_id": plan["project"], "spec": spec})


def find_marker(page: dict, marker: str, run_id: str) -> dict | None:
    """Only structured payload output for this run counts as computation evidence."""
    for record in page.get("records", []):
        message = record.get("message")
        if not isinstance(message, str):
            continue
        try:
            value = json.loads(message)
        except ValueError:
            continue
        if isinstance(value, dict) and value.get("marker") == marker and value.get("run_id") == run_id:
            return value
    return None


def poll_markers(client, workload_id, run_id, progress):
    page = client.workload_log_page(workload_id, after=progress.get("cursor", ""), limit=200)
    markers = progress.setdefault("markers", {})
    for name in ("checkpoint_ready", "training_restored", "urgent_complete"):
        marker = find_marker(page, name, run_id)
        if marker is not None:
            markers[name] = marker
    cursor = page.get("next_cursor")
    if isinstance(cursor, str):
        progress["cursor"] = cursor
    return markers


TERMINAL = {"succeeded", "failed", "cancelled"}


def _owned_workload(plan, item, request):
    if (
        item.org_id != plan["org"]
        or item.project_id != plan["project"]
        or item.spec.name != request.spec.name
        or item.spec.required_labels != {"eyvan_fleet": plan["fleet_name"]}
    ):
        raise RuntimeError("workload ownership does not match the generated demo")


def _owned_fleet(plan, status):
    pools = status.get("pools", [])
    if len(pools) != 1 or pools[0].get("labels", {}).get("eyvan_demo_run") != plan["run_id"]:
        raise RuntimeError("fleet ownership changed; refusing to operate on it")


class _OperationExpired(BaseException):
    """Cross retrying transport layers without being mistaken for a retryable failure."""


def bounded_call(operation, seconds):
    import signal
    import threading

    if not hasattr(signal, "setitimer") or threading.current_thread() is not threading.main_thread():
        raise RuntimeError("paid execution requires POSIX timers in the main thread")
    if seconds <= 0:
        raise TimeoutError("operation deadline expired")
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise RuntimeError("refusing to replace another operation's interval timer")
    previous = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise _OperationExpired

    try:
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            return operation()
        except _OperationExpired:
            raise TimeoutError("operation deadline expired") from None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class DeadlineClient:
    """Bound each complete SDK operation, including transport retry and body consumption."""

    def __init__(self, client, deadline):
        self.client = client
        self.deadline = deadline

    def __getattr__(self, name):
        import time

        operation = getattr(self.client, name)

        def call(*args, **kwargs):
            return bounded_call(lambda: operation(*args, **kwargs), min(15, self.deadline - time.monotonic()))

        return call


def _retry_cleanup(operation):
    for attempt in range(3):
        try:
            return operation()
        except Exception:
            if attempt == 2:
                raise


@contextlib.contextmanager
def defer_termination(receipt):
    import signal

    previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}

    def defer(signum, frame):
        receipt.setdefault("cleanup", {})["interrupted"] = True

    try:
        for number in previous:
            signal.signal(number, defer)
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def cleanup_owned(plan, client, transport, pending, receipt, *, fleet_started, save=lambda: None):
    """Finish bounded owned cleanup even if termination is requested during it."""
    with defer_termination(receipt):
        _cleanup_owned(plan, client, transport, pending, receipt, fleet_started=fleet_started, save=save)


def _cleanup_owned(plan, client, transport, pending, receipt, *, fleet_started, save):
    """Disable acquisition, then wait for owned cancellation and API cell deletion.

    Cancellation acceptance is not terminal completion; conflicts while preemption
    executes remain retryable within the same 120-second cleanup budget. API cell
    deletion is still not independent cloud-absence proof.
    """
    import time

    from eyvan.sdk.client.transport import EyvanHTTPError

    deadline = time.monotonic() + 120
    client, transport = DeadlineClient(client, deadline), DeadlineClient(transport, deadline)
    cleanup = receipt.setdefault("cleanup", {})
    cleanup.update(errors=[], api_cells_deleted=False, provider_absence_verified=False, teardown_requested=False)
    path = "/api/v1/fleets/" + plan["fleet_name"]

    def error(operation, exc):
        cleanup["errors"].append({"operation": operation, "type": type(exc).__name__})

    def persist_cleanup():
        try:
            save()
        except Exception as exc:
            error("write-receipt", exc)

    cancellation_requested = set()
    unresolved = dict(pending)

    def cancel(role, entry):
        try:
            if entry["id"] is None:
                # Reconcile an ambiguous POST using the original request/key, never a new submission identity.
                item = _retry_cleanup(
                    lambda: client.submit_workload(entry["request"], idempotency_key=plan["run_id"] + "-" + role)
                )
                _owned_workload(plan, item, entry["request"])
                entry["id"] = item.id
                receipt.setdefault("workloads", {})[role] = item.id
                persist_cleanup()
            item = _retry_cleanup(lambda: client.get_workload(entry["id"]))
            _owned_workload(plan, item, entry["request"])
            if item.state.value in TERMINAL:
                return True
            if role not in cancellation_requested:
                _retry_cleanup(
                    lambda: client.workload_action(item.id, "cancel", idempotency_key=plan["run_id"] + "-cancel-" + role)
                )
                cancellation_requested.add(role)
            # Acceptance can mean CANCELLING, not terminal. Read back on later polls.
            return False
        except EyvanHTTPError as exc:
            if exc.status_code == 409 and exc.detail == (
                "preemption is already executing; retry cancellation after it completes"
            ):
                # This refusal does not bind the key. Keep the same owned request
                # eligible after a bounded delay, even if the fleet is already empty.
                return False
            error("cancel-" + role, exc)
            return True
        except Exception as exc:
            error("cancel-" + role, exc)
            return True

    def disable():
        status = transport.request_checked("GET", path).json()
        _owned_fleet(plan, status)
        response = transport.mutate("DELETE", path, idempotency_key=plan["run_id"] + "-teardown")
        response.raise_for_status()

    try:
        # Disable acquisition before cancellation retries can exhaust the cleanup budget.
        if fleet_started:
            _retry_cleanup(disable)
            cleanup["teardown_requested"] = True
    except Exception as exc:
        error("disable-fleet", exc)
    finally:
        waiting_for_cells = cleanup["teardown_requested"]
        while True:
            for role, entry in list(unresolved.items()):
                if cancel(role, entry):
                    del unresolved[role]
            if waiting_for_cells and time.monotonic() < deadline:
                try:
                    status = transport.request_checked("GET", path).json()
                    _owned_fleet(plan, status)
                    cells = status.get("cells")
                    if not isinstance(cells, list):
                        raise RuntimeError("cell deletion status is unavailable")
                    if all(cell.get("state") == "deleted" for cell in cells):
                        cleanup["api_cells_deleted"] = True
                        waiting_for_cells = False
                except Exception as exc:
                    error("wait-for-cell-deletion", exc)
                    waiting_for_cells = False
            remaining = deadline - time.monotonic()
            if (not unresolved and not waiting_for_cells) or remaining <= 0:
                break
            time.sleep(min(2, remaining))
        for role in unresolved:
            error("cancel-" + role, TimeoutError())
        if waiting_for_cells:
            error("wait-for-cell-deletion", TimeoutError())
        persist_cleanup()


def verify_restoration(plan, batch, restored):
    """Do not confuse ordinary execution retry or provider loss with priority restoration."""
    if (
        batch.attempt != 2
        or batch.failure_count != 0
        or batch.preemption_count != 0
        or batch.capacity_loss_count != 0
        or batch.checkpoint_uri != plan["checkpoint_uri"]
        or restored.get("restored_step") != 20
        or restored.get("final_step") != 40
        or restored.get("device_count") != 4
        or restored.get("matches_uninterrupted_control") is not True
        or not isinstance(restored.get("checkpoint_instance"), str)
        or not restored["checkpoint_instance"]
        or restored.get("restored_instance") != restored["checkpoint_instance"]
    ):
        raise RuntimeError("priority checkpoint restoration evidence is incomplete")


def run_workflow(plan, client, transport, payload_source, *, save):
    """Run only against the supplied clients; never select or supervise a local runtime."""
    import time

    receipt = {
        "plan": plan,
        "priority_restoration_verified": False,
        "workloads": {},
        "observations": [],
        "scope": "Same-host scheduler priority restoration; not natural Spot reclamation",
    }
    pending = {}
    fleet_started = False
    deadline = time.monotonic() + plan["timeout"]
    raw_client, raw_transport = client, transport
    client, transport = DeadlineClient(client, deadline), DeadlineClient(transport, deadline)
    path = "/api/v1/fleets/" + plan["fleet_name"]

    def persist():
        save(receipt)

    def submit(role):
        request = workload_request(plan, role, payload_source)
        receipt.setdefault("admission_keys", {})[role] = plan["run_id"] + "-" + role
        persist()  # Retain ambiguous admission identity before sending the POST.
        pending[role] = {"request": request, "id": None}  # Only now may the POST have been sent.
        item = client.submit_workload(request, idempotency_key=plan["run_id"] + "-" + role)
        _owned_workload(plan, item, request)
        pending[role]["id"] = item.id
        receipt["workloads"][role] = item.id
        persist()
        return item.id

    def observe():
        status = transport.request_checked("GET", path).json()
        _owned_fleet(plan, status)
        cells = [
            {
                key: cell.get(key)
                for key in ("id", "state", "capacity_class", "provider_project", "provider_failure_code")
            }
            for cell in status.get("cells", [])
        ]
        if any(cell["capacity_class"] != "spot" or cell["provider_project"] != plan["gcp_project"] for cell in cells):
            raise RuntimeError("fleet observation violates Spot/project constraints")
        if any(cell["provider_failure_code"] for cell in cells):
            raise RuntimeError("provider reports a capacity failure")
        items = {}
        for role, entry in pending.items():
            item = client.get_workload(entry["id"])
            _owned_workload(plan, item, entry["request"])
            if item.state.value in {"failed", "cancelled"}:
                raise RuntimeError("a demo workload terminated unsuccessfully")
            items[role] = item
        observation = {
            "cells": cells,
            "workloads": {
                role: {"id": item.id, "state": item.state.value, "attempt": item.attempt} for role, item in items.items()
            },
        }
        if not receipt["observations"] or observation != receipt["observations"][-1]:
            receipt["observations"].append(observation)
            persist()
        return items

    try:
        persist()
        existing = transport.request("GET", path)
        if existing.status_code != 404:
            raise RuntimeError("generated fleet must be absent before this invocation")
        fleet_started = True
        receipt["fleet_apply_started"] = True
        persist()
        response = transport.mutate("PUT", path, json=plan["fleet_request"], idempotency_key=plan["run_id"] + "-apply")
        response.raise_for_status()
        batch_id = submit("batch")
        batch_logs = {}
        while time.monotonic() < deadline:
            items = observe()
            ready = poll_markers(client, batch_id, plan["run_id"], batch_logs).get("checkpoint_ready")
            if ready is not None and ready.get("step") == 20:
                receipt["checkpoint_ready"] = ready
                break
            if items["batch"].state.value in TERMINAL:
                raise RuntimeError("batch ended before checkpoint readiness")
            time.sleep(2)
        else:
            raise TimeoutError("checkpoint readiness deadline")
        urgent_id = submit("urgent")
        urgent_logs = {}
        while time.monotonic() < deadline:
            items = observe()
            if all(item.state.value in {"failed", "cancelled"} for item in items.values()):
                raise RuntimeError("workloads ended without restoration")
            if all(item.state.value == "succeeded" for item in items.values()):
                restored = poll_markers(client, batch_id, plan["run_id"], batch_logs).get("training_restored")
                urgent = poll_markers(client, urgent_id, plan["run_id"], urgent_logs).get("urgent_complete")
                if restored is not None and urgent is not None:
                    verify_restoration(plan, items["batch"], restored)
                    receipt.update(priority_restoration_verified=True, restored=restored, urgent=urgent)
                    break
            time.sleep(2)
        else:
            raise TimeoutError("priority restoration deadline")
    except (Exception, KeyboardInterrupt) as exc:
        receipt["failure_type"] = type(exc).__name__
    finally:
        cleanup_owned(plan, raw_client, raw_transport, pending, receipt, fleet_started=fleet_started, save=persist)
    return receipt


@contextlib.contextmanager
def termination_cleanup():
    import signal

    previous = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, terminate)
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("url", "org", "project", "gcp-project", "zone", "agent-url", "artifact-prefix"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--timeout", type=float, default=2400)
    parser.add_argument(
        "--execute", action="store_true", help="Permit paid Spot allocation through the explicit Eyvan plane"
    )
    parser.add_argument("--receipt-dir", help="Required with --execute; retain this private directory for recovery")
    args = vars(parser.parse_args(argv))
    execute = args.pop("execute")
    receipt_dir = args.pop("receipt_dir")
    plan = build_plan(**args)
    if not execute:
        print(json.dumps({"execute": False, "plan": plan}, indent=2))
        return 0
    if receipt_dir is None:
        parser.error("--execute requires --receipt-dir")

    import hashlib
    import os
    import tempfile
    from pathlib import Path

    import httpx

    from eyvan._internal.plane_tls import control_plane_httpx_options
    from eyvan.sdk.client import ControlPlaneEndpoint, SyncApiTransport, resolve_control_plane_identity
    from eyvan.sdk.client.sync import EyvanClient

    source = Path(__file__).with_name("spot_training_payload.py").read_text(encoding="utf-8")
    plan["payload_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    directory = Path(receipt_dir)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    receipt_path = directory / (plan["run_id"] + ".json")
    if receipt_path.exists():
        raise ValueError("refusing to replace an existing run receipt")

    def save(receipt):
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as output:
            temporary = Path(output.name)
            try:
                json.dump(receipt, output, indent=2)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        os.replace(temporary, receipt_path)

    endpoint = ControlPlaneEndpoint.parse(plan["url"])
    identity = resolve_control_plane_identity(
        endpoint,
        token=os.environ.get("EYVAN_TOKEN"),
        dev_org=plan["org"],
        dev_project=plan["project"],
        dev_roles=("fleet_admin",),
    )
    print(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "fleet_name": plan["fleet_name"],
                "warning": "This allocates paid Spot TPU capacity. Keep the control plane awake.",
            }
        ),
        flush=True,
    )
    with (
        termination_cleanup(),
        httpx.Client(base_url=endpoint.url, timeout=5, trust_env=False, **control_plane_httpx_options()) as http,
    ):
        client = EyvanClient(endpoint.url, identity=identity, client=http)
        transport = SyncApiTransport(endpoint, identity=identity, client=http)
        receipt = run_workflow(plan, client, transport, source, save=save)
    passed = (
        receipt["priority_restoration_verified"]
        and receipt["cleanup"]["api_cells_deleted"]
        and not receipt["cleanup"]["errors"]
    )
    print(
        json.dumps(
            {
                "priority_restoration_verified": receipt["priority_restoration_verified"],
                "api_cells_deleted": receipt["cleanup"]["api_cells_deleted"],
                "provider_absence_verified": False,
                "receipt": str(receipt_path),
            }
        ),
        flush=True,
    )
    if receipt.get("failure_type") == "KeyboardInterrupt" or receipt["cleanup"].get("interrupted"):
        return 130
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
