# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""CLI coverage for workload-record side effects."""

from __future__ import annotations

import json
import typing as tp
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

import eyvan.sdk.commands.workloads as workload_commands
from eyvan._internal.json_types import JsonObject, JsonValue
from eyvan.capacity.autoscale import AutoscaleConfigCache
from eyvan.records import (
    AutoscaleBlockReason,
    CapacityClass,
    CommandExecution,
    GcpQueueAdmissionDenialTarget,
    ResourceGang,
    ResourceVector,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.sdk.client.identity import ClientIdentity
from eyvan.sdk.commands.state import CliSession, ContextOverrides
from eyvan.sdk.commands.workloads import workload
from eyvan.sdk.contexts import EyvanContext, EyvanContextConfig, FileContextStore

QueryValue: tp.TypeAlias = str | int | float | bool | None


@dataclass
class WorkloadResponseClient:
    """Small typed workload API boundary for Click command coverage."""

    records: tuple[JsonObject, ...]
    calls: list[tuple[str, str]] = field(default_factory=list)
    identity_headers: dict[str, str] = field(default_factory=dict)

    def request(
        self,
        method: str,
        path: str,
        *,
        params: tp.Mapping[str, QueryValue] | None = None,
        json: JsonObject | None = None,
        headers: tp.Mapping[str, str] | None = None,
    ) -> JsonValue:
        """Return durable workload API payloads for the requested read route."""

        del params, json, headers
        self.calls.append((method, path))
        if method != "GET":
            raise AssertionError(f"unexpected workload request method {method!r}")
        if path == "/api/v1/workloads":
            return list(self.records)
        prefix = "/api/v1/workloads/"
        if path.startswith(prefix):
            workload_id = path.removeprefix(prefix)
            for record in self.records:
                record_id = record.get("id")
                if record_id == workload_id:
                    return record
        raise AssertionError(f"unexpected workload request {path!r}")

    def request_text(
        self,
        method: str,
        path: str,
        *,
        params: tp.Mapping[str, QueryValue] | None = None,
        headers: tp.Mapping[str, str] | None = None,
    ) -> str:
        """Reject text endpoints because workload-read commands use JSON only."""

        del method, path, params, headers
        raise AssertionError("workload-read commands must not request text")

    def replace_identity_headers(self, headers: tp.Mapping[str, str]) -> None:
        """Keep the command-session protocol fully implemented for test construction."""

        self.identity_headers = dict(headers)


def _queue_admission_denied_workload() -> WorkloadRecord:
    """Return one full record whose immutable GCP target was queue-denied."""

    return WorkloadRecord(
        id="wl-cli-queue-denied",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="tests.queue_denied",
            execution=CommandExecution(command="python train.py"),
            gang=ResourceGang(per_replica=ResourceVector(cpu=32, tpu=16)),
            provider="gcp",
            region="us-east5",
            accelerator_type="v6e-16",
            capacity_class=CapacityClass.SPOT,
            required_executor="eyvan-native",
        ),
        state=WorkloadState.FAILED,
        failure_reason="GCP denied queue admission",
        autoscale_block_reason=AutoscaleBlockReason.QUEUE_ADMISSION_DENIED,
        queue_admission_denial=GcpQueueAdmissionDenialTarget(
            provider_project="gcp-project",
            zone="us-east5-a",
            accelerator_type="v6e-16",
            capacity_class=CapacityClass.SPOT,
            occurred_at=utcnow(),
        ),
        created_by="alice",
    )


def _session(tmp_path: Path, client: WorkloadResponseClient, cache_dir: Path) -> CliSession:
    """Build an active session with a deliberately non-default autoscale cache root."""

    context = EyvanContext(
        fleet="research-us-east5-a",
        org_id="acme",
        project_id="research",
        autoscale_cache_dir=str(cache_dir),
    )
    return CliSession(
        client=client,
        identity=ClientIdentity(dev_user="alice"),
        context_store=FileContextStore(tmp_path / "contexts.json"),
        context_config=EyvanContextConfig(active_context=context.fleet, contexts={context.fleet: context}),
        base_url="http://127.0.0.1:8080",
        org_id="acme",
        project_id="research",
        overrides=ContextOverrides(),
    )


@pytest.mark.parametrize(
    ("arguments", "expects_list"),
    (
        # `status` renders for a person and takes `--json` for the record;
        # `get` is the accessor and always emits it. The assertion below is
        # about what they cached, not how either renders.
        (("status", "wl-cli-queue-denied", "--json"), False),
        (("get", "wl-cli-queue-denied"), False),
        (("list",), True),
    ),
)
def test_workload_read_commands_cache_exact_durable_queue_denials(
    tmp_path: Path,
    arguments: tuple[str, ...],
    expects_list: bool,
) -> None:
    """Status, get, and list all suppress only the durable exact GCP target."""

    record = _queue_admission_denied_workload()
    document = record.model_dump(mode="json")
    cache_dir = tmp_path / "selected-autoscale"
    client = WorkloadResponseClient(records=(document,))

    result = CliRunner().invoke(workload, list(arguments), obj=_session(tmp_path, client, cache_dir))

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == ([document] if expects_list else document)
    assert AutoscaleConfigCache(cache_dir).blocks_queue_admission_denial(
        project="gcp-project",
        zone="us-east5-a",
        accelerator_type="v6e-16",
        capacity_class=CapacityClass.SPOT,
    )


def test_workload_status_keeps_the_api_response_when_local_denial_cache_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An optional local cache failure cannot hide a durable workload record."""

    class FailingAutoscaleConfigCache:
        """Record the attempted cache write before reporting an invalid local cache."""

        calls = 0

        def __init__(self, root: Path | str | None = None) -> None:
            del root

        def record_queue_admission_denial(
            self,
            *,
            project: str,
            zone: str,
            accelerator_type: str,
            capacity_class: CapacityClass,
            observed_at: datetime,
        ) -> None:
            del project, zone, accelerator_type, capacity_class, observed_at
            type(self).calls += 1
            raise ValueError("invalid local denial cache")

    record = _queue_admission_denied_workload()
    document = record.model_dump(mode="json")
    client = WorkloadResponseClient(records=(document,))
    monkeypatch.setattr(workload_commands, "AutoscaleConfigCache", FailingAutoscaleConfigCache)

    result = CliRunner().invoke(
        workload,
        ["status", record.id, "--json"],
        obj=_session(tmp_path, client, tmp_path / "selected-autoscale"),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == document
    assert FailingAutoscaleConfigCache.calls == 1


def test_a_failed_status_leads_with_the_provider_sentence_not_eyvans_category(tmp_path: Path) -> None:
    """The record answers "why is this stuck" in four different fields, and a
    person reading a status does not know which to look at. The provider's own
    words are the only ones that say what to go and fix."""

    record = _queue_admission_denied_workload()
    client = WorkloadResponseClient(records=(record.model_dump(mode="json"),))

    result = CliRunner().invoke(
        workload,
        ["status", record.id],
        obj=_session(tmp_path, client, tmp_path / "selected-autoscale"),
    )

    assert result.exit_code == 0, result.output
    assert "v6e-16" in result.output
    # The durable record is not what a status prints any more.
    assert "worker_resource_scope" not in result.output
    assert result.output.count("\n") < 20, "a status is a summary, not a record dump"


class _ScriptedClient:
    """Flexible scripted API client for workload CLI command regression tests."""

    def __init__(self, responses: dict[tuple[str, str], JsonValue]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str, JsonObject | None]] = []
        self.identity_headers: dict[str, str] = {}

    def request(
        self,
        method: str,
        path: str,
        *,
        params: tp.Mapping[str, QueryValue] | None = None,
        json: JsonObject | None = None,
        headers: tp.Mapping[str, str] | None = None,
    ) -> JsonValue:
        del params, headers
        self.calls.append((method, path, json))
        if (method, path) in self.responses:
            return self.responses[(method, path)]
        raise AssertionError(f"unexpected request {method} {path}")

    def request_text(
        self,
        method: str,
        path: str,
        *,
        params: tp.Mapping[str, QueryValue] | None = None,
        headers: tp.Mapping[str, str] | None = None,
    ) -> str:
        del method, path, params, headers
        return ""

    def replace_identity_headers(self, headers: tp.Mapping[str, str]) -> None:
        self.identity_headers = dict(headers)


def test_workload_progress_children_endpoints_and_kick_prefix_json(tmp_path: Path) -> None:
    base = _queue_admission_denied_workload().model_dump(mode="json")
    record = {**base, "id": "wl-cli-queue-denied-1", "spec": {**base["spec"], "name": "tests.queue_denied_1"}}
    second = {**base, "id": "wl-cli-queue-denied-2", "spec": {**base["spec"], "name": "tests.queue_denied_2"}}
    wid = str(record["id"])
    progress_payload: JsonObject = {
        "workload_id": wid,
        "workload_state": "RUNNING",
        "phase": "running",
        "pool": {"pool_id": "pool-a", "state": "ACTIVE"},
        "capacity": {"ready_cells": 2, "total_cells": 2},
        "scheduling_reason": None,
        "failure_reason": None,
        "autoscale_block_reason": None,
        "user_progress": [
            {"rank": 0, "step": 50, "total": 100, "phase": "train", "message": "loss=1.23"},
        ],
    }
    children_payload: JsonValue = [
        {"id": "wl-child-1", "spec": {"name": "step-0"}, "state": "RUNNING", "attempt": 1, "assigned_pool_id": "p1"}
    ]
    endpoints_payload: JsonValue = [
        {"workload_id": "wl-child-1", "endpoint": "http://10.0.0.1:8000", "revision": "rev-1"}
    ]
    client = _ScriptedClient(
        {
            ("GET", "/api/v1/workloads"): [record, second],
            ("GET", f"/api/v1/workloads/{wid}/progress"): progress_payload,
            ("GET", f"/api/v1/workloads/{wid}/children"): children_payload,
            ("GET", f"/api/v1/workloads/{wid}/endpoints"): endpoints_payload,
            ("POST", f"/api/v1/workloads/{wid}/kick"): {"id": wid, "state": "QUEUED"},
            ("POST", "/api/v1/workloads/wl-cli-queue-denied-2/kick"): {
                "id": "wl-cli-queue-denied-2",
                "state": "QUEUED",
            },
        }
    )
    session = _session(tmp_path, tp.cast(tp.Any, client), tmp_path / "autoscale")

    # workload progress human & JSON
    prog_human = CliRunner().invoke(workload, ["progress", wid], obj=session)
    assert prog_human.exit_code == 0, prog_human.output
    assert "RUNNING" in prog_human.output and "50/100" in prog_human.output and "loss=1.23" in prog_human.output
    prog_json = CliRunner().invoke(workload, ["progress", wid, "--json"], obj=session)
    assert prog_json.exit_code == 0, prog_json.output
    assert json.loads(prog_json.output)["workload_id"] == wid

    # workload children table & JSON
    child_human = CliRunner().invoke(workload, ["children", wid], obj=session)
    assert child_human.exit_code == 0, child_human.output
    assert "WORKLOAD" not in child_human.output and "wl-child-1" in child_human.output and "step-0" in child_human.output
    child_json = CliRunner().invoke(workload, ["children", wid, "--json"], obj=session)
    assert child_json.exit_code == 0 and json.loads(child_json.output)[0]["id"] == "wl-child-1"

    # workload endpoints table & JSON
    ep_human = CliRunner().invoke(workload, ["endpoints", wid], obj=session)
    assert ep_human.exit_code == 0, ep_human.output
    assert "http://10.0.0.1:8000" in ep_human.output and "rev-1" in ep_human.output
    ep_json = CliRunner().invoke(workload, ["endpoints", wid, "--json"], obj=session)
    assert ep_json.exit_code == 0 and json.loads(ep_json.output)[0]["endpoint"] == "http://10.0.0.1:8000"

    # workload kick --prefix --json emits ONE valid JSON array
    kick = CliRunner().invoke(workload, ["kick", "--prefix", "wl-cli-queue-denied", "--yes", "--json"], obj=session)
    assert kick.exit_code == 0, kick.output
    kicked = json.loads(kick.output)
    assert isinstance(kicked, list) and [item["id"] for item in kicked] == [wid, "wl-cli-queue-denied-2"]


def test_run_and_submit_forward_setup_secret_env_and_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from eyvan.sdk.client import easydel_setup
    from eyvan.sdk.commands.workloads import run_workload

    commit = "a" * 40
    line = f"{commit} refs/heads/main\n".encode()
    adv = b"001e# service=git-upload-pack\n0000" + f"{len(line) + 4:04x}".encode() + line + b"0000"
    monkeypatch.setattr(easydel_setup, "_fetch_advertised_refs", lambda _repo: adv)

    submitted_record = _queue_admission_denied_workload().model_dump(mode="json")
    client = _ScriptedClient({("POST", "/api/v1/workloads"): submitted_record})
    session = _session(tmp_path, tp.cast(tp.Any, client), tmp_path / "autoscale")

    # eyvan workload submit with --env, --with, --setup-url, --secret
    res_submit = CliRunner().invoke(
        workload,
        [
            "submit",
            "--name",
            "job-a",
            "--command",
            "python train.py",
            "--env",
            "FOO=bar",
            "--with",
            "einops>=0.8",
            "--setup-url",
            "https://example.com/setup.sh",
            "--secret",
            "HF_TOKEN=gcp-secret://proj-a/hf-token",
        ],
        obj=session,
    )
    assert res_submit.exit_code == 0, res_submit.output
    _, _, payload = client.calls[-1]
    assert payload is not None
    spec = tp.cast(dict[str, tp.Any], payload["spec"])
    assert spec["environment"] == {"FOO": "bar"}
    assert spec["dependencies"] == ["einops>=0.8"]
    assert spec["setup"]["url"] == "https://example.com/setup.sh"
    assert spec["secret_mounts"] == {"HF_TOKEN": "gcp-secret://proj-a/hf-token"}

    # eyvan workload submit --easydel
    res_ed = CliRunner().invoke(
        workload,
        ["submit", "--name", "job-ed", "--command", "python train.py", "--tpu", "8", "--easydel"],
        obj=session,
    )
    assert res_ed.exit_code == 0, res_ed.output
    _, _, ed_payload = client.calls[-1]
    assert ed_payload is not None
    ed_spec = tp.cast(dict[str, tp.Any], ed_payload["spec"])
    assert "--ref" in ed_spec["setup"]["args"] and commit in ed_spec["setup"]["args"]

    # eyvan run --no-wait --secret --setup <file>
    setup_file = tmp_path / "init.sh"
    setup_file.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
    client.responses[("POST", "/api/v1/fleets/research-us-east5-a/node-types/pool-a/workloads")] = submitted_record
    res_run = CliRunner().invoke(
        run_workload,
        [
            "--no-wait",
            "--secret",
            "API_KEY=file:///etc/eyvan/secrets/api-key",
            "--setup",
            str(setup_file),
            "pool-a",
            "--",
            "python",
            "train.py",
        ],
        obj=session,
    )
    assert res_run.exit_code == 0, res_run.output
    _, _, run_payload = client.calls[-1]
    assert run_payload is not None
    assert run_payload["secret_mounts"] == {"API_KEY": "file:///etc/eyvan/secrets/api-key"}
    assert "echo hi" in tp.cast(dict[str, tp.Any], run_payload["setup"])["script"]

    # Mutually exclusive setup flags refused
    conflict = CliRunner().invoke(
        workload,
        [
            "submit",
            "--name",
            "bad",
            "--command",
            "true",
            "--setup",
            "echo a",
            "--setup-url",
            "https://example.com/b.sh",
        ],
        obj=session,
    )
    assert conflict.exit_code != 0 and "at most one" in conflict.output

    # --easydel boolean flag must NOT consume positional TARGET on `eyvan run`.
    res_run_ed = CliRunner().invoke(
        run_workload,
        ["--no-wait", "--easydel", "pool-a", "--", "python", "train.py"],
        obj=session,
    )
    assert res_run_ed.exit_code == 0, res_run_ed.output
    method, path, run_ed_payload = client.calls[-1]
    assert (method, path) == ("POST", "/api/v1/fleets/research-us-east5-a/node-types/pool-a/workloads")
    assert run_ed_payload is not None and commit in tp.cast(dict[str, tp.Any], run_ed_payload["setup"])["args"]

    # --easydel-branch accepts an explicit commit/branch without hijacking TARGET.
    custom_commit = "b" * 40
    res_run_branch = CliRunner().invoke(
        run_workload,
        ["--no-wait", "--easydel-branch", custom_commit, "pool-a", "--", "python", "train.py"],
        obj=session,
    )
    assert res_run_branch.exit_code == 0, res_run_branch.output
    _, _, branch_payload = client.calls[-1]
    assert branch_payload is not None and custom_commit in tp.cast(dict[str, tp.Any], branch_payload["setup"])["args"]

    # Long (>255 char) single-line inline --setup script must not crash with OSError ENAMETOOLONG.
    long_inline = "echo " + ("x" * 300)
    res_long = CliRunner().invoke(
        workload,
        ["submit", "--name", "job-long", "--command", "true", "--setup", long_inline],
        obj=session,
    )
    assert res_long.exit_code == 0, res_long.output
    _, _, long_payload = client.calls[-1]
    assert long_payload is not None
    assert tp.cast(dict[str, tp.Any], long_payload["spec"])["setup"]["script"] == long_inline
