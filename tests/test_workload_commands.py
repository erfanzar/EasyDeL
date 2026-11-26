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
