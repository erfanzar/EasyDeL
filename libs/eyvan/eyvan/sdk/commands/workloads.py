# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Workload submission and lifecycle commands for the Eyvan CLI."""

from __future__ import annotations

import hashlib
import re
import typing as tp
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import click
import yaml
from pydantic import ValidationError

from eyvan._internal.json_types import JsonValue, is_json_object, model_json
from eyvan._internal.workspace import WorkspaceError, build_workspace
from eyvan.artifacts.managed import workspace_uri as managed_workspace_uri
from eyvan.capacity.autoscale import AutoscaleConfigCache, QueueAdmissionDenialRecorder
from eyvan.records import (
    DEFAULT_MAX_RETRIES,
    AuditEvent,
    CapacityClass,
    CommandExecution,
    PriorityBand,
    ResourceGang,
    ResourceVector,
    TpuChipRequest,
    WorkloadDefinition,
    WorkloadExplanation,
    WorkloadRecord,
    WorkloadSetup,
    WorkloadState,
    WorkspaceRef,
)
from eyvan.sdk.client.easydel_setup import DEFAULT_BRANCH as EASYDEL_DEFAULT_BRANCH
from eyvan.sdk.client.easydel_setup import EasyDeLSetup
from eyvan.sdk.client.idempotency import IdempotencyKeyFactory
from eyvan.sdk.client.jobs import SetupScript
from eyvan.sdk.client.requests import pool_workload_request
from eyvan.sdk.commands.common import JsonRenderer, emit_result, parse_environment, shell_command
from eyvan.sdk.commands.state import CliSession, RunTarget, WorkloadScope
from eyvan.sdk.follow import WorkloadFollower, parse_duration
from eyvan.sdk.job_receipt import JobSubmissionReceipt, JobTerminalOutput
from eyvan.sdk.spend import RunSpend
from eyvan.sdk.spend import render as spend_render
from eyvan.sdk.tables import Column
from eyvan.sdk.tables import field as table_field
from eyvan.sdk.tables import render as render_table
from eyvan.sdk.terminal import stream_is_terminal
from eyvan.sdk.timeline import render, timeline
from eyvan.sdk.workload_summary import format_user_progress
from eyvan.sdk.workload_summary import render as summarize_workload
from eyvan.server.api.schemas import (
    BatchSubmitRequest,
    DagNodeRequest,
    DagSubmitRequest,
    WorkloadEstimateRequest,
    WorkloadSubmitRequest,
)
from eyvan.server.service.batch import DEFAULT_MAX_FAN_OUT

WorkloadAction = tp.Literal["cancel", "checkpoint", "retry"]
_WORKLOAD_ACTIONS: tuple[WorkloadAction, ...] = ("cancel", "checkpoint", "retry")

#: What each mutation did, in the past tense, because it has already happened by
#: the time this is printed.
_ACTION_HEADLINES: dict[str, str] = {
    "cancel": "Cancelled",
    "checkpoint": "Asked for a checkpoint of",
    "retry": "Retrying",
}


def _resolve_cli_setup(
    *,
    setup: str | None,
    setup_url: str | None,
    easydel: bool = False,
    easydel_branch: str | None = None,
    accelerator: str = "tpu",
) -> WorkloadSetup | None:
    """Resolve ``--setup`` / ``--setup-url`` / ``--easydel`` into one ``WorkloadSetup``."""

    resolved_branch = easydel_branch if easydel_branch is not None else (EASYDEL_DEFAULT_BRANCH if easydel else None)
    provided = [
        flag
        for flag, value in (
            ("--setup", setup),
            ("--setup-url", setup_url),
            ("--easydel", resolved_branch),
        )
        if value is not None
    ]
    if len(provided) > 1:
        raise click.UsageError(f"pass at most one of {', '.join(provided)}")
    if not provided:
        return None
    try:
        if setup is not None:
            stripped = setup.strip()
            is_local_file = False
            if stripped and "\n" not in stripped:
                try:
                    is_local_file = Path(stripped).expanduser().is_file()
                except OSError:
                    is_local_file = False
            if is_local_file:
                return SetupScript(stripped).to_workload_setup()
            return WorkloadSetup(script=setup)
        if setup_url is not None:
            return WorkloadSetup(url=setup_url)
        assert resolved_branch is not None
        return EasyDeLSetup(branch=resolved_branch, accelerator=accelerator).to_workload_setup()
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc


def _chip_request(chips: int | None, chip_indexes: tp.Sequence[int]) -> TpuChipRequest | None:
    """Return the chip request ``--chips`` / ``--chip-index`` describe, or ``None`` for the whole slice."""

    if chips is None and not chip_indexes:
        return None
    indexes = tuple(chip_indexes) if chip_indexes else None
    count = chips if chips is not None else len(chip_indexes)
    return TpuChipRequest(count=count, indexes=indexes)


def _requirements(declared: tp.Sequence[str], path: Path | None) -> tuple[str, ...]:
    """Combine --with packages with a requirements file, read on this machine.

    Read here rather than on the host, and this is the whole reason the flag
    exists rather than passing `-r requirements.txt` through: that path names a
    file on *your* machine, and the host has never seen it. Comments and blank
    lines are dropped; an option line is refused, because `-r` or `--index-url`
    inside the file would do on the host exactly what passing it directly would.
    """

    packages = [item.strip() for item in declared if item.strip()]
    if path is not None:
        for line in path.read_text(encoding="utf-8").splitlines():
            # Requirements comments start a line or follow whitespace; a URL
            # fragment belongs to the requirement (hashes and VCS subdirectories).
            entry = re.split(r"(?:^|\s+)#", line, maxsplit=1)[0].strip()
            if not entry:
                continue
            if entry.startswith("-"):
                raise click.UsageError(
                    f"{path}: {entry!r} is an installer option, not a package. Eyvan ships the packages "
                    "themselves, so a file that redirects the install cannot travel with the job."
                )
            packages.append(entry)
    return tuple(dict.fromkeys(packages))


def _as_float(value: object) -> float:
    """Read a number from a JSON document, treating anything else as zero."""

    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def _as_str(value: object) -> str | None:
    """Read a string from a JSON document, or nothing when it is absent."""

    return value if isinstance(value, str) and value else None


def _float_map(value: object) -> dict[str, float]:
    """Read a map of numbers, dropping entries that are not numbers."""

    if not isinstance(value, dict):
        return {}
    return {
        str(key): float(item)
        for key, item in value.items()
        if isinstance(item, int | float) and not isinstance(item, bool)
    }


@dataclass(frozen=True, slots=True)
class WorkloadCommandService:
    """Typed application service used by workload Click callbacks."""

    session: CliSession
    renderer: JsonRenderer = field(default_factory=JsonRenderer)
    keys: IdempotencyKeyFactory = field(default_factory=lambda: IdempotencyKeyFactory(prefix="cli"))

    def _available_fleets(self, scope: WorkloadScope) -> tuple[str, ...]:
        payload = self.session.client.request(
            "GET",
            "/api/v1/fleets",
            params={"org_id": scope.org_id, "project_id": scope.project_id},
            headers=self.session.workload_headers(scope),
        )
        if not isinstance(payload, list) or not all(isinstance(fleet, str) for fleet in payload):
            raise click.ClickException(
                f"control plane at {self.session.base_url} returned an invalid fleet-name list; "
                "a version mismatch between this CLI and the plane is the usual cause -- align them "
                "(redeploy the plane, or upgrade this CLI) rather than retrying"
            )
        return tuple(sorted(set(payload)))

    @staticmethod
    def _submission_id(response: JsonValue) -> str | None:
        """Return the durable workload ID from a valid JSON submission response."""

        if not is_json_object(response):
            return None
        workload_id = response.get("id")
        return workload_id if isinstance(workload_id, str) and workload_id else None

    def _emit_submission_receipt(self, response: JsonValue, *, fleet: str | None) -> None:
        """Write human guidance to stderr without changing JSON stdout contracts."""

        workload_id = self._submission_id(response)
        if workload_id is None:
            return
        JobTerminalOutput(stream=click.get_text_stream("stderr")).submission(
            JobSubmissionReceipt(workload_id=workload_id, fleet=fleet)
        )

    @staticmethod
    def _workload_records(response: JsonValue) -> tuple[WorkloadRecord, ...]:
        """Parse complete workload records without changing a raw CLI response."""

        candidates: tuple[JsonValue, ...]
        if is_json_object(response):
            candidates = (response,)
        elif isinstance(response, list):
            candidates = tuple(response)
        else:
            return ()

        records: list[WorkloadRecord] = []
        for candidate in candidates:
            if not is_json_object(candidate):
                continue
            try:
                records.append(WorkloadRecord.model_validate(candidate))
            except ValidationError:
                continue
        return tuple(records)

    def _record_queue_admission_denials(self, response: JsonValue) -> None:
        """Persist provider-proven denials without changing workload CLI output."""

        records = self._workload_records(response)
        if not records:
            return
        try:
            recorder = QueueAdmissionDenialRecorder(AutoscaleConfigCache(self.session.autoscale_cache_dir))
            for record in records:
                recorder.record_workload(record)
        except (OSError, ValueError):
            # Local eligibility caching is optional; the workload response remains authoritative.
            return

    def _emit_workload_response(self, response: JsonValue) -> None:
        """Record any durable denial before emitting the unchanged JSON response."""

        self._record_queue_admission_denials(response)
        self.renderer.emit(response)

    def default_fleet(self, scope: WorkloadScope) -> str:
        """Use the active fleet, or infer the only submit-capable fleet."""

        active = self.session.active_context
        if active is not None and active.base_url == self.session.base_url and active.fleet is not None:
            return active.fleet
        fleets = self._available_fleets(scope)
        if len(fleets) == 1:
            return fleets[0]
        if not fleets:
            raise click.ClickException("no applied fleet is available; run `eyvan up` first")
        available = ", ".join(fleets)
        raise click.ClickException(
            f"multiple fleets are applied: {available}; run `eyvan use FLEET` once or use FLEET/TYPE"
        )

    def _remote_workspace_artifact_root(
        self,
        scope: WorkloadScope,
        *,
        fleet_name: str | None = None,
        prefer_remote: bool | None = None,
    ) -> str | None:
        """Return a remote GCS artifact root when a loopback plane submits to remote workers.

        A loopback control plane's ``eyvan://`` artifact URL resolves to ``127.0.0.1``,
        which remote GCP TPU/GPU hosts cannot reach. When a native GCP runtime profile is
        configured for this loopback plane and the target is remote, stage workspaces in
        its GCS bucket just like callable submissions do.
        """

        from eyvan.capacity.providers.gcp.runtime.profile import (
            GcpNativeRuntimeProfileError,
            GcpNativeRuntimeProfileStore,
        )
        from eyvan.capacity.providers.gcp.runtime.supervisor import runtime_profile_path_for
        from eyvan.sdk.client.endpoint import ControlPlaneEndpoint

        try:
            endpoint = ControlPlaneEndpoint.parse(self.session.base_url)
        except ValueError:
            return None
        if not endpoint.is_loopback:
            return None
        try:
            profile = GcpNativeRuntimeProfileStore(runtime_profile_path_for(endpoint)).load()
        except (FileNotFoundError, GcpNativeRuntimeProfileError):
            return None
        artifact_root = (profile.artifact_root_uri or "").strip()
        if not artifact_root:
            return None
        if prefer_remote is not None:
            return artifact_root if prefer_remote else None
        if not fleet_name:
            return None
        try:
            payload = self.session.client.request(
                "GET",
                "/api/v1/capacity",
                params={"org_id": scope.org_id, "project_id": scope.project_id},
                headers=self.session.workload_headers(scope),
            )
        except Exception:
            return None
        pools = payload.get("pools") if is_json_object(payload) else None
        if not isinstance(pools, list):
            return None
        matched = [p for p in pools if is_json_object(p) and p.get("fleet") == fleet_name]
        if not matched:
            return None
        has_remote_pool = any(
            isinstance(p.get("region"), str) and p["region"].strip() not in ("", "local") for p in matched
        )
        return artifact_root if has_remote_pool else None

    def _upload_workspace(
        self,
        root: Path,
        scope: WorkloadScope,
        *,
        fleet_name: str | None = None,
        prefer_remote: bool | None = None,
    ) -> WorkspaceRef:
        """Package the working tree and store it content-addressed, once.

        The digest *is* the storage key, so re-running an unchanged tree uploads
        nothing new and two developers on the same commit share one object.
        """

        try:
            bundle = build_workspace(root)
        except WorkspaceError as exc:
            raise click.ClickException(str(exc)) from exc
        remote_root = self._remote_workspace_artifact_root(
            scope,
            fleet_name=fleet_name,
            prefer_remote=prefer_remote,
        )
        if remote_root is not None:
            from eyvan.artifacts import artifact_store_from_uri

            uri = artifact_store_from_uri(remote_root).put_workspace(bundle.payload, bundle.sha256)
            click.echo(f"  workspace  {bundle.file_count} files, {bundle.size_bytes / 1e6:.1f} MB ({uri})", err=True)
            return WorkspaceRef(uri=uri, sha256=bundle.sha256)
        self.session.client.request(
            "PUT",
            f"/api/v1/artifacts/workspaces/{bundle.sha256}",
            params={"org_id": scope.org_id, "project_id": scope.project_id},
            content=bundle.payload,
            headers={"Content-Type": "application/gzip", "Idempotency-Key": f"workspace-{bundle.sha256}"},
        )
        click.echo(f"  workspace  {bundle.file_count} files, {bundle.size_bytes / 1e6:.1f} MB", err=True)
        return WorkspaceRef(uri=managed_workspace_uri(bundle.sha256), sha256=bundle.sha256)

    def run(
        self,
        *,
        target_value: str,
        org_id: str | None,
        project_id: str | None,
        name: str | None,
        environment: tuple[str, ...],
        max_retries: int,
        idempotency_key: str | None,
        command: tuple[str, ...],
        max_failure_retries: int = 0,
        workspace: Path | None = None,
        checkpointable: bool = False,
        checkpoint_hook: str | None = None,
        checkpoint_output_uri: str | None = None,
        can_submit_children: bool = False,
        priority_band: str = PriorityBand.INTERACTIVE.value,
        intra_band_priority: int = 0,
        checkpoint_grace_seconds: int = 0,
        estimated_duration_seconds: int = 3600,
        named_assets: tuple[str, ...] = (),
        pods: int = 1,
        zone: str | None = None,
        dependencies: tuple[str, ...] = (),
        wait: bool = True,
        chips: int | None = None,
        chip_indexes: tuple[int, ...] = (),
        secret_mounts: tuple[str, ...] = (),
        setup: WorkloadSetup | None = None,
    ) -> int:
        """Submit one pool-targeted command using context and fleet defaults.

        Returns:
            What the process should exit with: the followed job's outcome, or
            zero when ``wait`` is false and the job was merely handed over.
        """

        scope = self.session.workload_scope(org_id=org_id, project_id=project_id)
        default_fleet = None if "/" in target_value else self.default_fleet(scope)
        try:
            target = RunTarget.parse(target_value, default_fleet=default_fleet)
        except (ValidationError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        workspace_ref = (
            self._upload_workspace(workspace, scope, fleet_name=target.fleet) if workspace is not None else None
        )
        try:
            request = pool_workload_request(
                org_id=scope.org_id,
                project_id=scope.project_id,
                command=shell_command(command),
                name=name,
                environment=parse_environment(environment),
                secret_mounts=parse_environment(secret_mounts, option="--secret"),
                metadata=None,
                max_retries=max_retries,
                max_failure_retries=max_failure_retries,
                workspace=workspace_ref,
                checkpointable=checkpointable,
                checkpoint_hook=checkpoint_hook,
                checkpoint_output_uri=checkpoint_output_uri,
                can_submit_children=can_submit_children,
                priority_band=PriorityBand(priority_band),
                intra_band_priority=intra_band_priority,
                checkpoint_grace_seconds=checkpoint_grace_seconds,
                estimated_duration_seconds=estimated_duration_seconds,
                named_assets=named_assets,
                pods=pods,
                zone=zone,
                dependencies=dependencies,
                tpu_chips=_chip_request(chips, chip_indexes),
                setup=setup,
            )
        except (ValidationError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        response = self.session.client.request(
            "POST",
            f"/api/v1/fleets/{target.fleet}/node-types/{target.pool}/workloads",
            json=model_json(request),
            headers={
                **self.session.workload_headers(scope),
                "Idempotency-Key": self.keys.create("run", idempotency_key),
            },
        )
        workload_id = self._submission_id(response)
        if not wait or workload_id is None:
            # The handover contract, unchanged: the submission document on
            # stdout is what every script that pipes this command parses, and
            # it is only correct when stdout carries nothing else.
            self.renderer.emit(response)
            self._emit_submission_receipt(response, fleet=target.fleet)
            return 0
        self._emit_submission_receipt(response, fleet=target.fleet)
        return self.follow(workload_id)

    def follow(self, workload_id: str, *, since: timedelta | None = None) -> int:
        """Stream one workload's output until it stops, and report its outcome."""

        outcome = WorkloadFollower(
            client=self.session.client,
            output=JobTerminalOutput(stream=click.get_text_stream("stderr")),
            write=lambda line: click.echo(line),
            since=since,
        ).follow(workload_id)
        return outcome.exit_code

    def submit(
        self,
        request: WorkloadSubmitRequest,
        *,
        scope: WorkloadScope,
        idempotency_key: str | None,
    ) -> None:
        """Submit one fully specified workload request."""

        response = self.session.client.request(
            "POST",
            "/api/v1/workloads",
            json=model_json(request, exclude_unset=True),
            headers={
                **self.session.workload_headers(scope),
                "Idempotency-Key": self.keys.create("workload", idempotency_key),
            },
        )
        self.renderer.emit(response)
        self._emit_submission_receipt(response, fleet=None)

    def submit_batch(
        self,
        request: BatchSubmitRequest,
        *,
        scope: WorkloadScope,
        idempotency_key: str | None,
    ) -> None:
        """Fan one template out into an array of ordinary workloads."""

        response = self.session.client.request(
            "POST",
            "/api/v1/batches",
            json=model_json(request, exclude_unset=True),
            headers={
                **self.session.workload_headers(scope),
                "Idempotency-Key": self.keys.create("batch", idempotency_key),
            },
        )
        self.renderer.emit(response)

    def submit_dag(
        self,
        request: DagSubmitRequest,
        *,
        scope: WorkloadScope,
        idempotency_key: str | None,
    ) -> None:
        """Submit a pipeline of dependent workloads in one transaction."""

        response = self.session.client.request(
            "POST",
            "/api/v1/dags",
            json=model_json(request, exclude_unset=True),
            headers={
                **self.session.workload_headers(scope),
                "Idempotency-Key": self.keys.create("dag", idempotency_key),
            },
        )
        self.renderer.emit(response)

    def list(self, scope: WorkloadScope, state: WorkloadState | None) -> None:
        """List workload records and retain any durable queue-admission denials."""

        params: dict[str, str] = {"org_id": scope.org_id, "project_id": scope.project_id}
        if state is not None:
            params["state"] = state.value
        response = self.session.client.request(
            "GET",
            "/api/v1/workloads",
            params=params,
            headers=self.session.workload_headers(scope),
        )
        self._emit_workload_response(response)

    def show(self, workload_id: str, *, as_json: bool = False) -> None:
        """Display one workload, and retain any durable queue-admission denial.

        Summarised by default. The durable record is sixty-odd keys, most of
        them null, and the three facts somebody wants from a status -- what
        state it is in, what it runs, and why it is not running -- were
        scattered between ``autoscale_block_reason`` and
        ``worker_resource_scope``. ``--json`` still returns it verbatim, because
        anything parsing this needs the record and not the prose.
        """

        response = self.session.client.request("GET", f"/api/v1/workloads/{workload_id}")
        self._record_queue_admission_denials(response)
        if as_json:
            self.renderer.emit(response)
            return
        lines = summarize_workload(response)
        if not lines:
            # Unreadable to the summariser is not unreadable to the operator:
            # falling back to the record beats printing nothing at all.
            self.renderer.emit(response)
            return
        for line in lines:
            click.echo(line)
        # The summary says what state a run is in; the timeline is where "why
        # is it still pending" gets answered, and nobody finds a command they
        # were never told about.
        click.echo(f"  timeline  eyvan workload timeline {workload_id}")

    def process_status(self, workload_id: str) -> None:
        """Emit live process detail for one workload attempt."""

        self.renderer.emit(self.session.client.request("GET", f"/api/v1/workloads/{workload_id}/process-status"))

    def _events_about(self, entity_type: str, entity_id: str, *, scope: WorkloadScope, limit: int) -> list[AuditEvent]:
        """Read one entity's audit events, ignoring rows this client cannot parse."""

        payload = self.session.client.request(
            "GET",
            "/api/v1/events",
            params={
                "org_id": scope.org_id,
                "project_id": scope.project_id,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "limit": limit,
            },
            headers=self.session.workload_headers(scope),
        )
        events: list[AuditEvent] = []
        for entry in payload if isinstance(payload, list) else []:
            if not is_json_object(entry):
                continue
            try:
                events.append(AuditEvent.model_validate(entry))
            except ValidationError:
                # A newer plane recording something this client cannot model.
                # Skipped rather than fatal: an unreadable row must not take the
                # rest of the narrative with it.
                continue
        return events

    def timeline(self, workload_id: str, *, org_id: str | None, project_id: str | None, limit: int) -> None:
        """Print one ordered narrative of what happened to a workload.

        The workload's own events plus its leases', because "leased capacity"
        and "capacity released" are recorded against the lease and are two of
        the moments an operator most wants placed in the sequence.
        """

        scope = self.session.workload_scope(org_id=org_id, project_id=project_id)
        record = self.session.client.request("GET", f"/api/v1/workloads/{workload_id}")
        events = self._events_about("workload", workload_id, scope=scope, limit=limit)
        # Retry clears the aggregate's current lease references, but leaves the
        # historical lease ledger intact. Read it in the same authorized scope.
        leases = self.session.client.request(
            "GET",
            "/api/v1/leases",
            params={"org_id": scope.org_id, "project_id": scope.project_id, "workload_id": workload_id},
            headers=self.session.workload_headers(scope),
        )
        lease_ids = list(self._workload_lease_ids_from(record))
        for lease in leases if isinstance(leases, list) else []:
            if is_json_object(lease) and isinstance(lease.get("id"), str):
                lease_ids.append(tp.cast(str, lease["id"]))
        for lease_id in dict.fromkeys(lease_ids):
            events.extend(self._events_about("lease", lease_id, scope=scope, limit=limit))
        for line in render(timeline(events)):
            click.echo(line)

    def cost(
        self,
        workload_id: str,
        *,
        org_id: str | None,
        project_id: str | None,
        on_demand_hourly: float | None,
    ) -> None:
        """Print what one run cost, and what it would have cost on demand."""

        del org_id, project_id
        params = {} if on_demand_hourly is None else {"on_demand_hourly": on_demand_hourly}
        # Asked of the plane rather than assembled here. Totalling it locally
        # meant reading capacity pools for their unit prices, which needs
        # manage_fleet -- so this command 403'd for exactly the people who ask
        # what their own run cost, and its help never said so.
        document = self.session.client.request(
            "GET",
            f"/api/v1/workloads/{workload_id}/cost",
            params=params,
        )
        if not is_json_object(document):
            raise click.ClickException(
                f"control plane at {self.session.base_url} returned an invalid cost document; "
                "a version mismatch between this CLI and the plane is the usual cause -- align them "
                "(redeploy the plane, or upgrade this CLI) rather than retrying"
            )
        for line in spend_render(
            str(document.get("workload_id") or workload_id),
            str(document.get("state") or "unknown"),
            RunSpend(
                accelerator_hours=_float_map(document.get("accelerator_hours")),
                settled=_as_float(document.get("settled")),
                priced=bool(document.get("priced")),
                reserved=_as_float(document.get("reserved")),
                leases=int(_as_float(document.get("leases"))),
            ),
            shape=_as_str(document.get("shape")),
            capacity_class=_as_str(document.get("capacity_class")),
            on_demand_hourly=None
            if document.get("on_demand_hourly") is None
            else _as_float(document.get("on_demand_hourly")),
            on_demand_source=str(document.get("on_demand_source") or ""),
        ):
            click.echo(line)

    @staticmethod
    def _workload_lease_ids_from(record: JsonValue) -> tuple[str, ...]:
        """Return current lease ids explicitly named by a workload document."""

        if not is_json_object(record):
            return ()
        ids = record.get("lease_ids")
        if isinstance(ids, list):
            return tuple(value for value in ids if isinstance(value, str))
        single = record.get("lease_id")
        return (single,) if isinstance(single, str) else ()

    def explain(self, workload_id: str) -> None:
        """Emit why one workload is or is not currently placeable."""

        response = self.session.client.request("GET", f"/api/v1/workloads/{workload_id}/explain")
        self._emit_explanation_summary(response)
        self.renderer.emit(response)

    @staticmethod
    def _emit_explanation_summary(response: JsonValue) -> None:
        """Write a compact human summary to stderr, leaving JSON stdout unchanged."""

        stream = click.get_text_stream("stderr")
        if not stream_is_terminal(stream) or not is_json_object(response):
            return
        try:
            explanation = WorkloadExplanation.model_validate(response)
        except ValidationError:
            return
        verdict = "placeable now" if explanation.placeable else "not placeable"
        lines = [f"Eyvan: {explanation.workload_id} is {explanation.state.value}", f"  verdict     {verdict}"]
        if explanation.placement_cell_id is not None:
            lines.append(f"  cell        {explanation.placement_cell_id}")
        if explanation.message:
            prefix = f"{explanation.code.value}: " if explanation.code is not None else ""
            lines.append(f"  reason      {prefix}{explanation.message}")
        if explanation.considered_cell_count:
            shown = len(explanation.cells)
            omitted = explanation.truncated_cell_count
            summary = f"{explanation.considered_cell_count} rejected"
            if omitted:
                summary = f"{summary} (showing {shown}, {omitted} not shown)"
            lines.append(f"  cells       {summary}")
            lines.extend(f"    {cell.cell_id}  {cell.code.value}  {cell.message}" for cell in explanation.cells)
        if explanation.candidate_zones:
            # Why this zone and not that one. The belief state has decided it
            # for every cross-zone workload since alternates existed; printing
            # the evidence beside the choice is what makes it reviewable.
            lines.append("  zones")
            for zone in explanation.candidate_zones:
                marker = "->" if zone.preferred else "  "
                where = zone.zone or zone.pool_id
                evidence = []
                if zone.free_owned_cells:
                    evidence.append(f"{zone.free_owned_cells} already yours")
                if zone.free_shared_cells:
                    evidence.append(f"{zone.free_shared_cells} free")
                if zone.cooling_cells:
                    evidence.append(f"{zone.cooling_cells} cooling")
                if zone.consecutive_failures:
                    evidence.append(f"{zone.consecutive_failures} consecutive failures")
                lines.append(f"    {marker} {where}  {', '.join(evidence) or 'nothing free, nothing failing'}")
        if explanation.autoscale_block_reason is not None:
            lines.append(f"  autoscale   {explanation.autoscale_block_reason.value}")
        if explanation.queue_admission_denial is not None:
            lines.append(f"  queue       admission denied for {explanation.queue_admission_denial}")
        click.echo("\n".join(lines), file=stream)

    def mutate(
        self,
        workload_id: str,
        action: WorkloadAction,
        idempotency_key: str | None,
        *,
        as_json: bool = False,
    ) -> None:
        """Apply one workload lifecycle mutation and say what it did.

        Cancelling printed the sixty-key record. Somebody who has just cancelled
        a job wants to know it is cancelled; the document is for something
        parsing, and stays behind ``--json``.
        """

        response = self._mutation_response(workload_id, action, idempotency_key)
        emit_result(
            self.renderer,
            response,
            f"{_ACTION_HEADLINES[action]} {workload_id}",
            as_json=as_json,
            fields=("state", "attempt", "failure_reason"),
        )

    def _mutation_response(self, workload_id: str, action: WorkloadAction, idempotency_key: str | None) -> JsonValue:
        """Apply a mutation without rendering, for direct and batch output."""

        return self.session.client.request(
            "POST",
            f"/api/v1/workloads/{workload_id}/{action}",
            headers={"Idempotency-Key": self.keys.create(action, idempotency_key)},
        )

    def _scoped_workloads(self, scope: WorkloadScope) -> tuple[WorkloadRecord, ...]:
        """Read every workload this scope can see."""

        return self._workload_records(
            self.session.client.request(
                "GET",
                "/api/v1/workloads",
                params={"org_id": scope.org_id, "project_id": scope.project_id},
                headers=self.session.workload_headers(scope),
            )
        )

    def matching(self, prefix: str, *, scope: WorkloadScope) -> tuple[WorkloadRecord, ...]:
        """Return every workload whose id or name starts with ``prefix``.

        Name as well as id, because the id is a uuid and the name is the thing
        a person chose. A sweep like ``--prefix sweep-lr-`` is exactly the shape
        a batch of experiments has, and matching only ids would make it useless
        for the case it exists for.
        """

        if not prefix.strip():
            # Everything starts with the empty string, and this is the argument
            # to a cancel. Refused rather than treated as "all".
            raise click.UsageError("--prefix must not be empty")
        return tuple(
            workload
            for workload in self._scoped_workloads(scope)
            if workload.id.startswith(prefix) or workload.spec.name.startswith(prefix)
        )

    def resolve(self, reference: str, *, org_id: str | None, project_id: str | None) -> str:
        """Return the workload id ``reference`` names, whatever form it took.

        Exact ids are answered without a lookup, so nothing changes for the
        normal path or for a scope this client cannot list. A prefix or a name
        is resolved against the scope and must identify exactly one workload:
        an ambiguous reference names the candidates instead of picking one,
        because the caller is usually about to cancel something.
        """

        if (
            reference.startswith("wl_")
            and len(reference) == 35
            and all(character in "0123456789abcdef" for character in reference[3:])
        ):
            return reference
        scope = self.session.workload_scope(org_id=org_id, project_id=project_id)
        try:
            candidates = self.matching(reference, scope=scope)
        except Exception:
            # A scope this client cannot list is not a reason to refuse an id it
            # was given. The route decides whether it exists.
            return reference
        if not candidates:
            return reference
        if len(candidates) == 1:
            return candidates[0].id
        shown = ", ".join(f"{workload.id} ({workload.spec.name})" for workload in candidates[:5])
        more = "" if len(candidates) <= 5 else f", and {len(candidates) - 5} more"
        raise click.ClickException(
            f"{reference!r} matches {len(candidates)} workloads: {shown}{more}; "
            "pass one of the full workload ids instead"
        )

    def mutate_prefix(
        self,
        prefix: str,
        action: WorkloadAction,
        *,
        scope: WorkloadScope,
        confirmed: bool,
        idempotency_key: str | None = None,
        as_json: bool = False,
    ) -> None:
        """Apply one lifecycle mutation to every workload matching ``prefix``.

        Confirmation is required for more than one, and the matches are listed
        first. A batch cancel is the single most expensive mistake this CLI can
        make -- a mistyped prefix ends other people's training runs -- and the
        cost of showing what is about to happen is one line each.
        """

        matches = self.matching(prefix, scope=scope)
        if not matches:
            raise click.ClickException(f"no workload in {scope.org_id}/{scope.project_id} starts with {prefix!r}")
        if len(matches) > 1 and not confirmed:
            for workload in matches:
                click.echo(f"  {workload.id}  {workload.state.value:<12}{workload.spec.name}", err=True)
            raise click.ClickException(f"{action} would apply to {len(matches)} workloads; pass --yes to confirm")
        # A sweep is one operator intent, not every future action on these
        # workloads. Explicit keys allow replay across invocations; otherwise
        # create a fresh intent once, keeping each job's transport retries stable.
        intent = self.keys.create(action, idempotency_key)
        digest = hashlib.sha256(intent.encode()).hexdigest()
        responses: list[JsonValue] = []
        for workload in matches:
            key = f"{action}-{workload.id}-{digest}"
            if as_json:
                responses.append(self._mutation_response(workload.id, action, key))
            else:
                self.mutate(workload.id, action, key)
        if as_json:
            self.renderer.emit(responses)


@click.command("run")
@click.argument("target", required=False)
@click.option(
    "--tpu",
    "tpu_type",
    metavar="TYPE",
    help="TPU shape to run on, for example v6e-8. Use this instead of naming a pool: the fleet is asked which "
    "pool provides it, and pools of that shape in other zones become alternates.",
)
@click.option("--gpu", "gpu_type", metavar="TYPE", help="GPU shape to run on, for example nvidia-tesla-a100.")
@click.option("--cpu", "cpu_only", is_flag=True, help="Run on CPU capacity.")
@click.option(
    "--pods",
    type=int,
    default=1,
    show_default=True,
    help="How many slices to allocate together, atomically. The same shape TpuConfig(pods=N) asks for.",
)
@click.option(
    "--chips",
    type=int,
    default=None,
    help="Run on this many chips of one TPU host instead of the whole slice (any free ones). Needs a pool with "
    "chip_sharing: true. The same request TpuConfig(chips=N) makes.",
)
@click.option(
    "--chip-index",
    "chip_indexes",
    type=int,
    multiple=True,
    help="Run on exactly this chip (global index in the slice; repeatable, all on one host). The same request "
    "TpuConfig(chips_indexes=[...]) makes.",
)
@click.option(
    "--zone",
    help="Pin this job to one zone. A constraint, not a preference: without it the fleet places the job wherever "
    "its pools of this shape have capacity.",
)
@click.option(
    "--with",
    "dependencies",
    multiple=True,
    metavar="PACKAGE",
    help="Install this package on the host before the command runs, over whatever the image provides. "
    "Repeatable. Use instead of rebuilding an image to add one library.",
)
@click.option(
    "--with-requirements",
    "requirements_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Read packages from a requirements file and install them like --with. Read here, not on the host: "
    "the fleet never opens a path from your machine.",
)
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.option("--name")
@click.option("--env", "environment", multiple=True, metavar="KEY=VALUE")
@click.option(
    "--secret",
    "secret_mounts",
    multiple=True,
    metavar="NAME=REFERENCE",
    help="Mount a secret as an environment variable by reference (resolved at dispatch, never stored as a value).",
)
@click.option(
    "--setup",
    "setup_script",
    help="Bash setup step run on each host before the command (inline script or path to a local script file).",
)
@click.option(
    "--setup-url",
    "setup_url",
    metavar="HTTPS_URL",
    help="HTTPS URL of a bash setup script fetched and run on each host before the command.",
)
@click.option(
    "--easydel",
    "install_easydel",
    is_flag=True,
    default=False,
    help=f"Install EasyDeL before running (defaults to {EASYDEL_DEFAULT_BRANCH!r}; override with --easydel-branch).",
)
@click.option(
    "--easydel-branch",
    "easydel_branch",
    default=None,
    metavar="REF",
    help="EasyDeL branch, tag, or commit to install before running (implies --easydel).",
)
@click.option("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, show_default=True)
@click.option(
    "--max-failure-retries",
    type=int,
    default=0,
    show_default=True,
    help="Retries for the job's own non-zero exit. Separate from --max-retries, which pays for capacity the fleet lost.",
)
@click.option("--idempotency-key")
@click.option(
    "--workspace",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    is_flag=False,
    flag_value=".",
    default=None,
    help="Upload this directory (respecting .gitignore) and run the command inside it. Bare --workspace means '.'.",
)
@click.option(
    "--checkpointable",
    is_flag=True,
    default=False,
    help="This job can be checkpointed. Pair it with --checkpoint-hook: the scheduler only checkpoints on "
    "preemption when both are set, so on spot this is the difference between resuming and restarting from zero.",
)
@click.option("--checkpoint-hook", help="Command run on the cell to write a checkpoint before the job is stopped.")
@click.option("--checkpoint-output-uri", help="Where the hook writes, normally a gs:// prefix.")
@click.option(
    "--can-submit-children",
    is_flag=True,
    default=False,
    help="Hand this job a capability to submit children of its own, under this workload's owner.",
)
@click.option(
    "--priority-band",
    type=click.Choice(tuple(band.value for band in PriorityBand)),
    default=PriorityBand.INTERACTIVE.value,
    show_default=True,
    help="Which band this job competes in. A long training run belongs in batch, and the pool cannot know that.",
)
@click.option("--intra-band-priority", type=int, default=0, show_default=True)
@click.option(
    "--checkpoint-grace-seconds",
    type=int,
    default=0,
    show_default=True,
    help="Seconds the checkpoint hook is given before the job is stopped.",
)
@click.option(
    "--duration-s",
    "estimated_duration_seconds",
    type=int,
    default=3600,
    show_default=True,
    help="How long this job is expected to run. Every cost figure is an estimate over it.",
)
@click.option(
    "--named-asset",
    "named_assets",
    multiple=True,
    help="A large input this job pulls, so cells that recently pulled it are preferred. A hint, never a pin.",
)
@click.option(
    "--no-wait",
    "no_wait",
    is_flag=True,
    default=False,
    help="Print the submission document and return immediately, instead of streaming the job. Use this in scripts: "
    "stdout is then the JSON record and nothing else.",
)
@click.argument("command", nargs=-1, required=False, type=click.UNPROCESSED)
@click.pass_obj
def run_workload(
    session: CliSession,
    target: str | None,
    tpu_type: str | None,
    gpu_type: str | None,
    cpu_only: bool,
    pods: int,
    chips: int | None,
    chip_indexes: tuple[int, ...],
    zone: str | None,
    dependencies: tuple[str, ...],
    requirements_file: Path | None,
    org_id: str | None,
    project_id: str | None,
    name: str | None,
    environment: tuple[str, ...],
    secret_mounts: tuple[str, ...],
    setup_script: str | None,
    setup_url: str | None,
    install_easydel: bool,
    easydel_branch: str | None,
    max_retries: int,
    max_failure_retries: int,
    idempotency_key: str | None,
    workspace: Path | None,
    checkpointable: bool,
    checkpoint_hook: str | None,
    checkpoint_output_uri: str | None,
    can_submit_children: bool,
    priority_band: str,
    intra_band_priority: int,
    checkpoint_grace_seconds: int,
    estimated_duration_seconds: int,
    named_assets: tuple[str, ...],
    no_wait: bool,
    command: tuple[str, ...],
) -> None:
    """Queue COMMAND on hardware you name, or on TYPE / FLEET/TYPE.

    `--tpu v6e-8` asks the fleet which pool provides that shape, so nobody has
    to learn a pool-naming scheme to run anything. Naming TARGET still pins one
    pool exactly, which is what you want when the choice matters.

    The job's output streams to stdout and its progress to stderr, and this
    command exits 0 when it succeeded, 1 when it failed and 2 when it was
    cancelled. Ctrl-C detaches; it does not cancel, because the job is durable
    and outlives whoever submitted it. `--no-wait` prints the submission
    document and returns instead.
    """

    shapes = {"--tpu": tpu_type, "--gpu": gpu_type, "--cpu": "cpu" if cpu_only else None}
    named = {flag: value for flag, value in shapes.items() if value is not None}
    if len(named) > 1:
        raise click.UsageError(f"pass exactly one of {', '.join(sorted(named))}; a job runs on one kind of hardware")
    if named:
        # TARGET is a positional in front of a variadic COMMAND, so click hands
        # us the command's first word as the target. When the hardware is named
        # by flag there is no pool argument, and that word belongs to the
        # command -- otherwise `--tpu v6e-8 python train.py` would try to place
        # work on a pool called "python".
        if target is not None:
            command = (target, *command)
        target = next(iter(named.values()))

    if not command:
        raise click.UsageError("Missing argument 'COMMAND...'.")
    declared = _requirements(dependencies, requirements_file)
    setup_accel = "cuda" if gpu_type is not None else "cpu" if cpu_only else "tpu"
    resolved_setup = _resolve_cli_setup(
        setup=setup_script,
        setup_url=setup_url,
        easydel=install_easydel,
        easydel_branch=easydel_branch,
        accelerator=setup_accel,
    )
    exit_code = WorkloadCommandService(session).run(
        target_value=target,
        org_id=org_id,
        project_id=project_id,
        name=name,
        environment=environment,
        max_retries=max_retries,
        max_failure_retries=max_failure_retries,
        idempotency_key=idempotency_key,
        command=command,
        workspace=workspace,
        checkpointable=checkpointable,
        checkpoint_hook=checkpoint_hook,
        checkpoint_output_uri=checkpoint_output_uri,
        can_submit_children=can_submit_children,
        priority_band=priority_band,
        intra_band_priority=intra_band_priority,
        checkpoint_grace_seconds=checkpoint_grace_seconds,
        estimated_duration_seconds=estimated_duration_seconds,
        named_assets=named_assets,
        pods=pods,
        zone=zone,
        dependencies=declared,
        wait=not no_wait,
        chips=chips,
        chip_indexes=chip_indexes,
        secret_mounts=secret_mounts,
        setup=resolved_setup,
    )
    if exit_code:
        # The job's outcome is this command's outcome, so `eyvan run ... && ...`
        # means what it looks like it means.
        raise SystemExit(exit_code)


@click.command("exec")
@click.argument("workload_id")
@click.option("--rank", type=int, default=None, help="Run on one rank only. Omitted runs on every rank.")
@click.option("--timeout", "timeout_s", type=float, default=30.0, show_default=True)
@click.argument("command", nargs=-1, required=True, type=click.UNPROCESSED)
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def exec_in_workload(
    session: CliSession,
    workload_id: str,
    rank: int | None,
    timeout_s: float,
    command: tuple[str, ...],
    as_json: bool,
) -> None:
    """Run COMMAND beside a running workload, on every rank by default.

    Beside it, not inside its cgroup: a process there competes for the memory
    the trainer is sized against, and its death can take the trainer with it.
    Every rank by default because the question this answers is usually "which
    rank is stuck", and one rank cannot distinguish a hung collective from a
    slow step.
    """

    service = WorkloadCommandService(session)
    response = session.client.request(
        "POST",
        f"/api/v1/workloads/{workload_id}/exec",
        json={"command": list(command), "timeout_s": timeout_s, "rank": rank},
    )
    if is_json_object(response) and response.get("output_suppressed") is True and response.get("results") == []:
        emit_result(service.renderer, response, "Exec not available", as_json=as_json)
        click.echo("exec is not available for workloads with secret mounts; nothing ran", err=True)
        raise SystemExit(1)
    if as_json:
        service.renderer.emit(response)
    else:
        where = "every rank" if rank is None else f"rank {rank}"
        click.echo(f"Ran {' '.join(command)} on {where}")
        _render_exec_results(response)
    failed = _failed_exec_ranks(response)
    if failed:
        # The document above is complete either way; the exit status is what a
        # script branches on. A refused diagnostic must exit non-zero so callers
        # can distinguish it from success without parsing the body.
        click.echo(f"exec failed on rank(s) {', '.join(str(index) for index in failed)}", err=True)
        raise SystemExit(1)


def _render_exec_entry(label: str, entry: JsonValue) -> None:
    """Render one rank/host exec answer with exit status, error, stdout, and stderr."""

    if not is_json_object(entry):
        return
    if entry.get("output_suppressed") is True:
        click.echo(f"[{label}] output suppressed")
        return
    error = entry.get("error")
    if error is not None:
        click.echo(f"[{label}] error: {error}", err=True)
        return
    if entry.get("observed") is False:
        detail = entry.get("detail") or "unobserved"
        click.echo(f"[{label}] unobserved: {detail}", err=True)
        return
    exit_code = entry.get("exit_code")
    has_code = isinstance(exit_code, int) and not isinstance(exit_code, bool)
    header = f"[{label}] exit={exit_code}" if has_code else f"[{label}]"
    click.echo(header)
    stdout = entry.get("stdout")
    if isinstance(stdout, str) and stdout:
        click.echo(stdout, nl=not stdout.endswith("\n"))
    stderr = entry.get("stderr")
    if isinstance(stderr, str) and stderr:
        click.echo(stderr, err=True, nl=not stderr.endswith("\n"))


def _render_exec_results(response: JsonValue) -> None:
    """Print per-rank command outputs for ``eyvan workload exec`` human mode."""

    if not is_json_object(response):
        return
    results = response.get("results")
    if not isinstance(results, list):
        return
    for index, item in enumerate(results):
        if not is_json_object(item):
            continue
        rank = item.get("rank")
        rank_label = f"rank {rank if isinstance(rank, int) and not isinstance(rank, bool) else index}"
        hosts = item.get("ranks")
        if isinstance(hosts, list) and hosts:
            for host_index, host in enumerate(hosts):
                if not is_json_object(host):
                    continue
                sub_rank = host.get("rank")
                host_label = (
                    f"{rank_label}.{sub_rank}"
                    if isinstance(sub_rank, int) and not isinstance(sub_rank, bool)
                    else f"{rank_label}.{host_index}"
                )
                _render_exec_entry(host_label, host.get("result"))
        else:
            _render_exec_entry(rank_label, item)


def _failed_exec_ranks(response: JsonValue) -> list[int]:
    """Return the ranks whose exec was refused, unreachable, or exited non-zero.

    A slice or gang answers for its hosts under ``ranks``; one failed host
    fails the rank that owns it. An absent exit code alone is not a failure:
    suppressed output withholds it on purpose.
    """

    if not is_json_object(response):
        return []
    results = response.get("results")
    if not isinstance(results, list):
        return []
    failed: list[int] = []
    for index, result in enumerate(results):
        if not is_json_object(result):
            continue
        rank = result.get("rank")
        if _exec_result_failed(result):
            failed.append(rank if isinstance(rank, int) else index)
    return failed


def _exec_result_failed(result: JsonValue) -> bool:
    """Return whether one exec answer, or any host nested under it, failed."""

    if not is_json_object(result):
        return False
    exit_code = result.get("exit_code")
    if result.get("error") is not None or result.get("observed") is False:
        return True
    if isinstance(exit_code, int) and not isinstance(exit_code, bool) and exit_code != 0:
        return True
    hosts = result.get("ranks")
    return isinstance(hosts, list) and any(
        _exec_result_failed(host.get("result")) for host in hosts if is_json_object(host)
    )


@click.group("workload")
def workload() -> None:
    """Submit and operate whole workloads."""


def _render_progress_lines(response: dict[str, JsonValue]) -> list[str]:
    """Render one ``WorkloadProgress`` payload as a concise multi-line summary."""

    wid = str(response.get("workload_id") or "")
    state = str(response.get("workload_state") or "unknown")
    phase = str(response.get("phase") or "")
    header = f"{wid}  {state}" if not phase else f"{wid}  {state}  ({phase})"
    lines = [header]
    pool = response.get("pool")
    if is_json_object(pool):
        pool_id = pool.get("pool_id") or pool.get("fleet_name") or pool.get("id")
        pool_state = pool.get("state") or pool.get("status")
        if pool_id:
            suffix = f" ({pool_state})" if pool_state else ""
            lines.append(f"  pool      {pool_id}{suffix}")
    capacity = response.get("capacity")
    if is_json_object(capacity):
        ready = capacity.get("ready_cells")
        total = capacity.get("total_cells")
        if isinstance(ready, int) and isinstance(total, int):
            lines.append(f"  capacity  {ready}/{total} ready cells")
    for label, key in (
        ("reason", "scheduling_reason"),
        ("failure", "failure_reason"),
        ("blocked", "autoscale_block_reason"),
    ):
        value = response.get(key)
        if isinstance(value, str) and value:
            lines.append(f"  {label:<8}  {value}")
    for entry in format_user_progress(response.get("user_progress")):
        lines.append(f"  progress  {entry}")
    return lines


@workload.command("children")
@click.argument("workload_id")
@click.option("--json", "as_json", is_flag=True, help="Emit the child workload records as JSON.")
@click.pass_obj
def list_children(session: CliSession, workload_id: str, as_json: bool) -> None:
    """List the workloads one workload submitted or fanned out to.

    The only view of a job tree. Without it a batch, a DAG or an in-job fan-out
    is a parent id and a guess: every child is an ordinary workload in the
    listing, indistinguishable from work a human submitted.
    """

    response = session.client.request("GET", f"/api/v1/workloads/{workload_id}/children")
    if as_json or not isinstance(response, list):
        JsonRenderer().emit(response)
        return
    columns = (
        Column("id", table_field("id")),
        Column("name", table_field("spec", "name")),
        Column("state", table_field("state")),
        Column("attempt", table_field("attempt"), align=">"),
        Column("pool", table_field("assigned_pool_id")),
    )
    for line in render_table(response, columns, empty=f"No child workloads for {workload_id}."):
        click.echo(line)


@workload.command("endpoints")
@click.argument("workload_id")
@click.option("--json", "as_json", is_flag=True, help="Emit the endpoint records as JSON.")
@click.pass_obj
def list_endpoints(session: CliSession, workload_id: str, as_json: bool) -> None:
    """List the addresses a service's replicas registered.

    Registration is the readiness signal -- an app says it can serve *after* it
    can -- so this is also the answer to "is the service actually up", which no
    other view gives.
    """

    response = session.client.request("GET", f"/api/v1/workloads/{workload_id}/endpoints")
    if as_json or not isinstance(response, list):
        JsonRenderer().emit(response)
        return
    columns = (
        Column("workload", table_field("workload_id")),
        Column("endpoint", table_field("endpoint")),
        Column("revision", table_field("revision")),
    )
    for line in render_table(response, columns, empty=f"No active endpoints for {workload_id}."):
        click.echo(line)


@workload.command("progress")
@click.argument("workload_id")
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.option("--json", "as_json", is_flag=True, help="Emit the progress document as JSON.")
@click.pass_obj
def workload_progress(
    session: CliSession,
    workload_id: str,
    org_id: str | None,
    project_id: str | None,
    as_json: bool,
) -> None:
    """Show autoscaling, worker, and in-job progress for WORKLOAD_ID."""

    service = WorkloadCommandService(session)
    resolved = service.resolve(workload_id, org_id=org_id, project_id=project_id)
    response = session.client.request("GET", f"/api/v1/workloads/{resolved}/progress")
    if as_json or not is_json_object(response):
        service.renderer.emit(response)
        return
    for line in _render_progress_lines(response):
        click.echo(line)


@workload.command("log-page")
@click.argument("workload_id")
@click.option("--after", default="", help="Resume cursor from a previous page.")
@click.option("--limit", type=int, default=200, show_default=True)
@click.pass_obj
def log_page(session: CliSession, workload_id: str, after: str, limit: int) -> None:
    """Read one structured, per-rank page of a workload's logs.

    Distinct from ``workload logs``, which returns concatenated text with rank
    banners. This keeps records attributed and resumable, which is what a
    multi-host job needs and what a banner cannot express.
    """

    JsonRenderer().emit(
        session.client.request(
            "GET",
            f"/api/v1/workloads/{workload_id}/log-page",
            params={"after": after, "limit": str(limit)},
        )
    )


@workload.command("estimate")
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.option("--name", default="estimate", show_default=True)
@click.option("--command", default="true", show_default=True, help="Not run; a spec needs one to validate.")
@click.option("--cpu", type=float, default=1.0, show_default=True)
@click.option("--memory-gib", type=float, default=1.0, show_default=True)
@click.option("--gpu", type=int, default=0, show_default=True)
@click.option("--tpu", type=int, default=0, show_default=True)
@click.option("--replicas", type=int, default=1, show_default=True)
@click.option(
    "--capacity-class",
    type=click.Choice(tuple(capacity.value for capacity in CapacityClass)),
    default=CapacityClass.ON_DEMAND.value,
    show_default=True,
)
@click.option("--accelerator-type")
@click.option("--provider")
@click.option("--region")
@click.option(
    "--duration-s",
    "estimated_duration_seconds",
    type=int,
    default=3600,
    show_default=True,
    help="How long the job is expected to run. Every figure is an estimate over this number, so the answer is only "
    "as honest as it is.",
)
@click.pass_obj
def estimate_workload(
    session: CliSession,
    org_id: str | None,
    project_id: str | None,
    name: str,
    command: str,
    cpu: float,
    memory_gib: float,
    gpu: int,
    tpu: int,
    replicas: int,
    capacity_class: str,
    accelerator_type: str | None,
    provider: str | None,
    region: str | None,
    estimated_duration_seconds: int,
) -> None:
    """Ask what a workload would cost before submitting it.

    The route has existed since budgets did; nothing outside the raw HTTP API
    could reach it, so the question it answers still required submitting a job
    and reading the resulting lease.

    Not a scheduling dry run. It consults declared pool attributes and prices,
    never live cell availability -- an estimate that claimed to predict placement
    would be wrong the moment the fleet moved.
    """

    scope = session.workload_scope(org_id=org_id, project_id=project_id)
    try:
        request = WorkloadEstimateRequest(
            org_id=scope.org_id,
            project_id=scope.project_id,
            spec=WorkloadDefinition(
                name=name,
                execution=CommandExecution(kind="command", command=command),
                gang=ResourceGang(
                    replicas=replicas,
                    per_replica=ResourceVector(cpu=cpu, memory_gib=memory_gib, gpu=gpu, tpu=tpu),
                ),
                capacity_class=CapacityClass(capacity_class),
                accelerator_type=accelerator_type,
                provider=provider,
                region=region,
                estimated_duration_seconds=estimated_duration_seconds,
            ),
        )
    except (ValidationError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    JsonRenderer().emit(
        session.client.request(
            "POST",
            "/api/v1/workloads/estimate",
            json=model_json(request),
            headers=session.workload_headers(scope),
        )
    )


@workload.command("submit")
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.option("--name", required=True)
@click.option("--command", required=True)
@click.option("--cpu", type=float, default=1.0, show_default=True)
@click.option("--memory-gib", type=float, default=1.0, show_default=True)
@click.option("--gpu", type=int, default=0, show_default=True)
@click.option("--tpu", type=int, default=0, show_default=True)
@click.option("--replicas", type=int, default=1, show_default=True)
@click.option(
    "--capacity-class",
    type=click.Choice(tuple(capacity.value for capacity in CapacityClass)),
    default=CapacityClass.ON_DEMAND.value,
    show_default=True,
)
@click.option("--accelerator-type")
@click.option("--provider")
@click.option("--region")
@click.option(
    "--additional-capacity-class",
    "additional_capacity_classes",
    multiple=True,
    type=click.Choice(tuple(capacity.value for capacity in CapacityClass)),
    help="Further capacity classes this job will accept, in preference order after --capacity-class. This is how a "
    "spot-first job says it would rather fall back than wait: the scheduler has always ranked a portfolio.",
)
@click.option(
    "--allowed-region",
    "allowed_regions",
    multiple=True,
    help="Regions this job will accept. Mutually exclusive with --region, which pins one.",
)
@click.option(
    "--allowed-zone",
    "allowed_zones",
    multiple=True,
    help="Zones this job will accept. Same rule as --allowed-region.",
)
@click.option(
    "--allowed-accelerator-type",
    "allowed_accelerator_types",
    multiple=True,
    help="Accelerator types this job will accept. Mutually exclusive with --accelerator-type.",
)
@click.option(
    "--named-asset",
    "named_assets",
    multiple=True,
    help="Logical names of large inputs this job pulls -- a dataset URI, a checkpoint, an image. Only a hint: cells "
    "that recently ran something naming the same asset are preferred, so a stale guess costs one pull.",
)
@click.option("--required-executor")
@click.option(
    "--priority-band",
    type=click.Choice(tuple(band.value for band in PriorityBand)),
    default=PriorityBand.INTERACTIVE.value,
    show_default=True,
    help="Scheduling band. A higher band may preempt a lower one for its capacity.",
)
@click.option(
    "--intra-band-priority",
    type=int,
    default=0,
    show_default=True,
    help="Rank within the band; higher wins.",
)
@click.option("--checkpointable", is_flag=True)
@click.option("--checkpoint-hook")
@click.option("--checkpoint-output-uri")
@click.option("--max-retries", type=int, default=0, show_default=True)
@click.option(
    "--max-failure-retries",
    type=int,
    default=0,
    show_default=True,
    help="Retries for the job's own non-zero exit. Separate from --max-retries, which pays for capacity the fleet lost.",
)
@click.option(
    "--max-preemption-retries",
    type=int,
    default=10,
    show_default=True,
    help="Restarts allowed after the provider reclaims the capacity. Separate from --max-retries because a spot "
    "reclamation is expected and cheap to retry, while a stockout means the pool is empty.",
)
@click.option("--env", "environment", multiple=True, metavar="KEY=VALUE", help="Environment variable (repeatable).")
@click.option(
    "--with",
    "dependencies",
    multiple=True,
    metavar="PACKAGE",
    help="Install this package on the host before the command runs (repeatable).",
)
@click.option(
    "--with-requirements",
    "requirements_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Read packages from a local requirements file and install them like --with.",
)
@click.option(
    "--workspace",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    is_flag=False,
    flag_value=".",
    default=None,
    help="Upload this directory (respecting .gitignore) and run the command inside it. Bare --workspace means '.'.",
)
@click.option(
    "--setup",
    "setup_script",
    help="Bash setup step run on each host before the command (inline script or path to a local script file).",
)
@click.option(
    "--setup-url",
    "setup_url",
    metavar="HTTPS_URL",
    help="HTTPS URL of a bash setup script fetched and run on each host before the command.",
)
@click.option(
    "--easydel",
    "install_easydel",
    is_flag=True,
    default=False,
    help=f"Install EasyDeL before running (defaults to {EASYDEL_DEFAULT_BRANCH!r}; override with --easydel-branch).",
)
@click.option(
    "--easydel-branch",
    "easydel_branch",
    default=None,
    metavar="REF",
    help="EasyDeL branch, tag, or commit to install before running (implies --easydel).",
)
@click.option(
    "--secret",
    "secret_mounts",
    multiple=True,
    metavar="NAME=REFERENCE",
    help=(
        "Mount a secret as an environment variable, by reference. The reference is resolved at dispatch and the "
        "value is never stored in the spec -- unlike --env, which is persisted. The project policy's "
        "allowed_secret_references must permit the reference."
    ),
)
@click.option(
    "--can-submit-children",
    is_flag=True,
    default=False,
    help=(
        "Hand this job a capability to submit children of its own. Children run under this workload's owner "
        "rather than the workload, inherit its scope and quota, are cancelled with it, and are capped at 256 "
        "live at once. Off by default: a credential nobody asked for is one nobody is watching."
    ),
)
@click.option("--idempotency-key")
@click.pass_obj
def submit_workload(
    session: CliSession,
    org_id: str | None,
    project_id: str | None,
    name: str,
    command: str,
    cpu: float,
    memory_gib: float,
    gpu: int,
    tpu: int,
    replicas: int,
    capacity_class: str,
    accelerator_type: str | None,
    provider: str | None,
    region: str | None,
    additional_capacity_classes: tuple[str, ...],
    allowed_regions: tuple[str, ...],
    allowed_zones: tuple[str, ...],
    allowed_accelerator_types: tuple[str, ...],
    named_assets: tuple[str, ...],
    required_executor: str | None,
    priority_band: str,
    intra_band_priority: int,
    checkpointable: bool,
    checkpoint_hook: str | None,
    checkpoint_output_uri: str | None,
    max_retries: int,
    max_failure_retries: int,
    max_preemption_retries: int,
    environment: tuple[str, ...],
    dependencies: tuple[str, ...],
    requirements_file: Path | None,
    workspace: Path | None,
    setup_script: str | None,
    setup_url: str | None,
    install_easydel: bool,
    easydel_branch: str | None,
    secret_mounts: tuple[str, ...],
    can_submit_children: bool,
    idempotency_key: str | None,
) -> None:
    """Submit a workload with explicit resource and placement intent."""

    service = WorkloadCommandService(session)
    scope = session.workload_scope(org_id=org_id, project_id=project_id)
    mounts = parse_environment(secret_mounts, option="--secret")
    env_map = parse_environment(environment)
    declared = _requirements(dependencies, requirements_file)
    setup_accel = "tpu" if tpu > 0 else "cuda" if gpu > 0 else "cpu"
    resolved_setup = _resolve_cli_setup(
        setup=setup_script,
        setup_url=setup_url,
        easydel=install_easydel,
        easydel_branch=easydel_branch,
        accelerator=setup_accel,
    )
    workspace_ref = (
        service._upload_workspace(
            workspace,
            scope,
            prefer_remote=bool(provider == "gcp" or tpu > 0 or gpu > 0),
        )
        if workspace is not None
        else None
    )
    optional_fields: dict[str, tp.Any] = {}
    if workspace_ref is not None:
        optional_fields["workspace"] = workspace_ref
    if env_map:
        optional_fields["environment"] = env_map
    if declared:
        optional_fields["dependencies"] = declared
    if resolved_setup is not None:
        optional_fields["setup"] = resolved_setup
    try:
        request = WorkloadSubmitRequest(
            org_id=scope.org_id,
            project_id=scope.project_id,
            spec=WorkloadDefinition(
                name=name,
                execution=CommandExecution(kind="command", command=command),
                gang=ResourceGang(
                    replicas=replicas,
                    per_replica=ResourceVector(cpu=cpu, memory_gib=memory_gib, gpu=gpu, tpu=tpu),
                ),
                capacity_class=CapacityClass(capacity_class),
                additional_capacity_classes=tuple(CapacityClass(value) for value in additional_capacity_classes),
                accelerator_type=accelerator_type,
                provider=provider,
                region=region,
                allowed_regions=allowed_regions,
                allowed_zones=allowed_zones,
                allowed_accelerator_types=allowed_accelerator_types,
                named_assets=named_assets,
                required_executor=required_executor,
                priority_band=PriorityBand(priority_band),
                intra_band_priority=intra_band_priority,
                checkpointable=checkpointable,
                checkpoint_hook=checkpoint_hook,
                checkpoint_output_uri=checkpoint_output_uri,
                max_retries=max_retries,
                max_failure_retries=max_failure_retries,
                max_preemption_retries=max_preemption_retries,
                secret_mounts=mounts,
                can_submit_children=can_submit_children,
                **optional_fields,
            ),
        )
    except (ValidationError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    service.submit(request, scope=scope, idempotency_key=idempotency_key)


@workload.command("submit-batch")
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.option("--name", required=True)
@click.option("--command", required=True)
@click.option("--count", type=int, required=True, help="Number of tasks to create.")
@click.option("--name-prefix", default="", help="Task names become {prefix}-{index}; defaults to --name.")
@click.option("--cpu", type=float, default=0.0, show_default=True)
@click.option("--memory-gib", type=float, default=0.0, show_default=True)
@click.option("--gpu", type=int, default=0, show_default=True)
@click.option("--tpu", type=int, default=0, show_default=True)
@click.option("--replicas", type=int, default=1, show_default=True)
@click.option(
    "--capacity-class",
    type=click.Choice(tuple(capacity.value for capacity in CapacityClass)),
    default=CapacityClass.ON_DEMAND.value,
    show_default=True,
)
@click.option("--accelerator-type")
@click.option("--max-fan-out", type=int, default=DEFAULT_MAX_FAN_OUT, show_default=True)
@click.option("--idempotency-key")
@click.pass_obj
def submit_batch(
    session: CliSession,
    org_id: str | None,
    project_id: str | None,
    name: str,
    command: str,
    count: int,
    name_prefix: str,
    cpu: float,
    memory_gib: float,
    gpu: int,
    tpu: int,
    replicas: int,
    capacity_class: str,
    accelerator_type: str | None,
    max_fan_out: int,
    idempotency_key: str | None,
) -> None:
    """Submit an array of N identical tasks, each with EYVAN_BATCH_INDEX set.

    The whole array commits in one transaction. The returned parent is a
    coordinator that is never scheduled; cancel it to cancel every task.
    """

    scope = session.workload_scope(org_id=org_id, project_id=project_id)
    try:
        request = BatchSubmitRequest(
            org_id=scope.org_id,
            project_id=scope.project_id,
            template=WorkloadDefinition(
                name=name,
                execution=CommandExecution(kind="command", command=command),
                gang=ResourceGang(
                    replicas=replicas,
                    per_replica=ResourceVector(cpu=cpu, memory_gib=memory_gib, gpu=gpu, tpu=tpu),
                ),
                capacity_class=CapacityClass(capacity_class),
                accelerator_type=accelerator_type,
            ),
            count=count,
            name_prefix=name_prefix,
            max_fan_out=max_fan_out,
        )
    except (ValidationError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    WorkloadCommandService(session).submit_batch(request, scope=scope, idempotency_key=idempotency_key)


def _parse_dag_nodes(document: object) -> tuple[DagNodeRequest, ...]:
    """Build node requests from a parsed pipeline document.

    Structured input gets a file rather than repeated flags: a DAG is a set of
    edges, and every flag syntax that encodes edges is harder to read than the
    YAML it replaces.
    """

    if not isinstance(document, dict) or not isinstance(document.get("nodes"), list):
        raise click.ClickException("pipeline file must be a mapping with a 'nodes' list")
    nodes: list[DagNodeRequest] = []
    for index, entry in enumerate(document["nodes"]):
        if not isinstance(entry, dict):
            raise click.ClickException(f"pipeline node {index} must be a mapping")
        name = entry.get("name")
        command = entry.get("command")
        if not isinstance(name, str) or not isinstance(command, str):
            raise click.ClickException(f"pipeline node {index} needs string 'name' and 'command' fields")
        depends_on = entry.get("depends_on", [])
        if not isinstance(depends_on, list) or not all(isinstance(item, str) for item in depends_on):
            raise click.ClickException(f"pipeline node {name!r}: 'depends_on' must be a list of node names")
        resources = entry.get("resources", {})
        if not isinstance(resources, dict):
            raise click.ClickException(f"pipeline node {name!r}: 'resources' must be a mapping")
        try:
            nodes.append(
                DagNodeRequest(
                    name=name,
                    template=WorkloadDefinition(
                        name=name,
                        execution=CommandExecution(kind="command", command=command),
                        gang=ResourceGang(
                            replicas=entry.get("replicas", 1),
                            per_replica=ResourceVector(
                                cpu=float(resources.get("cpu", 0.0)),
                                memory_gib=float(resources.get("memory_gib", 0.0)),
                                gpu=resources.get("gpu", 0),
                                tpu=resources.get("tpu", 0),
                            ),
                        ),
                        capacity_class=CapacityClass(entry.get("capacity_class", CapacityClass.ON_DEMAND.value)),
                        accelerator_type=entry.get("accelerator_type"),
                    ),
                    depends_on=tuple(depends_on),
                )
            )
        except (ValidationError, ValueError) as exc:
            raise click.ClickException(f"pipeline node {name!r}: {exc}") from exc
    return tuple(nodes)


@workload.command("submit-dag")
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.option(
    "--file",
    "pipeline_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="YAML or JSON pipeline document.",
)
@click.option("--name-prefix", default="", help="Node names become {prefix}-{node}.")
@click.option("--max-fan-out", type=int, default=DEFAULT_MAX_FAN_OUT, show_default=True)
@click.option("--idempotency-key")
@click.pass_obj
def submit_dag(
    session: CliSession,
    org_id: str | None,
    project_id: str | None,
    pipeline_file: str,
    name_prefix: str,
    max_fan_out: int,
    idempotency_key: str | None,
) -> None:
    """Submit a pipeline of dependent workloads from a file, in one transaction.

    \b
    nodes:
      - name: prepare
        command: python prepare.py
      - name: train
        command: python train.py
        depends_on: [prepare]
        replicas: 4
        accelerator_type: v5p-16
        resources: {tpu: 4}

    A node runs only after every node it depends on has SUCCEEDED. If one fails
    or is cancelled, its dependents are cancelled rather than left waiting.
    Dependencies order execution only -- results travel as eyvan:// URIs.
    """

    scope = session.workload_scope(org_id=org_id, project_id=project_id)
    try:
        with Path(pipeline_file).expanduser().open(encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise click.ClickException(f"could not read {pipeline_file}: {exc}") from exc
    try:
        request = DagSubmitRequest(
            org_id=scope.org_id,
            project_id=scope.project_id,
            nodes=_parse_dag_nodes(document),
            name_prefix=name_prefix,
            max_fan_out=max_fan_out,
        )
    except (ValidationError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    WorkloadCommandService(session).submit_dag(request, scope=scope, idempotency_key=idempotency_key)


@workload.command("list")
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.option("--state", type=click.Choice(tuple(state.value for state in WorkloadState)))
@click.pass_obj
def list_workloads(
    session: CliSession,
    org_id: str | None,
    project_id: str | None,
    state: str | None,
) -> None:
    """List workloads in a project, optionally filtered by lifecycle state."""

    scope = session.workload_scope(org_id=org_id, project_id=project_id)
    WorkloadCommandService(session).list(scope, None if state is None else WorkloadState(state))


@workload.command("get")
@click.argument("workload_id")
@click.pass_obj
def get_workload(session: CliSession, workload_id: str) -> None:
    """Get one workload record.

    The record, verbatim. `status` is the same fetch rendered for a person;
    this is the accessor, and something is piping it.
    """

    WorkloadCommandService(session).show(workload_id, as_json=True)


@workload.command("status")
@click.argument("workload_id")
@click.option("--json", "as_json", is_flag=True, help="Emit the durable record instead of a summary.")
@click.pass_obj
def workload_status(session: CliSession, workload_id: str, as_json: bool) -> None:
    """Show what one workload is, what it runs, and why it is or is not running."""

    WorkloadCommandService(session).show(workload_id, as_json=as_json)


@workload.command("timeline")
@click.argument("workload_id")
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.option("--limit", type=int, default=200, show_default=True, help="Most events to read per entity.")
@click.pass_obj
def workload_timeline(
    session: CliSession,
    workload_id: str,
    org_id: str | None,
    project_id: str | None,
    limit: int,
) -> None:
    """Show what happened to WORKLOAD_ID, in order, with elapsed times.

    Submitted, admitted, leased, dispatched, preempted, retried, settled -- one
    column, each line offset from the first. Every fact here was already in the
    audit ledger; what was missing was reading them together, which is how
    "eight minutes queued, thirty provisioning, two preemptions" becomes
    visible at all.
    """

    WorkloadCommandService(session).timeline(workload_id, org_id=org_id, project_id=project_id, limit=limit)


@workload.command("cost")
@click.argument("workload_id")
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.option(
    "--on-demand-price",
    "on_demand_hourly",
    type=float,
    help="On-demand price per accelerator-hour, to compare against. Eyvan carries no price catalogue, so without "
    "an on-demand pool declaring one this is the only way to see the comparison.",
)
@click.pass_obj
def workload_cost(
    session: CliSession,
    workload_id: str,
    org_id: str | None,
    project_id: str | None,
    on_demand_hourly: float | None,
) -> None:
    """Show what WORKLOAD_ID cost, and what it would have cost on demand.

    The sentence a spot-first scheduler exists to say. Accelerator-hours are
    measured and always known; money is reported only where a pool declares a
    price, and the on-demand comparison only where somebody declared that
    price too -- an invented saving is the figure a team puts in a budget.
    """

    WorkloadCommandService(session).cost(
        workload_id,
        org_id=org_id,
        project_id=project_id,
        on_demand_hourly=on_demand_hourly,
    )


@workload.command("explain")
@click.argument("workload_id")
@click.pass_obj
def workload_explain(session: CliSession, workload_id: str) -> None:
    """Explain why one workload is or is not currently placeable."""

    WorkloadCommandService(session).explain(workload_id)


@workload.command("process-status")
@click.argument("workload_id")
@click.pass_obj
def workload_process_status(session: CliSession, workload_id: str) -> None:
    """Show live process detail for a workload attempt.

    Use this when a job is RUNNING but not progressing: it reports each rank's
    systemd state and cgroup process list, which distinguishes a healthy step
    loop from a hung collective.
    """

    WorkloadCommandService(session).process_status(workload_id)


@workload.command("logs")
@click.argument("workload_id")
@click.option(
    "--follow",
    "-f",
    is_flag=True,
    default=False,
    help="Keep printing output until the workload stops, then exit with its outcome.",
)
@click.option(
    "--since",
    metavar="DURATION",
    help="Show only output newer than this, spelled 30s, 10m, 2h or 1d.",
)
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.pass_obj
def workload_logs(
    session: CliSession,
    workload_id: str,
    follow: bool,
    since: str | None,
    org_id: str | None,
    project_id: str | None,
) -> None:
    """Print driver logs for a workload attempt.

    Reads the live cell while one exists and this attempt's archive once it
    does not, so a preempted or finished job is still readable. With --follow
    this reattaches to a running job: output on stdout, progress on stderr, and
    an exit status of 0, 1 or 2 for succeeded, failed or cancelled.

    ``--since`` bounds what is printed, not what is read: a log position is an
    opaque backend cursor and none of the journals Eyvan reads can seek by
    time, so the window is applied here rather than pretended at the source.
    With ``--since`` but no ``--follow`` the window is printed once and the
    command returns; it does not wait for the job to stop.
    """

    service = WorkloadCommandService(session)
    resolved = service.resolve(workload_id, org_id=org_id, project_id=project_id)
    try:
        window = None if since is None else parse_duration(since)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    if not follow and window is None:
        click.echo(session.client.request_text("GET", f"/api/v1/workloads/{resolved}/logs"))
        return
    if not follow:
        # ``--since`` is documented as a filter on what is printed, so on its
        # own it must not attach: the follower polls until the workload stops,
        # and on a queued or running job that is a hang where a dump was asked
        # for. One drain of the paged log -- windowed exactly the way the
        # follower windows it -- and done.
        WorkloadFollower(
            client=session.client,
            output=JobTerminalOutput(stream=click.get_text_stream("stderr")),
            write=lambda line: click.echo(line),
            since=window,
        )._drain(resolved)
        return
    exit_code = service.follow(resolved, since=window)
    if exit_code:
        raise SystemExit(exit_code)


@click.command("kick")
@click.argument("workload_id", required=False)
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.option("--prefix", help="Apply to every workload in this scope whose id or name starts with PREFIX.")
@click.option(
    "--yes",
    "-y",
    "confirmed",
    is_flag=True,
    default=False,
    help="Do not ask before kicking more than one workload.",
)
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def kick_workload(
    session: CliSession,
    workload_id: str | None,
    org_id: str | None,
    project_id: str | None,
    prefix: str | None,
    confirmed: bool,
    as_json: bool,
) -> None:
    """Interrupt this attempt and let WORKLOAD_ID reschedule elsewhere.

    For an attempt wedged on a host that is not coming back while the job itself
    is fine. ``cancel`` ends the job; waiting for a timeout keeps the accelerator
    busy doing nothing until it fires.

    Charged to the preemption budget rather than the failure budget: a spot job
    sets ``--max-retries 0`` because it expects to be interrupted, so charging a
    kick there would end the job you were trying to move.

    WORKLOAD_ID may be a full id, a unique id prefix or a name; ``--prefix``
    kicks every match, which is what a zone going bad under a sweep looks like.
    """

    service = WorkloadCommandService(session)
    if prefix is not None:
        if workload_id is not None:
            raise click.UsageError("pass a workload id or --prefix, not both")
        scope = session.workload_scope(org_id=org_id, project_id=project_id)
        matches = service.matching(prefix, scope=scope)
        if not matches:
            raise click.ClickException(f"no workload in {scope.org_id}/{scope.project_id} starts with {prefix!r}")
        if len(matches) > 1 and not confirmed:
            for match in matches:
                click.echo(f"  {match.id}  {match.state.value:<12}{match.spec.name}", err=True)
            raise click.ClickException(f"kick would apply to {len(matches)} workloads; pass --yes to confirm")
        responses: list[JsonValue] = []
        for match in matches:
            response = session.client.request("POST", f"/api/v1/workloads/{match.id}/kick")
            if as_json:
                responses.append(response)
            else:
                emit_result(service.renderer, response, f"Kicked {match.id}", as_json=False)
        if as_json:
            service.renderer.emit(responses)
        return
    if workload_id is None:
        raise click.UsageError("give a workload id, or --prefix to match several")
    resolved = service.resolve(workload_id, org_id=org_id, project_id=project_id)
    response = session.client.request("POST", f"/api/v1/workloads/{resolved}/kick")
    emit_result(service.renderer, response, f"Kicked {resolved}", as_json=as_json)


def _workload_action_command(action: WorkloadAction) -> click.Command:
    @click.command(action)
    @click.argument("workload_id", required=False)
    @click.option("--idempotency-key")
    @click.option("--org", "org_id")
    @click.option("--project", "project_id")
    @click.option(
        "--prefix",
        help="Apply to every workload in this scope whose id or name starts with PREFIX.",
    )
    @click.option(
        "--yes",
        "-y",
        "confirmed",
        is_flag=True,
        default=False,
        help="Do not ask before applying this to more than one workload.",
    )
    @click.option("--json", "as_json", is_flag=True, help="Emit the workload record as JSON (one array for --prefix).")
    @click.pass_obj
    def command(
        session: CliSession,
        workload_id: str | None,
        idempotency_key: str | None,
        org_id: str | None,
        project_id: str | None,
        prefix: str | None,
        confirmed: bool,
        as_json: bool,
    ) -> None:
        service = WorkloadCommandService(session)
        if prefix is not None:
            if workload_id is not None:
                raise click.UsageError("pass a workload id or --prefix, not both")
            scope = session.workload_scope(org_id=org_id, project_id=project_id)
            service.mutate_prefix(
                prefix, action, scope=scope, confirmed=confirmed, idempotency_key=idempotency_key, as_json=as_json
            )
            return
        if workload_id is None:
            raise click.UsageError("give a workload id, or --prefix to match several")
        service.mutate(
            service.resolve(workload_id, org_id=org_id, project_id=project_id),
            action,
            idempotency_key,
            as_json=as_json,
        )

    command.help = (
        f"{action.capitalize()} one workload, or every workload matching --prefix.\n\n"
        "WORKLOAD_ID may be a full id, a unique id prefix, or a workload's name when only one carries it. "
        "Nobody types a uuid from memory, and the id in a log line is usually the only copy anyone has."
    )
    return command


for _action in _WORKLOAD_ACTIONS:
    workload.add_command(_workload_action_command(_action))

workload.add_command(kick_workload)
workload.add_command(exec_in_workload)


def register_workload_commands(group: click.Group) -> None:
    """Attach workload commands to the root command group."""

    group.add_command(run_workload)
    group.add_command(workload)


__all__ = (
    "WorkloadCommandService",
    "register_workload_commands",
    "run_workload",
    "workload",
    "workload_explain",
    "workload_process_status",
    "workload_status",
)
