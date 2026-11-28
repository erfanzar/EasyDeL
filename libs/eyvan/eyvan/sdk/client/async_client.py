# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Asynchronous typed Eyvan SDK client."""

from __future__ import annotations

import asyncio
import time
import typing as tp

import httpx

from eyvan._internal.json_types import JsonObject, model_json
from eyvan.artifacts import StoredCallableArtifact
from eyvan.artifacts.managed import ManagedArtifactKind, parse_managed_uri
from eyvan.records import (
    DEFAULT_MAX_RETRIES,
    AuditEvent,
    BudgetRecord,
    CallableExecution,
    CapacityClass,
    CellRecord,
    CrossRegionTransferPolicy,
    LeaseRecord,
    PolicyRecord,
    PriorityBand,
    TpuChipRequest,
    WorkloadEstimate,
    WorkloadProgress,
    WorkloadRecord,
    WorkloadSetup,
    WorkloadState,
)
from eyvan.sdk.client.identity import ClientIdentity
from eyvan.sdk.client.requests import (
    pool_callable_request,
    pool_request_json,
    pool_workload_request,
    workload_list_params,
)
from eyvan.sdk.client.sync import UNREACHABLE_GRACE_S
from eyvan.sdk.client.transport import AsyncApiTransport, EyvanUnavailableError, read_deadline
from eyvan.sdk.client.types import HeaderMap, QueryParams
from eyvan.server.api.schemas import CapacityPoolResponse, WorkloadEstimateRequest, WorkloadSubmitRequest
from eyvan.server.service.state import TERMINAL_WORKLOAD_STATES
from eyvan.worker.runtime.callables import CallableArtifact

AsyncWorkloadObserver: tp.TypeAlias = tp.Callable[[WorkloadRecord], tp.Awaitable[None]]


class AsyncEyvanClient:
    """Asynchronous Eyvan API client for services and notebooks."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        identity: ClientIdentity | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._transport = AsyncApiTransport(base_url, identity=identity, client=client)
        self._client = self._transport.client

    async def _mutating_request(
        self,
        method: str,
        path: str,
        *,
        idempotency_key: str | None,
        params: QueryParams | None = None,
        json: JsonObject | None = None,
        content: bytes | None = None,
        headers: HeaderMap | None = None,
    ) -> httpx.Response:
        """Submit a mutation through the shared retrying transport."""

        return await self._transport.mutate(
            method,
            path,
            idempotency_key=idempotency_key,
            params=params,
            json=json,
            content=content,
            headers=headers,
        )

    async def submit_workload(
        self,
        request: WorkloadSubmitRequest,
        *,
        idempotency_key: str | None = None,
    ) -> WorkloadRecord:
        """Submit one typed workload definition."""

        response = await self._mutating_request(
            "POST",
            "/api/v1/workloads",
            json=model_json(request),
            idempotency_key=idempotency_key,
        )
        return WorkloadRecord.model_validate(response.json())

    async def upload_managed_callable(
        self,
        artifact: CallableArtifact,
        *,
        org_id: str,
        project_id: str,
    ) -> StoredCallableArtifact:
        """Upload one content-addressed callable through Eyvan-managed storage."""

        response = await self._mutating_request(
            "PUT",
            f"/api/v1/artifacts/callables/{artifact.sha256}",
            params={"org_id": org_id, "project_id": project_id},
            content=artifact.payload,
            headers={"Content-Type": "application/octet-stream"},
            idempotency_key=f"artifact-{artifact.sha256}",
        )
        uri = str(response.json()["uri"])
        return StoredCallableArtifact(
            uri=uri,
            sha256=artifact.sha256,
            python=artifact.python,
            format_version=artifact.format_version,
            signature=artifact.signature,
        )

    async def upload_managed_workspace(
        self,
        payload: bytes,
        sha256: str,
        *,
        org_id: str,
        project_id: str,
    ) -> str:
        """Upload one content-addressed workspace bundle and return its URI."""

        response = await self._mutating_request(
            "PUT",
            f"/api/v1/artifacts/workspaces/{sha256}",
            params={"org_id": org_id, "project_id": project_id},
            content=payload,
            headers={"Content-Type": "application/gzip"},
            idempotency_key=f"workspace-{sha256}",
        )
        return str(response.json()["uri"])

    async def read_managed_artifact(self, uri: str, *, org_id: str, project_id: str) -> bytes:
        """Read a managed result through the caller's authenticated project scope."""

        reference = parse_managed_uri(uri)
        if reference.kind is not ManagedArtifactKind.RESULTS:
            raise ValueError("authenticated managed reads currently support result artifacts")
        response = await self._transport.request_checked(
            "GET",
            f"/api/v1/artifacts/results/{reference.object_id}",
            params={"org_id": org_id, "project_id": project_id},
        )
        return response.content

    async def run(
        self,
        fleet_name: str,
        node_type: str,
        command: str,
        *,
        org_id: str,
        project_id: str,
        pods: int = 1,
        zone: str | None = None,
        name: str | None = None,
        environment: tp.Mapping[str, str] | None = None,
        secret_mounts: tp.Mapping[str, str] | None = None,
        metadata: tp.Mapping[str, str] | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_failure_retries: int = 0,
        max_preemption_retries: int = 10,
        checkpointable: bool = False,
        checkpoint_hook: str | None = None,
        checkpoint_output_uri: str | None = None,
        can_submit_children: bool = False,
        priority_band: PriorityBand = PriorityBand.INTERACTIVE,
        intra_band_priority: int = 0,
        checkpoint_grace_seconds: int = 0,
        estimated_duration_seconds: int = 3600,
        named_assets: tuple[str, ...] = (),
        dependencies: tuple[str, ...] = (),
        setup: WorkloadSetup | None = None,
        cross_region_transfer_policy: CrossRegionTransferPolicy | None = None,
        idempotency_key: str | None = None,
    ) -> WorkloadRecord:
        """Queue a command on a configured fleet node type."""

        request = pool_workload_request(
            org_id=org_id,
            project_id=project_id,
            command=command,
            pods=pods,
            zone=zone,
            name=name,
            environment=environment,
            secret_mounts=secret_mounts,
            metadata=metadata,
            max_retries=max_retries,
            max_failure_retries=max_failure_retries,
            max_preemption_retries=max_preemption_retries,
            checkpointable=checkpointable,
            checkpoint_hook=checkpoint_hook,
            checkpoint_output_uri=checkpoint_output_uri,
            can_submit_children=can_submit_children,
            priority_band=priority_band,
            intra_band_priority=intra_band_priority,
            checkpoint_grace_seconds=checkpoint_grace_seconds,
            estimated_duration_seconds=estimated_duration_seconds,
            named_assets=named_assets,
            dependencies=dependencies,
            setup=setup,
            cross_region_transfer_policy=cross_region_transfer_policy,
        )
        response = await self._mutating_request(
            "POST",
            f"/api/v1/fleets/{fleet_name}/node-types/{node_type}/workloads",
            json=pool_request_json(request),
            idempotency_key=idempotency_key,
        )
        return WorkloadRecord.model_validate(response.json())

    async def submit_pool_callable(
        self,
        fleet_name: str,
        accelerator_type: str,
        execution: CallableExecution,
        *,
        org_id: str,
        project_id: str,
        pods: int = 1,
        capacity_class: CapacityClass = CapacityClass.SPOT,
        name: str | None = None,
        environment: tp.Mapping[str, str] | None = None,
        secret_mounts: tp.Mapping[str, str] | None = None,
        metadata: tp.Mapping[str, str] | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_failure_retries: int = 0,
        max_preemption_retries: int = 10,
        checkpointable: bool = False,
        checkpoint_hook: str | None = None,
        checkpoint_output_uri: str | None = None,
        can_submit_children: bool = False,
        priority_band: PriorityBand = PriorityBand.INTERACTIVE,
        intra_band_priority: int = 0,
        checkpoint_grace_seconds: int = 0,
        estimated_duration_seconds: int = 3600,
        named_assets: tuple[str, ...] = (),
        dependencies: tuple[str, ...] = (),
        atomic: bool = False,
        tpu_chips: TpuChipRequest | None = None,
        setup: WorkloadSetup | None = None,
        cross_region_transfer_policy: CrossRegionTransferPolicy | None = None,
        idempotency_key: str | None = None,
    ) -> WorkloadRecord:
        """Queue a trusted callable on the fleet pool matching an accelerator type."""

        request = pool_callable_request(
            org_id=org_id,
            project_id=project_id,
            accelerator_type=accelerator_type,
            pods=pods,
            atomic=atomic,
            tpu_chips=tpu_chips,
            capacity_class=capacity_class,
            execution=execution,
            name=name,
            environment=environment,
            secret_mounts=secret_mounts,
            metadata=metadata,
            max_retries=max_retries,
            max_failure_retries=max_failure_retries,
            max_preemption_retries=max_preemption_retries,
            checkpointable=checkpointable,
            checkpoint_hook=checkpoint_hook,
            checkpoint_output_uri=checkpoint_output_uri,
            can_submit_children=can_submit_children,
            priority_band=priority_band,
            intra_band_priority=intra_band_priority,
            checkpoint_grace_seconds=checkpoint_grace_seconds,
            estimated_duration_seconds=estimated_duration_seconds,
            named_assets=named_assets,
            dependencies=dependencies,
            setup=setup,
            cross_region_transfer_policy=cross_region_transfer_policy,
        )
        response = await self._mutating_request(
            "POST",
            f"/api/v1/fleets/{fleet_name}/callables",
            json=pool_request_json(request),
            idempotency_key=idempotency_key,
        )
        return WorkloadRecord.model_validate(response.json())

    async def estimate_workload(self, request: WorkloadEstimateRequest) -> WorkloadEstimate:
        """Ask what a workload would cost, before submitting anything.

        Not a mutating request and no idempotency key: it writes nothing, so a
        replay is just another read.
        """

        response = await self._transport.request_checked("POST", "/api/v1/workloads/estimate", json=model_json(request))
        return WorkloadEstimate.model_validate(response.json())

    async def list_workloads(
        self,
        *,
        org_id: str,
        project_id: str,
        state: WorkloadState | None = None,
    ) -> list[WorkloadRecord]:
        """List typed workloads visible in one project."""

        response = await self._transport.request_checked(
            "GET",
            "/api/v1/workloads",
            params=workload_list_params(org_id=org_id, project_id=project_id, state=state),
        )
        return [WorkloadRecord.model_validate(item) for item in response.json()]

    async def list_capacity(
        self,
        *,
        org_id: str,
        project_id: str,
        accelerator_type: str | None = None,
        capacity_class: CapacityClass | None = None,
    ) -> list[CapacityPoolResponse]:
        """List enabled pools available to a workload scope, optionally filtered by accelerator shape."""

        params: QueryParams = {"org_id": org_id, "project_id": project_id}
        if accelerator_type is not None:
            params["accelerator_type"] = accelerator_type
        if capacity_class is not None:
            params["capacity_class"] = capacity_class.value
        response = await self._transport.request_checked("GET", "/api/v1/capacity", params=params)
        return [CapacityPoolResponse.model_validate(item) for item in response.json()["pools"]]

    async def get_workload(self, workload_id: str) -> WorkloadRecord:
        """Return one typed workload record."""

        response = await self._transport.request_checked("GET", f"/api/v1/workloads/{workload_id}")
        return WorkloadRecord.model_validate(response.json())

    async def get_workload_progress(self, workload_id: str) -> WorkloadProgress:
        """Return typed autoscaling and execution progress for one workload."""

        response = await self._transport.request_checked("GET", f"/api/v1/workloads/{workload_id}/progress")
        return WorkloadProgress.model_validate(response.json())

    async def wait_workload(
        self,
        workload_id: str,
        *,
        timeout: float | None = None,
        poll_interval: float = 1.0,
        on_update: AsyncWorkloadObserver | None = None,
    ) -> WorkloadRecord:
        """Wait asynchronously until a workload reaches a terminal durable state.

        A control plane that cannot be reached is forgiven for
        ``UNREACHABLE_GRACE_S`` -- the same grace the synchronous client
        grants -- so a restart or a rolling deploy does not end the watching
        while the durable workload keeps running unwatched.

        Args:
            workload_id: Durable workload identifier.
            timeout: Maximum number of seconds to wait, or ``None`` to wait indefinitely.
            poll_interval: Delay between control-plane observations.
            on_update: Optional async observer invoked with every fetched workload record.
        """

        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        deadline = None if timeout is None else time.monotonic() + timeout
        # Mirrors ``EyvanClient.wait_workload``: the workload is durable across
        # a plane restart, a proxy hiccup, or a rolling deploy, so one dropped
        # poll must not kill the watch. Only transport failures are forgiven;
        # a 404 or a 403 is an answer and is raised at once.
        unreachable_since: float | None = None
        with read_deadline(deadline if timeout else None):
            while True:
                if timeout and deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(f"workload {workload_id!r} did not finish within {timeout:g} seconds")
                try:
                    workload = await self.get_workload(workload_id)
                except TimeoutError as exc:
                    if not timeout or deadline is None or time.monotonic() < deadline:
                        raise
                    raise TimeoutError(f"workload {workload_id!r} did not finish within {timeout:g} seconds") from exc
                except EyvanUnavailableError as exc:
                    now = time.monotonic()
                    if unreachable_since is None:
                        unreachable_since = now
                    elif now - unreachable_since >= UNREACHABLE_GRACE_S:
                        raise
                    if deadline is not None and now >= deadline:
                        raise TimeoutError(
                            f"workload {workload_id!r} did not finish within {timeout:g} seconds "
                            "and the control plane is currently unreachable"
                        ) from exc
                    await asyncio.sleep(
                        min(poll_interval, max((deadline - now) if deadline is not None else poll_interval, 0))
                    )
                    continue
                unreachable_since = None
                if timeout and deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(f"workload {workload_id!r} did not finish within {timeout:g} seconds")
                if on_update is not None:
                    remaining = max(deadline - time.monotonic(), 0) if timeout and deadline is not None else None
                    try:
                        async with asyncio.timeout(remaining):
                            await on_update(workload)
                    except TimeoutError as exc:
                        if not timeout or deadline is None or time.monotonic() < deadline:
                            raise
                        raise TimeoutError(
                            f"workload {workload_id!r} did not finish within {timeout:g} seconds"
                        ) from exc
                if timeout and deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(f"workload {workload_id!r} did not finish within {timeout:g} seconds")
                if workload.state in TERMINAL_WORKLOAD_STATES:
                    return workload
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(f"workload {workload_id!r} did not finish within {timeout:g} seconds")
                delay = poll_interval if deadline is None else min(poll_interval, max(deadline - time.monotonic(), 0))
                await asyncio.sleep(delay)

    async def workload_action(
        self,
        workload_id: str,
        action: tp.Literal["cancel", "checkpoint", "retry"],
        *,
        idempotency_key: str | None = None,
    ) -> WorkloadRecord:
        """Apply a durable workload action."""

        response = await self._mutating_request(
            "POST",
            f"/api/v1/workloads/{workload_id}/{action}",
            idempotency_key=idempotency_key,
        )
        return WorkloadRecord.model_validate(response.json())

    async def workload_log_page(self, workload_id: str, *, after: str, limit: int) -> dict[str, tp.Any]:
        """Read one page of a workload's archived output."""

        response = await self._transport.request_checked(
            "GET",
            f"/api/v1/workloads/{workload_id}/log-page",
            params={"after": after, "limit": limit},
        )
        return response.json()

    async def workload_logs(self, workload_id: str) -> str:
        """Return currently available workload logs."""

        response = await self._transport.request_checked("GET", f"/api/v1/workloads/{workload_id}/logs")
        return response.text

    async def force_finalize_preemption(
        self,
        command_id: str,
        *,
        reason: str,
        idempotency_key: str | None = None,
    ) -> WorkloadRecord:
        """Force finalization of one preemption command."""

        response = await self._mutating_request(
            "POST",
            f"/api/v1/preemptions/{command_id}/force-finalize",
            json={"reason": reason},
            idempotency_key=idempotency_key,
        )
        return WorkloadRecord.model_validate(response.json())

    async def list_cells(self) -> list[CellRecord]:
        """List registered execution cells."""

        response = await self._transport.request_checked("GET", "/api/v1/cells")
        return [CellRecord.model_validate(item) for item in response.json()]

    async def cell_action(
        self,
        cell_id: str,
        action: tp.Literal["drain", "delete"],
        *,
        idempotency_key: str | None = None,
    ) -> CellRecord:
        """Drain or delete one execution cell."""

        method = "POST" if action == "drain" else "DELETE"
        path = f"/api/v1/cells/{cell_id}/drain" if action == "drain" else f"/api/v1/cells/{cell_id}"
        response = await self._mutating_request(method, path, idempotency_key=idempotency_key)
        return CellRecord.model_validate(response.json())

    async def list_leases(self, *, org_id: str, project_id: str) -> list[LeaseRecord]:
        """List project capacity leases."""

        response = await self._transport.request_checked(
            "GET", "/api/v1/leases", params={"org_id": org_id, "project_id": project_id}
        )
        return [LeaseRecord.model_validate(item) for item in response.json()]

    async def list_events(self, *, org_id: str, project_id: str, after_sequence: int = 0) -> list[AuditEvent]:
        """List project audit events after a durable sequence number."""

        response = await self._transport.request_checked(
            "GET",
            "/api/v1/events",
            params={"org_id": org_id, "project_id": project_id, "after_sequence": after_sequence},
        )
        return [AuditEvent.model_validate(item) for item in response.json()]

    async def list_budgets(self, *, org_id: str) -> list[BudgetRecord]:
        """List organization budgets."""

        response = await self._transport.request_checked("GET", "/api/v1/budgets", params={"org_id": org_id})
        return [BudgetRecord.model_validate(item) for item in response.json()]

    async def list_policies(self, *, org_id: str) -> list[PolicyRecord]:
        """List organization policies."""

        response = await self._transport.request_checked("GET", "/api/v1/policies", params={"org_id": org_id})
        return [PolicyRecord.model_validate(item) for item in response.json()]

    async def aclose(self) -> None:
        """Close the underlying asynchronous HTTP connection pool."""

        await self._transport.aclose()

    async def __aenter__(self) -> AsyncEyvanClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


__all__ = ("AsyncEyvanClient",)
