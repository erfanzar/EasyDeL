# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""HTTP client for the versioned cell-agent contract."""

from __future__ import annotations

import typing as tp

import httpx
from pydantic import ValidationError

from eyvan._internal.json_types import JsonObject, is_json_object
from eyvan._internal.trace_context import outgoing_trace_headers
from eyvan.capacity.drivers.base import CapacityProviderFailureCode
from eyvan.identity.mtls import MutualTlsFiles, client_ssl_context
from eyvan.worker.agent.models import (
    AgentBuild,
    AgentCapabilities,
    AgentObservation,
    CapacityProviderFailure,
    CapacityRequest,
    CellStageAction,
    CellStageActionRequest,
    CellStageActionResult,
    CheckpointRequest,
    GangReclaimNoticeRequest,
    GangWorkloadMutationRequest,
    GangWorkloadRequest,
    MutationConflict,
    MutationConflictCode,
    ReclaimNoticeRequest,
    WorkloadMutationRequest,
    WorkloadObservation,
    WorkloadRequest,
)

#: How long one staging action may take, by what it does.
#:
#: A fetch is minutes of network for a large asset, so it gets the ceiling
#: reconciliation and submission already use rather than the generic 30 seconds
#: -- a timeout there is recorded as *inconclusive*, which is honest but costs
#: the transfer.
#:
#: Measuring and evicting must not inherit that patience. They are a stat and an
#: unlink, and they are what a hung agent is discovered by: the staging phase
#: shares a leader cycle with placement and takes up to
#: ``MAX_STAGE_ACTIONS_PER_PASS`` round trips, so a generous timeout on the cheap
#: half is how one unreachable fleet delays every other phase for an hour.
_STAGE_TIMEOUTS: dict[CellStageAction, float] = {
    CellStageAction.MEASURE: 60.0,
    CellStageAction.STAGE: 1800.0,
    CellStageAction.EVICT: 300.0,
}

#: An exec request legitimately holds the connection for as long as the command
#: itself was budgeted to run, so the per-request HTTP timeout is that budget
#: plus a margin for spawn and serialization overhead, capped just past the
#: wire contract's ``le=600``. Deriving it is what keeps a long diagnostic from
#: being recorded as a rank error while it is still running on the agent.
_EXEC_HTTP_TIMEOUT_MARGIN_S = 30.0
_EXEC_HTTP_TIMEOUT_MAX_S = 660.0


def _wire(request: tp.Any) -> JsonObject:
    """Serialize a request for an agent that may be older than this head.

    ``exclude_none`` is the whole point. Agent-side models are ``extra="forbid"``
    -- deliberately, so a rolled-back replica cannot be handed a field it would
    persist and then fail to read back -- and every optional field on them
    defaults to ``None``. Omitting a ``None`` therefore parses to exactly the
    same object on a matching agent, while an agent that predates the field
    accepts the payload instead of rejecting the whole configuration.

    Sending them cost a real run: a head one commit ahead sent
    ``runtime.python: null`` to an agent whose ``NativeRuntimeConfig`` had no
    such field, every allocation died on ``extra_forbidden``, and the operator
    was told "provider rejected capacity request" -- about a provider that had
    never been called.
    """

    return request.model_dump(mode="json", exclude_none=True)


class CellAgentBusyError(RuntimeError):
    """The cell agent cannot acquire its shared mutation lease yet."""


class CellAgentStaleMutationError(RuntimeError):
    """The cell agent rejected an obsolete generation or fencing token."""


class CellAgentUnsupportedCapabilityError(RuntimeError):
    """The cell agent's backend does not implement the requested operation.

    Terminal, not transient. Without this the agent's deliberate 501 arrived as a
    generic ``HTTPStatusError``, indistinguishable from the agent being broken --
    so "this backend cannot run checkpoint hooks" was retried on every tick and
    spent the workload's retry budget on an answer that will never change.
    """


class CellAgentCapacityProviderError(RuntimeError):
    """The cell agent returned an operator-safe capacity-provider failure."""

    def __init__(self, detail: str, *, code: CapacityProviderFailureCode) -> None:
        self.code = code
        super().__init__(detail)


class CellAgentInvalidConfigurationError(RuntimeError):
    """The agent refused a configuration that can never succeed as written.

    Terminal, like :class:`CellAgentUnsupportedCapabilityError` and for the same
    reason: retrying an unchanged provider option cannot start working. Carrying
    the agent's own sentence matters more here than anywhere else, because the
    fix is a line of YAML and the message names it exactly.
    """


class CellAgentStagingUnsupportedError(RuntimeError):
    """This agent has no staging executor for this cell -- not a failed action.

    Raised for exactly two answers, because they mean the same thing to a
    control plane: HTTP 404, which is an agent that predates the staging wire
    method entirely, and HTTP 501, which is a current agent whose execution
    backend does not run workloads on the disk this process owns.

    It exists so neither is read as a failure. An agent mid-rollout must look
    like a deployment that configured no executor -- nothing staged, nothing
    recorded, nothing charged, nothing retried -- rather than like a fleet that
    is refusing to answer, which would write an unproven row against every cell
    and charge its budget for bytes nobody attempted to fetch.
    """


class CellAgentUnavailableError(RuntimeError):
    """The cell agent could not answer, so nothing was learned about the cell.

    The mirror of :class:`CellAgentUnsupportedCapabilityError`: that one exists so
    a deliberate answer is not read as the agent being broken, and this one exists
    so the agent being broken is not read as an answer.

    An unstructured 5xx means the agent itself failed -- its own storage backend
    was unreachable, it is restarting, a proxy is in the way. None of that is a
    statement about the *capacity*, and treating it as one is expensive: the
    control plane records ``CapacityStatus.PROVIDER_ERROR``, restarts the provider
    phase clock, and feeds ``ProvisionOutcome.PROVIDER_ERROR`` into the zone's
    belief -- retiring capacity for a timeout the provider never caused and
    biasing future placement against a zone that did nothing wrong.

    A structured 502 (:class:`CellAgentCapacityProviderError`) is the opposite
    case and keeps its own handling -- there the agent successfully reported that
    the *provider* failed.
    """


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code == 409:
        try:
            conflict = MutationConflict.model_validate_json(response.content)
        except (ValidationError, ValueError):
            response.raise_for_status()
            return
        if conflict.code is MutationConflictCode.DRIVER_BUSY:
            raise CellAgentBusyError(conflict.detail)
        if conflict.code is MutationConflictCode.STALE_MUTATION:
            raise CellAgentStaleMutationError(conflict.detail)
    if response.status_code == 400:
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if isinstance(payload, dict) and payload.get("code") == "invalid_configuration":
            raise CellAgentInvalidConfigurationError(str(payload.get("detail") or "").strip() or "invalid configuration")
    if response.status_code == 501:
        detail = ""
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if isinstance(payload, dict) and isinstance(payload.get("detail"), str):
            detail = payload["detail"].strip()
        if not detail:
            detail = str(response.text or "").strip() or "the cell agent backend does not support this operation"
        raise CellAgentUnsupportedCapabilityError(detail)
    if response.status_code == 502:
        try:
            failure = CapacityProviderFailure.model_validate_json(response.content)
        except (ValidationError, ValueError):
            # A proxy's Bad Gateway is not a statement about the provider.
            # Only a validated agent failure may bypass the unavailable branch.
            pass
        else:
            raise CellAgentCapacityProviderError(failure.detail, code=failure.code)
    if response.status_code >= 500:
        # Everything the agent says deliberately has already been handled above,
        # so anything left in the 5xx range is the agent failing to answer.
        detail = str(response.text or "").strip() or f"the cell agent returned {response.status_code}"
        raise CellAgentUnavailableError(detail)
    response.raise_for_status()


class CellAgentClient:
    """Control-plane client for one cell agent, optionally over mutual TLS."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        client: httpx.Client | None = None,
        mtls: MutualTlsFiles | None = None,
    ):
        """Build the transport, presenting a client certificate only when asked.

        ``mtls=None`` -- the default at every call site -- leaves the transport
        byte for byte what it is today. This is the *client* half of the rollout
        documented in :mod:`eyvan.identity.mtls`: the control plane starts
        presenting a certificate at step 4, strictly before any agent is moved to
        ``required`` at step 5. Reversing those two is what locks a control plane
        out of the agents holding its running workloads.

        An explicit ``client`` wins, because a caller that built its own
        transport has already decided its TLS posture and silently replacing it
        would be worse than ignoring the argument.
        """

        headers = {"Authorization": f"Bearer {token}"} if token else None
        if client is not None:
            self.client = client
            return
        if mtls is not None and httpx.URL(base_url).scheme != "https":
            raise ValueError("mutual TLS requires an HTTPS cell-agent endpoint")
        context = client_ssl_context(mtls) if mtls is not None else None
        if context is None:
            self.client = httpx.Client(base_url=base_url, headers=headers, timeout=30.0)
        else:
            self.client = httpx.Client(base_url=base_url, headers=headers, timeout=30.0, verify=context)

    def _send(self, method: str, url: str, **kwargs: tp.Any) -> httpx.Response:
        """Issue one request, mapping "could not ask" onto one typed error.

        A 5xx is a *response*, so `_raise_for_status` can classify it. A dropped
        connection, a DNS failure or a read timeout produces no response at all
        -- httpx raises before `_raise_for_status` is ever reached -- so those
        used to escape as bare transport exceptions and be caught by the
        controller's `except Exception`, which records them as
        `CapacityStatus.PROVIDER_ERROR`.

        That is the same misclassification :class:`CellAgentUnavailableError`
        exists to prevent: agent transport failure says nothing about provider
        capacity.

        Routing every call through here keeps the taxonomy in one place instead
        of twelve `except` clauses that will drift.
        """

        trace_headers = outgoing_trace_headers()
        if trace_headers:
            headers = httpx.Headers(kwargs.get("headers"))
            headers.update(trace_headers)
            kwargs["headers"] = headers
        try:
            response = self.client.request(method, url, **kwargs)
        except httpx.TransportError as exc:
            raise CellAgentUnavailableError(f"the cell agent could not be reached: {exc}") from exc
        _raise_for_status(response)
        return response

    def capabilities(self) -> AgentCapabilities:
        response = self._send("GET", "/api/v1/capabilities")
        return AgentCapabilities.model_validate_json(response.content)

    def build(self) -> AgentBuild | None:
        """Return what code this agent runs, or ``None`` if it cannot say.

        A 404 is the answer from an agent that predates the route, and it is not
        an error: that agent is reachable and working, it simply cannot describe
        itself. ``None`` keeps "this agent does not report a build" distinct
        from "this agent could not be reached", which is a difference an
        operator chasing version skew has to be able to see.

        Every other failure still raises. An agent that answers 500 to a plain
        read is not one whose build should be quietly reported as unknown.
        """

        try:
            response = self._send("GET", "/api/v1/build")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise
        return AgentBuild.model_validate_json(response.content)

    def worker_builds(self, cell_id: str) -> list[dict[str, object]] | None:
        """Return each native worker's build for one cell, or ``None`` if unasked.

        ``None`` from an agent that predates the route (404) or whose execution
        driver cannot be asked at all (200 with a null body), an empty list from
        one whose backend runs the workload in the agent process itself. The
        answers are different facts and the caller reports them differently.
        """

        try:
            response = self._send("GET", f"/api/v1/cells/{cell_id}/worker-builds")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise
        payload = response.json()
        if payload is None:
            return None
        if not isinstance(payload, list):
            raise ValueError("cell agent worker builds must be a JSON array")
        return payload

    def inventory(self) -> list[dict[str, object]]:
        """Return every cell this agent holds durable state for."""

        response = self._send("GET", "/api/v1/cells")
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError("cell agent inventory must be a JSON array")
        return payload

    def observe(self, cell_id: str) -> AgentObservation:
        response = self._send("GET", f"/api/v1/cells/{cell_id}")
        return AgentObservation.model_validate(response.json())

    def reconcile(self, cell_id: str, request: WorkloadMutationRequest) -> AgentObservation:
        response = self._send(
            "POST",
            f"/api/v1/cells/{cell_id}/reconcile",
            json=_wire(request),
            timeout=1800,
        )
        return AgentObservation.model_validate(response.json())

    def apply_capacity(self, cell_id: str, request: CapacityRequest) -> AgentObservation:
        response = self._send(
            "PUT",
            f"/api/v1/cells/{cell_id}/capacity",
            json=_wire(request),
            timeout=1800,
        )
        return AgentObservation.model_validate(response.json())

    def submit(self, cell_id: str, request: WorkloadRequest) -> WorkloadObservation:
        # Submission holds the cell-agent's durable fence through remote
        # launch. On GCS that requires several CAS calls before the native
        # worker responds, so the generic 30-second HTTP timeout can expire
        # after a job was accepted but before its remote ID is persisted.
        response = self._send(
            "POST",
            f"/api/v1/cells/{cell_id}/workloads",
            json=_wire(request),
            timeout=1800,
        )
        return WorkloadObservation.model_validate(response.json())

    def submit_gang(self, coordinator_cell_id: str, request: GangWorkloadRequest) -> WorkloadObservation:
        """Submit one workload group through its deterministic leader cell agent."""

        response = self._send(
            "POST",
            f"/api/v1/cells/{coordinator_cell_id}/workload-gangs",
            json=_wire(request),
            timeout=1800,
        )
        return WorkloadObservation.model_validate(response.json())

    def workload(self, cell_id: str, remote_id: str) -> WorkloadObservation:
        response = self._send("GET", f"/api/v1/cells/{cell_id}/workloads/{remote_id}")
        return WorkloadObservation.model_validate(response.json())

    def gang_workload(
        self,
        coordinator_cell_id: str,
        workload_id: str,
        remote_id: str,
    ) -> WorkloadObservation:
        """Read aggregate status for a previously submitted workload group."""

        response = self._send("GET", f"/api/v1/cells/{coordinator_cell_id}/workload-gangs/{workload_id}/{remote_id}")
        return WorkloadObservation.model_validate(response.json())

    def stop(self, cell_id: str, remote_id: str, request: WorkloadMutationRequest) -> WorkloadObservation:
        response = self._send(
            "POST",
            f"/api/v1/cells/{cell_id}/workloads/{remote_id}/stop",
            json=_wire(request),
        )
        return WorkloadObservation.model_validate(response.json())

    def stop_gang(
        self,
        coordinator_cell_id: str,
        workload_id: str,
        remote_id: str,
        request: GangWorkloadMutationRequest,
    ) -> WorkloadObservation:
        """Stop every rank in one native workload group."""

        response = self._send(
            "POST",
            f"/api/v1/cells/{coordinator_cell_id}/workload-gangs/{workload_id}/{remote_id}/stop",
            json=_wire(request),
        )
        return WorkloadObservation.model_validate(response.json())

    def retire(self, cell_id: str, remote_id: str, request: WorkloadMutationRequest) -> WorkloadObservation:
        response = self._send(
            "POST",
            f"/api/v1/cells/{cell_id}/workloads/{remote_id}/retire",
            json=_wire(request),
            timeout=600,
        )
        return WorkloadObservation.model_validate(response.json())

    def retire_gang(
        self,
        coordinator_cell_id: str,
        workload_id: str,
        remote_id: str,
        request: GangWorkloadMutationRequest,
    ) -> WorkloadObservation:
        """Retire every rank in one native workload group."""

        response = self._send(
            "POST",
            f"/api/v1/cells/{coordinator_cell_id}/workload-gangs/{workload_id}/{remote_id}/retire",
            json=_wire(request),
            timeout=600,
        )
        return WorkloadObservation.model_validate(response.json())

    def checkpoint(self, cell_id: str, remote_id: str, request: CheckpointRequest) -> WorkloadObservation:
        response = self._send(
            "POST",
            f"/api/v1/cells/{cell_id}/workloads/{remote_id}/checkpoint",
            json=_wire(request),
        )
        return WorkloadObservation.model_validate(response.json())

    def reclaim_notice(self, cell_id: str, remote_id: str, request: ReclaimNoticeRequest) -> WorkloadObservation | None:
        """Deliver a graceful reclaim notice, returning ``None`` if unsupported by the agent."""

        try:
            response = self._send(
                "POST",
                f"/api/v1/cells/{cell_id}/workloads/{remote_id}/reclaim-notice",
                json=_wire(request),
            )
        except CellAgentUnsupportedCapabilityError:
            return None
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise
        return WorkloadObservation.model_validate(response.json())

    def reclaim_notice_gang(
        self,
        coordinator_cell_id: str,
        workload_id: str,
        remote_id: str,
        request: GangReclaimNoticeRequest,
    ) -> WorkloadObservation | None:
        """Deliver a graceful reclaim notice across every gang member, or ``None`` if unsupported."""

        try:
            response = self._send(
                "POST",
                f"/api/v1/cells/{coordinator_cell_id}/workload-gangs/{workload_id}/{remote_id}/reclaim-notice",
                json=_wire(request),
            )
        except CellAgentUnsupportedCapabilityError:
            return None
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise
        return WorkloadObservation.model_validate(response.json())

    def logs(self, cell_id: str, remote_id: str) -> str:
        response = self._send("GET", f"/api/v1/cells/{cell_id}/workloads/{remote_id}/logs")
        return response.text

    def gang_logs(self, coordinator_cell_id: str, workload_id: str, remote_id: str) -> str:
        """Read rank-prefixed logs from every member of a native workload group."""

        response = self._send(
            "GET", f"/api/v1/cells/{coordinator_cell_id}/workload-gangs/{workload_id}/{remote_id}/logs"
        )
        return response.text

    def exec_in_workload(
        self,
        cell_id: str,
        remote_id: str,
        command: tp.Sequence[str],
        *,
        timeout_s: float = 30.0,
        runtime_rank: int | None = None,
    ) -> JsonObject:
        """Run one diagnostic command beside a live workload on one cell.

        The agent may hold the answer for the whole ``timeout_s`` while the
        command runs, so the request carries an HTTP timeout derived from that
        budget instead of the client default -- which expired mid-command and
        made the observation record a rank error for a rank still running.
        """

        document: JsonObject = {"command": list(command), "timeout_s": timeout_s}
        if runtime_rank is not None:
            document["runtime_rank"] = runtime_rank
        response = self._send(
            "POST",
            f"/api/v1/cells/{cell_id}/workloads/{remote_id}/exec",
            json=document,
            timeout=min(timeout_s + _EXEC_HTTP_TIMEOUT_MARGIN_S, _EXEC_HTTP_TIMEOUT_MAX_S),
        )
        payload = response.json()
        if not is_json_object(payload):
            raise ValueError("cell agent exec result must be a JSON object")
        return payload

    def log_page(self, cell_id: str, remote_id: str, *, after: str = "", limit: int = 200) -> JsonObject:
        """Read one resumable page of structured log records from one cell."""

        response = self._send(
            "GET",
            f"/api/v1/cells/{cell_id}/workloads/{remote_id}/log-page",
            params={"after": after, "limit": limit},
        )
        payload = response.json()
        if not is_json_object(payload):
            raise ValueError("cell agent log page must be a JSON object")
        return payload

    def process_status(self, cell_id: str, remote_id: str) -> JsonObject:
        """Read live process detail for one workload attempt on one cell."""

        response = self._send("GET", f"/api/v1/cells/{cell_id}/workloads/{remote_id}/process-status")
        payload = response.json()
        if not is_json_object(payload):
            raise ValueError("cell agent process status must be a JSON object")
        return payload

    def stage_asset(self, cell_id: str, request: CellStageActionRequest) -> CellStageActionResult:
        """Measure, materialize or remove one named asset on one cell's disk.

        Raises:
            CellAgentStagingUnsupportedError: This agent has no staging executor
                for this cell, either because it predates the wire method or
                because its backend does not own the disk. Read as "no
                executor", never as a failed action.
            CellAgentStaleMutationError: The generation no longer owns the cell.
            CellAgentUnavailableError: The agent could not answer at all.
        """

        try:
            response = self._send(
                "POST",
                f"/api/v1/cells/{cell_id}/stage-actions",
                json=_wire(request),
                timeout=_STAGE_TIMEOUTS[request.action],
            )
        except CellAgentUnsupportedCapabilityError as exc:
            raise CellAgentStagingUnsupportedError(str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                # An agent that predates managed staging, or a proxy in front of
                # one. Either way this control plane has no executor here, which
                # is a fact about the deployment rather than about the asset.
                raise CellAgentStagingUnsupportedError(
                    f"cell agent for {cell_id!r} does not implement managed staging"
                ) from exc
            raise
        return CellStageActionResult.model_validate_json(response.content)

    def drain(self, cell_id: str, request: WorkloadMutationRequest) -> AgentObservation:
        response = self._send("POST", f"/api/v1/cells/{cell_id}/drain", json=_wire(request))
        return AgentObservation.model_validate(response.json())

    def release(self, cell_id: str, request: WorkloadMutationRequest) -> AgentObservation:
        response = self._send(
            "DELETE",
            f"/api/v1/cells/{cell_id}",
            json=_wire(request),
            timeout=700,
        )
        return AgentObservation.model_validate(response.json())

    def close(self) -> None:
        self.client.close()
