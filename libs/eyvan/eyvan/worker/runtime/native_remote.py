# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""HTTP execution driver for one Eyvan native worker.

**Deployment order: native workers first, then agents, then the controller.**
This driver calls one route older workers do not serve --
``POST /api/v1/cells/{cell_id}/stage-actions``, the worker half of managed
staging. A worker that predates it answers 404 and a worker that declares no
source prefixes answers 501, and both become
:class:`~eyvan.capacity.drivers.base.UnsupportedCapabilityError` here, which the
agent above reports to the control plane as *no staging executor*: absence of a
capability, never a failed action. So an agent upgraded ahead of its workers
stages nothing, records nothing and charges nothing.
"""

from __future__ import annotations

import os
import typing as tp

import httpx
from pydantic import ValidationError

from eyvan._internal.json_types import JsonObject, JsonValue, is_json_object, model_json
from eyvan._internal.trace_context import outgoing_trace_headers
from eyvan.capacity.drivers.base import (
    ExecutionWorkloadObservation,
    MutationContext,
    StaleMutationError,
    UnsupportedCapabilityError,
)
from eyvan.identity.mtls import MutualTlsFiles, client_ssl_context
from eyvan.worker.agent.models import (
    AgentBuild,
    CellStageAction,
    CellStageActionResult,
    CellStageVerdict,
    CheckpointRequest,
    MutationConflict,
    MutationConflictCode,
    MutationModel,
    NativeStageActionRequest,
    NativeWorkerHealth,
    WorkloadMutationRequest,
    WorkloadReference,
    WorkloadRequest,
)
from eyvan.worker.agent.models import WorkloadObservation as AgentWorkloadObservation
from eyvan.worker.runtime.log_records import LogPage, log_page_from_json
from eyvan.worker.runtime.stage_executor import AssetStageResult, AssetStageVerdict

#: Wall-clock bound on one staging hop, per action. A dataset pull is minutes of
#: network, a measurement is one HEAD, and an eviction is a few unlinks.
#:
#: Each is deliberately under the control plane's own bound on the outer hop, so
#: a fetch that runs long ends here first and the agent still answers. An
#: inconclusive verdict is something the ledger can record against a named host;
#: an outer request that died says nothing about which disk holds what.
_STAGE_TIMEOUTS: dict[CellStageAction, float] = {
    CellStageAction.MEASURE: 45.0,
    CellStageAction.STAGE: 1500.0,
    CellStageAction.EVICT: 240.0,
}

#: Exec waits synchronously for the command's budget (up to 600 seconds on the
#: wire). Match the outer agent hop's response margin and bounded HTTP policy.
_EXEC_HTTP_TIMEOUT_MARGIN_S = 30.0
_EXEC_HTTP_TIMEOUT_MAX_S = 660.0

#: What one staging verdict off the worker's wire means to this host.
_STAGE_VERDICTS: dict[CellStageVerdict, AssetStageVerdict] = {
    CellStageVerdict.OK: AssetStageVerdict.OK,
    CellStageVerdict.REFUSED: AssetStageVerdict.REFUSED,
    CellStageVerdict.INCONCLUSIVE: AssetStageVerdict.INCONCLUSIVE,
}


class NativeRemoteExecutor:
    """Delegate fenced workload operations to one native worker over HTTP.

    Also the agent's route to that worker's disk. ``measure``, ``stage`` and
    ``evict`` satisfy
    :class:`~eyvan.worker.runtime.stage_executor.AssetStageDriver`, so an agent
    whose workloads run on another machine forwards staging exactly as it
    forwards every other operation: to the process that owns the thing being
    acted on. The executor itself stays on the worker -- there is one
    implementation of fetch, measure and evict, and this is a route to it.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        secret: str | None = None,
        client: httpx.Client | None = None,
        mtls: MutualTlsFiles | None = None,
    ) -> None:
        """Build the transport; ``mtls=None`` keeps today's bearer-only posture.

        The inner half of the rollout in :mod:`eyvan.identity.mtls`: the agent
        presents its certificate to the worker at step 4, before the worker is
        moved to ``required`` at step 5. The bearer secret is still required
        either way -- mutual TLS authenticates the *host*, not the fenced
        mutation, and dropping the bearer would widen what a certificate grants.
        """

        if not endpoint or endpoint != endpoint.strip():
            raise ValueError("native worker endpoint must be non-empty and trimmed")
        if not isinstance(secret, str) or not secret.strip():
            raise ValueError("native worker requests require a non-empty bearer secret")
        self.endpoint = endpoint.rstrip("/")
        self._headers = {"Authorization": f"Bearer {secret}"}
        self._owns_client = client is None
        if client is None and mtls is not None and httpx.URL(self.endpoint).scheme != "https":
            raise ValueError("mutual TLS requires an HTTPS native-worker endpoint")
        context = client_ssl_context(mtls) if (client is None and mtls is not None) else None
        if client is not None:
            self.client = client
        elif context is None:
            self.client = httpx.Client(base_url=self.endpoint, timeout=30.0)
        else:
            self.client = httpx.Client(base_url=self.endpoint, timeout=30.0, verify=context)
        try:
            retire_wait_s = float(os.getenv("EYVAN_SUPERSEDE_WAIT_TIMEOUT_S", "300"))
        except ValueError:
            retire_wait_s = 300.0
        self._retire_timeout_s = max(retire_wait_s, 0.1) + 15.0

    @staticmethod
    def _mutation(context: MutationContext) -> MutationModel:
        return MutationModel(
            idempotency_key=context.idempotency_key,
            generation=context.generation,
            fencing_token=context.fencing_token,
            correlation_id=context.correlation_id,
        )

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> httpx.Response:
        if response.status_code == 400:
            try:
                payload = response.json()
            except ValueError:
                payload = None
            if isinstance(payload, dict) and payload.get("code") == "invalid_configuration":
                # Re-raised as the ValueError it was on the worker, so the cell
                # agent's own handler answers the plane with the same verdict.
                raise ValueError(str(payload.get("detail") or "the native worker refused the request as invalid"))
        if response.status_code == 409:
            try:
                conflict = MutationConflict.model_validate_json(response.content)
            except (ValidationError, ValueError):
                response.raise_for_status()
                return response
            if conflict.code is MutationConflictCode.STALE_MUTATION:
                raise StaleMutationError(conflict.detail)
        response.raise_for_status()
        return response

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: JsonObject | None = None,
        params: tp.Mapping[str, str | int] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        query = dict(params) if params else None
        if timeout is None:
            return self._send(method, path, json=json, params=query)
        return self._send(method, path, json=json, params=query, timeout=timeout)

    def _send(self, method: str, path: str, **kwargs: tp.Any) -> httpx.Response:
        """Issue one request, healing the one failure a retry is *provably* safe for.

        A pooled keep-alive the server already closed surfaces as
        ``RemoteProtocolError`` on the first request that reuses it. This can
        happen when polling is slower than the worker's keep-alive idle timeout
        and must not be mistaken for rank death. Retrying that one shape, for
        reads only, is safe by construction: no bytes reached
        the server, so nothing was executed twice. Mutations are never retried
        here; their fencing is the caller's to redo.
        """

        trace_headers = outgoing_trace_headers()
        headers = {**self._headers, **trace_headers} if trace_headers else self._headers
        try:
            return self.client.request(method, path, headers=headers, **kwargs)
        except httpx.RemoteProtocolError:
            if method != "GET":
                raise
            return self.client.request(method, path, headers=headers, **kwargs)

    def worker_builds(self) -> list[dict[str, object]]:
        """Return this cell's one native worker, in the slice executor's shape.

        A single-host cell has exactly one worker and it is this one. Expose it
        through the same probe as a slice so build checks cannot mistake a
        missing probe for evidence that the cell has no separate worker.
        """

        reported = self.build()
        return [
            {
                "rank": 0,
                "endpoint": self.endpoint,
                "build": None if reported is None else reported.model_dump(mode="json"),
            }
        ]

    def build(self) -> AgentBuild | None:
        """Return what code this worker runs, or ``None`` if it cannot say.

        Never raises. One rank that predates the route, or that cannot be
        reached at all, must not fail a sweep across a whole slice -- the caller
        distinguishes the ranks that answered from the ranks that did not, and
        an exception here would collapse that into "the slice is unknown".
        """

        try:
            response = self._request("GET", "/api/v1/build", timeout=5.0 if self._owns_client else None)
        except httpx.HTTPError:
            return None
        if response.status_code != 200:
            return None
        try:
            return AgentBuild.model_validate_json(response.content)
        except ValidationError:
            return None

    def health(self) -> bool:
        """Return whether the native worker accepts HTTP requests."""

        try:
            response = self._request(
                "GET",
                "/api/v1/health",
                timeout=5.0 if self._owns_client else None,
            )
            if response.status_code != 200:
                return False
            NativeWorkerHealth.model_validate_json(response.content)
            return True
        except (httpx.HTTPError, ValidationError, ValueError):
            return False

    def submit(self, spec: tp.Mapping[str, JsonValue], context: MutationContext) -> str:
        response = self._raise_for_status(
            self._request(
                "POST",
                "/api/v1/workloads",
                json=model_json(WorkloadRequest(mutation=self._mutation(context), spec=dict(spec))),
            )
        )
        return WorkloadReference.model_validate(response.json()).remote_id

    def status(self, remote_id: str) -> ExecutionWorkloadObservation:
        response = self._raise_for_status(self._request("GET", f"/api/v1/workloads/{remote_id}"))
        payload = response.json()
        try:
            observation = AgentWorkloadObservation.model_validate(payload)
        except ValidationError as exc:
            detail = ""
            if is_json_object(payload):
                message = payload.get("message")
                if isinstance(message, str) and message.strip():
                    detail = f"worker message {message.strip()!r}; "
                if "status" not in payload:
                    detail = f"omitted status; {detail}"
            raise ValueError(f"{detail}invalid native worker observation: {exc}") from exc
        return ExecutionWorkloadObservation(
            remote_id=observation.remote_id,
            status=observation.status,
            message=observation.message,
        )

    def stop(self, remote_id: str, context: MutationContext) -> None:
        self._raise_for_status(
            self._request(
                "POST",
                f"/api/v1/workloads/{remote_id}/stop",
                json=model_json(WorkloadMutationRequest(mutation=self._mutation(context))),
            )
        )

    def retire(self, remote_id: str, context: MutationContext) -> None:
        self._raise_for_status(
            self._request(
                "POST",
                f"/api/v1/workloads/{remote_id}/retire",
                json=model_json(WorkloadMutationRequest(mutation=self._mutation(context))),
                timeout=self._retire_timeout_s if self._owns_client else None,
            )
        )

    def checkpoint(
        self,
        remote_id: str,
        *,
        hook: str,
        output_uri: str | None,
        context: MutationContext,
    ) -> str:
        response = self._raise_for_status(
            self._request(
                "POST",
                f"/api/v1/workloads/{remote_id}/checkpoint",
                json=model_json(
                    CheckpointRequest(
                        mutation=self._mutation(context),
                        hook=hook,
                        output_uri=output_uri,
                    )
                ),
            )
        )
        return WorkloadReference.model_validate(response.json()).remote_id

    def logs(self, remote_id: str) -> str:
        return self._raise_for_status(self._request("GET", f"/api/v1/workloads/{remote_id}/logs")).text

    def log_page(self, remote_id: str, *, after: str = "", limit: int = 200) -> LogPage:
        """Fetch one structured page from the worker that owns this workload.

        This is the hop that was missing. Both ends already had everything: the
        worker's systemd runtime produces pages, and `log_records` knows how to
        merge and render them -- there was simply no route between them, so the
        capability was absent on precisely the fleets it was written for.

        A transport failure becomes an unavailable page rather than an
        exception, matching what the systemd runtime does locally. `logs()`
        raises because a string cannot say "the host is unreachable"; a page can.
        """

        try:
            response = self._request(
                "GET", f"/api/v1/workloads/{remote_id}/log-page", params={"after": after, "limit": limit}
            )
        except httpx.HTTPError as error:
            return LogPage(unavailable=True, detail=f"native worker was unreachable: {error}")
        if response.status_code == 501:
            return LogPage(unavailable=True, detail=response.text[:500])
        if response.status_code >= 400:
            return LogPage(unavailable=True, detail=f"native worker answered {response.status_code}")
        return log_page_from_json(response.json())

    def measure(self, cell_id: str, asset: str, *, max_bytes: int) -> AssetStageResult:
        """Return what this asset would occupy on the worker's disk, fetching nothing."""

        return self._stage_action(cell_id, CellStageAction.MEASURE, asset, max_bytes=max_bytes)

    def stage(self, cell_id: str, asset: str, *, max_bytes: int) -> AssetStageResult:
        """Materialize the asset on the worker's disk and report the bytes it proved."""

        return self._stage_action(cell_id, CellStageAction.STAGE, asset, max_bytes=max_bytes)

    def evict(self, cell_id: str, asset: str) -> AssetStageResult:
        """Remove every copy of the asset from the worker's disk."""

        return self._stage_action(cell_id, CellStageAction.EVICT, asset, max_bytes=0)

    def _stage_action(
        self,
        cell_id: str,
        action: CellStageAction,
        asset: str,
        *,
        max_bytes: int,
    ) -> AssetStageResult:
        """Issue one staging action against the worker that owns this cell's disk.

        The three answers this hop can add to the executor's own are separated
        on purpose, because the control plane acts on each differently.

        *No executor* -- 404 from a worker that predates the route, 501 from one
        that declares no source prefixes -- is raised rather than returned. It is
        not a verdict about the asset, and returning it as one would either
        block the asset permanently or charge a budget for bytes nobody
        attempted to fetch.

        *A rejected request* -- 400 or 422 -- is refused. The worker read it and
        said no, and the same bytes get the same answer tomorrow.

        *Anything else* -- a credential the worker would not accept, a 5xx, a
        timeout, a body that will not parse -- is inconclusive. None of it is
        evidence about the disk. A worker rebuilt under a new provider
        generation answers 401 to a driver still holding the old credential, and
        that says nothing about whether the asset could be staged; recording it
        as a refusal would block the asset long after the rebuild that caused
        it.
        """

        request = NativeStageActionRequest(action=action, asset=asset, max_bytes=max_bytes)
        try:
            response = self._request(
                "POST",
                f"/api/v1/cells/{cell_id}/stage-actions",
                json=model_json(request),
                timeout=_STAGE_TIMEOUTS[action] if self._owns_client else None,
            )
        except httpx.HTTPError as error:
            return AssetStageResult(
                verdict=AssetStageVerdict.INCONCLUSIVE,
                detail=f"native worker was unreachable: {error}",
            )
        if response.status_code in {404, 501}:
            raise UnsupportedCapabilityError(
                f"the native worker holding cell {cell_id!r} has no staging executor: {response.text[:500]}"
            )
        if response.status_code in {400, 422}:
            return AssetStageResult(
                verdict=AssetStageVerdict.REFUSED,
                detail=f"the native worker rejected this staging request: HTTP {response.status_code}",
            )
        if response.status_code >= 400:
            return AssetStageResult(
                verdict=AssetStageVerdict.INCONCLUSIVE,
                detail=f"native worker answered HTTP {response.status_code}",
            )
        try:
            result = CellStageActionResult.model_validate_json(response.content)
        except (ValidationError, ValueError) as error:
            return AssetStageResult(
                verdict=AssetStageVerdict.INCONCLUSIVE,
                detail=f"native worker returned an unreadable staging result: {error}",
            )
        return AssetStageResult(
            verdict=_STAGE_VERDICTS[result.verdict],
            size_bytes=result.size_bytes,
            digest=result.digest,
            detail=result.detail,
            path=result.path,
        )

    def process_status(self, remote_id: str) -> JsonObject:
        """Return live process detail for one workload on this host."""

        response = self._raise_for_status(self._request("GET", f"/api/v1/workloads/{remote_id}/process-status"))
        payload = response.json()
        if not is_json_object(payload):
            raise ValueError("native worker process status must be a JSON object")
        return payload

    def exec_in_workload(self, remote_id: str, command: tp.Sequence[str], *, timeout_s: float = 30.0) -> JsonObject:
        """Run one diagnostic command beside a workload on this host.

        Owned transports wait for the command's execution budget plus response
        overhead, rather than expiring at the default 30-second HTTP timeout.
        Caller-supplied clients retain their own timeout policy.
        """

        response = self._raise_for_status(
            self._request(
                "POST",
                f"/api/v1/workloads/{remote_id}/exec",
                json={"command": list(command), "timeout_s": timeout_s},
                timeout=min(timeout_s + _EXEC_HTTP_TIMEOUT_MARGIN_S, _EXEC_HTTP_TIMEOUT_MAX_S)
                if self._owns_client
                else None,
            )
        )
        payload = response.json()
        if not is_json_object(payload):
            raise ValueError("native worker exec result must be a JSON object")
        return payload

    def close(self) -> None:
        """Close the internally owned HTTP connection pool."""

        if self._owns_client:
            self.client.close()
        self._headers.clear()


__all__ = ("NativeRemoteExecutor",)
