# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Operating a service: proving it answers, replacing it, and resizing it.

Three leader phases, each off by default, and all three built on the same shape
the rest of this controller uses -- read in a transaction, close it, talk to the
world, reopen a transaction and re-check the generation the plan was made
against. A health probe is a network round trip against a machine that may be
exactly the unreachable one; holding a row lock across it would park the
transaction for the whole timeout and stall every phase behind it.

The one rule that shapes everything here is that a probe has three answers, not
two. A 503 is the replica saying it is not serving. A timeout is *this control
plane* saying it could not reach the replica, which is a statement about a
network and not about a process -- and acting on it as if it were a verdict
deletes healthy accelerators fleet-wide and simultaneously, because whatever
broke reachability broke it for every replica at once.

Nothing here changes a default. ``service_health_interval_s``,
``service_replica_lifecycle_interval_s`` and ``service_autoscale_interval_s``
are all zero unless an operator sets them, a service is probed only once someone
writes a health path onto it, and autoscaling additionally requires the service
to name both replica bounds. A fleet upgraded into this build behaves exactly as
it did before it.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
import typing as tp
from datetime import datetime

import httpx

from eyvan._internal.observability import trace_span
from eyvan.records import (
    CellRecord,
    CoordinationKind,
    LeaseRecord,
    ServiceDecisionRecord,
    ServiceObservation,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.server.controller._support import logger
from eyvan.server.persistence.base import RepositorySession
from eyvan.server.service.errors import ConflictError, InputError, NotFoundError
from eyvan.server.service.services import (
    ReplicaObservation,
    ReplicaVerdict,
    ServiceHealthProbe,
    ServicePolicy,
    ServiceProbeResult,
    apply_probe,
    plan_autoscale,
    plan_replica_lifecycle,
    probe_from_response,
    probe_unreachable,
    service_observation,
    service_policy,
)
from eyvan.server.service.state import TERMINAL_WORKLOAD_STATES

#: Largest health-response body this control plane will read. A health endpoint
#: answers with a status and at most a few numbers; anything larger is either a
#: misconfigured route or a replica pointing the leader at a log stream, and the
#: leader reading it to the end is a denial of service one bad service away.
MAX_HEALTH_BODY_BYTES = 64 * 1024


def health_probe_url(endpoint: str, path: str) -> str:
    """Join a registered endpoint and a health path without inventing either.

    Args:
        endpoint: The address the replica registered for itself.
        path: The absolute health path from the service policy.

    Returns:
        The URL to probe.
    """

    return f"{endpoint.rstrip('/')}/{path.lstrip('/')}"


def _checked_probe_address(value: str) -> str:
    """Normalize mapped IPv4 before applying the registered-endpoint policy."""

    address = ipaddress.ip_address(value)
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    if address.is_unspecified:
        raise ValueError("service endpoint must not resolve to an unspecified address")
    if (
        address.is_loopback
        or address.is_link_local
        or (address.version == 6 and address in ipaddress.ip_network("fc00::/7"))
    ):
        raise ValueError("service endpoint must not resolve to a loopback, link-local, or unique-local address")
    return str(address)


class _PinnedHealthTransport(httpx.AsyncHTTPTransport):
    """Resolve once, validate every answer, and give HTTPX only a numeric host.

    Merely checking DNS before a normal HTTPX request leaves a second lookup
    free to rebind the name. Pinning the URL's host prevents that lookup; the
    original Host header and TLS SNI/verification hostname remain unchanged.
    """

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        async def pinned_request() -> httpx.Response:
            host = request.url.host
            try:
                ipaddress.ip_address(host)
            except ValueError:
                resolved = await asyncio.get_running_loop().getaddrinfo(host, request.url.port, type=socket.SOCK_STREAM)
                addresses = [_checked_probe_address(entry[4][0]) for entry in resolved]
            else:
                addresses = [_checked_probe_address(host)]
            addresses = list(dict.fromkeys(addresses))
            if not addresses:
                raise OSError("service endpoint DNS returned no addresses")
            # HTTPX already built Host from the original URL. httpcore uses
            # sni_hostname for both SNI and certificate hostname verification.
            for index, address in enumerate(addresses):
                pinned = httpx.Request(
                    request.method,
                    request.url.copy_with(host=address),
                    headers=request.headers,
                    stream=request.stream,
                    extensions={**request.extensions, "sni_hostname": host},
                )
                try:
                    return await super(_PinnedHealthTransport, self).handle_async_request(pinned)
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    if index == len(addresses) - 1:
                        raise
            raise OSError("service endpoint DNS returned no addresses")  # pragma: no cover

        timeout = request.extensions.get("timeout", {}).get("connect")
        try:
            # Include DNS in the connection budget instead of allowing a slow
            # resolver to stall a complete health pass indefinitely.
            return await asyncio.wait_for(pinned_request(), timeout=timeout)
        except TimeoutError as exc:
            raise httpx.ConnectTimeout("service endpoint resolution/connection timed out", request=request) from exc


class HttpServiceHealthProbe:
    """Ask one replica over HTTP whether it is serving.

    The shipped implementation of
    :class:`~eyvan.server.service.services.ServiceHealthProbe`, and the reason
    that protocol exists rather than this class being called directly: the
    transport is the one part of health checking that cannot be tested without a
    network, and a fleet whose replicas speak something else should be able to
    bind its own without forking the reconciler.

    Every transport failure becomes :data:`ServiceProbeOutcome.UNKNOWN`. That is
    the whole contract: a refused connection, a DNS failure, a TLS error and a
    timeout are all facts about reachability, and none of them is the replica
    saying anything.
    """

    async def __call__(self, *, endpoint: str, path: str, timeout_s: float) -> ServiceProbeResult:
        """Probe one replica once.

        Args:
            endpoint: The address the replica registered for itself.
            path: The absolute health path from the service policy.
            timeout_s: How long to wait before giving up and reporting unknown.

        Returns:
            What the exchange established, which may be that it established
            nothing.
        """

        url = health_probe_url(endpoint, path)
        try:
            # Check the actual HTTPX URL, not only the registration-time parser.
            # This guard also applies when deployments inject a custom transport.
            target = httpx.URL(url)
            try:
                ipaddress.ip_address(target.host)
            except ValueError:
                pass
            else:
                _checked_probe_address(target.host)
            # HTTPX timeouts apply per I/O. A peer trickling bytes must not
            # extend the total leader budget through an unbounded body read.
            async with asyncio.timeout(timeout_s):
                async with httpx.AsyncClient(
                    timeout=timeout_s,
                    follow_redirects=False,
                    trust_env=False,
                    transport=_PinnedHealthTransport(trust_env=False),
                ) as client:
                    async with client.stream("GET", target) as response:
                        status = response.status_code
                        body = await _bounded_body(response)
        except TimeoutError:
            return probe_unreachable(f"TimeoutError: health probe exceeded total deadline of {timeout_s}s")
        except (httpx.HTTPError, OSError, ValueError) as exc:
            return probe_unreachable(f"{type(exc).__name__}: {exc}")
        return probe_from_response(status, body)


async def _bounded_body(response: httpx.Response) -> str:
    """Read at most :data:`MAX_HEALTH_BODY_BYTES` of one health response."""

    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        chunks.append(chunk)
        total += len(chunk)
        if total >= MAX_HEALTH_BODY_BYTES:
            break
    return b"".join(chunks)[:MAX_HEALTH_BODY_BYTES].decode("utf-8", errors="replace")


class _PlannedProbe(tp.NamedTuple):
    """One replica this pass intends to probe, frozen at the moment it was read.

    ``generation`` is the whole point of carrying a tuple rather than an id: the
    answer that comes back is only about the process that was running when the
    probe went out, and a replica requeued in the meantime must not inherit it.
    """

    replica_id: str
    generation: int
    endpoint: str
    authority_updated_at: datetime


class _ControllerServices:
    """Health, replica lifecycle, and bounded autoscaling for managed services.

    A concern of :class:`~eyvan.server.controller.EyvanController`; not
    instantiated on its own.
    """

    #: How a replica is asked whether it is serving. A class attribute rather
    #: than a constructor argument because it is stateless and because every
    #: deployment that has not written a health path onto a service never calls
    #: it at all. A deployment whose replicas speak something other than HTTP
    #: assigns its own.
    service_health_probe: ServiceHealthProbe = HttpServiceHealthProbe()

    async def probe_service_health_once(self) -> int:
        """Probe every registered replica of every service that declares a path.

        Three steps in the order that keeps the network out of the database:
        read the replicas and their stored counters in one transaction, close
        it, probe outside every transaction, then write the answers back in one
        short transaction that re-reads each replica and refuses an answer about
        a generation that has since moved.

        Returns:
            How many durable health records this pass wrote.
        """

        recorded = 0
        for service in await self._live_services():
            policy = self._policy_of(service)
            if policy is None:
                continue
            with trace_span("eyvan.service_health", service=service.id):
                recorded += await self._probe_one_service(service.id, policy)
        return recorded

    async def reconcile_service_replicas_once(self) -> int:
        """Replace replicas whose claim on a host is provably gone.

        A replica is retired, never duplicated: the successor is started by the
        ordinary replica-count reconciliation once this one is terminal, so at
        no point do two processes believe they are the same replica. Every
        retirement is fenced on the replica's generation inside the transaction
        that acts, exactly as workload dispatch is fenced on a cell's -- without
        that, a requeue between observing a replica and retiring it means the
        cancellation lands on the replacement.

        Returns:
            How many replicas were retired in this pass.
        """

        replaced = 0
        for service in await self._live_services():
            policy = self._policy_of(service)
            if policy is None:
                continue
            observations = await self._observe_replicas(service.id)
            observed = service_observation(policy, observations)
            verdicts = [verdict for verdict in plan_replica_lifecycle(policy, observations) if verdict.replace]
            if not verdicts:
                continue
            # Bounded by the availability the operator declared for a roll. The
            # same number governs both because they are the same question: how
            # many replicas of this service may be missing at once.
            budget = max(1, policy.max_unavailable)
            for verdict in verdicts[:budget]:
                decision = await self._replace_one_replica(service.id, verdict, observed)
                if decision is None:
                    continue
                # After the commit, never before: a line claiming a replacement
                # that a fence then refused is the one an operator reading back
                # an incident would take as fact.
                logger.info(
                    "replaced replica %s of service %s: %s",
                    verdict.replica_id,
                    service.id,
                    verdict.reason,
                )
                replaced += 1
        return replaced

    async def autoscale_services_once(self) -> int:
        """Resize every service whose operator turned autoscaling on.

        The decision is a pure function of durable rows -- the policy an operator
        wrote, the replicas' own records, and the newest committed decision,
        which is also the cooldown clock. Nothing here reads a provider, a clock
        skew, or an in-memory counter, so the whole decision table is testable
        without any of them.

        Returns:
            How many services had their declared replica count changed.
        """

        scaled = 0
        for service in await self._live_services():
            policy = self._policy_of(service)
            if policy is None or not policy.autoscale:
                continue
            observations = await self._observe_replicas(service.id, with_placement=False)
            observed = service_observation(policy, observations)
            async with self.repository.transaction() as transaction:
                newest = await transaction.list_service_decisions(service.id, limit=1)
            decision = plan_autoscale(
                policy,
                observed,
                now=utcnow(),
                last_decision=newest[0] if newest else None,
            )
            if not decision.changes:
                logger.debug("service %s holds at %d replicas: %s", service.id, decision.to_replicas, decision.reason)
                continue
            committed = await self._commit_scale(
                service,
                from_replicas=decision.from_replicas,
                to_replicas=decision.to_replicas,
                reason=decision.reason,
                observed=observed,
            )
            if committed is not None:
                scaled += 1
        return scaled

    # ----------------------------------------------------------------- reading

    async def _live_services(self) -> list[WorkloadRecord]:
        """Return the services this leader may act on.

        A service coordinator never runs, so it stays ``PENDING`` for its whole
        life; that is what makes this one indexed read rather than a scan.
        """

        async with self.repository.transaction() as transaction:
            workloads = await transaction.list_workloads(state=WorkloadState.PENDING)
        return [workload for workload in workloads if workload.coordination_kind is CoordinationKind.SERVICE]

    def _policy_of(self, service: WorkloadRecord) -> ServicePolicy | None:
        """Return one service's declared policy, or ``None`` when it has none.

        A malformed policy is logged and skipped rather than raised. The phase
        runs over every service in the fleet, and one tenant's bad metadata must
        not stop another tenant's replicas being probed.
        """

        try:
            return service_policy(service.spec.metadata)
        except InputError:
            logger.warning("service %s has a malformed policy; skipping", service.id, exc_info=True)
            return None

    async def _observe_replicas(
        self,
        service_id: str,
        *,
        with_placement: bool = True,
    ) -> list[ReplicaObservation]:
        """Read one service's replicas and the durable rows their claims rest on.

        Args:
            service_id: The service whose replicas are being read.
            with_placement: Whether to resolve each replica's lease and cell.
                The lifecycle planner needs them -- they are what proves a claim
                on a host is stale -- while the autoscaler reads only counts and
                health, so it skips two lookups per replica.

        Returns:
            One observation per replica, in stable replica-id order.
        """

        async with self.repository.transaction() as transaction:
            children = await transaction.list_child_workloads(service_id)
            health = {record.replica_id: record for record in await transaction.list_service_replica_health(service_id)}
            placement: dict[str, tuple[LeaseRecord | None, CellRecord | None]] = {}
            if with_placement:
                for child in children:
                    placement[child.id] = await self._placement_of(transaction, child)
        return [
            ReplicaObservation(
                replica=child,
                lease=placement.get(child.id, (None, None))[0],
                cell=placement.get(child.id, (None, None))[1],
                health=health.get(child.id),
            )
            for child in sorted(children, key=lambda child: child.id)
        ]

    @staticmethod
    async def _placement_of(
        transaction: RepositorySession,
        replica: WorkloadRecord,
    ) -> tuple[LeaseRecord | None, CellRecord | None]:
        """Resolve the lease and cell one replica claims, without inventing either."""

        if replica.lease_id is None:
            return None, None
        lease = await transaction.get_lease(replica.lease_id)
        if lease is None:
            return None, None
        return lease, await transaction.get_cell(lease.cell_id)

    # ------------------------------------------------------------------ health

    async def _probe_one_service(self, service_id: str, policy: ServicePolicy) -> int:
        """Collect one service's dead counters, then probe the replicas it has.

        The collection runs even for a service that declares no health path, so
        removing a path stops the probing without stranding the rows the probing
        left behind -- otherwise those rows would only ever be collected by the
        very phase that had just been told to stop looking at them.
        """

        async with self.repository.transaction() as transaction:
            children = await transaction.list_child_workloads(service_id)
            stored = {record.replica_id: record for record in await transaction.list_service_replica_health(service_id)}
        live = {child.id for child in children if child.state not in TERMINAL_WORKLOAD_STATES}
        await self._forget_retired_health(service_id, [key for key in stored if key not in live])
        if not policy.health_path:
            return 0
        planned = [
            _PlannedProbe(
                replica_id=child.id,
                generation=child.generation,
                endpoint=child.service_endpoint,
                authority_updated_at=child.updated_at,
            )
            # RUNNING says a process started; a registered endpoint says it
            # answered. Probing an address nobody published would make every
            # replica unknown for the whole of its startup and call that a fact.
            for child in children
            if child.state not in TERMINAL_WORKLOAD_STATES and child.service_endpoint
        ]
        if not planned:
            return 0

        # Bounded fan-out: a fleet with a thousand replicas must not open a
        # thousand sockets from the leader on one pass.
        gate = asyncio.Semaphore(max(1, self.settings.service_health_concurrency))

        async def probe(target: _PlannedProbe) -> tuple[_PlannedProbe, ServiceProbeResult]:
            async with gate:
                return target, await self._probe_once(target, policy)

        results = await asyncio.gather(*(probe(target) for target in planned))
        return await self._record_probes(service_id, results)

    async def _probe_once(self, target: _PlannedProbe, policy: ServicePolicy) -> ServiceProbeResult:
        """Run one probe outside every transaction, turning any escape into unknown.

        A probe implementation that raises has still established nothing about
        the replica, so it lands where every other unreachability lands. Letting
        it propagate would lose the whole pass -- including the definite answers
        every other replica already gave.
        """

        try:
            # Bound custom probes as well as the shipped HTTP implementation.
            async with asyncio.timeout(policy.health_timeout_s):
                return await self.service_health_probe(
                    endpoint=target.endpoint,
                    path=policy.health_path,
                    timeout_s=policy.health_timeout_s,
                )
        except Exception as exc:  # a probe transport may raise anything at all
            logger.debug("health probe of replica %s raised", target.replica_id, exc_info=True)
            return probe_unreachable(f"the health probe raised {type(exc).__name__}: {exc}")

    async def _record_probes(
        self,
        service_id: str,
        results: tp.Sequence[tuple[_PlannedProbe, ServiceProbeResult]],
    ) -> int:
        """Fold one pass's probe results into durable counters, each one fenced.

        One transaction for the whole service rather than one per replica: this
        reads the service's health rows, and a per-replica transaction would read
        and lock all of them once per replica. A thousand-replica service would
        then do a thousand locking scans of a thousand rows on every pass, which
        is a cost that grows as the square of a number an operator chose.

        There is no external I/O in here -- every probe has already answered --
        so the transaction is bounded by database work alone.

        Args:
            service_id: The service the answers are about.
            results: What each planned probe established.

        Returns:
            How many records were written. An answer about a replica that has
            since been requeued or retired is dropped, which is ordinary: the
            next pass probes what is actually there now.
        """

        recorded = 0
        async with self.repository.transaction() as transaction:
            stored = {record.replica_id: record for record in await transaction.list_service_replica_health(service_id)}
            for target, result in results:
                replica = await transaction.get_workload(target.replica_id)
                if replica is None or replica.parent_workload_id != service_id:
                    continue
                if (
                    replica.generation != target.generation
                    or replica.state in TERMINAL_WORKLOAD_STATES
                    or replica.service_endpoint != target.endpoint
                    # updated_at is a conservative authority fence: it also
                    # detects an endpoint that changed away and back (ABA).
                    or replica.updated_at != target.authority_updated_at
                ):
                    continue
                await transaction.put_service_replica_health(
                    apply_probe(
                        stored.get(target.replica_id),
                        result,
                        service_id=service_id,
                        replica_id=target.replica_id,
                        replica_generation=target.generation,
                        endpoint=target.endpoint,
                        now=utcnow(),
                    )
                )
                recorded += 1
        return recorded

    async def _forget_retired_health(self, service_id: str, replica_ids: tp.Sequence[str]) -> None:
        """Drop counters for replicas that no longer exist.

        A service is the one workload shape designed to restart, so keeping a row
        per dead replica is unbounded growth by construction. This is the only
        collector those rows have.
        """

        for replica_id in sorted(replica_ids):
            async with self.repository.transaction() as transaction:
                await transaction.delete_service_replica_health(service_id, replica_id)

    # --------------------------------------------------------------- lifecycle

    async def _replace_one_replica(
        self,
        service_id: str,
        verdict: ReplicaVerdict,
        observed: ServiceObservation,
    ) -> ServiceDecisionRecord | None:
        """Retire one replica, tolerating the races that are ordinary here."""

        try:
            return await self.service.replace_service_replica(
                self._principal,
                service_id=service_id,
                replica_id=verdict.replica_id,
                expected_generation=verdict.generation,
                reason=verdict.reason,
                observed=observed,
                history_limit=self.settings.service_decision_history,
            )
        except (ConflictError, NotFoundError):
            # It finished, was cancelled, or was requeued between the plan and
            # the act. The next pass observes what is actually there.
            logger.debug(
                "replica %s of service %s could not be retired",
                verdict.replica_id,
                service_id,
                exc_info=True,
            )
            return None

    # -------------------------------------------------------------- autoscaling

    async def _commit_scale(
        self,
        service: WorkloadRecord,
        *,
        from_replicas: int,
        to_replicas: int,
        reason: str,
        observed: ServiceObservation,
    ) -> ServiceDecisionRecord | None:
        """Commit one bounded resize, fenced on the service's own generation.

        Logged after the commit rather than before it, so the log never claims a
        resize that a fence refused -- which is the one line an operator reading
        back an incident would take as fact.
        """

        try:
            decision = await self.service.scale_service(
                self._principal,
                service_id=service.id,
                expected_generation=service.generation,
                from_replicas=from_replicas,
                to_replicas=to_replicas,
                reason=reason,
                observed=observed,
                history_limit=self.settings.service_decision_history,
            )
        except (ConflictError, NotFoundError):
            logger.debug("service %s could not be resized", service.id, exc_info=True)
            return None
        if decision is None:
            logger.info(
                "service %s was not resized from %d to %d: the observation was superseded",
                service.id,
                from_replicas,
                to_replicas,
            )
            return None
        logger.info(
            "scaled service %s from %d to %d replicas: %s",
            service.id,
            from_replicas,
            to_replicas,
            reason,
        )
        return decision


__all__ = (
    "MAX_HEALTH_BODY_BYTES",
    "HttpServiceHealthProbe",
    "_ControllerServices",
    "health_probe_url",
)
