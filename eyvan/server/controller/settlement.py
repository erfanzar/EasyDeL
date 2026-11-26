# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Terminal release, billing, retention, and what capacity failures teach."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import typing as tp
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import timedelta

from eyvan._internal.compat import StrEnum
from eyvan._internal.json_types import JsonObject, JsonValue
from eyvan.artifacts.managed import ManagedArtifactKind, log_manifest_uri, log_uri
from eyvan.capacity import (
    CapacityStatus,
)
from eyvan.capacity.drivers.base import CapacityProviderFailureCode, provision_outcome
from eyvan.domain import (
    CapacityBelief,
    CapacityPoolKey,
    ProvisionOutcome,
)
from eyvan.records import (
    AuditEvent,
    AutoscaleBlockReason,
    CapacityBeliefState,
    CellRecord,
    CellState,
    CoordinationKind,
    GcpQueueAdmissionDenialTarget,
    LeaseRecord,
    LeaseState,
    ManagedArtifactRecord,
    ManagedArtifactState,
    OutboxRecord,
    WorkloadRecord,
    WorkloadState,
    candidate_pool_templates,
    utcnow,
)
from eyvan.server.controller._support import (
    _LIVE_WORKLOAD_STATES,
    StaleOutboxCommand,
    _provider_phase_start,
    logger,
)
from eyvan.server.invariants import Invariant, check_fleet
from eyvan.server.persistence.base import RepositorySession
from eyvan.server.service.errors import AuthorizationError, ConflictError, InputError, NotFoundError
from eyvan.server.service.provider_diagnostics import safe_provider_diagnostic
from eyvan.server.service.services import CHILD_REVISION, ServicePolicy, child_revision, plan_service, service_policy
from eyvan.server.service.state import ACTIVE_LEASE_STATES, TERMINAL_WORKLOAD_STATES
from eyvan.worker.runtime.log_records import (
    LOG_ARCHIVE_LEGACY_RANK_SCAN_LIMIT,
    LOG_ARCHIVE_MANIFEST_KEY,
    LOG_ARCHIVE_MANIFEST_VERSION,
)

#: Lines archived per runtime rank. A ceiling rather than everything, because
#: the archive is written inside the release path: an unbounded read would make
#: settling a chatty job slow enough to matter, and the last lines are what
#: explain a failure. One extra line is requested so hitting the ceiling can be
#: distinguished from exceeding it and truncation is explicit in the manifest.
_LOG_ARCHIVE_LINE_LIMIT = 1000

#: Schema version of the *standalone* terminal archive manifest object.
#:
#: Deliberately not the same tag as ``LOG_ARCHIVE_MANIFEST_VERSION``. That
#: version belongs to the array line embedded inside rank zero's own blob and
#: keeps its meaning unchanged; this one belongs to a different object at a
#: different URI. Two payload schemas never share one version tag, so a rolling
#: reader that knows only the embedded index cannot be handed a document it
#: would misread.
LOG_ARCHIVE_MANIFEST_SCHEMA_V2 = 2

#: Ceiling on the number of physical rank objects one settlement may write.
#:
#: A declared attempt width is bounded only by ``runtime_process_count`` per
#: cell times the gang size, so "one object per expected rank" is otherwise an
#: unbounded amount of I/O inside the release path -- the path that must not be
#: slow enough to strand an accelerator. Ranks that produced output, or whose
#: fate is anything other than "silent", are written first; the manifest still
#: records every remaining expected rank, and says how many objects it left out.
_LOG_ARCHIVE_MAX_RANK_OBJECTS = 1024

#: Ceiling on how many ranks the manifest enumerates. A manifest that could not
#: name every expected rank says so through ``enumeration_capped`` and is never
#: reported ``complete``.
_LOG_ARCHIVE_MAX_MANIFEST_RANKS = 10_000

#: Operator switch for the DR-0002 stage-2 manifest object.
#:
#: Default off. DR-0002's own rollout ships a feature-disabled v2 writer first
#: and switches emission only once every serving and rollback image carries the
#: reader; enabling it here changes nothing an existing reader parses, because
#: the embedded v1 index keeps being written byte-for-byte and the new object
#: lives at a URI no v1 reader looks at. Read through ``EyvanSettings`` first so
#: a later typed setting supersedes the variable without touching this module.
_LOG_ARCHIVE_MANIFEST_V2_SETTING = "log_archive_manifest_v2_enabled"
_LOG_ARCHIVE_MANIFEST_V2_ENVIRONMENT_VARIABLE = "EYVAN_LOG_ARCHIVE_MANIFEST_V2_ENABLED"


def _replica_idempotency_suffix(service_id: str, revision: str, slot: int) -> str:
    """Return the stable fingerprint one replica submit is minted under.

    A random suffix would give two leaders watching one service two different
    keys for the same replica -- the duplicate this key exists to prevent --
    so the key is a hash of exactly the facts that name the replica.
    """

    material = f"{service_id}:{revision}:{slot}".encode()
    return hashlib.sha256(material).hexdigest()[:12]


class LogArchiveRankStatus(StrEnum):
    """What became of one expected global rank in a terminal archive.

    The distinction the embedded v1 index cannot make: it lists observed ranks
    only, so a rank that legitimately printed nothing, a rank whose host was
    unreachable, and a rank whose object failed to upload are all equally
    absent from it.
    """

    #: Everything the rank produced is in its object.
    ARCHIVED = "archived"
    #: The rank was reachable and produced nothing. Its object exists and is empty.
    QUIET = "quiet"
    #: Lines were captured, but the per-rank ceiling dropped the oldest ones.
    TRUNCATED = "truncated"
    #: The live read could not reach this rank, so nothing was captured for it.
    UNREACHABLE = "unreachable"
    #: The object could not be committed; its contents are unknown or absent.
    FAILED = "failed"


#: Statuses that mean this rank's output was captured in full.
_COMPLETE_RANK_STATUSES = frozenset({LogArchiveRankStatus.ARCHIVED, LogArchiveRankStatus.QUIET})


@dataclass(frozen=True, slots=True)
class _ArchiveTopology:
    """The attempt shape every expected global rank is derived from."""

    width: int
    digest: str
    cells: tuple[tuple[str, int, int], ...]

    def as_json(self) -> JsonObject:
        """Return the manifest projection of this topology."""

        return {
            "source": "leases",
            "width": self.width,
            "digest": self.digest,
            "cells": [
                {"cell": cell_id, "generation": generation, "width": width} for cell_id, generation, width in self.cells
            ],
        }


@dataclass(frozen=True, slots=True)
class _ArchivedRank:
    """One expected global rank's fate and the exact object that holds it."""

    rank: int
    status: LogArchiveRankStatus
    uri: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    detail: str = ""
    #: Whether a later settlement of this attempt could still land this object.
    #: An inconclusive write can; an immutable object that already holds other
    #: bytes cannot. Not part of the manifest: it describes this writer's
    #: knowledge, not the archive.
    retryable: bool = False

    def as_json(self) -> JsonObject:
        """Return the manifest projection of this rank."""

        return {
            "rank": self.rank,
            "status": self.status.value,
            "uri": self.uri,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "detail": self.detail,
        }


class _ControllerSettlement:
    """Terminal release, billing, retention, and what capacity failures teach.

    A concern of :class:`~eyvan.server.controller.EyvanController`; not
    instantiated on its own.
    """

    async def _release_terminal(
        self,
        workload: WorkloadRecord,
        lease: LeaseRecord,
        target: WorkloadState,
        action: str,
        *,
        failure_reason: str | None = None,
        expected_dispatch_command: OutboxRecord | None = None,
    ) -> None:
        if workload.lease_id != lease.id:
            raise StaleOutboxCommand(f"lease {lease.id!r} no longer owns workload {workload.id!r}")
        # Archive before settling. Settlement releases the lease, and releasing
        # a cell destroys its journal -- so the output disappears exactly when
        # the capacity bill stops, which is also when someone starts asking what
        # happened. Failures here are swallowed on purpose: a job that ran must
        # still be recorded as finished even if its logs could not be saved.
        #
        # The guard is outermost rather than only inside the per-object writes.
        # An archive that raised anywhere -- reading the page, deriving the
        # attempt topology, committing the manifest -- would abort settlement and
        # leave a finished workload holding an accelerator because its logs were
        # unreadable. Capacity is settled on every path.
        try:
            async with self._log_archive_lock(workload.id, workload.attempt):
                await self._archive_terminal_logs(workload)
        except Exception:
            logger.warning("could not archive logs for %s before release", workload.id, exc_info=True)
        await self.service.settle_terminal_workload(
            self._principal,
            workload.id,
            target,
            action=action,
            failure_reason=failure_reason,
            expected_generation=workload.generation,
            expected_lease_id=lease.id,
            expected_state=workload.state,
            expected_dispatch_command=expected_dispatch_command,
        )

    async def _archive_terminal_logs(self, workload: WorkloadRecord) -> None:
        """Copy one attempt's structured logs into managed storage.

        Best effort, and deliberately so. The alternative -- letting an archive
        failure abort settlement -- would leave a finished workload holding its
        lease because its logs were unreadable, turning a diagnostic gap into a
        stranded accelerator.

        Skipped entirely when managed artifacts are not configured, which is the
        local development case: there is nowhere to put them and the journal is
        on the same machine anyway.
        """

        if self.artifact_access is None or workload.remote_id is None or workload.spec.secret_mounts:
            # Durable secret-bearing executions intentionally have no archived
            # output: exact streaming redaction cannot be reconstructed after a
            # worker restart without persisting the plaintext values themselves.
            return
        # Resume where the incremental archive stopped, never from the start of
        # the journal. The chunks already hold everything up to that cursor, and
        # re-reading it here would store a second copy of every line and make
        # replay show each of them twice. With no chunks the cursor is empty and
        # this is the read it has always been.
        chunk_cursor = await self._last_log_chunk_cursor(workload)
        try:
            page = await self.workload_log_page(
                workload.id,
                after=chunk_cursor,
                limit=_LOG_ARCHIVE_LINE_LIMIT + 1,
                live_only=True,
            )
        except Exception:
            logger.warning("could not read logs for %s before release", workload.id, exc_info=True)
            return
        records = page.get("records")
        if not isinstance(records, list):
            return
        by_rank: dict[int, list[JsonObject]] = {}
        for entry in records:
            if not isinstance(entry, dict):
                continue
            rank = entry.get("rank")
            global_rank = rank if isinstance(rank, int) and not isinstance(rank, bool) and rank >= 0 else 0
            by_rank.setdefault(global_rank, []).append(entry)
        raw_truncated = page.get("truncated_ranks")
        unreachable = {
            rank
            for rank in (raw_truncated if isinstance(raw_truncated, list) else [])
            if isinstance(rank, int) and not isinstance(rank, bool) and rank >= 0
        }
        truncated = set(unreachable)
        # The provider was asked for one sentinel line beyond the advertised
        # ceiling. Trim independently per global rank so one chatty process does
        # not consume every other rank's diagnostic tail. Keep the last lines:
        # they are the ones nearest the terminal failure this archive explains.
        #
        # Tracked separately as well as in the union: "this rank kept only its
        # tail" and "this rank could not be read at all" are different incidents,
        # and the v1 index has one field for both.
        line_truncated: set[int] = set()
        for rank, lines in by_rank.items():
            if len(lines) > _LOG_ARCHIVE_LINE_LIMIT:
                by_rank[rank] = lines[-_LOG_ARCHIVE_LINE_LIMIT:]
                truncated.add(rank)
                line_truncated.add(rank)
        page_unavailable = bool(page.get("unavailable"))

        archived_ranks = sorted(set(by_rank) | truncated)
        manifest_ranks: list[JsonValue] = list(archived_ranks)
        manifest_truncated: list[JsonValue] = list(sorted(truncated))
        manifest_body: JsonObject = {
            "version": LOG_ARCHIVE_MANIFEST_VERSION,
            "ranks": manifest_ranks,
            "truncated_ranks": manifest_truncated,
            "unavailable": page_unavailable,
            "line_limit_per_rank": _LOG_ARCHIVE_LINE_LIMIT,
        }
        # A JSON array rather than a record-shaped object keeps old readers
        # compatible: their archive decoder ignores non-object lines instead of
        # rendering the manifest as a blank log message during a rolling upgrade.
        manifest: JsonValue = [LOG_ARCHIVE_MANIFEST_KEY, manifest_body]
        # Rank zero always carries the manifest, even when the workload emitted
        # no lines or rank zero itself was unreachable. That distinguishes a
        # successfully archived empty page from a missing archive and lets replay
        # find sparse ranks without guessing where the first gap means "end".
        #
        # Old readers scan contiguously and stop at the first gap. Bridge gaps
        # only as far as a real rank they can actually reach inside their fixed
        # compatibility window. Sparse high ranks are written directly because
        # the manifest names them; writing every intervening empty object would
        # make one corrupt/high rank an O(max_rank) release operation.
        legacy_visible = [rank for rank in archived_ranks if rank < LOG_ARCHIVE_LEGACY_RANK_SCAN_LIMIT]
        compatibility_ranks = set(range(max(legacy_visible) + 1)) if legacy_visible else {0}
        observed_ranks = {0, *archived_ranks, *compatibility_ranks}

        topology: _ArchiveTopology | None = None
        expected: set[int] = set()
        write_ranks = set(observed_ranks)
        manifest_v2 = self._log_archive_manifest_v2_enabled()
        if manifest_v2:
            # Every expected rank comes from the attempt's *declared* width, not
            # from whichever ranks happened to answer. A rank that was never
            # reachable produces no record anywhere else, so a set derived from
            # observation can only ever describe the ranks that were fine.
            topology = await self._archive_topology(workload)
            if topology is not None:
                expected = set(range(topology.width))
            headroom = max(_LOG_ARCHIVE_MAX_RANK_OBJECTS - len(write_ranks), 0)
            write_ranks |= set(sorted(expected - write_ranks)[:headroom])

        # A rank the live read could not reach at all is unreachable, not quiet.
        # When the whole read failed, that is true of every expected rank.
        unreachable_ranks = unreachable | (expected if page_unavailable else set())
        intended = {
            rank: self._intended_rank_status(
                rank,
                observed=set(by_rank),
                unreachable=unreachable_ranks,
                line_truncated=line_truncated,
            )
            for rank in sorted(write_ranks | expected)
        }

        # Concurrently, bounded by the same knob the live archiver uses. Each
        # rank's object is independent and immutable, and this runs before the
        # workload is settled -- inside the observe-workloads phase, which every
        # other workload waits behind. Serial uploads would delay terminal
        # status and observations of unrelated workloads by the sum of their latencies.
        gate = asyncio.Semaphore(max(int(self.settings.log_archive_concurrency), 1))

        async def archive_rank(rank: int) -> _ArchivedRank:
            lines: list[JsonValue] = list(by_rank.get(rank, ()))
            if rank == 0:
                lines.insert(0, manifest)
            payload = "\n".join(json.dumps(line, sort_keys=True) for line in lines).encode()
            async with gate:
                return await self._archive_rank_object(workload, rank, payload, intended=intended[rank])

        ordered_ranks = sorted(write_ranks)
        results = await asyncio.gather(*(archive_rank(rank) for rank in ordered_ranks), return_exceptions=True)
        # Every rank has finished before anything propagates: no upload is left
        # running unowned behind a failure, and the first failure in rank order
        # is the one reported, as the sequential walk reported it.
        for result in results:
            if isinstance(result, BaseException):
                raise result
        outcomes: dict[int, _ArchivedRank] = dict(zip(ordered_ranks, tp.cast(list[_ArchivedRank], results), strict=True))

        if not manifest_v2:
            return
        unfinished = sorted(rank for rank, outcome in outcomes.items() if outcome.retryable)
        if unfinished:
            # Publishing now would spend the attempt's one manifest URI on a
            # provisional answer. The URI is immutable, so a reservation taken
            # for these bytes is the reservation a later settlement -- which may
            # actually land these objects -- would need for the real ones. The
            # rank objects that did land are immutable and still there; the next
            # settlement reuses them and commits the manifest that describes the
            # finished archive.
            logger.warning(
                "log archive manifest for %s deferred; rank(s) %s may still be written",
                workload.id,
                ", ".join(str(rank) for rank in unfinished),
            )
            return
        # Last, and only now. Every object's fate is known, so the manifest
        # states what exists rather than what the writer intended to write.
        await self._commit_log_archive_manifest(
            workload,
            topology=topology,
            outcomes=outcomes,
            intended=intended,
            unavailable=page_unavailable,
        )

    def _log_archive_manifest_v2_enabled(self) -> bool:
        """Return whether this replica commits the standalone manifest object.

        Off unless an operator turns it on. DR-0002 rolls the writer out before
        the emission switch, so an image without the reader must never be the
        one that starts producing manifests other replicas depend on.
        """

        configured = getattr(self.settings, _LOG_ARCHIVE_MANIFEST_V2_SETTING, None)
        if configured is not None:
            return bool(configured)
        raw = os.getenv(_LOG_ARCHIVE_MANIFEST_V2_ENVIRONMENT_VARIABLE)
        return raw is not None and raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _intended_rank_status(
        rank: int,
        *,
        observed: set[int],
        unreachable: set[int],
        line_truncated: set[int],
    ) -> LogArchiveRankStatus:
        """Classify one expected rank from what the live read could see."""

        if rank in unreachable:
            return LogArchiveRankStatus.UNREACHABLE
        if rank in line_truncated:
            return LogArchiveRankStatus.TRUNCATED
        return LogArchiveRankStatus.ARCHIVED if rank in observed else LogArchiveRankStatus.QUIET

    async def _archive_topology(self, workload: WorkloadRecord) -> _ArchiveTopology | None:
        """Derive one attempt's declared rank space from its leased cells.

        The same ordering and per-cell width the live read uses, so a rank means
        the same physical worker in the archive as it did while the job ran.
        ``None`` when the membership cannot be resolved -- an archive that cannot
        prove the shape must not claim to be complete.
        """

        try:
            members = await self._load_workload_members(workload)
            ordered = self._ordered_gang_member_pairs(members) if len(members) > 1 else tuple(members)
        except Exception:
            logger.debug("could not resolve the archive topology for %s", workload.id, exc_info=True)
            return None
        if not ordered:
            return None
        cells = tuple((cell.id, cell.generation, cell.runtime_process_count) for _lease, cell in ordered)
        fingerprint = json.dumps(
            {
                "attempt": workload.attempt,
                "cells": [
                    {"cell": cell_id, "generation": generation, "width": width} for cell_id, generation, width in cells
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return _ArchiveTopology(
            width=sum(width for _cell, _generation, width in cells),
            digest=f"sha256:{hashlib.sha256(fingerprint).hexdigest()}",
            cells=cells,
        )

    async def _active_managed_artifact(
        self,
        workload: WorkloadRecord,
        uri: str,
    ) -> ManagedArtifactRecord | None:
        """Return the committed ledger row for ``uri``, if one already exists."""

        try:
            async with self.repository.transaction() as transaction:
                record = await transaction.get_managed_artifact(workload.org_id, workload.project_id, uri)
        except Exception:
            logger.debug("could not read the artifact ledger for %s", uri, exc_info=True)
            return None
        if record is None or record.state is not ManagedArtifactState.ACTIVE or record.sha256 is None:
            return None
        return record

    async def _archive_rank_object(
        self,
        workload: WorkloadRecord,
        rank: int,
        payload: bytes,
        *,
        intended: LogArchiveRankStatus,
    ) -> _ArchivedRank:
        """Commit or reuse one rank's immutable object and report what happened.

        Managed artifact URIs are immutable, so a settlement retry whose live
        read produced different bytes cannot overwrite its predecessor: it would
        conflict, quarantine the row, and lose the archive that already existed.
        An already-active object is therefore reused, and the manifest describes
        the bytes that are actually stored.
        """

        uri = log_uri(workload.id, workload.attempt, rank)
        digest = hashlib.sha256(payload).hexdigest()
        existing = await self._active_managed_artifact(workload, uri)
        if existing is not None:
            if existing.sha256 == digest:
                return _ArchivedRank(rank, intended, uri, existing.sha256, existing.size_bytes)
            observed = LogArchiveRankStatus.ARCHIVED if existing.size_bytes > 0 else LogArchiveRankStatus.QUIET
            return _ArchivedRank(
                rank,
                observed,
                uri,
                existing.sha256,
                existing.size_bytes,
                detail="an earlier settlement of this attempt committed different bytes",
            )
        try:
            record = await self.service.upload_managed_artifact(
                org_id=workload.org_id,
                project_id=workload.project_id,
                uri=uri,
                payload=payload,
                actor="system:log-archiver",
            )
        except Exception as exc:
            logger.warning("could not archive logs for %s rank %d", workload.id, rank, exc_info=True)
            # The class name only. This document is readable by the tenant, and a
            # provider exception body is exactly the place a credential or an
            # internal path leaks into one.
            #
            # A conflict is the object's final answer: the URI is immutable and
            # already holds something else, so no retry changes it. Anything else
            # -- an inconclusive write, an outage, a quota -- may still land, and
            # the manifest must not be published as though it had not.
            return _ArchivedRank(
                rank,
                LogArchiveRankStatus.FAILED,
                uri,
                detail=type(exc).__name__,
                retryable=not isinstance(exc, ConflictError),
            )
        if record.state is not ManagedArtifactState.ACTIVE or record.sha256 != digest:
            return _ArchivedRank(
                rank,
                LogArchiveRankStatus.FAILED,
                uri,
                record.sha256,
                record.size_bytes,
                detail="the upload did not activate the bytes it was given",
            )
        return _ArchivedRank(rank, intended, uri, record.sha256, record.size_bytes)

    async def _commit_log_archive_manifest(
        self,
        workload: WorkloadRecord,
        *,
        topology: _ArchiveTopology | None,
        outcomes: dict[int, _ArchivedRank],
        intended: dict[int, LogArchiveRankStatus],
        unavailable: bool,
    ) -> None:
        """Commit the immutable schema-v2 manifest for one terminal attempt.

        Written last and at its own URI. The manifest -- not the first missing
        object -- defines completeness, so it can only be honest once every blob
        has either landed or failed. An existing manifest is never rewritten: the
        first one committed for an attempt is the record of that attempt.
        """

        uri = log_manifest_uri(workload.id, workload.attempt)
        if await self._active_managed_artifact(workload, uri) is not None:
            return
        enumerated = sorted(set(outcomes) | set(intended))
        named = enumerated[:_LOG_ARCHIVE_MAX_MANIFEST_RANKS]
        enumeration_capped = len(named) != len(enumerated)
        entries: list[JsonValue] = []
        statuses: list[LogArchiveRankStatus] = []
        objects_omitted = 0
        for rank in named:
            outcome = outcomes.get(rank)
            if outcome is None:
                objects_omitted += 1
                outcome = _ArchivedRank(
                    rank,
                    intended.get(rank, LogArchiveRankStatus.QUIET),
                    detail="no object was written; this attempt reached its per-settlement object ceiling",
                )
            entries.append(outcome.as_json())
            statuses.append(outcome.status)
        complete = (
            topology is not None
            and not enumeration_capped
            and objects_omitted == 0
            and not unavailable
            and all(status in _COMPLETE_RANK_STATUSES for status in statuses)
        )
        body: JsonObject = {
            "schema": LOG_ARCHIVE_MANIFEST_SCHEMA_V2,
            "kind": "workload-log-archive",
            "workload": workload.id,
            "attempt": workload.attempt,
            "topology": (
                topology.as_json()
                if topology is not None
                else {"source": "unknown", "width": None, "digest": None, "cells": []}
            ),
            "line_limit_per_rank": _LOG_ARCHIVE_LINE_LIMIT,
            "object_limit": _LOG_ARCHIVE_MAX_RANK_OBJECTS,
            "objects_omitted": objects_omitted,
            "enumeration_capped": enumeration_capped,
            "unavailable": unavailable,
            "complete": complete,
            "ranks": entries,
        }
        payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        try:
            await self.service.upload_managed_artifact(
                org_id=workload.org_id,
                project_id=workload.project_id,
                uri=uri,
                payload=payload,
                actor="system:log-archiver",
            )
        except Exception:
            # The rank objects are already committed and immutable, so the next
            # settlement of this attempt reuses them and commits the manifest
            # then. Losing the manifest costs discoverability, never bytes.
            logger.warning("could not commit the log archive manifest for %s", workload.id, exc_info=True)

    async def _record_capacity_stockout(self, cell: CellRecord) -> None:
        """Persist a STOCKOUT belief for a cell reclaimed before it ever became READY.

        Guarded on the cell's generation so a concurrent recreate does not have its
        fresh belief overwritten by this late write.
        """

        availability = self._observe_capacity_belief(cell, ProvisionOutcome.STOCKOUT)
        async with self.repository.transaction() as transaction:
            current = await transaction.get_cell(cell.id)
            if current is None or current.generation != cell.generation:
                return
            await transaction.put_cell(current.model_copy(update={"availability": availability, "updated_at": utcnow()}))

    @staticmethod
    def _observe_capacity_belief(
        cell: CellRecord,
        outcome: ProvisionOutcome,
        *,
        latency_s: float | None = None,
    ) -> CapacityBeliefState:
        current = cell.availability
        belief = CapacityBelief(
            key=CapacityPoolKey(
                provider=cell.provider,
                project=cell.provider_project or "",
                zone=cell.zone or cell.region,
                accelerator_type=cell.accelerator_type or "unknown",
                capacity_class=cell.capacity_class.value,
            ),
            success_evidence=current.success_evidence,
            failure_evidence=current.failure_evidence,
            provision_latencies_s=current.provision_latencies_s,
            interruptions=current.interruptions,
            successful_hours=current.successful_hours,
            consecutive_failures=current.consecutive_failures,
            cooldown_until=current.cooldown_until,
            last_observed_at=current.last_observed_at,
        ).observe(
            outcome,
            latency_s=latency_s,
            # The cell's own copy, taken from its pool when it was created. A
            # curve retuned since then applies to the machines created after
            # it, not retroactively to how these were already being judged.
            evidence_decay=cell.availability_policy.evidence_decay,
            max_backoff_s=cell.availability_policy.max_backoff_s,
            backoff_base_s=cell.availability_policy.backoff_base_s,
            default_backoff_s=cell.availability_policy.default_backoff_s,
        )
        return CapacityBeliefState(
            success_evidence=belief.success_evidence,
            failure_evidence=belief.failure_evidence,
            provision_latencies_s=belief.provision_latencies_s,
            interruptions=belief.interruptions,
            successful_hours=belief.successful_hours,
            consecutive_failures=belief.consecutive_failures,
            cooldown_until=belief.cooldown_until,
            last_observed_at=belief.last_observed_at,
        )

    async def _record_capacity_failure(
        self,
        command: OutboxRecord,
        error: str,
        *,
        failure_code: CapacityProviderFailureCode | None,
    ) -> None:
        transition: tuple[CellRecord, CellRecord] | None = None
        async with self.repository.transaction() as transaction:
            cell = await transaction.get_cell(command.aggregate_id)
            if cell is None or cell.generation != command.generation or cell.state is not CellState.REGISTERING:
                return
            # Was: search ``error`` for the words "quota" and "capacity". That
            # read a provider's arbitrary prose as a classification, so
            # "capacity pool not found" -- a configuration mistake -- became
            # evidence that a zone had run out of accelerators, and fed it to
            # the belief that decides where the next workload is placed. The
            # provider now says which of these it means.
            outcome = provision_outcome(failure_code)
            observed_at = utcnow()
            unavailable_reason = failure_code.value if failure_code is not None else "provider_failure"
            updated = cell.model_copy(
                update={
                    "state": CellState.UNAVAILABLE,
                    "provider_phase": CapacityStatus.PROVIDER_ERROR,
                    "provider_failure_code": failure_code,
                    "provider_failure_occurred_at": observed_at if failure_code is not None else None,
                    "provider_failure_event_id": command.id if failure_code is not None else None,
                    "provider_failure_generation": cell.generation if failure_code is not None else None,
                    "provider_phase_started_at": _provider_phase_start(
                        cell,
                        CapacityStatus.PROVIDER_ERROR,
                        observed_at,
                    ),
                    "unavailable_reason": unavailable_reason,
                    # The reason is a code; this is what actually happened. It
                    # was dropped here, so a cell that died on a cell-agent
                    # config rejection showed the operator "provider rejected
                    # capacity request" -- about a provider that had never been
                    # called -- while the real pydantic error existed only in a
                    # log file on whichever machine ran the head.
                    "unavailable_detail": error[:2048] or None,
                    "availability": self._observe_capacity_belief(cell, outcome),
                    "updated_at": observed_at,
                }
            )
            await transaction.put_cell(updated)
            if failure_code is CapacityProviderFailureCode.QUEUE_ADMISSION_DENIED:
                await self._fail_autoscale_workload_for_queue_admission_denial(transaction, updated, error)
            transition = cell, updated
        if transition is not None:
            self._capacity_lifecycle.transition(*transition, reason=error)

    @staticmethod
    async def _has_alternative_pool(
        transaction: RepositorySession,
        workload: WorkloadRecord,
        *,
        denied_cell: CellRecord,
    ) -> bool:
        """Return whether another enabled pool could still host this workload.

        Deliberately coarse: it asks whether an *enabled pool of a different
        shape* exists that the workload's declared portfolio admits, not whether
        that pool has capacity right now. Capacity is what the next reconcile
        cycle decides, and predicting it here would duplicate the planner while
        being wrong more often.
        """

        spec = workload.spec
        candidates = candidate_pool_templates(spec.required_labels)
        if candidates:
            # A workload pinned to pools can only ever use those pools. Judging
            # it by any enabled pool of this shape and region would count
            # unrelated fleets as supply. Exclude every candidate that denied
            # this workload so denied pools cannot vouch for one another.
            denied = {denied_cell.pool_id}
            denied.update(
                cell.pool_id
                for cell in await transaction.list_cells_for_autoscale_request(workload.id)
                if cell.provider_failure_code is CapacityProviderFailureCode.QUEUE_ADMISSION_DENIED
            )
            enabled = {pool.id for pool in await transaction.list_capacity_pools() if pool.enabled}
            return any(pool_id in enabled and pool_id not in denied for pool_id in candidates)
        classes = {entry.value for entry in spec.capacity_class_portfolio}
        accelerators = set(spec.allowed_accelerator_types) | (
            {spec.accelerator_type} if spec.accelerator_type else set()
        )
        regions = set(spec.allowed_regions) | ({spec.region} if spec.region else set())
        for pool in await transaction.list_capacity_pools():
            if not pool.enabled or pool.id == denied_cell.pool_id:
                continue
            if str(pool.capacity_class) not in classes:
                continue
            if accelerators and pool.accelerator_type not in accelerators:
                continue
            if regions and pool.region not in regions:
                continue
            return True
        return False

    async def _fail_autoscale_workload_for_queue_admission_denial(
        self,
        transaction: RepositorySession,
        cell: CellRecord,
        diagnostic: str,
    ) -> None:
        """Block the pending workload that owns a denied queue request.

        **Terminal only when this pool was the last option.** A denial says this
        *shape in this zone* is unavailable; with a portfolio the workload has
        other admissible pools, and failing it outright discards a run the fleet
        could still place. When another pool remains, the workload stays PENDING
        with ``autoscale_block_reason`` set for explainability, and the ordinary
        reconcile cycle tries the next-best pool -- which is machinery that
        already ships, because ``cell_provides_pool_supply`` stops counting a
        failed cell as supply and ``CapacitySimulation.build`` already generates a
        candidate per admissible pool.
        """

        workload_id = cell.autoscale_request_workload_id
        if workload_id is None:
            return
        workload = await transaction.get_workload(workload_id)
        if workload is None or workload.state is not WorkloadState.PENDING:
            return
        failure_reason = safe_provider_diagnostic(diagnostic)
        if await self._has_alternative_pool(transaction, workload, denied_cell=cell):
            # Not terminal. Record why this attempt stalled so ``workload explain``
            # can say "one pool refused, others remain" rather than going quiet.
            await transaction.put_workload(
                workload.model_copy(
                    update={
                        "autoscale_block_reason": AutoscaleBlockReason.QUEUE_ADMISSION_DENIED,
                        "scheduling_reason": (f"{failure_reason}; other admissible pools remain, waiting to retry"),
                        "updated_at": utcnow(),
                    }
                )
            )
            return
        project = cell.provider_project
        zone = cell.zone
        accelerator_type = cell.accelerator_type
        target = (
            None
            if cell.provider != "gcp" or project is None or zone is None or accelerator_type is None
            else GcpQueueAdmissionDenialTarget(
                provider_project=project,
                zone=zone,
                accelerator_type=accelerator_type,
                capacity_class=cell.capacity_class,
                occurred_at=(
                    cell.provider_failure_occurred_at if cell.provider_failure_generation == cell.generation else None
                ),
                event_id=cell.provider_failure_event_id,
            )
        )
        await self.service._transition_workload(
            transaction,
            self._principal,
            workload,
            WorkloadState.FAILED,
            "workload.queue_admission_denied",
            updates={
                "failure_reason": failure_reason,
                "autoscale_block_reason": AutoscaleBlockReason.QUEUE_ADMISSION_DENIED,
                "queue_admission_denial": target,
                "scheduling_reason": None,
            },
        )

    async def _record_cell_agent_fence_conflict(
        self,
        *,
        cell_id: str,
        generation: int,
        diagnostic: str,
    ) -> None:
        """Mark a cell for replacement without treating an agent fence conflict as cloud failure."""

        transition: tuple[CellRecord, CellRecord] | None = None
        async with self.repository.transaction() as transaction:
            current = await transaction.get_cell(cell_id)
            if (
                current is None
                or current.generation != generation
                or current.state not in {CellState.REGISTERING, CellState.READY, CellState.UNAVAILABLE}
            ):
                return
            observed_at = utcnow()
            updated = current.model_copy(
                update={
                    "state": CellState.UNAVAILABLE,
                    "provider_phase": CapacityStatus.NEEDS_RECREATE,
                    "provider_phase_started_at": _provider_phase_start(
                        current,
                        CapacityStatus.NEEDS_RECREATE,
                        observed_at,
                    ),
                    "unavailable_reason": "agent_fence_conflict",
                    "unavailable_detail": diagnostic[:2048],
                    "availability": self._observe_capacity_belief(current, ProvisionOutcome.INTERRUPTED),
                    "updated_at": observed_at,
                }
            )
            await transaction.put_cell(updated)
            transition = current, updated
        if transition is not None:
            self._capacity_lifecycle.transition(*transition, reason=diagnostic)
        if transition is None:
            return
        _previous, unavailable = transition
        if unavailable.pool_id is not None:
            return
        try:
            recreated, replayed = await self.service.recreate_cell(
                self._principal,
                unavailable.id,
                expected_generation=unavailable.generation,
                idempotency_key=f"agent-fence-recreate:{unavailable.id}:g{unavailable.generation}",
            )
        except (ConflictError, NotFoundError) as exc:
            logger.info(
                "cell %s will wait for lease-safe recreation after an agent fence conflict: %s", unavailable.id, exc
            )
            return
        if not replayed:
            self._capacity_lifecycle.transition(
                unavailable,
                recreated,
                reason="cell-agent fence conflict requested a fresh capacity generation",
            )

    async def accrue_cell_billing_once(self) -> int:
        """Advance the fleet's cell billing clocks and return the cells charged.

        Delegates the arithmetic to the service so the accrual runs in one
        transaction with the lease read that decides which cells were idle -- a
        lease released between the two reads would otherwise make a busy cell
        look idle for the whole interval.
        """

        accrual = await self.service.accrue_cell_billing()
        if accrual.idle_seconds > 0:
            logger.debug(
                "Accrued %.0fs across %d cells, %.0fs idle (%.4f idle cost)",
                accrual.accrued_seconds,
                accrual.cells,
                accrual.idle_seconds,
                accrual.idle_cost,
            )
        return accrual.cells

    async def settle_orphaned_workloads_once(self) -> int:
        """Settle a live workload whose lease or cell row no longer exists.

        ``observe_workloads_once`` skips such a workload -- it cannot observe what it
        cannot resolve -- so without this it stays RUNNING forever: never polled,
        never settled, and counted against its project's quota, budget and fair share
        while holding no capacity at all. One hand-repaired or lost row therefore
        permanently shrinks a tenant's effective quota.

        Guarded by a grace period rather than firing on the first miss. Nothing should
        write a workload and its lease in separate transactions -- dispatch commits
        both together -- but settling a workload is irreversible, so requiring the
        condition to persist past the lease TTL costs one cycle and removes the whole
        class of "a race made this look orphaned".
        """

        grace = timedelta(seconds=max(self.settings.lease_ttl_s, 1))
        now = utcnow()
        async with self.repository.transaction() as transaction:
            live = [
                workload
                for workload in await transaction.list_workloads()
                if workload.state in _LIVE_WORKLOAD_STATES and now - workload.updated_at > grace
            ]
            lease_ids = {lease.id for lease in await transaction.list_leases()}
            cell_ids = {cell.id for cell in await transaction.list_cells()}
            leases_by_id = {lease.id: lease for lease in await transaction.list_leases()}

        orphans: list[tuple[WorkloadRecord, str]] = []
        for workload in live:
            members = self._workload_lease_ids(workload)
            if not members:
                continue
            missing_leases = sorted(set(members) - lease_ids)
            if missing_leases:
                orphans.append((workload, f"lease {missing_leases[0]!r} no longer exists"))
                continue
            missing_cells = sorted(
                {leases_by_id[lease_id].cell_id for lease_id in members} - cell_ids,
            )
            if missing_cells:
                orphans.append((workload, f"cell {missing_cells[0]!r} no longer exists"))

        settled = 0
        for workload, reason in orphans:
            try:
                await self.service.settle_terminal_workload(
                    self._principal,
                    workload.id,
                    WorkloadState.FAILED,
                    action="workload.orphaned",
                    failure_reason=(
                        f"workload is unobservable: {reason}; if the host is truly gone, settle it "
                        "with `eyvan workload-force-finalize <id> --reason '<what you checked>'`"
                    ),
                    expected_generation=workload.generation,
                    expected_state=workload.state,
                    settle_without_leases=True,
                )
            except (ConflictError, NotFoundError) as exc:
                # It recovered or moved on between the scan and the write; the next
                # cycle re-evaluates from fresh state. Logged rather than silently
                # skipped: a sweep that exists to clear stuck workloads must not be
                # able to fail quietly at the one step that clears them.
                logger.warning("orphan sweep could not settle workload %s: %s", workload.id, exc)
                continue
            settled += 1
        return settled

    async def settle_orphaned_leases_once(self) -> int:
        """Release leases that no live workload claims.

        ``settle_orphaned_workloads_once`` settles by *workload*, so a lease its
        workload stopped referencing -- because the workload finished, vanished,
        or retried onto different capacity -- is reachable from no path at all. It
        keeps a cell allocated and its cost unsettled forever: capacity the fleet
        pays for and no tenant is charged for, which is the one leak an operator
        cannot see in any existing view.

        The same grace period the workload sweep uses, for the same reason.
        Nothing should write a lease and its workload in separate transactions,
        but releasing a lease that a dispatch is mid-way through attaching would
        kill live work, and one cycle of patience removes the entire class of
        "a race made this look orphaned".
        """

        grace = timedelta(seconds=max(self.settings.lease_ttl_s, 1))
        now = utcnow()
        async with self.repository.transaction() as transaction:
            leases = await transaction.list_leases()
            workloads = {workload.id: workload for workload in await transaction.list_workloads()}
        candidates = [
            lease
            for lease in leases
            if lease.state in ACTIVE_LEASE_STATES
            and now - lease.updated_at > grace
            and self.service._lease_is_orphaned(lease, workloads.get(lease.workload_id))
        ]

        settled = 0
        for lease in candidates:
            owner = workloads.get(lease.workload_id)
            reason = "orphaned: workload no longer exists" if owner is None else "orphaned: workload released it"
            try:
                if await self.service.settle_orphaned_lease(
                    self._principal,
                    lease.id,
                    reason=reason,
                    expected_generation=lease.generation,
                ):
                    settled += 1
            except (ConflictError, NotFoundError) as exc:
                # It was re-attached or settled between the scan and the write;
                # the next cycle re-evaluates. Logged rather than swallowed: a
                # sweep whose whole purpose is reclaiming capacity must not be
                # able to fail quietly at the step that reclaims it.
                logger.warning("orphan sweep could not settle lease %s: %s", lease.id, exc)
        return settled

    async def quarantine_expired_leases_once(self) -> int:
        """Fence expired leases without killing work or reallocating ambiguous capacity."""

        now = utcnow()
        changed = 0
        async with self.repository.transaction() as transaction:
            candidate_ids = [
                lease.id
                for lease in await transaction.list_leases()
                if lease.state in {LeaseState.PENDING, LeaseState.ACTIVE, LeaseState.REVOKING}
                and lease.expires_at <= now
            ]
        for lease_id in candidate_ids:
            async with self.repository.transaction() as transaction:
                lease = await transaction.get_lease(lease_id)
                if lease is None:
                    continue
                if lease.state not in {LeaseState.PENDING, LeaseState.ACTIVE, LeaseState.REVOKING}:
                    continue
                if lease.expires_at > now:
                    continue
                await transaction.put_lease(
                    lease.model_copy(update={"state": LeaseState.QUARANTINED, "updated_at": now})
                )
                await transaction.append_event(
                    AuditEvent(
                        id=f"evt_{uuid.uuid4().hex}",
                        actor=self._principal.subject,
                        org_id=lease.org_id,
                        project_id=lease.project_id,
                        entity_type="lease",
                        entity_id=lease.id,
                        action="lease.quarantined",
                        previous_state=lease.state.value,
                        new_state=LeaseState.QUARANTINED.value,
                    )
                )
                changed += 1
        return changed

    async def settle_coordinator_workloads_once(self) -> int:
        """Derive a batch parent's terminal state from its children.

        A coordinator never holds a lease, so nothing else can ever settle it --
        it would sit PENDING forever while all 500 of its tasks succeeded, and
        every listing would report the array as still queued.

        The mapping is deliberately pessimistic: one failed task fails the
        array, because a partial array is not a result. A cancelled task without
        any failure reports CANCELLED, which is what the cascade produced.
        """

        settled = 0
        async with self.repository.transaction() as transaction:
            # Filtered on the indexed state column rather than scanning every
            # workload the fleet has ever run. This is a leader hot-path phase, and
            # the terminal records it used to read back are the fastest-growing table
            # in the system. Narrowing ``coordination_kind`` itself needs a typed
            # column and is deferred to the coordination-columns migration; PENDING
            # alone already removes the unbounded part.
            candidate_ids = [
                workload.id
                for workload in await transaction.list_workloads(state=WorkloadState.PENDING)
                # A service is deliberately excluded. Its children are
                # periodically all-terminal -- that is what a restart *is* -- so
                # this settler would declare it SUCCEEDED at the first moment
                # every replica happened to be between lives, and it would stop
                # forever. The failure would look like a service that quietly
                # went away, with a terminal state nobody asked for.
                if workload.coordination_kind is not None and workload.coordination_kind is not CoordinationKind.SERVICE
            ]
        for workload_id in candidate_ids:
            async with self.repository.transaction() as transaction:
                parent = await transaction.get_workload(workload_id)
                if (
                    parent is None
                    or parent.coordination_kind is None
                    or parent.coordination_kind is CoordinationKind.SERVICE
                    or parent.state is not WorkloadState.PENDING
                ):
                    continue
                children = await transaction.list_child_workloads(workload_id)
                if not children or any(child.state not in TERMINAL_WORKLOAD_STATES for child in children):
                    continue
                states = {child.state for child in children}
                if WorkloadState.FAILED in states:
                    target = WorkloadState.FAILED
                elif WorkloadState.CANCELLED in states:
                    target = WorkloadState.CANCELLED
                else:
                    target = WorkloadState.SUCCEEDED
                await transaction.put_workload(
                    parent.model_copy(
                        update={
                            "state": target,
                            "scheduling_reason": (
                                f"{parent.coordination_kind.value} of {len(children)} members settled"
                            ),
                            "updated_at": utcnow(),
                        }
                    )
                )
                await transaction.append_event(
                    AuditEvent(
                        id=f"evt_{uuid.uuid4().hex}",
                        actor=self._principal.subject,
                        org_id=parent.org_id,
                        project_id=parent.project_id,
                        entity_type="workload",
                        entity_id=parent.id,
                        action="workload.batch_settled",
                        previous_state=parent.state.value,
                        new_state=target.value,
                    )
                )
                settled += 1
        return settled

    async def export_metrics_once(self) -> int:
        """Publish fleet state, capacity inventory, and spend to Prometheus.

        Four of the seven metric families were declared, exported at ``/metrics``,
        and written by nothing -- so fleet state, capacity inventory and cost,
        the three things an operator actually dashboards, always read zero. A
        gauge that is always zero is worse than an absent one: it looks like an
        answer.

        Runs on its own slow interval rather than inside the scheduler, because
        gauges describe a level rather than a rate and re-reading the snapshot on
        every scheduling pass would buy nothing.
        """

        if self.metrics is None:
            return 0
        async with self.repository.transaction() as transaction:
            workloads = await transaction.list_workloads()
            cells = await transaction.list_cells()
            budgets = await transaction.list_budgets()

        by_state: Counter[str] = Counter(workload.state.value for workload in workloads)
        self.metrics.workloads.clear()
        for state, count in by_state.items():
            self.metrics.workloads.labels(state=state).set(count)

        accelerators: Counter[tuple[str, str, str]] = Counter()
        for cell in cells:
            if cell.state is CellState.DELETED:
                continue
            key = (cell.state.value, str(cell.capacity_class), cell.accelerator_type or "none")
            accelerators[key] += int(cell.capacity.gpu) + int(cell.capacity.tpu)
        self.metrics.capacity.clear()
        for (state, capacity_class, accelerator_type), total in accelerators.items():
            self.metrics.capacity.labels(
                state=state,
                capacity_class=capacity_class,
                accelerator_type=accelerator_type,
            ).set(total)

        self.metrics.project_spend.clear()
        for budget in budgets:
            self.metrics.project_spend.labels(
                org=budget.org_id,
                project=budget.project_id or "",
                kind="spent",
            ).set(float(budget.spent))

        # Progress series are written by the route as reports arrive, not from
        # this snapshot -- reading them would be one query per live workload. So
        # this phase prunes rather than rebuilds: a workload that has reached a
        # terminal state can report nothing further, and its series would
        # otherwise outlive it for the process's lifetime. ``workload`` is the
        # one unbounded label in the whole metric set, which is why the pruning
        # is not optional.
        live = {workload.id for workload in workloads if workload.state not in TERMINAL_WORKLOAD_STATES}
        for workload_id, rank in self._progress_series_labels():
            if workload_id not in live:
                self.metrics.workload_progress.remove(workload_id, rank)

        await self._export_invariant_violations()
        return len(by_state)

    async def _export_invariant_violations(self) -> None:
        """Evaluate the fleet's own durability invariants and publish the result.

        Exporting the checks makes violations alertable without an operator
        running `eyvan capacity invariants` by hand, including leases naming
        missing cells and stuck outbox commands.

        Published from this phase rather than a new one because that is what it
        is: a level, on a slow interval, next to the other gauges. Its own
        failure is contained like every phase's, and it must not be the reason a
        metrics export stops -- the invariant read is the newest and least
        trusted part of it.
        """

        if self.metrics is None:
            return
        try:
            violations = await check_fleet(self.repository, audit_retention_s=self.settings.audit_retention_s)
        except Exception:
            logger.exception("invariant check failed; its gauges keep their previous values")
            return
        counts = Counter(violation.invariant.value for violation in violations)
        for invariant in Invariant:
            self.metrics.invariant_violations.labels(invariant=invariant.value).set(counts.get(invariant.value, 0))
        if violations:
            # One line naming all of them, rather than one line each: these come
            # in correlated groups -- a stranded cell violates I1 and its lease
            # violates I9 -- and an operator wants the shape, not the list.
            logger.warning(
                "fleet invariant violations: %s",
                "; ".join(f"{name} x{count}" for name, count in sorted(counts.items())),
            )

    def _progress_series_labels(self) -> tuple[tuple[str, str], ...]:
        """Return the (workload, rank) label pairs currently held by the gauge."""

        if self.metrics is None:
            return ()
        return tuple(
            (values[0], values[1]) for values in list(self.metrics.workload_progress._metrics) if len(values) == 2
        )

    async def reconcile_services_once(self) -> int:
        """Bring every service to its declared replica count and revision.

        One action per pass per service, deliberately. The planner is pure and
        could hand back a whole batch, but performing it in one transaction
        would make a partially-applied roll indistinguishable from a completed
        one after a crash -- and a service reconciles again in a second anyway.

        Nothing here is a new workload state. A replica is an ordinary workload
        that succeeds or fails and stays terminal; the service simply asks for
        another one.
        """

        reconciled = 0
        async with self.repository.transaction() as transaction:
            services = [
                workload
                for workload in await transaction.list_workloads(state=WorkloadState.PENDING)
                if workload.coordination_kind is CoordinationKind.SERVICE
            ]
        for service in services:
            async with self.repository.transaction() as transaction:
                current = await transaction.get_workload(service.id)
                if current is None or current.coordination_kind is not CoordinationKind.SERVICE:
                    continue
                children = await transaction.list_child_workloads(current.id)
            try:
                policy = service_policy(current.spec.metadata)
            except InputError:
                logger.warning("service %s has a malformed policy; skipping", current.id, exc_info=True)
                continue
            if policy is None:
                continue
            plan = plan_service(policy, children, now=utcnow())
            if plan.cancel:
                await self._cancel_service_replica(current, plan.cancel[0])
                reconciled += 1
                continue
            if plan.submit > 0:
                # The slot this submit fills: the next ordinal this revision has
                # never minted, counted over every child it ever created.
                # Terminal replicas keep their ordinals, so a replacement mints
                # a fresh key, while two leaders that read the same children
                # mint the same one.
                slot = sum(1 for child in children if child_revision(child) == policy.revision)
                await self._start_service_replica(current, policy, slot=slot)
                reconciled += 1
        return reconciled

    async def _start_service_replica(self, service: WorkloadRecord, policy: ServicePolicy, *, slot: int) -> None:
        """Submit one replica of a service at its current revision."""

        spec = service.spec.model_copy(
            update={
                "metadata": {
                    **{
                        name: value
                        for name, value in service.spec.metadata.items()
                        if not name.startswith("eyvan.service/")
                    },
                    CHILD_REVISION: policy.revision,
                },
                # A replica is single-attempt: the service is the thing that
                # restarts, so a replica retrying inside itself would hide a
                # crash loop from the backoff that exists to bound it.
                "max_retries": 0,
                "max_failure_retries": 0,
            }
        )
        try:
            await self.service._submit_service_replica(
                parent=service,
                spec=spec,
                # Deterministic per revision and per slot, so a crash between
                # the submit and the commit does not start two replicas: two
                # leaders that read the same children mint the same key, and
                # the fence turns the second submit into a replay.
                idempotency_key=(
                    f"replica:{policy.revision}:{_replica_idempotency_suffix(service.id, policy.revision, slot)}"
                ),
            )
        except (ConflictError, AuthorizationError):
            logger.warning("could not start a replica for service %s", service.id, exc_info=True)

    async def _cancel_service_replica(self, service: WorkloadRecord, replica_id: str) -> None:
        """Retire one replica, leaving the service itself untouched."""

        try:
            await self.service.cancel_workload(self._principal, replica_id, idempotency_key=f"retire:{replica_id}")
        except (ConflictError, NotFoundError):
            # It finished on its own between the plan and the act, which is the
            # ordinary race for a service and not an error.
            logger.debug("replica %s of service %s was already terminal", replica_id, service.id)

    async def collect_artifacts_once(self) -> int:
        """Recover lifecycle claims, then collect only under explicit opt-in."""

        retention_s = self.settings.artifact_retention_s
        log_retention_s = self.settings.log_retention_s
        retention_by_kind = {ManagedArtifactKind.LOGS.value: float(log_retention_s) if log_retention_s > 0 else None}
        async with self.repository.transaction() as tx:
            projects = await tx.list_projects()
            artifact_scopes = await tx.list_managed_artifact_projects(limit=1001)
        if len(artifact_scopes) > 1000:
            raise RuntimeError("artifact lifecycle scope discovery exceeded 1000 entries; the sweep is inconclusive")
        artifact_scope_set = set(artifact_scopes)
        scopes = sorted({(project.org_id, project.id) for project in projects} | artifact_scope_set)
        deleted = reclaimed = 0
        for org_id, project_id in scopes:
            if (org_id, project_id) in artifact_scope_set:
                recovery = await self.service.reconcile_artifact_lifecycle_once(
                    org_id=org_id,
                    project_id=project_id,
                    recovery_grace_s=self.settings.artifact_recovery_grace_s,
                )
                if recovery["capped"] or recovery["inconclusive"]:
                    logger.warning(
                        "artifact lifecycle recovery was incomplete for %s/%s (capped=%d, inconclusive=%d)",
                        org_id,
                        project_id,
                        recovery["capped"],
                        recovery["inconclusive"],
                    )
            if retention_s <= 0:
                continue
            collection = await self.service.collect_artifacts_once(
                org_id=org_id,
                project_id=project_id,
                principal=self._principal,
                grace_period_s=float(retention_s),
                retention_by_kind=retention_by_kind,
                dry_run=not self.settings.artifact_destructive_collection_enabled,
            )
            if getattr(collection, "capped", False):
                logger.warning("artifact collection scan was capped for %s/%s", org_id, project_id)
            if not collection.dry_run:
                deleted += len(collection.collected_uris)
                reclaimed += collection.reclaimed_bytes
            elif collection.collected_uris:
                logger.info(
                    "artifact collection dry-run found %d candidate(s) in %s/%s",
                    len(collection.collected_uris),
                    org_id,
                    project_id,
                )
        if deleted:
            logger.info(
                "collected %d unreferenced managed artifact(s), reclaiming %d bytes",
                deleted,
                reclaimed,
            )
        return deleted

    async def mark_superseded_results_once(self) -> int:
        """Mark results a later attempt replaced, so collection can see them.

        Without this the superseded marker is never written, so a replaced
        result is never a collection candidate and the retention window above
        can never apply to it.
        """

        return await self.service.mark_superseded_results_once()

    async def expire_budget_grants_once(self) -> int:
        """Return headroom from federated budget grants whose expiry has passed.

        A grant's expiry is mandatory at construction, so leaving the sweep
        uncalled meant every grant lived forever: the granting plane never got
        its headroom back and a tenant could outspend its federated budget
        indefinitely.
        """

        expired = await self.service.expire_budget_grants_once()
        if expired:
            logger.info("expired %d federated budget grant(s)", expired)
        return expired

    async def prune_retained_state_once(self) -> dict[str, int]:
        """Delete state past its retention window and report what was removed.

        Runs leader-only and in bounded batches. A window of ``0`` disables
        pruning for that table, preserving the historical behaviour where
        nothing was ever deleted -- which also meant an idempotency key was
        honored forever, so a caller reusing one months later silently replayed
        the original response instead of acting.
        """

        now = utcnow()
        windows = (
            ("outbox", self.settings.outbox_retention_s, "prune_outbox"),
            ("audit_events", self.settings.audit_retention_s, "prune_events"),
            ("idempotency_keys", self.settings.idempotency_retention_s, "prune_idempotency"),
        )
        removed: dict[str, int] = {}
        failures: list[str] = []
        for name, retention_s, method in windows:
            if retention_s <= 0:
                continue
            cutoff = now - timedelta(seconds=retention_s)
            try:
                async with self.repository.transaction() as transaction:
                    prune = getattr(transaction, method)
                    count = await prune(before=cutoff, limit=self.settings.retention_batch_limit)
            except Exception as exc:  # one broken window must not mask the rest
                # These windows are independent tables with independent hazards.
                # Aborting the loop on the first failure is how a backend that
                # could not prune audit events also stopped pruning idempotency
                # keys -- so replays kept being honored forever, silently.
                failures.append(f"{name}: {exc}")
                logger.exception("retention pruning failed for %s", name)
                continue
            if count:
                removed[name] = count
        if removed:
            logger.info("retention pruned %s", ", ".join(f"{name}={count}" for name, count in sorted(removed.items())))
        if failures:
            raise RuntimeError("retention pruning incomplete: " + "; ".join(failures))
        return removed
