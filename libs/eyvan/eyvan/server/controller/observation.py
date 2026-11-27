# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reading live cell and workload state back into durable records."""

from __future__ import annotations

import asyncio
import hashlib
import json
import typing as tp
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta

from eyvan._internal.json_types import JsonObject
from eyvan.artifacts.managed import log_manifest_uri, log_uri
from eyvan.capacity import (
    DEFINITIVE_CAPACITY_LOSS_STATUSES,
    PENDING_CAPACITY_STATUSES,
    READY_CAPACITY_STATUSES,
    CapacityStatus,
)
from eyvan.capacity.drivers.base import CapacityProviderFailureCode, ExecutionWorkloadStatus, provision_outcome
from eyvan.capacity.planning import (
    cell_execution_runtime_is_pending,
)
from eyvan.domain import (
    ProvisionOutcome,
)
from eyvan.records import (
    AuditEvent,
    CellRecord,
    CellState,
    CoordinationKind,
    ExpansionAdmissionState,
    LeaseRecord,
    LeaseState,
    WorkloadAttemptTopology,
    WorkloadRecord,
    WorkloadState,
    WorkloadTopologyCell,
    utcnow,
)
from eyvan.secrets import redact_secret_material
from eyvan.server.controller._support import (
    _LIVE_WORKLOAD_STATES,
    _PREEMPTION_OWNED_TERMINAL_WORKLOAD_STATUSES,
    _RECOVERABLE_WORKLOAD_STATUSES,
    StaleOutboxCommand,
    _provider_phase_start,
    logger,
)
from eyvan.server.persistence.base import RepositorySession
from eyvan.server.service.errors import ConflictError, LogCursorRejectedError, NotFoundError
from eyvan.server.service.expansion import bounded_release
from eyvan.server.service.state import ACTIVE_LEASE_STATES, TERMINAL_WORKLOAD_STATES
from eyvan.worker.agent.client import (
    CellAgentBusyError,
    CellAgentCapacityProviderError,
    CellAgentInvalidConfigurationError,
    CellAgentStaleMutationError,
    CellAgentUnavailableError,
)
from eyvan.worker.agent.models import (
    GangWorkloadMutationRequest,
    MutationModel,
    WorkloadMutationRequest,
)
from eyvan.worker.runtime.log_records import (
    CURSOR_ATTEMPT_CHANGED,
    CURSOR_KIND_WORKLOAD_LOG,
    CURSOR_MODE_ARCHIVE,
    CURSOR_MODE_LIVE,
    CURSOR_RESET_REQUIRED,
    CURSOR_UNKNOWN_RANK,
    LOG_ARCHIVE_LEGACY_RANK_SCAN_LIMIT,
    LOG_ARCHIVE_MANIFEST_KEY,
    LOG_ARCHIVE_MANIFEST_VERSION,
    LOG_CHUNK_HEADER_KEY,
    ArchiveGap,
    CursorContext,
    CursorFresh,
    CursorIncompatible,
    CursorInvalid,
    CursorPosition,
    CursorResume,
    LogPage,
    LogRecord,
    decode_log_cursor,
    encode_archive_offset,
    encode_log_cursor,
    encode_unplaced_archive_position,
    is_archive_position,
    log_cursor_v2_emission_enabled,
    log_page_from_json,
    merge_pages,
    resolve_archive_position,
)
from eyvan.worker.runtime.log_records import (
    join_log_cursor as _join_log_cursor,
)
from eyvan.worker.runtime.node_health import (
    HOST_FAULT_AUDIT_ENTITY,
    HOST_FAULT_HOST_QUARANTINED,
    HOST_FAULT_REASON,
    HOST_FAULT_REPLACEMENT_DECLINED,
    HostFault,
    HostReplacementBudget,
    format_host_fault_marker,
    host_faults_by_rank,
    host_replacement_scope,
    match_host_fault,
    parse_host_fault_marker,
)

#: How many contiguous ranks a pre-manifest archive read probes. New archives
#: carry their exact rank set and do not use this compatibility ceiling.
_LOG_ARCHIVE_MANIFEST_SCHEMA_V2 = 2
#: Rank fates a schema-2 manifest can record. Kept as plain strings rather than
#: importing the writer's enum so the reader stays decoupled from settlement and
#: keeps working against a manifest written by a newer control plane.
_ARCHIVE_STATUS_QUIET = "quiet"
_ARCHIVE_STATUS_TRUNCATED = "truncated"
_ARCHIVE_STATUS_READABLE = frozenset({"archived", _ARCHIVE_STATUS_TRUNCATED})
_ARCHIVE_STATUS_UNREACHED = frozenset({"unreachable", "failed"})

_ARCHIVED_RANK_SCAN_LIMIT = LOG_ARCHIVE_LEGACY_RANK_SCAN_LIMIT
_SECRET_OUTPUT_SUPPRESSED = "output suppressed for secret-bearing execution"

#: Remote outcomes whose application archives every rank's output before the
#: lease is released. Only these leave the observation pass's critical path.
_SETTLING_WORKLOAD_STATUSES = frozenset(
    {ExecutionWorkloadStatus.SUCCEEDED, ExecutionWorkloadStatus.FAILED, ExecutionWorkloadStatus.STOPPED}
)
#: Answer to a ``live_only`` read of an attempt whose capacity is gone. Said
#: rather than answered from the archive, so an archive writer cannot be handed
#: its own output back and store a second copy of it.
_LIVE_READ_HAS_NO_CELL = "no cell holds this attempt's journal any more"

#: Newest node-health audit events one replacement-budget decision reads. Small
#: on purpose: the entity type carries nothing but these decisions, so a fleet
#: that fills it has a systemic outage rather than ordinary traffic -- and the
#: caller refuses to replace anything when the scan does fill, because a spend it
#: could not establish is not a budget it may spend.
_HOST_FAULT_EVENT_SCAN_LIMIT = 512

#: What a page says about the rank identity it is serving.
_TOPOLOGY_FROZEN = "frozen"
_TOPOLOGY_UNKNOWN = "unknown"
_TOPOLOGY_CHANGED = "changed"


def _live_cursor_context(
    workload: WorkloadRecord,
    topology: WorkloadAttemptTopology | None,
    *,
    ranks: tuple[int, ...] | None = None,
) -> CursorContext:
    """Describe the read a presented cursor has to address.

    Opacity is not addressing: a structurally perfect cursor minted for another
    workload, attempt, or attempt topology still names positions in a journal
    this read is not looking at. The frozen topology digest is what makes the
    last of those checkable at all.
    """

    return CursorContext(
        kind=CURSOR_KIND_WORKLOAD_LOG,
        mode=CURSOR_MODE_LIVE,
        attempt=workload.attempt,
        workload=workload.id,
        topology="" if topology is None else topology.digest,
        ranks=ranks,
    )


def _cell_leaf_positions(next_cursor: str, *, width: int) -> tuple[dict[int, str], bool]:
    """Split one cell's returned cursor into per-local-rank leaf positions.

    A single-process cell has exactly one position, so its whole opaque cursor
    *is* that leaf and is never parsed -- which is also the only way a backend
    cursor that happens to look like an envelope survives the round trip. A
    multi-process cell must return the per-local-rank envelope its slice executor
    writes; anything else cannot be attributed to a host, and guessing would give
    one journal another journal's position.

    The cell's own attempt field is deliberately not constrained. It echoes
    whatever the controller sent it, which is zero on a first read, and refusing
    the position for that would strand every multi-host cell on its second poll.

    The boolean says whether the value was understood. ``False`` means the caller
    must retain the positions it already had rather than advance.
    """

    if not next_cursor:
        return {}, True
    if width == 1:
        return {0: next_cursor}, True
    outcome = decode_log_cursor(
        next_cursor,
        expected=CursorContext(
            kind=CURSOR_KIND_WORKLOAD_LOG,
            mode=CURSOR_MODE_LIVE,
            ranks=tuple(range(width)),
        ),
    )
    if not isinstance(outcome, CursorResume) or not outcome.positions:
        return {}, False
    return outcome.cursors, True


def _leading_cell_leaf_positions(value: str) -> tuple[dict[int, str], bool]:
    """Split the first cell's cursor with no frozen width to check it against.

    The first cell's global rank offset is zero by construction, so its inner
    ranks *are* global ranks whatever its width turns out to be -- that much
    needs no witness, and it is the whole of a single-cell attempt. Only the
    envelope a multi-host cell emits is unpacked; anything else is one journal's
    opaque position, exactly as :func:`_cell_leaf_positions` treats a
    single-process cell, so a backend cursor that happens to look like one of
    ours survives the round trip.
    """

    if not value:
        return {}, True
    outcome = decode_log_cursor(value, expected=None)
    if isinstance(outcome, CursorResume) and outcome.version != 1 and outcome.positions:
        return outcome.cursors, True
    return {0: value}, True


def _log_page_from_document(payload: JsonObject, *, rank: int, runtime_process_count: int = 1) -> LogPage:
    """Rebuild one cell page and translate its local ranks into global ranks.

    ``rank`` is the cell's first global rank. A multi-host runtime already
    attributes records to its inner host, so replacing every record with the
    outer cell index collapses all hosts in that cell into one identity. Ordered
    cells plus each durable ``runtime_process_count`` instead define a stable
    flattened rank space: ``global = cell_offset + local``. Truncated inner ranks
    use the same mapping.

    Decoding is delegated rather than repeated. This function used to build
    ``LogRecord`` itself and omitted ``timestamp`` entirely, so every record read
    over the wire arrived without one and chronological interleaving was lost.

    A response containing an out-of-range local rank is partial, not merely
    malformed decoration: advancing its nested cursor would permanently skip the
    dropped line. The whole cell is marked truncated and its position is retained
    so a corrected retry can replay it.
    """

    if runtime_process_count < 1:
        raise ValueError("structured log runtime process count must be positive")
    page = log_page_from_json(payload)
    records: list[LogRecord] = []
    malformed_rank = False
    for record in page.records:
        local_rank = record.rank
        if (
            isinstance(local_rank, bool)
            or not isinstance(local_rank, int)
            or not 0 <= local_rank < runtime_process_count
        ):
            malformed_rank = True
            continue
        records.append(replace(record, rank=rank + local_rank))

    truncated: set[int] = set()
    for local_rank in page.truncated_ranks:
        if (
            isinstance(local_rank, bool)
            or not isinstance(local_rank, int)
            or not 0 <= local_rank < runtime_process_count
        ):
            malformed_rank = True
            continue
        truncated.add(rank + local_rank)
    expected = set(range(rank, rank + runtime_process_count))
    if page.unavailable and not truncated:
        truncated.update(expected)
    detail = page.detail
    if malformed_rank:
        truncated.update(expected)
        invalid_detail = "cell returned a log rank outside its durable runtime process width"
        detail = f"{detail}; {invalid_detail}" if detail else invalid_detail
    return LogPage(
        records=tuple(records),
        next_cursor="" if malformed_rank or page.unavailable else page.next_cursor,
        unavailable=page.unavailable,
        detail=detail,
        truncated_ranks=tuple(sorted(truncated)),
    )


def _log_page_document(
    page: LogPage,
    *,
    attempt: int,
    attempt_changed: bool = False,
    cursors: tp.Mapping[int, str] | None = None,
    next_cursor: str | None = None,
    topology_state: str = _TOPOLOGY_UNKNOWN,
    topology_digest: str | None = None,
    expected_ranks: int | None = None,
    missing_ranks: tp.Sequence[int] = (),
) -> JsonObject:
    """Project a merged page onto the wire, carrying the attempt it belongs to.

    ``next_cursor`` encodes every rank's position and the attempt they came
    from. The attempt has to travel with it: without it an attempt boundary is
    undetectable, and a follower resumes a retry from its predecessor's
    position. Pass it pre-encoded when the caller knows which rank space the
    positions live in; ``cursors`` is the per-cell shortcut for the paths that
    have no frozen rank space to encode against.

    ``topology`` and ``topology_state`` are additive and describe what the ranks
    on this page mean. ``frozen`` says every rank names the process it named at
    dispatch; ``unknown`` says the attempt predates the frozen record and its
    rank identity is not proven; ``changed`` says the fleet no longer matches the
    frozen map, so the page is explicitly incomplete rather than renumbered.

    ``missing_ranks`` names every expected rank the returned cursor carries no
    position for. Those ranks resume from the start of their journal, and saying
    so is the difference between a partial resume and one presented as whole.
    """

    document: JsonObject = {
        "attempt": attempt,
        "attempt_changed": attempt_changed,
        "records": [dict(record.as_json()) for record in page.records],
        "next_cursor": _join_log_cursor(attempt, cursors or {}) if next_cursor is None else next_cursor,
        "unavailable": page.unavailable,
        "detail": page.detail,
        "truncated_ranks": list(page.truncated_ranks),
        "topology_state": topology_state,
        "topology": topology_digest,
        "expected_ranks": expected_ranks,
        "missing_ranks": list(missing_ranks),
        "completeness_unknown": topology_state == _TOPOLOGY_UNKNOWN,
    }
    return document


def _archive_cursor_context(workload: WorkloadRecord) -> CursorContext:
    """Describe the archive read a cursor is minted for, or presented against.

    Archive positions address immutable objects, not backend journals, so the
    mode is part of the cursor's identity: a live position replayed against the
    archive addresses nothing, and an archive offset handed to journald is not a
    journald cursor. The attempt is carried for the same reason the live read
    carries it -- a retry archives separately, and its objects are different
    objects.

    The topology digest is deliberately not constrained. An archive's rank space
    is the manifest's, which survives the cells the frozen topology describes.
    """

    return CursorContext(
        kind=CURSOR_KIND_WORKLOAD_LOG,
        mode=CURSOR_MODE_ARCHIVE,
        attempt=workload.attempt,
        workload=workload.id,
    )


def _is_archive_cursor(after: str) -> bool:
    """Return whether a presented cursor was minted by an archive read.

    Either half is proof. A bound envelope says so in its ``mode``; the envelope
    emitted by default carries no mode, so the tagged positions say it instead.
    Deciding this wrong in either direction is the whole failure: a live cursor
    read as archive positions places nothing, and archive offsets read as cell
    positions are handed to a journal that never issued them.
    """

    if not after:
        return False
    outcome = decode_log_cursor(after, expected=None)
    if not isinstance(outcome, CursorResume):
        return False
    if outcome.mode == CURSOR_MODE_ARCHIVE:
        return True
    values = [position.cursor for position in outcome.positions if position.cursor]
    return bool(values) and all(is_archive_position(value) for value in values)


def _archived_entries(payload: bytes) -> list[object]:
    """Decode one archived object into the JSON values it holds, line by line."""

    decoded: list[object] = []
    for line in payload.decode(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            decoded.append(json.loads(line))
        except ValueError:
            continue
    return decoded


def _archived_rank_records(entries: tp.Sequence[object], *, rank: int) -> tuple[LogRecord, ...]:
    """Rebuild one rank's archived lines, in the order they were written.

    File order is the archive's own order and the only thing an offset can name,
    so it is preserved here rather than sorted: chronological interleaving is the
    merged page's job, and doing it before the offsets are computed would make a
    resumed read skip lines.
    """

    page = log_page_from_json({"records": [entry for entry in entries if isinstance(entry, dict)]})
    return tuple(replace(record, rank=rank) for record in page.records)


@dataclass(frozen=True, slots=True)
class _ArchiveReplay:
    """One attempt's archive as it stands, before a follower's window is applied.

    Attributes:
        ranks: Every readable rank's lines, in file order.
        truncated: Ranks whose output is known to be incomplete or absent.
        detail: The sentence describing this archive, already composed.
        unavailable: Whether the writer recorded that it could read nothing.
    """

    ranks: dict[int, tuple[LogRecord, ...]] = field(default_factory=dict)
    truncated: frozenset[int] = frozenset()
    detail: str = ""
    unavailable: bool = False


def _chunk_header(payload: bytes) -> JsonObject:
    """Return one chunk's header, or an empty mapping if it carries none."""

    for entry in _archived_entries(payload):
        if (
            isinstance(entry, list)
            and len(entry) == 2
            and entry[0] == LOG_CHUNK_HEADER_KEY
            and isinstance(entry[1], dict)
        ):
            return entry[1]
    return {}


def _chunk_truncated_ranks(header: JsonObject) -> set[int]:
    """Return the ranks one chunk recorded as incomplete when it was written."""

    raw = header.get("truncated_ranks")
    return {
        rank
        for rank in (raw if isinstance(raw, list) else [])
        if isinstance(rank, int) and not isinstance(rank, bool) and rank >= 0
    }


def _joined_archive_replays(earlier: _ArchiveReplay, later: _ArchiveReplay) -> _ArchiveReplay:
    """Join archives in file order, retaining only one copy of identified overlap.

    A terminal writer on another replica can read before an in-flight chunk is
    published. Nonempty per-rank cursors identify that overlap; equality of the
    complete record guards against cursor collisions with different content.
    Missing or repeated cursors cannot establish identity and are never removed.
    In particular messages and timestamps alone are not record identities.
    """

    ranks = {rank: tuple(lines) for rank, lines in earlier.ranks.items()}
    for rank, lines in later.ranks.items():
        previous = ranks.get(rank, ())
        earlier_counts: dict[str, int] = {}
        later_counts: dict[str, int] = {}
        for record in previous:
            earlier_counts[record.cursor] = earlier_counts.get(record.cursor, 0) + 1
        for record in lines:
            later_counts[record.cursor] = later_counts.get(record.cursor, 0) + 1
        identified = {
            record
            for record in previous
            if record.cursor and earlier_counts[record.cursor] == 1 and later_counts.get(record.cursor) == 1
        }
        ranks[rank] = previous + tuple(record for record in lines if record not in identified)
    details = [detail for detail in (earlier.detail, later.detail) if detail]
    return _ArchiveReplay(
        ranks=ranks,
        truncated=earlier.truncated | later.truncated,
        detail="; ".join(details),
        # Unavailable only when neither half could produce a line. One half
        # having nothing to say is ordinary: an attempt that was reclaimed has
        # no terminal archive, and one that predates chunking has no chunks.
        unavailable=earlier.unavailable and later.unavailable and not any(ranks.values()),
    )


def _dag_release_order(node: WorkloadRecord) -> tuple[str, str]:
    """Return the stable order dependent nodes are released in.

    Read off the nodes themselves so a bounded release resumes where it stopped.
    Names are unique inside one DAG submission; the id breaks the tie for a node
    whose name is not, so two replicas never disagree about which node is next.
    """

    return (node.spec.name, node.id)


def _topology_mismatch(frozen: WorkloadTopologyCell, current: CellRecord, observed_endpoints: tuple[str, ...]) -> str:
    """Return why a live cell no longer matches its frozen entry, or ``""``.

    The comparison is the point of freezing at all. ``runtime_process_count`` is
    rewritten by ordinary capacity observation, so a cell that reports a
    different width mid-attempt would move every following cell's global ranks if
    the offsets were recomputed. Here it makes the cell's own range explicitly
    truncated and leaves every other rank exactly where it was.
    """

    if current.generation != frozen.generation:
        return (
            f"cell {frozen.cell_id} is at generation {current.generation}, "
            f"not the generation {frozen.generation} this attempt was dispatched against"
        )
    if current.runtime_process_count != frozen.runtime_process_count:
        return (
            f"cell {frozen.cell_id} now reports {current.runtime_process_count} runtime processes, "
            f"not the {frozen.runtime_process_count} frozen for this attempt"
        )
    if frozen.worker_endpoints and observed_endpoints != frozen.worker_endpoints:
        return (
            f"cell {frozen.cell_id} reports {len(observed_endpoints)} worker endpoints in a different order "
            f"or count than the {len(frozen.worker_endpoints)} frozen for this attempt"
        )
    return ""


def _heartbeat_only(current: CellRecord, updated: CellRecord, now: datetime, *, interval_s: float) -> bool:
    """Return whether an observation may skip persisting a heartbeat-only change.

    True only when the two records differ in nothing but ``last_heartbeat_at``
    and ``updated_at``, a heartbeat was already persisted, and that heartbeat is
    younger than ``interval_s``. A zero interval never skips, which is the
    behaviour before the interval existed.
    """

    if interval_s <= 0 or current.last_heartbeat_at is None or updated.last_heartbeat_at is None:
        return False
    if now - current.last_heartbeat_at >= timedelta(seconds=interval_s):
        return False
    stamped = current.model_copy(
        update={"last_heartbeat_at": updated.last_heartbeat_at, "updated_at": updated.updated_at}
    )
    return stamped == updated


class _ControllerObservation:
    """Reading live cell and workload state back into durable records.

    A concern of :class:`~eyvan.server.controller.EyvanController`; not
    instantiated on its own.
    """

    async def observe_workloads_once(self) -> int:
        async with self.repository.transaction() as transaction:
            workloads = [
                workload
                for workload in await transaction.list_workloads()
                if workload.state in _LIVE_WORKLOAD_STATES
                and (workload.remote_id or workload.state is WorkloadState.LEASED)
            ]
            leases = {lease.id: lease for lease in await transaction.list_leases()}
            cells = {cell.id: cell for cell in await transaction.list_cells()}
            dispatch_commands = await transaction.list_outbox_for_aggregates(
                "workload",
                {workload.id: workload.generation for workload in workloads},
                topic="workload.dispatch",
            )
        observed = 0
        started: list[asyncio.Task[None]] = []
        for workload in workloads:
            if (workload.id, workload.generation) in self._settling_workloads:
                # Its terminal settlement is still archiving in the background;
                # observing it again would only race that settlement.
                continue
            lease_ids = self._workload_lease_ids(workload)
            members = tuple(leases.get(lease_id) for lease_id in lease_ids)
            concrete_leases = tuple(lease for lease in members if lease is not None)
            if not members or len(concrete_leases) != len(members):
                continue
            member_cells = tuple(cells.get(lease.cell_id) for lease in concrete_leases)
            concrete_cells = tuple(cell for cell in member_cells if cell is not None)
            if len(concrete_cells) != len(member_cells):
                continue
            pairs = tuple(zip(concrete_leases, concrete_cells, strict=True))
            lease, cell = pairs[0]
            is_group = len(pairs) > 1

            lost_pair = next(
                (
                    member
                    for member in pairs
                    if member[1].state in {CellState.DELETING, CellState.DELETED}
                    or (
                        member[1].state is CellState.UNAVAILABLE
                        and member[1].provider_phase in DEFINITIVE_CAPACITY_LOSS_STATUSES
                    )
                    or (member[1].state is CellState.REGISTERING and not cell_execution_runtime_is_pending(member[1]))
                ),
                None,
            )
            if lost_pair is not None:
                _lost_lease, lost_cell = lost_pair
                capacity_lost = lost_cell.state in {CellState.DELETING, CellState.DELETED} or (
                    lost_cell.state is CellState.UNAVAILABLE
                    and lost_cell.provider_phase in DEFINITIVE_CAPACITY_LOSS_STATUSES
                )
                reason = (
                    f"execution cell capacity is {lost_cell.unavailable_reason}"
                    if lost_cell.state is CellState.UNAVAILABLE
                    else f"execution cell is {lost_cell.state.value}"
                    if capacity_lost
                    else "execution cell capacity was recreated"
                )
                try:
                    if workload.state is WorkloadState.LEASED and workload.remote_id is None:
                        await self.service.requeue_unstarted_workload(
                            self._principal,
                            workload.id,
                            scheduling_reason=reason,
                            expected_generation=workload.generation,
                            expected_lease_id=lease.id,
                            expected_cell_generation=lost_cell.generation,
                            expected_cell_state=lost_cell.state,
                            expected_unavailable_reason=lost_cell.unavailable_reason,
                            expected_cell_id=lost_cell.id,
                        )
                    else:
                        if is_group:
                            await self._stop_terminal_workload_peers(
                                workload,
                                lease,
                                cell,
                                ExecutionWorkloadStatus.LOST,
                            )
                        await self._recover_lost_workload(workload, lease, reason, lost_cell=lost_cell)
                except Exception as exc:
                    logger.info("workload %s loss reconciliation became stale: %s", workload.id, exc)
                observed += 1
                continue
            changed_pair = next(
                (member for member in pairs if member[0].cell_generation != member[1].generation),
                None,
            )
            if changed_pair is not None and workload.remote_id is None:
                _changed_lease, changed_cell = changed_pair
                try:
                    await self.service.requeue_unstarted_workload(
                        self._principal,
                        workload.id,
                        scheduling_reason="cell generation changed before dispatch",
                        expected_generation=workload.generation,
                        expected_lease_id=lease.id,
                        expected_cell_generation=changed_cell.generation,
                        expected_cell_state=changed_cell.state,
                        expected_unavailable_reason=changed_cell.unavailable_reason,
                        expected_cell_id=changed_cell.id,
                    )
                except Exception as exc:
                    logger.info("workload %s undispatched recovery became stale: %s", workload.id, exc)
                observed += 1
                continue
            if workload.remote_id is None:
                releasable_undispatched_leases = all(
                    member.state in {LeaseState.PENDING, LeaseState.QUARANTINED} for member in concrete_leases
                ) and any(member.state is LeaseState.QUARANTINED for member in concrete_leases)
                if releasable_undispatched_leases and not self._has_live_dispatch_command(
                    workload,
                    lease_ids,
                    dispatch_commands,
                ):
                    try:
                        await self.service.requeue_unstarted_workload(
                            self._principal,
                            workload.id,
                            scheduling_reason="undispatched lease expired after its dispatch command retired",
                            expected_generation=workload.generation,
                            expected_lease_id=lease.id,
                            expected_cell_generation=cell.generation,
                            expected_cell_state=cell.state,
                            expected_unavailable_reason=cell.unavailable_reason,
                            expected_cell_id=cell.id,
                        )
                    except Exception as exc:
                        logger.info("workload %s expired undispatched recovery became stale: %s", workload.id, exc)
                    observed += 1
                continue
            try:
                if is_group:
                    coordinator_lease, coordinator_cell = self._ordered_gang_member_pairs(pairs)[0]
                    remote = await asyncio.to_thread(
                        self._agent_factory(coordinator_cell).gang_workload,
                        coordinator_lease.cell_id,
                        workload.id,
                        workload.remote_id,
                    )
                else:
                    remote = await asyncio.to_thread(self._agent_factory(cell).workload, cell.id, workload.remote_id)
            except Exception as exc:
                # Redacted even for workloads without secret mounts: the
                # exception text itself can carry capability tokens or URLs
                # that were part of the failed observation round trip.
                detail = _SECRET_OUTPUT_SUPPRESSED if workload.spec.secret_mounts else redact_secret_material(str(exc))
                logger.warning("workload %s observation failed: %s", workload.id, detail)
                continue
            apply = self._apply_observed_workload(workload, lease, cell, pairs, remote.status, remote.message)
            if remote.status in _SETTLING_WORKLOAD_STATUSES:
                key = (workload.id, workload.generation)
                task = asyncio.ensure_future(apply)
                self._settling_workloads[key] = task
                task.add_done_callback(lambda _task, key=key: self._settling_workloads.pop(key, None))
                started.append(task)
            else:
                await apply
            observed += 1
        if started:
            # Terminal settlement archives every rank's output before the lease
            # is released, and against a remote store that costs seconds per
            # rank -- 197 s on a 16-host slice, during which no other workload
            # was observed or scheduled. Wait at most one scheduler interval:
            # a fast settlement completes inside this pass exactly as before,
            # and a slow one keeps running while the cycle moves on. Its key
            # stays in ``_settling_workloads`` until it finishes, so no later
            # pass settles the same attempt twice; every write it makes is
            # fenced on generation and lease, so it stays safe against phases
            # that run meanwhile.
            # A non-positive interval keeps the fixed cadence everywhere else,
            # so it keeps the old fully synchronous settlement here too.
            interval = self.settings.scheduler_interval_s
            await asyncio.wait(started, timeout=interval if interval > 0 else None)
        return observed

    async def _apply_observed_workload(
        self,
        workload: WorkloadRecord,
        lease: LeaseRecord,
        cell: CellRecord,
        pairs: tuple[tuple[LeaseRecord, CellRecord], ...],
        status: ExecutionWorkloadStatus,
        message: str | None,
    ) -> None:
        """Apply one observation, containing its failure the way the pass always has."""

        try:
            if len(pairs) > 1:
                await self._recover_group_member_leases(workload, pairs, status)
            await self._apply_workload_observation(workload, lease, cell, status, message, observed_members=pairs)
        except Exception as exc:
            logger.info("workload %s observation became stale: %s", workload.id, exc)

    async def _recover_group_member_leases(
        self,
        workload: WorkloadRecord,
        members: tuple[tuple[LeaseRecord, CellRecord], ...],
        status: ExecutionWorkloadStatus,
    ) -> None:
        """Renew quarantined non-leader group leases after a healthy group observation."""

        if status not in _RECOVERABLE_WORKLOAD_STATUSES:
            return
        for lease, cell in members[1:]:
            if lease.state is not LeaseState.QUARANTINED or lease.revocation_reason == "preemption":
                continue
            await self._recover_observed_lease(workload, lease, cell)

    async def _recover_observed_lease(self, workload: WorkloadRecord, lease: LeaseRecord, cell: CellRecord) -> bool:
        """Recover and renew only the still-owned execution, in one transaction."""

        async with self.repository.transaction() as transaction:
            current_workload = await transaction.get_workload(workload.id)
            if (
                current_workload is None
                or current_workload.state not in {WorkloadState.LEASED, WorkloadState.DISPATCHING, WorkloadState.RUNNING}
                or current_workload.state is not workload.state
                or current_workload.generation != workload.generation
                or current_workload.attempt != workload.attempt
                or current_workload.remote_id != workload.remote_id
                or current_workload.preemption_id != workload.preemption_id
                or self._workload_lease_ids(current_workload) != self._workload_lease_ids(workload)
            ):
                return False
            current = await transaction.get_lease(lease.id)
            if (
                current is None
                or current.id not in self._workload_lease_ids(current_workload)
                or current.workload_id != workload.id
                or current.cell_id != cell.id
                or current.generation != lease.generation
                or current.fencing_token != lease.fencing_token
                or current.cell_generation != lease.cell_generation
                or current.state is not LeaseState.QUARANTINED
                or current.revocation_reason == "preemption"
                or current.revocation_reason != lease.revocation_reason
            ):
                return False
            current_cell = await transaction.get_cell(cell.id)
            if (
                current_cell is None
                or current_cell.generation != cell.generation
                or current_cell.generation != current.cell_generation
                or current_cell.state not in {CellState.READY, CellState.DRAINING}
            ):
                return False
            now = utcnow()
            await transaction.put_lease(
                current.model_copy(
                    update={
                        "state": LeaseState.ACTIVE,
                        "run_started_at": current.run_started_at or now,
                        "expires_at": now + timedelta(seconds=self.settings.lease_ttl_s),
                        "updated_at": now,
                    }
                )
            )
            await transaction.append_event(
                AuditEvent(
                    id=f"evt_{uuid.uuid4().hex}",
                    actor=self._principal.subject,
                    org_id=current.org_id,
                    project_id=current.project_id,
                    entity_type="lease",
                    entity_id=current.id,
                    action="lease.recovered",
                    previous_state=current.state.value,
                    new_state=LeaseState.ACTIVE.value,
                )
            )
            return True

    async def _apply_workload_observation(
        self,
        workload: WorkloadRecord,
        lease: LeaseRecord,
        cell: CellRecord,
        status: ExecutionWorkloadStatus,
        message: str | None,
        *,
        observed_members: tuple[tuple[LeaseRecord, CellRecord], ...] | None = None,
    ) -> None:
        if workload.spec.secret_mounts:
            message = _SECRET_OUTPUT_SUPPRESSED
        if status is ExecutionWorkloadStatus.INDETERMINATE:
            await self._escalate_unreadable_execution(workload, cell, message)
            return
        self._indeterminate_since.pop((workload.id, workload.generation), None)
        preemption_owned = lease.revocation_reason == "preemption" and lease.state in {
            LeaseState.REVOKING,
            LeaseState.QUARANTINED,
        }
        if preemption_owned and status in _PREEMPTION_OWNED_TERMINAL_WORKLOAD_STATUSES:
            return
        if lease.state is LeaseState.QUARANTINED and not preemption_owned and status in _RECOVERABLE_WORKLOAD_STATUSES:
            if not await self._recover_observed_lease(workload, lease, cell):
                return
        if status is ExecutionWorkloadStatus.RUNNING and workload.state is WorkloadState.DISPATCHING:
            await self.service.transition_workload(
                self._principal,
                workload.id,
                WorkloadState.RUNNING,
                expected_generation=workload.generation,
                expected_lease_id=lease.id,
                expected_state=workload.state,
            )
        elif status is ExecutionWorkloadStatus.SUCCEEDED:
            current = workload
            if workload.state is WorkloadState.DISPATCHING:
                current = await self.service.transition_workload(
                    self._principal,
                    workload.id,
                    WorkloadState.RUNNING,
                    expected_generation=workload.generation,
                    expected_lease_id=lease.id,
                    expected_state=workload.state,
                )
            await self._release_terminal(current, lease, WorkloadState.SUCCEEDED, "workload.succeeded")
        elif status in {ExecutionWorkloadStatus.FAILED, ExecutionWorkloadStatus.STOPPED}:
            if status is ExecutionWorkloadStatus.FAILED:
                await self._stop_terminal_workload_peers(workload, lease, cell, status)
                if not preemption_owned and await self._quarantine_faulted_host(
                    workload, message, observed_members or ((lease, cell),)
                ):
                    # The machine, not the workload, is what failed. Its cell is
                    # now a definitive capacity loss, so the loss path in
                    # ``observe_workloads_once`` requeues this attempt against
                    # the capacity budget on the next pass -- rather than this
                    # one spending a user-code retry to land on the same host.
                    return
            if workload.state is WorkloadState.CANCELLING and not preemption_owned:
                await self._release_terminal(workload, lease, WorkloadState.CANCELLED, "workload.cancelled")
            elif (
                status is ExecutionWorkloadStatus.FAILED
                and not preemption_owned
                and workload.failure_count < workload.spec.max_failure_retries
            ):
                if workload.lease_id != lease.id:
                    raise StaleOutboxCommand(f"lease {lease.id!r} no longer owns workload {workload.id!r}")
                await self.service.settle_failed_workload(
                    self._principal,
                    workload.id,
                    checkpoint_uri=workload.checkpoint_uri,
                    scheduling_reason=(
                        f"execution failed; retrying user-code failure "
                        f"({workload.failure_count + 1} of {workload.spec.max_failure_retries})"
                    ),
                    expected_generation=workload.generation,
                    expected_lease_id=lease.id,
                    expected_state=workload.state,
                )
            elif not preemption_owned:
                # Persist the executor's diagnostic before releasing the cell;
                # its logs may otherwise be the only account of the failure.
                await self._release_terminal(
                    workload,
                    lease,
                    WorkloadState.FAILED,
                    "workload.failed",
                    failure_reason=message or "execution backend reported failure without a diagnostic",
                )
        elif status in {ExecutionWorkloadStatus.LOST, ExecutionWorkloadStatus.NOT_FOUND}:
            await self._stop_terminal_workload_peers(workload, lease, cell, status)
            await self._recover_lost_workload(
                workload,
                lease,
                message or f"execution backend reported {status.value}",
            )

    async def _stop_terminal_workload_peers(
        self,
        workload: WorkloadRecord,
        lease: LeaseRecord,
        cell: CellRecord,
        status: ExecutionWorkloadStatus,
    ) -> None:
        """Fence every reachable rank before settling a failed slice workload.

        A native multi-host execution can report a failed or lost rank while
        peers are still waiting in a JAX collective. The original lease fence
        remains authoritative for this attempt, so best-effort fan-out stop
        happens before the workload is released or requeued. A missing host is
        expected for ``LOST`` and must not prevent recovery of the lease.
        """

        if workload.remote_id is None:
            return
        if len(self._workload_lease_ids(workload)) > 1:
            try:
                members = await self._load_workload_members(workload)
                ordered = self._ordered_gang_member_pairs(members)
                coordinator_lease, coordinator_cell = ordered[0]
                request = GangWorkloadMutationRequest(
                    members=self._gang_members(
                        ordered,
                        idempotency_prefix=f"terminal-stop:{workload.id}:g{workload.generation}",
                    )
                )
                await asyncio.to_thread(
                    self._agent_factory(coordinator_cell).stop_gang,
                    coordinator_lease.cell_id,
                    workload.id,
                    workload.remote_id,
                    request,
                )
            except Exception as exc:
                logger.warning(
                    "workload %s reported %s and group peer stop was incomplete: %s",
                    workload.id,
                    status.value,
                    _SECRET_OUTPUT_SUPPRESSED if workload.spec.secret_mounts else exc,
                )
            return
        request = WorkloadMutationRequest(
            mutation=MutationModel(
                idempotency_key=f"terminal-stop:{workload.id}:g{workload.generation}",
                generation=workload.generation,
                fencing_token=lease.fencing_token,
            )
        )
        try:
            await asyncio.to_thread(self._agent_factory(cell).stop, cell.id, workload.remote_id, request)
        except Exception as exc:
            logger.warning(
                "workload %s reported %s and peer stop was incomplete: %s",
                workload.id,
                status.value,
                _SECRET_OUTPUT_SUPPRESSED if workload.spec.secret_mounts else exc,
            )

    async def _quarantine_faulted_host(
        self,
        workload: WorkloadRecord,
        message: str | None,
        members: tuple[tuple[LeaseRecord, CellRecord], ...],
    ) -> bool:
        """Replace the host behind a failure that names an unrecoverable machine fault.

        Eyvan's failure vocabulary describes attempts, not machines, so a TPU
        that never initializes spends the workload's user-code retry budget and
        then fails it -- while the dead host keeps its lease, keeps its cell
        undeletable, and blocks ``eyvan down``. When the attempt's own output
        carries one of :data:`HOST_FAULT_SIGNATURES`, the machine is what failed:
        its cell is marked a definitive capacity loss so the existing loss path
        recovers the workload against its *capacity* budget, and the pool
        replaces the host instead of handing it back.

        Attribution is the hard half and is deliberately strict. A gang's
        observation is the concatenation of every rank's, so a fault is
        attributed only through the rank that reported it and the frozen attempt
        topology that says which cell owns that rank. An attempt with no frozen
        topology, or a message with no per-rank structure, leaves the owning host
        *unknown* -- and unknown is not "all of them". Quarantining a gang's
        healthy peers because one rank's TPU died would destroy good accelerators
        on exactly the evidence that names a single bad one.

        Returns whether at least one cell was quarantined.
        """

        budget = self._host_replacement_budget
        if not budget.enabled or not message:
            return False
        faults = await self._attribute_host_faults(workload, members, message)
        leases_by_cell = {cell.id: lease for lease, cell in members}
        quarantined = False
        for cell_id, fault in sorted(faults.items()):
            if await self._quarantine_faulted_cell(workload, cell_id, fault, budget, leases_by_cell[cell_id]):
                quarantined = True
        return quarantined

    async def _attribute_host_faults(
        self,
        workload: WorkloadRecord,
        members: tuple[tuple[LeaseRecord, CellRecord], ...],
        message: str,
    ) -> dict[str, HostFault]:
        """Return the host fault each of this workload's cells is evidenced to have.

        A single-cell workload owns every rank in its own observation, so the
        whole diagnostic is that cell's. Anything wider has to name the rank, and
        the rank has to resolve through the topology frozen at dispatch -- the
        record that exists because recomputing a rank map from today's cells
        silently renumbers hosts.
        """

        cell_ids = {cell.id for _lease, cell in members}
        if len(members) == 1:
            fault = match_host_fault(message)
            return {} if fault is None else {members[0][1].id: fault}
        faults_by_rank = host_faults_by_rank(message)
        if not faults_by_rank:
            return {}
        async with self.repository.transaction() as transaction:
            topology = await transaction.get_workload_topology(workload.id, workload.attempt)
        if topology is None:
            logger.warning(
                "workload %s reported a host fault but attempt %s has no frozen rank space, so no host owns it",
                workload.id,
                workload.attempt,
            )
            return {}
        attributed: dict[str, HostFault] = {}
        for rank, fault in sorted(faults_by_rank.items()):
            owner = topology.cell_for_rank(rank)
            if owner is None or owner.cell_id not in cell_ids:
                logger.warning(
                    "workload %s reported a host fault at rank %s, which no leased cell of this attempt owns",
                    workload.id,
                    rank,
                )
                continue
            attributed.setdefault(owner.cell_id, fault)
        return attributed

    async def _quarantine_faulted_cell(
        self,
        workload: WorkloadRecord,
        cell_id: str,
        fault: HostFault,
        budget: HostReplacementBudget,
        observed_lease: LeaseRecord,
    ) -> bool:
        """Mark one cell as a lost host, inside its pool's replacement budget."""

        now = utcnow()
        transition: tuple[CellRecord, CellRecord] | None = None
        refusal: tuple[CellRecord, str, str] | None = None
        async with self.repository.transaction() as transaction:
            # Attribution preceded this transaction (and an agent round trip).
            # Lock the execution first, then its lease and cell: a historical
            # lease must never diagnose the machine that replaced or reused it.
            current_workload = await transaction.get_workload(workload.id)
            if (
                current_workload is None
                or current_workload.state not in _LIVE_WORKLOAD_STATES
                or current_workload.state is not workload.state
                or current_workload.generation != workload.generation
                or current_workload.attempt != workload.attempt
                or current_workload.remote_id != workload.remote_id
                or self._workload_lease_ids(current_workload) != self._workload_lease_ids(workload)
            ):
                return False
            owning_lease = await transaction.get_lease(observed_lease.id)
            if (
                owning_lease is None
                or owning_lease.id not in self._workload_lease_ids(current_workload)
                or owning_lease.workload_id != workload.id
                or owning_lease.cell_id != cell_id
                or owning_lease.generation != observed_lease.generation
                or owning_lease.fencing_token != observed_lease.fencing_token
                or owning_lease.cell_generation != observed_lease.cell_generation
                or owning_lease.state not in {LeaseState.PENDING, LeaseState.ACTIVE, LeaseState.QUARANTINED}
                or owning_lease.revocation_reason == "preemption"
            ):
                return False
            current = await transaction.get_cell(cell_id)
            if current is None or current.generation != owning_lease.cell_generation:
                return False
            existing = parse_host_fault_marker(current.unavailable_reason, current.unavailable_detail)
            if (
                existing is not None
                and existing.workload_id == workload.id
                and existing.generation == workload.generation
            ):
                # Already promoted for this execution. Rewriting it would charge
                # the pool's budget a second time for one broken machine.
                return True
            if current.state not in {CellState.REGISTERING, CellState.READY, CellState.UNAVAILABLE}:
                # Draining, deleting, or gone: the cell is already leaving and the
                # lifecycle that owns it does not need a second opinion.
                return False
            scope = host_replacement_scope(current.pool_id, current.provider, current.zone or current.region)
            spent, complete = await self._host_replacements_spent(transaction, scope, now, budget)
            # Declared before the branches so the audit write below cannot read an
            # unbound name: a refusal leaves the cell exactly as it was, and only
            # the promoting branch produces a new record.
            updated: CellRecord | None = None
            if not complete:
                refusal = (
                    current,
                    "unknown",
                    f"more than {_HOST_FAULT_EVENT_SCAN_LIMIT} node-health events in the last "
                    f"{budget.window_s:.0f}s, so this scope's spend could not be established",
                )
            elif spent >= budget.max_replacements:
                refusal = (
                    current,
                    "exhausted",
                    f"{scope} has already replaced {spent} host(s) in the last {budget.window_s:.0f}s, "
                    f"which is its budget of {budget.max_replacements}",
                )
            else:
                updated = current.model_copy(
                    update={
                        "state": CellState.UNAVAILABLE,
                        "provider_phase": CapacityStatus.NEEDS_RECREATE,
                        "provider_phase_started_at": _provider_phase_start(
                            current,
                            CapacityStatus.NEEDS_RECREATE,
                            now,
                        ),
                        "unavailable_reason": HOST_FAULT_REASON,
                        "unavailable_detail": format_host_fault_marker(
                            fault,
                            workload_id=workload.id,
                            generation=workload.generation,
                            observed_at=now,
                        ),
                        "availability": self._observe_capacity_belief(current, ProvisionOutcome.HOST_FAULT),
                        "updated_at": now,
                    }
                )
                await transaction.put_cell(updated)
                transition = current, updated
            await transaction.append_event(
                AuditEvent(
                    id=f"evt_{uuid.uuid4().hex}",
                    actor=self._principal.subject,
                    entity_type=HOST_FAULT_AUDIT_ENTITY,
                    entity_id=cell_id,
                    action=(HOST_FAULT_REPLACEMENT_DECLINED if refusal is not None else HOST_FAULT_HOST_QUARANTINED),
                    previous_state=current.state.value,
                    new_state=(updated or current).state.value,
                    payload={
                        "code": fault.code.value,
                        "diagnosis": fault.diagnosis,
                        "evidence": fault.evidence,
                        "workload_id": workload.id,
                        "generation": workload.generation,
                        "scope": scope,
                        "budget": budget.max_replacements,
                        "window_s": budget.window_s,
                        "spent": spent,
                        **({"declined": refusal[1], "detail": refusal[2]} if refusal is not None else {}),
                    },
                )
            )
        if refusal is not None:
            stalled, _outcome, detail = refusal
            # Reported, not retried. Replacing hosts one after another through a
            # zone that is handing out broken machines costs a full provision and
            # a destroy per attempt and never converges, so the budget stops and
            # says so; the workload then follows its ordinary failure path.
            logger.error(
                "cell %s reported host fault %s (%s) and was not replaced: %s",
                stalled.id,
                fault.code.value,
                fault.evidence,
                detail,
            )
            return False
        if transition is not None:
            self._capacity_lifecycle.transition(
                *transition,
                reason=f"host fault {fault.code.value}: {fault.diagnosis}",
            )
        return True

    @staticmethod
    async def _host_replacements_spent(
        transaction: RepositorySession,
        scope: str,
        now: datetime,
        budget: HostReplacementBudget,
    ) -> tuple[int, bool]:
        """Return how many hosts one scope has replaced in the window, and whether that is known.

        Counted from the audit ledger rather than from the surviving cell
        records, because the records do not survive: an unpooled cell is
        recreated *in place*, and ``recreate_cell`` clears
        ``unavailable_reason`` -- so a count taken from cells would reset to zero
        on the very replacement it is supposed to be bounding, and a zone full of
        bad machines would loop forever one recreate at a time.

        The second element is whether the window was seen whole. A scan that
        fills its limit has not established anything, and an unestablished spend
        is not an available budget: the caller refuses rather than assuming the
        cheaper answer.

        Single-writer by construction -- only the elected leader observes
        workloads, and it does so one workload at a time -- so the read and the
        append that follows it do not race another replacement decision.
        """

        events = await transaction.list_events(
            entity_type=HOST_FAULT_AUDIT_ENTITY,
            occurred_at_or_after=now - timedelta(seconds=budget.window_s),
            newest_first=True,
            limit=_HOST_FAULT_EVENT_SCAN_LIMIT,
        )
        spent = sum(
            1 for event in events if event.action == HOST_FAULT_HOST_QUARANTINED and event.payload.get("scope") == scope
        )
        return spent, len(events) < _HOST_FAULT_EVENT_SCAN_LIMIT

    async def _escalate_unreadable_execution(
        self,
        workload: WorkloadRecord,
        cell: CellRecord,
        message: str | None,
    ) -> None:
        """Replace the host of an execution nobody has been able to read.

        An INDETERMINATE observation says the runtime could not be read, not
        that the workload finished, so the controller keeps polling. Polling
        forever would leave an unreadable workload holding capacity indefinitely.

        Settling the workload alone would be unsafe -- it might still be running,
        and releasing the lease would let a second workload onto the same
        accelerator. The cell is the only thing that can be reclaimed with
        certainty, so drain it: replacing the host destroys whatever survived,
        and the existing cell-loss path then recovers the workload.
        """

        grace = timedelta(seconds=self.settings.indeterminate_execution_grace_s)
        key = (workload.id, workload.generation)
        first_seen = self._indeterminate_since.setdefault(key, utcnow())
        if utcnow() - first_seen < grace:
            return
        if cell.state is not CellState.READY:
            # Already registering or on its way out; the cell-loss path owns it.
            return
        self._indeterminate_since.pop(key, None)
        logger.warning(
            "draining cell %s: execution of workload %s has been unreadable for %ss (%s)",
            cell.id,
            workload.id,
            self.settings.indeterminate_execution_grace_s,
            message or "no diagnostic",
        )
        try:
            await self.drain_cell(
                self._principal,
                cell.id,
                idempotency_key=f"unreadable-execution:{workload.id}:{workload.generation}",
                expected_workload=workload,
                expected_cell_generation=cell.generation,
            )
        except ConflictError:
            # A terminal observation or replacement won while this RPC waited.
            # Its current execution must not inherit the old attempt's drain.
            return

    async def _recover_lost_workload(
        self,
        workload: WorkloadRecord,
        lease: LeaseRecord,
        reason: str,
        *,
        lost_cell: CellRecord | None = None,
    ) -> None:
        if workload.lease_id != lease.id:
            return
        preemption_owned = lease.revocation_reason == "preemption" and lease.state in {
            LeaseState.REVOKING,
            LeaseState.QUARANTINED,
        }
        if workload.state is WorkloadState.CANCELLING and not preemption_owned:
            await self.service.settle_terminal_workload(
                self._principal,
                workload.id,
                WorkloadState.CANCELLED,
                action="workload.cancelled_after_loss",
                expected_generation=workload.generation,
                expected_lease_id=lease.id,
                expected_state=workload.state,
                expected_cell_generation=None if lost_cell is None else lost_cell.generation,
                expected_cell_state=None if lost_cell is None else lost_cell.state,
                expected_unavailable_reason=None if lost_cell is None else lost_cell.unavailable_reason,
                expected_cell_id=None if lost_cell is None else lost_cell.id,
            )
            return
        # The cloud reclaiming capacity, not Eyvan preempting a lower-priority
        # job for it. They are charged to different budgets because they mean
        # opposite things about whether running again can work: a spot
        # reclamation is the expected ending of a spot job, while a capacity
        # loss may be a pool that has nothing left to give.
        provider_preempted = (
            lost_cell is not None and lost_cell.provider_failure_code is CapacityProviderFailureCode.PREEMPTED
        )
        spent, budget = (
            (workload.preemption_count, workload.spec.max_preemption_retries)
            if provider_preempted
            else (workload.capacity_loss_count, workload.spec.max_retries)
        )
        if (workload.checkpoint_uri or preemption_owned) and spent < budget:
            await self.service.settle_preempted_workload(
                self._principal,
                workload.id,
                checkpoint_uri=workload.checkpoint_uri,
                scheduling_reason=(
                    f"cell lost; resuming from {workload.checkpoint_uri} ({spent + 1} of {budget})"
                    if workload.checkpoint_uri
                    else f"cell lost during preemption; restarting without a checkpoint ({spent + 1} of {budget})"
                ),
                action="workload.recovery_queued",
                expected_generation=workload.generation,
                expected_lease_id=lease.id,
                expected_state=workload.state,
                require_preemption_owner=preemption_owned,
                provider_preempted=provider_preempted,
                expected_cell_generation=None if lost_cell is None else lost_cell.generation,
                expected_cell_state=None if lost_cell is None else lost_cell.state,
                expected_unavailable_reason=None if lost_cell is None else lost_cell.unavailable_reason,
                expected_cell_id=None if lost_cell is None else lost_cell.id,
            )
            return
        if lost_cell is not None and spent < budget:
            await self.service.settle_preempted_workload(
                self._principal,
                workload.id,
                scheduling_reason=(
                    f"{reason}; retrying provider preemption ({spent + 1} of {budget})"
                    if provider_preempted
                    else f"{reason}; retrying capacity loss ({spent + 1} of {budget})"
                ),
                action="workload.capacity_recovery_queued",
                expected_generation=workload.generation,
                expected_lease_id=lease.id,
                expected_state=workload.state,
                provider_preempted=provider_preempted,
                expected_cell_generation=lost_cell.generation,
                expected_cell_state=lost_cell.state,
                expected_unavailable_reason=lost_cell.unavailable_reason,
                expected_cell_id=lost_cell.id,
            )
            return
        await self.service.settle_terminal_workload(
            self._principal,
            workload.id,
            WorkloadState.FAILED,
            action="workload.lost",
            failure_reason=reason,
            expected_generation=workload.generation,
            expected_lease_id=lease.id,
            expected_state=workload.state,
            expected_cell_generation=None if lost_cell is None else lost_cell.generation,
            expected_cell_state=None if lost_cell is None else lost_cell.state,
            expected_unavailable_reason=None if lost_cell is None else lost_cell.unavailable_reason,
            expected_cell_id=None if lost_cell is None else lost_cell.id,
        )

    async def observe_cells_once(self) -> int:
        """Advance capacity state while serializing with lifecycle transitions."""

        return await self._observe_cells_once()

    async def _observe_cells_once(self) -> int:
        """Advance provider capacity state and refresh the schedulable inventory."""

        now = utcnow()
        async with self.repository.transaction() as transaction:
            disabled_pool_ids = {pool.id for pool in await transaction.list_capacity_pools() if not pool.enabled}
            candidates = [
                cell
                for cell in await transaction.list_cells()
                if cell.state in {CellState.REGISTERING, CellState.READY, CellState.UNAVAILABLE}
                and (cell.availability.cooldown_until is None or cell.availability.cooldown_until <= now)
                and cell.pool_id not in disabled_pool_ids
            ]
            # Only REGISTERING cells consult the outbox, and only for their own
            # current generation. Query exactly those aggregates instead of
            # scanning the whole table: the previous limit=10_000 scan was
            # O(cells x outbox) per cycle and, past that many rows, could miss
            # an active command entirely and observe a cell mid-provision.
            registering = {cell.id: cell.generation for cell in candidates if cell.state is CellState.REGISTERING}
            active_capacity_applies = {
                (command.aggregate_id, command.generation)
                for command in await transaction.list_outbox_for_aggregates("cell", registering, topic="capacity.apply")
                if self._is_active_capacity_apply(command)
            }
            cells = [
                cell
                for cell in candidates
                if not (
                    cell.state is CellState.REGISTERING
                    and (cell.provider_phase is None or (cell.id, cell.generation) in active_capacity_applies)
                )
            ]
        # An agent that refused the last asks is asked less often; skipping it
        # writes nothing, exactly like the failed ask it replaces.
        self._agent_backoff.retain({(cell.id, cell.generation) for cell in cells})
        cells = [cell for cell in cells if self._agent_backoff.ready(cell, now)]
        semaphore = asyncio.Semaphore(max(1, self.settings.cell_reconcile_concurrency))

        async def reconcile(cell: CellRecord) -> bool:
            async with semaphore:
                return await self._observe_cell_once(cell, now)

        def finished(task: asyncio.Task[bool], key: tuple[str, int]) -> None:
            self._observing_cells.pop(key, None)
            if not task.cancelled() and task.exception() is not None:
                logger.warning("cell %s observation failed", key[0], exc_info=task.exception())

        started: list[asyncio.Task[bool]] = []
        for cell in cells:
            key = (cell.id, cell.generation)
            if key in self._observing_cells:
                # Its previous observation is still waiting on the agent; a
                # second one would only queue behind the same capacity lock.
                continue
            task = asyncio.ensure_future(reconcile(cell))
            self._observing_cells[key] = task
            task.add_done_callback(lambda done, key=key: finished(done, key))
            started.append(task)
        if not started:
            return 0
        # A worker that never answers costs the agent's whole readiness timeout
        # on every observation. Do not let it delay scheduling and observation
        # of unrelated workloads on every cycle.
        # Wait at most one scheduler interval; a slower observation keeps
        # running and lands on a later cycle, and until then the cell keeps the
        # state it was last observed in -- the same staleness the cell
        # reconcile interval already accepts. A non-positive interval keeps the
        # old fully synchronous pass.
        interval = self.settings.scheduler_interval_s
        done, _pending = await asyncio.wait(started, timeout=interval if interval > 0 else None)
        errors = [task.exception() for task in done if not task.cancelled() and task.exception() is not None]
        if errors:
            # Same as before: an observation that raised inside the pass fails
            # the phase, so the phases that require a fresh view are skipped.
            raise tp.cast(BaseException, errors[0])
        return sum(1 for task in done if not task.cancelled() and task.result())

    def _note_agent_answered(self, cell: CellRecord) -> None:
        """Clear an unreachable-agent streak, saying so once if there was one."""

        if self._agent_backoff.succeeded(cell):
            logger.info("cell %s agent is reachable again", cell.id)

    async def _observe_cell_once(self, cell: CellRecord, now: datetime) -> bool:
        """Fence one agent observation against its fleet and cell lifecycle changes."""

        async with self._capacity_operation(cell):
            current = await self._current_observable_cell(cell, now)
            if current is None:
                return False
            return await self._observe_cell_once_locked(current, now)

    async def _current_observable_cell(self, cell: CellRecord, now: datetime) -> CellRecord | None:
        """Return the current cell only when its original observation remains safe."""

        async with self.repository.transaction() as transaction:
            current = await transaction.get_cell(cell.id)
            if current is None or current.generation != cell.generation or current.state is not cell.state:
                return None
            if current.state not in {CellState.REGISTERING, CellState.READY, CellState.UNAVAILABLE}:
                return None
            if current.state is CellState.REGISTERING:
                # One aggregate, one generation: an indexed lookup rather than a
                # full-table scan repeated per cell.
                if any(
                    self._is_active_capacity_apply(command)
                    for command in await transaction.list_outbox_for_aggregates(
                        "cell", {current.id: current.generation}, topic="capacity.apply"
                    )
                ):
                    return None
                if current.provider_phase is None:
                    return None
            if current.unavailable_reason == HOST_FAULT_REASON:
                # Quarantined for a matched host fault. The provider's opinion of
                # this machine is not the evidence in question -- a queued
                # resource stays ACTIVE while its accelerator refuses to
                # initialize -- so letting an ordinary observation report it
                # HEALTHY would clear the quarantine and hand the same broken
                # host back to the scheduler. Only recreation, which replaces the
                # machine and takes a new generation, returns it to service.
                return None
            if current.availability.cooldown_until is not None and current.availability.cooldown_until > now:
                return None
            if current.pool_id is not None:
                pool = await transaction.get_capacity_pool(current.pool_id)
                if pool is not None and not pool.enabled:
                    return None
            return current

    async def _observe_cell_once_locked(self, cell: CellRecord, now: datetime) -> bool:
        """Reconcile and persist one cell without blocking unrelated cells."""

        agent_responded = False
        materialized_assets: tuple[str, ...] = cell.materialized_assets
        provider_detail: str | None = None
        provider_failure_code = cell.provider_failure_code
        unavailable_reason: str | None = (
            cell.unavailable_reason if provider_failure_code is not None else "agent_unreachable"
        )
        observed_tpu_vms: int | None = None
        # Runtime width is durable. A provider error or a currently unreachable
        # host must not collapse it and shift every later cell's log ranks.
        runtime_process_count = cell.runtime_process_count
        try:
            mutation = MutationModel(
                idempotency_key=f"reconcile:{cell.id}:{cell.generation}",
                generation=cell.generation,
                fencing_token=f"cell:{cell.id}:{cell.generation}",
            )
            observation = await asyncio.to_thread(
                self._agent_factory(cell).reconcile,
                cell.id,
                WorkloadMutationRequest(mutation=mutation),
            )
        except CellAgentBusyError:
            logger.debug("cell %s capacity reconciliation is already active", cell.id)
            return False
        except CellAgentStaleMutationError as exc:
            logger.debug("cell %s reconciliation was superseded by a newer agent fence", cell.id)
            # Keep the agent's own words. It distinguishes three different causes
            # -- an older generation, a fencing token that does not own the
            # current one, and a tombstoned cell -- and this path used to collapse
            # all three into one sentence that says only "a fence was rejected".
            # That is the difference between an operator reading the cause and
            # inferring it from a GCS document by hand.
            await self._record_cell_agent_fence_conflict(
                cell_id=cell.id,
                generation=cell.generation,
                diagnostic=(
                    f"cell-agent reconciliation rejected the current fence: {exc}"
                    if str(exc)
                    else "cell-agent reconciliation rejected the current fence"
                ),
            )
            return False
        except CellAgentUnavailableError as exc:
            # Nothing was learned, so nothing is written. Recording this as
            # PROVIDER_ERROR would restart the provider phase clock and feed the
            # zone's belief a failure the provider never caused; see
            # CellAgentUnavailableError for the incident that motivated it.
            #
            # Leaving the record untouched is also what bounds the blast radius:
            # the cell keeps its real phase and its real clock, so a permanently
            # dead agent still retires the cell on `queue_timeout_s` /
            # `provisioning_timeout_s` exactly as it would have. Capacity cannot
            # leak because of this branch -- it can only survive a blip.
            if self._agent_backoff.failed(cell, now):
                logger.warning("cell %s could not be observed: %s", cell.id, exc)
            else:
                # Said once per streak at warning level; repeating it every pass
                # grew the local log by tens of megabytes for one dead agent.
                logger.debug("cell %s still could not be observed: %s", cell.id, exc)
            return False
        except CellAgentCapacityProviderError as exc:
            self._note_agent_answered(cell)
            provider_detail = redact_secret_material(str(exc))
            provider_failure_code = exc.code
            unavailable_reason = exc.code.value
            state = CellState.UNAVAILABLE
            remote_state = CapacityStatus.PROVIDER_ERROR
            availability = self._observe_capacity_belief(cell, ProvisionOutcome.PROVIDER_ERROR)
        except Exception as exc:
            provider_detail = redact_secret_material(str(exc)) or type(exc).__name__
            state = CellState.UNAVAILABLE
            remote_state = CapacityStatus.PROVIDER_ERROR
            availability = self._observe_capacity_belief(cell, ProvisionOutcome.PROVIDER_ERROR)
        else:
            agent_responded = True
            self._note_agent_answered(cell)
            provider_detail = observation.detail or None
            remote_state = observation.state
            observed_tpu_vms = observation.observed_tpu_vms
            reported_runtime_process_count = observation.runtime_process_count or observation.observed_tpu_vms
            if reported_runtime_process_count is not None:
                runtime_process_count = reported_runtime_process_count
            reported_failure_code = observation.failure_code
            materialized_assets = _reported_assets(observation.metadata)
            if observation.generation != cell.generation and remote_state not in DEFINITIVE_CAPACITY_LOSS_STATUSES:
                logger.warning(
                    "cell %s reported generation %s while generation %s is desired",
                    cell.id,
                    observation.generation,
                    cell.generation,
                )
                return False
            if remote_state in READY_CAPACITY_STATUSES:
                state = CellState.READY
                unavailable_reason = None
                provider_failure_code = None
                first_ready = cell.billing_started_at is None and not cell.availability.provision_latencies_s
                availability = (
                    self._observe_capacity_belief(
                        cell,
                        ProvisionOutcome.SUCCESS,
                        latency_s=max(0.0, (now - cell.created_at).total_seconds()) if first_ready else None,
                    )
                    if cell.state is not CellState.READY
                    else cell.availability
                )
            elif remote_state in PENDING_CAPACITY_STATUSES:
                state = CellState.REGISTERING
                unavailable_reason = None
                provider_failure_code = None
                availability = (
                    self._observe_capacity_belief(cell, ProvisionOutcome.INTERRUPTED)
                    if cell.state is CellState.READY
                    else cell.availability
                )
            else:
                # Every non-ready, non-pending observation used to be recorded as
                # a STOCKOUT belief unless the status was explicitly a quota
                # halt. So an ordinary spot reclamation -- the common ending on
                # this fleet -- taught the placement engine that the pool had no
                # capacity, and the next placement routed away from a zone that
                # was in fact fine. A reported reason now decides.
                state = CellState.UNAVAILABLE
                provider_failure_code = reported_failure_code or provider_failure_code
                unavailable_reason = (
                    reported_failure_code.value
                    if reported_failure_code is not None
                    # Unchanged when nothing new was reported: a code carried
                    # over from a previous observation keeps the prose that came
                    # with it, rather than being flattened into its own name.
                    else (cell.unavailable_reason if provider_failure_code is not None else remote_state.value)
                )
                # A provider draining capacity without saying why has not
                # reported an outcome yet: the state that follows it does. Do not
                # score an intermediate STOPPING state as PROVIDER_ERROR and
                # impose cooldown before a possible Spot preemption is classified.
                availability = (
                    cell.availability
                    if reported_failure_code is None and remote_state is CapacityStatus.DRAINING
                    else self._observe_capacity_belief(
                        cell,
                        provision_outcome(reported_failure_code, remote_state),
                    )
                )
        transition: tuple[CellRecord, CellRecord] | None = None
        async with self.repository.transaction() as transaction:
            current = await transaction.get_cell(cell.id)
            if current is None or current.generation != cell.generation or current.state is not cell.state:
                return False
            if current.unavailable_reason in {"agent_fence_conflict", HOST_FAULT_REASON}:
                # A quarantine written while the agent RPC above was in flight.
                # Both markers keep the state and generation this observation
                # snapshotted, so the guard above cannot see them, and the write
                # below would erase them -- re-admitting a cell the controller
                # has slated for recreation or host replacement. Only a new
                # generation returns such a cell to observation.
                return False
            phase_started_at = _provider_phase_start(current, remote_state, now)
            new_failure = provider_failure_code is not None and (
                current.provider_failure_code is not provider_failure_code
                or (
                    current.provider_failure_generation is not None
                    and current.provider_failure_generation != current.generation
                )
            )
            # An unchanged legacy failure has no occurrence clock. Preserve that
            # unknown rather than stamping a heartbeat as a fresh provider event.
            failure_at = now if new_failure else current.provider_failure_occurred_at
            failure_id = uuid.uuid4().hex if new_failure else current.provider_failure_event_id
            failure_generation = current.generation if new_failure else current.provider_failure_generation
            updated = current.model_copy(
                update={
                    "state": state,
                    "provider_phase": remote_state,
                    "provider_failure_code": provider_failure_code,
                    "provider_failure_occurred_at": failure_at if provider_failure_code is not None else None,
                    "provider_failure_event_id": failure_id if provider_failure_code is not None else None,
                    "provider_failure_generation": failure_generation if provider_failure_code is not None else None,
                    "observed_tpu_vms": observed_tpu_vms,
                    "runtime_process_count": runtime_process_count,
                    "materialized_assets": materialized_assets,
                    "billing_started_at": (
                        now
                        if state is CellState.READY and current.billing_started_at is None
                        else current.billing_started_at
                    ),
                    "billing_ticked_at": (
                        now
                        if state is CellState.READY and current.billing_started_at is None
                        else current.billing_ticked_at
                    ),
                    "provider_phase_started_at": phase_started_at,
                    "unavailable_reason": unavailable_reason,
                    "unavailable_detail": (provider_detail or None) if unavailable_reason else None,
                    "availability": availability,
                    "last_heartbeat_at": now if agent_responded else current.last_heartbeat_at,
                    "updated_at": now,
                }
            )
            if provider_failure_code is not CapacityProviderFailureCode.QUEUE_ADMISSION_DENIED and _heartbeat_only(
                current,
                updated,
                now,
                interval_s=self.settings.cell_heartbeat_persist_interval_s,
            ):
                # Nothing but the heartbeat moved, and the persisted one is still
                # recent. Writing it would re-serialize the whole local state
                # every observation interval for a timestamp only the dashboard
                # shows; any other change to the cell is written at once.
                return True
            await transaction.put_cell(updated)
            if provider_failure_code is CapacityProviderFailureCode.QUEUE_ADMISSION_DENIED:
                await self._fail_autoscale_workload_for_queue_admission_denial(
                    transaction,
                    updated,
                    provider_detail or "",
                )
            transition = current, updated
        if transition is not None:
            self._capacity_lifecycle.transition(*transition, reason=provider_detail)
        return True

    async def release_ready_dag_nodes_once(self) -> int:
        """Place-enable DAG nodes whose predecessors have all succeeded.

        A gated node is stored ``schedulable=False``, which the scheduler already
        filters on, so releasing it is a single flag flip rather than a second
        admission concept.

        A predecessor that FAILED or was CANCELLED cancels its dependents instead of
        stranding them. Leaving them PENDING forever is the same defect the orphan
        sweep exists to clear -- a workload nothing will ever settle, counted against
        quota and fair share while holding nothing -- and CANCELLED states the truth:
        this never ran because what it waited for did not succeed.
        """

        released = 0
        async with self.repository.transaction() as transaction:
            gated = [
                workload
                for workload in await transaction.list_workloads(state=WorkloadState.PENDING)
                if workload.depends_on and not workload.schedulable
            ]
            predecessor_states = {
                workload.id: workload.state
                for workload in await transaction.list_workloads()
                if any(workload.id in node.depends_on for node in gated)
            }

        ready: list[WorkloadRecord] = []
        for node in gated:
            states = [predecessor_states.get(predecessor) for predecessor in node.depends_on]
            if any(state in {WorkloadState.FAILED, WorkloadState.CANCELLED} or state is None for state in states):
                # A missing predecessor counts as failed: it was pruned or lost, and
                # a dependent that can never learn its outcome must not wait forever.
                try:
                    await self.service.cancel_workload(
                        self._principal,
                        node.id,
                        idempotency_key=f"dag:abandon:{node.id}:g{node.generation}",
                    )
                except (ConflictError, NotFoundError) as exc:
                    logger.warning("could not abandon DAG node %s: %s", node.id, exc)
                    continue
                released += 1
                continue
            if not all(state is WorkloadState.SUCCEEDED for state in states):
                continue
            ready.append(node)

        # Grouped by coordinator because the bound is per expansion. An
        # abandonment above is deliberately outside the bound: cancelling a node
        # frees a place in the queue rather than taking one.
        for coordinator_id in sorted({node.parent_workload_id or "" for node in ready}):
            released += await self._release_dag_nodes_for(
                coordinator_id or None,
                [node for node in ready if (node.parent_workload_id or "") == coordinator_id],
            )
        return released

    async def _release_dag_nodes_for(
        self,
        coordinator_id: str | None,
        ready: tp.Sequence[WorkloadRecord],
    ) -> int:
        """Release one DAG's ready nodes, bounded by the wave that admitted it.

        Every input is a durable row read inside this transaction. The plan comes
        from the expansion ledger, the occupancy from the coordinator's children,
        and which node goes next from the nodes' own stable order -- so a leader
        that dies here resumes with the same answer instead of re-deriving one
        from a counter it may not have committed.

        The whole release is one transaction. A leader that dies mid-release
        therefore commits all of it or none of it: nothing is released twice, and
        a node that was not released is still gated for the next pass.

        Args:
            coordinator_id: The DAG coordinator these nodes belong to, or
                ``None`` for a node with no coordinator, which no plan governs.
            ready: Nodes whose predecessors have all succeeded.

        Returns:
            How many nodes were made schedulable.
        """

        async with self.repository.transaction() as transaction:
            wave_size: int | None = None
            competing = 0
            if coordinator_id is not None:
                ledger = await transaction.get_expansion_admission(coordinator_id)
                if ledger is not None and ledger.state is ExpansionAdmissionState.ADMITTING:
                    wave_size = ledger.wave_size
                    # Counted over *every* child of the expansion: an entry root
                    # the wave phase released and a dependent this phase released
                    # occupy the queue identically, and the two paths bound each
                    # other only if they measure the same thing.
                    competing = sum(
                        1
                        for child in await transaction.list_child_workloads(coordinator_id)
                        if child.schedulable and child.state not in TERMINAL_WORKLOAD_STATES
                    )
            current: list[WorkloadRecord] = []
            for node in sorted(ready, key=_dag_release_order):
                stored = await transaction.get_workload(node.id)
                if stored is None or stored.schedulable or stored.state is not WorkloadState.PENDING:
                    continue
                current.append(stored)
            releasing = bounded_release(current, competing=competing, wave_size=wave_size)
            now = utcnow()
            for child in releasing:
                await transaction.put_workload(
                    child.model_copy(
                        update={
                            "schedulable": True,
                            "scheduling_reason": "dependencies satisfied; waiting for scheduling",
                            "updated_at": now,
                        }
                    )
                )
                await transaction.append_event(
                    AuditEvent(
                        id=f"evt_{uuid.uuid4().hex}",
                        actor=self._principal.subject,
                        org_id=child.org_id,
                        project_id=child.project_id,
                        entity_type="workload",
                        entity_id=child.id,
                        action="workload.dag_released",
                    )
                )
            return len(releasing)

    async def release_admitted_groups_once(self) -> int:
        """Release a heterogeneous group once every member can be placed.

        A group's members are differently shaped, so they cannot be one gang. Left
        as independent workloads, the members that fit lease immediately and burn
        accelerator hours for as long as the members that do not stay unplaceable.
        Gating them all on the whole group's feasibility is the one thing groups buy
        over siblings.

        The check is a *feasibility* gate, not a reservation, and the difference is
        worth stating where it is implemented: between this answer and the members'
        actual leases another workload can take the capacity, because the core
        scheduler has no reservation primitive. What this prevents is half a group
        holding accelerators while the rest is structurally infeasible -- the common
        failure -- not a race between two placement passes.

        Members are evaluated as if schedulable, since the scheduler filters out the
        flag this method exists to flip; and they are released in one transaction, so
        a leader that dies mid-release cannot leave a group half-open.
        """

        released = 0
        async with self.repository.transaction() as transaction:
            coordinators = [
                workload
                for workload in await transaction.list_workloads(state=WorkloadState.PENDING)
                if workload.coordination_kind is CoordinationKind.GROUP
            ]
            members_by_parent = {
                coordinator.id: await transaction.list_child_workloads(coordinator.id) for coordinator in coordinators
            }

        for coordinator in coordinators:
            members = members_by_parent.get(coordinator.id, [])
            gated = [member for member in members if not member.schedulable]
            if not gated or any(member.state is not WorkloadState.PENDING for member in gated):
                continue
            feasible = await self.service.group_is_placeable(
                self._principal, member_ids=tuple(member.id for member in gated)
            )
            if not feasible:
                continue
            async with self.repository.transaction() as transaction:
                current = [await transaction.get_workload(member.id) for member in gated]
                if any(item is None or item.schedulable or item.state is not WorkloadState.PENDING for item in current):
                    # State moved under us. Re-evaluate next cycle rather than
                    # releasing a partial group, which is the exact outcome the
                    # gate exists to prevent.
                    continue
                for item in current:
                    assert item is not None
                    await transaction.put_workload(
                        item.model_copy(
                            update={
                                "schedulable": True,
                                "scheduling_reason": "group admitted; waiting for scheduling",
                                "updated_at": utcnow(),
                            }
                        )
                    )
                    await transaction.append_event(
                        AuditEvent(
                            id=f"evt_{uuid.uuid4().hex}",
                            actor=self._principal.subject,
                            org_id=item.org_id,
                            project_id=item.project_id,
                            entity_type="workload",
                            entity_id=item.id,
                            action="workload.group_released",
                            payload={"group_id": coordinator.id, "members": len(current)},
                        )
                    )
                released += len(current)
        return released

    async def workload_logs(self, workload_id: str) -> str:
        """Read driver logs for an active or terminal workload attempt."""

        async with self.repository.transaction() as transaction:
            workload = await transaction.get_workload(workload_id)
        if workload is None:
            raise LookupError(f"workload {workload_id!r} does not exist")
        # Three different answers that were all the empty string, which read to a
        # caller as "this workload produced no output".
        if workload.spec.secret_mounts:
            return (
                "[eyvan] logs are suppressed for this workload because it mounts secrets; "
                "its output may contain their values."
            )
        if workload.remote_id is None:
            return f"[eyvan] this workload has not been dispatched to a cell yet (state: {workload.state.value})."
        try:
            members = await self._load_workload_members(workload)
        except StaleOutboxCommand:
            members = []
        if not members:
            # Exactly what the orphan sweep leaves behind: it settles a workload
            # whose leases are already gone. Reading logs then either indexed an
            # empty list or raised StaleOutboxCommand, and both answered HTTP 500
            # -- on precisely the records an operator reads logs for, because
            # something went wrong with them. `/log-page` handled the same state
            # cleanly, so the two surfaces disagreed about one workload.
            return await self._archived_driver_logs(
                workload,
                reason="the capacity this workload ran on has been released, so its logs are no longer readable "
                "from the cell",
            )
        # A lease can still name a cell whose execution is gone: the workload
        # finished, its container was reaped, and the observation no longer
        # carries a management UID. The agent answers that with an invalid
        # configuration error, which reached the operator as a bare HTTP 500 --
        # on exactly the workload they are reading logs for, because it failed.
        # The empty-members branch above already decided this class of answer is
        # a sentence rather than a 500; this is the same state one step later.
        try:
            if len(members) > 1:
                coordinator_lease, coordinator_cell = self._ordered_gang_member_pairs(members)[0]
                return await asyncio.to_thread(
                    self._agent_factory(coordinator_cell).gang_logs,
                    coordinator_lease.cell_id,
                    workload.id,
                    workload.remote_id,
                )
            _lease, cell = members[0]
            return await asyncio.to_thread(self._agent_factory(cell).logs, cell.id, workload.remote_id)
        except CellAgentInvalidConfigurationError as exc:
            return await self._archived_driver_logs(
                workload,
                reason=f"this workload's execution is no longer readable from its cell ({exc}) because the run "
                "ended and its container was reaped",
            )

    async def _archived_driver_logs(self, workload: WorkloadRecord, *, reason: str) -> str:
        """Render this attempt's archived output when no cell can serve it.

        The whole point of archiving. A live read is only possible while the
        capacity exists, and the read an operator actually needs is the one after
        it is gone -- the workload was preempted, the slice was released, the
        container was reaped. Both of those states used to end at a sentence
        explaining that archived output was "the remaining source" without ever
        going and getting it, so the archive settlement writes was reachable
        through ``/log-page`` and unreachable through ``logs``.

        Ranks are rendered in the live gang's own section format rather than as
        one merged stream, because that is the shape an operator comparing two
        ranks of a hung collective already knows how to read. A rank the archive
        could not replay keeps its section and says so: dropping it would turn a
        partial archive into a complete-looking log, which is the failure the
        per-rank discipline exists to prevent.

        Args:
            workload: The attempt whose archive is being read.
            reason: Why the live cell could not serve it, in the voice of the
                sentence this returns. Carried into both answers so the operator
                is told what happened to the capacity, not only what was found.

        Returns:
            The archived output, or a sentence naming why none was recoverable.
        """

        page = await self._archived_log_page(workload)
        by_rank: dict[int, list[LogRecord]] = {}
        for record in page.records:
            by_rank.setdefault(record.rank, []).append(record)
        if not by_rank:
            detail = page.detail or "this attempt has no archived logs"
            return f"[eyvan] {reason}, and no archived output was recoverable ({detail})."
        sections = [f"[eyvan] {reason}. Replaying archived output ({page.detail})."]
        for rank in sorted(set(by_rank) | set(page.truncated_ranks)):
            lines = by_rank.get(rank, ())
            marker = " (incomplete)" if rank in page.truncated_ranks else ""
            body = "\n".join(record.message for record in lines) or "[eyvan] no archived lines for this rank"
            sections.append(f"===== rank {rank} archived{marker} =====\n{body}")
        return "\n".join(sections)

    async def workload_log_page(
        self,
        workload_id: str,
        *,
        after: str = "",
        limit: int = 200,
        live_only: bool = False,
    ) -> JsonObject:
        """Return one resumable page of structured logs for a workload attempt.

        The cursor is a versioned opaque document carrying the attempt and one
        backend position per ordered cell. A multi-host cell's position is
        itself opaque and may contain another fan-out cursor. A cursor from a
        previous attempt is not resumed from: the read restarts and reports the
        attempt change.

        Silently continuing is the failure that matters here. A retry replaces
        the remote execution entirely, so the old cursor means nothing to the
        new one; a follower that kept using it would be shown the new attempt's
        first lines appended to the old attempt's tail, and would conclude the
        job had resumed where it left off.

        Ranks come from the topology frozen at dispatch, never from whatever the
        cells report now. An attempt with no frozen topology is served with its
        rank identity reported as unknown rather than reconstructed: today's
        cells cannot prove which host was rank N while that attempt started.

        ``live_only`` refuses the archive fallback and reports the cell's
        absence instead. Readers want the fallback; the two *writers* of the
        archive must not have it, because a read that quietly returns archived
        lines would make them archive their own output a second time -- growing
        without bound and duplicating every line on replay.
        """

        async with self.repository.transaction() as transaction:
            workload = await transaction.get_workload(workload_id)
            topology = (
                None if workload is None else await transaction.get_workload_topology(workload_id, workload.attempt)
            )
        if workload is None:
            raise LookupError(f"workload {workload_id!r} does not exist")
        frozen_state = _TOPOLOGY_UNKNOWN if topology is None else _TOPOLOGY_FROZEN
        frozen_digest = None if topology is None else topology.digest
        frozen_ranks = None if topology is None else topology.total_rank_count
        if workload.spec.secret_mounts:
            return _log_page_document(
                LogPage(unavailable=True, detail=_SECRET_OUTPUT_SUPPRESSED),
                attempt=workload.attempt,
                topology_state=frozen_state,
                topology_digest=frozen_digest,
                expected_ranks=frozen_ranks,
            )
        if workload.remote_id is None:
            return _log_page_document(
                LogPage(),
                attempt=workload.attempt,
                topology_state=frozen_state,
                topology_digest=frozen_digest,
                expected_ranks=frozen_ranks,
            )

        archive_attempt_changed = False
        if _is_archive_cursor(after):
            # A follower already reading the archive. Its positions address
            # immutable objects, so there is no live journal to consult and no
            # cell whose absence has to be discovered first.
            archive_attempt_changed, archive_positions = self._decoded_archive_cursor(after, workload)
            if not archive_attempt_changed and live_only:
                return _log_page_document(
                    LogPage(unavailable=True, detail=_LIVE_READ_HAS_NO_CELL),
                    attempt=workload.attempt,
                    topology_state=frozen_state,
                    topology_digest=frozen_digest,
                    expected_ranks=frozen_ranks,
                )
            if not archive_attempt_changed:
                archived = await self._archived_log_page(workload, positions=archive_positions, limit=limit)
                return _log_page_document(
                    archived,
                    attempt=workload.attempt,
                    next_cursor=archived.next_cursor,
                    topology_state=frozen_state,
                    topology_digest=frozen_digest,
                    expected_ranks=frozen_ranks,
                )
            # A retry. Its journals are new objects, and they may still be live,
            # so the read restarts on the ordinary path instead of inside the
            # previous attempt's archive.
            after = ""

        attempt_changed, cursors, cursor_is_flat = self._decoded_cursor(after, workload, topology)
        attempt_changed = attempt_changed or archive_attempt_changed

        try:
            members = await self._load_workload_members(workload)
        except StaleOutboxCommand:
            # No current lease: the cell is released and its journal went with
            # it. This is the ordinary path for any finished workload, not an
            # error -- and falling back to the archive is the entire reason the
            # archive exists.
            members = ()
        if members and not any(lease.state in ACTIVE_LEASE_STATES for lease, _cell in members):
            # Every lease is already given back, but the cell outlived it: an
            # attached TPU is not destroyed on release, it is re-fenced at a new
            # generation for the next job. Its journal no longer belongs to this
            # attempt, so reading it live was refused as a generation mismatch
            # and the attempt's archived output became unreachable. A released
            # attempt is read from its archive, exactly like a deleted cell's.
            members = ()
        if not members and live_only:
            # An archive writer asking for live lines. There are none, and
            # saying so is the whole point of the flag: the fallback below would
            # hand it this attempt's own archive to archive again.
            return _log_page_document(
                LogPage(unavailable=True, detail=_LIVE_READ_HAS_NO_CELL),
                attempt=workload.attempt,
                attempt_changed=attempt_changed,
                topology_state=frozen_state,
                topology_digest=frozen_digest,
                expected_ranks=frozen_ranks,
            )
        if not members:
            # The handover. The follower is holding live positions and the
            # journals they name are gone, so they are translated into the
            # archive's own offsets rather than discarded -- discarding them is
            # what replayed the whole archive on every poll.
            positions, position_gap = self._archive_positions_from_live(
                {} if attempt_changed else cursors,
                cursor_is_flat=cursor_is_flat,
                topology=topology,
            )
            archived = await self._archived_log_page(
                workload,
                positions=positions,
                position_gap=position_gap,
                limit=limit,
            )
            return _log_page_document(
                archived,
                attempt=workload.attempt,
                attempt_changed=attempt_changed,
                next_cursor=archived.next_cursor,
                topology_state=frozen_state,
                topology_digest=frozen_digest,
                expected_ranks=frozen_ranks,
            )
        if topology is None:
            return await self._unproven_log_page(
                workload,
                members,
                cursors={} if cursor_is_flat else cursors,
                attempt_changed=attempt_changed,
                limit=limit,
            )
        return await self._frozen_log_page(
            workload,
            topology,
            members,
            positions=cursors,
            cursor_is_flat=cursor_is_flat,
            attempt_changed=attempt_changed,
            limit=limit,
        )

    def _decoded_cursor(
        self,
        after: str,
        workload: WorkloadRecord,
        topology: WorkloadAttemptTopology | None,
    ) -> tuple[bool, dict[int, str], bool]:
        """Decode one presented cursor, or refuse the read outright.

        Returns whether the attempt changed, the positions, and whether those
        positions name global ranks (a bound cursor) rather than cell indices.

        Only empty input starts a read. A malformed, oversized, foreign, or
        unresolvable cursor raises :class:`LogCursorRejectedError` instead of
        being collapsed back into "start from the beginning" -- the two are
        indistinguishable to a client, so the silent version re-reads the whole
        journal and never says why. An attempt change is the one exception the
        design names: the positions genuinely no longer exist, and DR-0002
        requires reporting it and beginning the new attempt at empty positions.

        Raises:
            LogCursorRejectedError: If the cursor cannot address this read.
        """

        outcome = decode_log_cursor(after, expected=_live_cursor_context(workload, topology))
        if isinstance(outcome, CursorInvalid) and topology is not None and not after.startswith("eyvan-log-v"):
            # The pre-envelope grammar nests an unescaped delimiter inside opaque
            # leaves, so it is reversible only against a known rank set. The one
            # that was ever emitted addressed *cells*, so that is the witness --
            # supplied explicitly rather than guessed, which is what DR-0002 means
            # by accepting v1 only where it is reversible.
            outcome = decode_log_cursor(
                after,
                expected=_live_cursor_context(workload, topology, ranks=tuple(range(len(topology.cells)))),
            )
        if isinstance(outcome, CursorFresh):
            return False, {}, False
        if isinstance(outcome, CursorIncompatible) and outcome.reason == CURSOR_ATTEMPT_CHANGED:
            # A retry runs a different execution; the old positions name lines in
            # journals that no longer exist. Reported, then restarted -- which is
            # the only refusal DR-0002 lets resolve to empty positions.
            return True, {}, False
        if (
            isinstance(outcome, CursorInvalid)
            and topology is None
            and not after.startswith("eyvan-log-v")
            and outcome.reason == CURSOR_RESET_REQUIRED
        ):
            # An attempt that started before its rank space was frozen has no
            # witness for a legacy cursor, and never will: refusing would leave
            # that follower with a position it can never make valid. Every
            # in-flight attempt is in exactly this state on the upgrade that
            # introduces the frozen topology, so this is the ordinary path once,
            # not an edge case. The read restarts and the page already reports the
            # rank space as unknown.
            return False, {}, False
        if isinstance(outcome, (CursorInvalid, CursorIncompatible)):
            raise LogCursorRejectedError(
                reason=outcome.reason,
                detail=outcome.detail,
                incompatible=isinstance(outcome, CursorIncompatible),
                workload_id=workload.id,
                attempt=workload.attempt,
                topology=None if topology is None else topology.digest,
                expected_ranks=None if topology is None else topology.total_rank_count,
            )
        positions = outcome.cursors
        if outcome.bound and topology is not None:
            unknown = sorted(rank for rank in positions if not topology.contains_rank(rank))
            if unknown:
                raise LogCursorRejectedError(
                    reason=CURSOR_UNKNOWN_RANK,
                    detail=(
                        "log cursor names rank(s) outside this attempt: " + ", ".join(str(rank) for rank in unknown)
                    ),
                    incompatible=True,
                    workload_id=workload.id,
                    attempt=workload.attempt,
                    topology=topology.digest,
                    expected_ranks=topology.total_rank_count,
                )
        return False, positions, outcome.bound

    def _decoded_archive_cursor(self, after: str, workload: WorkloadRecord) -> tuple[bool, dict[int, str]]:
        """Decode a cursor presented against this attempt's archive.

        The same discipline as the live decode, against the archive read instead:
        only empty input starts from the beginning, an attempt change is reported
        and restarted because the retry archived different objects, and anything
        else is refused rather than collapsed into a first read.

        Returns:
            Whether the attempt changed, and the per-rank archive positions.

        Raises:
            LogCursorRejectedError: If the cursor cannot address this archive.
        """

        outcome = decode_log_cursor(after, expected=_archive_cursor_context(workload))
        if isinstance(outcome, CursorFresh):
            return False, {}
        if isinstance(outcome, CursorIncompatible) and outcome.reason == CURSOR_ATTEMPT_CHANGED:
            return True, {}
        if isinstance(outcome, (CursorInvalid, CursorIncompatible)):
            raise LogCursorRejectedError(
                reason=outcome.reason,
                detail=outcome.detail,
                incompatible=isinstance(outcome, CursorIncompatible),
                workload_id=workload.id,
                attempt=workload.attempt,
            )
        return False, outcome.cursors

    def _archive_positions_from_live(
        self,
        cursors: tp.Mapping[int, str],
        *,
        cursor_is_flat: bool,
        topology: WorkloadAttemptTopology | None,
    ) -> tuple[dict[int, str], str]:
        """Express a live cursor's positions in the archive's global rank space.

        A bound cursor already names global ranks. A per-cell cursor names cells,
        and only the frozen topology says which ranks a cell held. Without it,
        the first cell is still placeable -- its global rank offset is zero by
        construction -- and every later cell is not, because the widths in front
        of it were never recorded.

        A cell that cannot be placed is *named on the page*, and the ranks behind
        it are served from the archive's first line. That is the same answer the
        live decode gives a cursor from before the rank space was frozen, and for
        the same reason: refusing would leave that follower holding a position it
        can never make valid. The repetition happens at most once, because this
        page returns an archive cursor that does place every rank -- unlike the
        read this replaces, which replayed the whole archive on every poll and
        returned no cursor at all.

        Returns:
            The positions by global rank, and the sentence naming any cell that
            could not be placed (``""`` when all of them were).
        """

        if not cursors:
            return {}, ""
        if cursor_is_flat:
            return {rank: value for rank, value in cursors.items() if value}, ""
        positions: dict[int, str] = {}
        unplaced: list[int] = []
        for index, value in sorted(cursors.items()):
            if not value:
                continue
            frozen = topology.cells[index] if topology is not None and 0 <= index < len(topology.cells) else None
            if frozen is None and index != 0:
                # Only the first cell's global rank offset is knowable without a
                # frozen rank space: it is zero by construction. Every later cell
                # begins after widths this attempt never recorded, and inventing
                # them would give one rank another rank's archived lines.
                unplaced.append(index)
                continue
            leaves, understood = (
                _leading_cell_leaf_positions(value)
                if frozen is None
                else _cell_leaf_positions(value, width=frozen.runtime_process_count)
            )
            if not understood:
                unplaced.append(index)
                continue
            offset = 0 if frozen is None else frozen.global_rank_offset
            for local, leaf in leaves.items():
                positions[offset + local] = leaf
        gap = ""
        if unplaced:
            gap = (
                "the presented cursor names cell(s) whose ranks this attempt never froze ("
                + ", ".join(str(index) for index in unplaced)
                + "), so those ranks are served from the start of the archive"
            )
        return positions, gap

    async def _frozen_log_page(
        self,
        workload: WorkloadRecord,
        topology: WorkloadAttemptTopology,
        members: tp.Sequence[tuple[LeaseRecord, CellRecord]],
        *,
        positions: tp.Mapping[int, str],
        cursor_is_flat: bool,
        attempt_changed: bool,
        limit: int,
    ) -> JsonObject:
        """Serve one live page against the rank space frozen at dispatch.

        Fan-out is grouped at the cell RPC boundary only. Inside this method
        every position is a global rank: one per expected rank on the way out,
        regrouped into a cell-local envelope for exactly as long as the request
        to that cell is in flight, and mapped straight back afterwards.
        """

        if len(members) > 1:
            # Keeps the gang's one-trusted-endpoint validation, whose failure is
            # a configuration error rather than a page to serve.
            self._ordered_gang_member_pairs(members)
        current = {cell.id: cell for _lease, cell in members}
        if not cursor_is_flat:
            positions = self._positions_from_cell_cursors(topology, positions)

        live: dict[int, str] = {
            rank: value for rank, value in positions.items() if topology.contains_rank(rank) and value
        }
        pages: dict[int, LogPage] = {}
        changed = False
        for frozen in topology.cells:
            offset = frozen.global_rank_offset
            width = frozen.runtime_process_count
            declared = tuple(range(offset, offset + width))
            cell = current.get(frozen.cell_id)
            mismatch = (
                f"cell {frozen.cell_id} no longer holds this attempt"
                if cell is None
                else _topology_mismatch(frozen, cell, self._cell_worker_endpoints(cell))
            )
            if mismatch:
                # Not renumbered and not dropped: the frozen range this cell owns
                # is exactly what is missing, and every other rank keeps its
                # identity and its position.
                changed = True
                pages[offset] = LogPage(unavailable=True, detail=mismatch, truncated_ranks=declared)
                continue
            try:
                payload = await asyncio.to_thread(
                    self._agent_factory(cell).log_page,
                    cell.id,
                    workload.remote_id,
                    after=self._cell_after_cursor(live, frozen, attempt=workload.attempt),
                    limit=limit,
                )
            except Exception as exc:
                # The cell boundary failed before its inner ranks could answer;
                # every frozen host is therefore explicitly truncated.
                pages[offset] = LogPage(unavailable=True, detail=str(exc), truncated_ranks=declared)
                continue
            page = _log_page_from_document(payload, rank=offset, runtime_process_count=width)
            leaves, understood = _cell_leaf_positions(page.next_cursor, width=width)
            if not understood:
                detail = f"cell {frozen.cell_id} returned a cursor its frozen worker order cannot explain"
                page = LogPage(
                    records=page.records,
                    unavailable=page.unavailable,
                    detail=f"{page.detail}; {detail}" if page.detail else detail,
                    truncated_ranks=tuple(sorted(set(page.truncated_ranks) | set(declared))),
                )
            for local, leaf in leaves.items():
                live[offset + local] = leaf
            pages[offset] = page
        return _log_page_document(
            merge_pages(pages) if pages else LogPage(),
            attempt=workload.attempt,
            attempt_changed=attempt_changed,
            next_cursor=self._encode_frozen_cursor(workload, topology, live),
            topology_state=_TOPOLOGY_CHANGED if changed else _TOPOLOGY_FROZEN,
            topology_digest=topology.digest,
            expected_ranks=topology.total_rank_count,
            missing_ranks=[rank for rank in range(topology.total_rank_count) if rank not in live],
        )

    @classmethod
    def _encode_frozen_cursor(
        cls,
        workload: WorkloadRecord,
        topology: WorkloadAttemptTopology,
        positions: tp.Mapping[int, str],
    ) -> str:
        """Encode the attempt's positions in whichever rank space is being emitted.

        The flat per-global-rank envelope and the per-cell one flip together with
        the codec's own emission flag, because they are one decision: a replica
        that emits global ranks inside the envelope an older replica reads as cell
        indices would send that replica to the wrong journal. Both are decodable
        here regardless, which is what makes the rollback direction safe.
        """

        if not log_cursor_v2_emission_enabled():
            return _join_log_cursor(workload.attempt, cls._cell_cursors(topology, positions, attempt=workload.attempt))
        leaves: list[CursorPosition] = []
        for frozen in topology.cells:
            for local in range(frozen.runtime_process_count):
                rank = frozen.global_rank_offset + local
                cursor = positions.get(rank, "")
                if cursor:
                    leaves.append(CursorPosition(rank=rank, cursor=cursor, cell=frozen.cell_id, local=local))
        return encode_log_cursor(leaves, context=_live_cursor_context(workload, topology), bound=True)

    @staticmethod
    def _cell_after_cursor(
        positions: tp.Mapping[int, str],
        frozen: WorkloadTopologyCell,
        *,
        attempt: int,
    ) -> str:
        """Regroup global-rank positions into what one cell understands."""

        offset = frozen.global_rank_offset
        leaves = {
            local: positions[offset + local]
            for local in range(frozen.runtime_process_count)
            if positions.get(offset + local)
        }
        if frozen.runtime_process_count == 1:
            return leaves.get(0, "")
        return _join_log_cursor(attempt, leaves)

    @classmethod
    def _cell_cursors(
        cls,
        topology: WorkloadAttemptTopology,
        positions: tp.Mapping[int, str],
        *,
        attempt: int,
    ) -> dict[int, str]:
        """Encode positions in the per-cell shape a pre-flat replica understands."""

        encoded: dict[int, str] = {}
        for index, frozen in enumerate(topology.cells):
            value = cls._cell_after_cursor(positions, frozen, attempt=attempt)
            if value:
                encoded[index] = value
        return encoded

    @staticmethod
    def _positions_from_cell_cursors(
        topology: WorkloadAttemptTopology,
        cursors: tp.Mapping[int, str],
    ) -> dict[int, str]:
        """Read per-cell positions back into the one global rank space.

        The decoder half of the rollout: a cursor issued before flat emission is
        resolved to exact per-rank positions rather than restarted, which is what
        lets a replica move in either direction without replaying a journal.
        """

        positions: dict[int, str] = {}
        for index, frozen in enumerate(topology.cells):
            value = cursors.get(index, "")
            leaves, understood = _cell_leaf_positions(value, width=frozen.runtime_process_count)
            if not understood:
                continue
            for local, leaf in leaves.items():
                positions[frozen.global_rank_offset + local] = leaf
        return positions

    async def _unproven_log_page(
        self,
        workload: WorkloadRecord,
        members: tp.Sequence[tuple[LeaseRecord, CellRecord]],
        *,
        cursors: tp.Mapping[int, str],
        attempt_changed: bool,
        limit: int,
    ) -> JsonObject:
        """Serve an attempt dispatched before its rank space was ever frozen.

        The offsets here are recomputed from live cells, which is exactly the
        thing the frozen record exists to stop -- so the page says so. Nothing is
        written back: reconstructing a map from today's inventory and storing it
        would turn an unproven guess into a durable claim, and this attempt's
        cells may already have been reconciled to a different width.
        """

        # ``_ordered_gang_member_pairs`` validates a *gang* -- it refuses fewer
        # than two cells -- so a single-cell workload uses its one member
        # directly. The gang path gets the deterministic cell ordering that
        # makes rank N the same host on every read.
        ordered = self._ordered_gang_member_pairs(members) if len(members) > 1 else tuple(members)
        pages: dict[int, LogPage] = {}
        advanced: dict[int, str] = {}
        global_rank = 0
        for cell_index, (_lease, cell) in enumerate(ordered):
            try:
                payload = await asyncio.to_thread(
                    self._agent_factory(cell).log_page,
                    cell.id,
                    workload.remote_id,
                    # The outer cursor addresses cells. A multi-host cell's value
                    # is itself an opaque cursor addressing each inner journal.
                    after=cursors.get(cell_index, ""),
                    limit=limit,
                )
            except Exception as exc:
                # The cell boundary failed before its inner ranks could answer;
                # every declared host is therefore explicitly truncated.
                page = LogPage(
                    unavailable=True,
                    detail=str(exc),
                    truncated_ranks=tuple(range(global_rank, global_rank + cell.runtime_process_count)),
                )
            else:
                page = _log_page_from_document(
                    payload,
                    rank=global_rank,
                    runtime_process_count=cell.runtime_process_count,
                )
            pages[global_rank] = page
            # A cell that produced no new lines (or is temporarily unreachable)
            # must keep its previous nested position. Dropping it would replay
            # every inner journal in that cell on the next poll.
            position = page.next_cursor or cursors.get(cell_index, "")
            if position:
                advanced[cell_index] = position
            global_rank += cell.runtime_process_count
        return _log_page_document(
            merge_pages(pages) if pages else LogPage(),
            attempt=workload.attempt,
            attempt_changed=attempt_changed,
            cursors=advanced,
            topology_state=_TOPOLOGY_UNKNOWN,
            topology_digest=None,
            expected_ranks=None,
        )

    async def validate_log_cursor(self, workload_id: str, after: str) -> None:
        """Refuse a log cursor before a response has been committed.

        The paged route can answer a refusal with a status code because nothing
        has been sent yet. A stream cannot: by the time the generator runs, the
        client already has ``200``, and raising inside it ends the stream with no
        way to say why. This is the preflight for that case, and it decodes
        against exactly the same read the page would, so the two can never
        disagree about whether a cursor is usable.

        Args:
            workload_id: Workload whose journal is being followed.
            after: The cursor exactly as the client returned it.

        Raises:
            LookupError: If the workload does not exist.
            LogCursorRejectedError: If the cursor cannot address this read.
        """

        async with self.repository.transaction() as transaction:
            workload = await transaction.get_workload(workload_id)
            topology = (
                None if workload is None else await transaction.get_workload_topology(workload_id, workload.attempt)
            )
        if workload is None:
            raise LookupError(f"workload {workload_id!r} does not exist")
        if _is_archive_cursor(after):
            # A browser reconnecting to a workload that finished while it was
            # away is holding an archive position. Preflighting it against the
            # live read would refuse the cursor this endpoint just issued.
            self._decoded_archive_cursor(after, workload)
            return
        self._decoded_cursor(after, workload, topology)

    async def _manifest_archive_replay(self, workload: WorkloadRecord) -> _ArchiveReplay | None:
        """Read a terminal archive from its schema-2 manifest, or ``None``.

        ``None`` means "there is no manifest to read" and hands the caller back to
        the embedded-index decoder. Every other outcome is answered here, because
        once a manifest exists it is the only record that knows which ranks were
        *expected*: falling back after finding one would silently downgrade a
        named gap into a rank nobody ever mentions.

        Bytes are verified against the digest and size the writer recorded. A
        mismatch is reported as a gap and never replayed -- an archive that does
        not match its own manifest is not evidence about what a rank printed.

        Records stay separated by rank rather than merged here, because a
        follower resumes per rank: merging first would lose each line's place in
        its own object, which is the only thing an archive offset can name.
        """

        if self.artifact_access is None:
            return None
        try:
            payload = await self.service.read_managed_artifact(
                org_id=workload.org_id,
                project_id=workload.project_id,
                uri=log_manifest_uri(workload.id, workload.attempt),
            )
        except NotFoundError:
            return None
        except Exception:
            # A manifest that exists but cannot be read conclusively must not fall
            # through to the embedded index: that index describes fewer ranks, so
            # the reply would look more complete than the evidence supports.
            return _ArchiveReplay(detail="the archive manifest could not be read conclusively", unavailable=True)

        try:
            document = json.loads(payload.decode())
        except (UnicodeDecodeError, ValueError):
            return _ArchiveReplay(detail="the archive manifest is unreadable", unavailable=True)
        if not isinstance(document, dict) or document.get("schema") != _LOG_ARCHIVE_MANIFEST_SCHEMA_V2:
            return _ArchiveReplay(detail="the archive manifest is not a schema-2 log manifest", unavailable=True)

        raw_entries = document.get("ranks")
        if not isinstance(raw_entries, list):
            return _ArchiveReplay(detail="the archive manifest declares no ranks", unavailable=True)

        ranks: dict[int, tuple[LogRecord, ...]] = {}
        truncated: set[int] = set()
        gaps: list[str] = []
        fetched = await self._read_archive_objects(
            workload,
            [
                item["uri"]
                for item in raw_entries
                if isinstance(item, dict)
                and item.get("status") in _ARCHIVE_STATUS_READABLE
                and isinstance(item.get("uri"), str)
                and item["uri"]
            ],
        )
        for item in raw_entries:
            if not isinstance(item, dict):
                continue
            rank = item.get("rank")
            if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
                continue
            status = item.get("status")
            if status in _ARCHIVE_STATUS_UNREACHED:
                # Named, never read: the writer already knows there are no bytes.
                truncated.add(rank)
                continue
            if status == _ARCHIVE_STATUS_QUIET:
                continue
            if status not in _ARCHIVE_STATUS_READABLE:
                truncated.add(rank)
                gaps.append(f"{rank} (unknown status)")
                continue
            uri = item.get("uri")
            if not isinstance(uri, str) or not uri:
                truncated.add(rank)
                gaps.append(f"{rank} (no object)")
                continue
            blob = fetched.get(uri)
            if not isinstance(blob, bytes):
                truncated.add(rank)
                gaps.append(f"{rank} (unreadable)")
                continue
            expected_size = item.get("size_bytes")
            expected_digest = item.get("sha256")
            if isinstance(expected_size, int) and not isinstance(expected_size, bool) and len(blob) != expected_size:
                truncated.add(rank)
                gaps.append(f"{rank} (size mismatch)")
                continue
            if isinstance(expected_digest, str) and expected_digest:
                if hashlib.sha256(blob).hexdigest() != expected_digest.removeprefix("sha256:"):
                    truncated.add(rank)
                    gaps.append(f"{rank} (digest mismatch)")
                    continue
            ranks[rank] = _archived_rank_records(_archived_entries(blob), rank=rank)
            if status == _ARCHIVE_STATUS_TRUNCATED:
                truncated.add(rank)

        detail = "archived"
        if document.get("complete") is not True:
            detail += "; the archive is incomplete"
        omitted = document.get("objects_omitted")
        if isinstance(omitted, int) and not isinstance(omitted, bool) and omitted > 0:
            detail += f"; {omitted} rank object(s) were not written"
        if document.get("enumeration_capped") is True:
            detail += "; the expected rank list was capped"
        if gaps:
            detail += "; rank archives could not be replayed: " + ", ".join(gaps)
        return _ArchiveReplay(
            ranks=ranks,
            truncated=frozenset(truncated),
            detail=detail,
            unavailable=bool(document.get("unavailable")),
        )

    async def _chunked_archive_replay(self, workload: WorkloadRecord) -> _ArchiveReplay | None:
        """Read one attempt's incremental chunks back, in the order written.

        These are the lines that were saved *while* the workload ran, so they
        are the only ones that exist at all when a machine is reclaimed rather
        than released. ``None`` means this attempt has no chunks -- either it
        predates incremental archival or the phase is switched off -- and the
        terminal archive is then served exactly as it was before.

        A chunk that cannot be read is a hole in the middle of a stream, not the
        end of one: the ranks it held are reported truncated and the remaining
        chunks are still served, because stopping at the first unreadable object
        would hide every line after it behind one lost write.
        """

        chunks = await self._list_log_chunks(workload)
        if not chunks:
            return None
        ranks: dict[int, list[LogRecord]] = {}
        missing: list[int] = []
        malformed = 0
        truncated: set[int] = set()
        fetched = await self._read_archive_objects(workload, [uri for _sequence, uri in chunks])
        for sequence, uri in chunks:
            payload = fetched.get(uri)
            if not isinstance(payload, bytes):
                missing.append(sequence)
                continue
            entries = [entry for entry in _archived_entries(payload) if isinstance(entry, dict)]
            header = _chunk_header(payload)
            truncated.update(_chunk_truncated_ranks(header))
            for record in log_page_from_json({"records": entries}).records:
                if record.rank < 0:
                    # A rank the writer could not place. Counted rather than
                    # filed under rank zero: two malformed ranks collide there
                    # and the page then looks complete to whoever reads it.
                    malformed += 1
                    continue
                ranks.setdefault(record.rank, []).append(record)
        detail = f"replayed from {len(chunks)} archived chunk(s)"
        if missing:
            detail += "; chunk(s) could not be read: " + ", ".join(str(sequence) for sequence in sorted(missing))
            # Which ranks lost lines is unknowable from an object that will not
            # open, so every rank the stream carries is marked incomplete.
            truncated.update(ranks)
        if malformed:
            detail += f"; {malformed} archived line(s) named no valid rank"
        return _ArchiveReplay(
            ranks={rank: tuple(lines) for rank, lines in sorted(ranks.items())},
            truncated=frozenset(truncated),
            detail=detail,
            unavailable=bool(missing) and not any(ranks.values()),
        )

    async def _read_archive_objects(
        self, workload: WorkloadRecord, uris: tp.Sequence[str]
    ) -> dict[str, bytes | Exception]:
        """Read archive objects concurrently; each answer is its bytes or the error it raised.

        One at a time, a 16-rank archive in object storage took over 70 s to
        replay -- past the SDK's request timeout, so a finished job's logs read
        as "control plane unavailable". Bounded by ``log_archive_concurrency``,
        the same ceiling the writer uses. Callers keep their own per-object
        failure handling: an error is returned in place, never raised.
        """

        gate = asyncio.Semaphore(max(int(self.settings.log_archive_concurrency), 1))

        async def read(uri: str) -> tuple[str, bytes | Exception]:
            async with gate:
                try:
                    return uri, await self.service.read_managed_artifact(
                        org_id=workload.org_id,
                        project_id=workload.project_id,
                        uri=uri,
                    )
                except Exception as exc:
                    return uri, exc

        return dict(await asyncio.gather(*(read(uri) for uri in dict.fromkeys(uris))))

    async def _indexed_archive_replay(self, workload: WorkloadRecord) -> _ArchiveReplay:
        """Read an archive that carries its index inside rank zero's own object.

        The shape written before the standalone manifest, and the only shape
        already-terminal attempts have. Kept exactly as it was, per rank, so an
        upgrade does not make an existing archive unreadable.
        """

        async def read_rank(rank: int) -> bytes:
            return await self.service.read_managed_artifact(
                org_id=workload.org_id,
                project_id=workload.project_id,
                uri=log_uri(workload.id, workload.attempt, rank),
            )

        try:
            first_payload = await read_rank(0)
        except NotFoundError:
            return _ArchiveReplay(detail="this attempt has no archived logs", unavailable=True)
        except Exception:
            return _ArchiveReplay(detail="archived logs could not be read conclusively", unavailable=True)

        manifest: JsonObject | None = None
        record_entries: dict[int, list[JsonObject]] = {0: []}
        for entry in _archived_entries(first_payload):
            if (
                isinstance(entry, list)
                and len(entry) == 2
                and entry[0] == LOG_ARCHIVE_MANIFEST_KEY
                and isinstance(entry[1], dict)
                and entry[1].get("version") == LOG_ARCHIVE_MANIFEST_VERSION
            ):
                manifest = entry[1]
            elif isinstance(entry, dict):
                record_entries[0].append(entry)

        truncated: set[int] = set()
        unavailable = False
        missing_archives: list[int] = []
        if manifest is not None:
            raw_ranks = manifest.get("ranks")
            raw_truncated = manifest.get("truncated_ranks")
            ranks = {
                rank
                for rank in (raw_ranks if isinstance(raw_ranks, list) else [])
                if isinstance(rank, int) and not isinstance(rank, bool) and rank >= 0
            }
            truncated.update(
                rank
                for rank in (raw_truncated if isinstance(raw_truncated, list) else [])
                if isinstance(rank, int) and not isinstance(rank, bool) and rank >= 0
            )
            # A writer declares every real or truncated rank in ``ranks``. Union
            # here as rolling compatibility with an early manifest that named a
            # truncated rank only in the dedicated field; if its object contains
            # surviving lines they must still be replayed.
            ranks.update(truncated)
            unavailable = bool(manifest.get("unavailable"))
            others = sorted(ranks - {0})
            fetched = await self._read_archive_objects(
                workload, [log_uri(workload.id, workload.attempt, rank) for rank in others]
            )
            for rank in others:
                blob = fetched.get(log_uri(workload.id, workload.attempt, rank))
                if isinstance(blob, bytes):
                    record_entries[rank] = [entry for entry in _archived_entries(blob) if isinstance(entry, dict)]
                else:
                    # The manifest says this rank existed. Losing its object is
                    # therefore explicit truncation, not the end of a contiguous
                    # scan and not a silent empty rank.
                    truncated.add(rank)
                    missing_archives.append(rank)
        else:
            # Legacy archives had no manifest and wrote ranks contiguously from
            # zero. Retain that decoder so already-terminal workloads remain
            # readable after the control plane is upgraded.
            for rank in range(1, _ARCHIVED_RANK_SCAN_LIMIT):
                try:
                    record_entries[rank] = [
                        entry for entry in _archived_entries(await read_rank(rank)) if isinstance(entry, dict)
                    ]
                except Exception:
                    break

        detail = "archived"
        if missing_archives:
            detail += "; rank archives could not be read: " + ", ".join(str(rank) for rank in missing_archives)
        return _ArchiveReplay(
            ranks={rank: _archived_rank_records(entries, rank=rank) for rank, entries in sorted(record_entries.items())},
            truncated=frozenset(truncated),
            detail=detail,
            unavailable=unavailable,
        )

    async def _archived_log_page(
        self,
        workload: WorkloadRecord,
        *,
        positions: tp.Mapping[int, str] | None = None,
        position_gap: str = "",
        limit: int | None = None,
    ) -> LogPage:
        """Read one attempt's archived logs back out of managed storage.

        A schema-2 manifest object is authoritative when present: it names every
        *expected* rank, its fate, and its object's digest and size, so a rank
        that printed nothing is distinguishable from one whose host was never
        reached, and bytes that contradict it are reported as a gap rather than
        replayed. Older archives carry a manifest in rank zero. It records the exact sparse
        global-rank set and which ranks were unavailable at collection time, so
        replay does not turn a partial live read into a complete-looking one.
        Archives written before the manifest remain readable through the former
        contiguous-rank scan.

        ``positions`` is where a follower had reached, expressed per global rank,
        and is what makes a live-to-archive handover continue instead of starting
        the archive again on every poll. The returned page carries the archive
        cursor to present next.

        Args:
            workload: The attempt whose archive is being read.
            positions: Per-rank positions the follower presented. Empty or
                ``None`` is a first read of the archive.
            position_gap: Which part of the presented cursor could not be
                expressed per rank at all, when any of it could not. Carried onto
                the page, so a read that restarted part of the archive says so.
            limit: Most lines to serve per rank, or ``None`` for all of them.

        Returns:
            The page, whose ``next_cursor`` is the archive position to resume at.
        """

        if self.artifact_access is None:
            return LogPage(unavailable=True, detail="managed artifacts are not configured on this replica")
        chunked = await self._chunked_archive_replay(workload)
        replay = await self._manifest_archive_replay(workload)
        if replay is None:
            replay = await self._indexed_archive_replay(workload)
        if chunked is not None:
            replay = _joined_archive_replays(chunked, replay)
        return self._archive_window(
            workload,
            replay,
            positions=positions or {},
            position_gap=position_gap,
            limit=limit,
        )

    def _archive_window(
        self,
        workload: WorkloadRecord,
        replay: _ArchiveReplay,
        *,
        positions: tp.Mapping[int, str],
        position_gap: str,
        limit: int | None,
    ) -> LogPage:
        """Serve the slice of one archive a follower has not been shown yet.

        Each rank is placed independently and advanced independently, because
        that is how the objects are written: one immutable object per rank, in
        the order its lines were captured. A rank whose position cannot be placed
        contributes no lines and keeps its position, so the gap is reported again
        on the next poll rather than becoming a silent replay of that object.

        ``limit`` bounds each rank rather than the merged page, matching the live
        read -- which passes the same number to every cell it fans out to -- so
        one chatty rank cannot consume another rank's share of the page.
        """

        served: dict[int, LogPage] = {}
        advanced: dict[int, str] = {}
        truncated = set(replay.truncated)
        gaps: list[str] = []
        for rank in sorted(replay.ranks):
            lines = replay.ranks[rank]
            placement = resolve_archive_position(
                positions.get(rank, ""),
                [record.cursor for record in lines],
                rank=rank,
            )
            if isinstance(placement, ArchiveGap):
                truncated.add(rank)
                gaps.append(f"{rank} ({placement.reason})")
                advanced[rank] = encode_unplaced_archive_position(placement.position)
                continue
            taken = lines[placement.offset :] if limit is None else lines[placement.offset : placement.offset + limit]
            reached = placement.offset + len(taken)
            if taken:
                served[rank] = LogPage(records=taken)
            if reached:
                # A rank still at zero needs no position: an absent one already
                # means "from the start", and emitting one per silent rank would
                # put a rank of the attempt's whole width into every cursor.
                advanced[rank] = encode_archive_offset(reached)
        for rank, presented in positions.items():
            if rank in replay.ranks or not presented:
                continue
            # A position for a rank this archive does not hold. Retained rather
            # than dropped: a later settlement can still land that object, and a
            # dropped position would replay it from the first line when it does.
            advanced[rank] = presented if is_archive_position(presented) else encode_unplaced_archive_position(presented)

        detail = replay.detail
        if gaps:
            detail += "; archived positions could not be placed: " + ", ".join(gaps)
        if position_gap:
            detail += f"; {position_gap}"
        return LogPage(
            records=merge_pages(served).records if served else (),
            next_cursor=encode_log_cursor(advanced, context=_archive_cursor_context(workload)),
            # Read from the whole archive, never from this window: a follower that
            # has already been served every line is caught up, not unable to read.
            unavailable=(replay.unavailable or bool(truncated)) and not any(replay.ranks.values()),
            detail=detail,
            truncated_ranks=tuple(sorted(truncated)),
        )

    async def workload_exec(
        self,
        workload_id: str,
        command: tp.Sequence[str],
        *,
        timeout_s: float = 30.0,
        rank: int | None = None,
    ) -> JsonObject:
        """Run one diagnostic command beside every rank of a workload.

        Fan-out is the default because the question exec answers is almost
        always "which rank is stuck", and asking one rank cannot distinguish a
        hung collective from a slow step. ``rank`` narrows it once the answer is
        known.

        One rank failing does not fail the call: a result set missing the rank
        you most wanted to see is still more useful than an exception, and the
        failure is reported in place beside the ranks that answered.
        """

        async with self.repository.transaction() as transaction:
            workload = await transaction.get_workload(workload_id)
        if workload is None:
            raise LookupError(f"workload {workload_id!r} does not exist")
        if workload.spec.secret_mounts:
            return {
                "workload_id": workload.id,
                "attempt": workload.attempt,
                "results": [],
                "output_suppressed": True,
            }
        if workload.remote_id is None:
            raise ConflictError(f"workload {workload_id!r} is not running on any cell")
        try:
            members = await self._load_workload_members(workload)
        except StaleOutboxCommand as exc:
            # A terminal workload keeps its remote_id but loses its lease, so
            # "has a remote id" is not "is still somewhere". Letting the
            # internal staleness error escape turned an ordinary race -- the job
            # finished between the read and the exec -- into a 500.
            raise ConflictError(f"workload {workload_id!r} no longer holds a cell: {exc}") from exc
        ordered = self._ordered_gang_member_pairs(members) if len(members) > 1 else tuple(members)
        selected: tuple[CellRecord, int, int] | None = None
        if rank is not None:
            async with self.repository.transaction() as transaction:
                topology = await transaction.get_workload_topology(workload.id, workload.attempt)
            if topology is None or not topology.contains_rank(rank):
                raise ConflictError(f"workload {workload_id!r} has no frozen rank {rank}")
            current_cells = {cell.id: cell for _lease, cell in ordered}
            for frozen in topology.cells:
                if frozen.global_rank_offset <= rank < frozen.global_rank_offset + frozen.runtime_process_count:
                    cell = current_cells.get(frozen.cell_id)
                    if cell is None or _topology_mismatch(frozen, cell, self._cell_worker_endpoints(cell)):
                        raise ConflictError("runtime topology changed before selected-rank exec")
                    selected = (cell, rank - frozen.global_rank_offset, frozen.runtime_process_count)
                    break
        results: list[JsonObject] = []
        for index, (_lease, cell) in enumerate(ordered):
            if selected is not None and cell.id != selected[0].id:
                continue
            result_rank = rank if rank is not None else index
            kwargs: dict[str, object] = {"timeout_s": timeout_s}
            if selected is not None and selected[2] > 1:
                kwargs["runtime_rank"] = selected[1]
            try:
                payload = await asyncio.to_thread(
                    self._agent_factory(cell).exec_in_workload,
                    cell.id,
                    workload.remote_id,
                    list(command),
                    **kwargs,
                )
            except Exception as exc:
                results.append({"rank": result_rank, "cell_id": cell.id, "error": str(exc)})
                continue
            results.append({**payload, "rank": result_rank, "cell_id": cell.id})
        if rank is not None and not results:
            raise ConflictError(f"workload {workload_id!r} has no rank {rank}")
        document: JsonObject = {
            "workload_id": workload_id,
            "attempt": workload.attempt,
            "results": list(results),
        }
        return document

    async def workload_process_status(self, workload_id: str) -> JsonObject:
        """Read live process detail for one workload attempt.

        For a gang this reports the coordinator cell, whose slice executor
        already fans out across every rank -- which is the view that makes a
        hung collective legible.
        """

        async with self.repository.transaction() as transaction:
            workload = await transaction.get_workload(workload_id)
        if workload is None:
            raise LookupError(f"workload {workload_id!r} does not exist")
        if workload.spec.secret_mounts:
            return {
                "workload_id": workload.id,
                "observed": False,
                "detail": _SECRET_OUTPUT_SUPPRESSED,
                "output_suppressed": True,
            }
        if workload.remote_id is None:
            return {"workload_id": workload_id, "observed": False, "detail": "workload has no dispatched attempt"}
        try:
            members = await self._load_workload_members(workload)
        except StaleOutboxCommand:
            return {"workload_id": workload_id, "observed": False, "detail": "workload no longer holds a cell"}
        lease, cell = self._ordered_gang_member_pairs(members)[0] if len(members) > 1 else members[0]
        return await asyncio.to_thread(
            self._agent_factory(cell).process_status,
            lease.cell_id,
            workload.remote_id,
        )


def _reported_assets(metadata: tp.Mapping[str, object] | None) -> tuple[str, ...]:
    """Read an agent's materialized-asset report out of its observation metadata.

    Carried in the existing free-form metadata rather than as a new wire field,
    so an agent that never learns to report keeps validating and a control plane
    that predates the convention keeps parsing. Locality is a preference either
    way, and a new required field would be a deployment ordering problem in
    exchange for nothing.

    Anything that is not a list of non-empty strings is ignored rather than
    rejected. This runs on the observation hot path, and refusing an otherwise
    good observation over a malformed hint would trade a cache miss for an
    unobservable cell.
    """

    if not isinstance(metadata, dict):
        return ()
    reported = metadata.get("materialized_assets")
    if not isinstance(reported, list):
        return ()
    names = [value for value in reported if isinstance(value, str) and value.strip()]
    # Deduplicated and ordered, so two agents reporting the same set produce the
    # same index and placement does not depend on report ordering.
    return tuple(sorted(dict.fromkeys(name.strip() for name in names)))
