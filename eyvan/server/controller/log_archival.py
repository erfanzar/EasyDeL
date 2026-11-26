# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Incremental log archival: making a run's output outlive its machine."""

from __future__ import annotations

import asyncio
import json

from eyvan._internal.json_types import JsonObject, JsonValue
from eyvan.artifacts.managed import (
    LOG_CHUNK_MAX_SEQUENCE,
    log_chunk_sequence,
    log_chunk_uri,
    log_chunk_uri_prefix,
)
from eyvan.records import WorkloadRecord, WorkloadState
from eyvan.server.controller._support import logger
from eyvan.server.service.errors import ConflictError
from eyvan.worker.runtime.log_records import LOG_CHUNK_HEADER_KEY, LOG_CHUNK_VERSION

#: States whose output is worth capturing before the capacity can vanish.
#: RUNNING only: a workload that has not dispatched has no journal, and one that
#: is terminal is settlement's to archive -- chunking it there would race the
#: terminal writer for the same lines.
_ARCHIVABLE_WORKLOAD_STATES = frozenset({WorkloadState.RUNNING})

#: Rows read per page while recovering a writer's position from storage.
_CHUNK_RECOVERY_PAGE = 256

#: Cover the entire supported sequence space, plus an end-of-prefix probe.
#: A smaller bound would recover an occupied next sequence for valid long runs
#: and silently omit their later chunks on replay. Keep a finite guard against
#: malformed ledgers, but refuse an incomplete scan rather than trust its prefix.
_CHUNK_RECOVERY_MAX_PAGES = LOG_CHUNK_MAX_SEQUENCE // _CHUNK_RECOVERY_PAGE + 2


class _LogChunkProgress:
    """Where one attempt's incremental archive has reached.

    Held in memory on the leader and recovered from storage when it is missing,
    which is what makes a leader change cost one listing rather than a restarted
    archive. Storage is the truth: the cursor lives in the chunk it was captured
    with, so a leader that dies between writing a chunk and updating this object
    loses nothing.
    """

    __slots__ = ("cursor", "exhausted", "sequence")

    def __init__(self, *, sequence: int, cursor: str) -> None:
        self.sequence = sequence
        self.cursor = cursor
        #: Set once the sequence space is spent, so the ceiling is logged once
        #: rather than on every interval for the rest of the run.
        self.exhausted = False


class _ControllerLogArchival:
    """Incremental log archival: making a run's output outlive its machine.

    A concern of :class:`~eyvan.server.controller.EyvanController`; not
    instantiated on its own.

    Settlement archives logs on the release path, which covers every workload
    that *ends*. It cannot cover a workload that is *taken*: a preempted slice
    is gone before the controller learns it existed, so the terminal read finds
    no journal and archives nothing. On a spot-first fleet that is not an edge
    case -- it is the failure operators most need the logs for, and the one
    state in which there were none.

    This phase closes that window by copying the live journal into immutable
    chunks while the workload runs. Everything already written is safe when the
    host disappears; only the lines since the last chunk are lost, bounded by
    the archive interval rather than by the run's length.
    """

    async def archive_running_logs_once(self) -> int:
        """Copy new output from every running workload into a durable chunk.

        Returns:
            How many workloads had a chunk written this pass.
        """

        if self.artifact_access is None:
            # Nowhere to put them. The local development case, where the journal
            # is on the same machine as the reader anyway.
            return 0
        limit = max(int(self.settings.log_archive_chunk_lines), 1)
        async with self.repository.transaction() as transaction:
            candidates = [
                workload
                for workload in await transaction.list_workloads(states=frozenset(_ARCHIVABLE_WORKLOAD_STATES))
                if workload.remote_id is not None
                and workload.lease_id is not None
                # Durable secret-bearing executions intentionally have no
                # archived output, for the reason settlement states: exact
                # streaming redaction cannot be reconstructed later without
                # persisting the plaintext values themselves.
                and not workload.spec.secret_mounts
            ]
        self._forget_stale_log_chunk_progress({(workload.id, workload.attempt) for workload in candidates})
        if not candidates:
            return 0
        gate = asyncio.Semaphore(max(int(self.settings.log_archive_concurrency), 1))

        async def archive(workload: WorkloadRecord) -> bool:
            async with gate, self._log_archive_lock(workload.id, workload.attempt):
                try:
                    return await self._archive_log_chunk(workload, limit=limit)
                except Exception:
                    # One unreachable cell must not stop the fleet's archive.
                    logger.warning("could not archive live logs for %s", workload.id, exc_info=True)
                    return False

        written = await asyncio.gather(*(archive(workload) for workload in candidates))
        return sum(1 for chunk in written if chunk)

    def _forget_stale_log_chunk_progress(self, live: set[tuple[str, int]]) -> None:
        """Drop cached positions for attempts this leader no longer archives.

        Unbounded growth otherwise: every workload the leader ever archived
        would keep a cursor for the process's lifetime. Dropping is free --
        storage holds the same position, so a workload that becomes archivable
        again simply recovers it.
        """

        for key in [key for key in self._log_chunk_progress if key not in live]:
            del self._log_chunk_progress[key]

    async def _archive_log_chunk(self, workload: WorkloadRecord, *, limit: int) -> bool:
        """Write one chunk of one attempt's new output, if it produced any.

        Args:
            workload: The running attempt to capture.
            limit: Most lines to take per rank this pass.

        Returns:
            Whether a chunk was written.
        """

        key = (workload.id, workload.attempt)
        progress = self._log_chunk_progress.get(key)
        if progress is None:
            progress = await self._recover_log_chunk_progress(workload)
            self._log_chunk_progress[key] = progress
        if progress.sequence > LOG_CHUNK_MAX_SEQUENCE:
            if not progress.exhausted:
                progress.exhausted = True
                logger.warning(
                    "log chunk sequence exhausted for %s attempt %d; live output is no longer being archived",
                    workload.id,
                    workload.attempt,
                )
            return False

        page = await self.workload_log_page(workload.id, after=progress.cursor, limit=limit, live_only=True)
        attempt = page.get("attempt")
        if isinstance(attempt, int) and attempt != workload.attempt:
            # The workload retried underneath this pass. The cursor named the
            # previous attempt's journals and this page belongs to a different
            # rank space, so it is not appended to the previous attempt's chunk
            # stream -- the next pass recovers the new attempt's own position.
            del self._log_chunk_progress[key]
            return False
        records = page.get("records")
        if not isinstance(records, list) or not records:
            # No new lines. The cursor may still have advanced past positions
            # that carried none, and keeping that advance is what stops every
            # pass from re-reading the same empty window.
            next_cursor = page.get("next_cursor")
            if isinstance(next_cursor, str) and next_cursor:
                progress.cursor = next_cursor
            return False

        header: JsonObject = {
            "version": LOG_CHUNK_VERSION,
            "sequence": progress.sequence,
            "attempt": workload.attempt,
            # The position *after* these lines, so recovery reads one object to
            # learn where to resume. A cursor kept only in the writer's memory
            # would restart the archive on every leader change.
            "cursor": page.get("next_cursor") if isinstance(page.get("next_cursor"), str) else "",
            "unavailable": bool(page.get("unavailable")),
            "truncated_ranks": [
                rank
                for rank in (page.get("truncated_ranks") if isinstance(page.get("truncated_ranks"), list) else [])
                if isinstance(rank, int) and not isinstance(rank, bool)
            ],
            "detail": page.get("detail") if isinstance(page.get("detail"), str) else "",
        }
        lines: list[JsonValue] = [[LOG_CHUNK_HEADER_KEY, header]]
        lines.extend(record for record in records if isinstance(record, dict))
        payload = "\n".join(json.dumps(line, sort_keys=True) for line in lines).encode()
        uri = log_chunk_uri(workload.id, workload.attempt, progress.sequence)
        try:
            await self.service.upload_managed_artifact(
                org_id=workload.org_id,
                project_id=workload.project_id,
                uri=uri,
                payload=payload,
                actor="system:log-archiver",
            )
        except ConflictError:
            # This sequence already holds different bytes: a previous pass wrote
            # it and this leader's position is behind storage. Storage wins --
            # the cached position is dropped so the next pass recovers the real
            # one rather than conflicting forever at the same sequence.
            logger.warning("log chunk %s already exists with different content; re-reading position", uri)
            del self._log_chunk_progress[key]
            return False
        cursor = header["cursor"]
        progress.cursor = cursor if isinstance(cursor, str) else ""
        progress.sequence += 1
        return True

    async def _last_log_chunk_cursor(self, workload: WorkloadRecord) -> str:
        """Return the live position this attempt's incremental archive reached.

        What settlement resumes from, so the terminal archive holds the lines
        after the last chunk rather than a second copy of every line before it.
        Empty when nothing was chunked, which is the read settlement has always
        done.

        Never raises. A ledger this cannot query is a reason to archive the
        whole journal again -- duplicating lines -- rather than to abandon the
        terminal archive on the one path that still has a live cell to read.
        """

        if self.artifact_access is None:
            return ""
        progress = self._log_chunk_progress.get((workload.id, workload.attempt))
        if progress is not None:
            return progress.cursor
        try:
            return (await self._recover_log_chunk_progress(workload)).cursor
        except Exception:
            logger.warning("could not read the chunk archive position for %s", workload.id, exc_info=True)
            return ""

    async def _list_log_chunks(self, workload: WorkloadRecord) -> list[tuple[int, str]]:
        """Return one attempt's chunk objects, in the order they were written.

        Sequence order, which is also line order: a chunk holds exactly the
        lines read between its predecessor's cursor and its own. The ledger is
        the index -- chunk sequences are zero-padded, so its URI ordering is
        numeric ordering and no separate list has to be kept in step with the
        objects it names.
        """

        prefix = log_chunk_uri_prefix(workload.id, workload.attempt)
        after = prefix
        found: dict[int, str] = {}
        pages = 0
        async with self.repository.transaction() as transaction:
            while pages < _CHUNK_RECOVERY_MAX_PAGES:
                pages += 1
                rows = await transaction.list_managed_artifacts_page(
                    org_id=workload.org_id,
                    project_id=workload.project_id,
                    after_uri=after,
                    limit=_CHUNK_RECOVERY_PAGE,
                )
                if not rows:
                    break
                matching = [row for row in rows if row.uri.startswith(prefix)]
                for row in matching:
                    sequence = log_chunk_sequence(row.uri, workload.id, workload.attempt)
                    if sequence is not None:
                        found[sequence] = row.uri
                if len(matching) < len(rows):
                    # A row outside the prefix. The ledger is URI-ordered, so
                    # every chunk of this attempt has already been seen.
                    break
                after = rows[-1].uri
            else:
                # Neither a writer nor a reader may treat a bounded prefix as
                # complete: the next sequence may already exist, and replay
                # would otherwise claim that omitted lines never existed.
                raise RuntimeError(
                    f"chunk archive scan incomplete for {workload.id} attempt {workload.attempt} after {pages} pages"
                )
        return [(sequence, found[sequence]) for sequence in sorted(found)]

    async def _recover_log_chunk_progress(self, workload: WorkloadRecord) -> _LogChunkProgress:
        """Read one attempt's last chunk back to learn where to resume.

        Chunk sequences are zero-padded, so the ledger's URI order is their
        numeric order and the last matching row is the last chunk. Reading its
        header recovers the exact live position it stopped at, which is what
        makes a leader change cost one listing instead of a replayed archive.
        """

        chunks = await self._list_log_chunks(workload)
        if not chunks:
            return _LogChunkProgress(sequence=0, cursor="")
        last_sequence, last_uri = chunks[-1]
        cursor = ""
        try:
            payload = await self.service.read_managed_artifact(
                org_id=workload.org_id,
                project_id=workload.project_id,
                uri=last_uri,
            )
        except Exception:
            # Unreadable last chunk. Resume after it with an empty cursor rather
            # than overwriting it: an empty cursor re-reads the journal from its
            # start, which duplicates lines already archived, while reusing the
            # sequence would conflict against an object that exists. Duplicated
            # lines are recoverable; a refused write is not.
            logger.warning("could not read the last log chunk %s to resume from it", last_uri, exc_info=True)
        else:
            cursor = _chunk_header_cursor(payload)
        return _LogChunkProgress(sequence=last_sequence + 1, cursor=cursor)


def _chunk_header_cursor(payload: bytes) -> str:
    """Return the live position one chunk stopped at, or empty if it names none."""

    for line in payload.decode(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            return ""
        if (
            isinstance(entry, list)
            and len(entry) == 2
            and entry[0] == LOG_CHUNK_HEADER_KEY
            and isinstance(entry[1], dict)
        ):
            cursor = entry[1].get("cursor")
            return cursor if isinstance(cursor, str) else ""
        # The header is the first line by construction. Anything else in that
        # position means this object was not written by this writer.
        return ""
    return ""


__all__ = ("_ControllerLogArchival",)
