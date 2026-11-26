# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Structured logs, self-reported progress, exec gating, and profiling.

A training loop's own notion of progress -- "step 4200 of 50000" -- is neither a
result nor a log line, and had nowhere to go. Three design choices make this
channel cheap enough to keep, and each is asserted here:

**Latest per rank, no history.** Upserted on ``(workload, rank)``, so the table is
bounded by live ranks rather than by how chatty user code is. Unbounded
user-metric history inside the control-plane database would be the fastest-growing
table in the system, written by code Eyvan does not control.

**A stale attempt cannot report.** Attempt 1 finishing its last step after attempt
2 started would otherwise walk the displayed progress backwards, which reads as a
job that lost work.

**No new polling loop.** ``GET /workloads/{id}/progress`` already exists and is
already polled by the dashboard and ``client/callables.py``; user progress rides
on it rather than adding a second thing to poll.
"""

from __future__ import annotations

import pytest

from eyvan.capacity.drivers.base import CellCapabilities
from eyvan.records import (
    CommandExecution,
    Principal,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    UserProgressReport,
    WorkloadDefinition,
    WorkloadRecord,
)
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.worker.runtime.log_records import (
    LogPage,
    LogRecord,
    merge_pages,
    parse_journal_json,
    stream_for_priority,
)
from eyvan.worker.runtime.registry import ExecCapableExecutionDriver, IntrospectableExecutionDriver


def _principal() -> Principal:
    return Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )


async def _service_with_workload(attempt: int = 1) -> tuple[ControlPlaneService, WorkloadRecord]:
    repository = InMemoryRepository()
    await repository.open()
    service = ControlPlaneService(repository=repository)
    workload = WorkloadRecord(
        id="wl-1",
        org_id="acme",
        project_id="research",
        created_by="alice",
        attempt=attempt,
        spec=WorkloadDefinition(
            name="train",
            execution=CommandExecution(kind="command", command="python train.py"),
            gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
        ),
    )
    async with repository.transaction() as tx:
        await tx.put_workload(workload)
    return service, workload


# --- 4a: structured log records ---------------------------------------------


def test_an_unrecognized_priority_is_unknown_not_guessed_as_stdout() -> None:
    """An operator debugging a crash must not read stderr as ordinary output."""

    assert stream_for_priority("3") == "stderr"
    assert stream_for_priority("6") == "stdout"
    assert stream_for_priority(None) == "unknown"
    assert stream_for_priority("banana") == "unknown"
    assert stream_for_priority("99") == "unknown"


def test_journal_json_is_parsed_into_records_with_cursors() -> None:
    payload = (
        '{"MESSAGE": "starting", "PRIORITY": "6", "__CURSOR": "c1", "__REALTIME_TIMESTAMP": "1700000000000000"}\n'
        '{"MESSAGE": "boom", "PRIORITY": "3", "__CURSOR": "c2", "_PID": "42"}\n'
    )

    page = parse_journal_json(payload, rank=2)

    assert [record.message for record in page.records] == ["starting", "boom"]
    assert [record.stream for record in page.records] == ["stdout", "stderr"]
    assert page.next_cursor == "c2"
    assert page.records[1].pid == 42
    assert all(record.rank == 2 for record in page.records)


def test_a_truncated_final_line_does_not_discard_the_page() -> None:
    """Live reads end mid-line exactly when a job is producing the most output."""

    payload = '{"MESSAGE": "one", "PRIORITY": "6", "__CURSOR": "c1"}\n{"MESSAGE": "tw'

    page = parse_journal_json(payload)

    assert [record.message for record in page.records] == ["one"]


def test_a_non_utf8_message_is_replaced_rather_than_dropped() -> None:
    """journald returns a byte array for those; losing the line loses the crash."""

    payload = '{"MESSAGE": [104, 105, 255], "PRIORITY": "6", "__CURSOR": "c1"}'

    page = parse_journal_json(payload)

    assert page.records[0].message.startswith("hi")


def test_an_unreachable_rank_is_reported_in_place_not_dropped() -> None:
    """A complete-looking log for a partially-failed slice is the failure to avoid."""

    merged = merge_pages(
        {
            0: LogPage(records=(LogRecord(message="rank zero", rank=0),)),
            1: LogPage(unavailable=True, detail="host unreachable"),
        }
    )

    assert [record.message for record in merged.records] == ["rank zero"]
    assert merged.truncated_ranks == (1,)
    assert "rank 1" in merged.detail
    assert merged.unavailable is False, "some output was read; the page is partial, not empty"


def test_every_rank_failing_reports_unavailable_rather_than_empty() -> None:
    """Zero lines read and unable to read are different answers."""

    merged = merge_pages({0: LogPage(unavailable=True, detail="down"), 1: LogPage(unavailable=True)})

    assert merged.unavailable is True
    assert merged.records == ()


# --- 4c: progress reports ---------------------------------------------------


@pytest.mark.asyncio
async def test_a_report_is_stored_and_returned_per_rank() -> None:
    service, _workload = await _service_with_workload()

    await service.record_progress_report(workload_id="wl-1", rank=0, attempt=1, step=10, total=100)
    await service.record_progress_report(workload_id="wl-1", rank=1, attempt=1, step=12, total=100)

    reports = await service.list_progress_reports(_principal(), "wl-1")
    assert [report.rank for report in reports] == [0, 1]
    assert [report.step for report in reports] == [10, 12]


@pytest.mark.asyncio
async def test_a_second_report_replaces_the_first_rather_than_accumulating() -> None:
    """The table is bounded by live ranks, not by how chatty user code is."""

    service, _workload = await _service_with_workload()

    for step in (10, 20, 30):
        await service.record_progress_report(workload_id="wl-1", rank=0, attempt=1, step=step, total=100)

    reports = await service.list_progress_reports(_principal(), "wl-1")
    assert len(reports) == 1
    assert reports[0].step == 30


@pytest.mark.asyncio
async def test_a_stale_attempt_cannot_walk_progress_backwards() -> None:
    """Attempt 1's last step arriving after attempt 2 started reads as lost work."""

    service, _workload = await _service_with_workload(attempt=2)
    await service.record_progress_report(workload_id="wl-1", rank=0, attempt=2, step=5, total=100)

    await service.record_progress_report(workload_id="wl-1", rank=0, attempt=1, step=99, total=100)

    reports = await service.list_progress_reports(_principal(), "wl-1")
    assert [report.step for report in reports] == [5]


@pytest.mark.asyncio
async def test_reports_from_a_superseded_attempt_are_withheld_on_read() -> None:
    service, workload = await _service_with_workload(attempt=1)
    await service.record_progress_report(workload_id="wl-1", rank=0, attempt=1, step=42, total=100)
    async with service.repository.transaction() as tx:
        await tx.put_workload(workload.model_copy(update={"attempt": 2}))

    assert await service.list_progress_reports(_principal(), "wl-1") == []


@pytest.mark.asyncio
async def test_user_progress_rides_the_route_that_is_already_polled() -> None:
    """No new polling loop: the dashboard and callables client already poll this."""

    from eyvan.server.service.workload_progress import WorkloadProgressResolver

    service, _workload = await _service_with_workload()
    await service.record_progress_report(workload_id="wl-1", rank=0, attempt=1, step=7, total=10, message="warming up")

    progress = await WorkloadProgressResolver(service, service.repository).resolve(_principal(), "wl-1")

    assert [report.step for report in progress.user_progress] == [7]
    assert progress.user_progress[0].message == "warming up"


def test_a_step_without_a_total_reports_no_fraction() -> None:
    """Rendering that as 0% is worse than rendering nothing."""

    assert UserProgressReport(rank=0, attempt=1, step=5).fraction is None
    assert UserProgressReport(rank=0, attempt=1, step=5, total=10).fraction == 0.5
    # A declared total of zero is "I do not know how many steps there are", not
    # "I am finished". (step=5, total=0 is refused outright by the validator.)
    assert UserProgressReport(rank=0, attempt=1, step=0, total=0).fraction is None


def test_a_step_beyond_its_total_is_refused() -> None:
    with pytest.raises(ValueError, match="exceed"):
        UserProgressReport(rank=0, attempt=1, step=11, total=10)


def test_a_fraction_is_clamped_rather_than_exceeding_one() -> None:
    report = UserProgressReport(rank=0, attempt=1, step=10, total=10)

    assert report.fraction == 1.0


# --- 4d: the supports_exec split --------------------------------------------


def test_exec_permission_now_defaults_to_denied() -> None:
    """A default-on capability nothing read is how it became dead."""

    capabilities = CellCapabilities(
        provider="manual", executor="eyvan-native", accelerator_types=(), capacity_classes=()
    )

    assert capabilities.supports_exec is False


def test_the_two_exec_gates_answer_different_questions() -> None:
    """ "May anyone do this here" is not "can this backend do it at all"."""

    class _CanExec:
        def exec_in_workload(self, remote_id, command, *, timeout_s):
            del remote_id, command, timeout_s
            return {}

    class _CannotExec:
        pass

    assert isinstance(_CanExec(), ExecCapableExecutionDriver)
    assert not isinstance(_CannotExec(), ExecCapableExecutionDriver)


def test_introspection_is_a_protocol_not_a_getattr_probe() -> None:
    """The registry already establishes this seam for close and health."""

    class _Introspectable:
        def process_status(self, remote_id):
            del remote_id
            return {}

    class _Opaque:
        pass

    assert isinstance(_Introspectable(), IntrospectableExecutionDriver)
    assert not isinstance(_Opaque(), IntrospectableExecutionDriver)


# --- 4e: profiling ----------------------------------------------------------


def test_each_rank_gets_its_own_profile_destination() -> None:
    """Same reason ranks get their own checkpoint URI: writers would race."""

    from eyvan.worker.runtime.systemd_transient import _rank_profile_uri

    assert _rank_profile_uri("gs://acme/run", 0) == "gs://acme/run/profiles/rank-0"
    assert _rank_profile_uri("gs://acme/run", 3) == "gs://acme/run/profiles/rank-3"


def test_profiles_are_written_beside_checkpoints_not_among_them() -> None:
    """A resume that read trace files as state would fail in a confusing way."""

    from eyvan.worker.runtime.systemd_transient import _rank_checkpoint_uri, _rank_profile_uri

    checkpoint = _rank_checkpoint_uri("gs://acme/run", 0)
    profile = _rank_profile_uri("gs://acme/run", 0)

    assert not profile.startswith(checkpoint)
    assert checkpoint != profile


def test_no_output_uri_means_no_profile_destination() -> None:
    """Empty, not a fabricated path: there is nowhere durable to write."""

    from eyvan.worker.runtime.systemd_transient import _rank_profile_uri

    assert _rank_profile_uri(None, 0) == ""
    assert _rank_profile_uri("", 0) == ""
