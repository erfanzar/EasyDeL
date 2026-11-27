# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Coordinated native execution across every host in one TPU slice group.

A slice is *one cell with several machines*. Every other layer treats it as a
single unit -- one fence, one lease, one stage row -- while the bytes and the
processes live on `n` hosts, and this module is where the one and the `n` meet.

Managed staging is the second thing to cross that boundary, after execution
itself. W4 and W5 shipped an executor that puts an asset on a disk and a route
that reaches the worker owning it, but the agent could only bind a stage driver
when a cell had exactly *one* worker endpoint. A cell with more resolved to
:class:`NativeSliceExecutor`, which did not satisfy ``AssetStageDriver``, so it
read as *no staging executor*: correct, honest, and the reason every real TPU
slice -- a ``v5p-16`` is already two hosts -- staged nothing at all.

:class:`NativeSliceStageDriver` is the missing half. It fans one action to every
host and answers once, and the whole of its difficulty is in refusing to
average:

**Partial success is not success.** A stage row says *this cell holds this
asset*, and placement is steered by it. Three hosts of four holding a dataset is
a cell that does not hold it -- rank 3 opens nothing -- so the aggregate is
unproven, exactly as a truncated single-host fetch is. Unproven still charges
the budget, because the bytes that did land are on real disks.

**Disagreement is a refusal, not a vote.** Two hosts that hash the same asset
name to different bytes are not a cell holding one asset; they are two objects
sharing a name, and a collective that reads them will produce silently wrong
numbers. The first answer to arrive is not the truth, and neither is the average
of the sizes.

**Eviction is unanimous or unproven.** A host that could not be reached may
still hold the bytes, so the row stays charged and stays present rather than
being deleted on the strength of the hosts that did answer.

**Fenced around the fan-out.** The agent checks the cell's generation before
any host is contacted, and for the two actions that touch a disk the driver
reads the fence again before dispatching every host at once and once more after
they have all answered. A fan-out that begins under a generation this agent no
longer owns is not dispatched at all, and one whose fence moves mid-flight
reports the bytes its hosts already wrote as unproven rather than as a hit --
the machines may belong to somebody else's plan now.

Hosts are asked in parallel, so a wide stage costs its slowest host rather than
the sum of them. A fetch that outlives the control plane's patience is still
reported as unproven -- never as a false hit.
"""

from __future__ import annotations

import concurrent.futures
import copy
import logging
import threading
import typing as tp
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass, replace
from time import monotonic
from urllib.parse import urlparse

import httpx
from pydantic import ValidationError

from eyvan._internal.compat import StrEnum
from eyvan._internal.json_types import JsonObject, JsonValue
from eyvan.capacity.drivers.base import (
    ExecutionWorkloadObservation,
    ExecutionWorkloadResult,
    ExecutionWorkloadStatus,
    MutationContext,
    UnsupportedCapabilityError,
    rollback_context,
)
from eyvan.identity.mtls import MutualTlsFiles, client_ssl_context
from eyvan.worker.distributed_environment import chip_host_of, without_distributed_topology_overrides
from eyvan.worker.runtime._multihost_site import (
    CROSS_REGION_TRANSFER_BANNED_EXIT_CODE,
    CROSS_REGION_TRANSFER_BANNED_MARKER,
    CROSS_REGION_TRANSFER_BANNED_REASON,
    DEFAULT_GANG_INITIALIZATION,
    GangInitializationPolicy,
    cross_region_ban_failure_reason,
    extract_cross_region_ban_marker,
    gang_formation_failure_reason,
    gang_initialization_outcome,
)
from eyvan.worker.runtime.log_records import (
    CURSOR_KIND_WORKLOAD_LOG,
    CURSOR_MODE_LIVE,
    CursorContext,
    CursorFresh,
    CursorResume,
    LogPage,
    decode_log_cursor,
    join_log_cursor,
    merge_pages,
)
from eyvan.worker.runtime.native_remote import _EXEC_HTTP_TIMEOUT_MARGIN_S, NativeRemoteExecutor
from eyvan.worker.runtime.stage_executor import AssetStageDriver, AssetStageResult, AssetStageVerdict
from eyvan.worker.runtime.systemd_transient import _rank_checkpoint_uri

#: Where a resuming workload is told to read its checkpoint from. Must be
#: derived per rank exactly as the hook derives its write destination.
_RESUME_CHECKPOINT_ENV = "EYVAN_RESUME_CHECKPOINT_URI"

_MULTIHOST_SITE_PATH = "/opt/eyvan-native/multihost"
_COORDINATOR_PORT = 12345
_RETIRE_SUMMARY_LIMIT = 4
_FAILED_STATUSES = frozenset({ExecutionWorkloadStatus.FAILED})
_LOST_STATUSES = frozenset({ExecutionWorkloadStatus.LOST, ExecutionWorkloadStatus.NOT_FOUND})
_RUNNING_STATUSES = frozenset(
    {
        ExecutionWorkloadStatus.PENDING,
        ExecutionWorkloadStatus.SUBMITTED,
        ExecutionWorkloadStatus.RUNNING,
        ExecutionWorkloadStatus.STOPPING,
        ExecutionWorkloadStatus.CHECKPOINTING,
    }
)
_MutationAction: tp.TypeAlias = tp.Literal["stop", "retire"]

#: How many hosts one aggregated staging detail spells out before it is elided.
#:
#: The same bound the retire summary uses, and for the same reason: this string
#: is persisted in a stage row whose detail column is truncated, and a slice can
#: be hundreds of hosts wide. Four names are enough to tell an operator which
#: rank to go and look at.
_STAGE_SUMMARY_LIMIT = 4

#: Which of the three staging operations one fan-out is performing.
_StageAction: tp.TypeAlias = tp.Literal["measure", "stage", "evict"]

#: Re-reads the generation that currently owns the cell being staged.
#:
#: Supplied by whoever already fenced the request -- the cell agent -- because
#: this module holds no capacity record of its own. Raising means the fence
#: could not be established, which is never read as "it still holds".
StageFence: tp.TypeAlias = tp.Callable[[], int]

logger = logging.getLogger(__name__)

# A wide cell must not create one OS thread per rank. Queued commands share
# the original deadline, rather than starting a fresh budget in each wave.
_EXEC_CONCURRENCY_LIMIT = 16

_T = tp.TypeVar("_T")


def _each_host(
    workers: tp.Sequence[NativeRemoteExecutor],
    read: tp.Callable[[int, NativeRemoteExecutor], _T],
) -> list[_T]:
    """Run one read-only call on every host at once and return the answers in rank order.

    A slice is asked about by the controller's observation pass, which every
    other workload waits behind, so a host-at-a-time walk made each poll cost
    the *sum* of its hosts -- linear in slice width. Concurrently it costs the
    slowest host, bounded by the same thread ceiling ``exec`` uses. Only reads
    go through here: mutations keep their sequential, fence-checked walks.
    """

    if len(workers) <= 1:
        return [read(rank, worker) for rank, worker in enumerate(workers)]
    with ThreadPoolExecutor(
        max_workers=min(len(workers), _EXEC_CONCURRENCY_LIMIT), thread_name_prefix="eyvan-read"
    ) as pool:
        futures = [pool.submit(copy_context().run, read, rank, worker) for rank, worker in enumerate(workers)]
        return [future.result() for future in futures]


def _exec_fanout(
    workers: tp.Sequence[tuple[int, str, NativeRemoteExecutor]],
    remote_id: str,
    command: tp.Sequence[str],
    *,
    timeout_s: float,
) -> list[JsonValue]:
    """Run bounded diagnostic requests with one command deadline and rank order.

    NativeRemoteExecutor's HTTP timeout is the remaining command budget plus
    its existing 30-second response margin, matching the outer agent hop. On
    cancellation, stop queued dispatch and drain in-flight bounded HTTP calls:
    Python threads cannot safely be killed, and abandoning them leaks work.
    """

    cancelled = threading.Event()
    deadline = monotonic() + timeout_s

    def run(rank: int, endpoint: str, worker: NativeRemoteExecutor, *, single: bool = False) -> JsonObject:
        remaining = timeout_s if single else deadline - monotonic()
        if cancelled.is_set() or remaining <= 0:
            result: JsonValue = {"observed": False, "detail": "exec deadline expired or dispatch cancelled"}
        else:
            try:
                result = worker.exec_in_workload(remote_id, list(command), timeout_s=remaining)
                if not single and monotonic() > deadline + _EXEC_HTTP_TIMEOUT_MARGIN_S:
                    result = {"observed": False, "detail": "exec response arrived after the shared deadline"}
            except httpx.TransportError as exc:
                result = {"observed": False, "detail": f"native worker is unreachable: {type(exc).__name__}: {exc}"}
            except (httpx.HTTPStatusError, RuntimeError, ValueError) as exc:
                result = {"observed": False, "detail": f"exec failed: {type(exc).__name__}: {exc}"}
            except BaseException:
                cancelled.set()
                raise
        return {"rank": rank, "endpoint": endpoint, "result": result}

    if len(workers) == 1:
        # Keep an exact-rank/one-host command's full budget and caller thread.
        return [run(*workers[0], single=True)]
    results: list[JsonValue] = [None] * len(workers)
    pool = ThreadPoolExecutor(max_workers=min(len(workers), _EXEC_CONCURRENCY_LIMIT), thread_name_prefix="eyvan-exec")
    futures: dict[concurrent.futures.Future[JsonObject], int] = {}
    try:
        for index, worker in enumerate(workers):
            futures[pool.submit(copy_context().run, run, *worker)] = index
        for future in concurrent.futures.as_completed(futures):
            results[futures[future]] = future.result()
    except BaseException:
        cancelled.set()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return results


class JaxClusterDetectionMode(StrEnum):
    """Supported JAX discovery contracts for one native TPU allocation."""

    EXPLICIT = "explicit"
    GCE_TPU = "gcetpu"


class NativeSliceWorkerResult(tp.TypedDict):
    """Wire-compatible observation for one host in a native TPU slice."""

    rank: int
    endpoint: str
    remote_id: str
    status: ExecutionWorkloadStatus
    message: str


class NativeSliceWorkloadResult(ExecutionWorkloadResult):
    """Aggregate slice status plus the typed observation from every rank."""

    workers: list[NativeSliceWorkerResult]


@dataclass(frozen=True, slots=True)
class _SliceWorkerObservation:
    """One typed worker observation together with its stable rank metadata."""

    rank: int
    endpoint: str
    observation: ExecutionWorkloadObservation

    def as_result(self) -> NativeSliceWorkerResult:
        """Return the worker detail in the established mapping wire shape."""

        return {
            "rank": self.rank,
            "endpoint": self.endpoint,
            "remote_id": self.observation.remote_id,
            "status": self.observation.status,
            "message": self.observation.message,
        }


def _indeterminate_observation(remote_id: str, reason: str) -> ExecutionWorkloadObservation:
    """Represent an explicit failure to obtain a definitive worker observation."""

    return ExecutionWorkloadObservation(
        remote_id=remote_id,
        status=ExecutionWorkloadStatus.INDETERMINATE,
        message=reason,
    )


def _invalid_observation_reason(
    error: Exception,
    *,
    endpoint: str,
) -> str:
    """Keep a validated worker message when Pydantic rejects its observation."""

    if isinstance(error, ValidationError):
        for detail in error.errors():
            if "input" not in detail:
                continue
            payload = detail["input"]
            if not isinstance(payload, dict) or "message" not in payload:
                continue
            message = payload["message"]
            if isinstance(message, str) and message.strip():
                return f"native worker {endpoint} returned an invalid observation: {message}"
    return f"native worker {endpoint} returned an invalid observation: {type(error).__name__}: {error}"


def _normalize_worker_observation(
    observation: ExecutionWorkloadObservation,
    *,
    expected_remote_id: str,
    endpoint: str,
) -> ExecutionWorkloadObservation:
    """Require one typed observation to describe the requested remote workload."""

    if observation.remote_id != expected_remote_id:
        return _indeterminate_observation(
            expected_remote_id,
            f"native worker {endpoint} returned a mismatched remote_id",
        )
    if observation.status is ExecutionWorkloadStatus.INDETERMINATE and not observation.message:
        return _indeterminate_observation(
            expected_remote_id,
            f"native worker {endpoint} reported an indeterminate status without a reason",
        )
    return observation


def _aggregate_status(workers: tp.Sequence[_SliceWorkerObservation]) -> ExecutionWorkloadStatus:
    """Reduce typed rank observations without manufacturing a default status."""

    states = tuple(worker.observation.status for worker in workers)
    if not states:
        raise RuntimeError("native slice has no workers to observe")
    if all(state is ExecutionWorkloadStatus.SUCCEEDED for state in states):
        return ExecutionWorkloadStatus.SUCCEEDED
    if all(state is ExecutionWorkloadStatus.NOT_FOUND for state in states):
        return ExecutionWorkloadStatus.NOT_FOUND
    if all(state is ExecutionWorkloadStatus.STOPPED for state in states):
        return ExecutionWorkloadStatus.STOPPED
    if any(state in _FAILED_STATUSES for state in states):
        return ExecutionWorkloadStatus.FAILED
    if any(state is ExecutionWorkloadStatus.INDETERMINATE for state in states):
        return ExecutionWorkloadStatus.INDETERMINATE
    if any(state in _RUNNING_STATUSES for state in states) and all(
        state in _RUNNING_STATUSES | {ExecutionWorkloadStatus.SUCCEEDED} for state in states
    ):
        # Hosts can reach a terminal systemd state a few seconds apart. A
        # completed rank alongside otherwise active ranks is still converging.
        return ExecutionWorkloadStatus.RUNNING
    if any(
        state in _LOST_STATUSES | {ExecutionWorkloadStatus.STOPPED, ExecutionWorkloadStatus.SUCCEEDED}
        for state in states
    ):
        # A vanished or stopped rank prevents distributed JAX progress even
        # while peers remain active.
        return ExecutionWorkloadStatus.LOST
    if any(state in _RUNNING_STATUSES for state in states):
        return ExecutionWorkloadStatus.RUNNING
    raise RuntimeError(f"native slice has no aggregate rule for worker states {states!r}")


def _aggregate_message(workers: tp.Sequence[_SliceWorkerObservation]) -> str:
    """Render each explicit rank observation, preserving failure reasons."""

    return "; ".join(
        f"rank {worker.rank}={worker.observation.status.value}"
        + (f": {worker.observation.message}" if worker.observation.message else "")
        for worker in workers
    )


def _bounded_retire_summary(failures: tp.Sequence[str]) -> str:
    """Render a bounded per-rank retire failure summary."""

    summary = "; ".join(failures[:_RETIRE_SUMMARY_LIMIT])
    remaining = len(failures) - _RETIRE_SUMMARY_LIMIT
    if remaining > 0:
        summary += f"; +{remaining} more"
    return summary


def _augment_error_detail(error: BaseException, detail: str) -> None:
    """Append cleanup detail to an error without changing its type."""

    if error.args and isinstance(error.args[0], str):
        error.args = (f"{error.args[0]}; {detail}", *error.args[1:])
    else:
        error.args = (*error.args, detail)


def _reported_exit_status(detail: JsonObject) -> int | None:
    """Return one host's reported process exit status, or ``None`` if unknown.

    ``None`` covers three different things -- the host was not observed, the
    unit is gone, the payload predates the field -- and all three mean the same
    here: this rank offers no evidence either way, which is not the same as
    evidence that it exited cleanly.
    """

    if detail.get("observed") is not True:
        return None
    status = detail.get("exec_main_status")
    return status if isinstance(status, int) and not isinstance(status, bool) else None


@dataclass(frozen=True, slots=True)
class _HostStageAnswer:
    """What one host of the slice answered about one asset."""

    rank: int
    endpoint: str
    result: AssetStageResult

    def describe(self, render: tp.Callable[[AssetStageResult], str] | None = None) -> str:
        """Render this host's answer for an aggregated detail line.

        ``render`` names the value being compared instead of the verdict, which
        is what a disagreement has to say. "rank 2 = ok" tells an operator
        nothing about which machine to go and look at; the digest it answered
        with does.
        """

        if render is not None:
            return f"rank {self.rank} ({self.endpoint})={render(self.result)}"
        rendered = f"rank {self.rank} ({self.endpoint})={self.result.verdict.value}"
        detail = self.result.detail.strip()
        return f"{rendered}: {detail}" if detail else rendered


def _observed_digest(result: AssetStageResult) -> str:
    """Name the digest one host proved, or say that it proved none."""

    return result.digest or "no digest"


def _observed_size(result: AssetStageResult) -> str:
    """Name the number of bytes one host reported."""

    return f"{result.size_bytes} bytes"


def _stage_summary(
    answers: tp.Sequence[_HostStageAnswer],
    render: tp.Callable[[AssetStageResult], str] | None = None,
) -> str:
    """Render a bounded per-host summary of one staging fan-out."""

    rendered = [answer.describe(render) for answer in answers[:_STAGE_SUMMARY_LIMIT]]
    remaining = len(answers) - _STAGE_SUMMARY_LIMIT
    if remaining > 0:
        rendered.append(f"+{remaining} more")
    return "; ".join(rendered)


def _sole_value(values: tp.Sequence[object]) -> tuple[bool, object]:
    """Return whether every host reported the same thing, and what it was.

    ``(False, None)`` for a disagreement, which every caller treats as a fact
    about the *cell* rather than as a value to pick a winner from.
    """

    distinct = set(values)
    if len(distinct) == 1:
        return True, next(iter(distinct))
    return False, None


@dataclass(frozen=True, slots=True)
class _SliceStageFanout:
    """Every host's answer to one staging action, and whether all were asked."""

    hosts: int
    answers: tuple[_HostStageAnswer, ...]
    #: Why these answers do not stand for the whole cell: hosts that were
    #: never contacted, or a fence that moved after every host had answered.
    #: Empty when the answers cover the cell.
    stopped: str = ""

    @property
    def complete(self) -> bool:
        """Return whether every host in this cell answered."""

        return not self.stopped and len(self.answers) == self.hosts

    def matching(self, verdict: AssetStageVerdict) -> tuple[_HostStageAnswer, ...]:
        """Return the hosts that answered with one verdict."""

        return tuple(answer for answer in self.answers if answer.result.verdict is verdict)

    def unreached(self) -> AssetStageResult:
        """Report a fan-out whose answers prove nothing about the cell.

        Never a refusal and never a success. The hosts that were asked may hold
        bytes and the ones that were not may hold anything at all, so the only
        honest aggregate is that presence on this cell is unknown. A fan-out
        whose every host answered -- the post-dispatch fence move -- reads the
        same way: its answers describe a generation the cell has already moved
        past, not a cell that holds or lacks the asset.
        """

        if len(self.answers) == self.hosts:
            # Every host was asked and answered, and the stop came after. "Only
            # 4 of 4 hosts were asked" would contradict the record an operator
            # is reading to decide what happened to their bytes.
            detail = f"all {self.hosts} hosts answered before the stop: {self.stopped}"
        else:
            detail = f"only {len(self.answers)} of {self.hosts} hosts were asked: {self.stopped}"
        summary = _stage_summary(self.answers)
        return AssetStageResult(
            verdict=AssetStageVerdict.INCONCLUSIVE,
            detail=f"{detail}; {summary}" if summary else detail,
        )


class NativeSliceStageDriver:
    """Materialize, measure and remove one named asset on every host of one cell.

    Satisfies :class:`~eyvan.worker.runtime.stage_executor.AssetStageDriver`, so
    an agent forwards to it exactly as it forwards to the single-worker route:
    the cell is the unit the control plane records, and this is what makes that
    record true of a cell with more than one machine behind it.

    Bound to the generation the caller fenced against, and constructed per
    action rather than cached. :class:`NativeSliceExecutor` is cached by the
    agent's resolver across generations -- the cache key is the endpoint set and
    the worker credential, not the fence -- so a staging view that outlived one
    generation would be the way a plan for a replaced machine reaches the
    machine that replaced it.

    Holds no transport of its own. The per-host executors belong to the slice
    executor that produced this view, and closing them here would shut the
    execution path for the workload as a side effect of a cache operation.
    """

    def __init__(
        self,
        workers: tp.Sequence[NativeRemoteExecutor],
        endpoints: tp.Sequence[str],
        *,
        generation: int,
        fence: StageFence | None = None,
    ) -> None:
        """Bind one staging view to the hosts and the generation it may act for.

        Args:
            workers: One native-worker driver per host, in rank order.
            endpoints: The matching endpoints, used only to name a host in a
                result an operator will read.
            generation: The cell generation this view is allowed to act for.
            fence: Re-reads the generation that owns the cell now, once before
                the hosts are dispatched and once after they have all answered.
                ``None`` disables the re-check, which is what a caller that
                cannot cheaply re-read its own record must ask for explicitly.

        Raises:
            ValueError: There are no hosts, or the endpoints do not match them.
        """

        if len(workers) != len(endpoints):
            raise ValueError("native slice staging needs one endpoint per worker")
        if not workers:
            raise ValueError("native slice staging requires at least one host")
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise ValueError("native slice staging generation must be a non-negative integer")
        self._workers = tuple(workers)
        self._endpoints = tuple(endpoints)
        self._generation = generation
        self._fence = fence

    def measure(self, cell_id: str, asset: str, *, max_bytes: int) -> AssetStageResult:
        """Return what this asset would occupy on this cell, fetching nothing."""

        return self._reduce_measure(self._fanout("measure", cell_id, asset, max_bytes=max_bytes, mutating=False))

    def stage(self, cell_id: str, asset: str, *, max_bytes: int) -> AssetStageResult:
        """Materialize the asset on every host and report what the cell can prove."""

        return self._reduce_stage(self._fanout("stage", cell_id, asset, max_bytes=max_bytes, mutating=True))

    def evict(self, cell_id: str, asset: str) -> AssetStageResult:
        """Remove the asset from every host, or leave the stage unproven."""

        return self._reduce_evict(self._fanout("evict", cell_id, asset, max_bytes=0, mutating=True))

    def _fanout(
        self,
        action: _StageAction,
        cell_id: str,
        asset: str,
        *,
        max_bytes: int,
        mutating: bool,
    ) -> _SliceStageFanout:
        """Apply one staging action to every host, in rank order.

        ``max_bytes`` is handed to each host unchanged rather than divided among
        them. A slice holds one copy of the asset per machine and the ledger
        charges one copy per cell, so the bound the control plane computed is
        the bound each host must honour; dividing it would refuse an asset that
        fits every disk it is going to land on.

        ``mutating`` separates the two actions that touch a disk from the one
        that only asks, and it decides two things.

        For a mutating action the fence is read once before the hosts are
        dispatched and re-read once after they have all answered. The caller
        checked it immediately before resolving this view, and the hosts are
        asked at once, so there is no between-hosts moment left to re-check at:
        the two reads bracket the one fan-out. Measuring skips the fence
        entirely, because there is no mutation for a moved fence to protect a
        machine from.

        A host with no staging route is absence of a capability -- *if nothing
        has happened yet*. Every host is attempted at once, so once the
        answers show that a mutating action reached a disk, the same answer is
        no longer "nothing was attempted" -- reporting it that way would put an
        untouched row back for a cell that has already been partly emptied --
        and the fan-out reads as unproven, exactly as an unreachable host does.

        Raises:
            UnsupportedCapabilityError: A host has no staging route and nothing
                had been done to any disk. The cell cannot hold an asset one of
                its machines will not fetch, so this is absence of a capability
                for the whole cell -- nothing recorded, nothing charged --
                exactly as it is for a cell with one host.
        """

        # All hosts are asked at once: a wide stage costs its slowest host,
        # not the sum of them. What the sequential walk bought -- stopping
        # before an *unreached* host when the fence moved underneath it -- is
        # replaced by checking the fence before dispatch and again after every
        # host has answered; a host already writing when the fence moves was
        # never protected by the between-hosts check either, and the higher
        # fencing on generations remains what actually decides ownership.
        if mutating:
            moved = self._fence_moved()
            if moved:
                return self._stopped(action, asset, cell_id, 0, (), moved)

        pairs = list(zip(self._endpoints, self._workers, strict=True))
        answers_by_rank: list[_HostStageAnswer | None] = [None] * len(pairs)
        capability_failures: list[tuple[int, str, UnsupportedCapabilityError]] = []
        with ThreadPoolExecutor(max_workers=len(pairs), thread_name_prefix=f"eyvan-{action}") as pool:
            futures = {
                pool.submit(copy_context().run, self._act, worker, action, cell_id, asset, max_bytes=max_bytes): (
                    rank,
                    endpoint,
                )
                for rank, (endpoint, worker) in enumerate(pairs)
            }
            for future in concurrent.futures.as_completed(futures):
                rank, endpoint = futures[future]
                try:
                    result = future.result()
                except UnsupportedCapabilityError as exc:
                    capability_failures.append(
                        (
                            rank,
                            f"rank {rank} of cell {cell_id!r} ({endpoint}) has no staging executor, "
                            f"so this cell has none: {exc}",
                            exc,
                        )
                    )
                    continue
                except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                    result = AssetStageResult(
                        verdict=AssetStageVerdict.INCONCLUSIVE,
                        detail=f"native worker {endpoint} could not be asked: {type(exc).__name__}: {exc}",
                    )
                answers_by_rank[rank] = _HostStageAnswer(rank=rank, endpoint=endpoint, result=result)

        # A host with no staging route is absence of a capability only if
        # nothing reached a disk. In parallel everything was attempted at once,
        # so any successful or inconclusive-but-dispatched sibling means the
        # action is under way: report stopped rather than "not capable".
        acted = [answer for answer in answers_by_rank if answer is not None]
        if capability_failures:
            capability_failures.sort(key=lambda item: item[0])
            rank, reason, exc = capability_failures[0]
            if not mutating or not acted:
                raise UnsupportedCapabilityError(reason) from exc
            return self._stopped(action, asset, cell_id, rank, acted, reason)

        if mutating:
            moved = self._fence_moved()
            if moved:
                first_unanswered = next(
                    (rank for rank, answer in enumerate(answers_by_rank) if answer is None), len(answers_by_rank)
                )
                return self._stopped(action, asset, cell_id, first_unanswered, acted, moved)
        return _SliceStageFanout(
            hosts=len(self._workers), answers=tuple(answer for answer in answers_by_rank if answer is not None)
        )

    def _stopped(
        self,
        action: _StageAction,
        asset: str,
        cell_id: str,
        rank: int,
        answers: tp.Sequence[_HostStageAnswer],
        reason: str,
    ) -> _SliceStageFanout:
        """Record a fan-out that stopped at one host, and say so out loud."""

        logger.warning(
            "native slice %s of %s on cell %s stopped at rank %s after %s hosts: %s",
            action,
            asset,
            cell_id,
            rank,
            len(answers),
            reason,
        )
        return _SliceStageFanout(hosts=len(self._workers), answers=tuple(answers), stopped=reason)

    @staticmethod
    def _act(
        worker: NativeRemoteExecutor,
        action: _StageAction,
        cell_id: str,
        asset: str,
        *,
        max_bytes: int,
    ) -> AssetStageResult:
        """Apply one closed staging action to one host."""

        if action == "measure":
            return worker.measure(cell_id, asset, max_bytes=max_bytes)
        if action == "stage":
            return worker.stage(cell_id, asset, max_bytes=max_bytes)
        if action == "evict":
            return worker.evict(cell_id, asset)
        raise ValueError(f"unsupported native slice staging action {action!r}")

    def _fence_moved(self) -> str:
        """Return why this view no longer owns the cell, or an empty string.

        A fence that cannot be read is treated as moved. It is not evidence that
        the generation changed, and it is deliberately not evidence that it did
        not: continuing to write to hosts on the strength of a failed check is
        how an unobservable state becomes an assumed one.
        """

        if self._fence is None:
            return ""
        try:
            current = self._fence()
        except Exception as exc:
            # Deliberately every exception. The fence is somebody else's
            # callable, and a fan-out that continued because the check itself
            # broke in an unforeseen way is the exact failure this guards.
            return f"the cell's generation could not be re-read: {type(exc).__name__}: {exc}"
        if current != self._generation:
            return f"the cell moved to generation {current} while generation {self._generation} was acting"
        return ""

    def _reduce_measure(self, fanout: _SliceStageFanout) -> AssetStageResult:
        """Return what the whole cell would have to admit to hold this asset.

        The largest declared size, not the first and not the mean. Hosts
        legitimately disagree here -- one may already hold the asset and report
        its own file, another reads the source -- and the number the control
        plane reserves against has to cover the host that needs the most.
        Disagreement is therefore reported rather than refused; it is
        :meth:`_reduce_stage` that treats it as a fact about the bytes, because
        only there is it an *observation* of what landed.
        """

        if not fanout.complete:
            return fanout.unreached()
        refused = fanout.matching(AssetStageVerdict.REFUSED)
        if refused:
            return AssetStageResult(
                verdict=AssetStageVerdict.REFUSED,
                detail=(
                    f"{len(refused)} of {fanout.hosts} hosts will never hold this asset, "
                    f"so this cell cannot: {_stage_summary(refused)}"
                ),
            )
        unproven = fanout.matching(AssetStageVerdict.INCONCLUSIVE)
        if unproven:
            return AssetStageResult(
                verdict=AssetStageVerdict.INCONCLUSIVE,
                detail=(
                    f"{len(unproven)} of {fanout.hosts} hosts could not say what this asset would cost: "
                    f"{_stage_summary(unproven)}"
                ),
            )
        sizes = [answer.result.size_bytes for answer in fanout.answers]
        largest = max(sizes)
        detail = f"every one of {fanout.hosts} hosts can be asked; the largest declares {largest} bytes"
        agreed, _ = _sole_value(sizes)
        if not agreed:
            detail = f"{detail}; hosts declare {_stage_summary(fanout.answers, _observed_size)}"
        return AssetStageResult(verdict=AssetStageVerdict.OK, size_bytes=largest, detail=detail)

    def _reduce_stage(self, fanout: _SliceStageFanout) -> AssetStageResult:
        """Return one answer about the cell, refusing to average its hosts.

        Three hosts of four is not a cache hit. The workload's rank on the
        fourth machine opens nothing, so the cell does not hold the asset -- and
        the bytes that did land are on real disks, so the control plane's
        unproven row keeps charging for them exactly as it does for a truncated
        single-host fetch.
        """

        if not fanout.complete:
            return fanout.unreached()
        refused = fanout.matching(AssetStageVerdict.REFUSED)
        if refused:
            return AssetStageResult(
                verdict=AssetStageVerdict.REFUSED,
                detail=(
                    f"{len(refused)} of {fanout.hosts} hosts refused this asset, so this cell will never hold it: "
                    f"{_stage_summary(refused)}"
                ),
            )
        unproven = fanout.matching(AssetStageVerdict.INCONCLUSIVE)
        if unproven:
            return AssetStageResult(
                verdict=AssetStageVerdict.INCONCLUSIVE,
                detail=(
                    f"{fanout.hosts - len(unproven)} of {fanout.hosts} hosts staged this asset, so the cell does "
                    f"not hold it: {_stage_summary(unproven)}"
                ),
            )
        agreed_size, size = _sole_value([answer.result.size_bytes for answer in fanout.answers])
        if not agreed_size:
            return AssetStageResult(
                verdict=AssetStageVerdict.REFUSED,
                detail=(
                    "this cell's hosts hold different numbers of bytes under one asset name, so they are not one "
                    f"asset: {_stage_summary(fanout.answers, _observed_size)}"
                ),
            )
        digests = [answer.result.digest for answer in fanout.answers]
        if any(digest is None for digest in digests) and any(digest is not None for digest in digests):
            # Absence of a digest is not disagreement, and manufacturing one
            # from the hosts that did answer would be inventing the very
            # evidence this comparison exists to demand.
            return AssetStageResult(
                verdict=AssetStageVerdict.INCONCLUSIVE,
                detail=(
                    "some of this cell's hosts proved no digest, so its copies cannot be shown to match: "
                    f"{_stage_summary(fanout.answers, _observed_digest)}"
                ),
            )
        agreed_digest, digest = _sole_value(digests)
        if not agreed_digest:
            return AssetStageResult(
                verdict=AssetStageVerdict.REFUSED,
                detail=(
                    "this cell's hosts hold different bytes under one asset name, so they are not one asset: "
                    f"{_stage_summary(fanout.answers, _observed_digest)}"
                ),
            )
        _, path = _sole_value([answer.result.path for answer in fanout.answers])
        return AssetStageResult(
            verdict=AssetStageVerdict.OK,
            size_bytes=tp.cast(int, size),
            digest=tp.cast("str | None", digest),
            detail=f"staged on every one of {fanout.hosts} hosts in this cell",
            path=tp.cast("str | None", path),
        )

    def _reduce_evict(self, fanout: _SliceStageFanout) -> AssetStageResult:
        """Confirm removal only when every host confirmed it.

        Anything short of unanimous is unproven rather than refused, whatever
        the individual hosts said. The caller's question is whether the bytes
        are gone from this cell, a host that did not answer may still hold them,
        and the row that survives an unproven eviction is what stops the ledger
        claiming a disk it never emptied. The reason each host gave is kept in
        the detail, and the control plane's attempt budget is what stops an
        eviction that can never complete from being retried forever.
        """

        proven = fanout.matching(AssetStageVerdict.OK)
        if fanout.complete and len(proven) == fanout.hosts:
            return AssetStageResult(
                verdict=AssetStageVerdict.OK,
                size_bytes=sum(answer.result.size_bytes for answer in proven),
                detail=f"removed from every one of {fanout.hosts} hosts in this cell",
            )
        if not fanout.complete:
            return fanout.unreached()
        remaining = tuple(answer for answer in fanout.answers if answer.result.verdict is not AssetStageVerdict.OK)
        return AssetStageResult(
            verdict=AssetStageVerdict.INCONCLUSIVE,
            detail=(
                f"{len(proven)} of {fanout.hosts} hosts confirmed removal, so this asset may still be on the cell: "
                f"{_stage_summary(remaining)}"
            ),
        )


@tp.runtime_checkable
class MultiHostStageCapableDriver(tp.Protocol):
    """An execution driver that stages across the several hosts of one cell.

    The seam between "this driver can put bytes somewhere" and "this driver
    knows *where*". A single-worker driver answers the first by satisfying
    :class:`~eyvan.worker.runtime.stage_executor.AssetStageDriver` directly,
    because its one host is the answer to the second. A slice cannot: it must be
    told which generation it is acting for before it may touch any of its hosts,
    so it offers this instead and hands back a view bound to that fence.

    Runtime-checkable for the same reason the rest of ``runtime/registry``'s
    capability protocols are: a backend that cannot do this says so by not
    satisfying it, rather than by returning something that looks like a result.
    """

    def asset_stage_driver(self, *, generation: int, fence: StageFence | None = None) -> AssetStageDriver: ...


class NativeSliceExecutor:
    """Fan one fenced workload out to every host in an atomic TPU slice group.

    Each remote worker receives the same logical submission ID, but rank-local
    JAX environment variables. The configured ``sitecustomize`` hook performs
    ``jax.distributed.initialize`` before the workload imports JAX.
    """

    def __init__(
        self,
        endpoints: tp.Sequence[str],
        *,
        secret: str,
        clients: tp.Sequence[httpx.Client] | None = None,
        jax_cluster_detection: JaxClusterDetectionMode = JaxClusterDetectionMode.EXPLICIT,
        initialization: GangInitializationPolicy = DEFAULT_GANG_INITIALIZATION,
        mtls: MutualTlsFiles | None = None,
    ) -> None:
        normalized = tuple(endpoint.rstrip("/") for endpoint in endpoints)
        if not normalized or any(not endpoint or endpoint != endpoint.strip() for endpoint in normalized):
            raise ValueError("native slice execution requires non-empty worker endpoints")
        if len(normalized) != len(set(normalized)):
            raise ValueError("native slice execution worker endpoints must be unique")
        coordinator = urlparse(normalized[0])
        if coordinator.scheme not in {"http", "https"} or not coordinator.hostname:
            raise ValueError("native slice coordinator endpoint must be an HTTP URL with a host")
        if clients is not None and len(clients) != len(normalized):
            raise ValueError("native slice clients must match the number of worker endpoints")
        if not isinstance(jax_cluster_detection, JaxClusterDetectionMode):
            raise TypeError("native slice JAX cluster detection must be a JaxClusterDetectionMode")
        if not isinstance(initialization, GangInitializationPolicy):
            raise TypeError("native slice initialization must be a GangInitializationPolicy")
        owned_endpoints = (
            normalized
            if clients is None
            else tuple(endpoint for endpoint, client in zip(normalized, clients, strict=True) if client is None)
        )
        if mtls is not None and owned_endpoints:
            if any(urlparse(endpoint).scheme != "https" for endpoint in owned_endpoints):
                raise ValueError("mutual TLS requires HTTPS native-worker endpoints")
            client_ssl_context(mtls)
        self.endpoints = normalized
        self.jax_cluster_detection = jax_cluster_detection
        self.initialization = initialization
        self._coordinator_address = f"{coordinator.hostname}:{_COORDINATOR_PORT}"
        self._workers = tuple(
            NativeRemoteExecutor(
                endpoint,
                secret=secret,
                client=client,
                **({"mtls": mtls} if mtls is not None else {}),
            )
            for endpoint, client in zip(normalized, clients or (None,) * len(normalized), strict=True)
        )

    def health(self) -> bool:
        """Return whether every host accepts authenticated native-worker requests."""

        return all(worker.health() for worker in self._workers)

    def worker_builds(self) -> list[dict[str, object]]:
        """Return what code each host in this slice is running.

        One entry per rank, always, in rank order. A rank that could not answer
        carries ``None`` rather than being dropped: the caller is asking whether
        a slice is uniform, and a shorter list would answer that question by
        omitting the ranks it could not check.

        These processes are the ones that execute the workload, and they are
        installed from the wheel their cell was provisioned with -- which is not
        necessarily the one the head stages now, because a cell outlives a
        re-staged head.
        """

        collected: list[dict[str, object]] = []
        for rank, (endpoint, worker) in enumerate(zip(self.endpoints, self._workers, strict=True)):
            reported = worker.build()
            collected.append(
                {
                    "rank": rank,
                    "endpoint": endpoint,
                    "build": None if reported is None else reported.model_dump(mode="json"),
                }
            )
        return collected

    def asset_stage_driver(self, *, generation: int, fence: StageFence | None = None) -> NativeSliceStageDriver:
        """Return a staging view over this slice, bound to one cell generation.

        Deliberately a separate object rather than three more methods on this
        class. This executor is cached by the agent's driver resolver and
        outlives a generation bump that leaves the endpoints and the worker
        credential unchanged, so a staging surface hanging off it directly would
        carry no fence at all -- and staging is the one cell operation whose
        wire has no generation of its own to check on arrival.

        Args:
            generation: The cell generation the caller has already fenced
                against, and the only one the returned view may act for.
            fence: Re-reads the generation that owns the cell now, once before
                the hosts of an action that touches a disk are dispatched and
                once after they have all answered.

        Returns:
            A driver satisfying
            :class:`~eyvan.worker.runtime.stage_executor.AssetStageDriver` that
            fans each action to every host in this slice and answers once.
        """

        return NativeSliceStageDriver(self._workers, self.endpoints, generation=generation, fence=fence)

    def _targets(self, remote_id: str) -> list[tuple[int, NativeRemoteExecutor]]:
        """Return the ``(rank, worker)`` pairs that own ``remote_id``.

        Every host for a whole-slice job; only the owning host for a chip share,
        whose remote ID names it (see ``chip_submission_id``).
        """

        host = chip_host_of(remote_id)
        if host is None:
            return list(enumerate(self._workers))
        if host >= len(self._workers):
            raise ValueError(f"chip share names host {host}, but this slice has {len(self._workers)} hosts")
        return [(host, self._workers[host])]

    def _ranked_spec(self, spec: tp.Mapping[str, JsonValue], rank: int) -> JsonObject:
        """Copy one JSON workload spec and add its rank-local environment."""

        payload: JsonObject = copy.deepcopy(dict(spec))
        raw_environment = payload.get("environment")
        if raw_environment is None:
            source_environment: JsonObject = {}
        elif isinstance(raw_environment, dict):
            source_environment = raw_environment
        else:
            raise ValueError("native slice workload environment must be a mapping")
        environment: dict[str, str] = {}
        for key, value in source_environment.items():
            if not isinstance(value, str):
                raise ValueError("native slice workload environment must be a string mapping")
            environment[key] = value
        cross_region_environment = {
            key: value
            for key, value in environment.items()
            if key == "EYVAN_HOST_REGION" or key.startswith("EYVAN_CROSS_REGION_")
        }
        environment = without_distributed_topology_overrides(environment)
        environment.update(cross_region_environment)
        # The checkpoint hook writes to ``{base}/rank-{n}``, so a resume handed
        # the bare base URI reads an object that was never written. It does not
        # fail: the workload sees "no checkpoint" and restarts from scratch,
        # which on a live v5p-16 silently discarded 252 completed training
        # steps after a successful preemption and checkpoint. The two paths
        # have to agree, so derive the resume URI the same way.
        resume_uri = environment.get(_RESUME_CHECKPOINT_ENV, "")
        if resume_uri:
            environment[_RESUME_CHECKPOINT_ENV] = _rank_checkpoint_uri(resume_uri, rank)
        python_path = environment.get("PYTHONPATH", "")
        environment.update(
            {
                "EYVAN_DISTRIBUTED_PROCESS_COUNT": str(len(self._workers)),
                "EYVAN_DISTRIBUTED_PROCESS_ID": str(rank),
                "EYVAN_JAX_CLUSTER_DETECTION": self.jax_cluster_detection.value,
                "EYVAN_JAX_MULTIHOST": "1",
                "EYVAN_REPLICA_COUNT": str(len(self._workers)),
                "EYVAN_REPLICA_INDEX": str(rank),
                "JAX_COORDINATOR_PORT": str(_COORDINATOR_PORT),
                "PYTHONPATH": f"{_MULTIHOST_SITE_PATH}:{python_path}" if python_path else _MULTIHOST_SITE_PATH,
            }
        )
        if self.jax_cluster_detection is JaxClusterDetectionMode.EXPLICIT:
            environment.update(
                {
                    "JAX_COORDINATOR_ADDRESS": self._coordinator_address,
                    "JAX_PROCESS_COUNT": str(len(self._workers)),
                    "JAX_PROCESS_ID": str(rank),
                    "JAX_PROCESS_INDEX": str(rank),
                }
            )
        # Last, and after the topology scrub above, so the deadline a cold slice
        # gets is the one this control plane configured. A caller who submits
        # its own copy has already had it stripped -- the whole ``EYVAN_JAX_``
        # prefix is Eyvan's -- and could otherwise hand every rank a one-second
        # barrier and blame the fleet.
        environment.update(self.initialization.environment())
        payload["environment"] = environment
        return payload

    def submit(self, spec: tp.Mapping[str, JsonValue], context: MutationContext) -> str:
        """Submit one identical logical workload to every slice host.

        A submission is only considered accepted when every worker returns the
        same logical ID. Any partial launch is fenced and retired before the
        original error is raised. A retire failure during that cleanup leaves a
        rank running, so it is logged at warning level and appended to the
        raised error to keep the orphaned rank observable.
        """

        raw_submission_id = spec.get("submission_id")
        if raw_submission_id is None:
            expected_remote_id = context.idempotency_key
        elif isinstance(raw_submission_id, str) and raw_submission_id.strip():
            expected_remote_id = raw_submission_id.strip()
        else:
            raise ValueError("native slice submission requires a non-empty submission ID")
        chip_host = chip_host_of(expected_remote_id)
        attempted: list[tuple[int, NativeRemoteExecutor, set[str]]] = []
        try:
            for rank, worker in self._targets(expected_remote_id):
                remote_ids = {expected_remote_id}
                attempted.append((rank, worker, remote_ids))
                # A chip share is its own single-process runtime on one host: the
                # controller already wrote its libtpu environment, and the
                # slice-wide JAX world must not be layered over it.
                payload = dict(spec) if chip_host is not None else self._ranked_spec(spec, rank)
                remote_id = worker.submit(payload, context)
                remote_ids.add(remote_id)
                if remote_id != expected_remote_id:
                    raise RuntimeError(
                        f"native slice worker returned unexpected remote ID {remote_id!r}; "
                        f"expected {expected_remote_id!r}"
                    )
            return expected_remote_id
        except BaseException as launch_error:
            retire_failures: list[str] = []
            for rank, worker, remote_ids in reversed(attempted):
                for attempted_remote_id in sorted(remote_ids):
                    try:
                        worker.retire(attempted_remote_id, rollback_context(context))
                    except Exception as exc:
                        retire_failures.append(f"rank {rank} ({worker.endpoint}): {type(exc).__name__}")
            if retire_failures:
                detail = f"partial-launch cleanup could not retire {_bounded_retire_summary(retire_failures)}"
                logger.warning("native slice submission failed and %s", detail)
                _augment_error_detail(launch_error, detail)
            raise

    def status(self, remote_id: str) -> NativeSliceWorkloadResult:
        """Aggregate typed rank observations into one durable slice state."""

        def observe(rank: int, worker: NativeRemoteExecutor) -> _SliceWorkerObservation:
            endpoint = self.endpoints[rank]
            try:
                observation = _normalize_worker_observation(
                    worker.status(remote_id),
                    expected_remote_id=remote_id,
                    endpoint=endpoint,
                )
            except httpx.TransportError as exc:
                # Same reasoning as the gang executor: an unreachable endpoint
                # is a fact about the connection, not the rank. Reading it as
                # LOST failed whole slices over a closed keep-alive; reading it
                # as INDETERMINATE lets the next poll say what actually
                # happened, and sustained unreachability escalates through the
                # indeterminate grace path instead of killing live peers.
                observation = _indeterminate_observation(
                    remote_id,
                    f"native worker {endpoint} is unreachable: {type(exc).__name__}: {exc}",
                )
            except (httpx.HTTPStatusError, RuntimeError, ValueError) as exc:
                observation = _indeterminate_observation(
                    remote_id,
                    _invalid_observation_reason(exc, endpoint=endpoint),
                )
            return _SliceWorkerObservation(rank=rank, endpoint=endpoint, observation=observation)

        if chip_host_of(remote_id) is not None:
            workers = [observe(rank, worker) for rank, worker in self._targets(remote_id)]
        else:
            workers = _each_host(self._workers, observe)
        status = _aggregate_status(workers)
        message = _aggregate_message(workers)
        if status is ExecutionWorkloadStatus.FAILED:
            terminal_reason = self._terminal_failure_reason(remote_id, workers)
            if terminal_reason is not None:
                message = f"{terminal_reason} {message}"
        aggregate = ExecutionWorkloadObservation(
            remote_id=remote_id,
            status=status,
            message=message,
        )
        return {
            "remote_id": aggregate.remote_id,
            "status": aggregate.status,
            "message": aggregate.message,
            "workers": [worker.as_result() for worker in workers],
        }

    def _cross_region_marker_detail(
        self,
        remote_id: str,
        ranks: tp.Sequence[int],
        workers: tp.Sequence[_SliceWorkerObservation],
    ) -> str | None:
        """Extract the cross-region ban stderr marker from worker observations or logs."""

        by_rank = {worker.rank: worker for worker in workers}
        for rank in sorted(set(ranks)):
            obs = by_rank.get(rank)
            if obs is not None:
                detail = extract_cross_region_ban_marker(obs.observation.message)
                if detail:
                    return detail
        for rank in sorted(set(ranks)):
            if not 0 <= rank < len(self._workers):
                continue
            try:
                logs = self._workers[rank].logs(remote_id)
            except (httpx.HTTPError, RuntimeError, ValueError):
                continue
            detail = extract_cross_region_ban_marker(logs)
            if detail:
                return detail
        return None

    def _terminal_failure_reason(
        self,
        remote_id: str,
        workers: tp.Sequence[_SliceWorkerObservation],
    ) -> str | None:
        """Return a structured terminal failure prefix for exit code 95 or gang-init failure."""

        marker_ranks = [
            worker.rank
            for worker in workers
            if CROSS_REGION_TRANSFER_BANNED_MARKER in worker.observation.message
            or CROSS_REGION_TRANSFER_BANNED_REASON in worker.observation.message.lower()
        ]
        if not self.initialization.fatal:
            if not marker_ranks:
                return None
            return cross_region_ban_failure_reason(
                banned_ranks=marker_ranks,
                marker_detail=self._cross_region_marker_detail(remote_id, marker_ranks, workers),
            )

        banned_ranks: list[int] = list(marker_ranks)
        deadline_ranks: list[int] = []
        error_ranks: list[int] = []
        for rank, worker in enumerate(self._workers):
            try:
                detail = worker.process_status(remote_id)
            except (httpx.HTTPError, RuntimeError, ValueError):
                continue
            exit_status = _reported_exit_status(detail)
            if exit_status == CROSS_REGION_TRANSFER_BANNED_EXIT_CODE:
                banned_ranks.append(rank)
                continue
            outcome = gang_initialization_outcome(exit_status)
            if outcome == "deadline":
                deadline_ranks.append(rank)
            elif outcome == "error":
                error_ranks.append(rank)
        if banned_ranks:
            return cross_region_ban_failure_reason(
                banned_ranks=banned_ranks,
                marker_detail=self._cross_region_marker_detail(remote_id, banned_ranks, workers),
            )
        if not deadline_ranks and not error_ranks:
            return None
        return gang_formation_failure_reason(
            deadline_s=self.initialization.initialization_timeout_s,
            deadline_ranks=deadline_ranks,
            error_ranks=error_ranks,
        )

    def stop(self, remote_id: str, context: MutationContext) -> None:
        """Stop the workload on every host, preserving the first failure."""

        self._fanout_mutation("stop", remote_id, context)

    def retire(self, remote_id: str, context: MutationContext) -> None:
        """Retire the workload on every host, preserving the first failure."""

        self._fanout_mutation("retire", remote_id, context)

    def _fanout_mutation(self, action: _MutationAction, remote_id: str, context: MutationContext) -> None:
        """Apply one closed mutation action to every worker."""

        first_error: BaseException | None = None
        for _rank, worker in self._targets(remote_id):
            try:
                if action == "stop":
                    worker.stop(remote_id, context)
                elif action == "retire":
                    worker.retire(remote_id, context)
                else:
                    raise ValueError(f"unsupported native slice mutation action {action!r}")
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def checkpoint(
        self,
        remote_id: str,
        *,
        hook: str,
        output_uri: str | None,
        context: MutationContext,
    ) -> str:
        """Launch one rank-aware checkpoint hook across every host in the slice.

        The hook runs on all hosts under a single shared logical ID, and each
        host's worker inherits *that host's* source environment -- so every rank
        keeps its JAX coordinator address, process id, and process count and the
        hook forms the same cluster the workload itself did. Without that, ranks
        would each come up as a single-process job and race on one output URI.

        Launch is all-or-nothing, mirroring ``submit``: a partial launch is
        fenced and retired before the original error is raised, because a
        surviving rank would hold accelerators and could still write to the
        checkpoint destination. Completion is *not* asserted here -- callers
        poll ``status``, which only reports success once every rank has
        succeeded.
        """

        if not isinstance(hook, str) or not hook.strip():
            raise ValueError("native slice checkpoint hook must be non-empty")
        expected_remote_id = context.idempotency_key
        attempted: list[tuple[int, NativeRemoteExecutor, set[str]]] = []
        try:
            for rank, worker in self._targets(remote_id):
                remote_ids = {expected_remote_id}
                attempted.append((rank, worker, remote_ids))
                checkpoint_id = worker.checkpoint(
                    remote_id,
                    hook=hook,
                    output_uri=output_uri,
                    context=context,
                )
                remote_ids.add(checkpoint_id)
                if checkpoint_id != expected_remote_id:
                    raise RuntimeError(
                        f"native slice worker returned unexpected checkpoint ID {checkpoint_id!r}; "
                        f"expected {expected_remote_id!r}"
                    )
            return expected_remote_id
        except BaseException as launch_error:
            retire_failures: list[str] = []
            for rank, worker, remote_ids in reversed(attempted):
                for attempted_remote_id in sorted(remote_ids):
                    try:
                        worker.retire(attempted_remote_id, rollback_context(context))
                    except Exception as exc:
                        retire_failures.append(f"rank {rank} ({worker.endpoint}): {type(exc).__name__}")
            if retire_failures:
                detail = f"partial checkpoint cleanup could not retire {_bounded_retire_summary(retire_failures)}"
                logger.warning("native slice checkpoint failed and %s", detail)
                _augment_error_detail(launch_error, detail)
            raise

    def logs(self, remote_id: str) -> str:
        """Return rank-prefixed logs from every TPU VM host when available."""

        sections: list[str] = []
        for rank, worker in self._targets(remote_id):
            endpoint = self.endpoints[rank]
            try:
                logs = worker.logs(remote_id)
            except httpx.TransportError as exc:
                logs = f"native worker is unreachable: {type(exc).__name__}: {exc}"
            except (httpx.HTTPStatusError, RuntimeError, ValueError) as exc:
                logs = f"native worker log retrieval failed: {type(exc).__name__}: {exc}"
            sections.append(f"===== rank {rank} {endpoint} =====\n{logs}")
        return "\n".join(sections)

    def log_page(self, remote_id: str, *, after: str = "", limit: int = 200) -> LogPage:
        """Return one structured page covering every host in this slice.

        The control plane fans out across *cells*, while this layer exposes
        every host's local rank inside one cell. The controller offsets those
        local ranks into the workload's deterministic global rank space; without
        this merge a v5p-16 would report only whichever host answered first.

        Each host resumes from its own position. A single cursor is meaningless
        to a journal that did not issue it, which is why `merge_pages` refuses to
        invent one, and the same per-rank encoding the control plane uses is
        reused rather than respelled.

        An unreachable host is reported *in place* via `truncated_ranks` rather
        than dropped: a fan-out that silently returned the hosts it could reach
        would show a complete-looking log for a half-dead slice, which is the
        failure this whole aggregation exists to prevent.

        A cursor this slice cannot resolve is answered as an unavailable page
        naming the codec's refusal, not by silently restarting every host. The
        two are indistinguishable to the control plane, and the silent one
        replays the whole slice's journal while reporting a clean read.
        """

        # A chip share runs on one host, so its cursor and its pages name that
        # host's rank only; a whole-slice job reads every host as before.
        targets = self._targets(remote_id)
        target_ranks = tuple(rank for rank, _ in targets)
        expected = CursorContext(
            kind=CURSOR_KIND_WORKLOAD_LOG,
            mode=CURSOR_MODE_LIVE,
            ranks=target_ranks,
        )
        outcome = decode_log_cursor(after, expected=expected)
        if not isinstance(outcome, (CursorFresh, CursorResume)):
            return LogPage(
                unavailable=True,
                detail=f"{outcome.reason}: {outcome.detail}",
                truncated_ranks=target_ranks,
            )
        _attempt = outcome.attempt if isinstance(outcome, CursorResume) else None
        cursors = outcome.cursors if isinstance(outcome, CursorResume) else {}

        def read(rank: int, worker: NativeRemoteExecutor) -> LogPage:
            try:
                return worker.log_page(remote_id, after=cursors.get(rank, ""), limit=limit)
            except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                return LogPage(unavailable=True, detail=f"{type(exc).__name__}: {exc}")

        pages: dict[int, LogPage] = {}
        next_cursors: dict[int, str] = {}
        answers = _each_host([worker for _, worker in targets], lambda index, worker: read(targets[index][0], worker))
        for rank, page in zip(target_ranks, answers, strict=True):
            pages[rank] = LogPage(
                records=tuple(replace(record, rank=rank) for record in page.records),
                next_cursor=page.next_cursor,
                unavailable=page.unavailable,
                detail=page.detail,
                truncated_ranks=(rank,) if page.truncated_ranks else (),
            )
            # A backend legitimately returns no cursor when this poll produced
            # no new lines. Keep the position it was asked from; dropping it
            # would restart that host on the next outer poll and duplicate its
            # tail. The same applies while a host is temporarily unreachable.
            advanced = cursors.get(rank, "") if page.unavailable else (page.next_cursor or cursors.get(rank, ""))
            if advanced:
                next_cursors[rank] = advanced
        merged = merge_pages(pages)
        # ``merge_pages`` deliberately returns no cursor; the slice-level one is
        # the per-host positions joined, which is the only thing that can resume
        # every host.
        return LogPage(
            records=merged.records,
            next_cursor=join_log_cursor(_attempt or 0, next_cursors),
            unavailable=merged.unavailable,
            detail=merged.detail,
            truncated_ranks=merged.truncated_ranks,
        )

    def process_status(self, remote_id: str) -> JsonObject:
        """Return per-rank process detail across every host in the slice.

        A hung multi-host job is exactly the case where one rank differs from
        the rest, so an unreachable host is reported in place rather than
        failing the whole probe.
        """

        ranks: list[JsonValue] = []
        for rank, worker in self._targets(remote_id):
            endpoint = self.endpoints[rank]
            try:
                detail: JsonValue = worker.process_status(remote_id)
            except httpx.TransportError as exc:
                detail = {"observed": False, "detail": f"native worker is unreachable: {type(exc).__name__}: {exc}"}
            except (httpx.HTTPStatusError, RuntimeError, ValueError) as exc:
                detail = {"observed": False, "detail": f"process status failed: {type(exc).__name__}: {exc}"}
            ranks.append({"rank": rank, "endpoint": endpoint, "status": detail})
        return {"remote_id": remote_id, "ranks": ranks}

    def exec_in_workload_rank(
        self, remote_id: str, command: tp.Sequence[str], *, timeout_s: float, runtime_rank: int
    ) -> JsonObject:
        """Address exactly one inner runtime without contacting other hosts."""

        return self.exec_in_workload(remote_id, command, timeout_s=timeout_s, runtime_rank=runtime_rank)

    def exec_in_workload(
        self,
        remote_id: str,
        command: tp.Sequence[str],
        *,
        timeout_s: float = 30.0,
        runtime_rank: int | None = None,
    ) -> JsonObject:
        """Run one diagnostic command beside this workload on every host.

        Every host, because the question exec answers is usually "which rank is
        stuck", and one rank cannot tell a hung collective from a slow step.

        An unreachable host is reported in place, exactly as ``process_status``
        and the structured-log fan-out report one. A probe that returned only the
        hosts it could reach would show a complete-looking answer for a
        half-dead slice.
        """

        if runtime_rank is not None and (
            isinstance(runtime_rank, bool)
            or not isinstance(runtime_rank, int)
            or not 0 <= runtime_rank < len(self._workers)
        ):
            raise ValueError("runtime rank does not belong to this slice")
        workers = [
            (rank, endpoint, worker)
            for rank, (endpoint, worker) in enumerate(zip(self.endpoints, self._workers, strict=True))
            if runtime_rank is None or rank == runtime_rank
        ]
        return {"remote_id": remote_id, "ranks": _exec_fanout(workers, remote_id, command, timeout_s=timeout_s)}

    def close(self) -> None:
        """Close internally owned native-worker clients."""

        for worker in self._workers:
            worker.close()


__all__ = (
    "MultiHostStageCapableDriver",
    "NativeSliceExecutor",
    "NativeSliceStageDriver",
    "StageFence",
)
