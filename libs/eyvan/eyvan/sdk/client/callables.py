# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""SDK adapter that submits decorated callables through Eyvan's HTTP API."""

from __future__ import annotations

import typing as tp
from collections import deque
from dataclasses import replace

from pydantic import ValidationError
from rich.text import Text

from eyvan.artifacts import ArtifactStore
from eyvan.capacity.acquisition_backoff import is_acquisition_backoff_reason
from eyvan.capacity.autoscale import AutoscaleConfigCache, QueueAdmissionDenialRecorder
from eyvan.records import (
    CallableExecution,
    CapacityClass,
    PlacementMode,
    WorkloadProgress,
    WorkloadProgressPhase,
    WorkloadRecord,
    WorkloadState,
)
from eyvan.sdk.client.jobs import (
    DEFAULT_RECLAIM_GRACE_SECONDS,
    AcceleratorConfig,
    CallableSubmission,
    EyvanJob,
    EyvanPlacementError,
    JobCompletion,
    format_reclaim_status,
    placement_error_labels,
    reject_remote_callable_signature,
    reject_remote_callable_signing_key,
)
from eyvan.sdk.client.sync import EyvanClient
from eyvan.sdk.client.transport import EyvanHTTPError, EyvanUnavailableError
from eyvan.sdk.follow import EXIT_DETACHED, _LogWindow
from eyvan.sdk.job_receipt import (
    JobLifecycleState,
    JobLifecycleUpdate,
    JobSubmissionReceipt,
    JobTerminalOutput,
)
from eyvan.sdk.lifecycle import (
    TERMINAL_STATES,
    running_workload_update,
    terminal_workload_update,
    worker_count,
    worker_request,
)
from eyvan.server.api.schemas import CapacityClassUnavailableErrorResponse, NoMatchingCapacityPoolErrorResponse
from eyvan.worker.runtime.results import load_result


def _job_words(line: str) -> Text:
    """The job's own words, keeping whatever colours it printed itself.

    A training script that colours its own output sent those bytes through
    journald to arrive here; reading them back as styles is what lets them
    survive a terminal, and stripping them is what keeps redirected output
    clean -- escape codes in a log file are corruption, not decoration. A
    line with no escapes is returned untouched, because most lines have none
    and the check is one comparison.
    """

    return Text.from_ansi(line) if "\x1b" in line else Text(line)


def _chorus_groups(lines: tp.Sequence[tuple[int, str]]) -> list[tuple[int, ...]]:
    """Group one page's ``(rank, message)`` lines into what renders as one line each.

    A gang's ranks print the same program, but their lines reach the page
    interleaved microseconds apart, so "identical and adjacent" split one
    16-host ``print`` into runs of 2x, 5x and single workers. The unit here is
    instead *the k-th time a rank said this exact message*: rank 3's first
    ``start`` and rank 9's first ``start`` are one chorus however far apart they
    landed, while a rank repeating itself is never merged with itself.

    Every rank's own lines keep their order. Groups are emitted in a merge over
    the ranks' sequences: a group is emitted once it is next for every rank in
    it, earliest first appearance first. When ranks disagree on order (A says
    X then Y, B says Y then X) no group is ready, and the earliest one is
    emitted for just the ranks that have it next -- the chorus splits rather
    than reordering anybody's output. A page without rank data is one rank,
    so nothing in it can merge.

    Returns:
        Page indices per rendered line, in display order; every index once.
    """

    occurrences: dict[tuple[int, str], int] = {}
    group_ids: dict[tuple[str, int], int] = {}
    members: list[dict[int, int]] = []  # per group: rank -> page index
    first_seen: list[int] = []
    queues: dict[int, deque[int]] = {}
    for index, (rank, message) in enumerate(lines):
        nth = occurrences.get((rank, message), 0)
        occurrences[(rank, message)] = nth + 1
        group = group_ids.setdefault((message, nth), len(members))
        if group == len(members):
            members.append({})
            first_seen.append(index)
        members[group][rank] = index
        queues.setdefault(rank, deque()).append(group)

    ordered: list[tuple[int, ...]] = []
    while queues:
        holders: dict[int, list[int]] = {}
        for rank, queue in queues.items():
            holders.setdefault(queue[0], []).append(rank)
        ready = [group for group, ranks in holders.items() if len(ranks) == len(members[group])]
        chosen = min(ready or holders, key=first_seen.__getitem__)
        ranks = holders[chosen]
        ordered.append(tuple(sorted(members[chosen].pop(rank) for rank in ranks)))
        for rank in ranks:
            queues[rank].popleft()
            if not queues[rank]:
                del queues[rank]
    return ordered


def _attributed(prefix: str, line: str) -> Text:
    """One job line with its attribution tinted and its words left alone.

    The tint rides on a span over the prefix only. ``Text``'s own style
    argument tints everything appended after it, so ``Text(prefix,
    style="cyan").append(line)`` painted the job's words cyan too -- and a
    user reading their own banner in Eyvan's colour is being told, wrongly,
    that the job said it that way. The words are the job's; the prefix is
    ours; only ours gets our colour.
    """

    attributed = Text()
    attributed.append(prefix, style="cyan")
    attributed.append_text(_job_words(line))
    return attributed


R = tp.TypeVar("R")
_ATOMIC_ALLOCATION = "atomic allocation"


def _allocation(placement: PlacementMode, cells: int) -> str:
    """Name what the plane is asking the provider for, in the job's own allocation mode.

    Every job used to be narrated as "one atomic allocation". An elastic job --
    the default -- is satisfied cell by cell from warm and newly provisioned
    slices, so that sentence described a request the plane was not making.
    """

    if placement is PlacementMode.ATOMIC:
        return f"one {_ATOMIC_ALLOCATION}"
    return "one elastic cell" if cells == 1 else f"{cells} elastic cells"


def _allocation_requests(placement: PlacementMode, cells: int) -> str:
    """Name the provider requests being sent, in the job's own allocation mode."""

    if placement is PlacementMode.ATOMIC:
        return f"one {_ATOMIC_ALLOCATION} request"
    return "one elastic cell request" if cells == 1 else f"{cells} elastic cell requests"


def _tpu_vm_count(count: int) -> str:
    """Format a GCP-observed TPU VM count without conflating it with chips."""

    return f"{count} TPU VM" if count == 1 else f"{count} TPU VMs"


def _existing_hardware_class(error: EyvanHTTPError, request: CallableSubmission) -> CapacityClass | None:
    """Return the fleet's one existing-hardware class for this shape, as the plane stated it.

    An attached TPU or operator-owned host already runs and is already billed
    whatever Eyvan does, so running there in its real class -- even on-demand --
    starts no spend nobody asked for. Only the plane can see a pool's backend
    for an ordinary project user, so this trusts its typed refusal and nothing
    else: an older plane's untyped 409, a different fleet or shape, or more than
    one candidate class all keep the original error.
    """

    try:
        response = CapacityClassUnavailableErrorResponse.model_validate_json(error.response.content)
    except ValidationError:
        return None
    context = response.context
    if context.fleet != request.fleet or context.requested_accelerator_type != request.tpu.type:
        return None
    if len(context.existing_hardware_capacity_classes) != 1:
        return None
    try:
        return CapacityClass(context.existing_hardware_capacity_classes[0])
    except ValueError:
        return None


def _placement_error(
    error: EyvanHTTPError,
    *,
    config: AcceleratorConfig | None = None,
) -> EyvanPlacementError | None:
    """Translate the typed no-pool API response into the public SDK error."""

    try:
        response = NoMatchingCapacityPoolErrorResponse.model_validate_json(error.response.content)
    except ValidationError:
        return None
    accelerator_label, config_name = placement_error_labels(config)
    return EyvanPlacementError(
        fleet=response.context.fleet,
        requested_accelerator_type=response.context.requested_accelerator_type,
        available_accelerator_types=response.context.available_enabled_accelerator_types,
        fleets_providing_requested_type=response.context.fleets_providing_requested_type,
        available_fleets=response.context.available_fleets,
        accelerator_label=accelerator_label,
        config_name=config_name,
    )


class _CallableProgressRenderer:
    """Render change-only autoscaling progress for one blocking callable result wait."""

    def __init__(
        self,
        client: EyvanClient,
        output: JobTerminalOutput,
        queue_admission_recorder: QueueAdmissionDenialRecorder | None,
    ) -> None:
        self._client = client
        self._output = output
        self._queue_admission_recorder = queue_admission_recorder
        self._last_update: JobLifecycleUpdate | None = None
        self._last_diagnostic: str | None = None
        #: Where the streamed transcript stopped reading. Ray-style: a blocking
        #: wait tails the job's own output, so the run reads like a local process.
        self._log_cursor = ""

    def __call__(self, workload: WorkloadRecord) -> None:
        """Render one control-plane observation without repeating unchanged state."""

        terminal = workload.state in TERMINAL_STATES
        if terminal:
            # Settlement archives remaining output before the terminal state is
            # visible; drain first so the job's last lines precede its verdict.
            self._drain_output(workload.id, terminal=True)
            if self._queue_admission_recorder is not None:
                self._queue_admission_recorder.record_workload(workload)
            self._emit(self._terminal_workload_update(workload))
            return
        reclaim_status = format_reclaim_status(workload)
        try:
            progress = self._client.get_workload_progress(workload.id)
        except EyvanHTTPError as error:
            # Own-workload status need not include access to project capacity details.
            if error.status_code not in {403, 404}:
                raise
            if self._queue_admission_recorder is not None:
                self._queue_admission_recorder.record_workload(workload)
            self._emit(self._with_reclaim_status(self._fallback_or_terminal_update(workload), reclaim_status))
            return
        except EyvanUnavailableError:
            if self._queue_admission_recorder is not None:
                self._queue_admission_recorder.record_workload(workload)
            self._emit(self._with_reclaim_status(self._fallback_or_terminal_update(workload), reclaim_status))
            self._drain_output(workload.id)
            return
        if self._queue_admission_recorder is not None:
            self._queue_admission_recorder.record_workload(workload)
        self._emit(
            self._with_reclaim_status(
                self._progress_update(progress, placement=workload.spec.placement),
                reclaim_status,
            )
        )
        self._drain_output(workload.id)

    @staticmethod
    def _with_reclaim_status(update: JobLifecycleUpdate, reclaim_status: str | None) -> JobLifecycleUpdate:
        """Overlay live reclaim/checkpoint countdown messaging onto ``update`` when active."""

        if reclaim_status is None:
            return update
        if update.state in {
            JobLifecycleState.REQUESTING,
            JobLifecycleState.QUEUED,
            JobLifecycleState.READY,
            JobLifecycleState.CHECKPOINTING,
        }:
            return replace(update, summary=reclaim_status)
        return replace(update, diagnostic=reclaim_status)

    def _emit(self, update: JobLifecycleUpdate) -> None:
        if update == self._last_update:
            return
        if update.diagnostic == self._last_diagnostic:
            update = replace(update, diagnostic=None)
        if update == self._last_update:
            return
        previous = self._last_update
        if (
            previous is not None
            and update.state == previous.state
            and update.state not in {JobLifecycleState.SUCCEEDED, JobLifecycleState.FAILED, JobLifecycleState.CANCELLED}
        ):
            # Same state, new progress: draw only the line under the heading.
            self._output.lifecycle_detail(update)
        else:
            self._output.lifecycle(update)
        self._last_update = update
        if update.diagnostic is not None:
            self._last_diagnostic = update.diagnostic

    def _drain_output(self, workload_id: str, *, terminal: bool = False) -> None:
        """Print produced output this renderer has not shown yet.

        The control-plane log page is the same surface ``eyvan workload logs
        --follow`` reads; polling it here makes a blocking callable wait tail
        its job like ``ray job submit`` does. Live polls are bounded, but a
        terminal observation has no later poll to recover an unread tail.
        Reading is best-effort: an unreachable plane or an attempt that is
        still archiving must never break the wait that owns it.
        """

        pages = 0
        seen_cursors: set[str] = set()
        attempt_restarted = False
        while terminal or pages < 16:
            if self._log_cursor in seen_cursors:
                return
            seen_cursors.add(self._log_cursor)
            pages += 1
            try:
                payload = self._client.workload_log_page(workload_id, after=self._log_cursor, limit=200)
            except (EyvanUnavailableError, EyvanHTTPError):
                return
            window = _LogWindow.parse(payload)
            if window.attempt_changed:
                # A preemption retry started a new attempt; its output begins
                # at a fresh cursor and the retired attempt's pages are gone.
                # Permit one restart, not an endless sequence of reset pages.
                if attempt_restarted:
                    return
                attempt_restarted = True
                seen_cursors.clear()
                self._log_cursor = ""
                continue
            self._emit_page(window)
            # Display filtering can hide every line on an advancing raw page.
            # Only cursor exhaustion/non-progress proves this drain is done.
            if not window.cursor or window.cursor == self._log_cursor:
                self._log_cursor = window.cursor or self._log_cursor
                return
            self._log_cursor = window.cursor

    def _emit_page(self, window: _LogWindow) -> None:
        """Emit one page of job output, collapsing a chorus into one line.

        A gang says its startup lines once per host, because every host says
        them: a 4x v6e-8 job printed the same EasyDeL banner four times and
        the blocking wait showed every voice. A *run* of byte-identical
        messages from distinct ranks now renders once with its headcount --
        ``(4x Worker)`` -- because which hosts said it is exactly what a
        chorus erases. Two lines that merely look similar do not collapse:
        the key is exact equality, so a genuine straggler or a per-rank value
        stays its own line. A rank repeating *its own* line also stays
        visible, because a repeat is signal, not chorus -- and a page without
        rank data cannot prove distinct voices and never collapses.
        """

        lines = window.lines
        for group in _chorus_groups(tuple((rank, message.rstrip("\n")) for rank, message, _ in lines)):
            index = group[0]
            line = lines[index][1].rstrip("\n")
            if len(group) == 1:
                # Ray-style: a blocking wait decorates each line with the
                # worker that produced it -- pid and host when the archive
                # reports them, nothing invented when it does not.
                provenance = window.provenance[index]
                if provenance[0] is None:
                    self._output.output_line(_job_words(line))
                else:
                    prefix = (
                        f"(Worker pid={provenance[0]}, {provenance[1]}) "
                        if provenance[1]
                        else f"(Worker pid={provenance[0]}) "
                    )
                    # Attribution recedes next to the job's own words: cyan
                    # when a terminal renders it, plain bytes everywhere else.
                    self._output.output_line(_attributed(prefix, line))
            else:
                self._output.output_line(_attributed(f"({len(group)}x Worker) ", line))

    @staticmethod
    def _required_workers(progress: WorkloadProgress) -> int:
        """Return the complete number of native workers required by the request."""

        request = progress.request
        return request.cells * request.workers_per_cell

    @staticmethod
    def _scope(progress: WorkloadProgress) -> str:
        """Describe the requested worker shape and selected provider location."""

        pool = progress.pool
        if pool is None:
            return worker_request(None, _CallableProgressRenderer._required_workers(progress))
        accelerator_type = pool.accelerator_type if pool.accelerator_type is not None else pool.node_type
        location = pool.zone if pool.zone is not None else pool.region
        return f"{worker_request(accelerator_type, _CallableProgressRenderer._required_workers(progress))} | {location}"

    @staticmethod
    def _ready_workers(progress: WorkloadProgress) -> str:
        """Return the current native-worker readiness fraction."""

        return f"{progress.capacity.ready_workers}/{_CallableProgressRenderer._required_workers(progress)} workers ready"

    @staticmethod
    def _diagnostic(progress: WorkloadProgress) -> str | None:
        """Choose the most specific durable control-plane diagnostic."""

        if progress.retry is not None and progress.retry.diagnostic:
            return progress.retry.diagnostic
        if progress.failure_reason:
            return progress.failure_reason
        return progress.scheduling_reason

    def _progress_update(
        self,
        progress: WorkloadProgress,
        *,
        placement: PlacementMode = PlacementMode.ATOMIC,
    ) -> JobLifecycleUpdate:
        """Map an exhaustive control-plane phase to a terminal lifecycle update."""

        phase = progress.phase
        allocation = _allocation(placement, progress.request.cells)
        required_workers = self._required_workers(progress)
        ready_workers = self._ready_workers(progress)
        diagnostic = self._diagnostic(progress)
        scope = self._scope(progress)

        if phase is WorkloadProgressPhase.WAITING_FOR_AUTOSCALER:
            return JobLifecycleUpdate(
                JobLifecycleState.REQUESTING,
                scope,
                f"{ready_workers} | requesting {allocation}",
            )
        if phase is WorkloadProgressPhase.REQUESTING_CAPACITY:
            return JobLifecycleUpdate(
                JobLifecycleState.REQUESTING,
                scope,
                f"{ready_workers} | sending {_allocation_requests(placement, progress.request.cells)} to GCP",
            )
        if phase is WorkloadProgressPhase.WAITING_FOR_CAPACITY:
            waiting = progress.capacity.waiting_workers
            if waiting == 0:
                summary = f"{ready_workers} | waiting for GCP capacity"
            else:
                summary = f"{ready_workers} | queued in GCP's capacity queue"
            return JobLifecycleUpdate(JobLifecycleState.QUEUED, scope, summary)
        if phase is WorkloadProgressPhase.PROVISIONING:
            provisioning = progress.capacity.provisioning_workers
            if provisioning:
                provisioned = f"the {_ATOMIC_ALLOCATION}" if placement is PlacementMode.ATOMIC else allocation
                summary = f"{ready_workers} | GCP is provisioning {provisioned}"
            else:
                summary = ready_workers
            return JobLifecycleUpdate(JobLifecycleState.PROVISIONING, scope, summary)
        if phase is WorkloadProgressPhase.CAPACITY_ALLOCATED:
            return JobLifecycleUpdate(
                JobLifecycleState.ALLOCATED,
                scope,
                f"{ready_workers} | GCP allocated capacity; TPU workers are starting",
            )
        if phase is WorkloadProgressPhase.NATIVE_WORKERS_CONNECTING:
            observed_tpu_vms = progress.capacity.observed_tpu_vms
            detail = "capacity allocated; native workers are connecting"
            if observed_tpu_vms is not None:
                detail = f"GCP reported {_tpu_vm_count(observed_tpu_vms)}; native workers are connecting"
            return JobLifecycleUpdate(
                JobLifecycleState.CONNECTING,
                scope,
                f"{ready_workers} | {detail}",
            )
        if phase is WorkloadProgressPhase.WAITING_FOR_SCHEDULER:
            return JobLifecycleUpdate(JobLifecycleState.READY, scope, f"{ready_workers} | waiting for placement")
        if phase is WorkloadProgressPhase.WAITING_FOR_DISPATCH:
            return JobLifecycleUpdate(JobLifecycleState.DISPATCHING, scope, "capacity leased | waiting for dispatch")
        if phase is WorkloadProgressPhase.READY_FOR_DISPATCH:
            return JobLifecycleUpdate(JobLifecycleState.DISPATCHING, scope, f"{ready_workers} | dispatching workload")
        if phase is WorkloadProgressPhase.DISPATCHING:
            return JobLifecycleUpdate(JobLifecycleState.DISPATCHING, scope, "starting native workers")
        if phase is WorkloadProgressPhase.RUNNING:
            return JobLifecycleUpdate(JobLifecycleState.RUNNING, scope, f"{ready_workers} | workload is running")
        if phase is WorkloadProgressPhase.CHECKPOINTING:
            return JobLifecycleUpdate(
                JobLifecycleState.CHECKPOINTING,
                scope,
                "writing checkpoint before releasing capacity",
            )
        if phase is WorkloadProgressPhase.CANCELLING:
            return JobLifecycleUpdate(JobLifecycleState.CANCELLING, scope, "releasing workload capacity")
        if phase is WorkloadProgressPhase.PREEMPTED:
            lost = progress.capacity.lost_workers
            if lost == 0:
                summary = f"Spot capacity lost | re-requesting {worker_count(required_workers)}"
            else:
                summary = f"Spot capacity lost: {worker_count(lost)} | re-requesting {worker_count(required_workers)}"
            return JobLifecycleUpdate(
                JobLifecycleState.PREEMPTED,
                scope,
                summary,
                diagnostic,
            )
        if phase is WorkloadProgressPhase.QUOTA_BLOCKED:
            return JobLifecycleUpdate(
                JobLifecycleState.BLOCKED,
                scope,
                "capacity request blocked by quota",
                diagnostic,
            )
        if phase is WorkloadProgressPhase.PROVIDER_FAILURE:
            # A rejected capacity attempt on a pending workload is not the
            # job's death. BLOCKED distinguishes a retryable capacity refusal
            # from the terminal FAILED receipt: the plane is still trying and
            # the attempt is counted. When the acquisition budget runs out the workload
            # settles on its own terminal verdict, and that is the receipt
            # that means it.
            if is_acquisition_backoff_reason(progress.scheduling_reason):
                # The pool's last requests were all refused and it is waiting
                # before the next one. Show the pool-level error and retry time;
                # the workload attempt count alone hides failures across fresh cells.
                return JobLifecycleUpdate(
                    JobLifecycleState.BLOCKED,
                    scope,
                    "capacity requests keep failing; the plane is backing off before asking again",
                    progress.scheduling_reason,
                )
            retry = progress.retry
            attempt = f" (attempt {retry.attempts})" if retry is not None and retry.attempts else ""
            return JobLifecycleUpdate(
                JobLifecycleState.BLOCKED,
                scope,
                f"capacity request rejected; the plane is retrying{attempt}",
                diagnostic,
            )
        if phase is WorkloadProgressPhase.QUEUE_ADMISSION_DENIED:
            target = progress.queue_admission_denial
            denial_scope = worker_request(None if target is None else target.accelerator_type, required_workers)
            if target is not None:
                denial_scope = f"{denial_scope} | {target.zone}"
            return JobLifecycleUpdate(
                JobLifecycleState.BLOCKED,
                denial_scope,
                "GCP denied queue admission; no queued resource was created",
                diagnostic,
            )
        if phase is WorkloadProgressPhase.TERMINAL:
            return self._terminal_progress_update(progress)
        raise AssertionError(f"unhandled workload progress phase {phase.value!r}")

    _terminal_workload_update = staticmethod(terminal_workload_update)

    @staticmethod
    def _fallback_or_terminal_update(workload: WorkloadRecord) -> JobLifecycleUpdate:
        """Retain terminal output when the optional progress endpoint is unavailable."""

        if workload.state in TERMINAL_STATES:
            return _CallableProgressRenderer._terminal_workload_update(workload)
        return _CallableProgressRenderer._fallback_update(workload)

    def _terminal_progress_update(self, progress: WorkloadProgress) -> JobLifecycleUpdate:
        """Render the terminal phase while preserving the selected pool context."""

        scope = self._scope(progress)
        if progress.workload_state is WorkloadState.SUCCEEDED:
            return JobLifecycleUpdate(JobLifecycleState.SUCCEEDED, scope, "job completed")
        if progress.workload_state is WorkloadState.FAILED:
            return JobLifecycleUpdate(JobLifecycleState.FAILED, scope, "job failed", progress.failure_reason)
        if progress.workload_state is WorkloadState.CANCELLED:
            return JobLifecycleUpdate(JobLifecycleState.CANCELLED, scope, "job cancelled")
        raise AssertionError(
            f"workload progress for {progress.workload_id!r} is terminal with {progress.workload_state.value!r}"
        )

    _fallback_update = staticmethod(running_workload_update)


class _SdkJobBackend:
    """Adapt synchronous workload operations to the callable job protocol."""

    def __init__(
        self,
        client: EyvanClient,
        *,
        terminal_output: JobTerminalOutput,
        queue_admission_recorder: QueueAdmissionDenialRecorder | None,
    ) -> None:
        self.client = client
        self._terminal_output = terminal_output
        self._queue_admission_recorder = queue_admission_recorder

    @staticmethod
    def _completion(workload: WorkloadRecord) -> JobCompletion:
        return JobCompletion(
            state=workload.state,
            succeeded=workload.state is WorkloadState.SUCCEEDED,
            message=workload.failure_reason or workload.scheduling_reason,
        )

    def wait(self, job_id: str, timeout: float | None = None) -> JobCompletion:
        """Wait for the workload's terminal control-plane state."""

        workload = self.client.wait_workload(job_id, timeout=timeout)
        if self._queue_admission_recorder is not None:
            self._queue_admission_recorder.record_workload(workload)
        return self._completion(workload)

    def wait_with_reclaim_updates(self, job_id: str, timeout: float | None = None) -> JobCompletion:
        """Wait for terminal state while surfacing live reclaim notices on stderr."""

        last_reclaim_message: str | None = None

        def on_update(workload: WorkloadRecord) -> None:
            nonlocal last_reclaim_message
            message = format_reclaim_status(workload)
            if message is not None and message != last_reclaim_message:
                self._terminal_output.note(message)
                last_reclaim_message = message

        workload = self.client.wait_workload(job_id, timeout=timeout, on_update=on_update)
        if self._queue_admission_recorder is not None:
            self._queue_admission_recorder.record_workload(workload)
        return self._completion(workload)

    def wait_with_progress(self, job_id: str, timeout: float | None = None) -> JobCompletion:
        """Wait for a callable workload while rendering durable lifecycle transitions.

        Ctrl-C detaches this client; it does not cancel. The workload is durable
        and nothing here owns it, so a client that cancelled on Ctrl-C would
        destroy a training run because somebody wanted their terminal back. The
        follower has always said this; a callable raised a KeyboardInterrupt
        traceback through ``time.sleep`` instead, which reads exactly like the
        job died with it.
        """

        renderer = _CallableProgressRenderer(self.client, self._terminal_output, self._queue_admission_recorder)
        try:
            workload = self.client.wait_workload(job_id, timeout=timeout, on_update=renderer)
        except KeyboardInterrupt:
            self._terminal_output.detached(job_id)
            raise SystemExit(EXIT_DETACHED) from None
        if self._queue_admission_recorder is not None:
            self._queue_admission_recorder.record_workload(workload)
        return self._completion(workload)

    def status(self, job_id: str) -> WorkloadState:
        """Return the workload's current typed state."""

        workload = self.client.get_workload(job_id)
        if self._queue_admission_recorder is not None:
            self._queue_admission_recorder.record_workload(workload)
        return workload.state

    def logs(self, job_id: str) -> str:
        """Return currently available workload logs."""

        return self.client.workload_logs(job_id)

    def cancel(self, job_id: str) -> WorkloadRecord:
        """Request cancellation and return the updated workload record."""

        workload = self.client.workload_action(job_id, "cancel")
        if self._queue_admission_recorder is not None:
            self._queue_admission_recorder.record_workload(workload)
        return workload


class EyvanCallableSubmitter(tp.Generic[R]):
    """Submit unsigned decorator invocations through the standalone Eyvan HTTP API."""

    def __init__(
        self,
        client: EyvanClient,
        artifact_store: ArtifactStore,
        *,
        signing_key: bytes | None = None,
        terminal_output: JobTerminalOutput | None = None,
        queue_admission_cache: AutoscaleConfigCache | None = None,
    ) -> None:
        reject_remote_callable_signing_key(signing_key)
        self.client = client
        self.artifact_store = artifact_store
        self._terminal_output = terminal_output if terminal_output is not None else JobTerminalOutput()
        recorder = None if queue_admission_cache is None else QueueAdmissionDenialRecorder(queue_admission_cache)
        self._backend = _SdkJobBackend(
            client,
            terminal_output=self._terminal_output,
            queue_admission_recorder=recorder,
        )

    def _reserved_substitute(self, request: CallableSubmission, error: EyvanHTTPError) -> CapacityClass | None:
        """Return RESERVED when an unnamed class was refused by a reserved-only fleet.

        Only an unset ``capacity_class`` qualifies: a class the caller named is
        exact. Only RESERVED is substituted -- capacity the operator already holds
        and pays for either way -- never on-demand, which would start billing for
        something nobody asked for. Any doubt (another status, an unreadable
        listing, the requested class actually present) keeps the original error.
        """

        if getattr(request.tpu, "capacity_class_explicit", True) or error.status_code != 409:
            return None
        try:
            pools = self.client.list_capacity(
                org_id=request.org_id,
                project_id=request.project_id,
                accelerator_type=request.tpu.type,
            )
        except (EyvanHTTPError, EyvanUnavailableError):
            return None
        offered = {pool.capacity_class for pool in pools if pool.fleet == request.fleet and pool.enabled}
        if request.tpu.capacity_class in offered:
            return None
        if CapacityClass.RESERVED in offered:
            return CapacityClass.RESERVED
        return _existing_hardware_class(error, request)

    def _fleet_owning_pool(self, request: CallableSubmission) -> str | None:
        """Return the fleet when ``request.fleet`` is actually one of its pool ids.

        A pool id is the fleet name plus a suffix (``tpu-x-c56c04d2-v5p-128``),
        printed beside the fleet in every listing, so it is the easiest wrong
        name to copy. The match is exact and must be unique; anything else keeps
        the original placement error.
        """

        try:
            pools = self.client.list_capacity(org_id=request.org_id, project_id=request.project_id)
        except (EyvanHTTPError, EyvanUnavailableError):
            return None
        owners = {pool.fleet for pool in pools if pool.pool == request.fleet and pool.fleet != request.fleet}
        return owners.pop() if len(owners) == 1 else None

    def _create_workload(self, request: CallableSubmission, key: str) -> WorkloadRecord:
        """Submit once, running an unnamed class on the fleet's reservation when it has no other."""

        execution = CallableExecution(
            artifact_uri=request.artifact.uri,
            sha256=request.artifact.sha256,
            python=request.artifact.python,
            format_version=request.artifact.format_version,
            signature=request.artifact.signature,
            result_uri=request.result_uri,
            workspace_uri=request.workspace_uri,
            workspace_sha256=request.workspace_sha256,
            workspace_signature=request.workspace_signature,
        )

        def submit(capacity_class: CapacityClass, idempotency_key: str) -> WorkloadRecord:
            return self.client.submit_pool_callable(
                request.fleet,
                request.tpu.type,
                execution,
                org_id=request.org_id,
                project_id=request.project_id,
                pods=request.tpu.pods,
                capacity_class=capacity_class,
                name=request.callable_name[:200],
                metadata={"eyvan_invocation_id": request.invocation_id},
                secret_mounts=request.tpu.secret_mounts,
                max_retries=request.tpu.max_retries,
                max_failure_retries=request.tpu.max_failure_retries,
                max_preemption_retries=request.tpu.max_preemption_retries,
                checkpointable=request.tpu.checkpointable,
                checkpoint_hook=request.tpu.checkpoint_hook,
                checkpoint_output_uri=request.tpu.checkpoint_output_uri,
                checkpoint_grace_seconds=getattr(request.tpu, "reclaim_grace_seconds", DEFAULT_RECLAIM_GRACE_SECONDS),
                reclaim_policy=getattr(request.tpu, "reclaim_policy", None),
                can_submit_children=request.tpu.can_submit_children,
                dependencies=tuple(request.tpu.dependencies),
                atomic=getattr(request.tpu, "atomic", False),
                tpu_chips=getattr(request.tpu, "chip_request", None),
                setup=getattr(request.tpu, "setup_request", None),
                cross_region_transfer_policy=getattr(request.tpu, "cross_region_transfer_policy", None),
                idempotency_key=idempotency_key,
            )

        try:
            return submit(request.tpu.capacity_class, key)
        except EyvanHTTPError as error:
            if _placement_error(error, config=request.tpu) is not None:
                raise
            substitute = self._reserved_substitute(request, error)
            if substitute is None:
                raise
            self._terminal_output.note(
                f"fleet {request.fleet} offers {request.tpu.type} only as {substitute.value} capacity, "
                f"not {request.tpu.capacity_class.value}; running there because no capacity_class was set "
                f"(set capacity_class explicitly to require {request.tpu.capacity_class.value})"
            )
            # A distinct key: the plane fingerprints the class, so replaying the
            # refused key with another class would be reported as a key reuse.
            return submit(substitute, f"{key}:{substitute.value}")

    def submit_callable(self, request: CallableSubmission) -> EyvanJob[R]:
        """Create a callable workload and bind its durable result reader."""

        reject_remote_callable_signature(request.artifact.signature)
        try:
            workload = self._create_workload(request, request.invocation_id)
        except EyvanHTTPError as error:
            placement_error = _placement_error(error, config=request.tpu)
            if placement_error is None:
                raise
            # A name with pools of its own is a real fleet lacking this type,
            # not a mistyped pool id; only an empty answer warrants the lookup.
            owner = None if placement_error.available_accelerator_types else self._fleet_owning_pool(request)
            if owner is None:
                raise placement_error from None
            self._terminal_output.note(
                f"{request.fleet} is a pool of fleet {owner}, not a fleet; submitting to {owner} "
                f'(use init(fleet="{owner}") to name it directly)'
            )
            request = replace(request, fleet=owner)
            try:
                # Fleet name is part of the plane's fingerprint, so the corrected
                # request carries its own key rather than replaying the refused one.
                workload = self._create_workload(request, f"{request.invocation_id}:fleet")
            except EyvanHTTPError as retry_error:
                retried = _placement_error(retry_error, config=request.tpu)
                if retried is None:
                    raise
                raise retried from None

        self._terminal_output.submission(
            JobSubmissionReceipt(
                workload_id=workload.id,
                fleet=request.fleet,
                resource_shape=worker_request(request.tpu.type, request.tpu.pods),
                atomic=getattr(request.tpu, "atomic", False),
            )
        )

        workload_execution = workload.spec.execution
        result_uri = (
            workload_execution.result_uri if isinstance(workload_execution, CallableExecution) else request.result_uri
        )

        def read_result(uri: str) -> R:
            current_execution = self.client.get_workload(workload.id).spec.execution
            if isinstance(current_execution, CallableExecution) and current_execution.result_uri:
                uri = current_execution.result_uri
            return tp.cast(R, load_result(self.artifact_store.read(uri)))

        return EyvanJob(
            id=workload.id,
            result_uri=result_uri or request.result_uri,
            backend=self._backend,
            result_reader=read_result,
            progress_waiter=self._backend.wait_with_progress,
        )


__all__ = ("EyvanCallableSubmitter",)
