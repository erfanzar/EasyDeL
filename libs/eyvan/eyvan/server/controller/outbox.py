# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Claiming, delivering, and recovering durable outbox commands."""

from __future__ import annotations

import asyncio
import logging
import typing as tp
from datetime import timedelta

from pydantic import ValidationError

from eyvan._internal.correlation import traced_operation
from eyvan._internal.observability import trace_span
from eyvan._internal.trace_context import remote_trace_parent
from eyvan.capacity.audit import CapacityLifecycleEvent
from eyvan.capacity.drivers.base import CapacityProviderFailureCode
from eyvan.records import (
    CellState,
    OutboxRecord,
    OutboxState,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.secrets import redact_secret_material
from eyvan.server.controller._support import (
    _RETRIABLE_OUTBOX_STATES,
    CapacityApplySuperseded,
    StaleOutboxCommand,
    _DispatchLeaseMembership,
    logger,
)
from eyvan.server.service.errors import ConflictError, NotFoundError
from eyvan.worker.agent.client import (
    CellAgentBusyError,
    CellAgentCapacityProviderError,
    CellAgentInvalidConfigurationError,
    CellAgentStaleMutationError,
    CellAgentUnavailableError,
    CellAgentUnsupportedCapabilityError,
)

_SECRET_OUTPUT_SUPPRESSED = "output suppressed for secret-bearing execution"


#: Ceiling on the delay between retries of a transient failure. Lower than the
#: non-transient ceiling on purpose: something is waiting on these -- a cancel, a
#: dispatch -- and the condition is expected to clear.
_TRANSIENT_RETRY_CEILING_S = 60


#: How long a command may keep failing before its log line stops being routine.
#: A cancel that has not landed in five minutes is not a blip, and an operator
#: reading warnings at this rate will not notice one more of them.
_STUCK_COMMAND_S = 300.0


def _exception_origin(exc: BaseException) -> str:
    """Name an exception's class and the innermost Eyvan frame that raised it.

    Both are code identifiers, never data, so they are safe to keep when a
    secret-bearing workload's message must be suppressed. They locate the fault
    without exposing the suppressed exception message.
    """

    location = ""
    traceback = exc.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        module = str(frame.f_globals.get("__name__", ""))
        if module.startswith("eyvan."):
            location = f" in {module}.{frame.f_code.co_name}:{traceback.tb_lineno}"
        traceback = traceback.tb_next
    return f"{type(exc).__name__}{location}"


def _suppressed_diagnostic(origin: str | None) -> str:
    """Return the secret-safe diagnostic, keeping the failure's code location."""

    return _SECRET_OUTPUT_SUPPRESSED if origin is None else f"{_SECRET_OUTPUT_SUPPRESSED} ({origin})"


def transient_retry_delay_s(age_s: float) -> int:
    """Return how long to wait before retrying a command failing transiently.

    Grows with the command's own age rather than with an attempt count, because
    a transient failure deliberately gives the attempt back -- an agent
    restarting mid-dispatch must not spend a workload's retry budget. Nothing
    else grew, so a condition that only looked transient retried once a second
    indefinitely: watched live for twenty minutes against a worker that answered
    500 to every stop, while four TPU VMs stayed leased.

    A blip still comes back in a second. A command that has been failing for
    minutes backs off, to a ceiling low enough that recovery is still noticed
    promptly -- something is waiting on these.
    """

    return min(_TRANSIENT_RETRY_CEILING_S, max(1, int(max(0.0, age_s) // 30)))


def command_is_stuck(age_s: float) -> bool:
    """Return whether a failing command has stopped being a routine retry."""

    return age_s >= _STUCK_COMMAND_S


class _ControllerOutbox:
    """Claiming, delivering, and recovering durable outbox commands.

    A concern of :class:`~eyvan.server.controller.EyvanController`; not
    instantiated on its own.
    """

    async def dispatch_outbox_once(self, *, limit: int = 20) -> int:
        """Deliver claimable commands, overlapping aggregates and ordering within one.

        Delivery was strictly serial, and one command can legitimately take
        minutes: a preemption checkpoints a real training step and waits for it.
        Everything else on the replica queued behind it -- capacity applies,
        dispatches, cancellations -- so one slow victim stalled a whole fleet's
        mutations. No test saw it, because correctness was never in question and
        only throughput suffered.

        Two properties constrain how this may be parallelised, and both are kept.

        A command is claimed only by a worker about to run it, never in a batch up
        front. A controller that claims work it has not started is hoarding: if it
        dies, those commands wait out the claim TTL before anyone else may take
        them, and a peer that could have delivered them immediately sees nothing
        available.

        Commands for one aggregate are delivered one at a time, because a cell's
        drain must not overtake its apply. Ordering was never required *between*
        aggregates, which is where all the available parallelism is. Ids are
        consumed in sequence order, so the earlier command of an aggregate is the
        one claimed first.
        """

        if limit <= 0:
            return 0
        command_ids = list(await self.repository.snapshot_claimable_outbox_ids(limit=limit))
        if not command_ids:
            return 0

        pending = iter(command_ids)
        aggregate_locks: dict[tuple[str, str], asyncio.Lock] = {}
        delivered = 0

        async def _worker() -> None:
            nonlocal delivered
            while True:
                command_id = next(pending, None)
                if command_id is None:
                    return
                command = await self.repository.claim_outbox_id(
                    worker_id=self.worker_id,
                    command_id=command_id,
                    lease_seconds=self.settings.outbox_claim_ttl_s,
                )
                if command is None:
                    continue
                key = (command.aggregate_type, command.aggregate_id)
                lock = aggregate_locks.setdefault(key, asyncio.Lock())
                async with lock:
                    await self._dispatch_claimed_outbox(command)
                delivered += 1

        workers = min(max(1, self.settings.outbox_dispatch_concurrency), len(command_ids))
        # ``gather`` rather than a task group: one worker raising past the
        # handlers in ``_dispatch_claimed_outbox`` must not cancel the others,
        # whose commands are unrelated and already claimed.
        outcomes = await asyncio.gather(*(_worker() for _ in range(workers)), return_exceptions=True)
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                logger.exception("outbox delivery worker failed", exc_info=outcome)
        return delivered

    async def _dispatch_claimed_outbox(self, command: OutboxRecord) -> None:
        """Deliver and settle one command that this controller has just claimed."""

        # Recovery can enqueue a replacement command after dispatch fails. Keep
        # the producer parent attached until those handlers and settlement finish.
        with remote_trace_parent(
            command.traceparent,
            enabled=self.settings.correlation_enabled and command.correlation_id is not None,
        ):
            await self._dispatch_claimed_outbox_in_context(command)

    async def _dispatch_claimed_outbox_in_context(self, command: OutboxRecord) -> None:
        """Run delivery and its recovery handlers under the command's parent."""

        try:
            # Delivery is the one place a provider or cell agent can block for
            # minutes, and ``topic`` is exactly the "driver operation" that
            # ``eyvan_reconcile_seconds`` was declared to be labeled by. The
            # histogram and the span existed with no call site anywhere, so
            # neither a slow provider nor a slow agent produced any signal.
            with (
                self._measure_reconcile(command.topic),
                traced_operation("outbox." + command.topic, command.correlation_id),
                trace_span(
                    "eyvan.outbox_dispatch",
                    topic=command.topic,
                    command_id=command.id,
                ),
            ):
                await self._dispatch_with_claim_heartbeat(command)
        except CapacityApplySuperseded as exc:
            logger.info("discarding superseded capacity apply %s: %s", command.id, exc)
            await self._discard_outbox(command, provider_diagnostic=exc.provider_diagnostic)
        except StaleOutboxCommand as exc:
            logger.info("discarding stale outbox command %s: %s", command.id, exc)
            await self._finish_outbox(command)
        except CellAgentStaleMutationError as exc:
            diagnostic = await self._safe_outbox_diagnostic(command, str(exc))
            logger.info("discarding cell-agent stale mutation %s: %s", command.id, diagnostic)
            if command.topic == "capacity.apply":
                await self._record_cell_agent_fence_conflict(
                    cell_id=command.aggregate_id,
                    generation=command.generation,
                    diagnostic=diagnostic,
                )
                await self._discard_outbox(command, provider_diagnostic=diagnostic)
            elif command.topic == "capacity.drain":
                await self._recover_stale_cell_lifecycle_command(
                    command,
                    expected_state=CellState.DRAINING,
                    diagnostic=diagnostic,
                )
            elif command.topic == "capacity.release":
                await self._recover_stale_cell_lifecycle_command(
                    command,
                    expected_state=CellState.DELETING,
                    diagnostic=diagnostic,
                )
            elif command.topic == "workload.dispatch":
                await self._recover_stale_workload_command(
                    command,
                    expected_state=WorkloadState.LEASED,
                    diagnostic=diagnostic,
                )
            elif command.topic == "workload.cancel":
                await self._recover_stale_workload_command(
                    command,
                    expected_state=WorkloadState.CANCELLING,
                    diagnostic=diagnostic,
                )
            elif command.topic == "workload.checkpoint":
                await self._recover_stale_workload_command(
                    command,
                    expected_state=WorkloadState.CHECKPOINTING,
                    diagnostic=diagnostic,
                )
            elif command.topic == "workload.preempt":
                await self._recover_stale_preemption_command(command, diagnostic=diagnostic)
            elif command.topic == "workload.reclaim":
                await self._recover_stale_reclaim_command(command, diagnostic=diagnostic)
            else:
                await self._discard_outbox(command, provider_diagnostic=diagnostic)
        except CellAgentBusyError as exc:
            await self._finish_outbox(command, error=str(exc), transient=True, error_origin=_exception_origin(exc))
        except CellAgentUnavailableError as exc:
            # Transient, and specifically *not* a provider verdict. An agent that
            # cannot be reached has told us nothing about the provider, and this
            # fell into the bare handler below -- which spends an attempt and, for
            # ``capacity.apply``, records a PROVIDER_ERROR against the cell. A
            # restarting agent or a network blip therefore became durable evidence
            # that GCP had failed, which is how a healthy pool acquires a failure
            # it never had. Unknown is a third answer here, exactly as it is
            # everywhere else in this package.
            await self._finish_outbox(command, error=str(exc), transient=True, error_origin=_exception_origin(exc))
        except CellAgentUnsupportedCapabilityError as exc:
            # The backend does not implement this operation; retrying cannot
            # change that. Secret-bearing diagnostics are suppressed before
            # either logging or durable settlement.
            diagnostic = await self._safe_outbox_diagnostic(command, str(exc))
            logger.warning(
                "cell agent does not support %s for command %s: %s",
                command.topic,
                command.id,
                diagnostic,
            )
            await self._finish_outbox(
                command,
                error=f"unsupported by cell agent backend: {diagnostic}",
                terminal=True,
            )
        except CellAgentInvalidConfigurationError as exc:
            diagnostic = await self._safe_outbox_diagnostic(command, str(exc))
            logger.warning("cell agent rejected the configuration for command %s: %s", command.id, diagnostic)
            await self._finish_outbox(command, error=f"invalid configuration: {diagnostic}", terminal=True)
        except CellAgentCapacityProviderError as exc:
            await self._finish_outbox(
                command, error=str(exc), failure_code=exc.code, error_origin=_exception_origin(exc)
            )
        except Exception as exc:
            await self._finish_outbox(command, error=str(exc), error_origin=_exception_origin(exc))
        else:
            await self._finish_outbox(command)

    def _outbox_claim_heartbeat_s(self) -> float:
        """Return a bounded renewal cadence for long-running outbox commands."""

        claim_ttl_s = max(self.settings.outbox_claim_ttl_s, 1)
        return min(30.0, max(0.1, claim_ttl_s / 3))

    @staticmethod
    def _consume_background_dispatch_result(task: asyncio.Task[None]) -> None:
        """Consume a detached RPC result after its durable claim was lost."""

        if task.cancelled():
            return
        try:
            task.exception()
        except asyncio.CancelledError:
            return

    async def _dispatch_with_claim_heartbeat(self, command: OutboxRecord) -> None:
        """Deliver one command while keeping its durable ownership lease alive.

        Provider creates can legitimately run for minutes. The command remains
        idempotent, but a process that dies during the RPC must be reclaimable
        soon rather than hiding a pending workload behind a long claim TTL.
        """

        active_command = await self._renew_outbox_claim(command)
        dispatch = asyncio.create_task(self._dispatch(active_command))
        try:
            while not dispatch.done():
                completed, _ = await asyncio.wait((dispatch,), timeout=self._outbox_claim_heartbeat_s())
                if not completed:
                    active_command = await self._renew_outbox_claim(active_command)
            await dispatch
        except asyncio.CancelledError:
            if not dispatch.done():
                dispatch.add_done_callback(self._consume_background_dispatch_result)
            raise
        except Exception:
            if not dispatch.done():
                dispatch.add_done_callback(self._consume_background_dispatch_result)
            raise

    async def _renew_outbox_claim(self, command: OutboxRecord) -> OutboxRecord:
        """Extend this controller's current claim without reviving a stale command."""

        async with self.repository.transaction() as transaction:
            current = await transaction.get_outbox(command.id)
            if current is None:
                raise StaleOutboxCommand(f"outbox command {command.id!r} no longer exists")
            now = utcnow()
            if (
                current.state is not OutboxState.PROCESSING
                or current.claimed_by != self.worker_id
                or current.claimed_until is None
                or current.claimed_until <= now
                or current.topic != command.topic
                or current.aggregate_type != command.aggregate_type
                or current.aggregate_id != command.aggregate_id
                or current.generation != command.generation
            ):
                raise StaleOutboxCommand(f"outbox command {command.id!r} claim expired or moved to another worker")
            renewed = current.model_copy(
                update={
                    "claimed_until": now + timedelta(seconds=max(self.settings.outbox_claim_ttl_s, 1)),
                    "updated_at": now,
                }
            )
            await transaction.put_outbox(renewed)
            return renewed

    async def _dispatch(self, command: OutboxRecord) -> None:
        if command.topic == "capacity.apply":
            await self._dispatch_capacity(command)
        elif command.topic == "capacity.drain":
            await self._dispatch_capacity_drain(command)
        elif command.topic == "capacity.release":
            await self._dispatch_capacity_release(command)
        elif command.topic == "workload.dispatch":
            await self._dispatch_workload(command)
        elif command.topic == "workload.cancel":
            await self._dispatch_cancel(command)
        elif command.topic == "workload.checkpoint":
            await self._dispatch_checkpoint(command)
        elif command.topic == "workload.preempt":
            await self._dispatch_preemption(command)
        elif command.topic == "workload.reclaim":
            await self._dispatch_reclaim(command)
        else:
            raise ValueError(f"unsupported outbox topic {command.topic!r}")

    async def _recover_stale_cell_lifecycle_command(
        self,
        command: OutboxRecord,
        *,
        expected_state: CellState,
        diagnostic: str,
    ) -> None:
        """Replace an obsolete drain or release mutation before retiring its outbox record."""

        try:
            previous = await self.service.get_cell(self._principal, command.aggregate_id)
            refenced, replayed = await self.service.refence_cell_lifecycle_command(
                self._principal,
                command.aggregate_id,
                expected_generation=command.generation,
                expected_state=expected_state,
                idempotency_key=f"agent-fence-refence:{command.id}",
            )
        except (ConflictError, NotFoundError) as exc:
            logger.info("stale %s command %s no longer owns its cell: %s", command.topic, command.id, exc)
        except Exception as exc:
            logger.warning("could not refence stale %s command %s: %s", command.topic, command.id, exc)
            await self._finish_outbox(command, error=f"stale mutation recovery failed: {exc}", transient=True)
            return
        else:
            if not replayed:
                self._capacity_lifecycle.event(
                    CapacityLifecycleEvent.RETRY,
                    refenced,
                    previous=previous,
                    reason=f"cell-agent rejected {command.topic}; issued generation {refenced.generation}",
                )
        await self._discard_outbox(command, provider_diagnostic=diagnostic)

    async def _recover_stale_workload_command(
        self,
        command: OutboxRecord,
        *,
        expected_state: WorkloadState,
        diagnostic: str,
    ) -> None:
        """Settle a workload whose agent fence makes remote execution ownership uncertain."""

        raw_lease_id = command.payload.get("lease_id")
        if not isinstance(raw_lease_id, str) or not raw_lease_id:
            await self._discard_outbox(command, provider_diagnostic=diagnostic)
            return
        async with self.repository.transaction() as transaction:
            workload = await transaction.get_workload(command.aggregate_id)
        if (
            workload is None
            or workload.generation != command.generation
            or workload.lease_id != raw_lease_id
            or workload.state is not expected_state
        ):
            await self._discard_outbox(command, provider_diagnostic=diagnostic)
            return
        try:
            members = await self._load_workload_members(workload)
        except StaleOutboxCommand as exc:
            logger.info("stale %s command %s has no current workload members: %s", command.topic, command.id, exc)
            await self._discard_outbox(command, provider_diagnostic=diagnostic)
            return
        if any(lease.generation != command.generation for lease, _cell in members):
            await self._discard_outbox(command, provider_diagnostic=diagnostic)
            return
        for lease, cell in members:
            await self._record_cell_agent_fence_conflict(
                cell_id=cell.id,
                generation=lease.cell_generation,
                diagnostic=f"workload {workload.id!r} {command.topic} fence conflict: {diagnostic}",
            )
        failure_reason = f"cell-agent {command.topic.removeprefix('workload.')} fence conflict: {diagnostic}"
        # Archive before settling, exactly as the ordinary terminal paths do in
        # ``_release_terminal``: settlement releases the lease, and releasing a
        # cell destroys its journal -- cancel/checkpoint victims ran, so the
        # tail of their output exists only right now. Best effort on purpose:
        # a job that ran must still be recorded as finished even if its logs
        # could not be saved.
        try:
            await self._archive_terminal_logs(workload)
        except Exception:
            logger.warning("could not archive logs for %s before release", workload.id, exc_info=True)
        try:
            await self.service.settle_terminal_workload(
                self._principal,
                workload.id,
                WorkloadState.FAILED,
                action="workload.agent_fence_conflict",
                failure_reason=failure_reason,
                expected_generation=workload.generation,
                expected_lease_id=raw_lease_id,
                expected_state=expected_state,
            )
        except (ConflictError, NotFoundError) as exc:
            logger.info("stale %s command %s settlement lost ownership: %s", command.topic, command.id, exc)
        else:
            await self.reconcile_unmanaged_capacity_recoveries_once()
        await self._discard_outbox(command, provider_diagnostic=diagnostic)

    async def _recover_stale_preemption_command(self, command: OutboxRecord, *, diagnostic: str) -> None:
        """Abort a preemption before stopping, or retain it for explicit operator finalization."""

        try:
            aborted = await self._abort_preemption_checkpoint_phase(command, diagnostic)
        except StaleOutboxCommand as exc:
            logger.info("stale preemption command %s already changed ownership: %s", command.id, exc)
            await self._discard_outbox(command, provider_diagnostic=diagnostic)
            return
        if aborted:
            await self._finish_outbox(command)
            return
        logger.error(
            "preemption command %s lost an agent fence after stopping began; explicit force-finalization is required",
            command.id,
        )
        await self._discard_outbox(command, provider_diagnostic=diagnostic)

    @staticmethod
    def _has_live_dispatch_command(
        workload: WorkloadRecord,
        lease_ids: tuple[str, ...],
        commands: tp.Iterable[OutboxRecord],
    ) -> bool:
        """Return whether a retryable command can still submit this exact lease membership.

        A malformed active command is deliberately treated as live. It must be
        retired by normal outbox delivery before releasing its lease, because a
        controller cannot prove that a malformed persisted payload has no
        submission side effect.
        """

        for command in commands:
            if (
                command.topic != "workload.dispatch"
                or command.aggregate_type != "workload"
                or command.aggregate_id != workload.id
                or command.generation != workload.generation
                or command.state not in _RETRIABLE_OUTBOX_STATES
            ):
                continue
            try:
                membership = _DispatchLeaseMembership.model_validate(command.payload)
            except ValidationError:
                return True
            if membership.ids == lease_ids:
                return True
        return False

    async def _safe_outbox_diagnostic(self, command: OutboxRecord, error: str) -> str:
        """Suppress agent/provider output for a declared-secret workload."""

        if command.aggregate_type == "workload":
            async with self.repository.transaction() as transaction:
                workload = await transaction.get_workload(command.aggregate_id)
            if workload is not None and workload.spec.secret_mounts:
                return _SECRET_OUTPUT_SUPPRESSED
        return redact_secret_material(error)

    async def _finish_outbox(
        self,
        command: OutboxRecord,
        *,
        error: str | None = None,
        failure_code: CapacityProviderFailureCode | None = None,
        transient: bool = False,
        terminal: bool = False,
        error_origin: str | None = None,
    ) -> bool:
        """Settle an outbox command only while this controller still owns its live claim.

        ``transient`` retries soon without spending an attempt. ``terminal``
        settles the command for good, keeping its diagnostic: it is for answers
        that will never change on a retry, such as a backend reporting that it
        does not implement the operation at all. ``error_origin`` names where
        the failure was raised; it survives secret suppression, which the
        message itself does not.
        """

        now = utcnow()
        async with self.repository.transaction() as transaction:
            current = await transaction.get_outbox(command.id)
            if (
                current is None
                or current.state is not OutboxState.PROCESSING
                or current.claimed_by != self.worker_id
                or current.claimed_until is None
                or current.claimed_until <= now
            ):
                return False
            if error is not None:
                if current.aggregate_type == "workload":
                    workload = await transaction.get_workload(current.aggregate_id)
                    error = (
                        _suppressed_diagnostic(error_origin)
                        if workload is not None and workload.spec.secret_mounts
                        else redact_secret_material(error)
                    )
                else:
                    error = redact_secret_material(error)
            if error is None:
                if current.attempts > 1 or current.last_error is not None:
                    # A command that failed and then succeeded is about to lose
                    # every trace of why: `last_error` is cleared here and nothing
                    # logged the failures. That combination produced a workload
                    # that sat "waiting for dispatch" for 21 minutes across 10
                    # attempts and then settled with attempts=10 and last_error
                    # NULL -- an operator, and the author of this comment, had no
                    # way to see the cause. Say it once, on the way past.
                    logger.warning(
                        "outbox %s (%s for %s %s) succeeded after %d attempt(s); last failure was: %s",
                        current.id,
                        current.topic,
                        current.aggregate_type,
                        current.aggregate_id,
                        current.attempts,
                        current.last_error or "not retained",
                    )
                updated = current.model_copy(
                    update={
                        "state": OutboxState.SUCCEEDED,
                        "claimed_by": None,
                        "claimed_until": None,
                        "last_error": None,
                        "failure_code": None,
                        "updated_at": now,
                    }
                )
            elif terminal or failure_code in {
                CapacityProviderFailureCode.QUEUE_ADMISSION_DENIED,
                CapacityProviderFailureCode.CONFIGURATION,
            }:
                logger.warning(
                    "outbox %s (%s for %s %s) will not be retried: %s",
                    current.id,
                    current.topic,
                    current.aggregate_type,
                    current.aggregate_id,
                    error,
                )
                updated = current.model_copy(
                    update={
                        "state": OutboxState.DISCARDED,
                        "claimed_by": None,
                        "claimed_until": None,
                        "last_error": error[:2000],
                        "failure_code": failure_code,
                        "updated_at": now,
                    }
                )
            else:
                # A transient failure must not consume the retry budget -- an
                # agent restarting mid-dispatch is not the workload's fault --
                # so `attempts` is given back below. That left nothing at all
                # growing, though: a condition that merely *looks* transient and
                # never recovers retried once a second forever, at an unchanged
                # attempt number, with an identical warning each time. Watched
                # live for twenty minutes while a cancelled multislice held four
                # TPU VMs, and nothing in the plane escalated.
                #
                # Age is what grows in that case, and it needs no new field. A
                # blip still retries in a second; a stuck command backs off and
                # says how long it has been stuck.
                age_s = max(0.0, (now - current.created_at).total_seconds())
                delay = (
                    transient_retry_delay_s(age_s) if transient else min(300, 5 * (2 ** min(current.attempts - 1, 6)))
                )
                # Logged here rather than at each raise site, because the two
                # handlers that matter most -- the capacity-provider error and the
                # bare `except Exception` -- logged nothing at all, so a dispatch
                # could fail indefinitely in complete silence while the client
                # showed a steady "waiting for dispatch".
                stuck = command_is_stuck(age_s)
                logger.log(
                    logging.ERROR if stuck else logging.WARNING,
                    "outbox %s (%s for %s %s) failed on attempt %d after %.0fs, retrying in %ds: %s",
                    current.id,
                    current.topic,
                    current.aggregate_type,
                    current.aggregate_id,
                    current.attempts,
                    age_s,
                    delay,
                    error,
                )
                updated = current.model_copy(
                    update={
                        "state": OutboxState.FAILED,
                        "attempts": max(0, current.attempts - 1) if transient else current.attempts,
                        "claimed_by": None,
                        "claimed_until": None,
                        "last_error": error[:2000],
                        "failure_code": failure_code,
                        "available_at": now + timedelta(seconds=delay),
                        "updated_at": now,
                    }
                )
            await transaction.put_outbox(updated)
        if error is not None and command.topic == "capacity.apply" and not transient:
            await self._record_capacity_failure(command, error, failure_code=failure_code)
        return True

    async def _discard_outbox(self, command: OutboxRecord, *, provider_diagnostic: str | None = None) -> None:
        """Finish a superseded command without erasing its prior provider diagnostic."""

        now = utcnow()
        async with self.repository.transaction() as transaction:
            current = await transaction.get_outbox(command.id)
            if (
                current is None
                or current.state is not OutboxState.PROCESSING
                or current.claimed_by != self.worker_id
                or current.claimed_until is None
                or current.claimed_until <= now
            ):
                return
            retained_diagnostic = current.last_error if current.last_error is not None else provider_diagnostic
            await transaction.put_outbox(
                current.model_copy(
                    update={
                        "state": OutboxState.DISCARDED,
                        "claimed_by": None,
                        "claimed_until": None,
                        "last_error": None if retained_diagnostic is None else retained_diagnostic[:2000],
                        "updated_at": now,
                    }
                )
            )
