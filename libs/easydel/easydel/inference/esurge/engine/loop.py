# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The eSurge scheduler-thread state machine.

:class:`EngineLoop` owns the background thread that drives inference
iterations: drain parser-stop signals, ``schedule()``, execute the model via
the step coordinator, and apply ``update_from_output``. Two loop shapes are
implemented — :meth:`_run_sync` and :meth:`_run_overlap` (async dispatch with
deferred drain) — selected by ``overlap_execution`` and the runner's async
capability. All engine coupling is constructor-injected callables; the
scheduler lock is the engine's own RLock, shared so the lock order contract
(``scheduler_lock -> request_lock -> output_lock``) is unchanged.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
import typing

from easydel.utils import flags

from ..distributed.coordinator import LocalCoordinator, StepHandle
from ..logger import logger
from ..scheduler.dp_scheduler import DPSchedulerClosed

if typing.TYPE_CHECKING:
    from ..scheduler import Scheduler, SchedulerOutput

# Default to fail-fast so benchmark runs don't spin for hours on fatal errors.
# Set `EASYDEL_MAX_SCHEDULER_ERRORS=10` (or higher) to restore retry behavior.
MAX_CONSECUTIVE_SCHEDULER_ERRORS = flags.get_int(flags.EASYDEL_MAX_SCHEDULER_ERRORS)


class EngineLoop:
    """Background scheduler loop with explicit, constructor-injected coupling.

    Owns the daemon thread previously defined as the ``_scheduler_loop``
    closure inside ``eSurge.initiate()``. The engine constructs one
    per :meth:`eSurge.initiate` call and stops it via
    :meth:`request_stop` / :meth:`join`.

    Args:
        scheduler: Live :class:`Scheduler` driving request admission.
        runner: The engine's :class:`eSurgeRunner`.
        coordinator: Step coordinator; ``None`` falls back to a
            :class:`LocalCoordinator` over ``runner`` at loop start.
        scheduler_lock: The engine's scheduler RLock — SHARED with the
            engine, never a new lock.
        emit_outputs: Engine callback queuing engine outputs for the
            output pipeline (``eSurge._enqueue_engine_outputs``).
        drain_parser_stops: Engine callback draining queued parser stop
            signals; invoked while ``scheduler_lock`` is held.
        on_fatal: Engine callback recording a fatal scheduler error and
            waking waiters (``eSurge._abort_scheduler_due_to_error``).
        heartbeat: Engine callback stamping the scheduler heartbeat.
        handle_profiling_step: Engine callback ticking the profiler
            step budget.
        is_nonrecoverable: Classifier for errors that must bypass the
            retry budget.
        overlap_execution: ``runtime_config.overlap_execution``.
        async_scheduling: ``runtime_config.async_scheduling`` (used for
            the verbose lifecycle log line).
        runner_verbose: ``runtime_config.runner_verbose``.
        info: Engine-level info logger callable.
    """

    def __init__(
        self,
        *,
        scheduler: Scheduler,
        runner,
        coordinator,
        scheduler_lock: threading.RLock,
        emit_outputs: typing.Callable[[typing.Any], None],
        drain_parser_stops: typing.Callable[[], None],
        on_fatal: typing.Callable[[BaseException], None],
        heartbeat: typing.Callable[[], None],
        handle_profiling_step: typing.Callable[[], None],
        is_nonrecoverable: typing.Callable[[BaseException], bool],
        overlap_execution: bool,
        async_scheduling: bool,
        runner_verbose: bool,
        info: typing.Callable[..., None],
    ) -> None:
        self.scheduler = scheduler
        self.runner = runner
        self._coordinator = coordinator
        self._scheduler_lock = scheduler_lock
        self._emit_outputs = emit_outputs
        self._drain_parser_stops = drain_parser_stops
        self._on_fatal = on_fatal
        self._heartbeat = heartbeat
        self._handle_profiling_step = handle_profiling_step
        self._is_nonrecoverable = is_nonrecoverable
        self._overlap_execution = bool(overlap_execution)
        self._async_scheduling = bool(async_scheduling)
        self._runner_verbose = bool(runner_verbose)
        self._info = info

        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        """Whether the loop has been started and not yet asked to stop."""
        return self._running

    @running.setter
    def running(self, value: bool) -> None:
        self._running = bool(value)

    @property
    def thread(self) -> threading.Thread | None:
        """The loop's daemon thread, or ``None`` before start / after join."""
        return self._thread

    def start(self) -> None:
        """Flip the running flag and spawn the daemon scheduler thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def request_stop(self) -> None:
        """Ask the loop to exit after its current iteration."""
        self._running = False

    def join(self, timeout: float | None = None) -> None:
        """Join the loop thread; clears the thread reference once it exits."""
        thread = self._thread
        if thread is None:
            return
        thread.join(timeout=timeout)
        if not thread.is_alive() and self._thread is thread:
            self._thread = None

    @staticmethod
    def _can_prefetch_scheduler_output(scheduler: Scheduler, current: SchedulerOutput) -> bool:
        """Decide whether ``schedule()`` can run while ``current`` is still in flight.

        The overlap path may compute the next batch ahead of time only when
        no decision in that batch depends on the sampled token of the
        current one. Two situations forbid prefetch and force the
        dispatcher to wait for ``update_from_output()`` first:

        * The async scheduler has already inserted output placeholders
          (``num_output_placeholders > 0``) — running ``schedule()`` again
          before those placeholders are filled would either double-budget
          the request or generate stale page-table updates.
        * The request has reached prompt-length parity
          (``num_computed_tokens >= num_tokens``) — its next decision (e.g.
          whether to terminate on EOS) depends on the as-yet-unobserved
          sampled token, so re-entering the scheduler would speculatively
          extend a request that may finish.

        Returns ``False`` defensively whenever exception handling kicks in
        (e.g. if the scheduler maps are mutated mid-iteration), trading
        some throughput for correctness.

        Args:
            scheduler: Live :class:`Scheduler` instance whose
                ``requests`` map is queried per request id.
            current: The :class:`SchedulerOutput` representing the
                in-flight step.

        Returns:
            ``True`` iff every request in ``current`` is in the safe
            "pure prefill, no termination decision yet" state, so the
            next batch can be assembled now.
        """
        try:
            for rid in current.num_scheduled_tokens:
                req = scheduler.requests.get(rid)
                if req is None:
                    continue
                if getattr(req, "num_output_placeholders", 0) > 0:
                    return False
                if req.num_computed_tokens >= req.num_tokens:
                    return False
        except Exception:
            return False
        return True

    @staticmethod
    def _zero_schedule_needs_update(scheduler_output: SchedulerOutput) -> bool:
        """Return whether a zero-token schedule still has bookkeeping to flush."""
        return bool(
            scheduler_output.finished_req_ids
            or scheduler_output.preempted_req_ids
            or scheduler_output.scheduled_new_reqs
            or scheduler_output.scheduled_cached_reqs.num_reqs
        )

    @staticmethod
    def _idle_scheduler_tick() -> None:
        """Back off the background scheduler when there is no executable work."""
        time.sleep(0.001)

    def _overlap_loop_enabled(self) -> bool:
        """Return whether the lifecycle loop should use async handles.

        ``async_scheduling=True`` is not itself enough to remove
        sampled-token materialization from the launch path: the
        synchronous loop calls ``execute_model()``, which must
        resolve the token before returning. The overlap loop calls
        ``execute_model_async()`` instead, then lets the runner's
        capability check decide whether the next launch may be
        issued before draining the previous sampled token. This is
        required for both TP/SPMD and PP decode; otherwise each
        token step is serialized by the host lifecycle even when the
        device kernels themselves are short.
        """
        if not self._overlap_execution:
            return False
        return hasattr(self.runner, "execute_model_async")

    def _drain(self, future, scheduler_output: SchedulerOutput) -> None:
        """Block on an in-flight async step and apply its outputs to the scheduler.

        Used by the overlap loop after dispatching a step via
        :meth:`eSurgeRunner.execute_model_async`. Waits for the host
        copies of sampled tokens to land, runs the scheduler's
        ``update_from_output`` on the resulting :class:`ModelRunnerOutput`,
        forwards engine-side outputs (sampled tokens, finished requests)
        to the output pipeline for client delivery, and decrements the
        profiler step counter when an active trace is running.

        Args:
            future: Async handle returned by
                :meth:`eSurgeRunner.execute_model_async` (an
                ``_AsyncExecutionHandle`` or ``concurrent.futures.Future``),
                possibly wrapped in a :class:`StepHandle`.
            scheduler_output: The :class:`SchedulerOutput` that produced
                ``future``; needed by ``update_from_output`` to map
                sampled tokens back to the right requests.
        """
        wait_start = time.perf_counter()
        if isinstance(future, StepHandle):
            coordinator = self._coordinator
            if coordinator is not None:
                model_output = coordinator.drain(future)
            else:
                model_output = self.runner.wait_for_execution(future.runner_handle)
        else:
            model_output = self.runner.wait_for_execution(future)
        wait_time = time.perf_counter() - wait_start
        total_tokens = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0)
        update_start = time.perf_counter()
        with self._scheduler_lock:
            engine_outputs = self.scheduler.update_from_output(scheduler_output, model_output)
        update_time = time.perf_counter() - update_start
        process_start = time.perf_counter()
        self._emit_outputs(engine_outputs)
        process_time = time.perf_counter() - process_start
        if total_tokens:
            total_time = wait_time + update_time + process_time
            self.runner.log_it(
                "[perf] overlap_drain tok=%d wait=%.2fms update=%.2fms process=%.2fms total=%.2fms wait_tps=%.1f",
                total_tokens,
                wait_time * 1e3,
                update_time * 1e3,
                process_time * 1e3,
                total_time * 1e3,
                total_tokens / wait_time if wait_time > 0 else 0.0,
            )
        self._handle_profiling_step()

    def _run(self) -> None:
        """Background scheduler thread driving inference iterations.

        Two implementations, selected by :meth:`_overlap_loop_enabled`:

        * **Synchronous loop** (:meth:`_run_sync`) — calls ``schedule()``,
          executes the model, applies ``update_from_output()``, and
          forwards engine outputs to the parser worker. Errors are
          absorbed up to ``MAX_CONSECUTIVE_SCHEDULER_ERRORS`` retries;
          some invariant violations (per the injected
          ``is_nonrecoverable`` classifier) abort immediately.
        * **Async-overlap loop** (:meth:`_run_overlap`) — uses async
          runner handles to prefetch the next ``schedule()`` while the
          previous step's device tokens are still in flight, with the
          runner's async-drain capability gate and
          :meth:`_can_prefetch_scheduler_output` gating when this is
          safe. CPU bookkeeping still runs in scheduler order when the
          older handle is drained.

        Both branches publish heartbeats via the injected ``heartbeat``
        callback and forward outputs through ``emit_outputs`` so the
        per-token detokenize/parse path stays off the scheduler thread.
        """
        self._info("Starting background scheduler loop")
        if self._coordinator is None:
            self._coordinator = LocalCoordinator(self.runner)

        use_overlap_loop = self._overlap_loop_enabled()
        if self._runner_verbose:
            logger.info(
                "Using %s eSurge scheduler lifecycle (async_scheduling=%s, overlap_execution=%s)",
                "async-overlap" if use_overlap_loop else "synchronous",
                self._async_scheduling,
                self._overlap_execution,
            )

        if not use_overlap_loop:
            self._run_sync()
            return
        self._run_overlap()

    def _run_sync(self) -> None:
        """Synchronous loop body: schedule → execute_sync → update, in order."""
        coordinator = self._coordinator
        consecutive_errors = 0
        max_consecutive_errors = MAX_CONSECUTIVE_SCHEDULER_ERRORS

        _diag_iter = 0
        _diag_last_log = time.time()
        _diag_debug = int(getattr(logger, "level", logging.INFO)) <= logging.DEBUG

        while self._running:
            try:
                _diag_iter += 1
                prof_loop_t0 = time.perf_counter()
                prof_phase = prof_loop_t0
                with self._scheduler_lock:
                    self._drain_parser_stops()
                    scheduler_output = self.scheduler.schedule()
                prof_sched = time.perf_counter() - prof_phase
                _n = len(scheduler_output.num_scheduled_tokens) if scheduler_output else 0
                _now = time.time()
                # ``scheduler.running``/``waiting`` are RPC snapshots under
                # DPScheduler worker processes: each access pickles a rank's
                # full EngineRequest list back to the coordinator. Python
                # evaluates logger arguments eagerly, so this diagnostic must
                # stay behind an explicit level check or every decode step
                # pays for a log line that is then discarded.
                if _diag_debug and (_n > 0 or (_now - _diag_last_log) > 30):
                    logger.debug(
                        "loop iter=%d sched=%d run=%d wait=%d",
                        _diag_iter,
                        _n,
                        len(self.scheduler.running),
                        len(self.scheduler.waiting),
                    )
                    _diag_last_log = _now
                self._heartbeat()
                coordinator.check_health()
                if scheduler_output.total_num_scheduled_tokens == 0 and not self._zero_schedule_needs_update(
                    scheduler_output
                ):
                    self._idle_scheduler_tick()
                    consecutive_errors = 0
                    continue
                prof_dispatch = 0.0
                prof_phase = time.perf_counter()
                model_output = coordinator.execute_sync(scheduler_output)
                prof_execute = time.perf_counter() - prof_phase
                prof_phase = time.perf_counter()
                with self._scheduler_lock:
                    engine_outputs = self.scheduler.update_from_output(scheduler_output, model_output)
                prof_update = time.perf_counter() - prof_phase
                prof_phase = time.perf_counter()
                self._emit_outputs(engine_outputs)
                prof_process = time.perf_counter() - prof_phase
                if self._runner_verbose and scheduler_output.total_num_scheduled_tokens > 0:
                    logger.info(
                        "[esurge-prof-core] it=%06d tok=%d reqs=%d sched=%.3fms "
                        "dispatch=%.3fms execute=%.3fms update=%.3fms process=%.3fms total=%.3fms",
                        _diag_iter,
                        int(scheduler_output.total_num_scheduled_tokens),
                        len(scheduler_output.num_scheduled_tokens),
                        prof_sched * 1e3,
                        prof_dispatch * 1e3,
                        prof_execute * 1e3,
                        prof_update * 1e3,
                        prof_process * 1e3,
                        (time.perf_counter() - prof_loop_t0) * 1e3,
                    )
                consecutive_errors = 0
                self._heartbeat()
            except KeyboardInterrupt:
                self._info("Scheduler loop interrupted by user")
                break
            except Exception as e:
                consecutive_errors += 1
                traceback.print_exc()
                logger.error(
                    "Scheduler loop error (attempt %d/%d): %s",
                    consecutive_errors,
                    max_consecutive_errors,
                    e,
                )
                if self._is_nonrecoverable(e):
                    logger.critical(
                        "Fatal scheduler error (non-recoverable invariant violation): %s",
                        e,
                    )
                    self._on_fatal(e)
                    break

                if consecutive_errors >= max_consecutive_errors:
                    logger.critical(
                        f"Scheduler loop encountered {consecutive_errors} consecutive errors. "
                        "Stopping scheduler to prevent resource exhaustion."
                    )
                    self._on_fatal(e)
                    break
                time.sleep(0.01)
        self._info("Background scheduler loop stopped")

    def _run_overlap(self) -> None:
        """Async-overlap loop body: dispatch next step before draining the previous."""
        coordinator = self._coordinator
        consecutive_errors = 0
        max_consecutive_errors = MAX_CONSECUTIVE_SCHEDULER_ERRORS

        pending_execution: tuple[typing.Any, SchedulerOutput] | None = None
        prefetched_schedule: SchedulerOutput | None = None

        if not coordinator.supports_overlap:
            raise ValueError(
                "Distributed step synchronization failure: overlap_execution=True is not supported "
                "with remote distributed workers."
            )

        def _can_dispatch_next_before_drain(current: SchedulerOutput) -> bool:
            """Whether async decode may launch before draining tokens.

            AsyncScheduler tracks one-step-ahead decode with optimistic
            output placeholders. When the runner says the current step
            is eligible, the next step can be queued before
            host-materializing the previous sampled token; the runner
            patches that placeholder from the previous device token via
            ``DeviceInputTokenHandoff``. CPU update/streaming still
            happens in scheduler order when the older handle is drained.
            """
            try:
                return bool(self.runner.can_dispatch_next_before_async_drain(current))
            except Exception:
                return False

        def _can_prefetch_next(current: SchedulerOutput) -> bool:
            """Whether the *plain* overlap path may run ``schedule()`` early.

            Async scheduling uses optimistic output placeholders and
            runner-side deferred sampled-token state. Advancing the
            scheduler before the previous async result is drained can
            make those two state machines observe different request
            lengths. Plain overlap remains enabled for non-async
            scheduler outputs by delegating to
            :meth:`_can_prefetch_scheduler_output`.
            """
            if current.async_scheduling:
                return False
            return self._can_prefetch_scheduler_output(self.scheduler, current)

        def _execute_zero_token_schedule(scheduler_output: SchedulerOutput) -> None:
            """Run a zero-token schedule synchronously through the runner.

            Some scheduler outputs do not actually generate tokens
            (e.g. parser-driven termination housekeeping). The
            overlap path falls back to a synchronous
            ``execute_model`` for those so the bookkeeping still
            passes through ``update_from_output`` and the parser
            pipeline without disturbing the async pipeline carry.
            """
            if self._runner_verbose and self._zero_schedule_needs_update(scheduler_output):
                logger.info("[esurge-loop] executing zero-token scheduler output")
            model_output = coordinator.execute_sync(scheduler_output)
            with self._scheduler_lock:
                engine_outputs = self.scheduler.update_from_output(scheduler_output, model_output)
            self._emit_outputs(engine_outputs)
            self._handle_profiling_step()

        def _execute_positive_schedule(
            scheduler_output: SchedulerOutput,
        ) -> tuple[typing.Any, SchedulerOutput] | None:
            """Dispatch decode asynchronously and prefill synchronously.

            The async-overlap path defers sampled-token materialization
            so decode step ``N+1`` can consume step ``N``'s sampled
            token through device-side handoff. That contract only holds
            for one-token decode windows. Chunked prefill windows do
            not produce a client-visible sampled token, so running them
            through an async handle only adds a deferred host copy that
            the scheduler cannot use. Execute those synchronously and
            return ``None``; return a pending async handle only for
            decode-shaped schedules.
            """
            defer_safe = self.runner.can_dispatch_next_before_async_drain(scheduler_output)
            if self._runner_verbose:
                logger.info(
                    "[esurge-loop] dispatch tok=%d reqs=%d async=%s defer_safe=%s",
                    int(scheduler_output.total_num_scheduled_tokens or 0),
                    len(scheduler_output.num_scheduled_tokens),
                    bool(scheduler_output.async_scheduling),
                    bool(defer_safe),
                )
            if defer_safe:
                return coordinator.execute_async(scheduler_output), scheduler_output

            model_output = coordinator.execute_sync(scheduler_output)
            with self._scheduler_lock:
                engine_outputs = self.scheduler.update_from_output(scheduler_output, model_output)
            self._emit_outputs(engine_outputs)
            self._handle_profiling_step()
            return None

        while self._running:
            try:
                coordinator.check_health()
                if pending_execution is not None:
                    future, prev_sched_out = pending_execution

                    if _can_dispatch_next_before_drain(prev_sched_out):
                        with self._scheduler_lock:
                            self._drain_parser_stops()
                            next_sched_out = self.scheduler.schedule()
                        self._heartbeat()

                        if next_sched_out.total_num_scheduled_tokens > 0:
                            next_pending: tuple[typing.Any, SchedulerOutput] | None = None
                            if self.runner.can_dispatch_next_before_async_drain(next_sched_out):
                                next_pending = (coordinator.execute_async(next_sched_out), next_sched_out)
                            try:
                                self._drain(future, prev_sched_out)
                            except Exception as e:
                                pending_execution = None
                                prefetched_schedule = None
                                logger.critical(
                                    "Async deferred drain failed after launching the next step. "
                                    "Aborting scheduler because scheduler state has advanced."
                                )
                                self._on_fatal(e)
                                break
                            if next_pending is None:
                                next_pending = _execute_positive_schedule(next_sched_out)
                            pending_execution = next_pending
                            consecutive_errors = 0
                            self._heartbeat()
                            continue

                        try:
                            self._drain(future, prev_sched_out)
                            if self._zero_schedule_needs_update(next_sched_out):
                                _execute_zero_token_schedule(next_sched_out)
                        except Exception as e:
                            pending_execution = None
                            prefetched_schedule = None
                            logger.critical(
                                "Async drain failed after preparing an empty follow-up schedule. "
                                "Aborting scheduler because scheduler state has advanced."
                            )
                            self._on_fatal(e)
                            break
                        pending_execution = None
                        consecutive_errors = 0
                        self._heartbeat()
                        continue

                    if prefetched_schedule is None and _can_prefetch_next(prev_sched_out):
                        with self._scheduler_lock:
                            self._drain_parser_stops()
                            prefetched_schedule = self.scheduler.schedule()

                    try:
                        self._drain(future, prev_sched_out)
                    except Exception as e:
                        if prefetched_schedule is not None:
                            pending_execution = None
                            prefetched_schedule = None
                            logger.critical(
                                "Overlap drain failed after speculatively advancing scheduler state. "
                                "Aborting scheduler instead of retrying with an unexecuted prefetched batch."
                            )
                            self._on_fatal(e)
                            break
                        raise
                    pending_execution = None

                if prefetched_schedule is not None:
                    scheduler_output = prefetched_schedule
                    prefetched_schedule = None
                else:
                    with self._scheduler_lock:
                        self._drain_parser_stops()
                        scheduler_output = self.scheduler.schedule()
                if self._runner_verbose and (
                    scheduler_output.total_num_scheduled_tokens > 0 or self._zero_schedule_needs_update(scheduler_output)
                ):
                    logger.info(
                        "[esurge-loop] scheduled tok=%d reqs=%d running=%d waiting=%d async=%s",
                        int(scheduler_output.total_num_scheduled_tokens or 0),
                        len(scheduler_output.num_scheduled_tokens),
                        len(self.scheduler.running),
                        len(self.scheduler.waiting),
                        bool(scheduler_output.async_scheduling),
                    )
                self._heartbeat()

                if scheduler_output.total_num_scheduled_tokens == 0:
                    if self._zero_schedule_needs_update(scheduler_output):
                        _execute_zero_token_schedule(scheduler_output)
                    else:
                        self._idle_scheduler_tick()
                        pending_execution = None
                else:
                    pending_execution = _execute_positive_schedule(scheduler_output)

                consecutive_errors = 0
                self._heartbeat()
            except KeyboardInterrupt:
                self._info("Scheduler loop interrupted by user")
                break
            except DPSchedulerClosed:
                # Not a fault: the scheduler's rank workers are gone because
                # teardown began. Retrying cannot succeed, and the retries are
                # not free -- five of them delayed a rank past jax.distributed's
                # shutdown barrier and its peer aborted the job.
                pending_execution = None
                self._info("Scheduler loop stopping: DP scheduler is shut down")
                break
            except Exception as e:
                pending_execution = None
                prefetched_schedule = None
                consecutive_errors += 1
                traceback.print_exc()
                logger.error(
                    "Scheduler loop error (attempt %d/%d): %s",
                    consecutive_errors,
                    max_consecutive_errors,
                    e,
                )
                if self._is_nonrecoverable(e):
                    logger.critical(
                        "Fatal scheduler error (non-recoverable invariant violation): %s",
                        e,
                    )
                    self._on_fatal(e)
                    break

                if consecutive_errors >= max_consecutive_errors:
                    logger.critical(
                        f"Scheduler loop encountered {consecutive_errors} consecutive errors. "
                        "Stopping scheduler to prevent resource exhaustion."
                    )
                    self._on_fatal(e)
                    break
                time.sleep(0.01)

        if pending_execution is not None:
            try:
                self._drain(*pending_execution)
            except Exception as e:
                traceback.print_exc()
                logger.error("Error processing pending batch: %s", e)

        self._info("Background scheduler loop stopped")
