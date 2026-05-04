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

"""Lifecycle mixin for the eSurge engine.

Handles process-level lifecycle for :class:`eSurge`:

- ``initiate`` / ``terminate`` / ``pause`` / ``resume``
- AOT compilation orchestration of the scheduler/runner
- Signal handling and graceful shutdown
- Idle-reset monitor thread that drains caches after periods of inactivity
- Profiling start/stop wrappers

Exposes :class:`EngineLifecycleMixin`, mixed into :class:`eSurge`.
"""

from __future__ import annotations

import os
import queue
import signal
import threading
import time
import traceback
import typing

import jax
import spectrax as spx

from ..logger import logger
from ..request import EngineRequestStatus
from ..scheduler import Scheduler, SchedulerOutput

MAX_CONSECUTIVE_SCHEDULER_ERRORS = int(os.environ.get("EASURGE_MAX_SCHEDULER_ERRORS", "5"))
_SCHEDULER_HEARTBEAT_WARN_S = float(os.environ.get("EASURGE_HEARTBEAT_WARN_S", "120"))
_SCHEDULER_HEARTBEAT_WARN_INTERVAL_S = float(os.environ.get("EASURGE_HEARTBEAT_WARN_INTERVAL_S", "30"))

if typing.TYPE_CHECKING:
    from easydel.infra import EasyDeLBaseModule


class EngineLifecycleMixin:
    """Mixin managing the scheduler lifecycle for the eSurge engine.

    Provides methods to start, stop, pause, and resume the background
    scheduler thread that drives inference. Also handles error classification,
    fatal abort propagation, model weight hot-swapping, JAX profiling,
    and signal-based diagnostics for debugging hangs or OOM kills.

    Methods:
        initiate: Start the background scheduler thread.
        terminate: Stop the scheduler gracefully.
        pause: Temporarily pause scheduling without clearing state.
        resume: Resume a paused scheduler.
        update_model_weights: Hot-swap model weights while idle.
        start_profiling: Begin a JAX profiler trace.
        stop_profiling: End the active profiler trace.
        release_model_state: Free model weights to reduce memory usage.
    """

    @staticmethod
    def _is_nonrecoverable_scheduler_error(exc: BaseException) -> bool:
        """Decide whether a scheduler exception should bypass the retry budget.

        Most exceptions are absorbed up to ``MAX_CONSECUTIVE_SCHEDULER_ERRORS``
        retries, since a transient failure (rate-limited HBM allocator, a
        flaky tokenizer worker, etc.) often clears on the next iteration.
        Some failures, however, leave the engine in an inconsistent state
        that the next iteration will only worsen — for instance a DP-local
        page-table invariant violation indicates the scheduler and runner
        disagree on which DP shard owns a request, and retrying drives
        further pages onto the wrong shard. Those flavours are matched here
        and force an immediate abort via
        :meth:`_abort_scheduler_due_to_error`.

        Args:
            exc: Exception raised by the scheduler loop body.

        Returns:
            ``True`` for failures that should *not* be retried (currently:
            ``ValueError`` whose message contains
            ``"Non-DP-local page IDs detected"`` or ``"Distributed step
            synchronization failure"``); ``False`` otherwise.
        """
        msg = str(exc)
        return isinstance(exc, ValueError) and (
            "Non-DP-local page IDs detected" in msg or "Distributed step synchronization failure" in msg
        )

    @staticmethod
    def _model_overrides_esurge_graphdef(model: EasyDeLBaseModule) -> bool:
        """Whether the model class supplies its own ``esurge_graphdef`` override.

        Some wrapper models (e.g. embedding-augmented LMs) cannot be
        rebuilt by eSurge's default ``_esurge_graphdef_from_graphdef``
        because their construction depends on runtime state that no longer
        exists at hot-swap time. They opt out by exposing a class-level
        ``esurge_graphdef`` attribute / property; weight-update logic
        consults this flag to decide whether to delegate eSurge graph
        construction to the underlying compatible model instead.

        Args:
            model: Loaded EasyDeL module to inspect.

        Returns:
            ``True`` iff the model's *class* (not the instance) has an
            ``esurge_graphdef`` entry in its dict — checking the class is
            important because most concrete models inherit a default that
            should not register as an override.
        """
        return type(model).__dict__.get("esurge_graphdef") is not None

    @classmethod
    def _split_graph_components_for_weight_update(cls, model: EasyDeLBaseModule):
        """Run :meth:`split_module` on the right model for an eSurge hot-swap.

        Engines that wrap their model in a non-eSurge-compatible class need
        their ``esurge_compatible_model`` (the inner LM that actually carries
        the executable graph) split, not the wrapper. This helper resolves
        the correct module, calls ``split_module()`` on it, and returns the
        full split tuple along with the resolved module so callers don't
        have to repeat the wrapper-detection logic.

        Args:
            model: Wrapped or bare module supplied to :meth:`update_model_weights`.

        Returns:
            Tuple ``(split_model, graphdef, graphstate, graphother)`` where
            ``split_model`` is either ``model.esurge_compatible_model``
            (when the wrapper opts out via :meth:`_model_overrides_esurge_graphdef`)
            or ``model`` itself, and the remaining three are the components
            produced by :meth:`split_module` on that module.
        """
        split_model = model
        if cls._model_overrides_esurge_graphdef(model):
            try:
                split_model = model.esurge_compatible_model
            except Exception:
                split_model = model
        split_graphdef, split_graphstate, split_graphother = split_model.split_module()
        return split_model, split_graphdef, split_graphstate, split_graphother

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

    def _apply_parser_stop_requests_locked(self, stop_string_finishes: dict[str, str]) -> None:
        """Apply parser-detected stop strings while the scheduler lock is held.

        Text parsing runs on the output worker, not on the generation thread.
        When that worker discovers a stop string it must still tell the
        scheduler to free the request, but it should never grab
        ``_scheduler_lock`` itself. The worker enqueues a tiny stop-signal
        packet and the generation loop drains it at scheduler-safe points.

        Args:
            stop_string_finishes: Mapping from request id to matched stop
                string. The caller must hold ``_scheduler_lock``.
        """
        if not stop_string_finishes:
            return
        for rid, stop_reason in stop_string_finishes.items():
            request = self.scheduler.requests.get(rid)
            if request is not None:
                request.stop_reason = stop_reason
        self.scheduler.finish_requests(stop_string_finishes.keys(), EngineRequestStatus.FINISHED_STOPPED)

    def _enqueue_parser_stop_requests(self, stop_string_finishes: dict[str, str]) -> None:
        """Queue parser-detected stop strings for the generation thread.

        This is intentionally non-blocking with respect to ``_scheduler_lock``.
        It keeps reasoning/tool/stop-string parsing from contending with the
        scheduler loop. Lightweight tests that do not define
        ``_parser_stop_queue`` fall back to the old inline behaviour.
        """
        if not stop_string_finishes:
            return
        stop_queue = getattr(self, "_parser_stop_queue", None)
        if stop_queue is None:
            with self._scheduler_lock:
                self._apply_parser_stop_requests_locked(dict(stop_string_finishes))
            return
        stop_queue.put(dict(stop_string_finishes))

    def _drain_parser_stop_requests_locked(self) -> None:
        """Drain queued parser stop signals on the generation thread.

        Called immediately before ``scheduler.schedule()`` while
        ``_scheduler_lock`` is already held. Merging all currently queued
        packets keeps the scheduler transition tiny and deterministic.
        """
        stop_queue = getattr(self, "_parser_stop_queue", None)
        if stop_queue is None:
            return
        merged: dict[str, str] = {}
        while True:
            try:
                merged.update(stop_queue.get_nowait())
            except queue.Empty:
                break
        self._apply_parser_stop_requests_locked(merged)

    @classmethod
    def _resolve_graphdef_for_weight_update(cls, model: EasyDeLBaseModule, split_graphdef=None):
        """Resolve the graphdef to use for a model-weight refresh.

        Some wrapper types override ``esurge_graphdef`` to delegate eSurge graph
        construction to an underlying base LM because the wrapper itself cannot
        be lazily rebuilt from config. Respect that override instead of forcing
        ``_esurge_graphdef_from_graphdef(...)`` on the wrapper graphdef.
        """
        if cls._model_overrides_esurge_graphdef(model):
            try:
                compatible_model = model.esurge_compatible_model
            except Exception:
                compatible_model = None
            if compatible_model is not None:
                return split_graphdef if split_graphdef is not None else compatible_model.graphdef
            return model.esurge_graphdef

        base_graphdef = split_graphdef if split_graphdef is not None else model.graphdef
        return model._esurge_graphdef_from_graphdef(base_graphdef)

    def _abort_scheduler_due_to_error(self, exc: BaseException) -> None:
        """Record a fatal scheduler error and wake all waiting callers.

        Args:
            exc: The exception that caused the scheduler failure.

        Note:
            This method records the exception and traceback, stops the scheduler,
            and signals all waiting events so that blocking calls (generate/stream/chat)
            can raise the error immediately.
        """
        # Record the failure so waiting callers can raise immediately.
        with self._request_lock:
            self._scheduler_exception = exc
            self._scheduler_exception_tb = traceback.format_exc()

        # Stop the scheduler and wake up any waiters (generate/stream/chat).
        self._scheduler_running = False
        self._output_event.set()
        with self._request_lock:
            events = list(self._request_events.values())
        for ev in events:
            ev.set()

    def _raise_if_scheduler_failed(self) -> None:
        """Check for scheduler failure and raise if one occurred.

        Raises:
            RuntimeError: If the scheduler encountered a fatal error, with
                the original exception and traceback included.
        """
        exc = self._scheduler_exception
        if exc is None:
            # No crash, but check for heartbeat staleness (possible hang).
            self._check_scheduler_heartbeat()
            return
        tb = self._scheduler_exception_tb
        if tb:
            raise RuntimeError(f"eSurge scheduler crashed: {exc}\n{tb}") from exc
        raise RuntimeError(f"eSurge scheduler crashed: {exc}") from exc

    def _install_signal_diagnostics(self) -> None:
        """Install signal handlers that log engine state before exit.

        Registers handlers for SIGTERM and SIGUSR1 (where available) to
        dump scheduler/request state before the process is killed. This
        makes OOM-kills and external termination diagnosable.
        """

        def _dump_state(signum, frame):
            sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
            try:
                running = getattr(self, "num_running_requests", "?")
                pending = getattr(self, "num_pending_requests", "?")
                sched_alive = getattr(self, "_scheduler_running", "?")
                heartbeat = getattr(self, "_scheduler_heartbeat", None)
                hb_age = f"{time.monotonic() - heartbeat:.1f}s ago" if heartbeat else "never"
                logger.critical(
                    "eSurge received signal %s — dumping state before exit: "
                    "scheduler_running=%s running_reqs=%s pending_reqs=%s "
                    "last_heartbeat=%s",
                    sig_name,
                    sched_alive,
                    running,
                    pending,
                    hb_age,
                )
            except Exception:
                logger.critical("eSurge received signal %s (state dump failed)", signum)
            # Re-raise with default handler so the process actually terminates.
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        for sig in (signal.SIGTERM,):
            try:
                prev = signal.getsignal(sig)
                # Only install if the current handler is the default.
                if prev in (signal.SIG_DFL, None):
                    signal.signal(sig, _dump_state)
            except (OSError, ValueError):
                pass  # Not main thread or signal not available

    def _update_scheduler_heartbeat(self) -> None:
        """Stamp the scheduler-loop progress watermark with the current monotonic time.

        Called twice per iteration of the scheduler loop (once after
        ``schedule()`` lands, once after ``update_from_output()`` lands)
        so :meth:`_check_scheduler_heartbeat` can distinguish a hung loop
        from one that is busy on a long step.
        """
        self._scheduler_heartbeat = time.monotonic()

    def _check_scheduler_heartbeat(self) -> None:
        """Emit a WARNING when the scheduler heartbeat is staler than the threshold.

        Polled from :meth:`_raise_if_scheduler_failed` (which is invoked
        on every poll of :meth:`generate` / :meth:`stream`). Compares the
        last heartbeat against
        :data:`_SCHEDULER_HEARTBEAT_WARN_S`; when stale, logs at WARNING
        with the age and rate-limits subsequent warnings to one per
        :data:`_SCHEDULER_HEARTBEAT_WARN_INTERVAL_S` so a hang doesn't
        flood the logs. No-op when the scheduler isn't running or no
        heartbeat has been recorded yet.
        """
        if not getattr(self, "_scheduler_running", False):
            return
        heartbeat = getattr(self, "_scheduler_heartbeat", None)
        if heartbeat is None:
            return
        now = time.monotonic()
        age = now - heartbeat
        if age > _SCHEDULER_HEARTBEAT_WARN_S:
            last_warn = getattr(self, "_scheduler_heartbeat_last_warn", 0.0)
            if now - last_warn < _SCHEDULER_HEARTBEAT_WARN_INTERVAL_S:
                return
            self._scheduler_heartbeat_last_warn = now
            logger.warning(
                "Scheduler heartbeat stale: last update %.1fs ago (threshold=%.0fs). Possible hang or deadlock.",
                age,
                _SCHEDULER_HEARTBEAT_WARN_S,
            )

    def _track_finished_output(self, request_id: str) -> None:
        """Track and evict completed RequestOutput objects to cap memory usage.

        Manages the finished request history using a FIFO eviction policy.
        When the number of retained outputs exceeds max_request_outputs,
        the oldest outputs are removed to free memory.

        Args:
            request_id: ID of the request that just finished.
        """
        max_outputs = self._max_request_outputs
        if max_outputs is None:
            return
        if max_outputs <= 0:
            self._request_outputs.pop(request_id, None)
            return
        if request_id in self._finished_request_ids:
            return
        self._finished_request_ids.append(request_id)
        while len(self._finished_request_ids) > max_outputs:
            old_id = self._finished_request_ids.popleft()
            if old_id == request_id:
                continue
            self._request_outputs.pop(old_id, None)

    def _start_engine_output_worker(self) -> None:
        """Start the background parser/output worker when the engine owns one.

        The scheduler loop is the hot generation path. It should apply model
        tokens to scheduler state and immediately return to launching work.
        Detokenization, reasoning/tool parsing, stream-delta assembly, and
        request-event wakeups are handled by this single FIFO worker so output
        processing overlaps the next device step while preserving per-token
        ordering.

        Lightweight lifecycle test harnesses do not define
        ``_engine_output_queue``; those keep the historical synchronous path.
        """
        output_queue = getattr(self, "_engine_output_queue", None)
        if output_queue is None:
            return

        thread = getattr(self, "_engine_output_thread", None)
        if thread is not None and thread.is_alive():
            return

        def _output_loop() -> None:
            while True:
                engine_outputs = output_queue.get()
                try:
                    if engine_outputs is None:
                        return
                    self._process_engine_outputs(engine_outputs)
                except Exception as exc:
                    traceback.print_exc()
                    logger.error("Engine output worker failed: %s", exc)
                    self._scheduler_exception = exc
                    self._scheduler_exception_tb = traceback.format_exc()
                    self._scheduler_running = False
                    with self._request_lock:
                        for event in self._request_events.values():
                            event.set()
                    self._output_event.set()
                finally:
                    output_queue.task_done()

        self._engine_output_thread = threading.Thread(target=_output_loop, name="eSurgeOutputWorker", daemon=True)
        self._engine_output_thread.start()

    def _enqueue_engine_outputs(self, engine_outputs) -> None:
        """Queue engine outputs for asynchronous parsing, or process inline.

        Args:
            engine_outputs: Scheduler-produced output bundle from
                ``update_from_output``.
        """
        if not engine_outputs:
            return
        output_queue = getattr(self, "_engine_output_queue", None)
        thread = getattr(self, "_engine_output_thread", None)
        if output_queue is None or thread is None or not thread.is_alive():
            self._process_engine_outputs(engine_outputs)
            return
        output_queue.put(engine_outputs)

    def _stop_engine_output_worker(self) -> None:
        """Drain and stop the asynchronous parser/output worker."""
        output_queue = getattr(self, "_engine_output_queue", None)
        thread = getattr(self, "_engine_output_thread", None)
        if output_queue is None or thread is None:
            return
        if thread.is_alive():
            output_queue.put(None)
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning("Engine output worker did not stop gracefully")
                return
        if getattr(self, "_engine_output_thread", None) is thread:
            self._engine_output_thread = None

    def initiate(self) -> None:
        """Start (or wake) the background scheduler so requests can flow.

        Two flavours of behaviour, gated by ``distributed_mode``:

        * **Worker rank** in distributed mode — boots the control server
          via :meth:`DistributedController.start`, ensures KV pages are
          allocated, and returns; no scheduler thread is spawned because
          the leader will dispatch each step over RPC.
        * **Leader rank or single-host** — re-allocates KV pages if they
          were previously destroyed, clears any prior crash state,
          installs SIGTERM diagnostics, and spawns the daemon scheduler
          thread that runs the main control loop.

        The scheduler-loop body itself comes in two shapes depending on
        ``runtime_config.overlap_execution``:

        1. **Synchronous path** — call ``schedule()``, run the model
           via :meth:`eSurgeRunner.execute_model`, and apply the result
           with ``update_from_output()`` on the same thread.
        2. **Overlap path** — dispatch the model asynchronously via
           :meth:`eSurgeRunner.execute_model_async`, use
           :meth:`_can_prefetch_scheduler_output` to decide whether to
           compute the next batch while the device is still busy, use the
           runner's async-drain capability gate for placeholder-token decode,
           and drain the prior step via :meth:`_drain_runner_future`.

        Both paths are wrapped in a retry budget
        (:data:`MAX_CONSECUTIVE_SCHEDULER_ERRORS`) and routed through
        :meth:`_is_nonrecoverable_scheduler_error` so invariant violations
        terminate immediately instead of looping. The
        :meth:`_update_scheduler_heartbeat` call after each iteration
        keeps :meth:`_check_scheduler_heartbeat` in :meth:`_raise_if_scheduler_failed`
        able to detect hangs.

        State on exit: ``_scheduler_running=True``, idle monitor armed,
        ``_paused=False``. No-op if the loop is already running.
        """
        with self._scheduler_lock:
            distributed_controller = getattr(self, "_distributed_controller", None)
            if distributed_controller is not None:
                distributed_controller.start()
                if distributed_controller.is_worker:
                    if self.runner.executor_manager.kv_pages is None:
                        self.runner.initialize_kv_cache()
                        self._kv_cache_valid = True

                    self._scheduler_exception = None
                    self._scheduler_exception_tb = None
                    self._scheduler_running = False
                    self._touch_activity()
                    self._start_idle_monitor()
                    self._paused = False
                    self._info("Distributed worker control server is running (scheduler loop disabled).")
                    return

            if self._scheduler_running:
                self._info("Scheduler loop is already running")
                return

            if self.runner.executor_manager.kv_pages is None:
                self.runner.initialize_kv_cache()
                self._kv_cache_valid = True

            self._start_engine_output_worker()

            # Clear any previous crash state before starting a fresh scheduler thread.
            self._scheduler_exception = None
            self._scheduler_exception_tb = None
            self._scheduler_heartbeat = None

            # Install signal diagnostics (best-effort, main thread only).
            try:
                self._install_signal_diagnostics()
            except Exception:
                pass

            def _scheduler_loop():
                self._info("Starting background scheduler loop")
                consecutive_errors = 0
                max_consecutive_errors = MAX_CONSECUTIVE_SCHEDULER_ERRORS
                distributed_controller = getattr(self, "_distributed_controller", None)

                _diag_iter = 0
                _diag_last_log = time.time()
                def _overlap_loop_enabled() -> bool:
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
                    if not self.runtime_config.overlap_execution:
                        return False
                    return hasattr(self.runner, "execute_model_async")

                use_overlap_loop = _overlap_loop_enabled()
                if getattr(self.runtime_config, "runner_verbose", False):
                    logger.info(
                        "Using %s eSurge scheduler lifecycle "
                        "(async_scheduling=%s, overlap_execution=%s)",
                        "async-overlap" if use_overlap_loop else "synchronous",
                        bool(self.runtime_config.async_scheduling),
                        bool(self.runtime_config.overlap_execution),
                    )
                if not use_overlap_loop:
                    while self._scheduler_running:
                        try:
                            _diag_iter += 1
                            prof_loop_t0 = time.perf_counter()
                            prof_phase = prof_loop_t0
                            with self._scheduler_lock:
                                self._drain_parser_stop_requests_locked()
                                scheduler_output = self.scheduler.schedule()
                            prof_sched = time.perf_counter() - prof_phase
                            _n = len(scheduler_output.num_scheduled_tokens) if scheduler_output else 0
                            _now = time.time()
                            if _n > 0 or (_now - _diag_last_log) > 30:
                                logger.debug(
                                    "loop iter=%d sched=%d run=%d wait=%d",
                                    _diag_iter,
                                    _n,
                                    len(self.scheduler.running),
                                    len(self.scheduler.waiting),
                                )
                                _diag_last_log = _now
                            self._update_scheduler_heartbeat()
                            dispatch = None
                            if distributed_controller is not None and distributed_controller.has_remote_workers:
                                prof_phase = time.perf_counter()
                                dispatch = distributed_controller.dispatch_step(scheduler_output)
                                prof_dispatch = time.perf_counter() - prof_phase
                            else:
                                prof_dispatch = 0.0
                            prof_phase = time.perf_counter()
                            model_output = self.runner.execute_model(scheduler_output)
                            prof_execute = time.perf_counter() - prof_phase
                            if dispatch is not None:
                                prof_phase = time.perf_counter()
                                distributed_controller.verify_step(dispatch, model_output)
                                prof_dispatch += time.perf_counter() - prof_phase
                            prof_phase = time.perf_counter()
                            with self._scheduler_lock:
                                engine_outputs = self.scheduler.update_from_output(scheduler_output, model_output)
                            prof_update = time.perf_counter() - prof_phase
                            prof_phase = time.perf_counter()
                            self._enqueue_engine_outputs(engine_outputs)
                            prof_process = time.perf_counter() - prof_phase
                            if (
                                getattr(self.runtime_config, "runner_verbose", False)
                                and scheduler_output.total_num_scheduled_tokens > 0
                            ):
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
                            # Reset error counter on success
                            consecutive_errors = 0
                            self._update_scheduler_heartbeat()
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
                            if self._is_nonrecoverable_scheduler_error(e):
                                logger.critical(
                                    "Fatal scheduler error (non-recoverable invariant violation): %s",
                                    e,
                                )
                                self._abort_scheduler_due_to_error(e)
                                break

                            if consecutive_errors >= max_consecutive_errors:
                                logger.critical(
                                    f"Scheduler loop encountered {consecutive_errors} consecutive errors. "
                                    "Stopping scheduler to prevent resource exhaustion."
                                )
                                self._abort_scheduler_due_to_error(e)
                                break
                            time.sleep(0.01)
                    self._info("Background scheduler loop stopped")
                    return

                pending_execution: tuple[typing.Any, SchedulerOutput] | None = None
                prefetched_schedule: SchedulerOutput | None = None

                if distributed_controller is not None and distributed_controller.has_remote_workers:
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
                    # Async scheduling uses optimistic output placeholders and
                    # runner-side deferred sampled-token state. Advancing the
                    # scheduler before the previous async result is drained can
                    # make those two state machines observe different request
                    # lengths. Plain overlap remains enabled for non-async
                    # scheduler outputs.
                    if current.async_scheduling:
                        return False
                    return self._can_prefetch_scheduler_output(self.scheduler, current)

                def _execute_zero_token_schedule(scheduler_output: SchedulerOutput) -> None:
                    model_output = self.runner.execute_model(scheduler_output)
                    with self._scheduler_lock:
                        engine_outputs = self.scheduler.update_from_output(scheduler_output, model_output)
                    self._enqueue_engine_outputs(engine_outputs)
                    self._handle_profiling_step()

                while self._scheduler_running:
                    try:
                        if pending_execution is not None:
                            future, prev_sched_out = pending_execution

                            if _can_dispatch_next_before_drain(prev_sched_out):
                                with self._scheduler_lock:
                                    self._drain_parser_stop_requests_locked()
                                    next_sched_out = self.scheduler.schedule()
                                self._update_scheduler_heartbeat()

                                if next_sched_out.total_num_scheduled_tokens > 0:
                                    next_future = self.runner.execute_model_async(next_sched_out)
                                    try:
                                        self._drain_runner_future(future, prev_sched_out)
                                    except Exception as e:
                                        pending_execution = None
                                        prefetched_schedule = None
                                        logger.critical(
                                            "Async deferred drain failed after launching the next step. "
                                            "Aborting scheduler because scheduler state has advanced."
                                        )
                                        self._abort_scheduler_due_to_error(e)
                                        break
                                    pending_execution = (next_future, next_sched_out)
                                    consecutive_errors = 0
                                    self._update_scheduler_heartbeat()
                                    continue

                                try:
                                    self._drain_runner_future(future, prev_sched_out)
                                    _execute_zero_token_schedule(next_sched_out)
                                except Exception as e:
                                    pending_execution = None
                                    prefetched_schedule = None
                                    logger.critical(
                                        "Async drain failed after preparing an empty follow-up schedule. "
                                        "Aborting scheduler because scheduler state has advanced."
                                    )
                                    self._abort_scheduler_due_to_error(e)
                                    break
                                pending_execution = None
                                consecutive_errors = 0
                                self._update_scheduler_heartbeat()
                                continue

                            if prefetched_schedule is None and _can_prefetch_next(prev_sched_out):
                                with self._scheduler_lock:
                                    self._drain_parser_stop_requests_locked()
                                    prefetched_schedule = self.scheduler.schedule()

                            try:
                                self._drain_runner_future(future, prev_sched_out)
                            except Exception as e:
                                if prefetched_schedule is not None:
                                    pending_execution = None
                                    prefetched_schedule = None
                                    logger.critical(
                                        "Overlap drain failed after speculatively advancing scheduler state. "
                                        "Aborting scheduler instead of retrying with an unexecuted prefetched batch."
                                    )
                                    self._abort_scheduler_due_to_error(e)
                                    break
                                raise
                            pending_execution = None

                        if prefetched_schedule is not None:
                            scheduler_output = prefetched_schedule
                            prefetched_schedule = None
                        else:
                            with self._scheduler_lock:
                                self._drain_parser_stop_requests_locked()
                                scheduler_output = self.scheduler.schedule()
                        self._update_scheduler_heartbeat()

                        if scheduler_output.total_num_scheduled_tokens == 0:
                            _execute_zero_token_schedule(scheduler_output)
                        else:
                            future = self.runner.execute_model_async(scheduler_output)
                            pending_execution = (future, scheduler_output)

                        # Reset error counter on success
                        consecutive_errors = 0
                        self._update_scheduler_heartbeat()
                    except KeyboardInterrupt:
                        self._info("Scheduler loop interrupted by user")
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
                        if self._is_nonrecoverable_scheduler_error(e):
                            logger.critical(
                                "Fatal scheduler error (non-recoverable invariant violation): %s",
                                e,
                            )
                            self._abort_scheduler_due_to_error(e)
                            break

                        if consecutive_errors >= max_consecutive_errors:
                            logger.critical(
                                f"Scheduler loop encountered {consecutive_errors} consecutive errors. "
                                "Stopping scheduler to prevent resource exhaustion."
                            )
                            self._abort_scheduler_due_to_error(e)
                            break
                        time.sleep(0.01)

                if pending_execution is not None:
                    try:
                        self._drain_runner_future(*pending_execution)
                    except Exception as e:
                        traceback.print_exc()
                        logger.error("Error processing pending batch: %s", e)

                self._info("Background scheduler loop stopped")

            self._scheduler_running = True
            self._scheduler_thread = threading.Thread(target=_scheduler_loop, daemon=True)
            self._scheduler_thread.start()
            self._info("Background scheduler initiated")
            self._touch_activity()
            self._start_idle_monitor()
            self._paused = False

    def terminate(self) -> None:
        """Tear down the scheduler loop and release subordinate resources.

        Engine-stop path. Steps:

        1. Stop the idle-reset watchdog so it cannot fire mid-shutdown.
        2. Set ``_scheduler_running = False`` under ``_scheduler_lock``, then
           release the lock before joining the scheduler thread. The release is
           important because the scheduler may need the same lock to apply the
           final runner output before it can exit. Workers in distributed mode
           are also asked to shut down their control servers here.
        3. If a profiler trace is active, stop it via
           :meth:`stop_profiling` (best-effort; failures are demoted to
           DEBUG).
        4. Forward to :meth:`eSurgeRunner.shutdown` so resident PP-stage
           worker threads inside the :class:`ModelStepExecutor` are
           joined.
        5. Clear runner buffers via :meth:`_reset_runner_state_if_idle`
           when the engine is genuinely idle, so a subsequent
           :meth:`initiate` starts from a clean slate.

        State on exit: the scheduler thread is gone, the runner has no
        in-flight state, and no background timers are active. Idempotent
        — calling it on an already-terminated engine logs an info line
        and returns.
        """
        self._stop_idle_monitor()
        scheduler_thread: threading.Thread | None = None
        with self._scheduler_lock:
            if not self._scheduler_running:
                distributed_controller = getattr(self, "_distributed_controller", None)
                if distributed_controller is not None and distributed_controller.is_worker:
                    try:
                        distributed_controller.shutdown()
                    except Exception:
                        logger.debug("Distributed worker controller shutdown encountered an error", exc_info=True)
                self._stop_engine_output_worker()
                self._info("Scheduler loop is not running")
                return
            self._info("Stopping background scheduler loop...")
            self._scheduler_running = False
            scheduler_thread = self._scheduler_thread

        if scheduler_thread is not None:
            scheduler_thread.join(timeout=5.0)
            if scheduler_thread.is_alive():
                logger.warning("Scheduler thread did not stop gracefully")

        with self._scheduler_lock:
            if self._scheduler_thread is scheduler_thread:
                self._scheduler_thread = None

        self._stop_engine_output_worker()
        self._info("Background scheduler terminated")
        if self._profiling_active:
            try:
                self.stop_profiling()
            except Exception:
                logger.debug("Profiler stop encountered an error", exc_info=True)
        if hasattr(self.runner, "shutdown"):
            try:
                self.runner.shutdown()
            except Exception:
                logger.debug("Runner shutdown encountered an error", exc_info=True)
        # Clear runner buffers if idle to avoid stale state on next start.
        self._reset_runner_state_if_idle("terminate")

    def pause(self) -> None:
        """Stop the scheduler loop while preserving request bookkeeping.

        Suspended state is intentionally non-destructive for requests:
        the waiting and running queues survive, and active streaming
        clients merely stop receiving updates until :meth:`resume`.
        Internally this terminates the scheduler thread (so any
        in-flight model step is drained), drains the resident PP worker
        pool, and — when ``cache_config.destroy_pages_on_pause`` is set
        — frees the KV pages to reclaim HBM (skipped if requests are
        still active to avoid corrupting their state). Finally,
        :meth:`_reset_runner_state_if_idle` clears the runner's
        sequence buffer if the engine is genuinely idle.

        Sets ``_paused=True``. Idempotent — pausing an already-paused
        engine logs an info line and returns.
        """
        if not self._scheduler_running:
            self._info("Scheduler loop already paused or not running")
            self._paused = True
            return

        self._info("Pausing eSurge scheduler loop...")
        self.terminate()
        self._paused = True
        self._drain_pipeline_workers("pause")
        if self.cache_config.destroy_pages_on_pause:
            if self.num_running_requests > 0 or self.num_pending_requests > 0:
                logger.warning(
                    f"Active or pending requests detected; skipping KV cache destruction (num running requests "
                    f"{self.num_running_requests} | num pending requests {self.num_pending_requests})."
                )
            else:
                self.runner.destroy_kv_cache()
                self._kv_cache_valid = False
                self._log_cache_event("kv_cache_destroyed", {"reason": "pause"})
        # Always try to clear runner buffers when idle to avoid stale state.
        self._reset_runner_state_if_idle("pause")

    def resume(self) -> None:
        """Reopen the engine for inference after a :meth:`pause`.

        Drains any leftover PP-worker state, reinitializes the KV cache
        when ``destroy_pages_on_pause`` had freed it, and delegates to
        :meth:`initiate` to spin a fresh scheduler thread. Requests that
        were waiting before the pause are immediately considered for
        scheduling on the next iteration. Idempotent for already-running
        engines.
        """
        if self._scheduler_running:
            self._info("Scheduler loop already running")
            return
        self._info("Resuming eSurge scheduler loop...")
        self._drain_pipeline_workers("resume")
        if self.cache_config.destroy_pages_on_pause and not self._kv_cache_valid:
            self.runner.initialize_kv_cache()
            self._kv_cache_valid = True
            self._log_cache_event("kv_cache_reinitialized", {"reason": "resume"})
        self.initiate()

    def release_model_state(self, *, clear_compiled_cache: bool = False) -> None:
        """Release runner-held model weights/state references to reduce memory.

        The engine remains reusable. Call `update_model_weights(...)` before
        serving again.

        Args:
            clear_compiled_cache: Whether to clear compiled model/sampler caches.
        """
        if self.num_running_requests > 0 or self.num_pending_requests > 0:
            logger.warning(
                "Skipping model-state release because requests are active or pending (running=%d, pending=%d).",
                self.num_running_requests,
                self.num_pending_requests,
            )
            return

        if self._scheduler_running:
            self.pause()
        else:
            self._drain_pipeline_workers("release_model_state")

        self.runner.release_model_state(clear_compiled_cache=clear_compiled_cache)
        self._paused = True

    def update_model_weights(
        self,
        model: EasyDeLBaseModule | None = None,
        *,
        graphdef=None,
        graphstate=None,
        graphother=None,
        restart_scheduler: bool = True,
    ) -> None:
        """Hot-swap the underlying model weights/graphs.

        The engine must be idle (no pending or running requests) before calling
        this method. It temporarily stops the scheduler loop, refreshes runner
        state, rebuilds the scheduler, and optionally restarts background serving.

        Args:
            model: Optional EasyDeLBaseModule carrying the new weights.
            graphdef: Optional graphdef override.
            graphstate: Optional graphstate override.
            graphother: Optional graphother override.
            restart_scheduler: Restart the scheduler thread if it was previously
                running (default: True).

        Raises:
            RuntimeError: If there are active or pending requests.
            ValueError: If no model/graph data is provided.
        """
        if self.num_running_requests > 0 or self.num_pending_requests > 0:
            raise RuntimeError("Cannot update model weights while requests are active or pending")

        if model is None and graphdef is None and graphstate is None and graphother is None:
            raise ValueError("No new model or graph components provided for update")

        was_running = self._scheduler_running
        if was_running:
            self.terminate()

        self._drain_pipeline_workers("update_model_weights")

        split_graphdef = None
        split_graphstate = None
        split_graphother = None
        using_compatible_split = False
        if model is not None and (graphdef is None or graphstate is None or graphother is None):
            using_compatible_split = self._model_overrides_esurge_graphdef(model)
            _split_model, split_graphdef, split_graphstate, split_graphother = (
                self._split_graph_components_for_weight_update(model)
            )

        if model is None:
            model = spx.bind(graphdef, graphstate.overlay(graphother))
        if using_compatible_split and graphdef is None:
            if graphstate is None:
                graphstate = split_graphstate
            if graphother is None:
                graphother = split_graphother
        else:
            if graphstate is None:
                graphstate = split_graphstate
            if graphother is None:
                graphother = split_graphother
        if graphdef is None:
            graphdef = self._resolve_graphdef_for_weight_update(model, split_graphdef)

        self.runner.update_model_weights(
            graphdef=graphdef,
            graphstate=graphstate,
            graphother=graphother,
            reset_state=True,
        )
        if not self.runner.executor_manager.has_compiled_variants():
            # Compilation uses the current KV pages as a template for sharding/shape.
            # Ensure pages exist when coming back from a released model state.
            if self.runner.executor_manager.kv_pages is None:
                self.runner.initialize_kv_cache()
            self.runner.compile(max_num_batched_tokens=self._scheduler_max_num_batched_tokens)
        self._kv_cache_valid = self.runner.executor_manager.kv_pages is not None
        cache_event = "kv_cache_reinitialized" if self._kv_cache_valid else "kv_cache_destroyed"
        self._log_cache_event(cache_event, {"reason": "update_model_weights"})

        with self._request_lock, self._output_lock:
            self._active_requests.clear()
            self._request_outputs.clear()
            self._finished_request_ids.clear()
            self._request_events.clear()

        self.scheduler = Scheduler.from_runner(
            self.runner,
            max_num_batched_tokens=self._scheduler_max_num_batched_tokens,
            enable_prefix_caching=self.cache_config.enable_prefix_caching,
        )

        if restart_scheduler and was_running:
            self.initiate()

    def _reset_runner_state_if_idle(self, reason: str) -> None:
        """Reset runner buffers when there are no active/pending requests.

        Args:
            reason: Reason for the reset (for logging).
        """
        if not hasattr(self.runner, "reset_state"):
            return
        if self.num_running_requests > 0 or self.num_pending_requests > 0:
            logger.warning(
                "Skipping runner state reset during %s because there are active or pending requests "
                "(running=%d, pending=%d)",
                reason,
                self.num_running_requests,
                self.num_pending_requests,
            )
            return
        try:
            self.runner.reset_state()
            self._info("Runner state reset (%s)", reason)
        except Exception:
            logger.debug("Runner state reset encountered an error during %s", reason, exc_info=True)

    def start_profiling(
        self,
        output_dir: str,
        num_batches: int = 10,
        host_tracer_level: int | None = None,
        python_tracer_level: int | None = None,
    ) -> None:
        """Start a JAX profiler trace for the next num_batches scheduler updates.

        Enables JAX profiling to capture performance traces that can be visualized
        in TensorBoard or the Chrome trace viewer.

        Args:
            output_dir: Directory to write profiler output files.
            num_batches: Number of scheduler batches to trace before auto-stopping.
            host_tracer_level: Optional host tracer level (1-4). Higher levels
                capture more detail but add overhead.
            python_tracer_level: Optional Python tracer level for function-level
                profiling.

        Raises:
            RuntimeError: If a profiling session is already active.
            ValueError: If num_batches is not positive.
        """
        if self._profiling_active:
            raise RuntimeError("A profiling session is already active")
        if num_batches <= 0:
            raise ValueError("num_batches must be positive")

        profiler_options = jax.profiler.ProfileOptions()
        if host_tracer_level is not None:
            profiler_options.host_tracer_level = host_tracer_level
        if python_tracer_level is not None:
            profiler_options.python_tracer_level = python_tracer_level

        jax.profiler.start_trace(output_dir, profiler_options=profiler_options)
        self._profiling_active = True
        self._profiling_steps_remaining = num_batches
        self._profiling_output_dir = output_dir
        self._profiling_host_level = host_tracer_level
        self._profiling_python_level = python_tracer_level
        self._info(
            "Started profiler trace -> %s (batches=%d, host_tracer_level=%s, python_tracer_level=%s)",
            output_dir,
            num_batches,
            host_tracer_level,
            python_tracer_level,
        )

    def stop_profiling(self) -> None:
        """Stop the active JAX profiler trace, if any.

        Safe to call even if profiling is not active - will no-op.
        Resets all profiling state.
        """
        if not self._profiling_active:
            return
        try:
            jax.profiler.stop_trace()
            self._info("Stopped profiler trace -> %s", self._profiling_output_dir)
        finally:
            self._profiling_active = False
            self._profiling_steps_remaining = 0
            self._profiling_output_dir = None
            self._profiling_host_level = None
            self._profiling_python_level = None

    def _drain_runner_future(self, future, scheduler_output: SchedulerOutput) -> None:
        """Block on an in-flight async step and apply its outputs to the scheduler.

        Used by the overlap scheduler-loop path after dispatching a step
        via :meth:`eSurgeRunner.execute_model_async`. Waits for the host
        copies of sampled tokens to land, runs the scheduler's
        ``update_from_output`` on the resulting :class:`ModelRunnerOutput`,
        forwards engine-side outputs (sampled tokens, finished requests)
        to :meth:`_process_engine_outputs` for client delivery, and
        decrements the profiler step counter when an active trace is
        running.

        Args:
            future: Async handle returned by
                :meth:`eSurgeRunner.execute_model_async` (an
                ``_AsyncExecutionHandle`` or ``concurrent.futures.Future``).
            scheduler_output: The :class:`SchedulerOutput` that produced
                ``future``; needed by ``update_from_output`` to map
                sampled tokens back to the right requests.
        """
        wait_start = time.perf_counter()
        model_output = self.runner.wait_for_execution(future)
        wait_time = time.perf_counter() - wait_start
        total_tokens = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0)
        update_start = time.perf_counter()
        with self._scheduler_lock:
            engine_outputs = self.scheduler.update_from_output(scheduler_output, model_output)
        update_time = time.perf_counter() - update_start
        process_start = time.perf_counter()
        self._enqueue_engine_outputs(engine_outputs)
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

    def _handle_profiling_step(self) -> None:
        """Tick down the profiler step budget and auto-stop the trace at zero.

        Called from both scheduler-loop paths (sync and overlap) after
        each scheduler iteration completes. When a profiler trace is not
        active, returns immediately. Otherwise decrements
        ``_profiling_steps_remaining`` and, if it reaches zero, finalizes
        the trace via :meth:`stop_profiling` so users get a self-bounded
        profile without having to remember to stop it manually.
        """
        if not self._profiling_active:
            return
        if self._profiling_steps_remaining > 0:
            self._profiling_steps_remaining -= 1
        if self._profiling_steps_remaining <= 0:
            self.stop_profiling()
