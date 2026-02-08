# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

from __future__ import annotations

import os
import threading
import time
import traceback
import typing
from typing import Any

import flax
import jax
from eformer.loggings import get_logger

from ..scheduler import Scheduler, SchedulerOutput

logger = get_logger("eSurgeEngine")
MAX_CONSECUTIVE_SCHEDULER_ERRORS = int(os.environ.get("EASURGE_MAX_SCHEDULER_ERRORS", "5"))

if typing.TYPE_CHECKING:
    from easydel.infra import EasyDeLBaseModule


class EngineLifecycleMixin:
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
            return
        tb = self._scheduler_exception_tb
        if tb:
            raise RuntimeError(f"eSurge scheduler crashed: {exc}\n{tb}") from exc
        raise RuntimeError(f"eSurge scheduler crashed: {exc}") from exc

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

    def initiate(self) -> None:
        """Start the background scheduler thread.

        Initiates a daemon thread that continuously runs the scheduler loop,
        processing requests from the queue and updating outputs. This must
        be called before using generate() or stream() methods.

        The scheduler thread will:
        1. Schedule requests from the waiting queue
        2. Execute model forward passes
        3. Update request outputs with generated tokens
        4. Signal waiting threads when updates are available
        """
        with self._scheduler_lock:
            if self._scheduler_running:
                self._info("Scheduler loop is already running")
                return

            if self.runner.executor_manager.kv_pages is None:
                self.runner.initialize_kv_cache()
                self._kv_cache_valid = True

            # Clear any previous crash state before starting a fresh scheduler thread.
            self._scheduler_exception = None
            self._scheduler_exception_tb = None

            def _scheduler_loop():
                self._info("Starting background scheduler loop")
                consecutive_errors = 0
                max_consecutive_errors = MAX_CONSECUTIVE_SCHEDULER_ERRORS

                if not self._overlap_execution:
                    while self._scheduler_running:
                        try:
                            with self._scheduler_lock:
                                scheduler_output = self.scheduler.schedule()
                            model_output = self.runner.execute_model(scheduler_output)
                            with self._scheduler_lock:
                                engine_outputs = self.scheduler.update_from_output(scheduler_output, model_output)
                            if engine_outputs:
                                self._process_engine_outputs(engine_outputs)
                            # Reset error counter on success
                            consecutive_errors = 0
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

                pending_future: tuple[Any, SchedulerOutput] | None = None
                prefetched_schedule: SchedulerOutput | None = None

                def _can_prefetch_next(current: SchedulerOutput) -> bool:
                    # Only prefetch when the current batch is guaranteed not to
                    # generate new output tokens. That keeps scheduler state
                    # deterministic (no token-dependent stop conditions) while
                    # overlapping schedule work with device execution.
                    try:
                        for rid in current.num_scheduled_tokens:
                            req = self.scheduler.requests.get(rid)
                            if req is None:
                                continue
                            if req.num_computed_tokens >= req.num_tokens:
                                return False
                    except Exception:
                        return False
                    return True

                while self._scheduler_running:
                    try:
                        if pending_future is not None:
                            future, prev_sched_out = pending_future

                            # Opportunistically prefetch the next schedule while the
                            # current batch is still running on device (prefill-only).
                            if prefetched_schedule is None and _can_prefetch_next(prev_sched_out):
                                with self._scheduler_lock:
                                    prefetched_schedule = self.scheduler.schedule()

                            self._drain_runner_future(future, prev_sched_out)
                            pending_future = None

                        if prefetched_schedule is not None:
                            scheduler_output = prefetched_schedule
                            prefetched_schedule = None
                        else:
                            with self._scheduler_lock:
                                scheduler_output = self.scheduler.schedule()
                        future = self.runner.execute_model_async(scheduler_output)
                        pending_future = (future, scheduler_output)
                        # Reset error counter on success
                        consecutive_errors = 0
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

                        if consecutive_errors >= max_consecutive_errors:
                            logger.critical(
                                f"Scheduler loop encountered {consecutive_errors} consecutive errors. "
                                "Stopping scheduler to prevent resource exhaustion."
                            )
                            self._abort_scheduler_due_to_error(e)
                            break
                        time.sleep(0.01)

                if pending_future is not None:
                    try:
                        self._drain_runner_future(*pending_future)
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
        """Stop the background scheduler thread.

        Gracefully shuts down the scheduler loop and waits for the thread
        to terminate. Should be called when the engine is no longer needed
        to free resources.
        """
        self._stop_idle_monitor()
        with self._scheduler_lock:
            if not self._scheduler_running:
                self._info("Scheduler loop is not running")
                return
            self._info("Stopping background scheduler loop...")
            self._scheduler_running = False
            if self._scheduler_thread:
                self._scheduler_thread.join(timeout=5.0)
                if self._scheduler_thread.is_alive():
                    logger.warning("Scheduler thread did not stop gracefully")
                self._scheduler_thread = None
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
        """Pause the background scheduler without clearing queued state.

        Temporarily stops the scheduler thread while preserving request state.
        Use resume() to restart processing. Optionally destroys KV cache to
        free memory if destroy_pages_on_pause is enabled.

        Note:
            Does not abort pending requests - they will resume when resume() is called.
        """
        if not self._scheduler_running:
            self._info("Scheduler loop already paused or not running")
            self._paused = True
            return

        self._info("Pausing eSurge scheduler loop...")
        self.terminate()
        self._paused = True
        self._drain_pipeline_workers("pause")
        if self.destroy_pages_on_pause:
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
        """Resume the scheduler if it was paused.

        Restarts the background scheduler thread after a pause(). If KV cache
        was destroyed during pause, it will be reinitialized before processing
        resumes.

        Note:
            Safe to call even if the scheduler is already running - will no-op.
        """
        if self._scheduler_running:
            self._info("Scheduler loop already running")
            return
        self._info("Resuming eSurge scheduler loop...")
        self._drain_pipeline_workers("resume")
        if self.destroy_pages_on_pause and not self._kv_cache_valid:
            self.runner.initialize_kv_cache()
            self._kv_cache_valid = True
            self._log_cache_event("kv_cache_reinitialized", {"reason": "resume"})
        self.initiate()

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

        if model is None:
            model = flax.nnx.merge(graphdef, graphstate, graphother)
        if graphstate is None:
            graphstate = model.graphstate
        if graphother is None:
            graphother = model.graphother
        graphdef = model.esurge_graphdef

        self.runner.update_model_weights(
            graphdef=graphdef,
            graphstate=graphstate,
            graphother=graphother,
            reset_state=True,
        )
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
            enable_prefix_caching=self._scheduler_enable_prefix_caching,
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
        """Wait for an async runner execution and process results.

        Args:
            future: The async execution future to wait on.
            scheduler_output: The scheduler output that triggered the execution.
        """
        model_output = self.runner.wait_for_execution(future)
        with self._scheduler_lock:
            engine_outputs = self.scheduler.update_from_output(scheduler_output, model_output)
        if engine_outputs:
            self._process_engine_outputs(engine_outputs)
        self._handle_profiling_step()

    def _handle_profiling_step(self) -> None:
        """Handle profiling step counter and auto-stop when complete.

        Decrements the profiling step counter and stops profiling when
        the configured number of batches has been traced.
        """
        if not self._profiling_active:
            return
        if self._profiling_steps_remaining > 0:
            self._profiling_steps_remaining -= 1
        if self._profiling_steps_remaining <= 0:
            self.stop_profiling()
