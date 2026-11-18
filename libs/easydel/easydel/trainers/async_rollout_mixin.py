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

"""Shared asynchronous rollout lookahead mixin for rollout-capable trainers."""

from __future__ import annotations

import concurrent.futures
import contextlib
import time
import typing as tp
from collections import deque
from dataclasses import dataclass

import jax
from eformer.loggings import get_logger

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossMetrics
from easydel.utils.helpers import capture_time

from ._shared import OwnedPolicySnapshot
from ._shared.rollout_lifecycle import after_futures
from .base_trainer import EasyDeLBreakRequest, EasyDeLPreemptionSignal, EasyDeLTimerError
from .metrics import BaseProgressBar, MetricsTracker, StepMetrics

logger = get_logger(__name__)


@dataclass
class AsyncRolloutResult:
    """Preprocessed rollout batch produced by a background worker thread."""

    batch: dict[str, jax.Array]
    informations: dict[str, float | int | str]
    produced_at_step: int
    preprocessing_time: float


class AsyncRolloutMixin:
    """Mixin enabling ``arguments.async_mode=True`` on any rollout-capable trainer.

    When ``arguments.async_mode`` is ``False`` (the default), ``_train_epoch``
    delegates directly to the synchronous ``super()._train_epoch(...)`` path.
    When ``arguments.async_mode`` is ``True``, rollout preprocessing runs in a
    background thread against a params-only :class:`OwnedPolicySnapshot` while
    the main thread executes compiled gradient updates, enforcing
    ``max_inflight_tasks``, ``queue_maxsize``, ``weight_sync_steps``, and
    ``max_staleness``.
    """

    supports_async_rollout: tp.ClassVar[bool] = True

    def _take_rollout_snapshot(
        self,
        state: EasyDeLState,
        *,
        policy_step: int,
        cache_scope_key: str,
    ) -> OwnedPolicySnapshot:
        """Build an inference-only policy snapshot for background rollouts."""
        return OwnedPolicySnapshot.from_training_state(
            state,
            policy_step=policy_step,
            cache_scope_key=cache_scope_key,
        )

    def _sync_rollout_snapshot(
        self,
        snapshot: OwnedPolicySnapshot | None,
        policy_state: EasyDeLState,
        policy_step: int,
        *,
        force: bool = False,
        can_release: bool = True,
        cache_scope_key: str = "async-rollout",
    ) -> tuple[OwnedPolicySnapshot, float]:
        """Refresh the rollout snapshot when ``weight_sync_steps`` has elapsed."""
        sync_interval = int(self.arguments.weight_sync_steps)  # type: ignore[attr-defined]
        if snapshot is not None and not force and policy_step - snapshot.policy_step < sync_interval:
            return snapshot, 0.0
        with capture_time() as sync_time:
            if snapshot is not None:
                if can_release:
                    snapshot.release()
                else:
                    logger.debug("Rollout in flight during policy sync; deferring snapshot release to GC")
                snapshot = None
            snapshot = self._take_rollout_snapshot(
                policy_state,
                policy_step=policy_step,
                cache_scope_key=cache_scope_key,
            )
        return snapshot, float(sync_time())

    def _store_buffered_grpo_batch(
        self,
        model_batch: dict[str, jax.Array],
        metrics: dict[str, float | int | str],
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        """Disable GRPO batch-reuse caching while async rollout lookahead is active."""
        if bool(getattr(self, "_async_grpo_force_old_logps", False)):
            metrics = dict(metrics)
            metrics["generation_reused"] = 0
            metrics["generation_reuse_span"] = 1
            metrics["generation_reuse_remaining"] = 0
            self._buffered_grpo_batch = None
            self._buffered_grpo_remaining = 0
        else:
            parent_store = getattr(super(), "_store_buffered_grpo_batch", None)
            if parent_store is not None:
                model_batch, metrics = parent_store(model_batch, metrics)
        if bool(getattr(self.arguments, "async_mode", False)):  # type: ignore[attr-defined]
            return model_batch, {
                **metrics,
                "async_rollout/max_inflight_tasks": int(self.arguments.max_inflight_tasks),  # type: ignore[attr-defined]
                "async_rollout/max_staleness": int(self.arguments.max_staleness),  # type: ignore[attr-defined]
                "async_rollout/weight_sync_steps": int(self.arguments.weight_sync_steps),  # type: ignore[attr-defined]
                "async_rollout/esurge_async_scheduling": int(bool(self.arguments.esurge_async_scheduling)),  # type: ignore[attr-defined]
                "async_rollout/esurge_overlap_execution": int(bool(self.arguments.esurge_overlap_execution)),  # type: ignore[attr-defined]
            }
        return model_batch, metrics

    def _generation_reuse_span(self) -> int:
        """Force sampling-policy log-probs for async rollouts without batch reuse."""
        parent_span = getattr(super(), "_generation_reuse_span", None)
        span = int(parent_span()) if parent_span is not None else 1
        if bool(getattr(self, "_async_grpo_force_old_logps", False)):
            return max(span, 2)
        return span

    def _preprocess_async_rollout(
        self,
        *,
        state: EasyDeLState,
        batch: dict[str, object],
        produced_at_step: int,
    ) -> AsyncRolloutResult:
        """Generate, score, and pack one rollout batch in the background worker."""
        with capture_time() as preprocessing_time_fn:
            self._async_grpo_force_old_logps = True
            try:
                model_batch, informations = self._preprocess_batch_input(  # type: ignore[attr-defined]
                    state=state,
                    batch=batch,
                    is_train=True,
                )
            finally:
                self._async_grpo_force_old_logps = False
        informations = dict(informations)
        informations["async_rollout/rollout_produced_at_step"] = produced_at_step
        return AsyncRolloutResult(
            batch=model_batch,
            informations=informations,
            produced_at_step=produced_at_step,
            preprocessing_time=float(preprocessing_time_fn()),
        )

    def _execute_preprocessed_train_step(
        self,
        state: EasyDeLState,
        batch: dict[str, jax.Array],
        informations: dict[str, float | int | str],
    ) -> tuple[EasyDeLState, LossMetrics, BaseException | None]:
        """Run the compiled training step on an already-preprocessed rollout batch."""
        if self.pruning_module is not None:  # type: ignore[attr-defined]
            state = state.replace(
                graphstate=self.pruning_module.pre_forward_update(  # type: ignore[attr-defined]
                    state.graphstate,
                    state.opt_state,
                )
            )
        metrics = LossMetrics()
        try:
            state, metrics = jax.block_until_ready(
                self.sharded_training_step_function(  # type: ignore[attr-defined]
                    state,
                    batch,
                    *self._train_shared_fn_extra_args,  # type: ignore[attr-defined]
                    *self._train_shared_fn_static_args,  # type: ignore[attr-defined]
                )
            )
            if self.pruning_module is not None:  # type: ignore[attr-defined]
                state = state.replace(
                    graphstate=self.pruning_module.post_gradient_update(  # type: ignore[attr-defined]
                        state.graphstate,
                        state.opt_state,
                    )
                )
            if informations:
                merged = dict(informations)
                if metrics.other_metrics is not None:
                    merged.update(metrics.other_metrics)
                metrics.other_metrics = merged
            return state, metrics, None
        except (
            KeyboardInterrupt,
            EasyDeLTimerError,
            EasyDeLBreakRequest,
            TypeError,
        ) as run_exception:
            return state, metrics, run_exception
        except Exception as run_exception:
            if self._is_memory_oom_exception(run_exception):  # type: ignore[attr-defined]
                annotated_exception = self._augment_memory_oom_exception(run_exception)  # type: ignore[attr-defined]
                return state, metrics, annotated_exception
            raise

    def _train_epoch(
        self,
        state: EasyDeLState,
        train_dataset,
        train_iter,
        metrics_tracker: MetricsTracker,
        step_metrics: StepMetrics,
        pbar: BaseProgressBar,
        epoch: int,
        *,
        epoch_start_step: int | None = None,
        epoch_end_step: int | None = None,
    ):
        """Dispatch to async rollout lookahead when ``arguments.async_mode=True``."""
        if not isinstance(self, AsyncRolloutMixin) or bool(getattr(self.arguments, "async_mode", False)):  # type: ignore[attr-defined]
            return AsyncRolloutMixin._train_epoch_async(
                self,
                state=state,
                train_dataset=train_dataset,
                train_iter=train_iter,
                metrics_tracker=metrics_tracker,
                step_metrics=step_metrics,
                pbar=pbar,
                epoch=epoch,
                epoch_start_step=epoch_start_step,
                epoch_end_step=epoch_end_step,
            )
        return super()._train_epoch(  # type: ignore[misc]
            state=state,
            train_dataset=train_dataset,
            train_iter=train_iter,
            metrics_tracker=metrics_tracker,
            step_metrics=step_metrics,
            pbar=pbar,
            epoch=epoch,
            epoch_start_step=epoch_start_step,
            epoch_end_step=epoch_end_step,
        )

    def _train_epoch_async(
        self,
        state: EasyDeLState,
        train_dataset,
        train_iter,
        metrics_tracker: MetricsTracker,
        step_metrics: StepMetrics,
        pbar: BaseProgressBar,
        epoch: int,
        *,
        epoch_start_step: int | None = None,
        epoch_end_step: int | None = None,
    ):
        """Run one training epoch with asynchronous rollout lookahead."""
        if not getattr(self, "_async_grpo_rollout_idle", True):
            raise RuntimeError("A previous rollout worker is still retiring; wait before starting another epoch.")
        data_collator = self.data_collator  # type: ignore[attr-defined]
        if data_collator is None:

            def data_collator(x):
                return x

        if self.max_training_steps is None:  # type: ignore[attr-defined]
            raise RuntimeError("max_training_steps must be set before training")
        if epoch_start_step is None or epoch_end_step is None:
            epoch_start_step, epoch_end_step = self._get_epoch_step_bounds(epoch)  # type: ignore[attr-defined]
        epoch_total_steps = max(epoch_end_step - epoch_start_step, 1)
        run_exception: BaseException | None = None
        pending_future: concurrent.futures.Future[AsyncRolloutResult] | None = None
        pending_data_time = 0.0
        pending_sync_time = 0.0
        pending_batch = None
        rollout_snapshot: OwnedPolicySnapshot | None = None
        rollout_cache_scope_key = f"{state.esurge_cache_scope_key}-async-rollout"
        max_workers = 1
        queue_capacity = int(self.arguments.max_inflight_tasks)  # type: ignore[attr-defined]
        if self.arguments.queue_maxsize:  # type: ignore[attr-defined]
            queue_capacity = min(queue_capacity, int(self.arguments.queue_maxsize))  # type: ignore[attr-defined]
        pending_rollouts: deque[tuple[concurrent.futures.Future, dict[str, object], float, float]] = deque()
        submitted_futures: set[concurrent.futures.Future] = set()
        rollout_timeout = min(
            float(self.arguments.request_timeout),  # type: ignore[attr-defined]
            float(self.arguments.heartbeat_stale_after_s),  # type: ignore[attr-defined]
        )

        def fetch_batch() -> tuple[dict[str, object], float]:
            nonlocal train_iter
            with capture_time() as data_collection_time:
                raw_batch, train_iter = self._get_next_batch(train_iter, train_dataset)  # type: ignore[attr-defined]
                collated = data_collator(raw_batch)
            return collated, float(data_collection_time())

        def submit_rollout(
            executor: concurrent.futures.ThreadPoolExecutor,
            *,
            rollout_state: EasyDeLState,
            rollout_batch: dict[str, object],
            produced_at_step: int,
        ) -> concurrent.futures.Future[AsyncRolloutResult]:
            submitted_futures.difference_update(future for future in tuple(submitted_futures) if future.done())
            future = executor.submit(
                self._preprocess_async_rollout,
                state=rollout_state,
                batch=rollout_batch,
                produced_at_step=produced_at_step,
            )
            submitted_futures.add(future)
            return future

        def ensure_rollout_snapshot(
            policy_state: EasyDeLState,
            policy_step: int,
            *,
            force: bool = False,
        ) -> tuple[OwnedPolicySnapshot, float]:
            nonlocal rollout_snapshot
            if pending_rollouts and rollout_snapshot is not None:
                return rollout_snapshot, 0.0
            rollout_snapshot, sync_seconds = self._sync_rollout_snapshot(
                rollout_snapshot,
                policy_state,
                policy_step,
                force=force,
                can_release=pending_future is None,
                cache_scope_key=rollout_cache_scope_key,
            )
            return rollout_snapshot, sync_seconds

        def release_rollout_snapshot() -> None:
            nonlocal rollout_snapshot
            if rollout_snapshot is not None:
                rollout_snapshot.release()
                rollout_snapshot = None
            self._async_grpo_retiring_futures = ()
            self._async_grpo_rollout_idle = True

        with contextlib.ExitStack() as cleanup:
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="async-trainer-rollout",
            )

            def retire_workers():
                self._async_grpo_rollout_idle = False
                readers = tuple(submitted_futures)
                for future in readers:
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                self._async_grpo_retiring_futures = readers
                after_futures(readers, release_rollout_snapshot)

            cleanup.callback(retire_workers)
            while True:
                with capture_time() as iteration_time:
                    current_step = int(jax.device_get(state.step))
                    if current_step >= self.max_training_steps or current_step >= epoch_end_step:  # type: ignore[attr-defined]
                        break

                    try:
                        step_metrics.start_step()
                        state = self.on_step_start(state=state, step=current_step)  # type: ignore[attr-defined]

                        with capture_time() as rollout_wait_time:
                            refresh_due = (
                                rollout_snapshot is not None
                                and current_step - rollout_snapshot.policy_step >= int(self.arguments.weight_sync_steps)  # type: ignore[attr-defined]
                            )
                            if refresh_due and pending_rollouts:
                                queued = list(pending_rollouts)
                                pending_rollouts.clear()
                                for future, _, _, _ in queued:
                                    if not future.cancel():
                                        future.result(timeout=rollout_timeout)
                                snapshot, sync_time = ensure_rollout_snapshot(state, current_step, force=True)
                                for _, raw_batch, data_time, _ in queued:
                                    future = submit_rollout(
                                        executor,
                                        rollout_state=snapshot.state,
                                        rollout_batch=raw_batch,
                                        produced_at_step=snapshot.policy_step,
                                    )
                                    pending_rollouts.append((future, raw_batch, data_time, sync_time))
                            if pending_rollouts:
                                pending_future, pending_batch, pending_data_time, pending_sync_time = (
                                    pending_rollouts.popleft()
                                )
                            if pending_future is None:
                                pending_batch, pending_data_time = fetch_batch()
                                snapshot, pending_sync_time = ensure_rollout_snapshot(state, current_step)
                                pending_future = submit_rollout(
                                    executor,
                                    rollout_state=snapshot.state,
                                    rollout_batch=pending_batch,
                                    produced_at_step=snapshot.policy_step,
                                )
                            if not bool(getattr(self, "_async_grpo_train_compiled_once", False)):
                                first_timeout = float(
                                    getattr(
                                        self.arguments,  # type: ignore[attr-defined]
                                        "first_rollout_timeout",
                                        max(rollout_timeout, 3600.0),
                                    )
                                )
                                deadline = time.monotonic() + first_timeout
                                while True:
                                    remaining_s = deadline - time.monotonic()
                                    if remaining_s <= 0:
                                        raise TimeoutError(f"First async rollout timed out after {first_timeout:.1f}s.")
                                    try:
                                        rollout = pending_future.result(timeout=min(60.0, remaining_s))
                                        break
                                    except concurrent.futures.TimeoutError:
                                        elapsed_s = first_timeout - max(0.0, deadline - time.monotonic())
                                        logger.debug(
                                            "Waiting for first async rollout / cold compile (%.0fs / %.0fs)...",
                                            elapsed_s,
                                            first_timeout,
                                        )
                            else:
                                rollout = pending_future.result(timeout=rollout_timeout)
                        rollout_wait_seconds = float(rollout_wait_time())
                        rollout_sync_seconds = pending_sync_time
                        pending_sync_time = 0.0
                        pending_future = None
                    except (
                        KeyboardInterrupt,
                        EasyDeLTimerError,
                        EasyDeLBreakRequest,
                        EasyDeLPreemptionSignal,
                    ) as exc:
                        if pending_future is not None:
                            pending_future.cancel()
                        return state, exc, train_iter

                    rollout_staleness = max(0, current_step - int(rollout.produced_at_step))
                    if rollout_staleness > int(self.arguments.max_staleness):  # type: ignore[attr-defined]
                        queued_batches = []
                        while pending_rollouts:
                            future, queued_batch, data_time, _ = pending_rollouts.popleft()
                            if not future.cancel():
                                future.result(timeout=rollout_timeout)
                            queued_batches.append((queued_batch, data_time))
                        snapshot, rollout_sync_seconds = ensure_rollout_snapshot(
                            state,
                            current_step,
                            force=True,
                        )
                        pending_future = submit_rollout(
                            executor,
                            rollout_state=snapshot.state,
                            rollout_batch=tp.cast(dict[str, object], pending_batch),
                            produced_at_step=snapshot.policy_step,
                        )
                        rollout = pending_future.result(timeout=rollout_timeout)
                        pending_future = None
                        rollout_staleness = 0
                        rollout_wait_seconds = 0.0
                        for queued_batch, data_time in queued_batches:
                            future = submit_rollout(
                                executor,
                                rollout_state=snapshot.state,
                                rollout_batch=queued_batch,
                                produced_at_step=snapshot.policy_step,
                            )
                            pending_rollouts.append((future, queued_batch, data_time, 0.0))

                    compiled_train_ready = bool(getattr(self, "_async_grpo_train_compiled_once", False))
                    schedule_next = (
                        compiled_train_ready
                        and current_step + 1 < self.max_training_steps  # type: ignore[attr-defined]
                        and current_step + 1 < epoch_end_step
                    )
                    next_data_time = 0.0
                    if schedule_next:
                        remaining = min(self.max_training_steps, epoch_end_step) - current_step - 1  # type: ignore[attr-defined]
                        while len(pending_rollouts) < min(queue_capacity, remaining):
                            next_batch, data_time = fetch_batch()
                            next_data_time += data_time
                            snapshot, sync_time = ensure_rollout_snapshot(state, current_step)
                            future = submit_rollout(
                                executor,
                                rollout_state=snapshot.state,
                                rollout_batch=next_batch,
                                produced_at_step=snapshot.policy_step,
                            )
                            pending_rollouts.append((future, next_batch, data_time, sync_time))

                    rollout.informations["async_rollout/rollout_staleness"] = rollout_staleness
                    rollout.informations["async_rollout/rollout_wait_time"] = rollout_wait_seconds
                    rollout.informations["async_rollout/rollout_preprocessing_time"] = rollout.preprocessing_time
                    rollout.informations["async_rollout/policy_sync_time"] = rollout_sync_seconds
                    rollout.informations["async_rollout/next_rollout_scheduled"] = int(schedule_next)
                    if rollout_snapshot is not None:
                        rollout.informations["async_rollout/policy_snapshot_gib"] = rollout_snapshot.nbytes / 1024**3

                    with self.train_tracker.trace_compilation():  # type: ignore[attr-defined]
                        with capture_time() as execution_time:
                            state, metrics, run_exception = self._execute_preprocessed_train_step(
                                state=state,
                                batch=rollout.batch,
                                informations=rollout.informations,
                            )
                            metrics.execution_time = execution_time()
                            current_step = int(jax.device_get(state.step))
                    self._async_grpo_train_compiled_once = True
                    if run_exception is not None:
                        if pending_future is not None:
                            pending_future.cancel()
                        return state, run_exception, train_iter

                    self._maybe_start_profiler(current_step)  # type: ignore[attr-defined]
                    try:
                        mean_loss, mean_accuracy = metrics_tracker.update(
                            loss=metrics.loss,
                            accuracy=metrics.accuracy,
                            step=current_step,
                        )
                        metrics = self.apply_training_hooks(metrics=metrics)  # type: ignore[attr-defined]
                        train_metrics = step_metrics.calculate(
                            metrics=metrics,
                            current_step=current_step,
                            learning_rate=(
                                self.scheduler(current_step)  # type: ignore[attr-defined]
                                if self.scheduler is not None  # type: ignore[attr-defined]
                                else self.arguments.learning_rate  # type: ignore[attr-defined]
                            ),
                            epoch=epoch,
                            epoch_progress=min(max((current_step - epoch_start_step) / epoch_total_steps, 0.0), 1.0),
                            flops_per_token=self._backward_flops_per_token,  # type: ignore[attr-defined]
                            extra_flops_per_token=self._extra_backward_flops_per_token,  # type: ignore[attr-defined]
                            batch_size=self.training_batch_size,  # type: ignore[attr-defined]
                            seq_length=self.arguments.max_length,  # type: ignore[attr-defined]
                            mean_loss=mean_loss,
                            mean_accuracy=mean_accuracy,
                            mode="train",
                        )
                        train_metrics["performance/data_collection_time"] = float(pending_data_time)
                        train_metrics["performance/async_next_data_collection_time"] = float(next_data_time)
                        state, metrics = self.on_step_end(  # type: ignore[attr-defined]
                            state=state,
                            metrics=metrics,
                            step=current_step,
                        )
                        with capture_time() as logging_time:
                            self.log_metrics(  # type: ignore[attr-defined]
                                metrics=train_metrics,
                                pbar=pbar,
                                step=current_step,
                                mode="train",
                            )
                        if self._should_save_tpu_preemption_checkpoint(current_step):  # type: ignore[attr-defined]
                            self._save_tpu_preemption_checkpoint(state=state, step=current_step)  # type: ignore[attr-defined]
                            return state, EasyDeLPreemptionSignal("TPU preemption checkpoint saved"), train_iter
                        with capture_time() as weight_distribution_time:
                            self.log_weight_distribution(state=state, step=current_step)  # type: ignore[attr-defined]
                        with capture_time() as watchers_time:
                            self.log_watchers(state=state, step=current_step)  # type: ignore[attr-defined]
                        with capture_time() as generation_time:
                            try:
                                self.maybe_generate(state=state, step=current_step, metrics=metrics)  # type: ignore[attr-defined]
                            except Exception as exc:
                                logger.warning("Preview generation hook failed: %s", exc, exc_info=True)
                        with capture_time() as benchmark_time:
                            try:
                                self.maybe_benchmark(state=state, step=current_step)  # type: ignore[attr-defined]
                            except Exception as exc:
                                logger.warning("Benchmark hook failed: %s", exc, exc_info=True)
                        with capture_time() as checkpoint_time:
                            self._save_checkpoint_for_step(  # type: ignore[attr-defined]
                                state=state,
                                step=current_step,
                                merge_lora_before_save=self.arguments.merge_lora_before_save,  # type: ignore[attr-defined]
                            )
                        with capture_time() as evaluation_time:
                            if self._should_run_evaluation(current_step):  # type: ignore[attr-defined]
                                for _ in self.eval(model_state=state):  # type: ignore[attr-defined]
                                    ...
                        self.log_metrics(  # type: ignore[attr-defined]
                            metrics={
                                "performance/iteration_time": float(iteration_time()),
                                "performance/logging_time": float(logging_time()),
                                "performance/weight_distribution_time": float(weight_distribution_time()),
                                "performance/watchers_time": float(watchers_time()),
                                "performance/generation_time": float(generation_time()),
                                "performance/benchmark_time": float(benchmark_time()),
                                "performance/checkpoint_time": float(checkpoint_time()),
                                "performance/evaluation_time": float(evaluation_time()),
                            },
                            pbar=pbar,
                            step=current_step,
                            mode="train",
                            update_progress=False,
                        )
                    except (
                        KeyboardInterrupt,
                        EasyDeLTimerError,
                        EasyDeLBreakRequest,
                        EasyDeLPreemptionSignal,
                        TypeError,
                    ) as exc:
                        if pending_future is not None:
                            pending_future.cancel()
                        return state, exc, train_iter

                    log_step_info = getattr(self, "_log_step_info", None)
                    if log_step_info is not None:
                        if getattr(self.arguments, "performance_mode", False):  # type: ignore[attr-defined]
                            with jax.spmd_mode("allow_all"):
                                log_step_info(
                                    pbar=pbar,
                                    metrics=metrics,
                                    train_metrics=train_metrics,
                                    step=current_step,
                                    epoch=epoch,
                                    metrics_tracker=metrics_tracker,
                                    iteration_time=iteration_time(),
                                )
                        else:
                            log_step_info(
                                pbar=pbar,
                                metrics=metrics,
                                train_metrics=train_metrics,
                                step=current_step,
                                epoch=epoch,
                                metrics_tracker=metrics_tracker,
                                iteration_time=iteration_time(),
                            )
                    maybe_stop_profiler = getattr(self, "_maybe_stop_profiler", None)
                    if maybe_stop_profiler is not None:
                        maybe_stop_profiler(current_step)
                    maybe_stop_step_trace = getattr(self, "_maybe_stop_step_trace", None)
                    if maybe_stop_step_trace is not None:
                        maybe_stop_step_trace(current_step)
                    should_block = getattr(self, "_profiler_should_block_until_ready", None)
                    if should_block is not None and should_block():
                        jax.block_until_ready(state)

        return state, run_exception, train_iter


__all__ = (
    "AsyncRolloutMixin",
    "AsyncRolloutResult",
)
