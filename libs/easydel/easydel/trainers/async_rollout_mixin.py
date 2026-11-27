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
import threading
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

_THREAD_FLAGS_LOCK = threading.Lock()


def with_legacy_async_grpo_aliases(metrics: dict[str, tp.Any]) -> dict[str, tp.Any]:
    """Return ``metrics`` plus a deprecated ``async_grpo/<name>`` copy of every ``async_rollout/<name>`` key.

    The ``async_grpo/*`` namespace is deprecated in favour of ``async_rollout/*``
    and is only emitted by trainers that set
    :attr:`AsyncRolloutMixin.emit_legacy_async_grpo_metrics`.
    """
    prefix = "async_rollout/"
    aliased = dict(metrics)
    for key, value in metrics.items():
        if key.startswith(prefix):
            aliased.setdefault("async_grpo/" + key[len(prefix) :], value)
    return aliased


@dataclass
class AsyncRolloutResult:
    """Preprocessed rollout batch produced by a background worker thread."""

    batch: dict[str, jax.Array]
    informations: dict[str, float | int | str]
    produced_at_step: int
    preprocessing_time: float
    minibatches: tuple[dict[str, jax.Array], ...] = ()


def _steps_per_rollout(trainer: object) -> int:
    """Optimizer steps per consumed rollout (``1`` when the trainer defines no hook)."""
    hook = getattr(trainer, "_optimizer_steps_per_rollout", None)
    steps = int(hook()) if hook is not None else 1
    if steps < 1:
        raise ValueError(f"_optimizer_steps_per_rollout() must be >= 1, got {steps}")
    return steps


class AsyncRolloutMixin:
    """Mixin enabling ``arguments.async_mode=True`` on any rollout-capable trainer.

    When ``arguments.async_mode`` is ``False`` (the default), ``_train_epoch``
    delegates directly to the synchronous ``super()._train_epoch(...)`` path.
    When ``arguments.async_mode`` is ``True``, rollout preprocessing runs in a
    background thread against a params-only :class:`OwnedPolicySnapshot` while
    the main thread executes compiled gradient updates, enforcing
    ``max_inflight_tasks``, ``queue_maxsize``, ``weight_sync_steps``, and
    ``max_staleness``.

    Lookahead rollouts use deadline admission: a rollout anchored at snapshot
    version ``v`` must be consumed by step ``v + max_staleness``. A lookahead
    rollout is only fetched and started when every queued deadline stays
    feasible (see :func:`~easydel.trainers._online_rl.async_rollout.admission_is_feasible`);
    otherwise it is deferred and retried on a later step, once a newer
    snapshot makes it admissible. The take-side staleness check that
    regenerates an over-stale rollout remains as a safety net. Per-epoch
    counts are logged as ``async_rollout/admission_deferred`` and
    ``async_rollout/stale_dropped``.

    Attributes:
        emit_legacy_async_grpo_metrics: When ``True``, every ``async_rollout/*``
            metric is also logged under the deprecated ``async_grpo/*`` name.
    """

    supports_async_rollout: tp.ClassVar[bool] = True
    emit_legacy_async_grpo_metrics: tp.ClassVar[bool] = False

    def _optimizer_steps_per_rollout(self) -> int:
        """Number of optimizer steps (``state.step`` increments) one consumed rollout produces.

        Async bookkeeping (staleness, deadline admission, weight-sync refresh, lookahead counts)
        runs in rollout units, i.e. policy versions; a step lag converts to versions by
        floor-dividing by this value. Trainers that replay a rollout for several updates override it.
        """
        return 1

    @property
    def _async_grpo_force_old_logps(self) -> bool:
        """Per-thread flag set while the background worker preprocesses a rollout.

        Thread-local so main-thread work (evaluation, previews) running while a
        rollout is being generated never observes the worker's flag.
        """
        flags = self.__dict__.get("_async_rollout_thread_flags")
        return bool(getattr(flags, "force_old_logps", False)) if flags is not None else False

    @_async_grpo_force_old_logps.setter
    def _async_grpo_force_old_logps(self, value: bool) -> None:
        flags = self.__dict__.get("_async_rollout_thread_flags")
        if flags is None:
            with _THREAD_FLAGS_LOCK:
                flags = self.__dict__.setdefault("_async_rollout_thread_flags", threading.local())
        flags.force_old_logps = bool(value)

    @property
    def _is_async_rollout_worker_thread(self) -> bool:
        """Whether the current thread is executing a background async rollout worker task."""
        flags = self.__dict__.get("_async_rollout_thread_flags")
        return bool(getattr(flags, "in_worker", False)) if flags is not None else False

    @property
    def _async_rollout_policy_version(self) -> int | None:
        """Snapshot step (policy version) active on the current async rollout worker thread."""
        flags = self.__dict__.get("_async_rollout_thread_flags")
        if flags is None:
            return None
        value = getattr(flags, "policy_version", None)
        return int(value) if value is not None else None

    @_async_rollout_policy_version.setter
    def _async_rollout_policy_version(self, value: int | None) -> None:
        flags = self.__dict__.get("_async_rollout_thread_flags")
        if flags is None:
            with _THREAD_FLAGS_LOCK:
                flags = self.__dict__.setdefault("_async_rollout_thread_flags", threading.local())
        flags.policy_version = None if value is None else int(value)

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
        """Refresh the rollout snapshot once ``weight_sync_steps`` rollouts (policy versions) have elapsed."""
        sync_interval = int(self.arguments.weight_sync_steps) * _steps_per_rollout(self)  # type: ignore[attr-defined]
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
            self._async_rollout_policy_version = int(produced_at_step)
            try:
                model_batch, informations = self._preprocess_batch_input(  # type: ignore[attr-defined]
                    state=state,
                    batch=batch,
                    is_train=True,
                )
            finally:
                self._async_grpo_force_old_logps = False
                self._async_rollout_policy_version = None
        informations = dict(informations)
        informations["async_rollout/rollout_produced_at_step"] = produced_at_step

        steps_per_generation = max(1, int(getattr(self, "steps_per_generation", 1) or 1))
        num_iterations = max(1, int(getattr(self, "num_iterations", 1) or 1))
        reuse_span = max(1, steps_per_generation * num_iterations)
        slice_chunks = getattr(self, "_slice_model_batch_chunks", None)
        if reuse_span > 1 and callable(slice_chunks):
            prompt_ids = model_batch.get("prompt_ids")
            expected_step_batch = int(getattr(self.arguments, "total_batch_size", 1) or 1)  # type: ignore[attr-defined]
            if (
                steps_per_generation > 1
                and prompt_ids is not None
                and int(prompt_ids.shape[0]) >= steps_per_generation * expected_step_batch
                and int(prompt_ids.shape[0]) % steps_per_generation == 0
            ):
                chunks = slice_chunks(model_batch, steps_per_generation)
                schedule = tuple(mb for _ in range(num_iterations) for mb in chunks)
            else:
                schedule = tuple(dict(model_batch) for _ in range(reuse_span))
            first_batch = schedule[0]
        else:
            schedule = (model_batch,)
            first_batch = model_batch

        return AsyncRolloutResult(
            batch=first_batch,
            informations=informations,
            produced_at_step=produced_at_step,
            preprocessing_time=float(preprocessing_time_fn()),
            minibatches=schedule,
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
            if getattr(self, "_has_buckets", False):
                bi = self._bucket_for_step  # type: ignore[attr-defined]
                base_graphdef = state.graphdef
                bucket_state = state.replace(graphdef=self._bucket_graphdefs[bi])  # type: ignore[attr-defined]
                state, metrics = jax.block_until_ready(
                    self._bucket_step_fns[bi](  # type: ignore[attr-defined]
                        bucket_state,
                        batch,
                        *self._bucket_extra_args[bi],  # type: ignore[attr-defined]
                        *self._bucket_static_args[bi],  # type: ignore[attr-defined]
                    )
                )
                state = state.replace(graphdef=base_graphdef)
            else:
                dispatch = getattr(self, "_dispatch_train_step", None)
                if callable(dispatch):
                    state, metrics = jax.block_until_ready(dispatch(state, batch))
                else:
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
                if hasattr(metrics, "replace"):
                    metrics = metrics.replace(other_metrics=merged)
                else:
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

    def _wait_for_retiring_async_rollouts(self, timeout: float | None = None) -> bool:
        """Wait until any retiring background rollout workers finish and release their snapshot."""
        idle_event = getattr(self, "_async_rollout_idle_event", None)
        if idle_event is None or idle_event.is_set():
            return True
        if timeout is None:
            args = getattr(self, "arguments", None)
            if args is not None:
                candidates = [
                    float(v)
                    for v in (
                        getattr(args, "request_timeout", None),
                        getattr(args, "heartbeat_stale_after_s", None),
                    )
                    if v is not None and float(v) > 0
                ]
                timeout = min(candidates) if candidates else 30.0
            else:
                timeout = 30.0
        return bool(idle_event.wait(timeout=timeout))

    def release_generation_runtime(
        self,
        *,
        state: EasyDeLState | None = None,
        extra_states: tp.Iterable[EasyDeLState | None] = (),
        clear_esurge_compiled_cache: bool = False,
    ) -> None:
        """Wait for any retiring async rollout workers before releasing the generation runtime."""
        if self._async_grpo_force_old_logps or self._is_async_rollout_worker_thread:
            # Running on a background rollout worker thread: never block on _async_rollout_idle_event
            # (which only sets after this worker's own future finishes -> self-deadlock). If workers
            # are retiring, return immediately so this worker finishes and lets the main thread tear down.
            idle_event = getattr(self, "_async_rollout_idle_event", None)
            if idle_event is not None and not idle_event.is_set():
                return
        elif not self._wait_for_retiring_async_rollouts():
            logger.warning(
                "Retiring async rollout workers did not finish before release_generation_runtime timeout; "
                "skipping eSurge teardown to avoid racing an active worker."
            )
            return
        parent_release = getattr(super(), "release_generation_runtime", None)
        if callable(parent_release):
            parent_release(
                state=state,
                extra_states=extra_states,
                clear_esurge_compiled_cache=clear_esurge_compiled_cache,
            )

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
        # Reuse the live state's eSurge cache scope so eval()/maybe_generate()
        # and rollouts share one eSurge engine rather than allocating 2x KV cache.
        rollout_cache_scope_key = state.esurge_cache_scope_key
        max_workers = 1
        queue_capacity = int(self.arguments.max_inflight_tasks)  # type: ignore[attr-defined]
        if self.arguments.queue_maxsize:  # type: ignore[attr-defined]
            queue_capacity = min(queue_capacity, int(self.arguments.queue_maxsize))  # type: ignore[attr-defined]
        sync_interval = int(self.arguments.weight_sync_steps)  # type: ignore[attr-defined]
        max_staleness = int(self.arguments.max_staleness)  # type: ignore[attr-defined]
        steps_per_generation = max(1, int(getattr(self, "steps_per_generation", 1) or 1))
        num_iterations = max(1, int(getattr(self, "num_iterations", 1) or 1))
        concat_chunks = getattr(self, "_concat_raw_prompt_batches", None)
        reuse_span = max(1, steps_per_generation * num_iterations) if callable(concat_chunks) else 1
        is_multihost = int(jax.process_count()) > 1
        if is_multihost:
            logger.warning_once(
                "Async rollout background lookahead disabled on multi-host meshes (process_count=%d) "
                "to prevent concurrent JAX/eSurge collective-order deadlocks.",
                int(jax.process_count()),
            )
        # (future, raw batch, data time, sync time, anchor policy step)
        pending_rollouts: deque[tuple[concurrent.futures.Future, dict[str, object], float, float, int]] = deque()
        active_reuse_batches: deque[tuple[dict[str, jax.Array], dict[str, float | int | str], int, float]] = deque()
        admission_deferred = 0
        stale_dropped = 0
        submitted_futures: set[concurrent.futures.Future] = set()
        rollout_timeout = min(
            float(self.arguments.request_timeout),  # type: ignore[attr-defined]
            float(self.arguments.heartbeat_stale_after_s),  # type: ignore[attr-defined]
        )

        has_buckets = bool(getattr(self, "_has_buckets", False)) and getattr(self, "_bucket_rule", None) is not None

        def fetch_batch(target_step: int | None = None) -> tuple[dict[str, object], float]:
            nonlocal train_iter
            with capture_time() as data_collection_time:
                if has_buckets:
                    bi = self._bucket_rule.select(0 if target_step is None else int(target_step))  # type: ignore[attr-defined]
                    bucket_loader = self._bucket_loaders[bi]  # type: ignore[attr-defined]
                    bucket_collator = self._bucket_collators[bi]  # type: ignore[attr-defined]
                    if steps_per_generation <= 1 or not callable(concat_chunks):
                        raw_batch, self._bucket_iters[bi] = self._get_next_batch(  # type: ignore[attr-defined]
                            self._bucket_iters[bi],  # type: ignore[attr-defined]
                            bucket_loader,
                        )
                        collated = bucket_collator(raw_batch)
                    else:
                        chunks = []
                        for _ in range(steps_per_generation):
                            raw_chunk, self._bucket_iters[bi] = self._get_next_batch(  # type: ignore[attr-defined]
                                self._bucket_iters[bi],  # type: ignore[attr-defined]
                                bucket_loader,
                            )
                            chunks.append(bucket_collator(raw_chunk))
                        collated = concat_chunks(chunks)
                elif steps_per_generation <= 1 or not callable(concat_chunks):
                    raw_batch, train_iter = self._get_next_batch(train_iter, train_dataset)  # type: ignore[attr-defined]
                    collated = data_collator(raw_batch)
                else:
                    chunks = []
                    for _ in range(steps_per_generation):
                        raw_chunk, train_iter = self._get_next_batch(train_iter, train_dataset)  # type: ignore[attr-defined]
                        chunks.append(data_collator(raw_chunk))
                    collated = concat_chunks(chunks)
            return collated, float(data_collection_time())

        def wait_for_inflight_rollouts() -> None:
            """Wait for currently queued/running rollouts to finish without discarding them."""
            for future, _, _, _, _ in tuple(pending_rollouts):
                if not future.done():
                    future.result(timeout=rollout_timeout)

        def submit_rollout(
            executor: concurrent.futures.ThreadPoolExecutor,
            *,
            rollout_state: EasyDeLState,
            rollout_batch: dict[str, object],
            produced_at_step: int,
        ) -> concurrent.futures.Future[AsyncRolloutResult]:
            submitted_futures.difference_update(future for future in tuple(submitted_futures) if future.done())

            def _run_worker() -> AsyncRolloutResult:
                flags = self.__dict__.get("_async_rollout_thread_flags")
                if flags is None:
                    with _THREAD_FLAGS_LOCK:
                        flags = self.__dict__.setdefault("_async_rollout_thread_flags", threading.local())
                flags.in_worker = True
                try:
                    return self._preprocess_async_rollout(
                        state=rollout_state,
                        batch=rollout_batch,
                        produced_at_step=produced_at_step,
                    )
                finally:
                    flags.in_worker = False

            future = executor.submit(_run_worker)
            submitted_futures.add(future)
            return future

        # Deferred: the ``_online_rl`` package imports trainers that import this mixin.
        from ._online_rl.async_rollout import admission_is_feasible

        # Bookkeeping runs in rollout units (policy versions): one consumed rollout advances
        # ``state.step`` by ``steps_per_rollout`` optimizer steps.
        steps_per_rollout = _steps_per_rollout(self)

        def rollout_lag(consume_step: int, anchor_step: int) -> int:
            return max(0, (consume_step - anchor_step) // steps_per_rollout)

        def admits(anchor_step: int, current_step: int) -> bool:
            # Rollout ``r`` (1-based) of the lookahead queue is consumed at step
            # ``current_step + r * steps_per_rollout``; a rollout anchored at ``a`` may be consumed while
            # ``rollout_lag <= max_staleness``, i.e. up to rollout index ``deadline`` below.
            def deadline(anchor: int) -> int:
                return (anchor - current_step + (max_staleness + 1) * steps_per_rollout - 1) // steps_per_rollout

            deadlines = [deadline(queued[4]) for queued in pending_rollouts]
            return admission_is_feasible([*deadlines, deadline(anchor_step)], head_version=1)

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

        idle_event = getattr(self, "_async_rollout_idle_event", None)
        if idle_event is None:
            idle_event = self._async_rollout_idle_event = threading.Event()
            idle_event.set()

        def release_rollout_snapshot() -> None:
            # Runs as a future done-callback, where exceptions are swallowed: always mark the worker idle.
            nonlocal rollout_snapshot
            try:
                if rollout_snapshot is not None:
                    snapshot_to_release = rollout_snapshot
                    rollout_snapshot = None
                    snapshot_to_release.release()
            except Exception:
                logger.warning("Releasing the async rollout policy snapshot failed", exc_info=True)
            finally:
                self._async_grpo_retiring_futures = ()
                self._async_grpo_rollout_idle = True
                idle_event.set()

        with contextlib.ExitStack() as cleanup:
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="async-trainer-rollout",
            )

            def retire_workers():
                self._async_grpo_rollout_idle = False
                idle_event.clear()
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
                    if has_buckets:
                        self._bucket_for_step = self._bucket_rule.select(current_step)  # type: ignore[attr-defined]

                    try:
                        step_metrics.start_step()
                        state = self.on_step_start(state=state, step=current_step)  # type: ignore[attr-defined]

                        with capture_time() as rollout_wait_time:
                            if active_reuse_batches:
                                reused_batch, reused_infos, reused_produced_step, reused_pre_time = (
                                    active_reuse_batches.popleft()
                                )
                                reused_infos = dict(reused_infos)
                                reused_infos["generation_reused"] = 1
                                reused_infos["generation_reuse_remaining"] = len(active_reuse_batches)
                                rollout = AsyncRolloutResult(
                                    batch=reused_batch,
                                    informations=reused_infos,
                                    produced_at_step=reused_produced_step,
                                    preprocessing_time=reused_pre_time,
                                )
                                pending_batch = None
                                pending_data_time = 0.0
                                pending_sync_time = 0.0
                            else:
                                if pending_rollouts:
                                    pending_future, pending_batch, pending_data_time, pending_sync_time, _ = (
                                        pending_rollouts.popleft()
                                    )
                                if pending_future is None:
                                    pending_batch, pending_data_time = fetch_batch(current_step)
                                    # Never generate from a snapshot that is already past its deadline.
                                    snapshot_expired = (
                                        rollout_snapshot is not None
                                        and rollout_lag(current_step, rollout_snapshot.policy_step) > max_staleness
                                    )
                                    snapshot, pending_sync_time = ensure_rollout_snapshot(
                                        state,
                                        current_step,
                                        force=snapshot_expired,
                                    )
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
                                            raise TimeoutError(
                                                f"First async rollout timed out after {first_timeout:.1f}s."
                                            )
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

                        rollout_staleness = rollout_lag(current_step, int(rollout.produced_at_step))
                        if rollout_staleness > max_staleness and pending_batch is not None:
                            # Safety net; deadline admission keeps this rare.
                            # Cancel unstarted futures first before waiting on any
                            # running worker so queued tasks are not drained and wasted.
                            stale_dropped += 1
                            queued_batches = []
                            queued_items = list(pending_rollouts)
                            pending_rollouts.clear()
                            for future, _, _, _, _ in queued_items:
                                future.cancel()
                            for future, queued_batch, data_time, _, _ in queued_items:
                                if not future.cancelled():
                                    future.result(timeout=rollout_timeout)
                                queued_batches.append((queued_batch, data_time))
                            snapshot, resync_seconds = ensure_rollout_snapshot(
                                state,
                                current_step,
                                force=True,
                            )
                            rollout_sync_seconds += resync_seconds
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
                                pending_rollouts.append((future, queued_batch, data_time, 0.0, snapshot.policy_step))

                        # Enqueue any additional minibatches produced for steps_per_generation / num_iterations.
                        if len(rollout.minibatches) > 1:
                            for extra_mb in rollout.minibatches[1:]:
                                active_reuse_batches.append(
                                    (
                                        dict(extra_mb),
                                        dict(rollout.informations),
                                        int(rollout.produced_at_step),
                                        float(rollout.preprocessing_time),
                                    )
                                )

                        compiled_train_ready = bool(getattr(self, "_async_grpo_train_compiled_once", False))
                        step_limit = min(self.max_training_steps, epoch_end_step)  # type: ignore[attr-defined]
                        schedule_next = (
                            not is_multihost
                            and compiled_train_ready
                            and max_staleness > 0
                            and current_step + steps_per_rollout < step_limit
                        )
                        next_data_time = 0.0
                        scheduled_lookahead = 0
                        if schedule_next:
                            # Refresh the snapshot *after* consuming valid rollouts when no
                            # worker is reading the old snapshot and a weight sync is already
                            # due, so lookahead work for upcoming steps runs on fresh weights
                            # instead of being discarded at the next consumption step.
                            if not pending_rollouts and rollout_snapshot is not None:
                                if rollout_lag(current_step, rollout_snapshot.policy_step) >= sync_interval:
                                    rollout_snapshot, refresh_sync_time = ensure_rollout_snapshot(
                                        state,
                                        current_step,
                                        force=True,
                                    )
                                    rollout_sync_seconds += refresh_sync_time

                            # Bound lookahead lag so we never speculatively generate rollouts that
                            # would cross a weight_sync_steps refresh boundary and get discarded.
                            # With weight_sync_steps=1, allow 1-step pipelining (capped by max_staleness).
                            sync_lag_cap = 1 if sync_interval == 1 else sync_interval - 1
                            remaining_steps = step_limit - current_step
                            # Rollouts still to consume after this one: ceil(remaining_steps / k) - 1.
                            remaining = -(-remaining_steps // steps_per_rollout) - 1
                            while len(pending_rollouts) < min(queue_capacity, remaining):
                                next_unscheduled_step = (
                                    current_step
                                    + (1 + len(active_reuse_batches) + len(pending_rollouts) * reuse_span)
                                    * steps_per_rollout
                                )
                                if next_unscheduled_step >= step_limit:
                                    break
                                snapshot, sync_time = ensure_rollout_snapshot(state, current_step)
                                next_lag = rollout_lag(next_unscheduled_step, int(snapshot.policy_step))
                                # Check deadline admission first so staleness-capped lookahead logs
                                # admission_deferred cleanly.
                                if not admits(int(snapshot.policy_step), current_step) or next_lag > max_staleness:
                                    admission_deferred += 1
                                    rollout_sync_seconds += sync_time
                                    break
                                if next_lag > sync_lag_cap:
                                    rollout_sync_seconds += sync_time
                                    break
                                next_batch, data_time = fetch_batch(next_unscheduled_step)
                                next_data_time += data_time
                                future = submit_rollout(
                                    executor,
                                    rollout_state=snapshot.state,
                                    rollout_batch=next_batch,
                                    produced_at_step=snapshot.policy_step,
                                )
                                pending_rollouts.append((future, next_batch, data_time, sync_time, snapshot.policy_step))
                                scheduled_lookahead += 1

                        rollout.informations["async_rollout/rollout_staleness"] = rollout_staleness
                        rollout.informations["rl/policy_lag"] = int(rollout_staleness)
                        rollout.informations["async_rollout/rollout_wait_time"] = rollout_wait_seconds
                        rollout.informations["async_rollout/rollout_preprocessing_time"] = rollout.preprocessing_time
                        rollout.informations["async_rollout/policy_sync_time"] = rollout_sync_seconds
                        rollout.informations["async_rollout/next_rollout_scheduled"] = int(scheduled_lookahead > 0)
                        rollout.informations["async_rollout/admission_deferred"] = admission_deferred
                        rollout.informations["async_rollout/stale_dropped"] = stale_dropped
                        if rollout_snapshot is not None:
                            rollout.informations["async_rollout/policy_snapshot_gib"] = rollout_snapshot.nbytes / 1024**3
                        if getattr(self, "emit_legacy_async_grpo_metrics", False):
                            rollout.informations = with_legacy_async_grpo_aliases(rollout.informations)
                    except (
                        KeyboardInterrupt,
                        EasyDeLTimerError,
                        EasyDeLBreakRequest,
                        EasyDeLPreemptionSignal,
                    ) as exc:
                        if pending_future is not None:
                            pending_future.cancel()
                        return state, exc, train_iter

                    # Refresh startup behavior parity gate on the consumer thread so prefetched lookahead
                    # batches never inherit a stale behavior_parity_gate=1 after step 0 completes.
                    step_batch = rollout.batch
                    refresh_gate = getattr(self, "_refresh_behavior_parity_gate_for_step", None)
                    if callable(refresh_gate):
                        step_batch = refresh_gate(step_batch)
                    elif (
                        getattr(self, "_startup_behavior_parity_checked", False)
                        and isinstance(step_batch, dict)
                        and "behavior_parity_gate" in step_batch
                    ):
                        step_batch = dict(step_batch)
                        existing_gate = step_batch["behavior_parity_gate"]
                        if isinstance(existing_gate, jax.Array):
                            step_batch["behavior_parity_gate"] = jax.numpy.zeros_like(existing_gate)
                        else:
                            step_batch["behavior_parity_gate"] = 0

                    with self.train_tracker.trace_compilation():  # type: ignore[attr-defined]
                        with capture_time() as execution_time:
                            state, metrics, run_exception = self._execute_preprocessed_train_step(
                                state=state,
                                batch=step_batch,
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
                        sync_ref_due = (
                            bool(getattr(self.arguments, "sync_ref_model", False))  # type: ignore[attr-defined]
                            and getattr(self, "ref_state", None) is not None
                            and current_step % max(int(getattr(self.arguments, "ref_model_sync_steps", 1) or 1), 1) == 0  # type: ignore[attr-defined]
                        )
                        if sync_ref_due:
                            # Idle background rollout workers before mutating self.ref_state.
                            wait_for_inflight_rollouts()
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
                        gen_interval = getattr(self.arguments, "generation_interval", None)  # type: ignore[attr-defined]
                        gen_due = bool(gen_interval and gen_interval > 0 and current_step % int(gen_interval) == 0)
                        if gen_due:
                            wait_for_inflight_rollouts()
                        with capture_time() as generation_time:
                            try:
                                self.maybe_generate(state=state, step=current_step, metrics=metrics)  # type: ignore[attr-defined]
                            except Exception as exc:
                                logger.warning("Preview generation hook failed: %s", exc, exc_info=True)
                        bench_interval = getattr(self.arguments, "benchmark_interval", None)  # type: ignore[attr-defined]
                        bench_due = bool(
                            bench_interval and bench_interval > 0 and current_step % int(bench_interval) == 0
                        )
                        if bench_due:
                            wait_for_inflight_rollouts()
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
                                wait_for_inflight_rollouts()
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
    "with_legacy_async_grpo_aliases",
)
