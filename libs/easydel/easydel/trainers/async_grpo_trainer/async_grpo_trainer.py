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
"""Async GRPO trainer backed by local eSurge async execution."""

from __future__ import annotations

import concurrent.futures
import contextlib
import threading
import typing as tp
from collections import deque
from dataclasses import dataclass

import jax

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.errors import EasyDeLBreakRequest, EasyDeLPreemptionSignal, EasyDeLTimerError
from easydel.infra.loss_utils import LossMetrics
from easydel.utils import Registry
from easydel.utils.helpers import capture_time, get_logger

from .._shared import OwnedPolicySnapshot
from .._shared.rollout_lifecycle import after_futures
from ..group_relative_policy_optimization import GRPOTrainer
from ..metrics import BaseProgressBar, MetricsTracker, StepMetrics
from .async_grpo_config import AsyncGRPOConfig

logger = get_logger(__name__)


@dataclass
class _AsyncRolloutResult:
    """Preprocessed rollout batch produced by the AsyncGRPO background worker."""

    batch: dict[str, jax.Array]
    informations: dict[str, float | int | str]
    produced_at_step: int
    preprocessing_time: float
    minibatches: tuple[dict[str, jax.Array], ...] = ()


@Registry.register("trainer", "async_grpo")
class AsyncGRPOTrainer(GRPOTrainer):
    """AsyncGRPO trainer using GRPO updates with eSurge async rollouts.

    The trainer keeps the inherited GRPO loss, reward, and reference-logprob
    implementation, but its config forces local eSurge generation to use async
    scheduler handling and overlap execution. No external inference-server path
    or string model loader is used.
    """

    arguments: AsyncGRPOConfig

    def __init__(
        self,
        arguments: AsyncGRPOConfig,
        model: EasyDeLBaseModule | EasyDeLState | None,
        reward_funcs: object | list[object],
        train_dataset: object | None = None,
        eval_dataset: object | dict[str, object] | None = None,
        processing_class: object | None = None,
        reward_processing_classes: object | list[object] | None = None,
        data_tokenize_fn: tp.Callable[..., object] | None = None,
        tools: list[dict | str | tp.Callable[..., object]] | None = None,
        environment_factory: tp.Callable[[], object] | None = None,
    ) -> None:
        """Initialize the AsyncGRPO public surface on top of GRPOTrainer.

        Args:
            arguments: AsyncGRPO config containing GRPO rollout settings plus
                local scheduling metadata such as inflight, staleness, and
                weight-sync limits.
            model: Initialized EasyDeL policy module or state used by the
                inherited GRPO trainer.
            reward_funcs: Reward callables or reward states used to score
                generated completions.
            train_dataset: Prompt dataset for training rollouts.
            eval_dataset: Optional prompt dataset or named evaluation mapping.
            processing_class: Tokenizer or processor used by generation and
                reward preprocessing.
            reward_processing_classes: Optional processors paired with reward
                functions.
            data_tokenize_fn: Optional tokenizer override accepted by GRPO.
            tools: Optional tool definitions exposed to rollout generation.
            environment_factory: Optional local environment factory for
                tool/environment feedback paths.

        Raises:
            TypeError: If ``arguments`` is not an ``AsyncGRPOConfig``.
        """
        if not isinstance(arguments, AsyncGRPOConfig):
            raise TypeError(f"arguments must be AsyncGRPOConfig, got {type(arguments)}")
        self._async_grpo_tls = threading.local()
        super().__init__(
            arguments=arguments,
            model=model,
            reward_funcs=reward_funcs,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            data_tokenize_fn=data_tokenize_fn,
            tools=tools,
            environment_factory=environment_factory,
        )

    def _get_async_tls(self) -> threading.local:
        tls = getattr(self, "_async_grpo_tls", None)
        if tls is None:
            tls = threading.local()
            self._async_grpo_tls = tls
        return tls

    def _is_forcing_old_logps(self) -> bool:
        return bool(getattr(self._get_async_tls(), "force_old_logps", False))

    def _get_next_batch(self, data_iter, dataloader):
        """Advance the dataloader directly without GRPO's reuse pause.

        ``AsyncGRPOTrainer`` manages multi-step and multi-iteration reuse
        explicitly via ``active_reuse_batches`` and ``fetch_batch()``, so
        inheriting :meth:`GRPOTrainer._get_next_batch`'s reuse pause would
        replay cached prompts across distinct generation cycles or mis-size
        multi-chunk fetches when ``steps_per_generation > 1``.
        """
        return super(GRPOTrainer, self)._get_next_batch(data_iter, dataloader)

    def _should_compute_old_per_token_logps(
        self,
        *,
        is_train: bool,
        needs_dppo: bool,
        needs_esurge_correction: bool,
        needs_off_policy_mask: bool,
    ) -> bool:
        """Always snapshot behavior log-probs for asynchronous rollout batches."""
        if is_train and self._is_forcing_old_logps():
            return True
        return super()._should_compute_old_per_token_logps(
            is_train=is_train,
            needs_dppo=needs_dppo,
            needs_esurge_correction=needs_esurge_correction,
            needs_off_policy_mask=needs_off_policy_mask,
        )

    def _take_rollout_snapshot(
        self,
        state: EasyDeLState,
        *,
        policy_step: int,
        cache_scope_key: str,
    ) -> OwnedPolicySnapshot:
        """Build an inference-only policy snapshot for background rollouts.

        The snapshot owns separate JAX buffers for model graph leaves so the
        training step can keep donating the live training state. Optimizer
        transform and optimizer slots are intentionally dropped: eSurge
        generation and policy log-prob scoring only need the model graph.
        """
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
        force: bool,
        can_release: bool,
        cache_scope_key: str,
    ) -> tuple[OwnedPolicySnapshot, float]:
        """Return an up-to-date rollout snapshot and the seconds spent syncing.

        The snapshot is reused until ``weight_sync_steps`` optimizer steps have
        elapsed since it was taken (or ``force`` demands a fresh policy). On a
        refresh the outgoing snapshot is released *before* its replacement is
        allocated, so only one extra copy of the policy is ever resident;
        reference counting cannot achieve that because the cached eSurge engine
        keeps its own reference to the graph trees it last generated from.

        Releasing is only safe when nothing is reading the outgoing snapshot,
        which the caller asserts through ``can_release``; the engine that
        generated from it must be idle and will pick the replacement's weights
        up through ``get_esurge``'s weight refresh before it executes again.

        Args:
            snapshot: Current snapshot, or ``None`` on the first sync.
            policy_state: Live training state to copy from.
            policy_step: Optimizer step of ``policy_state``.
            force: Refresh regardless of the sync interval.
            can_release: Whether no rollout is reading ``snapshot``. When
                ``False`` the outgoing snapshot is merely dereferenced.
            cache_scope_key: eSurge cache scope for the snapshot's engine.

        Returns:
            ``(snapshot, sync_seconds)`` where ``sync_seconds`` is ``0.0`` when
            the existing snapshot was reused.
        """
        sync_interval = int(self.arguments.weight_sync_steps)
        if snapshot is not None and not force and policy_step - snapshot.policy_step < sync_interval:
            return snapshot, 0.0
        with capture_time() as sync_time:
            if snapshot is not None:
                if can_release:
                    snapshot.release()
                else:
                    # The caller/worker retains ownership until its readers retire.
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
        """Store the GRPO rollout batch and attach AsyncGRPO execution metrics.

        When called from the background async rollout worker thread, avoids
        mutating shared trainer reuse caches (the main loop schedules any
        ``steps_per_generation`` / ``num_iterations`` minibatches cleanly).
        """
        if self._is_forcing_old_logps():
            reuse_span = self._generation_reuse_span()
            metrics = dict(metrics)
            metrics["generation_reused"] = 0
            metrics["generation_reuse_span"] = reuse_span
            metrics["generation_reuse_remaining"] = max(0, reuse_span - 1)
        else:
            model_batch, metrics = super()._store_buffered_grpo_batch(model_batch, metrics)
        return model_batch, {
            **metrics,
            "async_grpo/max_inflight_tasks": int(self.arguments.max_inflight_tasks),
            "async_grpo/max_staleness": int(self.arguments.max_staleness),
            "async_grpo/weight_sync_steps": int(self.arguments.weight_sync_steps),
            "async_grpo/esurge_async_scheduling": int(bool(self.arguments.esurge_async_scheduling)),
            "async_grpo/esurge_overlap_execution": int(bool(self.arguments.esurge_overlap_execution)),
        }

    def _preprocess_async_rollout(
        self,
        *,
        state: EasyDeLState,
        batch: dict[str, object],
        produced_at_step: int,
    ) -> _AsyncRolloutResult:
        """Generate, score, and pack one GRPO rollout for asynchronous reuse.

        This method is executed by the AsyncGRPO worker thread. It intentionally
        reuses :meth:`GRPOTrainer._preprocess_batch_input` so reward routing,
        eSurge generation, reference log-prob computation, tool/environment
        handling, and GRPO batch schema stay identical to the synchronous
        trainer. ``produced_at_step`` is recorded so the consumer can reject
        rollouts that exceed ``max_staleness`` before training on them.
        """
        tls = self._get_async_tls()
        with capture_time() as preprocessing_time_fn:
            tls.force_old_logps = True
            try:
                model_batch, informations = super()._preprocess_batch_input(
                    state=state,
                    batch=batch,
                    is_train=True,
                )
            finally:
                tls.force_old_logps = False
        informations = dict(informations)
        informations["async_grpo/rollout_produced_at_step"] = produced_at_step

        steps_per_generation = max(1, int(getattr(self, "steps_per_generation", 1) or 1))
        num_iterations = max(1, int(getattr(self, "num_iterations", 1) or 1))
        reuse_span = max(1, steps_per_generation * num_iterations)
        if reuse_span > 1:
            prompt_ids = model_batch.get("prompt_ids")
            expected_step_batch = int(getattr(self.arguments, "total_batch_size", 1) or 1)
            if (
                steps_per_generation > 1
                and prompt_ids is not None
                and int(prompt_ids.shape[0]) >= steps_per_generation * expected_step_batch
                and int(prompt_ids.shape[0]) % steps_per_generation == 0
            ):
                chunks = self._slice_model_batch_chunks(model_batch, steps_per_generation)
                schedule = tuple(mb for _ in range(num_iterations) for mb in chunks)
            else:
                schedule = tuple(dict(model_batch) for _ in range(reuse_span))
            first_batch = schedule[0]
        else:
            schedule = (model_batch,)
            first_batch = model_batch

        return _AsyncRolloutResult(
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
        """Run the compiled GRPO update on an already-preprocessed rollout batch.

        AsyncGRPO preprocesses rollout batches in a background worker. Calling
        the inherited ``_execute_train_step`` would regenerate synchronously, so
        this method mirrors the compiled-call part of the base trainer and
        merges the worker-produced information into ``metrics.other_metrics``.
        """
        if self.pruning_module is not None:
            state = state.replace(
                graphstate=self.pruning_module.pre_forward_update(
                    state.graphstate,
                    state.opt_state,
                )
            )
        metrics = LossMetrics()
        try:
            state, metrics = jax.block_until_ready(
                self.sharded_training_step_function(
                    state,
                    batch,
                    *self._train_shared_fn_extra_args,
                    *self._train_shared_fn_static_args,
                )
            )
            if informations:
                merged = dict(informations)
                if metrics.other_metrics is not None:
                    merged.update(metrics.other_metrics)
                metrics = metrics.replace(other_metrics=merged)
            if self.pruning_module is not None:
                state = state.replace(
                    graphstate=self.pruning_module.post_gradient_update(
                        state.graphstate,
                        state.opt_state,
                    )
                )
            return state, metrics, None
        except (
            KeyboardInterrupt,
            EasyDeLTimerError,
            EasyDeLBreakRequest,
            TypeError,
        ) as run_exception:
            return state, metrics, run_exception
        except Exception as run_exception:
            if self._is_memory_oom_exception(run_exception):
                annotated_exception = self._augment_memory_oom_exception(run_exception)
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
        """Run one training epoch with asynchronous rollout lookahead.

        The loop bounds running plus queued rollouts by ``max_inflight_tasks``
        and ``queue_maxsize``. A single worker owns mutable generation buffers.
        At each weight-sync boundary old readers retire and their prompts are
        replayed using the refreshed policy. Consumption also enforces
        ``max_staleness`` against the current learner step.

        Rollouts read a params-only :class:`OwnedPolicySnapshot` rather than the
        live donated state. Refresh releases the outgoing snapshot before
        allocating its replacement. Every exit cancels queued work without an
        unconditional executor join; the last running reader releases the owned
        snapshot when it exits. A subsequent epoch cannot reuse trainer buffers
        until those readers have retired.
        """
        if not getattr(self, "_async_grpo_rollout_idle", True):
            raise RuntimeError("A previous rollout worker is still retiring; wait before starting another epoch.")
        data_collator = self.data_collator
        if data_collator is None:

            def data_collator(x):
                return x

        if self.max_training_steps is None:
            raise RuntimeError("max_training_steps must be set before training")
        if epoch_start_step is None or epoch_end_step is None:
            epoch_start_step, epoch_end_step = self._get_epoch_step_bounds(epoch)
        epoch_total_steps = max(epoch_end_step - epoch_start_step, 1)
        run_exception: Exception | None = None
        pending_future: concurrent.futures.Future[_AsyncRolloutResult] | None = None
        pending_data_time = 0.0
        pending_sync_time = 0.0
        pending_batch = None
        rollout_snapshot: OwnedPolicySnapshot | None = None
        # Reuse the live state's eSurge cache scope so eval()/maybe_generate()
        # and rollouts share one eSurge engine rather than allocating 2x KV cache.
        rollout_cache_scope_key = state.esurge_cache_scope_key
        # GRPO preprocessing mutates trainer buffers and shares one local
        # inference engine. Serialize execution but honor the configured bound
        # on submitted tasks (running plus queued), rather than capping it at 2.
        max_workers = 1
        queue_capacity = int(self.arguments.max_inflight_tasks)
        if self.arguments.queue_maxsize:
            queue_capacity = min(queue_capacity, int(self.arguments.queue_maxsize))
        sync_interval = int(self.arguments.weight_sync_steps)
        max_staleness = int(self.arguments.max_staleness)
        steps_per_generation = max(1, int(getattr(self, "steps_per_generation", 1) or 1))
        num_iterations = max(1, int(getattr(self, "num_iterations", 1) or 1))
        reuse_span = max(1, steps_per_generation * num_iterations)
        is_multihost = int(jax.process_count()) > 1
        if is_multihost:
            logger.warning_once(
                "AsyncGRPO background lookahead disabled on multi-host meshes (process_count=%d) "
                "to prevent concurrent JAX/eSurge collective-order deadlocks.",
                int(jax.process_count()),
            )

        pending_rollouts: deque[tuple[concurrent.futures.Future, dict[str, object], float, float]] = deque()
        active_reuse_batches: deque[tuple[dict[str, jax.Array], dict[str, float | int | str], int, float]] = deque()
        submitted_futures: set[concurrent.futures.Future] = set()
        rollout_timeout = min(float(self.arguments.request_timeout), float(self.arguments.heartbeat_stale_after_s))

        def fetch_batch() -> tuple[dict[str, object], float]:
            nonlocal train_iter
            with capture_time() as data_collection_time:
                if steps_per_generation <= 1:
                    raw_batch, train_iter = self._get_next_batch(train_iter, train_dataset)
                    collated = data_collator(raw_batch)
                else:
                    chunks = []
                    for _ in range(steps_per_generation):
                        raw_chunk, train_iter = self._get_next_batch(train_iter, train_dataset)
                        chunks.append(data_collator(raw_chunk))
                    collated = self._concat_raw_prompt_batches(chunks)
            return collated, float(data_collection_time())

        def wait_for_inflight_rollouts() -> None:
            """Wait for currently queued/running rollouts to finish without discarding them."""
            for future, _, _, _ in tuple(pending_rollouts):
                if not future.done():
                    future.result(timeout=rollout_timeout)

        def submit_rollout(
            executor: concurrent.futures.ThreadPoolExecutor,
            *,
            rollout_state: EasyDeLState,
            rollout_batch: dict[str, object],
            produced_at_step: int,
        ) -> concurrent.futures.Future[_AsyncRolloutResult]:
            # Retain only unfinished readers; completed futures may own large
            # rollout batches and must not accumulate for a whole epoch.
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
            """Refresh the loop's rollout snapshot through :meth:`_sync_rollout_snapshot`."""
            nonlocal rollout_snapshot
            if pending_rollouts and rollout_snapshot is not None:
                # Every queued task owns this policy version. Drain/cancel them
                # before refreshing so no worker reads explicitly freed buffers.
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
            """Free the snapshot's device buffers once no worker can read them."""
            nonlocal rollout_snapshot
            if rollout_snapshot is not None:
                rollout_snapshot.release()
                rollout_snapshot = None
            self._async_grpo_retiring_futures = ()
            self._async_grpo_rollout_idle = True

        with contextlib.ExitStack() as cleanup:
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="async-grpo-rollout",
            )

            def retire_workers():
                # Every exit (including heartbeat timeout and hook exceptions)
                # cancels queued jobs. Never join a stuck provider here. Keep
                # the owner alive and release only after the last reader exits.
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
                    if current_step >= self.max_training_steps or current_step >= epoch_end_step:
                        break

                    try:
                        step_metrics.start_step()
                        state = self.on_step_start(state=state, step=current_step)

                        with capture_time() as rollout_wait_time:
                            if active_reuse_batches:
                                reused_batch, reused_infos, reused_produced_step, reused_pre_time = (
                                    active_reuse_batches.popleft()
                                )
                                reused_infos = dict(reused_infos)
                                reused_infos["generation_reused"] = 1
                                reused_infos["generation_reuse_remaining"] = len(active_reuse_batches)
                                rollout = _AsyncRolloutResult(
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
                    if rollout_staleness > max_staleness and pending_batch is not None:
                        # Cancel unstarted futures first before waiting on any
                        # running worker so queued tasks are not drained and wasted.
                        queued_batches = []
                        queued_items = list(pending_rollouts)
                        pending_rollouts.clear()
                        for future, _, _, _ in queued_items:
                            future.cancel()
                        for future, queued_batch, data_time, _ in queued_items:
                            if not future.cancelled():
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
                    step_limit = min(self.max_training_steps, epoch_end_step)
                    schedule_next = (
                        not is_multihost
                        and compiled_train_ready
                        and max_staleness > 0
                        and current_step + 1 < step_limit
                    )
                    next_data_time = 0.0
                    if schedule_next:
                        # Refresh the snapshot *after* consuming valid rollouts when no
                        # worker is reading the old snapshot, so lookahead work for
                        # upcoming steps runs on fresh weights instead of being discarded.
                        next_unscheduled_step = (
                            current_step + 1 + len(active_reuse_batches) + len(pending_rollouts) * reuse_span
                        )
                        if not pending_rollouts and rollout_snapshot is not None:
                            needs_refresh = (
                                current_step - rollout_snapshot.policy_step >= sync_interval
                                or next_unscheduled_step - rollout_snapshot.policy_step > max_staleness
                            )
                            if needs_refresh:
                                rollout_snapshot, refresh_sync_time = ensure_rollout_snapshot(
                                    state,
                                    current_step,
                                    force=True,
                                )
                                rollout_sync_seconds += refresh_sync_time

                        lookahead_budget = min(max_staleness, max(sync_interval, 1))
                        while len(pending_rollouts) < queue_capacity:
                            next_unscheduled_step = (
                                current_step + 1 + len(active_reuse_batches) + len(pending_rollouts) * reuse_span
                            )
                            if next_unscheduled_step >= step_limit:
                                break
                            snapshot, sync_time = ensure_rollout_snapshot(state, current_step)
                            # Do not submit lookahead work on a snapshot that would be
                            # stale or superseded before it is consumed.
                            if next_unscheduled_step - snapshot.policy_step > lookahead_budget:
                                break
                            next_batch, data_time = fetch_batch()
                            next_data_time += data_time
                            future = submit_rollout(
                                executor,
                                rollout_state=snapshot.state,
                                rollout_batch=next_batch,
                                produced_at_step=snapshot.policy_step,
                            )
                            pending_rollouts.append((future, next_batch, data_time, sync_time))

                    rollout.informations["async_grpo/rollout_staleness"] = rollout_staleness
                    rollout.informations["async_grpo/rollout_wait_time"] = rollout_wait_seconds
                    rollout.informations["async_grpo/rollout_preprocessing_time"] = rollout.preprocessing_time
                    rollout.informations["async_grpo/policy_sync_time"] = rollout_sync_seconds
                    rollout.informations["async_grpo/next_rollout_scheduled"] = int(schedule_next)
                    if rollout_snapshot is not None:
                        rollout.informations["async_grpo/policy_snapshot_gib"] = rollout_snapshot.nbytes / 1024**3

                    with self.train_tracker.trace_compilation():
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

                    self._maybe_start_profiler(current_step)
                    try:
                        mean_loss, mean_accuracy = metrics_tracker.update(
                            loss=metrics.loss,
                            accuracy=metrics.accuracy,
                            step=current_step,
                        )
                        metrics = self.apply_training_hooks(metrics=metrics)
                        train_metrics = step_metrics.calculate(
                            metrics=metrics,
                            current_step=current_step,
                            learning_rate=(
                                self.scheduler(current_step)
                                if self.scheduler is not None
                                else self.arguments.learning_rate
                            ),
                            epoch=epoch,
                            epoch_progress=min(max((current_step - epoch_start_step) / epoch_total_steps, 0.0), 1.0),
                            flops_per_token=self._backward_flops_per_token,
                            extra_flops_per_token=self._extra_backward_flops_per_token,
                            batch_size=self.training_batch_size,
                            seq_length=self.arguments.max_length,
                            mean_loss=mean_loss,
                            mean_accuracy=mean_accuracy,
                            mode="train",
                        )
                        train_metrics["performance/data_collection_time"] = float(pending_data_time)
                        train_metrics["performance/async_next_data_collection_time"] = float(next_data_time)
                        sync_ref_due = (
                            bool(getattr(self.arguments, "sync_ref_model", False))
                            and getattr(self, "ref_state", None) is not None
                            and current_step % max(int(getattr(self.arguments, "ref_model_sync_steps", 1) or 1), 1) == 0
                        )
                        if sync_ref_due:
                            # Idle background rollout workers before mutating self.ref_state.
                            wait_for_inflight_rollouts()
                        state, metrics = self.on_step_end(
                            state=state,
                            metrics=metrics,
                            step=current_step,
                        )
                        with capture_time() as logging_time:
                            self.log_metrics(
                                metrics=train_metrics,
                                pbar=pbar,
                                step=current_step,
                                mode="train",
                            )
                        if self._should_save_tpu_preemption_checkpoint(current_step):
                            self._save_tpu_preemption_checkpoint(state=state, step=current_step)
                            return state, EasyDeLPreemptionSignal("TPU preemption checkpoint saved"), train_iter
                        with capture_time() as weight_distribution_time:
                            self.log_weight_distribution(state=state, step=current_step)
                        with capture_time() as watchers_time:
                            self.log_watchers(state=state, step=current_step)
                        gen_interval = getattr(self.arguments, "generation_interval", None)
                        gen_due = bool(gen_interval and gen_interval > 0 and current_step % int(gen_interval) == 0)
                        if gen_due:
                            wait_for_inflight_rollouts()
                        with capture_time() as generation_time:
                            try:
                                self.maybe_generate(state=state, step=current_step, metrics=metrics)
                            except Exception as exc:
                                logger.warning("Preview generation hook failed: %s", exc, exc_info=True)
                        bench_interval = getattr(self.arguments, "benchmark_interval", None)
                        bench_due = bool(
                            bench_interval and bench_interval > 0 and current_step % int(bench_interval) == 0
                        )
                        if bench_due:
                            wait_for_inflight_rollouts()
                        with capture_time() as benchmark_time:
                            try:
                                self.maybe_benchmark(state=state, step=current_step)
                            except Exception as exc:
                                logger.warning("Benchmark hook failed: %s", exc, exc_info=True)
                        with capture_time() as checkpoint_time:
                            self._save_checkpoint_for_step(
                                state=state,
                                step=current_step,
                                merge_lora_before_save=self.arguments.merge_lora_before_save,
                            )
                        with capture_time() as evaluation_time:
                            if self._should_run_evaluation(current_step):
                                wait_for_inflight_rollouts()
                                for _ in self.eval(model_state=state):
                                    ...
                        self.log_metrics(
                            metrics={
                                "performance/logging_time": float(logging_time()),
                                "performance/weight_distribution_time": float(weight_distribution_time()),
                                "performance/watchers_time": float(watchers_time()),
                                "performance/generation_time": float(generation_time()),
                                "performance/benchmark_time": float(benchmark_time()),
                                "performance/checkpoint_time": float(checkpoint_time()),
                                "performance/evaluation_time": float(evaluation_time()),
                                "performance/iteration_time": float(iteration_time()),
                            },
                            pbar=pbar,
                            step=current_step,
                            mode="train",
                            update_progress=False,
                        )
                        if self._profiler_should_block_until_ready():
                            state, metrics = jax.block_until_ready((state, metrics))
                    except (KeyboardInterrupt, EasyDeLTimerError, EasyDeLBreakRequest, TypeError) as exc:
                        if pending_future is not None:
                            pending_future.cancel()
                        return state, exc, train_iter

                    pending_data_time = next_data_time

        return state, run_exception, train_iter
