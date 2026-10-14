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
import typing as tp
from dataclasses import dataclass

import jax

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.errors import EasyDeLBreakRequest, EasyDeLPreemptionSignal, EasyDeLTimerError
from easydel.infra.loss_utils import LossMetrics
from easydel.utils import Registry
from easydel.utils.helpers import capture_time, get_logger

from .._shared import OwnedPolicySnapshot
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
                    # Unreachable through `_train_epoch`, which consumes the
                    # in-flight rollout before every sync. Dereference instead so
                    # a future edit cannot free buffers a worker is still reading.
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

        The underlying buffering behavior is inherited from :class:`GRPOTrainer`.
        This override records both AsyncGRPO scheduling limits and the concrete
        eSurge async/overlap flags that make rollout generation asynchronous in
        EasyDeL.
        """
        if bool(getattr(self, "_async_grpo_force_old_logps", False)):
            metrics = dict(metrics)
            metrics["generation_reused"] = 0
            metrics["generation_reuse_span"] = 1
            metrics["generation_reuse_remaining"] = 0
            self._buffered_grpo_batch = None
            self._buffered_grpo_remaining = 0
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

    def _generation_reuse_span(self) -> int:
        """Force sampling-policy log-probs for async rollouts without reuse.

        GRPO computes ``old_per_token_logps`` when a generated batch may be
        reused. AsyncGRPO also needs those log-probs because a rollout can be
        consumed after one or more policy updates. The actual batch reuse cache
        stays disabled in :meth:`_store_buffered_grpo_batch` while this flag is
        active.
        """
        span = super()._generation_reuse_span()
        if bool(getattr(self, "_async_grpo_force_old_logps", False)):
            return max(span, 2)
        return span

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
        with capture_time() as preprocessing_time_fn:
            self._async_grpo_force_old_logps = True
            try:
                model_batch, informations = super()._preprocess_batch_input(
                    state=state,
                    batch=batch,
                    is_train=True,
                )
            finally:
                self._async_grpo_force_old_logps = False
        informations = dict(informations)
        informations["async_grpo/rollout_produced_at_step"] = produced_at_step
        return _AsyncRolloutResult(
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

        The loop keeps one pending rollout future. At step ``N`` it trains on
        the current preprocessed rollout while a worker generates/scores the
        rollout for step ``N + 1`` using the step-``N`` policy state. The next
        step consumes that future if its policy staleness is within
        ``max_staleness``; otherwise it regenerates synchronously with the
        current state.

        Rollouts read a params-only :class:`OwnedPolicySnapshot` rather than the
        live state, which the compiled step donates. Exactly one snapshot is
        resident at a time: :meth:`_sync_rollout_snapshot` frees the outgoing
        one before allocating its replacement, and the epoch's cleanup releases
        the last one once the rollout workers have joined.
        """
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
        rollout_cache_scope_key = f"{state.esurge_cache_scope_key}-async-grpo-rollout"
        max_workers = max(1, min(int(self.arguments.max_inflight_tasks), 2))

        def fetch_batch() -> tuple[dict[str, object], float]:
            nonlocal train_iter
            with capture_time() as data_collection_time:
                raw_batch, train_iter = self._get_next_batch(train_iter, train_dataset)
                collated = data_collator(raw_batch)
            return collated, float(data_collection_time())

        def submit_rollout(
            executor: concurrent.futures.ThreadPoolExecutor,
            *,
            rollout_state: EasyDeLState,
            rollout_batch: dict[str, object],
            produced_at_step: int,
        ) -> concurrent.futures.Future[_AsyncRolloutResult]:
            return executor.submit(
                self._preprocess_async_rollout,
                state=rollout_state,
                batch=rollout_batch,
                produced_at_step=produced_at_step,
            )

        def ensure_rollout_snapshot(
            policy_state: EasyDeLState,
            policy_step: int,
            *,
            force: bool = False,
        ) -> tuple[OwnedPolicySnapshot, float]:
            """Refresh the loop's rollout snapshot through :meth:`_sync_rollout_snapshot`."""
            nonlocal rollout_snapshot
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

        with contextlib.ExitStack() as cleanup:
            # Registered before the executor so it unwinds *after* the executor
            # has joined its rollout workers: the snapshot is then unreachable
            # and its buffers are freed on every exit path (break, return, raise)
            # rather than being held until the next epoch overwrites it.
            cleanup.callback(release_rollout_snapshot)
            executor = cleanup.enter_context(
                concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix="async-grpo-rollout",
                )
            )
            while True:
                with capture_time() as iteration_time:
                    current_step = int(jax.device_get(state.step))
                    if current_step >= self.max_training_steps or current_step >= epoch_end_step:
                        break

                    try:
                        step_metrics.start_step()
                        state = self.on_step_start(state=state, step=current_step)

                        with capture_time() as rollout_wait_time:
                            if pending_future is None:
                                pending_batch, pending_data_time = fetch_batch()
                                snapshot, pending_sync_time = ensure_rollout_snapshot(state, current_step)
                                pending_future = submit_rollout(
                                    executor,
                                    rollout_state=snapshot.state,
                                    rollout_batch=pending_batch,
                                    produced_at_step=snapshot.policy_step,
                                )
                            rollout = pending_future.result(timeout=float(self.arguments.request_timeout))
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
                    if rollout_staleness > int(self.arguments.max_staleness):
                        snapshot, rollout_sync_seconds = ensure_rollout_snapshot(
                            state,
                            current_step,
                            force=True,
                        )
                        rollout = self._preprocess_async_rollout(
                            state=snapshot.state,
                            batch=tp.cast(dict[str, object], pending_batch),
                            produced_at_step=snapshot.policy_step,
                        )
                        rollout_staleness = 0
                        rollout_wait_seconds = 0.0

                    compiled_train_ready = bool(getattr(self, "_async_grpo_train_compiled_once", False))
                    schedule_next = (
                        compiled_train_ready
                        and current_step + 1 < self.max_training_steps
                        and current_step + 1 < epoch_end_step
                    )
                    next_data_time = 0.0
                    if schedule_next:
                        next_batch, next_data_time = fetch_batch()
                        snapshot, pending_sync_time = ensure_rollout_snapshot(state, current_step)
                        pending_future = submit_rollout(
                            executor,
                            rollout_state=snapshot.state,
                            rollout_batch=next_batch,
                            produced_at_step=snapshot.policy_step,
                        )
                        pending_batch = next_batch

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
                        with capture_time() as generation_time:
                            try:
                                self.maybe_generate(state=state, step=current_step, metrics=metrics)
                            except Exception:
                                ...
                        with capture_time() as benchmark_time:
                            try:
                                self.maybe_benchmark(state=state, step=current_step)
                            except Exception:
                                ...
                        with capture_time() as checkpoint_time:
                            self._save_checkpoint_for_step(
                                state=state,
                                step=current_step,
                                merge_lora_before_save=self.arguments.merge_lora_before_save,
                            )
                        with capture_time() as evaluation_time:
                            if self._should_run_evaluation(current_step):
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
