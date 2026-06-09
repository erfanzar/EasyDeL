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
import typing as tp
from dataclasses import dataclass

import jax

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.errors import EasyDeLBreakRequest, EasyDeLPreemptionSignal, EasyDeLTimerError
from easydel.infra.loss_utils import LossMetrics
from easydel.utils import Registry
from easydel.utils.helpers import capture_time

from ..group_relative_policy_optimization import GRPOTrainer
from ..metrics import BaseProgressBar, MetricsTracker, StepMetrics
from .async_grpo_config import AsyncGRPOConfig


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

    @staticmethod
    def _copy_rollout_leaf(value: object) -> object:
        """Return an independent copy for JAX array leaves used by the actor."""
        if isinstance(value, jax.Array):
            return value.copy()
        return value

    def _copy_rollout_policy_state(self, state: EasyDeLState, *, cache_scope_key: str) -> EasyDeLState:
        """Build an inference-only policy snapshot for background rollouts.

        The snapshot owns separate JAX buffers for model graph leaves so the
        training step can keep donating the live training state. Optimizer
        transform and optimizer slots are intentionally dropped: eSurge
        generation and policy log-prob scoring only need the model graph.
        """
        rollout_state = state.replace(
            step=self._copy_rollout_leaf(state.step),
            graphstate=jax.tree_util.tree_map(self._copy_rollout_leaf, state.graphstate),
            graphother=jax.tree_util.tree_map(self._copy_rollout_leaf, state.graphother),
            tx=None,
            opt_state=None,
            esurge_cache_scope_key=cache_scope_key,
        )
        return jax.block_until_ready(rollout_state)

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
        rollout_state_snapshot: EasyDeLState | None = None
        rollout_state_step = -1
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

        def ensure_rollout_state(
            policy_state: EasyDeLState,
            policy_step: int,
            *,
            force: bool = False,
        ) -> tuple[EasyDeLState, int, float]:
            nonlocal rollout_state_snapshot, rollout_state_step
            sync_interval = int(self.arguments.weight_sync_steps)
            if rollout_state_snapshot is not None and not force and policy_step - rollout_state_step < sync_interval:
                return rollout_state_snapshot, rollout_state_step, 0.0
            old_snapshot = rollout_state_snapshot
            rollout_state_snapshot = None
            del old_snapshot
            with capture_time() as sync_time:
                rollout_state_snapshot = self._copy_rollout_policy_state(
                    policy_state,
                    cache_scope_key=rollout_cache_scope_key,
                )
            rollout_state_step = policy_step
            return rollout_state_snapshot, rollout_state_step, float(sync_time())

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="async-grpo-rollout",
        ) as executor:
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
                                rollout_state, rollout_policy_step, pending_sync_time = ensure_rollout_state(
                                    state,
                                    current_step,
                                )
                                pending_future = submit_rollout(
                                    executor,
                                    rollout_state=rollout_state,
                                    rollout_batch=pending_batch,
                                    produced_at_step=rollout_policy_step,
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
                        rollout_state, rollout_policy_step, rollout_sync_seconds = ensure_rollout_state(
                            state,
                            current_step,
                            force=True,
                        )
                        rollout = self._preprocess_async_rollout(
                            state=rollout_state,
                            batch=tp.cast(dict[str, object], pending_batch),
                            produced_at_step=rollout_policy_step,
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
                        rollout_state, rollout_policy_step, pending_sync_time = ensure_rollout_state(
                            state,
                            current_step,
                        )
                        pending_future = submit_rollout(
                            executor,
                            rollout_state=rollout_state,
                            rollout_batch=next_batch,
                            produced_at_step=rollout_policy_step,
                        )
                        pending_batch = next_batch

                    rollout.informations["async_grpo/rollout_staleness"] = rollout_staleness
                    rollout.informations["async_grpo/rollout_wait_time"] = rollout_wait_seconds
                    rollout.informations["async_grpo/rollout_preprocessing_time"] = rollout.preprocessing_time
                    rollout.informations["async_grpo/policy_sync_time"] = rollout_sync_seconds
                    rollout.informations["async_grpo/next_rollout_scheduled"] = int(schedule_next)

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
