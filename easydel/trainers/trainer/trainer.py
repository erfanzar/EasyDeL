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

"""Main Trainer implementation for EasyDeL.

This module contains the core Trainer class that orchestrates the complete
training pipeline for neural network models using JAX/spectrax. The trainer
provides a high-level interface for:

- Distributed training across multiple devices and hosts
- Automatic mixed precision training
- Gradient accumulation for large batch sizes
- Comprehensive checkpointing and recovery
- Integration with various data loaders (Grain, TensorFlow datasets)
- Metrics tracking and logging (WandB, TensorBoard)
- Memory-efficient training with sharding strategies

The Trainer class is designed to be flexible and extensible, supporting
various model architectures including language models, vision models,
and multimodal architectures.
"""

import collections.abc
import concurrent.futures
import typing as tp

import jax

from easydel.infra.base_state import EasyDeLState
from easydel.infra.errors import EasyDeLBreakRequest, EasyDeLPreemptionSignal, EasyDeLTimerError
from easydel.infra.loss_utils import LossMetrics
from easydel.infra.sharding import replicated_named_sharding
from easydel.utils import Registry
from easydel.utils.helpers import capture_time, get_logger  # pyright: ignore[reportPrivateLocalImportUsage]

from ..base_trainer import BaseTrainer, TrainerConfigureFunctionOutput  # pyright: ignore[reportPrivateLocalImportUsage]
from ..metrics import BaseProgressBar, MetricsTracker, StepMetrics
from ..trainer_protocol import TrainerOutput
from ..training_utils import compile_trainer_step, resolve_straight_through_emulator
from ._fn import evaluation_step, training_step

logger = get_logger(__name__)


class _TrainBatchPrefetcher:
    """One-batch lookahead for host-side dataloader fetch and collation.

    Hides the latency of fetching and collating the *next* training
    batch by running it on a background thread while the device is busy
    with the current step. Maintains a single pending future at a time
    so memory usage stays bounded; callers explicitly opt in to
    scheduling the next batch via ``schedule_next``.

    Attributes:
        _trainer: Owning :class:`Trainer` (used to call
            ``_get_next_batch``).
        _data_iter: Active dataloader iterator; replaced after each
            successful fetch so the iterator can advance.
        _dataloader: Source dataloader (Grain or TFDS), kept for
            iterator re-creation when an iterator is exhausted.
        _data_collator: Callable applied to the raw batch before
            returning it to the trainer.
        _executor: Single-worker thread pool that runs ``_load``.
        _future: Pending fetch future or ``None`` between submissions.
        _closed: Sticky flag that suppresses further submissions after
            :meth:`close` is called.
    """

    def __init__(
        self,
        trainer: "Trainer",
        data_iter: collections.abc.Iterator[tp.Any],
        dataloader: collections.abc.Iterable[tp.Any],
        data_collator: tp.Callable[[tp.Any], tp.Any],
    ) -> None:
        """Initialize the prefetcher and submit the first fetch.

        Args:
            trainer: Owning :class:`Trainer` instance.
            data_iter: Active dataloader iterator to read the next batch
                from.
            dataloader: Source dataloader; used to re-create the
                iterator when needed.
            data_collator: Callable applied to each raw batch to produce
                the collated batch returned by :meth:`next`.
        """
        self._trainer = trainer
        self._data_iter = data_iter
        self._dataloader = dataloader
        self._data_collator = data_collator
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="easydel-train-batch-prefetch",
        )
        self._future: concurrent.futures.Future[tuple[tp.Any, collections.abc.Iterator[tp.Any], float]] | None = None
        self._closed = False
        self._submit()

    def _load(
        self,
        data_iter: collections.abc.Iterator[tp.Any],
    ) -> tuple[tp.Any, collections.abc.Iterator[tp.Any], float]:
        """Fetch and collate one batch on the background thread.

        Args:
            data_iter: Iterator to pull the next batch from.

        Returns:
            Tuple ``(batch, data_iter, elapsed_seconds)`` containing the
            collated batch, the (possibly updated) iterator, and the
            wall time of the fetch + collate.
        """
        with capture_time() as data_collection_time:
            batch, data_iter = self._trainer._get_next_batch(data_iter, self._dataloader)
            batch = self._data_collator(batch)
        return batch, data_iter, float(data_collection_time())

    def _submit(self) -> None:
        """Submit a new fetch future if the prefetcher is still open."""
        if self._closed:
            return
        self._future = self._executor.submit(self._load, self._data_iter)

    def next(self, *, schedule_next: bool) -> tuple[tp.Any, float, float]:
        """Return the prefetched batch, optionally scheduling the next fetch.

        Args:
            schedule_next: When ``True``, submit a new fetch future
                immediately after retrieving the current one so the
                next call overlaps with device work.

        Returns:
            Tuple ``(batch, wait_seconds, data_time)`` where
            ``wait_seconds`` is the wall time spent blocking on the
            future result and ``data_time`` is the producer-side time
            spent inside :meth:`_load`.

        Raises:
            RuntimeError: If the prefetcher has been closed before a
                batch was made available.
        """
        if self._future is None:
            self._submit()
        if self._future is None:
            raise RuntimeError("Batch prefetcher was closed before a batch was available.")

        with capture_time() as wait_time:
            batch, self._data_iter, data_time = self._future.result()
        wait_seconds = float(wait_time())
        self._future = None
        if schedule_next:
            self._submit()
        return batch, wait_seconds, data_time

    @property
    def data_iter(self) -> collections.abc.Iterator[tp.Any]:
        """The current dataloader iterator (advanced after each fetch)."""
        return self._data_iter

    def close(self) -> None:
        """Cancel any pending fetch and shut down the background executor.

        Idempotent; subsequent calls are no-ops. The captured iterator
        remains accessible through :attr:`data_iter` so callers can
        resume reading from it after closing the prefetcher.
        """
        self._closed = True
        if self._future is not None:
            self._future.cancel()
            self._future = None
        self._executor.shutdown(wait=False, cancel_futures=True)


@Registry.register("trainer", "base")
class Trainer(BaseTrainer):
    """
    Main trainer implementation for EasyDeL models.

    This class provides a complete training and evaluation pipeline for JAX-based
    models with support for distributed training, gradient accumulation, mixed
    precision, and various optimization strategies.

    The trainer handles:
    - Distributed training across multiple devices and hosts
    - Automatic checkpointing and resumption
    - Gradient accumulation for large effective batch sizes
    - Learning rate scheduling and optimization
    - Comprehensive metrics tracking and logging
    - Memory-efficient data loading with Grain or TensorFlow datasets

    Key Features:
    - JIT compilation of training and evaluation steps
    - Automatic mixed precision training
    - Support for model and data parallelism
    - Integration with WandB and TensorBoard
    - Flexible data collation and preprocessing

    Example:
        >>> trainer = Trainer(
        ...     arguments=training_args,
        ...     model=model,
        ...     dataset_train=train_dataset,
        ...     dataset_eval=eval_dataset
        ... )
        >>> output = trainer.train()
    """

    def create_grain_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """Return the default Grain collator (identity) for the base trainer.

        The base :class:`Trainer` assumes Grain's per-shard pipeline has
        already produced ready-to-consume batched arrays (typically via
        upstream :class:`grain.transforms.Batch` or a custom
        per-trainer collator). This default therefore returns an
        identity callable: any padding / truncation the model needs
        must be done by the dataset transforms or by a subclass that
        overrides this hook.

        Subclasses with non-trivial padding (e.g. DPO, ORPO, GRPO,
        reward) override this method to return a dedicated
        ``Grain``-compatible collator.

        Args:
            max_sequence_length: Accepted for API symmetry with
                :meth:`create_tfds_collect_function`; ignored by the
                identity collator.
            truncation_mode: Accepted for API symmetry; ignored by the
                identity collator.

        Returns:
            A callable ``batch -> batch`` that forwards Grain output
            verbatim.
        """

        def collate_fn(batch):
            """Identity collator used by Grain when no truncation is needed.

            Args:
                batch: Iterable of per-example dicts produced by Grain.

            Returns:
                The original batch unchanged.
            """
            return batch

        return collate_fn

    def create_tfds_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """Return a TFDS collator that stacks per-example dicts into batched arrays.

        TFDS yields lists of per-example dicts; the returned closure
        builds a batched ``dict[str, jax.Array]`` by stacking
        same-named fields. Two type-specific behaviours are baked in:

        * Fields whose values fail ``jax.numpy.array(...)`` conversion
          (typically Python objects) are silently dropped, except for
          the special ``"tools"`` key which is forwarded as a Python
          list so chat-template tool schemas survive batching.
        * For causal-LM models (``model.lossfn_type == "ForCausalLM"``)
          per-example tensors are first sliced to
          ``max_sequence_length`` along the last axis -- ``keep_end``
          retains the trailing window (default; preserves the
          completion suffix) and ``keep_start`` retains the leading
          window. Non-causal models are stacked unchanged.

        Args:
            max_sequence_length: Maximum number of trailing/leading
                tokens to retain per causal-LM example.
            truncation_mode: ``"keep_end"`` (default) or
                ``"keep_start"`` -- selects which window survives the
                slice for causal-LM examples.

        Returns:
            A callable ``list[dict[str, Any]] -> dict[str, jax.Array]``
            ready to pass to a TFDS pipeline as the collate step.
        """

        def collate_fn(batch):
            """Stack a list of TFDS-style example dicts into batched arrays.

            For causal-LM models the per-example tensors are first
            truncated to ``max_sequence_length`` according to
            ``truncation_mode``.

            Args:
                batch: List of per-example dicts.

            Returns:
                dict: Batched arrays keyed by feature name. ``tools`` is
                kept as a python list; non-array fields that fail
                ``jax.numpy.array(...)`` conversion are silently dropped.
            """
            results = {}
            for key in batch[0].keys():
                if key == "tools":
                    results[key] = [example.get(key) for example in batch]
                    continue
                data_sample = batch[0][key]
                try:
                    data_sample = jax.numpy.array(data_sample)
                except TypeError:
                    continue
                if self.model.lossfn_type == "ForCausalLM":
                    if truncation_mode == "keep_end":
                        corrected_sequence = [jax.numpy.array(f[key])[..., -max_sequence_length:] for f in batch]
                    else:
                        corrected_sequence = [jax.numpy.array(f[key])[..., :max_sequence_length] for f in batch]
                    results[key] = jax.numpy.stack(corrected_sequence)
                else:
                    corrected_sequence = [jax.numpy.array(f[key]) for f in batch]
                    results[key] = jax.numpy.stack(corrected_sequence)
            return results

        return collate_fn

    def create_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"],
    ) -> tp.Callable:
        """
        Creates a function to collect and process batches of data for training or evaluation.

        This function handles padding or truncating sequences to the specified `max_sequence_length`
        based on the chosen `truncation_mode`.

        Args:
            max_sequence_length (int): The maximum allowed sequence length.
            truncation_mode (typing.tp.Literal["keep_end", "keep_start"], optional):
                The truncation mode. Defaults to "keep_end".

        Returns:
            tp.Callable: A function that takes a batch of data and returns a processed batch.
        """
        return (
            self.create_grain_collect_function(
                max_sequence_length=max_sequence_length,
                truncation_mode=truncation_mode,
            )
            if self.arguments.use_grain
            else self.create_tfds_collect_function(
                max_sequence_length=max_sequence_length,
                truncation_mode=truncation_mode,
            )
        )

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """
        Configure and JIT-compile training and evaluation step functions.

        This method is crucial for performance as it:
        1. Sets up proper sharding specifications for distributed training
        2. JIT-compiles the step functions with appropriate static arguments
        3. Configures input/output sharding for efficient data movement
        4. Sets up the checkpoint manager for model persistence

        The compilation process traces through the computation graph once
        and generates optimized XLA code for subsequent executions.

        Returns:
            TrainerConfigureFunctionOutput: Contains:
                - sharded_training_step_function: JIT-compiled training function
                  with gradient computation and parameter updates
                - sharded_evaluation_step_function: JIT-compiled evaluation function
                  for forward passes only
                - mesh: Device mesh for distributed computation
                - checkpoint_manager: AsyncCheckpointManager for saving/loading

        Note:
            - Static arguments are traced at compile time and cannot change
            - The donate_argnums=(0,) for training allows in-place updates
            - Empty sharding specs indicate replication across devices
        """
        empty_sharding = replicated_named_sharding(self.model.mesh)
        straight_through_emulator = resolve_straight_through_emulator(
            quantization_mode=self.arguments.quantization_mode,
            quantization_group_size=self.arguments.quantization_group_size,
            quantization_bits=self.arguments.quantization_bits,
            tensor_straight_through=self.arguments.tensor_straight_through,
            straight_through_emulator=self.arguments.straight_through_emulator,
        )
        self._train_shared_fn_static_args = (
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            straight_through_emulator,
        )
        sharded_training_step_function = compile_trainer_step(
            training_step,
            static_argnums=(2, 3, 4, 5, 6),
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            schedule=self.arguments.mpmd_scheduler,
            mesh=self.mesh,
        )

        self._eval_shared_fn_static_args = (
            self.arguments.loss_config,
            self.arguments.step_partition_spec,
        )
        sharded_evaluation_step_function = compile_trainer_step(
            evaluation_step,
            static_argnums=(2, 3),
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=(empty_sharding),
            schedule=self.arguments.mpmd_scheduler,
            mesh=self.mesh,
        )

        mesh = self.model.mesh
        self.arguments.ensure_checkpoint_path()
        checkpoint_manager = self.arguments.get_streaming_checkpointer()

        return TrainerConfigureFunctionOutput(
            sharded_training_step_function=sharded_training_step_function,
            sharded_evaluation_step_function=sharded_evaluation_step_function,
            mesh=mesh,
            checkpoint_manager=checkpoint_manager,
        )

    def _get_epoch_step_bounds(self, epoch: int) -> tuple[int, int]:
        """Return the global step range assigned to an epoch."""
        if self.max_training_steps is None:
            raise RuntimeError("max_training_steps must be set before training")
        total_epochs = max(int(self.arguments.num_train_epochs), 1)
        return (
            (epoch * self.max_training_steps) // total_epochs,
            ((epoch + 1) * self.max_training_steps) // total_epochs,
        )

    def _get_resume_epoch(self, step: int) -> int:
        """Map a global training step back to its epoch index."""
        if self.max_training_steps is None:
            raise RuntimeError("max_training_steps must be set before training")
        total_epochs = max(int(self.arguments.num_train_epochs), 1)
        if step >= self.max_training_steps:
            return total_epochs
        for epoch in range(total_epochs):
            _, epoch_end_step = self._get_epoch_step_bounds(epoch)
            if step < epoch_end_step:
                return epoch
        return total_epochs

    def _run_training_loop(
        self,
        state: EasyDeLState,
        metrics_tracker: MetricsTracker,
        step_metrics: StepMetrics,
    ):
        """Execute the main training loop across all epochs.

        Orchestrates the entire training process, managing:

        - Epoch iteration with proper resumption handling
        - Progress tracking and reporting
        - Batch processing and gradient updates
        - Checkpoint saving at specified intervals
        - Early stopping on interruption or time limits

        Resume is keyed off the saved global training step. The trainer
        fast-forwards the training iterator by the already-consumed batch
        count and continues from the correct point inside the current epoch.

        Args:
            state: Initial model state with parameters and optimizer state.
            metrics_tracker: Accumulates metrics across training steps.
            step_metrics: Calculates per-step metrics like throughput.

        Returns:
            tuple: ``(TrainerOutput, exception)`` where ``TrainerOutput``
            contains final state and checkpoint info and ``exception``
            is any error that caused training to stop (``None`` on
            normal completion).

        Note:
            - Progress bar is disabled on non-primary processes by default.
            - Training can be interrupted with Ctrl+C and will save state.
            - Automatic resumption updates the progress bar to show continuation.
        """
        disabled = False
        if jax.process_index() != 0 and not self.arguments.log_all_workers:
            disabled = True
        self._runtime_trace(
            "training_loop.begin",
            max_training_steps=self.max_training_steps,
            num_train_epochs=self.arguments.num_train_epochs,
        )
        pbar = self.create_progress_bar(
            total=self.max_training_steps,
            disabled=disabled,
            desc="training process",
        )

        initial_step = int(jax.device_get(state.step))
        start_epoch = 0
        train_iter = iter(self.dataloader_train)
        self._runtime_trace("training_loop.iterator.created", initial_step=initial_step)

        if initial_step > 0:
            if self.max_training_steps is None:
                raise RuntimeError("max_training_steps must be set before training")
            pbar.update(min(initial_step, self.max_training_steps))
            start_epoch = self._get_resume_epoch(initial_step)
            if initial_step < self.max_training_steps:
                logger.info(
                    f"Resuming training from step {initial_step}; fast-forwarding dataloader by {initial_step} batches."
                )
                train_iter = self._fast_forward_batches(train_iter, self.dataloader_train, initial_step)
            else:
                logger.info(
                    f"Resumed state is already at step {initial_step}, which is at or beyond "
                    f"max_training_steps={self.max_training_steps}."
                )
        try:
            run_exception = None
            self._runtime_trace("training_loop.mesh.enter")
            with self.mesh:
                self._runtime_trace("training_loop.mesh.entered")
                for epoch in range(start_epoch, self.arguments.num_train_epochs):
                    epoch_start_step, epoch_end_step = self._get_epoch_step_bounds(epoch)
                    if epoch_start_step >= epoch_end_step:
                        continue
                    self._runtime_trace(
                        "training_loop.epoch.begin",
                        epoch=epoch,
                        epoch_start_step=epoch_start_step,
                        epoch_end_step=epoch_end_step,
                    )
                    state, run_exception, train_iter = self._train_epoch(
                        state=state,
                        train_dataset=self.dataloader_train,
                        train_iter=train_iter,
                        metrics_tracker=metrics_tracker,
                        step_metrics=step_metrics,
                        pbar=pbar,
                        epoch=epoch,
                        epoch_start_step=epoch_start_step,
                        epoch_end_step=epoch_end_step,
                    )

                    current_step = int(jax.device_get(state.step))
                    self._runtime_trace(
                        "training_loop.epoch.end",
                        epoch=epoch,
                        current_step=current_step,
                        run_exception=type(run_exception).__name__ if run_exception is not None else None,
                    )
                    if current_step >= self.max_training_steps:
                        break
                    if run_exception is not None:
                        break
            self._runtime_trace(
                "training_loop.end",
                final_step=int(jax.device_get(state.step)),
                run_exception=type(run_exception).__name__ if run_exception is not None else None,
            )
            return self._prepare_training_output(state=state, run_exception=run_exception), run_exception
        except BaseException as exc:
            self._runtime_trace("training_loop.exception", exc_type=type(exc).__name__, exc=str(exc))
            raise
        finally:
            # Stop the JAX profiler trace (if one was started after step 1).
            # Guarded internally so this is a no-op when profiling was disabled.
            self._stop_profiler()
            pbar.close()

    def _run_evaluation(
        self,
        state: EasyDeLState,
        metrics_tracker: MetricsTracker,
        step_metrics: StepMetrics,
    ):
        """Run the core evaluation loop on the validation dataset.

        Iterates over the evaluation dataset, performs evaluation steps,
        updates metrics, and yields metrics for each step. A progress
        bar is used to indicate evaluation progress.

        Args:
            state: The model state used for evaluation.
            metrics_tracker: Tracker for accumulating evaluation metrics.
            step_metrics: Object to calculate metrics per evaluation step.

        Yields:
            dict: A dictionary of evaluation metrics for each evaluation
            step.
        """
        disabled = False
        if jax.process_index() != 0 and not self.arguments.log_all_workers:
            disabled = True
        pbar = self.create_progress_bar(
            total=self.max_evaluation_steps,
            disabled=disabled,
            desc="evaluation process",
        )

        eval_iter = iter(self.dataloader_eval)
        try:
            with self.mesh:
                yield from self._eval_epoch(
                    state=state,
                    eval_dataset=self.dataloader_eval,
                    eval_iter=eval_iter,
                    metrics_tracker=metrics_tracker,
                    step_metrics=step_metrics,
                    pbar=pbar,
                )
        finally:
            pbar.close()

    def _train_epoch(
        self,
        state: EasyDeLState,
        train_dataset,
        train_iter,
        metrics_tracker: MetricsTracker,
        step_metrics: StepMetrics,
        pbar: BaseProgressBar,
        epoch: int,
        epoch_start_step: int | None = None,
        epoch_end_step: int | None = None,
    ):
        """Execute training for a single epoch.

        Processes batches within an epoch, handling:

        - Batch fetching and collation (with optional host-side prefetch).
        - Forward and backward passes.
        - Gradient accumulation if configured.
        - Metrics computation and logging.
        - Checkpoint saving at specified intervals.
        - Optional evaluation during training.
        - Training hooks for customization.

        Robust error handling lets training be gracefully interrupted
        (Ctrl+C, timeout, TPU preemption) and the current state /
        iterator are returned so the caller can persist them.

        Args:
            state: Current model state with parameters and optimizer.
            train_dataset: Training data source (dataset or dataloader).
            train_iter: Iterator over training batches.
            metrics_tracker: Accumulates loss and accuracy metrics.
            step_metrics: Computes per-step performance metrics.
            pbar: Progress bar for visual feedback.
            epoch: Current epoch number (0-indexed).
            epoch_start_step: Optional pre-computed global step at which
                this epoch starts; falls back to
                :meth:`_get_epoch_step_bounds` when omitted.
            epoch_end_step: Optional pre-computed global step at which
                this epoch ends.

        Returns:
            tuple: ``(updated_state, exception, iterator)`` where
            ``updated_state`` is the model state after the epoch,
            ``exception`` is any exception that interrupted training
            (``None`` on normal completion), and ``iterator`` is the
            updated batch iterator to use for the next epoch.

        Note:
            - Implements ``on_step_start`` and ``on_step_end`` hooks.
            - Applies training hooks for loss validation.
            - Saves checkpoints based on ``save_steps`` configuration.
            - Runs evaluation based on ``evaluation_steps`` configuration.
        """
        data_collator = self.data_collator
        if data_collator is None:

            def data_collator(x):
                """Identity collator used as a fallback when no collator is configured.

                Args:
                    x: Already-batched input.

                Returns:
                    The input unchanged.
                """
                return x

        if self.max_training_steps is None:
            raise RuntimeError("max_training_steps must be set before training")
        if epoch_start_step is None or epoch_end_step is None:
            epoch_start_step, epoch_end_step = self._get_epoch_step_bounds(epoch)
        epoch_total_steps = max(epoch_end_step - epoch_start_step, 1)
        run_exception: Exception | None = None
        self._runtime_trace(
            "train_epoch.begin",
            epoch=epoch,
            epoch_start_step=epoch_start_step,
            epoch_end_step=epoch_end_step,
            epoch_total_steps=epoch_total_steps,
        )
        prefetcher: _TrainBatchPrefetcher | None = None

        def close_prefetcher() -> collections.abc.Iterator[tp.Any]:
            nonlocal prefetcher, train_iter
            if prefetcher is not None:
                train_iter = prefetcher.data_iter
                prefetcher.close()
                prefetcher = None
            return train_iter

        if bool(getattr(self.arguments, "dataloader_prefetch", False)):
            prefetcher = _TrainBatchPrefetcher(
                trainer=self,
                data_iter=train_iter,
                dataloader=train_dataset,
                data_collator=data_collator,
            )
            self._runtime_trace("train_epoch.prefetch.enabled", epoch=epoch, buffer_size=1)

        while True:
            with capture_time() as iteration_time:
                current_step = int(jax.device_get(state.step))
                if current_step >= self.max_training_steps or current_step >= epoch_end_step:
                    self._runtime_trace(
                        "train_epoch.break",
                        epoch=epoch,
                        current_step=current_step,
                        max_training_steps=self.max_training_steps,
                        epoch_end_step=epoch_end_step,
                    )
                    break
                try:
                    self._runtime_trace("train_step.batch_fetch.begin", epoch=epoch, current_step=current_step)
                    prefetch_wait_time: float | None = None
                    prefetch_producer_time: float | None = None
                    with capture_time() as data_collection_time:
                        if prefetcher is None:
                            batch, train_iter = self._get_next_batch(train_iter, train_dataset)
                            self._runtime_trace(
                                "train_step.batch_fetch.end",
                                epoch=epoch,
                                current_step=current_step,
                                batch=self._runtime_batch_summary(batch),
                                prefetch=False,
                            )
                            self._runtime_trace("train_step.collate.begin", epoch=epoch, current_step=current_step)
                            batch = data_collator(batch)
                        else:
                            schedule_next = (
                                current_step + 1 < self.max_training_steps and current_step + 1 < epoch_end_step
                            )
                            batch, prefetch_wait_time, prefetch_producer_time = prefetcher.next(
                                schedule_next=schedule_next
                            )
                            train_iter = prefetcher.data_iter
                            self._runtime_trace(
                                "train_step.batch_fetch.end",
                                epoch=epoch,
                                current_step=current_step,
                                batch=self._runtime_batch_summary(batch),
                                prefetch=True,
                                prefetch_wait_time=prefetch_wait_time,
                                prefetch_producer_time=prefetch_producer_time,
                                prefetch_next_scheduled=schedule_next,
                            )
                        self._runtime_trace(
                            "train_step.collate.end",
                            epoch=epoch,
                            current_step=current_step,
                            batch=self._runtime_batch_summary(batch),
                            prefetch=prefetcher is not None,
                        )
                    step_metrics.start_step()
                    self._runtime_trace("train_step.on_step_start.begin", epoch=epoch, current_step=current_step)
                    state = self.on_step_start(state=state, step=current_step)
                    self._runtime_trace("train_step.on_step_start.end", epoch=epoch, current_step=current_step)
                except (KeyboardInterrupt, EasyDeLTimerError, EasyDeLBreakRequest, EasyDeLPreemptionSignal) as exc:
                    self._runtime_trace(
                        "train_step.setup.interrupt",
                        epoch=epoch,
                        current_step=current_step,
                        exc_type=type(exc).__name__,
                        exc=str(exc),
                    )
                    return state, exc, close_prefetcher()

                # Execute training step
                self._runtime_trace("train_step.execute.begin", epoch=epoch, current_step=current_step)
                with self.train_tracker.trace_compilation():
                    with capture_time() as execution_time:
                        state, metrics, run_exception = self._execute_train_step(state=state, batch=batch)
                        metrics.execution_time = execution_time()
                        current_step = int(jax.device_get(state.step))
                self._runtime_trace(
                    "train_step.execute.end",
                    epoch=epoch,
                    current_step=current_step,
                    execution_time=float(execution_time()),
                    run_exception=type(run_exception).__name__ if run_exception is not None else None,
                )
                if run_exception is not None:
                    self._runtime_trace(
                        "train_step.execute.run_exception",
                        epoch=epoch,
                        current_step=current_step,
                        exc_type=type(run_exception).__name__,
                        exc=str(run_exception),
                    )
                    return state, run_exception, close_prefetcher()
                # Start the JAX profiler once step 1 has fully completed.
                # The first step's wall-time is dominated by JIT compile;
                # skipping it gives a profile of steady-state training.
                self._maybe_start_profiler(current_step)
                try:
                    self._runtime_trace("train_step.host_metrics.begin", epoch=epoch, current_step=current_step)
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
                            self.scheduler(current_step) if self.scheduler is not None else self.arguments.learning_rate
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
                    train_metrics["performance/data_collection_time"] = float(data_collection_time())
                    if prefetch_wait_time is not None:
                        train_metrics["performance/data_prefetch_wait_time"] = float(prefetch_wait_time)
                    if prefetch_producer_time is not None:
                        train_metrics["performance/data_prefetch_producer_time"] = float(prefetch_producer_time)
                    train_step_time = train_metrics.get("performance/train_step_time")
                    if train_step_time is not None:
                        remaining_steps = max(self.max_training_steps - current_step, 0)
                        remaining_seconds = remaining_steps * float(train_step_time)
                        train_metrics["performance/remaining_minutes"] = remaining_seconds / 60.0
                        train_metrics["performance/remaining_hours"] = remaining_seconds / 3600.0
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
                        if jax.process_index() == 0 or self.arguments.log_all_workers:
                            logger.warning(
                                f"TPU preemption sync point reached at step {current_step}. Saving coordinated checkpoint."
                            )
                        self._save_tpu_preemption_checkpoint(state=state, step=current_step)
                        return state, EasyDeLPreemptionSignal("TPU preemption checkpoint saved"), close_prefetcher()
                    with capture_time() as weight_distribution_time:
                        self.log_weight_distribution(state=state, step=current_step)
                    with capture_time() as watchers_time:
                        self.log_watchers(state=state, step=current_step)
                    with capture_time() as generation_time:
                        try:
                            self.maybe_generate(state=state, step=current_step, metrics=metrics)
                        except Exception as exc:  # pragma: no cover - preview must not interrupt training
                            logger.warning(f"Preview generation hook failed: {exc}")
                    with capture_time() as benchmark_time:
                        try:
                            self.maybe_benchmark(state=state, step=current_step)
                        except Exception as exc:  # pragma: no cover - benchmarks must not interrupt training
                            logger.warning(f"Benchmark hook failed: {exc}")

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
                        self._runtime_trace("train_step.profiler_block.begin", epoch=epoch, current_step=current_step)
                        state, metrics = jax.block_until_ready((state, metrics))
                        self._runtime_trace("train_step.profiler_block.end", epoch=epoch, current_step=current_step)
                    self._runtime_trace("train_step.host_metrics.end", epoch=epoch, current_step=current_step)
                except (KeyboardInterrupt, EasyDeLTimerError, EasyDeLBreakRequest) as exc:
                    self._runtime_trace(
                        "train_step.host_metrics.interrupt",
                        epoch=epoch,
                        current_step=current_step,
                        exc_type=type(exc).__name__,
                        exc=str(exc),
                    )
                    return state, exc, close_prefetcher()
                except TypeError as exc:
                    self._runtime_trace(
                        "train_step.host_metrics.type_error",
                        epoch=epoch,
                        current_step=current_step,
                        exc_type=type(exc).__name__,
                        exc=str(exc),
                    )
                    return state, exc, close_prefetcher()
                if run_exception is not None:
                    break
        close_prefetcher()
        self._runtime_trace(
            "train_epoch.end",
            epoch=epoch,
            final_step=int(jax.device_get(state.step)),
            run_exception=type(run_exception).__name__ if run_exception is not None else None,
        )
        return state, run_exception, train_iter

    def _eval_epoch(
        self,
        state: EasyDeLState,
        eval_dataset,
        eval_iter,
        metrics_tracker: MetricsTracker,
        step_metrics: StepMetrics,
        pbar: BaseProgressBar,
    ):
        """Execute a single evaluation epoch.

        Iterates over the evaluation dataset, processes each batch
        through the compiled evaluation step, updates and logs metrics,
        and yields the per-step evaluation metrics. Does not update
        model parameters.

        Args:
            state: The model state used for evaluation.
            eval_dataset: The evaluation dataset (or an iterator over
                it).
            eval_iter: Iterator over evaluation batches.
            metrics_tracker: Tracker for accumulating evaluation
                metrics.
            step_metrics: Object to calculate step-level metrics.
            pbar: Progress bar instance for displaying evaluation
                progress.

        Yields:
            dict: A dictionary of evaluation metrics for each evaluation
            step.

        Raises:
            ValueError: If ``eval_dataset`` is ``None``.
        """
        if eval_dataset is None:
            raise ValueError("Make sure to pass eval dataset to trainer or set `do_eval` to `False`.")
        data_collator = self.data_collator
        if data_collator is None:

            def data_collator(x):
                """Identity collator used as a fallback when no collator is configured.

                Args:
                    x: Already-batched input.

                Returns:
                    The input unchanged.
                """
                return x

        global_step = int(jax.device_get(state.step))
        final_eval_metrics = None
        summary_metric_sums: dict[str, float] = {}
        summary_metric_counts: dict[str, int] = {}
        summary_metrics_helper = (
            step_metrics
            if hasattr(step_metrics, "accumulate_summary_metric") and hasattr(step_metrics, "summarize_metrics")
            else StepMetrics(self.arguments)
        )

        for current_step in range(1, self.max_evaluation_steps + 1):
            try:
                with capture_time() as data_collection_time:
                    batch, eval_iter = self._get_next_batch(eval_iter, eval_dataset)
                    batch = data_collator(batch)
                step_metrics.start_step()
                with self.evalu_tracker.trace_compilation():
                    with capture_time() as execution_time:
                        metrics = self._execute_eval_step(state, batch)
                        metrics.execution_time = execution_time()
                mean_loss, mean_accuracy = metrics_tracker.update(
                    metrics.loss,
                    metrics.accuracy,
                    current_step,
                )
                eval_metrics = step_metrics.calculate(
                    metrics=metrics,
                    current_step=current_step,
                    learning_rate=0.000,
                    epoch=0,
                    epoch_progress=None,
                    flops_per_token=self._forward_flops_per_token,
                    extra_flops_per_token=self._extra_forward_flops_per_token,
                    batch_size=self.evaluation_batch_size,
                    seq_length=self.arguments.max_length,
                    mean_loss=mean_loss,
                    mean_accuracy=mean_accuracy,
                    mode="eval",
                )
                eval_metrics["performance/eval/data_collection_time"] = float(data_collection_time())
                for metric_name, metric_value in eval_metrics.items():
                    summary_metrics_helper.accumulate_summary_metric(
                        summary_metric_sums=summary_metric_sums,
                        summary_metric_counts=summary_metric_counts,
                        metric_name=metric_name,
                        metric_value=metric_value,
                        mode="eval",
                    )
                self.log_metrics(
                    metrics=eval_metrics,
                    pbar=pbar,
                    step=current_step,
                    mode="eval",
                    log_to_backends=False,
                )
                final_eval_metrics = eval_metrics
                yield eval_metrics
            except (KeyboardInterrupt, EasyDeLTimerError, EasyDeLBreakRequest):
                break
            except TypeError:
                raise
        if final_eval_metrics is not None:
            summary_eval_metrics = summary_metrics_helper.summarize_metrics(
                last_metrics=final_eval_metrics,
                summary_metric_sums=summary_metric_sums,
                summary_metric_counts=summary_metric_counts,
                mode="eval",
            )
            self.log_metrics(
                metrics=summary_eval_metrics,
                pbar=pbar,
                step=global_step,
                mode="eval",
                update_progress=False,
                force_report=True,
            )

    def _execute_eval_step(self, state, batch) -> LossMetrics:
        """Run one evaluation step end-to-end (preprocess + compiled forward).

        First routes ``batch`` through :meth:`_preprocess_batch_input`
        in eval mode so any trainer-specific preprocessing (rollouts,
        teacher scoring, label masking, etc.) is applied. The returned
        ``informations`` dict carries auxiliary scalars produced by
        preprocessing that would otherwise be lost (generation
        timings, reward statistics, etc.) and is merged into
        ``metrics.other_metrics`` after the compiled step returns. The
        compiled :func:`evaluation_step` (or trainer-specific override)
        is then invoked with the cached eval-side static argument
        tuple and any ``_eval_shared_fn_extra_args`` (such as a teacher
        or reference state for distillation/preference trainers).

        Args:
            state: Current model state used for the evaluation forward.
            batch: Already-collated evaluation batch.

        Returns:
            :class:`LossMetrics` with ``loss``, ``accuracy``, and any
            auxiliary metrics from the trainer-specific eval step
            merged with the preprocessing ``informations`` dict.
        """
        batch, informations = self._preprocess_batch_input(
            state=state,
            batch=batch,
            is_train=False,
        )
        metrics = self.sharded_evaluation_step_function(
            state,
            batch,
            *self._eval_shared_fn_extra_args,
            *self._eval_shared_fn_static_args,
        )
        if len(informations) != 0:
            if metrics.other_metrics is not None:
                informations.update(metrics.other_metrics)
            metrics = metrics.replace(other_metrics=informations)
        return metrics

    def _execute_train_step(
        self,
        state,
        batch,
    ) -> tuple[EasyDeLState, LossMetrics, BaseException | None]:
        """
        Execute a single training step with gradient computation and updates.

        This method performs a complete training iteration:
        1. Pre-forward pruning updates (if configured)
        2. Batch preprocessing with custom hooks
        3. Forward pass and loss computation
        4. Backward pass and gradient computation
        5. Parameter updates via optimizer
        6. Post-gradient pruning updates (if configured)

        The method handles various training strategies:
        - Gradient accumulation (handled in the compiled function)
        - Mixed precision training (via dtype configuration)
        - Model pruning (via pruning_module hooks)
        - Custom preprocessing (via _preprocess_batch_input)

        Args:
            state: Current model state containing parameters and optimizer state
            batch: Preprocessed batch of training data as a dictionary

        Returns:
            tuple: (updated_state, metrics, exception) where:
                - updated_state: Model state after parameter updates
                - metrics: LossMetrics with loss, accuracy, and custom metrics
                - exception: Any exception caught during execution, None if successful

        Note:
            - Uses jax.block_until_ready to ensure synchronous execution
            - Exceptions are caught to allow graceful shutdown with state saving
            - Custom metrics from preprocessing are merged with training metrics
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
            self._runtime_trace("execute_train_step.preprocess.begin", batch=self._runtime_batch_summary(batch))
            batch, informations = self._preprocess_batch_input(
                state=state,
                batch=batch,
                is_train=True,
            )
            self._runtime_trace(
                "execute_train_step.preprocess.end",
                batch=self._runtime_batch_summary(batch),
                information_keys=tuple(informations.keys()) if isinstance(informations, dict) else None,
            )

            self._runtime_trace("execute_train_step.compiled_call.begin")
            state, metrics = jax.block_until_ready(
                self.sharded_training_step_function(
                    state,
                    batch,
                    *self._train_shared_fn_extra_args,
                    *self._train_shared_fn_static_args,
                )
            )
            self._runtime_trace(
                "execute_train_step.compiled_call.end",
                step=int(jax.device_get(state.step)),
                metrics_type=type(metrics).__name__,
            )

            if len(informations) != 0:
                if metrics.other_metrics is not None:
                    informations.update(metrics.other_metrics)
                metrics = metrics.replace(other_metrics=informations)

            # Apply post-gradient updates via the pruning module, if present.
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
            self._runtime_trace(
                "execute_train_step.control_exception",
                exc_type=type(run_exception).__name__,
                exc=str(run_exception),
            )
            return state, metrics, run_exception
        except Exception as run_exception:
            self._runtime_trace(
                "execute_train_step.exception",
                exc_type=type(run_exception).__name__,
                exc=str(run_exception),
            )
            if self._is_memory_oom_exception(run_exception):
                annotated_exception = self._augment_memory_oom_exception(run_exception)
                logger.error(str(annotated_exception))
                return state, metrics, annotated_exception
            raise

    def _finalize_training(self, output, run_exception):
        """
        Finalizes the training process and prepares the output.

        If evaluation is enabled, this method runs an additional evaluation pass before finishing.
        It then calls the finish method to perform any cleanup and returns the final output.

        Args:
            output: The output object containing the final state and metrics.
            run_exception: Any exception that was encountered during training.

        Returns:
            The final output object.
        """
        try:
            if self.arguments.do_eval:
                for _ in self.eval(output.state):
                    ...
        except RuntimeError:
            logger.info("Caught RuntimeError from eval function (mostly due to `StopIteration` being called manually)")
        self.finish()
        return output

    def train(self) -> TrainerOutput:
        """
        Execute the complete training pipeline.

        This is the main entry point for training. It orchestrates the entire
        training workflow from initialization to completion:

        1. Calls start_training_hook for custom initialization
        2. Sets up metrics tracking and logging infrastructure
        3. Logs initial configuration and model information
        4. Executes the main training loop across all epochs
        5. Handles interruptions and saves final checkpoints
        6. Runs final evaluation if configured
        7. Cleans up resources and returns results

        The method is designed to be robust to interruptions and will save
        the model state before exiting on errors or keyboard interrupts.

        Returns:
            TrainerOutput: Contains:
                - state: Final model state after training
                - mesh: Device mesh used for training
                - checkpoint_path: Path to the final checkpoint
                - last_save_file_name: Name of the last saved file

        Example:
            >>> trainer = Trainer(arguments=args, model=model, ...)
            >>> output = trainer.train()
            >>> print(f"Final loss: {output.state.metrics['loss']}")

        Note:
            - Automatically resumes from checkpoints if configured
            - Saves checkpoints periodically based on save_steps
            - Can be interrupted with Ctrl+C without losing progress
        """
        self._runtime_trace("train.begin")
        try:
            self._runtime_trace("train.start_training_hook.begin")
            self.start_training_hook()
            self._runtime_trace("train.start_training_hook.end")
            state = self.model_state
            metrics_tracker = MetricsTracker()
            step_metrics = StepMetrics(self.arguments)
            self._runtime_trace("train.setup_initial_metrics.begin")
            self._setup_initial_metrics(state)
            self._runtime_trace("train.setup_initial_metrics.end")
            output, run_exception = self._run_training_loop(
                state=self.model_state,
                metrics_tracker=metrics_tracker,
                step_metrics=step_metrics,
            )
            self._runtime_trace(
                "train.finalize.begin",
                run_exception=type(run_exception).__name__ if run_exception is not None else None,
            )
            output = self._finalize_training(output, run_exception)
            self._runtime_trace("train.end")
            return output
        except BaseException as exc:
            self._runtime_trace("train.exception", exc_type=type(exc).__name__, exc=str(exc))
            raise

    def eval(self, model_state: EasyDeLState) -> collections.abc.Iterator[dict]:
        """
        Evaluate the model on the evaluation dataset.

        This method performs model evaluation without gradient computation,
        yielding metrics for each evaluation step. It's useful for:
        - Periodic evaluation during training
        - Final model evaluation after training
        - Standalone evaluation of checkpoints

        The evaluation process:
        1. Switches to evaluation mode (no gradient computation)
        2. Iterates through the evaluation dataset
        3. Computes forward passes and metrics
        4. Yields metrics for monitoring and analysis
        5. Handles multi-host synchronization

        Args:
            model_state: Model state containing parameters for evaluation.
                        This can be different from the training state,
                        allowing evaluation of checkpoints or other models.

        Yields:
            dict: Evaluation metrics for each step, including:
                - loss: Average loss value
                - accuracy: Average accuracy (if applicable)
                - throughput: Tokens/samples per second
                - Additional model-specific metrics

        Raises:
            ValueError: If evaluation dataloader is not configured

        Example:
            >>> for metrics in trainer.eval(model_state):
            ...     print(f"Eval loss: {metrics['eval/loss']}")

        Note:
            - Evaluation is performed without gradient computation
            - Catches RuntimeError from multi-host synchronization issues
            - Progress bar shows evaluation progress in real-time
        """
        self.start_evaluation_hook()
        try:
            metrics_tracker = MetricsTracker()
            step_metrics = StepMetrics(self.arguments)
            yield from self._run_evaluation(
                state=model_state,
                metrics_tracker=metrics_tracker,
                step_metrics=step_metrics,
            )
        except RuntimeError:
            # In multi-host evaluation, RuntimeError might be raised; catch and continue.
            ...
