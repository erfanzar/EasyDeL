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
import typing as tp

import jax
import jax.experimental
import jax.lib
from eformer.escale import with_sharding_constraint
from jax.sharding import NamedSharding, PartitionSpec

from easydel.infra.base_state import EasyDeLState
from easydel.infra.errors import EasyDeLBreakRequest, EasyDeLTimerError
from easydel.infra.loss_utils import LossMetrics
from easydel.utils.compiling_utils import ejit
from easydel.utils.helpers import capture_time, get_logger

from ..base_trainer import BaseTrainer, TrainerConfigureFunctionOutput
from ..trainer_protocol import BaseProgressBar, MetricsTracker, StepMetrics
from ._fn import evaluation_step, training_step
from .modeling_output import TrainerOutput

logger = get_logger(__name__)


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
        """
        Creates a collate/collect function to process batches of data for training or evaluation.

        This function returns a callable that takes a batch (a list of dictionaries) and converts it
        into a dictionary of JAX arrays. For models of class "ForCausalLMLoss", it also performs
        truncation (either keeping the end or the start of the sequence) so that each sequence does not
        exceed the specified maximum length.

        Args:
            max_sequence_length (int): The maximum allowed sequence length.
            truncation_mode (tp.Literal["keep_end", "keep_start"], optional):
                Determines whether to keep the end or the start of the sequence when truncating.
                Defaults to "keep_end".

        Returns:
            tp.Callable: A function that takes a batch (list of dicts) and returns a processed dict of arrays.
        """

        def collate_fn(batch):
            return batch

        return collate_fn

    def create_tfds_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """
        Creates a collate/collect function to process batches of data for training or evaluation.

        This function returns a callable that takes a batch (a list of dictionaries) and converts it
        into a dictionary of JAX arrays. For models of class "ForCausalLMLoss", it also performs
        truncation (either keeping the end or the start of the sequence) so that each sequence does not
        exceed the specified maximum length.

        Args:
            max_sequence_length (int): The maximum allowed sequence length.
            truncation_mode (tp.Literal["keep_end", "keep_start"], optional):
                Determines whether to keep the end or the start of the sequence when truncating.
                Defaults to "keep_end".

        Returns:
            tp.Callable: A function that takes a batch (list of dicts) and returns a processed dict of arrays.
        """

        def collate_fn(batch):
            results = {}
            for key in batch[0].keys():
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
        empty_sharding = jax.sharding.NamedSharding(spec=PartitionSpec(), mesh=self.model.mesh)
        self._train_shared_fn_static_args = (
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
        )
        sharded_training_step_function = ejit(
            training_step,
            static_argnums=(2, 3, 4, 5),
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
        )

        self._eval_shared_fn_static_args = (
            self.arguments.loss_config,
            self.arguments.step_partition_spec,
        )
        sharded_evaluation_step_function = ejit(
            evaluation_step,
            static_argnums=(2, 3),
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=(empty_sharding),
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

    def _all_gather(self, arr: jax.Array) -> jax.Array:
        return jax.device_put(arr, NamedSharding(self.model.mesh, PartitionSpec()))

    def _one_to_all(self, arr: jax.Array) -> jax.Array:
        with self.mesh:
            arr = with_sharding_constraint(arr, PartitionSpec(None))
        return arr

    def _run_training_loop(
        self,
        state: EasyDeLState,
        metrics_tracker: MetricsTracker,
        step_metrics: StepMetrics,
    ):
        """
        Execute the main training loop across all epochs.

        This method orchestrates the entire training process, managing:
        - Epoch iteration with proper resumption handling
        - Progress tracking and reporting
        - Batch processing and gradient updates
        - Checkpoint saving at specified intervals
        - Early stopping on interruption or time limits

        The method handles resumption differently for Grain (seekable) and
        TensorFlow (non-seekable) datasets. For Grain, it can resume from
        the exact position, while for TF datasets it starts fresh but
        continues from the saved model state.

        Args:
            state: Initial model state with parameters and optimizer state
            metrics_tracker: Accumulates metrics across training steps
            step_metrics: Calculates per-step metrics like throughput

        Returns:
            tuple: (TrainerOutput, exception) where:
                - TrainerOutput contains final state and checkpoint info
                - exception is any error that caused training to stop

        Note:
            - Progress bar is disabled on non-primary processes by default
            - Training can be interrupted with Ctrl+C and will save state
            - Automatic resumption updates the progress bar to show continuation
        """
        disabled = False
        if jax.process_index() != 0 and not self.arguments.log_all_workers:
            disabled = True
        pbar = self.create_progress_bar(
            total=self.max_training_steps,
            disabled=disabled,
            desc="training process",
        )

        # Handle resumption based on dataset type
        initial_step = int(jax.device_get(state.step))
        start_epoch = 0

        if initial_step > 0:
            pbar.update(initial_step)
            steps_per_epoch = self.max_training_steps // self.arguments.num_train_epochs

            if self.arguments.use_grain:
                logger.info(f"Resuming Grain dataset from step {initial_step}")
                start_epoch = initial_step // steps_per_epoch
            else:
                logger.info(
                    f"Resuming training from step {initial_step} (non-seekable dataset, starting fresh data iteration)"
                )

        train_iter = iter(self.dataloader_train)
        try:
            run_exception = None
            with self.mesh:
                for epoch in range(start_epoch, self.arguments.num_train_epochs):
                    state, run_exception, train_iter = self._train_epoch(
                        state=state,
                        train_dataset=self.dataloader_train,
                        train_iter=train_iter,
                        metrics_tracker=metrics_tracker,
                        step_metrics=step_metrics,
                        pbar=pbar,
                        epoch=epoch,
                    )

                    current_step = int(jax.device_get(state.step))
                    if current_step >= self.max_training_steps:
                        break
                    if run_exception is not None:
                        break
            return self._prepare_training_output(state=state, run_exception=run_exception), run_exception
        finally:
            pbar.close()

    def _run_evaluation(
        self,
        state: EasyDeLState,
        metrics_tracker: MetricsTracker,
        step_metrics: StepMetrics,
    ):
        """
        Implements the core evaluation loop.

        Iterates over the evaluation dataset, performing evaluation steps, updating metrics, and yielding metrics
        for each evaluation step. A progress bar is used to indicate evaluation progress.

        Args:
            state (EasyDeLState): The model state used for evaluation.
            metrics_tracker (MetricsTracker): Tracker for accumulating evaluation metrics.
            step_metrics (StepMetrics): Object to calculate metrics per evaluation step.

        Yields:
            dict: A dictionary containing evaluation metrics for each step.
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
    ):
        """
        Execute training for a single epoch.

        This method processes batches within an epoch, handling:
        - Batch fetching and collation
        - Forward and backward passes
        - Gradient accumulation if configured
        - Metrics computation and logging
        - Checkpoint saving at specified intervals
        - Optional evaluation during training
        - Training hooks for customization

        The method includes robust error handling to gracefully handle
        interruptions and save state before exiting.

        Args:
            state: Current model state with parameters and optimizer
            train_dataset: Training data source (dataset or dataloader)
            train_iter: Iterator over training batches
            metrics_tracker: Accumulates loss and accuracy metrics
            step_metrics: Computes per-step performance metrics
            pbar: Progress bar for visual feedback
            epoch: Current epoch number (0-indexed)

        Returns:
            tuple: (updated_state, exception, iterator) where:
                - updated_state: Model state after the epoch
                - exception: Any exception that interrupted training
                - iterator: Updated batch iterator for next epoch

        Note:
            - Implements on_step_start and on_step_end hooks
            - Applies training hooks for loss validation
            - Saves checkpoints based on save_steps configuration
            - Runs evaluation based on evaluation_steps configuration
        """
        data_collator = self.data_collator
        if data_collator is None:

            def data_collator(x):
                return x

        steps_per_epoch = self.max_training_steps // self.arguments.num_train_epochs

        for _ in range(steps_per_epoch):
            current_step = int(jax.device_get(state.step))
            if current_step >= self.max_training_steps:
                break
            try:
                batch, train_iter = self._get_next_batch(train_iter, train_dataset)
                step_metrics.start_step()
                state = self.on_step_start(state=state, step=current_step)
            except (KeyboardInterrupt, EasyDeLTimerError, EasyDeLBreakRequest, StopIteration) as exect:
                return state, exect, train_iter

            # Execute training step
            with self.train_tracker.trace_compilation():
                with capture_time() as execution_time:
                    state, metrics, run_exception = self._execute_train_step(state=state, batch=data_collator(batch))
                    metrics.execution_time = execution_time()
                    current_step = int(jax.device_get(state.step))
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
                    learning_rate=self.scheduler(current_step)
                    if self.scheduler is not None
                    else self.arguments.learning_rate,
                    epoch=epoch,
                    flops_per_token=self._backward_flops_per_token,
                    extra_flops_per_token=self._extra_backward_flops_per_token,
                    batch_size=self.training_batch_size,
                    seq_length=self.arguments.max_sequence_length,
                    mean_loss=mean_loss,
                    mean_accuracy=mean_accuracy,
                    mode="train",
                )
                state, metrics = self.on_step_end(
                    state=state,
                    metrics=metrics,
                    step=current_step,
                )
                self.log_metrics(
                    metrics=train_metrics,
                    pbar=pbar,
                    step=current_step,
                    mode="train",
                )
                self.log_weight_distribution(state=state, step=current_step)
                # Save checkpoint if needed
                if self._should_save_checkpoint(current_step):
                    _ = self._save_state(
                        state=state,
                        milestone=True,
                        save_directory=self.arguments.save_directory,
                    )
                if self._should_run_evaluation(current_step):
                    for _ in self.eval(model_state=state):
                        ...
            except (KeyboardInterrupt, EasyDeLTimerError, EasyDeLBreakRequest, TypeError):
                return state, run_exception, train_iter
            if run_exception is not None:
                break
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
        """
        Performs evaluation over one epoch.

        Iterates over the evaluation dataset, processes each batch through the evaluation step,
        updates and logs metrics, and yields the evaluation metrics.

        Args:
            state (EasyDeLState): The model state used for evaluation.
            eval_dataset: The evaluation dataset (or an iterator over it).
            metrics_tracker (MetricsTracker): Tracker for accumulating evaluation metrics.
            step_metrics (StepMetrics): Object to calculate step-level metrics.
            pbar (BaseProgressBar): Progress bar instance for displaying evaluation progress.

        Yields:
            dict: A dictionary of evaluation metrics for each evaluation step.
        """
        assert eval_dataset is not None, "Make sure to pass eval dataset to trainer or set `do_eval` to `False`."
        data_collator = self.data_collator
        if data_collator is None:

            def data_collator(x):
                return x

        for current_step in range(1, self.max_evaluation_steps + 1):
            try:
                batch, eval_iter = self._get_next_batch(eval_iter, eval_dataset)
                step_metrics.start_step()
                with self.evalu_tracker.trace_compilation():
                    with capture_time() as execution_time:
                        metrics = self._execute_eval_step(state, data_collator(batch))
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
                    flops_per_token=self._forward_flops_per_token,
                    extra_flops_per_token=self._extra_forward_flops_per_token,
                    batch_size=self.evaluation_batch_size,
                    seq_length=self.arguments.max_sequence_length,
                    mean_loss=mean_loss,
                    mean_accuracy=mean_accuracy,
                    mode="eval",
                )
                self.log_metrics(
                    metrics=eval_metrics,
                    pbar=pbar,
                    step=current_step,
                    mode="eval",
                )
                yield eval_metrics
            except (KeyboardInterrupt, EasyDeLTimerError, EasyDeLBreakRequest, TypeError):
                break

    def _execute_eval_step(self, state, batch) -> LossMetrics:
        """
        Executes a single evaluation step.

        Args:
            state: The current model state.
            batch: A processed batch of evaluation data.

        Returns:
            LossMetrics: The loss metrics computed by the sharded evaluation step function.
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
    ) -> tuple[EasyDeLState, LossMetrics, Exception]:
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
            batch, informations = self._preprocess_batch_input(
                state=state,
                batch=batch,
                is_train=True,
            )

            state, metrics = jax.block_until_ready(
                self.sharded_training_step_function(
                    state,
                    batch,
                    *self._train_shared_fn_extra_args,
                    *self._train_shared_fn_static_args,
                )
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
            return state, metrics, run_exception

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
        self.start_training_hook()
        state = self.model_state
        metrics_tracker = MetricsTracker()
        step_metrics = StepMetrics(self.arguments)
        self._setup_initial_metrics(state)
        output, run_exception = self._run_training_loop(
            state=self.model_state,
            metrics_tracker=metrics_tracker,
            step_metrics=step_metrics,
        )
        return self._finalize_training(output, run_exception)

    def eval(self, model_state: EasyDeLState) -> tp.Iterator[dict]:
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
            AssertionError: If evaluation dataloader is not configured

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
