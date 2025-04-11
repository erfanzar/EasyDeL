# Trainer Protocol Documentation

## Introduction

The `BaseTrainerProtocol` defines the abstract interface that all trainer classes within the EasyDeL framework must adhere to. It acts as a contract, ensuring that different trainer implementations (e.g., for SFT, DPO, custom tasks) share a common set of core functionalities and properties.

Using a protocol offers several advantages:

* **Consistency**: Provides a predictable structure and behavior across various training types.
* **Modularity**: Allows different trainers to be used interchangeably where appropriate.
* **Extensibility**: Simplifies the creation of new, custom trainers by providing a clear blueprint of required components.
* **Type Safety**: Enforces type hints through Python's `typing.Protocol`, improving code reliability and maintainability.

The primary concrete implementation of this protocol in EasyDeL is the `BaseTrainer` class. For details on how to *use* the standard trainer, refer to the [BaseTrainer Documentation](./base_trainer.md). This document focuses on the requirements for *implementing* a class that conforms to the `BaseTrainerProtocol`.

## Protocol Requirements

Any class implementing `BaseTrainerProtocol` must define the following methods and properties:

### Required Methods

These methods define the core lifecycle and operations of a trainer.

* **`train(self, model_parameters: Optional[FlaxPreTrainedModel.params] = None, state: Optional[EasyDeLState] = None) -> EasyDeLState`**:
  * **Purpose**: The main entry point to start or resume the training process.
  * **Implementation**: Must orchestrate the entire training loop, including:
    * Loading data using configured dataloaders.
    * Iterating through epochs and steps.
    * Executing the core training step function (often JIT-compiled).
    * Managing and updating the `EasyDeLState` (model parameters, optimizer state, step count).
    * Handling metric logging and progress reporting.
    * Implementing checkpoint saving logic based on `TrainingArguments`.
    * Optionally triggering evaluation based on `TrainingArguments`.
    * Handling potential resumption from a checkpoint (`state` or loading from `ckpt_path` implicitly via `arguments`).
  * **Arguments**:
    * `model_parameters`: Optional initial model parameters (usually handled internally by loading from `arguments.model_name_or_path` or a checkpoint).
    * `state`: Optional `EasyDeLState` to resume training from.
  * **Returns**: The final `EasyDeLState` after training completes.

* **`evaluate(self, state: EasyDeLState, metric_calculator: Optional[Callable] = None) -> Dict[str, float]`**:
  * **Purpose**: Evaluate the model's performance on the evaluation dataset.
  * **Implementation**: Must iterate through the evaluation dataloader, execute the core evaluation step function (often JIT-compiled), aggregate metrics, and return the results. Should handle distributed evaluation correctly if applicable.
  * **Arguments**:
    * `state`: The `EasyDeLState` containing the model parameters to evaluate.
    * `metric_calculator`: An optional callable to compute custom metrics beyond simple loss.
  * **Returns**: A dictionary mapping metric names (e.g., `"eval_loss"`, `"accuracy"`) to their computed values.

* **`save_pretrained(self, ckpt_path: str, state: Optional[EasyDeLState] = None)`**:
  * **Purpose**: Save the current training progress (model, optimizer, configuration) to disk.
  * **Implementation**: Must serialize and save:
    * Model parameters (from `state.params`).
    * Optimizer state (from `state.tx_state`).
    * Training arguments (`self.arguments`).
    * Any other necessary metadata for resuming (e.g., tokenizer files if applicable).
    * Should handle potential sharding/distributed saving correctly.
  * **Arguments**:
    * `ckpt_path`: The directory path where the checkpoint should be saved.
    * `state`: The `EasyDeLState` to save. If `None`, the trainer should use its internal current state.

* **`configure_functions(self) -> TrainerConfigureFunctionOutput`**:
  * **Purpose**: Define and potentially JIT-compile the core training and evaluation step functions.
  * **Implementation**: Must return a `TrainerConfigureFunctionOutput` named tuple containing:
    * `train_step_fn`: The function that takes `EasyDeLState` and a batch, performs a single training step (forward, loss, backward, optimizer step), and returns the updated state and metrics.
    * `eval_step_fn`: The function that takes `EasyDeLState` and a batch, performs a forward pass, calculates evaluation metrics, and returns them.
  * **Returns**: `TrainerConfigureFunctionOutput(train_step_fn: Callable, eval_step_fn: Callable)`

* **`configure_model(self) -> TrainerConfigureModelOutput`**:
  * **Purpose**: Initialize the model and the initial `EasyDeLState`.
  * **Implementation**: Must load or create the model architecture based on `arguments`, initialize its parameters, set up the optimizer based on `arguments`, and bundle these into an `EasyDeLState`.
  * **Returns**: `TrainerConfigureModelOutput(model: Module, state: EasyDeLState)`

* **`configure_dataloaders(self) -> TrainerConfigureDataloaderOutput`**:
  * **Purpose**: Set up the dataloaders for training and evaluation.
  * **Implementation**: Must prepare and return the dataloaders based on `arguments` and the provided datasets (`self.dataset_train`, `self.dataset_eval`).
  * **Returns**: `TrainerConfigureDataloaderOutput(dataloader_train: Any, dataloader_eval: Optional[Any])`

### Required Properties

These properties ensure the trainer has access to essential configuration and data.

* **`arguments: TrainingArguments`**: An instance holding all training hyperparameters and settings.
* **`dataset_train: Dataset`**: The dataset used for training (typically a `datasets.Dataset` or compatible).
* **`dataset_eval: Optional[Dataset]`**: The dataset used for evaluation (optional).
* **`model: Module`**: The Flax/EasyDeL model module being trained (must be available after `configure_model` is called).
* **`_model_state: EasyDeLState`**: The internal `EasyDeLState` managed by the trainer (must be available after `configure_model` is called).
* **`dtype: jnp.dtype`**: The data type used for computations (e.g., `jnp.float32`, `jnp.bfloat16`). Derived from `arguments`.
* **`param_dtype: jnp.dtype`**: The data type used for model parameters (e.g., `jnp.float32`). Derived from `arguments`.
* **`scheduler: Optional[OptaxSchedule]`**: The learning rate scheduler (must be available after `configure_model` is called).
* **`optimizer: Optional[optax.GradientTransformation]`**: The Optax optimizer (must be available after `configure_model` is called).

## Implementation Skeleton Example

```python
from abc import ABC, abstractmethod
from typing import Optional, Dict, Callable, Any
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.linen import Module
from datasets import Dataset

from easydel.trainers.trainer_protocol import (
    BaseTrainerProtocol,
    TrainerConfigureFunctionOutput,
    TrainerConfigureModelOutput,
    TrainerConfigureDataloaderOutput
)
from easydel.trainers.training_args import TrainingArguments
# Assume EasyDeLState is defined appropriately, inheriting from TrainState
class EasyDeLState(TrainState):
    pass # Add necessary fields if extending

class MyCustomTrainer(BaseTrainerProtocol):
    def __init__(self, arguments: TrainingArguments, dataset_train: Dataset, dataset_eval: Optional[Dataset] = None, **kwargs):
        self.arguments = arguments
        self.dataset_train = dataset_train
        self.dataset_eval = dataset_eval
        self.dtype = getattr(jnp, arguments.dtype) if arguments.dtype else jnp.float32
        self.param_dtype = getattr(jnp, arguments.param_dtype) if arguments.param_dtype else jnp.float32

        # Initialize state variables that will be populated by configure_* methods
        self.model: Optional[Module] = None
        self._model_state: Optional[EasyDeLState] = None
        self.optimizer: Optional[optax.GradientTransformation] = None
        self.scheduler: Optional[optax.Schedule] = None
        self.dataloader_train: Optional[Any] = None
        self.dataloader_eval: Optional[Any] = None
        self.train_step_fn: Optional[Callable] = None
        self.eval_step_fn: Optional[Callable] = None

        # Call configuration methods (order might matter depending on dependencies)
        model_output = self.configure_model()
        self.model = model_output.model
        self._model_state = model_output.state
        # Assume optimizer/scheduler are set within configure_model or state

        dataloader_output = self.configure_dataloaders()
        self.dataloader_train = dataloader_output.dataloader_train
        self.dataloader_eval = dataloader_output.dataloader_eval

        function_output = self.configure_functions()
        self.train_step_fn = function_output.train_step_fn
        self.eval_step_fn = function_output.eval_step_fn

    @abstractmethod
    def configure_model(self) -> TrainerConfigureModelOutput:
        # Load model, create optimizer/scheduler, build EasyDeLState
        # model = ...
        # state = ...
        # self.optimizer = state.opt_state # Or however it's stored
        # self.scheduler = ...
        # return TrainerConfigureModelOutput(model=model, state=state)
        raise NotImplementedError

    @abstractmethod
    def configure_dataloaders(self) -> TrainerConfigureDataloaderOutput:
        # Create train/eval dataloaders based on self.dataset_* and self.arguments
        # train_loader = ...
        # eval_loader = ...
        # return TrainerConfigureDataloaderOutput(dataloader_train=train_loader, dataloader_eval=eval_loader)
        raise NotImplementedError

    @abstractmethod
    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        # Define train_step and eval_step logic, potentially JIT compile
        # train_fn = jax.jit(...)
        # eval_fn = jax.jit(...)
        # return TrainerConfigureFunctionOutput(train_step_fn=train_fn, eval_step_fn=eval_fn)
        raise NotImplementedError

    @abstractmethod
    def train(self, model_parameters=None, state=None) -> EasyDeLState:
        # Implement the main training loop using configured components
        # (dataloaders, train_step_fn, state management, logging, checkpointing)
        # current_state = state or self._model_state
        # ... loop over epochs/steps ...
        #    batch = next(self.dataloader_train)
        #    current_state, metrics = self.train_step_fn(current_state, batch)
        #    ... log metrics ...
        #    ... handle eval ...
        #    ... handle checkpointing ...
        # self._model_state = current_state
        # return self._model_state
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, state: EasyDeLState, metric_calculator: Optional[Callable] = None) -> Dict[str, float]:
        # Implement evaluation loop using self.dataloader_eval and self.eval_step_fn
        # metrics = {}
        # for batch in self.dataloader_eval:
        #    batch_metrics = self.eval_step_fn(state, batch)
        #    ... aggregate metrics ...
        # if metric_calculator:
        #    ... use calculator ...
        # return aggregated_metrics
        raise NotImplementedError

    @abstractmethod
    def save_pretrained(self, ckpt_path: str, state: Optional[EasyDeLState] = None):
        # Implement logic to save state.params, state.tx_state, arguments etc.
        # to ckpt_path
        # state_to_save = state or self._model_state
        # ... save logic ...
        raise NotImplementedError

## Best Practices for Implementation

1.  **State Management**: Ensure `EasyDeLState` is consistently updated and managed, especially `state.step`.
2.  **Immutability**: Respect JAX's functional nature. Training/evaluation step functions should be pure and return new states/metrics rather than modifying inputs in place.
3.  **Error Handling**: Implement robust handling for potential issues like OOM errors, data loading failures, or numerical instability (e.g., NaN losses). Consider adding try-except blocks around critical sections like the training step.
4.  **Logging**: Provide clear and configurable logging (e.g., using `logging` module or TensorBoard) for loss, metrics, learning rate, and system stats (memory usage). Leverage `arguments.logging_steps`.
5.  **Checkpointing**: Implement reliable checkpointing triggered by `arguments.save_strategy` and `arguments.save_steps`/`epochs`. Ensure checkpoints include everything needed to resume. Handle `arguments.save_total_limit`.
6.  **Distributed Training**: Design step functions (`train_step_fn`, `eval_step_fn`) and checkpointing logic with JAX parallelism (`pmap`, `shmap`) in mind. Ensure gradients are correctly aggregated and parameters are synchronized/saved across devices.
7.  **Resource Management**: Ensure proper cleanup of resources, especially when dealing with external libraries or large datasets.
