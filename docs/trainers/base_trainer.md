# BaseTrainer Documentation

## Introduction

The `BaseTrainer` class in EasyDeL provides a robust and flexible foundation for training transformer models. It encapsulates the common logic required for training loops, evaluation, checkpointing, and configuration, allowing users to focus on the specifics of their model and data.

`BaseTrainer` implements the `BaseTrainerProtocol`, ensuring a consistent interface across different training scenarios. Use `BaseTrainer` when you need a feature-rich, configurable trainer for standard training tasks like Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), etc., or as a starting point for more complex custom training workflows.

For details on the abstract interface that `BaseTrainer` implements, refer to the [Trainer Protocol Documentation](./trainer_protocol.md).

## Core Concepts

Understanding these core concepts is crucial for effectively using `BaseTrainer`:

* **`TrainingArguments`**: A dataclass holding all hyperparameters and configuration settings for the training process (e.g., learning rate, batch size, number of epochs, checkpointing strategy, logging options). You initialize the trainer with an instance of this class.
* **`EasyDeLState`**: A Flax `TrainState` subclass that bundles the model parameters, optimizer state, and potentially other training-related variables (like PRNG keys). `BaseTrainer` manages this state throughout the training process.
* **Training Loop**: The core process orchestrated by the `train()` method. It involves iterating over epochs and steps, fetching data batches, executing the training step function, logging metrics, and handling checkpointing.
* **Checkpointing**: `BaseTrainer` automatically saves the `EasyDeLState` (model parameters, optimizer state) and `TrainingArguments` at specified intervals (e.g., every N steps or epochs) to allow resuming training later. It also manages a configurable limit on the number of checkpoints to keep.
* **Distributed Training**: `BaseTrainer` is designed with JAX's `pmap` and `shmap` in mind, enabling efficient training across multiple devices (GPUs/TPUs) with minimal code changes required from the user for standard setups.

## Key Methods and Configuration

`BaseTrainer` relies on several key methods, some of which you might override for customization:

* **`__init__(self, arguments: TrainingArguments, ...)`**: Initializes the trainer. Requires `TrainingArguments` and dataset information.
* **`configure_model(self) -> TrainerConfigureModelOutput`**: (Abstract method in protocol, implemented in `BaseTrainer`) Initializes the model and the initial `EasyDeLState`. It typically loads a pretrained model based on `arguments`. Returns a `TrainerConfigureModelOutput` tuple containing the initialized `model` and `state`.
* **`configure_dataloaders(self) -> TrainerConfigureDataloaderOutput`**: (Abstract method in protocol, implemented in `BaseTrainer`) Sets up the training and evaluation dataloaders based on the provided datasets and `arguments`. Returns a `TrainerConfigureDataloaderOutput` tuple containing `dataloader_train` and `dataloader_eval`.
* **`configure_functions(self) -> TrainerConfigureFunctionOutput`**: (Abstract method in protocol, implemented in subclasses like `SFTTrainer`) Defines the core training and evaluation step functions, often applying `jax.jit` for performance. Returns a `TrainerConfigureFunctionOutput` tuple containing `train_step_fn` and `eval_step_fn`.
* **`create_collect_function(self) -> Callable`**: (Optional override) Defines how batches are collated and preprocessed before being passed to the training/evaluation step functions. Useful for custom padding or data manipulation.
* **`train(self, ckpt_path: Optional[str] = None) -> EasyDeLState`**: The main entry point to start the training process. It orchestrates the entire training loop, including epoch/step iteration, data loading, calling the `train_step_fn`, logging, evaluation (if configured), and checkpointing. Optionally takes a `ckpt_path` to resume training. Returns the final `EasyDeLState`.
* **`eval(self, state: EasyDeLState) -> Dict`**: Runs the evaluation loop using the `eval_step_fn` on the evaluation dataloader. Returns a dictionary of evaluation metrics. Typically called automatically by `train()` if an evaluation dataset and strategy are provided.
* **`save_pretrained(self, ckpt_path: str, state: Optional[EasyDeLState] = None)`**: Saves the current training state (`EasyDeLState`) and `TrainingArguments` to the specified `ckpt_path`. Called automatically during training based on `arguments.save_strategy`.

## Configuration Example (`TrainingArguments`)

```python
from easydel.trainers import TrainingArguments

args = TrainingArguments(
    model_name="MyFineTunedModel",
    num_train_epochs=3,
    learning_rate=2e-5,
    total_batch_size=32, # Effective batch size (per_device_train_batch_size * num_devices)
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    weight_decay=0.01,
    max_steps=-1, # If > 0, overrides num_train_epochs
    gradient_checkpointing="nothing_saveable", # Or "everything_saveable"
    sharding_array=(1, -1, 1, 1, 1), # Device mesh shape for model sharding
    use_fast_kernels=True,
    logging_steps=10,
    save_steps=100, # Save checkpoint every 100 steps
    save_total_limit=2, # Keep only the latest 2 checkpoints
    evaluation_strategy="steps", # Or "epoch"
    eval_steps=100, # Evaluate every 100 steps
    output_dir="runs/my_model_run"
    # Add other relevant TrainingArguments as needed
)

# Example usage (assuming MyTrainerSubclass exists):
# trainer = MyTrainerSubclass(arguments=args, ...)
# trainer.train()
```

## Customization

While `BaseTrainer` handles most standard scenarios, you can customize its behavior:

* **Subclassing**: Create a new class inheriting from `BaseTrainer` (or a more specific trainer like `SFTTrainer`) and override methods like `configure_functions` or `create_collect_function` for custom logic.
* **Hooks**: `BaseTrainer` provides hooks (empty methods like `on_step_start`, `on_step_end`, `on_log`) that you can override in your subclass to inject custom actions at specific points in the training loop without rewriting the entire loop.

```python
from easydel.trainers import BaseTrainer, TrainerConfigureFunctionOutput
import jax
import jax.numpy as jnp # Assuming jnp is needed

class CustomTrainer(BaseTrainer):
    def create_tfds_collect_function(self, *args, **kwargs): # when using tfds
        # Example: Custom data collation
        def collect_fn(batch):
            # Implement your custom batch processing logic here
            processed_batch = batch # Placeholder
            return processed_batch
        return collect_fn

    def create_grain_collect_function(self, *args, **kwargs): # when using grain
        # Example: Custom data collation
        def collect_fn(batch):
            # Implement your custom batch processing logic here
            processed_batch = batch # Placeholder
            return processed_batch
        return collect_fn

    def configure_functions(self, *args, **kwargs) -> TrainerConfigureFunctionOutput:
        # Define your custom train and eval steps
        def train_step(state, batch):
            # Implement custom forward/backward pass logic
            # This is highly dependent on your specific task
            loss = jnp.mean(batch.get("labels", 0.0)) # Placeholder loss
            metrics = {"loss": loss} # Placeholder metrics
            # Assuming gradients are computed somehow (e.g., using jax.grad)
            # grads = ...
            # new_state = state.apply_gradients(grads=grads) # Placeholder state update
            new_state = state # Placeholder
            return new_state, metrics

        def eval_step(state, batch):
            # Implement custom evaluation logic
            # This is highly dependent on your specific task
            loss = jnp.mean(batch.get("labels", 0.0)) # Placeholder loss
            metrics = {"eval_loss": loss} # Placeholder metrics
            return metrics

        return TrainerConfigureFunctionOutput(
            train_step_fn=jax.jit(train_step),
            eval_step_fn=jax.jit(eval_step)
        )

    def on_step_end(self, state, metrics, *args, **kwargs):
        # Example Hook: Print learning rate every step if scheduler exists
        if hasattr(self, "scheduler") and self.scheduler is not None:
             current_lr = self.scheduler(state.step)
             print(f"Step completed. Current LR: {current_lr}")
        # Call parent hook if needed for base functionality
        super().on_step_end(state, metrics, *args, **kwargs)
```
