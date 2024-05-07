import abc
import os
import sys
import threading
import time
import warnings
from abc import abstractmethod
from typing import Union, Callable, Optional, Any, Literal, Mapping, Iterator

import fjformer
import jax
import tensorflow as tf
import termcolor
import wandb
from numpy import ndarray
from datasets import Dataset, IterableDataset
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run
from .training_configurations import TrainArguments
from ..smi import get_capacity_matrix, initialise_tracking
from ..utils.utils import prefix_print, Timers
from dataclasses import dataclass
from ..modules.auto_easydel_model import AutoEasyDeLModelForCausalLM, AutoEasyDeLConfig
from ..modules.easydel_modelling_utils import EasyDeLFlaxPretrainedModel, EasyDeLPretrainedConfig
from optax import GradientTransformation, Schedule
from fjformer import CheckpointManager
from jax.sharding import Mesh

from ..etils.etils import get_logger

logger = get_logger(__name__)


@dataclass
class TrainerConfigureDataloaderFuncOutput:
    dataloader_train: Iterator[ndarray[Any, Any]]
    max_training_steps: int
    dataloader_eval: Optional[Iterator[ndarray[Any, Any]]] = None
    max_evaluation_steps: Optional[int] = None


@dataclass
class TrainerConfigureModelFuncOutput:
    model: EasyDeLFlaxPretrainedModel
    tx: GradientTransformation
    scheduler: Schedule
    config: Optional[EasyDeLPretrainedConfig] = None


@dataclass
class TrainerConfigureFunctionFuncOutput:
    create_sharded_state_from_params_function: Callable
    sharded_train_step_function: Callable
    mesh: Mesh
    checkpoint_manager: CheckpointManager
    initialize_state_function: Callable
    sharded_eval_step_function: Optional[Callable] = None


class BaseTrainer:
    def __init__(
            self,
            arguments: TrainArguments,
            dataset_train: Dataset,
            dataset_eval: Dataset = None,
            finetune: bool = True,
            checkpoint_path: Union[str, os.PathLike] = None,
            _do_init_fns: bool = True
    ):
        """
        The __init__ function is called when the class is instantiated.
        It sets up all the variables that are needed for training, including:
        - The timer to keep track of how long each epoch takes.
        - The dataloaders for both training and evaluation (if provided).
        - The model itself, which will be created from a checkpoint if one was provided.  Otherwise,
         it will be created from scratch using the arguments passed in by the user.
         Note that this function also handles creating a mesh if one was not already specified in arguments
         or loaded from a checkpoint file (see below).
          This means that you can pass in either

        :param self: Represent the instance of the class
        :param arguments: TrainArguments: Pass the arguments to the trainer
        :param dataset_train: Dataset: Pass the training dataset to the trainer
        :param dataset_eval: Dataset: Pass the validation dataset
        :param finetune: bool: Load the model from a checkpoint
        :param checkpoint_path: Union[str,os.PathLike] : Load the checkpoint path
        :param _do_init_fns: bool: Initialize the functions
        :return: Nothing, it just initializes the class

        """
        # Loggers
        self.timer = getattr(self, "timer", None)
        self.wandb_runtime: Run | RunDisabled | None = getattr(self, "wandb_runtime", None)

        # Data
        self.dataloader_train = getattr(self, "dataloader_train", None)
        self.dataloader_eval = getattr(self, "dataloader_eval", None)
        self.max_training_steps = getattr(self, "max_training_steps", None)
        self.max_evaluation_steps = getattr(self, "max_evaluation_steps", None)
        self.dataset_train = dataset_train
        self.dataset_eval = dataset_eval

        # Model Related
        self.model = getattr(self, "model", None)
        self.config = getattr(self, "config", None)
        self.scheduler = getattr(self, "scheduler", None)
        self.tx = getattr(self, "tx", None)
        self.model_state = getattr(self, "model_state", None)

        # LoRA Related
        self.rapture = arguments.rapture
        self.lora_parameters = getattr(self, "lora_parameters", None)
        self.lora_model = getattr(self, "lora_model", None)
        self.lora_tx = getattr(self, "lora_tx", None)
        self.lora_opt_state = getattr(self, "lora_opt_state", None)
        self.lora_apply_fn = getattr(self, "lora_apply_fn", None)

        # PJit functions
        self.create_sharded_state_from_params_function = getattr(
            self,
            "create_sharded_state_from_params_function",
            None
        )
        self.sharded_train_step_function = getattr(self, "sharded_train_step_function", None)
        self.sharded_eval_step_function = getattr(self, "sharded_eval_step_function", None)
        self.initialize_state_function = getattr(self, "initialize_state_function", None)
        self.mesh = getattr(self, "mesh", None)

        # Checkpoint Managers
        self.checkpoint_manager: fjformer.CheckpointManager | None = getattr(self, "checkpoint_manager", None)

        # EasyState
        self.state_shape = getattr(self, "state_shape", None)
        self.state_partition_spec = getattr(self, "state_partition_spec", None)
        self.sharded_state = getattr(self, "sharded_state", None)

        # Rest

        self.arguments = arguments
        self.finetune = finetune
        self.checkpoint_path = checkpoint_path
        self.dtype = arguments.dtype
        self.param_dtype = arguments.param_dtype
        if self.arguments.track_memory:
            if not self.arguments.performance_mode:
                initialise_tracking()
                self.arguments._stop_capturing_memory = False
                self._start_capturing_memory().start()
        if finetune:
            if checkpoint_path is None:
                prefix_print(
                    "Warning",
                    "In case of using `finetune = True` and Passing `checkpoint_path = None`"
                    " you should pass parameters in train function"
                )
        if _do_init_fns:
            self.initialize_trainer_utils()
        else:
            prefix_print(
                "Warning",
                "you have set `_do_init_fns = False` so function will not me initialized you have "
                f"to do in manually (simply with `trainer.initialize_trainer_utils()` )"
            )

    def __str__(self):
        string = f"{self.__class__.__name__}("
        for key, value in self.__dict__.items():
            try:
                string += value.__str__().replace("\n", "\n\t")
            except TypeError:
                ...
        string += ")"
        return string

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def finish():
        """
        The finish function is called when the experiment ends.
        It can be used to save data, upload files, or do any other cleanup tasks.

        :return: A dictionary of the run's metadata

        """
        wandb.finish()

    def _start_capturing_memory(self, dir_prefix: str = "/dev/shm" if sys.platform != "win32" else "."):
        def _start():
            while True:
                information_queries = {}
                for key in ["Used", "Usage Percent"]:
                    for device, info in get_capacity_matrix(dir_prefix=dir_prefix).items():
                        information_queries[f"accelerators/{device.replace('_', ' ')} ({key})"] = float(
                            info[key].replace("%", "").replace("GB", "")
                        )
                self.arguments._captured_memory = information_queries
                if self.arguments.stop_capturing_memory:
                    break
                time.sleep(1.5)

        return threading.Thread(target=_start)

    def initialize_trainer_utils(self):
        """
        The initialize_trainer_utils function is responsible for initializing the following:
            - wandb_runtime (if you use_wandb is True)
            - timer object (for logging time taken by various functions)
            - dataloader objects for training and evaluation data, along with max steps per epoch.
              The configure_dataloader function accomplishes this task.

        :param self: Represent the instance of the class
        :return: A tuple of functions

        """
        self.wandb_runtime = None
        if self.arguments.use_wandb:
            self.wandb_runtime = self.arguments.get_wandb_init()
        self.timer = Timers(
            use_wandb=False,
            tensorboard_writer=self.arguments.get_board()
        )

        self.timer("configure dataloaders").start()
        dataset_configurations = self.configure_dataloader()
        self.dataloader_train = dataset_configurations.dataloader_train
        self.max_training_steps = dataset_configurations.max_training_steps
        self.dataloader_eval = dataset_configurations.dataloader_eval
        self.max_evaluation_steps = dataset_configurations.max_evaluation_steps

        self.timer("configure dataloaders").stop()

        self.timer.log(["configure dataloaders"])

        self.timer("configure Model, Optimizer, Scheduler and Config").start()
        model_configurations = self.configure_model()
        model = model_configurations.model
        tx = model_configurations.tx
        scheduler = model_configurations.scheduler
        config = model_configurations.config
        self.model = model
        self.tx = tx
        self.scheduler = scheduler
        self.config = config
        if self.rapture is not None:
            lora_modules = self.rapture.apply_lora(
                module=model,
                parameters=self.arguments.rapture_config.parameters,
                tx=tx,
            )
            self.lora_parameters = lora_modules.lora_parameters
            self.lora_apply_fn = lora_modules.lora_module.__call__
            self.lora_opt_state = lora_modules.lora_opt_state
            self.lora_model = lora_modules.lora_module
            self.lora_tx = lora_modules.lora_tx

        self.timer("configure Model, Optimizer, Scheduler and Config").stop()
        self.timer.log(["configure Model, Optimizer, Scheduler and Config"])
        self.timer("configure functions and sharding them").start()
        function_configurations = self.configure_functions()
        self.create_sharded_state_from_params_function = \
            function_configurations.create_sharded_state_from_params_function
        self.sharded_train_step_function = function_configurations.sharded_train_step_function
        self.sharded_eval_step_function = function_configurations.sharded_eval_step_function
        self.mesh = function_configurations.mesh
        self.checkpoint_manager = function_configurations.checkpoint_manager
        self.initialize_state_function = function_configurations.initialize_state_function
        self.timer("configure functions and sharding them").stop()
        self.timer.log(["configure functions and sharding them"])

    @abstractmethod
    def create_collate_function(
            self,
            max_sequence_length: int,
            truncation_mode: Literal["keep_end", "keep_start"]
    ) -> Callable:
        raise NotImplementedError

    @abc.abstractmethod
    def configure_functions(self) -> TrainerConfigureFunctionFuncOutput:
        """
        The configure_functions function is responsible for configuring the functions that will be used in training.
        It does this by first defining a function called function_configurations, which initializes the model parameters and returns
        them as a EasyDeLState object. The EasyDeLState object contains all the information needed to train or evaluate
        on a batch of data, including:
        :param self: Access the class attributes
        :return: A TrainerConfigureFunctionFuncOutput object

        """
        raise NotImplementedError

    def configure_dataloader(self) -> TrainerConfigureDataloaderFuncOutput:
        """
        The configure_dataloader function is used to configure the dataloader for training and evaluation.

        :param self: Refer to the class instance itself
        :return: A TrainerConfigureDataloaderFuncOutput object

        """

        def create_tf_dataset(dataset: Dataset, is_train: bool) -> Iterator[ndarray[Any, Any]]:
            return (
                dataset.to_tf_dataset(
                    collate_fn=self.create_collate_function(
                        max_sequence_length=self.arguments.max_sequence_length,
                        truncation_mode=self.arguments.truncation_mode
                    ),
                    batch_size=self.arguments.total_batch_size,
                    drop_remainder=True,
                    shuffle=not is_train,
                    num_workers=self.arguments.dataloader_num_workers
                )
                .repeat(self.arguments.num_train_epochs if is_train else 1)
                .prefetch(tf.data.experimental.AUTOTUNE)
                .as_numpy_iterator()
            )

        def create_tf_dataset_from_iterable(dataset: IterableDataset, is_train: bool) -> Iterator[ndarray[Any, Any]]:
            return (
                tf.data.Dataset.from_generator(
                    lambda: dataset,
                    output_signature={
                        col: tf.TensorSpec(shape=(self.arguments.max_sequence_length,), dtype=tf.int32)
                        for col in next(iter(dataset)).keys()
                    }
                )
                .repeat(self.arguments.num_train_epochs if is_train else 1)
                .batch(self.arguments.total_batch_size, drop_remainder=False)
                .prefetch(tf.data.experimental.AUTOTUNE)
                .as_numpy_iterator()
            )

        def calculate_steps(dataset: Union[Dataset, IterableDataset], is_train: bool):
            """
            Return total number of steps to train or evaluate on.
            """
            if hasattr(dataset, "__len__"):
                num_steps = len(dataset) * (self.arguments.num_train_epochs if is_train else 1)
                max_steps = self.arguments.max_training_steps if is_train else self.arguments.max_evaluation_steps
                return min(num_steps, max_steps) if max_steps else num_steps
            else:
                num_steps = self.arguments.max_training_steps if is_train else self.arguments.max_evaluation_steps
                if not num_steps:
                    raise ValueError(
                        f"Specify the number of {'training' if is_train else 'evaluation'} steps for a generator/streaming dataset.")
                return num_steps

        def to_tf_dataloader(dataset: Union[Dataset, IterableDataset], is_train: bool):
            if hasattr(dataset, "__len__"):
                return create_tf_dataset(dataset, is_train)
            else:
                return create_tf_dataset_from_iterable(dataset, is_train)

        max_training_steps = calculate_steps(self.dataset_train, is_train=True)
        dataloader_train = to_tf_dataloader(self.dataset_train, is_train=True)

        if self.dataset_eval is not None and self.arguments.do_eval:
            max_evaluation_steps = calculate_steps(self.dataset_eval, is_train=False)
            dataloader_eval = to_tf_dataloader(self.dataset_eval, is_train=False)
        else:
            dataloader_eval, max_evaluation_steps = None, 0

        return TrainerConfigureDataloaderFuncOutput(
            dataloader_train=dataloader_train,
            max_training_steps=max_training_steps,
            dataloader_eval=dataloader_eval,
            max_evaluation_steps=max_evaluation_steps
        )

    def configure_model(self) -> TrainerConfigureModelFuncOutput:
        """
        The configure_model function is responsible for creating the model, optimizer and scheduler.

        :param self: Represent the instance of the class
        :return: A model, optimizer, scheduler and config  in TrainerConfigureModelFuncOutput Object

        """
        extra_configs = {} if self.arguments.extra_configs is None else self.arguments.extra_configs
        if self.arguments.model_class is not None:

            if not hasattr(self.arguments.configs_to_initialize_model_class["config"], "get_partition_rules"):
                assert self.arguments.custom_rule is not None, (
                    "if you are using custom model to init you must"
                    " pass custom_rule for partition rules "
                )

            self.arguments.configs_to_initialize_model_class["config"].axis_dims = self.arguments.sharding_array

            model = self.arguments.model_class(
                **self.arguments.configs_to_initialize_model_class,
                _do_init=False
            )

            config = self.arguments.configs_to_initialize_model_class["config"]

        else:
            extra_configs["gradient_checkpointing"] = self.arguments.gradient_checkpointing

            model = AutoEasyDeLModelForCausalLM.from_pretrained(
                self.arguments.model_huggingface_repo_id,
                dtype=self.arguments.dtype,
                param_dtype=self.arguments.param_dtype,
                _do_init=False
            )
            if hasattr(model, "config"):
                for k, v in extra_configs.items():
                    setattr(model.config, k, v)
                config = model.config
            else:
                config = None
                warnings.warn(
                    "Config is being set to None due to not detecting Model Configuration from taken Model "
                    "this will cause errors later."
                )
        tx, scheduler = self.arguments.get_optimizer_and_scheduler(self.max_training_steps)
        return TrainerConfigureModelFuncOutput(
            model=model,
            tx=tx,
            scheduler=scheduler,
            config=config
        )

    def _save_state(
            self,
            state: "EasyDeLState",
            gather_fns: Optional[Any | Mapping[str, Callable] | dict[Callable]],
            milestone: bool = False
    ) -> str:
        step = int(
            jax.device_get(
                state.step
            )
        ) + self.arguments.step_start_point if self.arguments.step_start_point is not None else int(
            jax.device_get(
                state.step
            )
        )
        checkpoint_name = f"{self.arguments.model_name}-S{step}"
        filename = f"{checkpoint_name}_{step}" if milestone else f"{checkpoint_name}"
        filename += ".easy"
        termcolor.cprint(f"Saving Model {filename}.", color="cyan", force_color=True)
        state.save_state(
            filename=filename,
            checkpoint_dir=os.path.join(self.arguments.save_dir, self.arguments.model_name),
            gather_fns=gather_fns,
            float_dtype=self.dtype,
            verbose=self.arguments.verbose,
            save_optimizer=self.arguments.save_optimizer_state,
        )
        return filename

    @abc.abstractmethod
    def train(self):
        """
        abstract of Train Function to train model
        """

    @abc.abstractmethod
    def eval(self, state):
        """
        abstract of Eval Function to evaluate model
        """
