# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
from __future__ import annotations

import os
import pprint
import shutil
import sys
import threading
import time
import typing as tp
import warnings
from abc import abstractmethod
from glob import glob
from logging import warning
from pathlib import Path

import flax
import flax.core
import jax
import numpy as np
import termcolor
import tqdm
from flax.core import unfreeze

from easydel.etils.easystate import EasyDeLState
from easydel.etils.errors import EasyDeLTimerError

try:
	import wandb  # noqa: F821 # type:ignore
except ImportError:
	wandb = None


from easydel import __version__
from easydel.etils.etils import get_logger
from easydel.infra.base_module import (
	EasyDeLBaseModule,
)
from easydel.smi import get_capacity_matrix, initialise_tracking
from easydel.utils import Timers

from .trainer_protocol import (
	BaseTrainerProtocol,
	TrainerConfigureDataloaderOutput,
	TrainerConfigureFunctionOutput,
	TrainerConfigureModelOutput,
	TrainerOutput,
)
from .training_configurations import TrainingArguments

if tp.TYPE_CHECKING:
	from datasets import Dataset, IterableDataset
else:
	Dataset = tp.Any
	IterableDataset = tp.Any

logger = get_logger(__name__)


class BaseTrainer(BaseTrainerProtocol):
	def __init__(
		self,
		arguments: tp.Optional[TrainingArguments] = None,
		model: tp.Optional[EasyDeLBaseModule] = None,
		dataset_train: tp.Optional[Dataset] = None,
		dataset_eval: tp.Optional[Dataset] = None,
		finetune: bool = True,
		checkpoint_path: tp.Optional[tp.Union[str, os.PathLike]] = None,
		_do_init_fns: bool = True,
	):
		assert arguments is not None, "training argument must be passed to Trainers."
		assert model is not None, "Model can not be None and it must be passed to Trainers."
		self.arguments = arguments
		self.dataset_train = dataset_train
		self.dataset_eval = dataset_eval
		self.finetune = finetune
		self.checkpoint_path = checkpoint_path
		self._initialize_attributes()
		self.model = model

		if _do_init_fns:
			self.initialize_trainer_utils()
		else:
			warnings.warn(
				"You have set `_do_init_fns = False`. Functions will not be initialized automatically. "
				"Call `trainer.initialize_trainer_utils()` manually.",
				stacklevel=2,
			)

		if self.arguments.track_memory:
			self._initialize_memory_tracking()

	def _initialize_attributes(self):
		# Initialize all attributes with default values
		self.timer = getattr(self, "timer", None)
		self.wandb_runtime = getattr(self, "wandb_runtime", None)
		self.dataloader_train = getattr(self, "dataloader_train", None)
		self.dataloader_eval = getattr(self, "dataloader_eval", None)
		self.max_training_steps = getattr(self, "max_training_steps", None)
		self.max_evaluation_steps = getattr(self, "max_evaluation_steps", None)
		self.model = getattr(self, "model", None)
		self.config = getattr(self, "config", None)
		self.scheduler = getattr(self, "scheduler", None)
		self.tx = getattr(self, "tx", None)
		self.model_state = getattr(self, "model_state", None)
		self.create_state_sharded = getattr(self, "create_state_sharded", None)
		self.sharded_training_step_function = getattr(
			self, "sharded_training_step_function", None
		)
		self.sharded_evaluation_step_function = getattr(
			self, "sharded_evaluation_step_function", None
		)
		self.mesh = getattr(self, "mesh", None)
		self.checkpoint_manager = getattr(self, "checkpoint_manager", None)
		self.state_shape = getattr(self, "state_shape", None)
		self.state_partition_spec = getattr(self, "state_partition_spec", None)
		self.state_named_sharding = getattr(self, "state_named_sharding", None)
		self.state = getattr(self, "state", None)
		self.pruning_module = getattr(self.arguments, "pruning_module", None)

	def _initialize_memory_tracking(self):
		if not self.arguments.performance_mode:
			initialise_tracking()
			self.arguments._stop_capturing_memory = False
			self._start_capturing_memory().start()

	def __str__(self):
		return pprint.pformat(self.__dict__, indent=2)

	__repr__ = __str__

	@staticmethod
	def finish():
		if wandb is not None:
			wandb.finish()

	def _start_capturing_memory(
		self,
		dir_prefix: str = "/dev/shm" if sys.platform != "win32" else ".",
	):
		def _start():
			try:
				while not self.arguments._stop_capturing_memory:
					information_queries = {
						f"accelerators/{device.replace('_', ' ')} ({key})": float(
							info[key].replace("%", "").replace("GB", "")
						)
						for key in ["Used", "Usage Percent"]
						for device, info in get_capacity_matrix(dir_prefix=dir_prefix).items()
					}
					self.arguments._captured_memory = information_queries
					time.sleep(1.5)
			except FileNotFoundError as err:
				if "directory: 'go'" in err.__str__():
					warning(
						"in order to capture memory you need to have `go-lang` already installed.(ignoring memory capture action)"
					)
				else:
					raise FileNotFoundError(err) from err

		return threading.Thread(target=_start)

	def initialize_trainer_utils(self):
		"""
		Initializes various utilities used by the trainer.

		This includes setting up Weights & Biases, initializing the training timer,
		configuring dataloaders, configuring the model and optimizer, sharding the
		model and reference model states, and configuring the training and evaluation functions.
		"""
		self._initialize_wandb()
		self._initialize_timer()
		self._configure_dataloaders()
		self._configure_model()
		self._configure_functions()
		self._configure_state()

	def _initialize_wandb(self):
		if self.arguments.use_wandb:
			self.wandb_runtime = self.arguments.get_wandb_init()

	def _initialize_timer(self):
		self.timer = Timers(
			use_wandb=False,
			tensorboard_writer=self.arguments.get_tensorboard,
		)

	def _configure_dataloaders(self):
		"""
		Configures the dataloaders for training and evaluation.

		This method retrieves the dataloaders from the `configure_dataloaders` method,
		sets the maximum training and evaluation steps, and logs the time taken for
		this configuration.
		"""
		with self.timer("configure dataloaders"):
			dataset_configurations = self.configure_dataloaders()
			self.dataloader_train = dataset_configurations.dataloader_train
			self.max_training_steps = dataset_configurations.max_training_steps
			self.dataloader_eval = dataset_configurations.dataloader_eval
			self.max_evaluation_steps = dataset_configurations.max_evaluation_steps
		self.timer.log("configure dataloaders")

	def _configure_model(self):
		"""
		Configures the model, optimizer, scheduler, and configuration.

		This method retrieves the model, optimizer, scheduler, and configuration from
		the `configure_model` method and configures LoRA (if enabled). It also logs
		the time taken for this configuration.
		"""
		with self.timer("configure Model, Optimizer, Scheduler and Config"):
			model_configurations = self.configure_model()
			self.model = model_configurations.model
			self.tx = model_configurations.tx
			self.scheduler = model_configurations.scheduler
			self.config = model_configurations.config

		self.timer.log("configure Model, Optimizer, Scheduler and Config")

	def _configure_functions(self):
		"""
		Configures and JIT-compiles the training and evaluation step functions.

		This method retrieves the configured functions from the `configure_functions`
		method, sets up the mesh, checkpoint manager, and state initialization
		function, and logs the time taken for this configuration.
		"""
		with self.timer("configure functions and sharding them"):
			function_configurations = self.configure_functions()
			self.create_state_sharded = function_configurations.create_state_sharded
			self.sharded_training_step_function = (
				function_configurations.sharded_training_step_function
			)
			self.sharded_evaluation_step_function = (
				function_configurations.sharded_evaluation_step_function
			)
			self.mesh = function_configurations.mesh
			self.checkpoint_manager = function_configurations.checkpoint_manager
		self.timer.log("configure functions and sharding them")

	def _configure_state(self):
		"""Configures and JIT-compiles the sharded state"""
		with self.timer("configure sharded state"):
			self.state = self.create_state_sharded()
		self.timer.log("configure sharded state")

	@abstractmethod
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
		raise NotImplementedError

	@abstractmethod
	def configure_functions(self) -> TrainerConfigureFunctionOutput:
		"""
		Configures and JIT-compiles the training and evaluation step functions.

		This method sets up the necessary functions for training and evaluation, including:
				- Initialization of the model state.
				- Sharding of the model parameters and optimizer state.
				- JIT-compilation of the training and evaluation step functions.

		Returns:
				TrainerConfigureFunctionOutput: An object containing the configured functions and other relevant information.
		"""
		raise NotImplementedError

	def configure_dataloaders(self) -> TrainerConfigureDataloaderOutput:
		"""
		Configures the dataloaders for training and evaluation.

		This method creates the training and evaluation dataloaders using the provided
		datasets and data collator. It also determines the maximum number of training
		and evaluation steps based on the dataset sizes and training arguments.

		Returns:
				TrainerConfigureDataloaderOutput: An object containing the configured dataloaders and the
																				maximum number of training and evaluation steps.
		"""

		def create_tf_dataset(
			dataset: Dataset,
			is_train: bool,
		) -> tp.Iterator[np.ndarray]:
			"""
			Creates a TensorFlow dataset from a Hugging Face Dataset.

			Args:
					dataset (Dataset): The Hugging Face Dataset.
					is_train (bool): Whether the dataset is for training.

			Returns:
					tp.Iterator[np.ndarray]: The TensorFlow dataset iterator.
			"""
			import tensorflow as tf

			return (
				dataset.to_tf_dataset(
					collate_fn=self.create_collect_function(
						max_sequence_length=self.arguments.max_sequence_length,
						truncation_mode=self.arguments.truncation_mode,
					),
					batch_size=self.arguments.total_batch_size
					* self.arguments.gradient_accumulation_steps,
					drop_remainder=True,
					shuffle=is_train,
					num_workers=self.arguments.dataloader_num_workers,
				)
				.repeat(self.arguments.num_train_epochs if is_train else 1)
				.prefetch(tf.data.AUTOTUNE)
				.as_numpy_iterator()
			)

		def create_tf_dataset_from_iterable(
			dataset: IterableDataset,
			is_train: bool,
		) -> tp.Iterator[np.ndarray]:
			"""
			Creates a TensorFlow dataset from an iterable Hugging Face Dataset.

			Args:
					dataset (IterableDataset): The iterable Hugging Face Dataset.
					is_train (bool): Whether the dataset is for training.

			Returns:
					tp.Iterator[np.ndarray]: The TensorFlow dataset iterator.
			"""
			import tensorflow as tf

			return (
				tf.data.Dataset.from_generator(
					lambda: dataset,
					output_signature={
						col: tf.TensorSpec(
							shape=(self.arguments.max_sequence_length,), dtype=tf.int32
						)
						for col in next(iter(dataset)).keys()
					},
				)
				.repeat(self.arguments.num_train_epochs if is_train else 1)
				.batch(
					self.arguments.total_batch_size * self.arguments.gradient_accumulation_steps,
					drop_remainder=False,
				)
				.prefetch(tf.data.AUTOTUNE)
				.as_numpy_iterator()
			)

		def calculate_steps(
			dataset: tp.Union[Dataset, IterableDataset],
			is_train: bool,
		) -> int:
			"""
			Calculates the number of training or evaluation steps based on dataset length and arguments.

			Args:
					dataset (tp.Union[Dataset, IterableDataset]): The dataset to calculate steps for.
					is_train (bool): Whether the dataset is for training.

			Returns:
					int: The number of steps.

			Raises:
					ValueError: If the dataset is a generator/streaming dataset and the number of steps is not specified.
			"""
			if hasattr(dataset, "__len__"):
				total_data_len = len(dataset)
				batch_size = (
					self.arguments.total_batch_size
					if is_train
					else self.arguments.eval_batch_size
				)
				num_steps = (
					(total_data_len + batch_size - 1)
					// batch_size
					* (self.arguments.num_train_epochs if is_train else 1)
				)
				max_steps = (
					self.arguments.max_training_steps
					if is_train
					else self.arguments.max_evaluation_steps
				)
				steps = min(num_steps, max_steps) if max_steps else num_steps
			else:
				num_steps = (
					self.arguments.max_training_steps
					if is_train
					else self.arguments.max_evaluation_steps
				)
				if not num_steps:
					raise ValueError(
						f"Specify the number of {'training' if is_train else 'evaluation'} steps for a generator/streaming dataset."
					)
				steps = num_steps
			return steps // self.arguments.gradient_accumulation_steps

		def to_tf_dataloader(
			dataset: tp.Union[Dataset, IterableDataset],
			is_train: bool,
		) -> tp.Iterator[np.ndarray]:
			"""
			Converts a Hugging Face Dataset to a TensorFlow dataloader.

			Args:
					dataset (tp.Union[Dataset, IterableDataset]): The Hugging Face Dataset.
					is_train (bool): Whether the dataset is for training.

			Returns:
					tp.Iterator[np.ndarray]: The TensorFlow dataloader iterator.
			"""
			if hasattr(dataset, "__len__"):
				return create_tf_dataset(dataset, is_train)
			else:
				return create_tf_dataset_from_iterable(dataset, is_train)

		max_training_steps = calculate_steps(self.dataset_train, is_train=True)
		print(max_training_steps)
		dataloader_train = to_tf_dataloader(self.dataset_train, is_train=True)

		if self.dataset_eval is not None and self.arguments.do_eval:
			max_evaluation_steps = calculate_steps(self.dataset_eval, is_train=False)
			dataloader_eval = to_tf_dataloader(self.dataset_eval, is_train=False)
		else:
			dataloader_eval, max_evaluation_steps = None, 0

		return TrainerConfigureDataloaderOutput(
			dataloader_train=dataloader_train,
			max_training_steps=max_training_steps,
			dataloader_eval=dataloader_eval,
			max_evaluation_steps=max_evaluation_steps,
		)

	def configure_model(self) -> TrainerConfigureModelOutput:
		"""
		Configures the model, optimizer, scheduler, and configuration.

		This method retrieves the model configuration from the model state, creates
		the optimizer and scheduler using the training arguments, and returns an
		object containing the configured model, optimizer, scheduler, and configuration.

		Returns:
				TrainerConfigureModelOutput: An object containing the configured model, optimizer, scheduler, and configuration.
		"""

		tx, scheduler = self.arguments.get_optimizer_and_scheduler(self.max_training_steps)
		if self.pruning_module is not None:
			tx = self.pruning_module.wrap_optax(tx)
		return TrainerConfigureModelOutput(
			model=self.model,
			tx=tx,
			scheduler=scheduler,
			config=self.model.config,
		)

	def _save_state(self, state: EasyDeLState, *args, **kwargs) -> str:
		step = self._get_current_step(state)
		self._manage_checkpoint_limit(self.arguments._get_save_directory())
		directory_name = self.arguments._get_save_directory_milestone(
			step=step,
			create=True,
		)

		logger.info(f"saving state {directory_name}.")

		state.save_state(
			save_directory=directory_name,
			float_dtype=self.model.dtype,
			verbose=self.arguments.verbose,
			save_optimizer=self.arguments.save_optimizer_state,
		)

		self._save_readme(directory_name)
		return str(directory_name)

	def _get_current_step(self, state):
		step = int(jax.device_get(state.step))
		if self.arguments.step_start_point is not None:
			step += self.arguments.step_start_point
		return step

	def _manage_checkpoint_limit(self, save_directory):
		if self.arguments.save_total_limit:
			checkpoint_files = glob(os.path.join(save_directory, "run-*"))
			checkpoint_files.sort(key=os.path.getmtime)
			for old_save_directory in checkpoint_files[: -self.arguments.save_total_limit]:
				shutil.rmtree(old_save_directory, ignore_errors=True)
				logger.info(f"Removed old directory: {old_save_directory}")

	def _save_readme(self, save_directory):
		with open(os.path.join(save_directory, "README.md"), "w") as f:
			f.write(self._get_information())

	def _format_partition_rules(self) -> str:
		"""Format partition rules with proper indentation and formatting."""
		try:
			return pprint.pformat(self.model.config.get_partition_rules(), indent=2, width=80)
		except Exception as e:
			logger.error(f"Error formatting partition rules: {str(e)}")
			return "Error retrieving partition rules"

	def _get_device_info(self) -> dict:
		"""Get information about available devices."""
		try:
			devices = jax.devices()
			return {"platform": devices[0].platform.upper(), "device_count": len(devices)}
		except Exception as e:
			logger.error(f"Error getting device info: {str(e)}")
			return {"platform": "UNKNOWN", "device_count": 0}

	def _get_information(self) -> str:
		"""
		Generate formatted information about the model and training setup.

		Returns:
				str: Formatted markdown string containing model and training information
		"""
		device_info = self._get_device_info()
		partition_rules = self._format_partition_rules()

		return f"""
# {self.arguments.model_name}

## 🚀 Trained With [EasyDeL](https://github.com/erfanzar/EasyDeL)

EasyDeL is an open-source framework designed to enhance and streamline the training process of machine learning
models. With a primary focus on Jax, EasyDeL aims to provide convenient and effective solutions for 
training Flax/Jax models on TPU/GPU, for both serving and training purposes.

## 📦 Installation & Usage
 
```python
from easydel import AutoEasyDeLModelForCausalLM
from jax import numpy as jnp, lax

model = AutoEasyDeLModelForCausalLM.from_pretrained(
    f"REPO_ID/{self.arguments.model_name}",
    dtype=...,
    param_dtype=...,
    precision=lax.Precision("fastest"),
    auto_shard_model=True,
)
```

## 🔧 Training Configuration

### Model Details
- **Architecture**: {self.config.model_type}
- **Platform**: {device_info['platform']}
- **Number of Devices**: {device_info['device_count']}

### Training Parameters
- **Learning Rate**: {self.arguments.learning_rate} → {self.arguments.learning_rate_end}
- **Optimizer**: {self.arguments.optimizer}
- **Scheduler**: {self.arguments.scheduler}
- **Warmup Steps**: {self.arguments.warmup_steps}
- **Weight Decay**: {self.arguments.weight_decay}
- **Loss Config**: {self.arguments.loss_config}

### Training Setup
- **Epochs**: {self.arguments.num_train_epochs}
- **Batch Size**: {self.arguments.total_batch_size}
- **Sequence Length**: {self.arguments.max_sequence_length} 
- **Dtype**: {self.model.dtype}
- **Params Dtype**: {self.model.param_dtype}

### Advanced Configuration
- **Gradient Checkpointing**: {self.model.config.gradient_checkpointing}  
- **Gradient Accumulation Steps**: {self.arguments.gradient_accumulation_steps}
- **Max Training Steps**: {self.arguments.max_training_steps}
- **Max Evaluation Steps**: {self.arguments.max_evaluation_steps}
- **Training Duration**: {self.arguments.training_time}

### Sharding Configuration
```python
# Partition Rules
{partition_rules}
```

---
*Generated with EasyDeL v{__version__}*
"""

	def save_information(self, output_path: tp.Union[str, Path]) -> None:
		"""
		Save the generated information to a markdown file.

		Args:
				output_path: Path where the markdown file should be saved
		"""
		try:
			output_path = Path(output_path)
			output_path.parent.mkdir(parents=True, exist_ok=True)

			info = self._get_information()
			with open(output_path, "w", encoding="utf-8") as f:
				f.write(info)

			logger.info(f"Information saved successfully to {output_path}")
		except Exception as e:
			logger.error(f"Error saving information: {str(e)}")
			raise

	def save_pretrained(
		self,
		state: EasyDeLState,
		save_directory: tp.Optional[str] = None,
		gather_fns: tp.Optional[
			tp.Any | tp.Mapping[str, tp.Callable] | dict[tp.Callable]
		] = None,
		to_torch: bool = False,
		base_hf_auto_class=None,
		easystate_to_huggingface_model_kwargs: tp.Optional[dict] = None,
		torch_save_pretrained_kwargs: tp.Optional[dict] = None,
	):
		save_directory = save_directory or os.path.join(
			self.arguments.save_directory, self.arguments.model_name
		)

		if base_hf_auto_class is None:
			from transformers import AutoModelForCausalLM as base_hf_auto_class
		if to_torch:
			return self._save_to_torch(
				state,
				save_directory,
				base_hf_auto_class,
				easystate_to_huggingface_model_kwargs,
				torch_save_pretrained_kwargs,
			)
		else:
			return self._save_state(
				state=state, gather_fns=gather_fns, save_directory=save_directory
			)

	def _save_to_torch(
		self,
		state,
		save_directory,
		base_hf_auto_class,
		easystate_to_huggingface_model_kwargs,
		torch_save_pretrained_kwargs,
	):
		from easydel.utils.parameters_transformation import (
			easystate_to_huggingface_model,
		)

		easystate_to_huggingface_model_kwargs = easystate_to_huggingface_model_kwargs or {}
		torch_save_pretrained_kwargs = torch_save_pretrained_kwargs or {}

		model_config = state.module_config or state.module.config_class
		model_type = model_config.model_type
		model_class = base_hf_auto_class._model_mapping[type(model_config)]

		hf_model_config = self._create_hf_model_config(state, model_config, model_type)

		hf_model = easystate_to_huggingface_model(
			state=state,
			base_huggingface_module=model_class,
			config=hf_model_config,
			**easystate_to_huggingface_model_kwargs,
		)

		self._save_readme(save_directory)
		hf_model.save_pretrained(save_directory, **torch_save_pretrained_kwargs)
		return hf_model

	def _create_hf_model_config(
		self,
		state: EasyDeLState,
		model_config,
		model_type,
	):
		from transformers import AutoConfig

		hf_model_config = AutoConfig.for_model(model_type=model_type)
		unsafe_dict = state.unsafe_dict(model_config.__dict__)
		blocked_statics = ["torch_dtype"]

		for k, v in unsafe_dict.items():
			if (
				not k.startswith("_")
				and k in hf_model_config.__dict__
				and k not in blocked_statics
			):
				if isinstance(v, str) and v.isnumeric():
					v = int(float(v)) if float(v).is_integer() else float(v)
				setattr(hf_model_config, k, v)

		return hf_model_config

	def specs_to_name_sharding(self, tree, mesh=None):
		mesh = mesh or self.mesh or self.model.mesh
		return jax.tree_util.tree_map(
			lambda spec: jax.sharding.NamedSharding(spec=spec, mesh=mesh),
			tree,
		)

	def calculate_number_total_flops_per_device(self, params):
		return (
			6
			* sum(x.size for x in jax.tree_util.tree_flatten(unfreeze(params))[0])
			* (self.arguments.total_batch_size * self.arguments.max_sequence_length)
		) / jax.device_count()

	@staticmethod
	def count_model_parameters(prm):
		"""Prints the number of model parameters in billions."""
		return sum(n.size for n in jax.tree_util.tree_flatten(flax.core.unfreeze(prm))[0])

	def _should_skip_step(self, current_step):
		"""Determine if current step should be skipped."""
		return (
			self.arguments.step_start_point is not None
			and self.arguments.step_start_point > current_step
		)

	def _should_save_checkpoint(self, current_step):
		"""Determine if checkpoint should be saved at current step."""
		return (
			self.arguments.save_steps is not None
			and current_step > 0
			and (current_step % self.arguments.save_steps) == 0
		)

	def _should_run_evaluation(self, current_step):
		"""Determine if evaluation process should be runned current step."""
		return (
			self.arguments.evaluation_steps is not None
			and current_step > 0
			and (current_step % self.arguments.evaluation_steps) == 0
		)

	def _prepare_training_output(
		self,
		state: EasyDeLState,
		run_exception: tp.Optional[Exception] = None,
	):
		if run_exception is not None:
			if isinstance(run_exception, KeyboardInterrupt):
				termcolor.cprint(
					"KeyboardInterrupt: Training interrupted. Saving current state...",
					color="yellow",
					force_color=True,
				)
			elif isinstance(run_exception, EasyDeLTimerError):
				termcolor.cprint(
					"Training reached maximum time limit. Saving current state...",
					color="yellow",
					force_color=True,
				)
			elif isinstance(run_exception, StopIteration):
				...  # simply just pass
			else:
				raise RuntimeError("EasyDeL Runtime dumped") from run_exception
		checkpoint_path = "SAVING_SKIPPED"
		filename = None

		# TODO: LoRA be added.
		# try:
		if self.arguments.do_last_save:
			filename = self._save_state(
				state=state,
				milestone=False,
				save_directory=self.arguments.save_directory,
			)
			if self.arguments.save_directory is not None:
				checkpoint_path = os.path.join(self.arguments.save_directory, filename)
		# except Exception as e:
		# 	termcolor.cprint(
		# 		f"Failed to save checkpoint on interruption: {str(e)}",
		# 		color="red",
		# 		force_color=True,
		# 	)

		return TrainerOutput(
			state=state,
			mesh=self.mesh,
			checkpoint_path=checkpoint_path,
			last_save_file_name=filename,
		)

	def _handle_training_interruption(
		self,
		state: EasyDeLState,
		exception: Exception,
		shard_fns: tp.Optional[tp.Any | tp.Mapping[str, tp.Callable] | dict[tp.Callable]],
		gather_fns: tp.Optional[tp.Any | tp.Mapping[str, tp.Callable] | dict[tp.Callable]],
	):
		"""Handle training interruption gracefully."""
		if isinstance(exception, KeyboardInterrupt):
			termcolor.cprint(
				"KeyboardInterrupt: Training interrupted. Saving current state...",
				color="yellow",
				force_color=True,
			)
		elif isinstance(exception, EasyDeLTimerError):
			termcolor.cprint(
				"Training reached maximum time limit. Saving current state...",
				color="yellow",
				force_color=True,
			)
		else:
			raise RuntimeError("EasyDeL Runtime dumped") from exception
		return self._prepare_training_output(
			state=state,
			checkpoint_manager=self.checkpoint_manager,
			shard_fns=shard_fns,
			gather_fns=gather_fns,
			run_exception=None,
		)

	def _setup_initial_metrics(self, state):
		"""Setup initial metrics logging."""
		# Calculate and log model size
		self.arguments.log_metrics(
			{
				"Number of Model Parameters (Billion)": self.count_model_parameters(
					state.graphstate
				)
			},
			step=0,
		)
		self._flops_per_device = (
			self.calculate_number_total_flops_per_device(params=state.graphstate) / 1e12
		)

	def _get_next_batch(self, train_iter):
		"""Get next batch from iterator, reinitializing if needed."""
		try:
			batch = next(train_iter)
		except StopIteration:
			train_iter = iter(self.dataloader_train)
			batch = next(train_iter)

		# Remove specified ids from batch if needed
		for id_to_pop in self.arguments.ids_to_pop_from_dataset:
			_ = batch.pop(id_to_pop, None)

		return batch

	def _log_metrics(
		self,
		metrics: tp.Dict[str, float],
		pbar: tqdm.tqdm,
		step: int,
		mode: str = "train",
	):
		"""Log metrics and update progress bar."""
		# Update progress bar
		pbar.set_postfix(
			**{k.replace(f"{mode}/", ""): v for k, v in metrics.items() if len(k) < 30}
		)
		pbar.update()
		# Log metrics if tracking is enabled
		if not self.arguments.performance_mode:
			self.arguments.log_metrics(
				metrics=metrics,
				step=step,
			)
