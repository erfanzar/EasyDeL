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
import time
import typing as tp
from abc import abstractmethod
from functools import cached_property
from glob import glob
from pathlib import Path

import contextlib2
import flax
import flax.core
import flax.nnx
import jax
import jax.extend
import numpy as np
import tqdm
from flax import nnx as nn
from flax.core import unfreeze
from jax._src.stages import Compiled

import easydel
from easydel.infra.base_state import EasyDeLState
from easydel.infra.errors import EasyDeLBreakRequest, EasyDeLTimerError
from easydel.infra.loss_utils import LossMetrics
from easydel.infra.utils import CompilationTracker
from easydel.utils.lazy_import import is_package_available
from easydel.utils.traversals import specs_to_name_sharding

try:
	import wandb  # noqa: F821 # type:ignore
except ImportError:
	wandb = None


from easydel import __version__
from easydel.infra.base_module import (
	EasyDeLBaseModule,
)
from easydel.utils import Timers
from easydel.utils.helpers import get_logger

from .trainer_protocol import (
	BaseProgressBar,
	BaseTrainerProtocol,
	JSONProgressBar,
	NullProgressBar,
	RichProgressBar,
	TqdmProgressBar,
	TrainerConfigureDataloaderOutput,
	TrainerConfigureFunctionOutput,
	TrainerConfigureModelOutput,
	TrainerOutput,
)
from .training_configurations import MetricsType, TrainingArguments

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
		model_state: tp.Optional[EasyDeLState] = None,
		model: tp.type[EasyDeLBaseModule] = None,
		dataset_train: tp.Optional[Dataset] = None,
		dataset_eval: tp.Optional[Dataset] = None,
		data_collator: tp.Optional[tp.Callable] = None,
		finetune: bool = True,
		checkpoint_path: tp.Optional[tp.Union[str, os.PathLike]] = None,
		**deprecated_kwargs,
	):
		assert arguments is not None, "training argument must be passed to Trainers."
		if model_state is not None and model is not None:
			raise ValueError("Either model or model_state should be passed, not both.")
		elif model_state is None and model is None:
			raise ValueError("Either model or model_state should be passed.")
		elif model_state is None:
			model_state = model.to_state()

		self.arguments = arguments
		self.model_state = model_state
		self._model = flax.nnx.eval_shape(lambda: self.model_state.model)
		self.dataset_train = dataset_train
		self.dataset_eval = dataset_eval
		self.data_collator = data_collator
		self.finetune = finetune
		self.checkpoint_path = checkpoint_path
		self._initialize_attributes()
		self.initialize_trainer_utils()

		if self.arguments.track_memory:
			self._initialize_memory_tracking()

	@property
	def model(self):
		return self._model

	@property
	def mesh(self):
		return self.model.mesh

	@mesh.setter
	def mesh(self, val):
		return val

	@property
	def training_batch_size(self):
		return self.arguments.total_batch_size * self.arguments.gradient_accumulation_steps

	@cached_property
	def is_process_zero(self):
		return self.arguments.is_process_zero

	@property
	def evaluation_batch_size(self):
		return self.arguments.eval_batch_size

	def _initialize_attributes(self):
		# Initialize all attributes with default values
		self.timer = getattr(self, "timer", None)
		self.wandb_runtime = getattr(self, "wandb_runtime", None)
		self.dataloader_train = getattr(self, "dataloader_train", None)
		self.dataloader_eval = getattr(self, "dataloader_eval", None)
		self.max_training_steps = getattr(self, "max_training_steps", None)
		self.max_evaluation_steps = getattr(self, "max_evaluation_steps", None)

		self.scheduler = getattr(self, "scheduler", None)
		self.tx = getattr(self, "tx", None)

		self.checkpoint_manager = getattr(self, "checkpoint_manager", None)  #
		self.pruning_module = getattr(self.arguments, "pruning_module", None)
		self.memory_monitor = getattr(self.arguments, "memory_monitor", None)

		self._model = getattr(self, "_model", None)
		self.config = getattr(self, "config", None)

		self.state_shardings = getattr(self, "state_shardings", None)
		self.model_state = getattr(self, "model_state", None)

		self._training_time_start = getattr(self, "_training_time_start", None)
		self._evaluation_time_start = getattr(self, "_evaluation_time_start", None)

		self.sharded_training_step_function = getattr(
			self,
			"sharded_training_step_function",
			None,
		)
		self.sharded_evaluation_step_function = getattr(
			self,
			"sharded_evaluation_step_function",
			None,
		)

		self.train_tracker = getattr(self, "train_tracker", CompilationTracker())
		self.evalu_tracker = getattr(self, "evalu_tracker", CompilationTracker())

	def _initialize_memory_tracking(self):
		if not self.arguments.performance_mode:
			self.memory_monitor = easydel.utils.analyze_memory.SMPMemoryMonitor(1)

	def __repr__(self):
		return pprint.pformat(self.__dict__, indent=2)

	__str__ = __repr__

	@staticmethod
	def finish():
		if wandb is not None:
			try:
				wandb.finish()
			except Exception:
				...

	def on_step_start(
		self,
		state: EasyDeLState,
		step: int,
	) -> EasyDeLState:
		"""hook process to call in start of the step."""
		return state

	def on_step_end(
		self,
		state: EasyDeLState,
		metrics: MetricsType,
		step: int,
	) -> tp.Tuple[EasyDeLState, MetricsType]:
		"""hook process to call in start of the step."""
		return state, metrics

	def _preprocess_batch_input(
		self,
		state: EasyDeLState,
		batch: tp.Dict[str, jax.Array],
		is_train: bool,
	) -> tp.Tuple[tp.Dict[str, jax.Array], tp.Dict[str, tp.Union[float, int, str]]]:
		return batch, {}

	def get_runstage_flops(self, is_training) -> tp.Union[float, tp.Tuple[float, bool]]:
		try:
			function = (
				self.sharded_training_step_function
				if is_training
				else self.sharded_evaluation_step_function
			)
			flops = function.cost_analysis()[0]["flops"]
		except Exception:
			flops = (
				self.train_tracker.cached_flops
				if is_training
				else self.evalu_tracker.cached_flops
			)
		return flops

	def _ensure_functions_compiled(self):
		self.compile_aot()

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
		self._configure_state()
		self._configure_functions()

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
			manager = (
				jax.default_device(jax.devices(self.arguments.offload_device))
				if self.arguments.offload_dataset
				else contextlib2.nullcontext()
			)
			with manager:
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
			self._model = model_configurations.model
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
			functions = self.configure_functions()
			self.sharded_training_step_function = functions.sharded_training_step_function
			self.sharded_evaluation_step_function = functions.sharded_evaluation_step_function
			self.mesh = functions.mesh
			self.checkpoint_manager = functions.checkpoint_manager
		self.timer.log("configure functions and sharding them")

	def _configure_state(self):
		"""Configures and JIT-compiles the sharded state"""
		with self.timer("configure sharded state"):
			from eformer.escale import match_partition_rules

			with self.model.mesh:
				if self.arguments.init_tx:
					self.model_state = self.model_state.init_tx(self.tx)

				shape = nn.eval_shape(lambda: self.model_state)
				rules = self.model.config.get_partition_rules()
				state_shardings = specs_to_name_sharding(match_partition_rules(rules, shape))
				self.state_shardings = state_shardings
				self.model_state = self.model_state.shard_with_shape(state_shardings)

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
			if not is_package_available("tensorflow"):
				raise ImportError(
					"Please install `tensorflow` to use the `tensorflow-datasets` conversion."
				)
			import tensorflow as tf  # type:ignore

			batch_size = self.training_batch_size if is_train else self.evaluation_batch_size

			return (
				dataset.to_tf_dataset(
					collate_fn=self.create_collect_function(
						max_sequence_length=self.arguments.max_sequence_length,
						truncation_mode=self.arguments.truncation_mode,
					),
					batch_size=batch_size,
					drop_remainder=True,
					shuffle=is_train and self.arguments.shuffle_train_dataset,
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

			if not is_package_available("tensorflow"):
				raise ImportError(
					"Please install `tensorflow` to use the `tensorflow-datasets` conversion."
				)
			import tensorflow as tf  # type:ignore

			batch_size = self.training_batch_size if is_train else self.evaluation_batch_size
			tf_data_mapping = {
				"float16": tf.float16,
				"float32": tf.float32,
				"float64": tf.float64,
				"int16": tf.int16,
				"int32": tf.int32,
				"int64": tf.int64,
				"bool": tf.bool,
			}
			return (
				tf.data.Dataset.from_generator(
					lambda: dataset,
					output_signature={
						col: tf.TensorSpec(
							shape=vals.shape[1:]
							if len(vals.shape) > 1 and vals.shape[0] == 1  # auto remove batch dim
							else vals.shape,
							dtype=tf_data_mapping[str(vals.dtype)],
						)
						for col, vals in next(iter(dataset)).items()
						if hasattr(vals, "shape")
					},
				)
				.repeat(self.arguments.num_train_epochs if is_train else 1)
				.batch(batch_size, drop_remainder=False)
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
					self.arguments.total_batch_size if is_train else self.evaluation_batch_size
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
				steps = (
					self.arguments.max_training_steps
					if is_train
					else self.arguments.max_evaluation_steps
				)
				if not steps:
					raise ValueError(
						f"Specify the number of {'training' if is_train else 'evaluation'} steps for a generator/streaming dataset."
					)
			if is_train:
				steps = steps // self.arguments.gradient_accumulation_steps
			return steps

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
		enable = True
		if self.arguments.process_zero_is_admin and not self.arguments.is_process_zero:
			enable = False
		state.save_state(
			save_directory=directory_name,
			float_dtype=self.model.param_dtype,
			verbose=self.arguments.verbose,
			save_optimizer=self.arguments.save_optimizer_state,
			enable=enable,
		)

		self._save_readme(directory_name)
		return str(directory_name)

	def _get_current_step(self, state):
		step = int(jax.device_get(state.step))
		if self.arguments.step_start_point is not None:
			step += self.arguments.step_start_point
		return step

	def _manage_checkpoint_limit(self, save_directory):
		def _save():
			checkpoint_files = glob(os.path.join(save_directory, "run-*"))
			checkpoint_files.sort(key=os.path.getmtime)
			for old_save_directory in checkpoint_files[: -self.arguments.save_total_limit]:
				shutil.rmtree(old_save_directory, ignore_errors=True)
				logger.info(f"Removed old directory: {old_save_directory}")

		if self.arguments.save_total_limit:
			if self.arguments.process_zero_is_admin:
				if self.is_process_zero:
					_save()
			else:
				_save()

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
			return {
				"platform": jax.local_devices()[0].platform.upper(),
				"device_count": jax.device_count(),
			}
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
- **Platform**: {device_info["platform"]}
- **Number of Devices**: {device_info["device_count"]}

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
- **Dtype**: {str(self.model.dtype)}
- **Params Dtype**: {str(self.model.param_dtype)}

### Advanced Configuration
- **Gradient Checkpointing**: {self.model.config.gradient_checkpointing}  
- **Gradient Accumulation Steps**: {self.arguments.gradient_accumulation_steps}
- **Max Training Steps**: {self.arguments.max_training_steps}
- **Max Evaluation Steps**: {self.arguments.max_evaluation_steps}
- **Training Duration**: {self.arguments.training_time_limit}

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
		easystate_to_huggingface_model_kwargs: tp.Optional[dict] = None,
		torch_save_pretrained_kwargs: tp.Optional[dict] = None,
	):
		save_directory = save_directory or os.path.join(
			self.arguments.save_directory, self.arguments.model_name
		)

		if to_torch:
			return self._save_to_torch(
				state=state,
				save_directory=save_directory,
				easystate_to_huggingface_model_kwargs=easystate_to_huggingface_model_kwargs,
				torch_save_pretrained_kwargs=torch_save_pretrained_kwargs,
			)
		else:
			return self._save_state(
				state=state,
				gather_fns=gather_fns,
				save_directory=save_directory,
			)

	def _save_to_torch(
		self,
		state: EasyDeLState,
		save_directory: tp.Union[str, os.PathLike],
		easystate_to_huggingface_model_kwargs: tp.Optional[dict] = None,
		torch_save_pretrained_kwargs: tp.Optional[dict] = None,
	):
		easystate_to_huggingface_model_kwargs = easystate_to_huggingface_model_kwargs or {}
		torch_save_pretrained_kwargs = torch_save_pretrained_kwargs or {}
		hf_model = state.model.to_torch(**easystate_to_huggingface_model_kwargs)
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
		return specs_to_name_sharding(tree, mesh)

	def calculate_number_total_flops(self, params, is_training=True):
		return 6 * sum(x.size for x in jax.tree_util.tree_flatten(unfreeze(params))[0])

	@staticmethod
	def count_model_parameters(prm):
		"""Prints the number of model parameters in billions."""
		return sum(n.size for n in jax.tree_util.tree_flatten(flax.core.unfreeze(prm))[0])

	def apply_training_hooks(self, metrics: LossMetrics) -> LossMetrics:
		if (
			self.arguments.loss_config is not None and self.arguments.loss_config.break_on_nan
		):
			if jax.numpy.isnan(metrics.loss):
				info = "Prevent Running Model Due to NaN Loss"
				logger.info(info)
				raise EasyDeLBreakRequest(info)
		if (
			self.arguments.training_time_seconds is not None
			and time.time() > self.arguments.training_time_seconds + self._training_time_start
		):
			info = "Prevent Running Model Due to Time Limit"
			logger.info(info)
			raise EasyDeLTimerError(info)
		return metrics

	def start_training_hook(self):
		self.get_runstage_flops(True)
		self._setup_static_metrics()
		self._training_time_start = time.time()

	def start_evaluation_hook(self):
		self.get_runstage_flops(False)
		self._setup_static_metrics()
		self._evaluation_time_start = time.time()

	def _setup_static_metrics(self): ...

	def compile_aot(self) -> bool:
		compiled = False

		def compile_function(function, dataloader, state, tag):
			if not isinstance(function, Compiled):
				logger.info("Compiling function: %s", tag)
				return function.lower(state, next(iter(dataloader))).compile()
			return function

		if self.dataloader_train is not None:
			self.sharded_training_step_function = compile_function(
				self.sharded_training_step_function,
				self.dataloader_train,
				self.model_state,
				"trainer.sharded_training_step_function",
			)
			compiled = True

		if self.dataloader_eval is not None:
			self.sharded_evaluation_step_function = compile_function(
				self.sharded_evaluation_step_function,
				self.dataloader_eval,
				self.model_state,
				"trainer.sharded_evaluation_step_function",
			)
			compiled = True

		return compiled

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
			and current_step % self.arguments.save_steps == 0
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
				logger.warning(
					"KeyboardInterrupt: Training interrupted. Saving current state..."
				)
			elif isinstance(run_exception, EasyDeLTimerError):
				logger.warning("Training reached maximum time limit. Saving current state...")
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
			logger.warning("KeyboardInterrupt: Training interrupted. Saving current state...")
		elif isinstance(exception, EasyDeLTimerError):
			logger.warning("Training reached maximum time limit. Saving current state...")
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
		model_size = self.count_model_parameters(state.graphstate)
		self.arguments.log_metrics(
			{
				"Number of Model Parameters (Billion)": model_size,
				"process_count": jax.process_count(),
				"device_count": jax.device_count(),
				"local_device_count": jax.local_device_count(),
				"platform": jax.extend.backend.get_backend().platform,
				"XLA_FLAGS": os.getenv("XLA_FLAGS", ""),
				"LIBTPU_INIT_ARGS": os.getenv("LIBTPU_INIT_ARGS", ""),
			},
			step=0,
			log_as="config",
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

	def create_progress_bar(
		self,
		total: int,
		desc: str = "",
		disabled: bool = False,
	) -> BaseProgressBar:
		"""Create a progress bar of the specified type."""
		if disabled:
			return NullProgressBar()
		rpr = self.arguments.progress_bar_type
		if rpr == "tqdm":
			ncols = int(os.getenv("TQDM_NCOLS", "0"))
			return TqdmProgressBar(
				tqdm.tqdm(
					total=total,
					desc=desc,
					disable=disabled,
					ncols=ncols if ncols > 0 else None,
				)
			)
		elif rpr == "rich":  # rich
			from rich.progress import Progress

			if hasattr(self, "_hidden_rich_pbar"):
				progress = self._hidden_rich_pbar
			else:
				from rich.progress import (
					BarColumn,
					Progress,
					SpinnerColumn,
					TextColumn,
					TimeRemainingColumn,
				)

				from .trainer_protocol import MetricsColumn

				progress = Progress(
					SpinnerColumn(),
					TextColumn("[bold blue]{task.description}"),
					BarColumn(bar_width=None),
					TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
					TimeRemainingColumn(),
					MetricsColumn(metrics_to_show=self.arguments.metrics_to_show_in_rich_pbar),
					expand=True,
					refresh_per_second=10,
					disable=disabled,
				)
				progress.start()
				self._hidden_rich_pbar = progress
			task_id = progress.add_task(desc, total=total)
			return RichProgressBar(progress, task_id)
		elif rpr == "json":
			return JSONProgressBar(desc=desc)
		else:
			raise NotImplementedError(f"Progress Bar type {rpr}'s not supported.")

	def log_weight_distribution(self, state: EasyDeLState, step: int):
		return self.arguments.log_weight_distribution(state=state, step=step)

	def log_metrics(
		self,
		metrics: MetricsType,
		pbar: BaseProgressBar,
		step: int,
		mode: str = "train",
	):
		"""Log metrics and update progress bar."""

		if step % self.arguments.log_steps == 0:
			if step == 0:
				pbar.reset()
			display_metrics = {
				k.replace("train/", "").replace("eval/", ""): v
				for k, v in metrics.items()
				if not (
					k.startswith("mlperf/")
					or k.startswith("train/grad_norm")
					or k.startswith("eval/grad_norm")
				)
			}
			# Update progress bar
			pbar.set_postfix(**display_metrics)
			update_size = 0 if step == 0 else self.arguments.log_steps
			pbar.update(update_size)
		if step % self.arguments.report_steps == 0:
			self.arguments.log_metrics(metrics=metrics, step=step)
