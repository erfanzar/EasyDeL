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

import typing as tp
import warnings
from collections import defaultdict
from functools import partial

import jax
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from tqdm.autonotebook import tqdm

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.errors import EasyDeLBreakRequest, EasyDeLTimerError
from easydel.infra.loss_utils import LossMetrics
from easydel.infra.utils import ProcessingClassType
from easydel.trainers.trainer.trainer import Trainer
from easydel.utils.helpers import get_logger

from ..base_trainer import (
	BaseTrainer,
	TrainerConfigureDataloaderOutput,
	TrainerConfigureFunctionOutput,
	TrainerConfigureModelOutput,
)
from ..prompt_utils import maybe_apply_chat_template, maybe_extract_prompt
from .dpo_config import DPOConfig
from .func_utils import concatenated_forward, evaluation_step, training_step
from .utils import DPODataCollatorWithPadding, build_tokenize

if tp.TYPE_CHECKING:
	from datasets import Dataset
else:
	Dataset = tp.Any

logger = get_logger(__name__)


class DPOTrainer(Trainer):
	"""
	Trainer for Direct Preference Optimization (DPO).

	This trainer handles the training, evaluation, and checkpointing of language models
	using the DPO algorithm. It supports sharding, gradient accumulation, mixed precision
	training, LoRA, and precomputed reference model log probabilities.
	"""

	arguments: DPOConfig

	def __init__(
		self,
		arguments: DPOConfig,
		model: tp.Union[EasyDeLBaseModule, EasyDeLState],
		reference_model: tp.Optional[tp.Union[EasyDeLBaseModule, EasyDeLState]] = None,
		processing_class: tp.Optional[ProcessingClassType] = None,
		train_dataset: tp.Optional[Dataset] = None,
		eval_dataset: tp.Optional[Dataset] = None,
		data_collator: tp.Optional[tp.Callable] = None,
		dataset_map_arguments: tp.Optional[dict] = None,
		low_mem_usage: bool = True,
		auto_fix_data: bool = True,
	):
		assert arguments is not None, (
			"You Have to pass arguments that will be used for training but you have passed"
			"`arguments=None`"
		)
		assert isinstance(
			arguments, DPOConfig
		), f"arguments type must be `DPOConfig` but got {type(arguments)}"

		assert (
			processing_class is not None
		), "processing_class must be specified to tokenize a DPO dataset."
		self.arguments = arguments
		self.auto_fix_data = auto_fix_data
		self.truncation_mode = arguments.truncation_mode
		self.processing_class = processing_class
		self.is_encoder_decoder = False
		self._precomputed_train_ref_log_probs = False
		self._precomputed_eval_ref_log_probs = False
		self.low_mem_usage = low_mem_usage

		arguments.padding_value = (
			arguments.padding_value
			if arguments.padding_value is not None
			else processing_class.pad_token_id
		)
		data_collator = (
			DPODataCollatorWithPadding(
				max_prompt_length=arguments.max_prompt_length,
				max_completion_length=arguments.max_completion_length,  # type: ignore
				pad_token_id=processing_class.pad_token_id,  # type: ignore
				label_pad_token_id=arguments.label_pad_token_id,
				is_encoder_decoder=arguments.is_encoder_decoder,
			)
			if data_collator is None
			else data_collator
		)
		self._stored_metrics = defaultdict(lambda: defaultdict(list))

		if dataset_map_arguments is None:
			dataset_map_arguments = {}

		processing_class = processing_class

		if train_dataset[-1].get("chosen", None) is None:
			warnings.warn(
				"couldn't find `chosen` column in dataset"
				", this might make DPOTrainer break down if you haven't customize trainer.",
				stacklevel=2,
			)
		if train_dataset[-1].get("rejected", None) is None:
			warnings.warn(
				"couldn't find `rejected` column in dataset"
				", this might make DPOTrainer break down if you haven't customize trainer.",
				stacklevel=2,
			)
		if train_dataset[-1].get("prompt", None) is None:
			warnings.warn(
				"couldn't find `prompt` column in dataset"
				", this might make DPOTrainer break down if you haven't customize trainer.",
				stacklevel=2,
			)
		train_dataset = train_dataset.map(
			maybe_extract_prompt,
			num_proc=arguments.dataset_num_proc,
			desc="Extracting Prompts",
		)
		train_dataset = train_dataset.map(
			maybe_apply_chat_template,
			fn_kwargs={"processing_class": processing_class},
			num_proc=arguments.dataset_num_proc,
			desc="Apply Chat Template",
		)
		if eval_dataset is not None:
			eval_dataset = eval_dataset.map(
				maybe_extract_prompt,
				num_proc=arguments.dataset_num_proc,
				desc="Eval - Extracting Prompts",
			)
			eval_dataset = eval_dataset.map(
				maybe_apply_chat_template,
				fn_kwargs={"processing_class": processing_class},
				num_proc=arguments.dataset_num_proc,
				desc="Eval - Apply Chat Template",
			)
		fn_kwargs = {
			"processing_class": self.processing_class,
			"processor": None,
		}
		if not isinstance(model, EasyDeLState):
			model = model.to_state()

		if not isinstance(reference_model, EasyDeLState):
			reference_model = reference_model.to_state()

		_tokenize = build_tokenize(
			model=model.model if arguments.is_encoder_decoder else None,
			args=arguments,
		)

		train_dataset = train_dataset.map(
			_tokenize,
			fn_kwargs=fn_kwargs,
			batched=True,
			num_proc=arguments.dataset_num_proc,
			writer_batch_size=10,
			desc="Tokenizing train dataset",
		)
		if eval_dataset is not None:
			eval_dataset = eval_dataset.map(
				_tokenize,
				fn_kwargs=fn_kwargs,
				batched=True,
				num_proc=arguments.dataset_num_proc,
				writer_batch_size=10,
				desc="Tokenizing eval dataset",
			)

		self.arguments = arguments

		self.is_in_train = False
		self.data_collator = data_collator
		self.train_dataset = train_dataset
		self.eval_dataset = eval_dataset
		self.processing_class = processing_class
		self.reference_state = reference_model
		self._loggers_initialized = False
		self.mesh = model.model.mesh

		self._cached_p_l_s = None
		self._cached_c_l_s = None
		self._cached_r_l_s = None
		super().__init__(
			model_state=model,
			arguments=arguments,
			dataset_train=train_dataset,
			dataset_eval=eval_dataset,
		)

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
		dataloader_train = self.get_train_dataloader()
		max_evaluation_steps = None
		dataloader_eval = None

		max_training_steps = (
			self.arguments.num_train_epochs * len(dataloader_train)
			if self.arguments.max_training_steps is None
			else self.arguments.max_training_steps
		)
		if self.eval_dataset is not None:
			dataloader_eval = self.get_eval_dataloader(self.eval_dataset)
			max_evaluation_steps = len(dataloader_eval)
		return TrainerConfigureDataloaderOutput(
			dataloader_eval=dataloader_eval,
			dataloader_train=dataloader_train,
			max_evaluation_steps=max_evaluation_steps,
			max_training_steps=max_training_steps,
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
		mesh = self.model.mesh
		empty_sharding = jax.sharding.NamedSharding(
			spec=PartitionSpec(),
			mesh=mesh,
		)
		input_sharding = jax.sharding.NamedSharding(
			spec=self.arguments.step_partition_spec,
			mesh=mesh,
		)
		# returns chosen_log_probs, rejected_log_probs, chosen_logits, rejected_logits

		partial_concatenated_forward = partial(
			concatenated_forward,
			is_encoder_decoder=self.arguments.is_encoder_decoder,
			padding_value=self.arguments.padding_value,
			label_pad_token_id=self.arguments.label_pad_token_id,
			fixed_max_length=self.arguments.max_length,
		)
		jited_concatenated_forward = jax.jit(
			partial_concatenated_forward,
			# in_shardings=(self.state_shardings.model, input_sharding),
			out_shardings=(empty_sharding, empty_sharding, empty_sharding, empty_sharding),
			static_argnames=[
				"is_encoder_decoder",
				"padding_value",
				"label_pad_token_id",
			],
		)
		sharded_training_step_function = jax.jit(
			partial(
				training_step,
				learning_rate_fn=self.scheduler,
				concatenated_forward=partial_concatenated_forward,
				reference_state=self.reference_state,
				beta=self.arguments.beta,
				label_smoothing=self.arguments.label_smoothing,
				loss_type=self.arguments.loss_type,
				reference_free=self.arguments.reference_free,
			),
			in_shardings=(self.state_shardings, input_sharding),
			out_shardings=(self.state_shardings, empty_sharding),
			static_argnames=[
				"learning_rate_fn",
				"concatenated_forward",
				"reference_state",
				"beta",
				"label_smoothing",
				"loss_type",
				"reference_free",
			],
		)

		sharded_evaluation_step_function = jax.jit(
			partial(
				evaluation_step,
				concatenated_forward=partial_concatenated_forward,
				reference_state=self.reference_state,
				beta=self.arguments.beta,
				label_smoothing=self.arguments.label_smoothing,
				loss_type=self.arguments.loss_type,
				reference_free=self.arguments.reference_free,
			),
			in_shardings=(self.state_shardings, input_sharding),
			out_shardings=empty_sharding,
			static_argnames=[
				"concatenated_forward",
				"reference_state",
				"beta",
				"label_smoothing",
				"loss_type",
				"reference_free",
			],
		)
		self.arguments.ensure_checkpoint_path()
		self.concatenated_forward = jited_concatenated_forward
		checkpoint_manager = self.arguments.get_streaming_checkpointer()

		return TrainerConfigureFunctionOutput(
			sharded_training_step_function=sharded_training_step_function,
			sharded_evaluation_step_function=sharded_evaluation_step_function,
			mesh=mesh,
			checkpoint_manager=checkpoint_manager,
		)

	def create_collect_function(
		self,
		max_sequence_length: int,
		truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
	) -> tp.Callable:
		"""
		Creates a data collection function for batching.

		For DPO training, this method simply returns the pre-configured `data_collator`.

		Args:
				max_sequence_length (int): The maximum sequence length (not used in this implementation).
				truncation_mode (tp.Literal["keep_end", "keep_start"], optional):
						The truncation mode (not used in this implementation). Defaults to "keep_end".

		Returns:
				tp.Callable: The data collator function.
		"""
		return self.data_collator

	def _get_train_dataloader(self) -> "tensorflow.data.Dataset":  # noqa #type:ignore
		"""
		Creates the training dataloader as a TensorFlow Dataset.

		This method retrieves the training dataset, applies the data collator, and converts
		it into a TensorFlow Dataset for efficient batching and data loading during training.

		Returns:
				tensorflow.data.Dataset: The training dataloader.

		Raises:
				ValueError: If the training dataset is not set.
		"""
		try:
			import tensorflow_datasets
		except ImportError as e:
			raise ImportError(
				"tensorflow_datasets is not installed, please install it by running `pip install tensorflow_datasets`"
			) from e
		if self.train_dataset is None:
			raise ValueError("Trainer: training requires a train_dataset.")

		train_dataset = self.train_dataset
		data_collator = self.data_collator

		return tensorflow_datasets.as_numpy(
			train_dataset.to_tf_dataset(
				batch_size=self.training_batch_size,
				collate_fn=data_collator,
				num_workers=self.arguments.dataloader_num_workers,
				shuffle=True if jax.process_count() == 1 else False,
				drop_remainder=True,
			)
		)

	def _get_eval_dataloader(
		self,
		eval_dataset: tp.Optional["Dataset"] = None,  # noqa #type:ignore
	) -> "tensorflow.data.Dataset":  # noqa #type:ignore
		"""
		Creates the evaluation dataloader as a TensorFlow Dataset.

		This method retrieves the evaluation dataset (either provided as an argument or
		from the `self.eval_dataset` attribute), applies the data collator, and converts
		it into a TensorFlow Dataset for efficient batching and data loading during evaluation.

		Args:
				eval_dataset (tp.Optional[Dataset], optional):
						An optional evaluation dataset to use. If None, `self.eval_dataset` is used. Defaults to None.

		Returns:
				tensorflow.data.Dataset: The evaluation dataloader.

		Raises:
				ValueError: If no evaluation dataset is provided or set.
		"""
		try:
			import tensorflow_datasets
		except ImportError as e:
			raise ImportError(
				"tensorflow_datasets is not installed, please install it by running `pip install tensorflow_datasets`"
			) from e

		if eval_dataset is None and self.eval_dataset is None:
			raise ValueError("Trainer: evaluation requires an eval_dataset.")
		eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

		return tensorflow_datasets.as_numpy(
			eval_dataset.to_tf_dataset(
				batch_size=self.evaluation_batch_size,
				collate_fn=self.data_collator,
				num_workers=self.arguments.dataloader_num_workers,
				shuffle=False,
				drop_remainder=True,
			)
		)

	def get_train_dataloader(self) -> "tensorflow.data.Dataset":  # noqa #type:ignore
		"""
		Returns the training dataloader, potentially with precomputed reference log probabilities.

		If `precompute_ref_log_probs` is enabled, this method computes the reference model's log
		probabilities for the chosen and rejected responses in the training dataset and adds
		them as columns to the dataset.

		Returns:
				tensorflow.data.Dataset: The training dataloader.
		"""

		try:
			import tensorflow_datasets
		except ImportError as e:
			raise ImportError(
				"tensorflow_datasets is not installed, please install it by running `pip install tensorflow_datasets`"
			) from e

		if (
			self.arguments.precompute_ref_log_probs
			and not self._precomputed_train_ref_log_probs
		):
			data_loader = tensorflow_datasets.as_numpy(
				self.train_dataset.to_tf_dataset(
					batch_size=self.training_batch_size,
					collate_fn=self.data_collator,
					num_workers=self.arguments.dataloader_num_workers,
					shuffle=False,
					drop_remainder=True,
				)
			)
			reference_chosen_log_probs = []
			ref_rejected_logps = []
			for padded_batch in tqdm(
				iterable=data_loader, desc="Train dataset reference log probs"
			):
				reference_chosen_logp, reference_rejected_logp = (
					self.compute_reference_log_probs(
						self.model_state,
						padded_batch,
					)
				)
				reference_chosen_log_probs.append(reference_chosen_logp)
				ref_rejected_logps.append(reference_rejected_logp)

			all_reference_chosen_log_probs = jnp.concatenate(reference_chosen_log_probs)
			all_ref_rejected_logps = jnp.concatenate(ref_rejected_logps)
			self.train_dataset = self.train_dataset.add_column(
				name="reference_chosen_log_probs", column=all_reference_chosen_log_probs
			)
			self.train_dataset = self.train_dataset.add_column(
				name="ref_rejected_logps",
				column=all_ref_rejected_logps,
			)

			self._precomputed_train_ref_log_probs = True
		return self._get_train_dataloader()

	def get_eval_dataloader(
		self,
		eval_dataset: tp.Optional["Dataset"] = None,  # noqa #type:ignore
	) -> "tensorflow.data.Dataset":  # noqa #type:ignore
		"""
		Returns the evaluation dataloader, potentially with precomputed reference log probabilities.

		If `precompute_ref_log_probs` is enabled, this method computes the reference model's log
		probabilities for the chosen and rejected responses in the evaluation dataset and adds
		them as columns to the dataset.

		Args:
				eval_dataset (tp.Optional[Dataset], optional):
						An optional evaluation dataset to use. If None, `self.eval_dataset` is used. Defaults to None.

		Returns:
				tensorflow.data.Dataset: The evaluation dataloader.
		"""

		try:
			import tensorflow_datasets
		except ImportError as e:
			raise ImportError(
				"tensorflow_datasets is not installed, please install it by running `pip install tensorflow_datasets`"
			) from e

		if eval_dataset is None and self.eval_dataset is None:
			raise ValueError("Trainer: evaluation requires an eval_dataset.")
		eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

		if (
			self.arguments.precompute_ref_log_probs
			and not self._precomputed_eval_ref_log_probs
		):
			# prepare dataloader
			data_loader = tensorflow_datasets.as_numpy(
				eval_dataset.to_tf_dataset(
					batch_size=self.evaluation_batch_size,
					collate_fn=self.data_collator,
					num_workers=self.arguments.dataloader_num_workers,
					shuffle=False,
					drop_remainder=True,
				)
			)

			reference_chosen_log_probs = []
			ref_rejected_logps = []
			for padded_batch in tqdm(
				iterable=data_loader, desc="Eval dataset reference log probs"
			):
				reference_chosen_logp, reference_rejected_logp = (
					self.compute_reference_log_probs(self.model_state, padded_batch)
				)
				reference_chosen_log_probs.append(reference_chosen_logp.cpu())
				ref_rejected_logps.append(reference_rejected_logp.cpu())

			all_reference_chosen_log_probs = jnp.concatenate(reference_chosen_log_probs)
			all_ref_rejected_logps = jnp.concatenate(ref_rejected_logps)

			eval_dataset = eval_dataset.add_column(
				name="reference_chosen_log_probs", column=all_reference_chosen_log_probs
			)
			eval_dataset = eval_dataset.add_column(
				name="ref_rejected_logps",
				column=all_ref_rejected_logps,
			)

			if self.eval_dataset is not None:
				self.eval_dataset = eval_dataset
			self._precomputed_eval_ref_log_probs = True

		return self._get_eval_dataloader(eval_dataset=eval_dataset)

	def compute_reference_log_probs(
		self,
		state: EasyDeLState,
		padded_batch: tp.Dict,
	) -> tuple[tp.Any, tp.Any]:
		"""
		Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset.

		Args:
				state (EasyDeLState): The EasyDeLState object of the model (used if no reference model is provided).
				padded_batch (tp.Dict): The padded batch of data.

		Returns:
				tuple[tp.Any, tp.Any]: A tuple containing the log probabilities for the chosen and rejected responses.
		"""

		if self.reference_state is None:
			outs = self.concatenated_forward(state.model, batch=padded_batch)
		else:
			outs = self.concatenated_forward(self.reference_state.model, batch=padded_batch)
		return outs["chosen_logps"], outs["rejected_logps"]

	def _execute_eval_step(self, state: EasyDeLState, batch) -> LossMetrics:
		"""Execute a single eval step."""
		batch = {key: jnp.asarray(value) for key, value in batch.items()}

		metrics = self.sharded_evaluation_step_function(state, batch)
		return metrics

	def _execute_train_step(
		self, state: EasyDeLState, batch
	) -> tp.Tuple[EasyDeLState, LossMetrics, Exception]:
		"""Execute a single training step."""
		try:
			batch = {key: jnp.asarray(value) for key, value in batch.items()}
			state, metrics = self.sharded_training_step_function(state, batch)
			return state, metrics, None
		except (KeyboardInterrupt, EasyDeLTimerError, EasyDeLBreakRequest) as run_exception:
			return state, metrics, run_exception


def check_unimplemented_abstract_methods():
	from inspect import getmembers, isfunction

	base_class = BaseTrainer
	derived_class = DPOTrainer
	abstract_methods = [
		method
		for method in getmembers(base_class, predicate=isfunction)
		if getattr(getattr(base_class, method[0]), "__isabstractmethod__", False)
	]

	not_implemented = [
		method[0]
		for method in abstract_methods
		if getattr(derived_class, method[0]) is getattr(base_class, method[0])
	]
	if len(not_implemented) != 0:
		warnings.warn(
			f"{DPOConfig.__name__} does not implement the following abstract methods: {', '.join(not_implemented)}. "
			f"Please ensure these methods are implemented in {DPOConfig.__name__}.",
			stacklevel=1,
			category=UserWarning,
		)


check_unimplemented_abstract_methods()
