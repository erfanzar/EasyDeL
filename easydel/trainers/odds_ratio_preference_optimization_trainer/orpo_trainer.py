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
import typing
import typing as tp
import warnings
from collections import defaultdict
from functools import partial

import jax
import numpy as np
from jax import jit
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.errors import EasyDeLBreakRequest, EasyDeLTimerError
from easydel.infra.loss_utils import LossMetrics
from easydel.trainers.trainer.trainer import Trainer
from easydel.utils.helpers import get_logger

from ..base_trainer import (
	BaseTrainer,
	TrainerConfigureDataloaderOutput,
	TrainerConfigureFunctionOutput,
)
from ..direct_preference_optimization_trainer.utils import (
	DPODataCollatorWithPadding,
)
from ._fn import concatenated_forward, orpo_step
from .orpo_config import ORPOConfig

if tp.TYPE_CHECKING:
	from datasets import Dataset
	from tensorflow import data
	from transformers import PreTrainedTokenizerBase

	TFDataset = data.Dataset

else:
	Dataset = tp.Any
	PreTrainedTokenizerBase = tp.Any
	TFDataset = tp.Any

logger = get_logger(__name__)


class ORPOTrainer(Trainer):
	def __init__(
		self,
		arguments: ORPOConfig,
		model: tp.Optional[tp.Union[EasyDeLBaseModule, EasyDeLState]] = None,
		data_collator: tp.Optional[DPODataCollatorWithPadding] = None,
		train_dataset: tp.Optional[Dataset] = None,
		eval_dataset: tp.Optional[tp.Union[Dataset, tp.Dict[str, Dataset]]] = None,
		tokenizer: tp.Optional[PreTrainedTokenizerBase] = None,
		dataset_num_proc: tp.Optional[int] = None,
		dataset_map_arguments: tp.Optional[tp.Dict[str, tp.Any]] = None,
		low_mem_usage: bool = False,
	):
		"""
		Initializes the ORPOTrainer.

		Args:
		    arguments (ORPOConfig): The training arguments.
		    data_collator (tp.Optional[DPODataCollatorWithPadding], optional): The data collator used for batching. Defaults to None.
		    train_dataset (tp.Optional[Dataset], optional): The training dataset. Defaults to None.
		    eval_dataset (tp.Optional[tp.Union[Dataset, tp.Dict[str, Dataset]]], optional): The evaluation dataset. Defaults to None.
		    tokenizer (tp.Optional[PreTrainedTokenizerBase], optional): The tokenizer used for preprocessing. Defaults to None.
		    dataset_num_proc (tp.Optional[int], optional): The number of processes to use for dataset mapping. Defaults to None.
		    dataset_map_arguments (tp.Optional[tp.Dict[str, tp.Any]], optional): Arguments to pass to the dataset `map` function for tokenization. Defaults to None.
		    low_mem_usage (bool, optional): Whether to prioritize low memory usage during training. Defaults to False.

		Raises:
		    ValueError: If `arguments` is not provided or is not a `TrainingArguments` instance, or if `tokenizer` is not provided.
		"""

		assert arguments is not None, (
			"You Have to pass arguments that will be used for training but you have passed"
			"`arguments=None`"
		)
		assert isinstance(
			arguments, ORPOConfig
		), f"arguments type must be `TrainingArguments` but got {type(arguments)}"

		if tokenizer is None:
			raise ValueError("tokenizer must be specified to tokenize a ORPO dataset.")

		arguments.padding_value = (
			arguments.padding_value
			if arguments.padding_value is not None
			else tokenizer.pad_token_id
		)
		self.arguments = arguments
		self.tokenizer = tokenizer
		self.low_mem_usage = low_mem_usage
		self.dataset_num_proc = dataset_num_proc
		data_collator = (
			DPODataCollatorWithPadding(
				max_prompt_length=arguments.max_prompt_length,
				max_completion_length=arguments.max_completion_length,
				pad_token_id=tokenizer.pad_token_id,
				label_pad_token_id=arguments.label_pad_token_id,
				is_encoder_decoder=False,
			)
			if data_collator is None
			else data_collator
		)
		self._stored_metrics = defaultdict(lambda: defaultdict(list))
		if dataset_map_arguments is None:
			dataset_map_arguments = {}

		with jax.default_device(arguments.offload_device):
			off_div = arguments.offload_device
			# avoids TypeError: cannot pickle 'jaxlib.xla_extension.Device' object
			arguments.offload_device = None
			train_dataset = train_dataset.map(
				self.tokenize_row,
				num_proc=dataset_num_proc,
				batched=False,
				**dataset_map_arguments,
			)
			if eval_dataset is not None:
				eval_dataset = eval_dataset.map(
					self.tokenize_row,
					num_proc=dataset_num_proc,
					batched=False,
					**dataset_map_arguments,
				)
			arguments.offload_device = off_div
		self.hp_name = None
		self.deepspeed = None
		self.is_in_train = False

		self.data_collator = data_collator
		self.train_dataset = train_dataset
		self.eval_dataset = eval_dataset
		self.tokenizer = tokenizer
		self._loggers_initialized = False
		if not isinstance(model, EasyDeLState):
			model = model.to_state()
		model_state = model
		self.mesh = model_state.model.mesh
		assert (
			arguments.padding_value is not None
		), "`padding_value` can not be set as `None` it must be an integer."

		self._cached_p_l_s = None
		self._cached_c_l_s = None
		self._cached_r_l_s = None

		super().__init__(
			arguments=arguments,
			model_state=model_state,
			dataset_train=train_dataset,
			dataset_eval=eval_dataset,
			finetune=True,
			checkpoint_path=None,
		)

	def build_tokenized_answer(
		self,
		prompt: str,
		answer: str,
	) -> tp.Dict[str, np.ndarray]:
		"""
		Tokenizes a prompt and answer pair, handling special tokens and padding/truncation.

		Args:
		    prompt (str): The prompt text.
		    answer (str): The answer text.

		Returns:
		    tp.Dict[str, np.ndarray]: A dictionary containing the tokenized prompt and answer, along with attention masks.

		Raises:
		    ValueError: If there's a mismatch in token lengths.
		"""
		full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
		prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

		# Convert to numpy arrays for consistency
		full_input_ids = np.array(full_tokenized["input_ids"])
		prompt_input_ids = np.array(prompt_input_ids)

		response_token_ids_start_idx = len(prompt_input_ids)

		# Check for potential off-by-one error
		if not np.array_equal(
			prompt_input_ids, full_input_ids[:response_token_ids_start_idx]
		):
			response_token_ids_start_idx -= 1
			if not np.array_equal(
				prompt_input_ids, full_input_ids[:response_token_ids_start_idx]
			):
				raise ValueError("Mismatch in prompt tokenization")

		prompt_attention_mask = full_tokenized["attention_mask"][
			:response_token_ids_start_idx
		]
		answer_input_ids = full_input_ids[response_token_ids_start_idx:]
		answer_attention_mask = full_tokenized["attention_mask"][
			response_token_ids_start_idx:
		]

		if len(prompt_input_ids) != len(prompt_attention_mask):
			raise ValueError(
				"Prompt input ids and attention mask should have the same length."
			)

		return {
			"prompt_input_ids": prompt_input_ids.astype(np.int32),
			"prompt_attention_mask": np.array(prompt_attention_mask, dtype=np.int32),
			"input_ids": answer_input_ids.astype(np.int32),
			"attention_mask": np.array(answer_attention_mask, dtype=np.int32),
		}

	def tokenize_row(
		self,
		feature: tp.Dict[str, str],
		state: tp.Optional[object] = None,
	) -> tp.Dict[str, np.ndarray]:
		"""
		Tokenizes a single row of data from the ORPO dataset.

		This method tokenizes the prompt, chosen response, and rejected response,
		handles padding and truncation, and prepares the data for input to the DPO model.

		Args:
		    feature (tp.Dict): A dictionary containing the "prompt", "chosen", and "rejected" texts.
		    state (EasyDeLState, optional): Not used in this implementation. Defaults to None.

		Returns:
		    tp.Dict: A dictionary containing the tokenized prompt, chosen response, and rejected response,
		          along with attention masks and labels.

		Raises:
		    ValueError: If the input data types are incorrect.
		"""
		prompt = self._validate_input(feature, "prompt")
		chosen = self._validate_input(feature, "chosen")
		rejected = self._validate_input(feature, "rejected")

		prompt_tokens = self._tokenize_prompt(prompt)
		chosen_tokens = self._tokenize_answer(prompt, chosen)
		rejected_tokens = self._tokenize_answer(prompt, rejected)

		chosen_sequence = self._create_sequence(prompt_tokens, chosen_tokens)
		rejected_sequence = self._create_sequence(prompt_tokens, rejected_tokens)

		return self._prepare_final_batch(
			prompt_tokens,
			chosen_sequence,
			rejected_sequence,
		)

	def _validate_input(self, feature: tp.Dict[str, str], key: str) -> str:
		"""Validates input and returns the corresponding value."""
		value = feature[key]
		if self.arguments.apply_chat_template and key in ["chosen", "rejected"]:
			value = self.tokenizer.apply_chat_template(value, tokenize=False)

		if not isinstance(value, str):
			raise ValueError(f"{key} should be a string but got {type(value)}")
		return value

	def _tokenize_prompt(self, prompt: str) -> tp.Dict[str, np.ndarray]:
		"""Tokenizes the prompt."""
		tokens = self.tokenizer(prompt, add_special_tokens=False, return_tensors="np")
		return {f"prompt_{k}": v for k, v in tokens.items()}

	def _tokenize_answer(self, prompt: str, answer: str) -> tp.Dict[str, np.ndarray]:
		"""Tokenizes the answer in context of the prompt."""
		full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
		prompt_length = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])

		return {
			"prompt_input_ids": np.array(
				full_tokenized["input_ids"][:prompt_length], dtype=np.int32
			),
			"prompt_attention_mask": np.array(
				full_tokenized["attention_mask"][:prompt_length], dtype=np.int32
			),
			"input_ids": np.array(
				full_tokenized["input_ids"][prompt_length:], dtype=np.int32
			),
			"attention_mask": np.array(
				full_tokenized["attention_mask"][prompt_length:], dtype=np.int32
			),
		}

	def _create_sequence(
		self,
		prompt_tokens: tp.Dict[str, np.ndarray],
		answer_tokens: tp.Dict[str, np.ndarray],
	) -> tp.Dict[str, np.ndarray]:
		"""Creates a full sequence by combining prompt and answer tokens."""
		sequence = {}
		for key in ["input_ids", "attention_mask"]:
			sequence[key] = np.concatenate(
				[
					self._add_special_token(
						prompt_tokens[f"prompt_{key}"],
						self.tokenizer.bos_token_id,
						start=True,
					),
					self._add_special_token(
						answer_tokens[key],
						self.tokenizer.eos_token_id,
						start=False,
					),
				],
				axis=1,
			)
		sequence["labels"] = self._create_labels(
			sequence["input_ids"], len(prompt_tokens["prompt_input_ids"])
		)
		return sequence

	def _add_special_token(
		self,
		array: np.ndarray,
		token_id: int,
		start: bool,
	) -> np.ndarray:
		"""Adds a special token to the start or end of an array."""
		token = np.array([[token_id]], dtype=np.int32)
		array = np.atleast_2d(array)
		return np.concatenate([token, array] if start else [array, token], axis=1)

	def _create_labels(self, input_ids: np.ndarray, prompt_length: int) -> np.ndarray:
		"""Creates labels for the sequence, masking the prompt part."""
		labels = input_ids.copy()
		labels[:, :prompt_length] = self.arguments.label_pad_token_id
		return labels

	def _prepare_final_batch(
		self,
		prompt_tokens: tp.Dict[str, np.ndarray],
		chosen_sequence: tp.Dict[str, np.ndarray],
		rejected_sequence: tp.Dict[str, np.ndarray],
	) -> tp.Dict[str, np.ndarray]:
		"""Prepares the final batch by padding and truncating sequences."""
		batch = {}
		for prefix, tokens in [
			("", prompt_tokens),
			("chosen_", chosen_sequence),
			("rejected_", rejected_sequence),
		]:
			for key, value in tokens.items():
				if key == "token_type_ids":
					continue
				max_length = (
					self.arguments.max_prompt_length
					if prefix == ""
					else self.arguments.max_completion_length
				)
				pad_value = (
					self.arguments.padding_value if key in ["input_ids", "labels"] else 0
				)
				batch[f"{prefix}{key}"] = self._pad_and_truncate(value, max_length, pad_value)
		return batch

	def _pad_and_truncate(
		self,
		array: np.ndarray,
		max_length: int,
		pad_value: int,
	) -> np.ndarray:
		"""Pads or truncates an array to the specified length."""
		if array.shape[1] < max_length:
			padding = np.full(
				(array.shape[0], max_length - array.shape[1]),
				pad_value,
				dtype=array.dtype,
			)
			return np.concatenate([array, padding], axis=1)
		return array[:, :max_length]

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
		partial_concatenated_forward = partial(
			concatenated_forward,
			is_encoder_decoder=self.arguments.is_encoder_decoder,
			padding_value=self.arguments.padding_value,
			label_pad_token_id=self.arguments.label_pad_token_id,
		)
		jited_concatenated_forward = jax.jit(
			partial_concatenated_forward,
			static_argnames=[
				"is_encoder_decoder",
				"padding_value",
				"label_pad_token_id",
			],
		)
		empty_sharding = jax.sharding.NamedSharding(spec=PartitionSpec(), mesh=mesh)
		sharded_training_step_function = jit(
			partial(
				orpo_step,
				concatenated_forward=partial_concatenated_forward,
				beta=self.arguments.beta,
				learning_rate_fn=self.scheduler,
				mode="train",
				partition_spec=self.arguments.step_partition_spec,
				gradient_accumulation_steps=self.arguments.gradient_accumulation_steps,
				loss_config=self.arguments.loss_config,
			),
			in_shardings=(self.state_shardings, empty_sharding),
			out_shardings=(self.state_shardings, empty_sharding),
			static_argnames=[
				"concatenated_forward",
				"beta",
				"learning_rate_fn",
				"mode",
				"partition_spec",
				"gradient_accumulation_steps",
				"loss_config",
			],
		)

		sharded_evaluation_step_function = jit(
			partial(
				orpo_step,
				concatenated_forward=partial_concatenated_forward,
				beta=self.arguments.beta,
				learning_rate_fn=self.scheduler,
				mode="eval",
				partition_spec=self.arguments.step_partition_spec,
				gradient_accumulation_steps=self.arguments.gradient_accumulation_steps,
				loss_config=self.arguments.loss_config,
			),
			in_shardings=(self.state_shardings, empty_sharding),
			out_shardings=(empty_sharding),
			static_argnames=[
				"concatenated_forward",
				"beta",
				"learning_rate_fn",
				"mode",
				"partition_spec",
				"gradient_accumulation_steps",
				"loss_config",
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
		truncation_mode: typing.Literal["keep_end", "keep_start"] = "keep_end",
	) -> tp.Callable:
		return self.data_collator

	def configure_dataloaders(self) -> TrainerConfigureDataloaderOutput:
		"""
		Configures the dataloaders for training and evaluation.

		This method creates the training and evaluation dataloaders using the provided
		datasets and data collator. It also determines the maximum number of training
		and evaluation steps based on the dataset sizes and training arguments.

		Returns:
		    TrainerConfigureDataloaderOutput: An object containing the configured dataloaders and the maximum number of training and evaluation steps.
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
			dataloader_train=dataloader_train,  # type:ignore
			max_training_steps=max_training_steps,
			dataloader_eval=dataloader_eval,
			max_evaluation_steps=max_evaluation_steps,
		)

	def _get_train_dataloader(self) -> TFDataset:
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
				shuffle=True,
				drop_remainder=True,
			)
		)

	def _get_eval_dataloader(
		self,
		eval_dataset: tp.Optional[Dataset] = None,
	) -> TFDataset:
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

	def get_train_dataloader(self) -> TFDataset:
		"""
		Returns the training dataloader

		Returns:
		    tensorflow.data.Dataset: The training dataloader.
		"""
		return self._get_train_dataloader()

	def get_eval_dataloader(self, eval_dataset: tp.Optional[Dataset] = None) -> TFDataset:
		"""
		Returns the evaluation dataloader
		Args:
		    eval_dataset (tp.Optional[Dataset], optional):
		        An optional evaluation dataset to use. If None, `self.eval_dataset` is used. Defaults to None.

		Returns:
		    tensorflow.data.Dataset: The evaluation dataloader.
		"""
		if eval_dataset is None and self.eval_dataset is None:
			raise ValueError("Trainer: evaluation requires an eval_dataset.")
		eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
		return self._get_eval_dataloader(eval_dataset=eval_dataset)

	def _execute_eval_step(
		self,
		state: EasyDeLState,
		batch,
	) -> LossMetrics:
		"""Execute a single eval step."""
		batch = {key: jnp.asarray(value) for key, value in batch.items()}

		metrics = self.sharded_evaluation_step_function(state, batch)
		return metrics

	def _execute_train_step(
		self,
		state: EasyDeLState,
		batch,
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
	derived_class = ORPOTrainer
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
			f"{ORPOConfig.__name__} does not implement the following abstract methods: {', '.join(not_implemented)}. "
			f"Please ensure these methods are implemented in {ORPOConfig.__name__}.",
			stacklevel=1,
			category=UserWarning,
		)


check_unimplemented_abstract_methods()
