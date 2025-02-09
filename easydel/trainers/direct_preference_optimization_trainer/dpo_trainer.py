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
from collections import defaultdict
from functools import partial

import jax
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from tqdm.autonotebook import tqdm

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils.helpers import get_logger
from easydel.utils.traversals import deepcopy_model

from ..base_trainer import TrainerConfigureFunctionOutput
from ..prompt_utils import maybe_apply_chat_template, maybe_extract_prompt
from ..trainer.trainer import Trainer
from ..training_configurations import MetricsType
from ..utils import DataCollatorForPreference
from ._fn import concatenated_forward, evaluation_step, training_step
from .dpo_config import DPOConfig

if tp.TYPE_CHECKING:
	from datasets import Dataset, IterableDataset
	from transformers import (
		BaseImageProcessor,
		FeatureExtractionMixin,
		PreTrainedTokenizerBase,
		ProcessorMixin,
	)
else:
	Dataset = tp.Any
	IterableDataset = tp.Any
	BaseImageProcessor = tp.Any
	FeatureExtractionMixin = tp.Any
	PreTrainedTokenizerBase = tp.Any
	ProcessorMixin = tp.Any

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
	):
		assert arguments is not None, (
			"You Have to pass arguments that will be used for training but you have passed"
			"`arguments=None`"
		)
		assert isinstance(arguments, DPOConfig), (
			f"arguments type must be `DPOConfig` but got {type(arguments)}"
		)

		assert processing_class is not None, (
			"processing_class must be specified to tokenize a DPO dataset."
		)
		self.arguments = arguments
		self.truncation_mode = arguments.truncation_mode
		self.processing_class = processing_class
		self.is_encoder_decoder = arguments.is_encoder_decoder
		self._precomputed_train_ref_log_probs = False
		self._precomputed_eval_ref_log_probs = False

		if arguments.padding_value is not None:
			self.padding_value = arguments.padding_value
		else:
			if (
				hasattr(processing_class, "pad_token_id")
				and processing_class.pad_token_id is not None
			):
				self.padding_value = processing_class.pad_token_id
			elif (
				hasattr(processing_class, "tokenizer")
				and processing_class.tokenizer.pad_token_id is not None
			):
				self.padding_value = processing_class.tokenizer.pad_token_id
			else:
				raise ValueError(
					"`padding_value` is not specified in `DPOConfig`, and `pad_token_id` is missing in the "
					"`processing_class`. Please either set the `padding_value` argument in `DPOConfig`, or set "
					"`tokenizer.pad_token` (e.g., `tokenizer.pad_token = tokenizer.eos_token`) before instantiating "
					"the trainer."
				)
		arguments.padding_value = self.padding_value
		input_data_collator = (
			DataCollatorForPreference(
				max_prompt_length=arguments.max_prompt_length,
				max_completion_length=arguments.max_completion_length,  # type: ignore
				pad_token_id=self.padding_value,  # type: ignore
				label_pad_token_id=arguments.label_pad_token_id,
				is_encoder_decoder=arguments.is_encoder_decoder,
			)
			if data_collator is None
			else data_collator
		)
		self.input_data_collator = input_data_collator

		self._stored_metrics = defaultdict(lambda: defaultdict(list))

		processing_class = processing_class

		if not isinstance(model, EasyDeLState):
			model = model.to_state()
		if reference_model is None:
			reference_model = deepcopy_model(model)
		if not isinstance(reference_model, EasyDeLState):
			reference_model = reference_model.to_state()

		train_dataset = self._prepare_dataset(
			train_dataset,
			processing_class,
			arguments,
			"train",
		)
		if eval_dataset is not None:
			if isinstance(eval_dataset, dict):
				eval_dataset = {
					key: self._prepare_dataset(dataset, processing_class, arguments, key)
					for key, dataset in eval_dataset.items()
				}
			else:
				eval_dataset = self._prepare_dataset(
					eval_dataset,
					processing_class,
					arguments,
					"eval",
				)

		self.arguments = arguments

		self.train_dataset = train_dataset
		self.eval_dataset = eval_dataset
		self.processing_class = processing_class
		self.reference_state = reference_model

		super().__init__(
			model_state=model,
			arguments=arguments,
			dataset_train=train_dataset,
			dataset_eval=eval_dataset,
			data_collator=None,
		)

	def _prepare_dataset(
		self,
		dataset: tp.Union[Dataset, IterableDataset],
		processing_class: tp.Union[
			PreTrainedTokenizerBase,
			BaseImageProcessor,
			FeatureExtractionMixin,
			ProcessorMixin,
		],
		arguments: DPOConfig,
		dataset_name: str,
	) -> tp.Union[Dataset, IterableDataset]:
		map_kwargs = {"writer_batch_size": 10}
		from datasets import Dataset

		if isinstance(dataset, Dataset):
			map_kwargs["num_proc"] = arguments.dataset_num_proc

		if isinstance(dataset, Dataset):
			map_kwargs["desc"] = f"Extracting prompt in {dataset_name} dataset"
		dataset = dataset.map(maybe_extract_prompt, **map_kwargs)

		if isinstance(dataset, Dataset):
			map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
		dataset = dataset.map(
			maybe_apply_chat_template,
			fn_kwargs={
				"tokenizer": processing_class,
				"tools": arguments.tools,
			},
			**map_kwargs,
		)

		if isinstance(dataset, Dataset):
			map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

		dataset = dataset.map(
			self.tokenize_row,  # if not self.is_vision_model else self.process_row,
			remove_columns=["prompt", "chosen", "rejected"],
			fn_kwargs={
				"processing_class": processing_class,
				"max_prompt_length": arguments.max_prompt_length,
				"max_completion_length": arguments.max_completion_length,
				"add_special_tokens": False,
			},
			**map_kwargs,
		)
		return dataset

	@staticmethod
	def tokenize_row(
		features,
		processing_class,
		max_prompt_length,
		max_completion_length,
		add_special_tokens,
	):
		"""
		Tokenize a row of the dataset.

		Args:
		    features (`dict[str, str]`):
		        Row of the dataset, should contain the keys `"prompt"`, `"chosen"`, and `"rejected"`.
		    processing_class (`PreTrainedTokenizerBase`):
		        Processing class used to process the data.
		    max_prompt_length (`int` or `None`):
		        Maximum length of the prompt sequence. If `None`, the prompt sequence is not truncated.
		    max_completion_length (`int` or `None`):
		        Maximum length of the completion sequences. If `None`, the completion sequences are not truncated.
		    add_special_tokens (`bool`):
		        Whether to add special tokens to the sequences. Typically used for encoder-decoder models. If `True`,
		        the prompt sequence will have a bos token prepended and an eos token appended. In any case, the
		        completion sequences will have an eos token appended.

		Returns:
		    `dict[str, list[int]]`:
		        Tokenized sequences with the keys `"prompt_input_ids"`, `"chosen_input_ids"`, and
		        `"rejected_input_ids".
		"""
		tokenizer = processing_class
		prompt_input_ids = tokenizer(
			features["prompt"],
			add_special_tokens=False,
		)["input_ids"]
		chosen_input_ids = tokenizer(
			features["chosen"],
			add_special_tokens=False,
		)["input_ids"]
		rejected_input_ids = tokenizer(
			features["rejected"],
			add_special_tokens=False,
		)["input_ids"]

		if add_special_tokens:
			if tokenizer.bos_token_id is not None:
				prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
			if tokenizer.eos_token_id is not None:
				prompt_input_ids = prompt_input_ids + [tokenizer.eos_token_id]
		chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
		rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

		if max_prompt_length is not None:
			prompt_input_ids = prompt_input_ids[-max_prompt_length:]
		if max_completion_length is not None:
			chosen_input_ids = chosen_input_ids[:max_completion_length]
			rejected_input_ids = rejected_input_ids[:max_completion_length]

		return {
			"prompt_input_ids": prompt_input_ids,
			"chosen_input_ids": chosen_input_ids,
			"rejected_input_ids": rejected_input_ids,
		}

	@staticmethod
	def process_row(
		features,
		processing_class,
		max_prompt_length,
		max_completion_length,
		add_special_tokens,
	):
		processor, tokenizer = (processing_class, processing_class.tokenizer)
		processed_features = processor(
			images=features["images"],
			text=features["prompt"],
			add_special_tokens=False,
		)

		prompt_input_ids = processed_features["input_ids"][0]
		pixel_values = processed_features["pixel_values"][0]
		chosen_input_ids = tokenizer(
			features["chosen"],
			add_special_tokens=False,
			return_tensors="jax",
		)["input_ids"]
		rejected_input_ids = tokenizer(
			features["rejected"],
			add_special_tokens=False,
			return_tensors="jax",
		)["input_ids"]

		if add_special_tokens:
			if tokenizer.bos_token_id is not None:
				prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
			if tokenizer.eos_token_id is not None:
				prompt_input_ids = prompt_input_ids + [tokenizer.eos_token_id]
		chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
		rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

		if max_prompt_length is not None:
			prompt_input_ids = prompt_input_ids[-max_prompt_length:]
		if max_completion_length is not None:
			chosen_input_ids = chosen_input_ids[:max_completion_length]
			rejected_input_ids = rejected_input_ids[:max_completion_length]

		output = {
			"prompt_input_ids": prompt_input_ids,
			"pixel_values": pixel_values,
			"chosen_input_ids": chosen_input_ids,
			"rejected_input_ids": rejected_input_ids,
		}

		if "pixel_attention_mask" in processed_features:
			output["pixel_attention_mask"] = processed_features["pixel_attention_mask"][0]
		if "image_sizes" in processed_features:
			output["image_sizes"] = processed_features["image_sizes"][0]

		return output

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
		# returns chosen_log_probs, rejected_log_probs, chosen_logits, rejected_logits

		partial_concatenated_forward = partial(
			concatenated_forward,
			is_encoder_decoder=self.arguments.is_encoder_decoder,
			padding_value=self.padding_value,
			label_pad_token_id=self.arguments.label_pad_token_id,
			max_length=self.arguments.max_length,
			loss_type=self.arguments.loss_type,
			aux_loss_enabled=self.arguments.aux_loss_enabled,
			truncation_mode=self.arguments.truncation_mode,
		)
		jited_concatenated_forward = jax.jit(
			partial_concatenated_forward,
			out_shardings=(empty_sharding,),
			static_argnames=[
				"is_encoder_decoder",
				"padding_value",
				"label_pad_token_id",
				"aux_loss_enabled",
				"truncation_mode",
				"loss_type",
			],
		)
		sharded_training_step_function = jax.jit(
			partial(
				training_step,
				learning_rate_fn=self.scheduler,
				concatenated_forward=partial_concatenated_forward,
				beta=self.arguments.beta,
				label_smoothing=self.arguments.label_smoothing,
				loss_type=self.arguments.loss_type,
				reference_free=self.arguments.reference_free,
			),
			in_shardings=(
				self.state_shardings,
				empty_sharding,
				self.reference_state.shardings,
			),
			out_shardings=(self.state_shardings, empty_sharding),
			donate_argnums=(0,),
			static_argnames=[
				"learning_rate_fn",
				"concatenated_forward",
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
			in_shardings=(
				self.state_shardings,
				empty_sharding,
				self.reference_state.shardings,
			),
			out_shardings=empty_sharding,
			static_argnames=[
				"concatenated_forward",
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
		return self.input_data_collator

	def configure_dataloaders(self):
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
		if self.train_dataset is not None:
			if (
				self.arguments.precompute_ref_log_probs
				and not self._precomputed_train_ref_log_probs
			):
				data_loader = tensorflow_datasets.as_numpy(
					self.train_dataset.to_tf_dataset(
						batch_size=self.training_batch_size,
						collate_fn=self.input_data_collator,
						num_workers=self.arguments.dataloader_num_workers,
						shuffle=False,
						drop_remainder=True,
					)
				)
				reference_chosen_log_probs = []
				ref_rejected_logps = []
				for padded_batch in tqdm(
					iterable=data_loader,
					desc="Train dataset reference log probs",
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
					name="reference_chosen_log_probs",
					column=all_reference_chosen_log_probs,
				)
				self.train_dataset = self.train_dataset.add_column(
					name="ref_rejected_logps",
					column=all_ref_rejected_logps,
				)

				self._precomputed_train_ref_log_probs = True
		if self.eval_dataset is not None:
			if (
				self.arguments.precompute_ref_log_probs
				and not self._precomputed_eval_ref_log_probs
			):
				data_loader = tensorflow_datasets.as_numpy(
					self.eval_dataset.to_tf_dataset(
						batch_size=self.evaluation_batch_size,
						collate_fn=self.input_data_collator,
						num_workers=self.arguments.dataloader_num_workers,
						shuffle=False,
						drop_remainder=True,
					)
				)
				reference_chosen_log_probs = []
				ref_rejected_logps = []
				for padded_batch in tqdm(
					iterable=data_loader,
					desc="Eval dataset reference log probs",
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
				self.eval_dataset = self.eval_dataset.add_column(
					name="reference_chosen_log_probs",
					column=all_reference_chosen_log_probs,
				)
				self.eval_dataset = self.eval_dataset.add_column(
					name="ref_rejected_logps",
					column=all_ref_rejected_logps,
				)

				self._precomputed_train_ref_log_probs = True
		return super().configure_dataloaders()

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

	@property
	def _train_shared_fn_extra_args(self) -> tp.Tuple[tp.Any]:
		return (self.reference_state,)

	@property
	def _eval_shared_fn_extra_args(self) -> tp.Tuple[tp.Any]:
		return (self.reference_state,)

	def on_step_end(
		self,
		state: EasyDeLState,
		metrics: MetricsType,
		step: int,
	) -> tp.Tuple[EasyDeLState, MetricsType]:
		"""hook process to call in start of the step."""

		if (
			self.arguments.sync_ref_model
			and self.reference_state is not None
			and (step % self.arguments.ref_model_sync_steps == 0)
		):
			self.reference_state = self.reference_state.replace(
				graphstate=deepcopy_model(state.graphstate)
			)
		return state, metrics
