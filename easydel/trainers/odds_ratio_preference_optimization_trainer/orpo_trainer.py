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
import typing as tp
from collections import defaultdict
from functools import partial

import jax
import numpy as np
from jax import jit
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils.helpers import get_logger

from ..base_trainer import (
	TrainerConfigureFunctionOutput,
)
from ..prompt_utils import (
	maybe_apply_chat_template,
	maybe_extract_prompt,
)
from ..trainer.trainer import Trainer
from ..utils import (
	DPODataCollatorWithPadding,
	add_bos_token_if_needed,
	add_eos_token_if_needed,
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
	arguments: ORPOConfig

	def __init__(
		self,
		arguments: ORPOConfig,
		model: tp.Optional[tp.Union[EasyDeLBaseModule, EasyDeLState]] = None,
		data_collator: tp.Optional[DPODataCollatorWithPadding] = None,
		train_dataset: tp.Optional[Dataset] = None,
		eval_dataset: tp.Optional[tp.Union[Dataset, tp.Dict[str, Dataset]]] = None,
		processing_class: tp.Optional[ProcessingClassType] = None,
	):
		assert arguments is not None, (
			"You Have to pass arguments that will be used for training but you have passed"
			"`arguments=None`"
		)
		assert isinstance(
			arguments, ORPOConfig
		), f"arguments type must be `ORPOConfig` but got {type(arguments)}"

		assert (
			processing_class is not None
		), "processing_class must be specified to tokenize a DPO dataset."
		self.arguments = arguments
		self.truncation_mode = arguments.truncation_mode
		self.processing_class = processing_class
		self.is_encoder_decoder = arguments.is_encoder_decoder

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
					"`padding_value` is not specified in `ORPOConfig`, and `pad_token_id` is missing in the "
					"`processing_class`. Please either set the `padding_value` argument in `ORPOConfig`, or set "
					"`tokenizer.pad_token` (e.g., `tokenizer.pad_token = tokenizer.eos_token`) before instantiating "
					"the trainer."
				)
		arguments.padding_value = self.padding_value
		input_data_collator = (
			DPODataCollatorWithPadding(
				max_prompt_length=arguments.max_prompt_length,
				max_completion_length=arguments.max_completion_length,
				pad_token_id=self.padding_value,
				label_pad_token_id=arguments.label_pad_token_id,
				is_encoder_decoder=arguments.is_encoder_decoder,
				prepadded=True,
			)
			if data_collator is None
			else data_collator
		)
		self.input_data_collator = input_data_collator

		self._stored_metrics = defaultdict(lambda: defaultdict(list))

		processing_class = processing_class

		if not isinstance(model, EasyDeLState):
			model = model.to_state()

		train_dataset = train_dataset.map(
			maybe_extract_prompt,
			num_proc=arguments.dataset_num_proc,
		)
		train_dataset = train_dataset.map(
			maybe_apply_chat_template,
			fn_kwargs={"tokenizer": processing_class},
			num_proc=arguments.dataset_num_proc,
		)
		train_dataset = train_dataset.map(
			self.tokenize_row,
			num_proc=arguments.dataset_num_proc,
		)
		if eval_dataset is not None:
			eval_dataset = eval_dataset.map(
				maybe_extract_prompt,
				num_proc=arguments.dataset_num_proc,
			)
			eval_dataset = eval_dataset.map(
				maybe_apply_chat_template,
				fn_kwargs={"tokenizer": processing_class},
				num_proc=arguments.dataset_num_proc,
			)
			eval_dataset = eval_dataset.map(
				self.tokenize_row,
				num_proc=arguments.dataset_num_proc,
			)
		self.arguments = arguments
		self.processing_class = processing_class
		super().__init__(
			model_state=model,
			arguments=arguments,
			dataset_train=train_dataset,
			dataset_eval=eval_dataset,
			data_collator=None,
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
		full_tokenized = self.processing_class(
			prompt + answer,
			add_special_tokens=False,
		)
		prompt_input_ids = self.processing_class(
			prompt,
			add_special_tokens=False,
		)["input_ids"]

		answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
		answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]
		full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])
		full_input_ids = np.array(full_tokenized["input_ids"])

		if len(full_input_ids) != len(full_concat_input_ids):
			raise ValueError(
				"Prompt input ids and answer input ids should have the same length."
			)
		response_token_ids_start_idx = len(prompt_input_ids)
		if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
			response_token_ids_start_idx -= 1

		prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
		prompt_attention_mask = full_tokenized["attention_mask"][
			:response_token_ids_start_idx
		]
		if len(prompt_input_ids) != len(prompt_attention_mask):
			raise ValueError(
				"Prompt input ids and attention mask should have the same length."
			)
		answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
		answer_attention_mask = full_tokenized["attention_mask"][
			response_token_ids_start_idx:
		]
		return dict(
			prompt_input_ids=prompt_input_ids,
			prompt_attention_mask=prompt_attention_mask,
			input_ids=answer_input_ids,
			attention_mask=answer_attention_mask,
		)

	def tokenize_row(
		self, feature: tp.Dict[str, str], state: tp.Optional[object] = None
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
		batch = {}
		prompt = feature["prompt"]
		chosen = feature["chosen"]
		rejected = feature["rejected"]

		if not self.is_encoder_decoder:
			if not isinstance(prompt, str):
				raise ValueError(f"prompt should be an str but got {type(prompt)}")
			prompt_tokens = self.processing_class(prompt, add_special_tokens=False)
			prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

			if not isinstance(chosen, str):
				raise ValueError(f"chosen should be an str but got {type(chosen)}")
			chosen_tokens = self.build_tokenized_answer(prompt, chosen)

			if not isinstance(rejected, str):
				raise ValueError(f"rejected should be an str but got {type(rejected)}")
			rejected_tokens = self.build_tokenized_answer(prompt, rejected)
			prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

			chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
			rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
			prompt_len_input_ids = min(
				chosen_prompt_len_input_ids, rejected_prompt_len_input_ids
			)

			for k, v in prompt_tokens.items():
				prompt_tokens[k] = v[:prompt_len_input_ids]
			num_diff_tokens = sum(
				[
					a != b
					for a, b in zip(
						chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"]
					)
				]
			)
			num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
			if num_diff_tokens > 1 or num_diff_len > 1:
				raise ValueError(
					"Chosen and rejected prompt_input_ids might only differ on the "
					"last token due to tokenizer merge ops."
				)
			prompt_tokens, chosen_tokens, rejected_tokens = add_bos_token_if_needed(
				self.processing_class.bos_token_id,
				prompt_len_input_ids,
				prompt_tokens,
				chosen_prompt_len_input_ids,
				chosen_tokens,
				rejected_prompt_len_input_ids,
				rejected_tokens,
			)

			chosen_tokens, rejected_tokens = add_eos_token_if_needed(
				self.processing_class.eos_token_id,
				chosen_tokens,
				rejected_tokens,
			)

			longer_response_length = max(
				len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"])
			)

			for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
				if (
					len(answer_tokens["prompt_input_ids"]) + longer_response_length
					> self.arguments.max_length
				):
					if self.truncation_mode == "keep_start":
						for k in ["prompt_input_ids", "prompt_attention_mask"]:
							answer_tokens[k] = answer_tokens[k][: self.arguments.max_prompt_length]
					elif self.truncation_mode == "keep_end":
						for k in ["prompt_input_ids", "prompt_attention_mask"]:
							answer_tokens[k] = answer_tokens[k][-self.arguments.max_prompt_length :]
					else:
						raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")
			for answer_tokens in [chosen_tokens, rejected_tokens]:
				if (
					len(answer_tokens["prompt_input_ids"]) + longer_response_length
					> self.arguments.max_length
				):
					for k in ["input_ids", "attention_mask"]:
						answer_tokens[k] = answer_tokens[k][
							: self.arguments.max_length - self.arguments.max_prompt_length
						]
			chosen_sequence_tokens = {
				k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k]
				for k in ["input_ids", "attention_mask"]
			}
			rejected_sequence_tokens = {
				k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k]
				for k in ["input_ids", "attention_mask"]
			}
			chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
			chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
				self.arguments.label_pad_token_id
			] * len(chosen_tokens["prompt_input_ids"])
			rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
			rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
				self.arguments.label_pad_token_id
			] * len(rejected_tokens["prompt_input_ids"])

			for k, toks in {
				"chosen_": chosen_sequence_tokens,
				"rejected_": rejected_sequence_tokens,
				"": prompt_tokens,
			}.items():
				for type_key, tokens in toks.items():
					if type_key == "token_type_ids":
						continue
					batch[f"{k}{type_key}"] = tokens

		else:
			chosen_tokens = self.processing_class(
				chosen,
				truncation=True,
				max_length=self.arguments.max_completion_length,
				add_special_tokens=True,
			)
			rejected_tokens = self.processing_class(
				rejected,
				truncation=True,
				max_length=self.arguments.max_completion_length,
				add_special_tokens=True,
			)
			prompt_tokens = self.processing_class(
				prompt,
				truncation=True,
				max_length=self.arguments.max_prompt_length,
				add_special_tokens=True,
			)

			batch["chosen_labels"] = chosen_tokens["input_ids"]
			batch["rejected_labels"] = rejected_tokens["input_ids"]
			batch["prompt_input_ids"] = prompt_tokens["input_ids"]
			batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

			if state is not None and hasattr(
				state.model,
				"prepare_decoder_input_ids_from_labels",
			):
				model = state.model
				batch["rejected_decoder_input_ids"] = (
					model.prepare_decoder_input_ids_from_labels(
						labels=jnp.asarray(batch["rejected_labels"])
					)
				)
				batch["chosen_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
					labels=jnp.asarray(batch["chosen_labels"])
				)

		for k in batch:
			if "labels" in k or self.is_encoder_decoder:
				pad_value = self.arguments.label_pad_token_id
			elif k.endswith("_input_ids"):
				pad_value = self.padding_value
			elif k.endswith("_attention_mask"):
				pad_value = 0
			batch[k] = batch[k] + [pad_value] * (self.arguments.max_length - len(batch[k]))
		return batch

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
			max_length=self.arguments.max_length,
		)
		jited_concatenated_forward = jax.jit(
			partial_concatenated_forward,
			static_argnames=[
				"is_encoder_decoder",
				"padding_value",
				"label_pad_token_id",
				"max_length",
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
			donate_argnums=(0,),
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
			out_shardings=empty_sharding,
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
