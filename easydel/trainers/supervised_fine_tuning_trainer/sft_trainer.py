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
import warnings

from easydel.etils.etils import get_logger
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.utils import ProcessingClassType

from ..base_trainer import TrainerConfigureDataloaderOutput
from ..trainer import Trainer
from ..utils import (
	create_constant_length_dataset,
	get_formatting_func_from_dataset,
)
from .sft_config import SFTConfig

if tp.TYPE_CHECKING:
	from datasets import Dataset
else:
	Dataset = tp.Any
logger = get_logger(__name__)


class SFTTrainer(Trainer):
	"""
	Trainer class for Supervised Fine-Tuning (SFT) of language models.

	This trainer extends the `Trainer` and provides functionalities
	specific to supervised fine-tuning tasks.
	"""

	def __init__(
		self,
		arguments: SFTConfig,
		processing_class: ProcessingClassType,
		model: tp.Optional[EasyDeLBaseModule] = None,
		train_dataset: tp.Optional[Dataset] = None,
		eval_dataset: tp.Optional[tp.Union[Dataset, tp.Dict[str, Dataset]]] = None,
		formatting_func: tp.Optional[tp.Callable] = None,
		_do_init_fns: bool = True,
	):
		if getattr(processing_class, "pad_token", None) is None:
			processing_class.pad_token = processing_class.eos_token
		assert isinstance(arguments, SFTConfig), "passed argument must be a `SFTConfig`."

		if formatting_func is None and arguments.dataset_text_field is None:
			formatting_func = get_formatting_func_from_dataset(
				train_dataset, processing_class
			)  # type: ignore

		if not arguments.packing:
			if arguments.dataset_text_field is None and formatting_func is None:
				raise ValueError(
					"You passed `packing=False` to the SFTTrainer, but you didn't pass a "
					"`dataset_text_field` or `formatting_func` argument."
				)

		if arguments.dataset_kwargs is None:
			arguments.dataset_kwargs = {}
		if train_dataset is not None:
			train_dataset = self._prepare_dataset(
				train_dataset,
				processing_class,
				arguments.packing,
				arguments.dataset_text_field,
				arguments.max_sequence_length,
				formatting_func,
				arguments.num_of_sequences,
				arguments.chars_per_token,
				remove_unused_columns=arguments.remove_unused_columns,
				**arguments.dataset_kwargs,
			)
		if eval_dataset is not None:
			_multiple = isinstance(eval_dataset, dict)
			_eval_datasets = eval_dataset if _multiple else {"singleton": eval_dataset}

			eval_packing = (
				arguments.packing if arguments.eval_packing is None else arguments.eval_packing
			)

			for _eval_dataset_name, _eval_dataset in _eval_datasets.items():
				_eval_datasets[_eval_dataset_name] = self._prepare_dataset(
					_eval_dataset,
					processing_class,
					eval_packing,
					arguments.dataset_text_field,
					arguments.max_sequence_length,
					formatting_func,
					arguments.num_of_sequences,
					arguments.chars_per_token,
					remove_unused_columns=arguments.remove_unused_columns,
					**arguments.dataset_kwargs,
				)
			if not _multiple:
				eval_dataset = _eval_datasets["singleton"]
		if (
			processing_class.padding_side is not None
			and processing_class.padding_side != "right"
		):
			warnings.warn(
				"You passed a processing_class with `padding_side` not equal to `right` to the SFTTrainer. This might lead "
				"to some unexpected behaviour due to overflow issues when training a model in half-precision. "
				"You might consider adding `processing_class.padding_side = 'right'` to your code.",
				stacklevel=1,
			)

		super().__init__(
			arguments=arguments,
			dataset_train=train_dataset,
			dataset_eval=eval_dataset,
			model=model,
			_do_init_fns=_do_init_fns,
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

		import tensorflow_datasets as tfds

		dataloader_train = tfds.as_numpy(
			self.dataset_train.to_tf_dataset(
				batch_size=self.arguments.total_batch_size
				* self.arguments.gradient_accumulation_steps,
				drop_remainder=True,
				num_workers=self.arguments.dataloader_num_workers,
				collate_fn=self.create_collect_function(
					max_sequence_length=self.arguments.max_sequence_length,
					truncation_mode=self.arguments.truncation_mode,
				),
			)
		)
		max_training_steps = (
			self.arguments.num_train_epochs * len(dataloader_train)
			if self.arguments.max_training_steps is None
			else self.arguments.max_training_steps
		)
		if self.dataset_eval is not None and self.arguments.do_eval:
			dataloader_eval = tfds.as_numpy(
				self.dataset_eval.to_tf_dataset(
					batch_size=self.arguments.eval_batch_size,
					drop_remainder=True,
					shuffle=True,
					num_workers=self.arguments.dataloader_num_workers,
					collate_fn=self.create_collect_function(
						max_sequence_length=self.arguments.max_sequence_length,
						truncation_mode=self.arguments.truncation_mode,
					),
				)
			)
			max_evaluation_steps = (
				len(dataloader_eval)
				if self.arguments.max_training_steps is None
				else self.arguments.max_training_steps
			)
		else:
			dataloader_eval, max_evaluation_steps = None, 0

		return TrainerConfigureDataloaderOutput(
			dataloader_train=dataloader_train,
			max_training_steps=max_training_steps,
			dataloader_eval=dataloader_eval,
			max_evaluation_steps=max_evaluation_steps,
		)

	def _prepare_dataset(
		self,
		dataset,
		processing_class,
		packing,
		dataset_text_field,
		max_seq_length,
		formatting_func,
		num_of_sequences,
		chars_per_token,
		remove_unused_columns=True,
		append_concat_token=True,
		add_special_tokens=True,
	):
		"""
		Prepares the dataset for training by applying tokenization and packing (if enabled).

		Args:
		    dataset (Dataset): The dataset to prepare.
		    processing_class (ProcessingClassType): The processing_class to use.
		    packing (bool): Whether to pack multiple sequences into a single sample.
		    dataset_text_field (str): The name of the text field in the dataset.
		    max_seq_length (int): The maximum sequence length.
		    formatting_func (tp.Callable): A formatting function to apply to each sample.
		    num_of_sequences (int): Number of sequences to pack in each sample (if packing is enabled).
		    chars_per_token (float): Average number of characters per token.
		    remove_unused_columns (bool, optional): Whether to remove unused columns. Defaults to True.
		    append_concat_token (bool, optional): Whether to append a concat token for packing. Defaults to True.
		    add_special_tokens (bool, optional): Whether to add special tokens during tokenization. Defaults to True.

		Returns:
		    Dataset: The processed dataset ready for training.

		Raises:
		    ValueError: If the dataset is None or if packing is enabled without a `dataset_text_field` or `formatting_func`.
		"""
		if dataset is None:
			raise ValueError("The dataset should not be None")

		if not packing:
			return self._prepare_non_packed_dataloader(
				processing_class,
				dataset,
				dataset_text_field,
				max_seq_length,
				formatting_func,
				add_special_tokens,
				remove_unused_columns,
			)

		else:
			return self._prepare_packed_dataloader(
				processing_class,
				dataset,
				dataset_text_field,
				max_seq_length,
				num_of_sequences,
				chars_per_token,
				formatting_func,
				append_concat_token,
				add_special_tokens,
			)

	def _prepare_non_packed_dataloader(
		self,
		processing_class,
		dataset,
		dataset_text_field,
		max_seq_length,
		formatting_func=None,
		add_special_tokens=True,
		remove_unused_columns=True,
	):
		"""
		Prepares a non-packed dataloader from the given dataset.

		This method tokenizes the text data in the dataset, truncates or pads sequences to a fixed length,
		and removes unused columns as specified. It's suitable for datasets where each sample represents
		a single sequence.

		Args:
		    processing_class: The processing_class to use for text encoding.
		    dataset (Dataset): The dataset to prepare.
		    dataset_text_field (str): The name of the text field in the dataset.
		    max_seq_length (int): The maximum sequence length.
		    formatting_func (tp.Callable, optional): A formatting function to apply to each sample before tokenization.
		        Defaults to None.
		    add_special_tokens (bool, optional): Whether to add special tokens during tokenization. Defaults to True.
		    remove_unused_columns (bool, optional): Whether to remove unused columns from the dataset. Defaults to True.

		Returns:
		    Dataset: The processed dataset ready for training.
		"""
		use_formatting_func = formatting_func is not None and dataset_text_field is None
		self._dataset_sanity_checked = False

		def tokenize(element):
			inner = (
				element[dataset_text_field]
				if not use_formatting_func
				else formatting_func(element)
			)
			outputs = processing_class(
				inner,
				add_special_tokens=add_special_tokens,
				truncation=True,
				padding="max_length",
				max_length=max_seq_length,
				return_overflowing_tokens=False,
				return_length=False,
			)

			if use_formatting_func and not self._dataset_sanity_checked:
				if not isinstance(formatting_func(element), list):
					raise ValueError(
						"The `formatting_func` should return a list of processed strings since it can lead"
						" to silent bugs."
					)
				else:
					self._dataset_sanity_checked = True

			return {
				"input_ids": outputs["input_ids"],
				"attention_mask": outputs["attention_mask"],
			}

		signature_columns = ["input_ids", "labels", "attention_mask"]

		extra_columns = list(set(dataset.column_names) - set(signature_columns))

		if not remove_unused_columns and len(extra_columns) > 0:
			warnings.warn(
				"You passed `remove_unused_columns=False` on a non-packed dataset. This might create some issues with "
				"the default collator and yield to errors. If you want to inspect dataset other columns "
				f"(in this case {extra_columns}), you can subclass `DataCollatorForLanguageModeling` in case you "
				"used the default collator and create your own data collator in order to inspect the "
				"unused dataset columns.",
				stacklevel=1,
			)

		tokenized_dataset = dataset.map(
			tokenize,
			batched=False,
			remove_columns=dataset.column_names if remove_unused_columns else None,
			num_proc=self.dataset_num_proc,
			batch_size=self.dataset_batch_size,
		)

		return tokenized_dataset

	@staticmethod
	def _prepare_packed_dataloader(
		processing_class,
		dataset,
		dataset_text_field,
		max_seq_length,
		num_of_sequences,
		chars_per_token,
		formatting_func=None,
		append_concat_token=True,
		add_special_tokens=True,
	):
		"""
		Prepares a packed dataloader from the given dataset.

		This method is designed for efficient training of language models by packing multiple
		sequences from the dataset into a single sample. This can be particularly beneficial
		for handling long sequences and optimizing GPU/TPU utilization.

		Args:
		    processing_class: The processing_class used for text encoding.
		    dataset (Dataset): The dataset to prepare.
		    dataset_text_field (str): The name of the text field in the dataset.
		    max_seq_length (int): The maximum length of each packed sequence.
		    num_of_sequences (int): The number of sequences to pack into a single sample.
		    chars_per_token (float): The average number of characters per token, used for estimating
		        the number of tokens in a text sequence.
		    formatting_func (tp.Callable, optional): A function to format each sample from the dataset
		        before packing. It should take a sample as input and return a dictionary with a "text"
		        key containing the processed text. Defaults to None.
		    append_concat_token (bool, optional): Whether to append a special concatenation token
		        between packed sequences. Defaults to True.
		    add_special_tokens (bool, optional): Whether to add special tokens (like BOS, EOS)
		        during tokenization. Defaults to True.

		Returns:
		    Dataset: The processed dataset with packed sequences.

		Raises:
		    ValueError: If both `dataset_text_field` and `formatting_func` are None, or if there's
		        an error during dataset packing.
		"""
		if dataset_text_field is not None or formatting_func is not None:
			if processing_class is None:
				raise ValueError(
					"You need to pass a processing_class when using `dataset_text_field` with `SFTTrainer`."
				)

			constant_length_iterator = create_constant_length_dataset(
				processing_class=processing_class,
				dataset=dataset,
				dataset_text_field=dataset_text_field,
				formatting_func=formatting_func,
				seq_length=max_seq_length,
				infinite=False,
				num_of_sequences=num_of_sequences,
				chars_per_token=chars_per_token,
				eos_token_id=processing_class.eos_token_id,
				append_concat_token=append_concat_token,
				add_special_tokens=add_special_tokens,
			)

			def data_generator(inner_constant_length_iterator):
				for d in inner_constant_length_iterator():
					yield d

			# Import Only and Only when needed, don't dst the runtime.
			from datasets import Dataset
			from datasets.arrow_writer import SchemaInferenceError
			from datasets.builder import DatasetGenerationError

			try:
				packed_dataset = Dataset.from_generator(
					data_generator,
					gen_kwargs={"inner_constant_length_iterator": constant_length_iterator},
				)
			except (DatasetGenerationError, SchemaInferenceError) as exc:
				raise ValueError(
					"Error occurred while packing the dataset. "
					"Make sure that your dataset has enough samples to at least yield one packed sequence.\n"
					"External Information : {}".format(exc)
				) from exc
			return packed_dataset
		else:
			raise ValueError(
				"You need to pass a `dataset_text_field` or `formatting_func` argument to the SFTTrainer if you want "
				"to use the `ConstantLengthDataset`."
			)
