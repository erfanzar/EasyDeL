import warnings
from typing import Union, Optional, Callable, Dict
from abc import ABC

from datasets.arrow_writer import SchemaInferenceError
from datasets.builder import DatasetGenerationError
from transformers import PreTrainedTokenizerBase
from datasets import Dataset
from ..base_trainer import TrainerConfigureDataloaderFuncOutput
from ..training_configurations import TrainArguments
from ..utils import (
    get_formatting_func_from_dataset,
    create_constant_length_dataset
)
from ..causal_language_model_trainer import CausalLanguageModelTrainer
from ...etils.etils import get_logger

import tensorflow_datasets as tfds

logger = get_logger(__name__)


class SFTTrainer(CausalLanguageModelTrainer, ABC):

    def __init__(
            self,
            arguments: TrainArguments,
            tokenizer: PreTrainedTokenizerBase,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            dataset_text_field: Optional[str] = None,
            packing: Optional[bool] = False,
            formatting_func: Optional[Callable] = None,
            num_of_sequences: Optional[int] = 1024,
            chars_per_token: Optional[float] = 3.6,
            dataset_num_proc: Optional[int] = None,
            dataset_batch_size: int = 1000,
            neftune_noise_alpha: Optional[float] = None,
            dataset_kwargs: Optional[Dict] = None,
            eval_packing: Optional[bool] = None,
            checkpoint_path: Optional[str] = None,
            remove_unused_columns=True,
            _do_init_fns: bool = True
    ):

        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.dataset_num_proc = dataset_num_proc
        self.dataset_batch_size = dataset_batch_size

        self._trainer_supports_neftune = hasattr(arguments, "neftune_noise_alpha")

        if neftune_noise_alpha is not None and self._trainer_supports_neftune:
            arguments.neftune_noise_alpha = neftune_noise_alpha
            warnings.warn(
                "You passed a `neftune_noise_alpha` argument to the SFTTrainer, the value you passed will override "
                "the one in the `TrainArguments`."
            )
        elif not self._trainer_supports_neftune:
            self.neftune_noise_alpha = neftune_noise_alpha

        if formatting_func is None and dataset_text_field is None:
            formatting_func = get_formatting_func_from_dataset(train_dataset, tokenizer)  # type: ignore

        if not packing:
            if dataset_text_field is None and formatting_func is None:
                raise ValueError(
                    "You passed `packing=False` to the SFTTrainer, but you didn't pass a "
                    "`dataset_text_field` or `formatting_func` argument."
                )

        if dataset_kwargs is None:
            dataset_kwargs = {}
        if train_dataset is not None:
            train_dataset = self._prepare_dataset(
                train_dataset,
                tokenizer,
                packing,
                dataset_text_field,
                arguments.max_sequence_length,
                formatting_func,
                num_of_sequences,
                chars_per_token,
                remove_unused_columns=remove_unused_columns,
                **dataset_kwargs,
            )
        if eval_dataset is not None:
            _multiple = isinstance(eval_dataset, dict)
            _eval_datasets = eval_dataset if _multiple else {"singleton": eval_dataset}

            eval_packing = packing if eval_packing is None else eval_packing

            for _eval_dataset_name, _eval_dataset in _eval_datasets.items():
                _eval_datasets[_eval_dataset_name] = self._prepare_dataset(
                    _eval_dataset,
                    tokenizer,
                    eval_packing,
                    dataset_text_field,
                    arguments.max_sequence_length,
                    formatting_func,
                    num_of_sequences,
                    chars_per_token,
                    remove_unused_columns=remove_unused_columns,
                    **dataset_kwargs,
                )
            if not _multiple:
                eval_dataset = _eval_datasets["singleton"]
        if tokenizer.padding_side is not None and tokenizer.padding_side != "right":
            warnings.warn(
                "You passed a tokenizer with `padding_side` not equal to `right` to the SFTTrainer. This might lead "
                "to some unexpected behaviour due to overflow issues when training a model in half-precision. "
                "You might consider adding `tokenizer.padding_side = 'right'` to your code."
            )

        super().__init__(
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            finetune=True,
            checkpoint_path=checkpoint_path,
            _do_init_fns=_do_init_fns,
        )

    def configure_dataloader(self) -> TrainerConfigureDataloaderFuncOutput:

        """
        The configure_dataloader function is used to configure the dataloader for training and evaluation.

        :param self: Refer to the class instance itself
        :return: A TrainerConfigureDataloaderFuncOutput object

        """

        dataloader_train = tfds.as_numpy(
            self.dataset_train.to_tf_dataset(
                batch_size=self.arguments.total_batch_size,
                drop_remainder=True,
                num_workers=self.arguments.dataloader_num_workers,
                collate_fn=self.create_collate_function(
                    max_sequence_length=self.arguments.max_sequence_length,
                    truncation_mode=self.arguments.truncation_mode
                )
            )
        )
        max_training_steps = self.arguments.num_train_epochs * len(
            dataloader_train
        ) if self.arguments.max_training_steps is None else self.arguments.max_training_steps
        if self.dataset_eval is not None and self.arguments.do_eval:
            dataloader_eval = tfds.as_numpy(
                self.dataset_eval.to_tf_dataset(
                    batch_size=self.arguments.total_batch_size,
                    drop_remainder=True,
                    shuffle=True,
                    num_workers=self.arguments.dataloader_num_workers,
                    collate_fn=self.create_collate_function(
                        max_sequence_length=self.arguments.max_sequence_length,
                        truncation_mode=self.arguments.truncation_mode
                    )
                )
            )
            max_evaluation_steps = len(
                dataloader_eval
            ) if self.arguments.max_training_steps is None else self.arguments.max_training_steps
        else:
            dataloader_eval, max_evaluation_steps = None, 0

        return TrainerConfigureDataloaderFuncOutput(
            dataloader_train=dataloader_train,
            max_training_steps=max_training_steps,
            dataloader_eval=dataloader_eval,
            max_evaluation_steps=max_evaluation_steps
        )

    def _prepare_dataset(
            self,
            dataset,
            tokenizer,
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
        if dataset is None:
            raise ValueError("The dataset should not be None")

        if not packing:
            return self._prepare_non_packed_dataloader(
                tokenizer,
                dataset,
                dataset_text_field,
                max_seq_length,
                formatting_func,
                add_special_tokens,
                remove_unused_columns,
            )

        else:
            return self._prepare_packed_dataloader(
                tokenizer,
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
            tokenizer,
            dataset,
            dataset_text_field,
            max_seq_length,
            formatting_func=None,
            add_special_tokens=True,
            remove_unused_columns=True,
    ):
        use_formatting_func = formatting_func is not None and dataset_text_field is None
        self._dataset_sanity_checked = False

        def tokenize(element):
            inner = element[dataset_text_field] if not use_formatting_func else formatting_func(element)
            outputs = tokenizer(
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

            return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

        signature_columns = ["input_ids", "labels", "attention_mask"]

        extra_columns = list(set(dataset.column_names) - set(signature_columns))

        if not remove_unused_columns and len(extra_columns) > 0:
            warnings.warn(
                "You passed `remove_unused_columns=False` on a non-packed dataset. This might create some issues with "
                "the default collator and yield to errors. If you want to inspect dataset other columns "
                f"(in this case {extra_columns}), you can subclass `DataCollatorForLanguageModeling` in case you "
                "used the default collator and create your own data collator in order to inspect the "
                "unused dataset columns."
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
            tokenizer,
            dataset,
            dataset_text_field,
            max_seq_length,
            num_of_sequences,
            chars_per_token,
            formatting_func=None,
            append_concat_token=True,
            add_special_tokens=True,
    ):
        if dataset_text_field is not None or formatting_func is not None:
            if tokenizer is None:
                raise ValueError(
                    "You need to pass a tokenizer when using `dataset_text_field` with `SFTTrainer`."
                )

            constant_length_iterator = create_constant_length_dataset(
                tokenizer=tokenizer,
                dataset=dataset,
                dataset_text_field=dataset_text_field,
                formatting_func=formatting_func,
                seq_length=max_seq_length,
                infinite=False,
                num_of_sequences=num_of_sequences,
                chars_per_token=chars_per_token,
                eos_token_id=tokenizer.eos_token_id,
                append_concat_token=append_concat_token,
                add_special_tokens=add_special_tokens,
            )

            def data_generator(inner_constant_length_iterator):
                for d in inner_constant_length_iterator():
                    yield d

            try:
                packed_dataset = Dataset.from_generator(
                    data_generator, gen_kwargs={"inner_constant_length_iterator": constant_length_iterator}
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
