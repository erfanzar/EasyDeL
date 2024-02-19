import copy
import os
import sys
import time
import typing
import warnings
from abc import ABC
from collections import defaultdict

import IPython
import chex
import flax.core
import jax
import termcolor
import torch
import wandb
from fjformer import match_partition_rules, make_shard_and_gather_fns
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.collectors import DPODataCollatorWithPadding
from typing import Optional, Literal, Dict, Union, Any, Tuple, List, Callable, Mapping
from .utils import pad_to_length
from jax.experimental.pjit import pjit
from datasets import Dataset
from jax import numpy as jnp

from ...smi import get_capacity_matrix, initialise_tracking, get_mem
from ...trainer.causal_language_model_trainer import TrainerOutput
from ...trainer.training_configurations import TrainArguments
from ...trainer.base_trainer import (
    BaseTrainer,
    TrainerConfigureFunctionFuncOutput,
    TrainerConfigureDataloaderFuncOutput,
    TrainerConfigureModelFuncOutput
)
from ...etils import EasyDelState, EasyDelTimerError
from transformers import PreTrainedTokenizerBase
from .partitioner_config import PartitionerConfig
from jax.sharding import PartitionSpec

from ...utils import Timers
from flax.struct import dataclass


class SFTTrainer(BaseTrainer):
    _tag_names = ["trl", "sft"]

    def __init__(
            self,
            model: Union[EasyDelState, str],
            args: TrainArguments,
            train_dataset: Dataset,
            data_collator: Optional[Callable] = None,

            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            dataset_text_field: Optional[str] = None,
            packing: Optional[bool] = False,
            formatting_func: Optional[Callable] = None,
            max_seq_length: Optional[int] = None,
            infinite: Optional[bool] = None,
            num_of_sequences: Optional[int] = 1024,
            chars_per_token: Optional[float] = 3.6,
            dataset_num_proc: Optional[int] = None,
            dataset_batch_size: int = 1000,
            neftune_noise_alpha: Optional[float] = None,
            model_init_kwargs: Optional[Dict] = None,
            dataset_kwargs: Optional[Dict] = None,
    ):
        ...

    def train(self, *args, **kwargs):
        ...

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

        # check if torch dataset / dataloader and do nothing
        if isinstance(
                dataset, (
                        torch.utils.data.IterableDataset,
                        torch.utils.data.Dataset,
                )
        ):
            return dataset

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
            outputs = tokenizer(
                element[dataset_text_field] if not use_formatting_func else formatting_func(element),
                add_special_tokens=add_special_tokens,
                truncation=True,
                padding=False,
                max_length=max_seq_length,
                return_overflowing_tokens=False,
                return_length=False,
            )

            if use_formatting_func and not self._dataset_sanity_checked:
                if not isinstance(formatting_func(element), list):
                    raise ValueError(
                        "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                    )
                else:
                    self._dataset_sanity_checked = True

            return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

        signature_columns = ["input_ids", "labels", "attention_mask"]

        extra_columns = list(set(dataset.column_names) - set(signature_columns))

        if not remove_unused_columns and len(extra_columns) > 0:
            warnings.warn(
                "You passed `remove_unused_columns=False` on a non-packed dataset. This might create some issues with the default collator and yield to errors. If you want to "
                f"inspect dataset other columns (in this case {extra_columns}), you can subclass `DataCollatorForLanguageModeling` in case you used the default collator and create your own data collator in order to inspect the unused dataset columns."
            )

        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names if remove_unused_columns else None,
            num_proc=self.dataset_num_proc,
            batch_size=self.dataset_batch_size,
        )

        return tokenized_dataset

    def _prepare_packed_dataloader(
            self,
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
                raise ValueError("You need to pass a tokenizer when using `dataset_text_field` with `SFTTrainer`.")

            constant_length_iterator = ConstantLengthDataset(
                tokenizer,
                dataset,
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

            def data_generator(constant_length_iterator):
                yield from constant_length_iterator

            try:
                packed_dataset = Dataset.from_generator(
                    data_generator, gen_kwargs={"constant_length_iterator": constant_length_iterator}
                )
            except (DatasetGenerationError, SchemaInferenceError) as exc:
                raise ValueError(
                    "Error occurred while packing the dataset. "
                    "Make sure that your dataset has enough samples to at least yield one packed sequence."
                ) from exc
            return packed_dataset
        else:
            raise ValueError(
                "You need to pass a `dataset_text_field` or `formatting_func` argument to the SFTTrainer if you want to use the `ConstantLengthDataset`."
            )

    def _trl_activate_neftune(self, model):

        unwrapped_model = unwrap_model(model)
        if is_peft_available() and isinstance(unwrapped_model, PeftModel):
            embeddings = unwrapped_model.base_model.model.get_input_embeddings()
        else:
            embeddings = unwrapped_model.get_input_embeddings()

        embeddings.neftune_noise_alpha = self.neftune_noise_alpha
        hook_handle = embeddings.register_forward_hook(neftune_post_forward_hook)
        self.neftune_hook_handle = hook_handle
        return model
