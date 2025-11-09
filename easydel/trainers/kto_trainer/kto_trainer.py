# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

import jax
import numpy as np
from datasets import Dataset, IterableDataset
from eformer.loggings import get_logger
from jax.sharding import PartitionSpec

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry
from easydel.utils.compiling_utils import ejit
from easydel.utils.traversals import deepcopy_model

from ..binary_classifier_optimization_trainer._fn import concatenated_forward
from ..prompt_utils import maybe_apply_chat_template, maybe_extract_prompt, maybe_unpair_preference_dataset
from ..trainer.trainer import Trainer
from ..trainer_protocol import TrainerConfigureFunctionOutput
from ..utils import BCODataCollatorGrain, BCODataCollatorTFDS
from ._fn import evaluation_step, training_step
from .kto_config import KTOConfig

logger = get_logger(__name__)


def _tokenize(
    batch: dict[str, list[tp.Any]],
    tokenizer,
) -> dict[str, list[tp.Any]]:
    """Tokenize prompts and completions separately.

    Args:
        batch: Dictionary containing 'prompt' and 'completion' lists.
        tokenizer: Tokenizer to use for encoding text.

    Returns:
        Dictionary with separate token IDs and attention masks for prompts and answers.
    """
    prompt_tokenized = tokenizer(batch["prompt"], add_special_tokens=False)
    prompt_input_ids = prompt_tokenized["input_ids"]
    prompt_attention_mask = prompt_tokenized["attention_mask"]

    prompt_and_completion = [
        prompt + completion for prompt, completion in zip(batch["prompt"], batch["completion"], strict=True)
    ]
    full_tokenized = tokenizer(prompt_and_completion, add_special_tokens=False)
    full_input_ids = full_tokenized["input_ids"]
    full_attention_mask = full_tokenized["attention_mask"]

    answer_input_ids = [f[len(p) :] for f, p in zip(full_input_ids, prompt_input_ids, strict=True)]
    answer_attention_mask = [f[len(p) :] for f, p in zip(full_attention_mask, prompt_attention_mask, strict=True)]

    full_input_ids = [np.asarray(f) for f in full_input_ids]

    response_token_ids_start_idx = [len(p) for p in prompt_input_ids]
    for idx, (prompt_ids, full_ids, start_idx) in enumerate(
        zip(prompt_input_ids, full_input_ids, response_token_ids_start_idx, strict=True)
    ):
        if not np.array_equal(prompt_ids, full_ids[:start_idx]):
            response_token_ids_start_idx[idx] -= 1

    prompt_input_ids = [f[:r] for f, r in zip(full_input_ids, response_token_ids_start_idx, strict=True)]
    prompt_attention_mask = [f[:r] for f, r in zip(full_attention_mask, response_token_ids_start_idx, strict=True)]
    answer_input_ids = [f[r:] for f, r in zip(full_input_ids, response_token_ids_start_idx, strict=True)]
    answer_attention_mask = [f[r:] for f, r in zip(full_attention_mask, response_token_ids_start_idx, strict=True)]

    return dict(
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        answer_input_ids=answer_input_ids,
        answer_attention_mask=answer_attention_mask,
    )


def _process_tokens(
    example: dict[str, tp.Any],
    *,
    prefix: str,
    tokenizer,
    is_encoder_decoder: bool,
    max_length: int,
    truncation_mode: tp.Literal["keep_end", "keep_start"],
    label_pad_token_id: int,
    max_prompt_length: int | None,
    max_completion_length: int | None,
) -> dict[str, tp.Any]:
    """Process and truncate tokenized sequences to fit length constraints.

    Args:
        example: Single example containing prompt and answer tokens.
        prefix: Prefix to add to output keys.
        tokenizer: Tokenizer for special token IDs.
        is_encoder_decoder: Whether the model uses encoder-decoder architecture.
        max_length: Maximum total sequence length.
        truncation_mode: How to truncate sequences ('keep_end' or 'keep_start').
        label_pad_token_id: Token ID used for padding labels.
        max_prompt_length: Maximum prompt length.
        max_completion_length: Maximum completion length.

    Returns:
        Dictionary with processed prompt and completion token sequences.
    """
    prompt_tokens = {
        "prompt_input_ids": example["prompt_input_ids"],
        "prompt_attention_mask": example["prompt_attention_mask"],
        "answer_input_ids": example["answer_input_ids"],
        "answer_attention_mask": example["answer_attention_mask"],
    }

    full_len = len(prompt_tokens["prompt_input_ids"]) + len(prompt_tokens["answer_input_ids"])

    available_length = max_length
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id

    if bos_token_id is not None and (
        len(prompt_tokens["prompt_input_ids"]) == 0 or prompt_tokens["prompt_input_ids"][0] != bos_token_id
    ):
        available_length -= 1
    if eos_token_id is not None and (
        len(prompt_tokens["answer_input_ids"]) == 0 or prompt_tokens["answer_input_ids"][-1] != eos_token_id
    ):
        available_length -= 1

    if max_prompt_length is not None and len(prompt_tokens["prompt_input_ids"]) > max_prompt_length:
        if truncation_mode == "keep_start":
            prompt_tokens["prompt_input_ids"] = prompt_tokens["prompt_input_ids"][:max_prompt_length]
            prompt_tokens["prompt_attention_mask"] = prompt_tokens["prompt_attention_mask"][:max_prompt_length]
        elif truncation_mode == "keep_end":
            prompt_tokens["prompt_input_ids"] = prompt_tokens["prompt_input_ids"][-max_prompt_length:]
            prompt_tokens["prompt_attention_mask"] = prompt_tokens["prompt_attention_mask"][-max_prompt_length:]
        else:
            raise ValueError(f"Unsupported truncation mode: {truncation_mode}")

    full_len = len(prompt_tokens["prompt_input_ids"]) + len(prompt_tokens["answer_input_ids"])
    if full_len > available_length:
        keep = max(available_length - len(prompt_tokens["prompt_input_ids"]), 0)
        prompt_tokens["answer_input_ids"] = prompt_tokens["answer_input_ids"][:keep]
        prompt_tokens["answer_attention_mask"] = prompt_tokens["answer_attention_mask"][:keep]

    completion_ids = prompt_tokens["prompt_input_ids"] + prompt_tokens["answer_input_ids"]
    completion_attention = prompt_tokens["prompt_attention_mask"] + prompt_tokens["answer_attention_mask"]

    if bos_token_id is not None:
        if len(prompt_tokens["prompt_input_ids"]) == 0 or prompt_tokens["prompt_input_ids"][0] != bos_token_id:
            completion_ids = [bos_token_id, *completion_ids]
            completion_attention = [1, *completion_attention]
            prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens["prompt_input_ids"]
            prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]

    if eos_token_id is not None and (
        len(prompt_tokens["answer_input_ids"]) == 0 or prompt_tokens["answer_input_ids"][-1] != eos_token_id
    ):
        completion_ids = [*completion_ids, eos_token_id]
        completion_attention = [*completion_attention, 1]

    completion_labels = completion_ids[:]
    completion_labels[: len(prompt_tokens["prompt_input_ids"])] = [label_pad_token_id] * len(
        prompt_tokens["prompt_input_ids"]
    )

    output = {
        f"{prefix}prompt_input_ids": np.asarray(prompt_tokens["prompt_input_ids"], dtype=np.int32),
        f"{prefix}prompt_attention_mask": np.asarray(prompt_tokens["prompt_attention_mask"], dtype=np.int32),
        f"{prefix}completion_input_ids": np.asarray(completion_ids, dtype=np.int32),
        f"{prefix}completion_attention_mask": np.asarray(completion_attention, dtype=np.int32),
        f"{prefix}completion_labels": np.asarray(completion_labels, dtype=np.int32),
    }

    if is_encoder_decoder and "completion_decoder_input_ids" in example:
        output[f"{prefix}completion_decoder_input_ids"] = np.asarray(
            example["completion_decoder_input_ids"], dtype=np.int32
        )

    if prefix == "" and "label" in example:
        output["label"] = bool(example["label"])

    return output


@Registry.register("trainer", "kto")
class KTOTrainer(Trainer):
    """Kahneman-Tversky Optimization trainer.

    Implements KTO training which uses human binary feedback (desirable/undesirable)
    to align language models. Unlike DPO which requires paired preferences, KTO
    works with unpaired binary labels.

    Args:
        arguments: KTO-specific training configuration.
        model: Policy model to train (EasyDeLBaseModule or EasyDeLState).
        reference_model: Reference model for KL penalty computation.
        processing_class: Tokenizer or processor for text encoding.
        train_dataset: Training dataset with prompt, completion, and label fields.
        eval_dataset: Optional evaluation dataset.
        data_collator: Optional custom data collator.
    """

    arguments: KTOConfig

    def __init__(
        self,
        arguments: KTOConfig,
        model: EasyDeLBaseModule | EasyDeLState | None,
        reference_model: EasyDeLBaseModule | EasyDeLState | None = None,
        processing_class: ProcessingClassType | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | IterableDataset | None = None,
        data_collator: BCODataCollatorTFDS | BCODataCollatorGrain | None = None,
    ):
        if not isinstance(arguments, KTOConfig):
            raise TypeError(f"`arguments` must be a `KTOConfig`, received {type(arguments)}")
        if processing_class is None:
            raise ValueError("`processing_class` must be provided for KTO training.")
        if model is None:
            raise ValueError("`model` must be supplied to the KTO trainer.")
        if train_dataset is None:
            raise ValueError("`train_dataset` must be provided for KTOTrainer.")

        if getattr(processing_class, "pad_token_id", None) is None and hasattr(processing_class, "eos_token"):
            processing_class.pad_token = processing_class.eos_token

        if isinstance(model, EasyDeLState):
            model_state = model
        else:
            model_state = model.to_state()

        if reference_model is None:
            reference_state = deepcopy_model(model_state)
        elif isinstance(reference_model, EasyDeLState):
            reference_state = reference_model
        elif isinstance(reference_model, EasyDeLBaseModule):
            reference_state = reference_model.to_state()
        else:
            reference_state = deepcopy_model(model_state)

        self.arguments = arguments
        if self.arguments.is_encoder_decoder is not None:
            self.is_encoder_decoder = self.arguments.is_encoder_decoder
        else:
            self.is_encoder_decoder = getattr(model_state.model.config, "is_encoder_decoder", False)
            self.arguments.is_encoder_decoder = self.is_encoder_decoder

        if arguments.padding_value is not None:
            self.padding_value = arguments.padding_value
        else:
            pad_token_id = getattr(processing_class, "pad_token_id", None)
            if pad_token_id is None and hasattr(processing_class, "tokenizer"):
                pad_token_id = getattr(processing_class.tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                raise ValueError("Tokenizer must expose a pad token or `padding_value` must be specified.")
            self.padding_value = pad_token_id
            self.arguments.padding_value = pad_token_id

        if arguments.max_completion_length is None and arguments.max_length is not None:
            arguments.max_completion_length = max(arguments.max_length - arguments.max_prompt_length, 1)

        if data_collator is None:
            input_data_collator_tfds = BCODataCollatorTFDS(
                max_prompt_length=arguments.max_prompt_length,
                max_completion_length=arguments.max_completion_length,
                pad_token_id=self.padding_value,
                label_pad_token_id=arguments.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )
            input_data_collator_grain = BCODataCollatorGrain(
                max_prompt_length=arguments.max_prompt_length,
                max_completion_length=arguments.max_completion_length,
                pad_token_id=self.padding_value,
                label_pad_token_id=arguments.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )
        else:
            input_data_collator_tfds = data_collator
            input_data_collator_grain = data_collator

        self.input_data_collator_tfds = input_data_collator_tfds
        self.input_data_collator_grain = input_data_collator_grain

        train_dataset = self._prepare_dataset(train_dataset, processing_class, arguments, "train")
        if eval_dataset is not None:
            if isinstance(eval_dataset, dict):
                eval_dataset = {
                    key: self._prepare_dataset(dataset, processing_class, arguments, key)
                    for key, dataset in eval_dataset.items()
                }
            else:
                eval_dataset = self._prepare_dataset(eval_dataset, processing_class, arguments, "eval")

        self.processing_class = processing_class
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.reference_state = reference_state
        self.calculate_kl = arguments.loss_type == "kto"
        self.aux_loss_enabled = getattr(model_state.model, "output_router_logits", False)
        self.aux_loss_coef = getattr(model_state.model, "router_aux_loss_coef", 0.0)

        super().__init__(
            arguments=arguments,
            model_state=model_state,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            data_collator=None,
        )

        if arguments.disable_dropout:
            self.model_state.model.eval()
            self.reference_state.model.eval()

    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: ProcessingClassType,
        arguments: KTOConfig,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        """Prepare dataset by extracting prompts, applying chat templates, and tokenizing.

        Args:
            dataset: Raw dataset to process.
            processing_class: Tokenizer or processor.
            arguments: Training configuration.
            dataset_name: Name for logging purposes.

        Returns:
            Processed dataset with tokenized fields.
        """
        map_kwargs = {}
        if isinstance(dataset, Dataset):
            map_kwargs["num_proc"] = arguments.dataset_num_proc
            map_kwargs["desc"] = f"Extracting prompt for {dataset_name}"

        dataset = dataset.map(maybe_extract_prompt, **map_kwargs)
        dataset = maybe_unpair_preference_dataset(dataset, arguments.dataset_num_proc, desc=f"Unpairing {dataset_name}")

        if isinstance(dataset, Dataset):
            map_kwargs["desc"] = f"Applying chat template to {dataset_name}"
        dataset = dataset.map(
            maybe_apply_chat_template,
            fn_kwargs={"tokenizer": processing_class},
            **map_kwargs,
        )

        if isinstance(dataset, Dataset):
            map_kwargs["desc"] = f"Tokenizing {dataset_name}"
        dataset = dataset.map(
            _tokenize,
            batched=True,
            fn_kwargs={"tokenizer": processing_class},
            **map_kwargs,
        )

        process_kwargs = {
            "prefix": "",
            "tokenizer": processing_class,
            "is_encoder_decoder": self.is_encoder_decoder,
            "max_length": arguments.max_length,
            "truncation_mode": arguments.truncation_mode,
            "label_pad_token_id": arguments.label_pad_token_id,
            "max_prompt_length": arguments.max_prompt_length,
            "max_completion_length": arguments.max_completion_length,
        }

        if isinstance(dataset, Dataset):
            map_kwargs["desc"] = f"Processing {dataset_name}"
        dataset = dataset.map(
            _process_tokens,
            fn_kwargs=process_kwargs,
            **map_kwargs,
        )
        return dataset

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """Configure JIT-compiled training and evaluation functions.

        Returns:
            Configuration containing compiled step functions and mesh.
        """
        mesh = self.model.mesh
        empty_sharding = jax.sharding.NamedSharding(spec=PartitionSpec(), mesh=mesh)

        def forward_fn(model, batch):
            return concatenated_forward(
                model,
                batch,
                is_encoder_decoder=self.arguments.is_encoder_decoder,
                label_pad_token_id=self.arguments.label_pad_token_id,
                padding_value=self.padding_value,
                max_length=self.arguments.max_length,
                truncation_mode=self.arguments.truncation_mode,
                aux_loss_enabled=self.aux_loss_enabled,
            )

        self.concatenated_forward = ejit(forward_fn, static_argnames=())

        self._train_shared_fn_static_args = (
            self.scheduler,
            forward_fn,
            self.arguments.beta,
            self.arguments.desirable_weight,
            self.arguments.undesirable_weight,
            self.arguments.loss_type,
            self.calculate_kl,
            self.aux_loss_coef,
            self.arguments.loss_config,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
        )

        ref_sharding = self.reference_state.shardings if self.reference_state is not None else empty_sharding

        train_static_argnums = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
        sharded_training_step_function = ejit(
            training_step,
            in_shardings=(self.state_shardings, empty_sharding, ref_sharding),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=train_static_argnums,
        )

        self._eval_shared_fn_static_args = (
            forward_fn,
            self.arguments.beta,
            self.arguments.desirable_weight,
            self.arguments.undesirable_weight,
            self.arguments.loss_type,
            self.calculate_kl,
            self.aux_loss_coef,
            self.arguments.step_partition_spec,
        )

        eval_static_argnums = (3, 4, 5, 6, 7, 8, 9, 10)
        sharded_evaluation_step_function = ejit(
            evaluation_step,
            in_shardings=(self.state_shardings, empty_sharding, ref_sharding),
            out_shardings=empty_sharding,
            static_argnums=eval_static_argnums,
        )

        self.sharded_training_step_function = sharded_training_step_function
        self.sharded_evaluation_step_function = sharded_evaluation_step_function
        self._train_shared_fn_extra_args = (self.reference_state,)
        self._eval_shared_fn_extra_args = (self.reference_state,)

        checkpoint_manager = self.arguments.get_streaming_checkpointer()

        return TrainerConfigureFunctionOutput(
            sharded_training_step_function=sharded_training_step_function,
            sharded_evaluation_step_function=sharded_evaluation_step_function,
            mesh=mesh,
            checkpoint_manager=checkpoint_manager,
        )

    def create_grain_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """Create data collator for Grain data loading.

        Args:
            max_sequence_length: Maximum sequence length.
            truncation_mode: How to truncate sequences.

        Returns:
            Grain-compatible data collator.
        """
        return self.input_data_collator_grain

    def create_tfds_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """Create data collator for TFDS data loading.

        Args:
            max_sequence_length: Maximum sequence length.
            truncation_mode: How to truncate sequences.

        Returns:
            TFDS-compatible data collator.
        """
        return self.input_data_collator_tfds
