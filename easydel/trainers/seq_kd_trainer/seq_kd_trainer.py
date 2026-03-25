# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Sequence-level Knowledge Distillation (SeqKD) trainer.

The teacher generates text completions from prompts and the student trains
on them with standard cross-entropy loss.  No teacher logits are required,
so this works with API-based teachers via ``teacher_fn``.
"""

from __future__ import annotations

import typing as tp

import numpy as np
from eformer.loggings import get_logger
from jax import numpy as jnp

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry
from easydel.utils.helpers import capture_time

from ..prompt_transforms import GRPOPreprocessTransform
from ..trainer import Trainer
from .seq_kd_config import SeqKDConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset  # pyright: ignore[reportMissingTypeStubs]

    from easydel.data.core.protocols import ShardedDataSource

logger = get_logger(__name__)


@Registry.register("trainer", "seq_kd")
class SeqKDTrainer(Trainer):
    """Sequence-level Knowledge Distillation trainer.

    The teacher generates text completions from prompts and the student is
    trained on those completions using standard cross-entropy loss.  This is
    a black-box distillation method: no teacher logits or hidden states are
    required.

    Two teacher modes are supported:
        1. **Local teacher model** (``teacher_model``): An EasyDeL model that
           generates completions via ``generate_unified``.
        2. **External teacher function** (``teacher_fn``): A callable
           ``(prompts: list[str]) -> list[str]`` for API-based teachers.
           When ``num_generations_per_prompt > 1``, prompts are repeated in
           prompt-major order and the callable must return one completion per
           repeated prompt.

    Training loop:
        1. Sample prompts from the dataset
        2. Teacher generates completions (local model or API)
        3. Build training batch with teacher completions as labels
        4. Student trains with standard CE loss (same as SFT)

    Example:
        >>> config = SeqKDConfig(
        ...     max_prompt_length=512,
        ...     max_completion_length=256,
        ...     learning_rate=2e-5,
        ... )
        >>> trainer = SeqKDTrainer(
        ...     arguments=config,
        ...     student_model=student,
        ...     teacher_model=teacher,
        ...     train_dataset=dataset,
        ...     processing_class=tokenizer,
        ... )
        >>> trainer.train()
    """

    teacher_state: EasyDeLState | None
    teacher_fn: tp.Callable[[list[str]], list[str]] | None
    arguments: SeqKDConfig

    def __init__(
        self,
        arguments: SeqKDConfig,
        processing_class: ProcessingClassType,
        student_model: EasyDeLBaseModule | EasyDeLState | None = None,
        teacher_model: EasyDeLBaseModule | EasyDeLState | None = None,
        teacher_fn: tp.Callable[[list[str]], list[str]] | None = None,
        train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        eval_dataset: Dataset | IterableDataset | ShardedDataSource | dict[str, Dataset] | None = None,
    ):
        tokenizer = processing_class
        if hasattr(processing_class, "tokenizer"):
            tokenizer = processing_class.tokenizer
        if getattr(tokenizer, "pad_token", None) is None and hasattr(tokenizer, "eos_token"):
            tokenizer.pad_token = tokenizer.eos_token

        if not isinstance(arguments, SeqKDConfig):
            raise TypeError("passed argument must be a `SeqKDConfig`.")
        if teacher_model is None and teacher_fn is None:
            raise ValueError("Either `teacher_model` or `teacher_fn` must be provided.")

        self.arguments = arguments

        if not isinstance(student_model, EasyDeLState):
            student_model = student_model.to_state()

        if teacher_model is not None and not isinstance(teacher_model, EasyDeLState):
            teacher_model = teacher_model.to_state()

        self.teacher_state = teacher_model
        self.teacher_fn = teacher_fn
        self.processing_class = processing_class

        pad_token_id = getattr(processing_class, "pad_token_id", None)
        if pad_token_id is None and hasattr(processing_class, "tokenizer"):
            pad_token_id = getattr(processing_class.tokenizer, "pad_token_id", None)
        self.padding_value = 0 if pad_token_id is None else int(pad_token_id)

        super().__init__(
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            model_state=student_model,
            data_collator=None,
            processing_class=processing_class,
        )

    def _get_preprocess_transform(self) -> GRPOPreprocessTransform | None:
        """Get preprocessing transform for prompt-only datasets."""
        if self._is_pretokenized():
            return None
        return GRPOPreprocessTransform(
            tokenizer=self.processing_class,
            max_prompt_length=self.arguments.max_prompt_length,
            skip_apply_chat_template=self.arguments.skip_apply_chat_template,
            tools=getattr(self.arguments, "tools", None),
        )

    def _is_pretokenized(self) -> bool:
        """Check whether the source already yields token IDs."""
        if self._train_source is None:
            return False
        try:
            sample = next(iter(self._train_source.open_shard(self._train_source.shard_names[0])))
            return "input_ids" in sample
        except (StopIteration, IndexError):
            return False

    def create_grain_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """Create Grain data collator for prompt-only batches."""
        from ..utils import GRPODataCollatorGrain

        return GRPODataCollatorGrain(
            max_prompt_length=self.arguments.max_prompt_length,
            pad_token_id=self.padding_value,
        )

    def create_tfds_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """Create TFDS data collator for prompt-only batches."""
        from ..utils import GRPODataCollatorTFDS

        return GRPODataCollatorTFDS(
            max_prompt_length=self.arguments.max_prompt_length,
            pad_token_id=self.padding_value,
        )

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, tp.Any],
        is_train: bool,
    ) -> tuple[dict[str, tp.Any], dict[str, float | int | str]]:
        """Generate completions from teacher and prepare a CE training batch.

        If ``teacher_fn`` is set, prompts are decoded to text, optionally
        repeated per prompt when ``num_generations_per_prompt > 1``, passed to
        the callable, and the returned completions are re-tokenized.  Otherwise
        ``generate_unified`` is called on the local teacher model.
        """
        batch = self._purify_batch(batch)

        generation_time: float = 0.0
        with capture_time() as preprocessing_time_fn:
            prompt_ids, prompt_mask = batch["input_ids"], batch["attention_mask"]
            generated_completions_text: list[str] | None = None

            if self.teacher_fn is not None:
                with capture_time() as generation_time_fn:
                    prompt_texts = self.processing_class.batch_decode(
                        prompt_ids,
                        skip_special_tokens=True,
                    )
                    generation_factor = int(
                        getattr(self.arguments, "generation_num_return_sequences", None)
                        or getattr(self.arguments, "num_generations_per_prompt", 1)
                        or 1
                    )
                    generation_factor = max(generation_factor, 1)
                    expanded_prompt_texts = [
                        prompt_text for prompt_text in prompt_texts for _ in range(generation_factor)
                    ]
                    completion_texts = list(self.teacher_fn(expanded_prompt_texts))
                    if len(completion_texts) != len(expanded_prompt_texts):
                        raise ValueError(
                            "`teacher_fn` must return exactly one completion per prompt. "
                            f"Expected {len(expanded_prompt_texts)} completions for "
                            f"{len(prompt_texts)} prompts with generation_factor={generation_factor}, "
                            f"but got {len(completion_texts)}."
                        )
                    encoded = self.processing_class(
                        completion_texts,
                        padding="max_length",
                        max_length=self.arguments.max_completion_length,
                        truncation=True,
                        return_tensors="np",
                        add_special_tokens=False,
                    )
                    generated_completions_text = completion_texts
                    completion_ids = jnp.array(encoded["input_ids"])
                    completion_mask = jnp.array(encoded["attention_mask"])
                generation_time = generation_time_fn()
            else:
                with capture_time() as generation_time_fn:
                    results = self.generate_unified(
                        input_ids=prompt_ids,
                        attention_mask=prompt_mask,
                        state=self.teacher_state,
                        apply_chat_template=False,
                        shard_inputs=False,
                        all_gather=False,
                    )
                    prompt_ids = results.prompt_ids
                    prompt_mask = results.prompt_mask
                    completion_ids = results.completion_ids
                    generated_completions_text = self._coerce_generation_texts(results.text)
                generation_time = generation_time_fn()

                completion_mask = self._make_attn_mask(completion_ids)

            # Handle generation factor (num_generations_per_prompt > 1)
            generation_factor = completion_ids.shape[0] // max(prompt_mask.shape[0], 1)
            generation_factor = max(generation_factor, 1)
            expanded_prompt_ids = prompt_ids.repeat(generation_factor, 0) if generation_factor > 1 else prompt_ids
            expanded_prompt_mask = prompt_mask.repeat(generation_factor, 0) if generation_factor > 1 else prompt_mask

            # Concatenate prompt + completion into full sequences
            input_ids_full = jnp.concatenate([expanded_prompt_ids, completion_ids], axis=1)
            attention_mask_full = jnp.concatenate([expanded_prompt_mask, completion_mask], axis=1)

            # Build labels: same as input_ids, but mask prompt tokens with -100
            # ForCausalLMLoss shifts internally, so labels[i] = input_ids[i] is correct.
            prompt_len = expanded_prompt_ids.shape[1]
            labels = np.array(input_ids_full)
            labels[:, :prompt_len] = -100
            labels[np.array(attention_mask_full) == 0] = -100

            # Gather to replicated sharding
            input_ids_full = self._all_gather(input_ids_full)
            attention_mask_full = self._all_gather(attention_mask_full)
            labels = self._all_gather(jnp.asarray(labels))

        preprocessing_time = preprocessing_time_fn()

        metrics_dict: dict[str, float | int | str] = {
            "completion_length": float(jnp.mean(jnp.sum(completion_mask, -1))),
            "generation_time": generation_time,
            "preprocessing_time": preprocessing_time,
        }
        self._log_training_generations_to_wandb(
            state=state,
            prompts=expanded_prompt_ids,
            prompt_mask=expanded_prompt_mask,
            completions=generated_completions_text,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            generation_time=generation_time,
            source="teacher",
        )

        return (
            {
                "input_ids": input_ids_full,
                "attention_mask": attention_mask_full,
                "labels": labels,
            },
            metrics_dict,
        )
