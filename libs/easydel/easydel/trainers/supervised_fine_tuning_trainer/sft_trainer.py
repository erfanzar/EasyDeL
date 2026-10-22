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
"""Supervised Fine-Tuning (SFT) trainer.

Wires :class:`SFTTrainer`, EasyDeL's standard causal-LM cross-entropy
trainer.  The module relies on the lazy
:class:`SFTPreprocessTransform` to render chat templates and tokenise
text, and optionally on
:class:`easydel.data.transforms.pack.PackedShardedSource` to pack short
sequences into fixed-length blocks.  Public entry point is
:class:`SFTTrainer`.
"""

from __future__ import annotations

import typing as tp

from eformer.loggings import get_logger

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.sharding import replicated_named_sharding
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry

from ..model_loading import apply_chat_template_path, apply_tokenizer_overrides, reject_string_model_id
from ..prompt_transforms import SFTPreprocessTransform
from ..trainer import Trainer
from ..trainer._fn import evaluation_step, training_step
from ..trainer_protocol import TrainerConfigureFunctionOutput
from ..training_utils import (
    compile_trainer_step,
    normalize_completion_masks_and_labels,
    resolve_mask_prompt,
    resolve_straight_through_emulator,
)
from ..utils import DataCollatorForCompletionOnlyLM, get_formatting_func_from_dataset
from .sft_config import SFTConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset  # pyright: ignore[reportMissingTypeStubs]

    from easydel.data.core.protocols import ShardedDataSource

logger = get_logger(__name__)


@Registry.register("trainer", "sft")
class SFTTrainer(Trainer):
    """Supervised Fine-Tuning trainer for language models.

    Implements standard supervised fine-tuning for both base and instruction-tuned
    models. Supports various data formats including conversational datasets,
    completion-only training, and packed sequences for efficient training.

    Key features:
    - Automatic dataset formatting and tokenization via lazy transforms
    - Support for conversational/chat templates
    - Sequence packing for improved efficiency
    - Completion-only loss (ignore prompt tokens)
    - Multi-turn conversation handling

    The trainer uses lazy preprocessing transforms that are applied during
    iteration, providing better performance than eager HF .map() calls.

    Attributes:
        arguments: SFTConfig with training hyperparameters
        tokenizer: Tokenizer for text processing
        formatting_func: Optional function to format examples

    Example:
        >>> config = SFTConfig(
        ...     per_device_train_batch_size=4,
        ...     learning_rate=2e-5,
        ...     packing=True,
        ...     max_length=2048
        ... )
        >>> trainer = SFTTrainer(
        ...     arguments=config,
        ...     model=model,
        ...     train_dataset=dataset,
        ...     processing_class=tokenizer,
        ...     formatting_func=lambda x: x["text"]  # Optional
        ... )
        >>> trainer.train()

    Note:
        For conversational datasets, the trainer expects either:
        - A 'messages' column with chat format
        - A custom formatting_func to extract text
        - A dataset_text_field pointing to the text column
    """

    def __init__(
        self,
        arguments: SFTConfig,
        processing_class: ProcessingClassType,
        model: EasyDeLBaseModule | EasyDeLState | None = None,
        train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        eval_dataset: Dataset | IterableDataset | ShardedDataSource | dict[str, Dataset] | None = None,
        formatting_func: tp.Callable | None = None,
        data_collator: DataCollatorForCompletionOnlyLM | None = None,
    ):
        """Initialize the supervised fine-tuning trainer.

        Args:
            arguments (SFTConfig): SFT configuration; must be an
                :class:`SFTConfig`.
            processing_class (ProcessingClassType): Tokenizer / processor.
                If ``pad_token`` is unset and ``eos_token`` exists,
                pad is set to eos in place.
            model (EasyDeLBaseModule | EasyDeLState | None): Model to be
                fine-tuned. Plain modules are converted to a state via
                ``model.to_state(...)``.
            train_dataset (Dataset | IterableDataset | ShardedDataSource | None):
                Training dataset.
            eval_dataset (Dataset | IterableDataset | ShardedDataSource |
                dict[str, Dataset] | None): Optional evaluation
                dataset(s).
            formatting_func (tp.Callable | None): Optional callable
                converting a dataset row into a single string. When
                omitted and ``arguments.dataset_text_field`` is also
                ``None``, an automatic formatter is inferred from the
                dataset (when possible).
            data_collator (DataCollatorForCompletionOnlyLM | None):
                Optional completion-only data collator.

        Raises:
            TypeError: If ``arguments`` is not an :class:`SFTConfig`.
        """
        if not isinstance(arguments, SFTConfig):
            raise TypeError("passed argument must be a `SFTConfig`.")

        tokenizer = processing_class
        if hasattr(processing_class, "tokenizer"):
            tokenizer = processing_class.tokenizer
        apply_tokenizer_overrides(tokenizer, eos_token=arguments.eos_token, pad_token=arguments.pad_token)
        if arguments.chat_template_path is not None:
            apply_chat_template_path(tokenizer, arguments.chat_template_path)
        if getattr(tokenizer, "pad_token", None) is None and hasattr(tokenizer, "eos_token"):
            tokenizer.pad_token = tokenizer.eos_token

        # Auto-detect formatting function if not provided
        if formatting_func is None and arguments.dataset_text_field is None and train_dataset is not None:
            formatting_func = get_formatting_func_from_dataset(train_dataset, processing_class)

        self.arguments = arguments
        self.tokenizer = tokenizer
        self._formatting_func = formatting_func
        self._dataset_text_field = arguments.dataset_text_field

        if isinstance(model, str):
            reject_string_model_id(model, role="policy model")
        if not isinstance(model, EasyDeLState):
            model = model.to_state(trainable_selector=arguments.trainable_selector)

        super().__init__(
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            model_state=model,
            data_collator=data_collator,
            processing_class=processing_class,
        )

    @staticmethod
    def _apply_tokenizer_overrides(tokenizer: object, *, eos_token: str | None, pad_token: str | None) -> None:
        """Apply token overrides through the shared helper (compatibility API)."""
        apply_tokenizer_overrides(tokenizer, eos_token=eos_token, pad_token=pad_token)

    def _get_preprocess_transform(self) -> SFTPreprocessTransform | None:
        """Build the lazy SFT preprocessing transform for the data source.

        Returns a transform that handles:
        - Formatting function application
        - Format detection (conversational vs text)
        - Chat template application
        - Tokenization with optional completion masking

        The transform is skipped (``None`` returned) when the bound
        training source already exposes an ``"input_ids"`` field, as
        reported by :meth:`_is_pretokenized`. The ``mask_prompt`` flag
        on the returned transform combines ``assistant_only_loss`` and
        the legacy ``completion_only_loss`` setting, with the latter
        taking precedence when explicitly set.

        Returns:
            SFTPreprocessTransform | None: The configured preprocessing
            transform, or ``None`` when the dataset is already
            tokenized.
        """

        dataset_kwargs = self.arguments.dataset_kwargs or {}
        if bool(dataset_kwargs.get("skip_prepare_dataset", False)):
            return None

        if self._is_pretokenized():
            return None

        mask_prompt = resolve_mask_prompt(self.arguments)

        return SFTPreprocessTransform(
            tokenizer=self.processing_class,
            max_length=self.arguments.max_length,
            text_field=self._dataset_text_field or "text",
            mask_prompt=mask_prompt,
            padding=(
                False
                if getattr(self.arguments, "packing", False) or getattr(self.arguments, "padding_free", False)
                else "max_length"
            ),
            truncation_mode=self.arguments.truncation_mode,
            pad_to_multiple_of=self.arguments.pad_to_multiple_of,
            formatting_func=self._formatting_func,
        )

    @staticmethod
    def _apply_chat_template_path(tokenizer: object, chat_template_path: str) -> None:
        """Assign a chat template through the shared helper (compatibility API)."""
        apply_chat_template_path(tokenizer, chat_template_path)

    def _apply_preprocess_transforms(self) -> None:
        """Run the base preprocessor and optionally wrap the source in a packer.

        After the standard tokenisation transform attached by the base
        :class:`Trainer` runs, this override preserves the legacy SFT
        ``packing=True`` knob. The shared ``sequence_packing=True`` flag
        is handled by the base implementation so other offline
        single-stream trainers use the same wrapper.

        Side effects:
            Replaces ``self._train_source`` (and, when configured,
            ``self._eval_source``) with packed views in place.
        """
        super()._apply_preprocess_transforms()

        if getattr(self.arguments, "sequence_packing", False) or not getattr(self.arguments, "packing", False):
            return

        eval_packing = getattr(self.arguments, "eval_packing", None)
        if eval_packing is None:
            eval_packing = self.arguments.packing

        self._apply_sequence_packing_to_sources(
            train_enabled=True,
            eval_enabled=bool(eval_packing),
        )

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """Compile SFT train/eval steps with the active sharding and QAT config."""
        empty_sharding = replicated_named_sharding(self.model.mesh)
        straight_through_emulator = resolve_straight_through_emulator(
            quantization_mode=self.arguments.quantization_mode,
            quantization_group_size=self.arguments.quantization_group_size,
            quantization_bits=self.arguments.quantization_bits,
            tensor_straight_through=self.arguments.tensor_straight_through,
            straight_through_emulator=self.arguments.straight_through_emulator,
        )
        self._train_shared_fn_static_args = (
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            straight_through_emulator,
            self.arguments.loss_type,
        )
        sharded_training_step_function = compile_trainer_step(
            training_step,
            static_argnums=(2, 3, 4, 5, 6, 7),
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            schedule=self.arguments.mpmd_scheduler,
            arguments=self.arguments,
            mesh=self.mesh,
        )

        self._eval_shared_fn_static_args = (
            self.arguments.loss_config,
            self.arguments.step_partition_spec,
            self.arguments.loss_type,
        )
        sharded_evaluation_step_function = compile_trainer_step(
            evaluation_step,
            static_argnums=(2, 3, 4),
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=empty_sharding,
            schedule=self.arguments.mpmd_scheduler,
            arguments=self.arguments,
            mesh=self.mesh,
        )

        self.arguments.ensure_checkpoint_path()
        checkpoint_manager = self.arguments.get_streaming_checkpointer()
        return TrainerConfigureFunctionOutput(
            sharded_training_step_function=sharded_training_step_function,
            sharded_evaluation_step_function=sharded_evaluation_step_function,
            mesh=self.model.mesh,
            checkpoint_manager=checkpoint_manager,
        )

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, tp.Any],
        is_train: bool,
    ) -> tuple[dict[str, tp.Any], dict[str, float | int | str]]:
        """Normalize completion masks and ``labels`` after the base preprocessor.

        Renames any ``assistant_masks`` field to ``completion_mask``,
        intersects ``completion_mask`` with ``attention_mask`` and, when
        ``labels`` is missing, derives it from ``input_ids`` masked to
        ``-100`` outside the completion / attention regions. When only
        ``labels`` is provided, ``completion_mask`` is recovered as
        ``labels != -100``.

        Args:
            state (EasyDeLState): Current model state (forwarded to the
                base preprocessor).
            batch (dict[str, tp.Any]): Raw input batch, possibly already
                post-processed by the base preprocessor.
            is_train (bool): Whether this preprocessing is for training.

        Returns:
            tuple[dict[str, tp.Any], dict[str, float | int | str]]:
            ``(processed_batch, auxiliary_metrics)``.
        """
        batch, infos = super()._preprocess_batch_input(state=state, batch=batch, is_train=is_train)

        return normalize_completion_masks_and_labels(batch, assistant_mask_fallback_on_empty=True), infos
