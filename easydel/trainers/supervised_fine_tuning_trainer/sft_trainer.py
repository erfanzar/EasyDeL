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
from __future__ import annotations

import typing as tp

import numpy as np
from eformer.loggings import get_logger

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry

from ..prompt_transforms import SFTPreprocessTransform
from ..trainer import Trainer
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
                Optional completion-only data collator. Allowed only
                when ``arguments.packing`` is True.

        Raises:
            TypeError: If ``arguments`` is not an :class:`SFTConfig`.
            ValueError: If ``data_collator`` is supplied while
                ``arguments.packing`` is False.
        """
        if not isinstance(arguments, SFTConfig):
            raise TypeError("passed argument must be a `SFTConfig`.")

        tokenizer = processing_class
        if hasattr(processing_class, "tokenizer"):
            tokenizer = processing_class.tokenizer
        if getattr(tokenizer, "pad_token", None) is None and hasattr(tokenizer, "eos_token"):
            tokenizer.pad_token = tokenizer.eos_token

        # Auto-detect formatting function if not provided
        if formatting_func is None and arguments.dataset_text_field is None and train_dataset is not None:
            formatting_func = get_formatting_func_from_dataset(train_dataset, processing_class)

        if not arguments.packing and data_collator:
            raise ValueError(
                "You passed `packing=False` to the SFTTrainer, but you didn't pass a "
                "`dataset_text_field` or `formatting_func` argument."
            )

        # Store for use in _get_preprocess_transform
        self.arguments = arguments
        self.tokenizer = tokenizer
        self._formatting_func = formatting_func
        self._dataset_text_field = arguments.dataset_text_field

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

    def _get_preprocess_transform(self) -> SFTPreprocessTransform | None:
        """Get SFT preprocessing transform for ShardedDataSource.

        Returns a transform that handles:
        - Formatting function application
        - Format detection (conversational vs text)
        - Chat template application
        - Tokenization with optional completion masking

        Returns:
            SFTPreprocessTransform or None if data is already tokenized.
        """

        # Skip if already tokenized
        if self._is_pretokenized():
            return None

        mask_prompt = bool(getattr(self.arguments, "assistant_only_loss", False))
        completion_only_loss = getattr(self.arguments, "completion_only_loss", None)
        if completion_only_loss is not None:
            mask_prompt = bool(completion_only_loss)

        return SFTPreprocessTransform(
            tokenizer=self.processing_class,
            max_length=self.arguments.max_length,
            text_field=self._dataset_text_field or "text",
            mask_prompt=mask_prompt,
            formatting_func=self._formatting_func,
        )

    def _is_pretokenized(self) -> bool:
        """Detect whether the bound training source already exposes tokenised text.

        Peeks at the first row of the first shard and reports whether
        the column ``"input_ids"`` is present. The presence of that
        field is the signal the trainer uses to skip
        :class:`SFTPreprocessTransform` (chat-template rendering and
        tokenisation) and feed rows directly to the data collator. The
        method is defensive against unset sources, empty shard lists,
        and shards yielding no rows.

        Returns:
            ``True`` when the first sample of the first shard contains
            ``"input_ids"``; ``False`` otherwise.
        """
        if self._train_source is None:
            return False
        try:
            sample = next(iter(self._train_source.open_shard(self._train_source.shard_names[0])))
            return "input_ids" in sample
        except (StopIteration, IndexError):
            return False

    def _apply_preprocess_transforms(self) -> None:
        """Run the base preprocessor and optionally wrap the source in a packer.

        After the standard tokenisation transform attached by the base
        :class:`Trainer` runs, this override consults
        ``arguments.packing`` and ``arguments.eval_packing`` and, when
        either is enabled, wraps the corresponding shard source in a
        :class:`PackedShardedSource`. The packer fills fixed-length
        blocks of ``arguments.max_length`` tokens with multiple
        sequences separated by EOS, exposing per-sequence
        ``segment_ids`` so attention can be restricted to within-document
        boundaries.

        Strategy mapping mirrors the public ``packing_strategy`` field:
        ``"bfd"`` (the SFT default) maps to the underlying
        ``"first_fit"`` packer; ``"wrapped"`` maps to ``"greedy"``. When
        the tokenizer has no ``eos_token_id`` the pad token is used as a
        fallback delimiter and a warning is logged.

        Side effects:
            Replaces ``self._train_source`` (and, when configured,
            ``self._eval_source``) with packed views in place.
        """
        # First apply standard tokenization transform
        super()._apply_preprocess_transforms()

        # Then apply packing if enabled
        if not getattr(self.arguments, "packing", False):
            return

        from easydel.data.transforms.pack import PackedShardedSource

        # Get packing parameters
        seq_length = self.arguments.max_length
        eos_token_id = getattr(self.processing_class, "eos_token_id", None)
        pad_token_id = getattr(self.processing_class, "pad_token_id", 0)

        if eos_token_id is None:
            logger.warning("No eos_token_id found, using pad_token_id for packing")
            eos_token_id = pad_token_id

        # Map strategy names
        strategy_map = {"bfd": "first_fit", "wrapped": "greedy"}
        strategy = strategy_map.get(self.arguments.packing_strategy, "greedy")

        # Apply packing to train source
        if self._train_source is not None:
            self._train_source = PackedShardedSource(
                source=self._train_source,
                seq_length=seq_length,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                strategy=strategy,
                include_segment_ids=True,
            )

        # Apply packing to eval source if eval_packing is enabled
        eval_packing = getattr(self.arguments, "eval_packing", None)
        if eval_packing is None:
            eval_packing = self.arguments.packing

        if eval_packing and self._eval_source is not None:
            self._eval_source = PackedShardedSource(
                source=self._eval_source,
                seq_length=seq_length,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                strategy=strategy,
                include_segment_ids=True,
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

        if "assistant_masks" in batch:
            if "completion_mask" not in batch:
                batch["completion_mask"] = batch["assistant_masks"]
            batch.pop("assistant_masks", None)

        attention_mask = batch.get("attention_mask")
        completion_mask = batch.get("completion_mask")

        if completion_mask is not None:
            completion_mask_np = np.asarray(completion_mask)
            if attention_mask is not None:
                completion_mask_np = completion_mask_np * np.asarray(attention_mask)
            completion_dtype = (
                np.asarray(attention_mask).dtype if attention_mask is not None else completion_mask_np.dtype
            )
            batch["completion_mask"] = completion_mask_np.astype(completion_dtype, copy=False)

            if "labels" not in batch and "input_ids" in batch:
                labels = np.asarray(batch["input_ids"]).astype(np.int32, copy=False)
                labels[completion_mask_np == 0] = -100
                if attention_mask is not None:
                    labels[np.asarray(attention_mask) == 0] = -100
                batch["labels"] = labels

        if "labels" in batch and "completion_mask" not in batch:
            labels_np = np.asarray(batch["labels"])
            if (labels_np == -100).any():
                completion_mask_np = (labels_np != -100).astype(np.int32)
                if attention_mask is not None:
                    completion_mask_np = completion_mask_np * np.asarray(attention_mask)
                batch["completion_mask"] = completion_mask_np

        return batch, infos
