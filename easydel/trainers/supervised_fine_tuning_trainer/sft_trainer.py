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
    from datasets import Dataset, IterableDataset

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
            model = model.to_state()

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

        return SFTPreprocessTransform(
            tokenizer=self.processing_class,
            max_length=self.arguments.max_length,
            text_field=self._dataset_text_field or "text",
            mask_prompt=getattr(self.arguments, "completion_only_loss", False),
            formatting_func=self._formatting_func,
        )

    def _is_pretokenized(self) -> bool:
        """Check if dataset already has tokenized fields."""
        if self._train_source is None:
            return False
        try:
            sample = next(iter(self._train_source.open_shard(self._train_source.shard_names[0])))
            return "input_ids" in sample
        except (StopIteration, IndexError):
            return False

    def _apply_preprocess_transforms(self) -> None:
        """Apply preprocessing transforms including optional packing.

        Extends base implementation to add packing support when `packing=True`
        in the SFT config. Packing combines multiple sequences into fixed-length
        blocks for more efficient training.
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
