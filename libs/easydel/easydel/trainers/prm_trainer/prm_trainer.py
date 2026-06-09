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
"""Process reward model preprocessing and trainer."""

from __future__ import annotations

import typing as tp

from easydel.infra.base_state import EasyDeLState
from easydel.utils import Registry

from ..model_loading import disable_state_dropout, reject_string_model_id
from ..reward_trainer import RewardTrainer
from ..trainer import Trainer
from .prm_config import PRMConfig
from .prm_preprocess import PRMPreprocessTransform


@Registry.register("trainer", "prm")
class PRMTrainer(Trainer):
    """Token-classification trainer for process reward model supervision.

    PRM expects an initialized EasyDeL token-classification module or state and
    a tokenizer/processor. Raw PRM examples are transformed into token labels by
    :class:`PRMPreprocessTransform`; already-tokenized sources pass through.
    """

    arguments: PRMConfig

    def __init__(
        self,
        arguments: PRMConfig,
        model: tp.Any,
        processing_class: tp.Any,
        train_dataset: tp.Any = None,
        eval_dataset: tp.Any = None,
        data_collator: tp.Callable | None = None,
    ) -> None:
        """Initialize process-reward token-classification training.

        Args:
            arguments: PRM configuration with sequence lengths, tokenizer
                overrides, and dropout behavior.
            model: Initialized EasyDeL token-classification module or state.
                String model ids are rejected because EasyDeL trainers receive
                already-created model objects.
            processing_class: Tokenizer or processor used by PRM preprocessing.
            train_dataset: Raw or pretokenized process-reward training data.
            eval_dataset: Optional raw or pretokenized evaluation data.
            data_collator: Optional collator override passed to the base
                trainer.

        Raises:
            TypeError: If ``arguments`` is not ``PRMConfig``.
            ValueError: If the processing class is missing or the model is a
                string identifier instead of an initialized object.
        """
        if not isinstance(arguments, PRMConfig):
            raise TypeError(f"arguments must be PRMConfig, got {type(arguments)}")
        if processing_class is None:
            raise ValueError("processing_class must be specified.")
        if isinstance(model, str):
            raise ValueError(
                "EasyDeL PRMTrainer does not accept token-classification model ids. "
                "Load an initialized EasyDeL token-classification module/state first."
            )
        if isinstance(model, str):
            reject_string_model_id(model, role="token-classification model")
        tokenizer = processing_class.tokenizer if hasattr(processing_class, "tokenizer") else processing_class
        RewardTrainer._apply_tokenizer_overrides(tokenizer, eos_token=arguments.eos_token, pad_token=arguments.pad_token)
        if getattr(tokenizer, "pad_token", None) is None and hasattr(tokenizer, "eos_token"):
            tokenizer.pad_token = tokenizer.eos_token
        if not isinstance(model, EasyDeLState):
            model = model.to_state(trainable_selector=arguments.trainable_selector)
        if arguments.disable_dropout:
            model = disable_state_dropout(model)
        self.arguments = arguments
        self.processing_class = processing_class
        self.data_collator = data_collator
        super().__init__(
            arguments=arguments,
            model_state=model,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            data_collator=data_collator,
            processing_class=processing_class,
        )

    def _get_preprocess_transform(self) -> PRMPreprocessTransform | None:
        """Return PRM preprocessing unless the source already has labels.

        The transform is configured from trainer arguments, including step
        separator, max lengths, final-step-only supervision, and padding
        multiple. Pretokenized sources skip this path entirely.
        """
        if self._is_pretokenized():
            return None
        return PRMPreprocessTransform(
            tokenizer=self.processing_class,
            step_separator=self.arguments.step_separator,
            max_length=self.arguments.max_length,
            max_completion_length=self.arguments.max_completion_length,
            train_on_last_step_only=self.arguments.train_on_last_step_only,
            is_eval=False,
            pad_to_multiple_of=self.arguments.pad_to_multiple_of,
        )

    @staticmethod
    def tokenize_row(
        features: dict[str, tp.Any],
        tokenizer: tp.Any,
        step_separator: str = "\n",
        max_length: int | None = 1024,
        max_completion_length: int | None = None,
        train_on_last_step_only: bool = False,
        is_eval: bool = False,
    ) -> dict[str, list[int]]:
        """Convenience wrapper around :meth:`PRMPreprocessTransform.tokenize_row`.

        Tests and offline preprocessing scripts can call this method without
        constructing a trainer, while still using the same tokenization and
        sparse-label rules as runtime PRM training.
        """
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        return PRMPreprocessTransform.tokenize_row(
            features,
            tokenizer=tokenizer,
            step_separator=step_separator,
            max_length=max_length,
            max_completion_length=max_completion_length,
            train_on_last_step_only=train_on_last_step_only,
            is_eval=is_eval,
            pad_token_id=pad_token_id,
        )
