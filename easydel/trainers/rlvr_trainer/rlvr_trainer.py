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

"""Reinforcement Learning with Verifiable Rewards (RLVR) trainer.

RLVR is a single-turn RL pipeline where rewards come from deterministic
verifiable functions (math answer checking, code test execution, format
compliance) rather than learned reward models. It extends GRPO by
automatically constructing the reward function ensemble from verifier
configuration.

The training loop is identical to GRPO:

1. Generate ``N`` completions per prompt.
2. Score each completion with verifiable reward functions.
3. Compute group-relative advantages.
4. Update the policy with the GRPO/DAPO/CISPO loss.

The key difference is that reward functions are built-in verifiers
rather than external reward models, making training fully self-contained
and reproducible.

Inspired by the RLVR paradigm from DeepSeek-R1 and Alibaba's ROLL.
"""

from __future__ import annotations

import typing as tp

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry
from easydel.utils.helpers import get_logger

from ..group_relative_policy_optimization.grpo_trainer import GRPOTrainer
from .reward_verifiers import (
    FormatVerifier,
    LengthPenaltyVerifier,
    MathVerifier,
)
from .rlvr_config import RLVRConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from easydel.data.core.protocols import ShardedDataSource

logger = get_logger(__name__)

RewardFunc = EasyDeLBaseModule | EasyDeLState | tp.Callable[[list, list], list[float]]


@Registry.register("trainer", "rlvr")
class RLVRTrainer(GRPOTrainer):
    """Reinforcement Learning with Verifiable Rewards trainer.

    Extends GRPOTrainer to automatically build a reward function
    ensemble from verifiable reward specifications in ``RLVRConfig``.
    The trainer constructs ``MathVerifier``, ``CodeVerifier``,
    ``FormatVerifier``, and ``LengthPenaltyVerifier`` instances based
    on the configuration and combines them with any user-provided
    reward functions.

    This provides a fully self-contained RL training pipeline:
    generate completions, verify correctness with rule-based checks,
    compute group-relative advantages, and update the policy — all
    without a separate reward model.

    Args:
        arguments: ``RLVRConfig`` with training hyperparameters.
        model: Language model or state for the policy.
        reward_funcs: Backward-compatible alias for additional reward
            functions beyond the built-in verifiers. These are appended
            to the verifier ensemble.
        external_reward_funcs: Explicit external reward functions to
            append after the built-in verifiers.
        train_dataset: Training dataset with prompts and optional
            verifier sideband fields such as gold answers.
        eval_dataset: Optional evaluation dataset.
        processing_class: Tokenizer or processor.
        reward_processing_classes: Processing classes for the full
            verifier + external reward list. This remains supported for
            backward compatibility.
        external_reward_processing_classes: Processing classes only for
            ``external_reward_funcs``. When provided, built-in verifiers
            are automatically padded with ``None``.
        external_reward_weights: Optional weights for
            ``external_reward_funcs`` only.
        data_tokenize_fn: Optional tokenization function.

    Example:
        >>> from datasets import load_dataset
        >>> config = RLVRConfig(
        ...     max_prompt_length=1024,
        ...     max_completion_length=2048,
        ...     num_return_sequences=4,
        ...     answer_key="answer",
        ...     format_pattern=r"\\\\boxed\\{.+\\}",
        ...     format_reward_weight=0.1,
        ... )
        >>> gsm8k = load_dataset("openai/gsm8k", "main", split="train")
        >>> trainer = RLVRTrainer(
        ...     arguments=config,
        ...     model=model,
        ...     train_dataset=gsm8k,
        ...     processing_class=tokenizer,
        ... )
        >>> trainer.train()
    """

    arguments: RLVRConfig

    def __init__(
        self,
        arguments: RLVRConfig,
        model: EasyDeLBaseModule | EasyDeLState | None,
        reward_funcs: RewardFunc | list[RewardFunc] | None = None,
        external_reward_funcs: RewardFunc | list[RewardFunc] | None = None,
        train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        eval_dataset: Dataset | IterableDataset | ShardedDataSource | dict[str, Dataset] | None = None,
        processing_class: ProcessingClassType | None = None,
        reward_processing_classes: ProcessingClassType | None = None,
        external_reward_processing_classes: ProcessingClassType | list[ProcessingClassType] | None = None,
        external_reward_weights: list[float] | None = None,
        data_tokenize_fn: tp.Callable | None = None,
    ):
        if not isinstance(arguments, RLVRConfig):
            raise TypeError(f"arguments must be RLVRConfig, got {type(arguments)}")

        verifiers, weights = self._build_verifiers(arguments)
        external_rewards = self._coerce_reward_func_list(reward_funcs)
        external_rewards.extend(self._coerce_reward_func_list(external_reward_funcs))

        if external_rewards:
            verifiers.extend(external_rewards)
            if external_reward_weights is not None:
                user_weights = list(external_reward_weights)
            else:
                user_weights = list(arguments.reward_weights or [1.0] * len(external_rewards))
            if len(user_weights) != len(external_rewards):
                raise ValueError(
                    "The number of external reward weights must match the number of external reward functions."
                )
            weights.extend(user_weights)

        if not verifiers:
            if arguments.answer_key:
                verifiers = [MathVerifier(answer_key=arguments.answer_key)]
                weights = [1.0]
                logger.warning(
                    "No verifiers configured and no reward_funcs provided. "
                    "Falling back to MathVerifier with answer_key='%s'.",
                    arguments.answer_key,
                )
            else:
                raise ValueError(
                    "RLVR requires at least one verifier or external reward function. "
                    "Set `answer_key`, `format_pattern`, or pass external rewarders."
                )

        reward_processing_classes = self._merge_reward_processing_classes(
            verifier_count=len(verifiers) - len(external_rewards),
            external_reward_count=len(external_rewards),
            reward_processing_classes=reward_processing_classes,
            external_reward_processing_classes=external_reward_processing_classes,
        )

        arguments.reward_weights = weights

        super().__init__(
            arguments=arguments,
            model=model,
            reward_funcs=verifiers,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            data_tokenize_fn=data_tokenize_fn,
        )

    @staticmethod
    def _build_verifiers(config: RLVRConfig) -> tuple[list[tp.Callable], list[float]]:
        """Construct reward verifiers from config.

        Returns:
            Tuple of (verifier_list, weight_list).
        """
        verifiers: list[tp.Callable] = []
        weights: list[float] = []

        if config.answer_key:
            verifiers.append(MathVerifier(answer_key=config.answer_key))
            weights.append(1.0)

        if config.format_pattern and config.format_reward_weight > 0:
            verifiers.append(FormatVerifier(pattern=config.format_pattern))
            weights.append(config.format_reward_weight)

        if config.length_penalty_target is not None and config.length_penalty_weight > 0:
            verifiers.append(LengthPenaltyVerifier(target_length=config.length_penalty_target))
            weights.append(config.length_penalty_weight)

        return verifiers, weights

    @staticmethod
    def _coerce_reward_func_list(
        reward_funcs: RewardFunc | list[RewardFunc] | None,
    ) -> list[RewardFunc]:
        if reward_funcs is None:
            return []
        return list(reward_funcs) if isinstance(reward_funcs, list) else [reward_funcs]

    @staticmethod
    def _coerce_processing_class_list(
        processing_classes: ProcessingClassType | list[ProcessingClassType] | None,
        *,
        expected_len: int,
        field_name: str,
    ) -> list[ProcessingClassType | None]:
        if expected_len == 0:
            return []
        if processing_classes is None:
            return [None] * expected_len
        if isinstance(processing_classes, list):
            if len(processing_classes) != expected_len:
                raise ValueError(f"`{field_name}` must have length {expected_len}, got {len(processing_classes)}.")
            return list(processing_classes)
        if expected_len == 1:
            return [processing_classes]
        raise ValueError(
            f"`{field_name}` must be a list of length {expected_len} when multiple reward functions are used."
        )

    @classmethod
    def _merge_reward_processing_classes(
        cls,
        *,
        verifier_count: int,
        external_reward_count: int,
        reward_processing_classes: ProcessingClassType | list[ProcessingClassType] | None,
        external_reward_processing_classes: ProcessingClassType | list[ProcessingClassType] | None,
    ) -> list[ProcessingClassType | None] | None:
        if external_reward_processing_classes is not None and reward_processing_classes is not None:
            raise ValueError(
                "Pass either `reward_processing_classes` for the full reward list or "
                "`external_reward_processing_classes` for external rewards only, not both."
            )
        total_count = verifier_count + external_reward_count
        if total_count == 0:
            return None
        if reward_processing_classes is not None:
            return cls._coerce_processing_class_list(
                reward_processing_classes,
                expected_len=total_count,
                field_name="reward_processing_classes",
            )
        if external_reward_processing_classes is not None:
            external_classes = cls._coerce_processing_class_list(
                external_reward_processing_classes,
                expected_len=external_reward_count,
                field_name="external_reward_processing_classes",
            )
            return [None] * verifier_count + external_classes
        return None
