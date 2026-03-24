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

"""Configuration for Reinforcement Learning with Verifiable Rewards (RLVR).

RLVR is a single-turn generation-then-train pipeline that uses rule-based
verifiable reward functions (math answer checking, code test execution,
format compliance) instead of learned reward models. It combines
naturally with GRPO for critic-free advantage estimation.

This is the EasyDeL implementation of the RLVR paradigm used by
DeepSeek-R1 and Alibaba's ROLL framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..group_relative_policy_optimization.grpo_config import GRPOConfig


@Registry.register("trainer-arguments", "rlvr")
@dataclass
class RLVRConfig(GRPOConfig):
    """Configuration for Reinforcement Learning with Verifiable Rewards.

    Extends GRPOConfig with parameters specific to verifiable-reward
    training. The key difference from standard GRPO is that rewards
    come from deterministic rule-based verifiers (math correctness,
    code test execution, format checks) rather than learned reward
    models.

    RLVR generates multiple completions per prompt, scores them with
    verifiable reward functions, computes group-relative advantages,
    and updates the policy — all without a critic network.

    Attributes:
        answer_key: Optional column name in the dataset containing
            gold answers for math verification. Set to ``None`` to
            disable the built-in math verifier and rely entirely on
            external rewarders.
        test_key: Column name containing code test cases for
            code verification.
        format_pattern: Optional regex pattern that completions
            must match to receive a format reward.
        format_reward_weight: Weight for the format compliance
            reward (combined with content rewards).
        length_penalty_target: Target completion length for
            the length penalty verifier. Set to ``None`` to disable;
            ``0`` is accepted for backward compatibility.
        length_penalty_weight: Weight for the length penalty
            reward component.
        max_len_mask: If True, mask out completions that hit
            the maximum length (no EOS token) from the loss.
        reward_clip_range: Clip raw rewards to
            ``[-reward_clip_range, reward_clip_range]``.
            Set to ``None`` to disable clipping; ``0`` is accepted for
            backward compatibility.
        difficulty_key: Optional column name for per-sample
            difficulty scores used in weighted loss computation.
        difficulty_loss_weight: If True, weight the loss by
            the difficulty score of each sample.

    Example:
        >>> config = RLVRConfig(
        ...     max_prompt_length=1024,
        ...     max_completion_length=2048,
        ...     num_return_sequences=8,
        ...     answer_key="answer",
        ...     loss_type="dapo",
        ...     beta=0.04,
        ... )
    """

    trainer_prefix: str | None = field(
        default="RLVR",
        metadata={"help": "Default prefix name for trainer."},
    )

    answer_key: str | None = field(
        default="answer",
        metadata={
            "help": "Dataset column containing gold answers for math verification. Set to None to disable the math verifier."
        },
    )
    test_key: str = field(
        default="tests",
        metadata={"help": "Dataset column containing code test cases."},
    )
    format_pattern: str | None = field(
        default=None,
        metadata={"help": "Regex pattern completions must match for format reward."},
    )
    format_reward_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for format compliance reward component."},
    )
    length_penalty_target: int | None = field(
        default=None,
        metadata={
            "help": "Target length for length penalty verifier. `None` disables; `0` is accepted for backward compatibility."
        },
    )
    length_penalty_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for length penalty reward component."},
    )
    max_len_mask: bool = field(
        default=True,
        metadata={"help": "Mask completions without EOS from the loss."},
    )
    reward_clip_range: float | None = field(
        default=None,
        metadata={
            "help": "Clip rewards to [-range, range]. `None` disables; `0` is accepted for backward compatibility."
        },
    )
    difficulty_key: str | None = field(
        default=None,
        metadata={"help": "Dataset column for per-sample difficulty scores."},
    )
    difficulty_loss_weight: bool = field(
        default=False,
        metadata={"help": "Weight loss by per-sample difficulty."},
    )

    def __post_init__(
        self,
        max_sequence_length: int | None,
        quantization_block: int | None,
    ):
        super().__post_init__(
            max_sequence_length=max_sequence_length,
            quantization_block=quantization_block,
        )
        if self.length_penalty_target is not None:
            normalized_length_penalty_target = int(self.length_penalty_target)
            self.length_penalty_target = (
                normalized_length_penalty_target if normalized_length_penalty_target > 0 else None
            )
        if self.reward_clip_range is not None:
            normalized_reward_clip_range = float(self.reward_clip_range)
            self.reward_clip_range = normalized_reward_clip_range if normalized_reward_clip_range > 0.0 else None
        if self.mask_truncated_completions is False and self.max_len_mask:
            self.mask_truncated_completions = True

    __hash__ = hash_fn
