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

"""Configuration for Agentic MoshPit training.

This module defines AgenticMoshPitConfig, which extends GRPOConfig with
parameters for multi-turn environment interaction, tool-calling,
and agentic reward computation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..group_relative_policy_optimization.grpo_config import GRPOConfig


@Registry.register("trainer-arguments", "agentic-moshpit")
@dataclass
class AgenticMoshPitConfig(GRPOConfig):
    """Configuration class for Agentic MoshPit training.

    Extends GRPOConfig with parameters for multi-turn environment interaction,
    where the model acts as an agent interacting with an environment over
    multiple turns. Supports tool-calling, step-level rewards, and various
    advantage estimation strategies.

    The agentic training loop:
    1. Reset environment -> get task observation
    2. Generate response (with optional tool calls)
    3. Step environment -> get next observation + reward
    4. Repeat until episode ends
    5. Compute advantages and update policy (GRPO loss)

    Key additions over standard GRPO:
    - Multi-turn environment interaction instead of single-turn generation
    - Support for step-level rewards and discounted returns
    - Tool-calling support with configurable tool schemas
    - Multiple advantage estimators (episode, step, gigpo, agentic_reinforce)

    Attributes:
        max_steps: Maximum number of environment steps per episode.
        group_size: Number of rollouts per environment seed for GRPO grouping.
            Different rollouts with the same seed provide variance for
            group-relative advantage computation.
        reward_mode: How rewards are computed:
            - "episode": Terminal reward only (standard GRPO)
            - "step": Discounted per-step returns
            - "gigpo": Combined episode + step rewards
        advantage_estimator: Advantage computation method:
            - "grpo": Standard group-relative (episode-level)
            - "reinforce": REINFORCE with baseline subtraction
            - "gigpo": GiGPO combined episode + step
            - "step_reinforce": Step-level discounted returns
            - "agentic_reinforce": Segment-aware discounted returns
        step_reward_gamma: Discount factor for step-level rewards.
        episode_reward_weight: Weight for episode-level rewards in GiGPO.
        step_reward_weight: Weight for step-level rewards in GiGPO.
        system_prompt: System prompt prepended to all agent conversations.
        tool_names: List of registered tool names to make available.
        tool_schemas: Explicit tool schemas (overrides tool_names).
        tool_caller: Tool call parser identifier. Can be either:
            - A registered parser name from ``easydel.inference.tools``
              (e.g., "hermes", "openai", "qwen3_coder", "mistral", etc.)
            - A regex pattern string prefixed with "regex:" that matches
              tool calls in the model output (e.g., "regex:<tool_call>(.*?)</tool_call>")
        max_tool_calls_per_step: Maximum tool calls allowed per agent turn.
        num_env_groups: Number of distinct environment seeds per batch.
            Total rollouts = num_env_groups * group_size.
    """

    trainer_prefix: str | None = field(
        default="AgenticMoshPit",
        metadata={"help": "Default prefix name for trainer."},
    )

    max_steps: int = field(
        default=10,
        metadata={"help": "Maximum number of environment steps per episode."},
    )
    group_size: int = field(
        default=4,
        metadata={"help": "Number of rollouts per environment seed for grouping."},
    )
    num_env_groups: int = field(
        default=4,
        metadata={"help": "Number of distinct environment seeds per training batch."},
    )

    reward_mode: str = field(
        default="episode",
        metadata={"help": "Reward mode: 'episode', 'step', or 'gigpo'."},
    )
    advantage_estimator: str = field(
        default="grpo",
        metadata={"help": "Advantage estimator: 'grpo', 'reinforce', 'gigpo', 'step_reinforce', 'agentic_reinforce'."},
    )
    step_reward_gamma: float = field(
        default=0.95,
        metadata={"help": "Discount factor for step-level rewards."},
    )
    episode_reward_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for episode-level rewards in GiGPO mode."},
    )
    step_reward_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for step-level rewards in GiGPO mode."},
    )

    system_prompt: str | None = field(
        default=None,
        metadata={"help": "System prompt prepended to agent conversations."},
    )
    tool_names: list[str] | None = field(
        default=None,
        metadata={"help": "List of registered tool names to make available."},
    )
    tool_schemas: list[dict] | None = field(
        default=None,
        metadata={"help": "Explicit tool schemas for chat template formatting."},
    )
    tool_caller: str | None = field(
        default=None,
        metadata={
            "help": (
                "Tool call parser. Either a registered parser name from "
                "easydel.inference.tools (e.g., 'hermes', 'openai', 'qwen3_coder'), "
                "or a regex pattern prefixed with 'regex:' for custom extraction."
            )
        },
    )
    max_tool_calls_per_step: int = field(
        default=5,
        metadata={"help": "Maximum tool calls allowed per agent turn."},
    )
    reasoning_parser: str | None = field(
        default=None,
        metadata={
            "help": (
                "Reasoning parser name from easydel.inference.reasoning "
                "(e.g., 'qwen3', 'deepseek_r1', 'mistral', 'granite'). "
                "Determines which start/end tags are used for reasoning blocks. "
                "If None, auto-detected from model architecture."
            )
        },
    )

    def validate_agentic_knobs(self) -> None:
        """Validate Agentic MoshPit bounds and reject unsupported inherited GRPO knobs.

        Called both from ``__post_init__`` and from ``AgenticMoshPitTrainer.__init__``
        so post-construction config mutations are also caught before training starts.
        """
        for field_name in ("max_steps", "group_size", "num_env_groups"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"`{field_name}` must be a positive integer; got {value!r}.")

        gamma = self.step_reward_gamma
        if isinstance(gamma, bool) or not isinstance(gamma, int | float) or not math.isfinite(gamma) or not (0.0 <= gamma <= 1.0):
            raise ValueError(f"`step_reward_gamma` must be a finite float in [0.0, 1.0]; got {gamma!r}.")

        if self.num_iterations > 1:
            raise ValueError(
                f"AgenticMoshPit does not support `num_iterations > 1` (got {self.num_iterations}); "
                "agentic rollouts perform one on-policy update per collected batch."
            )
        if self.steps_per_generation is not None and self.steps_per_generation > 1:
            raise ValueError(
                f"AgenticMoshPit does not support `steps_per_generation > 1` (got {self.steps_per_generation})."
            )
        if self.mask_truncated_completions:
            raise ValueError(
                "AgenticMoshPit does not support `mask_truncated_completions=True`; "
                "multi-turn episodes track truncation per episode rather than single-turn EOS."
            )
        if self.off_policy_mask_threshold is not None:
            raise ValueError(
                "AgenticMoshPit does not support `off_policy_mask_threshold`; "
                f"got {self.off_policy_mask_threshold!r}."
            )
        divergence_type = getattr(self, "divergence_type", None)
        if self.loss_type == "dppo" and divergence_type in {"topk_tv", "topk_kl"}:
            raise ValueError(
                f"AgenticMoshPit does not support DPPO top-k divergence ({divergence_type!r}); "
                "multi-turn rollouts do not emit top-k sampling log-prob tensors."
            )

    def __post_init__(
        self,
        max_sequence_length: int | None,
        quantization_block: int | None,
    ):
        """Finalize Agentic-MoshPit-specific config invariants.

        Aligns ``num_generations`` and ``num_return_sequences`` with the
        configured ``group_size`` when they were left at their defaults,
        delegates to the GRPO base for shared validation, then validates
        bounds, ``reward_mode``, ``advantage_estimator``, and unsupported
        inherited GRPO knobs.

        Args:
            max_sequence_length: Forwarded to the GRPO ``__post_init__``
                for legacy alias handling.
            quantization_block: Forwarded to the GRPO ``__post_init__``
                for legacy alias handling.

        Raises:
            ValueError: If any Agentic MoshPit bound, mode, estimator, or
                unsupported GRPO knob is invalid.
        """
        if isinstance(self.group_size, bool) or not isinstance(self.group_size, int) or self.group_size <= 0:
            raise ValueError(f"`group_size` must be a positive integer; got {self.group_size!r}.")

        if self.num_generations is None and self.num_return_sequences == 4:
            self.num_return_sequences = self.group_size
            self.num_generations = self.group_size

        super().__post_init__(
            max_sequence_length=max_sequence_length,
            quantization_block=quantization_block,
        )

        valid_reward_modes = ("episode", "step", "gigpo")
        if self.reward_mode not in valid_reward_modes:
            raise ValueError(f"Invalid reward_mode: {self.reward_mode}. Must be one of {valid_reward_modes}.")

        valid_estimators = ("grpo", "reinforce", "gigpo", "step_reinforce", "agentic_reinforce")
        if self.advantage_estimator not in valid_estimators:
            raise ValueError(
                f"Invalid advantage_estimator: {self.advantage_estimator}. Must be one of {valid_estimators}."
            )
        if self.reward_mode != "episode" and self.advantage_estimator == "grpo":
            self.advantage_estimator = {
                "step": "step_reinforce",
                "gigpo": "gigpo",
            }[self.reward_mode]

        self.validate_agentic_knobs()

    __hash__ = hash_fn
