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

"""Agentic MoshPit trainer module for EasyDeL.

This module implements Agentic MoshPit, a multi-turn reinforcement learning
approach where the language model interacts with environments as an agent.
Unlike standard GRPO which does single-turn generation + reward scoring,
Agentic MoshPit runs full episode rollouts with multi-turn interactions.

The module includes:
- AgenticMoshPitConfig: Configuration with environment and tool parameters
- AgenticMoshPitTrainer: Trainer with multi-turn rollout loop
- AgenticEnvironment: Abstract environment interface (reset/step)
- Tool system: Registry of tools the agent can call during episodes
- RolloutManager: Orchestrates multi-turn agent-environment interaction
- Advantage utilities: Episode-level, step-level, GiGPO, and more

Inspired by Alibaba's ROLL agentic pipeline, adapted for EasyDeL's
JAX-native distributed training architecture.

Key Features:
- Multi-turn environment interaction with tool-calling support
- Multiple advantage estimators (GRPO, step, GiGPO, agentic_reinforce)
- Dense and sparse reward modes
- Group-relative advantage computation for stable training
- Reuses GRPO's proven loss function (dapo, grpo, bnpo, cispo, etc.)

Example:
    >>> from easydel.trainers import AgenticMoshPitConfig, AgenticMoshPitTrainer
    >>> from easydel.trainers.agentic_moshpit import AgenticEnvironment, ResetResult, StepResult
    >>>
    >>> class MathEnv(AgenticEnvironment):
    ...     def reset(self, seed=None):
    ...         return ResetResult("What is 6 * 7?")
    ...     def step(self, action):
    ...         return StepResult("", 1.0 if "42" in action else 0.0, terminated=True)
    ...
    >>> config = AgenticMoshPitConfig(
    ...     max_steps=5,
    ...     group_size=4,
    ...     num_env_groups=8,
    ... )
    >>> trainer = AgenticMoshPitTrainer(
    ...     arguments=config,
    ...     model=model,
    ...     env_factory=lambda: MathEnv(),
    ...     processing_class=tokenizer,
    ... )
    >>> trainer.train()
"""

from .agentic_moshpit_config import AgenticMoshPitConfig
from .agentic_moshpit_trainer import AgenticMoshPitTrainer
from .env_manager import RolloutManager, TrajectoryResult, TurnRecord
from .environment import (
    AgenticEnvironment,
    ResetResult,
    StepResult,
    ToolEnvWrapper,
    create_tool_call_parser,
)
from .self_play import (
    CallableQuestionGenerator,
    GeneratedQuestion,
    LocalQuestionGenerator,
    OpenAIQuestionGenerator,
    QuestionGenerator,
    SelfPlayEnvironment,
)
from .tools import (
    BashTool,
    CalculatorTool,
    FileReadTool,
    FunctionTool,
    JSONProcessorTool,
    NotepadTool,
    PythonCodeTool,
    RegexTool,
    Tool,
    UnitConverterTool,
    WebFetchTool,
    WikipediaTool,
    function_to_json,
    list_tools,
    make_tool,
    register_tool,
)
from .utils import (
    compute_advantages_episode,
    compute_advantages_gigpo,
    compute_advantages_step,
    compute_discounted_returns,
    normalize_rewards_batch,
    normalize_rewards_group,
)

__all__ = (
    "AgenticEnvironment",
    "AgenticMoshPitConfig",
    "AgenticMoshPitTrainer",
    "BashTool",
    "CalculatorTool",
    "CallableQuestionGenerator",
    "FileReadTool",
    "FunctionTool",
    "GeneratedQuestion",
    "JSONProcessorTool",
    "LocalQuestionGenerator",
    "NotepadTool",
    "OpenAIQuestionGenerator",
    "PythonCodeTool",
    "QuestionGenerator",
    "RegexTool",
    "ResetResult",
    "RolloutManager",
    "SelfPlayEnvironment",
    "StepResult",
    "Tool",
    "ToolEnvWrapper",
    "TrajectoryResult",
    "TurnRecord",
    "UnitConverterTool",
    "WebFetchTool",
    "WikipediaTool",
    "compute_advantages_episode",
    "compute_advantages_gigpo",
    "compute_advantages_step",
    "compute_discounted_returns",
    "create_tool_call_parser",
    "function_to_json",
    "list_tools",
    "make_tool",
    "normalize_rewards_batch",
    "normalize_rewards_group",
    "register_tool",
)
