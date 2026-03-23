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

"""Environment abstractions for agentic MoshPit training.

This module defines the core environment protocol and wrappers for
multi-turn agent training. Environments follow a Gymnasium-like interface
where the agent interacts through text observations and actions.

The module provides:
- AgenticEnvironment: Protocol defining the reset/step interface
- BaseAgenticEnvironment: Abstract base class with common utilities
- ToolEnvWrapper: Wraps an environment with tool-calling support

Environments produce text observations and accept text actions (model outputs).
Rewards can be sparse (terminal only) or dense (per-step).
"""

from __future__ import annotations

import abc
import json
import typing as tp
from dataclasses import dataclass, field

if tp.TYPE_CHECKING:
    from .tools import Tool


@dataclass
class StepResult:
    """Result from a single environment step.

    Attributes:
        observation: Text observation returned by the environment.
        reward: Scalar reward for this step (0.0 if sparse reward mode).
        terminated: Whether the episode ended (goal reached or failed).
        truncated: Whether the episode was cut short (e.g., max steps).
        info: Additional metadata from the environment.
    """

    observation: str
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: dict[str, tp.Any] = field(default_factory=dict)


@dataclass
class ResetResult:
    """Result from an environment reset.

    Attributes:
        observation: Initial text observation (task description / prompt).
        info: Additional metadata (e.g., ground truth, difficulty).
    """

    observation: str
    info: dict[str, tp.Any] = field(default_factory=dict)


class AgenticEnvironment(abc.ABC):
    """Abstract base class for agentic training environments.

    Environments provide a text-based interaction interface where:
    1. ``reset(seed)`` initializes an episode and returns an observation
    2. ``step(action)`` processes the agent's text response and returns
       the next observation, reward, and termination signals

    Subclass this to create custom environments for math, code, QA,
    tool-use, game-playing, or any multi-turn task.

    Example:
        >>> class MathEnv(AgenticEnvironment):
        ...     def reset(self, seed=None):
        ...         self.answer = 42
        ...         return ResetResult("What is 6 * 7?")
        ...     def step(self, action):
        ...         correct = "42" in action
        ...         return StepResult(
        ...             observation="Correct!" if correct else "Wrong.",
        ...             reward=1.0 if correct else 0.0,
        ...             terminated=True,
        ...         )
    """

    @abc.abstractmethod
    def reset(self, seed: int | None = None) -> ResetResult:
        """Reset the environment and return an initial observation.

        Args:
            seed: Optional random seed for deterministic episode generation.
                Different seeds should produce different episodes.

        Returns:
            ResetResult with initial observation and metadata.
        """

    @abc.abstractmethod
    def step(self, action: str) -> StepResult:
        """Process the agent's action and return the next state.

        Args:
            action: The agent's text response (decoded model output).

        Returns:
            StepResult with next observation, reward, and termination flags.
        """

    def close(self) -> None:  # noqa: B027
        """Clean up any resources held by the environment."""

    @property
    def max_steps(self) -> int | None:
        """Optional maximum number of steps per episode.

        Returns None if there's no intrinsic step limit (the trainer's
        config ``max_steps`` will be used instead).
        """
        return None


def create_tool_call_parser(
    tool_caller: str | None,
    tokenizer: tp.Any | None = None,
) -> tp.Callable[[str], list[tuple[str, str]]]:
    """Create a tool call parser from a tool_caller specification.

    The ``tool_caller`` can be:
    - A registered parser name from ``easydel.inference.tools``
      (e.g., "hermes", "openai", "qwen3_coder", "mistral", etc.).
      These use the same parsers as eSurge inference.
    - A regex pattern prefixed with ``"regex:"`` for custom extraction.
      The regex should capture a JSON object containing "name" and
      "arguments" keys.
    - ``None`` to use the default hermes-style parser.

    Args:
        tool_caller: Parser identifier string or None.
        tokenizer: Tokenizer instance (required for registered parsers).

    Returns:
        Callable that takes model output text and returns a list of
        ``(tool_name, tool_args_json)`` tuples. Returns empty list if
        no tool calls are found.
    """
    if tool_caller is None:
        return _default_tool_call_parser

    if tool_caller.startswith("regex:"):
        pattern = tool_caller[len("regex:") :]
        return _make_regex_parser(pattern)

    return _make_inference_tool_parser(tool_caller, tokenizer)


def _default_tool_call_parser(action: str) -> list[tuple[str, str]]:
    """Parse tool calls in hermes format: <tool_call>{"name": "...", "arguments": ...}</tool_call>."""
    import json
    import re

    results = []
    for match in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", action, re.DOTALL):
        try:
            data = json.loads(match.group(1))
            name = data.get("name", "")
            args = json.dumps(data.get("arguments", {}))
            results.append((name, args))
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def _make_regex_parser(pattern: str) -> tp.Callable[[str], list[tuple[str, str]]]:
    """Create a parser from a regex pattern."""
    import json
    import re

    compiled = re.compile(pattern, re.DOTALL)

    def parser(action: str) -> list[tuple[str, str]]:
        results = []
        for match in compiled.finditer(action):
            try:
                text = match.group(1) if match.lastindex else match.group(0)
                data = json.loads(text)
                name = data.get("name", "")
                args = json.dumps(data.get("arguments", {}))
                results.append((name, args))
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
        return results

    return parser


def _make_inference_tool_parser(
    parser_name: str,
    tokenizer: tp.Any | None,
) -> tp.Callable[[str], list[tuple[str, str]]]:
    """Create a parser using easydel.inference.tools registered parsers."""
    from easydel.inference.tools import ToolParserManager

    parser_cls = ToolParserManager.get_tool_parser(parser_name)
    if tokenizer is None:
        raise ValueError(
            f"Tokenizer is required when using registered tool parser '{parser_name}'. Pass a tokenizer to the trainer."
        )
    parser_instance = parser_cls(tokenizer)

    def parser(action: str) -> list[tuple[str, str]]:
        result = parser_instance.extract_tool_calls(action, request=None)
        if not result.tools_called:
            return []
        calls = []
        for tc in result.tool_calls:
            name = tc.function.name or ""
            args = tc.function.arguments or "{}"
            calls.append((name, args))
        return calls

    return parser


class ToolEnvWrapper(AgenticEnvironment):
    """Wraps an environment with tool-calling capabilities.

    This wrapper intercepts the agent's actions, checks for tool calls
    (function calls in the model output), executes them, and injects
    the tool results as observations before passing to the underlying
    environment.

    When a tool call is detected, the wrapper:
    1. Parses tool calls from the action text using the configured parser
       (from ``easydel.inference.tools`` or a regex pattern)
    2. Executes each tool and captures the result
    3. Returns the tool results as the observation (no env.step)
    4. The agent then gets another turn to incorporate the tool results

    When no tool call is detected, the action is forwarded to the
    underlying environment's ``step()`` method.

    Args:
        env: The underlying environment to wrap.
        tools: List of Tool instances available to the agent.
        tool_call_parser: Callable that extracts tool calls from model output.
            Returns a list of ``(tool_name, tool_args_json)`` tuples.
            Use ``create_tool_call_parser()`` to create one from a
            ``tool_caller`` config value.
        max_tool_calls_per_step: Maximum tool calls allowed in a single agent turn.
    """

    def __init__(
        self,
        env: AgenticEnvironment,
        tools: list[Tool],
        tool_call_parser: tp.Callable[[str], list[tuple[str, str]]] | None = None,
        max_tool_calls_per_step: int = 5,
    ):
        self.env = env
        self.tools = {tool.name: tool for tool in tools}
        self.tool_call_parser = tool_call_parser or _default_tool_call_parser
        self.max_tool_calls_per_step = max_tool_calls_per_step
        self._tool_calls_this_step = 0

    def reset(self, seed: int | None = None) -> ResetResult:
        self._tool_calls_this_step = 0
        return self.env.reset(seed=seed)

    @staticmethod
    def _coerce_tool_args_json(tool_args: tp.Any) -> str:
        if tool_args is None:
            return "{}"
        if isinstance(tool_args, str):
            return tool_args
        try:
            return json.dumps(tool_args)
        except (TypeError, ValueError):
            return str(tool_args)

    def _normalize_structured_tool_calls(
        self,
        tool_calls: tp.Any | None,
    ) -> list[tuple[str, str]]:
        if not isinstance(tool_calls, list):
            return []

        normalized: list[tuple[str, str]] = []
        for call in tool_calls:
            name = ""
            args: tp.Any = "{}"
            if isinstance(call, tuple) and len(call) == 2:
                name, args = call
            elif isinstance(call, dict):
                function = call.get("function")
                if isinstance(function, dict):
                    name = function.get("name") or call.get("name") or ""
                    args = (
                        function.get("arguments")
                        if function.get("arguments") is not None
                        else call.get("arguments", call.get("args", "{}"))
                    )
                else:
                    name = call.get("name") or ""
                    args = call.get("arguments", call.get("args", "{}"))
            else:
                function = getattr(call, "function", None)
                if function is not None:
                    name = getattr(function, "name", "") or getattr(call, "name", "") or ""
                    args = getattr(function, "arguments", None)
                    if args is None:
                        args = getattr(call, "arguments", getattr(call, "args", "{}"))
                else:
                    name = getattr(call, "name", "") or ""
                    args = getattr(call, "arguments", getattr(call, "args", "{}"))

            if name:
                normalized.append((name, self._coerce_tool_args_json(args)))
        return normalized

    def _execute_tool_calls(self, parsed_calls: list[tuple[str, str]]) -> StepResult | None:
        if not parsed_calls or self._tool_calls_this_step >= self.max_tool_calls_per_step:
            return None

        results = []
        for tool_name, tool_args in parsed_calls:
            if tool_name in self.tools and self._tool_calls_this_step < self.max_tool_calls_per_step:
                self._tool_calls_this_step += 1
                result = self.tools[tool_name].execute(tool_args)
                results.append(f"[{tool_name}]: {result}")
        if not results:
            return None

        return StepResult(
            observation="\n".join(results),
            reward=0.0,
            terminated=False,
            truncated=False,
            info={
                "tool_calls": [{"name": n, "args": a} for n, a in parsed_calls if n in self.tools],
            },
        )

    def step_with_tool_calls(
        self,
        action: str,
        *,
        tool_calls: tp.Any | None = None,
    ) -> StepResult:
        self._tool_calls_this_step = 0
        parsed_calls = self._normalize_structured_tool_calls(tool_calls)
        step_result = self._execute_tool_calls(parsed_calls)
        if step_result is not None:
            return step_result

        parsed_calls = self.tool_call_parser(action)
        step_result = self._execute_tool_calls(parsed_calls)
        if step_result is not None:
            return step_result

        return self.env.step(action)

    def step(self, action: str) -> StepResult:
        return self.step_with_tool_calls(action)

    def close(self) -> None:
        self.env.close()

    @property
    def max_steps(self) -> int | None:
        return self.env.max_steps
