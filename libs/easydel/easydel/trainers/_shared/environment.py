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

"""Shared text environments, tool-call parsers, and guarded tool wrappers.

Environments follow a Gymnasium-like interface where the agent interacts
through text observations and actions. Rewards may be terminal or per-step.
"""

from __future__ import annotations

import abc
import json
import typing as tp
from dataclasses import dataclass, field

from ..agentic_rollout.dispatcher import ToolDispatcher
from ..agentic_rollout.guards import TwoStageToolGuard
from ..agentic_rollout.schema import GuardContext, NormalizedToolCall

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

    def reset_batch(self, environments: list[AgenticEnvironment], seeds: list[int | None]) -> list[ResetResult]:
        """Reset a group, allowing environments to batch expensive backend calls."""
        return [env.reset(seed=seed) for env, seed in zip(environments, seeds, strict=True)]

    def finalize_batch(self, environments: list[AgenticEnvironment]) -> list[StepResult | None]:
        """Resolve deferred terminal rewards; ordinary environments need no work."""
        return [None for _ in environments]

    def configure_generation(
        self,
        generate_fn: tp.Callable,
        *,
        tokenizer: tp.Any = None,
        strip_reasoning_fn: tp.Callable[[str], str] | None = None,
    ) -> None:
        """Attach optional generation services used by model-backed environments."""
        return None

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
        """Extract ``(tool_name, tool_args_json)`` pairs from a regex match.

        Args:
            action: Raw model output potentially containing tool calls.

        Returns:
            A list of ``(name, args_json_string)`` tuples for every
            successful match; malformed JSON entries are skipped.
        """
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
        """Delegate to a registered ``easydel.inference.tools`` parser.

        Args:
            action: Raw model output to inspect.

        Returns:
            A list of ``(name, args_json_string)`` tuples for every tool
            call recognised by the registered parser; an empty list when
            the parser reports no tools.
        """
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

    This wrapper intercepts the agent's actions, checks for tool calls,
    executes them, and injects the results as observations. Actions with
    no tool calls are forwarded to the underlying environment's ``step``.

    Args:
        env: The underlying environment to wrap.
        tools: List of Tool instances available to the agent.
        tool_call_parser: Extracts ``(tool_name, tool_args_json)`` tuples.
        max_tool_calls_per_step: Maximum tool calls in a single turn.
    """

    def __init__(
        self,
        env: AgenticEnvironment,
        tools: list[Tool],
        tool_call_parser: tp.Callable[[str], list[tuple[str, str]]] | None = None,
        max_tool_calls_per_step: int = 5,
        tool_guard: TwoStageToolGuard | None = None,
    ):
        """Cache the inner environment and tool dispatch configuration.

        Args:
            env: The base :class:`AgenticEnvironment` being wrapped.
            tools: Registered :class:`Tool` instances, keyed by name.
            tool_call_parser: Text parser; defaults to a JSON detector.
            max_tool_calls_per_step: Hard cap on invocations per turn.
            tool_guard: Optional guard applied before tool execution.
        """
        self.env = env
        self.tools = {tool.name: tool for tool in tools}
        self.tool_call_parser = tool_call_parser or _default_tool_call_parser
        self.max_tool_calls_per_step = max_tool_calls_per_step
        self._tool_calls_this_step = 0
        self.guard_context = GuardContext(episode_id="uninitialized")

        def executor(tool):
            return lambda **kwargs: tool.execute(json.dumps(kwargs))

        self.dispatcher = ToolDispatcher({name: executor(tool) for name, tool in self.tools.items()}, guard=tool_guard)

    def reset_batch(self, environments: list[AgenticEnvironment], seeds: list[int | None]) -> list[ResetResult]:
        """Preserve inner batching and reset wrapper-local episode state."""
        if not all(isinstance(env, ToolEnvWrapper) for env in environments):
            return super().reset_batch(environments, seeds)
        reset_batch = getattr(self.env, "reset_batch", None)
        if not callable(reset_batch):
            return super().reset_batch(environments, seeds)
        results = reset_batch([env.env for env in environments], seeds)
        for env, seed, result in zip(environments, seeds, results, strict=True):
            env._tool_calls_this_step = 0
            env._reset_tools()
            env.guard_context = GuardContext(episode_id=str(seed), intent=result.observation)
        return results

    def finalize_batch(self, environments: list[AgenticEnvironment]) -> list[StepResult | None]:
        """Forward deferred terminal reward resolution to the wrapped environments."""
        if not all(isinstance(env, ToolEnvWrapper) for env in environments):
            return super().finalize_batch(environments)
        finalize_batch = getattr(self.env, "finalize_batch", None)
        if not callable(finalize_batch):
            return super().finalize_batch(environments)
        return finalize_batch([env.env for env in environments])

    def configure_generation(self, generate_fn: tp.Callable, **kwargs) -> None:
        """Forward generation services through tool wrappers."""
        self.env.configure_generation(generate_fn, **kwargs)

    def _reset_tools(self) -> None:
        """Reset episode-owned tools without requiring structural tools to subclass Tool."""
        for tool in self.tools.values():
            reset = getattr(tool, "reset_episode", None)
            if callable(reset):
                reset()

    def reset(self, seed: int | None = None) -> ResetResult:
        """Reset the wrapped environment, episode-local tools and per-turn counters.

        Args:
            seed: Optional seed forwarded to the inner environment.

        Returns:
            The :class:`ResetResult` produced by the inner environment.
        """
        self._tool_calls_this_step = 0
        self._reset_tools()
        result = self.env.reset(seed=seed)
        self.guard_context = GuardContext(episode_id=str(seed), intent=result.observation)
        return result

    @staticmethod
    def _coerce_tool_args_json(tool_args: tp.Any) -> str:
        """Coerce tool arguments into JSON, preserving already encoded strings."""
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
        """Flatten dicts, objects, and tuples into ``(name, args_json)`` pairs."""
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
        """Execute up to the per-turn cap and return a tool-result observation."""
        if not parsed_calls or self._tool_calls_this_step >= self.max_tool_calls_per_step:
            return None

        selected = [(name, args) for name, args in parsed_calls if name in self.tools][
            : self.max_tool_calls_per_step - self._tool_calls_this_step
        ]
        if not selected:
            return None
        try:
            canonical_calls = []
            for index, (name, args) in enumerate(selected):
                tool = self.tools[name]
                normalize = getattr(tool, "normalize_arguments", None)
                if callable(normalize):
                    arguments = normalize(args)
                else:
                    # Structural tools own their serialized-argument protocol.
                    # The schema supplies a meaningful guard key for raw/scalar
                    # inputs, but execution must still receive the original text.
                    try:
                        arguments = json.loads(args)
                    except json.JSONDecodeError:
                        arguments = args
                    if not isinstance(arguments, dict):
                        properties = tool.chat_schema.get("parameters", {}).get("properties", {})
                        arguments = {next(iter(properties), "arguments"): arguments}
                canonical_calls.append(NormalizedToolCall.create(name, arguments, call_id=str(index)))

            def execute(call):
                name, args = selected[int(call.call_id)]
                return self.tools[name].execute(args)

            dispatched = self.dispatcher.dispatch_many(canonical_calls, self.guard_context, execute=execute)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return StepResult(observation=f"Error: invalid tool arguments: {exc}")
        self._tool_calls_this_step += len(dispatched)
        return StepResult(
            observation="\n".join(f"[{result.call.name}]: {result.observation}" for result in dispatched),
            reward=sum(result.execution.reward for result in dispatched),
            terminated=any(result.terminated for result in dispatched),
            truncated=any(result.truncated for result in dispatched),
            info={
                "tool_calls": [{"name": name, "args": args} for name, args in selected],
                "guard_events": tuple(self.guard_context.events),
            },
        )

    def step_with_tool_calls(
        self,
        action: str,
        *,
        tool_calls: tp.Any | None = None,
    ) -> StepResult:
        """Execute structured tool calls, then text calls, or forward the action.

        Args:
            action: The agent's raw text action.
            tool_calls: Optional pre-parsed structured tool-call list.

        Returns:
            Tool output observations or the inner environment's response.
        """
        self._tool_calls_this_step = 0
        self.guard_context.advance_step()
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
        """Step without pre-parsed tool-call hints."""
        return self.step_with_tool_calls(action)

    def close(self) -> None:
        """Close the inner environment, releasing its resources."""
        self.env.close()

    @property
    def max_steps(self) -> int | None:
        """Return the underlying environment's ``max_steps`` (if any)."""
        return self.env.max_steps
