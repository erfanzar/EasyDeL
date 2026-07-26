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

"""Centralized guarded dispatch for agentic rollout tools."""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field

from .guards import GuardAction, GuardDecision, TwoStageToolGuard
from .schema import (
    AtomicInteractionStep,
    GuardContext,
    NormalizedToolCall,
    normalize_tool_calls,
)

BLOCKED_TOOL_OBSERVATION = "Tool call blocked by policy. Continue without a tool result."


@dataclass(frozen=True, slots=True)
class ToolExecution:
    """Normalized result returned by a tool executor."""

    observation: str = field(repr=False)
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: tp.Mapping[str, tp.Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class DispatchResult:
    """Guard and execution result for one normalized call."""

    call: NormalizedToolCall = field(repr=False)
    execution: ToolExecution
    blocked: bool
    guard_decision: GuardDecision | None = None

    @property
    def observation(self) -> str:
        """Return the observation inserted into the rollout."""
        return self.execution.observation

    @property
    def terminated(self) -> bool:
        """Whether an allowed tool result terminates the environment."""
        return self.execution.terminated

    @property
    def truncated(self) -> bool:
        """Whether an allowed tool result truncates the environment."""
        return self.execution.truncated


ToolExecutor = tp.Callable[..., tp.Any]


class ToolDispatcher:
    """Resolve, guard, and execute all rollout tool calls in one place.

    Args:
        tools: Mapping from normalized tool names to callables or objects with
            a callable ``call`` method. Executors receive decoded JSON
            arguments as keyword arguments.
        guard: Optional two-stage guard. Without one, calls are allowed.
        blocked_observation: Stable dummy observation returned for every
            blocked call. It must not include the call or its arguments.
    """

    def __init__(
        self,
        tools: tp.Mapping[str, ToolExecutor | tp.Any],
        *,
        guard: TwoStageToolGuard | None = None,
        blocked_observation: str = BLOCKED_TOOL_OBSERVATION,
    ) -> None:
        if not blocked_observation:
            raise ValueError("blocked_observation must not be empty")
        self._tools = dict(tools)
        self.guard = guard
        self.blocked_observation = blocked_observation

    def dispatch(self, call: tp.Any, context: GuardContext) -> DispatchResult:
        """Dispatch one tool call through the batched implementation."""
        return self.dispatch_many((call,), context)[0]

    def dispatch_many(
        self,
        calls: tp.Iterable[tp.Any],
        context: GuardContext,
    ) -> tuple[DispatchResult, ...]:
        """Guard all calls as a batch, then execute only allowed calls."""
        normalized = normalize_tool_calls(calls)
        if not normalized:
            return ()
        if self.guard is None:
            decisions: tuple[GuardDecision | None, ...] = (None,) * len(normalized)
        else:
            decisions = self.guard.evaluate_many(normalized, context)

        results = []
        for call, decision in zip(normalized, decisions, strict=True):
            if decision is not None and decision.action is GuardAction.BLOCK:
                results.append(
                    DispatchResult(
                        call=call,
                        execution=ToolExecution(
                            observation=self.blocked_observation,
                            terminated=False,
                            truncated=False,
                            info={
                                "blocked": True,
                                "call_hash": decision.event.call_hash,
                                "reason_code": decision.reason_code,
                            },
                        ),
                        blocked=True,
                        guard_decision=decision,
                    )
                )
                continue

            results.append(
                DispatchResult(
                    call=call,
                    execution=self._execute(call),
                    blocked=False,
                    guard_decision=decision,
                )
            )
        return tuple(results)

    def dispatch_step(
        self,
        assistant_action: str,
        calls: tp.Iterable[tp.Any],
        context: GuardContext,
        *,
        assistant_token_ids: tp.Sequence[int] = (),
        encode_observation: tp.Callable[[str], tp.Sequence[int]] | None = None,
        observation_separator: str = "\n",
    ) -> AtomicInteractionStep:
        """Dispatch calls and return one atomic action-observation step.

        All calls in an assistant action share one guard step index. The
        context advances only after the complete pair is constructed.
        """
        normalized = normalize_tool_calls(calls)
        results = self.dispatch_many(normalized, context)
        observation = observation_separator.join(result.observation for result in results)
        observation_token_ids = () if encode_observation is None else tuple(encode_observation(observation))
        info = {
            "dispatch": tuple(
                {
                    "blocked": result.blocked,
                    "call_hash": result.call.fingerprint,
                }
                for result in results
            )
        }
        step = AtomicInteractionStep(
            assistant_action=assistant_action,
            observation=observation,
            assistant_token_ids=tuple(assistant_token_ids),
            observation_token_ids=observation_token_ids,
            tool_calls=normalized,
            reward=sum(result.execution.reward for result in results),
            terminated=any(result.terminated for result in results),
            truncated=any(result.truncated for result in results),
            info=info,
        )
        context.advance_step()
        return step

    def _execute(self, call: NormalizedToolCall) -> ToolExecution:
        try:
            executor = self._tools[call.name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {call.name}") from exc
        execute = getattr(executor, "call", executor)
        if not callable(execute):
            raise TypeError(f"tool executor for {call.name!r} is not callable")
        result = execute(**call.arguments)
        return self._normalize_execution(result)

    @staticmethod
    def _normalize_execution(result: tp.Any) -> ToolExecution:
        if isinstance(result, ToolExecution):
            return result
        if isinstance(result, str):
            return ToolExecution(result)
        if all(hasattr(result, field_name) for field_name in ("observation", "terminated", "truncated")):
            return ToolExecution(
                observation=str(result.observation),
                reward=float(getattr(result, "reward", 0.0)),
                terminated=bool(result.terminated),
                truncated=bool(result.truncated),
                info=dict(getattr(result, "info", {})),
            )
        return ToolExecution(str(result))
