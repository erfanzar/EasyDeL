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

"""Fast deterministic ``<state, tools, task, verifier>`` RL environment."""

from __future__ import annotations

import asyncio
import collections.abc
import copy
import inspect
import json
import math
import typing as tp

from easydel.trainers._shared.environment import AgenticEnvironment, ResetResult, StepResult, create_tool_call_parser
from easydel.trainers._shared.tools import function_to_json

from ._base import register_environment
from .swe_repo import unscorable_step


@register_environment(["synthetic_tool", "stateful_tool"])
class SyntheticToolEnvironment(AgenticEnvironment):
    """Deterministic multi-turn tool-use environment over an isolated state dict.

    Follows the DeepSeek-V3.2 synthetic agentic task pattern
    ``<state, tools, task, verifier>``: each ``reset()`` deep-copies
    ``initial_state``, tool callables inspect/mutate ``state`` across turns,
    and ``verifier(state, final_response)`` computes the terminal reward once
    the agent calls ``submit_result`` or responds without further tool calls.
    """

    def __init__(
        self,
        prompt: str = "",
        *,
        initial_state: collections.abc.Mapping[str, tp.Any] | None = None,
        tools: collections.abc.Mapping[str, tp.Callable[..., tp.Any]]
        | collections.abc.Sequence[tp.Callable[..., tp.Any]] = (),
        verifier: tp.Callable[[dict[str, tp.Any], str], float | bool] | None = None,
        tool_caller: str | None = None,
        tokenizer: tp.Any | None = None,
        max_tool_calls_per_step: int = 5,
        format_penalty: float = 0.0,
        halt_on_invalid_tool: bool = False,
        max_steps: int = 10,
    ):
        self._prompt = prompt
        self._initial_state = copy.deepcopy(dict(initial_state)) if initial_state is not None else {}
        if isinstance(tools, collections.abc.Mapping):
            self._tools = dict(tools)
        else:
            self._tools = {getattr(fn, "__name__", f"tool_{i}"): fn for i, fn in enumerate(tools)}
        self._verifier = verifier
        self._parser = create_tool_call_parser(tool_caller, tokenizer=tokenizer)
        self._max_tool_calls_per_step = int(max_tool_calls_per_step)
        fp = float(format_penalty)
        if not math.isfinite(fp):
            raise ValueError(f"`format_penalty` must be finite; got {format_penalty}.")
        self._format_penalty = fp
        self._halt_on_invalid_tool = bool(halt_on_invalid_tool)
        self._max_steps = int(max_steps)
        self._state: dict[str, tp.Any] = {}
        self._step_count = 0

    @property
    def max_steps(self) -> int | None:
        return self._max_steps

    @property
    def state(self) -> dict[str, tp.Any]:
        return self._state

    @property
    def tool_schemas(self) -> list[dict[str, tp.Any]]:
        """OpenAI-compatible JSON schemas for all registered environment tools."""
        schemas: list[dict[str, tp.Any]] = []
        for name, fn in self._tools.items():
            raw = function_to_json(fn)
            fn_spec = dict(raw["function"])
            fn_spec["name"] = name
            params = dict(fn_spec.get("parameters", {}))
            props = dict(params.get("properties", {}))
            props.pop("state", None)
            req = [r for r in params.get("required", []) if r != "state"]
            params["properties"] = props
            params["required"] = req
            fn_spec["parameters"] = params
            schemas.append({"type": "function", "function": fn_spec})
        return schemas

    def reset(self, seed: int | None = None) -> ResetResult:
        self._state = copy.deepcopy(self._initial_state)
        self._step_count = 0
        return ResetResult(
            observation=self._prompt,
            info={"tool_names": list(self._tools.keys()), "env_error": False},
        )

    async def reset_async(self, seed: int | None = None) -> ResetResult:
        return await asyncio.to_thread(self.reset, seed)

    async def step_async(self, action: str) -> StepResult:
        return await asyncio.to_thread(self.step, action)

    def _invoke_tool(self, fn: tp.Callable[..., tp.Any], kwargs: dict[str, tp.Any]) -> tp.Any:
        clean_kwargs = {k: v for k, v in kwargs.items() if k != "state"}
        sig = inspect.signature(fn)
        if "state" in sig.parameters:
            return fn(state=self._state, **clean_kwargs)
        return fn(**clean_kwargs)

    def _score(self, final_answer: str) -> float:
        if self._verifier is None:
            return 0.0
        val = float(self._verifier(self._state, final_answer))
        if not math.isfinite(val):
            raise ValueError(f"verifier returned non-finite reward: {val!r}")
        return val

    def _safe_score(self, final_answer: str) -> tuple[float, StepResult | None]:
        try:
            return self._score(final_answer), None
        except Exception as exc:
            return 0.0, unscorable_step(f"verifier failed: {type(exc).__name__}: {exc}")

    def step(self, action: str) -> StepResult:
        self._step_count += 1
        calls = self._parser(action)
        if not calls:
            if "<tool_call>" in action and self._halt_on_invalid_tool:
                return StepResult(
                    observation="Malformed tool call.",
                    reward=-abs(self._format_penalty),
                    terminated=True,
                    truncated=False,
                    info={"step": self._step_count, "format_error": True, "env_error": False},
                )
            reward, err_step = self._safe_score(action)
            if err_step is not None:
                return err_step
            return StepResult(
                observation="Submitted.",
                reward=reward,
                terminated=True,
                truncated=False,
                info={"step": self._step_count, "submitted": True, "env_error": False},
            )

        observations: list[str] = []
        submitted_answer: str | None = None

        for name, raw_args in calls[: self._max_tool_calls_per_step]:
            try:
                kwargs = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                if not isinstance(kwargs, dict):
                    raise ValueError("Tool arguments must decode to a JSON object.")
            except Exception as exc:
                if self._halt_on_invalid_tool:
                    return StepResult(
                        observation=f"Invalid tool arguments for '{name}': {exc}",
                        reward=-abs(self._format_penalty),
                        terminated=True,
                        truncated=False,
                        info={"step": self._step_count, "format_error": True, "env_error": False},
                    )
                observations.append(f"[{name}]: Error parsing arguments: {exc}")
                continue

            if name == "submit_result" and name not in self._tools:
                submitted_answer = str(kwargs.get("answer_text", kwargs.get("answer", json.dumps(kwargs))))
                break

            fn = self._tools.get(name)
            if fn is None:
                if self._halt_on_invalid_tool:
                    return StepResult(
                        observation=f"Unknown tool '{name}'.",
                        reward=-abs(self._format_penalty),
                        terminated=True,
                        truncated=False,
                        info={"step": self._step_count, "format_error": True, "env_error": False},
                    )
                observations.append(f"[{name}]: Unknown tool '{name}'.")
                continue

            try:
                result = self._invoke_tool(fn, kwargs)
                rendered = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
                observations.append(f"[{name}]: {rendered}")
            except Exception as exc:
                observations.append(f"[{name}]: Error: {exc}")

        if submitted_answer is not None:
            reward, err_step = self._safe_score(submitted_answer)
            if err_step is not None:
                return err_step
            return StepResult(
                observation="Result submitted.",
                reward=reward,
                terminated=True,
                truncated=False,
                info={"step": self._step_count, "submitted": True, "env_error": False},
            )

        at_limit = self._step_count >= self._max_steps
        if at_limit:
            reward, err_step = self._safe_score(action)
            if err_step is not None:
                return err_step
        else:
            reward = 0.0
        return StepResult(
            observation="\n\n".join(observations),
            reward=reward,
            terminated=False,
            truncated=at_limit,
            info={"step": self._step_count, "tool_calls": len(calls), "env_error": False},
        )


__all__ = ("SyntheticToolEnvironment",)
