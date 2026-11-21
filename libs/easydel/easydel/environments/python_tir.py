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

"""Stateful Tool-Integrated Reasoning (TIR) Python RL environment."""

from __future__ import annotations

import json
import re
import typing as tp

from easydel.trainers._shared.environment import ResetResult, StepResult, create_tool_call_parser
from easydel.trainers.rlvr_trainer.reward_verifiers import _extract_answer, _normalize_number

from ._base import SandboxEnvironment, register_environment
from .runtime import ContainerConfig, KiriRuntime, SandboxInfraError, SandboxPool
from .swe_repo import unscorable_step

_PY_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```|<code>\s*(.*?)\s*</code>", re.DOTALL)

_REPL_RUNNER = r"""
import importlib, os, pickle, sys, traceback, types

STATE_FILE = ".ed_tir_state.pkl"
ns = {"__name__": "__main__"}
if os.path.exists(STATE_FILE):
    with open(STATE_FILE, "rb") as _f:
        saved = pickle.load(_f)
    for mod_alias, mod_name in saved.get("modules", {}).items():
        try:
            ns[mod_alias] = importlib.import_module(mod_name)
        except Exception:
            pass
    ns.update(saved.get("values", {}))

code = sys.stdin.read()
try:
    compiled = compile(code, "<tir>", "exec")
    exec(compiled, ns)
except Exception:
    traceback.print_exc()
    sys.exit(1)

modules = {}
values = {}
for k, v in list(ns.items()):
    if k.startswith("__"):
        continue
    if isinstance(v, types.ModuleType):
        modules[k] = v.__name__
        continue
    try:
        pickle.dumps(v)
        values[k] = v
    except Exception:
        pass

with open(STATE_FILE, "wb") as _f:
    pickle.dump({"modules": modules, "values": values}, _f)
"""


def _extract_python_snippets(action: str, parser: tp.Callable[[str], list[tuple[str, str]]]) -> list[str]:
    snippets: list[str] = []
    for name, raw_args in parser(action):
        if name in {"python", "python_code"}:
            try:
                parsed = json.loads(raw_args)
                if isinstance(parsed, dict) and "code" in parsed:
                    snippets.append(str(parsed["code"]))
            except json.JSONDecodeError:
                continue
    if snippets:
        return snippets
    for g1, g2 in _PY_BLOCK_RE.findall(action):
        snippet = (g1 or g2).strip()
        if snippet:
            snippets.append(snippet)
    return snippets


@register_environment(["python_tir", "tir"])
class StatefulPythonEnvironment(SandboxEnvironment):
    """Multi-turn Python REPL environment for Tool-Integrated Reasoning (TIR).

    Persists variables and module imports across turns within an episode inside
    the sandbox session, returning stdout or tracebacks as intermediate
    observations until the policy emits a final answer without code blocks (or
    reaches ``max_steps``), then grades against ``answer`` or ``verifier``.
    """

    def __init__(
        self,
        prompt: str = "",
        *,
        answer: str | float | int | None = None,
        verifier: tp.Callable[[str], float | bool] | None = None,
        tool_caller: str | None = None,
        tokenizer: tp.Any | None = None,
        snippet_timeout: float = 10.0,
        max_steps: int = 6,
        runtime: KiriRuntime | SandboxPool | None = None,
        backend: str = "kiri",
        endpoint: str | None = None,
        container_config: ContainerConfig | None = None,
        affinity_key: str | int | None = None,
    ):
        super().__init__(
            runtime=runtime,
            backend=backend,
            endpoint=endpoint,
            container_config=container_config,
            max_steps=max_steps,
            affinity_key=affinity_key,
        )
        self._prompt = prompt
        self._answer = str(answer) if answer is not None else None
        self._verifier = verifier
        self._parser = create_tool_call_parser(tool_caller, tokenizer=tokenizer)
        self._snippet_timeout = float(snippet_timeout)

    _infra_failure_step = staticmethod(unscorable_step)

    def reset(self, seed: int | None = None) -> ResetResult:
        try:
            session = self._open_session(seed=seed)
            session.write_files({".ed_tir_runner.py": _REPL_RUNNER})
        except SandboxInfraError as exc:
            return ResetResult(
                observation=self._prompt,
                info={"env_error": True, "env_error_message": str(exc)},
            )
        return ResetResult(observation=self._prompt, info={"env_error": False})

    def _grade_final_answer(self, action: str) -> float:
        if self._verifier is not None:
            return float(self._verifier(action))
        if self._answer is None:
            return 0.0
        gold_num = _normalize_number(self._answer)
        if gold_num is not None:
            pred_num = _extract_answer(action)
            return 1.0 if pred_num == gold_num else 0.0
        boxed = re.findall(r"\\boxed\{([^}]+)\}", action)
        candidate = boxed[-1].strip() if boxed else action.strip()
        return 1.0 if candidate == self._answer.strip() else 0.0

    def grade_single_turn(self, action: str) -> StepResult:
        """Grade ``action`` as a final answer without executing any code block it contains.

        Used by single-turn reward adapters (:func:`make_environment_reward_fn`), where there is no next turn
        to show tool output to: the completion is always graded terminally.

        Args:
            action: Complete model completion.

        Returns:
            A terminated :class:`StepResult` carrying the answer reward.
        """
        self._step_count += 1
        try:
            reward = self._grade_final_answer(action)
        except Exception as exc:
            return self._infra_failure_step(f"answer verifier failed: {type(exc).__name__}: {exc}")
        return StepResult(
            observation="Correct." if reward > 0.0 else "Incorrect.",
            reward=reward,
            terminated=True,
            truncated=False,
            info={"step": self._step_count, "correct": reward > 0.0, "env_error": False},
        )

    def step(self, action: str) -> StepResult:
        self._step_count += 1
        snippets = _extract_python_snippets(action, self._parser)
        if snippets and self._step_count < self._max_steps:
            outputs: list[str] = []
            for code in snippets:
                res = self.session.run_python(
                    "import runpy; runpy.run_path('.ed_tir_runner.py')",
                    stdin=code,
                    timeout=self._snippet_timeout,
                )
                if res.infra_error:
                    return self._infra_failure_step(res.infra_error)
                outputs.append(res.combined_output(4096))
            return StepResult(
                observation="\n\n".join(outputs),
                reward=0.0,
                terminated=False,
                truncated=False,
                info={"step": self._step_count, "executed_blocks": len(snippets), "env_error": False},
            )

        try:
            reward = self._grade_final_answer(action)
        except Exception as exc:
            return self._infra_failure_step(f"answer verifier failed: {type(exc).__name__}: {exc}")
        truncated = bool(snippets) and self._step_count >= self._max_steps and reward == 0.0
        return StepResult(
            observation="Correct." if reward > 0.0 else "Incorrect.",
            reward=reward,
            terminated=not truncated,
            truncated=truncated,
            info={"step": self._step_count, "correct": reward > 0.0, "env_error": False},
        )


__all__ = ("StatefulPythonEnvironment",)
