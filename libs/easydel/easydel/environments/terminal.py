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

"""Terminal-Bench style stateful Linux CLI RL environment."""

from __future__ import annotations

import collections.abc
import json
import re
import typing as tp

from easydel.trainers._shared.environment import ResetResult, StepResult, create_tool_call_parser

from ._base import SandboxEnvironment, register_environment
from .runtime import ContainerConfig, KiriRuntime, SandboxInfraError, SandboxPool

_BASH_BLOCK_RE = re.compile(r"```(?:bash|sh)\s*\n(.*?)```|<bash>\s*(.*?)\s*</bash>", re.DOTALL)
_REWARD_MARKER_RE = re.compile(r"REWARD\s*=\s*(-?\d+(?:\.\d+)?)")


def _extract_shell_commands(action: str, parser: tp.Callable[[str], list[tuple[str, str]]]) -> list[str]:
    commands: list[str] = []
    for name, raw_args in parser(action):
        if name == "bash":
            try:
                parsed = json.loads(raw_args)
                if isinstance(parsed, dict) and "command" in parsed:
                    commands.append(str(parsed["command"]))
            except json.JSONDecodeError:
                continue
    if commands:
        return commands
    for g1, g2 in _BASH_BLOCK_RE.findall(action):
        cmd = (g1 or g2).strip()
        if cmd:
            commands.append(cmd)
    return commands


@register_environment(["terminal", "bash_task"])
class TerminalEnvironment(SandboxEnvironment):
    """Stateful multi-turn shell environment graded by a post-episode verification script.

    Seeds an isolated workspace via ``files`` and an optional ``setup_script``,
    executes multi-turn ``bash`` commands, and runs ``verify_script`` only when
    the agent submits (no command emitted) or hits ``max_steps``—preventing the
    agent from reading the grading logic during interaction.
    """

    def __init__(
        self,
        task_prompt: str = "",
        *,
        verify_script: str = "true",
        setup_script: str | None = None,
        files: collections.abc.Mapping[str, bytes | str] | None = None,
        tool_caller: str | None = None,
        tokenizer: tp.Any | None = None,
        verify_timeout: float = 30.0,
        max_steps: int = 10,
        runtime: KiriRuntime | SandboxPool | None = None,
        backend: str = "kiri",
        endpoint: str | None = None,
        container_config: ContainerConfig | None = None,
        affinity_key: str | int | None = None,
    ):
        cfg = container_config if container_config is not None else ContainerConfig(use_jail="required")
        super().__init__(
            runtime=runtime,
            backend=backend,
            endpoint=endpoint,
            container_config=cfg,
            max_steps=max_steps,
            affinity_key=affinity_key,
        )
        self._task_prompt = task_prompt
        self._verify_script = verify_script
        self._setup_script = setup_script
        self._initial_files = dict(files) if files is not None else None
        self._parser = create_tool_call_parser(tool_caller, tokenizer=tokenizer)
        self._verify_timeout = float(verify_timeout)

    def reset(self, seed: int | None = None) -> ResetResult:
        try:
            session = self._open_session(seed=seed)
            if self._initial_files:
                session.write_files(self._initial_files)
            if self._setup_script:
                res = session.run_bash(self._setup_script, timeout=self._verify_timeout)
                if res.infra_error:
                    return ResetResult(
                        observation=self._task_prompt,
                        info={"env_error": True, "env_error_message": res.infra_error},
                    )
                if not res.ok:
                    return ResetResult(
                        observation=self._task_prompt,
                        info={
                            "env_error": True,
                            "env_error_message": f"setup_script failed: {res.combined_output(1024)}",
                        },
                    )
        except SandboxInfraError as exc:
            return ResetResult(
                observation=self._task_prompt,
                info={"env_error": True, "env_error_message": str(exc)},
            )
        return ResetResult(observation=self._task_prompt, info={"env_error": False})

    def _run_verification(self, final_action: str) -> tuple[float, str, str | None]:
        res = self.session.exec(
            ["/bin/sh", "-c", self._verify_script],
            env={"AGENT_FINAL_RESPONSE": final_action},
            timeout=self._verify_timeout,
        )
        if res.infra_error:
            return 0.0, "", res.infra_error
        matches = _REWARD_MARKER_RE.findall(res.stdout)
        if matches:
            return max(0.0, min(1.0, float(matches[-1]))), res.combined_output(2048), None
        return (1.0 if res.ok else 0.0), res.combined_output(2048), None

    def step(self, action: str) -> StepResult:
        self._step_count += 1
        commands = _extract_shell_commands(action, self._parser)
        if commands and self._step_count < self._max_steps:
            outputs: list[str] = []
            for cmd in commands:
                res = self.session.run_bash(cmd)
                if res.infra_error:
                    return self._infra_failure_step(res.infra_error)
                outputs.append(res.combined_output(4096))
            return StepResult(
                observation="\n\n".join(outputs),
                reward=0.0,
                terminated=False,
                truncated=False,
                info={"step": self._step_count, "commands_run": len(commands), "env_error": False},
            )

        if commands and self._step_count == self._max_steps:
            for cmd in commands:
                res = self.session.run_bash(cmd)
                if res.infra_error:
                    return self._infra_failure_step(res.infra_error)

        try:
            reward, verify_out, infra_err = self._run_verification(action)
        except SandboxInfraError as exc:
            return self._infra_failure_step(exc)

        if infra_err is not None:
            return self._infra_failure_step(infra_err)

        truncated = bool(commands) and self._step_count >= self._max_steps and reward < 1.0
        return StepResult(
            observation=verify_out,
            reward=reward,
            terminated=not truncated,
            truncated=truncated,
            info={"step": self._step_count, "verified": reward > 0.0, "env_error": False},
        )


__all__ = ("TerminalEnvironment",)
