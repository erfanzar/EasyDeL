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
import math
import posixpath
import re
import secrets
import typing as tp

from easydel.trainers._shared.environment import ResetResult, StepResult, create_tool_call_parser

from ._base import SandboxEnvironment, register_environment
from .runtime import ContainerConfig, ContainerSandboxSession, KiriRuntime, SandboxInfraError, SandboxPool
from .swe_repo import kill_agent_processes, unscorable_step

_PLAIN_REWARD_RE = re.compile(r"^REWARD\s*=\s*-?\d+(?:\.\d+)?\s*$", re.MULTILINE)
_BASH_BLOCK_RE = re.compile(r"```(?:bash|sh)\s*\n(.*?)```|<bash>\s*(.*?)\s*</bash>", re.DOTALL)
_VERIFY_HOOK_BASENAMES = (
    "sitecustomize.py",
    "usercustomize.py",
    "conftest.py",
    "pytest.ini",
    ".pytest.ini",
    "pytest.toml",
    ".pytest.toml",
    "pytest.py",
    "pytest",
    "_pytest",
)
_VERIFY_HOOK_SUFFIXES = (".pth",)


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


def _parse_bounded_reward(raw_text: str) -> float:
    """Parse a finite float and clamp to ``[0.0, 1.0]``; non-finite or invalid strings return ``0.0``."""
    try:
        val = float(raw_text)
    except ValueError:
        return 0.0
    if not math.isfinite(val):
        return 0.0
    return max(0.0, min(1.0, val))


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
        cfg = (
            container_config
            if container_config is not None
            else ContainerConfig(use_jail="required", read_only_rootfs=True, workspace_tmpfs_mb=256)
        )
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

    _infra_failure_step = staticmethod(unscorable_step)

    def reset(self, seed: int | None = None) -> ResetResult:
        try:
            session = self._open_session(seed=seed)
            if self._initial_files:
                session.write_files(self._initial_files)
            if self._setup_script:
                res = session.run_bash(self._setup_script, timeout=self._verify_timeout)
                if res.infra_error:
                    self.close()
                    return ResetResult(
                        observation=self._task_prompt,
                        info={"env_error": True, "env_error_message": res.infra_error},
                    )
                if not res.ok:
                    self.close()
                    return ResetResult(
                        observation=self._task_prompt,
                        info={
                            "env_error": True,
                            "env_error_message": f"setup_script failed: {res.combined_output(1024)}",
                        },
                    )
        except SandboxInfraError as exc:
            self.close()
            return ResetResult(
                observation=self._task_prompt,
                info={"env_error": True, "env_error_message": str(exc)},
            )
        return ResetResult(observation=self._task_prompt, info={"env_error": False})

    def _uses_legacy_reward_marker(self) -> bool:
        """Whether the trusted ``verify_script`` itself emits the old plain ``REWARD=`` marker.

        Decided from the script text only, never from its output: verify output can echo agent-written files, so
        an output-based check would let the policy turn a failing episode into an unscorable (discarded) one.
        """
        script = self._verify_script
        return "REWARD=" in script and "ED_REWARD_NONCE" not in script and "ED_REWARD_FILE" not in script

    def _reward_file_path(self, nonce: str) -> str:
        """Per-verification score path, outside the agent workspace when the backend allows it."""
        if isinstance(self.session, ContainerSandboxSession):
            return f"/tmp/.ed_verify_{nonce}/reward"
        # Local sessions run each command in a fresh ``/tmp``; the workspace is the only persistent location.
        return posixpath.join(self.session.workdir, f".ed_verify_{nonce}", "reward")

    def _neutralize_workspace_hooks(self) -> str | None:
        """Remove agent-planted Python startup / test hooks from the workspace before running ``verify_script``."""
        session = self.session
        # Restore user permissions on `.` and non-symlink directories first so `chmod 555 .` / `chmod 000`
        # cannot prevent hook cleanup or creation of the local verification directory.
        chmod_res = session.exec(
            [
                "/bin/sh",
                "-c",
                "chmod u+rwx . 2>/dev/null; find . -maxdepth 6 ! -type l -exec chmod u+rwx -- {} + 2>/dev/null || true",
            ],
            timeout=15.0,
        )
        if chmod_res.infra_error:
            return chmod_res.infra_error
        name_tests: list[str] = []
        for name in [*_VERIFY_HOOK_BASENAMES, *(f"*{s}" for s in _VERIFY_HOOK_SUFFIXES)]:
            name_tests += [*(["-o"] if name_tests else []), "-name", name]
        clean_res = session.exec(
            ["find", ".", "-maxdepth", "6", "(", *name_tests, ")", "-prune", "-exec", "rm", "-rf", "--", "{}", "+"],
            timeout=15.0,
        )
        if clean_res.infra_error:
            return clean_res.infra_error
        # Restore any hook file that was explicitly part of the trusted initial task files.
        if self._initial_files:
            restore: dict[str, bytes | str] = {}
            for path, content in self._initial_files.items():
                base = posixpath.basename(path)
                if base in _VERIFY_HOOK_BASENAMES or base.endswith(_VERIFY_HOOK_SUFFIXES):
                    restore[path] = content
            if restore:
                try:
                    session.write_files(restore)
                except (OSError, ValueError):
                    pass
        return None

    def _run_verification(self, final_action: str) -> tuple[float, str, str | None]:
        """Run ``verify_script`` and return ``(reward, output, infra_error)``.

        The verifier reports its score through channels the agent cannot predict. A fresh nonce is drawn only
        after the agent has finished, and the script receives ``ED_REWARD_FILE`` (a nonce-named path that is
        created empty right before verification) and ``ED_REWARD_NONCE``. The reward is read, in order, from:

        1. ``ED_REWARD_FILE`` (a finite float in ``[0, 1]``; non-finite values score ``0.0``);
        2. a stdout line ``REWARD[<ED_REWARD_NONCE>]=<float>``;
        3. the script's exit status (``0`` -> ``1.0``).

        Plain ``REWARD=<float>`` markers are never trusted, because verify output can echo agent-written files.
        When neither nonce channel was used, the script text uses the old plain ``REWARD=`` convention and its
        stdout carries such a line, the result is unscorable instead of silently falling back to the exit status.

        Before verification every agent process still running in the session is killed; if that is not possible
        the episode is unscorable.
        """
        session = self.session
        kill_err = kill_agent_processes(session)
        if kill_err is not None:
            return 0.0, "", kill_err
        hook_err = self._neutralize_workspace_hooks()
        if hook_err is not None:
            return 0.0, "", hook_err
        nonce = secrets.token_hex(16)
        reward_path = self._reward_file_path(nonce)
        reward_dir = posixpath.dirname(reward_path)
        prep = session.exec(
            ["/bin/sh", "-c", 'mkdir -p "$1" && : > "$2"', "sh", reward_dir, reward_path],
            timeout=10.0,
        )
        if prep.infra_error:
            try:
                session.exec(["rm", "-rf", "--", reward_dir], timeout=10.0)
            except SandboxInfraError:
                pass
            return 0.0, "", prep.infra_error
        if not prep.ok:
            try:
                session.exec(["rm", "-rf", "--", reward_dir], timeout=10.0)
            except SandboxInfraError:
                pass
            # Non-zero exit without infra_error indicates agent workspace tampering (e.g. broken perms/dir):
            # score 0.0 instead of converting a failed rollout into discard=True.
            return 0.0, f"could not create verifier reward file: {prep.combined_output(1024)}", None

        file_text = ""
        try:
            verify_env = {
                "AGENT_FINAL_RESPONSE": final_action,
                "ED_REWARD_FILE": reward_path,
                "ED_REWARD_NONCE": nonce,
                "PYTHONNOUSERSITE": "1",
                "PYTHONSAFEPATH": "1",
                "PYTHONPATH": "",
                "PYTHONSTARTUP": "",
                "BASH_ENV": "",
                "ENV": "",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
            res = session.exec(
                ["/bin/sh", "-c", self._verify_script],
                env=verify_env,
                timeout=self._verify_timeout,
            )
            if res.infra_error:
                return 0.0, "", res.infra_error
            output = res.combined_output(2048).replace(nonce, "<nonce>")
            try:
                file_text = session.read_file(reward_path).strip()
            except Exception:
                file_text = ""
        finally:
            try:
                session.exec(["rm", "-rf", "--", reward_dir], timeout=10.0)
            except SandboxInfraError:
                pass

        if file_text:
            return _parse_bounded_reward(file_text), output, None
        marker = re.compile(rf"^REWARD\[{re.escape(nonce)}\]\s*=\s*(-?\d+(?:\.\d+)?)\s*$", re.MULTILINE)
        matches = marker.findall(res.stdout)
        if matches:
            return _parse_bounded_reward(matches[-1]), output, None
        if self._uses_legacy_reward_marker() and _PLAIN_REWARD_RE.search(res.stdout):
            # Old convention: grading it by exit status would silently change the score, so refuse to score.
            return (
                0.0,
                output,
                "verify_script printed a plain 'REWARD=<x>' line, which is no longer honoured (agent-written files "
                "can echo it). Report the score with 'echo \"REWARD[$ED_REWARD_NONCE]=<x>\"' or by writing it to "
                '"$ED_REWARD_FILE".',
            )
        return (1.0 if res.ok else 0.0), output, None

    def step(self, action: str) -> StepResult:
        self._step_count += 1
        commands = _extract_shell_commands(action, self._parser)
        if commands and self._step_count < self._max_steps:
            outputs: list[str] = []
            for cmd in commands:
                try:
                    res = self.session.run_bash(cmd)
                except SandboxInfraError as exc:
                    return self._infra_failure_step(exc)
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
                try:
                    res = self.session.run_bash(cmd)
                except SandboxInfraError as exc:
                    return self._infra_failure_step(exc)
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
