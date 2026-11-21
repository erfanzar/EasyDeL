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

"""Base sandbox-backed RL environments, session-bound tools, and registry helpers."""

from __future__ import annotations

import asyncio
import collections.abc
import concurrent.futures
import typing as tp

from easydel.trainers._shared.environment import (
    AgenticEnvironment,
    ResetResult,
    StepResult,
    ToolEnvWrapper,
    create_tool_call_parser,
)
from easydel.trainers._shared.tools import Tool
from easydel.utils import Registry

from .runtime import ContainerConfig, KiriRuntime, SandboxInfraError, SandboxPool, SandboxSession


def register_environment(name: str | collections.abc.Sequence[str]):
    """Register an :class:`AgenticEnvironment` subclass in EasyDeL's ``Registry``."""
    return Registry.register("environment", name)


def list_environments() -> list[str]:
    """Return all registered environment names."""
    return Registry.list_implementations("environment")


def build_environment(
    spec: str | collections.abc.Mapping[str, tp.Any] | AgenticEnvironment,
    **overrides: tp.Any,
) -> AgenticEnvironment:
    """Instantiate a registered :class:`AgenticEnvironment` from a name or dict spec.

    Args:
        spec: Registered environment name (e.g. ``"code_judge"``), mapping with
            ``"name"`` or ``"type"`` plus constructor kwargs, or an already
            instantiated :class:`AgenticEnvironment`.
        **overrides: Additional keyword arguments merged into constructor kwargs.

    Returns:
        An :class:`AgenticEnvironment` instance.
    """
    if isinstance(spec, AgenticEnvironment):
        if overrides:
            raise ValueError("Cannot pass constructor overrides when `spec` is already an AgenticEnvironment.")
        return spec
    if isinstance(spec, str):
        env_cls = Registry.get_or_raise("environment", spec)
        return env_cls(**overrides)
    if isinstance(spec, collections.abc.Mapping):
        params = dict(spec)
        env_name = params.pop("name", None) or params.pop("type", None)
        if not isinstance(env_name, str) or not env_name:
            raise ValueError("Environment mapping spec must include a non-empty 'name' or 'type' string.")
        params.update(overrides)
        env_cls = Registry.get_or_raise("environment", env_name)
        return env_cls(**params)
    raise TypeError(f"Unsupported environment spec type: {type(spec).__name__}")


def make_environment_factory(
    spec: str | collections.abc.Mapping[str, tp.Any] | tp.Callable[..., AgenticEnvironment],
    **default_kwargs: tp.Any,
) -> tp.Callable[..., AgenticEnvironment]:
    """Build a reusable ``environment_factory`` callable for synchronous and async trainers."""
    if callable(spec) and not isinstance(spec, type):
        if not default_kwargs:
            return spec
        return lambda **kw: spec(**{**default_kwargs, **kw})
    if isinstance(spec, type) and issubclass(spec, AgenticEnvironment):
        return lambda **kw: spec(**{**default_kwargs, **kw})
    frozen_spec = dict(spec) if isinstance(spec, collections.abc.Mapping) else spec
    return lambda **kw: build_environment(frozen_spec, **{**default_kwargs, **kw})


class SandboxBashTool(Tool):
    """Run shell commands inside an episode's stateful :class:`SandboxSession`."""

    name = "bash"

    def __init__(self, session_getter: tp.Callable[[], SandboxSession], *, timeout: float | None = None):
        self._get_session = session_getter
        self._timeout = timeout

    def call(self, command: str) -> str:
        """Execute a shell command in the episode workspace and return stdout/stderr.

        command: Shell command to execute inside the workspace.
        """
        res = self._get_session().run_bash(command, timeout=self._timeout)
        return res.combined_output()


class SandboxPythonTool(Tool):
    """Run isolated Python snippets inside an episode's :class:`SandboxSession`."""

    name = "python"

    def __init__(self, session_getter: tp.Callable[[], SandboxSession], *, timeout: float | None = None):
        self._get_session = session_getter
        self._timeout = timeout

    def call(self, code: str) -> str:
        """Execute Python code in the episode workspace and return printed output.

        code: Python source code to execute.
        """
        res = self._get_session().run_python(code, timeout=self._timeout)
        return res.combined_output()


class SandboxReadFileTool(Tool):
    """Read a UTF-8 file from an episode's :class:`SandboxSession` workspace."""

    name = "read_file"

    def __init__(self, session_getter: tp.Callable[[], SandboxSession], *, max_chars: int = 16384):
        self._get_session = session_getter
        self._max_chars = max_chars

    def call(self, path: str) -> str:
        """Read a UTF-8 text file from the workspace.

        path: Workspace-relative file path to read.
        """
        try:
            text = self._get_session().read_file(path, max_chars=self._max_chars)
        except Exception as exc:
            return f"Error reading '{path}': {exc}"
        if len(text) > self._max_chars:
            return text[: self._max_chars] + "\n... (file truncated)"
        return text


class SandboxWriteFileTool(Tool):
    """Write a complete UTF-8 file inside an episode's :class:`SandboxSession`."""

    name = "write_file"

    def __init__(self, session_getter: tp.Callable[[], SandboxSession]):
        self._get_session = session_getter

    def call(self, path: str, content: str) -> str:
        """Create or overwrite a UTF-8 text file in the workspace.

        path: Workspace-relative destination path.
        content: Complete file contents to write.
        """
        try:
            self._get_session().write_files({path: content})
            return f"Wrote {len(content)} characters to {path}."
        except Exception as exc:
            return f"Error writing '{path}': {exc}"


class SandboxEditFileTool(Tool):
    """Replace one exact substring in a file inside an episode's :class:`SandboxSession`."""

    name = "edit_file"
    max_file_chars = 4 * 1024 * 1024
    """Largest file (in characters) the tool loads; bigger files are refused instead of read whole."""

    def __init__(self, session_getter: tp.Callable[[], SandboxSession]):
        self._get_session = session_getter

    def call(self, path: str, old_text: str, new_text: str) -> str:
        """Replace a unique span ``old_text`` with ``new_text`` in ``path``.

        path: Workspace-relative file path to edit.
        old_text: Exact existing text span (must appear exactly once).
        new_text: Replacement text.
        """
        session = self._get_session()
        try:
            # read_file returns at most max_chars + 1 characters, so a longer result means the file is too large.
            current = session.read_file(path, max_chars=self.max_file_chars)
        except Exception as exc:
            return f"Error reading '{path}': {exc}"
        if len(current) > self.max_file_chars:
            return f"Error: '{path}' is larger than {self.max_file_chars} characters; edit_file refuses to load it."
        count = current.count(old_text)
        if count == 0:
            return f"Error: `old_text` not found in '{path}'."
        if count > 1:
            return f"Error: `old_text` matched {count} times in '{path}'; include more surrounding context."
        updated = current.replace(old_text, new_text, 1)
        try:
            session.write_files({path: updated})
            return f"Updated {path}."
        except Exception as exc:
            return f"Error writing '{path}': {exc}"


class SandboxEnvironment(AgenticEnvironment):
    """Base multi-turn RL environment backed by a stateful :class:`SandboxSession`.

    Supports both synchronous trainer loops (including parallel ``reset_batch`` /
    ``step_batch`` execution) and non-blocking ``reset_async`` / ``step_async``
    coroutines for asynchronous RL collectors.
    """

    def __init__(
        self,
        *,
        runtime: KiriRuntime | SandboxPool | None = None,
        backend: str = "kiri",
        endpoint: str | None = None,
        container_config: ContainerConfig | None = None,
        max_steps: int = 8,
        affinity_key: str | int | None = None,
    ):
        self._container_config = container_config or ContainerConfig()
        if isinstance(runtime, SandboxPool):
            self._pool: SandboxPool | None = runtime
            self._runtime: KiriRuntime | None = None
        elif isinstance(runtime, KiriRuntime):
            self._pool = None
            self._runtime = runtime
        else:
            self._pool = None
            self._runtime = KiriRuntime(backend=backend, endpoint=endpoint, config=self._container_config)
        self._max_steps = int(max_steps)
        self._affinity_key = affinity_key
        self._session: SandboxSession | None = None
        self._step_count = 0

    @property
    def max_steps(self) -> int | None:
        return self._max_steps

    @property
    def session(self) -> SandboxSession:
        if self._session is None:
            raise RuntimeError("Environment session is not initialized; call reset() first.")
        return self._session

    def _open_session(self, seed: int | None = None) -> SandboxSession:
        self.close()
        key = self._affinity_key if self._affinity_key is not None else seed
        if self._pool is not None:
            self._session = self._pool.acquire(affinity_key=key, config=self._container_config)
        else:
            assert self._runtime is not None
            self._session = self._runtime.create_session(config=self._container_config)
        self._step_count = 0
        return self._session

    def build_workspace_tools(
        self,
        *,
        enable_bash: bool = True,
        enable_python: bool = False,
        enable_files: bool = True,
    ) -> list[Tool]:
        """Return session-bound tools for :class:`ToolEnvWrapper` or custom dispatch."""
        getter = lambda: self.session  # noqa: E731
        built: list[Tool] = []
        if enable_bash:
            built.append(SandboxBashTool(getter, timeout=self._container_config.timeout))
        if enable_python:
            built.append(SandboxPythonTool(getter, timeout=self._container_config.timeout))
        if enable_files:
            built.extend(
                [
                    SandboxReadFileTool(getter, max_chars=self._container_config.max_output_chars),
                    SandboxWriteFileTool(getter),
                    SandboxEditFileTool(getter),
                ]
            )
        return built

    def as_tool_env(
        self,
        *,
        tool_caller: str | None = None,
        tokenizer: tp.Any | None = None,
        max_tool_calls_per_step: int = 5,
        extra_tools: collections.abc.Sequence[Tool] = (),
        enable_bash: bool = True,
        enable_python: bool = False,
        enable_files: bool = True,
    ) -> ToolEnvWrapper:
        """Wrap this environment with :class:`ToolEnvWrapper` and bound sandbox tools."""
        tools = [
            *self.build_workspace_tools(enable_bash=enable_bash, enable_python=enable_python, enable_files=enable_files),
            *extra_tools,
        ]
        parser = create_tool_call_parser(tool_caller, tokenizer=tokenizer)
        return ToolEnvWrapper(
            env=self,
            tools=tools,
            tool_call_parser=parser,
            max_tool_calls_per_step=max_tool_calls_per_step,
        )

    async def reset_async(self, seed: int | None = None) -> ResetResult:
        """Non-blocking ``reset`` for asynchronous rollout loops."""
        return await asyncio.to_thread(self.reset, seed)

    async def step_async(self, action: str) -> StepResult:
        """Non-blocking ``step`` for asynchronous rollout loops."""
        return await asyncio.to_thread(self.step, action)

    def reset_batch(self, environments: list[AgenticEnvironment], seeds: list[int | None]) -> list[ResetResult]:
        """Reset a batch of environments concurrently using a thread pool."""
        if len(environments) <= 1:
            return [env.reset(seed=s) for env, s in zip(environments, seeds, strict=True)]
        workers = min(32, len(environments))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="edenv-reset") as pool:
            futs = [pool.submit(env.reset, seed=s) for env, s in zip(environments, seeds, strict=True)]
            return [f.result() for f in futs]

    @staticmethod
    def step_batch(
        environments: collections.abc.Sequence[AgenticEnvironment],
        actions: collections.abc.Sequence[str],
        *,
        max_workers: int = 32,
    ) -> list[StepResult]:
        """Step multiple environments concurrently to overlap sandbox/Kiri latency."""
        if len(environments) != len(actions):
            raise ValueError(f"environments ({len(environments)}) and actions ({len(actions)}) length mismatch.")
        if not environments:
            return []
        if len(environments) == 1:
            return [environments[0].step(actions[0])]
        workers = min(max(1, max_workers), len(environments))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="edenv-step") as pool:
            futs = [pool.submit(env.step, act) for env, act in zip(environments, actions, strict=True)]
            return [f.result() for f in futs]

    @staticmethod
    def _infra_failure_step(error: Exception | str) -> StepResult:
        return StepResult(
            observation=f"Environment infrastructure error: {error}",
            reward=0.0,
            terminated=True,
            truncated=True,
            info={"env_error": True, "env_error_message": str(error)},
        )

    def close(self) -> None:
        if self._session is not None:
            try:
                self._session.close()
            finally:
                self._session = None


__all__ = (
    "SandboxBashTool",
    "SandboxEditFileTool",
    "SandboxEnvironment",
    "SandboxInfraError",
    "SandboxPythonTool",
    "SandboxReadFileTool",
    "SandboxWriteFileTool",
    "build_environment",
    "list_environments",
    "make_environment_factory",
    "register_environment",
)
