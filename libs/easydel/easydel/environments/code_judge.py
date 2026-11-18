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

"""Single-turn and multi-turn code execution RL environment."""

from __future__ import annotations

import collections.abc
import re
import typing as tp
import uuid

from easydel.trainers._shared.environment import ResetResult, StepResult

from ._base import SandboxEnvironment, register_environment
from .runtime import ContainerConfig, ExecResult, KiriRuntime, SandboxInfraError, SandboxPool

_FENCED_CODE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\s*\n(.*?)```", re.DOTALL)
_TAGGED_CODE_RE = re.compile(r"<code>\s*(.*?)\s*</code>", re.DOTALL)

_LANGUAGE_SPECS: dict[str, tuple[str, list[str] | None, list[str]]] = {
    "python": ("solution.py", None, ["python3", "-I", "solution.py"]),
    "bash": ("solution.sh", None, ["/bin/sh", "solution.sh"]),
    "javascript": ("solution.js", None, ["node", "solution.js"]),
    "cpp": ("solution.cpp", ["g++", "-O2", "-std=c++20", "solution.cpp", "-o", "solution.bin"], ["./solution.bin"]),
    "rust": ("solution.rs", ["rustc", "-O", "solution.rs", "-o", "solution.bin"], ["./solution.bin"]),
    "go": ("solution.go", ["go", "build", "-o", "solution.bin", "solution.go"], ["./solution.bin"]),
}


def extract_code_block(text: str) -> str | None:
    """Extract the last fenced or ``<code>`` block from ``text``."""
    fenced = _FENCED_CODE_RE.findall(text)
    if fenced:
        return fenced[-1].strip()
    tagged = _TAGGED_CODE_RE.findall(text)
    if tagged:
        return tagged[-1].strip()
    return None


@register_environment(["code_judge", "code"])
class CodeJudgeEnvironment(SandboxEnvironment):
    """Competitive-coding and unit-test RL environment.

    Supports both stdin/stdout test cases (``[{"input": "...", "output": "..."}]``)
    and assertion/unit-test suites (``test_code``) across Python, Bash,
    JavaScript, C++, Rust, and Go. When ``max_steps > 1``, compiler errors and
    failed test diagnostics are returned as observations so the policy can
    iteratively debug its solution.
    """

    def __init__(
        self,
        prompt: str = "",
        *,
        stdin_tests: collections.abc.Sequence[collections.abc.Mapping[str, str]] | None = None,
        test_code: str | None = None,
        language: str = "python",
        reward_mode: tp.Literal["binary", "pass_rate"] = "binary",
        compile_timeout: float = 20.0,
        test_timeout: float = 5.0,
        max_steps: int = 1,
        runtime: KiriRuntime | SandboxPool | None = None,
        backend: str = "kiri",
        endpoint: str | None = None,
        container_config: ContainerConfig | None = None,
        affinity_key: str | int | None = None,
    ):
        lang = language.lower()
        if lang not in _LANGUAGE_SPECS:
            raise ValueError(f"Unsupported language {language!r}; choose from {sorted(_LANGUAGE_SPECS)}.")
        if reward_mode not in {"binary", "pass_rate"}:
            raise ValueError(f"reward_mode must be 'binary' or 'pass_rate'; got {reward_mode!r}.")
        if test_code and lang != "python":
            raise ValueError(f"`test_code` is only supported for language='python'; got {language!r}.")
        super().__init__(
            runtime=runtime,
            backend=backend,
            endpoint=endpoint,
            container_config=container_config,
            max_steps=max_steps,
            affinity_key=affinity_key,
        )
        self._prompt = prompt
        self._stdin_tests = [dict(case) for case in (stdin_tests or ())]
        self._test_code = test_code
        self._language = lang
        self._reward_mode = reward_mode
        self._compile_timeout = float(compile_timeout)
        self._test_timeout = float(test_timeout)

    def reset(self, seed: int | None = None) -> ResetResult:
        try:
            session = self._open_session(seed=seed)
        except SandboxInfraError as exc:
            return ResetResult(
                observation=self._prompt,
                info={"env_error": True, "env_error_message": str(exc)},
            )
        return ResetResult(
            observation=self._prompt,
            info={
                "language": self._language,
                "unjailed": not session.jailed,
                "in_process_test_runner": bool(self._test_code and self._language == "python"),
            },
        )

    def _evaluate_code(self, code: str) -> tuple[int, int, str, str | None]:
        src_name, compile_cmd, run_cmd = _LANGUAGE_SPECS[self._language]
        session = self.session

        if self._test_code and self._language == "python":
            # Write ONLY the untrusted solution to disk (never the sentinel or test_code).
            # Feed the runner over stdin (`-I -`) so open(__file__) and /proc/self/cmdline
            # cannot read the sentinel. Note: in-process execution shares module co_consts
            # with caller frames (flagged via `in_process_test_runner`).
            import sys as _sys

            from .runtime import LocalSandboxSession

            sentinel = f"__ED_UNIT_TESTS_PASSED_{uuid.uuid4().hex}__"
            session.write_files({src_name: code})
            runner_src = (
                "import os as _ed_os\n"
                "_ed_exec = exec\n"
                "_ed_compile = compile\n"
                "_ed_print = print\n"
                f"_ed_ns = {{'__name__': 'solution', '__file__': _ed_os.path.abspath({src_name!r})}}\n"
                f"with open({src_name!r}, 'r', encoding='utf-8') as _ed_f:\n"
                "    _ed_exec(_ed_compile(_ed_f.read(), _ed_ns['__file__'], 'exec'), _ed_ns)\n"
                f"_ed_exec(_ed_compile({self._test_code!r}, '<tests>', 'exec'), _ed_ns)\n"
                f"_ed_print({sentinel!r}, flush=True)\n"
            )
            py_bin = _sys.executable if isinstance(session, LocalSandboxSession) else "python3"
            res = session.exec([py_bin, "-I", "-"], stdin=runner_src, timeout=self._test_timeout)
            if res.infra_error:
                return 0, 1, "", res.infra_error
            if res.ok and sentinel in res.stdout:
                return 1, 1, "All unit tests passed.", None
            clean_out = res.combined_output(max_chars=4096).replace(sentinel, "").strip()
            if res.ok:
                clean_out = f"Solution exited before unit tests completed.\n{clean_out}".strip()
            return 0, 1, clean_out, None

        session.write_files({src_name: code})
        if compile_cmd is not None:
            comp: ExecResult = session.exec(compile_cmd, timeout=self._compile_timeout)
            if comp.infra_error:
                return 0, max(1, len(self._stdin_tests)), "", comp.infra_error
            if not comp.ok:
                return 0, max(1, len(self._stdin_tests)), f"Compilation failed:\n{comp.combined_output(4096)}", None

        exec_argv = list(run_cmd)
        if self._language == "python":
            # Use local interpreter path when running in LocalSandboxSession.
            res_probe = session.run_python("print('ok')", timeout=2.0)
            if res_probe.infra_error:
                return 0, max(1, len(self._stdin_tests)), "", res_probe.infra_error

        if not self._stdin_tests:
            run_res = (
                session.run_python(code, timeout=self._test_timeout)
                if self._language == "python"
                else session.exec(exec_argv, timeout=self._test_timeout)
            )
            if run_res.infra_error:
                return 0, 1, "", run_res.infra_error
            if run_res.ok:
                return 1, 1, run_res.stdout.strip() or "Execution succeeded.", None
            return 0, 1, run_res.combined_output(4096), None

        passed = 0
        first_failure = ""
        for idx, case in enumerate(self._stdin_tests):
            inp = str(case.get("input", ""))
            expected = str(case.get("output", "")).strip()
            run_res = (
                session.run_python(code, stdin=inp, timeout=self._test_timeout)
                if self._language == "python"
                else session.exec(exec_argv, stdin=inp, timeout=self._test_timeout)
            )
            if run_res.infra_error:
                return passed, len(self._stdin_tests), "", run_res.infra_error
            actual = run_res.stdout.strip()
            if run_res.ok and self._outputs_match(actual, expected):
                passed += 1
            elif not first_failure:
                detail = run_res.combined_output(2048) if not run_res.ok else f"got {actual!r}, expected {expected!r}"
                first_failure = f"Test {idx + 1}/{len(self._stdin_tests)} failed: {detail}"
        summary = "All test cases passed." if passed == len(self._stdin_tests) else first_failure
        return passed, len(self._stdin_tests), summary, None

    @staticmethod
    def _outputs_match(actual: str, expected: str) -> bool:
        act_lines = [line.rstrip() for line in actual.strip().splitlines()]
        exp_lines = [line.rstrip() for line in expected.strip().splitlines()]
        return act_lines == exp_lines

    def step(self, action: str) -> StepResult:
        self._step_count += 1
        code = extract_code_block(action)
        if not code:
            at_limit = self._step_count >= self._max_steps
            return StepResult(
                observation="No fenced code block found. Wrap your solution in ```...```.",
                reward=0.0,
                terminated=at_limit,
                truncated=at_limit,
                info={"passed": 0, "total": max(1, len(self._stdin_tests)), "env_error": False},
            )

        try:
            passed, total, feedback, infra_err = self._evaluate_code(code)
        except SandboxInfraError as exc:
            return self._infra_failure_step(exc)

        if infra_err is not None:
            return self._infra_failure_step(infra_err)

        all_passed = passed == total and total > 0
        at_limit = self._step_count >= self._max_steps
        done = all_passed or at_limit
        if self._reward_mode == "binary":
            reward = 1.0 if all_passed else 0.0
        else:
            reward = (float(passed) / float(total)) if (done and total > 0) else 0.0

        return StepResult(
            observation=feedback,
            reward=reward,
            terminated=all_passed or (at_limit and self._max_steps == 1),
            truncated=at_limit and not all_passed and self._max_steps > 1,
            info={
                "passed": passed,
                "total": total,
                "all_passed": all_passed,
                "env_error": False,
                "in_process_test_runner": bool(self._test_code and self._language == "python"),
            },
        )


__all__ = ("CodeJudgeEnvironment", "extract_code_block")
