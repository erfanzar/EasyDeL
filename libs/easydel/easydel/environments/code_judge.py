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
import json
import re
import secrets
import typing as tp

from easydel.trainers._shared.environment import ResetResult, StepResult

from ._base import SandboxEnvironment, register_environment
from .runtime import ContainerConfig, ExecResult, KiriRuntime, SandboxInfraError, SandboxPool
from .swe_repo import unscorable_step

# ---------------------------------------------------------------------------------------------------------------------
# Isolated unit-test grading (``test_code`` mode).
#
# The grading decision never shares a process with the untrusted solution:
#
# * The *runner* (trusted) is fed to ``python -I -`` over stdin. Before doing anything else it marks itself
#   non-dumpable (so same-uid processes cannot read its memory or ``/proc/<pid>/fd``), points fd 0 at
#   ``/dev/null`` and closes every other inherited descriptor.
# * The runner spawns the *solution server* as a child process with fresh pipes. The child loads
#   ``solution.py`` and answers call requests over its own stdin/stdout pipes; every value crossing the
#   boundary goes through a closed, exact-type codec (``None``/bool/int/float/str/bytes/list/tuple/set/
#   frozenset/dict), so the child can only ever return plain data -- never objects with custom ``__eq__``.
# * The test code executes inside the runner, against proxies of the solution's top-level names. Assertions,
#   builtins and control flow all live in the runner's interpreter, which the solution cannot touch.
# * The env generates a fresh random nonce per evaluation and embeds it only in the runner source (stdin,
#   consumed before the child exists). The runner prints exactly one ``<nonce> <json>`` result line; the env
#   ignores every other byte of output. The child has neither the runner's stdout nor the nonce, so it cannot
#   forge a result: exiting early, printing markers, rebinding builtins or walking frames only affects the child.
# * If the child dies, times out or violates the protocol mid-test, the runner records a fatal error that the
#   tests cannot catch away, and the evaluation fails.
#
# Limitation: tests can only exchange the codec's plain data types with the solution. A solution function that
# returns another object (class instance, generator, ...) makes the call fail, and in-place mutation of arguments
# is not visible to the tests.
# ---------------------------------------------------------------------------------------------------------------------

_VALUE_CODEC_SRC = r"""
_ED_MAX_DEPTH = 64


def _ed_enc(value, depth=0):
    if depth > _ED_MAX_DEPTH:
        raise TypeError("value nested too deeply")
    kind = type(value)
    if value is None:
        return ["n"]
    if kind is bool:
        return ["b", value]
    if kind is int:
        return ["i", str(value)]
    if kind is float:
        return ["f", repr(value)]
    if kind is str:
        return ["s", value]
    if kind is bytes:
        return ["y", value.hex()]
    if kind is list:
        return ["l", [_ed_enc(item, depth + 1) for item in value]]
    if kind is tuple:
        return ["t", [_ed_enc(item, depth + 1) for item in value]]
    if kind is set:
        return ["S", [_ed_enc(item, depth + 1) for item in value]]
    if kind is frozenset:
        return ["F", [_ed_enc(item, depth + 1) for item in value]]
    if kind is dict:
        return ["d", [[_ed_enc(k, depth + 1), _ed_enc(v, depth + 1)] for k, v in value.items()]]
    raise TypeError("unsupported value of type " + kind.__name__)


def _ed_dec(raw, depth=0):
    if depth > _ED_MAX_DEPTH or type(raw) is not list or not raw or type(raw[0]) is not str:
        raise ValueError("malformed value")
    tag = raw[0]
    if tag == "n" and len(raw) == 1:
        return None
    if len(raw) != 2:
        raise ValueError("malformed value")
    body = raw[1]
    if tag == "b" and type(body) is bool:
        return body
    if tag == "i" and type(body) is str:
        return int(body)
    if tag == "f" and type(body) is str:
        return float(body)
    if tag == "s" and type(body) is str:
        return body
    if tag == "y" and type(body) is str:
        return bytes.fromhex(body)
    if type(body) is not list:
        raise ValueError("malformed value")
    if tag == "l":
        return [_ed_dec(item, depth + 1) for item in body]
    if tag == "t":
        return tuple(_ed_dec(item, depth + 1) for item in body)
    if tag == "S":
        return {_ed_dec(item, depth + 1) for item in body}
    if tag == "F":
        return frozenset(_ed_dec(item, depth + 1) for item in body)
    if tag == "d":
        out = {}
        for pair in body:
            if type(pair) is not list or len(pair) != 2:
                raise ValueError("malformed value")
            out[_ed_dec(pair[0], depth + 1)] = _ed_dec(pair[1], depth + 1)
        return out
    raise ValueError("malformed value")
"""

_SOLUTION_SERVER_SRC = (
    "import json, os, sys, types\n"
    + _VALUE_CODEC_SRC
    + r"""
_ed_in = os.fdopen(os.dup(0), "r", encoding="utf-8", errors="replace")
_ed_out = os.fdopen(os.dup(1), "w", encoding="utf-8")
_ed_null = os.open(os.devnull, os.O_RDONLY)
os.dup2(_ed_null, 0)
os.close(_ed_null)
os.dup2(2, 1)


def _ed_send(obj):
    _ed_out.write(json.dumps(obj) + "\n")
    _ed_out.flush()


_ed_path = os.path.abspath(sys.argv[1])
_ed_ns = {"__name__": "solution", "__file__": _ed_path}
with open(_ed_path, "r", encoding="utf-8") as _ed_f:
    _ed_src = _ed_f.read()
exec(compile(_ed_src, _ed_path, "exec"), _ed_ns)

_ed_names = {}
for _ed_k, _ed_v in list(_ed_ns.items()):
    if _ed_k.startswith("__") or isinstance(_ed_v, types.ModuleType):
        continue
    if callable(_ed_v):
        _ed_names[_ed_k] = ["call"]
        continue
    try:
        _ed_names[_ed_k] = ["val", _ed_enc(_ed_v)]
    except Exception:
        pass
_ed_send({"names": _ed_names})

for _ed_line in _ed_in:
    _ed_req = json.loads(_ed_line)
    _ed_rid = _ed_req.get("id")
    try:
        _ed_fn = _ed_ns[_ed_req["name"]]
        _ed_args = [_ed_dec(a) for a in _ed_req["args"]]
        _ed_kwargs = {k: _ed_dec(v) for k, v in _ed_req["kwargs"].items()}
        _ed_result = _ed_fn(*_ed_args, **_ed_kwargs)
    except BaseException as _ed_exc:
        _ed_send({"id": _ed_rid, "exc": [type(_ed_exc).__name__, str(_ed_exc)[:2000]]})
        continue
    try:
        _ed_payload = {"id": _ed_rid, "ok": _ed_enc(_ed_result)}
    except Exception as _ed_exc:
        _ed_payload = {"id": _ed_rid, "unsupported": str(_ed_exc)[:2000]}
    _ed_send(_ed_payload)
"""
)

_TEST_RUNNER_TEMPLATE = (
    "import builtins, json, os, select, subprocess, sys, threading, time, traceback\n"
    + _VALUE_CODEC_SRC
    + r'''
try:
    import ctypes as _ed_ctypes

    _ed_ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE=0: no ptrace / no /proc/<pid>/fd for peers.
except Exception:
    pass
_ed_null = os.open(os.devnull, os.O_RDONLY)
os.dup2(_ed_null, 0)
os.close(_ed_null)
os.closerange(3, 4096)

_ED_NONCE = __ED_NONCE__
_ED_SOLUTION = __ED_SOLUTION__
_ED_TESTS = __ED_TESTS__
_ED_BUDGET = __ED_BUDGET__
_ED_SERVER = __ED_SERVER__
_ED_MAX_LINE = 8 * 1024 * 1024


class _EdFatal(BaseException):
    """Unrecoverable solution-process failure; deliberately not an ``Exception`` so tests cannot swallow it."""


class SolutionError(Exception):
    """Non-builtin exception raised inside the solution process."""


_ed_state = {"fatal": None, "next_id": 0, "buf": b""}
_ed_deadline = time.monotonic() + _ED_BUDGET
_ed_child = subprocess.Popen(
    [sys.executable, "-I", "-c", _ED_SERVER, _ED_SOLUTION],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
_ed_err_chunks = []


def _ed_drain_stderr():
    kept = 0
    while True:
        chunk = _ed_child.stderr.read(4096)
        if not chunk:
            return
        if kept < 8192:
            _ed_err_chunks.append(chunk[: 8192 - kept])
            kept += len(chunk)


_ed_err_thread = threading.Thread(target=_ed_drain_stderr, daemon=True)
_ed_err_thread.start()


def _ed_fail(message):
    if _ed_state["fatal"] is None:
        _ed_state["fatal"] = message
    raise _EdFatal(message)


def _ed_readline():
    fd = _ed_child.stdout.fileno()
    while b"\n" not in _ed_state["buf"]:
        remaining = _ed_deadline - time.monotonic()
        if remaining <= 0:
            _ed_fail("solution timed out before unit tests completed")
        ready, _, _ = select.select([fd], [], [], remaining)
        if not ready:
            _ed_fail("solution timed out before unit tests completed")
        chunk = os.read(fd, 65536)
        if not chunk:
            _ed_fail("solution process exited before unit tests completed")
        _ed_state["buf"] += chunk
        if len(_ed_state["buf"]) > _ED_MAX_LINE:
            _ed_fail("solution process violated the result protocol (oversized message)")
    line, _ed_state["buf"] = _ed_state["buf"].split(b"\n", 1)
    try:
        msg = json.loads(line.decode("utf-8"))
    except ValueError:
        _ed_fail("solution process violated the result protocol (invalid message)")
    if type(msg) is not dict:
        _ed_fail("solution process violated the result protocol (invalid message)")
    return msg


def _ed_raise_remote(spec):
    if type(spec) is not list or len(spec) != 2 or type(spec[0]) is not str or type(spec[1]) is not str:
        _ed_fail("solution process violated the result protocol (invalid exception)")
    name, text = spec
    cls = getattr(builtins, name, None)
    if isinstance(cls, type) and issubclass(cls, Exception):
        try:
            exc = cls(text)
        except Exception:
            exc = SolutionError(name + ": " + text)
    else:
        exc = SolutionError(name + ": " + text)
    raise exc


def _ed_make_proxy(name):
    def _proxy(*args, **kwargs):
        if _ed_state["fatal"] is not None:
            raise _EdFatal(_ed_state["fatal"])
        _ed_state["next_id"] += 1
        rid = _ed_state["next_id"]
        try:
            request = {
                "id": rid,
                "name": name,
                "args": [_ed_enc(a) for a in args],
                "kwargs": {str(k): _ed_enc(v) for k, v in kwargs.items()},
            }
        except TypeError as exc:
            raise TypeError("cannot pass argument to solution function " + repr(name) + ": " + str(exc)) from None
        try:
            _ed_child.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
            _ed_child.stdin.flush()
        except OSError:
            _ed_fail("solution process exited before unit tests completed")
        msg = _ed_readline()
        if msg.get("id") != rid:
            _ed_fail("solution process violated the result protocol (unexpected response)")
        if "ok" in msg:
            try:
                return _ed_dec(msg["ok"])
            except (ValueError, TypeError, OverflowError):
                _ed_fail("solution process violated the result protocol (invalid value)")
        if "exc" in msg:
            _ed_raise_remote(msg["exc"])
        if "unsupported" in msg:
            _ed_fail("solution function " + repr(name) + " returned a value tests cannot receive: " + str(msg["unsupported"]))
        _ed_fail("solution process violated the result protocol (unexpected response)")

    _proxy.__name__ = name
    _proxy.__qualname__ = name
    return _proxy


_ed_passed = False
_ed_error = ""
try:
    _ed_hello = _ed_readline()
    _ed_names = _ed_hello.get("names")
    if type(_ed_names) is not dict:
        _ed_fail("solution process violated the result protocol (missing names)")
    # Builtins and runner-reserved names stay authoritative: a solution defining ``abs``/``len``/``SolutionError``
    # (or any ``_ed_*`` / dunder name) must not shadow what the trusted tests resolve.
    _ed_reserved = frozenset(dir(builtins)) | {"SolutionError"}
    _ed_test_ns = {"__name__": "__easydel_tests__", "__builtins__": builtins}
    for _ed_k, _ed_spec in _ed_names.items():
        if type(_ed_k) is not str or not _ed_k.isidentifier() or type(_ed_spec) is not list:
            continue
        if _ed_k in _ed_reserved or _ed_k.startswith(("_ed_", "__")):
            continue
        if _ed_spec == ["call"]:
            _ed_test_ns[_ed_k] = _ed_make_proxy(_ed_k)
        elif len(_ed_spec) == 2 and _ed_spec[0] == "val":
            try:
                _ed_test_ns[_ed_k] = _ed_dec(_ed_spec[1])
            except (ValueError, TypeError, OverflowError):
                _ed_fail("solution process violated the result protocol (invalid value)")
    _ed_test_ns["SolutionError"] = SolutionError
    _ed_test_ns["__builtins__"] = builtins
    _ed_real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        exec(compile(_ED_TESTS, "<tests>", "exec"), _ed_test_ns)
    finally:
        sys.stdout = _ed_real_stdout
    _ed_passed = _ed_state["fatal"] is None
    _ed_error = _ed_state["fatal"] or ""
except _EdFatal as _ed_exc:
    _ed_error = _ed_state["fatal"] or str(_ed_exc)
except BaseException:
    _ed_error = _ed_state["fatal"] or traceback.format_exc(limit=6)
finally:
    try:
        _ed_child.kill()
    except OSError:
        pass
    try:
        _ed_child.wait(timeout=2.0)
    except Exception:
        pass
    _ed_err_thread.join(timeout=1.0)

if _ed_state["fatal"] is not None:
    _ed_passed = False
    _ed_error = _ed_state["fatal"]
_ed_stderr = b"".join(_ed_err_chunks).decode("utf-8", "replace")
_ed_record = {"passed": bool(_ed_passed), "error": _ed_error[-4000:], "solution_stderr": _ed_stderr[-2000:]}
sys.stdout.write(_ED_NONCE + " " + json.dumps(_ed_record) + "\n")
sys.stdout.flush()
'''
)


def _build_unit_test_runner(nonce: str, solution_path: str, test_code: str, budget: float) -> str:
    """Render the trusted test-runner source for one evaluation.

    Args:
        nonce: Fresh per-evaluation secret that prefixes the single result line.
        solution_path: Workspace-relative path of the solution file.
        test_code: Trusted assertion/unit-test code executed inside the runner.
        budget: Seconds the runner allows the solution process before failing it.

    Returns:
        Python source for ``python -I -``.
    """
    return (
        _TEST_RUNNER_TEMPLATE.replace("__ED_NONCE__", repr(nonce))
        .replace("__ED_SOLUTION__", repr(solution_path))
        .replace("__ED_TESTS__", repr(test_code))
        .replace("__ED_BUDGET__", repr(float(budget)))
        .replace("__ED_SERVER__", repr(_SOLUTION_SERVER_SRC))
    )


def _parse_runner_record(stdout: str, nonce: str) -> dict[str, tp.Any] | None:
    """Return the runner's result record, or ``None`` unless exactly one output line carries ``nonce``.

    The runner prints the nonce exactly once. Zero occurrences means no result; more than one means something
    besides the runner learned the nonce (e.g. a leaked descriptor or a readable runner source), so the output is
    untrustworthy and the evaluation is treated as unscored (reward 0).
    """
    prefix = nonce + " "
    carrying = [line for line in stdout.splitlines() if nonce in line]
    if len(carrying) != 1 or not carrying[0].startswith(prefix):
        return None
    try:
        record = json.loads(carrying[0][len(prefix) :])
    except ValueError:
        return None
    return record if isinstance(record, dict) else None


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

    _infra_failure_step = staticmethod(unscorable_step)

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
                "isolated_test_runner": bool(self._test_code and self._language == "python"),
            },
        )

    def _evaluate_code(self, code: str) -> tuple[int, int, str, str | None]:
        src_name, compile_cmd, run_cmd = _LANGUAGE_SPECS[self._language]
        session = self.session

        if self._test_code and self._language == "python":
            # The solution runs in a child process of a trusted runner; see the module-level design notes.
            # Only the runner's single nonce-prefixed result line decides the reward.
            import sys as _sys

            from .runtime import LocalSandboxSession

            nonce = f"__ED_RESULT_{secrets.token_hex(16)}__"
            session.write_files({src_name: code})
            runner_src = _build_unit_test_runner(nonce, src_name, self._test_code, self._test_timeout)
            py_bin = _sys.executable if isinstance(session, LocalSandboxSession) else "python3"
            res = session.exec([py_bin, "-I", "-"], stdin=runner_src, timeout=self._test_timeout + 5.0)
            if res.infra_error:
                return 0, 1, "", res.infra_error
            record = _parse_runner_record(res.stdout, nonce)
            if record is None:
                detail = res.combined_output(max_chars=4096).replace(nonce, "").strip()
                if res.timed_out:
                    return 0, 1, f"Unit tests timed out.\n{detail}".strip(), None
                return 0, 1, f"Unit-test runner produced no result.\n{detail}".strip(), None
            if record.get("passed") is True:
                return 1, 1, "All unit tests passed.", None
            feedback = str(record.get("error", "")).strip()
            child_err = str(record.get("solution_stderr", "")).strip()
            if child_err:
                feedback = f"{feedback}\n--- solution stderr ---\n{child_err}".strip()
            return 0, 1, feedback or "Unit tests failed.", None

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
                "isolated_test_runner": bool(self._test_code and self._language == "python"),
            },
        )


__all__ = ("CodeJudgeEnvironment", "extract_code_block")
