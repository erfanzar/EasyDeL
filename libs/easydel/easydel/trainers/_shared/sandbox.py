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

"""Hardened out-of-process sandbox for running untrusted model-generated code.

RL-with-verifiable-rewards and agentic tool use both execute strings emitted
by the policy (Python snippets, shell commands, unit tests). Running those
in-process with ``exec`` is unsafe: a segfault / ``os._exit`` / unbounded
allocation in the generated code takes down the trainer worker, ``signal.alarm``
timeouts only fire on the main thread, and the code sees the parent process's
full builtins, environment, and filesystem.

:class:`Sandbox` runs each snippet in a **child process** with:

* a hard wall-clock timeout enforced by killing the child's *process group*
  (so forked grandchildren die too) — works from any thread;
* ``RLIMIT_CPU`` and ``RLIMIT_AS`` (CPU-seconds and address-space caps) via a
  ``preexec_fn``, so a busy-loop or allocation bomb is bounded;
* a scrubbed environment (secrets / proxy vars dropped) and a fresh temporary
  working directory (writes cannot touch the repo or checkpoints);
* the Python interpreter launched in isolated mode (``-I``) so it ignores
  ``PYTHONPATH`` / user site-packages.

When ``bubblewrap`` (``bwrap``) or ``nsjail`` is available on ``PATH`` the
command is additionally wrapped for filesystem / PID / **network** namespace
isolation (network egress cannot be blocked without one of these on an
unprivileged host — that is documented, not silently assumed).

This is a defense-in-depth improvement over in-process ``exec``; it is not a
substitute for a VM/container boundary against a determined adversary.
"""

from __future__ import annotations

import dataclasses
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import typing as tp

_POSIX = os.name == "posix"
# Return code of a child killed by the RLIMIT_CPU backstop. SIGXCPU is POSIX-only,
# and no rlimits are installed off POSIX, so no return code maps to a budget kill
# there.
_CPU_BUDGET_RETURNCODE: int | None = -signal.SIGXCPU if _POSIX else None

# Environment variables that are safe to expose to sandboxed processes. Anything
# else (API keys, cloud creds, proxies, JAX/XLA/TPU coordinator vars) is dropped.
_ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "TZ", "HOME", "TMPDIR")


@dataclasses.dataclass(frozen=True)
class SandboxResult:
    """Outcome of one sandboxed execution.

    Attributes:
        ok: True when the child exited with code 0 and did not time out.
        stdout: Captured standard output (truncated to ``max_output``).
        stderr: Captured standard error (truncated to ``max_output``).
        returncode: Child exit code, or ``None`` if it never exited cleanly.
        timed_out: True if the execution exhausted its time budget -- either the
            wall-clock timeout fired and the group was killed, or the ``RLIMIT_CPU``
            backstop terminated the child with ``SIGXCPU`` first.
    """

    ok: bool
    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool


class Sandbox:
    """Run untrusted Python / shell in a resource-limited child process.

    Attributes:
        _timeout (float): Wall-clock budget (seconds) per execution.
        _cpu_seconds (int): ``RLIMIT_CPU`` soft/hard cap.
        _memory_mb (int): ``RLIMIT_AS`` address-space cap (MiB); 0 disables it.
        _max_output (int): Byte cap applied to captured stdout/stderr.
        _network (bool): Allow network egress (only enforceable under a jail).
        _jail (list[str] | None): Resolved bwrap/nsjail wrapper argv, or None.
    """

    def __init__(
        self,
        *,
        timeout: float = 10.0,
        cpu_seconds: int | None = None,
        memory_mb: int = 2048,
        max_output: int = 8192,
        network: bool = False,
        use_jail: str = "auto",
    ):
        """Initialize a sandbox.

        Args:
            timeout: Wall-clock budget in seconds; the child's process group is
                SIGKILLed when it elapses.
            cpu_seconds: ``RLIMIT_CPU`` cap; defaults to ``ceil(timeout) + 1``.
            memory_mb: ``RLIMIT_AS`` address-space cap in MiB (0 to disable).
            max_output: Byte cap for captured stdout/stderr.
            network: Permit network egress. Only actually enforced when a jail
                (bwrap/nsjail) is used; without one, egress cannot be blocked.
            use_jail: ``"auto"`` (wrap with bwrap/nsjail if present, else plain
                subprocess), ``"required"`` (raise if neither is available), or
                ``"off"`` (never wrap).

        Raises:
            RuntimeError: If ``use_jail="required"`` but no jailer is on PATH.
        """
        self._timeout = float(timeout)
        # ceil, not truncation: a fractional timeout such as 1.5 would otherwise get
        # a 2-second CPU cap, leaving the rlimit backstop only half a second behind
        # the wall clock. A busy child then trips RLIMIT_CPU before a contended
        # parent thread observes its own timeout, and the kill is reported as a
        # plain non-zero exit instead of a timeout.
        self._cpu_seconds = int(cpu_seconds) if cpu_seconds is not None else math.ceil(self._timeout) + 1
        self._memory_mb = int(memory_mb)
        self._max_output = int(max_output)
        self._network = bool(network)
        self._jail = self._resolve_jail(use_jail)

    def _resolve_jail(self, use_jail: str) -> list[str] | None:
        """Resolve a bwrap/nsjail wrapper argv for the current settings.

        Args:
            use_jail: The ``use_jail`` policy (``"auto"``/``"required"``/``"off"``).

        Returns:
            The wrapper argv prefix, or ``None`` for plain subprocess.

        Raises:
            RuntimeError: If ``"required"`` and no jailer is available.
        """
        if use_jail == "off" or not _POSIX:
            if use_jail == "required":
                raise RuntimeError("Sandbox jail required but unavailable on this platform.")
            return None
        bwrap = shutil.which("bwrap")
        if bwrap:
            argv = [bwrap, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--die-with-parent"]
            if not self._network:
                argv.append("--unshare-net")
            return argv
        nsjail = shutil.which("nsjail")
        if nsjail:
            argv = [nsjail, "-Mo", "--really_quiet", "--"]
            return argv if self._network else [nsjail, "-Mo", "--really_quiet", "--disable_clone_newnet", "--"]
        if use_jail == "required":
            raise RuntimeError("Sandbox jail required but neither bwrap nor nsjail is on PATH.")
        return None

    def _limits(self) -> tp.Callable[[], None] | None:
        """Build the ``preexec_fn`` that applies rlimits in the child.

        Returns:
            A no-arg callable for ``subprocess`` ``preexec_fn``, or ``None`` on
            non-POSIX platforms.
        """
        if not _POSIX:
            return None
        cpu = self._cpu_seconds
        mem_bytes = self._memory_mb * 1024 * 1024 if self._memory_mb > 0 else 0

        def _apply() -> None:
            """Apply CPU / address-space rlimits inside the forked child."""
            import resource

            # Soft below hard so exhausting the CPU budget raises SIGXCPU, which is
            # attributable, rather than a bare SIGKILL that is indistinguishable
            # from an OOM kill. The hard limit still guarantees termination if the
            # child installs a SIGXCPU handler.
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
            if mem_bytes:
                try:
                    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
                except (ValueError, OSError):
                    pass

        return _apply

    def _run(self, argv: list[str], *, stdin: str | None = None) -> SandboxResult:
        """Execute ``argv`` in the sandbox and capture its result.

        Runs in a new session (process group) so a timeout can kill any
        grandchildren; applies the jail wrapper and rlimits; enforces the
        wall-clock timeout with a group SIGKILL.

        Args:
            argv: Program + arguments to execute.
            stdin: Optional string piped to the child's standard input.

        Returns:
            A :class:`SandboxResult`.
        """
        full_argv = (self._jail + argv) if self._jail else argv
        env = {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}
        env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
        tmp = tempfile.mkdtemp(prefix="edsandbox-")
        try:
            proc = subprocess.Popen(
                full_argv,
                stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=tmp,
                env=env,
                text=True,
                preexec_fn=self._limits() if not self._jail else None,  # jail owns its own limits
                start_new_session=_POSIX,
            )
            try:
                out, err = proc.communicate(input=stdin, timeout=self._timeout)
                return SandboxResult(
                    ok=proc.returncode == 0,
                    stdout=out[: self._max_output],
                    stderr=err[: self._max_output],
                    returncode=proc.returncode,
                    # RLIMIT_CPU is the backstop for the wall clock, so a child it
                    # kills has exhausted its time budget just as surely as one the
                    # timeout path reaps -- and it can win the race whenever this
                    # thread is slow to observe its own deadline.
                    timed_out=_CPU_BUDGET_RETURNCODE is not None and proc.returncode == _CPU_BUDGET_RETURNCODE,
                )
            except subprocess.TimeoutExpired:
                self._kill_group(proc)
                out, err = proc.communicate()
                return SandboxResult(
                    ok=False,
                    stdout=(out or "")[: self._max_output],
                    stderr=(err or "")[: self._max_output],
                    returncode=proc.returncode,
                    timed_out=True,
                )
        except Exception as exc:  # pragma: no cover - launch failure (e.g. missing interpreter)
            return SandboxResult(
                ok=False, stdout="", stderr=f"{type(exc).__name__}: {exc}", returncode=None, timed_out=False
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @staticmethod
    def _kill_group(proc: subprocess.Popen) -> None:
        """SIGKILL the child's whole process group (best-effort).

        Args:
            proc: The running child process.
        """
        if not _POSIX:
            proc.kill()
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()

    def run_python(self, code: str, *, stdin: str | None = None) -> SandboxResult:
        """Run a Python snippet in the sandbox (isolated interpreter).

        Args:
            code: Python source to execute.
            stdin: Optional standard input for the snippet.

        Returns:
            A :class:`SandboxResult` (``ok`` iff it exited 0 without timing out).
        """
        return self._run([sys.executable, "-I", "-c", code], stdin=stdin)

    def run_bash(self, command: str, *, shell: str = "/bin/bash") -> SandboxResult:
        """Run a shell command in the sandbox.

        Args:
            command: Shell command line.
            shell: Shell interpreter to use.

        Returns:
            A :class:`SandboxResult`.
        """
        return self._run([shell, "-c", command])
