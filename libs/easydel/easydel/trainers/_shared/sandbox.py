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
import threading
import time
import typing as tp

# Pipe readers retain only the configured prefix; excess output is drained in
# fixed-size chunks so neither normal completion nor timeout can exhaust RAM.
_CAPTURE_CHUNK_SIZE = 8192
_RUNTIME_PATH = "/usr/local/bin:/usr/bin:/bin"

_POSIX = os.name == "posix"
# Return code of a child killed by the RLIMIT_CPU backstop. SIGXCPU is POSIX-only,
# and no rlimits are installed off POSIX, so no return code maps to a budget kill
# there.
_CPU_BUDGET_RETURNCODE: int | None = -signal.SIGXCPU if _POSIX else None

# Environment variables that are safe to expose to sandboxed processes. Anything
# else (API keys, cloud creds, proxies, JAX/XLA/TPU coordinator vars) is dropped.
_ENV_ALLOWLIST = ("LANG", "LC_ALL", "TZ")


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
        _max_output (int): Character cap applied independently to stdout/stderr.
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
            max_output: Character cap for each captured stream. Excess output is
                drained and discarded without buffering it.
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
        self._max_output = max(0, int(max_output))
        self._network = bool(network)
        self._jail_owns_limits = False
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
            argv = [
                bwrap,
                "--unshare-user",
                "--unshare-pid",
                "--unshare-ipc",
                "--unshare-uts",
                "--cap-drop",
                "ALL",
                "--die-with-parent",
            ]
            # Start with bwrap's empty root, not a read-only view of the host.
            # System executables/libraries and the base Python installation are
            # sufficient for isolated Python and shell execution. In particular,
            # never mount the venv prefix: it may contain the trainer repository.
            runtime_dirs = ["/usr", "/bin", "/sbin", "/lib", "/lib64"]
            base_prefix = os.path.realpath(sys.base_prefix)
            if not any(base_prefix == path or base_prefix.startswith(path + "/") for path in runtime_dirs):
                if base_prefix == "/":
                    raise RuntimeError("Cannot safely expose a Python runtime rooted at /.")
                runtime_dirs.append(base_prefix)
            for path in runtime_dirs:
                if os.path.exists(path):
                    argv += ["--ro-bind", path, path]
            argv += ["--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]
            if not self._network:
                argv.append("--unshare-net")
            else:
                # DNS needs only these configuration files, never all of /etc.
                for path in ("/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf"):
                    if os.path.exists(path):
                        argv += ["--ro-bind", path, path]
            return argv
        nsjail = shutil.which("nsjail")
        if nsjail:
            argv = [nsjail, "-Mo", "--really_quiet"]
            # nsjail isolates the network namespace by default; --disable_clone_newnet
            # shares the host network, so it is only passed when egress is allowed.
            if self._network:
                argv.append("--disable_clone_newnet")
            # nsjail resets the child's rlimits to its own defaults, overriding any
            # inherited ones, so the sandbox caps are passed as nsjail flags.
            argv += ["--rlimit_cpu", str(self._cpu_seconds)]
            argv += ["--rlimit_as", str(self._memory_mb) if self._memory_mb > 0 else "inf"]
            self._jail_owns_limits = True
            return [*argv, "--"]
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

            # Ignored and blocked signals survive fork+exec. A host process that ignores or blocks SIGXCPU (a
            # long-lived trainer or test runner, a native runtime) would otherwise make the soft CPU limit a no-op,
            # and the child would die by the hard limit's SIGKILL, which is not reported as a timeout.
            try:
                signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGXCPU})
                signal.signal(signal.SIGXCPU, signal.SIG_DFL)
            except (ValueError, OSError):
                pass

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
        env = {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}
        env["PATH"] = _RUNTIME_PATH
        tmp = tempfile.mkdtemp(prefix="edsandbox-")
        env.update(HOME=tmp, TMPDIR=tmp)
        if self._jail and not self._jail_owns_limits:
            # Mount only this execution's writable directory. Resolve the venv
            # executable to its base interpreter, avoiding a mount of host home
            # or the repository just to follow the venv's interpreter symlink.
            full_argv = [*self._jail, "--bind", tmp, tmp, "--chdir", tmp, "--", os.path.realpath(argv[0]), *argv[1:]]
        else:
            full_argv = (self._jail + argv) if self._jail else argv
        try:
            proc = subprocess.Popen(
                full_argv,
                stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=tmp,
                env=env,
                text=True,
                errors="replace",
                # rlimits survive exec, so they bound a bwrap-wrapped child too; only
                # nsjail overrides them, and it receives the caps as flags instead.
                preexec_fn=None if self._jail_owns_limits else self._limits(),
                start_new_session=_POSIX,
            )
            out, err, timed_out = self._capture(proc, stdin)
            # RLIMIT_CPU can win the race against the wall-clock deadline.
            timed_out |= _CPU_BUDGET_RETURNCODE is not None and proc.returncode == _CPU_BUDGET_RETURNCODE
            return SandboxResult(
                ok=proc.returncode == 0 and not timed_out,
                stdout=out,
                stderr=err,
                returncode=proc.returncode,
                timed_out=timed_out,
            )
        except Exception as exc:  # pragma: no cover - launch failure (e.g. missing interpreter)
            return SandboxResult(
                ok=False, stdout="", stderr=f"{type(exc).__name__}: {exc}", returncode=None, timed_out=False
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _capture(self, proc: subprocess.Popen, stdin: str | None) -> tuple[str, str, bool]:
        """Drain both pipes concurrently, retaining at most one prefix per stream.

        Reader and optional stdin-writer threads share the execution deadline.
        This also times out descendants holding pipes open after the child exits.
        Excess output is discarded, including after a timeout kill, rather than
        accumulated by ``communicate()``.
        """
        buffers: list[list[str]] = [[], []]

        def read_stream(stream: tp.TextIO, parts: list[str]) -> None:
            remaining = self._max_output
            with stream:
                while chunk := stream.read(_CAPTURE_CHUNK_SIZE):
                    if remaining:
                        prefix = chunk[:remaining]
                        parts.append(prefix)
                        remaining -= len(prefix)

        def write_stdin() -> None:
            with proc.stdin:
                try:
                    proc.stdin.write(stdin)
                    proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass  # The child may exit without consuming its input.

        workers = [
            threading.Thread(target=read_stream, args=(stream, parts), daemon=True)
            for stream, parts in zip((proc.stdout, proc.stderr), buffers, strict=True)
        ]
        if stdin is not None:
            workers.append(threading.Thread(target=write_stdin, daemon=True))
        deadline = time.monotonic() + self._timeout
        for worker in workers:
            worker.start()
        timed_out = False
        try:
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
            for worker in workers:
                worker.join(timeout=max(0.0, deadline - time.monotonic()))
            timed_out = any(worker.is_alive() for worker in workers)
        except subprocess.TimeoutExpired:
            timed_out = True
        if timed_out:
            self._kill_group(proc)
            proc.wait()
            # A descendant could deliberately leave the process group. Do not
            # let its open pipe defeat the deadline; daemon readers still retain
            # only the bounded prefix until the remaining descriptors close.
            for worker in workers:
                worker.join(timeout=0.1)
        return "".join(buffers[0]), "".join(buffers[1]), timed_out

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
            # start_new_session makes the child's PID its process-group ID.
            # Keep using that ID if wait() already reaped the group leader while
            # grandchildren still hold stdout/stderr open.
            os.killpg(proc.pid, signal.SIGKILL)
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
