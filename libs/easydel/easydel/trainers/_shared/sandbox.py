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
import hashlib
import hmac
import math
import os
import secrets
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import typing as tp
import warnings

try:
    import resource
except ImportError:  # pragma: no cover - non-POSIX
    resource = None  # type: ignore[assignment]

# Pipe readers retain only the configured prefix; excess output is drained in
# fixed-size chunks so neither normal completion nor timeout can exhaust RAM.
_CAPTURE_CHUNK_SIZE = 8192
_RUNTIME_PATH = "/usr/local/bin:/usr/bin:/bin"

_POSIX = os.name == "posix"
# Return codes of a child killed by the RLIMIT_CPU backstop. SIGXCPU is
# POSIX-only, and no rlimits are installed off POSIX. Direct kills report
# -SIGXCPU while jailed/shell-wrapped children commonly report 128 + SIGXCPU.
_CPU_BUDGET_RETURNCODE: int | None = -signal.SIGXCPU if _POSIX else None
_CPU_BUDGET_RETURNCODES: frozenset[int] = (
    frozenset({-signal.SIGXCPU, 128 + signal.SIGXCPU}) if _POSIX else frozenset()
)

_VALID_JAIL_MODES = frozenset({"auto", "required", "off", "bwrap", "nsjail"})

# Environment variables that are safe to expose to sandboxed processes. Anything
# else (API keys, cloud creds, proxies, JAX/XLA/TPU coordinator vars) is dropped.
_ENV_ALLOWLIST = ("LANG", "LC_ALL", "TZ")

# Cached results of one-time jail usability probes keyed by (jail_kind, prefix).
_JAIL_PROBE_LOCK = threading.Lock()
_JAIL_PROBE_CACHE: dict[tuple[str, tuple[str, ...]], tuple[bool, str]] = {}

_SENTINEL_HARNESS_SOURCE = """\
def _easydel_run_verified():
    import _thread as _cthread
    import collections as _collections
    import functools as _functools
    import gc as _gc
    import hmac as _hmac_mod
    import operator as _operator
    import os as _os
    import re as _re
    import secrets as _secrets
    import sys as _sys

    try:
        import _hashlib as _chashlib

        _c_hmac_digest = _chashlib.hmac_digest
    except Exception:
        _c_hmac_digest = None

    if _c_hmac_digest is None:
        import hashlib as _hashlib

        _py_hmac_digest = _hmac_mod.digest
        _sha256_mod = _hashlib.sha256

        def _hmac_sha256(_key, _msg):
            return _py_hmac_digest(_key, _msg, _sha256_mod)
    else:

        def _hmac_sha256(_key, _msg):
            return _c_hmac_digest(_key, _msg, "sha256")

    _compare_digest = _hmac_mod.compare_digest

    _raw_cmd_r = int(_sys.argv[1])
    _raw_ack_w = int(_sys.argv[2])
    del _sys.argv[1:]

    # Cache all C primitives, builtins, and exception types before untrusted code
    # executes so monkeypatching builtins/threading/os/hmac/gc/sys has zero effect.
    _os_read = _os.read
    _os_write = _os.write
    _os_close = _os.close
    _os_dup2 = _os.dup2
    _os_set_inheritable = _os.set_inheritable
    _os_exit = _os._exit
    _exec = exec
    _compile = compile
    _getattr = getattr
    _isinstance = isinstance
    _len = len
    _bytes = bytes
    _bytearray = bytearray
    _ba_extend = bytearray.extend
    _ba_decode = bytearray.decode
    _ba_clear = bytearray.clear
    _bytes_decode = bytes.decode
    _int_from_bytes = int.from_bytes
    _int_to_bytes = int.to_bytes
    _str = str
    _BaseException = BaseException
    _PermissionError = PermissionError
    _OSError = OSError
    _deque = _collections.deque
    _map = map
    _partial = _functools.partial
    _op_call = _operator.call
    _start_joinable_thread = _getattr(_cthread, "start_joinable_thread", None)
    _start_new_thread = _cthread.start_new_thread
    _allocate_lock = _cthread.allocate_lock
    _stdout = _sys.stdout
    _stderr = _sys.stderr
    _harness_file = __file__
    _proc_secret_search = _re.compile(r"(?:^|/)proc(?:/|$)").search

    def _relocate_fd(_src_fd, _avoid):
        _upper = 1023
        try:
            import resource as _resource

            _soft, _ = _resource.getrlimit(_resource.RLIMIT_NOFILE)
            if _soft > 0:
                _upper = min(_upper, int(_soft) - 1)
        except Exception:
            pass
        _lo = max(_src_fd, *_avoid, 63) + 1
        if _lo < _upper:
            for _ in range(32):
                _cand = _lo + _secrets.randbelow(_upper - _lo + 1)
                if _cand in _avoid:
                    continue
                try:
                    _os.fstat(_cand)
                except OSError:
                    try:
                        _os_dup2(_src_fd, _cand, inheritable=False)
                        _os_close(_src_fd)
                        return _cand
                    except OSError:
                        pass
        _os_set_inheritable(_src_fd, False)
        return _src_fd

    _cmd_r = _relocate_fd(_raw_cmd_r, {_raw_cmd_r, _raw_ack_w})
    _ack_w = _relocate_fd(_raw_ack_w, {_cmd_r, _raw_ack_w})
    del _raw_cmd_r, _raw_ack_w

    def _read_exact(_n):
        # Accumulate into a non-GC-tracked bytearray and return None on EOF/error
        # so main-thread protocol I/O allocates ZERO GC-tracked objects.
        _buf = _bytearray()
        _rem = _n
        while _rem > 0:
            try:
                _chunk = _os_read(_cmd_r, _rem)
            except _OSError:
                return None
            if not _chunk:
                return None
            _ba_extend(_buf, _chunk)
            _rem -= _len(_chunk)
        return _buf

    def _write_all(_data):
        while _data:
            try:
                _written = _os_write(_ack_w, _data)
            except _OSError:
                return False
            if _written <= 0:
                return False
            _data = _data[_written:]
        return True

    def _xor_ctr(_key, _nonce, _data):
        # CTR stream cipher using C HMAC-SHA256 blocks with ZERO GC-tracked allocations.
        _n = _len(_data)
        _out = _bytearray()
        _off = 0
        _blk = 0
        while _off < _n:
            _end = _off + 32
            if _end > _n:
                _end = _n
            _sz = _end - _off
            _ks = _hmac_sha256(_key, b"k2:" + _nonce + _int_to_bytes(_blk, 4, "big"))
            _x = _int_from_bytes(_data[_off:_end], "big") ^ _int_from_bytes(_ks[:_sz], "big")
            _ba_extend(_out, _int_to_bytes(_x, _sz, "big"))
            _off = _end
            _blk += 1
        return _out

    def _wipe_and_close(_key_buf):
        _i = 0
        _klen = _len(_key_buf)
        while _i < _klen:
            _key_buf[_i] = 0
            _i += 1
        _ba_clear(_key_buf)
        try:
            _os_close(_cmd_r)
        except _OSError:
            pass
        try:
            _os_close(_ack_w)
        except _OSError:
            pass

    def _flush_and_fail(_key_buf):
        _wipe_and_close(_key_buf)
        try:
            _stdout.flush()
        except _BaseException:
            pass
        try:
            _stderr.flush()
        except _BaseException:
            pass
        _os_exit(1)

    _blocked_events = frozenset(
        {
            "sys._current_frames",
            "sys.addaudithook",
            "sys.monitoring.register_callback",
            "gc.get_objects",
            "gc.get_referrers",
            "gc.get_referents",
            "ctypes.string_at",
            "ctypes.wstring_at",
            "ctypes.cdata",
            "ctypes.cdata/buffer",
            "ctypes.create_string_buffer",
            "ctypes.call_function",
            "ctypes.dlsym",
            "ctypes.dlopen",
        }
    )
    _path_events = frozenset({"open", "os.listdir", "os.scandir", "os.chdir", "os.symlink", "os.link"})

    def _check_path_arg(_target):
        if _isinstance(_target, _bytes):
            _text = _bytes_decode(_target, "utf-8", "replace")
        elif _isinstance(_target, _str):
            _text = _target
        else:
            return
        if _text and _proc_secret_search(_text):
            raise _PermissionError("Access to /proc is denied inside sandboxed verification.")

    def _audit_hook(_event, _args):
        # Unconditionally active on ALL threads at ALL times (the harness itself
        # never triggers any blocked event).
        if _event == "sys._getframe":
            _f = _args[0] if _args else None
            while _f is not None:
                if _getattr(_getattr(_f, "f_code", None), "co_filename", None) == _harness_file:
                    raise _PermissionError("Access to sandbox harness frame is denied.")
                _f = _getattr(_f, "f_back", None)
        elif _event in _blocked_events:
            raise _PermissionError(f"Audit event {_event!r} is denied inside sandboxed verification.")
        elif _event in _path_events:
            if _args:
                _check_path_arg(_args[0])
                if _len(_args) > 1 and _event in ("os.symlink", "os.link"):
                    _check_path_arg(_args[1])

    _ns = {"__name__": "__main__"}
    _hmac_key = _read_exact(32)
    if _hmac_key is None:
        _wipe_and_close(_bytearray())
        _os_exit(1)

    # Read Phase-1 nonce & candidate source BEFORE untrusted code runs.
    _nonce1 = _read_exact(16)
    _code_len_raw = _read_exact(4) if _nonce1 is not None else None
    _code_bytes = _read_exact(_int_from_bytes(_code_len_raw, "big")) if _code_len_raw is not None else None
    if _code_bytes is None:
        _wipe_and_close(_hmac_key)
        _os_exit(1)
    _code_tag = _hmac_sha256(_hmac_key, b"c1:" + _code_bytes)
    _code_src = _ba_decode(_code_bytes, "utf-8")
    del _code_len_raw, _code_bytes

    # Pre-allocate AND pre-spawn both pure-C worker threads BEFORE any untrusted
    # code runs so:
    # 1. Both phases run with ZERO Python caller frames (sys._getframe(0).f_back is None).
    # 2. Post-Phase-1 main-thread execution allocates ZERO GC-tracked objects and
    #    never decrefs any untrusted object, making cyclic GC / __del__ / gc.callbacks
    #    hijacking on the main thread impossible even under gc.set_threshold(1, 1, 1).
    # 3. Because worker threads are already spawned via _thread (not threading.Thread),
    #    threading.settrace/setprofile installed during Phase 1 cannot touch Phase 2.
    _p1_src_box = [_code_src]
    _p1_done = _bytearray(1)
    _p1_go = _allocate_lock()
    _p1_go.acquire()
    _p1_exec = _partial(
        _deque,
        _map(_exec, _map(_compile, _p1_src_box, ("<solution>",), ("exec",)), (_ns,)),
        0,
    )
    _p1_target = _partial(
        _deque,
        _map(_op_call, (_p1_go.acquire, _p1_exec, _partial(_p1_done.__setitem__, 0, 1))),
        0,
    )

    _p2_src_box = [""]
    _p2_done = _bytearray(1)
    _p2_go = _allocate_lock()
    _p2_go.acquire()
    _p2_exec = _partial(
        _deque,
        _map(_exec, _map(_compile, _p2_src_box, ("<tests>",), ("exec",)), (_ns,)),
        0,
    )
    _p2_target = _partial(
        _deque,
        _map(_op_call, (_p2_go.acquire, _p2_exec, _partial(_p2_done.__setitem__, 0, 1))),
        0,
    )

    if _start_joinable_thread is not None:
        _handle1 = _start_joinable_thread(_p1_target)
        _handle2 = _start_joinable_thread(_p2_target)
        _join1 = _handle1.join
        _join2 = _handle2.join
    else:
        _p1_fin = _allocate_lock()
        _p1_fin.acquire()
        _p2_fin = _allocate_lock()
        _p2_fin.acquire()

        def _legacy_worker(_target, _fin):
            try:
                _target()
            finally:
                _fin.release()

        _start_new_thread(_legacy_worker, (_p1_target, _p1_fin))
        _start_new_thread(_legacy_worker, (_p2_target, _p2_fin))
        _join1 = _p1_fin.acquire
        _join2 = _p2_fin.acquire

    # Replace gc.callbacks module attribute with a decoy list so CPython's internal
    # interp->gc.callbacks stays permanently empty and unreachable.
    _gc.callbacks = []
    _gc.collect()
    _sys.addaudithook(_audit_hook)

    # Phase 1: release pre-spawned C worker thread 1 and authenticate completion.
    del _code_src
    _p1_go.release()
    _join1()
    if not _p1_done[0]:
        _flush_and_fail(_hmac_key)
    if not _write_all(_hmac_sha256(_hmac_key, b"p1:" + _nonce1 + _code_tag)):
        _wipe_and_close(_hmac_key)
        _os_exit(1)

    # Phase 2: receive AEAD-encrypted & MAC-authenticated tests only after Phase 1
    # authenticates, verify tag2 before decrypting, execute on pre-spawned worker 2,
    # then authenticate completion bound to tag2.
    _nonce2 = _read_exact(16)
    _ct_len_raw = _read_exact(4) if _nonce2 is not None else None
    _ct = _read_exact(_int_from_bytes(_ct_len_raw, "big")) if _ct_len_raw is not None else None
    _tag2 = _read_exact(32) if _ct is not None else None
    if _tag2 is None:
        _wipe_and_close(_hmac_key)
        _os_exit(1)
    if not _compare_digest(_tag2, _hmac_sha256(_hmac_key, b"t2:" + _nonce2 + _ct)):
        _wipe_and_close(_hmac_key)
        _os_exit(1)
    _test_src = _ba_decode(_xor_ctr(_hmac_key, _nonce2, _ct), "utf-8")
    _p2_src_box[0] = _test_src
    _p2_go.release()
    _join2()
    if not _p2_done[0]:
        _flush_and_fail(_hmac_key)
    if not _write_all(_hmac_sha256(_hmac_key, b"p2:" + _nonce2 + _tag2)):
        _wipe_and_close(_hmac_key)
        _os_exit(1)

    # Phase 3: authenticate fresh post-test challenge generated only after Phase 2 succeeds.
    _challenge = _read_exact(32)
    if _challenge is None:
        _wipe_and_close(_hmac_key)
        _os_exit(1)
    if not _write_all(_hmac_sha256(_hmac_key, b"p3:" + _challenge)):
        _wipe_and_close(_hmac_key)
        _os_exit(1)

    # Zero secret key and close pipes BEFORE normal interpreter shutdown/flush.
    _wipe_and_close(_hmac_key)


_easydel_run_verified()
"""


def validate_use_jail(use_jail: str) -> str:
    """Validate the ``use_jail`` policy string.

    Args:
        use_jail: One of ``"auto"``, ``"required"``, ``"off"``, ``"bwrap"``,
            or ``"nsjail"``.

    Returns:
        The validated policy string.

    Raises:
        ValueError: If ``use_jail`` is not a recognized policy.
    """
    if not isinstance(use_jail, str) or use_jail not in _VALID_JAIL_MODES:
        allowed = ", ".join(sorted(_VALID_JAIL_MODES))
        raise ValueError(f"Invalid use_jail={use_jail!r}; expected one of: {allowed}.")
    return use_jail


def _build_jail_command(
    kind: str | None,
    jail_prefix: list[str] | None,
    argv: list[str],
    *,
    tmp: str,
    env: dict[str, str],
    ro_binds: tp.Sequence[str] = (),
    pass_fds: tp.Sequence[int] = (),
) -> list[str]:
    """Assemble the full subprocess command for a given jail backend."""
    if not jail_prefix or kind is None:
        return list(argv)
    resolved_exe = os.path.realpath(argv[0])
    if kind == "bwrap":
        cmd = [*jail_prefix]
        for path in ro_binds:
            cmd += ["--ro-bind", path, path]
        cmd += ["--bind", tmp, tmp, "--chdir", tmp, "--", resolved_exe, *argv[1:]]
        return cmd
    if kind == "nsjail":
        cmd = [*jail_prefix]
        for path in ro_binds:
            cmd += ["-R", path]
        cmd += ["-B", f"{tmp}:{tmp}", "--cwd", tmp]
        for fd in pass_fds:
            cmd += ["--pass_fd", str(int(fd))]
        for key, value in sorted(env.items()):
            cmd += ["-E", f"{key}={value}"]
        cmd += ["--", resolved_exe, *argv[1:]]
        return cmd
    return list(argv)


def _probe_jail(kind: str, prefix: list[str]) -> tuple[bool, str]:
    """Probe once (cached) whether a jailer can actually spawn a child process.

    Catches environments where ``bwrap`` / ``nsjail`` is installed on ``PATH``
    but blocked by AppArmor, Docker seccomp, or restricted user namespaces.
    """
    cache_key = (kind, tuple(prefix))
    with _JAIL_PROBE_LOCK:
        cached = _JAIL_PROBE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    tmp = tempfile.mkdtemp(prefix="edsandbox-probe-")
    try:
        probe_env = {"PATH": _RUNTIME_PATH, "HOME": tmp, "TMPDIR": tmp}
        probe_argv = _build_jail_command(
            kind,
            prefix,
            [sys.executable, "-I", "-c", "pass"],
            tmp=tmp,
            env=probe_env,
        )
        proc = subprocess.run(
            probe_argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            cwd=tmp,
            env=probe_env,
            text=True,
            errors="replace",
            timeout=5.0,
            check=False,
        )
        if proc.returncode == 0:
            result = (True, "")
        else:
            detail = (proc.stderr or proc.stdout or f"exit code {proc.returncode}").strip()
            result = (False, detail.splitlines()[0][:200] if detail else f"exit code {proc.returncode}")
    except (OSError, subprocess.TimeoutExpired) as exc:
        result = (False, f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    with _JAIL_PROBE_LOCK:
        _JAIL_PROBE_CACHE[cache_key] = result
    return result


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
        _file_size_mb (int): ``RLIMIT_FSIZE`` and tmpfs cap (MiB); 0 disables it.
        _max_procs (int | None): Optional ``RLIMIT_NPROC`` / ``--rlimit_nproc`` cap.
        _max_output (int): Character cap applied independently to stdout/stderr.
        _network (bool): Allow network egress (only enforceable under a jail).
        _use_jail (str): Validated jail policy string.
        _jail_kind (str | None): Resolved jailer name (``"bwrap"``/``"nsjail"``/``None``).
        _jail (list[str] | None): Resolved bwrap/nsjail wrapper argv prefix, or None.
    """

    def __init__(
        self,
        *,
        timeout: float = 10.0,
        cpu_seconds: int | None = None,
        memory_mb: int = 2048,
        file_size_mb: int = 64,
        max_procs: int | None = None,
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
            file_size_mb: ``RLIMIT_FSIZE`` and ``/tmp`` tmpfs cap in MiB (0 to
                disable). Defaults to 64 MiB.
            max_procs: Optional process-count limit. Applied via
                ``--rlimit_nproc`` in ``nsjail`` (which defaults to 64 when
                ``None``), and via ``RLIMIT_NPROC`` in ``preexec_fn`` only when
                explicitly set to a positive integer (because Linux accounts
                ``RLIMIT_NPROC`` per host UID across all threads). Pass ``0``
                to disable ``--rlimit_nproc`` in ``nsjail``.
            max_output: Character cap for each captured stream. Excess output is
                drained and discarded without buffering it.
            network: Permit network egress. Only actually enforced when a jail
                (bwrap/nsjail) is used; without one, egress cannot be blocked.
            use_jail: ``"auto"`` (wrap with bwrap/nsjail if present and usable,
                else warn and fall back to plain subprocess), ``"required"``
                (raise if no usable jailer is available), ``"bwrap"`` /
                ``"nsjail"`` (require that specific jailer), or ``"off"``
                (never wrap).

        Raises:
            ValueError: If ``use_jail`` is unrecognized or ``max_procs`` is negative.
            RuntimeError: If an explicit jail mode is requested but unavailable
                or fails its usability probe.
        """
        use_jail = validate_use_jail(use_jail)
        if max_procs is not None:
            max_procs = int(max_procs)
            if max_procs < 0:
                raise ValueError(f"max_procs must be >= 0, got {max_procs}.")

        self._timeout = float(timeout)
        # ceil, not truncation: a fractional timeout such as 1.5 would otherwise get
        # a 2-second CPU cap, leaving the rlimit backstop only half a second behind
        # the wall clock. A busy child then trips RLIMIT_CPU before a contended
        # parent thread observes its own timeout, and the kill is reported as a
        # plain non-zero exit instead of a timeout.
        self._cpu_seconds = int(cpu_seconds) if cpu_seconds is not None else math.ceil(self._timeout) + 1
        self._memory_mb = max(0, int(memory_mb))
        self._file_size_mb = max(0, int(file_size_mb))
        self._max_procs = max_procs
        self._max_output = max(0, int(max_output))
        self._network = bool(network)
        self._use_jail = use_jail
        self._jail_kind: str | None = None
        self._jail_owns_limits = False
        self._jail = self._resolve_jail(use_jail)

    def _runtime_ro_dirs(self) -> list[str]:
        """Return minimal read-only host paths required for Python/shell execution."""
        # Start with an empty root, not a read-only view of the host. System
        # executables/libraries and the base Python installation are sufficient
        # for isolated Python and shell execution. Never mount the venv prefix:
        # it may live alongside or inside the trainer repository.
        runtime_dirs = ["/usr", "/bin", "/sbin", "/lib", "/lib64"]
        base_prefix = os.path.realpath(sys.base_prefix)
        if not any(base_prefix == path or base_prefix.startswith(path + "/") for path in runtime_dirs):
            if base_prefix == "/":
                raise RuntimeError("Cannot safely expose a Python runtime rooted at /.")
            runtime_dirs.append(base_prefix)
        mounts = [path for path in runtime_dirs if os.path.exists(path)]
        if self._network:
            # DNS needs only these configuration files, never all of /etc.
            for path in ("/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf"):
                if os.path.exists(path):
                    mounts.append(path)
        return mounts

    def _build_bwrap_prefix(self, bwrap: str) -> list[str]:
        """Build the static ``bwrap`` argv prefix (excluding per-run binds)."""
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
        for path in self._runtime_ro_dirs():
            argv += ["--ro-bind", path, path]
        argv += ["--dev", "/dev", "--proc", "/proc"]
        if self._file_size_mb > 0:
            argv += ["--size", str(self._file_size_mb * 1024 * 1024), "--tmpfs", "/tmp"]
        else:
            argv += ["--tmpfs", "/tmp"]
        if not self._network:
            argv.append("--unshare-net")
        return argv

    def _build_nsjail_prefix(self, nsjail: str) -> list[str]:
        """Build the static ``nsjail`` argv prefix (excluding per-run mounts/env)."""
        argv = [nsjail, "-Mo", "--really_quiet"]
        # nsjail isolates the network namespace by default; --disable_clone_newnet
        # shares the host network, so it is only passed when egress is allowed.
        if self._network:
            argv.append("--disable_clone_newnet")
        for path in self._runtime_ro_dirs():
            argv += ["-R", path]
        for dev_path in ("/dev/null", "/dev/urandom"):
            if os.path.exists(dev_path):
                argv += ["-R", dev_path]
        if self._file_size_mb > 0:
            argv += ["--mount", f"none:/tmp:tmpfs:size={self._file_size_mb * 1024 * 1024}"]
        else:
            argv += ["-T", "/tmp"]
        # nsjail resets the child's rlimits to its own defaults, overriding any
        # inherited ones, so the sandbox caps are passed as nsjail flags.
        argv += ["--rlimit_cpu", str(self._cpu_seconds)]
        argv += ["--rlimit_as", str(self._memory_mb) if self._memory_mb > 0 else "inf"]
        argv += ["--rlimit_fsize", str(self._file_size_mb) if self._file_size_mb > 0 else "inf"]
        if self._max_procs is None:
            argv += ["--rlimit_nproc", "64"]
        else:
            argv += ["--rlimit_nproc", str(self._max_procs) if self._max_procs > 0 else "inf"]
        return argv

    def _resolve_jail(self, use_jail: str) -> list[str] | None:
        """Resolve and probe a bwrap/nsjail wrapper argv for the current settings.

        Args:
            use_jail: Validated ``use_jail`` policy.

        Returns:
            The wrapper argv prefix, or ``None`` for plain subprocess.

        Raises:
            ValueError: If ``use_jail`` is invalid.
            RuntimeError: If an explicit jail policy was requested but no usable
                jailer is available.
        """
        use_jail = validate_use_jail(use_jail)
        if use_jail == "off":
            return None
        if not _POSIX:
            if use_jail == "auto":
                warnings.warn(
                    "Sandbox jail unavailable on non-POSIX platform; falling back to unjailed subprocess execution. "
                    "Pass use_jail='off' to silence this warning.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return None
            raise RuntimeError(f"Sandbox jail {use_jail!r} required but unavailable on this platform.")

        candidates = ("bwrap", "nsjail") if use_jail in ("auto", "required") else (use_jail,)
        reasons: list[str] = []
        for kind in candidates:
            binary = shutil.which(kind)
            if not binary:
                reasons.append(f"{kind} not found on PATH")
                continue
            prefix = self._build_bwrap_prefix(binary) if kind == "bwrap" else self._build_nsjail_prefix(binary)
            usable, reason = _probe_jail(kind, prefix)
            if usable:
                self._jail_kind = kind
                self._jail_owns_limits = kind == "nsjail"
                return prefix
            reasons.append(f"{kind} probe failed: {reason}")

        detail = "; ".join(reasons) or "no jailer available"
        if use_jail == "auto":
            warnings.warn(
                f"Sandbox jail unavailable ({detail}); falling back to unjailed host subprocess execution. "
                "Pass use_jail='off' to silence this warning or use_jail='required' to fail closed.",
                RuntimeWarning,
                stacklevel=3,
            )
            return None
        raise RuntimeError(f"Sandbox jail {use_jail!r} required but unavailable ({detail}).")

    def _build_command(
        self,
        argv: list[str],
        *,
        tmp: str,
        env: dict[str, str],
        ro_binds: tp.Sequence[str] = (),
        pass_fds: tp.Sequence[int] = (),
    ) -> list[str]:
        """Build the final command argv for execution inside ``tmp``."""
        return _build_jail_command(
            self._jail_kind,
            self._jail,
            argv,
            tmp=tmp,
            env=env,
            ro_binds=ro_binds,
            pass_fds=pass_fds,
        )

    def _limits(self) -> tp.Callable[[], None] | None:
        """Build the ``preexec_fn`` that applies rlimits in the child.

        Returns:
            A no-arg callable for ``subprocess`` ``preexec_fn``, or ``None`` on
            non-POSIX platforms.
        """
        if not _POSIX or resource is None:
            return None
        cpu = self._cpu_seconds
        mem_bytes = self._memory_mb * 1024 * 1024 if self._memory_mb > 0 else 0
        fsize_bytes = self._file_size_mb * 1024 * 1024 if self._file_size_mb > 0 else 0
        max_procs = self._max_procs
        res_mod = resource

        def _apply() -> None:
            """Apply CPU / address-space / file-size / proc rlimits in the child."""
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
            res_mod.setrlimit(res_mod.RLIMIT_CPU, (cpu, cpu + 1))
            if mem_bytes:
                try:
                    res_mod.setrlimit(res_mod.RLIMIT_AS, (mem_bytes, mem_bytes))
                except (ValueError, OSError):
                    pass
            if fsize_bytes and hasattr(res_mod, "RLIMIT_FSIZE"):
                try:
                    res_mod.setrlimit(res_mod.RLIMIT_FSIZE, (fsize_bytes, fsize_bytes))
                except (ValueError, OSError):
                    pass
            if max_procs is not None and max_procs > 0 and hasattr(res_mod, "RLIMIT_NPROC"):
                try:
                    _, hard = res_mod.getrlimit(res_mod.RLIMIT_NPROC)
                    cap = max_procs if hard == res_mod.RLIM_INFINITY else min(max_procs, hard)
                    res_mod.setrlimit(res_mod.RLIMIT_NPROC, (cap, cap))
                except (ValueError, OSError):
                    pass

        return _apply

    def _run(
        self,
        argv: list[str] | tp.Callable[[str, str | None], list[str]],
        *,
        stdin: str | None = None,
        extra_env: dict[str, str] | tp.Callable[[str, str | None], dict[str, str]] | None = None,
        ro_files: dict[str, str] | None = None,
        work_files: dict[str, str] | None = None,
        pass_fds: tuple[int, ...] = (),
        on_spawn: tp.Callable[[subprocess.Popen], None] | None = None,
        on_after_wait: tp.Callable[[str], None] | None = None,
    ) -> SandboxResult:
        """Execute ``argv`` in the sandbox and capture its result.

        Runs in a new session (process group) so a timeout can kill any
        grandchildren; applies the jail wrapper and rlimits; enforces the
        wall-clock timeout with a group SIGKILL.

        Args:
            argv: Program + arguments to execute, or a callable ``(tmp, ro_dir) -> list[str]``.
            stdin: Optional string piped to the child's standard input.
            extra_env: Optional extra environment variables (or callable
                ``(tmp, ro_dir) -> dict[str, str]``) merged into the scrubbed
                child environment.
            ro_files: Optional mapping of filename -> contents staged in a
                separate read-only temp directory bound into jails.
            work_files: Optional mapping of filename -> contents staged inside
                the writable temp working directory ``tmp``.
            pass_fds: Optional inherited file descriptors forwarded to the child.
            on_spawn: Optional callback invoked immediately after ``Popen`` succeeds.
            on_after_wait: Optional callback invoked with ``tmp`` right after
                the child exits and before ``tmp`` is removed.

        Returns:
            A :class:`SandboxResult`.
        """
        env = {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}
        env["PATH"] = _RUNTIME_PATH
        tmp = tempfile.mkdtemp(prefix="edsandbox-")
        ro_dir: str | None = None
        env.update(HOME=tmp, TMPDIR=tmp)

        try:
            ro_binds: list[str] = []
            if ro_files:
                ro_dir = tempfile.mkdtemp(prefix="edsandbox-ro-")
                for name, content in ro_files.items():
                    file_path = os.path.join(ro_dir, name)
                    with open(file_path, "w", encoding="utf-8") as handle:
                        handle.write(content)
                ro_binds.append(ro_dir)

            if work_files:
                for name, content in work_files.items():
                    file_path = os.path.join(tmp, name)
                    with open(file_path, "w", encoding="utf-8") as handle:
                        handle.write(content)

            if extra_env is not None:
                env.update(extra_env(tmp, ro_dir) if callable(extra_env) else extra_env)

            resolved_argv = argv(tmp, ro_dir) if callable(argv) else list(argv)
            full_argv = self._build_command(resolved_argv, tmp=tmp, env=env, ro_binds=ro_binds, pass_fds=pass_fds)
            popen_kwargs: dict[str, tp.Any] = {
                "stdin": subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "cwd": tmp,
                "env": env,
                "text": True,
                "errors": "replace",
                # rlimits survive exec, so they bound a bwrap-wrapped child too; only
                # nsjail overrides them, and it receives the caps as flags instead.
                "preexec_fn": None if self._jail_owns_limits else self._limits(),
                "start_new_session": _POSIX,
            }
            if _POSIX and pass_fds:
                popen_kwargs["pass_fds"] = pass_fds
            proc = subprocess.Popen(full_argv, **popen_kwargs)
            captured = False
            try:
                if on_spawn is not None:
                    on_spawn(proc)
                captured = True
                out, err, timed_out = self._capture(proc, stdin)
            except BaseException:
                if not captured:
                    self._cleanup_process(proc)
                raise
            # RLIMIT_CPU can win the race against the wall-clock deadline.
            # Jailed/shell-wrapped children may exit with 128 + SIGXCPU instead of -SIGXCPU.
            timed_out |= proc.returncode in _CPU_BUDGET_RETURNCODES
            if on_after_wait is not None:
                on_after_wait(tmp)
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
            if ro_dir is not None:
                shutil.rmtree(ro_dir, ignore_errors=True)

    def _cleanup_process(
        self,
        proc: subprocess.Popen,
        started_workers: tp.Sequence[threading.Thread] = (),
    ) -> None:
        """Kill the child process group, reap ``proc``, join started threads, and close pipes."""
        self._kill_group(proc)
        if proc.poll() is None:
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                    proc.wait()
                except OSError:
                    pass
        for worker in started_workers:
            if worker.is_alive():
                worker.join(timeout=0.1)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None and not stream.closed:
                try:
                    stream.close()
                except (OSError, ValueError, RuntimeError):
                    pass

    def _capture(self, proc: subprocess.Popen, stdin: str | None) -> tuple[str, str, bool]:
        """Drain both pipes concurrently, retaining at most one prefix per stream.

        Reader and optional stdin-writer threads share the execution deadline.
        This also times out descendants holding pipes open after the child exits.
        Excess output is discarded, including after a timeout kill, rather than
        accumulated by ``communicate()``.
        """
        buffers: list[list[str]] = [[], []]
        started: list[threading.Thread] = []
        timed_out = False

        def read_stream(stream: tp.TextIO, parts: list[str]) -> None:
            remaining = self._max_output
            try:
                with stream:
                    while chunk := stream.read(_CAPTURE_CHUNK_SIZE):
                        if remaining:
                            prefix = chunk[:remaining]
                            parts.append(prefix)
                            remaining -= len(prefix)
            except (OSError, ValueError):
                pass

        def write_stdin() -> None:
            try:
                with proc.stdin:
                    proc.stdin.write(stdin)
                    proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass  # The child may exit without consuming its input (including on close()).

        try:
            workers = [
                threading.Thread(target=read_stream, args=(stream, parts), daemon=True)
                for stream, parts in zip((proc.stdout, proc.stderr), buffers, strict=True)
            ]
            if stdin is not None:
                workers.append(threading.Thread(target=write_stdin, daemon=True))
            deadline = time.monotonic() + self._timeout
            for worker in workers:
                worker.start()
                started.append(worker)
            try:
                proc.wait(timeout=max(0.0, deadline - time.monotonic()))
                for worker in started:
                    worker.join(timeout=max(0.0, deadline - time.monotonic()))
                timed_out = any(worker.is_alive() for worker in started)
            except subprocess.TimeoutExpired:
                timed_out = True
        finally:
            self._cleanup_process(proc, started)
        return "".join(buffers[0]), "".join(buffers[1]), timed_out

    @staticmethod
    def _kill_group(proc: subprocess.Popen) -> None:
        """SIGKILL the child's whole process group (best-effort).

        Args:
            proc: The running child process.
        """
        if not _POSIX:
            if proc.poll() is None:
                proc.kill()
            return
        try:
            # start_new_session makes the child's PID its process-group ID.
            # Keep using that ID if wait() already reaped the group leader while
            # grandchildren still hold stdout/stderr open.
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            if proc.poll() is None:
                proc.kill()

    def run_python(self, code: str, *, stdin: str | None = None) -> SandboxResult:
        """Run a Python snippet in the sandbox via a temp script file.

        Using a temporary file avoids Linux ``MAX_ARG_STRLEN`` (128 KiB)
        ``E2BIG`` failures that occur with ``python -c``.

        Args:
            code: Python source to execute.
            stdin: Optional standard input for the snippet.

        Returns:
            A :class:`SandboxResult` (``ok`` iff it exited 0 without timing out).
        """
        return self._run(
            lambda tmp, ro_dir: [sys.executable, "-I", os.path.join(ro_dir or tmp, "script.py")],
            stdin=stdin,
            ro_files={"script.py": code},
        )

    def run_python_with_sentinel(
        self,
        code: str,
        test_code: str,
        *,
        stdin: str | None = None,
    ) -> tuple[SandboxResult, bool]:
        """Run untrusted ``code`` followed by ``test_code`` with out-of-band verification.

        Never places any sentinel secret or path in the child's environment
        (preventing ``/proc/self/environ`` leaks), argv, temp directory, or
        untrusted execution frames. Instead, on POSIX the parent drives a
        3-phase HMAC-SHA256 authenticated protocol over two private inherited
        pipes:

        1. Send a 256-bit per-run HMAC key and only ``code`` (``test_code`` is
           not yet in the child's memory), then wait for a payload-bound
           phase-1 HMAC.
        2. Encrypt ``test_code`` under the per-run key (CTR + HMAC tag), send
           it only after phase 1 authenticates, and wait for a tag-bound
           phase-2 HMAC.
        3. Generate a fresh 256-bit cryptographic challenge *after* phase 2
           completes, send it to the child, and verify an exact phase-3 HMAC +
           EOF + clean zero exit status.

        Untrusted phases execute on pre-spawned pure-C worker threads (zero
        Python caller frames, zero post-Phase-1 GC-tracked allocations on the
        harness thread) under an unconditional ``sys.addaudithook`` blocking
        harness-frame, GC, ``ctypes``, and ``/proc`` introspection.

        Args:
            code: Untrusted candidate Python source.
            test_code: Test/assertion Python source executed after ``code``.
            stdin: Optional standard input.

        Returns:
            ``(result, passed)`` where ``passed`` is True iff ``result.ok`` and
            the post-test cryptographic challenge-response succeeded.
        """
        if not _POSIX:  # pragma: no cover - non-POSIX fallback
            code_bytes = code.encode("utf-8")
            test_bytes = test_code.encode("utf-8")
            combined = code_bytes.decode("utf-8") + "\n\n" + test_bytes.decode("utf-8")
            res = self.run_python(combined, stdin=stdin)
            return res, bool(res.ok)

        code_bytes = code.encode("utf-8")
        test_bytes = test_code.encode("utf-8")
        cmd_r, cmd_w = os.pipe()
        try:
            ack_r, ack_w = os.pipe()
        except BaseException:
            for fd in (cmd_r, cmd_w):
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise
        open_fds: dict[str, int | None] = {
            "cmd_r": cmd_r,
            "cmd_w": cmd_w,
            "ack_r": ack_r,
            "ack_w": ack_w,
        }
        fd_lock = threading.Lock()

        def _close_fd(key: str) -> None:
            with fd_lock:
                fd = open_fds[key]
                open_fds[key] = None
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass

        sentinel_matched = False
        protocol_thread: threading.Thread | None = None

        def _write_all(fd: int, data: bytes) -> None:
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("Short write on sandbox protocol pipe.")
                view = view[written:]

        def _read_exact(fd: int, n: int) -> bytes:
            chunks: list[bytes] = []
            remaining = n
            while remaining > 0:
                chunk = os.read(fd, remaining)
                if not chunk:
                    raise EOFError("Sandbox protocol EOF.")
                chunks.append(chunk)
                remaining -= len(chunk)
            return b"".join(chunks)

        def _xor_ctr(key: bytes, nonce: bytes, data: bytes) -> bytes:
            out = bytearray()
            offset = 0
            block_idx = 0
            total = len(data)
            while offset < total:
                end = min(offset + 32, total)
                size = end - offset
                keystream = hmac.digest(key, b"k2:" + nonce + block_idx.to_bytes(4, "big"), hashlib.sha256)
                xored = int.from_bytes(data[offset:end], "big") ^ int.from_bytes(keystream[:size], "big")
                out.extend(xored.to_bytes(size, "big"))
                offset = end
                block_idx += 1
            return bytes(out)

        def _drive_protocol() -> None:
            nonlocal sentinel_matched
            try:
                hmac_key = secrets.token_bytes(32)
                _write_all(cmd_w, hmac_key)

                nonce1 = secrets.token_bytes(16)
                code_tag = hmac.digest(hmac_key, b"c1:" + code_bytes, hashlib.sha256)
                _write_all(cmd_w, nonce1 + struct.pack("!I", len(code_bytes)) + code_bytes)
                expected_mac1 = hmac.digest(hmac_key, b"p1:" + nonce1 + code_tag, hashlib.sha256)
                if not secrets.compare_digest(_read_exact(ack_r, 32), expected_mac1):
                    return

                nonce2 = secrets.token_bytes(16)
                ct = _xor_ctr(hmac_key, nonce2, test_bytes)
                tag2 = hmac.digest(hmac_key, b"t2:" + nonce2 + ct, hashlib.sha256)
                _write_all(cmd_w, nonce2 + struct.pack("!I", len(ct)) + ct + tag2)
                expected_mac2 = hmac.digest(hmac_key, b"p2:" + nonce2 + tag2, hashlib.sha256)
                if not secrets.compare_digest(_read_exact(ack_r, 32), expected_mac2):
                    return

                challenge = secrets.token_bytes(32)
                _write_all(cmd_w, challenge)
                _close_fd("cmd_w")
                expected_mac3 = hmac.digest(hmac_key, b"p3:" + challenge, hashlib.sha256)
                reply = bytearray()
                while len(reply) < 33:
                    chunk = os.read(ack_r, 33 - len(reply))
                    if not chunk:
                        break
                    reply.extend(chunk)
                sentinel_matched = len(reply) == 32 and secrets.compare_digest(bytes(reply), expected_mac3)
            except (OSError, EOFError, ValueError):
                sentinel_matched = False
            finally:
                _close_fd("cmd_w")
                _close_fd("ack_r")

        def _on_spawn(_proc: subprocess.Popen) -> None:
            nonlocal protocol_thread
            # Close child ends in parent immediately so EOF reflects child exit.
            _close_fd("cmd_r")
            _close_fd("ack_w")
            protocol_thread = threading.Thread(target=_drive_protocol, daemon=True)
            protocol_thread.start()

        def _on_after_wait(_tmp: str) -> None:
            if protocol_thread is not None:
                protocol_thread.join(timeout=2.0)

        try:
            result = self._run(
                lambda tmp, ro_dir: [
                    sys.executable,
                    "-I",
                    os.path.join(ro_dir or tmp, "harness.py"),
                    str(cmd_r),
                    str(ack_w),
                ],
                stdin=stdin,
                ro_files={"harness.py": _SENTINEL_HARNESS_SOURCE},
                pass_fds=(cmd_r, ack_w),
                on_spawn=_on_spawn,
                on_after_wait=_on_after_wait,
            )
        finally:
            _close_fd("cmd_r")
            _close_fd("ack_w")
            _close_fd("cmd_w")
            if protocol_thread is not None and protocol_thread.is_alive():
                protocol_thread.join(timeout=0.5)
            _close_fd("ack_r")
        return result, bool(result.ok and sentinel_matched)

    def run_bash(self, command: str, *, shell: str = "/bin/bash") -> SandboxResult:
        """Run a shell command in the sandbox.

        Args:
            command: Shell command line.
            shell: Shell interpreter to use.

        Returns:
            A :class:`SandboxResult`.
        """
        if len(command.encode("utf-8", errors="replace")) > 65536:
            return self._run(
                lambda tmp, ro_dir: [shell, os.path.join(ro_dir or tmp, "script.sh")],
                ro_files={"script.sh": command},
            )
        return self._run([shell, "-c", command])
