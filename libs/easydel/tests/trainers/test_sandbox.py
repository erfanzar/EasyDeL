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

"""Tests for the hardened out-of-process sandbox used by RLVR / agentic tools.

Pure-Python, POSIX. Validates the guarantees that in-process ``exec`` +
``signal.alarm`` could not give: reliable off-main-thread timeouts, crash
containment, and environment/working-directory isolation.
"""

import os
import shutil
import signal
import sys
import threading
import time
import tracemalloc

import pytest
from easydel.trainers._shared.sandbox import Sandbox

pytestmark = pytest.mark.skipif(os.name != "posix", reason="sandbox relies on POSIX process groups / rlimits")


def test_normal_python_run_captures_stdout():
    res = Sandbox(timeout=10.0).run_python("print(sum(range(101)))")
    assert res.ok is True
    assert res.timed_out is False
    assert res.returncode == 0
    assert res.stdout.strip() == "5050"


@pytest.mark.parametrize("timed_out", [False, True])
def test_output_capture_memory_is_bounded_even_during_timeout(timed_out):
    """Large child streams must never be fully buffered in the trainer process."""
    cap = 257
    code = "import os, time\nfor _ in range(8192):\n    os.write(1, b'x' * 4096)\n    os.write(2, b'y' * 4096)\n"
    if timed_out:
        code += "time.sleep(30)\n"
    # The child emits 32 MiB on EACH pipe using small reusable buffers. The old
    # communicate()-then-slice implementation retains tens of MiB in this parent;
    # the public result must be the same prefix with <2 MiB capture overhead.
    tracemalloc.start()
    try:
        res = Sandbox(timeout=2.0 if timed_out else 10.0, max_output=cap, use_jail="off").run_python(code)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert res.stdout == "x" * cap
    assert res.stderr == "y" * cap
    assert res.timed_out is timed_out
    assert res.ok is not timed_out
    assert peak < 2 * 1024 * 1024


@pytest.mark.parametrize("cap", [0, 7, 8192])
def test_capture_preserves_text_prefix_and_handles_concurrent_stdin(cap):
    # Larger than a pipe buffer: writing input before draining output deadlocks.
    data = "é\n" * 65536
    res = Sandbox(timeout=10.0, max_output=cap, use_jail="off").run_python(
        "import sys\nfor line in sys.stdin:\n    sys.stdout.write(line)\n    sys.stderr.write(line)", stdin=data
    )
    assert res.ok, res.stderr
    assert res.stdout == data[:cap]
    assert res.stderr == data[:cap]


def test_exception_is_contained_not_ok():
    res = Sandbox(timeout=10.0).run_python("raise ValueError('boom')")
    assert res.ok is False
    assert res.returncode not in (0, None)
    assert "ValueError" in res.stderr


def test_hard_crash_does_not_kill_caller():
    # os._exit bypasses Python's exception handling entirely — in-process exec
    # would take the trainer down; here it is confined to the child.
    res = Sandbox(timeout=10.0).run_python("import os; os._exit(42)")
    assert res.ok is False
    assert res.returncode == 42
    # reaching this line proves the caller (this test process) survived.


def test_infinite_loop_times_out():
    res = Sandbox(timeout=1.5).run_python("while True:\n    pass")
    assert res.timed_out is True
    assert res.ok is False


def test_timeout_kills_grandchildren_after_group_leader_exits(tmp_path):
    # The rlimit preexec_fn forces a full fork() of the caller; deep into a long test session that alone can
    # take a few hundred ms, so the budget leaves room for the child to start and print before the deadline.
    marker = tmp_path / "escaped-timeout"
    grandchild = f"import time; time.sleep(4); open({str(marker)!r}, 'w').write('still running')"
    res = Sandbox(timeout=1.5, use_jail="off").run_python(
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-I', '-c', {grandchild!r}])\n"
        "print('spawned', flush=True)\n"
        "time.sleep(30)"
    )
    assert res.timed_out is True
    assert res.ok is False
    assert res.stdout.strip() == "spawned"
    time.sleep(4.2)
    assert not marker.exists()


def test_cpu_rlimit_backstop_is_reported_as_a_timeout():
    """A child killed by RLIMIT_CPU exhausted its time budget, so it must say so.

    The backstop wins the race against the wall clock whenever the calling thread
    is slow to observe its own deadline (contended interpreter, GIL pressure). It
    used to surface as a plain non-zero exit, which turns an infinite loop into a
    generic failure for anything scoring sandbox results. Forced deterministically
    here by leaving the wall clock far out of reach.
    """
    res = Sandbox(timeout=30.0, cpu_seconds=1).run_python("while True:\n    pass")

    assert res.timed_out is True
    assert res.ok is False
    assert res.returncode == -signal.SIGXCPU


@pytest.mark.parametrize("disposition", ["ignored", "blocked"])
def test_cpu_rlimit_is_reported_when_the_caller_ignores_or_blocks_sigxcpu(disposition):
    """Ignored/blocked signals survive fork+exec; the child must still die by SIGXCPU, not the hard-limit SIGKILL."""
    previous_handler = signal.getsignal(signal.SIGXCPU)
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    try:
        if disposition == "ignored":
            signal.signal(signal.SIGXCPU, signal.SIG_IGN)
        else:
            signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGXCPU})
        res = Sandbox(timeout=30.0, cpu_seconds=1, use_jail="off").run_python("while True:\n    pass")
    finally:
        signal.signal(signal.SIGXCPU, previous_handler)
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

    assert res.timed_out is True
    assert res.returncode == -signal.SIGXCPU


def test_default_cpu_budget_clears_a_fractional_wall_clock():
    """The CPU cap must sit above the wall clock, per the documented ceil+1."""
    assert Sandbox(timeout=1.5)._cpu_seconds == 3
    assert Sandbox(timeout=2.0)._cpu_seconds == 3
    assert Sandbox(timeout=10.0)._cpu_seconds == 11


@pytest.mark.parametrize("network", [False, True])
def test_nsjail_network_flag_matches_policy(monkeypatch, network):
    """nsjail isolates the network by default; --disable_clone_newnet shares the host's."""
    import easydel.trainers._shared.sandbox as sandbox_mod

    monkeypatch.setattr(sandbox_mod.shutil, "which", lambda name: "/usr/bin/nsjail" if name == "nsjail" else None)
    box = Sandbox(timeout=2.0, memory_mb=512, network=network, use_jail="required")
    assert box._jail[0] == "/usr/bin/nsjail"
    assert box._jail[-1] == "--"
    assert ("--disable_clone_newnet" in box._jail) is network
    # nsjail resets rlimits, so the caps travel as flags and no preexec_fn is used.
    assert box._jail[box._jail.index("--rlimit_cpu") + 1] == "3"
    assert box._jail[box._jail.index("--rlimit_as") + 1] == "512"
    assert box._jail_owns_limits is True


@pytest.mark.parametrize("network", [False, True])
def test_bwrap_isolates_host_paths_and_pids_and_keeps_rlimits(monkeypatch, network):
    """Required bwrap exposes runtime mounts, not host root/home/repo or PIDs."""
    import easydel.trainers._shared.sandbox as sandbox_mod

    monkeypatch.setattr(sandbox_mod.shutil, "which", lambda name: "/usr/bin/bwrap" if name == "bwrap" else None)
    box = Sandbox(timeout=2.0, network=network, use_jail="required")
    assert ("--unshare-net" in box._jail) is not network
    for flag in ("--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--die-with-parent"):
        assert flag in box._jail
    assert box._jail[box._jail.index("--cap-drop") + 1] == "ALL"
    sources = [box._jail[i + 1] for i, arg in enumerate(box._jail) if arg == "--ro-bind"]
    allowed = {"/usr", "/bin", "/sbin", "/lib", "/lib64", os.path.realpath(sys.base_prefix)}
    if network:
        allowed.update({"/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf"})
    assert set(sources) <= allowed
    assert "/" not in sources
    assert os.getcwd() not in sources
    assert os.path.expanduser("~") not in sources
    assert box._jail_owns_limits is False
    assert callable(box._limits())


@pytest.mark.parametrize("network", [False, True])
def test_required_bwrap_hides_host_sentinel_and_pids(tmp_path, network):
    """Exercise real namespace isolation when bubblewrap is usable on the host."""
    if not shutil.which("bwrap"):
        pytest.skip("bubblewrap is not installed")
    box = Sandbox(timeout=10.0, network=network, use_jail="required")
    probe = box.run_python("print('ready')")
    if not probe.ok and any(
        reason in probe.stderr for reason in ("Operation not permitted", "No permissions to create", "Permission denied")
    ):
        pytest.skip(f"bubblewrap cannot create namespaces here: {probe.stderr.strip()}")
    assert probe.ok, probe.stderr
    sentinel = tmp_path / "host-secret"
    sentinel.write_text("not for generated code")
    res = box.run_python(
        f"import os; assert not os.path.exists({str(sentinel)!r}); "
        f"assert not os.path.exists('/proc/{os.getpid()}'); "
        "assert os.environ['HOME'] == os.getcwd(); print('isolated')"
    )
    assert res.ok, res.stderr
    assert res.stdout.strip() == "isolated"


def test_timeout_works_off_main_thread():
    # The old signal.alarm approach only fires on the main thread; the sandbox
    # uses subprocess + process-group kill, so it must time out from a worker
    # thread too.
    box = {}

    def worker():
        box["res"] = Sandbox(timeout=1.5).run_python("while True:\n    pass")

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=15)
    assert not t.is_alive()
    assert box["res"].timed_out is True


def test_environment_is_scrubbed():
    os.environ["EDEL_FAKE_SECRET"] = "leak-me"
    try:
        res = Sandbox(timeout=10.0).run_python("import os; print(os.environ.get('EDEL_FAKE_SECRET', 'ABSENT'))")
    finally:
        del os.environ["EDEL_FAKE_SECRET"]
    assert res.stdout.strip() == "ABSENT"


def test_host_home_path_and_tmpdir_are_not_inherited(monkeypatch):
    monkeypatch.setenv("HOME", "/host-private-home")
    monkeypatch.setenv("PATH", "/host-private-bin")
    monkeypatch.setenv("TMPDIR", "/host-private-temp")
    res = Sandbox(timeout=10.0, use_jail="off").run_python(
        "import os; "
        "assert os.environ['HOME'] == os.environ['TMPDIR'] == os.getcwd(); "
        "assert 'host-private' not in os.environ['PATH']; print('scrubbed')"
    )
    assert res.ok, res.stderr
    assert res.stdout.strip() == "scrubbed"


def test_cwd_is_a_throwaway_tempdir():
    repo_cwd = os.getcwd()
    res = Sandbox(timeout=10.0).run_python("import os; print(os.getcwd())")
    child_cwd = res.stdout.strip()
    assert child_cwd != repo_cwd
    assert not os.path.exists(child_cwd)  # removed after the run


def test_run_bash():
    res = Sandbox(timeout=10.0).run_bash("echo hello | wc -c")
    assert res.ok is True
    assert res.stdout.strip() == "6"


def test_code_verifier_uses_sandbox():
    from easydel.trainers.rlvr_trainer import CodeVerifier

    v = CodeVerifier(timeout=10.0)
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    bad = "```python\ndef add(a, b):\n    return a - b\n```"
    batch = {"tests": ["assert add(2, 3) == 5", "assert add(2, 3) == 5"]}
    assert v(completions=[good, bad], batch=batch) == [1.0, 0.0]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
