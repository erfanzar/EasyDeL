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
import threading

import pytest
from easydel.trainers._shared.sandbox import Sandbox

pytestmark = pytest.mark.skipif(os.name != "posix", reason="sandbox relies on POSIX process groups / rlimits")


def test_normal_python_run_captures_stdout():
    res = Sandbox(timeout=10.0).run_python("print(sum(range(101)))")
    assert res.ok is True
    assert res.timed_out is False
    assert res.returncode == 0
    assert res.stdout.strip() == "5050"


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
