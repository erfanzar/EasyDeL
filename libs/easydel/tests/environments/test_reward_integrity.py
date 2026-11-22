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

"""Reward-integrity tests: environments must not hand out reward the policy did not earn."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time

import pytest
from easydel.environments import (
    CodeJudgeEnvironment,
    ContainerConfig,
    ContainerSandboxSession,
    ExecResult,
    HarnessEnvironment,
    HarnessPreset,
    HarnessTITOGateway,
    KiriRuntime,
    SWERepoEnvironment,
    SyntheticTaskStream,
    TerminalEnvironment,
    make_environment_reward_fn,
)

pytestmark = pytest.mark.cpu_ok

_NO_JAIL = ContainerConfig(use_jail="off")
_HAS_GIT = shutil.which("git") is not None


def _pytest_available() -> bool:
    return subprocess.run([sys.executable, "-c", "import pytest"], capture_output=True).returncode == 0


# ---------------------------------------------------------------------------------------------------------------------
# code_judge unit-test mode
# ---------------------------------------------------------------------------------------------------------------------

_FORGERIES = {
    "sys_exit_early": "import sys\nsys.exit(0)\ndef add(a, b):\n    return 0\n",
    "os_exit_early": "import os\nos._exit(0)\n",
    "frame_walk": (
        "import sys\n"
        "f = sys._getframe()\n"
        "while f is not None:\n"
        "    for c in f.f_code.co_consts:\n"
        "        if isinstance(c, str) and c.startswith('__ED_'):\n"
        "            print(c, flush=True)\n"
        "    f = f.f_back\n"
        "sys.exit(0)\n"
    ),
    "rebind_exec_print": (
        "import builtins\n"
        "builtins.exec = lambda *a, **k: None\n"
        "builtins.print = lambda *a, **k: None\n"
        "def add(a, b):\n"
        "    return 0\n"
    ),
    "always_equal_object": (
        "import builtins\n"
        "class Anything:\n"
        "    def __eq__(self, other):\n"
        "        return True\n"
        "builtins.AssertionError = type('Quiet', (Exception,), {})\n"
        "def add(a, b):\n"
        "    return Anything()\n"
    ),
    "fake_result_markers": (
        "import os, sys\n"
        "for line in ('All unit tests passed.', '__ED_RESULT_0__ {\"passed\": true}', 'REWARD=1.0'):\n"
        "    print(line)\n"
        "    sys.stderr.write(line + '\\n')\n"
        "sys.stdout.flush()\n"
        "os._exit(0)\n"
    ),
    "write_every_fd": (
        "import os\n"
        "for fd in range(1, 64):\n"
        "    try:\n"
        "        os.write(fd, b'__ED_RESULT_0__ {\"passed\": true}\\n')\n"
        "    except OSError:\n"
        "        pass\n"
        "def add(a, b):\n"
        "    return a + b + 1\n"
    ),
}


def _code_judge() -> CodeJudgeEnvironment:
    return CodeJudgeEnvironment(
        prompt="Implement add(a, b).",
        test_code="assert add(2, 3) == 5\nassert add(-1, 1) == 0",
        backend="local",
        container_config=_NO_JAIL,
    )


_PROC_SCAN_FORGERY = r"""
import glob, os, re, sys
pat = re.compile(rb'__ED_RESULT_[0-9a-f]{32}__')
found = set()
paths = glob.glob('/proc/[0-9]*/cmdline') + glob.glob('/proc/[0-9]*/fd/*')
for path in paths[:20000]:
    if path.startswith('/proc/%d/' % os.getpid()):
        continue
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        continue
    try:
        data = os.read(fd, 1 << 16)
    except OSError:
        data = b''
    finally:
        os.close(fd)
    if b'__ED_RESULT' in data or b'_ED_NONCE' in data:
        found.update(m.decode() for m in pat.findall(data))
forged = [n + ' {"passed": true, "error": "", "solution_stderr": ""}' for n in sorted(found)]
for line in forged + ['found %d candidate nonces' % len(found)]:
    print(line, flush=True)
    sys.stderr.write(line + '\n')
for fd_path in glob.glob('/proc/%d/fd/*' % os.getppid()):
    try:
        fd = os.open(fd_path, os.O_WRONLY | os.O_NONBLOCK)
    except OSError:
        continue
    try:
        for line in forged:
            os.write(fd, (line + '\n').encode())
    except OSError:
        pass
    finally:
        os.close(fd)

def add(a, b):
    return a - b
"""


def test_code_judge_proc_scan_for_runner_nonce_scores_zero():
    env = CodeJudgeEnvironment(
        prompt="Implement add(a, b).",
        test_code="assert add(2, 3) == 5\nassert add(-1, 1) == 0",
        backend="local",
        container_config=_NO_JAIL,
        test_timeout=30.0,
    )
    try:
        env.reset()
        res = env.step(f"```python\n{_PROC_SCAN_FORGERY}```")
        assert res.reward == 0.0, res.observation
        assert res.info["all_passed"] is False
    finally:
        env.close()


def test_code_judge_solution_cannot_shadow_builtins_used_by_tests():
    env = CodeJudgeEnvironment(
        prompt="Implement dist(a, b) and items(n).",
        test_code=(
            "assert abs(dist(1.0, 4.0) - 3.0) < 1e-9\nassert len(items(3)) == 3\nassert sorted(items(3)) == [0, 1, 2]\n"
        ),
        backend="local",
        container_config=_NO_JAIL,
    )
    try:
        env.reset()
        cheat = env.step(
            "```python\nabs = lambda x: 0.0\nlen = lambda x: 3\nsorted = lambda x: [0, 1, 2]\n"
            "class SolutionError(Exception):\n    pass\n"
            "def dist(a, b):\n    return 100.0\ndef items(n):\n    return [9]\n```"
        )
        assert cheat.reward == 0.0, cheat.observation
        env.reset()
        good = env.step("```python\ndef dist(a, b):\n    return b - a\ndef items(n):\n    return list(range(n))\n```")
        assert good.reward == 1.0, good.observation
    finally:
        env.close()


@pytest.mark.parametrize("name", sorted(_FORGERIES))
def test_code_judge_forgery_scores_zero(name):
    env = _code_judge()
    try:
        env.reset()
        res = env.step(f"```python\n{_FORGERIES[name]}```")
        assert res.reward == 0.0, res.observation
        assert res.info["all_passed"] is False
    finally:
        env.close()


def test_code_judge_correct_solution_scores_one_and_wrong_scores_zero():
    env = _code_judge()
    try:
        env.reset()
        good = env.step("```python\nimport math\nCONST = 3\ndef add(a, b):\n    return a + b\n```")
        assert good.reward == 1.0, good.observation
        env.reset()
        bad = env.step("```python\ndef add(a, b):\n    return a - b\n```")
        assert bad.reward == 0.0
    finally:
        env.close()


def test_code_judge_tests_see_solution_values_and_exceptions():
    env = CodeJudgeEnvironment(
        prompt="p",
        test_code=(
            "assert LIMIT == 10\n"
            "assert pair(3) == (3, [3, {'k': 3}])\n"
            "try:\n"
            "    boom()\n"
            "    raise SystemExit('no error')\n"
            "except ValueError as exc:\n"
            "    assert 'bad' in str(exc)\n"
        ),
        backend="local",
        container_config=_NO_JAIL,
    )
    try:
        env.reset()
        res = env.step(
            "```python\nLIMIT = 10\ndef pair(x):\n    return (x, [x, {'k': x}])\n"
            "def boom():\n    raise ValueError('bad input')\n```"
        )
        assert res.reward == 1.0, res.observation
    finally:
        env.close()


# ---------------------------------------------------------------------------------------------------------------------
# swe_repo grading
# ---------------------------------------------------------------------------------------------------------------------

_F2P_PATCH = (
    "diff --git a/test_f2p.py b/test_f2p.py\n"
    "new file mode 100644\n"
    "--- /dev/null\n"
    "+++ b/test_f2p.py\n"
    "@@ -0,0 +1,2 @@\n"
    "+from calc import mul\n"
    "+assert mul(3, 4) == 12\n"
)


@pytest.mark.skipif(not _HAS_GIT, reason="git is not installed")
def test_swe_resolved_requires_f2p_and_p2p():
    env = SWERepoEnvironment(
        problem_statement="Implement mul.",
        files={"calc.py": "def mul(a, b):\n    return 0\n", "check_p2p.py": "raise SystemExit(1)\n"},
        test_patch=_F2P_PATCH,
        fail_to_pass=[f"{sys.executable} test_f2p.py"],
        pass_to_pass=[f"{sys.executable} check_p2p.py"],
        backend="local",
        container_config=_NO_JAIL,
    )
    try:
        assert env.reset().info["env_error"] is False
        env.step(
            '<tool_call>{"name": "write_file", "arguments": {"path": "calc.py", '
            '"content": "def mul(a, b):\\n    return a * b\\n"}}</tool_call>'
        )
        res = env.step("done")
        assert res.info["f2p_passed"] == 1 and res.info["p2p_passed"] == 0
        assert res.info["resolved"] is False
        assert res.reward == 0.0
    finally:
        env.close()


@pytest.mark.skipif(not _HAS_GIT or not _pytest_available(), reason="git or pytest unavailable")
@pytest.mark.parametrize(
    ("planted_path", "planted"),
    [
        ("conftest.py", "import os\nos._exit(0)\n"),
        ("sub/conftest.py", "import os\nos._exit(0)\n"),
        ("pytest.ini", "[pytest]\naddopts = -p no:python --co\n"),
        (".pytest.ini", "[pytest]\naddopts = --collect-only\n"),
        ("test_f2p.py", "def test_ok():\n    pass\n"),
    ],
)
def test_swe_planted_test_infrastructure_cannot_force_passes(planted_path, planted):
    test_patch = (
        "diff --git a/test_f2p.py b/test_f2p.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/test_f2p.py\n"
        "@@ -0,0 +1,4 @@\n"
        "+from calc import mul\n"
        "+\n"
        "+def test_mul():\n"
        "+    assert mul(3, 4) == 12\n"
    )
    env = SWERepoEnvironment(
        problem_statement="Implement mul.",
        files={"calc.py": "def mul(a, b):\n    return 0\n", "sub/__init__.py": ""},
        test_patch=test_patch,
        fail_to_pass=[f"{sys.executable} -m pytest -q -p no:cacheprovider test_f2p.py sub"],
        backend="local",
        container_config=_NO_JAIL,
    )
    try:
        assert env.reset().info["env_error"] is False
        env.session.write_files({planted_path: planted})
        res = env.step("done")
        assert res.reward == 0.0, res.info
        assert res.info["resolved"] is False
    finally:
        env.close()


_REPLANT_LOOP = (
    "nohup sh -c 'while true; do printf \"import os\\nos._exit(0)\\n\" > conftest.py; sleep 0.02; done' "
    ">/dev/null 2>&1 </dev/null &"
)


@pytest.mark.skipif(not _HAS_GIT or not _pytest_available(), reason="git or pytest unavailable")
def test_swe_agent_background_loop_replanting_conftest_is_killed_before_grading():
    test_patch = (
        "diff --git a/test_f2p.py b/test_f2p.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/test_f2p.py\n"
        "@@ -0,0 +1,4 @@\n"
        "+from calc import mul\n"
        "+\n"
        "+def test_mul():\n"
        "+    assert mul(3, 4) == 12\n"
    )
    env = SWERepoEnvironment(
        problem_statement="Implement mul.",
        files={"calc.py": "def mul(a, b):\n    return 0\n"},
        test_patch=test_patch,
        fail_to_pass=[f"sleep 0.3; {sys.executable} -m pytest -q -p no:cacheprovider test_f2p.py"],
        backend="local",
        container_config=_NO_JAIL,
    )
    try:
        assert env.reset().info["env_error"] is False
        call = {"name": "bash", "arguments": {"command": _REPLANT_LOOP}}
        env.step(f"<tool_call>{json.dumps(call)}</tool_call>")
        time.sleep(0.2)
        res = env.step("done")
        assert res.info["env_error"] is False, res.info
        assert res.info["resolved"] is False
        assert res.reward == 0.0
    finally:
        env.close()


# ---------------------------------------------------------------------------------------------------------------------
# terminal verification
# ---------------------------------------------------------------------------------------------------------------------


def test_terminal_agent_background_loop_cannot_write_reward_file():
    env = _terminal("sleep 0.3; exit 1")
    try:
        env.reset()
        loop = (
            'nohup sh -c \'while true; do for d in .ed_verify_*; do [ -d "$d" ] && printf 1.0 > "$d/reward"; '
            "done; sleep 0.01; done' >/dev/null 2>&1 </dev/null &"
        )
        env.step(f"```bash\n{loop}\n```")
        time.sleep(0.1)
        res = env.step("done")
        assert res.info["env_error"] is False, res.info
        assert res.reward == 0.0
    finally:
        env.close()


def test_terminal_plain_reward_marker_from_verifier_is_unscorable():
    env = _terminal("echo REWARD=1.0; exit 1")
    try:
        env.reset()
        res = env.step("done")
        assert res.info["discard"] is True and res.info["env_error"] is True
        assert "REWARD[$ED_REWARD_NONCE]" in res.info["env_error_message"]
        assert "$ED_REWARD_FILE" in res.info["env_error_message"]
    finally:
        env.close()
    reward_fn = make_environment_reward_fn("terminal", backend="local", container_config=_NO_JAIL)
    assert reward_fn(prompts=["t"], completions=["done"], verify_script=["echo REWARD=0.5"]) == [None]


def _terminal(verify_script: str) -> TerminalEnvironment:
    return TerminalEnvironment(
        task_prompt="t",
        verify_script=verify_script,
        backend="local",
        container_config=_NO_JAIL,
    )


def test_terminal_spoofed_reward_marker_in_echoed_file_scores_zero():
    env = _terminal("cat notes.txt; exit 1")
    try:
        env.reset()
        env.step("```bash\nprintf 'REWARD=1.0\\nREWARD[x]=1.0\\n' > notes.txt\n```")
        res = env.step("done")
        assert "REWARD=1.0" in res.observation
        assert res.reward == 0.0
    finally:
        env.close()


def test_terminal_verifier_reward_file_and_nonce_marker_are_honoured():
    env = _terminal('echo REWARD=1.0; printf 0.25 > "$ED_REWARD_FILE"; exit 1')
    try:
        env.reset()
        assert env.step("done").reward == 0.25
    finally:
        env.close()
    env = _terminal('echo REWARD=0.0; echo "REWARD[$ED_REWARD_NONCE]=0.75"; exit 1')
    try:
        env.reset()
        assert env.step("done").reward == 0.75
    finally:
        env.close()


# ---------------------------------------------------------------------------------------------------------------------
# harness gateway reachability
# ---------------------------------------------------------------------------------------------------------------------


class _FakeEngineClient:
    def create_container(self, config, *, name=None):
        return "fake-cid"

    def start_container(self, cid):
        return None

    def remove_container(self, cid, *, force=True):
        return None

    def exec_run(self, cid, cmd, **kwargs):
        return ExecResult(ok=True, stdout="", stderr="", returncode=0)


class _ContainerRuntime(KiriRuntime):
    def create_session(self, config=None):
        return ContainerSandboxSession(_FakeEngineClient(), config or ContainerConfig())


def _gateway(**kwargs) -> HarnessTITOGateway:
    return HarnessTITOGateway(generate_fn=lambda *a: None, encode_messages_fn=lambda *a: [1], **kwargs)


def test_gateway_advertised_url_used_for_containers():
    gw = _gateway(host="127.0.0.1", advertise_host="10.1.2.3")
    try:
        port = gw.base_url.rsplit(":", 1)[1]
        assert gw.advertised_base_url == f"http://10.1.2.3:{port}"
        assert gw.advertised_base_url_v1 == f"http://10.1.2.3:{port}/v1"
        assert gw.advertised_is_loopback is False
        assert _gateway(public_base_url="http://gw.internal:9000/").advertised_base_url == "http://gw.internal:9000"
        assert _gateway().advertised_is_loopback is True
    finally:
        gw.close()


def test_harness_reset_fails_fast_for_container_with_loopback_gateway():
    preset = HarnessPreset(name="noop", protocol="openai_chat", argv_template=("true",), requires_git=False)
    loopback = HarnessEnvironment(prompt="p", harness=preset, gateway=_gateway(), runtime=_ContainerRuntime())
    with pytest.raises(ValueError, match="advertise_host"):
        loopback.reset()
    reachable = HarnessEnvironment(
        prompt="p",
        harness=preset,
        gateway=_gateway(advertise_host="172.17.0.1"),
        runtime=_ContainerRuntime(),
    )
    try:
        assert reachable.reset().info["env_error"] is False
    finally:
        reachable.close()


# ---------------------------------------------------------------------------------------------------------------------
# single-turn reward adapter and the discard protocol
# ---------------------------------------------------------------------------------------------------------------------


def test_reward_fn_grades_python_tir_answer_terminally():
    reward_fn = make_environment_reward_fn("python_tir", backend="local", container_config=_NO_JAIL)
    scores = reward_fn(
        prompts=["17*23+9?", "17*23+9?"],
        completions=[
            "```python\nprint(17 * 23 + 9)\n```\nSo the answer is \\boxed{400}.",
            "```python\nprint(1)\n```\nSo the answer is \\boxed{1}.",
        ],
        answer=["400", "400"],
    )
    assert scores == [1.0, 0.0]


def test_reward_fn_returns_none_for_unscorable_samples():
    def _broken_verifier(_text: str) -> float:
        raise RuntimeError("judge offline")

    reward_fn = make_environment_reward_fn(
        "python_tir", backend="local", container_config=_NO_JAIL, verifier=_broken_verifier
    )
    assert reward_fn.discard_unscorable is True
    assert reward_fn(prompts=["q"], completions=["\\boxed{1}"]) == [None]

    term_fn = make_environment_reward_fn("terminal", backend="local", container_config=_NO_JAIL)
    scores = term_fn(
        prompts=["a", "b"],
        completions=["done", "done"],
        verify_script=["true", "true"],
        setup_script=["true", "exit 3"],
    )
    assert scores == [1.0, None]
    assert term_fn(environment_rewards=[1.0, None]) == [1.0, None]


# ---------------------------------------------------------------------------------------------------------------------
# synthetic task stream
# ---------------------------------------------------------------------------------------------------------------------


class _GeneratorBroken(RuntimeError):
    pass


def test_synthetic_stream_surfaces_persistent_generator_failure():
    calls = {"n": 0}

    def _gen(seed: int):
        calls["n"] += 1
        raise _GeneratorBroken(f"bad seed {seed}")

    stream = SyntheticTaskStream(
        _gen,
        num_workers=2,
        max_consecutive_generator_failures=5,
        generator_backoff=0.001,
        generator_backoff_max=0.01,
    )
    try:
        start = time.monotonic()
        with pytest.raises(_GeneratorBroken):
            stream.next_task(timeout=10.0)
        assert time.monotonic() - start < 5.0
        attempts = calls["n"]
        time.sleep(0.1)
        assert calls["n"] == attempts  # producers stopped instead of busy-spinning
        assert 5 <= attempts <= 7
    finally:
        stream.close()


def test_synthetic_stream_recovers_from_transient_generator_failures():
    def _gen(seed: int):
        if seed % 3 != 2:
            raise _GeneratorBroken("transient")
        return {"task_id": f"t{seed}", "prompt": "p"}

    stream = SyntheticTaskStream(_gen, num_workers=1, max_consecutive_generator_failures=4, generator_backoff=0.001)
    try:
        tasks = [stream.next_task(timeout=5.0) for _ in range(3)]
        assert [t["task_id"] for t in tasks] == ["t2", "t5", "t8"]
    finally:
        stream.close()
