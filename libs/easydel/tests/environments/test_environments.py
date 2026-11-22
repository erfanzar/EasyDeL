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

"""CPU tests for RL environments, Kiri Engine socket client, and async rollout collector."""

from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import socket
import struct
import tarfile
import tempfile
import threading
import time
import typing as tp
import urllib.parse
import urllib.request

import pytest
from easydel.environments import (
    AsyncEnvironmentRolloutCollector,
    CodeJudgeEnvironment,
    ContainerConfig,
    EngineSocketClient,
    GatewayTurnOutput,
    GeneratedTurn,
    HarnessEnvironment,
    HarnessPreset,
    HarnessTITOGateway,
    KiriRuntime,
    LocalSandboxSession,
    SandboxEnvironment,
    SandboxInfraError,
    SandboxPool,
    StatefulPythonEnvironment,
    SWERepoEnvironment,
    SyntheticToolEnvironment,
    TerminalEnvironment,
    build_environment,
    list_environments,
    list_harnesses,
    make_environment_factory,
)
from easydel.trainers._online_rl.trajectory import OnlineRLTrajectoryCollator
from easydel.utils import Registry

pytestmark = pytest.mark.cpu_ok


class _MockKiriDaemon:
    """In-process Unix-domain-socket server speaking Kiri's Docker Engine v1.41 API."""

    def __init__(self, sock_path: str):
        self.sock_path = sock_path
        self.containers: dict[str, dict[str, tp.Any]] = {}
        self.execs: dict[str, dict[str, tp.Any]] = {}
        self.pulled_images: list[str] = []
        self.inspect_override: tuple[int, dict[str, tp.Any]] | None = None
        self.reset_on_exec_start: bool = False
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(sock_path)
        self._server.listen(16)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._counter = 0

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        try:
            self._server.close()
        except OSError:
            pass
        self._thread.join(timeout=2.0)

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        try:
            reader = conn.makefile("rb")
            req_line = reader.readline().decode("latin-1").strip()
            if not req_line:
                return
            method, target, _ = req_line.split(" ", 2)
            headers: dict[str, str] = {}
            while True:
                line = reader.readline()
                if not line or line in (b"\r\n", b"\n"):
                    break
                k, v = line.decode("latin-1").rstrip("\r\n").split(":", 1)
                headers[k.lower().strip()] = v.strip()
            length = int(headers.get("content-length", "0"))
            body = reader.read(length) if length > 0 else b""
            parsed_url = urllib.parse.urlparse(target)
            path = parsed_url.path
            qs = urllib.parse.parse_qs(parsed_url.query)

            if method == "GET" and path == "/_ping":
                self._send(conn, 200, b"OK", content_type="text/plain")
                return
            if method == "GET" and path == "/version":
                payload = json.dumps({"Version": "27.0.0", "KiriVersion": "0.3.0"}).encode()
                self._send_chunked(conn, 200, [payload[:12], payload[12:]])
                return
            if method == "GET" and path.startswith("/v1.41/images/") and path.endswith("/json"):
                self._send(conn, 404, json.dumps({"message": "No such image"}).encode())
                return
            if method == "POST" and path == "/v1.41/images/create":
                img = qs.get("fromImage", [""])[0]
                self.pulled_images.append(img)
                self._send(conn, 200, b'{"status":"Pulled"}\n')
                return
            if method == "POST" and path == "/v1.41/containers/create":
                self._counter += 1
                cid = f"kiri_cid_{self._counter}"
                spec = json.loads(body.decode("utf-8"))
                self.containers[cid] = {"spec": spec, "started": False, "files": {}}
                self._send(conn, 201, json.dumps({"Id": cid}).encode())
                return
            if method == "POST" and path.startswith("/v1.41/containers/") and path.endswith("/start"):
                cid = path.split("/")[3]
                self.containers[cid]["started"] = True
                # Mimic Kiri keeping 204 connections open with no body.
                conn.sendall(b"HTTP/1.1 204 No Content\r\n\r\n")
                time.sleep(0.05)
                return
            if method == "DELETE" and path.startswith("/v1.41/containers/"):
                cid = path.split("/")[3]
                self.containers.pop(cid, None)
                conn.sendall(b"HTTP/1.1 204 No Content\r\n\r\n")
                return
            if method == "PUT" and path.endswith("/archive"):
                cid = path.split("/")[3]
                base = qs.get("path", ["/workspace"])[0].rstrip("/")
                with tarfile.open(fileobj=io.BytesIO(body), mode="r:*") as tf:
                    for member in tf.getmembers():
                        if member.isfile():
                            fobj = tf.extractfile(member)
                            assert fobj is not None
                            full = f"{base}/{member.name}".replace("//", "/")
                            self.containers[cid]["files"][full] = fobj.read()
                self._send(conn, 200, b"")
                return
            if method == "GET" and path.endswith("/archive"):
                cid = path.split("/")[3]
                req_path = qs.get("path", [""])[0]
                raw = self.containers[cid]["files"].get(req_path)
                if raw is None:
                    self._send(conn, 404, json.dumps({"message": "not found"}).encode())
                    return
                buf = io.BytesIO()
                with tarfile.open(fileobj=buf, mode="w") as tf:
                    info = tarfile.TarInfo(name=os.path.basename(req_path))
                    info.size = len(raw)
                    tf.addfile(info, io.BytesIO(raw))
                self._send(conn, 200, buf.getvalue(), content_type="application/x-tar")
                return
            if method == "POST" and path.endswith("/exec"):
                cid = path.split("/")[3]
                self._counter += 1
                eid = f"exec_{self._counter}"
                self.execs[eid] = {"cid": cid, "spec": json.loads(body.decode("utf-8")), "exit": 0}
                self._send(conn, 201, json.dumps({"Id": eid}).encode())
                return
            if method == "POST" and path.startswith("/v1.41/exec/") and path.endswith("/start"):
                eid = path.split("/")[3]
                rec = self.execs[eid]
                cmd = rec["spec"]["Cmd"]
                cid = rec["cid"]
                # exec_run wraps user commands as [sh, -c, watchdog, sh, pid_file, backstop_s, stdin_path, *cmd].
                stdin_file = ""
                if len(cmd) >= 7 and cmd[:2] == ["/bin/sh", "-c"] and cmd[3] == "sh":
                    stdin_file = cmd[6]
                    cmd = cmd[7:]
                if self.reset_on_exec_start and cmd[:2] not in (["mkdir", "-p"], ["rm", "-f"]):
                    conn.close()
                    return
                if cmd[:2] == ["mkdir", "-p"] or cmd[:2] == ["rm", "-f"]:
                    rec["exit"] = 0
                    self._send(conn, 200, b"")
                    return
                cmdline = " ".join(cmd)
                if stdin_file:
                    stdin_bytes = self.containers[cid]["files"].get(stdin_file, b"")
                    out = b"STDIN=" + stdin_bytes
                    err = b"warn-line\n"
                    rec["exit"] = 0
                elif "fail_cmd" in cmdline:
                    out = b""
                    err = b"boom\n"
                    rec["exit"] = 2
                else:
                    out = f"ran:{cmdline}\n".encode()
                    err = b""
                    rec["exit"] = 0
                framed = b""
                if out:
                    framed += struct.pack(">BxxxI", 1, len(out)) + out
                if err:
                    framed += struct.pack(">BxxxI", 2, len(err)) + err
                self._send(conn, 200, framed, content_type="application/vnd.docker.multiplexed-stream")
                return
            if method == "GET" and path.startswith("/v1.41/exec/") and path.endswith("/json"):
                if self.inspect_override is not None:
                    st, payload = self.inspect_override
                    self._send(conn, st, json.dumps(payload).encode())
                    return
                eid = path.split("/")[3]
                code = self.execs[eid]["exit"]
                self._send(conn, 200, json.dumps({"ExitCode": code, "Running": False}).encode())
                return
            self._send(conn, 404, json.dumps({"message": f"unhandled {method} {path}"}).encode())
        finally:
            try:
                conn.close()
            except OSError:
                pass

    @staticmethod
    def _send(conn: socket.socket, status: int, body: bytes, content_type: str = "application/json") -> None:
        head = (
            f"HTTP/1.1 {status} OK\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii")
        conn.sendall(head + body)

    @staticmethod
    def _send_chunked(conn: socket.socket, status: int, chunks: list[bytes]) -> None:
        conn.sendall(
            f"HTTP/1.1 {status} OK\r\nTransfer-Encoding: chunked\r\nContent-Type: application/json\r\n\r\n".encode(
                "ascii"
            )
        )
        for ch in chunks:
            conn.sendall(f"{len(ch):x}\r\n".encode("ascii") + ch + b"\r\n")
        conn.sendall(b"0\r\n\r\n")


def test_kiri_socket_wire_protocol_and_container_session():
    short_dir = tempfile.mkdtemp(prefix="kiri-", dir="/tmp")
    sock_path = os.path.join(short_dir, "docker.sock")
    daemon = _MockKiriDaemon(sock_path)
    daemon.start()
    try:
        cfg = ContainerConfig(image="alpine:3.20", memory_mb=512, cpus=2.0, pids_limit=64, network=False)
        runtime = KiriRuntime(backend="kiri", endpoint=f"unix://{sock_path}", config=cfg)
        endpoint, flavor = runtime.resolve()
        assert endpoint == f"unix://{sock_path}"
        assert flavor == "kiri"

        with runtime.create_session() as session:
            assert daemon.pulled_images == ["alpine:3.20"]
            session.write_files({"sub/hello.txt": "hello-kiri"})
            assert session.read_file("sub/hello.txt") == "hello-kiri"

            res_ok = session.exec(["echo", "hi"], stdin="payload-123")
            assert res_ok.ok is True
            assert res_ok.stdout == "STDIN=payload-123"
            assert res_ok.stderr == "warn-line\n"

            res_fail = session.run_bash("fail_cmd")
            assert res_fail.ok is False
            assert res_fail.returncode == 2
            assert "boom" in res_fail.stderr

            # Verify HostConfig security/resource limits sent to Kiri.
            created_specs = [c["spec"] for c in daemon.containers.values()]
            assert len(created_specs) == 1
            hc = created_specs[0]["HostConfig"]
            assert hc["NetworkMode"] == "none"
            assert hc["Memory"] == 512 * 1024 * 1024
            assert hc["NanoCpus"] == 2_000_000_000
            assert hc["PidsLimit"] == 64
            assert "no-new-privileges" in hc["SecurityOpt"]

        # Container removed on context exit.
        assert daemon.containers == {}
    finally:
        daemon.close()
        shutil.rmtree(short_dir, ignore_errors=True)


def test_local_sandbox_session_persistence_timeout_and_escape_guard():
    cfg = ContainerConfig(timeout=1.0, memory_mb=512, use_jail="off")
    with LocalSandboxSession(cfg) as session:
        root = session.workdir
        session.write_files({"notes/a.txt": "step1"})
        res = session.run_bash("cat notes/a.txt && echo step2 > notes/b.txt")
        assert res.ok
        assert res.stdout.strip() == "step1"
        assert session.read_file("notes/b.txt").strip() == "step2"
        assert res.to_sandbox_result().ok is True

        with pytest.raises(ValueError, match="escapes sandbox"):
            session.write_files({"../../outside.txt": "nope"})

        slow = session.run_python("import time; time.sleep(5)", timeout=0.3)
        assert slow.ok is False
        assert slow.timed_out is True
    assert not os.path.exists(root)


def test_sandbox_pool_consistent_affinity_and_unreachable_kiri_error():
    pool = SandboxPool(
        endpoints=["http://10.0.0.1:2375", "http://10.0.0.2:2375", "http://10.0.0.3:2375"],
        backend="kiri",
    )
    assert pool.size == 3
    idx_a1 = pool.select_index("rollout-42")
    idx_a2 = pool.select_index("rollout-42")
    assert idx_a1 == idx_a2
    rr = [pool.select_index(None) for _ in range(4)]
    assert rr == [0, 1, 2, 0]

    bad_rt = KiriRuntime(backend="kiri", endpoint="unix:///tmp/nonexistent-kiri-test.sock")
    with pytest.raises(SandboxInfraError):
        bad_rt.create_session()


def test_registry_and_code_judge_single_and_multiturn():
    assert set(list_environments()) >= {"code_judge", "swe_repo", "terminal", "python_tir", "synthetic_tool"}
    assert Registry.get_or_raise("environment", "code_judge") is CodeJudgeEnvironment

    env = build_environment(
        {
            "name": "code_judge",
            "prompt": "Read two integers and print their sum.",
            "stdin_tests": [{"input": "2 3\n", "output": "5"}, {"input": "10 -4\n", "output": "6"}],
            "reward_mode": "pass_rate",
            "max_steps": 2,
            "backend": "local",
            "container_config": ContainerConfig(use_jail="off"),
        }
    )
    try:
        reset = env.reset()
        assert "sum" in reset.observation

        # Turn 1: buggy solution -> feedback returned, episode continues.
        s1 = env.step("Here is my attempt:\n```python\nprint(5)\n```")
        assert s1.terminated is False
        assert s1.reward == 0.0
        assert "Test 2/2 failed" in s1.observation

        # Turn 2: fixed solution -> passes all tests.
        s2 = env.step("Fixed:\n```python\na, b = map(int, input().split())\nprint(a + b)\n```")
        assert s2.terminated is True
        assert s2.reward == 1.0
        assert s2.info["all_passed"] is True
    finally:
        env.close()


def test_swe_repo_environment_end_to_end_fix_and_test_patch():
    env = SWERepoEnvironment(
        problem_statement="Fix `divide(a, b)` in `calc.py` so division by zero raises ValueError and `add` still works.",
        files={
            "calc.py": ("def add(a, b):\n    return a + b\n\ndef divide(a, b):\n    return a // b\n"),
            "test_calc.py": "from calc import add\nassert add(2, 3) == 5\n",
        },
        test_patch=(
            "diff --git a/test_f2p.py b/test_f2p.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/test_f2p.py\n"
            "@@ -0,0 +1,7 @@\n"
            "+from calc import divide\n"
            "+assert divide(8, 2) == 4\n"
            "+try:\n"
            "+    divide(1, 0)\n"
            "+    raise SystemExit(1)\n"
            "+except ValueError:\n"
            "+    pass\n"
        ),
        fail_to_pass=["python3 test_f2p.py"],
        pass_to_pass=["python3 test_calc.py"],
        backend="local",
        container_config=ContainerConfig(use_jail="off"),
    )
    try:
        env.reset()
        r1 = env.step('<tool_call>{"name": "read_file", "arguments": {"path": "calc.py"}}</tool_call>')
        assert "def divide" in r1.observation
        assert r1.terminated is False

        edit_payload = {
            "name": "edit_file",
            "arguments": {
                "path": "calc.py",
                "old_text": "def divide(a, b):\n    return a // b",
                "new_text": "def divide(a, b):\n    if b == 0:\n        raise ValueError('zero')\n    return a // b",
            },
        }
        r2 = env.step(f"<tool_call>{json.dumps(edit_payload)}</tool_call>")
        assert "Updated calc.py" in r2.observation
        assert r2.terminated is False

        r3 = env.step("I have fixed the division-by-zero bug in `calc.py`.")
        assert r3.terminated is True
        assert r3.reward == 1.0
        assert r3.info["resolved"] is True
        assert "+    if b == 0:" in r3.info["patch"]
    finally:
        env.close()


def test_terminal_and_stateful_python_tir_environments():
    term = TerminalEnvironment(
        task_prompt="Count lines in data.txt and write the integer to count.txt.",
        files={"data.txt": "alpha\nbeta\ngamma\n"},
        verify_script='test "$(cat count.txt | tr -d " \\n")" = "3"',
        backend="local",
        container_config=ContainerConfig(use_jail="off"),
    )
    try:
        term.reset()
        t1 = term.step("```bash\nwc -l < data.txt > count.txt\n```")
        assert t1.terminated is False
        t2 = term.step("Task complete.")
        assert t2.terminated is True
        assert t2.reward == 1.0
    finally:
        term.close()

    tir = StatefulPythonEnvironment(
        prompt="Compute 17 * 23 + 9 using Python.",
        answer="400",
        backend="local",
        container_config=ContainerConfig(use_jail="off"),
    )
    try:
        tir.reset()
        p1 = tir.step("```python\nimport math\nx = 17 * 23\nprint(x)\n```")
        assert p1.terminated is False
        assert "391" in p1.observation

        # `x` and `math` persist from turn 1!
        p2 = tir.step("```python\ny = x + int(math.isqrt(81))\nprint(y)\n```")
        assert "400" in p2.observation

        p3 = tir.step("Therefore the answer is \\boxed{400}.")
        assert p3.terminated is True
        assert p3.reward == 1.0
    finally:
        tir.close()


def test_synthetic_tool_and_parallel_step_batch():
    def book_hotel(state: dict[str, tp.Any], city: str, price: int) -> str:
        """Book a hotel in a city.

        city: Target city name.
        price: Nightly rate.
        """
        state["bookings"].append({"city": city, "price": price})
        return f"Booked {city} at {price}"

    factory = make_environment_factory(
        "synthetic_tool",
        prompt="Book Hangzhou under 500 and submit.",
        initial_state={"bookings": []},
        tools={"book_hotel": book_hotel},
        verifier=lambda st, ans: (
            1.0 if st["bookings"] == [{"city": "Hangzhou", "price": 400}] and "done" in ans else 0.0
        ),
    )
    envs = [factory() for _ in range(3)]
    assert envs[0].tool_schemas[0]["function"]["name"] == "book_hotel"
    assert "state" not in envs[0].tool_schemas[0]["function"]["parameters"]["properties"]

    for e in envs:
        e.reset()
    r_step1 = SandboxEnvironment.step_batch(
        envs,
        [
            '<tool_call>{"name": "book_hotel", "arguments": {"city": "Hangzhou", "price": 400}}</tool_call>',
            '<tool_call>{"name": "book_hotel", "arguments": {"city": "Hangzhou", "price": 900}}</tool_call>',
            "done without booking",
        ],
    )
    assert r_step1[0].terminated is False
    assert r_step1[2].terminated is True
    assert r_step1[2].reward == 0.0

    final0 = envs[0].step('<tool_call>{"name": "submit_result", "arguments": {"answer_text": "all done"}}</tool_call>')
    final1 = envs[1].step("all done")
    assert final0.reward == 1.0
    assert final1.reward == 0.0

    # Verify async coroutine interface as well.
    async def _run_async() -> float:
        e = factory()
        await e.reset_async()
        await e.step_async(
            '<tool_call>{"name": "book_hotel", "arguments": {"city": "Hangzhou", "price": 400}}</tool_call>'
        )
        done = await e.step_async("done")
        return done.reward

    assert asyncio.run(_run_async()) == 1.0


def test_async_environment_rollout_collector_tito_staleness_and_group_padding():
    def _encode(text: str) -> list[int]:
        return [max(1, ord(ch) % 64) for ch in text[:16]] or [1]

    class _FastSlowOrCrashEnv(SyntheticToolEnvironment):
        def __init__(self, mode: str = "fast"):
            self._mode = mode
            super().__init__(
                prompt=f"mode={mode}",
                initial_state={"steps": 0},
                tools={"tick": lambda state: state.__setitem__("steps", state["steps"] + 1) or "ticked"},
                verifier=lambda state, _ans: 1.0 if state["steps"] >= 1 else 0.0,
            )

        def step(self, action: str):
            if self._mode == "crash":
                return SandboxEnvironment._infra_failure_step("simulated kiri socket drop")
            if self._mode == "slow":
                time.sleep(0.08)
            return super().step(action)

    def _gen_turn(policy_tag: str, context_ids: list[int], turn_idx: int) -> GeneratedTurn:
        assert len(context_ids) > 0
        if turn_idx == 0:
            txt = '<tool_call>{"name": "tick", "arguments": {}}</tool_call>'
            toks = [10, 11, 12]
        else:
            txt = f"final-{policy_tag}"
            toks = [20, 21]
        return GeneratedTurn(token_ids=toks, text=txt, logprobs=[-0.5] * len(toks))

    collector = AsyncEnvironmentRolloutCollector(
        lambda mode="fast": _FastSlowOrCrashEnv(mode=mode),
        generate_turn_fn=_gen_turn,
        encode_fn=_encode,
        max_workers=4,
        max_outstanding=16,
        max_turns=4,
        max_staleness=1,
        drop_env_errors=True,
    )
    try:
        # 1) Completion-order: slow submitted first, fast finishes first!
        collector.publish_policy("pi_v1", version=1)
        collector.submit_episode(prompt_index=0, group_id=0, env_kwargs={"mode": "slow"})
        collector.submit_episode(prompt_index=1, group_id=1, env_kwargs={"mode": "fast"})
        first_only, _ = collector.collect_episodes(1, current_policy_version=1, timeout_per_episode=5.0)
        assert len(first_only) == 1
        assert first_only[0].prompt_index == 1  # Fast beat slow!

        # Let the slow v1 episode finish, then advance policy to v4 so v1 (lag=3 > max_staleness=1) is dropped!
        time.sleep(0.22)
        collector.publish_policy("pi_v4", version=4)
        collector.submit_episode(prompt_index=2, group_id=2, env_kwargs={"mode": "fast"})
        collator = OnlineRLTrajectoryCollator(max_length=128, pad_token_id=0)
        batch, metrics = collector.collect_collated_batch(
            1,
            current_policy_version=4,
            collator=collator,
            timeout_per_episode=5.0,
        )
        assert metrics["async_env/dropped_stale"] == 1
        assert batch["input_ids"].shape[0] == 1
        # TITO check: prompt & observation tokens have action_mask == False, only generated tokens (5 total) are True!
        assert int(batch["action_mask"].sum()) == 5
        assert batch["behavior_logps"] is not None

        # 2) GLM-5 group recovery: Group 0 has 3 fast + 1 crash (G=4 -> >=2 valid -> padded to 4);
        #    Group 1 has 1 fast + 3 crash (1 < 2 -> group dropped).
        collector.publish_policy("pi_v5", version=5)
        for k in range(4):
            collector.submit_episode(
                prompt_index=0,
                group_id=0,
                env_kwargs={"mode": "crash" if k == 3 else "fast"},
            )
        for k in range(4):
            collector.submit_episode(
                prompt_index=1,
                group_id=1,
                env_kwargs={"mode": "fast" if k == 0 else "crash"},
            )

        # Drain manually via collect_groups helper on a dedicated homogeneous/per-group call:
        # First clear the 8 submitted above via collect_episodes to verify env_error drop count!
        valid_eps, err_stats = collector.collect_episodes(8, current_policy_version=5, timeout_per_episode=5.0)
        assert len(valid_eps) == 4
        assert err_stats["dropped_env_error"] == 4

        # Now test collect_groups directly with partial crashes via stateful counter:
        calls = {"g0": 0, "g1": 0}
        lock = threading.Lock()

        def _group_env_factory(gid: str = "g0"):
            with lock:
                calls[gid] += 1
                idx = calls[gid]
            if gid == "g0":
                return _FastSlowOrCrashEnv(mode="crash" if idx == 4 else "fast")
            return _FastSlowOrCrashEnv(mode="fast" if idx == 1 else "crash")

        group_collector = AsyncEnvironmentRolloutCollector(
            _group_env_factory,
            generate_turn_fn=_gen_turn,
            encode_fn=_encode,
            max_workers=4,
            max_outstanding=16,
            max_staleness=2,
        )
        try:
            group_collector.publish_policy("pi_v1", version=1)
            gbatch = group_collector.collect_groups(
                num_groups=2,
                group_size=4,
                current_policy_version=1,
                env_kwargs_per_group=[{"gid": "g0"}, {"gid": "g1"}],
                timeout_per_episode=5.0,
            )
            assert gbatch.dropped_env_error == 4
            assert gbatch.dropped_groups == 1
            assert gbatch.padded_samples == 1
            assert len(gbatch.episodes) == 4
            assert all(ep.group_id == 0 for ep in gbatch.episodes)
        finally:
            group_collector.close()
    finally:
        collector.close()


def test_kiri_exec_inspect_failure_null_exit_and_io_reset_are_infra_errors():
    short_dir = tempfile.mkdtemp(prefix="kiri-err-", dir="/tmp")
    sock_path = os.path.join(short_dir, "docker.sock")
    daemon = _MockKiriDaemon(sock_path)
    daemon.start()
    try:
        client = EngineSocketClient(f"unix://{sock_path}")
        cfg = ContainerConfig(image="alpine:3.20", pull_if_missing=False)
        cid = client.create_container(cfg)
        client.start_container(cid)

        # 1) Failed exec inspect (HTTP 500) must be an infra_error, not ok=True.
        daemon.inspect_override = (500, {"message": "inspect crashed"})
        r_500 = client.exec_run(cid, ["echo", "ok"])
        assert r_500.ok is False
        assert r_500.infra_error is not None
        assert "exec inspect failed" in r_500.infra_error

        # 2) Null ExitCode must be an infra_error, not exit 0.
        daemon.inspect_override = (200, {"ExitCode": None, "Running": False})
        r_null = client.exec_run(cid, ["echo", "ok"])
        assert r_null.ok is False
        assert r_null.infra_error is not None
        assert "null ExitCode" in r_null.infra_error

        # 3) Socket drop/reset during exec start must be wrapped as infra_error.
        daemon.inspect_override = None
        daemon.reset_on_exec_start = True
        r_drop = client.exec_run(cid, ["echo", "ok"])
        assert r_drop.ok is False
        assert r_drop.infra_error is not None
    finally:
        daemon.close()
        shutil.rmtree(short_dir, ignore_errors=True)


def test_code_judge_blocks_early_exit_reward_hack():
    env = CodeJudgeEnvironment(
        prompt="Implement add(a, b).",
        test_code="assert add(2, 3) == 5",
        max_steps=1,
        backend="local",
        container_config=ContainerConfig(use_jail="off"),
    )
    try:
        env.reset()
        # Early sys.exit(0) before tests run must NOT pass!
        hack1 = env.step("```python\nimport sys\nsys.exit(0)\ndef add(a, b):\n    return 0\n```")
        assert hack1.reward == 0.0
        assert "before unit tests completed" in hack1.observation

        # Real passing implementation must pass!
        env.reset()
        real = env.step("```python\ndef add(a, b):\n    return a + b\n```")
        assert real.reward == 1.0
    finally:
        env.close()


def test_code_judge_blocks_file_and_cmdline_sentinel_inspection():
    env = CodeJudgeEnvironment(
        prompt="Implement add(a, b).",
        test_code="assert add(2, 3) == 5",
        max_steps=1,
        backend="local",
        container_config=ContainerConfig(use_jail="off"),
    )
    try:
        r0 = env.reset()
        assert r0.info["isolated_test_runner"] is True

        # Reading __file__ (solution.py) and /proc/self/cmdline must NOT reveal any sentinel.
        hack_file = env.step(
            "```python\n"
            "import re, sys\n"
            "for p in (__file__, '/proc/self/cmdline'):\n"
            "    try:\n"
            "        data = open(p, 'rb').read().decode('latin-1', errors='ignore')\n"
            "        for m in re.findall(r'__ED_UNIT_TESTS_PASSED_[0-9a-f]+__', data):\n"
            "            print(m, flush=True)\n"
            "    except OSError:\n"
            "        pass\n"
            "sys.exit(0)\n"
            "```"
        )
        assert hack_file.reward == 0.0
    finally:
        env.close()


def test_code_judge_container_backend_blocks_fd0_and_tmp_stdin_leak():
    """Verify a container-backed code judge cannot forge a pass, and exec_run's wire layout."""
    import shutil
    import subprocess
    import sys
    import tempfile

    from easydel.environments.runtime import ContainerSandboxSession, ExecResult

    class _LocalContainerFakeClient:
        def __init__(self, root: str):
            self.root = root
            self.tmp_dir = os.path.join(root, "tmp")
            self.ws_dir = os.path.join(root, "workspace")
            os.makedirs(self.tmp_dir, exist_ok=True)
            os.makedirs(self.ws_dir, exist_ok=True)

        def create_container(self, config, *, name=None):
            return "fake-cid"

        def start_container(self, cid):
            return None

        def remove_container(self, cid, *, force=True):
            return None

        def put_files(self, cid, files, *, base_dir="/workspace"):
            dest = self.tmp_dir if base_dir == "/tmp" else self.ws_dir
            for rel, content in files.items():
                raw = content.encode("utf-8") if isinstance(content, str) else bytes(content)
                target = os.path.join(dest, rel)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "wb") as f:
                    f.write(raw)

        def get_file(self, cid, path):
            return b""

        def exec_run(
            self,
            cid,
            cmd,
            *,
            workdir=None,
            env=None,
            stdin=None,
            timeout=30.0,
            max_output_chars=16384,
            max_stdin_bytes=None,
        ):
            # Stand-in for the engine: run the command locally with stdin piped straight in.
            # The production wrapper's fd handling is covered against EngineSocketClient itself in
            # test_runtime_sandbox.py::test_container_exec_stdin_file_is_held_only_by_the_command.
            if cmd[:2] == ["mkdir", "-p"]:
                return ExecResult(ok=True, stdout="", stderr="", returncode=0)
            exec_cmd = [sys.executable if part == "python3" else part for part in cmd]
            proc = subprocess.run(
                exec_cmd, cwd=self.ws_dir, input=stdin, capture_output=True, text=True, timeout=timeout
            )
            return ExecResult(
                ok=(proc.returncode == 0), stdout=proc.stdout, stderr=proc.stderr, returncode=proc.returncode
            )

    root = tempfile.mkdtemp(prefix="ed-fake-ctr-")
    try:
        fake_client = _LocalContainerFakeClient(root)
        sess = ContainerSandboxSession(fake_client, ContainerConfig())
        seen_cmds: list[list[str]] = []

        class _SpyClient(EngineSocketClient):
            def __init__(self):
                pass

            def put_files(self, container_id, files, *, base_dir="/workspace"):
                fake_client.put_files(container_id, files, base_dir=base_dir)

            def _request(
                self,
                method,
                path,
                *,
                body=None,
                content_type="application/json",
                timeout=30.0,
                max_body_bytes=None,
            ):
                if path.endswith("/exec") and isinstance(body, dict):
                    seen_cmds.append(list(body["Cmd"]))
                    return type("Resp", (), {"status": 201, "body": b'{"Id":"e1"}', "headers": {}})()
                return type("Resp", (), {"status": 200, "body": b"{}", "headers": {}})()

        class _FakeRuntime(KiriRuntime):
            def create_session(self, *, config=None, backend=None):
                return sess

        spy = _SpyClient()
        env = CodeJudgeEnvironment(
            prompt="Implement add(a, b).",
            test_code="assert add(2, 3) == 5",
            max_steps=1,
            runtime=_FakeRuntime(backend="local"),
        )
        env.reset()
        hack = env.step(
            "```python\n"
            "import glob, os, re, sys\n"
            "for p in ['/proc/self/fd/0', *glob.glob('/tmp/.ed_stdin_*')]:\n"
            "    try:\n"
            "        with open(p, 'rb') as f:\n"
            "            data = f.read().decode('latin-1', errors='ignore')\n"
            "        for m in re.findall(r'__ED_UNIT_TESTS_PASSED_[0-9a-f]+__', data):\n"
            "            print(m, flush=True)\n"
            "    except OSError:\n"
            "        pass\n"
            "for fn in (lambda: os.read(3, 65536), lambda: os.pread(3, 65536, 0)):\n"
            "    try:\n"
            "        data = fn().decode('latin-1', errors='ignore')\n"
            "        for m in re.findall(r'__ED_UNIT_TESTS_PASSED_[0-9a-f]+__', data):\n"
            "            print(m, flush=True)\n"
            "    except OSError:\n"
            "        pass\n"
            "sys.exit(0)\n"
            "```"
        )
        assert hack.reward == 0.0
        try:
            spy.exec_run("cid", ["python3", "-I", "-"], stdin="print(1)")
        except Exception:
            pass
        # Wire contract with the engine: [sh, -c, watchdog, sh, pid_file, backstop_s, stdin_path, *cmd].
        assert seen_cmds, "exec_run never created an exec"
        sent = seen_cmds[0]
        assert sent[:2] == ["/bin/sh", "-c"] and sent[3] == "sh"
        assert sent[6].startswith("/tmp/.ed_stdin_")
        assert sent[7:] == ["python3", "-I", "-"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_local_sandbox_close_kills_background_process_groups():
    sess = LocalSandboxSession(ContainerConfig(use_jail="off", timeout=5.0))
    try:
        res = sess.run_bash("sleep 60 >/dev/null 2>&1 & echo $!")
        assert res.ok is True
        bg_pid = int(res.stdout.strip().splitlines()[-1])
        os.kill(bg_pid, 0)  # Alive before close()
    finally:
        sess.close()

    with pytest.raises(ProcessLookupError):
        os.kill(bg_pid, 0)


def test_code_judge_sentinel_file_inspection_adversarial():
    env = CodeJudgeEnvironment(
        prompt="Implement add(a, b).",
        test_code="assert add(2, 3) == 5",
        max_steps=1,
        backend="local",
        container_config=ContainerConfig(use_jail="off"),
    )
    try:
        env.reset()
        # Adversarial solution walks caller frame co_consts and exits 0.
        hack = env.step(
            "```python\n"
            "import sys\n"
            "f = sys._getframe()\n"
            "while f is not None:\n"
            "    for c in f.f_code.co_consts:\n"
            "        if isinstance(c, str) and c.startswith('__ED_UNIT_TESTS_PASSED_'):\n"
            "            print(c, flush=True)\n"
            "    f = f.f_back\n"
            "sys.exit(0)\n"
            "```"
        )
        assert hack.reward == 0.0
    finally:
        env.close()


def test_unjailed_warning_and_required_jail_for_shell_envs():
    if shutil.which("bwrap") is None:
        # use_jail="auto" without bwrap never runs unjailed: reset reports an infra error.
        cj = CodeJudgeEnvironment(prompt="hi", test_code="assert True", backend="local")
        res = cj.reset()
        assert res.info["env_error"] is True
        assert "bwrap" in res.info["env_error_message"]
        cj.close()

        # Only an explicit use_jail="off" runs unjailed, and the session says so.
        cj_off = CodeJudgeEnvironment(
            prompt="hi", test_code="assert True", backend="local", container_config=ContainerConfig(use_jail="off")
        )
        res_off = cj_off.reset()
        assert res_off.info["unjailed"] is True
        cj_off.close()

        # TerminalEnvironment and SWERepoEnvironment default to use_jail="required".
        term = TerminalEnvironment(task_prompt="hi", backend="local")
        term_res = term.reset()
        assert term_res.info["env_error"] is True
        assert "bwrap" in term_res.info["env_error_message"]

        swe = SWERepoEnvironment(problem_statement="hi", backend="local")
        swe_res = swe.reset()
        assert swe_res.info["env_error"] is True
        assert "bwrap" in swe_res.info["env_error_message"]


def test_harness_tito_gateway_protocols_compaction_subagents_and_cli_execution():
    assert {"xerxes", "claude_code", "codex", "hermes", "pi", "opencode", "aider", "goose"}.issubset(
        set(list_harnesses())
    )

    turn_counter = {"n": 0}

    def _encode_msgs(msgs: list[dict[str, tp.Any]], _tools: list[dict[str, tp.Any]] | None) -> list[int]:
        return [10 + (i % 50) for i in range(max(1, len(msgs) * 3))]

    def _gen_turn(ctx_tokens: list[int], msgs: list[dict[str, tp.Any]], _tools: tp.Any) -> GatewayTurnOutput:
        assert len(ctx_tokens) > 0
        turn_counter["n"] += 1
        n = turn_counter["n"]
        return GatewayTurnOutput(
            token_ids=[100 + n, 200 + n],
            text=f"reply-{n}",
            logprobs=[-0.2, -0.3],
            policy_version=7,
        )

    gw = HarnessTITOGateway(generate_fn=_gen_turn, encode_messages_fn=_encode_msgs, model_name="test-policy").start()
    try:
        # 1) Test OpenAI chat/completions + prefix stitching + compaction split + subagent stream
        _ep_id, api_key = gw.open_episode(trajectory_id=42)
        headers_root = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        def _post(path: str, payload: dict[str, tp.Any], extra_headers: dict[str, str] | None = None) -> tuple[int, str]:
            hdrs = {**headers_root, **(extra_headers or {})}
            req = urllib.request.Request(
                f"{gw.base_url}{path}",
                data=json.dumps(payload).encode("utf-8"),
                headers=hdrs,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                return resp.status, resp.read().decode("utf-8")

        # Turn 1 (OpenAI Chat)
        st, raw = _post("/v1/chat/completions", {"messages": [{"role": "user", "content": "fix bug"}]})
        assert st == 200
        assert json.loads(raw)["choices"][0]["message"]["content"] == "reply-1"

        # Turn 2 (prefix extended -> stays in segment 0!)
        st, raw = _post(
            "/v1/chat/completions",
            {
                "messages": [
                    {"role": "user", "content": "fix bug"},
                    {"role": "assistant", "content": "reply-1"},
                    {"role": "user", "content": "tool output"},
                ]
            },
        )
        assert st == 200
        assert json.loads(raw)["choices"][0]["message"]["content"] == "reply-2"

        # Turn 3 (compacted history -> non-prefix -> seals segment 0 and opens segment 1!)
        st, raw = _post("/v1/chat/completions", {"messages": [{"role": "user", "content": "compacted summary"}]})
        assert st == 200
        assert json.loads(raw)["choices"][0]["message"]["content"] == "reply-3"

        # Subagent call on separate stream (Anthropic /v1/messages SSE)
        st, sse_raw = _post(
            "/v1/messages",
            {"messages": [{"role": "user", "content": "subagent check"}], "stream": True},
            extra_headers={"X-Agent-ID": "subagent-1"},
        )
        assert st == 200
        assert "message_stop" in sse_raw

        # OpenAI /v1/responses endpoint
        st, resp_raw = _post(
            "/v1/responses",
            {"input": "final check"},
            extra_headers={"X-Agent-ID": "subagent-2"},
        )
        assert st == 200
        assert json.loads(resp_raw)["output_text"] == "reply-5"

        trajs_root, meta_root = gw.finalize_episode(api_key, reward=1.0, subagent_mode="root_only")
        assert meta_root["segments"] == 2  # 2 compacted root segments, subagents excluded!
        assert meta_root["oldest_policy_version"] == 7
        assert all(t.trajectory_id == 42 for t in trajs_root)
        assert float(trajs_root[-1].rewards.sum()) == 1.0

        # 2) End-to-end HarnessEnvironment launching a mock CLI script inside LocalSandboxSession
        mock_cli = (
            "import json, os, sys, urllib.request\n"
            "url = os.environ['OPENAI_BASE_URL'].rstrip('/') + '/chat/completions'\n"
            "req = urllib.request.Request(\n"
            "    url,\n"
            "    data=json.dumps({'messages': [{'role': 'user', 'content': sys.argv[1]}]}).encode(),\n"
            "    headers={'Authorization': 'Bearer ' + os.environ['OPENAI_API_KEY'], 'Content-Type': 'application/json'},\n"
            "    method='POST',\n"
            ")\n"
            "with urllib.request.urlopen(req) as resp:\n"
            "    data = json.loads(resp.read().decode())\n"
            "with open('result.txt', 'w') as f:\n"
            "    f.write(data['choices'][0]['message']['content'])\n"
        )
        custom_preset = HarnessPreset(
            name="mock_cli",
            protocol="openai_chat",
            argv_template=("python3", "mock_cli.py", "{prompt}"),
            env_template={"OPENAI_BASE_URL": "{base_url_v1}", "OPENAI_API_KEY": "{api_key}"},
            requires_git=False,
        )
        henv = HarnessEnvironment(
            prompt="hello-harness",
            harness=custom_preset,
            gateway=gw,
            files={"mock_cli.py": mock_cli},
            verify_script="grep -q 'reply-6' result.txt",
            backend="local",
            container_config=ContainerConfig(network=True, use_jail="off"),
        )
        try:
            henv.reset()
            hres = henv.step()
            assert hres.reward == 1.0
            assert hres.info["resolved"] is True
            assert len(henv.last_trajectories) == 1
            assert int(henv.last_trajectories[0].action_mask.sum()) == 2
        finally:
            henv.close()
    finally:
        gw.close()


def test_local_sandbox_avoids_preexec_fn_and_enforces_rlimits(monkeypatch):
    import subprocess as _sp

    captured: list[dict[str, tp.Any]] = []
    real_popen = _sp.Popen

    def _spy_popen(*args, **kwargs):
        captured.append({"args": args, "kwargs": dict(kwargs)})
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(_sp, "Popen", _spy_popen)
    sess = LocalSandboxSession(ContainerConfig(use_jail="off", memory_mb=512, timeout=5.0))
    try:
        res = sess.run_python(
            "import resource; print(resource.getrlimit(resource.RLIMIT_CPU)[0])",
            timeout=5.0,
        )
        assert res.ok is True
        assert int(res.stdout.strip()) <= 6
        assert len(captured) == 1
        assert captured[0]["kwargs"].get("preexec_fn") is None
        assert captured[0]["kwargs"].get("start_new_session") is True
    finally:
        sess.close()


@pytest.mark.parametrize("repo_kwargs", [{}, {"repo": "A.b-1/kiri_2"}])
def test_kiri_installer_mock_release_tarball(monkeypatch, tmp_path, repo_kwargs):
    from easydel.environments import install_kiri, kiri_installer

    slug = repo_kwargs.get("repo", kiri_installer.KIRI_GITHUB_REPO)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        payload = b"#!/bin/sh\necho kiri-0.2.1\n"
        info = tarfile.TarInfo(name="Kiri-0.2.1/kiri")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    tar_bytes = buf.getvalue()

    class _FakeResp:
        def __init__(self, data: bytes):
            self._data = data

        def read(self) -> bytes:
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def _fake_urlopen(req, timeout=20.0):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "api.github.com" in url:
            assert url == f"https://api.github.com/repos/{slug}/releases/tags/v0.2.1"
            meta = {
                "tag_name": "v0.2.1",
                "assets": [
                    {
                        "name": "Kiri-0.2.1-x86_64.tar.gz",
                        "browser_download_url": "https://example.invalid/kiri-x86_64.tar.gz",
                    },
                    {
                        "name": "Kiri-0.2.1-arm64.tar.gz",
                        "browser_download_url": "https://example.invalid/kiri-arm64.tar.gz",
                    },
                ],
            }
            return _FakeResp(json.dumps(meta).encode("utf-8"))
        return _FakeResp(tar_bytes)

    import hashlib

    tar_sha = hashlib.sha256(tar_bytes).hexdigest()
    monkeypatch.setattr(
        kiri_installer,
        "PINNED_KIRI_SHA256",
        {
            "v0.2.1": {
                "Kiri-0.2.1-x86_64.tar.gz": tar_sha,
                "Kiri-0.2.1-arm64.tar.gz": tar_sha,
            }
        },
    )
    monkeypatch.setattr(kiri_installer.urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(kiri_installer.shutil, "which", lambda _name: None)

    installed = install_kiri(install_dir=tmp_path, force=True, **repo_kwargs)
    assert installed == str(tmp_path / "kiri")
    assert os.access(installed, os.X_OK)
    assert b"kiri-0.2.1" in open(installed, "rb").read()

    # Corrupted / tampered download must fail SHA256 verification.
    monkeypatch.setattr(
        kiri_installer,
        "PINNED_KIRI_SHA256",
        {"v0.2.1": {"Kiri-0.2.1-x86_64.tar.gz": "0" * 64, "Kiri-0.2.1-arm64.tar.gz": "0" * 64}},
    )
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        install_kiri(install_dir=tmp_path, force=True, **repo_kwargs)

    # Also test pure-Python .deb `ar` extraction (Linux default asset).
    ar_header = b"data.tar.gz     0           0     0     100644  " + f"{len(tar_bytes):<10}".encode("ascii") + b"`\n"
    deb_bytes = b"!<arch>\n" + ar_header + tar_bytes + (b"\n" if len(tar_bytes) % 2 else b"")
    extracted = kiri_installer._extract_ar_data_tar(deb_bytes)
    assert extracted == tar_bytes


class _FlakyGate:
    """Verification-gate stub: raises for the first ``fail_times`` calls, then accepts."""

    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.calls = 0
        self._lock = threading.Lock()

    def verify(self, task):
        from easydel.environments import TaskSanityReport

        with self._lock:
            self.calls += 1
            failing = self.calls <= self.fail_times
        if failing:
            raise ConnectionError("gate backend hiccup")
        return TaskSanityReport(accepted=True, reason="ok")

    def stats(self):
        return {"accepted": max(0, self.calls - self.fail_times)}


def test_synthetic_stream_transient_gate_error_does_not_poison_stream():
    from easydel.environments import SyntheticTaskStream

    def _gen(seed: int):
        return {"task_id": "only", "prompt": "p"} if seed == 0 else None

    stream = SyntheticTaskStream(_gen, verification_gate=_FlakyGate(1), num_workers=1, generator_backoff=0.001)
    try:
        # The single real task hit the transient gate error; a later timeout is a TimeoutError, not that stale error.
        with pytest.raises(TimeoutError):
            stream.next_task(timeout=0.5)
    finally:
        stream.close()

    tasks = iter(range(100))

    def _gen2(seed: int):
        return {"task_id": f"t{next(tasks)}", "prompt": "p"}

    stream = SyntheticTaskStream(
        _gen2,
        verification_gate=_FlakyGate(2),
        num_workers=1,
        max_consecutive_generator_failures=3,
        generator_backoff=0.001,
    )
    try:
        assert stream.next_task(timeout=5.0)["task_id"] == "t2"
    finally:
        stream.close()


def test_synthetic_stream_persistent_gate_errors_stop_the_stream():
    from easydel.environments import SyntheticTaskStream

    gate = _FlakyGate(10**9)
    stream = SyntheticTaskStream(
        lambda seed: {"task_id": f"t{seed}", "prompt": "p"},
        verification_gate=gate,
        num_workers=1,
        max_consecutive_generator_failures=3,
        generator_backoff=0.001,
    )
    try:
        with pytest.raises(ConnectionError):
            stream.next_task(timeout=10.0)
        calls = gate.calls
        time.sleep(0.1)
        assert gate.calls == calls == 3
    finally:
        stream.close()
