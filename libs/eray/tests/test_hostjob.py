# Copyright 2026 The EasyDeL/eray Author @erfanzar (Erfan Zare Chavoshi).
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

"""`eray tpu job`: the generated per-host scripts, run for real with local bash."""

import os
import subprocess
import time

import pytest
from click.testing import CliRunner
from eray.cli import hostjob
from eray.cli.main import cli
from eray.cli.utils import TpuInfo


def _bash(script: str) -> tuple[int, dict[str, str], str]:
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    return result.returncode, hostjob.parse_report(result.stdout), result.stdout


def test_lifecycle_start_status_logs_stop(tmp_path):
    root = str(tmp_path / "jobs")
    rc, fields, _ = _bash(
        hostjob.start_script(
            "svc",
            ["bash", "-c", "echo hello $GREETING; sleep 60"],
            root=root,
            cwd=str(tmp_path),
            env={"GREETING": "world"},
            tpu_check=False,
        )
    )
    assert (rc, fields["state"]) == (0, "started")
    pid = fields["pid"]

    rc, fields, _ = _bash(hostjob.status_script("svc", root=root))
    assert (fields["state"], fields["pid"]) == ("running", pid)

    rc, fields, _ = _bash(hostjob.start_script("svc", ["true"], root=root, tpu_check=False))
    assert (rc, fields["state"], fields["error"]) == (3, "running", "already-running")

    time.sleep(0.2)
    assert "hello world" in _bash(hostjob.logs_script("svc", root=root))[2]
    assert (tmp_path / "jobs" / "svc" / "cmd").read_text().startswith("bash -c")

    rc, fields, _ = _bash(hostjob.stop_script("svc", root=root, grace_s=5, tpu_check=False))
    assert (rc, fields["state"], fields["forced"]) == (0, "stopped", "0")
    rc, fields, _ = _bash(hostjob.status_script("svc", root=root))
    assert fields["state"] in {"stopped", "exited"}
    with pytest.raises(ProcessLookupError):
        os.killpg(int(pid), 0)  # the whole process group must be gone


def test_stop_escalates_to_sigkill_for_a_term_ignoring_job(tmp_path):
    root = str(tmp_path / "jobs")
    rc, fields, _ = _bash(
        hostjob.start_script("stubborn", ["bash", "-c", "trap '' TERM; sleep 60 & wait"], root=root, tpu_check=False)
    )
    assert rc == 0, fields
    rc, fields, _ = _bash(hostjob.stop_script("stubborn", root=root, grace_s=0.5, tpu_check=False))
    assert (rc, fields["state"], fields["forced"]) == (0, "stopped", "1")


def test_a_command_that_exits_immediately_is_reported_with_its_code(tmp_path):
    root = str(tmp_path / "jobs")
    rc, fields, _ = _bash(hostjob.start_script("bad", ["bash", "-c", "exit 9"], root=root, tpu_check=False))
    assert (rc, fields["state"], fields["rc"]) == (6, "exited", "9")
    _, fields, _ = _bash(hostjob.status_script("bad", root=root))
    assert (fields["state"], fields["rc"]) == ("exited", "9")
    _, fields, _ = _bash(hostjob.status_script("never", root=root))
    assert fields["state"] == "absent"


def test_held_devices_block_start(tmp_path):
    device = tmp_path / "dev0"
    device.write_text("")
    holder = subprocess.Popen(["bash", "-c", f"exec 3<{device}; sleep 30"])
    try:
        time.sleep(0.3)
        script = hostjob.preflight_script("svc", root=str(tmp_path / "jobs"), device_glob=str(device))
        rc, fields, _ = _bash(script)
        if rc == 0:
            pytest.skip("fuser is not available on this machine")
        assert (rc, fields["state"], fields["error"]) == (4, "blocked", "tpu-busy")
        assert str(holder.pid) in fields["holders"].split(",")
    finally:
        holder.kill()
        holder.wait()
    rc, fields, _ = _bash(hostjob.preflight_script("svc", root=str(tmp_path / "jobs"), device_glob=str(device)))
    assert (rc, fields["state"]) == (0, "ready")


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: hostjob.job_dir("~/j", "bad name"), "job name"),
        (lambda: hostjob.job_dir("relative/root", "ok"), "absolute"),
        (lambda: hostjob.start_script("ok", [], tpu_check=False), "command"),
        (lambda: hostjob.start_script("ok", ["true"], env={"1BAD": "x"}, tpu_check=False), "environment"),
        (lambda: hostjob.preflight_script("ok", device_glob="/dev/x; rm -rf /"), "device glob"),
    ],
)
def test_unsafe_inputs_are_rejected(call, match):
    with pytest.raises(ValueError, match=match):
        call()


def test_arguments_are_quoted_not_interpreted(tmp_path):
    root = str(tmp_path / "jobs")
    marker = tmp_path / "pwned"
    rc, _, _ = _bash(
        hostjob.start_script(
            "q", ["bash", "-c", 'echo "$1"; sleep 5', "x", f"; touch {marker}"], root=root, tpu_check=False
        )
    )
    assert rc == 0
    time.sleep(0.3)
    assert not marker.exists()
    assert f"; touch {marker}" in _bash(hostjob.logs_script("q", root=root))[2]
    _bash(hostjob.stop_script("q", root=root, grace_s=2, tpu_check=False))


def test_cli_start_refuses_everything_when_one_host_is_not_ready(monkeypatch):
    tpu = TpuInfo.from_ips(["10.0.0.1", "10.0.0.2"], "v5p-16")
    calls: list[str] = []

    def fake_run(_tpu, script, **_kwargs):
        calls.append("preflight" if "state=ready" in script else "start")
        return [
            hostjob.HostReport(0, "10.0.0.1", 0, {"state": "ready"}),
            hostjob.HostReport(1, "10.0.0.2", 4, {"state": "blocked", "error": "tpu-busy", "holders": "123"}),
        ]

    monkeypatch.setattr("eray.cli.main._resolve_tpu", lambda *a: (tpu, None, None))
    monkeypatch.setattr(hostjob, "run_everywhere", fake_run)
    result = CliRunner().invoke(cli, ["tpu", "job", "start", "--name", "svc", "--", "python", "-c", "pass"])
    assert result.exit_code != 0
    assert "nothing started" in result.output
    assert "holders=123" in result.output
    assert calls == ["preflight"], "no host may start while another is not ready"


def test_sync_copies_to_a_different_path_on_this_host_and_skips_the_same_path(tmp_path, monkeypatch):
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "sub" / "serve.yaml").write_text("port: 8000\n")
    (src / "skip.log").write_text("x")
    monkeypatch.setattr(hostjob, "_local_ips", lambda: {"10.0.0.1"})
    tpu = TpuInfo.from_ips(["10.0.0.1"], "v5p-8")

    dest = tmp_path / "elsewhere" / "deploy"
    results = hostjob.sync_tree(tpu, str(src), str(dest), user="u", ssh_key="/nonexistent", excludes=["*.log"])
    assert [(index, result.returncode) for index, result in results] == [(0, 0)]
    assert (dest / "sub" / "serve.yaml").read_text() == "port: 8000\n"
    assert not (dest / "skip.log").exists()

    assert hostjob.sync_tree(tpu, str(src), str(src) + "/", user="u", ssh_key="/nonexistent") == []
