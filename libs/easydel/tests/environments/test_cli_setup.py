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

"""CPU tests for `easydel setup cpu-vm` and `easydel setup kiri` CLI commands."""

from __future__ import annotations

import pytest
from easydel.scripts import cli

pytestmark = pytest.mark.cpu_ok


def test_easydel_setup_cpu_vm_defaults_to_ssh_tunnel_and_accept_new(capsys):
    rc = cli.main(["setup", "cpu-vm", "10.128.0.11", "10.128.0.12", "--num-daemons", "2", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"$KIRI_BIN" daemon' in out
    assert "sha256sum -c -" in out
    assert "http://127.0.0.1:2375" in out
    assert "http://127.0.0.1:2376" in out
    assert "http://127.0.0.1:2377" in out
    assert "http://127.0.0.1:2378" in out
    ssh_cmd = cli._ssh_base_cmd(target="10.128.0.11", ssh_key=None)
    assert "StrictHostKeyChecking=accept-new" in ssh_cmd
    assert "UserKnownHostsFile=/dev/null" not in ssh_cmd


def test_easydel_setup_cpu_vm_expose_tcp_refuses_public_and_unspecified_ips_and_warns_private(capsys):
    with pytest.raises(ValueError, match="Refusing to expose"):
        cli.main(["setup", "cpu-vm", "8.8.8.8", "--expose-tcp", "--dry-run"])
    with pytest.raises(ValueError, match="Refusing to expose"):
        cli.main(["setup", "cpu-vm", "10.128.0.11", "--expose-tcp", "--bind-host", "0.0.0.0", "--dry-run"])

    rc = cli.main(["setup", "cpu-vm", "10.128.0.11", "--expose-tcp", "--num-daemons", "2", "--dry-run"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "UNAUTHENTICATED" in captured.err
    assert "http://10.128.0.11:2375" in captured.out
    assert "http://10.128.0.11:2376" in captured.out


def test_easydel_setup_cpu_vm_tunnel_resolves_remote_home(monkeypatch, capsys):
    calls: list[list[str]] = []

    def _fake_check_output(cmd, text=True):
        calls.append(list(cmd))
        return "/home/remoteuser\n"

    def _fake_run(cmd, check=True):
        calls.append(list(cmd))

    monkeypatch.setattr(cli.subprocess, "check_output", _fake_check_output)
    monkeypatch.setattr(cli.subprocess, "run", _fake_run)
    monkeypatch.setattr(cli.EngineSocketClient, "ping", lambda self: True)

    rc = cli.main(["setup", "cpu-vm", "erfan@10.128.0.20", "--port", "2400"])
    assert rc == 0
    tunnel_cmds = [c for c in calls if "-L" in c]
    assert len(tunnel_cmds) == 1
    l_arg = tunnel_cmds[0][tunnel_cmds[0].index("-L") + 1]
    assert l_arg == "127.0.0.1:2400:/home/remoteuser/.kiri/run/docker.sock"
    assert "$HOME" not in l_arg


def test_easydel_setup_kiri_local_shm_pool(monkeypatch, tmp_path, capsys):
    from easydel.environments import SandboxPool, kiri_installer

    spawned_envs: list[dict[str, str]] = []

    def _fake_popen(cmd, env=None, **kwargs):
        spawned_envs.append(dict(env or {}))
        sock_dir = cli.os.path.join(env["KIRI_DATA_DIR"], "run")
        cli.os.makedirs(sock_dir, exist_ok=True)
        with open(cli.os.path.join(sock_dir, "docker.sock"), "w", encoding="utf-8") as f:
            f.write("")

        class _Proc:
            pass

        return _Proc()

    monkeypatch.setattr(cli, "install_kiri", lambda **kw: "/usr/local/bin/kiri")
    monkeypatch.setattr(kiri_installer.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(cli.EngineSocketClient, "ping", lambda self: True)

    data_root = tmp_path / "kiri_shm"
    rc = cli.main(["setup", "kiri", "--data-dir", str(data_root), "--num-daemons", "2", "--start"])
    assert rc == 0
    out = capsys.readouterr().out
    assert str(data_root / "vm-0" / ".kiri" / "run" / "docker.sock") in out
    assert str(data_root / "vm-1" / ".kiri" / "run" / "docker.sock") in out
    assert len(spawned_envs) == 2
    assert spawned_envs[0]["HOME"] == str(data_root / "vm-0")
    assert spawned_envs[1]["KIRI_DATA_DIR"] == str(data_root / "vm-1" / ".kiri")

    pool = SandboxPool.from_local_shm(num_daemons=2, data_dir=str(data_root), auto_install=False)
    assert pool.size == 2


def test_sandbox_pool_auto_expands_shm_and_pod_shm_specs(monkeypatch, tmp_path):
    from easydel.environments import SandboxPool, kiri_installer

    monkeypatch.setattr(kiri_installer, "install_kiri", lambda **kw: "/usr/local/bin/kiri")
    monkeypatch.setattr(
        kiri_installer,
        "ensure_kiri_shm_pool",
        lambda num_daemons=4, **kw: [f"unix:///dev/shm/kiri/vm-{i}/.kiri/run/docker.sock" for i in range(num_daemons)],
    )
    monkeypatch.setattr(kiri_installer, "_start_unix_tcp_bridge", lambda sock, ip, port: f"http://{ip}:{port}")

    local_pool = SandboxPool(endpoints="shm:8", backend="kiri")
    assert local_pool.size == 8

    pod_pool = SandboxPool.from_pod_shm(
        num_daemons_per_host=8,
        data_dir=str(tmp_path),
        host_ips=["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"],
        auto_install=False,
    )
    assert pod_pool.size == 32
