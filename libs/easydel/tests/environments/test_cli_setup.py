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
    rc = cli.main(["setup", "cpu-vm", "10.0.0.11", "10.0.0.12", "--num-daemons", "2", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"$KIRI_BIN" daemon' in out
    assert "sha256sum -c -" in out
    assert "http://127.0.0.1:2375" in out
    assert "http://127.0.0.1:2376" in out
    assert "http://127.0.0.1:2377" in out
    assert "http://127.0.0.1:2378" in out
    ssh_cmd = cli._ssh_base_cmd(target="10.0.0.11", ssh_key=None)
    assert "StrictHostKeyChecking=accept-new" in ssh_cmd
    assert "UserKnownHostsFile=/dev/null" not in ssh_cmd


def test_easydel_setup_cpu_vm_has_no_remote_tcp_listener(capsys):
    # The unauthenticated remote TCP forwarder was removed: the flags are rejected outright.
    for flags in (["--expose-tcp"], ["--bind-host", "10.0.0.11"]):
        with pytest.raises(SystemExit):
            cli.main(["setup", "cpu-vm", "10.0.0.11", *flags, "--dry-run"])
    capsys.readouterr()

    rc = cli.main(["setup", "cpu-vm", "10.0.0.11", "--num-daemons", "2", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "TCP-LISTEN" not in out and "socat" not in out and "srv.bind" not in out
    assert "http://10.0.0.11" not in out
    assert "http://127.0.0.1:2375" in out and "http://127.0.0.1:2376" in out


@pytest.mark.parametrize(
    ("data_dir", "expected_root"),
    [("~/kiri", "$HOME/kiri"), ("~", "$HOME"), ("$HOME/x/", "$HOME/x"), ("/dev/shm/kiri", "/dev/shm/kiri")],
)
def test_cpu_vm_data_dir_tilde_becomes_remote_home(capsys, data_dir, expected_root):
    rc = cli.main(["setup", "cpu-vm", "10.0.0.11", "--num-daemons", "2", "--data-dir", data_dir, "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert f'mkdir -p "{expected_root}/vm-0/.kiri/run"' in out
    assert '"~' not in out


@pytest.mark.parametrize("data_dir", ["~other/kiri", "relative/dir", "/tmp/$(id)", '/tmp/a"b', "/tmp/`x`"])
def test_cpu_vm_data_dir_rejects_unsafe_forms(data_dir):
    with pytest.raises(ValueError, match="--data-dir"):
        cli.main(["setup", "cpu-vm", "10.0.0.11", "--data-dir", data_dir, "--dry-run"])


def test_cpu_vm_tunnel_resolves_tilde_data_dir_on_remote(monkeypatch):
    monkeypatch.setattr(cli.subprocess, "check_output", lambda cmd, text=True: "/home/remoteuser\n")
    path = cli._resolve_remote_sock_path(ssh_target="u@10.0.0.1", ssh_key=None, num_daemons=2, data_dir="~/pool", idx=1)
    assert path == "/home/remoteuser/pool/vm-1/.kiri/run/docker.sock"


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

    rc = cli.main(["setup", "cpu-vm", "user@10.0.0.20", "--port", "2400"])
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

    def _fake_shm_pool(num_daemons=4, **kw):
        return [f"unix:///dev/shm/kiri/vm-{i}/.kiri/run/docker.sock" for i in range(num_daemons)], []

    bridges: list[tuple[str, str, int, str | None, list[str] | None]] = []

    def _fake_bridge(sock, ip, port, *, token=None, allowed_peers=None):
        bridges.append((sock, ip, port, token, list(allowed_peers) if allowed_peers else None))
        return f"http://{ip}:{port}"

    monkeypatch.setattr(kiri_installer, "_ensure_shm_pool", _fake_shm_pool)
    monkeypatch.setattr(kiri_installer, "_start_unix_tcp_bridge", _fake_bridge)
    monkeypatch.setattr(kiri_installer, "_detect_private_ip", lambda: "10.0.0.1")

    local_pool = SandboxPool(endpoints="shm:8", backend="kiri")
    assert local_pool.size == 8

    hosts = ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"]
    # Cross-host pools are opt-in and require a shared bridge token.
    with pytest.raises(ValueError, match="expose_bridge"):
        SandboxPool.from_pod_shm(num_daemons_per_host=8, data_dir=str(tmp_path), host_ips=hosts)
    assert bridges == []

    token = "s3cret-token-0123456789"
    pod_pool = SandboxPool.from_pod_shm(
        num_daemons_per_host=8,
        data_dir=str(tmp_path),
        host_ips=hosts,
        expose_bridge=True,
        bridge_token=token,
    )
    assert pod_pool.size == 32
    assert len(bridges) == 8
    assert all(b[1] == "10.0.0.1" and b[3] == token and b[4] == hosts for b in bridges)

    # Without bridging, a pod spec stays host-local (no listener is started).
    bridges.clear()
    host_local = SandboxPool.from_pod_shm(num_daemons_per_host=4, data_dir=str(tmp_path))
    assert host_local.size == 4
    assert bridges == []
