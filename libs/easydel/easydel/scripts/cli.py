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

"""Top-level ``easydel`` CLI (``easydel setup cpu-vm ...``, ``easydel setup kiri``, ...)."""

from __future__ import annotations

import argparse
import base64
import ipaddress
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

from easydel.environments.kiri_installer import (
    DEFAULT_KIRI_VERSION,
    add_kiri_repo_argument,
    install_kiri,
    resolve_kiri_asset_url,
    validate_kiri_download_url,
    validate_kiri_repo,
    validate_kiri_version,
)
from easydel.environments.runtime import EngineSocketClient

_UNSAFE_REMOTE_PATH_CHARS = frozenset('"`\\$:\n\r\x00')
_REMOTE_CURL_MAX_BYTES = 256 * 1024 * 1024


def _validate_ssh_target(target: str) -> str:
    """Validate an SSH target (``host`` or ``user@host``), blocking option injection."""
    if not isinstance(target, str):
        raise ValueError(f"Invalid SSH target {target!r}: expected a string.")
    raw = target.strip()
    if not raw or any(ord(ch) < 0x20 or ord(ch) == 0x7F or ch.isspace() for ch in raw):
        raise ValueError(f"Invalid SSH target {target!r}: empty or contains whitespace/control characters.")
    if raw.startswith("-"):
        raise ValueError(f"Invalid SSH target {target!r}: must not start with '-'.")
    if "@" in raw:
        user, host = raw.split("@", 1)
        if not user or not host or user.startswith("-") or host.startswith("-") or "@" in host:
            raise ValueError(f"Invalid SSH target {target!r}: malformed user@host.")
    return raw


def _validate_ssh_user(user: str | None) -> str | None:
    if user is None:
        return None
    raw = user.strip()
    if not raw or raw.startswith("-") or "@" in raw or any(ord(ch) < 0x20 or ord(ch) == 0x7F or ch.isspace() for ch in raw):
        raise ValueError(f"Invalid --user {user!r}.")
    return raw


def _normalize_remote_data_dir(data_dir: str | None) -> str | None:
    """Normalize a remote ``--data-dir`` so the remote shell resolves the home directory.

    A leading ``~`` / ``~/`` becomes ``$HOME`` / ``$HOME/`` (``~`` is never expanded
    inside the double quotes the provisioning script uses). ``~user`` paths, colons
    (which break OpenSSH ``-L port:sock_path``), and shell metacharacters are rejected.

    Raises:
        ValueError: For ``~user`` forms, unsafe characters, colons, or relative paths.
    """
    if data_dir is None:
        return None
    raw = data_dir.strip()
    if raw == "~":
        raw = "$HOME"
    elif raw.startswith("~/"):
        raw = "$HOME/" + raw[2:]
    elif raw.startswith("~"):
        raise ValueError(f"--data-dir {data_dir!r}: '~user' paths are not supported; use an absolute path.")
    rest = raw[len("$HOME") :] if raw == "$HOME" or raw.startswith("$HOME/") else raw
    if any(ch in _UNSAFE_REMOTE_PATH_CHARS for ch in rest):
        raise ValueError(f"--data-dir {data_dir!r} contains unsafe characters or ':'.")
    if rest is raw and not raw.startswith("/"):
        raise ValueError(f"--data-dir {data_dir!r} must be absolute, '~/...' or '$HOME/...'.")
    return raw.rstrip("/") or "/"


def _format_http_endpoint(host: str, port: int) -> str:
    try:
        ip = ipaddress.ip_address(host)
        if ip.version == 6:
            return f"http://[{host}]:{port}"
    except ValueError:
        pass
    return f"http://{host}:{port}"


def _build_remote_kiri_script(
    *,
    deb_url_amd64: str,
    deb_sha_amd64: str,
    deb_url_arm64: str,
    deb_sha_arm64: str,
    num_daemons: int,
    data_dir: str | None,
) -> str:
    """Build a self-contained bash script that installs Kiri + boots N daemons on a CPU VM.

    The daemons listen only on their Unix sockets; nothing is bound to TCP on
    the remote host. Clients reach them through SSH-forwarded local ports.
    """
    if num_daemons <= 0:
        raise ValueError("--num-daemons must be positive.")
    validate_kiri_download_url(deb_url_amd64)
    validate_kiri_download_url(deb_url_arm64)
    data_dir = _normalize_remote_data_dir(data_dir)
    data_root = data_dir or "$HOME/.kiri-vms"
    lines = [
        "set -euo pipefail",
        'ARCH="$(uname -m)"',
        'case "$ARCH" in',
        f"  x86_64|amd64) URL={shlex.quote(deb_url_amd64)}; SHA={shlex.quote(deb_sha_amd64)} ;;",
        f"  aarch64|arm64) URL={shlex.quote(deb_url_arm64)}; SHA={shlex.quote(deb_sha_arm64)} ;;",
        '  *) echo "Unsupported arch: $ARCH" >&2; exit 1 ;;',
        "esac",
        'mkdir -p "$HOME/.local/bin"',
        'TMP_DEB=""',
        'TMP_DIR=""',
        'TMP_BIN=""',
        "cleanup() {",
        '  [ -n "${TMP_DEB:-}" ] && rm -rf "$TMP_DEB"',
        '  [ -n "${TMP_DIR:-}" ] && rm -rf "$TMP_DIR"',
        '  [ -n "${TMP_BIN:-}" ] && rm -rf "$TMP_BIN"',
        "  return 0",
        "}",
        "trap cleanup EXIT",
        'if [ ! -x "$HOME/.local/bin/kiri" ]; then',
        '  TMP_DEB="$(mktemp /tmp/kiri-lite.XXXXXX.deb)"',
        f'  curl -fsSL --proto \'=https\' --proto-redir \'=https\' --tlsv1.2 --max-filesize {_REMOTE_CURL_MAX_BYTES} "$URL" -o "$TMP_DEB"',
        '  echo "$SHA  $TMP_DEB" | sha256sum -c -',
        '  TMP_DIR="$(mktemp -d /tmp/kiri-extract.XXXXXX)"',
        '  TMP_BIN="$(mktemp "$HOME/.local/bin/.kiri.XXXXXX")"',
        '  (cd "$TMP_DIR" && ar x "$TMP_DEB" && tar -xf data.tar* --wildcards "*/kiri" -O > "$TMP_BIN")',
        '  chmod +x "$TMP_BIN"',
        '  mv -f "$TMP_BIN" "$HOME/.local/bin/kiri"',
        '  TMP_BIN=""',
        '  rm -rf "$TMP_DEB" "$TMP_DIR"',
        '  TMP_DEB=""',
        '  TMP_DIR=""',
        "fi",
        'KIRI_BIN="$HOME/.local/bin/kiri"',
        "_kiri_sock_alive() {",
        '  [ -S "$1" ] && curl -fsS --max-time 2 --unix-socket "$1" http://localhost/_ping >/dev/null 2>&1',
        "}",
    ]

    for idx in range(num_daemons):
        if num_daemons == 1 and data_dir is None:
            daemon_home = "$HOME"
            sock_path = "$HOME/.kiri/run/docker.sock"
            log_path = "/tmp/kiri-daemon-0.log"
            kiri_data = "$HOME/.kiri"
        else:
            daemon_home = f"{data_root}/vm-{idx}"
            sock_path = f"{daemon_home}/.kiri/run/docker.sock"
            log_path = f"/tmp/kiri-daemon-{idx}.log"
            kiri_data = f"{daemon_home}/.kiri"

        lines.extend(
            [
                f'mkdir -p "{daemon_home}/.kiri/run" "{kiri_data}"',
                f'if ! _kiri_sock_alive "{sock_path}"; then',
                f'  rm -f "{sock_path}"',
                f'  HOME="{daemon_home}" KIRI_DATA_DIR="{kiri_data}" setsid nohup "$KIRI_BIN" daemon > "{log_path}" 2>&1 < /dev/null &',
                f'  for _i in $(seq 1 100); do if _kiri_sock_alive "{sock_path}"; then break; fi; sleep 0.1; done',
                f'  if ! _kiri_sock_alive "{sock_path}"; then',
                f'    echo "Kiri daemon failed to answer /_ping on {sock_path}" >&2',
                "    exit 1",
                "  fi",
                "fi",
            ]
        )
    lines.append('echo "KIRI_CPU_VM_READY"')
    return "\n".join(lines) + "\n"


def _resolve_ssh_key(ssh_key: str | None) -> str | None:
    if ssh_key:
        return ssh_key
    gce_key = os.path.expanduser("~/.ssh/google_compute_engine")
    if os.path.exists(gce_key):
        return gce_key
    return None


def _ssh_base_cmd(*, target: str, ssh_key: str | None) -> list[str]:
    safe_target = _validate_ssh_target(target)
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
    resolved_key = _resolve_ssh_key(ssh_key)
    if resolved_key:
        cmd.extend(["-i", resolved_key])
    cmd.extend(["--", safe_target])
    return cmd


def _resolve_remote_sock_path(
    *,
    ssh_target: str,
    ssh_key: str | None,
    num_daemons: int,
    data_dir: str | None,
    idx: int,
) -> str:
    """Resolve the absolute remote Unix socket path (expanding ``$HOME`` on the remote host)."""
    data_dir = _normalize_remote_data_dir(data_dir)
    if data_dir and not data_dir.startswith("$HOME"):
        return f"{data_dir}/vm-{idx}/.kiri/run/docker.sock"
    remote_home = subprocess.check_output(
        [*_ssh_base_cmd(target=ssh_target, ssh_key=ssh_key), 'printf %s "$HOME"'],
        text=True,
    ).strip()
    if not remote_home.startswith("/") or ":" in remote_home:
        raise RuntimeError(f"Unexpected remote $HOME from {ssh_target}: {remote_home!r}")
    if num_daemons == 1 and not data_dir:
        return f"{remote_home}/.kiri/run/docker.sock"
    if data_dir:
        resolved_data_dir = remote_home + data_dir[len("$HOME") :]
    else:
        resolved_data_dir = f"{remote_home}/.kiri-vms"
    return f"{resolved_data_dir}/vm-{idx}/.kiri/run/docker.sock"


def _close_ssh_tunnels(opened_tunnels: list[tuple[str, str]], ctl_dir: str | None) -> None:
    """Tear down any background ``ssh -fN`` tunnels opened via ControlMaster sockets."""
    for ctl_path, target in opened_tunnels:
        try:
            subprocess.run(["ssh", "-S", ctl_path, "-O", "exit", "--", target], check=False)
        except Exception:
            pass
    if ctl_dir is not None:
        shutil.rmtree(ctl_dir, ignore_errors=True)


def cmd_setup_cpu_vm(args: argparse.Namespace) -> int:
    """Provision one or more remote CPU VMs with Kiri and return their SandboxPool endpoints."""
    if args.num_daemons <= 0:
        raise ValueError("--num-daemons must be positive.")

    repo = validate_kiri_repo(args.repo)
    ver = validate_kiri_version(args.kiri_version)
    args.data_dir = _normalize_remote_data_dir(args.data_dir)
    safe_user = _validate_ssh_user(args.user)
    targets = [
        _validate_ssh_target(raw_host if "@" in raw_host or not safe_user else f"{safe_user}@{raw_host}")
        for raw_host in args.hosts
    ]
    if args.dry_run:
        from easydel.environments.kiri_installer import PINNED_KIRI_SHA256

        ver_num = ver.lstrip("v")
        pinned = PINNED_KIRI_SHA256.get(ver, {})
        amd64_name = f"kiri-lite_{ver_num}_amd64.deb"
        arm64_name = f"kiri-lite_{ver_num}_arm64.deb"
        release_url = f"https://github.com/{repo}/releases/download/{ver}"
        deb_amd64 = f"{release_url}/{amd64_name}"
        sha_amd64 = pinned.get(amd64_name, args.sha256_amd64 or "<sha256-unavailable>")
        deb_arm64 = f"{release_url}/{arm64_name}"
        sha_arm64 = pinned.get(arm64_name, args.sha256_arm64 or "<sha256-unavailable>")
    else:
        _, _, deb_amd64, sha_amd64 = resolve_kiri_asset_url(
            repo=repo, version=ver, machine="x86_64", system="linux", expected_sha256=args.sha256_amd64
        )
        _, _, deb_arm64, sha_arm64 = resolve_kiri_asset_url(
            repo=repo, version=ver, machine="arm64", system="linux", expected_sha256=args.sha256_arm64
        )

    endpoints: list[str] = []
    opened_tunnels: list[tuple[str, str]] = []
    ctl_dir: str | None = None
    if not args.dry_run:
        ctl_dir = tempfile.mkdtemp(prefix="ed-ssh-", dir="/tmp")

    try:
        for host_idx, ssh_target in enumerate(targets):
            script = _build_remote_kiri_script(
                deb_url_amd64=deb_amd64,
                deb_sha_amd64=sha_amd64,
                deb_url_arm64=deb_arm64,
                deb_sha_arm64=sha_arm64,
                num_daemons=args.num_daemons,
                data_dir=args.data_dir,
            )
            encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
            remote_cmd = f"echo {shlex.quote(encoded)} | base64 -d | bash"
            ssh_cmd = [*_ssh_base_cmd(target=ssh_target, ssh_key=args.ssh_key), remote_cmd]

            if args.dry_run:
                print(f"[dry-run] {ssh_target}:")
                print(script)
                for idx in range(args.num_daemons):
                    endpoints.append(_format_http_endpoint("127.0.0.1", args.port + host_idx * args.num_daemons + idx))
                continue

            print(f"Provisioning Kiri on {ssh_target}...", flush=True)
            subprocess.run(ssh_cmd, check=True)

            for idx in range(args.num_daemons):
                local_port = args.port + host_idx * args.num_daemons + idx
                sock_path = _resolve_remote_sock_path(
                    ssh_target=ssh_target,
                    ssh_key=args.ssh_key,
                    num_daemons=args.num_daemons,
                    data_dir=args.data_dir,
                    idx=idx,
                )
                resolved_key = _resolve_ssh_key(args.ssh_key)
                assert ctl_dir is not None
                ctl_path = os.path.join(ctl_dir, f"t-{host_idx}-{idx}.sock")
                tunnel_cmd = [
                    "ssh",
                    "-fN",
                    "-M",
                    "-S",
                    ctl_path,
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ExitOnForwardFailure=yes",
                    "-o",
                    "StrictHostKeyChecking=accept-new",
                    *(["-i", resolved_key] if resolved_key else []),
                    "-L",
                    f"127.0.0.1:{local_port}:{sock_path}",
                    "--",
                    ssh_target,
                ]
                subprocess.run(tunnel_cmd, check=True)
                opened_tunnels.append((ctl_path, ssh_target))
                endpoint = _format_http_endpoint("127.0.0.1", local_port)
                client = EngineSocketClient(endpoint)
                ok = False
                for _ in range(30):
                    if client.ping():
                        ok = True
                        break
                    time.sleep(0.2)
                if not ok:
                    raise RuntimeError(f"Kiri endpoint {endpoint} did not answer /_ping after provisioning {ssh_target}.")
                endpoints.append(endpoint)
    except BaseException:
        _close_ssh_tunnels(opened_tunnels, ctl_dir)
        raise

    if ctl_dir is not None:
        try:
            os.rmdir(ctl_dir)
        except OSError:
            pass

    print("\nKiri CPU VM pool ready!")
    print(f"endpoints = {endpoints!r}")
    return 0


def cmd_setup_kiri(args: argparse.Namespace) -> int:
    """Install and optionally start one or more Kiri daemons locally (e.g. in /dev/shm)."""
    from easydel.environments.kiri_installer import ensure_kiri_shm_pool

    if args.num_daemons <= 0:
        raise ValueError("--num-daemons must be positive.")
    version = validate_kiri_version(args.version)

    path = install_kiri(
        version=version,
        repo=args.repo,
        install_dir=args.install_dir,
        force=args.force,
        start_daemon=bool(args.start and args.data_dir is None and args.num_daemons == 1),
        expected_sha256=args.sha256,
    )
    print(f"Kiri ready at: {path}")
    if args.start and (args.data_dir is not None or args.num_daemons > 1):
        target_dir = args.data_dir or "/dev/shm/kiri"
        endpoints = ensure_kiri_shm_pool(
            num_daemons=args.num_daemons,
            data_dir=target_dir,
            kiri_bin=path,
            auto_install=False,
            persist=True,
        )
        print(f"endpoints = {endpoints!r}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="easydel", description="EasyDeL CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    setup_parser = sub.add_parser("setup", help="Provision local or remote runtimes.")
    setup_sub = setup_parser.add_subparsers(dest="setup_target", required=True)

    cpu_vm = setup_sub.add_parser(
        "cpu-vm",
        help="Install & start Kiri on remote CPU VM(s) over SSH; endpoints are SSH tunnels (nothing listens on remote TCP).",
    )
    add_kiri_repo_argument(cpu_vm)
    cpu_vm.add_argument("hosts", nargs="+", help="Target CPU VM IP(s) or user@host.")
    cpu_vm.add_argument("--user", default=None, help="SSH username if not included in host.")
    cpu_vm.add_argument("--ssh-key", default=None, help="Path to SSH private key.")
    cpu_vm.add_argument(
        "--port",
        type=int,
        default=2375,
        help="Base local port for the SSH-forwarded Kiri Engine API endpoints (default: 2375).",
    )
    cpu_vm.add_argument(
        "--num-daemons",
        type=int,
        default=1,
        help="Number of independent Kiri daemons to start per CPU VM (default: 1).",
    )
    cpu_vm.add_argument("--data-dir", default=None, help="Base KIRI_DATA_DIR on remote CPU VM.")
    cpu_vm.add_argument(
        "--kiri-version",
        default=DEFAULT_KIRI_VERSION,
        help=f"Kiri release tag (default: {DEFAULT_KIRI_VERSION}); unpinned tags also need --sha256-amd64/-arm64.",
    )
    cpu_vm.add_argument("--sha256-amd64", default=None, help="Expected SHA256 of the amd64 .deb (unpinned tags).")
    cpu_vm.add_argument("--sha256-arm64", default=None, help="Expected SHA256 of the arm64 .deb (unpinned tags).")
    cpu_vm.add_argument("--dry-run", action="store_true", help="Print provisioning script without running SSH.")
    cpu_vm.set_defaults(func=cmd_setup_cpu_vm)

    kiri_local = setup_sub.add_parser("kiri", help="Install (and optionally start) Kiri on this host.")
    kiri_local.add_argument(
        "--version",
        default=DEFAULT_KIRI_VERSION,
        help=f"Kiri release tag (default: {DEFAULT_KIRI_VERSION}); unpinned tags also need --sha256.",
    )
    add_kiri_repo_argument(kiri_local)
    kiri_local.add_argument("--sha256", default=None, help="Expected SHA256 of the release asset (unpinned tags).")
    kiri_local.add_argument("--install-dir", default=None, help="Target directory (default: ~/.local/bin).")
    kiri_local.add_argument("--data-dir", default=None, help="Optional RAM-backed data dir (e.g. /dev/shm/kiri).")
    kiri_local.add_argument("--num-daemons", type=int, default=1, help="Number of local Kiri daemons to start.")
    kiri_local.add_argument("--force", action="store_true", help="Reinstall even if already present.")
    kiri_local.add_argument("--start", action="store_true", help="Start `kiri daemon` after install.")
    kiri_local.set_defaults(func=cmd_setup_kiri)

    elarge_p = sub.add_parser("elarge", help="Run EasyDeL eLarge YAML train/eval/serve runner.")
    elarge_p.add_argument("elarge_args", nargs=argparse.REMAINDER)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if argv_list and argv_list[0] == "elarge":
        from easydel.scripts.elarge import main as elarge_main

        return int(elarge_main(argv_list[1:]) or 0)
    parser = build_parser()
    args = parser.parse_args(argv_list)
    if args.command == "elarge":
        from easydel.scripts.elarge import main as elarge_main

        sys.argv = ["easydel-elarge", *args.elarge_args]
        elarge_main()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
