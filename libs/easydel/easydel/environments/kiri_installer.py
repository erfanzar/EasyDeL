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

"""Automatic installer and daemon bootstrapper for Kiri (``github.com/libracoder/kiri``)."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import shutil
import stat
import subprocess
import tarfile
import threading
import time
import urllib.request

KIRI_GITHUB_REPO = "libracoder/kiri"
KIRI_RELEASES_API = f"https://api.github.com/repos/{KIRI_GITHUB_REPO}/releases"
DEFAULT_KIRI_VERSION = "v0.2.1"

PINNED_KIRI_SHA256: dict[str, dict[str, str]] = {
    "v0.2.1": {
        "kiri-lite_0.2.1_amd64.deb": "5e7b1f6d8239aa590b314fd2ccc8f1472ba1427f414b35efa86ccedd6a6571b2",
        "kiri-lite_0.2.1_arm64.deb": "76fe00facf4d61aefe90fbb0e44774942b1507e810c4e7b47ea34fddafe0193d",
        "Kiri-0.2.1-x86_64.tar.gz": "332d313ba59b2939b0bc76a9b55ac30f683537a34e6a652565c7c39f84b42209",
        "Kiri-0.2.1-arm64.tar.gz": "7be1c28b08f2e74ac2ef4f8972a3da951f89a4e35f10a461dcecb0fad9b94af2",
    }
}

_KIRI_BOOT_LOCK = threading.Lock()


def _normalize_arch(machine: str | None = None) -> tuple[str, str]:
    raw = (machine or platform.machine()).lower()
    if raw in {"x86_64", "amd64"}:
        return "x86_64", "amd64"
    if raw in {"aarch64", "arm64"}:
        return "arm64", "arm64"
    raise RuntimeError(f"Unsupported architecture for Kiri binary release: {raw!r}")


def _extract_ar_data_tar(deb_bytes: bytes) -> bytes:
    """Extract the ``data.tar*`` member from a Debian ``.deb`` ``ar`` archive in pure Python."""
    if not deb_bytes.startswith(b"!<arch>\n"):
        raise RuntimeError("Invalid .deb archive header.")
    pos = 8
    n = len(deb_bytes)
    while pos + 60 <= n:
        header = deb_bytes[pos : pos + 60]
        pos += 60
        name = header[:16].decode("ascii", errors="replace").strip().rstrip("/")
        size = int(header[48:58].decode("ascii", errors="replace").strip())
        member_data = deb_bytes[pos : pos + size]
        pos += size + (size % 2)
        if name.startswith("data.tar"):
            return member_data
    raise RuntimeError("No data.tar member found in .deb archive.")


def resolve_kiri_asset_url(
    *,
    version: str = DEFAULT_KIRI_VERSION,
    machine: str | None = None,
    system: str | None = None,
    timeout: float = 20.0,
) -> tuple[str, str, str, str]:
    """Resolve ``(tag_name, asset_name, download_url, sha256_hex)`` for ``libracoder/kiri``."""
    resolved_ver = DEFAULT_KIRI_VERSION if version in ("", "latest") else version
    arch_tar, arch_deb = _normalize_arch(machine)
    sys_name = (system or platform.system()).lower()
    api_url = f"{KIRI_RELEASES_API}/tags/{resolved_ver}"
    req = urllib.request.Request(
        api_url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "easydel-kiri-installer"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    tag = str(payload.get("tag_name") or resolved_ver)
    assets = payload.get("assets", [])
    pinned_for_tag = PINNED_KIRI_SHA256.get(tag, {})

    def _pick_digest(asset: dict[str, object], name: str) -> str:
        if name in pinned_for_tag:
            return pinned_for_tag[name]
        digest = str(asset.get("digest") or "")
        if digest.startswith("sha256:"):
            return digest.removeprefix("sha256:")
        raise RuntimeError(f"No pinned or release SHA256 digest available for {name!r} ({tag!r}).")

    if sys_name == "linux":
        deb_suffix = f"_{arch_deb}.deb"
        for asset in assets:
            name = str(asset.get("name", ""))
            url = str(asset.get("browser_download_url", ""))
            if name.startswith("kiri-lite_") and name.endswith(deb_suffix) and url:
                return tag, name, url, _pick_digest(asset, name)

    tar_suffix = f"-{arch_tar}.tar.gz"
    for asset in assets:
        name = str(asset.get("name", ""))
        url = str(asset.get("browser_download_url", ""))
        if name.startswith("Kiri-") and name.endswith(tar_suffix) and url:
            return tag, name, url, _pick_digest(asset, name)

    raise RuntimeError(f"No Kiri binary release found for {sys_name}/{arch_tar} in {tag!r}.")


def _extract_kiri_binary_from_tar(tar_stream: bytes, target_bin: str) -> str:
    with tarfile.open(fileobj=io.BytesIO(tar_stream), mode="r:*") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            if os.path.basename(member.name) != "kiri":
                continue
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            tmp_path = f"{target_bin}.tmp.{os.getpid()}"
            with open(tmp_path, "wb") as out_f:
                shutil.copyfileobj(extracted, out_f)
            os.chmod(
                tmp_path,
                stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH,
            )
            os.replace(tmp_path, target_bin)
            return target_bin
    raise RuntimeError("Kiri release archive did not contain a `kiri` binary.")


def install_kiri(
    *,
    version: str = DEFAULT_KIRI_VERSION,
    install_dir: str | os.PathLike[str] | None = None,
    force: bool = False,
    start_daemon: bool = False,
    timeout: float = 60.0,
) -> str:
    """Download, SHA256-verify, and install the ``kiri`` binary from ``libracoder/kiri``.

    Installs into ``install_dir`` (default ``~/.local/bin``) without requiring
    ``sudo``. Verifies the downloaded archive's SHA256 digest before extracting.
    """
    existing = shutil.which("kiri")
    if existing and not force:
        if start_daemon:
            ensure_kiri_daemon(kiri_bin=existing, auto_install=False)
        return existing

    target_dir = os.path.expanduser(os.fspath(install_dir) if install_dir is not None else "~/.local/bin")
    os.makedirs(target_dir, exist_ok=True)
    target_bin = os.path.join(target_dir, "kiri")
    if os.path.exists(target_bin) and not force:
        if start_daemon:
            ensure_kiri_daemon(kiri_bin=target_bin, auto_install=False)
        return target_bin

    _tag, asset_name, url, expected_sha256 = resolve_kiri_asset_url(version=version, timeout=min(timeout, 20.0))
    req = urllib.request.Request(url, headers={"User-Agent": "easydel-kiri-installer"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        archive_bytes = resp.read()

    actual_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    if actual_sha256.lower() != expected_sha256.lower():
        raise RuntimeError(f"SHA256 mismatch for {asset_name}: expected {expected_sha256}, got {actual_sha256}.")

    if asset_name.endswith(".deb"):
        data_tar = _extract_ar_data_tar(archive_bytes)
        installed_path = _extract_kiri_binary_from_tar(data_tar, target_bin)
    else:
        installed_path = _extract_kiri_binary_from_tar(archive_bytes, target_bin)

    if start_daemon:
        ensure_kiri_daemon(kiri_bin=installed_path, auto_install=False)
    return installed_path


def ensure_kiri_daemon(
    *,
    kiri_bin: str | None = None,
    auto_install: bool = True,
    wait_seconds: float = 15.0,
    data_dir: str | os.PathLike[str] | None = None,
) -> str | None:
    """Ensure the Kiri daemon is running and return its ``unix://`` socket URI.

    When ``data_dir`` is provided (for example ``"/dev/shm/kiri"`` on TPU hosts
    with large CPU RAM and small root SSDs), ``HOME`` and ``KIRI_DATA_DIR`` are
    directed into ``<data_dir>`` so all container images, overlays, and sockets
    live in RAM-backed tmpfs with zero root-SSD usage.
    """
    from .runtime import EngineSocketClient

    if data_dir is not None:
        base_home = os.path.abspath(os.path.expanduser(os.fspath(data_dir)))
        kiri_data = os.path.join(base_home, ".kiri")
        candidates = [os.path.join(kiri_data, "run", "docker.sock")]
    else:
        base_home = None
        kiri_data = None
        candidates = [
            "/dev/shm/kiri/.kiri/run/docker.sock",
            os.path.expanduser("~/.kiri/run/docker.sock"),
            "/run/kiri/docker.sock",
        ]

    for sock_path in candidates:
        if os.path.exists(sock_path):
            uri = f"unix://{sock_path}"
            if EngineSocketClient(uri).ping():
                return uri

    with _KIRI_BOOT_LOCK:
        for sock_path in candidates:
            if os.path.exists(sock_path):
                uri = f"unix://{sock_path}"
                if EngineSocketClient(uri).ping():
                    return uri

        exe = kiri_bin or shutil.which("kiri")
        if exe is None:
            local_kiri = os.path.expanduser("~/.local/bin/kiri")
            if os.path.exists(local_kiri):
                exe = local_kiri
            elif auto_install:
                exe = install_kiri()
            else:
                return None

        daemon_env = dict(os.environ)
        if base_home is not None and kiri_data is not None:
            os.makedirs(os.path.join(kiri_data, "run"), exist_ok=True)
            daemon_env["HOME"] = base_home
            daemon_env["KIRI_DATA_DIR"] = kiri_data

        subprocess.Popen(
            [exe, "daemon"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=daemon_env,
            start_new_session=True,
        )

        deadline = time.monotonic() + max(1.0, float(wait_seconds))
        while time.monotonic() < deadline:
            for sock_path in candidates:
                if os.path.exists(sock_path):
                    uri = f"unix://{sock_path}"
                    if EngineSocketClient(uri).ping():
                        return uri
            time.sleep(0.1)
    return None


def parse_shm_endpoint_spec(spec: str) -> tuple[str, int] | None:
    """Parse ``"shm"``, ``"shm:N"``, or ``"pod-shm:N"`` into ``(scope, count)``."""
    raw = spec.strip().lower()
    if raw in {"shm", "local-shm"}:
        return ("local", 4)
    if raw.startswith(("shm:", "local-shm:")):
        _, num_str = raw.split(":", 1)
        return ("local", max(1, int(num_str)))
    if raw == "pod-shm":
        return ("pod", 8)
    if raw.startswith("pod-shm:"):
        _, num_str = raw.split(":", 1)
        return ("pod", max(1, int(num_str)))
    return None


def _detect_private_ip() -> str:
    """Return this host's primary non-loopback IPv4 address."""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.255.255.255", 1))
            ip = probe.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
    except OSError:
        pass
    try:
        resolved = socket.gethostbyname(socket.gethostname())
        if resolved and not resolved.startswith("127."):
            return resolved
    except OSError:
        pass
    return "127.0.0.1"


_TCP_BRIDGES: dict[tuple[str, int], threading.Thread] = {}
_TCP_BRIDGE_LOCK = threading.Lock()


def _start_unix_tcp_bridge(sock_path: str, bind_host: str, port: int) -> str:
    """Start an in-process background TCP->Unix bridge for pod-wide Kiri access."""
    import socket

    endpoint = f"http://{bind_host}:{port}"
    key = (bind_host, int(port))
    with _TCP_BRIDGE_LOCK:
        existing = _TCP_BRIDGES.get(key)
        if existing is not None and existing.is_alive():
            return endpoint

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind((bind_host, int(port)))
        except OSError:
            srv.close()
            return endpoint
        srv.listen(128)

        def _pipe(src: socket.socket, dst: socket.socket) -> None:
            try:
                while True:
                    chunk = src.recv(65536)
                    if not chunk:
                        break
                    dst.sendall(chunk)
            except OSError:
                pass
            finally:
                for s in (src, dst):
                    try:
                        s.close()
                    except OSError:
                        pass

        def _serve() -> None:
            while True:
                try:
                    client, _ = srv.accept()
                except OSError:
                    break
                try:
                    unix_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    unix_sock.connect(sock_path)
                except OSError:
                    client.close()
                    continue
                threading.Thread(target=_pipe, args=(client, unix_sock), daemon=True).start()
                threading.Thread(target=_pipe, args=(unix_sock, client), daemon=True).start()

        t = threading.Thread(target=_serve, name=f"kiri-tcp-bridge-{port}", daemon=True)
        t.start()
        _TCP_BRIDGES[key] = t
    return endpoint


def ensure_kiri_shm_pool(
    num_daemons: int = 4,
    *,
    data_dir: str | os.PathLike[str] = "/dev/shm/kiri",
    kiri_bin: str | None = None,
    auto_install: bool = True,
    wait_seconds: float = 15.0,
) -> list[str]:
    """Start or reuse ``num_daemons`` local Kiri daemons backed by ``/dev/shm`` tmpfs."""
    import concurrent.futures

    count = max(1, int(num_daemons))
    root = os.path.abspath(os.path.expanduser(os.fspath(data_dir)))

    exe = kiri_bin or shutil.which("kiri")
    if exe is None:
        local_kiri = os.path.expanduser("~/.local/bin/kiri")
        if os.path.exists(local_kiri):
            exe = local_kiri
        elif auto_install:
            exe = install_kiri()

    def _boot_one(idx: int) -> str:
        vm_dir = root if count == 1 else os.path.join(root, f"vm-{idx}")
        uri = ensure_kiri_daemon(
            kiri_bin=exe,
            auto_install=False,
            wait_seconds=wait_seconds,
            data_dir=vm_dir,
        )
        if uri is None:
            raise RuntimeError(f"Failed to start Kiri daemon {idx} under {vm_dir!r}.")
        return uri

    if count == 1:
        return [_boot_one(0)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(count, 16)) as pool:
        return list(pool.map(_boot_one, range(count)))


def ensure_kiri_pod_shm_pool(
    num_daemons_per_host: int = 8,
    *,
    data_dir: str | os.PathLike[str] = "/dev/shm/kiri",
    base_port: int = 2375,
    host_ips: list[str] | tuple[str, ...] | None = None,
    kiri_bin: str | None = None,
    auto_install: bool = True,
    wait_seconds: float = 15.0,
) -> list[str]:
    """Start ``num_daemons_per_host`` RAM-backed Kiri daemons on this host and gather pod endpoints.

    On a 4-host ``v5p-32`` pod with ``num_daemons_per_host=8``, each TPU host
    boots 8 local Kiri daemons in ``/dev/shm/kiri/vm-0..7`` (using ~26 vCPUs and
    RAM-backed tmpfs per daemon), bridges them on its private VPC IP, and
    exchanges endpoints across JAX hosts when ``jax.process_count() > 1`` so all
    32 Kiri daemons are reachable as one unified pool.
    """
    local_uris = ensure_kiri_shm_pool(
        num_daemons=num_daemons_per_host,
        data_dir=data_dir,
        kiri_bin=kiri_bin,
        auto_install=auto_install,
        wait_seconds=wait_seconds,
    )
    my_ip = _detect_private_ip()
    local_http = [
        _start_unix_tcp_bridge(uri.removeprefix("unix://"), my_ip, base_port + idx)
        for idx, uri in enumerate(local_uris)
    ]

    if host_ips:
        return [
            f"http://{ip}:{base_port + idx}"
            for ip in host_ips
            for idx in range(len(local_uris))
        ]

    try:
        import jax

        n_procs = jax.process_count()
    except Exception:
        n_procs = 1

    if n_procs <= 1:
        return local_uris

    import numpy as np
    from jax.experimental import multihost_utils

    ip_bytes = my_ip.encode("utf-8")[:64].ljust(64, b"\x00")
    local_arr = np.frombuffer(ip_bytes, dtype=np.uint8)[None, :]
    gathered = np.asarray(multihost_utils.process_allgather(local_arr, tiled=True))
    all_ips: list[str] = []
    for row in gathered:
        decoded = bytes(row.tolist()).rstrip(b"\x00").decode("utf-8", errors="ignore")
        if decoded and decoded not in all_ips:
            all_ips.append(decoded)
    if not all_ips:
        return local_http
    return [
        f"http://{ip}:{base_port + idx}"
        for ip in all_ips
        for idx in range(len(local_uris))
    ]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``easydel-install-kiri``."""
    parser = argparse.ArgumentParser(description="Install the Kiri container runtime (libracoder/kiri).")
    parser.add_argument(
        "--version",
        default=DEFAULT_KIRI_VERSION,
        help=f"Pinned GitHub release tag (default: {DEFAULT_KIRI_VERSION}).",
    )
    parser.add_argument("--install-dir", default=None, help="Target directory (default: ~/.local/bin).")
    parser.add_argument("--force", action="store_true", help="Reinstall even if kiri is already present.")
    parser.add_argument("--start", action="store_true", help="Start `kiri daemon` in the background after install.")
    args = parser.parse_args(argv)

    path = install_kiri(
        version=args.version,
        install_dir=args.install_dir,
        force=args.force,
        start_daemon=args.start,
    )
    print(f"Installed Kiri at: {path}")
    return 0


__all__ = (
    "DEFAULT_KIRI_VERSION",
    "KIRI_GITHUB_REPO",
    "PINNED_KIRI_SHA256",
    "ensure_kiri_daemon",
    "install_kiri",
    "main",
    "resolve_kiri_asset_url",
)
