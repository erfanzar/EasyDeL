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

"""Installer, daemon lifecycle and pod bridge for the Kiri container engine (release repo: ``KIRI_GITHUB_REPO``).

Installing Kiri downloads a SHA256-verified release (pinned digests for the tags
in ``PINNED_KIRI_SHA256``, a caller-supplied digest for any other tag) and only happens on
an explicit request (``easydel setup kiri``, ``easydel-install-kiri`` or
``auto_install=True``). Daemons booted through :func:`ensure_kiri_daemon` or
:func:`ensure_kiri_shm_pool` are owned by this process: they are tracked in a
module registry and stopped by :func:`shutdown_kiri_daemons` (also registered
with :mod:`atexit`), which never signals a daemon this process did not start.
``persist=True`` leaves a daemon running after the process exits (the CLI uses
it).

Cross-host access goes through an in-process TCP bridge. Bridging is opt-in and
binds loopback by default. A bridge bound to a routable address requires a
shared secret token sent as the ``X-EasyDeL-Sandbox-Token`` header, accepts
peers only from an allow-list of host IPs, and refuses container/exec requests
that would escape the sandbox (privileged mode, host binds or mounts, host
namespaces, devices, added capabilities).
"""

from __future__ import annotations

import argparse
import atexit
import dataclasses
import hashlib
import hmac
import io
import ipaddress
import json
import os
import platform
import re
import shutil
import socket
import stat
import subprocess
import tarfile
import threading
import time
import typing as tp
import urllib.parse
import urllib.request

from .runtime import (
    SANDBOX_TOKEN_HEADER,
    EngineSocketClient,
    KiriDaemonBudget,
    SandboxInfraError,
    _terminate_process_groups,
)

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
_MISSING_KIRI_MESSAGE = (
    "No `kiri` binary was found and automatic installation is disabled. Install it explicitly with "
    "`easydel setup kiri` (or `easydel-install-kiri`), pass `kiri_bin=`, or pass `auto_install=True` to "
    f"download the pinned, SHA256-verified {DEFAULT_KIRI_VERSION} release."
)
_MIN_BRIDGE_TOKEN_CHARS = 16
_GITHUB_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def validate_kiri_repo(repo: str) -> str:
    """Validate that ``repo`` is a strict ``owner/name`` GitHub repository slug.

    Args:
        repo: Repository slug to validate.

    Returns:
        The validated ``owner/name`` slug.

    Raises:
        ValueError: If ``repo`` is not a two-segment ``owner/name`` matching
            ``^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$``, or if either component is ``.`` or ``..``.
    """
    if not isinstance(repo, str) or not _GITHUB_REPO_RE.fullmatch(repo):
        raise ValueError(
            f"Invalid GitHub repository slug {repo!r}: expected 'owner/name' matching "
            "'^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$'."
        )
    owner, name = repo.split("/", 1)
    if owner in {".", ".."} or name in {".", ".."}:
        raise ValueError(f"Invalid GitHub repository slug {repo!r}: components must not be '.' or '..'.")
    return repo


def _kiri_releases_api(repo: str = KIRI_GITHUB_REPO) -> str:
    """Build the GitHub Releases API base URL for ``repo`` after validating the slug."""
    validated = validate_kiri_repo(repo)
    if validated == KIRI_GITHUB_REPO:
        return KIRI_RELEASES_API
    return f"https://api.github.com/repos/{validated}/releases"


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
    repo: str = KIRI_GITHUB_REPO,
    machine: str | None = None,
    system: str | None = None,
    timeout: float = 20.0,
    expected_sha256: str | None = None,
) -> tuple[str, str, str, str]:
    """Resolve ``(tag_name, asset_name, download_url, sha256_hex)`` for a release from ``repo``.

    Only the tags in :data:`PINNED_KIRI_SHA256` are pinned: their digest is part of this source tree. The
    release API's own ``digest`` field comes from the same place as the download, so it is never trusted; any
    other tag needs a caller-supplied ``expected_sha256``.

    Args:
        version: Release tag (``""``/``"latest"`` mean :data:`DEFAULT_KIRI_VERSION`).
        repo: GitHub ``owner/name`` release repository (defaults to :data:`KIRI_GITHUB_REPO`).
        machine: Machine architecture (defaults to this host's).
        system: Operating system name (defaults to this host's).
        timeout: Release-API request timeout in seconds.
        expected_sha256: SHA256 hex digest of the selected asset, required for tags without a pinned digest.
            For a pinned asset it must equal the pinned value.

    Raises:
        RuntimeError: When no asset matches, or the tag has no pinned digest and ``expected_sha256`` is missing,
            or ``expected_sha256`` contradicts the pinned digest.
    """
    releases_api = _kiri_releases_api(repo)
    resolved_ver = DEFAULT_KIRI_VERSION if version in ("", "latest") else version
    arch_tar, arch_deb = _normalize_arch(machine)
    sys_name = (system or platform.system()).lower()
    api_url = f"{releases_api}/tags/{resolved_ver}"
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
        del asset  # The API's own digest field is not a pin (same source as the download).
        expected = expected_sha256.strip().lower() if expected_sha256 else None
        pinned = pinned_for_tag.get(name)
        if pinned is not None:
            if expected is not None and expected != pinned.lower():
                raise RuntimeError(f"expected_sha256 for {name!r} contradicts the pinned digest of {tag!r}.")
            return pinned
        if expected is None:
            raise RuntimeError(
                f"{tag!r} has no pinned SHA256 for {name!r}; pinned tags are {sorted(PINNED_KIRI_SHA256)}. "
                "Pass the asset's expected SHA256 explicitly (expected_sha256= / --sha256) to install it."
            )
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise RuntimeError("expected_sha256 must be 64 hexadecimal characters.")
        return expected

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
    repo: str = KIRI_GITHUB_REPO,
    install_dir: str | os.PathLike[str] | None = None,
    force: bool = False,
    start_daemon: bool = False,
    timeout: float = 60.0,
    expected_sha256: str | None = None,
) -> str:
    """Download, SHA256-verify, and install ``kiri`` from the GitHub ``owner/name`` given by ``repo``.

    Installs into ``install_dir`` (default ``~/.local/bin``) without requiring
    ``sudo``. Verifies the downloaded archive's SHA256 digest before extracting:
    the pinned digest for tags in :data:`PINNED_KIRI_SHA256`, otherwise the
    caller's ``expected_sha256`` (a tag with neither is refused). A daemon
    started with ``start_daemon=True`` is persistent: it keeps running after
    this process exits.
    """
    validate_kiri_repo(repo)
    existing = shutil.which("kiri")
    if existing and not force:
        if start_daemon:
            ensure_kiri_daemon(kiri_bin=existing, persist=True)
        return existing

    target_dir = os.path.expanduser(os.fspath(install_dir) if install_dir is not None else "~/.local/bin")
    os.makedirs(target_dir, exist_ok=True)
    target_bin = os.path.join(target_dir, "kiri")
    if os.path.exists(target_bin) and not force:
        if start_daemon:
            ensure_kiri_daemon(kiri_bin=target_bin, persist=True)
        return target_bin

    _tag, asset_name, url, expected_sha256 = resolve_kiri_asset_url(
        repo=repo, version=version, timeout=min(timeout, 20.0), expected_sha256=expected_sha256
    )
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
        ensure_kiri_daemon(kiri_bin=installed_path, persist=True)
    return installed_path


# ---------------------------------------------------------------------------
# Owned-daemon registry
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _OwnedDaemon:
    uri: str
    proc: subprocess.Popen
    pgid: int


_OWNED_DAEMONS: dict[str, _OwnedDaemon] = {}
_ENDPOINT_BUDGETS: dict[str, KiriDaemonBudget] = {}
_REGISTRY_LOCK = threading.RLock()


def owned_kiri_daemons() -> list[str]:
    """Return the ``unix://`` endpoints of the daemons this process started and still owns."""
    with _REGISTRY_LOCK:
        return list(_OWNED_DAEMONS)


def kiri_daemon_budget(endpoint: str) -> KiriDaemonBudget | None:
    """Return the resource budget registered for ``endpoint`` by this process, if any."""
    with _REGISTRY_LOCK:
        return _ENDPOINT_BUDGETS.get(endpoint)


def shutdown_kiri_daemons(endpoints: list[str] | tuple[str, ...] | None = None, *, grace: float = 2.0) -> list[str]:
    """Stop Kiri daemons (and their TCP bridges) that this process started.

    Each owned daemon was launched as the leader of its own process group; the
    group is sent SIGTERM and, after ``grace`` seconds, SIGKILL. Daemons started
    by other processes, or with ``persist=True``, are never signalled.

    Args:
        endpoints: ``unix://`` endpoints to stop; ``None`` stops every owned daemon.
        grace: Seconds between SIGTERM and SIGKILL.

    Returns:
        The endpoints of the daemons that were stopped.
    """
    with _REGISTRY_LOCK:
        wanted = None if endpoints is None else set(endpoints)
        targets = [d for uri, d in _OWNED_DAEMONS.items() if wanted is None or uri in wanted]
        for d in targets:
            _OWNED_DAEMONS.pop(d.uri, None)
            _ENDPOINT_BUDGETS.pop(d.uri, None)
        stopped_paths = {d.uri.removeprefix("unix://") for d in targets}
        bridges = [(key, b) for key, b in _TCP_BRIDGES.items() if wanted is None or b.sock_path in stopped_paths]
        for key, b in bridges:
            _TCP_BRIDGES.pop(key, None)
            _ENDPOINT_BUDGETS.pop(b.endpoint, None)
    for _key, b in bridges:
        b.close()
    live = [d.pgid for d in targets if d.proc.poll() is None]
    _terminate_process_groups(live, grace=grace)
    for d in targets:
        try:
            d.proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
    return [d.uri for d in targets]


atexit.register(shutdown_kiri_daemons)


def _resolve_kiri_exe(kiri_bin: str | None, auto_install: bool) -> str:
    exe = kiri_bin or shutil.which("kiri")
    if exe is not None:
        return exe
    local_kiri = os.path.expanduser("~/.local/bin/kiri")
    if os.path.exists(local_kiri):
        return local_kiri
    if auto_install:
        return install_kiri()
    raise SandboxInfraError(_MISSING_KIRI_MESSAGE)


def _socket_candidates(data_dir: str | os.PathLike[str] | None) -> tuple[str | None, str | None, list[str]]:
    if data_dir is not None:
        base_home = os.path.abspath(os.path.expanduser(os.fspath(data_dir)))
        kiri_data = os.path.join(base_home, ".kiri")
        return base_home, kiri_data, [os.path.join(kiri_data, "run", "docker.sock")]
    return (
        None,
        None,
        [
            "/dev/shm/kiri/.kiri/run/docker.sock",
            os.path.expanduser("~/.kiri/run/docker.sock"),
            "/run/kiri/docker.sock",
        ],
    )


def _ping_any(candidates: list[str]) -> str | None:
    for sock_path in candidates:
        if os.path.exists(sock_path):
            uri = f"unix://{sock_path}"
            if EngineSocketClient(uri).ping():
                return uri
    return None


def _check_storage_budget(path: str, tmpfs_mb: int) -> None:
    probe = path
    while not os.path.exists(probe) and os.path.dirname(probe) != probe:
        probe = os.path.dirname(probe)
    free = shutil.disk_usage(probe).free
    need = int(tmpfs_mb) * 1024 * 1024
    if free < need:
        raise SandboxInfraError(
            f"Kiri daemon storage budget is {tmpfs_mb} MiB but only {free // (1024 * 1024)} MiB is free under "
            f"{probe!r}; lower KiriDaemonBudget.tmpfs_mb, start fewer daemons, or free space."
        )


def _ensure_daemon(
    *,
    kiri_bin: str | None = None,
    auto_install: bool = False,
    wait_seconds: float = 15.0,
    data_dir: str | os.PathLike[str] | None = None,
    budget: KiriDaemonBudget | None = None,
    persist: bool = False,
) -> tuple[str, bool]:
    """Return ``(uri, started)`` where ``started`` is ``True`` when this call booted an owned daemon."""
    base_home, kiri_data, candidates = _socket_candidates(data_dir)
    uri = _ping_any(candidates)
    if uri is not None:
        return uri, False

    with _KIRI_BOOT_LOCK:
        uri = _ping_any(candidates)
        if uri is not None:
            return uri, False

        exe = _resolve_kiri_exe(kiri_bin, auto_install)
        if budget is not None and budget.tmpfs_mb > 0:
            _check_storage_budget(base_home or os.path.expanduser("~"), budget.tmpfs_mb)

        daemon_env = dict(os.environ)
        if base_home is not None and kiri_data is not None:
            os.makedirs(os.path.join(kiri_data, "run"), exist_ok=True)
            daemon_env["HOME"] = base_home
            daemon_env["KIRI_DATA_DIR"] = kiri_data

        proc = subprocess.Popen(
            [exe, "daemon"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=daemon_env,
            start_new_session=True,
        )

        deadline = time.monotonic() + max(1.0, float(wait_seconds))
        while time.monotonic() < deadline:
            uri = _ping_any(candidates)
            if uri is not None:
                with _REGISTRY_LOCK:
                    if not persist:
                        _OWNED_DAEMONS[uri] = _OwnedDaemon(uri=uri, proc=proc, pgid=proc.pid)
                    if budget is not None:
                        _ENDPOINT_BUDGETS[uri] = budget
                return uri, not persist
            time.sleep(0.1)

        # The daemon never came up: it is ours, so tear its process group down.
        pid = getattr(proc, "pid", None)
        if isinstance(pid, int):
            _terminate_process_groups([pid])
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
    raise SandboxInfraError(f"Kiri daemon {exe!r} did not answer /_ping within {wait_seconds}s (tried {candidates}).")


def ensure_kiri_daemon(
    *,
    kiri_bin: str | None = None,
    auto_install: bool = False,
    wait_seconds: float = 15.0,
    data_dir: str | os.PathLike[str] | None = None,
    budget: KiriDaemonBudget | None = None,
    persist: bool = False,
) -> str:
    """Ensure a Kiri daemon is running and return its ``unix://`` socket URI.

    A reachable daemon is reused as is. Otherwise a new daemon is started as the
    leader of its own process group and, unless ``persist=True``, registered as
    owned by this process (stopped by :func:`shutdown_kiri_daemons` or at exit).

    Args:
        kiri_bin: Path to the ``kiri`` binary (default: ``PATH``, then ``~/.local/bin/kiri``).
        auto_install: Download and install Kiri when no binary is found. Off by
            default: installation is an explicit ``easydel setup kiri`` step.
        wait_seconds: How long to wait for the new daemon to answer ``/_ping``.
        data_dir: Optional data directory (for example ``"/dev/shm/kiri"`` on hosts
            with large RAM and small root disks); ``HOME`` and ``KIRI_DATA_DIR``
            of the daemon are pointed into it.
        budget: Optional per-daemon budget. Its ``tmpfs_mb`` is checked against the
            free space of ``data_dir`` before booting, and containers created
            through this endpoint by this process are clamped to it.
        persist: Leave the daemon running after this process exits.

    Returns:
        The daemon's ``unix://`` endpoint.

    Raises:
        SandboxInfraError: When no binary is available (and ``auto_install`` is
            off), the storage budget does not fit, or the daemon does not start.
    """
    uri, _started = _ensure_daemon(
        kiri_bin=kiri_bin,
        auto_install=auto_install,
        wait_seconds=wait_seconds,
        data_dir=data_dir,
        budget=budget,
        persist=persist,
    )
    return uri


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


# ---------------------------------------------------------------------------
# Authenticated, policy-filtered TCP -> Unix bridge
# ---------------------------------------------------------------------------

_MAX_BRIDGE_HEAD_BYTES = 64 * 1024
_MAX_BRIDGE_BODY_BYTES = 256 * 1024 * 1024
_BRIDGE_REQUEST_TIMEOUT_S = 30.0
_API_VERSION_RE = re.compile(r"^/v\d+(?:\.\d+)*(?=/)")
_EXEC_CREATE_RE = re.compile(r"/containers/[^/]+/exec")
_DENIED_API_PREFIXES = ("/plugins", "/swarm", "/services", "/secrets", "/configs", "/nodes", "/session", "/build")
_HOST_NAMESPACE_KEYS = ("NetworkMode", "PidMode", "IpcMode", "UTSMode", "UsernsMode", "CgroupnsMode")
_HOP_HEADERS = ("connection", "keep-alive", "proxy-connection")


_ALLOWED_MOUNT_TYPES = ("volume", "tmpfs")
_LOCAL_VOLUME_DRIVERS = ("", "local")
_PATH_SAFE_CHARS = "/:@!$&'()*+,;=-._~"
_HEADER_TOKEN_RE = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
_BUDGET_HOST_KEYS = ("Memory", "NanoCpus", "PidsLimit", "Tmpfs")


def _has_control_chars(text: str, *, allow_tab: bool) -> bool:
    """Whether ``text`` holds a C0 control character, DEL, or (for targets) a space."""
    for ch in text:
        code = ord(ch)
        if code == 0x7F or (code < 0x20 and not (allow_tab and ch == "\t")):
            return True
        if not allow_tab and ch == " ":
            return True
    return False


class _BridgePolicyError(ValueError):
    """A bridged request is refused by the sandbox policy (message is returned to the client)."""


def _reject_duplicate_keys(pairs: list[tuple[str, tp.Any]]) -> dict[str, tp.Any]:
    out: dict[str, tp.Any] = {}
    for key, value in pairs:
        if key in out:
            raise _BridgePolicyError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def _casefold_keys(value: tp.Any) -> tp.Any:
    """Recursively case-fold every object key, refusing keys that collide case-insensitively.

    Go engines match JSON object keys to struct fields case-insensitively (with
    Unicode simple folding), so the policy must see the keys the way the engine
    does: ``{"hostconfig": {"PRIVILEGED": true}}`` means ``HostConfig.Privileged``.
    """
    if isinstance(value, dict):
        folded: dict[str, tp.Any] = {}
        for key, item in value.items():
            fkey = key.casefold()
            if fkey in folded:
                raise _BridgePolicyError(f"JSON keys collide case-insensitively: {key!r}")
            folded[fkey] = _casefold_keys(item)
        return folded
    if isinstance(value, list):
        return [_casefold_keys(item) for item in value]
    return value


def _normalize_bridge_target(target: str) -> tuple[str, str]:
    """Return ``(policy_path, forwarded_target)`` for an origin-form request target.

    Only origin-form targets starting with exactly one ``/`` are accepted.
    Empty segments (repeated slashes) are collapsed and ``.``/``..`` segments
    are refused. The policy checks the fully percent-decoded path (an encoded
    ``%2F`` counts as a separator there); the forwarded target is rebuilt from
    the same collapsed segments, each re-encoded canonically, with the
    original query string.

    Raises:
        _BridgePolicyError: For absolute-form, ``//``-prefixed or traversing targets.
    """
    if not target.startswith("/") or target.startswith("//"):
        raise _BridgePolicyError("request target must be an origin-form path starting with a single '/'")
    if "#" in target:
        raise _BridgePolicyError("request target must not carry a fragment")
    raw_path, sep, query = target.partition("?")
    try:
        raw_segments = [urllib.parse.unquote(seg, errors="strict") for seg in raw_path.split("/") if seg]
    except UnicodeDecodeError as exc:
        raise _BridgePolicyError("request path is not valid percent-encoded UTF-8") from exc
    decoded = "/".join(raw_segments)
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in decoded):
        raise _BridgePolicyError("request path contains control characters")
    segments = [seg for seg in decoded.split("/") if seg]
    if any(seg in (".", "..") for seg in segments):
        raise _BridgePolicyError("request path must not contain '.' or '..' segments")
    path = "/" + "/".join(segments)
    segment_safe = _PATH_SAFE_CHARS.replace("/", "")
    forwarded = "/" + "/".join(urllib.parse.quote(seg, safe=segment_safe) for seg in raw_segments)
    return path, forwarded + (sep + query if sep else "")


def _container_create_violations(spec: dict) -> list[str]:
    """List the sandbox-escaping settings requested by a case-folded container-create body.

    HostConfig fields are checked both inside ``HostConfig`` and at the body root, where engines that keep the
    legacy embedded-HostConfig layout still honour them.
    """
    host_config = spec.get("hostconfig") or {}
    if not isinstance(host_config, dict):
        return ["HostConfig must be an object"]
    root = {k: v for k, v in spec.items() if k != "hostconfig"}
    found = [f"{v} (at the body root)" for v in _host_config_violations(root)]
    return found + _host_config_violations(host_config)


def _host_config_violations(host_config: dict) -> list[str]:
    """List the sandbox-escaping settings in one case-folded HostConfig-shaped mapping."""
    found: list[str] = []
    if host_config.get("privileged"):
        found.append("Privileged")
    if host_config.get("binds"):
        found.append("Binds (host bind mounts)")
    if host_config.get("volumesfrom"):
        found.append("VolumesFrom")
    if str(host_config.get("volumedriver") or "").lower() not in _LOCAL_VOLUME_DRIVERS:
        found.append("VolumeDriver (non-local volume driver)")
    mounts = host_config.get("mounts") or []
    if not isinstance(mounts, list):
        found.append("Mounts must be a list")
        mounts = []
    for mount in mounts:
        if not isinstance(mount, dict) or str(mount.get("type", "")).lower() not in _ALLOWED_MOUNT_TYPES:
            found.append("Mounts (only volume and tmpfs mounts are allowed)")
            break
        options = mount.get("volumeoptions")
        if options is not None and (not isinstance(options, dict) or options.get("driverconfig")):
            found.append("Mounts (volume DriverConfig)")
            break
    if host_config.get("devices") or host_config.get("devicerequests") or host_config.get("devicecgrouprules"):
        found.append("Devices")
    if host_config.get("capadd"):
        found.append("CapAdd")
    for key in _HOST_NAMESPACE_KEYS:
        value = str(host_config.get(key.casefold()) or "").lower()
        if value == "host" or value.startswith("container:"):
            found.append(f"{key}={value}")
    if any("unconfined" in str(opt).lower() for opt in host_config.get("securityopt") or []):
        found.append("SecurityOpt unconfined")
    return found


def _volume_create_violations(spec: dict) -> list[str]:
    """List the host-reaching settings requested by a case-folded volume-create body."""
    found: list[str] = []
    if str(spec.get("driver") or "").lower() not in _LOCAL_VOLUME_DRIVERS:
        found.append("non-local Driver")
    if spec.get("driveropts"):
        found.append("DriverOpts")
    return found


def _clamp_create_spec(spec: dict[str, tp.Any], budget: KiriDaemonBudget) -> None:
    """Clamp a container-create body's resource limits to ``budget`` in place.

    Limit keys are matched case-insensitively (as the engine does) and rewritten under their canonical names
    inside ``HostConfig``; body-root copies (legacy layout) are dropped so they cannot override the clamp.

    Raises:
        _BridgePolicyError: When a limit is not an integer.
    """
    folded_limits = {k.casefold(): k for k in _BUDGET_HOST_KEYS}
    hc_key = next((k for k in spec if k.casefold() == "hostconfig"), "HostConfig")
    host_config = dict(spec.get(hc_key) or {})
    for key in list(host_config):
        canon = folded_limits.get(key.casefold())
        if canon is not None and key != canon:
            host_config[canon] = host_config.pop(key)
    try:
        host_config = budget.clamp_host_config(host_config)
    except (TypeError, ValueError) as exc:
        raise _BridgePolicyError(f"container create refused: invalid resource limit ({exc})") from exc
    for key in list(spec):
        if key.casefold() in folded_limits:
            del spec[key]
    spec[hc_key] = host_config


def _bridge_policy_check(
    method: str, target: str, body: bytes, budget: KiriDaemonBudget | None = None
) -> tuple[str, bytes]:
    """Apply the sandbox policy to one bridged request.

    Args:
        method: Request method.
        target: Origin-form request target.
        body: Request body.
        budget: Resource budget of the bridged daemon; approved container creates are clamped to it.

    Returns:
        ``(forwarded_target, forwarded_body)``: the normalised target and, for
        policed JSON endpoints, the re-serialised body the policy approved.

    Raises:
        _BridgePolicyError: When the request is refused.
    """
    if not method.isalpha() or method != method.upper():
        raise _BridgePolicyError("request method must be an upper-case token")
    path, forwarded = _normalize_bridge_target(target)
    path = _API_VERSION_RE.sub("", path, count=1)
    if any(path == p or path.startswith(p + "/") for p in _DENIED_API_PREFIXES):
        raise _BridgePolicyError(f"{path} is not available through the sandbox bridge")
    if method != "POST":
        return forwarded, body
    kind = None
    if path == "/containers/create":
        kind = "container"
    elif path == "/volumes/create":
        kind = "volume"
    elif _EXEC_CREATE_RE.fullmatch(path):
        kind = "exec"
    if kind is None:
        return forwarded, body
    try:
        spec = json.loads(body.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys) if body else {}
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _BridgePolicyError("request body must be valid JSON") from exc
    if not isinstance(spec, dict):
        raise _BridgePolicyError("request body must be a JSON object")
    folded = _casefold_keys(spec)
    if kind == "container":
        found = _container_create_violations(folded)
        if found:
            raise _BridgePolicyError("container create refused: " + ", ".join(found))
        if budget is not None:
            _clamp_create_spec(spec, budget)
            return forwarded, json.dumps(spec).encode("utf-8")
    elif kind == "volume":
        found = _volume_create_violations(folded)
        if found:
            raise _BridgePolicyError("volume create refused: " + ", ".join(found))
    elif folded.get("privileged"):
        raise _BridgePolicyError("privileged exec refused")
    # Forward exactly the document that was checked, so engine-side parsing cannot diverge.
    return forwarded, (json.dumps(spec).encode("utf-8") if body else body)


def _bridge_policy_violation(method: str, target: str, body: bytes) -> str | None:
    """Return why a bridged request is refused, or ``None`` when it may be forwarded."""
    try:
        _bridge_policy_check(method, target, body)
    except _BridgePolicyError as exc:
        return str(exc)
    return None


class _BridgeRejected(Exception):
    def __init__(self, status: int, reason: str, message: str):
        super().__init__(message)
        self.status = status
        self.reason = reason
        self.message = message


def _send_http_error(conn: socket.socket, status: int, reason: str, message: str) -> None:
    body = json.dumps({"message": message}).encode("utf-8")
    head = (
        f"HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
    ).encode("ascii")
    try:
        conn.sendall(head + body)
    except OSError:
        pass


def _read_bridge_request(conn: socket.socket) -> tuple[str, str, str, list[tuple[str, str]], bytes]:
    buf = bytearray()
    while b"\r\n\r\n" not in buf:
        if len(buf) > _MAX_BRIDGE_HEAD_BYTES:
            raise _BridgeRejected(431, "Request Header Fields Too Large", "request head too large")
        chunk = conn.recv(65536)
        if not chunk:
            raise _BridgeRejected(400, "Bad Request", "incomplete request head")
        buf += chunk
    head, _, rest = bytes(buf).partition(b"\r\n\r\n")
    # Engines may treat a bare CR or LF as a line break; the bridge must see exactly the lines they see.
    if b"\r" in head.replace(b"\r\n", b"") or b"\n" in head.replace(b"\r\n", b""):
        raise _BridgeRejected(400, "Bad Request", "bare CR or LF in request head")
    lines = head.decode("latin-1").split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) != 3 or not parts[2].startswith("HTTP/1."):
        raise _BridgeRejected(400, "Bad Request", "malformed request line")
    method, target, version = parts
    if _has_control_chars(target, allow_tab=False):
        raise _BridgeRejected(400, "Bad Request", "control characters in request target")
    headers: list[tuple[str, str]] = []
    seen: dict[str, str] = {}
    for line in lines[1:]:
        if ":" not in line:
            raise _BridgeRejected(400, "Bad Request", "malformed header line")
        key, value = line.split(":", 1)
        if not _HEADER_TOKEN_RE.fullmatch(key):
            raise _BridgeRejected(400, "Bad Request", "header name is not an RFC 7230 token")
        value = value.strip(" \t")
        if _has_control_chars(value, allow_tab=True):
            raise _BridgeRejected(400, "Bad Request", f"control characters in {key} header")
        lowered = key.lower()
        if lowered in seen and lowered in ("content-length", SANDBOX_TOKEN_HEADER.lower()):
            raise _BridgeRejected(400, "Bad Request", f"duplicate {key} header")
        seen[lowered] = value
        headers.append((key, value))
    if "transfer-encoding" in seen:
        raise _BridgeRejected(411, "Length Required", "chunked request bodies are not accepted by the bridge")
    try:
        length = int(seen.get("content-length", "0"))
    except ValueError as exc:
        raise _BridgeRejected(400, "Bad Request", "invalid Content-Length") from exc
    if length < 0:
        raise _BridgeRejected(400, "Bad Request", "invalid Content-Length")
    if length > _MAX_BRIDGE_BODY_BYTES:
        raise _BridgeRejected(413, "Payload Too Large", "request body too large")
    body = bytearray(rest)
    while len(body) < length:
        chunk = conn.recv(min(65536, length - len(body)))
        if not chunk:
            raise _BridgeRejected(400, "Bad Request", "incomplete request body")
        body += chunk
    return method, target, version, headers, bytes(body[:length])


def _shutdown_on_client_close(client: socket.socket, upstream: socket.socket) -> None:
    # Bytes sent after the single authenticated request are discarded, never forwarded.
    try:
        while client.recv(65536):
            pass
    except OSError:
        pass
    try:
        upstream.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass


def _handle_bridge_connection(client: socket.socket, sock_path: str, token: str | None) -> None:
    upstream: socket.socket | None = None
    try:
        client.settimeout(_BRIDGE_REQUEST_TIMEOUT_S)
        try:
            method, target, version, headers, body = _read_bridge_request(client)
            if token is not None:
                presented = next((v for k, v in headers if k.lower() == SANDBOX_TOKEN_HEADER.lower()), None)
                if presented is None or not hmac.compare_digest(presented.encode("utf-8"), token.encode("utf-8")):
                    raise _BridgeRejected(401, "Unauthorized", f"missing or invalid {SANDBOX_TOKEN_HEADER}")
            try:
                target, body = _bridge_policy_check(method, target, body, kiri_daemon_budget(f"unix://{sock_path}"))
            except _BridgePolicyError as exc:
                raise _BridgeRejected(403, "Forbidden", str(exc)) from exc
        except _BridgeRejected as rejected:
            _send_http_error(client, rejected.status, rejected.reason, rejected.message)
            return
        dropped = (*_HOP_HEADERS, SANDBOX_TOKEN_HEADER.lower(), "content-length")
        forwarded = [f"{method} {target} {version}"]
        forwarded += [f"{k}: {v}" for k, v in headers if k.lower() not in dropped]
        forwarded.append(f"Content-Length: {len(body)}")
        forwarded.append("Connection: close")
        try:
            upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            upstream.settimeout(_BRIDGE_REQUEST_TIMEOUT_S)
            upstream.connect(sock_path)
            upstream.sendall(("\r\n".join(forwarded) + "\r\n\r\n").encode("latin-1") + body)
        except OSError as exc:
            _send_http_error(client, 502, "Bad Gateway", f"sandbox engine unreachable: {exc}")
            return
        upstream.settimeout(None)
        client.settimeout(None)
        threading.Thread(target=_shutdown_on_client_close, args=(client, upstream), daemon=True).start()
        while True:
            chunk = upstream.recv(65536)
            if not chunk:
                break
            client.sendall(chunk)
    except OSError:
        pass
    finally:
        for s in (upstream, client):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass


def _peer_allowed(host: str, allowed: frozenset | None) -> bool:
    try:
        ip = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return False
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if allowed is None:
        return ip.is_loopback
    return ip in allowed


@dataclasses.dataclass
class _Bridge:
    endpoint: str
    sock_path: str
    server: socket.socket
    token: str | None
    allowed: frozenset | None
    stop: threading.Event = dataclasses.field(default_factory=threading.Event)
    thread: threading.Thread | None = None

    def serve(self) -> None:
        while not self.stop.is_set():
            try:
                client, addr = self.server.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            if not _peer_allowed(str(addr[0]), self.allowed):
                try:
                    client.close()
                except OSError:
                    pass
                continue
            threading.Thread(
                target=_handle_bridge_connection,
                args=(client, self.sock_path, self.token),
                daemon=True,
            ).start()

    def is_alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def close(self) -> None:
        self.stop.set()
        try:
            self.server.close()
        except OSError:
            pass
        if self.thread is not None:
            self.thread.join(timeout=1.0)


_TCP_BRIDGES: dict[tuple[str, int], _Bridge] = {}


def _format_endpoint(host: str, port: int) -> str:
    return f"http://[{host}]:{port}" if ":" in host else f"http://{host}:{port}"


def _start_unix_tcp_bridge(
    sock_path: str,
    bind_host: str = "127.0.0.1",
    port: int = 2375,
    *,
    token: str | None = None,
    allowed_peers: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Start an in-process TCP -> Unix bridge to a Kiri daemon socket and return its ``http://`` endpoint.

    The bridge handles exactly one request per connection. It rejects requests
    lacking the shared ``token`` (sent as :data:`SANDBOX_TOKEN_HEADER`, which is
    stripped before forwarding), and refuses container/exec creates that ask
    for privileged mode, host binds or mounts, host namespaces, devices, or
    added capabilities.

    Args:
        sock_path: Path of the daemon's Unix socket.
        bind_host: IP literal to bind. Loopback (default) serves only local peers.
        port: TCP port to bind.
        token: Shared secret (at least 16 characters). Required for non-loopback binds.
        allowed_peers: Peer IPs allowed to connect. Required for non-loopback
            binds; loopback binds without it accept only loopback peers.

    Returns:
        The bridge endpoint.

    Raises:
        ValueError: For a non-IP ``bind_host``, a missing/short token or missing
            allow-list on a routable bind, or a port already bridged to another socket.
        SandboxInfraError: When the port cannot be bound.
    """
    try:
        bind_ip = ipaddress.ip_address(bind_host)
    except ValueError as exc:
        raise ValueError(f"bind_host must be an IP literal, got {bind_host!r}.") from exc
    if token is not None and len(token) < _MIN_BRIDGE_TOKEN_CHARS:
        raise ValueError(f"Bridge token must be at least {_MIN_BRIDGE_TOKEN_CHARS} characters.")
    if not bind_ip.is_loopback:
        if not token:
            raise ValueError(
                f"Refusing to expose the sandbox engine on {bind_host} without a shared secret token; "
                "pass token=... (and the same token to SandboxPool/KiriRuntime)."
            )
        if not allowed_peers:
            raise ValueError(f"Refusing to expose the sandbox engine on {bind_host} without an allowed_peers list.")
    allowed = frozenset(ipaddress.ip_address(p) for p in allowed_peers) if allowed_peers else None

    endpoint = _format_endpoint(bind_host, int(port))
    key = (bind_host, int(port))
    with _REGISTRY_LOCK:
        existing = _TCP_BRIDGES.get(key)
        if existing is not None and existing.is_alive():
            if existing.sock_path != sock_path:
                raise ValueError(f"{endpoint} already bridges {existing.sock_path!r}, not {sock_path!r}.")
            return endpoint

        srv = socket.socket(socket.AF_INET6 if bind_ip.version == 6 else socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind((bind_host, int(port)))
            srv.listen(128)
        except OSError as exc:
            srv.close()
            raise SandboxInfraError(f"Failed to bind sandbox bridge on {bind_host}:{port}: {exc}") from exc
        srv.settimeout(0.25)
        bridge = _Bridge(endpoint=endpoint, sock_path=sock_path, server=srv, token=token, allowed=allowed)
        bridge.thread = threading.Thread(target=bridge.serve, name=f"kiri-tcp-bridge-{port}", daemon=True)
        bridge.thread.start()
        _TCP_BRIDGES[key] = bridge
        budget = _ENDPOINT_BUDGETS.get(f"unix://{sock_path}")
        if budget is not None:
            _ENDPOINT_BUDGETS[endpoint] = budget
    return endpoint


# ---------------------------------------------------------------------------
# RAM-backed pools
# ---------------------------------------------------------------------------


def _ensure_shm_pool(
    num_daemons: int = 4,
    *,
    data_dir: str | os.PathLike[str] = "/dev/shm/kiri",
    kiri_bin: str | None = None,
    auto_install: bool = False,
    wait_seconds: float = 15.0,
    budget: KiriDaemonBudget | None = None,
    persist: bool = False,
) -> tuple[list[str], list[str]]:
    """Return ``(endpoints, started_by_this_call)`` for a local ``/dev/shm`` daemon pool."""
    import concurrent.futures

    count = max(1, int(num_daemons))
    root = os.path.abspath(os.path.expanduser(os.fspath(data_dir)))
    budget = budget if budget is not None else KiriDaemonBudget()
    vm_dirs = [root if count == 1 else os.path.join(root, f"vm-{idx}") for idx in range(count)]

    exe = kiri_bin
    if any(_ping_any(_socket_candidates(d)[2]) is None for d in vm_dirs):
        exe = _resolve_kiri_exe(kiri_bin, auto_install)

    def _boot_one(vm_dir: str) -> tuple[str, bool]:
        return _ensure_daemon(
            kiri_bin=exe,
            auto_install=False,
            wait_seconds=wait_seconds,
            data_dir=vm_dir,
            budget=budget,
            persist=persist,
        )

    results: list[tuple[str, bool]] = []
    errors: list[BaseException] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(count, 16)) as pool:
        for fut in [pool.submit(_boot_one, d) for d in vm_dirs]:
            try:
                results.append(fut.result())
            except BaseException as exc:
                errors.append(exc)
    started = [uri for uri, was_started in results if was_started]
    if errors:
        shutdown_kiri_daemons(started)
        raise errors[0]
    with _REGISTRY_LOCK:
        for uri, _ in results:
            _ENDPOINT_BUDGETS.setdefault(uri, budget)
    return [uri for uri, _ in results], started


def ensure_kiri_shm_pool(
    num_daemons: int = 4,
    *,
    data_dir: str | os.PathLike[str] = "/dev/shm/kiri",
    kiri_bin: str | None = None,
    auto_install: bool = False,
    wait_seconds: float = 15.0,
    budget: KiriDaemonBudget | None = None,
    persist: bool = False,
) -> list[str]:
    """Start or reuse ``num_daemons`` local Kiri daemons backed by ``/dev/shm`` tmpfs.

    Args:
        num_daemons: Number of daemons (each gets ``<data_dir>/vm-<i>``).
        data_dir: Root of the RAM-backed data directories.
        kiri_bin: Path to the ``kiri`` binary.
        auto_install: Download Kiri when no binary is found (off by default).
        wait_seconds: Per-daemon boot timeout.
        budget: Per-daemon resource budget (default :class:`KiriDaemonBudget`:
            4 CPUs, 8 GiB memory, 1024 pids and 4 GiB of tmpfs per daemon).
            Free space is checked before booting each daemon and containers
            created by this process through the endpoints are clamped to it.
        persist: Leave the daemons running after this process exits.

    Returns:
        The daemons' ``unix://`` endpoints.
    """
    endpoints, _started = _ensure_shm_pool(
        num_daemons,
        data_dir=data_dir,
        kiri_bin=kiri_bin,
        auto_install=auto_install,
        wait_seconds=wait_seconds,
        budget=budget,
        persist=persist,
    )
    return endpoints


def _gather_pod_ips(my_ip: str) -> list[str]:
    try:
        import jax

        n_procs = jax.process_count()
    except Exception:
        n_procs = 1
    if n_procs <= 1:
        return [my_ip]

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
    return all_ips or [my_ip]


def _ensure_pod_shm_pool(
    num_daemons_per_host: int = 8,
    *,
    data_dir: str | os.PathLike[str] = "/dev/shm/kiri",
    base_port: int = 2375,
    host_ips: list[str] | tuple[str, ...] | None = None,
    kiri_bin: str | None = None,
    auto_install: bool = False,
    wait_seconds: float = 15.0,
    budget: KiriDaemonBudget | None = None,
    expose_bridge: bool = False,
    bridge_token: str | None = None,
    allowed_peers: list[str] | tuple[str, ...] | None = None,
) -> tuple[list[str], list[str]]:
    if not expose_bridge and host_ips:
        raise ValueError(
            "host_ips requires expose_bridge=True and a bridge_token: remote hosts can only reach this host's "
            "daemons through the authenticated bridge."
        )
    if expose_bridge and not bridge_token:
        raise ValueError("expose_bridge=True requires a shared bridge_token (identical on every host).")
    local_uris, started = _ensure_shm_pool(
        num_daemons_per_host,
        data_dir=data_dir,
        kiri_bin=kiri_bin,
        auto_install=auto_install,
        wait_seconds=wait_seconds,
        budget=budget,
    )
    if not expose_bridge:
        return local_uris, started

    try:
        my_ip = _detect_private_ip()
        all_ips = list(host_ips) if host_ips else _gather_pod_ips(my_ip)
        peers = list(allowed_peers) if allowed_peers is not None else list(all_ips)
        if my_ip not in peers:
            peers.append(my_ip)
        for idx, uri in enumerate(local_uris):
            _start_unix_tcp_bridge(
                uri.removeprefix("unix://"),
                my_ip,
                base_port + idx,
                token=bridge_token,
                allowed_peers=peers,
            )
    except BaseException:
        shutdown_kiri_daemons(started)
        raise
    return [_format_endpoint(ip, base_port + idx) for ip in all_ips for idx in range(len(local_uris))], started


def ensure_kiri_pod_shm_pool(
    num_daemons_per_host: int = 8,
    *,
    data_dir: str | os.PathLike[str] = "/dev/shm/kiri",
    base_port: int = 2375,
    host_ips: list[str] | tuple[str, ...] | None = None,
    kiri_bin: str | None = None,
    auto_install: bool = False,
    wait_seconds: float = 15.0,
    budget: KiriDaemonBudget | None = None,
    expose_bridge: bool = False,
    bridge_token: str | None = None,
    allowed_peers: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Start ``num_daemons_per_host`` RAM-backed Kiri daemons on this host and optionally bridge them pod-wide.

    Without ``expose_bridge`` (default) nothing listens on the network and each
    host uses only its own local daemons. With ``expose_bridge=True`` every
    daemon is bridged on this host's private IP behind ``bridge_token`` and an
    allow-list of peer IPs (default: the pod's host IPs, gathered across JAX
    processes when ``host_ips`` is not given), and the endpoints of all hosts
    are returned. Clients must send the same token (``SandboxPool(token=...)``).

    Args:
        num_daemons_per_host: Daemons booted on this host.
        data_dir: Root of the RAM-backed data directories.
        base_port: First bridge port; daemon ``i`` uses ``base_port + i``.
        host_ips: Explicit pod host IPs (requires ``expose_bridge=True``).
        kiri_bin: Path to the ``kiri`` binary.
        auto_install: Download Kiri when no binary is found (off by default).
        wait_seconds: Per-daemon boot timeout.
        budget: Per-daemon resource budget (see :func:`ensure_kiri_shm_pool`).
        expose_bridge: Bridge the daemons on the private IP for other hosts.
        bridge_token: Shared secret required by the bridges.
        allowed_peers: Peer IPs allowed to connect (default: the pod host IPs).

    Returns:
        Local ``unix://`` endpoints, or every host's bridge endpoints when bridging.
    """
    endpoints, _started = _ensure_pod_shm_pool(
        num_daemons_per_host,
        data_dir=data_dir,
        base_port=base_port,
        host_ips=host_ips,
        kiri_bin=kiri_bin,
        auto_install=auto_install,
        wait_seconds=wait_seconds,
        budget=budget,
        expose_bridge=expose_bridge,
        bridge_token=bridge_token,
        allowed_peers=allowed_peers,
    )
    return endpoints


def add_kiri_repo_argument(parser: argparse.ArgumentParser) -> None:
    """Add the release repository option shared by the Kiri setup commands."""
    parser.add_argument(
        "--repo",
        type=validate_kiri_repo,
        default=KIRI_GITHUB_REPO,
        metavar="OWNER/NAME",
        help=f"GitHub release repository (default: {KIRI_GITHUB_REPO}).",
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``easydel-install-kiri``."""
    parser = argparse.ArgumentParser(description=f"Install the Kiri container runtime ({KIRI_GITHUB_REPO}).")
    parser.add_argument(
        "--version",
        default=DEFAULT_KIRI_VERSION,
        help=(
            f"GitHub release tag (default: {DEFAULT_KIRI_VERSION}). Only {sorted(PINNED_KIRI_SHA256)} are pinned; "
            "any other tag also needs --sha256."
        ),
    )
    add_kiri_repo_argument(parser)
    parser.add_argument("--sha256", default=None, help="Expected SHA256 of the release asset (unpinned tags).")
    parser.add_argument("--install-dir", default=None, help="Target directory (default: ~/.local/bin).")
    parser.add_argument("--force", action="store_true", help="Reinstall even if kiri is already present.")
    parser.add_argument(
        "--start", action="store_true", help="Start a persistent `kiri daemon` in the background after install."
    )
    args = parser.parse_args(argv)

    path = install_kiri(
        version=args.version,
        repo=args.repo,
        install_dir=args.install_dir,
        force=args.force,
        start_daemon=args.start,
        expected_sha256=args.sha256,
    )
    print(f"Installed Kiri at: {path}")
    return 0


__all__ = (
    "DEFAULT_KIRI_VERSION",
    "KIRI_GITHUB_REPO",
    "PINNED_KIRI_SHA256",
    "ensure_kiri_daemon",
    "ensure_kiri_pod_shm_pool",
    "ensure_kiri_shm_pool",
    "install_kiri",
    "kiri_daemon_budget",
    "main",
    "owned_kiri_daemons",
    "resolve_kiri_asset_url",
    "shutdown_kiri_daemons",
)
