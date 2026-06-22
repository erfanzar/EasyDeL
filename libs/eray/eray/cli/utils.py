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

"""Shared utilities for the eray CLI: gcloud wrappers, SSH backends, logging, TPU discovery."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass

logger = logging.getLogger("eray.cli")

# ANSI colors
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
BLUE = "\033[0;34m"
NC = "\033[0m"


def info(msg: str) -> None:
    logger.info(msg)
    print(f"{BLUE}[INFO]{NC} {msg}", file=sys.stderr)


def success(msg: str) -> None:
    logger.info(msg)
    print(f"{GREEN}[SUCCESS]{NC} {msg}", file=sys.stderr)


def warning(msg: str) -> None:
    logger.warning(msg)
    print(f"{YELLOW}[WARNING]{NC} {msg}", file=sys.stderr)


def error(msg: str) -> None:
    logger.error(msg)
    print(f"{RED}[ERROR]{NC} {msg}", file=sys.stderr)


# ── Command execution ────────────────────────────────────────────


def run_command(
    cmd: list[str],
    *,
    timeout: int = 300,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess:
    """Run a subprocess command and return the result."""
    logger.debug(f"Running: {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        timeout=timeout,
        check=check,
    )


# ── gcloud wrappers ──────────────────────────────────────────────


def gcloud(args: list[str], *, timeout: int = 300, check: bool = True) -> str:
    """Run a gcloud command and return stdout."""
    result = run_command(["gcloud", *args], timeout=timeout, check=check)
    return result.stdout.strip()


def gcloud_json(args: list[str], *, timeout: int = 300) -> dict | list:
    """Run a gcloud command and parse JSON output."""
    result = run_command(["gcloud", *args, "--format=json"], timeout=timeout, check=True)
    return json.loads(result.stdout)


def check_gcloud() -> bool:
    """Verify gcloud CLI is installed and authenticated."""
    try:
        run_command(["gcloud", "--version"], timeout=10, capture=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return True


def get_active_account() -> str | None:
    """Get the currently active gcloud account."""
    try:
        result = gcloud(["config", "get-value", "account"], check=False)
        if result and result != "(unset)":
            return result
    except Exception:
        pass
    return None


def _metadata_value(path: str) -> str | None:
    """Query GCE metadata server (works inside TPU/VM instances)."""
    try:
        result = run_command(
            [
                "curl",
                "-fsS",
                f"http://metadata.google.internal/computeMetadata/v1/{path}",
                "-H",
                "Metadata-Flavor: Google",
            ],
            timeout=5,
            check=False,
        )
        val = result.stdout.strip()
        return val if val else None
    except Exception:
        return None


def detect_project() -> str | None:
    """Detect GCP project from metadata server, then gcloud config."""
    proj = _metadata_value("project/project-id")
    if proj:
        return proj
    try:
        result = gcloud(["config", "get-value", "project"], check=False)
        if result and result != "(unset)":
            return result
    except Exception:
        pass
    return None


def detect_zone() -> str | None:
    """Detect GCP zone from metadata server, then gcloud config."""
    zone_raw = _metadata_value("instance/zone")
    if zone_raw:
        return zone_raw.split("/")[-1]
    try:
        result = gcloud(["config", "get-value", "compute/zone"], check=False)
        if result and result != "(unset)":
            return result
    except Exception:
        pass
    return None


# ── TpuInfo ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class TpuInfo:
    """Discovered TPU metadata.

    Can represent either a gcloud-managed TPU (name/project/zone set)
    or a direct-IP connection (name/project/zone are None).

    Attributes:
        name: TPU resource name, or None for direct-IP mode.
        project: GCP project ID, or None for direct-IP mode.
        zone: GCP zone, or None for direct-IP mode.
        accelerator_type: e.g. "v4-32".
        internal_ips: List of internal IPs, one per worker host.
        num_hosts: Number of worker hosts.
        state: TPU state (READY, CREATING, etc.).
    """

    name: str | None
    project: str | None
    zone: str | None
    accelerator_type: str
    internal_ips: list[str]
    num_hosts: int
    state: str

    @property
    def is_gcloud_managed(self) -> bool:
        """True if this TPU was discovered via gcloud (has a name)."""
        return self.name is not None

    @property
    def tpu_version(self) -> str:
        """Extract version from accelerator type. e.g. 'v4-32' → 'v4'."""
        return self.accelerator_type.split("-")[0]

    @property
    def slice_size(self) -> str:
        """Extract slice size from accelerator type. e.g. 'v4-32' → '32'."""
        parts = self.accelerator_type.split("-")
        return parts[1] if len(parts) > 1 else "8"

    @property
    def chips_per_host(self) -> int:
        """Approximate chips per host from the slice topology."""
        total = int(self.slice_size)
        return max(total // self.num_hosts, 1) if self.num_hosts else total

    @classmethod
    def from_ips(
        cls,
        ips: list[str],
        tpu_type: str,
        *,
        state: str = "UNKNOWN",
    ) -> TpuInfo:
        """Build a TpuInfo for direct-IP mode (no gcloud).

        Args:
            ips: List of host internal IPs.
            tpu_type: Accelerator type string (e.g. "v4-32").
            state: Synthetic state label.
        """
        return cls(
            name=None,
            project=None,
            zone=None,
            accelerator_type=tpu_type,
            internal_ips=list(ips),
            num_hosts=len(ips),
            state=state,
        )


# ── gcloud TPU discovery ─────────────────────────────────────────


def discover_tpu(name: str, project: str, zone: str) -> TpuInfo:
    """Describe a TPU and extract connection metadata.

    Uses `gcloud compute tpus tpu-vm describe` to get the accelerator
    type, internal IPs, and worker count.
    """
    info(f"Describing TPU {name} in {project}/{zone}...")
    raw = gcloud_json(
        [
            "compute",
            "tpus",
            "tpu-vm",
            "describe",
            name,
            "--project",
            project,
            "--zone",
            zone,
        ]
    )
    if not isinstance(raw, dict):
        raise ValueError(f"Unexpected gcloud response: {raw}")

    acc_type_raw = raw.get("acceleratorType", "v4-8")
    if "/" in acc_type_raw:
        acc_type_raw = acc_type_raw.split("/")[-1]

    endpoints = raw.get("networkEndpoints", [])
    ips = [ep.get("ipAddress", "") for ep in endpoints if ep.get("ipAddress")]
    if not ips:
        raise ValueError(f"No internal IPs found for TPU {name}")

    return TpuInfo(
        name=name,
        project=project,
        zone=zone,
        accelerator_type=acc_type_raw,
        internal_ips=ips,
        num_hosts=len(ips),
        state=raw.get("state", "UNKNOWN"),
    )


def list_tpus_in_zone(project: str, zone: str) -> list[dict]:
    """List all TPU VMs in a zone.

    Returns raw gcloud dicts with keys: name, state, acceleratorType,
    health, networkEndpoints, etc.
    """
    raw = gcloud_json(
        [
            "compute",
            "tpus",
            "tpu-vm",
            "list",
            "--project",
            project,
            "--zone",
            zone,
        ]
    )
    if not isinstance(raw, list):
        raw = [raw] if raw else []
    return raw


def list_tpus_in_project(project: str) -> list[dict]:
    """List all TPU VMs across ALL zones in a project.

    Adds a "zone" key to each TPU dict since gcloud --filter doesn't
    include zone by default when listing all zones.
    """
    raw = gcloud_json(
        [
            "compute",
            "tpus",
            "tpu-vm",
            "list",
            "--project",
            project,
            "--zone",
            "-",  # gcloud wildcard — scans all locations in the project
        ]
    )
    if not isinstance(raw, list):
        raw = [raw] if raw else []
    # Enrich with zone info from the name path
    for tpu in raw:
        name = tpu.get("name", "")
        if "/" in name:
            parts = name.split("/")
            # gcloud TPU name format: projects/{project}/locations/{zone}/nodes/{name}
            for marker in ("zones", "locations"):
                if marker in parts:
                    idx = parts.index(marker)
                    if idx + 1 < len(parts):
                        tpu["zone"] = parts[idx + 1]
                        break
    return raw


# ── SSH backends ─────────────────────────────────────────────────


def ssh_tpu_worker(
    tpu: TpuInfo,
    worker: int,
    command: str,
    *,
    timeout: int = 300,
) -> subprocess.CompletedProcess:
    """Run a command on a specific TPU worker via gcloud ssh.

    Only works when tpu.is_gcloud_managed is True.
    """
    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        tpu.name,
        "--worker",
        str(worker),
        "--project",
        tpu.project,
        "--zone",
        tpu.zone,
        "--command",
        command,
        "--no-tty-through-ssh",
    ]
    return run_command(cmd, timeout=timeout, check=False)


def ssh_to_ip(
    ip: str,
    command: str,
    *,
    timeout: int = 300,
    user: str | None = None,
    ssh_key: str | None = None,
    port: int = 22,
) -> subprocess.CompletedProcess:
    """Run a command on a host via plain SSH (direct-IP mode).

    Args:
        ip: Target host IP address.
        command: Shell command to execute on the host.
        timeout: Command timeout in seconds.
        user: SSH user (default: current user).
        ssh_key: Path to SSH private key.
        port: SSH port (default 22).
    """
    target = f"{user}@{ip}" if user else ip
    cmd = ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-p", str(port)]
    if ssh_key:
        cmd.extend(["-i", ssh_key])
    cmd.extend([target, command])
    return run_command(cmd, timeout=timeout, check=False)


def run_on_host(
    tpu: TpuInfo,
    host_index: int,
    command: str,
    *,
    timeout: int = 300,
    user: str | None = None,
    ssh_key: str | None = None,
) -> subprocess.CompletedProcess:
    """Run a command on a specific host.

    Dispatches to gcloud SSH or direct SSH depending on tpu.is_gcloud_managed.
    """
    if tpu.is_gcloud_managed:
        return ssh_tpu_worker(tpu, host_index, command, timeout=timeout)
    else:
        return ssh_to_ip(
            tpu.internal_ips[host_index],
            command,
            timeout=timeout,
            user=user,
            ssh_key=ssh_key,
        )


def run_on_all_hosts(
    tpu: TpuInfo,
    command: str,
    *,
    timeout: int = 300,
    user: str | None = None,
    ssh_key: str | None = None,
) -> list[tuple[int, subprocess.CompletedProcess]]:
    """Run a command on ALL hosts in parallel.

    Dispatches to gcloud SSH or direct SSH per host.
    Returns a list of (host_index, result) tuples sorted by index.
    """
    import concurrent.futures

    results: list[tuple[int, subprocess.CompletedProcess]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=tpu.num_hosts) as pool:
        futures = {
            pool.submit(run_on_host, tpu, w, command, timeout=timeout, user=user, ssh_key=ssh_key): w
            for w in range(tpu.num_hosts)
        }
        for future in concurrent.futures.as_completed(futures):
            worker = futures[future]
            results.append((worker, future.result()))

    results.sort(key=lambda x: x[0])
    return results


# ── Ray resource builder ─────────────────────────────────────────


def build_ray_resource_flags(tpu: TpuInfo, is_head: bool) -> str:
    """Build the Ray --resources JSON for a TPU host.

    Mirrors the resource allocation from tpu_setup.sh:
      Head:  TPU, TPU-{version}, TPU-{version}-{slice}-head, accelerator_type:TPU-{VERSION}, head-node
      Worker: TPU, TPU-{version}, accelerator_type:TPU-{VERSION}
    """
    version = tpu.tpu_version
    slice_size = tpu.slice_size
    version_upper = version.upper()

    if is_head:
        resources = {
            "TPU": tpu.chips_per_host,
            f"TPU-{version}": tpu.chips_per_host,
            f"TPU-{version}-{slice_size}-head": 1,
            f"TPU-{version}-{slice_size}-global-head": 1,
            f"accelerator_type:TPU-{version_upper}": 1,
            "head-node": 1,
            "ray-cluster-head": 1,
        }
    else:
        resources = {
            "TPU": tpu.chips_per_host,
            f"TPU-{version}": tpu.chips_per_host,
            f"accelerator_type:TPU-{version_upper}": 1,
        }
    return json.dumps(resources, separators=(",", ":"))
