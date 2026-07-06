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

"""GCP access primitives shared across eray: gcloud subprocess + GCE metadata.

Click-free and CLI-free so both `eray.cli` and `eray.provision` can build on
it without import cycles. All GCP interaction in eray goes through the gcloud
CLI (subprocess) — no google-api client dependencies.
"""

from __future__ import annotations

import json
import logging
import subprocess

logger = logging.getLogger("eray.gcp")

GCE_METADATA_BASE = "http://metadata.google.internal/computeMetadata/v1"


def run_command(
    cmd: list[str],
    *,
    timeout: int = 300,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess:
    """Run a subprocess command and return the result.

    Args:
        cmd: Command and arguments as a list of strings.
        timeout: Maximum time to wait for the command in seconds.
        check: If True, raise CalledProcessError on non-zero exit.
        capture: If True, capture stdout and stderr.

    Returns:
        A subprocess.CompletedProcess with returncode, stdout, and stderr.

    Raises:
        subprocess.CalledProcessError: If check is True and the command exits with a non-zero code.
        subprocess.TimeoutExpired: If the command exceeds the timeout.
    """
    logger.debug(f"Running: {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        timeout=timeout,
        check=check,
    )


def gcloud(args: list[str], *, timeout: int = 300, check: bool = True) -> str:
    """Run a gcloud command and return stdout.

    Args:
        args: gcloud subcommand and arguments as a list of strings.
        timeout: Maximum time to wait for the command in seconds.
        check: If True, raise CalledProcessError on non-zero exit.

    Returns:
        The stripped stdout output from the command.

    Raises:
        subprocess.CalledProcessError: If check is True and gcloud exits with a non-zero code.
    """
    result = run_command(["gcloud", *args], timeout=timeout, check=check)
    return result.stdout.strip()


def gcloud_json(args: list[str], *, timeout: int = 300) -> dict | list:
    """Run a gcloud command and parse JSON output.

    Args:
        args: gcloud subcommand and arguments as a list of strings.
        timeout: Maximum time to wait for the command in seconds.

    Returns:
        Parsed JSON output as a dict or list.

    Raises:
        subprocess.CalledProcessError: If gcloud exits with a non-zero code.
        json.JSONDecodeError: If the output is not valid JSON.
    """
    result = run_command(["gcloud", *args, "--format=json"], timeout=timeout, check=True)
    return json.loads(result.stdout)


def _gce_metadata(path: str, timeout: float = 2.0) -> str | None:
    """Read one value from the GCE metadata server.

    Args:
        path: Metadata path below the v1 root, e.g.
            ``instance/attributes/accelerator-type``.
        timeout: Request timeout in seconds; kept short so non-GCP machines
            fail fast.

    Returns:
        The value as a stripped string, or None when the metadata server is
        unreachable (not on GCP) or the key does not exist.
    """
    import requests

    try:
        r = requests.get(f"{GCE_METADATA_BASE}/{path}", headers={"Metadata-Flavor": "Google"}, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.text.strip()
    except Exception:
        return None


def detect_project() -> str | None:
    """Detect GCP project from metadata server, then gcloud config.

    Returns:
        The GCP project ID, or None if it cannot be determined.
    """
    proj = _gce_metadata("project/project-id")
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
    """Detect GCP zone from metadata server, then gcloud config.

    Returns:
        The GCP zone name, or None if it cannot be determined.
    """
    zone_raw = _gce_metadata("instance/zone")
    if zone_raw:
        return zone_raw.split("/")[-1]
    try:
        result = gcloud(["config", "get-value", "compute/zone"], check=False)
        if result and result != "(unset)":
            return result
    except Exception:
        pass
    return None
