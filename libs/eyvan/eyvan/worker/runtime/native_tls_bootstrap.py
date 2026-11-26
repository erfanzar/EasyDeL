# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Stage an explicitly supplied immutable GCP worker TLS bundle, never issue one.

Only the bootstrap systemd unit calls this helper. Its root-owned runtime
parent is private; workload processes must not have access. Fleet endpoint/SAN
selection and per-cell certificate issuance are separate operator contracts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import subprocess
import tempfile
from pathlib import Path

from cryptography.hazmat.primitives.serialization import load_pem_private_key

from eyvan._internal.subprocess_env import provider_cli_env
from eyvan.identity.mtls import MutualTlsConfigurationError, MutualTlsFiles, MutualTlsMode, server_ssl_context

_FIELDS = ("trust_bundle", "certificate", "private_key")
_VERSION = re.compile(r"projects/[A-Za-z0-9-]+/secrets/[A-Za-z0-9_-]+/versions/[1-9][0-9]*", re.ASCII)


def validate_tls_secret_resource(resource: str) -> str:
    """Refuse aliases, inline material and shell/systemd metacharacters."""
    if not isinstance(resource, str) or len(resource) > 1024 or _VERSION.fullmatch(resource) is None:
        raise ValueError("worker TLS requires an immutable Secret Manager version resource")
    return resource


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate TLS bundle field")
        result[name] = value
    return result


def install_native_worker_tls(resource: str, destination: Path) -> MutualTlsFiles:
    """Fetch, validate and publish a complete owner-only bundle or publish nothing.

    The destination must be absent in an existing owner-only parent. Systemd
    removes the runtime directory when stopping the service; this helper never
    overwrites an existing key directory. No provider stderr or payload appears
    in diagnostics. It validates CA parsing and key-pair consistency, not SAN
    reachability, issuance policy, expiry or trust by a remote client.
    """
    validate_tls_secret_resource(resource)
    parent = destination.parent
    try:
        info = parent.stat()
        if (
            not destination.is_absolute()
            or parent.is_symlink()
            or not parent.is_dir()
            or info.st_uid != os.geteuid()
            or info.st_mode & 0o077
            or os.path.lexists(destination)
        ):
            raise ValueError("worker TLS destination must be absent in a private owned directory")
        fetched = subprocess.run(
            ["gcloud", "secrets", "versions", "access", resource, "--quiet"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=30,
            env=provider_cli_env(),
        ).stdout
        if len(fetched) > 65536:
            raise ValueError("oversized TLS bundle")
        bundle = json.loads(fetched, object_pairs_hook=_unique_object)
        if not isinstance(bundle, dict) or set(bundle) != set(_FIELDS):
            raise ValueError("invalid TLS bundle fields")
        if any(not isinstance(bundle[name], str) or not bundle[name].strip() for name in _FIELDS):
            raise ValueError("invalid TLS bundle material")
        # Refuse encrypted keys without OpenSSL's interactive password prompt.
        load_pem_private_key(bundle["private_key"].encode(), password=None)
        with tempfile.TemporaryDirectory(prefix=".tls-", dir=parent) as scratch:
            stage = Path(scratch)
            stage.chmod(0o700)
            for name in _FIELDS:
                path = stage / f"{name}.pem"
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "wb") as output:
                    os.fchmod(output.fileno(), 0o600)
                    output.write(bundle[name].encode())
            files = MutualTlsFiles(**{name: str(stage / f"{name}.pem") for name in _FIELDS})
            server_ssl_context(MutualTlsMode.REQUIRED, files)
            stage.rename(destination)
    except (OSError, ValueError, TypeError, subprocess.SubprocessError, ssl.SSLError, MutualTlsConfigurationError):
        raise ValueError("worker TLS bundle could not be installed; check reference, material and directory") from None
    return MutualTlsFiles(**{name: str(destination / f"{name}.pem") for name in _FIELDS})


def main() -> None:
    """Run the silent systemd pre-start stage, with bounded private failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource", required=True)
    parser.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args()
    try:
        install_native_worker_tls(args.resource, args.directory)
    except ValueError:
        parser.exit(1, "Eyvan worker TLS bootstrap failed; verify the pinned bundle and private runtime directory.\n")


if __name__ == "__main__":
    main()
