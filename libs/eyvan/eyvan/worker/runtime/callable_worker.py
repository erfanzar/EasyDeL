# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Trusted-cell entry point for a signed callable workload artifact."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

from eyvan._internal.subprocess_env import provider_cli_env
from eyvan.worker.runtime import callable_protocol
from eyvan.worker.runtime.callable_protocol import GcsObjectTransfer, UriArtifactTransfer, execute_callable

# Preserve test seams for the shared HTTP retry implementation.  These are
# module objects, so monkeypatching them here affects callable_protocol too.
time = callable_protocol.time
urllib = callable_protocol.urllib


class GcloudStorageTransfer(GcsObjectTransfer):
    """Use the native worker's installed gcloud CLI for GCS object transfer."""

    @staticmethod
    def _uri(bucket_name: str, object_name: str) -> str:
        return f"gs://{bucket_name}/{object_name}"

    def read_object(self, bucket_name: str, object_name: str) -> bytes:
        """Download one callable artifact with the trusted native worker identity."""

        return subprocess.run(
            ["gcloud", "storage", "cat", self._uri(bucket_name, object_name)],
            check=True,
            capture_output=True,
            env=provider_cli_env(),
        ).stdout

    def write_object(self, bucket_name: str, object_name: str, payload: bytes) -> None:
        """Upload one result artifact with the trusted native worker identity."""

        with tempfile.NamedTemporaryFile("wb", suffix=".result", delete=False) as temporary:
            temporary.write(payload)
            temporary_path = Path(temporary.name)
        try:
            subprocess.run(
                ["gcloud", "storage", "cp", str(temporary_path), self._uri(bucket_name, object_name)],
                check=True,
                capture_output=True,
                env=provider_cli_env(),
            )
        finally:
            temporary_path.unlink(missing_ok=True)


class GcsMetadataTransfer(GcsObjectTransfer):
    """Use the TPU VM's metadata-server identity for GCS JSON API access."""

    _METADATA_TOKEN_URL = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"

    def _access_token(self) -> str:
        request = urllib.request.Request(
            self._METADATA_TOKEN_URL,
            headers={"Metadata-Flavor": "Google"},
        )
        # The metadata server is link-local: reach it directly, with none of the
        # caller's proxy settings in the path. And trust only a reply that
        # identifies itself as the metadata service -- anything else answering
        # on that address (an SSRF proxy, a stray local service, a captive
        # portal) must not be handing this worker a credential. See
        # ``eyvan.identity.metadata`` for the same rule on the identity path.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=10) as response:
            if response.headers.get("Metadata-Flavor", "") != "Google":
                raise RuntimeError("the metadata token reply did not identify itself as the Google metadata service")
            payload = json.loads(response.read().decode())
        return payload["access_token"]

    def read_object(self, bucket_name: str, object_name: str) -> bytes:
        """Download one callable artifact through the GCS JSON API."""

        url = (
            "https://storage.googleapis.com/storage/v1/b/"
            f"{urllib.parse.quote(bucket_name, safe='')}/o/"
            f"{urllib.parse.quote(object_name, safe='')}?alt=media"
        )
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {self._access_token()}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()

    def write_object(self, bucket_name: str, object_name: str, payload: bytes) -> None:
        """Upload one result artifact through the GCS JSON API."""

        url = (
            "https://storage.googleapis.com/upload/storage/v1/b/"
            f"{urllib.parse.quote(bucket_name, safe='')}/o"
            f"?uploadType=media&name={urllib.parse.quote(object_name, safe='')}"
        )
        request = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self._access_token()}",
                "Content-Type": "application/octet-stream",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            response.read()


def main() -> int:
    """Run a signed callable through the native worker's GCS transport."""

    return execute_callable(
        None,
        transfer=UriArtifactTransfer(GcsMetadataTransfer(), os.environ),
        environment=os.environ,
    )


if __name__ == "__main__":
    raise SystemExit(main())
