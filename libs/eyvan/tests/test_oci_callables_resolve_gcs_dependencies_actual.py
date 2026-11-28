# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""OCI callables must resolve GCS dependencies only for actual GCS transfers."""

from __future__ import annotations

import io
import sys
import types
import urllib.request
from pathlib import Path

import pytest

from eyvan.worker.runtime import oci_callable_runner
from eyvan.worker.runtime.callable_protocol import CallableArtifact, dump_callable, load_result

_SIGNING_KEY = b"writer-117-signing-key"


def _invocation(monkeypatch: pytest.MonkeyPatch, source_uri: str, result_uri: str) -> tuple[CallableArtifact, list[str]]:
    def target(value: int) -> dict[str, int]:
        return {"value": value + 5}

    artifact = dump_callable(target, signing_key=_SIGNING_KEY)
    assert artifact.signature is not None
    monkeypatch.setenv("EYVAN_CALLABLE_SIGNING_KEY", _SIGNING_KEY.decode())
    monkeypatch.setenv("EYVAN_CALL_ARGS_JSON", "[7]")
    monkeypatch.setenv("EYVAN_CALL_KWARGS_JSON", "{}")
    monkeypatch.delenv("EYVAN_NATIVE_WORKER_STATE_DIR", raising=False)
    monkeypatch.delenv("EYVAN_REPLICA_INDEX", raising=False)
    return artifact, [
        "--uri",
        source_uri,
        "--result-uri",
        result_uri,
        "--sha256",
        artifact.sha256,
        "--python",
        artifact.python,
        "--signature",
        artifact.signature,
    ]


@pytest.mark.parametrize("file_uri", [False, True])
def test_oci_local_callable_does_not_require_google_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, file_uri: bool
) -> None:
    source = tmp_path / "callable.pkl"
    result = tmp_path / "result.bin"
    artifact, argv = _invocation(
        monkeypatch,
        source.as_uri() if file_uri else str(source),
        result.as_uri() if file_uri else str(result),
    )
    source.write_bytes(artifact.payload)
    # Models an image without the optional dependency, even if the host has it.
    monkeypatch.setitem(sys.modules, "google.cloud", None)

    assert oci_callable_runner.main(argv) == 0
    assert load_result(result.read_bytes(), signing_key=_SIGNING_KEY) == {"value": 12}


@pytest.mark.parametrize("scheme", ["https", "http"])
def test_oci_http_callable_does_not_construct_an_adc_client(monkeypatch: pytest.MonkeyPatch, scheme: str) -> None:
    source_uri = f"{scheme}://artifacts.example/callable"
    result_uri = f"{scheme}://artifacts.example/result"
    artifact, argv = _invocation(monkeypatch, source_uri, result_uri)
    monkeypatch.setenv("EYVAN_ALLOW_INSECURE_ARTIFACT_HTTP", "1")
    monkeypatch.setenv("WRITER_117_READ_TOKEN", "read-capability")
    monkeypatch.setenv("WRITER_117_WRITE_TOKEN", "write-capability")
    argv.extend(["--uri-token-env", "WRITER_117_READ_TOKEN", "--result-token-env", "WRITER_117_WRITE_TOKEN"])
    cloud = types.ModuleType("google.cloud")
    storage = types.ModuleType("google.cloud.storage")

    def forbidden_client() -> None:
        raise AssertionError("non-GCS callable must not resolve ADC")

    monkeypatch.setattr(storage, "Client", forbidden_client, raising=False)
    monkeypatch.setattr(cloud, "storage", storage, raising=False)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud)
    requests: list[urllib.request.Request] = []
    results: list[bytes] = []

    def urlopen(request: urllib.request.Request, *, timeout: int, **options: object) -> io.BytesIO:
        assert timeout == 60
        requests.append(request)
        if request.get_method() == "GET":
            assert request.full_url == source_uri
            assert request.get_header("X-eyvan-artifact-token") == "read-capability"
            return io.BytesIO(artifact.payload)
        assert request.get_method() == "PUT"
        assert request.full_url == result_uri
        assert request.get_header("X-eyvan-artifact-token") == "write-capability"
        assert request.data is not None
        results.append(request.data)
        return io.BytesIO(b"")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    assert oci_callable_runner.main(argv) == 0
    assert [request.get_method() for request in requests] == ["GET", "PUT"]
    assert len(results) == 1
    assert load_result(results[0], signing_key=_SIGNING_KEY) == {"value": 12}


def test_oci_gcs_callable_still_requires_google_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    _, argv = _invocation(monkeypatch, "gs://bucket/callable", "gs://bucket/result")
    monkeypatch.setitem(sys.modules, "google.cloud", None)

    with pytest.raises(RuntimeError, match="require google-cloud-storage in the workload image"):
        oci_callable_runner.main(argv)
