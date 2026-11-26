# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import http.client
import io
import sys
import urllib.error
import urllib.request
from dataclasses import replace
from types import TracebackType

import pytest

from eyvan._internal.json_types import JsonObject
from eyvan.capacity.drivers.base import (
    CapacityStatus,
    CellObservation,
    ExecutionWorkloadStatus,
    MutationContext,
)
from eyvan.worker.runtime import callable_worker
from eyvan.worker.runtime.callable_worker import main as callable_worker_main
from eyvan.worker.runtime.callables import dump_callable, load_callable
from eyvan.worker.runtime.results import dump_result, load_result


def _increment(value: int) -> int:
    return value + 1


def test_callable_artifact_checks_digest_signature_and_python() -> None:
    artifact = dump_callable(_increment, signing_key=b"secret")

    loaded = load_callable(artifact, signing_key=b"secret")

    assert loaded(4) == 5
    with pytest.raises(ValueError, match="digest"):
        load_callable(replace(artifact, payload=artifact.payload + b"broken"), signing_key=b"secret")
    with pytest.raises(ValueError, match="signature"):
        load_callable(artifact, signing_key=b"wrong")
    with pytest.raises(ValueError, match="Python"):
        load_callable(replace(artifact, python="0.0"), signing_key=b"secret")


def test_callable_result_round_trips_arbitrary_python_and_checks_integrity() -> None:
    value = {"tuple": (1, 2), "bytes": b"payload"}
    artifact = dump_result(value, signing_key=b"secret")

    assert load_result(artifact, signing_key=b"secret") == value
    with pytest.raises(ValueError, match="digest"):
        load_result(artifact + b"broken", signing_key=b"secret")
    with pytest.raises(ValueError, match="signature"):
        load_result(artifact, signing_key=b"wrong")


def test_callable_worker_commits_result_artifact(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    callable_artifact = dump_callable(_increment)
    callable_path = tmp_path / "callable.pkl"
    result_path = tmp_path / "result.bin"
    callable_path.write_bytes(callable_artifact.payload)
    monkeypatch.setenv("EYVAN_CALL_ARGS_JSON", "[4]")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eyvan-callable-worker",
            "--uri",
            str(callable_path),
            "--sha256",
            callable_artifact.sha256,
            "--python",
            callable_artifact.python,
            "--result-uri",
            str(result_path),
        ],
    )

    callable_worker_main()

    assert load_result(result_path.read_bytes()) == 5


@pytest.mark.parametrize(
    ("environment", "writes_result"),
    [
        ({"EYVAN_REPLICA_COUNT": "2", "EYVAN_REPLICA_INDEX": "0"}, True),
        ({"EYVAN_REPLICA_COUNT": "2", "EYVAN_REPLICA_INDEX": "1"}, False),
        (
            {
                "EYVAN_DISTRIBUTED_PROCESS_COUNT": "2",
                "EYVAN_DISTRIBUTED_PROCESS_ID": "0",
                "EYVAN_REPLICA_COUNT": "2",
                "EYVAN_REPLICA_INDEX": "0",
            },
            True,
        ),
        (
            {
                "EYVAN_DISTRIBUTED_PROCESS_COUNT": "2",
                "EYVAN_DISTRIBUTED_PROCESS_ID": "1",
                "EYVAN_REPLICA_COUNT": "2",
                "EYVAN_REPLICA_INDEX": "1",
            },
            False,
        ),
    ],
)
def test_callable_worker_only_the_leader_writes_a_gang_result(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    writes_result: bool,
) -> None:
    callable_artifact = dump_callable(_increment)
    callable_path = tmp_path / "callable.pkl"
    result_path = tmp_path / "result.bin"
    callable_path.write_bytes(callable_artifact.payload)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("EYVAN_CALL_ARGS_JSON", "[4]")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eyvan-callable-worker",
            "--uri",
            str(callable_path),
            "--sha256",
            callable_artifact.sha256,
            "--python",
            callable_artifact.python,
            "--result-uri",
            str(result_path),
        ],
    )

    callable_worker_main()

    assert result_path.exists() is writes_result
    if writes_result:
        assert load_result(result_path.read_bytes()) == 5


@pytest.mark.parametrize(
    "environment",
    [
        {"EYVAN_REPLICA_COUNT": "2"},
        {"EYVAN_REPLICA_COUNT": "2", "EYVAN_REPLICA_INDEX": "2"},
        {
            "EYVAN_DISTRIBUTED_PROCESS_COUNT": "2",
            "EYVAN_DISTRIBUTED_PROCESS_ID": "1",
            "EYVAN_REPLICA_COUNT": "2",
            "EYVAN_REPLICA_INDEX": "0",
        },
    ],
)
def test_callable_worker_rejects_invalid_gang_result_ownership(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
) -> None:
    callable_artifact = dump_callable(_increment)
    callable_path = tmp_path / "callable.pkl"
    result_path = tmp_path / "result.bin"
    callable_path.write_bytes(callable_artifact.payload)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("EYVAN_CALL_ARGS_JSON", "[4]")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eyvan-callable-worker",
            "--uri",
            str(callable_path),
            "--sha256",
            callable_artifact.sha256,
            "--python",
            callable_artifact.python,
            "--result-uri",
            str(result_path),
        ],
    )

    with pytest.raises(RuntimeError, match="Eyvan callable"):
        callable_worker_main()

    assert not result_path.exists()


class FakeHTTPResponse:
    def __init__(self, payload: bytes = b""):
        self.payload = payload

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


class IncompleteHTTPResponse(FakeHTTPResponse):
    def read(self) -> bytes:
        raise http.client.IncompleteRead(b"partial")


def test_callable_worker_transfers_managed_artifacts_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = dump_callable(_increment)
    uploaded: list[urllib.request.Request] = []
    source_attempts = [0]
    sleeps: list[float] = []

    def urlopen(request: str | urllib.request.Request, *, timeout: int) -> FakeHTTPResponse:
        assert timeout == 60
        if isinstance(request, urllib.request.Request) and request.get_method() == "GET":
            assert request.full_url == "http://control-plane/callable"
            assert request.get_header("X-eyvan-artifact-token") == "read-token"
            source_attempts[0] += 1
            if source_attempts[0] == 1:
                raise urllib.error.HTTPError(
                    request.full_url,
                    503,
                    "temporarily unavailable",
                    {"Retry-After": "0"},
                    io.BytesIO(),
                )
            return FakeHTTPResponse(artifact.payload)
        assert isinstance(request, urllib.request.Request)
        uploaded.append(request)
        if len(uploaded) == 1:
            raise urllib.error.URLError("connection reset after write")
        if len(uploaded) == 2:
            return IncompleteHTTPResponse()
        return FakeHTTPResponse()

    monkeypatch.setattr(callable_worker.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(callable_worker.time, "sleep", sleeps.append)
    monkeypatch.setenv("EYVAN_MANAGED_CALLABLE_URL", "http://control-plane/callable")
    monkeypatch.setenv("EYVAN_MANAGED_CALLABLE_TOKEN", "read-token")
    monkeypatch.setenv("EYVAN_MANAGED_RESULT_URL", "http://control-plane/result")
    monkeypatch.setenv("EYVAN_MANAGED_RESULT_TOKEN", "write-token")
    monkeypatch.setenv("EYVAN_ALLOW_INSECURE_ARTIFACT_HTTP", "1")
    monkeypatch.setenv("EYVAN_CALL_ARGS_JSON", "[4]")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eyvan-callable-worker",
            "--uri-env",
            "EYVAN_MANAGED_CALLABLE_URL",
            "--uri-token-env",
            "EYVAN_MANAGED_CALLABLE_TOKEN",
            "--sha256",
            artifact.sha256,
            "--python",
            artifact.python,
            "--result-uri-env",
            "EYVAN_MANAGED_RESULT_URL",
            "--result-token-env",
            "EYVAN_MANAGED_RESULT_TOKEN",
        ],
    )

    callable_worker_main()

    assert source_attempts == [2]
    assert len(uploaded) == 3
    assert uploaded[2].full_url == "http://control-plane/result"
    assert uploaded[2].get_method() == "PUT"
    assert uploaded[2].get_header("X-eyvan-artifact-token") == "write-token"
    assert load_result(uploaded[2].data or b"") == 5
    assert sleeps == [0.0, 0.25, 0.5]


def test_callable_worker_rejects_insecure_managed_artifact_http(monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = dump_callable(_increment)
    monkeypatch.setenv("EYVAN_MANAGED_CALLABLE_URL", "http://control-plane/callable?token=read-token")
    monkeypatch.delenv("EYVAN_ALLOW_INSECURE_ARTIFACT_HTTP", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eyvan-callable-worker",
            "--uri-env",
            "EYVAN_MANAGED_CALLABLE_URL",
            "--sha256",
            artifact.sha256,
            "--python",
            artifact.python,
        ],
    )

    with pytest.raises(ValueError, match="HTTP callable artifacts require"):
        callable_worker_main()


def _gang() -> JsonObject:
    return {
        "replicas": 2,
        "per_replica": {
            "cpu": 4,
            "memory_gib": 16,
            "gpu": 1,
            "tpu": 2,
            "custom": {"accelerator_type:v6e": 1, "fabric_bandwidth": 0.5},
        },
        "placement_strategy": "STRICT_SPREAD",
    }


def _fence(context: MutationContext) -> dict[str, str]:
    return {
        "eyvan_generation": str(context.generation),
        "eyvan_fencing_token": context.fencing_token,
    }


def test_execution_workload_indeterminate_status_has_an_explicit_wire_value() -> None:
    assert ExecutionWorkloadStatus.INDETERMINATE.value == "INDETERMINATE"


def test_resolver_raises_instead_of_asserting_when_worker_token_derivation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import eyvan.worker.runtime.native_bootstrap as native_bootstrap
    from eyvan.worker.runtime.registry import ExecutionDriverResolver

    monkeypatch.setattr(
        native_bootstrap,
        "derive_native_worker_token",
        lambda *args, **kwargs: None,
    )
    resolver = ExecutionDriverResolver(
        backend="eyvan-native",
        native_worker_token="root-secret",
    )
    observation = CellObservation(
        "cell-a",
        CapacityStatus.HEALTHY,
        1,
        execution_endpoint="http://worker",
        metadata={"management_uid": "uid-1"},
    )

    with pytest.raises(ValueError, match="could not derive a native worker token"):
        resolver.resolve("cell-a", observation)
