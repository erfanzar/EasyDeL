# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pin the unsigned standalone boundary separately from optional runtime HMAC."""

from __future__ import annotations

import typing as tp
from pathlib import Path

import pytest

from eyvan.artifacts import ArtifactStore
from eyvan.sdk.client.callables import EyvanCallableSubmitter
from eyvan.sdk.client.sync import EyvanClient
from eyvan.worker.runtime.callable_protocol import (
    UriArtifactTransfer,
    dump_callable,
    execute_callable,
    load_result,
)


class _NoCloudTransfer:
    """Fail if the local contract proof ever reaches a provider boundary."""

    def read_object(self, bucket_name: str, object_name: str) -> bytes:
        raise AssertionError("local contract proof must not contact GCS")

    def write_object(self, bucket_name: str, object_name: str, payload: bytes) -> None:
        raise AssertionError("local contract proof must not contact GCS")


def test_worker_reference_distinguishes_standalone_from_optional_hmac() -> None:
    document = (Path(__file__).resolve().parents[1] / "docs/reference/worker-runtime.md").read_text()
    callable_section = document.split("## 4. Callable lifecycle", 1)[1].split("## 5. JAX", 1)[0]
    normalized = " ".join(callable_section.split())

    assert "Operators should configure the callable signing key" not in normalized
    assert "standalone SDK/API callables are unsigned" in normalized
    assert "rejects caller signing keys and artifact signatures" in normalized
    assert "custom submitter" in normalized
    assert "does not distribute a producer/verifier key" in normalized
    assert "digest alone does not make an untrusted callable safe" in normalized


@pytest.mark.parametrize("signed", [False, True])
def test_local_execution_separates_unsigned_artifacts_from_optional_hmac(tmp_path: Path, signed: bool) -> None:
    key = b"local-contract-key"
    artifact = dump_callable(lambda: {"value": 19}, signing_key=key if signed else None)
    assert (artifact.signature is not None) is signed
    source = tmp_path / "callable.pkl"
    result = tmp_path / "result.bin"
    source.write_bytes(artifact.payload)
    argv = [
        "--uri",
        str(source),
        "--sha256",
        artifact.sha256,
        "--python",
        artifact.python,
        "--result-uri",
        str(result),
    ]
    if artifact.signature is not None:
        argv.extend(["--signature", artifact.signature])
    environment = {"EYVAN_CALLABLE_SIGNING_KEY": key.decode()}
    transfer = UriArtifactTransfer(_NoCloudTransfer(), environment)

    if not signed:
        with pytest.raises(ValueError, match="callable artifact signature mismatch"):
            execute_callable(argv, transfer=transfer, environment=environment)
        assert not result.exists()
        assert execute_callable(argv, transfer=transfer, environment={}) == 0
        assert load_result(result.read_bytes()) == {"value": 19}
    else:
        assert execute_callable(argv, transfer=transfer, environment=environment) == 0
        assert load_result(result.read_bytes(), signing_key=key) == {"value": 19}


def test_standalone_submitter_refuses_signing_configuration_before_io() -> None:
    # Constructor validation runs before either inert dependency is accessed.
    with pytest.raises(ValueError, match="remote callable signing keys are not supported"):
        EyvanCallableSubmitter(
            tp.cast(EyvanClient, object()),
            tp.cast(ArtifactStore, object()),
            signing_key=b"local-contract-key",
        )
