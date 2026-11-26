# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from eyvan.worker.runtime.callables import dump_callable
from eyvan.worker.runtime.oci_callable_stager import OCI_CALLABLE_RUNNER_FILENAME, stage_oci_callable_runner
from eyvan.worker.runtime.results import load_result


def _install_fake_gcs_client(directory: Path) -> None:
    package = directory / "google" / "cloud"
    package.mkdir(parents=True)
    (directory / "google" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "storage.py").write_text(
        """
import os
from pathlib import Path


class Blob:
    def __init__(self, path: Path) -> None:
        self.path = path

    def download_as_bytes(self) -> bytes:
        return self.path.read_bytes()

    def upload_from_string(self, data: bytes, *, content_type: str) -> None:
        assert content_type == "application/octet-stream"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(data)


class Bucket:
    def __init__(self, root: Path, name: str) -> None:
        self.root = root
        self.name = name

    def blob(self, blob_name: str) -> Blob:
        return Blob(self.root / self.name / blob_name)


class Client:
    def __init__(self) -> None:
        self.root = Path(os.environ["EYVAN_TEST_GCS_ROOT"])

    def bucket(self, bucket_name: str) -> Bucket:
        return Bucket(self.root, bucket_name)
""".lstrip(),
        encoding="utf-8",
    )
    for package_name in ("eyvan", "ray", "eray"):
        (directory / f"{package_name}.py").write_text(
            f'raise RuntimeError("{package_name} must not be imported by the OCI callable runner")\n',
            encoding="utf-8",
        )


def _runner_environment(fake_modules: Path, gcs_root: Path) -> dict[str, str]:
    environment = dict(os.environ)
    inherited_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(value for value in (str(fake_modules), inherited_python_path) if value)
    environment["EYVAN_TEST_GCS_ROOT"] = str(gcs_root)
    environment["EYVAN_CALL_ARGS_JSON"] = json.dumps([7])
    environment["EYVAN_CALL_KWARGS_JSON"] = json.dumps({"offset": 5})
    environment["EYVAN_CALLABLE_SIGNING_KEY"] = "oci-test-signing-key"
    return environment


def _run_runner(
    runner: Path,
    *,
    sha256: str,
    python: str,
    signature: str,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (
            sys.executable,
            str(runner),
            "--uri",
            "gs://test-bucket/callables/invocation.pkl",
            "--sha256",
            sha256,
            "--python",
            python,
            "--format-version",
            "1",
            "--signature",
            signature,
            "--result-uri",
            "gs://test-bucket/results/invocation.bin",
        ),
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )


def test_staged_oci_runner_executes_a_signed_gcs_callable_without_eyvan_ray_or_eray(tmp_path: Path) -> None:
    def target(value: int, *, offset: int) -> dict[str, int]:
        return {"value": value + offset}

    signing_key = b"oci-test-signing-key"
    callable_artifact = dump_callable(target, signing_key=signing_key)
    assert callable_artifact.signature is not None
    gcs_root = tmp_path / "gcs"
    artifact_path = gcs_root / "test-bucket" / "callables" / "invocation.pkl"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(callable_artifact.payload)
    fake_modules = tmp_path / "fake-modules"
    _install_fake_gcs_client(fake_modules)
    runner = stage_oci_callable_runner(tmp_path / "native-worker-state" / "oci-callable-runner")
    environment = _runner_environment(fake_modules, gcs_root)

    incompatible = _run_runner(
        runner,
        sha256=callable_artifact.sha256,
        python="0.0",
        signature=callable_artifact.signature,
        environment=environment,
    )

    assert incompatible.returncode != 0
    assert "callable requires Python 0.0" in incompatible.stderr
    assert not (gcs_root / "test-bucket" / "results" / "invocation.bin").exists()

    completed = _run_runner(
        runner,
        sha256=callable_artifact.sha256,
        python=callable_artifact.python,
        signature=callable_artifact.signature,
        environment=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert runner.name == OCI_CALLABLE_RUNNER_FILENAME
    result_path = gcs_root / "test-bucket" / "results" / "invocation.bin"
    assert load_result(result_path.read_bytes(), signing_key=signing_key) == {"value": 12}
