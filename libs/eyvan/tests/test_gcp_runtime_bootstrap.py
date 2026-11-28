# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import hashlib
import subprocess
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from eyvan.capacity.providers.gcp.runtime.bootstrap import (
    GcpRuntimeBootstrap,
    GcpRuntimeBootstrapStep,
    GcpRuntimeBucketEnsureError,
    GcpRuntimePreflightError,
    GcpRuntimeWheelBuildError,
    GcpRuntimeWheelStageError,
)
from eyvan.capacity.providers.gcp.runtime.profile import GcpNativeRuntimeProfile


@dataclass
class FakeWheelBuilder:
    """Return one pre-created wheel path and record build requests."""

    path: Path
    calls: int = 0

    def build(self) -> Path:
        self.calls += 1
        return self.path


@dataclass
class FakeGcloudRunner:
    """Small in-memory GCS and preflight command boundary."""

    objects: dict[str, bytes] = field(default_factory=dict)
    failures: dict[tuple[str, ...], tuple[int, bytes, bytes]] = field(default_factory=dict)
    copy_races: set[str] = field(default_factory=set)
    calls: list[tuple[str, ...]] = field(default_factory=list)
    timeouts: list[float] = field(default_factory=list)

    def __call__(
        self,
        args: tp.Sequence[str],
        *,
        check: bool,
        capture_output: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[bytes]:
        command = tuple(args)
        self.calls.append(command)
        self.timeouts.append(timeout)
        assert check is False
        assert capture_output is True
        configured = self.failures.get(command)
        if configured is not None:
            return subprocess.CompletedProcess(command, configured[0], stdout=configured[1], stderr=configured[2])
        if command[:4] == ("gcloud", "storage", "objects", "describe"):
            uri = command[4]
            if uri in self.objects:
                return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")
            return subprocess.CompletedProcess(command, 1, stdout=b"", stderr=b"404 object not found")
        if command[:3] == ("gcloud", "storage", "cp"):
            source = Path(command[4])
            destination = command[5]
            if destination in self.copy_races:
                self.copy_races.remove(destination)
                self.objects[destination] = source.read_bytes()
                return subprocess.CompletedProcess(command, 1, stdout=b"", stderr=b"already exists")
            if destination in self.objects:
                return subprocess.CompletedProcess(command, 1, stdout=b"", stderr=b"already exists")
            self.objects[destination] = source.read_bytes()
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")


def profile() -> GcpNativeRuntimeProfile:
    return GcpNativeRuntimeProfile(
        project_id="project-a",
        artifact_root_uri="gs://eyvan-runtime-artifacts/eyvan",
        agent_shared_secret="a" * 32,
        native_worker_root_secret="b" * 32,
    )


def create_wheel(tmp_path: Path, payload: bytes = b"eyvan native worker wheel") -> Path:
    wheel = tmp_path / "eyvan-0.1.0-py3-none-any.whl"
    wheel.write_bytes(payload)
    return wheel


def test_preflight_uses_only_read_only_gcloud_commands_and_derives_destinations(tmp_path: Path) -> None:
    runner = FakeGcloudRunner()
    bootstrap = GcpRuntimeBootstrap(FakeWheelBuilder(create_wheel(tmp_path)), runner=runner, command_timeout_s=15)
    runtime_profile = profile()

    checked = bootstrap.preflight(runtime_profile)

    assert checked.project_id == "project-a"
    assert checked.artifact_bucket_uri == "gs://eyvan-runtime-artifacts"
    assert checked.fleet_state_uri == "gs://eyvan-runtime-artifacts/eyvan/control-plane/fleet-state.json"
    assert checked.fleet_state_uri == runtime_profile.fleet_state_uri
    assert checked.checks == (
        GcpRuntimeBootstrapStep.PROJECT_ACCESS,
        GcpRuntimeBootstrapStep.STORAGE_ACCESS,
        GcpRuntimeBootstrapStep.TPU_ACCESS,
    )
    assert runner.calls == [
        ("gcloud", "projects", "describe", "project-a", "--format=value(projectId)"),
        (
            "gcloud",
            "storage",
            "buckets",
            "describe",
            "gs://eyvan-runtime-artifacts",
            "--project=project-a",
            "--format=value(name)",
        ),
        (
            "gcloud",
            "compute",
            "tpus",
            "locations",
            "list",
            "--project=project-a",
            "--limit=1",
            "--format=value(locationId)",
        ),
    ]
    assert runner.timeouts == [15.0, 15.0, 15.0]


def test_bootstrap_stages_a_content_addressed_wheel_with_no_clobber(tmp_path: Path) -> None:
    payload = b"immutable native worker wheel"
    wheel = create_wheel(tmp_path, payload)
    builder = FakeWheelBuilder(wheel)
    runner = FakeGcloudRunner()
    bootstrap = GcpRuntimeBootstrap(builder, runner=runner)
    original = profile()

    updated = bootstrap.bootstrap(original)

    digest = hashlib.sha256(payload).hexdigest()
    target_uri = f"gs://eyvan-runtime-artifacts/eyvan/runtime/wheels/{digest}/{wheel.name}"
    assert updated is not original
    assert updated.native_worker_wheel_uri == target_uri
    assert original.native_worker_wheel_uri is None
    assert runner.objects == {target_uri: payload}
    assert builder.calls == 1
    copy = next(command for command in runner.calls if command[:3] == ("gcloud", "storage", "cp"))
    assert copy == ("gcloud", "storage", "cp", "--no-clobber", str(wheel), target_uri, "--project=project-a")


def test_existing_content_addressed_wheel_is_reused_without_upload(tmp_path: Path) -> None:
    payload = b"existing native worker wheel"
    wheel = create_wheel(tmp_path, payload)
    digest = hashlib.sha256(payload).hexdigest()
    target_uri = f"gs://eyvan-runtime-artifacts/eyvan/runtime/wheels/{digest}/{wheel.name}"
    runner = FakeGcloudRunner(objects={target_uri: payload})
    bootstrap = GcpRuntimeBootstrap(FakeWheelBuilder(wheel), runner=runner)

    updated = bootstrap.stage_native_worker_wheel(profile())

    assert updated.native_worker_wheel_uri == target_uri
    assert not any(command[:3] == ("gcloud", "storage", "cp") for command in runner.calls)


def test_no_clobber_race_is_confirmed_with_a_read_before_returning_a_profile(tmp_path: Path) -> None:
    payload = b"raced native worker wheel"
    wheel = create_wheel(tmp_path, payload)
    digest = hashlib.sha256(payload).hexdigest()
    target_uri = f"gs://eyvan-runtime-artifacts/eyvan/runtime/wheels/{digest}/{wheel.name}"
    runner = FakeGcloudRunner(copy_races={target_uri})
    bootstrap = GcpRuntimeBootstrap(FakeWheelBuilder(wheel), runner=runner)

    updated = bootstrap.stage_native_worker_wheel(profile())

    assert updated.native_worker_wheel_uri == target_uri
    describes = [command for command in runner.calls if command[:4] == ("gcloud", "storage", "objects", "describe")]
    assert len(describes) == 2
    assert runner.objects[target_uri] == payload


@pytest.mark.parametrize(
    ("step", "command"),
    [
        (
            GcpRuntimeBootstrapStep.PROJECT_ACCESS,
            ("gcloud", "projects", "describe", "project-a", "--format=value(projectId)"),
        ),
        (
            GcpRuntimeBootstrapStep.STORAGE_ACCESS,
            (
                "gcloud",
                "storage",
                "buckets",
                "describe",
                "gs://eyvan-runtime-artifacts",
                "--project=project-a",
                "--format=value(name)",
            ),
        ),
        (
            GcpRuntimeBootstrapStep.TPU_ACCESS,
            (
                "gcloud",
                "compute",
                "tpus",
                "locations",
                "list",
                "--project=project-a",
                "--limit=1",
                "--format=value(locationId)",
            ),
        ),
    ],
)
def test_preflight_reports_the_failing_access_boundary(
    tmp_path: Path,
    step: GcpRuntimeBootstrapStep,
    command: tuple[str, ...],
) -> None:
    runner = FakeGcloudRunner(failures={command: (1, b"", b"PERMISSION_DENIED: test access")})
    bootstrap = GcpRuntimeBootstrap(FakeWheelBuilder(create_wheel(tmp_path)), runner=runner)

    with pytest.raises(GcpRuntimePreflightError, match="PERMISSION_DENIED") as raised:
        bootstrap.preflight(profile())

    assert raised.value.step is step


def test_wheel_builder_failure_is_typed_and_does_not_issue_gcloud_commands(tmp_path: Path) -> None:
    invalid_wheel = tmp_path / "eyvan.tar.gz"
    invalid_wheel.write_bytes(b"not a wheel")
    runner = FakeGcloudRunner()
    bootstrap = GcpRuntimeBootstrap(FakeWheelBuilder(invalid_wheel), runner=runner)

    with pytest.raises(GcpRuntimeWheelBuildError, match="PEP 427 wheel filename") as raised:
        bootstrap.stage_native_worker_wheel(profile())

    assert raised.value.step is GcpRuntimeBootstrapStep.WHEEL_BUILD
    assert runner.calls == []


def test_wheel_upload_permission_error_is_typed_and_keeps_profile_immutable(tmp_path: Path) -> None:
    payload = b"native worker wheel"
    wheel = create_wheel(tmp_path, payload)
    digest = hashlib.sha256(payload).hexdigest()
    target_uri = f"gs://eyvan-runtime-artifacts/eyvan/runtime/wheels/{digest}/{wheel.name}"
    copy = ("gcloud", "storage", "cp", "--no-clobber", str(wheel), target_uri, "--project=project-a")
    runner = FakeGcloudRunner(failures={copy: (1, b"", b"PERMISSION_DENIED: cannot create object")})
    original = profile()
    bootstrap = GcpRuntimeBootstrap(FakeWheelBuilder(wheel), runner=runner)

    with pytest.raises(GcpRuntimeWheelStageError, match="PERMISSION_DENIED") as raised:
        bootstrap.stage_native_worker_wheel(original)

    assert raised.value.step is GcpRuntimeBootstrapStep.WHEEL_STAGE
    assert original.native_worker_wheel_uri is None


def test_destinations_reject_non_sha256_digest() -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        GcpRuntimeBootstrap.destinations_for(profile(), "not-a-digest", "eyvan-0.1.0-py3-none-any.whl")


def test_destinations_preserve_a_valid_wheel_filename_under_its_content_hash() -> None:
    wheel_sha256 = "a" * 64
    wheel_filename = "eyvan-1.2.3-py3-none-any.whl"

    destinations = GcpRuntimeBootstrap.destinations_for(profile(), wheel_sha256, wheel_filename)

    assert destinations.native_worker_wheel_uri == (
        f"gs://eyvan-runtime-artifacts/eyvan/runtime/wheels/{wheel_sha256}/{wheel_filename}"
    )


@pytest.mark.parametrize(
    "wheel_filename",
    ("eyvan.whl", "eyvan-1.0.tar.gz", "../eyvan-1.0-py3-none-any.whl", "eyvan-1.0-py3-none-any.whl/"),
)
def test_destinations_reject_invalid_or_unsafe_wheel_filenames(wheel_filename: str) -> None:
    with pytest.raises(ValueError, match="PEP 427 wheel filename"):
        GcpRuntimeBootstrap.destinations_for(profile(), "a" * 64, wheel_filename)


_BUCKET_DESCRIBE = (
    "gcloud",
    "storage",
    "buckets",
    "describe",
    "gs://eyvan-runtime-artifacts",
    "--project=project-a",
    "--format=value(name)",
)
_BUCKET_CREATE_PREFIX = ("gcloud", "storage", "buckets", "create", "gs://eyvan-runtime-artifacts")


def test_preflight_tolerates_a_missing_artifact_bucket(tmp_path: Path) -> None:
    """ "The bucket does not exist yet" is a state, not an access failure."""

    runner = FakeGcloudRunner(failures={_BUCKET_DESCRIBE: (1, b"", b"404 bucket not found")})
    bootstrap = GcpRuntimeBootstrap(FakeWheelBuilder(create_wheel(tmp_path)), runner=runner)

    checked = bootstrap.preflight(profile())

    assert GcpRuntimeBootstrapStep.STORAGE_ACCESS in checked.checks


def test_preflight_still_fails_on_a_real_bucket_access_error(tmp_path: Path) -> None:
    runner = FakeGcloudRunner(failures={_BUCKET_DESCRIBE: (1, b"", b"PERMISSION_DENIED")})
    bootstrap = GcpRuntimeBootstrap(FakeWheelBuilder(create_wheel(tmp_path)), runner=runner)

    with pytest.raises(GcpRuntimePreflightError, match="PERMISSION_DENIED") as raised:
        bootstrap.preflight(profile())

    assert raised.value.step is GcpRuntimeBootstrapStep.STORAGE_ACCESS


def test_ensure_artifact_bucket_is_a_noop_when_the_bucket_exists(tmp_path: Path) -> None:
    runner = FakeGcloudRunner()
    bootstrap = GcpRuntimeBootstrap(FakeWheelBuilder(create_wheel(tmp_path)), runner=runner)

    assert bootstrap.ensure_artifact_bucket(profile()) == "gs://eyvan-runtime-artifacts"
    assert not [call for call in runner.calls if call[: len(_BUCKET_CREATE_PREFIX)] == _BUCKET_CREATE_PREFIX]


def test_ensure_artifact_bucket_creates_an_absent_bucket_with_uniform_access(tmp_path: Path) -> None:
    """First-time setup never depends on a hand-created bucket."""

    runner = FakeGcloudRunner(failures={_BUCKET_DESCRIBE: (1, b"", b"404 bucket not found")})
    bootstrap = GcpRuntimeBootstrap(FakeWheelBuilder(create_wheel(tmp_path)), runner=runner)

    assert bootstrap.ensure_artifact_bucket(profile()) == "gs://eyvan-runtime-artifacts"
    creates = [call for call in runner.calls if call[: len(_BUCKET_CREATE_PREFIX)] == _BUCKET_CREATE_PREFIX]
    assert creates == [
        (
            "gcloud",
            "storage",
            "buckets",
            "create",
            "gs://eyvan-runtime-artifacts",
            "--project=project-a",
            "--location=us-central1",
            "--uniform-bucket-level-access",
        )
    ]


def test_ensure_artifact_bucket_confirms_a_create_race(tmp_path: Path) -> None:
    """A concurrent initializer's win is as good as ours once it describes."""

    runner = FakeGcloudRunner(failures={_BUCKET_DESCRIBE: (1, b"", b"404 bucket not found")})
    bootstrap = GcpRuntimeBootstrap(FakeWheelBuilder(create_wheel(tmp_path)), runner=runner)

    def fail_create_once(args, *, check, capture_output, timeout):
        command = tuple(args)
        if command[: len(_BUCKET_CREATE_PREFIX)] == _BUCKET_CREATE_PREFIX:
            # The raced winner answers the confirming describe afterwards.
            runner.failures.pop(_BUCKET_DESCRIBE, None)
            return subprocess.CompletedProcess(command, 1, stdout=b"", stderr=b"already exists")
        return runner(args, check=check, capture_output=capture_output, timeout=timeout)

    bootstrap = GcpRuntimeBootstrap(FakeWheelBuilder(create_wheel(tmp_path)), runner=fail_create_once)

    assert bootstrap.ensure_artifact_bucket(profile()) == "gs://eyvan-runtime-artifacts"


def test_ensure_artifact_bucket_raises_typed_error_when_creation_is_refused(tmp_path: Path) -> None:
    runner = FakeGcloudRunner(
        failures={
            _BUCKET_DESCRIBE: (1, b"", b"404 bucket not found"),
        }
    )

    def refuse_create(args, *, check, capture_output, timeout):
        command = tuple(args)
        if command[: len(_BUCKET_CREATE_PREFIX)] == _BUCKET_CREATE_PREFIX:
            return subprocess.CompletedProcess(
                command, 1, stdout=b"", stderr=b"PERMISSION_DENIED: storage.buckets.create"
            )
        return runner(args, check=check, capture_output=capture_output, timeout=timeout)

    bootstrap = GcpRuntimeBootstrap(FakeWheelBuilder(create_wheel(tmp_path)), runner=refuse_create)

    with pytest.raises(GcpRuntimeBucketEnsureError, match="PERMISSION_DENIED") as raised:
        bootstrap.ensure_artifact_bucket(profile())

    assert raised.value.step is GcpRuntimeBootstrapStep.STORAGE_ENSURE
