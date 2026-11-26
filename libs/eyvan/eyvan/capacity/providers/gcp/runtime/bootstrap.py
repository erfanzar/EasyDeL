# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Bounded GCP preflight and immutable native-worker wheel staging."""

from __future__ import annotations

import hashlib
import re
import stat
import subprocess
import typing as tp
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urlsplit

from eyvan._internal.compat import StrEnum
from eyvan._internal.subprocess_env import provider_cli_env
from eyvan.capacity.providers.gcp.runtime.profile import (
    GcpNativeRuntimeProfile,
    GcpNativeRuntimeProfileError,
    validate_native_worker_wheel_filename,
)

_WHEEL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DEFAULT_COMMAND_TIMEOUT_S = 60.0
_DEFAULT_MAX_WHEEL_BYTES = 512 * 1024 * 1024
#: Location used when Eyvan creates the artifact bucket itself. Artifacts are
#: wheels and pickled callables -- small, written once, read at dispatch -- so a
#: predictable fixed region beats a clever one. Operators who care name their
#: own bucket with --artifact-root instead.
DEFAULT_ARTIFACT_BUCKET_LOCATION = "us-central1"


class GcpRuntimeBootstrapStep(StrEnum):
    """One explicit boundary in native GCP runtime preparation."""

    PROJECT_ACCESS = "project_access"
    STORAGE_ACCESS = "storage_access"
    STORAGE_ENSURE = "storage_ensure"
    TPU_ACCESS = "tpu_access"
    WHEEL_BUILD = "wheel_build"
    WHEEL_STAGE = "wheel_stage"


class GcpRuntimeBootstrapError(RuntimeError):
    """Base error with a stable preparation step for CLI and API diagnostics."""

    def __init__(self, step: GcpRuntimeBootstrapStep, detail: str):
        self.step = step
        self.detail = detail
        super().__init__(f"GCP native runtime {step.value.replace('_', ' ')} failed: {detail}")


class GcpRuntimePreflightError(GcpRuntimeBootstrapError):
    """A non-mutating GCP project, bucket, or TPU access probe failed."""


class GcpRuntimeBucketEnsureError(GcpRuntimeBootstrapError):
    """The artifact bucket did not exist and could not be created."""


class GcpRuntimeWheelBuildError(GcpRuntimeBootstrapError):
    """The injected local wheel builder did not yield a safe wheel file."""


class GcpRuntimeWheelStageError(GcpRuntimeBootstrapError):
    """The content-addressed native-worker wheel could not be staged in GCS."""


class GcloudRunner(tp.Protocol):
    """Typed subprocess boundary for every GCP command issued by the bootstrapper."""

    def __call__(
        self,
        args: tp.Sequence[str],
        *,
        check: bool,
        capture_output: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[bytes]: ...


class NativeWorkerWheelBuilder(tp.Protocol):
    """Build or locate one local Eyvan native-worker wheel."""

    def build(self) -> Path: ...


def _run_gcloud(
    args: tp.Sequence[str],
    *,
    check: bool,
    capture_output: bool,
    timeout: float,
) -> subprocess.CompletedProcess[bytes]:
    """Run one bounded GCP CLI command without inheriting terminal output."""

    return subprocess.run(args, check=check, capture_output=capture_output, timeout=timeout, env=provider_cli_env())


@dataclass(frozen=True, slots=True)
class GcpRuntimeDestinations:
    """Stable GCS destinations derived from a native GCP runtime profile."""

    artifact_bucket_uri: str
    fleet_state_uri: str
    native_worker_wheel_uri: str


@dataclass(frozen=True, slots=True)
class GcpRuntimePreflight:
    """Successful non-mutating access checks for one native GCP runtime profile."""

    project_id: str
    artifact_bucket_uri: str
    fleet_state_uri: str
    checks: tuple[GcpRuntimeBootstrapStep, ...]


class GcpRuntimeBootstrap:
    """Prepare a native GCP profile without provisioning any TPU capacity.

    The class is intentionally narrow: it verifies that the configured account
    can read the project, artifact bucket, and Cloud TPU locations, creates the
    artifact bucket when it is absent (the one mutation it owns), then stages a
    locally built wheel under its SHA-256. It never creates a TPU, modifies IAM,
    changes quotas, or persists the returned profile.
    """

    def __init__(
        self,
        wheel_builder: NativeWorkerWheelBuilder,
        *,
        runner: GcloudRunner = _run_gcloud,
        command_timeout_s: float = _DEFAULT_COMMAND_TIMEOUT_S,
        max_wheel_bytes: int = _DEFAULT_MAX_WHEEL_BYTES,
    ):
        if isinstance(command_timeout_s, bool) or not isinstance(command_timeout_s, (int, float)):
            raise ValueError("command_timeout_s must be a positive number")
        if command_timeout_s <= 0:
            raise ValueError("command_timeout_s must be a positive number")
        if isinstance(max_wheel_bytes, bool) or not isinstance(max_wheel_bytes, int) or max_wheel_bytes <= 0:
            raise ValueError("max_wheel_bytes must be a positive integer")
        self._wheel_builder = wheel_builder
        self._runner = runner
        self.command_timeout_s = float(command_timeout_s)
        self.max_wheel_bytes = max_wheel_bytes

    def preflight(self, profile: GcpNativeRuntimeProfile) -> GcpRuntimePreflight:
        """Verify read access to the configured GCP project, bucket, and TPU API.

        Every command in this method is read-only. A successful preflight does
        not imply that Spot capacity is currently available or that the account
        may create TPU resources; those are provider-side admission checks.

        A missing artifact bucket passes here and is created by
        :meth:`ensure_artifact_bucket`: "the bucket does not exist yet" is a
        state, not an access failure, and failing preflight on it made every
        first run require a hand-created bucket.
        """

        root_uri, artifact_bucket_uri = self._artifact_root(profile)
        project = profile.project_id
        self._run_read_only(
            GcpRuntimeBootstrapStep.PROJECT_ACCESS,
            ("gcloud", "projects", "describe", project, "--format=value(projectId)"),
        )
        self._run_bucket_access_probe(artifact_bucket_uri, project=project)
        self._run_read_only(
            GcpRuntimeBootstrapStep.TPU_ACCESS,
            (
                "gcloud",
                "compute",
                "tpus",
                "locations",
                "list",
                f"--project={project}",
                "--limit=1",
                "--format=value(locationId)",
            ),
        )
        return GcpRuntimePreflight(
            project_id=project,
            artifact_bucket_uri=artifact_bucket_uri,
            fleet_state_uri=f"{root_uri}/control-plane/fleet-state.json",
            checks=(
                GcpRuntimeBootstrapStep.PROJECT_ACCESS,
                GcpRuntimeBootstrapStep.STORAGE_ACCESS,
                GcpRuntimeBootstrapStep.TPU_ACCESS,
            ),
        )

    def _run_bucket_access_probe(self, artifact_bucket_uri: str, *, project: str) -> None:
        """Pass when the bucket is readable or merely absent; fail on real access errors."""

        result = self._invoke(
            GcpRuntimeBootstrapStep.STORAGE_ACCESS,
            (
                "gcloud",
                "storage",
                "buckets",
                "describe",
                artifact_bucket_uri,
                f"--project={project}",
                "--format=value(name)",
            ),
            GcpRuntimePreflightError,
        )
        if result.returncode == 0 or self._is_missing(result):
            return
        raise GcpRuntimePreflightError(
            GcpRuntimeBootstrapStep.STORAGE_ACCESS,
            self._failed_command_detail("gcloud access check failed", result),
        )

    def ensure_artifact_bucket(
        self,
        profile: GcpNativeRuntimeProfile,
        *,
        location: str = DEFAULT_ARTIFACT_BUCKET_LOCATION,
    ) -> str:
        """Create the profile's artifact bucket when it is absent, and return its URI.

        This is the one mutating step in runtime preparation: Eyvan owns the
        bucket it stages wheels and callables into, the way the autoscalers it
        replaces own theirs, so an operator never hand-creates one. Creation is
        idempotent -- a describe-first check plus a create that treats "already
        exists" as success when the describe says the bucket is ours to read.
        """

        _, artifact_bucket_uri = self._artifact_root(profile)
        project = profile.project_id
        describe_args = (
            "gcloud",
            "storage",
            "buckets",
            "describe",
            artifact_bucket_uri,
            f"--project={project}",
            "--format=value(name)",
        )
        existing = self._invoke(
            GcpRuntimeBootstrapStep.STORAGE_ENSURE,
            describe_args,
            GcpRuntimeBucketEnsureError,
            f"could not inspect artifact bucket {artifact_bucket_uri}",
        )
        if existing.returncode == 0:
            return artifact_bucket_uri
        if not self._is_missing(existing):
            raise GcpRuntimeBucketEnsureError(
                GcpRuntimeBootstrapStep.STORAGE_ENSURE,
                self._failed_command_detail(f"could not inspect artifact bucket {artifact_bucket_uri}", existing),
            )
        created = self._invoke(
            GcpRuntimeBootstrapStep.STORAGE_ENSURE,
            (
                "gcloud",
                "storage",
                "buckets",
                "create",
                artifact_bucket_uri,
                f"--project={project}",
                f"--location={location}",
                "--uniform-bucket-level-access",
            ),
            GcpRuntimeBucketEnsureError,
            f"could not create artifact bucket {artifact_bucket_uri}",
        )
        if created.returncode == 0:
            return artifact_bucket_uri
        # A concurrent initializer can win the create race; the bucket is then
        # exactly as good as if this call had created it.
        confirmed = self._invoke(
            GcpRuntimeBootstrapStep.STORAGE_ENSURE,
            describe_args,
            GcpRuntimeBucketEnsureError,
            f"could not confirm artifact bucket {artifact_bucket_uri} after create failed",
        )
        if confirmed.returncode == 0:
            return artifact_bucket_uri
        raise GcpRuntimeBucketEnsureError(
            GcpRuntimeBootstrapStep.STORAGE_ENSURE,
            self._failed_command_detail(f"could not create artifact bucket {artifact_bucket_uri}", created),
        )

    def bootstrap(self, profile: GcpNativeRuntimeProfile) -> GcpNativeRuntimeProfile:
        """Run preflight, ensure the artifact bucket, then pin the profile to a staged wheel."""

        self.preflight(profile)
        self.ensure_artifact_bucket(profile)
        return self.stage_native_worker_wheel(profile)

    def stage_native_worker_wheel(self, profile: GcpNativeRuntimeProfile) -> GcpNativeRuntimeProfile:
        """Stage one local wheel under its SHA-256 without overwriting an object."""

        wheel_path = self._build_wheel()
        wheel_sha256 = self._wheel_sha256(wheel_path)
        destinations = self.destinations_for(profile, wheel_sha256, wheel_path.name)
        target_uri = destinations.native_worker_wheel_uri
        describe_args = (
            "gcloud",
            "storage",
            "objects",
            "describe",
            target_uri,
            f"--project={profile.project_id}",
            "--format=value(name)",
        )
        existing = self._run_stage_command(
            describe_args,
            detail_prefix=f"could not inspect staged native-worker wheel {target_uri}",
        )
        if existing.returncode == 0:
            return self._with_staged_wheel(profile, target_uri)
        if not self._is_missing(existing):
            raise GcpRuntimeWheelStageError(
                GcpRuntimeBootstrapStep.WHEEL_STAGE,
                self._failed_command_detail(f"could not inspect staged native-worker wheel {target_uri}", existing),
            )

        copied = self._run_stage_command(
            (
                "gcloud",
                "storage",
                "cp",
                "--no-clobber",
                str(wheel_path),
                target_uri,
                f"--project={profile.project_id}",
            ),
            detail_prefix=f"could not stage native-worker wheel at {target_uri}",
        )
        if copied.returncode == 0:
            return self._with_staged_wheel(profile, target_uri)
        if self._is_existing_collision(copied):
            confirmed = self._run_stage_command(
                describe_args,
                detail_prefix=f"could not confirm raced native-worker wheel at {target_uri}",
            )
            if confirmed.returncode == 0:
                return self._with_staged_wheel(profile, target_uri)
        raise GcpRuntimeWheelStageError(
            GcpRuntimeBootstrapStep.WHEEL_STAGE,
            self._failed_command_detail(f"could not stage native-worker wheel at {target_uri}", copied),
        )

    @staticmethod
    def destinations_for(
        profile: GcpNativeRuntimeProfile,
        wheel_sha256: str,
        wheel_filename: str,
    ) -> GcpRuntimeDestinations:
        """Derive immutable runtime paths without writing a fleet-state object."""

        if not _WHEEL_SHA256.fullmatch(wheel_sha256):
            raise ValueError("wheel_sha256 must be a lowercase SHA-256 digest")
        try:
            wheel_filename = validate_native_worker_wheel_filename(wheel_filename)
        except GcpNativeRuntimeProfileError as exc:
            raise ValueError("wheel_filename must be a concrete PEP 427 wheel filename") from exc
        root_uri, artifact_bucket_uri = GcpRuntimeBootstrap._artifact_root(profile)
        return GcpRuntimeDestinations(
            artifact_bucket_uri=artifact_bucket_uri,
            fleet_state_uri=f"{root_uri}/control-plane/fleet-state.json",
            native_worker_wheel_uri=f"{root_uri}/runtime/wheels/{wheel_sha256}/{wheel_filename}",
        )

    @staticmethod
    def _artifact_root(profile: GcpNativeRuntimeProfile) -> tuple[str, str]:
        """Return the normalized artifact root and its bucket URI from a validated profile."""

        parsed = urlsplit(profile.artifact_root_uri)
        if parsed.scheme != "gs" or not parsed.netloc:
            raise ValueError("native GCP runtime profile has an invalid GCS artifact root")
        return profile.artifact_root_uri.rstrip("/"), f"gs://{parsed.netloc}"

    @staticmethod
    def _with_staged_wheel(profile: GcpNativeRuntimeProfile, target_uri: str) -> GcpNativeRuntimeProfile:
        """Return the immutable profile copy that pins native workers to one wheel object."""

        return replace(profile, native_worker_wheel_uri=target_uri)

    def _run_read_only(self, step: GcpRuntimeBootstrapStep, args: tuple[str, ...]) -> None:
        """Run one preflight probe and translate CLI failures into typed diagnostics."""

        result = self._invoke(step, args, GcpRuntimePreflightError)
        if result.returncode != 0:
            raise GcpRuntimePreflightError(step, self._failed_command_detail("gcloud access check failed", result))

    def _run_stage_command(self, args: tuple[str, ...], *, detail_prefix: str) -> subprocess.CompletedProcess[bytes]:
        """Run one bounded staging command with a wheel-stage error on process failure."""

        return self._invoke(GcpRuntimeBootstrapStep.WHEEL_STAGE, args, GcpRuntimeWheelStageError, detail_prefix)

    def _invoke(
        self,
        step: GcpRuntimeBootstrapStep,
        args: tuple[str, ...],
        error_type: type[GcpRuntimeBootstrapError],
        detail_prefix: str = "gcloud command failed",
    ) -> subprocess.CompletedProcess[bytes]:
        """Invoke the injected command boundary and preserve only actionable failures."""

        try:
            return self._runner(args, check=False, capture_output=True, timeout=self.command_timeout_s)
        except FileNotFoundError as exc:
            raise error_type(step, f"{detail_prefix}: gcloud is not installed or is not on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise error_type(step, f"{detail_prefix}: gcloud exceeded {self.command_timeout_s:g}s") from exc
        except OSError as exc:
            raise error_type(step, f"{detail_prefix}: could not start gcloud: {exc}") from exc

    def _build_wheel(self) -> Path:
        """Obtain one safe local wheel from the injected builder."""

        try:
            candidate = self._wheel_builder.build()
        except (OSError, subprocess.SubprocessError) as exc:
            raise GcpRuntimeWheelBuildError(
                GcpRuntimeBootstrapStep.WHEEL_BUILD,
                f"native-worker wheel builder failed: {exc}",
            ) from exc
        if not isinstance(candidate, Path):
            raise GcpRuntimeWheelBuildError(
                GcpRuntimeBootstrapStep.WHEEL_BUILD,
                "native-worker wheel builder must return pathlib.Path",
            )
        try:
            metadata = candidate.lstat()
        except OSError as exc:
            raise GcpRuntimeWheelBuildError(
                GcpRuntimeBootstrapStep.WHEEL_BUILD,
                f"native-worker wheel {candidate} is not readable: {exc}",
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise GcpRuntimeWheelBuildError(
                GcpRuntimeBootstrapStep.WHEEL_BUILD,
                f"native-worker wheel {candidate} must be a regular file",
            )
        try:
            validate_native_worker_wheel_filename(candidate.name)
        except GcpNativeRuntimeProfileError as exc:
            raise GcpRuntimeWheelBuildError(
                GcpRuntimeBootstrapStep.WHEEL_BUILD,
                f"native-worker wheel {candidate} must use a concrete PEP 427 wheel filename",
            ) from exc
        if metadata.st_size > self.max_wheel_bytes:
            raise GcpRuntimeWheelBuildError(
                GcpRuntimeBootstrapStep.WHEEL_BUILD,
                f"native-worker wheel {candidate} exceeds the {self.max_wheel_bytes}-byte staging limit",
            )
        return candidate

    def _wheel_sha256(self, wheel_path: Path) -> str:
        """Hash a bounded local wheel without holding it all in memory."""

        digest = hashlib.sha256()
        total = 0
        try:
            with wheel_path.open("rb") as wheel:
                while chunk := wheel.read(1024 * 1024):
                    total += len(chunk)
                    if total > self.max_wheel_bytes:
                        raise GcpRuntimeWheelBuildError(
                            GcpRuntimeBootstrapStep.WHEEL_BUILD,
                            f"native-worker wheel {wheel_path} exceeds the {self.max_wheel_bytes}-byte staging limit",
                        )
                    digest.update(chunk)
        except GcpRuntimeWheelBuildError:
            raise
        except OSError as exc:
            raise GcpRuntimeWheelBuildError(
                GcpRuntimeBootstrapStep.WHEEL_BUILD,
                f"could not hash native-worker wheel {wheel_path}: {exc}",
            ) from exc
        return digest.hexdigest()

    @staticmethod
    def _is_missing(result: subprocess.CompletedProcess[bytes]) -> bool:
        """Recognize the GCS CLI's object-not-found outcomes without masking access errors."""

        detail = GcpRuntimeBootstrap._command_output(result).lower()
        return result.returncode != 0 and ("404" in detail or "not found" in detail or "no urls matched" in detail)

    @staticmethod
    def _is_existing_collision(result: subprocess.CompletedProcess[bytes]) -> bool:
        """Recognize a no-clobber race that can safely be confirmed by a read."""

        detail = GcpRuntimeBootstrap._command_output(result).lower()
        return result.returncode != 0 and (
            "already exists" in detail or "precondition" in detail or "conditionnotmet" in detail
        )

    @staticmethod
    def _failed_command_detail(prefix: str, result: subprocess.CompletedProcess[bytes]) -> str:
        """Build bounded CLI diagnostics without leaking unbounded process output."""

        output = GcpRuntimeBootstrap._command_output(result)
        if output:
            return f"{prefix}: {output[:800]}"
        return f"{prefix} (gcloud exited with status {result.returncode})"

    @staticmethod
    def _command_output(result: subprocess.CompletedProcess[bytes]) -> str:
        """Decode captured gcloud output consistently for diagnostics and classification."""

        return b"\n".join((result.stderr or b"", result.stdout or b"")).decode("utf-8", "replace").strip()


__all__ = (
    "DEFAULT_ARTIFACT_BUCKET_LOCATION",
    "GcloudRunner",
    "GcpRuntimeBootstrap",
    "GcpRuntimeBootstrapError",
    "GcpRuntimeBootstrapStep",
    "GcpRuntimeBucketEnsureError",
    "GcpRuntimeDestinations",
    "GcpRuntimePreflight",
    "GcpRuntimePreflightError",
    "GcpRuntimeWheelBuildError",
    "GcpRuntimeWheelStageError",
    "NativeWorkerWheelBuilder",
)
