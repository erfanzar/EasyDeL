# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""One-time local setup for Eyvan's native GCP runtime."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import typing as tp
import zipfile
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

import eyvan
from eyvan._internal.json_types import JsonObject
from eyvan._internal.subprocess_env import provider_cli_env
from eyvan.capacity.autoscale import AutoscaleConfigCache, validate_gcp_project_id
from eyvan.capacity.providers.gcp.runtime.bootstrap import (
    GcpRuntimeBootstrap,
    GcpRuntimeBootstrapError,
    GcpRuntimePreflight,
)
from eyvan.capacity.providers.gcp.runtime.profile import (
    DEFAULT_GCP_NATIVE_RUNTIME_PROFILE_PATH,
    GcpNativeRuntimeProfile,
    GcpNativeRuntimeProfileError,
    GcpNativeRuntimeProfileStore,
    default_artifact_root_uri,
)
from eyvan.capacity.providers.gcp.runtime.supervisor import LocalAgentState, LocalAgentSupervisor
from eyvan.sdk.client.endpoint import DEFAULT_CONTROL_PLANE_URL, ControlPlaneEndpoint
from eyvan.sdk.commands.autoscale import autoscale_console, gcloud_default_project, is_interactive
from eyvan.sdk.commands.deployment import stale_components
from eyvan.sdk.commands.state import CliSession
from eyvan.sdk.terminal import print_block
from eyvan.server.local.daemon import LocalDaemonError, LocalDaemonResult, LocalDaemonSupervisor

_DEFAULT_BUILD_TIMEOUT_S = 300.0
_VALIDATION_AGENT_SECRET = "a" * 32
_VALIDATION_NATIVE_WORKER_SECRET = "b" * 32
_RECORD_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
_DIST_INFO_SUFFIX = ".dist-info"
#: Files an installer writes into ``.dist-info`` that the wheel never carried.
#: PEP 376 defines the first five and PEP 610 ``direct_url.json``; uv adds two
#: of its own. Repacking them would make the staged wheel depend on which tool
#: installed Eyvan rather than on Eyvan.
_INSTALLER_OWNED_DIST_INFO_FILES = frozenset(
    {
        "INSTALLER",
        "RECORD",
        "RECORD.jws",
        "RECORD.p7s",
        "REQUESTED",
        "direct_url.json",
        "uv_build.json",
        "uv_cache.json",
    }
)
#: A wheel is a zip and a zip stores modification times, so repacking the same
#: installation twice would otherwise produce two different SHA-256 digests --
#: and the staged object is content-addressed, so every ``runtime deploy`` would
#: upload a fresh copy of identical bytes.
_WHEEL_ENTRY_DATE_TIME = (1980, 1, 1, 0, 0, 0)
_WHEEL_ENTRY_MODE = 0o644


class NativeWorkerWheelBuildError(OSError):
    """The local Eyvan wheel build required for native workers did not complete."""


class GcpRuntimeInitializationError(RuntimeError):
    """The bootstrapper did not return the complete profile contract required by the CLI."""


class GcpRuntimeRestartError(RuntimeError):
    """A private native GCP runtime cannot be safely restarted."""


class NativeWorkerWheelSource(tp.Protocol):
    """One local Eyvan wheel plus the temporary directory that holds it."""

    def build(self) -> Path:
        """Produce the wheel that native workers will install."""

        ...

    def close(self) -> None:
        """Remove whatever the build left on local disk."""

        ...


class UvBuildRunner(tp.Protocol):
    """Typed boundary for the local ``uv build`` invocation."""

    def __call__(
        self,
        args: tp.Sequence[str],
        *,
        cwd: Path,
        check: bool,
        capture_output: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[bytes]:
        """Run one local build command."""

        ...


class RuntimeProfileStore(tp.Protocol):
    """Private profile persistence required by native runtime initialization."""

    def load_or_create(
        self,
        *,
        project_id: str,
        artifact_root_uri: str,
        native_worker_wheel_uri: str | None = None,
    ) -> GcpNativeRuntimeProfile:
        """Load the stable profile or create its owner-only credentials once."""

        ...

    def save(self, profile: GcpNativeRuntimeProfile) -> Path:
        """Persist the staged-wheel profile atomically."""

        ...


class RuntimeBootstrapper(tp.Protocol):
    """Typed preparation contract used by the runtime setup service."""

    def preflight(self, profile: GcpNativeRuntimeProfile) -> GcpRuntimePreflight:
        """Check GCP project, storage, and TPU access without creating capacity."""

        ...

    def ensure_artifact_bucket(self, profile: GcpNativeRuntimeProfile) -> str:
        """Create the profile's artifact bucket when it is absent."""

        ...

    def stage_native_worker_wheel(self, profile: GcpNativeRuntimeProfile) -> GcpNativeRuntimeProfile:
        """Stage the content-addressed Eyvan native-worker wheel."""

        ...


def _run_uv_build(
    args: tp.Sequence[str],
    *,
    cwd: Path,
    check: bool,
    capture_output: bool,
    timeout: float,
) -> subprocess.CompletedProcess[bytes]:
    """Run a bounded local wheel build without streaming its output into the CLI."""

    return subprocess.run(
        args, cwd=cwd, check=check, capture_output=capture_output, timeout=timeout, env=provider_cli_env()
    )


class UvNativeWorkerWheelBuilder:
    """Build one temporary Eyvan wheel from the installed source checkout.

    The wheel only lives long enough for ``GcpRuntimeBootstrap`` to hash and
    stage it under an immutable GCS object name. It is never placed in the
    runtime profile or passed to a daemon command line.
    """

    def __init__(
        self,
        *,
        package_root: Path | None = None,
        runner: UvBuildRunner = _run_uv_build,
        timeout_s: float = _DEFAULT_BUILD_TIMEOUT_S,
    ):
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
            raise ValueError("timeout_s must be a positive number")
        resolved_root = package_root if package_root is not None else _default_package_root()
        if resolved_root is None:
            raise NativeWorkerWheelBuildError(
                "`uv build` needs a directory containing pyproject.toml and this Eyvan is not running from a source "
                "checkout; use InstalledDistributionWheelBuilder, which repacks the installed distribution instead"
            )
        self.package_root = resolved_root.resolve()
        self._runner = runner
        self.timeout_s = float(timeout_s)
        self._output_directory: Path | None = None

    def build(self) -> Path:
        """Build and return the single wheel produced in a private temporary directory."""

        self.close()
        if not self.package_root.is_dir():
            raise NativeWorkerWheelBuildError(f"Eyvan package root {self.package_root} does not exist")
        output_directory = Path(tempfile.mkdtemp(prefix="eyvan-native-worker-"))
        self._output_directory = output_directory
        command = ("uv", "build", "--wheel", "--out-dir", str(output_directory), str(self.package_root))
        try:
            completed = self._runner(
                command,
                cwd=self.package_root,
                check=False,
                capture_output=True,
                timeout=self.timeout_s,
            )
        except FileNotFoundError as exc:
            raise NativeWorkerWheelBuildError("uv is not installed or is not on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise NativeWorkerWheelBuildError(f"uv build exceeded {self.timeout_s:g}s") from exc
        except OSError as exc:
            raise NativeWorkerWheelBuildError(f"could not start uv build: {exc}") from exc
        if completed.returncode != 0:
            detail = _process_output(completed)
            message = "uv build failed"
            if detail:
                message = f"{message}: {detail[:800]}"
            raise NativeWorkerWheelBuildError(message)
        wheels = tuple(sorted(output_directory.glob("*.whl")))
        if len(wheels) != 1:
            raise NativeWorkerWheelBuildError(
                f"uv build must produce exactly one Eyvan wheel, found {len(wheels)} in {output_directory}"
            )
        return wheels[0]

    def close(self) -> None:
        """Remove the temporary local wheel directory once staging is complete."""

        output_directory = self._output_directory
        self._output_directory = None
        if output_directory is not None:
            shutil.rmtree(output_directory, ignore_errors=True)


def _default_package_root() -> Path | None:
    """Return the directory holding Eyvan's ``pyproject.toml``, or ``None``.

    Anchored on the package rather than counted in parent hops: a hop encodes
    how deep this module happens to sit, so regrouping the package silently
    pointed ``uv build`` at a directory with no project file in it.

    ``None`` means this Eyvan was installed rather than checked out. Returning
    the project-less directory anyway -- which is what this used to do -- made
    ``uv build`` fail with "does not appear to be a Python project", so the
    published wheel and every image built from it could not run
    ``eyvan runtime init`` at all. :func:`native_worker_wheel_builder` reads the
    ``None`` and repacks the installed distribution instead.
    """

    root = Path(eyvan.__file__).resolve().parent.parent
    if (root / "pyproject.toml").is_file():
        return root
    return None


def installed_eyvan_distribution() -> metadata.Distribution:
    """Return the installed distribution that provides the running ``eyvan``.

    The identity check is the point. Staging some *other* installed Eyvan would
    put code on every worker in the fleet that nobody here is running, and the
    workload would succeed on it -- the drift ``runtime deploy`` exists to
    remove, reintroduced one layer down.
    """

    try:
        distribution = metadata.distribution("eyvan")
    except metadata.PackageNotFoundError as exc:
        raise NativeWorkerWheelBuildError(
            "Eyvan is importable but is not installed as a distribution, so there is nothing to repack into a "
            "native-worker wheel; install Eyvan from a wheel or run this command from a source checkout"
        ) from exc
    installed = Path(os.fspath(distribution.locate_file("eyvan/__init__.py"))).resolve()
    running = Path(eyvan.__file__).resolve()
    if installed != running:
        raise NativeWorkerWheelBuildError(
            f"the installed Eyvan distribution provides {installed}, but this process imported {running}; "
            "staging the installed one would deploy code this command is not running"
        )
    return distribution


class InstalledDistributionWheelBuilder:
    """Repack the installed Eyvan distribution into one wheel, with no source tree.

    ``uv build`` needs a project directory and an installed wheel has none, so
    the distribution this project publishes could not provision a fleet with
    itself. It needs no project directory: the native-worker wheel *is* this
    distribution, and every byte of it is already on disk, named and hashed by
    the installer in ``RECORD``. Repacking those bytes reproduces the wheel that
    was installed rather than building a second one that might differ from it.

    The result is byte-stable for a given installation, because the staged GCS
    object is addressed by its SHA-256 and a wheel whose hash moved on every
    invocation would re-upload identical bytes forever.
    """

    def __init__(self, distribution: metadata.Distribution):
        self._distribution = distribution
        self._output_directory: Path | None = None

    def build(self) -> Path:
        """Return one repacked wheel in a private temporary directory."""

        self.close()
        install_root = Path(os.fspath(self._distribution.locate_file(""))).resolve()
        rows = self._record_rows()
        dist_info = self._dist_info_directory(rows)
        self._refuse_files_installed_outside_the_root(rows)
        recorded = self._recorded_files(rows, dist_info=dist_info, install_root=install_root)
        if not any(name.startswith("eyvan/") for name, _path, _digest in recorded):
            raise NativeWorkerWheelBuildError(
                f"the Eyvan distribution installed at {install_root} records no packaged module files -- an editable "
                "or otherwise source-linked install cannot be repacked; run this command from its source checkout"
            )
        wheel_name = f"{dist_info[: -len(_DIST_INFO_SUFFIX)]}-{self._wheel_tag(install_root / dist_info)}.whl"
        output_directory = Path(tempfile.mkdtemp(prefix="eyvan-native-worker-"))
        self._output_directory = output_directory
        wheel_path = output_directory / wheel_name
        self._write_wheel(wheel_path, recorded, dist_info=dist_info)
        return wheel_path

    def close(self) -> None:
        """Remove the temporary local wheel directory once staging is complete."""

        output_directory = self._output_directory
        self._output_directory = None
        if output_directory is not None:
            shutil.rmtree(output_directory, ignore_errors=True)

    def _record_rows(self) -> tuple[tuple[str, ...], ...]:
        """Parse the installer-written ``RECORD`` inventory of this distribution."""

        record = self._distribution.read_text("RECORD")
        if not record:
            raise NativeWorkerWheelBuildError(
                "the installed Eyvan distribution has no RECORD, so its installed files cannot be listed; "
                "reinstall Eyvan from a wheel or run this command from a source checkout"
            )
        return tuple(tuple(row) for row in csv.reader(record.splitlines()) if row)

    @staticmethod
    def _dist_info_directory(rows: tp.Sequence[tp.Sequence[str]]) -> str:
        """Read the ``.dist-info`` directory name out of RECORD's own entry."""

        for row in rows:
            name = row[0]
            if name.endswith(f"{_DIST_INFO_SUFFIX}/RECORD"):
                return name[: -len("/RECORD")]
        raise NativeWorkerWheelBuildError(
            "the installed Eyvan distribution's RECORD does not name its own .dist-info directory"
        )

    def _refuse_files_installed_outside_the_root(self, rows: tp.Sequence[tp.Sequence[str]]) -> None:
        """Refuse an installation that put files anywhere but ``site-packages``.

        Console launchers are the one legitimate case, and dropping them is
        correct: an installer regenerates them from ``entry_points.txt``.
        Anything else outside the install root came from a wheel ``.data``
        directory, and this repack cannot tell which scheme it belonged to --
        so it would quietly stage a wheel that installs less than the one it
        was made from. Refusing keeps that from being a silent difference.
        """

        launchers = {
            entry_point.name
            for entry_point in self._distribution.entry_points
            if entry_point.group in ("console_scripts", "gui_scripts")
        }
        stray = sorted(
            row[0]
            for row in rows
            if (row[0].startswith("/") or ".." in row[0].split("/"))
            and row[0].rsplit("/", 1)[-1].removesuffix(".exe") not in launchers
        )
        if stray:
            raise NativeWorkerWheelBuildError(
                f"the installed Eyvan distribution placed {stray[0]} outside its package directory; that file came "
                "from a wheel data directory this repack cannot reconstruct, so no native-worker wheel is staged"
            )

    @classmethod
    def _recorded_files(
        cls,
        rows: tp.Sequence[tp.Sequence[str]],
        *,
        dist_info: str,
        install_root: Path,
    ) -> tuple[tuple[str, Path, bytes | None], ...]:
        """Select the RECORD entries a wheel actually carries, in archive order."""

        selected = [
            (row[0], install_root.joinpath(*row[0].split("/")), cls._recorded_digest(row[1] if len(row) > 1 else ""))
            for row in rows
            if cls._is_wheel_member(row[0], dist_info=dist_info)
        ]
        selected.sort(key=lambda entry: entry[0])
        return tuple(selected)

    @staticmethod
    def _is_wheel_member(name: str, *, dist_info: str) -> bool:
        """Return whether one RECORD path belongs inside the repacked wheel.

        Console scripts are recorded as ``../../../bin/eyvan``: they live
        outside the install root, and an installer regenerates them from
        ``entry_points.txt``. Bytecode is never shipped in a wheel. Everything
        else under the install root is exactly what the wheel put there.
        """

        parts = name.split("/")
        if not name or "\\" in name or "\x00" in name or any(part in ("", ".", "..") for part in parts):
            return False
        if name.endswith(".pyc") or "__pycache__" in parts:
            return False
        return not (parts[0] == dist_info and "/".join(parts[1:]) in _INSTALLER_OWNED_DIST_INFO_FILES)

    @staticmethod
    def _recorded_digest(value: str) -> bytes | None:
        """Decode a RECORD hash, which installers write base64 *or* hex encoded.

        ``None`` means "recorded with a hash this build cannot check", never
        "recorded wrong": a corrupt SHA-256 is refused rather than skipped.
        """

        algorithm, separator, encoded = value.partition("=")
        if not separator or algorithm != "sha256":
            return None
        if _RECORD_HEX_SHA256.fullmatch(encoded):
            return bytes.fromhex(encoded)
        try:
            digest = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        except ValueError as exc:
            raise NativeWorkerWheelBuildError(f"installed Eyvan RECORD has an undecodable SHA-256 {value!r}") from exc
        if len(digest) != hashlib.sha256().digest_size:
            raise NativeWorkerWheelBuildError(f"installed Eyvan RECORD has a truncated SHA-256 {value!r}")
        return digest

    @staticmethod
    def _wheel_tag(dist_info_path: Path) -> str:
        """Rebuild the filename tag triple from the installed ``WHEEL`` metadata."""

        wheel_metadata_path = dist_info_path / "WHEEL"
        try:
            wheel_metadata = wheel_metadata_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise NativeWorkerWheelBuildError(f"could not read {wheel_metadata_path}: {exc}") from exc
        columns: tuple[list[str], list[str], list[str]] = ([], [], [])
        for line in wheel_metadata.splitlines():
            key, separator, raw_value = line.partition(":")
            value = raw_value.strip()
            if not separator:
                continue
            if key.strip() == "Root-Is-Purelib" and value.lower() != "true":
                raise NativeWorkerWheelBuildError(
                    f"{wheel_metadata_path} declares Root-Is-Purelib {value!r}; a platlib-rooted Eyvan install "
                    "cannot be repacked without its original .data layout"
                )
            if key.strip() != "Tag":
                continue
            parts = value.split("-")
            if len(parts) != 3 or not all(parts):
                raise NativeWorkerWheelBuildError(f"{wheel_metadata_path} declares an unusable wheel tag {value!r}")
            for column, part in zip(columns, parts, strict=True):
                if part not in column:
                    column.append(part)
        if not all(columns):
            raise NativeWorkerWheelBuildError(f"{wheel_metadata_path} declares no wheel tag to name the repack with")
        return "-".join(".".join(column) for column in columns)

    @classmethod
    def _write_wheel(
        cls,
        wheel_path: Path,
        recorded: tp.Sequence[tuple[str, Path, bytes | None]],
        *,
        dist_info: str,
    ) -> None:
        """Write the repacked archive and the fresh RECORD that describes it."""

        record_lines: list[str] = []
        with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, path, expected in recorded:
                payload = cls._read_recorded_file(path, name)
                digest = hashlib.sha256(payload).digest()
                if expected is not None and digest != expected:
                    raise NativeWorkerWheelBuildError(
                        f"installed file {path} does not match the hash RECORD holds for {name}; reinstall Eyvan "
                        "before staging a native-worker wheel"
                    )
                cls._write_entry(archive, name, payload)
                encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
                record_lines.append(f"{name},sha256={encoded},{len(payload)}")
            record_name = f"{dist_info}/RECORD"
            record_lines.append(f"{record_name},,")
            cls._write_entry(archive, record_name, ("\n".join(record_lines) + "\n").encode("utf-8"))

    @staticmethod
    def _write_entry(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
        """Add one archive member with pinned metadata so repacks stay byte-stable."""

        entry = zipfile.ZipInfo(name, date_time=_WHEEL_ENTRY_DATE_TIME)
        entry.compress_type = zipfile.ZIP_DEFLATED
        entry.external_attr = _WHEEL_ENTRY_MODE << 16
        archive.writestr(entry, payload)

    @staticmethod
    def _read_recorded_file(path: Path, name: str) -> bytes:
        """Read one installed file, refusing anything that is not a regular file."""

        try:
            info = path.lstat()
        except OSError as exc:
            raise NativeWorkerWheelBuildError(
                f"installed file {path} recorded as {name} is not readable: {exc}"
            ) from exc
        if not stat.S_ISREG(info.st_mode):
            raise NativeWorkerWheelBuildError(f"installed file {path} recorded as {name} must be a regular file")
        try:
            return path.read_bytes()
        except OSError as exc:
            raise NativeWorkerWheelBuildError(f"could not read installed file {path}: {exc}") from exc


def native_worker_wheel_builder() -> NativeWorkerWheelSource:
    """Return the wheel source this installation can actually use.

    A source checkout still builds from source, so ``runtime deploy`` keeps
    meaning "deploy what is in this tree". An installed wheel has no project
    directory for ``uv build`` and needs none, because the native-worker wheel
    is this same distribution.
    """

    package_root = _default_package_root()
    if package_root is not None:
        return UvNativeWorkerWheelBuilder(package_root=package_root)
    return InstalledDistributionWheelBuilder(installed_eyvan_distribution())


@dataclass(frozen=True, slots=True)
class RuntimeInitializationResult:
    """Non-secret result of preparing one local native GCP runtime."""

    profile_path: Path
    profile: GcpNativeRuntimeProfile
    preflight: GcpRuntimePreflight

    def as_document(self) -> JsonObject:
        """Return automation-safe setup output without local credentials."""

        native_worker_wheel_uri = self.profile.native_worker_wheel_uri
        if native_worker_wheel_uri is None:
            raise AssertionError("a staged native runtime profile must include a worker wheel URI")
        return {
            "artifact_root_uri": self.profile.artifact_root_uri,
            "checks": [step.value for step in self.preflight.checks],
            "native_worker_wheel_uri": native_worker_wheel_uri,
            "profile_path": str(self.profile_path),
            "project_id": self.profile.project_id,
        }


def replacement_trusted_callable_projects() -> tuple[str, ...]:
    """Return the callable trust the replacement control plane will start with.

    The supervisor launches the new process with the *invoking* environment, so
    this is read here rather than from the outgoing process -- a POSIX process
    cannot recover another's environment, and inventing one would be worse than
    reporting the one that will actually apply.
    """

    raw = os.getenv("EYVAN_TRUSTED_CALLABLE_PROJECTS", "")
    return tuple(part.strip() for part in raw.split(",") if part.strip())


@dataclass(frozen=True, slots=True)
class RuntimeRestartResult:
    """Non-secret result of restarting the local native GCP API-and-agent runtime."""

    endpoint: ControlPlaneEndpoint
    control_plane: LocalDaemonResult
    agent_endpoint: ControlPlaneEndpoint
    agent_state: LocalAgentState
    trusted_callable_projects: tuple[str, ...] = ()

    @property
    def callable_trust_warning(self) -> str | None:
        """Warn when the replacement will refuse the callables the operator submits.

        ``EYVAN_TRUSTED_CALLABLE_PROJECTS`` is read once at control-plane start,
        so restarting from a shell that does not export it silently narrows the
        runtime: the restart reports success and the loss only surfaces later as
        a 403 on the next callable submission, which names the workload rather
        than the restart that caused it.
        """

        if self.trusted_callable_projects:
            return None
        return (
            "EYVAN_TRUSTED_CALLABLE_PROJECTS is unset, so this control plane trusts callables only in the "
            "default loopback project; submissions from any other org/project will be refused with 403. "
            "Export it before restarting to keep the previous trust."
        )

    def as_document(self) -> JsonObject:
        """Return automation-safe restart output without profile credentials."""

        return {
            "agent": {
                "endpoint": self.agent_endpoint.url,
                "state": self.agent_state.value,
            },
            "callable_trust_warning": self.callable_trust_warning,
            "control_plane": {
                "endpoint": self.control_plane.endpoint.url,
                "pid": self.control_plane.pid,
                "state": self.control_plane.state.value,
            },
            "endpoint": self.endpoint.url,
            "trusted_callable_projects": list(self.trusted_callable_projects),
        }


class GcpRuntimeInitializer:
    """Create private runtime credentials once, preflight GCP, and stage Eyvan's worker wheel."""

    def __init__(self, profile_store: RuntimeProfileStore, bootstrap: RuntimeBootstrapper):
        self._profile_store = profile_store
        self._bootstrap = bootstrap

    def initialize(self, *, project_id: str, artifact_root_uri: str) -> RuntimeInitializationResult:
        """Prepare a reusable native runtime without provisioning a TPU."""

        profile = self._profile_store.load_or_create(
            project_id=project_id,
            artifact_root_uri=artifact_root_uri,
        )
        preflight = self._bootstrap.preflight(profile)
        self._bootstrap.ensure_artifact_bucket(profile)
        staged_profile = self._bootstrap.stage_native_worker_wheel(profile)
        if staged_profile.native_worker_wheel_uri is None:
            raise GcpRuntimeInitializationError("native runtime wheel staging completed without a worker wheel URI")
        profile_path = self._profile_store.save(staged_profile)
        return RuntimeInitializationResult(profile_path=profile_path, profile=staged_profile, preflight=preflight)


def suggested_runtime_project(*, defaults_path: Path | None, cache_dir: Path | None) -> str | None:
    """Return a valid autoscale project first, then the active gcloud project when available."""

    try:
        defaults = AutoscaleConfigCache(cache_dir).load_defaults(defaults_path)
    except ValueError:
        defaults = None
    if defaults is not None and defaults.project is not None:
        try:
            return validate_gcp_project_id(defaults.project)
        except ValueError:
            pass
    candidate = gcloud_default_project()
    if candidate is None:
        return None
    try:
        return validate_gcp_project_id(candidate)
    except ValueError:
        return None


def normalize_runtime_artifact_root(project_id: str, artifact_root_uri: str) -> str:
    """Validate public artifact input through the authoritative runtime-profile schema.

    The transient profile is never persisted. Its placeholder secrets exist only
    because the profile schema deliberately validates all fields together.
    """

    return GcpNativeRuntimeProfile(
        project_id=project_id,
        artifact_root_uri=artifact_root_uri,
        agent_shared_secret=_VALIDATION_AGENT_SECRET,
        native_worker_root_secret=_VALIDATION_NATIVE_WORKER_SECRET,
    ).artifact_root_uri


def _process_output(result: subprocess.CompletedProcess[bytes]) -> str:
    """Decode bounded local build diagnostics consistently."""

    return b"\n".join((result.stderr or b"", result.stdout or b"")).decode("utf-8", "replace").strip()


def _prompt_project(console: Console, default: str | None) -> str:
    """Prompt until the GCP project is a canonical project ID."""

    while True:
        candidate = Prompt.ask("GCP project ID", default=default, console=console).strip()
        try:
            return validate_gcp_project_id(candidate)
        except ValueError as exc:
            console.print(f"[bold red]Invalid GCP project[/]: {exc}")


def _prompt_artifact_root(console: Console, project_id: str, default: str | None) -> str:
    """Prompt until the GCS root is accepted by the shared profile schema."""

    while True:
        candidate = Prompt.ask("GCS artifact root", default=default, console=console).strip()
        try:
            return normalize_runtime_artifact_root(project_id, candidate)
        except GcpNativeRuntimeProfileError as exc:
            console.print(f"[bold red]Invalid GCS artifact root[/]: {exc}")


def _render_runtime_review(
    console: Console,
    *,
    project_id: str,
    artifact_root_uri: str,
    profile_path: Path,
) -> None:
    """Render only public setup choices before any cloud-facing work begins."""

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Setting", style="bold cyan", min_width=22)
    table.add_column("Value", style="white", overflow="fold")
    table.add_row("GCP project", project_id)
    table.add_row("GCS artifact root", artifact_root_uri)
    table.add_row("Private profile", str(profile_path))
    print_block(console, Panel(table, title="[bold cyan]Review native GCP runtime[/]", border_style="cyan"))


def _render_runtime_ready(console: Console, result: RuntimeInitializationResult) -> None:
    """Render the initialized runtime without showing local credential material."""

    payload = result.as_document()
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Setting", style="bold cyan", min_width=22)
    table.add_column("Value", style="white", overflow="fold")
    table.add_row("GCP project", result.profile.project_id)
    table.add_row("GCS artifact root", result.profile.artifact_root_uri)
    table.add_row("Native worker package", str(payload["native_worker_wheel_uri"]))
    table.add_row("Private profile", str(result.profile_path))
    print_block(console, Panel(table, title="[bold green]Native GCP runtime ready[/]", border_style="green"))


def _render_runtime_restart_review(
    console: Console,
    *,
    endpoint: ControlPlaneEndpoint,
    profile_path: Path,
) -> None:
    """Render the public target of one explicit local daemon replacement."""

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Setting", style="bold cyan", min_width=22)
    table.add_column("Value", style="white", overflow="fold")
    table.add_row("Control plane", endpoint.url)
    table.add_row("Private profile", str(profile_path))
    print_block(console, Panel(table, title="[bold yellow]Restart native GCP runtime[/]", border_style="yellow"))


def _render_runtime_restarted(console: Console, result: RuntimeRestartResult) -> None:
    """Render a completed runtime restart without displaying profile credentials."""

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Service", style="bold cyan", min_width=22)
    table.add_column("State", style="white", overflow="fold")
    table.add_row("Control plane", result.control_plane.state.value)
    table.add_row("Cell agent", result.agent_state.value)
    if result.trusted_callable_projects:
        table.add_row("Trusted callables", ", ".join(result.trusted_callable_projects))
    print_block(console, Panel(table, title="[bold green]Native GCP runtime restarted[/]", border_style="green"))
    warning = result.callable_trust_warning
    if warning is not None:
        print_block(console, Panel(warning, title="[bold yellow]Callable trust narrowed[/]", border_style="yellow"))


def _initialize_native_runtime(
    *,
    project_id: str,
    artifact_root_uri: str,
    profile_path: Path,
) -> RuntimeInitializationResult:
    """Build, preflight, stage, and persist one native GCP runtime profile."""

    wheel_builder = native_worker_wheel_builder()
    try:
        initializer = GcpRuntimeInitializer(
            GcpNativeRuntimeProfileStore(profile_path),
            GcpRuntimeBootstrap(wheel_builder),
        )
        return initializer.initialize(project_id=project_id, artifact_root_uri=artifact_root_uri)
    finally:
        wheel_builder.close()


def _restart_native_runtime(
    *,
    endpoint: ControlPlaneEndpoint,
    profile_path: Path,
) -> RuntimeRestartResult:
    """Replace a verified local API and then prepare its matching native GCP agent."""

    profile_store = GcpNativeRuntimeProfileStore(profile_path)
    try:
        profile = profile_store.load()
    except FileNotFoundError as exc:
        raise GcpRuntimeRestartError("the native GCP runtime is not initialized; run `eyvan runtime init`") from exc
    except GcpNativeRuntimeProfileError as exc:
        raise GcpRuntimeRestartError(f"the private native GCP runtime profile is invalid: {exc}") from exc
    if profile.native_worker_wheel_uri is None:
        raise GcpRuntimeRestartError(
            "the native GCP runtime has no staged Eyvan worker package; run `eyvan runtime init`"
        )
    control_plane_supervisor = LocalDaemonSupervisor()
    control_plane = control_plane_supervisor.restart(endpoint, runtime_profile_path=profile_store.path)
    try:
        agent = LocalAgentSupervisor().restart(profile_store.path, profile)
    except BaseException as agent_error:
        try:
            control_plane_supervisor.stop_started(control_plane, runtime_profile_path=profile_store.path)
        except BaseException as cleanup_error:
            if isinstance(agent_error, Exception):
                if isinstance(cleanup_error, LocalDaemonError):
                    raise GcpRuntimeRestartError(
                        "Eyvan could not restart its local GCP cell agent and could not stop the newly restarted "
                        f"control plane: {cleanup_error}"
                    ) from agent_error
                raise
            # Cancellation still belongs to the caller even if rollback fails.
            # Keep its identity and disclose cleanup failure without masking it.
            agent_error.add_note(
                "Eyvan could not stop the newly restarted control plane during runtime interruption "
                f"({type(cleanup_error).__name__})."
            )
        raise
    return RuntimeRestartResult(
        trusted_callable_projects=replacement_trusted_callable_projects(),
        endpoint=endpoint,
        control_plane=control_plane,
        agent_endpoint=agent.endpoint,
        agent_state=agent.state,
    )


def _runtime_restart_endpoint(context: click.Context) -> ControlPlaneEndpoint:
    """Resolve the root CLI endpoint while keeping direct command tests self-contained."""

    root_session = context.find_root().obj
    if isinstance(root_session, CliSession):
        return ControlPlaneEndpoint.parse(root_session.base_url)
    return ControlPlaneEndpoint.parse(DEFAULT_CONTROL_PLANE_URL)


@click.group("runtime")
def runtime_group() -> None:
    """Prepare the local native GCP runtime used by Eyvan fleets."""


@runtime_group.command("init")
@click.option("--project", "project_id", help="GCP project ID. Defaults to autoscale defaults or active gcloud project.")
@click.option(
    "--artifact-root",
    help="GCS bucket or prefix used for Eyvan artifacts and native worker staging. "
    "Defaults to gs://eyvan-<project>, which Eyvan creates when absent.",
)
@click.option(
    "--profile",
    "profile_path",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Override the private ~/.eyvan/gcp-native-runtime.json profile path.",
)
@click.option(
    "--defaults",
    "defaults_path",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Read the project suggestion from this autoscale defaults YAML file.",
)
@click.option("--cache-dir", type=click.Path(path_type=Path, file_okay=False), help="Override ~/.eyvan/autoscale.")
@click.option("--yes", is_flag=True, help="Skip the interactive review confirmation.")
@click.option("--json", "as_json", is_flag=True, help="Emit the non-secret initialized runtime summary as JSON.")
def initialize_runtime(
    project_id: str | None,
    artifact_root: str | None,
    profile_path: Path | None,
    defaults_path: Path | None,
    cache_dir: Path | None,
    yes: bool,
    as_json: bool,
) -> None:
    """Create private local credentials and stage Eyvan's native GCP worker wheel."""

    console = autoscale_console(err=as_json)
    interactive = is_interactive()
    if interactive:
        print_block(
            console,
            Panel(
                "Choose the GCP project and GCS root once. Eyvan creates local private credentials automatically.",
                title="[bold cyan]Eyvan Native GCP Runtime[/]",
                border_style="cyan",
            ),
        )
    if project_id is None:
        project_id = suggested_runtime_project(defaults_path=defaults_path, cache_dir=cache_dir)
    if project_id is None:
        if not interactive:
            raise click.ClickException(
                "could not resolve a GCP project; pass --project or run `eyvan autoscale init` in a terminal"
            )
        project_id = _prompt_project(console, default=None)
    else:
        try:
            project_id = validate_gcp_project_id(project_id)
        except ValueError as exc:
            raise click.ClickException(
                f"invalid GCP project: {exc}; pass one you can see in `gcloud projects list` with --project"
            ) from exc

    if artifact_root is None:
        # The bucket is Eyvan-managed: derived from the project and created by
        # the bootstrap when absent, so setup never depends on a hand-made one.
        derived_default = default_artifact_root_uri(project_id)
        if interactive:
            artifact_root = _prompt_artifact_root(console, project_id, default=derived_default)
        else:
            artifact_root = derived_default
    else:
        try:
            artifact_root = normalize_runtime_artifact_root(project_id, artifact_root)
        except GcpNativeRuntimeProfileError as exc:
            raise click.ClickException(
                f"invalid GCS artifact root: {exc}; pass a bucket you own as --artifact-root gs://<bucket>/eyvan"
            ) from exc

    resolved_profile_path = GcpNativeRuntimeProfileStore(
        profile_path if profile_path is not None else DEFAULT_GCP_NATIVE_RUNTIME_PROFILE_PATH
    ).path
    if not yes:
        if not interactive:
            raise click.ClickException("runtime init requires --yes outside a terminal")
        _render_runtime_review(
            console,
            project_id=project_id,
            artifact_root_uri=artifact_root,
            profile_path=resolved_profile_path,
        )
        if not Confirm.ask("Initialize this runtime", default=True, console=console):
            raise click.Abort()

    try:
        result = _initialize_native_runtime(
            project_id=project_id,
            artifact_root_uri=artifact_root,
            profile_path=resolved_profile_path,
        )
    except (
        GcpNativeRuntimeProfileError,
        GcpRuntimeBootstrapError,
        NativeWorkerWheelBuildError,
        GcpRuntimeInitializationError,
    ) as exc:
        raise click.ClickException(f"could not initialize native GCP runtime: {exc}") from exc
    if as_json:
        click.echo(json.dumps(result.as_document(), indent=2, sort_keys=True))
    else:
        _render_runtime_ready(console, result)


@runtime_group.command("restart")
@click.option(
    "--profile",
    "profile_path",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Override the private ~/.eyvan/gcp-native-runtime.json profile path.",
)
@click.option("--yes", is_flag=True, help="Confirm replacement of a verified local Eyvan control plane.")
@click.option("--json", "as_json", is_flag=True, help="Emit the non-secret restart summary as JSON.")
@click.pass_context
def restart_runtime(
    context: click.Context,
    profile_path: Path | None,
    yes: bool,
    as_json: bool,
) -> None:
    """Replace the locally supervised API with the selected native GCP runtime profile."""

    console = autoscale_console(err=as_json)
    endpoint = _runtime_restart_endpoint(context)
    resolved_profile_path = GcpNativeRuntimeProfileStore(
        profile_path if profile_path is not None else DEFAULT_GCP_NATIVE_RUNTIME_PROFILE_PATH
    ).path
    if not yes:
        if not is_interactive():
            raise click.ClickException("runtime restart requires --yes outside a terminal")
        _render_runtime_restart_review(
            console,
            endpoint=endpoint,
            profile_path=resolved_profile_path,
        )
        if not Confirm.ask("Restart this local runtime", default=False, console=console):
            raise click.Abort()
    try:
        result = _restart_native_runtime(endpoint=endpoint, profile_path=resolved_profile_path)
    except (GcpRuntimeRestartError, LocalDaemonError) as exc:
        raise click.ClickException(f"could not restart native GCP runtime: {exc}") from exc
    if as_json:
        click.echo(json.dumps(result.as_document(), indent=2, sort_keys=True))
    else:
        _render_runtime_restarted(console, result)


@dataclass(frozen=True, slots=True)
class RuntimeDeployResult:
    """Non-secret result of redeploying every local Eyvan component at once."""

    wheel_uri: str
    restart: RuntimeRestartResult
    stale_after: tuple[str, ...]

    def as_document(self) -> JsonObject:
        """Return automation-safe deploy output without profile credentials."""

        document = dict(self.restart.as_document())
        document["native_worker_wheel_uri"] = self.wheel_uri
        document["stale_components"] = list(self.stale_after)
        return document


class GcpRuntimeDeployError(RuntimeError):
    """A local runtime could not be redeployed from the current source."""


def _deploy_native_runtime(*, endpoint: ControlPlaneEndpoint, profile_path: Path) -> RuntimeDeployResult:
    """Stage the wheel and restart every local component from one source tree.

    The three surfaces were previously updated by three separate commands, so
    forgetting one produced a fleet that worked and was wrong -- most expensively
    when a fix lived in the cell agent while only the TPU hosts were redeployed.
    Doing them together is the point; verifying afterwards is what makes it a
    guarantee rather than a convention.
    """

    store = GcpNativeRuntimeProfileStore(profile_path)
    try:
        profile = store.load()
    except FileNotFoundError as exc:
        raise GcpRuntimeDeployError("the native GCP runtime is not initialized; run `eyvan runtime init`") from exc
    except GcpNativeRuntimeProfileError as exc:
        raise GcpRuntimeDeployError(f"the private native GCP runtime profile is invalid: {exc}") from exc

    wheel_builder = native_worker_wheel_builder()
    try:
        staged = GcpRuntimeBootstrap(wheel_builder).stage_native_worker_wheel(profile)
    except GcpRuntimeBootstrapError as exc:
        raise GcpRuntimeDeployError(f"could not stage the native worker wheel: {exc}") from exc
    finally:
        wheel_builder.close()
    if staged.native_worker_wheel_uri is None:
        raise GcpRuntimeDeployError("wheel staging completed without a worker wheel URI")
    store.save(staged)

    restart = _restart_native_runtime(endpoint=endpoint, profile_path=store.path)

    stale = stale_components(runtime_profile_path=store.path)
    if stale:
        # Reporting success here would recreate the exact failure this command
        # exists to remove: a deployment that looks complete and is not.
        raise GcpRuntimeDeployError(
            "redeploy finished but these components still run older code than the source: " + ", ".join(stale)
        )
    return RuntimeDeployResult(wheel_uri=staged.native_worker_wheel_uri, restart=restart, stale_after=stale)


@runtime_group.command("deploy")
@click.option("--profile", "profile_path", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--yes", is_flag=True, help="Confirm redeploying every local Eyvan component.")
@click.option("--json", "as_json", is_flag=True, help="Emit the non-secret deploy summary as JSON.")
@click.pass_context
def deploy_runtime(context: click.Context, profile_path: Path | None, yes: bool, as_json: bool) -> None:
    """Rebuild and redeploy every local component from the current source.

    Stages the worker wheel and restarts the control plane and cell agent as one
    step, then verifies that nothing is left running older code.
    """

    console = autoscale_console(err=as_json)
    endpoint = _runtime_restart_endpoint(context)
    resolved_profile_path = GcpNativeRuntimeProfileStore(
        profile_path if profile_path is not None else DEFAULT_GCP_NATIVE_RUNTIME_PROFILE_PATH
    ).path
    if not yes:
        if not is_interactive():
            raise click.ClickException("runtime deploy requires --yes outside a terminal")
        if not Confirm.ask("Redeploy every local Eyvan component", default=False, console=console):
            raise click.Abort()
    try:
        result = _deploy_native_runtime(endpoint=endpoint, profile_path=resolved_profile_path)
    except (GcpRuntimeDeployError, GcpRuntimeRestartError, LocalDaemonError) as exc:
        raise click.ClickException(f"could not redeploy the native GCP runtime: {exc}") from exc
    if as_json:
        click.echo(json.dumps(result.as_document(), indent=2, sort_keys=True))
    else:
        _render_runtime_restarted(console, result.restart)
        console.print("[green]every local component now runs the current source[/]")


def register_runtime_commands(group: click.Group) -> None:
    """Attach native GCP runtime setup commands to the root CLI."""

    group.add_command(runtime_group)


__all__ = (
    "GcpRuntimeDeployError",
    "GcpRuntimeInitializationError",
    "GcpRuntimeInitializer",
    "GcpRuntimeRestartError",
    "InstalledDistributionWheelBuilder",
    "NativeWorkerWheelBuildError",
    "NativeWorkerWheelSource",
    "RuntimeDeployResult",
    "RuntimeInitializationResult",
    "RuntimeRestartResult",
    "UvNativeWorkerWheelBuilder",
    "initialize_runtime",
    "installed_eyvan_distribution",
    "native_worker_wheel_builder",
    "normalize_runtime_artifact_root",
    "register_runtime_commands",
    "restart_runtime",
    "runtime_group",
    "suggested_runtime_project",
)
