# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Disk-backed, restart-independent local ``eyvan gcp connect`` jobs.

Only validated request fields are persisted. Credentials stay in the owner's
normal CLI credential stores/environment, never in argv or job metadata.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
import typing as tp
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from eyvan._internal.ssh import SshDestination
from eyvan._internal.subprocess_env import provider_cli_env
from eyvan._internal.workspace import create_private_directory
from eyvan.capacity.fleet_config import validate_fleet_name
from eyvan.capacity.providers.gcp.tpu_nodes import validate_node_name

JOB_ID_PATTERN = r"^[0-9a-f]{32}$"
MAX_DOCUMENT_BYTES = 4 * 1024 * 1024


class AttachmentRequest(BaseModel):
    """Closed, credential-free inputs accepted by the attachment CLI."""

    model_config = ConfigDict(extra="forbid")

    target: str = Field(min_length=1, max_length=320)
    kind: tp.Literal["ssh", "tpu"] = "ssh"
    fleet: str | None = Field(default=None, max_length=63)
    project: str | None = Field(default=None, max_length=63)
    zone: str | None = Field(default=None, max_length=63)

    @model_validator(mode="after")
    def validate_options(self) -> AttachmentRequest:
        """Reject option injection and shell syntax before invoking any process."""

        if self.target != self.target.strip() or any(ord(char) < 32 for char in self.target):
            raise ValueError("target must be trimmed and contain no control characters")
        if self.kind == "ssh":
            SshDestination.parse(self.target)
        else:
            # The Cloud TPU API's own rule: real node names carry underscores
            # (team_spot2), which a hyphen-only pattern refused with a 422.
            try:
                validate_node_name(self.target)
            except ValueError:
                raise ValueError("invalid TPU name") from None
        if self.fleet is not None:
            validate_fleet_name(self.fleet)
        if self.project is not None and not re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", self.project):
            raise ValueError("invalid GCP project ID")
        if self.zone is not None and not re.fullmatch(r"[a-z]+[0-9]?-[a-z]+[0-9]+-[a-z]", self.zone):
            raise ValueError("invalid GCP zone")
        return self


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _private_directory(path: Path) -> None:
    create_private_directory(path)
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
        raise ValueError("attachment state must be an owner-only directory, not a symlink")


def _write(path: Path, payload: dict[str, tp.Any]) -> None:
    """Publish one new owner-only document atomically without following links."""

    temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read(path: Path) -> dict[str, tp.Any]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "r") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
            raise ValueError("attachment state must be an owner-only regular file")
        text = stream.read(MAX_DOCUMENT_BYTES + 1)
    if len(text) > MAX_DOCUMENT_BYTES:
        raise ValueError("attachment document is too large")
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("attachment CLI returned invalid JSON")
    return value


def cli_environment() -> dict[str, str]:
    """Use local credential discovery, not an inherited remote API bearer token."""

    environment = provider_cli_env()
    environment.pop("EYVAN_TOKEN", None)
    return environment


def cli_argv(payload: dict[str, tp.Any], *, dry_run: bool = False) -> list[str]:
    """Build a shell-free command pinned to the serving plane, with no secrets."""

    request = AttachmentRequest.model_validate(payload["request"])
    argv = [
        sys.executable,
        "-m",
        "eyvan.sdk.cli",
        "--url",
        payload["url"],
        "--user",
        payload["user"],
        "--role",
        "fleet_admin",
        "gcp",
        "connect",
        f"--{request.kind}",
        request.target,
    ]
    for option in ("fleet", "project", "zone"):
        value = getattr(request, option)
        if value is not None:
            argv.extend((f"--{option}", value))
    argv.append("--json")
    if dry_run:
        argv.append("--dry-run")
    return argv


def cli_error(progress: list[str], fallback: str) -> str:
    """Return the CLI's human diagnostic, never a subprocess/environment repr."""

    for line in reversed(progress):
        if line.startswith("Error:"):
            return line.removeprefix("Error:").strip()[:2000]
    return fallback


def run_plan(payload: dict[str, tp.Any]) -> dict[str, tp.Any]:
    """Run the read-only CLI preview, with a bounded synchronous deadline."""

    result = subprocess.run(
        cli_argv(payload, dry_run=True),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        env=cli_environment(),
    )
    if result.returncode:
        raise ValueError(cli_error(result.stderr.splitlines(), "Attachment preview failed; check local SSH access."))
    try:
        plan = json.loads(result.stdout)
        if not isinstance(plan, dict) or not isinstance(plan.get("fleet"), str) or plan.get("dry_run") is not True:
            raise ValueError
    except (ValueError, TypeError):
        raise ValueError("Attachment CLI returned invalid JSON for the preview.") from None
    return plan


class AttachmentJobs:
    """Read and launch owner-only jobs independently of any application instance."""

    def __init__(self, directory: Path | None = None):
        self.directory = directory or Path.home() / ".eyvan" / "attachments"

    def start(self, payload: dict[str, tp.Any]) -> dict[str, tp.Any]:
        """Detach a tiny supervisor so CLI redeployment cannot kill its own job."""

        _private_directory(self.directory)
        job_id = uuid.uuid4().hex
        directory = self.directory / job_id
        directory.mkdir(mode=0o700)
        _write(directory / "request.json", {**payload, "id": job_id, "started_at": _now()})
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "eyvan.server.local.attach_job", str(directory)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=cli_environment(),
            )
            _write(directory / "runner.json", {"pid": process.pid})
        except OSError:
            _write(
                directory / "exit.json",
                {"code": 1, "finished_at": _now(), "error": "Could not start the local attachment runner."},
            )
        return self.get(job_id)

    def get(self, job_id: str) -> dict[str, tp.Any]:
        """Recover status and a bounded progress tail from durable files."""

        if not re.fullmatch(JOB_ID_PATTERN, job_id):
            raise ValueError("invalid attachment ID")
        directory = self.directory / job_id
        if not directory.exists():
            raise FileNotFoundError(job_id)
        _private_directory(directory)
        request = _read(directory / "request.json")
        progress: list[str] = []
        try:
            descriptor = os.open(directory / "progress.log", os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(descriptor, "rb") as stream:
                stream.seek(0, os.SEEK_END)
                stream.seek(max(0, stream.tell() - 65536))
                progress = stream.read().decode("utf-8", errors="replace").splitlines()[-300:]
        except FileNotFoundError:
            pass
        status, result, error, finished_at = "running", None, None, None
        try:
            completion = _read(directory / "exit.json")
        except FileNotFoundError:
            completion = None
        if completion is not None:
            finished_at = completion["finished_at"]
            if completion["code"] == 0:
                try:
                    result = _read(directory / "result.json")
                    status = "succeeded"
                except (OSError, ValueError):
                    status, error = "failed", "Attachment CLI returned invalid JSON; check the fleet inventory."
            else:
                status = "failed"
                error = completion.get("error") or cli_error(progress, "Attachment failed; check local SSH access.")
        else:
            try:
                os.kill(_read(directory / "runner.json")["pid"], 0)
            except ProcessLookupError:
                status, error = (
                    "failed",
                    "Attachment runner stopped unexpectedly; check the fleet inventory before retrying.",
                )
            except FileNotFoundError:
                pass  # A concurrent reader may arrive before Popen returns.
        return {
            "id": job_id,
            "target": request["request"]["target"],
            "status": status,
            "progress": progress,
            "result": result,
            "error": error,
            "started_at": request["started_at"],
            "finished_at": finished_at,
        }

    def recent(self, *, url: str) -> list[dict[str, tp.Any]]:
        """List recent jobs for this exact local plane, not another local profile."""

        if not self.directory.exists():
            return []
        _private_directory(self.directory)
        jobs = []
        for directory in self.directory.iterdir():
            if not re.fullmatch(JOB_ID_PATTERN, directory.name):
                continue
            try:
                if _read(directory / "request.json")["url"] == url:
                    jobs.append(self.get(directory.name))
            except (OSError, ValueError, KeyError):
                continue
        return sorted(jobs, key=lambda job: job["started_at"], reverse=True)[:30]


def run_job(directory: Path) -> int:
    """Run the CLI child and always publish completion after its output closes."""

    _private_directory(directory)
    payload = _read(directory / "request.json")
    error = None
    try:
        stdout_fd = os.open(directory / "result.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        stderr_fd = os.open(directory / "progress.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(stdout_fd, "w") as stdout, os.fdopen(stderr_fd, "w") as stderr:
            result = subprocess.run(
                cli_argv(payload),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                text=True,
                env=cli_environment(),
            )
            code = result.returncode
    except Exception:
        code, error = 1, "Local attachment runner failed; check the fleet inventory before retrying."
    _write(directory / "exit.json", {"code": code, "finished_at": _now(), "error": error})
    return code


if __name__ == "__main__":
    raise SystemExit(run_job(Path(sys.argv[1])))
