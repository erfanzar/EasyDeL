# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Content-addressed storage for trusted Eyvan callable artifacts."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
import typing as tp
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from eyvan._internal.subprocess_env import provider_cli_env
from eyvan._internal.workspace import create_private_directory
from eyvan.artifacts.managed import result_uri as managed_result_uri
from eyvan.worker.runtime.callables import CallableArtifact

_RESULT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True, slots=True)
class StoredCallableArtifact:
    """Immutable reference to a serialized callable in durable storage."""

    uri: str
    sha256: str
    python: str
    format_version: int
    signature: str | None = None


class ArtifactStore(tp.Protocol):
    """Storage boundary used by callable submission and result retrieval."""

    def put(self, artifact: CallableArtifact) -> StoredCallableArtifact:
        """Store a callable once under its digest and return its reference."""

        ...

    def put_workspace(self, payload: bytes, sha256: str) -> str:
        """Store a workspace bundle once under its digest and return its URI."""

        ...

    def read(self, uri: str) -> bytes:
        """Read an artifact or result manifest from ``uri``."""

        ...

    def allocate_result_uri(self, invocation_id: str) -> str:
        """Allocate a unique result-manifest URI for an invocation."""

        ...


class CommandRunner(tp.Protocol):
    """Injectable subprocess boundary for GCS operations."""

    def __call__(
        self,
        args: tp.Sequence[str],
        *,
        check: bool,
        capture_output: bool,
    ) -> subprocess.CompletedProcess[bytes]: ...


class GcsCommandError(RuntimeError):
    """A ``gcloud`` invocation failed, reporting what the CLI itself said.

    ``subprocess`` captures the CLI's stderr and then buries it inside
    ``CalledProcessError``, so a traceback shows an exit status and nothing
    else. A user submitting a workload has then lost the only sentence that
    named the fault -- credentials, quota, network, a bucket that is gone --
    and what was a one-line answer becomes an investigation. The command and
    its stderr travel with this error instead, so the next failure names
    itself.
    """


#: How much of a CLI's stderr an error carries. Enough for any gcloud
#: diagnostic, short enough that a traceback stays readable.
_COMMAND_ERROR_STDERR_LIMIT = 2000


def _gcs_command_error(
    args: tp.Sequence[str],
    returncode: int,
    stderr: str,
) -> GcsCommandError:
    """Build the error that names the command and its own failure reason."""

    if len(stderr) > _COMMAND_ERROR_STDERR_LIMIT:
        stderr = f"{stderr[:_COMMAND_ERROR_STDERR_LIMIT]}… [truncated]"
    return GcsCommandError(
        f"'{' '.join(args)}' failed with exit status {returncode}"
        + (f": {stderr}" if stderr else " and printed nothing to stderr")
    )


def _gcloud_checked(
    runner: CommandRunner,
    args: tp.Sequence[str],
) -> subprocess.CompletedProcess[bytes]:
    """Run one ``gcloud`` command, refusing to swallow its failure reason."""

    try:
        return runner(args, check=True, capture_output=True)
    except subprocess.CalledProcessError as error:
        stderr = (error.stderr or b"").decode("utf-8", "replace").strip()
        raise _gcs_command_error(args, error.returncode, stderr) from error


def _default_gcloud_runner(
    args: tp.Sequence[str],
    *,
    check: bool,
    capture_output: bool,
) -> subprocess.CompletedProcess[bytes]:
    """Run ``gcloud`` without the caller's Python interpreter steering.

    The default used to be a bare ``subprocess.run``, which leaked a submitting
    process's ``PYTHONPATH`` into gcloud's own interpreter; see
    ``eyvan._internal.subprocess_env``.
    """

    return subprocess.run(args, check=check, capture_output=capture_output, env=provider_cli_env())


class ManagedArtifactClient(tp.Protocol):
    """Authenticated SDK operations used by the default managed store."""

    def upload_managed_callable(
        self,
        artifact: CallableArtifact,
        *,
        org_id: str,
        project_id: str,
    ) -> StoredCallableArtifact: ...

    def upload_managed_workspace(
        self,
        payload: bytes,
        sha256: str,
        *,
        org_id: str,
        project_id: str,
    ) -> str: ...

    def read_managed_artifact(self, uri: str, *, org_id: str, project_id: str) -> bytes: ...


def _validate_artifact(artifact: CallableArtifact) -> None:
    digest = hashlib.sha256(artifact.payload).hexdigest()
    if digest != artifact.sha256:
        raise ValueError("callable artifact digest does not match its payload")


def _validate_workspace(payload: bytes, sha256: str) -> None:
    if not isinstance(payload, bytes):
        raise TypeError("workspace bundle payload must be bytes")
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError("workspace bundle digest does not match its payload")


def _validate_invocation_id(invocation_id: str) -> None:
    if not _RESULT_ID.fullmatch(invocation_id):
        raise ValueError("invocation_id must contain only letters, digits, dots, underscores, or hyphens")


def _stored(uri: str, artifact: CallableArtifact) -> StoredCallableArtifact:
    return StoredCallableArtifact(
        uri=uri,
        sha256=artifact.sha256,
        python=artifact.python,
        format_version=artifact.format_version,
        signature=artifact.signature,
    )


def _local_path(value: str | os.PathLike[str]) -> Path:
    raw = os.fspath(value)
    if raw.startswith("file://"):
        parsed = urlparse(raw)
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError("local artifact file URIs cannot name a remote host")
        return Path(unquote(parsed.path))
    return Path(raw).expanduser()


class LocalArtifactStore:
    """Content-addressed artifact store rooted in a local directory."""

    def __init__(self, root: str | os.PathLike[str]):
        self.root = _local_path(root).resolve()

    def _put_digest_object(self, payload: bytes, sha256: str, *, kind: str, extension: str) -> Path:
        """Write one immutable digest-named object and re-verify what landed."""

        target = self.root / kind / f"{sha256}{extension}"
        create_private_directory(target.parent)
        if not target.exists():
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
                    temporary.write(payload)
                    temporary.flush()
                    os.fsync(temporary.fileno())
                    temporary_path = Path(temporary.name)
                try:
                    os.link(temporary_path, target)
                except FileExistsError:
                    pass
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
        if hashlib.sha256(target.read_bytes()).hexdigest() != sha256:
            raise ValueError(f"stored {kind} artifact at {target} has an invalid digest")
        return target

    def put(self, artifact: CallableArtifact) -> StoredCallableArtifact:
        """Persist ``artifact`` without replacing an existing digest object."""

        _validate_artifact(artifact)
        target = self._put_digest_object(artifact.payload, artifact.sha256, kind="callables", extension=".pkl")
        return _stored(target.as_uri(), artifact)

    def put_workspace(self, payload: bytes, sha256: str) -> str:
        """Persist a workspace bundle and return its content-addressed URI."""

        _validate_workspace(payload, sha256)
        return self._put_digest_object(payload, sha256, kind="workspaces", extension=".tar.gz").as_uri()

    def read(self, uri: str) -> bytes:
        """Read bytes from a local path or file URI confined to the store root."""

        if uri.startswith("gs://"):
            raise ValueError("LocalArtifactStore cannot read GCS URIs")
        path = _local_path(uri).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"local artifact read {uri!r} escapes the store root {self.root}")
        return path.read_bytes()

    def allocate_result_uri(self, invocation_id: str) -> str:
        """Return the invocation's unique local result-manifest URI."""

        _validate_invocation_id(invocation_id)
        results = self.root / "results"
        create_private_directory(results)
        return (results / f"{invocation_id}.result").as_uri()


class GCSArtifactStore:
    """Content-addressed artifact store backed by ``gcloud storage``."""

    def __init__(self, root_uri: str, *, runner: CommandRunner = None):
        root_uri = root_uri.rstrip("/")
        parsed = urlparse(root_uri)
        if parsed.scheme != "gs" or not parsed.netloc:
            raise ValueError("GCS artifact storage requires a gs://bucket[/prefix] URI")
        self.root_uri = root_uri
        self._runner = _default_gcloud_runner if runner is None else runner

    def _verify_stored_digest(self, uri: str, sha256: str, *, kind: str) -> None:
        """Re-hash a stored digest-named object; a foreign one is rejected."""

        if hashlib.sha256(self.read(uri)).hexdigest() != sha256:
            raise ValueError(f"stored {kind} artifact at {uri} has an invalid digest")

    def _put_digest_object(self, payload: bytes, sha256: str, *, kind: str, extension: str) -> str:
        """Upload one immutable digest-named object only when it is absent.

        A pre-existing object under the digest name is fetched and re-hashed; a
        corrupted or foreign object is rejected instead of trusted. A
        concurrent submitter landing the same digest between our check and our
        upload is that case arriving late, not a failure: ``--no-clobber``
        answers 412, the losing upload verifies what the winner wrote, and
        submission continues -- identical callables submitted side by side are
        one object, and none of their submitters crashes.
        """

        uri = f"{self.root_uri}/{kind}/{sha256}{extension}"
        described = self._runner(
            ["gcloud", "storage", "objects", "describe", uri],
            check=False,
            capture_output=True,
        )
        if described.returncode == 0:
            self._verify_stored_digest(uri, sha256, kind=kind)
            return uri
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False) as temporary:
                temporary.write(payload)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            try:
                _gcloud_checked(
                    self._runner,
                    ["gcloud", "storage", "cp", "--no-clobber", str(temporary_path), uri],
                )
            except GcsCommandError as error:
                if "HTTPError 412" not in str(error):
                    raise
                # The object appeared between the describe and this upload:
                # a concurrent submitter landed the same digest first. The
                # name is the content's digest, so what they wrote is what
                # we were uploading -- verify it and continue, exactly as if
                # it had been there all along.
                self._verify_stored_digest(uri, sha256, kind=kind)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return uri

    def put(self, artifact: CallableArtifact) -> StoredCallableArtifact:
        """Upload ``artifact`` only when its digest object is absent."""

        _validate_artifact(artifact)
        uri = self._put_digest_object(artifact.payload, artifact.sha256, kind="callables", extension=".pkl")
        return _stored(uri, artifact)

    def put_workspace(self, payload: bytes, sha256: str) -> str:
        """Upload a workspace bundle and return its content-addressed URI."""

        _validate_workspace(payload, sha256)
        return self._put_digest_object(payload, sha256, kind="workspaces", extension=".tar.gz")

    def read(self, uri: str) -> bytes:
        """Read an object with ``gcloud storage cat``."""

        if not uri.startswith("gs://"):
            raise ValueError("GCSArtifactStore can only read gs:// URIs")
        return _gcloud_checked(self._runner, ["gcloud", "storage", "cat", uri]).stdout

    def allocate_result_uri(self, invocation_id: str) -> str:
        """Return the invocation's unique GCS result-manifest URI."""

        _validate_invocation_id(invocation_id)
        return f"{self.root_uri}/results/{invocation_id}.result"


class EyvanManagedArtifactStore:
    """Default artifact store proxied through the authenticated Eyvan API."""

    def __init__(self, client: ManagedArtifactClient, *, org_id: str, project_id: str):
        self.client = client
        self.org_id = org_id
        self.project_id = project_id

    def put(self, artifact: CallableArtifact) -> StoredCallableArtifact:
        """Upload a callable to the control plane's configured storage backend."""

        _validate_artifact(artifact)
        return self.client.upload_managed_callable(artifact, org_id=self.org_id, project_id=self.project_id)

    def put_workspace(self, payload: bytes, sha256: str) -> str:
        """Upload a workspace bundle through the authenticated control-plane API."""

        _validate_workspace(payload, sha256)
        return self.client.upload_managed_workspace(payload, sha256, org_id=self.org_id, project_id=self.project_id)

    def read(self, uri: str) -> bytes:
        """Read a managed result through the authenticated control-plane API."""

        return self.client.read_managed_artifact(uri, org_id=self.org_id, project_id=self.project_id)

    def allocate_result_uri(self, invocation_id: str) -> str:
        """Allocate a logical result URI without exposing the physical backend."""

        _validate_invocation_id(invocation_id)
        return managed_result_uri(invocation_id)


def artifact_store_from_uri(uri: str) -> ArtifactStore:
    """Create the native artifact store selected by ``uri``."""

    if uri.startswith("gs://"):
        return GCSArtifactStore(uri)
    return LocalArtifactStore(uri)
