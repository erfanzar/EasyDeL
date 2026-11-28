# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Integrity and confinement behavior of the native artifact stores."""

from __future__ import annotations

import subprocess
import typing as tp
from pathlib import Path

import pytest

from eyvan.artifacts import GCSArtifactStore, GcsCommandError, LocalArtifactStore
from eyvan.worker.runtime.callables import dump_callable


def _add(left: int, right: int = 1) -> int:
    return left + right


class FakeGcloudRunner:
    """In-memory ``gcloud storage`` double used by the GCS store tests."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.calls: list[list[str]] = []

    def __call__(
        self,
        args: tp.Sequence[str],
        *,
        check: bool,
        capture_output: bool,
    ) -> subprocess.CompletedProcess[bytes]:
        command = list(args)
        self.calls.append(command)
        uri = command[-1]
        if command[2:4] == ["objects", "describe"]:
            return subprocess.CompletedProcess(command, 0 if uri in self.objects else 1, stdout=b"", stderr=b"")
        if command[2] == "cp":
            self.objects[uri] = Path(command[-2]).read_bytes()
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")
        if command[2] == "cat":
            return subprocess.CompletedProcess(command, 0, stdout=self.objects[uri], stderr=b"")
        raise AssertionError(command)


def test_local_store_round_trips_a_put_within_the_root(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    artifact = dump_callable(_add)

    reference = store.put(artifact)

    assert store.read(reference.uri) == artifact.payload


def test_local_read_is_confined_to_the_store_root(tmp_path) -> None:
    root = tmp_path / "artifacts"
    store = LocalArtifactStore(root)
    store.put(dump_callable(_add))
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"foreign-bytes")

    with pytest.raises(ValueError, match="escapes the store root"):
        store.read(str(outside))
    with pytest.raises(ValueError, match="escapes the store root"):
        store.read(outside.as_uri())
    with pytest.raises(ValueError, match="escapes the store root"):
        store.read(str(root / "callables" / ".." / ".." / "outside.bin"))


def test_local_read_rejects_symlinks_escaping_the_root(tmp_path) -> None:
    root = tmp_path / "artifacts"
    store = LocalArtifactStore(root)
    store.put(dump_callable(_add))
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"foreign-bytes")
    (root / "callables" / "link.bin").symlink_to(outside)

    with pytest.raises(ValueError, match="escapes the store root"):
        store.read(str(root / "callables" / "link.bin"))


def test_local_read_still_rejects_gcs_uris(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")

    with pytest.raises(ValueError, match="cannot read GCS URIs"):
        store.read("gs://bucket/object.pkl")


def test_gcs_put_uploads_once_and_verifies_a_preexisting_object() -> None:
    runner = FakeGcloudRunner()
    store = GCSArtifactStore("gs://artifacts/eyvan", runner=runner)
    artifact = dump_callable(_add)

    first = store.put(artifact)
    second = store.put(artifact)

    assert second == first
    assert sum(command[2] == "cp" for command in runner.calls) == 1
    assert sum(command[2] == "cat" for command in runner.calls) == 1
    assert store.read(first.uri) == artifact.payload


def test_gcs_put_rejects_a_corrupted_preexisting_object() -> None:
    runner = FakeGcloudRunner()
    store = GCSArtifactStore("gs://artifacts/eyvan", runner=runner)
    artifact = dump_callable(_add)
    uri = f"gs://artifacts/eyvan/callables/{artifact.sha256}.pkl"
    runner.objects[uri] = b"tampered-payload"

    with pytest.raises(ValueError, match="invalid digest"):
        store.put(artifact)
    assert not any(command[2] == "cp" for command in runner.calls)


def test_a_failed_upload_reports_what_gcloud_said() -> None:
    """A swallowed CLI stderr turned every upload failure into a mystery.

    ``subprocess`` captured gcloud's diagnostic and ``CalledProcessError``
    buried it, so a user's traceback said "exit status 1" and nothing else --
    credentials, quota, and network blips all looked identical. The store now
    refuses to report a CLI failure without the CLI's own words for it.
    """

    class FailingRunner(FakeGcloudRunner):
        """Describe says the object is absent; the upload then fails."""

        def __call__(self, args, *, check, capture_output):
            command = list(args)
            if command[2] == "cp" and check:
                raise subprocess.CalledProcessError(
                    returncode=1,
                    cmd=command,
                    stderr=b"ERROR: (gcloud.storage.cp) Your credentials are not valid.\n",
                )
            return super().__call__(args, check=check, capture_output=capture_output)

    store = GCSArtifactStore("gs://artifacts/eyvan", runner=FailingRunner())

    with pytest.raises(GcsCommandError, match="Your credentials are not valid") as caught:
        store.put(dump_callable(_add))

    message = str(caught.value)
    assert "gcloud storage cp" in message, "the failing command must be named"
    assert "exit status 1" in message


def test_a_failed_read_reports_what_gcloud_said() -> None:
    class FailingRunner(FakeGcloudRunner):
        def __call__(self, args, *, check, capture_output):
            command = list(args)
            if command[2] == "cat" and check:
                raise subprocess.CalledProcessError(
                    returncode=1,
                    cmd=command,
                    stderr=b"ERROR: (gcloud.storage.cat) No such object: bucket/x\n",
                )
            return super().__call__(args, check=check, capture_output=capture_output)

    store = GCSArtifactStore("gs://artifacts/eyvan", runner=FailingRunner())

    with pytest.raises(GcsCommandError, match="No such object"):
        store.read("gs://artifacts/eyvan/callables/missing.pkl")


def test_a_silent_cli_failure_says_it_printed_nothing() -> None:
    class SilentRunner(FakeGcloudRunner):
        def __call__(self, args, *, check, capture_output):
            command = list(args)
            if command[2] == "cp" and check:
                raise subprocess.CalledProcessError(returncode=1, cmd=command, stderr=None)
            return super().__call__(args, check=check, capture_output=capture_output)

    store = GCSArtifactStore("gs://artifacts/eyvan", runner=SilentRunner())

    with pytest.raises(GcsCommandError, match="printed nothing to stderr"):
        store.put(dump_callable(_add))
