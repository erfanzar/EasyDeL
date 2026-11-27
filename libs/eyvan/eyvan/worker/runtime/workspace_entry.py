# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Materialize a workspace bundle, then exec the workload's own command.

The callable path has always been able to ship a working tree, because
``callable_worker`` fetches and unpacks one before unpickling. A plain command
had no equivalent, so ``eyvan run`` could only launch what was already baked
into the host image -- which is why using your own code on an accelerator meant
building a container.

This is that missing half, and deliberately nothing more. It verifies, extracts,
enters, and then ``exec``s. Replacing the process rather than spawning a child
keeps the workload's PID the one systemd supervises, so signals, exit codes,
cgroup accounting and ``process_status`` all continue to describe the real
process -- a wrapper that stayed resident would break every one of those.

Verification order matches :mod:`eyvan.worker.runtime.callable_protocol` and matters for
the same reason: the digest and signature are checked *before* anything is
written to disk, because the extracted tree becomes the process working
directory and can shadow whatever the workload resolves by name.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import shutil
import sys
import tempfile
import typing as tp
import urllib.parse
import urllib.request
from pathlib import Path

if __package__:
    from eyvan._internal.plane_tls import control_plane_urlopen_options
else:  # staged standalone copy in the OCI bridge mount (tests/test_oci_job_wrappers.py)
    # Inside a workload image ``eyvan`` is not importable. The stager puts this
    # file beside its stdlib-only siblings, and the wrapper runs with ``-I``,
    # which keeps the script's own directory off sys.path -- so add it here.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from plane_tls import control_plane_urlopen_options  # type: ignore[import-not-found,no-redef]

DEFAULT_WORKSPACE_DIRNAME = ".eyvan-workspace"

#: Written only after an extraction succeeds. This is completion metadata,
#: not an integrity guarantee for the writable execution tree: later
#: invocations always extract into their own private directory.
COMPLETE_MARKER = ".eyvan-extraction-complete"


_METADATA_TOKEN_URL = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"


def _fetch_gcs(uri: str) -> bytes:
    """Download one ``gs://bucket/object`` bundle using GCE metadata credentials."""

    remainder = uri.removeprefix("gs://")
    bucket_name, sep, object_name = remainder.partition("/")
    if not sep or not bucket_name or not object_name:
        raise ValueError(f"invalid GCS workspace URI: {uri}")
    token_request = urllib.request.Request(
        _METADATA_TOKEN_URL,
        headers={"Metadata-Flavor": "Google"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(token_request, timeout=10) as token_response:
        if token_response.headers.get("Metadata-Flavor", "") != "Google":
            raise RuntimeError("the metadata token reply did not identify itself as the Google metadata service")
        token_payload = json.loads(token_response.read().decode("utf-8"))
    access_token = token_payload["access_token"]
    url = (
        "https://storage.googleapis.com/storage/v1/b/"
        f"{urllib.parse.quote(bucket_name, safe='')}/o/"
        f"{urllib.parse.quote(object_name, safe='')}?alt=media"
    )
    object_request = urllib.request.Request(url, headers={"Authorization": f"Bearer {access_token}"})
    with urllib.request.urlopen(object_request, timeout=60) as response:
        return tp.cast(bytes, response.read())


def _fetch(uri: str, *, token: str | None) -> bytes:
    """Return the bundle bytes for a local path, GCS URI, or HTTP(S) managed URL."""

    if "://" not in uri or uri.startswith("file://"):
        return Path(uri.removeprefix("file://")).read_bytes()
    if uri.startswith("gs://"):
        return _fetch_gcs(uri)
    request = urllib.request.Request(uri)
    if not uri.startswith(("http://", "https://")):
        raise ValueError(f"unsupported workspace URI scheme: {uri}")
    if token:
        request.add_header("X-Eyvan-Artifact-Token", token)
    with urllib.request.urlopen(request, **control_plane_urlopen_options()) as response:
        return tp.cast(bytes, response.read())


def materialize(
    payload: bytes,
    *,
    sha256: str,
    signature: str | None,
    signing_key: bytes | None,
    base: Path,
) -> Path:
    """Verify and unpack a bundle under ``base``, returning where it landed.

    Both checks run before anything is written. Each invocation extracts the
    verified bytes into a new private directory: a previous workload may have
    modified its writable cwd or added shadow modules, so neither a digest
    directory name nor a completion marker makes that tree safe to reuse.
    Completed trees are retained because a sibling workload or child process
    may still be using them; reclaiming them requires execution ownership.
    """

    digest = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(digest, sha256):
        raise ValueError("workspace bundle does not match its content-addressed digest")
    if signing_key is not None:
        expected = hmac.new(signing_key, payload, hashlib.sha256).hexdigest()
        if signature is None or not hmac.compare_digest(expected, signature):
            raise ValueError("workspace bundle signature mismatch")
    if __package__:
        from eyvan._internal.workspace import create_private_directory, extract_workspace
    else:  # staged standalone copy: the hardened extractor is a sibling file
        from workspace import create_private_directory, extract_workspace  # type: ignore[import-not-found,no-redef]

    create_private_directory(base)
    # Never reuse, overwrite, or remove a sibling's execution tree. Unique
    # directories isolate retries and concurrent ranks from prior cwd writes.
    destination = Path(tempfile.mkdtemp(dir=base, prefix=f"{digest[:12]}-"))
    try:
        extract_workspace(payload, destination)
        (destination / COMPLETE_MARKER).write_text(digest, encoding="utf-8")
    except BaseException:
        # Only this invocation owns this incomplete tree. It has not been
        # returned or entered yet, so cleanup cannot disturb a live workload.
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return destination


def _default_workspace_base() -> Path:
    """Return where extracted workspaces land when ``EYVAN_WORKSPACE_DIR`` is unset.

    Prefers an explicit workspace or agent state directory, then
    ``~/.eyvan/workspaces`` when ``HOME`` is set and writable so local runs do
    not create untracked directories inside the user's working tree, and falls
    back to ``./.eyvan-workspace`` when no user home is available (e.g. ``env -i``
    in an isolated container).
    """

    configured = os.environ.get("EYVAN_WORKSPACE_DIR")
    if configured:
        return Path(configured)
    agent_state = os.environ.get("EYVAN_AGENT_STATE_DIR")
    if agent_state:
        return Path(agent_state).expanduser() / "workspaces"
    home = os.environ.get("HOME")
    if home:
        candidate = Path(home).expanduser() / ".eyvan" / "workspaces"
        probe = candidate
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        if probe.is_dir() and os.access(probe, os.W_OK | os.X_OK):
            return candidate
    return Path.cwd() / DEFAULT_WORKSPACE_DIRNAME


def main(argv: tp.Sequence[str] | None = None) -> int:
    """Materialize the workspace named by the arguments, then exec the command."""

    parser = argparse.ArgumentParser(prog="eyvan-workspace-entry")
    parser.add_argument("--workspace-uri")
    parser.add_argument("--workspace-uri-env")
    parser.add_argument("--workspace-token-env")
    parser.add_argument("--workspace-sha256", required=True)
    parser.add_argument("--workspace-signature")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args(list(argv) if argv is not None else None)

    command = list(arguments.command)
    # argparse hands the wrapper its own end-of-options marker back inside the
    # remainder; consume exactly that one. Any later `--` belongs to the
    # workload's command line -- `python tool.py -- --fast` means something --
    # and must be exec'd verbatim.
    if command and command[0] == "--":
        del command[0]
    if not command:
        parser.error("a command to execute is required")

    uri = arguments.workspace_uri
    token: str | None = None
    if arguments.workspace_uri_env:
        uri = os.environ.get(arguments.workspace_uri_env)
    if arguments.workspace_token_env:
        token = os.environ.get(arguments.workspace_token_env)
    if not uri:
        parser.error("a workspace URI is required")

    signing_key_value = os.environ.get("EYVAN_CALLABLE_SIGNING_KEY")
    signing_key = signing_key_value.encode() if signing_key_value else None
    base = _default_workspace_base()

    workspace = materialize(
        _fetch(uri, token=token),
        sha256=arguments.workspace_sha256,
        signature=arguments.workspace_signature,
        signing_key=signing_key,
        base=base,
    )
    if __package__:
        from eyvan._internal.workspace import discover_workspace_python_paths, prepend_pythonpath
    else:  # staged standalone copy: the hardened workspace helpers are a sibling file
        from workspace import (  # type: ignore[import-not-found,no-redef]
            discover_workspace_python_paths,
            prepend_pythonpath,
        )

    # Resolve workspace import paths before chdir in case ``base`` was relative.
    python_paths = discover_workspace_python_paths(workspace)
    os.environ["PYTHONPATH"] = prepend_pythonpath(python_paths, os.environ.get("PYTHONPATH"))
    os.chdir(workspace)
    # `exec` rather than spawn: the workload must keep the PID systemd started,
    # or signals, exit status and process introspection all describe a wrapper
    # instead of the job.
    os.execvp(command[0], command)
    raise AssertionError("execvp returned")  # pragma: no cover - unreachable


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    sys.exit(main())
