# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Dependency-light protocol shared by native and OCI callable workers.

This module deliberately imports no Eyvan package modules.  The OCI executor
stages this exact file beside its runner, while the native worker imports it
through the normal package path.  Keeping the format, integrity checks, rank
ownership, and URI behavior here prevents the two execution paths from
drifting.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import http.client
import inspect
import json
import os
import sys
import tempfile
import time
import typing as tp
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

import cloudpickle

try:  # staged OCI layout: siblings on sys.path, no package
    # Sibling first, deliberately. This file is also executed as a bare module
    # staged next to its siblings inside an OCI image, where importing ``eyvan``
    # is not merely unavailable but forbidden -- the runner must see only the
    # bridge, not the host environment. A package-first import would reach for
    # ``eyvan`` on that path, and the fallback would not catch what a blocked
    # package raises. Nothing can shadow this on a native worker: the module is
    # already in ``sys.modules`` before any user workspace joins ``sys.path``.
    from http_retry import HTTP_ATTEMPTS, RETRYABLE_HTTP_STATUS, retry_delay  # type: ignore[import-not-found]
    from plane_tls import control_plane_urlopen_options  # type: ignore[import-not-found]
except ImportError:
    from eyvan._internal.plane_tls import control_plane_urlopen_options
    from eyvan.worker.runtime.http_retry import HTTP_ATTEMPTS, RETRYABLE_HTTP_STATUS, retry_delay

FORMAT_VERSION = 1
"""Current callable and result artifact wire-format version."""

_RESULT_MAGIC = b"EYVAN-CALLABLE-RESULT\n"

JsonValue: tp.TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
"""Recursive JSON values accepted as decorated-callable arguments."""


@dataclass(frozen=True, slots=True)
class CallableArtifact:
    """Serialized callable with compatibility and integrity metadata."""

    payload: bytes
    sha256: str
    python: str
    format_version: int = FORMAT_VERSION
    signature: str | None = None


@dataclass(frozen=True, slots=True)
class ReplicaContext:
    """Validated rank metadata for one callable worker process."""

    index: int
    count: int

    @classmethod
    def from_environment(cls, environment: tp.Mapping[str, str]) -> ReplicaContext:
        """Read either single-replica or native multihost rank metadata."""

        replica = cls._read_pair(environment, "EYVAN_REPLICA_INDEX", "EYVAN_REPLICA_COUNT")
        distributed = cls._read_pair(
            environment,
            "EYVAN_DISTRIBUTED_PROCESS_ID",
            "EYVAN_DISTRIBUTED_PROCESS_COUNT",
        )
        if replica is not None and distributed is not None and replica != distributed:
            raise RuntimeError("Eyvan callable replica and distributed process metadata disagree")
        return distributed or replica or cls(index=0, count=1)

    @classmethod
    def _read_pair(
        cls,
        environment: tp.Mapping[str, str],
        index_name: str,
        count_name: str,
    ) -> ReplicaContext | None:
        """Parse one paired rank/count environment contract."""

        raw_index = environment.get(index_name)
        raw_count = environment.get(count_name)
        if raw_index is None and raw_count is None:
            return None
        if raw_index is None or raw_count is None:
            raise RuntimeError(f"Eyvan callable requires both {index_name} and {count_name}")
        try:
            index = int(raw_index)
            count = int(raw_count)
        except ValueError as exc:
            raise RuntimeError("Eyvan callable has invalid replica metadata") from exc
        if count < 1 or index < 0 or index >= count:
            raise RuntimeError("Eyvan callable has invalid replica metadata")
        return cls(index=index, count=count)

    @property
    def owns_result(self) -> bool:
        """Return whether this replica persists the one durable job result."""

        return self.index == 0


class GcsObjectTransfer(tp.Protocol):
    """The small GCS boundary implemented by each execution environment."""

    def read_object(self, bucket_name: str, object_name: str) -> bytes:
        """Return one GCS object's bytes."""

        ...

    def write_object(self, bucket_name: str, object_name: str, payload: bytes) -> None:
        """Persist one GCS object's bytes."""

        ...


class ArtifactTransfer(tp.Protocol):
    """Read callable and write result payloads through accepted URI schemes."""

    def read(self, uri: str, *, token: str | None = None) -> bytes:
        """Read one payload."""

        ...

    def write(self, uri: str, payload: bytes, *, token: str | None = None) -> None:
        """Write one payload."""

        ...


def dump_callable(fn: tp.Callable[..., object], *, signing_key: bytes | None = None) -> CallableArtifact:
    """Serialize a callable for a trusted Eyvan project."""

    if not callable(fn):
        raise TypeError("callable workloads require a callable object")
    payload = cloudpickle.dumps(fn)
    digest = hashlib.sha256(payload).hexdigest()
    signature = hmac.new(signing_key, payload, hashlib.sha256).hexdigest() if signing_key else None
    return CallableArtifact(
        payload=payload,
        sha256=digest,
        python=f"{sys.version_info.major}.{sys.version_info.minor}",
        signature=signature,
    )


def load_callable(
    artifact: CallableArtifact,
    *,
    signing_key: bytes | None = None,
) -> tp.Callable[..., object]:
    """Validate and deserialize one trusted callable artifact.

    Deserialization executes pickle opcodes. Callers must enforce trusted
    project policy before invoking this function.
    """

    if artifact.format_version != FORMAT_VERSION:
        raise ValueError(f"unsupported callable format {artifact.format_version}")
    current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if artifact.python != current_python:
        raise ValueError(f"callable requires Python {artifact.python}; current runtime is {current_python}")
    digest = hashlib.sha256(artifact.payload).hexdigest()
    if not hmac.compare_digest(digest, artifact.sha256):
        raise ValueError("callable artifact digest mismatch")
    if signing_key is not None:
        expected = hmac.new(signing_key, artifact.payload, hashlib.sha256).hexdigest()
        if artifact.signature is None or not hmac.compare_digest(expected, artifact.signature):
            raise ValueError("callable artifact signature mismatch")
    loaded = cloudpickle.loads(artifact.payload)
    if not callable(loaded):
        raise ValueError("callable artifact did not deserialize to a callable")
    return tp.cast(tp.Callable[..., object], loaded)


def dump_result(value: object, *, signing_key: bytes | None = None) -> bytes:
    """Serialize one Python result into a self-validating binary artifact."""

    payload = cloudpickle.dumps(value)
    header = {
        "format_version": FORMAT_VERSION,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "signature": hmac.new(signing_key, payload, hashlib.sha256).hexdigest() if signing_key else None,
    }
    encoded_header = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return _RESULT_MAGIC + encoded_header + b"\n" + payload


def load_result(artifact: bytes, *, signing_key: bytes | None = None) -> object:
    """Validate and deserialize one trusted callable result artifact."""

    if not artifact.startswith(_RESULT_MAGIC):
        raise ValueError("invalid callable result artifact")
    try:
        encoded_header, payload = artifact[len(_RESULT_MAGIC) :].split(b"\n", 1)
        header = json.loads(encoded_header)
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid callable result artifact") from exc
    if not isinstance(header, dict) or header.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported callable result format {header.get('format_version')!r}")
    current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if header.get("python") != current_python:
        raise ValueError(f"callable result requires Python {header.get('python')}; current runtime is {current_python}")
    digest = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(digest, str(header.get("sha256") or "")):
        raise ValueError("callable result artifact digest mismatch")
    if signing_key is not None:
        expected = hmac.new(signing_key, payload, hashlib.sha256).hexdigest()
        signature = header.get("signature")
        if not isinstance(signature, str) or not hmac.compare_digest(expected, signature):
            raise ValueError("callable result artifact signature mismatch")
    return cloudpickle.loads(payload)


def _local_path(uri: str) -> Path:
    """Return a local path for the local-artifact fallback."""

    if not uri.startswith("file://"):
        return Path(uri).expanduser()
    parsed = urlsplit(uri)
    if parsed.netloc not in {"", "localhost"}:
        raise ValueError("local callable artifact URIs cannot name a remote host")
    return Path(unquote(parsed.path))


def _gcs_location(uri: str) -> tuple[str, str]:
    """Parse one complete ``gs://bucket/object`` URI."""

    parsed = urlsplit(uri)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("GCS callable artifacts must use a complete gs://bucket/object URI") from exc
    object_name = unquote(parsed.path.lstrip("/"))
    if (
        parsed.scheme != "gs"
        or not parsed.netloc
        or not object_name
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
        or port is not None
    ):
        raise ValueError("GCS callable artifacts must use a complete gs://bucket/object URI")
    return parsed.netloc, object_name


def _request_bytes(request: str | urllib.request.Request) -> bytes:
    """Read an HTTP artifact with bounded retries."""

    for attempt in range(HTTP_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=60, **control_plane_urlopen_options()) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRYABLE_HTTP_STATUS or attempt == HTTP_ATTEMPTS - 1:
                raise
            delay = retry_delay(exc, attempt)
            exc.close()
        except (urllib.error.URLError, http.client.IncompleteRead, TimeoutError, ConnectionError):
            if attempt == HTTP_ATTEMPTS - 1:
                raise
            delay = retry_delay(None, attempt)
        time.sleep(delay)
    raise AssertionError("unreachable")


class UriArtifactTransfer:
    """Shared local, HTTP, and GCS URI transfer behavior for callable workers."""

    def __init__(self, gcs: GcsObjectTransfer, environment: tp.Mapping[str, str]) -> None:
        self._gcs = gcs
        self._environment = environment

    def read(self, uri: str, *, token: str | None = None) -> bytes:
        """Read a callable payload from the requested URI."""

        if uri.startswith("gs://"):
            bucket_name, object_name = _gcs_location(uri)
            return self._gcs.read_object(bucket_name, object_name)
        if uri.startswith(("https://", "http://")):
            self._validate_http(uri, "callable artifacts")
            request: str | urllib.request.Request = uri
            if token:
                request = urllib.request.Request(uri, headers={"X-Eyvan-Artifact-Token": token})
            return _request_bytes(request)
        return _local_path(uri).read_bytes()

    def write(self, uri: str, payload: bytes, *, token: str | None = None) -> None:
        """Persist a callable result through the requested URI."""

        if uri.startswith("gs://"):
            bucket_name, object_name = _gcs_location(uri)
            self._gcs.write_object(bucket_name, object_name, payload)
            return
        if uri.startswith(("https://", "http://")):
            self._validate_http(uri, "callable results")
            headers = {"Content-Type": "application/octet-stream"}
            if token:
                headers["X-Eyvan-Artifact-Token"] = token
            request = urllib.request.Request(uri, data=payload, headers=headers, method="PUT")
            _request_bytes(request)
            return
        path = _local_path(uri)
        # Inlined rather than imported: this module is copied into the OCI
        # callable runner, which must not import eyvan. mkdir(parents=True)
        # applies its mode to the leaf only, so each level is made explicitly.
        _missing: list[Path] = []
        _current = path.parent
        while not _current.exists():
            _missing.append(_current)
            if _current.parent == _current:
                break
            _current = _current.parent
        for _directory in reversed(_missing):
            _directory.mkdir(mode=0o700, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(payload)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, path)
        except BaseException:
            # The temp file is created with delete=False; a failed write must
            # not leave it behind, because nothing prunes this directory.
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise

    def _validate_http(self, uri: str, subject: str) -> None:
        if uri.startswith("http://") and self._environment.get("EYVAN_ALLOW_INSECURE_ARTIFACT_HTTP", "").lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            raise ValueError(f"HTTP {subject} require EYVAN_ALLOW_INSECURE_ARTIFACT_HTTP=1")


async def _resolve_awaitable(awaitable: tp.Awaitable[object]) -> object:
    """Bridge an arbitrary awaitable through ``asyncio.run`` with exact typing."""

    return await awaitable


def _call_arguments(environment: tp.Mapping[str, str]) -> tuple[list[JsonValue], dict[str, JsonValue]]:
    """Decode and validate the controller's bound callable invocation arguments."""

    raw_args = tp.cast(JsonValue, json.loads(environment.get("EYVAN_CALL_ARGS_JSON", "[]")))
    raw_kwargs = tp.cast(JsonValue, json.loads(environment.get("EYVAN_CALL_KWARGS_JSON", "{}")))
    if not isinstance(raw_args, list):
        raise ValueError("EYVAN_CALL_ARGS_JSON must decode to a JSON array")
    if not isinstance(raw_kwargs, dict) or any(not isinstance(name, str) for name in raw_kwargs):
        raise ValueError("EYVAN_CALL_KWARGS_JSON must decode to a JSON object with string keys")
    return tp.cast(list[JsonValue], raw_args), tp.cast(dict[str, JsonValue], raw_kwargs)


def _workspace_extractor() -> tp.Callable[..., int]:
    """Return ``extract_workspace`` from whichever context this module runs in.

    This file is executed two ways: as ``eyvan.worker.runtime.callable_protocol`` on a
    native worker, and as a bare sibling module staged next to
    ``workspace.py`` inside an OCI image. Importing the hardened extractor
    rather than reimplementing it keeps exactly one copy of the tar-slip
    defence.
    """

    try:
        from eyvan._internal.workspace import extract_workspace
    except ImportError:  # staged OCI layout: siblings on sys.path, no package
        from workspace import extract_workspace  # type: ignore[no-redef]
    return extract_workspace


_WORKSPACE_BASE_ANCHOR: Path | None = None


def _materialize_workspace(
    payload: bytes,
    *,
    sha256: str,
    signature: str | None,
    signing_key: bytes | None,
    environment: tp.Mapping[str, str],
) -> Path:
    """Verify, unpack, and enter a workspace bundle before the callable loads.

    Both checks run *before* extraction, and extraction runs before the
    workspace is put on ``sys.path``. The ordering is the point: this directory
    becomes ``sys.path[0]`` and the process cwd, so anything it contains can
    shadow a module the signed callable resolves by reference. Verifying the
    bundle after unpacking it would already be too late.
    """

    digest = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(digest, sha256):
        raise ValueError("workspace bundle does not match its content-addressed digest")
    if signing_key is not None:
        expected = hmac.new(signing_key, payload, hashlib.sha256).hexdigest()
        if signature is None or not hmac.compare_digest(expected, signature):
            # Without this, signing the callable is decorative whenever a
            # workspace is attached: cloudpickle resolves imports by reference,
            # so an unsigned bundle chooses what those references find.
            raise ValueError("workspace bundle signature mismatch")
    # The anchor survives os.chdir below: an in-process retry must land in the
    # same base, not nest a workspace inside the previous one.
    global _WORKSPACE_BASE_ANCHOR
    explicit = environment.get("EYVAN_WORKSPACE_DIR")
    if explicit:
        base = Path(explicit)
    else:
        # An explicit directory always wins; the implicit one is anchored so
        # an in-process retry lands beside the first extraction instead of
        # nesting inside it.
        if _WORKSPACE_BASE_ANCHOR is None or not _WORKSPACE_BASE_ANCHOR.is_dir():
            _WORKSPACE_BASE_ANCHOR = Path.cwd() / ".eyvan-workspace"
        base = _WORKSPACE_BASE_ANCHOR
    # Archive bytes can be cached by digest, but execution trees are writable:
    # an earlier attempt may have changed or removed submitted files. Extract
    # verified bytes afresh into a private, unique tree for each invocation.
    # Concurrent invocations never write, reuse, or remove each other's tree.
    _ensure_private_dir(base)
    destination = Path(tempfile.mkdtemp(dir=base, prefix=f"{sha256[:12]}-"))
    try:
        _workspace_extractor()(payload, destination)
        (destination / ".eyvan-extraction-complete").write_text(sha256, encoding="utf-8")
    except BaseException:
        # Only this invocation owns this incomplete tree; it has not been
        # entered or added to sys.path yet, so removing it is safe.
        _remove_tree(destination)
        raise
    # Completed trees may still be cwd/sys.path of live workloads or children.
    # Without an execution ownership ledger, never evict a sibling here.
    # Enter the workspace so relative paths behave as they did for the caller,
    # and make its modules importable by the unpickling that follows.
    os.chdir(destination)
    sys.path.insert(0, str(destination))
    return destination


#: Content-addressed read cache for callable and workspace bytes. The digest
#: IS the validity check -- a file named by sha256 can only be served if it
#: was written after a byte-for-byte match -- so caching needs no invalidation
#: logic, and a preemption retry skips the re-download Ray pays every time.
_CACHE_MAX_FILES = 128


def _cache_root(environment: tp.Mapping[str, str]) -> Path | None:
    """Resolve the cache directory, or ``None`` when caching is unavailable.

    Caching opts in through the native worker's existing state directory: the
    worker process always knows one, while ad-hoc invocations (and tests that
    count fetches against a fake origin) stay exactly as they were.
    """

    base = environment.get("EYVAN_NATIVE_WORKER_STATE_DIR")
    if not base:
        return None
    root = Path(base) / "callable-cache"
    try:
        _ensure_private_dir(root)
        return root
    except OSError:
        return None


def _remove_tree(root: Path) -> None:
    """Best-effort recursive removal of an invocation's own incomplete tree."""

    import shutil

    shutil.rmtree(root, ignore_errors=True)


def _ensure_private_dir(root: Path) -> None:
    # Inlined rather than imported: this module is copied into the OCI
    # callable runner, which must not import eyvan.
    missing: list[Path] = []
    cursor = root
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    for directory in reversed(missing):
        # Another invocation may create the shared base after our exists check.
        directory.mkdir(mode=0o700, exist_ok=True)
        existing_mode = directory.stat().st_mode & 0o777
        if existing_mode != 0o700:
            directory.chmod(0o700)


def _prune_cache(root: Path) -> None:
    """Keep the cache bounded; oldest entries go first."""

    entries = sorted(
        (path for path in root.iterdir() if path.is_file() and path.suffix == ".pkl"),
        key=lambda path: path.stat().st_mtime,
    )
    for path in entries[: max(len(entries) - _CACHE_MAX_FILES, 0)]:
        path.unlink(missing_ok=True)


def _read_cached(
    transfer: ArtifactTransfer,
    *,
    uri: str,
    token: str | None,
    expected_sha256: str,
    cache_root: Path | None,
) -> bytes:
    """Serve content-addressed bytes from the cell's cache, filling on miss."""

    if not expected_sha256 or cache_root is None:
        return transfer.read(uri, token=token)
    candidate = cache_root / f"{expected_sha256}.pkl"
    try:
        return candidate.read_bytes()
    except OSError:
        pass
    payload = transfer.read(uri, token=token)
    if hmac.compare_digest(hashlib.sha256(payload).hexdigest(), expected_sha256):
        try:
            temporary = candidate.with_name(f"{candidate.name}.{os.getpid()}.tmp")
            temporary.write_bytes(payload)
            os.replace(temporary, candidate)
            _prune_cache(cache_root)
        except OSError:
            # Pruning counts only .pkl entries, so remove partial fills when
            # possible without turning an optional cache failure into a job failure.
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return payload


def execute_callable(
    argv: tp.Sequence[str] | None,
    *,
    transfer: ArtifactTransfer,
    environment: tp.Mapping[str, str],
) -> int:
    """Run one signed callable artifact through a concrete URI transfer adapter."""

    parser = argparse.ArgumentParser(description="Execute a trusted Eyvan callable artifact")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--uri")
    source.add_argument("--uri-env")
    parser.add_argument("--uri-token-env")
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--format-version", type=int, default=FORMAT_VERSION)
    parser.add_argument("--signature")
    result_target = parser.add_mutually_exclusive_group()
    result_target.add_argument("--result-uri")
    result_target.add_argument("--result-uri-env")
    parser.add_argument("--result-token-env")
    workspace_target = parser.add_mutually_exclusive_group()
    workspace_target.add_argument("--workspace-uri")
    workspace_target.add_argument("--workspace-uri-env")
    parser.add_argument("--workspace-token-env")
    parser.add_argument("--workspace-sha256")
    parser.add_argument("--workspace-signature")
    args = parser.parse_args(None if argv is None else list(argv))

    signing_key_value = environment.get("EYVAN_CALLABLE_SIGNING_KEY")
    signing_key = signing_key_value.encode() if signing_key_value else None
    artifact_uri = args.uri or environment.get(args.uri_env or "", "")
    if not artifact_uri:
        raise ValueError("callable artifact URI environment variable is empty")
    artifact_token = environment.get(args.uri_token_env or "") or None
    result_uri = args.result_uri or environment.get(args.result_uri_env or "") or None
    result_token = environment.get(args.result_token_env or "") or None

    workspace_uri = args.workspace_uri or environment.get(args.workspace_uri_env or "") or None
    if workspace_uri:
        if not args.workspace_sha256:
            # Not optional: the bundle is content-addressed storage's only
            # integrity claim, and it decides what the callable imports.
            raise ValueError("--workspace-sha256 is required when a workspace bundle is attached")
        # The workspace lands before the callable is unpickled: unpickling may
        # import modules that only exist inside the shipped working directory.
        _materialize_workspace(
            _read_cached(
                transfer,
                uri=workspace_uri,
                token=environment.get(args.workspace_token_env or "") or None,
                expected_sha256=args.workspace_sha256,
                cache_root=_cache_root(environment),
            ),
            sha256=args.workspace_sha256,
            signature=args.workspace_signature,
            signing_key=signing_key,
            environment=environment,
        )

    fn = load_callable(
        CallableArtifact(
            payload=_read_cached(
                transfer,
                uri=artifact_uri,
                token=artifact_token,
                expected_sha256=args.sha256,
                cache_root=_cache_root(environment),
            ),
            sha256=args.sha256,
            python=args.python,
            format_version=args.format_version,
            signature=args.signature,
        ),
        signing_key=signing_key,
    )
    call_args, call_kwargs = _call_arguments(environment)
    result = fn(*call_args, **call_kwargs)
    if inspect.isawaitable(result):
        result = asyncio.run(_resolve_awaitable(tp.cast(tp.Awaitable[object], result)))
    if result_uri and ReplicaContext.from_environment(environment).owns_result:
        transfer.write(result_uri, dump_result(result, signing_key=signing_key), token=result_token)
    elif result is not None:
        print(json.dumps(result, default=repr))
    return 0


__all__ = (
    "FORMAT_VERSION",
    "ArtifactTransfer",
    "CallableArtifact",
    "GcsObjectTransfer",
    "ReplicaContext",
    "UriArtifactTransfer",
    "dump_callable",
    "dump_result",
    "execute_callable",
    "load_callable",
    "load_result",
)
