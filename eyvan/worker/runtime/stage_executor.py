# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Put a named asset on this host's disk, prove what landed, and take it off.

The control plane already owned every decision about staging -- the state
machine, the byte budget, the eviction policy, the fence and the durable ledger
-- and owned no way to move a byte. ``CellStageExecutor`` was an injected
protocol with no implementation, so every stage read as absent and placement
never changed. This module is the half that touches a disk.

**Deployment order: native workers first, then agents, then the controller.**
Staging reaches a disk over two hops, and each is a wire method an older process
does not serve. The agent serves
``POST /api/v1/cells/{cell_id}/stage-actions``; a native worker serves the same
path for the cells whose disk it owns. Either one, when it predates its route,
answers HTTP 404, which the caller reports as *this host has no staging
executor* -- absence of a capability, not a failed action. Nothing is recorded,
nothing is charged and nothing is retried, which is exactly the behaviour of a
deployment that configures no executor at all. Every prefix of the order above
is therefore inert rather than broken, and the feature appears from the disk
upwards.

Which hop an agent uses is decided by whose disk the workload runs on.
``native-local`` runs it as a child of the agent process, so the agent holds the
executor itself. ``eyvan-native`` runs it on another machine, so the agent
fences the request against its own durable record and forwards it to that
machine's worker, which holds the executor. The class below is the same class in
both cases: there is one implementation of fetch, measure and evict, and the
second deployment shape is a route to it rather than a copy of it.

Four rules decide every answer here, and each exists because its opposite is a
lie the control plane would then act on:

**Bounded.** ``max_bytes`` is a term of the request, never a hint. Nothing is
fetched whose declared length exceeds it, and a source that streams past it mid
transfer is refused rather than truncated -- an executor that overruns turns the
cell budget into a number that bounds nothing.

**Resumable, and a partial fetch is inconclusive.** Bytes land in a ``.part``
file that survives a failure, so the next attempt resumes at the offset the last
one reached. The published path appears only after the whole declared length
arrived and verified, so a truncated transfer is never mistaken for a copy.

**Observed, never expected.** The size and digest reported by :meth:`stage` are
read back off this host's disk. The digest an asset *declares* is only ever used
to reject bytes, never to describe them.

**A digest mismatch quarantines.** Bytes that do not hash to what the asset
declared are moved aside and refused permanently. Serving them would hand a
workload silently wrong data, and deleting them would destroy the evidence of
which copy was wrong.

Sources are an allowlist an operator declares, and it is empty by default.
``WorkloadDefinition.named_assets`` is *user* input, so an executor that fetched
whatever it was handed would read any local file the agent can read and issue
any HTTP request the agent's network permits -- including one to a cloud
metadata service. A fleet that has not declared a prefix stages nothing, and
says so as a refusal rather than by pretending an asset does not exist.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import re
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

from eyvan._internal.compat import StrEnum
from eyvan._internal.workspace import create_private_directory

logger = logging.getLogger(__name__)

#: Environment variable naming the directory staged assets are written under.
#:
#: Read here rather than through :class:`~eyvan.config.EyvanSettings` for the
#: same reason the native worker reads its workload directory here: this runs on
#: the execution host, where the control plane's settings object is not the
#: configuration in force.
STAGE_DIRECTORY_ENVIRONMENT = "EYVAN_CELL_STAGE_DIR"

#: Environment variable listing the asset prefixes this host may fetch.
#:
#: Whitespace- or comma-separated. Empty -- the default -- stages nothing.
STAGE_SOURCES_ENVIRONMENT = "EYVAN_CELL_STAGE_SOURCES"

#: Bytes read or written per chunk. Large enough that a multi-gigabyte transfer
#: is not dominated by syscalls, small enough that the byte bound is enforced
#: within a megabyte of where it was crossed.
CHUNK_BYTES = 1 << 20

_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_INTEGRITY_FRAGMENT = re.compile(r"^sha256[=:]([0-9a-f]{64})$")
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class AssetStageVerdict(StrEnum):
    """What one staging action established, in the only three shapes there are.

    The names match the control plane's own taxonomy on purpose: ``REFUSED`` is
    terminal and becomes a blocked stage, ``INCONCLUSIVE`` is transient and
    becomes an unknown one, and there is deliberately no fourth verdict meaning
    "probably fine".
    """

    #: The action completed and the reported bytes are a fact about this disk.
    OK = "ok"
    #: This request can never succeed as written. Retrying it unchanged is the
    #: unbounded loop the control plane's blocked state exists to end.
    REFUSED = "refused"
    #: Nothing was established. The bytes may or may not be there.
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class AssetStageResult:
    """One executor answer: a verdict, and the bytes it can prove.

    ``size_bytes`` and ``digest`` are populated only for :attr:`
    AssetStageVerdict.OK`. A refused or inconclusive action proved nothing about
    the disk, and filling these in with what was *expected* is precisely the
    invented evidence this design refuses.
    """

    verdict: AssetStageVerdict
    size_bytes: int = 0
    #: ``sha256:<hex>`` computed over the bytes on this host, or ``None`` when
    #: the action did not read them.
    digest: str | None = None
    detail: str = ""
    #: Where the asset lives on this host, when it is present.
    path: str | None = None


@dataclass(frozen=True, slots=True)
class AssetReference:
    """A parsed asset address: where the bytes are, and what they must hash to.

    An asset may carry an integrity fragment -- ``https://host/x.tar#sha256=<hex>``
    -- and when it does, bytes that hash to anything else are quarantined rather
    than published. A reference without one is still staged; its content is then
    unverified, which is reported as an observed digest and never as a claim
    that the bytes are the right ones.
    """

    raw: str
    scheme: str
    location: str
    expected_digest: str | None = None


class AssetRefusedError(RuntimeError):
    """This source can never satisfy this request as written."""


class AssetTransferError(RuntimeError):
    """The transfer did not complete and nothing was established by it."""


@dataclass(frozen=True, slots=True)
class AssetStream:
    """One resumable read, and the offset it actually starts at.

    ``offset`` is reported rather than assumed: a server may ignore a range
    request and send the whole body, and a caller that trusted its own request
    would splice those bytes onto the middle of a partial file.
    """

    offset: int
    chunks: tp.Iterator[bytes]


class AssetSource(tp.Protocol):
    """Where one scheme's bytes come from, and how long they are.

    Both methods are blocking. Neither is called while the control plane holds a
    transaction: the agent runs them on its own host, on a request thread.
    """

    def size(self, reference: AssetReference) -> int:
        """Return the asset's declared length in bytes.

        Raises:
            AssetRefusedError: The asset cannot be fetched as named.
            AssetTransferError: The length could not be established.
        """
        ...

    def stream(self, reference: AssetReference, *, offset: int) -> tp.ContextManager[AssetStream]:
        """Open the asset's bytes, resuming at ``offset`` where supported.

        Raises:
            AssetRefusedError: The asset cannot be fetched as named.
            AssetTransferError: The read could not be started.
        """
        ...


class FileAssetSource:
    """Read an asset from a filesystem this host already mounts.

    Only regular files, and never through a symbolic link. A directory is
    refused rather than walked: a tree has no single length to bound the fetch
    by and no single digest to verify it with, so staging one would give up both
    of this module's guarantees at once.

    The symlink rule is a privilege boundary, not tidiness. The allowlist is a
    prefix on a *name*, and a workload that already runs on this host can drop
    ``/data/public/x -> /etc/shadow`` inside a declared prefix and then name it.
    The agent may run as root; the workload does not. Following that link copies
    a file the workload could not read into a directory it can.
    """

    def _path(self, reference: AssetReference) -> Path:
        path = Path(reference.location)
        if not path.is_absolute():
            raise AssetRefusedError(f"asset {reference.raw!r} must name an absolute path")
        # The final component is not the only one a hostile name can aim
        # through. A link dropped on an *intermediate* directory inside the
        # declared prefix -- ``d -> /etc``, then name ``d/shadow`` -- redirects
        # the whole lookup, and the stat()/open() below follow it; the final
        # component would hash as an ordinary file. So every component is
        # checked, which is the allowlist-as-name-boundary rule applied all the
        # way down instead of only at its last hop.
        walked = Path(path.anchor)
        for part in path.parts[1:]:
            walked = walked / part
            if walked.is_symlink():
                raise AssetRefusedError(
                    f"asset {reference.raw!r} is reached through the symbolic link {walked}; "
                    "only real files can be staged"
                )
        return path

    def size(self, reference: AssetReference) -> int:
        """Return the file's length, refusing anything that is not a regular file."""

        path = self._path(reference)
        try:
            status = path.stat()
        except FileNotFoundError as exc:
            raise AssetRefusedError(f"asset {reference.raw!r} does not exist on this host") from exc
        except OSError as exc:
            raise AssetTransferError(f"could not read asset {reference.raw!r}: {exc}") from exc
        if not path.is_file():
            raise AssetRefusedError(f"asset {reference.raw!r} is not a regular file; only files can be staged")
        return status.st_size

    @contextlib.contextmanager
    def stream(self, reference: AssetReference, *, offset: int) -> tp.Iterator[AssetStream]:
        """Open the file at ``offset``, which a local read always honours."""

        path = self._path(reference)
        try:
            handle = path.open("rb")
            handle.seek(offset)
        except FileNotFoundError as exc:
            raise AssetRefusedError(f"asset {reference.raw!r} does not exist on this host") from exc
        except OSError as exc:
            raise AssetTransferError(f"could not open asset {reference.raw!r}: {exc}") from exc
        # No try/except around the yield: an error raised by the *caller* while
        # it writes is a failure of this host's disk, not of the read, and
        # relabelling it here would report the wrong side as unreachable.
        try:
            yield AssetStream(offset=offset, chunks=iter(lambda: handle.read(CHUNK_BYTES), b""))
        finally:
            handle.close()


class HttpAssetSource:
    """Fetch an asset over HTTP, resuming with a byte range where offered.

    A source that does not declare ``Content-Length`` is refused: without a
    length there is nothing to bound the transfer by and nothing to compare the
    written bytes against, so a truncated body would be indistinguishable from a
    complete one and would be published as a confirmed stage.
    """

    def __init__(self, *, timeout_s: float = 300.0, client: tp.Any = None) -> None:
        self._timeout_s = timeout_s
        self._client = client

    def _transport(self) -> tp.Any:
        if self._client is not None:
            return contextlib.nullcontext(self._client)
        import httpx

        return httpx.Client(timeout=self._timeout_s, follow_redirects=True)

    @staticmethod
    def _refuse(status_code: int, reference: AssetReference) -> None:
        """Raise the failure a status code actually establishes.

        A 4xx is the server stating this request is wrong, which retrying
        unchanged cannot fix. Everything else -- 5xx, a proxy error, a timeout --
        says nothing about the asset, and recording it as a verdict about the
        asset is how a transient outage becomes a permanently blocked stage.
        """

        if 400 <= status_code < 500:
            raise AssetRefusedError(f"asset {reference.raw!r} was rejected with HTTP {status_code}")
        raise AssetTransferError(f"asset {reference.raw!r} could not be read: HTTP {status_code}")

    def size(self, reference: AssetReference) -> int:
        """Return the declared ``Content-Length``, refusing a source without one."""

        import httpx

        try:
            with self._transport() as client:
                response = client.request("HEAD", reference.location)
        except httpx.HTTPError as exc:
            raise AssetTransferError(f"asset {reference.raw!r} could not be reached: {exc}") from exc
        if response.status_code >= 400:
            self._refuse(response.status_code, reference)
        declared = response.headers.get("content-length")
        if declared is None or not declared.strip().isdigit():
            raise AssetRefusedError(
                f"asset {reference.raw!r} declares no content length, so the fetch cannot be bounded"
            )
        return int(declared.strip())

    @contextlib.contextmanager
    def stream(self, reference: AssetReference, *, offset: int) -> tp.Iterator[AssetStream]:
        """Open a ranged read, reporting the offset the server actually honoured."""

        import httpx

        headers = {"Range": f"bytes={offset}-"} if offset > 0 else {}
        try:
            with self._transport() as client, client.stream("GET", reference.location, headers=headers) as response:
                if response.status_code >= 400:
                    self._refuse(response.status_code, reference)
                # 206 means the range was honoured. Anything else is the whole
                # body, and splicing that onto a partial file would corrupt it.
                started = offset if response.status_code == 206 else 0
                yield AssetStream(offset=started, chunks=response.iter_bytes(CHUNK_BYTES))
        except httpx.HTTPError as exc:
            raise AssetTransferError(f"asset {reference.raw!r} transfer failed: {exc}") from exc


def parse_asset_reference(asset: str) -> AssetReference:
    """Parse one named asset into a scheme, a location and a declared digest.

    Raises:
        ValueError: The asset is empty, untrimmed, or carries a NUL byte.
    """

    if not isinstance(asset, str) or not asset.strip() or asset != asset.strip():
        raise ValueError("a named asset must be a non-empty trimmed string")
    if "\x00" in asset:
        raise ValueError("a named asset must not contain NUL bytes")
    location, _, fragment = asset.rpartition("#")
    integrity = _INTEGRITY_FRAGMENT.fullmatch(fragment) if location else None
    expected: str | None = None
    if integrity is not None:
        expected = f"sha256:{integrity.group(1)}"
    else:
        # Not an integrity fragment, so it is part of the name. A path may
        # legitimately contain '#', and guessing otherwise would silently fetch
        # a different object than the one the workload declared.
        location = asset
    split = urlsplit(location)
    scheme = split.scheme.lower()
    if scheme in {"", "file"}:
        path = unquote(split.path) if scheme == "file" else location
        return AssetReference(raw=asset, scheme="file", location=path, expected_digest=expected)
    return AssetReference(raw=asset, scheme=scheme, location=location, expected_digest=expected)


def default_asset_sources() -> dict[str, AssetSource]:
    """Return the schemes this host can fetch without extra credentials."""

    return {"file": FileAssetSource(), "http": HttpAssetSource(), "https": HttpAssetSource()}


def stage_sources_from_environment(value: str | None = None) -> tuple[str, ...]:
    """Return the asset prefixes an operator declared fetchable on this host.

    Empty is the default and means this host stages nothing. That is deliberate:
    named assets arrive from a user's workload spec, so an allowlist that
    defaults to "anything" is a request forgery primitive with a cache attached.
    """

    raw = os.getenv(STAGE_SOURCES_ENVIRONMENT, "") if value is None else value
    return tuple(sorted({item for item in re.split(r"[\s,]+", raw) if item}))


def _within_prefix(asset: str, prefix: str) -> bool:
    """Return whether one asset lies under a declared prefix, on a boundary.

    A bare ``startswith`` would admit ``/data/publicly-readable`` under a prefix
    of ``/data/public``, and ``https://cdn-evil/x`` under ``https://cdn``. The
    boundary is what makes an allowlist an allowlist rather than a string test.
    """

    if asset == prefix:
        return True
    return asset.startswith(prefix if prefix.endswith("/") else f"{prefix}/")


def _digest_file(path: Path) -> str:
    """Return ``sha256:<hex>`` over one file's bytes, read back off the disk.

    A second pass over the file rather than a hash accumulated during the
    transfer, because a resumed fetch never sees the bytes it did not download
    and a hasher cannot be restored from a previous process. The cost is one
    extra sequential read; the alternative is a digest that is only correct when
    nothing failed.
    """

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


def _asset_key(asset: str) -> str:
    """Return a collision-free, filesystem-safe name for one asset.

    Content-addressed by the asset *name*, with a readable tail so an operator
    reading ``ls`` output can tell which dataset a directory holds. The digest
    prefix is what makes it unique; the tail is only for humans.
    """

    fingerprint = hashlib.sha256(asset.encode("utf-8")).hexdigest()[:32]
    tail = _SAFE_NAME.sub("-", PurePosixPath(asset.split("#", 1)[0]).name)[-48:].strip("-")
    return f"{fingerprint}-{tail}" if tail else fingerprint


def _cell_directory(cell_id: str) -> str:
    """Return one cell's directory name, refusing anything that could escape it."""

    if not isinstance(cell_id, str) or not cell_id.strip() or cell_id != cell_id.strip():
        raise ValueError("a cell ID must be a non-empty trimmed string")
    if "\x00" in cell_id or "/" in cell_id or "\\" in cell_id or cell_id in {".", ".."}:
        raise ValueError("a cell ID must not contain path separators")
    return cell_id


@dataclass(frozen=True, slots=True)
class _AssetPaths:
    """Where one asset's published, in-flight and quarantined copies live."""

    published: Path
    partial: Path
    quarantined: Path


@tp.runtime_checkable
class AssetStageDriver(tp.Protocol):
    """The host-side half of managed staging, as anything forwarding to it sees it.

    :class:`LocalAssetStageExecutor` satisfies it by running on the disk, and
    :class:`~eyvan.worker.runtime.native_remote.NativeRemoteExecutor` satisfies
    it by forwarding to the worker that does. Runtime-checkable so a caller can
    ask a resolved execution driver whether it stages at all, the way
    ``runtime/registry`` already asks about structured logs and process
    introspection -- an execution backend that cannot stage says so by not
    satisfying this, rather than by returning something that looks like a
    result.
    """

    def measure(self, cell_id: str, asset: str, *, max_bytes: int) -> AssetStageResult: ...

    def stage(self, cell_id: str, asset: str, *, max_bytes: int) -> AssetStageResult: ...

    def evict(self, cell_id: str, asset: str) -> AssetStageResult: ...


@dataclass(slots=True)
class LocalAssetStageExecutor:
    """Materialize, measure and remove named assets on the disk it runs on.

    Constructed with the directory it owns and the prefixes it may fetch from.
    It owns nothing outside that directory and reads nothing outside those
    prefixes, so the blast radius of a hostile asset name is a refusal.

    Deliberately not the cell-agent client and deliberately not the control
    plane's ``CellStageExecutor``: this is the host-side half, and the wire
    between them is one RPC. Keeping them apart is what lets the same class
    serve an agent that runs workloads as its own children and a native worker
    reached over HTTP, without either one importing the control plane. Both
    deployment shapes construct this class; neither reimplements it.
    """

    root: Path
    #: Asset prefixes this host may fetch. Empty stages nothing.
    allowed_prefixes: tuple[str, ...] = ()
    sources: dict[str, AssetSource] = field(default_factory=default_asset_sources)

    def __post_init__(self) -> None:
        """Reject a root that is not an absolute directory this host owns."""

        self.root = Path(self.root).expanduser()
        if not self.root.is_absolute():
            raise ValueError("a cell stage root must be an absolute path")
        self.allowed_prefixes = tuple(sorted({item for item in self.allowed_prefixes if item}))

    @classmethod
    def from_environment(cls, default_root: str | os.PathLike[str]) -> LocalAssetStageExecutor:
        """Build the executor an agent runs with, from this host's environment.

        The root is resolved rather than required to be absolute. An agent whose
        state directory is relative is a supported configuration everywhere else
        in this package, and refusing one here would turn managed staging into a
        reason the agent will not start at all.
        """

        root = os.getenv(STAGE_DIRECTORY_ENVIRONMENT) or str(default_root)
        return cls(root=Path(root).expanduser().resolve(), allowed_prefixes=stage_sources_from_environment())

    def _paths(self, cell_id: str, asset: str) -> _AssetPaths:
        cell_root = self.root / _cell_directory(cell_id)
        key = _asset_key(asset)
        return _AssetPaths(
            published=cell_root / "assets" / key,
            partial=cell_root / "partial" / f"{key}.part",
            quarantined=cell_root / "quarantine" / key,
        )

    def _reference(self, asset: str) -> AssetReference:
        """Parse and admit one asset, or say exactly why it will never be fetched."""

        reference = parse_asset_reference(asset)
        if not any(_within_prefix(asset, prefix) for prefix in self.allowed_prefixes):
            raise AssetRefusedError(
                f"asset {asset!r} matches no declared stage source on this host; "
                f"set {STAGE_SOURCES_ENVIRONMENT} to the prefixes this fleet may fetch"
            )
        # Every scheme, not only ``file``. A declared prefix is a boundary, and
        # ``https://cdn/public/../private`` walks out of one exactly as
        # ``/data/public/../private`` does -- on the server's filesystem rather
        # than on this one, which makes it somebody else's incident.
        if ".." in PurePosixPath(reference.location).parts:
            raise AssetRefusedError(f"asset {asset!r} must not traverse out of its declared source")
        if reference.scheme not in self.sources:
            raise AssetRefusedError(f"this host cannot fetch {reference.scheme}:// assets")
        return reference

    def measure(self, cell_id: str, asset: str, *, max_bytes: int) -> AssetStageResult:
        """Return what this asset would occupy here, without fetching it.

        A copy already on this disk answers with its own length, which is an
        observation. Anything else answers with the length the source declares,
        which is an estimate the control plane reserves against and replaces
        with the observed length the moment :meth:`stage` confirms.

        ``max_bytes`` is reported back rather than enforced: measuring costs
        nothing, and the control plane's own budget arithmetic is the authority
        on whether a measured size may be admitted. Enforcing it twice, in two
        places that can disagree, is how a headroom check and a budget check
        come to name different numbers.
        """

        try:
            reference = self._reference(asset)
            paths = self._paths(cell_id, asset)
            if paths.published.is_file():
                return AssetStageResult(
                    verdict=AssetStageVerdict.OK,
                    size_bytes=paths.published.stat().st_size,
                    detail=f"already staged on this host; this cell may admit {max_bytes} bytes",
                    path=str(paths.published),
                )
            declared = self.sources[reference.scheme].size(reference)
            return AssetStageResult(
                verdict=AssetStageVerdict.OK,
                size_bytes=declared,
                detail=f"the source declares {declared} bytes; this cell may admit {max_bytes}",
            )
        except (AssetRefusedError, ValueError) as exc:
            return AssetStageResult(verdict=AssetStageVerdict.REFUSED, detail=str(exc))
        except (AssetTransferError, OSError) as exc:
            return AssetStageResult(verdict=AssetStageVerdict.INCONCLUSIVE, detail=str(exc))

    def stage(self, cell_id: str, asset: str, *, max_bytes: int) -> AssetStageResult:
        """Materialize the asset and report the bytes that are provably here."""

        try:
            reference = self._reference(asset)
            paths = self._paths(cell_id, asset)
            if paths.published.is_file():
                return self._verify(reference, paths, max_bytes=max_bytes, detail="already staged on this host")
            return self._fetch(reference, paths, max_bytes=max_bytes)
        except (AssetRefusedError, ValueError) as exc:
            return AssetStageResult(verdict=AssetStageVerdict.REFUSED, detail=str(exc))
        except (AssetTransferError, OSError) as exc:
            return AssetStageResult(verdict=AssetStageVerdict.INCONCLUSIVE, detail=str(exc))

    def evict(self, cell_id: str, asset: str) -> AssetStageResult:
        """Remove every copy of the asset, returning once the bytes are gone.

        Idempotent: an asset that is already absent is a successful eviction,
        because the caller's question is whether the bytes are on the disk and
        the answer is no. Quarantined and partial copies go too -- they are the
        same reserved bytes as far as the disk is concerned.
        """

        try:
            paths = self._paths(cell_id, asset)
        except ValueError as exc:
            return AssetStageResult(verdict=AssetStageVerdict.REFUSED, detail=str(exc))
        freed = 0
        try:
            for path in (paths.published, paths.partial, paths.quarantined):
                if not path.exists():
                    continue
                freed += path.stat().st_size
                path.unlink()
        except OSError as exc:
            return AssetStageResult(verdict=AssetStageVerdict.INCONCLUSIVE, detail=f"could not remove {asset!r}: {exc}")
        return AssetStageResult(
            verdict=AssetStageVerdict.OK,
            size_bytes=freed,
            detail=f"removed {freed} bytes",
        )

    def _verify(
        self,
        reference: AssetReference,
        paths: _AssetPaths,
        *,
        max_bytes: int,
        detail: str,
    ) -> AssetStageResult:
        """Read back a published copy and either confirm it or quarantine it."""

        observed = paths.published.stat().st_size
        if observed > max_bytes:
            return AssetStageResult(
                verdict=AssetStageVerdict.REFUSED,
                detail=f"asset holds {observed} bytes; this cell may admit {max_bytes}",
            )
        digest = _digest_file(paths.published)
        if reference.expected_digest is not None and digest != reference.expected_digest:
            return self._quarantine(reference, paths.published, paths, digest)
        return AssetStageResult(
            verdict=AssetStageVerdict.OK,
            size_bytes=observed,
            digest=digest,
            detail=detail,
            path=str(paths.published),
        )

    def _fetch(self, reference: AssetReference, paths: _AssetPaths, *, max_bytes: int) -> AssetStageResult:
        """Transfer the asset into place, bounded, resumable and verified."""

        declared = self.sources[reference.scheme].size(reference)
        if declared > max_bytes:
            return AssetStageResult(
                verdict=AssetStageVerdict.REFUSED,
                detail=f"asset needs {declared} bytes; this cell may admit {max_bytes}",
            )
        create_private_directory(paths.partial.parent)
        create_private_directory(paths.published.parent)
        resume = paths.partial.stat().st_size if paths.partial.is_file() else 0
        if resume > declared:
            # The source is shorter than what is already on disk, so it is not
            # the same object. Resuming would splice two different assets.
            resume = 0
        if resume == declared and declared:
            # A crash between the transfer and the publish left a complete
            # .part. Resuming it would ask the source for ``bytes=declared-``,
            # which a compliant server (GCS, S3) answers 416 -- a 4xx, and so a
            # refusal recorded against bytes that are already on this disk and
            # would verify. The transfer is done; read it back and publish it
            # below without contacting the source again.
            written = declared
        else:
            written = self._transfer(reference, paths.partial, offset=resume, max_bytes=max_bytes)
        if written > max_bytes:
            paths.partial.unlink(missing_ok=True)
            return AssetStageResult(
                verdict=AssetStageVerdict.REFUSED,
                detail=f"asset exceeded the {max_bytes} byte bound it was admitted under",
            )
        if written != declared:
            return AssetStageResult(
                verdict=AssetStageVerdict.INCONCLUSIVE,
                detail=f"transferred {written} of {declared} bytes; the partial copy is resumable",
            )
        digest = _digest_file(paths.partial)
        if reference.expected_digest is not None and digest != reference.expected_digest:
            return self._quarantine(reference, paths.partial, paths, digest)
        os.replace(paths.partial, paths.published)
        return AssetStageResult(
            verdict=AssetStageVerdict.OK,
            size_bytes=written,
            digest=digest,
            detail="staged",
            path=str(paths.published),
        )

    def _transfer(self, reference: AssetReference, partial: Path, *, offset: int, max_bytes: int) -> int:
        """Copy bytes into the partial file, stopping one chunk past the bound.

        Returns the number of bytes on disk afterwards, which is the resume
        point for the next attempt. Overrunning by at most one chunk is
        deliberate: it is how the caller can tell "this asset is larger than its
        bound" from "this transfer stopped early", and the overrun is discarded
        rather than published.
        """

        with self.sources[reference.scheme].stream(reference, offset=offset) as stream:
            if stream.offset and not partial.is_file():
                # The source resumed a transfer whose earlier bytes are not on
                # this disk. Writing from its offset would leave a hole and
                # publish a file of the right length and the wrong content.
                raise AssetTransferError("the source resumed a transfer this host holds no partial copy of")
            written = stream.offset
            mode = "r+b" if stream.offset else "wb"
            with partial.open(mode) as handle:
                handle.seek(stream.offset)
                handle.truncate(stream.offset)
                try:
                    for chunk in stream.chunks:
                        handle.write(chunk)
                        written += len(chunk)
                        if written > max_bytes:
                            break
                finally:
                    # Flushed before any failure propagates, so the resume point
                    # is durable rather than whatever the buffer happened to
                    # hold when the connection dropped.
                    handle.flush()
                    os.fsync(handle.fileno())
        return written

    def _quarantine(
        self,
        reference: AssetReference,
        source: Path,
        paths: _AssetPaths,
        digest: str,
    ) -> AssetStageResult:
        """Move bytes that failed their declared digest aside, and refuse them.

        Never deleted here and never published. Deleting destroys the evidence
        of which copy was wrong, and publishing hands a workload silently
        corrupt data -- the one failure this whole verification exists to catch.
        Exactly one quarantined copy per asset is kept, so a source that is
        permanently wrong cannot fill the disk by being retried.
        """

        create_private_directory(paths.quarantined.parent)
        os.replace(source, paths.quarantined)
        logger.warning(
            "quarantined asset %s: declared %s, observed %s; kept at %s",
            reference.raw,
            reference.expected_digest,
            digest,
            paths.quarantined,
        )
        return AssetStageResult(
            verdict=AssetStageVerdict.REFUSED,
            detail=(
                f"asset declared {reference.expected_digest} and observed {digest}; "
                f"the bytes were quarantined at {paths.quarantined} and not served"
            ),
        )


__all__ = (
    "CHUNK_BYTES",
    "STAGE_DIRECTORY_ENVIRONMENT",
    "STAGE_SOURCES_ENVIRONMENT",
    "AssetReference",
    "AssetRefusedError",
    "AssetSource",
    "AssetStageDriver",
    "AssetStageResult",
    "AssetStageVerdict",
    "AssetStream",
    "AssetTransferError",
    "FileAssetSource",
    "HttpAssetSource",
    "LocalAssetStageExecutor",
    "default_asset_sources",
    "parse_asset_reference",
    "stage_sources_from_environment",
)
