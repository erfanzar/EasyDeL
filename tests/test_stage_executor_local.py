# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The half of managed staging that touches a disk.

Everything above this was decidable without a fleet -- what is provably
present, what the budget allows, what may be deleted -- and every one of those
decisions was made against a protocol nothing implemented. These tests are about
the bytes.

Four properties, and each one is a lie the control plane would otherwise act on:

**Bounded.** A source larger than the bound it was admitted under is refused
rather than truncated. An executor that overruns turns the cell budget into a
number that bounds nothing, and the ledger cannot tell the difference.

**A partial fetch is inconclusive, never present.** The published path appears
only after the whole declared length arrived, so a dropped connection leaves a
resumable ``.part`` file and an unproven row -- not a confirmed stage of half a
dataset that placement will then steer jobs toward.

**Observed, never expected.** The size and digest reported are read back off
this disk. The digest an asset declares is only ever used to reject bytes.

**A mismatch quarantines.** Bytes that hash to the wrong thing are moved aside
and refused permanently: serving them hands a workload silently wrong data, and
deleting them destroys the evidence of which copy was wrong.
"""

from __future__ import annotations

import hashlib
import typing as tp
from pathlib import Path

import httpx
import pytest

from eyvan.worker.runtime.stage_executor import (
    STAGE_SOURCES_ENVIRONMENT,
    AssetStageVerdict,
    AssetStream,
    HttpAssetSource,
    LocalAssetStageExecutor,
    parse_asset_reference,
    stage_sources_from_environment,
)

_CELL = "cell-a"


def _digest(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _source_file(tmp_path: Path, payload: bytes, name: str = "dataset.bin") -> Path:
    path = tmp_path / "source" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _executor(tmp_path: Path, *, allowed: tuple[str, ...] | None = None, sources: dict | None = None):
    return LocalAssetStageExecutor(
        root=tmp_path / "stage",
        allowed_prefixes=allowed if allowed is not None else (str(tmp_path / "source"),),
        **({"sources": sources} if sources is not None else {}),
    )


class _TruncatingSource:
    """A source that declares one length and delivers less of it, on demand.

    Models the failure that matters: a transfer that stops mid-stream without
    an error anyone can see, which is what a dropped connection looks like from
    the receiving end after the bytes stop arriving.
    """

    def __init__(self, payload: bytes, *, deliver: int) -> None:
        self.payload = payload
        self.deliver = deliver
        self.offsets: list[int] = []

    def size(self, _reference) -> int:
        return len(self.payload)

    def stream(self, _reference, *, offset: int):
        self.offsets.append(offset)
        chunk = self.payload[offset : offset + self.deliver]
        return _stream(AssetStream(offset=offset, chunks=iter([chunk])))


class _UnrangedSource:
    """A source that ignores a byte range and always sends the whole body."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def size(self, _reference) -> int:
        return len(self.payload)

    def stream(self, _reference, *, offset: int):
        del offset
        return _stream(AssetStream(offset=0, chunks=iter([self.payload])))


def _stream(value: AssetStream) -> tp.ContextManager[AssetStream]:
    import contextlib

    @contextlib.contextmanager
    def opened() -> tp.Iterator[AssetStream]:
        yield value

    return opened()


# --------------------------------------------------------------------------- round trip


def test_a_fetch_reports_the_size_and_digest_it_read_back_off_the_disk(tmp_path: Path) -> None:
    """The whole point of an executor: bytes land, and the report is an observation."""

    payload = b"dataset-bytes" * 64
    source = _source_file(tmp_path, payload)
    executor = _executor(tmp_path)

    measured = executor.measure(_CELL, str(source), max_bytes=len(payload))
    staged = executor.stage(_CELL, str(source), max_bytes=len(payload))

    assert measured.verdict is AssetStageVerdict.OK
    assert measured.size_bytes == len(payload)
    assert staged.verdict is AssetStageVerdict.OK
    assert staged.size_bytes == len(payload)
    assert staged.digest == _digest(payload)
    assert staged.path is not None
    assert Path(staged.path).read_bytes() == payload


def test_an_eviction_removes_the_bytes_and_a_second_one_is_still_a_success(tmp_path: Path) -> None:
    """Idempotent by design: the caller asked whether the bytes are gone."""

    payload = b"x" * 4096
    source = _source_file(tmp_path, payload)
    executor = _executor(tmp_path)
    staged = executor.stage(_CELL, str(source), max_bytes=len(payload))

    first = executor.evict(_CELL, str(source))
    second = executor.evict(_CELL, str(source))

    assert first.verdict is AssetStageVerdict.OK
    assert first.size_bytes == len(payload)
    assert second.verdict is AssetStageVerdict.OK
    assert second.size_bytes == 0
    assert staged.path is not None and not Path(staged.path).exists()


def test_two_cells_stage_the_same_asset_into_separate_paths(tmp_path: Path) -> None:
    """Otherwise evicting one cell's copy silently removes another cell's."""

    payload = b"shared" * 100
    source = _source_file(tmp_path, payload)
    executor = _executor(tmp_path)

    first = executor.stage("cell-a", str(source), max_bytes=len(payload))
    second = executor.stage("cell-b", str(source), max_bytes=len(payload))
    executor.evict("cell-a", str(source))

    assert first.path != second.path
    assert second.path is not None and Path(second.path).read_bytes() == payload


# --------------------------------------------------------------------------- the byte bound


def test_an_asset_larger_than_its_bound_is_refused_before_a_byte_is_written(tmp_path: Path) -> None:
    """The bound is a term of the request; an executor that overruns unbounds the budget."""

    payload = b"y" * 5000
    source = _source_file(tmp_path, payload)
    executor = _executor(tmp_path)

    result = executor.stage(_CELL, str(source), max_bytes=4999)

    assert result.verdict is AssetStageVerdict.REFUSED
    assert "4999" in result.detail
    assert list((tmp_path / "stage").rglob("*.part")) == []


def test_a_source_that_streams_past_its_bound_is_refused_and_leaves_nothing(tmp_path: Path) -> None:
    """A declared length is a claim; the executor still counts what arrives.

    A source that under-declares would otherwise fill a disk the budget believes
    is half empty -- and the ledger would confirm the smaller number, because
    the ledger records what the executor reports.
    """

    class _Liar:
        def size(self, _reference) -> int:
            return 10

        def stream(self, _reference, *, offset: int):
            del offset
            return _stream(AssetStream(offset=0, chunks=iter([b"z" * 4096, b"z" * 4096])))

    executor = _executor(tmp_path, allowed=("liar://",), sources={"liar": _Liar()})

    result = executor.stage(_CELL, "liar://asset", max_bytes=10)

    assert result.verdict is AssetStageVerdict.REFUSED
    assert "bound" in result.detail
    # Not merely unpublished: the overrun is discarded, so a retry cannot resume
    # into bytes the budget already refused.
    assert [path for path in (tmp_path / "stage").rglob("*") if path.is_file()] == []


# --------------------------------------------------------------------------- partial fetches


def test_a_partial_fetch_is_inconclusive_and_never_published(tmp_path: Path) -> None:
    """Half a dataset must never look like a cache hit."""

    payload = b"p" * 8192
    source = _TruncatingSource(payload, deliver=2048)
    executor = _executor(tmp_path, allowed=("part://",), sources={"part": source})

    result = executor.stage(_CELL, "part://dataset", max_bytes=len(payload))

    assert result.verdict is AssetStageVerdict.INCONCLUSIVE
    assert "2048 of 8192" in result.detail
    assert result.size_bytes == 0
    assert result.digest is None
    assert result.path is None
    assert not any(path.is_file() for path in (tmp_path / "stage").rglob("assets/*"))


def test_a_retry_resumes_from_where_the_partial_fetch_stopped(tmp_path: Path) -> None:
    """Resumability is what makes a bounded, interruptible fetch affordable.

    Without it, a transfer that dies at 90% starts again at zero on every pass,
    and a large asset on a flaky link is never staged at all.
    """

    payload = bytes(range(256)) * 32
    source = _TruncatingSource(payload, deliver=2048)
    executor = _executor(tmp_path, allowed=("part://",), sources={"part": source})

    first = executor.stage(_CELL, "part://dataset", max_bytes=len(payload))
    source.deliver = len(payload)
    second = executor.stage(_CELL, "part://dataset", max_bytes=len(payload))

    assert first.verdict is AssetStageVerdict.INCONCLUSIVE
    assert source.offsets == [0, 2048]
    assert second.verdict is AssetStageVerdict.OK
    assert second.size_bytes == len(payload)
    assert second.digest == _digest(payload)
    assert second.path is not None and Path(second.path).read_bytes() == payload


def test_a_source_that_ignores_the_range_restarts_instead_of_splicing(tmp_path: Path) -> None:
    """Trusting our own range request would corrupt the file in silence.

    The bytes would be the right length and the wrong content, which is the one
    outcome a digest catches only when the asset declared one.
    """

    payload = b"abcdefgh" * 512
    truncating = _TruncatingSource(payload, deliver=1024)
    executor = _executor(tmp_path, allowed=("part://",), sources={"part": truncating})
    executor.stage(_CELL, "part://dataset", max_bytes=len(payload))

    executor.sources["part"] = _UnrangedSource(payload)
    result = executor.stage(_CELL, "part://dataset", max_bytes=len(payload))

    assert result.verdict is AssetStageVerdict.OK
    assert result.path is not None and Path(result.path).read_bytes() == payload


# --------------------------------------------------------------------------- integrity


def test_a_digest_mismatch_quarantines_the_bytes_instead_of_serving_them(tmp_path: Path) -> None:
    """Publishing them hands a workload silently wrong data; deleting them loses the evidence."""

    payload = b"not-what-was-declared"
    source = _source_file(tmp_path, payload)
    declared = f"{source}#sha256:{'0' * 64}"
    executor = _executor(tmp_path)

    result = executor.stage(_CELL, declared, max_bytes=len(payload))

    quarantined = list((tmp_path / "stage").rglob("quarantine/*"))
    assert result.verdict is AssetStageVerdict.REFUSED
    assert "quarantined" in result.detail
    assert result.path is None
    assert [path.read_bytes() for path in quarantined] == [payload]
    assert not any(path.is_file() for path in (tmp_path / "stage").rglob("assets/*"))


def test_a_declared_digest_that_matches_is_staged_normally(tmp_path: Path) -> None:
    """Verification must not be a second way to fail an honest fetch."""

    payload = b"verified-bytes" * 32
    source = _source_file(tmp_path, payload)
    declared = f"{source}#{_digest(payload).replace(':', '=')}"
    executor = _executor(tmp_path)

    result = executor.stage(_CELL, declared, max_bytes=len(payload))

    assert result.verdict is AssetStageVerdict.OK
    assert result.digest == _digest(payload)


def test_a_second_quarantine_of_the_same_asset_keeps_exactly_one_copy(tmp_path: Path) -> None:
    """A permanently wrong source must not fill the disk by being retried."""

    payload = b"still-wrong"
    source = _source_file(tmp_path, payload)
    declared = f"{source}#sha256:{'1' * 64}"
    executor = _executor(tmp_path)

    executor.stage(_CELL, declared, max_bytes=len(payload))
    executor.stage(_CELL, declared, max_bytes=len(payload))

    assert len(list((tmp_path / "stage").rglob("quarantine/*"))) == 1


def test_an_asset_without_a_declared_digest_reports_what_was_observed(tmp_path: Path) -> None:
    """Unverified content is still measured content; it is simply not vouched for."""

    payload = b"unverified" * 10
    source = _source_file(tmp_path, payload)
    executor = _executor(tmp_path)

    result = executor.stage(_CELL, str(source), max_bytes=len(payload))

    assert result.digest == _digest(payload)


# --------------------------------------------------------------------------- what may be fetched


def test_an_asset_matching_no_declared_source_is_refused(tmp_path: Path) -> None:
    """Named assets are user input; an executor that fetches anything is a forgery primitive."""

    executor = _executor(tmp_path, allowed=())

    result = executor.stage(_CELL, "http://169.254.169.254/computeMetadata/v1/", max_bytes=1024)

    assert result.verdict is AssetStageVerdict.REFUSED
    assert STAGE_SOURCES_ENVIRONMENT in result.detail


def test_a_file_outside_the_declared_prefix_is_refused(tmp_path: Path) -> None:
    """The allowlist is a prefix on the asset, so traversal out of it must not pass."""

    executor = _executor(tmp_path)

    result = executor.stage(_CELL, f"{tmp_path / 'source'}/../secret", max_bytes=1024)

    assert result.verdict is AssetStageVerdict.REFUSED
    assert "traverse" in result.detail


def test_a_scheme_this_host_cannot_fetch_is_refused_rather_than_retried(tmp_path: Path) -> None:
    """Retrying an unchanged request that cannot work is the loop blocked exists to end."""

    executor = _executor(tmp_path, allowed=("gs://acme/",))

    result = executor.stage(_CELL, "gs://acme/datasets/pile-v3", max_bytes=1024)

    assert result.verdict is AssetStageVerdict.REFUSED
    assert "gs://" in result.detail


def test_a_directory_is_refused_because_it_has_no_length_or_digest(tmp_path: Path) -> None:
    """A tree gives up both guarantees at once, so it is refused rather than half-staged."""

    directory = tmp_path / "source" / "tree"
    directory.mkdir(parents=True)
    executor = _executor(tmp_path)

    result = executor.stage(_CELL, str(directory), max_bytes=1024)

    assert result.verdict is AssetStageVerdict.REFUSED
    assert "regular file" in result.detail


def test_a_missing_file_is_refused_rather_than_retried_forever(tmp_path: Path) -> None:
    """Absent is a fact about the asset, and asking again tomorrow gets the same answer."""

    executor = _executor(tmp_path)
    (tmp_path / "source").mkdir(parents=True, exist_ok=True)

    absent = executor.stage(_CELL, str(tmp_path / "source" / "gone"), max_bytes=1024)

    assert absent.verdict is AssetStageVerdict.REFUSED
    assert "does not exist" in absent.detail


# --------------------------------------------------------------------------- addressing


@pytest.mark.parametrize(
    ("asset", "expected"),
    [
        ("/data/set.bin", None),
        ("file:///data/set.bin", None),
        (f"/data/set.bin#sha256:{'a' * 64}", f"sha256:{'a' * 64}"),
        (f"https://h/x.tar#sha256={'b' * 64}", f"sha256:{'b' * 64}"),
        ("/data/set#1", None),
    ],
)
def test_an_integrity_fragment_is_read_only_when_it_is_one(asset: str, expected: str | None) -> None:
    """A path may legitimately contain '#', and guessing would fetch a different object."""

    reference = parse_asset_reference(asset)

    assert reference.expected_digest == expected
    if expected is None and "#" in asset:
        assert reference.location.endswith("#1")


def test_declared_sources_are_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty is the default, and it stages nothing."""

    monkeypatch.delenv(STAGE_SOURCES_ENVIRONMENT, raising=False)
    assert stage_sources_from_environment() == ()

    monkeypatch.setenv(STAGE_SOURCES_ENVIRONMENT, "gs://acme/  https://cdn/ ,gs://acme/")
    assert stage_sources_from_environment() == ("gs://acme/", "https://cdn/")


def test_a_relative_stage_root_is_refused(tmp_path: Path) -> None:
    """A cache rooted at a working directory is a cache nobody can find or bound."""

    with pytest.raises(ValueError, match="absolute"):
        LocalAssetStageExecutor(root=Path("relative/stage"))


@pytest.mark.parametrize("cell_id", ["../escape", "a/b", ""])
def test_a_cell_id_that_could_escape_its_directory_is_refused(tmp_path: Path, cell_id: str) -> None:
    """Cell IDs reach here from a record; a separator in one must not select a path."""

    executor = _executor(tmp_path)

    result = executor.evict(cell_id, "/data/x")

    assert result.verdict is AssetStageVerdict.REFUSED


# --------------------------------------------------------------------------- http


def test_an_http_source_resumes_with_a_byte_range() -> None:
    """The header is the whole reason a large fetch is affordable on a flaky link."""

    payload = b"h" * 4096
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("range"))
        if request.method == "HEAD":
            return httpx.Response(200, headers={"content-length": str(len(payload))})
        start = int((request.headers.get("range") or "bytes=0-").removeprefix("bytes=").split("-")[0])
        return httpx.Response(206 if start else 200, content=payload[start:])

    source = HttpAssetSource(client=httpx.Client(transport=httpx.MockTransport(handler)))
    reference = parse_asset_reference("https://cdn/asset.bin")

    assert source.size(reference) == len(payload)
    with source.stream(reference, offset=1024) as stream:
        assert stream.offset == 1024
        assert b"".join(stream.chunks) == payload[1024:]
    assert seen[-1] == "bytes=1024-"


def test_an_http_source_without_a_content_length_is_refused() -> None:
    """Nothing to bound the fetch by and nothing to compare the written bytes against."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    source = HttpAssetSource(client=httpx.Client(transport=httpx.MockTransport(handler)))
    executor = LocalAssetStageExecutor(
        root=Path("/tmp/eyvan-stage-test"),
        allowed_prefixes=("https://cdn/",),
        sources={"https": source},
    )

    result = executor.measure(_CELL, "https://cdn/asset.bin", max_bytes=4096)

    assert result.verdict is AssetStageVerdict.REFUSED
    assert "content length" in result.detail


@pytest.mark.parametrize(
    ("status", "verdict"),
    [(404, AssetStageVerdict.REFUSED), (403, AssetStageVerdict.REFUSED), (503, AssetStageVerdict.INCONCLUSIVE)],
)
def test_an_http_status_is_classified_by_what_it_establishes(status: int, verdict: AssetStageVerdict) -> None:
    """A 5xx says nothing about the asset, and blocking on it outlives the outage."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status)

    source = HttpAssetSource(client=httpx.Client(transport=httpx.MockTransport(handler)))
    executor = LocalAssetStageExecutor(
        root=Path("/tmp/eyvan-stage-test"),
        allowed_prefixes=("https://cdn/",),
        sources={"https": source},
    )

    assert executor.measure(_CELL, "https://cdn/asset.bin", max_bytes=4096).verdict is verdict


def test_a_prefix_admits_only_what_lies_under_it(tmp_path: Path) -> None:
    """A bare string test would admit ``/data/publicly-readable`` under ``/data/public``.

    The boundary is what makes an allowlist an allowlist. Without it a declared
    source silently covers every sibling whose name it happens to prefix.
    """

    (tmp_path / "source").mkdir(parents=True, exist_ok=True)
    sibling = tmp_path / "sourcecode" / "secret"
    sibling.parent.mkdir(parents=True, exist_ok=True)
    sibling.write_bytes(b"private")
    executor = _executor(tmp_path)

    result = executor.stage(_CELL, str(sibling), max_bytes=4096)

    assert result.verdict is AssetStageVerdict.REFUSED
    assert STAGE_SOURCES_ENVIRONMENT in result.detail


def test_a_url_that_walks_out_of_its_prefix_is_refused(tmp_path: Path) -> None:
    """The traversal check is not a filesystem concern; it is a prefix concern."""

    executor = _executor(tmp_path, allowed=("https://cdn/public/",))

    result = executor.measure(_CELL, "https://cdn/public/../private/key", max_bytes=4096)

    assert result.verdict is AssetStageVerdict.REFUSED
    assert "traverse" in result.detail


def test_a_source_that_resumes_bytes_this_host_never_wrote_is_inconclusive(tmp_path: Path) -> None:
    """Writing from that offset leaves a hole and publishes the wrong content.

    Right length, wrong bytes, and a digest only catches it when the asset
    declared one -- so the transfer is abandoned instead.
    """

    class _AlwaysResumes:
        def size(self, _reference) -> int:
            return 4096

        def stream(self, _reference, *, offset: int):
            del offset
            return _stream(AssetStream(offset=2048, chunks=iter([b"r" * 2048])))

    executor = _executor(tmp_path, allowed=("resume://",), sources={"resume": _AlwaysResumes()})

    result = executor.stage(_CELL, "resume://asset", max_bytes=4096)

    assert result.verdict is AssetStageVerdict.INCONCLUSIVE
    assert "no partial copy" in result.detail
    assert [path for path in (tmp_path / "stage").rglob("*") if path.is_file()] == []


def test_a_relative_agent_state_directory_still_builds_an_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relative state directory is supported everywhere else, so it must not
    become the reason an agent refuses to start.
    """

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(STAGE_SOURCES_ENVIRONMENT, raising=False)

    executor = LocalAssetStageExecutor.from_environment(Path("agent") / "stages")

    assert executor.root.is_absolute()
    assert executor.root == (tmp_path / "agent" / "stages").resolve()


def test_a_symbolic_link_inside_a_declared_source_is_refused(tmp_path: Path) -> None:
    """The allowlist is a prefix on a name, and a link redirects the bytes behind it.

    A workload already running on this host can drop a link inside a declared
    prefix and then declare it. The agent may run as root and the workload does
    not, so following it copies a file the workload could not read into a
    directory it can.
    """

    secret = tmp_path / "secret"
    secret.write_bytes(b"root-only")
    (tmp_path / "source").mkdir(parents=True, exist_ok=True)
    link = tmp_path / "source" / "innocent.bin"
    link.symlink_to(secret)
    executor = _executor(tmp_path)

    result = executor.stage(_CELL, str(link), max_bytes=4096)

    assert result.verdict is AssetStageVerdict.REFUSED
    assert "symbolic link" in result.detail
    assert [path for path in (tmp_path / "stage").rglob("*") if path.is_file()] == []
