# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Offline regressions for independently reported tracking format/store findings."""

from __future__ import annotations

import gzip
import io
import json
import os
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import pytest

from eyvan.track import _format as fmt
from eyvan.track import _gcs, _store
from eyvan.track._gcs import GcsTrackingStore
from eyvan.track._run import replay_spill
from eyvan.track._store import LocalTrackingStore, TrackingStoreError, store_for_uri
from eyvan.track.tracker.index import TrackingIndex


def segment_key(stream: str = "meta", *, seq: int = 0, org: str = "org") -> str:
    """Return a canonical segment key scoped for indexing."""

    return f"{org}/project/run/{stream}/{fmt.SegmentName(1, 0, 'bbbbbbbbbbbb', seq).filename}"


class OfflineGcs:
    """Small paginated GCS peer that never performs network operations."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.pages: list[str] = []

    def __call__(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> tuple[int, dict[str, str], bytes]:
        assert headers["Authorization"] == "Bearer test-token"
        parsed = urlsplit(url)
        query = parse_qs(parsed.query)
        if method == "POST":
            assert body is not None
            name = query["name"][0]
            if name in self.objects:
                return 412, {}, b""
            self.objects[name] = body
            return 200, {}, b"{}"
        if parsed.path.endswith("/o"):
            self.pages.append(query.get("pageToken", [""])[0])
            names = sorted(name for name in self.objects if name.startswith(query["prefix"][0]))
            offset = int(query.get("pageToken", ["0"])[0])
            result: dict[str, object] = {"items": [{"name": name} for name in names[offset : offset + 2]]}
            if offset + 2 < len(names):
                result["nextPageToken"] = str(offset + 2)
            return 200, {}, json.dumps(result).encode()
        name = unquote(parsed.path.split("/o/", 1)[1])
        return (200, {}, self.objects[name]) if name in self.objects else (404, {}, b"")


@pytest.mark.parametrize("operation", ["get", "put", "put_existing", "list"])
def test_local_store_refuses_directory_symlinks(tmp_path: Path, operation: str) -> None:
    """Finding 1: directory symlinks cannot redirect any store operation."""

    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret").write_bytes(b"private")
    (root / "run").symlink_to(outside, target_is_directory=True)
    store = LocalTrackingStore(root)
    with pytest.raises(TrackingStoreError):
        if operation == "get":
            store.get("run/secret")
        elif operation == "put":
            store.put("run/new", b"payload")
        elif operation == "put_existing":
            store.put("run/secret", b"private")
        else:
            store.list("run")
    assert not (outside / "new").exists()
    assert (outside / "secret").read_bytes() == b"private"


@pytest.mark.parametrize("operation", ["get", "put"])
def test_local_store_refuses_leaf_symlinks(tmp_path: Path, operation: str) -> None:
    """Finding 1: even an identical existing object cannot be a symlink."""

    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "secret"
    outside.write_bytes(b"private")
    (root / "object").symlink_to(outside)
    store = LocalTrackingStore(root)
    with pytest.raises(TrackingStoreError):
        if operation == "get":
            store.get("object")
        else:
            store.put("object", b"private")
    assert outside.read_bytes() == b"private"


def test_local_store_parent_replacement_cannot_redirect_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding 1: replacing a verified parent cannot redirect the eventual link."""

    root, outside = tmp_path / "root", tmp_path / "outside"
    (root / "run").mkdir(parents=True)
    outside.mkdir()
    store = LocalTrackingStore(root)
    real_link = os.link

    def replace_parent(source: str, destination: str, **kwargs: object) -> None:
        (root / "run").rename(root / "original-run")
        (root / "run").symlink_to(outside, target_is_directory=True)
        (outside / Path(source).name).write_bytes(b"attacker-controlled")
        real_link(source, destination, **kwargs)

    monkeypatch.setattr(os, "link", replace_parent)
    store.put("run/object", b"payload")
    assert not (outside / "object").exists()
    assert (root / "original-run" / "object").read_bytes() == b"payload"


def test_gcs_listing_skips_unusable_names_across_pages_and_scans(tmp_path: Path) -> None:
    """Finding 2: one unusable GCS object must not poison another tenant's scan."""

    peer = OfflineGcs()
    peer.objects = {
        "tracking/bad/project/run/scalars//junk": b"junk",
        "tracking/bad/project/./junk": b"junk",
        "tracking/bad/project/../junk": b"junk",
        "tracking/" + segment_key(org="good"): fmt.encode_segment([{"op": "summary", "t": 1, "summary": {"ok": 1}}]),
    }
    store = GcsTrackingStore("gs://bucket/tracking", token_provider=lambda: "test-token", transport=peer)
    assert store.list() == [segment_key(org="good")]
    assert "2" in peer.pages
    index = TrackingIndex(tmp_path / "index.sqlite3")
    assert index.scan(store).ingested_segments == 1
    assert index.run_detail("good", "project", "run")["summary"] == {"ok": 1}
    assert index.scan(store).ingested_segments == 0
    store.put(segment_key(seq=1, org="good"), fmt.encode_segment([{"op": "summary", "t": 2, "summary": {"ok": 2}}]))
    assert index.scan(store).ingested_segments == 1
    assert index.run_detail("good", "project", "run")["summary"] == {"ok": 2}


def test_decode_rejects_gzip_expansion_and_keeps_indexing_other_segments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding 3: exercise a small ceiling rather than allocate a real gzip bomb."""

    monkeypatch.setattr(fmt, "MAX_SEGMENT_BYTES", 64 * 1024, raising=False)
    bomb = fmt.encode_segment([{"op": "summary", "t": 1, "summary": {"note": "x" * (64 * 1024)}}])
    assert len(bomb) < 256
    with pytest.raises(ValueError, match="limit"):
        fmt.decode_segment(bomb)
    store = LocalTrackingStore(tmp_path / "store")
    store.put(segment_key(), bomb)
    store.put(segment_key("scalars"), fmt.encode_segment([{"t": 2, "step": 1, "v": {"loss": 3}}]))
    index = TrackingIndex(tmp_path / "index.sqlite3")
    scanned = index.scan(store)
    assert (scanned.ingested_segments, scanned.errors) == (2, 1)
    assert len(index.run_detail("org", "project", "run")["errors"]) == 1
    assert index.series("org", "project", "run", ["loss"])["series"][0]["points"] == [[1, 3, 3, 3]]
    assert index.scan(store).ingested_segments == 0


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_unicode_separators_roundtrip_summary_and_index(tmp_path: Path, separator: str) -> None:
    """Finding 10: literal Unicode separators are text, not NDJSON delimiters."""

    records = [
        {"op": "summary", "t": 1, "summary": {"note": f"before{separator}after"}},
        {"op": "summary", "t": 2, "summary": {"other": "retained"}},
    ]
    encoded = fmt.encode_segment(records)
    assert separator.encode() in gzip.decompress(encoded)
    assert fmt.decode_segment(encoded) == records
    store = LocalTrackingStore(tmp_path / "store")
    store.put(segment_key(), encoded)
    index = TrackingIndex(tmp_path / "index.sqlite3")
    assert index.scan(store).errors == 0
    assert index.run_detail("org", "project", "run")["summary"] == {
        "note": f"before{separator}after",
        "other": "retained",
    }


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_unicode_separators_preserve_completions_tables_and_neighbor_records(tmp_path: Path, separator: str) -> None:
    """Finding 27: completion and table text retains legal Unicode unchanged."""

    text = f"before{separator}after"
    record = {
        "t": 1,
        "step": 1,
        "group": "group-1",
        "prompt": text,
        "answer": None,
        "policy_version": None,
        "responses": [
            {"text": text, "rewards": {}, "reward": 1, "advantage": 0, "tokens": 1, "kl": None, "truncated": None}
        ],
    }
    records = [record, dict(record, group="group-2", step=2, t=2)]
    assert fmt.decode_segment(fmt.encode_segment(records)) == records
    store = LocalTrackingStore(tmp_path / "store")
    store.put(segment_key("completions"), fmt.encode_segment(records))
    table = {"t": 3, "step": 3, "key": "cases", "columns": ["text"], "rows": [[text], ["retained"]]}
    store.put(segment_key("tables"), fmt.encode_segment([table]))
    index = TrackingIndex(tmp_path / "index.sqlite3")
    assert index.scan(store).errors == 0
    groups = index.completions("org", "project", "run")
    assert groups["total"] == 2
    detail = index.completion_detail("org", "project", "run", groups["groups"][0]["id"])
    assert detail["prompt"] == text and detail["responses"][0]["text"] == text
    assert index.tables("org", "project", "run", "cases", store=store)["rows"] == [[text], ["retained"]]


@pytest.mark.parametrize("prefix", ["runs%252Fone", "runs%23one", "runs%3Fone"])
def test_gcs_canonical_root_roundtrip_preserves_spill_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefix: str
) -> None:
    """Finding 11: replay uses exactly the original GCS prefix after URI parsing."""

    peer = OfflineGcs()
    monkeypatch.setattr(_gcs, "_urllib_transport", peer)
    monkeypatch.setattr(GcsTrackingStore, "_metadata_token", lambda self: "test-token")
    original = store_for_uri(f"gs://bucket/{prefix}")
    reconstructed = store_for_uri(original.root)
    key = segment_key()
    original.put(key, b"payload")
    reconstructed.put(key, b"payload")
    assert len(peer.objects) == 1
    peer.objects.clear()
    spill = tmp_path / "spill"
    spill.mkdir()
    marker = spill / "segment.json"
    marker.write_text(json.dumps({"store": original.root, "key": key}))
    marker.with_suffix(".seg").write_bytes(b"payload")
    result = replay_spill(spill)
    assert (result.replayed, result.failed) == (1, 0)
    assert peer.objects == {f"{unquote(prefix)}/{key}": b"payload"}


@pytest.mark.parametrize(
    "uri",
    [
        "https://storage.googleapis.com/bucket/tracking?access_token=TOPSECRET",
        "file://user:TOPSECRET@localhost/tmp/tracking",  # pragma: allowlist secret
    ],
)
def test_rejected_store_uri_errors_are_credential_free(tmp_path: Path, uri: str) -> None:
    """Finding 12: direct validation errors and replay results never include secrets."""

    with pytest.raises(ValueError) as error:
        store_for_uri(uri)
    assert "TOPSECRET" not in str(error.value)
    spill = tmp_path / "spill"
    spill.mkdir()
    marker = spill / "segment.json"
    marker.write_text(json.dumps({"store": uri, "key": segment_key()}))
    marker.with_suffix(".seg").write_bytes(b"payload")
    result = replay_spill(spill)
    assert (result.replayed, result.failed) == (0, 1)
    assert result.errors and all("TOPSECRET" not in message for message in result.errors)


@pytest.mark.parametrize(
    ("constant", "limit", "raw"),
    [
        ("MAX_COMPRESSED_SEGMENT_BYTES", 16, b'{"note":"normal"}\n'),
        ("MAX_SEGMENT_LINE_BYTES", 16, b'{"note":"xxxxxxxxxxxxxxxx"}\n'),
        ("MAX_SEGMENT_RECORDS", 2, b"{}\n{}\n{}\n"),
    ],
)
def test_decode_checks_each_resource_budget_before_parsing(
    monkeypatch: pytest.MonkeyPatch, constant: str, limit: int, raw: bytes
) -> None:
    """Finding 3: compressed bytes, individual lines and record counts are bounded."""

    monkeypatch.setattr(fmt, constant, limit)
    with pytest.raises(ValueError, match="limit"):
        fmt.decode_segment(gzip.compress(raw))


def test_decode_budget_is_cumulative_over_gzip_members(monkeypatch: pytest.MonkeyPatch) -> None:
    """Concatenated gzip members cannot reset the decompressed byte budget."""

    monkeypatch.setattr(fmt, "MAX_SEGMENT_BYTES", 8)
    first = gzip.compress(b"{}\n")
    assert fmt.decode_segment(first + first) == [{}, {}]
    with pytest.raises(ValueError, match="limit"):
        fmt.decode_segment(first + first + first)


def test_decode_accepts_exact_limits_crlf_and_missing_final_newline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Normal historical gzip NDJSON still decodes, including exact-budget input."""

    raw = b'{}\r\n\n{"note":"legacy"}'
    monkeypatch.setattr(fmt, "MAX_SEGMENT_BYTES", len(raw))
    monkeypatch.setattr(fmt, "MAX_SEGMENT_LINE_BYTES", len(raw))
    assert fmt.decode_segment(gzip.compress(raw)) == [{}, {"note": "legacy"}]
    assert fmt.decode_segment(fmt.encode_segment([])) == []


def test_local_store_read_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Finding 3: a large on-disk object is stopped during reading, before decoding."""

    store = LocalTrackingStore(tmp_path / "store")
    store.put("object", b"x" * 129)
    monkeypatch.setattr(_store, "MAX_OBJECT_BYTES", 128)
    with pytest.raises(_store.TrackingSizeLimitError, match="limit"):
        store.get("object")


@pytest.mark.parametrize("status", [200, 403])
def test_gcs_urllib_transport_bounds_response_reads(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    """Finding 3: default HTTP reads stop at the limit even without Content-Length."""

    from urllib.error import HTTPError

    class Response(io.BytesIO):
        def __init__(self) -> None:
            super().__init__(b"x" * 4096)
            self.headers: dict[str, str] = {}
            self.status = status
            self.consumed = 0

        def read(self, size: int = -1) -> bytes:
            assert size >= 0
            result = super().read(size)
            self.consumed += len(result)
            return result

    response = Response()

    class Opener:
        def open(self, request: object, timeout: float) -> Response:
            if status != 200:
                raise HTTPError("https://example.invalid", status, "failed", {}, response)
            return response

    monkeypatch.setattr(_gcs, "build_opener", lambda *args: Opener())
    monkeypatch.setattr(_store, "MAX_OBJECT_BYTES", 128)
    with pytest.raises(_store.TrackingSizeLimitError, match="limit"):
        _gcs._urllib_transport("GET", "https://example.invalid", {}, None)
    assert response.consumed == 129


def test_gcs_custom_transport_cannot_bypass_response_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even an injected HTTP peer obeys the store's response byte limit."""

    peer = OfflineGcs()
    peer.objects["tracking/object"] = b"x" * 129
    monkeypatch.setattr(_gcs, "MAX_OBJECT_BYTES", 128)
    store = GcsTrackingStore("gs://bucket/tracking", token_provider=lambda: "test-token", transport=peer)
    with pytest.raises(_store.TrackingSizeLimitError, match="limit"):
        store.get("object")


def test_local_store_listing_ignores_symlink_entries(tmp_path: Path) -> None:
    """Finding 1: root listing follows neither internal nor external symlink entries."""

    root, outside = tmp_path / "root", tmp_path / "outside"
    (root / "real").mkdir(parents=True)
    outside.mkdir()
    (root / "real" / "object").write_bytes(b"safe")
    (outside / "secret").write_bytes(b"private")
    (root / "inside-link").symlink_to(root / "real", target_is_directory=True)
    (root / "outside-link").symlink_to(outside, target_is_directory=True)
    (root / "leaf-link").symlink_to(outside / "secret")
    store = LocalTrackingStore(root)
    assert store.list() == ["real/object"]
    with pytest.raises(TrackingStoreError):
        store.put("inside-link/object", b"safe")
