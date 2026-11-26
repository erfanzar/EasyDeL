# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Exercise the GCS JSON protocol without cloud credentials or network access."""

from __future__ import annotations

import json
from urllib.parse import parse_qs, unquote, urlsplit

import pytest

from eyvan.track._gcs import GcsTrackingStore
from eyvan.track._store import SegmentConflictError, TrackingStoreError


class GcsServer:
    """In-memory HTTP peer implementing the exercised GCS JSON endpoints."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.calls: list[tuple[str, str, dict[str, str], bytes | None]] = []
        self.metadata_calls = 0
        self.failure: int | None = None

    def __call__(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> tuple[int, dict[str, str], bytes]:
        self.calls.append((method, url, headers, body))
        parsed = urlsplit(url)
        if parsed.hostname == "metadata.google.internal":
            self.metadata_calls += 1
            assert headers == {"Metadata-Flavor": "Google"}
            return 200, {}, json.dumps({"access_token": "metadata-secret", "expires_in": 3600}).encode()
        assert headers["Authorization"] in {"Bearer test-secret", "Bearer metadata-secret"}
        if self.failure is not None:
            return self.failure, {}, b'{"error":"test-secret metadata-secret"}'
        query = parse_qs(parsed.query)
        if method == "POST":
            assert query["ifGenerationMatch"] == ["0"]
            assert query["uploadType"] == ["media"]
            name = query["name"][0]
            if name in self.objects:
                return 412, {}, b""
            assert body is not None
            self.objects[name] = body
            return 200, {}, b"{}"
        if parsed.path.endswith("/o"):
            prefix = query.get("prefix", [""])[0]
            names = sorted(name for name in self.objects if name.startswith(prefix))
            offset = int(query.get("pageToken", ["0"])[0])
            page: dict[str, object] = {"items": [{"name": name} for name in names[offset : offset + 2]]}
            if offset + 2 < len(names):
                page["nextPageToken"] = str(offset + 2)
            return 200, {}, json.dumps(page).encode()
        assert query["alt"] == ["media"]
        name = unquote(parsed.path.split("/o/", 1)[1])
        if name not in self.objects:
            return 404, {}, b"missing"
        return 200, {}, self.objects[name]


def test_create_replay_conflict_and_missing_object() -> None:
    server = GcsServer()
    store = GcsTrackingStore("gs://bucket/tracking/", token_provider=lambda: "test-secret", transport=server)
    assert store.root == "gs://bucket/tracking"
    store.put("org/project/run/scalars/segment", b"original")
    store.put("org/project/run/scalars/segment", b"original")
    assert store.get("org/project/run/scalars/segment") == b"original"
    assert server.objects == {"tracking/org/project/run/scalars/segment": b"original"}
    with pytest.raises(SegmentConflictError):
        store.put("org/project/run/scalars/segment", b"different")
    with pytest.raises(FileNotFoundError):
        store.get("absent")


def test_paginated_listing_stays_under_root_and_relative_prefix() -> None:
    server = GcsServer()
    server.objects = {
        "tracking/org/project/c": b"c",
        "tracking/org/project/a": b"a",
        "tracking/org/project/b": b"b",
        "tracking/org/other/a": b"other",
        "tracking/organization/a": b"not-org",
        "tracking-other/org/project/a": b"outside",
    }
    store = GcsTrackingStore("gs://bucket/tracking", token_provider=lambda: "test-secret", transport=server)
    assert store.list("org/project/") == ["org/project/a", "org/project/b", "org/project/c"]
    assert store.list("org") == ["org/other/a", "org/project/a", "org/project/b", "org/project/c"]
    assert store.list() == [
        "org/other/a",
        "org/project/a",
        "org/project/b",
        "org/project/c",
        "organization/a",
    ]
    assert any("pageToken=" in call[1] for call in server.calls)


def test_metadata_token_is_cached_and_not_in_the_url() -> None:
    server = GcsServer()
    store = GcsTrackingStore("gs://bucket", transport=server)
    store.put("one", b"1")
    store.put("two", b"2")
    assert store.list() == ["one", "two"]
    assert server.metadata_calls == 1
    assert all("secret" not in call[1] for call in server.calls)


def test_metadata_token_refreshes_before_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    server = GcsServer()
    clock = [0.0]
    monkeypatch.setattr("eyvan.track._gcs.time.monotonic", lambda: clock[0])
    store = GcsTrackingStore("gs://bucket", transport=server)
    store.put("one", b"1")
    clock[0] = 3539
    store.put("two", b"2")
    assert server.metadata_calls == 1
    clock[0] = 3541
    assert store.list() == ["one", "two"]
    assert server.metadata_calls == 2


def test_token_provider_failure_is_credential_free() -> None:
    server = GcsServer()

    def broken_provider() -> str:
        raise RuntimeError("test-secret")

    store = GcsTrackingStore("gs://bucket", token_provider=broken_provider, transport=server)
    with pytest.raises(TrackingStoreError) as error:
        store.list()
    assert "secret" not in str(error.value)
    assert not server.calls


@pytest.mark.parametrize("operation", ["put", "get", "list"])
def test_error_body_and_transport_exception_cannot_leak_token(operation: str) -> None:
    server = GcsServer()
    server.failure = 403
    store = GcsTrackingStore("gs://bucket", token_provider=lambda: "test-secret", transport=server)
    args = ("key", b"data") if operation == "put" else ("key",) if operation == "get" else ()
    with pytest.raises(TrackingStoreError) as error:
        getattr(store, operation)(*args)
    assert "403" in str(error.value)
    assert "secret" not in str(error.value)

    def broken(*args: object) -> tuple[int, dict[str, str], bytes]:
        raise OSError("request contained Bearer test-secret")

    broken_store = GcsTrackingStore("gs://bucket", token_provider=lambda: "test-secret", transport=broken)
    with pytest.raises(TrackingStoreError) as error:
        getattr(broken_store, operation)(*args)
    assert "secret" not in str(error.value)
    assert error.value.__suppress_context__


@pytest.mark.parametrize(
    "uri",
    [
        "gs://",
        "https://bucket/prefix",
        "gs://user:secret@bucket/root",  # pragma: allowlist secret
        "gs://bucket/../x",
    ],
)
def test_invalid_roots_are_refused(uri: str) -> None:
    with pytest.raises(ValueError):
        GcsTrackingStore(uri, token_provider=lambda: "test-secret", transport=GcsServer())


@pytest.mark.parametrize("key", ["../escape", "/absolute", "one//two", "one/../two"])
def test_unsafe_keys_are_refused_before_request(key: str) -> None:
    server = GcsServer()
    store = GcsTrackingStore("gs://bucket", token_provider=lambda: "test-secret", transport=server)
    with pytest.raises(ValueError):
        store.put(key, b"no")
    assert not server.calls
