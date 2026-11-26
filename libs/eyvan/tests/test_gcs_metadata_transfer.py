# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The callable worker's metadata token path must trust only the metadata service.

``eyvan.identity.metadata`` documents why the flavour header is checked on the
response and not just the request: anything that can answer on the metadata
address -- an SSRF proxy, a stray local service, a captive portal -- can
otherwise hand the worker a credential of its choosing. The token fetch must
also reach the link-local server directly, with none of the caller's proxy
settings in the path.
"""

from __future__ import annotations

import typing as tp
import urllib.request

import pytest

from eyvan.worker.runtime import callable_worker


class _FakeHeaders(dict):
    """Case-insensitive mapping, the way real HTTP response headers behave."""

    def get(self, name: str, default: tp.Any = None) -> tp.Any:  # type: ignore[override]
        for key, value in self.items():
            if key.lower() == name.lower():
                return value
        return default


class _FakeResponse:
    def __init__(self, headers: dict[str, str], body: bytes = b"{}"):
        self.headers = _FakeHeaders(headers)
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _install_opener(
    monkeypatch: pytest.MonkeyPatch,
    response: _FakeResponse,
) -> list[tp.Any]:
    """Route ``build_opener`` through a fake and return the handlers it was given."""

    seen: list[tp.Any] = []

    class FakeOpener:
        def __init__(self, *handlers: tp.Any) -> None:
            seen.extend(handlers)

        def open(self, request: urllib.request.Request, timeout: float | None = None) -> _FakeResponse:
            assert request.get_header("Metadata-flavor") == "Google"
            assert timeout == 10
            return response

    monkeypatch.setattr(callable_worker.urllib.request, "build_opener", lambda *handlers: FakeOpener(*handlers))
    return seen


def test_token_fetch_reaches_the_metadata_server_directly_and_returns_the_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _FakeResponse({"Metadata-Flavor": "Google"}, b'{"access_token": "tok", "expires_in": 3587}')

    handlers = _install_opener(monkeypatch, response)

    assert callable_worker.GcsMetadataTransfer()._access_token() == "tok"
    proxy_handlers = [handler for handler in handlers if isinstance(handler, urllib.request.ProxyHandler)]
    assert len(proxy_handlers) == 1
    assert proxy_handlers[0].proxies == {}


def test_a_reply_that_does_not_identify_itself_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _FakeResponse({}, b'{"access_token": "attacker-chosen"}')
    _install_opener(monkeypatch, response)

    with pytest.raises(RuntimeError, match="did not identify itself"):
        callable_worker.GcsMetadataTransfer()._access_token()
