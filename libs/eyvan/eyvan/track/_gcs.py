# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Standard-library GCS JSON API access for immutable tracking segments.

No Google SDK or tracker dependencies are imported in training processes.
Credentials are obtained from the GCE metadata service unless a caller supplies
its own token provider. Response bodies and transport exceptions are deliberately
not included in error messages: either may contain a credential.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
import typing as tp
from urllib.error import HTTPError
from urllib.parse import quote, unquote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from eyvan.track._store import (
    MAX_OBJECT_BYTES,
    SegmentConflictError,
    TrackingSizeLimitError,
    TrackingStoreError,
    _read_bounded,
    validate_key,
)

Transport = tp.Callable[[str, str, dict[str, str], bytes | None], tuple[int, tp.Mapping[str, str], bytes]]
TokenProvider = tp.Callable[[], str]
_METADATA_URL = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
_API_ROOT = "https://storage.googleapis.com"


class _NoRedirect(HTTPRedirectHandler):
    """Refuse redirects rather than forwarding bearer credentials elsewhere."""

    def redirect_request(self, req: Request, fp: tp.Any, code: int, msg: str, headers: tp.Any, newurl: str) -> None:
        return None


def _urllib_transport(
    method: str, url: str, headers: dict[str, str], body: bytes | None
) -> tuple[int, tp.Mapping[str, str], bytes]:
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with build_opener(_NoRedirect()).open(request, timeout=30) as response:
            return response.status, dict(response.headers.items()), _read_bounded(response)
    except HTTPError as error:
        with error:
            return error.code, dict(error.headers.items()), _read_bounded(error)


class GcsTrackingStore:
    """A create-only tracking store rooted at ``gs://bucket/prefix``.

    Args:
        uri: Bucket URI and optional object-name prefix.
        token_provider: Optional callable returning an OAuth access token. The
            default reads and caches the default service account's metadata token.
        transport: Optional HTTP peer for offline tests. Receives method, URL,
            headers and body; returns status, response headers and response bytes.
    """

    def __init__(
        self, uri: str, *, token_provider: TokenProvider | None = None, transport: Transport | None = None
    ) -> None:
        parsed = urlsplit(uri)
        if (
            parsed.scheme != "gs"
            or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]", parsed.netloc)
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("tracking GCS URI must be gs://bucket with an optional relative prefix")
        self._bucket = parsed.netloc
        self._prefix = validate_key(unquote(parsed.path).lstrip("/"), allow_empty=True)
        self._transport = transport or _urllib_transport
        self._token_provider = token_provider or self._metadata_token
        self._token_lock = threading.Lock()
        self._cached_token: str | None = None
        self._token_expires_at = 0.0

    @property
    def root(self) -> str:
        """Return this store's canonical bucket URI and prefix."""

        return f"gs://{self._bucket}" + (f"/{quote(self._prefix, safe='/')}" if self._prefix else "")

    def _request(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None = None
    ) -> tuple[int, tp.Mapping[str, str], bytes]:
        try:
            status, response_headers, payload = self._transport(method, url, headers, body)
            if len(payload) > MAX_OBJECT_BYTES:
                raise TrackingSizeLimitError("tracking GCS response exceeds the byte limit")
            return status, response_headers, payload
        except TrackingSizeLimitError:
            raise
        except Exception:
            raise TrackingStoreError("tracking GCS HTTP request failed") from None

    def _metadata_token(self) -> str:
        with self._token_lock:
            now = time.monotonic()
            if self._cached_token is not None and now < self._token_expires_at:
                return self._cached_token
            status, _, body = self._request("GET", _METADATA_URL, {"Metadata-Flavor": "Google"})
            if status != 200:
                raise TrackingStoreError(f"tracking GCS metadata token request returned HTTP {status}")
            try:
                record = json.loads(body)
                token = record["access_token"]
                expires_in = float(record["expires_in"])
                if not isinstance(token, str) or not token or not math.isfinite(expires_in) or expires_in <= 0:
                    raise ValueError
            except (KeyError, TypeError, ValueError, UnicodeError):
                raise TrackingStoreError("tracking GCS metadata service returned an invalid token response") from None
            self._cached_token = token
            # Refresh before expiry, including for unusually short-lived tokens.
            self._token_expires_at = now + expires_in - min(60.0, expires_in / 10)
            return token

    def _headers(self) -> dict[str, str]:
        try:
            token = self._token_provider()
            if not isinstance(token, str) or not token or "\r" in token or "\n" in token:
                raise ValueError
        except Exception:
            raise TrackingStoreError("could not obtain a tracking GCS access token") from None
        return {"Authorization": f"Bearer {token}"}

    def _name(self, key: str) -> str:
        key = validate_key(key)
        return f"{self._prefix}/{key}" if self._prefix else key

    def put(self, key: str, data: bytes) -> None:
        """Create a segment, accepting a replay only when its bytes are identical."""

        name = self._name(key)
        query = urlencode({"uploadType": "media", "name": name, "ifGenerationMatch": "0"})
        url = f"{_API_ROOT}/upload/storage/v1/b/{quote(self._bucket, safe='')}/o?{query}"
        headers = {**self._headers(), "Content-Type": "application/octet-stream"}
        status, _, _ = self._request("POST", url, headers, data)
        if 200 <= status < 300:
            return
        if status == 412:
            if self.get(key) == data:
                return
            raise SegmentConflictError("tracking segment already exists with different content")
        raise TrackingStoreError(f"tracking GCS upload returned HTTP {status}")

    def get(self, key: str) -> bytes:
        """Read object bytes; missing objects raise ``FileNotFoundError``."""

        name = self._name(key)
        url = f"{_API_ROOT}/storage/v1/b/{quote(self._bucket, safe='')}/o/{quote(name, safe='')}?alt=media"
        status, _, body = self._request("GET", url, self._headers())
        if status == 404:
            raise FileNotFoundError(key)
        if status != 200:
            raise TrackingStoreError(f"tracking GCS download returned HTTP {status}")
        return body

    def list(self, prefix: str = "") -> list[str]:
        """List relative keys recursively below a directory prefix, with pagination."""

        relative = validate_key(prefix, allow_empty=True)
        root_prefix = f"{self._prefix}/" if self._prefix else ""
        object_prefix = root_prefix + (f"{relative}/" if relative else "")
        keys: set[str] = set()
        page_token = ""
        seen_tokens: set[str] = set()
        while True:
            params = {"prefix": object_prefix}
            if page_token:
                params["pageToken"] = page_token
            url = f"{_API_ROOT}/storage/v1/b/{quote(self._bucket, safe='')}/o?{urlencode(params)}"
            status, _, body = self._request("GET", url, self._headers())
            if status != 200:
                raise TrackingStoreError(f"tracking GCS listing returned HTTP {status}")
            try:
                record = json.loads(body)
                for item in record.get("items", []):
                    name = item["name"]
                    if not isinstance(name, str) or not name.startswith(object_prefix) or name.endswith("/"):
                        continue
                    key = name[len(root_prefix) :]
                    try:
                        normalized = validate_key(key)
                    except ValueError:
                        # GCS permits object names that are not tracking keys.
                        # Ignore each one without poisoning the shared listing.
                        continue
                    keys.add(normalized)
                page_token = record.get("nextPageToken", "")
                if not isinstance(page_token, str) or page_token in seen_tokens:
                    raise ValueError
            except (KeyError, AttributeError, TypeError, ValueError, UnicodeError):
                raise TrackingStoreError("tracking GCS listing returned an invalid response") from None
            if not page_token:
                return sorted(keys)
            seen_tokens.add(page_token)


__all__ = ("GcsTrackingStore",)
