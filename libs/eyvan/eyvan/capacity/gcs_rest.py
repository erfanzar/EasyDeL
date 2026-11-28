# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reading and conditionally writing one GCS object without spawning gcloud.

Every fleet-state mutation is a compare-and-swap against a GCS object, and the
subprocess backend does it with three ``gcloud`` invocations -- describe, cat,
then a conditional copy. Interpreter startup adds CPU and latency to every
mutation, which accumulates as releases serialize on the document.

Same conditional semantics, over the JSON API: a media download reports the
object's generation in ``x-goog-generation``, and an upload takes
``ifGenerationMatch``. The credential comes from the path already established
for the other API readers here -- one ``gcloud auth print-access-token``,
inheriting whatever login the operator has.

**The subprocess backend stays.** It works where a raw HTTPS call may not, and a
caller that cannot mint a token falls back to it rather than failing.
"""

from __future__ import annotations

import json
import typing as tp
from dataclasses import dataclass, field

from eyvan._internal.json_types import JsonObject


def _default_token() -> str:
    """Mint a token via the shared gcloud helper, imported on use.

    Deferred because the helper lives under ``capacity.providers.gcp``, which
    imports this module's package -- taking it at module scope makes
    ``fleet_state`` a partially initialised import for anything that reaches it
    through a provider.
    """

    from eyvan.capacity.providers.gcp.accelerator_catalog import gcloud_access_token

    return gcloud_access_token()


#: Bound on one request. Generous because a conditional write that times out is
#: indeterminate and costs a reconciliation to resolve, tight enough that a hung
#: socket cannot hold a fleet lease open.
REQUEST_TIMEOUT_S = 30.0

#: Answers that mean "this credential is no longer good", as opposed to
#: "this object is not yours" -- both are 401/403 on the wire, and retrying once
#: with a fresh token tells them apart without a second way to be wrong.
_AUTH_FAILURES = frozenset({401, 403})

_OBJECT = "https://storage.googleapis.com/storage/v1/b/{bucket}/o/{object}"
_UPLOAD = "https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o"


class GcsRestError(RuntimeError):
    """The object could not be read or written over the GCS JSON API."""


class GcsPreconditionFailed(GcsRestError):
    """The object changed since it was read, so the conditional write was refused."""


def split_uri(uri: str) -> tuple[str, str]:
    """Split ``gs://bucket/path/to/object`` into its bucket and object name."""

    if not uri.startswith("gs://") or uri.endswith("/"):
        raise ValueError("fleet state URI must be a gs:// object path")
    bucket, _, name = uri[len("gs://") :].partition("/")
    if not bucket or not name:
        raise ValueError("fleet state URI must name a bucket and an object")
    return bucket, name


@dataclass
class GcsRestObject:
    """One GCS object, read and conditionally written over the JSON API.

    Attributes:
        uri: The ``gs://`` object path.
        token_source: Where the bearer credential comes from. Injected so a test
            never shells out and a caller can supply its own.
    """

    uri: str
    token_source: tp.Callable[[], str] = _default_token
    _token: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.bucket, self.name = split_uri(self.uri)

    def _headers(self, *, refresh: bool = False) -> dict[str, str]:
        """Return the bearer header, minting a token when there is none or on demand.

        ``refresh`` is what makes a long-lived process work at all. A Google
        access token lasts about an hour; caching one for the life of the
        process means every call after that hour is a 401, which this reports as
        a plain read failure -- and the caller then falls back to a subprocess.
        """

        if refresh or self._token is None:
            # Deferred for the same provider import-cycle reason as _default_token.
            from eyvan.capacity.providers.gcp.accelerator_catalog import AcceleratorCatalogError

            try:
                self._token = self.token_source()
            except AcceleratorCatalogError:
                # Token subprocess diagnostics can contain credential material.
                # Preserve per-call fallback without forwarding those diagnostics.
                raise GcsRestError("could not mint a GCS access token") from None
        return {"Authorization": f"Bearer {self._token}"}

    def read(self) -> tuple[JsonObject | None, int]:
        """Return the object's parsed body and generation, or ``(None, 0)`` if absent.

        Generation zero is GCS's own spelling for "this object does not exist",
        and is what ``ifGenerationMatch`` takes to mean "create only if still
        absent" -- so the missing case needs no separate signalling.
        """

        from urllib.parse import quote

        import httpx

        url = _OBJECT.format(bucket=self.bucket, object=quote(self.name, safe=""))
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT_S, trust_env=False) as client:
                response = client.get(url, params={"alt": "media"}, headers=self._headers())
                if response.status_code in _AUTH_FAILURES:
                    response = client.get(url, params={"alt": "media"}, headers=self._headers(refresh=True))
        except httpx.HTTPError as exc:
            raise GcsRestError(f"could not read {self.uri}: {exc}") from exc
        if response.status_code == 404:
            return None, 0
        if response.status_code != 200:
            raise GcsRestError(f"reading {self.uri} answered {response.status_code}")
        generation = response.headers.get("x-goog-generation")
        if generation is None or not generation.isdigit():
            # Without a generation there is nothing to compare and swap against,
            # and writing unconditionally would overwrite a concurrent mutation.
            raise GcsRestError(f"{self.uri} was read without a generation header")
        try:
            body = response.json()
        except ValueError as exc:
            raise GcsRestError(f"{self.uri} is not valid JSON") from exc
        if not isinstance(body, dict):
            raise GcsRestError(f"{self.uri} is not a JSON object")
        return tp.cast("JsonObject", body), int(generation)

    def write(self, document: JsonObject, generation: int) -> None:
        """Replace the object only while it is still at ``generation``.

        Raises:
            GcsPreconditionFailed: Something else wrote first. The caller re-reads
                and reapplies its mutation; that is the compare-and-swap.
            GcsRestError: The write could not be attempted or its outcome is not
                known.
        """

        import httpx

        payload = json.dumps(document, indent=2, sort_keys=True).encode()
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT_S, trust_env=False) as client:
                params = {
                    "uploadType": "media",
                    "name": self.name,
                    "ifGenerationMatch": str(generation),
                }
                url = _UPLOAD.format(bucket=self.bucket)
                response = client.post(
                    url,
                    params=params,
                    headers={**self._headers(), "Content-Type": "application/json"},
                    content=payload,
                )
                if response.status_code in _AUTH_FAILURES:
                    # Safe to repeat: ifGenerationMatch makes the write
                    # idempotent -- if the first attempt somehow landed, the
                    # second is refused as a precondition failure, which the
                    # caller already handles by re-reading.
                    response = client.post(
                        url,
                        params=params,
                        headers={**self._headers(refresh=True), "Content-Type": "application/json"},
                        content=payload,
                    )
        except httpx.HTTPError as exc:
            raise GcsRestError(f"could not write {self.uri}: {exc}") from exc
        if response.status_code == 412:
            raise GcsPreconditionFailed(f"{self.uri} changed since it was read at generation {generation}")
        if response.status_code not in {200, 201}:
            raise GcsRestError(f"writing {self.uri} answered {response.status_code}")


__all__ = ("REQUEST_TIMEOUT_S", "GcsPreconditionFailed", "GcsRestError", "GcsRestObject", "split_uri")
