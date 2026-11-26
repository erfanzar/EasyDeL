# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Click-facing adapter over Eyvan's shared synchronous HTTP transport."""

from __future__ import annotations

import json
import typing as tp

import click
import httpx

from eyvan._internal.json_types import JsonObject, JsonValue, is_json_object, is_json_value
from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
from eyvan.sdk.client.transport import EyvanUnavailableError, SyncApiTransport
from eyvan.sdk.client.types import HeaderMap, QueryParams

_MUTATING_METHODS = frozenset({"DELETE", "PATCH", "POST", "PUT"})


class EyvanApiError(click.ClickException):
    """An HTTP API failure with its status available to Click commands."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        super().__init__(f"{status_code}: {detail}")


class EyvanCliUnavailableError(EyvanUnavailableError, click.ClickException):
    """An unreachable control plane, carried in the shape Click commands report.

    Subclasses both on purpose. Commands that give an unreachable plane its own
    diagnosis -- every ``doctor`` check does -- catch ``EyvanUnavailableError``;
    Click's top-level handling sees a ``ClickException`` and prints one clean
    ``Error:`` line instead of a traceback. Wrapping the error as a plain
    ``ClickException`` here used to break the first half of that: the typed
    handlers could never fire, because the exception arriving at them was not
    an ``EyvanUnavailableError`` at all.
    """

    def __init__(self, endpoint: ControlPlaneEndpoint, method: str) -> None:
        EyvanUnavailableError.__init__(self, endpoint, method)
        # ``ClickException.__init__`` sets the state Click's own reporting
        # reads (``.message``, ``.show_color``) and the cooperative call above
        # never reaches it; the message text is already in ``args``.
        click.ClickException.__init__(self, self.args[0])


def _name_a_version_skew(status_code: int, detail: object) -> str | None:
    """Say "your CLI is newer than your plane" when that is what happened.

    FastAPI answers a field it does not know with 422 and
    ``extra_forbidden``, which is exactly right and completely unhelpful: the
    operator reads that ``dependencies`` are not permitted, goes looking for the
    flag they used, and finds it documented and working. The cause is never in
    the request -- it is that the plane predates the field.

    Only for that one shape. A 422 naming a field the caller really did get
    wrong must keep saying so, because most of them are.
    """

    if status_code != 422 or not isinstance(detail, list):
        return None
    unknown = [
        ".".join(str(part) for part in item.get("loc", [])[1:])
        for item in detail
        if isinstance(item, dict) and item.get("type") == "extra_forbidden"
    ]
    if not unknown:
        return None
    fields = ", ".join(name for name in unknown if name) or "a field"
    return (
        f"the control plane does not know {fields}, which this client sends. That is a version "
        "skew rather than a bad request: the plane is older than this CLI. Upgrade it -- "
        "`eyvan runtime deploy --yes` for a local plane, or redeploy the head's wheel -- "
        "or use a client matching the plane"
    )


class EyvanClient:
    """JSON CLI adapter that delegates HTTP behavior to ``SyncApiTransport``."""

    def __init__(self, base_url: str, *, headers: HeaderMap):
        endpoint = ControlPlaneEndpoint.parse(base_url)
        self._transport = SyncApiTransport(
            endpoint,
            headers=headers,
            trust_env=not endpoint.is_loopback,
        )

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        detail = response.text
        try:
            payload = response.json()
        except ValueError:
            return detail
        if is_json_object(payload):
            api_detail = payload.get("detail")
            if api_detail is not None:
                return _name_a_version_skew(response.status_code, api_detail) or str(api_detail)
        return detail

    @staticmethod
    def _json(response: httpx.Response) -> JsonValue:
        if response.status_code == httpx.codes.NO_CONTENT:
            # 204 is what a successful DELETE returns, and an empty body is the
            # whole point of it. Parsing it as JSON reported every successful
            # deletion as `Error: 204: ... non-JSON response body: ''` and
            # exited 1 -- so `eyvan quota delete` removed the quota and then
            # told the operator, and any script wrapping it, that it had failed.
            return None
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            excerpt = " ".join(response.text.split())[:200]
            raise EyvanApiError(
                response.status_code,
                f"control plane returned a non-JSON response body: {excerpt!r}",
            ) from exc
        if not is_json_value(payload):
            raise EyvanApiError(response.status_code, "control plane returned a non-JSON response value")
        return payload

    def _response(
        self,
        method: str,
        path: str,
        *,
        params: QueryParams | None,
        json: JsonObject | None,
        headers: HeaderMap | None,
        content: bytes | None = None,
    ) -> httpx.Response:
        request_headers = dict(headers or {})
        idempotency_key = request_headers.pop("Idempotency-Key", None)
        try:
            if method.upper() in _MUTATING_METHODS:
                return self._transport.mutate(
                    method,
                    path,
                    idempotency_key=idempotency_key,
                    params=params,
                    json=json,
                    content=content,
                    headers=request_headers,
                )
            return self._transport.request_checked(
                method,
                path,
                params=params,
                json=json,
                headers=request_headers,
            )
        except httpx.HTTPStatusError as exc:
            response = exc.response
            raise EyvanApiError(response.status_code, self._error_detail(response)) from exc
        except EyvanUnavailableError as exc:
            # Re-raised as itself-in-subclass-form, not swallowed: the message
            # Click prints is unchanged, but a command that distinguishes
            # "the plane is gone" from any other failure can now catch it.
            raise EyvanCliUnavailableError(exc.endpoint, exc.method) from exc

    def request(
        self,
        method: str,
        path: str,
        *,
        params: QueryParams | None = None,
        json: JsonObject | None = None,
        content: bytes | None = None,
        headers: HeaderMap | None = None,
    ) -> JsonValue:
        """Send one JSON API request through the shared transport.

        ``content`` carries a raw body for the content-addressed artifact routes,
        which take bytes rather than JSON.
        """

        return self._json(self._response(method, path, params=params, json=json, content=content, headers=headers))

    def replace_identity_headers(self, headers: HeaderMap) -> None:
        """Replace only Eyvan authentication headers on the persistent HTTP client."""

        identity_headers = (
            "Authorization",
            "X-Eyvan-Local-Credential",
            "X-Eyvan-Org",
            "X-Eyvan-Project",
            "X-Eyvan-Roles",
            "X-Eyvan-User",
        )
        for name in identity_headers:
            self._transport.client.headers.pop(name, None)
        self._transport.client.headers.update(headers)

    def request_text(
        self,
        method: str,
        path: str,
        *,
        params: QueryParams | None = None,
        headers: HeaderMap | None = None,
    ) -> str:
        """Send one API request and return its response body."""

        return self._response(method, path, params=params, json=None, headers=headers).text

    def request_bytes(
        self,
        method: str,
        path: str,
        *,
        params: QueryParams | None = None,
        headers: HeaderMap | None = None,
    ) -> tuple[bytes, str]:
        """Send one API request and return its raw body and media type (for blob downloads)."""

        response = self._response(method, path, params=params, json=None, headers=headers)
        media_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        return response.content, media_type

    def stream_lines(
        self,
        method: str,
        path: str,
        *,
        params: QueryParams | None = None,
        headers: HeaderMap | None = None,
        json: JsonObject | None = None,
    ) -> tp.Iterator[str]:
        """Yield response lines as they arrive, for server-sent event and NDJSON streams.

        A generator rather than a callback: the caller decides when to stop, and
        the ``with`` block closes the connection on the way out however the
        iteration ended -- including a ``KeyboardInterrupt``, which is how anyone
        actually exits ``--follow``.
        """

        client = self._transport.client
        request_headers = dict(headers or {})
        try:
            with client.stream(
                method,
                path,
                params=dict(params or {}),
                headers=request_headers,
                json=json,
                timeout=None,
            ) as response:
                if response.status_code >= 400:
                    response.read()
                    raise EyvanApiError(response.status_code, self._error_detail(response))
                yield from response.iter_lines()
        except httpx.HTTPError as exc:
            raise click.ClickException(f"event stream failed: {exc}") from exc


__all__ = ("EyvanApiError", "EyvanCliUnavailableError", "EyvanClient")
