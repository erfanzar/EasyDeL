# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Synchronous and asynchronous HTTP transports for Eyvan API clients."""

from __future__ import annotations

import asyncio
import json
import time
import typing as tp
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import httpx

from eyvan._internal.client_version import CLIENT_VERSION_HEADER, client_version
from eyvan._internal.json_types import JsonObject, is_json_object
from eyvan._internal.plane_tls import control_plane_httpx_options
from eyvan.sdk.client.endpoint import DEFAULT_CONTROL_PLANE_URL, ControlPlaneEndpoint
from eyvan.sdk.client.idempotency import IdempotencyKeyFactory
from eyvan.sdk.client.identity import ClientIdentity
from eyvan.sdk.client.retry import RetryPolicy
from eyvan.sdk.client.types import HeaderMap, QueryParams
from eyvan.sdk.team_credentials import plane_trust_httpx_options
from eyvan.server.local.auth import LOCAL_DEVELOPMENT_CREDENTIAL_REQUIRED_DETAIL


def _plane_httpx_options(url: str) -> dict[str, tp.Any]:
    """Return TLS options for ``url``: trust `eyvan login` pinned for it, else the head CA, else defaults."""

    return plane_trust_httpx_options(url) or control_plane_httpx_options()


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _local_credential_refusal_diagnosis(url: httpx.URL) -> str:
    """Explain a loopback credential refusal in terms of this machine's real state.

    The plane answered the exact refusal that means "I hold a development
    credential you did not present". That is not a permissions problem to
    debug blind: the cause is almost always a plane started under a different
    HOME -- a stray test daemon, a plane from another checkout -- or one that
    predates this machine's current credential. Say which of those this
    profile's own records support, so the operator stops the right process
    instead of uninstalling.
    """

    from eyvan.server.local.auth import DEFAULT_LOCAL_DEVELOPMENT_CREDENTIAL_PATH
    from eyvan.server.local.daemon import LocalDaemonError, LocalDaemonPaths, LocalDaemonTarget, read_owner_only_record

    endpoint = ControlPlaneEndpoint(str(url.copy_with(path="/", query=None, fragment=None)))
    record: object = None
    try:
        target = LocalDaemonTarget.from_endpoint(endpoint)
        if target is not None:
            record = read_owner_only_record(LocalDaemonPaths().record_path(target))
    except (LocalDaemonError, OSError):
        record = None
    credential = DEFAULT_LOCAL_DEVELOPMENT_CREDENTIAL_PATH
    credential_state = "present" if credential.expanduser().exists() else "absent"
    lines = [
        f"A control plane is listening at {endpoint.url} but refused this machine's local",
        f"development credential ({credential}, {credential_state}). That plane holds a",
        "credential of its own: typically one started under a different HOME -- a stray",
        "test daemon, or a plane from another checkout -- or one started before this",
        "credential was issued.",
    ]
    pid = getattr(record, "pid", None)
    if pid is not None:
        lines.append(f"This profile's PID record names pid {pid}: stop that process and retry.")
    else:
        lines.append("This profile has no PID record for that endpoint, so it did not start it;")
        lines.append("find that process (its host and port are above) and stop it, then retry.")
    return "\n".join(lines)


class EyvanHTTPError(httpx.HTTPStatusError):
    """An unsuccessful Eyvan response with its structured API detail preserved."""

    def __init__(self, response: httpx.Response, detail: str) -> None:
        self.status_code = response.status_code
        self.detail = detail
        message = f"Eyvan API returned {response.status_code} {response.reason_phrase}"
        if detail:
            message = f"{message}: {detail}"
        if response.status_code == 422 and "extra_forbidden" in detail:
            # Name what this is. Eyvan's records are extra="forbid" deliberately,
            # so a rejected unknown field means this client sent something the
            # control plane's models do not carry -- a version skew, not a
            # malformed request. Reported as a bare pydantic dump it reads like a
            # bug in the payload: `eyvan up` failed exactly this way against a
            # head one commit behind, with nothing naming the cause.
            message = (
                f"{message}\n\nThis is a version skew: the control plane rejected a field this client sent. "
                "Compare builds with 'eyvan capacity builds', then bring the plane up to this source with "
                "'eyvan runtime deploy'."
            )
        if (
            response.status_code == 401
            and detail == LOCAL_DEVELOPMENT_CREDENTIAL_REQUIRED_DETAIL
            and (response.request.url.host or "") in _LOOPBACK_HOSTS
        ):
            # Same discipline as the version-skew note above: this refusal has
            # exactly one meaning, and the machine already holds the evidence
            # for which plane is answering. A bare "401" sent operators to
            # re-login against a plane that would refuse every credential
            # except one from a HOME that no longer exists.
            message = f"{message}\n\n{_local_credential_refusal_diagnosis(response.request.url)}"
        super().__init__(message, request=response.request, response=response)


class EyvanUnavailableError(ConnectionError):
    """A request that could not reach the configured Eyvan control plane."""

    #: The mutation's idempotency key, attached by ``mutate`` when every retry
    #: failed: after an ambiguous admission the caller needs it to query or
    #: cancel instead of resubmitting blind.
    idempotency_key: str | None = None

    def __init__(self, endpoint: ControlPlaneEndpoint, method: str) -> None:
        self.endpoint = endpoint
        self.method = method.upper()
        message = f"Eyvan control plane is unavailable at {endpoint.url}."
        if endpoint.is_loopback:
            message = (
                f"{message} Start it with 'eyvan-server --log-level info' and keep it running while jobs are active."
            )
        else:
            message = f"{message} Check the configured control-plane URL and network access."
        super().__init__(message)


def _response_detail(response: httpx.Response) -> str:
    """Extract a stable message from an Eyvan JSON error response."""

    try:
        payload = response.json()
    except ValueError:
        return response.text.strip()
    if is_json_object(payload) and "detail" in payload:
        detail = payload["detail"]
        return detail if isinstance(detail, str) else json.dumps(detail, separators=(",", ":"), sort_keys=True)
    return response.text.strip()


def raise_for_status(response: httpx.Response) -> httpx.Response:
    """Raise a typed Eyvan error for unsuccessful responses and return successes."""

    if not response.is_success:
        raise EyvanHTTPError(response, _response_detail(response))
    return response


#: Methods a retry cannot duplicate the effect of.
_SAFE_METHODS = frozenset({"GET", "HEAD"})
_READ_DEADLINE: ContextVar[float | None] = ContextVar("eyvan_sdk_read_deadline", default=None)


@contextmanager
def read_deadline(deadline: float | None) -> Iterator[None]:
    """Bound SDK reads in this execution context, including observer progress reads."""

    previous = _READ_DEADLINE.get()
    if previous is not None:
        deadline = previous if deadline is None else min(previous, deadline)
    token = _READ_DEADLINE.set(deadline)
    try:
        yield
    finally:
        _READ_DEADLINE.reset(token)


def _remaining_read_time() -> float | None:
    deadline = _READ_DEADLINE.get()
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("SDK read deadline expired")
    return remaining


def _bounded_read_timeout(configured: httpx.Timeout, remaining: float) -> httpx.Timeout:
    """Cap every HTTPX phase without relaxing an injected client's stricter limits."""

    return httpx.Timeout(
        **{name: remaining if value is None else min(value, remaining) for name, value in configured.as_dict().items()}
    )


def _read_retry_delay(delay: float) -> float:
    remaining = _remaining_read_time()
    return delay if remaining is None else min(delay, remaining)


def _mutation_headers(
    headers: HeaderMap | None,
    idempotency_key: str | None,
    factory: IdempotencyKeyFactory,
) -> tuple[str, dict[str, str]]:
    """Prefer a caller header over the argument; collapse equal keys, reject conflicts."""

    # HTTPX's mapping view comma-joins duplicate fields: inspect individual
    # values so conflicting keys cannot silently become a different key.
    caller_items = headers.multi_items() if isinstance(headers, httpx.Headers) else list((headers or {}).items())
    supplied = [value for name, value in caller_items if name.lower() == "idempotency-key"]
    if supplied and any(value != supplied[0] for value in supplied[1:]):
        raise ValueError("conflicting Idempotency-Key header values")
    key = factory.create(supplied=supplied[0] if supplied else idempotency_key)
    outgoing = {name: value for name, value in dict(headers or {}).items() if name.lower() != "idempotency-key"}
    outgoing["Idempotency-Key"] = key
    return key, outgoing


def _merge_request_headers(injected: HeaderMap, headers: HeaderMap | None) -> HeaderMap | None:
    """Let a request key replace injected defaults regardless of casing."""

    caller_headers = dict(headers or {})
    overridden = {name.lower() for name in caller_headers}
    injected = {name: value for name, value in injected.items() if name.lower() not in overridden}
    merged = {**injected, **caller_headers}
    return merged or None


class SyncApiTransport:
    """Typed synchronous HTTP boundary with mutation retry handling."""

    def __init__(
        self,
        endpoint: str | ControlPlaneEndpoint = DEFAULT_CONTROL_PLANE_URL,
        *,
        identity: ClientIdentity | None = None,
        headers: HeaderMap | None = None,
        client: httpx.Client | None = None,
        trust_env: bool = True,
        retry_policy: RetryPolicy = RetryPolicy(),
        idempotency_factory: IdempotencyKeyFactory = IdempotencyKeyFactory(),
    ) -> None:
        self.endpoint = ControlPlaneEndpoint.parse(endpoint)
        self.retry_policy = retry_policy
        self.idempotency_factory = idempotency_factory
        client_headers = {
            CLIENT_VERSION_HEADER: client_version(),
            **(identity or ClientIdentity()).headers(),
            **dict(headers or {}),
        }
        if client is None:
            client = httpx.Client(
                base_url=self.endpoint.url,
                headers=client_headers,
                timeout=30,
                trust_env=trust_env,
                # A head's own CA, added to system trust; see eyvan._internal.plane_tls.
                **_plane_httpx_options(self.endpoint.url),
            )
            self._own_client = True
            self._injected_headers: HeaderMap = {}
        else:
            self._own_client = False
            self._injected_headers = client_headers
        self._client = client

    @property
    def client(self) -> httpx.Client:
        """Return the underlying HTTPX client for compatibility integrations."""

        return self._client

    def request(
        self,
        method: str,
        path: str,
        *,
        params: QueryParams | None = None,
        json: JsonObject | None = None,
        content: bytes | None = None,
        headers: HeaderMap | None = None,
    ) -> httpx.Response:
        """Send one request without applying status handling or retries."""

        remaining = _remaining_read_time() if method.upper() in _SAFE_METHODS else None
        options = {} if remaining is None else {"timeout": _bounded_read_timeout(self._client.timeout, remaining)}
        try:
            response = self._client.request(
                method,
                path,
                params=params,
                json=json,
                content=content,
                headers=self._request_headers(headers),
                **options,
            )
        except httpx.TransportError:
            if remaining is not None:
                _remaining_read_time()
            raise EyvanUnavailableError(self.endpoint, method) from None
        if remaining is not None:
            _remaining_read_time()
        return response

    def _request_headers(self, headers: HeaderMap | None) -> HeaderMap | None:
        """Merge identity headers onto caller headers without mutating injected clients."""

        return _merge_request_headers(self._injected_headers, headers)

    def request_checked(
        self,
        method: str,
        path: str,
        *,
        params: QueryParams | None = None,
        json: JsonObject | None = None,
        content: bytes | None = None,
        headers: HeaderMap | None = None,
    ) -> httpx.Response:
        """Send one request and require a successful response.

        A read is retried on the statuses a mutation is: the server answers a
        PostgreSQL serialization conflict with 503 and ``Retry-After``, and a
        read loses nothing by asking again. A transient conflict must not end
        a client's ``wait()`` while its workload is still running.
        """

        if method.upper() not in _SAFE_METHODS:
            return raise_for_status(
                self.request(method, path, params=params, json=json, content=content, headers=headers)
            )
        for attempt in range(self.retry_policy.max_attempts):
            response = self.request(method, path, params=params, json=json, content=content, headers=headers)
            delay = self.retry_policy.delay(status_code=response.status_code, headers=response.headers, attempt=attempt)
            if delay is None or attempt == self.retry_policy.max_attempts - 1:
                return raise_for_status(response)
            time.sleep(_read_retry_delay(delay))
        raise AssertionError("retry policy must allow at least one attempt")

    def mutate(
        self,
        method: str,
        path: str,
        *,
        idempotency_key: str | None = None,
        params: QueryParams | None = None,
        json: JsonObject | None = None,
        content: bytes | None = None,
        headers: HeaderMap | None = None,
    ) -> httpx.Response:
        """Send one mutation, retaining its idempotency key across retries."""

        key, request_headers = _mutation_headers(headers, idempotency_key, self.idempotency_factory)
        for attempt in range(self.retry_policy.max_attempts):
            try:
                response = self.request(
                    method,
                    path,
                    params=params,
                    json=json,
                    content=content,
                    headers=request_headers,
                )
            except EyvanUnavailableError as exc:
                if attempt == self.retry_policy.max_attempts - 1:
                    # The plane may have admitted the request before the
                    # response was lost; surface the key so the caller can
                    # query or cancel by it instead of resubmitting blind.
                    exc.idempotency_key = key
                    raise
                time.sleep(self.retry_policy.backoff(attempt))
                continue
            delay = self.retry_policy.delay(
                status_code=response.status_code,
                headers=response.headers,
                attempt=attempt,
            )
            if delay is None or attempt == self.retry_policy.max_attempts - 1:
                return raise_for_status(response)
            time.sleep(delay)
        raise AssertionError("retry policy must allow at least one attempt")

    def close(self) -> None:
        """Close the underlying synchronous HTTP client when owned by this transport."""

        if self._own_client:
            self._client.close()


class AsyncApiTransport:
    """Typed asynchronous HTTP boundary with mutation retry handling."""

    def __init__(
        self,
        endpoint: str | ControlPlaneEndpoint = DEFAULT_CONTROL_PLANE_URL,
        *,
        identity: ClientIdentity | None = None,
        headers: HeaderMap | None = None,
        client: httpx.AsyncClient | None = None,
        trust_env: bool = True,
        retry_policy: RetryPolicy = RetryPolicy(),
        idempotency_factory: IdempotencyKeyFactory = IdempotencyKeyFactory(),
    ) -> None:
        self.endpoint = ControlPlaneEndpoint.parse(endpoint)
        self.retry_policy = retry_policy
        self.idempotency_factory = idempotency_factory
        client_headers = {
            CLIENT_VERSION_HEADER: client_version(),
            **(identity or ClientIdentity()).headers(),
            **dict(headers or {}),
        }
        if client is None:
            client = httpx.AsyncClient(
                base_url=self.endpoint.url,
                headers=client_headers,
                timeout=30,
                trust_env=trust_env,
                # A head's own CA, added to system trust; see eyvan._internal.plane_tls.
                **_plane_httpx_options(self.endpoint.url),
            )
            self._own_client = True
            self._injected_headers: HeaderMap = {}
        else:
            self._own_client = False
            self._injected_headers = client_headers
        self._client = client

    @property
    def client(self) -> httpx.AsyncClient:
        """Return the underlying async HTTPX client for compatibility integrations."""

        return self._client

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: QueryParams | None = None,
        json: JsonObject | None = None,
        content: bytes | None = None,
        headers: HeaderMap | None = None,
    ) -> httpx.Response:
        """Send one request without applying status handling or retries."""

        remaining = _remaining_read_time() if method.upper() in _SAFE_METHODS else None
        options = {} if remaining is None else {"timeout": _bounded_read_timeout(self._client.timeout, remaining)}
        try:
            async with asyncio.timeout(remaining):
                response = await self._client.request(
                    method,
                    path,
                    params=params,
                    json=json,
                    content=content,
                    headers=self._request_headers(headers),
                    **options,
                )
        except httpx.TransportError:
            if remaining is not None:
                _remaining_read_time()
            raise EyvanUnavailableError(self.endpoint, method) from None
        if remaining is not None:
            _remaining_read_time()
        return response

    def _request_headers(self, headers: HeaderMap | None) -> HeaderMap | None:
        """Merge identity headers onto caller headers without mutating injected clients."""

        return _merge_request_headers(self._injected_headers, headers)

    async def request_checked(
        self,
        method: str,
        path: str,
        *,
        params: QueryParams | None = None,
        json: JsonObject | None = None,
        content: bytes | None = None,
        headers: HeaderMap | None = None,
    ) -> httpx.Response:
        """Send one request and require a successful response; reads retry as in the sync transport."""

        if method.upper() not in _SAFE_METHODS:
            response = await self.request(method, path, params=params, json=json, content=content, headers=headers)
            return raise_for_status(response)
        for attempt in range(self.retry_policy.max_attempts):
            response = await self.request(method, path, params=params, json=json, content=content, headers=headers)
            delay = self.retry_policy.delay(status_code=response.status_code, headers=response.headers, attempt=attempt)
            if delay is None or attempt == self.retry_policy.max_attempts - 1:
                return raise_for_status(response)
            await asyncio.sleep(_read_retry_delay(delay))
        raise AssertionError("retry policy must allow at least one attempt")

    async def mutate(
        self,
        method: str,
        path: str,
        *,
        idempotency_key: str | None = None,
        params: QueryParams | None = None,
        json: JsonObject | None = None,
        content: bytes | None = None,
        headers: HeaderMap | None = None,
    ) -> httpx.Response:
        """Send one mutation, retaining its idempotency key across retries."""

        key, request_headers = _mutation_headers(headers, idempotency_key, self.idempotency_factory)
        for attempt in range(self.retry_policy.max_attempts):
            try:
                response = await self.request(
                    method,
                    path,
                    params=params,
                    json=json,
                    content=content,
                    headers=request_headers,
                )
            except EyvanUnavailableError as exc:
                if attempt == self.retry_policy.max_attempts - 1:
                    # See the synchronous twin above.
                    exc.idempotency_key = key
                    raise
                await asyncio.sleep(self.retry_policy.backoff(attempt))
                continue
            delay = self.retry_policy.delay(
                status_code=response.status_code,
                headers=response.headers,
                attempt=attempt,
            )
            if delay is None or attempt == self.retry_policy.max_attempts - 1:
                return raise_for_status(response)
            await asyncio.sleep(delay)
        raise AssertionError("retry policy must allow at least one attempt")

    async def aclose(self) -> None:
        """Close the underlying asynchronous HTTP client when owned by this transport."""

        if self._own_client:
            await self._client.aclose()


__all__ = (
    "AsyncApiTransport",
    "EyvanHTTPError",
    "EyvanUnavailableError",
    "SyncApiTransport",
    "raise_for_status",
)
