# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ZeroMQ REQ-socket client for the eSurge auth worker subprocess.

Provides :class:`AuthWorkerClient` plus thin client-side mirrors of the
worker's domain exceptions (:class:`PermissionDenied`,
:class:`RateLimitExceeded`, :class:`QuotaExceeded`). The client is
designed to be created once per FastAPI worker and shared across
request handlers; every method serialises send/recv behind a per-client
lock so concurrent calls remain wire-safe.

Most methods deserialize the worker's response into an
:class:`ApiKeyMetadata` (or list thereof) using
:meth:`AuthWorkerClient._deserialize_metadata`, which deliberately
omits the ``hashed_key`` field because the worker does not transmit
secret-derived material over the socket.
"""

from __future__ import annotations

import threading
import typing as tp

import zmq

from .auth_models import ApiKeyMetadata, ApiKeyPermissions, ApiKeyRole, ApiKeyStatus, QuotaConfig, RateLimitConfig


class PermissionDenied(Exception):
    """Client-side mirror of the auth worker's ``PermissionDenied``.

    Re-raised by :meth:`AuthWorkerClient._request` whenever the worker
    returns ``status == "error"`` with ``exception_type == "PermissionDenied"``,
    so callers can distinguish authorisation failures from rate-limit
    and quota errors using a normal ``except`` clause without touching
    the worker's internal exception classes.
    """

    pass


class RateLimitExceeded(Exception):
    """Client-side mirror of the auth worker's ``RateLimitExceeded``.

    Re-raised by :meth:`AuthWorkerClient._request` when the worker
    reports a rate-limit breach. The message string mirrors the
    server-side message and identifies the window
    (e.g. ``"... 60 requests/minute"``).
    """

    pass


class QuotaExceeded(Exception):
    """Client-side mirror of the auth worker's ``QuotaExceeded``.

    Re-raised by :meth:`AuthWorkerClient._request` when the worker
    detects that a lifetime or monthly quota would be breached. Use this
    to surface 429/402-style errors back to API callers.
    """

    pass


class AuthWorkerClient:
    """Thread-safe ZeroMQ REQ client for the eSurge auth worker.

    Wraps a single REQ socket plus a ``threading.Lock`` so multiple FastAPI
    request handlers can share one client instance without interleaving
    sends and receives. Each public method maps 1:1 to a command consumed
    by ``worker_main.py`` and re-raises the worker's domain exceptions
    (:class:`PermissionDenied`, :class:`RateLimitExceeded`,
    :class:`QuotaExceeded`) on the calling thread.

    Returned :class:`ApiKeyMetadata` objects are reconstructed from the
    wire payload by :meth:`_deserialize_metadata`; the raw ``hashed_key``
    field is *not* sent over the socket and is left empty client-side.
    """

    def __init__(self, endpoint: str):
        """Connect a REQ socket to the auth worker at ``endpoint``.

        Args:
            endpoint: ZeroMQ endpoint URI. Must be non-empty.

        Raises:
            ValueError: If ``endpoint`` is the empty string or ``None``.
        """
        if not endpoint:
            raise ValueError("Auth worker endpoint must be provided.")
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(endpoint)
        self._lock = threading.Lock()
        self._endpoint = endpoint

    def _request(self, payload: dict) -> dict:
        """Send one REQ/REP round-trip and translate worker errors.

        Serialises ``payload`` with :meth:`zmq.Socket.send_pyobj`, blocks
        for the reply, and inspects its ``status`` field. Errors carrying
        the special ``exception_type`` markers are re-raised as the
        matching client-side mirror class so callers see the same
        exception hierarchy as if they were using
        :class:`EnhancedApiKeyManager` directly.

        Args:
            payload: Command dictionary; must include a ``cmd`` key.

        Returns:
            dict: The full worker response (always contains ``status``).

        Raises:
            PermissionDenied: Worker reported a permission failure.
            RateLimitExceeded: Worker reported a rate-limit breach.
            QuotaExceeded: Worker reported a quota breach.
            RuntimeError: Any other ``status="error"`` reply.
        """
        with self._lock:
            self._socket.send_pyobj(payload)
            resp = self._socket.recv_pyobj()
            if resp.get("status") == "error":
                # Re-raise specific exceptions
                exception_type = resp.get("exception_type")
                message = resp.get("message", "Auth worker failed")
                if exception_type == "PermissionDenied":
                    raise PermissionDenied(message)
                elif exception_type == "RateLimitExceeded":
                    raise RateLimitExceeded(message)
                elif exception_type == "QuotaExceeded":
                    raise QuotaExceeded(message)
                else:
                    raise RuntimeError(message)
            return resp

    def generate_api_key(
        self,
        name: str,
        role: ApiKeyRole = ApiKeyRole.USER,
        description: str | None = None,
        created_by: str | None = None,
        expires_in_days: int | None = None,
        rate_limits: RateLimitConfig | None = None,
        quota: QuotaConfig | None = None,
        permissions: ApiKeyPermissions | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, tp.Any] | None = None,
    ) -> tuple[str, ApiKeyMetadata]:
        """Ask the worker to mint a fresh ``sk-...`` API key.

        Forwards to ``EnhancedApiKeyManager.generate_api_key`` on the
        worker side; the returned raw key is the only point at which
        the secret is observable.

        Args:
            name: Human-readable label.
            role: Access control role. Defaults to :data:`ApiKeyRole.USER`.
            description: Optional free-form description.
            created_by: Creator identifier stored on the metadata.
            expires_in_days: Optional TTL in days; ``None`` for no expiry.
            rate_limits: Optional :class:`RateLimitConfig` overriding
                the default open-ended limits.
            quota: Optional :class:`QuotaConfig` overriding the default
                open-ended quota.
            permissions: Optional :class:`ApiKeyPermissions`.
            tags: Optional organizational tags.
            metadata: Optional arbitrary user-defined metadata payload.

        Returns:
            tuple[str, ApiKeyMetadata]: ``(raw_key, metadata)``; the raw
            key must be returned to the caller immediately and never
            logged.
        """
        resp = self._request(
            {
                "cmd": "generate_api_key",
                "name": name,
                "role": role,
                "description": description,
                "created_by": created_by,
                "expires_in_days": expires_in_days,
                "rate_limits": rate_limits,
                "quota": quota,
                "permissions": permissions,
                "tags": tags,
                "metadata": metadata,
            }
        )
        return resp["raw_key"], self._deserialize_metadata(resp["metadata"])

    def validate_key(self, raw_key: str | None) -> ApiKeyMetadata | None:
        """Resolve a raw key to live metadata or ``None`` for invalid keys.

        Args:
            raw_key: Bearer token presented by the client. ``None`` and
                the empty string round-trip and reliably return ``None``.

        Returns:
            ApiKeyMetadata | None: Active metadata when the key is valid
            and not revoked / suspended / expired; ``None`` otherwise.
        """
        resp = self._request({"cmd": "validate_key", "raw_key": raw_key})
        metadata_dict = resp.get("metadata")
        return self._deserialize_metadata(metadata_dict) if metadata_dict else None

    def authorize_request(
        self,
        raw_key: str | None,
        ip_address: str | None = None,
        endpoint: str | None = None,
        model: str | None = None,
        requested_tokens: int = 0,
    ) -> ApiKeyMetadata:
        """Run the worker's full authorisation pipeline for a request.

        Forwards to ``EnhancedApiKeyManager.authorize_request`` on the
        worker; rejections from any stage (validity, IP, endpoint /
        model permissions, rate limits, quotas, per-request token
        ceiling) are re-raised locally as the matching mirror exception.

        Args:
            raw_key: Bearer token from the client. ``None`` / empty
                always fails with :class:`PermissionDenied`.
            ip_address: Client IP for allow/blocklist enforcement.
            endpoint: Path being accessed; checked against the key's
                ``allowed_endpoints``.
            model: Model name being requested; checked against the
                key's ``allowed_models``.
            requested_tokens: Projected token cost; checked against
                both per-request ceiling and rate limit / quota windows.

        Returns:
            ApiKeyMetadata: Live metadata for the authorising key with
            ``last_used_at`` already refreshed by the worker.

        Raises:
            PermissionDenied: For invalid keys or permission failures.
            RateLimitExceeded: When a sliding-window limit fires.
            QuotaExceeded: When a cumulative quota is breached.
        """
        resp = self._request(
            {
                "cmd": "authorize_request",
                "raw_key": raw_key,
                "ip_address": ip_address,
                "endpoint": endpoint,
                "model": model,
                "requested_tokens": requested_tokens,
            }
        )
        return self._deserialize_metadata(resp["metadata"])

    def record_usage(
        self,
        raw_key: str | None,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Tell the worker to bump per-key counters after a served request.

        Args:
            raw_key: Raw API key used for the served request.
            prompt_tokens: Input tokens consumed.
            completion_tokens: Output tokens generated.
        """
        self._request(
            {
                "cmd": "record_usage",
                "raw_key": raw_key,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        )

    def revoke_key(self, key_id: str, revoked_by: str | None = None) -> bool:
        """Permanently disable a key via the worker.

        Revocation is irreversible — :meth:`reactivate_key` only works
        on suspended keys.

        Args:
            key_id: Internal key identifier (``"key_..."``).
            revoked_by: Optional actor recorded in the worker's audit log.

        Returns:
            bool: ``True`` when a key with that id existed and was
            revoked; ``False`` otherwise.
        """
        resp = self._request({"cmd": "revoke_key", "key_id": key_id, "revoked_by": revoked_by})
        return resp["success"]

    def suspend_key(self, key_id: str, suspended_by: str | None = None) -> bool:
        """Temporarily disable a key via the worker.

        Reversible with :meth:`reactivate_key`.

        Args:
            key_id: Internal key identifier.
            suspended_by: Optional actor recorded in the worker's audit log.

        Returns:
            bool: ``True`` when the key existed and was suspended.
        """
        resp = self._request({"cmd": "suspend_key", "key_id": key_id, "suspended_by": suspended_by})
        return resp["success"]

    def reactivate_key(self, key_id: str, reactivated_by: str | None = None) -> bool:
        """Move a suspended key back to active state via the worker.

        Refuses to act on revoked or unknown keys.

        Args:
            key_id: Internal key identifier.
            reactivated_by: Optional actor recorded in the worker's audit log.

        Returns:
            bool: ``True`` when the key existed, was suspended (not
            revoked) and is now active.
        """
        resp = self._request({"cmd": "reactivate_key", "key_id": key_id, "reactivated_by": reactivated_by})
        return resp["success"]

    def delete_key(self, key_id: str, deleted_by: str | None = None) -> bool:
        """Hard-delete a key from the worker store.

        Unlike :meth:`revoke_key`, deletion drops the metadata record
        and clears the worker's rate-limit windows for the key.

        Args:
            key_id: Internal key identifier.
            deleted_by: Optional actor recorded in the worker's audit log.

        Returns:
            bool: ``True`` when the key existed and was removed.
        """
        resp = self._request({"cmd": "delete_key", "key_id": key_id, "deleted_by": deleted_by})
        return resp["success"]

    def get_key_by_id(self, key_id: str) -> ApiKeyMetadata | None:
        """Fetch a key's metadata record from the worker.

        Args:
            key_id: Internal key identifier (``"key_..."``).

        Returns:
            ApiKeyMetadata | None: Reconstructed metadata when the key
            exists; ``None`` when no record is found.
        """
        resp = self._request({"cmd": "get_key_by_id", "key_id": key_id})
        metadata_dict = resp.get("metadata")
        return self._deserialize_metadata(metadata_dict) if metadata_dict else None

    def list_keys(
        self,
        role: ApiKeyRole | None = None,
        status: ApiKeyStatus | None = None,
        tags: list[str] | None = None,
    ) -> list[ApiKeyMetadata]:
        """List managed keys, optionally filtered by role / status / tags.

        Filters are AND-ed together on the worker side; tag filtering
        requires every requested tag to be present on the candidate.

        Args:
            role: Restrict to keys with this :class:`ApiKeyRole`.
            status: Restrict to keys with this :class:`ApiKeyStatus`.
            tags: Restrict to keys whose ``tags`` are a superset of
                this iterable.

        Returns:
            list[ApiKeyMetadata]: Matching key records in worker order.
        """
        resp = self._request({"cmd": "list_keys", "role": role, "status": status, "tags": tags})
        return [self._deserialize_metadata(k) for k in resp["keys"]]

    def update_key(
        self,
        key_id: str,
        name: str | None = None,
        description: str | None = None,
        role: ApiKeyRole | None = None,
        expires_in_days: int | None = None,
        rate_limits: RateLimitConfig | None = None,
        quota: QuotaConfig | None = None,
        permissions: ApiKeyPermissions | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, tp.Any] | None = None,
        updated_by: str | None = None,
    ) -> bool:
        """Apply a partial update to a managed key via the worker.

        ``None`` arguments are treated as "leave alone"; non-``None``
        values overwrite the corresponding field on the worker.
        ``metadata`` is merged into the existing dict rather than
        replacing it.

        Args:
            key_id: Internal key identifier to update.
            name: New display name.
            description: New description.
            role: New :class:`ApiKeyRole`.
            expires_in_days: New TTL in days (converted to an absolute
                timestamp on the worker side).
            rate_limits: New :class:`RateLimitConfig` (full replacement).
            quota: New :class:`QuotaConfig` (full replacement).
            permissions: New :class:`ApiKeyPermissions` (full replacement).
            tags: New tag list (full replacement).
            metadata: User-metadata patch merged into the existing dict.
            updated_by: Optional actor recorded in the worker audit log.

        Returns:
            bool: ``True`` when the key existed and was updated.
        """
        resp = self._request(
            {
                "cmd": "update_key",
                "key_id": key_id,
                "name": name,
                "description": description,
                "role": role,
                "expires_in_days": expires_in_days,
                "rate_limits": rate_limits,
                "quota": quota,
                "permissions": permissions,
                "tags": tags,
                "metadata": metadata,
                "updated_by": updated_by,
            }
        )
        return resp["success"]

    def rotate_key(self, key_id: str, rotated_by: str | None = None) -> tuple[str, ApiKeyMetadata] | None:
        """Issue a fresh secret for an existing key while preserving metadata.

        Usage counters, permissions, audit history and key identifier
        are all preserved on the worker side.

        Args:
            key_id: Internal key identifier to rotate.
            rotated_by: Optional actor recorded in the audit log.

        Returns:
            tuple[str, ApiKeyMetadata] | None: ``(new_raw_key, metadata)``
            on success; ``None`` when no key with that id is registered.
            The new raw key must be returned to the caller immediately.
        """
        resp = self._request({"cmd": "rotate_key", "key_id": key_id, "rotated_by": rotated_by})
        if resp.get("status") == "ok" and "raw_key" in resp:
            return resp["raw_key"], self._deserialize_metadata(resp["metadata"])
        return None

    def get_audit_logs(
        self,
        limit: int = 100,
        key_id: str | None = None,
        action: str | None = None,
    ) -> list[dict]:
        """Fetch newest-first audit-log entries from the worker.

        Args:
            limit: Maximum number of entries to return after filtering.
            key_id: When set, restrict to entries for this key id.
            action: When set, restrict to entries with exactly this
                ``action`` slug (e.g. ``"request_authorized"``).

        Returns:
            list[dict]: Raw audit-log dicts ordered newest-first; the
            client does not deserialise them into
            :class:`AuditLogEntry` so consumers can inspect or stream
            them directly.
        """
        resp = self._request({"cmd": "get_audit_logs", "limit": limit, "key_id": key_id, "action": action})
        return resp["logs"]

    def get_statistics(self) -> dict[str, tp.Any]:
        """Fetch aggregate auth statistics from the worker.

        Returns:
            dict[str, tp.Any]: The same blob produced by
            :meth:`EnhancedApiKeyManager.get_statistics`, containing
            lifecycle counts (active / suspended / revoked / expired),
            cumulative request / token totals, per-role breakdown and
            audit-log size.
        """
        resp = self._request({"cmd": "get_statistics"})
        return resp["statistics"]

    def shutdown(self) -> None:
        """Send ``shutdown`` to the worker then close the local socket.

        Errors from the round-trip (e.g. the worker has already exited)
        are swallowed so the local socket is always released.
        """
        try:
            self._request({"cmd": "shutdown"})
        except Exception:
            pass
        finally:
            self.close()

    def close(self):
        """Close the local REQ socket without notifying the worker.

        Used by :meth:`AuthWorkerManager.shutdown` in the attached
        (non-owning) mode where terminating the upstream worker is the
        caller's responsibility.
        """
        self._socket.close(0)

    @property
    def enabled(self) -> bool:
        """Whether the client is active.

        Returns:
            bool: Always ``True`` for this worker client; provided for
            interface compatibility with no-op stubs used when auth is
            disabled.
        """
        return True

    def _deserialize_metadata(self, data: dict[str, tp.Any]) -> ApiKeyMetadata:
        """Rebuild an :class:`ApiKeyMetadata` from a wire-format dict.

        Args:
            data: The mapping returned by ``worker_main.py`` for any
                command that yields a key record. Sensitive fields like
                ``hashed_key`` are not sent over the wire.

        Returns:
            ApiKeyMetadata: A reconstructed dataclass with nested rate
            limit / quota / permissions sub-dataclasses populated.
        """
        return ApiKeyMetadata(
            key_id=data["key_id"],
            key_prefix=data["key_prefix"],
            hashed_key=data.get("hashed_key", ""),  # Not sent over wire
            name=data["name"],
            description=data.get("description"),
            role=ApiKeyRole(data["role"]),
            status=ApiKeyStatus(data["status"]),
            created_at=data["created_at"],
            created_by=data.get("created_by"),
            expires_at=data.get("expires_at"),
            last_used_at=data.get("last_used_at"),
            last_rotated_at=data.get("last_rotated_at"),
            total_requests=data.get("total_requests", 0),
            total_prompt_tokens=data.get("total_prompt_tokens", 0),
            total_completion_tokens=data.get("total_completion_tokens", 0),
            monthly_requests=data.get("monthly_requests", 0),
            monthly_tokens=data.get("monthly_tokens", 0),
            last_reset_month=data.get("last_reset_month", 0),
            rate_limits=RateLimitConfig(**data.get("rate_limits", {})),
            quota=QuotaConfig(**data.get("quota", {})),
            permissions=ApiKeyPermissions(**data.get("permissions", {})),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )
