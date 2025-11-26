# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""ZeroMQ client for communicating with auth worker process."""

from __future__ import annotations

import threading
import typing as tp

import zmq

from .auth_models import ApiKeyMetadata, ApiKeyPermissions, ApiKeyRole, ApiKeyStatus, QuotaConfig, RateLimitConfig


class PermissionDenied(Exception):
    """Raised when permission check fails."""

    pass


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    pass


class QuotaExceeded(Exception):
    """Raised when quota limit is exceeded."""

    pass


class AuthWorkerClient:
    """Client for communicating with auth worker process via ZMQ.

    Args:
            endpoint: The ZeroMQ endpoint of the auth worker.

    Raises:
            ValueError: If endpoint is not provided.
    """

    def __init__(self, endpoint: str):
        if not endpoint:
            raise ValueError("Auth worker endpoint must be provided.")
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(endpoint)
        self._lock = threading.Lock()
        self._endpoint = endpoint

    def _request(self, payload: dict) -> dict:
        """Send a request to the worker and return the response.

        Args:
                payload: The request payload to send.

        Returns:
                The response from the worker.

        Raises:
                RuntimeError: If the worker returns an error.
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
        """Generate a new API key.

        Args:
                name: Human-readable name for the key.
                role: Access control role.
                description: Optional description.
                created_by: User/service creating the key.
                expires_in_days: Number of days until expiration.
                rate_limits: Rate limiting configuration.
                quota: Usage quota configuration.
                permissions: Granular permissions.
                tags: List of tags for organization.
                metadata: Additional metadata.

        Returns:
                Tuple of (raw_key, metadata).
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
        """Validate a raw API key and return its metadata.

        Args:
                raw_key: The raw API key to validate.

        Returns:
                ApiKeyMetadata if valid, None otherwise.
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
        """Authorize a request and perform all security checks.

        Args:
                raw_key: Raw API key from the request.
                ip_address: Client IP address.
                endpoint: API endpoint being accessed.
                model: Model being requested.
                requested_tokens: Number of tokens being requested.

        Returns:
                ApiKeyMetadata if authorized.

        Raises:
                PermissionDenied: If authorization fails.
                RateLimitExceeded: If rate limit is exceeded.
                QuotaExceeded: If quota is exceeded.
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
        """Record token usage for a key.

        Args:
                raw_key: Raw API key.
                prompt_tokens: Number of prompt tokens used.
                completion_tokens: Number of completion tokens generated.
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
        """Revoke an API key.

        Args:
                key_id: ID of the key to revoke.
                revoked_by: User/service revoking the key.

        Returns:
                True if revoked, False if not found.
        """
        resp = self._request({"cmd": "revoke_key", "key_id": key_id, "revoked_by": revoked_by})
        return resp["success"]

    def suspend_key(self, key_id: str, suspended_by: str | None = None) -> bool:
        """Suspend an API key.

        Args:
                key_id: ID of the key to suspend.
                suspended_by: User/service suspending the key.

        Returns:
                True if suspended, False if not found.
        """
        resp = self._request({"cmd": "suspend_key", "key_id": key_id, "suspended_by": suspended_by})
        return resp["success"]

    def reactivate_key(self, key_id: str, reactivated_by: str | None = None) -> bool:
        """Reactivate a suspended API key.

        Args:
                key_id: ID of the key to reactivate.
                reactivated_by: User/service reactivating the key.

        Returns:
                True if reactivated, False if not found or revoked.
        """
        resp = self._request({"cmd": "reactivate_key", "key_id": key_id, "reactivated_by": reactivated_by})
        return resp["success"]

    def delete_key(self, key_id: str, deleted_by: str | None = None) -> bool:
        """Permanently delete an API key.

        Args:
                key_id: ID of the key to delete.
                deleted_by: User/service deleting the key.

        Returns:
                True if deleted, False if not found.
        """
        resp = self._request({"cmd": "delete_key", "key_id": key_id, "deleted_by": deleted_by})
        return resp["success"]

    def get_key_by_id(self, key_id: str) -> ApiKeyMetadata | None:
        """Get key metadata by key ID.

        Args:
                key_id: ID of the key.

        Returns:
                ApiKeyMetadata if found, None otherwise.
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
        """List API keys with optional filtering.

        Args:
                role: Filter by role.
                status: Filter by status.
                tags: Filter by tags.

        Returns:
                List of matching ApiKeyMetadata objects.
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
        """Update API key configuration.

        Args:
                key_id: ID of the key to update.
                name: New name.
                description: New description.
                role: New role.
                expires_in_days: New expiration.
                rate_limits: New rate limits.
                quota: New quota.
                permissions: New permissions.
                tags: New tags.
                metadata: New metadata.
                updated_by: User/service updating the key.

        Returns:
                True if updated, False if not found.
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
        """Rotate an API key.

        Args:
                key_id: ID of the key to rotate.
                rotated_by: User/service rotating the key.

        Returns:
                Tuple of (new_raw_key, metadata) if successful, None if not found.
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
        """Get audit log entries.

        Args:
                limit: Maximum number of entries to return.
                key_id: Filter by key ID.
                action: Filter by action type.

        Returns:
                List of audit log entry dicts.
        """
        resp = self._request({"cmd": "get_audit_logs", "limit": limit, "key_id": key_id, "action": action})
        return resp["logs"]

    def get_statistics(self) -> dict[str, tp.Any]:
        """Get overall statistics about API keys and usage.

        Returns:
                Dictionary with aggregate statistics.
        """
        resp = self._request({"cmd": "get_statistics"})
        return resp["statistics"]

    def shutdown(self) -> None:
        """Shutdown the auth worker and close the connection."""
        try:
            self._request({"cmd": "shutdown"})
        except Exception:
            pass
        finally:
            self.close()

    def close(self):
        """Close the ZeroMQ socket."""
        self._socket.close(0)

    @property
    def enabled(self) -> bool:
        """Check if auth worker is enabled (always True for worker client)."""
        return True

    def _deserialize_metadata(self, data: dict[str, tp.Any]) -> ApiKeyMetadata:
        """Deserialize metadata dict to ApiKeyMetadata object."""
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
