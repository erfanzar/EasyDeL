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

"""Enhanced authentication models and schemas for eSurge API server."""

from __future__ import annotations

import time
import typing as tp
from dataclasses import dataclass, field
from enum import StrEnum


class ApiKeyRole(StrEnum):
    """Role-based access control levels for API keys.

    Attributes:
        ADMIN: Full access including key management operations.
        USER: Standard access to inference endpoints.
        READONLY: Read-only access (metrics, health, list models).
        SERVICE: Service account with specific custom permissions.
    """

    ADMIN = "admin"  # Full access including key management
    USER = "user"  # Standard access to inference endpoints
    READONLY = "readonly"  # Read-only access (metrics, health, list models)
    SERVICE = "service"  # Service account with specific permissions


class ApiKeyStatus(StrEnum):
    """API key lifecycle status.

    Attributes:
        ACTIVE: Key is active and can authorize requests.
        SUSPENDED: Key is temporarily disabled (can be reactivated).
        EXPIRED: Key has passed its expiration date.
        REVOKED: Key is permanently disabled.
    """

    ACTIVE = "active"
    SUSPENDED = "suspended"
    EXPIRED = "expired"
    REVOKED = "revoked"


@dataclass
class RateLimitConfig:
    """Rate limiting configuration for an API key.

    All limits are optional. When set to ``None``, no limit is enforced
    for that window.

    Attributes:
        requests_per_minute: Max requests allowed per minute.
        requests_per_hour: Max requests allowed per hour.
        requests_per_day: Max requests allowed per day.
        tokens_per_minute: Max tokens allowed per minute.
        tokens_per_hour: Max tokens allowed per hour.
        tokens_per_day: Max tokens allowed per day.
    """

    requests_per_minute: int | None = None
    requests_per_hour: int | None = None
    requests_per_day: int | None = None
    tokens_per_minute: int | None = None
    tokens_per_hour: int | None = None
    tokens_per_day: int | None = None

    def as_dict(self) -> dict[str, tp.Any]:
        """Serialize the rate limit config to a plain dictionary.

        Returns:
            dict[str, tp.Any]: A JSON-serializable mapping of every limit
            field to its current value. Unset limits are emitted as ``None``.
        """
        return {
            "requests_per_minute": self.requests_per_minute,
            "requests_per_hour": self.requests_per_hour,
            "requests_per_day": self.requests_per_day,
            "tokens_per_minute": self.tokens_per_minute,
            "tokens_per_hour": self.tokens_per_hour,
            "tokens_per_day": self.tokens_per_day,
        }


@dataclass
class QuotaConfig:
    """Usage quota limits for an API key.

    All limits are optional. When set to ``None``, no quota is enforced
    for that metric.

    Attributes:
        max_total_tokens: Lifetime cumulative token limit.
        max_total_requests: Lifetime cumulative request limit.
        monthly_token_limit: Monthly token limit (resets each calendar month).
        monthly_request_limit: Monthly request limit (resets each calendar month).
    """

    max_total_tokens: int | None = None  # Lifetime token limit
    max_total_requests: int | None = None  # Lifetime request limit
    monthly_token_limit: int | None = None
    monthly_request_limit: int | None = None

    def as_dict(self) -> dict[str, tp.Any]:
        """Serialize the quota config to a plain dictionary.

        Returns:
            dict[str, tp.Any]: JSON-serializable mapping of quota fields.
            Unset limits are emitted as ``None``.
        """
        return {
            "max_total_tokens": self.max_total_tokens,
            "max_total_requests": self.max_total_requests,
            "monthly_token_limit": self.monthly_token_limit,
            "monthly_request_limit": self.monthly_request_limit,
        }


@dataclass
class ApiKeyPermissions:
    """Granular permissions for an API key.

    ``None`` values for list fields mean "no restriction" (all allowed).

    Attributes:
        allowed_models: Allowlist of model names. ``None`` allows all.
        allowed_endpoints: Allowlist of API endpoints. ``None`` allows all.
        allowed_ip_addresses: IP allowlist. ``None`` means no IP restriction.
        blocked_ip_addresses: IP blocklist (checked before allowlist).
        enable_streaming: Whether streaming responses are permitted.
        enable_function_calling: Whether function calling is permitted.
        max_tokens_per_request: Per-request token ceiling.
    """

    allowed_models: list[str] | None = None  # None = all models allowed
    allowed_endpoints: list[str] | None = None  # None = all endpoints allowed
    allowed_ip_addresses: list[str] | None = None  # None = no IP restrictions
    blocked_ip_addresses: list[str] | None = None
    enable_streaming: bool = True
    enable_function_calling: bool = True
    max_tokens_per_request: int | None = None

    def as_dict(self) -> dict[str, tp.Any]:
        """Serialize the permissions to a plain dictionary.

        Returns:
            dict[str, tp.Any]: JSON-serializable mapping of every
            permission field. ``None`` lists denote "no restriction".
        """
        return {
            "allowed_models": self.allowed_models,
            "allowed_endpoints": self.allowed_endpoints,
            "allowed_ip_addresses": self.allowed_ip_addresses,
            "blocked_ip_addresses": self.blocked_ip_addresses,
            "enable_streaming": self.enable_streaming,
            "enable_function_calling": self.enable_function_calling,
            "max_tokens_per_request": self.max_tokens_per_request,
        }


@dataclass
class ApiKeyMetadata:
    """Complete metadata record for a managed API key.

    Combines identification, lifecycle state, usage tracking, rate limits,
    quotas, and granular permissions in a single dataclass. Instances are
    stored in-memory by ``EnhancedApiKeyManager`` and persisted to disk
    via ``AuthStorage``.

    Attributes:
        key_id: Internal unique identifier (e.g. ``key_abc123...``).
        key_prefix: Display-safe prefix of the raw key (e.g. ``sk-abc123...``).
        hashed_key: SHA-256 hex digest of the raw key.
        name: Human-readable name.
        description: Optional description.
        role: Access control role.
        status: Current lifecycle status.
        created_at: Unix timestamp of creation.
        created_by: Creator user or service name.
        expires_at: Optional expiration timestamp.
        last_used_at: Timestamp of last authorized request.
        last_rotated_at: Timestamp of last key rotation.
        total_requests: Lifetime request count.
        total_prompt_tokens: Lifetime prompt token count.
        total_completion_tokens: Lifetime completion token count.
        monthly_requests: Current month's request count.
        monthly_tokens: Current month's token count.
        last_reset_month: Month number of the last monthly counter reset.
        rate_limits: Rate limiting configuration.
        quota: Usage quota configuration.
        permissions: Granular permissions.
        tags: Organizational tags.
        metadata: Arbitrary user-defined metadata.
    """

    key_id: str  # Internal unique identifier
    key_prefix: str  # First 8 chars for display (e.g., "sk-abc123...")
    hashed_key: str  # Hashed version of the full key
    name: str  # Human-readable name
    description: str | None = None
    role: ApiKeyRole = ApiKeyRole.USER
    status: ApiKeyStatus = ApiKeyStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    created_by: str | None = None  # User/service that created this key
    expires_at: float | None = None  # Unix timestamp, None = no expiration
    last_used_at: float | None = None
    last_rotated_at: float | None = None

    # Usage tracking
    total_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    monthly_requests: int = 0
    monthly_tokens: int = 0
    last_reset_month: int = field(default_factory=lambda: time.localtime().tm_mon)

    # Configuration
    rate_limits: RateLimitConfig = field(default_factory=RateLimitConfig)
    quota: QuotaConfig = field(default_factory=QuotaConfig)
    permissions: ApiKeyPermissions = field(default_factory=ApiKeyPermissions)

    # Metadata
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, tp.Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Whether the key has passed its expiration timestamp.

        Returns:
            bool: ``True`` only when ``expires_at`` is set and lies in the
            past.
        """
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def is_active(self) -> bool:
        """Whether the key may currently authorize requests.

        Returns:
            bool: ``True`` when the status is ``ACTIVE`` and the key has
            not expired.
        """
        return self.status == ApiKeyStatus.ACTIVE and not self.is_expired()

    def update_last_used(self) -> None:
        """Stamp ``last_used_at`` with the current time."""
        self.last_used_at = time.time()

    def reset_monthly_counters_if_needed(self) -> None:
        """Reset monthly token/request counters when a new calendar month starts.

        Compares the current month against ``last_reset_month`` and zeros
        the monthly counters when they differ.
        """
        current_month = time.localtime().tm_mon
        if current_month != self.last_reset_month:
            self.monthly_requests = 0
            self.monthly_tokens = 0
            self.last_reset_month = current_month

    def as_dict(self, include_sensitive: bool = False) -> dict[str, tp.Any]:
        """Serialize the metadata to a JSON-compatible dictionary.

        Args:
            include_sensitive: When ``True``, include the SHA-256
                ``hashed_key``. The raw key is never returned in any form.

        Returns:
            dict[str, tp.Any]: A flat mapping of every public field plus
            derived totals (``total_tokens``, ``is_expired``).
        """
        data = {
            "key_id": self.key_id,
            "key_prefix": self.key_prefix,
            "name": self.name,
            "description": self.description,
            "role": self.role.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "expires_at": self.expires_at,
            "last_used_at": self.last_used_at,
            "last_rotated_at": self.last_rotated_at,
            "is_expired": self.is_expired(),
            "total_requests": self.total_requests,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "monthly_requests": self.monthly_requests,
            "monthly_tokens": self.monthly_tokens,
            "rate_limits": self.rate_limits.as_dict(),
            "quota": self.quota.as_dict(),
            "permissions": self.permissions.as_dict(),
            "tags": self.tags,
            "metadata": self.metadata,
        }

        if include_sensitive:
            data["hashed_key"] = self.hashed_key

        return data


@dataclass
class AuditLogEntry:
    """Audit log entry for tracking API key operations.

    Attributes:
        timestamp: Unix timestamp of the event.
        key_id: API key ID involved, if applicable.
        action: Action name (e.g. ``key_created``, ``request_authorized``).
        actor: User or service that performed the action.
        ip_address: Client IP address, if available.
        details: Additional context about the event.
        success: Whether the action succeeded.
    """

    timestamp: float = field(default_factory=time.time)
    key_id: str | None = None
    action: str = ""  # e.g., "key_created", "key_revoked", "request_authorized", "request_denied"
    actor: str | None = None  # Who performed the action
    ip_address: str | None = None
    details: dict[str, tp.Any] = field(default_factory=dict)
    success: bool = True

    def as_dict(self) -> dict[str, tp.Any]:
        """Serialize the audit log entry to a plain dictionary.

        Returns:
            dict[str, tp.Any]: JSON-serializable mapping of every field on
            this entry.
        """
        return {
            "timestamp": self.timestamp,
            "key_id": self.key_id,
            "action": self.action,
            "actor": self.actor,
            "ip_address": self.ip_address,
            "details": self.details,
            "success": self.success,
        }
