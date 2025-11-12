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

"""Enhanced authentication models and schemas for eSurge API server."""

from __future__ import annotations

import time
import typing as tp
from dataclasses import dataclass, field
from enum import Enum


class ApiKeyRole(str, Enum):
    """Role-based access control levels for API keys."""

    ADMIN = "admin"  # Full access including key management
    USER = "user"  # Standard access to inference endpoints
    READONLY = "readonly"  # Read-only access (metrics, health, list models)
    SERVICE = "service"  # Service account with specific permissions


class ApiKeyStatus(str, Enum):
    """API key lifecycle status."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    EXPIRED = "expired"
    REVOKED = "revoked"


@dataclass
class RateLimitConfig:
    """Rate limiting configuration for an API key."""

    requests_per_minute: int | None = None
    requests_per_hour: int | None = None
    requests_per_day: int | None = None
    tokens_per_minute: int | None = None
    tokens_per_hour: int | None = None
    tokens_per_day: int | None = None

    def as_dict(self) -> dict[str, tp.Any]:
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
    """Usage quota limits for an API key."""

    max_total_tokens: int | None = None  # Lifetime token limit
    max_total_requests: int | None = None  # Lifetime request limit
    monthly_token_limit: int | None = None
    monthly_request_limit: int | None = None

    def as_dict(self) -> dict[str, tp.Any]:
        return {
            "max_total_tokens": self.max_total_tokens,
            "max_total_requests": self.max_total_requests,
            "monthly_token_limit": self.monthly_token_limit,
            "monthly_request_limit": self.monthly_request_limit,
        }


@dataclass
class ApiKeyPermissions:
    """Granular permissions for an API key."""

    allowed_models: list[str] | None = None  # None = all models allowed
    allowed_endpoints: list[str] | None = None  # None = all endpoints allowed
    allowed_ip_addresses: list[str] | None = None  # None = no IP restrictions
    blocked_ip_addresses: list[str] | None = None
    enable_streaming: bool = True
    enable_function_calling: bool = True
    max_tokens_per_request: int | None = None

    def as_dict(self) -> dict[str, tp.Any]:
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
    """Extended metadata for an API key."""

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
        """Check if the key has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def is_active(self) -> bool:
        """Check if the key is active and usable."""
        return self.status == ApiKeyStatus.ACTIVE and not self.is_expired()

    def update_last_used(self) -> None:
        """Update the last used timestamp."""
        self.last_used_at = time.time()

    def reset_monthly_counters_if_needed(self) -> None:
        """Reset monthly counters if we're in a new month."""
        current_month = time.localtime().tm_mon
        if current_month != self.last_reset_month:
            self.monthly_requests = 0
            self.monthly_tokens = 0
            self.last_reset_month = current_month

    def as_dict(self, include_sensitive: bool = False) -> dict[str, tp.Any]:
        """Serialize key metadata to dictionary.

        Args:
                include_sensitive: If True, include hashed_key. Never include the raw key.
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
    """Audit log entry for tracking API key operations."""

    timestamp: float = field(default_factory=time.time)
    key_id: str | None = None
    action: str = ""  # e.g., "key_created", "key_revoked", "request_authorized", "request_denied"
    actor: str | None = None  # Who performed the action
    ip_address: str | None = None
    details: dict[str, tp.Any] = field(default_factory=dict)
    success: bool = True

    def as_dict(self) -> dict[str, tp.Any]:
        return {
            "timestamp": self.timestamp,
            "key_id": self.key_id,
            "action": self.action,
            "actor": self.actor,
            "ip_address": self.ip_address,
            "details": self.details,
            "success": self.success,
        }
