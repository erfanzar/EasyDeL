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

"""Enhanced API key manager with security, rate limiting, and audit logging."""

from __future__ import annotations

import hashlib
import secrets
import threading
import time
import typing as tp
from collections import defaultdict, deque
from pathlib import Path

from easydel.workers.loggers import get_logger

from .auth_models import (
    ApiKeyMetadata,
    ApiKeyPermissions,
    ApiKeyRole,
    ApiKeyStatus,
    AuditLogEntry,
    QuotaConfig,
    RateLimitConfig,
)
from .auth_storage import AuthStorage

logger = get_logger("AuthManager")


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    pass


class QuotaExceeded(Exception):
    """Raised when quota limit is exceeded."""

    pass


class PermissionDenied(Exception):
    """Raised when permission check fails."""

    pass


class EnhancedApiKeyManager:
    """Production-grade API key manager with security, RBAC, rate limiting, and audit logging.

    Features:
    - Secure key storage with SHA-256 hashing
    - Role-based access control (RBAC)
    - Per-key rate limiting (requests/min, hour, day + tokens/min, hour, day)
    - Per-key quotas (lifetime and monthly limits)
    - IP allowlist/blocklist
    - API key expiration and rotation
    - Comprehensive audit logging
    - Thread-safe operations
    """

    def __init__(
        self,
        require_api_key: bool = False,
        admin_key: str | None = None,
        enable_audit_logging: bool = True,
        max_audit_entries: int = 10000,
        storage_dir: str | Path | None = None,
        enable_persistence: bool = True,
        auto_save: bool = True,
        save_interval: float = 60.0,
    ) -> None:
        """Initialize the enhanced API key manager.

        Args:
                require_api_key: If True, all requests must provide a valid API key.
                admin_key: Optional admin key for initial setup. If provided, creates an admin key.
                enable_audit_logging: Enable audit logging for all operations.
                max_audit_entries: Maximum number of audit log entries to keep in memory.
                storage_dir: Directory to store auth data. Defaults to ~/.cache/esurge-auth/
                enable_persistence: Enable persistent storage to disk (default: True).
                auto_save: Enable automatic periodic saving (default: True).
                save_interval: Seconds between auto-saves (default: 60.0).
        """
        self.require_api_key = require_api_key
        self.enable_audit_logging = enable_audit_logging
        self.max_audit_entries = max_audit_entries
        self.enable_persistence = enable_persistence

        # Key storage: hashed_key -> ApiKeyMetadata
        self._keys: dict[str, ApiKeyMetadata] = {}
        # Key lookup: key_id -> hashed_key for faster lookups
        self._key_id_to_hash: dict[str, str] = {}
        # Reverse lookup: raw_key -> hashed_key (only used during validation)
        self._raw_to_hash_cache: dict[str, str] = {}

        # Auto-save routines call back into helper methods that also grab this lock,
        # so we need a re-entrant lock to prevent deadlocks when the same thread
        # re-acquires it during persistence.
        self._lock = threading.RLock()

        # Rate limiting tracking: key_id -> time window -> deque of timestamps
        self._rate_limit_windows: dict[str, dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
        # Token usage tracking for rate limits
        self._token_usage_windows: dict[str, dict[str, deque]] = defaultdict(lambda: defaultdict(deque))

        # Audit log
        self._audit_log: deque[AuditLogEntry] = deque(maxlen=max_audit_entries)

        # Initialize persistent storage
        self.storage: AuthStorage | None = None
        if enable_persistence:
            self.storage = AuthStorage(
                storage_dir=storage_dir,
                auto_save=auto_save,
                save_interval=save_interval,
            )
            self._load_from_storage()

        # Create admin key if provided (after loading from storage)
        if admin_key:
            self._create_initial_admin_key(admin_key)

    def _hash_key(self, key: str) -> str:
        """Hash an API key using SHA-256.

        Args:
                key: Raw API key string.

        Returns:
                Hexadecimal hash of the key.
        """
        return hashlib.sha256(key.encode()).hexdigest()

    def _load_from_storage(self) -> None:
        """Load auth data from persistent storage."""
        if not self.storage:
            return

        try:
            # Load keys
            keys = self.storage.load_keys()
            for hashed_key, metadata in keys.items():
                self._keys[hashed_key] = metadata
                self._key_id_to_hash[metadata.key_id] = hashed_key
            logger.info(f"Loaded {len(keys)} API keys from storage")

            # Load audit logs
            if self.enable_audit_logging:
                logs = self.storage.load_audit_logs()
                for log in logs[-self.max_audit_entries :]:  # Only load up to max
                    self._audit_log.append(log)
                logger.info(f"Loaded {len(self._audit_log)} audit log entries from storage")

        except Exception as e:
            logger.error(f"Failed to load from storage: {e}")

    def _save_to_storage(self) -> None:
        """Save auth data to persistent storage."""
        if not self.storage:
            return

        try:
            # Save keys
            self.storage.save_keys(self._keys)

            # Save audit logs
            if self.enable_audit_logging:
                self.storage.save_audit_logs(list(self._audit_log))

            # Save usage stats
            stats = self.get_statistics()
            self.storage.save_usage_stats(stats)

            logger.debug("Auth data saved to storage")
        except Exception as e:
            logger.error(f"Failed to save to storage: {e}")

    def _auto_save_if_needed(self) -> None:
        """Trigger auto-save if conditions are met."""
        if self.storage and self.storage.should_auto_save():
            self._save_to_storage()

    def _mark_dirty_and_save(self) -> None:
        """Mark storage as dirty and trigger auto-save if needed."""
        if self.storage:
            self.storage.mark_dirty()
            self._auto_save_if_needed()

    def _create_initial_admin_key(self, key: str) -> None:
        """Create initial admin key during initialization."""
        # Check if admin key already exists in storage
        hashed = self._hash_key(key)
        if hashed in self._keys:
            logger.info("Admin key already exists in storage, skipping creation")
            # Cache the raw key for validation
            self._raw_to_hash_cache[key] = hashed
            return

        try:
            self.create_api_key(
                raw_key=key,
                name="Initial Admin Key",
                role=ApiKeyRole.ADMIN,
                created_by="system",
            )
            logger.info("Initial admin key created successfully")
        except Exception as e:
            logger.error(f"Failed to create initial admin key: {e}")

    def _log_audit(
        self,
        action: str,
        key_id: str | None = None,
        actor: str | None = None,
        ip_address: str | None = None,
        details: dict[str, tp.Any] | None = None,
        success: bool = True,
    ) -> None:
        """Log an audit entry.

        Args:
                action: Action performed (e.g., "key_created", "request_authorized").
                key_id: API key ID involved in the action.
                actor: User/service performing the action.
                ip_address: IP address of the request.
                details: Additional details about the action.
                success: Whether the action was successful.
        """
        if not self.enable_audit_logging:
            return

        entry = AuditLogEntry(
            key_id=key_id,
            action=action,
            actor=actor,
            ip_address=ip_address,
            details=details or {},
            success=success,
        )
        self._audit_log.append(entry)

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
        """Generate a new random API key with configuration.

        Args:
                name: Human-readable name for the key.
                role: Access control role.
                description: Optional description.
                created_by: User/service creating the key.
                expires_in_days: Number of days until expiration (None = never expires).
                rate_limits: Rate limiting configuration.
                quota: Usage quota configuration.
                permissions: Granular permissions.
                tags: List of tags for organization.
                metadata: Additional metadata.

        Returns:
                Tuple of (raw_key, metadata). Store raw_key securely - it won't be retrievable later.
        """
        raw_key = f"sk-{secrets.token_urlsafe(48)}"
        metadata_obj = self.create_api_key(
            raw_key=raw_key,
            name=name,
            role=role,
            description=description,
            created_by=created_by,
            expires_in_days=expires_in_days,
            rate_limits=rate_limits,
            quota=quota,
            permissions=permissions,
            tags=tags,
            metadata=metadata,
        )
        return raw_key, metadata_obj

    def create_api_key(
        self,
        raw_key: str,
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
    ) -> ApiKeyMetadata:
        """Create an API key with a user-provided raw key.

        Args:
                raw_key: The raw API key string (will be hashed for storage).
                name: Human-readable name for the key.
                role: Access control role.
                description: Optional description.
                created_by: User/service creating the key.
                expires_in_days: Number of days until expiration (None = never expires).
                rate_limits: Rate limiting configuration.
                quota: Usage quota configuration.
                permissions: Granular permissions.
                tags: List of tags for organization.
                metadata: Additional metadata.

        Returns:
                ApiKeyMetadata object.

        Raises:
                ValueError: If key is invalid or already exists.
        """
        if not raw_key or len(raw_key) < 16:
            raise ValueError("API key must be at least 16 characters long")

        hashed_key = self._hash_key(raw_key)

        with self._lock:
            if hashed_key in self._keys:
                raise ValueError("API key already exists")

            key_id = f"key_{secrets.token_hex(16)}"
            key_prefix = raw_key[:12] + "..." if len(raw_key) > 12 else raw_key

            expires_at = None
            if expires_in_days is not None:
                expires_at = time.time() + (expires_in_days * 86400)

            key_metadata = ApiKeyMetadata(
                key_id=key_id,
                key_prefix=key_prefix,
                hashed_key=hashed_key,
                name=name,
                description=description,
                role=role,
                created_by=created_by,
                expires_at=expires_at,
                rate_limits=rate_limits or RateLimitConfig(),
                quota=quota or QuotaConfig(),
                permissions=permissions or ApiKeyPermissions(),
                tags=tags or [],
                metadata=metadata or {},
            )

            self._keys[hashed_key] = key_metadata
            self._key_id_to_hash[key_id] = hashed_key
            self._raw_to_hash_cache[raw_key] = hashed_key

            self._log_audit(
                action="key_created",
                key_id=key_id,
                actor=created_by,
                details={"name": name, "role": role.value, "expires_at": expires_at},
                success=True,
            )

            logger.info(f"Created API key: {key_id} (name: {name}, role: {role.value})")

            # Mark storage as dirty and auto-save if needed
            if self.storage:
                self.storage.mark_dirty()
                self._auto_save_if_needed()

            return key_metadata

    def validate_key(self, raw_key: str | None) -> ApiKeyMetadata | None:
        """Validate a raw API key and return its metadata.

        Args:
                raw_key: The raw API key to validate.

        Returns:
                ApiKeyMetadata if valid, None otherwise.
        """
        if not raw_key:
            return None

        # Check cache first
        hashed_key = self._raw_to_hash_cache.get(raw_key)
        if hashed_key is None:
            hashed_key = self._hash_key(raw_key)
            # Cache the result for faster subsequent lookups
            self._raw_to_hash_cache[raw_key] = hashed_key

        metadata = self._keys.get(hashed_key)
        if metadata is None:
            return None

        # Check if key is active
        if not metadata.is_active():
            return None

        return metadata

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
        # Validate key
        metadata = self.validate_key(raw_key)
        if metadata is None:
            self._log_audit(
                action="request_denied",
                details={"reason": "invalid_key"},
                ip_address=ip_address,
                success=False,
            )
            raise PermissionDenied("Invalid or inactive API key")

        # Update last used timestamp
        metadata.update_last_used()

        # Check IP restrictions
        if not self._check_ip_permissions(metadata, ip_address):
            self._log_audit(
                action="request_denied",
                key_id=metadata.key_id,
                ip_address=ip_address,
                details={"reason": "ip_blocked"},
                success=False,
            )
            raise PermissionDenied(f"Access denied from IP address: {ip_address}")

        # Check endpoint permissions
        if not self._check_endpoint_permissions(metadata, endpoint):
            self._log_audit(
                action="request_denied",
                key_id=metadata.key_id,
                ip_address=ip_address,
                details={"reason": "endpoint_blocked", "endpoint": endpoint},
                success=False,
            )
            raise PermissionDenied(f"Access denied to endpoint: {endpoint}")

        # Check model permissions
        if model and not self._check_model_permissions(metadata, model):
            self._log_audit(
                action="request_denied",
                key_id=metadata.key_id,
                ip_address=ip_address,
                details={"reason": "model_blocked", "model": model},
                success=False,
            )
            raise PermissionDenied(f"Access denied to model: {model}")

        # Check rate limits
        self._check_rate_limits(metadata)

        # Check quotas
        self._check_quotas(metadata, requested_tokens)

        # Check per-request token limit
        if metadata.permissions.max_tokens_per_request:
            if requested_tokens > metadata.permissions.max_tokens_per_request:
                raise PermissionDenied(
                    f"Requested tokens ({requested_tokens}) exceeds limit ({metadata.permissions.max_tokens_per_request})"
                )

        self._log_audit(
            action="request_authorized",
            key_id=metadata.key_id,
            ip_address=ip_address,
            details={"endpoint": endpoint, "model": model},
            success=True,
        )

        return metadata

    def _check_ip_permissions(self, metadata: ApiKeyMetadata, ip_address: str | None) -> bool:
        """Check if the IP address is allowed."""
        if ip_address is None:
            return True

        permissions = metadata.permissions

        # Check blocklist first
        if permissions.blocked_ip_addresses:
            if ip_address in permissions.blocked_ip_addresses:
                return False

        # Check allowlist
        if permissions.allowed_ip_addresses:
            return ip_address in permissions.allowed_ip_addresses

        return True

    def _check_endpoint_permissions(self, metadata: ApiKeyMetadata, endpoint: str | None) -> bool:
        """Check if the endpoint is allowed."""
        if endpoint is None:
            return True

        permissions = metadata.permissions
        if permissions.allowed_endpoints is None:
            return True

        return endpoint in permissions.allowed_endpoints

    def _check_model_permissions(self, metadata: ApiKeyMetadata, model: str | None) -> bool:
        """Check if the model is allowed."""
        if model is None:
            return True

        permissions = metadata.permissions
        if permissions.allowed_models is None:
            return True

        return model in permissions.allowed_models

    def _check_rate_limits(self, metadata: ApiKeyMetadata) -> None:
        """Check if rate limits are exceeded.

        Raises:
                RateLimitExceeded: If any rate limit is exceeded.
        """
        rate_limits = metadata.rate_limits
        current_time = time.time()
        key_id = metadata.key_id

        # Check requests per minute
        if rate_limits.requests_per_minute:
            window = self._rate_limit_windows[key_id]["requests_minute"]
            self._clean_window(window, current_time, 60)
            if len(window) >= rate_limits.requests_per_minute:
                raise RateLimitExceeded(f"Rate limit exceeded: {rate_limits.requests_per_minute} requests/minute")
            window.append(current_time)

        # Check requests per hour
        if rate_limits.requests_per_hour:
            window = self._rate_limit_windows[key_id]["requests_hour"]
            self._clean_window(window, current_time, 3600)
            if len(window) >= rate_limits.requests_per_hour:
                raise RateLimitExceeded(f"Rate limit exceeded: {rate_limits.requests_per_hour} requests/hour")
            window.append(current_time)

        # Check requests per day
        if rate_limits.requests_per_day:
            window = self._rate_limit_windows[key_id]["requests_day"]
            self._clean_window(window, current_time, 86400)
            if len(window) >= rate_limits.requests_per_day:
                raise RateLimitExceeded(f"Rate limit exceeded: {rate_limits.requests_per_day} requests/day")
            window.append(current_time)

    def _check_quotas(self, metadata: ApiKeyMetadata, requested_tokens: int) -> None:
        """Check if quotas are exceeded.

        Raises:
                QuotaExceeded: If any quota is exceeded.
        """
        metadata.reset_monthly_counters_if_needed()
        quota = metadata.quota

        # Check lifetime token limit
        if quota.max_total_tokens:
            if (
                metadata.total_prompt_tokens + metadata.total_completion_tokens + requested_tokens
                > quota.max_total_tokens
            ):
                raise QuotaExceeded(f"Total token quota exceeded: {quota.max_total_tokens}")

        # Check lifetime request limit
        if quota.max_total_requests:
            if metadata.total_requests >= quota.max_total_requests:
                raise QuotaExceeded(f"Total request quota exceeded: {quota.max_total_requests}")

        # Check monthly token limit
        if quota.monthly_token_limit:
            if metadata.monthly_tokens + requested_tokens > quota.monthly_token_limit:
                raise QuotaExceeded(f"Monthly token quota exceeded: {quota.monthly_token_limit}")

        # Check monthly request limit
        if quota.monthly_request_limit:
            if metadata.monthly_requests >= quota.monthly_request_limit:
                raise QuotaExceeded(f"Monthly request quota exceeded: {quota.monthly_request_limit}")

    def _clean_window(self, window: deque, current_time: float, window_size: int) -> None:
        """Remove expired entries from a time window."""
        cutoff = current_time - window_size
        while window and window[0] < cutoff:
            window.popleft()

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
        if not raw_key:
            return

        metadata = self.validate_key(raw_key)
        if metadata is None:
            return

        with self._lock:
            metadata.total_requests += 1
            metadata.total_prompt_tokens += max(prompt_tokens, 0)
            metadata.total_completion_tokens += max(completion_tokens, 0)
            metadata.monthly_requests += 1
            metadata.monthly_tokens += max(prompt_tokens + completion_tokens, 0)

        # Track token rate limits
        self._record_token_rate_limit(metadata, prompt_tokens + completion_tokens)

        # Mark storage as dirty for next auto-save
        if self.storage:
            self.storage.mark_dirty()

    def _record_token_rate_limit(self, metadata: ApiKeyMetadata, tokens: int) -> None:
        """Record token usage for rate limiting."""
        rate_limits = metadata.rate_limits
        current_time = time.time()
        key_id = metadata.key_id

        if rate_limits.tokens_per_minute:
            window = self._token_usage_windows[key_id]["tokens_minute"]
            self._clean_window(window, current_time, 60)
            window.append((current_time, tokens))

        if rate_limits.tokens_per_hour:
            window = self._token_usage_windows[key_id]["tokens_hour"]
            self._clean_window(window, current_time, 3600)
            window.append((current_time, tokens))

        if rate_limits.tokens_per_day:
            window = self._token_usage_windows[key_id]["tokens_day"]
            self._clean_window(window, current_time, 86400)
            window.append((current_time, tokens))

    def revoke_key(self, key_id: str, revoked_by: str | None = None) -> bool:
        """Revoke an API key.

        Args:
                key_id: ID of the key to revoke.
                revoked_by: User/service revoking the key.

        Returns:
                True if revoked, False if not found.
        """
        with self._lock:
            hashed_key = self._key_id_to_hash.get(key_id)
            if hashed_key is None:
                return False

            metadata = self._keys.get(hashed_key)
            if metadata is None:
                return False

            metadata.status = ApiKeyStatus.REVOKED

            self._log_audit(
                action="key_revoked",
                key_id=key_id,
                actor=revoked_by,
                details={"name": metadata.name},
                success=True,
            )

            logger.info(f"Revoked API key: {key_id} by {revoked_by}")
            self._mark_dirty_and_save()
            return True

    def suspend_key(self, key_id: str, suspended_by: str | None = None) -> bool:
        """Suspend an API key (can be reactivated later).

        Args:
                key_id: ID of the key to suspend.
                suspended_by: User/service suspending the key.

        Returns:
                True if suspended, False if not found.
        """
        with self._lock:
            hashed_key = self._key_id_to_hash.get(key_id)
            if hashed_key is None:
                return False

            metadata = self._keys.get(hashed_key)
            if metadata is None:
                return False

            metadata.status = ApiKeyStatus.SUSPENDED

            self._log_audit(
                action="key_suspended",
                key_id=key_id,
                actor=suspended_by,
                details={"name": metadata.name},
                success=True,
            )

            logger.info(f"Suspended API key: {key_id} by {suspended_by}")
            self._mark_dirty_and_save()
            return True

    def reactivate_key(self, key_id: str, reactivated_by: str | None = None) -> bool:
        """Reactivate a suspended API key.

        Args:
                key_id: ID of the key to reactivate.
                reactivated_by: User/service reactivating the key.

        Returns:
                True if reactivated, False if not found or revoked.
        """
        with self._lock:
            hashed_key = self._key_id_to_hash.get(key_id)
            if hashed_key is None:
                return False

            metadata = self._keys.get(hashed_key)
            if metadata is None or metadata.status == ApiKeyStatus.REVOKED:
                return False

            metadata.status = ApiKeyStatus.ACTIVE

            self._log_audit(
                action="key_reactivated",
                key_id=key_id,
                actor=reactivated_by,
                details={"name": metadata.name},
                success=True,
            )

            logger.info(f"Reactivated API key: {key_id} by {reactivated_by}")
            self._mark_dirty_and_save()
            return True

    def delete_key(self, key_id: str, deleted_by: str | None = None) -> bool:
        """Permanently delete an API key.

        Args:
                key_id: ID of the key to delete.
                deleted_by: User/service deleting the key.

        Returns:
                True if deleted, False if not found.
        """
        with self._lock:
            hashed_key = self._key_id_to_hash.get(key_id)
            if hashed_key is None:
                return False

            metadata = self._keys.get(hashed_key)
            if metadata is None:
                return False

            # Remove from all data structures
            del self._keys[hashed_key]
            del self._key_id_to_hash[key_id]

            # Clear from cache
            for raw_key, cached_hash in list(self._raw_to_hash_cache.items()):
                if cached_hash == hashed_key:
                    del self._raw_to_hash_cache[raw_key]

            # Clear rate limit windows
            if key_id in self._rate_limit_windows:
                del self._rate_limit_windows[key_id]
            if key_id in self._token_usage_windows:
                del self._token_usage_windows[key_id]

            self._log_audit(
                action="key_deleted",
                key_id=key_id,
                actor=deleted_by,
                details={"name": metadata.name},
                success=True,
            )

            logger.info(f"Deleted API key: {key_id} by {deleted_by}")
            self._mark_dirty_and_save()
            return True

    def get_key_by_id(self, key_id: str) -> ApiKeyMetadata | None:
        """Get key metadata by key ID.

        Args:
                key_id: ID of the key.

        Returns:
                ApiKeyMetadata if found, None otherwise.
        """
        hashed_key = self._key_id_to_hash.get(key_id)
        if hashed_key is None:
            return None
        return self._keys.get(hashed_key)

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
                tags: Filter by tags (must have all tags).

        Returns:
                List of matching ApiKeyMetadata objects.
        """
        with self._lock:
            keys = list(self._keys.values())

        if role:
            keys = [k for k in keys if k.role == role]
        if status:
            keys = [k for k in keys if k.status == status]
        if tags:
            keys = [k for k in keys if all(tag in k.tags for tag in tags)]

        return keys

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
                expires_in_days: New expiration (from now).
                rate_limits: New rate limits.
                quota: New quota.
                permissions: New permissions.
                tags: New tags.
                metadata: New metadata (merged with existing).
                updated_by: User/service updating the key.

        Returns:
                True if updated, False if not found.
        """
        with self._lock:
            key_meta = self.get_key_by_id(key_id)
            if key_meta is None:
                return False

            changes = {}
            if name is not None:
                key_meta.name = name
                changes["name"] = name
            if description is not None:
                key_meta.description = description
                changes["description"] = description
            if role is not None:
                key_meta.role = role
                changes["role"] = role.value
            if expires_in_days is not None:
                key_meta.expires_at = time.time() + (expires_in_days * 86400)
                changes["expires_at"] = key_meta.expires_at
            if rate_limits is not None:
                key_meta.rate_limits = rate_limits
                changes["rate_limits"] = "updated"
            if quota is not None:
                key_meta.quota = quota
                changes["quota"] = "updated"
            if permissions is not None:
                key_meta.permissions = permissions
                changes["permissions"] = "updated"
            if tags is not None:
                key_meta.tags = tags
                changes["tags"] = tags
            if metadata is not None:
                key_meta.metadata.update(metadata)
                changes["metadata"] = "updated"

            self._log_audit(
                action="key_updated",
                key_id=key_id,
                actor=updated_by,
                details=changes,
                success=True,
            )

            logger.info(f"Updated API key: {key_id} by {updated_by}")
            self._mark_dirty_and_save()
            return True

    def rotate_key(self, key_id: str, rotated_by: str | None = None) -> tuple[str, ApiKeyMetadata] | None:
        """Rotate an API key (generate new key, preserve metadata).

        Args:
                key_id: ID of the key to rotate.
                rotated_by: User/service rotating the key.

        Returns:
                Tuple of (new_raw_key, metadata) if successful, None if not found.
        """
        old_metadata = self.get_key_by_id(key_id)
        if old_metadata is None:
            return None

        # Generate new key
        new_raw_key = f"sk-{secrets.token_urlsafe(48)}"
        new_hashed_key = self._hash_key(new_raw_key)

        with self._lock:
            # Remove old key
            old_hashed_key = self._key_id_to_hash.get(key_id)
            if old_hashed_key:
                del self._keys[old_hashed_key]

            # Update metadata
            old_metadata.hashed_key = new_hashed_key
            old_metadata.key_prefix = new_raw_key[:12] + "..."
            old_metadata.last_rotated_at = time.time()

            # Store with new hash
            self._keys[new_hashed_key] = old_metadata
            self._key_id_to_hash[key_id] = new_hashed_key
            self._raw_to_hash_cache[new_raw_key] = new_hashed_key

            self._log_audit(
                action="key_rotated",
                key_id=key_id,
                actor=rotated_by,
                details={"name": old_metadata.name},
                success=True,
            )

            logger.info(f"Rotated API key: {key_id} by {rotated_by}")
            self._mark_dirty_and_save()
            return new_raw_key, old_metadata

    def get_audit_logs(
        self,
        limit: int = 100,
        key_id: str | None = None,
        action: str | None = None,
    ) -> list[AuditLogEntry]:
        """Get audit log entries.

        Args:
                limit: Maximum number of entries to return.
                key_id: Filter by key ID.
                action: Filter by action type.

        Returns:
                List of AuditLogEntry objects (most recent first).
        """
        logs = list(reversed(self._audit_log))

        if key_id:
            logs = [log for log in logs if log.key_id == key_id]
        if action:
            logs = [log for log in logs if log.action == action]

        return logs[:limit]

    def get_statistics(self) -> dict[str, tp.Any]:
        """Get overall statistics about API keys and usage.

        Returns:
                Dictionary with aggregate statistics.
        """
        with self._lock:
            total_keys = len(self._keys)
            active_keys = sum(1 for k in self._keys.values() if k.status == ApiKeyStatus.ACTIVE)
            suspended_keys = sum(1 for k in self._keys.values() if k.status == ApiKeyStatus.SUSPENDED)
            revoked_keys = sum(1 for k in self._keys.values() if k.status == ApiKeyStatus.REVOKED)
            expired_keys = sum(1 for k in self._keys.values() if k.is_expired())

            total_requests = sum(k.total_requests for k in self._keys.values())
            total_tokens = sum(k.total_prompt_tokens + k.total_completion_tokens for k in self._keys.values())

            keys_by_role = defaultdict(int)
            for k in self._keys.values():
                keys_by_role[k.role.value] += 1

        return {
            "total_keys": total_keys,
            "active_keys": active_keys,
            "suspended_keys": suspended_keys,
            "revoked_keys": revoked_keys,
            "expired_keys": expired_keys,
            "total_requests_all_keys": total_requests,
            "total_tokens_all_keys": total_tokens,
            "keys_by_role": dict(keys_by_role),
            "audit_log_entries": len(self._audit_log),
        }

    @property
    def enabled(self) -> bool:
        """Check if API key management is enabled."""
        return self.require_api_key or bool(self._keys)
