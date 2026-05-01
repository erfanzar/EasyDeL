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
    """Raised when an API key exceeds one of its sliding-window rate limits.

    Thrown by :meth:`EnhancedApiKeyManager._check_rate_limits` when the
    accumulated request count or token usage in the current minute / hour
    / day window meets or exceeds the corresponding limit configured on
    the key's :class:`RateLimitConfig`. The exception ``args[0]`` carries a
    human-readable message naming the breached window so handlers can
    surface ``Retry-After``-style hints.
    """

    pass


class QuotaExceeded(Exception):
    """Raised when an API key exceeds a lifetime or monthly quota.

    Thrown by :meth:`EnhancedApiKeyManager._check_quotas` when the total or
    monthly usage tracked on :class:`ApiKeyMetadata` would surpass the
    limits set in its :class:`QuotaConfig`. Distinct from
    :class:`RateLimitExceeded`, which only applies to short rolling
    windows; quotas are cumulative counters and recover only on monthly
    reset (or never, for lifetime limits).
    """

    pass


class PermissionDenied(Exception):
    """Raised when an authorisation check rejects a request.

    Used by :meth:`EnhancedApiKeyManager.authorize_request` for any
    rejection that is *not* a rate-limit or quota breach: invalid /
    inactive key, IP allow/blocklist failure, endpoint or model not
    permitted by the key's :class:`ApiKeyPermissions`, or per-request
    token ceiling violation. The audit log records the specific reason
    via the ``details`` payload so callers can map it back to a 401/403
    response.
    """

    pass


class EnhancedApiKeyManager:
    """In-process API key manager with RBAC, rate limiting, and audit logging.

    Acts as the single source of truth for API key state in the eSurge
    auth subsystem. Lives either inside the API server process (for
    embedded deployments) or behind :class:`AuthWorkerManager` /
    :class:`AuthWorkerClient` so multiple API workers can share state.

    A single manager owns:

    * The hashed-key store (``_keys`` mapping ``hashed_key -> ApiKeyMetadata``)
      and the ``key_id -> hashed_key`` index used for O(1) lookup by id.
    * A raw-key cache (``_raw_to_hash_cache``) so :meth:`validate_key`
      avoids hashing the same secret on every request.
    * Per-key sliding-window deques in ``_rate_limit_windows`` (request
      counts) and ``_token_usage_windows`` (token counts) used by
      :meth:`_check_rate_limits`.
    * A bounded :class:`AuditLogEntry` deque for in-memory audit
      history.
    * An optional :class:`AuthStorage` companion that persists keys,
      audit logs, and aggregate statistics to disk and reloads them on
      restart.

    All mutating operations acquire :attr:`_lock` (a re-entrant lock so
    auto-save callbacks can fire from inside a held section without
    deadlocking) and mark the storage dirty so the next auto-save tick
    flushes them.
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
        """Build the manager and (optionally) hydrate it from disk.

        Constructs the in-memory key store, initialises the bounded
        audit-log deque, prepares the per-key sliding-window deques used
        by :meth:`_check_rate_limits`, and - when ``enable_persistence``
        is set - opens an :class:`AuthStorage` and reloads any keys /
        audit logs from previous runs. If ``admin_key`` is provided, an
        ``ADMIN``-role record for it is created (or refreshed) after the
        reload.

        The internal lock is a :class:`threading.RLock` because mutating
        helpers re-enter via the auto-save path.

        Args:
            require_api_key: When ``True``, the API server using this
                manager rejects requests that lack a valid key. When
                ``False``, validation is best-effort and unauthenticated
                callers are still served (useful for local dev).
            admin_key: Optional bootstrap admin key. If supplied and not
                already present, a record with
                :data:`ApiKeyRole.ADMIN` is created.
            enable_audit_logging: Whether mutating operations and
                authorisation outcomes are recorded in the in-memory
                audit log (and persisted, when storage is enabled).
            max_audit_entries: Capacity of the bounded audit-log deque.
                Older entries are evicted FIFO once the limit is hit.
            storage_dir: Filesystem location for persistent auth data.
                Defaults to ``~/.cache/esurge-auth/`` when ``None`` and
                ``enable_persistence`` is ``True``.
            enable_persistence: Whether to attach an :class:`AuthStorage`
                and reload state from disk. Disable for stateless tests.
            auto_save: Whether the storage should periodically flush
                dirty state (only meaningful when persistence is on).
            save_interval: Minimum seconds between auto-saves. Combined
                with the ``_dirty`` flag to throttle write amplification.
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
            str: Hexadecimal SHA-256 digest used as the in-memory and
            on-disk identifier for the key.
        """
        return hashlib.sha256(key.encode()).hexdigest()

    def _load_from_storage(self) -> None:
        """Repopulate the in-memory state from :class:`AuthStorage`.

        Loads keys (and their reverse-lookup indices) and the recent audit
        log entries. Failures are logged and swallowed so the manager still
        starts up with an empty state.
        """
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
        """Flush keys, audit logs and aggregate stats to disk.

        No-op when persistence is disabled. Errors are logged but never
        raised so a transient I/O failure cannot bring down the auth path.
        """
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
        """Run :meth:`_save_to_storage` when the auto-save interval has elapsed.

        Driven by :meth:`AuthStorage.should_auto_save`; safe to call from
        every mutating code path.
        """
        if self.storage and self.storage.should_auto_save():
            self._save_to_storage()

    def _mark_dirty_and_save(self) -> None:
        """Mark storage as dirty and possibly auto-save.

        Convenience helper used from every mutating method; combines
        ``storage.mark_dirty`` with :meth:`_auto_save_if_needed`.
        """
        if self.storage:
            self.storage.mark_dirty()
            self._auto_save_if_needed()

    def _create_initial_admin_key(self, key: str) -> None:
        """Bootstrap an admin key on first manager start.

        If the supplied key already exists in storage, only the in-memory
        cache is refreshed; otherwise a new ``ADMIN``-role record is
        created.

        Args:
            key: Raw admin key string supplied to the constructor.
        """
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
        """Append a single :class:`AuditLogEntry` to the in-memory ring.

        No-op when ``enable_audit_logging`` is ``False``. The bounded
        ``deque`` evicts the oldest entry when the configured maximum
        is reached, keeping the manager memory-bounded under sustained
        traffic.

        Args:
            action: Short slug describing the event,
                e.g. ``"key_created"``, ``"request_denied"``,
                ``"key_rotated"``. Used as the primary filter by
                :meth:`get_audit_logs`.
            key_id: Optional internal key identifier the action targets.
            actor: Optional user / service name that initiated the
                action; ``None`` for anonymous or system events.
            ip_address: Optional client IP for request-level events.
            details: Extra structured context (reason codes, target
                model names, ...). ``None`` is normalized to ``{}``.
            success: ``True`` for happy-path events, ``False`` for
                rejections.
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
        """Mint a fresh ``sk-...`` key with cryptographically random secret.

        Generates a 384-bit URL-safe token (``secrets.token_urlsafe(48)``)
        prefixed with ``"sk-"`` and forwards the rest of the arguments
        to :meth:`create_api_key`. The returned ``raw_key`` is the
        *only* point at which the secret is observable; subsequent
        operations only ever store and compare its SHA-256 digest.

        Args:
            name: Human-readable label shown in admin UIs.
            role: Access control role. Defaults to :data:`ApiKeyRole.USER`.
            description: Optional free-form description.
            created_by: Optional creator identifier (user / service);
                stored on the metadata and emitted in the audit log.
            expires_in_days: Optional time-to-live in days; ``None``
                means the key never expires.
            rate_limits: Optional :class:`RateLimitConfig` overriding
                the default open-ended limits.
            quota: Optional :class:`QuotaConfig` overriding the default
                open-ended quota.
            permissions: Optional :class:`ApiKeyPermissions` for
                model/endpoint/IP allowlists and per-request token caps.
            tags: Optional organizational tags (used as filters by
                :meth:`list_keys`).
            metadata: Optional arbitrary user-defined metadata payload.

        Returns:
            tuple[str, ApiKeyMetadata]: ``(raw_key, metadata)``. The raw
            key MUST be returned to the requesting user immediately and
            never logged or stored - only its hash survives in
            :attr:`_keys`.
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
        """Register a caller-supplied key (e.g. for migration / bootstrap).

        Used internally by :meth:`generate_api_key` and by
        :meth:`_create_initial_admin_key` when a deployment supplies an
        out-of-band admin secret. Hashes ``raw_key``, builds an
        :class:`ApiKeyMetadata` record (assigning a fresh
        ``"key_<hex>"`` id and recording the first 12 chars as a
        display prefix), inserts it into the in-memory store under the
        manager lock, and audits as ``"key_created"``.

        Args:
            raw_key: The secret to register. Must be at least 16
                characters; only the SHA-256 hash is stored.
            name: Human-readable label.
            role: Access control role.
            description: Optional description.
            created_by: Creator user / service name (audit log).
            expires_in_days: Optional TTL converted into an absolute
                ``expires_at`` Unix timestamp; ``None`` for no expiry.
            rate_limits, quota, permissions: Optional configuration
                objects (defaults to all-open when ``None``).
            tags: Optional organizational tags.
            metadata: Optional user-defined payload.

        Returns:
            ApiKeyMetadata: The newly inserted record.

        Raises:
            ValueError: If ``raw_key`` is shorter than 16 characters or
                its hash is already registered.
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
        """Resolve a raw API key string to its live metadata, or reject it.

        Hashes ``raw_key`` (caching the digest in
        ``_raw_to_hash_cache`` for subsequent lookups), then verifies
        that the corresponding key exists *and* satisfies
        :meth:`ApiKeyMetadata.is_active`, i.e. status is ``ACTIVE`` and
        the expiration timestamp has not passed. Returns ``None`` for
        any of: missing input, unknown key, revoked / suspended /
        expired key.

        Args:
            raw_key: The raw secret presented by the client (typically
                from an ``Authorization: Bearer`` header). ``None`` and
                empty string are accepted and short-circuited.

        Returns:
            ApiKeyMetadata | None: The live metadata when authorisation
            should proceed; ``None`` to refuse the request.
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
        """Run the full authorisation pipeline for an incoming request.

        Sequentially applies, in order: :meth:`validate_key`, IP
        allow/blocklist (:meth:`_check_ip_permissions`), endpoint
        allowlist (:meth:`_check_endpoint_permissions`), model
        allowlist (:meth:`_check_model_permissions`), rate limits
        (:meth:`_check_rate_limits`), cumulative quotas
        (:meth:`_check_quotas`), and finally the per-request token
        ceiling. Each rejection is recorded in the audit log with the
        specific ``reason`` so downstream observability tools can map
        denials back to the failing check.

        Args:
            raw_key: Bearer token presented by the client. ``None`` /
                empty fails immediately with :class:`PermissionDenied`.
            ip_address: Client IP for allow/blocklist enforcement.
                ``None`` skips IP checks (e.g. intra-process callers).
            endpoint: Path being accessed; matched against the key's
                ``allowed_endpoints``.
            model: Model name being requested; matched against the
                key's ``allowed_models``.
            requested_tokens: Estimated token cost; checked against the
                key's per-request cap and rate-limit / quota windows.

        Returns:
            ApiKeyMetadata: The authorising key's metadata, with
            ``last_used_at`` updated to ``time.time()``.

        Raises:
            PermissionDenied: For invalid keys, IP / endpoint / model
                rejections, or per-request token-ceiling violations.
            RateLimitExceeded: When a sliding-window limit fires.
            QuotaExceeded: When a cumulative quota is breached.
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
        self._check_rate_limits(metadata, requested_tokens)

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
        """Decide whether ``ip_address`` is permitted to use the key.

        Args:
            metadata: The key being checked.
            ip_address: Client IP. ``None`` always passes (e.g. for
                process-local clients).

        Returns:
            bool: ``True`` when the IP is not in the blocklist and either
            no allowlist is set or the IP is on it.
        """
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
        """Decide whether ``endpoint`` is permitted by the key.

        Args:
            metadata: The key being checked.
            endpoint: Endpoint path. ``None`` always passes.

        Returns:
            bool: ``True`` when no allowlist is configured or the endpoint
            is on it.
        """
        if endpoint is None:
            return True

        permissions = metadata.permissions
        if permissions.allowed_endpoints is None:
            return True

        return endpoint in permissions.allowed_endpoints

    def _check_model_permissions(self, metadata: ApiKeyMetadata, model: str | None) -> bool:
        """Decide whether ``model`` is permitted by the key.

        Args:
            metadata: The key being checked.
            model: Requested model name. ``None`` always passes.

        Returns:
            bool: ``True`` when no allowlist is configured or the model is
            on it.
        """
        if model is None:
            return True

        permissions = metadata.permissions
        if permissions.allowed_models is None:
            return True

        return model in permissions.allowed_models

    def _check_rate_limits(self, metadata: ApiKeyMetadata, requested_tokens: int = 0) -> None:
        """Validate every configured rolling-window rate limit for a key.

        Walks the per-key request and token sliding windows
        (minute / hour / day) maintained in
        ``_rate_limit_windows`` and ``_token_usage_windows``. For request
        windows the call appends a timestamp on success; for token
        windows the call only *checks* the projected total
        (``current_used + requested_tokens``) - the actual append is
        deferred to :meth:`record_usage` once the real token count is
        known.

        Args:
            metadata: Live metadata for the key being charged. Its
                ``rate_limits`` (a :class:`RateLimitConfig`) drives which
                windows are inspected.
            requested_tokens: Estimated upper bound on tokens this
                request will consume; clamped to ``>= 0`` before use.
                Pass ``0`` when the request has no token component.

        Raises:
            RateLimitExceeded: When any of the configured windows would
                be saturated by serving this request. The first breached
                window wins; the message identifies it
                (e.g. ``"... 1000 requests/hour"``).
        """
        rate_limits = metadata.rate_limits
        current_time = time.time()
        key_id = metadata.key_id
        requested_tokens = max(requested_tokens, 0)

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

        # Check token rate limits (pre-check using requested_tokens)
        if rate_limits.tokens_per_minute:
            window = self._token_usage_windows[key_id]["tokens_minute"]
            self._clean_window(window, current_time, 60)
            used = sum(tokens for _, tokens in window)
            if used + requested_tokens > rate_limits.tokens_per_minute:
                raise RateLimitExceeded(f"Rate limit exceeded: {rate_limits.tokens_per_minute} tokens/minute")

        if rate_limits.tokens_per_hour:
            window = self._token_usage_windows[key_id]["tokens_hour"]
            self._clean_window(window, current_time, 3600)
            used = sum(tokens for _, tokens in window)
            if used + requested_tokens > rate_limits.tokens_per_hour:
                raise RateLimitExceeded(f"Rate limit exceeded: {rate_limits.tokens_per_hour} tokens/hour")

        if rate_limits.tokens_per_day:
            window = self._token_usage_windows[key_id]["tokens_day"]
            self._clean_window(window, current_time, 86400)
            used = sum(tokens for _, tokens in window)
            if used + requested_tokens > rate_limits.tokens_per_day:
                raise RateLimitExceeded(f"Rate limit exceeded: {rate_limits.tokens_per_day} tokens/day")

    def _check_quotas(self, metadata: ApiKeyMetadata, requested_tokens: int) -> None:
        """Validate every configured cumulative quota for a key.

        Calls :meth:`ApiKeyMetadata.reset_monthly_counters_if_needed`
        first so the monthly counters reflect the current calendar
        month, then compares the projected totals against the limits in
        ``metadata.quota`` (:class:`QuotaConfig`). Lifetime token / request
        limits and monthly token / request limits are all checked; the
        first breach raises.

        Unlike rate limits, quotas are not sliding windows - they only
        recover via the monthly reset (or never, for lifetime limits).

        Args:
            metadata: Live metadata for the key being charged.
            requested_tokens: Token cost projected for this request.

        Raises:
            QuotaExceeded: When any quota would be breached by serving
                this request. The message names the quota
                (e.g. ``"Monthly token quota exceeded: 1_000_000"``).
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
        """Drop entries older than ``current_time - window_size`` from a deque.

        Supports both bare-timestamp deques (for request rate limits) and
        ``(timestamp, tokens)`` deques (for token rate limits).

        Args:
            window: Sliding-window deque to prune in-place.
            current_time: Reference time, typically ``time.time()``.
            window_size: Window length in seconds.
        """
        cutoff = current_time - window_size
        while window:
            head = window[0]
            timestamp = head[0] if isinstance(head, (tuple, list)) else head
            if timestamp < cutoff:
                window.popleft()
            else:
                break

    def record_usage(
        self,
        raw_key: str | None,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Bump per-key counters after a successfully served request.

        Updates lifetime and monthly request / token totals on the
        :class:`ApiKeyMetadata`, appends an entry to the appropriate
        rolling token rate-limit window, and marks storage dirty so the
        next auto-save flushes the new totals to disk. Negative inputs
        are clamped to zero. Silently no-ops when ``raw_key`` is empty
        or no metadata is found (e.g. the key was deleted between
        authorisation and accounting).

        Args:
            raw_key: The raw secret used for the served request; rebuilt
                back into a metadata record via :meth:`validate_key`.
            prompt_tokens: Number of input tokens consumed.
            completion_tokens: Number of output tokens generated.
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
        self._mark_dirty_and_save()

    def _record_token_rate_limit(self, metadata: ApiKeyMetadata, tokens: int) -> None:
        """Append a ``(timestamp, tokens)`` entry to the active token windows.

        Args:
            metadata: Key whose rate-limit windows are updated.
            tokens: Token count for this request, used by future
                :meth:`_check_rate_limits` calls.
        """
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
        """Permanently disable an API key.

        Marks the key's status as :data:`ApiKeyStatus.REVOKED`; revoked
        keys are kept on disk for auditability but :meth:`validate_key`
        will refuse them and they cannot be reactivated (use
        :meth:`reactivate_key` only on suspended keys). The action is
        recorded in the audit log under ``"key_revoked"``.

        Args:
            key_id: Internal key identifier returned by
                :meth:`generate_api_key` / :meth:`create_api_key` (e.g.
                ``"key_abc..."``); not the raw secret.
            revoked_by: Optional actor (user or service name) recorded in
                the audit log. ``None`` leaves the actor unset.

        Returns:
            bool: ``True`` when the key existed and was revoked,
            ``False`` when no key with the given id was found.
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
        """Temporarily disable an API key.

        Sets the status to :data:`ApiKeyStatus.SUSPENDED` so subsequent
        :meth:`validate_key` calls refuse the key, but unlike
        :meth:`revoke_key` the action is reversible via
        :meth:`reactivate_key`. Logged as ``"key_suspended"``.

        Args:
            key_id: Internal key identifier.
            suspended_by: Optional actor recorded in the audit log.

        Returns:
            bool: ``True`` when the key existed and was suspended,
            ``False`` otherwise.
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
        """Move a suspended key back to :data:`ApiKeyStatus.ACTIVE`.

        Refuses to act on keys that were :meth:`revoke_key`-d (revocation
        is permanent) or that no longer exist. Logged as
        ``"key_reactivated"``.

        Args:
            key_id: Internal key identifier.
            reactivated_by: Optional actor recorded in the audit log.

        Returns:
            bool: ``True`` when the key existed, was not revoked, and
            was reactivated; ``False`` otherwise.
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
        """Hard-delete a key from the manager and forget all of its state.

        Unlike :meth:`revoke_key`, deletion drops the metadata record
        from in-memory storage, removes the raw-key cache entry, and
        clears the per-key rate-limit windows. Audit log entries
        previously logged for the key remain so the operation is still
        traceable. Logged as ``"key_deleted"``.

        Args:
            key_id: Internal key identifier.
            deleted_by: Optional actor recorded in the audit log.

        Returns:
            bool: ``True`` when the key existed and was removed,
            ``False`` otherwise.
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
        """Fetch the in-memory metadata record for a managed key.

        Lookup goes through the ``key_id -> hashed_key -> metadata``
        index maintained alongside the raw key cache, so this is O(1).
        Does not validate the key's lifecycle status; use
        :meth:`validate_key` (with the raw key) when authorising
        requests.

        Args:
            key_id: Internal key identifier (``"key_..."``).

        Returns:
            ApiKeyMetadata | None: The live metadata object, or ``None``
            if no key with that id is registered.
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
        """Snapshot all keys, optionally filtered by role / status / tags.

        Iterates the in-memory key store under the manager lock and
        returns a list copy so callers can mutate the result without
        affecting live state. Filters are AND-ed together; tag filtering
        requires the candidate key to carry every tag in ``tags``.

        Args:
            role: Restrict to keys with this :class:`ApiKeyRole`. ``None``
                disables role filtering.
            status: Restrict to keys with this :class:`ApiKeyStatus`.
                ``None`` disables status filtering.
            tags: Restrict to keys whose ``tags`` list is a superset of
                this iterable. ``None`` or empty disables tag filtering.

        Returns:
            list[ApiKeyMetadata]: Matching key records, in the same
            order they appear in the in-memory store.
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
        """Apply a partial update to a key's metadata.

        Each ``None`` argument is treated as "leave alone"; non-``None``
        values overwrite the corresponding field on the key's
        :class:`ApiKeyMetadata`. The ``metadata`` dict is *merged*
        rather than replaced so callers can layer additional context
        without clobbering existing entries. The applied changes are
        captured in the audit log under ``"key_updated"``.

        Args:
            key_id: Internal key identifier to update.
            name: New display name; ``None`` keeps the existing one.
            description: New description.
            role: New :class:`ApiKeyRole`.
            expires_in_days: New TTL in days; converted to an absolute
                Unix timestamp relative to ``time.time()``.
            rate_limits: New :class:`RateLimitConfig` (full replacement).
            quota: New :class:`QuotaConfig` (full replacement).
            permissions: New :class:`ApiKeyPermissions` (full replacement).
            tags: New tag list (full replacement).
            metadata: User metadata patch merged into the existing dict.
            updated_by: Optional actor recorded in the audit log.

        Returns:
            bool: ``True`` when the key existed and was updated,
            ``False`` when no key with the given id is registered.
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
        """Issue a fresh secret for an existing key while preserving its metadata.

        Generates a new ``sk-...`` token, replaces the in-memory
        hashed-key index entry, updates the display prefix and
        ``last_rotated_at`` timestamp, and audits as ``"key_rotated"``.
        Lifetime / monthly counters, quotas, permissions and audit
        history are all preserved so rotation does not reset usage
        accounting.

        Args:
            key_id: Internal key identifier to rotate.
            rotated_by: Optional actor recorded in the audit log.

        Returns:
            tuple[str, ApiKeyMetadata] | None: ``(new_raw_key, metadata)``
            when rotation succeeded; ``None`` when no key with the
            given id is registered. The new raw key must be returned
            to the caller immediately - only its hash is retained.
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
        """Return audit log entries newest-first, optionally filtered.

        Reads from the bounded ``deque`` populated by
        :meth:`_log_audit`. Filtering happens after reversal so the
        ``limit`` always counts entries that match the predicates.

        Args:
            limit: Maximum number of entries to return after filtering.
                Pass a large value to dump the full window.
            key_id: When set, keep only entries whose ``key_id`` matches
                this id (e.g. for per-key audit trails).
            action: When set, keep only entries whose ``action`` matches
                exactly (e.g. ``"request_authorized"``,
                ``"key_revoked"``).

        Returns:
            list[AuditLogEntry]: Filtered entries ordered from newest to
            oldest, truncated to ``limit``.
        """
        logs = list(reversed(self._audit_log))

        if key_id:
            logs = [log for log in logs if log.key_id == key_id]
        if action:
            logs = [log for log in logs if log.action == action]

        return logs[:limit]

    def get_statistics(self) -> dict[str, tp.Any]:
        """Aggregate per-key counters into a server-wide statistics blob.

        Walks every registered key under the manager lock to compute
        lifecycle counts (active / suspended / revoked / expired),
        cumulative request and token totals, and a per-role breakdown.
        Useful for emission via the ``/admin/stats`` endpoint and for
        the periodic snapshot persisted by :class:`AuthStorage`.

        Returns:
            dict[str, Any]: Aggregate statistics with the following keys:
            ``total_keys``, ``active_keys``, ``suspended_keys``,
            ``revoked_keys``, ``expired_keys``,
            ``total_requests_all_keys``, ``total_tokens_all_keys``,
            ``keys_by_role`` (``role_name -> count``),
            ``audit_log_entries``.
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
        """Whether API key management is active.

        Returns:
            bool: ``True`` when keys are required (``require_api_key`` is
            set) or when at least one key has been registered.
        """
        return self.require_api_key or bool(self._keys)
