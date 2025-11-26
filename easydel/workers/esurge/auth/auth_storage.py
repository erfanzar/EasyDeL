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

"""Persistent storage for authentication data and usage tracking."""

from __future__ import annotations

import json
import threading
import time
import typing as tp
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

logger = get_logger("AuthStorage")


class AuthStorage:
    """Persistent storage manager for authentication data.

    Stores API keys, usage statistics, and audit logs to disk for persistence
    across server restarts. Data is stored in ~/.cache/esurge-auth/ by default.

    File structure:
            ~/.cache/esurge-auth/
                    keys.json           - API key metadata (without raw keys)
                    audit_logs.json     - Audit log entries
                    usage_stats.json    - Aggregated usage statistics

    Features:
    - Automatic saving on changes
    - Thread-safe operations
    - JSON serialization with proper typing
    - Atomic file writes (write to temp, then rename)
    - Automatic backup on save
    """

    def __init__(
        self,
        storage_dir: str | Path | None = None,
        auto_save: bool = True,
        save_interval: float = 60.0,
    ) -> None:
        """Initialize the auth storage manager.

        Args:
                storage_dir: Directory to store auth data. Defaults to ~/.cache/esurge-auth/
                auto_save: Enable automatic periodic saving (default: True)
                save_interval: Seconds between auto-saves (default: 60.0)
        """
        if storage_dir is None:
            storage_dir = Path.home() / ".cache" / "esurge-auth"
        else:
            storage_dir = Path(storage_dir)

        self.storage_dir = storage_dir
        self.auto_save = auto_save
        self.save_interval = save_interval

        # File paths
        self.keys_file = self.storage_dir / "keys.json"
        self.audit_logs_file = self.storage_dir / "audit_logs.json"
        self.stats_file = self.storage_dir / "usage_stats.json"

        # Thread safety
        self._lock = threading.Lock()
        self._dirty = False  # Track if data needs saving
        self._last_save = time.time()

        # Create storage directory
        self._ensure_storage_dir()

        logger.info(f"Auth storage initialized at: {self.storage_dir}")

    def _ensure_storage_dir(self) -> None:
        """Create storage directory if it doesn't exist."""
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Storage directory created/verified: {self.storage_dir}")
        except Exception as e:
            logger.error(f"Failed to create storage directory: {e}")
            raise

    def _atomic_write(self, file_path: Path, data: str) -> None:
        """Write data to file atomically (write to temp, then rename).

        Args:
                file_path: Target file path
                data: String data to write
        """
        temp_file = file_path.with_suffix(".tmp")
        backup_file = file_path.with_suffix(".bak")

        try:
            # Write to temp file
            temp_file.write_text(data, encoding="utf-8")

            # Backup existing file if it exists
            if file_path.exists():
                if backup_file.exists():
                    backup_file.unlink()
                file_path.rename(backup_file)

            # Rename temp to target
            temp_file.rename(file_path)

            logger.debug(f"Successfully saved: {file_path}")
        except Exception as e:
            logger.error(f"Failed to write {file_path}: {e}")
            # Restore backup if available
            if backup_file.exists() and not file_path.exists():
                backup_file.rename(file_path)
            raise
        finally:
            # Clean up temp file if it still exists
            if temp_file.exists():
                temp_file.unlink()

    def save_keys(self, keys: dict[str, ApiKeyMetadata]) -> None:
        """Save API key metadata to disk.

        Args:
                keys: Dictionary mapping hashed_key -> ApiKeyMetadata
        """
        with self._lock:
            try:
                # Convert to serializable format
                keys_data = {hashed_key: self._serialize_key_metadata(metadata) for hashed_key, metadata in keys.items()}

                data = {
                    "version": "1.0",
                    "saved_at": time.time(),
                    "total_keys": len(keys_data),
                    "keys": keys_data,
                }

                json_data = json.dumps(data, indent=2, ensure_ascii=False)
                self._atomic_write(self.keys_file, json_data)
                self._dirty = False
                self._last_save = time.time()
                logger.info(f"Saved {len(keys_data)} API keys to disk")
            except Exception as e:
                logger.error(f"Failed to save keys: {e}")
                raise

    def load_keys(self) -> dict[str, ApiKeyMetadata]:
        """Load API key metadata from disk.

        Returns:
                Dictionary mapping hashed_key -> ApiKeyMetadata
        """
        if not self.keys_file.exists():
            logger.info("No existing keys file found, starting fresh")
            return {}

        with self._lock:
            try:
                data = json.loads(self.keys_file.read_text(encoding="utf-8"))
                keys_data = data.get("keys", {})

                keys = {
                    hashed_key: self._deserialize_key_metadata(key_data) for hashed_key, key_data in keys_data.items()
                }

                logger.info(f"Loaded {len(keys)} API keys from disk")
                return keys
            except Exception as e:
                logger.error(f"Failed to load keys: {e}")
                # Try to load from backup
                backup_file = self.keys_file.with_suffix(".bak")
                if backup_file.exists():
                    logger.info("Attempting to load from backup...")
                    try:
                        data = json.loads(backup_file.read_text(encoding="utf-8"))
                        keys_data = data.get("keys", {})
                        keys = {
                            hashed_key: self._deserialize_key_metadata(key_data)
                            for hashed_key, key_data in keys_data.items()
                        }
                        logger.info(f"Successfully loaded {len(keys)} keys from backup")
                        return keys
                    except Exception as backup_error:
                        logger.error(f"Backup load also failed: {backup_error}")
                return {}

    def save_audit_logs(self, logs: list[AuditLogEntry]) -> None:
        """Save audit logs to disk.

        Args:
                logs: List of audit log entries
        """
        with self._lock:
            try:
                logs_data = [self._serialize_audit_log(log) for log in logs]

                data = {
                    "version": "1.0",
                    "saved_at": time.time(),
                    "total_logs": len(logs_data),
                    "logs": logs_data,
                }

                json_data = json.dumps(data, indent=2, ensure_ascii=False)
                self._atomic_write(self.audit_logs_file, json_data)
                logger.info(f"Saved {len(logs_data)} audit log entries to disk")
            except Exception as e:
                logger.error(f"Failed to save audit logs: {e}")
                raise

    def load_audit_logs(self) -> list[AuditLogEntry]:
        """Load audit logs from disk.

        Returns:
                List of audit log entries
        """
        if not self.audit_logs_file.exists():
            logger.info("No existing audit logs file found, starting fresh")
            return []

        with self._lock:
            try:
                data = json.loads(self.audit_logs_file.read_text(encoding="utf-8"))
                logs_data = data.get("logs", [])

                logs = [self._deserialize_audit_log(log_data) for log_data in logs_data]

                logger.info(f"Loaded {len(logs)} audit log entries from disk")
                return logs
            except Exception as e:
                logger.error(f"Failed to load audit logs: {e}")
                return []

    def save_usage_stats(self, stats: dict[str, tp.Any]) -> None:
        """Save aggregated usage statistics to disk.

        Args:
                stats: Dictionary of usage statistics
        """
        with self._lock:
            try:
                data = {
                    "version": "1.0",
                    "saved_at": time.time(),
                    "stats": stats,
                }

                json_data = json.dumps(data, indent=2, ensure_ascii=False)
                self._atomic_write(self.stats_file, json_data)
                logger.debug("Saved usage statistics to disk")
            except Exception as e:
                logger.error(f"Failed to save usage stats: {e}")
                raise

    def load_usage_stats(self) -> dict[str, tp.Any]:
        """Load aggregated usage statistics from disk.

        Returns:
                Dictionary of usage statistics
        """
        if not self.stats_file.exists():
            return {}

        with self._lock:
            try:
                data = json.loads(self.stats_file.read_text(encoding="utf-8"))
                return data.get("stats", {})
            except Exception as e:
                logger.error(f"Failed to load usage stats: {e}")
                return {}

    def _serialize_key_metadata(self, metadata: ApiKeyMetadata) -> dict[str, tp.Any]:
        """Serialize ApiKeyMetadata to JSON-compatible dict."""
        return {
            "key_id": metadata.key_id,
            "key_prefix": metadata.key_prefix,
            "hashed_key": metadata.hashed_key,
            "name": metadata.name,
            "description": metadata.description,
            "role": metadata.role.value,
            "status": metadata.status.value,
            "created_at": metadata.created_at,
            "created_by": metadata.created_by,
            "expires_at": metadata.expires_at,
            "last_used_at": metadata.last_used_at,
            "last_rotated_at": metadata.last_rotated_at,
            "total_requests": metadata.total_requests,
            "total_prompt_tokens": metadata.total_prompt_tokens,
            "total_completion_tokens": metadata.total_completion_tokens,
            "monthly_requests": metadata.monthly_requests,
            "monthly_tokens": metadata.monthly_tokens,
            "last_reset_month": metadata.last_reset_month,
            "rate_limits": {
                "requests_per_minute": metadata.rate_limits.requests_per_minute,
                "requests_per_hour": metadata.rate_limits.requests_per_hour,
                "requests_per_day": metadata.rate_limits.requests_per_day,
                "tokens_per_minute": metadata.rate_limits.tokens_per_minute,
                "tokens_per_hour": metadata.rate_limits.tokens_per_hour,
                "tokens_per_day": metadata.rate_limits.tokens_per_day,
            },
            "quota": {
                "max_total_tokens": metadata.quota.max_total_tokens,
                "max_total_requests": metadata.quota.max_total_requests,
                "monthly_token_limit": metadata.quota.monthly_token_limit,
                "monthly_request_limit": metadata.quota.monthly_request_limit,
            },
            "permissions": {
                "allowed_models": metadata.permissions.allowed_models,
                "allowed_endpoints": metadata.permissions.allowed_endpoints,
                "allowed_ip_addresses": metadata.permissions.allowed_ip_addresses,
                "blocked_ip_addresses": metadata.permissions.blocked_ip_addresses,
                "enable_streaming": metadata.permissions.enable_streaming,
                "enable_function_calling": metadata.permissions.enable_function_calling,
                "max_tokens_per_request": metadata.permissions.max_tokens_per_request,
            },
            "tags": metadata.tags,
            "metadata": metadata.metadata,
        }

    def _deserialize_key_metadata(self, data: dict[str, tp.Any]) -> ApiKeyMetadata:
        """Deserialize JSON dict to ApiKeyMetadata."""
        return ApiKeyMetadata(
            key_id=data["key_id"],
            key_prefix=data["key_prefix"],
            hashed_key=data["hashed_key"],
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
            last_reset_month=data.get("last_reset_month", time.localtime().tm_mon),
            rate_limits=RateLimitConfig(**data.get("rate_limits", {})),
            quota=QuotaConfig(**data.get("quota", {})),
            permissions=ApiKeyPermissions(**data.get("permissions", {})),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )

    def _serialize_audit_log(self, log: AuditLogEntry) -> dict[str, tp.Any]:
        """Serialize AuditLogEntry to JSON-compatible dict."""
        return {
            "timestamp": log.timestamp,
            "key_id": log.key_id,
            "action": log.action,
            "actor": log.actor,
            "ip_address": log.ip_address,
            "details": log.details,
            "success": log.success,
        }

    def _deserialize_audit_log(self, data: dict[str, tp.Any]) -> AuditLogEntry:
        """Deserialize JSON dict to AuditLogEntry."""
        return AuditLogEntry(
            timestamp=data["timestamp"],
            key_id=data.get("key_id"),
            action=data["action"],
            actor=data.get("actor"),
            ip_address=data.get("ip_address"),
            details=data.get("details", {}),
            success=data.get("success", True),
        )

    def mark_dirty(self) -> None:
        """Mark data as needing save."""
        self._dirty = True

    def should_auto_save(self) -> bool:
        """Check if auto-save should be triggered.

        Returns:
                True if auto_save is enabled and interval has elapsed
        """
        if not self.auto_save:
            return False
        return self._dirty and (time.time() - self._last_save) >= self.save_interval

    def clear_all(self) -> None:
        """Clear all stored data (for testing/reset)."""
        with self._lock:
            for file_path in [self.keys_file, self.audit_logs_file, self.stats_file]:
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Deleted: {file_path}")
            logger.info("All auth storage cleared")
