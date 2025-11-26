"""API key management utilities for eSurge servers."""

from __future__ import annotations

import secrets
import threading
import time
import typing as tp
from dataclasses import dataclass, field


@dataclass
class ApiKeyUsage:
    """Tracks usage statistics for a single API key."""

    key: str
    label: str | None = None
    created_at: float = field(default_factory=time.time)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_requests: int = 0

    def as_dict(self) -> dict[str, tp.Any]:
        """Serialize usage data for monitoring endpoints."""
        return {
            "key": self.key,
            "label": self.label,
            "created_at": self.created_at,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_requests": self.total_requests,
        }


class ApiKeyManager:
    """Runtime API key registry with usage tracking."""

    def __init__(
        self,
        api_keys: tp.Sequence[str] | dict[str, tp.Mapping[str, tp.Any]] | None = None,
        require_api_key: bool = False,
    ) -> None:
        self.require_api_key = require_api_key
        self._keys: dict[str, ApiKeyUsage] = {}
        self._lock = threading.Lock()

        if api_keys:
            self._load_initial_keys(api_keys)

    def _load_initial_keys(
        self,
        api_keys: tp.Sequence[str] | dict[str, tp.Mapping[str, tp.Any]],
    ) -> None:
        if isinstance(api_keys, dict):
            iterator = api_keys.items()
        else:
            iterator = ((key, None) for key in api_keys)

        for key, metadata in iterator:
            if not isinstance(key, str) or not key:
                continue

            label = None
            if isinstance(metadata, dict):
                label = tp.cast(str | None, metadata.get("label"))

            self._keys[key] = ApiKeyUsage(key=key, label=label)

    def generate_api_key(self, label: str | None = None) -> ApiKeyUsage:
        """Create a new random API key entry."""
        key = secrets.token_urlsafe(32)
        return self.register_api_key(key=key, label=label)

    def register_api_key(self, key: str, label: str | None = None) -> ApiKeyUsage:
        """Register a user-provided API key."""
        if not key:
            raise ValueError("API key cannot be empty")

        with self._lock:
            usage = self._keys.get(key)
            if usage:
                if label:
                    usage.label = label
                return usage

            usage = ApiKeyUsage(key=key, label=label)
            self._keys[key] = usage
            return usage

    def validate_key(self, key: str | None) -> ApiKeyUsage | None:
        """Return usage entry for a key if it exists."""
        if not key:
            return None
        return self._keys.get(key)

    def record_usage(self, key: str | None, prompt_tokens: int, completion_tokens: int) -> None:
        """Update usage stats for a key."""
        if not key:
            return

        usage = self._keys.get(key)
        if usage is None:
            return

        with self._lock:
            usage.prompt_tokens += max(prompt_tokens, 0)
            usage.completion_tokens += max(completion_tokens, 0)
            usage.total_requests += 1

    @property
    def enabled(self) -> bool:
        return self.require_api_key or bool(self._keys)

    def usage_snapshot(self) -> dict[str, dict[str, tp.Any]]:
        """Return a serializable snapshot of all usage counters."""
        with self._lock:
            return {key: usage.as_dict() for key, usage in self._keys.items()}
