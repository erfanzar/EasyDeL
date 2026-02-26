"""API key management utilities for eSurge servers.

This module provides lightweight API key management functionality for tracking
and validating API keys used to authenticate requests to eSurge inference servers.
It offers basic key registration, validation, and usage tracking capabilities.

For production deployments requiring advanced features like rate limiting, quotas,
RBAC, and persistent storage, use the enhanced authentication system from
`easydel.workers.esurge.auth` instead.

Classes:
    ApiKeyUsage: Dataclass tracking usage statistics for a single API key.
    ApiKeyManager: Thread-safe registry for managing API keys and their usage.

Example:
    Basic usage with key generation::

        manager = ApiKeyManager(require_api_key=True)
        usage = manager.generate_api_key(label="production-client")
        print(f"Generated key: {usage.key}")

    Pre-loading keys from configuration::

        keys = {
            "sk-abc123": {"label": "client-a"},
            "sk-xyz789": {"label": "client-b"},
        }
        manager = ApiKeyManager(api_keys=keys, require_api_key=True)

Note:
    This module is intended for simple use cases. For enterprise deployments,
    consider using `EnhancedApiKeyManager` from the workers auth module which
    provides persistent storage, audit logging, and advanced access control.
"""

from __future__ import annotations

import secrets
import threading
import time
import typing as tp
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


@dataclass
class ApiKeyUsage:
    """Tracks usage statistics for a single API key.

    This dataclass maintains cumulative usage metrics for an API key including
    token counts and request totals. It is thread-safe when used with the
    ApiKeyManager which handles synchronization.

    Attributes:
        key: The API key string (typically a secure random token).
        label: Optional human-readable label for identifying the key's purpose.
        created_at: Unix timestamp when the key was registered.
        prompt_tokens: Cumulative count of prompt tokens processed.
        completion_tokens: Cumulative count of completion tokens generated.
        total_requests: Total number of requests made with this key.

    Example:
        Creating and tracking usage::

            usage = ApiKeyUsage(key="sk-abc123", label="production")
            usage.prompt_tokens += 100
            usage.completion_tokens += 50
            usage.total_requests += 1
            print(usage.as_dict())
    """

    key: str
    label: str | None = None
    created_at: float = field(default_factory=time.time)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_requests: int = 0

    def as_dict(self) -> dict[str, tp.Any]:
        """Serialize usage data to a dictionary for monitoring endpoints.

        Converts all usage statistics to a JSON-serializable dictionary format
        suitable for API responses and logging.

        Returns:
            Dictionary containing all usage fields with their current values:
                - key: The API key string
                - label: Optional key label
                - created_at: Unix timestamp of creation
                - prompt_tokens: Total prompt tokens used
                - completion_tokens: Total completion tokens generated
                - total_requests: Total request count
        """
        return {
            "key": self.key,
            "label": self.label,
            "created_at": self.created_at,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_requests": self.total_requests,
        }


class ApiKeyManager:
    """Thread-safe runtime API key registry with usage tracking.

    This class provides a simple in-memory key management system for authenticating
    API requests. It supports pre-loading keys from configuration, generating new
    random keys, and tracking usage statistics per key.

    The manager is thread-safe and suitable for multi-threaded server environments.
    All key operations are protected by a lock to prevent race conditions.

    Attributes:
        require_api_key: If True, authentication is mandatory for all requests.

    Example:
        Creating a manager with pre-loaded keys::

            manager = ApiKeyManager(
                api_keys={"sk-secret": {"label": "prod"}},
                require_api_key=True
            )

        Generating and using keys::

            manager = ApiKeyManager(require_api_key=True)
            usage = manager.generate_api_key(label="new-client")

            # Later, validate incoming requests
            if manager.validate_key(request_key):
                # Process request
                manager.record_usage(request_key, prompt_tokens=100, completion_tokens=50)

    Note:
        This is a basic implementation without persistence. Keys are lost when
        the process terminates. For persistent key storage, use the enhanced
        authentication system from `easydel.workers.esurge.auth`.
    """

    def __init__(
        self,
        api_keys: Sequence[str] | dict[str, Mapping[str, tp.Any]] | None = None,
        require_api_key: bool = False,
    ) -> None:
        """Initialize the API key manager.

        Args:
            api_keys: Initial keys to register. Can be:
                - A sequence of key strings (labels will be None)
                - A dict mapping key strings to metadata dicts with optional "label" field
                - None for no pre-loaded keys
            require_api_key: If True, all requests must include a valid API key.
                If False, requests without keys may be allowed when no keys are registered.
        """
        self.require_api_key = require_api_key
        self._keys: dict[str, ApiKeyUsage] = {}
        self._lock = threading.Lock()

        if api_keys:
            self._load_initial_keys(api_keys)

    def _load_initial_keys(
        self,
        api_keys: Sequence[str] | dict[str, Mapping[str, tp.Any]],
    ) -> None:
        """Load initial API keys into the registry.

        Processes the provided keys and creates ApiKeyUsage entries for each.
        Empty or non-string keys are silently skipped.

        Args:
            api_keys: Keys to load, either as a sequence of strings or a dict
                mapping key strings to metadata dictionaries.
        """
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
        """Generate and register a new cryptographically secure API key.

        Creates a new random API key using `secrets.token_urlsafe()` which
        produces a URL-safe base64-encoded string suitable for use in HTTP
        headers and query parameters.

        Args:
            label: Optional human-readable label for the key.

        Returns:
            ApiKeyUsage object containing the new key and its initial statistics.

        Example:
            >>> manager = ApiKeyManager()
            >>> usage = manager.generate_api_key(label="mobile-app")
            >>> print(f"New key: {usage.key}")
        """
        key = secrets.token_urlsafe(32)
        return self.register_api_key(key=key, label=label)

    def register_api_key(self, key: str, label: str | None = None) -> ApiKeyUsage:
        """Register a user-provided API key in the registry.

        If the key already exists, updates its label (if provided) and returns
        the existing usage entry. Otherwise, creates a new entry.

        Args:
            key: The API key string to register. Must be non-empty.
            label: Optional human-readable label for the key.

        Returns:
            ApiKeyUsage object for the registered key.

        Raises:
            ValueError: If the key is empty or None.

        Example:
            >>> manager = ApiKeyManager()
            >>> usage = manager.register_api_key("sk-custom-key", label="custom")
        """
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
        """Validate an API key and return its usage entry if valid.

        Checks if the provided key exists in the registry. This method does
        not modify any state and is safe to call concurrently.

        Args:
            key: The API key to validate. Can be None.

        Returns:
            ApiKeyUsage object if the key exists, None otherwise.
            Returns None if key is None or empty.

        Example:
            >>> manager = ApiKeyManager(api_keys=["sk-valid"])
            >>> if manager.validate_key("sk-valid"):
            ...     print("Key is valid")
        """
        if not key:
            return None
        return self._keys.get(key)

    def record_usage(self, key: str | None, prompt_tokens: int, completion_tokens: int) -> None:
        """Record token usage for a completed request.

        Updates the cumulative token counts and request counter for the
        specified key. Negative token values are clamped to zero.

        This method is thread-safe and can be called concurrently from
        multiple request handlers.

        Args:
            key: The API key that made the request. If None or not found,
                the call is silently ignored.
            prompt_tokens: Number of prompt tokens consumed by the request.
            completion_tokens: Number of completion tokens generated.

        Example:
            >>> manager.record_usage("sk-key", prompt_tokens=150, completion_tokens=75)
        """
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
        """Check if authentication is enabled.

        Returns True if either `require_api_key` is True or if any keys
        have been registered. This property is used to determine whether
        incoming requests should be authenticated.

        Returns:
            True if authentication is enabled, False otherwise.
        """
        return self.require_api_key or bool(self._keys)

    def usage_snapshot(self) -> dict[str, dict[str, tp.Any]]:
        """Return a serializable snapshot of all usage counters.

        Creates a copy of the current usage statistics for all registered
        keys. Useful for monitoring endpoints and logging.

        Returns:
            Dictionary mapping API key strings to their serialized usage data.
            Each value is the result of calling `ApiKeyUsage.as_dict()`.

        Example:
            >>> snapshot = manager.usage_snapshot()
            >>> for key, usage in snapshot.items():
            ...     print(f"{key}: {usage['total_requests']} requests")
        """
        with self._lock:
            return {key: usage.as_dict() for key, usage in self._keys.items()}
