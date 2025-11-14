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

"""Authentication and authorization module for eSurge workers.

This module provides a comprehensive authentication and authorization system for
eSurge workers, including:

- API key management with RBAC (Role-Based Access Control)
- Rate limiting and quota management
- Audit logging and usage tracking
- Persistent storage for auth data
- ZeroMQ-based worker process for authentication operations

Example:
    Basic usage with auth manager::

        from easydel.workers.esurge.auth import EnhancedApiKeyManager, ApiKeyRole

        # Create auth manager
        auth_manager = EnhancedApiKeyManager(
            require_api_key=True,
            admin_key="your-admin-key",
        )

        # Generate a new API key
        raw_key, metadata = auth_manager.generate_api_key(
            name="My API Key",
            role=ApiKeyRole.USER,
        )

        # Authorize a request
        metadata = auth_manager.authorize_request(
            raw_key=raw_key,
            ip_address="127.0.0.1",
            endpoint="/inference",
        )

    Using with ZeroMQ worker::

        from easydel.workers.esurge.auth import AuthWorkerManager

        # Start auth worker
        manager = AuthWorkerManager(admin_key="your-admin-key")
        client = manager.start()

        # Use client for auth operations
        raw_key, metadata = client.generate_api_key(
            name="Test Key",
            role=ApiKeyRole.USER,
        )

        # Shutdown when done
        manager.shutdown()
"""

from .auth_manager import (
    EnhancedApiKeyManager,
    PermissionDenied,
    QuotaExceeded,
    RateLimitExceeded,
)
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
from .worker_manager import AuthWorkerManager
from .zmq_client import AuthWorkerClient

__all__ = [
    "ApiKeyMetadata",
    "ApiKeyPermissions",
    "ApiKeyRole",
    "ApiKeyStatus",
    "AuditLogEntry",
    "AuthStorage",
    "AuthWorkerClient",
    "AuthWorkerManager",
    "EnhancedApiKeyManager",
    "PermissionDenied",
    "QuotaConfig",
    "QuotaExceeded",
    "RateLimitConfig",
    "RateLimitExceeded",
]
