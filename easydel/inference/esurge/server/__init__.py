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

"""eSurge Server module providing OpenAI-compatible API endpoints.

This module exposes the eSurge inference engine through a FastAPI-based REST API
server that is fully compatible with the OpenAI API specification. It provides
comprehensive authentication, rate limiting, quota management, and audit logging
capabilities for production deployments.

Key Components:
    eSurgeApiServer: Main API server class with OpenAI-compatible endpoints.
    eSurgeAdapter: Adapter bridging eSurge engine with the API server infrastructure.
    AuthEndpointsMixin: Mixin providing admin endpoints for API key management.
    CreateApiKeyRequest: Pydantic model for API key creation requests.
    UpdateApiKeyRequest: Pydantic model for API key update requests.

Authentication System:
    The module integrates with the enhanced authentication system from
    `easydel.workers.esurge.auth`, providing:
    - Role-based access control (RBAC) with Admin, User, and Service roles
    - Rate limiting at multiple time granularities (minute, hour, day)
    - Token and request quotas with monthly reset capabilities
    - IP-based access control (allowlists and blocklists)
    - Comprehensive audit logging for compliance

Example:
    Basic server setup with a single model::

        from easydel.inference.esurge import eSurge
        from easydel.inference.esurge.server import eSurgeApiServer

        # Create eSurge instance
        esurge = eSurge.from_pretrained("model-name")

        # Create and run server
        server = eSurgeApiServer(esurge, require_api_key=True)
        server.run(host="0.0.0.0", port=8000)

    Multi-model deployment::

        esurge_map = {
            "gpt-4": eSurge.from_pretrained("model-a"),
            "gpt-3.5-turbo": eSurge.from_pretrained("model-b"),
        }
        server = eSurgeApiServer(esurge_map, enable_function_calling=True)

See Also:
    - `easydel.inference.esurge.esurge_engine`: Core eSurge inference engine
    - `easydel.workers.esurge.auth`: Authentication and authorization components
"""

# Import auth components from workers
from easydel.workers.esurge.auth import (
    ApiKeyMetadata,
    ApiKeyPermissions,
    ApiKeyRole,
    ApiKeyStatus,
    AuthStorage,
    EnhancedApiKeyManager,
    PermissionDenied,
    QuotaConfig,
    QuotaExceeded,
    RateLimitConfig,
    RateLimitExceeded,
)

from .api_server import ServerStatus, create_error_response, eSurgeAdapter, eSurgeApiServer
from .auth_endpoints import AuthEndpointsMixin, CreateApiKeyRequest, UpdateApiKeyRequest

__all__ = (
    "ApiKeyMetadata",
    "ApiKeyPermissions",
    "ApiKeyRole",
    "ApiKeyStatus",
    "AuthEndpointsMixin",
    "AuthStorage",
    "CreateApiKeyRequest",
    "EnhancedApiKeyManager",
    "PermissionDenied",
    "QuotaConfig",
    "QuotaExceeded",
    "RateLimitConfig",
    "RateLimitExceeded",
    "ServerStatus",
    "ToolParserWorkerClient",
    "ToolParserWorkerManager",
    "UpdateApiKeyRequest",
    "create_error_response",
    "eSurgeAdapter",
    "eSurgeApiServer",
)
