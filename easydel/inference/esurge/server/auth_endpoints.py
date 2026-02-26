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

"""Admin endpoints for API key management.

This module centralizes every privileged API route exposed by the inference
server: creation, rotation, suspension, auditing, and statistics. Treat it as
an operator handbook baked into code. Each handler performs its own access
control and returns structured JSON so that dashboards, CLIs, or automation can
interact with the auth subsystem without sprinkling logic across the codebase.

The file is intentionally verbose. Each endpoint documents the corresponding
HTTP method and path, describes the purpose of the operation, and highlights the
required role. That level of detail doubles as living documentation for the
security posture of the deployment—reviewers can scan this module to understand
exactly which admin capabilities exist and which invariants (audit trails,
metadata updates, rate-limit policies) are enforced at the boundary.
"""

from __future__ import annotations

import typing as tp

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import auth models from workers
from easydel.workers.esurge.auth.auth_models import (
    ApiKeyPermissions,
    ApiKeyRole,
    ApiKeyStatus,
    QuotaConfig,
    RateLimitConfig,
)


class CreateApiKeyRequest(BaseModel):
    """Request model for creating a new API key.

    All optional fields default to None, which means unlimited/unrestricted access:
    - expires_in_days=None → Key never expires
    - Rate limit fields=None → No rate limiting
    - Quota fields=None → No usage limits
    - allowed_models=None → Access to all models
    - allowed_endpoints=None → Access to all endpoints
    - allowed_ip_addresses=None → No IP restrictions
    - max_tokens_per_request=None → No per-request token limit
    """

    name: str = Field(..., description="Human-readable name for the key")
    role: ApiKeyRole = Field(ApiKeyRole.USER, description="Access control role")
    description: str | None = Field(None, description="Optional description")
    expires_in_days: int | None = Field(None, description="Days until expiration (None = never expires)")

    # Rate limits (None = unlimited)
    requests_per_minute: int | None = Field(None, description="Max requests per minute (None = unlimited)")
    requests_per_hour: int | None = Field(None, description="Max requests per hour (None = unlimited)")
    requests_per_day: int | None = Field(None, description="Max requests per day (None = unlimited)")
    tokens_per_minute: int | None = Field(None, description="Max tokens per minute (None = unlimited)")
    tokens_per_hour: int | None = Field(None, description="Max tokens per hour (None = unlimited)")
    tokens_per_day: int | None = Field(None, description="Max tokens per day (None = unlimited)")

    # Quotas (None = unlimited)
    max_total_tokens: int | None = Field(None, description="Lifetime token limit (None = unlimited)")
    max_total_requests: int | None = Field(None, description="Lifetime request limit (None = unlimited)")
    monthly_token_limit: int | None = Field(None, description="Monthly token limit (None = unlimited)")
    monthly_request_limit: int | None = Field(None, description="Monthly request limit (None = unlimited)")

    # Permissions (None = unrestricted access)
    allowed_models: list[str] | None = Field(None, description="Allowed models (None = all models)")
    allowed_endpoints: list[str] | None = Field(None, description="Allowed endpoints (None = all endpoints)")
    allowed_ip_addresses: list[str] | None = Field(
        None, description="Allowed IPs (None = all IPs). If set, only these IPs can use the key"
    )
    blocked_ip_addresses: list[str] | None = Field(None, description="Blocked IPs (None = no blocks)")
    enable_streaming: bool = Field(True, description="Allow streaming requests")
    enable_function_calling: bool = Field(True, description="Allow function calling")
    max_tokens_per_request: int | None = Field(None, description="Max tokens per single request (None = unlimited)")

    # Metadata
    tags: list[str] | None = Field(None, description="Tags for organization")
    metadata: dict[str, tp.Any] | None = Field(None, description="Additional custom metadata")


class UpdateApiKeyRequest(BaseModel):
    """Request model for updating an API key.

    Only provided fields will be updated. Omitted fields remain unchanged.
    Setting a field to None means unlimited/unrestricted:
    - expires_in_days=None → Key never expires
    - Rate limit fields=None → No rate limiting
    - Quota fields=None → No usage limits
    - allowed_models=None → Access to all models
    - allowed_endpoints=None → Access to all endpoints
    - allowed_ip_addresses=None → No IP restrictions
    """

    name: str | None = Field(None, description="Update key name")
    description: str | None = Field(None, description="Update description")
    role: ApiKeyRole | None = Field(None, description="Update role")
    expires_in_days: int | None = Field(None, description="Days until expiration (None = never expires)")

    # Rate limits (None = unlimited)
    requests_per_minute: int | None = Field(None, description="Max requests per minute (None = unlimited)")
    requests_per_hour: int | None = Field(None, description="Max requests per hour (None = unlimited)")
    requests_per_day: int | None = Field(None, description="Max requests per day (None = unlimited)")
    tokens_per_minute: int | None = Field(None, description="Max tokens per minute (None = unlimited)")
    tokens_per_hour: int | None = Field(None, description="Max tokens per hour (None = unlimited)")
    tokens_per_day: int | None = Field(None, description="Max tokens per day (None = unlimited)")

    # Quotas (None = unlimited)
    max_total_tokens: int | None = Field(None, description="Lifetime token limit (None = unlimited)")
    max_total_requests: int | None = Field(None, description="Lifetime request limit (None = unlimited)")
    monthly_token_limit: int | None = Field(None, description="Monthly token limit (None = unlimited)")
    monthly_request_limit: int | None = Field(None, description="Monthly request limit (None = unlimited)")

    # Permissions (None = unrestricted access)
    allowed_models: list[str] | None = Field(None, description="Allowed models (None = all models)")
    allowed_endpoints: list[str] | None = Field(None, description="Allowed endpoints (None = all endpoints)")
    allowed_ip_addresses: list[str] | None = Field(None, description="Allowed IPs (None = all IPs)")
    blocked_ip_addresses: list[str] | None = Field(None, description="Blocked IPs (None = no blocks)")
    enable_streaming: bool | None = Field(None, description="Allow streaming requests")
    enable_function_calling: bool | None = Field(None, description="Allow function calling")
    max_tokens_per_request: int | None = Field(None, description="Max tokens per request (None = unlimited)")

    # Metadata
    tags: list[str] | None = Field(None, description="Tags for organization")
    metadata: dict[str, tp.Any] | None = Field(None, description="Additional custom metadata")


class ApiKeyResponse(BaseModel):
    """Response model for API key operations.

    This model standardizes the structure of API key information returned
    from admin endpoints. It includes both identification metadata and
    usage statistics while omitting sensitive configuration details.

    The `key` field is only populated during key creation or rotation
    operations. In all other contexts, only the `key_prefix` is available
    for identification purposes.

    Attributes:
        key: Raw API key secret (only returned on creation/rotation).
        key_id: Unique identifier for the key record.
        key_prefix: First few characters of the key for display purposes.
        name: Human-readable name assigned to the key.
        description: Optional description of the key's purpose.
        role: Access control role (admin, user, service).
        status: Current key status (active, suspended, revoked, expired).
        created_at: Unix timestamp when the key was created.
        expires_at: Unix timestamp when the key expires, or None if no expiration.
        last_used_at: Unix timestamp of the last request using this key.
        total_requests: Cumulative count of requests made with this key.
        total_tokens: Cumulative count of tokens processed with this key.
        message: Optional status message about the operation.
    """

    key: str | None = Field(None, description="Raw API key (only returned on creation)")
    key_id: str
    key_prefix: str
    name: str
    description: str | None = None
    role: str
    status: str
    created_at: float
    expires_at: float | None = None
    last_used_at: float | None = None
    total_requests: int
    total_tokens: int
    message: str | None = None


class AuthEndpointsMixin:
    """Mixin providing admin endpoints for API key management.

    This mixin should be added to eSurgeApiServer to enable key management endpoints.
    Requires the server to have an `auth_manager` attribute of type EnhancedApiKeyManager.
    """

    def _require_admin_role(self, raw_request: Request) -> None:
        """Verify that the request is from an admin API key.

        Args:
                raw_request: FastAPI request object.

        Raises:
                HTTPException: If not authorized as admin.
        """
        # First, try to get API key from request state (if _authorize_request was already called)
        api_key = getattr(raw_request.state, "api_key", None)

        # If not set in state, extract it manually
        if not api_key:
            # Try Authorization header (Bearer token)
            auth_header = raw_request.headers.get("Authorization")
            if auth_header and auth_header.lower().startswith("bearer "):
                api_key = auth_header.split(" ", 1)[1].strip()

            # Try X-API-Key header
            if not api_key:
                api_key = raw_request.headers.get("X-API-Key")

            # Try query parameter
            if not api_key:
                api_key = raw_request.query_params.get("api_key")

        if not api_key:
            raise HTTPException(status_code=401, detail="API key required for this endpoint")

        # Validate the key and check role
        if not hasattr(self, "auth_manager"):
            raise HTTPException(status_code=500, detail="Auth manager not initialized")

        metadata = self.auth_manager.validate_key(api_key)
        if not metadata:
            raise HTTPException(status_code=401, detail="Invalid or inactive API key")

        if metadata.role != ApiKeyRole.ADMIN:
            raise HTTPException(status_code=403, detail="Admin role required for this endpoint")

        # Store in state for later use
        raw_request.state.api_key = api_key
        raw_request.state.api_key_metadata = metadata

    async def create_api_key_endpoint(self, request: CreateApiKeyRequest, raw_request: Request) -> JSONResponse:
        """Admin endpoint to create a new API key.

        POST /v1/admin/keys
        Requires admin role.

        Args:
                request: Key creation parameters.
                raw_request: FastAPI request object.

        Returns:
                JSONResponse with created key details including the raw key (only time it's shown).
        """
        self._require_admin_role(raw_request)

        # Build configuration objects
        rate_limits = RateLimitConfig(
            requests_per_minute=request.requests_per_minute,
            requests_per_hour=request.requests_per_hour,
            requests_per_day=request.requests_per_day,
            tokens_per_minute=request.tokens_per_minute,
            tokens_per_hour=request.tokens_per_hour,
            tokens_per_day=request.tokens_per_day,
        )

        quota = QuotaConfig(
            max_total_tokens=request.max_total_tokens,
            max_total_requests=request.max_total_requests,
            monthly_token_limit=request.monthly_token_limit,
            monthly_request_limit=request.monthly_request_limit,
        )

        permissions = ApiKeyPermissions(
            allowed_models=request.allowed_models,
            allowed_endpoints=request.allowed_endpoints,
            allowed_ip_addresses=request.allowed_ip_addresses,
            blocked_ip_addresses=request.blocked_ip_addresses,
            enable_streaming=request.enable_streaming,
            enable_function_calling=request.enable_function_calling,
            max_tokens_per_request=request.max_tokens_per_request,
        )

        # Get the admin's key for audit trail
        admin_key = getattr(raw_request.state, "api_key", None)
        admin_metadata = self.auth_manager.validate_key(admin_key)
        created_by = admin_metadata.name if admin_metadata else "unknown"

        # Generate the key
        raw_key, metadata = self.auth_manager.generate_api_key(
            name=request.name,
            role=request.role,
            description=request.description,
            created_by=created_by,
            expires_in_days=request.expires_in_days,
            rate_limits=rate_limits,
            quota=quota,
            permissions=permissions,
            tags=request.tags,
            metadata=request.metadata,
        )

        return JSONResponse(
            {
                "key": raw_key,  # Only time the raw key is shown!
                "key_id": metadata.key_id,
                "key_prefix": metadata.key_prefix,
                "name": metadata.name,
                "description": metadata.description,
                "role": metadata.role.value,
                "status": metadata.status.value,
                "created_at": metadata.created_at,
                "expires_at": metadata.expires_at,
                "message": "API key created successfully. Store this key securely - it won't be shown again!",
            },
            status_code=201,
        )

    async def list_api_keys_endpoint(
        self,
        raw_request: Request,
        role: str | None = None,
        status: str | None = None,
    ) -> JSONResponse:
        """Admin endpoint to list API keys.

        GET /v1/admin/keys?role=user&status=active
        Requires admin role.

        Args:
                raw_request: FastAPI request object.
                role: Optional role filter.
                status: Optional status filter.

        Returns:
                JSONResponse with list of keys (without raw keys).
        """
        self._require_admin_role(raw_request)

        role_filter = None
        if role is not None:
            try:
                role_filter = ApiKeyRole(role)
            except ValueError as exc:
                allowed_roles = ", ".join(item.value for item in ApiKeyRole)
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid role filter: {role!r}. Allowed values: {allowed_roles}",
                ) from exc

        status_filter = None
        if status is not None:
            try:
                status_filter = ApiKeyStatus(status)
            except ValueError as exc:
                allowed_statuses = ", ".join(item.value for item in ApiKeyStatus)
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status filter: {status!r}. Allowed values: {allowed_statuses}",
                ) from exc

        keys = self.auth_manager.list_keys(role=role_filter, status=status_filter)

        return JSONResponse(
            {
                "keys": [k.as_dict(include_sensitive=False) for k in keys],
                "total": len(keys),
            }
        )

    async def get_api_key_endpoint(self, key_id: str, raw_request: Request) -> JSONResponse:
        """Admin endpoint to get details of a specific API key.

        GET /v1/admin/keys/{key_id}
        Requires admin role.

        Args:
                key_id: ID of the key to retrieve.
                raw_request: FastAPI request object.

        Returns:
                JSONResponse with key details.
        """
        self._require_admin_role(raw_request)

        metadata = self.auth_manager.get_key_by_id(key_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"API key not found: {key_id}")

        return JSONResponse(metadata.as_dict(include_sensitive=False))

    async def update_api_key_endpoint(
        self,
        key_id: str,
        request: UpdateApiKeyRequest,
        raw_request: Request,
    ) -> JSONResponse:
        """Admin endpoint to update an API key.

        PATCH /v1/admin/keys/{key_id}
        Requires admin role.

        Args:
                key_id: ID of the key to update.
                request: Updated key parameters.
                raw_request: FastAPI request object.

        Returns:
                JSONResponse with updated key details.
        """
        self._require_admin_role(raw_request)
        existing = self.auth_manager.get_key_by_id(key_id)
        if not existing:
            raise HTTPException(status_code=404, detail=f"API key not found: {key_id}")

        provided_fields = set(request.model_fields_set)

        # Build configuration objects if any fields are set
        rate_limits = None
        rate_limit_fields = {
            "requests_per_minute",
            "requests_per_hour",
            "requests_per_day",
            "tokens_per_minute",
            "tokens_per_hour",
            "tokens_per_day",
        }
        if provided_fields.intersection(rate_limit_fields):
            rate_limits = RateLimitConfig(
                requests_per_minute=(
                    request.requests_per_minute
                    if "requests_per_minute" in provided_fields
                    else existing.rate_limits.requests_per_minute
                ),
                requests_per_hour=(
                    request.requests_per_hour
                    if "requests_per_hour" in provided_fields
                    else existing.rate_limits.requests_per_hour
                ),
                requests_per_day=(
                    request.requests_per_day
                    if "requests_per_day" in provided_fields
                    else existing.rate_limits.requests_per_day
                ),
                tokens_per_minute=(
                    request.tokens_per_minute
                    if "tokens_per_minute" in provided_fields
                    else existing.rate_limits.tokens_per_minute
                ),
                tokens_per_hour=(
                    request.tokens_per_hour
                    if "tokens_per_hour" in provided_fields
                    else existing.rate_limits.tokens_per_hour
                ),
                tokens_per_day=(
                    request.tokens_per_day
                    if "tokens_per_day" in provided_fields
                    else existing.rate_limits.tokens_per_day
                ),
            )

        quota = None
        quota_fields = {
            "max_total_tokens",
            "max_total_requests",
            "monthly_token_limit",
            "monthly_request_limit",
        }
        if provided_fields.intersection(quota_fields):
            quota = QuotaConfig(
                max_total_tokens=(
                    request.max_total_tokens
                    if "max_total_tokens" in provided_fields
                    else existing.quota.max_total_tokens
                ),
                max_total_requests=(
                    request.max_total_requests
                    if "max_total_requests" in provided_fields
                    else existing.quota.max_total_requests
                ),
                monthly_token_limit=(
                    request.monthly_token_limit
                    if "monthly_token_limit" in provided_fields
                    else existing.quota.monthly_token_limit
                ),
                monthly_request_limit=(
                    request.monthly_request_limit
                    if "monthly_request_limit" in provided_fields
                    else existing.quota.monthly_request_limit
                ),
            )

        permissions = None
        permission_fields = {
            "allowed_models",
            "allowed_endpoints",
            "allowed_ip_addresses",
            "blocked_ip_addresses",
            "enable_streaming",
            "enable_function_calling",
            "max_tokens_per_request",
        }
        if provided_fields.intersection(permission_fields):
            permissions = ApiKeyPermissions(
                allowed_models=(
                    request.allowed_models
                    if "allowed_models" in provided_fields
                    else existing.permissions.allowed_models
                ),
                allowed_endpoints=(
                    request.allowed_endpoints
                    if "allowed_endpoints" in provided_fields
                    else existing.permissions.allowed_endpoints
                ),
                allowed_ip_addresses=(
                    request.allowed_ip_addresses
                    if "allowed_ip_addresses" in provided_fields
                    else existing.permissions.allowed_ip_addresses
                ),
                blocked_ip_addresses=(
                    request.blocked_ip_addresses
                    if "blocked_ip_addresses" in provided_fields
                    else existing.permissions.blocked_ip_addresses
                ),
                enable_streaming=(
                    request.enable_streaming
                    if "enable_streaming" in provided_fields
                    else existing.permissions.enable_streaming
                ),
                enable_function_calling=(
                    request.enable_function_calling
                    if "enable_function_calling" in provided_fields
                    else existing.permissions.enable_function_calling
                ),
                max_tokens_per_request=(
                    request.max_tokens_per_request
                    if "max_tokens_per_request" in provided_fields
                    else existing.permissions.max_tokens_per_request
                ),
            )

        # Get the admin's key for audit trail
        admin_key = getattr(raw_request.state, "api_key", None)
        admin_metadata = self.auth_manager.validate_key(admin_key)
        updated_by = admin_metadata.name if admin_metadata else "unknown"

        success = self.auth_manager.update_key(
            key_id=key_id,
            name=request.name,
            description=request.description,
            role=request.role,
            expires_in_days=request.expires_in_days,
            rate_limits=rate_limits,
            quota=quota,
            permissions=permissions,
            tags=request.tags,
            metadata=request.metadata,
            updated_by=updated_by,
        )

        if not success:
            raise HTTPException(status_code=404, detail=f"API key not found: {key_id}")

        updated_metadata = self.auth_manager.get_key_by_id(key_id)
        return JSONResponse(
            {
                **updated_metadata.as_dict(include_sensitive=False),
                "message": "API key updated successfully",
            }
        )

    async def revoke_api_key_endpoint(self, key_id: str, raw_request: Request) -> JSONResponse:
        """Admin endpoint to revoke an API key.

        DELETE /v1/admin/keys/{key_id}/revoke
        Requires admin role.

        Args:
                key_id: ID of the key to revoke.
                raw_request: FastAPI request object.

        Returns:
                JSONResponse confirming revocation.
        """
        self._require_admin_role(raw_request)

        admin_key = getattr(raw_request.state, "api_key", None)
        admin_metadata = self.auth_manager.validate_key(admin_key)
        revoked_by = admin_metadata.name if admin_metadata else "unknown"

        success = self.auth_manager.revoke_key(key_id, revoked_by=revoked_by)
        if not success:
            raise HTTPException(status_code=404, detail=f"API key not found: {key_id}")

        return JSONResponse({"message": f"API key {key_id} revoked successfully", "key_id": key_id})

    async def suspend_api_key_endpoint(self, key_id: str, raw_request: Request) -> JSONResponse:
        """Admin endpoint to suspend an API key.

        POST /v1/admin/keys/{key_id}/suspend
        Requires admin role.

        Args:
                key_id: ID of the key to suspend.
                raw_request: FastAPI request object.

        Returns:
                JSONResponse confirming suspension.
        """
        self._require_admin_role(raw_request)

        admin_key = getattr(raw_request.state, "api_key", None)
        admin_metadata = self.auth_manager.validate_key(admin_key)
        suspended_by = admin_metadata.name if admin_metadata else "unknown"

        success = self.auth_manager.suspend_key(key_id, suspended_by=suspended_by)
        if not success:
            raise HTTPException(status_code=404, detail=f"API key not found: {key_id}")

        return JSONResponse({"message": f"API key {key_id} suspended successfully", "key_id": key_id})

    async def reactivate_api_key_endpoint(self, key_id: str, raw_request: Request) -> JSONResponse:
        """Admin endpoint to reactivate a suspended API key.

        POST /v1/admin/keys/{key_id}/reactivate
        Requires admin role.

        Args:
                key_id: ID of the key to reactivate.
                raw_request: FastAPI request object.

        Returns:
                JSONResponse confirming reactivation.
        """
        self._require_admin_role(raw_request)

        admin_key = getattr(raw_request.state, "api_key", None)
        admin_metadata = self.auth_manager.validate_key(admin_key)
        reactivated_by = admin_metadata.name if admin_metadata else "unknown"

        success = self.auth_manager.reactivate_key(key_id, reactivated_by=reactivated_by)
        if not success:
            raise HTTPException(status_code=404, detail=f"API key not found or cannot be reactivated: {key_id}")

        return JSONResponse({"message": f"API key {key_id} reactivated successfully", "key_id": key_id})

    async def delete_api_key_endpoint(self, key_id: str, raw_request: Request) -> JSONResponse:
        """Admin endpoint to permanently delete an API key.

        DELETE /v1/admin/keys/{key_id}
        Requires admin role.

        Args:
                key_id: ID of the key to delete.
                raw_request: FastAPI request object.

        Returns:
                JSONResponse confirming deletion.
        """
        self._require_admin_role(raw_request)

        admin_key = getattr(raw_request.state, "api_key", None)
        admin_metadata = self.auth_manager.validate_key(admin_key)
        deleted_by = admin_metadata.name if admin_metadata else "unknown"

        success = self.auth_manager.delete_key(key_id, deleted_by=deleted_by)
        if not success:
            raise HTTPException(status_code=404, detail=f"API key not found: {key_id}")

        return JSONResponse({"message": f"API key {key_id} deleted permanently", "key_id": key_id})

    async def rotate_api_key_endpoint(self, key_id: str, raw_request: Request) -> JSONResponse:
        """Admin endpoint to rotate an API key.

        POST /v1/admin/keys/{key_id}/rotate
        Requires admin role.

        Args:
                key_id: ID of the key to rotate.
                raw_request: FastAPI request object.

        Returns:
                JSONResponse with new raw key (only time it's shown).
        """
        self._require_admin_role(raw_request)

        admin_key = getattr(raw_request.state, "api_key", None)
        admin_metadata = self.auth_manager.validate_key(admin_key)
        rotated_by = admin_metadata.name if admin_metadata else "unknown"

        result = self.auth_manager.rotate_key(key_id, rotated_by=rotated_by)
        if not result:
            raise HTTPException(status_code=404, detail=f"API key not found: {key_id}")

        new_raw_key, metadata = result
        return JSONResponse(
            {
                "key": new_raw_key,  # New raw key
                "key_id": metadata.key_id,
                "key_prefix": metadata.key_prefix,
                "name": metadata.name,
                "last_rotated_at": metadata.last_rotated_at,
                "message": "API key rotated successfully. Store the new key securely - it won't be shown again!",
            }
        )

    async def get_api_key_stats_endpoint(self, raw_request: Request) -> JSONResponse:
        """Admin endpoint to get overall API key statistics.

        GET /v1/admin/keys/stats
        Requires admin role.

        Args:
                raw_request: FastAPI request object.

        Returns:
                JSONResponse with aggregate statistics.
        """
        self._require_admin_role(raw_request)

        stats = self.auth_manager.get_statistics()
        return JSONResponse(stats)

    async def get_audit_logs_endpoint(
        self,
        raw_request: Request,
        limit: int = 100,
        key_id: str | None = None,
        action: str | None = None,
    ) -> JSONResponse:
        """Admin endpoint to get audit logs.

        GET /v1/admin/audit-logs?limit=100&key_id=xxx&action=key_created
        Requires admin role.

        Args:
                raw_request: FastAPI request object.
                limit: Maximum number of entries to return.
                key_id: Optional key ID filter.
                action: Optional action type filter.

        Returns:
                JSONResponse with audit log entries.
        """
        self._require_admin_role(raw_request)

        logs = self.auth_manager.get_audit_logs(limit=limit, key_id=key_id, action=action)
        return JSONResponse({"logs": [log.as_dict() for log in logs], "total": len(logs)})
