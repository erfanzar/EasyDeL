# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Typed dependencies and value objects shared by Eyvan CLI commands."""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass
from pathlib import Path

from eyvan._internal.json_types import JsonObject, JsonValue
from eyvan.capacity.fleet_config import validate_fleet_name
from eyvan.sdk.client.identity import ClientIdentity
from eyvan.sdk.contexts import (
    DEFAULT_ORG_ID,
    DEFAULT_PROJECT_ID,
    EyvanContext,
    EyvanContextConfig,
    FileContextStore,
)


class ApiClient(tp.Protocol):
    """Minimal control-plane transport required by CLI commands."""

    def request(
        self,
        method: str,
        path: str,
        *,
        params: tp.Mapping[str, str | int | float | bool | None] | None = None,
        json: JsonObject | None = None,
        headers: tp.Mapping[str, str] | None = None,
    ) -> JsonValue:
        """Send a JSON request and return its decoded response."""

        ...

    def request_text(
        self,
        method: str,
        path: str,
        *,
        params: tp.Mapping[str, str | int | float | bool | None] | None = None,
        headers: tp.Mapping[str, str] | None = None,
    ) -> str:
        """Send a request and return its text response."""

        ...

    def replace_identity_headers(self, headers: tp.Mapping[str, str]) -> None:
        """Replace the authentication headers used for subsequent API requests."""

        ...

    def request_bytes(
        self,
        method: str,
        path: str,
        *,
        params: tp.Mapping[str, str | int | float | bool | None] | None = None,
        headers: tp.Mapping[str, str] | None = None,
    ) -> tuple[bytes, str]:
        """Send a request and return its raw body and media type."""

        ...

    def stream_lines(
        self,
        method: str,
        path: str,
        *,
        params: tp.Mapping[str, str | int | float | bool | None] | None = None,
        headers: tp.Mapping[str, str] | None = None,
        json: JsonObject | None = None,
    ) -> tp.Iterator[str]:
        """Send a request and yield response lines as they arrive (SSE or NDJSON)."""

        ...


ApiClientFactory: tp.TypeAlias = tp.Callable[[str, tp.Mapping[str, str]], ApiClient]
"""Build an API client for a control-plane URL and identity headers."""


@dataclass(frozen=True, slots=True)
class ContextOverrides:
    """Explicit CLI or environment values that outrank persisted context."""

    base_url: str | None = None
    org_id: str | None = None
    project_id: str | None = None


@dataclass(frozen=True, slots=True)
class WorkloadScope:
    """Logical authorization scope for one workload submission."""

    org_id: str
    project_id: str


@dataclass(frozen=True, slots=True)
class RunTarget:
    """Resolved fleet and pool selected by a shorthand or explicit target."""

    fleet: str
    pool: str

    @classmethod
    def parse(cls, value: str, *, default_fleet: str | None = None) -> RunTarget:
        """Parse ``TYPE`` or ``FLEET/TYPE`` into a validated target."""

        parts = value.split("/")
        if len(parts) == 1 and parts[0] and default_fleet is not None:
            fleet, pool = default_fleet, parts[0]
        elif len(parts) == 2 and all(parts):
            fleet, pool = parts
        else:
            raise ValueError("run target must be TYPE or FLEET/TYPE")
        try:
            validate_fleet_name(fleet)
            validate_fleet_name(pool)
        except ValueError as exc:
            raise ValueError(f"invalid run target: {exc}") from exc
        return cls(fleet=fleet, pool=pool)


@dataclass(slots=True)
class CliSession:
    """All typed state and dependencies available to one CLI invocation."""

    client: ApiClient
    identity: ClientIdentity
    context_store: FileContextStore
    context_config: EyvanContextConfig
    base_url: str
    org_id: str | None
    project_id: str | None
    overrides: ContextOverrides
    client_factory: ApiClientFactory | None = None

    @property
    def active_context(self) -> EyvanContext | None:
        """Return the active persisted context for this invocation."""

        return self.context_config.active

    @property
    def autoscale_cache_dir(self) -> Path | None:
        """Return the active fleet's selected autoscale cache root, if any."""

        active = self.active_context
        if active is None or active.autoscale_cache_dir is None:
            return None
        return Path(active.autoscale_cache_dir)

    def workload_scope(self, *, org_id: str | None = None, project_id: str | None = None) -> WorkloadScope:
        """Resolve explicit, active, or conventional default workload scope."""

        return WorkloadScope(
            org_id=org_id or self.org_id or DEFAULT_ORG_ID,
            project_id=project_id or self.project_id or DEFAULT_PROJECT_ID,
        )

    def workload_headers(self, scope: WorkloadScope) -> dict[str, str]:
        """Return request-local development headers for an explicitly selected scope."""

        if self.identity.token is not None or self.identity.dev_user is None:
            return {}
        return ClientIdentity(
            dev_user=self.identity.dev_user,
            dev_org=scope.org_id,
            dev_project=scope.project_id,
            dev_roles=self.identity.dev_roles,
            local_development_credential=self.identity.local_development_credential,
        ).headers()

    def replace_identity(self, identity: ClientIdentity) -> None:
        """Apply a refreshed identity to this session and its concrete API client."""

        self.client.replace_identity_headers(identity.headers())
        self.identity = identity

    def refresh_loopback_identity(self) -> None:
        """Re-resolve development material after a local plane may have minted it.

        A session resolves its identity before the command body runs, and the
        local plane may not exist yet: on a machine with no development
        credential, starting the plane mints one and the plane then requires
        it, so the pre-start identity would be refused by the plane the same
        command just started -- first run, first command, 401. Re-resolving
        after the plane exists picks up the credential that plane demands.
        No-op for token identities, remote endpoints, and identities that
        already carry the credential.
        """

        if self.identity.token is not None:
            return
        if self.identity.local_development_credential is not None:
            return
        from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
        from eyvan.sdk.client.identity import resolve_control_plane_identity

        endpoint = ControlPlaneEndpoint.parse(self.base_url)
        if not endpoint.is_loopback:
            return
        resolved = resolve_control_plane_identity(
            endpoint,
            dev_user=self.identity.dev_user,
            dev_org=self.identity.dev_org,
            dev_project=self.identity.dev_project,
            dev_roles=self.identity.dev_roles,
        )
        if resolved.local_development_credential is None:
            # Still nothing to present: leave the session exactly as it was
            # rather than swapping an equal identity underneath the client.
            return
        self.replace_identity(resolved)

    def retarget_loopback_endpoint(self, base_url: str) -> bool:
        """Point this session at another literal loopback plane, if that is safe.

        Two development planes on one machine differ only by port, and a fleet
        saved against one of them should not be refused merely because another
        local fleet is active. Only loopback-to-loopback moves are made, never
        over an explicit ``--url`` or with a bearer token: different local
        listeners need not share credentials. Anything else leaves the session
        unchanged and returns ``False`` so the caller keeps its stricter behavior.
        """

        from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
        from eyvan.sdk.client.identity import resolve_control_plane_identity

        if self.client_factory is None or self.overrides.base_url is not None or self.identity.token is not None:
            return False
        target = ControlPlaneEndpoint.parse(base_url)
        current = ControlPlaneEndpoint.parse(self.base_url)
        if not (target.is_loopback and current.is_loopback):
            return False
        identity = resolve_control_plane_identity(
            target,
            token=self.identity.token,
            dev_user=self.identity.dev_user,
            dev_org=self.identity.dev_org,
            dev_project=self.identity.dev_project,
            dev_roles=self.identity.dev_roles,
        )
        self.client = self.client_factory(target.url, identity.headers())
        self.identity = identity
        self.base_url = target.url
        return True

    def activate_fleet(
        self,
        fleet: str,
        *,
        org_id: str | None = None,
        project_id: str | None = None,
        autoscale_cache_dir: Path | None = None,
    ) -> EyvanContext:
        """Persist ``fleet`` as active without borrowing another fleet's scope."""

        existing = self.context_config.contexts.get(fleet)
        context = EyvanContext(
            fleet=fleet,
            org_id=(
                org_id or self.overrides.org_id or (existing.org_id if existing is not None else None) or DEFAULT_ORG_ID
            ),
            project_id=(
                project_id
                or self.overrides.project_id
                or (existing.project_id if existing is not None else None)
                or DEFAULT_PROJECT_ID
            ),
            # Mirrors ``select_context`` rather than taking the session URL
            # unconditionally. The two sat in this file with opposite precedence,
            # and the consequence ran the wrong way: `eyvan --url <private-ip> up`
            # -- the exact command `head up` prints -- overwrote a fleet's saved
            # reachable endpoint with an address only the VPC can route, and that
            # became the default for every later command on the machine.
            base_url=self.overrides.base_url or (existing.base_url if existing is not None else self.base_url),
            autoscale_cache_dir=(
                str(autoscale_cache_dir)
                if autoscale_cache_dir is not None
                else (existing.autoscale_cache_dir if existing is not None else None)
            ),
        )
        self.context_config = self.context_store.activate(context)
        self.org_id = context.org_id
        self.project_id = context.project_id
        return context

    def save_plane_context(
        self,
        base_url: str,
        *,
        org_id: str | None = None,
        project_id: str | None = None,
    ) -> tuple[str, EyvanContext]:
        """Make a plane the active context before any fleet is chosen.

        ``eyvan login`` on a plane with no fleet, or several, still has to make
        every later command target that plane; the fleet is then inferred or
        named per command (or chosen once with ``eyvan use FLEET``).
        """

        from eyvan.sdk.contexts import plane_context_name

        name = plane_context_name(base_url)
        existing = self.context_config.contexts.get(name)
        context = EyvanContext(
            fleet=None,
            org_id=org_id
            or self.overrides.org_id
            or (existing.org_id if existing is not None else None)
            or DEFAULT_ORG_ID,
            project_id=project_id
            or self.overrides.project_id
            or (existing.project_id if existing is not None else None)
            or DEFAULT_PROJECT_ID,
            base_url=base_url,
        )
        self.context_config = self.context_store.activate(context, name=name)
        self.base_url = context.base_url
        self.org_id = context.org_id
        self.project_id = context.project_id
        return name, context

    def select_context(
        self,
        fleet: str,
        *,
        org_id: str | None = None,
        project_id: str | None = None,
    ) -> EyvanContext:
        """Select a named fleet, preserving its saved endpoint and scope defaults."""

        existing = self.context_config.contexts.get(fleet)
        selected_org = (
            org_id
            or self.overrides.org_id
            or (existing.org_id if existing is not None else self.org_id)
            or DEFAULT_ORG_ID
        )
        selected_project = (
            project_id or self.overrides.project_id or (existing.project_id if existing is not None else self.project_id)
        ) or DEFAULT_PROJECT_ID
        context = EyvanContext(
            fleet=fleet,
            org_id=selected_org,
            project_id=selected_project,
            base_url=self.overrides.base_url or (existing.base_url if existing is not None else self.base_url),
            autoscale_cache_dir=existing.autoscale_cache_dir if existing is not None else None,
        )
        self.context_config = self.context_store.activate(context)
        self.base_url = context.base_url
        self.org_id = context.org_id
        self.project_id = context.project_id
        return context
