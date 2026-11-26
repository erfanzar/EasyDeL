# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Scoped transfer URLs for control-plane-managed callable artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote, urlencode, urlsplit, urlunsplit

from eyvan.artifacts.managed import (
    ArtifactAction,
    ArtifactTokenSigner,
    ManagedArtifactKind,
    ManagedArtifactStore,
    parse_managed_uri,
)


@dataclass(frozen=True, slots=True)
class ManagedArtifactTransfer:
    """A token-free transfer URL and its separately transported credential."""

    url: str
    token: str


@dataclass(frozen=True, slots=True)
class ManagedArtifactAccess:
    """Bind managed storage, transfer authorization, and the public API route."""

    store: ManagedArtifactStore
    signer: ArtifactTokenSigner
    public_url: str
    token_ttl_seconds: int

    def __post_init__(self) -> None:
        public_url = self.public_url.strip().rstrip("/")
        parsed = urlsplit(public_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("managed artifact public URL must be an absolute HTTP or HTTPS URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("managed artifact public URL cannot contain credentials, a query, or a fragment")
        try:
            _ = parsed.port
        except ValueError as exc:
            raise ValueError("managed artifact public URL has an invalid port") from exc
        if self.token_ttl_seconds <= 0:
            raise ValueError("managed artifact token TTL must be positive")
        object.__setattr__(self, "public_url", urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", "")))

    def _transfer(
        self,
        action: ArtifactAction,
        uri: str,
        *,
        org_id: str,
        project_id: str,
        workload_id: str,
        ttl_seconds: int | None = None,
        expires: bool = True,
    ) -> ManagedArtifactTransfer:
        reference = parse_managed_uri(uri)
        token = self.signer.issue(
            action,
            reference.uri,
            org_id=org_id,
            project_id=project_id,
            workload_id=workload_id,
            ttl_seconds=(self.token_ttl_seconds if ttl_seconds is None else ttl_seconds) if expires else None,
            expires=expires,
        )
        query = urlencode(
            {
                "org_id": org_id,
                "project_id": project_id,
                "workload_id": workload_id,
            }
        )
        url = f"{self.public_url}/api/v1/artifacts/{reference.kind.value}/{reference.object_id}/content?{query}"
        return ManagedArtifactTransfer(url=url, token=token)

    def callable_download_transfer(
        self,
        uri: str,
        *,
        org_id: str,
        project_id: str,
        workload_id: str,
        ttl_seconds: int | None = None,
    ) -> ManagedArtifactTransfer:
        """Issue a scoped callable download transfer for one workload."""

        reference = parse_managed_uri(uri)
        if reference.kind is not ManagedArtifactKind.CALLABLES:
            raise ValueError("callable download requires an eyvan://callables URI")
        return self._transfer(
            ArtifactAction.READ,
            reference.uri,
            org_id=org_id,
            project_id=project_id,
            workload_id=workload_id,
            ttl_seconds=ttl_seconds,
        )

    def workspace_download_transfer(
        self,
        uri: str,
        *,
        org_id: str,
        project_id: str,
        workload_id: str,
        ttl_seconds: int | None = None,
    ) -> ManagedArtifactTransfer:
        """Issue a scoped workspace-bundle download transfer for one workload.

        A bundle is fetched like a callable, so it uses the expiring READ
        capability rather than the durable generation-revoked WRITE one.
        """

        reference = parse_managed_uri(uri)
        if reference.kind is not ManagedArtifactKind.WORKSPACES:
            raise ValueError("workspace download requires an eyvan://workspaces URI")
        return self._transfer(
            ArtifactAction.READ,
            reference.uri,
            org_id=org_id,
            project_id=project_id,
            workload_id=workload_id,
            ttl_seconds=ttl_seconds,
        )

    def result_upload_transfer(
        self,
        uri: str,
        *,
        org_id: str,
        project_id: str,
        workload_id: str,
    ) -> ManagedArtifactTransfer:
        """Issue a generation-revoked result upload transfer for one workload."""

        reference = parse_managed_uri(uri)
        if reference.kind is not ManagedArtifactKind.RESULTS:
            raise ValueError("result upload requires an eyvan://results URI")
        return self._transfer(
            ArtifactAction.WRITE,
            reference.uri,
            org_id=org_id,
            project_id=project_id,
            workload_id=workload_id,
            expires=False,
        )

    def progress_report_url(self, *, workload_id: str, org_id: str, project_id: str) -> str:
        """Return the address a running attempt posts its own progress to.

        No token of its own: the progress route authenticates with the attempt's
        result-write capability, which is already scoped to this workload,
        already revoked by a generation bump, and already delivered to every
        rank. This method exists so the URL is built where every other
        control-plane address is built, rather than assembled inside a job from
        a base URL it would otherwise have to be told separately.
        """

        query = urlencode({"org_id": org_id, "project_id": project_id})
        return f"{self.public_url}/api/v1/workloads/{quote(workload_id, safe='')}/progress-reports?{query}"


__all__ = ("ManagedArtifactAccess", "ManagedArtifactTransfer")
