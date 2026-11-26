# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""SDK identity headers and explicit identity resolution."""

from __future__ import annotations

import getpass
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
from eyvan.server.local.auth import (
    DEFAULT_LOCAL_DEVELOPMENT_CREDENTIAL_PATH,
    LOCAL_DEVELOPMENT_CREDENTIAL_HEADER,
    LocalDevelopmentCredentialStore,
)


@dataclass(frozen=True, slots=True)
class ClientIdentity:
    """Authentication material used by Eyvan API clients."""

    token: str | None = field(default=None, repr=False)
    dev_user: str | None = None
    dev_org: str | None = None
    dev_project: str | None = None
    dev_roles: tuple[str, ...] = ("operator",)
    local_development_credential: str | None = field(default=None, repr=False)

    def headers(self) -> dict[str, str]:
        """Return the wire headers for this identity.

        A bearer token deliberately takes precedence over development headers.
        """

        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        if not self.dev_user:
            return {}
        return {
            "X-Eyvan-User": self.dev_user,
            "X-Eyvan-Roles": ",".join(self.dev_roles),
            **({"X-Eyvan-Org": self.dev_org} if self.dev_org else {}),
            **({"X-Eyvan-Project": self.dev_project} if self.dev_project else {}),
            **(
                {LOCAL_DEVELOPMENT_CREDENTIAL_HEADER: self.local_development_credential}
                if self.local_development_credential
                else {}
            ),
        }


def local_development_subject(user_reader: tp.Callable[[], str] | None = None) -> str:
    """Return a stable non-empty subject for loopback development auth."""

    reader = user_reader or getpass.getuser
    try:
        subject = reader().strip()
    except (KeyError, OSError):
        subject = ""
    return subject or "local-user"


def resolve_identity(
    *,
    token: str | None = None,
    dev_user: str | None = None,
    dev_org: str | None = None,
    dev_project: str | None = None,
    dev_roles: str | tp.Iterable[str] | None = None,
    local_development_credential: str | None = None,
) -> ClientIdentity:
    """Build a client identity from explicit, already-resolved configuration."""

    if token is not None and not token.strip():
        raise ValueError("token must be a non-empty string")
    if dev_roles is None:
        roles = ("operator",)
    elif isinstance(dev_roles, str):
        roles = tuple(role.strip() for role in dev_roles.split(",") if role.strip())
    else:
        roles = tuple(role.strip() for role in dev_roles if role.strip())
    if dev_user is not None and not roles:
        raise ValueError("dev_roles must name at least one role")
    return ClientIdentity(
        token=token,
        dev_user=dev_user,
        dev_org=dev_org,
        dev_project=dev_project,
        dev_roles=roles,
        local_development_credential=local_development_credential,
    )


def resolve_control_plane_identity(
    endpoint: ControlPlaneEndpoint,
    *,
    token: str | None = None,
    dev_user: str | None = None,
    dev_org: str | None = None,
    dev_project: str | None = None,
    dev_roles: str | tp.Iterable[str] | None = None,
    local_development_credential_path: Path | str = DEFAULT_LOCAL_DEVELOPMENT_CREDENTIAL_PATH,
) -> ClientIdentity:
    """Resolve explicit auth and synthesize a subject only for loopback development."""

    if token is None and dev_user is None and endpoint.is_loopback:
        dev_user = local_development_subject()
    local_development_credential = None
    if token is None and endpoint.is_loopback:
        credential = LocalDevelopmentCredentialStore(local_development_credential_path).load_if_present()
        if credential is not None:
            local_development_credential = credential.value
    return resolve_identity(
        token=token,
        dev_user=dev_user,
        dev_org=dev_org,
        dev_project=dev_project,
        dev_roles=dev_roles,
        local_development_credential=local_development_credential,
    )


__all__ = (
    "ClientIdentity",
    "local_development_subject",
    "resolve_control_plane_identity",
    "resolve_identity",
)
