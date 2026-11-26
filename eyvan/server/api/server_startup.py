# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Typed startup policy for the Eyvan API server."""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass, replace
from urllib.parse import urlsplit

from eyvan._internal.compat import StrEnum
from eyvan._internal.platform import load_platform_config, refuse_removed_team_variable, resolve_team_mode
from eyvan.config import EyvanSettings, RepositoryBackend
from eyvan.server.api.server_bind import ServerBindPolicy

_DEV_AUTH_ENVIRONMENT_VARIABLE = "EYVAN_DEV_AUTH_ENABLED"
_REPOSITORY_BACKEND_ENVIRONMENT_VARIABLE = "EYVAN_REPOSITORY_BACKEND"
_METRICS_TOKEN_ENVIRONMENT_VARIABLE = "EYVAN_METRICS_TOKEN"
_ARTIFACT_TOKEN_KEY_ENVIRONMENT_VARIABLE = "EYVAN_ARTIFACT_TOKEN_KEY"

NETWORK_REQUIRES_POSTGRES = (
    "a network-bound Eyvan server requires EYVAN_REPOSITORY_BACKEND=postgres so every replica "
    "shares durable control-plane state"
)
NETWORK_REQUIRES_TLS = (
    "a network-bound Eyvan server refuses to serve plaintext HTTP: OIDC bearer tokens, bootstrap "
    "credentials, and capability tokens would cross the network unencrypted. Set EYVAN_TLS_CERT_FILE "
    "and EYVAN_TLS_KEY_FILE to serve TLS, or set EYVAN_INSECURE_PLAINTEXT_BIND=1 (or pass "
    "--insecure-plaintext to eyvan-server) to keep the pre-0.1 plaintext behavior as an explicit, "
    "reviewed decision"
)
TLS_REQUIRES_BOTH_FILES = (
    "TLS needs both halves of the listener identity: set EYVAN_TLS_CERT_FILE and "
    "EYVAN_TLS_KEY_FILE together; a certificate without its key cannot terminate anything"
)
LOCAL_REPOSITORY_IS_LOOPBACK_ONLY = "the local Eyvan repository may only be used with a literal loopback server bind"
DEVELOPMENT_AUTH_IS_LOOPBACK_ONLY = (
    "Eyvan development authentication may only be used with a literal loopback server bind; "
    "disable it or configure OIDC for a network-bound deployment"
)
# ``/metrics`` exports eyvan_project_spend_usd labeled by (org, project).
# Unauthenticated on a network bind, that is every tenant's spend readable by
# anyone who can reach the port -- and the shipped manifests have no
# NetworkPolicy in front of it. Refusing at startup is the only version of this
# that cannot be forgotten; a warning in a log is a gap with documentation
# attached.
NETWORK_REQUIRES_METRICS_TOKEN = (
    "a network-bound Eyvan server requires EYVAN_METRICS_TOKEN so /metrics does not expose "
    "per-tenant spend to anyone who can reach the port"
)
NON_LOCAL_REPOSITORY_REQUIRES_ARTIFACT_TOKEN_KEY = (
    "EYVAN_ARTIFACT_TOKEN_KEY must be set when the Eyvan repository backend is not local; an ephemeral "
    "per-process signing key would break managed-artifact token validation across replicas and restarts"
)

TEAM_MODE_REQUIRES_PUBLIC_URL = (
    "team mode on a network bind needs EYVAN_PUBLIC_URL set to the address people actually open "
    "(https://..., not a loopback address): invite links are built from it, and every dashboard cookie "
    "write is refused unless its Origin matches it"
)

BOOTSTRAP_TOKEN_AND_ISSUER_CONFLICT = (
    "EYVAN_BOOTSTRAP_TOKEN and EYVAN_OIDC_ISSUER are both set; a bootstrap token exists only to reach a head that "
    "has no identity provider, and silently preferring one of two configured identities is how a temporary "
    "credential outlives the gap it was created for"
)


class DevelopmentAuthMode(StrEnum):
    """How development-header authentication was selected for this process."""

    LOOPBACK_DEFAULT = "loopback_default"
    EXPLICIT = "explicit"
    DISABLED = "disabled"


class ServerControllerMode(StrEnum):
    """Whether this API process actively reconciles workloads and capacity."""

    ACTIVE = "active"
    API_ONLY = "api_only"


class StartupConfigurationSource(StrEnum):
    """Where the settings under evaluation came from.

    A process configured from ``EYVAN_*`` must run on Postgres to bind a network
    address, because nothing else keeps replicas consistent. Explicitly injected
    settings are a different situation -- an embedding harness passes its own
    repository object -- so only the local repository, which has no cross-replica
    story at all, is refused there. Keeping the distinction here rather than in
    two call sites is what lets one function answer for both.
    """

    ENVIRONMENT = "environment"
    EXPLICIT = "explicit"


@dataclass(frozen=True, slots=True)
class StartupRequirement:
    """One setting a bind mode requires, with the reason it cannot be defaulted."""

    environment_variable: str
    explanation: str

    def describe(self) -> str:
        """Render this requirement as ``NAME: why it exists``."""

        return f"{self.environment_variable}: {self.explanation}"


def unsatisfied_startup_requirements(
    bind: ServerBindPolicy,
    settings: EyvanSettings,
    *,
    source: StartupConfigurationSource,
) -> tuple[StartupRequirement, ...]:
    """Return every requirement this bind does not satisfy, in refusal order.

    The one source of truth for "what will the server refuse to start without".
    It exists because these refusals were discovered serially: each is correct,
    each names its variable, and each was reached only after the previous one had
    been fixed and the process restarted. Collecting them means the operator gets
    the whole list on the first attempt, and ``eyvan doctor`` can ask the same
    question without starting anything.

    Order matches the order the individual refusals used to fire in, so the first
    entry is still the one a reader of an old traceback would have seen.
    """

    if bind.is_loopback:
        return ()
    requirements: list[StartupRequirement] = []
    if not settings.allow_insecure_plaintext_bind:
        if settings.tls_cert_file is None and settings.tls_key_file is None:
            requirements.append(StartupRequirement("EYVAN_TLS_CERT_FILE", NETWORK_REQUIRES_TLS))
        elif settings.tls_cert_file is None or settings.tls_key_file is None:
            requirements.append(StartupRequirement("EYVAN_TLS_CERT_FILE", TLS_REQUIRES_BOTH_FILES))
    if source is StartupConfigurationSource.ENVIRONMENT:
        if settings.repository_backend is not RepositoryBackend.POSTGRES:
            requirements.append(StartupRequirement(_REPOSITORY_BACKEND_ENVIRONMENT_VARIABLE, NETWORK_REQUIRES_POSTGRES))
    elif settings.repository_backend is RepositoryBackend.LOCAL:
        requirements.append(
            StartupRequirement(_REPOSITORY_BACKEND_ENVIRONMENT_VARIABLE, LOCAL_REPOSITORY_IS_LOOPBACK_ONLY)
        )
    if settings.bootstrap_token and settings.oidc_issuer:
        requirements.append(
            StartupRequirement(
                "EYVAN_BOOTSTRAP_TOKEN",
                BOOTSTRAP_TOKEN_AND_ISSUER_CONFLICT,
            )
        )
    if settings.team_mode and not _team_public_url_is_usable(settings):
        requirements.append(StartupRequirement("EYVAN_PUBLIC_URL", TEAM_MODE_REQUIRES_PUBLIC_URL))
    if settings.dev_auth_enabled:
        requirements.append(StartupRequirement(_DEV_AUTH_ENVIRONMENT_VARIABLE, DEVELOPMENT_AUTH_IS_LOOPBACK_ONLY))
    if not settings.metrics_token:
        requirements.append(StartupRequirement(_METRICS_TOKEN_ENVIRONMENT_VARIABLE, NETWORK_REQUIRES_METRICS_TOKEN))
    if settings.artifact_token_key is None and _signing_key_must_be_explicit(settings, source):
        requirements.append(
            StartupRequirement(
                _ARTIFACT_TOKEN_KEY_ENVIRONMENT_VARIABLE,
                NON_LOCAL_REPOSITORY_REQUIRES_ARTIFACT_TOKEN_KEY,
            )
        )
    return tuple(requirements)


def _team_public_url_is_usable(settings: EyvanSettings) -> bool:
    """Return whether a network-bound team plane's public URL can carry invites and cookie writes.

    It must name a non-loopback host, and use https unless plaintext was
    explicitly accepted: a password typed into an http page crosses the network
    in the clear even when the API listener itself has TLS in front of it.
    """

    split = urlsplit(settings.public_url.strip())
    if split.scheme not in {"http", "https"} or not split.hostname:
        return False
    if split.scheme == "http" and not settings.allow_insecure_plaintext_bind:
        return False
    try:
        return not ipaddress.ip_address(split.hostname).is_loopback
    except ValueError:
        return split.hostname.lower() != "localhost"


def _signing_key_must_be_explicit(settings: EyvanSettings, source: StartupConfigurationSource) -> bool:
    """Return whether ``eyvan.server.api.app`` will refuse an implicit artifact signing key.

    Evaluated against the backend the process will actually run on. From the
    environment a network bind must be Postgres, so the key is required no matter
    what ``EYVAN_REPOSITORY_BACKEND`` currently says -- reporting it only once the
    backend had already been corrected is exactly the serialization this exists to
    remove. Injected settings are taken at face value, because a memory-backed
    harness legitimately mints an ephemeral key and only gets a warning.
    """

    backend = (
        RepositoryBackend.POSTGRES if source is StartupConfigurationSource.ENVIRONMENT else settings.repository_backend
    )
    return backend not in {RepositoryBackend.LOCAL, RepositoryBackend.MEMORY}


def format_requirement_bullets(requirements: tuple[StartupRequirement, ...]) -> str:
    """Render requirements one per line as ``  - NAME: why it exists``.

    Shared with ``eyvan doctor`` so the two surfaces emit the same bytes for the
    same finding, and so a test can parse either one the same way.
    """

    return "\n".join(f"  - {requirement.describe()}" for requirement in requirements)


def describe_unsatisfied_requirements(requirements: tuple[StartupRequirement, ...]) -> str:
    """Render one refusal naming every unsatisfied requirement and why each exists."""

    count = len(requirements)
    noun = "requirement" if count == 1 else "requirements"
    return (
        f"a network-bound Eyvan server has {count} unsatisfied {noun}:\n"
        f"{format_requirement_bullets(requirements)}\n"
        "ask for this list before starting a server with: eyvan doctor --server-host <bind address>"
    )


def server_environment_settings(*, team: bool = False, insecure_plaintext: bool = False) -> EyvanSettings:
    """Resolve what an ``eyvan-server`` process started here would run with, before bind defaults.

    ``EYVAN_*`` settings, then the platform file named by
    ``EYVAN_PLATFORM_CONFIG_PATH`` (its ``team`` key), then the command-line
    flags. The server and ``eyvan doctor`` both call this, so the preflight
    answers for the same team mode the server will enforce, including a team
    mode that only the platform file turned on. Nothing is written to
    ``os.environ``: ``--insecure-plaintext`` is a settings replacement.

    Raises:
        ValueError: An ``EYVAN_*`` value or the platform file does not parse.
    """

    refuse_removed_team_variable()
    settings = EyvanSettings.from_env()
    if insecure_plaintext:
        settings = replace(settings, allow_insecure_plaintext_bind=True)
    return resolve_team_mode(settings, flag=team, platform=load_platform_config(settings.platform_config_path))


@dataclass(frozen=True, slots=True)
class ServerStartupSettings:
    """Security-relevant server settings resolved for one bind address."""

    bind: ServerBindPolicy
    settings: EyvanSettings
    development_auth_mode: DevelopmentAuthMode

    @classmethod
    def from_environment(
        cls,
        bind: ServerBindPolicy,
        *,
        team: bool = False,
        insecure_plaintext: bool = False,
    ) -> ServerStartupSettings:
        """Apply zero-configuration local defaults only to an unconfigured literal loopback bind.

        ``team`` and ``insecure_plaintext`` are the ``eyvan-server`` flags of the
        same names; see :func:`server_environment_settings`.
        """

        settings = server_environment_settings(team=team, insecure_plaintext=insecure_plaintext)
        if bind.is_loopback and _REPOSITORY_BACKEND_ENVIRONMENT_VARIABLE not in os.environ:
            settings = replace(settings, repository_backend=RepositoryBackend.LOCAL)
        if bind.is_loopback and _DEV_AUTH_ENVIRONMENT_VARIABLE not in os.environ:
            return cls(
                bind=bind,
                settings=replace(settings, dev_auth_enabled=True),
                development_auth_mode=DevelopmentAuthMode.LOOPBACK_DEFAULT,
            )
        return cls._resolve(bind, settings, source=StartupConfigurationSource.ENVIRONMENT)

    @classmethod
    def from_settings(cls, bind: ServerBindPolicy, settings: EyvanSettings) -> ServerStartupSettings:
        """Describe explicit settings without changing their authentication policy."""

        return cls._resolve(bind, settings, source=StartupConfigurationSource.EXPLICIT)

    @classmethod
    def _resolve(
        cls,
        bind: ServerBindPolicy,
        settings: EyvanSettings,
        *,
        source: StartupConfigurationSource,
    ) -> ServerStartupSettings:
        """Refuse once for every unsatisfied requirement, or accept the settings as given.

        One raise rather than the first of several. Every explanation is still
        here in full: standing a head up meant reading a traceback, fixing the one
        variable it named, restarting, and finding the next -- three times, on a
        machine that was already provisioned and billing.
        """

        unsatisfied = unsatisfied_startup_requirements(bind, settings, source=source)
        if unsatisfied:
            raise ValueError(describe_unsatisfied_requirements(unsatisfied))
        mode = DevelopmentAuthMode.EXPLICIT if settings.dev_auth_enabled else DevelopmentAuthMode.DISABLED
        return cls(bind=bind, settings=settings, development_auth_mode=mode)

    @property
    def controller_mode(self) -> ServerControllerMode:
        """Return the controller mode exposed by the health endpoint."""

        return ServerControllerMode.ACTIVE if self.settings.controller_enabled else ServerControllerMode.API_ONLY


__all__ = (
    "DEVELOPMENT_AUTH_IS_LOOPBACK_ONLY",
    "LOCAL_REPOSITORY_IS_LOOPBACK_ONLY",
    "NETWORK_REQUIRES_METRICS_TOKEN",
    "NETWORK_REQUIRES_POSTGRES",
    "NON_LOCAL_REPOSITORY_REQUIRES_ARTIFACT_TOKEN_KEY",
    "TEAM_MODE_REQUIRES_PUBLIC_URL",
    "DevelopmentAuthMode",
    "ServerControllerMode",
    "ServerStartupSettings",
    "StartupConfigurationSource",
    "StartupRequirement",
    "describe_unsatisfied_requirements",
    "format_requirement_bullets",
    "server_environment_settings",
    "unsatisfied_startup_requirements",
)
