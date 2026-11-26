# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""What a cell agent knows about its own identity, and what it admits it does not.

Thin on purpose. :mod:`eyvan.identity.metadata` decides *what the answer is*;
this decides *when the question is asked* on a worker host and hands back one
operator-safe sentence. Keeping the two apart is what lets the decision be
tested without a network and the plumbing be tested without a provider.

Resolved once per process rather than per request. A metadata answer changes
only when the host is replaced, and a probe on a request path is a network call
in front of every dispatch -- with a timeout that would show up as the agent
being slow rather than as an identity problem.

Nothing here reaches for a token. The agent learns which principal it runs as;
anything that needs to *act* as that principal uses the provider SDK, which
fetches and holds its own credential. That is why this module can be called from
a log line.
"""

from __future__ import annotations

import threading
import typing as tp

from eyvan.config import EyvanSettings
from eyvan.identity.metadata import (
    MetadataFetch,
    WorkloadIdentity,
    WorkloadIdentitySource,
    resolve_workload_identity,
)

_LOCK = threading.Lock()
_CACHED: WorkloadIdentity | None = None


def agent_workload_identity(
    settings: EyvanSettings,
    *,
    fetch: MetadataFetch | None = None,
    refresh: bool = False,
) -> WorkloadIdentity:
    """Return this agent host's provider-asserted identity.

    Args:
        settings: Process settings carrying the operator flag and timeout.
        fetch: Transport override, for tests.
        refresh: Re-probe instead of returning the cached answer.

    Returns:
        The identity, which is ``DISABLED`` when the flag is off, ``ABSENT`` only
        on a conclusive negative from the metadata service, and ``UNKNOWN``
        whenever the question could not be answered. ``UNKNOWN`` is never
        cached: an inability to observe is transient by definition, and caching
        it would turn one DNS hiccup at boot into a permanent verdict.
    """

    global _CACHED
    with _LOCK:
        if _CACHED is not None and not refresh:
            return _CACHED
        identity = resolve_workload_identity(
            enabled=settings.provider_metadata_identity_enabled,
            timeout_s=settings.provider_metadata_identity_timeout_s,
            fetch=fetch,
        )
        if identity.source is not WorkloadIdentitySource.UNKNOWN:
            _CACHED = identity
        return identity


def reset_agent_workload_identity() -> None:
    """Forget the cached answer. Exists for tests and for a deliberate re-probe."""

    global _CACHED
    with _LOCK:
        _CACHED = None


def describe_agent_workload_identity(identity: WorkloadIdentity) -> str:
    """Return one line safe to log, including for the cases with no identity.

    The absent and unknown cases are spelled out rather than dropped. An agent
    that silently logs nothing about its identity is indistinguishable from one
    that was never asked, and that ambiguity is what a rollout of metadata
    identity has to be able to see through.
    """

    return identity.describe()


__all__: tp.Final[tuple[str, ...]] = (
    "agent_workload_identity",
    "describe_agent_workload_identity",
    "reset_agent_workload_identity",
)
