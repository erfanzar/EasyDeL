# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Workload identity taken from the provider metadata path, or honestly absent.

A long-lived service-account key is a file that grants everything the workload
can do, forever, to anyone who reads it -- and it has to be distributed, stored,
and rotated, which is three more places it can leak. The provider already knows
which principal a host is running as and will say so over its metadata path,
with no material to distribute at all. This resolves that answer.

**Identity, never credentials.** Nothing here asks the metadata service for an
access token, and nothing here returns one. Eyvan needs to know *who* a host is,
so it reads the principal; a process that needs to *act* as that principal uses
the provider SDK, which reads the metadata service itself and keeps the token
inside the library that fetched it. That split is why this module can be called
from anywhere, including from a diagnostic, without a leak being possible.

**Behind a flag, defaulting to today's behaviour.** With the flag off the answer
is :attr:`WorkloadIdentitySource.DISABLED` -- not "absent", because a disabled
probe has learned nothing about the host.

**Absent is a finding; unreachable is not.** ``ABSENT`` is returned only when the
metadata service itself answered, identified itself with the provider's flavour
header, and reported no identity. A DNS failure, a refused connection, a
timeout, a proxy, or a 5xx all produce ``UNKNOWN``: none of them distinguishes
"this host has no workload identity" from "a firewall is in the way", and
recording the first when the second is true is how a fleet ends up believing a
correctly-configured host is unidentifiable.

The flavour header is checked on success too. Without it, anything that can
answer on that address -- an SSRF proxy, a stray local service, a captive portal
-- can assert a principal, and the whole point of metadata identity is that the
platform is the one asserting it.
"""

from __future__ import annotations

import http.client
import re
import typing as tp
import urllib.error
import urllib.request
from dataclasses import dataclass

from eyvan._internal.compat import StrEnum

#: Google Compute/TPU metadata root. The path is a constant rather than a
#: setting: an operator-supplied URL is an SSRF sink, and there is no legitimate
#: reason for one host to read another's identity.
GCP_METADATA_IDENTITY_URL: tp.Final = (
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email"
)
#: Required on both the request and the response. On the request it defeats
#: browser-style cross-origin reads; on the response it proves the answer came
#: from the metadata service rather than from whatever else replied.
METADATA_FLAVOR_HEADER: tp.Final = "Metadata-Flavor"
METADATA_FLAVOR_GOOGLE: tp.Final = "Google"

_MAX_IDENTITY_BYTES: tp.Final = 4096
_PRINCIPAL = re.compile(r"[A-Za-z0-9._%+-]{1,128}@[A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]{1,63}){1,8}\Z")


class WorkloadIdentitySource(StrEnum):
    """Where a host's asserted identity came from, including "nowhere"."""

    #: The probe was not run. Says nothing about the host.
    DISABLED = "disabled"
    #: The provider metadata service asserted this principal.
    METADATA = "metadata"
    #: The metadata service answered and reported no workload identity.
    ABSENT = "absent"
    #: The metadata service could not be consulted, or its answer was not
    #: attributable to it. Never collapsed into ``ABSENT``.
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class WorkloadIdentity:
    """One host's asserted identity, or an explicit statement that there is none.

    ``detail`` is an operator-safe sentence built from fixed text and, at most, a
    status code. Response bodies never reach it: if a deployment ever pointed
    this at a token path, the body would be a credential.
    """

    source: WorkloadIdentitySource
    principal: str = ""
    detail: str = ""

    @property
    def is_proven(self) -> bool:
        """Return whether the provider itself asserted this identity."""

        return self.source is WorkloadIdentitySource.METADATA

    def describe(self) -> str:
        """Return one operator-safe line naming the source and its consequence."""

        if self.source is WorkloadIdentitySource.METADATA:
            return f"provider metadata identity: {self.principal}"
        return f"provider metadata identity unavailable ({self.source}): {self.detail}"


class MetadataResponse(tp.NamedTuple):
    """The only three things this module reads from a metadata reply."""

    status: int
    flavor: str
    body: str


MetadataFetch: tp.TypeAlias = tp.Callable[[str, float], MetadataResponse]


def resolve_workload_identity(
    *,
    enabled: bool,
    url: str = GCP_METADATA_IDENTITY_URL,
    timeout_s: float = 2.0,
    fetch: MetadataFetch | None = None,
) -> WorkloadIdentity:
    """Ask the provider metadata path who this host is.

    Args:
        enabled: The operator flag. ``False`` -- the default everywhere -- makes
            this a no-op returning ``DISABLED``.
        url: Metadata identity path. Defaults to the GCP one; overridable only
            so tests can point it at a local server.
        timeout_s: Per-request timeout in seconds.
        fetch: Transport, for tests. Defaults to a bounded stdlib request.

    Returns:
        A :class:`WorkloadIdentity` that is ``ABSENT`` only on a conclusive
        negative from an authenticated metadata service, and ``UNKNOWN``
        whenever the question could not be answered.
    """

    if not enabled:
        return WorkloadIdentity(
            source=WorkloadIdentitySource.DISABLED,
            detail="provider metadata identity is disabled; this host's identity was not probed",
        )
    transport = fetch or _fetch
    try:
        response = transport(url, float(timeout_s))
    except Exception:
        # Transport exceptions carry request URLs, proxy configuration and, from
        # custom transports, arbitrary text. None of it is needed to say "we
        # could not ask", and one of them one day will not be safe to repeat.
        return WorkloadIdentity(
            source=WorkloadIdentitySource.UNKNOWN,
            detail="the provider metadata service could not be reached",
        )
    if not isinstance(response, MetadataResponse):
        return WorkloadIdentity(
            source=WorkloadIdentitySource.UNKNOWN,
            detail="the provider metadata transport returned an unusable reply",
        )
    if response.flavor != METADATA_FLAVOR_GOOGLE:
        # Something answered on the metadata address that is not the metadata
        # service. Trusting it would let anything reachable assert a principal.
        return WorkloadIdentity(
            source=WorkloadIdentitySource.UNKNOWN,
            detail="the reply on the metadata path did not identify itself as the provider metadata service",
        )
    if response.status == 404:
        return WorkloadIdentity(
            source=WorkloadIdentitySource.ABSENT,
            detail="the provider metadata service reports no workload identity for this host",
        )
    if response.status != 200:
        return WorkloadIdentity(
            source=WorkloadIdentitySource.UNKNOWN,
            detail=f"the provider metadata service answered with HTTP {int(response.status)}",
        )
    principal = response.body.strip() if isinstance(response.body, str) else ""
    if not _PRINCIPAL.fullmatch(principal):
        # A principal is an email-shaped name. Anything else on this path is
        # either a different endpoint or something impersonating one, and in the
        # first case it could be a token -- so it is never echoed.
        return WorkloadIdentity(
            source=WorkloadIdentitySource.UNKNOWN,
            detail="the provider metadata service returned a value that is not a workload principal",
        )
    return WorkloadIdentity(source=WorkloadIdentitySource.METADATA, principal=principal)


class _MetadataUnreachable(RuntimeError):
    """Internal marker; callers see ``UNKNOWN`` rather than this."""


def _fetch(url: str, timeout_s: float) -> MetadataResponse:
    """Read a bounded metadata reply with the provider's required request header.

    Not retried. A metadata service on the same host either answers immediately
    or is not there, and an identity probe that blocks a launch while it retries
    is worse than an identity probe that reports ``UNKNOWN``.
    """

    request = urllib.request.Request(url, method="GET", headers={METADATA_FLAVOR_HEADER: METADATA_FLAVOR_GOOGLE})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = response.read(_MAX_IDENTITY_BYTES)
            flavor = response.headers.get(METADATA_FLAVOR_HEADER, "")
            return MetadataResponse(status=int(response.status), flavor=str(flavor), body=_text(payload))
    except urllib.error.HTTPError as exc:
        flavor = exc.headers.get(METADATA_FLAVOR_HEADER, "") if exc.headers is not None else ""
        status = int(exc.code)
        exc.close()
        # The body of an error reply is not read: on a misconfigured path it is
        # the one place a credential could appear in a failure message.
        return MetadataResponse(status=status, flavor=str(flavor), body="")
    except (urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError, OSError) as exc:
        raise _MetadataUnreachable("the provider metadata service could not be reached") from exc


def _text(payload: bytes) -> str:
    """Decode a metadata body without raising on provider noise."""

    try:
        return payload.decode("ascii")
    except UnicodeDecodeError:
        return ""


__all__ = (
    "GCP_METADATA_IDENTITY_URL",
    "METADATA_FLAVOR_GOOGLE",
    "METADATA_FLAVOR_HEADER",
    "MetadataFetch",
    "MetadataResponse",
    "WorkloadIdentity",
    "WorkloadIdentitySource",
    "resolve_workload_identity",
)
