# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Where a delivery may be sent, and how a receiver knows it is genuine.

Two questions with one answer each, kept out of the service so both are testable
without a plane.

The address check is the security half. A webhook is a URL somebody in a project
writes down, and the control plane is the thing that fetches it -- which makes
an unchecked subscription a request forgery primitive pointed at whatever the
plane can reach that the submitter cannot: the cloud metadata service on
169.254.169.254, an admin endpoint on loopback, a database on the private
network. Nothing comes back to the submitter, so this is not about reading a
response; it is about the POST itself being an action somewhere it should not
reach.

The signature is the other half, in the other direction: it is how the receiver
knows a delivery came from this plane and not from anyone who learned the URL.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import socket
import typing as tp
from urllib.parse import urlparse

#: Header carrying the HMAC of the body, `sha256=<hex>`.
SIGNATURE_HEADER = "X-Eyvan-Signature"

#: Header carrying the audit sequence a delivery ends at, so a receiver can
#: detect a gap rather than assuming it has seen everything.
SEQUENCE_HEADER = "X-Eyvan-Sequence"


class WebhookTargetRefused(ValueError):
    """Raised when a subscription names an address the plane must not fetch."""


def sign(secret: str, body: bytes) -> str:
    """Return the signature a receiver verifies a delivery with."""

    return f"sha256={hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()}"


def _resolved_addresses(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Return every address ``host`` resolves to, or nothing when it will not.

    Every one is checked, not the first. A name resolving to one public address
    and one private address is the ordinary shape of a DNS rebinding attempt,
    and checking only what `getaddrinfo` happens to return first would let the
    private one through.
    """

    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return []
    addresses = []
    for info in infos:
        try:
            addresses.append(ipaddress.ip_address(info[4][0]))
        except ValueError:  # pragma: no cover - getaddrinfo returned a non-address
            continue
    return addresses


def validate_target(
    url: str,
    *,
    allow_private: bool = False,
    resolve: tp.Callable[[str], list[ipaddress.IPv4Address | ipaddress.IPv6Address]] | None = None,
) -> None:
    """Refuse a subscription URL the control plane must not fetch.

    Args:
        url: The address a delivery would be POSTed to.
        allow_private: Permit loopback and private ranges. For a development
            plane and for fleets whose receiver genuinely is on the same private
            network; off by default because the safe reading of "somebody in a
            project asked us to POST to 169.254.169.254" is no.
        resolve: Injected in tests. Defaults to the system resolver.

    Raises:
        WebhookTargetRefused: If the scheme is not HTTP(S), the host is missing,
            or any address it resolves to is one the plane should not reach.
    """

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise WebhookTargetRefused(f"webhook URL must be http or https, not {parsed.scheme or 'a missing scheme'}")
    if not parsed.hostname:
        raise WebhookTargetRefused("webhook URL must name a host")
    if allow_private:
        return

    resolver = resolve or _resolved_addresses
    try:
        literal = [ipaddress.ip_address(parsed.hostname)]
    except ValueError:
        literal = []
    addresses = literal or resolver(parsed.hostname)
    if not addresses:
        # Unresolvable now is not proof of anything, and refusing it would make
        # subscription creation depend on DNS being healthy at that moment. The
        # delivery path checks again, which is where it matters.
        return
    for address in addresses:
        if address.is_loopback or address.is_private or address.is_link_local or address.is_reserved:
            raise WebhookTargetRefused(
                f"webhook URL resolves to {address}, which is on this plane's own network rather than "
                "the receiver's; a subscription is a request the control plane makes, so an internal "
                "address makes it a way to reach what the submitter cannot"
            )


def resolve_target_addresses(
    url: str,
    *,
    resolve: tp.Callable[[str], list[ipaddress.IPv4Address | ipaddress.IPv6Address]] | None = None,
) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Return the addresses a delivery for ``url`` may be pinned to.

    A delivery that connects to one of these after validating exactly this
    answer cannot be rebound: the address the guard vetted and the address the
    POST opens are the same, which a fetch that re-resolves the name cannot
    promise. Empty means there is nothing to pin -- ``url`` already names an
    address literal, or its name resolved to nothing this time. Callers must
    not fetch an unresolved hostname as written; they must retry the delivery.

    Args:
        url: The address a delivery would be POSTed to.
        resolve: Injected in tests. Defaults to the system resolver.
    """

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return []
    try:
        ipaddress.ip_address(parsed.hostname)
    except ValueError:
        return (resolve or _resolved_addresses)(parsed.hostname)
    return []  # an address literal cannot resolve differently later


def matches(
    event_entity_type: str,
    event_action: str,
    entity_types: tp.Sequence[str],
    actions: tp.Sequence[str],
) -> bool:
    """Return whether one audit event belongs in a subscription's delivery.

    An empty filter means everything. A filter that had to be written before
    anything arrived would be got wrong once and never revisited.
    """

    if entity_types and event_entity_type not in entity_types:
        return False
    return not (actions and event_action not in actions)


__all__ = (
    "SEQUENCE_HEADER",
    "SIGNATURE_HEADER",
    "WebhookTargetRefused",
    "matches",
    "resolve_target_addresses",
    "sign",
    "validate_target",
)
