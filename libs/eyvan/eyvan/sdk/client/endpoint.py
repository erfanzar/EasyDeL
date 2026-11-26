# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validated control-plane endpoint values."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from urllib.parse import urlsplit

DEFAULT_CONTROL_PLANE_URL = "http://127.0.0.1:8080"


def _normalized_url(value: str) -> str:
    """Return one validated, trailing-slash-free HTTP(S) URL."""

    normalized = value.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("control-plane endpoint must be an absolute HTTP or HTTPS URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("control-plane endpoint cannot contain credentials, a query, or a fragment")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("control-plane endpoint has an invalid port") from exc
    return normalized


@dataclass(frozen=True, slots=True)
class ControlPlaneEndpoint:
    """A normalized absolute URL for an Eyvan control plane."""

    url: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "url", _normalized_url(self.url))

    @classmethod
    def parse(cls, value: str | ControlPlaneEndpoint) -> ControlPlaneEndpoint:
        """Validate and normalize an absolute HTTP(S) control-plane URL."""

        if isinstance(value, cls):
            return value
        return cls(value)

    @property
    def is_loopback(self) -> bool:
        """Return whether this endpoint resolves explicitly to the local host."""

        hostname = urlsplit(self.url).hostname
        if hostname is None:
            return False
        if hostname.casefold() == "localhost":
            return True
        try:
            return ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            return False
