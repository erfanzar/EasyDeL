# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A workload-registered service endpoint must not aim the control plane inward.

The health probe GETs ``endpoint + health_path`` from the control-plane host
itself, so a replica that could register ``http://169.254.169.254/`` would turn
every health pass into a request the plane makes on the workload's behalf --
with the status and detail landing in probe counters the workload can read.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from eyvan.server.api.schemas import ServiceEndpointRequest


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://127.0.0.1:8080",
        "http://[::1]:8080",
        "http://169.254.169.254/latest/meta-data/",
        "http://[fe80::1]:8080",
        "http://[fd00::1]:8080",
        "http://[::ffff:127.0.0.1]:8080",
        "http://[::ffff:7f00:1]:8080",
        "http://[::ffff:169.254.169.254]:8080",
        "http://[::ffff:a9fe:a9fe]:8080",
    ],
)
def test_an_endpoint_that_resolves_into_the_plane_or_its_metadata_is_refused(endpoint: str) -> None:
    """Loopback names the plane, link-local the metadata service, unique-local
    the host's own private segments."""

    with pytest.raises(ValidationError, match="loopback, link-local, or unique-local"):
        ServiceEndpointRequest(endpoint=endpoint)


@pytest.mark.parametrize("endpoint", ["ftp://10.0.0.1:8080", "10.0.0.1:8080", "http://"])
def test_an_endpoint_without_an_http_target_is_refused(endpoint: str) -> None:
    """The probe is an HTTP GET, so anything it cannot fetch is only a string."""

    with pytest.raises(ValidationError, match="http"):
        ServiceEndpointRequest(endpoint=endpoint)


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://10.0.0.1:8080",
        "https://192.168.1.10:8443",
        "http://8.8.8.8",
        "http://[::ffff:10.0.0.1]:8080",
        "https://[::ffff:192.168.1.10]:8443",
    ],
)
def test_endpoints_on_site_local_or_public_space_are_accepted(endpoint: str) -> None:
    """Replicas register where they actually serve; that is not the plane."""

    assert ServiceEndpointRequest(endpoint=endpoint).endpoint == endpoint
