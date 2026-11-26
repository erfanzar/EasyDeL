# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Provider metadata identity, and the difference between absent and unknown.

The interesting tests here are the negative ones. Anything can return "this host
is service-account X" when the metadata server answers; what separates a usable
identity source from a dangerous one is what it says when the answer is missing,
when something *else* answers, and when the network is simply in the way.

Three refusals, each earned by a real failure mode:

* a firewall between the host and ``metadata.google.internal`` looks exactly
  like a host with no identity unless the two are kept apart -- so only a
  conclusive negative from the metadata service itself becomes ``ABSENT``;
* anything reachable on the metadata address can assert a principal unless the
  provider's flavour header is checked on the *response*; and
* a deployment that ever points this at a token path would put a credential in
  whatever the "identity" turns into -- so a body that is not principal-shaped
  is refused and never echoed.
"""

from __future__ import annotations

import logging

import pytest

from eyvan.config import EyvanSettings
from eyvan.identity.metadata import (
    METADATA_FLAVOR_GOOGLE,
    MetadataResponse,
    WorkloadIdentitySource,
    resolve_workload_identity,
)
from eyvan.worker.identity import (
    agent_workload_identity,
    describe_agent_workload_identity,
    reset_agent_workload_identity,
)

PRINCIPAL = "eyvan-worker@example-project.iam.gserviceaccount.com"
#: What an access-token endpoint returns. Never a value this package handles,
#: which is exactly why a test has to prove it never becomes one.
TOKEN_SHAPED_BODY = "ya29.a0AfB_byBOGUS-ACCESS-TOKEN-VALUE-THAT-MUST-NEVER-SURFACE"


def _fetch(response: MetadataResponse):
    calls: list[tuple[str, float]] = []

    def fetch(url: str, timeout_s: float) -> MetadataResponse:
        calls.append((url, timeout_s))
        return response

    fetch.calls = calls  # type: ignore[attr-defined]
    return fetch


@pytest.fixture(autouse=True)
def _clear_cached_identity():
    reset_agent_workload_identity()
    yield
    reset_agent_workload_identity()


# --------------------------------------------------------------------------- present


def test_metadata_identity_is_used_when_the_provider_asserts_one() -> None:
    """The point of the feature: identity with no long-lived key anywhere."""

    identity = resolve_workload_identity(
        enabled=True,
        fetch=_fetch(MetadataResponse(status=200, flavor=METADATA_FLAVOR_GOOGLE, body=PRINCIPAL + "\n")),
    )

    assert identity.source is WorkloadIdentitySource.METADATA
    assert identity.principal == PRINCIPAL
    assert identity.is_proven
    assert identity.describe() == f"provider metadata identity: {PRINCIPAL}"


def test_the_flag_defaults_to_off_and_asks_nothing() -> None:
    """Off is not "absent": a probe that never ran learned nothing about the host."""

    fetch = _fetch(MetadataResponse(status=200, flavor=METADATA_FLAVOR_GOOGLE, body=PRINCIPAL))

    assert EyvanSettings().provider_metadata_identity_enabled is False
    identity = resolve_workload_identity(enabled=False, fetch=fetch)

    assert identity.source is WorkloadIdentitySource.DISABLED
    assert identity.principal == ""
    assert not identity.is_proven
    assert fetch.calls == []


# --------------------------------------------------------------------------- absent


def test_no_metadata_identity_is_reported_absent_explicitly() -> None:
    """A metadata service that answers "there is none" is a finding, not a gap."""

    identity = resolve_workload_identity(
        enabled=True,
        fetch=_fetch(MetadataResponse(status=404, flavor=METADATA_FLAVOR_GOOGLE, body="")),
    )

    assert identity.source is WorkloadIdentitySource.ABSENT
    assert identity.principal == ""
    assert "no workload identity" in identity.detail
    assert "absent" in identity.describe()


# --------------------------------------------------------------------------- unknown, never inferred


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (None, "an unreachable metadata service is a firewall as often as it is a missing one"),
        (
            MetadataResponse(status=500, flavor=METADATA_FLAVOR_GOOGLE, body=""),
            "a metadata service that failed did not answer the question",
        ),
        (
            MetadataResponse(status=403, flavor=METADATA_FLAVOR_GOOGLE, body=""),
            "a refused read is not proof that there is nothing to read",
        ),
        (
            MetadataResponse(status=200, flavor="", body=PRINCIPAL),
            "without the provider's flavour header, anything reachable can assert a principal",
        ),
    ],
)
def test_an_unanswerable_probe_is_unknown_rather_than_absent(response, reason: str) -> None:
    """Never turn "I could not find out" into "there is nothing there"."""

    def failing(url: str, timeout_s: float) -> MetadataResponse:
        raise OSError("connection refused")

    identity = resolve_workload_identity(enabled=True, fetch=failing if response is None else _fetch(response))

    assert identity.source is WorkloadIdentitySource.UNKNOWN, reason
    assert identity.source is not WorkloadIdentitySource.ABSENT
    assert not identity.is_proven


def test_a_token_shaped_body_is_refused_and_never_echoed() -> None:
    """The one path on which a credential could reach a diagnostic.

    A misconfigured deployment pointing this at the access-token endpoint must
    not produce an "identity" that is a bearer token, and must not put it in the
    detail, the description, or a log line.
    """

    caplog_records: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: caplog_records.append(record.getMessage())  # type: ignore[method-assign]
    logging.getLogger().addHandler(handler)
    try:
        identity = resolve_workload_identity(
            enabled=True,
            fetch=_fetch(MetadataResponse(status=200, flavor=METADATA_FLAVOR_GOOGLE, body=TOKEN_SHAPED_BODY)),
        )
    finally:
        logging.getLogger().removeHandler(handler)

    assert identity.source is WorkloadIdentitySource.UNKNOWN
    assert identity.principal == ""
    for text in (identity.detail, identity.describe(), *caplog_records):
        assert TOKEN_SHAPED_BODY not in text


def test_an_error_status_body_is_not_read_at_all() -> None:
    """A metadata error body is the other place a value could hide."""

    identity = resolve_workload_identity(
        enabled=True,
        fetch=_fetch(MetadataResponse(status=500, flavor=METADATA_FLAVOR_GOOGLE, body=TOKEN_SHAPED_BODY)),
    )

    assert identity.source is WorkloadIdentitySource.UNKNOWN
    assert TOKEN_SHAPED_BODY not in identity.detail
    assert TOKEN_SHAPED_BODY not in identity.describe()


# --------------------------------------------------------------------------- the agent's own plumbing


def test_the_agent_caches_a_conclusive_answer_and_re_probes_an_unknown_one() -> None:
    """Caching an unknown would make one DNS hiccup at boot a permanent verdict."""

    settings = EyvanSettings(provider_metadata_identity_enabled=True)

    def failing(url: str, timeout_s: float) -> MetadataResponse:
        raise OSError("connection refused")

    first = agent_workload_identity(settings, fetch=failing)
    assert first.source is WorkloadIdentitySource.UNKNOWN

    answering = _fetch(MetadataResponse(status=200, flavor=METADATA_FLAVOR_GOOGLE, body=PRINCIPAL))
    second = agent_workload_identity(settings, fetch=answering)
    assert second.source is WorkloadIdentitySource.METADATA

    third = agent_workload_identity(settings, fetch=failing)
    assert third.source is WorkloadIdentitySource.METADATA
    assert third.principal == PRINCIPAL


def test_the_agent_states_every_outcome_including_having_none() -> None:
    """An agent that logs nothing when it has no identity is indistinguishable
    from one nobody asked, and that ambiguity is what a rollout has to see
    through."""

    settings = EyvanSettings(provider_metadata_identity_enabled=True)
    absent = agent_workload_identity(
        settings,
        fetch=_fetch(MetadataResponse(status=404, flavor=METADATA_FLAVOR_GOOGLE, body="")),
    )

    line = describe_agent_workload_identity(absent)

    assert "absent" in line
    assert "no workload identity" in line
    reset_agent_workload_identity()
    disabled = describe_agent_workload_identity(agent_workload_identity(EyvanSettings()))
    assert "disabled" in disabled
