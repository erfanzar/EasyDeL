# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pushing lifecycle events out, and the two ways that goes wrong.

The audit trail was readable over SSE, by a client willing to hold a connection
open -- the wrong shape for a Slack workspace or a pager, so "tell me when my
job finishes" meant writing a poller.

The two failure modes worth most of these tests:

A subscription is a URL somebody in a project writes down and the *control
plane* fetches, which makes an unchecked one a request-forgery primitive
pointed at whatever the plane can reach and the submitter cannot -- the metadata
service, an admin endpoint on loopback. Nothing comes back to the submitter, so
this is not about reading a response; the POST itself is the action.

And a delivery loop that gets its cursor wrong is invisible: it either replays
somebody's history into an incident channel, or silently stops, and a stopped
integration looks exactly like a quiet fleet.
"""

from __future__ import annotations

import ipaddress
import json

import httpx
import pytest

from eyvan.config import EyvanSettings
from eyvan.records import AuditEvent, Principal, Role, RoleGrant, WebhookRecord
from eyvan.server.controller.webhook_delivery import _ControllerWebhookDelivery, delivery_body
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.errors import AuthorizationError, NotFoundError
from eyvan.server.service.webhooks import (
    SEQUENCE_HEADER,
    SIGNATURE_HEADER,
    WebhookTargetRefused,
    matches,
    sign,
    validate_target,
)

SECRET = "k" * 32


def _principal(role: Role = Role.PROJECT_ADMIN, *, org: str = "acme", project: str = "research") -> Principal:
    return Principal(subject="alice", grants=(RoleGrant(role=role, org_id=org, project_id=project),))


# --------------------------------------------------------------------------- where a delivery may go


@pytest.mark.parametrize(
    "address",
    ["127.0.0.1", "169.254.169.254", "10.0.0.5", "192.168.1.10", "::1"],
)
def test_the_plane_refuses_to_fetch_its_own_network(address: str) -> None:
    """The whole security argument. 169.254.169.254 is the metadata service.

    A subscription is a request *this plane* makes, with this plane's network
    position. Somebody who can write one and cannot reach an internal address
    otherwise has just been handed a way to POST to it.
    """

    with pytest.raises(WebhookTargetRefused, match="own network"):
        validate_target(f"https://[{address}]/hook" if ":" in address else f"https://{address}/hook")


def test_every_address_a_name_resolves_to_is_checked() -> None:
    """One public and one private is the ordinary shape of a rebinding attempt.

    Checking whichever address the resolver happens to return first would let
    the private one through on the next lookup.
    """

    def resolve(host: str):
        del host
        return [ipaddress.ip_address("93.184.216.34"), ipaddress.ip_address("127.0.0.1")]

    with pytest.raises(WebhookTargetRefused):
        validate_target("https://mixed.invalid/hook", resolve=resolve)


def test_a_public_address_is_allowed() -> None:
    validate_target("https://hooks.example.com/eyvan", resolve=lambda _host: [ipaddress.ip_address("93.184.216.34")])


def test_a_private_target_is_allowed_where_an_operator_says_so() -> None:
    """A development plane, and fleets whose receiver shares the network."""

    validate_target("http://127.0.0.1:9000/hook", allow_private=True)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "gopher://example.com/", "not-a-url"])
def test_only_http_targets_are_accepted(url: str) -> None:
    with pytest.raises(WebhookTargetRefused):
        validate_target(url)


def test_a_name_that_will_not_resolve_now_is_not_refused_now() -> None:
    """Otherwise creating a subscription depends on DNS being healthy that second.

    The delivery path checks again, which is where it decides anything.
    """

    validate_target("https://not-yet.invalid/hook", resolve=lambda _host: [])


# --------------------------------------------------------------------------- proving a delivery is genuine


def test_the_signature_is_over_the_body_a_receiver_sees() -> None:
    body = b'{"events":[]}'

    assert sign(SECRET, body) == sign(SECRET, body)
    assert sign(SECRET, body) != sign("j" * 32, body)
    assert sign(SECRET, body) != sign(SECRET, b'{"events":[1]}')


# --------------------------------------------------------------------------- what gets delivered


@pytest.mark.parametrize(
    ("entity_types", "actions", "expected"),
    [
        ((), (), True),
        (("workload",), (), True),
        (("cell",), (), False),
        ((), ("workload.settled",), True),
        ((), ("workload.cancelled",), False),
        (("workload",), ("workload.cancelled",), False),
    ],
)
def test_an_empty_filter_means_everything(entity_types, actions, expected) -> None:
    """A filter that must be right before anything arrives is got wrong once."""

    assert matches("workload", "workload.settled", entity_types, actions) is expected


def _webhook(**overrides) -> WebhookRecord:
    fields = {
        "id": "wh_1",
        "org_id": "acme",
        "project_id": "research",
        "url": "https://hooks.example.com/eyvan",
        "secret": SECRET,
        "created_by": "alice",
        **overrides,
    }
    return WebhookRecord(**fields)


def test_a_batch_carries_the_events_and_names_the_subscription() -> None:
    """One request per event is how an integration gets itself rate-limited."""

    events = [
        AuditEvent(id="evt-1", sequence=7, actor="alice", entity_type="workload", entity_id="wl-1", action="a"),
        AuditEvent(id="evt-2", sequence=8, actor="alice", entity_type="workload", entity_id="wl-1", action="b"),
    ]

    body = delivery_body(_webhook(), events)

    assert body["webhook_id"] == "wh_1"
    assert [event["id"] for event in body["events"]] == ["evt-1", "evt-2"]


# --------------------------------------------------------------------------- the delivery loop


class _Plane(_ControllerWebhookDelivery):
    """The delivery mixin over an in-memory repository and a scripted receiver."""

    def __init__(self, repository, *, responses=None, allow_private: bool = True) -> None:
        self.repository = repository
        self.settings = EyvanSettings(
            artifact_token_key="k" * 32,
            webhook_allow_private_targets=allow_private,
        )
        self.sent: list[tuple[str, bytes, dict]] = []
        self.extensions: list[dict | None] = []
        self._responses = list(responses or [])

    async def _post(self, url, content, headers, extensions=None):
        self.sent.append((url, content, dict(headers)))
        self.extensions.append(extensions)
        if self._responses:
            outcome = self._responses.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
        return httpx.Response(200, request=httpx.Request("POST", url))


async def _seed(repository, *, events: int = 3, **overrides) -> WebhookRecord:
    webhook = _webhook(**overrides)
    async with repository.transaction() as tx:
        await tx.put_webhook(webhook)
        for index in range(events):
            await tx.append_event(
                AuditEvent(
                    id=f"evt-{index}",
                    actor="scheduler",
                    entity_type="workload",
                    entity_id=f"wl-{index}",
                    action="workload.settled",
                    org_id="acme",
                    project_id="research",
                )
            )
    return webhook


@pytest.mark.asyncio
async def test_a_delivery_carries_a_signature_the_receiver_can_check(monkeypatch) -> None:
    repository = InMemoryRepository()
    plane = _Plane(repository)
    await _seed(repository)
    _patch_transport(monkeypatch, plane)

    assert await plane.deliver_webhooks_once() == 1

    url, body, headers = plane.sent[0]
    assert url == "https://hooks.example.com/eyvan"
    assert headers[SIGNATURE_HEADER] == sign(SECRET, body)
    events = json.loads(body)["events"]
    assert events, "a delivery with no events is a request nobody asked for"
    # The sequence a receiver resumes from, so a gap is detectable rather than
    # assumed away.
    async with plane.repository.transaction() as tx:
        stored = await tx.get_webhook("wh_1")
    assert stored is not None
    assert headers[SEQUENCE_HEADER] == str(stored.delivered_sequence)


@pytest.mark.asyncio
async def test_a_delivery_connects_to_the_address_it_validated(monkeypatch) -> None:
    """The POST may not re-resolve the name after the guard has checked it.

    A low-TTL name alternating public and private answers passes a check that
    resolves once and connects after resolving again -- the rebinding a
    delivery pins itself against the address it validated to make impossible.
    """

    monkeypatch.setattr(
        "eyvan.server.service.webhooks._resolved_addresses",
        lambda _host: [ipaddress.ip_address("93.184.216.34")],
    )
    repository = InMemoryRepository()
    plane = _Plane(repository, allow_private=False)
    await _seed(repository)
    _patch_transport(monkeypatch, plane)

    assert await plane.deliver_webhooks_once() == 1

    url, _body, headers = plane.sent[0]
    assert url == "https://93.184.216.34/eyvan"
    assert headers["Host"] == "hooks.example.com"
    assert plane.extensions[0] == {"sni_hostname": "hooks.example.com"}


@pytest.mark.asyncio
async def test_an_unresolved_delivery_is_retried_without_a_second_dns_lookup(monkeypatch) -> None:
    """An empty first answer must not fall back to an unpinned hostname fetch.

    A second lookup could return a private address after the first lookup had
    no answer, allowing a rebinding attack to reach an internal service.
    """

    answers = iter(
        [
            [],
            [ipaddress.ip_address("93.184.216.34")],
            [ipaddress.ip_address("127.0.0.1")],
        ]
    )
    calls = 0

    def resolve(_host):
        nonlocal calls
        calls += 1
        return next(answers)

    monkeypatch.setattr("eyvan.server.service.webhooks._resolved_addresses", resolve)
    repository = InMemoryRepository()
    plane = _Plane(repository, allow_private=False)
    await _seed(repository)
    _patch_transport(monkeypatch, plane)

    assert await plane.deliver_webhooks_once() == 0
    assert plane.sent == []
    async with repository.transaction() as tx:
        stored = await tx.get_webhook("wh_1")
    assert stored is not None
    assert stored.consecutive_failures == 1
    assert calls == 1

    assert await plane.deliver_webhooks_once() == 1
    assert plane.sent[0][0] == "https://93.184.216.34/eyvan"
    assert calls == 2


@pytest.mark.asyncio
async def test_a_successful_delivery_is_not_sent_twice(monkeypatch) -> None:
    """The cursor advances on success, which is what makes this at-least-once."""

    repository = InMemoryRepository()
    plane = _Plane(repository)
    await _seed(repository)
    _patch_transport(monkeypatch, plane)

    await plane.deliver_webhooks_once()
    await plane.deliver_webhooks_once()

    assert len(plane.sent) == 1


@pytest.mark.asyncio
async def test_a_refused_delivery_is_retried(monkeypatch) -> None:
    """At-least-once: a receiver that never saw an event cannot ask for it."""

    repository = InMemoryRepository()
    plane = _Plane(repository, responses=[httpx.ConnectError("refused")])
    await _seed(repository)
    _patch_transport(monkeypatch, plane)

    await plane.deliver_webhooks_once()
    await plane.deliver_webhooks_once()

    assert len(plane.sent) == 2
    assert json.loads(plane.sent[0][1])["events"] == json.loads(plane.sent[1][1])["events"]


@pytest.mark.asyncio
async def test_a_filtered_out_event_still_advances_the_cursor(monkeypatch) -> None:
    """Otherwise a narrow subscription re-reads the whole ledger every cycle.

    Nothing is delivered, so nothing looks wrong -- while the read gets slower
    every day the fleet is used.
    """

    repository = InMemoryRepository()
    plane = _Plane(repository)
    await _seed(repository, actions=("nothing.matches",))
    _patch_transport(monkeypatch, plane)

    await plane.deliver_webhooks_once()

    assert plane.sent == []
    async with repository.transaction() as tx:
        stored = await tx.get_webhook("wh_1")
    assert stored is not None and stored.delivered_sequence > 0


@pytest.mark.asyncio
async def test_a_receiver_that_never_recovers_is_disabled_with_a_reason(monkeypatch) -> None:
    """Retrying forever spends the plane's time; going quiet explains nothing."""

    repository = InMemoryRepository()
    plane = _Plane(repository, responses=[httpx.ConnectError("refused")] * 20)
    await _seed(repository)
    _patch_transport(monkeypatch, plane)

    for _ in range(12):
        await plane.deliver_webhooks_once()

    async with repository.transaction() as tx:
        stored = await tx.get_webhook("wh_1")
    assert stored is not None
    assert not stored.enabled
    assert stored.disabled_reason and "ConnectError" in stored.disabled_reason


@pytest.mark.asyncio
@pytest.mark.parametrize("parameter", ("X-Amz-Signature", "X-Goog-Signature", "custom-credential"))
@pytest.mark.parametrize("failure", ("status", "transport", "unexpected"))
async def test_delivery_diagnostics_never_copy_signed_target_credentials(
    monkeypatch, caplog, parameter, failure
) -> None:
    credential = "test-only-signed-query-marker"
    url = f"https://hooks.example.com/eyvan?{parameter}={credential}"
    repository = InMemoryRepository()
    plane = _Plane(repository)
    await _seed(repository, url=url)
    sent: list[str] = []

    async def post(_client, target, *, content, headers, extensions=None):
        sent.append(target)
        if failure == "transport":
            raise httpx.ConnectError(f"receiver refused {target}")
        if failure == "unexpected":
            raise RuntimeError(f"receiver failed at {target}")
        return httpx.Response(503, request=httpx.Request("POST", target))

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    for _ in range(12):
        assert await plane.deliver_webhooks_once() == 0

    async with repository.transaction() as tx:
        stored = await tx.get_webhook("wh_1")
    assert stored is not None
    assert stored.delivered_sequence == 0
    assert credential not in caplog.text
    if failure == "unexpected":
        assert stored.enabled
        assert stored.consecutive_failures == 0
        assert len(sent) == 12
    else:
        assert not stored.enabled
        assert stored.consecutive_failures == 10
        assert len(sent) == 10
        assert stored.disabled_reason
        assert credential not in stored.disabled_reason
        assert (
            "HTTPStatusError" in stored.disabled_reason
            if failure == "status"
            else "ConnectError" in stored.disabled_reason
        )
    assert all(target == url for target in sent)


@pytest.mark.asyncio
async def test_one_broken_receiver_does_not_stop_the_others(monkeypatch) -> None:
    """A dead integration is one dead integration, not a stalled notifier."""

    repository = InMemoryRepository()
    plane = _Plane(repository, responses=[httpx.ConnectError("refused")])
    await _seed(repository)
    async with repository.transaction() as tx:
        await tx.put_webhook(_webhook(id="wh_2", url="https://second.example.com/eyvan"))
    _patch_transport(monkeypatch, plane)

    await plane.deliver_webhooks_once()

    assert [url for url, _body, _headers in plane.sent] == [
        "https://hooks.example.com/eyvan",
        "https://second.example.com/eyvan",
    ]


def _patch_transport(monkeypatch, plane: _Plane) -> None:
    """Route the mixin's HTTP call into the scripted receiver.

    Patched at the client rather than by injecting a transport, so the headers
    and body asserted here are the ones the real code builds.
    """

    async def post(self, url, *, content, headers, extensions=None):
        del self
        return await plane._post(url, content, headers, extensions)

    monkeypatch.setattr(httpx.AsyncClient, "post", post)


# --------------------------------------------------------------------------- who may subscribe


@pytest.mark.asyncio
async def test_a_subscription_starts_from_now_rather_than_from_the_beginning() -> None:
    """A new integration replaying a month into an incident channel is the
    first impression this feature would otherwise make."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    async with repository.transaction() as tx:
        for index in range(5):
            await tx.append_event(
                AuditEvent(
                    id=f"old-{index}",
                    actor="scheduler",
                    entity_type="workload",
                    entity_id="wl-old",
                    action="workload.settled",
                    org_id="acme",
                    project_id="research",
                )
            )

    webhook = await service.create_webhook(
        _principal(),
        org_id="acme",
        project_id="research",
        url="https://hooks.example.com/eyvan",
        secret=SECRET,
        allow_private_targets=True,
    )

    assert webhook.delivered_sequence >= 5


@pytest.mark.asyncio
async def test_a_read_never_returns_the_signing_key() -> None:
    """A signature anybody can forge proves nothing about who sent a delivery."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    await service.create_webhook(
        _principal(),
        org_id="acme",
        project_id="research",
        url="https://hooks.example.com/eyvan",
        secret=SECRET,
        allow_private_targets=True,
    )

    listed = await service.list_webhooks(_principal(), org_id="acme", project_id="research")

    assert listed[0].secret != SECRET


@pytest.mark.asyncio
async def test_subscribing_needs_more_than_permission_to_read() -> None:
    """A subscription sends this project's history somewhere of its choosing."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)

    with pytest.raises(AuthorizationError):
        await service.create_webhook(
            _principal(Role.PROJECT_VIEWER),
            org_id="acme",
            project_id="research",
            url="https://hooks.example.com/eyvan",
            secret=SECRET,
            allow_private_targets=True,
        )


@pytest.mark.asyncio
async def test_another_tenant_cannot_delete_a_subscription() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    webhook = await service.create_webhook(
        _principal(),
        org_id="acme",
        project_id="research",
        url="https://hooks.example.com/eyvan",
        secret=SECRET,
        allow_private_targets=True,
    )
    outsider = _principal(org="rival")

    with pytest.raises(AuthorizationError):
        await service.delete_webhook(outsider, webhook.id)


@pytest.mark.asyncio
async def test_deleting_something_that_is_not_there_says_so() -> None:
    service = ControlPlaneService(InMemoryRepository())

    with pytest.raises(NotFoundError):
        await service.delete_webhook(_principal(), "wh_missing")
