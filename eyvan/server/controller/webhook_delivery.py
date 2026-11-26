# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pushing lifecycle events out, so nobody has to poll for them.

The audit trail has always been readable -- over SSE, by a client willing to
hold a connection open. That is the wrong shape for the things people actually
want notified: a Slack workspace, a pager, a small script on somebody's laptop.
So "tell me when my job finishes" meant writing a poller, and pollers are how a
control plane acquires a hundred clients asking the same question every five
seconds.

Three properties this delivery has to have, and each one is a way it could
quietly go wrong:

**A dead receiver must not stall the plane.** Deliveries run last, on their own
interval, with a bounded timeout, and each subscription's failure is isolated
from the others. An endpoint that accepts a connection and never answers is the
ordinary failure here, not the exotic one.

**The cursor advances only on success**, which makes delivery at-least-once
rather than at-most-once. A receiver that sees a duplicate can dedupe on the
event id; one that never sees an event has no way to know.

**A filtered-out event still advances the cursor.** Otherwise a subscription
narrowed to one action re-reads everything before its filter forever, and gets
slower every day the fleet is used.
"""

from __future__ import annotations

import ipaddress
import json
import typing as tp

import httpx

from eyvan._internal.json_types import JsonObject
from eyvan.records import AuditEvent, WebhookRecord, utcnow
from eyvan.server.controller._support import logger
from eyvan.server.service.webhooks import (
    SEQUENCE_HEADER,
    SIGNATURE_HEADER,
    WebhookTargetRefused,
    matches,
    resolve_target_addresses,
    sign,
    validate_target,
)

#: Events read per subscription per cycle. Bounded so one subscription that has
#: been down for a day cannot hold the phase open while it catches up -- it
#: catches up over several cycles instead, and every other subscription keeps
#: being delivered meanwhile.
_DELIVERY_BATCH = 50

#: Consecutive failures before a subscription is disabled. A receiver that has
#: been refusing for this long is not coming back on its own, and retrying it
#: forever spends the plane's time and fills its logs. Disabling *records why*,
#: because a subscription that silently went quiet cannot be told apart from a
#: fleet with nothing happening in it.
_FAILURE_LIMIT = 10

#: How long one delivery may take. Short: this runs on the leader, and a
#: receiver that needs longer than this to accept a POST is one that will be
#: retried rather than waited for.
_DELIVERY_TIMEOUT_S = 10.0


def delivery_body(webhook: WebhookRecord, events: tp.Sequence[AuditEvent]) -> JsonObject:
    """Return the document one delivery carries.

    A batch rather than one event per request. A busy fleet emits events far
    faster than a receiver's rate limit, and a delivery per event is how an
    integration gets itself throttled by its own chat provider.
    """

    return {
        "webhook_id": webhook.id,
        "org_id": webhook.org_id,
        "project_id": webhook.project_id,
        "delivered_at": utcnow().isoformat(),
        "events": [event.model_dump(mode="json") for event in events],
    }


def _host_header(url: httpx.URL) -> str:
    """Return the Host header a fetch of ``url`` as written would carry.

    A POST sent to a pinned address would otherwise announce the address as
    its host; the receiver answers for the name it was given, so that is what
    the header must say.
    """

    host = f"[{url.host}]" if ":" in url.host else url.host
    return host if url.port is None else f"{host}:{url.port}"


class _ControllerWebhookDelivery:
    """Deliver committed audit events to each project's subscriptions."""

    async def deliver_webhooks_once(self) -> int:
        """Send one batch to every subscription with something to send.

        Returns:
            How many subscriptions were delivered to, so a caller can tell a
            quiet cycle from a cycle that did nothing because it was broken.
        """

        async with self.repository.transaction() as transaction:
            webhooks = [item for item in await transaction.list_webhooks() if item.enabled]
        if not webhooks:
            return 0

        delivered = 0
        for webhook in webhooks:
            try:
                if await self._deliver_one(webhook):
                    delivered += 1
            except Exception as error:
                # Isolated deliberately. A subscription pointing at something
                # that raises in a way nothing here anticipated is one broken
                # integration, not a reason to stop notifying everybody else.
                logger.warning("webhook %s delivery failed unexpectedly (%s)", webhook.id, type(error).__name__)
        return delivered

    async def _deliver_one(self, webhook: WebhookRecord) -> bool:
        """Send this subscription's next batch, returning whether anything went."""

        async with self.repository.transaction() as transaction:
            events = await transaction.list_events(
                org_id=webhook.org_id,
                project_id=webhook.project_id,
                after_sequence=webhook.delivered_sequence,
                limit=_DELIVERY_BATCH,
            )
        if not events:
            return False

        # The cursor covers everything read, not everything sent. A subscription
        # filtered to one action would otherwise re-read the whole ledger before
        # its filter on every cycle, forever, getting slower as the fleet is used.
        highest = max(event.sequence for event in events)
        wanted = [
            event for event in events if matches(event.entity_type, event.action, webhook.entity_types, webhook.actions)
        ]
        if not wanted:
            await self._advance(webhook, highest)
            return False

        body = json.dumps(delivery_body(webhook, wanted), separators=(",", ":")).encode()
        try:
            # Re-checked at delivery, not only at creation: a name that resolved
            # publicly when the subscription was written can resolve to loopback
            # now, and that gap is the whole point of a rebinding attack.
            #
            # The check validates the one resolution this delivery pins to, so
            # the POST cannot re-resolve the name to an address nobody vetted.
            # Where private targets are allowed at all there is nothing to pin:
            # the guard has no address policy to enforce.
            resolved = () if self.settings.webhook_allow_private_targets else resolve_target_addresses(webhook.url)
            target = httpx.URL(webhook.url)
            headers = {
                "Content-Type": "application/json",
                SIGNATURE_HEADER: sign(webhook.secret, body),
                SEQUENCE_HEADER: str(highest),
            }
            extensions: dict[str, str] = {}
            if resolved:
                validate_target(
                    webhook.url,
                    allow_private=self.settings.webhook_allow_private_targets,
                    resolve=lambda _host: resolved,
                )
                headers["Host"] = _host_header(target)
                if target.scheme == "https":
                    # Connect to the vetted address; speak TLS for the name the
                    # receiver answers as, so its certificate still verifies
                    # against the hostname it was issued for.
                    extensions["sni_hostname"] = target.host
                target = target.copy_with(host=str(resolved[0]))
            elif self.settings.webhook_allow_private_targets:
                # Private targets intentionally retain the normal hostname
                # lookup behavior: the operator has opted out of this guard.
                validate_target(webhook.url, allow_private=True)
            else:
                try:
                    ipaddress.ip_address(target.host)
                except ValueError:
                    # An empty answer is not safe to fetch as a hostname. A
                    # second lookup during the POST could return a private
                    # address that this delivery never validated.
                    raise WebhookTargetRefused("webhook hostname did not resolve to an address") from None
                # Address literals do not undergo DNS rebinding; validate the
                # literal itself and send it as written.
                validate_target(webhook.url, allow_private=False)
            async with httpx.AsyncClient(timeout=_DELIVERY_TIMEOUT_S, follow_redirects=False) as client:
                response = await client.post(
                    str(target),
                    content=body,
                    headers=headers,
                    extensions=extensions,
                )
            response.raise_for_status()
        except (httpx.HTTPError, WebhookTargetRefused) as error:
            # Exception text can echo signed URLs, headers or response bytes.
            # Keep only a controlled category/status; the subscription id already
            # identifies the target without guessing credential parameter names.
            detail = type(error).__name__
            if isinstance(error, httpx.HTTPStatusError):
                detail += f": HTTP {error.response.status_code}"
            await self._record_failure(webhook, detail)
            return False

        await self._advance(webhook, highest)
        return True

    async def _advance(self, webhook: WebhookRecord, sequence: int) -> None:
        """Record that this subscription has been delivered through ``sequence``."""

        async with self.repository.transaction() as transaction:
            current = await transaction.get_webhook(webhook.id)
            if current is None:
                return
            await transaction.put_webhook(
                current.model_copy(
                    update={
                        # Never moved backwards: this cycle's read and the write
                        # are separate transactions, so a concurrent delivery may
                        # already have gone further.
                        "delivered_sequence": max(current.delivered_sequence, sequence),
                        "consecutive_failures": 0,
                        "updated_at": utcnow(),
                    }
                )
            )

    async def _record_failure(self, webhook: WebhookRecord, detail: str) -> None:
        """Count one failure, and disable the subscription once it is hopeless."""

        async with self.repository.transaction() as transaction:
            current = await transaction.get_webhook(webhook.id)
            if current is None:
                return
            failures = current.consecutive_failures + 1
            disabled = failures >= _FAILURE_LIMIT
            await transaction.put_webhook(
                current.model_copy(
                    update={
                        "consecutive_failures": failures,
                        "enabled": not disabled,
                        # Recorded whether or not it disabled anything: the first
                        # question about a quiet integration is what the receiver
                        # last said, and nowhere else keeps it.
                        "disabled_reason": (
                            f"disabled after {failures} consecutive failures; last: {detail}"[:512] if disabled else None
                        ),
                        "updated_at": utcnow(),
                    }
                )
            )
        logger.warning("webhook %s delivery failed (%d consecutive): %s", webhook.id, failures, detail)
