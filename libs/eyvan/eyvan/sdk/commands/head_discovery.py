# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Finding the heads a project already has.

Every other head command needs you to already know which head you mean: `head
status` takes a fleet file, `head down` takes one too, and both build the
instance name from it. That is fine once you know, and useless on the day you
walk up to a project somebody else configured -- or your own, three months later
-- and the only question is *what is already running here*.

The answer is a label search. `head up` stamps every instance it creates with
``eyvan-managed=control-plane-head`` and ``eyvan-head-fleet=<fleet>``, and it
refuses to adopt an instance missing them. Those same two labels are what makes
discovery honest rather than a name-prefix guess: an instance this finds is one
Eyvan created and would agree to manage, and an instance it misses is one Eyvan
would refuse anyway.

**Reads only.** Nothing here creates, adopts, or reconfigures anything. It is
the one head operation safe to run against a project you are not sure about,
which is exactly when you want it.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass

from eyvan._internal.json_types import JsonObject, is_json_object
from eyvan.sdk.commands.head import (
    HEAD_FLEET_LABEL_KEY,
    HEAD_MANAGED_LABEL_KEY,
    HEAD_MANAGED_LABEL_VALUE,
    HeadAction,
    HeadCommandError,
    HeadCommandRunner,
    HeadStep,
    _run_gcloud,
)

#: How long a discovery listing may take. Longer than a describe: a project-wide
#: instance list crosses every zone, and a timeout here costs a retry rather
#: than a half-finished mutation.
DISCOVERY_TIMEOUT_S = 90.0


@dataclass(frozen=True, slots=True)
class DiscoveredHead:
    """One control-plane head found in a project.

    Attributes:
        fleet: The fleet this head was created for, from its own label. This is
            the argument every other head command wants.
        instance: The VM name.
        zone: The zone it lives in.
        status: GCP's instance status -- ``RUNNING``, ``TERMINATED``, and so on.
            Reported rather than filtered: a stopped head is still a head, and
            hiding it would send somebody to create a second one beside it.
        address: The internal IPv4 address, when it has one. Absent for a
            stopped instance, which is why it is optional rather than blank.
        machine_type: The shape, short form.
    """

    fleet: str
    instance: str
    zone: str
    status: str
    address: str | None = None
    machine_type: str | None = None

    @property
    def url(self) -> str | None:
        """Return the control-plane URL this head serves, when it is reachable.

        ``None`` for a head with no address rather than a URL that cannot
        resolve: a caller that got a string would try it, fail on connect, and
        report a network problem instead of a stopped instance.
        """

        return None if self.address is None else f"http://{self.address}:8080"


def discovery_action(project: str) -> HeadAction:
    """Build the read that lists every Eyvan-managed head in ``project``.

    Filtered server-side on the ownership label rather than client-side on the
    name. A project can hold thousands of instances, and a filter that ships the
    whole list back to be sieved locally is one that times out on exactly the
    fleets big enough to need this command.
    """

    return HeadAction(
        step=HeadStep.INSTANCE_LOOKUP,
        summary=f"list instances in {project} carrying Eyvan's control-plane-head label",
        args=(
            "compute",
            "instances",
            "list",
            "--project",
            project,
            "--filter",
            f"labels.{HEAD_MANAGED_LABEL_KEY}={HEAD_MANAGED_LABEL_VALUE}",
            "--format",
            "json",
        ),
    )


def parse_heads(document: str) -> list[DiscoveredHead]:
    """Read the discovered heads out of a gcloud instance listing.

    An entry missing the fleet label is skipped rather than guessed at. The
    fleet name is what every following command is given, and inventing one from
    the instance name would send `head down` at a resource under a name nobody
    wrote.
    """

    try:
        payload = json.loads(document or "[]")
    except json.JSONDecodeError as exc:
        raise HeadCommandError(HeadStep.INSTANCE_LOOKUP, f"gcloud returned unreadable JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise HeadCommandError(HeadStep.INSTANCE_LOOKUP, "gcloud returned a non-list instance document")

    heads: list[DiscoveredHead] = []
    for item in payload:
        if not is_json_object(item):
            continue
        labels = item.get("labels")
        fleet = labels.get(HEAD_FLEET_LABEL_KEY) if is_json_object(labels) else None
        if not isinstance(fleet, str) or not fleet:
            continue
        heads.append(
            DiscoveredHead(
                fleet=fleet,
                instance=str(item.get("name") or ""),
                zone=_short(item.get("zone")),
                status=str(item.get("status") or "UNKNOWN"),
                address=_internal_address(item),
                machine_type=_short(item.get("machineType")) or None,
            )
        )
    return sorted(heads, key=lambda head: (head.fleet, head.instance))


def discover_heads(
    project: str,
    *,
    runner: HeadCommandRunner | None = None,
    timeout_s: float = DISCOVERY_TIMEOUT_S,
) -> list[DiscoveredHead]:
    """Return every Eyvan-managed control-plane head in ``project``.

    Args:
        project: The GCP project to search.
        runner: Injected command boundary, as every other head operation takes.
        timeout_s: Bound on the listing.

    Raises:
        HeadCommandError: If gcloud is absent, times out, or fails. A failed
            listing is never reported as an empty project -- "there are no
            heads here" is the answer that makes somebody create a duplicate.
    """

    action = discovery_action(project)
    execute = runner or _run_gcloud
    try:
        result = execute(action.args, stdin_text=None, timeout=timeout_s)
    except FileNotFoundError as exc:
        raise HeadCommandError(action.step, "gcloud is not installed or is not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise HeadCommandError(action.step, f"gcloud instance listing timed out after {timeout_s:g} seconds") from exc
    except OSError as exc:
        raise HeadCommandError(action.step, f"could not start gcloud: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[:400]
        raise HeadCommandError(action.step, f"could not list instances in {project}: {detail}")
    return parse_heads(result.stdout)


def _short(value: object) -> str:
    """Return the last path segment of a GCP self-link, or the value itself."""

    return str(value).rsplit("/", 1)[-1] if isinstance(value, str) and value else ""


def _internal_address(instance: JsonObject) -> str | None:
    """Return the instance's first internal IPv4 address, if it has one."""

    interfaces = instance.get("networkInterfaces")
    if not isinstance(interfaces, list):
        return None
    for interface in interfaces:
        if not is_json_object(interface):
            continue
        address = interface.get("networkIP")
        if isinstance(address, str) and address:
            return address
    return None


def configured_project(runner: HeadCommandRunner | None = None, *, timeout_s: float = 20.0) -> str | None:
    """Return the project gcloud is currently configured for, if any.

    Returned rather than required, so a caller can offer it as a default and
    still ask. A machine with no gcloud, a timed-out lookup, or one where nobody
    has run ``gcloud config set project``, gets ``None`` and a prompt -- never a
    guess.
    """

    execute = runner or _run_gcloud
    try:
        result = execute(("config", "get-value", "project"), stdin_text=None, timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = (result.stdout or "").strip()
    if result.returncode != 0 or not value or value in {"(unset)", "None"}:
        return None
    return value


__all__ = (
    "DISCOVERY_TIMEOUT_S",
    "DiscoveredHead",
    "configured_project",
    "discover_heads",
    "discovery_action",
    "parse_heads",
)
