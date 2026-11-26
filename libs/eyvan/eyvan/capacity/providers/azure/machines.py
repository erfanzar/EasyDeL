# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""What one Eyvan cell looks like as an Azure VM, and who owns it.

The same ownership story as EC2, for the same reason: ARM has no notion of a
cell, so which incarnation asked for a machine lives in tags. An instance whose
tags do not match is a leak to delete rather than capacity to adopt.

Azure makes one part easier and one part harder.

Easier: **the VM name is the idempotency key.** A ``PUT`` to the same path is the
same machine, so a retried create needs no derived token -- it either returns
what exists or makes what does not. The cell id becomes the VM name, sanitized to
what Azure accepts.

Harder: **eviction is a policy, not a default.** A Spot VM whose eviction policy
is ``Deallocate`` stops but keeps its disks, its NIC and its name -- so the fleet
stops paying for compute and keeps paying for storage, holding a machine no
scheduler can place work on. ``Delete`` is what makes a reclamation actually
release, which is the same reasoning that makes EC2 spot requests one-time here.
"""

from __future__ import annotations

import base64
import re
import typing as tp

from eyvan._internal.json_types import JsonObject, JsonValue
from eyvan.capacity.drivers.base import MutationContext
from eyvan.capacity.providers.azure.api import AzureVirtualMachine

#: Tag naming the cell a VM belongs to.
CELL_TAG = "eyvan-cell"

#: Tag naming the incarnation that asked for it.
GENERATION_TAG = "eyvan-generation"

#: Tag carrying the fencing token of the creating mutation.
FENCE_TAG = "eyvan-fence"

#: Provisioning states in which the VM still exists and still bills.
LIVE_STATES = frozenset({"Creating", "Updating", "Succeeded", "Failed"})

#: Longest VM name Azure accepts on Linux. Longer cell ids are hashed into the
#: suffix rather than truncated: truncation makes two distinct cells collide on
#: one machine, which is the worst outcome available here.
_MAX_NAME = 64

_ILLEGAL = re.compile(r"[^A-Za-z0-9-]")


def vm_name(cell_id: str) -> str:
    """Return the VM name for a cell, within what Azure accepts.

    Deterministic, because it is the idempotency key: a name derived differently
    on a retry would create a second machine instead of finding the first.
    """

    cleaned = _ILLEGAL.sub("-", cell_id).strip("-") or "cell"
    if len(cleaned) <= _MAX_NAME:
        return cleaned
    import hashlib

    digest = hashlib.sha256(cell_id.encode()).hexdigest()[:12]
    return f"{cleaned[: _MAX_NAME - 13]}-{digest}"


def ownership_tags(cell_id: str, context: MutationContext) -> dict[str, str]:
    """Return the tags that make a VM's owner readable from ARM alone."""

    return {
        CELL_TAG: cell_id,
        GENERATION_TAG: str(context.generation),
        FENCE_TAG: context.fencing_token,
    }


def owns(machine: AzureVirtualMachine, cell_id: str, context: MutationContext) -> bool:
    """Return whether this VM belongs to *this* incarnation of the cell."""

    return (
        machine.tags.get(CELL_TAG) == cell_id
        and machine.tags.get(GENERATION_TAG) == str(context.generation)
        and machine.tags.get(FENCE_TAG) == context.fencing_token
    )


def is_live(machine: AzureVirtualMachine) -> bool:
    """Return whether this VM still exists and still costs money."""

    return machine.provisioning_state in LIVE_STATES


def vm_body(
    cell_id: str,
    desired: tp.Mapping[str, JsonValue],
    context: MutationContext,
    *,
    location: str,
) -> JsonObject:
    """Build the ARM document that creates one cell's VM.

    Args:
        cell_id: The cell being created.
        desired: The capacity document, already validated by the driver.
        context: The fencing context, which supplies the ownership tags.
        location: The Azure region, which lives on the resource rather than in
            the path.

    Returns:
        The request body for ``PUT .../virtualMachines/{name}``.
    """

    properties: JsonObject = {
        "hardwareProfile": {"vmSize": str(desired["vm_size"])},
        "storageProfile": {
            "imageReference": _image_reference(desired),
            "osDisk": {"createOption": "FromImage", "deleteOption": "Delete"},
        },
        "osProfile": {
            "computerName": vm_name(cell_id),
            "adminUsername": str(desired.get("admin_username") or "eyvan"),
            "linuxConfiguration": {"disablePasswordAuthentication": True},
        },
        "networkProfile": {
            "networkInterfaces": [
                {
                    "id": str(desired["network_interface_id"]),
                    # The NIC outlives the VM unless it is told not to. Left
                    # behind, it holds an address and a name a later cell of the
                    # same id then cannot use.
                    "properties": {"deleteOption": "Delete"},
                }
            ]
        },
    }

    capacity_class = str(desired.get("capacity_class") or "on_demand").replace("-", "_")
    if capacity_class == "spot":
        properties["priority"] = "Spot"
        # `Delete`, never `Deallocate`. A deallocated Spot VM keeps its disks,
        # its NIC and its name: the fleet stops paying for compute and keeps
        # paying for storage, holding a machine no scheduler can place work on.
        properties["evictionPolicy"] = "Delete"
        properties["billingProfile"] = {"maxPrice": -1}

    user_data = desired.get("user_data")
    if isinstance(user_data, str) and user_data:
        # ARM's userData contract is base-64: sent raw, a cloud-init script is
        # decoded to garbage by the platform while provisioningState still
        # reads Succeeded, so the corruption never surfaces as an error.
        properties["userData"] = base64.b64encode(user_data.encode("utf-8")).decode("ascii")

    body: JsonObject = {
        "location": location,
        "tags": ownership_tags(cell_id, context),
        "properties": properties,
    }
    zone = desired.get("zone")
    if isinstance(zone, str) and zone:
        body["zones"] = [zone]
    return body


def _image_reference(desired: tp.Mapping[str, JsonValue]) -> JsonObject:
    """Return the image the VM boots, from either spelling an operator uses."""

    image_id = desired.get("image_id")
    if isinstance(image_id, str) and image_id:
        return {"id": image_id}
    return {
        "publisher": str(desired.get("image_publisher") or "Canonical"),
        "offer": str(desired.get("image_offer") or "ubuntu-24_04-lts"),
        "sku": str(desired.get("image_sku") or "server"),
        "version": str(desired.get("image_version") or "latest"),
    }


def observation_metadata(machine: AzureVirtualMachine) -> JsonObject:
    """Return what a cell observation carries about its VM."""

    return {
        "vm_name": machine.name,
        "provisioning_state": machine.provisioning_state,
        "vm_size": machine.vm_size,
        "zone": machine.zone,
        "private_ip": machine.private_ip,
    }


__all__ = (
    "CELL_TAG",
    "FENCE_TAG",
    "GENERATION_TAG",
    "LIVE_STATES",
    "is_live",
    "observation_metadata",
    "ownership_tags",
    "owns",
    "vm_body",
    "vm_name",
)
