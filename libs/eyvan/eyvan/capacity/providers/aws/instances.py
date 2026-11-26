# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""What one Eyvan cell looks like as EC2 instances, and who owns them.

EC2 has no notion of a cell, so ownership lives in tags. That is not a
convenience -- it is the whole fencing mechanism on this provider. A control
plane that crashed between "launch" and "record" comes back to an instance it
cannot distinguish from somebody else's unless the instance says, in a tag, which
cell and which *generation* asked for it.

Three tags, and each answers a different question:

* ``eyvan:cell`` -- which cell this belongs to. Without it a leaked instance is
  indistinguishable from an unrelated one and gets left running forever.
* ``eyvan:generation`` -- which incarnation asked for it. An instance from an
  earlier generation is a leak to terminate, not capacity to adopt: adopting it
  is how two generations end up sharing a machine.
* ``eyvan:fence`` -- the fencing token of the mutation that created it. Two
  control-plane replicas both believing they are leader produce the same cell id
  and the same generation; only the token differs.

The client token is derived from the same three, because EC2's own idempotency
is keyed on it: retrying an inconclusive ``RunInstances`` with the same token
returns the original instance instead of launching a second one. Deriving it
rather than generating it is what makes the retry safe.
"""

from __future__ import annotations

import base64
import hashlib
import typing as tp

from eyvan._internal.json_types import JsonObject, JsonValue
from eyvan.capacity.drivers.base import MutationContext
from eyvan.capacity.providers.aws.api import Ec2Instance

#: Tag naming the cell an instance belongs to.
CELL_TAG = "eyvan:cell"

#: Tag naming the incarnation that asked for it.
GENERATION_TAG = "eyvan:generation"

#: Tag carrying the fencing token of the creating mutation.
FENCE_TAG = "eyvan:fence"

#: EC2 instance states that still hold hardware and still cost money. An
#: instance leaving these is gone; one inside them is a leak if nothing owns it.
LIVE_STATES = frozenset({"pending", "running", "stopping", "stopped", "shutting-down"})

#: States in which an instance is on its way out and cannot be revived.
TERMINAL_STATES = frozenset({"shutting-down", "terminated"})


def ownership_tags(cell_id: str, context: MutationContext) -> dict[str, str]:
    """Return the tags that make an instance's owner readable from EC2 alone."""

    return {
        CELL_TAG: cell_id,
        GENERATION_TAG: str(context.generation),
        FENCE_TAG: context.fencing_token,
    }


def client_token(cell_id: str, context: MutationContext) -> str:
    """Return the idempotency token for one cell incarnation's launch.

    Derived, not generated. EC2 returns the original instance for a repeated
    token, so a retry after a call that may or may not have launched something
    returns that same instance rather than a second one -- which is the only way
    an inconclusive launch can be retried safely.

    Hashed because EC2 bounds the token at 64 characters and a cell id plus a
    fencing token routinely exceeds that; truncating either instead would make
    two distinct incarnations collide.
    """

    material = f"{cell_id}\n{context.generation}\n{context.fencing_token}".encode()
    return hashlib.sha256(material).hexdigest()[:48]


def owns(instance: Ec2Instance, cell_id: str, context: MutationContext) -> bool:
    """Return whether this instance belongs to *this* incarnation of the cell."""

    return (
        instance.tags.get(CELL_TAG) == cell_id
        and instance.tags.get(GENERATION_TAG) == str(context.generation)
        and instance.tags.get(FENCE_TAG) == context.fencing_token
    )


def is_live(instance: Ec2Instance) -> bool:
    """Return whether this instance still holds hardware."""

    return instance.state in LIVE_STATES and instance.state not in TERMINAL_STATES


def launch_parameters(
    cell_id: str,
    desired: tp.Mapping[str, JsonValue],
    context: MutationContext,
) -> dict[str, str]:
    """Build the ``RunInstances`` parameters for one cell.

    Args:
        cell_id: The cell being created.
        desired: The capacity document, already validated by the driver.
        context: The fencing context, which supplies both the ownership tags and
            the idempotency token.

    Returns:
        Form parameters for the EC2 Query API.
    """

    params: dict[str, str] = {
        "ImageId": str(desired["image_id"]),
        "InstanceType": str(desired["instance_type"]),
        "MinCount": "1",
        "MaxCount": "1",
        "ClientToken": client_token(cell_id, context),
    }
    subnet = desired.get("subnet_id")
    if isinstance(subnet, str) and subnet:
        params["SubnetId"] = subnet
    zone = desired.get("zone")
    if isinstance(zone, str) and zone:
        params["Placement.AvailabilityZone"] = zone
    profile = desired.get("iam_instance_profile")
    if isinstance(profile, str) and profile:
        params["IamInstanceProfile.Name"] = profile
    key_name = desired.get("key_name")
    if isinstance(key_name, str) and key_name:
        params["KeyName"] = key_name
    for index, group in enumerate(_string_list(desired.get("security_group_ids")), start=1):
        params[f"SecurityGroupId.{index}"] = group
    user_data = desired.get("user_data")
    if isinstance(user_data, str) and user_data:
        # The EC2 Query API decodes UserData itself, so the bytes go up
        # base64-encoded: raw text here reaches the instance as decoded garbage
        # -- a corrupted bootstrap, with no error anywhere naming the cause.
        params["UserData"] = base64.b64encode(user_data.encode("utf-8")).decode("ascii")

    capacity_class = str(desired.get("capacity_class") or "on_demand").replace("-", "_")
    if capacity_class == "spot":
        # `one-time` rather than `persistent`: a persistent request re-launches
        # the instance itself after a reclamation, behind Eyvan's back, and the
        # fleet would then be paying for capacity no cell owns and no scheduler
        # can place work on.
        params["InstanceMarketOptions.MarketType"] = "spot"
        params["InstanceMarketOptions.SpotOptions.SpotInstanceType"] = "one-time"
        params["InstanceMarketOptions.SpotOptions.InstanceInterruptionBehavior"] = "terminate"

    # Tagged at launch, in the same call. A separate CreateTags afterwards leaves
    # a window in which a crash produces an instance nothing can recognise -- and
    # that window is exactly when a crash is most likely, because it is the same
    # instant the record is being written.
    tags = ownership_tags(cell_id, context)
    params["TagSpecification.1.ResourceType"] = "instance"
    for index, (key, value) in enumerate(sorted(tags.items()), start=1):
        params[f"TagSpecification.1.Tag.{index}.Key"] = key
        params[f"TagSpecification.1.Tag.{index}.Value"] = value
    return params


def _string_list(value: JsonValue | None) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def observation_metadata(instance: Ec2Instance) -> JsonObject:
    """Return what a cell observation carries about its instance."""

    return {
        "instance_id": instance.instance_id,
        "instance_state": instance.state,
        "instance_type": instance.instance_type,
        "availability_zone": instance.availability_zone,
        "private_ip": instance.private_ip,
        "public_ip": instance.public_ip,
    }


__all__ = (
    "CELL_TAG",
    "FENCE_TAG",
    "GENERATION_TAG",
    "LIVE_STATES",
    "TERMINAL_STATES",
    "client_token",
    "is_live",
    "launch_parameters",
    "observation_metadata",
    "ownership_tags",
    "owns",
)
