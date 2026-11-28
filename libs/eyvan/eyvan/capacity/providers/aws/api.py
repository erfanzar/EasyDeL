# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Talking to EC2, and telling its two kinds of failure apart.

**No new dependency.** The EC2 Query API is form-encoded parameters and an XML
response, and ``httpx`` is already core. A client library would bring an object
model that then has to be kept in agreement with the handful of actions Eyvan
actually calls.

The distinction this module exists to make is between a call that *conclusively*
failed and one whose outcome is unknown. AWS reports both as an error document,
and treating them alike is how a control plane either abandons a running
instance it is still paying for, or launches a second one beside it. So:

* An error code AWS documents as a client fault -- a malformed request, an
  unauthorized principal, a subnet that does not exist -- is conclusive.
  Retrying is guaranteed to fail identically, so the cell fails with a reason.
* A timeout, a refused connection, a 5xx, and the throttling codes are
  inconclusive. The call may have taken effect. The cell stays where it is and
  the next reconcile looks again.

``RunInstances`` is called with a client token, which is EC2's own idempotency
mechanism: the same token within its window returns the original instance rather
than launching a second one. That is what makes a retry after an inconclusive
launch safe, and it is the reason the token is derived from the fencing context
rather than generated fresh.
"""

from __future__ import annotations

import typing as tp
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx

from eyvan.capacity.drivers.base import CapacityProviderError, CapacityProviderFailureCode
from eyvan.capacity.providers.aws.signing import AwsCredentials, authorization_header

#: EC2 Query API version this client speaks. Pinned: the action parameters below
#: are the ones this version accepts, and letting the service choose would make
#: a response shape change arrive as an unexplained parse failure.
API_VERSION = "2016-11-15"

#: How long one EC2 call may take. A control-plane reconcile holds the leader,
#: so a call that needs longer than this is one to retry rather than wait for.
DEFAULT_TIMEOUT_S = 20.0

#: Error codes that mean the request itself is wrong and will stay wrong.
#: Everything not named here is treated as inconclusive, which is the safe
#: default: a code this list has never seen is one nobody has reasoned about,
#: and concluding "permanently broken" from it would strand a running instance.
CONCLUSIVE_ERROR_CODES = frozenset(
    {
        "AuthFailure",
        "Blocked",
        "DryRunOperation",
        "InvalidAMIID.Malformed",
        "InvalidAMIID.NotFound",
        "InvalidInstanceType",
        "InvalidKeyPair.NotFound",
        "InvalidParameterCombination",
        "InvalidParameterValue",
        "InvalidSecurityGroupID.NotFound",
        "InvalidSubnetID.NotFound",
        "MissingParameter",
        "OptInRequired",
        "UnauthorizedOperation",
        "VPCIdNotSpecified",
    }
)

#: Codes that specifically mean "not now", which is inconclusive *and* worth
#: naming: a fleet that reads these as capacity exhaustion backs off from a zone
#: that is merely rate-limiting the account.
THROTTLING_ERROR_CODES = frozenset({"RequestLimitExceeded", "Throttling", "SlowDown"})

#: Codes that mean the provider has no capacity of this shape right now. Their
#: own outcome, because "the zone is empty" and "the request is wrong" call for
#: entirely different backoffs.
CAPACITY_ERROR_CODES = frozenset(
    {
        "InsufficientHostCapacity",
        "InsufficientInstanceCapacity",
        "InsufficientReservedInstanceCapacity",
        "Unsupported",
    }
)


@dataclass(frozen=True, slots=True)
class Ec2Instance:
    """One instance, as much of it as a capacity driver needs."""

    instance_id: str
    state: str
    #: Tags, which is where the cell id, generation and fencing token live. EC2
    #: has no other place to record who owns an instance.
    tags: dict[str, str] = field(default_factory=dict)
    private_ip: str | None = None
    public_ip: str | None = None
    instance_type: str | None = None
    availability_zone: str | None = None
    #: Present when EC2 has told us this spot instance is going away. The
    #: distinction that makes a reclamation reportable as a preemption rather
    #: than as a machine that vanished.
    spot_interruption: str | None = None


class Ec2Api(tp.Protocol):
    """The four EC2 actions a capacity driver needs, and nothing else."""

    def run_instances(self, params: tp.Mapping[str, str]) -> list[Ec2Instance]: ...

    def describe_instances(self, filters: tp.Mapping[str, str]) -> list[Ec2Instance]: ...

    def terminate_instances(self, instance_ids: tp.Sequence[str]) -> None: ...

    def create_tags(self, instance_ids: tp.Sequence[str], tags: tp.Mapping[str, str]) -> None: ...

    def describe_spot_instance_requests(self, instance_ids: tp.Sequence[str]) -> dict[str, str]: ...


def parse_spot_instance_requests(document: str) -> dict[str, str]:
    """Map each spot request's instance id to its status code.

    ``marked-for-termination`` and friends are EC2's only pollable signal that a
    spot reclamation has begun -- DescribeInstances never carries it -- so this
    is what turns a vanishing machine into a reportable preemption. Requests
    without an attached instance id are skipped.
    """

    root = ElementTree.fromstring(document)
    statuses: dict[str, str] = {}
    for node in root.iter():
        if _strip_namespace(node.tag) != "spotInstanceRequestSet":
            continue
        for item in node:
            if _strip_namespace(item.tag) != "item":
                continue
            instance_id: str | None = None
            code: str | None = None
            for child in item:
                name = _strip_namespace(child.tag)
                if name == "instanceId":
                    instance_id = _text(child)
                elif name == "status":
                    code = next((_text(part) for part in child if _strip_namespace(part.tag) == "code"), None)
            if instance_id is not None and code is not None:
                statuses[instance_id] = code
    return statuses


def _text(node: ElementTree.Element | None) -> str | None:
    return None if node is None or node.text is None else node.text.strip()


def _strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_instances(document: str) -> list[Ec2Instance]:
    """Read every instance out of a DescribeInstances or RunInstances response.

    Namespace-insensitive on purpose: the EC2 response namespace carries the API
    version, and a client that matches on the full tag stops parsing the day the
    service is asked for a different one.
    """

    root = ElementTree.fromstring(document)
    instances: list[Ec2Instance] = []
    for node in root.iter():
        if _strip_namespace(node.tag) != "instancesSet":
            continue
        for item in node:
            if _strip_namespace(item.tag) != "item":
                continue
            fields: dict[str, tp.Any] = {"tags": {}}
            for child in item:
                name = _strip_namespace(child.tag)
                if name == "instanceId":
                    fields["instance_id"] = _text(child)
                elif name == "instanceState":
                    fields["state"] = next((_text(part) for part in child if _strip_namespace(part.tag) == "name"), None)
                elif name == "privateIpAddress":
                    fields["private_ip"] = _text(child)
                elif name == "ipAddress":
                    fields["public_ip"] = _text(child)
                elif name == "instanceType":
                    fields["instance_type"] = _text(child)
                elif name == "placement":
                    fields["availability_zone"] = next(
                        (_text(part) for part in child if _strip_namespace(part.tag) == "availabilityZone"),
                        None,
                    )
                elif name == "tagSet":
                    for tag in child:
                        key = next((_text(part) for part in tag if _strip_namespace(part.tag) == "key"), None)
                        value = next((_text(part) for part in tag if _strip_namespace(part.tag) == "value"), None)
                        if key is not None:
                            fields["tags"][key] = value or ""
            if fields.get("instance_id"):
                instances.append(
                    Ec2Instance(
                        instance_id=str(fields["instance_id"]),
                        state=str(fields.get("state") or "unknown"),
                        tags=fields["tags"],
                        private_ip=fields.get("private_ip"),
                        public_ip=fields.get("public_ip"),
                        instance_type=fields.get("instance_type"),
                        availability_zone=fields.get("availability_zone"),
                    )
                )
    return instances


def parse_error(document: str) -> tuple[str, str]:
    """Return the ``(code, message)`` of an EC2 error response.

    A response this cannot parse yields an empty code, which routes to the
    inconclusive branch -- the safe reading of "AWS said something and we could
    not tell what".
    """

    try:
        root = ElementTree.fromstring(document)
    except ElementTree.ParseError:
        return "", document[:200]
    code = ""
    message = ""
    for node in root.iter():
        name = _strip_namespace(node.tag)
        if name == "Code" and not code:
            code = _text(node) or ""
        elif name == "Message" and not message:
            message = _text(node) or ""
    return code, message or document[:200]


def failure_for(code: str, message: str) -> CapacityProviderError:
    """Classify one EC2 error into the outcome the fleet reasons about.

    The classification, not a translation. Everything Eyvan does next -- fail the
    cell, back off this zone, or try again unchanged -- follows from which of
    these three it is, and an unknown code takes the branch that assumes the call
    may have taken effect.
    """

    if code in CAPACITY_ERROR_CODES:
        return CapacityProviderError(
            f"EC2 has no capacity for this request ({code}): {message}",
            code=CapacityProviderFailureCode.STOCKOUT,
        )
    if code in THROTTLING_ERROR_CODES:
        # Not a stockout. A fleet that reads throttling as an empty zone backs
        # off from capacity that is there, because the account asked too fast.
        return CapacityProviderError(
            f"EC2 is rate limiting this account ({code}): {message}",
            code=CapacityProviderFailureCode.PROVIDER_UNAVAILABLE,
        )
    if code in CONCLUSIVE_ERROR_CODES:
        return CapacityProviderError(
            f"EC2 refused this request ({code}): {message}",
            code=CapacityProviderFailureCode.CONFIGURATION,
        )
    return CapacityProviderError(
        f"EC2 call did not conclude ({code or 'unclassified'}): {message}",
        code=CapacityProviderFailureCode.PROVIDER_UNAVAILABLE,
    )


@dataclass
class HttpEc2Api:
    """The EC2 Query API over ``httpx``.

    **Unverified against AWS.** Every behaviour in this provider that a test
    covers is covered against a fake; this class is the seam where the fake ends
    and a real account begins, and nothing here has been run against one. It is
    written to the documented Query API and pinned to one version so a
    disagreement shows up as a parse failure naming the response rather than as
    silently wrong capacity.
    """

    region: str
    credentials: AwsCredentials
    endpoint: str | None = None
    timeout_s: float = DEFAULT_TIMEOUT_S
    client: httpx.Client | None = None

    @property
    def host(self) -> str:
        return self.endpoint or f"ec2.{self.region}.amazonaws.com"

    def _call(self, action: str, params: tp.Mapping[str, str]) -> str:
        body = {**params, "Action": action, "Version": API_VERSION}
        # One encoding, used for the signature and for the wire. AWS recomputes
        # the payload digest from the body it receives, so hashing any encoding
        # other than the bytes sent -- a re-sorted urlencode posted against
        # httpx's insertion-order one -- is SignatureDoesNotMatch on every call.
        encoded_body = _encoded_body(body)
        headers = authorization_header(
            self.credentials,
            method="POST",
            path="/",
            params={},
            headers={"host": self.host, "content-type": "application/x-www-form-urlencoded; charset=utf-8"},
            region=self.region,
            service="ec2",
            moment=datetime.now(UTC),
            payload_sha256=_body_digest(encoded_body),
        )
        client = self.client or httpx.Client(timeout=self.timeout_s)
        try:
            response = client.post(f"https://{self.host}/", content=encoded_body, headers=headers)
        except httpx.HTTPError as error:
            # Inconclusive by construction: the request may have reached EC2 and
            # launched something. Concluding otherwise is how a fleet abandons an
            # instance it is still paying for.
            raise CapacityProviderError(
                f"EC2 {action} did not complete: {error}",
                code=CapacityProviderFailureCode.PROVIDER_UNAVAILABLE,
            ) from error
        finally:
            if self.client is None:
                client.close()
        if response.status_code >= 400:
            raise failure_for(*parse_error(response.text))
        return response.text

    def run_instances(self, params: tp.Mapping[str, str]) -> list[Ec2Instance]:
        return parse_instances(self._call("RunInstances", params))

    def describe_instances(self, filters: tp.Mapping[str, str]) -> list[Ec2Instance]:
        params: dict[str, str] = {}
        for index, (name, value) in enumerate(sorted(filters.items()), start=1):
            params[f"Filter.{index}.Name"] = name
            params[f"Filter.{index}.Value.1"] = value
        return parse_instances(self._call("DescribeInstances", params))

    def terminate_instances(self, instance_ids: tp.Sequence[str]) -> None:
        params = {f"InstanceId.{index}": value for index, value in enumerate(instance_ids, start=1)}
        self._call("TerminateInstances", params)

    def create_tags(self, instance_ids: tp.Sequence[str], tags: tp.Mapping[str, str]) -> None:
        params = {f"ResourceId.{index}": value for index, value in enumerate(instance_ids, start=1)}
        for index, (key, value) in enumerate(sorted(tags.items()), start=1):
            params[f"Tag.{index}.Key"] = key
            params[f"Tag.{index}.Value"] = value
        self._call("CreateTags", params)

    def describe_spot_instance_requests(self, instance_ids: tp.Sequence[str]) -> dict[str, str]:
        # The action is keyed by SpotInstanceRequestId.N, which this seam does
        # not hold -- callers pass instance ids -- so the ids go as the action's
        # documented `instance-id` filter instead. Sent as a bare InstanceId.N
        # parameter they never reach EC2 at all: the name is not part of the
        # action, and the call degrades to describing every spot request in the
        # region, of which the answer is then parsed for the instances asked
        # about by accident.
        params: dict[str, str] = {}
        for index, instance_id in enumerate(instance_ids, start=1):
            params[f"Filter.{index}.Name"] = "instance-id"
            params[f"Filter.{index}.Value.1"] = instance_id
        return parse_spot_instance_requests(self._call("DescribeSpotInstanceRequests", params))


def _encoded_body(body: tp.Mapping[str, str]) -> str:
    """Encode the form body exactly as it will be sent.

    Built once so the signed bytes and the wire bytes cannot drift apart: this
    string is both what the payload digest hashes and what is posted. httpx
    sends a string body through verbatim, which is what makes that guarantee
    hold regardless of how httpx itself would have encoded a dict.
    """

    from urllib.parse import urlencode

    return urlencode(sorted(body.items()))


def _body_digest(encoded_body: str) -> str:
    import hashlib

    return hashlib.sha256(encoded_body.encode("utf-8")).hexdigest()


__all__ = (
    "API_VERSION",
    "CAPACITY_ERROR_CODES",
    "CONCLUSIVE_ERROR_CODES",
    "THROTTLING_ERROR_CODES",
    "Ec2Api",
    "Ec2Instance",
    "HttpEc2Api",
    "failure_for",
    "parse_error",
    "parse_instances",
)
