# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The EC2 driver, and the fake EC2 it is held against.

The second cloud, written against the boundary the first one was cleaned up to
expose. What is asserted here is not "AWS works" -- nothing in this file has run
against an AWS account, and saying otherwise would be the most expensive kind of
wrong. It is that the driver's *ownership* discipline holds: an instance is
adopted only when its tags prove which cell and which incarnation asked for it,
a launch that may or may not have happened is retried without doubling, and
release frees the substrate before it forgets the pointer to it.

The fake models the parts of EC2 that decide those questions: tags, the client
token's idempotency, instance states, and the two ways a call can fail.
"""

from __future__ import annotations

import base64
import typing as tp
from contextlib import contextmanager

import pytest

from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.aws import AwsEc2CapacityDriver
from eyvan.capacity.drivers.base import (
    CapacityProviderError,
    CapacityProviderFailureCode,
    MutationContext,
    StaleMutationError,
)
from eyvan.capacity.fleet_state import FleetStateStore
from eyvan.capacity.providers.aws.api import Ec2Instance, HttpEc2Api, failure_for, parse_error, parse_instances
from eyvan.capacity.providers.aws.instances import CELL_TAG, FENCE_TAG, GENERATION_TAG, client_token
from eyvan.capacity.providers.aws.signing import AwsCredentials


class FakeEc2:
    """The parts of EC2 that decide whether the driver is correct.

    Client-token idempotency is modelled because it is the mechanism the driver
    relies on to retry an inconclusive launch: a fake that launched a second
    instance for a repeated token would make the driver look wrong, and one that
    ignored the token entirely would make it look right for free.
    """

    def __init__(self) -> None:
        self.instances: dict[str, Ec2Instance] = {}
        self.launches: list[dict[str, str]] = []
        self.terminated: list[str] = []
        self._tokens: dict[str, str] = {}
        self._next = 1
        self._fail: Exception | None = None
        self.spot_statuses: dict[str, str] = {}

    # -- failure injection ----------------------------------------------------

    @contextmanager
    def failing(self, error: Exception) -> tp.Iterator[None]:
        self._fail = error
        try:
            yield
        finally:
            self._fail = None

    def _check(self) -> None:
        if self._fail is not None:
            raise self._fail

    # -- the four actions -----------------------------------------------------

    def run_instances(self, params: tp.Mapping[str, str]) -> list[Ec2Instance]:
        self._check()
        self.launches.append(dict(params))
        token = params.get("ClientToken", "")
        if token and token in self._tokens:
            # EC2's own idempotency: the same token inside its window returns the
            # original instance. Without this the driver's retry story is untested.
            return [self.instances[self._tokens[token]]]
        instance_id = f"i-{self._next:012x}"
        self._next += 1
        tags = {
            params[f"TagSpecification.1.Tag.{index}.Key"]: params[f"TagSpecification.1.Tag.{index}.Value"]
            for index in range(1, 9)
            if f"TagSpecification.1.Tag.{index}.Key" in params
        }
        instance = Ec2Instance(
            instance_id=instance_id,
            state="running",
            tags=tags,
            private_ip=f"10.0.0.{self._next}",
            instance_type=params.get("InstanceType"),
            availability_zone=params.get("Placement.AvailabilityZone"),
        )
        self.instances[instance_id] = instance
        if token:
            self._tokens[token] = instance_id
        return [instance]

    def describe_instances(self, filters: tp.Mapping[str, str]) -> list[Ec2Instance]:
        self._check()
        matched = list(self.instances.values())
        for name, value in filters.items():
            if name.startswith("tag:"):
                key = name.removeprefix("tag:")
                matched = [item for item in matched if item.tags.get(key) == value]
        return matched

    def terminate_instances(self, instance_ids: tp.Sequence[str]) -> None:
        self._check()
        for instance_id in instance_ids:
            self.terminated.append(instance_id)
            current = self.instances.get(instance_id)
            if current is not None:
                self.instances[instance_id] = Ec2Instance(
                    instance_id=current.instance_id,
                    state="terminated",
                    tags=current.tags,
                )

    def create_tags(self, instance_ids: tp.Sequence[str], tags: tp.Mapping[str, str]) -> None:
        self._check()
        for instance_id in instance_ids:
            current = self.instances[instance_id]
            self.instances[instance_id] = Ec2Instance(
                instance_id=current.instance_id,
                state=current.state,
                tags={**current.tags, **tags},
                private_ip=current.private_ip,
            )

    def describe_spot_instance_requests(self, instance_ids: tp.Sequence[str]) -> dict[str, str]:
        """Return the reclamation notices this fake has been told about."""

        self._check()
        return {
            instance_id: self.spot_statuses[instance_id]
            for instance_id in instance_ids
            if instance_id in self.spot_statuses
        }

    # -- what conformance counts ---------------------------------------------

    def live_for(self, cell_id: str) -> int:
        return sum(
            1
            for item in self.instances.values()
            if item.tags.get(CELL_TAG) == cell_id and item.state not in {"terminated", "shutting-down"}
        )


def desired(**overrides: object) -> dict[str, object]:
    document = {
        "capacity_backend": "aws-ec2",
        "executor": "eyvan-native",
        "capacity_class": "on_demand",
        "accelerator_type": "cpu",
        "region": "us-east-1",
        "zone": "us-east-1a",
        "image_id": "ami-0123456789abcdef0",
        "instance_type": "m6i.large",
        "subnet_id": "subnet-abc",
    }
    document.update(overrides)
    return document


def context(key: str, *, generation: int = 1, token: str = "fence-1") -> MutationContext:
    return MutationContext(idempotency_key=key, generation=generation, fencing_token=token)


@pytest.fixture
def driver(tmp_path) -> AwsEc2CapacityDriver:
    return AwsEc2CapacityDriver(
        FleetStateStore.from_uri(str(tmp_path / "state.json")),
        api=FakeEc2(),
    )


# --------------------------------------------------------------------------- launching


def test_a_launch_tags_the_instance_with_who_asked_for_it(driver) -> None:
    """EC2 offers nowhere else to record ownership, so the tags are the fence."""

    driver.apply_capacity("cell-a", desired(), context("apply-1"))

    instance = next(iter(driver.api.instances.values()))
    assert instance.tags[CELL_TAG] == "cell-a"
    assert instance.tags[GENERATION_TAG] == "1"
    assert instance.tags[FENCE_TAG] == "fence-1"


def test_tags_are_applied_in_the_launch_call_rather_than_afterwards(driver) -> None:
    """A separate CreateTags leaves a window holding an unrecognisable instance.

    That window is precisely when a crash is likeliest, because it is the same
    instant the durable record is being written.
    """

    driver.apply_capacity("cell-a", desired(), context("apply-1"))

    launch = driver.api.launches[0]
    assert launch["TagSpecification.1.ResourceType"] == "instance"
    assert any(launch[key] == "cell-a" for key in launch if key.endswith(".Value"))


def test_an_inconclusive_launch_is_retried_without_doubling(driver) -> None:
    """The failure mode this whole client-token derivation exists for.

    A retry that launched a second instance would double the bill silently, and
    one that gave up would abandon a machine nobody is watching.
    """

    driver.apply_capacity("cell-a", desired(), context("apply-1"))
    driver.apply_capacity("cell-a", desired(), context("apply-2"))

    assert driver.api.live_for("cell-a") == 1


def test_the_client_token_is_derived_from_the_fence_rather_than_generated(driver) -> None:
    """Generated afresh, EC2's idempotency window would never match anything."""

    first = client_token("cell-a", context("k1"))
    same = client_token("cell-a", context("k2"))
    other_generation = client_token("cell-a", context("k1", generation=2))

    assert first == same, "the token identifies the incarnation, not the call"
    assert first != other_generation, "a new incarnation must not adopt the previous one's instance"
    assert len(first) <= 64, "EC2 bounds the token at 64 characters"


def test_a_spot_cell_asks_for_a_one_time_request(driver) -> None:
    """A persistent request re-launches behind Eyvan's back.

    The fleet would then be paying for a machine no cell owns and no scheduler
    can place work on -- the exact shape of the leak this driver is built to
    prevent, arriving from the provider's own convenience feature.
    """

    driver.apply_capacity("cell-a", desired(capacity_class="spot"), context("apply-1"))

    launch = driver.api.launches[0]
    assert launch["InstanceMarketOptions.MarketType"] == "spot"
    assert launch["InstanceMarketOptions.SpotOptions.SpotInstanceType"] == "one-time"


def test_user_data_reaches_ec2_base64_encoded(driver) -> None:
    """The Query API decodes UserData itself: sent raw, a cloud-init script
    boots as garbage while the instance still reads running."""

    script = "#!/bin/sh\necho booted > /var/eyvan/booted\n"
    driver.apply_capacity("cell-a", desired(user_data=script), context("apply-1"))

    wire = driver.api.launches[0]["UserData"]
    assert wire != script, "raw text on the wire is what EC2 decodes to garbage"
    assert base64.b64decode(wire, validate=True).decode("utf-8") == script


# --------------------------------------------------------------------------- ownership


def test_an_instance_from_a_previous_generation_is_not_adopted(driver) -> None:
    """Adopting one is how two incarnations end up sharing a machine."""

    driver.api.instances["i-leaked"] = Ec2Instance(
        instance_id="i-leaked",
        state="running",
        tags={CELL_TAG: "cell-a", GENERATION_TAG: "1", FENCE_TAG: "fence-1"},
        private_ip="10.9.9.9",
    )

    observation = driver.apply_capacity("cell-a", desired(), context("apply-1", generation=2, token="fence-2"))

    assert observation.metadata["instance_id"] != "i-leaked"


def test_another_owners_instance_is_not_adopted(driver) -> None:
    """Two replicas both believing they are leader produce the same generation."""

    driver.api.instances["i-other"] = Ec2Instance(
        instance_id="i-other",
        state="running",
        tags={CELL_TAG: "cell-a", GENERATION_TAG: "1", FENCE_TAG: "somebody-else"},
    )

    observation = driver.apply_capacity("cell-a", desired(), context("apply-1"))

    assert observation.metadata["instance_id"] != "i-other"


def test_a_cell_whose_instance_vanished_waits_rather_than_reporting_health(driver) -> None:
    """Publishing an endpoint for capacity that is not there is the failure."""

    driver.apply_capacity("cell-a", desired(), context("apply-1"))
    driver.api.instances.clear()

    observation = driver.observe("cell-a")

    assert observation.state is CapacityStatus.WAITING
    assert observation.execution_endpoint is None


def test_a_spot_reclamation_is_reported_as_a_preemption(driver) -> None:
    """Mapped onto the outcome the fleet already reasons about.

    The preemption budget, the retry accounting and the belief model then work
    unchanged, which is most of the value of building against the boundary
    instead of beside it.
    """

    driver.apply_capacity("cell-a", desired(capacity_class="spot"), context("apply-1"))
    instance_id = next(iter(driver.api.instances))
    current = driver.api.instances[instance_id]
    driver.api.instances[instance_id] = Ec2Instance(
        instance_id=current.instance_id,
        state="running",
        tags=current.tags,
        private_ip=current.private_ip,
        spot_interruption="instance-terminated-by-price",
    )

    observation = driver.observe("cell-a")

    assert observation.metadata["preemption_reason"] == "instance-terminated-by-price"


# --------------------------------------------------------------------------- release


def test_release_terminates_the_instance_before_forgetting_it(driver) -> None:
    """The other order abandons a billing machine if the process dies between."""

    driver.apply_capacity("cell-a", desired(), context("apply-1"))

    driver.release("cell-a", context("release-1"))

    assert driver.api.live_for("cell-a") == 0
    assert driver.state.get("cell-a") is None


def test_release_is_idempotent(driver) -> None:
    driver.apply_capacity("cell-a", desired(), context("apply-1"))
    driver.release("cell-a", context("release-1"))

    observation = driver.release("cell-a", context("release-2"))

    assert observation.state is CapacityStatus.RELEASED


def test_a_stale_generation_cannot_release(driver) -> None:
    driver.apply_capacity("cell-a", desired(), context("apply-1", generation=3, token="fence-3"))

    with pytest.raises(StaleMutationError):
        driver.release("cell-a", context("release-1", generation=2, token="fence-2"))

    assert driver.api.live_for("cell-a") == 1


# --------------------------------------------------------------------------- refusing a bad document


@pytest.mark.parametrize(
    ("field", "value"),
    [("image_id", ""), ("instance_type", ""), ("region", ""), ("capacity_class", "flex_start")],
)
def test_a_document_that_cannot_launch_is_refused_by_name(driver, field: str, value: str) -> None:
    """An operator reading their own fleet YAML beats one reading an AWS code."""

    with pytest.raises(ValueError, match=field.split("_")[0]):
        driver.apply_capacity("cell-a", desired(**{field: value}), context("apply-1"))


def test_the_driver_refuses_to_exist_without_a_client() -> None:
    """A driver that built its own client would read credentials at import."""

    with pytest.raises(ValueError, match="requires an EC2 API client"):
        AwsEc2CapacityDriver()


# --------------------------------------------------------------------------- reading EC2's answers


def test_an_error_code_nobody_has_reasoned_about_is_inconclusive() -> None:
    """The safe default: concluding "broken" strands a running instance."""

    assert failure_for("SomethingNewInvented", "x").code is CapacityProviderFailureCode.PROVIDER_UNAVAILABLE


def test_a_stockout_is_not_a_configuration_error() -> None:
    """One says the zone is empty and the other says the request is wrong."""

    assert failure_for("InsufficientInstanceCapacity", "x").code is CapacityProviderFailureCode.STOCKOUT
    assert failure_for("InvalidAMIID.NotFound", "x").code is CapacityProviderFailureCode.CONFIGURATION


def test_throttling_is_not_a_stockout() -> None:
    """A fleet that confuses them backs off from capacity that is there."""

    assert failure_for("RequestLimitExceeded", "x").code is CapacityProviderFailureCode.PROVIDER_UNAVAILABLE


def test_instances_parse_out_of_a_real_shaped_response() -> None:
    """Namespace-insensitive, because the namespace carries the API version."""

    document = """<?xml version="1.0"?>
    <DescribeInstancesResponse xmlns="http://ec2.amazonaws.com/doc/2016-11-15/">
      <reservationSet><item><instancesSet><item>
        <instanceId>i-abc</instanceId>
        <instanceState><code>16</code><name>running</name></instanceState>
        <privateIpAddress>10.0.0.4</privateIpAddress>
        <instanceType>m6i.large</instanceType>
        <placement><availabilityZone>us-east-1a</availabilityZone></placement>
        <tagSet><item><key>eyvan:cell</key><value>cell-a</value></item></tagSet>
      </item></instancesSet></item></reservationSet>
    </DescribeInstancesResponse>"""

    instances = parse_instances(document)

    assert len(instances) == 1
    assert instances[0].instance_id == "i-abc"
    assert instances[0].state == "running"
    assert instances[0].private_ip == "10.0.0.4"
    assert instances[0].availability_zone == "us-east-1a"
    assert instances[0].tags["eyvan:cell"] == "cell-a"


def test_an_unparseable_error_is_still_inconclusive() -> None:
    """ "AWS said something and we could not tell what" is not a conclusion."""

    code, message = parse_error("<html>502 Bad Gateway</html>")

    assert code == ""
    assert failure_for(code, message).code is CapacityProviderFailureCode.PROVIDER_UNAVAILABLE


def test_a_transport_failure_is_never_conclusive(driver) -> None:
    """The call may have launched something. Concluding otherwise abandons it."""

    with (
        driver.api.failing(
            CapacityProviderError("connection refused", code=CapacityProviderFailureCode.PROVIDER_UNAVAILABLE)
        ),
        pytest.raises(CapacityProviderError) as raised,
    ):
        driver.apply_capacity("cell-a", desired(), context("apply-1"))

    assert raised.value.code is CapacityProviderFailureCode.PROVIDER_UNAVAILABLE


def test_a_marked_spot_instance_is_reported_as_a_preemption(tmp_path) -> None:
    """A reclamation notice becomes the typed PREEMPTED outcome, not a vanishing."""

    api = FakeEc2()
    driver = AwsEc2CapacityDriver(FleetStateStore.from_uri(str(tmp_path / "state.json")), api=api)
    applied = driver.apply_capacity("cell-a", desired(), context("apply", generation=2, token="leader-2"))
    instance_id = applied.metadata["instance_id"]
    api.spot_statuses[instance_id] = "marked-for-termination"

    observation = driver.observe("cell-a")

    assert observation.failure_code is CapacityProviderFailureCode.PREEMPTED
    assert observation.metadata["preemption_reason"] == "marked-for-termination"


def test_an_unmarked_spot_instance_carries_no_preemption(tmp_path) -> None:
    api = FakeEc2()
    driver = AwsEc2CapacityDriver(FleetStateStore.from_uri(str(tmp_path / "state.json")), api=api)
    applied = driver.apply_capacity("cell-a", desired(), context("apply", generation=2, token="leader-2"))
    api.spot_statuses[applied.metadata["instance_id"]] = "fulfilled"

    observation = driver.observe("cell-a")

    assert "preemption_reason" not in observation.metadata


def test_spot_request_status_codes_are_parsed_from_the_query_response() -> None:
    from eyvan.capacity.providers.aws.api import parse_spot_instance_requests

    document = """<DescribeSpotInstanceRequestsResponse xmlns="http://ec2.amazonaws.com/doc/2016-11-15/">
      <spotInstanceRequestSet>
        <item>
          <instanceId>i-1</instanceId>
          <status><code>marked-for-termination</code></status>
        </item>
        <item>
          <instanceId>i-2</instanceId>
          <status><code>fulfilled</code></status>
        </item>
      </spotInstanceRequestSet>
    </DescribeSpotInstanceRequestsResponse>"""

    assert parse_spot_instance_requests(document) == {"i-1": "marked-for-termination", "i-2": "fulfilled"}


class _XmlResponse:
    """The two things the EC2 client reads off a response."""

    def __init__(self, text: str) -> None:
        self.status_code = 200
        self.text = text


class _RecordingClient:
    """The httpx seam HttpEc2Api posts through, holding the encoded body."""

    def __init__(self, document: str) -> None:
        self._document = document
        self.body: str | None = None

    def post(self, url: str, content: str | None = None, headers: tp.Mapping[str, str] | None = None) -> _XmlResponse:
        self.body = content
        return _XmlResponse(self._document)


def test_describe_spot_instance_requests_asks_about_the_instances_it_was_given() -> None:
    """DescribeSpotInstanceRequests has no InstanceId parameter.

    Sent anyway, the ids never reach EC2 and the call degrades to describing
    every spot request in the region. The action is keyed by spot request id,
    which this seam does not hold -- callers pass instance ids -- so the ids go
    as the action's documented ``instance-id`` filter instead.
    """

    document = """<DescribeSpotInstanceRequestsResponse xmlns="http://ec2.amazonaws.com/doc/2016-11-15/">
      <spotInstanceRequestSet><item>
        <instanceId>i-1</instanceId>
        <status><code>marked-for-termination</code></status>
      </item></spotInstanceRequestSet>
    </DescribeSpotInstanceRequestsResponse>"""
    client = _RecordingClient(document)
    api = HttpEc2Api(region="us-east-1", credentials=AwsCredentials("AKIDEXAMPLE", "example-secret"), client=client)

    statuses = api.describe_spot_instance_requests(["i-1"])

    assert statuses == {"i-1": "marked-for-termination"}
    body = client.body
    assert "Filter.1.Name=instance-id" in body
    assert "Filter.1.Value.1=i-1" in body
    assert "InstanceId.1=" not in body, "a bare InstanceId.N is not part of the action and is ignored"
