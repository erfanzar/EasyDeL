# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""EC2 capacity: one instance per cell, fenced by tags.

The second cloud. Everything here was written against the boundary the first one
was cleaned up to expose -- the six-method ``CapacityDriver`` protocol, the
backend capability table, the twelve-property conformance suite -- so that
"support AWS" is a driver and a capability entry rather than an edit to the
scheduler.

**Ownership lives in tags**, because EC2 offers nowhere else to put it. That is
the entire fencing story on this provider: an instance carries the cell it
belongs to, the generation that asked for it, and the fencing token of the
mutation that created it, and an instance failing to match all three is a leak to
terminate rather than capacity to adopt. Adopting one is how two generations end
up sharing a machine, and how a control plane that crashed mid-launch comes back
believing somebody else's instance is its own.

**A launch is retried with the same client token.** EC2 keys idempotency on it,
so a retry after a call that may or may not have launched something returns the
original instance rather than a second one. That is what makes an inconclusive
``RunInstances`` recoverable instead of a coin flip between abandoning capacity
and doubling it.

**A spot reclamation is a preemption, not a disappearance.** It maps onto the
``PREEMPTED`` outcome the fleet already reasons about, so the preemption budget,
the retry accounting and the belief model work unchanged -- which is most of the
value of doing this against an existing boundary rather than beside it.

**Verified against a fake EC2, not against AWS.** The signing is checked against
AWS's own published vectors, and everything in this file is checked against a
substrate that behaves the way the documented Query API says EC2 does. Nothing
here has been run against a real account, and until it has, "supported" means it
passes conformance -- which is a stronger claim than most providers ship with
and a weaker one than having run.
"""

from __future__ import annotations

import typing as tp
from dataclasses import replace
from datetime import datetime
from threading import RLock

from eyvan._internal.compat import UTC
from eyvan._internal.json_types import JsonObject, JsonValue, is_json_object
from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.base import (
    CapacityProviderError,
    CapacityProviderFailureCode,
    CellCapabilities,
    CellObservation,
    MutationContext,
    StaleMutationError,
)
from eyvan.capacity.fleet_state import FleetCellNotRegisteredError, FleetCellRecord, FleetStateStore, StaleFenceError
from eyvan.capacity.providers.aws.api import Ec2Api, Ec2Instance
from eyvan.capacity.providers.aws.instances import (
    CELL_TAG,
    FENCE_TAG,
    GENERATION_TAG,
    client_token,
    is_live,
    launch_parameters,
    observation_metadata,
    owns,
)

#: Where the instance id of the current incarnation is remembered. Kept in the
#: durable record so release can terminate precisely what this cell created,
#: rather than everything a tag search happens to return.
_INSTANCE_KEY = "aws_instance_id"


class AwsEc2CapacityDriver:
    """Provision one EC2 instance per Eyvan capacity cell."""

    provider_name = "aws-ec2"
    execution_backend = "eyvan-native"
    accelerator_types = ("cpu", "gpu")
    #: No flex-start: AWS sells a bounded window as a Capacity Block, which is a
    #: reservation bought in advance rather than a queued request, so mapping it
    #: onto that class would misdescribe what an operator gets.
    capacity_classes = ("on-demand", "spot", "reserved")

    def __init__(self, state: FleetStateStore | None = None, *, api: Ec2Api | None = None) -> None:
        if api is None:  # pragma: no cover - constructed with an api in every test
            raise ValueError(
                "the AWS driver requires an EC2 API client; build one with HttpEc2Api(region=..., credentials=...)"
            )
        self.state = state or FleetStateStore.from_uri(None)
        self.api = api
        self._mutation_lock = RLock()

    def capabilities(self) -> CellCapabilities:
        """Describe EC2-backed capacity.

        ``resource_enforcement`` is ``"none"``: an instance is a whole machine
        and Eyvan does not put a cgroup around the workload on it. Claiming
        ``"backend"`` would tell the scheduler the substrate bounds a job's
        memory, which nothing here does.
        """

        return CellCapabilities(
            provider=self.provider_name,
            executor=self.execution_backend,
            accelerator_types=self.accelerator_types,
            capacity_classes=self.capacity_classes,
            capacity_backends=("aws-ec2",),
            supports_callables=True,
            supports_checkpoint=True,
            # One instance per cell for now. Placement groups would make a
            # multi-machine cell possible; claiming it before the driver builds
            # one would let the durable model accept a gang nothing can allocate.
            max_gang_replicas=1,
            supports_exec=False,
            resource_enforcement="none",
        )

    # ---- durable record helpers, shared in shape with the other drivers -----

    @staticmethod
    def _accept_mutation(record: FleetCellRecord, context: MutationContext) -> bool:
        if context.generation < record.generation:
            raise StaleMutationError(
                f"generation {context.generation} is older than cell {record.name!r} generation {record.generation}"
            )
        if (
            context.generation == record.generation
            and record.fencing_token is not None
            and record.fencing_token != context.fencing_token
        ):
            raise StaleMutationError(f"fencing token does not own cell {record.name!r} generation {record.generation}")
        return context.idempotency_key not in record.idempotency_keys

    @staticmethod
    def _stamp(record: FleetCellRecord, context: MutationContext) -> None:
        record.generation = context.generation
        record.fencing_token = context.fencing_token
        record.idempotency_keys = [*record.idempotency_keys, context.idempotency_key][-64:]

    @staticmethod
    def _required_string(desired: tp.Mapping[str, JsonValue], name: str) -> str:
        value = desired.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"aws capacity field {name!r} must be a non-empty string")
        return value.strip()

    @staticmethod
    def _options(desired: tp.Mapping[str, JsonValue]) -> JsonObject:
        for key in ("provider_config", "provider_options"):
            value = desired.get(key)
            if is_json_object(value):
                return value
        return {}

    def _validate_desired(self, desired: tp.Mapping[str, JsonValue]) -> JsonObject:
        """Return the flattened launch document, or refuse it conclusively.

        Refusing here rather than at EC2 is the difference between an operator
        reading their own fleet YAML and reading an AWS error code: both are
        failures, and only one of them names the field.
        """

        backend = self._required_string(desired, "capacity_backend").replace("_", "-")
        if backend not in {"aws", "aws-ec2", "ec2"}:
            raise ValueError(f"aws capacity requires capacity_backend 'aws-ec2', got {backend!r}")
        executor = self._required_string(desired, "executor")
        if executor != self.execution_backend:
            raise ValueError(f"aws capacity requires executor {self.execution_backend!r}, got {executor!r}")
        capacity_class = self._required_string(desired, "capacity_class").replace("_", "-")
        if capacity_class not in self.capacity_classes:
            choices = ", ".join(self.capacity_classes)
            raise ValueError(f"unsupported aws capacity class {capacity_class!r}; choose one of {choices}")

        options = self._options(desired)
        flattened: JsonObject = {**options, **{key: value for key, value in desired.items() if key not in options}}
        flattened["capacity_class"] = capacity_class.replace("-", "_")
        for name in ("image_id", "instance_type", "region"):
            if not isinstance(flattened.get(name), str) or not str(flattened[name]).strip():
                raise ValueError(f"aws capacity field {name!r} must be a non-empty string")
        return flattened

    # ---- observation --------------------------------------------------------

    def observe(self, cell_id: str) -> CellObservation:
        """Return the cell's live state, reconciled against the durable record."""

        record = self.state.get(cell_id)
        if record is None:
            return CellObservation(
                cell_id=cell_id,
                state=CapacityStatus.ABSENT,
                generation=0,
                observed_at=datetime.now(UTC),
            )
        instance = self._current_instance(record)
        if self._disowned_instances(record):
            # Hardware carrying this cell's name that the fence disowns. The
            # controller has to be told to *replace* the cell rather than
            # schedule onto it: reporting healthy here would send work to a
            # machine of a generation nobody is tracking, and the endpoint would
            # be published beside it.
            return CellObservation(
                cell_id=record.name,
                state=CapacityStatus.NEEDS_RECREATE,
                generation=record.generation,
                execution_endpoint=None,
                observed_at=datetime.now(UTC),
                metadata={
                    "provider": self.provider_name,
                    "capacity_backend": self.provider_name,
                    "resource_id": f"aws:{record.name}",
                    "detail": "EC2 holds instances tagged for this cell that its fencing token disowns",
                },
            )
        return self._observation(record, instance)

    def _disowned_instances(self, record: FleetCellRecord) -> list[Ec2Instance]:
        """Return live instances tagged for this cell that its fence disowns.

        Counted rather than ignored. An instance whose generation or fencing
        token is not this cell's is either a previous incarnation nobody cleaned
        up or another replica's -- and in both cases it costs money, holds
        hardware, and must not be scheduled onto.
        """

        disowned = []
        for instance in self.api.describe_instances({f"tag:{CELL_TAG}": record.name}):
            if not is_live(instance):
                continue
            same_generation = instance.tags.get(GENERATION_TAG) == str(record.generation)
            same_fence = record.fencing_token is None or instance.tags.get(FENCE_TAG) == record.fencing_token
            if not (same_generation and same_fence):
                disowned.append(instance)
        return disowned

    def _current_instance(self, record: FleetCellRecord) -> Ec2Instance | None:
        """Return the instance this incarnation owns, if EC2 still has one.

        Searched by tag rather than only by remembered id: a launch that returned
        inconclusively may have created an instance whose id was never written
        down, and the tag is the only way back to it.
        """

        instances = self.api.describe_instances({f"tag:{CELL_TAG}": record.name})
        owned = [
            instance
            for instance in instances
            if is_live(instance)
            and instance.tags.get("eyvan:generation") == str(record.generation)
            and (record.fencing_token is None or instance.tags.get("eyvan:fence") == record.fencing_token)
        ]
        if not owned:
            return None
        remembered = record.metadata.get(_INSTANCE_KEY)
        for instance in owned:
            if instance.instance_id == remembered:
                return self._with_spot_interruption(instance)
        return self._with_spot_interruption(owned[0])

    def _with_spot_interruption(self, instance: Ec2Instance) -> Ec2Instance:
        """Attach EC2's reclamation notice to the instance, when one exists.

        DescribeInstances never carries the signal: it lives on the spot
        request, and only ``DescribeSpotInstanceRequests`` answers it. A
        provider that cannot ask (an older fake, a narrower seam) changes
        nothing; a marked instance comes back carrying the reason, which is
        what makes the reclamation reportable as a preemption rather than as
        a machine that vanished.
        """

        ask = getattr(self.api, "describe_spot_instance_requests", None)
        if not callable(ask):
            return instance
        statuses = ask([instance.instance_id])
        code = statuses.get(instance.instance_id) if isinstance(statuses, dict) else None
        if isinstance(code, str) and code.startswith("marked-for-"):
            return replace(instance, spot_interruption=code)
        return instance

    def _observation(self, record: FleetCellRecord, instance: Ec2Instance | None) -> CellObservation:
        """Project the record and its instance into one observation."""

        metadata: JsonObject = {
            "provider": self.provider_name,
            "capacity_backend": self.provider_name,
            "resource_id": f"aws:{record.name}",
            "desired_state": record.desired_state,
            "provider_generation": record.provider_generation,
            **record.metadata,
        }
        state = record.state
        endpoint = record.execution_endpoint
        failure_code: CapacityProviderFailureCode | None = None
        if record.desired_state != "up":
            # Drain has already decided. A running instance is exactly what a
            # draining cell has -- deriving the state from it here would report
            # the cell healthy again on the next observation and undo the drain.
            return CellObservation(
                cell_id=record.name,
                state=state,
                generation=record.generation,
                execution_endpoint=None,
                observed_at=datetime.now(UTC),
                runtime_process_count=1,
                metadata=metadata,
            )
        if instance is not None:
            metadata.update(observation_metadata(instance))
            if instance.spot_interruption:
                # A reclamation, not a machine that vanished. Reported so the
                # fleet charges it to the preemption budget rather than to the
                # one that decides a pool cannot give capacity at all -- and
                # carried as the typed PREEMPTED failure code, not just prose,
                # because routing reads the code and humans read the text.
                metadata["preemption_reason"] = instance.spot_interruption
            if instance.state == "running":
                state = CapacityStatus.HEALTHY
                endpoint = endpoint or instance.private_ip
            elif instance.state == "pending":
                state = CapacityStatus.WAITING
            if instance.spot_interruption:
                failure_code = CapacityProviderFailureCode.PREEMPTED
        elif record.desired_state == "up" and record.state is not CapacityStatus.ABSENT:
            # The record wants a machine and EC2 has none carrying its tags.
            # Waiting rather than healthy: publishing an endpoint for capacity
            # that is not there is the failure this whole ownership check exists
            # to prevent.
            state = CapacityStatus.WAITING
            endpoint = None

        return CellObservation(
            cell_id=record.name,
            state=state,
            generation=record.generation,
            execution_endpoint=endpoint,
            observed_at=datetime.now(UTC),
            runtime_process_count=1,
            failure_code=failure_code,
            metadata=metadata,
        )

    # ---- mutations ----------------------------------------------------------

    def apply_capacity(
        self,
        cell_id: str,
        desired: tp.Mapping[str, JsonValue],
        context: MutationContext,
    ) -> CellObservation:
        """Launch this cell's instance, or adopt the one this incarnation owns."""

        document = self._validate_desired(desired)
        with self._mutation_lock, self.state.exclusive():

            def claim(record: FleetCellRecord) -> None:
                if not self._accept_mutation(record, context):
                    return
                record.desired_state = "up"
                record.zone = str(document.get("zone") or "") or None
                record.accelerator_type = str(desired.get("accelerator_type") or "cpu")
                record.capacity_class = str(document["capacity_class"]).replace("_", "-")
                record.metadata["capacity_backend"] = self.provider_name
                record.metadata["region"] = str(document["region"])
                self._stamp(record, context)

            try:
                record = self.state.mutate_or_create_record(cell_id, claim, generation=context.generation)
            except StaleFenceError as exc:
                raise StaleMutationError(str(exc)) from exc

        existing = self._current_instance(record)
        if existing is None:
            # The same token on every retry, so an inconclusive launch returns
            # the instance it may already have created rather than a second one.
            launched = self.api.run_instances(launch_parameters(cell_id, document, context))
            existing = next((instance for instance in launched if owns(instance, cell_id, context)), None)
            if existing is None:
                raise CapacityProviderError(
                    f"EC2 accepted the launch for {cell_id!r} but returned no instance carrying its ownership tags; "
                    "the next reconcile will look for one rather than launching a second",
                    code=CapacityProviderFailureCode.PROVIDER_UNAVAILABLE,
                )

        with self._mutation_lock, self.state.exclusive():

            def record_instance(record: FleetCellRecord) -> None:
                record.metadata[_INSTANCE_KEY] = existing.instance_id
                record.provider_generation = max(record.provider_generation, context.generation)
                record.state = CapacityStatus.HEALTHY if existing.state == "running" else CapacityStatus.WAITING
                if existing.state == "running":
                    record.execution_endpoint = existing.private_ip

            try:
                record = self.state.mutate_record(cell_id, record_instance, generation=context.generation)
            except StaleFenceError as exc:
                raise StaleMutationError(str(exc)) from exc
        return self._observation(record, existing)

    def reconcile(self, cell_id: str, context: MutationContext) -> CellObservation:
        """Re-read EC2 and keep the durable record agreeing with it."""

        record = self.state.get(cell_id)
        if record is None:
            return self.observe(cell_id)
        instance = self._current_instance(record)
        with self._mutation_lock, self.state.exclusive():

            def mutate(current: FleetCellRecord) -> None:
                if not self._accept_mutation(current, context):
                    return
                if instance is None:
                    current.execution_endpoint = None
                    current.metadata.pop(_INSTANCE_KEY, None)
                    current.state = CapacityStatus.WAITING if current.desired_state == "up" else current.state
                else:
                    current.metadata[_INSTANCE_KEY] = instance.instance_id
                    if instance.state == "running":
                        current.state = CapacityStatus.HEALTHY
                        current.execution_endpoint = instance.private_ip
                self._stamp(current, context)

            try:
                record = self.state.mutate_record(cell_id, mutate, generation=context.generation)
            except FleetCellNotRegisteredError:
                return self.observe(cell_id)
            except StaleFenceError as exc:
                raise StaleMutationError(str(exc)) from exc
        return self._observation(record, instance)

    def drain(self, cell_id: str, context: MutationContext) -> CellObservation:
        """Stop new placement while leaving the instance for release to terminate."""

        with self._mutation_lock, self.state.exclusive():

            def mutate(record: FleetCellRecord) -> None:
                record.desired_state = "down"
                record.state = CapacityStatus.DRAINING
                self._stamp(record, context)

            try:
                record = self.state.drain_record(cell_id, context.generation, context.fencing_token, mutate)
            except StaleFenceError as exc:
                raise StaleMutationError(str(exc)) from exc
        if record is None:
            return CellObservation(
                cell_id=cell_id,
                state=CapacityStatus.DRAINING,
                generation=context.generation,
                observed_at=datetime.now(UTC),
            )
        return self._observation(record, self._current_instance(record))

    def release(self, cell_id: str, context: MutationContext) -> CellObservation:
        """Terminate this cell's instance and forget it.

        The substrate is freed *before* the record is tombstoned. The other order
        loses the only pointer to a running instance if the process dies between
        the two -- and an EC2 instance nothing points at bills until somebody
        finds it by hand.
        """

        record = self.state.get(cell_id)
        if record is not None:
            # The fence is checked *before* anything is terminated. Freeing the
            # substrate first is right against the crash -- an instance nothing
            # points at bills until somebody finds it by hand -- but doing it
            # before establishing ownership lets a stale replica destroy a live
            # cell that a newer generation is using.
            if context.generation < record.generation or (
                context.generation == record.generation
                and record.fencing_token is not None
                and record.fencing_token != context.fencing_token
            ):
                raise StaleMutationError(
                    f"generation {context.generation} does not own cell {cell_id!r} "
                    f"generation {record.generation}; refusing to terminate its instances"
                )
            for instance in self.api.describe_instances({f"tag:{CELL_TAG}": cell_id}):
                if is_live(instance):
                    self.api.terminate_instances([instance.instance_id])

        with self._mutation_lock, self.state.exclusive():

            def owned(current: FleetCellRecord) -> bool:
                if self._accept_mutation(current, context):
                    self._stamp(current, context)
                if current.generation != context.generation or current.fencing_token != context.fencing_token:
                    raise StaleMutationError(f"cell {cell_id!r} changed ownership before release")
                return True

            try:
                self.state.tombstone_record(cell_id, context.generation, context.fencing_token, owned)
            except StaleFenceError as exc:
                raise StaleMutationError(str(exc)) from exc
        return CellObservation(
            cell_id=cell_id,
            state=CapacityStatus.RELEASED,
            generation=context.generation,
            observed_at=datetime.now(UTC),
        )

    def _client_token(self, cell_id: str, context: MutationContext) -> str:
        """Expose the derived idempotency token for diagnostics and tests."""

        return client_token(cell_id, context)


__all__ = ("AwsEc2CapacityDriver",)
