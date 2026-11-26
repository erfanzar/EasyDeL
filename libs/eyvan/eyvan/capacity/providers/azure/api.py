# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Talking to Azure Resource Manager, and telling its two kinds of failure apart.

**No new dependency**, for the reason the Kubernetes and EC2 transports give:
ARM is JSON over HTTPS with a bearer token, ``httpx`` is already core, and an SDK
would bring an object model to keep in agreement with the four calls Eyvan makes.

Azure differs from EC2 in one way that simplifies everything here: **a virtual
machine is addressed by name**. A ``PUT`` to the same path with the same body is
the same machine, so the resource name *is* the idempotency key and there is no
client token to derive. That removes the entire class of "did my launch happen"
ambiguity EC2 needs a token to answer -- a retried create either returns the
machine that exists or creates the one that does not.

What does not simplify is the failure classification, which is the same problem
everywhere: a call that conclusively failed and a call whose outcome is unknown
look alike, and treating them alike either abandons a running VM or builds a
second one. ARM at least says so in the status code, so the split here is by
status rather than by a code table -- 4xx is the request, 5xx and a transport
failure are not.

Long-running operations are accepted asynchronously: a create returns ``201`` and
the VM reaches ``Succeeded`` later. The driver therefore reads provisioning state
rather than assuming the response body describes a finished machine.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field

import httpx

from eyvan._internal.json_types import JsonObject, is_json_object
from eyvan.capacity.drivers.base import CapacityProviderError, CapacityProviderFailureCode

#: ARM API version this client speaks. Pinned for the reason EC2's is: letting
#: the service choose makes a response-shape change arrive as a parse failure
#: nobody can date.
API_VERSION = "2024-07-01"

#: How long one ARM call may take. This runs on the leader.
DEFAULT_TIMEOUT_S = 30.0

#: ARM error codes that mean the request is wrong and will stay wrong, whatever
#: the status code says. Checked as well as the status because ARM reports some
#: conclusive failures as 409.
CONCLUSIVE_ERROR_CODES = frozenset(
    {
        "AuthorizationFailed",
        "InvalidParameter",
        "InvalidResourceReference",
        "InvalidTemplateDeployment",
        "MissingSubscriptionRegistration",
        "ResourceGroupNotFound",
        "SkuNotAvailable",
        "SubscriptionNotFound",
    }
)

#: Codes that mean the region has no capacity of this size right now. Their own
#: outcome, because a zone that is full and a request that is wrong call for
#: entirely different backoffs.
CAPACITY_ERROR_CODES = frozenset(
    {
        "AllocationFailed",
        "OverconstrainedAllocationRequest",
        "ZonalAllocationFailed",
    }
)

#: Provisioning states in which a VM still exists and still costs money.
LIVE_PROVISIONING_STATES = frozenset({"Creating", "Updating", "Succeeded", "Failed"})


@dataclass(frozen=True, slots=True)
class AzureVirtualMachine:
    """One VM, as much of it as a capacity driver needs."""

    name: str
    provisioning_state: str
    tags: dict[str, str] = field(default_factory=dict)
    private_ip: str | None = None
    #: Resource ids of the VM's network interfaces. Real ARM documents never
    #: embed addresses here -- they carry only references -- so a driver that
    #: needs an address resolves one of these.
    nic_ids: tuple[str, ...] = ()
    vm_size: str | None = None
    zone: str | None = None
    #: Set when Azure has deallocated a Spot VM. The signal that separates a
    #: reclamation from a machine somebody deleted, which are charged to
    #: different budgets and mean opposite things about retrying.
    eviction_reason: str | None = None


class AzureApi(tp.Protocol):
    """The four ARM calls a capacity driver needs, and nothing else."""

    def create_or_update_vm(self, name: str, body: JsonObject) -> AzureVirtualMachine: ...

    def get_vm(self, name: str) -> AzureVirtualMachine | None: ...

    def list_vms(self) -> list[AzureVirtualMachine]: ...

    def delete_vm(self, name: str) -> None: ...

    def get_network_interface(self, nic_id: str) -> JsonObject | None: ...


def parse_vm(payload: JsonObject) -> AzureVirtualMachine:
    """Read one VM out of an ARM resource document."""

    properties = payload.get("properties") if is_json_object(payload.get("properties")) else {}
    hardware = properties.get("hardwareProfile") if is_json_object(properties.get("hardwareProfile")) else {}
    zones = payload.get("zones")
    tags = payload.get("tags") if is_json_object(payload.get("tags")) else {}
    return AzureVirtualMachine(
        name=str(payload.get("name") or ""),
        provisioning_state=str(properties.get("provisioningState") or "Unknown"),
        tags={str(key): str(value) for key, value in tags.items()},
        private_ip=_private_ip(properties),
        nic_ids=_nic_ids(properties),
        vm_size=str(hardware.get("vmSize")) if hardware.get("vmSize") else None,
        zone=str(zones[0]) if isinstance(zones, list) and zones else None,
        eviction_reason=_eviction_reason(properties),
    )


def _private_ip(properties: tp.Mapping[str, tp.Any]) -> str | None:
    """Return the VM's first private address, when the document carries one.

    ARM does not put the address on the VM: it lives on a network interface,
    which is a separate resource. A document that carries it has been assembled
    by something that already looked, so this reads it rather than issuing a
    second call from inside a parser; real ARM documents carry only interface
    *references*, which ``_nic_ids`` collects for exactly that follow-up.
    """

    network = properties.get("networkProfile")
    if not is_json_object(network):
        return None
    interfaces = network.get("networkInterfaces")
    if not isinstance(interfaces, list):
        return None
    for interface in interfaces:
        if not is_json_object(interface):
            continue
        address = interface.get("privateIPAddress")
        if isinstance(address, str) and address:
            return address
    return None


def _nic_ids(properties: tp.Mapping[str, tp.Any]) -> tuple[str, ...]:
    """Return the resource ids of the VM's network interfaces, in order."""

    network = properties.get("networkProfile")
    if not is_json_object(network):
        return ()
    interfaces = network.get("networkInterfaces")
    if not isinstance(interfaces, list):
        return ()
    ids = [
        str(interface["id"])
        for interface in interfaces
        if isinstance(interface, dict) and isinstance(interface.get("id"), str) and interface["id"]
    ]
    return tuple(ids)


def parse_nic_private_ip(payload: JsonObject | None) -> str | None:
    """Read the private address out of one ARM network-interface document.

    The address lives at ``properties.ipConfigurations[*].properties.privateIPAddress``;
    anything else (a different shape, a missing configuration) yields ``None``
    rather than a guess.
    """

    if not is_json_object(payload):
        return None
    properties = payload.get("properties") if is_json_object(payload.get("properties")) else {}
    configurations = properties.get("ipConfigurations")
    if not isinstance(configurations, list):
        return None
    for configuration in configurations:
        if not is_json_object(configuration):
            continue
        inner = configuration.get("properties") if is_json_object(configuration.get("properties")) else {}
        address = inner.get("privateIPAddress")
        if isinstance(address, str) and address:
            return address
    return None


def _eviction_reason(properties: tp.Mapping[str, tp.Any]) -> str | None:
    """Return why a Spot VM was reclaimed, if it was.

    Read from the instance view's power state rather than inferred from the VM
    being absent: a deleted VM and an evicted one are both gone, and only one of
    them is a preemption.
    """

    view = properties.get("instanceView")
    if not is_json_object(view):
        return None
    for status in view.get("statuses") or []:
        if not is_json_object(status):
            continue
        code = str(status.get("code") or "")
        if code == "PowerState/deallocated" and properties.get("priority") == "Spot":
            return str(status.get("displayStatus") or "evicted")
    return None


def failure_for(status_code: int, payload: JsonObject | str) -> CapacityProviderError:
    """Classify one ARM failure into the outcome the fleet reasons about.

    A 5xx is inconclusive by construction: ARM accepted the request and
    something downstream failed, and the resource may exist. A 4xx is the
    request, except for the capacity codes, which say something about the region
    rather than about what was asked for.
    """

    code, message = _error_fields(payload)
    if code in CAPACITY_ERROR_CODES:
        return CapacityProviderError(
            f"Azure has no capacity for this request ({code}): {message}",
            code=CapacityProviderFailureCode.STOCKOUT,
        )
    if status_code == 429:
        return CapacityProviderError(
            f"Azure is throttling this subscription: {message}",
            code=CapacityProviderFailureCode.PROVIDER_UNAVAILABLE,
        )
    if code in CONCLUSIVE_ERROR_CODES or 400 <= status_code < 500:
        return CapacityProviderError(
            f"Azure refused this request ({code or status_code}): {message}",
            code=CapacityProviderFailureCode.CONFIGURATION,
        )
    return CapacityProviderError(
        f"Azure call did not conclude ({code or status_code}): {message}",
        code=CapacityProviderFailureCode.PROVIDER_UNAVAILABLE,
    )


def _error_fields(payload: JsonObject | str) -> tuple[str, str]:
    if isinstance(payload, str):
        return "", payload[:200]
    error = payload.get("error")
    if is_json_object(error):
        return str(error.get("code") or ""), str(error.get("message") or "")[:200]
    return "", str(payload)[:200]


@dataclass
class HttpAzureApi:
    """ARM over ``httpx``, scoped to one resource group.

    **Unverified against Azure.** Everything the tests cover is covered against a
    fake; this class is where the fake ends and a subscription begins. It is
    written to the documented REST surface and pinned to one API version so a
    disagreement shows up as a parse failure naming the response rather than as
    silently wrong capacity.
    """

    subscription_id: str
    resource_group: str
    location: str
    #: Supplied rather than fetched. A token provider is an auth concern -- a
    #: managed identity, a client credential, a federated token -- and a
    #: transport that acquired its own would make the choice for every operator.
    token: tp.Callable[[], str]
    endpoint: str = "https://management.azure.com"
    timeout_s: float = DEFAULT_TIMEOUT_S
    client: httpx.Client | None = None

    def _path(self, name: str | None = None) -> str:
        base = (
            f"{self.endpoint}/subscriptions/{self.subscription_id}"
            f"/resourceGroups/{self.resource_group}/providers/Microsoft.Compute/virtualMachines"
        )
        return base if name is None else f"{base}/{name}"

    def _call(
        self,
        method: str,
        url: str,
        body: JsonObject | None = None,
        *,
        params: tp.Mapping[str, str] | None = None,
    ) -> JsonObject | None:
        client = self.client or httpx.Client(timeout=self.timeout_s)
        try:
            response = client.request(
                method,
                url,
                params={"api-version": API_VERSION, **(params or {})},
                json=body,
                headers={"Authorization": f"Bearer {self.token()}", "Content-Type": "application/json"},
            )
        except httpx.HTTPError as error:
            # Inconclusive: the request may have reached ARM and created
            # something. Concluding otherwise abandons a VM that bills.
            raise CapacityProviderError(
                f"Azure {method} did not complete: {error}",
                code=CapacityProviderFailureCode.PROVIDER_UNAVAILABLE,
            ) from error
        finally:
            if self.client is None:
                client.close()
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            raise failure_for(response.status_code, _payload(response))
        return _payload(response) if response.content else None

    def create_or_update_vm(self, name: str, body: JsonObject) -> AzureVirtualMachine:
        payload = self._call("PUT", self._path(name), body)
        return parse_vm(payload or {"name": name, "properties": {"provisioningState": "Creating"}})

    def get_vm(self, name: str) -> AzureVirtualMachine | None:
        # ARM omits the instance view from a plain GET, and the view is the only
        # place a Spot eviction is reported: without $expand, _eviction_reason
        # can never fire on a real subscription no matter what it looks for.
        payload = self._call("GET", self._path(name), params={"$expand": "instanceView"})
        return None if payload is None else parse_vm(payload)

    def list_vms(self) -> list[AzureVirtualMachine]:
        payload = self._call("GET", self._path())
        if payload is None:
            return []
        values = payload.get("value")
        return [parse_vm(item) for item in values if is_json_object(item)] if isinstance(values, list) else []

    def delete_vm(self, name: str) -> None:
        self._call("DELETE", self._path(name))

    def get_network_interface(self, nic_id: str) -> JsonObject | None:
        # Interface ids are absolute resource paths; the api-version query
        # parameter is what ARM needs beside them.
        return self._call("GET", f"{self.endpoint}/{nic_id.lstrip('/')}")


def _payload(response: httpx.Response) -> JsonObject | str:
    try:
        decoded = response.json()
    except ValueError:
        return response.text
    return decoded if is_json_object(decoded) else response.text


__all__ = (
    "API_VERSION",
    "CAPACITY_ERROR_CODES",
    "CONCLUSIVE_ERROR_CODES",
    "LIVE_PROVISIONING_STATES",
    "AzureApi",
    "AzureVirtualMachine",
    "HttpAzureApi",
    "failure_for",
    "parse_vm",
)
