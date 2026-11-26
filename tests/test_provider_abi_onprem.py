# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""On-premises SSH capacity: the claim, the fence, and the marker script itself.

The fake fleet below models what the remote script does; the last section runs
the *real* script through ``/bin/sh`` so that model is not the only thing
tested. What no host test can cover is a real ``ssh`` binary against a real
host -- ``OpenSshCommandRunner`` is deliberately the thinnest possible seam for
exactly that reason.
"""

from __future__ import annotations

import subprocess
import typing as tp
from contextlib import contextmanager
from pathlib import Path

import pytest

from eyvan.capacity import CapacityStatus
from eyvan.capacity.conformance import ProviderConformanceHarness
from eyvan.capacity.drivers.base import (
    CapacityDriver,
    CapacityProviderError,
    CapacityProviderFailureCode,
    MutationContext,
    StaleMutationError,
)
from eyvan.capacity.drivers.onprem_ssh import (
    MARKER_SCRIPT,
    OnPremSshCapacityDriver,
    RemoteResult,
    RemoteTransportError,
    marker_fingerprint,
)
from eyvan.capacity.drivers.registry import DEFAULT_CAPACITY_BACKENDS
from eyvan.capacity.fleet_state import FleetStateStore, LocalBackend
from eyvan.config import EyvanSettings

HOSTS = ("eyvan@rack-a", "eyvan@rack-b")


def _context(key: str, generation: int = 1, token: str = "fence-1") -> MutationContext:
    return MutationContext(idempotency_key=key, generation=generation, fencing_token=token)


def _marker(cell: str, uid: str, generation: int, fence: str) -> str:
    return f"cell_id={cell}\nmanagement_uid={uid}\nprovider_generation={generation}\nfence={fence}\n"


def _field(payload: str, name: str) -> str:
    for line in payload.splitlines():
        key, separator, value = line.partition("=")
        if separator and key == name:
            return value
    return ""


class FakeSshFleet:
    """An in-memory rack that answers the marker protocol like a real host.

    An independent implementation of the same state machine as ``MARKER_SCRIPT``
    -- not a call into it -- so a change to one is caught by the script tests at
    the bottom of this module rather than mirrored silently into both.
    """

    def __init__(self) -> None:
        self.markers: dict[str, str] = {}
        self.lifecycle_commands: list[tuple[str, tuple[str, ...]]] = []
        self.unreachable = False
        self.refuse_claims = False
        self.lifecycle_exit_code = 0

    def run(
        self,
        host: str,
        argv: tp.Sequence[str],
        *,
        stdin: str = "",
        timeout_s: float,
    ) -> RemoteResult:
        """Answer one remote command, or fail the transport inconclusively."""

        del timeout_s
        if self.unreachable:
            raise RemoteTransportError(f"connection to {host} refused")
        if tuple(argv[:3]) != ("sh", "-s", "--"):
            self.lifecycle_commands.append((host, tuple(argv)))
            return RemoteResult(exit_code=self.lifecycle_exit_code, stdout="", stderr="")
        assert stdin == MARKER_SCRIPT, "the marker protocol must ship the audited script, not a built command"
        action, _path, cell, uid, generation, fence = argv[3:9]
        command = tuple(argv[9:])
        current = self.markers.get(host)
        if action == "observe":
            if current is None:
                return RemoteResult(exit_code=4, stdout="", stderr="")
            return RemoteResult(exit_code=0, stdout=current, stderr="")
        if action == "claim":
            if self.refuse_claims:
                return RemoteResult(exit_code=3, stdout="", stderr="host is claimed elsewhere")
            if current is not None and not self._supersedable(current, cell, uid, generation, fence):
                return RemoteResult(exit_code=3, stdout="", stderr="host is claimed elsewhere")
            payload = _marker(cell, uid, int(generation), fence)
            self.markers[host] = payload
            if command:
                self.lifecycle_commands.append((host, command))
                if self.lifecycle_exit_code:
                    return RemoteResult(
                        exit_code=8, stdout="", stderr=f"lifecycle command exited {self.lifecycle_exit_code}"
                    )
            return RemoteResult(exit_code=0, stdout=payload, stderr="")
        if action == "release":
            if current is not None:
                expected = _marker(cell, uid, int(generation), fence)
                if current != expected:
                    return RemoteResult(exit_code=3, stdout="", stderr="host is claimed elsewhere")
                if command:
                    self.lifecycle_commands.append((host, command))
                    if self.lifecycle_exit_code:
                        return RemoteResult(
                            exit_code=8, stdout="", stderr=f"lifecycle command exited {self.lifecycle_exit_code}"
                        )
                self.markers.pop(host)
            return RemoteResult(exit_code=0, stdout="", stderr="")
        return RemoteResult(exit_code=6, stdout="", stderr="unknown action")

    @staticmethod
    def _supersedable(current: str, cell: str, uid: str, generation: str, fence: str) -> bool:
        if _field(current, "cell_id") != cell or _field(current, "management_uid") != uid:
            return False
        observed = _field(current, "provider_generation")
        return observed.isdigit() and (
            int(observed) < int(generation) or (int(observed) == int(generation) and _field(current, "fence") == fence)
        )


class OnPremSshConformanceHarness(ProviderConformanceHarness):
    """Run the conformance suite against the on-premises driver.

    The worked example ``docs/providers.md`` points an out-of-tree provider at:
    six hooks, each a fact about a substrate only this provider can produce.
    """

    def __init__(self, tmp_path: Path, factory: type[OnPremSshCapacityDriver] = OnPremSshCapacityDriver) -> None:
        self.tmp_path = tmp_path
        self.factory = factory
        self._built = 0

    @property
    def provider_name(self) -> str:
        """Name the provider under test."""

        return "onprem-ssh"

    def new_driver(self) -> CapacityDriver:
        """Return a driver over a private fake rack and durable document."""

        self._built += 1
        state = FleetStateStore(LocalBackend(self.tmp_path / f"fleet-{self._built}.json"))
        return self.factory(
            state,
            runner=FakeSshFleet(),
            native_worker_port=8100,
            native_worker_state_dir="/var/lib/eyvan/native-worker",
        )

    @staticmethod
    def fleet(driver: CapacityDriver) -> FakeSshFleet:
        """Return the fake rack a driver was built over."""

        return tp.cast(FakeSshFleet, tp.cast(OnPremSshCapacityDriver, driver).runner)

    def desired_capacity(self, cell_id: str) -> dict:
        """Return a two-host on-premises pool declaration."""

        del cell_id
        return {
            "provider": "onprem-ssh",
            "capacity_backend": "onprem-ssh",
            "executor": "eyvan-native",
            "accelerator_type": "gpu",
            "capacity_class": "reserved",
            "region": "rack-1",
            "provider_options": {
                "hosts": list(HOSTS),
                "worker_port": 8100,
                "state_dir": "/var/lib/eyvan/native-worker",
                "start_command": ["/usr/local/bin/eyvan-worker", "start"],
                "stop_command": ["/usr/local/bin/eyvan-worker", "stop"],
            },
        }

    def provider_incarnations(self, driver: CapacityDriver, cell_id: str) -> int:
        """Count distinct claimed incarnations of ``cell_id`` across the rack."""

        fleet = self.fleet(driver)
        return len(
            {
                (_field(payload, "provider_generation"), _field(payload, "fence"))
                for payload in fleet.markers.values()
                if _field(payload, "cell_id") == cell_id
            }
        )

    def leak_previous_incarnation(self, driver: CapacityDriver, cell_id: str) -> None:
        """Leave one host holding a claim from a fencing token nobody owns.

        Written unconditionally rather than by editing whatever the driver
        claimed: a provider that never claimed anything must still face a
        substrate holding something, or the check would be measuring the
        harness instead of the driver.
        """

        fleet = self.fleet(driver)
        payload = fleet.markers.get(HOSTS[0], "")
        fleet.markers[HOSTS[0]] = _marker(
            cell_id,
            _field(payload, "management_uid") or "uid-of-a-previous-incarnation",
            int(_field(payload, "provider_generation") or 1),
            marker_fingerprint("a-token-this-cell-never-held"),
        )

    @contextmanager
    def provider_unreachable(self, driver: CapacityDriver) -> tp.Iterator[None]:
        """Make every ssh call fail without establishing an outcome."""

        fleet = self.fleet(driver)
        fleet.unreachable = True
        try:
            yield
        finally:
            fleet.unreachable = False

    @contextmanager
    def provider_rejects_permanently(self, driver: CapacityDriver) -> tp.Iterator[None]:
        """Make every host conclusively refuse the claim."""

        fleet = self.fleet(driver)
        fleet.refuse_claims = True
        try:
            yield
        finally:
            fleet.refuse_claims = False


def _driver(tmp_path: Path) -> tuple[OnPremSshCapacityDriver, FakeSshFleet, dict]:
    harness = OnPremSshConformanceHarness(tmp_path)
    driver = tp.cast(OnPremSshCapacityDriver, harness.new_driver())
    return driver, harness.fleet(driver), harness.desired_capacity("cell-a")


# --------------------------------------------------------------------------- the driver


def test_apply_claims_every_host_and_publishes_their_endpoints(tmp_path) -> None:
    """A claimed rack is HEALTHY and its worker endpoints are the cell's."""

    driver, fleet, desired = _driver(tmp_path)

    observation = driver.apply_capacity("cell-a", desired, _context("apply"))

    assert observation.state is CapacityStatus.HEALTHY
    assert observation.metadata["native_worker_endpoints"] == ["http://rack-a:8100", "http://rack-b:8100"]
    assert observation.runtime_process_count == 2
    assert set(fleet.markers) == set(HOSTS)
    assert _field(fleet.markers["eyvan@rack-a"], "fence") == marker_fingerprint("fence-1")
    assert ("eyvan@rack-a", ("/usr/local/bin/eyvan-worker", "start")) in fleet.lifecycle_commands


def test_a_host_claimed_by_another_cell_is_refused_conclusively(tmp_path) -> None:
    """Two Eyvan cells cannot both own one machine, and retrying cannot fix it."""

    driver, fleet, desired = _driver(tmp_path)
    fleet.markers["eyvan@rack-b"] = _marker("someone-elses-cell", "uid", 4, "f" * 32)

    with pytest.raises(CapacityProviderError) as raised:
        driver.apply_capacity("cell-a", desired, _context("apply"))

    assert raised.value.code is CapacityProviderFailureCode.CONFIGURATION
    assert raised.value.retryable is False
    assert _field(fleet.markers["eyvan@rack-b"], "cell_id") == "someone-elses-cell"


def test_an_unreachable_rack_is_unobserved_not_absent(tmp_path) -> None:
    """SSH failing is a maintenance window, not evidence the hardware is gone."""

    driver, fleet, desired = _driver(tmp_path)
    driver.apply_capacity("cell-a", desired, _context("apply"))
    fleet.unreachable = True

    observation = driver.observe("cell-a")

    assert observation.state is CapacityStatus.UNOBSERVED
    assert observation.failure_code is CapacityProviderFailureCode.PROVIDER_UNAVAILABLE
    assert observation.metadata["native_worker_endpoints"] == []


def test_a_host_that_lost_its_marker_is_recovered_by_reconcile(tmp_path) -> None:
    """A rebuilt host converges back instead of stranding working hardware."""

    driver, fleet, desired = _driver(tmp_path)
    driver.apply_capacity("cell-a", desired, _context("apply"))
    fleet.markers.pop("eyvan@rack-b")

    assert driver.observe("cell-a").state is CapacityStatus.NEEDS_RECREATE

    reconciled = driver.reconcile("cell-a", _context("reconcile"))

    assert reconciled.state is CapacityStatus.HEALTHY
    assert set(fleet.markers) == set(HOSTS)


def test_release_stops_the_workers_and_drops_every_claim(tmp_path) -> None:
    """Release is what makes an on-premises host reusable by another cell."""

    driver, fleet, desired = _driver(tmp_path)
    driver.apply_capacity("cell-a", desired, _context("apply"))

    released = driver.release("cell-a", _context("release", generation=2))

    assert released.state is CapacityStatus.RELEASED
    assert fleet.markers == {}
    assert ("eyvan@rack-a", ("/usr/local/bin/eyvan-worker", "stop")) in fleet.lifecycle_commands


def test_a_release_replay_removes_the_markers_a_crash_left_behind(tmp_path) -> None:
    """The tombstone carries the host identity, so a replay is not a no-op."""

    driver, fleet, desired = _driver(tmp_path)
    driver.apply_capacity("cell-a", desired, _context("apply"))
    claimed_marker = fleet.markers["eyvan@rack-a"]
    release = _context("release", generation=2)
    driver.release("cell-a", release)
    # A crash after the tombstone, before the host was cleaned: the marker is
    # back and the durable record is already gone.
    fleet.markers["eyvan@rack-a"] = claimed_marker

    replayed = driver.release("cell-a", release)

    assert replayed.state is CapacityStatus.RELEASED
    assert fleet.markers == {}


def test_release_never_deletes_another_cells_claim(tmp_path) -> None:
    """The one destructive mistake this driver could make, it does not make."""

    driver, fleet, desired = _driver(tmp_path)
    driver.apply_capacity("cell-a", desired, _context("apply"))
    fleet.markers["eyvan@rack-a"] = _marker("another-cell", "uid", 9, "e" * 32)

    driver.release("cell-a", _context("release", generation=2))

    assert _field(fleet.markers["eyvan@rack-a"], "cell_id") == "another-cell"


def test_drain_keeps_the_claim_so_a_running_workload_stays_fenced(tmp_path) -> None:
    """A draining cell is still running work; an unclaimed host is claimable."""

    driver, fleet, desired = _driver(tmp_path)
    driver.apply_capacity("cell-a", desired, _context("apply"))

    drained = driver.drain("cell-a", _context("drain"))

    assert drained.state is CapacityStatus.DRAINING
    assert set(fleet.markers) == set(HOSTS)
    assert driver.observe("cell-a").metadata["native_worker_endpoints"] == []


def test_a_stale_fencing_token_cannot_release_the_rack(tmp_path) -> None:
    """The durable fence is checked before any host is touched."""

    driver, fleet, desired = _driver(tmp_path)
    driver.apply_capacity("cell-a", desired, _context("apply"))

    with pytest.raises(StaleMutationError):
        driver.release("cell-a", _context("release", generation=1, token="someone-else"))

    assert set(fleet.markers) == set(HOSTS)


def test_management_uid_is_derived_so_worker_tokens_stay_stable(tmp_path) -> None:
    """A random UID would rotate the per-cell token and lock out live workers."""

    first, _, desired = _driver(tmp_path / "a")
    second, _, _ = _driver(tmp_path / "b")

    left = first.apply_capacity("cell-a", desired, _context("apply"))
    right = second.apply_capacity("cell-a", desired, _context("apply"))

    assert left.metadata["management_uid"] == right.metadata["management_uid"]


def test_a_start_command_failure_is_conclusive_not_a_retry_loop(tmp_path) -> None:
    """A launcher that exits non-zero will do so again; the operator must see it."""

    driver, fleet, desired = _driver(tmp_path)
    fleet.lifecycle_exit_code = 2

    with pytest.raises(CapacityProviderError) as raised:
        driver.apply_capacity("cell-a", desired, _context("apply"))

    assert raised.value.retryable is False


@pytest.mark.parametrize(
    ("hosts", "match"),
    (
        ([], "non-empty 'hosts' list"),
        (["rack-a", "rack-a"], "must not contain duplicates"),
        (["http://rack-a:8100"], "ssh destination"),
        (["rack a"], "ssh destination"),
        ([" "], "non-empty string"),
    ),
)
def test_the_host_inventory_is_validated_before_anything_is_claimed(tmp_path, hosts, match) -> None:
    driver, fleet, desired = _driver(tmp_path)
    desired["provider_options"]["hosts"] = hosts

    with pytest.raises(ValueError, match=match):
        driver.apply_capacity("cell-a", desired, _context("apply"))
    assert fleet.markers == {}


def test_on_premises_capacity_refuses_spot(tmp_path) -> None:
    """Operator-owned hardware is never provider-preemptible."""

    driver, _, desired = _driver(tmp_path)
    desired["capacity_class"] = "spot"

    with pytest.raises(ValueError, match="never provider-preemptible"):
        driver.apply_capacity("cell-a", desired, _context("apply"))


def test_the_registry_builds_the_on_premises_driver_from_settings(tmp_path) -> None:
    """The new backend is reachable the way an operator selects it."""

    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))

    driver = DEFAULT_CAPACITY_BACKENDS.create(
        "onprem-ssh",
        state=state,
        settings=EyvanSettings(capacity_backend="onprem-ssh", native_worker_port=9100),
    )

    assert isinstance(driver, OnPremSshCapacityDriver)
    assert driver.native_worker_port == 9100


# --------------------------------------------------------------------------- the real script


def _sh(marker: Path, action: str, cell: str, generation: int, fence: str = "f" * 32) -> subprocess.CompletedProcess:
    """Run the shipped marker script through a real POSIX shell."""

    return subprocess.run(
        ["sh", "-s", "--", action, str(marker), cell, "uid-1", str(generation), fence],
        input=MARKER_SCRIPT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_marker_script_claims_observes_and_releases(tmp_path) -> None:
    """The audited script, run by /bin/sh, implements the protocol it claims."""

    marker = tmp_path / "state" / "eyvan-cell.marker"

    claimed = _sh(marker, "claim", "cell-a", 1)
    assert claimed.returncode == 0
    assert _field(claimed.stdout, "cell_id") == "cell-a"
    assert marker.read_text() == claimed.stdout

    observed = _sh(marker, "observe", "cell-a", 1)
    assert observed.returncode == 0
    assert _field(observed.stdout, "provider_generation") == "1"

    released = _sh(marker, "release", "cell-a", 1)
    assert released.returncode == 0
    assert not marker.exists()

    assert _sh(marker, "observe", "cell-a", 1).returncode == 4


def test_the_marker_script_refuses_another_cells_host(tmp_path) -> None:
    """The claim is the leak detector; it must never take a foreign host."""

    marker = tmp_path / "eyvan-cell.marker"
    assert _sh(marker, "claim", "cell-a", 1).returncode == 0

    assert _sh(marker, "claim", "cell-b", 9).returncode == 3
    assert _sh(marker, "release", "cell-b", 9).returncode == 3
    assert _field(marker.read_text(), "cell_id") == "cell-a"


def test_the_marker_script_refuses_an_older_incarnation_and_supersedes_a_newer_claim(tmp_path) -> None:
    """An older generation cannot overwrite the claim that replaced it."""

    marker = tmp_path / "eyvan-cell.marker"
    assert _sh(marker, "claim", "cell-a", 5).returncode == 0

    assert _sh(marker, "claim", "cell-a", 4).returncode == 3
    assert _field(marker.read_text(), "provider_generation") == "5"
    assert _sh(marker, "claim", "cell-a", 6).returncode == 0
    assert _field(marker.read_text(), "provider_generation") == "6"


def test_the_marker_script_refuses_a_corrupt_marker(tmp_path) -> None:
    """Something wrote it, and it was not this incarnation."""

    marker = tmp_path / "eyvan-cell.marker"
    marker.write_text("cell_id=cell-a\nprovider_generation=not-a-number\n")

    assert _sh(marker, "claim", "cell-a", 1).returncode == 3


def test_the_marker_script_reports_an_unknown_action(tmp_path) -> None:
    """A protocol mismatch is an error, never a silent success."""

    assert _sh(tmp_path / "eyvan-cell.marker", "destroy", "cell-a", 1).returncode == 6
