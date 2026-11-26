# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Local real-shell regressions for fenced on-premises ownership."""

from __future__ import annotations

import copy
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.base import CapacityProviderError, MutationContext
from eyvan.capacity.drivers.onprem_ssh import (
    MARKER_SCRIPT,
    OnPremSshCapacityDriver,
    OpenSshCommandRunner,
    RemoteResult,
    RemoteTransportError,
)
from eyvan.capacity.fleet_state import FleetStateStore, LocalBackend


def context(key: str, generation: int = 1) -> MutationContext:
    return MutationContext(idempotency_key=key, generation=generation, fencing_token="fence-1")


class LocalRack:
    """Execute the actual remote protocol locally, never SSH or worker binaries."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir()
        self.environment = None
        self.log = root / "lifecycle.log"
        self.command = root / "worker"
        self.command.write_text('#!/bin/sh\nprintf "%s\\n" "$1" >> "$2"\n')
        self.command.chmod(0o700)

    def run(self, host, argv, *, stdin="", timeout_s):
        if tuple(argv[:3]) == ("sh", "-s", "--"):
            args = list(argv)
            args[4] = str(self.root / host / str(args[4]).lstrip("/"))
        else:
            args = list(argv)
        completed = subprocess.run(
            args, input=stdin, text=True, capture_output=True, timeout=timeout_s, check=False, env=self.environment
        )
        return RemoteResult(completed.returncode, completed.stdout, completed.stderr)

    def marker(self, host="rack-a", state_dir="/state"):
        return self.root / host / state_dir.lstrip("/") / "eyvan-cell.marker"

    def events(self):
        return self.log.read_text().splitlines() if self.log.exists() else []


def driver_and_desired(tmp_path):
    rack = LocalRack(tmp_path / "rack")
    driver = OnPremSshCapacityDriver(FleetStateStore(LocalBackend(tmp_path / "fleet.json")), runner=rack)
    desired = {
        "provider": "onprem-ssh",
        "capacity_backend": "onprem-ssh",
        "executor": "eyvan-native",
        "accelerator_type": "cpu",
        "capacity_class": "reserved",
        "region": "lab",
        "provider_options": {
            "hosts": ["rack-a", "rack-b"],
            "state_dir": "/state",
            "start_command": [str(rack.command), "start", str(rack.log)],
            "stop_command": [str(rack.command), "stop", str(rack.log)],
        },
    }
    return driver, rack, desired


@pytest.mark.parametrize("change", ["hosts", "state_dir"])
def test_inventory_change_refused_without_discarding_cleanup(tmp_path, change):
    driver, rack, desired = driver_and_desired(tmp_path)
    driver.apply_capacity("cell-a", desired, context("apply"))
    original = driver.state.get("cell-a")
    replacement = copy.deepcopy(desired)
    replacement["provider_options"][change] = ["rack-b"] if change == "hosts" else "/other-state"
    with pytest.raises(CapacityProviderError, match="release"):
        driver.apply_capacity("cell-a", replacement, context("replace", 2))
    assert driver.state.get("cell-a") == original
    assert rack.events() == ["start", "start"]
    assert driver.release("cell-a", context("release", 3)).state is CapacityStatus.RELEASED
    assert driver.release("cell-a", context("release", 3)).state is CapacityStatus.RELEASED
    assert not rack.marker("rack-a").exists()
    assert not rack.marker("rack-b").exists()
    assert rack.events().count("stop") >= 2


def script(marker, action="claim", cell="cell-a", generation=1, uid="uid-a", fence="fence-a", command=(), env=None):
    return subprocess.run(
        ["/bin/sh", "-s", "--", action, str(marker), cell, uid, str(generation), fence, *command],
        input=MARKER_SCRIPT,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
        env=env,
    )


@pytest.mark.parametrize("foreign", ["cell", "generation", "uid", "fence", "absent"])
def test_release_and_tombstone_replay_do_not_stop_foreign_incarnation(tmp_path, foreign):
    driver, rack, desired = driver_and_desired(tmp_path)
    driver.apply_capacity("cell-a", desired, context("apply"))
    marker = rack.marker()
    payload = marker.read_text()
    if foreign == "cell":
        payload = payload.replace("cell_id=cell-a", "cell_id=cell-b")
    elif foreign == "generation":
        payload = payload.replace("provider_generation=1", "provider_generation=9")
    elif foreign == "uid":
        payload = payload.replace("management_uid=", "management_uid=foreign-")
    elif foreign == "fence":
        payload = payload.replace("fence=", "fence=foreign-")
    if foreign == "absent":
        marker.unlink()
    else:
        marker.write_text(payload)
    release = context("release", 2)
    assert driver.release("cell-a", release).state is CapacityStatus.RELEASED
    assert rack.events() == ["start", "start", "stop"]  # only the still-owned rack-b
    assert driver.release("cell-a", release).state is CapacityStatus.RELEASED
    assert rack.events() == ["start", "start", "stop"]
    if foreign != "absent":
        assert marker.read_text() == payload


def barrier(tmp_path, executable):
    """Pause the actual publication/removal after ownership checks, without replacing it."""
    wrapper = tmp_path / "bin"
    wrapper.mkdir()
    ready, resume = tmp_path / "ready", tmp_path / "resume"
    path = wrapper / executable
    path.write_text(
        f'#!/bin/sh\n: > "$READY"\nwhile [ ! -f "$RESUME" ]; do sleep 0.01; done\nexec /bin/{executable} "$@"\n'
    )
    path.chmod(0o700)
    env = dict(os.environ, PATH=f"{wrapper}:{os.environ['PATH']}", READY=str(ready), RESUME=str(resume))
    return env, ready, resume


def wait_ready(ready):
    deadline = time.monotonic() + 5
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ready.exists(), "real script did not reach publication/removal barrier"


def test_concurrent_foreign_claim_cannot_overwrite_acknowledged_owner(tmp_path):
    marker = tmp_path / "state" / "eyvan-cell.marker"
    env, ready, resume = barrier(tmp_path, "mv")
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(script, marker, env=env)
        try:
            wait_ready(ready)
            second = script(marker, cell="cell-b")
        finally:
            resume.touch()
        accepted = first.result()
    assert accepted.returncode == 0
    assert second.returncode != 0
    assert "cell_id=cell-a\n" in marker.read_text()
    assert script(marker, cell="cell-b").returncode == 3


def test_release_cannot_unlink_replacement_claim(tmp_path):
    marker = tmp_path / "state" / "eyvan-cell.marker"
    assert script(marker).returncode == 0
    env, ready, resume = barrier(tmp_path, "rm")
    with ThreadPoolExecutor(max_workers=1) as pool:
        releasing = pool.submit(script, marker, "release", env=env)
        try:
            wait_ready(ready)
            replacement = script(marker, generation=2, fence="fence-b")
        finally:
            resume.touch()
        released = releasing.result()
    assert released.returncode == 0
    assert replacement.returncode != 0
    assert script(marker, generation=2, fence="fence-b").returncode == 0
    assert script(marker, "release").returncode == 3
    assert "provider_generation=2\n" in marker.read_text()


@pytest.mark.parametrize("field", ["uid", "fence"])
def test_equal_generation_claim_cannot_replace_identity(tmp_path, field):
    marker = tmp_path / "eyvan-cell.marker"
    assert script(marker).returncode == 0
    before = marker.read_text()
    replacement = {field: "foreign"}
    assert script(marker, **replacement).returncode == 3
    assert marker.read_text() == before


def test_new_durable_fence_reconcile_advances_remote_identity(tmp_path):
    driver, rack, desired = driver_and_desired(tmp_path)
    driver.apply_capacity("cell-a", desired, context("apply"))
    successor = MutationContext(idempotency_key="reconcile", generation=2, fencing_token="fence-2")
    assert driver.reconcile("cell-a", successor).state is CapacityStatus.HEALTHY
    assert driver.reconcile("cell-a", successor).state is CapacityStatus.HEALTHY
    assert "provider_generation=2\n" in rack.marker().read_text()
    driver.release("cell-a", MutationContext(idempotency_key="release", generation=3, fencing_token="fence-3"))
    assert not rack.marker().exists()


def test_drain_new_fence_keeps_exact_predecessor_cleanup_identity(tmp_path):
    driver, rack, desired = driver_and_desired(tmp_path)
    driver.apply_capacity("cell-a", desired, context("apply"))
    before = rack.marker().read_text()
    driver.drain("cell-a", MutationContext(idempotency_key="drain", generation=2, fencing_token="fence-2"))
    assert rack.marker().read_text() == before
    driver.release("cell-a", MutationContext(idempotency_key="release", generation=3, fencing_token="fence-3"))
    assert not rack.marker().exists()
    assert rack.events() == ["start", "start", "stop", "stop"]


@pytest.mark.parametrize("phase", ["start", "stop"])
def test_lifecycle_failure_keeps_marker_and_refuses_success(tmp_path, phase):
    driver, rack, desired = driver_and_desired(tmp_path)
    if phase == "start":
        rack.command.write_text("#!/bin/sh\nexit 2\n")
        with pytest.raises(CapacityProviderError) as raised:
            driver.apply_capacity("cell-a", desired, context("apply"))
    else:
        driver.apply_capacity("cell-a", desired, context("apply"))
        rack.command.write_text("#!/bin/sh\nexit 2\n")
        with pytest.raises(CapacityProviderError) as raised:
            driver.release("cell-a", context("release", 2))
    assert raised.value.retryable is False
    assert rack.marker().exists()
    assert driver.state.get("cell-a") is not None
    assert not rack.marker().with_name("eyvan-cell.marker.lock").exists()


def test_abandoned_lock_is_fail_closed_and_retryable(tmp_path):
    driver, rack, desired = driver_and_desired(tmp_path)
    driver.apply_capacity("cell-a", desired, context("apply"))
    lock = rack.marker().with_name("eyvan-cell.marker.lock")
    lock.mkdir()
    before = rack.marker().read_text()
    with pytest.raises(CapacityProviderError) as raised:
        driver.release("cell-a", context("release", 2))
    assert raised.value.retryable is True
    assert lock.exists()
    assert rack.marker().read_text() == before
    assert rack.events() == ["start", "start"]
    assert driver.state.get("cell-a") is not None


def test_stop_holds_lock_until_child_completes(tmp_path):
    marker = tmp_path / "state" / "eyvan-cell.marker"
    assert script(marker).returncode == 0
    env, ready, resume = barrier(tmp_path, "unused")
    waiting = tmp_path / "stop"
    waiting.write_text('#!/bin/sh\n: > "$READY"\nwhile [ ! -f "$RESUME" ]; do sleep 0.01; done\n')
    waiting.chmod(0o700)
    with ThreadPoolExecutor(max_workers=1) as pool:
        stopping = pool.submit(script, marker, "release", command=[str(waiting)], env=env)
        try:
            wait_ready(ready)
            replacement = script(marker, generation=2, fence="fence-b")
            assert replacement.returncode != 0
            assert "provider_generation=1\n" in marker.read_text()
        finally:
            resume.touch()
        assert stopping.result().returncode == 0
    assert not marker.exists()
    assert script(marker, generation=2, fence="fence-b").returncode == 0


@pytest.mark.parametrize("operation", ["apply", "reconcile"])
def test_partial_replacement_retains_every_exact_cleanup_identity(tmp_path, operation):
    driver, rack, desired = driver_and_desired(tmp_path)
    driver.apply_capacity("cell-a", desired, context("apply"))
    original_run = rack.run

    def interrupted(host, argv, *, stdin="", timeout_s):
        if host == "rack-b" and argv[3] == "claim":
            raise RemoteTransportError("simulated transport failure before replacement claim")
        return original_run(host, argv, stdin=stdin, timeout_s=timeout_s)

    rack.run = interrupted
    successor = MutationContext(idempotency_key="replace", generation=2, fencing_token="fence-2")
    with pytest.raises(CapacityProviderError):
        if operation == "apply":
            driver.apply_capacity("cell-a", desired, successor)
        else:
            driver.reconcile("cell-a", successor)
    rack.run = original_run
    # A fresh driver must recover both predecessor and partially written successor.
    recovered = OnPremSshCapacityDriver(driver.state, runner=rack)
    release = MutationContext(idempotency_key="release", generation=3, fencing_token="fence-3")
    assert recovered.release("cell-a", release).state is CapacityStatus.RELEASED
    assert recovered.release("cell-a", release).state is CapacityStatus.RELEASED
    assert not rack.marker("rack-a").exists()
    assert not rack.marker("rack-b").exists()
    assert rack.events().count("stop") == 2


def test_ssh_shell_boundary_preserves_lifecycle_argv(monkeypatch, tmp_path):
    """Emulate only SSH's remote shell serialization, with no SSH binary/network."""
    actual_run = subprocess.run
    monkeypatch.chdir(tmp_path)

    def local_remote_shell(argv, **kwargs):
        assert argv[0] == "ssh"
        remote_argv = argv[argv.index("--") + 1 :]
        return actual_run(["/bin/sh", "-c", " ".join(remote_argv)], **kwargs)

    monkeypatch.setattr("eyvan.capacity.drivers.onprem_ssh.subprocess.run", local_remote_shell)
    marker = tmp_path / "eyvan-cell.marker"
    output = tmp_path / "argv.log"
    worker = tmp_path / "worker"
    worker.write_text('#!/bin/sh\nprintf "%s\\n" "$1" > "$2"\n')
    worker.chmod(0o700)
    argument = "literal spaces ; printf bad"
    result = OpenSshCommandRunner().run(
        "rack-a",
        ["sh", "-s", "--", "claim", str(marker), "cell-a", "uid-a", "1", "fence-a", str(worker), argument, str(output)],
        stdin=MARKER_SCRIPT,
        timeout_s=5,
    )
    assert result.exit_code == 0
    assert output.read_text() == argument + "\n"


def test_public_applies_from_independent_stores_cannot_both_start_same_host(tmp_path):
    first, rack, desired = driver_and_desired(tmp_path)
    second = OnPremSshCapacityDriver(FleetStateStore(LocalBackend(tmp_path / "other-fleet.json")), runner=rack)
    env, ready, resume = barrier(tmp_path, "mv")
    rack.environment = env
    with ThreadPoolExecutor(max_workers=1) as pool:
        applying = pool.submit(first.apply_capacity, "cell-a", desired, context("apply-a"))
        try:
            wait_ready(ready)
            with pytest.raises(CapacityProviderError) as raised:
                second.apply_capacity("cell-b", desired, context("apply-b"))
            assert raised.value.retryable is True
            assert second.observe("cell-b").metadata["native_worker_endpoints"] == []
            assert rack.events() == []
        finally:
            resume.touch()
        assert applying.result().state is CapacityStatus.HEALTHY
    assert rack.events() == ["start", "start"]
    assert first.observe("cell-a").state is CapacityStatus.HEALTHY
    assert second.observe("cell-b").metadata["native_worker_endpoints"] == []


def test_tombstone_replay_cleans_only_exact_restored_predecessor(tmp_path):
    driver, rack, desired = driver_and_desired(tmp_path)
    driver.apply_capacity("cell-a", desired, context("apply"))
    original = rack.marker().read_text()
    release = context("release", 2)
    driver.release("cell-a", release)
    rack.marker().write_text(original)
    assert driver.release("cell-a", release).state is CapacityStatus.RELEASED
    assert not rack.marker().exists()
    assert rack.events().count("stop") == 3


def test_empty_default_commands_do_not_run_lifecycle_on_foreign_host(tmp_path):
    driver, rack, desired = driver_and_desired(tmp_path)
    desired["provider_options"].pop("start_command")
    desired["provider_options"].pop("stop_command")
    driver.apply_capacity("cell-a", desired, context("apply"))
    foreign = rack.marker().read_text().replace("cell_id=cell-a", "cell_id=cell-b")
    rack.marker().write_text(foreign)
    release = context("release", 2)
    driver.release("cell-a", release)
    driver.release("cell-a", release)
    assert rack.marker().read_text() == foreign
    assert rack.events() == []


def test_changed_inventory_replay_is_refused_before_effects(tmp_path):
    driver, rack, desired = driver_and_desired(tmp_path)
    driver.apply_capacity("cell-a", desired, context("apply"))
    before = driver.state.get("cell-a")
    desired["provider_options"]["hosts"] = ["rack-b"]
    with pytest.raises(CapacityProviderError, match="release"):
        driver.apply_capacity("cell-a", desired, context("apply"))
    assert driver.state.get("cell-a") == before
    assert rack.events() == ["start", "start"]
