# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for Eyvan-managed JAX distributed initialization.

These exercise ``_multihost_site`` -- the module ``native_bootstrap`` actually
stages onto TPU hosts. They used to exercise a thin ``runtime/multihost.py``
facade that nothing in production imported, so the scrub-before-initialize
ordering was behaviourally covered only in the copy nobody ran; the shipped copy
lived in a generated shell heredoc and could be asserted only as a string.
"""

from __future__ import annotations

import os
import sys
import typing as tp
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from eyvan.worker.runtime import _multihost_site
from eyvan.worker.runtime._multihost_site import initialize_worker_runtime as initialize_from_environment


class _FakeDistributed:
    def __init__(self) -> None:
        self.initialize_calls: list[dict[str, str | int | None]] = []
        #: Topology variables still present at the moment JAX was initialized.
        #: Captured rather than checked afterwards, because *afterwards* both a
        #: correct and a too-late scrub look identical -- and a too-late scrub is
        #: exactly the bug: JAX has already read the override by then.
        self.topology_at_initialize: dict[str, str] = {}

    @staticmethod
    def is_initialized() -> bool:
        return False

    def initialize(
        self,
        *,
        cluster_detection_method: str,
        coordinator_address: str | None = None,
        num_processes: int | None = None,
        process_id: int | None = None,
        initialization_timeout: int | None = None,
        heartbeat_timeout_seconds: int | None = None,
    ) -> None:
        self.initialization_timeout = initialization_timeout
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds
        self.topology_at_initialize = {
            name: value
            for name, value in os.environ.items()
            if name in {"GCE_METADATA_IP", "JAX_LOCAL_DEVICE_IDS", "MEGASCALE_NUM_SLICES", "TPU_SKIP_MDS_QUERY"}
        }
        self.initialize_calls.append(
            {
                "coordinator_address": coordinator_address,
                "num_processes": num_processes,
                "process_id": process_id,
                "cluster_detection_method": cluster_detection_method,
            }
        )


class _FakeJax(ModuleType):
    def __init__(self, distributed: _FakeDistributed) -> None:
        super().__init__("jax")
        self.distributed = distributed


def test_initialize_from_environment_disables_platform_cluster_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    distributed = _FakeDistributed()
    monkeypatch.setitem(sys.modules, "jax", _FakeJax(distributed))
    monkeypatch.setenv("EYVAN_JAX_MULTIHOST", "1")
    monkeypatch.setenv("JAX_COORDINATOR_ADDRESS", "10.12.0.10:12345")
    monkeypatch.setenv("JAX_PROCESS_COUNT", "4")
    monkeypatch.setenv("JAX_PROCESS_ID", "3")

    initialize_from_environment()

    assert distributed.initialize_calls == [
        {
            "coordinator_address": "10.12.0.10:12345",
            "num_processes": 4,
            "process_id": 3,
            "cluster_detection_method": "deactivate",
        }
    ]


def test_standalone_multihost_module_runs_without_importing_eyvan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    distributed = _FakeDistributed()
    monkeypatch.setitem(sys.modules, "jax", _FakeJax(distributed))
    monkeypatch.setenv("EYVAN_JAX_MULTIHOST", "1")
    monkeypatch.setenv("JAX_COORDINATOR_ADDRESS", "10.12.0.10:12345")
    monkeypatch.setenv("JAX_PROCESS_COUNT", "4")
    monkeypatch.setenv("JAX_PROCESS_ID", "1")
    module_path = Path(_multihost_site.__file__)
    spec = spec_from_file_location("eyvan_multihost", module_path)
    assert spec is not None and spec.loader is not None
    standalone = module_from_spec(spec)
    spec.loader.exec_module(standalone)

    standalone.initialize_from_environment()

    assert distributed.initialize_calls[0]["cluster_detection_method"] == "deactivate"


def test_initialize_from_gce_tpu_metadata_leaves_topology_arguments_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    distributed = _FakeDistributed()
    monkeypatch.setitem(sys.modules, "jax", _FakeJax(distributed))
    monkeypatch.setenv("EYVAN_JAX_MULTIHOST", "1")
    monkeypatch.setenv("EYVAN_JAX_CLUSTER_DETECTION", "gcetpu")
    monkeypatch.setenv("GCE_METADATA_IP", "attacker.invalid")
    monkeypatch.setenv("JAX_LOCAL_DEVICE_IDS", "0")
    monkeypatch.setenv("MEGASCALE_NUM_SLICES", "99")
    monkeypatch.setenv("TPU_SKIP_MDS_QUERY", "1")

    initialize_from_environment()

    assert distributed.initialize_calls == [
        {
            "coordinator_address": None,
            "num_processes": None,
            "process_id": None,
            "cluster_detection_method": "gcetpu",
        }
    ]
    assert "GCE_METADATA_IP" not in os.environ
    assert "JAX_LOCAL_DEVICE_IDS" not in os.environ
    assert "MEGASCALE_NUM_SLICES" not in os.environ
    assert "TPU_SKIP_MDS_QUERY" not in os.environ


def test_initialize_from_gce_tpu_metadata_removes_explicit_process_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    distributed = _FakeDistributed()
    monkeypatch.setitem(sys.modules, "jax", _FakeJax(distributed))
    monkeypatch.setenv("EYVAN_JAX_MULTIHOST", "1")
    monkeypatch.setenv("EYVAN_JAX_CLUSTER_DETECTION", "gcetpu")
    monkeypatch.setenv("JAX_PROCESS_ID", "0")

    initialize_from_environment()

    assert "JAX_PROCESS_ID" not in os.environ
    assert distributed.initialize_calls[0]["cluster_detection_method"] == "gcetpu"


def test_gce_topology_overrides_are_scrubbed_before_jax_reads_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Order, not outcome: after the call both orderings look the same.

    Under ``gcetpu`` detection JAX reads topology from GCE metadata during
    ``initialize``. An inherited override removed *after* that has already been
    read, so asserting only the final environment cannot tell a correct scrub
    from a useless one.
    """

    distributed = _FakeDistributed()
    monkeypatch.setitem(sys.modules, "jax", _FakeJax(distributed))
    monkeypatch.setenv("EYVAN_JAX_MULTIHOST", "1")
    monkeypatch.setenv("EYVAN_JAX_CLUSTER_DETECTION", "gcetpu")
    monkeypatch.setenv("GCE_METADATA_IP", "attacker.invalid")
    monkeypatch.setenv("MEGASCALE_NUM_SLICES", "99")

    _multihost_site.initialize_worker_runtime()

    assert distributed.initialize_calls, "JAX must actually have been initialized"
    assert distributed.topology_at_initialize == {}, "the scrub ran too late to matter"


def test_the_heartbeat_deadline_survives_a_model_load(monkeypatch) -> None:
    """JAX's 100-second default kills a host that is busy, not hung.

    This module runs from ``sitecustomize``, so JAX is initialized at interpreter
    startup and the workload's own work happens afterwards -- weight download,
    conversion, first compile -- none of which heartbeats.

    The default must allow cold model loading rather than mistake a busy host
    for a failed distributed peer.
    """

    distributed = _FakeDistributed()
    monkeypatch.setenv("EYVAN_JAX_MULTIHOST", "1")
    monkeypatch.setenv("EYVAN_JAX_CLUSTER_DETECTION", "gcetpu")
    monkeypatch.delenv("EYVAN_JAX_HEARTBEAT_TIMEOUT_S", raising=False)
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(distributed=distributed))

    initialize_from_environment()

    assert distributed.heartbeat_timeout_seconds == _multihost_site.DEFAULT_HEARTBEAT_TIMEOUT_S
    assert distributed.heartbeat_timeout_seconds >= 600, "a cold model pull must not read as a dead host"


def test_an_operator_can_lengthen_the_deadlines(monkeypatch) -> None:
    """A larger model or a slower link is a setting, not a rebuild."""

    distributed = _FakeDistributed()
    monkeypatch.setenv("EYVAN_JAX_MULTIHOST", "1")
    monkeypatch.setenv("EYVAN_JAX_CLUSTER_DETECTION", "gcetpu")
    monkeypatch.setenv("EYVAN_JAX_HEARTBEAT_TIMEOUT_S", "1800")
    monkeypatch.setenv("EYVAN_JAX_INIT_TIMEOUT_S", "1200")
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(distributed=distributed))

    initialize_from_environment()

    assert distributed.heartbeat_timeout_seconds == 1800
    assert distributed.initialization_timeout == 1200


def test_a_nonsense_deadline_is_refused_rather_than_silently_defaulted(monkeypatch) -> None:
    """Ignoring it would hand back a default the operator thought they replaced."""

    distributed = _FakeDistributed()
    monkeypatch.setenv("EYVAN_JAX_MULTIHOST", "1")
    monkeypatch.setenv("EYVAN_JAX_CLUSTER_DETECTION", "gcetpu")
    monkeypatch.setenv("EYVAN_JAX_HEARTBEAT_TIMEOUT_S", "not-a-number")
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(distributed=distributed))

    with pytest.raises(RuntimeError, match="integer"):
        initialize_from_environment()


def test_a_zero_deadline_is_refused(monkeypatch) -> None:
    """Zero would mean "declare every host dead immediately"."""

    distributed = _FakeDistributed()
    monkeypatch.setenv("EYVAN_JAX_MULTIHOST", "1")
    monkeypatch.setenv("EYVAN_JAX_CLUSTER_DETECTION", "gcetpu")
    monkeypatch.setenv("EYVAN_JAX_HEARTBEAT_TIMEOUT_S", "0")
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(distributed=distributed))

    with pytest.raises(RuntimeError, match="positive"):
        initialize_from_environment()


def test_a_subprocess_does_not_rejoin_and_kill_the_cluster(monkeypatch) -> None:
    """The cause of a v5p inference job dying with no host having died.

    ``sitecustomize`` runs in every interpreter that inherits ``PYTHONPATH``, and
    the workload environment carries both that path and ``EYVAN_JAX_MULTIHOST=1``.
    Any Python subprocess -- ``gcloud`` is one -- therefore joined the
    coordination service as a duplicate task carrying its parent's
    ``JAX_PROCESS_ID``, then exited when its work was done. The service saw a
    task die and tore down every rank: "another task died", no host dead.

    Ray never hits this: it injects no sitecustomize, so user code initializes
    once, itself.
    """

    distributed = _FakeDistributed()
    monkeypatch.setenv("EYVAN_JAX_MULTIHOST", "1")
    monkeypatch.setenv("EYVAN_JAX_CLUSTER_DETECTION", "gcetpu")
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(distributed=distributed))

    initialize_from_environment()

    assert len(distributed.initialize_calls) == 1
    # A child inherits a copy of this environment; the marker must already be
    # gone from it, or that child joins the cluster and its exit kills the job.
    assert "EYVAN_JAX_MULTIHOST" not in os.environ


def test_the_marker_is_consumed_even_for_explicit_detection(monkeypatch) -> None:
    """Both detection paths spawn subprocesses; neither may leak the marker."""

    distributed = _FakeDistributed()
    monkeypatch.setenv("EYVAN_JAX_MULTIHOST", "1")
    monkeypatch.setenv("EYVAN_JAX_CLUSTER_DETECTION", "explicit")
    monkeypatch.setenv("JAX_COORDINATOR_ADDRESS", "10.0.0.1:8476")
    monkeypatch.setenv("JAX_PROCESS_COUNT", "2")
    monkeypatch.setenv("JAX_PROCESS_ID", "0")
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(distributed=distributed))

    initialize_from_environment()

    assert "EYVAN_JAX_MULTIHOST" not in os.environ


def test_a_second_call_in_the_same_process_is_a_no_op(monkeypatch) -> None:
    """Consuming the marker must not make a re-entrant call misbehave."""

    distributed = _FakeDistributed()
    monkeypatch.setenv("EYVAN_JAX_MULTIHOST", "1")
    monkeypatch.setenv("EYVAN_JAX_CLUSTER_DETECTION", "gcetpu")
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(distributed=distributed))

    initialize_from_environment()
    initialize_from_environment()

    assert len(distributed.initialize_calls) == 1


class _Clock:
    """A monotonic clock the barrier below advances by the time it makes a rank wait."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _GangBarrier:
    """A registration barrier with the one property that loses cold gangs.

    JAX's ``RegisterTask`` makes every already-registered task wait for the
    slowest peer and aborts *all* of them when the deadline passes first. So
    whether a gang forms is decided by one comparison: the deadline the ranks
    were given against the arrival of the last host. That deadline is the number
    Eyvan now owns, which is why it is the only thing this fake models.
    """

    def __init__(self, clock: _Clock, *, last_arrival_s: float | None) -> None:
        self._clock = clock
        self._last_arrival_s = last_arrival_s
        self.initialize_calls: list[dict[str, object]] = []

    @staticmethod
    def is_initialized() -> bool:
        return False

    def initialize(self, **arguments: object) -> None:
        self.initialize_calls.append(arguments)
        deadline = arguments["initialization_timeout"]
        assert isinstance(deadline, int)
        arrival = self._last_arrival_s
        if arrival is None or arrival > deadline:
            self._clock.advance(deadline)
            raise RuntimeError("DEADLINE_EXCEEDED: RegisterTask timed out waiting for tasks to register")
        self._clock.advance(arrival)


class _RankTerminated(Exception):
    """Stands in for the rank's process exit so a test can survive it."""

    def __init__(self, exit_status: int) -> None:
        super().__init__(f"rank exited with status {exit_status}")
        self.exit_status = exit_status


def _multihost_rank(monkeypatch: pytest.MonkeyPatch, barrier: _GangBarrier, clock: _Clock) -> list[int]:
    """Wire one rank onto a barrier and return where its exit status lands."""

    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(distributed=barrier))
    monkeypatch.setattr(_multihost_site, "time", SimpleNamespace(monotonic=clock.monotonic))
    exits: list[int] = []

    def terminate(exit_status: int) -> tp.NoReturn:
        exits.append(exit_status)
        raise _RankTerminated(exit_status)

    monkeypatch.setattr(_multihost_site, "terminate_rank", terminate)
    monkeypatch.setenv("EYVAN_JAX_MULTIHOST", "1")
    monkeypatch.setenv("EYVAN_JAX_CLUSTER_DETECTION", "gcetpu")
    return exits


def _apply(monkeypatch: pytest.MonkeyPatch, environment: dict[str, str]) -> None:
    """Set exactly the coordination environment an executor would inject."""

    for name, value in environment.items():
        monkeypatch.setenv(name, value)


def _marker(captured: str, marker: str) -> str:
    """Return the single marker line a rank wrote, failing if it wrote none."""

    lines = [line for line in captured.splitlines() if line.startswith(marker)]
    assert len(lines) == 1, f"expected exactly one {marker!r} line in {captured!r}"
    return lines[0]


def test_a_late_rank_does_not_abort_a_forming_gang(monkeypatch, capsys) -> None:
    """The failure this workstream exists for, reduced to its one comparison.

    A host that is 450 seconds late is ordinary on a cold gang restart after a
    preemption: environment sync, imports, a GCS read. Under JAX's 300-second
    default the ranks that *did* arrive abort, and the gang -- every host, every
    accelerator -- is lost to nothing but start-up variance. Under the deadline
    Eyvan injects, the same gang forms.
    """

    clock = _Clock()
    barrier = _GangBarrier(clock, last_arrival_s=450.0)
    _multihost_rank(monkeypatch, barrier, clock)
    _apply(monkeypatch, _multihost_site.GangInitializationPolicy().environment())

    _multihost_site.initialize_from_environment()

    assert barrier.initialize_calls[0]["initialization_timeout"] == 600
    joined = _marker(capsys.readouterr().err, _multihost_site.GANG_INITIALIZATION_JOINED_MARKER)
    assert "waited_s=450.0" in joined


def test_the_same_late_rank_loses_the_gang_under_jaxs_default(monkeypatch, capsys) -> None:
    """The contrast that makes the deadline load-bearing rather than decorative."""

    clock = _Clock()
    barrier = _GangBarrier(clock, last_arrival_s=450.0)
    _multihost_rank(monkeypatch, barrier, clock)
    # Fatality off isolates what this test is about -- the deadline firing at
    # all -- from what the fatal tests cover, which is what happens after.
    _apply(
        monkeypatch,
        _multihost_site.GangInitializationPolicy(initialization_timeout_s=300, fatal=False).environment(),
    )

    with pytest.raises(RuntimeError, match="DEADLINE_EXCEEDED"):
        _multihost_site.initialize_from_environment()

    failed = _marker(capsys.readouterr().err, _multihost_site.GANG_INITIALIZATION_FAILED_MARKER)
    assert "waited_s=300.0" in failed
    assert "outcome=deadline" in failed


def test_a_stuck_initialization_ends_at_the_deadline_with_a_distinct_status(monkeypatch, capsys) -> None:
    """A peer that never arrives must end the rank, not be swallowed.

    ``site.py`` catches every ``Exception`` raised from ``sitecustomize``,
    prints one line and lets the interpreter continue -- so without this a rank
    whose gang never formed runs the workload as an unwitting single process and
    can exit 0. The exit status is what makes "the gang never formed"
    recoverable from the host afterwards.
    """

    clock = _Clock()
    barrier = _GangBarrier(clock, last_arrival_s=None)
    exits = _multihost_rank(monkeypatch, barrier, clock)
    _apply(monkeypatch, _multihost_site.GangInitializationPolicy(fatal=True).environment())

    with pytest.raises(_RankTerminated):
        _multihost_site.initialize_from_environment()

    assert exits == [_multihost_site.GANG_INITIALIZATION_DEADLINE_EXIT_CODE]
    failed = _marker(capsys.readouterr().err, _multihost_site.GANG_INITIALIZATION_FAILED_MARKER)
    assert "waited_s=600.0" in failed
    assert "deadline_s=600" in failed
    assert "outcome=deadline" in failed


def test_a_join_that_fails_early_is_not_reported_as_a_deadline(monkeypatch, capsys) -> None:
    """A refused connection is not a slow gang, and says so.

    Both are "the gang never formed", but they call for opposite actions --
    lengthen the window, or go and fix this host -- so they carry different exit
    statuses. Calling a two-second failure a deadline would send an operator to
    raise a timeout that was never reached.
    """

    clock = _Clock()

    class _RefusedBarrier(_GangBarrier):
        def initialize(self, **arguments: object) -> None:
            self.initialize_calls.append(arguments)
            self._clock.advance(2.0)
            raise RuntimeError("UNAVAILABLE: failed to connect to the coordination service")

    barrier = _RefusedBarrier(clock, last_arrival_s=None)
    exits = _multihost_rank(monkeypatch, barrier, clock)
    _apply(monkeypatch, _multihost_site.GangInitializationPolicy(fatal=True).environment())

    with pytest.raises(_RankTerminated):
        _multihost_site.initialize_from_environment()

    assert exits == [_multihost_site.GANG_INITIALIZATION_ERROR_EXIT_CODE]
    failed = _marker(capsys.readouterr().err, _multihost_site.GANG_INITIALIZATION_FAILED_MARKER)
    assert "outcome=error" in failed
    assert "waited_s=2.0" in failed


def test_without_the_fatal_flag_a_failed_join_keeps_todays_behaviour(monkeypatch, capsys) -> None:
    """Fatality off, per an operator's explicit choice -- and it is not a safe one.

    The error propagates exactly as it always has, which means ``site.py``
    swallows it and the workload runs on. The marker line is still written, so
    the failure is at least visible in the rank's log before it is ignored.
    """

    clock = _Clock()
    barrier = _GangBarrier(clock, last_arrival_s=None)
    exits = _multihost_rank(monkeypatch, barrier, clock)
    _apply(monkeypatch, _multihost_site.GangInitializationPolicy(fatal=False).environment())

    with pytest.raises(RuntimeError, match="DEADLINE_EXCEEDED"):
        _multihost_site.initialize_from_environment()

    assert exits == []
    assert _multihost_site.GANG_INITIALIZATION_FAILED_MARKER in capsys.readouterr().err


def test_a_single_host_job_reads_no_gang_deadline_at_all(monkeypatch, capsys) -> None:
    """One host has no gang to form, so none of this may apply to it.

    Asserted with values that would fail loudly if they were read: a deadline
    that is not a number, and a barrier that aborts every gang it is given.
    """

    clock = _Clock()
    barrier = _GangBarrier(clock, last_arrival_s=None)
    exits = _multihost_rank(monkeypatch, barrier, clock)
    monkeypatch.delenv("EYVAN_JAX_MULTIHOST", raising=False)
    monkeypatch.setenv(_multihost_site.INITIALIZATION_TIMEOUT_ENV, "not-a-number")
    monkeypatch.setenv(_multihost_site.FATAL_INITIALIZATION_ENV, "not-a-boolean")

    _multihost_site.initialize_from_environment()

    assert barrier.initialize_calls == []
    assert exits == []
    assert capsys.readouterr().err == ""


def test_a_nonsense_fatality_flag_is_refused_rather_than_read_as_off(monkeypatch) -> None:
    """Reading it as off would silently restore the behaviour it disables."""

    clock = _Clock()
    barrier = _GangBarrier(clock, last_arrival_s=1.0)
    _multihost_rank(monkeypatch, barrier, clock)
    monkeypatch.setenv(_multihost_site.FATAL_INITIALIZATION_ENV, "yes-please")

    with pytest.raises(RuntimeError, match=_multihost_site.FATAL_INITIALIZATION_ENV):
        _multihost_site.initialize_from_environment()


def test_a_policy_cannot_be_built_with_a_deadline_that_aborts_every_gang() -> None:
    """Zero or negative means "abort immediately", which is never what was meant."""

    with pytest.raises(ValueError, match="positive"):
        _multihost_site.GangInitializationPolicy(initialization_timeout_s=0)
    with pytest.raises(ValueError, match="positive"):
        _multihost_site.GangInitializationPolicy(heartbeat_timeout_s=-1)


def test_the_shipped_default_bounds_what_an_unformed_gang_can_do() -> None:
    """Configuring nothing still ends a rank whose gang never formed.

    The deadlines keep their historical values -- those are measured against
    cold starts, not against safety -- but fatality ships on: without it a gang
    that lost a host keeps running as unwitting single-process jobs that can
    exit 0. An operator who wants the old behaviour says ``fatal=False``.
    """

    policy = _multihost_site.GangInitializationPolicy()

    assert policy.environment() == {
        _multihost_site.INITIALIZATION_TIMEOUT_ENV: str(_multihost_site.DEFAULT_INITIALIZATION_TIMEOUT_S),
        _multihost_site.HEARTBEAT_TIMEOUT_ENV: str(_multihost_site.DEFAULT_HEARTBEAT_TIMEOUT_S),
        _multihost_site.FATAL_INITIALIZATION_ENV: "1",
    }


def test_the_staged_copy_builds_a_policy_without_being_registered_as_a_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The staged file is exec'd on a host that has no ``eyvan`` package at all.

    Anything in it that resolves its own annotations through ``sys.modules`` --
    ``dataclass`` does -- fails at interpreter start, inside ``sitecustomize``,
    in every workload process, before a line of user code runs.
    """

    spec = spec_from_file_location("eyvan_multihost", Path(_multihost_site.__file__))
    assert spec is not None and spec.loader is not None
    staged = module_from_spec(spec)
    monkeypatch.delitem(sys.modules, "eyvan_multihost", raising=False)
    spec.loader.exec_module(staged)

    assert staged.GangInitializationPolicy(initialization_timeout_s=1800, fatal=True).environment() == {
        "EYVAN_JAX_INIT_TIMEOUT_S": "1800",
        "EYVAN_JAX_HEARTBEAT_TIMEOUT_S": str(_multihost_site.DEFAULT_HEARTBEAT_TIMEOUT_S),
        "EYVAN_JAX_INIT_FATAL": "1",
    }


def test_an_interrupted_rank_is_not_recorded_as_a_gang_that_never_formed(monkeypatch, capsys) -> None:
    """Somebody stopped this workload; nothing was wrong with the gang.

    Labelling it a failed join would send an operator to lengthen a deadline
    that was never reached, and would spend an exit status that means something
    else.
    """

    clock = _Clock()

    class _InterruptedBarrier(_GangBarrier):
        def initialize(self, **arguments: object) -> None:
            self.initialize_calls.append(arguments)
            self._clock.advance(30.0)
            raise KeyboardInterrupt

    barrier = _InterruptedBarrier(clock, last_arrival_s=None)
    exits = _multihost_rank(monkeypatch, barrier, clock)
    _apply(monkeypatch, _multihost_site.GangInitializationPolicy(fatal=True).environment())

    with pytest.raises(KeyboardInterrupt):
        _multihost_site.initialize_from_environment()

    assert exits == []
    assert _multihost_site.GANG_INITIALIZATION_FAILED_MARKER not in capsys.readouterr().err


class _OlderJaxDistributed(_FakeDistributed):
    """``jax.distributed`` as JAX 0.6.2 shipped it: no ``heartbeat_timeout_seconds``."""

    def initialize(  # type: ignore[override]
        self,
        *,
        cluster_detection_method: str,
        coordinator_address: str | None = None,
        num_processes: int | None = None,
        process_id: int | None = None,
        initialization_timeout: int | None = None,
    ) -> None:
        super().initialize(
            cluster_detection_method=cluster_detection_method,
            coordinator_address=coordinator_address,
            num_processes=num_processes,
            process_id=process_id,
            initialization_timeout=initialization_timeout,
        )


@pytest.mark.parametrize("detection", ["gcetpu", "explicit"])
def test_a_jax_without_the_heartbeat_argument_still_forms_the_gang(monkeypatch, capsys, detection: str) -> None:
    """Drop unsupported JAX tuning arguments with a diagnostic while retaining gang identity arguments."""

    distributed = _OlderJaxDistributed()
    monkeypatch.setenv("EYVAN_JAX_MULTIHOST", "1")
    monkeypatch.setenv("EYVAN_JAX_CLUSTER_DETECTION", detection)
    if detection == "explicit":
        monkeypatch.setenv("JAX_COORDINATOR_ADDRESS", "10.0.0.2:8476")
        monkeypatch.setenv("JAX_PROCESS_COUNT", "2")
        monkeypatch.setenv("JAX_PROCESS_ID", "1")
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(distributed=distributed))

    initialize_from_environment()

    assert len(distributed.initialize_calls) == 1
    assert distributed.initialization_timeout == _multihost_site.DEFAULT_INITIALIZATION_TIMEOUT_S
    if detection == "explicit":
        assert distributed.initialize_calls[0]["process_id"] == 1
        assert distributed.initialize_calls[0]["num_processes"] == 2
    assert "does not accept heartbeat_timeout_seconds" in capsys.readouterr().err


def test_a_jax_that_takes_any_keyword_still_gets_the_heartbeat(monkeypatch) -> None:
    received: dict[str, object] = {}

    class _Permissive:
        @staticmethod
        def is_initialized() -> bool:
            return False

        @staticmethod
        def initialize(**arguments: object) -> None:
            received.update(arguments)

    monkeypatch.setenv("EYVAN_JAX_MULTIHOST", "1")
    monkeypatch.setenv("EYVAN_JAX_CLUSTER_DETECTION", "gcetpu")
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(distributed=_Permissive()))

    initialize_from_environment()

    assert received["heartbeat_timeout_seconds"] == _multihost_site.DEFAULT_HEARTBEAT_TIMEOUT_S


def test_a_one_process_world_joins_nothing_and_raises_nothing(monkeypatch) -> None:
    """Each cell of an elastic job is its own world; a single-host cell's world is one
    process. That printed "process configuration is invalid" on every rank."""

    distributed = _FakeDistributed()
    monkeypatch.setenv("EYVAN_JAX_MULTIHOST", "1")
    monkeypatch.setenv("EYVAN_JAX_CLUSTER_DETECTION", "explicit")
    monkeypatch.setenv("JAX_COORDINATOR_ADDRESS", "10.0.0.2:8476")
    monkeypatch.setenv("JAX_PROCESS_COUNT", "1")
    monkeypatch.setenv("JAX_PROCESS_ID", "0")
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(distributed=distributed))

    initialize_from_environment()

    assert distributed.initialize_calls == []


@pytest.mark.parametrize(("count", "process_id"), (("0", "0"), ("1", "1"), ("2", "2")))
def test_an_impossible_world_is_still_refused(monkeypatch, count: str, process_id: str) -> None:
    monkeypatch.setenv("EYVAN_JAX_MULTIHOST", "1")
    monkeypatch.setenv("EYVAN_JAX_CLUSTER_DETECTION", "explicit")
    monkeypatch.setenv("JAX_COORDINATOR_ADDRESS", "10.0.0.2:8476")
    monkeypatch.setenv("JAX_PROCESS_COUNT", count)
    monkeypatch.setenv("JAX_PROCESS_ID", process_id)
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(distributed=_FakeDistributed()))

    with pytest.raises(RuntimeError, match="invalid"):
        initialize_from_environment()
