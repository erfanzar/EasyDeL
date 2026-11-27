# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Standalone JAX distributed initialization for native worker Python paths.

This module is *staged* onto TPU hosts by ``native_bootstrap`` as
``eyvan_multihost.py``, next to ``eyvan_distributed_environment.py``, and driven
from a generated ``sitecustomize.py``. It is therefore imported under two
different names -- as part of the ``eyvan`` package here, and as a bare module
there -- which is why the one cross-module import below is written to work either
way rather than assuming the package layout.
"""

from __future__ import annotations

import os
import sys
import time
import typing as tp


def _arm_cross_region_guard(environment: tp.Mapping[str, str]) -> None:
    """Arm Eyvan's cross-region GCS transfer guard when policy env vars are present.

    Tries the staged sibling module name first (TPU/GPU hosts where ``eyvan`` is
    not on the workload site path), falling back to ``eyvan.storage._region_guard``.
    """

    policy = (environment.get("EYVAN_CROSS_REGION_POLICY") or "").strip().lower()
    if not policy or policy == "allow":
        return
    try:
        from eyvan_region_guard import (  # type: ignore[import-not-found]
            arm_cross_region_guard_from_environment,
        )
    except ImportError:
        try:
            from eyvan.storage._region_guard import arm_cross_region_guard_from_environment
        except ImportError:
            return
    arm_cross_region_guard_from_environment(environment)


def _scrub_gce_topology_overrides(environment: tp.MutableMapping[str, str]) -> None:
    """Drop topology values that must come from GCE TPU metadata discovery.

    The import is tried under the staged module name first because that is the
    context that matters: on a TPU host there is no ``eyvan`` package on the
    site path, only the two files the bootstrap installed.
    """

    try:
        from eyvan_distributed_environment import (  # type: ignore[import-not-found]
            remove_gce_tpu_topology_overrides,
        )
    except ImportError:
        from eyvan.worker.distributed_environment import remove_gce_tpu_topology_overrides
    remove_gce_tpu_topology_overrides(environment)


#: JAX declares a task dead if it misses this many seconds of heartbeats, and
#: its own default is 100. That is far too short for the work that happens
#: *after* initialization on a real ML host: this module runs from
#: ``sitecustomize``, so JAX is initialized at interpreter startup and the user's
#: code then downloads weights, converts them, and compiles -- none of which
#: heartbeats. The timeout must accommodate this work to avoid killing busy
#: ranks during model loading or compilation.
#:
#: Eyvan makes this call on the workload's behalf, so the default is Eyvan's to
#: choose. Fifteen minutes accommodates a cold model pull and a first compile;
#: a genuinely hung host is still detected, just not a busy one.
DEFAULT_HEARTBEAT_TIMEOUT_S = 900

#: Skew between hosts *reaching* initialization: image pull, container start,
#: interpreter boot. JAX defaults to 300.
DEFAULT_INITIALIZATION_TIMEOUT_S = 600

#: Every name below starts with ``EYVAN_JAX_`` on purpose:
#: ``distributed_environment.is_distributed_topology_environment_name`` strips
#: that whole prefix from a submitted environment, so these are Eyvan's to set
#: and a caller cannot pre-seed one. Naming them here rather than at the
#: injection site keeps the producer and the consumer of the contract in one
#: file -- a slice executor that injected ``EYVAN_JAX_INIT_TIMEOUT`` while this
#: module read ``EYVAN_JAX_INIT_TIMEOUT_S`` would silently ship the default.
INITIALIZATION_TIMEOUT_ENV = "EYVAN_JAX_INIT_TIMEOUT_S"
HEARTBEAT_TIMEOUT_ENV = "EYVAN_JAX_HEARTBEAT_TIMEOUT_S"
FATAL_INITIALIZATION_ENV = "EYVAN_JAX_INIT_FATAL"

#: Exit status a rank uses when it waited out the whole initialization deadline
#: without the gang forming, and when it failed to join for any other reason.
#:
#: Two codes rather than one because they call for opposite operator actions:
#: a deadline says the gang was still assembling and the window was too short,
#: an early error says something is wrong with this rank. Both are still "the
#: gang never formed", which is what separates them from a workload failure.
#:
#: 93 and 94 sit above ``sysexits.h`` (64-78) and below the shell's
#: signal-encoded range (128+), so neither can be confused with a conventional
#: status. A workload that chooses 93 for itself would be misread -- it never
#: gets the chance, because a rank that reaches these codes exits before user
#: code is imported.
GANG_INITIALIZATION_DEADLINE_EXIT_CODE = 93
GANG_INITIALIZATION_ERROR_EXIT_CODE = 94

#: Exit code, stderr marker, and stable failure-reason prefix for Eyvan's
#: runtime cross-region GCS transfer ban.
CROSS_REGION_TRANSFER_BANNED_EXIT_CODE = 95
CROSS_REGION_TRANSFER_BANNED_MARKER = "eyvan-cross-region-transfer-banned"
CROSS_REGION_TRANSFER_BANNED_REASON = "cross-region-transfer-banned"

#: Single-line markers written to this rank's stderr, which systemd journals and
#: Eyvan's log page serves. The failure marker carries the *observed* wait; the
#: exit code carries only which kind of failure it was.
GANG_INITIALIZATION_JOINED_MARKER = "eyvan-gang-init-joined"
GANG_INITIALIZATION_FAILED_MARKER = "eyvan-gang-init-failed"

#: Stable machine-readable prefix an executor puts in front of the terminal
#: message when the gang never formed. It reaches ``WorkloadRecord.failure_reason``
#: unchanged, which is the point: "the gang never formed" and "the workload
#: failed" were previously the same terminal record.
GANG_NEVER_FORMED_REASON = "gang-init-never-formed"

#: How close to the deadline a failed join has to land to be called a deadline
#: rather than an error. A fraction rather than a fixed slack because the
#: deadline is configurable across three orders of magnitude, and JAX's own
#: barrier returns a little either side of the value it was given.
_DEADLINE_FRACTION = 0.95

#: Bound on the JAX diagnostic copied into the marker line. The line has to stay
#: one line: it is read back by grep and by eye, not parsed.
_ERROR_DETAIL_LIMIT = 240


class GangInitializationPolicy:
    """The coordination deadlines Eyvan applies to one multi-host workload.

    Constructed on the control-plane side and injected into every rank's
    environment; read back here, on the host, by
    :func:`initialize_from_environment`.

    Written out by hand rather than as a ``dataclass`` because this file is also
    *staged* onto TPU hosts and executed at interpreter start: ``dataclass``
    resolves string annotations through ``sys.modules[cls.__module__]``, which a
    loader that execs the staged copy without registering it does not have, and
    the failure would land in every workload process before any of them ran a
    line of user code. Treat instances as immutable.

    Attributes:
        initialization_timeout_s: How long a rank waits for every peer to reach
            the registration barrier. JAX defaults to 300, which a cold gang
            restart routinely exceeds while hosts are still syncing an
            environment -- and the ranks that *did* arrive abort the whole gang.
        heartbeat_timeout_s: How long a joined rank may go without heartbeating
            before the coordination service declares it dead.
        fatal: Whether a rank that never joined ends its own workload. On by
            default: ``site.py`` swallows every ordinary exception raised from
            ``sitecustomize``, so without it a failed join leaves the workload
            running as an unwitting single-process job that can exit 0. A rank
            that cannot form its gang has nothing useful to compute.
    """

    __slots__ = ("fatal", "heartbeat_timeout_s", "initialization_timeout_s")

    def __init__(
        self,
        initialization_timeout_s: int = DEFAULT_INITIALIZATION_TIMEOUT_S,
        heartbeat_timeout_s: int = DEFAULT_HEARTBEAT_TIMEOUT_S,
        fatal: bool = True,
    ) -> None:
        """Validate one gang's deadlines, refusing values that bound nothing.

        Args:
            initialization_timeout_s: Seconds a rank waits at the barrier.
            heartbeat_timeout_s: Seconds a joined rank may miss heartbeats for.
            fatal: Whether a rank that never joined ends its own workload.

        Raises:
            ValueError: If a deadline is not a positive whole number of seconds,
                or ``fatal`` is not a boolean.
        """

        for name, value in (
            ("initialization_timeout_s", initialization_timeout_s),
            ("heartbeat_timeout_s", heartbeat_timeout_s),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"gang {name} must be an integer number of seconds")
            if value <= 0:
                raise ValueError(f"gang {name} must be positive")
        if not isinstance(fatal, bool):
            raise ValueError("gang initialization fatality must be a boolean")
        self.initialization_timeout_s = initialization_timeout_s
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self.fatal = fatal

    def __repr__(self) -> str:
        """Return the deadlines this policy applies."""

        return (
            f"GangInitializationPolicy(initialization_timeout_s={self.initialization_timeout_s}, "
            f"heartbeat_timeout_s={self.heartbeat_timeout_s}, fatal={self.fatal})"
        )

    def __eq__(self, other: object) -> bool:
        """Compare two policies by the deadlines they apply."""

        if not isinstance(other, GangInitializationPolicy):
            return NotImplemented
        return (
            self.initialization_timeout_s == other.initialization_timeout_s
            and self.heartbeat_timeout_s == other.heartbeat_timeout_s
            and self.fatal == other.fatal
        )

    def __hash__(self) -> int:
        """Hash the deadlines this policy applies."""

        return hash((self.initialization_timeout_s, self.heartbeat_timeout_s, self.fatal))

    def environment(self) -> dict[str, str]:
        """Return the Eyvan-owned coordination variables for one rank."""

        return {
            INITIALIZATION_TIMEOUT_ENV: str(self.initialization_timeout_s),
            HEARTBEAT_TIMEOUT_ENV: str(self.heartbeat_timeout_s),
            FATAL_INITIALIZATION_ENV: "1" if self.fatal else "0",
        }


#: What a rank gets when nothing configured it. Identical to the values this
#: module has always defaulted to, so an unconfigured deployment sees no change.
DEFAULT_GANG_INITIALIZATION = GangInitializationPolicy()


def gang_initialization_outcome(exit_status: int | None) -> str | None:
    """Return why a rank never joined the gang, read back from its exit status.

    Args:
        exit_status: The process exit status a host reported, or ``None`` when
            the host could not be asked.

    Returns:
        ``"deadline"``, ``"error"``, or ``None`` when this exit says nothing
        about gang formation. ``None`` is not "the gang formed" -- an ordinary
        workload failure and an unreadable host both land here.
    """

    if exit_status == GANG_INITIALIZATION_DEADLINE_EXIT_CODE:
        return "deadline"
    if exit_status == GANG_INITIALIZATION_ERROR_EXIT_CODE:
        return "error"
    return None


def gang_formation_failure_reason(
    *,
    deadline_s: int,
    deadline_ranks: tp.Sequence[int],
    error_ranks: tp.Sequence[int],
) -> str:
    """Render the terminal reason for a gang that never formed.

    Args:
        deadline_s: The initialization deadline Eyvan actually applied.
        deadline_ranks: Ranks that waited the deadline out.
        error_ranks: Ranks whose join failed before the deadline.

    Returns:
        One line beginning with :data:`GANG_NEVER_FORMED_REASON`.
    """

    parts: list[str] = []
    if deadline_ranks:
        ranks = ", ".join(str(rank) for rank in sorted(deadline_ranks))
        parts.append(f"rank(s) {ranks} waited the full {deadline_s}s initialization deadline")
    if error_ranks:
        ranks = ", ".join(str(rank) for rank in sorted(error_ranks))
        parts.append(f"rank(s) {ranks} failed to join before the {deadline_s}s deadline")
    detail = "; ".join(parts) if parts else f"no rank joined within the {deadline_s}s deadline"
    return (
        f"{GANG_NEVER_FORMED_REASON}: the gang never formed -- {detail}. "
        f"Each rank's observed wait is on its {GANG_INITIALIZATION_FAILED_MARKER} log line."
    )


def _timeout(name: str, default: int) -> int:
    """Return a positive integer timeout from the environment, or the default."""

    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer number of seconds") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be positive")
    return value


def _fatal_initialization() -> bool:
    """Return whether a rank that never joined must end its own workload."""

    raw = os.getenv(FATAL_INITIALIZATION_ENV, "")
    if raw in {"", "0", "false", "False"}:
        return False
    if raw in {"1", "true", "True"}:
        return True
    raise RuntimeError(f"{FATAL_INITIALIZATION_ENV} must be 0 or 1")


def _rank_identity() -> tuple[str, str]:
    """Return this rank's index and its gang size, or ``unknown`` for either.

    Under ``gcetpu`` detection JAX discovers both from metadata and Eyvan has
    scrubbed them from the environment by the time this runs, so there is
    genuinely nothing to report. Saying ``unknown`` keeps that distinct from
    rank 0 of 1.
    """

    return os.getenv("JAX_PROCESS_ID") or "unknown", os.getenv("JAX_PROCESS_COUNT") or "unknown"


def _emit(line: str) -> None:
    """Write one marker line to this rank's journaled stderr, and flush it.

    Flushed because the caller may be about to ``os._exit``, which does not.
    Never raises: this runs from ``sitecustomize``, where an interpreter with no
    usable stderr would otherwise turn a diagnostic into a failure to initialize
    JAX at all -- and ``site.py`` would swallow *that*, leaving the workload
    running unjoined.
    """

    try:
        stream = sys.stderr
        if stream is None:
            return
        stream.write(f"{line}\n")
        stream.flush()
    except Exception:
        return


#: How a rank that never joined ends itself. Indirected so a test can observe
#: the exit instead of taking the interpreter down with it.
#:
#: ``os._exit`` rather than an exception, because ``site.py`` catches every
#: ``Exception`` raised from ``sitecustomize``, prints one line, and lets the
#: interpreter carry on -- which is exactly how a gang that never formed became
#: a workload that "succeeded".
terminate_rank: tp.Callable[[int], tp.NoReturn] = os._exit


#: ``jax.distributed.initialize`` tuning arguments that not every JAX release
#: accepts. The gang-identity arguments are required and always passed; these
#: are passed only when the installed JAX names them, so unsupported tuning
#: keywords cannot fail initialization before user code runs.
_OPTIONAL_INITIALIZE_ARGUMENTS = frozenset({"heartbeat_timeout_seconds", "initialization_timeout"})


def _supported_arguments(initialize: tp.Callable[..., tp.Any], arguments: tp.Mapping[str, tp.Any]) -> dict[str, tp.Any]:
    """Drop the optional tuning arguments the installed ``initialize`` does not take."""

    import inspect

    try:
        parameters = inspect.signature(initialize).parameters
    except (TypeError, ValueError):
        return dict(arguments)
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return dict(arguments)
    supported = {}
    for name, value in arguments.items():
        if name in parameters or name not in _OPTIONAL_INITIALIZE_ARGUMENTS:
            supported[name] = value
        else:
            print(
                f"eyvan: this JAX's distributed.initialize() does not accept {name}; joining without it",
                file=sys.stderr,
                flush=True,
            )
    return supported


def _initialize_distributed(
    distributed: tp.Any,
    arguments: tp.Mapping[str, tp.Any],
    *,
    deadline_s: int,
    fatal: bool,
) -> None:
    """Join the gang, reporting how long the wait actually was.

    Args:
        distributed: ``jax.distributed``.
        arguments: Keyword arguments for ``jax.distributed.initialize``.
        deadline_s: The initialization deadline being applied, for reporting.
        fatal: Whether a failed join ends this rank's workload.

    Raises:
        BaseException: Whatever JAX raised, unchanged, when ``fatal`` is off.
    """

    rank, processes = _rank_identity()
    started = time.monotonic()
    try:
        distributed.initialize(**_supported_arguments(distributed.initialize, arguments))
    except (KeyboardInterrupt, SystemExit):
        # An interrupted rank is not a rank that could not join. Claiming
        # otherwise would send an operator to lengthen a deadline that was never
        # reached, over a workload somebody stopped on purpose.
        raise
    except BaseException as error:
        waited = time.monotonic() - started
        outcome = "deadline" if waited >= deadline_s * _DEADLINE_FRACTION else "error"
        detail = f"{type(error).__name__}: {error}".replace("\n", " ").replace("\r", " ")[:_ERROR_DETAIL_LIMIT]
        _emit(
            f"{GANG_INITIALIZATION_FAILED_MARKER} rank={rank} processes={processes} "
            f"waited_s={waited:.1f} deadline_s={deadline_s} outcome={outcome} error={detail}"
        )
        if fatal:
            terminate_rank(
                GANG_INITIALIZATION_DEADLINE_EXIT_CODE if outcome == "deadline" else GANG_INITIALIZATION_ERROR_EXIT_CODE
            )
        raise
    _emit(
        f"{GANG_INITIALIZATION_JOINED_MARKER} rank={rank} processes={processes} "
        f"waited_s={time.monotonic() - started:.1f} deadline_s={deadline_s}"
    )


def initialize_worker_runtime(environment: tp.MutableMapping[str, str] | None = None) -> None:
    """Arm locality guards, prepare the environment, and initialize JAX, in that order.

    The order is the point and it used to live in a generated shell heredoc,
    where it could only ever be asserted as a *string*. Under ``gcetpu``
    detection, JAX reads topology from GCE metadata, so any inherited override
    must be removed *before* initialization -- afterwards it has already been
    read. Here it is one Python function a test can actually run.
    """

    target = os.environ if environment is None else environment
    _arm_cross_region_guard(target)
    if target.get("EYVAN_JAX_CLUSTER_DETECTION") == "gcetpu":
        _scrub_gce_topology_overrides(target)
    initialize_from_environment()


def _consume_multihost_marker() -> None:
    """Stop this process's children from re-joining the JAX cluster.

    ``sitecustomize`` runs in *every* Python interpreter that inherits
    ``PYTHONPATH``, and the workload's environment carries both that path and
    ``EYVAN_JAX_MULTIHOST=1``. So any Python subprocess the workload spawns --
    and ``gcloud`` is itself a Python program -- would load this module, see the
    marker, and call ``jax.distributed.initialize()`` carrying its *parent's*
    ``JAX_PROCESS_ID``. It joins the coordination service as a duplicate task,
    finishes its work, and exits. The service sees a task die and tears the
    whole cluster down: every rank reports "another task died" and no host has
    died at all.

    Ray never hits this because it injects no ``sitecustomize``; user code calls
    ``initialize()`` once, itself, at a time it chooses.

    Removing the marker from ``os.environ`` after initializing is what breaks the
    inheritance: this process is already joined, and every interpreter it starts
    from here on sees no marker and returns immediately. The ``PYTHONPATH`` entry
    is left alone deliberately -- staged sibling modules are still importable,
    and without the marker this module is a no-op anyway.

    Also removes the earlier symptom: a child with no JAX installed printed
    "Error in sitecustomize ... ModuleNotFoundError: No module named 'jax'",
    because the import is reached only past this guard.
    """

    os.environ.pop("EYVAN_JAX_MULTIHOST", None)


def initialize_from_environment() -> None:
    """Initialize JAX from Eyvan's explicit multi-host environment."""

    if os.getenv("EYVAN_JAX_MULTIHOST") != "1":
        return
    _consume_multihost_marker()
    deadline_s = _timeout(INITIALIZATION_TIMEOUT_ENV, DEFAULT_INITIALIZATION_TIMEOUT_S)
    heartbeat_s = _timeout(HEARTBEAT_TIMEOUT_ENV, DEFAULT_HEARTBEAT_TIMEOUT_S)
    fatal = _fatal_initialization()
    detection_method = os.getenv("EYVAN_JAX_CLUSTER_DETECTION", "explicit")
    if detection_method == "gcetpu":
        import jax

        if not jax.distributed.is_initialized():
            _initialize_distributed(
                jax.distributed,
                {
                    "cluster_detection_method": "gcetpu",
                    "initialization_timeout": deadline_s,
                    "heartbeat_timeout_seconds": heartbeat_s,
                },
                deadline_s=deadline_s,
                fatal=fatal,
            )
        return
    if detection_method != "explicit":
        raise RuntimeError(f"Eyvan does not support JAX cluster detection method {detection_method!r}")
    coordinator_address = os.getenv("JAX_COORDINATOR_ADDRESS")
    process_count = os.getenv("JAX_PROCESS_COUNT")
    process_id = os.getenv("JAX_PROCESS_ID")
    if not coordinator_address or process_count is None or process_id is None:
        raise RuntimeError("Eyvan multi-host JAX workload is missing distributed process configuration")
    try:
        num_processes = int(process_count)
        rank = int(process_id)
    except ValueError as exc:
        raise RuntimeError("Eyvan multi-host JAX process configuration must be integers") from exc
    if num_processes == 1 and rank == 0:
        # An elastic job is one hardware world per cell, and a single-host
        # cell's world is one process: there is nothing to join. This is valid
        # even when the elastic workload spans several such cells.
        return
    if num_processes < 1 or rank < 0 or rank >= num_processes:
        raise RuntimeError("Eyvan multi-host JAX process configuration is invalid")
    import jax

    if not jax.distributed.is_initialized():
        _initialize_distributed(
            jax.distributed,
            {
                "coordinator_address": coordinator_address,
                "num_processes": num_processes,
                "process_id": rank,
                "cluster_detection_method": "deactivate",
                "initialization_timeout": deadline_s,
                "heartbeat_timeout_seconds": heartbeat_s,
            },
            deadline_s=deadline_s,
            fatal=fatal,
        )


__all__ = (
    "CROSS_REGION_TRANSFER_BANNED_EXIT_CODE",
    "CROSS_REGION_TRANSFER_BANNED_MARKER",
    "CROSS_REGION_TRANSFER_BANNED_REASON",
    "DEFAULT_GANG_INITIALIZATION",
    "DEFAULT_HEARTBEAT_TIMEOUT_S",
    "DEFAULT_INITIALIZATION_TIMEOUT_S",
    "FATAL_INITIALIZATION_ENV",
    "GANG_INITIALIZATION_DEADLINE_EXIT_CODE",
    "GANG_INITIALIZATION_ERROR_EXIT_CODE",
    "GANG_INITIALIZATION_FAILED_MARKER",
    "GANG_INITIALIZATION_JOINED_MARKER",
    "GANG_NEVER_FORMED_REASON",
    "HEARTBEAT_TIMEOUT_ENV",
    "INITIALIZATION_TIMEOUT_ENV",
    "GangInitializationPolicy",
    "gang_formation_failure_reason",
    "gang_initialization_outcome",
    "initialize_from_environment",
    "initialize_worker_runtime",
)
