# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Environment ownership for Eyvan-managed distributed runtimes.

Two families of topology variable live here and they are deliberately kept
apart.

:data:`DISTRIBUTED_TOPOLOGY_ENVIRONMENT_NAMES` is the TPU/JAX family. It is
reserved at *submission* time -- ``records.WorkloadDefinition`` refuses a
workload that names any of them -- as well as stripped at dispatch.

:data:`GPU_TOPOLOGY_ENVIRONMENT_NAMES` is the GPU family, and it is only
stripped at dispatch, on the path that then replaces it. The difference is not
an oversight. Reserving these names at submission would start refusing workloads
that a running control plane accepts today -- ``CUDA_VISIBLE_DEVICES`` in a
submitted environment is ordinary and harmless on every non-GPU-coordinated
path -- and a rolling upgrade that rejects a previously valid submission is a
worse failure than the one being prevented. What matters is that a *coordinated*
rank never inherits one, and that is a property of the dispatch path, which owns
the value it writes. A workload that is not GPU-coordinated sees exactly what it
sees today.

**Nothing here is hardware-qualified.** No GPU has executed an Eyvan workload.
The variables below are the ones a GPU runtime reads; that they are computed and
delivered correctly is host-testable, and is all that is claimed.
"""

from __future__ import annotations

import re
import typing as tp
from dataclasses import dataclass

from eyvan._internal.compat import StrEnum

DISTRIBUTED_TOPOLOGY_ENVIRONMENT_NAMES = frozenset(
    {
        "EYVAN_DISTRIBUTED_PROCESS_COUNT",
        "EYVAN_DISTRIBUTED_PROCESS_ID",
        "EYVAN_JAX_CLUSTER_DETECTION",
        "EYVAN_JAX_MULTIHOST",
        "EYVAN_REPLICA_COUNT",
        "EYVAN_REPLICA_INDEX",
        "GCE_METADATA_IP",
        "JAX_COORDINATOR_ADDRESS",
        "JAX_COORDINATOR_BIND_ADDRESS",
        "JAX_COORDINATOR_PORT",
        "JAX_LOCAL_DEVICE_IDS",
        "JAX_PARTITION_INDEX",
        "JAX_PROCESS_COUNT",
        "JAX_PROCESS_ID",
        "JAX_PROCESS_INDEX",
        "JAX_SLICE_INDEX",
        "MEGASCALE_COORDINATOR_ADDRESS",
        "MEGASCALE_NUM_SLICES",
        "MEGASCALE_SLICE_ID",
        "TPU_PROCESS_ADDRESSES",
        "TPU_SKIP_MDS_QUERY",
        "TPU_WORKER_HOSTNAMES",
        "TPU_WORKER_ID",
    }
)

#: Device-visibility and rendezvous variables a GPU runtime reads. A rank that
#: inherited any of these from a submission would compute on a device another
#: rank owns, or join a rendezvous the control plane never created -- so the
#: GPU-coordinated dispatch path removes every one of them and then writes back
#: only those it can state without guessing (see :func:`gpu_rank_environment`).
#:
#: Deliberately *not* every ``NCCL_`` name: ``NCCL_DEBUG``, ``NCCL_SOCKET_IFNAME``
#: and the transport tunables are an operator's to set and say nothing about
#: which rank a process is. Only ``NCCL_COMM_ID``, which names a rendezvous, is
#: taken.
GPU_TOPOLOGY_ENVIRONMENT_NAMES = frozenset(
    {
        "CUDA_DEVICE_ORDER",
        "CUDA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "GROUP_RANK",
        "HIP_VISIBLE_DEVICES",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "NCCL_COMM_ID",
        "NODE_RANK",
        "NVIDIA_VISIBLE_DEVICES",
        "RANK",
        "ROCR_VISIBLE_DEVICES",
        "WORLD_SIZE",
    }
)

# These host identity and launch-path values are assigned by the native runtime.
# The complete ``EYVAN_`` namespace is reserved below as well: controller
# dispatch, callable/workspace transfer, checkpoint/profile collection, Docker,
# and future runtime features must be able to inject their own variables without
# a secret mount being silently overwritten after its value is materialized.
RUNTIME_OWNED_ENVIRONMENT_NAMES = frozenset(
    {
        "HOME",
        "LOGNAME",
        "PATH",
        "USER",
        "VIRTUAL_ENV",
        "XDG_CACHE_HOME",
    }
    | DISTRIBUTED_TOPOLOGY_ENVIRONMENT_NAMES
)


def is_distributed_topology_environment_name(name: str) -> bool:
    """Return whether Eyvan, JAX, or GCP owns a topology variable."""

    return name in DISTRIBUTED_TOPOLOGY_ENVIRONMENT_NAMES or name.startswith(("EYVAN_JAX_", "MEGASCALE_"))


def is_runtime_owned_environment_name(name: str) -> bool:
    """Return whether a secret mount would collide with runtime-owned state.

    Eyvan owns its entire environment namespace rather than an enumerated
    snapshot. This keeps a newly injected checkpoint, callable, source, or
    container variable from becoming an overwrite of an already-materialized
    secret in older submission code.
    """

    return (
        name in RUNTIME_OWNED_ENVIRONMENT_NAMES
        or name.startswith("EYVAN_")
        or is_distributed_topology_environment_name(name)
    )


def without_distributed_topology_overrides(environment: tp.Mapping[str, str]) -> dict[str, str]:
    """Copy an environment without caller-controlled topology variables."""

    return {name: value for name, value in environment.items() if not is_distributed_topology_environment_name(name)}


def is_gpu_topology_environment_name(name: str) -> bool:
    """Return whether Eyvan owns a GPU device-visibility or rendezvous variable."""

    return name in GPU_TOPOLOGY_ENVIRONMENT_NAMES


def without_gpu_topology_overrides(environment: tp.Mapping[str, str]) -> dict[str, str]:
    """Copy an environment without caller-controlled GPU topology variables.

    Applied only where the caller is about to be given Eyvan's own values. A
    submission that carried ``CUDA_VISIBLE_DEVICES=0`` on every rank would put
    every rank of a gang on device zero of its host and leave the rest idle;
    stripping first is what makes the assignment below authoritative rather than
    a suggestion the submission can override.
    """

    return {name: value for name, value in environment.items() if not is_gpu_topology_environment_name(name)}


#: Rendezvous port for the JAX coordinator every native gang forms. Held here,
#: beside the GPU port, because the two are one operator-visible fact: these are
#: the ports a deployment's firewall must open *between gang hosts*, and
#: ``worker/runtime/registry.py`` records what happens when they are not -- every
#: rank died in ``RegisterTask`` with DEADLINE_EXCEEDED against an address no
#: rule permitted. Neither number is derived from a workload, so a firewall
#: written once keeps working; that is the property, and it is why the collision
#: below is refused rather than dodged by moving a port.
DEFAULT_NATIVE_COORDINATOR_PORT = 12345

#: Rendezvous port for GPU collectives. Deliberately not the JAX coordinator
#: port: a workload is free to initialize JAX *and* a torch process group, and
#: two services asking for one port is a failure that would look like a network
#: fault. Eyvan owns both numbers, so it can simply not collide with itself.
DEFAULT_GPU_RENDEZVOUS_PORT = 12346

#: Every port a native gang's rendezvous occupies on its coordinator's host, in
#: the order an operator opens them.
#:
#: Both are constants, so two gangs whose coordinators land on one machine are
#: handed the *same* endpoint and a rank of one can join the other's coordinator.
#: Nothing in this module can see a second gang, so nothing here can detect that;
#: what this tuple exists for is to let the one component that can -- gang
#: dispatch, in ``server/controller/workload_commands.py`` -- name the exact
#: sockets it is refusing to hand out twice, from this definition rather than
#: from a second copy of the numbers.
NATIVE_RENDEZVOUS_PORTS: tuple[int, ...] = (DEFAULT_NATIVE_COORDINATOR_PORT, DEFAULT_GPU_RENDEZVOUS_PORT)


class GpuRankLayout(StrEnum):
    """What a "rank" is, and therefore what ``WORLD_SIZE`` counts.

    Two readings of one gang are both legitimate and they are not compatible,
    so the deployment picks one instead of the code guessing.

    :attr:`HOST_LAUNCHER` is what Eyvan has always dispatched. A rank is a
    host-local *launcher*: it is handed every device Eyvan gave that host and is
    expected to fan out to one worker per device, so ``WORLD_SIZE`` counts
    devices, ``LOCAL_WORLD_SIZE`` counts this host's devices, and ``RANK`` and
    ``LOCAL_RANK`` are deliberately unset -- they name a worker process inside a
    launcher Eyvan does not run. It follows that a host can hold only one rank:
    two launchers on one host have no way to say which of them is which.

    :attr:`RANK_PROCESS` says a rank is the process Eyvan actually starts. That
    is the reading under which several ranks can share a host: each gets its own
    slice of that host's devices, ``LOCAL_RANK`` names which slice, and
    ``WORLD_SIZE`` counts processes, because a ``RANK`` that does not index into
    ``WORLD_SIZE`` is how a collective hangs instead of failing.

    The device facts do not move between the two: ``CUDA_VISIBLE_DEVICES`` and
    the ``EYVAN_GPU_*`` device counts mean the same thing in both, so nothing a
    workload needs to know about hardware is lost by choosing one.

    Neither layout is hardware-qualified. No GPU has executed an Eyvan workload.
    """

    HOST_LAUNCHER = "host-launcher"
    RANK_PROCESS = "rank-process"


#: The layout every dispatch used before this existed, and the default
#: everywhere. Opting in is a deployment decision because it changes
#: ``WORLD_SIZE`` from a device count to a process count.
DEFAULT_GPU_RANK_LAYOUT = GpuRankLayout.HOST_LAUNCHER


@dataclass(frozen=True, slots=True)
class GpuDeviceAssignment:
    """The devices one rank of a GPU gang may use, host-local and gang-wide.

    ``host_devices`` are the indices the rank's process sees, which is what
    ``CUDA_VISIBLE_DEVICES`` takes. ``global_devices`` are the same devices'
    positions in the gang-wide device order, which is what makes "exact and
    non-overlapping" checkable: two ranks on different hosts both see local
    index ``0``, and only the global ordinals can say whether that is one device
    or two.

    ``host_index`` and ``local_rank`` are what let a host carry more than one
    rank. They are separate from ``rank`` because the three answer different
    questions -- which process am I in the gang, which machine am I on, and
    which process am I on that machine -- and they coincide only while every
    host holds exactly one rank, which is the shape Eyvan dispatched before
    per-host grouping existed.

    Attributes:
        rank: The gang rank this assignment belongs to.
        host_devices: Host-local device indices, ascending.
        global_devices: Gang-wide device ordinals, ascending and contiguous.
        host_index: Position of this rank's host in the gang's host order.
        local_rank: Position of this rank among the ranks on its own host.
        local_rank_count: How many ranks share this rank's host.
    """

    rank: int
    host_devices: tuple[int, ...]
    global_devices: tuple[int, ...]
    host_index: int
    local_rank: int
    local_rank_count: int

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise ValueError("a GPU device assignment rank cannot be negative")
        if not self.host_devices or len(self.host_devices) != len(self.global_devices):
            raise ValueError("a GPU device assignment must map every host device to one global device")
        if self.host_index < 0:
            raise ValueError("a GPU device assignment host index cannot be negative")
        if self.local_rank_count < 1:
            raise ValueError("a GPU device assignment host must carry at least one rank")
        if not 0 <= self.local_rank < self.local_rank_count:
            raise ValueError("a GPU device assignment local rank must index the ranks on its own host")

    @property
    def device_count(self) -> int:
        """Return how many devices this rank owns."""

        return len(self.host_devices)

    @property
    def global_device_offset(self) -> int:
        """Return this rank's first gang-wide device ordinal."""

        return self.global_devices[0]


@dataclass(frozen=True, slots=True)
class GpuDevicePlan:
    """Every rank's device assignment for one GPU gang.

    Built by :func:`plan_gpu_devices` and never by hand, so the non-overlap
    property is established once rather than re-argued at each call site.
    """

    assignments: tuple[GpuDeviceAssignment, ...]

    @property
    def host_count(self) -> int:
        """Return how many distinct hosts this plan covers.

        Not the number of ranks. The two are equal exactly while every host
        holds one rank, which is the only shape that existed before per-host
        grouping, so this keeps reporting what it always reported there.
        """

        return len({assignment.host_index for assignment in self.assignments})

    @property
    def rank_count(self) -> int:
        """Return how many ranks -- dispatched processes -- this plan covers."""

        return len(self.assignments)

    @property
    def device_count(self) -> int:
        """Return how many devices this plan covers in total."""

        return sum(assignment.device_count for assignment in self.assignments)

    @property
    def shares_a_host(self) -> bool:
        """Return whether any host in this plan carries more than one rank."""

        return any(assignment.local_rank_count > 1 for assignment in self.assignments)

    def rank_environment(
        self,
        rank: int,
        *,
        rendezvous_host: str,
        rendezvous_port: int = DEFAULT_GPU_RENDEZVOUS_PORT,
        gpu_model: str | None = None,
        layout: GpuRankLayout = DEFAULT_GPU_RANK_LAYOUT,
    ) -> dict[str, str]:
        """Return the GPU coordination environment Eyvan owns for one rank.

        What is written back is only what Eyvan can state without guessing. The
        device visibility, the device counts, the host index and the rendezvous
        endpoint follow from the placement Eyvan made, and are identical under
        both layouts.

        What differs is who a "rank" is, which is :class:`GpuRankLayout`'s whole
        subject. Under :attr:`~GpuRankLayout.HOST_LAUNCHER` -- the default, and
        every dispatch before per-host grouping existed -- ``RANK`` and
        ``LOCAL_RANK`` stay unset because they name a worker inside a launcher
        Eyvan does not run, and filling them with the host's index would be
        right for the first worker and wrong for every other one. Under
        :attr:`~GpuRankLayout.RANK_PROCESS` the rank *is* the process Eyvan
        started, so both are facts about Eyvan's own dispatch and are written,
        along with the ``WORLD_SIZE`` they have to index into.

        Args:
            rank: The gang rank to render.
            rendezvous_host: Host part of the collective rendezvous endpoint.
            rendezvous_port: Port part of that endpoint.
            gpu_model: The attached accelerator, when the catalog records one.
            layout: What a rank is in this deployment.

        Returns:
            A mapping of environment variable names to values.

        Raises:
            IndexError: If ``rank`` is not in this plan.
            ValueError: If the rendezvous host is empty, or if a host carries
                several ranks under a layout that has no name for the second
                one.
        """

        if not rendezvous_host or rendezvous_host != rendezvous_host.strip():
            raise ValueError("a GPU rendezvous host must be non-empty and trimmed")
        if not isinstance(layout, GpuRankLayout):
            raise TypeError("a GPU rank environment layout must be a GpuRankLayout")
        if layout is GpuRankLayout.HOST_LAUNCHER and self.shares_a_host:
            # Refused rather than rendered: both ranks would be told they are
            # node zero of that host and be handed the launcher contract, under
            # which nothing distinguishes them.
            raise ValueError(
                f"the {GpuRankLayout.HOST_LAUNCHER.value!r} GPU rank layout has no name for a second rank on one "
                f"host; use {GpuRankLayout.RANK_PROCESS.value!r} to place several ranks per host"
            )
        assignment = self.assignments[rank]
        visible = ",".join(str(index) for index in assignment.host_devices)
        environment = {
            # PCI bus order rather than CUDA's default "fastest first", so the
            # index a rank is given means the same device on every host of a
            # heterogeneous gang and matches what nvidia-smi reports.
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": visible,
            "NVIDIA_VISIBLE_DEVICES": visible,
        }
        if layout is GpuRankLayout.RANK_PROCESS:
            environment.update(
                {
                    "RANK": str(assignment.rank),
                    "LOCAL_RANK": str(assignment.local_rank),
                    # Processes on this host, not devices on it. The device count
                    # is still stated, under EYVAN_GPU_LOCAL_DEVICE_COUNT below.
                    "LOCAL_WORLD_SIZE": str(assignment.local_rank_count),
                    "WORLD_SIZE": str(self.rank_count),
                    "GROUP_RANK": str(assignment.host_index),
                    "NODE_RANK": str(assignment.host_index),
                }
            )
        else:
            environment.update(
                {
                    "LOCAL_WORLD_SIZE": str(assignment.device_count),
                    "NODE_RANK": str(assignment.rank),
                    "WORLD_SIZE": str(self.device_count),
                }
            )
        environment.update(
            {
                "MASTER_ADDR": rendezvous_host,
                "MASTER_PORT": str(rendezvous_port),
                "EYVAN_GPU_HOST_COUNT": str(self.host_count),
                "EYVAN_GPU_HOST_INDEX": str(assignment.host_index),
                "EYVAN_GPU_LOCAL_DEVICE_COUNT": str(assignment.device_count),
                "EYVAN_GPU_GLOBAL_DEVICE_COUNT": str(self.device_count),
                "EYVAN_GPU_GLOBAL_DEVICE_OFFSET": str(assignment.global_device_offset),
            }
        )
        if gpu_model is not None:
            environment["EYVAN_GPU_MODEL"] = gpu_model
        return environment


def gpu_rank_placement_error(
    devices_per_rank: tp.Sequence[int],
    rank_hosts: tp.Sequence[str],
) -> str | None:
    """Return why these ranks cannot be given disjoint device slices, or ``None``.

    Separate from :func:`plan_gpu_devices` because two callers need the same
    answer for opposite reasons. A hand-built gang is *refused* when it cannot
    be placed, because its author stated something impossible; a gang the
    resolver assembles from observations is *downgraded* to the uncoordinated
    dispatch it gets today, because partial or awkward knowledge about a fleet
    is ordinary and failing the dispatch would break a workload that runs now.
    Both need the reason, and neither should re-derive it.

    Args:
        devices_per_rank: Device count of each rank's host, in rank order.
        rank_hosts: The host each rank lands on, in the same order. Ranks
            sharing a value share a machine.

    Returns:
        An operator-readable reason, or ``None`` when every host's devices
        divide evenly among the ranks placed on it.
    """

    if len(devices_per_rank) != len(rank_hosts):
        return f"{len(rank_hosts)} rank hosts cannot describe {len(devices_per_rank)} ranks"
    declared: dict[str, int] = {}
    ranks_per_host: dict[str, int] = {}
    for rank, (host, count) in enumerate(zip(rank_hosts, devices_per_rank, strict=True)):
        if not isinstance(host, str) or not host.strip() or host != host.strip():
            return f"rank {rank} must name a non-empty trimmed host, not {host!r}"
        ranks_per_host[host] = ranks_per_host.get(host, 0) + 1
        if host in declared and declared[host] != count:
            # One machine cannot carry two different numbers of devices, so this
            # is two sources disagreeing and there is nothing here that can
            # arbitrate. Splitting either number would pin a rank to devices
            # chosen by the loser of a coin flip.
            return f"host {host!r} is declared with both {declared[host]} and {count} GPUs"
        declared[host] = count
    for host, count in declared.items():
        ranks = ranks_per_host[host]
        if count % ranks:
            # An uneven split is expressible and is still refused: the leftover
            # devices would either sit idle behind a plan that claims to cover
            # the host, or make one rank larger than its peers, and a collective
            # sized by the smallest rank would then be silently wrong.
            return f"host {host!r} carries {count} GPUs, which {ranks} ranks cannot divide evenly"
    return None


def plan_gpu_devices(
    devices_per_rank: tp.Sequence[int],
    *,
    rank_hosts: tp.Sequence[str] | None = None,
) -> GpuDevicePlan:
    """Assign every gang rank an exact, non-overlapping set of GPU devices.

    Without ``rank_hosts`` this is one rank per host, which is the only shape
    Eyvan dispatched before per-host grouping: the native worker runs one
    process per endpoint, and a host-local launcher inside it fans out to the
    devices this function made visible. Hosts may hold different numbers of
    devices -- a single-GPU host and an eight-GPU host in one gang is the
    heterogeneous case -- so the gang-wide ordinals are accumulated rather than
    assumed uniform.

    With ``rank_hosts``, ranks that name the same host split that host's devices
    between them: the first rank on a host takes its first share, the second the
    next, and so on. ``devices_per_rank`` still states what the *host* carries,
    not what the rank ends up with -- it is read from one catalog row per cell,
    and the row describes a machine. A plan built with all-distinct hosts is
    identical to one built without them, which is what keeps the shape that
    ships today unchanged.

    Args:
        devices_per_rank: Device count of each rank's host, in rank order.
        rank_hosts: The host each rank lands on, in rank order, or ``None`` when
            no rank shares a host with another.

    Returns:
        The plan covering every rank.

    Raises:
        ValueError: If no rank was given, if any host's device count is not a
            positive integer, or if the ranks placed on one host cannot divide
            its devices. A host whose device count is unknown is refused rather
            than treated as zero: an unobservable device count is not evidence
            of an empty host, and a zero would silently hand the rank an empty
            ``CUDA_VISIBLE_DEVICES``, which means "no GPU at all".
    """

    counts = tuple(devices_per_rank)
    if not counts:
        raise ValueError("a GPU device plan requires at least one rank")
    for rank, count in enumerate(counts):
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"rank {rank} must declare a positive integer GPU device count, not {count!r}")
    # Without hosts, every rank is keyed by itself, which is the same statement
    # as "no rank shares a machine" and keeps one code path below instead of two.
    hosts = tuple(rank_hosts) if rank_hosts is not None else tuple(f"rank-{rank}" for rank in range(len(counts)))
    if rank_hosts is not None:
        reason = gpu_rank_placement_error(counts, hosts)
        if reason is not None:
            raise ValueError(f"a GPU device plan cannot place these ranks: {reason}")
    host_indices: dict[str, int] = {}
    ranks_per_host: dict[str, int] = {}
    for host in hosts:
        host_indices.setdefault(host, len(host_indices))
        ranks_per_host[host] = ranks_per_host.get(host, 0) + 1
    assignments: list[GpuDeviceAssignment] = []
    local_ranks: dict[str, int] = {}
    offset = 0
    for rank, (host, count) in enumerate(zip(hosts, counts, strict=True)):
        local_rank = local_ranks.get(host, 0)
        local_ranks[host] = local_rank + 1
        share = count // ranks_per_host[host]
        first = local_rank * share
        assignments.append(
            GpuDeviceAssignment(
                rank=rank,
                host_devices=tuple(range(first, first + share)),
                global_devices=tuple(range(offset, offset + share)),
                host_index=host_indices[host],
                local_rank=local_rank,
                local_rank_count=ranks_per_host[host],
            )
        )
        offset += share
    return GpuDevicePlan(assignments=tuple(assignments))


#: Suffix on a chip-share submission ID naming the slice host that owns its chips.
#: Routing reads it back from the remote ID itself, so a restarted agent or a
#: resolver rebuilt from scratch still sends status, logs and retire to the one
#: host that runs the job. A whole-slice ID (``...-a<n>-g<n>``) never ends this way.
_CHIP_HOST_SUFFIX = re.compile(r"-tpuhost(\d{1,5})$")

#: libtpu's default per-process port; a chip share offsets it by its first chip.
_TPU_PROCESS_BASE_PORT = 8476

#: ``TPU_CHIPS_PER_PROCESS_BOUNDS`` per host-local chip set, measured on a v5p
#: host on 2026-11-02: each pair accepts exactly one orientation and the other
#: aborts libtpu; diagonal pairs {0,2} and {1,3} accept none and are absent.
_CHIP_BOUNDS: dict[tuple[int, ...], str] = {
    (0, 1): "1,2,1",
    (2, 3): "1,2,1",
    (1, 2): "2,1,1",
    (0, 3): "2,1,1",
    (0, 1, 2, 3): "2,2,1",
}


def chip_submission_id(base: str, host: int) -> str:
    """Return the submission ID of a chip share running on slice host ``host``."""

    if isinstance(host, bool) or not isinstance(host, int) or host < 0:
        raise ValueError("chip share host must be a non-negative integer")
    return f"{base}-tpuhost{host}"


def chip_host_of(remote_id: str) -> int | None:
    """Return the slice host a chip-share remote ID runs on, or ``None`` for a whole-slice job."""

    match = _CHIP_HOST_SUFFIX.search(remote_id)
    return None if match is None else int(match.group(1))


def tpu_chip_environment(local_chips: tp.Sequence[int]) -> dict[str, str]:
    """Return the libtpu environment that confines one process to ``local_chips`` of its host.

    The process is its own single-process TPU runtime: process bounds 1x1x1, a
    private libtpu port derived from its first chip (so neighbours on one host
    never share one), and the shared ``/tmp/libtpu_lockfile`` bypassed because
    several runtimes on one host are the point.
    """

    chips = tuple(sorted(local_chips))
    if not chips or len(set(chips)) != len(chips) or any(chip < 0 for chip in chips):
        raise ValueError("a chip share needs unique non-negative local chip indexes")
    bounds = "1,1,1" if len(chips) == 1 else _CHIP_BOUNDS.get(chips)
    if bounds is None:
        raise ValueError(f"chips {list(chips)} have no libtpu process bounds; one process cannot hold them")
    port = _TPU_PROCESS_BASE_PORT + chips[0]
    return {
        "TPU_VISIBLE_CHIPS": ",".join(str(chip) for chip in chips),
        "TPU_CHIPS_PER_PROCESS_BOUNDS": bounds,
        "TPU_PROCESS_BOUNDS": "1,1,1",
        "TPU_PROCESS_PORT": str(port),
        "TPU_PROCESS_ADDRESSES": f"localhost:{port}",
        "CLOUD_TPU_TASK_ID": "0",
        "ALLOW_MULTIPLE_LIBTPU_LOAD": "1",
    }


def remove_gce_tpu_topology_overrides(environment: tp.MutableMapping[str, str]) -> None:
    """Remove values that must come from GCE TPU metadata discovery."""

    for name in DISTRIBUTED_TOPOLOGY_ENVIRONMENT_NAMES:
        if not name.startswith("EYVAN_") and name != "JAX_COORDINATOR_PORT":
            environment.pop(name, None)


__all__ = (
    "DEFAULT_GPU_RANK_LAYOUT",
    "DEFAULT_GPU_RENDEZVOUS_PORT",
    "DEFAULT_NATIVE_COORDINATOR_PORT",
    "DISTRIBUTED_TOPOLOGY_ENVIRONMENT_NAMES",
    "GPU_TOPOLOGY_ENVIRONMENT_NAMES",
    "NATIVE_RENDEZVOUS_PORTS",
    "RUNTIME_OWNED_ENVIRONMENT_NAMES",
    "GpuDeviceAssignment",
    "GpuDevicePlan",
    "GpuRankLayout",
    "chip_host_of",
    "chip_submission_id",
    "gpu_rank_placement_error",
    "is_distributed_topology_environment_name",
    "is_gpu_topology_environment_name",
    "is_runtime_owned_environment_name",
    "plan_gpu_devices",
    "remove_gce_tpu_topology_overrides",
    "tpu_chip_environment",
    "without_distributed_topology_overrides",
    "without_gpu_topology_overrides",
)
