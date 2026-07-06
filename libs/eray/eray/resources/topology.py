# Copyright 2026 The EasyDeL/eray Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Host chip-split planning for running multiple processes on one TPU host.

A TPU host is normally driven by a single process that owns every chip on the
host. The libtpu/PJRT runtime, however, allows a host to be carved into
multiple processes, each owning a disjoint subset of chips, controlled purely
through environment variables (the same mechanism torch_xla uses for its
process-per-chip execution model).

This module computes those environment variables. It is pure planning logic —
no Ray, no TPU — so it is fully testable on CPU. The actual task launching
lives in :class:`eray.pool.device_host.DeviceHostActor.run_split_remote_fn`.

Two split modes are supported:

``isolated``
    Every split is a fully independent runtime: its own process, its own
    chips, no communication with sibling splits. Chip assignment is delegated
    to Ray's TPU accelerator manager, which assigns disjoint chips to
    concurrent tasks requesting fractional ``TPU`` resources and writes
    ``TPU_VISIBLE_CHIPS`` / ``TPU_CHIPS_PER_HOST_BOUNDS`` / ``TPU_HOST_BOUNDS``
    into the task environment. Ray only writes the bounds variables for 1- and
    2-chip subsets, so those are the supported split sizes (plus the trivial
    1-split whole-host case). Use this for independent replicas, e.g. one
    small serving instance per chip.

``cooperative``
    All splits together form one multi-process runtime over the host's chips
    (one process per chip), able to communicate over ICI via
    ``jax.distributed`` — the torch_xla single-host multiprocess layout. The
    shared topology variables (``TPU_PROCESS_BOUNDS``,
    ``TPU_CHIPS_PER_PROCESS_BOUNDS``, ``TPU_PROCESS_ADDRESSES``) are emitted
    statically, while the chip-dependent variables (``TPU_VISIBLE_CHIPS``,
    ``CLOUD_TPU_TASK_ID``, ``TPU_PROCESS_PORT``) are bound inside each task
    from the chip Ray actually assigned — pinning them statically is not
    possible, because Ray resolves its chip assignment *through* the
    process-visible ids at task start and a pre-injected single-chip value
    crashes that lookup. ``RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS`` is set
    so Ray does not additionally write single-chip host-bounds variables
    that would contradict the shared process topology.

Note:
    Hardware-validated on a v5p-8 (4 chips): isolated 1-chip splits, isolated
    2-chip splits, and cooperative one-process-per-chip all initialize and run
    compute concurrently. Constraint found on hardware: 2-chip sub-slices only
    form for aligned chip pairs ({0,1}, {2,3}); other pairs crash libtpu's
    SliceBuilder — the split runners check the assignment and raise instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

SplitModeT = Literal["isolated", "cooperative"]

DEFAULT_SPLIT_BASE_PORT = 8476
"""Default first port for cooperative-mode process endpoints (torch_xla's base)."""

DEFAULT_HOST_CHIP_GRIDS: dict[int, tuple[int, int, int]] = {
    1: (1, 1, 1),
    2: (1, 2, 1),
    4: (2, 2, 1),
    8: (2, 4, 1),
}
"""Physical chip-grid layout of a single host, keyed by chip count.

4 chips per host is the v4/v5p host layout (2x2x1); 8 chips per host is the
v5e/v6e full-host layout (2x4). Callers with exotic topologies can pass an
explicit ``chip_grid`` instead.
"""

_ISOLATED_SUPPORTED_CHIPS_PER_SPLIT = (1, 2)
"""Subset sizes Ray's TPU accelerator manager writes bounds env vars for."""


@dataclass(frozen=True)
class HostSplitPlan:
    """A validated plan for splitting one TPU host across multiple processes.

    Produced by :func:`plan_host_split`; consumed by
    ``DeviceHostActor.run_split_remote_fn`` which launches one Ray task per
    split with the planned environment and a fractional ``TPU`` resource
    request.

    Attributes:
        num_chips: Total chips on the host being split.
        num_splits: Number of processes the host is carved into.
        chips_per_split: Chips owned by each split (``num_chips // num_splits``).
        mode: ``"isolated"`` (independent runtimes) or ``"cooperative"``
            (one multi-process runtime over all splits).
        env_per_split: Environment variables for each split, index-aligned
            with launch order. Always includes ``ERAY_SPLIT_ID``,
            ``ERAY_NUM_SPLITS``, and ``ERAY_SPLIT_MODE`` markers. In
            cooperative mode the chip-dependent variables are absent here —
            they are bound in-process from Ray's chip assignment (and
            ``ERAY_SPLIT_ID`` is rebound to the assigned chip id).
        chip_assignments: Explicit chip ids owned by each split, or None when
            chip assignment is delegated to Ray.
        base_port: First port for cooperative process endpoints; split on
            chip ``c`` listens on ``base_port + c``.
    """

    num_chips: int
    num_splits: int
    chips_per_split: int
    mode: SplitModeT
    env_per_split: tuple[dict[str, str], ...] = field(repr=False)
    chip_assignments: tuple[tuple[int, ...], ...] | None = None
    base_port: int = DEFAULT_SPLIT_BASE_PORT

    def resources_per_split(self) -> dict[str, float]:
        """Ray custom-resource request for one split's task.

        Returns:
            Mapping with a fractional ``TPU`` reservation sized to this
            split's chip count, suitable for ``ray.remote(resources=...)``.
        """
        return {"TPU": float(self.chips_per_split)}


def _split_markers(split_id: int, num_splits: int, mode: SplitModeT) -> dict[str, str]:
    """Build the eray bookkeeping variables common to every split env.

    Args:
        split_id: Zero-based index of the split on its host.
        num_splits: Total number of splits on the host.
        mode: Split mode the plan was built with.

    Returns:
        Environment fragment identifying the split to user code.
    """
    return {
        "ERAY_SPLIT_ID": str(split_id),
        "ERAY_NUM_SPLITS": str(num_splits),
        "ERAY_SPLIT_MODE": mode,
    }


@dataclass(frozen=True)
class HostPartitionPlan:
    """A validated heterogeneous partition of one TPU host across runs.

    Unlike :class:`HostSplitPlan` (uniform splits of one function), a
    partition gives each run its own chip count — e.g. ``(2, 1, 1)`` on a
    4-chip host. All runs are isolated runtimes with chips assigned
    disjointly by Ray's fractional ``TPU`` scheduling. Produced by
    :func:`plan_host_partition`; consumed by
    ``DeviceHostActor.run_swarm_remote_fn``.

    Attributes:
        num_chips: Total chips on the host being partitioned.
        chip_counts: Chips owned by each run, index-aligned with run id.
            May sum to less than ``num_chips`` (leftover chips stay idle).
        env_per_run: Environment variables for each run, index-aligned with
            run id: ``ERAY_RUN_ID``, ``ERAY_NUM_RUNS``, ``ERAY_RUN_CHIPS``,
            and ``ERAY_RUN_NAME`` when a name was given.
    """

    num_chips: int
    chip_counts: tuple[int, ...]
    env_per_run: tuple[dict[str, str], ...] = field(repr=False)

    def resources_per_run(self) -> tuple[dict[str, float], ...]:
        """Ray custom-resource requests, one per run.

        Returns:
            Tuple of mappings with each run's fractional ``TPU`` reservation,
            index-aligned with run id.
        """
        return tuple({"TPU": float(c)} for c in self.chip_counts)


def plan_host_partition(
    num_chips: int,
    chip_counts: tuple[int, ...] | list[int],
    *,
    run_names: tuple[str | None, ...] | list[str | None] | None = None,
) -> HostPartitionPlan:
    """Plan a heterogeneous partition of one TPU host's chips across runs.

    Every run is an isolated runtime (see :func:`plan_host_split`); chip
    assignment is delegated to Ray, which requires each run's chip count to
    be a subset Ray can bound (1 or 2 chips) or the whole host.

    Args:
        num_chips: Number of chips on the host (e.g. 4 on a v4/v5p host).
        chip_counts: Chips per run, e.g. ``(2, 1, 1)``. Must sum to at most
            ``num_chips``; leftover chips stay idle.
        run_names: Optional human-readable name per run, surfaced to the run
            as ``ERAY_RUN_NAME``. Must be index-aligned with ``chip_counts``.

    Returns:
        A frozen :class:`HostPartitionPlan` with per-run environments.

    Raises:
        ValueError: If counts are empty/non-positive, a count is neither
            1, 2, nor the whole host, the counts oversubscribe the host, or
            ``run_names`` is misaligned.
    """
    if num_chips <= 0:
        raise ValueError(f"num_chips must be positive, got {num_chips}")
    for c in chip_counts:
        if float(c) != int(c):
            raise ValueError(f"TPU chip counts must be whole chips, got {c}")
    counts = tuple(int(c) for c in chip_counts)
    if not counts:
        raise ValueError("chip_counts must not be empty")
    if run_names is not None and len(run_names) != len(counts):
        raise ValueError(f"run_names has {len(run_names)} entries for {len(counts)} runs")
    for c in counts:
        if c <= 0:
            raise ValueError(f"chip counts must be positive, got {c}")
        if c not in _ISOLATED_SUPPORTED_CHIPS_PER_SPLIT and c != num_chips:
            raise ValueError(
                f"a run of {c} chips on a {num_chips}-chip host is not supported: Ray's TPU accelerator "
                f"manager only writes chip-bounds environment for subsets of "
                f"{_ISOLATED_SUPPORTED_CHIPS_PER_SPLIT} chips (or the whole host)."
            )
    if sum(counts) > num_chips:
        raise ValueError(f"chip_counts {counts} oversubscribe the host: sum {sum(counts)} > {num_chips} chips")

    envs = []
    for i, c in enumerate(counts):
        env = {
            "ERAY_RUN_ID": str(i),
            "ERAY_NUM_RUNS": str(len(counts)),
            "ERAY_RUN_CHIPS": str(c),
        }
        if run_names is not None and run_names[i]:
            env["ERAY_RUN_NAME"] = str(run_names[i])
        envs.append(env)
    return HostPartitionPlan(num_chips, counts, tuple(envs))


def plan_host_split(
    num_chips: int,
    num_splits: int,
    *,
    mode: SplitModeT = "isolated",
    base_port: int = DEFAULT_SPLIT_BASE_PORT,
    chip_grid: tuple[int, int, int] | None = None,
) -> HostSplitPlan:
    """Plan how to carve one TPU host's chips into multiple processes.

    Args:
        num_chips: Number of chips on the host (e.g. 4 on a v4/v5p host).
        num_splits: Number of processes to carve the host into. Must divide
            ``num_chips`` evenly.
        mode: ``"isolated"`` for fully independent per-split runtimes (chip
            assignment delegated to Ray's fractional ``TPU`` scheduling), or
            ``"cooperative"`` for one multi-process runtime with explicit
            process topology (requires exactly one chip per split).
        base_port: First port for cooperative process endpoints; split ``i``
            listens on ``base_port + i``. Unused in isolated mode.
        chip_grid: Physical chip layout of the host as ``(x, y, z)``.
            Defaults to the known layout for ``num_chips``
            (:data:`DEFAULT_HOST_CHIP_GRIDS`). Only consulted in cooperative
            mode, where it becomes ``TPU_PROCESS_BOUNDS``.

    Returns:
        A frozen :class:`HostSplitPlan` with per-split environments.

    Raises:
        ValueError: If counts are non-positive, ``num_splits`` does not divide
            ``num_chips``, an isolated split size is outside Ray's supported
            1/2-chip subsets, or ``chip_grid`` does not match ``num_chips``.
        NotImplementedError: For cooperative splits with more than one chip
            per process (the multi-chip-per-process grid assignment is not a
            verified layout; use one process per chip).
    """
    if num_chips <= 0:
        raise ValueError(f"num_chips must be positive, got {num_chips}")
    if num_splits <= 0:
        raise ValueError(f"num_splits must be positive, got {num_splits}")
    if num_chips % num_splits != 0:
        raise ValueError(f"num_splits={num_splits} must divide num_chips={num_chips} evenly")

    chips_per_split = num_chips // num_splits

    if mode == "isolated":
        envs = tuple(_split_markers(i, num_splits, mode) for i in range(num_splits))
        if num_splits == 1:
            # Whole host in one process — nothing to partition.
            return HostSplitPlan(num_chips, num_splits, chips_per_split, mode, envs, (tuple(range(num_chips)),))
        if chips_per_split not in _ISOLATED_SUPPORTED_CHIPS_PER_SPLIT:
            raise ValueError(
                f"isolated splits of {chips_per_split} chips are not supported: Ray's TPU accelerator "
                f"manager only writes chip-bounds environment for subsets of "
                f"{_ISOLATED_SUPPORTED_CHIPS_PER_SPLIT} chips. Choose num_splits so each split gets 1 or 2 chips."
            )
        # Chip ids are assigned by Ray at schedule time (disjoint across
        # concurrent tasks); the plan intentionally does not pin them.
        return HostSplitPlan(num_chips, num_splits, chips_per_split, mode, envs, None)

    if mode == "cooperative":
        if chips_per_split != 1:
            raise NotImplementedError(
                f"cooperative mode supports exactly one chip per split (got {chips_per_split}); "
                f"the one-process-per-chip layout is the only verified process topology. "
                f"Use num_splits={num_chips}."
            )
        grid = chip_grid if chip_grid is not None else DEFAULT_HOST_CHIP_GRIDS.get(num_chips)
        if grid is None:
            raise ValueError(
                f"no default chip grid known for a {num_chips}-chip host; pass chip_grid=(x, y, z) explicitly"
            )
        if len(grid) != 3 or any(g <= 0 for g in grid):
            raise ValueError(f"chip_grid must be three positive ints, got {grid}")
        if math.prod(grid) != num_chips:
            raise ValueError(f"chip_grid {grid} covers {math.prod(grid)} chips, expected {num_chips}")

        process_bounds = ",".join(str(g) for g in grid)
        # Endpoint list is ordered by chip id; each task binds its own
        # TPU_PROCESS_PORT / CLOUD_TPU_TASK_ID / TPU_VISIBLE_CHIPS in-process
        # once Ray has assigned it a chip.
        addresses = ",".join(f"localhost:{base_port + c}" for c in range(num_chips))
        shared = {
            # Ray must not write single-chip TPU_CHIPS_PER_HOST_BOUNDS /
            # TPU_HOST_BOUNDS — they would contradict the shared process grid.
            "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS": "1",
            "TPU_PROCESS_BOUNDS": process_bounds,
            "TPU_CHIPS_PER_PROCESS_BOUNDS": "1,1,1",
            "TPU_PROCESS_ADDRESSES": addresses,
        }
        envs = tuple({**_split_markers(i, num_splits, mode), **shared} for i in range(num_splits))
        return HostSplitPlan(num_chips, num_splits, chips_per_split, mode, envs, None, base_port)

    raise ValueError(f"unknown split mode {mode!r}; expected 'isolated' or 'cooperative'")
