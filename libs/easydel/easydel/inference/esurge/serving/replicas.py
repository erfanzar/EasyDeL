# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Data-parallel replicas of one serving launch, each in its own JAX world.

With ``dp = N > 1`` in ``model.sharding_axis_dims`` the launch is unchanged: every host runs the same
``serving serve --config X.yaml`` and joins the whole slice. When each ``dp`` slice of the deployment's global mesh is
a contiguous box of TPU hosts owned by no other slice, every process works out its replica from that mesh, exchanges
its address through the slice's key-value store, leaves the slice world and re-executes the same command inside its
replica's own sub-slice world. Each replica is then a complete ``dp = 1`` engine (weights, KV and prefix cache,
scheduler, step loop, JAX coordination service) and fails alone. Once its engine is up, replica 0's leader starts the
front: the same command as a host-only process that serves the one OpenAI endpoint over a router whose members are a
request-plane client per replica leader. No replica shares an interpreter with the HTTP and routing work.

When the slices are not separate host boxes (several slices on one host, non-TPU backends, pipeline stages), the
launch keeps the single rank-major engine.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import typing as tp

import numpy as np

DP_AXIS = "dp"
_HOST_KEY = "easydel/esurge/serving/replica_host/{port}/{process}"


def _parse_accelerator_type(value: str | None) -> tuple[str, int | None] | None:
    """Parse a ``TPU_ACCELERATOR_TYPE`` string like ``'v5litepod-16'`` into ``(prefix, total_cores)``."""
    if not value:
        return None
    raw = str(value).strip().lower()
    if not raw:
        return None
    prefix, sep, count = raw.rpartition("-")
    if sep and prefix and count.isdigit():
        return prefix, int(count)
    return raw, None


def _generation_from_prefix(prefix: str) -> tuple[str, int] | None:
    """Return ``(generation_family, physical_cores_per_chip)`` for an accelerator-type prefix."""
    norm = prefix.strip().lower().replace("_", "-")
    for token, family, cores in (
        ("v6e", "v6e", 1),
        ("ct6e", "v6e", 1),
        ("trillium", "v6e", 1),
        ("v5litepod", "v5e", 1),
        ("v5lite", "v5e", 1),
        ("v5e", "v5e", 1),
        ("ct5e", "v5e", 1),
        ("ct5lp", "v5e", 1),
        ("v5p", "v5p", 2),
        ("ct5p", "v5p", 2),
        ("v4i", "v4i", 1),
        ("v4lite", "v4i", 1),
        ("v4", "v4", 2),
        ("ct4", "v4", 2),
        ("v3", "v3", 2),
        ("v2", "v2", 2),
        ("tpu7x", "tpu7x", 2),
        ("v7x", "tpu7x", 2),
        ("tpu7", "tpu7", 2),
        ("v7", "tpu7", 2),
    ):
        if norm == token or norm.startswith(f"{token}-"):
            return family, cores
    return None


def _spec_from_device_kind(device_kind: str | None) -> tuple[str, str, int] | None:
    """Map a JAX ``device.device_kind`` string to ``(accelerator_prefix, family, cores_per_chip)``."""
    if not device_kind:
        return None
    norm = " ".join(str(device_kind).strip().lower().split())
    if not norm:
        return None
    if "v6e" in norm or "v6 lite" in norm or "trillium" in norm:
        return "v6e", "v6e", 1
    if "v5 lite" in norm or "v5e" in norm or "v5lite" in norm:
        return "v5litepod", "v5e", 1
    if "v5p" in norm or norm == "tpu v5":
        return "v5p", "v5p", 2
    if "v4 lite" in norm or "v4i" in norm:
        return "v4i", "v4i", 1
    if "v4" in norm:
        return "v4", "v4", 2
    if "v3" in norm:
        return "v3", "v3", 2
    if "v2" in norm:
        return "v2", "v2", 2
    if "7x" in norm:
        return "tpu7x", "tpu7x", 2
    if "tpu7" in norm or "tpu 7" in norm:
        return "tpu7", "tpu7", 2
    return None


def _infer_accelerator_spec(
    all_devices: tp.Sequence[tp.Any],
    local_devices: tp.Sequence[tp.Any],
    slice_extent: tp.Sequence[int],
    chips_per_host: tp.Sequence[int],
    env: tp.Mapping[str, str] | None = None,
) -> tuple[str, int]:
    """Derive ``(accelerator_prefix, cores_per_chip)`` from live devices and inherited ``TPU_ACCELERATOR_TYPE``."""
    slice_chips = max(1, int(np.prod(slice_extent)))
    host_chips = max(1, int(np.prod(chips_per_host)))
    devices_per_chip = max(1, len(local_devices) // host_chips) if local_devices else 1

    device_kind: str | None = None
    for device in (*local_devices, *all_devices):
        kind = getattr(device, "device_kind", None)
        if isinstance(kind, str) and kind.strip():
            device_kind = kind
            break

    source_env = os.environ if env is None else env
    env_parsed = _parse_accelerator_type(source_env.get("TPU_ACCELERATOR_TYPE"))
    dev_spec = _spec_from_device_kind(device_kind)

    if dev_spec is not None:
        dev_prefix, dev_family, dev_cores = dev_spec
        if env_parsed is not None:
            env_prefix, env_total = env_parsed
            env_gen = _generation_from_prefix(env_prefix)
            if env_gen is not None and env_gen[0] == dev_family:
                if env_total is not None and env_total > 0 and env_total % slice_chips == 0:
                    return env_prefix, max(1, env_total // slice_chips)
                return env_prefix, max(dev_cores, devices_per_chip)
        return dev_prefix, max(dev_cores, devices_per_chip)

    if env_parsed is not None:
        env_prefix, env_total = env_parsed
        if env_total is not None and env_total > 0 and env_total % slice_chips == 0:
            return env_prefix, max(1, env_total // slice_chips)
        env_gen = _generation_from_prefix(env_prefix)
        if env_gen is not None:
            return env_prefix, max(env_gen[1], devices_per_chip)
        return env_prefix, devices_per_chip

    return "v5p", max(2, devices_per_chip)


@dataclasses.dataclass(frozen=True)
class BoxLayout:
    """A replica's devices as a contiguous box of the slice's chip torus.

    Attributes:
        origin: Lowest chip coordinate of the box.
        extent: Chips per dimension of the box.
        slice_extent: Chips per dimension of the whole slice (a box dimension spanning it keeps its wraparound).
        chips_per_host: Chips per dimension of one host.
        accelerator_prefix: TPU accelerator-type prefix (e.g. ``"v5p"``, ``"v4"``, ``"v5litepod"``, ``"v6e"``).
            When omitted, derived from ``TPU_ACCELERATOR_TYPE`` in the environment.
        cores_per_chip: Physical TensorCores per chip counted in ``TPU_ACCELERATOR_TYPE`` (2 for v2/v3/v4/v5p,
            1 for v5e/v5litepod/v6e).
    """

    origin: tuple[int, int, int]
    extent: tuple[int, int, int]
    slice_extent: tuple[int, int, int]
    chips_per_host: tuple[int, int, int]
    accelerator_prefix: str | None = None
    cores_per_chip: int | None = None

    @property
    def host_bounds(self) -> tuple[int, int, int]:
        return tuple(e // c for e, c in zip(self.extent, self.chips_per_host, strict=True))  # type: ignore[return-value]

    def host_position(self, chip: tp.Sequence[int]) -> tuple[int, int, int]:
        """(z, y, x) position, inside the box, of the host owning ``chip``: libtpu's worker order."""
        x, y, z = ((int(chip[i]) - self.origin[i]) // self.chips_per_host[i] for i in range(3))
        return (z, y, x)

    def libtpu_env(
        self,
        hostnames: tp.Sequence[str],
        rank: int,
        *,
        env: tp.Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """The libtpu environment that makes this box its own TPU slice, for the worker at ``rank``."""
        chips = int(np.prod(self.extent))
        if self.accelerator_prefix is not None and self.cores_per_chip is not None:
            prefix, cores_per_chip = self.accelerator_prefix, int(self.cores_per_chip)
        else:
            prefix, cores_per_chip = _infer_accelerator_spec((), (), self.slice_extent, self.chips_per_host, env=env)
        return {
            "TPU_WORKER_HOSTNAMES": ",".join(hostnames),
            "TPU_WORKER_ID": str(int(rank)),
            "TPU_SKIP_MDS_QUERY": "1",
            "TPU_ACCELERATOR_TYPE": f"{prefix}-{cores_per_chip * chips}",
            "TPU_TOPOLOGY": "x".join(str(v) for v in self.extent),
            "TPU_TOPOLOGY_WRAP": ",".join(
                "true" if e == s else "false" for e, s in zip(self.extent, self.slice_extent, strict=True)
            ),
            "TPU_HOST_BOUNDS": ",".join(str(v) for v in self.host_bounds),
            "TPU_CHIPS_PER_HOST_BOUNDS": ",".join(str(v) for v in self.chips_per_host),
        }


@dataclasses.dataclass(frozen=True)
class ReplicaPlan:
    """This process's place in its replica's JAX world, handed across the re-exec.

    Attributes:
        index: Replica index.
        rank: This process's index in the replica world.
        num_processes: Processes in the replica world.
        coordinator: ``host:port`` of the replica world's JAX coordination service (its rank 0).
        leaders: Address of every replica's planned rank 0 (its JAX coordinator), by replica index.
        token_fd: Inherited pipe holding the shared request-plane token (never on the command line).
        role: ``"engine"`` for a replica process, ``"front"`` for the process serving the OpenAI endpoint.
        hosts: Every replica's host addresses in plan order, by replica index. libtpu numbers a replica world's
            processes by its own host order, which need not match the plan's, so the engine leader
            (``jax.process_index() == 0``) can be any of these hosts; the front finds it by asking each.
    """

    index: int
    rank: int
    num_processes: int
    coordinator: str
    leaders: tuple[str, ...]
    token_fd: int = -1
    role: str = "engine"
    hosts: tuple[tuple[str, ...], ...] = ()
    front_index: int = 0
    front_socket_dir: str = ""

    @property
    def count(self) -> int:
        return len(self.leaders)

    @property
    def is_front(self) -> bool:
        """Whether this process is the front: no TPU work, a router over every replica, the OpenAI endpoint."""
        return self.role == "front"

    @property
    def spawns_front(self) -> bool:
        """Whether this process is in the replica whose engine leader starts the front (replica 0).

        Only the process that turns out to lead replica 0's engine starts it; which process that is is known only
        inside the replica world (``jax.process_index() == 0``), not from :attr:`rank`.
        """
        return self.role == "engine" and self.index == 0

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_json(cls, text: str) -> ReplicaPlan:
        data = json.loads(text)
        data["leaders"] = tuple(data["leaders"])
        data["hosts"] = tuple(tuple(h) for h in data.get("hosts", ()))
        return cls(**data)

    def read_token(self) -> str:
        """Read (once) the request-plane token the slice world handed over through the inherited pipe."""
        chunks = []
        while chunk := os.read(self.token_fd, 4096):
            chunks.append(chunk)
        os.close(self.token_fd)
        return b"".join(chunks).decode()


def split_dp_slices(device_mesh: np.ndarray, axis_names: tp.Sequence[str]) -> list[np.ndarray] | None:
    """The devices of each ``dp`` slice of a global mesh (flattened in mesh order), when no process owns two slices.

    Returns:
        One device array per slice, or ``None`` for ``dp <= 1``, no ``dp`` axis, or processes straddling slices.
    """
    names = tuple(axis_names)
    if DP_AXIS not in names:
        return None
    axis = names.index(DP_AXIS)
    devices = np.asarray(device_mesh, dtype=object)
    if int(devices.shape[axis]) <= 1:
        return None
    slices = [np.take(devices, [r], axis=axis).reshape(-1) for r in range(int(devices.shape[axis]))]
    owners = [{int(d.process_index) for d in s} for s in slices]
    if sum(len(o) for o in owners) != len(set().union(*owners)):
        return None
    return slices


def box_layout(box: tp.Sequence[tp.Any], all_devices: tp.Sequence[tp.Any], local_devices: tp.Sequence[tp.Any]):
    """The box a replica's chips form inside the slice torus, or ``None`` when they are not a whole-host box."""
    coords = np.asarray([tuple(d.coords)[:3] for d in box])
    origin = coords.min(axis=0)
    extent = coords.max(axis=0) - origin + 1
    unique_coords = {tuple(int(v) for v in c) for c in coords}
    local = np.asarray([tuple(d.coords)[:3] for d in local_devices])
    local_unique = {tuple(int(v) for v in c) for c in local}
    if not local_unique or len(local_devices) % len(local_unique) != 0:
        return None
    devices_per_chip = len(local_devices) // len(local_unique)
    if int(np.prod(extent)) != len(unique_coords) or len(box) != len(unique_coords) * devices_per_chip:
        return None
    full = np.asarray([tuple(d.coords)[:3] for d in all_devices]).max(axis=0) + 1
    per_host = local.max(axis=0) - local.min(axis=0) + 1
    if np.any(extent % per_host) or np.any(origin % per_host):
        return None
    as_tuple = lambda a: tuple(int(v) for v in a)  # noqa: E731
    prefix, cores_per_chip = _infer_accelerator_spec(all_devices, local_devices, full, per_host)
    return BoxLayout(
        as_tuple(origin),
        as_tuple(extent),
        as_tuple(full),
        as_tuple(per_host),
        accelerator_prefix=prefix,
        cores_per_chip=cores_per_chip,
    )


def plan_replicas(
    device_mesh: np.ndarray,
    axis_names: tp.Sequence[str],
    *,
    process_index: int,
    all_devices: tp.Sequence[tp.Any],
    local_devices: tp.Sequence[tp.Any],
    address: str,
    client: tp.Any,
    port: int,
    timeout_s: float,
) -> tuple[ReplicaPlan, dict[str, str]] | None:
    """Work out this process's replica world from the global mesh and the slice's key-value store.

    Every process of the slice must call this with the same mesh: the decision to split (or not) is a pure function
    of it, so all processes agree. Each publishes its address and host position and reads every other's.

    Args:
        device_mesh: The deployment's global device mesh.
        axis_names: Its axis names.
        process_index: This process's index in the slice world.
        all_devices: Every device of the slice.
        local_devices: This process's devices.
        address: This host's internal address.
        client: The slice world's ``jax.distributed`` key-value client.
        port: Port of each replica world's coordination service.
        timeout_s: How long to wait for the other processes' addresses.

    Returns:
        ``(plan, libtpu environment)``, or ``None`` when the launch keeps one rank-major engine.
    """
    slices = split_dp_slices(device_mesh, axis_names)
    if slices is None:
        return None
    layouts = [box_layout(s, all_devices, local_devices) for s in slices]
    if any(layout is None for layout in layouts):
        return None
    me = int(process_index)
    mine = next(r for r, s in enumerate(slices) if any(int(d.process_index) == me for d in s))
    layout = layouts[mine]
    own_chip = min(tuple(d.coords)[:3] for d in local_devices)
    client.key_value_set(
        _HOST_KEY.format(port=int(port), process=me),
        json.dumps({"address": address, "position": layout.host_position(own_chip)}),
        allow_overwrite=True,
    )
    timeout_ms = max(1, int(float(timeout_s) * 1000))
    orders: list[list[tuple[tuple[int, ...], str, int]]] = []
    for s in slices:
        members = sorted({int(d.process_index) for d in s})
        entries = []
        for p in members:
            info = json.loads(client.blocking_key_value_get(_HOST_KEY.format(port=int(port), process=p), timeout_ms))
            entries.append((tuple(info["position"]), str(info["address"]), p))
        orders.append(sorted(entries))
    order = orders[mine]
    rank = next(i for i, (_, _, p) in enumerate(order) if p == me)
    hostnames = [addr for _, addr, _ in order]
    coord_host = f"[{hostnames[0]}]" if ":" in hostnames[0] and not hostnames[0].startswith("[") else hostnames[0]
    plan = ReplicaPlan(
        index=mine,
        rank=rank,
        num_processes=len(order),
        coordinator=f"{coord_host}:{int(port)}",
        leaders=tuple(o[0][1] for o in orders),
        hosts=tuple(tuple(addr for _, addr, _ in o) for o in orders),
    )
    return plan, layout.libtpu_env(hostnames, rank)


def reexec_into_replica(plan: ReplicaPlan, env: tp.Mapping[str, str], token: str) -> tp.NoReturn:
    """Replace this process with the same command inside its replica's world.

    Call after leaving the slice world (``jax.distributed.shutdown``). libtpu opens ``/dev/vfio`` and its lock file
    without close-on-exec; every inherited descriptor is marked close-on-exec here so the kernel releases the chips
    during ``execve``, otherwise the new image finds them busy. The token travels in a pipe, not on the command line.
    """
    read_fd = _token_pipe(token)
    for name in os.listdir("/proc/self/fd"):
        fd = int(name)
        if fd > 2 and fd != read_fd:
            try:
                os.set_inheritable(fd, False)
            except OSError:
                pass
    os.set_inheritable(read_fd, True)
    child_env = {**os.environ, **env, "ENABLE_DISTRIBUTED_INIT": "0"}
    argv = relaunch_argv(dataclasses.replace(plan, token_fd=read_fd))
    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(sys.executable, argv, child_env)


def spawn_front(plan: ReplicaPlan, token: str, index: int = 0) -> subprocess.Popen:
    """Start the front: the same command, as a host-only process serving the OpenAI endpoint over every replica.

    The front runs JAX on the host CPU with no distributed world, so HTTP handling, routing and per-request
    tokenisation and streaming never share an interpreter with a replica's step loop. It stays in the launch's process
    group (``eray tpu job stop`` stops it with the rest) and is a separate process from replica 0, so it keeps routing
    to the other replicas when replica 0's world fails.

    Args:
        plan: The calling replica-0 leader's plan.
        token: The shared request-plane token, handed over through a pipe.

    Returns:
        The front's process.
    """
    read_fd = _token_pipe(token)
    front = dataclasses.replace(plan, role="front", token_fd=read_fd, front_index=int(index))
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "ENABLE_DISTRIBUTED_INIT": "0"}
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        return subprocess.Popen(relaunch_argv(front), env=env, pass_fds=(read_fd,))
    finally:
        os.close(read_fd)


def relaunch_argv(plan: ReplicaPlan) -> list[str]:
    """This process's own command line, carrying ``plan`` (any plan it was started with is replaced)."""
    argv = list(sys.orig_argv[1:])
    if "--replica-plan" in argv:
        at = argv.index("--replica-plan")
        del argv[at : at + 2]
    return [sys.executable, *argv, "--replica-plan", plan.to_json()]


def _token_pipe(token: str) -> int:
    """A pipe already holding ``token`` and closed for writing; returns its read end."""
    read_fd, write_fd = os.pipe()
    os.write(write_fd, token.encode())
    os.close(write_fd)
    return read_fd
