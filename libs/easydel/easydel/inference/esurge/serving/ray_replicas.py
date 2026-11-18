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

"""dp replicas of an eSurge engine built through the classic Python API from a Ray driver.

When ``eLargeModel(...).build_esurge()`` (or ``build_esurge(cfg)``) runs in a Ray driver and the mesh has ``dp > 1``
and ``pp == 1``, the engine is not built in the driver. Instead:

1. **Probe.** One Ray task per TPU node joins the whole slice's JAX world, builds the deployment's global mesh and
   works out (:func:`.replicas.plan_replicas`) which dp replica the host belongs to, its rank there and the libtpu
   environment that makes the replica's hosts their own TPU slice. The task leaves the world and exits, releasing
   the chips.
2. **Replicas.** One Ray task per TPU node, started with its replica's libtpu environment, joins its replica's own
   JAX world and builds a complete ``dp = 1`` engine through the same ``build_esurge`` with the launch-wide limits
   split over the replicas. The replica leader serves its request plane; the other hosts replay its steps.
3. **Front.** The driver (JAX on CPU, never the TPU) returns a router over a request-plane client per replica leader.
   It is ready as soon as one replica answers and attaches the others as they come up. ``close()`` stops the
   replica tasks.

The returned router is engine-shaped, so ``eSurgeApiServer(surge).run(...)`` serves it unchanged.
"""

from __future__ import annotations

import copy
import secrets
import sys
import time
import typing as tp
from collections.abc import Mapping

from eformer.loggings import get_logger

from .config import EASYDEL_MESH_AXES

logger = get_logger(__name__)

DEFAULT_CONTROL_PORT = 19666
"""Request-plane port of each replica leader when the config does not set ``distributed_control_port``."""
REPLICA_HEARTBEAT_TIMEOUT_S = 120.0
"""Worker silence a replica tolerates (the value the YAML serving path uses)."""
PLAN_TIMEOUT_S = 600.0
"""How long the probe waits for every host of the slice."""
_DEFAULT_DIMS = (1, 1, 1, 1, -1, 1)


def running_as_ray_driver() -> bool:
    """Whether this process is a connected Ray driver (not a Ray worker), without importing Ray."""
    ray = sys.modules.get("ray")
    if ray is None:
        return False
    try:
        if not ray.is_initialized():
            return False
        from ray._private import worker

        return worker.global_worker.mode == worker.SCRIPT_MODE
    except Exception:
        return False


def mesh_of(cfg: Mapping[str, tp.Any]) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """The eLarge config's mesh dims and axis names."""
    sharding = cfg.get("sharding") or {}
    dims = tuple(int(v) for v in sharding.get("axis_dims", _DEFAULT_DIMS))
    names = tuple(str(v) for v in sharding.get("axis_names", EASYDEL_MESH_AXES))
    return dims, names


def wants_replicas(cfg: Mapping[str, tp.Any]) -> bool:
    """Whether ``build_esurge`` on this config should launch dp replicas instead of building one engine here."""
    dims, names = mesh_of(cfg)
    if "dp" not in names or "pp" not in names:
        return False
    return dims[names.index("dp")] > 1 and dims[names.index("pp")] == 1 and running_as_ray_driver()


def control_port_of(cfg: Mapping[str, tp.Any]) -> int:
    distributed = (cfg.get("esurge") or {}).get("distributed") or {}
    return int(distributed.get("distributed_control_port") or DEFAULT_CONTROL_PORT)


def replica_config(
    cfg: Mapping[str, tp.Any],
    *,
    count: int,
    token: str,
    control_port: int,
    bind_host: str,
) -> dict[str, tp.Any]:
    """One replica's eLarge config: ``dp`` collapsed to 1, launch-wide limits split, a request plane added.

    Args:
        cfg: The deployment's (normalized) eLarge config.
        count: Number of replicas.
        token: Shared request-plane secret.
        control_port: Port the replica leader's request plane binds.
        bind_host: Address this host's request plane binds.

    Returns:
        A new config; ``cfg`` is not modified.
    """
    from easydel.infra.elarge.builders import normalize_esurge_sections

    from .app import per_replica_sections

    out = copy.deepcopy(dict(cfg))
    sharding = dict(out.get("sharding") or {})
    dims, names = mesh_of(out)
    replica_dims = list(dims)
    replica_dims[names.index("dp")] = 1
    sharding["axis_dims"] = tuple(replica_dims)
    out["sharding"] = sharding
    sections = {name: value for name, value in normalize_esurge_sections(out.get("esurge")).items() if value}
    sections = per_replica_sections(sections, count)
    sections["distributed"] = {
        "coordination": "zmq",
        "distributed_auth_token": token,
        "distributed_control_port": int(control_port),
        "distributed_control_bind_host": bind_host,
        "distributed_heartbeat_timeout_s": REPLICA_HEARTBEAT_TIMEOUT_S,
    }
    out["esurge"] = sections
    return out


def replica_task_env(libtpu_env: Mapping[str, str]) -> dict[str, str]:
    """Environment of one replica task: its sub-slice libtpu variables, kept as planned.

    A task that reserves every chip of its node gets ``TPU_HOST_BOUNDS`` / ``TPU_CHIPS_PER_HOST_BOUNDS`` removed by
    Ray's TPU accelerator manager, which would turn the replica's sub-slice back into the whole slice; Ray's own
    switch tells it to leave them alone. The replica joins its world explicitly, not at ``import easydel``.
    """
    return {
        **{str(k): str(v) for k, v in libtpu_env.items()},
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS": "1",
        "ENABLE_DISTRIBUTED_INIT": "0",
    }


LIBTPU_LOCKFILE = "/tmp/libtpu_lockfile"


def _tolerate_foreign_libtpu_lockfile() -> bool:
    """Let libtpu start next to a ``/tmp/libtpu_lockfile`` this user cannot write (left by another user's job).

    libtpu takes that host-wide lock at startup and refuses ("TPU is already in use ... probably owned by another
    user") when another user's lockfile is there, even if no process holds the chips. The same rule as eray's host
    launcher: only then is libtpu's own ``ALLOW_MULTIPLE_LIBTPU_LOAD`` set; the other user's file is never touched.
    Whether the chips are really free is checked separately (:func:`_require_free_chips`).

    Returns:
        Whether the lock was bypassed.
    """
    import os

    if os.path.exists(LIBTPU_LOCKFILE) and not os.access(LIBTPU_LOCKFILE, os.W_OK):
        os.environ["ALLOW_MULTIPLE_LIBTPU_LOAD"] = "1"
        return True
    return False


def _probe(dims: tuple[int, ...], names: tuple[str, ...], port: int, timeout_s: float):
    """Ray task: join the slice world, plan this host's replica, leave. Returns ``(plan, env)`` or ``None``."""
    import jax

    _tolerate_foreign_libtpu_lockfile()  # before libtpu loads (create_mesh below)
    # Importing easydel to unpickle this task may already have joined the slice world.
    if not jax.distributed.is_initialized():
        jax.distributed.initialize()
    try:
        import ray
        import spectrax as spx
        from jax._src.distributed import global_state
        from jax.experimental import multihost_utils

        import easydel as ed

        from .replicas import plan_replicas

        _require_free_chips()  # before the TPU backend starts in create_mesh
        mesh = spx.to_jax_mesh(ed.EasyDeLBaseConfig.create_mesh(sharding_axis_dims=dims, sharding_axis_names=names))
        planned = plan_replicas(
            mesh.devices,
            mesh.axis_names,
            process_index=jax.process_index(),
            all_devices=jax.devices(),
            local_devices=jax.local_devices(),
            address=ray.util.get_node_ip_address(),
            client=global_state.client,
            port=port,
            timeout_s=timeout_s,
        )
        multihost_utils.sync_global_devices("easydel-esurge-ray-replica-probe")
        return planned
    finally:
        jax.distributed.shutdown()


CHIP_RELEASE_TIMEOUT_S = 180.0
"""How long a task waits for this host's TPU devices to be released (a process that just exited may still hold them)."""


def _wait_for_free_chips(timeout_s: float) -> list[str]:
    """Wait until this host's TPU device files can be opened, whoever held them.

    A process that held the chips (the previous task, or an engine that was just killed) can keep them busy for a
    while after it is gone; libtpu then fails with "already in use". The device is opened and closed at once.

    Returns:
        Device files still busy when the wait timed out (empty when they are free).
    """
    import errno
    import glob
    import os

    deadline = time.monotonic() + float(timeout_s)
    while True:
        busy = []
        for path in sorted(glob.glob("/dev/vfio/[0-9]*")) + sorted(glob.glob("/dev/accel*")):
            try:
                fd = os.open(path, os.O_RDWR)
            except OSError as exc:
                if exc.errno == errno.EBUSY:
                    busy.append(path)
                continue
            os.close(fd)
        if not busy or time.monotonic() >= deadline:
            return busy
        time.sleep(1.0)


def _require_free_chips() -> None:
    busy = _wait_for_free_chips(CHIP_RELEASE_TIMEOUT_S)
    if busy:
        raise RuntimeError(f"TPU devices {busy} on this host are still in use after {CHIP_RELEASE_TIMEOUT_S:.0f}s")


def _run_replica(cfg: dict[str, tp.Any], plan: tp.Any, timeout_s: float) -> None:
    """Ray task: join this replica's JAX world and run its engine until it shuts down or the task is cancelled."""
    import jax

    _tolerate_foreign_libtpu_lockfile()
    _require_free_chips()
    jax.distributed.initialize(
        plan.coordinator, plan.num_processes, plan.rank, initialization_timeout=max(1, int(timeout_s))
    )
    from easydel.infra.elarge.builders import build_esurge

    engine = build_esurge(cfg)
    try:
        if engine.is_coordination_leader:
            # The request plane is served by the coordinator's IO thread; the front reaches the engine through it.
            while True:
                time.sleep(1.0)
        engine.wait_for_shutdown()
    finally:
        engine.close()


def _tpu_nodes(ray: tp.Any) -> list[dict[str, tp.Any]]:
    nodes = [n for n in ray.nodes() if n.get("Alive") and float(n.get("Resources", {}).get("TPU", 0) or 0) > 0]
    return sorted(nodes, key=lambda n: n["NodeManagerAddress"])


def _cpu_only_driver() -> None:
    """Keep the driver off the TPU: one of the replicas runs on this host."""
    import jax
    from jax._src import xla_bridge

    if "tpu" in xla_bridge._backends:
        raise RuntimeError(
            "this Ray driver already initialized the TPU backend, so the replica on this host cannot get its chips; "
            "build the dp>1 engine before anything in the driver touches a TPU device"
        )
    if not xla_bridge._backends:
        jax.config.update("jax_platforms", "cpu")


def maybe_launch_replicas(cfg: Mapping[str, tp.Any]) -> tp.Any | None:
    """Launch dp replicas over the Ray cluster's TPU nodes when this is a Ray driver and the mesh asks for them.

    Returns:
        A router over the replicas (engine-shaped; ``close()`` stops them), or ``None`` to build one engine as before.

    Raises:
        RuntimeError: When the dp slices are not whole-host boxes, or a replica task fails before any replica is up.
    """
    if not wants_replicas(cfg):
        return None
    import ray

    nodes = _tpu_nodes(ray)
    if not nodes:
        return None
    _cpu_only_driver()
    return _launch(ray, cfg, nodes)


def _launch(ray: tp.Any, cfg: Mapping[str, tp.Any], nodes: list[dict[str, tp.Any]]) -> tp.Any:
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from easydel.inference.esurge.distributed.remote_engine import RemoteEngineHandle
    from easydel.inference.esurge.distributed.routed_engine import RoutedEngine

    from .app import ReplicaAttacher

    dims, names = mesh_of(cfg)
    port = control_port_of(cfg)

    def placed(node):
        return {
            "scheduling_strategy": NodeAffinitySchedulingStrategy(node["NodeID"], soft=False),
            "resources": {"TPU": float(node["Resources"]["TPU"])},
            "num_cpus": 0,
            "max_retries": 0,
        }

    probe = ray.remote(max_calls=1)(_probe)
    logger.info("dp=%d over %d TPU hosts: planning replicas", dims[names.index("dp")], len(nodes))
    planned = ray.get([probe.options(**placed(n)).remote(dims, names, port + 1, PLAN_TIMEOUT_S) for n in nodes])
    if any(p is None for p in planned):
        raise RuntimeError(
            f"mesh {dims} cannot run as dp replicas from a Ray driver: its dp slices are not whole-host boxes of the "
            "slice. Use dp=1, or start the engine on every host instead of from a driver."
        )
    leaders = planned[0][0].leaders
    count = len(leaders)
    token = secrets.token_hex(32)
    model = cfg.get("model") or {}
    tokenizer = model.get("tokenizer") or model.get("name_or_path")

    replica = ray.remote(max_calls=1)(_run_replica)
    tasks = []
    for node, (plan, env) in zip(nodes, planned, strict=True):
        node_cfg = replica_config(cfg, count=count, token=token, control_port=port, bind_host=node["NodeManagerAddress"])
        options = placed(node)
        options["runtime_env"] = {"env_vars": replica_task_env(env)}
        tasks.append(replica.options(**options).remote(node_cfg, plan, PLAN_TIMEOUT_S))
    logger.info("started %d replica tasks; replica leaders %s", len(tasks), list(leaders))

    def connect(addr: str):
        return RemoteEngineHandle(addr, port, auth_token=token, tokenizer_source=tokenizer, connect_timeout_s=10.0)

    attacher: ReplicaAttacher | None = None

    def stop() -> None:
        if attacher is not None:
            attacher.stop()
        for task in tasks:
            try:
                ray.cancel(task, force=True)
            except Exception:
                pass

    class RayRoutedEngine(RoutedEngine):
        """The front's router; ``close()`` also stops the replica tasks."""

        def close(self) -> None:
            try:
                super().close()
            finally:
                stop()

    attacher = ReplicaAttacher(leaders, connect=connect, log=logger.info, router_factory=RayRoutedEngine)
    try:
        while attacher.router is None:
            attacher.attach_round()
            if attacher.router is None:
                done, _ = ray.wait(tasks, num_returns=1, timeout=5.0)
                if done:
                    ray.get(done[0])  # raises the replica's error
                    raise RuntimeError("a replica task exited before any replica came up")
    except BaseException:
        stop()
        raise
    attacher.start()
    return attacher.router
