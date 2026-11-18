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

"""dp replicas of an eSurge engine when the same script runs on every host of a TPU slice.

The classic multi-host launch runs one script on every host: ``import easydel`` joins the slice's JAX world and every
host calls ``build_esurge()``. With ``dp > 1`` and ``pp == 1``, every host moves into its dp replica's own JAX world
before the TPU backend starts, and the deployment runs exactly like ``serving serve`` with ``dp: N``:

1. **Plan.** A short-lived subprocess per host joins a planning world (the slice coordinator's host, two ports above
   the control port), builds the deployment's global mesh on the real devices and works out
   (:func:`.replicas.plan_replicas`) the host's replica, its rank there and the libtpu environment that makes the
   replica's hosts their own TPU slice. The planning world also agrees on a fresh request-plane token. The result
   comes back over a pipe and the subprocess exits, releasing the chips.
2. **Switch.** All hosts leave the slice world together.
3. **Replicas.** A non-leader host joins its replica's world in this process and builds a complete ``dp = 1`` engine
   (``build_esurge`` with the launch-wide limits split over the replicas); its replay loop runs as before. On each
   replica leader the engine runs in a child process (rank 0 of the replica world, holding the chips), and this
   process becomes a host-only front: ``build_esurge`` returns a router over every replica leader's request plane,
   so ``eSurgeApiServer(surge)`` never shares an interpreter with a step loop.

After the switch ``jax.process_index()`` is the rank inside the replica world (``0`` in a front), so a script's usual
``if jax.process_index() == 0: serve`` serves on every replica leader: each is an endpoint for all replicas.
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import subprocess
import sys
import time
import typing as tp
from collections.abc import Mapping, Sequence

from eformer.loggings import get_logger

from .ray_replicas import (
    PLAN_TIMEOUT_S,
    control_port_of,
    maybe_launch_replicas,
    mesh_of,
    replica_config,
    running_as_ray_driver,
)

logger = get_logger(__name__)

REPLICA_PORT_OFFSET = 1
"""Replica worlds' JAX coordination port, relative to the request-plane control port (as in ``serving serve``)."""
PROBE_PORT_OFFSET = 2
"""Planning world's JAX coordination port, relative to the request-plane control port."""
ENGINE_STOP_TIMEOUT_S = 60.0
"""How long the front waits for its replica leader's engine process to exit on ``close()``."""
_PROBE_CODE = "from easydel.inference.esurge.serving.host_replicas import probe_main; probe_main()"
_ENGINE_CODE = "from easydel.inference.esurge.serving.host_replicas import leader_engine_main; leader_engine_main()"


def launch_dp_replicas(cfg: Mapping[str, tp.Any]) -> tp.Any | None:
    """dp replicas for ``build_esurge``: from a Ray driver, or in place on every host of a slice; ``None`` otherwise."""
    replicas = maybe_launch_replicas(cfg)
    if replicas is None:
        replicas = maybe_enter_host_replica(cfg)
    return replicas


def slice_world() -> tuple[str, int, int] | None:
    """``(coordinator host, process count, process id)`` of the joined multi-process JAX world, or ``None``."""
    from jax._src.distributed import global_state

    if global_state.client is None or int(global_state.num_processes or 0) <= 1:
        return None
    host = str(global_state.coordinator_address).rsplit(":", 1)[0]
    return host, int(global_state.num_processes), int(global_state.process_id)


def on_tpu_host() -> bool:
    """Whether this process may run on a TPU (device files present, TPU not excluded by ``jax_platforms``)."""
    import jax

    platforms = str(jax.config.jax_platforms or "")
    if platforms and "tpu" not in platforms.split(","):
        return False
    return bool(glob.glob("/dev/vfio/[0-9]*") or glob.glob("/dev/accel*"))


def wants_host_replicas(cfg: Mapping[str, tp.Any]) -> bool:
    """Whether ``build_esurge`` on this host should move into its dp replica's own world."""
    dims, names = mesh_of(cfg)
    if "dp" not in names or "pp" not in names:
        return False
    if dims[names.index("dp")] <= 1 or dims[names.index("pp")] != 1:
        return False
    return not running_as_ray_driver() and slice_world() is not None and on_tpu_host()


def child_env(base: Mapping[str, str], path: Sequence[str], extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment of a planning or engine child: this process's import path, no distributed join at import."""
    return {
        **base,
        **(extra or {}),
        "ENABLE_DISTRIBUTED_INIT": "0",
        "PYTHONPATH": os.pathsep.join(p for p in path if p),
    }


def encode_probe_result(planned: tuple[tp.Any, dict[str, str]] | None, token: str) -> bytes:
    if planned is None:
        return json.dumps({"plan": None}).encode()
    plan, env = planned
    return json.dumps({"plan": plan.to_json(), "env": dict(env), "token": token}).encode()


def decode_probe_result(data: bytes) -> tuple[tp.Any, dict[str, str], str] | None:
    """``(plan, libtpu environment, request-plane token)``, or ``None`` when the slices are not host boxes."""
    from .replicas import ReplicaPlan

    obj = json.loads(data.decode())
    if obj["plan"] is None:
        return None
    return ReplicaPlan.from_json(obj["plan"]), {str(k): str(v) for k, v in obj["env"].items()}, str(obj["token"])


def run_probe(request: Mapping[str, tp.Any]) -> tuple[tp.Any, dict[str, str], str] | None:
    """Run the planning subprocess; its result (which carries the token) comes back over a pipe, never argv or logs.

    Raises:
        RuntimeError: When the subprocess fails.
    """
    read_fd, write_fd = os.pipe()
    try:
        argv = [sys.executable, "-c", _PROBE_CODE, json.dumps({**request, "result_fd": write_fd})]
        proc = subprocess.Popen(argv, env=child_env(os.environ, sys.path), pass_fds=(write_fd,))
    except BaseException:
        os.close(read_fd)
        raise
    finally:
        os.close(write_fd)
    with os.fdopen(read_fd, "rb") as result:
        data = result.read()
    code = proc.wait()
    if code != 0 or not data:
        raise RuntimeError(f"dp replica planning subprocess failed (exit code {code})")
    return decode_probe_result(data)


def probe_main() -> None:
    """Planning subprocess: join the planning world, plan this host's replica on the real devices, report, leave."""
    request = json.loads(sys.argv[1])
    out_fd = int(request["result_fd"])
    timeout_s = float(request["timeout_s"])
    import jax

    from .ray_replicas import _require_free_chips, _tolerate_foreign_libtpu_lockfile

    _tolerate_foreign_libtpu_lockfile()
    jax.distributed.initialize(
        request["coordinator"],
        int(request["num_processes"]),
        int(request["process_id"]),
        initialization_timeout=max(1, int(timeout_s)),
        coordinator_bind_address=request["coordinator"],
    )
    try:
        import spectrax as spx
        from jax._src.distributed import global_state
        from jax.experimental import multihost_utils

        import easydel as ed

        from .cluster import resolve_bind_host, shared_coordination_token
        from .replicas import plan_replicas

        _require_free_chips()
        mesh = spx.to_jax_mesh(
            ed.EasyDeLBaseConfig.create_mesh(
                sharding_axis_dims=tuple(request["dims"]), sharding_axis_names=tuple(request["names"])
            )
        )
        port = int(request["replica_port"])
        planned = plan_replicas(
            mesh.devices,
            mesh.axis_names,
            process_index=jax.process_index(),
            all_devices=jax.devices(),
            local_devices=jax.local_devices(),
            address=resolve_bind_host("internal"),
            client=global_state.client,
            port=port,
            timeout_s=timeout_s,
        )
        token = ""
        if planned is not None:
            token = shared_coordination_token(
                port, is_leader=jax.process_index() == 0, timeout_s=timeout_s, client=global_state.client
            )
        multihost_utils.sync_global_devices("easydel-esurge-host-replica-plan")
        payload = memoryview(encode_probe_result(planned, token))
        while payload:
            payload = payload[os.write(out_fd, payload) :]
    finally:
        os.close(out_fd)
        jax.distributed.shutdown()


def maybe_enter_host_replica(cfg: Mapping[str, tp.Any]) -> tp.Any | None:
    """Move this host into its dp replica's own JAX world, when the launch allows it.

    Returns:
        On a replica leader, a router over every replica (engine-shaped); on other hosts, the replica's engine;
        ``None`` to build one engine as before (dp=1, a Ray driver, one process, or dp slices that are not host boxes).

    Raises:
        RuntimeError: When the JAX backend was started before ``build_esurge()``: replicas need it unstarted.
    """
    if not wants_host_replicas(cfg):
        return None
    from jax._src import xla_bridge

    dims, names = mesh_of(cfg)
    if xla_bridge.backends_are_initialized():
        raise RuntimeError(
            f"dp={dims[names.index('dp')]}: the JAX backend ({', '.join(sorted(xla_bridge._backends))}) was started "
            "before build_esurge(), so this host can no longer move into its dp replica's own JAX world. Something "
            "in the script ran jax.devices()/jax.device_count()/jax.default_backend(), created a mesh, built or "
            "loaded a model, or put an array on a device before build_esurge(). Call build_esurge() first (it loads "
            "the model itself), or use dp=1."
        )
    world = slice_world()
    assert world is not None
    host, num_processes, process_id = world
    port = control_port_of(cfg)
    request = {
        "coordinator": f"{host}:{port + PROBE_PORT_OFFSET}",
        "num_processes": num_processes,
        "process_id": process_id,
        "dims": list(dims),
        "names": list(names),
        "replica_port": port + REPLICA_PORT_OFFSET,
        "timeout_s": PLAN_TIMEOUT_S,
    }
    logger.info("dp=%d over %d hosts: planning replicas", dims[names.index("dp")], num_processes)
    planned = run_probe(request)
    if planned is None:
        logger.warning("dp slices are not separate host boxes; building one rank-major engine over the whole mesh")
        return None
    plan, env, token = planned
    return _enter_replica(cfg, plan, env, token, port)


def _enter_replica(cfg: Mapping[str, tp.Any], plan: tp.Any, env: Mapping[str, str], token: str, port: int) -> tp.Any:
    import jax
    from jax._src.distributed import global_state

    from .cluster import resolve_bind_host

    # Leave the slice world together: no client may outlive the coordination service on the slice's process 0.
    global_state.client.wait_at_barrier("easydel-esurge-host-replica-switch", int(PLAN_TIMEOUT_S * 1000))
    jax.distributed.shutdown()
    node_cfg = replica_config(
        cfg, count=plan.count, token=token, control_port=port, bind_host=resolve_bind_host("internal")
    )
    logger.info(
        "dp replica %d of %d: worker %d/%d (topology %s, coordinator %s)",
        plan.index,
        plan.count,
        plan.rank,
        plan.num_processes,
        env.get("TPU_TOPOLOGY"),
        plan.coordinator,
    )
    if plan.rank == 0:
        return _leader_front(cfg, node_cfg, plan, env, token, port)

    from easydel.infra.elarge.builders import build_esurge

    from .ray_replicas import _require_free_chips, _tolerate_foreign_libtpu_lockfile

    os.environ.update(env)  # read by libtpu when the backend starts, which happens only after this point
    _tolerate_foreign_libtpu_lockfile()
    _require_free_chips()
    jax.distributed.initialize(
        plan.coordinator,
        plan.num_processes,
        plan.rank,
        initialization_timeout=max(1, int(PLAN_TIMEOUT_S)),
        coordinator_bind_address=plan.coordinator,
    )
    return build_esurge(node_cfg)


def become_single_process() -> None:
    """After leaving the slice world, make this process a one-process JAX program (index 0 of 1).

    ``jax.distributed.shutdown()`` drops the client but keeps the old process id and count, and JAX's CPU client
    takes its process index from them, so a front would otherwise report its slice-world rank (and a script's
    ``if jax.process_index() == 0: serve`` would skip serving on it). Call before the backend starts.
    """
    from jax._src.distributed import global_state

    global_state.process_id = 0
    global_state.num_processes = 1
    global_state.coordinator_address = None


def leader_engine_main() -> None:
    """A replica leader's engine process: rank 0 of the replica world, its ``dp = 1`` engine and request plane.

    Reads ``(replica config, plan)`` from the inherited pipe (the config carries the token, so it never reaches argv)
    and runs until the front that started it exits.
    """
    read_fd, parent = int(sys.argv[1]), int(sys.argv[2])
    with os.fdopen(read_fd, "rb") as source:
        node_cfg, plan = pickle.load(source)
    import jax

    from easydel.infra.elarge.builders import build_esurge

    from .ray_replicas import _require_free_chips, _tolerate_foreign_libtpu_lockfile

    _tolerate_foreign_libtpu_lockfile()
    _require_free_chips()
    jax.distributed.initialize(
        plan.coordinator,
        plan.num_processes,
        plan.rank,
        initialization_timeout=max(1, int(PLAN_TIMEOUT_S)),
        coordinator_bind_address=plan.coordinator,
    )
    engine = build_esurge(node_cfg)
    try:
        # The coordinator's IO thread serves the request plane; this process lives as long as its front.
        while os.getppid() == parent:
            time.sleep(1.0)
    finally:
        engine.close()


def start_leader_engine(node_cfg: Mapping[str, tp.Any], plan: tp.Any, env: Mapping[str, str]) -> subprocess.Popen:
    """Start the replica leader's engine in a child process holding the chips; config and plan go over a pipe."""
    read_fd, write_fd = os.pipe()
    try:
        argv = [sys.executable, "-c", _ENGINE_CODE, str(read_fd), str(os.getpid())]
        proc = subprocess.Popen(argv, env=child_env(os.environ, sys.path, env), pass_fds=(read_fd,))
    except BaseException:
        os.close(write_fd)
        raise
    finally:
        os.close(read_fd)
    with os.fdopen(write_fd, "wb") as sink:
        pickle.dump((dict(node_cfg), plan), sink)
    return proc


def _leader_front(
    cfg: Mapping[str, tp.Any],
    node_cfg: Mapping[str, tp.Any],
    plan: tp.Any,
    env: Mapping[str, str],
    token: str,
    port: int,
) -> tp.Any:
    """This process becomes a host-only front over every replica; its replica's engine runs in a child process."""
    import jax

    from easydel.inference.esurge.distributed.remote_engine import RemoteEngineHandle
    from easydel.inference.esurge.distributed.routed_engine import RoutedEngine

    from .app import ReplicaAttacher

    jax.config.update("jax_platforms", "cpu")  # the engine child owns this host's chips
    become_single_process()
    proc = start_leader_engine(node_cfg, plan, env)
    logger.info("replica %d leader engine started in process %d; this process is its front", plan.index, proc.pid)
    model = cfg.get("model") or {}
    tokenizer = model.get("tokenizer") or model.get("name_or_path")

    def connect(addr: str) -> tp.Any:
        return RemoteEngineHandle(addr, port, auth_token=token, tokenizer_source=tokenizer, connect_timeout_s=10.0)

    attacher: ReplicaAttacher | None = None

    def stop_engine() -> None:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=ENGINE_STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                proc.kill()

    class ReplicaRoutedEngine(RoutedEngine):
        """The front's router; ``close()`` also stops attaching replicas and this replica's engine process."""

        def close(self) -> None:
            try:
                if attacher is not None:
                    attacher.stop()
                super().close()
            finally:
                stop_engine()

    attacher = ReplicaAttacher(plan.leaders, connect=connect, log=logger.info, router_factory=ReplicaRoutedEngine)
    try:
        while attacher.router is None:
            attacher.attach_round()
            if attacher.router is None:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"replica {plan.index}'s engine process exited with code {proc.returncode} before any replica "
                        "came up"
                    )
                time.sleep(5.0)
    except BaseException:
        attacher.stop()
        stop_engine()
        raise
    attacher.start()
    return attacher.router
