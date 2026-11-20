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

"""Run one host's share of an eSurge OpenAI-compatible deployment.

Every host of the slice runs :func:`serve` with the same :class:`ServingConfig`. All of them build the engine (the
model is sharded over the whole slice); the coordination leader then serves HTTP while the others replay its steps
until it shuts the world down.
"""

from __future__ import annotations

import contextlib
import signal
import time
import typing as tp
from collections.abc import Iterator

from .cluster import public_urls, resolve_bind_host, shared_coordination_token
from .config import EASYDEL_MESH_AXES, ServingConfig
from .keys import load_or_create_api_key

ENGINE_SECTIONS = frozenset({"runtime", "cache", "context", "workers", "parsing", "vision"})
WORKER_SHUTDOWN_WAIT_S = 20.0
"""How long a worker that got SIGTERM waits for the leader's Shutdown before closing on its own."""
FRONT_STABLE_S = 300.0
"""A front that ran this long before exiting is restarted without backoff growth."""
FRONT_MAX_BACKOFF_S = 60.0
"""Longest wait between restarts of a front that keeps exiting."""
FRONT_PROCESSES = 8
"""Front processes replica 0's leader starts. They share the endpoint port (SO_REUSEPORT, the kernel spreads
connections) and each routes over every replica, so HTTP, routing and streaming are not bound to one interpreter."""
REPLICA_ATTACH_INTERVAL_S = 5.0
"""How often the front retries replicas it has not attached yet (or whose engine died)."""


class ShutdownRequested(BaseException):
    """Raised in the main thread by SIGTERM while serving, so the engine is closed on the way out.

    A ``BaseException`` so ``except Exception`` blocks in loading or serving code cannot swallow it.
    """


@contextlib.contextmanager
def sigterm_raises() -> Iterator[None]:
    """Turn SIGTERM into :class:`ShutdownRequested` in the main thread while the context is open.

    Enter it before the engine is built: the engine installs its diagnostic SIGTERM handler only over the default
    action, so it stays out. Uvicorn captures SIGTERM for its graceful HTTP shutdown and re-raises it afterwards with
    the previous handler restored; with this handler the leader then leaves ``server.run()`` through an exception and
    still closes the engine, which tells the workers to stop. With the diagnostic handler the re-raised signal killed
    the leader before ``close()`` and left every worker replaying until libtpu lost its peer.
    """

    def _raise(signum: int, frame: tp.Any) -> None:
        raise ShutdownRequested(signal.Signals(signum).name)

    previous = signal.signal(signal.SIGTERM, _raise)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def build_loading_kwargs(
    config: ServingConfig,
    text_config: tp.Any | None,
    *,
    ed: tp.Any,
    sharding_axis_dims: tp.Sequence[int] | None = None,
) -> dict[str, tp.Any]:
    """Pretrained-loader kwargs for ``config.model``.

    Args:
        config: The deployment.
        text_config: The checkpoint's text config, needed only when quantization uses the model's default pattern.
        ed: The ``easydel`` module (injected so this stays importable without loading JAX in tests).
        sharding_axis_dims: Mesh dims to load with instead of the deployment's (one dp replica's dims).

    Raises:
        ValueError: When quantization asks for the model's default pattern and the model has none.
    """
    import jax.numpy as jnp

    model = config.model
    dims = tuple(sharding_axis_dims) if sharding_axis_dims is not None else model.sharding_axis_dims
    kwargs: dict[str, tp.Any] = {
        "dtype": jnp.dtype(model.dtype),
        "param_dtype": jnp.dtype(model.param_dtype),
        "sharding_axis_dims": dims,
        "sharding_axis_names": EASYDEL_MESH_AXES,
        "auto_shard_model": True,
    }
    config_kwargs = dict(model.config_overrides)
    if model.attn_mechanism:
        config_kwargs["attn_mechanism"] = model.attn_mechanism
    if config_kwargs:
        kwargs["config_kwargs"] = config_kwargs
    if model.quantization is not None:
        pattern = model.quantization.pattern
        if pattern is None:
            default = getattr(text_config, "default_quantization_pattern", None)
            if default is None:
                raise ValueError(
                    f"{type(text_config).__name__} has no default_quantization_pattern; set model.quantization.pattern"
                )
            pattern = default(model.quantization.dtype)
        kwargs["quantization_config"] = ed.QuantizationConfig(dtype=model.quantization.dtype, pattern=pattern)
        kwargs["apply_quantization"] = True
    return kwargs


def build_engine_sections(
    config: ServingConfig, *, world_size: int, token: str | None, control_bind_host: str | None
) -> dict[str, tp.Any]:
    """Per-section eSurge kwargs; adds the ZMQ ``distributed`` section for a multi-host world.

    Raises:
        ValueError: On an ``esurge`` section eSurge does not accept here.
    """
    unknown = sorted(set(config.esurge) - ENGINE_SECTIONS)
    if unknown:
        raise ValueError(f"`esurge` has unknown sections {unknown}; allowed: {sorted(ENGINE_SECTIONS)}")
    sections = {name: dict(value or {}) for name, value in config.esurge.items()}
    if world_size > 1:
        if not token or not control_bind_host:
            raise ValueError("a multi-host world needs a coordination token and a control bind host")
        sections["distributed"] = {
            "coordination": "zmq",
            "distributed_auth_token": token,
            "distributed_control_port": config.coordination.control_port,
            "distributed_control_bind_host": control_bind_host,
            "distributed_heartbeat_timeout_s": config.coordination.heartbeat_timeout_s,
        }
    return sections


# Engine limits a deployment states for the whole launch: with dp replicas each replica gets its share.
_PER_REPLICA_LIMITS = (("runtime", "max_num_seqs"), ("runtime", "max_num_batched_tokens"), ("cache", "max_cache_tokens"))


def per_replica_sections(sections: dict[str, tp.Any], replicas: int) -> dict[str, tp.Any]:
    """Split launch-wide engine limits over ``replicas`` independent engines.

    A dp>1 deployment states rows, batched tokens, token buckets and KV capacity for the whole launch (the rank-major
    engine splits them over its dp ranks). Run as one engine per dp replica, each engine gets ``1/replicas`` of them,
    rounded up, so the same config serves the same totals.

    Args:
        sections: Engine sections as built by :func:`build_engine_sections`.
        replicas: Number of replicas.

    Returns:
        New sections; ``sections`` is not modified.
    """
    out = {name: dict(value) for name, value in sections.items()}

    def share(value: tp.Any) -> int:
        return max(1, -(-int(value) // replicas))

    for section, key in _PER_REPLICA_LIMITS:
        if out.get(section, {}).get(key) is not None:
            out[section][key] = share(out[section][key])
    buckets = out.get("runtime", {}).get("max_num_seq_buckets")
    if buckets:
        out["runtime"]["max_num_seq_buckets"] = sorted({share(b) for b in buckets})
    return out


def replica_dims(config: ServingConfig) -> tuple[int, ...]:
    """One dp replica's mesh dims: the deployment's dims with ``dp`` collapsed to 1."""
    dims = [int(v) for v in config.model.sharding_axis_dims]
    dims[EASYDEL_MESH_AXES.index("dp")] = 1
    return tuple(dims)


def _log(rank: int, message: str) -> None:
    print(f"[esurge-serving rank={rank}] {message}", flush=True)


def serve(config: ServingConfig, plan: tp.Any | None = None) -> None:
    """Join the slice, build the engine, and serve (leader) or replay steps (workers) until shutdown.

    With ``dp > 1`` and dp slices that are separate host boxes, every host first leaves the slice world and re-executes
    the same command inside its replica's own JAX world (``plan`` is then set); see :mod:`.replicas`.

    SIGTERM (``eray tpu job stop``) ends it cleanly on every rank: the engine is closed and the call returns.
    """
    with sigterm_raises():
        try:
            _serve(config, plan)
        except ShutdownRequested as exc:
            print(f"[esurge-serving] stopped by {exc}", flush=True)


def _replica_port(config: ServingConfig) -> int:
    """Port of each replica world's JAX coordination service (next to the ZMQ control port)."""
    return int(config.coordination.control_port) + 1


def _split_into_replicas(config: ServingConfig, rank: int, token: str | None, ed: tp.Any) -> None:
    """In the slice world: re-execute this host inside its dp replica's own JAX world, or return to stay rank-major."""
    dims = tuple(int(v) for v in config.model.sharding_axis_dims)
    if dims[EASYDEL_MESH_AXES.index("dp")] == 1 or dims[EASYDEL_MESH_AXES.index("pp")] != 1 or not token:
        return
    import jax
    import spectrax as spx
    from jax.experimental import multihost_utils

    from .replicas import plan_replicas, reexec_into_replica

    if jax.default_backend() != "tpu":
        return
    mesh = spx.to_jax_mesh(
        ed.EasyDeLBaseConfig.create_mesh(sharding_axis_dims=dims, sharding_axis_names=EASYDEL_MESH_AXES)
    )
    from jax._src.distributed import global_state

    planned = plan_replicas(
        mesh.devices,
        mesh.axis_names,
        process_index=rank,
        all_devices=jax.devices(),
        local_devices=jax.local_devices(),
        address=resolve_bind_host("internal"),
        client=global_state.client,
        port=_replica_port(config),
        timeout_s=float(config.coordination.token_timeout_s),
    )
    if planned is None:
        _log(rank, "dp slices are not separate host boxes; keeping one rank-major engine over the whole mesh")
        return
    plan, env = planned
    _log(
        rank,
        f"dp={plan.count}: re-executing into replica {plan.index} as worker {plan.rank}/{plan.num_processes} "
        f"(topology {env['TPU_TOPOLOGY']}, coordinator {plan.coordinator})",
    )
    multihost_utils.sync_global_devices("easydel-esurge-serving-replica-split")
    jax.distributed.shutdown()
    reexec_into_replica(plan, env, token)


def _serve(config: ServingConfig, plan: tp.Any | None = None) -> None:
    if plan is not None and plan.is_front:
        _serve_front(config, plan)
        return
    started = time.monotonic()
    import jax

    if plan is not None:
        # Inside a dp replica: its own JAX world, joined before anything touches the TPU backend.
        jax.distributed.initialize(
            plan.coordinator,
            plan.num_processes,
            plan.rank,
            initialization_timeout=int(config.coordination.token_timeout_s),
            coordinator_bind_address=plan.coordinator,  # the replica leader's internal address, not every interface
        )

    import easydel as ed  # joins the multi-host JAX world (TPU slices are detected from GCE metadata)
    from easydel.utils.compilation_cache import cache_namespace, configure_compilation_cache

    configure_compilation_cache(cache_namespace(config.model.checkpoint), explicit=config.compilation_cache_dir)

    world_size = int(jax.process_count())
    rank = int(jax.process_index())
    is_leader = rank == 0
    tag = "" if plan is None else f" (dp replica {plan.index} of {plan.count})"
    _log(rank, f"joined world of {world_size} processes{tag}; easydel {ed.__version__} from {ed.__file__}")

    token = control_host = None
    if plan is not None:
        token = plan.read_token()
        control_host = resolve_bind_host(config.coordination.bind_host)
    elif world_size > 1:
        token = shared_coordination_token(
            config.coordination.control_port, is_leader=is_leader, timeout_s=config.coordination.token_timeout_s
        )
        control_host = resolve_bind_host(config.coordination.bind_host)
        _split_into_replicas(config, rank, token, ed)  # returns only when the launch stays one rank-major engine

    profiles = is_leader if plan is None else (plan.spawns_front and is_leader)
    if profiles and config.server.profiler_port is not None:
        jax.profiler.start_server(config.server.profiler_port)
        _log(rank, f"JAX profiler server on port {config.server.profiler_port}")

    text_config = None
    if config.model.quantization is not None and config.model.quantization.pattern is None:
        text_config = ed.AutoEasyDeLConfig.from_pretrained(config.model.checkpoint).get_text_config()
    sections = build_engine_sections(config, world_size=world_size, token=token, control_bind_host=control_host)
    if plan is None:
        loading = build_loading_kwargs(config, text_config, ed=ed)
    else:
        loading = build_loading_kwargs(config, text_config, ed=ed, sharding_axis_dims=replica_dims(config))
        sections = per_replica_sections(sections, plan.count)
    engine = ed.eSurge(
        model=config.model.checkpoint,
        processor=config.model.tokenizer_source,
        loading_kwargs=loading,
        **sections,
    )
    _log(rank, f"engine ready in {time.monotonic() - started:.1f}s (leader={engine.is_coordination_leader})")
    if plan is not None and engine.is_coordination_leader:
        table = getattr(getattr(engine, "runner", None), "step_time_table", None)
        if table:
            steps = {t: round(ms, 3) for t, ms in table.items()}
            _log(rank, f"replica {plan.index} model-step ms per token bucket: {steps}")
    try:
        if plan is not None and engine.is_coordination_leader:
            # A replica leader: the front's router reaches its engine over its request plane.
            _log(rank, f"replica {plan.index} leader serving its request plane on {control_host}")
            fronts: dict[int, tp.Any] = {}
            started_at: dict[int, float] = {}
            restarts: dict[int, int] = {}
            due: dict[int, float] = {}
            if plan.spawns_front:
                from .replicas import spawn_front

                for index in range(FRONT_PROCESSES):
                    fronts[index] = spawn_front(plan, str(token), index=index)
                    started_at[index], restarts[index] = time.monotonic(), 0
                pids = [f.pid for f in fronts.values()]
                _log(
                    rank,
                    f"started {len(fronts)} fronts (pids {pids}) for the OpenAI endpoint over {plan.count} replicas",
                )
            while True:
                now = time.monotonic()
                for index, front in list(fronts.items()):
                    if front is None:
                        if now >= due[index]:
                            fronts[index] = spawn_front(plan, str(token), index=index)
                            started_at[index] = time.monotonic()
                            _log(rank, f"restarted front {index} (pid {fronts[index].pid})")
                        continue
                    if front.poll() is None:
                        continue
                    # A front holds no engine state: restart it instead of taking this replica down with it.
                    lived = now - started_at[index]
                    restarts[index] = 0 if lived > FRONT_STABLE_S else restarts[index] + 1
                    wait = min(FRONT_MAX_BACKOFF_S, 2.0 ** restarts[index])
                    _log(
                        rank,
                        f"front {index} (pid {front.pid}) exited with code {front.returncode} after {lived:.0f}s; "
                        f"restarting it in {wait:.0f}s",
                    )
                    fronts[index], due[index] = None, now + wait
                time.sleep(1.0)
        if not engine.is_coordination_leader:
            try:
                engine.wait_for_shutdown()
            except ShutdownRequested:
                # `eray tpu job stop` signals every host at once; let the leader's Shutdown arrive before closing.
                _log(rank, f"SIGTERM; waiting up to {WORKER_SHUTDOWN_WAIT_S:.0f}s for the leader's shutdown")
                engine.wait_for_shutdown(timeout=WORKER_SHUTDOWN_WAIT_S)
                raise
            _log(rank, "leader shut the world down")
            return
        _serve_http(config, engine, rank, ed)
    finally:
        engine.close()


def _serve_front(config: ServingConfig, plan: tp.Any) -> None:
    """The front of a dp>1 launch: host-only, a router over every replica's request plane, the OpenAI endpoint.

    Started by replica 0's leader with JAX on the host CPU and no distributed world, so it never touches the TPU.
    """
    import easydel as ed

    token = plan.read_token()
    tag = f"front {plan.front_index}"
    _log(0, f"{tag} of {plan.count} dp replicas (host-only); easydel {ed.__version__} from {ed.__file__}")
    attacher = ReplicaAttacher(
        replica_candidates(plan), connect=_replica_connector(config, token), log=lambda m: _log(0, f"{tag}: {m}")
    )
    routed = attacher.wait_for_first()
    attacher.start()
    try:
        _serve_http(config, routed, 0, ed, shared_port=True)
    finally:
        attacher.stop()
        routed.close()


def replica_candidates(plan: tp.Any) -> tuple[tuple[str, ...], ...]:
    """Every host that may lead each replica's engine, by replica index (planned rank 0 first)."""
    if plan.hosts:
        return tuple(tuple(h) for h in plan.hosts)
    return tuple((leader,) for leader in plan.leaders)


def port_open(addr: str, port: int, timeout_s: float = 1.0) -> bool:
    """Whether something accepts TCP connections on ``addr:port`` (a cheap check before a request-plane handshake)."""
    import socket

    try:
        with socket.create_connection((addr, int(port)), timeout=timeout_s):
            return True
    except OSError:
        return False


def _served_model_type(config: ServingConfig) -> str | None:
    """The checkpoint's (text) ``model_type``, the parser auto-detection hint; ``None`` when it cannot be read."""
    try:
        import easydel as ed

        return getattr(
            ed.AutoEasyDeLConfig.from_pretrained(config.model.checkpoint).get_text_config(), "model_type", None
        )
    except Exception as exc:
        _log(0, f"could not read the checkpoint's model_type for parser auto-detection: {exc!r}")
        return None


def _replica_connector(config: ServingConfig, token: str) -> tp.Callable[[str], tp.Any]:
    """Open a request-plane client to one replica leader; raises ``RequestPlaneUnavailable`` while it is not up."""
    from easydel.inference.esurge.distributed.remote_engine import RemoteEngineHandle
    from easydel.inference.esurge.distributed.request_plane import RequestPlaneUnavailable

    # The front renders every reply itself, so it runs the same tool/reasoning parsers the engines would.
    parsing = dict(config.esurge.get("parsing") or {})
    model_type = _served_model_type(config)

    def connect(addr: str) -> tp.Any:
        # Only a replica's engine leader listens on the control port; skip the others without a handshake wait.
        if not port_open(addr, config.coordination.control_port):
            raise RequestPlaneUnavailable(f"nothing listening on {addr}:{config.coordination.control_port}")
        return RemoteEngineHandle(
            addr,
            config.coordination.control_port,
            auth_token=str(token),
            tokenizer_source=config.model.tokenizer_source,
            connect_timeout_s=10.0,
            tool_parser=parsing.get("tool_parser"),
            reasoning_parser=parsing.get("reasoning_parser"),
            model_type=model_type,
        )

    return connect


class ReplicaAttacher:
    """Builds the front's router from the replicas that are up, and keeps attaching the rest.

    The front serves as soon as one replica answers. Replicas still compiling (or restarting after a failure) are
    attached in the background when their request plane comes up, so one slow or dead replica never holds the
    endpoint down; a replica whose engine died is re-attached in its slot when it comes back.

    Args:
        leaders: By replica index, the address of its engine leader, or every address that may lead it (tried in
            order; the first that answers is attached).
        connect: Opens a client to one leader; raises ``RequestPlaneUnavailable`` while it is not up.
        log: Progress logger.
        interval_s: Wait between attach rounds.
        router_factory: Builds the router from the first attached member; defaults to ``RoutedEngine``.
    """

    def __init__(
        self,
        leaders: tp.Sequence[str | tp.Sequence[str]],
        *,
        connect: tp.Callable[[str], tp.Any],
        log: tp.Callable[[str], None],
        interval_s: float = REPLICA_ATTACH_INTERVAL_S,
        router_factory: tp.Callable[..., tp.Any] | None = None,
    ) -> None:
        import threading

        self.candidates = tuple((entry,) if isinstance(entry, str) else tuple(entry) for entry in leaders)
        # The address each replica was last attached at (its first candidate until then).
        self.leaders = [c[0] for c in self.candidates]
        self._router_factory = router_factory
        self._connect = connect
        self._log = log
        self._interval_s = float(interval_s)
        self._slots: dict[int, int] = {}  # replica index -> router member index
        self.router: tp.Any | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _try(self, index: int) -> tp.Any | None:
        from easydel.inference.esurge.distributed.request_plane import RequestPlaneError, RequestPlaneUnavailable

        for addr in self.candidates[index]:
            try:
                handle = self._connect(addr)
            except RequestPlaneUnavailable:
                continue
            except RequestPlaneError as exc:
                # Up but refusing (its world failed and the process has not exited yet): try again next round.
                self._log(f"replica {index} at {addr} refused: {exc}")
                continue
            self.leaders[index] = addr
            return handle
        return None

    def attach_round(self) -> int:
        """Try every replica that is not attached or whose engine died once; returns how many were attached."""
        from easydel.inference.esurge.distributed.routed_engine import RoutedEngine

        attached = 0
        for index in range(len(self.leaders)):
            slot = self._slots.get(index)
            old = None if slot is None or self.router is None else self.router.members[slot]
            if old is not None and getattr(old, "alive", True):
                continue
            handle = self._try(index)
            if handle is None:
                continue
            if self.router is None:
                factory = self._router_factory or RoutedEngine
                self.router = factory([handle], planned_members=len(self.leaders))
                self._slots[index] = 0
            elif slot is None:
                self._slots[index] = self.router.add_member(handle)
            else:
                self.router.replace_member(slot, handle)
                close = getattr(old, "close", None)
                if close is not None:
                    close()
            attached += 1
            verb = "re-attached" if old is not None else "attached"
            self._log(f"replica {index} {verb} at {self.leaders[index]} ({len(self._slots)}/{len(self.leaders)})")
        return attached

    def wait_for_first(self) -> tp.Any:
        """Block until at least one replica is attached; returns the router."""
        started = time.monotonic()
        last_note = started
        while self.router is None:
            self.attach_round()
            if self.router is None:
                if time.monotonic() - last_note > 60.0:
                    self._log(f"no replica up yet after {time.monotonic() - started:.0f}s; still waiting")
                    last_note = time.monotonic()
                time.sleep(self._interval_s)
        return self.router

    def start(self) -> None:
        """Keep attaching missing and dead replicas in the background."""
        import threading

        def _loop() -> None:
            while not self._stop.wait(self._interval_s):
                try:
                    self.attach_round()
                except Exception as exc:
                    self._log(f"replica attach round failed: {exc!r}")

        self._thread = threading.Thread(target=_loop, name="esurge-replica-attacher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30.0)


def reuse_port_socket(host: str, port: int) -> tp.Any:
    """A listening TCP socket on ``host:port`` that other processes may bind too (SO_REUSEPORT)."""
    import socket

    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((host, int(port)))
    sock.listen(2048)
    return sock


def _serve_http(config: ServingConfig, engine: tp.Any, rank: int, ed: tp.Any, *, shared_port: bool = False) -> None:
    key, created = load_or_create_api_key(config.server.api_key_path)
    name = config.model.served_name or engine.esurge_name
    server = ed.eSurgeApiServer(
        {name: engine},
        require_api_key=True,
        enable_persistence=False,
        enable_cors=False,
        max_concurrent_generations=config.server.max_concurrent_generations,
    )
    metadata = server.auth_manager.create_api_key(raw_key=key, name=config.server.key_name, created_by="serving")
    host = resolve_bind_host(config.server.host)
    urls = public_urls(host, config.server.port)
    _log(
        rank,
        f"READY model={name} key_id={metadata.key_id} key_file={config.server.api_key_path} "
        f"({'created' if created else 'existing'}) urls={urls}",
    )
    if not shared_port:
        server.run(host=host, port=config.server.port, workers=1, log_level=config.server.log_level, reload=False)
        return
    import asyncio

    import uvicorn

    from easydel.inference.inference_engine_interface import TIMEOUT_KEEP_ALIVE

    sock = reuse_port_socket(host, config.server.port)
    try:
        import uvloop

        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass
    uv_config = uvicorn.Config(
        server.app,
        log_level=config.server.log_level,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        server_header=False,
        date_header=True,
    )
    uvicorn.Server(uv_config).run(sockets=[sock])
