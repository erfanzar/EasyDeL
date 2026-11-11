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


def build_loading_kwargs(config: ServingConfig, text_config: tp.Any | None, *, ed: tp.Any) -> dict[str, tp.Any]:
    """Pretrained-loader kwargs for ``config.model``.

    Args:
        config: The deployment.
        text_config: The checkpoint's text config, needed only when quantization uses the model's default pattern.
        ed: The ``easydel`` module (injected so this stays importable without loading JAX in tests).

    Raises:
        ValueError: When quantization asks for the model's default pattern and the model has none.
    """
    import jax.numpy as jnp

    model = config.model
    kwargs: dict[str, tp.Any] = {
        "dtype": jnp.dtype(model.dtype),
        "param_dtype": jnp.dtype(model.param_dtype),
        "sharding_axis_dims": model.sharding_axis_dims,
        "sharding_axis_names": EASYDEL_MESH_AXES,
        "auto_shard_model": True,
    }
    if model.attn_mechanism:
        kwargs["config_kwargs"] = {"attn_mechanism": model.attn_mechanism}
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


def _log(rank: int, message: str) -> None:
    print(f"[esurge-serving rank={rank}] {message}", flush=True)


def serve(config: ServingConfig) -> None:
    """Join the slice, build the engine, and serve (leader) or replay steps (workers) until shutdown.

    SIGTERM (``eray tpu job stop``) ends it cleanly on every rank: the engine is closed and the call returns.
    """
    with sigterm_raises():
        try:
            _serve(config)
        except ShutdownRequested as exc:
            print(f"[esurge-serving] stopped by {exc}", flush=True)


def _serve(config: ServingConfig) -> None:
    started = time.monotonic()
    import jax

    import easydel as ed  # joins the multi-host JAX world (TPU slices are detected from GCE metadata)
    from easydel.utils.compilation_cache import cache_namespace, configure_compilation_cache

    configure_compilation_cache(cache_namespace(config.model.checkpoint), explicit=config.compilation_cache_dir)

    world_size = int(jax.process_count())
    rank = int(jax.process_index())
    is_leader = rank == 0
    _log(rank, f"joined world of {world_size} processes; easydel {ed.__version__} from {ed.__file__}")

    token = control_host = None
    if world_size > 1:
        token = shared_coordination_token(
            config.coordination.control_port, is_leader=is_leader, timeout_s=config.coordination.token_timeout_s
        )
        control_host = resolve_bind_host(config.coordination.bind_host)

    text_config = None
    if config.model.quantization is not None and config.model.quantization.pattern is None:
        text_config = ed.AutoEasyDeLConfig.from_pretrained(config.model.checkpoint).get_text_config()
    engine = ed.eSurge(
        model=config.model.checkpoint,
        processor=config.model.tokenizer_source,
        loading_kwargs=build_loading_kwargs(config, text_config, ed=ed),
        **build_engine_sections(config, world_size=world_size, token=token, control_bind_host=control_host),
    )
    _log(rank, f"engine ready in {time.monotonic() - started:.1f}s (leader={engine.is_coordination_leader})")
    try:
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


def _serve_http(config: ServingConfig, engine: tp.Any, rank: int, ed: tp.Any) -> None:
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
    server.run(host=host, port=config.server.port, workers=1, log_level=config.server.log_level, reload=False)
