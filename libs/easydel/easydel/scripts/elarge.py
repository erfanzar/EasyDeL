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

"""Unified eLargeModel runner driven by YAML.

This module provides a single entrypoint to run common workflows (train/eval/etc.)
using `easydel.infra.eLargeModel`, configured from a YAML file.

Usage:
    python -m easydel.scripts.elarge --config path/to/config.yaml

YAML schema (recommended):
    config:
      model:
        name_or_path: Qwen/Qwen3-8B
      trainer:
        trainer_type: sft
      mixture:
        informs:
          - type: json
            data_files: train.jsonl
            content_field: text
    actions:
      - validate
      - print
      - train
      - eval:
          tasks: ["gsm8k", "mmlu"]
          num_fewshot: 5
          output_path: eval_results.json
      - serve:
          host: "0.0.0.0"
          port: 11556

You can also omit the `config:` wrapper and place ELM config keys at the root,
as long as `actions:` exists.
"""

from __future__ import annotations

import json
import pprint
from dataclasses import dataclass, field
from typing import Any

from eformer.aparser import DataClassArgumentParser
from eformer.loggings import get_logger

logger = get_logger("eLargeScript")


@dataclass
class ElargeArgs:
    """Command-line arguments for the eLargeModel YAML runner.

    Attributes:
        config: Path to the YAML configuration file.
        dry_run: If ``True``, parse and print the config/actions without
            executing them.
    """

    config: str | None = field(default=None, metadata={"help": "Path to YAML config.", "aliases": ["-c"]})
    dry_run: bool = field(default=False, metadata={"help": "Parse and print config/actions, then exit."})


def _load_yaml(path: str) -> dict[str, Any]:
    """Load and parse a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        SystemExit: If PyYAML is not installed or the file is malformed.
    """
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "PyYAML is required to run `python -m easydel.scripts.elarge`.\n"
            "Install it with `pip install pyyaml` (or add it to your environment)."
        ) from exc

    with open(path, encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise SystemExit(f"YAML root must be a mapping/dict, got {type(loaded).__name__}.")
    return loaded


def _extract_config_and_actions(doc: dict[str, Any]) -> tuple[dict[str, Any], list[Any]]:
    """Extract the eLargeModel config and action list from a parsed YAML document.

    Supports multiple layout conventions: ``config:`` key, ``elarge_model:``
    key, ``elm:`` key, or all top-level keys except ``actions:``.

    Args:
        doc: Parsed YAML document as a dictionary.

    Returns:
        Tuple of (config_dict, actions_list).

    Raises:
        SystemExit: If ``actions`` key is missing or has an invalid type.
    """
    actions = doc.get("actions")
    if actions is None:
        raise SystemExit("Config YAML must contain an `actions:` list.")

    config = None
    for key in ("config", "elarge_model", "elm"):
        if key in doc:
            config = doc[key]
            break
    if config is None:
        config = {k: v for k, v in doc.items() if k != "actions"}

    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise SystemExit(f"`config` must be a mapping/dict, got {type(config).__name__}.")

    if isinstance(actions, list):
        return config, actions
    if isinstance(actions, str | dict):
        return config, [actions]
    raise SystemExit(f"`actions` must be a list (or a single action), got {type(actions).__name__}.")


def _parse_action(item: Any) -> tuple[str, Any | None]:
    """Parse a single action entry from the YAML actions list.

    Handles three formats:
    - A plain string (e.g. ``"train"``).
    - A single-key dict (e.g. ``{eval: {tasks: [...]}}``)
    - A dict with one ``None``-valued key (common YAML indentation mistake).

    Args:
        item: Raw action item from the YAML actions list.

    Returns:
        Tuple of (action_name, optional_parameters).

    Raises:
        SystemExit: If the action item cannot be parsed.
    """
    if isinstance(item, str):
        return item, None
    if isinstance(item, dict):
        if len(item) == 1:
            name, value = next(iter(item.items()))
            if not isinstance(name, str):
                raise SystemExit("Action name must be a string.")
            return name, value

        # Recover from a common YAML indentation mistake:
        #   - serve:
        #     host: ...
        #     port: ...
        # which parses as {"serve": None, "host": "...", "port": ...}.
        none_keys = [key for key, value in item.items() if value is None]
        if len(none_keys) == 1 and isinstance(none_keys[0], str):
            name = none_keys[0]
            params = {key: value for key, value in item.items() if key != name}
            return name, params or None

        raise SystemExit(
            "Action mappings must have exactly one key (e.g. `{eval: {...}}`). "
            "If you meant to pass parameters, indent them under the action key (e.g. `- serve: {host: ..., port: ...}` "
            "or `- serve:\\n    host: ...`)."
        )
    raise SystemExit(f"Invalid action item type: {type(item).__name__}.")


def _require_mapping(value: Any, *, ctx: str) -> dict[str, Any]:
    """Assert that ``value`` is a dict and return it.

    Args:
        value: Value to validate.
        ctx: Description used in the error message.

    Returns:
        The validated dictionary.

    Raises:
        SystemExit: If ``value`` is not a mapping.
    """
    if not isinstance(value, dict):
        raise SystemExit(f"{ctx} must be a mapping/dict, got {type(value).__name__}.")
    return value


def _require_str(value: Any, *, ctx: str) -> str:
    """Assert that ``value`` is a non-empty string and return it.

    Args:
        value: Value to validate.
        ctx: Description used in the error message.

    Returns:
        The validated string.

    Raises:
        SystemExit: If ``value`` is not a non-empty string.
    """
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"{ctx} must be a non-empty string.")
    return value


def _plan_replica_tpu_envs(count: int) -> list[dict[str, str]]:
    """Carve this host's TPU chips into per-replica visibility environments.

    Detects the chip count lock-free via ``/dev/accel*`` (the parent must
    never initialize a TPU runtime — the replicas own the chips) and writes
    the same visibility variables Ray's TPU accelerator manager uses for
    1/2/4-chip subsets. Chips are assigned contiguously so 2-chip splits
    land on aligned pairs ({0,1}, {2,3}).

    Args:
        count: Number of replica processes to carve the host into.

    Returns:
        One environment dict per replica; empty dicts when the host has no
        TPU chips (CPU/GPU replicas manage their own visibility).

    Raises:
        SystemExit: If the chip count does not divide evenly or the
            per-replica subset size has no known bounds layout.
    """
    import glob

    num_chips = len(glob.glob("/dev/accel*")) or len(glob.glob("/dev/vfio/*")) - (
        1 if glob.glob("/dev/vfio/vfio") else 0
    )
    if num_chips <= 0:
        return [{} for _ in range(count)]
    if num_chips % count != 0:
        raise SystemExit(f"`serve.replicas.count={count}` must divide the host's {num_chips} TPU chips evenly.")
    chips_per_replica = num_chips // count
    bounds = {1: "1,1,1", 2: "1,2,1", 4: "2,2,1"}.get(chips_per_replica)
    if bounds is None:
        raise SystemExit(
            f"Unsupported chips-per-replica={chips_per_replica}; single-host splits support 1, 2, or 4 chips."
        )
    envs = []
    for index in range(count):
        chip_ids = range(index * chips_per_replica, (index + 1) * chips_per_replica)
        envs.append(
            {
                "TPU_VISIBLE_CHIPS": ",".join(str(chip) for chip in chip_ids),
                "TPU_CHIPS_PER_HOST_BOUNDS": bounds,
                "TPU_HOST_BOUNDS": "1,1,1",
            }
        )
    return envs


def _serve_replicated(
    *,
    elm: Any,
    replicas_params: dict[str, Any],
    server_kwargs: dict[str, Any],
    host: str,
    port: int,
    log_level: str,
    ssl_keyfile: str | None,
    ssl_certfile: str | None,
) -> None:
    """Serve N independent engine replicas behind one routed API server.

    Two substrates:

    * **Spawn** (default): launch ``count`` subprocesses of this same
      runner, each executing a generated ``serve_replica`` overlay of the
      current config — its own JAX world on a carved chip subset, exposing
      the request plane on ``base_control_port + i``. This is single-host
      chip-split data parallelism in one command.
    * **Attach** (``endpoints:`` set): connect to externally-launched
      replicas (e.g. one ``serve_replica`` run per pod host via eray) and
      only run the router here.

    The parent stays off the accelerator entirely (its JAX is pinned to
    CPU); it renders through :class:`RemoteEngineHandle` clients and fronts
    them with a prefix-affinity + JSQ :class:`RoutedEngine`.

    Args:
        elm: The parsed ``eLargeModel`` (config source; never built here).
        replicas_params: The ``serve.replicas`` mapping.
        server_kwargs: Validated ``eSurgeApiServer`` kwargs.
        host: API-server bind host.
        port: API-server port.
        log_level: Uvicorn log level.
        ssl_keyfile: Optional TLS key path.
        ssl_certfile: Optional TLS cert path.

    Raises:
        SystemExit: On invalid parameters, a replica exiting during
            startup, or the ready deadline lapsing.
    """
    import hashlib
    import os
    import subprocess
    import sys
    import tempfile
    import time

    if os.environ.get("_ELARGE_DP_ROUTER") != "1":
        # This process already imported easydel/jax, which initializes the
        # local accelerator at import time — and the replicas need those
        # chips. Re-exec ourselves CPU-pinned: the exec replaces the
        # process image, releasing the TPU before replicas spawn. libtpu
        # opens its device fds (/dev/vfio/*, /dev/accel*) without
        # close-on-exec, so mark every inherited fd CLOEXEC first or the
        # re-exec'd router would keep the chips locked forever.
        import fcntl

        for fd_name in os.listdir("/proc/self/fd"):
            try:
                fd = int(fd_name)
                if fd > 2:
                    flags = fcntl.fcntl(fd, fcntl.F_GETFD)
                    fcntl.fcntl(fd, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)
            except (OSError, ValueError):
                continue
        environ = dict(os.environ)
        environ["_ELARGE_DP_ROUTER"] = "1"
        environ["JAX_PLATFORMS"] = "cpu"
        environ["ENABLE_DISTRIBUTED_INIT"] = "0"
        logger.info("Re-executing the DP-serve router CPU-pinned; replicas own the accelerator.")
        os.execve(sys.executable, [sys.executable, "-m", "easydel.scripts.elarge", *sys.argv[1:]], environ)

    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required for `serve.replicas`.") from exc

    endpoints = replicas_params.get("endpoints")

    # Replica count + per-replica config come from the mesh `dp` axis:
    # replica-DP peels `dp` into independent processes, so the count
    # defaults to that axis size and each replica's own mesh collapses `dp`
    # to 1 (its remaining axes — usually `tp` — size against its carved
    # chip subset). An explicit `serve.replicas.count` overrides the
    # derived value. json-roundtrip makes the config yaml-safe and its dims
    # mutable for the collapse.
    config_dict = json.loads(json.dumps(elm.to_dict(), default=str))
    sharding = config_dict.get("sharding") or {}
    axis_names = list(sharding.get("axis_names") or ())
    axis_dims = list(sharding.get("axis_dims") or ())
    has_dp = "dp" in axis_names and len(axis_dims) == len(axis_names)
    dp_size = int(axis_dims[axis_names.index("dp")]) if has_dp else 1

    explicit_count = replicas_params.get("count")
    if explicit_count is not None:
        count = int(explicit_count)
    elif endpoints:
        count = len(endpoints)
    elif dp_size > 1:
        count = dp_size
    else:
        raise SystemExit(
            "`serve.replicas` needs a replica count: set the mesh `dp` axis > 1 in `sharding.axis_dims`, "
            "pass `serve.replicas.count`, or provide `serve.replicas.endpoints`."
        )
    if not endpoints and count < 2:
        raise SystemExit(f"`serve.replicas` resolved count={count}; need >= 2 replicas (or `endpoints`).")

    # Each replica is its own process, so its mesh must not also fan out dp.
    if has_dp:
        axis_dims[axis_names.index("dp")] = 1
        sharding["axis_dims"] = axis_dims
        config_dict["sharding"] = sharding

    base_control_port = int(replicas_params.get("base_control_port", 19700))
    ready_timeout_s = float(replicas_params.get("ready_timeout_s", 1800.0))
    auth_token = replicas_params.get("auth_token") or hashlib.sha256(
        json.dumps(elm.to_dict(), sort_keys=True, default=str).encode()
    ).hexdigest()

    # The replicas own the accelerator; any incidental JAX use in this
    # router process must never grab the TPU (libtpu locks per chip).
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    procs: list[subprocess.Popen] = []
    log_paths: list[str] = []
    open_logs: list[Any] = []
    if endpoints:
        targets = []
        for entry in endpoints:
            entry = str(entry)
            addr, _, entry_port = entry.rpartition(":")
            if not addr or not entry_port.isdigit():
                raise SystemExit(f"`serve.replicas.endpoints` entries must be 'host:port', got {entry!r}.")
            targets.append((addr, int(entry_port)))
    else:
        workdir = str(replicas_params.get("workdir") or tempfile.mkdtemp(prefix="elarge-dp-serve-"))
        os.makedirs(workdir, exist_ok=True)
        chip_envs = _plan_replica_tpu_envs(count)
        platform_override = replicas_params.get("platform")
        targets = []
        for index in range(count):
            control_port = base_control_port + index
            overlay = {
                "config": config_dict,
                "actions": [
                    {
                        "serve_replica": {
                            "control_port": control_port,
                            "auth_token": auth_token,
                            "bind_host": "127.0.0.1",
                        }
                    }
                ],
            }
            overlay_path = os.path.join(workdir, f"replica_{index}.yaml")
            with open(overlay_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(overlay, handle, sort_keys=False)

            env = dict(os.environ)
            env.update(chip_envs[index])
            env["ENABLE_DISTRIBUTED_INIT"] = "0"
            # Replicas auto-detect their accelerator; drop the router's
            # CPU pin and re-exec marker unless the YAML forces a platform.
            env.pop("JAX_PLATFORMS", None)
            env.pop("_ELARGE_DP_ROUTER", None)
            if platform_override:
                env["JAX_PLATFORMS"] = str(platform_override)

            log_path = os.path.join(workdir, f"replica_{index}.log")
            log_file = open(log_path, "w")
            log_paths.append(log_path)
            open_logs.append(log_file)
            procs.append(
                subprocess.Popen(
                    [sys.executable, "-m", "easydel.scripts.elarge", "--config", overlay_path],
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            )
            targets.append(("127.0.0.1", control_port))
        logger.info("Spawned %d replica processes (logs under %s); waiting for their request planes.", count, workdir)

    from easydel.inference import eSurgeApiServer
    from easydel.inference.esurge.distributed.remote_engine import RemoteEngineHandle
    from easydel.inference.esurge.distributed.request_plane import RequestPlaneUnavailable
    from easydel.inference.esurge.distributed.routed_engine import RoutedEngine

    handles: list[RemoteEngineHandle] = []
    try:
        deadline = time.time() + ready_timeout_s
        for index, (addr, control_port) in enumerate(targets):
            while True:
                if procs and procs[index].poll() is not None:
                    tail = ""
                    if log_paths:
                        with open(log_paths[index], encoding="utf-8", errors="replace") as handle:
                            tail = handle.read()[-2000:]
                    raise SystemExit(f"Replica {index} exited during startup (code {procs[index].returncode}).\n{tail}")
                try:
                    handles.append(
                        RemoteEngineHandle(
                            addr,
                            control_port,
                            auth_token=auth_token,
                            tokenizer_source=replicas_params.get("tokenizer_source"),
                            connect_timeout_s=10.0,
                        )
                    )
                    logger.info("Replica %d ready at %s:%d.", index, addr, control_port)
                    break
                except RequestPlaneUnavailable:
                    # Still compiling before it binds the control socket.
                    if time.time() > deadline:
                        raise SystemExit(
                            f"Replica {index} at {addr}:{control_port} not ready within {ready_timeout_s}s."
                        ) from None
                    time.sleep(5.0)
                except Exception as exc:
                    raise SystemExit(f"Replica {index} at {addr}:{control_port} attach failed: {exc}") from exc

        router = RoutedEngine(handles)
        server = eSurgeApiServer(router, **server_kwargs)
        logger.info("Routing across %d replicas; serving on %s:%d.", len(handles), host, port)
        server.run(
            host=host,
            port=port,
            workers=1,
            log_level=log_level,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
            reload=False,
        )
    finally:
        for handle in handles:
            try:
                handle.close()
            except Exception:
                pass
        for proc in procs:
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        for log_file in open_logs:
            log_file.close()


def _run_action(elm: Any, name: str, value: Any | None) -> None:
    """Execute a single named action on an eLargeModel instance.

    Supported actions: ``validate``, ``print``/``show``, ``dump_config``,
    ``to_json``, ``to_yaml``, ``train``, ``eval``, ``serve``,
    ``serve_replica``.

    Args:
        elm: An ``eLargeModel`` instance.
        name: Action name string.
        value: Optional parameters for the action (from YAML).

    Raises:
        SystemExit: If the action is unknown or parameters are invalid.
    """
    action = name.strip().lower().replace("-", "_")

    if action == "validate":
        elm.validate()
        return

    if action in {"print", "show"}:
        logger.info(elm)
        return

    if action in {"dump_config", "print_config", "config"}:
        logger.info(pprint.pformat(elm.to_dict()))
        return

    if action in {"to_json", "save_json", "write_json"}:
        if value is None:
            raise SystemExit("`to_json` requires a path (string) or `{path: ...}`.")
        if isinstance(value, dict):
            path = _require_str(value.get("path"), ctx="`to_json.path`")
        else:
            path = _require_str(value, ctx="`to_json` value")
        elm.to_json(path)
        return

    if action in {"to_yaml", "save_yaml", "write_yaml"}:
        if value is None:
            raise SystemExit("`to_yaml` requires a path (string) or `{path: ...}`.")
        if isinstance(value, dict):
            path = _require_str(value.get("path"), ctx="`to_yaml.path`")
        else:
            path = _require_str(value, ctx="`to_yaml` value")
        elm.to_yaml(path)
        return

    if action == "train":
        elm.train()
        return

    if action == "eval":
        params = _require_mapping(value, ctx="`eval` action")
        tasks = params.get("tasks")
        if tasks is None:
            raise SystemExit("`eval` action requires `tasks` (string or list of strings).")

        num_fewshot = params.get("num_fewshot", 0)
        if not isinstance(num_fewshot, int):
            raise SystemExit("`eval.num_fewshot` must be an int.")

        output_path = params.get("output_path")
        if output_path is not None and not isinstance(output_path, str):
            raise SystemExit("`eval.output_path` must be a string if provided.")

        results = elm.eval(
            tasks=tasks,
            num_fewshot=num_fewshot,
            output_path=output_path,
        )
        if params.get("print_results", False):
            logger.info(json.dumps(results, indent=2))
        return

    if action == "serve_replica":
        # One DP replica: its own JAX world exposing the request plane on a
        # workerless ZMQ leader; a routed API server (or any request-plane
        # client) fronts it. Blocks until the parent terminates the process.
        params = _require_mapping(value, ctx="`serve_replica` action")
        try:
            control_port = int(params["control_port"])
        except (KeyError, TypeError, ValueError) as e:
            raise SystemExit("`serve_replica.control_port` (int) is required.") from e
        auth_token = _require_str(params.get("auth_token"), ctx="`serve_replica.auth_token`")
        bind_host = str(params.get("bind_host", "0.0.0.0"))
        elm.set_esurge(
            coordination="zmq",
            distributed_auth_token=auth_token,
            distributed_control_port=control_port,
            distributed_control_bind_host=bind_host,
        )
        elm.validate()
        surge = elm.build_esurge()
        logger.info(
            "Replica engine %r ready: request plane on %s:%d.",
            surge.esurge_name,
            bind_host,
            control_port,
        )
        import threading

        threading.Event().wait()
        return

    if action in {"serve", "server"}:
        params: dict[str, Any] = {}
        if value is not None:
            params = _require_mapping(value, ctx="`serve` action")

        host = params.get("host", "0.0.0.0")
        if not isinstance(host, str) or not host.strip():
            raise SystemExit("`serve.host` must be a non-empty string.")

        port = params.get("port", 11556)
        try:
            port_int = int(port)
        except (TypeError, ValueError) as e:
            raise SystemExit("`serve.port` must be an int.") from e

        workers = params.get("workers", 1)
        try:
            workers_int = int(workers)
        except (TypeError, ValueError) as e:
            raise SystemExit("`serve.workers` must be an int.") from e
        if workers_int != 1:
            raise SystemExit("`serve.workers` must be 1 when serving an in-process eSurge engine.")

        log_level = params.get("log_level", "info")
        if not isinstance(log_level, str) or not log_level.strip():
            raise SystemExit("`serve.log_level` must be a non-empty string.")

        reload = bool(params.get("reload", False))
        if reload:
            raise SystemExit("`serve.reload` is not supported for in-process servers. Use `workers: 1` and restart.")

        ssl_keyfile = params.get("ssl_keyfile")
        if ssl_keyfile is not None and not isinstance(ssl_keyfile, str):
            raise SystemExit("`serve.ssl_keyfile` must be a string if provided.")

        ssl_certfile = params.get("ssl_certfile")
        if ssl_certfile is not None and not isinstance(ssl_certfile, str):
            raise SystemExit("`serve.ssl_certfile` must be a string if provided.")

        try:
            from easydel.inference import eSurgeApiServer
        except ImportError as e:
            raise SystemExit("Serving requires `fastapi` and `uvicorn` to be installed in your environment.") from e

        tool_parser_name = params.get("tool_parser_name", "auto")
        if not isinstance(tool_parser_name, str) or not tool_parser_name.strip():
            raise SystemExit("`serve.tool_parser_name` must be a non-empty string.")
        tool_parser_name = tool_parser_name.strip()
        if tool_parser_name.lower() != "auto":
            esurge_config = elm.config.get("esurge", {})
            parsing_config = esurge_config.get("parsing", {}) if isinstance(esurge_config, dict) else {}
            configured_tool_parser = parsing_config.get("tool_parser", esurge_config.get("tool_parser"))
            if configured_tool_parser is None:
                logger.warning(
                    "`serve.tool_parser_name=%s` is deprecated; migrating it to `esurge.tool_parser` for this run.",
                    tool_parser_name,
                )
                elm.set_esurge(tool_parser=tool_parser_name)
            elif str(configured_tool_parser).strip() != tool_parser_name:
                raise SystemExit(
                    "`serve.tool_parser_name` is deprecated and disagrees with `esurge.tool_parser`. "
                    "Remove the `serve.tool_parser_name` override and keep only `esurge.tool_parser`."
                )
            else:
                logger.warning(
                    "`serve.tool_parser_name=%s` is deprecated and ignored; using `esurge.tool_parser`.",
                    tool_parser_name,
                )

        server_kwargs: dict[str, Any] = {
            "oai_like_processor": bool(params.get("oai_like_processor", True)),
            "enable_function_calling": bool(params.get("enable_function_calling", True)),
            "require_api_key": bool(params.get("require_api_key", False)),
            "admin_key": params.get("admin_key"),
        }
        if server_kwargs["admin_key"] is not None and not isinstance(server_kwargs["admin_key"], str):
            raise SystemExit("`serve.admin_key` must be a string if provided.")

        # BaseInferenceApiServer kwargs (optional)
        if "enable_cors" in params:
            server_kwargs["enable_cors"] = bool(params["enable_cors"])
        if "cors_origins" in params:
            cors_origins = params["cors_origins"]
            if cors_origins is not None and not isinstance(cors_origins, list):
                raise SystemExit("`serve.cors_origins` must be a list of strings (or null).")
            if isinstance(cors_origins, list) and any(not isinstance(v, str) for v in cors_origins):
                raise SystemExit("`serve.cors_origins` must be a list of strings (or null).")
            server_kwargs["cors_origins"] = cors_origins

        if "replicas" in params:
            # Data-parallel serving: N independent replica engines behind
            # one routed server. The parent never builds a model or touches
            # the accelerator, so skip validate() (each replica validates).
            # An empty `replicas:` block is valid — count derives from the
            # mesh `dp` axis inside _serve_replicated.
            replicas_value = params["replicas"]
            _serve_replicated(
                elm=elm,
                replicas_params=_require_mapping(replicas_value or {}, ctx="`serve.replicas`"),
                server_kwargs=server_kwargs,
                host=host,
                port=port_int,
                log_level=log_level,
                ssl_keyfile=ssl_keyfile,
                ssl_certfile=ssl_certfile,
            )
            return

        elm.validate()

        import jax

        if int(jax.process_count()) > 1:
            # Serving on a multi-process runtime needs the step-coordination
            # plane: requests arrive at rank 0 only, but the jitted step is a
            # global collective. Enable coordination="zmq" automatically with
            # a config-derived auth token (identical on every host because
            # the YAML is identical), unless the YAML configured it already.
            esurge_section = elm.config.get("esurge", {}) or {}
            dist_section = esurge_section.get("distributed", {}) if isinstance(esurge_section, dict) else {}
            has_plane = str(dist_section.get("coordination", "")) == "zmq"
            if not has_plane:
                import hashlib
                import json as _json

                auth = dist_section.get("distributed_auth_token") or hashlib.sha256(
                    _json.dumps(elm.config, sort_keys=True, default=str).encode()
                ).hexdigest()
                elm.set_esurge(coordination="zmq", distributed_auth_token=auth)
                logger.info(
                    "Multi-process serve: enabled esurge step coordination "
                    "(coordination='zmq', %d processes, rank %d).",
                    int(jax.process_count()),
                    int(jax.process_index()),
                )

        surge = elm.build_esurge()

        if int(jax.process_count()) > 1 and int(jax.process_index()) != 0:
            logger.info("Worker rank %d: replaying leader steps; the API server runs on rank 0.", jax.process_index())
            replay_thread = getattr(surge, "_worker_replay_thread", None)
            if replay_thread is None:
                raise SystemExit(
                    "Worker rank has no replay loop; ensure esurge.distributed.coordination='zmq' on every host."
                )
            replay_thread.join()
            return

        server = eSurgeApiServer(surge, **server_kwargs)
        server.run(
            host=host,
            port=port_int,
            workers=workers_int,
            log_level=log_level,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
            reload=False,
        )
        return

    raise SystemExit(f"Unknown action: {name!r}.")


def main(argv: list[str] | None = None) -> None:
    """Entry point for the eLargeModel YAML runner.

    Parses a YAML config file, constructs an ``eLargeModel`` instance,
    and executes the declared actions in order.

    Args:
        argv: Command-line arguments. Uses ``sys.argv`` when ``None``.
    """
    parser = DataClassArgumentParser(
        ElargeArgs,
        description="Unified YAML runner for easydel.infra.eLargeModel.",
    )
    parser.add_argument("config_pos", nargs="?", help="Path to YAML config.")
    args, extra = parser.parse_args_into_dataclasses(args=argv, look_for_args_file=False)

    config_path = args.config or getattr(extra, "config_pos", None)
    if not config_path:
        raise SystemExit("Missing config path. Provide `--config path.yaml` (or a positional path).")

    doc = _load_yaml(config_path)
    _, actions = _extract_config_and_actions(doc)
    from easydel.infra import eLargeModel

    elm = eLargeModel.from_yaml(config_path)

    if args.dry_run:
        logger.info("actions:")
        logger.info(pprint.pformat(actions))
        logger.info("\nnormalized_config:")
        logger.info(pprint.pformat(elm.to_dict()))
        return

    for item in actions:
        name, value = _parse_action(item)
        _run_action(elm, name, value)


if __name__ == "__main__":
    main()
