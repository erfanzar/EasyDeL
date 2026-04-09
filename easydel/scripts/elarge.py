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
    if not isinstance(value, dict):
        raise SystemExit(f"{ctx} must be a mapping/dict, got {type(value).__name__}.")
    return value


def _require_str(value: Any, *, ctx: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"{ctx} must be a non-empty string.")
    return value


def _run_action(elm: Any, name: str, value: Any | None) -> None:
    """Execute a single named action on an eLargeModel instance.

    Supported actions: ``validate``, ``print``/``show``, ``dump_config``,
    ``to_json``, ``to_yaml``, ``train``, ``eval``, ``serve``.

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
            configured_tool_parser = elm.config.get("esurge", {}).get("tool_parser")
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

        elm.validate()
        surge = elm.build_esurge()
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
