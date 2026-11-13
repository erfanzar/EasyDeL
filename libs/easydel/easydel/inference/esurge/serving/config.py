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

"""Typed configuration for serving one eSurge engine over the OpenAI API on a multi-host TPU slice.

One YAML file describes a deployment; every host of the slice loads the same file. Sections:

* ``model``: checkpoint, tokenizer, mesh, dtypes, attention kernel and optional quantization.
* ``esurge``: the engine's own sections (``runtime``, ``cache``, ``context``, ``workers``, ``parsing``), passed
  verbatim to :class:`~easydel.inference.esurge.config.eSurgeConfig`, which validates them. ``distributed`` is
  not allowed here: coordination is owned by the ``coordination`` section.
* ``server``: HTTP bind address, port and the API-key file.
* ``coordination``: the ZMQ step-coordination control plane.

The literal ``internal`` for a bind address means this VM's primary internal (VPC) address.
"""

from __future__ import annotations

import dataclasses
import typing as tp
from collections.abc import Mapping
from pathlib import Path

import yaml

INTERNAL_ADDRESS = "internal"
"""Bind-address value that resolves to the host's internal VPC address at startup."""

EASYDEL_MESH_AXES = ("pp", "dp", "fsdp", "ep", "tp", "sp")


class ServingConfigError(ValueError):
    """The serving YAML does not describe a valid deployment."""


def _section(data: Mapping[str, tp.Any], name: str, cls: type) -> tp.Any:
    """Build dataclass ``cls`` from ``data[name]``, rejecting unknown keys."""
    raw = data.get(name) or {}
    if not isinstance(raw, Mapping):
        raise ServingConfigError(f"`{name}` must be a mapping, got {type(raw).__name__}")
    fields = {field.name for field in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - fields)
    if unknown:
        raise ServingConfigError(f"`{name}` has unknown keys {unknown}; allowed: {sorted(fields)}")
    try:
        return cls(**raw)
    except TypeError as exc:
        raise ServingConfigError(f"`{name}`: {exc}") from exc


def _port(value: tp.Any, what: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 65535:
        raise ServingConfigError(f"{what} must be an integer port in 1..65535, got {value!r}")
    return value


@dataclasses.dataclass(frozen=True)
class QuantizationSpec:
    """Load-time quantization.

    Attributes:
        dtype: EasyDeL quantization dtype, e.g. ``"w8a8"``.
        pattern: Regex over parameter paths. ``None`` asks the model's text config for its
            ``default_quantization_pattern(dtype)``.
    """

    dtype: str
    pattern: str | None = None


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    """What to load and how to shard it.

    Attributes:
        checkpoint: EasyDeL checkpoint id or path (``gs://`` is supported).
        tokenizer: Hugging Face tokenizer id or local path. Defaults to ``checkpoint``; set it when the
            checkpoint is on GCS, which Hugging Face tokenizers cannot read.
        served_name: Model name clients send in ``model``. Defaults to the engine's own name.
        sharding_axis_dims: Mesh sizes in EasyDeL order ``(pp, dp, fsdp, ep, tp, sp)``; ``-1`` fills.
        dtype: Activation dtype.
        param_dtype: Parameter storage dtype.
        attn_mechanism: Attention kernel name, or ``None`` for the model default.
        quantization: Optional load-time quantization.
    """

    checkpoint: str
    tokenizer: str | None = None
    served_name: str | None = None
    sharding_axis_dims: tuple[int, ...] = (1, 1, -1, 1, 1, 1)
    dtype: str = "bfloat16"
    param_dtype: str = "bfloat16"
    attn_mechanism: str | None = None
    quantization: QuantizationSpec | None = None

    def __post_init__(self) -> None:
        if not self.checkpoint:
            raise ServingConfigError("`model.checkpoint` is required")
        dims = tuple(int(d) for d in self.sharding_axis_dims)
        if len(dims) != len(EASYDEL_MESH_AXES):
            raise ServingConfigError(
                f"`model.sharding_axis_dims` needs {len(EASYDEL_MESH_AXES)} entries {EASYDEL_MESH_AXES}"
            )
        if any(d == 0 or d < -1 for d in dims) or dims.count(-1) > 1:
            raise ServingConfigError("`model.sharding_axis_dims` entries must be positive, with at most one -1")
        object.__setattr__(self, "sharding_axis_dims", dims)
        if isinstance(self.quantization, Mapping):
            object.__setattr__(self, "quantization", _section({"q": self.quantization}, "q", QuantizationSpec))

    @property
    def tokenizer_source(self) -> str:
        return self.tokenizer or self.checkpoint


@dataclasses.dataclass(frozen=True)
class ServerSpec:
    """The OpenAI-compatible HTTP server, run by the coordination leader only.

    Attributes:
        host: Bind address; ``internal`` (default) is this VM's internal address, ``0.0.0.0`` all interfaces.
        port: HTTP port.
        api_key_file: File holding the one API key. Read when present, created (mode 0600) when absent, so
            the key is stable across relaunches.
        key_name: Label of that key in the auth manager.
        log_level: uvicorn log level.
        max_concurrent_generations: In-flight generation cap; ``None`` uses the engine's request cap.
    """

    host: str = INTERNAL_ADDRESS
    port: int = 8000
    api_key_file: str = "~/.config/esurge/api_key"
    key_name: str = "esurge-serving"
    log_level: str = "info"
    max_concurrent_generations: int | None = None

    def __post_init__(self) -> None:
        _port(self.port, "`server.port`")
        if not self.api_key_file:
            raise ServingConfigError("`server.api_key_file` is required: the server never runs without a key")

    @property
    def api_key_path(self) -> Path:
        return Path(self.api_key_file).expanduser()


@dataclasses.dataclass(frozen=True)
class CoordinationSpec:
    """ZMQ step-coordination plane between the leader and the worker processes.

    The shared token is generated per launch on the leader and handed to workers through the JAX
    coordination service; it is never written to disk and is unrelated to the API key.

    Attributes:
        control_port: Leader control-plane port.
        bind_host: Leader control-plane bind address (``internal`` by default; never exposed publicly).
        heartbeat_timeout_s: Worker silence tolerated before the world aborts.
        token_timeout_s: How long workers wait for the leader to publish the token.
    """

    control_port: int = 19666
    bind_host: str = INTERNAL_ADDRESS
    heartbeat_timeout_s: float = 120.0
    token_timeout_s: float = 600.0

    def __post_init__(self) -> None:
        _port(self.control_port, "`coordination.control_port`")
        if float(self.heartbeat_timeout_s) <= 0 or float(self.token_timeout_s) <= 0:
            raise ServingConfigError("`coordination` timeouts must be positive")


@dataclasses.dataclass(frozen=True)
class ServingConfig:
    """A complete serving deployment. Build it with :meth:`load` or :meth:`from_mapping`.

    Attributes:
        model: What to load.
        esurge: Engine sections (``runtime``, ``cache``, ...) for :class:`eSurgeConfig`.
        server: HTTP server.
        coordination: Multi-host control plane.
        compilation_cache_dir: JAX persistent compilation cache directory. ``None`` keeps a directory JAX already
            has and otherwise, on Google Cloud, uses an owned regional bucket automatically
            (:mod:`easydel.utils.compilation_cache`).
    """

    model: ModelSpec
    esurge: Mapping[str, tp.Any] = dataclasses.field(default_factory=dict)
    server: ServerSpec = dataclasses.field(default_factory=ServerSpec)
    coordination: CoordinationSpec = dataclasses.field(default_factory=CoordinationSpec)
    compilation_cache_dir: str | None = None

    _TOP_LEVEL: tp.ClassVar[frozenset[str]] = frozenset(
        {"model", "esurge", "server", "coordination", "compilation_cache_dir"}
    )

    @classmethod
    def from_mapping(cls, data: Mapping[str, tp.Any]) -> ServingConfig:
        """Validate a parsed YAML mapping.

        Raises:
            ServingConfigError: On unknown keys, missing required values or invalid values.
        """
        if not isinstance(data, Mapping):
            raise ServingConfigError("the serving config must be a mapping")
        unknown = sorted(set(data) - cls._TOP_LEVEL)
        if unknown:
            raise ServingConfigError(f"unknown top-level keys {unknown}; allowed: {sorted(cls._TOP_LEVEL)}")
        if "model" not in data:
            raise ServingConfigError("`model` section is required")
        esurge = data.get("esurge") or {}
        if not isinstance(esurge, Mapping):
            raise ServingConfigError("`esurge` must be a mapping of engine sections")
        if "distributed" in esurge:
            raise ServingConfigError("`esurge.distributed` is owned by the `coordination` section; remove it")
        return cls(
            model=_section(data, "model", ModelSpec),
            esurge=dict(esurge),
            server=_section(data, "server", ServerSpec),
            coordination=_section(data, "coordination", CoordinationSpec),
            compilation_cache_dir=data.get("compilation_cache_dir"),
        )

    @classmethod
    def load(cls, path: str | Path) -> ServingConfig:
        """Read and validate a serving YAML file."""
        with open(Path(path).expanduser(), encoding="utf-8") as handle:
            return cls.from_mapping(yaml.safe_load(handle) or {})
