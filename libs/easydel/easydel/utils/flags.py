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

"""Typed registry for EasyDeL-owned environment flags.

A single catalog of the environment variables EasyDeL reads, each with a type,
default, owning subsystem, and one-line doc. The typed getters
(:func:`get_bool` / :func:`get_int` / :func:`get_float` / :func:`get_str`) read
``os.environ`` **at call time**, so a flag set after import (e.g. by a test) is
honored and stays togglable per run.

EasyDeL-owned flags use the ``EASYDEL_`` prefix, except for the
``ESURGE_`` tokenizer-worker settings, which keep their names (the old
``EASURGE_`` names were removed with no alias). Flags owned by the foundation
libraries keep their own prefixes (``EJKERNEL_``,
``EFORMER_``) and are not registered here; ``EASYDEL_CACHE_COMPILES`` /
``EASYDEL_RECOMPILE_FORCE`` are read by ejkernel's ``callib`` by intentional
cross-package agreement.

:func:`render_markdown` emits a documentation table of every registered flag,
grouped by subsystem, for ``docs/environment_variables.md``.
"""

from __future__ import annotations

import os
import typing as tp
from dataclasses import dataclass

from .helpers import check_bool_flag

__all__ = [
    "FlagSpec",
    "get_bool",
    "get_float",
    "get_int",
    "get_str",
    "registered_flags",
    "render_markdown",
]


@dataclass(frozen=True)
class FlagSpec:
    """Definition of a single EasyDeL environment flag.

    Attributes:
        name: Existing environment variable name (usually ``EASYDEL_...``).
        kind: One of ``"bool"``, ``"int"``, ``"float"``, ``"str"``.
        default: Value used when the variable is unset (or invalid).
        subsystem: Human-readable owner group used for documentation.
        doc: One-line description of what the flag does.
        aliases: Existing legacy environment names, checked only when the
            canonical name is unset.
        true_values: Optional case-sensitive true values for flags with legacy
            parsing semantics; ``None`` uses ``check_bool_flag``.
    """

    name: str
    kind: str
    default: tp.Any
    subsystem: str
    doc: str
    aliases: tuple[str, ...] = ()
    true_values: tuple[str, ...] | None = None


_REGISTRY: dict[str, FlagSpec] = {}


def _register(
    name: str,
    kind: str,
    default: tp.Any,
    subsystem: str,
    doc: str,
    *,
    aliases: tuple[str, ...] = (),
    true_values: tuple[str, ...] | None = None,
) -> str:
    """Register a flag spec and return its name (for use as a module constant)."""
    if name in _REGISTRY:
        raise ValueError(f"flag {name!r} is already registered")
    _REGISTRY[name] = FlagSpec(
        name=name, kind=kind, default=default, subsystem=subsystem, doc=doc, aliases=aliases, true_values=true_values
    )
    return name


def registered_flags() -> tuple[FlagSpec, ...]:
    """Return all registered flag specs (sorted by subsystem then name)."""
    return tuple(sorted(_REGISTRY.values(), key=lambda s: (s.subsystem, s.name)))


def _spec(name: str) -> FlagSpec:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"{name!r} is not a registered EasyDeL flag (see easydel.utils.flags).") from exc


def _environment_name(name: str) -> str:
    """Select the canonical environment key, falling back to existing aliases."""
    for candidate in (name, *_spec(name).aliases):
        if candidate in os.environ:
            return candidate
    return name


def get_bool(name: str) -> bool:
    """Read a registered boolean flag from the environment at call time.

    Uses :func:`easydel.utils.helpers.check_bool_flag` truthiness
    (``true``/``yes``/``ok``/``1``/``easy``), unless the spec declares
    case-sensitive legacy ``true_values``. Falls back to the registered default
    when unset.
    """
    spec = _spec(name)
    environment_name = _environment_name(name)
    if spec.true_values is not None:
        raw = os.environ.get(environment_name)
        return bool(spec.default) if raw is None else raw in spec.true_values
    return check_bool_flag(environment_name, bool(spec.default))


def get_int(name: str) -> int:
    """Read a registered integer flag from the environment at call time."""
    default = int(_spec(name).default)
    raw = os.environ.get(_environment_name(name))
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def get_float(name: str) -> float:
    """Read a registered float flag from the environment at call time."""
    default = float(_spec(name).default)
    raw = os.environ.get(_environment_name(name))
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def get_str(name: str) -> str | None:
    """Read a registered string flag from the environment at call time.

    Returns the registered default (commonly ``None``) when unset or empty.
    """
    raw = os.environ.get(_environment_name(name))
    if raw is None or raw == "":
        return _spec(name).default
    return raw


def render_markdown() -> str:
    """Render the registered flags as a Markdown document grouped by subsystem."""
    lines = [
        "<!-- generated by easydel.utils.flags.render_markdown - do not edit by hand -->",
        "# EasyDeL-owned environment flags",
        "",
        "Typed, call-time flags registered in `easydel/utils/flags.py`. Booleans use "
        "`check_bool_flag` truthiness (`true`/`yes`/`ok`/`1`/`easy`) unless a flag documents legacy true values.",
        "",
    ]
    subsystem = None
    for spec in registered_flags():
        if spec.subsystem != subsystem:
            subsystem = spec.subsystem
            lines += ["", f"## {subsystem}", "", "| Flag | Type | Default | Description |", "| --- | --- | --- | --- |"]
        default = "none" if spec.default is None else f"`{spec.default}`"
        aliases = f" Legacy aliases: {', '.join(f'`{alias}`' for alias in spec.aliases)}." if spec.aliases else ""
        lines.append(f"| `{spec.name}` | {spec.kind} | {default} | {spec.doc}{aliases} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Catalog of EasyDeL-owned eSurge / speculative-decoding flags.
#
# The names below double as importable constants so call sites read
# ``flags.EASYDEL_SPEC_RECURRENT_REPLAY`` instead of a bare string literal.
# --------------------------------------------------------------------------- #

EASYDEL_PREFERRED_HOST_COPY_INDEX = _register(
    "EASYDEL_PREFERRED_HOST_COPY_INDEX",
    "int",
    0,
    "Checkpoint conversion",
    "Device index used for host-mediated tensor transfers.",
    aliases=("EASYDEL_PERFRED_HOST_COPY_INDEX",),
)
EASYDEL_PREFERRED_HOST_COPY = _register(
    "EASYDEL_PREFERRED_HOST_COPY",
    "str",
    "cpu",
    "Checkpoint conversion",
    "Backend for host-mediated tensor transfers; 'none' selects JAX's default backend.",
    aliases=("EASYDEL_PERFRED_HOST_COPY",),
)

EASYDEL_MAX_SCHEDULER_ERRORS = _register(
    "EASYDEL_MAX_SCHEDULER_ERRORS",
    "int",
    5,
    "eSurge engine",
    "Max consecutive scheduler errors before the eSurge engine loop stops.",
)
EASYDEL_HEARTBEAT_WARN_S = _register(
    "EASYDEL_HEARTBEAT_WARN_S",
    "float",
    120.0,
    "eSurge engine",
    "Seconds without a scheduler heartbeat before a stall warning is logged.",
)
EASYDEL_HEARTBEAT_WARN_INTERVAL_S = _register(
    "EASYDEL_HEARTBEAT_WARN_INTERVAL_S",
    "float",
    30.0,
    "eSurge engine",
    "Minimum seconds between repeated scheduler-stall warnings.",
)
EASYDEL_TOKENIZER_ENDPOINT = _register(
    "EASYDEL_TOKENIZER_ENDPOINT",
    "str",
    None,
    "eSurge engine",
    "Overrides the tokenizer worker ZeroMQ endpoint.",
)
EASYDEL_DETOKENIZER_ENDPOINT = _register(
    "EASYDEL_DETOKENIZER_ENDPOINT",
    "str",
    None,
    "eSurge engine",
    "Overrides the detokenizer worker ZeroMQ endpoint.",
)
EASYDEL_WORKER_STARTUP_TIMEOUT = _register(
    "EASYDEL_WORKER_STARTUP_TIMEOUT",
    "float",
    120.0,
    "eSurge engine",
    "Seconds to wait for a tokenizer/detokenizer worker to bind before timing out.",
)
ESURGE_WORKER_TRUST_REMOTE_CODE = _register(
    "ESURGE_WORKER_TRUST_REMOTE_CODE",
    "bool",
    True,
    "eSurge engine",
    "Default tokenizer trust_remote_code; legacy true values are exactly '1', 'on', 'yes' (case-sensitive).",
    true_values=("1", "on", "yes"),
)
ESURGE_DETOKENIZER_CONTEXT_WINDOW = _register(
    "ESURGE_DETOKENIZER_CONTEXT_WINDOW",
    "int",
    4,
    "eSurge engine",
    "Token context window used by the worker's incremental detokenizer.",
)

EASYDEL_SYNC_INPUTS_FOR_TIMING = _register(
    "EASYDEL_SYNC_INPUTS_FOR_TIMING",
    "bool",
    False,
    "eSurge runner",
    "Sync step inputs for accurate timing (adds a device round-trip).",
)
EASYDEL_ENABLE_DP_LOCAL_PAGE_PATH = _register(
    "EASYDEL_ENABLE_DP_LOCAL_PAGE_PATH",
    "bool",
    True,
    "eSurge runner",
    "Enable the data-parallel local-page attention path when the request count divides evenly.",
)

EASYDEL_SPEC_RECURRENT_REPLAY = _register(
    "EASYDEL_SPEC_RECURRENT_REPLAY",
    "bool",
    False,
    "eSurge spec-decode",
    "Bit-exact greedy under spec-decode via sequential-GDN replay of the accepted prefix (slower; opt-in).",
)
EASYDEL_DISABLE_BATCHED_DRAFT = _register(
    "EASYDEL_DISABLE_BATCHED_DRAFT",
    "bool",
    False,
    "eSurge spec-decode",
    "A/B fallback: force the per-request MTP draft path instead of the batched drafter.",
)
EASYDEL_DISABLE_BATCHED_EMIT = _register(
    "EASYDEL_DISABLE_BATCHED_EMIT",
    "bool",
    False,
    "eSurge spec-decode",
    "A/B fallback: force the per-request MTP emit path instead of the batched emit.",
)
EASYDEL_MTP_PERSIST_KV = _register(
    "EASYDEL_MTP_PERSIST_KV",
    "bool",
    False,
    "eSurge spec-decode",
    "Persist the MTP drafter KV cache across verify windows.",
)
EASYDEL_DISABLE_DYNAMIC_SPEC = _register(
    "EASYDEL_DISABLE_DYNAMIC_SPEC",
    "bool",
    False,
    "eSurge spec-decode",
    "A/B escape hatch: force the static num_draft_tokens (ignore num_draft_tokens_per_batch_size schedules).",
)
EASYDEL_DISABLE_DRAFT_IN_VERIFY = _register(
    "EASYDEL_DISABLE_DRAFT_IN_VERIFY",
    "bool",
    False,
    "eSurge spec-decode",
    "A/B escape hatch: force the two-dispatch verify->draft flow instead of the fused draft-in-verify program.",
)
