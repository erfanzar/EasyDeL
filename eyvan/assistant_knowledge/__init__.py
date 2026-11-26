# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""EasyDeL trainer knowledge for the fleet assistant.

The data is static: ``easydel_trainers.json`` is generated from the EasyDeL
trainer config sources by ``scripts/generate_easydel_catalog.py`` (pure ``ast``,
no EasyDeL/JAX import), and ``easydel_essentials.json`` is hand-curated. Both
ship as package data and are read with :mod:`importlib.resources`, so the
control plane never needs EasyDeL installed.

The four public functions map 1:1 onto assistant tools:

* :func:`list_trainers` -> ``list_easydel_trainers``
* :func:`describe_trainer` -> ``describe_easydel_trainer``
* :func:`validate_trainer_arguments` -> ``validate_easydel_arguments``
* :func:`render_launch` -> the ``easydel_trainer`` / ``trainer_arguments`` part of ``draft_workload``

Launches target the eLarge YAML runner shipped inside EasyDeL,
``python -m easydel.scripts.elarge --config FILE`` (``easydel/scripts/elarge.py``),
which builds an ``eLargeModel`` from the YAML and runs its ``actions``.
"""

from __future__ import annotations

import copy
import difflib
import functools
import json
import math
import re
import shlex
from collections.abc import Mapping, Sequence
from importlib import resources
from typing import Any

import yaml

__all__ = [
    "DEFAULT_MAX_CHARS",
    "DEFAULT_SHARDING",
    "ELARGE_MODULE",
    "TORCH_CPU_VERSION",
    "UnknownTrainerError",
    "catalog_info",
    "describe_trainer",
    "list_trainers",
    "render_launch",
    "requirement_name",
    "resolve_trainer_id",
    "torch_cpu_requirements",
    "validate_trainer_arguments",
]

CATALOG_RESOURCE = "easydel_trainers.json"
ESSENTIALS_RESOURCE = "easydel_essentials.json"
#: Upper bound on the serialized size of :func:`describe_trainer` output (an LLM tool result).
DEFAULT_MAX_CHARS = 12_000
_MIN_MAX_CHARS = 2_000
_MAX_MAX_CHARS = 60_000
_HELP_CHARS = 300
_TYPE_CHARS = 120
_MAX_CHOICES = 24
_MAX_CONSTRAINTS = 10
#: The eLarge YAML runner: ``easydel/scripts/elarge.py`` (``python -m easydel.scripts.elarge --config``).
ELARGE_MODULE = "easydel.scripts.elarge"
DEFAULT_CONFIG_PATH = "easydel-launch.yaml"
#: Mesh used by EasyDeL's own eLarge CLI examples (docs/scripts/elarge_cli.md): FSDP over every chip.
DEFAULT_SHARDING: dict[str, Any] = {
    "axis_dims": [1, -1, 1, 1, 1],
    "axis_names": ["dp", "fsdp", "ep", "tp", "sp"],
    "auto_shard_model": True,
}
_HEREDOC = "EASYDEL_ELARGE_CONFIG_EOF"
_AUX_ROLES = ("reference_model", "reward_model", "teacher_model", "target_model")

#: The CPU-only PyTorch build EasyDeL's Hugging Face loader imports
#: (``EasyBridgeMixin._from_torch_pretrained``). The default Linux wheel pulls
#: several GB of CUDA libraries a TPU host never uses, and a shared TPU disk can
#: hold a couple of GB free. One wheel per supported worker interpreter and
#: architecture; the PEP 508 markers let the worker's own installer pick the
#: one matching *its* Python, which the plane drafting the job cannot know.
TORCH_CPU_VERSION = "2.15.0"
_TORCH_CPU_PYTHONS = ("3.11", "3.12", "3.13")
_TORCH_CPU_MACHINES = ("x86_64", "aarch64")


def torch_cpu_requirements() -> list[str]:
    """Return the marker-selected CPU-only torch requirements, one per (Python, machine)."""

    requirements: list[str] = []
    for python in _TORCH_CPU_PYTHONS:
        tag = "cp" + python.replace(".", "")
        for machine in _TORCH_CPU_MACHINES:
            url = (
                "https://download.pytorch.org/whl/cpu/"
                f"torch-{TORCH_CPU_VERSION}%2Bcpu-{tag}-{tag}-manylinux_2_28_{machine}.whl"
            )
            requirements.append(
                f'torch @ {url} ; python_version == "{python}" and platform_machine == "{machine}" '
                'and sys_platform == "linux"'
            )
    return requirements


def requirement_name(requirement: str) -> str | None:
    """Return the normalized project name of a requirement string, or None when it has none."""

    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    return re.sub(r"[-_.]+", "-", match.group(1)).lower() if match else None


# eLarge ``build_trainer`` (easydel/infra/elarge/model.py) raises without these inputs.
# Keyed by eLarge trainer_type. Reward: GRPO family, online_dpo, sdpo, compaction_rl/sao,
# ppo, xpo/nash-md raise "requires `reward_model` (config key) or `reward_funcs`".
_REQUIRES_REWARD = frozenset(
    {
        "async_grpo",
        "compaction_rl",
        "dppo",
        "gfpo",
        "grpo",
        "grpo_with_replay_buffer",
        "gspo",
        "gspo_token",
        "nash-md",
        "online_dpo",
        "papo",
        "ppo",
        "rloo",
        "sao",
        "sdpo",
        "xpo",
    }
)
_REQUIRES_TARGET = frozenset({"speculative_decoding"})
#: Trainers whose required inputs cannot be expressed in YAML at all.
_YAML_UNSUPPORTED = {
    "agentic-moshpit": "eLarge requires a Python `env_factory` callable for agentic-moshpit; YAML cannot provide it.",
}
#: TRL / Hugging Face spellings users reach for, mapped to the EasyDeL field name.
_COMMON_RENAMES = {
    "batch_size": "total_batch_size",
    "per_device_train_batch_size": "total_batch_size",
    "train_batch_size": "total_batch_size",
    "lr": "learning_rate",
    "epochs": "num_train_epochs",
    "num_epochs": "num_train_epochs",
    "max_steps": "max_training_steps",
    "output_dir": "save_directory",
    "lr_scheduler_type": "scheduler",
    "lr_scheduler": "scheduler",
    "max_seq_length": "max_length",
    "max_seq_len": "max_length",
    "warmup_ratio": "warmup_steps",
    "logging_steps": "log_steps",
    "eval_steps": "evaluation_steps",
    "max_grad_norm": "clip_grad",
    "report_to": "use_wandb",
    "optim": "optimizer",
}


class UnknownTrainerError(ValueError):
    """The trainer id is not in the EasyDeL catalog."""

    def __init__(self, trainer_id: object, suggestions: Sequence[str]) -> None:
        self.trainer_id = trainer_id
        self.suggestions = list(suggestions)
        hint = f"; did you mean {', '.join(self.suggestions)}?" if self.suggestions else ""
        super().__init__(f"unknown EasyDeL trainer {trainer_id!r}{hint}")


# --------------------------------------------------------------------------- data


def _load(name: str) -> dict[str, Any]:
    data = json.loads(resources.files(__name__).joinpath(name).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"{name} must hold a JSON object")
    return data


@functools.cache
def _catalog() -> dict[str, Any]:
    return _load(CATALOG_RESOURCE)


@functools.cache
def _essentials() -> dict[str, Any]:
    return _load(ESSENTIALS_RESOURCE)


@functools.cache
def _alias_map() -> dict[str, str]:
    trainers: dict[str, Any] = _catalog()["trainers"]
    mapping: dict[str, str] = {}
    for alias, canonical in _catalog().get("elarge", {}).get("trainer_type_aliases", {}).items():
        if canonical in trainers:
            mapping[alias.lower()] = canonical
    for trainer_id, info in trainers.items():
        for alias in info.get("aliases", []):
            mapping[alias.lower()] = trainer_id
    return mapping


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False))


def resolve_trainer_id(trainer_id: str) -> str:
    """Return the canonical catalog id for ``trainer_id`` (case, ``-``/``_`` and alias tolerant).

    Raises:
        UnknownTrainerError: No trainer matches; ``suggestions`` holds close ids.
    """

    trainers: dict[str, Any] = _catalog()["trainers"]
    if not isinstance(trainer_id, str) or not trainer_id.strip():
        raise UnknownTrainerError(trainer_id, [])
    raw = trainer_id.strip()
    key = raw.lower()
    aliases = _alias_map()
    for candidate in (raw, key, key.replace("-", "_"), key.replace("_", "-")):
        if candidate in trainers:
            return candidate
        if candidate in aliases:
            return aliases[candidate]
    pool = sorted(set(trainers) | set(aliases))
    suggestions: list[str] = []
    for match in difflib.get_close_matches(key, pool, n=4, cutoff=0.5):
        resolved = match if match in trainers else aliases[match]
        if resolved not in suggestions:
            suggestions.append(resolved)
    raise UnknownTrainerError(trainer_id, suggestions[:3])


def _resolved_fields(trainer_id: str) -> dict[str, tuple[dict[str, Any], str]]:
    """``name -> (field entry, declaring class)``; derived declarations override base ones."""

    catalog = _catalog()
    resolved: dict[str, tuple[dict[str, Any], str]] = {}
    for cls in reversed(catalog["trainers"][trainer_id]["mro"]):
        for name, entry in catalog["classes"][cls]["fields"].items():
            resolved[name] = (entry, cls)
    return resolved


def _elarge_defaults(trainer_id: str) -> dict[str, Any]:
    elarge = _catalog().get("elarge", {})
    etype = _catalog()["trainers"][trainer_id]["elarge_trainer_type"]
    defaults = dict(elarge.get("base_defaults", {}))
    defaults.update(elarge.get("trainer_defaults", {}).get(etype, {}))
    return defaults


def _essential_spec(trainer_id: str) -> dict[str, Any]:
    """Merge the curated common and trainer-specific essentials (fields, one_of, launch inputs)."""

    essentials = _essentials()
    common = essentials.get("common", {})
    specific = essentials.get("trainers", {}).get(trainer_id, {})
    fields = dict(common.get("fields", {}))
    fields.update(specific.get("fields", {}))
    names = _resolved_fields(trainer_id)
    fields = {name: why for name, why in fields.items() if name in names}
    one_of = [copy.deepcopy(group) for group in [*common.get("one_of", []), *specific.get("one_of", [])]]
    return {"fields": fields, "one_of": one_of, "launch": dict(common.get("launch", {}))}


def _launch_inputs(trainer_id: str) -> dict[str, dict[str, Any]]:
    """Auxiliary models (``<role>_model`` blocks) the trainer uses, with whether eLarge needs them."""

    etype = _catalog()["trainers"][trainer_id]["elarge_trainer_type"]
    curated = _essentials().get("trainers", {}).get(trainer_id, {}).get("launch_inputs", {})
    inputs = {role: {"required": bool(spec["required"]), "why": spec["why"]} for role, spec in curated.items()}
    if etype in _REQUIRES_REWARD:
        inputs.setdefault("reward_model", {"why": "eLarge build_trainer raises without a reward model."})
        inputs["reward_model"]["required"] = True
    if etype in _REQUIRES_TARGET:
        inputs.setdefault("target_model", {"why": "eLarge build_trainer raises without a target model."})
        inputs["target_model"]["required"] = True
    return inputs


def catalog_info() -> dict[str, Any]:
    """Catalog provenance plus the essentials every trainer shares (training length, model, data, mesh)."""

    catalog = _catalog()
    common = _essentials().get("common", {})
    return {
        "easydel_revision": catalog.get("easydel_revision"),
        "generated_by": catalog.get("generated_by"),
        "trainer_count": len(catalog["trainers"]),
        "launcher": f"python -m {ELARGE_MODULE} --config <file.yaml>",
        "common_essential": {
            "fields": dict(common.get("fields", {})),
            "one_of": copy.deepcopy(common.get("one_of", [])),
            "launch": dict(common.get("launch", {})),
        },
    }


# --------------------------------------------------------------------------- list / describe


def list_trainers() -> list[dict[str, Any]]:
    """Every EasyDeL trainer: id, config class, one-line summary and its trainer-specific essentials.

    Essentials shared by all trainers (learning rate, batch size, length, checkpointing,
    model, dataset, sharding) are in :func:`catalog_info` and in every
    :func:`describe_trainer` result, so they are not repeated per entry here.
    """

    catalog = _catalog()
    curated = _essentials().get("trainers", {})
    out: list[dict[str, Any]] = []
    for trainer_id in sorted(catalog["trainers"]):
        info = catalog["trainers"][trainer_id]
        entry: dict[str, Any] = {
            "id": trainer_id,
            "class": info["class"],
            "summary": info.get("summary") or "",
            "essential": sorted(curated.get(trainer_id, {}).get("fields", {})),
        }
        if info.get("aliases"):
            entry["aliases"] = list(info["aliases"])
        if info["elarge_trainer_type"] != trainer_id:
            entry["elarge_trainer_type"] = info["elarge_trainer_type"]
        required = sorted(role for role, spec in _launch_inputs(trainer_id).items() if spec["required"])
        if required:
            entry["requires"] = required
        if info["elarge_trainer_type"] in _YAML_UNSUPPORTED:
            entry["yaml_launch"] = False
        out.append(entry)
    return out


def _compact_field(
    name: str,
    entry: dict[str, Any],
    owner: str,
    own_class: str,
    why: str | None,
    elarge_defaults: Mapping[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {"name": name, "type": _clip(entry["type"], _TYPE_CHARS)}
    if "default" in entry:
        default = entry["default"]
        if _size(default) > 200:
            out["default_expr"] = _clip(json.dumps(default, ensure_ascii=False), 120)
        else:
            out["default"] = copy.deepcopy(default)
    elif "default_expr" in entry:
        out["default_expr"] = _clip(entry["default_expr"], 120)
    elif "default_factory" in entry:
        out["default_factory"] = _clip(entry["default_factory"], 120)
    if name in elarge_defaults and ("default" not in entry or elarge_defaults[name] != entry["default"]):
        value = elarge_defaults[name]
        if _size(value) <= 200:
            out["elarge_default"] = copy.deepcopy(value)
    if entry.get("required"):
        out["required"] = True
    if entry.get("init_only"):
        out["init_only"] = True
    help_text = entry.get("help") or entry.get("doc")
    if help_text:
        out["help"] = _clip(help_text, _HELP_CHARS)
    choices = entry.get("choices")
    if choices:
        out["choices"] = copy.deepcopy(choices[:_MAX_CHOICES])
        if len(choices) > _MAX_CHOICES:
            out["choices_omitted"] = len(choices) - _MAX_CHOICES
        if entry.get("choices_allow_list"):
            out["choices_allow_list"] = True
    if owner != own_class:
        out["inherited_from"] = owner
    if why:
        out["essential"] = why
    return out


def _matches(terms: Sequence[str], name: str, entry: Mapping[str, Any]) -> bool:
    haystack = " ".join(
        [name, name.replace("_", " "), str(entry.get("help") or ""), str(entry.get("doc") or "")]
    ).lower()
    # Any term: an LLM asks for several fields at once ("beta loss_type"), and
    # requiring every term in one field returned nothing for that.
    return any(term in haystack for term in terms)


def describe_trainer(
    trainer_id: str,
    *,
    include_inherited: bool = True,
    query: str | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> dict[str, Any]:
    """Describe a trainer's arguments compactly enough for an LLM tool result.

    Args:
        trainer_id: Catalog id or alias (``"dpo"``, ``"DPO"``, ``"nash_md"``, ``"spec"``).
        include_inherited: Include fields declared by base classes (``TrainingArguments`` etc.).
        query: Whitespace-separated terms; a field is kept when any term occurs in its
            name (``_`` read as a space) or help/doc text. Case-insensitive.
        max_chars: Bound on the JSON size of the result (clamped to 2,000..60,000).

    Returns:
        Header (trainer, class, module, summary, mro, revision, essentials, launch inputs)
        plus ``fields``: essentials first, then the trainer's own fields, then inherited ones.
        Fields that do not fit in detail are listed by name in ``more_fields``; ``truncated``
        and ``note`` say so and how to narrow with ``query``.

    Raises:
        UnknownTrainerError: ``trainer_id`` is not in the catalog.
    """

    budget = max(_MIN_MAX_CHARS, min(int(max_chars), _MAX_MAX_CHARS))
    tid = resolve_trainer_id(trainer_id)
    catalog = _catalog()
    info = catalog["trainers"][tid]
    own_class = info["class"]
    spec = _essential_spec(tid)
    essential_fields: dict[str, str] = spec["fields"]
    elarge_defaults = _elarge_defaults(tid)
    terms = query.lower().split() if query else []

    fields = _resolved_fields(tid)
    selected = [
        (name, entry, owner)
        for name, (entry, owner) in fields.items()
        if (include_inherited or owner == own_class) and (not terms or _matches(terms, name, entry))
    ]
    selected.sort(key=lambda item: (item[0] not in essential_fields, item[2] != own_class, item[0]))

    result: dict[str, Any] = {
        "trainer": tid,
        "class": own_class,
        "module": info["module"],
        "summary": info.get("summary") or "",
        "mro": list(info["mro"]),
        "elarge_trainer_type": info["elarge_trainer_type"],
        "easydel_revision": catalog.get("easydel_revision"),
        "essential": [{"field": name, "why": why} for name, why in essential_fields.items()],
        "essential_one_of": spec["one_of"],
        "launch_inputs": {**{key: {"required": True, "why": why} for key, why in spec["launch"].items()}},
        "total_fields": len(fields),
        "matched_fields": len(selected),
        "fields": [],
    }
    for role, role_spec in _launch_inputs(tid).items():
        result["launch_inputs"][role] = dict(role_spec)
    if query:
        result["query"] = query
    if info["elarge_trainer_type"] in _YAML_UNSUPPORTED:
        result["yaml_launch"] = _YAML_UNSUPPORTED[info["elarge_trainer_type"]]
    if any("elarge_default" in _compact_field(n, e, o, own_class, None, elarge_defaults) for n, e, o in selected):
        result["defaults_note"] = (
            "`elarge_default` is the value the eLarge YAML runner applies when the key is omitted; "
            "it overrides the dataclass `default`."
        )

    reserve = 600  # note + truncation flags
    size = _size(result) + reserve
    remaining = list(selected)
    detailed_budget = budget - min(3_000, 24 * len(selected)) if len(selected) > 12 else budget
    while remaining:
        name, entry, owner = remaining[0]
        compact = _compact_field(name, entry, owner, own_class, essential_fields.get(name), elarge_defaults)
        cost = _size(compact) + 2
        if size + cost > detailed_budget and result["fields"]:
            break
        result["fields"].append(compact)
        size += cost
        remaining.pop(0)
    more: list[str] = []
    omitted = 0
    for name, _entry, _owner in remaining:
        cost = len(name) + 4
        if size + cost <= budget:
            more.append(name)
            size += cost
        else:
            omitted += 1
    if more:
        result["more_fields"] = more
    result["shown_fields"] = len(result["fields"])
    result["truncated"] = bool(remaining)
    if remaining:
        result["note"] = (
            f"{len(result['fields'])} of {len(selected)} matching fields shown in detail; "
            f"{len(more)} more listed by name in more_fields"
            + (f" and {omitted} omitted" if omitted else "")
            + ". Call again with query=<field name or topic> for their docs and defaults."
        )
    while _size(result) > budget and result.get("more_fields"):
        result["more_fields"].pop()
    return result


# --------------------------------------------------------------------------- validate


def _kind(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list | tuple):
        return "list"
    if isinstance(value, Mapping):
        return "dict"
    return type(value).__name__


def _has_nonfinite(value: Any) -> bool:
    """Whether a value is, or contains, a NaN or infinite float."""

    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, list | tuple):
        return any(_has_nonfinite(item) for item in value)
    if isinstance(value, Mapping):
        return any(_has_nonfinite(item) for item in value.values())
    return False


def _choices_for(trainer_id: str, name: str, entry: Mapping[str, Any]) -> tuple[list[Any], bool] | None:
    if entry.get("choices"):
        return list(entry["choices"]), bool(entry.get("choices_allow_list"))
    classes = _catalog()["classes"]
    for cls in _catalog()["trainers"][trainer_id]["mro"]:
        validated = classes[cls].get("validated_choices", {})
        if name in validated:
            return list(validated[name]), "list" in entry.get("accepts", [])
    return None


def _suggest(name: str, names: Sequence[str]) -> str | None:
    renamed = _COMMON_RENAMES.get(name.lower())
    if renamed in names:
        return renamed
    lowered = {candidate.lower(): candidate for candidate in names}
    if name.lower() in lowered:
        return lowered[name.lower()]
    match = difflib.get_close_matches(name, list(names), n=1, cutoff=0.6)
    if match:
        return match[0]
    containing = [candidate for candidate in names if name.lower() in candidate.lower()]
    return min(containing, key=len) if len(name) >= 4 and containing else None


def validate_trainer_arguments(trainer_id: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Check proposed trainer arguments against the catalog before drafting a job.

    Returns:
        ``ok`` (True only when there are no unknown names, invalid choices, type warnings
        or missing essentials), ``unknown`` names with ``suggestions`` (closest real field),
        ``missing_essential`` (``{"field", "why"}`` or ``{"any_of", "why"}`` the assistant
        must ask about), ``type_warnings``, ``invalid_choices``, plus ``constraints``
        (``__post_init__`` checks touching the given fields), ``launch_inputs_to_confirm``
        and ``notes``.

    Raises:
        UnknownTrainerError: ``trainer_id`` is not in the catalog.
        TypeError: ``arguments`` is not a mapping.
    """

    tid = resolve_trainer_id(trainer_id)
    if not isinstance(arguments, Mapping):
        raise TypeError("arguments must be a mapping of EasyDeL field name to value")
    fields = _resolved_fields(tid)
    names = sorted(fields)
    unknown: list[str] = []
    suggestions: dict[str, str] = {}
    type_warnings: list[dict[str, Any]] = []
    invalid_choices: list[dict[str, Any]] = []
    notes: list[str] = []
    info = _catalog()["trainers"][tid]

    for key in sorted(str(k) for k in arguments):
        value = arguments[key]
        if key == "trainer_type":
            notes.append("trainer_type is set from the trainer id; drop it from the arguments.")
            continue
        if key not in fields:
            unknown.append(key)
            suggestion = _suggest(key, names)
            if suggestion:
                suggestions[key] = suggestion
            continue
        entry = fields[key][0]
        kind = _kind(value)
        if _has_nonfinite(value):
            # YAML writes these as .nan/.inf and EasyDeL accepts them, so a NaN
            # learning rate would otherwise become a "runnable" job that trains garbage.
            type_warnings.append(
                {"field": key, "value_type": "non-finite number", "expected": entry.get("type"), "accepts": ["finite"]}
            )
            continue
        accepts = entry.get("accepts")
        if accepts is not None and kind not in accepts:
            type_warnings.append({"field": key, "value_type": kind, "expected": entry["type"], "accepts": accepts})
            continue
        choice_spec = _choices_for(tid, key, entry)
        if choice_spec is None:
            continue
        choices, allow_list = choice_spec
        choice_kinds = {_kind(choice) for choice in choices}
        values: list[Any]
        if kind == "list" and allow_list:
            values = list(value)
        elif kind in choice_kinds:
            values = [value]
        else:
            values = []
        bad = [item for item in values if _kind(item) in choice_kinds and item not in choices]
        if bad:
            record: dict[str, Any] = {"field": key, "value": copy.deepcopy(value), "allowed": list(choices)}
            close = [
                match
                for item in bad
                if isinstance(item, str)
                for match in difflib.get_close_matches(item, [str(c) for c in choices], n=1, cutoff=0.5)
            ]
            if close:
                record["did_you_mean"] = close
            invalid_choices.append(record)

    spec = _essential_spec(tid)
    missing: list[dict[str, Any]] = [
        {"field": name, "why": why} for name, why in spec["fields"].items() if name not in arguments
    ]
    for group in spec["one_of"]:
        group_fields = [name for name in group.get("fields", []) if name in fields]
        if group_fields and not any(name in arguments for name in group_fields):
            missing.append({"any_of": group_fields, "why": group.get("why", "")})

    constraints: list[dict[str, Any]] = []
    classes = _catalog()["classes"]
    given = set(arguments)
    for cls in info["mro"]:
        for check in classes[cls].get("checks", []):
            if given.intersection(check["fields"]) and len(constraints) < _MAX_CONSTRAINTS:
                constraints.append({"if": check["if"], "message": check.get("message"), "declared_in": cls})

    launch = [{"input": key, "required": True, "why": why} for key, why in spec["launch"].items()]
    launch += [{"input": role, **role_spec} for role, role_spec in _launch_inputs(tid).items()]
    if unknown:
        notes.append(
            "The eLarge runner silently drops trainer keys that are not config fields, so unknown names "
            "would never raise at launch: fix or remove them."
        )
    if info["elarge_trainer_type"] in _YAML_UNSUPPORTED:
        notes.append(_YAML_UNSUPPORTED[info["elarge_trainer_type"]])

    return {
        "ok": not (unknown or type_warnings or invalid_choices or missing),
        "trainer": tid,
        "unknown": unknown,
        "suggestions": suggestions,
        "missing_essential": missing,
        "type_warnings": type_warnings,
        "invalid_choices": invalid_choices,
        "constraints": constraints,
        "launch_inputs_to_confirm": launch,
        "notes": notes,
    }


# --------------------------------------------------------------------------- render


_FILE_TYPES = (
    (".jsonl", "json"),
    (".json", "json"),
    (".parquet", "parquet"),
    (".csv", "csv"),
    (".tsv", "tsv"),
    (".arrow", "arrow"),
    (".txt", "text"),
)


def _dataset_inform(
    dataset: str | Mapping[str, Any],
    *,
    split: str,
    dataset_type: str | None,
    subset: str | None,
    content_field: str | None,
    assumptions: list[str],
) -> dict[str, Any]:
    if isinstance(dataset, Mapping):
        return dict(dataset)
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("dataset must be a dataset id, a file path/glob, or an eLarge mixture inform mapping")
    location = dataset.strip()
    source_type = dataset_type
    if source_type is None:
        lowered = location.lower().rstrip("*/")
        source_type = next((kind for suffix, kind in _FILE_TYPES if lowered.endswith(suffix)), None)
        if source_type is None and ("*" in location or "://" in location or location.startswith(("/", ".", "~"))):
            assumptions.append(f"dataset {location!r} has no recognised file suffix; set dataset_type explicitly.")
        elif source_type is None:
            source_type = "huggingface"
            assumptions.append(f"dataset {location!r} treated as a Hugging Face Hub dataset id.")
    inform: dict[str, Any] = {}
    if source_type is not None:
        inform["type"] = source_type
    inform["data_files"] = location
    if source_type in ("huggingface", "hf"):
        inform["split"] = split
        if subset:
            inform["dataset_split_name"] = subset
    if content_field:
        inform["content_field"] = content_field
    return inform


def _model_block(value: str | Mapping[str, Any], *, role: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        block = dict(value)
        if not block.get("name_or_path"):
            raise ValueError(f"{role} needs name_or_path")
        return block
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{role} must be a model id/path or a model mapping")
    return {"name_or_path": value.strip()}


def _sharding_problems(sharding: Mapping[str, Any], num_devices: int | None) -> list[str]:
    dims = sharding.get("axis_dims")
    axis_names = sharding.get("axis_names")
    if not isinstance(dims, list | tuple) or not all(isinstance(d, int) and not isinstance(d, bool) for d in dims):
        return ["sharding.axis_dims must be a list of integers"]
    problems: list[str] = []
    if isinstance(axis_names, list | tuple) and len(axis_names) != len(dims):
        problems.append("sharding.axis_dims and axis_names must have the same length")
    if sum(1 for d in dims if d == -1) > 1:
        problems.append("sharding.axis_dims may contain at most one -1")
    if any(d == 0 or d < -1 for d in dims):
        problems.append("sharding.axis_dims entries must be positive or -1")
    if num_devices is not None and not problems:
        fixed = 1
        for d in dims:
            if d != -1:
                fixed *= d
        if -1 in dims and num_devices % fixed:
            problems.append(f"{num_devices} devices are not divisible by the fixed mesh product {fixed}")
        elif -1 not in dims and fixed != num_devices:
            problems.append(f"mesh product {fixed} does not equal the {num_devices} devices")
    return problems


def render_launch(
    trainer_id: str,
    arguments: Mapping[str, Any],
    *,
    model: str | Mapping[str, Any],
    dataset: str | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    dataset_split: str = "train",
    dataset_type: str | None = None,
    dataset_subset: str | None = None,
    content_field: str | None = None,
    tokenizer: str | None = None,
    reference_model: str | Mapping[str, Any] | None = None,
    reward_model: str | Mapping[str, Any] | None = None,
    teacher_model: str | Mapping[str, Any] | None = None,
    target_model: str | Mapping[str, Any] | None = None,
    sharding: Mapping[str, Any] | None = None,
    num_devices: int | None = None,
    dtype: str = "bfloat16",
    config_path: str = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Render a runnable eLarge launch: YAML config, command line and a self-contained shell script.

    The command is ``python -m easydel.scripts.elarge --config <config_path>``; the YAML holds
    ``model``, auxiliary ``<role>_model`` blocks, ``loader``, ``sharding``, ``trainer``
    (``trainer_type`` + arguments), ``mixture`` and ``actions: [validate, train]``.

    Returns:
        ``runnable`` (False when the arguments fail validation errors, a required auxiliary
        model is missing, the mesh is inconsistent or the trainer cannot launch from YAML),
        ``blocking`` reasons, ``command``, ``argv``, ``config_path``, ``config_yaml``,
        ``config`` (the same data as a dict), ``script`` (writes the YAML then runs the
        command), ``dry_run_command``, ``validation`` (see :func:`validate_trainer_arguments`)
        and ``assumptions``.

    Raises:
        UnknownTrainerError: ``trainer_id`` is not in the catalog.
        ValueError: ``model``/``dataset`` are malformed or arguments are not JSON/YAML data.
    """

    tid = resolve_trainer_id(trainer_id)
    info = _catalog()["trainers"][tid]
    etype = info["elarge_trainer_type"]
    try:
        trainer_args = json.loads(json.dumps(dict(arguments)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"trainer arguments must be JSON-compatible data: {exc}") from exc
    validation = validate_trainer_arguments(tid, trainer_args)
    assumptions: list[str] = [
        f"Entry point: python -m {ELARGE_MODULE} (EasyDeL easydel/scripts/elarge.py, YAML runner for eLargeModel).",
        "EasyDeL, its dependencies and PyYAML must be installed on the worker.",
    ]
    blocking: list[str] = []

    model_block = _model_block(model, role="model")
    model_block.setdefault("tokenizer", tokenizer or model_block["name_or_path"])
    if etype == "reward" and "task" not in model_block:
        model_block["task"] = "sequence-classification"
        model_block.setdefault("extra_kwargs", {"num_labels": 1})
        assumptions.append("reward trainer: model loaded as sequence-classification with num_labels=1.")

    config: dict[str, Any] = {"model": model_block}
    aux_values = {
        "reference_model": reference_model,
        "reward_model": reward_model,
        "teacher_model": teacher_model,
        "target_model": target_model,
    }
    launch_inputs = _launch_inputs(tid)
    for role in _AUX_ROLES:
        value = aux_values[role]
        if value is None:
            spec = launch_inputs.get(role)
            if spec and spec.get("required"):
                blocking.append(f"{role} is required for {tid}: {spec['why']}")
            continue
        block = _model_block(value, role=role)
        if role == "reward_model" and not isinstance(value, Mapping):
            block.update({"task": "sequence-classification", "extra_kwargs": {"num_labels": 1}})
            assumptions.append("reward_model loaded as a sequence-classification head with num_labels=1.")
        if role not in launch_inputs:
            assumptions.append(f"{role} given, but {tid} is not known to use it; eLarge may ignore it.")
        config[role] = block
    if "reference_model" in launch_inputs and reference_model is None:
        assumptions.append("No reference_model: the trainer uses a frozen copy of the starting model.")

    config["loader"] = {"dtype": dtype, "param_dtype": dtype}
    if sharding is None:
        mesh: dict[str, Any] = copy.deepcopy(DEFAULT_SHARDING)
        assumptions.append("sharding not given: default mesh dp=1, fsdp=-1 (all chips), ep=tp=sp=1.")
    else:
        mesh = json.loads(json.dumps(dict(sharding)))
        mesh.setdefault("axis_names", list(DEFAULT_SHARDING["axis_names"]))
        mesh.setdefault("auto_shard_model", True)
    blocking.extend(_sharding_problems(mesh, num_devices))
    config["sharding"] = mesh

    config["trainer"] = {"trainer_type": etype, **{k: v for k, v in trainer_args.items() if k != "trainer_type"}}
    if isinstance(dataset, Mapping) or isinstance(dataset, str):
        informs = [
            _dataset_inform(
                dataset,
                split=dataset_split,
                dataset_type=dataset_type,
                subset=dataset_subset,
                content_field=content_field,
                assumptions=assumptions,
            )
        ]
    elif isinstance(dataset, Sequence) and dataset and all(isinstance(item, Mapping) for item in dataset):
        informs = [dict(item) for item in dataset if isinstance(item, Mapping)]
    else:
        raise ValueError("dataset must be a dataset id, a file path/glob, or eLarge mixture inform mapping(s)")
    config["mixture"] = {"informs": json.loads(json.dumps(informs))}
    config["actions"] = ["validate", "train"]

    if etype in _YAML_UNSUPPORTED:
        blocking.append(_YAML_UNSUPPORTED[etype])
    if validation["unknown"]:
        blocking.append(f"unknown trainer arguments (eLarge would silently drop them): {validation['unknown']}")
    if validation["invalid_choices"]:
        blocking.append(f"invalid choices for {[item['field'] for item in validation['invalid_choices']]}")
    if validation["type_warnings"]:
        blocking.append(f"wrong value types for {[item['field'] for item in validation['type_warnings']]}")
    if "save_directory" not in trainer_args:
        assumptions.append("save_directory not set: checkpoints go to the relative EasyDeL-Checkpoints directory.")
    # eLarge loads Hugging Face checkpoints through EasyDeL's torch bridge; a
    # job without torch dies before its first step ("install torch first").
    dependencies = torch_cpu_requirements()
    assumptions.append(
        f"Hugging Face checkpoints load through torch: the draft declares the CPU-only torch {TORCH_CPU_VERSION} "
        "wheel for the worker's Python (no CUDA libraries); a torch requirement you list yourself replaces it."
    )

    config_yaml = yaml.safe_dump(config, sort_keys=False, allow_unicode=True, default_flow_style=False)
    if _HEREDOC in config_yaml:
        raise ValueError("rendered configuration contains the heredoc delimiter")
    argv = ["python", "-m", ELARGE_MODULE, "--config", config_path]
    command = shlex.join(argv)
    script = f"cat > {shlex.quote(config_path)} <<'{_HEREDOC}'\n{config_yaml}{_HEREDOC}\n{command}\n"
    return {
        "trainer": tid,
        "elarge_trainer_type": etype,
        "runnable": not blocking,
        "blocking": blocking,
        "entry_point": ELARGE_MODULE,
        "command": command,
        "argv": argv,
        "dry_run_command": f"{command} --dry-run",
        "config_path": config_path,
        "config_yaml": config_yaml,
        "config": config,
        "script": script,
        "validation": validation,
        "assumptions": assumptions,
        "dependencies": dependencies,
    }
