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
"""NeMo Gym style GRPO trainer backed by EasyDeL/eSurge generation."""

from __future__ import annotations

import io
import json
import typing as tp
from pathlib import Path

import numpy as np
from eformer.paths import UniversalPath, ePath

from easydel.data.transforms.base import Transform


def _environment_reward_func(
    completions: tp.Sequence[object],
    *,
    environment_rewards: tp.Sequence[object] | None = None,
    **_: object,
) -> list[float]:
    """Default reward bridge for NeMo Gym environments.

    Environment feedback is passed through ``environment_rewards`` after the
    trainer steps each task environment. When no environment rewards are
    available, the function returns zeros so GRPO can still construct a valid
    reward tensor.
    """

    if environment_rewards is None:
        return [0.0 for _ in completions]
    return [0.0 if reward is None else float(reward) for reward in environment_rewards]


def _coerce_reward(reward: object) -> float:
    """Coerce an optional environment reward to float, treating ``None`` as ``0.0``."""
    if reward is None:
        return 0.0
    return float(reward)


def _decode_json_if_needed(value: object) -> object:
    """Decode byte/string JSON payloads while preserving non-JSON values.

    NeMo task metadata can come from JSONL files, datasets, or environment
    wrappers. NumPy scalars are unwrapped, byte values are decoded first, and
    invalid JSON strings are returned unchanged instead of failing.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes | bytearray):
        value = value.decode()
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return value
    return value


def _decode_json_container_if_needed(value: object) -> object:
    """Decode JSON object/array strings (used for serialized ``agent_ref`` payloads).

    Plain non-container strings such as ``"agent-a"`` or ``"42"`` are preserved
    as strings rather than coerced into numbers.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes | bytearray):
        value = value.decode()
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                parsed = json.loads(stripped)
            except (TypeError, ValueError):
                return value
            if isinstance(parsed, dict | list):
                return parsed
    return value


def _as_mapping(value: object) -> dict[str, object]:
    """Normalize dict, pydantic-like, JSON, or scalar payloads to a mapping.

    Dict inputs are copied, ``None`` yields an empty dict, objects with
    ``model_dump`` use that structured representation, and scalar fallback
    values are wrapped under ``"value"``.
    """
    if value is None:
        return {}
    value = _decode_json_if_needed(value)
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dict(dumped) if isinstance(dumped, dict) else {"value": dumped}
    return {"value": value}


def _get_from_mapping_or_object(value: object, key: str, default: object = None) -> object:
    """Read a field from a mapping or object with a fallback default.

    Different environment implementations return dictionaries or dataclass-like
    result objects. This helper keeps result normalization explicit while still
    preserving the requested default for missing optional fields.
    """
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _normalize_step_result(step_result: object) -> tuple[object, float, bool, bool, dict[str, object]]:
    """Normalize Gymnasium, NeMo, dict, and dataclass-like step payloads.

    The normalized shape is always ``(observation, reward, terminated,
    truncated, info)``. The accepted inputs cover Gymnasium five-tuples, classic
    Gym four-tuples, dictionaries, and objects exposing matching attributes.
    Unexpected tuple lengths raise ``ValueError`` rather than silently yielding
    ``0.0`` reward, and ``reward=None`` safely defaults to ``0.0``.
    """

    if isinstance(step_result, tuple):
        if len(step_result) == 5:
            observation, reward, terminated, truncated, info = step_result
            return observation, _coerce_reward(reward), bool(terminated), bool(truncated), _as_mapping(info)
        if len(step_result) == 4:
            observation, reward, done, info = step_result
            return observation, _coerce_reward(reward), bool(done), False, _as_mapping(info)
        raise ValueError(
            f"Expected a 4-tuple or 5-tuple environment step result, got tuple of length {len(step_result)}."
        )

    if isinstance(step_result, dict):
        reward = step_result.get("reward")
        if reward is None:
            reward = step_result.get("env_reward", 0.0)
        observation = step_result.get("observation", step_result.get("response", ""))
        terminated = step_result.get("terminated", step_result.get("done", False))
        truncated = step_result.get("truncated", False)
        info = step_result.get("info")
        if info is None:
            info = {}
        elif not isinstance(info, dict):
            info = {"value": info}
        if "num_turns" in step_result:
            info = {**info, "num_turns": step_result["num_turns"]}
        return observation, _coerce_reward(reward), bool(terminated), bool(truncated), info

    observation = _get_from_mapping_or_object(step_result, "observation", "")
    reward = _get_from_mapping_or_object(step_result, "reward", 0.0)
    terminated = _get_from_mapping_or_object(step_result, "terminated", False)
    truncated = _get_from_mapping_or_object(step_result, "truncated", False)
    info = _get_from_mapping_or_object(step_result, "info", {})
    return observation, _coerce_reward(reward), bool(terminated), bool(truncated), _as_mapping(info)


class _NeMoGymPreprocessTransform(Transform):
    """Wrap GRPO preprocessing while unconditionally preserving NeMo sidechannels.

    ``GRPOPreprocessTransform`` filters auxiliary columns via ``is_reward_sideband``,
    which drops numeric-only metadata dicts (such as ``{"reward": 0.1}``). This
    wrapper retains ``metadata_key`` and ``agent_ref_key`` across preprocessing.
    """

    def __init__(
        self,
        base_transform: Transform,
        *,
        metadata_key: str,
        agent_ref_key: str,
    ) -> None:
        self._base_transform = base_transform
        self._metadata_key = metadata_key
        self._agent_ref_key = agent_ref_key

    def __call__(self, example: dict[str, tp.Any]) -> dict[str, tp.Any]:
        result = self._base_transform(example)
        if not isinstance(result, dict):
            return result
        for key in (self._metadata_key, self._agent_ref_key):
            if key in example and key not in result and example[key] is not None:
                result[key] = example[key]
        return result

    def __repr__(self) -> str:
        return (
            f"_NeMoGymPreprocessTransform(base={self._base_transform!r}, "
            f"metadata_key={self._metadata_key!r}, agent_ref_key={self._agent_ref_key!r})"
        )


class _NeMoGymGrainCollator:
    """Wrap GRPO Grain collator so NeMo sidechannels bypass numeric array coercion."""

    def __init__(
        self,
        base_collator: tp.Callable[[dict[str, tp.Any]], dict[str, tp.Any]],
        *,
        metadata_key: str,
        agent_ref_key: str,
    ) -> None:
        self._base_collator = base_collator
        self._sidechannel_keys = (metadata_key, agent_ref_key)

    def __call__(self, feature: dict[str, tp.Any]) -> dict[str, tp.Any]:
        saved = {key: feature[key] for key in self._sidechannel_keys if key in feature and feature[key] is not None}
        if not saved:
            return self._base_collator(feature)
        stripped = {key: value for key, value in feature.items() if key not in saved}
        batch = self._base_collator(stripped)
        for key, value in saved.items():
            batch[key] = value
        return batch


class _NeMoGymTFDSCollator:
    """Wrap GRPO TFDS collator so NeMo sidechannels remain Python lists."""

    def __init__(
        self,
        base_collator: tp.Callable[[list[dict[str, tp.Any]]], dict[str, tp.Any]],
        *,
        metadata_key: str,
        agent_ref_key: str,
    ) -> None:
        self._base_collator = base_collator
        self._sidechannel_keys = (metadata_key, agent_ref_key)

    def __call__(self, features: list[dict[str, tp.Any]]) -> dict[str, tp.Any]:
        saved: dict[str, list[tp.Any]] = {}
        for key in self._sidechannel_keys:
            values = [feature.get(key) for feature in features]
            if any(value is not None for value in values):
                saved[key] = values
        if not saved:
            return self._base_collator(features)
        stripped = [{k: v for k, v in feature.items() if k not in saved} for feature in features]
        batch = self._base_collator(stripped)
        for key, values in saved.items():
            batch[key] = values
        return batch


def _schema_signature(value: object) -> object:
    """Return a structural type signature used to detect heterogeneous Arrow columns."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, dict):
        return ("dict", tuple(sorted((str(k), _schema_signature(v)) for k, v in value.items())))
    if isinstance(value, list):
        item_sigs = tuple(dict.fromkeys(_schema_signature(v) for v in value))
        return ("list", item_sigs)
    return type(value).__name__


def _values_have_incompatible_schemas(values: tp.Sequence[object]) -> bool:
    """Return ``True`` when non-None values cannot safely share one Arrow column type."""
    non_null = [v for v in values if v is not None]
    if len(non_null) <= 1:
        return False
    types = {type(v) for v in non_null}
    if len(types) > 1:
        # Allow int + float widening, but reject dict vs str, list vs str, etc.
        if not types <= {int, float}:
            return True
    if isinstance(non_null[0], dict):
        # Check for conflicting value types for shared keys across rows.
        key_types: dict[str, object] = {}
        for item in non_null:
            assert isinstance(item, dict)
            for k, v in item.items():
                sig = _schema_signature(v)
                if sig is None:
                    continue
                prev = key_types.get(k)
                if prev is None:
                    key_types[k] = sig
                elif prev != sig and not ({prev, sig} <= {"int", "float"}):
                    return True
    elif isinstance(non_null[0], list):
        item_sigs = {_schema_signature(v) for v in non_null}
        if len(item_sigs) > 1:
            # Lists of dicts with incompatible value types can fail Arrow conversion.
            elem_dicts = [elem for row in non_null for elem in row if isinstance(elem, dict)]
            if elem_dicts and _values_have_incompatible_schemas(elem_dicts):
                return True
    return False


def _normalize_heterogeneous_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Normalize heterogeneous ``agent_ref`` or ``prompt`` columns for Arrow compatibility."""
    if not rows:
        return rows

    normalized = [dict(row) for row in rows]

    agent_refs = [row.get("agent_ref") for row in normalized]
    if _values_have_incompatible_schemas(agent_refs):
        for row in normalized:
            ref = row.get("agent_ref")
            if ref is not None and not isinstance(ref, str):
                row["agent_ref"] = json.dumps(ref, sort_keys=True)

    prompts = [row.get("prompt") for row in normalized]
    if _values_have_incompatible_schemas(prompts):
        has_list = any(isinstance(p, list) for p in prompts)
        if has_list:
            for row in normalized:
                p = row.get("prompt")
                if isinstance(p, str):
                    row["prompt"] = [{"role": "user", "content": p}]
                elif isinstance(p, list):
                    clean_msgs = []
                    for msg in p:
                        if isinstance(msg, dict):
                            clean_msg = {}
                            for k, v in msg.items():
                                clean_msg[str(k)] = v if isinstance(v, str) or v is None else json.dumps(v, sort_keys=True)
                            clean_msgs.append(clean_msg)
                        else:
                            clean_msgs.append({"role": "user", "content": str(msg)})
                    row["prompt"] = clean_msgs
        else:
            for row in normalized:
                p = row.get("prompt")
                if p is not None and not isinstance(p, str):
                    row["prompt"] = json.dumps(p, sort_keys=True)

    return normalized


def _iter_jsonl_lines(path: str | Path | UniversalPath) -> tp.Iterator[str]:
    """Stream JSONL lines without double-loading local files into memory."""
    if isinstance(path, Path):
        with path.open("r", encoding="utf-8") as handle:
            yield from handle
        return
    if isinstance(path, str) and not path.startswith(("gs://", "s3://", "http://", "https://", "az://")):
        local_path = Path(path)
        if local_path.exists():
            with local_path.open("r", encoding="utf-8") as handle:
                yield from handle
            return
    task_path = path if isinstance(path, UniversalPath) else ePath(path)
    raw_str = str(task_path)
    if not raw_str.startswith(("gs://", "s3://", "http://", "https://", "az://")):
        local_path = Path(raw_str)
        if local_path.exists():
            with local_path.open("r", encoding="utf-8") as handle:
                yield from handle
            return
    yield from io.StringIO(task_path.read_text())


def load_nemo_gym_jsonl(path: str | Path | UniversalPath) -> object:
    """Load a NeMo Gym JSONL task file into a Dataset when available.

    Streams local files line-by-line, stores the original task JSON in
    ``metadata``, exposes a ``prompt`` (falling back to the row index when
    absent, matching TRL's NeMo Gym script convention), and safely normalizes
    heterogeneous ``agent_ref`` / ``prompt`` schemas so Arrow dataset
    construction does not fail on mixed-schema JSONL files.
    """

    rows: list[dict[str, object]] = []
    for idx, line in enumerate(_iter_jsonl_lines(path)):
        stripped = line.strip()
        if not stripped:
            continue
        item = json.loads(stripped)
        prompt = item.get("prompt")
        if prompt is None:
            params = item.get("responses_create_params", {})
            input_payload = params.get("input") if isinstance(params, dict) else None
            prompt = input_payload if input_payload is not None else str(idx)
        rows.append({"prompt": prompt, "metadata": json.dumps(item), "agent_ref": item.get("agent_ref")})

    try:
        from datasets import Dataset
    except ImportError:
        return rows

    rows = _normalize_heterogeneous_rows(rows)
    try:
        return Dataset.from_list(rows)
    except Exception:
        # Final fallback: serialize non-string agent_ref and heterogeneous prompts as JSON strings.
        safe_rows = []
        for row in rows:
            prompt = row.get("prompt")
            agent_ref = row.get("agent_ref")
            safe_rows.append(
                {
                    "prompt": prompt if isinstance(prompt, str) else json.dumps(prompt, sort_keys=True),
                    "metadata": row["metadata"],
                    "agent_ref": (
                        None
                        if agent_ref is None
                        else (agent_ref if isinstance(agent_ref, str) else json.dumps(agent_ref, sort_keys=True))
                    ),
                }
            )
        return Dataset.from_list(safe_rows)
