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

import json
import typing as tp
from pathlib import Path


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
    return [float(reward) for reward in environment_rewards]


def _decode_json_if_needed(value: object) -> object:
    """Decode byte/string JSON payloads while preserving non-JSON values.

    NeMo task metadata can come from JSONL files, datasets, or environment
    wrappers. Byte values are decoded first; invalid JSON strings are returned
    unchanged instead of being treated as hard failures.
    """
    if isinstance(value, bytes | bytearray):
        value = value.decode()
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return value
    return value


def _as_mapping(value: object) -> dict[str, object]:
    """Normalize dict, pydantic-like, JSON, or scalar payloads to a mapping.

    Dict inputs are copied, objects with ``model_dump`` use that structured
    representation, and scalar fallback values are wrapped under ``"value"``.
    This gives environment factories a predictable metadata shape.
    """
    value = _decode_json_if_needed(value)
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
    """

    if isinstance(step_result, tuple):
        if len(step_result) == 5:
            observation, reward, terminated, truncated, info = step_result
            return observation, float(reward), bool(terminated), bool(truncated), _as_mapping(info)
        if len(step_result) == 4:
            observation, reward, done, info = step_result
            return observation, float(reward), bool(done), False, _as_mapping(info)

    if isinstance(step_result, dict):
        reward = step_result.get("reward", step_result.get("env_reward", 0.0))
        observation = step_result.get("observation", step_result.get("response", ""))
        terminated = step_result.get("terminated", step_result.get("done", False))
        truncated = step_result.get("truncated", False)
        info = step_result.get("info", {})
        if not isinstance(info, dict):
            info = {"value": info}
        if "num_turns" in step_result:
            info = {**info, "num_turns": step_result["num_turns"]}
        return observation, float(reward), bool(terminated), bool(truncated), info

    observation = _get_from_mapping_or_object(step_result, "observation", "")
    reward = _get_from_mapping_or_object(step_result, "reward", 0.0)
    terminated = _get_from_mapping_or_object(step_result, "terminated", False)
    truncated = _get_from_mapping_or_object(step_result, "truncated", False)
    info = _get_from_mapping_or_object(step_result, "info", {})
    return observation, float(reward), bool(terminated), bool(truncated), _as_mapping(info)


def load_nemo_gym_jsonl(path: str | Path) -> object:
    """Load a NeMo Gym JSONL task file into a Dataset when available.

    Each row keeps the original task as JSON in ``metadata`` and exposes a
    prompt. When the task has no explicit prompt/input, the row index is used,
    matching TRL's NeMo Gym script convention.
    """

    rows: list[dict[str, object]] = []
    with Path(path).open() as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
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
    return Dataset.from_list(rows)
