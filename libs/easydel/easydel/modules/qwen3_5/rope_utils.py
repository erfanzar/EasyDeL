# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0
"""Shared Qwen3.5 dense/MoE configuration compatibility for multimodal RoPE."""

from collections.abc import Mapping


def hf_supports_mrope_rope_type() -> bool:
    """Return whether installed HF exposes an mRoPE validator, without torch models."""
    try:
        from transformers import modeling_rope_utils
    except Exception:
        return False
    validators = getattr(modeling_rope_utils, "ROPE_VALIDATION_FUNCTIONS", None)
    return isinstance(validators, Mapping) and "mrope" in validators


def normalize_rope_scaling_for_mrope(rope_scaling: dict | None) -> dict | None:
    """Normalize mRoPE keys for the installed HF version without mutating input.

    mRoPE validators receive ``rope_type='mrope'`` when available; older HF
    versions receive ``'default'`` while retaining all modality-specific keys.
    Explicit non-mRoPE scaling types are left unchanged.
    """
    if not isinstance(rope_scaling, dict):
        return rope_scaling
    normalized = dict(rope_scaling)
    if "mrope_section" in normalized or "mrope_interleaved" in normalized:
        rope_type = normalized.get("rope_type", normalized.get("type"))
        if rope_type in (None, "default", "mrope"):
            target_rope_type = "mrope" if hf_supports_mrope_rope_type() else "default"
            normalized["rope_type"] = target_rope_type
            normalized["type"] = target_rope_type
    return normalized
