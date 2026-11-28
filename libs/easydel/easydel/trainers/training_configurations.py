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
"""Top-level :class:`TrainingArguments` dataclass and supporting helpers.

This module defines the configuration surface shared by every concrete
EasyDeL trainer.  ``TrainingArguments`` exposes hyperparameters for the
optimizer, scheduler, sharding, dataloaders, logging/checkpointing,
generation and benchmarking, plus utility methods to construct an
optimizer/scheduler pair and resolve save directories.

Subclassed configs (DPO, GRPO, SFT, ...) extend this base with
algorithm-specific knobs while reusing the shared serialization,
checkpoint-resume, and W&B-init plumbing.
"""

from __future__ import annotations

import ast
import collections.abc
import datetime
import functools
import hashlib
import json
import math
import os
import random
import re
import secrets
import threading
import traceback
import types
import typing as tp
import warnings
import weakref
from copy import deepcopy
from dataclasses import InitVar, dataclass, field, fields

import jax
import jax.numpy as jnp
import numpy as np
import spectrax as spx
from eformer.loggings import get_logger
from eformer.optimizers import OptimizerFactory, SchedulerConfig
from eformer.paths import ePath, ePathLike
from jax.sharding import PartitionSpec
from optax import GradientTransformation  # pyright: ignore[reportMissingTypeStubs]
from spectrax.serialization import Checkpointer

from easydel.infra.elarge.benchmarking import normalize_benchmark_configs
from easydel.infra.elarge.types import BenchmarkConfig
from easydel.infra.errors import EasyDeLTimerError
from easydel.infra.etils import (
    AVAILABLE_OPTIMIZERS,
    AVAILABLE_SCHEDULERS,
    AVAILABLE_SPARSE_MODULE_TYPES,
    EasyDeLOptimizers,
    EasyDeLSchedulers,
    MpMdSchedulers,
)
from easydel.infra.loss_utils import LossConfig
from easydel.trainers.pose import PoSEConfig
from easydel.trainers.process_encoder import ProcessEncoderConfig
from easydel.trainers.randomized_yarn import RandomizedYaRNConfig
from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn
from easydel.utils.eyvan_helpers import (
    _ensure_local_eyvan_store_writable,
    is_eyvan_auto_track_enabled,
)
from easydel.utils.eyvan_helpers import (
    set_eyvan_auto_track_enabled as set_eyvan_auto_track_enabled,
)

from .metrics import LogWatcher, MetricsHistogram, compute_weight_stats, run_watchers
from .training_utils import (
    AFFINE_SUPPORTED_BITS as AFFINE_SUPPORTED_BITS,
)
from .training_utils import (
    FIXED_QUANTIZATION_BITS_BY_MODE as FIXED_QUANTIZATION_BITS_BY_MODE,
)
from .training_utils import (
    QuantizationMode as QuantizationMode,
)
from .training_utils import (
    validate_quantization_bits,
)
from .utils import JaxDistributedConfig

try:
    import wandb
except ImportError:
    wandb = None

from easydel.utils.helpers import SummaryWriter

if tp.TYPE_CHECKING:
    from jax import Array
    from torch import Tensor
else:
    Array, Tensor = [tp.Any] * 2


MetricsType = dict[str, float | list | tuple | np.ndarray | Array | Tensor | None]
logger = get_logger(__name__)

STE_QAT_QUANTIZATION_MODES: tuple[QuantizationMode, ...] = tp.get_args(QuantizationMode)
STE_QAT_QUANTIZATION_MODES_DOC = ", ".join(f"'{mode}'" for mode in STE_QAT_QUANTIZATION_MODES)


def get_safe_arr(xs):
    """Convert single-element arrays to Python scalars for safe logging.

    Args:
        xs: Value to normalize. Can be a numpy/JAX array or scalar.

    Returns:
        Python scalar if input is a size-1 array, otherwise the original value.
    """
    if isinstance(xs, np.ndarray | np.generic | jax.Array):
        if xs.size == 1:  # Only try .item() on size-1 arrays
            return xs.item()
        return xs
    return xs


def _is_dtype_like(value: tp.Any) -> bool:
    """Return True for numpy/JAX dtype objects and scalar dtype classes (e.g. ``jnp.bfloat16``)."""
    if isinstance(value, np.dtype):
        return True
    if isinstance(value, type):
        if issubclass(value, np.generic):
            return True
        if isinstance(getattr(value, "dtype", None), np.dtype):
            return True
    return False


def _parse_dtype(value: tp.Any) -> tp.Any:
    """Normalize a serialized or string dtype representation back to a JAX dtype."""
    if value is None or _is_dtype_like(value):
        return value
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        if token.startswith("<class '") and token.endswith("'>"):
            token = token[len("<class '") : -len("'>")].rsplit(".", 1)[-1]
        resolved = getattr(jnp, token, None)
        if resolved is not None and _is_dtype_like(resolved):
            return resolved
        try:
            return jnp.dtype(token)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid dtype specified: {value!r}") from exc
    return value


def _is_real_number(value: tp.Any) -> bool:
    """Return True for real numeric scalars, explicitly excluding booleans."""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


def _is_real_int(value: tp.Any) -> bool:
    """Return True for integer scalars, explicitly excluding booleans."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _expects_tuple_type(field_type: tp.Any) -> bool:
    """Return whether a type annotation is a tuple or an optional/union tuple (and not a list)."""
    if field_type is None:
        return False
    origin = tp.get_origin(field_type)
    if origin is tuple or field_type is tuple:
        return True
    if origin in (tp.Union, types.UnionType):
        args = [arg for arg in tp.get_args(field_type) if arg is not type(None)]
        has_tuple = any(tp.get_origin(arg) is tuple or arg is tuple for arg in args)
        has_list = any(tp.get_origin(arg) is list or arg is list for arg in args)
        return has_tuple and not has_list
    return False


def _close_summary_writer(writer: tp.Any) -> None:
    """Flush and close a TensorBoard SummaryWriter safely."""
    if writer is None:
        return
    try:
        if hasattr(writer, "flush"):
            writer.flush()
    except Exception:
        pass
    try:
        if hasattr(writer, "close"):
            writer.close()
    except Exception:
        pass


_EYVAN_TRACKING_URI_ENV = "EYVAN_TRACKING_URI"
_EYVAN_WORKLOAD_ID_ENV = "EYVAN_WORKLOAD_ID"
_EYVAN_DEFAULT_TAGS: tuple[str, ...] = ("EasyDeL", "Jax", "Train")
_EYVAN_MAX_TAGS = 64
_EYVAN_MAX_TAG_LEN = 64
_EYVAN_MAX_NAME_LEN = 256
_EYVAN_MAX_AUTO_NAME_LEN = 64
_EYVAN_MAX_KEY_LEN = 128
_EYVAN_MAX_HIST_BINS = 512
_EYVAN_MAX_EXTRA = 32
_EYVAN_MAX_ACTOR_LEN = 64
_EYVAN_MAX_ENV_LEN = 128
_EYVAN_MAX_CHECKPOINT_LEN = 256
_EYVAN_MAX_LABEL_LEN = 128
_EYVAN_MAX_ACTION_NAME_LEN = 64
_EYVAN_MAX_ACTION_NAMES = 256
_EYVAN_MAX_GROUP_SIZE = 64
_EYVAN_MAX_ROLLOUT_STEPS = 10_000
_EYVAN_MAX_TRACKS = 32
_EYVAN_MAX_TEXT_LEN = 32_768
_EYVAN_MAX_INT64 = (1 << 63) - 1
_EYVAN_MIN_INT64 = -(1 << 63)
_EYVAN_EPISODE_KINDS = frozenset({"train", "eval"})
_EYVAN_KEY_RE = re.compile(r"[A-Za-z0-9_./-]{1,128}")
_EYVAN_KEY_UNSAFE_RE = re.compile(r"[^A-Za-z0-9_./-]+")
_EYVAN_SLUG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_EYVAN_SLUG_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_EYVAN_TAG_SLUG_UNSAFE_RE = re.compile(r"[^a-z0-9._-]+")
_EYVAN_NAME_TOKEN_UNSAFE_RE = re.compile(r"[^a-z0-9-]+")
_EYVAN_FINISH_STATUSES = frozenset({"finished", "failed", "aborted"})
_EYVAN_INIT_LOCK = threading.Lock()
_EYVAN_GCP_CONTEXT_LOCK = threading.Lock()
_EYVAN_GCP_CONTEXT_CACHE: dict[int, tp.Any] = {}
_EYVAN_SYSTEM_RANDOM = secrets.SystemRandom()

_EYVAN_NAME_ADJECTIVES: tuple[str, ...] = (
    "agate",
    "amber",
    "arctic",
    "astral",
    "azure",
    "brisk",
    "cedar",
    "cobalt",
    "coral",
    "crisp",
    "crimson",
    "dawn",
    "dusk",
    "emerald",
    "feral",
    "flint",
    "frost",
    "gilded",
    "glacier",
    "golden",
    "granite",
    "harbor",
    "ivory",
    "jade",
    "keen",
    "lively",
    "lunar",
    "marble",
    "misty",
    "noble",
    "obsidian",
    "ochre",
    "opal",
    "pine",
    "polar",
    "quartz",
    "quiet",
    "radiant",
    "rapid",
    "scarlet",
    "silver",
    "solar",
    "steady",
    "storm",
    "swift",
    "velvet",
    "vivid",
    "zephyr",
)
_EYVAN_NAME_NOUNS: tuple[str, ...] = (
    "badger",
    "beacon",
    "bison",
    "canyon",
    "comet",
    "condor",
    "coyote",
    "crane",
    "crest",
    "drift",
    "eagle",
    "falcon",
    "fjord",
    "forge",
    "fox",
    "grove",
    "hawk",
    "heron",
    "horizon",
    "ibex",
    "kestrel",
    "lantern",
    "lynx",
    "meadow",
    "mesa",
    "nebula",
    "orbit",
    "osprey",
    "otter",
    "owl",
    "peak",
    "phoenix",
    "pioneer",
    "plume",
    "puma",
    "quasar",
    "raven",
    "ridge",
    "river",
    "spark",
    "sparrow",
    "summit",
    "swift",
    "tiger",
    "trail",
    "vector",
    "wolf",
    "wren",
)
_EYVAN_GENERIC_MODEL_NAMES: frozenset[str] = frozenset(
    {
        "",
        "auto",
        "base",
        "base-trainer",
        "base_trainer",
        "basetrainer",
        "default",
        "easydel",
        "easydel-model",
        "easydel_model",
        "model",
        "module",
        "none",
        "trainer",
        "unknown",
        "unnamed",
    }
)
_EYVAN_GENERIC_TRAINER_SLUGS: frozenset[str] = frozenset(
    {
        "",
        "base",
        "base_trainer",
        "basetrainer",
        "default",
        "none",
        "trainer",
        "training",
        "training_arguments",
        "trainingarguments",
    }
)
_EYVAN_TRAINER_ALIASES: dict[str, str] = {
    "agent_flow": "agentflow",
    "agent-flow": "agentflow",
    "agentflow": "agentflow",
    "agentic_moshpit": "agentic_moshpit",
    "agentic-moshpit": "agentic_moshpit",
    "agenticmoshpit": "agentic_moshpit",
    "async_grpo": "async_grpo",
    "asyncgrpo": "async_grpo",
    "bco": "bco",
    "compaction_rl": "compaction_rl",
    "compactionrl": "compaction_rl",
    "cpo": "cpo",
    "distillation": "distillation",
    "dpo": "dpo",
    "dppo": "dppo",
    "embedding": "embedding",
    "esurge": "esurge",
    "gdpo": "gdpo",
    "gfpo": "gfpo",
    "gkd": "gkd",
    "gmpo": "gmpo",
    "gold": "gold",
    "grpo": "grpo",
    "grpo_replay_buffer": "grpo_replay_buffer",
    "grpo_with_replay_buffer": "grpo_replay_buffer",
    "grpowithreplaybuffer": "grpo_replay_buffer",
    "gspo": "gspo",
    "gspo_token": "gspo_token",
    "gspotoken": "gspo_token",
    "kto": "kto",
    "minillm": "minillm",
    "nash_md": "nash_md",
    "nashmd": "nash_md",
    "nemo_gym": "nemo_gym",
    "nemogym": "nemo_gym",
    "on_policy_distillation": "on_policy_distillation",
    "onpolicydistillation": "on_policy_distillation",
    "online_actor_critic": "online_rl",
    "online_dpo": "online_dpo",
    "online_rl": "online_rl",
    "onlinedpo": "online_dpo",
    "onlinerl": "online_rl",
    "orpo": "orpo",
    "papo": "papo",
    "ppo": "ppo",
    "prm": "prm",
    "reinforce_plus_plus": "reinforce_plus_plus",
    "reinforceplusplus": "reinforce_plus_plus",
    "reward": "reward",
    "rloo": "rloo",
    "rlvr": "rlvr",
    "sao": "sao",
    "sdft": "sdft",
    "sdpo": "sdpo",
    "self_distillation": "sdft",
    "selfdistillation": "sdft",
    "seq_kd": "seq_kd",
    "seqkd": "seq_kd",
    "sft": "sft",
    "sparse_distillation": "sparse_distillation",
    "speculative_decoding": "speculative_decoding",
    "ssd": "ssd",
    "tpo": "tpo",
    "vapo": "vapo",
    "xpo": "xpo",
}
_EYVAN_RL_TRAINER_SLUGS: frozenset[str] = frozenset(
    {
        "agentflow",
        "agentic_moshpit",
        "async_grpo",
        "compaction_rl",
        "dppo",
        "gdpo",
        "gfpo",
        "gmpo",
        "grpo",
        "grpo_replay_buffer",
        "gspo",
        "gspo_token",
        "minillm",
        "nash_md",
        "nemo_gym",
        "online_dpo",
        "online_rl",
        "papo",
        "ppo",
        "reinforce_plus_plus",
        "rloo",
        "rlvr",
        "sao",
        "sdpo",
        "tpo",
        "vapo",
        "xpo",
    }
)
_EYVAN_GPU_CHIP_RE = re.compile(
    r"\b(gb200|b200|b100|h200|h100|a100|a10g|a10|l40s|l40|l4|v100|t4|mi325x|mi300x|mi250x|mi250)\b"
)
_EYVAN_RTX_CHIP_RE = re.compile(r"\brtx[\s_-]*(\d{4}(?:[\s_-]*ti)?)\b")
_EYVAN_TPU_GEN_RE = re.compile(r"\bv(\d+[a-z]?)(?:[-_\s]|$)")
_EYVAN_GCP_ZONE_RE = re.compile(r"^([a-z0-9]+-[a-z0-9]+)-[a-z]$")


def _import_eyvan():
    """Lazily import ``eyvan`` when installed, returning ``None`` if absent."""
    try:
        import eyvan
    except ImportError:
        return None
    return eyvan


def _import_eyvan_track():
    """Lazily import ``eyvan.track`` when installed, returning ``None`` if absent."""
    try:
        import eyvan.track as eyvan_track
    except ImportError:
        return None
    return eyvan_track


def _import_eyvan_storage():
    """Lazily import ``eyvan.storage`` when installed, returning ``None`` if absent."""
    try:
        import eyvan.storage as eyvan_storage
    except ImportError:
        return None
    return eyvan_storage


def _clear_eyvan_auto_caches() -> None:
    """Clear process-level Eyvan auto-detection caches (used by tests)."""
    with _EYVAN_GCP_CONTEXT_LOCK:
        _EYVAN_GCP_CONTEXT_CACHE.clear()


def _detect_eyvan_gcp_context() -> tp.Any:
    """Safely fetch and cache ``eyvan.storage.detect_gcp_context()`` without raising."""
    storage = _import_eyvan_storage()
    if storage is None:
        return None
    detect_fn = getattr(storage, "detect_gcp_context", None)
    if not callable(detect_fn):
        return None
    cache_key = id(detect_fn)
    with _EYVAN_GCP_CONTEXT_LOCK:
        if cache_key in _EYVAN_GCP_CONTEXT_CACHE:
            return _EYVAN_GCP_CONTEXT_CACHE[cache_key]
    try:
        ctx = detect_fn()
    except Exception:
        ctx = None
    with _EYVAN_GCP_CONTEXT_LOCK:
        _EYVAN_GCP_CONTEXT_CACHE[cache_key] = ctx
    return ctx


def _resolve_eyvan_default_tracking_uri(
    *,
    explicit_uri: str | None,
    project: str,
    org: str = "default",
) -> str | None:
    """Resolve the Eyvan tracking store URI with zero-config default bucket discovery.

    Resolution order:
    1. Explicit ``explicit_uri`` if non-empty.
    2. Pre-existing ``EYVAN_TRACKING_URI`` environment variable if set.
    3. ``eyvan.storage.resolve_default_tracking_uri(project=project, org=org, allow_create=False)``.
    4. ``None`` so ``eyvan.track.start()`` falls back to ``~/.eyvan/tracking/default/<project>``.
    """
    if isinstance(explicit_uri, str) and explicit_uri.strip():
        return explicit_uri.strip()
    env_uri = os.environ.get(_EYVAN_TRACKING_URI_ENV, "").strip()
    if env_uri:
        return env_uri
    storage = _import_eyvan_storage()
    if storage is None:
        return None
    resolver = getattr(storage, "resolve_default_tracking_uri", None)
    if not callable(resolver):
        return None
    try:
        resolved = resolver(project=project, org=org, allow_create=False)
    except Exception:
        return None
    if isinstance(resolved, str) and resolved.strip():
        return resolved.strip()
    return None


def _camel_to_snake(text: str) -> str:
    """Convert CamelCase identifiers into lowercase snake_case."""
    step1 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    step2 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", step1)
    return step2.lower()


def _normalize_eyvan_trainer_slug(raw: tp.Any) -> str | None:
    """Normalize a trainer prefix, class name, or config name to a concise lowercase slug."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    for suffix in ("Trainer", "Config", "Arguments"):
        if text.endswith(suffix) and len(text) > len(suffix):
            text = text[: -len(suffix)]
    compact_key = re.sub(r"[^a-z0-9]+", "", text.lower())
    if compact_key in _EYVAN_TRAINER_ALIASES:
        return _EYVAN_TRAINER_ALIASES[compact_key]
    snake = _camel_to_snake(text)
    snake = _EYVAN_TAG_SLUG_UNSAFE_RE.sub("_", snake)
    snake = re.sub(r"_+", "_", snake).strip("._-")
    if snake in _EYVAN_TRAINER_ALIASES:
        return _EYVAN_TRAINER_ALIASES[snake]
    if not snake or snake in _EYVAN_GENERIC_TRAINER_SLUGS or compact_key in _EYVAN_GENERIC_TRAINER_SLUGS:
        return None
    return snake[:_EYVAN_MAX_TAG_LEN]


def _extract_eyvan_model_slug(
    model_name: tp.Any = None,
    *,
    model: tp.Any = None,
    prefer_model_type: bool = True,
    for_run_name: bool = False,
) -> str | None:
    """Extract a clean lowercase model slug from ``model`` or ``model_name``, skipping generic placeholders."""
    candidates: list[str] = []
    if model is not None:
        inner_model = getattr(model, "model", model)
        model_cfg = getattr(inner_model, "config", None)
        model_type = getattr(model_cfg, "model_type", None) if model_cfg is not None else None
        if isinstance(model_type, str) and model_type.strip():
            candidates.append(model_type.strip())
        internal_type = getattr(inner_model, "_model_type", None)
        if isinstance(internal_type, str) and internal_type.strip():
            candidates.append(internal_type.strip())

    name_candidate: str | None = None
    if isinstance(model_name, str) and model_name.strip():
        raw_name = model_name.strip().rstrip("/")
        # Strip org/path prefix (e.g. "Qwen/Qwen3-8B" -> "Qwen3-8B", "/ckpt/qwen3/step-10" -> "qwen3").
        segments = [seg for seg in raw_name.replace("\\", "/").split("/") if seg and not seg.endswith(":")]
        if segments:
            chosen = segments[-1]
            if len(segments) >= 2 and re.fullmatch(r"(?:checkpoint|ckpt|step|run)[-_]?\d+", chosen.lower()):
                chosen = segments[-2]
            name_candidate = chosen

    ordered = [*candidates, name_candidate] if prefer_model_type else [name_candidate, *candidates]
    for item in ordered:
        if not item:
            continue
        lower = item.strip().lower()
        if lower in _EYVAN_GENERIC_MODEL_NAMES:
            continue
        if for_run_name:
            slug = _EYVAN_NAME_TOKEN_UNSAFE_RE.sub("-", lower)
            slug = re.sub(r"-{2,}", "-", slug).strip(".-")
        else:
            slug = _EYVAN_TAG_SLUG_UNSAFE_RE.sub("-", lower)
            slug = re.sub(r"-{2,}", "-", slug).strip("._-")
        if not slug or slug in _EYVAN_GENERIC_MODEL_NAMES:
            continue
        return slug[:_EYVAN_MAX_TAG_LEN]
    return None


def _extract_eyvan_env_slug(env_name: tp.Any) -> str | None:
    """Normalize an environment identifier into a compact lowercase tag slug."""
    if env_name is None:
        return None
    raw: str | None = None
    if isinstance(env_name, str):
        raw = env_name.strip()
    elif isinstance(env_name, collections.abc.Mapping):
        val = env_name.get("name") or env_name.get("type")
        if isinstance(val, str):
            raw = val.strip()
    elif isinstance(env_name, type):
        raw = env_name.__name__
    elif isinstance(env_name, functools.partial):
        fn_name = getattr(env_name.func, "__name__", None)
        if isinstance(fn_name, str) and fn_name != "<lambda>":
            raw = fn_name
    elif callable(env_name):
        fn_name = getattr(env_name, "__name__", None)
        if isinstance(fn_name, str) and fn_name != "<lambda>":
            raw = fn_name
    if not raw:
        return None
    slug = _camel_to_snake(raw.lstrip("_"))
    slug = _EYVAN_TAG_SLUG_UNSAFE_RE.sub("_", slug)
    slug = re.sub(r"_+", "_", slug).strip("._-")
    if not slug or slug in {"none", "unknown", "default"}:
        return None
    # Leave room for the "env:" prefix within Eyvan's 64-char tag limit.
    return slug[: _EYVAN_MAX_TAG_LEN - 4]


def _extract_quantization_method(obj: tp.Any) -> str | None:
    """Extract a normalized quantization method name from a mode string, config object, or mapping."""
    if obj is None or isinstance(obj, bool):
        return None
    if hasattr(obj, "value") and isinstance(obj.value, str):
        obj = obj.value
    if isinstance(obj, str):
        cleaned = _EYVAN_TAG_SLUG_UNSAFE_RE.sub("-", obj.strip().lower()).strip("._-")
        return cleaned[: _EYVAN_MAX_TAG_LEN - 6] if cleaned and cleaned not in {"none", "false"} else None
    if isinstance(obj, collections.abc.Mapping):
        for key in ("quant_method", "quant_type", "quant_dtype", "quantization_type", "mode", "method"):
            val = obj.get(key)
            if val is not None:
                found = _extract_quantization_method(val)
                if found:
                    return found
        return None
    for attr in ("quant_method", "quant_type", "quant_dtype", "quantization_type", "mode", "method"):
        val = getattr(obj, attr, None)
        if val is not None:
            found = _extract_quantization_method(val)
            if found:
                return found
    return None


def _normalize_eyvan_chip_generation(
    device_kind: str | None,
    accelerator_type: str | None = None,
) -> str | None:
    """Normalize a JAX ``device_kind`` or GCP ``accelerator_type`` into a concise chip generation tag."""
    for raw in (device_kind, accelerator_type):
        if not isinstance(raw, str) or not raw.strip():
            continue
        text = raw.strip().lower()
        if text in {"cpu", "host"}:
            continue
        if "v6e" in text or "v6 lite" in text or "v6_lite" in text or "v6lite" in text or "trillium" in text:
            return "v6e"
        if "v5e" in text or "v5 lite" in text or "v5_lite" in text or "v5lite" in text:
            return "v5e"
        if "v5p" in text:
            return "v5p"
        if "tpu" in text and re.search(r"\bv5(?:\b|[-_])", text):
            return "v5p"
        gpu_match = _EYVAN_GPU_CHIP_RE.search(text)
        if gpu_match is not None:
            return gpu_match.group(1)
        rtx_match = _EYVAN_RTX_CHIP_RE.search(text)
        if rtx_match is not None:
            return "rtx" + re.sub(r"[\s_-]+", "", rtx_match.group(1))
        if "tpu" in text or re.match(r"^v\d+[a-z]?-\d+$", text):
            tpu_match = _EYVAN_TPU_GEN_RE.search(text)
            if tpu_match is not None:
                return f"v{tpu_match.group(1)}"
    return None


def _detect_eyvan_hardware_tags() -> tuple[str, ...]:
    """Detect metadata-only hardware, topology, and GCP region tags without JAX collectives or compiles."""
    tags: list[str] = []
    platform: str | None = None
    total_devices = 0
    local_devices = 0
    process_count = 1
    device_kind: str | None = None

    try:
        raw_backend = str(jax.default_backend()).strip().lower()
        if raw_backend in {"cuda", "rocm", "gpu"}:
            platform = "gpu"
        elif raw_backend in {"tpu", "cpu"}:
            platform = raw_backend
        elif raw_backend:
            platform = _EYVAN_TAG_SLUG_UNSAFE_RE.sub("-", raw_backend).strip("._-") or None
    except Exception:
        platform = None

    try:
        total_devices = max(0, int(jax.device_count()))
    except Exception:
        total_devices = 0
    try:
        local_devices = max(0, int(jax.local_device_count()))
    except Exception:
        local_devices = 0
    try:
        process_count = max(1, int(jax.process_count()))
    except Exception:
        process_count = 1
    try:
        local_devs = jax.local_devices()
        if local_devs:
            kind = getattr(local_devs[0], "device_kind", None)
            if isinstance(kind, str) and kind.strip():
                device_kind = kind.strip()
    except Exception:
        device_kind = None

    gcp_ctx = _detect_eyvan_gcp_context()
    accel_type = getattr(gcp_ctx, "accelerator_type", None) if gcp_ctx is not None else None
    chip_gen = _normalize_eyvan_chip_generation(device_kind, accel_type)

    if platform:
        tags.append(platform)
    if chip_gen:
        tags.append(chip_gen)
        if process_count > 1:
            per_host = local_devices if local_devices > 0 else (total_devices // process_count)
            if per_host > 0:
                tags.append(f"{chip_gen}-{per_host}x{process_count}")
            elif total_devices > 0:
                tags.append(f"{chip_gen}-{total_devices}")
        elif total_devices > 0:
            tags.append(f"{chip_gen}-{total_devices}")
    if platform in {"tpu", "gpu"} and total_devices > 1:
        tags.append(f"{total_devices}-chips")

    if gcp_ctx is not None:
        raw_region = getattr(gcp_ctx, "region", None)
        if not (isinstance(raw_region, str) and raw_region.strip()):
            raw_zone = getattr(gcp_ctx, "zone", None)
            if isinstance(raw_zone, str) and raw_zone.strip():
                zone_match = _EYVAN_GCP_ZONE_RE.match(raw_zone.strip().lower())
                if zone_match is not None:
                    raw_region = zone_match.group(1)
        if isinstance(raw_region, str) and raw_region.strip():
            region_slug = _EYVAN_TAG_SLUG_UNSAFE_RE.sub("-", raw_region.strip().lower()).strip("._-")
            if region_slug:
                tags.append(region_slug[:_EYVAN_MAX_TAG_LEN])

    return tuple(dict.fromkeys(tags))


def _fallback_generate_eyvan_run_name(*, prefix: str | None = None, seed: int | None = None) -> str:
    """Generate a W&B-style ``<adj>-<noun>-<num>`` run name using only stdlib randomness."""
    rng = random.Random(seed) if seed is not None else _EYVAN_SYSTEM_RANDOM
    adj = rng.choice(_EYVAN_NAME_ADJECTIVES)
    noun = rng.choice(_EYVAN_NAME_NOUNS)
    num = rng.randint(1, 99)
    suffix = f"{adj}-{noun}-{num}"
    if prefix:
        clean_prefix = _EYVAN_NAME_TOKEN_UNSAFE_RE.sub("-", str(prefix).strip().lower())
        clean_prefix = re.sub(r"-{2,}", "-", clean_prefix).strip(".-")
        if clean_prefix:
            max_prefix = _EYVAN_MAX_AUTO_NAME_LEN - len(suffix) - 1
            if max_prefix > 0:
                clipped = clean_prefix[:max_prefix].rstrip(".-")
                if clipped:
                    return f"{clipped}-{suffix}"
    return suffix[:_EYVAN_MAX_AUTO_NAME_LEN]


def _generate_eyvan_wandb_style_name(*, prefix: str | None = None, seed: int | None = None) -> str:
    """Generate a readable W&B-style run name (<=64 chars) via ``eyvan.track.generate_run_name`` or stdlib fallback."""
    clean_prefix: str | None = None
    if prefix:
        norm = _EYVAN_NAME_TOKEN_UNSAFE_RE.sub("-", str(prefix).strip().lower())
        norm = re.sub(r"-{2,}", "-", norm).strip(".-")
        if norm:
            # Reserve room for a typical "-adjective-noun-99" suffix while staying <= 64 chars.
            clean_prefix = norm[:36].rstrip(".-") or None

    eyvan_track = _import_eyvan_track()
    gen_fn = getattr(eyvan_track, "generate_run_name", None) if eyvan_track is not None else None
    if callable(gen_fn):
        try:
            generated = gen_fn(prefix=clean_prefix, seed=seed) if seed is not None else gen_fn(prefix=clean_prefix)
            if isinstance(generated, str):
                candidate = generated.strip()
                if candidate:
                    if len(candidate) <= _EYVAN_MAX_AUTO_NAME_LEN:
                        return candidate
                    # Preserve the random suffix at the tail if prefix made the string exceed 64 chars.
                    if clean_prefix and candidate.startswith(f"{clean_prefix}-"):
                        tail = candidate[len(clean_prefix) + 1 :]
                        budget = _EYVAN_MAX_AUTO_NAME_LEN - len(tail) - 1
                        if budget > 0:
                            clipped_prefix = clean_prefix[:budget].rstrip(".-")
                            if clipped_prefix:
                                return f"{clipped_prefix}-{tail}"
                        return tail[:_EYVAN_MAX_AUTO_NAME_LEN].strip(".-") or "easydel-run"
                    return candidate[:_EYVAN_MAX_AUTO_NAME_LEN].rstrip(".-") or "easydel-run"
        except Exception:
            pass
    return _fallback_generate_eyvan_run_name(prefix=clean_prefix, seed=seed)


def _resolve_eyvan_run_name(
    args: tp.Any = None,
    *,
    explicit_name: str | None = None,
    wandb_name: str | None = None,
    model_name: str | None = None,
    model: tp.Any = None,
    trainer: tp.Any = None,
    trainer_slug: str | None = None,
    seed: int | None = None,
) -> str:
    """Resolve an Eyvan run name: explicit name -> ``wandb_name`` -> cached W&B-style auto name.

    Auto-generated names are <=64 characters, optionally prefixed with a compact
    ``<model>-<trainer>`` context slug (skipping generic placeholders), and cached on
    ``args.__dict__["_eyvan_auto_run_name"]`` so repeated calls on the same config instance
    return the exact same name without adding dataclass schema fields.
    """
    raw_explicit = explicit_name if explicit_name is not None else getattr(args, "eyvan_track_name", None)
    if isinstance(raw_explicit, str) and raw_explicit.strip():
        return _sanitize_eyvan_utf8_text(raw_explicit.strip(), _EYVAN_MAX_NAME_LEN).strip() or "easydel-run"

    raw_wandb = wandb_name if wandb_name is not None else getattr(args, "wandb_name", None)
    if isinstance(raw_wandb, str) and raw_wandb.strip():
        return _sanitize_eyvan_utf8_text(raw_wandb.strip(), _EYVAN_MAX_NAME_LEN).strip() or "easydel-run"

    if args is not None and seed is None and hasattr(args, "__dict__"):
        cached = args.__dict__.get("_eyvan_auto_run_name")
        if isinstance(cached, str) and cached.strip():
            return cached

    ctx = args.__dict__.get("_eyvan_runtime_context", {}) if args is not None and hasattr(args, "__dict__") else {}
    resolved_model_name = model_name if model_name is not None else getattr(args, "model_name", None)
    model_token = _extract_eyvan_model_slug(
        resolved_model_name,
        model=model,
        prefer_model_type=False,
        for_run_name=True,
    )
    if not model_token:
        ctx_model_slug = ctx.get("model_slug")
        if isinstance(ctx_model_slug, str) and ctx_model_slug:
            model_token = _EYVAN_NAME_TOKEN_UNSAFE_RE.sub("-", ctx_model_slug.lower()).strip(".-") or None

    resolved_trainer_token: str | None = None
    for candidate in (
        trainer_slug,
        *(ctx.get("extra_trainer_tags") or ()),
        ctx.get("trainer_slug"),
        getattr(args, "trainer_prefix", None),
        type(trainer).__name__ if trainer is not None else None,
        type(args).__name__ if args is not None else None,
    ):
        slug = _normalize_eyvan_trainer_slug(candidate)
        if slug:
            resolved_trainer_token = slug.replace("_", "-")
            break

    prefix_parts: list[str] = []
    if model_token:
        prefix_parts.append(model_token[:20].rstrip(".-"))
    if resolved_trainer_token and resolved_trainer_token not in prefix_parts:
        prefix_parts.append(resolved_trainer_token[:16].rstrip(".-"))
    prefix = "-".join(part for part in prefix_parts if part) or None

    generated = _generate_eyvan_wandb_style_name(prefix=prefix, seed=seed)
    if args is not None and seed is None and hasattr(args, "__dict__"):
        args.__dict__["_eyvan_auto_run_name"] = generated
    return generated


def _merge_eyvan_tags(
    auto_tags: tp.Iterable[tp.Any] | None,
    user_tags: tp.Iterable[tp.Any] | None,
) -> tuple[str, ...]:
    """Merge auto-detected and user tags cleanly within Eyvan's 64-tag / 64-char limits.

    User tags are always preserved (up to 64 unique tags); remaining slots are filled with
    deduplicated auto-detected tags in canonical order.
    """

    def _clean(items: tp.Iterable[tp.Any] | None) -> list[str]:
        out: list[str] = []
        seen_local: set[str] = set()
        for raw in items or ():
            if raw is None:
                continue
            tag = _sanitize_eyvan_utf8_text(str(raw).strip(), _EYVAN_MAX_TAG_LEN).strip()
            if not tag or tag in seen_local:
                continue
            seen_local.add(tag)
            out.append(tag)
        return out

    clean_user = _clean(user_tags)[:_EYVAN_MAX_TAGS]
    clean_auto = _clean(auto_tags)
    user_set = set(clean_user)
    auto_only = [tag for tag in clean_auto if tag not in user_set]

    auto_budget = max(0, _EYVAN_MAX_TAGS - len(clean_user))
    kept_auto = set(auto_only[:auto_budget])
    # Place canonical auto tags first and append user tags (keeping duplicates at their auto position).
    seen: set[str] = set()
    merged: list[str] = []
    for tag in (*clean_auto, *clean_user):
        if tag in seen:
            continue
        if tag not in user_set and tag not in kept_auto:
            continue
        seen.add(tag)
        merged.append(tag)
        if len(merged) >= _EYVAN_MAX_TAGS:
            break
    return tuple(merged)


def _auto_detect_eyvan_tags(
    args: tp.Any,
    *,
    trainer: tp.Any = None,
    model: tp.Any = None,
    env_name: str | None = None,
    extra_trainer_tags: tp.Iterable[str] | None = None,
    include_user_tags: bool = True,
) -> tuple[str, ...]:
    """Auto-detect rich framework, trainer, RL/feature, model, and hardware tags for Eyvan tracking.

    Inspects trainer/config/model metadata safely via ``getattr`` (zero circular imports) and
    merges with user ``eyvan_track_tags`` (preserving user tags within Eyvan's 64-tag / 64-char limits).
    """
    ctx = args.__dict__.get("_eyvan_runtime_context", {}) if args is not None and hasattr(args, "__dict__") else {}
    auto: list[str] = ["easydel", "jax"]

    # 1. Trainer identifiers (outer wrapper + inner trainer/prefix/config class) and RL mode tag.
    trainer_slugs: list[str] = []
    for candidate in (
        *(extra_trainer_tags or ()),
        *(ctx.get("extra_trainer_tags") or ()),
        ctx.get("trainer_slug"),
        getattr(args, "trainer_prefix", None),
        type(trainer).__name__ if trainer is not None else None,
        type(args).__name__ if args is not None else None,
    ):
        slug = _normalize_eyvan_trainer_slug(candidate)
        if slug and slug not in trainer_slugs:
            trainer_slugs.append(slug)
    auto.extend(trainer_slugs)

    resolved_env_raw = (
        env_name
        if env_name is not None
        else (ctx.get("env_name") or getattr(trainer, "_eyvan_env_name", None) or getattr(args, "environment", None))
    )
    env_slug = _extract_eyvan_env_slug(resolved_env_raw)

    is_icepop = bool(getattr(args, "esurge_importance_sampling_correction", False))
    is_async_rl = bool(getattr(args, "async_mode", False)) or bool(getattr(args, "asynchronous_rollouts", False))
    is_rl = (
        any(slug in _EYVAN_RL_TRAINER_SLUGS for slug in trainer_slugs)
        or is_icepop
        or bool(env_slug)
        or (is_async_rl and any(hasattr(args, attr) for attr in ("num_generations", "advantage_estimator", "cliprange")))
    )
    if is_rl:
        auto.append("rl")

    # 2. RL & training feature tags (detected cleanly via getattr).
    if is_icepop:
        auto.append("icepop")
        is_mode = getattr(args, "esurge_importance_sampling_mode", None)
        if isinstance(is_mode, str) and is_mode.strip():
            mode_slug = _EYVAN_TAG_SLUG_UNSAFE_RE.sub("_", is_mode.strip().lower()).strip("._-")
            if mode_slug:
                auto.append(mode_slug[:_EYVAN_MAX_TAG_LEN])

    if is_async_rl:
        auto.append("async_rl")

    if env_slug:
        auto.append(f"env:{env_slug}")

    beta = getattr(args, "beta", None)
    if _is_real_number(beta):
        beta_f = float(beta)
        if math.isfinite(beta_f) and beta_f > 0.0:
            auto.append("kl")

    # LoRA / PEFT detection.
    lora_active = bool(ctx.get("lora_active", False))
    peft_active = bool(ctx.get("peft_active", False))
    inner_model = getattr(model, "model", model) if model is not None else None
    if inner_model is not None and bool(getattr(inner_model, "lora_is_enabled", False)):
        lora_active = True
    if trainer is not None and bool(getattr(trainer, "lora_is_enabled", False)):
        lora_active = True
    trainable_selector = getattr(args, "trainable_selector", None)
    if isinstance(trainable_selector, str) and "lora" in trainable_selector.lower():
        lora_active = True
    elif isinstance(trainable_selector, collections.abc.Iterable) and not isinstance(trainable_selector, (str, bytes)):
        for sel in trainable_selector:
            sel_text = getattr(sel, "__name__", str(sel)).lower()
            if "lora" in sel_text:
                lora_active = True
                break
    if getattr(args, "lora_config", None) is not None or bool(getattr(args, "use_lora", False)):
        lora_active = True
    peft_cfg = getattr(args, "peft_config", None) or getattr(trainer, "peft_config", None)
    if peft_cfg is not None:
        peft_active = True
        if "lora" in type(peft_cfg).__name__.lower() or "lora" in str(getattr(peft_cfg, "peft_type", "")).lower():
            lora_active = True
    if lora_active:
        auto.append("lora")
        auto.append("peft")
    elif peft_active:
        auto.append("peft")

    # Quantization detection.
    quant_method = (
        _extract_quantization_method(getattr(args, "quantization_mode", None))
        or _extract_quantization_method(getattr(args, "quantization_config", None))
        or _extract_quantization_method(
            getattr(getattr(inner_model, "config", None), "quantization_config", None)
            if inner_model is not None
            else None
        )
        or _extract_quantization_method(
            getattr(inner_model, "quantization_config", None) if inner_model is not None else None
        )
        or (ctx.get("quant_method") if isinstance(ctx.get("quant_method"), str) else None)
    )
    if quant_method:
        auto.append(f"quant:{quant_method}")

    # 3. Model family slug (prefer model.config.model_type, fall back to sanitized model_name).
    model_slug = (
        _extract_eyvan_model_slug(None, model=inner_model, prefer_model_type=True)
        or (ctx.get("model_slug") if isinstance(ctx.get("model_slug"), str) else None)
        or _extract_eyvan_model_slug(getattr(args, "model_name", None), prefer_model_type=False)
    )
    if model_slug:
        auto.append(model_slug)

    # 4. Hardware & topology tags (metadata-only, non-blocking).
    auto.extend(_detect_eyvan_hardware_tags())

    user_tags = getattr(args, "eyvan_track_tags", None) if include_user_tags and args is not None else None
    return _merge_eyvan_tags(auto, user_tags)


def _close_eyvan_run(run: tp.Any, status: str = "finished") -> None:
    """Flush, finish, and join the background writer thread of an Eyvan tracking Run safely."""
    if run is None:
        return
    resolved_status = status if status in _EYVAN_FINISH_STATUSES else "finished"
    try:
        if not getattr(run, "finished", False) and hasattr(run, "finish"):
            run.finish(status=resolved_status)
    except Exception:
        pass
    thread = getattr(run, "_thread", None)
    if isinstance(thread, threading.Thread) and thread.is_alive() and thread is not threading.current_thread():
        try:
            thread.join(timeout=1.0)
        except Exception:
            pass


def _sanitize_eyvan_slug(value: tp.Any, fallback: str = "easydel") -> str:
    """Sanitize a project or scope identifier to Eyvan's ``[A-Za-z0-9][A-Za-z0-9._-]{0,127}`` slug."""
    if value is None:
        return fallback
    slug = _EYVAN_SLUG_UNSAFE_RE.sub("-", str(value).strip()).strip("-._")
    slug = slug[:_EYVAN_MAX_KEY_LEN]
    if not slug or slug in {".", ".."} or not _EYVAN_SLUG_RE.fullmatch(slug):
        return fallback
    return slug


def _sanitize_eyvan_key(key: tp.Any) -> str:
    """Sanitize a metric/histogram/config key to Eyvan's ``[A-Za-z0-9_./-]{1,128}`` regex."""
    raw = str(key).strip()
    if _EYVAN_KEY_RE.fullmatch(raw):
        return raw
    sanitized = _EYVAN_KEY_UNSAFE_RE.sub("_", raw)
    sanitized = re.sub(r"/{2,}", "/", sanitized).strip("_")
    if not sanitized:
        return "metric"
    if len(sanitized) > _EYVAN_MAX_KEY_LEN:
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
        sanitized = f"{sanitized[: _EYVAN_MAX_KEY_LEN - 9]}_{digest}"
    if not _EYVAN_KEY_RE.fullmatch(sanitized):
        return "metric"
    return sanitized


def _sanitize_eyvan_tags(tags: tp.Iterable[tp.Any] | None) -> tuple[str, ...]:
    """Merge default EasyDeL tags with user tags, deduplicating within Eyvan's 64-tag / 64-char limits."""
    seen: set[str] = set()
    cleaned: list[str] = []
    for raw in (*_EYVAN_DEFAULT_TAGS, *(tags or ())):
        if raw is None:
            continue
        tag = str(raw).strip()[:_EYVAN_MAX_TAG_LEN].strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        cleaned.append(tag)
        if len(cleaned) >= _EYVAN_MAX_TAGS:
            break
    return tuple(cleaned)


def _eyvan_in_workload_env() -> bool:
    """Return True when running inside an Eyvan workload with tracking env configured."""
    return bool(
        os.environ.get(_EYVAN_TRACKING_URI_ENV, "").strip() or os.environ.get(_EYVAN_WORKLOAD_ID_ENV, "").strip()
    )


def _effective_eyvan_rank(env: tp.Mapping[str, str] | None = None) -> int:
    """Return the effective Eyvan distributed process rank matching ``eyvan.track.start``."""
    src = os.environ if env is None else env
    for key in ("EYVAN_DISTRIBUTED_PROCESS_ID", "JAX_PROCESS_ID"):
        raw = (src.get(key) or "").strip()
        if raw:
            try:
                return max(0, int(raw))
            except ValueError:
                pass
    try:
        return max(0, int(jax.process_index()))
    except Exception:
        return 0


def _resolve_eyvan_rank_environment() -> dict[str, str] | None:
    """Return an environment copy carrying ``JAX_PROCESS_ID`` for nonzero JAX ranks when unset.

    Never mutates ``os.environ``. Returns ``None`` when an explicit Eyvan/JAX rank env var is
    already present (non-empty) or when the current process is rank 0.
    """
    if os.environ.get("EYVAN_DISTRIBUTED_PROCESS_ID", "").strip() or os.environ.get("JAX_PROCESS_ID", "").strip():
        return None
    try:
        proc_idx = int(jax.process_index())
    except Exception:
        return None
    if proc_idx <= 0:
        return None
    env = dict(os.environ)
    env["JAX_PROCESS_ID"] = str(proc_idx)
    return env


def _is_eyvan_real_numeric_dtype(dtype: tp.Any) -> bool:
    """Return True for real numeric or boolean NumPy/JAX dtypes, explicitly excluding complex dtypes."""
    if dtype == jnp.bfloat16:
        return True
    return (np.issubdtype(dtype, np.number) and not np.issubdtype(dtype, np.complexfloating)) or dtype == np.bool_


def _is_eyvan_local_jax_array(value: tp.Any) -> bool:
    """Return True when a JAX array can be read locally without cross-host collectives."""
    if getattr(value, "is_fully_addressable", True):
        return True
    if getattr(value, "is_fully_replicated", False):
        sharding = getattr(value, "sharding", None)
        return bool(getattr(sharding, "has_addressable_devices", True))
    return False


def _safe_eyvan_device_get(value: tp.Any) -> tuple[bool, tp.Any]:
    """Read a host or locally-addressable/replicated JAX array without triggering cross-host collectives."""
    if isinstance(value, jax.Array):
        if not _is_eyvan_local_jax_array(value):
            return False, None
        return True, jax.device_get(value)
    return True, value


def _sanitize_eyvan_utf8_text(value: tp.Any, limit: int) -> str:
    """Coerce text to strict UTF-8 (replacing lone surrogates) and clip to ``limit`` characters."""
    return str(value).encode("utf-8", errors="replace").decode("utf-8")[:limit]


def _normalize_eyvan_meta_value(value: tp.Any) -> tp.Any:
    """Convert JAX/NumPy scalars and arrays into host-native values for Eyvan config/summary records."""
    if isinstance(value, jax.Array):
        ok, value = _safe_eyvan_device_get(value)
        if not ok:
            return None
    if isinstance(value, np.ndarray):
        arr = value.astype(np.float32) if value.dtype == jnp.bfloat16 else value
        if arr.ndim == 0 or arr.size == 1:
            return arr.reshape(()).item()
        return arr.tolist()
    if isinstance(value, np.generic):
        if value.dtype == jnp.bfloat16:
            return float(value)
        return value.item()
    return value


def _to_eyvan_scalar(value: tp.Any) -> tuple[bool, float]:
    """Extract a Python float from a scalar or 0-d/size-1 array, returning ``(ok, float_value)``."""
    if isinstance(value, bool | np.bool_):
        return True, 1.0 if bool(value) else 0.0
    if isinstance(value, int | float):
        return True, float(value)
    if isinstance(value, np.generic):
        if np.issubdtype(value.dtype, np.complexfloating):
            return False, 0.0
        if value.dtype == jnp.bfloat16:
            return True, float(value)
        item = value.item()
        if isinstance(item, bool):
            return True, 1.0 if item else 0.0
        if isinstance(item, int | float):
            return True, float(item)
        if _is_eyvan_real_numeric_dtype(value.dtype):
            return True, float(value)
        return False, 0.0
    if isinstance(value, np.ndarray | jax.Array):
        ok_read, host_value = _safe_eyvan_device_get(value)
        if not ok_read:
            return False, 0.0
        arr = np.asarray(host_value)
        if not _is_eyvan_real_numeric_dtype(arr.dtype):
            return False, 0.0
        if arr.ndim == 0 or arr.size == 1:
            if arr.dtype == jnp.bfloat16:
                arr = arr.astype(np.float32)
            item = arr.reshape(()).item()
            if isinstance(item, bool | np.bool_):
                return True, 1.0 if bool(item) else 0.0
            if isinstance(item, int | float | np.integer | np.floating):
                return True, float(item)
    return False, 0.0


def _sanitize_eyvan_axes(axes: tp.Mapping[str, tp.Any] | None) -> dict[str, float] | None:
    """Sanitize extra Eyvan x-axes to ``{valid_key: finite_float}`` or ``None`` when empty."""
    if not isinstance(axes, tp.Mapping):
        return None
    cleaned: dict[str, float] = {}
    for key, raw in axes.items():
        if raw is None:
            continue
        ok, val = _to_eyvan_scalar(raw)
        if ok and math.isfinite(val):
            cleaned[_sanitize_eyvan_key(key)] = float(val)
    return cleaned or None


def _coerce_eyvan_int(
    value: tp.Any,
    *,
    minimum: int | None = None,
    maximum: int = _EYVAN_MAX_INT64,
    clamp_min: bool = False,
    allow_float_round: bool = False,
) -> int | None:
    """Coerce a Python/NumPy/JAX scalar into a bounded integer, rejecting bools and complex dtypes."""
    if value is None or isinstance(value, bool | np.bool_ | str | bytes | bytearray):
        return None
    if isinstance(value, jax.Array):
        ok_read, value = _safe_eyvan_device_get(value)
        if not ok_read:
            return None
    if isinstance(value, np.ndarray):
        if not _is_eyvan_real_numeric_dtype(value.dtype) or value.dtype == np.bool_:
            return None
        if value.ndim != 0 and value.size != 1:
            return None
        if value.dtype == jnp.bfloat16:
            value = value.astype(np.float32)
        value = value.reshape(()).item()
    elif isinstance(value, np.generic):
        if not _is_eyvan_real_numeric_dtype(value.dtype) or value.dtype == np.bool_:
            return None
        value = float(value) if value.dtype == jnp.bfloat16 else value.item()

    if isinstance(value, bool | np.bool_):
        return None
    if isinstance(value, int | np.integer):
        result = int(value)
    elif isinstance(value, float | np.floating):
        fval = float(value)
        if not math.isfinite(fval):
            return None
        if allow_float_round:
            result = round(fval)
        elif fval.is_integer():
            result = int(fval)
        else:
            return None
    else:
        return None

    if minimum is not None and result < minimum:
        if clamp_min:
            result = minimum
        else:
            return None
    if result > maximum:
        result = maximum
    return result


def _sanitize_eyvan_text_label(value: tp.Any, limit: int) -> str | None:
    """Coerce an optional short label into a non-empty bounded UTF-8 string."""
    if value is None or isinstance(value, bool | np.bool_):
        return None
    text = _sanitize_eyvan_utf8_text(str(value).strip(), limit)
    if not text:
        return None
    return text


def _coerce_eyvan_sequence(value: tp.Any) -> list[tp.Any] | None:
    """Coerce a 1-D array or sequence (excluding strings/mappings/complex arrays) into a Python list."""
    if value is None or isinstance(value, str | bytes | bytearray | tp.Mapping):
        return None
    if isinstance(value, jax.Array):
        ok_read, value = _safe_eyvan_device_get(value)
        if not ok_read:
            return None
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.complexfloating):
            return None
        if value.dtype == jnp.bfloat16:
            value = value.astype(np.float32)
        return value.reshape(-1).tolist()
    if isinstance(value, list | tuple):
        return list(value)
    if isinstance(value, collections.abc.Iterable):
        return list(value)
    return None


def _normalize_eyvan_rollout_actions(
    raw_actions: list[tp.Any],
    raw_action_names: list[tp.Any] | None,
) -> tuple[list[int | None], list[str] | None]:
    """Normalize rollout actions to integer indices plus optional bounded ``action_names``."""
    sanitized_names: list[str] | None = None
    if raw_action_names is not None:
        sanitized_names = [
            (_sanitize_eyvan_utf8_text(str(name).strip(), _EYVAN_MAX_ACTION_NAME_LEN) or "action")
            for name in raw_action_names[:_EYVAN_MAX_ACTION_NAMES]
        ]

    int_actions: list[int | None] = []
    all_int_like = True
    for item in raw_actions:
        if item is None:
            int_actions.append(None)
            continue
        coerced = _coerce_eyvan_int(item, minimum=_EYVAN_MIN_INT64, maximum=_EYVAN_MAX_INT64)
        if coerced is None:
            all_int_like = False
            break
        int_actions.append(coerced)
    if all_int_like:
        return int_actions, sanitized_names

    palette: list[str] = list(sanitized_names) if sanitized_names is not None else []
    name_to_idx: dict[str, int] = {name: idx for idx, name in enumerate(palette)}
    indexed_actions: list[int | None] = []
    for item in raw_actions:
        if item is None:
            indexed_actions.append(None)
            continue
        label = _sanitize_eyvan_utf8_text(str(item).strip(), _EYVAN_MAX_ACTION_NAME_LEN) or "action"
        idx = name_to_idx.get(label)
        if idx is None:
            if len(palette) < _EYVAN_MAX_ACTION_NAMES - 1:
                idx = len(palette)
                palette.append(label)
                name_to_idx[label] = idx
            else:
                other_idx = name_to_idx.get("other")
                if other_idx is None:
                    if len(palette) < _EYVAN_MAX_ACTION_NAMES:
                        other_idx = len(palette)
                        palette.append("other")
                    else:
                        other_idx = _EYVAN_MAX_ACTION_NAMES - 1
                        palette[other_idx] = "other"
                    name_to_idx["other"] = other_idx
                idx = other_idx
        indexed_actions.append(idx)
    return indexed_actions, palette or None


_TB_HISTOGRAM_MAX_SAMPLES = 4096


def _histogram_tuple_to_samples(
    bin_counts: tp.Any,
    bin_edges: tp.Any,
    *,
    max_samples: int = _TB_HISTOGRAM_MAX_SAMPLES,
) -> np.ndarray:
    """Convert pre-binned ``(bin_counts, bin_edges)`` to a bounded host sample array.

    Proportionally downsamples large histograms (such as LLM weight tensors with
    hundreds of millions of elements) to at most ``max_samples`` bin-center samples
    while preserving non-empty bins whenever possible, avoiding host CPU OOM.
    """
    counts_arr = np.asarray(jax.device_get(bin_counts) if isinstance(bin_counts, jax.Array) else bin_counts).reshape(-1)
    edges_arr = np.asarray(jax.device_get(bin_edges) if isinstance(bin_edges, jax.Array) else bin_edges).reshape(-1)
    if edges_arr.dtype == jnp.bfloat16:
        edges_arr = edges_arr.astype(np.float32)

    n_bins = min(counts_arr.size, max(edges_arr.size - 1, 0))
    if n_bins == 0:
        return np.empty((0,), dtype=np.float64)

    counts = np.maximum(np.rint(counts_arr[:n_bins].astype(np.float64)).astype(np.int64), 0)
    total = int(counts.sum())
    if total <= 0:
        return np.empty((0,), dtype=np.float64)

    if total > max_samples:
        scale = max_samples / float(total)
        scaled = np.rint(counts * scale).astype(np.int64)
        nonzero = counts > 0
        num_nonzero = int(nonzero.sum())
        preserve_nonzero = num_nonzero <= max_samples
        if preserve_nonzero:
            scaled = np.where(nonzero & (scaled == 0), 1, scaled)
        excess = int(scaled.sum()) - max_samples
        if excess > 0:
            floor_per_bin = 1 if preserve_nonzero else 0
            for idx in np.argsort(scaled)[::-1]:
                if excess <= 0:
                    break
                min_keep = floor_per_bin if nonzero[idx] else 0
                reducible = int(scaled[idx]) - min_keep
                if reducible > 0:
                    take = min(reducible, excess)
                    scaled[idx] -= take
                    excess -= take
        if int(scaled.sum()) == 0:
            nz_idx = np.flatnonzero(nonzero)
            stride = max(1, len(nz_idx) // max_samples)
            scaled[nz_idx[::stride][:max_samples]] = 1
        counts = scaled

    bin_centers = 0.5 * (edges_arr[:n_bins].astype(np.float64) + edges_arr[1 : n_bins + 1].astype(np.float64))
    return np.repeat(bin_centers, counts)


def _normalize_partition_spec_entry(value: tp.Any) -> tp.Any:
    """Recursively convert lists to tuples within a partition spec entry.

    Args:
        value: A partition spec entry that may contain nested lists or tuples.

    Returns:
        The entry with all lists converted to tuples, preserving other types as-is.
    """
    if isinstance(value, list):
        return tuple(_normalize_partition_spec_entry(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_normalize_partition_spec_entry(item) for item in value)
    return value


_PARTITION_SPEC_REPR_PREFIXES: tuple[str, ...] = ("PartitionSpec(", "P(")


def _parse_partition_spec(value: tp.Any) -> PartitionSpec:
    """Parse a value into a JAX PartitionSpec.

    Handles PartitionSpec instances, None, string representations (both
    ``"PartitionSpec(...)"`` and the JAX 0.10+ ``"P(...)"`` repr), and
    list/tuple inputs.

    Args:
        value: The value to parse. Can be a PartitionSpec, None, a string
            representation, or a list/tuple of axis names.

    Returns:
        A PartitionSpec constructed from the parsed value.

    Raises:
        ValueError: If a string value cannot be parsed into a valid spec.
    """
    if isinstance(value, PartitionSpec):
        return value
    if value is None:
        return PartitionSpec()

    parsed = value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return PartitionSpec()
        prefix = next((p for p in _PARTITION_SPEC_REPR_PREFIXES if stripped.startswith(p)), None)
        try:
            if prefix is not None and stripped.endswith(")"):
                inner = stripped[len(prefix) : -1].strip()
                parsed = [] if not inner else ast.literal_eval(f"[{inner}]")
            else:
                parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid `step_partition_spec` value: {value!r}") from exc

    if isinstance(parsed, PartitionSpec):
        return parsed
    if isinstance(parsed, (list, tuple)):
        return PartitionSpec(*tuple(_normalize_partition_spec_entry(item) for item in parsed))
    return PartitionSpec(_normalize_partition_spec_entry(parsed))


_REMOVED_TRAINING_ARGS: dict[str, str] = {
    "use_esurge_generation": "trainers always generate through eSurge",
}
"""Fields that saved configs may still carry, with why each was removed; loading drops them."""


def _apply_training_args_legacy_aliases(data: dict[str, tp.Any]) -> dict[str, tp.Any]:
    """Rewrite deprecated field names in a config dict to their current equivalents.

    Translates ``quantization_block`` to ``quantization_group_size`` when the
    new name is not already present, then removes the deprecated key. Keys of
    removed fields (:data:`_REMOVED_TRAINING_ARGS`) are dropped with one
    warning each.

    Args:
        data: A configuration dictionary, typically from deserialized JSON.

    Returns:
        A new dictionary with deprecated aliases replaced by current field names.
    """
    data = dict(data)
    if "max_sequence_length" in data:
        legacy_max_seq_len = data.pop("max_sequence_length")
        if legacy_max_seq_len is not None:
            warnings.warn(
                "`max_sequence_length` is deprecated; use `max_length` instead.",
                FutureWarning,
                stacklevel=2,
            )
            if data.get("max_length") is None:
                data["max_length"] = legacy_max_seq_len
            elif data["max_length"] != legacy_max_seq_len:
                warnings.warn(
                    f"Both `max_length` ({data['max_length']}) and `max_sequence_length` ({legacy_max_seq_len}) "
                    "are set; ignoring `max_sequence_length`.",
                    FutureWarning,
                    stacklevel=2,
                )
    if "quantization_block" in data:
        legacy_quant_block = data.pop("quantization_block")
        if legacy_quant_block is not None and data.get("quantization_group_size") is None:
            data["quantization_group_size"] = legacy_quant_block
    for key, reason in _REMOVED_TRAINING_ARGS.items():
        if key in data:
            value = data.pop(key)
            logger.warning(f"Ignoring removed training argument `{key}={value!r}` from a saved config: {reason}.")
    return data


AVAILABLE_BACKENDS: list[str] = ["cpu", "gpu", "tpu", None]

_OWN_OPTIMIZER_FIELD = {"steps": "max_training_steps", "mu_dtype": "tx_mu_dtype"}


def _looks_like_dtype(value: tp.Any) -> bool:
    """Return whether ``value`` can be interpreted as a dtype (and is not ``None`` or a bool)."""
    if value is None or isinstance(value, bool):
        return False
    try:
        jnp.dtype(value)
    except (TypeError, ValueError):
        return False
    return True


def _same_optimizer_kwarg(own: tp.Any, extra: tp.Any) -> bool:
    """Return whether an extra optimizer kwarg repeats the ``TrainingArguments`` value exactly.

    Dtype-valued settings compare through ``jnp.dtype`` so ``jnp.float32`` and ``"float32"`` agree, and
    plain numbers compare by value so ``1`` and ``1.0`` agree.

    Args:
        own: The value held by the ``TrainingArguments`` field.
        extra: The value supplied through ``extra_optimizer_kwargs``.

    Returns:
        ``True`` when the two values are equivalent, ``False`` otherwise (including incomparable values).
    """
    try:
        if own is extra:
            return True
        if _looks_like_dtype(own) and _looks_like_dtype(extra):
            return jnp.dtype(own) == jnp.dtype(extra)
        numeric = (int, float)
        if (
            isinstance(own, numeric)
            and isinstance(extra, numeric)
            and not isinstance(own, bool)
            and not isinstance(extra, bool)
        ):
            return float(own) == float(extra)
        return bool(own == extra)
    except Exception:
        return False


@Registry.register("trainer-arguments", "base")
@dataclass
class TrainingArguments:
    """
    Comprehensive configuration class for training and evaluation.

    This class encapsulates all training hyperparameters, optimization settings,
    data loading configuration, logging preferences, and hardware-specific options.
    It provides a centralized way to manage the complex configuration required for
    distributed training of large models.

    The configuration covers:
    - Training hyperparameters (learning rate, batch size, epochs)
    - Optimization settings (optimizer, scheduler, gradient clipping)
    - Data loading (dataset configuration, batch collation)
    - Checkpointing (save frequency, checkpoint limits)
    - Logging (WandB, TensorBoard, metrics reporting)
    - Hardware configuration (sharding, precision, device placement)
    - Performance optimization (compilation, memory tracking)

    Example:
        >>> args = TrainingArguments(
        ...     learning_rate=1e-4,
        ...     num_train_epochs=3,
        ...     total_batch_size=32,
        ...     save_steps=1000,
        ...     use_wandb=True
        ... )
    """

    auto_shard_states: bool = field(
        default=True,
        metadata={"help": "Whether to automatically shard model states across devices."},
    )
    aux_loss_enabled: bool = field(
        default=False,
        metadata={"help": "Whether to enable the auxiliary loss."},
    )
    pose: PoSEConfig = field(
        default_factory=PoSEConfig,
        metadata={"help": "PoSE (positional skip-wise) position_ids augmentation; disabled by default."},
    )
    randomized_yarn: RandomizedYaRNConfig = field(
        default_factory=RandomizedYaRNConfig,
        metadata={
            "help": "Randomized YaRN position_ids augmentation (sorted random positions under a length "
            "curriculum); disabled by default and mutually exclusive with `pose`."
        },
    )
    process_encoder: ProcessEncoderConfig = field(
        default_factory=ProcessEncoderConfig,
        metadata={
            "help": (
                "Run a VLM vision tower out of band (frozen, vision-bearing rows only) and hand "
                "precomputed patch embeddings to the train step; disabled by default."
            )
        },
    )
    buckets: tp.Any = field(
        default=None,
        metadata={
            "help": (
                "Optional list of easydel.trainers.buckets.TrainingBucket configs. When set together with "
                "`bucket_rule`, the trainer builds one compiled step function and one dataloader per "
                "bucket and selects which to use each step via `bucket_rule.select(step)`. "
                "Each bucket is a model-config variant (e.g. attn_mechanism override) plus its own "
                "max_length/dataloader; all buckets share one parameter/optimizer tree. "
                "None (default) disables bucketing and the trainer behaves exactly as before."
            )
        },
    )
    bucket_rule: tp.Any = field(
        default=None,
        metadata={
            "help": (
                "easydel.trainers.buckets.BucketRule selecting the active bucket index per optimizer "
                "step (e.g. ModBucketRule(mod=5, on_bucket=0, off_bucket=1) for two buckets, or "
                "PatternBucketRule(pattern=[0, 0, 1, 1, 2]) for an arbitrary repeating N-way ratio). "
                "Required when `buckets` is set; ignored otherwise. Must be serializable "
                "(CallableBucketRule is not — to_dict runs on every checkpoint save)."
            )
        },
    )
    bucket_datasets: tp.Any = field(
        default=None,
        metadata={
            "help": (
                "Optional list of datasets/ShardedDataSource, one per bucket in `buckets`, used to "
                "build each bucket's dataloader. Length must match `buckets`. When None and `buckets` "
                "is set, all buckets fall back to the primary `dataset_train`."
            )
        },
    )
    backend: str | None = field(
        default=None,
        metadata={"help": "The JAX backend to use (e.g., 'cpu', 'gpu', 'tpu').  If None, JAX will choose."},
    )
    clip_grad: float | None = field(
        default=None,
        metadata={"help": "The value at which to clip the gradients."},
    )
    custom_scheduler: tp.Callable[[int], tp.Any] | None = field(
        default=None,
        metadata={"help": "A custom scheduler function that takes the current step as input."},
    )
    dataloader_num_workers: int | None = field(
        default=0,
        metadata={"help": "The number of workers to use for the dataloader."},
    )
    dataloader_pin_memory: bool | None = field(
        default=False,
        metadata={"help": "Whether to pin memory for the dataloader."},
    )
    dataloader_prefetch: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to overlap trainer-level batch fetching and collation with the current "
                "compiled train step using background host workers. JAX preprocessing and execution "
                "remain on the main training thread."
            )
        },
    )
    dataloader_prefetch_workers: int = field(
        default=1,
        metadata={
            "help": (
                "Parallel host producer threads for the trainer-level batch prefetcher (requires "
                "`dataloader_prefetch`). Fetching from the dataloader iterator stays serialized so "
                "the stream order is deterministic; collation fans out across workers. Raise this "
                "when producing one batch (fetch + collate) takes longer than the device step — "
                "e.g. large global batches or heavy collators. The data collator must be "
                "thread-safe when set > 1."
            )
        },
    )
    dataloader_prefetch_buffer_size: int | None = field(
        default=None,
        metadata={
            "help": (
                "Maximum collated batches queued ahead by the batch prefetcher (per bucket when "
                "training buckets are enabled). Defaults to `dataloader_prefetch_workers`. Each "
                "queued batch holds one full global batch in host memory."
            )
        },
    )
    dataloader_fast_forward_on_resume: bool = field(
        default=True,
        metadata={
            "help": (
                "Fast-forward/seek the train dataloader by the resumed global step at the start of "
                "training. Disable for streaming pack-on-the-fly sources where no cheap seek exists and "
                "a sequential skip would re-render/re-pack the entire consumed prefix; the stream then "
                "restarts from batch 0 (same behavior training buckets document on resume) while the "
                "optimizer/scheduler/step state still resumes correctly."
            )
        },
    )
    do_eval: bool = field(
        default=False,
        metadata={"help": "Whether to run evaluation during training."},
    )
    do_last_save: bool = field(
        default=True,
        metadata={"help": "Whether to save the model at the end of training."},
    )
    do_train: bool = field(
        default=True,
        metadata={"help": "Whether to run training."},
    )
    eval_batch_size: int | None = field(
        default=None,
        metadata={"help": "The batch size to use for evaluation."},
    )
    eval_config_overrides: dict | None = field(
        default=None,
        metadata={
            "help": (
                "Model-config overrides applied ONLY to the evaluation forward, e.g. "
                "{'attn_mechanism': 'vanilla'} to evaluate on the reference attention kernel while "
                "training on a sparse one. Like training buckets, this builds a separate structure-"
                "preserving graphdef via lazy_init and swaps it into the state for the compiled eval "
                "step; parameter/optimizer buffers are shared, never copied, and never mutated by "
                "evaluation. Only structure-preserving keys are allowed. None (default) evaluates "
                "with the training config."
            )
        },
    )
    evaluation_steps: int | None = field(
        default=None,
        metadata={"help": "Run evaluation every X steps."},
    )
    extra_optimizer_kwargs: dict | None = field(
        default_factory=dict,
        metadata={"help": "Additional keyword arguments to pass to the optimizer."},
    )
    frozen_parameters: str | None = field(
        default=None,
        metadata={"help": "A regex pattern of parameters to freeze (not train)."},
    )
    grain_shard_index: int | None = field(
        default=None,
        metadata={
            "help": "sharding index to be used for grain dataloaders in both train and eval steps. For the "
            "parquetra / ArrayRecord lanes, None groups hosts by the batch shards their devices hold (hosts that "
            "share a shard read the same rows); set both index and count to override."
        },
    )
    grain_shard_count: int | None = field(
        default=None,
        metadata={
            "help": "sharding count to be used for grain dataloaders in both train and eval steps (lanes: None = "
            "number of batch-shard host groups, see grain_shard_index)."
        },
    )
    data_lane_local_batch: bool = field(
        default=False,
        metadata={
            "help": "When True on the grain data lanes (`parquetra_train_datasets` / `arrayrecord_train_*`), each "
            "batch-shard host group loads and collates only its own local batch rows instead of the full global "
            "batch, and `place_batch` lifts each group's local batch into the global sharded array. Requires a "
            "`data_collator` sized to the local group batch (see `data_lanes.resolve_lane_batch_size`). Default "
            "False preserves the global-batch contract."
        },
    )
    arrayrecord_train_files: str | list[str] | None = field(
        default=None,
        metadata={
            "help": "Path/glob/list of msgpack-row `.array_record` files; selects the grain ArrayRecord "
            "training lane (one source, no mixing). `arrayrecord_train_datasets` takes precedence."
        },
    )
    arrayrecord_train_datasets: dict[str, str | list[str]] | None = field(
        default=None,
        metadata={
            "help": "Per-dataset ArrayRecord sets `{name: glob}` for weighted mixing on the grain ArrayRecord "
            "lane. A set directory may carry a `_MANIFEST.tsv` (`relpath<TAB>nbytes<TAB>nrows` per shard), "
            "which skips the per-file footer reads and meters reads by compressed bytes."
        },
    )
    arrayrecord_mixture_weights: dict[str, float] | None = field(
        default=None,
        metadata={"help": "Weights `{name: weight}` for `arrayrecord_train_datasets`; None = row-count proportional."},
    )
    arrayrecord_assistant_only: dict[str, bool] | None = field(
        default=None,
        metadata={
            "help": "Per-dataset assistant-only loss for the ArrayRecord lane, keyed like "
            "`arrayrecord_train_datasets` (partial dicts inherit `assistant_only_loss`; unknown keys raise)."
        },
    )
    arrayrecord_num_threads: int = field(
        default=16,
        metadata={"help": "grain ReadOptions.num_threads shared by the ArrayRecord and parquetra lanes."},
    )
    arrayrecord_prefetch_buffer: int = field(
        default=64,
        metadata={"help": "grain ReadOptions.prefetch_buffer_size shared by the ArrayRecord and parquetra lanes."},
    )
    arrayrecord_read_rate_limit_mb_per_sec: float | None = field(
        default=None,
        metadata={
            "help": "Per-process read-bandwidth cap (MB/s) for ArrayRecord reads, enforced by one token bucket "
            "shared by all read threads. None or 0 disables it."
        },
    )
    arrayrecord_read_rate_burst_seconds: float = field(
        default=0.25,
        metadata={"help": "Token-bucket capacity of the ArrayRecord read limiter, in seconds of the configured rate."},
    )
    parquetra_train_datasets: dict[str, str | list[str]] | None = field(
        default=None,
        metadata={
            "help": "Per-dataset parquet globs `{name: glob}` for the grain random-access parquet lane: row-level "
            "global shuffle over raw parquet, tokenized in the loader with the trainer's preprocess transform. "
            "Mutually exclusive with the ArrayRecord lane. Requires an explicit `data_collator`."
        },
    )
    parquetra_mixture_weights: dict[str, float] | None = field(
        default=None,
        metadata={"help": "Weights `{name: weight}` for `parquetra_train_datasets`; None = row-count proportional."},
    )
    parquetra_assistant_only: dict[str, bool] | None = field(
        default=None,
        metadata={
            "help": "Per-dataset assistant-only loss for the parquetra lanes, keyed like "
            "`parquetra_train_datasets` (partial dicts inherit `assistant_only_loss`; unknown keys raise)."
        },
    )
    parquetra_tokenizer_id: str | None = field(
        default=None,
        metadata={"help": "Tokenizer id for in-loop tokenization when the trainer has no processing_class."},
    )
    parquetra_fast_tokenizer: str | None = field(
        default=None,
        metadata={
            "help": "Optional `module:attr` import path of a factory `factory(hf_tokenizer) -> encode` where "
            "`encode(text) -> list[int]` reproduces `hf_tokenizer(text, add_special_tokens=False)`. Rows are "
            "rendered with the chat template and encoded by it; the first `parquetra_fast_tokenizer_verify_rows` "
            "rows of every worker are checked against the full transform and any divergence raises. Only valid "
            "without prompt masking."
        },
    )
    parquetra_fast_tokenizer_verify_rows: int = field(
        default=64,
        metadata={"help": "Rows per worker the fast tokenizer must reproduce exactly before it runs unchecked."},
    )
    parquetra_pack_seq_len: int = field(
        default=0,
        metadata={
            "help": "When > 0 the parquetra lane packs: rows are pulled from the shuffled, per-host-sharded, "
            "weight-mixed row stream until `total_batch_size` windows of this length are full (first fit, whole "
            "documents, never truncated). The data_collator receives one window-set per call as a flat row list in "
            "placement order. Resume is seed-shifted: step n starts the stream a fresh build at step_start_point=n "
            "would produce, without materializing skipped batches. Requires `max_training_steps`."
        },
    )
    parquetra_pack_embed_cap: int = field(
        default=0,
        metadata={"help": "Max image-embed rows per packed window-set (sum of `embed_n_tok`); 0 = uncapped."},
    )
    parquetra_pack_misfit_patience: int = field(
        default=10,
        metadata={
            "help": "The packer closes a window-set after this many rows failed to fit; they carry to the next set. "
            "Higher = denser windows, more host work per set."
        },
    )
    parquetra_mp_workers: int = field(
        default=0,
        metadata={
            "help": "grain mp_prefetch worker processes for the parquetra lanes (0 = thread prefetch only). Order is "
            "preserved, so seeked resume is unchanged."
        },
    )
    parquetra_mp_worker_cpu_only: bool = field(
        default=True,
        metadata={
            "help": "Configure each spawned parquetra worker process for CPU-only JAX before it unpickles the "
            "pipeline (drops TPU_/MEGASCALE_/CLOUD_TPU_ variables and sets JAX_PLATFORMS=cpu in the CHILD only), so "
            "a worker cannot try to claim accelerators held by the trainer. The trainer's environment is untouched."
        },
    )
    parquet_metadata_cache_dir: str | None = field(
        default=None,
        metadata={
            "help": "Root (local or gs://) for durable parquet footer / row-group index sidecars used by the "
            "parquetra lanes; restarts then skip the footer scan. Only process 0 writes remote sidecars."
        },
    )
    assistant_header_text: str | None = field(
        default=None,
        metadata={
            "help": "Text that opens an assistant turn, used by the per-dataset assistant-only span scan. None "
            "derives it from the tokenizer's chat template (its generation prompt)."
        },
    )
    assistant_end_text: str | None = field(
        default=None,
        metadata={
            "help": "Text that closes an assistant turn (its first token is the span terminator). None derives it "
            "from the chat template (falls back to eos_token)."
        },
    )
    zero_supervision_report_fraction: float | None = field(
        default=0.05,
        metadata={
            "help": "Warn when more than this fraction of a batch's documents end with zero supervised tokens under "
            "assistant-only masking. None disables the warning; the census metrics are always reported."
        },
    )
    zero_supervision_abort_fraction: float | None = field(
        default=0.5,
        metadata={
            "help": "Raise when more than this fraction of a batch's documents end with zero supervised tokens "
            "while per-dataset assistant-only masking is active (a misconfiguration signal). None disables it."
        },
    )
    zero_supervision_dump_limit: int = field(
        default=3,
        metadata={"help": "Zero-supervision documents logged (once per process) when the report threshold fires."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "The number of steps to accumulate gradients over."},
    )
    gradient_accumulation_mode: str = field(
        default="scan",
        metadata={
            "help": "'scan' (default): the microbatch loop is a lax.scan inside the compiled step. 'execution': "
            "each microbatch is its own grad-program execution into a donated fp32 accumulator, followed by one "
            "apply-program execution, so fwd/bwd and optimizer temporaries do not stack. Only trainers whose "
            "step supports it (distillation) accept 'execution'."
        },
    )
    ids_to_pop_from_dataset: list[str] | None = field(
        default_factory=list,
        metadata={"help": "A list of dataset columns to remove before training."},
    )
    is_fine_tuning: bool = field(
        default=True,
        metadata={"help": "Whether the training is a fine-tuning run."},
    )
    init_tx: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the training state."},
    )
    jax_distributed_config: dict | None = field(
        default=None,
        metadata={"help": "Configuration for JAX distributed training."},
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "The learning rate."},
    )
    learning_rate_end: float | None = field(
        default=None,
        metadata={"help": "The final learning rate for linear decay schedulers."},
    )
    param_lr_multipliers: dict[str, float] | None = field(
        default=None,
        metadata={
            "help": (
                "Optional mapping of regex/substring parameter path patterns to learning-rate "
                "multipliers (e.g. {'loop_gates': 10.0, 'loop_iter_embeddings': 10.0}). "
                "Applied statelessly at the end of the optimizer chain (zero extra optimizer state)."
            )
        },
    )
    no_weight_decay_patterns: list[str] | None = field(
        default=None,
        metadata={
            "help": (
                "Optional list of regex/substring parameter path patterns excluded from weight decay "
                "(e.g. ['loop_gates', 'loop_iter_embeddings'])."
            )
        },
    )
    optimizer_stochastic_rounding: bool = field(
        default=False,
        metadata={
            "help": (
                "When True, applies unbiased IEEE/bitwise stochastic rounding when casting narrow-float "
                "(e.g. bfloat16/float16) parameter updates inside the optimizer chain, preventing sub-ULP "
                "update stagnation without requiring float32 master weights."
            )
        },
    )
    optimizer_stochastic_rounding_seed: int = field(
        default=0,
        metadata={
            "help": "Base integer seed for optimizer stochastic rounding (folded with step count and leaf index)."
        },
    )
    log_all_workers: bool = field(
        default=False,
        metadata={"help": "Whether to log metrics from all workers in a distributed setup."},
    )
    log_grad_norms: bool = field(
        default=True,
        metadata={"help": "Whether to log gradient norms."},
    )
    report_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to report metrics to a logger."},
    )
    log_steps: int = field(
        default=10,
        metadata={"help": "Log metrics every X steps."},
    )
    loss_config: LossConfig | None = field(
        default=None,
        metadata={"help": "Configuration for the loss function."},
    )
    lmhead_chunksize: int | None = field(
        default=None,
        metadata={"help": "Optional token chunk size for LM-head projection during training/inference."},
    )
    quantization_mode: QuantizationMode | None = field(
        default=None,
        metadata={
            "help": (
                "Quantization mode for quantization-aware training (QAT). "
                f"Supported values: {STE_QAT_QUANTIZATION_MODES_DOC}. "
                "When set (or when a straight-through callable is provided), trainers can apply a straight-through "
                "estimator (STE) transform to `state.graphstate` for the forward pass without permanently modifying "
                "the stored parameters."
            )
        },
    )
    quantization_group_size: int | None = field(
        default=None,
        metadata={
            "help": (
                "Quantization group size for group-wise quantizers (e.g. NF4/AFFINE/MXFP*/NVFP*). "
                "If None, the default group size for the selected quantization mode is used."
            )
        },
    )
    quantization_bits: int | None = field(
        default=None,
        metadata={
            "help": (
                "Quantization bit-width for QAT/STE quantizers. For `affine`, supported bits are 2..8. "
                "For fixed-width formats (`nf4`, `mxfp4`, `nvfp4`, `mxfp8`, `nvfp8`), when provided it must "
                "match the format width."
            )
        },
    )
    tensor_straight_through: tp.Callable[[jax.Array], jax.Array] | None = field(
        default=None,
        metadata={
            "help": (
                "Per-tensor straight-through transform used for QAT (e.g. tensor -> quantize(mode) -> dequantize "
                "with identity gradients). If `straight_through_emulator` is not provided, this callable can be "
                "mapped over `state.graphstate` via `jax.tree_util.tree_map`."
            )
        },
    )
    straight_through_emulator: tp.Callable[[spx.State], spx.State] | None = field(
        default=None,
        metadata={
            "help": (
                "Callable that maps a graphstate pytree -> new graphstate pytree for the forward pass. "
                "If None, and quantization is enabled, trainers can default to applying "
                "`jax.tree_util.tree_map(tensor_straight_through, graphstate)`."
            )
        },
    )
    low_mem_usage: bool = field(
        default=True,
        metadata={"help": "Whether to try to minimize memory usage."},
    )
    max_evaluation_steps: int | None = field(
        default=None,
        metadata={"help": "Maximum number of evaluation steps."},
    )
    max_length: int | None = field(
        default=None,
        metadata={"help": "The maximum sequence length."},
    )
    max_sequence_length: InitVar[int | None] = None
    quantization_block: InitVar[int | None] = None
    max_training_steps: int | None = field(
        default=None,
        metadata={"help": "The maximum number of training steps."},
    )
    per_epoch_training_steps: int | None = field(
        default=None,
        metadata={"help": "The maximum number of training step per each epoch."},
    )
    per_epoch_evaluation_steps: int | None = field(
        default=None,
        metadata={"help": "The maximum number of evaluation step per each epoch."},
    )
    model_name: str | None = field(
        default=None,
        metadata={"help": "The name of the model."},
    )
    model_parameters: dict | None = field(
        default=None,
        metadata={"help": "Model architecture config"},
    )
    metrics_to_show_in_rich_pbar: list[str] | None = field(
        default=None,
        metadata={"help": "Metrics to display in the rich progress bar."},
    )
    mpmd_scheduler: MpMdSchedulers | None = field(
        default=None,
        metadata={
            "help": (
                "Optional MPMD pipeline schedule (``spx.runtime.schedules.Schedule`` instance, "
                "e.g. ``Std1F1B(microbatches=N)`` / ``GPipe(microbatches=N)``). "
                "Forwarded to ``spx.jit(schedule=...)`` when the model mesh is MPMD. "
                "If None, the trainer step runs as a forward-only marker-cluster MPMD program "
                "(no microbatch interleaving / 1F1B). Ignored on SPMD meshes."
            )
        },
    )
    generation_top_p: float | None = field(
        default=None,
        metadata={"help": "Default nucleus sampling threshold used for preview generations."},
    )
    generation_top_k: int | None = field(
        default=None,
        metadata={"help": "Default top-k sampling threshold used for preview generations."},
    )
    generation_presence_penalty: float | None = field(
        default=None,
        metadata={"help": "Default presence penalty used for preview generations."},
    )
    generation_frequency_penalty: float | None = field(
        default=None,
        metadata={"help": "Default frequency penalty used for preview generations."},
    )
    generation_repetition_penalty: float | None = field(
        default=None,
        metadata={"help": "Default repetition penalty used for preview generations."},
    )
    generation_temperature: float | None = field(
        default=None,
        metadata={"help": "Default sampling temperature for preview generations."},
    )
    generation_do_sample: bool | None = field(
        default=None,
        metadata={"help": "Whether to enable sampling when generating previews (auto-inferred when None)."},
    )
    generation_num_return_sequences: int | None = field(
        default=None,
        metadata={"help": "Number of completions to sample per prompt for preview generations."},
    )
    generation_max_new_tokens: int | None = field(
        default=None,
        metadata={"help": "Maximum number of newly generated tokens for previews."},
    )
    generation_shard_inputs: bool = field(
        default=True,
        metadata={"help": "Whether generation previews should reuse the model's sharding plan."},
    )
    generation_interval: int | None = field(
        default=None,
        metadata={"help": "Run preview generation every X training steps (disabled when None)."},
    )
    generation_prompts: list[str | dict[str, tp.Any]] = field(
        default_factory=list,
        metadata={"help": "Static prompts (text or tokenized dicts) to sample during preview generation."},
    )
    generation_use_train_prompts: bool = field(
        default=False,
        metadata={"help": "When True, sample additional prompts from the training dataset for previews."},
    )
    generation_num_prompts: int | None = field(
        default=1,
        metadata={"help": "Number of prompts to use per preview generation call."},
    )
    generation_dataset_prompt_field: str | None = field(
        default="prompt",
        metadata={"help": "Dataset field to treat as prompt text when sampling from the training set."},
    )
    generation_extra_kwargs: dict[str, tp.Any] | None = field(
        default=None,
        metadata={"help": "Additional kwargs forwarded to `model.generate` for previews."},
    )
    generation_config_overrides: dict[str, tp.Any] | None = field(
        default=None,
        metadata={"help": "Attribute overrides applied to the copied generation config for previews."},
    )
    generation_seed: int | None = field(
        default=None,
        metadata={"help": "Seed for preview prompt sampling (None uses a random seed)."},
    )
    generation_preview_print: bool = field(
        default=False,
        metadata={"help": "Whether to print preview generations to Terminal."},
    )
    generation_log_to_wandb: bool = field(
        default=True,
        metadata={"help": "Whether to log preview generations to WandB when available."},
    )
    log_training_generations_to_wandb: bool = field(
        default=True,
        metadata={"help": "Whether to log rollout/training generations to WandB when available."},
    )
    benchmark_interval: int | None = field(
        default=None,
        metadata={"help": "Run configured lm-eval benchmark suites every X training steps (disabled when None)."},
    )
    benchmarks: list[BenchmarkConfig] | BenchmarkConfig | None = field(
        default_factory=list,
        metadata={"help": "Benchmark suite definitions executed during training when benchmark_interval is set."},
    )
    esurge_use_tqdm: bool = field(
        default=True,
        metadata={"help": "Whether to use tqdm progress bars for eSurge generations."},
    )
    esurge_hbm_utilization: float | None = field(
        default=None,
        metadata={
            "help": "HBM utilization target (0.0-1.0) for eSurge KV-cache sizing. Sizing priority: "
            "(1) esurge_max_cache_tokens if set -> cap the pool to that many tokens; "
            "(2) else this hbm_utilization if set -> size the pool from that HBM fraction; "
            "(3) else (both None, default) -> the trainer auto-computes a budget from the generation batch "
            "(max_length x total_batch_size x num_generations) so it doesn't over-allocate a huge pool from a "
            "blanket HBM fraction against the model's full context window."
        },
    )
    esurge_max_num_seqs: int | None = field(
        default=None,
        metadata={
            "help": "Maximum number of concurrent sequences for eSurge batch processing. None uses eSurge default."
        },
    )
    esurge_max_num_seq_buckets: list[int] | None = field(
        default=None,
        metadata={"help": "Optional explicit sequence-capacity buckets for eSurge runner compilation."},
    )
    esurge_min_input_pad: int | None = field(
        default=None,
        metadata={"help": "Minimum input padding for eSurge sequences. None uses eSurge default."},
    )
    esurge_page_size: int | None = field(
        default=32,
        metadata={"help": "Page size for eSurge KV cache management. None uses eSurge default."},
    )
    esurge_max_cache_tokens: int | None = field(
        default=None,
        metadata={
            "help": "Hard ceiling on the total eSurge KV-cache token capacity. When set, the page pool is "
            "sized to min(esurge_hbm_utilization-derived, this), so a generous hbm_utilization can't allocate "
            "(e.g.) a 1M-token cache when the run only needs 64k. When None AND esurge_hbm_utilization is also "
            "None (both default), the trainer auto-computes the budget as max_length x total_batch_size x num_generations."
        },
    )
    esurge_silent_mode: bool = field(
        default=True,
        metadata={"help": "Silence eSurge info logs (engine start/stop/resume, cache events)."},
    )
    esurge_runner_verbose: bool = field(
        default=False,
        metadata={"help": "Enable verbose eSurge runner performance logs (including `[perf]` lines)."},
    )
    esurge_max_num_batched_tokens: int | None = field(
        default=None,
        metadata={"help": "Maximum number of tokens to batch together for eSurge generation. None uses eSurge default."},
    )
    esurge_enable_prefix_caching: bool | None = field(
        default=None,
        metadata={"help": "Enable/disable eSurge prefix caching. None keeps engine default behavior."},
    )
    esurge_data_parallelism_axis: str | None = field(
        default=None,
        metadata={"help": "Mesh axis name used by eSurge as the data-parallel KV-page axis (e.g. 'dp')."},
    )
    esurge_async_scheduling: bool | None = field(
        default=None,
        metadata={"help": "Enable/disable eSurge async scheduler token sampling. None keeps eSurge default behavior."},
    )
    esurge_overlap_execution: bool | None = field(
        default=None,
        metadata={
            "help": "Enable/disable eSurge double-buffered execution overlap. None keeps eSurge default behavior."
        },
    )
    async_mode: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable asynchronous background rollout generation with policy snapshots and staleness "
                "control on rollout-capable trainers (GRPO/GSPO/GFPO/RLOO/RLVR/DPPO/SDPO/SDFT/SSD/PAPO/"
                "NashMD/XPO/MiniLLM/AgenticMoshPit/NeMoGym/PPO/OnlineDPO/OnPolicyDistillation/GKD/SeqKD)."
            )
        },
    )
    max_inflight_tasks: int = field(
        default=32,
        metadata={"help": "Maximum number of queued/inflight async rollout tasks when `async_mode=True`."},
    )
    max_staleness: int = field(
        default=1,
        metadata={"help": "Maximum allowed optimizer-step lag between rollout generation and consumption."},
    )
    queue_maxsize: int = field(
        default=0,
        metadata={"help": "Optional cap on buffered async rollout tasks (0 uses `max_inflight_tasks`)."},
    )
    weight_sync_steps: int = field(
        default=1,
        metadata={"help": "Optimizer steps between rollout policy snapshot refreshes when `async_mode=True`."},
    )
    request_timeout: float = field(
        default=120.0,
        metadata={"help": "Timeout in seconds waiting for a background async rollout task."},
    )
    first_rollout_timeout: float = field(
        default=3600.0,
        metadata={"help": "Timeout in seconds for the very first async rollout (covers cold XLA compile)."},
    )
    heartbeat_stale_after_s: float = field(
        default=120.0,
        metadata={"help": "Timeout in seconds before an async rollout worker is treated as stalled."},
    )
    num_train_epochs: int = field(
        default=10,
        metadata={"help": "The number of training epochs."},
    )
    offload_dataset: bool = field(
        default=False,
        metadata={"help": "Whether to offload the dataset to CPU or disk."},
    )
    offload_device_type: str = field(
        default="cpu",
        metadata={"help": "The device type to offload the dataset to (cpu or disk)."},
    )
    offload_device_index: int = field(
        default=0,
        metadata={"help": "The device index to offload the dataset to."},
    )
    optimizer: AVAILABLE_OPTIMIZERS = field(
        default=EasyDeLOptimizers.ADAMW,
        metadata={"help": "The optimizer to use."},
    )
    trainable_selector: tp.Any = field(
        default="parameters",
        metadata={
            "help": (
                "Selector used when converting models to EasyDeLState. "
                "Use 'parameters' for base weights, 'lora' for adapter-only training, "
                "or a tuple like ('parameters', 'lora') in Python code for combined training."
            )
        },
    )
    performance_mode: bool = field(
        default=False,
        metadata={"help": "Whether to enable performance mode (e.g., XLA compilation)."},
    )
    profiler_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "Optional output directory for the JAX profiler trace. When set, the trainer calls "
                "`jax.profiler.start_trace(profiler_path)` immediately after the first training step "
                "completes (so the trace excludes the initial JIT-compile of step 1) and "
                "`jax.profiler.stop_trace()` when training finishes. Only rank 0 records the trace "
                "unless `profiler_log_all_workers` is True. Set to None to disable profiling."
            )
        },
    )
    xla_compiler_options: dict | None = field(
        default=None,
        metadata={
            "help": (
                "Per-compile XLA compiler option overrides forwarded to the train-step jit "
                "(e.g. {'xla_tpu_enable_latency_hiding_scheduler': 'true'}). Safer than env "
                "XLA_FLAGS: unknown options raise a catchable error at compile instead of "
                "crashing workers at libtpu load. Applies to the marker-based jit path "
                "(including bucketed steps); the scheduled MPMD path (mpmd_scheduler with a "
                "registered scheduled-loss adapter) does not forward it."
            )
        },
    )
    profiler_duration_ms: int | None = field(
        default=None,
        metadata={
            "help": (
                "Bound the JAX profiler collection window to this many milliseconds after the "
                "trace starts. Large programs (e.g. chunked-scan layers) emit millions of device "
                "op events per step and can overflow the profiler's 2GB XSpace protobuf cap when "
                "tracing whole steps; a bounded time slice keeps the trace valid. None traces "
                "until `profiler_stop_step`/end of training."
            )
        },
    )
    profiler_enable_hlo_proto: bool | None = field(
        default=None,
        metadata={
            "help": (
                "Override ProfileOptions.enable_hlo_proto (embedding compiled HLO protos in the "
                "trace). Set False to shrink traces for very large programs. None keeps the "
                "library default."
            )
        },
    )
    profiler_stop_step: int | None = field(
        default=None,
        metadata={
            "help": (
                "Stop the JAX profiler trace once the global step reaches this value, so a "
                "production run can capture a bounded few-step trace instead of tracing to the "
                "end of training. The trace starts after step 1 (see `profiler_path`); e.g. "
                "profiler_stop_step=6 captures steps 2-6. None traces until training finishes."
            )
        },
    )
    profiler_log_all_workers: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether all distributed workers should record JAX profiler traces. This is separate "
                "from `log_all_workers` so profiling TPU workers does not enable per-worker metrics, "
                "W&B, progress bars, or checkpoint logging."
            )
        },
    )
    profiler_include_compile: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to start the JAX profiler before the first training step. This captures "
                "the initial compile/HLO data needed by xprof op views, at the cost of including "
                "compile time in the trace."
            )
        },
    )
    profiler_host_tracer_level: int | None = field(
        default=None,
        metadata={
            "help": (
                "Optional `jax.profiler.ProfileOptions().host_tracer_level` override (1-4). Higher "
                "values capture more host-side detail at the cost of more overhead. Only used when "
                "`profiler_path` is set."
            )
        },
    )
    profiler_python_tracer_level: int | None = field(
        default=None,
        metadata={
            "help": (
                "Optional `jax.profiler.ProfileOptions().python_tracer_level` override for Python "
                "function-level profiling. Only used when `profiler_path` is set."
            )
        },
    )
    pruning_module: tp.Any = field(
        default=None,
        metadata={"help": "The pruning module to use."},
    )
    process_zero_is_admin: bool = field(
        default=True,
        metadata={"help": "Whether the process with rank 0 is the admin process."},
    )
    progress_bar_type: tp.Literal["tqdm", "rich", "json"] = field(
        default="tqdm",
        metadata={"help": "The type of progress bar to use."},
    )
    remove_ckpt_after_load: bool = field(
        default=False,
        metadata={"help": "Whether to remove the checkpoint after loading it."},
    )
    remove_unused_columns: bool = field(
        default=True,
        metadata={"help": "Whether to remove unused columns from the dataset."},
    )
    report_steps: int = field(
        default=5,
        metadata={"help": "Report metrics every X steps."},
    )
    save_interval_minutes: float | None = field(
        default=None,
        metadata={"help": "Interval Minutes to save the checkpoint for state."},
    )
    save_directory: str | None = field(
        default="EasyDeL-Checkpoints",
        metadata={"help": "The directory to save checkpoints to."},
    )
    save_optimizer_state: bool = field(
        default=True,
        metadata={"help": "Whether to save the optimizer state along with the model."},
    )
    merge_lora_before_save: bool = field(
        default=True,
        metadata={
            "help": (
                "When True, regular and final trainer checkpoints save a merged LoRA export at the checkpoint root. "
                "Merged LoRA checkpoints also keep an unmerged LoRA copy under `_resume_model` so "
                "`EasyDeLState.load_state()` can resume the original training graph with or without optimizer state."
            )
        },
    )
    merge_lora_before_tpu_preemption_save: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether TPU preemption checkpoints should also export merged LoRA weights. Defaults to False so "
                "emergency saves stay on the faster resume-focused path."
            )
        },
    )
    save_steps: int | None = field(
        default=None,
        metadata={"help": "Save a checkpoint every X steps."},
    )
    save_total_limit: int | None = field(
        default=None,
        metadata={
            "help": (
                "Maximum number of permanent checkpoints to keep. Older checkpoints are deleted. "
                "Note: Temporary checkpoints (time-based) are managed separately by Checkpointer."
            )
        },
    )
    save_tpu_preemption_checkpoints: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to force-save a standard EasyDeL checkpoint when JAX's TPU preemption "
                "sync service reaches a safe save step."
            )
        },
    )
    scheduler: AVAILABLE_SCHEDULERS = field(
        default=EasyDeLSchedulers.NONE,
        metadata={"help": "The scheduler to use."},
    )
    shuffle_seed_train: int = field(
        default=64871,
        metadata={"help": "seed used for trainer dataloader shuffle."},
    )
    sparsify_module: bool = field(
        default=False,
        metadata={"help": "Whether to sparsify the model."},
    )
    sparse_module_type: AVAILABLE_SPARSE_MODULE_TYPES = field(
        default="bcoo",
        metadata={"help": "The type of sparse module to use."},
    )
    state_apply_fn_kwarguments_to_model: dict | None = field(
        default=None,
        metadata={"help": "Keyword arguments to pass to the state apply function."},
    )
    step_partition_spec: PartitionSpec = field(
        default=PartitionSpec(("dp", "fsdp"), "sp"),
        metadata={"help": "The partition specification for the training step."},
    )
    step_start_point: int | None = field(
        default=None,
        metadata={"help": "The step to start training from (for resuming)."},
    )
    force_step_start_point: bool = field(
        default=False,
        metadata={
            "help": (
                "Force `step_start_point` onto a loaded nonzero state as well. "
                "When enabled, the trainer overwrites both `state.step` and optimizer/scheduler "
                "count leaves even if the incoming state was loaded manually rather than resumed "
                "through the trainer's auto-resume path."
            )
        },
    )
    resume_if_possible: bool = field(
        default=True,
        metadata={"help": "Automatically resume from the latest checkpoint if available."},
    )
    shuffle_train_dataset: bool = field(
        default=True,
        metadata={"help": "Whether to shuffle the training dataset."},
    )
    total_batch_size: int = field(
        default=32,
        metadata={"help": "The total batch size."},
    )
    training_time_limit: str | None = field(
        default=None,
        metadata={"help": "The maximum training time (e.g., '1d', '2h30m')."},
    )
    host_phase_log_path: str | None = field(
        default=None,
        kw_only=True,
        metadata={
            "help": (
                "Optional per-host timing log: a local append-only JSONL file, or a cloud prefix "
                "containing immutable per-step JSONL chunks."
            )
        },
    )
    train_on_inputs: bool = field(
        default=True,
        metadata={"help": "Whether to train on the input data."},
    )
    trainer_prefix: str | None = field(
        default=None,
        metadata={"help": "default prefix name for trainer."},
    )
    truncation_mode: tp.Literal["keep_end", "keep_start"] = field(
        default="keep_end",
        metadata={"help": "The truncation mode to use."},
    )
    tx_mu_dtype: jnp.dtype | None = field(
        default=None,
        metadata={"help": "The dtype to use for the `tx.mu` variable."},
    )
    track_memory: bool | float = field(
        default=False,
        metadata={"help": "Whether to track memory usage. If a float, it sets the memory tracking interval in seconds."},
    )
    sequence_packing: bool = field(
        default=False,
        metadata={
            "help": (
                "Pack multiple short sequences into fixed-length blocks of `max_length` tokens to cut padding "
                "waste. Documents stay isolated (block-diagonal attention + per-document position reset; "
                "linear-attention/SSM state is reset at boundaries where supported). Honored by "
                "single-stream supervised/offline trainers such as SFT, distillation, and GOLD; rollout, "
                "pairwise-preference, reward, and embedding trainers warn and ignore it."
            )
        },
    )
    use_data_collator: bool = field(
        default=True,
        metadata={"help": "Whether to use a data collator."},
    )
    use_grain: bool = field(
        default=True,
        metadata={"help": "Whether to use grain instead of `tensorflow-datasets`."},
    )
    use_wandb: bool = field(
        default=True,
        metadata={"help": "Whether to use Weights & Biases for logging."},
    )
    use_eyvan_track: bool | None = field(
        default=None,
        metadata={
            "help": (
                "Whether to log experiment metrics to Eyvan (`eyvan.track`). "
                "`None` (default) auto-enables whenever `eyvan` is installed and `performance_mode=False` "
                "(and not suppressed by test isolation via `set_eyvan_auto_track_enabled(False)`); "
                "`True` forces on; `False` disables completely. Disabled automatically when `performance_mode=True`."
            )
        },
    )
    eyvan_track_uri: str | None = field(
        default=None,
        metadata={
            "help": (
                "Explicit Eyvan tracking store URI (`gs://...`, `file://...`, or local directory). "
                "When omitted, respects `EYVAN_TRACKING_URI`, then queries `eyvan.storage.resolve_default_tracking_uri` "
                "(configured default monitoring bucket), and finally falls back to `~/.eyvan/tracking/default/<project>`."
            )
        },
    )
    eyvan_track_project: str | None = field(
        default=None,
        metadata={
            "help": (
                "Local Eyvan tracking project slug (`[A-Za-z0-9][A-Za-z0-9._-]{0,127}`) used when no "
                "explicit store URI is configured. Defaults from `EasyDeL-{trainer_prefix}` or `easydel`."
            )
        },
    )
    eyvan_track_name: str | None = field(
        default=None,
        metadata={
            "help": (
                "Human-readable run name for Eyvan tracking. Falls back to `wandb_name` or "
                "a cached W&B-style auto-generated name (e.g. `qwen3-grpo-amber-falcon-42`)."
            )
        },
    )
    eyvan_track_group: str | None = field(
        default=None,
        metadata={"help": "Optional sweep or experiment group name for Eyvan tracking."},
    )
    eyvan_track_tags: list[str] | None = field(
        default=None,
        metadata={
            "help": (
                "Optional list of tags for Eyvan tracking; merged with auto-detected framework, trainer, "
                "RL/feature, model, and hardware tags, deduplicated, and clamped to Eyvan limits."
            )
        },
    )
    eyvan_track_flush_interval_s: float = field(
        default=5.0,
        metadata={"help": "Background flush interval in seconds for `eyvan.track.start(...)`."},
    )
    eyvan_report_progress: bool = field(
        default=True,
        metadata={
            "help": (
                "Forward rank-0 scalar step progress (`step`, `total`, finite scalar metrics) to "
                "`eyvan.report_progress(...)`. Acts as a fast no-op when `EYVAN_PROGRESS_URL` is absent."
            )
        },
    )
    verbose: bool = field(
        default=True,
        metadata={"help": "Whether to print verbose output."},
    )
    wandb_entity: str | None = field(
        default=None,
        metadata={"help": "The Weights & Biases entity."},
    )
    wandb_name: str | None = field(
        default=None,
        metadata={"help": "The Weights & Biases run name."},
    )
    warmup_steps: int = field(
        default=0,
        metadata={"help": "The number of warmup steps for the learning rate scheduler."},
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "The weight decay value."},
    )
    watchers: list[LogWatcher] = field(
        default_factory=list,
        metadata={
            "help": (
                "List of LogWatcher instances that define custom per-parameter "
                "metrics to log at independent intervals during training."
            ),
        },
    )
    weight_distribution_pattern: str = field(
        default=r".*",
        metadata={"help": "The pattern to use to extract weight distribution."},
    )
    weight_distribution_log_steps: int = field(
        default=500,
        metadata={"help": "log weight distribution every X steps."},
    )

    _can_log_metrics: bool | None = None
    _im_a_hidden_checkpoint_manager: Checkpointer | None = None

    @property
    def can_log_metrics(self):
        """Whether this process should log metrics.

        Returns ``False`` when the process is not rank-zero and
        ``log_all_workers`` is disabled. Can be overridden via the setter.

        Returns:
            bool: True if metrics logging is enabled for this process.
        """
        if self._can_log_metrics is None:
            if not self.is_process_zero and not self.log_all_workers:
                return False
            return self.report_metrics
        return self._can_log_metrics

    @can_log_metrics.setter
    def can_log_metrics(self, val):
        """Override the automatic metrics-logging decision.

        Args:
            val: Explicit boolean to force metrics logging on or off,
                or None to revert to the default behavior.
        """
        self._can_log_metrics = val

    @property
    def offload_device(self):
        """Return the JAX device used for parameter offloading.

        Resolves the device from ``offload_device_type`` and
        ``offload_device_index``.

        Returns:
            jax.Device: The target offload device.
        """
        return jax.devices(self.offload_device_type)[self.offload_device_index]

    @property
    def training_time_seconds(self) -> int | None:
        """Convert ``training_time_limit`` to total seconds.

        Returns:
            The training time limit in seconds, or None if no limit is set.
        """
        if self.training_time_limit is None:
            return None
        return self._time_to_seconds(self.training_time_limit)

    @functools.cached_property
    def is_process_zero(self):
        """Whether the current process is the rank-zero (main) process.

        Returns:
            bool: True if ``jax.process_index()`` is 0.
        """
        return jax.process_index() == 0

    def _handle_deprecated_max_sequence_length(self, max_sequence_length: int | None) -> None:
        """Migrate the deprecated ``max_sequence_length`` value to ``max_length``.

        Emits a FutureWarning and sets ``max_length`` when it has not been
        explicitly provided. If both are set to conflicting values,
        ``max_length`` takes precedence and a warning is emitted.

        Args:
            max_sequence_length: The deprecated sequence-length value, or None
                if it was not supplied.
        """
        if max_sequence_length is None:
            return
        warnings.warn(
            "`max_sequence_length` is deprecated; use `max_length` instead.",
            FutureWarning,
            stacklevel=2,
        )
        if self.max_length is None:
            self.max_length = max_sequence_length
            return
        if self.max_length == max_sequence_length:
            return

        # If `max_length` is still at the class default, treat the deprecated alias as the user's intent.
        max_length_default = None
        for field_obj in fields(self):
            if field_obj.name == "max_length":
                max_length_default = field_obj.default
                break

        if self.max_length == max_length_default:
            self.max_length = max_sequence_length
            return

        warnings.warn(
            f"Both `max_length` ({self.max_length}) and `max_sequence_length` ({max_sequence_length}) are set; "
            "ignoring `max_sequence_length`.",
            FutureWarning,
            stacklevel=2,
        )

    def _handle_deprecated_quantization_block(self, quantization_block: int | None) -> None:
        """Migrate the deprecated ``quantization_block`` value to ``quantization_group_size``.

        Emits a FutureWarning and sets ``quantization_group_size`` when it has
        not been explicitly provided. If both are set to conflicting values,
        ``quantization_group_size`` takes precedence and a warning is emitted.

        Args:
            quantization_block: The deprecated quantization block size, or None
                if it was not supplied.
        """
        if quantization_block is None:
            return
        warnings.warn(
            "`quantization_block` is deprecated; use `quantization_group_size` instead.",
            FutureWarning,
            stacklevel=2,
        )
        if self.quantization_group_size is None:
            self.quantization_group_size = quantization_block
            return
        if self.quantization_group_size == quantization_block:
            return

        warnings.warn(
            f"Both `quantization_group_size` ({self.quantization_group_size}) and "
            f"`quantization_block` ({quantization_block}) are set; ignoring `quantization_block`.",
            FutureWarning,
            stacklevel=2,
        )

    def __post_init__(
        self,
        max_sequence_length: int | None,
        quantization_block: int | None,
    ):
        """
        Post-initialization setup and validation.

        This method is automatically called after dataclass initialization.
        It performs:
        1. Configuration validation to catch invalid settings early
        2. Distributed training setup based on backend and platform
        3. Optimizer and scheduler configuration
        4. Logging infrastructure initialization
        5. Variable normalization and default value setting

        Raises:
            ValueError: If configuration validation fails
            ValueError: If required conditions are not met
        """
        self._handle_deprecated_max_sequence_length(max_sequence_length)
        self._handle_deprecated_quantization_block(quantization_block)
        if self.max_length is None:
            self.max_length = 4096
        self._ensure_variables()
        self._validate_config()
        self._setup_distributed()
        self._setup_optimizer()
        self._setup_logging()

    def _validate_config(self):
        """
        Validate configuration settings for correctness and compatibility.

        This method checks:
        - Gradient accumulation steps are positive
        - Backend is supported (CPU, GPU, TPU)
        - Other configuration constraints are met

        Raises:
            ValueError: If gradient_accumulation_steps < 1 or backend is not recognized
        """
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("`gradient_accumulation_steps` can't be lower than 1.")
        if self.total_batch_size <= 0:
            raise ValueError("`total_batch_size` must be > 0.")
        if self.eval_batch_size <= 0:
            raise ValueError("`eval_batch_size` must be > 0.")
        if self.max_length <= 0:
            raise ValueError("`max_length` must be > 0.")
        if self.log_steps <= 0:
            raise ValueError("`log_steps` must be > 0.")
        if self.report_steps <= 0:
            raise ValueError("`report_steps` must be > 0.")
        if self.warmup_steps < 0:
            raise ValueError("`warmup_steps` must be >= 0.")
        if self.gradient_accumulation_mode not in ("scan", "execution"):
            raise ValueError(
                f"`gradient_accumulation_mode` must be 'scan' or 'execution', got {self.gradient_accumulation_mode!r}."
            )
        if self.gradient_accumulation_mode == "execution" and getattr(self, "mpmd_scheduler", None) is not None:
            raise ValueError("`gradient_accumulation_mode='execution'` is not supported with `mpmd_scheduler`.")
        if self.training_time_limit is not None:
            self._time_to_seconds(self.training_time_limit)

        if bool(self.async_mode):
            self.esurge_async_scheduling = True
            self.esurge_overlap_execution = True
            if self.max_inflight_tasks <= 0:
                raise ValueError("`max_inflight_tasks` must be positive.")
            if self.max_staleness < 0:
                raise ValueError("`max_staleness` must be non-negative.")
            if self.queue_maxsize < 0:
                raise ValueError("`queue_maxsize` must be non-negative.")
            if self.weight_sync_steps <= 0:
                raise ValueError("`weight_sync_steps` must be positive.")
            if self.request_timeout <= 0:
                raise ValueError("`request_timeout` must be positive.")
            if self.first_rollout_timeout <= 0:
                raise ValueError("`first_rollout_timeout` must be positive.")
            if self.heartbeat_stale_after_s <= 0:
                raise ValueError("`heartbeat_stale_after_s` must be positive.")

        self.sequence_packing = bool(self.sequence_packing)

        if self.backend not in AVAILABLE_BACKENDS:
            raise ValueError(f"Backend {self.backend} is not recognized. Available backends: {AVAILABLE_BACKENDS}")
        if self.lmhead_chunksize is not None and self.lmhead_chunksize <= 0:
            raise ValueError("`lmhead_chunksize` must be > 0 when specified.")
        if self.quantization_group_size is not None and self.quantization_group_size <= 0:
            raise ValueError("`quantization_group_size` must be > 0 when specified.")
        validate_quantization_bits(self.quantization_mode, self.quantization_bits)
        if not np.isfinite(float(self.eyvan_track_flush_interval_s)) or float(self.eyvan_track_flush_interval_s) <= 0:
            raise ValueError("`eyvan_track_flush_interval_s` must be a finite number > 0.")
        self._validate_data_lane_config()

    def _validate_data_lane_config(self) -> None:
        """Validate the grain data-lane and assistant-only supervision fields.

        Raises:
            ValueError: On out-of-range values or when both grain lanes are configured.
        """
        arrayrecord_set = bool(self.arrayrecord_train_files or self.arrayrecord_train_datasets)
        if arrayrecord_set and self.parquetra_train_datasets:
            raise ValueError(
                "`arrayrecord_train_*` and `parquetra_train_datasets` are mutually exclusive; set only one grain lane."
            )
        for name in ("zero_supervision_report_fraction", "zero_supervision_abort_fraction"):
            value = getattr(self, name)
            if value is not None and (
                not _is_real_number(value) or not np.isfinite(float(value)) or not (0.0 <= float(value) <= 1.0)
            ):
                raise ValueError(f"`{name}` must be None or within [0, 1], got {value!r}.")
        for name, min_val in (
            ("zero_supervision_dump_limit", 0),
            ("arrayrecord_num_threads", 1),
            ("arrayrecord_prefetch_buffer", 0),
            ("parquetra_pack_seq_len", 0),
            ("parquetra_pack_embed_cap", 0),
            ("parquetra_pack_misfit_patience", 1),
            ("parquetra_mp_workers", 0),
            ("parquetra_fast_tokenizer_verify_rows", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool | np.bool_):
                raise TypeError(f"`{name}` must be an int, got {type(value).__name__}: {value!r}")
            if int(value) < min_val:
                raise ValueError(f"`{name}` must be >= {min_val}.")
        rate = self.arrayrecord_read_rate_limit_mb_per_sec
        if rate is not None:
            if not _is_real_number(rate) or rate < 0 or not np.isfinite(rate):
                raise ValueError(
                    "`arrayrecord_read_rate_limit_mb_per_sec` must be None, 0, or a positive finite number."
                )
        burst = self.arrayrecord_read_rate_burst_seconds
        if not _is_real_number(burst) or not np.isfinite(float(burst)) or float(burst) <= 0:
            raise ValueError("`arrayrecord_read_rate_burst_seconds` must be a finite number > 0.")

    def _setup_distributed(self):
        """
        Configure JAX for distributed training.

        This method initializes the JAX distributed configuration which handles:
        - Multi-host setup for distributed training
        - Device mesh creation for model parallelism
        - Communication backend configuration
        - Process coordination setup

        The actual implementation is delegated to JaxDistributedConfig.
        """

        JaxDistributedConfig.initialize(self.jax_distributed_config)

    def _build_optimizer_kwargs(self) -> dict[str, tp.Any]:
        """Build optimizer keyword arguments from the current live fields."""
        extra_optimizer_kwargs = self.extra_optimizer_kwargs if self.extra_optimizer_kwargs is not None else {}
        own_optimizer_kwargs = {
            "learning_rate": self.learning_rate,
            "learning_rate_end": self.learning_rate_end,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "warmup_steps": self.warmup_steps,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "weight_decay": self.weight_decay,
            "steps": self.max_training_steps,
            "clip_grad": self.clip_grad,
            "mu_dtype": _parse_dtype(self.tx_mu_dtype),
        }
        # Extra kwargs must not silently override TrainingArguments fields; unset tx_mu_dtype may be filled.
        colliding = {
            key: (own_optimizer_kwargs[key], value)
            for key, value in extra_optimizer_kwargs.items()
            if key in own_optimizer_kwargs
            and not (key == "mu_dtype" and own_optimizer_kwargs[key] is None)
            and not _same_optimizer_kwarg(own_optimizer_kwargs[key], value)
        }
        if colliding:
            details = "; ".join(
                f"extra_optimizer_kwargs[{key!r}]={extra!r} would override "
                f"TrainingArguments.{_OWN_OPTIMIZER_FIELD.get(key, key)}={own!r}"
                for key, (own, extra) in sorted(colliding.items())
            )
            raise ValueError(
                "extra_optimizer_kwargs collides with TrainingArguments' own optimizer settings: "
                f"{details}. Set the TrainingArguments field directly, or pass optimizer-specific settings under "
                "a name that does not shadow a TrainingArguments field."
            )
        return {**own_optimizer_kwargs, **extra_optimizer_kwargs}

    def _setup_optimizer(self):
        """
        Configure optimizer and learning rate scheduler settings.

        This method prepares the optimizer_kwargs dictionary with all necessary
        parameters for optimizer creation, including:
        - Learning rate and schedule parameters
        - Weight decay and gradient clipping
        - Optimizer-specific settings from extra_optimizer_kwargs
        - Data type specifications for optimizer states

        The actual optimizer creation is deferred until training setup.
        """
        self.optimizer_kwargs = self._build_optimizer_kwargs()

    def _setup_logging(self):
        """
        Configure logging infrastructure for training monitoring.

        This method handles the setup of various logging backends and ensures
        compatibility with performance mode:
        - Disables WandB in performance mode to reduce overhead
        - Disables metrics reporting in performance mode
        - Configures appropriate logging levels and destinations

        Performance mode prioritizes speed over detailed monitoring.
        """
        if self.use_wandb and self.performance_mode:
            logger.info("WandB logging disabled due to performance mode")
            self.use_wandb = False
        if self.performance_mode and self.use_eyvan_track is not False:
            if self.use_eyvan_track is True:
                logger.info("Eyvan tracking disabled due to performance mode")
            self.use_eyvan_track = False
        if self.report_metrics and self.performance_mode:
            logger.info("Metrics reporting disabled due to performance mode")
            self._can_log_metrics = False

    def _ensure_variables(self):
        """
        Ensure all configuration variables are properly initialized.

        This method:
        - Converts string representations to proper types (e.g., PartitionSpec)
        - Sets default values for optional parameters
        - Initializes complex configuration objects (e.g., LossConfig)
        - Normalizes configuration values for consistency

        This ensures the configuration is ready for use by the trainer.
        """

        def _coerce_float(value: tp.Any) -> tp.Any:
            """Best-effort coercion of a value to a Python float.

            Handles numeric types, NumPy scalars, and whitespace-stripped
            strings. Returns the original value unchanged if conversion is
            not possible (e.g., non-numeric strings, booleans, None).

            Args:
                value: The value to coerce.

            Returns:
                A Python float if conversion succeeds, otherwise the
                original value.
            """
            if value is None:
                return None
            if isinstance(value, bool):
                return value
            if isinstance(value, (float, int, np.floating, np.integer)):
                return float(value)
            if isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    return value
                try:
                    return float(stripped)
                except ValueError:
                    return value
            return value

        def _coerce_int(value: tp.Any) -> tp.Any:
            """Best-effort coercion of a value to a Python int.

            Handles int/float types, NumPy scalars, and whitespace-stripped
            strings. Floats are converted only when they represent whole
            numbers. Returns the original value unchanged if conversion is
            not possible.

            Args:
                value: The value to coerce.

            Returns:
                A Python int if conversion succeeds, otherwise the original
                value.
            """
            if value is None:
                return None
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, np.integer)):
                return int(value)
            if isinstance(value, float) and value.is_integer():
                return int(value)
            if isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    return value
                try:
                    return int(stripped)
                except ValueError:
                    try:
                        casted = float(stripped)
                    except ValueError:
                        return value
                    return int(casted) if casted.is_integer() else value
            return value

        self.learning_rate = _coerce_float(self.learning_rate)
        self.learning_rate_end = _coerce_float(self.learning_rate_end)
        self.weight_decay = _coerce_float(self.weight_decay)
        self.clip_grad = _coerce_float(self.clip_grad)
        self.warmup_steps = _coerce_int(self.warmup_steps)
        self.gradient_accumulation_steps = _coerce_int(self.gradient_accumulation_steps)
        self.lmhead_chunksize = _coerce_int(self.lmhead_chunksize)
        self.quantization_group_size = _coerce_int(self.quantization_group_size)
        self.quantization_bits = _coerce_int(self.quantization_bits)
        self.total_batch_size = _coerce_int(self.total_batch_size)
        self.eval_batch_size = _coerce_int(self.eval_batch_size)
        self.max_length = _coerce_int(self.max_length)
        self.log_steps = _coerce_int(self.log_steps)
        self.report_steps = _coerce_int(self.report_steps)
        self.eyvan_track_flush_interval_s = _coerce_float(self.eyvan_track_flush_interval_s)
        self.tx_mu_dtype = _parse_dtype(self.tx_mu_dtype)

        if self.use_eyvan_track is not None and not isinstance(self.use_eyvan_track, bool):
            raise TypeError(
                "`use_eyvan_track` must be a bool or None, got "
                f"{type(self.use_eyvan_track).__name__}: {self.use_eyvan_track!r}"
            )
        for name in ("eyvan_track_uri", "eyvan_track_project", "eyvan_track_name", "eyvan_track_group"):
            value = getattr(self, name, None)
            if value is not None:
                if not isinstance(value, str):
                    raise TypeError(f"`{name}` must be a string when provided, got {type(value).__name__}: {value!r}")
                setattr(self, name, value.strip() or None)
        if self.eyvan_track_tags is not None:
            if isinstance(self.eyvan_track_tags, str):
                self.eyvan_track_tags = [self.eyvan_track_tags]
            elif isinstance(self.eyvan_track_tags, collections.abc.Iterable):
                self.eyvan_track_tags = [str(tag) for tag in self.eyvan_track_tags]
            else:
                raise TypeError(
                    "`eyvan_track_tags` must be a sequence of strings when provided, got "
                    f"{type(self.eyvan_track_tags).__name__}."
                )
        if not _is_real_number(self.eyvan_track_flush_interval_s):
            raise TypeError(
                "`eyvan_track_flush_interval_s` must be a number, got "
                f"{type(self.eyvan_track_flush_interval_s).__name__}: {self.eyvan_track_flush_interval_s!r}"
            )
        self.eyvan_report_progress = bool(self.eyvan_report_progress)

        for name in ("learning_rate", "weight_decay"):
            value = getattr(self, name, None)
            if not _is_real_number(value):
                raise TypeError(f"`{name}` must be a number, got {type(value).__name__}: {value!r}")

        if self.learning_rate_end is not None and not _is_real_number(self.learning_rate_end):
            raise TypeError(
                "`learning_rate_end` must be a number when provided, got "
                f"{type(self.learning_rate_end).__name__}: {self.learning_rate_end!r}"
            )

        if self.param_lr_multipliers is not None:
            if not isinstance(self.param_lr_multipliers, dict):
                raise TypeError(
                    "`param_lr_multipliers` must be a dict[str, float] when provided, got "
                    f"{type(self.param_lr_multipliers).__name__}."
                )
            self.param_lr_multipliers = {str(k): float(v) for k, v in self.param_lr_multipliers.items()}

        if self.no_weight_decay_patterns is not None:
            if isinstance(self.no_weight_decay_patterns, str):
                self.no_weight_decay_patterns = [self.no_weight_decay_patterns]
            else:
                self.no_weight_decay_patterns = [str(p) for p in self.no_weight_decay_patterns]

        self.optimizer_stochastic_rounding = bool(self.optimizer_stochastic_rounding)
        if not _is_real_int(self.optimizer_stochastic_rounding_seed):
            raise TypeError(
                "`optimizer_stochastic_rounding_seed` must be an int, got "
                f"{type(self.optimizer_stochastic_rounding_seed).__name__}: {self.optimizer_stochastic_rounding_seed!r}"
            )
        self.optimizer_stochastic_rounding_seed = int(self.optimizer_stochastic_rounding_seed)

        if self.clip_grad is not None and not _is_real_number(self.clip_grad):
            raise TypeError(
                f"`clip_grad` must be a number when provided, got {type(self.clip_grad).__name__}: {self.clip_grad!r}"
            )

        for name in (
            "warmup_steps",
            "gradient_accumulation_steps",
            "total_batch_size",
            "max_length",
            "log_steps",
            "report_steps",
        ):
            value = getattr(self, name, None)
            if not _is_real_int(value):
                raise TypeError(f"`{name}` must be an int, got {type(value).__name__}: {value!r}")

        for name in ("eval_batch_size", "lmhead_chunksize", "quantization_group_size", "quantization_bits"):
            value = getattr(self, name, None)
            if value is not None and not _is_real_int(value):
                raise TypeError(f"`{name}` must be an int when provided, got {type(value).__name__}: {value!r}")

        if isinstance(self.quantization_mode, str):
            quantization_mode = self.quantization_mode.strip().lower()
            self.quantization_mode = tp.cast(QuantizationMode | None, quantization_mode or None)
        if self.quantization_mode is not None and self.quantization_mode not in STE_QAT_QUANTIZATION_MODES:
            raise ValueError(
                f"`quantization_mode` must be one of {STE_QAT_QUANTIZATION_MODES_DOC}, got {self.quantization_mode!r}."
            )

        self.step_partition_spec = _parse_partition_spec(self.step_partition_spec)

        self.eval_batch_size = self.eval_batch_size if self.eval_batch_size is not None else self.total_batch_size
        if self.pose is None:
            self.pose = PoSEConfig()
        elif isinstance(self.pose, dict):
            self.pose = PoSEConfig.from_dict(self.pose)
        if self.randomized_yarn is None:
            self.randomized_yarn = RandomizedYaRNConfig()
        elif isinstance(self.randomized_yarn, dict):
            self.randomized_yarn = RandomizedYaRNConfig.from_dict(self.randomized_yarn)
        if self.pose.enabled and self.randomized_yarn.enabled:
            raise ValueError(
                "`pose.enabled` and `randomized_yarn.enabled` are mutually exclusive: both rewrite position_ids."
            )
        if self.process_encoder is None:
            self.process_encoder = ProcessEncoderConfig()
        elif isinstance(self.process_encoder, dict):
            self.process_encoder = ProcessEncoderConfig.from_dict(self.process_encoder)
        if self.loss_config is None:
            self.loss_config = LossConfig()
        if isinstance(self.loss_config, dict):
            self.loss_config = LossConfig(**self.loss_config)
        if self.generation_interval is not None and self.generation_interval <= 0:
            logger.warning("`generation_interval` must be positive; disabling preview generation.")
            self.generation_interval = None
        if self.benchmark_interval is not None and self.benchmark_interval <= 0:
            logger.warning("`benchmark_interval` must be positive; disabling benchmark hooks.")
            self.benchmark_interval = None
        if self.generation_num_prompts is not None:
            self.generation_num_prompts = max(1, int(self.generation_num_prompts))
        if self.benchmarks is None:
            self.benchmarks = []
        elif isinstance(self.benchmarks, collections.abc.Mapping):
            self.benchmarks = [dict(self.benchmarks)]
        elif isinstance(self.benchmarks, collections.abc.Sequence) and not isinstance(self.benchmarks, (str, bytes)):
            self.benchmarks = [dict(benchmark) for benchmark in self.benchmarks]
        else:
            raise TypeError("`benchmarks` must be a BenchmarkConfig or a sequence of BenchmarkConfig values.")
        normalize_benchmark_configs(self.benchmarks)  # early validation; resolved again at runtime
        if self.esurge_max_num_seq_buckets is not None:
            self.esurge_max_num_seq_buckets = [int(v) for v in self.esurge_max_num_seq_buckets]
        if self.esurge_data_parallelism_axis is not None:
            self.esurge_data_parallelism_axis = str(self.esurge_data_parallelism_axis).strip()
            if not self.esurge_data_parallelism_axis:
                raise ValueError("`esurge_data_parallelism_axis` must be a non-empty string when provided.")

        def _inherit_generation_attr(attr, fallback_name):
            """Copy a fallback attribute into a generation attribute if unset.

            Sets ``self.<attr>`` to ``self.<fallback_name>`` when the
            generation attribute is None and the fallback holds a
            non-None, non-False value.

            Args:
                attr: Name of the generation attribute to populate.
                fallback_name: Name of the legacy/alternative attribute to
                    read from.
            """
            current = getattr(self, attr, None)
            if current is None and hasattr(self, fallback_name):
                fallback_value = getattr(self, fallback_name)
                if fallback_value is not None and fallback_value is not False:
                    setattr(self, attr, fallback_value)

        _inherit_generation_attr("generation_num_return_sequences", "num_generations_per_prompt")
        _inherit_generation_attr("generation_num_return_sequences", "num_return_sequences")
        _inherit_generation_attr("generation_top_p", "top_p")
        _inherit_generation_attr("generation_top_k", "top_k")
        _inherit_generation_attr("generation_presence_penalty", "presence_penalty")
        _inherit_generation_attr("generation_frequency_penalty", "frequency_penalty")
        _inherit_generation_attr("generation_repetition_penalty", "repetition_penalty")
        _inherit_generation_attr("generation_temperature", "temperature_sampling")
        _inherit_generation_attr("generation_max_new_tokens", "max_completion_length")

        self.generation_top_p = _coerce_float(self.generation_top_p)
        self.generation_temperature = _coerce_float(self.generation_temperature)
        self.generation_top_k = _coerce_int(self.generation_top_k)
        self.generation_presence_penalty = _coerce_float(self.generation_presence_penalty)
        self.generation_frequency_penalty = _coerce_float(self.generation_frequency_penalty)
        self.generation_repetition_penalty = _coerce_float(self.generation_repetition_penalty)
        self.generation_num_return_sequences = _coerce_int(self.generation_num_return_sequences)
        self.generation_max_new_tokens = _coerce_int(self.generation_max_new_tokens)

        if self.generation_extra_kwargs is not None and not isinstance(
            self.generation_extra_kwargs, collections.abc.Mapping
        ):
            raise TypeError(
                "`generation_extra_kwargs` must be a mapping when provided, got "
                f"{type(self.generation_extra_kwargs).__name__}: {self.generation_extra_kwargs!r}"
            )
        generation_extra_kwargs = dict(self.generation_extra_kwargs or {})

        generation_kwargs = getattr(self, "generation_kwargs", None)
        if generation_kwargs is not None:
            if not isinstance(generation_kwargs, collections.abc.Mapping):
                raise TypeError(
                    "`generation_kwargs` must be a mapping when provided, got "
                    f"{type(generation_kwargs).__name__}: {generation_kwargs!r}"
                )
            generation_extra_kwargs.update(generation_kwargs)

        for attr_name, kwarg_name in (("min_p", "min_p"),):
            value = getattr(self, attr_name, None)
            if value is not None and kwarg_name not in generation_extra_kwargs:
                generation_extra_kwargs[kwarg_name] = value
        for attr_name, kwarg_name in (
            ("presence_penalty", "presence_penalty"),
            ("frequency_penalty", "frequency_penalty"),
            ("repetition_penalty", "repetition_penalty"),
        ):
            value = getattr(self, attr_name, None)
            generation_value = getattr(self, f"generation_{attr_name}", None)
            if value is not None and generation_value is None and kwarg_name not in generation_extra_kwargs:
                generation_extra_kwargs[kwarg_name] = value
        self.generation_extra_kwargs = generation_extra_kwargs or None

        if self.generation_do_sample is None:
            if any(
                getattr(self, name, None) is not None
                for name in ("generation_top_p", "generation_top_k", "generation_temperature")
            ):
                self.generation_do_sample = True
        if self.generation_do_sample is None and hasattr(self, "do_sample"):
            do_sample = getattr(self, "do_sample", None)
            if do_sample is not None:
                self.generation_do_sample = do_sample

    @staticmethod
    def _time_to_seconds(time_str: str) -> int:
        """
        Convert a human-readable time string to seconds.

        Supports various time formats:
        - Days: "1d", "2day", "3days"
        - Hours: "23h", "2hour", "3hours"
        - Minutes: "50min", "30m", "45minutes"
        - Seconds: "30s", "120sec", "60seconds"
        - Compound durations: "1d2h30m", "2 hours 30 minutes"

        Args:
            time_str: The time string to convert

        Returns:
            int: The equivalent time in seconds

        Raises:
            ValueError: If the time format is not recognized

        Example:
            >>> _time_to_seconds("2h30min")
            9000
            >>> _time_to_seconds("90min")
            5400
        """
        unit_to_seconds = {
            "d": 86400,
            "day": 86400,
            "days": 86400,
            "h": 3600,
            "hour": 3600,
            "hours": 3600,
            "min": 60,
            "m": 60,
            "minute": 60,
            "minutes": 60,
            "s": 1,
            "sec": 1,
            "second": 1,
            "seconds": 1,
        }
        units = "|".join(sorted(unit_to_seconds, key=len, reverse=True))
        token = rf"(\d+)\s*({units})"
        if not isinstance(time_str, str) or re.fullmatch(rf"\s*(?:{token}\s*)+", time_str.lower()) is None:
            raise ValueError("Invalid time format. Use a duration such as `1d`, `2h30m`, `50min`, or `30s`.")
        return sum(int(value) * unit_to_seconds[unit] for value, unit in re.findall(token, time_str.lower()))

    def get_path(self) -> ePathLike:
        """Get the path to the checkpoint directory.

        Returns:
            ePathLike: The path to the checkpoint directory, combining
                      save_directory and model_name.

        Note:
            Creates a model-specific subdirectory within the main save directory.
        """
        return ePath(self.save_directory) / self.model_name  # pyright: ignore[reportReturnType]

    def ensure_checkpoint_path(self):
        """Create the checkpoint directory if it doesn't exist.

        Ensures the full checkpoint path including parent directories
        exists on the filesystem. Safe to call multiple times.

        Note:
            Uses mkdir with parents=True to create full directory tree.
        """
        path = self.get_path()
        path.mkdir(parents=True, exist_ok=True)

    def get_tx_template(self, possible_max: int | None = None) -> GradientTransformation:
        """Get the optimizer transformation without a specific step count.

        Useful for creating a template optimizer when the total number of
        training steps is not yet known (e.g., during checkpoint loading).

        Args:
            possible_max: Maximum step count to use. Defaults to 2^63-1.

        Returns:
            GradientTransformation: The configured Optax optimizer chain.
        """
        if possible_max is None:
            possible_max = 2**63 - 1
        optimizer = self.get_optimizer_and_scheduler(possible_max)[0]
        if self.pruning_module is not None:
            optimizer = self.pruning_module.wrap_optax(optimizer)
        return optimizer

    def get_optimizer_and_scheduler(self, steps: int | None = None):
        """
        Create and return the optimizer and learning rate scheduler.

        This method uses the OptimizerFactory to create the configured optimizer
        and scheduler based on the training arguments. It handles:
        - Standard optimizers (AdamW, SGD, etc.)
        - Learning rate schedules (linear, cosine, constant)
        - Gradient clipping and weight decay
        - Custom optimizers and schedulers

        Args:
            steps: Optional override for the number of training steps.
                   If not provided, uses the value from self.optimizer_kwargs.

        Returns:
            tuple: A tuple of (optimizer, scheduler) where:
                - optimizer: Optax GradientTransformation
                - scheduler: Learning rate schedule function

        Note:
            The optimizer is an Optax transformation chain that may include
            gradient clipping, weight decay, and other transformations.
        """

        optimizer_kwargs = deepcopy(self._build_optimizer_kwargs())
        if steps is not None:
            optimizer_kwargs["steps"] = steps
        scheduler = optimizer_kwargs.pop("scheduler", None)
        if scheduler == "none":
            scheduler = None
        if scheduler == EasyDeLSchedulers.NONE:
            scheduler = None
        warmup_steps = optimizer_kwargs.pop("warmup_steps", None)
        scheduler_config = SchedulerConfig(
            scheduler_type=scheduler,
            steps=optimizer_kwargs.pop("steps"),
            learning_rate=optimizer_kwargs.pop("learning_rate"),
            learning_rate_end=optimizer_kwargs.pop("learning_rate_end"),
            warmup_steps=warmup_steps or None,
            exponent=optimizer_kwargs.pop("exponent", 1),
        )
        optimizer_kwargs.pop("gradient_accumulation_steps", 0)
        optimizer, scheduler = OptimizerFactory.create(
            optimizer_type=optimizer_kwargs.pop("optimizer"),
            scheduler_config=scheduler_config,
            clip_grad=optimizer_kwargs.pop("clip_grad"),
            weight_decay=optimizer_kwargs.pop("weight_decay"),
            no_weight_decay_patterns=self.no_weight_decay_patterns,
            param_lr_multipliers=self.param_lr_multipliers,
            stochastic_rounding=self.optimizer_stochastic_rounding,
            stochastic_rounding_seed=self.optimizer_stochastic_rounding_seed,
            custom_scheduler=self.custom_scheduler,
            **optimizer_kwargs,
        )
        return optimizer, scheduler

    def get_streaming_checkpointer(self):
        """Get the asynchronous checkpoint manager.

        Returns:
            AsyncCheckpointManager: The checkpoint manager for handling
                                   asynchronous model checkpointing.

        Note:
            The AsyncCheckpointManager allows non-blocking checkpoint
            saves during training, improving training efficiency.
        """
        if self._im_a_hidden_checkpoint_manager is not None:
            return self._im_a_hidden_checkpoint_manager

        save_directory = self._get_save_directory()
        if save_directory is None:
            raise ValueError("A save_directory and model_name are required to create a checkpoint manager.")
        self._im_a_hidden_checkpoint_manager = Checkpointer(
            base_path=str(save_directory),
            save_interval=self.get_save_interval_timedelta(),
            step_policies=self.get_checkpoint_policies(),
        )

        return self._im_a_hidden_checkpoint_manager

    def get_checkpoint_policies(self):
        """Convert save_steps configuration to CheckpointInterval policies.

        Returns:
            list[CheckpointInterval]: List of checkpoint interval policies.
                Returns empty list if save_steps is None.

        Example:
            >>> args = TrainingArguments(save_steps=1000)
            >>> policies = args.get_checkpoint_policies()
            >>> # Returns: [CheckpointInterval(every=1000, until=None)]
        """
        from spectrax.serialization import CheckpointInterval

        if self.save_steps is None:
            return []
        return [CheckpointInterval(every=self.save_steps, until=None)]

    def get_save_interval_timedelta(self):
        """Get time-based checkpoint save interval as timedelta.

        Returns:
            timedelta | None: Time interval for temporary checkpoints,
                or None if no time-based saving is configured.

        Note:
            Currently returns None. Can be extended to support
            time-based checkpoint saving via new TrainingArguments field.
        """
        if self.save_interval_minutes is not None:
            return datetime.timedelta(minutes=self.save_interval_minutes)
        return None

    @property
    def _tensorboard(self):
        """Lazy initialization of the SpectraX TensorBoard writer.

        Returns:
            SummaryWriter | None: TensorBoard writer instance, or None if the
                save path is not configured or metrics logging is disabled on
                this worker.
        """
        if not self.can_log_metrics:
            return None
        writer = self.__dict__.get("_tensorboard_writer")
        if writer is not None:
            return writer
        from easydel.utils.helpers import SummaryWriter

        path = self._get_save_directory(create=True)
        if path is None:
            return None
        writer = SummaryWriter(log_dir=str(path))
        self.__dict__["_tensorboard_writer"] = writer
        weakref.finalize(self, _close_summary_writer, writer)
        return writer

    def get_tensorboard(self) -> SummaryWriter | None:
        """Get the SpectraX TensorBoard SummaryWriter for logging metrics.

        Returns:
            SummaryWriter | None: The TensorBoard writer instance, or None if it
                is not available, not configured, or disabled for this worker.
        """
        if not self.can_log_metrics:
            return None
        try:
            return self._tensorboard
        except ModuleNotFoundError:
            return None

    def close_tensorboard(self) -> None:
        """Flush and close the cached TensorBoard SummaryWriter, if one was opened."""
        writer = self.__dict__.pop("_tensorboard_writer", None)
        _close_summary_writer(writer)

    @staticmethod
    def resolve_eyvan_storage_path(
        *parts: str,
        region: str | None = None,
        zone: str | None = None,
        create: bool | None = None,
        **kwargs: tp.Any,
    ) -> str:
        """Resolve a regional Eyvan GCS path (``gs://eyvan_<region>/...``) for checkpoints or datasets.

        Delegates lazily to :func:`easydel.utils.eyvan_helpers.storage_path` so
        calling this helper never pulls in ``eyvan.sdk``, ``fastapi``, or ``uvicorn``.

        Args:
            *parts: Relative object path segments appended under the regional bucket
                (e.g. ``args.resolve_eyvan_storage_path("checkpoints", "qwen3-8b")``).
            region: Optional GCP region override (auto-detected when omitted).
            zone: Optional GCP zone override (implies its parent region).
            create: Consent-by-default bucket creation policy (``True``, ``False``, or ``None``).
            **kwargs: Additional keyword arguments forwarded to
                :func:`easydel.utils.eyvan_helpers.storage_path`.

        Returns:
            Canonical ``gs://`` URI string.
        """
        from easydel.utils.eyvan_helpers import storage_path

        return storage_path(*parts, region=region, zone=zone, create=create, **kwargs)

    @property
    def eyvan_tracking_enabled(self) -> bool:
        """Whether Eyvan experiment tracking is active for this process.

        When ``use_eyvan_track`` is ``None`` (the default), auto-enables whenever
        ``eyvan.track`` is importable (unless suppressed by
        :func:`set_eyvan_auto_track_enabled`), or whenever ``eyvan_track_uri`` or
        Eyvan workload environment variables are present. Always respects
        ``performance_mode``, ``can_log_metrics``, explicit ``use_eyvan_track=False``,
        and the warn-once init-failure guard (``_eyvan_init_failed``).
        """
        if getattr(self, "performance_mode", False):
            return False
        if not self.can_log_metrics:
            return False
        if self.__dict__.get("_eyvan_init_failed", False):
            return False
        use_eyvan = getattr(self, "use_eyvan_track", None)
        if use_eyvan is False:
            return False
        if use_eyvan is True:
            return True
        uri = getattr(self, "eyvan_track_uri", None)
        if isinstance(uri, str) and uri.strip():
            return True
        if _eyvan_in_workload_env():
            return True
        if not is_eyvan_auto_track_enabled():
            return False
        return _import_eyvan_track() is not None

    def _bind_eyvan_runtime_context(
        self,
        *,
        trainer: tp.Any = None,
        model: tp.Any = None,
        env_name: tp.Any = None,
        extra_trainer_tags: tp.Iterable[str] | None = None,
        auto_run_name: str | None = None,
    ) -> None:
        """Record lightweight trainer/model/env metadata in ``__dict__`` without retaining object references."""
        ctx: dict[str, tp.Any] = dict(self.__dict__.get("_eyvan_runtime_context") or {})

        merged_extra: list[str] = list(ctx.get("extra_trainer_tags") or ())
        for raw_tag in extra_trainer_tags or ():
            slug = _normalize_eyvan_trainer_slug(raw_tag)
            if slug and slug not in merged_extra:
                merged_extra.append(slug)
        if merged_extra:
            ctx["extra_trainer_tags"] = tuple(merged_extra)

        if trainer is not None:
            trainer_slug = _normalize_eyvan_trainer_slug(type(trainer).__name__)
            if trainer_slug:
                ctx["trainer_slug"] = trainer_slug
            if bool(getattr(trainer, "lora_is_enabled", False)):
                ctx["lora_active"] = True
            peft_cfg = getattr(trainer, "peft_config", None)
            if peft_cfg is not None:
                ctx["peft_active"] = True
                if (
                    "lora" in type(peft_cfg).__name__.lower()
                    or "lora" in str(getattr(peft_cfg, "peft_type", "")).lower()
                ):
                    ctx["lora_active"] = True

        if model is not None:
            inner_model = getattr(model, "model", model)
            model_slug = _extract_eyvan_model_slug(None, model=inner_model, prefer_model_type=True)
            if model_slug:
                ctx["model_slug"] = model_slug
            if bool(getattr(inner_model, "lora_is_enabled", False)):
                ctx["lora_active"] = True
            qcfg = getattr(getattr(inner_model, "config", None), "quantization_config", None) or getattr(
                inner_model, "quantization_config", None
            )
            qmethod = _extract_quantization_method(qcfg)
            if qmethod:
                ctx["quant_method"] = qmethod

        if env_name is not None:
            env_slug = _extract_eyvan_env_slug(env_name)
            if env_slug:
                ctx["env_name"] = env_slug

        if ctx:
            self.__dict__["_eyvan_runtime_context"] = ctx

        if isinstance(auto_run_name, str) and auto_run_name.strip():
            self.__dict__["_eyvan_auto_run_name"] = auto_run_name.strip()

    def resolve_eyvan_run_name(
        self,
        *,
        trainer: tp.Any = None,
        model: tp.Any = None,
        trainer_slug: str | None = None,
    ) -> str:
        """Resolve and cache the human-readable Eyvan run name for this config instance."""
        return _resolve_eyvan_run_name(self, trainer=trainer, model=model, trainer_slug=trainer_slug)

    def get_eyvan_track(
        self,
        *,
        trainer: tp.Any = None,
        model: tp.Any = None,
        env_name: str | None = None,
    ):
        """Lazily initialize and return the cached Eyvan tracking :class:`~eyvan.track.Run`.

        Args:
            trainer: Optional trainer instance used to enrich auto-detected tags and run name.
            model: Optional model or state instance used to extract model family, LoRA, and
                quantization tags.
            env_name: Optional environment identifier used to emit ``env:<slug>`` tags.

        Returns:
            eyvan.track.Run | None: The active Eyvan run handle, or ``None`` if
                Eyvan tracking is disabled, ``eyvan`` is not installed, or run
                initialization fails.
        """
        if not self.eyvan_tracking_enabled:
            return None
        if self.__dict__.get("_eyvan_init_failed", False):
            return None
        cached = self.__dict__.get("_eyvan_track_run")
        if cached is not None and not getattr(cached, "finished", False):
            return cached

        with _EYVAN_INIT_LOCK:
            if self.__dict__.get("_eyvan_init_failed", False):
                return None
            cached = self.__dict__.get("_eyvan_track_run")
            if cached is not None:
                if not getattr(cached, "finished", False):
                    return cached
                self.__dict__.pop("_eyvan_track_run", None)

            eyvan_track = _import_eyvan_track()
            if eyvan_track is None:
                if getattr(self, "use_eyvan_track", None) is True and not self.__dict__.get(
                    "_eyvan_track_missing_warned"
                ):
                    self.__dict__["_eyvan_track_missing_warned"] = True
                    warnings.warn(
                        "You have set `use_eyvan_track=True` but `eyvan` is not installed.",
                        stacklevel=2,
                    )
                return None

            prefix = (getattr(self, "trainer_prefix", None) or "").strip()
            default_project = f"EasyDeL-{prefix}" if prefix else "easydel"
            project = _sanitize_eyvan_slug(
                getattr(self, "eyvan_track_project", None) or default_project,
                fallback="easydel",
            )

            run_name = _resolve_eyvan_run_name(self, trainer=trainer, model=model)

            raw_group = getattr(self, "eyvan_track_group", None)
            group = str(raw_group).strip()[:_EYVAN_MAX_NAME_LEN].strip() or None if raw_group is not None else None
            tags = _auto_detect_eyvan_tags(self, trainer=trainer, model=model, env_name=env_name)

            raw_uri = getattr(self, "eyvan_track_uri", None)
            uri = _resolve_eyvan_default_tracking_uri(
                explicit_uri=raw_uri if isinstance(raw_uri, str) else None,
                project=project,
                org="default",
            )

            flush_interval_s = float(getattr(self, "eyvan_track_flush_interval_s", 5.0))
            if not np.isfinite(flush_interval_s) or flush_interval_s <= 0:
                flush_interval_s = 5.0

            ranks = "all" if getattr(self, "log_all_workers", False) else "zero"
            environment = _resolve_eyvan_rank_environment()
            effective_rank = _effective_eyvan_rank(environment)
            will_write = ranks == "all" or effective_rank == 0
            try:
                run_config = self.to_dict()
            except Exception:
                run_config = None

            try:
                if will_write:
                    _ensure_local_eyvan_store_writable(uri, project=project, org="default")
                run = eyvan_track.start(
                    name=run_name,
                    project=project,
                    group=group,
                    tags=tags,
                    config=run_config,
                    uri=uri,
                    ranks=ranks,
                    episode_ranks=ranks,
                    flush_interval_s=flush_interval_s,
                    environment=environment,
                )
            except Exception as exc:
                self.__dict__["_eyvan_init_failed"] = True
                warnings.warn(
                    f"Failed to initialize Eyvan tracking ({type(exc).__name__}: {exc}); "
                    "disabling Eyvan tracking for this run.",
                    UserWarning,
                    stacklevel=2,
                )
                return None

            self.__dict__["_eyvan_track_run"] = run
            old_finalizer = self.__dict__.pop("_eyvan_track_finalizer", None)
            if old_finalizer is not None:
                old_finalizer.detach()
            self.__dict__["_eyvan_track_finalizer"] = weakref.finalize(self, _close_eyvan_run, run)
            if effective_rank == 0 and getattr(run, "active", True):
                run_id = getattr(run, "id", None)
                store_root = getattr(run, "store_root", None)
                resolved_name = getattr(run, "name", run_name)
                logger.info(
                    f"Eyvan tracking active: name={resolved_name} run_id={run_id} "
                    f"store={store_root} (view with `eyvan dashboard`)"
                )
            return run

    def close_eyvan_track(self, status: str = "finished") -> None:
        """Idempotently flush, finish, and clear the cached Eyvan tracking run."""
        with _EYVAN_INIT_LOCK:
            finalizer = self.__dict__.pop("_eyvan_track_finalizer", None)
            if finalizer is not None:
                finalizer.detach()
            run = self.__dict__.pop("_eyvan_track_run", None)
        _close_eyvan_run(run, status=status)

    def get_wandb_init(self):
        """
        Initialize Weights & Biases for experiment tracking.

        This method creates a new WandB run with appropriate configuration:
        - Project name based on model name and optional prefix
        - Run name with timestamp if not specified
        - Configuration dictionary from training arguments
        - Standard tags for EasyDeL experiments

        The method handles process-level initialization, ensuring only the
        main process creates the WandB run in distributed settings.

        Returns:
            wandb.Run | None: The initialized WandB run object, or None if:
                - WandB is not installed
                - use_wandb is False
                - Not the main process and log_all_workers is False

        Note:
            WandB initialization is skipped in performance mode to reduce overhead.
        """
        if self.can_log_metrics:
            if not self.use_wandb:
                return None
            if wandb is None:
                warnings.warn(
                    "you have used `use_wandb=True` but you haven't install wandb.",
                    stacklevel=1,
                )
                return None
            wandb_name = self.wandb_name
            prefix = (self.trainer_prefix or "").strip()
            project_suffix = f"-{prefix}" if prefix else ""
            resolved_model_name = self.model_name if isinstance(self.model_name, str) and self.model_name else "model"
            safe_model_name = resolved_model_name.lower().replace("/", "-")
            if wandb_name is None:
                wandb_name = self.build_wandb_run_name(safe_model_name)
            else:
                wandb_name = wandb_name + "-" + safe_model_name
            return wandb.init(
                project=f"EasyDeL{project_suffix}",
                config=self.to_dict(),
                save_code=True,
                name=wandb_name,
                tags=["EasyDeL", "Jax", "Train", "LLM", "VLM"],
                entity=self.wandb_entity,
            )
        return None

    @staticmethod
    def _wandb_token(value: tp.Any, fallback: str = "na") -> str:
        """Normalize arbitrary config values into safe W&B name tokens."""
        if value is None:
            return fallback
        if hasattr(value, "value"):
            value = value.value
        token = str(value).strip().lower()
        token = token.replace("/", "-").replace(" ", "")
        token = re.sub(r"[^a-z0-9._-]", "-", token)
        token = re.sub(r"-{2,}", "-", token).strip("-")
        return token or fallback

    @staticmethod
    def _wandb_float_token(value: float | int | None, fallback: str = "none") -> str:
        """Format a numeric value as a compact string token for W&B run names.

        Args:
            value: The numeric value to format, or None.
            fallback: String to return when ``value`` is None.

        Returns:
            A compact string representation (e.g., ``"5e-05"``) or the
            fallback string.
        """
        if value is None:
            return fallback
        number = float(value)
        token = f"{number:g}".replace("+", "")
        return token

    def build_wandb_run_name(
        self,
        model_name: str,
        size_in_billion: float | None = None,
    ) -> str:
        """Build a structured default W&B run name.

        Format (when size_in_billion is provided):
            ``{model_name}-{size}b-b{batch}-lr{lr}-tx-{optimizer}``
        Format (when size_in_billion is None):
            ``{model_name}-b{batch}-lr{lr}-tx-{optimizer}``
        """
        batch = int(self.total_batch_size)
        lr = self._wandb_float_token(self.learning_rate, fallback="none")
        optimizer = self._wandb_token(self.optimizer, fallback="NA")
        if size_in_billion is not None:
            size = self._wandb_float_token(size_in_billion)
            return f"{model_name}-{size}b-b{batch}-lr{lr}-tx-{optimizer}"
        return f"{model_name}-b{batch}-lr{lr}-tx-{optimizer}"

    def ensure_training_time_limit(self, time_passed):
        """Check if training has exceeded the configured time limit.

        Args:
            time_passed: Elapsed training time in seconds.

        Raises:
            EasyDeLTimerError: If elapsed time exceeds training_time_limit.
        """
        if self.training_time_limit is not None and time_passed > self._time_to_seconds(self.training_time_limit):
            raise EasyDeLTimerError("Time Out")

    def log_metrics(
        self,
        metrics: MetricsType,
        step: int,
        log_as: tp.Literal["summary", "config"] | None = None,
        *,
        axes: tp.Mapping[str, float | int] | None = None,
    ):
        """
        Log metrics to configured logging backends.

        This method handles logging to multiple backends (WandB, TensorBoard,
        Eyvan) and supports various metric types including scalars, histograms,
        and distributions. It automatically filters and formats metrics for each
        backend's requirements.

        Args:
            metrics: Dictionary of metric names to values. Values can be:
                - Scalars (float, int)
                - Arrays (numpy, JAX arrays)
                - Histograms (tuple of bin_counts and bin_edges)
                - Tensors (automatically converted)
            step: The current training/evaluation step
            log_as: Special logging mode:
                - None: Regular step-based logging
                - "summary": Log as final summary
                - "config": Log as configuration
            axes: Optional extra x-axis values (e.g. ``{"tokens": ..., "wall_s": ...}``)
                forwarded to Eyvan tracking.

        Note:
            - Metrics are automatically filtered for None values
            - Array metrics are converted to appropriate formats
            - Gradient norm metrics are restructured for clarity
            - Logging only occurs if can_log_metrics is True
        """

        if self.can_log_metrics:
            filtered_metrics = {k: v for k, v in metrics.items() if v is not None}
            metrics = {self._restructure_metric_name(k): get_safe_arr(v) for k, v in filtered_metrics.items()}
            self._log_to_wandb(metrics, step, log_as)
            self._log_to_tensorboard(metrics, step, log_as)
            self._log_to_eyvan(metrics, step, log_as, axes=axes)

    def _restructure_metric_name(self, metric_name: str) -> str:
        """
        Restructures the metric name for logging.

        Args:
          metric_name (str): The original metric name.

        Returns:
          str: The restructured metric name.
        """
        if metric_name.startswith("train/grad_norm/"):
            return metric_name.replace("train/grad_norm/", "grad_norm/")
        return metric_name

    def log_weight_distribution(self, state, step: int):
        """
        Log weight distribution histograms and statistics.

        This method computes and logs detailed statistics about model weights,
        including histograms and summary statistics (mean, std, min, max).
        It's useful for monitoring training stability and detecting issues
        like gradient explosion or vanishing.

        Args:
            state: Model state containing parameters to analyze
            step: Current training step for logging

        Note:
            - Only logs at intervals defined by weight_distribution_log_steps
            - Uses weight_distribution_pattern to filter parameters
            - Computes statistics across all processes in distributed training
            - Logs both histograms and scalar statistics for each parameter
        """
        if self.weight_distribution_log_steps <= 0:
            return
        try:
            if (step % self.weight_distribution_log_steps) == 0:
                stats = compute_weight_stats(state.graphstate, self.weight_distribution_pattern)
                stats = jax.device_get(stats)

                metrics = {}
                for key, histogram in stats.items():
                    try:
                        if isinstance(histogram, MetricsHistogram):
                            path = key.replace("/", ".")
                            metrics[f"weights-histogram/{path}"] = (
                                histogram.bin_counts.tolist(),
                                histogram.bin_edges.tolist(),
                            )

                            base_path = key.removesuffix("/histogram").replace("/", ".")
                            metrics[f"weights-information/{base_path}/mean"] = float(
                                np.asarray(histogram.mean).reshape(()).item()
                            )
                            metrics[f"weights-information/{base_path}/std"] = float(
                                np.asarray(histogram.std).reshape(()).item()
                            )
                            metrics[f"weights-information/{base_path}/min"] = float(
                                np.asarray(histogram.min).reshape(()).item()
                            )
                            metrics[f"weights-information/{base_path}/max"] = float(
                                np.asarray(histogram.max).reshape(()).item()
                            )
                        else:
                            path = key.replace("/", ".")
                            metrics[f"weights-information/{path}"] = histogram
                    except Exception as e:
                        traceback.print_exc()
                        raise e

                self.log_metrics(metrics, step)

        except Exception as e:
            logger.warning(f"Failed to log weight distribution {e}...")

    def log_watchers(self, state, step: int):
        """Run all registered ``LogWatcher`` instances and log their metrics.

        Each watcher is only evaluated when ``step`` is a multiple of its
        ``interval``. Results are sent through the standard
        ``log_metrics`` pipeline (wandb / TensorBoard).

        Args:
            state: Model state whose ``graphstate`` contains the parameters.
            step: Current training step.
        """
        if not self.watchers:
            return
        try:
            metrics = run_watchers(self.watchers, state.graphstate, step)
            if metrics:
                metrics = jax.device_get(metrics)
                self.log_metrics(metrics, step)
        except Exception as e:
            logger.warning(f"Failed to log watchers: {e}...")

    def _log_to_wandb(
        self,
        metrics: dict[str, tp.Any],
        step: int,
        log_as: tp.Literal["summary", "config"] | None = None,
    ):
        """
        Log metrics to Weights & Biases (wandb).

        This method processes the given metrics and logs them to wandb if it's enabled and properly initialized.

        Args:
            metrics: A dictionary of metrics to log. Keys are metric names, values are the metric values.
            step: The current step or iteration number.
            log_as: How to log the metrics
                (None for regular logging, "summary" for wandb.summary, "config" for wandb.config)
        """
        if self.use_wandb and wandb is not None:
            if log_as == "summary":
                wandb.summary.update(metrics)
            elif log_as == "config":
                wandb.config.update(metrics)
            else:
                wandb_metrics = {}
                for key, value in metrics.items():
                    try:
                        if isinstance(value, tuple) and len(value) == 2:
                            bin_counts, bin_edges = value
                            if isinstance(bin_counts, list | np.ndarray | jax.Array) and isinstance(
                                bin_edges, list | np.ndarray | jax.Array
                            ):
                                bin_counts = np.asarray(
                                    jax.device_get(bin_counts) if isinstance(bin_counts, jax.Array) else bin_counts
                                ).reshape(-1)
                                bin_edges = np.asarray(
                                    jax.device_get(bin_edges) if isinstance(bin_edges, jax.Array) else bin_edges
                                ).reshape(-1)
                                if bin_edges.dtype == jnp.bfloat16:
                                    bin_edges = bin_edges.astype(np.float32)
                                np_histogram = (bin_counts, bin_edges)

                                wandb_metrics[key] = wandb.Histogram(np_histogram=np_histogram)
                                continue

                        wandb_metrics[key] = (
                            self._create_wandb_histogram(value)
                            if isinstance(value, float | int | list | tuple | np.ndarray | np.generic | jax.Array)
                            else value
                        )

                    except Exception as e:
                        warnings.warn(f"Failed to log metric {key} to wandb: {e}", stacklevel=3)

                wandb_metrics = {k: v for k, v in wandb_metrics.items() if v is not None}

                try:
                    wandb.log(wandb_metrics, step=step)
                except Exception as e:
                    warnings.warn(f"Failed to log metrics to wandb: {e}", stacklevel=3)

    def _log_to_tensorboard(
        self,
        metrics: dict[str, tp.Any],
        step: int,
        log_as: tp.Literal["summary", "config"] | None = None,
    ):
        """
        Log metrics to TensorBoard.

        Args:
            metrics: A dictionary of metrics to log
            step: The current step or iteration number
            log_as: Currently not used for TensorBoard
        """
        summary_writer = self.get_tensorboard()
        if summary_writer is not None:
            try:
                for key, value in metrics.items():
                    try:
                        if isinstance(value, float | int | np.floating | np.integer):
                            summary_writer.add_scalar(
                                key, float(value) if isinstance(value, np.floating) else value, step
                            )
                        elif isinstance(value, tuple) and len(value) == 2:
                            bin_counts, bin_edges = value
                            if isinstance(bin_counts, list | np.ndarray | jax.Array) and isinstance(
                                bin_edges, list | np.ndarray | jax.Array
                            ):
                                samples = _histogram_tuple_to_samples(bin_counts, bin_edges)
                                if samples.size > 0:
                                    summary_writer.add_histogram(key, samples, step)
                        elif isinstance(value, list | np.ndarray | jax.Array):
                            arr = np.asarray(jax.device_get(value) if isinstance(value, jax.Array) else value)
                            if arr.dtype == jnp.bfloat16:
                                arr = arr.astype(np.float32)
                            summary_writer.add_histogram(key, arr, step)
                    except Exception as e:
                        warnings.warn(f"Failed to log metric {key} to TensorBoard: {e}", stacklevel=1)
            finally:
                summary_writer.flush()

    def _log_eyvan_histogram_tuple(
        self,
        run: tp.Any,
        key: str,
        bin_counts: tp.Any,
        bin_edges: tp.Any,
        step: int,
        *,
        axes: dict[str, float] | None = None,
    ) -> None:
        """Record a pre-binned ``(bin_counts, bin_edges)`` histogram on an Eyvan run safely."""
        ok_counts, host_counts = _safe_eyvan_device_get(bin_counts)
        ok_edges, host_edges = _safe_eyvan_device_get(bin_edges)
        if not (ok_counts and ok_edges):
            return
        counts_arr = np.asarray(host_counts).reshape(-1)
        edges_arr = np.asarray(host_edges).reshape(-1)
        if not (_is_eyvan_real_numeric_dtype(counts_arr.dtype) and _is_eyvan_real_numeric_dtype(edges_arr.dtype)):
            return
        if counts_arr.dtype == jnp.bfloat16:
            counts_arr = counts_arr.astype(np.float32)
        if edges_arr.dtype == jnp.bfloat16:
            edges_arr = edges_arr.astype(np.float32)

        axes_kwargs = {"axes": axes} if axes else {}
        if 1 <= counts_arr.size <= _EYVAN_MAX_HIST_BINS and edges_arr.size == counts_arr.size + 1:
            counts_f = counts_arr.astype(np.float64)
            edges_f = edges_arr.astype(np.float64)
            if (
                np.all(np.isfinite(counts_f))
                and np.all(counts_f >= 0)
                and np.all(np.isfinite(edges_f))
                and np.all(np.diff(edges_f) > 0)
            ):
                count_list = np.rint(counts_f).astype(np.int64).tolist()
                run.histogram(key, counts=count_list, edges=edges_f.tolist(), step=step, **axes_kwargs)
                return

        samples = _histogram_tuple_to_samples(counts_arr, edges_arr)
        if samples.size > 0:
            run.histogram(key, values=samples, step=step, **axes_kwargs)

    def _log_to_eyvan(
        self,
        metrics: dict[str, tp.Any],
        step: int,
        log_as: tp.Literal["summary", "config"] | None = None,
        *,
        axes: tp.Mapping[str, float | int] | None = None,
    ) -> None:
        """Log metrics to Eyvan experiment tracking and optionally forward rank-0 progress.

        Never raises on any tracking or transport error so training is never interrupted.
        """
        run = self.get_eyvan_track()
        should_report_progress = (
            log_as is None
            and bool(getattr(self, "eyvan_report_progress", True))
            and not getattr(self, "performance_mode", False)
        )
        if run is None and not should_report_progress:
            return

        try:
            step_int = max(int(get_safe_arr(step)), 0)
        except Exception:
            step_int = 0

        if log_as == "summary":
            if run is not None:
                try:
                    sanitized = {
                        _sanitize_eyvan_key(k): _normalize_eyvan_meta_value(v)
                        for k, v in metrics.items()
                        if v is not None
                    }
                    if sanitized:
                        run.summary(**sanitized)
                except Exception as exc:
                    warnings.warn(f"Failed to log summary metrics to Eyvan: {exc}", stacklevel=2)
            return

        if log_as == "config":
            if run is not None:
                try:
                    sanitized = {
                        _sanitize_eyvan_key(k): _normalize_eyvan_meta_value(v)
                        for k, v in metrics.items()
                        if v is not None
                    }
                    if sanitized:
                        run.config(sanitized)
                except Exception as exc:
                    warnings.warn(f"Failed to log config metrics to Eyvan: {exc}", stacklevel=2)
            return

        clean_axes = _sanitize_eyvan_axes(axes)
        scalars: dict[str, float] = {}
        hist_tuples: list[tuple[str, tp.Any, tp.Any]] = []
        hist_arrays: list[tuple[str, tp.Any]] = []

        for key, value in metrics.items():
            if value is None:
                continue
            try:
                skey = _sanitize_eyvan_key(key)
                is_scalar, scalar_val = _to_eyvan_scalar(value)
                if is_scalar:
                    scalars[skey] = scalar_val
                    continue
                if run is None:
                    continue
                if isinstance(value, tuple) and len(value) == 2:
                    bin_counts, bin_edges = value
                    if isinstance(bin_counts, list | tuple | np.ndarray | jax.Array) and isinstance(
                        bin_edges, list | tuple | np.ndarray | jax.Array
                    ):
                        hist_tuples.append((skey, bin_counts, bin_edges))
                        continue
                if isinstance(value, list | tuple | np.ndarray | jax.Array):
                    hist_arrays.append((skey, value))
            except Exception as exc:
                warnings.warn(f"Failed to prepare metric {key!r} for Eyvan: {exc}", stacklevel=2)

        if run is not None:
            if scalars:
                try:
                    if clean_axes:
                        run.log(scalars, step=step_int, axes=clean_axes)
                    else:
                        run.log(scalars, step=step_int)
                except Exception as exc:
                    warnings.warn(f"Failed to log scalar metrics to Eyvan: {exc}", stacklevel=2)
            for skey, bin_counts, bin_edges in hist_tuples:
                try:
                    self._log_eyvan_histogram_tuple(run, skey, bin_counts, bin_edges, step_int, axes=clean_axes)
                except Exception as exc:
                    warnings.warn(f"Failed to log histogram {skey!r} to Eyvan: {exc}", stacklevel=2)
            for skey, raw_values in hist_arrays:
                try:
                    ok_vals, host_vals = _safe_eyvan_device_get(raw_values)
                    if not ok_vals:
                        continue
                    arr = np.asarray(host_vals)
                    if not _is_eyvan_real_numeric_dtype(arr.dtype):
                        continue
                    if arr.dtype == jnp.bfloat16:
                        arr = arr.astype(np.float32)
                    if arr.size > 0:
                        if clean_axes:
                            run.histogram(skey, values=arr.reshape(-1), step=step_int, axes=clean_axes)
                        else:
                            run.histogram(skey, values=arr.reshape(-1), step=step_int)
                except Exception as exc:
                    warnings.warn(f"Failed to log array histogram {skey!r} to Eyvan: {exc}", stacklevel=2)

        if should_report_progress:
            try:
                is_rank_zero = bool(self.is_process_zero)
            except Exception:
                is_rank_zero = False
            if is_rank_zero:
                eyvan = _import_eyvan()
                report_fn = getattr(eyvan, "report_progress", None) if eyvan is not None else None
                if callable(report_fn):
                    raw_total = getattr(self, "max_training_steps", None)
                    if raw_total is None:
                        raw_total = self.__dict__.get("_eyvan_total_steps")
                    total: int | None = None
                    if raw_total is not None and _is_real_int(raw_total):
                        candidate_total = int(raw_total)
                        if candidate_total >= 0:
                            total = max(candidate_total, step_int)
                    finite_scalars = {k: v for k, v in scalars.items() if math.isfinite(v)}
                    try:
                        report_fn(step=step_int, total=total, metrics=finite_scalars)
                    except Exception as exc:
                        warnings.warn(f"Failed to report progress to Eyvan: {exc}", stacklevel=2)

    def log_eyvan_completions(
        self,
        prompt: str,
        responses: tp.Sequence[str],
        *,
        rewards: tp.Any = None,
        advantages: tp.Any = None,
        group: str | int | None = None,
        answer: str | None = None,
        tokens: tp.Any = None,
        kl: tp.Any = None,
        truncated: tp.Any = None,
        policy_version: int | None = None,
        step: int | None = None,
    ) -> bool:
        """Record one LLM-RL completion group on the active Eyvan run without raising.

        Args:
            prompt: Prompt string shared by the completion group.
            responses: Non-empty sequence of completion strings (capped at 64).
            rewards: Optional per-response scalar rewards or mapping of
                ``{component_name: per_response_rewards}``.
            advantages: Optional per-response advantage values.
            group: Optional group identifier (1-128 chars).
            answer: Optional reference answer text.
            tokens: Optional per-response non-negative token counts.
            kl: Optional per-response finite KL values.
            truncated: Optional per-response truncation flags.
            policy_version: Optional non-negative behavior policy version.
            step: Optional non-negative learner step.

        Returns:
            ``True`` when queued onto an active Eyvan run, ``False`` otherwise.
        """
        run = self.get_eyvan_track()
        if run is None or getattr(run, "finished", False) or not getattr(run, "active", True):
            return False
        try:
            raw_responses = _coerce_eyvan_sequence(responses)
            if not raw_responses:
                return False
            response_texts = [
                _sanitize_eyvan_utf8_text(item, _EYVAN_MAX_TEXT_LEN) for item in raw_responses[:_EYVAN_MAX_GROUP_SIZE]
            ]
            size = len(response_texts)
            prompt_text = _sanitize_eyvan_utf8_text(prompt, _EYVAN_MAX_TEXT_LEN)
            answer_text = _sanitize_eyvan_utf8_text(answer, _EYVAN_MAX_TEXT_LEN) if answer is not None else None

            clean_rewards: tp.Any = None
            if rewards is not None:
                if isinstance(rewards, tp.Mapping):
                    clean_components: dict[str, list[float]] = {}
                    for comp_name, comp_vals in rewards.items():
                        seq = _coerce_eyvan_sequence(comp_vals)
                        if seq is None or len(seq) < size:
                            continue
                        skey = _sanitize_eyvan_key(comp_name)
                        col: list[float] = []
                        for val in seq[:size]:
                            ok, fval = _to_eyvan_scalar(val)
                            col.append(fval if ok else math.nan)
                        clean_components[skey] = col
                    clean_rewards = clean_components or None
                else:
                    seq = _coerce_eyvan_sequence(rewards)
                    if seq is not None and len(seq) >= size:
                        clean_rewards = [
                            fval if ok else math.nan for ok, fval in (_to_eyvan_scalar(val) for val in seq[:size])
                        ]

            clean_advantages: list[float] | None = None
            if advantages is not None:
                seq = _coerce_eyvan_sequence(advantages)
                if seq is not None and len(seq) >= size:
                    clean_advantages = [
                        fval if ok else math.nan for ok, fval in (_to_eyvan_scalar(val) for val in seq[:size])
                    ]

            clean_tokens: list[int | None] | None = None
            if tokens is not None:
                seq = _coerce_eyvan_sequence(tokens)
                if seq is not None and len(seq) >= size:
                    clean_tokens = [
                        _coerce_eyvan_int(val, minimum=0, clamp_min=True, allow_float_round=True) for val in seq[:size]
                    ]

            clean_kl: list[float | None] | None = None
            if kl is not None:
                seq = _coerce_eyvan_sequence(kl)
                if seq is not None and len(seq) >= size:
                    clean_kl = []
                    for val in seq[:size]:
                        if val is None:
                            clean_kl.append(None)
                            continue
                        ok, fval = _to_eyvan_scalar(val)
                        clean_kl.append(fval if (ok and math.isfinite(fval)) else None)

            clean_truncated: list[bool | None] | None = None
            if truncated is not None:
                seq = _coerce_eyvan_sequence(truncated)
                if seq is not None and len(seq) >= size:
                    clean_truncated = [None if val is None else bool(val) for val in seq[:size]]

            clean_group = _sanitize_eyvan_text_label(group, _EYVAN_MAX_LABEL_LEN)
            clean_policy_version = _coerce_eyvan_int(policy_version, minimum=0)
            clean_step = _coerce_eyvan_int(step, minimum=0, clamp_min=True)

            run.completions(
                prompt_text,
                response_texts,
                rewards=clean_rewards,
                advantages=clean_advantages,
                group=clean_group,
                answer=answer_text,
                tokens=clean_tokens,
                kl=clean_kl,
                truncated=clean_truncated,
                policy_version=clean_policy_version,
                step=clean_step,
            )
            return True
        except Exception as exc:
            warnings.warn(f"Failed to log completions to Eyvan: {exc}", stacklevel=2)
            return False

    def log_eyvan_episode(
        self,
        ret: tp.Any,
        length: int | None = None,
        *,
        success: bool | None = None,
        kind: str = "train",
        actor: str | int | None = None,
        env: str | None = None,
        seed: int | None = None,
        policy_version: int | None = None,
        learner_version: int | None = None,
        checkpoint: str | None = None,
        step: int | None = None,
        axes: tp.Mapping[str, tp.Any] | None = None,
        extra: tp.Mapping[str, tp.Any] | None = None,
    ) -> bool:
        """Record one completed RL episode on the active Eyvan run without raising.

        Args:
            ret: Episode return scalar (Python, NumPy, or JAX scalar; NaN/inf preserved).
            length: Optional non-negative episode length.
            success: Optional boolean task-success flag.
            kind: ``"train"`` or ``"eval"``.
            actor: Optional actor identifier (truncated to 64 chars).
            env: Optional environment name (truncated to 128 chars).
            seed: Optional integer environment seed.
            policy_version: Optional non-negative policy version that generated the episode.
            learner_version: Optional non-negative learner version for lag tracking.
            checkpoint: Optional checkpoint identifier (truncated to 256 chars).
            step: Optional non-negative learner step.
            axes: Optional extra x-axis values.
            extra: Optional mapping of up to 32 finite numeric fields.

        Returns:
            ``True`` when queued onto an active Eyvan run, ``False`` otherwise.
        """
        run = self.get_eyvan_track()
        if (
            run is None
            or getattr(run, "finished", False)
            or not getattr(run, "episodes_active", getattr(run, "active", True))
        ):
            return False
        try:
            ok_ret, ret_val = _to_eyvan_scalar(ret)
            if not ok_ret:
                return False
            clean_length = _coerce_eyvan_int(length, minimum=0, clamp_min=True, allow_float_round=True)
            clean_success = None if success is None else bool(success)
            clean_kind = kind if kind in _EYVAN_EPISODE_KINDS else "train"
            clean_actor = _sanitize_eyvan_text_label(actor, _EYVAN_MAX_ACTOR_LEN)
            clean_env = _sanitize_eyvan_text_label(env, _EYVAN_MAX_ENV_LEN)
            clean_seed = _coerce_eyvan_int(seed, minimum=_EYVAN_MIN_INT64, maximum=_EYVAN_MAX_INT64)
            clean_policy_version = _coerce_eyvan_int(policy_version, minimum=0)
            clean_learner_version = _coerce_eyvan_int(learner_version, minimum=0)
            clean_checkpoint = _sanitize_eyvan_text_label(checkpoint, _EYVAN_MAX_CHECKPOINT_LEN)
            clean_step = _coerce_eyvan_int(step, minimum=0, clamp_min=True)
            clean_axes = _sanitize_eyvan_axes(axes)

            clean_extra: dict[str, float] | None = None
            if isinstance(extra, tp.Mapping):
                sanitized_extra: dict[str, float] = {}
                for key, raw in extra.items():
                    if raw is None:
                        continue
                    ok_val, fval = _to_eyvan_scalar(raw)
                    if not (ok_val and math.isfinite(fval)):
                        continue
                    skey = _sanitize_eyvan_key(key)
                    if skey not in sanitized_extra and len(sanitized_extra) >= _EYVAN_MAX_EXTRA:
                        continue
                    sanitized_extra[skey] = float(fval)
                clean_extra = sanitized_extra or None

            run.episode(
                ret_val,
                clean_length,
                success=clean_success,
                kind=clean_kind,
                actor=clean_actor,
                env=clean_env,
                seed=clean_seed,
                policy_version=clean_policy_version,
                learner_version=clean_learner_version,
                checkpoint=clean_checkpoint,
                step=clean_step,
                axes=clean_axes,
                extra=clean_extra,
            )
            return True
        except Exception as exc:
            warnings.warn(f"Failed to log episode to Eyvan: {exc}", stacklevel=2)
            return False

    def log_eyvan_rollout(
        self,
        steps: tp.Any = None,
        *,
        tracks: tp.Mapping[str, tp.Any] | None = None,
        actions: tp.Any = None,
        action_names: tp.Any = None,
        every: int = 1,
        episode_id: str | int | None = None,
        actor: str | int | None = None,
        env: str | None = None,
        kind: str = "train",
        policy_version: int | None = None,
        step: int | None = None,
        axes: tp.Mapping[str, tp.Any] | None = None,
    ) -> bool:
        """Record a sampled RL trajectory on the active Eyvan run without raising.

        Accepts either per-step mappings (``steps``) or columnar ``tracks``/``actions``.
        String action labels passed via ``actions`` are automatically indexed into
        ``action_names`` so Eyvan's integer-action wire contract is always satisfied.

        Returns:
            ``True`` when the rollout was sampled and queued, ``False`` otherwise.
        """
        run = self.get_eyvan_track()
        if (
            run is None
            or getattr(run, "finished", False)
            or not getattr(run, "episodes_active", getattr(run, "active", True))
        ):
            return False
        try:
            raw_tracks: dict[str, list[tp.Any]] = {}
            raw_actions: list[tp.Any] | None = None

            if steps is not None:
                step_rows = _coerce_eyvan_sequence(steps)
                if not step_rows:
                    return False
                step_rows = step_rows[:_EYVAN_MAX_ROLLOUT_STEPS]
                track_order: list[str] = []
                step_has_action = False
                for row in step_rows:
                    if not isinstance(row, tp.Mapping):
                        return False
                    for key in row:
                        if key == "action":
                            step_has_action = True
                            continue
                        if key not in track_order:
                            track_order.append(key)
                for key in track_order:
                    skey = _sanitize_eyvan_key(key)
                    if skey not in raw_tracks and len(raw_tracks) >= _EYVAN_MAX_TRACKS:
                        continue
                    raw_tracks[skey] = [row.get(key) for row in step_rows]
                if actions is None and step_has_action:
                    raw_actions = [row.get("action") for row in step_rows]

            if tracks is not None and isinstance(tracks, tp.Mapping):
                for key, vals in tracks.items():
                    seq = _coerce_eyvan_sequence(vals)
                    if seq is None or not seq:
                        continue
                    skey = _sanitize_eyvan_key(key)
                    if skey not in raw_tracks and len(raw_tracks) >= _EYVAN_MAX_TRACKS:
                        continue
                    raw_tracks[skey] = seq[:_EYVAN_MAX_ROLLOUT_STEPS]

            if actions is not None:
                seq = _coerce_eyvan_sequence(actions)
                if seq is not None and seq:
                    raw_actions = seq[:_EYVAN_MAX_ROLLOUT_STEPS]

            lengths = [len(vals) for vals in raw_tracks.values()]
            if raw_actions is not None:
                lengths.append(len(raw_actions))
            if not lengths:
                return False
            common_len = min(min(lengths), _EYVAN_MAX_ROLLOUT_STEPS)
            if common_len < 1:
                return False

            clean_tracks: dict[str, list[float | None]] = {}
            for skey, vals in raw_tracks.items():
                encoded: list[float | None] = []
                valid_track = True
                for val in vals[:common_len]:
                    if val is None:
                        encoded.append(None)
                        continue
                    ok, fval = _to_eyvan_scalar(val)
                    if not ok:
                        valid_track = False
                        break
                    encoded.append(fval)
                if valid_track:
                    clean_tracks[skey] = encoded

            raw_action_names = _coerce_eyvan_sequence(action_names) if action_names is not None else None
            clean_actions: list[int | None] | None = None
            clean_action_names: list[str] | None = None
            if raw_actions is not None:
                clean_actions, clean_action_names = _normalize_eyvan_rollout_actions(
                    raw_actions[:common_len],
                    raw_action_names,
                )
            elif raw_action_names is not None:
                clean_action_names = [
                    (_sanitize_eyvan_utf8_text(str(name).strip(), _EYVAN_MAX_ACTION_NAME_LEN) or "action")
                    for name in raw_action_names[:_EYVAN_MAX_ACTION_NAMES]
                ]

            if not clean_tracks and clean_actions is None:
                return False

            clean_every = _coerce_eyvan_int(every, minimum=1, clamp_min=True) or 1
            clean_episode_id = _sanitize_eyvan_text_label(episode_id, _EYVAN_MAX_LABEL_LEN)
            clean_actor = _sanitize_eyvan_text_label(actor, _EYVAN_MAX_ACTOR_LEN)
            clean_env = _sanitize_eyvan_text_label(env, _EYVAN_MAX_ENV_LEN)
            clean_kind = kind if kind in _EYVAN_EPISODE_KINDS else "train"
            clean_policy_version = _coerce_eyvan_int(policy_version, minimum=0)
            clean_step = _coerce_eyvan_int(step, minimum=0, clamp_min=True)
            clean_axes = _sanitize_eyvan_axes(axes)

            return bool(
                run.rollout(
                    tracks=clean_tracks or None,
                    actions=clean_actions,
                    action_names=clean_action_names,
                    every=clean_every,
                    episode_id=clean_episode_id,
                    actor=clean_actor,
                    env=clean_env,
                    kind=clean_kind,
                    policy_version=clean_policy_version,
                    step=clean_step,
                    axes=clean_axes,
                )
            )
        except Exception as exc:
            warnings.warn(f"Failed to log rollout to Eyvan: {exc}", stacklevel=2)
            return False

    def log_eyvan_histogram(
        self,
        key: str,
        values: tp.Any,
        *,
        step: int | None = None,
        bins: int = 64,
        axes: tp.Mapping[str, tp.Any] | None = None,
    ) -> bool:
        """Record a numeric distribution histogram on the active Eyvan run without raising.

        Safely excludes complex dtypes, converts ``bfloat16`` to ``float32``, and
        sanitizes histogram keys and extra x-axes.

        Returns:
            ``True`` when queued onto an active Eyvan run, ``False`` otherwise.
        """
        run = self.get_eyvan_track()
        if run is None or getattr(run, "finished", False) or not getattr(run, "active", True):
            return False
        if values is None:
            return False
        try:
            ok_read, raw = _safe_eyvan_device_get(values)
            if not ok_read:
                return False
            arr = np.asarray(raw)
            if not _is_eyvan_real_numeric_dtype(arr.dtype):
                return False
            if arr.dtype == jnp.bfloat16:
                arr = arr.astype(np.float32)
            flat = arr.reshape(-1)
            if flat.size == 0:
                return False
            skey = _sanitize_eyvan_key(key)
            clean_bins = _coerce_eyvan_int(bins, minimum=1, maximum=_EYVAN_MAX_HIST_BINS, clamp_min=True) or 64
            clean_step = _coerce_eyvan_int(step, minimum=0, clamp_min=True)
            clean_axes = _sanitize_eyvan_axes(axes)
            if clean_axes:
                run.histogram(skey, values=flat, step=clean_step, bins=clean_bins, axes=clean_axes)
            else:
                run.histogram(skey, values=flat, step=clean_step, bins=clean_bins)
            return True
        except Exception as exc:
            warnings.warn(f"Failed to log histogram {key!r} to Eyvan: {exc}", stacklevel=2)
            return False

    def _create_wandb_histogram(self, value):
        """Create a wandb.Histogram object from the given value.

        Args:
            value: The value to convert into a wandb.Histogram. Can be:
                  - JAX array
                  - NumPy array or generic
                  - Other numeric types

        Returns:
            wandb.Histogram | None: A wandb.Histogram object if successful,
                                   None if an error occurs.

        Note:
            Handles dtype conversion for bfloat16 to float32 for
            compatibility with wandb. Automatically moves JAX arrays to CPU.
        """
        try:
            if isinstance(value, jax.Array | np.ndarray | np.generic):
                value = np.array(jax.device_get(value))
                if value.dtype == jnp.bfloat16:
                    value = value.astype(np.float32)
                return wandb.Histogram(value)
            return value
        except Exception as e:
            warnings.warn(f"Failed to create wandb histogram: {e}", stacklevel=1)
            return None

    @classmethod
    def _dict_from_json_file(cls, json_file: str | os.PathLike):
        """Load a JSON file and return its contents as a dictionary.

        Args:
            json_file: Path to the JSON configuration file.

        Returns:
            dict: The parsed JSON contents.
        """
        return json.loads(ePath(json_file).read_text())

    _RUNTIME_ONLY_FIELDS: tp.ClassVar[frozenset[str]] = frozenset(
        {
            "buckets",
            "bucket_datasets",
            "watchers",
            "custom_scheduler",
            "pruning_module",
            "tensor_straight_through",
            "straight_through_emulator",
            "mpmd_scheduler",
        }
    )

    def to_dict(self) -> dict[str, tp.Any]:
        """Serializes this instance to a dictionary.

        Returns:
            dict[str, tp.Any]: A dictionary containing all serializable fields.
        """
        # Runtime-only fields (callables, live watchers, bucket objects/dataloaders,
        # pruning wrappers, MPMD schedules) are not reconstructable from JSON and
        # must not leak as stringified objects into the saved projection.
        result = {}
        for field_obj in fields(self):
            value = getattr(self, field_obj.name)
            if value is Ellipsis:
                continue
            if field_obj.name.startswith("_"):
                continue
            if field_obj.name in self._RUNTIME_ONLY_FIELDS:
                continue
            if _is_dtype_like(value):
                result[field_obj.name] = jnp.dtype(value).name
            elif isinstance(value, tuple):
                result[field_obj.name] = [jnp.dtype(v).name if _is_dtype_like(v) else v for v in value]
            elif value is None:
                result[field_obj.name] = None
            elif hasattr(value, "to_dict") and callable(value.to_dict):
                result[field_obj.name] = value.to_dict()
            else:
                candidate = (
                    {k: jnp.dtype(v).name if _is_dtype_like(v) else v for k, v in value.items()}
                    if isinstance(value, dict)
                    else value
                )
                try:
                    json.dumps(candidate)
                    result[field_obj.name] = candidate
                except (TypeError, OverflowError):
                    result[field_obj.name] = str(value)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, tp.Any]):
        """Deserializes a dictionary into a TrainingArguments instance.

        Args:
            data: Dictionary containing field names and values.

        Returns:
            TrainingArguments: A new instance created from the dictionary.
        """
        data = _apply_training_args_legacy_aliases(data)

        processed_data = {}
        try:
            type_hints = tp.get_type_hints(cls)
        except Exception:
            type_hints = {}
            for base in reversed(cls.__mro__):
                type_hints.update(getattr(base, "__annotations__", {}))

        for field_obj in fields(cls):
            field_name = field_obj.name
            if field_name not in data:
                continue

            value = data[field_name]
            field_type = type_hints.get(field_name)

            if field_name in cls._RUNTIME_ONLY_FIELDS and (
                isinstance(value, str)
                or (
                    field_name == "watchers" and isinstance(value, list) and any(isinstance(item, str) for item in value)
                )
            ):
                continue

            if field_name == "tx_mu_dtype":
                processed_data[field_name] = _parse_dtype(value)
            elif field_name == "pose" and isinstance(value, dict):
                processed_data[field_name] = PoSEConfig.from_dict(value)
            elif field_name == "randomized_yarn" and isinstance(value, dict):
                processed_data[field_name] = RandomizedYaRNConfig.from_dict(value)
            elif field_name == "process_encoder" and isinstance(value, dict):
                processed_data[field_name] = ProcessEncoderConfig.from_dict(value)
            elif field_name == "bucket_rule" and isinstance(value, dict):
                # Reconstruct the BucketRule from its discriminator-tagged dict
                # ({"kind": "mod", ...}). Deferred import avoids a cycle.
                from easydel.trainers.buckets import BucketRule

                processed_data[field_name] = BucketRule.from_dict(value)
            elif value is not None and isinstance(value, list) and _expects_tuple_type(field_type):
                processed_data[field_name] = tuple(value)
            else:
                processed_data[field_name] = value

        return cls(**processed_data)

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        config_dict = self.to_dict()
        config_dict["trainer_config_class"] = self.__class__.__name__
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    @classmethod
    def load_arguments(cls, json_file: str | os.PathLike):
        """
        Load training arguments from a JSON file.

        This class method reconstructs a TrainingArguments instance from a
        previously saved JSON configuration file. It handles class resolution
        and proper type conversion.

        Args:
            json_file: Path to the JSON file containing saved arguments

        Returns:
            TrainingArguments: Reconstructed configuration object with all
                              settings from the saved file

        Note:
            The JSON file should contain a 'trainer_config_class' field
            for proper class resolution when using subclasses.
        """

        config_dict = cls._dict_from_json_file(json_file)
        return cls.load_from_json(config_dict)

    @classmethod
    def load_from_json(cls, config_dict):
        """Reconstruct a TrainingArguments instance from a parsed JSON dictionary.

        Args:
            config_dict: Dictionary parsed from a JSON configuration file.
                May contain a 'trainer_config_class' key for subclass resolution.

        Returns:
            TrainingArguments: Reconstructed configuration object.

        Raises:
            ValueError: If the trainer config class cannot be resolved.
        """
        config_dict = _apply_training_args_legacy_aliases(config_dict)
        if "trainer_config_class" in config_dict.keys():
            import easydel as ed

            class_name = config_dict.pop("trainer_config_class")
            resolved_cls = getattr(ed, class_name, None) if isinstance(class_name, str) else None
            if resolved_cls is None:
                if class_name == cls.__name__:
                    resolved_cls = cls
                else:
                    raise ValueError(f"We couldn't clarify the trainer config class {class_name!r} from provided json.")
            cls = resolved_cls
        return cls.from_dict(config_dict)

    def save_arguments(self, json_file_path: str | os.PathLike | ePathLike):
        """
        Save training arguments to a JSON file.

        This method serializes the current configuration to a JSON file,
        preserving all settings for later reconstruction. The saved file
        includes class information for proper deserialization.

        Args:
            json_file_path: Path where the JSON file will be saved.
                           Parent directories are created if needed.

        Note:
            The saved JSON includes a 'trainer_config_class' field to
            ensure proper class resolution when loading.
        """
        ePath(json_file_path).write_text(self.to_json_string())

    def _get_save_directory(self, create: bool = True) -> ePathLike | None:
        """Return the base checkpoint save directory.

        Args:
            create: If True, ensure the directory exists before returning.

        Returns:
            The checkpoint directory path, or None if no path is configured.
        """
        if self.save_directory is None or self.model_name is None:
            return None
        if create:
            self.ensure_checkpoint_path()
        return self.get_path()

    def _get_save_directory_milestone(self, step, create: bool = True) -> ePathLike:
        """Return the checkpoint directory for a specific training step.

        Creates a subdirectory named ``run-{step}`` under the base save
        directory.

        Args:
            step: The training step number used to name the subdirectory.
            create: If True, create the directory (and parents) if it does
                not exist.

        Returns:
            The milestone checkpoint directory path.
        """
        directory_name = f"run-{step}"
        savedir = self._get_save_directory(create=create)
        if savedir is None:
            return ePath("/dev/null")  # pyright: ignore[reportReturnType]
        save_directory = savedir / directory_name
        if create:
            save_directory.mkdir(exist_ok=True, parents=True)
        return save_directory

    __hash__ = hash_fn


def _get_max_sequence_length(self: TrainingArguments) -> int | None:
    """Getter for the deprecated ``max_sequence_length`` property.

    Returns:
        The current ``max_length`` value.
    """
    return self.max_length


def _set_max_sequence_length(self: TrainingArguments, value: int | None) -> None:
    """Setter for the deprecated ``max_sequence_length`` property.

    Emits a FutureWarning directing users to ``max_length``, then
    forwards the value.

    Args:
        value: The sequence length to set.
    """
    warnings.warn(
        "`max_sequence_length` is deprecated; use `max_length` instead.",
        FutureWarning,
        stacklevel=2,
    )
    self.max_length = value


TrainingArguments.max_sequence_length = property(
    _get_max_sequence_length,
    _set_max_sequence_length,
    doc="Deprecated alias for `max_length`.",
)


def _get_quantization_block(self: TrainingArguments) -> int | None:
    """Getter for the deprecated ``quantization_block`` property.

    Returns:
        The current ``quantization_group_size`` value.
    """
    return self.quantization_group_size


def _set_quantization_block(self: TrainingArguments, value: int | None) -> None:
    """Setter for the deprecated ``quantization_block`` property.

    Emits a FutureWarning directing users to ``quantization_group_size``,
    then forwards the value.

    Args:
        value: The quantization block size to set.
    """
    warnings.warn(
        "`quantization_block` is deprecated; use `quantization_group_size` instead.",
        FutureWarning,
        stacklevel=2,
    )
    self.quantization_group_size = value


TrainingArguments.quantization_block = property(
    _get_quantization_block,
    _set_quantization_block,
    doc="Deprecated alias for `quantization_group_size`.",
)
