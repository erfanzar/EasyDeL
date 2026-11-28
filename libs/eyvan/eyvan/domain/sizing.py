# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pure transformer-memory estimates, not a compiler or a capacity guarantee.

Model state is assumed fully sharded and activations data-parallel across chips.
No provider, filesystem, model-weight or network access is performed. Real kernels,
sharding, padding and allocator peaks need measurement; headroom is not a proof
that a compiled job will fit.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from eyvan.domain.accelerators import parse_tpu_accelerator_type

GIB = 1024**3
FP32_BYTES = 4
DTYPE_BYTES: Mapping[str, int] = MappingProxyType({"bfloat16": 2, "float16": 2, "float32": FP32_BYTES})
OPTIMIZER_STATE_FACTORS: Mapping[str, float] = MappingProxyType(
    {"adamw": 2.0, "adafactor": 0.1, "lion": 1.0, "sgd": 0.0, "none": 0.0}
)
TRAINERS = frozenset({"pretrain", "sft", "dpo", "orpo", "grpo", "inference"})
REFERENCE_TRAINERS = frozenset({"dpo", "grpo"})
PREFERENCE_TRAINERS = frozenset({"dpo", "orpo"})
ACTIVATION_FACTOR = 16
CHECKPOINTED_ACTIVATION_FACTOR = 2
INFERENCE_ACTIVATION_FACTOR = 1
PREFERENCE_ACTIVATION_MULTIPLIER = 2
LORA_ADAPTED_PROJECTIONS = 7
LORA_MATRICES_PER_PROJECTION = 2
KV_CACHE_TENSORS = 2
GATED_MLP_PROJECTIONS = 3
LAYER_NORMS = 2
DEFAULT_HIDDEN_SIZE = 4096
DEFAULT_NUM_LAYERS = 32
DEFAULT_VOCAB_SIZE = 32000
DEFAULT_HEADROOM = 0.85
MAX_REPORTED_FITS = 5

TPU_HBM_GIB: Mapping[str, float] = MappingProxyType(
    {"v4": 32.0, "v5e": 16.0, "v5litepod": 16.0, "v5p": 95.0, "v6e": 32.0, "tpu7x": 192.0}
)
# The parser returns public suffix units, not chips. These are the same unit
# conversions documented by capacity.hardware_catalog and autoscale.profiles.
TPU_SUFFIX_UNITS_PER_CHIP: Mapping[str, int] = MappingProxyType(
    {"v4": 2, "v5p": 2, "v5e": 1, "v5litepod": 1, "v6e": 1, "tpu7x": 1}
)
DEFAULT_CANDIDATE_TYPES: tuple[str, ...] = tuple(
    f"{generation}-{size}"
    for generation, sizes in (
        ("v5p", (8, 16, 32, 64, 128, 256, 512)),
        ("v4", (8, 16, 32, 64, 128, 256, 512)),
        ("v6e", (1, 4, 8, 16, 32, 64, 128, 256)),
        ("v5litepod", (1, 4, 8, 16, 32, 64, 128, 256)),
    )
    for size in sizes
)


def _integer(name: str, value: Any, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class ModelShape:
    """Resident parameter count and dimensions needed for a memory estimate.

    All experts remain resident even when only ``active_parameters`` participate
    in a token's computation. Missing KV dimensions conservatively use hidden_size
    as the combined KV width. Parameter-only callers should use the helper below.
    """

    parameters: int
    hidden_size: int
    num_layers: int
    vocab_size: int
    num_kv_heads: int | None = None
    head_dim: int | None = None
    active_parameters: int | None = None

    def __post_init__(self) -> None:
        for name in ("parameters", "hidden_size", "num_layers", "vocab_size"):
            _integer(name, getattr(self, name))
        for name in ("num_kv_heads", "head_dim"):
            value = getattr(self, name)
            if value is not None:
                _integer(name, value)
        active = (
            self.parameters if self.active_parameters is None else _integer("active_parameters", self.active_parameters)
        )
        if active > self.parameters:
            raise ValueError("active_parameters must not exceed parameters")
        object.__setattr__(self, "active_parameters", active)


def shape_from_parameter_count(
    parameters: int,
    *,
    hidden_size: int | None = None,
    num_layers: int | None = None,
    vocab_size: int | None = None,
) -> ModelShape:
    """Use supplied dimensions or rough 7B-style defaults: H=4096, L=32, V=32000.

    These defaults are placeholders, not inferred architecture. KV width falls
    back to H (multi-head attention); supplying a config is strongly preferred.
    """

    return ModelShape(
        parameters=parameters,
        hidden_size=DEFAULT_HIDDEN_SIZE if hidden_size is None else hidden_size,
        num_layers=DEFAULT_NUM_LAYERS if num_layers is None else num_layers,
        vocab_size=DEFAULT_VOCAB_SIZE if vocab_size is None else vocab_size,
    )


def _config_integer(config: Mapping[str, Any], name: str, *aliases: str, default: int | None = None) -> int:
    for key in (name, *aliases):
        if key in config and config[key] is not None:
            return _integer(key, config[key])
    if default is not None:
        return default
    raise ValueError(f"missing required model config field: {name}")


def model_shape_from_config(config: Mapping[str, Any]) -> ModelShape:
    """Count Llama/Mistral/Qwen2/Gemma-style decoder parameters without weights.

    Count embeddings, optional untied output head, Q/K/V/O projections, three
    gated-MLP projections, two per-layer RMSNorms and a final norm. For Mixtral
    and Qwen2-MoE, count every expert plus its router; active counts substitute
    experts-per-token. Optional Qwen shared experts and sparse-layer frequency
    are included. Small projection biases and architecture-specific extra norms
    are excluded. ``head_dim`` overrides H/attention_heads when provided.
    """

    hidden = _config_integer(config, "hidden_size")
    layers = _config_integer(config, "num_hidden_layers", "num_layers")
    vocab = _config_integer(config, "vocab_size")
    heads = _config_integer(config, "num_attention_heads")
    if config.get("head_dim") is None and hidden % heads:
        raise ValueError("hidden_size must be divisible by num_attention_heads when head_dim is missing")
    head_dim = _config_integer(config, "head_dim", default=hidden // heads)
    kv_heads = _config_integer(config, "num_key_value_heads", "num_kv_heads", default=heads)
    if kv_heads > heads or heads % kv_heads:
        raise ValueError("num_key_value_heads must divide num_attention_heads")
    tied = config.get("tie_word_embeddings", False)
    if not isinstance(tied, bool):
        raise ValueError("tie_word_embeddings must be a boolean")

    attention = hidden * (heads * head_dim) + KV_CACHE_TENSORS * hidden * (kv_heads * head_dim)
    attention += (heads * head_dim) * hidden
    common = vocab * hidden * (1 if tied else 2) + layers * (attention + LAYER_NORMS * hidden) + hidden
    experts = _config_integer(config, "num_local_experts", "num_experts", default=1)
    if experts == 1:
        intermediate = _config_integer(config, "intermediate_size")
        parameters = common + layers * GATED_MLP_PROJECTIONS * hidden * intermediate
        active = parameters
    else:
        intermediate = _config_integer(config, "moe_intermediate_size", "intermediate_size")
        active_experts = _config_integer(config, "num_experts_per_tok")
        if active_experts > experts:
            raise ValueError("num_experts_per_tok must not exceed the number of experts")
        sparse_step = _config_integer(config, "decoder_sparse_step", default=1)
        dense_indices = config.get("mlp_only_layers", [])
        if not isinstance(dense_indices, (list, tuple)):
            raise ValueError("mlp_only_layers must be a list of layer indices")
        for index in dense_indices:
            _integer("mlp_only_layers index", index, minimum=0)
            if index >= layers:
                raise ValueError("mlp_only_layers index must be less than num_hidden_layers")
        sparse_layers = sum((index + 1) % sparse_step == 0 and index not in dense_indices for index in range(layers))
        dense_layers = layers - sparse_layers
        dense_mlp = 0
        if dense_layers:
            dense_mlp = dense_layers * GATED_MLP_PROJECTIONS * hidden * _config_integer(config, "intermediate_size")
        shared_intermediate = config.get("shared_expert_intermediate_size", 0)
        _integer("shared_expert_intermediate_size", shared_intermediate, minimum=0)
        shared = GATED_MLP_PROJECTIONS * hidden * shared_intermediate + (hidden if shared_intermediate else 0)
        expert_mlp = GATED_MLP_PROJECTIONS * hidden * intermediate
        router = hidden * experts
        parameters = common + dense_mlp + sparse_layers * (expert_mlp * experts + router + shared)
        active = common + dense_mlp + sparse_layers * (expert_mlp * active_experts + router + shared)
    return ModelShape(parameters, hidden, layers, vocab, kv_heads, head_dim, active)


@dataclass(frozen=True, slots=True)
class SizingRequest:
    """Validated training or inference assumptions; batch is global, not per chip."""

    shape: ModelShape
    dtype: str = "bfloat16"
    optimizer: str = "adamw"
    trainer: str = "sft"
    sequence_length: int = 2048
    global_batch_size: int = 8
    gradient_checkpointing: bool = True
    lora_rank: int | None = None
    generation_tokens: int = 0
    num_generations: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.shape, ModelShape):
            raise ValueError("shape must be a ModelShape")
        for name, choices in (("dtype", DTYPE_BYTES), ("optimizer", OPTIMIZER_STATE_FACTORS), ("trainer", TRAINERS)):
            value = getattr(self, name)
            if not isinstance(value, str) or value not in choices:
                raise ValueError(f"unsupported {name}: {value!r}; choose from {', '.join(sorted(choices))}")
        for name in ("sequence_length", "global_batch_size", "num_generations"):
            _integer(name, getattr(self, name))
        _integer("generation_tokens", self.generation_tokens, minimum=0)
        if self.lora_rank is not None:
            _integer("lora_rank", self.lora_rank)
        if not isinstance(self.gradient_checkpointing, bool):
            raise ValueError("gradient_checkpointing must be a boolean")


@dataclass(frozen=True, slots=True)
class MemoryEstimate:
    """Aggregate bytes before sharding; components sum exactly to total_bytes."""

    parameters_bytes: int
    gradients_bytes: int
    optimizer_bytes: int
    reference_bytes: int
    activations_bytes: int
    generation_bytes: int
    total_bytes: int
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        components = (
            self.parameters_bytes,
            self.gradients_bytes,
            self.optimizer_bytes,
            self.reference_bytes,
            self.activations_bytes,
            self.generation_bytes,
        )
        for value in (*components, self.total_bytes):
            _integer("memory bytes", value, minimum=0)
        if sum(components) != self.total_bytes:
            raise ValueError("total_bytes must equal the sum of memory components")


def estimate_memory(request: SizingRequest) -> MemoryEstimate:
    """Estimate resident state, activations and KV cache using named constants.

    Weights/gradients use dtype bytes. Optimizers use fp32 state equivalents per
    trainable parameter: AdamW 2, Lion 1, Adafactor approximately 0.1 (factored
    shape-dependent proxy), SGD/none 0. All but none add an fp32 master copy for
    16-bit training. Inference has neither gradients nor optimizer state.

    LoRA approximates seven square-H adapted projections per layer: Q/K/V/O
    and gated-MLP gate/up/down, each with two rank-r matrices. Base weights stay
    frozen and adapter weights are additional. DPO/GRPO keep one unshared frozen
    reference copy; ORPO does not. All MoE experts stay resident.

    Activations = batch * seq * H * L * dtype_bytes * factor: 16 normally, 2
    with gradient checkpointing, 1 for inference. DPO/ORPO double this for paired
    sequences; GRPO multiplies by num_generations. GRPO/inference KV cache is
    2 * L * KV_width * dtype_bytes * (seq + generated_tokens) * batch * generations.
    """

    shape = request.shape
    dtype_bytes = DTYPE_BYTES[request.dtype]
    training = request.trainer != "inference"
    notes = [
        "Analytic estimate only: kernel workspaces, padding, communication and allocator peaks are excluded.",
        "Fully sharded model state and perfectly data-parallel activations/KV cache are assumed.",
    ]
    if shape.num_kv_heads is None or shape.head_dim is None:
        notes.append("Architecture dimensions may be rough defaults; missing KV dimensions use hidden_size (MHA).")
    if shape.active_parameters != shape.parameters:
        notes.append("All MoE experts remain resident; active parameters do not reduce weights or optimizer state.")
    adapter_parameters = 0
    if request.lora_rank is not None:
        adapter_parameters = min(
            shape.parameters,
            LORA_MATRICES_PER_PROJECTION
            * request.lora_rank
            * shape.hidden_size
            * LORA_ADAPTED_PROJECTIONS
            * shape.num_layers,
        )
        notes.append(
            "LoRA approximates Q/K/V/O plus gate/up/down as seven square-H projections per layer, capped at base count."
        )
    trainable = (adapter_parameters if request.lora_rank is not None else shape.parameters) if training else 0
    parameters_bytes = (shape.parameters + adapter_parameters) * dtype_bytes
    gradients_bytes = trainable * dtype_bytes
    optimizer_bytes = 0
    if training and request.optimizer != "none":
        state_bytes = math.ceil(trainable * FP32_BYTES * OPTIMIZER_STATE_FACTORS[request.optimizer])
        master_bytes = trainable * FP32_BYTES if dtype_bytes < FP32_BYTES else 0
        optimizer_bytes = state_bytes + master_bytes
        notes.append(f"{request.optimizer} uses fp32 optimizer state; 16-bit training adds an fp32 master copy.")
        if request.optimizer == "adafactor":
            notes.append("Adafactor factored state is approximated as 0.1 fp32 values per trainable parameter.")
    reference_bytes = shape.parameters * dtype_bytes if request.trainer in REFERENCE_TRAINERS else 0
    if reference_bytes:
        notes.append("DPO/GRPO includes one full frozen reference model; reference/base weight sharing is not assumed.")
    elif request.trainer == "orpo":
        notes.append("ORPO is reference-free; no frozen reference model is included.")
    factor = INFERENCE_ACTIVATION_FACTOR
    if training:
        factor = CHECKPOINTED_ACTIVATION_FACTOR if request.gradient_checkpointing else ACTIVATION_FACTOR
    if request.trainer in PREFERENCE_TRAINERS:
        factor *= PREFERENCE_ACTIVATION_MULTIPLIER
        notes.append("Preference training holds paired chosen/rejected sequences in the activation estimate.")
    if request.trainer == "grpo":
        factor *= request.num_generations
    activations_bytes = (
        request.global_batch_size * request.sequence_length * shape.hidden_size * shape.num_layers * dtype_bytes * factor
    )
    notes.append(f"Activations use batch*seq*H*layers*dtype_bytes*{factor}; actual trainer/compiler memory can differ.")
    generation_bytes = 0
    if request.trainer in {"grpo", "inference"}:
        kv_width = (
            shape.num_kv_heads * shape.head_dim
            if shape.num_kv_heads is not None and shape.head_dim is not None
            else shape.hidden_size
        )
        generation_bytes = (
            KV_CACHE_TENSORS
            * shape.num_layers
            * kv_width
            * dtype_bytes
            * (request.sequence_length + request.generation_tokens)
            * request.global_batch_size
            * request.num_generations
        )
        notes.append(
            "Generation keeps prompt+output KV cache for every batch item and generation at the requested dtype."
        )
    components = (
        parameters_bytes,
        gradients_bytes,
        optimizer_bytes,
        reference_bytes,
        activations_bytes,
        generation_bytes,
    )
    return MemoryEstimate(*components, sum(components), tuple(notes))


@dataclass(frozen=True, slots=True)
class SliceFit:
    """One candidate's actual-HBM utilization (not divided by the headroom cap)."""

    accelerator_type: str
    chips: int
    hbm_per_chip_gib: float
    per_chip_bytes: float
    fits: bool
    utilization: float


def fitting_slices(
    estimate: MemoryEstimate,
    accelerator_types: Iterable[str] = DEFAULT_CANDIDATE_TYPES,
    *,
    headroom: float = DEFAULT_HEADROOM,
) -> list[SliceFit]:
    """Evaluate known types, sorting all fit/non-fit results by chips then name.

    Per-chip bytes = total/chips under ideal full-state sharding/data parallelism.
    Fit means per-chip bytes <= physical HBM * headroom. Malformed types, unknown
    generations and fractional-chip suffixes are skipped, not labeled non-fitting.
    A fit is not provider availability, throughput or sharding compatibility.
    """

    if isinstance(headroom, bool) or not isinstance(headroom, (int, float)) or not 0 < headroom <= 1:
        raise ValueError("headroom must be a finite fraction in (0, 1]")
    results = []
    for accelerator_type in sorted(set(accelerator_types)):
        parsed = parse_tpu_accelerator_type(accelerator_type)
        if parsed is None or parsed.generation not in TPU_HBM_GIB:
            continue
        units_per_chip = TPU_SUFFIX_UNITS_PER_CHIP[parsed.generation]
        if parsed.size % units_per_chip:
            continue
        chips = parsed.size // units_per_chip
        hbm = TPU_HBM_GIB[parsed.generation]
        per_chip_bytes = estimate.total_bytes / chips
        utilization = per_chip_bytes / (hbm * GIB)
        results.append(SliceFit(accelerator_type, chips, hbm, per_chip_bytes, utilization <= headroom, utilization))
    return sorted(results, key=lambda result: (result.chips, result.accelerator_type))


def smallest_fit(
    estimate: MemoryEstimate,
    accelerator_types: Iterable[str] = DEFAULT_CANDIDATE_TYPES,
    *,
    headroom: float = DEFAULT_HEADROOM,
) -> SliceFit | None:
    """Return the fewest-chip fit; ties use accelerator_type lexical order."""

    return next(
        (candidate for candidate in fitting_slices(estimate, accelerator_types, headroom=headroom) if candidate.fits),
        None,
    )


def size_job(
    request: SizingRequest,
    accelerator_types: Iterable[str] = DEFAULT_CANDIDATE_TYPES,
    *,
    headroom: float = DEFAULT_HEADROOM,
) -> dict[str, Any]:
    """Return a compact JSON-ready estimate with GiB values rounded to 2 places.

    ``estimate`` keys end in ``_gib``; fitting candidates are capped at five.
    Recommendation is a memory-only fewest-chip choice, not a price/placement
    decision. Config counting ignores small biases and model-specific extra norms.
    """

    estimate = estimate_memory(request)
    fits = [candidate for candidate in fitting_slices(estimate, accelerator_types, headroom=headroom) if candidate.fits]
    return {
        "parameters": request.shape.parameters,
        "active_parameters": request.shape.active_parameters,
        "estimate": {
            "parameters_gib": round(estimate.parameters_bytes / GIB, 2),
            "gradients_gib": round(estimate.gradients_bytes / GIB, 2),
            "optimizer_gib": round(estimate.optimizer_bytes / GIB, 2),
            "reference_gib": round(estimate.reference_bytes / GIB, 2),
            "activations_gib": round(estimate.activations_bytes / GIB, 2),
            "generation_gib": round(estimate.generation_bytes / GIB, 2),
            "total_gib": round(estimate.total_bytes / GIB, 2),
        },
        "fits": [
            {
                "accelerator_type": candidate.accelerator_type,
                "chips": candidate.chips,
                "hbm_per_chip_gib": candidate.hbm_per_chip_gib,
                "per_chip_gib": round(candidate.per_chip_bytes / GIB, 2),
                "utilization": round(candidate.utilization, 2),
            }
            for candidate in fits[:MAX_REPORTED_FITS]
        ],
        "recommended": fits[0].accelerator_type if fits else None,
        "notes": [
            *estimate.notes,
            "Config parameter counts omit small biases and architecture-specific extra norms.",
            f"Memory-only recommendation with {headroom:.0%} of physical HBM usable; capacity is not checked.",
        ],
    }


__all__ = (
    "DEFAULT_CANDIDATE_TYPES",
    "TPU_HBM_GIB",
    "MemoryEstimate",
    "ModelShape",
    "SizingRequest",
    "SliceFit",
    "estimate_memory",
    "fitting_slices",
    "model_shape_from_config",
    "shape_from_parameter_count",
    "size_job",
    "smallest_fit",
)
